"""How bright the MetroY's laser lines are, and how CloudClean keeps them right: Revo Metro's surface presets, the
camera-settings model, the register writes that apply it, and the automatic exposure loop. Pure Python + numpy (no
OpenCV, no device), so all of it is unit tested on any machine.

Registers (docs/metroy-protocol.md, docs/revo-metro-internals.md, docs/metroy-handoff.md "Camera view"):

* 0x911 exposure in us - written ONLY together with frame time 0x910, in one shell command; 0x903 gain (0x10 per 1x);
  0xb07 marker light (the IR fill light that makes retro-reflective markers shine); 0xb08 laser pulse ("laser line
  exposure time"); 0x48d laser power, always 255 (start-up). Every register other than the 0x910/0x911 pair goes in a
  command of its own, with a pause (four chained writes made the scanner drop off the bus).
* Laser level -> pulse, as Revo Metro does: ``0xb08 = 45 + int(level * exposure_us / level_max)``. The level is the
  share of the exposure the laser is lit. Fitted on Revo's own logs for the general preset (level 44 -> 217, 38 -> 194,
  48 -> 233, 51 -> 245) [verified]. That the same formula holds for the dark (193 -> 801) and reflective (90 -> 397)
  presets is [inferred]: exposure / level_max is 3.92 in every preset, but no log of a dark session survives. Before
  this, CloudClean wrote 217 for every preset, so Dark integrated 1000 us of background around a 217 pulse: about a
  fifth of the stripe light Revo Metro gives a dark part.

Automatic exposure (CloudClean's loop, after Revo Metro's RPALaserAutoExposure, which runs every ~0.8 s in preview and
while scanning):

* measures every frame at the stripe centres CloudClean actually triangulated, in the central 70 % of the left image:
  the 95th percentile of their raw grey level ("stripe brightness") and the share that is saturated;
* acts only on frames whose points lie 220-380 mm away on average (Revo's window; nearer or farther the stripe
  brightness says more about distance than about exposure), and ignores frames captured before its last change took
  effect;
* every 0.8 s: inside the target band it holds. Otherwise it scales the laser level towards the target. When the
  level would have to pass its maximum it steps the gain up by one (max 5) and re-scales the level, and the marker
  light follows the gain from the scanner's brightnessGainMap so markers stay as bright as before; going down it does
  the reverse. Exposure itself is never changed by auto (Revo's AE does not either).
* The target - stripe brightness 200 of 255, band 175-230, at most 3 % saturated - is CloudClean's choice: Revo's
  target grey level was not recovered from its binaries.

Manual mode never changes anything on its own.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

BLACK_LEVEL = 16                    # sensor black level (8-bit)
PULSE_OFFSET = 45                   # 0xb08 at level 0 (Revo's fit)
MAX_PULSE = 1045                    # the longest pulse Revo Metro asks for in laser mode (dark preset at 255)
FRAME_TIME_US = 8000
GAIN_RANGE = (1, 5)                 # Revo's laser auto exposure goes up to gain 5
EXPOSURE_RANGE_US = (100, 2000)     # manual exposure; the laser presets use 200-1000
MARKER_LIGHT_RANGE = (0, 255)
SATURATED_GREY = 250                # a stripe centre at or above this is clipped

SURFACES = ("general", "dark", "reflective")
MODES = ("auto", "manual")


@dataclass(frozen=True)
class SurfacePreset:
    exposure_us: int
    gain: int
    laser_level: int
    level_max: int
    marker_light: int


# Revo Metro's cross-line presets for the MetroY Ultra (crosswire_param.json, model 0L8) [verified config]:
# exposure, gain, laserLuminance / laserLuminanceMax, fillLight
PRESETS: dict[str, SurfacePreset] = {
    "general": SurfacePreset(200, 1, 44, 51, 30),
    "dark": SurfacePreset(1000, 2, 193, 255, 3),
    "reflective": SurfacePreset(800, 1, 90, 204, 6),
}

# Marker light per gain 1..n when the gain changes (the scanner's /data/brightnessGainMap.json; these are the values
# of our scanner's copy, used when the file cannot be read off the device)
MARKER_LIGHT_BY_GAIN: dict[str, tuple[int, ...]] = {
    "general": (45, 21, 14, 10, 8, 7, 6, 5, 4, 4),
    "dark": (9, 4, 2, 1, 1),
    "reflective": (11, 5, 3, 2, 1),
}
_GAIN_MAP_KEYS = {"general": "lineGeneral", "dark": "lineDark", "reflective": "lineReflect"}
_XOR_KEYS = (b"RevoPoint", b"RevoScan")      # Revo Metro's config obfuscation (both keys, XORed together)


def parse_gain_map(data: bytes | str) -> dict[str, tuple[int, ...]]:
    """brightnessGainMap.json -> {surface: marker light for gain 1, 2, ...}. Accepts the plain file (as on the
    scanner) and Revo Metro's obfuscated PC copy. Surfaces missing from the file keep CloudClean's defaults.
    Raises ValueError when nothing usable is in it."""
    raw = data.encode() if isinstance(data, str) else bytes(data)
    doc = None
    for attempt in (raw, bytes(b ^ _XOR_KEYS[0][i % 9] ^ _XOR_KEYS[1][i % 8] for i, b in enumerate(raw))):
        try:
            doc = json.loads(attempt.decode("utf-8"))
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    if not isinstance(doc, dict):
        raise ValueError("not a brightness/gain map")
    out = dict(MARKER_LIGHT_BY_GAIN)
    found = False
    for surface, key in _GAIN_MAP_KEYS.items():
        rows = doc.get(key)
        if not isinstance(rows, list):
            continue
        try:
            pairs = sorted((int(r["gain"]), int(r["brightness"])) for r in rows)
        except (KeyError, TypeError, ValueError):
            continue
        if pairs and [g for g, _ in pairs] == list(range(1, len(pairs) + 1)):
            out[surface] = tuple(max(0, min(255, b)) for _, b in pairs)
            found = True
    if not found:
        raise ValueError("the brightness/gain map has no laser-line entries")
    return out


def marker_light_for(surface: str, gain: int, gain_map: dict | None = None) -> int:
    table = (gain_map or MARKER_LIGHT_BY_GAIN).get(surface) or MARKER_LIGHT_BY_GAIN[surface]
    return int(table[max(0, min(len(table), int(gain))) - 1])


def laser_pulse(level: int, exposure_us: int, level_max: int) -> int:
    """0xb08 for a laser level: Revo Metro's 45 + level * exposure / level_max (truncated), capped at Revo's maximum."""
    pulse = PULSE_OFFSET + int(int(level) * int(exposure_us) / int(level_max))
    return max(PULSE_OFFSET, min(MAX_PULSE, pulse))


# ----------------------------------------------------------------------------------------------- settings model
@dataclass(frozen=True)
class CameraSettings:
    surface: str = "general"
    mode: str = "auto"
    laser_level: int = 44
    exposure_us: int = 200
    gain: int = 1
    marker_light: int = 30

    @property
    def preset(self) -> SurfacePreset:
        return PRESETS[self.surface]

    @property
    def level_max(self) -> int:
        return self.preset.level_max

    @property
    def level_min(self) -> int:
        return max(1, round(0.05 * self.level_max))

    @property
    def pulse(self) -> int:
        return laser_pulse(self.laser_level, self.exposure_us, self.level_max)

    def registers(self) -> dict:
        """What goes to the scanner: exposure (with the frame time), gain, marker light and laser pulse."""
        return {"exposure_us": int(self.exposure_us), "gain": int(self.gain), "marker_light": int(self.marker_light),
                "pulse": self.pulse}

    def to_dict(self) -> dict:
        return {**asdict(self), "level_max": self.level_max, "laser_pulse": self.pulse,
                "laser_pct": round(100.0 * self.laser_level / self.level_max, 1)}

    @classmethod
    def from_preset(cls, surface: str = "general", mode: str = "auto", markers: bool = True) -> "CameraSettings":
        if surface not in PRESETS:
            raise ValueError(f"surface must be one of: {', '.join(SURFACES)}")
        if mode not in MODES:
            raise ValueError("mode must be auto or manual")
        p = PRESETS[surface]
        return cls(surface, mode, p.laser_level, p.exposure_us, p.gain, p.marker_light if markers else 0)

    def with_changes(self, changes: dict, markers: bool = True) -> "CameraSettings":
        """A validated copy. A new surface loads that surface's preset (values given in the same request win). Any
        of laser_level / exposure_us / gain / marker_light switches to manual unless mode is given too."""
        changes = {k: v for k, v in (changes or {}).items() if v is not None}
        unknown = set(changes) - {"surface", "mode", "laser_level", "exposure_us", "gain", "marker_light"}
        if unknown:
            raise ValueError(f"unknown camera setting(s): {', '.join(sorted(unknown))}")
        s = self
        if "surface" in changes and changes["surface"] != self.surface:
            s = CameraSettings.from_preset(changes["surface"], self.mode, markers)
        values = {k: changes[k] for k in ("laser_level", "exposure_us", "gain", "marker_light") if k in changes}
        mode = changes.get("mode", "manual" if values else s.mode)
        if mode not in MODES:
            raise ValueError("mode must be auto or manual")
        out = {}
        bounds = {"laser_level": (1, s.level_max), "exposure_us": EXPOSURE_RANGE_US, "gain": GAIN_RANGE,
                  "marker_light": MARKER_LIGHT_RANGE}
        words = {"laser_level": "Laser brightness", "exposure_us": "Exposure", "gain": "Gain",
                 "marker_light": "Marker light"}
        for k, v in values.items():
            try:
                if isinstance(v, bool):
                    raise TypeError
                f = float(v)
                if not math.isfinite(f):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"{words[k]} must be a number") from None
            lo, hi = bounds[k]
            if not lo <= f <= hi:
                raise ValueError(f"{words[k]} must be between {lo} and {hi}")
            out[k] = int(round(f))
        return replace(s, mode=mode, **out)


def camera_from_driver_settings(settings: dict, markers: bool = True, surface_key: str = "surface") -> CameraSettings:
    """The camera settings remembered with a driver's settings: surface + camera_mode; in manual mode also
    laser_level, exposure_us, gain and fill_light (marker light); -1 means "the surface preset's value"."""
    surface = settings.get(surface_key) or "general"
    surface = {"normal": "general", "shiny": "reflective"}.get(surface, surface)
    mode = settings.get("camera_mode") or "auto"
    cs = CameraSettings.from_preset(surface if surface in PRESETS else "general", mode if mode in MODES else "auto",
                                    markers)
    if cs.mode != "manual":
        return cs
    changes = {"mode": "manual"}
    for key, field_ in (("laser_level", "laser_level"), ("exposure_us", "exposure_us"), ("gain", "gain"),
                        ("fill_light", "marker_light")):
        v = settings.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
            changes[field_] = v
    try:
        return cs.with_changes(changes, markers)
    except ValueError:              # a remembered value out of range (e.g. another surface's level): the preset
        return cs


def remembered(cs: CameraSettings, surface_key: str = "surface", surface_value: str | None = None) -> dict:
    """Driver-setting values that remember `cs` for the next connect (the inverse of camera_from_driver_settings)."""
    manual = cs.mode == "manual"
    return {surface_key: surface_value or cs.surface, "camera_mode": cs.mode,
            "laser_level": cs.laser_level if manual else -1, "exposure_us": cs.exposure_us if manual else -1,
            "gain": cs.gain if manual else -1, "fill_light": cs.marker_light if manual else -1}


def limits(cs: CameraSettings) -> dict:
    """Slider ranges for the UI, and each surface's preset (what Revo Metro starts from)."""
    return {"laser_level": [cs.level_min, cs.level_max], "exposure_us": list(EXPOSURE_RANGE_US),
            "gain": list(GAIN_RANGE), "marker_light": list(MARKER_LIGHT_RANGE),
            "presets": {k: asdict(p) for k, p in PRESETS.items()}}


# ----------------------------------------------------------------------------------------------- register writes
def _preisp(register: int, value) -> str:
    return f"echo s 0x{register:x} {value} > /dev/rk_preisp"


REG_FRAME_TIME, REG_EXPOSURE, REG_GAIN, REG_FILL_LIGHT, REG_LASER_TIMES = 0x910, 0x911, 0x903, 0xB07, 0xB08

# what MetroyHid.startup() leaves behind: the state the first write plan starts from
STARTUP_REGISTERS = {"exposure_us": 200, "gain": 1, "marker_light": 30, "pulse": 217}


def write_plan(old: dict | None, new: dict, frame_time_us: int = FRAME_TIME_US) -> list[tuple[tuple[str, ...], str]]:
    """Shell commands that take the scanner from register state `old` (None = unknown: write everything) to `new`,
    as [(register keys it sets, command)]. Rules measured on the user's scanner:

    * frame time 0x910 and exposure 0x911 together, in ONE command (0x910 alone stalled the stream);
    * every other register in a command of its own (the caller pauses between commands);
    * a register that already holds the value is not written again.

    Order: a longer exposure is written before the longer laser pulse that goes with it, a shorter one after the
    shorter pulse, so the pulse never outlasts the exposure window in between."""
    def changed(k):
        return old is None or old.get(k) != new[k]

    exposure = ((("exposure_us",), f"{_preisp(REG_FRAME_TIME, int(frame_time_us))}; "
                                   f"{_preisp(REG_EXPOSURE, int(new['exposure_us']))}")
                if changed("exposure_us") else None)
    pulse = (("pulse",), _preisp(REG_LASER_TIMES, int(new["pulse"]))) if changed("pulse") else None
    gain = (("gain",), _preisp(REG_GAIN, f"0x{int(round(16 * new['gain'])):02x}")) if changed("gain") else None
    marker = (("marker_light",), _preisp(REG_FILL_LIGHT, int(new["marker_light"]))) if changed("marker_light") else None
    longer = old is None or new["exposure_us"] >= old.get("exposure_us", 0)
    order = [exposure, pulse, gain, marker] if longer else [pulse, gain, marker, exposure]
    return [step for step in order if step is not None]


# ----------------------------------------------------------------------------------------------- measuring a frame
@dataclass
class Light:
    """One frame's laser lines as the camera saw them."""
    t: float                    # capture time, s (monotonic clock)
    centres: int                # triangulated stripe centres measured (central 70 % of the left view)
    peak: float                 # their raw grey level, 95th percentile (0-255): "stripe brightness"
    saturated: float            # share of them at or above SATURATED_GREY
    depth_mm: float | None      # mean depth of the frame's points in that region
    points: int = 0             # points the frame triangulated in all

    def to_dict(self) -> dict:
        return asdict(self)


def measure_light(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, pts: np.ndarray, t: float = 0.0,
                  roi: float = 0.7) -> Light:
    """Stripe brightness at the accepted (triangulated) centres of a rectified left image. xs, ys, pts are what
    Triangulator.points_rectified(..., with_pixels=True) returns. A few microseconds of numpy per frame."""
    pts = np.asarray(pts).reshape(-1, 3)
    if not len(xs):
        return Light(t, 0, 0.0, 0.0, None, int(len(pts)))
    h, w = img.shape[:2]
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    margin_x, margin_y = 0.5 * (1 - roi) * w, 0.5 * (1 - roi) * h
    m = (xs >= margin_x) & (xs < w - margin_x) & (ys >= margin_y) & (ys < h - margin_y)
    if not m.any():
        return Light(t, 0, 0.0, 0.0, None, int(len(pts)))
    x0 = np.clip(np.floor(xs[m]).astype(np.int64), 0, w - 2)
    y = np.clip(ys[m].astype(np.int64), 0, h - 1)
    peak = np.maximum(img[y, x0], img[y, x0 + 1]).astype(np.float64)      # the sub-pixel centre lies between them
    depth = float(np.mean(pts[m, 2])) if len(pts) == len(xs) else None
    return Light(t, int(m.sum()), float(np.percentile(peak, 95)), float(np.mean(peak >= SATURATED_GREY)), depth,
                 int(len(pts)))


# ----------------------------------------------------------------------------------------------- automatic exposure
@dataclass
class AutoState:
    state: str = "off"          # off (manual) | waiting | adjusting | steady | limit
    message: str = ""           # what auto is doing, in plain words
    verdict: str = "none"       # the laser lines: none | dim | good | bright


class AutoExposure:
    TARGET = 200.0              # stripe brightness aimed for (95th percentile, 0-255)
    BAND = (175.0, 230.0)       # no change inside
    SATURATION_MAX = 0.03       # more saturated centres than this is too bright, whatever the percentile says
    INTERVAL_S = 0.8            # one decision per interval (Revo: ~0.8 s)
    SETTLE_S = 0.1              # after a change took effect, frames captured within this are still ignored
    DEPTH_MM = (220.0, 380.0)   # Revo acts only inside this mean depth
    MIN_FRAMES = 3
    MIN_CENTRES = 40
    MAX_STEP = 2.0              # largest change of laser x gain per decision

    APPLY_TIMEOUT_S = 3.0       # a change the scanner never confirms stops holding the loop after this

    def __init__(self, settings: CameraSettings, gain_map: dict | None = None, markers: bool = True):
        self.settings = settings
        self.gain_map = gain_map
        self.markers = markers
        self.status = AutoState()
        self._frames: list[Light] = []
        self._last_decision: float | None = None
        self._hold_until = -math.inf
        self._issued_at = -math.inf
        self.reset(settings)

    def reset(self, settings: CameraSettings, now: float | None = None) -> None:
        """New settings from outside (the user, a surface change): start measuring afresh. With `now`, the new
        settings are on their way to the scanner: frames are ignored until applied() says they arrived."""
        self.settings = settings
        self._frames = []
        self._last_decision = None
        if now is not None:
            self._hold_until, self._issued_at = math.inf, now
        if settings.mode != "auto":
            self.status = AutoState("off", "Manual: your settings stay as they are", self.status.verdict)
        else:
            self.status = AutoState("waiting", "Measuring the laser lines", self.status.verdict)

    def applied(self, t: float) -> None:
        """The scanner holds the settings asked for since time t (monotonic): frames before t + SETTLE_S are old."""
        self._hold_until = t + self.SETTLE_S

    def observe(self, light: Light) -> None:
        if light.t < self._hold_until:
            return
        self._frames.append(light)
        if len(self._frames) > 400:
            del self._frames[:200]

    def step(self, now: float) -> CameraSettings | None:
        """Decide (at most every INTERVAL_S). Returns the new settings when they change, else None."""
        if self._hold_until == math.inf and now - self._issued_at > self.APPLY_TIMEOUT_S:
            self._hold_until = self._issued_at + self.APPLY_TIMEOUT_S
        if self._last_decision is None:
            self._last_decision = now
            return None
        if now - self._last_decision < self.INTERVAL_S:
            return None
        self._last_decision = now
        frames, self._frames = self._frames, []
        lo, hi = self.DEPTH_MM
        lit = [f for f in frames if f.centres >= self.MIN_CENTRES]
        usable = [f for f in lit if f.depth_mm is not None and lo <= f.depth_mm <= hi]
        verdict = self.status.verdict
        if lit:
            verdict = self._verdict(float(np.median([f.peak for f in lit])),
                                    float(np.median([f.saturated for f in lit])))
        elif frames:
            verdict = "none"
        if self.settings.mode != "auto":
            self.status = AutoState("off", "Manual: your settings stay as they are", verdict)
            return None
        if len(usable) < self.MIN_FRAMES:
            if self._hold_until == math.inf:
                msg = "Applying the last change"
            elif len(lit) >= self.MIN_FRAMES:
                msg = (f"Waiting: hold the scanner {lo:.0f}-{hi:.0f} mm from the part (auto exposure only adjusts "
                       "in that range)")
            else:
                msg = "Waiting: no laser lines on the part yet"
            self.status = AutoState("waiting", msg, verdict)
            return None
        peak = float(np.median([f.peak for f in usable]))
        sat = float(np.median([f.saturated for f in usable]))
        new, message, state = self._decide(peak, sat)
        self.status = AutoState(state, message, self._verdict(peak, sat))
        if new is None or new == self.settings:
            return None
        self.settings = new
        self._hold_until, self._issued_at = math.inf, now     # until the scanner reports it applied the change
        return new

    def _verdict(self, peak: float, sat: float) -> str:
        if sat > self.SATURATION_MAX or peak > self.BAND[1]:
            return "bright"
        if peak < self.BAND[0]:
            return "dim"
        return "good"

    def _decide(self, peak: float, sat: float) -> tuple[CameraSettings | None, str, str]:
        s = self.settings
        if sat > self.SATURATION_MAX:
            ratio = 0.7 if sat > 0.15 else 0.8           # clipped: the percentile under-reads, step down
        elif peak > self.BAND[1] or peak < self.BAND[0]:
            ratio = (self.TARGET - BLACK_LEVEL) / max(peak - BLACK_LEVEL, 4.0)
        else:
            return None, "Steady: the laser lines are right", "steady"
        ratio = min(self.MAX_STEP, max(1.0 / self.MAX_STEP, ratio))
        want = s.laser_level * s.gain * ratio            # stripe light ~ laser level x gain
        gain = s.gain
        if ratio > 1 and want > s.level_max * gain:
            # as Revo Metro: the laser goes to full first; only a laser already at full brings more gain
            if s.laser_level >= s.level_max and gain < GAIN_RANGE[1]:
                gain += 1
        elif ratio < 1 and gain > GAIN_RANGE[0] and want <= s.level_max * (gain - 1):
            gain -= 1                                    # down: the least gain that still reaches it (less noise)
        level = int(min(s.level_max, max(s.level_min, round(want / gain))))
        marker = s.marker_light
        if gain != s.gain:
            marker = marker_light_for(s.surface, gain, self.gain_map) if self.markers else 0
        new = replace(s, laser_level=level, gain=gain, marker_light=marker)
        if new == s:
            if ratio > 1:
                hint = " - choose Dark" if s.surface != "dark" else " - move a little closer"
                return None, f"At full laser and gain: the lines stay dim{hint}", "limit"
            hint = " - choose Shiny" if s.surface != "reflective" else ""
            return None, f"At the lowest laser and gain: the lines stay too bright{hint}", "limit"
        if gain > s.gain:
            message = f"More camera gain ({gain}x): the laser is at full and the lines are still dim"
        elif gain < s.gain:
            message = f"Less camera gain ({gain}x): the lines are too bright"
        elif level > s.laser_level:
            message = "Raising the laser: the lines are dim"
        else:
            message = ("Lowering the laser: " + (f"{100 * sat:.0f} % of the lines are washed out"
                                                 if sat > self.SATURATION_MAX else "the lines are too bright"))
        return new, message, "adjusting"


# ----------------------------------------------------------------------------------------------- readout
@dataclass
class Readout:
    """Recent frames summarised for the UI."""
    frames: list[Light] = field(default_factory=list)
    window_s: float = 1.5

    def add(self, light: Light) -> None:
        self.frames.append(light)
        cut = light.t - self.window_s
        if len(self.frames) > 4 and self.frames[0].t < cut:
            self.frames = [f for f in self.frames if f.t >= cut]
        if len(self.frames) > 300:
            self.frames = self.frames[-150:]

    def summary(self) -> dict:
        lit = [f for f in self.frames if f.centres >= AutoExposure.MIN_CENTRES]
        depths = [f.depth_mm for f in lit if f.depth_mm is not None]
        return {"frames": len(self.frames),
                "lines_seen": bool(lit),
                "stripe_brightness": round(float(np.median([f.peak for f in lit])), 1) if lit else None,
                "saturated_pct": round(100.0 * float(np.median([f.saturated for f in lit])), 1) if lit else None,
                "points_per_frame": int(np.median([f.points for f in self.frames])) if self.frames else None,
                "depth_mm": round(float(np.median(depths)), 1) if depths else None,
                "target": AutoExposure.TARGET, "band": list(AutoExposure.BAND),
                "saturation_max_pct": 100.0 * AutoExposure.SATURATION_MAX}
