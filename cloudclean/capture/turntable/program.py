"""Step-and-scan programs: validation and the runner thread.

A program is ``{mode, interval_deg, frames_per_stop, direction, speed_s_per_rev, rotations: [{tilt_deg}],
sync_scan, dwell_s, settle_s, level_at_end, capture_timeout_s}``.

``mode="step"`` (Revo Metro's "Auto Turntable"): for each rotation, tilt to its angle, then per stop rotate
``interval_deg`` and let the scanner capture ``frames_per_stop`` frames, until the rotation adds up to 360 deg.
When 360 is not a multiple of the interval the last move of a rotation is shorter, so every rotation ends exactly
where it started (reported in ``moves``). ``mode="continuous"`` sweeps each rotation in one 360 deg move while
capturing ("Turntable Sync" style). A laser-line scanner (driver capability ``sweeps``, the MetroY) always runs
continuous: at a stop it sees the same few lines, so stop-and-go gives lines, never a surface (manager.py).

With ``sync_scan`` the runner drives the live capture session the way Revo Metro does [verified from its code and
logs]: capture is **paused while the platter moves** and resumed at each stop until ``frames_per_stop`` new frames
arrived; without a capture session (or with ``sync_scan=false``) it simply dwells ``dwell_s`` seconds per stop.
"""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime

from .base import TurntableError
from .protocol import whole_degrees

PROGRAM_KEYS = ("mode", "interval_deg", "frames_per_stop", "direction", "speed_s_per_rev", "rotations", "sync_scan",
                "dwell_s", "settle_s", "level_at_end", "capture_timeout_s")
DEFAULTS = {"mode": "step", "interval_deg": 30, "frames_per_stop": 3, "direction": "cw", "speed_s_per_rev": None,
            "rotations": [{"tilt_deg": 0}], "sync_scan": True, "dwell_s": 1.0, "settle_s": 0.3,
            "level_at_end": True, "capture_timeout_s": None}
DIRECTIONS = {"cw": "cw", "clockwise": "cw", "ccw": "ccw", "counterclockwise": "ccw", "anticlockwise": "ccw"}


def _number(value, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise TurntableError(f"{label} must be a number")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TurntableError(f"{label} must be a number") from None
    if not math.isfinite(value):
        raise TurntableError(f"{label} must be a finite number")
    return value


def _bool(value, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("1", "0", "true", "false", "yes", "no", "on", "off"):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise TurntableError(f"{label} must be true or false")


def validate_program(program: dict | None, capabilities: dict) -> dict:
    """Merge with defaults, check every value against the device's capabilities. Raises TurntableError.
    Returns the normalized plan including `notes` (e.g. values rounded to whole degrees)."""
    program = dict(program or {})
    unknown = sorted(k for k in program if k not in PROGRAM_KEYS)
    if unknown:
        raise TurntableError(f"Unknown program setting(s): {', '.join(unknown)}. Valid: {', '.join(PROGRAM_KEYS)}")
    p = {**DEFAULTS, **{k: v for k, v in program.items() if v is not None or k == "speed_s_per_rev"}}
    notes: list[str] = []

    mode = str(p["mode"]).strip().lower()
    if mode not in ("step", "continuous"):
        raise TurntableError("mode must be 'step' or 'continuous'")

    lo, hi = capabilities.get("interval_range") or [5, 30]
    interval_req = _number(p["interval_deg"], "interval_deg")
    interval = whole_degrees(interval_req, "rotation interval")
    if interval != interval_req:
        notes.append(f"Rotation interval {interval_req:g} deg rounded to {interval} deg (the turntable takes whole "
                     f"degrees).")
    if mode == "step" and not lo <= interval <= hi:
        raise TurntableError(f"Rotation interval must be between {lo} and {hi} degrees")

    frames = _number(p["frames_per_stop"], "frames_per_stop")
    if frames != int(frames) or not 1 <= frames <= 100:
        raise TurntableError("frames_per_stop must be a whole number between 1 and 100")

    direction = DIRECTIONS.get(str(p["direction"]).strip().lower())
    if direction is None:
        raise TurntableError("direction must be 'cw' (clockwise seen from above) or 'ccw'")

    speed = None
    if p["speed_s_per_rev"] is not None:
        s_lo, s_hi = capabilities.get("speed_range") or [25, 90]
        speed_req = _number(p["speed_s_per_rev"], "speed_s_per_rev")
        speed = whole_degrees(speed_req, "speed")
        if speed != speed_req:
            notes.append(f"Speed {speed_req:g} s/rev rounded to {speed} s/rev.")
        if not s_lo <= speed <= s_hi:
            raise TurntableError(f"Speed must be between {s_lo} and {s_hi} seconds per revolution")

    rotations_in = p["rotations"]
    if isinstance(rotations_in, (int, float)) and not isinstance(rotations_in, bool):
        rotations_in = [{"tilt_deg": 0}] * int(rotations_in)
    if not isinstance(rotations_in, list) or not rotations_in:
        raise TurntableError("rotations must be a list like [{\"tilt_deg\": 0}, {\"tilt_deg\": 20}]")
    max_rot = int(capabilities.get("max_rotations") or 5)
    if len(rotations_in) > max_rot:
        raise TurntableError(f"At most {max_rot} rotations are allowed (Revo Metro offers the same)")
    tilt_range = capabilities.get("tilt_range")
    rotations = []
    for i, rot in enumerate(rotations_in, 1):
        raw = rot.get("tilt_deg", 0) if isinstance(rot, dict) else rot
        if isinstance(rot, dict) and set(rot) - {"tilt_deg"}:
            raise TurntableError(f"Rotation {i}: only 'tilt_deg' can be set")
        tilt_req = _number(0 if raw is None else raw, f"Rotation {i} tilt_deg")
        tilt = whole_degrees(tilt_req, "tilt")
        if tilt != tilt_req:
            notes.append(f"Rotation {i}: tilt {tilt_req:g} deg rounded to {tilt} deg.")
        if not capabilities.get("tilt"):
            if tilt != 0:
                raise TurntableError("This turntable cannot tilt; use tilt_deg 0 for every rotation")
        elif not tilt_range[0] <= tilt <= tilt_range[1]:
            raise TurntableError(f"Rotation {i}: tilt must be between {tilt_range[0]} and {tilt_range[1]} degrees")
        rotations.append({"tilt_deg": tilt})

    dwell = _number(p["dwell_s"], "dwell_s")
    settle = _number(p["settle_s"], "settle_s")
    if not 0 <= dwell <= 600 or not 0 <= settle <= 30:
        raise TurntableError("dwell_s must be 0..600 s and settle_s 0..30 s")
    cap_timeout = p["capture_timeout_s"]
    if cap_timeout is not None:
        cap_timeout = _number(cap_timeout, "capture_timeout_s")
        if not 1 <= cap_timeout <= 3600:
            raise TurntableError("capture_timeout_s must be between 1 and 3600 s")

    if mode == "step":
        stops = math.ceil(360.0 / interval - 1e-9)
        moves = [interval] * (stops - 1) + [360 - interval * (stops - 1)]
    else:
        stops, moves = 1, [360]
    if moves[-1] != interval and mode == "step":
        notes.append(f"360 is not a multiple of {interval}: the last move of each rotation is {moves[-1]} deg so "
                     "every rotation ends where it started.")
    return {"mode": mode, "interval_deg": interval, "frames_per_stop": int(frames), "direction": direction,
            "speed_s_per_rev": speed, "rotations": rotations, "sync_scan": _bool(p["sync_scan"], "sync_scan"),
            "dwell_s": dwell, "settle_s": settle, "level_at_end": _bool(p["level_at_end"], "level_at_end"),
            "capture_timeout_s": cap_timeout, "stops_per_rotation": stops, "moves": moves, "notes": notes}


class _Stopped(Exception):
    pass


class ProgramRunner:
    """Runs one validated plan in a daemon thread. `capture` is a callable returning the workspace's
    CaptureManager (or None)."""

    def __init__(self, driver, plan: dict, capture=None, log=None):
        self.driver = driver
        self.plan = plan
        self._capture = capture or (lambda: None)
        self._log = log or (lambda msg: None)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._we_paused = False      # capture paused by us (resume only what we paused / started)
        self._we_started = False
        self.thread: threading.Thread | None = None
        n_rot = len(plan["rotations"])
        self.info = {"state": "starting", "mode": plan["mode"], "phase": None, "rotation": 0, "rotations": n_rot,
                     "stop": 0, "stops_per_rotation": plan["stops_per_rotation"],
                     "total_stops": plan["stops_per_rotation"] * n_rot, "completed_stops": 0, "progress": 0.0,
                     "interval_deg": plan["interval_deg"], "moves": plan["moves"],
                     "frames_per_stop": plan["frames_per_stop"], "direction": plan["direction"],
                     "speed_s_per_rev": plan["speed_s_per_rev"], "sync_scan": plan["sync_scan"],
                     "tilt_deg": None, "tilts": [r["tilt_deg"] for r in plan["rotations"]],
                     "turned_deg": 0, "started": datetime.now().isoformat(timespec="seconds"), "finished": None,
                     "elapsed_s": 0.0, "error": None, "warnings": [], "notes": list(plan["notes"]),
                     "capture": {"linked": False, "frames": 0}}
        self._t0 = time.monotonic()

    # ----------------------------------------------------------------- public
    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="turntable-program", daemon=True)
        self.thread.start()

    def stop(self, wait: float = 10.0) -> None:
        self._stop.set()
        try:
            self.driver.stop()
        except Exception as exc:  # stopping must never raise to the caller
            self._warn(f"Stop command failed: {exc}")
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=wait)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def snapshot(self) -> dict:
        with self._lock:
            info = dict(self.info)
            info["warnings"] = list(info["warnings"])
            info["capture"] = dict(info["capture"])
        if info["finished"] is None:
            info["elapsed_s"] = round(time.monotonic() - self._t0, 1)
        return info

    # ----------------------------------------------------------------- helpers
    def _set(self, **values) -> None:
        with self._lock:
            self.info.update(values)
            total = self.info["total_stops"] or 1
            self.info["progress"] = round(self.info["completed_stops"] / total, 4)

    def _warn(self, msg: str) -> None:
        with self._lock:
            if msg not in self.info["warnings"]:
                self.info["warnings"].append(msg)
        self._log(msg)

    def _check(self) -> None:
        if self._stop.is_set():
            raise _Stopped()

    def _sleep(self, seconds: float) -> None:
        if self._stop.wait(max(0.0, seconds)):
            raise _Stopped()

    # capture link ------------------------------------------------------
    def _cm(self):
        if not self.plan["sync_scan"]:
            return None
        try:
            cm = self._capture()
        except Exception:
            cm = None
        if cm is None or getattr(cm, "session", None) is None:
            self._warn("No capture session is connected: the program dwells at each stop instead of waiting for "
                       "frames. Connect the scanner in Capture to scan while it turns.")
            return None
        return cm

    def _capture_state(self, cm) -> dict:
        try:
            return cm.status()
        except Exception:
            return {}

    def _pause_capture(self) -> None:
        cm = self._cm() if self.plan["sync_scan"] else None
        if cm is None:
            return
        if self._capture_state(cm).get("state") == "running":
            try:
                cm.pause()
                self._we_paused = True
            except Exception as exc:
                self._warn(f"Could not pause the capture: {exc}")

    def _run_capture(self) -> bool:
        """Make sure the capture is running; True when it is."""
        cm = self._cm()
        if cm is None:
            return False
        state = self._capture_state(cm).get("state")
        try:
            if state == "paused":
                cm.resume()
            elif state in ("connected", "stopped", "finished"):
                cm.start()
                self._we_started = True
            elif state != "running":
                self._warn(f"The capture is '{state}', so the program cannot capture at the stops.")
                return False
        except Exception as exc:
            self._warn(f"Could not start the capture: {exc}")
            return False
        self._set(capture={**self.info["capture"], "linked": True})
        return True

    def _capture_at_stop(self) -> None:
        plan = self.plan
        frames_needed = plan["frames_per_stop"]
        if not self._run_capture():
            self._set(phase="dwelling")
            self._sleep(plan["dwell_s"])
            return
        cm = self._cm()
        self._set(phase="capturing")
        start = int(self._capture_state(cm).get("frames") or 0)
        timeout = plan["capture_timeout_s"] or max(15.0, 3.0 * frames_needed)
        deadline = time.monotonic() + timeout
        while True:
            st = self._capture_state(cm)
            got = int(st.get("frames") or 0) - start
            if got >= frames_needed:
                break
            if st.get("state") not in ("running",):
                self._warn(f"The capture stopped ({st.get('state')}); continuing without waiting for frames.")
                break
            if time.monotonic() > deadline:
                self._warn(f"Only {got} of {frames_needed} frames arrived within {timeout:.0f} s at a stop; "
                           "continuing.")
                break
            self._sleep(0.05)
        with self._lock:
            self.info["capture"]["frames"] += max(0, int(self._capture_state(cm).get("frames") or 0) - start)
        self._pause_capture()

    # ----------------------------------------------------------------- the program
    def _run(self) -> None:
        plan, drv = self.plan, self.driver
        sign = 1 if plan["direction"] == "cw" else -1
        state, error = "done", None
        try:
            self._set(state="running")
            if plan["speed_s_per_rev"] is not None:
                self._set(phase="setting speed")
                drv.set_speed(plan["speed_s_per_rev"])
            can_tilt = bool(drv.capabilities().get("tilt"))
            for r_index, rot in enumerate(plan["rotations"], 1):
                self._check()
                self._set(rotation=r_index, stop=0, tilt_deg=rot["tilt_deg"])
                if can_tilt:
                    current = drv.state().get("tilt_deg")
                    if current is None or abs(float(current) - rot["tilt_deg"]) > 0.5:
                        self._pause_capture()
                        self._set(phase="tilting")
                        self._log(f"Rotation {r_index}: tilting to {rot['tilt_deg']} deg")
                        res = drv.tilt(rot["tilt_deg"])
                        if res.get("stopped"):
                            raise _Stopped()
                        self._check()
                if plan["mode"] == "continuous":
                    self._set(stop=1, phase="rotating")
                    capturing = self._run_capture()
                    cm = self._cm() if capturing else None
                    start = int(self._capture_state(cm).get("frames") or 0) if cm else 0
                    try:
                        res = drv.rotate(sign * 360)
                    finally:
                        if cm is not None:
                            with self._lock:
                                self.info["capture"]["frames"] += max(
                                    0, int(self._capture_state(cm).get("frames") or 0) - start)
                    if res.get("stopped"):
                        raise _Stopped()
                    if capturing:
                        self._pause_capture()
                    with self._lock:
                        self.info["turned_deg"] += 360
                        self.info["completed_stops"] += 1
                    self._set()
                    continue
                for k, move in enumerate(plan["moves"], 1):
                    self._check()
                    self._pause_capture()
                    self._set(stop=k, phase="rotating")
                    res = drv.rotate(sign * move)
                    if res.get("stopped"):
                        raise _Stopped()
                    with self._lock:
                        self.info["turned_deg"] += move
                    self._set(phase="settling")
                    self._sleep(plan["settle_s"])
                    self._capture_at_stop()
                    with self._lock:
                        self.info["completed_stops"] += 1
                    self._set()
            if plan["level_at_end"] and can_tilt and any(r["tilt_deg"] for r in plan["rotations"]):
                self._check()
                self._set(phase="leveling")
                drv.tilt(0)
        except _Stopped:
            state = "stopped"
        except TurntableError as exc:
            state, error = "error", str(exc)
            try:
                drv.stop()
            except Exception:
                pass
        except Exception as exc:  # a program must never die silently
            state, error = "error", f"{type(exc).__name__}: {exc}"
            try:
                drv.stop()
            except Exception:
                pass
        finally:
            try:
                self._pause_capture()
            except Exception:
                pass
            if self.info["capture"]["linked"]:
                self._warn("The capture was left paused: save it (or resume scanning) in the Scan step.")
            self._set(state=state, error=error, phase=None, finished=datetime.now().isoformat(timespec="seconds"),
                      elapsed_s=round(time.monotonic() - self._t0, 1))
            self._log(f"Program {state}" + (f": {error}" if error else ""))
