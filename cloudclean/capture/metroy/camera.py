"""Live camera settings and the camera view for the MetroY (and the simulated scanner that mimics it).

CameraControl is the only writer of the exposure registers while the scanner streams. Changes are coalesced (a slider
being dragged sends many values; only the latest is written), applied one register per shell command with the
scanner's pause between commands (exposure.write_plan has the rules), and a register that already holds a value is
never written again. Writes run on their own thread so neither the frame stream nor a web request ever waits for
them.

The camera view: workers return a small copy of the rectified images (plus where they found stripe centres, markers
and saturated pixels) for a frame now and then while someone is looking; compose_view() turns that into one picture
(left, right or both side by side, with or without the overlay) and encode_jpeg() makes it a JPEG. Pillow only: no
OpenCV needed to show it.
"""
from __future__ import annotations

import io
import threading
import time
from dataclasses import asdict
from typing import Callable

import numpy as np

from .exposure import FRAME_TIME_US, AutoState, CameraSettings, limits, write_plan

# overlay colours (RGB): found laser lines, washed-out pixels, markers
LINE_RGB = (76, 214, 120)
SATURATED_RGB = (255, 64, 64)
MARKER_RGB = (90, 160, 255)


class CameraControl:
    def __init__(self, hid, gap_s: float = 0.15, frame_time_us: int = FRAME_TIME_US,
                 on_applied: Callable[[float], None] | None = None):
        self.hid = hid                      # anything with .shell(command)
        self.gap_s = gap_s
        self.frame_time_us = frame_time_us
        self.on_applied = on_applied
        self.applied: dict | None = None    # register values the scanner holds (None: unknown)
        self.commands = 0
        self.error: str | None = None
        self._wanted: dict | None = None
        self._last_cmd = -1e9
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = False

    # -- writing
    def assume(self, registers: dict) -> None:
        """The scanner is known to hold these (e.g. right after MetroyHid.startup())."""
        with self._cv:
            self.applied = dict(registers)

    def apply_now(self, registers: dict) -> int:
        """Write synchronously (before streaming starts). Returns the number of commands sent."""
        with self._cv:
            self._wanted = None
        return self._write(dict(registers))

    def request(self, registers: dict) -> None:
        """Ask for these values; the writer thread applies them soon. Never blocks."""
        with self._cv:
            self._wanted = dict(registers)
            self._cv.notify()

    def pending(self) -> bool:
        with self._cv:
            return self._wanted is not None

    def _write(self, wanted: dict) -> int:
        sent = 0
        plan = write_plan(self.applied, wanted, self.frame_time_us)
        for keys, command in plan:
            wait = self._last_cmd + self.gap_s - time.perf_counter()     # perf_counter: fine-grained everywhere
            if wait > 0:
                time.sleep(wait)
            self.hid.shell(command)
            self._last_cmd = time.perf_counter()
            sent += 1
            self.commands += 1
            with self._cv:
                if self.applied is None:
                    self.applied = {}
                for k in keys:
                    self.applied[k] = wanted[k]
                newer = self._wanted is not None
            if newer:
                break                      # a newer request: plan again from what is applied now
        with self._cv:
            done = self._wanted is None and (self.applied or {}) == wanted
        if done and self.on_applied is not None:
            # the frame after the last write is the first that shows all of it
            self.on_applied(time.monotonic())
        return sent

    # -- the writer thread
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="metroy-camera", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        self._thread = None

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._wanted is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                wanted, self._wanted = self._wanted, None
            try:
                self._write(wanted)
            except OSError as exc:         # the scanner went away: the stream reports it; stop writing
                self.error = f"could not change the camera settings ({exc})"
                return


def camera_payload(cs: CameraSettings, auto: AutoState, readout: dict, *, streaming: bool, preview: bool,
                   starting: bool = False, error: str | None = None, gain_map: str = "defaults",
                   applied: dict | None = None) -> dict:
    """GET /api/capture/camera, for every driver that has a camera (see routes_capture)."""
    return {"settings": cs.to_dict(), "limits": limits(cs), "auto": asdict(auto), "readout": readout,
            "streaming": streaming, "preview": preview, "starting": starting, "error": error,
            "gain_map": gain_map, "applied": applied}


def camera_brief(payload: dict) -> dict:
    """The few numbers the live status carries 5 times a second (the floating camera view reads them)."""
    s, r, a = payload["settings"], payload["readout"], payload["auto"]
    return {"mode": s["mode"], "surface": s["surface"], "laser_pct": s["laser_pct"], "gain": s["gain"],
            "verdict": a["verdict"], "auto_state": a["state"], "auto_message": a["message"],
            "stripe_brightness": r.get("stripe_brightness"), "saturated_pct": r.get("saturated_pct"),
            "points_per_frame": r.get("points_per_frame"), "streaming": payload["streaming"],
            "preview": payload["preview"]}


class PreviewLease:
    """Keeps a preview alive only while someone is looking: every view request renews it, and when renewals stop
    (the browser tab closed, the panel folded) `expire` is called once - the laser does not stay on for nobody."""

    def __init__(self, expire: Callable[[], None], lease_s: float = 6.0):
        self.expire = expire
        self.lease_s = lease_s
        self.until = 0.0
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return time.monotonic() < self.until

    def renew(self) -> None:
        with self._lock:
            self.until = time.monotonic() + self.lease_s
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._watch, name="camera-preview-lease", daemon=True)
                self._thread.start()

    def end(self) -> None:
        with self._lock:
            self.until = 0.0

    def _watch(self) -> None:
        while True:
            time.sleep(0.25)
            with self._lock:
                if self.until == 0.0:
                    self._thread = None
                    return
                expired = time.monotonic() >= self.until
                if expired:
                    self.until = 0.0
                    self._thread = None
            if expired:
                self.expire()
                return


# ----------------------------------------------------------------------------------------------- the picture
def make_view(rl: np.ndarray, rr: np.ndarray, scale: float, centres_l=None, centres_r=None, markers_l=(),
              markers_r=(), saturated: int = 250) -> dict:
    """A small copy of a rectified pair and what was found in it, for compose_view(). Runs in a worker (OpenCV)."""
    import cv2
    h, w = rl.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))

    def shrink(img):
        small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        sat = cv2.resize((img >= saturated).astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) > 0
        return small, np.packbits(sat)

    left, sat_l = shrink(rl)
    right, sat_r = shrink(rr)

    def pts(c):
        # one entry per view pixel: at this scale thousands of centres share a few pixels each
        c = np.zeros((0, 2), np.float32) if c is None else np.asarray(c, np.float64).reshape(-1, 2)
        c = c[np.isfinite(c).all(axis=1)] * scale
        return np.unique(np.rint(c).astype(np.int16), axis=0) if len(c) else np.zeros((0, 2), np.int16)

    def mk(blobs):
        return np.asarray([(b.x * scale, b.y * scale, b.radius * scale) for b in blobs], np.float32).reshape(-1, 3)

    return {"left": left, "right": right, "sat_left": sat_l, "sat_right": sat_r, "shape": left.shape,
            "centres_left": pts(centres_l), "centres_right": pts(centres_r),
            "markers_left": mk(markers_l), "markers_right": mk(markers_r)}


def compose_view(view: dict, cams: str = "both", overlay: bool = True, width: int = 960) -> np.ndarray:
    """One RGB picture: the left camera, the right one, or both side by side; the overlay paints found laser lines
    green, washed-out (saturated) pixels red and rings the markers blue."""
    from PIL import Image, ImageDraw
    h, w = view["shape"][:2]
    sides = {"left": ["left"], "right": ["right"]}.get(cams, ["left", "right"])
    gap = 4 if len(sides) == 2 else 0
    canvas = np.full((h, w * len(sides) + gap * (len(sides) - 1), 3), 10, np.uint8)
    rings = []
    for i, side in enumerate(sides):
        x0 = i * (w + gap)
        grey = view[side]
        tile = np.repeat(grey[:, :, None], 3, axis=2)
        if overlay:
            sat = np.unpackbits(view[f"sat_{side}"], count=h * w).reshape(h, w).astype(bool)
            tile[sat] = SATURATED_RGB
            c = view[f"centres_{side}"]
            if len(c):
                xs = np.clip(np.rint(c[:, 0]).astype(int), 0, w - 1)
                ys = np.clip(np.rint(c[:, 1]).astype(int), 0, h - 1)
                tile[ys, xs] = LINE_RGB
            rings += [(x0 + float(x), float(y), max(3.0, float(r) * 1.4)) for x, y, r in view[f"markers_{side}"]]
        canvas[:, x0:x0 + w] = tile
    img = Image.fromarray(canvas)
    if rings:
        draw = ImageDraw.Draw(img)
        for x, y, r in rings:
            draw.ellipse((x - r, y - r, x + r, y + r), outline=MARKER_RGB, width=2)
    width = int(max(16, min(1600 * len(sides), width)))
    if img.width != width:
        img = img.resize((width, max(1, round(img.height * width / img.width))), Image.BILINEAR)
    return np.asarray(img)


def encode_jpeg(rgb: np.ndarray, quality: int = 80) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    return buf.getvalue()
