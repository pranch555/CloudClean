"""A MetroY-like stereo IR camera for the simulated scanner: the practice scanner gets the same camera view, the same
settings model and the same automatic exposure as the real MetroY (cloudclean.capture.metroy.exposure), so all of it
can be tried - and tested - without hardware.

What it simulates: each laser point of a simulated frame lights up in proportion to laser pulse x gain x the part's
reflectance (a white, a black or a shiny part: the simulated driver's "Object surface"), falling with distance, with
laser speckle; bright enough and it is "found" (green in the overlay), too bright and it saturates (red). Markers in
a ring on the table shine with the marker light x gain. Noise grows with gain. The camera preset (General / Dark /
Shiny) is chosen separately, so a black part under the General preset shows exactly what the user saw on the real
scanner: dim lines that auto exposure cannot rescue until Dark is chosen.

The simulated camera never changes the simulated points: it only shows and measures.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from ..metroy.camera import camera_payload, compose_view, encode_jpeg
from ..metroy.exposure import (BLACK_LEVEL, PULSE_OFFSET, SATURATED_GREY, AutoExposure, CameraSettings, Light,
                               Readout, camera_from_driver_settings, remembered)

W, H = 480, 384                       # per camera (the simulated scanner's 30 degree field, aspect 0.8)
FOCAL = W * 0.5 / math.tan(math.radians(15.0))
BASELINE_MM = 30.0
K = 0.75                              # grey levels per (reflectance x gain x us of laser) at 250 mm
DETECT = 5.0                          # grey levels above black a centre needs to be found
MATERIAL = {"normal": (0.85, 0.3), "dark": (0.04, 0.3), "reflective": (0.35, 0.9)}   # reflectance, speckle sigma
RING = [(70.0 * math.cos(a), 70.0 * math.sin(a), 0.0) for a in np.radians(np.arange(0, 360, 30))]
MARKER_MM = 3.0


class SimulatedCamera:
    def __init__(self, settings: dict, markers: bool = True):
        self.material = MATERIAL.get(settings.get("surface", "normal"), MATERIAL["normal"])
        preset = settings.get("camera_surface") or "match"
        if preset == "match":
            preset = {"normal": "general", "dark": "dark", "reflective": "reflective"}.get(
                settings.get("surface", "normal"), "general")
        self.markers = markers
        self.camera = camera_from_driver_settings({**settings, "camera_surface": preset}, markers,
                                                  surface_key="camera_surface")
        self.ae = AutoExposure(self.camera, None, markers)
        self.readout = Readout()
        self.lock = threading.RLock()
        self._latest = None               # (frame, brightness per point)
        self._rng = np.random.default_rng(int(settings.get("seed", 0)) + 7)
        self.streaming = False
        self.preview = False

    # -- settings
    def set(self, changes: dict) -> CameraSettings:
        with self.lock:
            self.camera = self.camera.with_changes(changes, self.markers)
            self.ae.reset(self.camera, None)
            return self.camera

    def remembered(self) -> dict:
        return remembered(self.camera, surface_key="camera_surface")

    # -- frames
    def brightness(self, frame) -> np.ndarray:
        """Peak grey above black of every laser point of a simulated frame (before clipping)."""
        cs = self.camera
        pts = frame.points
        if not len(pts):
            return np.zeros(0)
        reflectance, speckle = self.material
        z = np.maximum(pts[:, 2].astype(np.float64), 50.0)
        laser_us = max(cs.pulse - PULSE_OFFSET, 0)
        spread = np.exp(self._rng.normal(0.0, speckle, len(pts)))
        return reflectance * spread * K * cs.gain * laser_us * (250.0 / z)

    def observe(self, frame, now: float | None = None) -> Light:
        """Measure a frame as the real scanner measures its stripe centres, and let auto exposure act on it."""
        now = time.monotonic() if now is None else now
        with self.lock:
            b = self.brightness(frame)
            pts = frame.points
            light = Light(now, 0, 0.0, 0.0, None, int(len(pts)))
            if len(pts):
                u, v = self._project(pts, 0.0)
                roi = (u > 0.15 * W) & (u < 0.85 * W) & (v > 0.15 * H) & (v < 0.85 * H) & (b >= DETECT)
                if roi.any():
                    grey = np.minimum(BLACK_LEVEL + b[roi], 255.0)
                    light = Light(now, int(roi.sum()), float(np.percentile(grey, 95)),
                                  float(np.mean(grey >= SATURATED_GREY)), float(np.mean(pts[roi, 2])),
                                  int(len(pts)))
            self._latest = (frame, b)
            self.readout.add(light)
            self.ae.observe(light)
            new = self.ae.step(now)
            if new is not None:
                self.camera = new
                self.ae.applied(now)          # the simulated camera takes new settings at once
        return light

    @staticmethod
    def _project(pts: np.ndarray, baseline: float):
        z = np.maximum(pts[:, 2], 1.0)
        return FOCAL * (pts[:, 0] - baseline) / z + W / 2, FOCAL * pts[:, 1] / z + H / 2

    # -- the picture
    def view(self, cams: str = "both", overlay: bool = True, width: int = 960) -> bytes | None:
        with self.lock:
            latest, cs = self._latest, self.camera
        if latest is None:
            return None
        frame, b = latest
        noise = 1.2 * cs.gain
        rng = np.random.default_rng()
        view = {"shape": (H, W)}
        for side, baseline in (("left", 0.0), ("right", BASELINE_MM)):
            img = np.zeros((H, W), np.float64)
            centres = np.zeros((0, 2), np.int16)
            if len(frame.points):
                u, v = self._project(frame.points, baseline)
                ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
                inside = (ui >= 1) & (ui < W - 1) & (vi >= 1) & (vi < H - 1)
                ui, vi, bb = ui[inside], vi[inside], b[inside]
                for du, dv, w in ((0, 0, 1.0), (1, 0, 0.55), (-1, 0, 0.55), (0, 1, 0.55), (0, -1, 0.55)):
                    np.maximum.at(img, (vi + dv, ui + du), bb * w)
                found = bb >= DETECT
                centres = np.unique(np.c_[ui[found], vi[found]].astype(np.int16), axis=0)
            rings = self._markers(img, frame, baseline, cs)
            grey = np.clip(BLACK_LEVEL + img + rng.normal(0.0, noise, img.shape), 0, 255).astype(np.uint8)
            view[side] = grey
            view[f"sat_{side}"] = np.packbits(grey >= SATURATED_GREY)
            view[f"centres_{side}"] = centres
            view[f"markers_{side}"] = rings
        return encode_jpeg(compose_view(view, cams, overlay, width))

    def _markers(self, img: np.ndarray, frame, baseline: float, cs: CameraSettings) -> np.ndarray:
        pose = frame.meta.get("true_pose")
        if pose is None or cs.marker_light <= 0:
            return np.zeros((0, 3), np.float32)
        T = np.asarray(pose, float)
        local = (np.asarray(RING) - T[:3, 3]) @ T[:3, :3]          # world -> sensor
        local = local[local[:, 2] > 50]
        if not len(local):
            return np.zeros((0, 3), np.float32)
        u, v = self._project(local, baseline)
        r = FOCAL * MARKER_MM / local[:, 2]
        level = min(240.0, cs.marker_light * cs.gain * 4.0)
        yy, xx = np.mgrid[0:H, 0:W]
        out = []
        for x, y, rad in zip(u, v, r):
            if -rad < x < W + rad and -rad < y < H + rad:
                x0, x1 = int(max(0, x - rad - 1)), int(min(W, x + rad + 2))
                y0, y1 = int(max(0, y - rad - 1)), int(min(H, y + rad + 2))
                d = np.hypot(xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y)
                img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], level * np.clip(rad + 0.5 - d, 0, 1))
                out.append((x, y, rad))
        return np.asarray(out, np.float32).reshape(-1, 3)

    # -- the API's view of it
    def payload(self) -> dict:
        with self.lock:
            cs, auto = self.camera, self.ae.status
        readout = self.readout.summary() if self.streaming else {}
        return camera_payload(cs, auto, readout, streaming=self.streaming, preview=self.preview,
                              gain_map="defaults")
