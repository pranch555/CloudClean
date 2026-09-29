"""Generic depth cameras with Linux aarch64 SDKs (optional): Intel RealSense (`pyrealsense2`) and Orbbec
(`pyorbbecsdk`). Frames arrive in sensor coordinates without pose; the capture session tracks them.

These cameras are far less accurate than a metrology scanner (millimetre-level noise) but useful for
large parts, rough coverage checks and for trying live capture on the DGX Spark directly."""
from __future__ import annotations

import importlib.util
import time

import numpy as np

from .base import Frame, ScannerDriver, option, setting


def _depth_settings(default_range=(200.0, 1000.0)) -> list[dict]:
    return [
        setting("resolution", "Depth resolution", "select", "848x480",
                options=[option("640x480", "640 x 480"), option("848x480", "848 x 480"),
                         option("1280x720", "1280 x 720")]),
        setting("fps", "Frame rate", "number", 15, min=5, max=90, step=1, unit="fps"),
        setting("min_distance", "Minimum distance", "number", default_range[0], min=50.0, max=5000.0, step=10.0,
                unit="mm"),
        setting("max_distance", "Maximum distance", "number", default_range[1], min=100.0, max=10000.0, step=10.0,
                unit="mm"),
        setting("decimate", "Keep every n-th pixel", "number", 2, min=1, max=8, step=1),
        setting("point_distance", "Point distance", "number", 1.0, min=0.1, max=10.0, step=0.1, unit="mm"),
        setting("turntable", "Turntable", "boolean", False),
    ]


class RealSenseDriver(ScannerDriver):
    id = "realsense"
    name = "Intel RealSense depth camera"
    kind = "depth_camera"
    description = "Live depth frames from an Intel RealSense D4xx camera (pyrealsense2)."

    def __init__(self):
        super().__init__()
        self._pipeline = None
        self._pc = None

    def availability(self) -> tuple[bool, str]:
        if importlib.util.find_spec("pyrealsense2") is None:
            return False, ("pyrealsense2 is not installed. Install Intel's librealsense Python bindings "
                           "(build from source on Linux aarch64) and connect a D4xx camera.")
        return True, ""

    def settings_schema(self) -> list[dict]:
        return _depth_settings((200.0, 1200.0))

    def capabilities(self) -> dict:
        s = self.settings
        rng = [s.get("min_distance", 200.0), s.get("max_distance", 1200.0)]
        return {"range_mm": rng, "optimal_mm": (rng[0] + rng[1]) / 2, "streaming": True, "provides_pose": False}

    def _connect(self) -> None:
        import pyrealsense2 as rs

        w, h = (int(x) for x in self.settings["resolution"].split("x"))
        self._rs = rs
        self._config = rs.config()
        self._config.enable_stream(rs.stream.depth, w, h, rs.format.z16, int(self.settings["fps"]))
        self._pc = rs.pointcloud()

    def _start(self) -> None:
        rs = self._rs
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(self._config)
        self._scale_mm = profile.get_device().first_depth_sensor().get_depth_scale() * 1000.0

    def read(self, timeout: float = 0.1) -> Frame | None:
        if not self.running or self._pipeline is None:
            time.sleep(min(timeout, 0.05))
            return None
        ok, frames = self._pipeline.try_wait_for_frames(int(timeout * 1000))
        if not ok:
            return None
        depth = frames.get_depth_frame()
        if not depth:
            return None
        vertices = np.asanyarray(self._pc.calculate(depth).get_vertices()).view(np.float32).reshape(-1, 3)
        h, w = depth.get_height(), depth.get_width()
        step = int(self.settings["decimate"])
        pts = vertices.reshape(h, w, 3)[::step, ::step].reshape(-1, 3) * 1000.0   # metres -> mm
        z = pts[:, 2]
        keep = (z >= self.settings["min_distance"]) & (z <= self.settings["max_distance"])
        pts = pts[keep]
        meta = {"distance_mm": float(np.median(pts[:, 2])) if len(pts) else None}
        return Frame(points=pts, timestamp=depth.get_timestamp() / 1000.0, meta=meta, coordinates="sensor")

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        super().stop()


class OrbbecDriver(ScannerDriver):
    id = "orbbec"
    name = "Orbbec depth camera"
    kind = "depth_camera"
    description = "Live depth frames from an Orbbec Femto / Gemini camera (pyorbbecsdk)."

    def __init__(self):
        super().__init__()
        self._pipeline = None

    def availability(self) -> tuple[bool, str]:
        if importlib.util.find_spec("pyorbbecsdk") is None:
            return False, ("pyorbbecsdk is not installed. Install Orbbec's Python SDK (aarch64 wheels are "
                           "available from Orbbec's GitHub releases) and connect the camera.")
        return True, ""

    def settings_schema(self) -> list[dict]:
        return _depth_settings((250.0, 1500.0))

    def capabilities(self) -> dict:
        s = self.settings
        rng = [s.get("min_distance", 250.0), s.get("max_distance", 1500.0)]
        return {"range_mm": rng, "optimal_mm": (rng[0] + rng[1]) / 2, "streaming": True, "provides_pose": False}

    def _connect(self) -> None:
        import pyorbbecsdk as ob

        self._ob = ob
        self._pipeline = ob.Pipeline()
        self._config = ob.Config()
        profiles = self._pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        w, h = (int(x) for x in self.settings["resolution"].split("x"))
        try:
            profile = profiles.get_video_stream_profile(w, h, ob.OBFormat.Y16, int(self.settings["fps"]))
        except Exception:
            profile = profiles.get_default_video_stream_profile()
        self._config.enable_stream(profile)

    def _start(self) -> None:
        self._pipeline.start(self._config)
        self._intrinsics = self._pipeline.get_camera_param().depth_intrinsic

    def read(self, timeout: float = 0.1) -> Frame | None:
        if not self.running or self._pipeline is None:
            time.sleep(min(timeout, 0.05))
            return None
        frames = self._pipeline.wait_for_frames(int(timeout * 1000))
        depth = frames.get_depth_frame() if frames is not None else None
        if depth is None:
            return None
        w, h = depth.get_width(), depth.get_height()
        z = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(h, w).astype(np.float32) * depth.get_depth_scale()
        step = int(self.settings["decimate"])
        z = z[::step, ::step]
        v, u = np.mgrid[0:h:step, 0:w:step]
        k = self._intrinsics
        keep = (z >= self.settings["min_distance"]) & (z <= self.settings["max_distance"])
        zz = z[keep]
        pts = np.stack([(u[keep] - k.cx) * zz / k.fx, (v[keep] - k.cy) * zz / k.fy, zz], axis=1)
        meta = {"distance_mm": float(np.median(zz)) if len(zz) else None}
        return Frame(points=pts, timestamp=depth.get_timestamp() / 1000.0, meta=meta, coordinates="sensor")

    def stop(self) -> None:
        if self._pipeline is not None and self.running:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        super().stop()
