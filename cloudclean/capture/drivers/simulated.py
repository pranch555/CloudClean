"""Simulated structured-light scanner: a known part, a sensor moving on a turntable / hand-held path,
ray-cast depth frames with realistic noise, dropouts, motion blur and scanning mistakes.

It makes the whole live capture experience demoable without hardware and gives the tests exact ground
truth (`part_mesh()` is the surface every frame samples; `Frame.meta["true_pose"]` is the real pose even
when the pose is withheld from the session)."""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import open3d as o3d

from .base import Frame, ScannerDriver, camera_settings_schema, option, revo_settings, setting

RANGE_MM = (150.0, 400.0)
OPTIMAL_MM = 250.0
H_FOV_DEG = 30.0
EXPOSURE_BLUR_S = 0.004          # sensor integration time used for motion blur
EPISODE_PERIOD_S = 20.0
FAST_WINDOW = (6.0, 7.2)         # seconds within each period: the user moves much too fast
FAR_WINDOW = (13.0, 14.5)        # the user pulls the scanner out of range

MODES = {  # nominal fps, perpendicular noise sigma (mm)
    "cross_laser": {"fps": 20.0, "noise": 0.020},
    "parallel_lines": {"fps": 15.0, "noise": 0.025},
    "full_field_ir": {"fps": 10.0, "noise": 0.030},
}
OPTIMAL_EXPOSURE_MS = {"normal": 8.0, "dark": 16.0, "reflective": 3.0}


# --------------------------------------------------------------------------- parts
def part_mesh(shape: str = "pebble") -> o3d.geometry.TriangleMesh:
    """Closed test part in mm, resting on z = 0 and centred on the turntable axis. Deterministic."""
    if shape == "pebble":
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=90)
        v = np.asarray(mesh.vertices).copy()
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        shaped = np.stack([x * (1 + 0.2 * z), y * (1 + 0.15 * x), z], axis=1)
        bump_dir = np.array([0.75, 0.6, 0.12])
        bump_dir /= np.linalg.norm(bump_dir)
        bump = 1.0 + 0.3 * np.exp(-np.sum((v - bump_dir) ** 2, axis=1) / 0.08)
        v = shaped * bump[:, None] * np.array([40.0, 25.0, 15.0])
        mesh.vertices = o3d.utility.Vector3dVector(v)
    elif shape == "bracket":
        def box(w, d, h, at):
            b = o3d.geometry.TriangleMesh.create_box(w, d, h)
            return b.translate(at)

        mesh = box(70.0, 40.0, 6.0, (-35.0, -20.0, 0.0))            # base plate
        mesh += box(6.0, 40.0, 45.0, (-35.0, -20.0, 0.0))           # upright
        mesh += box(40.0, 5.0, 10.0, (-29.0, -2.5, 6.0))            # rib along the base
        boss = o3d.geometry.TriangleMesh.create_cylinder(radius=8.0, height=12.0, resolution=48)
        mesh += boss.translate((15.0, 5.0, 6.0))
    else:
        raise ValueError(f"Unknown simulated part '{shape}'")
    v = np.asarray(mesh.vertices)
    c = v.mean(axis=0)
    mesh.translate((-c[0], -c[1], -v[:, 2].min()))
    mesh.compute_vertex_normals()
    return mesh


def surface_color(points: np.ndarray) -> np.ndarray:
    stripe = (np.floor(points[:, 0] / 10.0) % 2).astype(float)
    rgb = np.stack([0.25 + 0.6 * stripe, 0.45 + 0.01 * points[:, 2], 0.75 - 0.5 * stripe], axis=1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- geometry helpers
def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Sensor->world pose for a sensor at `eye` looking at `target` (x right, y down, z forward)."""
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, (1.0, 0.0, 0.0))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _window_level(t: float, window: tuple[float, float], ramp: float = 0.25) -> float:
    """0..1 level of an episode window repeating every EPISODE_PERIOD_S, with smooth ramps."""
    r = t % EPISODE_PERIOD_S
    a, b = window
    return _smoothstep((r - a) / ramp) * (1.0 - _smoothstep((r - (b - ramp)) / ramp))


def _window_cumulative(t: float, window: tuple[float, float]) -> float:
    """Seconds spent inside the window up to time t."""
    k, r = divmod(t, EPISODE_PERIOD_S)
    a, b = window
    return k * (b - a) + min(max(r - a, 0.0), b - a)


class SimulatedDriver(ScannerDriver):
    id = "simulated"
    name = "Simulated scanner"
    kind = "simulated"
    description = ("A virtual MetroY-style scanner looking at a known test part. Settings behave like the real "
                   "ones (noise, density, frame rate, dark/shiny dropouts) and it makes deliberate mistakes "
                   "(moving too fast, going out of range) so the live guidance can be tried without hardware.")

    def __init__(self):
        super().__init__()
        self._scene = None
        self._mesh = None
        self._index = 0
        self._t0 = 0.0
        self._done = False
        self._rng = np.random.default_rng(0)
        self._cam = None                       # the simulated camera view (simulated_camera.py)
        self._render_lock = threading.Lock()   # the preview thread and the capture thread both render
        self._preview_thread: threading.Thread | None = None
        from ..metroy.camera import PreviewLease
        self._lease = PreviewLease(lambda: self.camera_preview(False))

    def settings_schema(self) -> list[dict]:
        return revo_settings(0.1) + [
            setting("turntable_speed", "Turntable speed", "number", 20.0, min=2.0, max=90.0, step=1.0,
                    unit="deg/s"),
            setting("part", "Simulated part", "select", "pebble",
                    options=[option("pebble", "Pebble (smooth freeform)"), option("bracket", "Bracket (prismatic)")]),
            setting("provide_pose", "Device provides pose", "boolean", True,
                    help="Off: frames arrive in sensor coordinates and CloudClean has to track them itself."),
            setting("follow_turntable", "Follow the simulated turntable", "boolean", True,
                    help="With 'Turntable' on and the simulated turntable connected, the part turns and tilts "
                         "exactly as that turntable does (step-and-scan programs can be tried without hardware)."),
            setting("simulate_mistakes", "Simulate mistakes", "boolean", True,
                    help="Every 20 s: a burst of too-fast motion and a moment out of range."),
            setting("colors", "Colour camera", "boolean", True),
            setting("realtime", "Real time", "boolean", True, help="Off: produce frames as fast as possible."),
            setting("max_frames", "Stop after frames", "number", 0, min=0, max=1_000_000, step=1,
                    help="0 = run until stopped."),
            setting("seed", "Random seed", "number", 0, min=0, max=1_000_000, step=1),
            setting("camera_surface", "Camera preset", "select", "match",
                    options=[option("match", "Same as Object surface"), option("general", "General"),
                             option("dark", "Dark"), option("reflective", "Reflective / shiny")],
                    help="The simulated camera's preset (Scan -> Camera view changes it live). Object surface is "
                         "what the simulated part is made of."),
            *camera_settings_schema(),
        ]

    def capabilities(self) -> dict:
        return {"range_mm": list(RANGE_MM), "optimal_mm": OPTIMAL_MM, "streaming": True,
                "provides_pose": bool(self.settings.get("provide_pose", True)), "camera": True}

    # ----------------------------------------------------------------- lifecycle
    def _connect(self) -> None:
        s = self.settings
        self._mesh = part_mesh(s["part"])
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.core.Tensor(np.asarray(self._mesh.vertices, dtype=np.float32)),
                                  o3d.core.Tensor(np.asarray(self._mesh.triangles, dtype=np.uint32)))
        bb = self._mesh.get_axis_aligned_bounding_box()
        self._center = np.asarray(bb.get_center())
        mode = MODES[s["scan_mode"]]
        self.fps = mode["fps"]
        self._rng = np.random.default_rng(int(s["seed"]))
        phases = np.random.default_rng(int(s["seed"]) + 99).uniform(0, 2 * np.pi, 12)
        self._phases = phases
        from .simulated_camera import SimulatedCamera
        self._cam = SimulatedCamera(s)

    def _start(self) -> None:
        self._lease.end()
        self._index = 0
        self._done = False
        self._t0 = time.monotonic()
        if self._cam is not None:
            self._cam.streaming, self._cam.preview = True, False

    def stop(self) -> None:
        self._lease.end()
        super().stop()
        if self._cam is not None:
            self._cam.streaming = self._cam.preview = False

    # ----------------------------------------------------------------- camera view (simulated_camera.py)
    def camera(self) -> dict | None:
        return self._cam.payload() if self._cam is not None else None

    def set_camera(self, changes: dict) -> dict:
        if self._cam is None:
            raise ValueError("Connect the scanner first")
        self._cam.set(changes)
        self.settings.update(self._cam.remembered())
        return self.camera()

    def camera_remembered(self) -> dict:
        return self._cam.remembered() if self._cam is not None else {}

    def camera_preview(self, on: bool) -> None:
        if self._cam is None:
            raise ValueError("Connect the scanner first")
        if not on:
            self._lease.end()
            if not self.running:
                self._cam.streaming = self._cam.preview = False
            return
        if self.running:
            return
        self._lease.renew()
        self._cam.streaming = self._cam.preview = True
        if self._preview_thread is None or not self._preview_thread.is_alive():
            self._preview_thread = threading.Thread(target=self._preview_loop, name="sim-camera-preview", daemon=True)
            self._preview_thread.start()

    def _preview_loop(self) -> None:
        """Frames for the camera view while not scanning: shown and measured, never handed to the session. One thread
        for the whole connection: it idles while nobody looks or while the scan itself renders the frames."""
        t0 = time.monotonic()
        while self.connected and self._cam is not None:
            cam = self._cam
            if not self.running and self._lease.active:
                with self._render_lock:
                    frame = self.render(time.monotonic() - t0)
                cam.observe(frame)
            elif not self.running and cam.preview:
                cam.streaming = cam.preview = False
            time.sleep(0.1)

    def camera_view(self, cams: str = "both", overlay: bool = True, width: int = 960) -> bytes | None:
        if self._cam is None:
            return None
        if not self.running and self._cam.preview:
            self._lease.renew()
        return self._cam.view(cams, overlay, width) if self._cam.streaming else None

    def status(self) -> dict:
        if self._cam is None:
            return {}
        from ..metroy.camera import camera_brief
        return {"camera": camera_brief(self.camera())}

    @property
    def exhausted(self) -> bool:
        return self._done

    # ----------------------------------------------------------------- motion
    def _turntable_pose(self) -> tuple[float, float] | None:
        """(angle_deg, tilt_deg) of a connected simulated turntable to follow, else None."""
        if not (self.settings.get("turntable") and self.settings.get("follow_turntable", True)):
            return None
        try:
            from ..turntable.simulated import follow_pose
        except ImportError:
            return None
        return follow_pose()

    def _pose_on_turntable(self, t: float, angle_deg: float, tilt_deg: float) -> np.ndarray:
        """Scanner fixed in the room, part on the platter: rotated by -angle about z (positive angle = clockwise
        seen from above) and tilted by `tilt` about the horizontal axis across the scanner's line of sight: the
        scanner's elevation over the part becomes 35 deg - tilt (which way the real table tips relative to the
        scanner is not known). Returns the sensor->part pose."""
        ph = self._phases
        az, el = ph[0], math.radians(35.0)
        target = self._center
        eye = target + OPTIMAL_MM * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        eye = eye + 0.15 * np.array([math.sin(1.1 * t + ph[6]), math.sin(1.3 * t + ph[7]), math.sin(0.8 * t + ph[8])])
        a, b = math.radians(angle_deg), math.radians(tilt_deg)
        rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        u = np.array([-math.sin(az), math.cos(az), 0.0])           # tilt axis: horizontal, across the view
        ux = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
        ru = math.cos(b) * np.eye(3) + math.sin(b) * ux + (1.0 - math.cos(b)) * np.outer(u, u)
        room_to_part = rz @ ru          # inverse of part_to_room = Ru(-tilt) @ Rz(-angle); pivot at the origin
        return look_at(room_to_part @ eye, room_to_part @ target, room_to_part @ np.array([0.0, 0.0, 1.0]))

    def sensor_pose(self, t: float) -> tuple[np.ndarray, str | None]:
        s = self.settings
        follow = self._turntable_pose()
        if follow is not None:
            return self._pose_on_turntable(t, *follow), None
        mistakes = s["simulate_mistakes"]
        ph = self._phases
        fast = _window_level(t, FAST_WINDOW) if mistakes else 0.0
        far = _window_level(t, FAR_WINDOW) if mistakes else 0.0
        boost = math.radians(160.0) * _window_cumulative(t, FAST_WINDOW) if mistakes else 0.0
        if s["turntable"]:
            az = -math.radians(s["turntable_speed"]) * t + boost + ph[0]
            el = math.radians(35.0 + 2.0 * math.sin(2 * math.pi * t / 23.0))
            dist = OPTIMAL_MM
            jitter = 0.15
        else:
            az = math.radians(25.0) * t + boost + ph[0]
            el = math.radians(40.0 + 22.0 * math.sin(2 * math.pi * t / 17.0 + ph[1]))
            dist = OPTIMAL_MM + 35.0 * math.sin(2 * math.pi * t / 11.0 + ph[2])
            jitter = 1.5
        dist += 300.0 * far
        target = self._center + jitter * np.array([math.sin(0.7 * t + ph[3]), math.sin(0.9 * t + ph[4]),
                                                   0.5 * math.sin(0.5 * t + ph[5])])
        eye = target + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        eye += jitter * np.array([math.sin(1.1 * t + ph[6]), math.sin(1.3 * t + ph[7]), math.sin(0.8 * t + ph[8])])
        if fast > 0:
            eye += 15.0 * fast * np.array([math.sin(2 * math.pi * 2.3 * t), math.cos(2 * math.pi * 1.7 * t),
                                           math.sin(2 * math.pi * 3.1 * t)])
        episode = "fast" if fast > 0.5 else ("far" if far > 0.5 else None)
        return look_at(eye, target), episode

    # ----------------------------------------------------------------- frames
    def read(self, timeout: float = 0.1) -> Frame | None:
        if not self.running:
            time.sleep(min(timeout, 0.05))
            return None
        max_frames = int(self.settings["max_frames"])
        if max_frames and self._index >= max_frames:
            self._done = True
            time.sleep(min(timeout, 0.02))
            return None
        if self.settings["realtime"]:
            now = time.monotonic()
            due = self._t0 + self._index / self.fps
            if due > now:
                if due - now > timeout:
                    time.sleep(timeout)
                    return None
                time.sleep(due - now)
            elif now - due > 2.0 / self.fps:  # the consumer was slow: skip frames like a real stream does
                self._index = int((now - self._t0) * self.fps)
                if max_frames and self._index >= max_frames:
                    self._done = True
                    return None
        with self._render_lock:
            frame = self.render(self._index / self.fps)
        self._index += 1
        if self._cam is not None:
            self._cam.observe(frame)
        return frame

    def _rays(self, pose: np.ndarray) -> np.ndarray:
        s = self.settings
        rng = self._rng
        f = 0.5 / math.tan(math.radians(H_FOV_DEG) / 2)       # focal length in units of image width
        aspect = 0.8
        mode = s["scan_mode"]
        if mode == "full_field_ir":
            scale = float(np.clip(math.sqrt(0.1 / s["point_distance"]), 0.6, 1.4))
            W, H = int(420 * scale), int(336 * scale)
            du, dv = rng.uniform(0, 1, 2)
            u, v = np.meshgrid((np.arange(W) + du) / W - 0.5, ((np.arange(H) + dv) / H - 0.5) * aspect)
            u, v = u.ravel(), v.ravel()
        else:
            n_lines, samples = (22, 3000) if mode == "parallel_lines" else (7, 3000)
            offset = rng.uniform(0, 1)
            pos = (np.arange(n_lines) + offset) / n_lines - 0.5
            along = np.linspace(-0.5, 0.5, samples)
            u = np.repeat(pos, samples)
            v = np.tile(along * aspect, n_lines)
            if mode == "cross_laser":  # horizontal lines too
                u = np.concatenate([u, np.tile(along, n_lines)])
                v = np.concatenate([v, np.repeat(pos * aspect, samples)])
        d_cam = np.stack([u / f, v / f, np.ones_like(u)], axis=1)
        d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
        d = d_cam @ pose[:3, :3].T
        origin = np.broadcast_to(pose[:3, 3], d.shape)
        return np.hstack([origin, d]).astype(np.float32)

    def render(self, t: float) -> Frame:
        """Frame at simulation time t (seconds)."""
        s = self.settings
        rng = self._rng
        pose, episode = self.sensor_pose(t)
        rays = self._rays(pose)
        res = self._scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = res["t_hit"].numpy()
        normals = res["primitive_normals"].numpy()
        hit = np.isfinite(t_hit)
        distance = float(np.median(t_hit[hit])) if hit.any() else float(np.linalg.norm(pose[:3, 3] - self._center))
        ok = hit & (t_hit >= RANGE_MM[0]) & (t_hit <= RANGE_MM[1])
        d = rays[ok, 3:].astype(float)
        n = normals[ok].astype(float)
        depth = t_hit[ok].astype(float)
        cos_inc = np.abs(np.einsum("ij,ij->i", n, d))
        surface = s["surface"]
        brightness = float(s["laser_brightness"])

        # --- which points the sensor actually measures
        keep_p = np.full(len(depth), 0.985)
        if surface == "dark":
            keep_p[:] = min(0.25 + 0.07 * brightness, 0.95)
        elif surface == "reflective":
            keep_p[:] = 0.8 - 0.04 * max(brightness - 5.0, 0.0)
            keep_p[cos_inc > 0.93] = 0.12  # specular glare
        if s["exposure_mode"] == "manual":
            ratio = s["exposure_ms"] / OPTIMAL_EXPOSURE_MS[surface]
            keep_p *= math.exp(-math.log(ratio) ** 2 / 0.8)
        keep_p *= np.clip((cos_inc - 0.12) / 0.15, 0, 1)  # grazing angles return nothing
        keep = rng.random(len(depth)) < keep_p
        d, n, depth, cos_inc = d[keep], n[keep], depth[keep], cos_inc[keep]
        pts = pose[:3, 3] + d * depth[:, None]

        # --- noise perpendicular to the surface
        sigma = MODES[s["scan_mode"]]["noise"] * np.maximum((depth / OPTIMAL_MM) ** 2, 0.7)
        sigma *= {"normal": 1.0, "dark": 1.3, "reflective": 1.8}[surface]
        if s["exposure_mode"] == "manual":
            ratio = s["exposure_ms"] / OPTIMAL_EXPOSURE_MS[surface]
            sigma *= 1.0 + 0.3 * abs(math.log(ratio))
        noisy = pts + n * (rng.normal(size=len(pts)) * sigma)[:, None]
        colors = surface_color(pts) if s["colors"] else None

        # --- rare outliers along the ray (more on shiny parts)
        n_out = int(len(noisy) * (0.01 if surface == "reflective" else 0.001))
        if n_out:
            idx = rng.choice(len(noisy), n_out, replace=False)
            noisy[idx] += d[idx] * rng.uniform(-5, 5, n_out)[:, None]

        # --- motion blur: each point is measured at a slightly different instant
        inv0 = np.linalg.inv(pose)
        pose1, _ = self.sensor_pose(t + EXPOSURE_BLUR_S)
        inv1 = np.linalg.inv(pose1)
        p0 = noisy @ inv0[:3, :3].T + inv0[:3, 3]
        p1 = noisy @ inv1[:3, :3].T + inv1[:3, 3]
        local = p0 + (p1 - p0) * rng.uniform(-0.5, 0.5, len(p0))[:, None]

        tracking = s["tracking"]
        meta = {"distance_mm": round(distance, 2), "exposure_ms": s["exposure_ms"] if s["exposure_mode"] == "manual"
                else OPTIMAL_EXPOSURE_MS[surface], "tracking": tracking, "scan_mode": s["scan_mode"],
                "episode": episode, "sim_time": round(t, 4), "true_pose": pose.tolist()}
        device_pose = None
        if s["provide_pose"]:
            device_pose = pose.copy()
            err = {"marker": 0.002, "hybrid": 0.003, "feature": 0.005, "texture": 0.008}[tracking]
            device_pose[:3, 3] += rng.normal(scale=err, size=3)
        return Frame(points=local.astype(np.float32), colors=colors, pose=device_pose, timestamp=t, meta=meta,
                     coordinates="sensor")
