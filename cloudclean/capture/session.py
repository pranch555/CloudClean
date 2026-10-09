"""A live capture session: reads frames from a driver in a background thread, tracks and fuses them into
a voxel map and computes guidance.

Data kept per session (all numpy, bounded):

* **full resolution** – accepted points in per-frame float32 chunks (+ uint8 colours). At most
  `max_points_per_cell` points are kept per point-distance cell, so rescanning an area does not grow memory
  without bound. They are stored as measured.
* **surface field** (`fusion` = "average", the default) – every fused point of every frame, also where the cells
  above are full, summed into per-cell surface moments (fusion.SurfaceField). At save time the kept points move along
  the normal onto the surface those moments average out (fusion.project); edges, corners and anything that is not
  one clean surface keep the measured points. On the user's bust this took the surface thickness from 0.24 to
  0.075 mm with the same points (fusion.py has the numbers).
* **display map** – the first point in every `display_voxel` cell, append-only so clients can stream it
  incrementally; each has a density ratio (1.0 = target density reached).
* **density grid** – ~2.5 mm surface cells with point counts and summed view directions (guidance).
* **free-space grid** – coarse cells rays passed through (guidance, concavities).

Tracking for frames without a device pose: coarse-to-fine point-to-plane ICP of the voxelised frame
against the display map (cropped around the predicted pose, normals refreshed every few frames) starting
from a constant-velocity prediction; fitness below `min_fitness` means lost, and relocalisation runs
FPFH + RANSAC against the whole map (rate limited) until a frame fits again.

Whole-scan chunks (`coordinates="scan"`, bridge / folder drivers) are never fused blindly: the first one starts
the model, every later one is aligned and assessed against the model (`register.assess_pair`: new coverage,
alignment confidence, doubled surfaces). `fuse_mode` decides: "always" fuses, "auto" fuses only on a "merge"
recommendation, "ask" (default) keeps every later scan aside as a *pending scan* until the user decides
(`resolve_pending`: fuse | discard | keep_separate). Live frames are fused as before."""
from __future__ import annotations

import threading
import time
import traceback
import struct
import uuid
from collections import deque

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ..io import estimate_spacing
from ..register import MergeParams, assess_pair, scan_reasons
from .drivers.base import Frame, ScannerDriver, option, resolve_settings, setting
from .fusion import FUSION_CELL_MM, SurfaceField, project
from .grid import VoxelCounter, neighbor_keys, voxel_keys
from .guidance import DENSE_RATIO, analyze, smoothed_density

reg = o3d.pipelines.registration

SESSION_SETTINGS = [
    setting("fusion", "Fusion", "select", "average",
            options=[option("average", "Average every frame (thin, accurate surface)"),
                     option("raw", "Keep the raw points (every frame's noise stays)")],
            help="Average: every frame is averaged into the surface and the saved points lie on that average - the "
                 "scanner's noise and the tracking jitter cancel out, edges and corners stay as measured. Raw: the "
                 "first points measured in each cell are saved as they are."),
    setting("max_points_per_cell", "Max points per point-distance cell", "number", 3, min=1, max=100, step=1,
            help="Bounds memory on long sessions: extra passes over an area stop adding points once reached."),
    setting("max_points", "Point budget", "number", 0, min=0, max=100_000_000, step=100_000,
            help="Upper limit on the saved cloud. 0 = keep every fused point. The cloud is thinned evenly, so "
                 "the shape and its measured size are unchanged; point distance grows to match."),
    setting("display_voxel", "Live view resolution", "number", 0.0, min=0.0, max=20.0, step=0.1, unit="mm",
            help="0 = automatic."),
    setting("speed_limit", "Maximum scanner speed", "number", 150.0, min=10.0, max=5000.0, step=10.0,
            unit="mm/s", help="Frames with faster apparent motion are not fused (motion blur)."),
    setting("density_cell", "Guidance cell size", "number", 2.5, min=0.5, max=50.0, step=0.5, unit="mm"),
    setting("min_fitness", "Tracking lost below fitness", "number", 0.35, min=0.05, max=0.95, step=0.05),
    setting("alignment", "Align exported scans", "select", "auto",
            options=[option("auto", "Automatic registration to the model"),
                     option("none", "Keep coordinates (scans already aligned)")]),
    setting("fuse_mode", "Add exported scans to the model", "select", "ask",
            options=[option("ask", "Ask me for every further scan"),
                     option("auto", "Automatically when it adds coverage and aligns reliably"),
                     option("always", "Always (no check)")],
            help="Each export is a complete scan. Merging scans that cover the same area, or that do not align "
                 "exactly, creates doubled, layered surfaces - so by default every further scan is checked and "
                 "kept aside until you decide."),
]
PENDING_DECISIONS = ("fuse", "discard", "keep_separate")
SESSION_KEYS = {s["key"] for s in SESSION_SETTINGS}

MAX_DISPLAY_POINTS = 2_000_000
MAX_FRAME_GAP_S = 0.5            # frames further apart than this cannot be checked for motion blur
DENSITY_BUCKETS = np.array([0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0], np.float32)
SEVERITY_RANK = {"error": 3, "warning": 2, "info": 1, "ok": 0}


def _cloud(points, normals=None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    return pcd


def _transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]


class CaptureSession:
    def __init__(self, driver: ScannerDriver, settings: dict | None = None, log=None):
        settings = dict(settings or {})
        self.id = uuid.uuid4().hex[:10]
        self.driver = driver
        self.options = resolve_settings(SESSION_SETTINGS, {k: v for k, v in settings.items() if k in SESSION_KEYS})
        self._driver_input = {k: v for k, v in settings.items() if k not in SESSION_KEYS}
        self.driver_settings: dict = {}
        self.capabilities: dict = {}
        self.lock = threading.RLock()
        self._guidance_lock = threading.Lock()
        self.state = "created"      # created connected running paused stopped finished error closed
        self.error: str | None = None
        self.created = time.time()
        self.started_at: float | None = None
        self.running_time = 0.0
        self._run_since: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._log_fn = log
        self.logs: deque[str] = deque(maxlen=300)
        self.saved_asset_id: str | None = None
        self.saved_at_points = 0
        self.epoch = 0
        self.stats = {"frames": 0, "fused_frames": 0, "scans": 0,
                      "dropped": {"too_fast": 0, "tracking_lost": 0, "empty": 0}}
        self.scans: list[dict] = []
        self.pending: dict[str, dict] = {}     # scan_id -> scan chunk kept aside until the user decides
        self._resolved: deque[tuple[str, str]] = deque(maxlen=200)
        self.trajectory: deque[dict] = deque(maxlen=20_000)
        self.guidance: dict | None = None
        self.guidance_version = 0
        self.history: dict[str, dict] = {}
        self.coverage_timeline: list[list[float]] = []
        self._fps_times: deque[float] = deque(maxlen=30)
        self._recent_counts: deque[int] = deque(maxlen=50)
        self._tick = self._new_tick()
        self._last_heavy = 0.0
        self._heavy: dict | None = None
        self._last_light: dict = {}
        self._reset_map()

    # ----------------------------------------------------------------- helpers
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.logs.append(line)
        if self._log_fn:
            self._log_fn(msg)

    def _new_tick(self) -> dict:
        return {"frames": 0, "fast": 0, "max_speed": 0.0, "lost": 0, "empty": 0, "last": None, "device_reason": None,
                "device_markers": None}

    def _reset_map(self, display_voxel: float | None = None) -> None:
        with self.lock:
            self.point_distance: float | None = None
            self.fine = VoxelCounter()
            self.field: SurfaceField | None = None        # created with the first frame (needs the point distance)
            self.fusion_stats: dict | None = None
            self.density = VoxelCounter(vectors=True)
            self.free = VoxelCounter()
            self.display = VoxelCounter()
            self.chunks: list[tuple[np.ndarray, np.ndarray | None]] = []
            self.n_points = 0
            self.has_colors = False
            self.bbox = None
            self.display_voxel = display_voxel
            self.disp_n = 0
            self.disp_xyz = np.empty((0, 3), np.float32)
            self.disp_rgb = np.empty((0, 3), np.uint8)
            self.disp_density = np.empty(0, np.float32)
            self.disp_dkey = np.empty(0, np.int64)
            self.disp_bucket = np.empty(0, np.uint8)
            self.density_version = 0
            self.live_frame: np.ndarray | None = None   # the latest frame, as shown live (fused or not)
            self.live_version = 0
            self.live_view: dict | None = None             # its pose and markers, for the scanner's-eye view
            self.density_updates: deque[tuple[int, np.ndarray]] = deque(maxlen=30)
            self._dirty_dkeys: list[np.ndarray] = []
            self._model = None
            self._model_n = 0
            self._model_frames = 0
            self._reloc_features = None
            self._T_prev = self._T_prev2 = None
            self._t_prev = None
            self._pts_prev = None
            self._last_reloc = -1e9
            self.tracking = {"state": "none", "fitness": None, "rmse_mm": None}
            self.epoch += 1

    @property
    def cell(self) -> float:
        return float(self.options["density_cell"])

    @property
    def target_density(self) -> float:
        pd = self.point_distance or 0.1
        return 1.0 / (pd * pd)

    # ----------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        self.driver_settings = self.driver.connect(self._driver_input)
        self.capabilities = self.driver.capabilities()
        pd = float(self.driver_settings.get("point_distance") or 0)
        self._configured_pd = pd if pd > 0 else None
        self.state = "connected"
        self.log(f"Connected to {self.driver.name}")

    def start(self) -> None:
        if self.state in ("running", "paused"):
            return
        if self.state not in ("connected", "stopped", "finished"):
            raise RuntimeError(f"Cannot start a capture that is {self.state}")
        self.driver.start()
        self._stop.clear()
        self.started_at = self.started_at or time.time()
        self._run_since = time.monotonic()
        self.state = "running"
        self._thread = threading.Thread(target=self._run, name=f"capture-{self.id}", daemon=True)
        self._thread.start()
        self.log("Capture started")

    def pause(self) -> None:
        if self.state == "running":
            self._accumulate_time()
            self.state = "paused"
            self.log("Paused")

    def resume(self) -> None:
        if self.state == "paused":
            self._run_since = time.monotonic()
            # the scanner probably moved while paused: forget the motion model
            self._T_prev2 = None
            self.state = "running"
            self.log("Resumed")

    def _accumulate_time(self) -> None:
        if self._run_since is not None:
            self.running_time += time.monotonic() - self._run_since
            self._run_since = None

    def stop(self, final_state: str = "stopped") -> None:
        thread = self._thread
        self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=30)
        self._thread = None
        try:
            self.driver.stop()
        except Exception as exc:
            self.log(f"driver stop failed: {exc}")
        if self.state in ("running", "paused", "finished"):
            self._accumulate_time()
            self.state = final_state if self.state != "finished" else "finished"
            self.update_guidance(full=True)
            self.log("Capture stopped")

    def close(self) -> None:
        self.stop()
        try:
            self.driver.disconnect()
        except Exception as exc:
            self.log(f"driver disconnect failed: {exc}")
        self.state = "closed"

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        last_tick = 0.0
        failures = 0
        try:
            while not self._stop.is_set():
                if self.state == "paused":   # leave frames with the driver (bridge uploads stay queued)
                    self._stop.wait(0.1)
                    frame = None
                else:
                    frame = self.driver.read(0.1)
                if frame is not None:
                    try:
                        self.process_frame(frame)
                        failures = 0
                    except Exception as exc:  # one bad frame / export must not end the capture
                        failures += 1
                        self.log(f"Frame could not be processed: {exc}")
                        if failures >= 20:
                            raise
                elif self.state != "paused" and self.driver.exhausted:
                    self._accumulate_time()
                    self.state = "finished"
                    self.update_guidance(full=True)
                    self.log("Scanner stream ended")
                    return
                now = time.monotonic()
                if now - last_tick >= 0.2:
                    self.update_guidance()
                    last_tick = now
        except Exception as exc:
            self.error = str(exc)
            self.state = "error"
            self.log(traceback.format_exc())

    # ----------------------------------------------------------------- frames
    def process_frame(self, frame: Frame) -> dict:
        """Track + fuse one frame (called by the capture thread; tests may call it directly)."""
        self.stats["frames"] += 1
        self._fps_times.append(time.monotonic())
        if frame.coordinates == "scan":
            return self._process_scan(frame)
        return self._process_live(frame)

    def _distance_state(self, distance: float | None) -> str:
        rng = self.capabilities.get("range_mm")
        if distance is None or not rng:
            return "unknown"
        if distance < rng[0]:
            return "too_close"
        if distance > rng[1]:
            return "too_far"
        return "ok"

    def _process_live(self, frame: Frame) -> dict:
        pts = frame.points.astype(np.float64)
        n = len(pts)
        distance = frame.meta.get("distance_mm")
        if distance is None and frame.coordinates == "sensor" and n:
            distance = float(np.median(pts[:, 2]))
        state = self._distance_state(distance)
        near = frame.meta.get("near_mm")
        rng = self.capabilities.get("range_mm")
        if state == "ok" and near is not None and rng and near < rng[0]:
            # the part as a whole is in range but its nearest side is not: the scanner cannot measure it there (a
            # bust's head at 165-190 mm came out missing while the distance read 285)
            state = "too_close"
        info = {"points": n, "distance_mm": distance, "distance_state": state, "near_mm": near,
                "speed_mm_s": None, "too_fast": False, "tracking": self.tracking["state"], "fused": False,
                "episode": frame.meta.get("episode")}
        tick = self._tick
        tick["frames"] += 1
        if n < 50:
            self.stats["dropped"]["empty"] += 1
            tick["empty"] += 1
            tick["last"] = info
            if frame.pose is not None:  # keep the motion reference current for the speed check
                self._T_prev2, self._T_prev, self._t_prev = self._T_prev, frame.pose, frame.timestamp
            return info
        if info["distance_state"] == "ok":
            self._recent_counts.append(n)

        fitness = rmse = None
        if frame.coordinates == "world":
            T, state = np.eye(4), "world"
        elif frame.pose is not None:
            T, state = frame.pose, frame.meta.get("tracking_label", "device")   # e.g. "markers"
        elif frame.meta.get("device_tracking") in ("lost", "no_markers"):
            # the device tracks itself (e.g. by markers) and could not place this frame: never guess it by ICP
            T = self._T_prev if self._T_prev is not None else self._initial_pose(pts)
            state = "lost"
            tick["device_reason"] = frame.meta["device_tracking"]
            tick["device_markers"] = frame.meta.get("markers")
        else:
            T, state, fitness, rmse = self._track(pts, frame.timestamp)
        self._publish_live(pts, T, frame.meta.get("marker_points"), state, frame.meta.get("map_points"))
        # no fit yet (nothing matched) gives an infinite rmse: report it as unknown
        self.tracking = {"state": state,
                         "fitness": round(float(fitness), 3) if fitness is not None and np.isfinite(fitness) else None,
                         "rmse_mm": round(float(rmse), 4) if rmse is not None and np.isfinite(rmse) else None}
        info["tracking"] = state

        if state == "lost":
            self.stats["dropped"]["tracking_lost"] += 1
            tick["lost"] += 1
        else:
            if frame.coordinates == "sensor" and self._T_prev is not None and self._t_prev is not None:
                dt = frame.timestamp - self._t_prev
                if dt > MAX_FRAME_GAP_S:
                    info["resync"] = True     # motion since the last usable frame is unknown: skip this one
                elif dt > 1e-6:
                    sample = pts[:: max(1, n // 300)]
                    world = _transform(sample, T)
                    prev_local = _transform(world, np.linalg.inv(self._T_prev))
                    disp = np.linalg.norm(prev_local - sample, axis=1)
                    speed = float(np.percentile(disp, 90) / dt)
                    info["speed_mm_s"] = round(speed, 1)
                    tick["max_speed"] = max(tick["max_speed"], speed)
                    if speed > self.options["speed_limit"]:
                        info["too_fast"] = True
                        tick["fast"] += 1
            if info["too_fast"]:
                self.stats["dropped"]["too_fast"] += 1
            elif frame.meta.get("preview_only"):
                info["preview"] = True     # tracked and shown, not fused (e.g. while mapping markers first)
            elif not info.get("resync"):
                origin = T[:3, 3] if frame.coordinates == "sensor" else None
                self._fuse(_transform(pts, T), frame.colors, origin)
                self.stats["fused_frames"] += 1
                info["fused"] = True
            self._T_prev2, self._T_prev, self._t_prev = self._T_prev, T, frame.timestamp
        self.trajectory.append({"frame": self.stats["frames"], "t": frame.timestamp,
                                "pose": None if state == "lost" else np.asarray(T).copy(),
                                "true_pose": frame.meta.get("true_pose"), "tracking": state,
                                "fused": info["fused"]})
        tick["last"] = info
        return info

    def _publish_live(self, pts: np.ndarray, T: np.ndarray, markers=None, state: str = "", map_points=None,
                      limit: int = 15_000):
        """The current frame for the live preview - shown whether or not it is fused, so the operator always sees
        what the scanner sees (placed at its pose, or the last known one while tracking is lost) - plus the pose and
        the markers it saw, so the viewer can look from the scanner as Revo Metro does."""
        step = max(1, len(pts) // limit)
        mk = np.asarray(markers, float).reshape(-1, 3) if markers is not None and len(markers) else np.zeros((0, 3))
        with self.lock:
            self.live_frame = _transform(pts[::step], T).astype(np.float32)
            self.live_view = {"type": "live_view", "pose": np.asarray(T, float).ravel().round(4).tolist(),
                              "markers": _transform(mk, T).round(3).tolist() if len(mk) else [],
                              "tracking": state,
                              "map": np.asarray(map_points, float).round(3).tolist() if map_points is not None else None}
            self.live_version += 1

    def _process_scan(self, frame: Frame) -> dict:
        pts = frame.points.astype(np.float64)
        name = frame.meta.get("name") or frame.meta.get("file") or f"scan {self.stats['scans'] + 1}"
        info = {"name": name, "points": len(pts)}
        if len(pts) < 50:
            self.stats["dropped"]["empty"] += 1
            self.log(f"{name}: {frame.meta.get('error') or 'no points'} - skipped")
            info["error"] = frame.meta.get("error") or "no points"
            self.scans.append(info)
            return info
        if self.point_distance is None:
            self.point_distance = self._configured_pd or max(estimate_spacing(pts), 1e-3)
            self.log(f"Point distance {self.point_distance:.4f} mm")
        T = np.eye(4)
        fuse = True
        if self.n_points:
            T, assessment = self._assess_scan(pts, name)
            info.update(method=assessment["method"], fitness=assessment["fitness"], rmse=assessment["rmse"],
                        assessment=assessment)
            mode = self.options["fuse_mode"]
            fuse = mode == "always" or (mode == "auto" and assessment["recommendation"] == "merge")
            self.log(f"{name}: aligned ({info['method']}, fitness {info['fitness']:.3f}), new surface "
                     f"{assessment['coverage_gain_pct']:.1f}% -> {assessment['recommendation']}")
        else:
            info["method"] = "reference"
        info["transform"] = np.asarray(T).round(8).tolist()
        self.scans.append(info)
        self.tracking = {"state": "n/a", "fitness": info.get("fitness"), "rmse_mm": info.get("rmse")}
        self._tick["frames"] += 1
        self._tick["last"] = {"points": len(pts), "distance_mm": None, "distance_state": "unknown",
                              "speed_mm_s": None, "too_fast": False, "tracking": "n/a", "fused": fuse}
        if fuse:
            info["status"] = "fused"
            self._fuse(_transform(pts, T), frame.colors, None)
            self.stats["fused_frames"] += 1
        else:
            scan_id = uuid.uuid4().hex[:10]
            info.update(status="pending", scan_id=scan_id)
            colors = frame.colors if frame.colors is not None and len(frame.colors) == len(pts) else None
            with self.lock:
                self.pending[scan_id] = {"scan_id": scan_id, "name": name, "points": pts.astype(np.float32),
                                         "colors": colors, "transform": np.asarray(T), "info": info,
                                         "assessment": info["assessment"], "created": time.time(),
                                         "file": frame.meta.get("file")}
            self.log(f"{name}: kept aside - fuse, discard or keep it separate ({info['assessment']['headline']})")
        self.update_guidance(full=True)   # analyse coverage right away
        self.stats["scans"] += 1          # after the guidance, so status and guidance agree
        return info

    def _assess_scan(self, pts: np.ndarray, name: str) -> tuple[np.ndarray, dict]:
        """Align a scan chunk to the model and assess whether fusing it helps (see register.assess_pair)."""
        spacing = self.point_distance
        cloud = _cloud(pts)
        target = self.map_cloud(voxel=spacing, max_points=1_500_000)
        diag = float(np.linalg.norm(target.get_max_bound() - target.get_min_bound()))
        voxel = max(diag / 60.0, spacing * 4)
        if self.options["alignment"] == "none":
            params = MergeParams(method="none")
        else:
            quick = reg.evaluate_registration(cloud.voxel_down_sample(spacing * 2), target, spacing * 3).fitness
            params = MergeParams(method="icp" if quick >= 0.5 else "auto")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            T, a = assess_pair(cloud, target, params, spacing, voxel, log=self.log)
        reasons = scan_reasons(f"'{name}'", a, params)
        if a["recommendation"] == "merge":
            reasons.append("It adds new surface and aligns reliably - fusing is recommended.")
        elif a["recommendation"] == "use_best":
            reasons.append("Discard it (or keep it separate) unless it is better than what was fused.")
        else:
            reasons.append("Check the alignment before fusing it.")
        summary = {k: a.get(k) for k in ("recommendation", "method", "fitness", "rmse", "overlap",
                                         "coverage_gain_pct", "coverage_gain_mm2", "layering_mm", "separation_ratio",
                                         "doubled_surface", "ambiguous", "alternative_angle", "already_aligned",
                                         "noise_mm", "confident", "adds_coverage")}
        summary["reasons"] = reasons
        # the main concern for short messages: alignment problems first, then "adds only x % new surface"
        concerns = [r for r in reasons if r.startswith(f"'{name}':")] + [r for r in reasons if "adds only" in r]
        summary["headline"] = concerns[0] if concerns and a["recommendation"] != "merge" else reasons[-1]
        return T, summary

    # ----------------------------------------------------------------- pending scans
    def pending_scans(self) -> list[dict]:
        with self.lock:
            items = list(self.pending.values())
        return [{"scan_id": p["scan_id"], "name": p["name"], "points": len(p["points"]), "file": p.get("file"),
                 "created": p["created"], "assessment": p["assessment"]} for p in items]

    def get_pending(self, scan_id: str) -> dict:
        with self.lock:
            if scan_id not in self.pending:
                raise KeyError(scan_id)
            return self.pending[scan_id]

    def resolve_pending(self, scan_id: str, decision: str) -> dict:
        """fuse: add it to the model with its assessed pose; discard / keep_separate: drop it from the session
        (the caller saves a kept scan as its own asset first)."""
        if decision not in PENDING_DECISIONS:
            raise ValueError(f"decision must be one of: {', '.join(PENDING_DECISIONS)}")
        with self.lock:
            if scan_id not in self.pending:
                raise KeyError(scan_id)
            item = self.pending.pop(scan_id)
            if decision == "fuse":
                self._fuse(_transform(item["points"], item["transform"]), item["colors"], None)
                self.stats["fused_frames"] += 1
            item["info"]["status"] = {"fuse": "fused", "discard": "discarded", "keep_separate": "kept_separate"}[
                decision]
            self._resolved.append((scan_id, decision))
        self.log(f"{item['name']}: {item['info']['status'].replace('_', ' ')}")
        self.update_guidance(full=True)
        return item

    # ----------------------------------------------------------------- tracking
    def _initial_pose(self, pts: np.ndarray) -> np.ndarray:
        """World frame for untracked devices: Z up = sensor up, origin at the first frame's centroid."""
        R = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # sensor x->X, y(down)->-Z, z(forward)->Y
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ pts.mean(axis=0)
        return T

    def _tracking_model(self):
        with self.lock:
            n = self.disp_n
            grew = n - self._model_n
            stale = self._model is None or grew > 0.2 * max(self._model_n, 1) or (
                self._model_frames >= 5 and grew > 0.02 * max(self._model_n, 1))
            if not stale:
                self._model_frames += 1
                return self._model
            pts = self.disp_xyz[:n].astype(np.float64)
        v = self.display_voxel
        model = {}
        for level, voxel in (("fine", None), ("coarse", max(2.0, v * 4))):
            pcd = _cloud(pts) if voxel is None else _cloud(pts).voxel_down_sample(voxel)
            radius = (voxel or v) * (2.5 if voxel else 4.0)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=20))
            P = np.asarray(pcd.points)
            model[level] = {"cloud": pcd, "points": P, "normals": np.asarray(pcd.normals), "tree": cKDTree(P)}
        self._model = model
        self._model_n, self._model_frames = n, 0
        self._reloc_features = None
        return self._model

    @staticmethod
    def _icp_level(src: np.ndarray, level: dict, T: np.ndarray, thr: float, iterations: int):
        """Robust (Tukey) point-to-plane ICP against a model level with a persistent KD-tree."""
        T = np.asarray(T, dtype=float).copy()
        fitness, rmse = 0.0, float("inf")
        for it in range(iterations + 1):
            p = src @ T[:3, :3].T + T[:3, 3]
            dist, idx = level["tree"].query(p, k=1, distance_upper_bound=thr)
            ok = np.isfinite(dist)
            fitness = float(ok.mean()) if len(p) else 0.0
            if ok.sum() < 12:
                return T, 0.0, float("inf")
            rmse = float(np.sqrt(np.mean(dist[ok] ** 2)))
            if it == iterations:
                break
            ps, n = p[ok], level["normals"][idx[ok]]
            r = np.einsum("ij,ij->i", ps - level["points"][idx[ok]], n)
            w = np.clip(1.0 - (r / thr) ** 2, 0.0, None) ** 2
            J = np.hstack([np.cross(ps, n), n])
            A = (J * w[:, None]).T @ J + np.eye(6) * 1e-9
            x = np.linalg.solve(A, -(J * w[:, None]).T @ r)
            dT = np.eye(4)
            dT[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(x[:3])
            dT[:3, 3] = x[3:]
            T = dT @ T
            if np.linalg.norm(x[:3]) < 2e-5 and np.linalg.norm(x[3:]) < 1e-3:
                p = src @ T[:3, :3].T + T[:3, 3]
                dist, _ = level["tree"].query(p, k=1, distance_upper_bound=thr)
                ok = np.isfinite(dist)
                fitness = float(ok.mean())
                rmse = float(np.sqrt(np.mean(dist[ok] ** 2))) if ok.any() else float("inf")
                break
        return T, fitness, rmse

    def _icp(self, frame_cloud, model, init: np.ndarray):
        v = self.display_voxel
        src_c = np.asarray(frame_cloud.voxel_down_sample(max(2.0, v * 4)).points)
        src_f = np.asarray(frame_cloud.voxel_down_sample(max(1.2, v * 2.4)).points)
        T, _, _ = self._icp_level(src_c, model["coarse"], init, 6.0, 15)
        T, _, _ = self._icp_level(src_f, model["fine"], T, max(1.5, v * 3), 10)
        return self._icp_level(src_f, model["fine"], T, max(0.6, v * 1.2), 10)

    def _track(self, pts: np.ndarray, timestamp: float):
        with self.lock:
            empty = self.disp_n == 0
        if empty:
            return self._initial_pose(pts), "ok", None, None
        model = self._tracking_model()
        frame_cloud = _cloud(pts)
        min_fit = self.options["min_fitness"]
        if self._T_prev is None:
            candidates = []
        elif self._T_prev2 is not None:
            motion = np.linalg.inv(self._T_prev2) @ self._T_prev
            candidates = [self._T_prev @ motion, self._T_prev]
        else:
            candidates = [self._T_prev]
        best = (None, 0.0, float("inf"))
        for init in candidates:
            T, fit, rmse = self._icp(frame_cloud, model, init)
            if fit > best[1]:
                best = (T, fit, rmse)
            if fit >= 0.6:
                break
        T, fit, rmse = best
        if fit < min_fit:
            reloc = self._relocalize(frame_cloud, model, timestamp)
            if reloc is not None:
                T, fit, rmse = reloc
                self.log(f"Tracking recovered (fitness {fit:.2f})")
        if T is None or fit < min_fit:
            if T is None:
                T = self._T_prev if self._T_prev is not None else np.eye(4)
            return T, "lost", fit, rmse
        return T, ("ok" if fit >= 0.6 else "weak"), fit, rmse

    def _relocalize(self, frame_cloud, model, timestamp: float):
        if timestamp - self._last_reloc < 0.5:
            return None
        self._last_reloc = timestamp
        voxel = max(2.5, self.display_voxel * 5)
        radius_n, radius_f = voxel * 2.5, voxel * 5

        def features(pcd):
            down = pcd.voxel_down_sample(voxel)
            down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_n, max_nn=30))
            return down, reg.compute_fpfh_feature(down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_f,
                                                                                              max_nn=100))

        if self._reloc_features is None:
            self._reloc_features = features(model["fine"]["cloud"])
        t_down, t_feat = self._reloc_features
        s_down, s_feat = features(frame_cloud)
        if len(s_down.points) < 30 or len(t_down.points) < 30:
            return None
        dist = voxel * 1.5
        o3d.utility.random.seed(int(timestamp * 1000) % 100_000)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            res = reg.registration_ransac_based_on_feature_matching(
                s_down, t_down, s_feat, t_feat, True, dist, reg.TransformationEstimationPointToPoint(False), 3,
                [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(dist)],
                reg.RANSACConvergenceCriteria(200_000, 0.999))
        T, fit, rmse = self._icp(frame_cloud, model, np.asarray(res.transformation))
        return (T, fit, rmse) if fit >= 0.6 else None

    def _ratio(self, dkeys: np.ndarray) -> np.ndarray:
        """Density relative to the target (1.0 = reached) for density cells, block averaged."""
        uniq, inverse = np.unique(dkeys, return_inverse=True)
        ratio = smoothed_density(self.density.get, uniq, self.cell) / np.float32(self.target_density)
        return ratio[inverse.reshape(-1)]

    # ----------------------------------------------------------------- fusion
    def _grow(self, needed: int) -> None:
        cap = len(self.disp_xyz)
        if needed <= cap:
            return
        new_cap = max(needed, cap * 2, 65_536)

        def grow(a, shape_tail=()):
            b = np.zeros((new_cap, *shape_tail), a.dtype)
            b[:self.disp_n] = a[:self.disp_n]
            return b

        self.disp_xyz = grow(self.disp_xyz, (3,))
        self.disp_rgb = grow(self.disp_rgb, (3,))
        self.disp_density = grow(self.disp_density)
        self.disp_dkey = grow(self.disp_dkey)
        self.disp_bucket = grow(self.disp_bucket)

    def _fuse(self, world: np.ndarray, colors: np.ndarray | None, origin: np.ndarray | None) -> None:
        with self.lock:
            if self.point_distance is None:
                self.point_distance = self._configured_pd or max(estimate_spacing(world), 1e-3)
            if self.display_voxel is None:
                v = float(self.options["display_voxel"])
                if v <= 0:
                    diag = float(np.linalg.norm(np.percentile(world, 99, axis=0) - np.percentile(world, 1, axis=0)))
                    v = max(0.5, self.point_distance * 2, diag / 1500.0)
                self.display_voxel = v
            if self.field is None and self.options["fusion"] == "average":
                self.field = SurfaceField(max(FUSION_CELL_MM, self.point_distance))
            if self.field is not None:       # every point, also those the cell cap below turns away: the averaging
                self.field.add(world)
            cap = int(self.options["max_points_per_cell"])
            accepted = self.fine.add(voxel_keys(world, self.point_distance), cap=cap)[0]
            if not accepted.any():
                return
            w = world[accepted]
            rgb = colors[accepted] if colors is not None and len(colors) == len(world) else None
            if rgb is not None:
                self.has_colors = True
            self.chunks.append((w.astype(np.float32), rgb))
            self.n_points += len(w)
            lo, hi = w.min(axis=0), w.max(axis=0)
            self.bbox = (lo, hi) if self.bbox is None else (np.minimum(self.bbox[0], lo), np.maximum(self.bbox[1], hi))

            view = None
            if origin is not None:
                view = origin - w
                view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
            dkeys = voxel_keys(w, self.cell)
            _, du, _, _, dadded = self.density.add(dkeys, vectors=view)
            self._dirty_dkeys.append(du[dadded > 0])

            self._add_display(w, rgb, dkeys)
            if origin is not None:
                self._carve(w, origin)

    def _add_display(self, w, rgb, dkeys) -> None:
        acc = self.display.add(voxel_keys(w, self.display_voxel), cap=1)[0]
        k = int(acc.sum())
        if not k:
            return
        if self.disp_n + k > MAX_DISPLAY_POINTS:
            self._rebuild_display(self.display_voxel * 2)
            return
        self._grow(self.disp_n + k)
        s = slice(self.disp_n, self.disp_n + k)
        self.disp_xyz[s] = w[acc]
        self.disp_rgb[s] = rgb[acc] if rgb is not None else 200
        self.disp_dkey[s] = dkeys[acc]
        ratio = self._ratio(dkeys[acc])
        self.disp_density[s] = ratio
        self.disp_bucket[s] = np.searchsorted(DENSITY_BUCKETS, ratio)
        self.disp_n += k

    def _rebuild_display(self, voxel: float) -> None:
        self.log(f"Live view resolution reduced to {voxel:.2f} mm to keep the view responsive")
        self.display_voxel = voxel
        self.display = VoxelCounter()
        self.disp_n = 0
        self.density_updates.clear()
        self.epoch += 1
        for pts, rgb in self.chunks:
            self._add_display(pts.astype(np.float64), rgb, voxel_keys(pts, self.cell))
        self._model = None

    def _carve(self, w: np.ndarray, origin: np.ndarray) -> None:
        rng = np.random.default_rng(len(self.chunks))
        idx = rng.choice(len(w), min(len(w), 256), replace=False)
        vec = w[idx] - origin
        length = np.linalg.norm(vec, axis=1)
        usable = length - 3.0 * self.cell
        ok = usable > 0
        if not ok.any():
            return
        unit = vec[ok] / length[ok, None]
        fractions = np.linspace(0.0, 1.0, 16)
        # sample only the last 60 mm before the stop point: that is where the object is
        start = np.maximum(usable[ok] - 60.0, 0.0)
        dist = start[:, None] + (usable[ok] - start)[:, None] * fractions[None, :]
        samples = origin + unit[:, None, :] * dist[:, :, None]
        self.free.add(voxel_keys(samples.reshape(-1, 3), self.cell * 2))

    # ----------------------------------------------------------------- output
    def map_cloud(self, voxel: float | None = None, max_points: int | None = None) -> o3d.geometry.PointCloud:
        with self.lock:
            chunks = list(self.chunks)
        pts = np.concatenate([c[0] for c in chunks]) if chunks else np.empty((0, 3), np.float32)
        pcd = _cloud(pts)
        if voxel and max_points and len(pts) > max_points:
            pcd = pcd.voxel_down_sample(voxel)
        return pcd

    def build_cloud(self) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Full-resolution fused cloud (float64, colours when captured) and density per point (points/mm²)."""
        with self.lock:
            chunks = list(self.chunks)
            if not chunks:
                raise ValueError("Nothing has been captured yet")
            pts = np.concatenate([c[0] for c in chunks]).astype(np.float64)
            cells, inverse = np.unique(voxel_keys(pts, self.cell), return_inverse=True)
            density = smoothed_density(self.density.get, cells, self.cell)[inverse.reshape(-1)]
            has_colors = self.has_colors
            field = self.field.copy() if self.field is not None else None
        if field is not None:
            pts, stats = project(field, pts)
            self.fusion_stats = stats
            self.log(f"Averaged {stats['moved_pct']:.1f}% of the points onto the surface of {field.points:,} measured "
                     f"points (median move {stats['median_move_mm']:.3f} mm); the rest stay as measured")
        pcd = _cloud(pts)
        if has_colors:
            rgb = np.concatenate([c[1] if c[1] is not None else np.full((len(c[0]), 3), 128, np.uint8)
                                  for c in chunks])
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

        budget = int(self.options.get("max_points") or 0)
        if budget and len(pts) > budget:
            # Evenly spaced selection, never a voxel filter: voxel averaging invents point positions, while an even
            # pick keeps only real measurements and hits the budget exactly. Thinning alone would still pull the
            # bounding box in (the extreme points are the first casualties), so the per-axis extremes are always
            # retained - those are what the reported dimensions are measured from.
            keep = np.linspace(0, len(pts) - 1, budget).astype(np.int64)
            extremes = np.concatenate([pts.argmin(axis=0), pts.argmax(axis=0)])
            keep = np.unique(np.concatenate([keep, extremes]))[:budget]
            pcd = pcd.select_by_index(keep)
            density = density[keep]
            self.log(f"Point budget: kept {len(keep):,} of {len(pts):,} fused points")
        return pcd, density

    # ----------------------------------------------------------------- guidance
    def _flush_density_updates(self) -> None:
        with self.lock:
            if not self._dirty_dkeys:
                return
            dirty = np.unique(neighbor_keys(np.unique(np.concatenate(self._dirty_dkeys))).reshape(-1))
            self._dirty_dkeys = []
            n = self.disp_n
            if not n or not len(dirty):
                return
            idx = np.flatnonzero(np.isin(self.disp_dkey[:n], dirty))
            if not len(idx):
                return
            ratio = self._ratio(self.disp_dkey[idx])
            bucket = np.searchsorted(DENSITY_BUCKETS, ratio).astype(np.uint8)
            self.disp_density[idx] = ratio
            changed = idx[bucket != self.disp_bucket[idx]]
            self.disp_bucket[idx] = bucket
            if len(changed):
                self.density_version += 1
                self.density_updates.append((self.density_version, changed))

    def _heavy_analysis(self) -> dict:
        with self.lock:
            keys, counts, vecs = self.density.items()
            free_keys = self.free.items()[0]
        sensor = self._T_prev[:3, 3] if self._T_prev is not None else None
        return analyze(keys, counts, vecs, self.cell, self.target_density, free_keys, self.cell * 2, sensor,
                       bool(self.driver_settings.get("turntable", False)))

    def _msg(self, msgs: list, code: str, severity: str, message: str) -> None:
        msgs.append({"code": code, "severity": severity, "message": message})

    def update_guidance(self, full: bool = False) -> dict:
        with self._guidance_lock:   # the capture thread and request handlers (save / stop) both call this
            return self._update_guidance(full)

    def _update_guidance(self, full: bool) -> dict:
        now = time.monotonic()
        self._flush_density_updates()
        if full or now - self._last_heavy >= 1.0 and (self._heavy is None or self.stats["fused_frames"]):
            try:
                self._heavy = self._heavy_analysis()
            except Exception as exc:  # guidance must never kill the capture
                self.log(f"coverage analysis failed: {exc}")
            self._last_heavy = now
            if self._heavy is not None:
                self.coverage_timeline.append([round(self.elapsed, 2), self._heavy["completeness"]])
                if len(self.coverage_timeline) > 400:
                    self.coverage_timeline = self.coverage_timeline[::2]

        tick, self._tick = self._tick, self._new_tick()
        last = tick["last"] or self._last_light
        if tick["last"]:
            self._last_light = tick["last"]
        caps = self.capabilities
        msgs: list[dict] = []
        live = self.state not in ("stopped", "finished", "closed")
        if self.state == "error":
            self._msg(msgs, "error", "error", f"Capture failed: {self.error}")
        if live and tick["frames"]:
            if tick["lost"] and tick["lost"] >= tick["frames"] / 2:
                reason = tick.get("device_reason")
                seen = tick.get("device_markers")
                if reason == "no_markers" and seen:
                    # markers the operator can see in the camera are not necessarily usable: each must be found in
                    # BOTH cameras. "No markers in view" next to a picture full of markers read as a contradiction
                    text = (f"Only {seen} marker{'s' if seen != 1 else ''} found in both cameras - tracking needs 4. "
                            "Camera view rings the usable ones blue: look down onto the markers more steeply, or "
                            "come closer")
                elif reason == "no_markers":
                    text = ("No markers in view - the scanner tracks by markers in laser mode: keep at least 4 in "
                            "view (or switch the driver to geometry tracking)")
                elif reason == "lost":
                    text = "Markers not recognised - move back to where markers were already seen and hold steady"
                else:
                    text = "Tracking lost - move back to an area you already scanned and hold the scanner steady"
                self._msg(msgs, "tracking_lost", "error", text)
            if tick["fast"]:
                self._msg(msgs, "too_fast", "warning",
                          f"Moving too fast ({tick['max_speed']:.0f} mm/s, limit {self.options['speed_limit']:.0f}) "
                          "- slow down, these frames are not recorded")
            dstate = last.get("distance_state")
            rng = caps.get("range_mm")
            if dstate == "too_far":
                self._msg(msgs, "too_far", "warning", f"Too far ({last['distance_mm']:.0f} mm) - move closer, "
                          f"best around {caps.get('optimal_mm') or (rng[0] + rng[1]) / 2:.0f} mm")
            elif dstate == "too_close":
                best = caps.get('optimal_mm') or (rng[0] + rng[1]) / 2
                near = last.get("near_mm")
                if near is not None and last.get("distance_mm") is not None and last["distance_mm"] >= rng[0]:
                    self._msg(msgs, "too_close", "warning", f"Too close - the nearest part is {near:.0f} mm away and "
                              f"the scanner measures from {rng[0]:.0f} mm, so it is left out: move back, best around "
                              f"{best:.0f} mm")
                else:
                    self._msg(msgs, "too_close", "warning", f"Too close ({last['distance_mm']:.0f} mm) - move back, "
                              f"best around {best:.0f} mm")
            if tick["empty"] >= tick["frames"] and dstate not in ("too_far", "too_close"):
                self._msg(msgs, "no_data", "warning", "Nothing measured - point the scanner at the part")
            expected = max(float(np.median(self._recent_counts)) if len(self._recent_counts) >= 5 else 0.0,
                           float(caps.get("expected_points") or 0.0))
            if expected and dstate == "ok" and 50 <= last.get("points", 0) < 0.4 * expected:
                surface = self.driver_settings.get("surface")
                if caps.get("camera"):
                    # the camera view shows the laser lines and has the surface choice and Auto / Manual
                    if surface == "reflective":
                        hint = "shiny surface: open Scan → Camera view and lower the laser, or use scanning spray"
                    elif surface == "dark":
                        hint = "dark surface: open Scan → Camera view and keep it on Auto, or move a little closer"
                    else:
                        hint = ("open Scan → Camera view: choose Dark for a black part, Shiny for polished metal, "
                                "and keep it on Auto")
                elif surface == "reflective":
                    hint = "shiny surface: lower the brightness or exposure, or use scanning spray"
                elif surface == "dark":
                    hint = "dark surface: raise the laser brightness or use a longer exposure"
                else:
                    hint = ("check exposure (try Auto) and set Object surface to Dark or Reflective "
                            "if the part is dark or shiny")
                self._msg(msgs, "few_points", "warning", f"Few points per frame ({last['points']:,}) - {hint}")
            if self.tracking["state"] == "weak" and not tick["lost"]:
                self._msg(msgs, "tracking_weak", "info",
                          "Tracking is weak - include more shape detail in view or slow down")

        heavy = self._heavy or {}
        holes = heavy.get("holes", [])
        completeness = heavy.get("completeness", 0.0)
        with self.lock:
            pending = list(self.pending.values())
        if pending:
            first = pending[0]
            more = f" (+{len(pending) - 1} more waiting)" if len(pending) > 1 else ""
            self._msg(msgs, "pending_scan", "warning",
                      f"'{first['name']}' was not added to the model{more}: {first['assessment']['headline']} "
                      "Choose fuse, discard or keep separate.")
        if holes:
            self._msg(msgs, "hole", "info", holes[0]["message"])
        if not msgs:
            if self.state in ("running", "connected") and self.n_points:
                self._msg(msgs, "ok", "ok", f"Scanning well - coverage {completeness:.0%}")
            elif self.state in ("stopped", "finished") and self.n_points:
                self._msg(msgs, "done", "ok", f"Capture complete - coverage {completeness:.0%}")
            elif self.state == "paused":
                self._msg(msgs, "paused", "info", "Paused")
            else:
                self._msg(msgs, "waiting", "info", "Waiting for data from the scanner")
        top = max(msgs, key=lambda m: SEVERITY_RANK[m["severity"]])
        t = round(self.elapsed, 2)
        for m in msgs:
            h = self.history.setdefault(m["code"], {"count": 0, "first_s": t, "last_s": t, "severity": m["severity"],
                                                    "example": m["message"]})
            h["count"] += 1
            h["last_s"] = t

        dist = last.get("distance_mm")
        rng = caps.get("range_mm")
        sensor = None
        if self._T_prev is not None and self.tracking["state"] not in ("world", "n/a"):
            sensor = {"position": self._T_prev[:3, 3].round(2).tolist(),
                      "direction": self._T_prev[:3, 2].round(4).tolist()}
        guidance = {
            "type": "guidance", "session_id": self.id, "time": t, "state": self.state,
            "status": {"severity": top["severity"], "code": top["code"], "message": top["message"]},
            "messages": msgs,
            "tracking": dict(self.tracking),
            "speed": {"value_mm_s": round(tick["max_speed"], 1) if tick["frames"] else last.get("speed_mm_s"),
                      "limit_mm_s": self.options["speed_limit"], "too_fast": bool(tick["fast"])},
            "distance": {"value_mm": None if dist is None else round(float(dist), 1),
                         "min_mm": rng[0] if rng else None, "max_mm": rng[1] if rng else None,
                         "optimal_mm": caps.get("optimal_mm"), "state": last.get("distance_state", "unknown")},
            "frame": {"points": last.get("points", 0), "fused": last.get("fused", False)},
            "density": {k: heavy.get(k) for k in ("cell_mm", "target_per_mm2", "median_ratio", "cells", "dense_cells")},
            "coverage": {k: heavy.get(k) for k in ("completeness", "covered_area_mm2", "observed_area_mm2",
                                                   "missing_area_mm2", "estimated_area_mm2", "method")},
            "holes": holes,
            "sensor": sensor,
        }
        guidance["density"]["dense_ratio"] = DENSE_RATIO
        with self.lock:
            self.guidance = guidance
            self.guidance_version += 1
        return guidance

    # ----------------------------------------------------------------- status / streaming
    @property
    def elapsed(self) -> float:
        extra = time.monotonic() - self._run_since if self._run_since is not None else 0.0
        return self.running_time + extra

    def status(self) -> dict:
        times = list(self._fps_times)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 2 and times[-1] > times[0] else 0.0
        if self.state != "running" or (times and time.monotonic() - times[-1] > 2.0):
            fps = 0.0
        return {"type": "status", "active": True, "session_id": self.id, "driver": self.driver.id,
                "driver_name": self.driver.name, "state": self.state, "error": self.error,
                "frames": self.stats["frames"], "fused_frames": self.stats["fused_frames"],
                "scans": self.stats["scans"], "dropped": dict(self.stats["dropped"]),
                "points": self.n_points, "display_points": self.disp_n, "epoch": self.epoch,
                "fps": round(fps, 1), "elapsed_s": round(self.elapsed, 1),
                "point_distance_mm": self.point_distance, "display_voxel_mm": self.display_voxel,
                "tracking": self.tracking["state"], "has_colors": self.has_colors,
                "saved_asset_id": self.saved_asset_id, "pending_scans": len(self.pending),
                "unsaved": bool(self.n_points and self.n_points != self.saved_at_points or self.pending),
                "settings": {**self.driver_settings, **self.options},
                "device": self._device_status()}

    def _device_status(self) -> dict:
        try:
            return self.driver.status() or {}
        except Exception:  # a driver's status must never break the session's
            return {}

    def report(self) -> dict:
        heavy = self._heavy or {}
        return {"driver": self.driver.id, "driver_name": self.driver.name, "frames": self.stats["frames"],
                "fused_frames": self.stats["fused_frames"], "scans": self.scans,
                "pending_scans": [p["name"] for p in self.pending.values()],
                "dropped_frames": dict(self.stats["dropped"]), "duration": round(self.elapsed, 2),
                "points": self.n_points, "point_distance_mm": self.point_distance,
                "coverage": {k: heavy.get(k) for k in ("completeness", "covered_area_mm2", "observed_area_mm2",
                                                       "missing_area_mm2", "estimated_area_mm2", "method")},
                "holes": heavy.get("holes", []),
                "guidance": {"history": self.history, "coverage_timeline": self.coverage_timeline[-200:]},
                "fusion": {"method": self.options["fusion"],
                           "cell_mm": self.field.cell if self.field is not None else None,
                           "measured_points": self.field.points if self.field is not None else self.n_points,
                           **(self.fusion_stats or {})},
                "settings": {**self.driver_settings, **self.options}}

    def stream(self, cursor: dict, max_points: int = 250_000, chunk: int = 100_000) -> list:
        """Messages a stream client has not seen yet. `cursor` is per-client state, updated in place.

        Returns a list of JSON-able dicts (``reset``) and binary display messages (see routes_capture)."""
        out: list = []
        with self.lock:
            if cursor.get("session") != self.id or cursor.get("epoch") != self.epoch:
                cursor.update(session=self.id, epoch=self.epoch, sent=0, dversion=self.density_version)
                out.append({"type": "reset", "session_id": self.id, "epoch": self.epoch,
                            "display_voxel_mm": self.display_voxel, "total_points": self.disp_n})
            sent = cursor["sent"]
            if cursor["dversion"] < self.density_version:
                pending = [(v, idx) for v, idx in self.density_updates if v > cursor["dversion"]]
                if not pending or pending[0][0] != cursor["dversion"] + 1:
                    idx = np.arange(sent)     # fell too far behind: refresh every point already sent
                else:
                    idx = np.unique(np.concatenate([i for _, i in pending]))
                    idx = idx[idx < sent]
                cursor["dversion"] = self.density_version
                for start in range(0, len(idx), chunk):
                    part = idx[start:start + chunk]
                    out.append(pack_points(self.disp_xyz[part], None, self.disp_density[part], part))
            end = min(self.disp_n, sent + max_points)
            for start in range(sent, end, chunk):
                s = slice(start, min(start + chunk, end))
                out.append(pack_points(self.disp_xyz[s], self.disp_rgb[s] if self.has_colors else None,
                                       self.disp_density[s]))
            cursor["sent"] = end
            if self.live_frame is not None and cursor.get("live") != self.live_version:
                cursor["live"] = self.live_version
                out.append(pack_points(self.live_frame, live=True))
                if self.live_view is not None:
                    out.append(self.live_view)
            sent_pending = cursor.setdefault("pending", set())
            for scan_id, item in self.pending.items():
                if scan_id not in sent_pending:
                    sent_pending.add(scan_id)
                    out.append({"type": "pending_scan", "session_id": self.id, "scan_id": scan_id,
                                "name": item["name"], "points": len(item["points"]),
                                "assessment": item["assessment"]})
            for scan_id, decision in self._resolved:
                if scan_id in sent_pending:
                    sent_pending.discard(scan_id)
                    out.append({"type": "pending_resolved", "session_id": self.id, "scan_id": scan_id,
                                "decision": decision})
        return out


MAGIC = 0x43435054        # 'CCPT'
STREAM_VERSION = 1
FLAG_COLORS, FLAG_DENSITY, FLAG_INDICES, FLAG_LIVE = 1, 2, 4, 8   # LIVE: the current frame, replaces the last


def pack_points(xyz, rgb=None, density=None, indices=None, live: bool = False) -> bytes:
    """Binary display message: 16-byte header then float32 xyz, [uint8 rgb], [float32 density], [uint32 index]."""
    n = len(xyz)
    flags = (FLAG_COLORS if rgb is not None else 0) | (FLAG_DENSITY if density is not None else 0) | \
        (FLAG_INDICES if indices is not None else 0) | (FLAG_LIVE if live else 0)
    parts = [struct.pack("<IIII", MAGIC, STREAM_VERSION, n, flags), np.ascontiguousarray(xyz, "<f4").tobytes()]
    if rgb is not None:
        parts.append(np.ascontiguousarray(rgb, np.uint8).tobytes())
    if density is not None:
        parts.append(np.ascontiguousarray(density, "<f4").tobytes())
    if indices is not None:
        parts.append(np.ascontiguousarray(indices, "<u4").tobytes())
    return b"".join(parts)
