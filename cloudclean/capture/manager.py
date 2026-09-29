"""The single active capture session behind the web routes: connect / start / pause / resume / stop /
discard / save, pending scans (fuse / discard / keep separate), bridge uploads and streaming."""
from __future__ import annotations

import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

from ..web.settings import Settings
from .drivers.base import available_drivers, create_driver
from .session import PENDING_DECISIONS, SESSION_SETTINGS, CaptureSession

LIVE_STATES = ("running", "paused")


class CaptureError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class CaptureManager:
    def __init__(self, workspace, jobs=None):
        self.ws = workspace
        self.jobs = jobs
        self.settings = Settings(workspace.root)
        self.lock = threading.RLock()
        self.session: CaptureSession | None = None
        self.closed = False
        self.bridge_dir = Path(workspace.uploads_dir) / "capture_bridge"

    # ----------------------------------------------------------------- info
    def drivers(self) -> dict:
        remembered = self.settings.get("capture", {})
        return {"drivers": available_drivers(), "session_settings": SESSION_SETTINGS,
                "last_driver": remembered.get("last_driver"), "last_settings": remembered.get("driver_settings", {})}

    def status(self) -> dict:
        s = self.session
        if s is None:
            return {"type": "status", "active": False, "state": "idle", "session_id": None}
        return {**s.status(), "guidance": s.guidance, "logs": list(s.logs)[-20:]}

    def _require(self) -> CaptureSession:
        if self.session is None:
            raise CaptureError("No capture session - connect a scanner first", 409)
        return self.session

    def driver_command(self, name: str, args: dict | None = None) -> dict:
        with self.lock:
            s = self._require()
            if name in ("map_markers", "clear_map") and s.n_points:
                raise CaptureError("Save or discard the current scan first: a new marker map starts a new coordinate "
                                   "frame, and the points already captured would not line up with it", 409)
            try:
                result = s.driver.command(name, args or {})
            except ValueError as exc:
                raise CaptureError(str(exc)) from None
            s.log(f"{name}: {result}")
        return {"result": result, "status": self.status()}

    @staticmethod
    def _unsaved(s: CaptureSession) -> bool:
        return bool(s.n_points) and s.n_points != s.saved_at_points or bool(s.pending)

    # ----------------------------------------------------------------- lifecycle
    def _new_session(self, driver_id: str, settings: dict | None) -> CaptureSession:
        try:
            driver = create_driver(driver_id)
        except KeyError as exc:
            raise CaptureError(str(exc.args[0]))
        try:
            session = CaptureSession(driver, settings)
            session.connect()
        except (ValueError, RuntimeError, OSError) as exc:
            raise CaptureError(str(exc))
        return session

    def connect(self, driver_id: str, settings: dict | None = None) -> dict:
        with self.lock:
            current = self.session
            if current is not None:
                if current.state in LIVE_STATES:
                    raise CaptureError("A capture is running - stop it first", 409)
                if self._unsaved(current):
                    raise CaptureError("The current capture has unsaved data - save or discard it first", 409)
            session = self._new_session(driver_id, settings)
            if current is not None:
                current.close()
            self.session = session
            saved = self.settings.get("capture", {}).get("driver_settings", {})
            self.settings.update("capture", {"last_driver": driver_id,
                                             "driver_settings": {**saved, driver_id: settings or {}}})
            return self.status()

    def start(self) -> dict:
        with self.lock:
            s = self._require()
            try:
                s.start()
            except RuntimeError as exc:
                raise CaptureError(str(exc), 409)
            return self.status()

    def pause(self) -> dict:
        with self.lock:
            self._require().pause()
            return self.status()

    def resume(self) -> dict:
        with self.lock:
            self._require().resume()
            return self.status()

    def stop(self) -> dict:
        with self.lock:
            self._require().stop()
            return self.status()

    def discard(self) -> dict:
        with self.lock:
            if self.session is not None:
                self.session.close()
                self.session = None
            return self.status()

    def save(self, name: str | None = None, auto_process: bool | None = None, project_id: str | None = None) -> dict:
        """Save the fused cloud as an asset (in `project_id`, default the most recently updated project)."""
        with self.lock:
            s = self._require()
        if not s.n_points:
            raise CaptureError("Nothing has been captured yet")
        s.update_guidance(full=True)
        cloud, density = s.build_cloud()
        name = (name or "").strip() or f"Capture {datetime.now():%Y-%m-%d %H:%M}"
        report = s.report()
        meta = self.ws.add_geometry(cloud, name, "capture", [],
                                    params={"driver": s.driver.id, "settings": {**s.driver_settings, **s.options}},
                                    report=report, project=project_id)
        self.ws.add_scalars(meta["id"], "density", density, "points/mm²",
                            "Local point density while capturing (3x3x3 block of guidance cells)")
        s.saved_asset_id = meta["id"]
        s.saved_at_points = len(cloud.points)
        s.log(f"Saved '{name}' ({len(cloud.points):,} points)")
        if auto_process is None:
            auto_process = bool(self.settings.get("autopilot", {"auto_on_capture": True}).get("auto_on_capture", True))
        job = None
        if auto_process and self.jobs is not None:
            job = self.jobs.submit("autopilot", f"Autopilot {name}", {"asset_ids": [meta["id"]], "name": name})
        return {"asset": self.ws.get(meta["id"]), "job": job, "status": self.status()}

    # ----------------------------------------------------------------- pending scans
    def pending(self) -> dict:
        s = self.session
        return {"session_id": s.id if s else None, "pending": s.pending_scans() if s else []}

    def decide_pending(self, scan_id: str, decision: str) -> dict:
        """fuse: add the kept-aside scan to the model; discard: drop it; keep_separate: save it as its own asset
        (in its original file coordinates, the assessed pose is stored in the report) and drop it."""
        if decision not in PENDING_DECISIONS:
            raise CaptureError(f"decision must be one of: {', '.join(PENDING_DECISIONS)}")
        with self.lock:
            s = self._require()
        try:
            item = s.get_pending(scan_id)
        except KeyError:
            raise CaptureError(f"No pending scan {scan_id} (already decided?)", 404)
        asset = None
        if decision == "keep_separate":
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(item["points"].astype(np.float64)))
            if item["colors"] is not None:
                cloud.colors = o3d.utility.Vector3dVector(item["colors"].astype(np.float64) / 255.0)
            meta = self.ws.add_geometry(cloud, item["name"], "capture", [],
                                        params={"driver": s.driver.id, "source": item.get("file"),
                                                "kept_separate": True},
                                        report={"assessment": item["assessment"],
                                                "transform_to_model": np.asarray(item["transform"]).round(8).tolist()})
            asset = self.ws.get(meta["id"])
        try:
            s.resolve_pending(scan_id, decision)
        except KeyError:
            raise CaptureError(f"No pending scan {scan_id} (already decided?)", 404)
        return {"scan_id": scan_id, "decision": decision, "asset": asset, "status": self.status()}

    # ----------------------------------------------------------------- bridge
    def bridge_upload(self, tmp_path: Path, filename: str, source: str | None = None) -> dict:
        """Queue an uploaded export for the revo_bridge session, creating/starting that session if needed."""
        with self.lock:
            s = self.session
            if s is not None and s.driver.id != "revo_bridge":
                if s.state in LIVE_STATES or self._unsaved(s):
                    Path(tmp_path).unlink(missing_ok=True)
                    return {"queued": False, "status": self.status(),
                            "message": f"'{s.driver.name}' capture is active - the export was not added to it. "
                                       "Stop and save or discard that capture to use the bridge."}
                s.close()
                s = self.session = None
            if s is not None and s.state in ("error", "closed"):
                s.close()
                s = self.session = None
            if s is None:
                remembered = self.settings.get("capture", {}).get("driver_settings", {}).get("revo_bridge", {})
                try:
                    s = self._new_session("revo_bridge", remembered)
                except CaptureError:
                    s = self._new_session("revo_bridge", {})
                self.session = s
                s.log("Bridge upload received - capture session created")
            self.bridge_dir.mkdir(parents=True, exist_ok=True)
            dst = self.bridge_dir / f"{uuid.uuid4().hex}{Path(filename).suffix.lower()}"
            shutil.move(str(tmp_path), dst)
            s.driver.push(dst, Path(filename).stem, source or filename)
            s.log(f"Received {filename}")
            if s.state in ("connected", "stopped", "finished"):
                s.start()
            return {"queued": True, "message": f"{filename} added to the capture", "status": self.status()}

    # ----------------------------------------------------------------- stream / shutdown
    def stream(self, cursor: dict) -> list:
        s = self.session
        if s is None:
            if cursor.get("session") is not None:
                cursor.clear()
                return [{"type": "reset", "session_id": None, "epoch": 0, "display_voxel_mm": None,
                         "total_points": 0}]
            return []
        return s.stream(cursor)

    def shutdown(self) -> None:
        with self.lock:
            self.closed = True
            if self.session is not None:
                self.session.close()
                self.session = None
