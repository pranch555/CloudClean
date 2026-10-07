"""Live capture API (Feature 4). One capture session at a time, owned by `capture.manager.CaptureManager`.

HTTP
----
GET  /api/capture/drivers        -> {drivers: [{id, name, kind, description, available, reason, settings: [schema],
                                     capabilities: {range_mm, optimal_mm, streaming, provides_pose}}],
                                     session_settings: [schema], last_driver, last_settings}
POST /api/capture/connect        {driver, settings}  -> status (settings may mix driver and session_settings keys)
POST /api/capture/start|pause|resume|stop|discard    -> status
POST /api/capture/save           {name?, auto_process?} -> {asset, job|null, status}
                                 auto_process null = settings section autopilot.auto_on_capture (default true)
GET  /api/capture/status         -> status (+ "guidance": last guidance message, "logs": last 20 lines)
GET  /api/capture/pending        -> {session_id, pending: [{scan_id, name, points, file, created, assessment}]}
POST /api/capture/pending/{scan_id} {decision: fuse|discard|keep_separate}
                                 -> {scan_id, decision, asset|null (keep_separate: the scan saved as its own
                                     asset in its file coordinates), status}

Scan chunks (bridge / folder drivers: every export is a complete scan) are not fused blindly. The first one starts
the model; each later one is aligned and assessed against it. Session setting `fuse_mode`:
"ask" (default) keeps every later scan aside as a pending scan, "auto" fuses it only when it adds coverage and
aligns reliably (recommendation "merge") and keeps it aside otherwise, "always" fuses without asking.
`assessment` = {recommendation: merge|use_best|ask, headline, reasons: [sentences], method, fitness, rmse, overlap,
coverage_gain_pct, coverage_gain_mm2, layering_mm, separation_ratio, doubled_surface, ambiguous,
alternative_angle, already_aligned, noise_mm, confident, adds_coverage}. Frame-streaming drivers (simulated,
depth cameras) fuse frames as before. Bridge uploads are still also imported as their own assets.
POST /api/capture/bridge/upload  multipart `files` (+ optional form `source` = path on the scanning PC,
                                 `import_asset` = true|false, default true: also import the file as its own asset)
                                 -> {results: [{file, queued, message, import_job}], status}
GET  /api/capture/bridge/ping    -> {ok, server, extensions}  (used by `cloudclean bridge` on start-up)

Camera view (scanners with a live camera: the MetroY, the simulated scanner; others answer available=false)
------------------------------------------------------------------------------------------------------------
GET  /api/capture/camera         -> {available, reason?, state, driver,
                                     settings: {surface: general|dark|reflective, mode: auto|manual, laser_level,
                                                level_max, laser_pct, laser_pulse (0xb08), exposure_us, gain,
                                                marker_light},
                                     limits: {laser_level: [min, max], exposure_us, gain, marker_light: [min, max],
                                              presets: {surface: {exposure_us, gain, laser_level, level_max,
                                                                  marker_light}}},
                                     auto: {state: off|waiting|adjusting|steady|limit, message, verdict:
                                            none|dim|good|bright},
                                     readout: {stripe_brightness (0-255, 95th pct), saturated_pct, points_per_frame,
                                               depth_mm, lines_seen, target, band, saturation_max_pct},
                                     streaming, preview, starting, error, gain_map: scanner|defaults, applied}
POST /api/capture/camera         {surface?, mode?, laser_level?, exposure_us?, gain?, marker_light?} -> as GET, plus
                                 remembered: {the driver-setting values stored for the next connect}.
                                 Applied at once, also while scanning; a laser/exposure/gain/marker value switches
                                 to manual unless mode is given. Remembered with the driver settings (surface,
                                 camera_mode, laser_level, exposure_us, gain, fill_light).
POST /api/capture/camera/preview {on}  -> as GET. Streams the camera while connected and not scanning (laser on, auto
                                 exposure running, nothing tracked or fused). It ends by itself ~6 s after the last
                                 picture was asked for, and becomes the scan when scanning starts.
GET  /api/capture/camera/view?cams=both|left|right&overlay=1&width=960 -> image/jpeg (204 while there is none yet).
                                 Pictures are made only while someone asks (poll at 5-8 per second). Overlay: laser
                                 lines CloudClean found in green, washed-out pixels red, markers ringed blue.
The status message's `device.camera` carries {mode, surface, laser_pct, gain, verdict, auto_state, auto_message,
stripe_brightness, saturated_pct, points_per_frame, streaming, preview} live.

Settings schema entries: {key, label, type: number|boolean|select|text, default, min?, max?, step?,
options?: [{value, label}], unit?, help?}.

WebSocket /api/capture/stream
-----------------------------
Text messages (JSON):

* ``{"type": "status", "active": bool, "state": "idle|connected|running|paused|stopped|finished|error|closed",
  "session_id", "driver", "driver_name", "error", "frames", "fused_frames", "scans",
  "dropped": {"too_fast", "tracking_lost", "empty"}, "points" (full resolution), "display_points", "epoch",
  "fps", "elapsed_s", "point_distance_mm", "display_voxel_mm", "tracking", "has_colors", "saved_asset_id",
  "pending_scans" (count), "unsaved" (also true while scans are pending), "settings"}`` - about 5 per second. When no session exists only ``type, active=false,
  state="idle", session_id=null``.
* ``{"type": "guidance", "session_id", "time", "state",
  "status": {"severity": "ok|info|warning|error", "code", "message"},
  "messages": [{"code", "severity", "message"}],   # codes: tracking_lost too_fast too_far too_close no_data
                                                   # few_points tracking_weak hole ok done paused waiting error
                                                   # pending_scan
  "tracking": {"state": "ok|weak|lost|device|world|n/a|none", "fitness", "rmse_mm"},
  "speed": {"value_mm_s", "limit_mm_s", "too_fast"},
  "distance": {"value_mm", "min_mm", "max_mm", "optimal_mm", "state": "ok|too_close|too_far|unknown"},
  "frame": {"points", "fused"},
  "density": {"cell_mm", "target_per_mm2", "median_ratio", "cells", "dense_cells", "dense_ratio"},
  "coverage": {"completeness" (0..1), "covered_area_mm2", "observed_area_mm2", "missing_area_mm2",
               "estimated_area_mm2", "method"},
  "holes": [{"id", "kind": "missing|sparse|edge", "center": [x,y,z], "direction": [x,y,z] (unit vector to
             look from, world frame, Z up), "area_mm2", "region", "hint", "message"}],
  "sensor": {"position": [x,y,z], "direction": [x,y,z]} | null}`` - sent whenever it changes (<= 5 Hz).
* ``{"type": "reset", "session_id", "epoch", "display_voxel_mm", "total_points"}`` - drop all display points;
  sent on connect, when a new session starts and when the display map is rebuilt.
* ``{"type": "pending_scan", "session_id", "scan_id", "name", "points", "assessment": {...}}`` - a scan chunk was
  kept aside and waits for a decision (POST /api/capture/pending/{scan_id}); sent once per client (again after
  a resync). ``{"type": "pending_resolved", "session_id", "scan_id", "decision"}`` once it was decided.

Binary messages (display points, little endian, sections not padded):

    uint32 magic 0x43435054 ('CCPT') | uint32 version = 1 | uint32 count | uint32 flags
    float32 xyz * count
    uint8   rgb * count      if flags & 1
    float32 density * count  if flags & 2   (1.0 = target density reached; for a heatmap)
    uint32  index * count    if flags & 4

Without flag 4 the points are **new** and are appended in order (their index is their arrival position since
the last reset). With flag 4 the message **updates** existing points at the given indices (density changes;
xyz is repeated for convenience). Because rgb breaks 4-byte alignment, copy sections that follow it
(``buffer.slice``) before creating typed arrays. Clients may send ``{"type": "resync"}`` to get a reset and
the full map again.
"""
from __future__ import annotations

import asyncio
import time
import uuid
import weakref
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..accuracy import jsonable
from ..capture.manager import CaptureError, CaptureManager
from ..io import CAD_EXTS, SUPPORTED_EXTS

SCAN_EXTS = sorted(SUPPORTED_EXTS - CAD_EXTS)
_MANAGERS: "weakref.WeakValueDictionary[str, CaptureManager]" = weakref.WeakValueDictionary()


def manager_for(workspace_root) -> CaptureManager | None:
    """The capture manager serving a workspace (for tests and other features)."""
    return _MANAGERS.get(str(Path(workspace_root).resolve()))


class ConnectReq(BaseModel):
    driver: str
    settings: dict = {}


class PendingReq(BaseModel):
    decision: str


class SaveReq(BaseModel):
    name: str | None = None
    auto_process: bool | None = None
    project_id: str | None = None   # default: the most recently updated project


class CameraReq(BaseModel):
    surface: str | None = None
    mode: str | None = None
    laser_level: float | None = None
    exposure_us: float | None = None
    gain: float | None = None
    marker_light: float | None = None


class PreviewReq(BaseModel):
    on: bool = True


def create_router(workspace, jobs) -> APIRouter:
    manager = CaptureManager(workspace, jobs)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(manager.shutdown)

    router = APIRouter(lifespan=lifespan)
    _MANAGERS[str(Path(workspace.root).resolve())] = manager

    def call(fn, *args):
        # live numbers can be inf / NaN (no tracking yet, nothing in view): JSON has neither, so they go out as null
        try:
            return jsonable(fn(*args))
        except CaptureError as exc:
            raise HTTPException(exc.status, str(exc))

    @router.get("/api/capture/drivers")
    def drivers():
        return manager.drivers()

    @router.post("/api/capture/connect")
    def connect(req: ConnectReq):
        return call(manager.connect, req.driver, req.settings)

    @router.post("/api/capture/start")
    def start():
        return call(manager.start)

    @router.post("/api/capture/pause")
    def pause():
        return call(manager.pause)

    @router.post("/api/capture/resume")
    def resume():
        return call(manager.resume)

    @router.post("/api/capture/stop")
    def stop():
        return call(manager.stop)

    @router.post("/api/capture/discard")
    def discard():
        return call(manager.discard)

    def check_project(project_id: str | None) -> str | None:
        if project_id:
            try:
                workspace.get_project(project_id)
            except KeyError:
                raise HTTPException(400, f"Project {project_id} not found")
        return project_id

    @router.post("/api/capture/save")
    def save(req: SaveReq):
        try:
            return call(manager.save, req.name, req.auto_process, check_project(req.project_id))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.post("/api/capture/driver/{name}")
    def driver_command(name: str, args: dict | None = None):
        return call(manager.driver_command, name, args)

    @router.get("/api/capture/status")
    def status():
        return jsonable(manager.status())

    @router.get("/api/capture/camera")
    def camera():
        return jsonable(manager.camera())

    @router.post("/api/capture/camera")
    def set_camera(req: CameraReq):
        return call(manager.set_camera, req.model_dump(exclude_none=True))

    @router.post("/api/capture/camera/preview")
    def camera_preview(req: PreviewReq):
        return call(manager.camera_preview, req.on)

    @router.get("/api/capture/camera/view")
    def camera_view(cams: str = "both", overlay: bool = True, width: int = 960):
        if cams not in ("both", "left", "right"):
            raise HTTPException(400, "cams must be both, left or right")
        try:
            jpeg = manager.camera_view(cams, overlay, max(160, min(3200, width)))
        except CaptureError as exc:
            raise HTTPException(exc.status, str(exc))
        headers = {"Cache-Control": "no-store"}
        if jpeg is None:
            return Response(status_code=204, headers=headers)
        return Response(jpeg, media_type="image/jpeg", headers=headers)

    @router.get("/api/capture/pending")
    def pending():
        return manager.pending()

    @router.post("/api/capture/pending/{scan_id}")
    def decide_pending(scan_id: str, req: PendingReq):
        return call(manager.decide_pending, scan_id, req.decision)

    @router.get("/api/capture/bridge/ping")
    def bridge_ping():
        return {"ok": True, "server": "CloudClean", "extensions": SCAN_EXTS}

    @router.post("/api/capture/bridge/upload")
    async def bridge_upload(files: list[UploadFile] = File(...), source: str | None = Form(None),
                            import_asset: bool = Form(True), project_id: str | None = Form(None)):
        check_project(project_id)
        results = []
        for f in files:
            name = Path(f.filename or "scan").name
            ext = Path(name).suffix.lower()
            if ext not in SCAN_EXTS:
                raise HTTPException(400, f"Unsupported scan file type '{ext}' ({name})")
        for f in files:
            name = Path(f.filename or "scan").name
            ext = Path(name).suffix.lower()
            tmp = workspace.uploads_dir / f"bridge_{uuid.uuid4().hex}{ext}"
            with open(tmp, "wb") as out:
                while chunk := await f.read(4 << 20):
                    out.write(chunk)
            import_job = None
            if import_asset:
                copy = workspace.uploads_dir / f"{uuid.uuid4().hex}{ext}"
                await asyncio.to_thread(_copy, tmp, copy)
                import_job = jobs.submit("import", f"Import {name}", {"path": str(copy), "name": Path(name).stem,
                                                                      "temporary": True, "source": source or name,
                                                                      "project_id": project_id})
            try:
                res = await asyncio.to_thread(manager.bridge_upload, tmp, name, source)
            except CaptureError as exc:
                tmp.unlink(missing_ok=True)
                raise HTTPException(exc.status, str(exc))
            results.append({"file": name, "queued": res["queued"], "message": res["message"],
                            "import_job": import_job})
        return {"results": results, "status": manager.status()}

    @router.websocket("/api/capture/stream")
    async def stream(websocket: WebSocket):
        from starlette.concurrency import run_in_threadpool

        from .auth import COOKIE, KEY_HEADER, AuthError

        store = getattr(websocket.app.state, "auth", None)
        if store is not None:
            try:
                user, _ = await run_in_threadpool(store.identify, websocket.cookies.get(COOKIE),
                                                  websocket.headers.get(KEY_HEADER))
            except AuthError:   # the accounts file is damaged
                user = None
            if user is None:
                await websocket.close(code=4401)   # not signed in
                return
        await websocket.accept()
        cursor: dict = {}
        closed = asyncio.Event()

        async def reader():
            try:
                while True:
                    msg = await websocket.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    if "resync" in (msg.get("text") or ""):
                        cursor.clear()   # also re-sends pending_scan messages
            except Exception:
                pass
            finally:
                closed.set()

        task = asyncio.create_task(reader())
        last_status = 0.0
        seen_guidance = None
        try:
            while not closed.is_set() and not manager.closed:
                for msg in await asyncio.to_thread(manager.stream, cursor):
                    if isinstance(msg, bytes):
                        await websocket.send_bytes(msg)
                    else:
                        await websocket.send_json(jsonable(msg))
                now = time.monotonic()
                if now - last_status >= 0.2:
                    last_status = now
                    session = manager.session
                    if session is None:
                        await websocket.send_json(jsonable(manager.status()))
                    else:
                        await websocket.send_json(jsonable(session.status()))
                        key = (session.id, session.guidance_version)
                        if session.guidance is not None and key != seen_guidance:
                            seen_guidance = key
                            await websocket.send_json(jsonable(session.guidance))
                try:
                    await asyncio.wait_for(closed.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            task.cancel()
            if not closed.is_set():
                try:
                    await websocket.close()
                except Exception:
                    pass

    return router


def _copy(src: Path, dst: Path) -> None:
    import shutil

    shutil.copyfile(src, dst)
