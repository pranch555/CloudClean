"""Autopilot API: settings, folder watcher status, run on existing assets, drop files -> mesh files, and the
user's merge decision for runs that stopped with status "needs_decision".

POST /api/autopilot/decide {job_id, decision: merge|best|separate|cancel, best_index?}
    -> {job: new autopilot job | null, entry: the original history entry (now status "decided")}
    The new job continues from the cleaned scans (skip_clean) with the assessed transforms (merge) or the chosen
    scan (best, default = the assessment's best_index); its history entry has `decision_for` = the old job id.
"""
from __future__ import annotations

import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..autopilot import DECISIONS, SCAN_EXTS, AutopilotSettings, AutopilotState, Watcher, suggest_name
from ..io import CAD_EXTS
from .settings import Settings


class RunReq(BaseModel):
    asset_ids: list[str]
    name: str | None = None
    project_id: str | None = None   # results inherit their inputs' project; this only files imported files


class DecideReq(BaseModel):
    job_id: str
    decision: str
    best_index: int | None = None


def _same_folder(a, b) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def create_router(workspace, jobs) -> APIRouter:
    ws = workspace
    store = Settings(ws.root)
    state = AutopilotState(ws.root / "autopilot_state.json")
    lock = threading.Lock()
    holder: dict = {"watcher": None, "error": None}

    def saved() -> AutopilotSettings:
        return AutopilotSettings.load(store.get("autopilot", AutopilotSettings().to_dict()))

    def usable(settings: AutopilotSettings) -> AutopilotSettings:
        try:
            return settings.validate()
        except ValueError as exc:
            raise HTTPException(400, f"Autopilot settings are invalid: {exc}")

    def project_or_default(project_id: str | None) -> str:
        if project_id:
            try:
                return ws.get_project(project_id)["id"]
            except KeyError:
                raise HTTPException(400, f"Project {project_id} not found")
        return ws.default_project_id()

    def submit(name: str, payload: dict, files: list[str], settings: AutopilotSettings) -> dict:
        job = jobs.submit("autopilot", f"Autopilot {name}", {**payload, "name": name, "settings": settings.to_dict()})
        state.add(name, files, job)
        return job

    def stop_watcher() -> None:
        with lock:
            watcher, holder["watcher"] = holder["watcher"], None
        if watcher is not None:
            watcher.stop()

    def apply(settings: AutopilotSettings) -> None:
        """(Re)start or stop the watcher to match the settings."""
        stop_watcher()
        holder["error"] = None
        if not settings.watch_enabled:
            return
        if not settings.watch_folder or not Path(settings.watch_folder).is_dir():
            holder["error"] = f"Watch folder not found: {settings.watch_folder or '(not set)'}"
            return

        def on_item(name: str, files: list[str]) -> dict:
            # the watcher records its own history entry
            return jobs.submit("autopilot", f"Autopilot {name}",
                               {"paths": files, "name": name, "settings": settings.to_dict()})

        watcher = Watcher(settings, state, on_item, get_job=jobs.get, exclude=[ws.root])
        with lock:
            holder["watcher"] = watcher
        watcher.start()

    def start_if_enabled() -> None:
        if holder["watcher"] is not None and holder["watcher"].watching:
            return
        try:
            apply(saved().validate())
        except ValueError as exc:
            holder["error"] = f"Autopilot settings are invalid: {exc}"

    @asynccontextmanager
    async def lifespan(app):
        start_if_enabled()
        try:
            yield
        finally:
            stop_watcher()

    router = APIRouter(lifespan=lifespan)
    router.stop_watcher = stop_watcher  # for callers that manage the app without a lifespan (tests, scripts)

    # ------------------------------------------------------------------ settings
    @router.get("/api/autopilot/settings")
    def get_settings():
        return saved().to_dict()

    @router.put("/api/autopilot/settings")
    def put_settings(values: dict = Body(...)):
        current = saved()
        merged = current.to_dict()
        for key, value in values.items():
            if key in ("mesh", "compare") and isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        try:
            settings = AutopilotSettings.from_dict(merged).validate()
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc))

        if settings.watch_enabled:
            if not settings.watch_folder:
                raise HTTPException(400, "Choose a folder to watch before enabling watching")
            if not Path(settings.watch_folder).is_dir():
                raise HTTPException(400, f"The watch folder does not exist: {settings.watch_folder}")
        if settings.output_folder:
            if settings.watch_folder and _same_folder(settings.output_folder, settings.watch_folder):
                raise HTTPException(400, "The output folder must be different from the watch folder "
                                         "(a sub-folder of it is fine)")
            try:
                Path(settings.output_folder).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise HTTPException(400, f"Cannot create the output folder {settings.output_folder}: {exc}")
        if settings.reference_id:
            try:
                ref = ws.get(settings.reference_id)
            except KeyError:
                raise HTTPException(400, f"The CAD reference {settings.reference_id} does not exist")
            if ref["kind"] == "image":
                raise HTTPException(400, f"'{ref['name']}' is a photo, not a CAD reference")

        store.update("autopilot", settings.to_dict())
        apply(settings)
        return settings.to_dict()

    # ------------------------------------------------------------------ status
    @router.get("/api/autopilot/status")
    def status():
        settings = saved()
        watcher = holder["watcher"]
        if watcher is not None:
            info = watcher.status()
            info["error"] = info["error"] or holder["error"]
        else:
            state.refresh(jobs.get)
            info = {"watching": False, "folder": settings.watch_folder, "error": holder["error"],
                    "pending_items": [], "history": state.history()}
        return {"enabled": settings.watch_enabled, **info}

    # ------------------------------------------------------------------ run
    @router.post("/api/autopilot/run")
    def run(req: RunReq):
        if not req.asset_ids:
            raise HTTPException(400, "Select at least one scan")
        metas = []
        for asset_id in req.asset_ids:
            try:
                metas.append(ws.get(asset_id))
            except KeyError:
                raise HTTPException(404, f"Asset {asset_id} not found")
        if all(m["kind"] == "image" for m in metas):
            raise HTTPException(400, "Select at least one scan - photos cannot be meshed")
        settings = usable(saved())
        name = (req.name or "").strip() or suggest_name([m["name"] for m in metas if m["kind"] != "image"])
        payload = {"asset_ids": req.asset_ids}
        if req.project_id:
            payload["project_id"] = project_or_default(req.project_id)
        return submit(name, payload, [m["name"] for m in metas], settings)

    @router.post("/api/autopilot/upload")
    async def upload(files: list[UploadFile] = File(...), name: str | None = Form(None),
                     project_id: str | None = Form(None)):
        project = project_or_default(project_id)
        originals = [Path(f.filename or "upload").name for f in files]
        for original in originals:
            ext = Path(original).suffix.lower()
            if ext in CAD_EXTS:
                raise HTTPException(400, f"{original} is a CAD file - import it as a reference, not as a scan")
            if ext not in SCAN_EXTS:
                raise HTTPException(400, f"Unsupported file type '{ext}' ({original})")
        settings = usable(saved())
        paths: list[str] = []
        try:
            for f, original in zip(files, originals):
                tmp = ws.uploads_dir / f"{uuid.uuid4().hex}{Path(original).suffix.lower()}"
                paths.append(str(tmp))
                with open(tmp, "wb") as out:
                    while chunk := await f.read(4 << 20):
                        out.write(chunk)
        except Exception:
            for p in paths:
                Path(p).unlink(missing_ok=True)
            raise
        item = (name or "").strip() or suggest_name(originals)
        return submit(item, {"paths": paths, "temporary": True, "sources": dict(zip(paths, originals)),
                             "project_id": project}, originals, settings)

    # ------------------------------------------------------------------ merge decision
    @router.post("/api/autopilot/decide")
    def decide(req: DecideReq):
        if req.decision not in DECISIONS:
            raise HTTPException(400, f"decision must be one of: {', '.join(DECISIONS)}")
        state.refresh(jobs.get)
        with state.lock:
            entry = state.find(req.job_id)
            if entry is None:
                raise HTTPException(404, f"No autopilot run with job id {req.job_id}")
            if entry.get("status") != "needs_decision":
                raise HTTPException(409, f"This autopilot run does not wait for a decision (status: "
                                         f"{entry.get('status')})")
            pending = entry.get("decision") or {}
            ids = list(pending.get("cleaned_ids") or [])
            if len(ids) < 2:
                raise HTTPException(400, "This run has no scans to decide about")
            missing = []
            for asset_id in ids:
                try:
                    ws.get(asset_id)
                except KeyError:
                    missing.append(asset_id)
            if missing and req.decision != "cancel":
                raise HTTPException(400, f"The cleaned scans of this run were deleted ({', '.join(missing)}) - "
                                         "run autopilot again")
            now = datetime.now().isoformat(timespec="seconds")
            if req.decision == "cancel":
                entry = state.update(req.job_id, status="decided", finished=now,
                                     decided={"decision": "cancel", "job_id": None, "at": now})
                return {"job": None, "entry": dict(entry)}
            best = None
            if req.decision == "best":
                best = req.best_index if req.best_index is not None else pending.get("best_index")
                if best is None or not 0 <= int(best) < len(ids):
                    raise HTTPException(400, f"best_index must be between 0 and {len(ids) - 1}")
                best = int(best)
            try:
                settings = AutopilotSettings.load(pending.get("settings") or saved().to_dict()).validate()
            except ValueError as exc:
                raise HTTPException(400, f"Autopilot settings are invalid: {exc}")
            payload = {"asset_ids": ids, "skip_clean": True, "merge_decision": req.decision,
                       "transforms": pending.get("transforms") if req.decision == "merge" else None,
                       "best_index": best, "decision_for": req.job_id, "name": entry["item"],
                       "settings": settings.to_dict()}
            job = jobs.submit("autopilot", f"Autopilot {entry['item']} ({req.decision})", payload)
            state.add(entry["item"], entry.get("files", []), job, decision_for=req.job_id)
            entry = state.update(req.job_id, status="decided", finished=now,
                                 decided={"decision": req.decision, "job_id": job["id"], "at": now,
                                          "best_index": best})
            return {"job": job, "entry": dict(entry)}

    start_if_enabled()
    return router
