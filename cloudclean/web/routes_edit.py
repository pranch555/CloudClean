"""API routes for geometry editing, exporting to the server's disk and measuring."""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..edit import MESH, OP_CATALOGUE, PC, Snapper, measure, validate_ops
from .jobs_edit import export_asset


class EditReq(BaseModel):
    asset_id: str
    ops: list[dict]
    name: str | None = None


class ExportReq(BaseModel):
    asset_id: str
    format: str
    folder: str
    filename: str | None = None
    overwrite: bool = False


class MeasureReq(BaseModel):
    asset_id: str
    points: list[list[float]]


def create_router(workspace, jobs) -> APIRouter:
    ws = workspace
    router = APIRouter()
    # Loading a full-resolution scan and building its search structure takes seconds, and measuring is
    # interactive (one request per click), so keep the last few snappers.
    snappers: OrderedDict[str, Snapper] = OrderedDict()
    snap_lock = threading.Lock()

    def meta(asset_id: str) -> dict:
        try:
            return ws.get(asset_id)
        except KeyError:
            raise HTTPException(404, "Asset not found")

    def geometry_meta(asset_id: str) -> dict:
        m = meta(asset_id)
        if m["kind"] == "image":
            raise HTTPException(400, f"'{m['name']}' is a photo, not a scan")
        return m

    @router.get("/api/edit/ops")
    def edit_ops():
        return OP_CATALOGUE

    @router.post("/api/edit")
    def edit(req: EditReq):
        m = geometry_meta(req.asset_id)
        try:
            ops = validate_ops(req.ops, PC if m["kind"] == "pointcloud" else MESH)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        title = f"Edit {m['name']}: {', '.join(op['op'] for op in ops[:3])}" + (" …" if len(ops) > 3 else "")
        payload = {"asset_id": req.asset_id, "ops": ops, "name": (req.name or "").strip() or None}
        return jobs.submit("edit", title, payload)

    @router.post("/api/export")
    def export(req: ExportReq):
        geometry_meta(req.asset_id)
        try:
            path = export_asset(ws, req.asset_id, req.format, req.folder, req.filename, req.overwrite)
        except FileExistsError as exc:
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"path": str(path)}

    @router.post("/api/measure")
    def measure_points(req: MeasureReq):
        m = geometry_meta(req.asset_id)
        if not 1 <= len(req.points) <= 1000:
            raise HTTPException(400, "Give between 1 and 1000 points to measure")
        if any(len(p) != 3 for p in req.points) or not np.isfinite(np.asarray(req.points, float)).all():
            raise HTTPException(400, "Each point must be three finite numbers [x, y, z]")
        with snap_lock:
            snapper = snappers.get(req.asset_id)
            if snapper is None:
                snapper = Snapper(ws.load_geometry(req.asset_id))
                snappers[req.asset_id] = snapper
                while len(snappers) > 2:
                    snappers.popitem(last=False)
            snappers.move_to_end(req.asset_id)
        if not snapper.cloud and len(snapper.triangles) == 0:
            raise HTTPException(400, f"'{m['name']}' has no surface to measure on")
        return measure(snapper, req.points)

    return router
