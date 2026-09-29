"""Holes of a mesh: list them (size, position) and fill the chosen ones. See cloudclean/holes.py.

GET  /api/assets/{id}/holes              -> {holes: [{id, vertices, perimeter, diameter, area, center, normal, outer}],
                                             count, holes_to_fill, boundary_edges, nonmanifold_edges, watertight}
POST /api/holes/fill {asset_id, hole_ids?, max_diameter?, name?} -> job (kind fill_holes_selected)
"""
from __future__ import annotations

import threading
from collections import OrderedDict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..holes import find_holes, public_holes


class FillReq(BaseModel):
    asset_id: str
    hole_ids: list[int] | None = None
    max_diameter: float | None = None
    name: str | None = None


def create_router(workspace, jobs) -> APIRouter:
    ws = workspace
    router = APIRouter()
    cache: OrderedDict[tuple, dict] = OrderedDict()
    lock = threading.Lock()

    def mesh_meta(asset_id: str) -> dict:
        try:
            meta = ws.get(asset_id)
        except KeyError:
            raise HTTPException(404, "Asset not found")
        if meta["kind"] != "mesh":
            raise HTTPException(400, f"'{meta['name']}' is not a mesh: build a mesh first to find its holes")
        return meta

    @router.get("/api/assets/{asset_id}/holes")
    def holes(asset_id: str):
        meta = mesh_meta(asset_id)
        path = ws.asset_dir(asset_id) / "data.ply"
        key = (asset_id, path.stat().st_mtime_ns if path.exists() else 0)
        with lock:
            hit = cache.get(key)
        if hit is None:
            try:
                hit = public_holes(find_holes(ws.load_geometry(asset_id)))
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            hit["asset_id"] = asset_id
            hit["asset_name"] = meta["name"]
            with lock:
                cache[key] = hit
                while len(cache) > 8:
                    cache.popitem(last=False)
        return hit

    @router.post("/api/holes/fill")
    def fill(req: FillReq):
        meta = mesh_meta(req.asset_id)
        if req.hole_ids is not None and not req.hole_ids:
            raise HTTPException(400, "Choose at least one hole to fill")
        if req.max_diameter is not None and req.max_diameter < 0:
            raise HTTPException(400, "max_diameter must be 0 (all) or a positive size in mm")
        payload = {"asset_id": req.asset_id, "hole_ids": req.hole_ids, "max_diameter": req.max_diameter,
                   "name": req.name}
        return jobs.submit("fill_holes_selected", f"Fill holes in {meta['name']}", payload)

    return router
