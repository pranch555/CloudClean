"""Part understanding, measurement tools and regions (Contracts 2 and 3; details in docs/v3-backend.md).

GET  /api/assets/{id}/summary[?refresh=true]   -> Contract 3 summary (cached in the asset folder)
POST /api/assets/summaries {asset_ids?, refresh?} -> job (kind "summary": precompute summaries in the background)
POST /api/regions/resolve  {asset_id, region}  -> {resolved, count, total, bbox: {min, max} | null, description}
POST /api/measure/extent   {asset_id, direction, region?}
POST /api/measure/caliper  {asset_id, direction, region?}
POST /api/measure/diameter {asset_id, region?, axis_hint?}
POST /api/measure/sphere   {asset_id, region}
POST /api/measure/plane    {asset_id, region}
POST /api/measure/angle    {asset_id, region_a, region_b}
POST /api/measure/section  {asset_id, plane: {point, normal} | {direction, at}}

Measurements answer directly (no job): the last two loaded models are kept in memory and part frames come from
asset meta. Invalid input or an impossible fit -> HTTP 400 with a sentence; unknown asset -> 404.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import understand as U
from ..regions import check_region, describe_region, positions, region_mask, resolve_region


class RegionReq(BaseModel):
    asset_id: str
    region: dict


class DirectionReq(BaseModel):
    asset_id: str
    direction: Any
    region: dict | None = None


class FacesReq(BaseModel):
    asset_id: str
    direction: Any = "length"
    region: dict | None = None


class DiameterReq(BaseModel):
    asset_id: str
    region: dict | None = None
    axis_hint: Any = None


class AngleReq(BaseModel):
    asset_id: str
    region_a: dict
    region_b: dict


class SectionReq(BaseModel):
    asset_id: str
    plane: dict


def create_router(workspace, jobs) -> APIRouter:
    ws = workspace
    router = APIRouter(prefix="/api")

    def geometry_meta(asset_id: str) -> dict:
        try:
            m = ws.get(asset_id)
        except KeyError:
            raise HTTPException(404, "Asset not found")
        if m["kind"] == "image":
            raise HTTPException(400, f"'{m['name']}' is a photo, not a scan")
        return m

    def load(asset_id: str):
        meta = geometry_meta(asset_id)
        try:
            geom = U.cached_geometry(ws, asset_id)
        except KeyError:
            raise HTTPException(404, "Asset not found")
        return meta, geom

    def frame(asset_id: str, geom):
        return U.frame_for(ws, asset_id, geom)

    def run(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    def checked_region(region: dict | None, where: str = "region") -> dict | None:
        if region is None:
            return None
        return run(check_region, region, where)

    @router.get("/assets/{asset_id}/summary")
    def summary(asset_id: str, refresh: bool = False):
        geometry_meta(asset_id)
        return run(U.get_summary, ws, asset_id, refresh)

    @router.post("/assets/summaries")
    def summaries(values: dict | None = None):
        """Start a background `summary` job: {asset_ids?: [...], refresh?: bool} (default: every model whose
        summary is missing or stale)."""
        values = values or {}
        ids = values.get("asset_ids") or []
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise HTTPException(400, "asset_ids must be a list of asset ids")
        for asset_id in ids:
            geometry_meta(asset_id)
        return jobs.submit("summary", f"Summarise {len(ids) or 'all'} model(s)",
                           {"asset_ids": ids, "refresh": bool(values.get("refresh"))})

    @router.post("/regions/resolve")
    def resolve(req: RegionReq):
        _, geom = load(req.asset_id)
        region = checked_region(req.region)
        f = frame(req.asset_id, geom)
        resolved = run(resolve_region, geom, region, f)
        mask = run(region_mask, geom, region, f)
        pts = np.asarray(positions(geom))[mask]
        bbox = {"min": pts.min(axis=0).tolist(), "max": pts.max(axis=0).tolist()} if len(pts) else None
        return {"resolved": resolved, "count": int(mask.sum()), "total": int(len(mask)), "bbox": bbox,
                "description": describe_region(region)}

    @router.post("/measure/extent")
    def extent(req: DirectionReq):
        meta, geom = load(req.asset_id)
        region = checked_region(req.region)
        return {**run(U.measure_extent, geom, req.direction, region, frame(req.asset_id, geom)),
                "asset_id": req.asset_id}

    @router.post("/measure/caliper")
    def caliper(req: DirectionReq):
        meta, geom = load(req.asset_id)
        region = checked_region(req.region)
        return {**run(U.measure_caliper, geom, req.direction, region, frame(req.asset_id, geom),
                      (meta.get("stats") or {}).get("spacing")), "asset_id": req.asset_id}

    @router.post("/measure/faces")
    def faces(req: FacesReq):
        meta, geom = load(req.asset_id)
        region = checked_region(req.region)
        return {**run(U.measure_faces, geom, req.direction or "length", region, frame(req.asset_id, geom),
                      (meta.get("stats") or {}).get("spacing")), "asset_id": req.asset_id}

    @router.post("/measure/diameter")
    def diameter(req: DiameterReq):
        _, geom = load(req.asset_id)
        region = checked_region(req.region)
        return {**run(U.measure_diameter, geom, region, req.axis_hint, frame(req.asset_id, geom)),
                "asset_id": req.asset_id}

    @router.post("/measure/sphere")
    def sphere(req: RegionReq):
        _, geom = load(req.asset_id)
        region = checked_region(req.region)
        return {**run(U.measure_sphere, geom, region, frame(req.asset_id, geom)), "asset_id": req.asset_id}

    @router.post("/measure/plane")
    def plane(req: RegionReq):
        _, geom = load(req.asset_id)
        region = checked_region(req.region)
        return {**run(U.measure_plane, geom, region, frame(req.asset_id, geom)), "asset_id": req.asset_id}

    @router.post("/measure/angle")
    def angle(req: AngleReq):
        _, geom = load(req.asset_id)
        region_a = checked_region(req.region_a, "region_a")
        region_b = checked_region(req.region_b, "region_b")
        return {**run(U.measure_angle, geom, region_a, region_b, frame(req.asset_id, geom)),
                "asset_id": req.asset_id}

    @router.post("/measure/section")
    def section(req: SectionReq):
        meta, geom = load(req.asset_id)
        return {**run(U.measure_section, geom, req.plane, frame(req.asset_id, geom),
                      (meta.get("stats") or {}).get("spacing")), "asset_id": req.asset_id}

    return router
