"""API routes for measuring: thread analysis (pitch, diameters, standard match) and point-to-point distances."""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..analysis import axis_vector, nearest_standard_thread, point_distance, thread_analysis
from ..edit import Snapper

MAX_THREAD_POINTS = 2_000_000


class ThreadReq(BaseModel):
    asset_id: str
    region: dict | None = None


class DistanceReq(BaseModel):
    asset_id: str
    points: list[list[float]]
    axis: list[float] | str | None = None


def create_router(workspace, jobs) -> APIRouter:
    ws = workspace
    router = APIRouter()
    # Measuring is interactive (a request per click / selection): keep the last few loaded scans.
    geometries: OrderedDict[str, object] = OrderedDict()
    snappers: OrderedDict[str, Snapper] = OrderedDict()
    lock = threading.Lock()

    def geometry_meta(asset_id: str) -> dict:
        try:
            m = ws.get(asset_id)
        except KeyError:
            raise HTTPException(404, "Asset not found")
        if m["kind"] == "image":
            raise HTTPException(400, f"'{m['name']}' is a photo, not a scan")
        return m

    def cached(cache: OrderedDict, asset_id: str, make, keep: int):
        with lock:
            item = cache.get(asset_id)
            if item is None:
                item = make()
                cache[asset_id] = item
                while len(cache) > keep:
                    cache.popitem(last=False)
            cache.move_to_end(asset_id)
            return item

    @router.post("/api/measure/thread")
    def measure_thread(req: ThreadReq):
        m = geometry_meta(req.asset_id)
        geom = cached(geometries, req.asset_id, lambda: ws.load_geometry(req.asset_id), 1)
        logs: list[str] = []
        try:
            result = thread_analysis(geom, req.region, log=logs.append, max_points=MAX_THREAD_POINTS)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        try:
            result["standards"] = nearest_standard_thread(result["major_diameter"], result["pitch"], 3,
                                                          result["kind"])
        except ValueError:
            result["standards"] = []
        result["asset_id"] = req.asset_id
        result["asset_name"] = m["name"]
        result["log"] = logs
        return result

    @router.post("/api/measure/distance")
    def measure_distance(req: DistanceReq):
        m = geometry_meta(req.asset_id)
        if len(req.points) != 2 or any(len(p) != 3 for p in req.points) \
                or not np.isfinite(np.asarray(req.points, float)).all():
            raise HTTPException(400, "Give exactly two points, each three finite numbers [x, y, z]")
        if req.axis is not None:
            try:
                axis_vector(req.axis)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
        snapper = cached(snappers, req.asset_id, lambda: Snapper(ws.load_geometry(req.asset_id)), 2)
        if not snapper.cloud and len(snapper.triangles) == 0:
            raise HTTPException(400, f"'{m['name']}' has no surface to measure on")
        snapped, dist = snapper.snap(req.points)
        out = point_distance(snapped, req.axis)
        out["points"] = snapped.tolist()
        out["snap_distances"] = dist.tolist()
        return out

    return router
