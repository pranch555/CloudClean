"""Accuracy checks: scanner vs software (Contract 5, docs/v3-plan.md; usage in docs/accuracy.md).

* POST /api/accuracy/drift {asset_id, parent_id?, tolerance?}   how far the operation that made an asset moved /
  resized its parent's surface (clean must move nothing, a merge must be rigid, a mesh must follow the cloud)
* POST /api/accuracy/compare-scans {a_id, b_id, transform?, tolerance?, save_aligned?, name?}   job compare_scans:
  two scans of the same part -> relative scale (ppm), surface offset, separation (scanner repeatability)
* POST /api/accuracy/reference {asset_id, reference}   measure a known artefact (gauge block, ball bar, pin, thread)
* POST /api/accuracy/dimensions {asset_id, trim?}   robust part dimensions next to what the viewer's boxes show
* POST /api/measure/snap {asset_id, points}   picked points moved onto a local surface fit, with their uncertainty
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import accuracy as acc
from ..io import is_cloud

SCAN_KINDS = ("pointcloud", "mesh")
MAX_DRIFT_PARENTS = 4


class DriftReq(BaseModel):
    asset_id: str
    parent_id: str | None = None
    tolerance: float = 0.01


class CompareScansReq(BaseModel):
    a_id: str
    b_id: str
    transform: list[list[float]] | None = None
    tolerance: float = 0.02
    save_aligned: bool = True
    name: str | None = None


class ReferenceReq(BaseModel):
    asset_id: str
    reference: dict


class DimensionsReq(BaseModel):
    asset_id: str
    trim: float = acc.DEFAULT_TRIM


class SnapReq(BaseModel):
    asset_id: str
    points: list[list[float]]


SNAP_NEIGHBOURS = 24
MAX_SNAP_POINTS = 200


def parent_transforms(meta: dict, report: dict | None) -> dict[str, np.ndarray | None]:
    """Pose that maps each parent into the asset's coordinates, from the operation's report (None = unknown ->
    compared as it is, identity)."""
    parents = list(meta.get("parents") or [])
    report = report or {}
    op = meta.get("operation")
    out: dict[str, np.ndarray | None] = {p: None for p in parents}
    if op == "merge":
        for scan in report.get("scans") or []:
            i = int(scan.get("index", -1))
            if 0 <= i < len(parents):
                T = scan.get("transform")
                out[parents[i]] = np.eye(4) if T is None else np.asarray(T, float)
    elif op in ("edit", "compare") and parents and report.get("transform") is not None:
        out[parents[0]] = np.asarray(report["transform"], float)
    elif op == "compare_scans" and len(parents) >= 2 and report.get("transform") is not None:
        out[parents[1]] = np.asarray(report["transform"], float)
    return out


def create_router(workspace, jobs) -> APIRouter:
    ws = workspace
    router = APIRouter()
    lock = threading.Lock()
    geometries: OrderedDict[str, object] = OrderedDict()
    results: OrderedDict[tuple, dict] = OrderedDict()
    surfaces: OrderedDict[str, acc.SurfaceModel] = OrderedDict()

    def asset_meta(asset_id: str) -> dict:
        try:
            m = ws.get(asset_id)
        except KeyError:
            raise HTTPException(404, "Asset not found")
        return m

    def scan_meta(asset_id: str) -> dict:
        m = asset_meta(asset_id)
        if m["kind"] not in SCAN_KINDS:
            raise HTTPException(400, f"'{m['name']}' is a {m['kind']}, not a scan")
        return m

    def stamp(asset_id: str) -> float:
        try:
            return (ws.asset_dir(asset_id) / "data.ply").stat().st_mtime
        except OSError:
            return 0.0

    def geometry(asset_id: str):
        key = f"{asset_id}:{stamp(asset_id)}"
        with lock:
            g = geometries.get(key)
            if g is not None:
                geometries.move_to_end(key)
                return g
        g = ws.load_geometry(asset_id)
        with lock:
            geometries[key] = g
            while len(geometries) > 3:
                geometries.popitem(last=False)
        return g

    @router.post("/api/accuracy/drift")
    def drift(req: DriftReq):
        meta = scan_meta(req.asset_id)
        if not 0 < req.tolerance < 10:
            raise HTTPException(400, "tolerance must be a positive distance in scan units (e.g. 0.01)")
        scans = [p for p in (meta.get("parents") or []) if _is_scan(ws, p)]
        if not scans:
            raise HTTPException(400, f"'{meta['name']}' was not made from another scan (operation "
                                     f"'{meta.get('operation')}') - there is nothing to compare it with")
        if req.parent_id is not None:
            if req.parent_id not in scans:
                raise HTTPException(400, "parent_id is not a scan this asset was made from")
            scans = [req.parent_id]
        transforms = parent_transforms(meta, ws.report(req.asset_id))
        entries = []
        child = geometry(req.asset_id)
        for pid in scans[:MAX_DRIFT_PARENTS]:
            key = (req.asset_id, stamp(req.asset_id), pid, stamp(pid), round(req.tolerance, 9))
            with lock:
                cached = results.get(key)
            if cached is None:
                pmeta = asset_meta(pid)
                try:
                    res = acc.operation_drift(geometry(pid), child, meta.get("operation"), transforms.get(pid),
                                              max_points=100_000, tolerance=req.tolerance)
                except ValueError as exc:
                    raise HTTPException(400, str(exc))
                res = acc.jsonable({k: v for k, v in res.items() if k != "frame"})
                res.update(parent_id=pid, parent_name=pmeta["name"])
                with lock:
                    results[key] = res
                    while len(results) > 32:
                        results.popitem(last=False)
                cached = res
            entries.append(cached)
        first = entries[0]
        out = {"asset_id": req.asset_id, "asset_name": meta["name"], **first}
        out["parents"] = entries
        if len(entries) > 1:
            order = {"changed": 2, "moved": 1, "unchanged": 0}
            out["verdict"] = max((e["verdict"] for e in entries), key=lambda v: order.get(v, 0))
        return out

    @router.post("/api/accuracy/compare-scans")
    def compare_scans(req: CompareScansReq):
        ma, mb = scan_meta(req.a_id), scan_meta(req.b_id)
        if req.a_id == req.b_id:
            raise HTTPException(400, "Pick two different scans of the same part")
        if req.transform is not None:
            try:
                check = acc.transform_check(req.transform)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            if not check["rigid"]:
                raise HTTPException(400, "transform must be rigid (rotation + translation, no scale)")
        if not 0 < req.tolerance < 10:
            raise HTTPException(400, "tolerance must be a positive distance in scan units (e.g. 0.02)")
        return jobs.submit("compare_scans", f"Compare {mb['name']} with {ma['name']}", req.model_dump())

    @router.post("/api/accuracy/reference")
    def reference(req: ReferenceReq):
        meta = scan_meta(req.asset_id)
        try:
            res = acc.reference_check(geometry(req.asset_id), req.reference)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        out = acc.jsonable(res)
        out.update(asset_id=req.asset_id, asset_name=meta["name"])
        return out

    @router.post("/api/accuracy/dimensions")
    def dimensions(req: DimensionsReq):
        meta = scan_meta(req.asset_id)
        if not 0 <= req.trim < 0.05:
            raise HTTPException(400, "trim must be between 0 and 0.05 (fraction of points ignored at each end)")
        g = geometry(req.asset_id)
        dims = acc.robust_dimensions(g, trim=req.trim)
        stats = meta.get("stats") or {}
        return acc.jsonable({
            "asset_id": req.asset_id, "asset_name": meta["name"], "kind": meta["kind"],
            "dimensions": {n: dims[n] for n in acc.AXIS_NAMES}, "raw": dims["raw"], "trim": req.trim,
            "axes": dims["frame"]["axes"], "center": dims["frame"]["center"],
            "axis_aligned_box": dims["aabb"]["size"], "fitted_box": stats.get("oriented_dimensions"),
            "points": int(len(acc.geometry_points(g))), "is_cloud": is_cloud(g),
            "note": "dimensions = extents along the part's principal axes with the trimmed fraction of points ignored "
                    "at each end; axis_aligned_box depends on how the part lay in the scanner and is not a part size",
        })

    @router.post("/api/measure/snap")
    def snap(req: SnapReq):
        """Refine picked points: each moves onto a quadric fitted to its 24 nearest scan points (clouds) or onto the
        closest triangle (meshes). A two-point distance from snapped points scatters ~2.5x less than one from the
        nearest raw points (0.018 vs 0.048 mm at 0.03 mm scan noise); its 1-sigma uncertainty is
        sqrt(u_a^2 + u_b^2) from uncertainty_mm."""
        meta = scan_meta(req.asset_id)
        pts = np.asarray(req.points, float) if req.points else np.zeros((0, 3))
        if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts) or not np.isfinite(pts).all():
            raise HTTPException(400, "points must be a list of [x, y, z] (three finite numbers each)")
        if len(pts) > MAX_SNAP_POINTS:
            raise HTTPException(400, f"at most {MAX_SNAP_POINTS} points per request")
        key = f"{req.asset_id}:{stamp(req.asset_id)}"
        with lock:
            model = surfaces.get(key)
            if model is not None:
                surfaces.move_to_end(key)
        if model is None:
            g = geometry(req.asset_id)
            if acc.geometry_points(g).shape[0] < SNAP_NEIGHBOURS:
                raise HTTPException(400, f"'{meta['name']}' has too few points to fit a surface")
            model = acc.SurfaceModel(g, k=SNAP_NEIGHBOURS)
            with lock:
                surfaces[key] = model
                while len(surfaces) > 2:
                    surfaces.popitem(last=False)
        res = model.project(pts)
        return acc.jsonable({
            "asset_id": req.asset_id, "points": res["points"], "uncertainty_mm": res["uncertainty"],
            "normals": res["normals"], "on_surface_fit": res["ok"], "moved_mm": res["moved"],
            "method": f"local quadric fit of the {SNAP_NEIGHBOURS} nearest scan points" if model.is_cloud
            else "closest point on the mesh",
        })

    return router


def _is_scan(ws, asset_id: str) -> bool:
    try:
        return ws.get(asset_id)["kind"] in SCAN_KINDS
    except KeyError:
        return False
