"""Part understanding and measurements (Contract 3).

`summarize(geom)` describes a model in its own part frame (see cloudclean.regions): robust and full dimensions
along length / width / height, a 24-slice cross-section profile along the length, the flat faces and the
cylindrical sections it finds, and one or two plain sentences generated from those numbers. `get_summary(ws, id)`
caches the result as `summary.json` in the asset folder (recomputed when data.ply changes).

Measurement functions (`measure_extent`, `measure_caliper`, `measure_diameter`, `measure_sphere`, `measure_plane`,
`measure_angle`, `measure_section`) are used by the HTTP routes and by the assistant. They measure the data as it
is: extents use every point of the region, fits follow cloudclean.fitting (uniform subsample only to find a model,
statistics on every point), and every result reports `points_used`. All distances are in scan units (mm for
Revopoint scans); end points `a` / `b` are world coordinates so the viewer can draw a dimension line.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np

from . import fitting
from .io import Geometry, estimate_spacing, is_cloud
from .regions import (PART_AXES, PartFrame, direction_vector, part_frame, perpendicular_basis, positions,
                      region_mask, trimmed_range, vertex_weights)

SUMMARY_VERSION = 1
PROFILE_SLICES = 24
CYLINDER_SLICES = 40
DETECT_SAMPLE = 40_000          # points used to find planes (fits and statistics use every point)
SECTION_MAX_POINTS = 100_000     # polyline points returned by a section (evenly thinned beyond)


# --------------------------------------------------------------------------- small helpers
def _r(v, digits: int = 6):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [round(float(x), digits) for x in np.asarray(v, float).reshape(-1)]
    return round(float(v), digits)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def world_hint(v, limit_deg: float = 25.0) -> str:
    """'+z, top', '-x', ... when a direction is within limit_deg of a world axis, else ''."""
    v = _unit(v)
    k = int(np.argmax(np.abs(v)))
    if abs(v[k]) < math.cos(math.radians(limit_deg)):
        return ""
    sign = "+" if v[k] > 0 else "-"
    extra = {("+", 2): ", top", ("-", 2): ", bottom"}.get((sign, k), "")
    return f"{sign}{'xyz'[k]}{extra}"


def _fmt(x: float) -> str:
    """Numbers for sentences: 1 decimal from 10 mm, 2 below."""
    return f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"


def _selected(geom: Geometry, region, frame: PartFrame | None) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(positions(geom), dtype=np.float64)
    if region is None:
        mask = np.ones(len(pts), bool)
    else:
        mask = region_mask(geom, region, frame)
    if not mask.any():
        raise ValueError("The region contains no points of this model - check its position and size")
    return pts[mask], mask


def _geom_normals(geom: Geometry) -> np.ndarray | None:
    """Point / vertex normals without modifying the (possibly shared, cached) geometry."""
    if is_cloud(geom):
        return np.asarray(geom.normals) if geom.has_normals() else None
    if geom.has_vertex_normals():
        return np.asarray(geom.vertex_normals)
    if len(geom.triangles):
        from .edit import _vertex_normals

        return _vertex_normals(np.asarray(geom.vertices), np.asarray(geom.triangles))
    return None


# --------------------------------------------------------------------------- geometry cache
class GeometryCache:
    """The last few loaded models (measuring is interactive: one request per click)."""

    def __init__(self, keep: int = 2):
        self.keep = keep
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, ws, asset_id: str):
        path = ws.asset_dir(asset_id) / "data.ply"
        try:
            st = path.stat()
        except FileNotFoundError:
            raise KeyError(asset_id)
        key = (str(path), st.st_mtime_ns, st.st_size)
        with self._lock:
            geom = self._items.get(key)
            if geom is None:
                geom = ws.load_geometry(asset_id)
                self._items[key] = geom
                while len(self._items) > self.keep:
                    self._items.popitem(last=False)
            self._items.move_to_end(key)
            return geom

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


GEOMETRIES = GeometryCache()
_summary_locks: dict[str, threading.Lock] = {}
_summary_locks_guard = threading.Lock()


def cached_geometry(ws, asset_id: str):
    return GEOMETRIES.get(ws, asset_id)


def part_info(geom: Geometry) -> dict:
    """The part frame and dimensions stored in asset meta as `part` (cheap: ~0.3 s for 2 M points)."""
    return part_frame(geom).info()


def frame_for(ws, asset_id: str, geom: Geometry | None = None, store: bool = True) -> PartFrame:
    """The asset's part frame: from meta `part` when present, else computed (and stored in meta unless
    store=False - job workers must not rewrite the meta of existing assets)."""
    meta = ws.get(asset_id)
    info = meta.get("part")
    try:
        source = _source(ws, asset_id)
    except OSError:
        source = None
    if isinstance(info, dict) and "frame" in info and info.get("source") == source:
        try:
            return PartFrame.from_info(info)
        except (KeyError, TypeError, ValueError):
            pass
    frame = part_frame(geom if geom is not None else cached_geometry(ws, asset_id))
    if store:
        try:
            ws.update(asset_id, part={**frame.info(), "source": source})
        except (KeyError, OSError):
            pass
    return frame


# --------------------------------------------------------------------------- profile
def _mesh_slab_points(v: np.ndarray, tris: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Exact extreme points of a mesh surface between part coordinates z0..z1 along column 0 of v: vertices
    inside plus the intersections of the triangle edges with both bounding planes."""
    tz = v[tris, 0]
    tri = tris[(tz.max(axis=1) >= z0) & (tz.min(axis=1) <= z1)]
    if len(tri) == 0:
        return np.zeros((0, 3))
    used = v[np.unique(tri)]
    out = [used[(used[:, 0] >= z0) & (used[:, 0] <= z1)]]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = v[tri[:, i]], v[tri[:, j]]
        den = b[:, 0] - a[:, 0]
        for z in (z0, z1):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (z - a[:, 0]) / den
            ok = (den != 0) & (t >= 0) & (t <= 1)
            out.append(a[ok] + t[ok, None] * (b[ok] - a[ok]))
    return np.vstack(out)


def _profile(geom: Geometry, coords: np.ndarray, frame: PartFrame, slices: int = PROFILE_SLICES) -> dict:
    """Cross-section size (width, height: 0.5 % trimmed extents) of `slices` equal slices along the length.
    Coarse meshes (few vertices per slice) use the exact triangle / slab intersections instead of vertices."""
    length = float(frame.dims[0])
    edges = np.linspace(0.0, length, slices + 1)
    out = []
    if length <= 0:
        return {"axis": "length", "slices": out}
    inside = (coords[:, 0] >= 0) & (coords[:, 0] <= length)
    c = coords[inside]
    idx = np.minimum((c[:, 0] / length * slices).astype(int), slices - 1)
    counts = np.bincount(idx, minlength=slices)
    exact = (not is_cloud(geom)) and len(geom.triangles) and len(geom.triangles) <= 300_000 and counts.min() < 50
    order = np.argsort(idx, kind="stable")
    starts = np.r_[0, np.cumsum(counts)]
    tris = np.asarray(geom.triangles) if exact else None
    for s in range(slices):
        z0, z1 = float(edges[s]), float(edges[s + 1])
        if exact:
            q = _mesh_slab_points(coords, tris, z0, z1)
        else:
            q = c[order[starts[s]:starts[s + 1]]]
        item = {"from": _r(z0, 4), "to": _r(z1, 4), "width": None, "height": None, "count": int(counts[s])}
        if len(q):
            if len(q) >= 400 and not exact:
                lo, hi = np.percentile(q[:, 1:], [0.5, 99.5], axis=0)
            else:
                lo, hi = q[:, 1:].min(axis=0), q[:, 1:].max(axis=0)
            item["width"], item["height"] = _r(hi[0] - lo[0], 4), _r(hi[1] - lo[1], 4)
        out.append(item)
    return {"axis": "length", "slices": out}


# --------------------------------------------------------------------------- features: planes
def _occupied_area(uv: np.ndarray, cell: float) -> float:
    """Area covered by 2D points: occupied cells of a grid (closed with a 3 x 3 structuring element)."""
    from scipy import ndimage

    if len(uv) == 0 or cell <= 0:
        return 0.0
    ij = np.floor((uv - uv.min(axis=0)) / cell).astype(np.int64)
    shape = ij.max(axis=0) + 3
    if shape[0] * shape[1] > 25_000_000:  # absurdly fine grid: fall back to a coarser one
        return _occupied_area(uv, cell * math.sqrt(shape[0] * shape[1] / 25_000_000))
    grid = np.zeros(shape, bool)
    grid[ij[:, 0] + 1, ij[:, 1] + 1] = True
    grid = ndimage.binary_closing(grid, structure=np.ones((3, 3), bool))
    return float(grid.sum() * cell * cell)


def _plane_label(normal: np.ndarray, point: np.ndarray, frame: PartFrame) -> str:
    dots = frame.axes @ normal
    k = int(np.argmax(np.abs(dots)))
    hint = world_hint(normal, 20.0)
    hint = f" ({hint})" if hint else ""
    if abs(dots[k]) < math.cos(math.radians(20)):
        return "oblique face" + hint
    side = "max" if dots[k] > 0 else "min"
    noun = {0: "end", 1: "side", 2: "face"}[k]
    return f"{PART_AXES[k]}-{side} {noun}{hint}"


def _bending(uv: np.ndarray, res: np.ndarray) -> float:
    """Peak-to-valley of the quadratic part of a surface fitted to plane residuals (0 for a flat face)."""
    if len(uv) < 12:
        return 0.0
    scale = max(float(np.ptp(uv, axis=0).max()), 1e-12)
    u, v = (uv[:, 0] - uv[:, 0].mean()) / scale, (uv[:, 1] - uv[:, 1].mean()) / scale
    a = np.c_[u * u, u * v, v * v, u, v, np.ones_like(u)]
    coef, *_ = np.linalg.lstsq(a, res, rcond=None)
    quad = a[:, :3] @ coef[:3]
    return float(np.ptp(quad))


def detect_planes(geom: Geometry, pts: np.ndarray, frame: PartFrame, spacing: float, noise: float = 0.0,
                  max_planes: int = 6, min_fraction: float = 0.02, seed: int = 0) -> list[dict]:
    """Flat faces: iterative RANSAC on a subsample, then a least-squares refit on every point near the plane
    whose surface agrees with it (point normals for clouds, triangle normals for meshes - a plane through a curved
    or stepped part must not collect other surfaces). A candidate is kept when it covers >= min_fraction of the
    surface and does not bend (a strip of a curved surface is not a face)."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    n = len(pts)
    cloud = is_cloud(geom)
    if n < 100 and cloud:
        return []
    cos25, cos15, cos10 = (math.cos(math.radians(a)) for a in (25, 15, 10))
    tris = tri_area = tri_normal = None
    if not cloud:
        tris = np.asarray(geom.triangles)
        if len(tris) == 0:
            return []
        v = np.asarray(geom.vertices)
        cross = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
        tri_area = np.linalg.norm(cross, axis=1) / 2
        tri_normal = cross / np.maximum(2 * tri_area, 1e-300)[:, None]
    point_normals = _geom_normals(geom) if cloud else None
    if cloud:
        idx = fitting.sample_indices(n, DETECT_SAMPLE, seed)
        det = pts[idx]
        det_n = point_normals[idx] if point_normals is not None else None
    else:  # meshes: find faces on area-uniform samples of the triangles; fit on the vertices
        o3d.utility.random.seed(seed)
        samples = geom.sample_points_uniformly(number_of_points=min(DETECT_SAMPLE, max(20_000, n // 10)),
                                               use_triangle_normal=True)
        det, det_n = np.asarray(samples.points), np.asarray(samples.normals)
    oriented = det_n is not None
    if det_n is None:
        det_n = fitting._estimate_normals(det, 24)
    det_tree = cKDTree(det) if cloud and point_normals is None else None

    def agreeing(index: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Which of the points / vertices `index` lie on a surface parallel to the plane."""
        if cloud:
            if point_normals is not None:
                nrm = point_normals[index]
            else:
                nrm = det_n[det_tree.query(pts[index], k=1, workers=-1)[1]]
            return np.abs(nrm @ normal) >= cos25
        ok_vertex = np.zeros(n, bool)
        ok_vertex[tris[np.abs(tri_normal @ normal) >= cos25].ravel()] = True
        return ok_vertex[index]

    center = frame.to_world(frame.dims / 2)
    diag = float(np.linalg.norm(frame.dims_raw))
    # RANSAC band: 3 x the noise (a CAD mesh has none: a tiny fraction of its size)
    band0 = max(3.0 * noise, 1e-4 * diag)
    remaining = np.ones(len(det), bool)
    planes: list[dict] = []
    o3d.utility.random.seed(seed)
    for _ in range(max_planes + 6):
        if len(planes) >= max_planes or remaining.sum() < max(0.02 * len(det), 30):
            break
        sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(det[remaining]))
        model, inl = sub.segment_plane(distance_threshold=band0, ransac_n=3, num_iterations=500)
        normal = _unit(model[:3])
        point = -float(model[3]) / float(np.linalg.norm(model[:3])) * normal
        cand = np.flatnonzero(remaining)[np.asarray(inl, int)]
        band = band0
        near_idx = np.zeros(0, np.int64)
        # one pass over every point: the refit only tilts the plane slightly, so later bands stay in this subset
        pre = np.flatnonzero(np.abs((pts - point) @ normal) <= 3.0 * band0 + 2e-3 * diag)
        pre_pts = pts[pre]
        for _k in range(4):
            idx_band = pre[np.abs((pre_pts - point) @ normal) <= band]
            if len(idx_band) < 3:
                break
            near_idx = idx_band[agreeing(idx_band, normal)]
            if len(near_idx) < 3:
                break
            normal, point = fitting._svd_plane(pts[near_idx])
            res = (pts[near_idx] - point) @ normal
            band = min(max(3.0 * float(np.sqrt(np.mean(res ** 2))), band0), 2.0 * band0)
        agree = np.abs(det_n[cand] @ normal) >= cos15
        det_res = np.abs((det - point) @ normal)
        remaining &= ~((det_res <= max(band, band0)) & (np.abs(det_n @ normal) >= cos25))
        remaining[cand] = False
        if len(near_idx) < 3 or agree.mean() < 0.7:
            continue
        near = np.zeros(n, bool)
        near[near_idx] = True
        inliers = pts[near_idx]
        res = (inliers - point) @ normal
        u, v = fitting._basis(normal)
        uv = np.c_[(inliers - point) @ u, (inliers - point) @ v]
        sub_i = fitting.sample_indices(len(uv), 20_000, seed)
        rms = float(np.sqrt(np.mean(res ** 2)))
        if _bending(uv[sub_i], res[sub_i]) > max(0.5 * max(band, band0), 2 * rms):
            continue  # a strip of a curved surface
        if tris is not None:
            on = near[tris].all(axis=1) & (np.abs(tri_normal @ normal) >= cos10)
            area = float(tri_area[on].sum())
            fraction = area / max(float(tri_area.sum()), 1e-300)
            outward = float((tri_normal[on] * tri_area[on, None]).sum(axis=0) @ normal)
        else:
            area = _occupied_area(uv, 3.0 * spacing)
            fraction = float(len(near_idx) / n)
            outward = None
            if oriented:
                pick = near_idx[fitting.sample_indices(len(near_idx), 5000)]
                nrm = point_normals[pick] if point_normals is not None else None
                outward = float((nrm @ normal).sum()) if nrm is not None else None
        if fraction < min_fraction:
            continue
        # outward: the model's own normals when it has oriented ones, else away from the part centre
        flip = (outward < 0) if outward is not None else (normal @ (point - center) < 0)
        if flip:
            normal, res = -normal, -res
        centroid = inliers.mean(axis=0)
        planes.append({"point": _r(centroid), "normal": _r(normal, 9), "area_mm2": _r(area, 2),
                       "flatness": _r(float(res.max() - res.min()), 6), "rms": _r(rms, 6),
                       "points": int(len(near_idx)), "label": _plane_label(normal, centroid, frame)})
    planes.sort(key=lambda p: -p["area_mm2"])
    return planes


# --------------------------------------------------------------------------- features: cylinders
def _slice_circle(q: np.ndarray, noise: float) -> dict | None:
    """Quick robust circle of one cross-section (algebraic start, geometric refit, 3-sigma trimming)."""
    if len(q) < 30:
        return None
    try:
        c, r = fitting._kasa(q)
        # cheap rejection: a cross-section that is far from round needs no geometric refinement
        if fitting.robust_sigma(np.linalg.norm(q - c, axis=1) - r) > 0.05 * r + 3 * noise:
            return None
        keep = np.ones(len(q), bool)
        for _ in range(3):
            c, r = fitting._geometric_circle(q[keep], c, r, iterations=6)
            res = np.linalg.norm(q - c, axis=1) - r
            band = max(3.0 * float(np.sqrt(np.mean(res[keep] ** 2))), 2 * noise, 1e-9)
            keep = np.abs(res) <= band
            if keep.sum() < 10:
                return None
    except np.linalg.LinAlgError:
        return None
    rel = q[keep] - c
    ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360
    coverage = np.unique(np.minimum((ang / 10).astype(int), 35)).size * 10.0
    return {"c": c, "r": float(r), "rms": float(np.sqrt(np.mean(res[keep] ** 2))), "fraction": float(keep.mean()),
            "coverage": coverage}


def _grow_cylinder(cyl: dict, det: np.ndarray, bin_len: float, noise: float) -> None:
    """Extend a cylinder found on a run of slices over the contiguous surface it describes (the slices next to a
    shoulder or an end cap are not round, but the cylinder surface continues up to it): axial extent of the points
    on the fitted surface, walking out from the run while bins keep holding points (gaps of 2 bins allowed)."""
    a, c, r = np.asarray(cyl["axis"]), np.asarray(cyl["point"]), cyl["radius"]
    rel = det - c
    t = rel @ a
    radial = np.linalg.norm(rel - np.outer(t, a), axis=1)
    tt = np.sort(t[np.abs(radial - r) <= max(3.0 * cyl["rms"], 2.0 * noise, 1e-6 * r)])
    if len(tt) < 2:
        return
    nb = int(math.ceil((tt[-1] - tt[0]) / bin_len)) + 1
    counts = np.bincount(np.minimum(((tt - tt[0]) / bin_len).astype(int), nb - 1), minlength=nb)
    mid = int(np.clip((0.0 - tt[0]) / bin_len, 0, nb - 1))
    lo = hi = mid
    gap = 0
    while lo > 0 and gap <= 2:
        lo -= 1
        gap = 0 if counts[lo] else gap + 1
    lo += gap
    gap = 0
    while hi < nb - 1 and gap <= 2:
        hi += 1
        gap = 0 if counts[hi] else gap + 1
    hi -= gap
    inside = tt[(tt >= tt[0] + lo * bin_len) & (tt <= tt[0] + (hi + 1) * bin_len)]
    if len(inside) < 2:
        return
    t0, t1 = float(inside[0]), float(inside[-1])
    if t1 - t0 > cyl["length"]:
        cyl["length"] = t1 - t0
        cyl["point"] = _r(c + a * (t0 + t1) / 2)


def detect_cylinders(geom: Geometry, pts: np.ndarray, coords: np.ndarray, frame: PartFrame, spacing: float,
                     noise: float, max_cylinders: int = 4, seed: int = 0) -> list[dict]:
    """Cylindrical sections along the part axes (shanks, pins, turned diameters, round plates): runs of
    consecutive round cross-sections of equal size, then a least-squares cylinder on every point of the run."""
    n = len(pts)
    coarse = not is_cloud(geom) and n < 20_000 and len(geom.triangles) > 0
    if n < 200 and not coarse:
        return []
    if coarse:  # a CAD-like mesh has vertices only at a few rings: slice area-uniform samples of its surface
        import open3d as o3d

        o3d.utility.random.seed(seed)
        det = np.asarray(geom.sample_points_uniformly(number_of_points=100_000).points)
        cs = frame.coords(det)
    else:
        sample = fitting.sample_indices(n, 200_000, seed)
        det, cs = pts[sample], coords[sample]
    tol_floor = max(3.0 * noise, spacing, 1e-6)
    bin_len = max(float(frame.dims_raw.max()) / 400, 1e-9)
    found: list[dict] = []
    for k in (0, 2, 1):
        length = float(frame.dims[k])
        if length <= 0:
            continue
        i, j = [m for m in range(3) if m != k]
        span_max = 0.75 * max(frame.dims[i], frame.dims[j])
        inside = (cs[:, k] >= 0) & (cs[:, k] <= length)
        sc = cs[inside]
        sidx = np.minimum((sc[:, k] / length * CYLINDER_SLICES).astype(int), CYLINDER_SLICES - 1)
        order = np.argsort(sidx, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(sidx, minlength=CYLINDER_SLICES))]
        fits = []
        for s in range(CYLINDER_SLICES):
            q = sc[order[starts[s]:starts[s + 1]]][:, [i, j]]
            f = _slice_circle(q, noise)
            ok = (f is not None and f["fraction"] >= 0.9 and f["coverage"] >= 180 and f["r"] <= span_max
                  and f["rms"] <= max(0.01 * f["r"], 3.0 * noise, 0.5 * spacing))
            fits.append(f if ok else None)
        runs, current = [], []
        for s, f in enumerate(fits):
            if f is not None and current:
                prev = fits[current[-1]]
                tol = max(0.02 * f["r"], tol_floor)
                if abs(f["r"] - prev["r"]) <= tol and np.linalg.norm(f["c"] - prev["c"]) <= tol:
                    current.append(s)
                    continue
            if len(current) >= 3:
                runs.append(current)
            current = [s] if f is not None else []
        if len(current) >= 3:
            runs.append(current)
        for run in runs:
            z0, z1 = run[0] * length / CYLINDER_SLICES, (run[-1] + 1) * length / CYLINDER_SLICES
            sel = (coords[:, k] >= z0) & (coords[:, k] <= z1)
            if coarse:  # fit the mesh vertices on the run's cylinder (the samples only located it)
                span_pts = pts[sel]
                if len(span_pts) < 12:
                    continue
            else:
                if sel.sum() < 50:
                    continue
                span_pts = pts[sel]
            try:
                cyl = fitting.fit_cylinder(span_pts, axis_hint=frame.axes[k], seed=seed)
            except ValueError:
                continue
            r = cyl["radius"]
            axis = np.asarray(cyl["axis"])
            if abs(axis @ frame.axes[k]) < math.cos(math.radians(10)):
                continue
            good = (cyl["rms"] <= max(0.01 * r, 3.0 * noise, 0.5 * spacing) and cyl["coverage_deg"] >= 180
                    and cyl["inlier_fraction"] >= 0.8)
            long_enough = cyl["length"] >= 0.5 * cyl["diameter"] or (cyl["rms"] <= 0.005 * r
                                                                      and cyl["coverage_deg"] >= 270)
            if not (good and long_enough):
                continue
            _grow_cylinder(cyl, det, bin_len, noise)
            ends = frame.coords(np.asarray(cyl["point"]) + np.outer([-0.5, 0.5], cyl["length"] * axis))[:, k]
            cyl.update(axis_name=PART_AXES[k], span={"from": _r(ends.min(), 4), "to": _r(ends.max(), 4)})
            found.append(cyl)
    found.sort(key=lambda c: -c["length"])
    out = []
    for c in found:  # the same surface seen from two axes is one cylinder
        dup = any(abs(c["radius"] - o["radius"]) <= max(0.02 * o["radius"], tol_floor)
                  and np.linalg.norm(np.cross(np.asarray(c["point"]) - np.asarray(o["point"]), o["axis"]))
                  <= max(0.05 * o["radius"], tol_floor) for o in out)
        if not dup:
            out.append(c)
    keep = ("point", "axis", "radius", "diameter", "length", "rms", "coverage_deg", "points_used", "inliers",
            "axis_name", "span")
    return [{k: (_r(c[k]) if isinstance(c[k], float) else c[k]) for k in keep} for c in out[:max_cylinders]]


# --------------------------------------------------------------------------- description
def _end_text(profile: dict, frame: PartFrame, elongated: bool) -> str:
    slices = profile["slices"]
    sizes = [max(s["width"] or 0, s["height"] or 0) if s["count"] or s["width"] else None for s in slices]
    valid = [x for x in sizes if x]
    if len(valid) < 6:
        return ""
    k = max(2, len(slices) // 6)
    start = [x for x in sizes[:k] if x]
    end = [x for x in sizes[-k:] if x]
    if not start or not end:
        return ""
    s_med, e_med = float(np.median(start)), float(np.median(end))
    if max(s_med, e_med) < 1.25 * min(s_med, e_med):
        return ""
    side = "max" if e_med > s_med else "min"
    ordered = sizes[::-1] if side == "max" else sizes
    across = max(x for x in ordered[:k] if x)
    run = 0
    for x in ordered:
        if x is None or x < 0.8 * across:
            break
        run += 1
    head_len = run * frame.dims[0] / len(slices)
    hint = world_hint(frame.axes[0] if side == "max" else -frame.axes[0])
    where = f" ({hint})" if hint else ""
    head = ", likely a head" if elongated and head_len <= 0.4 * frame.dims[0] else ""
    return f"the length-{side} end{where} is wider ({across:.0f} mm across{head})"


def describe_shape(dims: dict, profile: dict, features: dict, frame: PartFrame) -> str:
    """One or two plain sentences generated from the numbers of a summary."""
    L, W, H = dims["length"], dims["width"], dims["height"]
    if W > 0 and L >= 2.0 * W:
        shape, elongated = "Elongated part", True
    elif W > 0 and H <= 0.25 * W:
        shape, elongated = f"Flat part (plate-like, {_fmt(H)} mm thick)", False
    else:
        shape, elongated = "Compact part", False
    parts = [f"{shape} {_fmt(L)} × {_fmt(W)} × {_fmt(H)} mm"]
    end = _end_text(profile, frame, elongated)
    if end:
        parts.append(end)
    cylinders = features.get("cylinders") or []
    if cylinders:
        c = cylinders[0]
        if c["axis_name"] == "length":
            parts.append(f"{c['length']:.0f} mm of the length is a Ø{_fmt(c['diameter'])} cylinder")
        else:
            parts.append(f"a Ø{_fmt(c['diameter'])} cylindrical section {_fmt(c['length'])} mm long along its "
                         f"{c['axis_name']}")
        if len(cylinders) > 1:
            parts[-1] += f" (plus {len(cylinders) - 1} more cylindrical section{'s' if len(cylinders) > 2 else ''})"
    text = "; ".join(parts) + "."
    planes = features.get("planes") or []
    if planes:
        p = planes[0]
        text += (f" {len(planes)} flat face{'s' if len(planes) > 1 else ''}; the largest is the {p['label']} "
                 f"(≈{p['area_mm2']:,.0f} mm²).")
    return text


# --------------------------------------------------------------------------- summary
def summarize(geom: Geometry, *, asset_id: str | None = None, spacing: float | None = None,
              frame: PartFrame | None = None, log=None) -> dict:
    """The Contract 3 summary of a geometry (see module docstring)."""
    started = time.perf_counter()
    pts = np.asarray(positions(geom), dtype=np.float64)
    n = len(pts)
    if n == 0:
        raise ValueError("The model has no points")
    cloud = is_cloud(geom)
    if not spacing or spacing <= 0:
        spacing = estimate_spacing(pts) or 1e-3
    frame = frame or part_frame(geom)
    coords = frame.coords(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    profile = _profile(geom, coords, frame)
    noise = fitting.local_noise(pts[fitting.sample_indices(n, 50_000, 3)]) if n > 32 else 0.0
    if not cloud:  # vertex neighbourhoods of a coarse mesh measure its curvature, not noise
        noise = min(noise, 0.5 * spacing, 5e-4 * float(np.linalg.norm(frame.dims_raw)))
    features = {"planes": [], "cylinders": []}
    # the two detectors are independent numpy / Open3D work (the GIL is released): run them side by side
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(2) as pool:
        jobs = {"cylinders": pool.submit(detect_cylinders, geom, pts, coords, frame, spacing, noise),
                "planes": pool.submit(detect_planes, geom, pts, frame, spacing, noise)}
        for key, future in jobs.items():
            try:
                features[key] = future.result()
            except Exception as exc:  # features are a bonus: never fail the summary
                if log:
                    log(f"{key} detection failed: {exc}")
    dims = frame.dims_dict()
    return {"asset_id": asset_id, "kind": "pointcloud" if cloud else "mesh", "units": "mm", "count": int(n),
            "spacing": _r(spacing), "noise": _r(noise),
            "aabb": {"min": _r(lo), "max": _r(hi), "size": _r(hi - lo)},
            "part_frame": frame.to_dict(), "dimensions": dims, "dimensions_raw": frame.dims_dict(raw=True),
            "profile": profile, "features": features,
            "description": describe_shape(dims, profile, features, frame),
            "computed_at": datetime.now().isoformat(timespec="seconds"), "version": SUMMARY_VERSION,
            "seconds": round(time.perf_counter() - started, 3)}


def _source(ws, asset_id: str) -> dict:
    st = (ws.asset_dir(asset_id) / "data.ply").stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def load_cached_summary(ws, asset_id: str) -> dict | None:
    """The cached summary if it is still valid (same data.ply, same version), else None. Never computes."""
    try:
        path = ws.asset_dir(asset_id) / "summary.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == SUMMARY_VERSION and data.get("source") == _source(ws, asset_id):
            return data
    except (OSError, ValueError, KeyError):
        pass
    return None


def _store_part_from_summary(ws, asset_id: str, meta: dict, summary: dict) -> None:
    """A summary computed by a job worker (which must not rewrite meta) leaves the asset without `part`: fill it in
    from the summary the first time the server reads it, so lists show the robust part size."""
    info = meta.get("part")
    if isinstance(info, dict) and info.get("source") == summary.get("source"):
        return
    try:
        ws.update(asset_id, part={"dimensions": summary["dimensions"],
                                  "dimensions_raw": summary.get("dimensions_raw") or summary["dimensions"],
                                  "frame": summary["part_frame"], "source": summary.get("source")})
    except (KeyError, OSError):
        pass


def get_summary(ws, asset_id: str, refresh: bool = False, store_part: bool = True) -> dict:
    """Cached summary of an asset (computed and stored in the asset folder when missing or stale)."""
    meta = ws.get(asset_id)
    if meta["kind"] == "image":
        raise ValueError(f"'{meta['name']}' is a photo, not a scan")
    with _summary_locks_guard:
        lock = _summary_locks.setdefault(f"{ws.root}/{asset_id}", threading.Lock())
    with lock:
        if not refresh:
            cached = load_cached_summary(ws, asset_id)
            if cached is not None:
                if store_part:
                    _store_part_from_summary(ws, asset_id, meta, cached)
                return cached
        geom = cached_geometry(ws, asset_id)
        frame = frame_for(ws, asset_id, geom, store=store_part)
        summary = summarize(geom, asset_id=asset_id, spacing=(meta.get("stats") or {}).get("spacing"), frame=frame)
        summary["source"] = _source(ws, asset_id)
        path = ws.asset_dir(asset_id) / "summary.json"
        tmp = path.with_name(f"summary.{os.getpid()}.{threading.get_ident()}.tmp")  # a job may write it too
        tmp.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return summary


# --------------------------------------------------------------------------- measurements
def _direction(direction, frame: PartFrame | None, geom) -> tuple[np.ndarray, str, bool]:
    d, label = direction_vector(direction, frame, geom)
    return d, label, isinstance(direction, str) and direction.lower() in PART_AXES


def _line_through(points: np.ndarray, d: np.ndarray, t0: float, t1: float) -> tuple[list, list]:
    c = points.mean(axis=0)
    base = c - (c @ d) * d
    return _r(base + t0 * d), _r(base + t1 * d)


def measure_extent(geom: Geometry, direction, region=None, frame: PartFrame | None = None) -> dict:
    """Full and robust (0.05 % trimmed; by surface area for meshes) extent along a direction.

    {length, robust_length, min, max, robust_min, robust_max, direction, direction_name, frame, a, b,
    extreme_points {min, max}, points_used}. min / max are part coordinates for part axes, else world."""
    d, label, part = _direction(direction, frame, geom)
    if part and frame is None:
        frame = part_frame(geom)
    pts, mask = _selected(geom, region, frame)
    t = pts @ d
    w = vertex_weights(geom)
    rlo, rhi = trimmed_range(t, None if w is None else w[mask])
    i0, i1 = int(np.argmin(t)), int(np.argmax(t))
    offset = float(frame.origin @ d) if part else 0.0
    a, b = _line_through(pts, d, float(t[i0]), float(t[i1]))
    return {"length": float(t[i1] - t[i0]), "robust_length": rhi - rlo, "min": float(t[i0]) - offset,
            "max": float(t[i1]) - offset, "robust_min": rlo - offset, "robust_max": rhi - offset,
            "direction": _r(d, 9), "direction_name": label, "frame": "part" if part else "world", "a": a, "b": b,
            "extreme_points": {"min": _r(pts[i0]), "max": _r(pts[i1])}, "points_used": int(len(pts))}


def _face(points: np.ndarray, d: np.ndarray, sign: float, band: float, noise: float, extreme: float) -> dict:
    """One caliper jaw: a plane fitted to the extreme band when it is a flat face across the direction,
    otherwise the contact point at the robust extreme."""
    face = {"kind": "contact", "point": None, "normal": _r(sign * d, 9), "rms": None, "points": int(len(points))}
    contact = points[int(np.argmin(np.abs(points @ d - extreme)))] if len(points) else None
    if len(points) >= 12:
        try:
            plane = fitting.fit_plane(points)
        except ValueError:
            plane = None
        if plane is not None:
            nrm = np.asarray(plane["normal"])
            if nrm @ d * sign < 0:
                nrm = -nrm
            angle = math.degrees(math.acos(min(1.0, abs(float(nrm @ d)))))
            if angle <= 10 and plane["inlier_fraction"] >= 0.5 and plane["rms"] <= max(1.5 * noise, 0.2 * band):
                return {"kind": "plane", "point": plane["point"], "normal": _r(nrm, 9), "rms": plane["rms"],
                        "flatness": plane["flatness"], "points": plane["inliers"], "angle_to_direction_deg": angle}
    if contact is not None:
        base = contact - (contact @ d - extreme) * d  # at the robust extreme level
        face["point"] = _r(base)
    return face


def measure_caliper(geom: Geometry, direction, region=None, frame: PartFrame | None = None,
                    spacing: float | None = None) -> dict:
    """Distance between the two opposing faces across a direction, like calipers: a plane is fitted to each
    extreme band (flat faces); a rounded end is touched at its robust extreme instead.

    {distance, parallelism_deg, face_a, face_b (kind plane|contact, point, normal, rms, points), a, b,
    extent (robust extent along the direction), direction, direction_name, points_used, warnings}."""
    d, label, part = _direction(direction, frame, geom)
    if part and frame is None:
        frame = part_frame(geom)
    pts, mask = _selected(geom, region, frame)
    if len(pts) < 20:
        raise ValueError(f"The region has only {len(pts)} points - too few for a caliper measurement")
    t = pts @ d
    w = vertex_weights(geom)
    lo, hi = trimmed_range(t, None if w is None else w[mask])
    extent = hi - lo
    if not spacing or spacing <= 0:
        spacing = estimate_spacing(pts[fitting.sample_indices(len(pts), 200_000)]) or 1e-3 * max(extent, 1e-9)
    noise = fitting.local_noise(pts[fitting.sample_indices(len(pts), 50_000, 5)]) if len(pts) > 32 else 0.0
    band = max(6.0 * noise, 2.0 * spacing, 1e-4 * max(extent, 1e-12))
    band = min(band, 0.25 * max(extent, 1e-12))
    face_a = _face(pts[t <= lo + band], d, -1.0, band, noise, lo)
    face_b = _face(pts[t >= hi - band], d, 1.0, band, noise, hi)
    na, nb = np.asarray(face_a["normal"]), np.asarray(face_b["normal"])
    pa, pb = np.asarray(face_a["point"]), np.asarray(face_b["point"])
    parallel = math.degrees(math.acos(min(1.0, abs(float(na @ nb)))))
    # measure along the mean jaw normal (oriented along d) at the middle of the region
    axis = _unit(nb - na) if np.linalg.norm(nb - na) > 1e-12 else d
    c = pts.mean(axis=0)

    def hit(p, nrm):  # where the line c + s * axis crosses the jaw plane
        den = float(nrm @ axis)
        return float((p - c) @ nrm / den) if abs(den) > 1e-9 else float((p - c) @ axis)

    sa, sb = hit(pa, na), hit(pb, nb)
    warnings = []
    if parallel > 1.0:
        warnings.append(f"The two faces are {parallel:.2f} deg from parallel: the distance is measured through the "
                        "middle of the region")
    if "plane" not in (face_a["kind"], face_b["kind"]):
        warnings.append("Neither end is a flat face across this direction: the distance is between the robust "
                        "extremes (like caliper jaws touching the highest points)")
    return {"distance": abs(sb - sa), "parallelism_deg": parallel, "face_a": face_a, "face_b": face_b,
            "a": _r(c + sa * axis), "b": _r(c + sb * axis), "extent": extent, "direction": _r(d, 9),
            "direction_name": label, "band": band, "noise": noise, "points_used": int(len(pts)),
            "warnings": warnings}


def measure_faces(geom: Geometry, direction="length", region=None, frame: PartFrame | None = None,
                  spacing: float | None = None, max_tilt_deg: float = 18.0, max_faces: int = 12) -> dict:
    """Every flat face across a direction and the distances between them: the ends of a part, the shoulder under a
    bolt head, the steps of a turned shaft. Faces are found from points whose normal lies within `max_tilt_deg` of
    the direction, grouped by position and fitted with robust planes; positions are taken where each plane crosses
    the part's centre line, so a slightly tilted face does not bias a step. Face-to-face, like calipers, instead of
    extents of a region whose cut would count as a face.

    {direction, direction_name, faces [{position (mm from the part's start along the direction), point, normal,
    tilt_deg, rms, flatness, sharpness (share of the nearby face-like points within the face's band: 1 = crisp),
    points, facing (+1 / -1 / None), radius_range [inner, outer] from the centre line, size_across [u, v]}], steps [{from, to, distance}], overall, a, b, points_used, warnings}."""
    d, label, part = _direction(direction, frame, geom)
    if part and frame is None:
        frame = part_frame(geom)
    pts, mask = _selected(geom, region, frame)
    normals = _geom_normals(geom)
    idx = fitting.sample_indices(len(pts), 400_000, 11)
    P = pts[idx]
    oriented = normals is not None
    if oriented:
        N = np.asarray(normals, float)[mask][idx]
    else:
        import open3d as o3d

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        N = np.asarray(pc.normals)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    if not spacing or spacing <= 0:
        spacing = estimate_spacing(P[fitting.sample_indices(len(P), 200_000)]) or 1e-3
    noise = fitting.local_noise(P[fitting.sample_indices(len(P), 50_000, 5)]) if len(P) > 32 else 0.0
    warnings: list[str] = []
    # reference: the part's start along the direction (part frame) or the robust start of the model
    if part:
        start = float(frame.origin @ d)
    else:
        w = vertex_weights(geom)
        start = trimmed_range(np.asarray(positions(geom)) @ d, w)[0]
    # centre line: through the middle of the selection's robust box across the direction
    u, v = perpendicular_basis(d, frame if part else None, label)
    pu, pv = P @ u, P @ v
    (ulo, uhi), (vlo, vhi) = trimmed_range(pu), trimmed_range(pv)
    uc, vc = (ulo + uhi) / 2, (vlo + vhi) / 2
    c0 = uc * u + vc * v

    cos = N @ d
    axial = np.abs(cos) >= math.cos(math.radians(max_tilt_deg))
    t = P[axial] @ d
    if len(t) < 30:
        raise ValueError(f"No flat faces across {label}: almost no surface faces that way")
    h = max(2.0 * noise, spacing / 4.0, 0.01)
    edges = np.arange(t.min() - 2 * h, t.max() + 3 * h, h)
    hist = np.histogram(t, edges)[0].astype(float)
    window = np.convolve(hist, np.ones(5), mode="same")
    need = max(60.0, 0.004 * len(t))
    candidates = [i for i in range(1, len(window) - 1)
                  if window[i] >= need and window[i] >= window[i - 1] and window[i] > window[i + 1]]
    candidates.sort(key=lambda i: -window[i])
    band = 3.0 * max(noise, h)
    faces: list[dict] = []
    taken: list[float] = []
    Pa, Na, ta = P[axial], N[axial], t
    for i in candidates:
        tc = (edges[i] + edges[i + 1]) / 2
        if any(abs(tc - x) < 2 * band for x in taken):
            continue
        sel = np.abs(ta - tc) <= band
        if sel.sum() < 30:
            continue
        # a flat face is sharp along the direction; a strip of a curved surface smears over its sagitta
        sharp = float(sel.sum()) / max(int((np.abs(ta - tc) <= 4.0 * band).sum()), 1)
        taken.append(tc)
        try:
            fit = fitting.fit_plane(Pa[sel], band=max(2.5 * noise, 0.5 * h))
        except ValueError:
            continue
        n = np.asarray(fit["normal"], float)
        if n @ d < 0:
            n = -n
        tilt = math.degrees(math.acos(min(1.0, abs(float(n @ d)))))
        if tilt > max_tilt_deg or fit["rms"] > 8.0 * max(noise, 0.002):
            continue  # a strip of a curved surface (the side of a shaft) is not a face
        p0 = np.asarray(fit["point"], float)
        # where the plane crosses the centre line
        s = float(((p0 - c0) @ n) / (d @ n))
        on = sel.copy()
        on[sel] = np.abs((Pa[sel] - p0) @ n) <= max(3.0 * fit["rms"], 2.0 * noise, 1e-6)
        across = sorted([float(np.ptp(Pa[on] @ u)), float(np.ptp(Pa[on] @ v))])
        if sharp < 0.75 and across[0] < 0.2 * across[1]:
            continue  # a long thin strip smeared along the direction: the side of a shaft seen edge on
        r = np.hypot(Pa[on] @ u - uc, Pa[on] @ v - vc)
        facing = None
        if oriented:
            m = float(np.median(Na[on] @ d))
            facing = 1 if m > 0.5 else -1 if m < -0.5 else None
        faces.append({"position": s - start, "_t": s, "point": _r(c0 + s * d), "normal": _r(n, 9),
                      "tilt_deg": round(tilt, 3), "rms": fit["rms"], "flatness": fit["flatness"],
                      "sharpness": round(sharp, 4),
                      "points": int(on.sum() * len(pts) / max(len(P), 1)), "facing": facing,
                      "radius_range": [float(np.percentile(r, 2)), float(np.percentile(r, 98))],
                      "size_across": [float(np.ptp(Pa[on] @ u)), float(np.ptp(Pa[on] @ v))]})
    if not faces:
        raise ValueError(f"No flat faces across {label} were found")
    # a thread's flanks leave small, slightly tilted patches at every turn: keep faces of a real size
    biggest = max(f["points"] for f in faces)
    faces = [f for f in faces if f["points"] >= 0.05 * biggest]
    if len(faces) > max_faces:
        faces = sorted(faces, key=lambda f: -f["points"])[:max_faces]
        warnings.append(f"Only the {max_faces} largest faces are listed")
    faces.sort(key=lambda f: f["_t"])
    for f in faces:
        if f["rms"] > 4.0 * max(noise, 1e-6):
            warnings.append(f"The face at {f['position']:.2f} mm is not flat (rms {f['rms']:.3f} mm): a fillet, "
                            "dimples or a doubled skin")
        if f["tilt_deg"] > 2.0:
            warnings.append(f"The face at {f['position']:.2f} mm is tilted {f['tilt_deg']:.1f}° to {label}")
    steps = [{"from": k, "to": k + 1, "distance": faces[k + 1]["_t"] - faces[k]["_t"]} for k in range(len(faces) - 1)]
    key = _label_faces(faces)
    if faces[0]["facing"] == 1:
        warnings.append(f"The start end face across {label} was not scanned: distances from it are not available")
    if len(faces) > 1 and faces[-1]["facing"] == -1:
        warnings.append(f"The far end face across {label} was not scanned: distances to it are not available")
    a = c0 + faces[0]["_t"] * d
    b = c0 + faces[-1]["_t"] * d
    overall = faces[-1]["_t"] - faces[0]["_t"]
    for f in faces:
        f["position"] = round(f.pop("_t") - start, 6)
    return {"direction": _r(d, 9), "direction_name": label, "faces": faces, "steps": steps,
            "key_distances": key, "overall": float(overall), "a": _r(a), "b": _r(b),
            "centre_line": {"point": _r(c0), "direction": _r(d, 9)}, "points_used": int(len(pts)),
            "noise": noise, "warnings": warnings}


def _label_faces(faces: list[dict]) -> list[dict]:
    """Name each face (end faces, shoulders, recess floors) and the distances people ask for (head height, length
    under the head, recess depth, overall). `facing` +1 looks towards the far end, so its material lies on the start
    side. The first face is the part's start face only if it looks outwards (towards the start); otherwise that end
    was not scanned. A face that looks back towards a real end face and fits inside that end face's hole is the
    floor of a recess (a socket) open at that end, not a shoulder. Adds `label` to every face; returns
    [{what, from, to, distance}]."""
    n = len(faces)
    if n == 0:
        return []
    first, last = faces[0], faces[-1]
    start_real = first["facing"] != 1
    end_real = n > 1 and last["facing"] != -1
    key: list[dict] = []

    def dist(i, j):
        return float(faces[j]["_t"] - faces[i]["_t"])

    def fits(f, end):
        return f["radius_range"][1] <= end["radius_range"][0] * 1.05 + 0.2

    ends = set()
    if start_real:
        first["label"] = "end face at the start"
        ends.add(0)
    if end_real:
        last["label"] = "end face at the far end"
        ends.add(n - 1)
    shoulders = []
    for k in range(n):
        if k in ends:
            continue
        f = faces[k]
        if f["facing"] == -1 and start_real and fits(f, first):
            f["label"] = "floor of a recess open at the start"
            key.append({"what": "recess depth from the start face", "from": 0, "to": k, "distance": dist(0, k)})
        elif f["facing"] == 1 and end_real and fits(f, last):
            f["label"] = "floor of a recess open at the far end"
            key.append({"what": "recess depth from the far end face", "from": k, "to": n - 1,
                        "distance": dist(k, n - 1)})
        elif f["facing"] == 1:
            f["label"] = "shoulder facing the far end (the part is wider on the start side)"
            shoulders.append(k)
        elif f["facing"] == -1:
            f["label"] = "shoulder facing the start (the part is wider on the far side)"
            shoulders.append(k)
        else:
            f["label"] = "step (which way it faces is unknown)"
            shoulders.append(k)
    for k in shoulders:
        f = faces[k]
        wide_start = f["facing"] == 1
        if start_real:
            key.append({"what": "head height (start face to shoulder)" if wide_start else
                        "length before the head (start face to shoulder)", "from": 0, "to": k, "distance": dist(0, k)})
        if end_real:
            key.append({"what": "length under the head (shoulder to far end face)" if wide_start else
                        "head height (shoulder to far end face)", "from": k, "to": n - 1, "distance": dist(k, n - 1)})
    if start_real and end_real:
        key.append({"what": "overall (end face to end face)", "from": 0, "to": n - 1, "distance": dist(0, n - 1)})
    return key


def _diameter_line(center: np.ndarray, axis: np.ndarray, radius: float, points: np.ndarray) -> tuple[list, list]:
    """End points of a diameter through the centre, across the side that holds the points."""
    rel = points - center
    radial = rel - np.outer(rel @ axis, axis)
    mean = radial.mean(axis=0)
    if np.linalg.norm(mean) < 1e-9 * max(radius, 1e-12):
        mean = fitting._basis(axis)[0]
    e = _unit(mean - (mean @ axis) * axis)
    return _r(center - radius * e), _r(center + radius * e)


def measure_diameter(geom: Geometry, region=None, axis_hint=None, frame: PartFrame | None = None) -> dict:
    """Cylinder fit (or a circle for a thin slice): {kind cylinder|circle, diameter, radius, axis, center, length,
    rms, coverage_deg, points_used, inliers, a, b (a diameter line), warnings}."""
    hint = None
    if axis_hint is not None:
        hint, _, part = _direction(axis_hint, frame, geom)
    pts, mask = _selected(geom, region, frame)
    normals = _geom_normals(geom)
    if normals is not None and len(normals) == len(mask):
        normals = normals[mask]
    if normals is not None:
        normals = normals[fitting.sample_indices(len(normals), 20_000, 1)]
    if len(pts) < 6:
        raise ValueError(f"The region has only {len(pts)} points - select more of the round surface")
    cyl = fitting.fit_cylinder(pts, axis_hint=hint, normals=normals)
    axis = np.asarray(cyl["axis"])
    warnings = []
    if cyl["length"] < 0.2 * cyl["radius"]:  # a thin ring: the axis is better defined by the plane of the ring
        circle = fitting.fit_circle_3d(pts, normal=hint)
        result = {"kind": "circle", "diameter": circle["diameter"], "radius": circle["radius"],
                  "axis": circle["normal"], "center": circle["center"], "length": cyl["length"],
                  "rms": circle["rms"], "coverage_deg": circle["coverage_deg"], "points_used": circle["points_used"],
                  "inliers": circle["inliers"], "points_in_region": int(len(pts))}
        axis, center = np.asarray(circle["normal"]), np.asarray(circle["center"])
    else:
        result = {"kind": "cylinder", "diameter": cyl["diameter"], "radius": cyl["radius"], "axis": cyl["axis"],
                  "center": cyl["point"], "length": cyl["length"], "rms": cyl["rms"],
                  "coverage_deg": cyl["coverage_deg"], "points_used": cyl["points_used"], "inliers": cyl["inliers"],
                  "points_in_region": cyl["points_in_region"]}
        center = np.asarray(cyl["point"])
    if result["coverage_deg"] < 120:
        warnings.append(f"Only {result['coverage_deg']:.0f} deg of the circumference is covered: the diameter is "
                        "less certain")
    if result["rms"] > 0.02 * result["radius"]:
        warnings.append(f"The points do not follow one {result['kind']} well (rms {result['rms']:.4g}): check the "
                        "region")
    if result["inliers"] < 0.8 * len(pts):
        warnings.append(f"{100 * (1 - result['inliers'] / len(pts)):.0f} % of the region's points are not on the "
                        f"{result['kind']} (other surfaces in the region)")
    result["a"], result["b"] = _diameter_line(center, axis, result["radius"], pts)
    result["warnings"] = warnings
    return result


def measure_sphere(geom: Geometry, region, frame: PartFrame | None = None) -> dict:
    """{center, radius, diameter, rms, coverage (0..1), points_used, inliers, a, b, warnings}."""
    pts, _ = _selected(geom, region, frame)
    fit = fitting.fit_sphere(pts)
    warnings = []
    if fit["coverage"] < 0.15:
        warnings.append(f"Only {100 * fit['coverage']:.0f} % of the sphere is covered: the radius is less certain")
    center = np.asarray(fit["center"])
    a, b = _diameter_line(center, fitting._basis(np.array([0.0, 0.0, 1.0]))[0], fit["radius"], pts)
    return {**{k: fit[k] for k in ("center", "radius", "diameter", "rms", "coverage", "points_used", "inliers",
                                   "points_in_region")}, "a": a, "b": b, "warnings": warnings}


def measure_plane(geom: Geometry, region, frame: PartFrame | None = None) -> dict:
    """{point, normal, rms, flatness, points_used, inliers, a, b (a short normal marker), warnings}."""
    pts, _ = _selected(geom, region, frame)
    fit = fitting.fit_plane(pts)
    normal, point = np.asarray(fit["normal"]), np.asarray(fit["point"])
    size = float(np.linalg.norm(np.ptp(pts, axis=0))) if len(pts) > 1 else 1.0
    warnings = []
    if fit["inlier_fraction"] < 0.8:
        warnings.append(f"{100 * (1 - fit['inlier_fraction']):.0f} % of the region's points are off the plane "
                        "(other surfaces in the region)")
    return {**{k: fit[k] for k in ("point", "normal", "rms", "flatness", "points_used", "inliers",
                                   "points_in_region")},
            "a": _r(point), "b": _r(point + 0.15 * size * normal), "warnings": warnings}


def _line_or_plane(geom: Geometry, region, frame) -> dict:
    pts, mask = _selected(geom, region, frame)
    if len(pts) < 3:
        raise ValueError("Each region needs at least 3 points")
    ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1] if len(pts) > 3 else np.array([1.0, 0.0, 0.0])
    if ev[1] < 0.02 * ev[0]:
        fit = fitting.fit_line(pts)
        return {"type": "line", "direction": fit["direction"], "point": fit["point"], "rms": fit["rms"],
                "length": fit["length"], "points_used": fit["points_used"]}
    fit = fitting.fit_plane(pts)
    out = {"type": "plane", "normal": fit["normal"], "point": fit["point"], "rms": fit["rms"],
           "flatness": fit["flatness"], "points_used": fit["points_used"]}
    normals = _geom_normals(geom)
    if normals is not None and len(normals) == len(mask):
        mean = normals[mask].mean(axis=0)
        if np.linalg.norm(mean) > 1e-9:
            out["outward"] = _r(np.asarray(fit["normal"]) * (1 if np.asarray(fit["normal"]) @ mean >= 0 else -1), 9)
    return out


def measure_angle(geom: Geometry, region_a, region_b, frame: PartFrame | None = None) -> dict:
    """Angle between a plane or line fitted in each region (a region that is long and thin is a line).

    {angle_deg (acute, 0..90), supplement_deg, normals_angle_deg (between the outward normals of two planes when
    the model has normals; the material angle at a convex edge is 180 minus it), fit_a, fit_b, a, b}."""
    fa, fb = _line_or_plane(geom, region_a, frame), _line_or_plane(geom, region_b, frame)
    va = np.asarray(fa.get("normal") or fa.get("direction"))
    vb = np.asarray(fb.get("normal") or fb.get("direction"))
    acute = fitting.angle_between(va, vb)
    if fa["type"] != fb["type"]:
        acute = 90.0 - acute  # line vs plane: angle to the plane, not to its normal
    out = {"angle_deg": acute, "supplement_deg": 180.0 - acute, "fit_a": fa, "fit_b": fb,
           "a": fa["point"], "b": fb["point"]}
    if fa.get("outward") and fb.get("outward"):
        c = float(np.clip(np.asarray(fa["outward"]) @ np.asarray(fb["outward"]), -1, 1))
        out["normals_angle_deg"] = math.degrees(math.acos(c))
    return out


# --------------------------------------------------------------------------- sections
def _section_plane(geom, plane: dict, frame: PartFrame | None) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                         np.ndarray, str]:
    """(point, normal, u, v, label) of a section plane {point, normal} | {direction: D, at: mm}."""
    if not isinstance(plane, dict):
        raise ValueError("plane must be {point: [x, y, z], normal: [x, y, z]} or {direction: D, at: mm}")
    if "direction" in plane:
        if set(plane) - {"direction", "at"} or "at" not in plane:
            raise ValueError("plane {direction, at}: give both, nothing else")
        at = plane["at"]
        if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at):
            raise ValueError("plane 'at' must be a number (mm from the part's robust minimum along the direction)")
        d, label, part = _direction(plane["direction"], frame, geom)
        if part and frame is None:
            frame = part_frame(geom)
        if part:
            base = float(frame.origin @ d)
        else:
            t = np.asarray(positions(geom)) @ d
            base = trimmed_range(t, vertex_weights(geom))[0]
        u, v = perpendicular_basis(d, frame if part else None, plane["direction"] if part else "")
        return d * (base + float(at)), d, u, v, f"{label} = {float(at):.4g}"
    if set(plane) != {"point", "normal"}:
        raise ValueError("plane must be {point: [x, y, z], normal: [x, y, z]} or {direction: D, at: mm}")
    raw = plane["point"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 3 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in raw):
        raise ValueError("plane 'point' must be three numbers [x, y, z]")
    point = np.asarray(raw, float)
    normal, label = direction_vector(plane["normal"], frame, geom)
    name = plane["normal"] if isinstance(plane["normal"], str) else ""
    if name in PART_AXES and frame is None:
        frame = part_frame(geom)
    u, v = perpendicular_basis(normal, frame if name in PART_AXES else None, name)
    return point, normal, u, v, f"plane ⟂ {label}"


def _chain_segments(ka: np.ndarray, kb: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> list[tuple[list, bool]]:
    """Join segments that share end keys (mesh edges) into polylines. Returns [(points, closed)]."""
    from collections import defaultdict

    adj = defaultdict(list)
    pos = {}
    for i in range(len(ka)):
        a, b = int(ka[i]), int(kb[i])
        if a == b:
            continue
        adj[a].append(b)
        adj[b].append(a)
        pos[a], pos[b] = pa[i], pb[i]
    seen_edges = set()
    lines = []
    # open chains start at degree-1 nodes, then the remaining loops
    starts = [k for k, nb in adj.items() if len(nb) == 1] + list(adj)
    for s in starts:
        if all(tuple(sorted((s, nb))) in seen_edges for nb in adj[s]):
            continue
        chain = [s]
        cur, prev = s, None
        while True:
            nxt = None
            for nb in adj[cur]:
                e = tuple(sorted((cur, nb)))
                if e not in seen_edges:
                    nxt = nb
                    seen_edges.add(e)
                    break
            if nxt is None:
                break
            chain.append(nxt)
            prev, cur = cur, nxt
            if cur == s:
                break
        if len(chain) >= 2:
            closed = chain[0] == chain[-1] and len(chain) > 3
            lines.append(([pos[k] for k in chain], closed))
    return lines


def _mesh_section(geom, point, normal) -> tuple[list[tuple[list, bool]], int]:
    v = np.asarray(geom.vertices)
    t = np.asarray(geom.triangles, dtype=np.int64)
    d = (v - point) @ normal
    side = d >= 0
    s = side[t]
    cross = s.any(axis=1) & ~s.all(axis=1)
    tri = t[cross]
    if len(tri) == 0:
        return [], 0
    n_v = len(v)
    keys, points = [], []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = tri[:, i], tri[:, j]
        ok = side[a] != side[b]
        da, db = d[a], d[b]
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = da / (da - db)
            p = v[a] + tt[:, None] * (v[b] - v[a])
        key = np.minimum(a, b) * n_v + np.maximum(a, b)
        keys.append(np.where(ok, key, -1))
        points.append(p)
    keys = np.stack(keys, axis=1)
    points = np.stack(points, axis=1)
    ka, kb, pa, pb = [], [], [], []
    for row in range(len(tri)):
        idx = np.flatnonzero(keys[row] >= 0)
        if len(idx) == 2:
            ka.append(keys[row, idx[0]])
            kb.append(keys[row, idx[1]])
            pa.append(points[row, idx[0]])
            pb.append(points[row, idx[1]])
    return _chain_segments(np.asarray(ka), np.asarray(kb), np.asarray(pa), np.asarray(pb)), int(len(tri))


def _cloud_section(pts, point, normal, u, v, spacing: float, noise: float = 0.0) -> tuple[list[tuple[list, bool]],
                                                                                          int, np.ndarray]:
    """Points within a thin band of the plane (one point spacing, at least 2 x noise), projected onto it, thinned
    on a grid of one spacing and chained by nearest neighbours; chains whose ends meet are joined."""
    from scipy.spatial import cKDTree

    half = max(spacing, 2.0 * noise, 1e-9)
    dist = (pts - point) @ normal
    band = pts[np.abs(dist) <= half]
    if len(band) == 0:
        return [], 0, np.zeros((0, 2))
    uv = np.c_[(band - point) @ u, (band - point) @ v]
    cell = max(spacing, 1e-9)
    key = np.floor(uv / cell).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(first), 2))
    np.add.at(sums, inverse, uv)
    thin = sums / np.bincount(inverse, minlength=len(first))[:, None]
    tree = cKDTree(thin)
    gap = float(np.median(tree.query(thin, k=2)[0][:, 1])) if len(thin) > 2 else cell
    max_step = max(3.0 * cell, 4.0 * gap)
    used = np.zeros(len(thin), bool)
    chains: list[list[int]] = []
    for start in np.lexsort((thin[:, 1], thin[:, 0])):
        if used[start]:
            continue
        used[start] = True
        chain = [int(start)]
        for direction in (0, 1):  # grow from both ends
            cur = chain[-1] if direction == 0 else chain[0]
            while True:
                dists, nbs = tree.query(thin[cur], k=min(16, len(thin)), distance_upper_bound=max_step)
                nxt = next((int(nb) for dd, nb in zip(np.atleast_1d(dists), np.atleast_1d(nbs))
                            if np.isfinite(dd) and nb < len(thin) and not used[nb]), None)
                if nxt is None:
                    break
                used[nxt] = True
                if direction == 0:
                    chain.append(nxt)
                else:
                    chain.insert(0, nxt)
                cur = nxt
        chains.append(chain)
    merged = True
    while merged and len(chains) > 1:  # join chains whose end points are close (in either orientation)
        merged = False
        for a in range(len(chains)):
            for b in range(a + 1, len(chains)):
                ca, cb = chains[a], chains[b]
                options = ((ca[-1], cb[0], lambda: ca + cb), (ca[-1], cb[-1], lambda: ca + cb[::-1]),
                           (ca[0], cb[-1], lambda: cb + ca), (ca[0], cb[0], lambda: cb[::-1] + ca))
                for end_a, end_b, join in options:
                    if np.linalg.norm(thin[end_a] - thin[end_b]) <= 2 * max_step:
                        chains[a] = join()
                        del chains[b]
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
    lines = []
    for chain in chains:
        if len(chain) < 3:
            continue
        closed = bool(np.linalg.norm(thin[chain[0]] - thin[chain[-1]]) <= 2 * max_step)
        pts3 = [point + thin[k, 0] * u + thin[k, 1] * v for k in chain]
        if closed:
            pts3.append(pts3[0])
        lines.append((pts3, closed))
    return lines, int(len(band)), uv


def measure_section(geom: Geometry, plane: dict, frame: PartFrame | None = None,
                    spacing: float | None = None) -> dict:
    """Cross-section with a plane: {polylines [[[x, y, z], ...]], closed [bool], width, height (extents in the
    plane along u / v), bbox2d {min, max}, basis {origin, u, v, normal}, length (total polyline length),
    points_used, plane_label}. Meshes: exact triangle / plane intersection; clouds: the points within half a
    point spacing of the plane, chained in order."""
    point, normal, u, v, label = _section_plane(geom, plane, frame)
    if is_cloud(geom):
        pts = np.asarray(geom.points)
        if not spacing or spacing <= 0:
            spacing = estimate_spacing(pts[fitting.sample_indices(len(pts), 200_000)]) or 1e-3
        noise = fitting.local_noise(pts[fitting.sample_indices(len(pts), 20_000, 7)]) if len(pts) > 32 else 0.0
        lines, used, uv = _cloud_section(pts, point, normal, u, v, spacing, noise)
    else:
        lines, used = _mesh_section(geom, point, normal)
        uv = None
    if not lines:
        raise ValueError(f"The plane ({label}) does not cut the model")
    total = sum(len(p) for p, _ in lines)
    step = max(1, int(math.ceil(total / SECTION_MAX_POINTS)))
    polylines, closed, length = [], [], 0.0
    all_pts = []
    for p, is_closed in lines:
        arr = np.asarray(p, float)
        length += float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()) if len(arr) > 1 else 0.0
        all_pts.append(arr)
        thin = arr[::step] if step > 1 else arr
        if step > 1 and not np.array_equal(thin[-1], arr[-1]):
            thin = np.vstack([thin, arr[-1]])
        polylines.append([_r(q) for q in thin])
        closed.append(bool(is_closed))
    stacked = np.vstack(all_pts)
    uv2 = np.c_[(stacked - point) @ u, (stacked - point) @ v] if uv is None else uv
    lo, hi = uv2.min(axis=0), uv2.max(axis=0)
    return {"polylines": polylines, "closed": closed, "width": float(hi[0] - lo[0]), "height": float(hi[1] - lo[1]),
            "bbox2d": {"min": _r(lo), "max": _r(hi)},
            "basis": {"origin": _r(point), "u": _r(u, 9), "v": _r(v, 9), "normal": _r(normal, 9)},
            "length": length, "points_used": used, "plane_label": label, "thinned_every": step}


# --------------------------------------------------------------------------- UI helpers
def overlay_items(kind: str, result: dict) -> list[dict]:
    """measure_overlay items (Contract 7) for a measurement result."""
    def item(label, value, a, b, unit="mm"):
        return {"label": label, "kind": kind, "value": round(float(value), 6), "unit": unit, "a": a, "b": b}

    if kind == "extent":
        return [item(f"{result['direction_name']} extent", result["length"], result["a"], result["b"])]
    if kind == "caliper":
        return [item(f"caliper {result['direction_name']}", result["distance"], result["a"], result["b"])]
    if kind == "faces":
        # one dimension line per step, drawn just outside the larger of the two faces, parallel to the direction
        faces, d = result["faces"], np.asarray(result["direction"], float)
        u, v = perpendicular_basis(d)
        out = []
        key = result.get("key_distances") or [{"what": f"faces {s['from'] + 1}→{s['to'] + 1}", **s}
                                              for s in result["steps"]]
        for i, s in enumerate(key):  # fanned round the axis so the lines never hide behind each other
            fa, fb = faces[s["from"]], faces[s["to"]]
            r = max(f["radius_range"][1] for f in faces[s["from"]:s["to"] + 1]) * (1.1 + 0.1 * (i // 4))
            side = math.cos(i * math.pi / 2) * u + math.sin(i * math.pi / 2) * v
            a = np.asarray(fa["point"], float) + r * side
            b = np.asarray(fb["point"], float) + r * side
            out.append(item(s["what"].split(" (")[0], s["distance"], _r(a), _r(b)))
        return out
    if kind == "diameter":
        return [item(f"Ø {result['kind']}", result["diameter"], result["a"], result["b"])]
    if kind == "sphere":
        return [item("Ø sphere", result["diameter"], result["a"], result["b"])]
    if kind == "plane":
        return [item("flatness", result["flatness"], result["a"], result["b"])]
    if kind == "angle":
        return [item("angle", result["angle_deg"], result["a"], result["b"], "deg")]
    if kind == "section":
        basis = result["basis"]
        o, u, v = (np.asarray(basis[k]) for k in ("origin", "u", "v"))
        lo, hi = np.asarray(result["bbox2d"]["min"]), np.asarray(result["bbox2d"]["max"])
        mid_v, mid_u = (lo[1] + hi[1]) / 2, (lo[0] + hi[0]) / 2
        return [item("section width", result["width"], _r(o + lo[0] * u + mid_v * v), _r(o + hi[0] * u + mid_v * v)),
                item("section height", result["height"], _r(o + mid_u * u + lo[1] * v), _r(o + mid_u * u + hi[1] * v))]
    if kind in ("distance", "points"):
        pts = result.get("points") or result.get("snapped") or []
        if len(pts) >= 2:
            value = result.get("distance", result.get("total", 0.0))
            return [item("distance", value, _r(pts[0]), _r(pts[-1]))]
    return []
