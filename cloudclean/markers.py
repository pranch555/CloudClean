"""Marker stickers: find the round holes they leave in a scan, and line scans up on the stickers they share.

The scanner does not capture the centre of a marker sticker, so every sticker on the part leaves a small, round
hole in the scan. Stickers are placed irregularly, so three or more of them seen in two scans fix the one way the
scans fit - exactly, even when the part looks alike from several sides (a 12-sided head, a thread), where the
surface alone leaves the pose ambiguous (docs/accuracy-investigation-2026-09-24.md).

find_markers   boundary points -> rings -> circle fits; a ring counts as a sticker hole when it is round, closed,
               sticker-sized, empty inside and surrounded by surface (the open edge of a partial scan is not).
match_markers  the rigid motion that maps one scan's stickers onto another's: hypotheses from pairs of stickers
               with matching distances, completed by distance consistency, solved with Kabsch, scored by inliers.
align_by_markers  stickers -> pose -> a short ICP refinement against the surface, with how far it moved.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .io import Geometry, estimate_spacing, to_cloud

Log = Callable[[str], None]

MIN_DIAMETER = 2.0      # mm: the hole left by a sticker (Revopoint markers are 3-6 mm)
MAX_DIAMETER = 12.0
MIN_COVERAGE = 300.0    # degrees of the rim that must be scanned all around the hole
MIN_COMMON = 3


def _cloud(geom) -> o3d.geometry.PointCloud:
    pcd = geom if isinstance(geom, o3d.geometry.PointCloud) else to_cloud(geom)
    return o3d.geometry.PointCloud(pcd)


def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic circle fit refined by a few Gauss-Newton steps on the geometric distance."""
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r = math.sqrt(max(float(sol[2]) + cx * cx + cy * cy, 1e-12))
    for _ in range(5):
        dx, dy = x - cx, y - cy
        rho = np.maximum(np.hypot(dx, dy), 1e-12)
        J = np.c_[-dx / rho, -dy / rho, -np.ones_like(rho)]
        step, *_ = np.linalg.lstsq(J, -(rho - r), rcond=None)
        cx, cy, r = cx + float(step[0]), cy + float(step[1]), r + float(step[2])
    return cx, cy, abs(r)


def _coverage(angles_deg: np.ndarray, bins: int = 36) -> float:
    hist = np.histogram(angles_deg, bins=bins, range=(-180.0, 180.0))[0]
    return float((hist > 0).mean() * 360.0)


def find_markers(geom, spacing: float | None = None, min_diameter: float = MIN_DIAMETER,
                 max_diameter: float = MAX_DIAMETER) -> dict:
    """The sticker holes of a scan (point cloud or mesh vertices): round empty discs in the surface.

    1. Points on an edge (a permissive boundary test) look across their gap: candidate hole centres are placed
       at several radii along the gap direction, and kept when the nearest scan point is about that far away.
    2. Candidates of one hole pile up at its centre; each pile is refined with a circle fit to the points
       around it, and kept when the hole is round, sticker-sized, empty inside and surrounded by surface (the
       open edge of a partial scan has surface on one side only, so it never counts).

    {markers: [{center, normal, diameter, rms, coverage_deg, points}], spacing, edge_points}"""
    if isinstance(geom, o3d.geometry.TriangleMesh) and len(geom.triangles):
        return _mesh_markers(geom, min_diameter, max_diameter)
    pcd = _cloud(geom)
    pts = np.asarray(pcd.points)
    if len(pts) < 100:
        return {"markers": [], "spacing": 0.0, "edge_points": 0}
    spacing = spacing or estimate_spacing(pcd.points)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    normals = np.asarray(pcd.normals)
    tree = cKDTree(pts)
    tp = o3d.t.geometry.PointCloud.from_legacy(pcd)
    _, mask = tp.compute_boundary_points(5.0 * spacing, 40, 100.0)
    edge = np.nonzero(mask.numpy().astype(bool))[0]
    if len(edge) > 60_000:  # enough to find every hole; keeps big scans fast
        edge = edge[np.random.default_rng(0).choice(len(edge), 60_000, replace=False)]
    if len(edge) < 8:
        return {"markers": [], "spacing": float(spacing), "edge_points": int(len(edge))}
    # 1. each edge point looks across its gap
    e = pts[edge]
    nb = tree.query(e, k=24)[1]
    g = e - pts[nb].mean(axis=1)
    n_e = normals[edge]
    g -= np.sum(g * n_e, axis=1)[:, None] * n_e
    length = np.linalg.norm(g, axis=1)
    keep = length > 0.1 * spacing
    e, g = e[keep], g[keep] / length[keep][:, None]
    # walking into a round hole from its rim, the rim point stays the nearest point until the walk reaches the
    # centre: the centre is the farthest step whose nearest point is still (about) as far as the step
    radii = np.arange(0.4 * min_diameter, max_diameter / 2 + 2.0 * spacing + 1e-9, max(0.1, spacing))
    steps = e[:, None, :] + g[:, None, :] * radii[None, :, None]
    dist = tree.query(steps.reshape(-1, 3))[0].reshape(len(e), len(radii))
    inside = dist >= 0.93 * radii[None, :]
    reach = np.where(inside.all(axis=1), -1, np.argmin(inside, axis=1) - 1)  # last step still inside the hole
    good = (reach >= 1) & (radii[np.clip(reach, 0, None)] >= min_diameter / 2 * 0.8)
    rows = np.nonzero(good)[0]
    centres = steps[rows, reach[rows]]
    dist = dist[rows, reach[rows]]
    if len(centres) < 5:
        return {"markers": [], "spacing": float(spacing), "edge_points": int(len(edge))}
    # 2. piles of candidates = holes
    labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centres))
                        .cluster_dbscan(eps=max(0.4, 3.0 * spacing), min_points=5))
    markers = []
    for lab in range(labels.max() + 1 if len(labels) else 0):
        sel = labels == lab
        c0 = np.median(centres[sel], axis=0)
        r0 = float(np.median(dist[sel]))
        if not (min_diameter / 2 * 0.8 <= r0 <= max_diameter / 2 * 1.2):
            continue
        near_idx = tree.query_ball_point(c0, r0 + max(2.0, 0.6 * r0))
        if len(near_idx) < 20:
            continue
        near = pts[near_idx]
        N = normals[near_idx]
        w, V = np.linalg.eigh(N.T @ N)
        n = V[:, -1]  # the surface's normal around the hole (sign-free)
        helper = np.eye(3)[int(np.argmin(np.abs(n)))]
        u = np.cross(n, helper)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        Y = near - c0
        flat = np.abs(Y @ n) < max(1.0, 0.35 * r0)
        x, y = Y[flat] @ u, Y[flat] @ v
        rr = np.hypot(x, y)
        rim = (rr > r0 - 2.0 * spacing) & (rr < r0 + 3.0 * spacing)
        if rim.sum() < 10:
            continue
        cx, cy, r = _fit_circle(x[rim], y[rim])
        # the fitted circle passes through the rim points just outside the hole
        d = 2.0 * r
        if not (min_diameter <= d <= max_diameter):
            continue
        rho = np.hypot(x - cx, y - cy)
        ring = (rho > r - 2.0 * spacing) & (rho < r + 2.0 * spacing)
        if ring.sum() < 10:
            continue
        rms = float(np.sqrt(np.mean((rho[ring] - r) ** 2)))
        cover = _coverage(np.degrees(np.arctan2(y[ring] - cy, x[ring] - cx)))
        inside = int((rho < 0.6 * r).sum())
        annulus = rho > 1.05 * r
        around = _coverage(np.degrees(np.arctan2(y[annulus] - cy, x[annulus] - cx)))
        if (rms > max(0.12 * r, 1.5 * spacing) or cover < MIN_COVERAGE or annulus.sum() < 12
                or inside > 0.02 * annulus.sum() + 2 or around < 300.0):
            continue
        centre = c0 + cx * u + cy * v
        if float(np.mean(N @ n)) < 0:
            n = -n  # point the normal like the surface around the hole
        markers.append({"center": centre.round(5).tolist(), "normal": n.round(6).tolist(), "diameter": round(d, 4),
                        "rms": round(rms, 5), "coverage_deg": round(cover, 1), "points": int(ring.sum())})
    # one hole found twice (two piles): keep the rounder
    markers.sort(key=lambda m: m["rms"])
    kept: list[dict] = []
    for m in markers:
        if all(np.linalg.norm(np.subtract(m["center"], k["center"])) > 0.5 * k["diameter"] for k in kept):
            kept.append(m)
    return {"markers": kept, "spacing": float(spacing), "edge_points": int(len(edge))}


def _mesh_markers(mesh: o3d.geometry.TriangleMesh, min_diameter: float, max_diameter: float) -> dict:
    """Sticker holes of a mesh: its hole outlines (cloudclean/holes.py) that are round and sticker-sized."""
    from .holes import find_holes

    v = np.asarray(mesh.vertices)
    markers = []
    for h in find_holes(mesh)["holes"]:
        if h["outer"] or not (min_diameter <= h["diameter"] <= max_diameter):
            continue
        # a circle has area / (perimeter^2 / 4 pi) = 1; ragged or long openings far less
        if 4 * math.pi * h["area"] / max(h["perimeter"] ** 2, 1e-12) < 0.8:
            continue
        loop = v[h["loop"]]
        n = np.asarray(h["normal"], float)
        n /= max(np.linalg.norm(n), 1e-12)
        helper = np.eye(3)[int(np.argmin(np.abs(n)))]
        u = np.cross(n, helper)
        u /= np.linalg.norm(u)
        w = np.cross(n, u)
        c = loop.mean(axis=0)
        cx, cy, r = _fit_circle((loop - c) @ u, (loop - c) @ w)
        rho = np.hypot((loop - c) @ u - cx, (loop - c) @ w - cy)
        markers.append({"center": (c + cx * u + cy * w).round(5).tolist(), "normal": n.round(6).tolist(),
                        "diameter": round(2 * r, 4), "rms": round(float(np.sqrt(np.mean((rho - r) ** 2))), 5),
                        "coverage_deg": 360.0, "points": int(len(loop))})
    return {"markers": markers, "spacing": 0.0, "edge_points": 0}


def _rigid(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    from .register import rigid_from_pairs

    return rigid_from_pairs(src, dst)


def _angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1) / 2))))


def match_markers(moving: list[dict], reference: list[dict], tolerance: float = 0.3,
                  min_common: int = MIN_COMMON) -> dict | None:
    """The rigid motion mapping `moving`'s stickers onto `reference`'s, or None when fewer than `min_common` agree.

    {T, pairs [(moving index, reference index)], common, rms_mm, ambiguous (another pose explains almost as many)}"""
    if len(moving) < min_common or len(reference) < min_common:
        return None
    P = np.array([m["center"] for m in moving], float)
    Q = np.array([m["center"] for m in reference], float)
    NP = np.array([m["normal"] for m in moving], float)
    NQ = np.array([m["normal"] for m in reference], float)
    dia_p = np.array([m["diameter"] for m in moving])
    dia_q = np.array([m["diameter"] for m in reference])
    DP = np.linalg.norm(P[:, None] - P[None], axis=2)
    DQ = np.linalg.norm(Q[:, None] - Q[None], axis=2)
    tree = cKDTree(Q)
    n, m = len(P), len(Q)
    iu = np.triu_indices(n, 1)
    order = np.argsort(-DP[iu])[:300]  # long pairs first: they fix the rotation best
    hypotheses: list[tuple[int, float, np.ndarray, list]] = []
    seen: set = set()

    def score(T):
        moved = P @ T[:3, :3].T + T[:3, 3]
        dist, nn = tree.query(moved)
        turned = NP @ T[:3, :3].T
        pairs, used = [], set()
        for p in np.argsort(dist):
            q = int(nn[p])
            if dist[p] > tolerance or q in used:
                continue
            if abs(float(turned[p] @ NQ[q])) < math.cos(math.radians(45)):
                continue
            if abs(dia_p[p] - dia_q[q]) > max(0.5, 0.25 * dia_q[q]):
                continue
            used.add(q)
            pairs.append((int(p), q))
        return pairs

    for idx in order:
        i, j = int(iu[0][idx]), int(iu[1][idx])
        d = DP[i, j]
        if d < 2.0:
            continue
        ks, ls = np.nonzero(np.abs(DQ - d) < tolerance)
        for k, l in zip(ks.tolist(), ls.tolist()):
            if k == l or (i, j, k, l) in seen:
                continue
            seen.add((i, j, k, l))
            E = np.abs(DP[i][:, None] - DQ[k][None, :]) + np.abs(DP[j][:, None] - DQ[l][None, :])
            E[[i, j], :] = np.inf
            E[:, [k, l]] = np.inf
            rows = np.argmin(E, axis=1)
            good = np.nonzero(E[np.arange(n), rows] < 2 * tolerance)[0]
            cand, used = [(i, k), (j, l)], {k, l}
            for p in good[np.argsort(E[good, rows[good]])]:
                q = int(rows[p])
                if q not in used:
                    used.add(q)
                    cand.append((int(p), q))
            if len(cand) < min_common:
                continue
            T = _rigid(P[[c[0] for c in cand]], Q[[c[1] for c in cand]])
            pairs = score(T)
            if len(pairs) < min_common:
                continue
            T = _rigid(P[[c[0] for c in pairs]], Q[[c[1] for c in pairs]])
            pairs = score(T)
            if len(pairs) < min_common:
                continue
            moved = P[[c[0] for c in pairs]] @ T[:3, :3].T + T[:3, 3]
            rms = float(np.sqrt(np.mean(np.sum((moved - Q[[c[1] for c in pairs]]) ** 2, axis=1))))
            hypotheses.append((len(pairs), rms, T, pairs))
            if len(pairs) == min(n, m) and rms < tolerance / 2:
                break
    if not hypotheses:
        return None
    hypotheses.sort(key=lambda h: (-h[0], h[1]))
    count, rms, T, pairs = hypotheses[0]
    centre = P.mean(axis=0)
    ambiguous = False
    for c2, _, T2, _ in hypotheses[1:]:
        if c2 < max(min_common, count - 1):
            break
        moved = np.linalg.norm((T[:3, :3] @ centre + T[:3, 3]) - (T2[:3, :3] @ centre + T2[:3, 3]))
        if _angle_deg(T[:3, :3].T @ T2[:3, :3]) > 2.0 or moved > 1.0:
            ambiguous = True
            break
    return {"T": T, "pairs": pairs, "common": count, "rms_mm": rms, "ambiguous": ambiguous}


def sticker_pose(moving, ref, spacing: float | None = None, log: Log = print,
                 tolerance: float | None = None) -> tuple[np.ndarray, dict]:
    """The pose of `moving` on `ref` from the stickers both scans share, before any surface refinement.

    Raises ValueError (plain words, with the counts) when fewer than 3 stickers are common."""
    src, tgt = _cloud(moving), _cloud(ref)
    spacing = spacing or float(np.median([estimate_spacing(src.points), estimate_spacing(tgt.points)]))
    found_m = find_markers(moving if isinstance(moving, o3d.geometry.TriangleMesh) else src, spacing)["markers"]
    found_r = find_markers(ref if isinstance(ref, o3d.geometry.TriangleMesh) else tgt, spacing)["markers"]
    tol = tolerance or max(0.3, 2.0 * spacing)
    log(f"  stickers: {len(found_m)} in the moving scan, {len(found_r)} in the reference")
    match = match_markers(found_m, found_r, tol)
    if match is None:
        if not found_m and not found_r:
            raise ValueError("No marker stickers were found in either scan. Only stickers on the part itself count "
                             "(not on the turntable or a mat), and the round holes they leave must stay open in the "
                             "exported scan.")
        raise ValueError(f"Fewer than {MIN_COMMON} marker stickers are common to both scans ({len(found_m)} found in "
                         f"one scan, {len(found_r)} in the other). Stickers on the part itself leave round holes in "
                         "each scan; use at least 3 that both scans see.")
    info = {"markers_moving": len(found_m), "markers_reference": len(found_r), "common": match["common"],
            "marker_rms_mm": round(match["rms_mm"], 4), "ambiguous": bool(match["ambiguous"]),
            "pairs": match["pairs"], "tolerance_mm": round(tol, 4)}
    log(f"  stickers: {match['common']} common, fit {match['rms_mm']:.3f} mm"
        + (" - another pose explains almost as many, check the result" if match["ambiguous"] else ""))
    return np.asarray(match["T"]), info


def align_by_markers(moving: Geometry, ref: Geometry, log: Log = print, tolerance: float | None = None,
                     refine: bool = True) -> tuple[np.ndarray, dict]:
    """Pose of `moving` on `ref` from the stickers both scans share, refined against the surface.

    Raises ValueError (plain words, with the counts) when fewer than 3 stickers are common."""
    src, tgt = _cloud(moving), _cloud(ref)
    spacing = float(np.median([estimate_spacing(src.points), estimate_spacing(tgt.points)]))
    T0, info = sticker_pose(moving, ref, spacing, log, tolerance)
    tol = info["tolerance_mm"]
    T = T0
    if refine:
        from .register import MergeParams, refine_icp

        fine = spacing * MergeParams().icp_threshold_multiplier
        T, fit, rmse = refine_icp(src, tgt, T0, [fine * 2, fine])
        centre = np.asarray(src.get_center())
        shift = float(np.linalg.norm((T[:3, :3] @ centre + T[:3, 3]) - (T0[:3, :3] @ centre + T0[:3, 3])))
        turn = _angle_deg(T0[:3, :3].T @ T[:3, :3])
        info.update(fitness=round(float(fit), 4), rmse=float(rmse), refine_shift_mm=round(shift, 4),
                    refine_turn_deg=round(turn, 4))
        if shift > max(1.0, 5 * tol) or turn > 1.0:
            info["warning"] = (f"the surface pulled the scan {shift:.2f} mm / {turn:.2f}° away from the stickers' "
                               "pose - check the stickers were not moved between scans")
            log(f"  WARNING: {info['warning']}")
    return np.asarray(T), info
