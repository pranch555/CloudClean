"""Inspection: align a scan to its CAD model / reference mesh and measure the deviation.

Nothing is ever rescaled: the scan is only moved rigidly into the CAD frame. A scale factor is
estimated and reported (it usually points at a calibration problem) but never applied."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .io import Geometry, estimate_spacing, to_cloud
from .params import ParamsMixin
from .register import _rival_pose, global_candidates, refine_icp

Log = Callable[[str], None]
o3c = o3d.core


@dataclass
class CompareParams(ParamsMixin):
    align: str = "auto"                 # auto (global search + ICP) | icp (roughly aligned already) | none
    tolerance: float = 0.1              # +/- band counted as "within tolerance" (scan units, mm)
    max_distance: float = 0.0           # farther points are not part of the part; 0 -> 3 % of reference diagonal
    sample_points: int = 0              # >0: measure a random subset of this many scan points (0 = all)
    estimate_scale: bool = True         # estimate (never apply) a scale factor between scan and CAD
    coverage_threshold: float = 0.0     # CAD surface counts as scanned within this distance; 0 -> auto


def rotation_angle(R: np.ndarray) -> float:
    """Rotation angle in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# --------------------------------------------------------------------------- reference surface
def is_closed(triangles: np.ndarray) -> bool:
    """Every edge shared by exactly two triangles (fast; TriangleMesh.is_watertight also tests
    self-intersection, which is very slow on large meshes)."""
    F = np.asarray(triangles, dtype=np.int64)
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges[:, 0] * (int(F.max()) + 1) + edges[:, 1], return_counts=True)
    return bool(len(F) and np.all(counts == 2))


class ReferenceSurface:
    """Reference mesh with consistently outward triangle normals and exact closest-point queries."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh, log: Log = print):
        mesh = o3d.geometry.TriangleMesh(mesh)
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) == 0:
            raise ValueError("The reference has no triangles - a CAD model or mesh is needed")
        try:
            consistent = bool(mesh.orient_triangles())
        except Exception:
            consistent = False
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.triangles, dtype=np.int64)
        self.watertight = is_closed(F)
        self.center = (V.min(0) + V.max(0)) / 2
        cross = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        if self.watertight:
            outward = np.einsum("ij,ij->", V[F[:, 0]] - self.center, cross) > 0      # 6 x signed volume
        else:  # open surface: area-weighted vote of normals pointing away from the centre
            outward = np.einsum("ij,ij->", (V[F].mean(1) - self.center), cross) >= 0
        if not outward:
            F = F[:, ::-1].copy()
            mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))
            cross = -cross
            log("  reference normals pointed inwards - flipped them")
        if not consistent:
            log("  WARNING: reference triangles could not be oriented consistently (non-manifold mesh); "
                "the sign of deviations may be wrong in places")
        self.consistent = consistent
        area2 = np.linalg.norm(cross, axis=1)
        self.normals = cross / np.maximum(area2, 1e-300)[:, None]
        self.areas = area2 / 2
        self.mesh = mesh
        self.mesh.compute_vertex_normals()
        self.vertices, self.triangles = V, F
        self.diagonal = float(np.linalg.norm(V.max(0) - V.min(0)))
        self.scene = o3d.t.geometry.RaycastingScene()
        # Coordinates relative to the centre keep float32 round-off in the ray tracer at the micron level.
        self.scene.add_triangles(o3c.Tensor((V - self.center).astype(np.float32)), o3c.Tensor(F.astype(np.uint32)))

    def query(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Signed distance (+ = outside), closest points and triangle normals for each point."""
        return self.closest(pts)[:3]

    def closest(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """query() plus the index of each point's closest triangle."""
        rel = np.asarray(pts, dtype=np.float64) - self.center
        res = self.scene.compute_closest_points(o3c.Tensor(rel.astype(np.float32)))
        tri = res["primitive_ids"].numpy().astype(np.int64)
        q = res["points"].numpy().astype(np.float64)
        n = self.normals[tri]
        plane = np.einsum("ij,ij->i", rel - self.vertices[self.triangles[tri, 0]] + self.center, n)
        unsigned = np.linalg.norm(rel - q, axis=1)
        signed = np.where(plane >= 0, 1.0, -1.0) * np.maximum(unsigned, np.abs(plane))
        return signed, q + self.center, n, tri

    def sample(self, count: int, seed: int = 0, triangles: bool = False):
        """Area-uniform random surface samples with their (outward) normals (and triangle indices)."""
        rng = np.random.default_rng(seed)
        tri = rng.choice(len(self.areas), size=count, p=self.areas / self.areas.sum())
        r1, r2 = rng.random(count), rng.random(count)
        s = np.sqrt(r1)
        a, b, c = (self.vertices[self.triangles[tri, k]] for k in range(3))
        pts = a * (1 - s)[:, None] + b * (s * (1 - r2))[:, None] + c * (s * r2)[:, None]
        return (pts, self.normals[tri], tri) if triangles else (pts, self.normals[tri])


# --------------------------------------------------------------------------- exact fitting
def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def fit_to_surface(ref: ReferenceSurface, pts: np.ndarray, T: np.ndarray, max_distance: float, floor: float,
                   with_scale: bool = False, iterations: int = 60) -> tuple[np.ndarray, dict]:
    """Robust point-to-plane fit against the exact reference surface (closest points on the triangles).

    Tukey weights with a scale from the residual spread (never below `floor`); points farther than
    max_distance are ignored. with_scale adds a uniform scale about the points' centroid (7 DOF)."""
    T = np.asarray(T, dtype=float).copy()
    k = max_distance
    info = {"iterations": 0, "scale": 1.0}
    for it in range(iterations):
        p = _apply(T, pts)
        r, _, n = ref.query(p)
        use = np.abs(r) < max_distance
        if use.sum() < 10:
            break
        sigma = 1.4826 * float(np.median(np.abs(r[use] - np.median(r[use]))))
        k = min(max(4.685 * sigma, floor), max_distance)
        w = np.where(np.abs(r) < k, (1 - (r / k) ** 2) ** 2, 0.0)
        sel = w > 0
        p, n, r, w = p[sel], n[sel], r[sel], w[sel]
        c = p.mean(0)
        cols = [np.cross(p - c, n), n]
        if with_scale:
            cols.append(np.einsum("ij,ij->i", p - c, n)[:, None])
        A = np.hstack(cols)
        Aw = A * w[:, None]
        try:
            x = np.linalg.solve(Aw.T @ A + 1e-12 * np.eye(A.shape[1]), -Aw.T @ r)
        except np.linalg.LinAlgError:
            break
        dT = np.eye(4)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(x[:3])
        s = 1.0 + (x[6] if with_scale else 0.0)
        dT[:3, :3] = s * R
        dT[:3, 3] = c + x[3:6] - s * R @ c        # rotate / scale about the centroid, then translate
        T = dT @ T
        info["scale"] *= s
        info["iterations"] = it + 1
        if np.linalg.norm(x[:3]) < 1e-7 and np.linalg.norm(x[3:6]) < 1e-6 * max(ref.diagonal, 1e-9) and \
                (not with_scale or abs(x[6]) < 1e-7):
            break
    info["k"] = float(k)
    return T, info


# --------------------------------------------------------------------------- global alignment
def _principal_frame(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = pts.mean(0)
    _, s, Vt = np.linalg.svd(pts - c, full_matrices=False)
    R = Vt.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return c, R, s / np.sqrt(len(pts))


def _pca_candidates(scan_pts: np.ndarray, ref_pts: np.ndarray) -> list[tuple[str, np.ndarray]]:
    cs, Rs, ss = _principal_frame(scan_pts)
    cr, Rr, sr = _principal_frame(ref_pts)
    close = lambda a, b: abs(sr[a] - sr[b]) < 0.15 * max(sr[a], sr[b])  # noqa: E731
    out = []
    for perm in permutations(range(3)):
        if any(perm[i] != i and not close(i, perm[i]) for i in range(3)):
            continue
        P = np.eye(3)[:, list(perm)]
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            R = Rr @ np.diag(signs) @ P.T @ Rs.T
            if np.linalg.det(R) < 0:
                R = Rr @ np.diag(signs) @ np.diag([1, 1, -1]) @ P.T @ Rs.T
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = cr - R @ cs
            out.append((f"pca{''.join(map(str, perm))}{''.join('+' if x > 0 else '-' for x in signs)}", T))
    return out


def _same_pose(A: np.ndarray, B: np.ndarray, pts: np.ndarray, dist: float) -> bool:
    return float(np.max(np.linalg.norm(_apply(A, pts) - _apply(B, pts), axis=1))) < dist


def align_to_reference(cloud: o3d.geometry.PointCloud, ref: ReferenceSurface, p: CompareParams, spacing: float,
                       max_distance: float, log: Log = print) -> tuple[np.ndarray, dict]:
    pts = np.asarray(cloud.points)
    rng = np.random.default_rng(0)
    tight = 2.0 * spacing
    fine = 2.5 * spacing
    voxel = max(ref.diagonal / 60.0, spacing * 4)
    floor = max(2.0 * spacing, 2.0 * p.tolerance)
    fit_pts = pts if len(pts) <= 150_000 else pts[rng.choice(len(pts), 150_000, replace=False)]

    if p.align == "none":
        return np.eye(4), {"method": "none", "candidates": []}

    n_samples = int(np.clip(ref.areas.sum() / max(spacing, 1e-9) ** 2, 50_000, 400_000))
    s_pts, s_nrm = ref.sample(n_samples)
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s_pts))
    target.normals = o3d.utility.Vector3dVector(s_nrm)
    coarse = [t for t in (voxel * 2.0, voxel * 0.8) if t > fine * 1.5] or [fine * 2]
    if p.align == "icp":
        coarse = [voxel * 4.0] + coarse
    loose = max(voxel * 0.4, tight)

    candidates: list[tuple[str, np.ndarray]] = [("identity", np.eye(4))]
    if p.align == "auto":
        shift = np.eye(4)
        shift[:3, 3] = s_pts.mean(0) - pts.mean(0)
        candidates.append(("centroid", shift))
        down = cloud.voxel_down_sample(voxel / 2)
        candidates += _pca_candidates(np.asarray(down.points), np.asarray(target.voxel_down_sample(voxel / 2).points))
        log(f"  global feature matching (voxel {voxel:.4f})")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            candidates += global_candidates(cloud, target, voxel, 3)
    elif p.align != "icp":
        raise ValueError(f"Unknown alignment '{p.align}' (use auto, icp or none)")

    eval_pts = np.asarray(cloud.voxel_down_sample(spacing * 2).points)
    if len(eval_pts) > 60_000:
        eval_pts = eval_pts[rng.choice(len(eval_pts), 60_000, replace=False)]
    check_pts = eval_pts[rng.choice(len(eval_pts), min(2000, len(eval_pts)), replace=False)]

    def evaluate(T: np.ndarray, threshold: float) -> tuple[float, float]:
        d = np.abs(ref.query(_apply(T, eval_pts))[0])
        near = d[d < threshold * 2.5]
        return float((d < threshold).mean()), float(np.sqrt((near ** 2).mean())) if len(near) else float("inf")

    # Stage 1: coarse ICP from every start, scored at a loose distance.
    pool: list[dict] = []

    def coarse_candidate(name: str, T0: np.ndarray) -> None:
        T, _, _ = refine_icp(cloud, target, T0, coarse)
        if any(_same_pose(T, c["T"], check_pts, voxel) for c in pool):
            return
        score, _ = evaluate(T, loose)
        pool.append({"name": name, "T": T, "coarse": score})

    for name, T0 in candidates:
        coarse_candidate(name, T0)
    if p.align == "auto" and pool:
        # Also try the best pose turned 180 degrees about its principal axes (exposes symmetric parts).
        best = max(pool, key=lambda c: c["coarse"])
        c, R, _ = _principal_frame(_apply(best["T"], eval_pts))
        for k in range(3):
            F = np.eye(4)
            F[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(R[:, k] * np.pi)
            F[:3, 3] = c - F[:3, :3] @ c
            coarse_candidate(f"flip-axis{k + 1}", F @ best["T"])

    # Stage 2: exact surface fit of the most promising poses, ranked by the fraction of points within
    # 2x point spacing of the CAD surface (a loose overlap can prefer a wrong, e.g. flipped, pose).
    scored: list[dict] = []
    for cand in sorted(pool, key=lambda c: c["coarse"], reverse=True)[:6]:
        T, _ = fit_to_surface(ref, eval_pts, cand["T"], max(voxel, 2 * floor), floor, iterations=30)
        if any(_same_pose(T, s["T"], check_pts, fine * 2) for s in scored):
            continue
        score, rms = evaluate(T, tight)
        scored.append({"name": cand["name"], "score": score, "rmse": rms, "T": T})
        log(f"    candidate {cand['name']:<13} coarse {cand['coarse']:.3f}  within {tight:.4f}: {score:.3f}  "
            f"rmse {rms:.5f}")

    ranked = sorted(scored, key=lambda s: (round(s["score"], 4), -s["rmse"]), reverse=True)
    best = ranked[0]
    T, fit_info = fit_to_surface(ref, fit_pts, best["T"], max_distance, floor)
    report = {"method": p.align, "best_candidate": best["name"], "tight_threshold": tight,
              "fit_iterations": fit_info["iterations"], "robust_k": fit_info["k"], "candidates": []}
    alternatives = []
    for s in ranked:
        angle = rotation_angle(best["T"][:3, :3].T @ s["T"][:3, :3])
        entry = {"name": s["name"], "score": round(s["score"], 4), "rmse": round(s["rmse"], 6),
                 "angle_from_best": round(angle, 2)}
        if len(report["candidates"]) < 6:
            report["candidates"].append(entry)
        if s is not best and angle > 15 and all(rotation_angle(a["T"][:3, :3].T @ s["T"][:3, :3]) > 15
                                               for a in alternatives):
            alternatives.append({**entry, "T": s["T"]})
    if alt := _rival_pose(alternatives, best["score"], best["rmse"]):
        report["ambiguous"] = True
        # the move (in the reference frame) from the best pose to the rival one: when it maps the reference onto
        # itself the part is symmetric that way and the choice does not matter (golden.py checks this)
        report["rival_delta"] = (alt["T"] @ np.linalg.inv(best["T"])).round(10).tolist()
        report["warning"] = (f"a pose rotated {alt['angle_from_best']:.0f} deg ({alt['name']}) fits almost as well "
                             f"({alt['score']:.3f} vs {best['score']:.3f}) - the part may be symmetric; check "
                             "the alignment visually")
        log(f"  WARNING: {report['warning']}")
    return T, report


# --------------------------------------------------------------------------- main entry
def _histogram(values: np.ndarray, tolerance: float, bins: int = 41) -> dict:
    limit = 4.0 * tolerance if tolerance > 0 else max(float(np.abs(values).max()) if len(values) else 1.0, 1e-6)
    edges = np.linspace(-limit, limit, bins + 1)
    counts, _ = np.histogram(values, bins=edges)   # out-of-range values go to under/overflow
    return {"edges": edges.round(8).tolist(), "counts": counts.astype(int).tolist(),
            "underflow": int((values < -limit).sum()), "overflow": int((values > limit).sum())}


def compare_to_reference(scan: Geometry, reference_mesh: o3d.geometry.TriangleMesh,
                         params: CompareParams | None = None, log: Log = print) -> dict:
    """Align a scan to a reference mesh (CAD tessellation or STL) and measure signed deviations.

    Returns {"aligned": scan cloud in the CAD frame, "deviation": signed distance per aligned point
    (+ = material outside the CAD surface, NaN = excluded), "reference": reference mesh copy (outward
    normals, finely subdivided for display), "reference_distance": per vertex of that mesh, distance to the
    nearest included scan point, "report": dict, and for further analysis (golden.py): "surface" (the
    ReferenceSurface), "closest" (closest reference point per aligned point), "triangle" (its triangle)}."""
    p = params or CompareParams()
    if isinstance(reference_mesh, o3d.geometry.PointCloud) or len(reference_mesh.triangles) == 0:
        raise ValueError("The reference must be a mesh (CAD model or STL), not a point cloud")
    if p.tolerance <= 0:
        raise ValueError("tolerance must be positive")
    cloud = o3d.geometry.PointCloud(to_cloud(scan))
    if len(cloud.points) < 100:
        raise ValueError("The scan has too few points to compare")
    if 0 < p.sample_points < len(cloud.points):
        idx = np.sort(np.random.default_rng(0).choice(len(cloud.points), p.sample_points, replace=False))
        cloud = cloud.select_by_index(idx)
        log(f"Using a random subset of {p.sample_points:,} scan points")

    ref = ReferenceSurface(reference_mesh, log)
    spacing = estimate_spacing(cloud.points)
    max_distance = p.max_distance if p.max_distance > 0 else 0.03 * ref.diagonal
    log(f"Comparing {len(cloud.points):,} scan points with a {len(ref.triangles):,}-triangle reference "
        f"(diagonal {ref.diagonal:.3f}, scan spacing {spacing:.5f}, tolerance +/-{p.tolerance}, "
        f"max distance {max_distance:.4f}, alignment {p.align})")
    warnings: list[str] = []

    T, alignment = align_to_reference(cloud, ref, p, spacing, max_distance, log)
    aligned = o3d.geometry.PointCloud(cloud).transform(T)
    pts = np.asarray(aligned.points)
    signed, closest, _, triangle = ref.closest(pts)
    included = np.abs(signed) <= max_distance
    deviation = np.where(included, signed, np.nan)
    d = signed[included]
    n_excl = int((~included).sum())
    if len(d) == 0:
        raise ValueError(f"No scan point lies within {max_distance:.4f} of the reference - the alignment failed "
                         "or the scan and CAD use different units")
    tight = 2.0 * spacing
    alignment["fitness"] = round(float((np.abs(signed) < max(tight, p.tolerance)).mean()), 4)
    alignment["rmse"] = float(np.sqrt(np.mean(d ** 2)))
    if alignment.get("warning"):
        warnings.append(alignment["warning"])
    excl_pct = 100.0 * n_excl / len(pts)
    log(f"  aligned: {alignment['fitness'] * 100:.1f} % of points within {max(tight, p.tolerance):.4f}, "
        f"{n_excl:,} points ({excl_pct:.1f} %) farther than {max_distance:.4f} excluded")
    if excl_pct > 20:
        warnings.append(f"{excl_pct:.0f} % of the scan is farther than {max_distance:.3f} from the CAD model and was "
                        "excluded - check the alignment, or clean away table / fixture points")
    if alignment["fitness"] < 0.5 and p.align != "none":
        warnings.append("less than half of the scan lies close to the CAD surface - the alignment may be wrong")

    within = float((np.abs(d) <= p.tolerance).mean() * 100)
    stats = {"points": int(len(d)), "excluded": n_excl, "mean": float(d.mean()), "std": float(d.std()),
             "rms": float(np.sqrt((d ** 2).mean())), "abs_mean": float(np.abs(d).mean()),
             "min": float(d.min()), "max": float(d.max()), "p05": float(np.percentile(d, 5)),
             "p95": float(np.percentile(d, 95)), "within_tolerance_pct": within}

    # Scale: 7-DOF fit on the included points, reported only.
    scale_estimate = None
    if p.estimate_scale:
        sub = pts[included]
        if len(sub) > 150_000:
            sub = sub[np.random.default_rng(1).choice(len(sub), 150_000, replace=False)]
        _, sinfo = fit_to_surface(ref, sub, np.eye(4), max_distance, max(2.0 * spacing, 2.0 * p.tolerance),
                                  with_scale=True)
        factor = float(sinfo["scale"])
        # fit_to_surface scales the scan onto the CAD; the scan is 1/factor times the true size
        scan_scale = 1.0 / factor
        scale_estimate = {"factor": round(scan_scale, 6), "percent": round((scan_scale - 1) * 100, 4),
                          "applied": False}
        log(f"  scale estimate: scan is {scale_estimate['percent']:+.3f} % of CAD size (not applied)")
        if abs(scale_estimate["percent"]) > 0.2:
            warnings.append(f"the scan appears {scale_estimate['percent']:+.2f} % larger than the CAD model - check "
                            "the scanner calibration (the scale was NOT corrected)")

    # Coverage: which part of the CAD surface was captured.
    cov_thr = p.coverage_threshold if p.coverage_threshold > 0 else max(3.0 * spacing, 2.0 * p.tolerance)
    tree = cKDTree(pts[included])
    n_cov = int(np.clip(ref.areas.sum() / cov_thr ** 2, 20_000, 300_000))
    s_pts, _ = ref.sample(n_cov, seed=2)
    covered_pct = float((tree.query(s_pts, k=1, workers=-1)[0] <= cov_thr).mean() * 100)
    display = o3d.geometry.TriangleMesh(ref.mesh)
    target_edge = max(cov_thr * 4, ref.diagonal / 100)   # CAD tessellations have huge flat triangles
    for _ in range(6):
        V, F = np.asarray(display.vertices), np.asarray(display.triangles)
        longest = np.max(np.linalg.norm(V[F] - V[F[:, [1, 2, 0]]], axis=2))
        if longest <= target_edge or len(F) * 4 > 200_000:
            break
        display = display.subdivide_midpoint(1)     # planar split: surface geometry is unchanged
    display.compute_vertex_normals()
    reference_distance = tree.query(np.asarray(display.vertices), k=1, workers=-1)[0]
    coverage = {"covered_area_pct": covered_pct, "threshold": cov_thr}
    log(f"  coverage: {covered_pct:.1f} % of the CAD surface has scan points within {cov_thr:.4f}")

    ref_min, ref_max = ref.vertices.min(0), ref.vertices.max(0)
    # Isolated noise points near the part would dominate a min/max: measure extents on points that have
    # scan neighbours (a surface), not on stray ones.
    inc_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[included]))
    dense, _ = inc_cloud.remove_radius_outlier(nb_points=8, radius=4.0 * max(spacing, 1e-9))
    inc_pts = np.asarray(dense.points) if len(dense.points) >= 10 else pts[included]
    scan_min, scan_max = inc_pts.min(0), inc_pts.max(0)
    dimensions = {"reference": (ref_max - ref_min).tolist(), "scan": (scan_max - scan_min).tolist(),
                  "difference": ((scan_max - scan_min) - (ref_max - ref_min)).tolist(),
                  "reference_min": ref_min.tolist(), "reference_max": ref_max.tolist(),
                  "scan_min": scan_min.tolist(), "scan_max": scan_max.tolist()}
    if covered_pct < 90:
        warnings.append(f"only {covered_pct:.0f} % of the CAD surface was scanned - overall scan dimensions of "
                        "uncovered directions are not meaningful")

    report = {"transform": np.asarray(T).round(10).tolist(), "alignment": alignment, "stats": stats,
              "histogram": _histogram(d, p.tolerance), "dimensions": dimensions, "coverage": coverage,
              "scale_estimate": scale_estimate, "tolerance": p.tolerance, "max_distance": max_distance,
              "spacing": spacing, "reference": {"triangles": len(ref.triangles), "watertight": ref.watertight,
                                                "consistent_normals": ref.consistent, "diagonal": ref.diagonal},
              "warnings": warnings, "params": p.to_dict()}
    log(f"Deviation: mean {stats['mean']:+.4f}, std {stats['std']:.4f}, rms {stats['rms']:.4f}, "
        f"p05 {stats['p05']:+.4f}, p95 {stats['p95']:+.4f}; {within:.1f} % within +/-{p.tolerance}")
    for w in warnings:
        log(f"  WARNING: {w}")
    return {"aligned": aligned, "deviation": deviation, "reference": display,
            "reference_distance": reference_distance, "report": report,
            "surface": ref, "closest": closest, "triangle": triangle}
