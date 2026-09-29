"""Aligning and merging multiple scans (point clouds or meshes) of the same object."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .io import Geometry, estimate_spacing, is_cloud, to_cloud
from .params import ParamsMixin

Log = Callable[[str], None]
reg = o3d.pipelines.registration


@dataclass
class MergeParams(ParamsMixin):
    method: str = "auto"                # auto (global + ICP, stickers first) | markers (stickers only) | icp | none
    feature_voxel: float = 0.0          # voxel for feature matching; 0 -> auto (bbox diagonal / 60)
    ransac_trials: int = 4              # independent global registration attempts, best one wins
    icp_threshold_multiplier: float = 2.5   # final ICP correspondence distance, x point spacing
    refine_passes: int = 1              # extra passes aligning each scan to all others (3+ scans)
    dedupe_voxel: float = 0.0           # >0: thin overlap regions to this voxel size (0 keeps all points)
    final_sor: bool = True              # light outlier removal on the merged cloud
    min_fitness: float = 0.2            # warn when a scan overlaps less than this after alignment
    min_gain: float = 0.03              # assessment: a scan must add this fraction of new surface to be worth merging
    layer_ratio: float = 1.5            # assessment: overlap surfaces apart by more than this x noise = doubled skin
    layer_min_mm: float = 0.02          # assessment: ... and by more than this (closer is within scanner accuracy)


def rigid_from_pairs(source_pts, target_pts) -> np.ndarray:
    """Least squares rigid transform (Kabsch) mapping source points onto target points."""
    A = np.asarray(source_pts, dtype=float)
    B = np.asarray(target_pts, dtype=float)
    if A.shape != B.shape or len(A) < 3:
        raise ValueError("Need at least 3 matching point pairs")
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


def _transformed(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(pcd).transform(T)


def _down_with_normals(pcd, voxel: float) -> o3d.geometry.PointCloud:
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    return down


def _fpfh(pcd, voxel: float):
    down = _down_with_normals(pcd, voxel)
    feat = reg.compute_fpfh_feature(down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return down, feat


def global_candidates(src, tgt, voxel: float, trials: int) -> list[tuple[str, np.ndarray]]:
    s_down, s_feat = _fpfh(src, voxel)
    t_down, t_feat = _fpfh(tgt, voxel)
    dist = voxel * 1.5
    out = []
    for i in range(trials):
        o3d.utility.random.seed(1000 + i)
        res = reg.registration_ransac_based_on_feature_matching(
            s_down, t_down, s_feat, t_feat, True, dist,
            reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(dist)],
            reg.RANSACConvergenceCriteria(2_000_000, 0.9995))
        out.append((f"ransac#{i + 1}", np.asarray(res.transformation)))
    try:
        res = reg.registration_fgr_based_on_feature_matching(
            s_down, t_down, s_feat, t_feat,
            reg.FastGlobalRegistrationOption(maximum_correspondence_distance=dist))
        out.append(("fgr", np.asarray(res.transformation)))
    except Exception:
        pass
    return out


ICP_MAX_SOURCE = 300_000   # points per ICP level; enough for sub-0.05 degree poses, keeps dense scans fast
ICP_MAX_TARGET = 600_000
ICP_RANK_SOURCE = 80_000   # points used while ranking candidate poses
# The final fine level needs a target denser than the correspondence distance: a 600k subsample of a 6M point
# scan has ~3x the spacing and under-reports fitness (measured 0.17 instead of 0.27).
ICP_FINAL_TARGET = 4_000_000


def cap_points(pcd: o3d.geometry.PointCloud, max_points: int, seed: int = 0) -> o3d.geometry.PointCloud:
    """Deterministic uniform random subsample (keeps colours/normals). Spreads samples over the whole surface, unlike
    a coarser voxel grid, so the fine ICP level still sees the true surface position."""
    n = len(pcd.points)
    if n <= max_points:
        return pcd
    idx = np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))
    return pcd.select_by_index(idx)


def _icp_level(src, tgt, thr: float, cache: dict | None, max_source: int = ICP_MAX_SOURCE,
               max_target: int | None = None):
    """Downsampled + capped source and normal-estimated target for one ICP level (reused across candidate poses)."""
    if max_target is None:
        max_target = max(ICP_MAX_TARGET * max_source // ICP_MAX_SOURCE, 150_000)
    key = (id(src), id(tgt), round(thr, 12), max_source, max_target)
    if cache is not None and key in cache:
        return cache[key]
    voxel = thr / 2.5
    s = cap_points(src.voxel_down_sample(voxel), max_source)
    t = cap_points(tgt.voxel_down_sample(voxel), max_target, seed=1)
    t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel, thr / 2) * 2.5, max_nn=30))
    if cache is not None:
        cache[key] = (s, t)
    return s, t


def refine_icp(src, tgt, init: np.ndarray, thresholds: list[float], cache: dict | None = None,
               max_source: int = ICP_MAX_SOURCE, max_target: int | None = None):
    """Coarse-to-fine robust point-to-plane ICP. Returns (transform, fitness, rmse).

    Each level is capped at ICP_MAX_SOURCE / ICP_MAX_TARGET points: multi-million point scans otherwise spend minutes
    per candidate pose at the fine level (measured 10.7 s per ICP call on 1.2M points) for no accuracy gain. Pass the
    same `cache` dict when refining many candidate poses of the same pair."""
    T = np.asarray(init, dtype=float)
    fitness, rmse = 0.0, float("inf")
    for thr in thresholds:
        s, t = _icp_level(src, tgt, thr, cache, max_source, max_target)
        est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=thr))
        res = reg.registration_icp(s, t, thr, T, est,
                                   reg.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7,
                                                              max_iteration=60))
        T, fitness, rmse = np.asarray(res.transformation), res.fitness, res.inlier_rmse
    return T, fitness, rmse


def color_agreement(src, tgt, T: np.ndarray, threshold: float) -> float | None:
    """Fraction of overlapping points whose colours match (None when a scan has no colour)."""
    if not (src.has_colors() and tgt.has_colors()):
        return None
    s = _transformed(src.voxel_down_sample(threshold), T)
    t = tgt.voxel_down_sample(threshold / 2)
    dist, idx = cKDTree(np.asarray(t.points)).query(np.asarray(s.points), k=1, distance_upper_bound=threshold,
                                                    workers=-1)
    valid = np.isfinite(dist)
    if valid.sum() < 50:
        return 0.0
    diff = np.abs(np.asarray(s.colors)[valid] - np.asarray(t.colors)[idx[valid]]).mean(axis=1)
    return float((diff < 0.12).mean())


def _rival_pose(alternatives: list[dict], best_score: float, best_rmse: float) -> dict | None:
    """The first other pose that fits almost as well as the best one (a symmetric part), if any."""
    if best_score <= 0:
        return None
    return next((a for a in alternatives
                 if a["score"] >= 0.9 * best_score and a["rmse"] <= 1.25 * max(best_rmse, 1e-9)), None)


def align_pair(src, tgt, params: MergeParams, spacing: float, voxel: float,
               init: np.ndarray | None = None, log: Log = print) -> tuple[np.ndarray, dict]:
    fine = spacing * params.icp_threshold_multiplier
    coarse_levels = [t for t in (voxel * 2.0, voxel * 0.8) if t > fine * 1.5]
    if params.method == "none":
        return np.eye(4), {"method": "none"}

    candidates: list[tuple[str, np.ndarray]] = []
    # marker stickers both scans share fix the pose even when the part looks alike from several sides
    stickers: dict | None = None
    T_stickers = np.eye(4)
    if init is None and params.method in ("auto", "markers"):
        from .markers import sticker_pose

        try:
            T_stickers, stickers = sticker_pose(src, tgt, spacing, log)
            candidates.append(("stickers", T_stickers))  # evaluated first: other poses that converge to it drop
        except ValueError as exc:
            if params.method == "markers":
                raise
            log(f"    stickers: {str(exc).split('. ')[0]}")
    if params.method == "markers":
        pass
    elif init is not None:
        candidates.append(("manual pairs", init))
    else:
        candidates.append(("identity", np.eye(4)))
        shift = np.eye(4)
        shift[:3, 3] = tgt.get_center() - src.get_center()
        candidates.append(("centroid", shift))
        if params.method == "auto":
            candidates += global_candidates(src, tgt, voxel, params.ransac_trials)

    levels = (coarse_levels or []) + [fine * 2]
    tight = spacing * 1.5
    src_eval = cap_points(src.voxel_down_sample(spacing * 2), ICP_MAX_SOURCE)
    scored = []
    evaluated: list[np.ndarray] = []
    icp_cache: dict = {}

    def evaluate(name, T0):
        # Rank by how many points land within noise level of the other scan after refinement.
        # Coarse overlap is misleading: a wrong (e.g. flipped) pose can overlap more at a loose tolerance.
        # ranking candidate poses only needs a coarse answer; the winner is refined at full cap below
        # the stickers' pose is already exact: coarse passes could only slide it (round parts turn freely)
        T, fit, rmse = refine_icp(src, tgt, T0, [fine * 2] if name == "stickers" else levels, icp_cache,
                                  ICP_RANK_SOURCE)
        for prev in evaluated:
            dR = prev[:3, :3].T @ T[:3, :3]
            if np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)) < np.radians(1.0) and \
                    np.linalg.norm(prev[:3, 3] - T[:3, 3]) < fine * 2:
                log(f"    candidate {name:<13} converged to an earlier pose")
                return
        evaluated.append(T)
        tight_fit = reg.evaluate_registration(src_eval, tgt, tight, T).fitness
        color = color_agreement(src, tgt, T, fine)
        score = tight_fit if color is None else tight_fit * (0.3 + 0.7 * color)
        scored.append((round(score, 4), -rmse, name, T))
        color_txt = "" if color is None else f"  colour match {color:.2f}"
        log(f"    candidate {name:<13} overlap {fit:.3f}  tight {tight_fit:.3f}  rmse {rmse:.5f}{color_txt}")

    for name, T0 in candidates:
        evaluate(name, T0)
    if init is None and params.method == "auto":
        # Also try the best pose flipped 180 degrees about the scan's principal axes. This exposes
        # symmetric parts where a different orientation fits (almost) as well.
        best_T = max(scored, key=lambda x: (x[0], x[1]))[3]
        pts = np.asarray(_transformed(src.voxel_down_sample(voxel), best_T).points)
        c = pts.mean(axis=0)
        _, _, axes = np.linalg.svd(pts - c, full_matrices=False)
        for k, axis in enumerate(axes):
            F = np.eye(4)
            F[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)
            F[:3, 3] = c - F[:3, :3] @ c
            evaluate(f"flip-axis{k + 1}", F @ best_T)

    ranked = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)
    best_score, _, name, T = ranked[0]
    by_stickers = next((x for x in scored if x[2] == "stickers"), None)
    stickers_lost = False
    if by_stickers is not None and name != "stickers":
        # Overlap is the wrong yardstick here: scans of opposite sides (face up / face down) share little surface
        # at the right pose, while a wrong face-on-face pose of a flat or symmetric part overlaps a lot. The
        # stickers are trusted when enough of them agree on one pose and the surface agrees locally: refining from
        # their pose barely moves it, and where the scans do overlap they sit as close as at the best surface fit.
        centre = np.asarray(src.get_center())
        refined = by_stickers[3]
        moved = float(np.linalg.norm((refined[:3, :3] @ centre + refined[:3, 3])
                                     - (T_stickers[:3, :3] @ centre + T_stickers[:3, 3])))
        turned = float(np.degrees(np.arccos(np.clip((np.trace(T_stickers[:3, :3].T @ refined[:3, :3]) - 1) / 2,
                                                    -1.0, 1.0))))
        trusted = (stickers["common"] >= 4 and not stickers["ambiguous"] and moved < max(1.0, 4.0 * spacing)
                   and turned < 1.0 and -by_stickers[1] <= 1.5 * -ranked[0][1])
        if trusted or by_stickers[0] >= 0.9 * best_score:
            best_score, _, name, T = by_stickers
            ranked = [by_stickers] + [x for x in ranked if x is not by_stickers]
        else:
            stickers_lost = True
    alternatives = []
    best_rmse = -ranked[0][1]
    for score, neg_rmse, other_name, other_T in ranked[1:]:
        dR = T[:3, :3].T @ other_T[:3, :3]
        angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        if angle > 15 and all(np.degrees(np.arccos(np.clip((np.trace(a["T"][:3, :3].T @ other_T[:3, :3]) - 1) / 2, -1, 1))) > 15
                              for a in alternatives):
            alternatives.append({"candidate": other_name, "score": score, "angle_from_best": round(float(angle), 1),
                                 "rmse": round(float(-neg_rmse), 5), "T": other_T})
    T, fit, rmse = refine_icp(src, tgt, T, [fine * 2, fine], icp_cache, ICP_MAX_SOURCE, ICP_FINAL_TARGET)
    # report quality against the full-resolution target so it does not depend on any subsampling
    quality = reg.evaluate_registration(cap_points(src.voxel_down_sample(fine / 2.5), ICP_MAX_SOURCE), tgt, fine, T)
    fit, rmse = quality.fitness, quality.inlier_rmse
    info = {"method": params.method, "best_candidate": name, "fitness": round(float(fit), 4),
            "rmse": float(rmse), "transform": np.asarray(T).round(8).tolist(),
            "alternatives": [{k: (v.round(8).tolist() if k == "T" else v) for k, v in a.items()}
                             for a in alternatives[:3]]}
    if stickers is not None:
        info["stickers"] = {k: stickers[k] for k in ("common", "marker_rms_mm", "markers_moving",
                                                       "markers_reference", "ambiguous")}
        if stickers_lost:
            info["stickers"]["disagree"] = True
            log("    WARNING: the stickers' pose fits the surface clearly worse than the best pose - were stickers "
                "moved between the scans? The best surface fit is used.")
    if name == "stickers" and not (stickers or {}).get("ambiguous"):
        log(f"    lined up on {stickers['common']} common stickers (fit {stickers['marker_rms_mm']:.3f} mm)")
    elif rival := _rival_pose(alternatives, best_score, best_rmse):
        # "fits almost as well" = about as much surface lands on the other scan AND it lands about as tightly: the
        # other pose of a symmetric part fits the same surface just as closely, a wrong pose that merely overlaps
        # a similar amount sits clearly looser (seen at 1.5x the rmse), and would flip the verdict run to run
        info["ambiguous"] = True
        info["alternatives"].sort(key=lambda a: a["candidate"] != rival["candidate"])   # the rival first
        log(f"    WARNING: a pose rotated {rival['angle_from_best']:.0f} deg fits almost as well - the part "
            "may be symmetric. Check the merge and use manual point pairs if it is wrong.")
    color = color_agreement(src, tgt, T, fine)
    if color is not None:
        info["color_agreement"] = round(color, 4)
    if fit < params.min_fitness:
        info["warning"] = (f"low overlap/fitness {fit:.2f} - alignment may be wrong. "
                           "Use manual point pairs or scan with more overlap.")
        log(f"    WARNING: {info['warning']}")
    return T, info


def _parse_transforms(transforms, n: int) -> list[np.ndarray | None]:
    """[4x4 | None] * n from a list (or {index: 4x4}); index 0 must be identity or None."""
    if transforms is None:
        return [None] * n
    if isinstance(transforms, dict):
        items = {int(k): v for k, v in transforms.items()}
        transforms = [items.get(i) for i in range(n)]
    if len(transforms) != n:
        raise ValueError(f"transforms must have one entry per scan ({n}), got {len(transforms)}")
    out: list[np.ndarray | None] = []
    for i, T in enumerate(transforms):
        if T is None:
            out.append(None)
            continue
        M = np.asarray(T, dtype=float)
        if M.shape != (4, 4) or not np.all(np.isfinite(M)):
            raise ValueError(f"transform for scan {i + 1} must be a 4x4 matrix")
        R = M[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0 or \
                not np.allclose(M[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError(f"transform for scan {i + 1} is not a rigid transform")
        out.append(M)
    return out


def merge_geometries(geoms: list[Geometry], params: MergeParams | None = None,
                     pairs: dict | None = None, log: Log = print, transforms=None):
    """Align every scan to the first one and merge them into one dense point cloud.

    pairs: optional manual correspondences {scan_index: {"source": [[x,y,z],...],
    "target": [[x,y,z],...]}} where source points lie on that scan and target points on scan 0.
    transforms: optional precomputed scan -> scan 0 poses (e.g. from `assess_merge` that the user approved), a
    list with one 4x4 (or None) per scan. A scan with a transform is placed with exactly that pose: no global
    registration, no ICP and no refinement pass touches it, so the approved alignment is what gets merged
    (fitness/rmse are still measured and reported). Scans without one (None) are aligned as usual.
    Returns (merged_cloud, transforms, report).
    """
    p = params or MergeParams()
    if len(geoms) < 2:
        raise ValueError("Merging needs at least two scans")
    pairs = {int(k): v for k, v in (pairs or {}).items()}
    clouds = [o3d.geometry.PointCloud(to_cloud(g)) for g in geoms]
    n = len(clouds)
    given = _parse_transforms(transforms, n)
    spacing = float(np.median([estimate_spacing(c.points) for c in clouds]))
    diag = max(float(np.linalg.norm(c.get_max_bound() - c.get_min_bound())) for c in clouds)
    voxel = p.feature_voxel if p.feature_voxel > 0 else max(diag / 60.0, spacing * 4)
    log(f"Merging {len(clouds)} scans (spacing {spacing:.5f}, feature voxel {voxel:.4f}, method {p.method})")

    transforms = [np.eye(4) if given[0] is None else given[0] for _ in range(n)]
    scans = [{"index": 0, "reference": True, "points": len(clouds[0].points)}]
    reference = _transformed(clouds[0], transforms[0])
    for i in range(1, n):
        target = reference.voxel_down_sample(spacing) if len(reference.points) > 1_500_000 else reference
        if given[i] is not None:
            T = given[i]
            fine = spacing * p.icp_threshold_multiplier
            res = reg.evaluate_registration(clouds[i], target, fine, T)
            info = {"method": "approved", "best_candidate": "approved transform",
                    "fitness": round(float(res.fitness), 4), "rmse": float(res.inlier_rmse),
                    "transform": np.asarray(T).round(8).tolist()}
            log(f"  scan {i + 1}/{n}: using the approved transform (fitness {res.fitness:.3f}) - not re-aligned")
        else:
            log(f"  aligning scan {i + 1}/{n} ({len(clouds[i].points):,} points)")
            init = None
            if i in pairs:
                init = rigid_from_pairs(pairs[i]["source"], pairs[i]["target"])
            T, info = align_pair(clouds[i], target, p, spacing, voxel, init, log)
        transforms[i] = T
        info.update(index=i, points=len(clouds[i].points))
        scans.append(info)
        reference += _transformed(clouds[i], T)

    free = [i for i in range(1, n) if given[i] is None]
    if n > 2 and p.method != "none" and free:
        fine = spacing * p.icp_threshold_multiplier
        for rp in range(p.refine_passes):
            log(f"  global refinement pass {rp + 1}")
            for i in free:
                others = o3d.geometry.PointCloud()
                for j in range(n):
                    if j != i:
                        others += _transformed(clouds[j], transforms[j])
                others = others.voxel_down_sample(spacing)
                moved = _transformed(clouds[i], transforms[i])
                dT, fit, rmse = refine_icp(moved, others, np.eye(4), [fine * 2, fine], None, ICP_MAX_SOURCE,
                                           ICP_FINAL_TARGET)
                quality = reg.evaluate_registration(cap_points(moved.voxel_down_sample(fine / 2.5), ICP_MAX_SOURCE),
                                                    others, fine, dT)
                fit, rmse = quality.fitness, quality.inlier_rmse
                transforms[i] = dT @ transforms[i]
                scans[i].update(fitness=round(float(fit), 4), rmse=float(rmse),
                                transform=transforms[i].round(8).tolist())

    merged = o3d.geometry.PointCloud()
    for c, T in zip(clouds, transforms):
        merged += _transformed(c, T)
    n_raw = len(merged.points)
    if p.dedupe_voxel > 0:
        merged = merged.voxel_down_sample(p.dedupe_voxel)
    if p.final_sor:
        merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    log(f"Merge done: {n_raw:,} combined points -> {len(merged.points):,}")
    report = {"scans": scans, "spacing": spacing, "feature_voxel": voxel, "combined_points": n_raw,
              "output_points": len(merged.points), "params": p.to_dict()}
    return merged, transforms, report


# --------------------------------------------------------------------------- merge assessment
NOISE_NEIGHBORS = 12
AREA_PER_CELL = 1.0 / 1.5   # a surface crosses ~1.5 voxels per voxel face area on average (random orientation)
HALF_NORMAL_MEDIAN = 0.6745  # median |d| of zero-mean Gaussian distances = 0.6745 sigma
LAYER_MAX_ANGLE_DEG = 40.0   # layering counts only where both scans' local surfaces face within this angle


def _local_planes(points: np.ndarray, neighbors: np.ndarray):
    """Centroid, unit normal and smallest eigenvalue of each neighbourhood (S, k, 3)."""
    nb = points[neighbors]
    mean = nb.mean(axis=1)
    c = nb - mean[:, None, :]
    cov = np.einsum("ski,skj->sij", c, c) / nb.shape[1]
    values, vectors = np.linalg.eigh(cov)
    return mean, vectors[:, :, 0], np.clip(values[:, 0], 0.0, None)


def estimate_noise(points, sample: int = 5000, k: int = NOISE_NEIGHBORS, seed: int = 0) -> float:
    """Measurement noise (1 sigma, normal direction) from local plane-fit residuals of k-neighbourhoods.

    The median over many neighbourhoods is robust to edges and remaining outliers; curvature of fine-spaced scans
    adds little at this neighbourhood size."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < k + 3:
        return 0.0
    idx = np.random.default_rng(seed).choice(len(pts), size=min(sample, len(pts)), replace=False)
    _, nn = cKDTree(pts).query(pts[idx], k=k, workers=-1)
    _, _, smallest = _local_planes(pts, nn)
    return float(np.median(np.sqrt(smallest * k / (k - 3))))


def _cells(points: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(points / cell).astype(np.int64)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    return uniq, counts


def _surface_grid(points: np.ndarray, spacing: float) -> tuple[float, int, float]:
    """(cell size, occupied surface cells, min points per cell) for area estimates.

    Cells start at 2.5x the point spacing and grow until a typical occupied cell holds >= 4 points, so randomly
    sampled scans do not leave empty cells (false 'new surface'). Cells with only stray points are ignored."""
    cell = 2.5 * spacing
    counts = np.zeros(0)
    for _ in range(6):
        _, counts = _cells(points, cell)
        if not len(counts) or np.median(counts) >= 4:
            break
        cell *= 1.4
    if not len(counts):
        return cell, 0, 2.0
    minimum = max(2.0, 0.2 * float(np.median(counts)))
    return cell, int((counts >= minimum).sum()), minimum


def _rotation_deg(T: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))


def assess_pair(src: o3d.geometry.PointCloud, ref: o3d.geometry.PointCloud, params: MergeParams, spacing: float,
                voxel: float, noise_src: float | None = None, noise_ref: float | None = None,
                log: Log = print) -> tuple[np.ndarray, dict]:
    """Align `src` to `ref` and measure whether merging it is worthwhile and safe.

    Returns (transform, info) with fitness, rmse, rmse_over_spacing, ambiguous, alternative_angle, already_aligned,
    overlap, coverage_gain (fraction of the reference area), coverage_gain_pct, coverage_gain_mm2, layering_mm
    (median point-to-surface distance in the overlap, counted only where both scans face the same way -
    layering_facing_fraction of the overlap), layering_p90_mm, separation_ratio (layering / what noise
    alone explains), doubled_surface, noise_mm, confident, adds_coverage, recommendation (merge|use_best|ask)."""
    p = params
    noise_src = estimate_noise(src.points) if noise_src is None else noise_src
    noise_ref = estimate_noise(ref.points) if noise_ref is None else noise_ref
    T, align = align_pair(src, ref, p, spacing, voxel, None, log)
    stickers = align.get("stickers")
    tight = spacing * 1.5
    src_eval = cap_points(src.voxel_down_sample(spacing * 2), ICP_MAX_SOURCE)
    info: dict = {"method": align.get("method"), "best_candidate": align.get("best_candidate"),
                  "points": len(src.points), "noise_mm": round(noise_src, 6)}
    fine = spacing * p.icp_threshold_multiplier
    if "fitness" in align:
        fitness, rmse = float(align["fitness"]), float(align["rmse"])
    else:  # method none: measure at the given pose
        res = reg.evaluate_registration(src, ref, fine, T)
        fitness, rmse = float(res.fitness), float(res.inlier_rmse)
    fit_identity = reg.evaluate_registration(src_eval, ref, tight, np.eye(4)).fitness
    fit_aligned = reg.evaluate_registration(src_eval, ref, tight, T).fitness
    info.update(fitness=round(fitness, 4), rmse=round(rmse, 6),
                rmse_over_spacing=round(rmse / spacing, 3) if spacing > 0 else None,
                ambiguous=bool(align.get("ambiguous", False)), stickers=stickers,
                already_aligned=bool(fit_aligned > 0.05 and fit_identity >= 0.9 * fit_aligned
                                     and _rotation_deg(T) < 0.5 and np.linalg.norm(T[:3, 3]) < 3 * spacing))
    alts = align.get("alternatives") or []
    if info["ambiguous"] and alts:
        info["alternative_angle"] = alts[0].get("angle_from_best")

    # overlap and new coverage
    s_cloud = src.voxel_down_sample(spacing) if len(src.points) > 1_500_000 else src
    s_pts = np.asarray(_transformed(s_cloud, T).points)
    r_pts = np.asarray(ref.points)
    tree = cKDTree(r_pts)
    radius = 3.0 * spacing
    dist, _ = tree.query(s_pts, k=1, distance_upper_bound=radius, workers=-1)
    near = np.isfinite(dist)
    info["overlap"] = round(float(near.mean()) if len(s_pts) else 0.0, 4)
    cell, ref_cells, ref_min = _surface_grid(r_pts, spacing)
    new_pts = s_pts[~near]
    new_cells = 0
    if len(new_pts):
        _, all_counts = _cells(s_pts, cell)
        _, counts = _cells(new_pts, cell)
        new_cells = int((counts >= max(2.0, 0.2 * float(np.median(all_counts)))).sum())
    gain = new_cells / ref_cells if ref_cells else 0.0
    info.update(coverage_gain=round(gain, 4), coverage_gain_pct=round(100 * gain, 2),
                coverage_gain_mm2=round(new_cells * cell * cell * AREA_PER_CELL, 3),
                reference_area_mm2=round(ref_cells * cell * cell * AREA_PER_CELL, 3), cell_mm=round(cell, 6))

    # layering: distance of overlapping points to the reference surface vs the noise of both scans - only where both
    # scans see the surface facing the same way. Scans of a thread from its two ends see opposite flanks: a flank
    # point of one scan next to the crest seen by the other is not a doubled skin, only a surface the other missed.
    overlap_pts = s_pts[near]
    combined = float(np.hypot(noise_src, noise_ref))
    info["combined_noise_mm"] = round(combined, 6)
    facing = None
    if len(overlap_pts) >= 100 and len(r_pts) >= NOISE_NEIGHBORS and len(s_pts) >= NOISE_NEIGHBORS:
        if len(overlap_pts) > 20_000:
            overlap_pts = overlap_pts[np.random.default_rng(1).choice(len(overlap_pts), 20_000, replace=False)]
        _, nn = tree.query(overlap_pts, k=NOISE_NEIGHBORS, workers=-1)
        mean, normal, _ = _local_planes(r_pts, nn)
        _, nn_src = cKDTree(s_pts).query(overlap_pts, k=NOISE_NEIGHBORS, workers=-1)
        _, normal_src, _ = _local_planes(s_pts, nn_src)
        facing = np.abs(np.einsum("ij,ij->i", normal, normal_src)) >= np.cos(np.radians(LAYER_MAX_ANGLE_DEG))
        info["layering_facing_fraction"] = round(float(facing.mean()), 4)
    if facing is not None and facing.sum() >= 50:
        d = np.abs(np.einsum("ij,ij->i", overlap_pts[facing] - mean[facing], normal[facing]))
        layering = float(np.median(d))
        expected = HALF_NORMAL_MEDIAN * combined
        ratio = layering / expected if expected > 0 else float("inf")
        info.update(layering_mm=round(layering, 6), layering_p90_mm=round(float(np.percentile(d, 90)), 6),
                    separation_ratio=round(ratio, 3),
                    doubled_surface=bool(ratio > p.layer_ratio and layering > p.layer_min_mm))
    else:
        info.update(layering_mm=None, layering_p90_mm=None, separation_ratio=None, doubled_surface=False)

    info["confident"] = bool(fitness >= p.min_fitness and not info["ambiguous"])
    info["adds_coverage"] = bool(gain >= p.min_gain)
    if info["ambiguous"]:   # the coverage estimate depends on a pose that cannot be trusted
        info["recommendation"] = "ask"
    elif not info["adds_coverage"]:
        info["recommendation"] = "use_best"
    elif info["confident"] and not info["doubled_surface"]:
        info["recommendation"] = "merge"
    else:
        info["recommendation"] = "ask"
    return T, info


def scan_reasons(label: str, s: dict, p: MergeParams) -> list[str]:
    out = []
    if s.get("ambiguous") and not s["adds_coverage"]:
        out.append(f"{label} adds only {s['coverage_gain_pct']:.1f}% new surface at the best pose found, but that "
                   "pose is uncertain.")
    elif not s["adds_coverage"]:
        out.append(f"{label} adds only {s['coverage_gain_pct']:.1f}% new surface (below {100 * p.min_gain:g}%): it "
                   "covers the same area as the reference, so merging would only add noise.")
    else:
        out.append(f"{label} adds {s['coverage_gain_pct']:.1f}% new surface ({s['coverage_gain_mm2']:.0f} mm²).")
    st = s.get("stickers") or {}
    if st and not st.get("disagree"):
        out.append(f"{label} lined up on {st['common']} marker stickers both scans share (fit "
                   f"{st['marker_rms_mm']:.3f} mm).")
    elif st.get("disagree"):
        out.append(f"{label}: the {st['common']} common stickers point to a different pose than the surface - were "
                   "stickers moved between the scans?")
    if s.get("ambiguous"):
        angle = s.get("alternative_angle")
        rotated = f" rotated {angle:.0f}°" if isinstance(angle, (int, float)) else ""
        out.append(f"{label}: a pose{rotated} fits almost as well - the part may be symmetric, so the alignment "
                   "cannot be trusted without checking.")
    if s["fitness"] < p.min_fitness:
        out.append(f"{label}: alignment fitness {s['fitness']:.2f} is below {p.min_fitness:g} - too little overlap "
                   "to align reliably.")
    if s.get("doubled_surface"):
        out.append(f"{label}: where the scans overlap their surfaces are {s['layering_mm']:.3f} mm apart (median), "
                   f"{s['separation_ratio']:.1f}x what the scan noise explains - merging would create a doubled, "
                   "layered skin.")
    return out


def assess_merge(geoms: list[Geometry], params: MergeParams | dict | None = None, log: Log = print) -> dict:
    """Check whether scans should be merged before doing it.

    Each scan i >= 1 is aligned to the union of the scans accepted so far (scan 0 plus every earlier scan that
    aligned confidently without a doubled surface) and measured with `assess_pair`. Recommendation:
    "single" (one scan), "merge" (every extra scan adds >= min_gain new surface, aligns confidently, no doubled
    surface), "use_best" (no extra scan adds meaningful surface - keep the best one: lowest noise, then most
    points), "ask" (anything else: uncertain alignment, doubled surfaces or a mix of useful and duplicate scans).
    `transforms` (scan -> scan 0, 4x4 lists) can be passed to `merge_geometries(transforms=...)`."""
    p = params if isinstance(params, MergeParams) else MergeParams.from_dict(params or {})
    if not geoms:
        raise ValueError("No scans to assess")
    clouds = [o3d.geometry.PointCloud(to_cloud(g)) for g in geoms]
    n = len(clouds)
    spacings = [estimate_spacing(c.points) for c in clouds]
    spacing = float(np.median(spacings))
    noises = [estimate_noise(c.points) for c in clouds]
    scans: list[dict] = [{"index": 0, "reference": True, "points": len(clouds[0].points),
                          "noise_mm": round(noises[0], 6), "spacing": round(spacings[0], 6)}]
    transforms = [np.eye(4) for _ in range(n)]
    result = {"recommendation": "single", "reasons": [], "best_index": 0, "scans": scans,
              "transforms": [np.eye(4).tolist()], "spacing": round(spacing, 6),
              "params": {"method": p.method, "min_gain": p.min_gain, "min_fitness": p.min_fitness,
                         "layer_ratio": p.layer_ratio}}
    if n == 1:
        result["reasons"] = ["Only one scan - nothing to merge."]
        return result

    diag = max(float(np.linalg.norm(c.get_max_bound() - c.get_min_bound())) for c in clouds)
    voxel = p.feature_voxel if p.feature_voxel > 0 else max(diag / 60.0, spacing * 4)
    log(f"Assessing {n} scans before merging (spacing {spacing:.5f}, method {p.method})")
    accepted = [0]
    union = o3d.geometry.PointCloud(clouds[0])
    for i in range(1, n):
        log(f"  scan {i + 1}/{n} ({len(clouds[i].points):,} points) vs "
            + ("the reference" if accepted == [0] else f"scans {', '.join(str(a + 1) for a in accepted)}"))
        target = union.voxel_down_sample(spacing) if len(union.points) > 1_500_000 else union
        noise_ref = noises[0] if len(accepted) == 1 else float(np.median([noises[a] for a in accepted]))
        T, info = assess_pair(clouds[i], target, p, spacing, voxel, noises[i], noise_ref, log)
        info.update(index=i, spacing=round(spacings[i], 6), compared_to=list(accepted))
        transforms[i] = T
        scans.append(info)
        layering = "-" if info["layering_mm"] is None else f"{info['layering_mm']:.4f} mm "             f"(x{info['separation_ratio']:.2f} of noise)"
        log(f"    fitness {info['fitness']:.3f}, overlap {info['overlap']:.2f}, new surface "
            f"{info['coverage_gain_pct']:.1f}%, layering {layering}, noise {info['noise_mm']:.4f} mm "
            f"-> {info['recommendation']}")
        if info["confident"] and not info["doubled_surface"]:
            accepted.append(i)
            union += _transformed(clouds[i], T)

    extra = scans[1:]
    reasons: list[str] = []
    for s in extra:
        reasons += scan_reasons(f"Scan {s['index'] + 1}", s, p)
    if all(s["recommendation"] == "merge" for s in extra):
        recommendation = "merge"
        reasons.append("Every scan adds new surface and aligns confidently without doubled surfaces - merging is "
                       "recommended.")
    elif all(s["recommendation"] == "use_best" for s in extra):
        recommendation = "use_best"
    else:
        recommendation = "ask"
        if any(s["recommendation"] == "use_best" for s in extra):
            reasons.append("Some scans only repeat surface that is already covered; consider merging only the scans "
                           "that add coverage.")
        reasons.append("The merge is uncertain - check the alignment before merging.")

    # lowest noise; noise within 5 % of the lowest counts as equal, then the scan with more points wins
    ties = [k for k in range(n) if noises[k] <= min(noises) * 1.05]
    best = max(ties, key=lambda k: (len(clouds[k].points), -noises[k]))
    if recommendation == "use_best":
        reasons.append(f"Use scan {best + 1} alone (lowest noise {noises[best]:.4f} mm, "
                       f"{len(clouds[best].points):,} points) instead of merging.")
    result.update(recommendation=recommendation, reasons=reasons, best_index=best,
                  transforms=[np.asarray(T).round(10).tolist() for T in transforms])
    log(f"Assessment: {recommendation}" + (f" (best scan {best + 1})" if recommendation == "use_best" else ""))
    return result


SUMMARY_SCAN_KEYS = ("index", "points", "noise_mm", "fitness", "rmse_over_spacing", "overlap", "coverage_gain_pct",
                     "coverage_gain_mm2", "layering_mm", "separation_ratio", "doubled_surface", "ambiguous",
                     "alternative_angle", "already_aligned", "recommendation", "stickers")


def assessment_summary(assessment: dict, names: list[str] | None = None) -> dict:
    """Compact assessment without transforms (for history entries, messages and the assistant)."""
    scans = []
    for s in assessment.get("scans", []):
        item = {k: s[k] for k in SUMMARY_SCAN_KEYS if k in s and s[k] is not None}
        if names and s.get("index", 0) < len(names):
            item["name"] = names[s["index"]]
        scans.append(item)
    return {"recommendation": assessment.get("recommendation"), "best_index": assessment.get("best_index"),
            "reasons": list(assessment.get("reasons", [])), "scans": scans}


def merge_caveat(report: dict) -> dict | None:
    """What a measurement taken on a merge (or anything made from it) should warn about, or None.

    Reads the pre-merge assessment the user saw (`report["assessment"]`, a summary) and the merge's own alignment
    flags. A pose that is ambiguous by one symmetry step (a 12-sided head, a thread) moves one scan's features
    along the axis, and scans that disagree leave a doubled skin whose outer layer is what extents measure
    (docs/accuracy-investigation-2026-09-24.md)."""
    scans = [s for s in (report.get("assessment") or {}).get("scans", []) if s.get("index", 0) > 0]
    scans += [s for s in report.get("scans", []) if s.get("index", 0) > 0 and s.get("ambiguous")]
    ambiguous = [s for s in scans if s.get("ambiguous")]
    doubled = [s for s in scans if s.get("doubled_surface")]
    if not ambiguous and not doubled:
        return None
    layering = max((s.get("layering_mm") or 0.0 for s in doubled), default=0.0)
    angles = [abs(float(s["alternative_angle"])) for s in ambiguous if isinstance(s.get("alternative_angle"), (int, float))]
    parts = []
    if ambiguous:
        parts.append("the scans could be aligned almost equally well " + (f"{min(angles):.0f}° apart" if angles else
                     "in another pose") + ", so features seen by only one scan may be shifted")
    if doubled:
        parts.append(f"where the scans overlap their surfaces are {layering:.2f} mm apart, so sizes across the "
                     "overlap can read up to that much too large")
    sentence = ("Measured on a merge that the pre-merge check flagged: " + "; ".join(parts) + ". Check a key "
                "dimension with calipers, or measure features on the single scan that saw them.")
    return {"ambiguous": bool(ambiguous), "alternative_angle": min(angles) if angles else None,
            "doubled_surface": bool(doubled), "layering_mm": round(layering, 4) if doubled else None,
            "sentence": sentence}


def transform_meshes(geoms: list[Geometry], transforms: list[np.ndarray]) -> o3d.geometry.TriangleMesh:
    """Apply merge transforms to mesh inputs and concatenate them (no remeshing)."""
    out = o3d.geometry.TriangleMesh()
    for g, T in zip(geoms, transforms):
        if is_cloud(g):
            continue
        out += o3d.geometry.TriangleMesh(g).transform(T)
    return out
