"""Fill the parts a scan missed with points from photos of the same part (docs/photo-fill.md).

The photo model is first lined up with the scan (photo_align.align_photo_model), then fitted onto the scan once more
where both show the part (scale, turn and shift, refine_onto_scan): the dense photo points can be a few percent off
their own cameras. Photo points that lie where the scan has nothing are then added, marked as coming from photos.

Photo surfaces are far less accurate than the scanner (about 1-2 mm against a few hundredths), so filled areas make
the model complete; they are never for measuring. The report says how far the photo surface sat from the scan where
both exist, and the golden model check leaves photo points out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .params import ParamsMixin

Log = Callable[[str], None]


@dataclass
class FillParams(ParamsMixin):
    gap: float = 0.0           # photo points farther than this from every scan point fill a gap (mm); 0: automatic
    margin_pct: float = 5.0    # only inside the scan's box grown by this share of its size
    min_points: int = 30       # smaller groups of photo points are left out as stray


def refine_onto_scan(photo_pts: np.ndarray, scan_pts: np.ndarray, spacing: float, log: Log = print) -> tuple[np.ndarray, dict]:
    """Fit the (roughly lined up) photo points onto the scan: scale, turn and shift, on the points near the scan.
    Returns (4x4 similarity, {scale, rmse_mm, fitness})."""
    size = float(np.linalg.norm(scan_pts.max(0) - scan_pts.min(0)))
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(photo_pts))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts))
    voxel = max(2.0 * spacing, size / 400)
    src_d, tgt_d = src.voxel_down_sample(voxel), tgt.voxel_down_sample(voxel)
    T = np.eye(4)
    info = {}
    for reach in (0.05 * size, 0.02 * size, 0.008 * size):
        reg = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d, reach, T, o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = reg.transformation
        info = {"fitness": float(reg.fitness), "rmse_mm": float(reg.inlier_rmse), "reach_mm": float(reach)}
    info["scale"] = float(np.cbrt(np.linalg.det(T[:3, :3])))
    log(f"  photo points fitted onto the scan: scale {info['scale']:.4f}, {info['fitness'] * 100:.0f} % within "
        f"{info['reach_mm']:.2f} mm, rms {info['rmse_mm']:.3f} mm")
    return T, info


def fill_gaps(scan_pts: np.ndarray, photo_pts: np.ndarray, spacing: float, params: FillParams | None = None,
              log: Log = print) -> dict:
    """Add the photo points (already in the scan's frame) that lie where the scan has nothing.

    Returns {"points": scan points then the added ones, "source": 0 per scan point, 1 per photo point,
    "added_index": which of photo_pts were added, "report": {added, gap_mm, photo_vs_scan_mm {median, p90},
    filled_area_mm2, left_out {...}, warnings}}."""
    p = params or FillParams()
    scan_pts = np.asarray(scan_pts, float)
    photo_pts = np.asarray(photo_pts, float)
    valid = np.flatnonzero(np.all(np.isfinite(photo_pts), axis=1))
    photo_pts = photo_pts[valid]
    lo, hi = scan_pts.min(0), scan_pts.max(0)
    size = float(np.linalg.norm(hi - lo))
    tree = cKDTree(scan_pts)
    d, _ = tree.query(photo_pts, k=1, workers=-1)

    # how far the photo surface sits from the scan where both have the part: the photos' own accuracy here
    on_scan = d < min(3.0, 0.03 * size)
    med = float(np.median(d[on_scan])) if on_scan.any() else float("nan")
    p90 = float(np.percentile(d[on_scan], 90)) if on_scan.any() else float("nan")
    gap = p.gap if p.gap > 0 else max(4.0 * spacing, 2.0 * (p90 if np.isfinite(p90) else 0.0), 0.5)

    # the table (or the scale sheet) under the part: a flat surface much bigger than the part, and what lies below it.
    # A part usually stands on it with its unscanned bottom touching it, so it would read as a missing face.
    table = np.zeros(len(photo_pts), dtype=bool)
    rest = np.flatnonzero(d > gap)
    centre = scan_pts.mean(0)
    for _ in range(3):
        if len(rest) < 100:
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(photo_pts[rest]))
        plane, inl = pc.segment_plane(max(gap / 2, 0.003 * size), 3, 400)
        inl = np.asarray(inl)
        n, off = np.asarray(plane[:3]), plane[3]
        flat = photo_pts[rest[inl]]
        u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        spread = max(np.ptp(flat @ u), np.ptp(flat @ np.cross(n, u)))
        if len(inl) < 0.05 * len(rest) or spread < 1.3 * float(np.max(hi - lo)):
            break
        side = np.sign(centre @ n + off) or 1.0      # the part is on this side of the table
        thr = max(gap / 2, 0.003 * size)
        if np.mean(side * (scan_pts @ n + off) < -thr) > 0.02:
            break                                   # the part reaches through it: not what it stands on
        below = side * (photo_pts @ n + off) < thr
        table |= below
        rest = rest[~below[rest]]
    grow = (hi - lo) * p.margin_pct / 100 + gap
    inside = np.all((photo_pts >= lo - grow) & (photo_pts <= hi + grow), axis=1)
    cand_index = np.flatnonzero((d > gap) & inside & ~table)
    cand = photo_pts[cand_index]
    left_out = {"table": int((table & (d > gap)).sum()), "outside_the_part": int(((d > gap) & ~inside & ~table).sum())}
    added, added_index = np.empty((0, 3)), np.empty(0, dtype=np.int64)
    if len(cand):
        photo_spacing = float(np.median(cKDTree(cand).query(cand, k=2, workers=-1)[0][:, 1])) if len(cand) > 2 else gap
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand))
        labels = np.asarray(pc.cluster_dbscan(eps=max(3.0 * photo_spacing, gap / 2), min_points=4))
        keep = np.zeros(len(cand), dtype=bool)
        stray = far = 0
        for lab in np.unique(labels[labels >= 0]):
            idx = np.flatnonzero(labels == lab)
            if len(idx) < p.min_points:
                stray += len(idx)
                continue
            # a group only fills a gap when it reaches the scan: groups floating away from it are clutter
            if tree.query(cand[idx], k=1, workers=-1)[0].min() > 3.0 * gap:
                far += len(idx)
                continue
            keep[idx] = True
        stray += int((labels < 0).sum())
        added, added_index = cand[keep], valid[cand_index[keep]]
        left_out.update(stray=stray, away_from_the_scan=far)
    area = 0.0
    if len(added) and on_scan.sum() > 50:
        # the photos' points per mm² where the scan (a thin surface, so its cells count area well) also has the part
        cell = max(4.0 * spacing, gap / 2)
        near = np.isfinite(cKDTree(photo_pts[on_scan]).query(scan_pts, k=1, distance_upper_bound=gap, workers=-1)[0])
        seen = len(np.unique(np.floor(scan_pts[near] / cell).astype(np.int64), axis=0)) * cell * cell
        if seen > 0:
            area = float(len(added) / (on_scan.sum() / seen))
    warnings = []
    if np.isfinite(med) and med > 0.5:
        warnings.append(f"The photo surface sits about {med:.1f} mm from the scan where both have the part (90 % "
                        f"within {p90:.1f} mm): the filled areas are only that accurate. Do not measure on them.")
    if not len(added):
        warnings.append("The photos show nothing the scan is missing (or only bits smaller than "
                        f"{gap:.1f} mm, which photos cannot fill).")
    log(f"  filled from photos: {len(added):,} points over about {area:.0f} mm² (gaps wider than {gap:.2f} mm); "
        f"photo surface vs scan: median {med:.3f} mm, 90 % within {p90:.3f} mm")
    report = {"added": int(len(added)), "gap_mm": round(gap, 4), "filled_area_mm2": round(area, 1),
              "photo_vs_scan_mm": {"median": round(med, 4) if np.isfinite(med) else None,
                                   "p90": round(p90, 4) if np.isfinite(p90) else None},
              "left_out": left_out, "warnings": warnings}
    points = np.vstack([scan_pts, added])
    source = np.r_[np.zeros(len(scan_pts)), np.ones(len(added))]
    return {"points": points, "source": source, "added_index": added_index, "report": report}
