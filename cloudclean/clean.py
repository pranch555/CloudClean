"""Point cloud cleaning: outliers, floating debris, support planes, normals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .io import estimate_spacing
from .params import ParamsMixin

Log = Callable[[str], None]


@dataclass
class CleanParams(ParamsMixin):
    # Distances marked "multiplier" are relative to the measured point spacing, so the same
    # settings work regardless of units (mm / m) or scan resolution.
    scale: float = 1.0                    # multiply coordinates (unit conversion, e.g. 1000 for m -> mm)
    crop_min: list | None = None          # [x, y, z] axis aligned crop box (scan units)
    crop_max: list | None = None
    dedupe: bool = True                   # drop exact duplicate points
    voxel_size: float = 0.0               # >0: downsample to this voxel size (0 keeps full density)
    remove_plane: bool = False            # remove the table / turntable plane under the object
    plane_distance_multiplier: float = 3.0
    plane_min_fraction: float = 0.05
    sor: bool = True                      # statistical outlier removal
    sor_neighbors: int = 24
    sor_std_ratio: float = 2.0
    radius_filter: bool = True            # remove points with too few neighbours
    radius_multiplier: float = 5.0
    radius_min_neighbors: int = 6
    cluster: bool = True                  # remove disconnected floating clusters
    cluster_eps_multiplier: float = 8.0
    cluster_keep_ratio: float = 0.05      # keep clusters >= this fraction of the largest one
    max_filter_removal: float = 0.35      # safety: skip a noise filter that would remove more than this
    normals: bool = True
    normal_neighbors: int = 30
    recompute_normals: bool = False       # ignore normals stored in the file

    PRESETS = {
        "light": dict(sor_std_ratio=3.0, radius_filter=False, cluster_keep_ratio=0.01),
        "standard": {},
        "aggressive": dict(sor_neighbors=32, sor_std_ratio=1.5, radius_multiplier=4.0,
                           radius_min_neighbors=10, cluster_keep_ratio=0.25, max_filter_removal=0.5),
    }

    @classmethod
    def preset(cls, name: str) -> "CleanParams":
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {', '.join(cls.PRESETS)}")
        return cls.from_dict(cls.PRESETS[name])


# --------------------------------------------------------------------------- normals
def orient_normals(pcd: o3d.geometry.PointCloud, log: Log = print, sample: int = 60_000) -> None:
    """Make normals consistent (MST propagation on a subsample) and point outward."""
    pts = np.asarray(pcd.points)
    n = len(pts)
    rng = np.random.default_rng(0)
    idx = np.arange(n) if n <= sample else np.sort(rng.choice(n, sample, replace=False))
    sub = pcd.select_by_index(idx)
    try:
        sub.orient_normals_consistent_tangent_plane(15)
        ref_pts, ref_normals = np.asarray(sub.points), np.asarray(sub.normals)
        normals = np.asarray(pcd.normals).copy()
        _, nn = cKDTree(ref_pts).query(pts, k=1, workers=-1)
        flip = np.einsum("ij,ij->i", normals, ref_normals[nn]) < 0
        normals[flip] *= -1
    except Exception as exc:  # pragma: no cover - depends on Open3D build
        log(f"  normal propagation failed ({exc}); using centroid orientation")
        normals = np.asarray(pcd.normals).copy()
    centroid = pts.mean(axis=0)
    if np.mean(np.einsum("ij,ij->i", pts - centroid, normals)) < 0:
        normals *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)


def ensure_normals(pcd: o3d.geometry.PointCloud, spacing: float, neighbors: int = 30,
                   recompute: bool = False, log: Log = print) -> None:
    if pcd.has_normals() and not recompute:
        pcd.normalize_normals()
        return
    log("  estimating normals")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(spacing, 1e-9) * 6, max_nn=neighbors))
    orient_normals(pcd, log)


# --------------------------------------------------------------------------- filters
def remove_support_plane(pcd, distance: float, min_fraction: float, log: Log = print):
    """Remove the largest plane only if it looks like a table/turntable the object sits on."""
    n = len(pcd.points)
    model, inliers = pcd.segment_plane(distance_threshold=distance, ransac_n=3, num_iterations=3000)
    inliers = np.asarray(inliers)
    if len(inliers) < min_fraction * n:
        log(f"  plane: largest plane has only {len(inliers) / n:.1%} of points - skipped")
        return pcd
    normal = np.asarray(model[:3], dtype=float)
    s = np.linalg.norm(normal)
    normal, d = normal / s, model[3] / s
    pts = np.asarray(pcd.points)
    signed = pts @ normal + d
    rest = np.ones(n, bool)
    rest[inliers] = False
    above = np.count_nonzero(signed[rest] > distance)
    below = np.count_nonzero(signed[rest] < -distance)
    if above + below < 0.02 * n:
        log("  plane: cloud is almost entirely planar - skipped")
        return pcd
    side = 1.0 if above >= below else -1.0
    if max(above, below) / (above + below) < 0.9:
        log("  plane: plane cuts through the object - not a support surface, skipped")
        return pcd

    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    def footprint(P):
        q = np.c_[P @ u, P @ v]
        lo, hi = np.percentile(q, [1, 99], axis=0)
        return float(np.prod(hi - lo))

    keep = side * signed > distance
    if footprint(pts[inliers]) < 1.3 * footprint(pts[keep]):
        log("  plane: plane is not wider than the object - probably part of the object, skipped")
        return pcd
    return pcd.select_by_index(np.flatnonzero(keep))


def keep_main_clusters(pcd, eps: float, keep_ratio: float, log: Log = print):
    """Connected-component filter. DBSCAN runs on a voxel grid, labels map back to all points."""
    pts = np.asarray(pcd.points)
    down = pcd.voxel_down_sample(eps / 3.0)
    labels = np.asarray(down.cluster_dbscan(eps=eps, min_points=3, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        log("  clusters: none found - skipped")
        return pcd
    _, nn = cKDTree(np.asarray(down.points)).query(pts, k=1, workers=-1)
    point_labels = labels[nn]
    counts = np.bincount(point_labels[point_labels >= 0])
    keep_labels = np.flatnonzero(counts >= keep_ratio * counts.max())
    log(f"  clusters: {len(counts)} found, keeping {len(keep_labels)}")
    return pcd.select_by_index(np.flatnonzero(np.isin(point_labels, keep_labels)))


# --------------------------------------------------------------------------- pipeline
def clean_point_cloud(pcd: o3d.geometry.PointCloud, params: CleanParams | None = None,
                      log: Log = print) -> tuple[o3d.geometry.PointCloud, dict]:
    p = params or CleanParams()
    pcd = o3d.geometry.PointCloud(pcd)
    steps: list[dict] = []
    n0 = len(pcd.points)
    log(f"Cleaning {n0:,} points")

    def record(name: str, before: int):
        after = len(pcd.points)
        steps.append({"step": name, "before": before, "after": after, "removed": before - after})
        log(f"  {name}: {before:,} -> {after:,} (removed {before - after:,})")

    before = len(pcd.points)
    pcd.remove_non_finite_points()
    if p.dedupe:
        pcd.remove_duplicated_points()
    record("sanitize", before)

    if p.scale != 1.0:
        pcd.scale(p.scale, center=np.zeros(3))
        log(f"  scaled by {p.scale}")

    if p.crop_min is not None or p.crop_max is not None:
        pts = np.asarray(pcd.points)
        lo = np.asarray(p.crop_min if p.crop_min is not None else pts.min(0) - 1, dtype=float)
        hi = np.asarray(p.crop_max if p.crop_max is not None else pts.max(0) + 1, dtype=float)
        before = len(pcd.points)
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
        record("crop", before)

    if len(pcd.points) < 50:
        raise ValueError("Fewer than 50 points left - check crop / scale settings")

    spacing = estimate_spacing(pcd.points)
    log(f"  point spacing: {spacing:.5f}")

    if p.voxel_size > 0:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(p.voxel_size)
        spacing = max(spacing, estimate_spacing(pcd.points))
        record("voxel downsample", before)

    if p.remove_plane:
        before = len(pcd.points)
        pcd = remove_support_plane(pcd, spacing * p.plane_distance_multiplier, p.plane_min_fraction, log)
        record("support plane", before)

    def guarded(name: str, result: o3d.geometry.PointCloud):
        nonlocal pcd
        before = len(pcd.points)
        removed = 1 - len(result.points) / before
        if removed > p.max_filter_removal:
            log(f"  {name}: would remove {removed:.0%} of points (> {p.max_filter_removal:.0%}) - skipped, "
                f"loosen its settings or raise max_filter_removal")
            return
        pcd = result
        record(name, before)

    if p.sor:
        result, _ = pcd.remove_statistical_outlier(nb_neighbors=p.sor_neighbors, std_ratio=p.sor_std_ratio)
        guarded("statistical outliers", result)

    if p.radius_filter:
        result, _ = pcd.remove_radius_outlier(nb_points=p.radius_min_neighbors,
                                              radius=spacing * p.radius_multiplier)
        guarded("sparse points", result)

    if p.cluster:
        before = len(pcd.points)
        pcd = keep_main_clusters(pcd, spacing * p.cluster_eps_multiplier, p.cluster_keep_ratio, log)
        record("floating clusters", before)

    if p.normals:
        ensure_normals(pcd, spacing, p.normal_neighbors, p.recompute_normals, log)

    report = {"input_points": n0, "output_points": len(pcd.points), "removed": n0 - len(pcd.points),
              "removed_fraction": round(1 - len(pcd.points) / max(n0, 1), 4), "spacing": spacing,
              "steps": steps, "params": p.to_dict()}
    log(f"Clean done: kept {len(pcd.points):,} of {n0:,} points ({report['removed_fraction']:.1%} removed)")
    return pcd, report
