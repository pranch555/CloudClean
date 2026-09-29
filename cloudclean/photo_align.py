"""Lining a model made from photos up with the owner's scan or mesh (job colour_from_photos).

A photo reconstruction has its own frame and an arbitrary scale (easily 25x off), and it also holds the table, the
patterned sheet and the background. `align_photo_model` finds the part in it (the cluster where the cameras look,
above the table the cameras look down on), guesses the scale from the part's size, tries every orientation, and then
refines scale, rotation and shift together with ICP: a similarity x_model = s R x_photo + t. The dense points are
only good to ~2 % and may be scaled a little against the cameras, so the final fit uses the reconstruction's own
triangulated points on the part (COLMAP's sparse.ply), which agree exactly with the cameras.

`camera_to_model` moves a photo camera into the model's frame with that similarity, so the photos can be projected
onto the model. Projection does not change when the whole scene is scaled, so only the camera's position scales.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .io import estimate_spacing, is_cloud
from .register import estimate_noise, global_candidates

Log = Callable[[str], None]

FIT_FAIL = 0.4         # below this share of the photo points on the model: the photos do not line up (refused)
PARTIAL_FIT_FAIL = 0.15  # ... when the model is one scan of an item the photos show more of (merging)
COVER_FAIL = 0.2       # ... or below this share of the model covered by the photo part
FIT_TRUST = 0.6        # trusted alignment: at least this share of the photo points on the model ...
COVER_TRUST = 0.35     # ... and at least this share of the model covered (the underside is never photographed)
DENSE_ERROR = 0.02     # the dense photo model is only good to ~2 % of the part size (measured: 1.8 mm on 97 mm)
CAMERA_POINT_ERROR = 0.004   # the reconstruction's own points agree with its cameras to well under this x size
MIN_CAMERA_POINTS = 30       # fewer on the part: the final fit falls back to the dense model (lower trust)
RIVAL_SHARE = 0.9      # another pose (> RIVAL_ANGLE away) fitting this well: the part may be symmetric
RIVAL_ANGLE = 15.0
GRID_ROTATIONS = 240   # orientations tried (super-Fibonacci grid, ~20 deg apart) besides the part-axes guesses
MAX_TARGET = 400_000   # model points used for fitting and checking (random subsample of bigger scans)
MAX_FINE = 20_000      # photo part points in the fine ICP
MAX_EVAL = 60_000      # ... when measuring the fit


# --------------------------------------------------------------------------- similarity transforms
def similarity(scale: float, R: np.ndarray, t) -> np.ndarray:
    S = np.eye(4)
    S[:3, :3] = scale * np.asarray(R, float)
    S[:3, 3] = np.asarray(t, float)
    return S


def split_similarity(S: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(scale, rotation, shift) of x -> s R x + t."""
    S = np.asarray(S, float)
    s = float(np.cbrt(np.linalg.det(S[:3, :3])))
    return s, S[:3, :3] / s, S[:3, 3].copy()


def camera_to_model(world_to_camera: np.ndarray, S: np.ndarray) -> np.ndarray:
    """A camera's world_to_camera [Rw|tw] (photo frame) in the model frame of x_model = s R x_photo + t:
    [Rw R^T | s tw - Rw R^T t]. The camera coordinates come out multiplied by s (model units); the image does not
    change because projection divides by depth."""
    s, R, t = split_similarity(S)
    W = np.asarray(world_to_camera, float)
    Rm = W[:3, :3] @ R.T
    out = np.eye(4)
    out[:3, :3] = Rm
    out[:3, 3] = s * W[:3, 3] - Rm @ t
    return out


def _apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def _query(tree: cKDTree, X: np.ndarray, **kw):
    """Nearest neighbours; threads only pay off for big queries (small ones spend most time starting them)."""
    return tree.query(X, workers=-1 if len(X) > 50_000 else 1, **kw)


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1.0, 1.0))))


def _rotvec(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * Kx + (1 - math.cos(theta)) * Kx @ Kx


def rotation_grid(n: int) -> np.ndarray:
    """n rotations spread evenly over all orientations (super-Fibonacci spirals, Alexa 2022)."""
    s = np.arange(n) + 0.5
    r, R = np.sqrt(s / n), np.sqrt(1.0 - s / n)
    a, b = 2 * np.pi * s / math.sqrt(2.0), 2 * np.pi * s / 1.533751168755204288118041
    x, y, z, w = r * np.sin(a), r * np.cos(a), R * np.sin(b), R * np.cos(b)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


# --------------------------------------------------------------------------- finding the part in the photo model
def camera_centres_axes(world_to_camera: list) -> tuple[np.ndarray, np.ndarray]:
    """Camera centres and viewing directions (optical axes) in the world."""
    W = np.asarray(world_to_camera, float).reshape(-1, 4, 4)
    R, t = W[:, :3, :3], W[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t), R[:, 2, :].copy()


def look_at_point(centres: np.ndarray, axes: np.ndarray, points: np.ndarray | None = None) -> np.ndarray:
    """Least-squares point closest to every camera's optical axis: where the photographer aimed. With (nearly)
    parallel axes the depth is taken from the points in front of the cameras."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in zip(centres, axes):
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ c
    w = np.linalg.eigvalsh(A)
    if len(centres) >= 2 and w[0] > 0.02 * w[-1]:
        return np.linalg.solve(A, b)
    c, d = centres.mean(0), axes.mean(0) / max(np.linalg.norm(axes.mean(0)), 1e-12)
    depth = 1.0
    if points is not None and len(points):
        z = (np.asarray(points) - c) @ d
        depth = float(np.median(z[z > 0])) if (z > 0).any() else 1.0
    return c + depth * d


def find_part(points: np.ndarray, centres: np.ndarray, axes: np.ndarray, log: Log = print) -> tuple[np.ndarray, dict]:
    """Indices of the photographed part in the photo model: the points around the look-at point, without the planes
    under / behind it that the cameras look at (table, sheet, wall), keeping the cluster(s) at the look-at point."""
    pts = np.asarray(points, float)
    L = look_at_point(centres, axes, pts)
    D = float(np.median(np.linalg.norm(centres - L, axis=1)))
    near = np.flatnonzero(np.linalg.norm(pts - L, axis=1) < 0.7 * D)   # what the photos show around the aim point
    if len(near) < 200:
        near = np.arange(len(pts))
    sub = pts[near]
    voxel = D / 400
    down = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub)).voxel_down_sample(voxel)
    while len(down.points) > 150_000:
        voxel *= 1.3
        down = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub)).voxel_down_sample(voxel)
    Dp = np.asarray(down.points)
    noise = estimate_noise(sub if len(sub) <= 2_000_000 else sub[:: len(sub) // 2_000_000 + 1])
    thr = max(2.5 * noise, 0.5 * voxel)   # voxel centroids of a flat table stay on it: the noise sets the band
    info = {"look_at": L.round(6).tolist(), "camera_distance": D, "planes_removed": 0, "clusters": 0}

    # The table: a plane the cameras look down on, with the aim point above it and (almost) nothing near the aim
    # point below it - a face of the part has the rest of the part below it. Later planes count only as walls,
    # roughly upright to the table: a face of the part parallel to the table (a flange, a plate) also has the rest
    # of the part above it once the table is gone.
    keep = np.ones(len(Dp), bool)
    tried = np.zeros(len(Dp), bool)
    planes = []
    for _ in range(5):
        cand = np.flatnonzero(keep & ~tried)
        if len(cand) < 300 or len(planes) >= 3:
            break
        o3d.utility.random.seed(7)
        model, inl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Dp[cand])).segment_plane(thr, 3, 2000)
        inl = cand[np.asarray(inl, dtype=np.int64)]
        if len(inl) < 0.05 * keep.sum():
            break
        n, d = np.asarray(model[:3], float), float(model[3])
        k = np.linalg.norm(n)
        n, d = n / k, d / k
        side = centres @ n + d
        up = 1.0 if (side > 0).mean() >= 0.5 else -1.0
        n, d = up * n, up * d
        rest = np.setdiff1d(np.flatnonzero(keep), inl)
        rest = rest[np.linalg.norm(Dp[rest] - L, axis=1) < 0.5 * D]
        below = float((Dp[rest] @ n + d < -2.5 * thr).mean()) if len(rest) else 1.0
        support = (side * up > 0).mean() >= 0.8 and L @ n + d > 2 * thr and below <= 0.1
        if support and planes:
            support = abs(float(n @ planes[0][0])) < 0.35
        if support:
            keep &= Dp @ n + d > 2.5 * thr
            planes.append((n, d))
        else:
            tried[inl] = True
    info["planes_removed"] = len(planes)

    kept = np.flatnonzero(keep)
    if len(kept) < 50:
        raise ValueError("the part could not be found in the photos")
    eps = 3 * max(voxel, estimate_spacing(Dp[kept]))
    labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Dp[kept])).cluster_dbscan(eps, 6))
    good = labels >= 0
    if not good.any():
        raise ValueError("the part could not be found in the photos")
    ids, counts = np.unique(labels[good], return_counts=True)
    dist_L = np.linalg.norm(Dp[kept] - L, axis=1)
    dmin = np.array([dist_L[labels == i].min() for i in ids])
    main = int(np.argmax(counts / (1 + (dmin / (0.08 * D)) ** 2)))
    reach = float(np.percentile(dist_L[labels == ids[main]], 95))
    chosen = [ids[main]] + [i for i, c, m in zip(ids, counts, dmin)
                            if i != ids[main] and m < reach and c >= 0.01 * counts[main]]
    part_down = Dp[kept[np.isin(labels, chosen)]]
    info["clusters"] = len(chosen)

    ok = np.isfinite(_query(cKDTree(part_down), sub, k=1, distance_upper_bound=1.8 * voxel)[0])
    for n, d in planes:
        ok &= sub @ n + d > 2.5 * thr
    idx = near[ok]
    info.update(points=int(len(idx)), noise=noise, planes=planes)
    return idx, info


def find_part_by_camera_points(points: np.ndarray, camera_points: np.ndarray, centres: np.ndarray, axes: np.ndarray,
                               log: Log = print) -> tuple[np.ndarray, dict] | None:
    """The part found with the reconstruction's own (sparse) points, which sit exactly on the table: the table plane
    comes from them, and so does the part's footprint and height. The dense model's table is measured against that
    plane (on real photos it sat ~2 % of the camera distance low and ~0.7 % thick, far more than its local noise, so
    a plane fitted to the dense points alone kept a ring of table around the part). None when the sparse points do
    not show a table with a part on it (then find_part decides from the dense model)."""
    cp = np.asarray(camera_points, float)
    cp = cp[np.all(np.isfinite(cp), axis=1)]
    pts = np.asarray(points, float)
    if len(cp) < 200:
        return None
    L = look_at_point(centres, axes, cp)
    D = float(np.median(np.linalg.norm(centres - L, axis=1)))
    near_c = cp[np.linalg.norm(cp - L, axis=1) < 0.7 * D]
    if len(near_c) < 200:
        return None
    o3d.utility.random.seed(7)
    model, inl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(near_c)).segment_plane(0.003 * D, 3, 3000)
    n, d = np.asarray(model[:3], float), float(model[3])
    k = np.linalg.norm(n)
    n, d = n / k, d / k
    if ((centres @ n + d) > 0).mean() < 0.5:
        n, d = -n, -d
    if len(inl) < 0.25 * len(near_c) or ((centres @ n + d) > 0).mean() < 0.8 or L @ n + d < 0.004 * D:
        return None
    above = near_c[near_c @ n + d > 0.004 * D]
    if len(above) < 100:
        return None
    # sparse points are few and uneven: link them at 2 % of the camera distance, not at their own spacing
    labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(above)).cluster_dbscan(
        max(3 * estimate_spacing(above), 0.02 * D), 5))
    good = labels >= 0
    if not good.any():
        return None
    Lp = L - (L @ n + d) * n
    foot = lambda X, h: np.linalg.norm(X - np.outer(h, n) - Lp, axis=1)   # distance from the aim point, along the table
    ids, counts = np.unique(labels[good], return_counts=True)
    dmin = np.array([foot(above[labels == i], above[labels == i] @ n + d).min() for i in ids])
    main = ids[int(np.argmax(counts / (1 + (dmin / (0.08 * D)) ** 2)))]
    part_c = above[labels == main]
    if len(part_c) < 100:
        return None
    hc = part_c @ n + d
    r_part = float(np.percentile(foot(part_c, hc), 95))
    top = float(np.percentile(hc, 99))
    near = np.flatnonzero(np.linalg.norm(pts - L, axis=1) < 0.7 * D)
    h = pts[near] @ n + d
    radial = foot(pts[near], h)
    table = (radial > 1.3 * r_part) & (np.abs(h) < 0.05 * D)
    if table.sum() >= 1000:
        level = float(np.median(h[table]))
        sigma = float(1.4826 * np.median(np.abs(h[table] - level)))
    else:
        level, sigma = 0.0, 0.003 * D
    band = level + max(3 * sigma, 0.004 * D)
    ok = (h > band) & (h < top + 0.05 * D) & (radial < 1.15 * r_part)
    d_c, _ = _query(cKDTree(part_c), pts[near[ok]], k=1, distance_upper_bound=0.04 * D)
    idx = near[ok][np.isfinite(d_c)]
    if len(idx) < 200:
        return None
    log(f"Found the part with the photos' own points: {len(part_c):,} of them stand on the table "
        f"(the dense model's table sits {level / D * 100:+.1f} % of the camera distance off it)")
    info = {"look_at": L.round(6).tolist(), "camera_distance": D, "planes_removed": 1, "clusters": 1,
            "points": int(len(idx)), "noise": sigma, "planes": [(n, d)], "by": "camera points",
            "table_offset_pct": round(level / D * 100, 3), "table_sigma_pct": round(sigma / D * 100, 3)}
    return idx, info


def camera_points_on_part(points: np.ndarray, planes: list, D: float) -> np.ndarray:
    """Mask of the reconstruction's own (sparse) points above the table and walls found in the dense model. Each
    plane is fitted again to the sparse points near it: the dense model may sit a little off the cameras."""
    pts = np.asarray(points, float)
    keep = np.ones(len(pts), bool)
    for n, d in planes:
        band = np.flatnonzero(np.abs(pts @ n + d) < 0.03 * D)
        if len(band) >= 30:
            o3d.utility.random.seed(7)
            model, _ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[band])).segment_plane(0.002 * D, 3, 1000)
            n2, d2 = np.asarray(model[:3], float), float(model[3])
            k = np.linalg.norm(n2) * (1.0 if n2 @ n >= 0 else -1.0)
            if abs(n2 @ n) / abs(k) > 0.95:
                n, d = n2 / k, d2 / k
        keep &= pts @ n + d > 0.004 * D
    return keep


# --------------------------------------------------------------------------- ICP with scale
def _umeyama(P: np.ndarray, Q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted similarity mapping P onto Q (Umeyama 1991)."""
    w = w / w.sum()
    mp, mq = w @ P, w @ Q
    Pc, Qc = P - mp, Q - mq
    U, S, Vt = np.linalg.svd((Qc * w[:, None]).T @ Pc)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / max(w @ (Pc ** 2).sum(1), 1e-300))
    return similarity(s, R, mq - s * R @ mp)


def _plane_step(P: np.ndarray, Q: np.ndarray, N: np.ndarray, w: np.ndarray) -> np.ndarray:
    """One linearised point-to-plane step for scale, rotation and shift (about the weighted centre of P)."""
    c = (w @ P) / w.sum()
    Pc = P - c
    A = np.hstack([np.cross(Pc, N), N, np.einsum("ij,ij->i", Pc, N)[:, None]])
    b = -np.einsum("ij,ij->i", P - Q, N)
    Aw = A * w[:, None]
    H = A.T @ Aw
    H += 1e-9 * np.trace(H) * np.eye(7)
    x = np.linalg.solve(H, Aw.T @ b)
    sR = math.exp(x[6]) * _rotvec(x[:3])
    T = np.eye(4)
    T[:3, :3] = sR
    T[:3, 3] = c + x[3:6] - sR @ c
    return T


def similarity_icp(src: np.ndarray, tgt: np.ndarray, tree: cKDTree, normals: np.ndarray | None, S: np.ndarray,
                   thresholds, iterations: int = 30, early_points: int | None = None,
                   fixed_scale: bool = False) -> np.ndarray:
    """Robust (Tukey) ICP that also estimates the scale: point-to-plane when target normals are given, else
    point-to-point. Levels before the last use at most `early_points` source points. fixed_scale: the scale of S
    is kept (each step is made rigid about the points it moves). Returns the refined similarity."""
    S = np.asarray(S, float).copy()
    few = src
    if early_points and len(src) > early_points:
        few = src[np.random.default_rng(5).choice(len(src), early_points, replace=False)]
    for level, thr in enumerate(thresholds):
        pts = src if level == len(thresholds) - 1 else few
        for _ in range(iterations):
            X = _apply(S, pts)
            d, idx = _query(tree, X, k=1, distance_upper_bound=thr)
            ok = np.isfinite(d)
            if ok.sum() < 12:
                break
            X, Q = X[ok], tgt[idx[ok]]
            if normals is None:
                w = (1 - (d[ok] / thr) ** 2) ** 2
                step = _umeyama(X, Q, w + 1e-12)
            else:
                N = normals[idx[ok]]
                r = np.einsum("ij,ij->i", X - Q, N)
                w = np.clip(1 - (r / thr) ** 2, 0, None) ** 2
                if w.sum() <= 0:
                    break
                step = _plane_step(X, Q, N, w + 1e-12)
            if fixed_scale:   # the same turn and the same move of the points' centre, without the scaling
                k = float(np.cbrt(np.linalg.det(step[:3, :3])))
                m = X.mean(0)
                rigid = np.eye(4)
                rigid[:3, :3] = step[:3, :3] / k
                rigid[:3, 3] = step[:3, :3] @ m + step[:3, 3] - rigid[:3, :3] @ m
                step = rigid
            S = step @ S
            if np.sqrt(np.mean(np.sum((_apply(step, X) - X) ** 2, axis=1))) < 1e-3 * thr:
                break
    return S


# --------------------------------------------------------------------------- the model to line up with
def _extents(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(axes as rows, robust 2-98 % sizes along them, centre of that box), longest axis first."""
    c = pts.mean(0)
    axes = np.linalg.eigh(np.cov((pts - c).T))[1].T
    coords = (pts - c) @ axes.T
    lo, hi = np.percentile(coords, [2, 98], axis=0)
    order = np.argsort(-(hi - lo))
    axes, lo, hi = axes[order], lo[order], hi[order]
    return axes, hi - lo, c + ((lo + hi) / 2) @ axes


class _Model:
    """The owner's model as points with normals (a mesh is sampled), at two resolutions, with search trees."""

    def __init__(self, geom):
        if is_cloud(geom):
            pts = np.asarray(geom.points, float)
            self.spacing = estimate_spacing(pts)
            cloud = geom
        else:
            mesh = o3d.geometry.TriangleMesh(geom)
            mesh.compute_triangle_normals()
            cloud = mesh.sample_points_uniformly(300_000, use_triangle_normal=True)
            pts = np.asarray(cloud.points, float)
            self.spacing = estimate_spacing(pts)
        if len(pts) < 100:
            raise ValueError("the model has too few points")
        sel = np.arange(len(pts))
        if len(pts) > MAX_TARGET:
            sel = np.sort(np.random.default_rng(0).choice(len(pts), MAX_TARGET, replace=False))
        work = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[sel]))
        self.axes, self.size, self.centre = _extents(pts[sel])
        self.diag = float(np.linalg.norm(self.size))
        sub_spacing = estimate_spacing(pts[sel])
        if cloud.has_normals():
            work.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals)[sel])
        else:
            work.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * sub_spacing, max_nn=30))
        self.pts, self.nrm = pts[sel], np.asarray(work.normals)
        self.tree = cKDTree(self.pts)
        self.sub_spacing = sub_spacing
        self.noise = estimate_noise(self.pts)
        coarse = work.voxel_down_sample(self.diag / 120)
        coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.diag / 30, max_nn=30))
        self.coarse, self.coarse_nrm = np.asarray(coarse.points), np.asarray(coarse.normals)
        self.coarse_tree = cKDTree(self.coarse)


def _fit(model: _Model, X: np.ndarray, tol: float) -> tuple[float, float]:
    """(share of X within tol of the model surface, rms of those distances). Point-to-plane distance to the nearest
    model point, so a subsampled model does not read as a gap."""
    reach = max(tol, 2.0 * model.sub_spacing)
    d, idx = _query(model.tree, X, k=1, distance_upper_bound=reach)
    near = np.isfinite(d)
    dist = np.full(len(X), np.inf)
    dist[near] = np.abs(np.einsum("ij,ij->i", X[near] - model.pts[idx[near]], model.nrm[idx[near]]))
    inside = dist < tol
    rmse = float(np.sqrt(np.mean(dist[inside] ** 2))) if inside.any() else float("inf")
    return float(inside.mean()), rmse


def _coverage(model_pts: np.ndarray, X: np.ndarray, tol: float) -> float:
    """Share of the model points within tol of the (aligned) photo part."""
    d, _ = _query(cKDTree(X), model_pts, k=1, distance_upper_bound=tol)
    return float(np.isfinite(d).mean())


def _f1(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def failed(name: str, why: str) -> str:
    return (f"The photos could not be lined up with {name}: {why}. Take more photos all the way round, with the "
            "whole part in view")


# --------------------------------------------------------------------------- the alignment
def align_photo_model(photo_points: np.ndarray, world_to_cameras: list, target, name: str = "the model",
                      poses: str = "colmap", log: Log = print, progress=None,
                      camera_points: np.ndarray | None = None, scale: float | None = None,
                      partial: bool = False) -> dict:
    """Similarity (scale, rotation, shift) from the photo model's frame into the target's (a scan or mesh, mm).

    photo_points: the dense photo model (robust, but only good to ~2 % and possibly scaled a little against the
    cameras) - used to find the part and search the pose. camera_points: the reconstruction's own triangulated points
    (COLMAP's sparse.ply), exactly consistent with the cameras - the final scale and pose are fitted to those on the
    part, since a scale error there shifts every projected colour. Without them the dense fit is used, lower trust.

    Returns {transform (4x4 photo -> model), scale, fitted_on ("camera points" | "dense model"), fitness (share of
    those photo points within `within_mm` of the model), rmse_mm, within_mm, photo_points, coverage (share of the
    model the dense part covers), dense_fitness, dense_scale_offset_pct, trusted, ambiguous, alternative_angle,
    warnings, part_points}. Raises ValueError (plain words) when the photos do not line up.

    scale: the photo -> model scale when it is already known (a second scan of the same item, in mm like the first):
    only turn and shift are searched. partial: the model is one scan of an item the photos show more of (a merge):
    far fewer of the photos' points lie on it, so the shares that refuse a fit are lower."""
    fit_fail = PARTIAL_FIT_FAIL if partial else FIT_FAIL
    progress = progress or (lambda *a: None)
    rng = np.random.default_rng(0)
    photo_points = np.asarray(photo_points, float)
    photo_points = photo_points[np.all(np.isfinite(photo_points), axis=1)]
    centres, axes = camera_centres_axes(world_to_cameras)
    found = None
    if camera_points is not None and len(camera_points):
        found = find_part_by_camera_points(photo_points, camera_points, centres, axes, log)
    try:
        part_idx, part = found or find_part(photo_points, centres, axes, log)
    except ValueError as exc:
        raise ValueError(failed(name, str(exc)))
    P = photo_points[part_idx]
    if len(P) < 200:
        raise ValueError(failed(name, "the part could not be found in the photos"))
    extra = f", removed {part['planes_removed']} table/background plane(s)" if part["planes_removed"] else ""
    log(f"Found the part in the photo model: {len(P):,} points{extra}")
    progress(0.15)

    model = _Model(target)
    p_axes, p_size, p_centre = _extents(P)
    ratios = model.size / np.maximum(p_size, 1e-12)
    s0 = float(np.exp(np.mean(np.log(ratios))))
    # the ICP finds the scale; two starts cover a photo part that is missing its underside or has table bits
    scales = [s0] + ([float(ratios[0])] if abs(math.log(ratios[0] / s0)) > 0.04 else [])
    if scale is not None:
        s0, scales = float(scale), [float(scale)]

    def pick(n, seed):
        return P if len(P) <= n else P[np.random.default_rng(seed).choice(len(P), n, replace=False)]

    coarse_src = pick(1000, 1) - p_centre
    fine_src = pick(MAX_FINE, 2)
    eval_src = pick(MAX_EVAL, 3)

    # orientations: the part axes matched (all sign / order choices) and an even grid; ranked with a quick fit
    rotations = []
    for perm in ((0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)):
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),
                      (-1, -1, -1)):
            R = model.axes.T @ np.diag(signs) @ p_axes[list(perm)]
            if np.linalg.det(R) > 0:
                rotations.append(R)
    rotations = np.concatenate([np.asarray(rotations), rotation_grid(GRID_ROTATIONS)])
    X = s0 * np.einsum("rij,nj->rni", rotations, coarse_src) + model.centre
    n_rot, n_pts = X.shape[:2]
    for reach in (0.25, 0.12, 0.06):   # a few shift-only steps: the photo part lacks the underside, the scan may not
        d, idx = _query(model.coarse_tree, X.reshape(-1, 3), k=1, distance_upper_bound=reach * model.diag)
        ok = np.isfinite(d).reshape(n_rot, n_pts)
        delta = np.zeros((n_rot * n_pts, 3))
        flat_ok = ok.ravel()
        delta[flat_ok] = model.coarse[idx[flat_ok]] - X.reshape(-1, 3)[flat_ok]
        delta = delta.reshape(n_rot, n_pts, 3)
        X = X + (delta.sum(1) / np.maximum(ok.sum(1), 1)[:, None])[:, None, :]
    d, _ = _query(model.coarse_tree, X.reshape(-1, 3), k=1)
    quick = (d.reshape(n_rot, n_pts) < 0.03 * model.diag).mean(1)
    candidates, chosen = [], []
    for r in np.argsort(-quick):
        if all(_angle_deg(rotations[r], rotations[c]) > 20 for c in chosen):
            chosen.append(r)
            anchor = X[r].mean(0) - s0 * rotations[r] @ coarse_src.mean(0)   # where the photo part's centre lands
            candidates += [("grid", similarity(s, rotations[r], anchor - s * rotations[r] @ p_centre))
                           for s in scales]
        if len(chosen) >= 8:
            break
    progress(0.35)

    # feature matching at the first scale guess finds the pose even when the shapes' centres differ a lot
    try:
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s0 * (pick(60_000, 4) - p_centre)))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model.pts))
        for label, T in global_candidates(src, tgt, model.diag / 40, 1):
            candidates.append((label, T @ similarity(s0, np.eye(3), -s0 * p_centre)))
    except Exception:
        pass

    coarse_pts = coarse_src + p_centre

    def coarse(S0):
        S = similarity_icp(coarse_pts, model.coarse, model.coarse_tree, None, S0,
                           [0.2 * model.diag, 0.1 * model.diag, 0.05 * model.diag], 10,
                           fixed_scale=scale is not None)
        X = _apply(S, coarse_pts)
        tol = 0.02 * model.diag
        fit = float((_query(model.coarse_tree, X, k=1)[0] < tol).mean())
        return S, _f1(fit, _coverage(model.coarse, X, 2 * tol))

    ranked = []
    for label, S0 in candidates:
        S, score = coarse(S0)
        ranked.append((score, label, S))
    ranked.sort(key=lambda x: -x[0])
    # the best pose turned 180 degrees about its own axes: exposes parts that look alike both ways round
    best = ranked[0][2]
    Xb = _apply(best, coarse_pts)
    c = Xb.mean(0)
    flips = np.linalg.svd(Xb - c, full_matrices=False)[2]
    for k, axis in enumerate(flips):
        F = np.eye(4)
        F[:3, :3] = _rotvec(axis * np.pi)
        F[:3, 3] = c - F[:3, :3] @ c
        S, score = coarse(F @ best)
        ranked.append((score, f"flip{k + 1}", S))
    ranked.sort(key=lambda x: -x[0])
    progress(0.6)

    # fine: point-to-plane with scale on the best few distinct poses, judged with the dense model's own accuracy
    distinct = []
    for score, label, S in ranked:
        R = split_similarity(S)[1]
        if all(_angle_deg(R, split_similarity(o[2])[1]) > 10 for o in distinct):
            distinct.append((score, label, S))
        if len(distinct) >= 3:
            break
    photo_noise = estimate_noise(P if len(P) <= 500_000 else P[:: len(P) // 500_000 + 1])
    photo_spacing = estimate_spacing(P if len(P) <= 200_000 else eval_src)

    def dense_tol(s):
        return max(3 * model.spacing, 2.5 * math.hypot(s * photo_noise, model.noise), DENSE_ERROR * model.diag)

    results = []
    for score, label, S in distinct:
        tol = dense_tol(split_similarity(S)[0])
        S = similarity_icp(fine_src, model.pts, model.tree, model.nrm, S, [2 * tol, tol, tol / 2], 20, 5000,
                           fixed_scale=scale is not None)
        s = split_similarity(S)[0]
        tol = dense_tol(s)
        X = _apply(S, eval_src)
        fit, rmse = _fit(model, X, tol)
        cover = _coverage(model.pts[rng.choice(len(model.pts), min(len(model.pts), 30_000), replace=False)], X,
                          max(tol, 2 * s * photo_spacing))
        results.append({"S": S, "fit": fit, "cover": cover, "rmse": rmse, "tol": tol, "label": label,
                        "score": _f1(fit, cover)})
    results.sort(key=lambda r: -r["score"])
    dense = results[0]
    s_dense, R, _ = split_similarity(dense["S"])
    rival = None
    for r in results[1:]:
        angle = _angle_deg(R, split_similarity(r["S"])[1])
        if angle > RIVAL_ANGLE and r["score"] >= RIVAL_SHARE * dense["score"] and \
                r["rmse"] <= 1.25 * max(dense["rmse"], 1e-12):
            rival = round(angle, 1)
            break
    if dense["fit"] < fit_fail:
        raise ValueError(failed(name, f"only {dense['fit']:.0%} of the part in the photos lies on the model"))
    if dense["cover"] < COVER_FAIL:
        raise ValueError(failed(name, f"the photos cover only {dense['cover']:.0%} of the model"))
    progress(0.85)

    # final: scale and pose from the points that agree exactly with the cameras
    warnings = []
    S, fitted_on, used = dense["S"], "dense model", len(eval_src)
    fit, rmse, tol = dense["fit"], dense["rmse"], dense["tol"]
    finals: list[dict] = []
    if camera_points is not None and len(camera_points):
        cp = np.asarray(camera_points, float)
        cp = cp[np.all(np.isfinite(cp), axis=1)]
        cp = cp[camera_points_on_part(cp, part["planes"], part["camera_distance"])]
        # only those on the part found in the photo model (not clutter elsewhere on the table), whatever the pose
        d_part, _ = _query(cKDTree(P[:: max(1, len(P) // 400_000)]), cp, k=1,
                           distance_upper_bound=0.03 * part["camera_distance"])
        cp = cp[np.isfinite(d_part)]
        tol_c = max(3 * model.spacing, CAMERA_POINT_ERROR * model.diag)
        # The dense model ranks poses only roughly: a hole pattern or a round part fits several ways on it. Every
        # strong candidate is refined on the exact points, and they decide (scored on all of them, so the poses
        # are compared on the same points).
        finals.clear()
        if len(cp) >= MIN_CAMERA_POINTS:
            for r in [x for x in results if x["score"] >= (0.5 if partial else 0.75) * dense["score"]][:6]:
                reach = max(2.5 * r["tol"], 0.03 * model.diag)   # room for the dense model's scale offset
                d, _ = _query(model.tree, _apply(r["S"], cp), k=1, distance_upper_bound=reach)
                near = cp[np.isfinite(d)]
                if len(near) < MIN_CAMERA_POINTS:
                    continue
                Sc = similarity_icp(near, model.pts, model.tree, model.nrm, r["S"],
                                    [reach, reach / 2, reach / 4, max(reach / 8, 2 * tol_c), 2 * tol_c, tol_c], 40,
                                    fixed_scale=scale is not None)
                f_all, rm = _fit(model, _apply(Sc, cp), tol_c)
                finals.append({"S": Sc, "fit": f_all, "rmse": rm})
        if finals:
            finals.sort(key=lambda f: -f["fit"])
            best = finals[0]
            R_best = split_similarity(best["S"])[1]
            rival = None
            for f in finals[1:]:
                angle = _angle_deg(R_best, split_similarity(f["S"])[1])
                if angle > RIVAL_ANGLE and f["fit"] >= RIVAL_SHARE * best["fit"]:
                    rival = round(angle, 1)
                    break
            S, fitted_on, used, tol = best["S"], "camera points", len(cp), tol_c
            fit, rmse = best["fit"], best["rmse"]
            if len(finals) > 1:
                log(f"  {len(finals)} candidate poses checked on the photos' own points: best {best['fit']:.0%}, "
                    f"next {finals[1]['fit']:.0%}")
            if fit < fit_fail:
                raise ValueError(failed(name, f"only {fit:.0%} of the photos' own points on the part lie on the "
                                              "model"))
        else:
            warnings.append(f"Only {len(cp)} of the points the photos were placed with lie on the part, so the scale "
                            "comes from the rougher dense model: colours may be shifted slightly. Photos with more "
                            "detail on the part help.")
    elif poses == "colmap":
        warnings.append("The points the photos were placed with are missing, so the scale comes from the rougher "
                        "dense model: colours may be shifted slightly.")
    s = split_similarity(S)[0]
    offset = 100 * (s_dense / s - 1)
    what = "of the photos' own points on the part" if fitted_on == "camera points" else "of the photo model"
    log(f"Lined the photos up with {name}: scale ×{s:.4g}, {fit:.0%} {what} within {tol:.2f} mm (rmse {rmse:.3f} "
        f"mm), the photos cover {dense['cover']:.0%} of the model")
    if fitted_on == "camera points" and abs(offset) >= 0.5:
        log(f"  the dense photo model was {offset:+.1f}% off in scale against the cameras - corrected")
    progress(0.95)

    if rival is not None:
        warnings.append(f"A pose turned {rival:.0f}° fits almost as well - the part looks alike from several sides, "
                        "so the colours could land turned. Check them on the model.")
    if poses != "colmap":
        warnings.append("The camera positions could only be estimated roughly, so the colours may be blurred or "
                        "shifted. More photos with more overlap and a patterned sheet under the part help.")
    if fit < FIT_TRUST or dense["cover"] < COVER_TRUST:
        warnings.append(f"Only part of the photos matches the model ({fit:.0%} on it, {dense['cover']:.0%} of the "
                        "model covered), so some colours may be misplaced. Check them on the model.")
    if abs(math.log(s / s0)) > math.log(1.3):
        warnings.append("The part in the photos and the model are not the same size - does the model show the "
                        "whole part?")
    for w in warnings:
        log("WARNING: " + w)
    geometry_ok = (fitted_on == "camera points" and fit >= FIT_TRUST and dense["cover"] >= COVER_TRUST
                   and poses == "colmap" and abs(math.log(s / s0)) <= math.log(1.3))
    trusted = geometry_ok and rival is None
    return {"transform": S, "scale": s, "fitted_on": fitted_on, "fitness": fit, "rmse_mm": rmse, "within_mm": tol,
            "photo_points": int(used), "coverage": dense["cover"], "dense_fitness": dense["fit"],
            "dense_scale_offset_pct": offset, "trusted": bool(trusted), "geometry_ok": bool(geometry_ok),
            "ambiguous": rival is not None,
            "alternative_angle": rival, "warnings": warnings, "part_points": int(len(P)),
            "planes_removed": part["planes_removed"],
            # the distinct placements tried last, best first (a scan of one side of an item can sit either way)
            "candidates": [f["S"] for f in finals] if finals else [r["S"] for r in results]}
