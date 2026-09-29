"""Accuracy checks: when a dimension looks off, is it the scanner or the software?

Nothing in this module changes geometry; every function measures and reports.

* `robust_dimensions`  part size from trimmed extents along the principal axes (not the axis-aligned box, which
                       depends on how the part happened to lie in the scanner's coordinate system)
* `operation_drift`    how far the operation that produced an asset moved or resized its parent's surface
* `compare_scans`      two scans of the same part aligned rigidly, then a similarity fit on their overlap: relative
                       scale (ppm), surface offset, per-axis extent differences and surface separation - this is
                       scanner repeatability, independent of any processing
* `reference_check`    measurement of a known artefact: gauge block / known length, sphere pair (ball bar),
                       diameter (pin, ring), thread pitch (an absolute scale check that needs no calipers)
* `fit_plane`, `fit_sphere`, `fit_circle`, `fit_cylinder`, `fit_thread`   small robust fitting helpers
* `SurfaceModel`       signed distance to a scanned cloud (local quadric fits, so thread crests and fillets are not
                       biased like with a plane fit) or to a mesh (exact closest point)

Units are scan units (mm for Revopoint scans). Signed distances are positive outside the reference surface
(material added). Robust fits use Tukey biweights with a MAD scale, so stray points do not pull the result.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .io import Geometry, estimate_spacing, is_cloud

Log = Callable[[str], None]

AXIS_NAMES = ("length", "width", "height")
DEFAULT_TRIM = 0.0005          # 0.05 % of the points trimmed at each end of every axis (as the part frame, Contract 3)
TUKEY_C = 4.685


def _quiet(msg: str) -> None:
    pass


# --------------------------------------------------------------------------- small helpers
def jsonable(obj):
    """Recursively convert numpy scalars / arrays to plain Python (NaN / inf become None)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def _mad_sigma(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def _tukey(r, c: float) -> np.ndarray:
    u = np.asarray(r, float) / max(float(c), 1e-300)
    return np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)


def _unit(v, what: str = "the direction") -> np.ndarray:
    a = np.asarray(v, float).reshape(-1)
    n = float(np.linalg.norm(a)) if a.shape == (3,) else float("nan")
    if a.shape != (3,) or not np.isfinite(n) or n == 0:
        raise ValueError(f"{what} must be a non-zero vector [x, y, z]")
    return a / n


def _basis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.eye(3)[int(np.argmin(np.abs(a)))]
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def _skew(w) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], float)


def _rotation(w) -> np.ndarray:
    """Rotation matrix of a rotation vector (Rodrigues)."""
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3) + _skew(w)
    K = _skew(w / th)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def rotation_angle_deg(R) -> float:
    R = np.asarray(R, float)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def apply_transform(T, points) -> np.ndarray:
    T = np.asarray(T, float)
    return np.asarray(points, float) @ T[:3, :3].T + T[:3, 3]


def _stats(d) -> dict | None:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    a = np.abs(d)
    return {"signed_mean": float(d.mean()), "median": float(np.median(d)), "mean": float(a.mean()),
            "rms": float(np.sqrt(np.mean(d * d))), "std": float(d.std()), "p05": float(np.percentile(d, 5)),
            "p95_signed": float(np.percentile(d, 95)), "median_abs": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()), "count": int(d.size)}


def _histogram(d, limit: float, bins: int = 41) -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    edges = np.linspace(-limit, limit, bins + 1)
    counts, _ = np.histogram(d, bins=edges)
    return {"edges": edges.tolist(), "counts": counts.tolist(), "underflow": int((d < -limit).sum()),
            "overflow": int((d > limit).sum())}


def _fmt_mm(x: float, signed: bool = True) -> str:
    return f"{x:+.3f} mm" if signed else f"{x:.3f} mm"


# --------------------------------------------------------------------------- points of any geometry
def geometry_points(geom: Geometry) -> np.ndarray:
    """All positions: cloud points or mesh vertices (float64)."""
    return np.asarray(geom.points if is_cloud(geom) else geom.vertices, dtype=float)


def _mesh_arrays(mesh):
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    return V, F


def _outward_sign(V: np.ndarray, F: np.ndarray, cross: np.ndarray) -> float:
    """+1 when the triangle winding points outward: signed volume of a closed mesh, flux through the triangles
    measured from the centre for an open one."""
    centre = V.mean(axis=0)
    flux = float(np.einsum("ij,ij->", V[F[:, 0]] - centre, cross))
    return 1.0 if flux >= 0 else -1.0


def surface_points(geom: Geometry, max_points: int = 2_000_000, seed: int = 0):
    """(points, normals or None) that represent the surface evenly.

    Clouds: the points (and their normals). Meshes: area-uniform random samples with outward face normals, so every
    square millimetre counts the same whatever the triangle sizes (Poisson meshes are much denser than the scan)."""
    rng = np.random.default_rng(seed)
    if is_cloud(geom):
        pts = np.asarray(geom.points, dtype=float)
        nrm = np.asarray(geom.normals, dtype=float) if geom.has_normals() else None
        if len(pts) > max_points:
            idx = np.sort(rng.choice(len(pts), max_points, replace=False))
            pts = pts[idx]
            nrm = None if nrm is None else nrm[idx]
        return pts, nrm
    V, F = _mesh_arrays(geom)
    if len(F) == 0:
        return V[:max_points].copy(), None
    cross = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area2 = np.linalg.norm(cross, axis=1)
    if not np.isfinite(area2).all() or area2.sum() <= 0:
        return V[:max_points].copy(), None
    sign = _outward_sign(V, F, cross)
    count = int(min(max_points, max(len(V), 100_000)))
    tri = rng.choice(len(F), size=count, p=area2 / area2.sum())
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    pts = (V[F[tri, 0]] * (1 - r1)[:, None] + V[F[tri, 1]] * (r1 * (1 - r2))[:, None]
           + V[F[tri, 2]] * (r1 * r2)[:, None])
    nrm = sign * cross[tri] / np.maximum(area2[tri], 1e-300)[:, None]
    return pts, nrm


def _estimate_normals(points: np.ndarray, log: Log = _quiet) -> np.ndarray:
    """Outward normals for a cloud stored without them (MST propagation, as cleaning does)."""
    from .clean import orient_normals

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    spacing = estimate_spacing(points) or 1.0
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=6 * spacing, max_nn=30))
    orient_normals(pcd, log)
    return np.asarray(pcd.normals, dtype=float).copy()


def transformed_geometry(geom: Geometry, T) -> Geometry:
    """Copy of geom moved by the 4x4 T in float64 (normals use the inverse transpose)."""
    T = np.asarray(T, float)
    N = np.linalg.inv(T[:3, :3]).T
    if is_cloud(geom):
        out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(apply_transform(T, np.asarray(geom.points))))
        if geom.has_normals():
            n = np.asarray(geom.normals) @ N.T
            out.normals = o3d.utility.Vector3dVector(n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None])
        return out
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(apply_transform(T, np.asarray(geom.vertices))),
                                    geom.triangles)
    if np.linalg.det(T[:3, :3]) < 0:
        out.triangles = o3d.utility.Vector3iVector(np.asarray(geom.triangles)[:, ::-1].copy())
    return out


# --------------------------------------------------------------------------- robust fits
def fit_plane(points, robust: bool = True, iterations: int = 12) -> dict:
    """Plane through points (weighted PCA, Tukey reweighting). Returns point, normal, rms, flatness (p99.5 - p0.5
    of the inlier residuals), points_used, inlier_fraction."""
    P = np.asarray(points, float)
    if len(P) < 3:
        raise ValueError("A plane fit needs at least 3 points")
    w = np.ones(len(P))
    c, n, r = P.mean(axis=0), np.array([0.0, 0.0, 1.0]), np.zeros(len(P))
    for _ in range(iterations if robust else 1):
        c = (P * w[:, None]).sum(axis=0) / w.sum()
        X = P - c
        cov = (X * w[:, None]).T @ X
        n = np.linalg.eigh(cov)[1][:, 0]
        r = X @ n
        if not robust:
            break
        s = max(_mad_sigma(r[w > 0]), 1e-12)
        w_new = _tukey(r, TUKEY_C * s)
        if w_new.sum() < 3:
            break
        done = np.allclose(w_new, w, atol=1e-4)
        w = w_new
        if done:
            break
    inl = w > 0
    ri = r[inl]
    return {"point": c, "normal": n, "rms": float(np.sqrt(np.mean(ri ** 2))) if len(ri) else 0.0,
            "flatness": float(np.percentile(ri, 99.5) - np.percentile(ri, 0.5)) if len(ri) else 0.0,
            "points_used": int(inl.sum()), "inlier_fraction": float(inl.mean()), "residuals": r}


def _weighted_refine(residual, x0, robust: bool, iterations: int = 6, floor: float = 0.0):
    """Gauss-Newton (scipy least_squares) with Tukey reweighting. Returns (x, residuals, weights, jacobian)."""
    x = np.asarray(x0, float)
    res = residual(x)
    w = np.ones(len(res))
    sol = None
    for it in range(iterations if robust else 1):
        sw = np.sqrt(w)
        sol = least_squares(lambda p: sw * residual(p), x, method="trf", x_scale="jac")
        x = sol.x
        res = residual(x)
        if not robust:
            break
        s = max(_mad_sigma(res[w > 0]), floor, 1e-12)
        w_new = _tukey(res, TUKEY_C * s)
        if w_new.sum() < len(x) + 1:
            break
        done = np.allclose(w_new, w, atol=1e-4)
        w = w_new
        if done and it > 0:
            break
    return x, res, w, sol.jac


def _covariance(J: np.ndarray, res: np.ndarray, w: np.ndarray) -> np.ndarray:
    dof = max(float(w.sum()) - J.shape[1], 1.0)
    sigma2 = float((w * res ** 2).sum()) / dof
    return np.linalg.pinv(J.T @ J) * sigma2


def fit_circle(xy, robust: bool = True) -> dict:
    """2D circle (algebraic start, geometric refinement). Returns center, radius, rms, radius_se, points_used."""
    Q = np.asarray(xy, float)
    if len(Q) < 5:
        raise ValueError("A circle fit needs at least 5 points")
    A = np.c_[2 * Q, np.ones(len(Q))]
    sol, *_ = np.linalg.lstsq(A, (Q ** 2).sum(axis=1), rcond=None)
    c0 = sol[:2]
    r0 = math.sqrt(max(sol[2] + c0 @ c0, 1e-300))

    def residual(x):
        return np.linalg.norm(Q - x[:2], axis=1) - x[2]

    x, res, w, J = _weighted_refine(residual, np.r_[c0, r0], robust)
    cov = _covariance(J, res, w)
    inl = w > 0
    return {"center": x[:2], "radius": float(x[2]), "rms": float(np.sqrt(np.mean(res[inl] ** 2))),
            "radius_se": float(math.sqrt(max(cov[2, 2], 0))), "points_used": int(inl.sum())}


def fit_sphere(points, robust: bool = True, max_points: int = 200_000, seed: int = 0) -> dict:
    """Sphere (algebraic start, geometric Tukey-reweighted refinement).
    Returns center, radius, diameter, rms, center_se, radius_se, points_used, inlier_fraction."""
    P = np.asarray(points, float)
    if len(P) < 10:
        raise ValueError(f"A sphere fit needs at least 10 points (the selection has {len(P)})")
    if len(P) > max_points:
        P = P[np.sort(np.random.default_rng(seed).choice(len(P), max_points, replace=False))]
    off = P.mean(axis=0)
    X = P - off
    A = np.c_[2 * X, np.ones(len(X))]
    sol, *_ = np.linalg.lstsq(A, (X ** 2).sum(axis=1), rcond=None)
    c0 = sol[:3]
    r0 = math.sqrt(max(sol[3] + c0 @ c0, 1e-300))

    def residual(x):
        return np.linalg.norm(X - x[:3], axis=1) - x[3]

    x, res, w, J = _weighted_refine(residual, np.r_[c0, r0], robust)
    cov = _covariance(J, res, w)
    inl = w > 0
    return {"center": x[:3] + off, "radius": float(x[3]), "diameter": float(2 * x[3]),
            "rms": float(np.sqrt(np.mean(res[inl] ** 2))), "center_se": np.sqrt(np.maximum(np.diag(cov)[:3], 0)),
            "radius_se": float(math.sqrt(max(cov[3, 3], 0))), "points_used": int(inl.sum()),
            "inlier_fraction": float(inl.mean())}


def _circle_algebraic_rel(xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    A = np.c_[2 * xy, np.ones(len(xy))]
    sol, *_ = np.linalg.lstsq(A, (xy ** 2).sum(axis=1), rcond=None)
    c = sol[:2]
    r = math.sqrt(max(sol[2] + c @ c, 1e-300))
    res = np.linalg.norm(xy - c, axis=1) - r
    return c, r, _mad_sigma(res) / max(r, 1e-300)


def fit_cylinder(points, normals=None, axis_hint=None, robust: bool = True, max_points: int = 150_000,
                 seed: int = 0) -> dict:
    """Cylinder (axis candidates from the normals and the point spread, the most circular cross-section wins, then a
    geometric Tukey-reweighted refinement of axis tilt, axis position and radius).

    Returns point (on the axis, middle of the covered length), axis, radius, diameter, rms, radius_se, length,
    coverage_deg, points_used, inlier_fraction."""
    P = np.asarray(points, float)
    if len(P) < 12:
        raise ValueError(f"A cylinder fit needs at least 12 points (the selection has {len(P)})")
    rng = np.random.default_rng(seed)
    if len(P) > max_points:
        idx = np.sort(rng.choice(len(P), max_points, replace=False))
        P = P[idx]
        normals = None if normals is None else np.asarray(normals, float)[idx]
    off = P.mean(axis=0)
    X = P - off
    extent = float(np.linalg.norm(np.ptp(X, axis=0))) or 1.0
    cands = []
    if axis_hint is not None:
        cands.append(_unit(axis_hint, "axis_hint"))
    if normals is not None and len(normals) == len(P):
        nn = np.asarray(normals, float)
        nn = nn / np.maximum(np.linalg.norm(nn, axis=1), 1e-300)[:, None]
        cands += list(np.linalg.eigh(nn.T @ nn)[1].T)
    cands += list(np.linalg.eigh(np.cov(X.T))[1].T)
    best = None
    for a in cands:
        a = _unit(a)
        u, v = _basis(a)
        c2, r, rel = _circle_algebraic_rel(np.c_[X @ u, X @ v])
        if not np.isfinite(r) or r > 50 * extent:
            continue
        if best is None or rel < best[0]:
            best = (rel, a, u, v, c2[0] * u + c2[1] * v, r)
    if best is None:
        raise ValueError("No cylindrical surface found in the selection")
    _, a0, u0, v0, c0, r0 = best

    def frame(x):
        a = a0 + x[0] * u0 + x[1] * v0
        a = a / np.linalg.norm(a)
        return a, c0 + x[2] * u0 + x[3] * v0

    def residual(x):
        a, c = frame(x)
        d = X - c
        return np.linalg.norm(d - np.outer(d @ a, a), axis=1) - x[4]

    x, res, w, J = _weighted_refine(residual, np.array([0, 0, 0, 0, r0], float), robust)
    cov = _covariance(J, res, w)
    a, c = frame(x)
    inl = w > 0
    t = (X - c) @ a
    tmid = 0.5 * (t[inl].min() + t[inl].max())
    q = (X - c) - np.outer(t, a)
    u, v = _basis(a)
    theta = np.arctan2(q[inl] @ v, q[inl] @ u)
    coverage = np.unique(np.floor((theta + math.pi) / (2 * math.pi) * 36).astype(int)).size * 10.0
    radius = abs(float(x[4]))
    return {"point": off + c + tmid * a, "axis": a, "radius": radius, "diameter": 2 * radius,
            "rms": float(np.sqrt(np.mean(res[inl] ** 2))), "radius_se": float(math.sqrt(max(cov[4, 4], 0))),
            "length": float(np.ptp(t[inl])), "coverage_deg": float(coverage), "points_used": int(inl.sum()),
            "inlier_fraction": float(inl.mean())}


# --------------------------------------------------------------------------- thread (helix) fit
def _thread_frame(X, a0, u0, v0, c0, x):
    a = a0 + x[0] * u0 + x[1] * v0
    a = a / np.linalg.norm(a)
    c = c0 + x[2] * u0 + x[3] * v0
    d = X - c
    t = d @ a
    q = d - t[:, None] * a
    r = np.linalg.norm(q, axis=1)
    u = u0 - (u0 @ a) * a
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return a, c, u, v, t, r, np.arctan2(q @ v, q @ u)


def _thread_design(t, theta, pitch, hand, harmonics):
    phi = 2 * math.pi * t / pitch - hand * theta
    mp = phi[:, None] * np.arange(1, harmonics + 1)
    return np.hstack([np.ones((len(t), 1)), np.cos(mp), np.sin(mp)])


def _thread_profile_fit(t, theta, r, pitch, hand, harmonics):
    A = _thread_design(t, theta, pitch, hand, harmonics)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    return coef, r - A @ coef


def _thread_model(phi, coef):
    m = np.arange(1, (len(coef) - 1) // 2 + 1)
    h = len(m)
    mp = np.asarray(phi, float)[:, None] * m
    return coef[0] + np.cos(mp) @ coef[1:1 + h] + np.sin(mp) @ coef[1 + h:]


def _thread_model_dphi(phi, coef):
    m = np.arange(1, (len(coef) - 1) // 2 + 1)
    h = len(m)
    mp = np.asarray(phi, float)[:, None] * m
    return -np.sin(mp) @ (m * coef[1:1 + h]) + np.cos(mp) @ (m * coef[1 + h:])


def _spectral_pitch(t, theta, r, spacing) -> float | None:
    """Coarse pitch from the spectrum of the radius along the axis in narrow angular sectors (20 deg): across a
    sector the helix moves the crests by only 1/18 of the pitch, while a whole turn would smear them out."""
    length = float(np.ptp(t))
    if length <= 0:
        return None
    b = max(spacing, length / 4096)
    nb = int(math.ceil(length / b)) + 1
    n_fft = 1 << int(math.ceil(math.log2(nb * 8)))
    power = np.zeros(n_fft // 2 + 1)
    used = 0
    sector = np.floor((theta + math.pi) / (2 * math.pi / 18)).astype(int) % 18
    t0 = float(t.min())
    for k in range(18):
        g = sector == k
        if g.sum() < 200:
            continue
        idx = np.clip(((t[g] - t0) / b).astype(int), 0, nb - 1)
        cnt = np.bincount(idx, minlength=nb)
        valid = cnt > 0
        if valid.sum() < max(8, nb // 8):
            continue
        xs = np.flatnonzero(valid)
        prof = np.bincount(idx, weights=r[g], minlength=nb)[valid] / cnt[valid]
        seg = np.interp(np.arange(xs[0], xs[-1] + 1), xs, prof)
        ii = np.arange(len(seg))
        seg = seg - np.polyval(np.polyfit(ii, seg, 1), ii)
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=n_fft)) ** 2
        if spec.sum() > 0:
            power += spec / spec.sum()
            used += 1
    if not used:
        return None
    freqs = np.fft.rfftfreq(n_fft, d=b)
    allowed = np.flatnonzero((freqs >= 2.5 / length) & (freqs <= 1 / max(4 * b, 0.05)))
    if len(allowed) < 3:
        return None
    i = allowed[int(np.argmax(power[allowed]))]
    f = freqs[i]
    if 0 < i < len(power) - 1:
        y0, y1, y2 = np.log(power[i - 1:i + 2] + 1e-300)
        den = y0 - 2 * y1 + y2
        if den < 0:
            f += 0.5 * (y0 - y2) / den * (freqs[1] - freqs[0])
    return 1 / f if f > 0 else None


def _interval_centre(phi_grid, above) -> float:
    """Centre phase of the longest circular run where `above` is True."""
    n = len(above)
    if above.all():
        return 0.0
    start = int(np.argmin(above))  # a False position: runs never wrap past it
    rolled = np.roll(above, -start)
    best, run_start, best_start = 0, None, 0
    for i in range(n + 1):
        if i < n and rolled[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start > best:
                best, best_start = i - run_start, run_start
            run_start = None
    centre = (start + best_start + (best - 1) / 2) % n
    return float(phi_grid[0] + centre * (phi_grid[1] - phi_grid[0]))


def fit_thread(points, normals=None, pitch_hint: float | None = None, harmonics: int = 10,
               fit_points: int = 40_000, segments: int = 0, seed: int = 1, log: Log = _quiet) -> dict:
    """Least-squares helix fit of a screw thread (external or internal).

    The axis, pitch and handedness come from a helix model (Fourier profile of the thread phase) fitted to the raw
    points; nothing is binned, so the pitch does not depend on how the scanner sampled the surface. Diameters are read
    from the raw points at the crest / root phase (major / minor) and from the model profile (pitch diameter = the
    radius where ridge and groove are equally wide). The lead is also fitted per axial segment: `lead` lists the
    axial offset of each segment from the ideal helix, which shows non-uniform axial scale (a straight trend = a
    pitch error, bends = local distortion)."""
    P = np.asarray(points, float)
    if len(P) < 500:
        raise ValueError(f"The thread selection has only {len(P)} points - select more of the threaded surface")
    off = P.mean(axis=0)
    X = P - off
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(X), size=min(fit_points, len(X)), replace=False))
    Xs = X[sub]
    ns = None if normals is None else np.asarray(normals, float)[sub]
    spacing = estimate_spacing(P) or float(np.linalg.norm(np.ptp(X, axis=0))) * 1e-4
    cyl = fit_cylinder(Xs, ns, robust=True, max_points=len(Xs))
    a0 = cyl["axis"]
    u0, v0 = _basis(a0)
    c0 = cyl["point"]  # in the same (centred) coordinates as Xs
    _, _, _, _, t, r, theta = _thread_frame(Xs, a0, u0, v0, c0, np.zeros(4))
    p0 = float(pitch_hint) if pitch_hint else _spectral_pitch(t, theta, r, spacing)
    if not p0:
        raise ValueError("No thread found: the selection is too short along its axis to show a repeating profile")
    grid = p0 * (1 + np.linspace(-0.1, 0.1, 81))
    costs = {h: np.array([np.mean(_thread_profile_fit(t, theta, r, p, h, 6)[1] ** 2) for p in grid])
             for h in (1, -1)}
    hand = 1 if costs[1].min() <= costs[-1].min() else -1
    c = costs[hand]
    i = int(np.argmin(c))
    pitch = float(grid[i])
    if 0 < i < len(grid) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        if den > 0:
            pitch += 0.5 * (c[i - 1] - c[i + 1]) / den * (grid[1] - grid[0])
    hand_ratio = float(costs[-hand].min() / max(costs[hand].min(), 1e-300))

    def fun(x):
        _, _, _, _, tt, rr, th = _thread_frame(Xs, a0, u0, v0, c0, x[:4])
        return _thread_profile_fit(tt, th, rr, pitch + x[4], hand, harmonics)[1]

    res0 = fun(np.zeros(5))
    s0 = _mad_sigma(res0) + 1e-9
    sol = least_squares(fun, np.zeros(5), loss="soft_l1", f_scale=2 * s0,
                        x_scale=[1e-3, 1e-3, spacing, spacing, pitch * 1e-3])
    x = sol.x
    pitch_fit = float(pitch + x[4])
    sigma = _mad_sigma(sol.fun) + 1e-12
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * sigma ** 2
        pitch_se = float(math.sqrt(max(cov[4, 4], 0.0)))
        axis_se_deg = float(np.degrees(math.sqrt(max(cov[0, 0] + cov[1, 1], 0.0))))
    except (np.linalg.LinAlgError, ValueError):
        pitch_se, axis_se_deg = None, None
    a, cc, u, v, _, _, _ = _thread_frame(Xs, a0, u0, v0, c0, x[:4])

    # every point in the final frame; profile coefficients from all inliers (normal equations in chunks)
    _, _, u, v, t_all, r_all, th_all = _thread_frame(X, a, u, v, cc, np.zeros(4))
    nc = 1 + 2 * harmonics
    ata, atb = np.zeros((nc, nc)), np.zeros(nc)
    for s in range(0, len(X), 200_000):
        sl = slice(s, s + 200_000)
        A = _thread_design(t_all[sl], th_all[sl], pitch_fit, hand, harmonics)
        ata += A.T @ A
        atb += A.T @ r_all[sl]
    coef = np.linalg.solve(ata + 1e-12 * np.eye(nc), atb)
    phi_all = 2 * math.pi * t_all / pitch_fit - hand * th_all
    resid = r_all - _thread_model(phi_all, coef)
    sigma_all = _mad_sigma(resid) + 1e-12
    inl = np.abs(resid) <= 4 * sigma_all
    r2 = float(1 - np.var(resid[inl]) / max(np.var(r_all[inl]), 1e-300))
    # Which part of the thread profile was actually scanned: seen from one side, the flanks facing away from the
    # scanner are hidden under the crests. There the Fourier model is an extrapolation and must not be read.
    nbins = 64
    wrapped = np.mod(phi_all, 2 * math.pi)
    counts = np.bincount((wrapped[inl] / (2 * math.pi) * nbins).astype(int) % nbins, minlength=nbins)
    covered = counts >= max(5.0, 0.1 * float(np.median(counts[counts > 0])) if (counts > 0).any() else 5.0)
    profile_coverage = float(covered.mean())
    grid_phi = np.linspace(0, 2 * math.pi, 2048, endpoint=False)
    grid_cov = covered[(grid_phi / (2 * math.pi) * nbins).astype(int) % nbins]
    model = _thread_model(grid_phi, coef)
    seen = model[grid_cov] if grid_cov.any() else model
    depth = float(np.ptp(seen))
    if r2 < 0.3 or depth < 3 * sigma_all or depth < 0.5 * spacing:
        raise ValueError("No thread found: the surface has no repeating helical profile clearly above the scan noise "
                         f"(profile depth {depth:.3f} vs noise {sigma_all:.3f})")
    top = seen.max() - 0.1 * depth
    bottom = seen.min() + 0.1 * depth
    phi_crest = _interval_centre(grid_phi, grid_cov & (model >= top))
    phi_root = _interval_centre(grid_phi, grid_cov & (model <= bottom))

    def near(phase):
        dphi = np.angle(np.exp(1j * (phi_all - phase)))
        return inl & (np.abs(dphi) <= math.pi / 16)

    crest_pts, root_pts = near(phi_crest), near(phi_root)
    major = 2 * float(np.median(r_all[crest_pts])) if crest_pts.sum() >= 20 else 2 * float(seen.max())
    minor = 2 * float(np.median(r_all[root_pts])) if root_pts.sum() >= 20 else 2 * float(seen.min())
    # the pitch diameter needs both flanks: where ridge and groove are equally wide
    both_flanks = profile_coverage >= 0.9
    pitch_diameter = 2 * float(np.median(model)) if both_flanks else None
    # Along +axis the phase grows, so where the profile falls with phase the flank's normal points along +axis.
    # Which flank is missing tells the user from which end the thread was not scanned.
    slope = np.gradient(model)
    band = (model > seen.min() + 0.25 * depth) & (model < seen.max() - 0.25 * depth)
    seen_plus = float(grid_cov[band & (slope < 0)].mean()) if (band & (slope < 0)).any() else 0.0
    seen_minus = float(grid_cov[band & (slope > 0)].mean()) if (band & (slope > 0)).any() else 0.0
    missing_flank = None
    if min(seen_plus, seen_minus) < 0.5:
        missing_flank = "+axis" if seen_plus < seen_minus else "-axis"
    half_angles = []
    for sign, seen_frac in ((-1, seen_plus), (1, seen_minus)):
        m = band & (np.sign(slope) == sign)
        if seen_frac >= 0.5 and m.any():
            drdt = float(np.median(slope[m])) / ((grid_phi[1] - grid_phi[0]) * pitch_fit / (2 * math.pi))
            half_angles.append(float(np.degrees(math.atan(1 / max(abs(drdt), 1e-9)))))
    kind = "unknown"
    if normals is not None:
        radial = X[sub] - cc - np.outer((X[sub] - cc) @ a, a)
        radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-300)[:, None]
        dots = np.einsum("ij,ij->i", np.asarray(normals, float)[sub], radial)
        med = float(np.median(dots))
        kind = "external" if med > 0.3 else "internal" if med < -0.3 else "unknown"

    # lead (axial position of the thread) and radius per axial segment, relative to the global helix model
    length = float(np.ptp(t_all[inl]))
    nseg = segments or int(np.clip(length / (2 * pitch_fit), 2, 24))
    edges = np.linspace(t_all[inl].min(), t_all[inl].max(), nseg + 1)
    lead = []
    for k in range(nseg):
        m = inl & (t_all >= edges[k]) & (t_all <= edges[k + 1])
        if m.sum() < 200:
            continue
        idx = np.flatnonzero(m)
        if len(idx) > 60_000:
            idx = np.sort(rng.choice(idx, 60_000, replace=False))
        dt, dr = 0.0, 0.0
        for _ in range(4):
            ph = 2 * math.pi * (t_all[idx] - dt) / pitch_fit - hand * th_all[idx]
            f = _thread_model(ph, coef) + dr
            g = _thread_model_dphi(ph, coef) * (2 * math.pi / pitch_fit)
            rr = r_all[idx] - f
            keep = np.abs(rr) <= 4 * sigma_all
            A = np.c_[-g[keep], np.ones(keep.sum())]
            step, *_ = np.linalg.lstsq(A, rr[keep], rcond=None)
            dt, dr = dt + float(step[0]), dr + float(step[1])
        lead.append({"t": float(0.5 * (edges[k] + edges[k + 1])), "axial_offset": dt, "radial_offset": dr,
                     "points": int(m.sum())})
    lead_slope_ppm = None
    lead_residual = None
    if len(lead) >= 3:
        tt = np.array([s["t"] for s in lead])
        oo = np.array([s["axial_offset"] for s in lead])
        k1, k0 = np.polyfit(tt, oo, 1)
        lead_slope_ppm = float(k1 * 1e6)
        lead_residual = float(np.ptp(oo - (k1 * tt + k0)))
    q = (X[inl] - cc) - np.outer(t_all[inl], a)
    ang = np.arctan2(q @ v, q @ u)
    coverage = np.unique(np.floor((ang + math.pi) / (2 * math.pi) * 36).astype(int)).size * 10.0
    model_params = {"origin": off + cc, "axis": a, "u": u, "v": v, "pitch": pitch_fit, "hand": hand, "coef": coef,
                    "noise": float(sigma_all), "covered_bins": covered, "crest_phase": phi_crest, "root_phase": phi_root}
    result = {"pitch": pitch_fit, "pitch_se": pitch_se, "tpi": 25.4 / pitch_fit, "model": model_params,
              "handedness": "right" if hand > 0 else "left", "hand_ratio": hand_ratio,
              "handedness_confident": bool(hand_ratio > 1.5 and coverage >= 60),
              "kind": kind, "axis": a, "center": off + cc + a * 0.5 * (t_all[inl].min() + t_all[inl].max()),
              "axis_se_deg": axis_se_deg, "major_diameter": major, "minor_diameter": minor,
              "pitch_diameter": pitch_diameter, "profile_coverage": profile_coverage, "both_flanks": both_flanks,
              "missing_flank": missing_flank, "flank_half_angles_deg": half_angles if len(half_angles) == 2 else None,
              "flank_angle_deg": sum(half_angles) if len(half_angles) == 2 else None,
              "depth": depth, "noise": float(sigma_all), "fit_r2": r2,
              "length": length, "coverage_deg": float(coverage), "points_used": int(inl.sum()),
              "inlier_fraction": float(inl.mean()), "spacing": float(spacing),
              "profile": {"phase": grid_phi[::16], "radius": model[::16]}, "lead": lead,
              "lead_slope_ppm": lead_slope_ppm, "lead_residual_pp": lead_residual}
    if not both_flanks:
        result["warnings"] = [f"Only {100 * profile_coverage:.0f} % of the thread profile was scanned (one flank is "
                              "hidden when a thread is scanned from one side): no pitch diameter. Scan it from both "
                              "ends, or measure a merge of such scans."]
    pd_text = f"{pitch_diameter:.4f}" if pitch_diameter is not None else "n/a (one flank not scanned)"
    log(f"Thread fit: pitch {pitch_fit:.5f} (se {pitch_se or 0:.5f}), {result['handedness']}-hand, major "
        f"{major:.4f}, pitch dia {pd_text}, minor {minor:.4f}, noise {sigma_all:.4f}, profile coverage "
        f"{100 * profile_coverage:.0f} %")
    return result


def thread_offsets(model: dict, points, edges) -> list[dict]:
    """Axial and radial offset of `points` from a fitted thread `model` (fit_thread()["model"]) in axial segments
    [edges[k], edges[k+1]] (axial coordinate from the model origin). Evaluating two aligned scans against the same
    model shows where along the thread they differ: equal offsets = same thread; a trend in the axial offset = a
    scale difference along the axis; a radial offset = one scan measures the thread thicker."""
    X = np.asarray(points, float) - model["origin"]
    a, u, v = model["axis"], model["u"], model["v"]
    pitch, hand, coef = model["pitch"], model["hand"], np.asarray(model["coef"], float)
    t = X @ a
    q = X - np.outer(t, a)
    r = np.linalg.norm(q, axis=1)
    th = np.arctan2(q @ v, q @ u)
    sigma = float(model.get("noise") or _mad_sigma(r - _thread_model(2 * math.pi * t / pitch - hand * th, coef)))
    usable = np.ones(len(t), bool)
    covered = model.get("covered_bins")
    if covered is not None:  # only where the model was fitted to data (not across a flank that was never scanned)
        covered = np.asarray(covered, bool)
        nb = len(covered)
        ph = np.mod(2 * math.pi * t / pitch - hand * th, 2 * math.pi)
        usable = covered[(ph / (2 * math.pi) * nb).astype(int) % nb]
    out = []
    for k in range(len(edges) - 1):
        m = np.flatnonzero((t >= edges[k]) & (t < edges[k + 1]) & usable)
        if len(m) < 200:
            out.append({"t": float(0.5 * (edges[k] + edges[k + 1])), "axial_offset": None, "radial_offset": None,
                        "points": int(len(m))})
            continue
        # coarse search over one pitch (the linearisation below only converges within ~P/8)
        trial = np.linspace(-pitch / 2, pitch / 2, 65)[:-1]
        cost = []
        for dt in trial:
            rr = r[m] - _thread_model(2 * math.pi * (t[m] - dt) / pitch - hand * th[m], coef)
            cost.append(np.median(np.abs(rr - np.median(rr))))
        dt, dr = float(trial[int(np.argmin(cost))]), 0.0
        for _ in range(6):
            ph = 2 * math.pi * (t[m] - dt) / pitch - hand * th[m]
            f = _thread_model(ph, coef) + dr
            g = _thread_model_dphi(ph, coef) * (2 * math.pi / pitch)
            rr = r[m] - f
            keep = np.abs(rr - np.median(rr)) <= 6 * max(sigma, _mad_sigma(rr))
            A = np.c_[-g[keep], np.ones(keep.sum())]
            step, *_ = np.linalg.lstsq(A, rr[keep], rcond=None)
            dt, dr = dt + float(step[0]), dr + float(step[1])
        out.append({"t": float(0.5 * (edges[k] + edges[k + 1])), "axial_offset": dt, "radial_offset": dr,
                    "points": int(len(m))})
    return out


# --------------------------------------------------------------------------- frames and dimensions
def principal_frame(points) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, axes, spread): axes rows = length, width, height directions (PCA, longest first), right-handed;
    each of the first two axes points towards the heavier tail of the part so the frame is reproducible."""
    P = np.asarray(points, float)
    if len(P) < 3:
        raise ValueError("Need at least 3 points for a part frame")
    c = P.mean(axis=0)
    X = P - c
    w, V = np.linalg.eigh(X.T @ X / len(X))
    order = np.argsort(w)[::-1]
    axes = V[:, order].T.copy()
    for k in range(2):
        if np.mean((X @ axes[k]) ** 3) < 0:
            axes[k] = -axes[k]
    axes[2] = np.cross(axes[0], axes[1])
    return c, axes, np.sqrt(np.maximum(w[order], 0.0))


def refine_axes(points, center, axes, trim: float = DEFAULT_TRIM, sample: int = 100_000, seed: int = 0) -> np.ndarray:
    """Turn principal axes into the axes of the smallest (trimmed) bounding box, pair by pair.

    Principal axes of a part with two similar extents are poorly determined: random sampling alone tilts those of a
    40 x 30 mm box by ~0.8 deg, which lengthens its extents by ~0.4 mm. Calipers settle on the smallest reading;
    so does this (coarse-to-fine search of the rotation in each axis plane that minimises the product of the two
    trimmed extents). Returns axes sorted longest first, right-handed."""
    X = np.asarray(points, float) - center
    if len(X) > sample:
        X = X[np.random.default_rng(seed).choice(len(X), sample, replace=False)]
    A = np.asarray(axes, float).copy()
    q = (trim, 1 - trim) if trim > 0 else (0.0, 1.0)

    def extent(v):
        s = X @ v
        lo, hi = np.quantile(s, q)
        return hi - lo

    for _ in range(2):
        for i, j in ((0, 1), (0, 2), (1, 2)):
            best, best_area = 0.0, extent(A[i]) * extent(A[j])
            for span, steps in ((math.radians(8), 33), (math.radians(0.6), 25), (math.radians(0.04), 17)):
                for ang in best + np.linspace(-span, span, steps):
                    c, s = math.cos(ang), math.sin(ang)
                    area = extent(c * A[i] + s * A[j]) * extent(-s * A[i] + c * A[j])
                    if area < best_area - 1e-12:
                        best, best_area = float(ang), area
            c, s = math.cos(best), math.sin(best)
            A[i], A[j] = c * A[i] + s * A[j], -s * A[i] + c * A[j]
    order = np.argsort([-extent(a) for a in A])
    A = A[order]
    A[2] = np.cross(A[0], A[1])
    return A


def resolve_direction(direction, axes=None) -> np.ndarray:
    """'x' | 'y' | 'z' | 'length' | 'width' | 'height' | [x, y, z] -> unit vector (part axes need `axes`)."""
    if isinstance(direction, str):
        key = direction.strip().lower()
        if key in ("x", "y", "z"):
            return np.eye(3)["xyz".index(key)]
        if key in AXIS_NAMES:
            if axes is None:
                raise ValueError(f"'{direction}' needs the part frame")
            return np.asarray(axes, float)[AXIS_NAMES.index(key)]
        raise ValueError("The direction must be 'x', 'y', 'z', 'length', 'width', 'height' or a vector [x, y, z]")
    return _unit(direction, "The direction")


def robust_dimensions(geom_or_points, trim: float = DEFAULT_TRIM, frame=None, max_points: int = 2_000_000,
                      seed: int = 0, refine: bool = True) -> dict:
    """Part size as a person with calipers would read it: extents along the part's axes (length >= width >=
    height), with `trim` of the points ignored at each end of every axis so a few stray points cannot inflate it.
    The axes are the principal axes turned to the smallest trimmed box (`refine_axes`).

    `frame` = (centre, axes) measures along given axes instead (e.g. a parent's frame, to compare two assets).
    Also returns the raw extents along the same axes and the axis-aligned box in scanner coordinates (what a
    viewer shows as X / Y / Z), which depends on how the part lay and is not a part dimension.

    Noise pushes trimmed extents outwards (about 2.5 noise sigma per end on a flat face); `measure_length` (planes
    fitted to the end faces) has no such bias."""
    if isinstance(geom_or_points, (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh)):
        pts, _ = surface_points(geom_or_points, max_points, seed)
        allp = geometry_points(geom_or_points)
    else:
        pts = allp = np.asarray(geom_or_points, float)
    if len(pts) < 3:
        raise ValueError("Need at least 3 points to measure dimensions")
    if not 0 <= trim < 0.5:
        raise ValueError("trim must be between 0 and 0.5")
    if frame is None:
        centre, axes, _ = principal_frame(pts)
        if refine:
            axes = refine_axes(pts, centre, axes, trim, seed=seed)
    else:
        centre, axes = np.asarray(frame[0], float), np.asarray(frame[1], float)
    Q = (pts - centre) @ axes.T
    lo = np.quantile(Q, trim, axis=0) if trim > 0 else Q.min(axis=0)
    hi = np.quantile(Q, 1 - trim, axis=0) if trim > 0 else Q.max(axis=0)
    R = (allp - centre) @ axes.T
    rlo, rhi = R.min(axis=0), R.max(axis=0)
    size, raw = hi - lo, rhi - rlo
    amin, amax = allp.min(axis=0), allp.max(axis=0)
    return {"length": float(size[0]), "width": float(size[1]), "height": float(size[2]), "trim": trim,
            "raw": {n: float(raw[i]) for i, n in enumerate(AXIS_NAMES)},
            "frame": {"center": centre, "axes": axes, "origin": centre + lo @ axes},
            "min": lo, "max": hi, "points_used": int(len(pts)),
            "aabb": {"min": amin, "max": amax, "size": amax - amin}}


def open3d_obb_extent(geom: Geometry, sample: int = 200_000) -> list[float]:
    """The 'fitted' box CloudClean's asset stats report (Open3D oriented box of a 200k sample), for comparison."""
    pts = geometry_points(geom)
    if len(pts) > sample:
        pts = pts[np.random.default_rng(0).choice(len(pts), sample, replace=False)]
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts))
    return sorted(np.asarray(obb.extent).tolist(), reverse=True)


# --------------------------------------------------------------------------- surface distances
def _quadric_rows(u, v):
    return np.stack([u * u, u * v, v * v, u, v, np.ones_like(u)], axis=-1)


class SurfaceModel:
    """Signed distance of query points to a surface.

    Clouds: a quadric height field is fitted to the k nearest scan points of each query in their local PCA frame
    (moving least squares), so curved features (thread crests, fillets) are not biased as with a tangent plane;
    the sign comes from the scan's normals (estimated and oriented outward when it has none). A query only gets a
    value when it lies over the fitted patch (not beyond the edge of the scanned area).
    Meshes: exact closest point on the triangles, sign from the outward face normal."""

    def __init__(self, geom: Geometry, k: int = 16, max_points: int | None = None, seed: int = 0,
                 log: Log = _quiet):
        self.is_cloud = is_cloud(geom)
        if self.is_cloud:
            pts, nrm = surface_points(geom, max_points or 10 ** 12, seed)
            if nrm is None or len(nrm) != len(pts):
                nrm = _estimate_normals(pts, log)
            self.points = pts
            self.normals = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
            self.tree = cKDTree(pts)
            self.k = int(min(k, len(pts)))
            self.spacing = estimate_spacing(pts)
        else:
            V, F = _mesh_arrays(geom)
            if len(F) == 0:
                raise ValueError("The mesh has no triangles")
            cross = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
            area2 = np.maximum(np.linalg.norm(cross, axis=1), 1e-300)
            self.tri_normals = _outward_sign(V, F, cross) * cross / area2[:, None]
            self.offset = V.mean(axis=0)
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(o3d.core.Tensor((V - self.offset).astype(np.float32)),
                                     o3d.core.Tensor(F.astype(np.uint32)))
            e = F[np.random.default_rng(seed).choice(len(F), min(len(F), 100_000), replace=False)]
            self.spacing = float(np.median(np.linalg.norm(V[e[:, 1]] - V[e[:, 0]], axis=1)))

    def query(self, q, normals=None, max_angle_deg: float = 30.0, chunk: int = 100_000):
        """(signed distance, unit normal at the foot point, gap = distance to the nearest scan point / surface,
        ok = a trustworthy value) for each query point.

        With `normals` (the query points' own outward normals) a point only counts where both surfaces face the same
        way (within max_angle_deg): it keeps, e.g., a thread flank that only one scan saw from being measured
        against the crest next to it in the other scan."""
        q = np.asarray(q, float).reshape(-1, 3)
        n = len(q)
        d, nrm = np.full(n, np.nan), np.zeros((n, 3))
        gap, ok = np.full(n, np.inf), np.zeros(n, bool)
        for s in range(0, n, chunk):
            sl = slice(s, min(s + chunk, n))
            d[sl], nrm[sl], gap[sl], ok[sl] = (self._cloud if self.is_cloud else self._mesh)(q[sl])
        if normals is not None:
            qn = np.asarray(normals, float).reshape(-1, 3)
            qn = qn / np.maximum(np.linalg.norm(qn, axis=1), 1e-300)[:, None]
            ok &= np.einsum("ij,ij->i", qn, nrm) >= math.cos(math.radians(max_angle_deg))
        return d, nrm, gap, ok

    def _cloud(self, q):
        return self._cloud_fit(q)[:4]

    def _cloud_fit(self, q):
        """Local quadric fit around each query: (signed distance, foot normal, gap, ok, residual sigma of the fit,
        leverage of the query position)."""
        dist, idx = self.tree.query(q, k=self.k, workers=-1)
        nb = self.points[idx]
        c = nb.mean(axis=1)
        X = nb - c[:, None, :]
        vec = np.linalg.eigh(np.einsum("mki,mkj->mij", X, X))[1]
        n, e1 = vec[:, :, 0].copy(), vec[:, :, 2].copy()
        flip = np.einsum("mki,mi->m", self.normals[idx], n) < 0
        n[flip] *= -1
        e2 = np.cross(n, e1)
        scale = np.maximum(dist[:, -1], 1e-12)
        u = np.einsum("mki,mi->mk", X, e1) / scale[:, None]
        v = np.einsum("mki,mi->mk", X, e2) / scale[:, None]
        h = np.einsum("mki,mi->mk", X, n)
        A = _quadric_rows(u, v)
        ata = np.einsum("mki,mkj->mij", A, A) + 1e-9 * np.eye(6)
        coef = np.linalg.solve(ata, np.einsum("mki,mk->mi", A, h)[..., None])[..., 0]
        rel = q - c
        uq = np.einsum("mi,mi->m", rel, e1) / scale
        vq = np.einsum("mi,mi->m", rel, e2) / scale
        hq = np.einsum("mi,mi->m", rel, n)
        hfit = np.einsum("mi,mi->m", _quadric_rows(uq, vq), coef)
        gu = (2 * coef[:, 0] * uq + coef[:, 1] * vq + coef[:, 3]) / scale
        gv = (coef[:, 1] * uq + 2 * coef[:, 2] * vq + coef[:, 4]) / scale
        normal = n - gu[:, None] * e1 - gv[:, None] * e2
        norm = np.linalg.norm(normal, axis=1)
        normal /= np.maximum(norm, 1e-300)[:, None]
        d = (hq - hfit) / np.maximum(norm, 1e-300)
        ok = (uq * uq + vq * vq <= 1.0) & np.isfinite(d) & (np.abs(gu) < 3) & (np.abs(gv) < 3)
        res = h - np.einsum("mki,mi->mk", A, coef)
        sigma = np.sqrt((res * res).sum(axis=1) / max(self.k - 6, 1))
        aq = _quadric_rows(uq, vq)
        leverage = np.einsum("mi,mi->m", aq, np.linalg.solve(ata, aq[..., None])[..., 0])
        return d, normal, dist[:, 0], ok, sigma, np.clip(leverage, 0.0, None)

    def project(self, q, iterations: int = 3) -> dict:
        """Move points onto the surface.

        Clouds: onto the local quadric of the k nearest scan points (moving least squares), so a picked point sits on
        the fitted surface instead of on one noisy scan point; `uncertainty` is the 1-sigma error of that position
        along the normal (fit residual x sqrt(leverage)). Where no patch fits (far from the scan, at a scan border)
        the nearest scan point is used, ok = False, uncertainty = the local noise. Meshes: the exact closest point on
        the triangles (uncertainty None - a mesh carries no noise estimate)."""
        q0 = np.asarray(q, float).reshape(-1, 3)
        if not self.is_cloud:
            res = self.scene.compute_closest_points(o3d.core.Tensor((q0 - self.offset).astype(np.float32)))
            foot = res["points"].numpy().astype(float) + self.offset
            n = self.tri_normals[res["primitive_ids"].numpy().astype(np.int64)]
            return {"points": foot, "normals": n, "uncertainty": [None] * len(q0), "ok": np.ones(len(q0), bool),
                    "moved": np.linalg.norm(foot - q0, axis=1)}
        _, near = self.tree.query(q0, k=1, workers=-1)
        p = self.points[near].copy()           # start on the scan, where the patch is well defined
        ok = np.zeros(len(q0), bool)
        for _ in range(iterations):
            d, n, _, ok, sigma, lev = self._cloud_fit(p)
            step = np.where(ok, d, 0.0)
            p = p - step[:, None] * n
        d, n, _, ok, sigma, lev = self._cloud_fit(p)
        ok &= np.abs(d) < max(self.spacing, 1e-9)
        points = np.where(ok[:, None], p - np.where(ok, d, 0.0)[:, None] * n, self.points[near])
        unc = np.where(ok, sigma * np.sqrt(lev), sigma)
        normals = np.where(ok[:, None], n, self.normals[near])
        return {"points": points, "normals": normals, "uncertainty": unc.tolist(), "ok": ok,
                "moved": np.linalg.norm(points - q0, axis=1)}

    def _mesh(self, q):
        res = self.scene.compute_closest_points(o3d.core.Tensor((q - self.offset).astype(np.float32)))
        foot = res["points"].numpy().astype(float) + self.offset
        tri = res["primitive_ids"].numpy().astype(np.int64)
        n = self.tri_normals[tri]
        diff = q - foot
        dist = np.linalg.norm(diff, axis=1)
        sign = np.where(np.einsum("ij,ij->i", diff, n) < 0, -1.0, 1.0)
        return sign * dist, n, dist, np.ones(len(q), bool)


# --------------------------------------------------------------------------- operation drift
def transform_check(T) -> dict:
    """Is a 4x4 a rigid transform? Determinant, orthonormality error, singular values (scale) and motion."""
    M = np.asarray(T, float)
    if M.shape != (4, 4) or not np.isfinite(M).all():
        raise ValueError("The transform must be a finite 4x4 matrix")
    R = M[:3, :3]
    sv = np.linalg.svd(R, compute_uv=False)
    det = float(np.linalg.det(R))
    orth = float(np.abs(R.T @ R - np.eye(3)).max())
    affine_row = bool(np.allclose(M[3], [0, 0, 0, 1], atol=1e-9))
    rigid = bool(abs(det - 1) < 1e-6 and orth < 1e-6 and affine_row)
    scale = float(np.cbrt(abs(det))) if det != 0 else 0.0
    Rn = R / scale if scale > 0 else R
    return {"determinant": det, "orthonormality_error": orth, "singular_values": sv,
            "scale_ppm": (float(sv.mean()) - 1) * 1e6, "rotation_deg": rotation_angle_deg(Rn),
            "translation": M[:3, 3], "rigid": rigid, "mirrored": bool(det < 0),
            "identity": bool(np.allclose(M, np.eye(4), atol=1e-9))}


def _dimension_change(before: dict, after: dict) -> dict:
    return {n: after[n] - before[n] for n in AXIS_NAMES}


def trimmed_extents(points, frame, trim: float = DEFAULT_TRIM) -> dict:
    """Extents along the frame axes with `trim` of the points ignored at each end."""
    centre, axes = np.asarray(frame[0], float), np.asarray(frame[1], float)
    Q = (np.asarray(points, float) - centre) @ axes.T
    lo, hi = np.quantile(Q, [trim, 1 - trim], axis=0)
    return {n: float(hi[i] - lo[i]) for i, n in enumerate(AXIS_NAMES)}


def band_extents(points, frame, width: float, trim: float = DEFAULT_TRIM) -> dict:
    """Extents along the frame axes measured at the median position of the outermost band of surface (all points
    within `width` inside the trimmed extreme) at each end.

    Unlike trimmed extents this does not grow with noise, so a noisy cloud and a smooth mesh of the same surface
    measure the same (a flat end face is measured exactly; a rounded end about width/2 inside its tip, equally for
    both). Used to compare an asset with its parent."""
    centre, axes = np.asarray(frame[0], float), np.asarray(frame[1], float)
    Q = (np.asarray(points, float) - centre) @ axes.T
    out = {}
    for i, name in enumerate(AXIS_NAMES):
        s = Q[:, i]
        lo_t, hi_t = np.quantile(s, [trim, 1 - trim])
        hi_band = s[(s >= hi_t - width) & (s <= hi_t + width)]
        lo_band = s[(s <= lo_t + width) & (s >= lo_t - width)]
        out[name] = float(np.median(hi_band) - np.median(lo_band))
    return out


def operation_drift(parent: Geometry, child: Geometry, operation: str | None = None, transform=None,
                    max_points: int = 200_000, tolerance: float = 0.01, reach: float | None = None, seed: int = 0,
                    log: Log = _quiet) -> dict:
    """How far `child` (made from `parent` by `operation`) moved or resized the parent's surface.

    `transform` (4x4) maps the parent into the child's coordinates when the operation moved it (a merge pose, an
    edit transform); it is checked for rigidity (a merge must never scale) and applied first.

    * identical_fraction: child points that are exactly parent points (cleaning only removes points: 1.0)
    * kept_fraction: parent points still present in the child
    * displacement (child relative to the parent surface, + = outside): identical points count as 0, the others
      are measured against the parent's local surface. Child surface with no parent data within `reach`
      (new coverage from another scan, holes filled by a watertight mesh) is left out and reported as
      unsupported_fraction.
    * reverse (meshes): the parent's points measured against the child surface, sign flipped so + = child outside.
    * removed_surface_points / removed_stray_points (when the child is a subset): removed points that lay on the
      kept surface (e.g. an edge thinned by a filter) versus stray points off it.
    * dimensions_before / dimensions_after: extents in the parent's part frame; dimension_change. A subset is
      compared with its parent minus the stray points (trimmed extents, same points and noise); anything else with
      `band_extents`, which do not depend on the noise (a smooth mesh vs a noisy cloud).
    * verdict: unchanged (no systematic movement, no size change beyond `tolerance`; random motion within 3x the
      parent's noise is not a change) | moved (rigid pose change only) | changed."""
    from .register import estimate_noise

    op = operation or "unknown"
    T = None if transform is None else np.asarray(transform, float)
    tcheck = transform_check(T) if T is not None else None
    base = parent if T is None or tcheck["identity"] else transformed_geometry(parent, T)
    log(f"Drift of '{op}': {len(geometry_points(child)):,} child vs {len(geometry_points(parent)):,} parent points")

    identical_fraction = kept_fraction = None
    removed_surface = removed_stray = None
    extra_surface = None     # removed points that were real surface (subset children)
    child_pts, child_nrm = surface_points(child, 10 ** 12 if is_cloud(child) else max_points, seed)
    parent_pts, _ = surface_points(base, 10 ** 12 if is_cloud(base) else max_points, seed)
    noise = estimate_noise(parent_pts)
    spacing = estimate_spacing(parent_pts)
    identical = np.zeros(len(child_pts), bool)
    subset = False
    if is_cloud(parent) and is_cloud(child):
        P = parent_pts
        scale = max(1.0, float(np.abs(child_pts).max()), float(np.abs(P).max()))
        eps = 1e-6 * scale
        dist, _ = cKDTree(P).query(child_pts, k=1, workers=-1)
        identical = dist <= eps
        dist2, _ = cKDTree(child_pts).query(P, k=1, workers=-1)
        kept = dist2 <= eps
        identical_fraction = float(identical.mean())
        kept_fraction = float(kept.mean())
        subset = identical_fraction >= 0.9999
        if subset and (~kept).any():
            removed = P[~kept]
            dd, _, gap, ok = SurfaceModel(child, seed=seed).query(removed)
            on = ok & (gap <= 3 * spacing) & (np.abs(dd) <= max(3 * noise, 0.5 * spacing))
            removed_surface, removed_stray = int(on.sum()), int((~on).sum())
            extra_surface = removed[on]
    rng = np.random.default_rng(seed)
    sample = np.arange(len(child_pts))
    if len(sample) > max_points:
        sample = np.sort(rng.choice(len(sample), max_points, replace=False))
    d = np.zeros(len(sample))
    moved = ~identical[sample]
    surface = None
    unsupported = 0
    if moved.any():
        surface = SurfaceModel(base, seed=seed, log=log)
        reach = reach if reach is not None else 3.0 * max(surface.spacing, 1e-9)
        qn = None if child_nrm is None else child_nrm[sample[moved]]
        dd, _, gap, ok = surface.query(child_pts[sample[moved]], qn, 45.0)
        good = ok & (gap <= reach)
        unsupported = int((~good).sum())
        dm = np.full(moved.sum(), np.nan)
        dm[good] = dd[good]
        d[moved] = dm
    displacement = _stats(d) or _stats(np.zeros(1))
    moved_stats = _stats(d[moved]) if moved.any() else None
    unsupported_fraction = unsupported / max(len(sample), 1)

    reverse = None
    if not is_cloud(child):
        ppts = parent_pts
        if len(ppts) > max_points:
            ppts = ppts[np.sort(rng.choice(len(ppts), max_points, replace=False))]
        dd, _, _, _ = SurfaceModel(child).query(ppts)
        reverse = _stats(-dd)

    # sizes: the parent's frame; a subset's parent without its stray points; extents that ignore noise
    if subset:
        ref_pts = child_pts if extra_surface is None else np.vstack([child_pts, extra_surface])
    else:
        ref_pts = parent_pts
    frame_dims = robust_dimensions(ref_pts, seed=seed)
    frame = (frame_dims["frame"]["center"], frame_dims["frame"]["axes"])
    width = max(4 * spacing, 8 * noise)
    if subset:   # the same points before and after: trimmed extents compare like with like
        before, after = trimmed_extents(ref_pts, frame), trimmed_extents(child_pts, frame)
    else:        # different surfaces / noise levels: extents that do not depend on the noise
        before, after = band_extents(ref_pts, frame, width), band_extents(child_pts, frame, width)
    change = _dimension_change(before, after)
    big_change = max(abs(v) for v in change.values())
    random_limit = max(tolerance, 3 * noise)

    if subset:
        same = big_change <= tolerance
    else:
        same = (abs(displacement["signed_mean"]) <= tolerance and displacement["p95"] <= random_limit
                and big_change <= tolerance and unsupported_fraction <= 0.02
                and (reverse is None or (abs(reverse["signed_mean"]) <= tolerance
                                         and reverse["p95"] <= random_limit)))
    if tcheck is not None and not tcheck["rigid"]:
        verdict = "changed"
    elif same:
        verdict = "moved" if tcheck is not None and not tcheck["identity"] else "unchanged"
    else:
        verdict = "changed"

    result = {"operation": op, "verdict": verdict, "tolerance": tolerance, "displacement": displacement,
              "moved_points": moved_stats, "unsupported_fraction": unsupported_fraction,
              "identical_fraction": identical_fraction, "kept_fraction": kept_fraction,
              "removed_surface_points": removed_surface, "removed_stray_points": removed_stray,
              "reverse": reverse, "transform_check": tcheck, "noise": noise, "band_width": width,
              "dimensions_before": before, "dimensions_after": after, "dimension_change": change,
              "frame": frame_dims["frame"], "points": {"parent": int(len(geometry_points(parent))),
                                                       "child": int(len(geometry_points(child)))}}
    result["sentence"] = _drift_sentence(result)
    log(result["sentence"])
    return result


def _drift_sentence(r: dict) -> str:
    parts = []
    tc = r.get("transform_check")
    if tc is not None and not tc["identity"]:
        if tc["rigid"]:
            parts.append(f"The parent was placed by a rigid transform (rotation {tc['rotation_deg']:.2f} deg, "
                         f"determinant {tc['determinant']:.6f}, no scale).")
        else:
            parts.append(f"WARNING: the transform is not rigid (determinant {tc['determinant']:.6f}, scale "
                         f"{tc['scale_ppm']:+.0f} ppm{', mirrored' if tc['mirrored'] else ''}) - it resizes the data.")
    if r["identical_fraction"] is not None:
        removed = r["points"]["parent"] - r["points"]["child"]
        if r["identical_fraction"] >= 0.9999:
            if removed > 0:
                parts.append(f"{removed:,} points were removed and none was moved")
                if r.get("removed_surface_points") is not None:
                    parts[-1] += (f" ({r['removed_stray_points']:,} stray points off the surface, "
                                  f"{r['removed_surface_points']:,} from the surface itself)")
                parts[-1] += "."
            else:
                parts.append("Every point is unchanged.")
        else:
            parts.append(f"{100 * r['identical_fraction']:.1f} % of the points are the parent's own, unmoved.")
    disp, mv = r["displacement"], r["moved_points"]
    if mv is not None and mv["count"] > 0:
        parts.append(f"The other surface lies {mv['signed_mean']:+.4f} mm from the parent on average (median "
                     f"|d| {mv['median_abs']:.4f} mm, p95 {mv['p95']:.4f} mm).")
    elif disp["p95"] > 0:
        parts.append(f"Surface displacement: mean {disp['signed_mean']:+.4f} mm, p95 {disp['p95']:.4f} mm.")
    if r["unsupported_fraction"] > 0.005:
        parts.append(f"{100 * r['unsupported_fraction']:.1f} % of the surface has no parent data nearby (added "
                     "coverage or filled holes).")
    if r.get("reverse"):
        rv = r["reverse"]
        parts.append(f"Measured from the parent's points the result sits {rv['signed_mean']:+.4f} mm away on "
                     f"average (p95 {rv['p95']:.4f} mm).")
    ch = r["dimension_change"]
    parts.append("Part size change: " + ", ".join(f"{n} {ch[n]:+.3f} mm" for n in AXIS_NAMES) + ".")
    lead = {"unchanged": "No measurable change.", "moved": "Moved rigidly; shape and size unchanged.",
            "changed": "Changed."}[r["verdict"]]
    return " ".join([lead] + parts)


# --------------------------------------------------------------------------- pose / similarity fits
def fit_pose(surface: SurfaceModel, points, T0=None, mode: str = "rigid", frame_axes=None, iterations: int = 40,
             max_distance: float | None = None, floor: float | None = None, normals=None,
             log: Log = _quiet) -> tuple[np.ndarray, dict]:
    """Robust point-to-surface fit of `points` onto `surface` (Tukey IRLS, pivot at the points' centroid).

    mode: rigid (6 dof) | similarity (+ uniform scale) | similarity_offset (+ scale + a uniform normal offset of
    the surface, i.e. a thickness difference) | offset (rigid + normal offset) | axes (rigid + one scale per axis of
    `frame_axes`) | axes_offset (axes + normal offset: on a round part a thickness difference would otherwise
    look like a radial scale).
    Returns (4x4 transform, info) with scale / offset / axis_scales, their standard errors (from the
    fit covariance; spatially correlated residuals make them optimistic), rms, inliers.
    `normals` (outward normals of `points`) restricts the fit to surface both sides see facing the same way."""
    if mode not in ("rigid", "similarity", "similarity_offset", "offset", "axes", "axes_offset"):
        raise ValueError(f"Unknown fit mode '{mode}'")
    pts = np.asarray(points, float)
    T = np.eye(4) if T0 is None else np.asarray(T0, float).copy()
    spacing = max(surface.spacing, 1e-9)
    max_distance = max_distance or 10 * spacing
    floor = floor if floor is not None else 0.2 * spacing
    F = np.asarray(frame_axes, float) if frame_axes is not None else np.eye(3)
    offset_total, scale_total, axis_total = 0.0, 1.0, np.ones(3)
    info: dict = {"mode": mode, "iterations": 0, "converged": False}
    cov, w, pivot = None, None, None
    pn = None if normals is None else np.asarray(normals, float)
    for it in range(iterations):
        p = apply_transform(T, pts)
        d, n, gap, ok = surface.query(p, None if pn is None else pn @ T[:3, :3].T)
        d = d - offset_total
        use = ok & np.isfinite(d) & (np.abs(d) < max_distance)
        if use.sum() < 50:
            raise ValueError("Too little overlap between the scans to fit them")
        sigma = _mad_sigma(d[use])
        k = min(max(TUKEY_C * sigma, TUKEY_C * floor), max_distance)
        w = np.where(use, _tukey(d, k), 0.0)
        sel = w > 0
        p, n, d, ws = p[sel], n[sel], d[sel], w[sel]
        if pivot is None:
            pivot = p.mean(axis=0)
        c = pivot
        cols = [np.cross(p - c, n), n]
        if mode in ("similarity", "similarity_offset"):
            cols.append(np.einsum("ij,ij->i", p - c, n)[:, None])
        if mode in ("axes", "axes_offset"):
            cols.append(((p - c) @ F.T) * (n @ F.T))
        if mode in ("similarity_offset", "offset", "axes_offset"):
            cols.append(-np.ones((len(p), 1)))
        A = np.hstack(cols)
        Aw = A * ws[:, None]
        H = Aw.T @ A + 1e-12 * np.eye(A.shape[1])
        x = np.linalg.solve(H, -Aw.T @ d)
        res = d + A @ x
        dof = max(ws.sum() - A.shape[1], 1.0)
        cov = np.linalg.inv(H) * float((ws * res ** 2).sum()) / dof
        S = np.eye(3)
        if mode in ("similarity", "similarity_offset"):
            S = (1 + x[6]) * np.eye(3)
            scale_total *= 1 + x[6]
        elif mode in ("axes", "axes_offset"):
            S = F.T @ np.diag(1 + x[6:9]) @ F
            axis_total *= 1 + x[6:9]
        if mode in ("similarity_offset", "offset", "axes_offset"):
            offset_total += float(x[-1])
        M = _rotation(x[:3]) @ S
        dT = np.eye(4)
        dT[:3, :3] = M
        dT[:3, 3] = c + x[3:6] - M @ c
        T = dT @ T
        info["iterations"] = it + 1
        small = np.linalg.norm(x[:3]) < 1e-7 and np.linalg.norm(x[3:6]) < 1e-5 * spacing
        if mode in ("similarity", "similarity_offset"):
            small = small and abs(x[6]) < 1e-7
        if mode in ("axes", "axes_offset"):
            small = small and np.all(np.abs(x[6:9]) < 1e-7)
        if mode in ("similarity_offset", "offset", "axes_offset"):
            small = small and abs(x[-1]) < 1e-5 * spacing
        if small:
            info["converged"] = True
            break
    # final statistics at the solution
    p = apply_transform(T, pts)
    d, _, gap, ok = surface.query(p, None if pn is None else pn @ T[:3, :3].T)
    d = d - offset_total
    use = ok & np.isfinite(d) & (np.abs(d) < max_distance)
    info.update(transform=T, rms=float(np.sqrt(np.mean(d[use] ** 2))) if use.any() else None,
                inliers=int((w > 0).sum()) if w is not None else 0, overlap=float(use.mean()),
                scale=float(scale_total), offset=offset_total,
                singular_values=np.linalg.svd(T[:3, :3], compute_uv=False))
    if cov is not None:
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        if mode in ("similarity", "similarity_offset"):
            info["scale_se"] = float(se[6])
        if mode in ("axes", "axes_offset"):
            info["axis_scales"] = axis_total.tolist()
            info["axis_scales_se"] = se[6:9]
        if mode in ("similarity_offset", "offset", "axes_offset"):
            info["offset_se"] = float(se[-1])
    return T, info


# --------------------------------------------------------------------------- scan vs scan
def _icp_candidates(ca, Ra, sa, cb, Rb, steps: int) -> list[tuple[str, np.ndarray]]:
    """Initial poses mapping scan B onto scan A from their principal frames: the four proper sign flips and, for a
    part that is round about its long axis (a bolt, a pin), rotations about that axis in `steps` increments."""
    out = []
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        R = Ra.T @ np.diag(signs) @ Rb
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = ca - R @ cb
        out.append((f"pca{''.join('+' if s > 0 else '-' for s in signs)}", T))
    if sa[1] < 1.15 * sa[2] and sa[0] > 1.15 * sa[1]:
        for base in (out[0], out[3]):
            for k in range(1, steps):
                ang = 2 * math.pi * k / steps
                R = _rotation(Ra[0] * ang) @ base[1][:3, :3]
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = ca - R @ cb
                out.append((f"{base[0]}@{360 * k / steps:.0f}", T))
    return out


def align_scans(a: Geometry, b: Geometry, steps: int = 24, max_points: int = 80_000, seed: int = 0,
                log: Log = _quiet, progress=None) -> tuple[np.ndarray, dict]:
    """Rigid pose of scan B in scan A's coordinates from many starting poses (principal-axis flips and rotations
    about the long axis), each refined by robust point-to-plane ICP; ranked by how many points land within noise of
    the other scan. Near-equal alternatives at a clearly different pose are reported: a symmetric part (a bolt is
    round, its head has flats) can fit in several poses."""
    from .register import cap_points, refine_icp

    reg = o3d.pipelines.registration
    pa, _ = surface_points(a, 2_000_000, seed)
    pb, nb = surface_points(b, 2_000_000, seed)
    spacing = float(np.median([estimate_spacing(pa), estimate_spacing(pb)]))
    ca, Ra, sa = principal_frame(pa)
    cb, Rb, _ = principal_frame(pb)
    A = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pa))
    B = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pb))
    diag = float(np.linalg.norm(np.ptp(pa, axis=0)))
    voxel = max(diag / 80.0, spacing * 4)
    fine = 2.5 * spacing
    levels = [t for t in (voxel * 2.0, voxel * 0.8) if t > fine * 1.5] + [fine * 2]
    tight = 1.5 * spacing
    b_eval = cap_points(B.voxel_down_sample(spacing * 2), max_points)
    cands = _icp_candidates(ca, Ra, sa, cb, Rb, steps)
    cache: dict = {}
    scored = []
    for i, (name, T0) in enumerate(cands):
        T, fit, rmse = refine_icp(B, A, T0, levels, cache, max_points)
        if any(rotation_angle_deg(T[:3, :3].T @ s["T"][:3, :3]) < 0.5 and
               np.linalg.norm(T[:3, 3] - s["T"][:3, 3]) < fine for s in scored):
            continue
        tight_fit = reg.evaluate_registration(b_eval, A, tight, T).fitness
        scored.append({"candidate": name, "T": T, "score": float(tight_fit), "rmse": float(rmse)})
        if progress:
            progress((i + 1) / len(cands))
    scored.sort(key=lambda s: (round(s["score"], 4), -s["rmse"]), reverse=True)
    # The leading poses are refined at full resolution and re-ranked by the fraction of B's points that lie within
    # the scan noise of A's surface (local quadric fits): much finer than the ICP overlap at 1.5x the spacing,
    # which cannot tell a pose rotated by one flat of a bolt head from the right one.
    from .register import estimate_noise

    surface = SurfaceModel(a, seed=seed)
    noise = math.hypot(estimate_noise(pa), estimate_noise(pb))
    within = max(3 * noise, 0.2 * spacing)
    reach = 3 * spacing + 5 * noise
    rng = np.random.default_rng(seed)
    pick = np.arange(len(pb)) if len(pb) <= 40_000 else rng.choice(len(pb), 40_000, replace=False)
    probe, probe_n = pb[pick], (None if nb is None else nb[pick])
    top = scored[:6]
    for s in top:
        T, fit, rmse = refine_icp(B, A, s["T"], [fine * 2, fine], cache, max_points * 3)
        s["T"] = T
        s["score"] = float(reg.evaluate_registration(b_eval, A, tight, T).fitness)
        s["rmse"] = float(rmse)
        d, _, gap, ok = surface.query(apply_transform(T, probe), None if probe_n is None else probe_n @ T[:3, :3].T)
        good = ok & (gap <= reach)
        s["fine_score"] = float(np.mean(good & (np.abs(d) <= within)))
        s["median_separation"] = float(np.median(np.abs(d[good]))) if good.any() else None
    top.sort(key=lambda s: s["fine_score"], reverse=True)
    best = top[0]
    alternatives = []
    for s in top[1:] + scored[6:]:
        ang = rotation_angle_deg(best["T"][:3, :3].T @ s["T"][:3, :3])
        shift = float(np.linalg.norm(s["T"][:3, 3] - best["T"][:3, 3]))
        if ang > 3 or shift > 5 * spacing:
            alternatives.append({"candidate": s["candidate"], "score": s["score"], "rmse": s["rmse"],
                                 "fine_score": s.get("fine_score"), "median_separation": s.get("median_separation"),
                                 "angle_from_best": ang, "shift_from_best": shift, "T": s["T"]})
    alternatives.sort(key=lambda s: (s["fine_score"] is not None, s["fine_score"] or 0, s["score"]), reverse=True)
    ambiguous = bool(alternatives and alternatives[0]["fine_score"] is not None
                     and alternatives[0]["fine_score"] >= 0.9 * best["fine_score"])
    info = {"best_candidate": best["candidate"], "score": best["score"], "fine_score": best["fine_score"],
            "median_separation": best["median_separation"], "within_mm": within, "rmse": best["rmse"],
            "candidates_tried": len(cands), "distinct_poses": len(scored), "ambiguous": ambiguous,
            "alternatives": alternatives[:5], "spacing": spacing}
    log(f"Alignment: {best['candidate']} ({100 * best['fine_score']:.1f} % of points within {within:.3f} mm, "
        f"median separation {best['median_separation'] or 0:.4f} mm)"
        + (f"; ambiguous - {alternatives[0]['candidate']} reaches {100 * alternatives[0]['fine_score']:.1f} % at "
           f"{alternatives[0]['angle_from_best']:.1f} deg" if ambiguous else ""))
    return best["T"], info


def compare_scans(a: Geometry, b: Geometry, transform=None, max_points: int = 60_000, tolerance: float = 0.02,
                  seed: int = 0, log: Log = _quiet, progress=None) -> dict:
    """Two scans of the same part: how consistent is the scanner?

    B is aligned to A rigidly (from `transform` when given, e.g. an approved merge pose, else by `align_scans`),
    refined against A's surface, and then fitted again with a uniform scale (similarity), with scale + surface
    offset, and with one scale per part axis. Reported:
    * scale_ppm: B's size relative to A (+ = B larger), with its standard error
    * offset_mm: B's surface outside A's (+) after the scale fit - a thickness difference (e.g. exposure)
    * axis_scale_ppm: the same per part axis (length / width / height of A), fitted together with the offset
    * separation: signed distances of B's points to A's surface where they overlap (rigid pose)
    * extents: robust dimensions of both in A's part frame, over all points and over the common area only
    * noise of each scan, the alignment (candidates, ambiguity) and a verdict: consistent | differ | uncertain."""
    prog = progress or (lambda f, label=None: None)
    from .register import estimate_noise

    pa, _ = surface_points(a, 2_000_000, seed)
    pb, nb = surface_points(b, 2_000_000, seed + 1)
    noise_a, noise_b = estimate_noise(pa), estimate_noise(pb)
    spacing = float(np.median([estimate_spacing(pa), estimate_spacing(pb)]))
    log(f"Comparing scans: A {len(pa):,} points (noise {noise_a:.4f}), B {len(pb):,} points (noise {noise_b:.4f}), "
        f"spacing {spacing:.4f}")
    prog(0.05, "aligning")
    if transform is not None:
        T0 = np.asarray(transform, float)
        align_info = {"best_candidate": "given transform", "ambiguous": False, "alternatives": []}
    else:
        T0, align_info = align_scans(a, b, seed=seed, log=log,
                                     progress=lambda f: prog(0.05 + 0.5 * f, "aligning"))
    prog(0.6, "fitting")
    surface = SurfaceModel(a, seed=seed, log=log)
    rng = np.random.default_rng(seed)
    pick = np.arange(len(pb)) if len(pb) <= max_points else np.sort(rng.choice(len(pb), max_points, replace=False))
    sb, snb = pb[pick], (None if nb is None else nb[pick])
    combined = float(math.hypot(noise_a, noise_b))
    max_distance = max(10 * spacing, 20 * combined)
    T_rigid, rigid = fit_pose(surface, sb, T0, "rigid", max_distance=max_distance, normals=snb, log=log)
    prog(0.7, "scale")
    frame_a = principal_frame(pa)
    _, sim = fit_pose(surface, sb, T_rigid, "similarity", max_distance=max_distance, normals=snb)
    _, simo = fit_pose(surface, sb, T_rigid, "similarity_offset", max_distance=max_distance, normals=snb)
    _, axes = fit_pose(surface, sb, T_rigid, "axes_offset", frame_axes=frame_a[1], max_distance=max_distance,
                       normals=snb)
    prog(0.85, "separation")

    # separation at the rigid pose, over the overlap (both scans see the surface, facing the same way)
    moved = apply_transform(T_rigid, sb)
    d, _, gap, ok = surface.query(moved, None if snb is None else snb @ T_rigid[:3, :3].T)
    reach = 3 * spacing + 5 * combined
    overlap = ok & (gap <= reach) & (np.abs(d) <= max_distance)
    sep = _stats(d[overlap])

    # extents in A's part frame: everything, and the common area only
    frame = (frame_a[0], frame_a[1])
    pb_moved = apply_transform(T_rigid, pb)
    dims_a = robust_dimensions(pa, frame=frame)
    dims_b = robust_dimensions(pb_moved, frame=frame)
    tree_b = cKDTree(pb_moved)
    near_a = tree_b.query(pa, k=1, distance_upper_bound=reach, workers=-1)[0] <= reach
    near_b = cKDTree(pa).query(pb_moved, k=1, distance_upper_bound=reach, workers=-1)[0] <= reach
    common = None
    if near_a.sum() > 100 and near_b.sum() > 100:
        ca_dims = robust_dimensions(pa[near_a], frame=frame)
        cb_dims = robust_dimensions(pb_moved[near_b], frame=frame)
        common = {"a": {n: ca_dims[n] for n in AXIS_NAMES}, "b": {n: cb_dims[n] for n in AXIS_NAMES},
                  "difference": {n: cb_dims[n] - ca_dims[n] for n in AXIS_NAMES}}

    # B scaled by 1/s matches A, so B is (1/s - 1) larger than A
    def rel(s):
        return (1.0 / s - 1.0) * 1e6

    scale_ppm = rel(sim["scale"])
    scale_se_ppm = sim.get("scale_se", 0.0) * 1e6
    axis_ppm = [rel(s) for s in axes.get("axis_scales", [1, 1, 1])]
    offset = simo["offset"]  # fit_pose models the distance of B's points to A's surface as this constant
    ambiguous = bool(align_info.get("ambiguous"))
    uncertain = ambiguous or sep is None or overlap.mean() < 0.2
    size_diff_mm = abs(scale_ppm) * 1e-6 * dims_a["length"]
    consistent = (not uncertain and sep is not None and size_diff_mm <= tolerance
                  and abs(offset) <= tolerance and sep["median_abs"] <= max(tolerance, 2 * combined))
    verdict = "uncertain" if uncertain else "consistent" if consistent else "differ"
    result = {
        "verdict": verdict, "tolerance": tolerance,
        "alignment": {k: v for k, v in align_info.items()},
        "transform": T_rigid, "rigid_fit": {k: rigid[k] for k in ("rms", "inliers", "overlap", "iterations")},
        "scale_ppm": scale_ppm, "scale_se_ppm": scale_se_ppm,
        "scale_with_offset_ppm": rel(simo["scale"]), "offset_mm": offset, "offset_se_mm": simo.get("offset_se"),
        "axis_scale_ppm": {n: axis_ppm[i] for i, n in enumerate(AXIS_NAMES)}, "axis_fit_offset_mm": axes["offset"],
        "axis_scale_se_ppm": {n: float(axes.get("axis_scales_se", [0, 0, 0])[i]) * 1e6
                              for i, n in enumerate(AXIS_NAMES)},
        "length_difference_from_scale_mm": scale_ppm * 1e-6 * dims_a["length"],
        "separation": sep, "separation_histogram": _histogram(d[overlap], max(0.05, 6 * combined + 0.02)),
        "overlap_fraction": float(overlap.mean()),
        "noise": {"a": noise_a, "b": noise_b, "combined": combined}, "spacing": spacing,
        "extents": {"a": {n: dims_a[n] for n in AXIS_NAMES}, "b": {n: dims_b[n] for n in AXIS_NAMES},
                    "difference": {n: dims_b[n] - dims_a[n] for n in AXIS_NAMES}},
        "extents_common": common, "frame": {"center": frame_a[0], "axes": frame_a[1]},
    }
    result["sentence"] = _compare_sentence(result)
    log(result["sentence"])
    prog(1.0, "done")
    return result


def _compare_sentence(r: dict) -> str:
    sep, n = r["separation"], r["noise"]
    parts = []
    if r["verdict"] == "uncertain":
        why = ("the alignment is ambiguous (the part is nearly symmetric)" if r["alignment"].get("ambiguous")
               else "the scans overlap too little")
        parts.append(f"Uncertain: {why}, so check the alignment before trusting these numbers.")
    ax = r["axis_scale_ppm"]
    length = r["extents"]["a"]["length"]
    parts.append(f"Along the part's length scan B reads {ax['length']:+.0f} ppm "
                 f"({ax['length'] * 1e-6 * length:+.3f} mm over "
                 f"{length:.1f} mm) against scan A; across it {ax['width']:+.0f} / {ax['height']:+.0f} ppm, with B's "
                 f"surface {r['axis_fit_offset_mm']:+.4f} mm outside A's (a thickness difference).")
    if sep is not None:
        parts.append(f"Where both saw the same surface ({100 * r['overlap_fraction']:.0f} % of B) they are "
                     f"{sep['median_abs']:.4f} mm apart (median; signed mean {sep['signed_mean']:+.4f}, p95 "
                     f"{sep['p95']:.4f}); the scan noise alone explains about {0.6745 * n['combined']:.4f} mm.")
    parts.append(f"(One uniform scale for everything: {r['scale_ppm']:+.0f} ppm ± {r['scale_se_ppm']:.0f}; with a "
                 f"thickness offset: {r['scale_with_offset_ppm']:+.0f} ppm and {r['offset_mm']:+.4f} mm.)")
    return " ".join(parts)


# --------------------------------------------------------------------------- reference artefacts
REFERENCE_TYPES = ("known_length", "sphere_pair", "diameter", "thread_pitch")


def _region_mask(geom_or_cloud, region) -> np.ndarray:
    """Region (Contract 2 kinds when cloudclean.regions is available, else box / spheres / screen selection)."""
    try:
        from .regions import region_mask as mask_fn  # all region kinds (workstream B)
    except ImportError:  # pragma: no cover - depends on the other workstream
        from .analysis import region_mask as mask_fn
    return np.asarray(mask_fn(geom_or_cloud, region), bool)


def region_points(geom: Geometry, region=None, max_points: int = 2_000_000, seed: int = 0):
    """(points, normals) of the surface inside a region (None = everything)."""
    pts, nrm = surface_points(geom, max_points, seed)
    if region is None or region == {}:
        return pts, nrm
    if is_cloud(geom) and len(pts) == len(geom.points):
        mask = _region_mask(geom, region)
    else:
        mask = _region_mask(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)), region)
    return pts[mask], (None if nrm is None else nrm[mask])


def measure_length(geom: Geometry, direction, region=None, band: float | None = None) -> dict:
    """Distance between the two end faces along a direction, measured like calipers: planes are fitted to the points
    in the extreme band at each end and the distance between them is taken along their mean normal. When the ends
    are not faces across the direction (a rounded end) the robust extent is returned instead (method 'extent')."""
    allp, _ = surface_points(geom)
    _, axes, _ = principal_frame(allp)
    dvec = resolve_direction(direction, axes)
    pts, _ = region_points(geom, region)
    if len(pts) < 50:
        raise ValueError(f"The region has only {len(pts)} points - select more of the part")
    s = pts @ dvec
    lo, hi = np.quantile(s, [0.0005, 0.9995])
    extent = float(hi - lo)
    spacing = estimate_spacing(pts)
    band = band or max(0.02 * extent, 6 * spacing)
    faces = []
    for side, m in (("start", s <= lo + band), ("end", s >= hi - band)):
        if m.sum() < 20:
            faces = []
            break
        f = fit_plane(pts[m])
        nrm = f["normal"] if f["normal"] @ dvec >= 0 else -f["normal"]
        faces.append({**f, "normal": nrm, "side": side})
    method = "extent"
    measured = extent
    parallel = None
    if len(faces) == 2 and all(abs(f["normal"] @ dvec) > math.cos(math.radians(10)) for f in faces):
        n_mean = faces[0]["normal"] + faces[1]["normal"]
        n_mean /= np.linalg.norm(n_mean)
        measured = float(abs((faces[1]["point"] - faces[0]["point"]) @ n_mean))
        parallel = float(np.degrees(np.arccos(np.clip(faces[0]["normal"] @ faces[1]["normal"], -1, 1))))
        method = "faces"
    se = None
    if method == "faces":
        se = float(math.hypot(*(f["rms"] / math.sqrt(max(f["points_used"], 1)) for f in faces)))
    a = faces[0]["point"] if method == "faces" else pts[np.argmin(s)]
    b = a + measured * dvec
    return {"measured": measured, "method": method, "direction": dvec, "parallelism_deg": parallel,
            "faces": [{k: f[k] for k in ("side", "point", "normal", "rms", "flatness", "points_used")} for f in faces],
            "extent": extent, "uncertainty": se, "a": a, "b": b, "points_used": int(len(pts))}


def _sphere_in(geom, region, name):
    pts, _ = region_points(geom, region)
    if len(pts) < 30:
        raise ValueError(f"{name} holds only {len(pts)} points - select a whole ball")
    return fit_sphere(pts)


def default_tolerance(nominal: float) -> float:
    """Assumed acceptance when none is given: 0.02 mm + 100 ppm of the nominal (a metrology-grade blue-laser scanner
    on a short artefact). Pass `tolerance` for your own scanner specification."""
    return 0.02 + 1e-4 * abs(nominal)


def reference_check(geom: Geometry, reference: dict, log: Log = _quiet) -> dict:
    """Measure a known artefact and compare with its nominal size.

    reference = {type: "known_length", direction, region?, nominal}
              | {type: "sphere_pair", region_a, region_b, nominal}          (ball bar / two calibrated spheres)
              | {type: "diameter", region, nominal}                         (pin / cylinder; ring gauge)
              | {type: "thread_pitch", region?, nominal?}                   (nominal pitch; default: nearest
                                                                             ISO / Unified / BSP standard)
    plus optional `tolerance` (default `default_tolerance(nominal)`).
    Returns {type, measured, nominal, error, error_ppm, error_pct, uncertainty, tolerance, verdict: ok | marginal
    | off, significant, sentence, details}."""
    if not isinstance(reference, dict):
        raise ValueError("reference must be an object with a type")
    kind = reference.get("type")
    if kind not in REFERENCE_TYPES:
        raise ValueError(f"reference.type must be one of: {', '.join(REFERENCE_TYPES)}")
    nominal = reference.get("nominal")
    if kind != "thread_pitch" or nominal is not None:
        try:
            nominal = float(nominal)
        except (TypeError, ValueError):
            raise ValueError("reference.nominal must be a number (the certified size)")
        if not math.isfinite(nominal) or nominal <= 0:
            raise ValueError("reference.nominal must be a positive number")
    details: dict = {}
    if kind == "known_length":
        if reference.get("direction") is None:
            raise ValueError("known_length needs a direction: 'x', 'y', 'z', 'length', 'width', 'height' or [x, y, z]")
        m = measure_length(geom, reference["direction"], reference.get("region"))
        measured, unc = m["measured"], m["uncertainty"]
        details = m
        what = f"length along {reference['direction']}"
    elif kind == "sphere_pair":
        if reference.get("region_a") is None or reference.get("region_b") is None:
            raise ValueError("sphere_pair needs region_a and region_b (one around each ball)")
        s1 = _sphere_in(geom, reference["region_a"], "region_a")
        s2 = _sphere_in(geom, reference["region_b"], "region_b")
        measured = float(np.linalg.norm(s2["center"] - s1["center"]))
        unc = float(np.linalg.norm(np.r_[s1["center_se"], s2["center_se"]]))
        details = {"sphere_a": s1, "sphere_b": s2, "a": s1["center"], "b": s2["center"]}
        what = "sphere centre distance"
    elif kind == "diameter":
        if reference.get("region") is None:
            raise ValueError("diameter needs a region around the cylindrical surface")
        pts, nrm = region_points(geom, reference["region"])
        if len(pts) < 30:
            raise ValueError(f"The region holds only {len(pts)} points - select more of the cylinder")
        c = fit_cylinder(pts, nrm, reference.get("axis_hint"))
        measured, unc = c["diameter"], 2 * c["radius_se"]
        details = c
        what = "diameter"
    else:
        pts, nrm = region_points(geom, reference.get("region"))
        th = fit_thread(pts, nrm, log=log)
        measured, unc = th["pitch"], th["pitch_se"]
        details = {k: th[k] for k in ("handedness", "major_diameter", "pitch_diameter", "minor_diameter", "axis",
                                      "center", "length", "coverage_deg", "noise", "lead", "lead_slope_ppm",
                                      "lead_residual_pp", "tpi", "kind")}
        if nominal is None:
            from .analysis import nearest_standard_thread

            best = nearest_standard_thread(th["major_diameter"], th["pitch"], 1, th["kind"])[0]
            nominal = float(best["pitch"])
            details["standard"] = best["designation"]
        what = "thread pitch"
    error = measured - nominal
    tol = float(reference.get("tolerance") or default_tolerance(nominal if kind != "thread_pitch" else 0.0))
    if kind == "thread_pitch" and not reference.get("tolerance"):
        tol = 0.0005 * nominal   # 500 ppm: commercial threads are not made to a better lead
    verdict = "ok" if abs(error) <= tol else "marginal" if abs(error) <= 2 * tol else "off"
    significant = bool(unc is not None and abs(error) > 2 * unc)
    ppm = error / nominal * 1e6
    sentence = (f"Measured {what} {measured:.4f} mm vs nominal {nominal:.4f} mm: {error:+.4f} mm "
                f"({ppm:+.0f} ppm, {error / nominal * 100:+.3f} %)"
                + (f", measurement uncertainty ±{unc:.4f} mm (1 sigma)" if unc is not None else "")
                + f". {'Within' if verdict == 'ok' else 'Close to' if verdict == 'marginal' else 'Outside'} the "
                  f"tolerance of ±{tol:.4f} mm.")
    if kind == "thread_pitch":
        sentence += (" A thread's lead is only as exact as its manufacture (typically within a few hundred ppm), so "
                     "this checks the scanner's scale coarsely.")
    return {"type": kind, "measured": measured, "nominal": nominal, "error": error, "error_ppm": ppm,
            "error_pct": error / nominal * 100, "uncertainty": unc, "tolerance": tol, "verdict": verdict,
            "significant": significant, "sentence": sentence, "details": details}
