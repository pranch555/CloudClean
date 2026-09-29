"""Robust geometric fits for measuring: plane, sphere, circle (2D and 3D), cylinder and line.

Every fit is least squares on the inliers of a robust start (MSAC for planes, spheres and circles, candidate axes
scored by the roundness of the cross-section for cylinders), iterated with 3-sigma trimming until the inlier set
is stable. Nothing is smoothed or resampled: the model search and the non-linear cylinder refinement use at most
`FIT_SAMPLE` points picked uniformly at random (fixed seed, so results are reproducible and unbiased), and the
reported numbers - rms, flatness, inliers, extents, coverage - are computed on every point that was given.

All functions return plain dicts (lists and floats, JSON-ready) and raise ValueError with a sentence when the
points cannot support the fit. Common keys: `points_used` (points in the final least-squares estimate),
`points_in_region` (points given), `inliers` (points within the final band on all given points), `rms` (root mean
square of the geometric residuals of the inliers, in scan units).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

FIT_SAMPLE = 200_000        # max points for the model search
CYLINDER_SAMPLE = 50_000    # max points for the non-linear cylinder refinement (radius error ~ noise / 220)
EVAL_SAMPLE = 5_000         # points scoring each MSAC hypothesis
MIN_BAND_REL = 1e-9         # inlier bands never shrink below this fraction of the data size


# --------------------------------------------------------------------------- helpers
def _as_points(points, dims: int = 3) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != dims or not np.isfinite(p).all():
        raise ValueError(f"expected an (n, {dims}) array of finite coordinates")
    return p


def sample_indices(n: int, size: int = FIT_SAMPLE, seed: int = 0) -> np.ndarray:
    """Sorted uniform random subset of range(n) (all of it when n <= size); deterministic for a seed."""
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size, replace=False))


def robust_sigma(residuals: np.ndarray) -> float:
    """1.4826 x median absolute deviation (= sigma for Gaussian residuals)."""
    r = np.asarray(residuals, float)
    if len(r) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def local_noise(points: np.ndarray, k: int = 16, samples: int = 1000, seed: int = 0) -> float:
    """Scanner noise estimate: median rms of local plane fits to the k nearest neighbours of sampled points
    (independent of the global shape as long as neighbourhoods are small compared with the curvature)."""
    p = np.asarray(points, float)
    if len(p) <= k:
        return 0.0
    tree = cKDTree(p)
    idx = sample_indices(len(p), samples, seed)
    _, nb = tree.query(p[idx], k=k, workers=-1)
    q = p[nb]
    q = q - q.mean(axis=1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", q, q) / k
    smallest = np.linalg.eigvalsh(cov)[:, 0]
    return float(np.sqrt(np.median(np.maximum(smallest, 0.0))))


def _scale(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.ptp(points, axis=0))) if len(points) > 1 else 1.0


def _basis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.eye(3)[int(np.argmin(np.abs(a)))]
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise ValueError("direction must be a non-zero vector")
    return v / n


def _canonical(v: np.ndarray) -> np.ndarray:
    """Sign convention for axes / normals without orientation: largest component positive."""
    return -v if v[int(np.argmax(np.abs(v)))] < 0 else v


def _r(v, digits: int = 6):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [round(float(x), digits) for x in np.asarray(v).reshape(-1)]
    return round(float(v), digits)


def _trim_loop(residual_fn, refit_fn, n_all: int, band0: float, floor: float, iterations: int = 10):
    """Alternate: inliers = |residual| <= band, refit on inliers, band = max(3 x inlier rms, floor)."""
    band = max(band0, floor)
    inliers = None
    for _ in range(iterations):
        res = residual_fn()
        new = np.abs(res) <= band
        if new.sum() < 3:
            break
        if inliers is not None and np.array_equal(new, inliers):
            break
        inliers = new
        refit_fn(inliers)
        res = residual_fn()
        rms = float(np.sqrt(np.mean(res[inliers] ** 2)))
        band = max(3.0 * rms, floor)
    if inliers is None:
        inliers = np.ones(n_all, bool)
    return inliers


def coverage_deg(points: np.ndarray, center: np.ndarray, axis: np.ndarray, bin_deg: float = 5.0) -> float:
    """Angular coverage around an axis: occupied bins of `bin_deg` degrees (needs >= 2 points per bin)."""
    if len(points) == 0:
        return 0.0
    u, v = _basis(axis)
    rel = points - center
    ang = np.degrees(np.arctan2(rel @ v, rel @ u)) % 360.0
    counts = np.bincount(np.minimum((ang / bin_deg).astype(int), int(360 / bin_deg) - 1),
                         minlength=int(360 / bin_deg))
    need = 2 if len(points) >= 4 * 360 / bin_deg else 1
    return float((counts >= need).sum() * bin_deg)


# --------------------------------------------------------------------------- plane
def _svd_plane(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane (total least squares): smallest eigenvector of the 3 x 3 scatter matrix."""
    c = p.mean(axis=0)
    x = p - c
    return _canonical(np.linalg.eigh(x.T @ x)[1][:, 0]), c


def _msac_plane(s: np.ndarray, band: float, iterations: int, rng) -> tuple[np.ndarray, np.ndarray] | None:
    m = len(s)
    ev = s[rng.choice(m, min(m, EVAL_SAMPLE), replace=False)]
    tri = np.array([rng.choice(m, 3, replace=False) for _ in range(iterations)])
    a, b, c = s[tri[:, 0]], s[tri[:, 1]], s[tri[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1)
    ok = length > 1e-12 * _scale(s) ** 2
    if not ok.any():
        return None
    n, a = n[ok] / length[ok, None], a[ok]
    res = ev @ n.T - np.einsum("ij,ij->i", a, n)[None, :]
    cost = np.minimum(res * res, band * band).sum(axis=0)
    k = int(np.argmin(cost))
    return n[k], a[k]


def fit_plane(points, *, band: float | None = None, seed: int = 0) -> dict:
    """Robust plane: {point (centroid of the inliers), normal (unit, largest component positive), rms, flatness
    (peak-to-valley of the inlier residuals), inliers, points_used, points_in_region, inlier_fraction, noise}."""
    p = _as_points(points)
    n = len(p)
    if n < 3:
        raise ValueError(f"A plane needs at least 3 points (the selection has {n})")
    rng = np.random.default_rng(seed)
    s = p[sample_indices(n, FIT_SAMPLE, seed)]
    scale = _scale(p)
    noise = local_noise(s) if len(s) > 32 else 0.0
    floor = max(2.0 * noise, MIN_BAND_REL * scale)
    t = band if band is not None else max(3.0 * noise, 1e-4 * scale)
    normal, point = _svd_plane(s)
    if len(s) >= 30:
        best = _msac_plane(s, t, 300, rng)
        if best is not None:
            normal, point = best
    state = {"normal": _unit(normal), "point": np.asarray(point, float)}

    def residual():
        return (p - state["point"]) @ state["normal"]

    def refit(mask):
        state["normal"], state["point"] = _svd_plane(p[mask])

    inliers = _trim_loop(residual, refit, n, t, floor)
    res = residual()[inliers]
    rms = float(np.sqrt(np.mean(res ** 2)))
    return {"point": _r(state["point"]), "normal": _r(state["normal"], 9), "rms": rms,
            "flatness": float(res.max() - res.min()) if len(res) else 0.0, "inliers": int(inliers.sum()),
            "points_used": int(inliers.sum()), "points_in_region": n, "inlier_fraction": float(inliers.mean()),
            "noise": noise}


# --------------------------------------------------------------------------- sphere
def _algebraic_sphere(p: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.c_[2 * p, np.ones(len(p))]
    b = np.einsum("ij,ij->i", p, p)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    c = sol[:3]
    return c, math.sqrt(max(sol[3] + c @ c, 1e-300))


def _geometric_sphere(p: np.ndarray, c: np.ndarray, r: float, iterations: int = 20) -> tuple[np.ndarray, float]:
    """Gauss-Newton on sum (|p - c| - r)^2."""
    for _ in range(iterations):
        d = p - c
        dist = np.linalg.norm(d, axis=1)
        dist = np.maximum(dist, 1e-300)
        res = dist - r
        jac = np.c_[-d / dist[:, None], -np.ones(len(p))]
        step, *_ = np.linalg.lstsq(jac, -res, rcond=None)
        c, r = c + step[:3], r + step[3]
        if np.linalg.norm(step) < 1e-12 * max(abs(r), 1.0):
            break
    return c, abs(r)


def fit_sphere(points, *, band: float | None = None, seed: int = 0) -> dict:
    """Robust sphere: {center, radius, diameter, rms, inliers, points_used, points_in_region, inlier_fraction,
    coverage (fraction of directions from the centre that hold points, 0..1)}."""
    p = _as_points(points)
    n = len(p)
    if n < 4:
        raise ValueError(f"A sphere needs at least 4 points (the selection has {n})")
    rng = np.random.default_rng(seed)
    s = p[sample_indices(n, FIT_SAMPLE, seed)]
    scale = _scale(p)
    noise = local_noise(s) if len(s) > 32 else 0.0
    floor = max(2.0 * noise, MIN_BAND_REL * scale)
    t = band if band is not None else max(3.0 * noise, 1e-4 * scale)
    c, r = _algebraic_sphere(s)
    if len(s) >= 40:  # MSAC over 4-point spheres, scored on a random subset
        ev = s[rng.choice(len(s), min(len(s), EVAL_SAMPLE), replace=False)]
        best_cost = np.minimum((np.linalg.norm(ev - c, axis=1) - r) ** 2, t * t).sum()
        for _ in range(200):
            q = s[rng.choice(len(s), 4, replace=False)]
            try:
                cq, rq = _algebraic_sphere(q)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(rq) or rq > 10 * scale:
                continue
            cost = np.minimum((np.linalg.norm(ev - cq, axis=1) - rq) ** 2, t * t).sum()
            if cost < best_cost:
                best_cost, c, r = cost, cq, rq
    state = {"c": c, "r": r}

    def residual():
        return np.linalg.norm(p - state["c"], axis=1) - state["r"]

    def refit(mask):
        q = p[mask]
        c0, r0 = _algebraic_sphere(q)
        state["c"], state["r"] = _geometric_sphere(q, c0, r0)

    inliers = _trim_loop(residual, refit, n, t, floor)
    res = residual()[inliers]
    dirs = p[inliers] - state["c"]
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1), 1e-300)[:, None]
    # coverage: occupied cells of a 20 x 40 latitude / longitude grid, area weighted
    lat = np.arcsin(np.clip(dirs[:, 2], -1, 1))
    lon = np.arctan2(dirs[:, 1], dirs[:, 0])
    li = np.minimum(((lat + math.pi / 2) / math.pi * 20).astype(int), 19)
    lo = np.minimum(((lon + math.pi) / (2 * math.pi) * 40).astype(int), 39)
    occupied = np.zeros((20, 40), bool)
    occupied[li, lo] = True
    band_area = np.diff(np.sin(np.linspace(-math.pi / 2, math.pi / 2, 21)))
    coverage = float((occupied * band_area[:, None]).sum() / (40 * band_area.sum()))
    return {"center": _r(state["c"]), "radius": float(state["r"]), "diameter": 2 * float(state["r"]),
            "rms": float(np.sqrt(np.mean(res ** 2))), "inliers": int(inliers.sum()),
            "points_used": int(inliers.sum()), "points_in_region": n, "inlier_fraction": float(inliers.mean()),
            "coverage": coverage, "noise": noise}


# --------------------------------------------------------------------------- circle
def _kasa(xy: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.c_[2 * xy, np.ones(len(xy))]
    b = np.einsum("ij,ij->i", xy, xy)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    c = sol[:2]
    return c, math.sqrt(max(sol[2] + c @ c, 1e-300))


def _geometric_circle(xy: np.ndarray, c: np.ndarray, r: float, iterations: int = 20) -> tuple[np.ndarray, float]:
    for _ in range(iterations):
        d = xy - c
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-300)
        res = dist - r
        jac = np.c_[-d / dist[:, None], -np.ones(len(xy))]
        step, *_ = np.linalg.lstsq(jac, -res, rcond=None)
        c, r = c + step[:2], r + step[2]
        if np.linalg.norm(step) < 1e-12 * max(abs(r), 1.0):
            break
    return c, abs(r)


def fit_circle_2d(xy, *, band: float | None = None, noise: float | None = None, seed: int = 0) -> dict:
    """Robust circle in 2D: {center [2], radius, rms, inliers, points_used, inlier_fraction, coverage_deg}."""
    q = _as_points(xy, 2)
    n = len(q)
    if n < 3:
        raise ValueError(f"A circle needs at least 3 points (the selection has {n})")
    rng = np.random.default_rng(seed)
    s = q[sample_indices(n, FIT_SAMPLE, seed)]
    scale = _scale(q)
    if noise is None:
        noise = 0.0
    floor = max(2.0 * noise, MIN_BAND_REL * scale)
    c, r = _kasa(s)
    if band is None:
        res0 = np.linalg.norm(s - c, axis=1) - r
        t = max(3.0 * noise, 2.5 * robust_sigma(res0), 1e-4 * scale)
    else:
        t = band
    if len(s) >= 30:  # MSAC over 3-point circles
        ev = s[rng.choice(len(s), min(len(s), EVAL_SAMPLE), replace=False)]
        best_cost = np.minimum((np.linalg.norm(ev - c, axis=1) - r) ** 2, t * t).sum()
        idx = np.array([rng.choice(len(s), 3, replace=False) for _ in range(200)])
        for tri in idx:
            try:
                cq, rq = _kasa(s[tri])
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(rq) or rq > 10 * scale:
                continue
            cost = np.minimum((np.linalg.norm(ev - cq, axis=1) - rq) ** 2, t * t).sum()
            if cost < best_cost:
                best_cost, c, r = cost, cq, rq
    state = {"c": c, "r": r}

    def residual():
        return np.linalg.norm(q - state["c"], axis=1) - state["r"]

    def refit(mask):
        c0, r0 = _kasa(q[mask])
        state["c"], state["r"] = _geometric_circle(q[mask], c0, r0)

    inliers = _trim_loop(residual, refit, n, t, floor)
    res = residual()[inliers]
    rel = q[inliers] - state["c"]
    ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360
    occupied = np.unique(np.minimum((ang / 5).astype(int), 71)).size
    return {"center": _r(state["c"]), "radius": float(state["r"]), "rms": float(np.sqrt(np.mean(res ** 2))),
            "inliers": int(inliers.sum()), "points_used": int(inliers.sum()), "points_in_region": n,
            "inlier_fraction": float(inliers.mean()), "coverage_deg": float(occupied * 5)}


def fit_circle_3d(points, *, normal=None, seed: int = 0) -> dict:
    """Circle in space: the plane of the points (or the given normal) and a robust circle in that plane.
    {center, normal, radius, diameter, rms (in-plane radial), plane_rms, inliers, points_used, coverage_deg}."""
    p = _as_points(points)
    if len(p) < 4:
        raise ValueError(f"A circle needs at least 4 points (the selection has {len(p)})")
    if normal is None:
        plane = fit_plane(p, seed=seed)
        nrm, origin = np.asarray(plane["normal"]), np.asarray(plane["point"])
    else:
        nrm, origin = _unit(normal), p.mean(axis=0)
    u, v = _basis(nrm)
    rel = p - origin
    xy = np.c_[rel @ u, rel @ v]
    noise = local_noise(p[sample_indices(len(p), 20_000, seed)]) if len(p) > 32 else 0.0
    circle = fit_circle_2d(xy, noise=noise, seed=seed)
    center = origin + circle["center"][0] * u + circle["center"][1] * v
    plane_res = rel @ nrm
    return {"center": _r(center), "normal": _r(_canonical(nrm), 9), "radius": circle["radius"],
            "diameter": 2 * circle["radius"], "rms": circle["rms"],
            "plane_rms": float(np.sqrt(np.mean((plane_res - plane_res.mean()) ** 2))),
            "inliers": circle["inliers"], "points_used": circle["points_used"], "points_in_region": len(p),
            "inlier_fraction": circle["inlier_fraction"], "coverage_deg": circle["coverage_deg"], "noise": noise}


# --------------------------------------------------------------------------- cylinder
def _cyl_frame(x: np.ndarray, a0, u0, v0, c0):
    a = a0 + x[0] * u0 + x[1] * v0
    a = a / np.linalg.norm(a)
    c = c0 + x[2] * u0 + x[3] * v0
    return a, c


def _axis_distance(p: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = p - c
    t = d @ a
    return np.linalg.norm(d - t[:, None] * a, axis=1)


def _estimate_normals(p: np.ndarray, k: int = 16) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(k))
    return np.asarray(pcd.normals).copy()


def _axis_candidates(s: np.ndarray, normals, axis_hint) -> list[np.ndarray]:
    cands = []
    if axis_hint is not None:
        cands.append(_unit(axis_hint))
    cands += list(np.linalg.eigh(np.cov(s.T))[1].T)
    if normals is None and len(s) >= 50:
        normals = _estimate_normals(s[sample_indices(len(s), 20_000, 1)])
    if normals is not None and len(normals):
        nn = normals / np.maximum(np.linalg.norm(normals, axis=1), 1e-300)[:, None]
        cands.append(np.linalg.eigh(nn.T @ nn)[1][:, 0])   # normals of a cylinder are perpendicular to its axis
    return [_unit(c) for c in cands]


def fit_cylinder(points, *, axis_hint=None, normals=None, seed: int = 0) -> dict:
    """Robust cylinder: {point (on the axis, middle of the inliers' axial extent), axis (unit), radius, diameter,
    length (axial extent of the inliers), rms, coverage_deg, inliers, points_used, points_in_region,
    inlier_fraction, axis_source}. `axis_hint` (a direction) is tried first and wins ties. The non-linear
    refinement uses at most CYLINDER_SAMPLE points (uniform random); rms, inliers, length and coverage use all."""
    from scipy.optimize import least_squares

    p = _as_points(points)
    n = len(p)
    if n < 6:
        raise ValueError(f"A cylinder needs at least 6 points (the selection has {n})")
    offset = p.mean(axis=0)
    x_all = p - offset
    s = x_all[sample_indices(n, CYLINDER_SAMPLE, seed)]
    scale = _scale(s)
    noise = local_noise(s[sample_indices(len(s), 50_000, seed)]) if len(s) > 32 else 0.0
    floor = max(2.0 * noise, MIN_BAND_REL * scale)
    probe = s[sample_indices(len(s), 20_000, seed + 1)]
    best = None
    for i, a in enumerate(_axis_candidates(s, normals, axis_hint)):
        u, v = _basis(a)
        xy = np.c_[probe @ u, probe @ v]
        try:
            c2, r2 = _kasa(xy)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(r2) or r2 > 20 * scale:
            continue
        res = np.linalg.norm(xy - c2, axis=1) - r2
        score = robust_sigma(res) / max(r2, 1e-12)
        if i == 0 and axis_hint is not None:
            score *= 0.9  # prefer the hint on near ties
        if best is None or score < best[0]:
            best = (score, a, c2[0] * u + c2[1] * v, r2, "hint" if i == 0 and axis_hint is not None else "fit")
    if best is None:
        raise ValueError("No cylindrical surface found in the selection")
    _, a0, c0, r0, source = best
    u0, v0 = _basis(a0)
    state = {"x": np.array([0.0, 0.0, 0.0, 0.0, r0])}

    def model_res(x, q):
        a, c = _cyl_frame(x, a0, u0, v0, c0)
        return _axis_distance(q, a, c) - x[4]

    def model_jac(x, q):
        """Analytic Jacobian of model_res: radial unit vectors e, axial coordinates t."""
        raw = a0 + x[0] * u0 + x[1] * v0
        norm = np.linalg.norm(raw)
        a, c = raw / norm, c0 + x[2] * u0 + x[3] * v0
        d = q - c
        t = d @ a
        radial = d - t[:, None] * a
        e = radial / np.maximum(np.linalg.norm(radial, axis=1), 1e-300)[:, None]
        eu, ev = e @ u0, e @ v0
        return np.c_[-t * eu / norm, -t * ev / norm, -eu, -ev, -np.ones(len(q))]

    res0 = model_res(state["x"], s)
    f_scale = max(2.0 * robust_sigma(res0), floor, 1e-9)
    sol = least_squares(model_res, state["x"], jac=model_jac, args=(s,), loss="soft_l1", f_scale=f_scale,
                        x_scale=[1e-2, 1e-2, max(r0, 1e-9) * 1e-2, max(r0, 1e-9) * 1e-2, max(r0, 1e-9) * 1e-2])
    state["x"] = sol.x
    band = max(3.0 * robust_sigma(model_res(state["x"], s)), floor)
    for _ in range(4):  # trimmed least squares on the sample
        inl = np.abs(model_res(state["x"], s)) <= band
        if inl.sum() < 6:
            break
        sol = least_squares(model_res, state["x"], jac=model_jac, args=(s[inl],))
        new_band = max(3.0 * float(np.sqrt(np.mean(sol.fun ** 2))), floor)
        state["x"] = sol.x
        if abs(new_band - band) <= 1e-3 * band:
            band = new_band
            break
        band = new_band
    a, c = _cyl_frame(state["x"], a0, u0, v0, c0)
    radius = abs(float(state["x"][4]))
    res_all = _axis_distance(x_all, a, c) - radius
    inliers = np.abs(res_all) <= band
    if inliers.sum() < 6:
        inliers = np.ones(n, bool)
    t = (x_all[inliers] - c) @ a
    t0, t1 = float(t.min()), float(t.max())
    center = c + a * (t0 + t1) / 2
    used = int(min(inliers.sum(), len(s)))
    a = _canonical(a)
    return {"point": _r(center + offset), "axis": _r(a, 9), "radius": radius, "diameter": 2 * radius,
            "length": t1 - t0, "rms": float(np.sqrt(np.mean(res_all[inliers] ** 2))),
            "coverage_deg": coverage_deg(x_all[inliers], center, a), "inliers": int(inliers.sum()),
            "points_used": used, "points_in_region": n, "inlier_fraction": float(inliers.mean()),
            "axis_source": source, "noise": noise}


# --------------------------------------------------------------------------- line
def fit_line(points, *, seed: int = 0) -> dict:
    """Robust 3D line: {point (centroid of the inliers), direction (unit, largest component positive), rms
    (perpendicular), length (extent of the inliers along the line), a, b (end points), inliers, points_used}."""
    p = _as_points(points)
    n = len(p)
    if n < 2:
        raise ValueError(f"A line needs at least 2 points (the selection has {n})")
    state = {}

    def refit(mask):
        q = p[mask]
        c = q.mean(axis=0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        state["c"], state["d"] = c, _canonical(vt[0])

    def residual():
        d = p - state["c"]
        return np.linalg.norm(d - (d @ state["d"])[:, None] * state["d"], axis=1)

    refit(np.ones(n, bool))
    r0 = residual()
    scale = _scale(p)
    inliers = _trim_loop(residual, refit, n, max(3.0 * float(np.median(r0)) * 1.5, MIN_BAND_REL * scale),
                         MIN_BAND_REL * scale)
    t = (p[inliers] - state["c"]) @ state["d"]
    res = residual()[inliers]
    return {"point": _r(state["c"]), "direction": _r(state["d"], 9), "rms": float(np.sqrt(np.mean(res ** 2))),
            "length": float(t.max() - t.min()), "a": _r(state["c"] + t.min() * state["d"]),
            "b": _r(state["c"] + t.max() * state["d"]), "inliers": int(inliers.sum()),
            "points_used": int(inliers.sum()), "points_in_region": n, "inlier_fraction": float(inliers.mean())}


def angle_between(u, v, *, lines: bool = True) -> float:
    """Acute angle in degrees (0..90) between two undirected directions."""
    u, v = _unit(u), _unit(v)
    c = abs(float(np.clip(u @ v, -1.0, 1.0)))
    return math.degrees(math.acos(c)) if lines else math.degrees(math.acos(float(np.clip(u @ v, -1, 1))))
