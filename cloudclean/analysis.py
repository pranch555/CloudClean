"""Measurement science: screw-thread analysis, standard-thread matching and axis-aware point distances.

Thread analysis works on the scan as captured (a mesh is sampled area-uniformly so every part of its surface
counts equally). Nothing is smoothed into the reported numbers: the axis, centre and pitch come from a
least-squares helix fit to the raw points, and crests / roots / diameters are read from the axial profile of
all points after the helix has been "unwrapped" (every point moved along the axis to where the same thread
flank passes the reference side), which averages thousands of points per profile bin."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

from .edit import screen_selection
from .io import Geometry, estimate_spacing, is_cloud

Log = Callable[[str], None]

FIT_POINTS = 40_000          # points used by the non-linear fits (the profile uses every point)
HARMONICS = 10               # Fourier harmonics of the thread profile in the helix fit
PROFILE_SAMPLES = 600        # max samples of the returned chart profile


# --------------------------------------------------------------------------- small helpers
def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise ValueError("The axis must be a non-zero vector")
    return v / n


def axis_vector(axis) -> np.ndarray:
    """'x' | 'y' | 'z' or a 3-vector -> unit vector."""
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key not in ("x", "y", "z"):
            raise ValueError("The axis must be 'x', 'y', 'z' or a vector [x, y, z]")
        return np.eye(3)["xyz".index(key)]
    a = np.asarray(axis, float).reshape(-1)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("The axis must be 'x', 'y', 'z' or a vector [x, y, z]")
    return _unit(a)


def _basis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.eye(3)[int(np.argmin(np.abs(a)))]
    u = _unit(np.cross(a, helper))
    return u, np.cross(a, u)


def _vec3(value, where: str) -> np.ndarray:
    v = np.asarray(value, float).reshape(-1) if isinstance(value, (list, tuple, np.ndarray)) else None
    if v is None or v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"{where} must be three numbers [x, y, z]")
    return v


def point_distance(points, axis=None) -> dict:
    """Distance between two points: straight, per coordinate and (with an axis) along / across the axis."""
    p = np.asarray(points, float)
    if p.shape != (2, 3) or not np.isfinite(p).all():
        raise ValueError("Give exactly two points, each three finite numbers [x, y, z]")
    d = p[1] - p[0]
    out = {"distance": float(np.linalg.norm(d)), "delta": d.tolist(),
           "dx": float(abs(d[0])), "dy": float(abs(d[1])), "dz": float(abs(d[2]))}
    if axis is not None:
        a = axis_vector(axis)
        along = float(d @ a)
        out.update(axis=a.tolist(), along_axis=abs(along), along_axis_signed=along,
                   perpendicular=float(np.linalg.norm(d - along * a)))
    return out


# --------------------------------------------------------------------------- gathering points
def region_mask(geom: Geometry, region) -> np.ndarray:
    """Boolean mask over points / vertices for a region: screen selection, box, spheres or None (all)."""
    pos = np.asarray(geom.points if is_cloud(geom) else geom.vertices)
    if region is None or region == {}:
        return np.ones(len(pos), bool)
    if not isinstance(region, dict):
        raise ValueError("The region must be an object: {view_projection, polygon}, {box} or {spheres}")
    if "view_projection" in region or "polygon" in region:
        vp, poly = region.get("view_projection"), region.get("polygon")
        vp_arr = np.asarray(vp, float).reshape(-1) if isinstance(vp, (list, tuple)) else np.zeros(0)
        if vp_arr.shape != (16,) or not np.isfinite(vp_arr).all():
            raise ValueError("A screen selection needs view_projection: 16 numbers")
        poly_arr = np.asarray(poly, float) if isinstance(poly, (list, tuple)) else np.zeros((0, 2))
        if poly_arr.ndim != 2 or poly_arr.shape[1] != 2 or len(poly_arr) < 3 or not np.isfinite(poly_arr).all():
            raise ValueError("A screen selection needs a polygon of at least 3 [x, y] points")
        return screen_selection(geom, vp_arr.tolist(), poly_arr.tolist(), bool(region.get("visible_only", False)))
    if "box" in region:
        box = region["box"]
        if not isinstance(box, dict):
            raise ValueError("box must be {min: [x, y, z], max: [x, y, z]}")
        lo, hi = _vec3(box.get("min"), "box.min"), _vec3(box.get("max"), "box.max")
        if np.any(lo > hi):
            raise ValueError("box.min must not be larger than box.max")
        return np.all((pos >= lo) & (pos <= hi), axis=1)
    if "spheres" in region:
        spheres = np.asarray(region["spheres"], float) if isinstance(region["spheres"], (list, tuple)) else None
        if spheres is None or spheres.ndim != 2 or spheres.shape[1] != 4 or len(spheres) == 0 \
                or not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0):
            raise ValueError("spheres must be a list of [x, y, z, radius] with positive radii")
        mask = np.zeros(len(pos), bool)
        for s in spheres:
            mask |= np.sum((pos - s[:3]) ** 2, axis=1) <= s[3] ** 2
        return mask
    raise ValueError("The region must be {view_projection, polygon, visible_only}, {box: {min, max}} "
                     "or {spheres: [[x, y, z, r], ...]}")


def gather_points(geom: Geometry, region=None, max_points: int = 2_000_000, seed: int = 0):
    """(points, normals or None, notes). Meshes give their selected vertices plus area-uniform surface samples
    on the triangles whose vertices are all selected."""
    rng = np.random.default_rng(seed)
    notes: list[str] = []
    mask = region_mask(geom, region)
    if is_cloud(geom):
        pts = np.asarray(geom.points)[mask]
        nrm = np.asarray(geom.normals)[mask] if geom.has_normals() else None
    else:
        verts, tris = np.asarray(geom.vertices), np.asarray(geom.triangles, dtype=np.int64)
        vn = np.zeros_like(verts)
        pts, nrm = verts[mask], None
        keep = mask[tris].all(axis=1) if len(tris) else np.zeros(0, bool)
        if keep.any():
            t = tris[keep]
            a, b, c = verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
            cross = np.cross(b - a, c - a)
            for k in range(3):
                np.add.at(vn, t[:, k], cross)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            count = int(np.clip(2 * mask.sum(), 50_000, max(max_points // 2, 1)))
            pick = rng.choice(len(t), size=count, p=area / area.sum())
            r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
            samples = (a[pick] * (1 - r1)[:, None] + b[pick] * (r1 * (1 - r2))[:, None]
                       + c[pick] * (r1 * r2)[:, None])
            fn = cross[pick] / np.maximum(2 * area[pick], 1e-300)[:, None]
            vsel = vn[mask]
            vsel = vsel / np.maximum(np.linalg.norm(vsel, axis=1), 1e-300)[:, None]
            pts, nrm = np.vstack([pts, samples]), np.vstack([vsel, fn])
            notes.append(f"Mesh: {int(mask.sum()):,} vertices plus {count:,} area-uniform surface samples")
    if len(pts) > max_points:
        idx = np.sort(rng.choice(len(pts), size=max_points, replace=False))
        notes.append(f"Randomly subsampled {len(pts):,} points to {max_points:,} for the analysis")
        pts = pts[idx]
        nrm = nrm[idx] if nrm is not None else None
    return np.asarray(pts, float), (np.asarray(nrm, float) if nrm is not None else None), notes


# --------------------------------------------------------------------------- axis fitting
def _circle_fit(xy: np.ndarray, iterations: int = 4) -> tuple[np.ndarray, float, float]:
    """Robust algebraic circle fit -> (centre, radius, relative MAD of the radial residual)."""
    w = np.ones(len(xy))
    A = np.c_[2 * xy, np.ones(len(xy))]
    rhs = np.sum(xy ** 2, axis=1)
    centre, radius, rel = np.zeros(2), 0.0, np.inf
    for _ in range(iterations):
        sw = np.sqrt(w)
        sol, *_ = np.linalg.lstsq(A * sw[:, None], rhs * sw, rcond=None)
        centre = sol[:2]
        radius = math.sqrt(max(sol[2] + centre @ centre, 1e-300))
        res = np.linalg.norm(xy - centre, axis=1) - radius
        mad = float(np.median(np.abs(res - np.median(res)))) + 1e-12
        rel = mad / radius
        w = 1.0 / np.maximum(1.0, np.abs(res) / (3 * 1.4826 * mad))
    return centre, radius, rel


def _frame(X: np.ndarray, a0, u0, v0, c0, x) -> tuple:
    """Axis a, centre c, axial t, radial vectors q, radius r, angle theta for parameters x (tilt 2, shift 2)."""
    a = _unit(a0 + x[0] * u0 + x[1] * v0)
    c = c0 + x[2] * u0 + x[3] * v0
    d = X - c
    t = d @ a
    q = d - t[:, None] * a
    r = np.linalg.norm(q, axis=1)
    u = _unit(u0 - (u0 @ a) * a)
    v = np.cross(a, u)
    theta = np.arctan2(q @ v, q @ u)
    return a, c, u, v, t, r, theta


def _initial_axis(X: np.ndarray, normals: np.ndarray | None, extent: float, log: Log):
    """Candidates: eigenvectors of sum(n n^T) (smallest for a plain cylinder, largest for a thread, whose flank
    normals are mostly axial) and of the point covariance. The one whose cross-section is most circular wins."""
    cands = []
    if normals is not None and len(normals):
        nn = normals / np.maximum(np.linalg.norm(normals, axis=1), 1e-300)[:, None]
        cands += list(np.linalg.eigh(nn.T @ nn)[1].T)
    cands += list(np.linalg.eigh(np.cov(X.T))[1].T)
    best = None
    for a in cands:
        a = _unit(a)
        u, v = _basis(a)
        xy = np.c_[X @ u, X @ v]
        centre, radius, rel = _circle_fit(xy)
        if not np.isfinite(radius) or radius > 20 * extent:
            continue
        if best is None or rel < best[0]:
            best = (rel, a, u, v, centre[0] * u + centre[1] * v, radius)
    if best is None:
        raise ValueError("No cylindrical surface found in the selection - select the threaded part of a screw or nut")
    return best[1:]


def _cylinder_refine(X, a0, u0, v0, c0, radius):
    def fun(x):
        return _frame(X, a0, u0, v0, c0, x)[5] - x[4]

    x0 = np.array([0.0, 0.0, 0.0, 0.0, radius])
    res0 = fun(x0)
    scale = 1.4826 * float(np.median(np.abs(res0 - np.median(res0)))) + 1e-9
    sol = least_squares(fun, x0, loss="soft_l1", f_scale=scale, x_scale=[1e-2, 1e-2, radius * 1e-2,
                                                                         radius * 1e-2, radius * 1e-2])
    a, c, *_ = _frame(X, a0, u0, v0, c0, sol.x)
    return a, c, float(sol.x[4])


# --------------------------------------------------------------------------- pitch
def _coarse_pitch(t: np.ndarray, theta: np.ndarray, r: np.ndarray, spacing: float) -> float | None:
    """Spectral pitch estimate. The profile is taken in narrow angular sectors (20 deg): the helix moves the
    crests by only P/18 across a sector whatever the pitch, while a full revolution would smear them out."""
    tmin, length = float(t.min()), float(np.ptp(t))
    if length <= 0:
        return None
    b = max(spacing, length / 4096)
    nb = int(math.ceil(length / b)) + 1
    n_fft = 1 << int(math.ceil(math.log2(nb * 8)))
    power = np.zeros(n_fft // 2 + 1)
    sectors = np.floor((theta + math.pi) / (2 * math.pi / 18)).astype(int)
    groups = [sectors == k for k in range(18)]
    groups = [g for g in groups if g.sum() >= max(200, nb)] or [np.ones(len(t), bool)]
    used = 0
    for g in groups:
        idx = np.clip(((t[g] - tmin) / b).astype(int), 0, nb - 1)
        cnt = np.bincount(idx, minlength=nb)
        valid = cnt > 0
        if valid.sum() < max(8, nb // 3):
            continue
        prof = np.bincount(idx, weights=r[g], minlength=nb)[valid] / cnt[valid]
        xs = np.flatnonzero(valid)
        full = np.interp(np.arange(nb), xs, prof)
        lo, hi = xs[0], xs[-1] + 1
        seg = full[lo:hi]
        ii = np.arange(len(seg))
        seg = seg - np.polyval(np.polyfit(ii, seg, 1), ii)
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=n_fft)) ** 2
        if spec.sum() > 0:
            power += spec / spec.sum()
            used += 1
    if not used:
        return None
    freqs = np.fft.rfftfreq(n_fft, d=b)
    allowed = np.flatnonzero((freqs >= 1 / max(length / 2.5, 1e-9)) & (freqs <= 1 / max(3 * b, 0.05)))
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


def _design(t, theta, pitch, hand, harmonics):
    phi = 2 * math.pi * t / pitch - hand * theta
    m = np.arange(1, harmonics + 1)
    mp = phi[:, None] * m
    return np.hstack([np.ones((len(t), 1)), np.cos(mp), np.sin(mp)])


def _profile_fit(t, theta, r, pitch, hand, harmonics):
    A = _design(t, theta, pitch, hand, harmonics)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    return coef, r - A @ coef


def _scan_pitch(t, theta, r, p0) -> tuple[float, int, float, float]:
    """Grid search pitch x handedness around the spectral estimate. Returns (pitch, hand, cost, cost other hand)."""
    grid = p0 * (1 + np.linspace(-0.1, 0.1, 81))
    costs = {}
    for hand in (1, -1):
        costs[hand] = np.array([np.mean(_profile_fit(t, theta, r, p, hand, 6)[1] ** 2) for p in grid])
    hand = 1 if costs[1].min() <= costs[-1].min() else -1
    c = costs[hand]
    i = int(np.argmin(c))
    pitch = grid[i]
    if 0 < i < len(grid) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        if den > 0:
            pitch += 0.5 * (c[i - 1] - c[i + 1]) / den * (grid[1] - grid[0])
    return float(pitch), hand, float(c.min()), float(costs[-hand].min())


# --------------------------------------------------------------------------- profile features
def _crossings(s: np.ndarray, prof: np.ndarray, level: float, min_run: int):
    """Runs above / below `level` (runs shorter than min_run bins are merged away) with sub-bin crossing
    positions between them. Returns list of (above: bool, start_bin, end_bin, s_start, s_end); s_start/s_end are
    None at the ends of the data."""
    above = prof > level
    starts = np.r_[0, np.flatnonzero(np.diff(above.astype(np.int8))) + 1]
    runs = [[bool(above[a]), int(a), int(b)] for a, b in zip(starts, np.r_[starts[1:], len(prof)])]
    changed = True
    while changed and len(runs) > 2:
        changed = False
        for k in range(1, len(runs) - 1):
            if runs[k][2] - runs[k][1] < min_run:
                runs[k - 1][2] = runs[k + 1][2]
                del runs[k:k + 2]
                changed = True
                break
    out = []
    for k, (flag, a, b) in enumerate(runs):
        def cross(i):  # crossing between bin i-1 and i
            y0, y1 = prof[i - 1] - level, prof[i] - level
            frac = y0 / (y0 - y1) if y0 != y1 else 0.5
            return float(s[i - 1] + np.clip(frac, 0, 1) * (s[i] - s[i - 1]))

        s_start = cross(a) if k > 0 else None
        s_end = cross(b) if k < len(runs) - 1 else None
        out.append((flag, a, b, s_start, s_end))
    return out


def _peak(s, prof, a, b, sign):
    """Sub-bin parabolic extreme of prof[a:b] (sign +1 max, -1 min) -> value."""
    seg = sign * prof[a:b]
    i = int(np.argmax(seg))
    val = seg[i]
    if 0 < i < len(seg) - 1:
        y0, y1, y2 = seg[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        if den < 0:
            val = y1 - 0.125 * (y0 - y2) ** 2 / den
    return float(sign * val)


def _linear_pitch(crests: np.ndarray, roots: np.ndarray, pitch: float):
    """Least squares position = pitch * index + offset (separate offsets for crests and roots)."""
    rows, rhs = [], []
    for kind, pos in ((0, crests), (1, roots)):
        if len(pos) == 0:
            continue
        idx = np.round((pos - pos[0]) / pitch)
        for k, p in zip(idx, pos):
            rows.append([k, kind == 0, kind == 1])
            rhs.append(p)
    A, y = np.asarray(rows, float), np.asarray(rhs)
    A = A[:, [0] + [j for j in (1, 2) if A[:, j].any()]]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ sol
    dof = len(y) - A.shape[1]
    se = None
    if dof > 0:
        cov = np.linalg.pinv(A.T @ A) * float(res @ res) / dof
        se = float(math.sqrt(max(cov[0, 0], 0.0)))
    return float(sol[0]), se, res


# --------------------------------------------------------------------------- thread analysis
def thread_analysis(geom: Geometry, region=None, log: Log = print, max_points: int = 2_000_000) -> dict:
    """Measure an external (screw) or internal (nut) thread. Raises ValueError when no thread is found.

    The crest-by-crest analysis reads every crest and root from the unwrapped profile. It needs complete crests,
    and a thread scanned from one side often has none: each crest hides the flank behind it (the real Revo Metro
    bolt scans of 2026-09 were refused with "No thread found"). Then a least-squares helix fit of all points
    (cloudclean.accuracy.fit_thread) answers instead: same result keys, pitch / major / minor from the fit, pitch
    diameter and flank angle only when both flanks were scanned (`method` says which path answered). A crest-by-
    crest result from one flank only (its profile is half interpolated, its crests are few and its diameters
    unreliable) also defers to the helix fit when that works."""
    crest_result = None
    try:
        crest_result = _thread_analysis_crests(geom, region, log, max_points)
        if crest_result["both_flanks"]:
            return crest_result
        reason = ValueError(f"only one flank was scanned, so its profile is incomplete ({crest_result['crest_count']} "
                            "crests found)")
    except ValueError as exc:
        if not str(exc).startswith("No thread found"):
            raise
        reason = exc
    pts, normals, notes = gather_points(geom, region, max_points)
    log(f"Crest-by-crest analysis: {reason}; fitting a helix to all {len(pts):,} points instead")
    from .accuracy import fit_thread

    try:
        th = fit_thread(pts, normals, log=log)
    except ValueError:
        if crest_result is not None:
            return crest_result
        raise reason
    return _helix_thread_result(th, pts, notes, str(reason), log)


def _flank_warning(missing: str, centre: np.ndarray, axis: np.ndarray, length: float) -> str:
    """Plain sentence naming the flank a one-sided scan did not see (missing = '+axis' | '-axis')."""
    end = np.asarray(centre, float) + (1 if missing == "+axis" else -1) * np.asarray(axis, float) * length / 2
    where = ", ".join(f"{x:.1f}" for x in end)
    return (f"Only one flank of the thread was scanned: the flanks facing the end of the thread near ({where}) are "
            "hidden - seen from one end, every crest hides the flank behind it. The pitch diameter and the flank "
            "angle need both flanks: scan the thread from that end as well, or measure a merge of both scans.")


def _missing_flank(normals, axis) -> str | None:
    """'+axis' / '-axis' when (almost) no scanned flank faces that way, else None (both flanks seen)."""
    if normals is None or not len(normals):
        return None
    nn = np.asarray(normals, float)
    nn = nn / np.maximum(np.linalg.norm(nn, axis=1), 1e-300)[:, None]
    na = nn @ np.asarray(axis, float)
    plus, minus = int((na > 0.3).sum()), int((na < -0.3).sum())
    if max(plus, minus) < 100 or min(plus, minus) >= 0.2 * max(plus, minus):
        return None
    return "+axis" if plus < minus else "-axis"


def _helix_thread_result(th: dict, pts: np.ndarray, notes: list, reason: str, log: Log) -> dict:
    """thread_analysis result (same keys) from a helix fit (cloudclean.accuracy.fit_thread)."""
    from .accuracy import thread_offsets

    m = th["model"]
    a, o, u, v = (np.asarray(m[k], float) for k in ("axis", "origin", "u", "v"))
    pitch, hand = float(m["pitch"]), int(m["hand"])
    X = pts - o
    t = X @ a
    q = X - np.outer(t, a)
    ang = np.arctan2(q @ v, q @ u)
    mc, ms = float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))
    theta_ref = math.atan2(ms, mc) if math.hypot(mc, ms) > 0.1 else 0.0
    side = math.cos(theta_ref) * u + math.sin(theta_ref) * v
    t_lo, t_hi = np.percentile(t, [0.5, 99.5])
    t_mid = 0.5 * (t_lo + t_hi)

    def positions(phase):   # axial positions of a thread phase on the reference side, complete turns only
        t0 = pitch * (phase + hand * theta_ref) / (2 * math.pi)
        k = np.arange(math.ceil((t_lo + pitch / 2 - t0) / pitch), math.floor((t_hi - pitch / 2 - t0) / pitch) + 1)
        return t0 + k * pitch

    tc = positions(m["crest_phase"])
    tr = positions(m["root_phase"])
    crests, radii = [], []
    if len(tc):
        offs = thread_offsets(m, pts, np.r_[tc - pitch / 2, tc[-1] + pitch / 2])
        for tk, off in zip(tc, offs):
            if off["axial_offset"] is None:
                continue
            crests.append(tk + off["axial_offset"])
            radii.append(th["major_diameter"] / 2 + off["radial_offset"])
    crests = np.asarray(crests, float)
    per_crest = np.diff(crests).tolist() if len(crests) > 1 else []
    max_dev = 0.0
    if len(crests) >= 3:
        turns = np.round((crests - crests[0]) / pitch)
        max_dev = float(np.max(np.abs(crests - np.polyval(np.polyfit(turns, crests, 1), turns))))
    crest_points = [(o + a * tk + side * rk).tolist() for tk, rk in zip(crests, radii)]

    # unwrapped axial profile (every point moved along the axis to where its flank passes the reference side)
    s_all = t - hand * pitch * np.angle(np.exp(1j * (ang - theta_ref))) / (2 * math.pi)
    r_all = np.linalg.norm(q, axis=1)
    b = float(np.clip(th["spacing"] / 2, pitch / 200, pitch / 16))
    idx = ((s_all - s_all.min()) / b).astype(int)
    cnt = np.bincount(idx)
    good = np.flatnonzero(cnt >= 3)
    mean = np.bincount(idx, weights=r_all)[good] / cnt[good]
    rmax = np.full(len(cnt), -np.inf)
    rmin = np.full(len(cnt), np.inf)
    np.maximum.at(rmax, idx, r_all)
    np.minimum.at(rmin, idx, r_all)
    centres = s_all.min() + (good + 0.5) * b - t_mid
    step = max(1, int(math.ceil(len(good) / PROFILE_SAMPLES)))
    profile = {"t": np.round(centres[::step], 6).tolist(), "r_mean": np.round(mean[::step], 6).tolist(),
               "r_max": np.round(rmax[good][::step], 6).tolist(), "r_min": np.round(rmin[good][::step], 6).tolist()}

    centre = o + a * t_mid
    why = reason[len("No thread found: "):] if reason.startswith("No thread found: ") else reason
    warnings = [f"The crest-by-crest analysis could not use this selection ({why}); the values come from a "
                "least-squares helix fit of all selected points."]
    if not th["both_flanks"]:
        warnings.append(_flank_warning(th["missing_flank"], centre, a, float(t_hi - t_lo)) if th.get("missing_flank")
                        else "Part of the thread profile was not scanned: no pitch diameter or flank angle.")
    score = 0.75
    if th["fit_r2"] < 0.8:
        warnings.append("The thread profile is noisy compared with its depth")
        score *= 0.7
    if th["coverage_deg"] < 90:
        warnings.append(f"The axis is poorly constrained (only {th['coverage_deg']:.0f} deg of the circumference "
                        "selected)")
        score *= 0.75
    if th["inlier_fraction"] < 0.8:
        warnings.append(f"{100 * (1 - th['inlier_fraction']):.0f} % of the selected points do not follow the thread "
                        "- the selection may include other surfaces")
        score *= 0.8
    confidence = "medium" if score >= 0.6 else "low"
    major, minor = th["major_diameter"], th["minor_diameter"]
    result = {
        "kind": th["kind"],
        "handedness": th["handedness"] if th["handedness_confident"] else "unknown",
        "pitch": th["pitch"],
        "pitch_se": th["pitch_se"],
        "pitch_helix_fit": th["pitch"],
        "tpi": 25.4 / th["pitch"],
        "per_crest_pitches": per_crest,
        "pitch_max_deviation": max_dev,
        "crest_count": int(len(crests)),
        "root_count": int(len(tr)),
        "major_diameter": major,
        "minor_diameter": minor,
        "pitch_diameter": th["pitch_diameter"],
        "thread_depth": (major - minor) / 2,
        "flank_angle_deg": th.get("flank_angle_deg"),
        "flank_half_angles_deg": th.get("flank_half_angles_deg"),
        "axis": a.tolist(),
        "center": centre.tolist(),
        "axis_uncertainty_deg": float(th["axis_se_deg"] or 0.0),
        "length": float(t_hi - t_lo),
        "angular_coverage_deg": float(th["coverage_deg"]),
        "points_used": int(len(pts)),
        "spacing": th["spacing"],
        "noise_rms": th["noise"],
        "fit_r2": th["fit_r2"],
        "inlier_fraction": th["inlier_fraction"],
        "crests": (crests - t_mid).tolist(),
        "roots": (tr - t_mid).tolist(),
        "crest_radii": [float(x) for x in radii],
        "crest_points": crest_points,
        "profile": profile,
        "confidence": confidence,
        "confidence_score": round(score, 3),
        "warnings": warnings,
        "notes": list(notes) + [f"helix fit of all points (crest by crest: {why})"],
        "method": "helix fit",
        "both_flanks": bool(th["both_flanks"]),
        "missing_flank": th.get("missing_flank"),
    }
    pd = "n/a" if th["pitch_diameter"] is None else f"{th['pitch_diameter']:.4f}"
    log(f"Helix fit: pitch {th['pitch']:.5f}, major {major:.4f}, minor {minor:.4f}, pitch diameter {pd}, "
        f"{len(crests)} crests located")
    return result


def _thread_analysis_crests(geom: Geometry, region=None, log: Log = print, max_points: int = 2_000_000) -> dict:
    """Crest-by-crest thread analysis (see thread_analysis)."""
    pts, normals, notes = gather_points(geom, region, max_points)
    if len(pts) < 500:
        raise ValueError(f"The selection has only {len(pts)} points - select more of the threaded surface")
    offset = pts.mean(axis=0)
    X = pts - offset
    rng = np.random.default_rng(1)
    sub = np.sort(rng.choice(len(X), size=min(FIT_POINTS, len(X)), replace=False))
    Xs = X[sub]
    spacing = estimate_spacing(pts)
    extent = float(np.linalg.norm(np.ptp(X, axis=0)))
    if spacing <= 0:
        spacing = extent * 1e-4
    log(f"Thread analysis on {len(pts):,} points (spacing {spacing:.4f})")

    ns = normals[sub] if normals is not None else None
    if ns is None:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xs))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
        ns = np.asarray(pc.normals)
    a0, u0, v0, c0, radius = _initial_axis(Xs, ns, extent, log)
    a1, c1, radius = _cylinder_refine(Xs, a0, u0, v0, c0, radius)
    u1, v1 = _basis(a1)
    _, _, _, _, t, r, theta = _frame(Xs, a1, u1, v1, c1, np.zeros(4))
    log(f"Cylinder: radius {radius:.4f}, length {np.ptp(t):.3f}")

    p0 = _coarse_pitch(t, theta, r, spacing)
    if p0 is None:
        raise ValueError("No thread found: the selection is too short along its axis to show a repeating profile")
    pitch, hand, cost, cost_other = _scan_pitch(t, theta, r, p0)
    log(f"Spectral pitch {p0:.4f}, grid pitch {pitch:.5f} ({'right' if hand > 0 else 'left'}-hand candidate)")

    # helix fit: axis tilt (2), axis shift (2), pitch; the profile is a Fourier series solved linearly inside
    def fun(x):
        _, _, _, _, tt, rr, th = _frame(Xs, a1, u1, v1, c1, x[:4])
        return _profile_fit(tt, th, rr, pitch + x[4], hand, HARMONICS)[1]

    res0 = fun(np.zeros(5))
    sigma0 = 1.4826 * float(np.median(np.abs(res0 - np.median(res0)))) + 1e-9
    sol = least_squares(fun, np.zeros(5), loss="soft_l1", f_scale=2 * sigma0,
                        x_scale=[1e-3, 1e-3, spacing, spacing, pitch * 1e-3])
    x = sol.x
    pitch_model = float(pitch + x[4])
    a, c, *_ = _frame(Xs, a1, u1, v1, c1, x[:4])
    residual = sol.fun
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-12
    axis_se_deg = None
    try:
        J = sol.jac
        cov = np.linalg.pinv(J.T @ J) * sigma ** 2
        axis_se_deg = float(np.degrees(math.sqrt(max(cov[0, 0] + cov[1, 1], 0.0))))
    except (np.linalg.LinAlgError, ValueError):
        pass

    # final frame: reference direction = side the points are on, centre at the middle of the selection
    _, _, u, v, t_all, r_all, th_all = _frame(X, a, *_basis(a), c, np.zeros(4))
    radial_mean = np.mean(np.cos(th_all)), np.mean(np.sin(th_all))
    theta_ref = math.atan2(radial_mean[1], radial_mean[0]) if math.hypot(*radial_mean) > 0.1 else 0.0
    u_ref = math.cos(theta_ref) * u + math.sin(theta_ref) * v
    c = c + a * (t_all.min() + t_all.max()) / 2
    _, _, u, v, t_all, r_all, th_all = _frame(X, a, u_ref, np.cross(a, u_ref), c, np.zeros(4))
    coverage = np.unique(np.floor((th_all + math.pi) / (2 * math.pi) * 36).astype(int)).size * 10.0
    s_all = t_all - hand * pitch_model * th_all / (2 * math.pi)
    coef, _ = _profile_fit(t_all[sub], th_all[sub], r_all[sub], pitch_model, hand, HARMONICS)
    model = _design(t_all, th_all, pitch_model, hand, HARMONICS) @ coef
    res_all = r_all - model
    inlier = np.abs(res_all) <= max(5 * sigma, 1e-9)
    inlier_fraction = float(inlier.mean())
    r2 = float(1 - np.var(residual) / max(np.var(r_all[sub]), 1e-300))
    amp = coef[1:]
    model_depth = float(np.ptp(_design(np.linspace(0, pitch_model, 256), np.zeros(256), pitch_model, 1,
                                       HARMONICS) @ np.r_[0.0, amp]))
    log(f"Helix fit: pitch {pitch_model:.5f}, noise {sigma:.4f}, depth {model_depth:.4f}, R2 {r2:.3f}")
    if r2 < 0.3 or model_depth < 3 * sigma or model_depth < spacing * 0.5:
        raise ValueError("No thread found: the surface has no repeating helical profile clearly above the scan "
                         f"noise (profile depth {model_depth:.3f} vs noise {sigma:.3f}). Select the threaded part "
                         "of the screw or nut.")

    # axial profile of every inlier point in the unwrapped frame
    s_in, r_in = s_all[inlier], r_all[inlier]
    b = float(np.clip(spacing / 2, pitch_model / 200, pitch_model / 16))
    smin = float(s_in.min())
    nb = int(math.ceil(np.ptp(s_in) / b)) + 1
    idx = np.clip(((s_in - smin) / b).astype(int), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    mean = np.bincount(idx, weights=r_in, minlength=nb) / np.maximum(cnt, 1)
    rmax = np.full(nb, -np.inf)
    rmin = np.full(nb, np.inf)
    np.maximum.at(rmax, idx, r_in)
    np.minimum.at(rmin, idx, r_in)
    good = cnt >= max(3, int(0.1 * np.median(cnt[cnt > 0])))
    first, last = np.flatnonzero(good)[[0, -1]]
    sl = slice(first, last + 1)
    centres = smin + (np.arange(nb) + 0.5) * b
    S, mean, rmax, rmin, good, cnt = centres[sl], mean[sl], rmax[sl], rmin[sl], good[sl], cnt[sl]
    # bins near the ends of a selection only hold part of the circumference (and, for a clipped selection, only
    # part of the depth): features there are biased, so they must lie where the coverage is complete
    # (flanks hold more surface per axial length than crests, so compare with bins at the same thread phase)
    phase_bin = np.floor((S / pitch_model) % 1.0 * 32).astype(int) % 32
    expected = np.ones(32)
    for k in range(32):
        if (phase_bin == k).any():
            expected[k] = max(np.percentile(cnt[phase_bin == k], 75), 1.0)
    window = max(1, int(round(0.25 * pitch_model / b)))
    ratio = np.convolve(cnt / expected[phase_bin], np.ones(window) / window, mode="same")
    full = ratio >= 0.6
    full_run = max(1, int(round(0.25 * pitch_model / b)))
    xs = np.flatnonzero(good)
    mean = np.interp(np.arange(len(S)), xs, mean[xs])
    rmax = np.interp(np.arange(len(S)), xs, rmax[xs])
    rmin = np.interp(np.arange(len(S)), xs, rmin[xs])

    hi, lo = np.percentile(mean, 95), np.percentile(mean, 5)
    mid = (hi + lo) / 2
    runs = _crossings(S, mean, mid, max(1, int(pitch_model / (6 * b))))
    crests, crest_r, roots, root_r = [], [], [], []
    for flag, i0, i1, s0, s1 in runs:
        if s0 is None or s1 is None or not (0.15 * pitch_model <= s1 - s0 <= 0.85 * pitch_model):
            continue
        if not full[max(i0 - full_run, 0):i1 + full_run].all():
            continue
        if flag:
            peak = _peak(S, mean, i0, i1, 1)
            if peak >= mid + 0.5 * (hi - mid):
                crests.append((s0 + s1) / 2)
                crest_r.append(peak)
        else:
            peak = _peak(S, mean, i0, i1, -1)
            if peak <= mid - 0.5 * (mid - lo):
                roots.append((s0 + s1) / 2)
                root_r.append(peak)
    crests, roots = np.asarray(crests), np.asarray(roots)
    if len(crests) < 2 or len(crests) + len(roots) < 3:
        raise ValueError(f"No thread found: only {len(crests)} complete crest(s) in the selection - select at least "
                         "three threads")
    pitch_lin, pitch_se, lin_res = _linear_pitch(crests, roots, pitch_model)
    ck = np.round((crests - crests[0]) / pitch_lin)
    per_crest = (np.diff(crests) / np.maximum(np.diff(ck), 1)).tolist()

    major = 2 * float(np.percentile(crest_r, 75))
    minor = 2 * float(np.percentile(root_r, 25)) if len(root_r) else None
    span = (S >= crests[0]) & (S <= crests[-1])
    pitch_diameter = 2 * float(np.median(mean[span])) if len(crests) >= 2 and span.sum() > 4 else None

    # flank angles from the raw points folded onto one pitch, between 25 % and 75 % of the depth
    flank = None
    if minor is not None:
        crest_offset = float(np.mean(crests - ck * pitch_lin))
        phase = (s_in - crest_offset) / pitch_lin
        phase = (phase - np.round(phase)) * pitch_lin
        depth = major / 2 - minor / 2
        f = (r_in - minor / 2) / max(depth, 1e-12)
        band = (f > 0.25) & (f < 0.75)
        half = []
        for side in (phase < 0, phase > 0):
            m = band & side
            if m.sum() < 50:
                break
            slope = np.polyfit(r_in[m], phase[m], 1)[0]
            half.append(float(np.degrees(math.atan(abs(slope)))))
        if len(half) == 2:
            flank = half

    kind = "unknown"
    if normals is not None:
        radial = X[sub] - c - np.outer((X[sub] - c) @ a, a)
        radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-300)[:, None]
        dots = np.einsum("ij,ij->i", normals[sub], radial)
        med = float(np.median(dots))
        kind = "external" if med > 0.3 else "internal" if med < -0.3 else "unknown"
    handed_known = cost_other > 1.5 * cost and coverage >= 60
    handedness = ("right" if hand > 0 else "left") if handed_known else "unknown"

    # confidence
    warnings, score = [], 1.0
    missing = _missing_flank(normals, a)
    if missing is not None:   # a profile with one flank interpolated has no valid pitch diameter / flank angle
        pitch_diameter, flank = None, None
        warnings.append(_flank_warning(missing, c + offset, a, float(np.ptp(t_all))))
    n_features = len(crests)
    if n_features < 4:
        warnings.append(f"Only {n_features} complete crests in the selection - the pitch is less certain; "
                        "select more threads if you can")
        score *= 0.6
    if pitch_se is not None and pitch_se > 0.002 * pitch_lin:
        warnings.append(f"The crest positions scatter (pitch standard error {pitch_se:.4f}) - noisy or damaged thread")
        score *= 0.7
    if r2 < 0.8:
        warnings.append("The thread profile is noisy compared with its depth")
        score *= 0.7
    if coverage < 90 or (axis_se_deg is not None and axis_se_deg > 0.5):
        warnings.append(f"The axis is poorly constrained (only {coverage:.0f} deg of the circumference selected)")
        score *= 0.75
    if inlier_fraction < 0.8:
        warnings.append(f"{100 * (1 - inlier_fraction):.0f} % of the selected points do not follow the thread - "
                        "the selection may include other surfaces")
        score *= 0.8
    if len(lin_res) and np.max(np.abs(lin_res)) > 0.1 * pitch_lin:
        warnings.append("Some crests are far from the regular pitch - check for damage or a wrong selection")
        score *= 0.7
    if abs(pitch_lin - pitch_model) > max(3 * (pitch_se or 0), 0.002 * pitch_lin):
        warnings.append(f"Crest spacing ({pitch_lin:.4f}) and helix fit ({pitch_model:.4f}) disagree")
        score *= 0.8
    confidence = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"

    to_world = lambda p: (p + offset).tolist()  # noqa: E731
    crest_points = [to_world(c + a * s + u * crest_r[i]) for i, s in enumerate(crests)]
    group = max(1, int(math.ceil(len(S) / PROFILE_SAMPLES)))
    n_groups = int(math.ceil(len(S) / group))
    edges = np.arange(n_groups) * group
    profile = {"t": np.add.reduceat(S, edges) / np.diff(np.r_[edges, len(S)]),
               "r_mean": np.add.reduceat(mean, edges) / np.diff(np.r_[edges, len(S)]),
               "r_max": np.maximum.reduceat(rmax, edges), "r_min": np.minimum.reduceat(rmin, edges)}
    profile = {k: np.round(v, 6).tolist() for k, v in profile.items()}

    result = {
        "kind": kind,
        "handedness": handedness,
        "pitch": pitch_lin,
        "pitch_se": pitch_se,
        "pitch_helix_fit": pitch_model,
        "tpi": 25.4 / pitch_lin,
        "per_crest_pitches": per_crest,
        "pitch_max_deviation": float(np.max(np.abs(lin_res))) if len(lin_res) else 0.0,
        "crest_count": int(len(crests)),
        "root_count": int(len(roots)),
        "major_diameter": major,
        "minor_diameter": minor,
        "pitch_diameter": pitch_diameter,
        "thread_depth": (major - minor) / 2 if minor is not None else None,
        "flank_angle_deg": sum(flank) if flank else None,
        "flank_half_angles_deg": flank,
        "axis": a.tolist(),
        "center": to_world(c),
        "axis_uncertainty_deg": axis_se_deg,
        "length": float(np.ptp(t_all)),
        "angular_coverage_deg": float(coverage),
        "points_used": int(len(pts)),
        "spacing": spacing,
        "noise_rms": sigma,
        "fit_r2": r2,
        "inlier_fraction": inlier_fraction,
        "crests": crests.tolist(),
        "roots": roots.tolist(),
        "crest_radii": [float(x) for x in crest_r],
        "crest_points": crest_points,
        "profile": profile,
        "confidence": confidence,
        "confidence_score": round(score, 3),
        "warnings": warnings,
        "notes": notes,
        "method": "crest by crest",
        "both_flanks": missing is None,
        "missing_flank": missing,
    }
    log(f"Pitch {pitch_lin:.5f} +/- {pitch_se or 0:.5f} from {len(crests)} crests, major {major:.4f}, "
        f"handedness {handedness}, confidence {confidence}")
    return result


# --------------------------------------------------------------------------- standard threads
_ISO_COARSE = {1: 0.25, 1.1: 0.25, 1.2: 0.25, 1.4: 0.3, 1.6: 0.35, 1.8: 0.35, 2: 0.4, 2.2: 0.45, 2.5: 0.45,
               3: 0.5, 3.5: 0.6, 4: 0.7, 4.5: 0.75, 5: 0.8, 6: 1.0, 7: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 14: 2.0,
               16: 2.0, 18: 2.5, 20: 2.5, 22: 2.5, 24: 3.0, 27: 3.0, 30: 3.5, 33: 3.5, 36: 4.0, 39: 4.0, 42: 4.5,
               45: 4.5, 48: 5.0, 52: 5.0, 56: 5.5, 60: 5.5, 64: 6.0}
_ISO_FINE = {1: [0.2], 1.2: [0.2], 1.4: [0.2], 1.6: [0.2], 1.8: [0.2], 2: [0.25], 2.5: [0.35], 3: [0.35],
             3.5: [0.35], 4: [0.5], 5: [0.5], 6: [0.75], 7: [0.75], 8: [1.0, 0.75], 10: [1.25, 1.0, 0.75],
             12: [1.5, 1.25, 1.0], 14: [1.5, 1.25, 1.0], 16: [1.5, 1.0], 18: [2.0, 1.5, 1.0], 20: [2.0, 1.5, 1.0],
             22: [2.0, 1.5, 1.0], 24: [2.0, 1.5, 1.0], 27: [2.0, 1.5, 1.0], 30: [3.0, 2.0, 1.5, 1.0],
             33: [3.0, 2.0, 1.5], 36: [3.0, 2.0, 1.5], 39: [3.0, 2.0, 1.5], 42: [4.0, 3.0, 2.0, 1.5],
             45: [4.0, 3.0, 2.0, 1.5], 48: [4.0, 3.0, 2.0, 1.5], 52: [4.0, 3.0, 2.0, 1.5], 56: [4.0, 3.0, 2.0, 1.5],
             60: [4.0, 3.0, 2.0, 1.5], 64: [4.0, 3.0, 2.0, 1.5]}
# ASME B1.1: (size, basic major diameter in inches, UNC threads per inch, UNF threads per inch)
_UNIFIED = [("#0", 0.0600, None, 80), ("#1", 0.0730, 64, 72), ("#2", 0.0860, 56, 64), ("#3", 0.0990, 48, 56),
            ("#4", 0.1120, 40, 48), ("#5", 0.1250, 40, 44), ("#6", 0.1380, 32, 40), ("#8", 0.1640, 32, 36),
            ("#10", 0.1900, 24, 32), ("#12", 0.2160, 24, 28), ("1/4", 0.2500, 20, 28), ("5/16", 0.3125, 18, 24),
            ("3/8", 0.3750, 16, 24), ("7/16", 0.4375, 14, 20), ("1/2", 0.5000, 13, 20), ("9/16", 0.5625, 12, 18),
            ("5/8", 0.6250, 11, 18), ("3/4", 0.7500, 10, 16), ("7/8", 0.8750, 9, 14), ("1", 1.0000, 8, 12),
            ("1-1/8", 1.1250, 7, 12), ("1-1/4", 1.2500, 7, 12), ("1-3/8", 1.3750, 6, 12), ("1-1/2", 1.5000, 6, 12)]
# ISO 228-1 (BSPP, 55 deg Whitworth form): (size, major diameter mm, threads per inch)
_BSP = [("G1/16", 7.723, 28), ("G1/8", 9.728, 28), ("G1/4", 13.157, 19), ("G3/8", 16.662, 19),
        ("G1/2", 20.955, 14), ("G5/8", 22.911, 14), ("G3/4", 26.441, 14), ("G7/8", 30.201, 14),
        ("G1", 33.249, 11), ("G1-1/4", 41.910, 11), ("G1-1/2", 47.803, 11), ("G2", 59.614, 11)]
# ISO 965-1: fundamental deviation es of position g and major diameter tolerance Td grade 6 (micrometres)
_ES_G = {0.2: 17, 0.25: 18, 0.3: 18, 0.35: 19, 0.4: 19, 0.45: 20, 0.5: 20, 0.6: 21, 0.7: 22, 0.75: 22, 0.8: 24,
         1.0: 26, 1.25: 28, 1.5: 32, 1.75: 34, 2.0: 38, 2.5: 42, 3.0: 48, 3.5: 53, 4.0: 60, 4.5: 63, 5.0: 71,
         5.5: 75, 6.0: 80}
_TD6 = {0.2: 56, 0.25: 67, 0.3: 75, 0.35: 85, 0.4: 95, 0.45: 100, 0.5: 106, 0.6: 125, 0.7: 140, 0.75: 140,
        0.8: 150, 1.0: 180, 1.25: 212, 1.5: 236, 1.75: 265, 2.0: 280, 2.5: 335, 3.0: 375, 3.5: 425, 4.0: 475,
        4.5: 500, 5.0: 530, 5.5: 560, 6.0: 600}


def _fmt(x: float) -> str:
    return f"{x:g}"


@lru_cache(maxsize=1)
def standard_threads() -> tuple[dict, ...]:
    out = []
    for d, p in _ISO_COARSE.items():
        out.append({"designation": f"M{_fmt(d)}x{_fmt(p)}", "series": "ISO metric coarse", "major_diameter": float(d),
                    "pitch": p, "flank_angle_deg": 60.0})
    for d, pitches in _ISO_FINE.items():
        for p in pitches:
            out.append({"designation": f"M{_fmt(d)}x{_fmt(p)}", "series": "ISO metric fine",
                        "major_diameter": float(d), "pitch": p, "flank_angle_deg": 60.0})
    for size, inch, unc, unf in _UNIFIED:
        for series, tpi in (("UNC", unc), ("UNF", unf)):
            if tpi:
                out.append({"designation": f"{size}-{tpi} {series}", "series": f"Unified {series}",
                            "major_diameter": round(inch * 25.4, 4), "pitch": 25.4 / tpi, "tpi": tpi,
                            "flank_angle_deg": 60.0})
    for size, d, tpi in _BSP:
        out.append({"designation": f"{size} (BSPP)", "series": "BSP parallel (ISO 228)", "major_diameter": d,
                    "pitch": 25.4 / tpi, "tpi": tpi, "flank_angle_deg": 55.0})
    return tuple(out)


def nearest_standard_thread(major_diameter: float, pitch: float, count: int = 3, kind: str = "unknown") -> list:
    """Closest standard threads by relative pitch (weighted most) and major diameter differences."""
    if not (np.isfinite(major_diameter) and np.isfinite(pitch) and major_diameter > 0 and pitch > 0):
        raise ValueError("Major diameter and pitch must be positive numbers")
    ranked = []
    for std in standard_threads():
        dp, dd = pitch - std["pitch"], major_diameter - std["major_diameter"]
        score = (dp / std["pitch"] / 0.02) ** 2 + (dd / std["major_diameter"] / 0.04) ** 2
        ranked.append((score, std, dp, dd))
    ranked.sort(key=lambda x: x[0])
    out = []
    for score, std, dp, dd in ranked[:count]:
        item = dict(std, pitch_deviation=dp, major_diameter_deviation=dd, score=round(float(score), 4),
                    match="close" if score < 1 else "possible" if score < 6 else "poor")
        p = std["pitch"]
        if std["series"].startswith("ISO metric") and p in _TD6:
            d = std["major_diameter"]
            tol = {}
            if kind in ("external", "unknown"):
                hi = d - _ES_G[p] / 1000
                lo = hi - _TD6[p] / 1000
                tol["6g"] = {"major_min": round(lo, 4), "major_max": round(hi, 4),
                             "within": bool(lo <= major_diameter <= hi)}
            if kind in ("internal", "unknown"):
                tol["6H"] = {"major_min": d, "within": bool(major_diameter >= d)}
            item["tolerance"] = tol
        out.append(item)
    return out
