"""A thread, groove by groove, from the outline (docs/outline-check.md, Threads groove by groove).

After the outline check has placed the golden model (cloudclean.outline_check: pose, and the roll about the axis
that lines the helix up), the thread's outline on each side of the part is read tooth by tooth, the way a thread is
checked on a shadowgraph. Each side separately (the two sides show the helix half a pitch apart):

- every tooth's position along the axis, from both of its flanks (a tooth moved along the axis moves one flank out
  and the other in; a fatter tooth moves both out): the spacing to the next tooth (individual pitch error) and the
  cumulative lead error from the first full tooth at the thread's start;
- every crest's radius and every root's radius, and every groove's depth: the crests' envelope (a straight line
  through the side's crests, so one nicked crest does not make its neighbours' grooves deeper) down to the root.

Everything is the photo's edge against the golden model's own edge at the same pose: the golden silhouette ray cast
at each outline point, blurred like the photo and read by the same edge finder (outline_check.model_edge), so a
narrow crest or root, which blur pulls in, reads the same in both. Values = golden + deviation. Teeth and grooves
that are not complete on the golden model itself (the chamfered start, the runout under a head) are listed as
excluded and not measured.

What a photo can show:
- Geometry. The teeth are on the outline only where the line of sight is nearly square to the axis: grazing rays
  cross several teeth and the outline closes up to the crests' envelope. On the 1"-8 screw 91251A917 the outline
  shows all 2.06 mm of depth at 85-90 degrees between the axis and the line of sight, 1.57 mm at 80, 0.47 mm at 70,
  0.16 mm at 57 (the MetroY looking 33 degrees down along a screw pointing at it). visible_depth() measures this on
  the golden model at the fitted pose; below a quarter of the depth nothing is measured and the check says how to
  lay the part.
- Perspective. The view is square to the axis only near the point under the camera (and a lying screw is tilted
  by its bigger head), so one photo reads part of a long thread; a tooth or groove past SQUARE_DEG is reported with
  the local angle. Samples where the golden model's own exact silhouette is not at the sample (a root seen past the
  next tooth's crest) are left out.
- Resolution. A root (and a crest) is measured only when its edge is on the clean outline and the blur model
  supports it: changing the photo's blur estimate by its assumed uncertainty (BLUR_REL_U) must move the reading by
  less than BLUR_SHARE of the tolerance (k = 2). Otherwise "groove depth not measurable at X px/mm; needs about Y
  px/mm", Y found by trying the same tests on the golden model with a camera that has more pixels per mm and the
  same blur in pixels. Where blur is not small against a root or crest, a real change of its height reads smaller
  (a filled root is wider, so blur pulls it out less): shape_response() models that on the golden tooth's profile
  and the reading is corrected, half the correction going into U.
- Light. A root is read against the paper seen through the groove; the part's own shadow there (front light beside
  the lens) makes it read shallow. paper_shading() compares the paper next to each sample with open paper further
  out; shaded grooves and teeth are not measured (SHADE_ROOT, SHADE_FLANK).

inspect_thread(...) -> {"sides": [{"teeth": [...], "grooves": [...]}], "summary": {...}, ...} or None (no thread);
summary_rows() turns it into measurement rows; thread_overlay(), thread_chart() draw it; print_thread() prints it.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

from .golden import _judge, _robust_sigma

K_U = 2.0
MIN_FLANK = 2            # outline points on each flank to place a tooth
BLUR_SPAN = (0.8, 1.25)  # the blur estimate is varied this much to see how much a value depends on it
BLUR_REL_U = 0.10        # standard uncertainty of the photo's blur estimate (relative; assumed: the real blur is not
                         # quite Gaussian)
BLUR_SHARE = 0.25        # a crest, root or groove depth is measured only if the blur model's part of its U (k = 2) is
                         # at most this share of the tolerance
SHADE_ROOT = 0.90        # a groove whose paper is darker than this share of open paper (median) is in the part's shadow
SHADE_FLANK = 0.94       # ... a tooth whose flanks' paper is darker than this (10th percentile)
SHADE_FAR_MM = 2.5       # open paper is looked for this far (and 1.5 and 2.2 times as far) beyond the edge
ON_OUTLINE_PX = 0.1      # a thread sample whose golden silhouette is further than this from its contour point (px)
                         # is left out
SQUARE_DEG = 85.0        # teeth further than this from square to the line of sight lose a flank or root to perspective
VISIBLE_SHARE = 0.25     # the teeth must show at least this share of their depth on the outline to be read at all
WINDOW_SHARE = 0.17      # the edge search window on a thread: this share of the pitch either way (at most window_px)
NEED_LADDER = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
EDGE_LOCAL_PX = 0.03     # local edge error that does not average out along a flank or crest (pixel grid,
                         # demosaicing, compression): px per tooth (assumed)
_LOG_SPAN = math.log(BLUR_SPAN[1] / BLUR_SPAN[0])


def _window(pitch_px: float, window_px: float) -> float:
    """The edge search window (px either way) on a thread: within the groove near its root (a sixth of the pitch;
    the outline check's own 0.28 pitch keeps the roots of a 60 degree thread off the clean outline)."""
    return float(np.clip(WINDOW_SHARE * pitch_px, 3.0, window_px))


def thread_zone(golden) -> dict | None:
    """The golden model's thread (its crest feature: pitch, radii, hand, full-thread range, helix phase)."""
    if golden.profile is None:
        return None
    for f in golden.features:
        if f["kind"] == "crest" and f.get("texture") == "thread":
            return f
    return None


def _tip_sign(golden, z: dict) -> float:
    """+1 when the thread starts (the end away from a head) at the low end of the axis, else -1."""
    pr = golden.profile
    if pr["head"] is not None:
        return 1.0 if pr["zones"][pr["head"]]["tc"] > z["tc"] else -1.0
    return 1.0


def view_angle(golden, T: np.ndarray, sheet) -> float | None:
    """The angle (degrees) between a round part's axis and the line of sight to its middle (90: square to it)."""
    from .outline_check import _to_camera

    if golden.profile is None:
        return None
    pr = golden.profile
    Rc, tc = _to_camera(T, sheet)
    Xc = (pr["o"] + pr["a"] * (pr["t0"] + pr["t1"]) / 2) @ Rc.T + tc
    a_c = Rc @ pr["a"]
    return float(np.degrees(np.arccos(np.clip(abs(a_c @ Xc) / np.linalg.norm(Xc), 0.0, 1.0))))


def visible_depth(golden, T: np.ndarray, cam, sheet, z: dict, n: int = 500) -> float:
    """How deep the thread's teeth look on the outline from this camera (mm): the golden silhouette's half-width
    across the projected axis, its in-and-out over each pitch, median along the full thread, mean of both sides."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    from .outline_check import _to_camera

    pr = golden.profile
    Rc, tc = _to_camera(T, sheet)
    Cg = -Rc.T @ tc
    f0, f1 = z["full"]
    tt = np.linspace(f0, f1, n)
    P = pr["o"] + np.outer(tt, pr["a"])
    Pc = P @ Rc.T + tc
    q = cam.project(Pc)
    dq = np.gradient(q, axis=0)
    dq /= np.maximum(np.linalg.norm(dq, axis=1, keepdims=True), 1e-12)
    nn = np.c_[-dq[:, 1], dq[:, 0]]
    ppm = cam.f / Pc[:, 2]
    reach = (z["R"] + 3.0) * ppm * 1.5
    k = max(3, int(round(z["pitch"] / (tt[1] - tt[0]))))
    out = []
    for sd in (1.0, -1.0):
        def solid(v):
            d = cam.rays(q + (sd * v)[:, None] * nn) @ Rc
            return golden.cast(Cg, d)

        lo, hi = np.zeros(n), reach.copy()
        ok = solid(lo) & ~solid(hi)
        for _ in range(24):
            mid = (lo + hi) / 2
            s = solid(mid)
            lo = np.where(s, mid, lo)
            hi = np.where(s, hi, mid)
        w = (lo + hi) / 2 / ppm
        span = maximum_filter1d(w, k) - minimum_filter1d(w, k)
        inner = ok & (tt > f0 + z["pitch"]) & (tt < f1 - z["pitch"])
        if inner.any():
            out.append(float(np.median(span[inner])))
    return float(np.mean(out)) if out else 0.0


def geometry_advice(alpha: float | None, vis: float, depth0: float) -> str:
    """Why the teeth are not on the outline, and how to lay the part."""
    return (f"the thread's teeth are not on the outline from this direction: the line of sight is "
            f"{alpha:.0f}° from the screw's axis and grazing rays cross several teeth, so the outline shows only "
            f"{vis:.2f} mm of the {depth0:.2f} mm tooth depth. Lay the screw across the view (its axis within about "
            "5° of square to the line of sight: with a camera tilted 35°, the screw lying across the tilt, not "
            "pointing at the camera)")


def _robust_line(u: np.ndarray, v: np.ndarray) -> tuple[Callable, Callable]:
    """A robust straight line v(u) (Tukey) and the standard error of its value at u."""
    n = len(u)
    if n < 3:
        m = float(np.median(v)) if n else 0.0
        s = max(_robust_sigma(v) if n > 1 else 0.0, 1e-4)
        return (lambda x: np.full(np.shape(x), m)), (lambda x: np.full(np.shape(x), s / math.sqrt(max(n, 1))))
    um = float(u.mean())
    A = np.c_[np.ones(n), u - um]
    w = np.ones(n)
    sol = np.zeros(2)
    s = 1e-4
    for _ in range(6):
        sw = np.sqrt(w)
        sol = np.linalg.lstsq(A * sw[:, None], v * sw, rcond=None)[0]
        e = v - A @ sol
        s = max(_robust_sigma(e[w > 0]), 1e-4)
        c = 4.685 * s
        w = np.where(np.abs(e) < c, (1 - (e / c) ** 2) ** 2, 0.0)
    good = w > 0
    ng = max(int(good.sum()), 2)
    suu = float(((u[good] - um) ** 2).sum()) or 1.0
    return ((lambda x: sol[0] + sol[1] * (np.asarray(x) - um)),
            (lambda x: s * np.sqrt(1.0 / ng + (np.asarray(x) - um) ** 2 / suu)))


def _fit_tooth(res, s_ax, s_rad, sig):
    """A tooth's shift along the axis and its flanks' radial shift from both flanks' samples (robust least squares):
    (axial mm, radial mm, standard uncertainty of the axial shift)."""
    A = np.c_[s_ax, s_rad]
    w = np.ones(len(res))
    sol = np.zeros(2)
    s = sig
    for _ in range(5):
        sw = np.sqrt(w)
        sol = np.linalg.lstsq(A * sw[:, None], res * sw, rcond=None)[0]
        e = res - A @ sol
        s = max(_robust_sigma(e[w > 0]), 0.5 * sig, 1e-6)
        c = 4.685 * s
        w = np.where(np.abs(e) < c, (1 - (e / c) ** 2) ** 2, 0.0)
    try:
        cov = np.linalg.inv((A * w[:, None]).T @ A) * s * s
    except np.linalg.LinAlgError:
        return float(sol[0]), float(sol[1]), math.inf
    return float(sol[0]), float(sol[1]), float(math.sqrt(max(cov[0, 0], 0.0)))


def _radial(dev: np.ndarray, corr: float, floor: float) -> tuple[float, float]:
    """A crest's or root's radial deviation (median, mm) and its statistical standard uncertainty. floor: one
    sample's noise (mm) from the whole thread's outline, for a crest or root with few samples of its own;
    neighbouring samples within the blur count as one (corr), never fewer than one."""
    n = len(dev)
    s = max(_robust_sigma(dev), floor) if n > 2 else floor
    return float(np.median(dev)), 1.2533 * max(s, 1e-4) * min(corr / math.sqrt(n), 1.0)


def paper_shading(img: np.ndarray, x: np.ndarray, n2: np.ndarray, half: float, model_prof: np.ndarray,
                  far_px: np.ndarray) -> np.ndarray:
    """How bright the paper next to each outline sample is against open paper further out (1: as bright; less: in a
    shadow), where the model shows open paper: the photo's profile across the edge (the edge finder's line) against
    dark + (paper - dark) x the model's blurred share of open paper, the paper level taken far_px further out
    (several distances, the brightest)."""
    from scipy.ndimage import map_coordinates

    t = np.arange(-half, half + 1e-9, 0.25)
    n = len(x)

    def sample(P):
        return map_coordinates(img, [P[..., 1].ravel(), P[..., 0].ravel()], order=1, mode="nearest",
                               prefilter=False).reshape(P.shape[:-1])

    prof = sample(x[:, None, :] + t[None, :, None] * n2[:, None, :])
    dark = np.median(prof[:, t <= -half / 2], axis=1)
    far = np.max(np.stack([sample(x + (half + k * far_px)[:, None] * n2) for k in (1.0, 1.5, 2.2)], 1), 1)
    out = np.full(n, np.nan)
    use = (t >= half / 2)[None, :] & (model_prof > 0.3)
    exp_ = (far - dark)[:, None] * model_prof
    ok = use & (exp_ > 1e-6)
    ratio = np.where(ok, (prof - dark[:, None]) / np.where(ok, exp_, 1.0), np.nan)
    good = ok.any(1)
    out[good] = np.nanmedian(ratio[good], axis=1)
    return out


def shape_response(tm: np.ndarray, m: np.ndarray, centre: float, kind: str, z: dict, sigma_mm: float, half_mm: float,
                   step_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """How the edge finder's reading of a crest or root follows a real change of its height when blur is not small
    against it: the golden tooth's profile (one meridian, tm / m) with its root filled flat (shallower) or deepened,
    or its crest cut flat (lower) or raised, by delta; blurred by sigma; read at its middle along the radius like the
    photo (the same estimator and window). Returns (deltas, reading(delta) - reading(0)), mm. A narrow root reads less
    than its real change (a filled root is wider, so blur pulls it out less than the golden root's)."""
    from scipy.ndimage import gaussian_filter

    from .outline_camera import profile_crossing

    p, R, minor = z["pitch"], z["R"], z["minor"]
    depth0 = R - minor
    d = min(0.01, sigma_mm / 3.0)
    xs = np.arange(centre - p / 2, centre + p / 2 + 1e-9, d)
    r = np.interp(xs, tm, m)
    ys = np.arange(minor - half_mm - 0.5, R + half_mm + 0.5, d)
    y0 = R if kind == "crest" else minor
    offs = np.arange(-half_mm, half_mm + 1e-9, step_mm)
    ix = len(xs) // 2
    deltas = np.round(np.arange(-0.4, 0.4001, 0.02), 4)
    reads = np.full(len(deltas), np.nan)
    for i, dl in enumerate(deltas):
        if kind == "root":
            rr = np.maximum(r, minor + dl) if dl > 0 else r + dl * np.clip(1 - (r - minor) / (0.25 * depth0), 0, 1)
        else:
            rr = np.minimum(r, R + dl) if dl < 0 else r + dl * np.clip(1 - (R - r) / (0.25 * depth0), 0, 1)
        sl = slice(max(ix - int(4 * sigma_mm / d) - 2, 0), ix + int(4 * sigma_mm / d) + 3)
        mat = (ys[:, None] < rr[None, :]).astype(np.float32)
        col = gaussian_filter(mat, sigma_mm / d, mode="nearest")[:, sl][:, ix - sl.start]
        prof = 1.0 - np.interp(y0 + offs, ys, col)
        off, found, _, _ = profile_crossing(prof[None, :], offs, half_mm)
        if found[0]:
            reads[i] = off[0]
    zero = reads[np.argmin(np.abs(deltas))]
    return deltas, reads - zero


def _corrected(meas: float, table) -> tuple[float, float, float]:
    """A crest's or root's measured change turned into its real change through shape_response's table: (real
    change, standard uncertainty of the correction - half of it, as the real defect's shape is not known - and the
    gain: real mm per mm read there, which every uncertainty of the reading is multiplied by)."""
    if table is None:
        return meas, 0.0, 1.0
    deltas, f = table
    ok = np.isfinite(f)
    if ok.sum() < 3:
        return meas, 0.0, 1.0
    dl, ff = deltas[ok], np.maximum.accumulate(f[ok])
    if not (ff[0] <= meas <= ff[-1]):
        return meas, abs(meas), 1.0              # outside what the model covers: no correction, all of it uncertain
    true = float(np.interp(meas, ff, dl))
    slope = np.gradient(ff, dl)
    gain = 1.0 / max(float(np.interp(true, dl, slope)), 0.2)
    return true, 0.5 * abs(true - meas), max(gain, 1.0)


def exact_offsets(golden, T, cam, sheet, x: np.ndarray, n2: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray]:
    """Where the golden model's own exact (unblurred) silhouette ends along each line x + v n2 (px), and whether
    the line runs from inside the part to outside it within half + 2 px."""
    from .outline_check import _to_camera

    Rc, tc = _to_camera(T, sheet)
    Cg = -Rc.T @ tc

    def solid(v):
        return golden.cast(Cg, cam.rays(x + v[:, None] * n2) @ Rc)

    reach = half + 2.0
    lo, hi = np.full(len(x), -reach), np.full(len(x), reach)
    ok = solid(lo) & ~solid(hi)
    for _ in range(24):
        mid = (lo + hi) / 2
        s_ = solid(mid)
        lo = np.where(s_, mid, lo)
        hi = np.where(s_, hi, mid)
    return (lo + hi) / 2, ok


def _thread_samples(golden, T, cam, sheet, z, step, half):
    """The thread's outline samples at pose T: (S, C, indices of the thread's crest / root / flank samples, kind of
    every sample)."""
    from .outline_check import classify, outline_samples

    S = outline_samples(golden, T, cam, sheet, step, half)
    C = classify(golden, S, T, cam, sheet)
    feats = golden.features
    kind = np.array([feats[f]["kind"] if f >= 0 and feats[f]["kind"] in ("crest", "root", "flank")
                     and feats[f].get("zone") == z["zone"] else "" for f in C["feat"]])
    return S, C, np.flatnonzero(kind != ""), kind


def resolution_need(golden, T, cam, sheet, z: dict, sigma: float, tol: float, ppm: float,
                    window_px: float) -> float | None:
    """The px per mm (same blur in pixels) at which the golden thread's roots and crests are on the clean outline
    and the blur model moves the groove depth, the crests and the roots by little enough (BLUR_SHARE of tol): the
    same tests as on the photo, on the golden model alone, with the camera's focal length scaled up. None: more
    than the ladder's top."""
    from .outline_check import _sensitivity, model_edge

    f0, f1 = z["full"]
    for k in NEED_LADDER:
        ck = cam.scaled_f(k)
        pitch_px = z["pitch"] * ppm * k
        half = _window(pitch_px, window_px)
        S, C, idx, kind = _thread_samples(golden, T, ck, sheet, z, max(pitch_px / 16, 0.5), half)
        t = C["t"]
        inner = (t > f0 + z["pitch"]) & (t < f1 - z["pitch"])
        roots = idx[(kind[idx] == "root") & inner[idx]]
        crests = idx[(kind[idx] == "crest") & inner[idx]]
        if len(roots) < 4 or len(crests) < 4:
            continue
        rng = np.random.default_rng(0)
        roots = rng.choice(roots, min(len(roots), 60), replace=False)
        crests = rng.choice(crests, min(len(crests), 60), replace=False)
        sel = np.r_[roots, crests]
        x, n2, X = S["x"][sel], S["n2"][sel], S["X"][sel]
        lo, f_lo = model_edge(golden, T, ck, sheet, x, n2, half, sigma * BLUR_SPAN[0], return_found=True)
        hi, f_hi = model_edge(golden, T, ck, sheet, x, n2, half, sigma * BLUR_SPAN[1], return_found=True)
        if np.mean(f_lo & f_hi) < 0.8:
            continue
        _, _, rhat = golden.axial(X)
        s_rad = _sensitivity(X, rhat, n2, T, ck, sheet)
        g = (hi - lo) / _LOG_SPAN / np.where(np.abs(s_rad) > 1e-9, s_rad, np.nan)
        g_r, g_c = float(np.nanmedian(g[:len(roots)])), float(np.nanmedian(g[len(roots):]))
        if 2 * BLUR_REL_U * max(abs(g_c - g_r), abs(g_c), abs(g_r)) <= BLUR_SHARE * tol:
            return k * ppm
    return None


def inspect_thread(golden, rest: dict, fit, photo, cam, sheet, D_px: float, sigma_blur: float, params,
                   rel_u: float, edges: Callable | None = None, T: np.ndarray | None = None) -> dict | None:
    """The thread of a round part, tooth by tooth and groove by groove on both sides of its outline. rel_u: relative
    standard uncertainty of lengths along the part (print scale, focal length, the part's height: from the check's
    own budget). edges: (x, n2, half) -> (offset px along n2, valid) to use instead of the photo's edge finder (the
    exact silhouette of a known part, for validation: no blur then). T: the golden model's pose (default: the fit's).
    None when the golden model has no thread."""
    from .outline_camera import edge_offsets
    from .outline_check import _sensitivity, _to_camera, model_edge

    z = thread_zone(golden)
    if z is None:
        return None
    tol = params.tolerance
    pr = golden.profile
    p, R, minor = z["pitch"], z["R"], z["minor"]
    hand = z.get("hand_sign", 1.0)
    depth0 = R - minor
    T = golden.rest_frame(rest, fit.p, fit.d) if T is None else T
    sgn = _tip_sign(golden, z)
    f0, f1 = z["full"]
    t_start = f0 if sgn > 0 else f1
    Rc, tc = _to_camera(T, sheet)
    ppm = float(cam.f / ((pr["o"] + pr["a"] * (f0 + f1) / 2) @ Rc.T + tc)[2])
    pitch_px = p * ppm
    alpha = view_angle(golden, T, sheet)
    vis = visible_depth(golden, T, cam, sheet, z)
    blur = 0.0 if edges is not None else float(sigma_blur)
    out = {"pitch": round(p, 5), "major_radius": round(R, 4), "minor_radius": round(minor, 4),
           "depth": round(depth0, 4), "hand": z.get("hand"), "full_thread": [round(f0, 3), round(f1, 3)],
           "numbered_from": "the thread's start (the end away from the head)", "px_per_mm": round(ppm, 3),
           "pitch_px": round(pitch_px, 1), "view_angle_deg": None if alpha is None else round(alpha, 1),
           "visible_depth": round(vis, 3), "blur_px": round(blur, 3), "tolerance": tol, "sides": [],
           "assumed": {"blur_rel_u": BLUR_REL_U, "edge_local_px": EDGE_LOCAL_PX}}
    if vis < VISIBLE_SHARE * depth0:
        out.update(measurable=False, reason=geometry_advice(alpha, vis, depth0))
        return out
    half = _window(pitch_px, params.window_px)
    step = float(np.clip(pitch_px / 30.0, 0.4, 1.5))
    S, C, idx, kind_all = _thread_samples(golden, T, cam, sheet, z, step, half)
    if len(idx) < 20:
        out.update(measurable=False, reason="too little of the thread's outline is clean (free of corners and "
                                            "hidden edges) in this photo")
        return out
    X, x, n2 = S["X"][idx], S["x"][idx], S["n2"][idx]
    kind = kind_all[idx]
    side = C["side"][idx]
    t, _, rhat = golden.axial(X)
    theta = np.arctan2(rhat @ pr["v"], rhat @ pr["u"])
    s_rad = _sensitivity(X, rhat, n2, T, cam, sheet)
    s_ax = _sensitivity(X, np.broadcast_to(pr["a"], X.shape), n2, T, cam, sheet)
    # the golden model's own exact silhouette along each line: a sample whose outline is not where its contour is
    # (a root seen past the next tooth's crest, near the perspective limit) measures something else - left out
    gold_off, gold_ok = exact_offsets(golden, T, cam, sheet, x, n2, half)
    on_outline = gold_ok & (np.abs(gold_off) < ON_OUTLINE_PX)
    if edges is None:
        r_px, valid, _ = edge_offsets(photo.linear, x, n2, half)
        valid &= on_outline
        pred, f_m, mprof = model_edge(golden, T, cam, sheet, x, n2, half, blur, return_profile=True)
        lo, f_lo = model_edge(golden, T, cam, sheet, x, n2, half, blur * BLUR_SPAN[0], return_found=True)
        hi, f_hi = model_edge(golden, T, cam, sheet, x, n2, half, blur * BLUR_SPAN[1], return_found=True)
        model_ok = f_m & f_lo & f_hi
        dlog = (hi - lo) / _LOG_SPAN                # px per unit of ln(blur)
        res = r_px - pred - D_px
        corr = math.sqrt(max(1.0, 2.0 * blur / step))   # neighbouring samples share the blur: not independent
        e_loc = EDGE_LOCAL_PX
        far = np.where(kind == "root", depth0 + SHADE_FAR_MM, SHADE_FAR_MM) * ppm
        shade = paper_shading(photo.linear, x, n2, half, mprof, far)
    else:
        r_px, valid = edges(x, n2, half)
        valid &= on_outline
        model_ok = np.ones(len(x), bool)
        dlog = np.zeros(len(x))
        res = r_px - gold_off
        corr, e_loc = 1.0, 0.0
        shade = np.ones(len(x))
    srad_safe = np.where(np.abs(s_rad) > 1e-9, s_rad, np.nan)
    a_c = Rc @ pr["a"]

    def alpha_at(tv: float) -> float:
        """The angle between the axis and the line of sight to the axis at t = tv (degrees)."""
        Xc = (pr["o"] + pr["a"] * tv) @ Rc.T + tc
        return float(np.degrees(np.arccos(np.clip(abs(a_c @ Xc) / np.linalg.norm(Xc), 0.0, 1.0))))

    need_cache: dict = {}

    def need_val():
        if "v" not in need_cache:
            need_cache["v"] = resolution_need(golden, T, cam, sheet, z, max(blur, 0.3), tol, ppm, params.window_px)
        return need_cache["v"]

    def need_text():
        nd = need_val()
        where = (f"needs about {nd:.0f} px/mm" if nd is not None else
                 f"needs more than {NEED_LADDER[-1] * ppm:.0f} px/mm")
        return (f"groove depth not measurable at {ppm:.1f} px/mm; {where} (iPhone 48 MP backlit, or the scanner "
                "closer)")

    A = {"kind": kind, "res": res, "dev": res / srad_safe, "g_mm": dlog / srad_safe, "valid": valid,
         "model_ok": model_ok, "s_ax": s_ax, "s_rad": s_rad, "shade": shade, "t": t, "on_outline": on_outline,
         "sig": max(_robust_sigma(res[valid]), 0.02) if valid.any() else 0.1, "corr": corr, "e_loc": e_loc}
    ctx = {"golden": golden, "z": z, "sgn": sgn, "t_start": t_start, "tol": tol, "rel_u": rel_u, "params": params,
           "need_text": need_text, "need_val": need_val, "alpha_at": alpha_at, "ppm": ppm,
           "blur_mm": blur / ppm, "half_mm": half / ppm, "step_mm": 0.25 / ppm}
    # each side's helix coordinate, and one sample's scatter about its own crest's or root's median (both sides)
    helix = {}
    for sd in (1.0, -1.0):
        on = side == sd
        if on.sum() >= 10:
            th0 = float(np.angle(np.mean(np.exp(1j * theta[on]))))
            thu = th0 + np.angle(np.exp(1j * (theta - th0)))
            helix[sd] = (th0, (t - z["root_t0"]) / p - hand * thu / (2 * np.pi))
    usable = valid & model_ok & on_outline & np.isfinite(A["dev"])
    A["scatter"] = {}
    for kk, key in (("crest", np.floor), ("root", np.round)):
        e = []
        for sd, (_, sv) in helix.items():
            ii = np.flatnonzero(usable & (side == sd) & (kind == kk))
            k_ = key(sv[ii])
            for v in np.unique(k_):
                g_ = A["dev"][ii[k_ == v]]
                if len(g_) >= 2:
                    e.append((g_ - np.median(g_)) * math.sqrt(len(g_) / (len(g_) - 1)))
        e = np.concatenate(e) if e else np.zeros(0)
        A["scatter"][kk] = float(_robust_sigma(e)) if len(e) >= 6 else None
    for sd, name in ((1.0, "A"), (-1.0, "B")):
        on = side == sd
        if sd not in helix:
            out["sides"].append({"side": name, "teeth": [], "grooves": [],
                                 "note": "this side of the thread is not on the clean outline"})
            continue
        th0, A["s"] = helix[sd]
        one = _read_side(name, on, th0, A, ctx)
        one["px"] = {"x": x[on], "res": res[on] / ppm, "kind": kind[on], "valid": valid[on]}
        one["angle_deg"] = round(math.degrees(th0), 2)
        if edges is None:
            sh = {k: shade[on & (kind == k) & np.isfinite(shade)] for k in ("crest", "flank", "root")}
            one["paper_shading"] = {k: round(float(np.median(v)), 3) if len(v) else None for k, v in sh.items()}
        out["sides"].append(one)
    out["measurable"] = any(tt.get("status") == "measured" for sd_ in out["sides"] for tt in sd_["teeth"])
    if not out["measurable"]:
        out["reason"] = "no tooth of the thread could be placed on the outline (too few clean flank points)"
    if "v" in need_cache:          # asked for when a root or crest was not on the clean outline
        nd = need_cache["v"]
        out["roots_need_px_per_mm"] = None if nd is None else round(nd, 1)
        out["resolution_enough"] = nd is not None and nd <= 1.05 * ppm
    out["summary"] = _summary(out, p, depth0)
    return out


def _read_side(name: str, on: np.ndarray, th0: float, A: dict, ctx: dict) -> dict:
    """One side of the thread (its samples `on`, its outline at angle th0 about the axis): its teeth (axial position
    from both flanks, crest radius) and grooves (root radius, depth below the crests' envelope), against the golden
    thread, complete teeth and grooves only. A: per-sample arrays (helix coordinate s, kind, residual res px, radial
    deviation dev mm, blur dependence g_mm, valid, model_ok, sensitivities s_ax / s_rad px per mm, paper shading);
    ctx: the thread and the photo."""
    from .outline_check import _meridians

    golden, z, sgn, tol, rel_u, params = (ctx[k] for k in ("golden", "z", "sgn", "tol", "rel_u", "params"))
    need_text, alpha_at, t_start = ctx["need_text"], ctx["alpha_at"], ctx["t_start"]
    s, kind, res, dev, g_mm, valid, model_ok, s_ax, s_rad, shade = (
        A[k] for k in ("s", "kind", "res", "dev", "g_mm", "valid", "model_ok", "s_ax", "s_rad", "shade"))
    sig, corr, e_loc = A["sig"], A["corr"], A["e_loc"]
    pr = golden.profile
    p, R, minor = z["pitch"], z["R"], z["minor"]
    hand = z.get("hand_sign", 1.0)
    depth0 = R - minor
    off = hand * th0 / (2 * np.pi)
    # the golden profile along this side's meridian: which teeth and grooves are complete
    tm = np.arange(z["t0"], z["t1"], p / 120)
    m = _meridians(golden.scene, golden.ref.center, pr["o"], pr["a"], pr["u"], pr["v"], tm, np.array([th0]),
                   R + 5)[0]
    k0 = int(math.ceil((z["t0"] - z["root_t0"]) / p - off)) - 1
    k1 = int(math.floor((z["t1"] - z["root_t0"]) / p - off)) + 1

    def t_at(sv):
        return z["root_t0"] + p * (sv + off)

    def env(tc, fn, empty):
        near = np.abs(tm - tc) <= p / 4
        return fn(m[near]) if near.any() else empty

    root_ok = {k: bool(env(t_at(k), np.min, np.inf) <= minor + 0.02) for k in range(k0, k1 + 1)}
    crest_ok = {j: bool(env(t_at(j + 0.5), np.max, 0.0) >= R - 0.02) for j in range(k0, k1 + 1)}
    # how a crest's and a root's readings follow a real change of their height through this photo's blur (a root or
    # crest that is off changes the width blur sees, which the golden model's own blurred edge does not know)
    mids = [k for k in range(k0, k1) if root_ok[k] and crest_ok[k]] or [int(round((k0 + k1) / 2))]
    km = mids[len(mids) // 2]
    tables = {"root": None, "crest": None}
    if ctx["blur_mm"] > 0:
        for kk, c0 in (("root", t_at(km)), ("crest", t_at(km + 0.5))):
            tables[kk] = shape_response(tm, m, c0, kk, z, ctx["blur_mm"], ctx["half_mm"], ctx["step_mm"])

    def where_(tv):
        return ("at the thread's start (chamfer)" if sgn * (tv - t_start) < (z["t1"] - z["t0"]) / 2 else
                "in the runout under the head" if pr["head"] is not None else "at the thread's end")

    def pos(tv):
        return round(sgn * (tv - t_start), 3)

    def perspective(tv) -> str:
        al = alpha_at(tv)
        return (f"the line of sight is {al:.0f}° from the axis here (perspective: it must be within about "
                f"{90 - SQUARE_DEG:.0f}° of square to it); a photo taken square over this part of the thread reads "
                "it") if al < SQUARE_DEG else ""

    def hidden(tv, what):
        """Why `what` is not on the clean outline at t = tv: the view (perspective) or the resolution."""
        why = perspective(tv)
        if why:
            return f"{what} not on the outline here: {why}"
        nd = ctx["need_val"]()
        if nd is None or nd > 1.1 * ctx["ppm"]:
            return f"{what} not on the clean outline; " + need_text()
        return (f"{what} not on the clean outline here (the line of sight is {alpha_at(tv):.0f}° from the axis, near "
                "the limit); a photo taken square over this part of the thread reads it")

    scatter = A["scatter"]

    def shaded(sel, q, limit):
        v = shade[sel & np.isfinite(shade)]
        return float(np.percentile(v, q)) if len(v) and np.percentile(v, q) < limit else None

    teeth, n = [], 0
    for j in sorted(range(k0, k1), key=lambda q: sgn * q):          # from the thread's start
        tc_ = t_at(j + 0.5)
        if not (z["t0"] - p / 2 <= tc_ <= z["t1"] + p / 2):
            continue
        tooth = {"side": name, "t": round(tc_, 4), "position": pos(tc_)}
        if not (root_ok[j] and root_ok[j + 1] and crest_ok[j]):
            tooth.update(status="excluded", reason="incomplete tooth " + where_(tc_))
            teeth.append(tooth)
            continue
        n += 1
        tooth["n"] = n
        fl = on & valid & (kind == "flank") & (np.floor(s) == j)
        trail, lead = fl & (s - j < 0.5), fl & (s - j >= 0.5)
        tooth["flank_points"] = [int(trail.sum()), int(lead.sum())]
        dark = shaded(fl, 10, SHADE_FLANK)
        if dark is not None:
            tooth["flank_reason"] = (f"the paper next to its flanks is {100 * (1 - dark):.0f} % darker than open "
                                     "paper (the part's own shadow: the light is not behind the part or at the "
                                     "lens); light it from behind (light pad)")
        elif trail.sum() >= MIN_FLANK and lead.sum() >= MIN_FLANK:
            sel = trail | lead
            ax_, rf_, uax = _fit_tooth(res[sel], s_ax[sel], s_rad[sel], sig)
            u_loc = e_loc / max(float(np.median(np.abs(s_ax[sel]))), 1e-9)
            tooth.update(axial=sgn * ax_, flank_radial=round(rf_, 4), u_axial=math.hypot(uax * corr, u_loc))
        else:
            why = perspective(tc_)
            tooth["flank_reason"] = "fewer than two clean outline points on one of its flanks" + (
                ": " + why if why else "")
        cr = on & (kind == "crest") & (np.abs(s - (j + 0.5)) < 0.25) & A["on_outline"]
        crv = cr & valid & model_ok
        tooth["crest_points"] = int(crv.sum())
        if crv.any():
            c, uc = _radial(dev[crv], corr, scatter["crest"] if scatter["crest"] is not None else
                            sig / max(float(np.median(np.abs(s_rad[crv]))), 1e-9))
            c, us_c, gain = _corrected(c, tables["crest"])
            srm = max(float(np.median(np.abs(s_rad[crv]))), 1e-9)
            gc = float(np.median(g_mm[crv]))
            ub = BLUR_REL_U * abs(gc)
            tooth.update(_c=c, _uc=gain * math.hypot(uc, e_loc / srm), _gc=gc, _ub=gain * ub,
                         _ue=gain * params.edge_u_px / srm if e_loc else 0.0, _us=us_c, _gain=gain)
            if 2 * ub > BLUR_SHARE * tol:
                tooth["crest_reason"] = (f"the crest's edge depends on the blur model (±{2 * ub:.3f} mm); "
                                         + need_text())
        elif cr.any():
            tooth["crest_reason"] = "its crest's edge was not found in the photo"
        else:
            tooth["crest_reason"] = hidden(tc_, "its crest is")
        teeth.append(tooth)
    placed = [tt for tt in teeth if "axial" in tt]
    first = placed[0] if placed else None
    for k, tooth in enumerate(teeth):
        if tooth.get("status") == "excluded":
            continue
        if "axial" in tooth:
            nxt = next((tt for tt in teeth[k + 1:] if tt.get("status") != "excluded"), None)
            if nxt is not None and "axial" in nxt and nxt["n"] == tooth["n"] + 1:
                err = nxt["axial"] - tooth["axial"]
                U = K_U * math.sqrt(tooth["u_axial"] ** 2 + nxt["u_axial"] ** 2 + (p * rel_u) ** 2)
                tooth.update(pitch=round(p + err, 4), pitch_error=round(err, 4), pitch_U=round(U, 4),
                             pitch_status=_judge(err, U, tol))
            if tooth is not first:
                dist = abs(tooth["t"] - first["t"])
                lead = tooth["axial"] - first["axial"]
                U = K_U * math.sqrt(tooth["u_axial"] ** 2 + first["u_axial"] ** 2 + (dist * rel_u) ** 2)
                tooth.update(lead_error=round(lead, 4), lead_U=round(U, 4), lead_status=_judge(lead, U, tol))
            else:
                tooth.update(lead_error=0.0, lead_U=0.0, lead_status="reference")
        if "_c" in tooth and "crest_reason" not in tooth:
            c = tooth["_c"]
            U = K_U * math.sqrt(tooth["_uc"] ** 2 + tooth["_ub"] ** 2 + tooth["_ue"] ** 2 + tooth["_us"] ** 2
                                + (R * rel_u) ** 2)
            tooth.update(crest_radius=round(R + c, 4), crest_error=round(c, 4), crest_U=round(U, 4),
                         crest_status=_judge(c, U, tol))
        tooth["status"] = "measured" if ("axial" in tooth or "crest_error" in tooth) else "not_measured"
        if tooth["status"] == "not_measured":
            tooth["reason"] = tooth.get("flank_reason") or tooth.get("crest_reason") or "not on the outline"
    for tooth in teeth:          # rounded only now: pitch and lead come from the unrounded shifts
        if "axial" in tooth:
            tooth["axial"] = round(tooth["axial"], 4)
            tooth["u_axial"] = round(tooth["u_axial"], 5)
    # the crests' envelope (a robust straight line along the side), for the grooves' depth
    cr_t = [tt for tt in teeth if "crest_error" in tt]
    env_fn, env_u = _robust_line(np.array([sgn * tt["t"] for tt in cr_t]), np.array([tt["_c"] for tt in cr_t]))
    g_c = float(np.median([tt["_gc"] for tt in cr_t])) if cr_t else 0.0
    us_env = float(np.median([tt["_us"] for tt in cr_t])) if cr_t else 0.0
    gain_c = float(np.median([tt["_gain"] for tt in cr_t])) if cr_t else 1.0
    grooves, gn = [], 0
    for k in sorted(range(k0 + 1, k1), key=lambda q: sgn * q):
        tr = t_at(k)
        if not (z["t0"] <= tr <= z["t1"]):
            continue
        g = {"side": name, "t": round(tr, 4), "position": pos(tr)}
        if not (root_ok[k] and crest_ok[k - 1] and crest_ok[k]):
            g.update(status="excluded", reason="incomplete groove " + where_(tr))
            grooves.append(g)
            continue
        gn += 1
        g["n"] = gn
        rt = on & (kind == "root") & (np.abs(s - k) < 0.25) & A["on_outline"]
        rv = rt & valid & model_ok
        g["root_points"] = int(rv.sum())
        if not rt.any():
            g.update(status="not_measured", reason=hidden(tr, "its root is"))
            grooves.append(g)
            continue
        if not rv.any():
            g.update(status="not_measured",
                     reason=("blur fills the groove: the model's own root edge vanishes; " + need_text()
                             if (rt & valid & ~model_ok).any() else "its root's edge was not found in the photo"))
            grooves.append(g)
            continue
        dark = shaded(rv, 50, SHADE_ROOT)
        if dark is not None:
            g.update(status="not_measured",
                     reason=(f"the paper seen through the groove is {100 * (1 - dark):.0f} % darker than open paper "
                             "(the part's own shadow: the light is not behind the part or at the lens), so its root "
                             "reads shallow; light it from behind (light pad)"))
            grooves.append(g)
            continue
        dr, ur = _radial(dev[rv], corr, scatter["root"] if scatter["root"] is not None else
                         sig / max(float(np.median(np.abs(s_rad[rv]))), 1e-9))
        dr, us_root, gain = _corrected(dr, tables["root"])
        srm = max(float(np.median(np.abs(s_rad[rv]))), 1e-9)
        ur = gain * math.hypot(ur, e_loc / srm)
        g_r = float(np.median(g_mm[rv]))
        ub_root = BLUR_REL_U * abs(g_r)
        if 2 * ub_root > BLUR_SHARE * tol:
            g.update(status="not_measured",
                     reason=f"the root's edge depends on the blur model (±{2 * ub_root:.3f} mm); " + need_text())
            grooves.append(g)
            continue
        ub_root *= gain
        Ur = K_U * math.sqrt(ur ** 2 + ub_root ** 2 + (gain * params.edge_u_px / srm if e_loc else 0.0) ** 2
                             + us_root ** 2 + (minor * rel_u) ** 2)
        g.update(root_radius=round(minor + dr, 4), root_error=round(dr, 4), root_U=round(Ur, 4),
                 root_status=_judge(dr, Ur, tol))
        if not cr_t:
            g.update(status="not_measured", reason="no crest of this side was measured (the depth needs them)")
            grooves.append(g)
            continue
        e_here = float(env_fn(sgn * tr))
        ue_here = float(env_u(sgn * tr)) * corr
        ub = BLUR_REL_U * abs(g_c - g_r) * max(gain, gain_c)
        dd = e_here - dr
        us = math.hypot(us_root, us_env)
        U = K_U * math.sqrt(ue_here ** 2 + ur ** 2 + ub ** 2 + us ** 2 + (depth0 * rel_u) ** 2)
        g.update(depth=round(depth0 + dd, 4), depth_error=round(dd, 4), depth_U=round(U, 4),
                 depth_status=_judge(dd, U, tol), status="measured", u_blur=round(ub, 5))
        grooves.append(g)
    for tt in teeth:
        for key in [q for q in tt if q.startswith("_")]:
            tt.pop(key)
    return {"side": name, "teeth": teeth, "grooves": grooves}


_RANK = {"off": 3, "close": 2, "ok": 1}


def _worst(rows: list[dict], key: str) -> dict | None:
    """The row whose verdict on `key` is worst (off > close > ok), the largest |error| among those."""
    cand = [r for r in rows if f"{key}_error" in r and r.get(f"{key}_status") in _RANK]
    if not cand:
        return None
    return max(cand, key=lambda r: (_RANK[r[f"{key}_status"]], abs(r[f"{key}_error"])))


def _reasons(texts: list, what: str) -> list[str]:
    """Reasons, the same ones counted once (numbers in them shown as their range): "what (12): ..."."""
    import re

    groups: dict[str, list] = {}
    for tx in texts:
        if tx:
            groups.setdefault(re.sub(r"\d+(\.\d+)?", "#", tx), []).append(tx)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        nums = [[float(v) for v in re.findall(r"\d+(?:\.\d+)?", tx)] for tx in items]
        text = key
        for col in zip(*nums):
            lo, hi = min(col), max(col)
            fmt = (lambda v: f"{v:g}")
            text = text.replace("#", fmt(lo) if lo == hi else f"{fmt(lo)}-{fmt(hi)}", 1)
        out.append(f"{what} ({len(items)}): {text}")
    return out


def _summary(out: dict, p: float, depth0: float) -> dict:
    """The thread's summary: mean and worst pitch error, worst cumulative lead error, worst crest, mean and worst
    groove depth error, counts and the reasons for what was not measured."""
    teeth = [tt for s in out["sides"] for tt in s["teeth"]]
    grooves = [g for s in out["sides"] for g in s["grooves"]]
    summ = {"teeth": sum(tt.get("status") != "excluded" for tt in teeth),
            "teeth_excluded": sum(tt.get("status") == "excluded" for tt in teeth),
            "teeth_placed": sum("axial" in tt for tt in teeth),
            "pitches": sum("pitch_error" in tt for tt in teeth),
            "crests_measured": sum("crest_error" in tt for tt in teeth),
            "grooves": sum(g.get("status") != "excluded" for g in grooves),
            "grooves_excluded": sum(g.get("status") == "excluded" for g in grooves),
            "grooves_measured": sum("depth_error" in g for g in grooves)}
    summ["grooves_not_measurable"] = summ["grooves"] - summ["grooves_measured"]
    summ["reasons"] = _reasons([g["reason"] for g in grooves if g.get("status") == "not_measured"],
                               "groove depths not measured")
    summ["reasons"] += _reasons([tt.get("flank_reason") for tt in teeth if tt.get("status") != "excluded"
                                 and "axial" not in tt], "tooth positions (pitch, lead) not measured")
    summ["reasons"] += _reasons([tt.get("crest_reason") for tt in teeth if tt.get("status") != "excluded"],
                                "crests not measured")
    pe = [tt for tt in teeth if "pitch_error" in tt]
    if pe:
        errs = np.array([tt["pitch_error"] for tt in pe])
        u_mean = math.sqrt(float(np.mean([(tt["pitch_U"] / K_U) ** 2 for tt in pe])) / len(pe))
        summ["pitch_mean"] = {"golden": round(p, 5), "value": round(p + float(errs.mean()), 5),
                              "error": round(float(errs.mean()), 5), "U": round(K_U * u_mean, 5)}
    for key, fld in (("pitch_worst", "pitch"), ("lead_worst", "lead"), ("crest_worst", "crest")):
        w = _worst(teeth, fld)
        if w is not None:
            summ[key] = {"side": w["side"], "tooth": w["n"], "position": w["position"], "error": w[f"{fld}_error"],
                         "U": w[f"{fld}_U"], "status": w[f"{fld}_status"]}
    de = [g for g in grooves if "depth_error" in g]
    if de:
        errs = np.array([g["depth_error"] for g in de])
        u_ind = math.sqrt(max(float(np.mean([(g["depth_U"] / K_U) ** 2 - g["u_blur"] ** 2 for g in de])), 0.0)
                          / len(de))
        u = math.sqrt(u_ind ** 2 + float(np.mean([g["u_blur"] for g in de])) ** 2)
        summ["depth_mean"] = {"golden": round(depth0, 4), "value": round(depth0 + float(errs.mean()), 4),
                              "error": round(float(errs.mean()), 4), "U": round(K_U * u, 4)}
        w = _worst(de, "depth")
        summ["depth_worst"] = {"side": w["side"], "groove": w["n"], "position": w["position"],
                               "error": w["depth_error"], "U": w["depth_U"], "status": w["depth_status"]}
    return summ


ROWS = [("pitch_worst", "Thread pitch error, worst tooth", "pitch_error"),
        ("lead_worst", "Thread lead error (cumulative), worst tooth", "lead_error"),
        ("crest_worst", "Thread crest radius error, worst tooth", "crest_error"),
        ("depth_mean", "Groove depth, mean", "depth"),
        ("depth_worst", "Groove depth error, worst groove", "depth_error")]


def summary_rows(thread: dict | None, tol: float) -> list[dict]:
    """The thread's summary as measurement rows (golden / photo / difference / U / verdict): the same rows for every
    photo so several photos combine ("worst" rows keep the worst photo's); not measured, with the reason, when the
    photo cannot show them."""
    rows = []
    if thread is None:
        return rows
    s = thread.get("summary", {}) if thread.get("measurable") else {}
    for key, name, kind in ROWS:
        golden = thread["depth"] if key == "depth_mean" else 0.0
        row = {"kind": kind, "name": name, "golden": round(golden, 4), "features": ["thread (groove by groove)"],
               "combine": "worst" if key.endswith("worst") else "mean"}
        if key not in s:
            if not thread.get("measurable"):
                reason = thread.get("reason", "")
            elif key.startswith("depth"):
                rs = [r for r in s.get("reasons", []) if r.startswith("groove depths")]
                reason = rs[0] if rs else "no groove's depth could be measured"
            elif key == "crest_worst":
                rs = [r for r in s.get("reasons", []) if r.startswith("crests")]
                reason = rs[0] if rs else "no crest could be measured"
            else:
                rs = [r for r in s.get("reasons", []) if r.startswith("tooth positions")]
                reason = rs[0] if rs else "too few teeth were placed on the outline"
            rows.append({**row, "photo": None, "difference": None, "uncertainty": None, "status": "not_measured",
                         "reason": "Not measured from this photo: " + reason})
            continue
        v = s[key]
        U = max(float(v["U"]), 0.0005)
        diff = float(v["error"])
        where = ""
        if "tooth" in v:
            where = f"side {v['side']}, tooth {v['tooth']} at {v['position']:.1f} mm"
        elif "groove" in v:
            where = f"side {v['side']}, groove {v['groove']} at {v['position']:.1f} mm"
        rows.append({**row, "where": where, "photo": round(golden + diff, 4), "difference": round(diff, 4),
                     "uncertainty": round(U, 4), "status": v.get("status") or _judge(diff, U, tol),
                     "budget": {"statistical": round(U / K_U, 5), "camera_and_scale": 0.0, "placement": 0.0,
                                "toner_spread_assumed": 0.0, "edge_assumed": 0.0, "k": K_U}})
    return rows


# --------------------------------------------------------------------------- printing and pictures
def print_thread(thread: dict | None, log=print, photo: str = "") -> None:
    """The per-tooth and per-groove table of each side, as plain text."""
    if thread is None:
        return
    log(f"Thread groove by groove{(' - ' + photo) if photo else ''}: pitch {thread['pitch']:.4f} mm, depth "
        f"{thread['depth']:.3f} mm, {thread['px_per_mm']:.1f} px/mm ({thread['pitch_px']:.0f} px per pitch), line "
        f"of sight {thread['view_angle_deg']}° from the axis, teeth {thread['visible_depth']:.2f} mm deep on the "
        "outline")
    if not thread.get("measurable"):
        log(f"  not measured: {thread.get('reason')}")
        return
    s = thread["summary"]
    log(f"  {s['teeth']} full teeth ({s['teeth_placed']} placed; {s['teeth_excluded']} incomplete ones left out), "
        f"{s['grooves']} full grooves ({s['grooves_measured']} depths measured, {s['grooves_not_measurable']} not "
        f"measurable; {s['grooves_excluded']} incomplete ones left out)")
    for r in s.get("reasons", []):
        log(f"  - {r}")

    def f(v, u=None):
        if v is None:
            return f"{'-':^16}"
        return f"{v:+.4f}" + (f" ±{u:.4f}" if u else "        ")

    for sd in thread["sides"]:
        log(f"  side {sd['side']}" + (f" ({sd['note']})" if sd.get("note") else ""))
        log("    tooth   at mm  pitch error      lead error       crest error      verdicts")
        for tt in sd["teeth"]:
            if tt.get("status") == "excluded":
                continue
            ver = [tt.get("pitch_status"), tt.get("lead_status"), tt.get("crest_status")]
            log(f"    {tt.get('n', '-'):>5} {tt['position']:7.2f}  {f(tt.get('pitch_error'), tt.get('pitch_U'))} "
                f"{f(tt.get('lead_error'), tt.get('lead_U'))} {f(tt.get('crest_error'), tt.get('crest_U'))} "
                + " / ".join(v or "-" for v in ver) + ("" if tt["status"] == "measured" else " (not measured)"))
        log("    groove  at mm  depth            depth error      root error       verdict")
        for g in sd["grooves"]:
            if g.get("status") == "excluded":
                continue
            if g["status"] != "measured":
                log(f"    {g.get('n', '-'):>5} {g['position']:7.2f}  not measured")
                continue
            log(f"    {g['n']:>5} {g['position']:7.2f}  {g['depth']:.4f}           {f(g['depth_error'], g['depth_U'])} "
                f"{f(g.get('root_error'), g.get('root_U'))} {g['depth_status']}")


def _colour(d, tol):
    from .outline_check import _colour as c

    return c(np.atleast_1d(np.asarray(d, np.float64)), tol)


def thread_overlay(photo, thread: dict, path, max_side: int = 2400) -> Path | None:
    """The thread close up: its outline points coloured by how far the photo's edge is out (red, more material) or
    in (blue) from the golden thread (mm across the outline, saturating at the tolerance; grey: no edge found).
    Upright."""
    from PIL import Image, ImageDraw

    from .outline_camera import upright

    if not thread or not thread.get("measurable"):
        return None
    tol = thread["tolerance"]
    sides = [s for s in thread["sides"] if "px" in s]
    if not sides:
        return None
    pts = np.concatenate([s["px"]["x"] for s in sides])
    lo, hi = pts.min(0), pts.max(0)
    pad = 25
    W, H = photo.size
    c0, r0 = int(max(0, lo[0] - pad)), int(max(0, lo[1] - pad))
    c1, r1 = int(min(W, hi[0] + pad)), int(min(H, hi[1] + pad))
    k = max(1.0, 30.0 / max(thread["pitch_px"], 1e-3), 1600.0 / max(c1 - c0, r1 - r0, 1))
    k = min(k, max_side / max(c1 - c0, r1 - r0, 1))
    im = Image.fromarray(photo.grey8[r0:r1, c0:c1]).convert("RGB")
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    rad = max(1.2, min(3.0, k * 0.8))
    for s in sides:
        P = (s["px"]["x"] - [c0, r0]) * k
        col = _colour(s["px"]["res"], tol)
        for (u, v), c, ok in zip(P, col, s["px"]["valid"]):
            fill = tuple(int(q) for q in c) if ok else (160, 160, 160)
            draw.ellipse([u - rad, v - rad, u + rad, v + rad], fill=fill)
    im = upright(im, photo.orientation)
    narrow = im.width < 720                      # the legend's text goes under the colour bar
    out = Image.new("RGB", (im.width, im.height + (58 if narrow else 40)), (250, 250, 248))
    out.paste(im, (0, 0))
    d2 = ImageDraw.Draw(out)
    bar_w = max(60, min(300, im.width - 40))
    y0 = im.height + 8
    for i in range(bar_w):
        tt = (i / (bar_w - 1)) * 2 - 1
        c = tuple(int(q) for q in _colour(tt * tol, tol)[0])
        d2.line([(20 + i, y0), (20 + i, y0 + 10)], fill=c)
    d2.text((20, y0 + 13), f"-{tol:g} mm", fill=(40, 40, 40))
    d2.text((20 + bar_w - 50, y0 + 13), f"+{tol:g} mm", fill=(40, 40, 40))
    s = thread.get("summary", {})
    d2.text((20, y0 + 30) if narrow else (40 + bar_w, y0),
            f"thread: {s.get('teeth_placed', 0)} teeth placed, {s.get('grooves_measured', 0)} of "
            f"{s.get('grooves', 0)} groove depths measured", fill=(40, 40, 40))
    path = Path(path)
    out.save(path)
    return path


def thread_chart(thread: dict, path, width: int = 1000) -> Path | None:
    """Pitch error per tooth with the cumulative lead error (lines), and groove depth error per groove, along the
    thread from its start, both sides, with U (k = 2) and the tolerance."""
    from PIL import Image, ImageDraw

    if not thread or not thread.get("measurable"):
        return None
    tol = thread["tolerance"]
    teeth = [tt for s in thread["sides"] for tt in s["teeth"] if "pitch_error" in tt or "lead_error" in tt]
    grooves = [g for s in thread["sides"] for g in s["grooves"] if "depth_error" in g]
    x_all = [tt["position"] for tt in teeth] + [g["position"] for g in grooves]
    if not x_all:
        return None
    x0, x1 = min(x_all) - 2, max(x_all) + 2
    panel_h, m_l, m_r, m_t = 240, 70, 20, 34
    img = Image.new("RGB", (width, 2 * panel_h + 3 * m_t + 36), (252, 252, 250))
    d = ImageDraw.Draw(img)
    colours = {"A": (37, 99, 235), "B": (234, 88, 12)}
    ink, grid = (45, 45, 45), (215, 215, 210)

    def X(xv):
        return m_l + (xv - x0) / (x1 - x0) * (width - m_l - m_r)

    def panel(top, title, series, extra=()):
        lim = max([abs(v) + u for _, v, u, _ in series] + [abs(v) for v in extra] + [tol * 1.2])

        def Y(yv):
            return top + panel_h / 2 - yv / lim * (panel_h / 2 - 12)

        d.rectangle([m_l, top, width - m_r, top + panel_h], outline=grid)
        for yv in (-tol, 0.0, tol):
            yy = Y(yv)
            d.line([(m_l, yy), (width - m_r, yy)], fill=(200, 60, 60) if yv else (170, 170, 170), width=1)
            d.text((6, yy - 6), f"{yv:+.2f}" if yv else " 0", fill=ink)
        d.text((m_l, top - 18), title, fill=ink)
        for sd, v, u, xv in series:
            c = colours[sd]
            xx, yy = X(xv), Y(v)
            d.line([(xx, Y(v - u)), (xx, Y(v + u))], fill=c, width=1)
            d.ellipse([xx - 3, yy - 3, xx + 3, yy + 3], fill=c)
        return Y

    s1 = [(tt["side"], tt["pitch_error"], tt["pitch_U"], tt["position"]) for tt in teeth if "pitch_error" in tt]
    leads = [tt["lead_error"] for tt in teeth if "lead_error" in tt]
    Y = panel(m_t, "Pitch error per tooth (to the next tooth), mm; lines: cumulative lead error from the first full "
                   "tooth", s1, leads)
    for sd in ("A", "B"):
        pts = sorted((tt["position"], tt["lead_error"]) for tt in teeth if tt["side"] == sd and "lead_error" in tt)
        if len(pts) > 1:
            d.line([(X(a), Y(b)) for a, b in pts], fill=colours[sd], width=1)
    s2 = [(g["side"], g["depth_error"], g["depth_U"], g["position"]) for g in grooves]
    top2 = 2 * m_t + panel_h
    panel(top2, "Groove depth error per groove (crest envelope to root), mm", s2)
    for top in (m_t, top2):                      # mm ticks along the thread
        for xv in range(int(math.ceil(x0 / 10.0)) * 10, int(x1) + 1, 10):
            d.line([(X(xv), top + panel_h), (X(xv), top + panel_h - 5)], fill=(150, 150, 150))
            d.text((X(xv) - 6, top + panel_h + 2), f"{xv}", fill=(110, 110, 110))
    yb = top2 + panel_h + 16
    d.text((m_l, yb), f"mm along the thread from its start ({x0 + 2:.0f} to {x1 - 2:.0f})", fill=ink)
    xl = m_l + 330
    for sd in ("A", "B"):
        d.ellipse([xl, yb + 3, xl + 8, yb + 11], fill=colours[sd])
        d.text((xl + 12, yb), f"side {sd}", fill=ink)
        xl += 70
    d.text((xl + 10, yb), f"red: ±{tol:g} mm tolerance; bars: U (k = 2)", fill=ink)
    path = Path(path)
    img.save(path)
    return path
