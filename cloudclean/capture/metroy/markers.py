"""Retro-reflective markers: detection in the rectified IR views, stereo matching and 3D, and marker-map tracking.

In laser mode Revo Metro tracks the scanner by markers only (docs/revo-metro-internals.md, section 7): a laser frame
holds ~17 stripes, far too sparse for geometric (ICP) tracking to hold, but a handful of markers stuck on or around
the part pin the pose exactly. Markers are lit by the scanner's IR fill light (register 0xb07) and return far more
light than any diffuse surface, so they show as bright, filled ellipses.

Detection (per rectified view)
    threshold well above the background, open with a disk that is wider than a laser stripe (so stripes vanish and
    marker blobs stay), keep components that fill their bounding ellipse and are not too elongated, and locate each
    centre by an ellipse fitted to the marker's sub-pixel rim, leaving out the rim pieces a laser stripe touches
    (_rim_fit; the laser lines cross the markers, and an intensity-weighted centroid follows their light).
Stereo
    a marker's partner sits on the same rectified row (within `epipolar_px`) with a similar size, at a disparity that
    means a depth in range, and both blobs are marker-sized at that depth; only pairs that are unique both ways are
    kept, then triangulated through Q. A blob that runs off the image and whose rim could not be fitted is never
    paired (Blob.partial). plausible()/masks() tell which blobs may be markers, so that only those need be kept out
    of the laser line search.
Tracking (MarkerTracker)
    a global marker map in world coordinates. Each frame is registered to it by nearest-marker association from the
    motion-predicted pose and a least-squares rigid fit (Kabsch); when that fails, by matching triangles of
    inter-marker distances (which do not depend on the pose) and a RANSAC fit, as Revo Metro's markerAlignByTriangle
    does. Registered markers refine the map; new ones join it once seen consistently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np


@dataclass
class MarkerParams:
    threshold_above: float = 25.0  # a marker is this much brighter than the image's median (the background). At
                                   # Revo Metro's 200 us exposure near markers read ~100-250 over a background of 16,
                                   # but markers further away and seen at a low angle (the far side of a turntable
                                   # plate) return far less light (retro-reflection falls off with the angle) and
                                   # read ~60-110. On the user's plate + bust recording (2026-10-06): 40 paired 10.2
                                   # markers a frame, 25 paired 17.8 (map 13 -> 23, pose jitter at the surface
                                   # 0.141 -> 0.058 mm RMS, all on the plate within 0.22 mm, p95); 20 began to take
                                   # laser-lit spots on the bust (32 of 700 off the plate)
    threshold_min: float = 40.0
    stripe_width_px: int = 9      # laser stripes are 5-8 px wide: an opening this wide removes them
    min_area_px: int = 40
    max_area_px: int = 6000
    min_fill: float = 0.6         # blob area / area of its fitted ellipse (Revo: saturation)
    min_axis_ratio: float = 0.25  # minor / major axis (Revo: ellipse_ab / minAb). A round marker seen 75 deg off its
                                  # normal is 0.26: the far markers of a plate viewed from the side are that flat
    epipolar_px: float = 1.5      # rectified row difference of a stereo pair (Revo: 1.0..1.5)
    size_ratio: float = 1.6       # left/right blob size may differ by this factor at most
    min_diameter_mm: float = 2.5  # a paired blob's size in mm (area-equivalent, at its triangulated depth). The
                                  # MetroY takes 6 mm markers only; seen at an angle they measure 3.3-5.5 mm
                                  # (p1-p99, the user's plate, 2026-10-09), laser spots on a white part ~1.4 mm
                                  # (median): 2.5 keeps every plate marker and drops 92 % of the spots. Measured at
                                  # the 200 us "General" exposure only: a longer exposure (Dark) or more fill light
                                  # grows both a marker's and a spot's thresholded blob, and is not checked
    z_min: float = 150.0
    z_max: float = 500.0
    centre: str = "rim"           # how a marker's centre is located (detect()): "rim" fits an ellipse to its sub-pixel
                                  # rim, leaving out the rim pieces a laser stripe touches; "centroid" is the former
                                  # intensity-weighted centroid, kept for comparison. The stripes cross the markers
                                  # and move between the two line families, so the centroid landed in two places
                                  # 2.5 px apart (0.34 mm in 3D) while within one family it held to 0.012 px. Static
                                  # plate, 344 + 279 frames (2026-10-09), centroid -> rim, median per marker: centre
                                  # std 0.89/0.75 -> 0.025/0.019 px, 3D std 0.17/0.15 -> 0.007 mm, inter-marker
                                  # distance std 0.108/0.092 -> 0.007 mm; pose jitter 220 mm out 0.074/0.064 ->
                                  # 0.013/0.007 mm (tools/metroy/marker_precision.py)
    rim_step_px: float = 1.0      # one rim profile per this much rim length
    rim_reach_px: float = 5.0     # a profile runs this far inside and outside the rim (less on thin blobs)


@dataclass
class Blob:
    x: float
    y: float
    radius: float                 # equivalent radius, px (of the thresholded, opened blob: min_diameter_mm uses it)
    axis_ratio: float
    rim_rms: float = float("nan")  # px, RMS distance of the rim points from the fitted ellipse; NaN = the rim could
                                   # not be fitted and the centre is a centroid (stripe pixels clipped)
    partial: bool = False          # runs off the image and its rim could not be fitted: the centre is the centroid
                                   # of the part in view, pixels off (1.6 px with a quarter cut off). Masked out of
                                   # the laser line search like any marker, but never paired


def detect(img: np.ndarray, p: MarkerParams | None = None) -> list[Blob]:
    p = p or MarkerParams()
    if p.centre not in ("rim", "centroid"):
        raise ValueError(f"marker centre must be 'rim' or 'centroid', not {p.centre!r}")
    threshold = max(p.threshold_min, float(np.median(img[::8, ::8])) + p.threshold_above)
    if img.dtype == np.uint8:          # img >= threshold, 6x faster than numpy's compare + astype
        bright = cv2.threshold(img, int(np.ceil(threshold)) - 1, 1, cv2.THRESH_BINARY)[1]
    else:
        bright = (img >= threshold).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.stripe_width_px, p.stripe_width_px))
    core = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k)
    # each 8-connected component by its outer contour: the same components as labelling the image, 6x faster
    # (labelling all 1.9 M pixels was most of detect()'s time). Two-level: outer contours (no parent) are the blobs,
    # including any blob that sits inside another one's hole; the second level are the holes
    cnts, hier = cv2.findContours(core, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    H, W = img.shape
    cand = []                          # (blob, ellipse, x0, y0, component mask)
    for c, (_, _, child, parent) in zip(cnts, hier[0] if hier is not None else ()):
        if parent >= 0:
            continue
        x0, y0, w, h = cv2.boundingRect(c)
        if w * h < p.min_area_px:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [c], -1, 1, -1, offset=(-x0, -y0))
        mask &= core[y0:y0 + h, x0:x0 + w]                    # holes are not part of the blob
        if child >= 0:                                        # ... nor a blob inside a hole
            _, lab = cv2.connectedComponents(mask, connectivity=8)
            mask = (lab == lab[c[0, 0, 1] - y0, c[0, 0, 0] - x0]).astype(np.uint8)
        area = cv2.countNonZero(mask)
        if not (p.min_area_px <= area <= p.max_area_px) or len(c) < 6:
            continue
        ell = cv2.fitEllipse(c)
        (_, _), (a, b), _ = ell
        major, minor = max(a, b), min(a, b)
        if major <= 0 or minor / major < p.min_axis_ratio:
            continue
        if area / (np.pi * a * b / 4.0) < p.min_fill:
            continue
        cand.append((Blob(0.0, 0.0, float(np.sqrt(area / np.pi)), float(minor / major)), ell, x0, y0, mask))
    if p.centre == "rim" and cand:
        fit = _rim_fit(img, [e for _, e, *_ in cand], p)
    else:
        fit = [None] * len(cand)
    out = []
    for (blob, ell, x0, y0, mask), f in zip(cand, fit):
        if f is not None:
            blob.x, blob.y, blob.rim_rms = f
        else:
            c = _centroid(img, x0, y0, mask, threshold, clip=p.centre == "rim")
            if c is None:
                continue
            blob.x, blob.y = c
            h, w = mask.shape
            blob.partial = p.centre == "rim" and (x0 == 0 or y0 == 0 or x0 + w == W or y0 + h == H)
        out.append(blob)
    return out


def _centroid(img: np.ndarray, x0: int, y0: int, mask: np.ndarray, threshold: float, clip: bool):
    """Intensity-weighted centroid over the blob grown by a pixel (captures the soft edge symmetrically). With
    `clip`, pixels brighter than the marker's own level (a laser stripe across it) count only with that level."""
    h, w = mask.shape
    grown = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    pad = 1
    ya, yb = max(0, y0 - pad), min(img.shape[0], y0 + h + pad)
    xa, xb = max(0, x0 - pad), min(img.shape[1], x0 + w + pad)
    m = np.zeros((yb - ya, xb - xa), np.float32)
    m[y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w] = grown[:, :]
    roi = img[ya:yb, xa:xb].astype(np.float32)
    if clip:                           # the marker's level: a low quantile of its pixels, as in _rim_pass
        level = float(np.quantile(img[y0:y0 + h, x0:x0 + w][mask > 0], MARKER_LEVEL_Q))
        roi = np.minimum(roi, level + max(10.0, 0.3 * (level - threshold * 0.5)))
    wts = np.clip(roi - threshold * 0.5, 0, None) * m
    s = wts.sum()
    if s <= 0:
        return None
    yy, xx = np.mgrid[ya:yb, xa:xb]
    return float((wts * xx).sum() / s), float((wts * yy).sum() / s)


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """img sampled at (x, y) by bilinear interpolation; NaN outside the image."""
    H, W = img.shape
    ok = (x >= 0) & (y >= 0) & (x <= W - 1) & (y <= H - 1)
    xc, yc = np.where(ok, x, 0).astype(np.float32), np.where(ok, y, 0).astype(np.float32)
    # the cell's corner, at most one short of the last row/column (x = W - 1 then interpolates with weight 1 on it;
    # float32 rounds 1598.99999 up to 1599)
    xi, yi = np.minimum(xc.astype(np.int32), W - 2), np.minimum(yc.astype(np.int32), H - 2)
    fx, fy = xc - xi, yc - yi
    flat = np.ascontiguousarray(img).reshape(-1)
    i = yi * W + xi
    a, b, c, d = (flat[i + o].astype(np.float32) for o in (0, 1, W, W + 1))
    top, bot = a + (b - a) * fx, c + (d - c) * fx
    return np.where(ok, top + (bot - top) * fy, np.float32(np.nan))


MARKER_LEVEL_Q = 0.25


def _rim_fit(img: np.ndarray, ells: list, p: MarkerParams, debug: dict | None = None) -> list:
    """Marker centres from an ellipse fitted to the sub-pixel rim, for all blobs of an image at once.

    The rim is where a marker's own light falls to the background: a laser stripe through the marker's middle never
    touches it, which is why the rim and not the blob's mass locates the centre (Luhmann 2014: a contour ellipse fit
    can drop disturbed edge pieces, a centroid cannot). Short profiles run across the rim along the normals of an
    ellipse; on each, the rim point is where the intensity crosses halfway between the marker's level just inside
    and the background just outside (an iso-contour point: it does not depend on the profile's direction, and
    symmetric blur or a saturated marker moves every rim point alike, not the centre). A profile is left out when
    anything brighter than the marker lies on its inside part or anything brighter than the background on its
    outside part (a stripe crossing the rim, a neighbour, the part's edge), and so are its two neighbours each side,
    which the stripe's soft flank still lifts. An ellipse is fitted to the rim points, trimming those that sit off it
    (3 robust sigma), twice.

    Two passes: the first runs the profiles along the blob's coarse ellipse (its thresholded outline), the second
    along the ellipse the first one found. The outline lies inside the rim - the threshold sits well up a dim
    marker's edge (83 % of a 17 -> 46 edge on the plate's far side), and the 9 px opening shortens the tips of thin
    ellipses - and profiles centred off the rim see the edge's tail on their outside part and are left out: on the
    user's plate a dim far marker kept 29 of 84 profiles in the first pass.

    Returns per blob (x, y, rim RMS px), or None when too little clean rim is left to trust the fit."""
    C0 = np.array([e[0] for e in ells], np.float64)                      # (n, 2)
    A = np.array([0.5 * e[1][0] for e in ells])                          # semi-axis along the angle's direction
    B = np.array([0.5 * e[1][1] for e in ells])
    th = np.radians([e[2] for e in ells])
    C = C0.copy()
    for pas in range(2):
        # (a cheaper first pass - half the profiles, or 11 samples - placed the second worse on dim thin markers)
        theta, used, rms, K, aux = _rim_pass(img, C, A, B, th, p)
        geo = _ellipse(theta, C, A, B, th)
        centre, ok = geo[0], geo[-1]
        ok &= used >= 12
        ok &= np.hypot(*(centre - C0).T) <= np.maximum(1.5, 0.15 * np.minimum(A, B))   # still the blob's own rim
        if pas == 0:                       # re-anchor where the first pass found a sane ellipse
            C = np.where(ok[:, None], centre, C)
            A, B, th = (np.where(ok, v, w) for v, w in zip(geo[1:4], (A, B, th)))
    if debug is not None:
        debug.update(aux)
    ok &= used >= np.maximum(12, 0.35 * K)
    return [(float(centre[i, 0]), float(centre[i, 1]), float(rms[i])) if ok[i] else None for i in range(len(ells))]


def _ellipse(theta: np.ndarray, C: np.ndarray, A: np.ndarray, B: np.ndarray, th: np.ndarray):
    """Conics a u^2 + b uv + c v^2 + d u + e v = 1 in the frame where the ellipse (C, A, B, th) is the unit circle,
    back in pixels: (centre (n, 2), semi-axis A, semi-axis B, angle, valid)."""
    a, b, c, d, e = theta.T
    det = 4 * a * c - b * b
    ok = det > 1e-12
    det = np.where(ok, det, 1.0)
    u0, v0 = (-2 * c * d + b * e) / det, (b * d - 2 * a * e) / det
    eu = np.c_[np.cos(th), np.sin(th)]
    ev = np.c_[-np.sin(th), np.cos(th)]
    centre = C + (A * u0)[:, None] * eu + (B * v0)[:, None] * ev
    Mq = np.stack([np.stack([a, b / 2], -1), np.stack([b / 2, c], -1)], -2)          # (n, 2, 2)
    k = 1 + a * u0 * u0 + b * u0 * v0 + c * v0 * v0
    ok &= k > 0
    J = np.stack([eu / A[:, None], ev / B[:, None]], 1)                               # q = J (p - C)
    Mp = np.einsum("nki,nkl,nlj->nij", J, Mq, J) / np.where(ok, k, 1.0)[:, None, None]
    lam, vec = np.linalg.eigh(Mp)
    ok &= lam[:, 0] > 0
    lam = np.where(ok[:, None], lam, 1.0)
    return centre, 1 / np.sqrt(lam[:, 0]), 1 / np.sqrt(lam[:, 1]), np.arctan2(vec[:, 1, 0], vec[:, 0, 0]), ok


def _rim_pass(img: np.ndarray, C: np.ndarray, A: np.ndarray, B: np.ndarray, th: np.ndarray, p: MarkerParams):
    """One pass of _rim_fit with the profiles along the ellipses (C, A, B, th): (conic per blob (n, 5) in the frame
    where that ellipse is the unit circle, rim points used, their RMS distance px, profiles per blob, details)."""
    n = len(C)
    eu = np.c_[np.cos(th), np.sin(th)]
    ev = np.c_[-np.sin(th), np.cos(th)]
    perim = np.pi * (3 * (A + B) - np.sqrt((3 * A + B) * (A + 3 * B)))
    K = np.clip(np.round(perim / p.rim_step_px), 24, 256).astype(int)
    # all blobs' profiles in one flat list: blob g[i], number j[i] of K[g[i]] around its rim
    start = np.r_[0, np.cumsum(K)[:-1]]
    g = np.repeat(np.arange(n), K)
    j = np.arange(len(g)) - start[g]
    t = 2 * np.pi * j / K[g]
    ct, st = np.cos(t), np.sin(t)
    anchor = C[g] + (A[g] * ct)[:, None] * eu[g] + (B[g] * st)[:, None] * ev[g]
    nrm = (ct / A[g])[:, None] * eu[g] + (st / B[g])[:, None] * ev[g]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    reach = np.clip(0.6 * np.minimum(A, B), 2.5, p.rim_reach_px)
    M = 17
    u = np.linspace(-1.0, 1.0, M)
    s = reach[g][:, None] * u[None, :]                                    # (P, M) offsets along the normal, px
    S = _bilinear(img, anchor[:, :1] + nrm[:, :1] * s, anchor[:, 1:] + nrm[:, 1:] * s)
    inside = np.isfinite(S).all(1)
    S = np.where(inside[:, None], S, 0.0)
    inner, outer = np.sort(S[:, u <= -0.4], 1), np.sort(S[:, u >= 0.4], 1)          # 5 samples each
    hi, lo = inner[:, inner.shape[1] // 2], outer[:, outer.shape[1] // 2]  # medians (odd counts)
    # the marker's own level: a low quantile, not the median - a stripe only ever adds light, and one running along a
    # small marker lit most of its profiles, which then all passed for clean (0.3-0.5 px off, rendered markers)
    hi_b = _group_median(np.where(inside, hi, np.nan), g, n, MARKER_LEVEL_Q)[g]
    lo_b = _group_median(np.where(inside, lo, np.nan), g, n)[g]
    con = hi_b - lo_b
    level = 0.5 * (hi + lo)
    above = S >= level[:, None]
    good = inside & (con > 5)
    # a stripe (or anything else bright) on the inside or outside part of the profile
    good &= inner[:, -1] <= hi_b + np.maximum(10.0, 0.3 * con)
    good &= outer[:, -1] <= lo_b + np.maximum(8.0, 0.2 * con)
    good &= (hi - lo) >= 0.5 * con
    # exactly one crossing, from the marker's level outward to the background's
    good &= above[:, 0] & ~above[:, -1] & (np.count_nonzero(above[:, 1:] != above[:, :-1], axis=1) == 1)
    # the stripe's soft flank lifts its neighbours too: drop two profiles each side of a rejected one
    bad = ~good
    near = bad.copy()
    for sh in (1, 2, -1, -2):
        near |= bad[start[g] + (j + sh) % K[g]]
    good &= ~near
    k = np.clip(np.argmin(above, axis=1), 1, M - 1)                       # first sample below the level
    r_ = np.arange(len(g))
    S0, S1 = S[r_, k - 1], S[r_, k]
    with np.errstate(invalid="ignore", divide="ignore"):
        off = s[r_, k - 1] + (S0 - level) / (S0 - S1) * (s[:, 1] - s[:, 0])
    good &= np.isfinite(off)
    P = anchor + nrm * np.where(good, off, 0.0)[:, None]                 # (P, 2) rim points, px
    # robust ellipse fit, in the frame where the anchor ellipse is the unit circle (well conditioned):
    # a u^2 + b uv + c v^2 + d u + e v = 1, solved per blob by weighted least squares
    d = P - C[g]
    Uq, Vq = (d * eu[g]).sum(1) / A[g], (d * ev[g]).sum(1) / B[g]
    D = np.c_[Uq * Uq, Uq * Vq, Vq * Vq, Uq, Vq]                          # (P, 5)
    DD = D[:, :, None] * D[:, None, :]
    w = good.astype(np.float64)
    for it in range(3):
        G = np.add.reduceat(DD * w[:, None, None], start, 0) + 1e-12 * np.eye(5)
        theta = np.linalg.solve(G, np.add.reduceat(D * w[:, None], start, 0)[..., None])[..., 0]   # (n, 5)
        T = theta[g]
        F = (D * T).sum(1) - 1.0
        gu = 2 * T[:, 0] * Uq + T[:, 1] * Vq + T[:, 3]
        gv = T[:, 1] * Uq + 2 * T[:, 2] * Vq + T[:, 4]
        gp = (gu / A[g])[:, None] * eu[g] + (gv / B[g])[:, None] * ev[g]
        r = F / np.maximum(np.linalg.norm(gp, axis=1), 1e-9)              # Sampson distance, px
        if it == 2:
            break
        sig = 1.4826 * np.nan_to_num(_group_median(np.where(good, np.abs(r), np.nan), g, n), nan=0.0)
        w = (good & (np.abs(r) < np.maximum(3.0 * sig, 0.06)[g])).astype(np.float64)
    used = np.add.reduceat(w, start)
    rms = np.sqrt(np.add.reduceat(w * r * r, start) / np.maximum(used, 1))
    return theta, used, rms, K, {"points": P, "clean": good, "used": w > 0, "blob": g, "residual": r}


def _group_median(v: np.ndarray, g: np.ndarray, n: int, q: float = 0.5) -> np.ndarray:
    """Median (or the q quantile) of v within each group g (g sorted ascending, every group non-empty); NaN values are
    left out, a group with none left gives NaN."""
    fin = np.isfinite(v)
    # one sort of group-offset keys (by group, then value; missing values last): 4x faster than a lexsort
    vs = np.sort(g * 1e7 + np.where(fin, np.clip(v, -1e6, 1e6), 2e6)) - g * 1e7
    size = np.bincount(g, minlength=n)
    cnt = np.bincount(g, weights=fin, minlength=n).astype(int)
    first = np.r_[0, np.cumsum(size)[:-1]]
    pos = q * np.maximum(cnt - 1, 0)
    lo_i = np.floor(pos).astype(int)
    frac = pos - lo_i
    hi_i = np.minimum(lo_i + 1, np.maximum(cnt - 1, 0))
    val = vs[first + lo_i] * (1 - frac) + vs[first + hi_i] * frac
    return np.where(cnt > 0, val, np.nan)


def mask(shape, blobs: list[Blob], grow: float = 1.8, pad_px: float = 4.0) -> np.ndarray:
    """Pixels covered by markers (a little beyond their rim): a marker's bright disc must never be taken for
    laser stripe, or every marker turns into a dense clump of false surface points."""
    m = np.zeros(shape, np.uint8)
    for b in blobs:
        cv2.circle(m, (int(round(b.x)), int(round(b.y))), int(np.ceil(b.radius * grow + pad_px)), 1, -1)
    return m.astype(bool)


def stereo(left: list[Blob], right: list[Blob], Q: np.ndarray, p: MarkerParams | None = None) -> np.ndarray:
    """(n, 3) marker positions in mm (rectified left camera frame) from unique left/right pairs."""
    return stereo_pairs(left, right, Q, p)[0]


def stereo_pairs(left: list[Blob], right: list[Blob], Q: np.ndarray, p: MarkerParams | None = None):
    """stereo(), plus which blobs made the pairs: (points (n, 3), left indices (n,), right indices (n,)). A blob in
    neither list was found in one picture but not matched in the other - the camera view shows it apart."""
    p = p or MarkerParams()
    none = np.zeros(0, int)
    if not left or not right:
        return np.zeros((0, 3)), none, none
    ok, _, big_l, big_r = _candidates(left, right, Q, p)
    # a marker has a known physical size: bright laser spots on a light part pair across the cameras too (they are
    # real surface points), but they are a millimetre or so across, and a marker map that takes them in follows the
    # laser instead of the part. The size test comes BEFORE uniqueness, and holds for both blobs of a candidate: a
    # spot in either view that happens to sit on a marker's row at a plausible depth used to make the marker's pair
    # ambiguous and vetoed a real marker (only the left blob's size was tested, and only after)
    ok &= big_l & big_r
    pairs = [(i, int(np.flatnonzero(ok[i])[0])) for i in range(len(left))
             if ok[i].sum() == 1 and ok[:, np.flatnonzero(ok[i])[0]].sum() == 1]
    if not pairs:
        return np.zeros((0, 3)), none, none
    i, j = np.array(pairs).T
    L = np.array([(b.x, b.y) for b in left])
    R = np.array([(b.x, b.y) for b in right])
    y = 0.5 * (L[i, 1] + R[j, 1])
    h = np.c_[L[i, 0], y, L[i, 0] - R[j, 0], np.ones(len(i))] @ Q.T
    return h[:, :3] / h[:, 3:4], i, j


def _candidates(left: list[Blob], right: list[Blob], Q: np.ndarray, p: MarkerParams):
    """Which left/right blobs could be one marker (same row, similar size, depth in range): (ok (nl, nr), z (nl, nr),
    left blob marker-sized at that depth, right blob marker-sized at that depth)."""
    L = np.array([(b.x, b.y, b.radius) for b in left]).reshape(-1, 3)
    R = np.array([(b.x, b.y, b.radius) for b in right]).reshape(-1, 3)
    dy = np.abs(L[:, None, 1] - R[None, :, 1])
    ratio = np.maximum(L[:, None, 2], R[None, :, 2]) / np.maximum(1e-6, np.minimum(L[:, None, 2], R[None, :, 2]))
    d = L[:, None, 0] - R[None, :, 0]
    w = d * Q[3, 2] + Q[3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(w != 0, Q[2, 3] / w, 0.0)
    ok = (dy < p.epipolar_px) & (ratio < p.size_ratio) & (z > p.z_min) & (z < p.z_max)
    ok &= ~np.array([b.partial for b in left], bool).reshape(-1, 1) & ~np.array([b.partial for b in right], bool)
    f = Q[2, 3]
    return ok, z, 2.0 * L[:, None, 2] * z / f >= p.min_diameter_mm, 2.0 * R[None, :, 2] * z / f >= p.min_diameter_mm


def plausible(left: list[Blob], right: list[Blob], Q: np.ndarray, p: MarkerParams | None = None):
    """Which blobs of each view may be markers: (left (nl,) bool, right (nr,) bool). A blob with a partner in the
    other view is one when it is marker-sized at that partner's depth (min_diameter_mm); a blob without any partner
    when it could be a marker somewhere in the depth range (radius at least that of a min_diameter_mm marker at
    z_max). Laser spots on a light part pair across the views at ~1.4 mm and are left out."""
    p = p or MarkerParams()
    f = Q[2, 3]
    r_far = 0.5 * p.min_diameter_mm * f / p.z_max
    rl = np.array([b.radius for b in left])
    rr = np.array([b.radius for b in right])
    if not left or not right:
        return rl >= r_far, rr >= r_far
    ok, _, big_l, big_r = _candidates(left, right, Q, p)
    keep_l = np.where(ok.any(1), (ok & big_l).any(1), rl >= r_far)
    keep_r = np.where(ok.any(0), (ok & big_r).any(0), rr >= r_far)
    return keep_l, keep_r


def masks(shape, left: list[Blob], right: list[Blob], Q: np.ndarray, p: MarkerParams | None = None):
    """mask() of the left and right view over the blobs that may be markers (plausible()). Masking every blob, as
    before, also cut a hole into the laser line at every bright spot on the part: the spot IS laser line."""
    keep_l, keep_r = plausible(left, right, Q, p)
    return (mask(shape, [b for b, k in zip(left, keep_l) if k]),
            mask(shape, [b for b, k in zip(right, keep_r) if k]))


# -- tracking ----------------------------------------------------------------------------------------------------

def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid 4x4 T minimising |T(src) - dst| (no scale: the markers are metric)."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cd - R @ cs
    return T


def apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


@dataclass
class TrackResult:
    pose: np.ndarray | None       # sensor -> world, None when the frame could not be registered
    state: str                    # "init", "tracked", "relocalized", "lost", "no_markers"
    markers: int                  # markers seen in the frame
    inliers: int                  # of them, registered to the map
    rmse_mm: float | None


@dataclass
class MarkerTracker:
    """Registers each frame's markers to a world marker map.

    Two ways to use it, mirroring Revo Metro:

    * map as you go: the map grows while scanning (new markers join after `confirm` consistent sightings);
    * map first ("marker scan"): sweep once over all markers without fusing surface (`frozen` False), then
      `refine()` the whole map at once and freeze it; the surface scan then tracks against a rigid map that a bad
      frame can never corrupt.

    Wrong registrations are what fuse a surface twice at an offset, so a pose is only accepted with evidence: at
    least `min_inliers` markers within `inlier_mm`, most of the frame's markers agreeing, and - when the pose is
    recovered from scratch - no second, different pose explaining the markers nearly as well (markers laid out in a
    regular grid fit in several places).
    """
    gate_mm: float = 5.0          # association radius after prediction (hand-held motion reaches ~3 mm per frame;
                                  # markers sit 20-40 mm apart)
    inlier_mm: float = 0.5        # a registered marker must land this close to its map position
    min_inliers: int = 3          # while tracking continuously from the predicted pose
    min_inliers_reloc: int = 4    # when recovering the pose from scratch
    min_agree: float = 0.5        # fraction of the frame's markers that must register
    side_tol_mm: float = 0.6      # triangle side lengths must agree this well to be a candidate match
    min_side_mm: float = 8.0      # triangles with shorter sides pin nothing down
    confirm: int = 3              # sightings before a new marker joins the map
    forget_after: int = 10        # lost frames in a row after which the motion prediction is no longer trusted
    frozen: bool = False          # a finished marker map: registered markers no longer move it, none are added
    initial: np.ndarray | None = None
    world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    seen: np.ndarray = field(default_factory=lambda: np.zeros(0))
    version: int = 0              # bumps whenever the map changes (for display)
    _candidates: list = field(default_factory=list)
    _obs: list = field(default_factory=list)        # (frame markers, frame->map index) for refine()
    _T1: np.ndarray | None = None
    _T2: np.ndarray | None = None
    _lost_run: int = 0
    _tri_cache: tuple | None = None

    def reset(self) -> None:
        self.world = np.zeros((0, 3))
        self.seen = np.zeros(0)
        self._candidates, self._obs = [], []
        self._T1 = self._T2 = None
        self._lost_run = 0
        self._tri_cache = None
        self.frozen = False
        self.version += 1

    def track(self, markers: np.ndarray) -> TrackResult:
        m = len(markers)
        if len(self.world) == 0:
            if m < self.min_inliers_reloc or _degenerate(markers):
                return TrackResult(None, "no_markers", m, 0, None)
            T = self.initial if self.initial is not None else np.eye(4)
            self.world = apply(T, markers)
            self.seen = np.ones(m)
            self._obs.append((markers.copy(), np.arange(m)))
            self._T2, self._T1 = self._T1, T
            self.version += 1
            return TrackResult(T, "init", m, m, 0.0)
        if m < self.min_inliers:
            return TrackResult(None, "no_markers", m, 0, None)

        state, T, idx = "tracked", None, None
        for guess in self._predictions():
            T, idx = self._refine(markers, guess, self.min_inliers)
            if T is not None:
                break
        if T is None:
            state = "relocalized"
            T = self._relocalize(markers)
            if T is not None:
                T, idx = self._refine(markers, T, self.min_inliers_reloc)
        if T is None:
            self._lost_run += 1
            if self._lost_run >= self.forget_after:
                self._T1 = self._T2 = None           # the motion since is unknown: recover from the markers alone
            return TrackResult(None, "lost", m, 0, None)
        self._lost_run = 0
        good = idx >= 0
        W = apply(T, markers)
        rmse = float(np.sqrt(np.mean(np.sum((W[good] - self.world[idx[good]]) ** 2, 1))))
        if not self.frozen:
            self._update_map(W, idx, well_posed=good.sum() >= self.min_inliers_reloc and rmse < 0.3)
            if len(self._obs) < 4000:
                self._obs.append((markers.copy(), idx.copy()))
        self._T2, self._T1 = self._T1, T
        return TrackResult(T, state, m, int(good.sum()), rmse)

    def refine(self, iterations: int = 10) -> float:
        """Whole-map adjustment over every frame recorded while mapping: alternately re-solve each frame's pose
        from its markers and each marker from all its sightings. Spreads what incremental mapping accumulated
        (drift) over the map instead of leaving it where the sweep ended. Returns the final RMS residual, mm."""
        obs = [(M, idx) for M, idx in self._obs if (idx >= 0).sum() >= self.min_inliers_reloc]
        rms = float("nan")
        if not obs or len(self.world) < 3:
            return rms
        for _ in range(iterations):
            sums = np.zeros_like(self.world)
            counts = np.zeros(len(self.world))
            poses = []
            for M, idx in obs:
                good = idx >= 0
                T = kabsch(M[good], self.world[idx[good]])
                poses.append(T)
                np.add.at(sums, idx[good], apply(T, M[good]))
                np.add.at(counts, idx[good], 1)
            seen = counts > 0
            self.world[seen] = sums[seen] / counts[seen, None]
            res = [apply(T, M[idx >= 0]) - self.world[idx[idx >= 0]] for (M, idx), T in zip(obs, poses)]
            rms = float(np.sqrt(np.mean(np.concatenate(res) ** 2) * 3))
        self._tri_cache = None
        self.version += 1
        return rms

    def freeze(self) -> dict:
        rms = self.refine()
        self.frozen = True
        self._candidates = []
        return {"markers": int(len(self.world)), "frames": len(self._obs), "rms_mm": rms}

    # -- internals
    def _predictions(self):
        if self._T1 is None:
            return []
        if self._T2 is None:
            return [self._T1]
        motion = self._T1 @ np.linalg.inv(self._T2)
        return [motion @ self._T1, self._T1]

    def _associate(self, W: np.ndarray, radius: float) -> np.ndarray:
        d = np.linalg.norm(W[:, None, :] - self.world[None, :, :], axis=2)
        idx = np.argmin(d, 1)
        idx[d[np.arange(len(W)), idx] > radius] = -1
        # one map marker per frame marker: drop the farther claims
        for j in np.unique(idx[idx >= 0]):
            claim = np.flatnonzero(idx == j)
            if len(claim) > 1:
                keep = claim[np.argmin(d[claim, j])]
                idx[claim[claim != keep]] = -1
        return idx

    def _accept(self, markers: np.ndarray, idx: np.ndarray, need: int, T: np.ndarray) -> bool:
        """Enough markers register, they pin the pose down, and most markers that could have registered did.

        A frame marker far from every map marker is simply not mapped yet: it is no evidence against the pose. Only
        one that lands within the association gate of a map marker without matching it contradicts the pose (two
        real markers never sit 0.5-5 mm apart). Counting every unregistered marker against the pose, as before,
        deadlocked a map started on a frame with few markers: a 4-marker map could never register half of a
        10-marker frame, and only registered frames may add markers (the user's turntable scan, 2026-10-09)."""
        good = idx >= 0
        n = int(good.sum())
        if n < need or _degenerate(markers[good]):
            return False
        if self.frozen or n >= self.min_agree * len(markers):
            # a finished map holds every marker there is: whatever does not register counts against the pose
            return n >= self.min_agree * len(markers)
        rest = apply(T, markers[~good])
        against = int((np.linalg.norm(rest[:, None] - self.world[None], axis=2).min(1) < self.gate_mm).sum()) \
            if len(rest) else 0
        if n < self.min_agree * (n + against) or n < self.min_inliers_reloc:
            return False
        # most of the frame is not explained (new markers coming into view of a small map): take the pose only if
        # the layout pins it down. On a grid or a ring a pose one spacing off registers as many markers and puts the
        # rest in empty map space (review 2026-10-09: 30 mm and 131-228 mm wrong poses accepted from a stale
        # prediction); the triangle search finds the pose that explains most, and refuses when two different ones
        # do equally well
        alt = self._relocalize(markers, use_prediction=False)   # the pose under test came from the prediction
        return alt is not None and _pose_distance(alt, T) < (2.0, 1.0)

    def _refine(self, markers: np.ndarray, T: np.ndarray, need: int):
        radius = self.gate_mm
        for _ in range(4):
            idx = self._associate(apply(T, markers), radius)
            good = idx >= 0
            if good.sum() < need or _degenerate(markers[good]):
                return None, idx
            T = kabsch(markers[good], self.world[idx[good]])
            radius = max(self.inlier_mm * 2, radius * 0.5)
        idx = self._associate(apply(T, markers), self.inlier_mm)
        if not self._accept(markers, idx, need, T):
            return None, idx
        good = idx >= 0
        return kabsch(markers[good], self.world[idx[good]]), idx

    def _map_triangles(self, max_side: float):
        """The map's triangles, sorted by their shortest side (a frame triangle's partners are a binary search
        away instead of a scan of every map triangle)."""
        key = (self.version, len(self.world), round(max_side))
        if self._tri_cache is None or self._tri_cache[0] != key:
            keys, tris = _triangles(self.world, self.min_side_mm, max_side=max_side)
            o = np.argsort(keys[:, 0], kind="stable")
            self._tri_cache = (key, (keys[o], [tris[i] for i in o]))
        return self._tri_cache[1]

    RELOC_TRIANGLES = 200   # frame triangles tried, the largest first: a few already find a random layout's pose,
                            # and on a symmetric one (ring, grid) any of them meets every rotated / shifted copy

    def _relocalize(self, markers: np.ndarray, use_prediction: bool = True) -> np.ndarray | None:
        """Pose from triangles of inter-marker distances (which the pose does not change), accepted only when it
        is clearly better than any other pose - on a regular marker grid several poses fit a few markers.

        Every triangle match used to be fitted and associated, so a 25-marker frame against a 38-marker map took
        1.5 s - on the capture thread, for every lost frame. Now the map's triangles are looked up in a sorted
        list, only the largest frame triangles are tried, and a triangle pose that repeats a pose already found
        (within 10 mm / 3 deg - the copies a symmetric layout produces differ by a marker spacing or a turn) is
        not fitted again."""
        if len(self.world) < 3 or len(markers) < self.min_inliers_reloc:
            return None
        extent = float(np.max(np.linalg.norm(markers[:, None] - markers[None], axis=2)))
        keys_map, tri_map = self._map_triangles(extent + 2 * self.side_tol_mm)
        if not len(keys_map):
            return None
        ft_keys, ft = _triangles(markers, self.min_side_mm)
        if not len(ft_keys):
            return None
        order = np.argsort(-ft_keys[:, 0], kind="stable")[:self.RELOC_TRIANGLES]
        first = keys_map[:, 0]
        cands = []
        for key, tri in zip(ft_keys[order], [ft[i] for i in order]):
            lo = np.searchsorted(first, key[0] - self.side_tol_mm, "left")
            hi = np.searchsorted(first, key[0] + self.side_tol_mm, "right")
            close = lo + np.flatnonzero(np.all(np.abs(keys_map[lo:hi] - key) < self.side_tol_mm, axis=1))
            for c in close[:60]:
                T = kabsch(markers[list(tri)], self.world[list(tri_map[c])])
                if any(_pose_distance(T, Tc) < (10.0, 3.0) for _, Tc in cands):
                    continue
                idx = self._associate(apply(T, markers), self.inlier_mm * 2)
                n = int((idx >= 0).sum())
                if n >= self.min_inliers_reloc:
                    # compare poses fitted to ALL their inliers: from one small triangle, noise alone moves a pose
                    # by millimetres, which would make one true pose look like two
                    for _ in range(3):              # to convergence: the inlier set settles in 1-2 rounds
                        good = idx >= 0
                        T = kabsch(markers[good], self.world[idx[good]])
                        idx = self._associate(apply(T, markers), self.inlier_mm * 2)
                    cands.append((int((idx >= 0).sum()), T))
        if not cands:
            return None
        cands.sort(key=lambda c: -c[0])
        best_n, best = cands[0]
        prediction = self._T1 if use_prediction else None
        full = best_n == len(markers)             # explains every marker in view
        for n, T in cands[1:]:
            if n < best_n - (0 if full else 1):     # a full explanation beats any that leaves a marker out
                break
            if _pose_distance(T, best) > (2.0, 1.0):                    # a genuinely different pose fits as well
                if prediction is not None and _pose_distance(best, prediction) < (20.0, 10.0) \
                        and not _pose_distance(T, prediction) < (20.0, 10.0):
                    continue                                              # the motion so far settles it
                return None
        return best

    def _update_map(self, W: np.ndarray, idx: np.ndarray, well_posed: bool) -> None:
        good = idx >= 0
        j = idx[good]
        w = self.seen[j]
        self.world[j] = (self.world[j] * w[:, None] + W[good]) / (w + 1)[:, None]
        self.seen[j] = np.minimum(w + 1, 50)       # cap: the map keeps adapting slowly
        if not well_posed:
            return                                  # only a firmly registered frame may introduce markers
        fresh = W[~good]
        if len(fresh) and len(self.world):
            # within the gate of a map marker it IS that marker, a little off (two markers never sit 5 mm apart):
            # re-adding it as new made a copy of every plate marker on each turn of the table (444 markers for ~30)
            near = np.linalg.norm(fresh[:, None] - self.world[None], axis=2).min(1) < self.gate_mm
            fresh = fresh[~near]
        kept = []
        for c in self._candidates:
            d = np.linalg.norm(fresh - c[0], axis=1) if len(fresh) else np.zeros(0)
            if len(d) and d.min() < self.inlier_mm * 2:
                k = int(np.argmin(d))
                c = ((c[0] * c[1] + fresh[k]) / (c[1] + 1), c[1] + 1)
                fresh = np.delete(fresh, k, axis=0)
            kept.append(c)
        kept += [(f, 1) for f in fresh]
        promote = [c for c in kept if c[1] >= self.confirm]
        if promote:
            self.world = np.vstack([self.world, [c[0] for c in promote]])
            self.seen = np.r_[self.seen, [c[1] for c in promote]]
            self.version += 1
        self._candidates = [c for c in kept if c[1] < self.confirm][-200:]


class _Dist(tuple):
    def __gt__(self, other):          # "farther than" in either translation (mm) or rotation (deg)
        return self[0] > other[0] or self[1] > other[1]

    def __lt__(self, other):
        return self[0] < other[0] and self[1] < other[1]


def _pose_distance(A: np.ndarray, B: np.ndarray) -> "_Dist":
    dR = A[:3, :3].T @ B[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return _Dist((float(np.linalg.norm(A[:3, 3] - B[:3, 3])), float(angle)))


def _degenerate(P: np.ndarray) -> bool:
    """Three or more markers on (nearly) one line leave a rotation free."""
    if len(P) < 3:
        return True
    s = np.linalg.svd(P - P.mean(0), compute_uv=False)
    return s[1] < 2.0          # mm: the second extent of the set must be real


def _triangles(P: np.ndarray, min_side: float, limit: int = 40000, max_side: float = np.inf):
    """Every triangle of markers with sides in [min_side, max_side]: its sides sorted (the pose-free key) and its
    vertices opposite those sides, in combination order, at most `limit`. Vectorised: the loop it replaces spent
    ~60 ms on 25 markers, most of a relocalization."""
    n = len(P)
    if n < 3:
        return np.zeros((0, 3)), []
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    abc = np.fromiter(combinations(range(n), 3), dtype=np.dtype((np.int64, 3)))
    a, b, c = abc.T
    sides = np.c_[D[b, c], D[a, c], D[a, b]]                     # side k is opposite vertex k
    ok = (sides.min(1) >= min_side) & (sides.max(1) <= max_side)
    sides, abc = sides[ok][:limit], abc[ok][:limit]
    o = np.argsort(sides, axis=1, kind="stable")                  # canonical order: vertices opposite sorted sides
    keys = np.take_along_axis(sides, o, axis=1)
    verts = np.take_along_axis(abc, o, axis=1)
    return keys.reshape(-1, 3), [tuple(v) for v in verts.tolist()]
