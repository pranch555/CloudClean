"""MetroY laser frame -> 3D points in millimetres, with sub-pixel stripe centres that hold up to measurement.

The stripes run diagonally across the image (about 30 degrees from vertical) and are 5-8 px wide. Locating each one
independently per row with a three-pixel parabola throws away most of the signal: laser speckle makes any single
row's profile lumpy, and the parabola is biased whenever the profile is not a parabola. Here instead:

1. Steger's ridge detector. Smooth with a 2D Gaussian (which already averages a few pixels *along* the stripe),
   take the Hessian, and use its dominant eigenvector as the stripe normal n. The centre is where the derivative of
   the image along n crosses zero. Evaluated along each rectified row, that zero crossing is exactly the sub-pixel x
   at which the stripe meets the epipolar line - which is what triangulation needs, with no per-row peak model.
2. Along-stripe fit. The centres are linked row to row into tracks, and each track is smoothed with a local
   quadratic (Savitzky-Golay) over a few rows. A stripe on a real surface is a smooth curve at this scale, so this
   averages speckle further without moving the centre. Tracks are cut wherever the stripe jumps (an edge or an
   occlusion), so the smoothing never bridges two surfaces.
3. Correspondence by laser line identity. All ~17 stripes of a frame look alike, so pairing them by order alone
   slips by a stripe wherever one view sees a stripe the other does not (steps, occlusions) - and a one-stripe slip
   is a ~40 mm error that still looks like a surface. The scanner's own laser calibration maps each line from the
   left view to the right one (laserfile); the line the right view confirms along a whole track fixes the pairing.
   The measured centre pair is then triangulated through the factory Q, and a neighbourhood agreement filter
   removes what is left of mismatches. (Order-preserving matching per row remains as the fallback without the
   laser calibration.)

Everything that depends only on the calibration (rectification maps) is computed once in Triangulator, so a stream
can call points() per frame.

Used by the MetroY capture driver; tools/metroy/ has the command line wrappers and validation scripts.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# the depth range the laser lines are calibrated over (metroExtra.bin ROIs span ~218..430 mm; Revo Metro's cross
# presets use 200..400). Outside it a stripe's line identity is extrapolated, so points there are not kept.
Z_MIN, Z_MAX = 190.0, 450.0


def read_calibration(path_or_text: str | bytes):
    """(calib dict, P1, P2, Q) from Revopoint's camparam.yaml (OpenCV FileStorage), a path or the file's contents."""
    if isinstance(path_or_text, bytes):
        path_or_text = path_or_text.decode("utf-8", "replace")
    in_memory = path_or_text.lstrip().startswith("%YAML")
    flags = cv2.FILE_STORAGE_READ | (cv2.FILE_STORAGE_MEMORY if in_memory else 0)
    fs = cv2.FileStorage(path_or_text, flags)
    if not fs.isOpened():
        raise ValueError("cannot read the stereo calibration")
    get = lambda k: fs.getNode(k).mat()
    # Revopoint's own key names, including the "stero" spelling
    calib = {"KL": get("cameraMatrixL"), "DL": get("distCoeffL"), "KR": get("cameraMatrixR"),
             "DR": get("distCoeffR"), "R": get("steroRotation"), "T": get("steroTranslation"),
             "size": (int(fs.getNode("width").string()), int(fs.getNode("height").string()))}
    P1, P2, Q = get("pL"), get("pR"), get("Q")
    missing = [k for k, v in {**calib, "pL": P1, "pR": P2, "Q": Q}.items() if v is None]
    if missing:
        raise ValueError(f"stereo calibration is missing {', '.join(missing)}")
    return calib, P1, P2, Q


def load_raw(path: str, size=(1600, 1200)) -> np.ndarray:
    w, h = size
    return np.fromfile(path, np.uint8).reshape(2 * h, w)


@dataclass
class Params:
    sigma: float = 2.0            # Gaussian scale for the ridge detector, px (stripe FWHM is 5-8 px)
    floor: float = 18.0           # smoothed intensity a stripe centre must reach (sensor black level is 16). At
                                  # Revo Metro's 200 us exposure stripes on dark parts peak only ~20-30; line
                                  # identity rejects noise, so the floor can sit just above black (Revo: threshold 10)
    prominence: float = 4.0       # ... and how far it must stand above the image 6 px either side, along n
    anisotropy: float = 0.5       # |lambda2| / |lambda1| must stay below this: a line, not a blob (markers)
    track_gap: float = 1.2        # px a stripe may move sideways between rows beyond its own slope
    min_track: int = 9            # rows; shorter fragments are noise or glints
    smooth_half: int = 6          # half window of the along-stripe fit, rows
    smooth_reject: float = 0.6    # px; a centre this far off its local fit is dropped
    trim_ends: int = 4            # rows dropped at each end of a track: where a stripe ends (shadow, occlusion, edge)
                                  # the Gaussian blends it with its surroundings and pulls the centre by up to ~1 mm
    line_tol_coarse: float = 6.0  # px in the right view, before the maps' offset is known (lines are ~180 px apart)
    line_tol: float = 2.5         # px, once it is removed
    line_min_support: float = 0.5  # fraction of a track the right view must confirm for its line
    line_margin: float = 0.35     # ... by this much more than any other line (regular stripe spacing makes a wrong
                                  # line meet *some* right stripe on part of a track; the true one meets it throughout)
    agree_mm: float = 1.5         # neighbourhood depth agreement
    agree_cell: int = 24          # px


class Triangulator:
    def __init__(self, calib: dict, P1: np.ndarray, P2: np.ndarray, Q: np.ndarray, params: Params | None = None,
                 extra: str | bytes | None = None):
        self.p = params or Params()
        self.size = calib["size"]
        R1, R2, *_ = cv2.stereoRectify(calib["KL"], calib["DL"], calib["KR"], calib["DR"], self.size,
                                       calib["R"], calib["T"], flags=0)   # flags=0 reproduces Revopoint's pL/pR
        self.m1 = cv2.initUndistortRectifyMap(calib["KL"], calib["DL"], R1, P1, self.size, cv2.CV_16SC2)
        self.m2 = cv2.initUndistortRectifyMap(calib["KR"], calib["DR"], R2, P2, self.size, cv2.CV_16SC2)
        self.P1, self.P2, self.Q = P1, P2, Q
        self.f = float(P1[0, 0])
        self.B = float(-P2[0, 3] / P2[0, 0])
        self.off = float(P1[0, 2] - P2[0, 2])
        self.line_sets: list = []
        if extra is not None:
            from . import laserfile
            sets = laserfile.parse(extra)
            # the two cross-line families (sections 0 and 1); frames alternate between them
            self.line_sets = [sets[i] for i in (0, 1) if i in sets]
        self.last: dict = {}

    @classmethod
    def from_yaml(cls, path_or_text: str | bytes, params: Params | None = None,
                  extra: str | bytes | None = None) -> "Triangulator":
        """From camparam.yaml, given as a path or as the file's contents (as read off the scanner)."""
        return cls(*read_calibration(path_or_text), params, extra)

    # -- per frame -------------------------------------------------------------------------------------------

    def rectify(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = self.size[1]
        rl = cv2.remap(frame[:h], *self.m1, cv2.INTER_LINEAR)
        rr = cv2.remap(frame[h:2 * h], *self.m2, cv2.INTER_LINEAR)
        return rl, rr

    def points(self, frame: np.ndarray, track: bool = True, with_pixels: bool = False, matching: str | None = None):
        """matching: 'lines' (laser line identity from the factory laser calibration; the default when it is
        loaded) or 'order' (stereo order only - fine on smooth surfaces, wrong across steps and occlusions)."""
        rl, rr = self.rectify(frame)
        return self.points_rectified(rl, rr, track, with_pixels, matching)

    def points_rectified(self, rl: np.ndarray, rr: np.ndarray, track: bool = True, with_pixels: bool = False,
                         matching: str | None = None, masks: tuple | None = None):
        """masks: optional (left, right) boolean images of pixels that are not laser stripes (markers)."""
        matching = matching or ("lines" if self.line_sets else "order")
        if matching == "lines" and not self.line_sets:
            raise ValueError("line matching needs the laser calibration (metroExtra.bin)")
        cl = stripe_centres(rl, self.p)
        cr = stripe_centres(rr, self.p)
        if masks is not None:
            cl, cr = _unmasked(cl, masks[0]), _unmasked(cr, masks[1])
        if track or matching == "lines":
            cl = smooth_tracks(cl, self.p)
            cr = smooth_tracks(cr, self.p)
        if matching == "lines":
            pts, xs, ys = self.triangulate_by_lines(cl, cr)
        else:
            self.last = {}
            pts, xs, ys = self.triangulate(cl, cr)
        keep = agreement(pts, xs, ys, self.p.agree_mm, self.p.agree_cell)
        if with_pixels:
            return pts[keep], xs[keep], ys[keep]
        return pts[keep]

    def triangulate(self, cl: "Centres", cr: "Centres"):
        xl_all, xr_all, y_all = [], [], []
        rows_l, rows_r = cl.by_row(), cr.by_row()
        for y, pa in rows_l.items():
            pb = rows_r.get(y)
            if pb is None:
                continue
            for a, b in match_row(pa, pb, self.f, self.B, self.off):
                xl_all.append(a)
                xr_all.append(b)
                y_all.append(y)
        if not xl_all:
            return np.zeros((0, 3), np.float32), np.zeros(0), np.zeros(0)
        xl, xr, y = np.array(xl_all), np.array(xr_all), np.array(y_all, np.float64)
        h = np.c_[xl, y, xl - xr, np.ones_like(xl)] @ self.Q.T
        pts = (h[:, :3] / h[:, 3:4]).astype(np.float32)
        return pts, xl, y

    def triangulate_by_lines(self, cl: "Centres", cr: "Centres"):
        """Pair left and right stripes by laser line identity, then triangulate by stereo.

        The factory laser calibration maps, for every laser line, a left pixel to the right-view x on the same row
        (see laserfile). Along a left track each line predicts where the partner must be; the right view confirms the
        true line over the whole track (~1 px) while every other line lands ~180 px off and only meets some stripe by
        chance, point here and there. The line that the right view confirms along most of the track wins. The pair of
        MEASURED centres is then triangulated through the stereo calibration - the maps choose, they do not measure.
        """
        p = self.p
        empty = (np.zeros((0, 3), np.float32), np.zeros(0), np.zeros(0))
        self.last = {"family": None, "tracks": 0, "assigned": 0, "paired": 0.0}
        if cl.track is None or not len(cl.x) or not len(cr.x):
            return empty
        order = np.argsort(cl.track, kind="stable")
        cuts = np.flatnonzero(np.diff(cl.track[order])) + 1
        tracks = [t for t in np.split(order, cuts) if len(t) >= p.min_track]
        self.last["tracks"] = len(tracks)
        if not tracks:
            return empty

        # right-view centres sorted by (row, x): one searchsorted answers "nearest right stripe on this row"
        ry = cr.y.astype(np.int64)
        rkey = ry * 4096.0 + cr.x
        rorder = np.argsort(rkey)
        rkey, rx, ry = rkey[rorder], cr.x[rorder], ry[rorder]

        def nearest(yq: np.ndarray, xq: np.ndarray):
            j = np.searchsorted(rkey, yq * 4096.0 + xq)
            j0, j1 = np.clip(j - 1, 0, len(rkey) - 1), np.clip(j, 0, len(rkey) - 1)
            d0 = np.where(ry[j0] == yq, rx[j0] - xq, np.inf)
            d1 = np.where(ry[j1] == yq, rx[j1] - xq, np.inf)
            pick1 = np.abs(d1) < np.abs(d0)
            return np.where(pick1, d1, d0), np.where(pick1, j1, j0)

        from .laserfile import monomials
        m = monomials(cl.x, cl.y)
        yq = cl.y.astype(np.int64)[:, None]
        best = None
        for fi, ls in enumerate(self.line_sets):          # which family is lit: the one the right view confirms
            delta, partner = nearest(yq, m @ ls.forward.T)
            ok = np.abs(delta) < p.line_tol_coarse
            score = sum(int(ok[t].sum(0).max()) for t in tracks)
            if best is None or score > best[0]:
                best = (score, fi, delta, partner)
        _, family, delta, partner = best
        ls = self.line_sets[family]

        # the maps carry a small offset against this rectification (1-2 px, same for all lines of a family):
        # measure it on tracks that are unambiguous even at the coarse tolerance, then tighten
        votes = []
        for t in tracks:
            support = (np.abs(delta[t]) < p.line_tol_coarse).sum(0)
            k = int(np.argmax(support))
            if support[k] >= 0.8 * len(t) and np.sort(support)[-2] < 0.1 * len(t):
                votes.append(np.median(delta[t, k]))
        bias = float(np.median(votes)) if votes else 0.0

        # a candidate counts only where the pairing it implies lands inside the calibrated depth range
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.f * self.B / (cl.x[:, None] - (rx[partner] if len(rx) else 0) - self.off)
        ok = (np.abs(delta - bias) < p.line_tol) & (z > Z_MIN) & (z < Z_MAX)
        line = np.full(len(cl.x), -1)
        for t in tracks:
            support = ok[t].sum(0) / len(t)
            order_k = np.argsort(support)[::-1]
            k, runner = int(order_k[0]), float(support[order_k[1]])
            if support[k] >= p.line_min_support and support[k] - runner >= p.line_margin:
                sel = t[ok[t, k]]
                line[sel] = k
        idx = np.flatnonzero(line >= 0)
        if not len(idx):
            self.last.update(family=family, bias_px=bias)
            return empty
        k = line[idx]
        d = delta[idx, k] - bias
        right = partner[idx, k]
        # the reverse map must agree too: it sends the right stripe back to the left one
        back = np.einsum("ij,ij->i", monomials(rx[right], cl.y[idx]), ls.reverse[k])
        rev = back - cl.x[idx]
        rev_bias = float(np.median(rev))
        keep = np.abs(rev - rev_bias) < p.line_tol
        idx, k, d, right = idx[keep], k[keep], d[keep], right[keep]
        # a right stripe centre pairs with one left centre only: keep the closest claim
        o = np.lexsort((np.abs(d), right))
        first = np.r_[True, np.diff(right[o]) != 0]
        idx, k, right = idx[o][first], k[o][first], right[o][first]

        xl, xr, y = cl.x[idx], rx[right], cl.y[idx].astype(np.float64)
        h = np.c_[xl, y, xl - xr, np.ones_like(xl)] @ self.Q.T
        pts = (h[:, :3] / h[:, 3:4]).astype(np.float32)
        inside = (pts[:, 2] > Z_MIN) & (pts[:, 2] < Z_MAX)
        pts, xl, y, idx, k = pts[inside], xl[inside], y[inside], idx[inside], k[inside]
        self.last = {"family": family, "bias_px": bias, "reverse_bias_px": rev_bias, "tracks": len(tracks),
                     "assigned": int(len(np.unique(cl.track[idx]))), "lines": sorted(set(k.tolist())),
                     "paired": float(len(idx) / len(cl.x))}
        return pts, xl, y


def _unmasked(c: "Centres", mask: np.ndarray) -> "Centres":
    keep = ~mask[c.y.astype(int), np.clip(np.rint(c.x).astype(int), 0, mask.shape[1] - 1)]
    return Centres(c.x[keep], c.y[keep], c.slope[keep], None if c.track is None else c.track[keep])


# -- stripe centres -------------------------------------------------------------------------------------------

@dataclass
class Centres:
    x: np.ndarray        # sub-pixel column
    y: np.ndarray        # row (integer valued)
    slope: np.ndarray    # dx/dy of the stripe at this point
    track: np.ndarray | None = None   # track id per centre, once linked

    def by_row(self) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        order = np.lexsort((self.x, self.y))
        ys, xs = self.y[order].astype(int), self.x[order]
        cuts = np.flatnonzero(np.diff(ys)) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(ys)]):
            if b > a:
                out[int(ys[a])] = xs[a:b].tolist()
        return out


def stripe_centres(img: np.ndarray, p: Params) -> Centres:
    """Steger ridge centres, one per stripe per row, as the zero crossing along the row of d/dn of the image."""
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), p.sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3, scale=1 / 4)
    gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3, scale=1 / 4)
    gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3, scale=1 / 4)

    half_tr = 0.5 * (gxx + gyy)
    disc = np.sqrt((0.5 * (gxx - gyy)) ** 2 + gxy ** 2)
    lam1 = half_tr - disc                  # most negative: curvature across a bright line
    lam2 = half_tr + disc
    # eigenvector of lam1, from whichever row of (H - lam1 I) is better conditioned
    ax, ay = gxy, lam1 - gxx
    bx, by = lam1 - gyy, gxy
    use_b = (bx * bx + by * by) > (ax * ax + ay * ay)
    nx = np.where(use_b, bx, ax)
    ny = np.where(use_b, by, ay)
    norm = np.sqrt(nx * nx + ny * ny) + 1e-12
    nx, ny = nx / norm, ny / norm
    flip = nx < 0                          # orient the normal to point right, so d/dn goes + to - across a ridge
    nx = np.where(flip, -nx, nx)
    ny = np.where(flip, -ny, ny)
    dn = nx * gx + ny * gy

    # a zero crossing from + to - between column x and x+1
    a, b = dn[:, :-1], dn[:, 1:]
    cross = (a > 0) & (b <= 0)
    strong = (lam1[:, :-1] < 0) & (np.abs(lam2[:, :-1]) < p.anisotropy * np.abs(lam1[:, :-1]))
    bright = g[:, :-1] >= p.floor
    ys, xs = np.nonzero(cross & strong & bright)
    if len(xs) == 0:
        return Centres(np.zeros(0), np.zeros(0), np.zeros(0))
    t = a[ys, xs] / (a[ys, xs] - b[ys, xs])
    xc = xs + t

    # prominence: the centre must stand above the image 6 px either side along the normal (rejects wide blobs)
    nxs, nys = nx[ys, xs], ny[ys, xs]
    h, w = g.shape
    def sample(dx, dy):
        u = np.clip(np.rint(xc + dx).astype(int), 0, w - 1)
        v = np.clip(np.rint(ys + dy).astype(int), 0, h - 1)
        return g[v, u]
    centre = g[ys, xs] * (1 - t) + g[ys, np.minimum(xs + 1, w - 1)] * t
    side = np.maximum(sample(6 * nxs, 6 * nys), sample(-6 * nxs, -6 * nys))
    ok = centre - side >= p.prominence
    # tangent (-ny, nx): dx/dy = -ny / nx
    slope = -nys / np.where(np.abs(nxs) < 1e-3, 1e-3, nxs)
    return Centres(xc[ok].astype(np.float64), ys[ok].astype(np.float64), slope[ok])


# -- along-stripe fitting --------------------------------------------------------------------------------------

def _savgol(half: int) -> np.ndarray:
    """Local quadratic least-squares smoothing weights over 2*half+1 samples (value at the centre)."""
    k = np.arange(-half, half + 1, dtype=np.float64)
    A = np.c_[np.ones_like(k), k, k * k]
    return np.linalg.pinv(A)[0]


def link_tracks(c: Centres, p: Params) -> list[np.ndarray]:
    """Chain centres row to row: each continues the nearest centre of the previous row that its slope predicts."""
    if not len(c.x):
        return []
    rows: dict[int, np.ndarray] = {}
    order = np.lexsort((c.x, c.y))
    ys = c.y[order].astype(int)
    cuts = np.flatnonzero(np.diff(ys)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(ys)]):
        rows[int(ys[a])] = order[a:b]
    track_of = np.full(len(c.x), -1)
    tracks: list[list[int]] = []
    prev_y, prev_idx = None, None
    for y in sorted(rows):
        idx = rows[y]
        if prev_y == y - 1 and prev_idx is not None and len(prev_idx):
            pred = c.x[prev_idx] + c.slope[prev_idx]
            # greedy, closest pairs first, one-to-one
            d = np.abs(c.x[idx][:, None] - pred[None, :])
            taken_prev = np.zeros(len(prev_idx), bool)
            taken_cur = np.zeros(len(idx), bool)
            for flat in np.argsort(d, axis=None):
                i, j = divmod(int(flat), len(prev_idx))
                if d[i, j] > p.track_gap:
                    break
                if taken_cur[i] or taken_prev[j]:
                    continue
                taken_cur[i] = taken_prev[j] = True
                t = track_of[prev_idx[j]]
                track_of[idx[i]] = t
                tracks[t].append(int(idx[i]))
        for i in idx:
            if track_of[i] < 0:
                track_of[i] = len(tracks)
                tracks.append([int(i)])
        prev_y, prev_idx = y, idx
    return [np.array(t) for t in tracks if len(t) >= p.min_track]


def smooth_tracks(c: Centres, p: Params) -> Centres:
    wts = _savgol(p.smooth_half)
    xs, ys, ss, ids = [], [], [], []
    for tid, t in enumerate(link_tracks(c, p)):
        x = c.x[t]
        n = len(x)
        if n < 2 * p.smooth_half + 1:
            # too short for the full window: a straight line fit is the most we can justify
            k = np.arange(n)
            fit = np.polyval(np.polyfit(k, x, 1), k)
        else:
            fit = np.convolve(x, wts[::-1], mode="same")
            # the ends: fit the first/last full window and evaluate it at the end samples
            k = np.arange(-p.smooth_half, p.smooth_half + 1)
            for sl, where in ((slice(0, 2 * p.smooth_half + 1), slice(0, p.smooth_half)),
                              (slice(n - 2 * p.smooth_half - 1, n), slice(n - p.smooth_half, n))):
                coef = np.polyfit(k, x[sl], 2)
                kk = np.arange(sl.start, sl.stop)[where.start - sl.start: where.stop - sl.start] - sl.start - p.smooth_half
                fit[where] = np.polyval(coef, kk)
        keep = np.abs(x - fit) <= p.smooth_reject
        k = np.arange(n)
        keep &= (k >= p.trim_ends) & (k < n - p.trim_ends)
        xs.append(fit[keep])
        ys.append(c.y[t][keep])
        ss.append(c.slope[t][keep])
        ids.append(np.full(int(keep.sum()), tid))
    if not xs:
        return Centres(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, int))
    return Centres(np.concatenate(xs), np.concatenate(ys), np.concatenate(ss), np.concatenate(ids))


# -- matching and filtering -------------------------------------------------------------------------------------

def match_row(pl: list[float], pr: list[float], f: float, B: float, off: float) -> list[tuple[float, float]]:
    """Order-preserving match (a stripe cannot overtake another), skipping a stripe on either side if needed."""
    if not pl or not pr:
        return []
    n, m = len(pl), len(pr)
    NEG = -1e9
    score = np.full((n + 1, m + 1), NEG)
    back = np.zeros((n + 1, m + 1), np.int8)
    score[0, 0] = 0.0
    a = np.asarray(pl)[:, None] - np.asarray(pr)[None, :] - off
    with np.errstate(divide="ignore"):
        z = np.where(a != 0, f * B / a, 0.0)
    valid = (z > Z_MIN) & (z < Z_MAX)
    for i in range(n + 1):
        for j in range(m + 1):
            s0 = score[i, j]
            if s0 == NEG:
                continue
            if i < n and j < m and valid[i, j] and s0 + 1.0 > score[i + 1, j + 1]:
                score[i + 1, j + 1], back[i + 1, j + 1] = s0 + 1.0, 1
            if i < n and s0 - 0.2 > score[i + 1, j]:
                score[i + 1, j], back[i + 1, j] = s0 - 0.2, 2
            if j < m and s0 - 0.2 > score[i, j + 1]:
                score[i, j + 1], back[i, j + 1] = s0 - 0.2, 3
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        b = back[i, j]
        if b == 1:
            pairs.append((pl[i - 1], pr[j - 1]))
            i, j = i - 1, j - 1
        elif b == 2:
            i -= 1
        elif b == 3:
            j -= 1
        else:
            break
    return pairs[::-1]


def agreement(pts: np.ndarray, xs: np.ndarray, ys: np.ndarray, agree_mm: float, cell: int) -> np.ndarray:
    """A mis-paired stripe lands at a depth nothing around it has; keep points near their cell's median depth."""
    keep = np.zeros(len(pts), bool)
    if not len(pts):
        return keep
    keys = (ys // cell).astype(np.int64) * 10_000 + (xs // cell).astype(np.int64)
    order = np.argsort(keys, kind="stable")
    ks, z = keys[order], pts[order, 2]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(ks)]):
        if b - a >= 4:
            idx = order[a:b]
            keep[idx] = np.abs(z[a:b] - np.median(z[a:b])) < agree_mm
    return keep


