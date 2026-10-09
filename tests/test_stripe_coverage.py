"""Stripe coverage: track ends, marker-cut tracks and the depth agreement filter (cloudclean.capture.metroy.stripes).

On the 2026-10-09 turntable recording a quarter of the laser stripe centres on the part never became points; three
causes were fixed (see stripes.Params.trim_ends and stripes.agreement) - each without admitting a wrong pairing. The
scene tests render a box on a floor lit by 17 laser sheets through the scanner's REAL factory rectification
(tests/data/metroy), like tests/test_metroy.py, and run the whole points_rectified path.
"""
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cloudclean.capture.metroy import laserfile  # noqa: E402
from cloudclean.capture.metroy import stripes as S  # noqa: E402

DATA = Path(__file__).parent / "data" / "metroy"


@pytest.fixture(scope="module")
def tri():
    return S.Triangulator.from_yaml(str(DATA / "camparam.yaml"), extra=str(DATA / "metroExtra.bin"))


# ----------------------------------------------------------------------------------------------- synthetic scene
# a box on a floor, 17 laser sheets from a projector beside the cameras, with laser shadows and occlusions (the same
# scene as tests/test_metroy.py, kept here so the two files can change independently)
FLOOR, TOP = 380.0, 330.0
BOX_X, BOX_Y = (-25.0, 65.0), (-40.0, 20.0)
PROJECTOR = np.array([80.0, -10.0, -20.0])


def _cast(o, d):
    o = np.broadcast_to(o, d.shape)
    best = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (FLOOR - o[:, 2]) / d[:, 2]
        best = np.where(t > 1e-9, np.minimum(best, t), best)
        t = (TOP - o[:, 2]) / d[:, 2]
        p = o + t[:, None] * d
        inside = (p[:, 0] > BOX_X[0]) & (p[:, 0] < BOX_X[1]) & (p[:, 1] > BOX_Y[0]) & (p[:, 1] < BOX_Y[1])
        best = np.where(inside & (t > 1e-9), np.minimum(best, t), best)
        for axis, walls, other, span in ((0, BOX_X, 1, BOX_Y), (1, BOX_Y, 0, BOX_X)):
            for w in walls:
                t = (w - o[:, axis]) / d[:, axis]
                p = o + t[:, None] * d
                hit = (p[:, other] > span[0]) & (p[:, other] < span[1]) & (p[:, 2] > TOP) & (p[:, 2] < FLOOR)
                best = np.where(hit & (t > 1e-9), np.minimum(best, t), best)
    return best


def _laser_planes(n=17):
    axis = np.array([0.47, -0.87, 0.2])
    axis /= np.linalg.norm(axis)
    base = np.cross(axis, [0.0, 0.0, 1.0])
    base /= np.linalg.norm(base)
    other = np.cross(axis, base)
    normals = [np.cos(a) * base + np.sin(a) * other for a in np.linspace(-0.26, 0.26, n)]
    return np.array(normals), np.array([nn @ PROJECTOR for nn in normals])


def _line_set(tri, normals, offsets):
    f, cx, cy, cx2 = tri.P1[0, 0], tri.P1[0, 2], tri.P1[1, 2], tri.P2[0, 2]
    rng = np.random.default_rng(1)
    recs = np.zeros((2 * len(normals), 20))
    for k, (nn, d) in enumerate(zip(normals, offsets)):
        u, v = rng.uniform(0, 1600, 4000), rng.uniform(0, 1200, 4000)
        ray = np.c_[(u - cx) / f, (v - cy) / f, np.ones_like(u)]
        z = d / (ray @ nn)
        ok = (z > 200) & (z < 450)
        P = ray[ok] * z[ok, None]
        xl, y = u[ok], v[ok]
        xr = f * (P[:, 0] - tri.B) / P[:, 2] + cx2
        fwd = np.linalg.lstsq(laserfile.monomials(xl, y), xr, rcond=None)[0]
        rev = np.linalg.lstsq(laserfile.monomials(xr, y), xl, rcond=None)[0]
        recs[k, :6] = recs[k, 6:12] = fwd
        recs[k + len(normals), :6] = recs[k + len(normals), 6:12] = rev
    return laserfile.LineSet(1, len(normals), recs)


def _render(tri, normals, offsets, origin, cx, seed=0):
    f, cy = tri.P1[0, 0], tri.P1[1, 2]
    v, u = np.mgrid[0:1200, 0:1600].astype(np.float64)
    d = np.c_[((u - cx) / f).ravel(), ((v - cy) / f).ravel(), np.ones(u.size)]
    t = _cast(origin, d)
    X = origin + t[:, None] * d
    lit = np.isfinite(t)
    lit &= _cast(PROJECTOR, X - PROJECTOR) > 1 - 1e-6                 # laser shadow
    stripe = 210 * np.exp(-0.5 * ((X @ normals.T - offsets) / 0.5) ** 2).max(1)
    img = 25 + np.where(lit, stripe, 0) + np.random.default_rng(seed).normal(0, 2, len(t))
    return np.clip(img, 0, 255).astype(np.uint8).reshape(1200, 1600)


def _depth_error(points):
    t = _cast(np.zeros(3), points / points[:, 2:3])
    return points[:, 2] - t


@pytest.fixture(scope="module")
def scene(tri):
    normals, offsets = _laser_planes()
    left = _render(tri, normals, offsets, np.zeros(3), tri.P1[0, 2])
    right = _render(tri, normals, offsets, np.array([tri.B, 0.0, 0.0]), tri.P2[0, 2], seed=1)
    return left, right, _line_set(tri, normals, offsets)


def _with_lines(tri, line_set, fn):
    saved = tri.line_sets
    tri.line_sets = [line_set]
    try:
        return fn()
    finally:
        tri.line_sets = saved


# ----------------------------------------------------------------------------------------------- track ends
def test_left_track_ends_are_paired_but_do_not_choose_the_line(tri, scene):
    left, right, line_set = scene
    pts, xs, ys = _with_lines(tri, line_set, lambda: tri.points_rectified(left, right, with_pixels=True))
    assert len(pts) > 5000
    err = np.abs(_depth_error(pts.astype(np.float64)))
    # no stripe paired with the wrong partner (a slip is 10-40 mm); the box's convex edges round off by < 3 mm
    assert err.max() < 3.0, f"worst point {err.max():.2f} mm off the surface"
    assert np.percentile(err, 99.5) < 0.1

    # the same frame with every end row dropped, as before: the same tracks are paired, with the same partners, and
    # the end rows only add points to them
    cl = S.smooth_tracks(S.stripe_centres(left, tri.p), tri.p, keep_ends=True)
    cr = S.smooth_tracks(S.stripe_centres(right, tri.p), tri.p)
    inner = ~cl.end
    old = S.Centres(cl.x[inner], cl.y[inner], cl.slope[inner], cl.track[inner])
    new_pts, new_x, new_y = _with_lines(tri, line_set, lambda: tri.triangulate_by_lines(cl, cr))
    old_pts, old_x, old_y = _with_lines(tri, line_set, lambda: tri.triangulate_by_lines(old, cr))
    # here both cameras' stripes end on the same rows, and right-view ends are never partners, so few end rows pair
    # (on the real bust, where the views' stripes end in different places, they were 10 % of the centres)
    assert len(new_pts) > len(old_pts), (len(new_pts), len(old_pts))
    new = {(round(x, 6), int(y)): z for x, y, z in zip(new_x, new_y, new_pts[:, 2])}
    same = [abs(new.get((round(x, 6), int(y)), np.inf) - z) < 1e-4 for x, y, z in zip(old_x, old_y, old_pts[:, 2])]
    assert np.mean(same) > 0.995                              # (a few right centres go to a closer end-row claim)


def test_a_track_cut_by_a_marker_mask_keeps_its_rows_there(tri):
    h, w = 400, 400
    cols = np.arange(w)
    img = np.empty((h, w), np.uint8)
    for y in range(h):                                        # a straight stripe, 0.5 px per row
        img[y] = np.clip(25 + 200 * np.exp(-0.5 * ((cols - (100 + 0.5 * y)) / 1.6) ** 2), 0, 255).astype(np.uint8)
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (200, 200), 12, 1, -1)                      # a "marker" on it
    mask = m.astype(bool)
    full = S.stripe_centres(img, tri.p)
    c = S._unmasked(full, mask)
    gone = np.setdiff1d(full.y, c.y)
    assert len(gone) > 10
    beside = np.r_[np.arange(gone.min() - tri.p.trim_ends, gone.min()), np.arange(gone.max() + 1,
                                                                                    gone.max() + 1 + tri.p.trim_ends)]
    with_mask = S.smooth_tracks(c, tri.p, mask)
    without = S.smooth_tracks(c, tri.p)
    assert not np.isin(beside, without.y).any()               # as before: each piece lost 4 rows at the marker
    assert np.isin(beside, with_mask.y).all()                 # now the rows next to the mask stay
    # the free ends (top and bottom of the image) are still trimmed
    assert with_mask.y.min() == without.y.min() and with_mask.y.max() == without.y.max()


# ----------------------------------------------------------------------------------------------- agreement
def _cell_points(z_of_row, x0=100.0, rows=range(0, 24), slope=0.5):
    ys = np.array(list(rows), float)
    xs = x0 + slope * ys
    z = np.array([z_of_row(y) for y in ys])
    return np.c_[np.zeros_like(z), np.zeros_like(z), z].astype(np.float32), xs, ys


def test_agreement_keeps_a_steep_stripe_whole():
    # one stripe crossing a cell on a surface seen at a grazing angle: depth changes 0.3 mm per row (7 mm over the
    # cell). The median test kept only the middle third of it
    pts, xs, ys = _cell_points(lambda y: 250 + 0.3 * y)
    keep = S.agreement(pts, xs, ys, 1.5, 24)
    assert keep.all()


def test_agreement_still_drops_a_mispaired_stripe():
    # two stripes in one cell, one paired one line off: 12 mm deeper. It must go, the true one must stay
    a, xa, ya = _cell_points(lambda y: 250 + 0.2 * y, x0=96, rows=range(0, 22))
    b, xb, yb = _cell_points(lambda y: 262 + 0.2 * y, x0=110, rows=range(4, 14))
    pts, xs, ys = np.r_[a, b], np.r_[xa, xb], np.r_[ya, yb]
    keep = S.agreement(pts, xs, ys, 1.5, 24)
    assert keep[:len(a)].all() and not keep[len(a):].any()


def test_agreement_judges_a_sparse_cell_by_its_neighbourhood():
    # a stripe's last two rows spill into the next cell: alone they cannot be judged, with the cells around they can
    a, xa, ya = _cell_points(lambda y: 250 + 0.05 * y, x0=60, rows=range(0, 26))     # rows 24, 25: next cell row
    keep = S.agreement(a, xa, ya, 1.5, 24)
    assert keep.all()
    wrong = a.copy()
    wrong[-2:, 2] += 15.0                                    # the same two rows, mis-paired
    keep = S.agreement(wrong, xa, ya, 1.5, 24)
    assert keep[:-2].all() and not keep[-2:].any()


def test_agreement_cost_stays_small():
    import time
    rng = np.random.default_rng(0)
    # a frame's worth: 17 stripes, 1000 rows each, on a tilted surface, broken into pieces of 5-60 rows
    ys = np.tile(np.arange(100, 1100, dtype=float), 17)
    xs = np.repeat(np.arange(17) * 90.0 + 50, 1000) + 0.5 * (ys - 100)
    keep_rows = (rng.random(len(ys)) < 0.97) & (np.sin(ys / rng.uniform(5, 60)) > -0.8)
    xs, ys = xs[keep_rows], ys[keep_rows]
    pts = np.c_[np.zeros_like(xs), np.zeros_like(xs), 250 + 0.02 * xs + 0.05 * ys + rng.normal(0, 0.05, len(xs))]
    S.agreement(pts, xs, ys, 1.5, 24)
    t0 = time.perf_counter()
    for _ in range(5):
        keep = S.agreement(pts, xs, ys, 1.5, 24)
    assert (time.perf_counter() - t0) / 5 < 0.05             # a few ms; a frame's whole budget is ~150 ms
    assert keep.mean() > 0.99


# ----------------------------------------------------------------------------------------------- tracing
def test_trace_accounts_for_every_left_centre(tri, scene):
    left, right, line_set = scene
    pts, xs, ys = _with_lines(tri, line_set, lambda: tri.points_rectified(left, right, with_pixels=True))
    tri.trace = True
    try:
        pts2, xs2, ys2 = _with_lines(tri, line_set, lambda: tri.points_rectified(left, right, with_pixels=True))
        tr = dict(tri.traced)
    finally:
        tri.trace = False
    assert np.array_equal(xs, xs2) and np.array_equal(ys, ys2)            # tracing changes nothing
    # left track ends are kept (and flagged); the right view never offers an end row as a partner
    assert tr["left"].end is not None and tr["left"].end.any()
    assert tr["right"].end is None
    why, final = tr["why"], tr["final"]
    assert len(why) == len(tr["left"].x)
    assert (why == S.ACCEPTED).sum() == len(pts)
    assert np.array_equal(tr["left"].x[final], xs) and np.array_equal(tr["left"].y[final], ys)
    raw = tr["left_raw"]
    assert np.array_equal(raw.y[tr["left"].src[final]], ys)               # back to the detector's centres
