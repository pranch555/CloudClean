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
    assert np.mean(same) > 0.999                              # inner rows keep their claims over end rows
    for c in (cl, old):                                       # the line decisions are the same with and without ends
        _with_lines(tri, line_set, lambda: tri.triangulate_by_lines(c, cr))
        info = {k: tri.last[k] for k in ("tracks", "assigned", "lines")}
        if c is cl:
            first = info
    assert info == first


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
def _stripe(z_of_row, x0=100.0, rows=range(0, 24), slope=0.5, track=0):
    """One stripe's points: rows, x along a 0.5 px/row diagonal, depth by row, and its track id."""
    ys = np.array(list(rows), float)
    xs = x0 + slope * ys
    z = np.array([z_of_row(y) for y in ys])
    return np.c_[np.zeros_like(z), np.zeros_like(z), z].astype(np.float32), xs, ys, np.full(len(ys), track)


def _cat(*stripes):
    return [np.concatenate(parts) for parts in zip(*stripes)]


def test_agreement_keeps_a_steep_stripe_whole():
    # one stripe crossing a cell on a surface seen at a grazing angle: depth changes 0.3 mm per row (7 mm over the
    # cell). The median test kept only its middle third
    pts, xs, ys, tr = _stripe(lambda y: 250 + 0.3 * y)
    assert S.agreement(pts, xs, ys, 1.5, 24, tr).all()
    assert S.agreement(pts, xs, ys, 1.5, 24).mean() < 0.5      # (without track ids: the median test, as before)


@pytest.mark.parametrize("slope, offset", [(0.0, 11.0), (0.2, 12.0), (0.3, 11.0), (0.5, 14.0), (1.0, 20.0),
                                           (1.0, 44.0)])
def test_agreement_drops_a_parallel_mispaired_stripe(slope, offset):
    # two parallel stripes 14 px apart in one cell, one paired a line off: a constant 11-44 mm deeper. Two parallel
    # stripes always lie on one tilted plane - a plane test would take the offset for a tilt across them (it did:
    # 100 % accepted at 0.3 mm/row). Depth followed along each stripe and compared at one row cannot be fooled
    a = _stripe(lambda y: 250 + slope * y, x0=96, rows=range(0, 24), track=1)
    b = _stripe(lambda y: 250 + offset + slope * y, x0=110, rows=range(2, 18), track=2)
    pts, xs, ys, tr = _cat(a, b)
    keep = S.agreement(pts, xs, ys, 1.5, 24, tr)
    assert not keep[len(a[0]):].any()                        # the mispaired stripe never survives
    assert keep[:len(a[0])].all()                            # the true one, the majority, does


@pytest.mark.parametrize("slope, offset", [(0.3, 11.0), (0.5, 14.0), (1.0, 20.0), (0.7, 30.0)])
def test_agreement_never_explains_a_mispair_as_a_tilt(slope, offset):
    # the review's case (2026-10-09): two parallel 20-row stripes 14 px apart, the second a line off. A plane through
    # the cell accepted all of the mispaired points; now none survive
    a = _stripe(lambda y: 250 + slope * y, x0=96, rows=range(0, 20), track=1)
    b = _stripe(lambda y: 250 + offset + slope * y, x0=110, rows=range(0, 20), track=2)
    pts, xs, ys, tr = _cat(a, b)
    assert not S.agreement(pts, xs, ys, 1.5, 24, tr)[20:].any()


def test_agreement_equal_stripes_in_conflict_both_go():
    a = _stripe(lambda y: 250 + 0.3 * y, x0=96, rows=range(0, 20), track=1)
    b = _stripe(lambda y: 261 + 0.3 * y, x0=110, rows=range(0, 20), track=2)
    pts, xs, ys, tr = _cat(a, b)
    assert not S.agreement(pts, xs, ys, 1.5, 24, tr).any()   # no way to tell which is right: neither is kept


def test_agreement_keeps_two_true_stripes_on_a_steep_surface():
    # neighbouring stripes on the same steep surface: along each, 0.4 mm per row; between them 0.8 mm
    a = _stripe(lambda y: 250 + 0.4 * y, x0=96, rows=range(0, 24), track=1)
    b = _stripe(lambda y: 250.8 + 0.4 * y, x0=110, rows=range(0, 24), track=2)
    pts, xs, ys, tr = _cat(a, b)
    assert S.agreement(pts, xs, ys, 1.5, 24, tr).all()


def test_agreement_judges_a_sparse_cell_by_its_neighbourhood():
    # a stripe's last two rows spill into the next cell: alone they cannot be judged, with the cells around they can
    pts, xs, ys, tr = _stripe(lambda y: 250 + 0.05 * y, x0=60, rows=range(0, 26))    # rows 24, 25: next cell row
    assert S.agreement(pts, xs, ys, 1.5, 24, tr).all()
    assert not S.agreement(pts, xs, ys, 1.5, 24)[-2:].any()   # (the median test dropped them)
    wrong = pts.copy()
    wrong[-2:, 2] += 15.0                                    # a second track, mis-paired, only those two rows
    tr2 = tr.copy()
    tr2[-2:] = 7
    keep = S.agreement(wrong, xs, ys, 1.5, 24, tr2)
    assert keep[:-2].all() and not keep[-2:].any()


def test_agreement_cost_stays_small():
    import time
    rng = np.random.default_rng(0)
    # a frame's worth: 17 stripes, 1000 rows each, on a tilted surface, broken into pieces
    ys = np.tile(np.arange(100, 1100, dtype=float), 17)
    xs = np.repeat(np.arange(17) * 90.0 + 50, 1000) + 0.5 * (ys - 100)
    tr = np.repeat(np.arange(17), 1000) * 1000 + (ys // 37).astype(int)
    keep_rows = (rng.random(len(ys)) < 0.97) & (np.sin(ys / rng.uniform(5, 60)) > -0.8)
    xs, ys, tr = xs[keep_rows], ys[keep_rows], tr[keep_rows]
    pts = np.c_[np.zeros_like(xs), np.zeros_like(xs), 250 + 0.02 * xs + 0.05 * ys + rng.normal(0, 0.05, len(xs))]
    S.agreement(pts, xs, ys, 1.5, 24, tr)
    t0 = time.perf_counter()
    for _ in range(5):
        keep = S.agreement(pts, xs, ys, 1.5, 24, tr)
    assert (time.perf_counter() - t0) / 5 < 0.1              # ~5 ms here; a frame's whole budget is ~150 ms
    assert keep.mean() > 0.99


# ----------------------------------------------------------------------------------------------- tracing
@pytest.mark.parametrize("with_masks", [False, True])
def test_trace_accounts_for_every_left_centre(tri, scene, with_masks):
    left, right, line_set = scene
    masks = None
    if with_masks:                                           # two "markers" on stripes, one per view
        masks = []
        for img in (left, right):
            m = np.zeros(img.shape, np.uint8)
            cv2.circle(m, (800, 600), 15, 1, -1)
            cv2.circle(m, (500, 400), 10, 1, -1)
            masks.append(m.astype(bool))
    run = lambda: tri.points_rectified(left, right, with_pixels=True, masks=masks)  # noqa: E731
    pts, xs, ys = _with_lines(tri, line_set, run)
    tri.trace = True
    try:
        pts2, xs2, ys2 = _with_lines(tri, line_set, run)
        tr = dict(tri.traced)
    finally:
        tri.trace = False
    assert not tri.traced or tri.trace is False
    assert np.array_equal(xs, xs2) and np.array_equal(ys, ys2)            # tracing changes nothing
    # left track ends are kept (and flagged); the right view never offers an end row as a partner
    assert tr["left"].end is not None and tr["left"].end.any()
    assert tr["right"].end is None
    why, final = tr["why"], tr["final"]
    assert len(why) == len(tr["left"].x)
    assert (why == S.ACCEPTED).sum() == len(pts)
    assert np.array_equal(tr["left"].x[final], xs) and np.array_equal(tr["left"].y[final], ys)
    src = tr["left"].src[final]                                            # back to the detector's centres
    if with_masks:
        src = tr["left_unmasked"].src[src]
        raw = tr["left_raw"]
        assert not masks[0][raw.y[src].astype(int), np.rint(raw.x[src]).astype(int)].any()
    assert np.array_equal(tr["left_raw"].y[src], ys)
    # outside tracing nothing is kept
    _with_lines(tri, line_set, run)
    assert tri.traced == {}


def test_order_matching_keeps_the_median_test():
    # stereo-order matching (no laser calibration) has no track ids for its points: the agreement there is the
    # cell median, unchanged
    pts, xs, ys, _ = _stripe(lambda y: 250 + 0.3 * y)
    old = np.zeros(len(pts), bool)
    old[np.abs(pts[:, 2] - np.median(pts[:, 2])) < 1.5] = True
    assert np.array_equal(S.agreement(pts, xs, ys, 1.5, 24), old)
