"""Marker centres (metroy/markers.py detect()): an ellipse fitted to the sub-pixel rim, leaving out what a laser stripe
touches, against the former intensity-weighted centroid.

The laser lines cross the markers and move between the two line families, so a centre a stripe drags lands in two
places: on the user's static plate (2026-10-09) the centroid jumped 2.5 px between alternate frames, the rim fit holds
to ~0.03 px. Markers here are rendered exactly (8x8 supersampled ellipse, optical blur, sensor noise, 8-bit), with the
true centre known to the last digit.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cloudclean.capture.metroy import markers  # noqa: E402

RIM = markers.MarkerParams()
CENTROID = markers.MarkerParams(centre="centroid")


def _render(centre=(100.37, 80.71), axes=(26.0, 13.0), angle=20.0, plateau=70.0, bg=17.0, stripe=None,
            noise=1.0, seed=0, shape=(160, 200)):
    """A marker (bright filled ellipse, semi-axes `axes`, rotated `angle` deg) over a dark background, as the IR
    cameras see it. stripe = (x, y, direction deg, peak, sigma px): a laser line through (x, y), `peak` on the marker
    and a faint 25 over the background (as on the user's dark plate)."""
    H, W = shape
    ss = 8
    yy, xx = (np.mgrid[0:H * ss, 0:W * ss] + 0.5) / ss - 0.5
    t = np.radians(angle)
    u = (xx - centre[0]) * np.cos(t) + (yy - centre[1]) * np.sin(t)
    v = -(xx - centre[0]) * np.sin(t) + (yy - centre[1]) * np.cos(t)
    cover = ((u / axes[0]) ** 2 + (v / axes[1]) ** 2 <= 1).reshape(H, ss, W, ss).mean((1, 3))
    img = cv2.GaussianBlur(bg + (plateau - bg) * cover, (0, 0), 0.8)
    if stripe is not None:
        sx, sy, d, peak, sigma = stripe
        y, x = np.mgrid[0:H, 0:W]
        dist = (x - sx) * np.cos(np.radians(d)) - (y - sy) * np.sin(np.radians(d))   # across the line
        gain = cv2.GaussianBlur(25.0 + (peak - 25.0) * cover, (0, 0), 0.8)
        img = img + gain * np.exp(-0.5 * (dist / sigma) ** 2)
    img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _one(img, p):
    blobs = markers.detect(img, p)
    assert len(blobs) == 1, [(b.x, b.y, b.radius) for b in blobs]
    return blobs[0]


def _err(b, centre=(100.37, 80.71)):
    return float(np.hypot(b.x - centre[0], b.y - centre[1]))


def test_clean_marker_centre_is_exact():
    b = _one(_render(), RIM)
    assert np.isfinite(b.rim_rms) and b.rim_rms < 0.1
    assert _err(b) < 0.04                   # no bias: over 16 draws the mean error is 0.002 px, the spread 0.01


def test_saturated_stripe_through_the_marker_does_not_drag_the_centre():
    """A laser line across the marker, 6 px off its centre, saturating (as on the user's plate): the centroid follows
    the stripe's light, the rim never sees it."""
    img = _render(stripe=(106.0, 80.0, 30.0, 400.0, 1.8))
    assert (img == 255).sum() > 30                                   # the stripe saturates
    rim, cen = _one(img, RIM), _one(img, CENTROID)
    assert _err(rim) < 0.05, _err(rim)
    assert _err(cen) > 0.3, _err(cen)


def test_stripe_across_the_rim_is_left_out():
    """The line crosses the rim near one end of the marker and runs on over the background, dimmer: the rim pieces
    it touches are left out and the rest of the rim still fixes the centre."""
    img = _render(stripe=(122.0, 88.0, 25.0, 300.0, 2.0))
    rim, cen = _one(img, RIM), _one(img, CENTROID)
    assert np.isfinite(rim.rim_rms)
    # less rim, less precision: over 16 draws at random sub-pixel centres up to 0.09 px (the centroid: up to 6 px)
    assert _err(rim) < 0.1, _err(rim)
    assert _err(cen) > 0.5, _err(cen)


def test_saturated_marker():
    """A near marker can saturate (Creaform sets its ring light to): the rim is still where the light falls off."""
    img = _render(plateau=255.0, stripe=(95.0, 80.0, 30.0, 400.0, 1.8))
    assert _err(_one(img, RIM)) < 0.05


def test_dim_thin_far_marker():
    """A far plate marker seen at a low angle: dim (17 -> 46) and thin (axis ratio 0.4). Its thresholded outline sits
    well inside the rim; the second pass re-centres the profiles on the rim (the first alone left out 2/3 of them on
    the user's plate and fell back to the centroid)."""
    c = (100.37, 80.71)
    img = _render(centre=c, axes=(18.0, 7.2), angle=77.0, plateau=46.0)
    b = _one(img, RIM)
    assert np.isfinite(b.rim_rms)
    assert _err(b, c) < 0.06                # 16 draws: up to 0.045 px


def test_centre_noise():
    """Sensor noise alone: spread of the centre over noise draws (the static recordings: 0.01-0.02 px per marker)."""
    pts = np.array([(b.x, b.y) for b in (_one(_render(seed=s, noise=1.5), RIM) for s in range(12))])
    assert pts.std(0).max() < 0.02
    assert np.hypot(*(pts.mean(0) - (100.37, 80.71))) < 0.02


def test_size_and_shape_are_those_of_the_thresholded_blob():
    """min_diameter_mm works on Blob.radius: the rim fit must not change it, only the centre."""
    img = _render(stripe=(106.0, 80.0, 30.0, 400.0, 1.8))
    rim, cen = _one(img, RIM), _one(img, CENTROID)
    assert (rim.radius, rim.axis_ratio) == (cen.radius, cen.axis_ratio)
    # the radius is the area of the thresholded, opened component
    thr = max(RIM.threshold_min, float(np.median(img[::8, ::8])) + RIM.threshold_above)
    core = cv2.morphologyEx((img >= thr).astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RIM.stripe_width_px,) * 2))
    n, _, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    assert n == 2 and rim.radius == pytest.approx(np.sqrt(st[1, cv2.CC_STAT_AREA] / np.pi))


def test_marker_cut_by_the_image_border():
    """A marker running off the image: what is left of the rim fixes the centre while there is enough of it; past
    that the centre is the centroid of the part in view, pixels off, and the blob is flagged partial - still masked
    out of the laser line search, never paired."""
    for cx, cut in ((182.0, False), (192.0, True)):                  # 1 px and 11 px of the 26 px half-axis cut off
        b = _one(_render(centre=(cx, 80.0), angle=0.0), RIM)
        assert b.partial == cut
        if not cut:
            assert np.isfinite(b.rim_rms) and _err(b, (cx, 80.0)) < 0.1
    d = 400.0
    left = [markers.Blob(1300.0, 500.0, 15.0, 0.8, partial=True)]
    right = [markers.Blob(1300.0 - d, 500.0, 15.0, 0.8)]
    assert len(markers.stereo_pairs(left, right, Q)[0]) == 0
    assert markers.plausible(left, right, Q)[0].tolist() == [True]


def test_several_markers_in_one_image_are_fitted_independently():
    """All blobs of an image are fitted together in one flat array: each centre must be what it is alone."""
    shape = (300, 400)
    spec = [((90.3, 80.6), (26.0, 13.0), 20.0, 70.0, (96.0, 80.0, 30.0, 400.0, 1.8)),
            ((300.7, 70.2), (18.0, 7.2), 77.0, 46.0, None),
            ((110.1, 220.9), (34.0, 20.0), -35.0, 255.0, (100.0, 230.0, 30.0, 400.0, 1.8)),
            ((290.4, 210.5), (14.0, 9.0), 60.0, 120.0, (292.0, 216.0, -40.0, 300.0, 1.8))]
    alone = [_render(centre=c, axes=a, angle=t, plateau=pl, stripe=st, shape=shape, seed=i)
             for i, (c, a, t, pl, st) in enumerate(spec)]
    together = alone[1].copy()                                        # each marker's own surroundings, pasted
    for (c, *_), img in zip(spec, alone):
        x, y = int(c[0]), int(c[1])
        together[y - 45:y + 45, x - 45:x + 45] = img[y - 45:y + 45, x - 45:x + 45]
    got = sorted(markers.detect(together, RIM), key=lambda b: (b.y > 150, b.x))
    assert len(got) == len(spec)
    for b, (c, *_), img in zip(got, spec, alone):
        single = _one(img, RIM)
        assert np.isfinite(b.rim_rms) and _err(b, c) < 0.15           # the small one keeps half its rim: noisier
        assert np.hypot(b.x - single.x, b.y - single.y) < 1e-6        # nothing leaks between the blobs


def test_a_blob_inside_another_ones_hole_is_its_own_blob():
    """A ring with a disc in its hole: two components, as labelling the image gives (the ring fails the fill test)."""
    img = np.full((200, 200), 17, np.uint8)
    cv2.circle(img, (100, 100), 70, 120, -1)
    cv2.circle(img, (100, 100), 45, 17, -1)
    cv2.circle(img, (100, 100), 18, 120, -1)
    blobs = markers.detect(img, CENTROID)
    assert len(blobs) == 1 and abs(blobs[0].x - 100) < 0.5 and abs(blobs[0].y - 100) < 0.5
    assert blobs[0].radius == pytest.approx(18.0, abs=0.6)                # the disc alone, not the ring's pixels


def test_stripe_along_a_small_marker():
    """A wide saturated stripe running along a small marker lights most of it. Its rim pieces next to the stripe looked
    clean against a marker level taken as the median over the rim (the stripe raised it) and pulled the fit 0.3-0.5 px;
    against a low quantile they are left out, too little rim is left, and the centre is the stripe-clipped centroid."""
    c = (290.4, 210.5)
    img = _render(centre=c, axes=(14.0, 9.0), angle=60.0, plateau=120.0, stripe=(292.0, 216.0, 25.0, 300.0, 2.0),
                  shape=(300, 400))
    b = _one(img, RIM)
    assert (np.isfinite(b.rim_rms) and _err(b, c) < 0.1) or _err(b, c) < 0.3, (b, _err(b, c))


def test_unknown_centre_method_is_refused():
    with pytest.raises(ValueError):
        markers.detect(_render(), markers.MarkerParams(centre="ellipse"))


def test_rim_sampling_at_the_image_edge():
    """Profiles reaching the last row or column: float32 rounds 1598.99999 up to 1599, whose right neighbour is off
    the image (an IndexError on the user's turning recording); off-image samples are NaN, never an error."""
    img = np.random.default_rng(1).integers(0, 255, (1200, 1600)).astype(np.uint8)
    x = np.array([0.0, 1598.9999999, 1599.0, 1599.0000001, -1e-9, 800.25, 1598.5])
    y = np.array([0.0, 1198.9999999, 1199.0, 600.0, 600.0, 1199.0, 1198.5])
    v = markers._bilinear(img, x, y)
    assert np.isnan(v[[3, 4]]).all() and np.isfinite(v[[0, 1, 2, 5, 6]]).all()
    assert v[2] == img[1199, 1599] and v[0] == img[0, 0]
    f = img.astype(np.float64)
    assert v[6] == pytest.approx(f[1198:1200, 1598:1600].mean(), abs=1e-3)
    assert v[5] == pytest.approx(0.75 * f[1199, 800] + 0.25 * f[1199, 801], abs=1e-3)


# ----------------------------------------------------------------------------------------------- stereo and masks
F, BASE = 1813.6, 128.5
Q = np.array([[1, 0, 0, -800.0], [0, 1, 0, -600.0], [0, 0, 0, F], [0, 0, 1 / BASE, 0]])     # right x = left x - d


def _stereo_blob(x, y, diameter_mm, z):
    return markers.Blob(x, y, 0.5 * diameter_mm * F / z, 0.8)


def test_a_laser_spot_on_the_marker_row_no_longer_vetoes_the_marker():
    """A bright laser spot (1.4 mm) on the same rectified row as a marker, at a disparity that also means a depth in
    range: it used to make the marker's pair ambiguous (two candidates), and the size test came only after."""
    z, d = 300.0, F * BASE / 300.0
    marker_l = _stereo_blob(1300.0, 500.0, 3.5, z)                  # a far plate marker at an angle: 3.3-5.5 mm
    marker_r = markers.Blob(1300.0 - d, 500.0, marker_l.radius, 0.8)
    spot_r = markers.Blob(1300.0 - d - 40.0, 500.3, marker_l.radius / 1.55, 0.8)   # 2.4 mm at its depth, on the row
    assert 2.0 * spot_r.radius * F * BASE / (d + 40.0) / F < 2.5
    for right in ([marker_r, spot_r], [spot_r, marker_r]):
        P, il, ir = markers.stereo_pairs([marker_l], right, Q)
        assert list(il) == [0] and right[ir[0]] is marker_r
        assert P[0, 2] == pytest.approx(z, rel=1e-6)
    # and the other way round: a spot in the left view next to the marker's left blob
    spot_l = markers.Blob(1300.0 + 40.0, 499.8, marker_l.radius / 1.55, 0.8)
    P, il, ir = markers.stereo_pairs([spot_l, marker_l], [marker_r], Q)
    assert list(il) == [1] and list(ir) == [0]


def test_masks_cover_markers_but_not_laser_spots():
    """Only blobs that may be markers are kept out of the laser line search: a spot that pairs as a 1.4 mm thing IS
    laser line (masking it cut a hole into the line); a blob seen in one view only is masked when it could be a marker
    somewhere in the depth range."""
    z, d = 300.0, F * BASE / 300.0
    marker = _stereo_blob(1300.0, 500.0, 4.5, z)
    spot = _stereo_blob(1100.0, 300.0, 1.4, z)
    lone_big = markers.Blob(400.0, 900.0, 9.0, 0.8)                  # no partner: could be a marker
    lone_small = markers.Blob(300.0, 1000.0, 0.5 * 2.5 * F / 500.0 - 0.5, 0.8)   # smaller than any marker in range
    left = [marker, spot, lone_big, lone_small]
    right = [markers.Blob(b.x - d, b.y, b.radius, b.axis_ratio) for b in (marker, spot)]
    keep_l, keep_r = markers.plausible(left, right, Q)
    assert keep_l.tolist() == [True, False, True, False] and keep_r.tolist() == [True, False]
    ml, mr = markers.masks((1200, 1600), left, right, Q)
    assert ml[500, 1300] and not ml[300, 1100] and ml[900, 400] and not ml[1000, 300]
    assert mr[500, int(1300 - d)] and not mr[300, int(1100 - d)]
    P, il, _ = markers.stereo_pairs(left, right, Q)
    assert list(il) == [0]                                           # the spot never becomes a marker
