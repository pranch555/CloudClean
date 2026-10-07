"""Native MetroY capture: laser calibration file, stripe triangulation with line identity, markers, tracking.

The geometry tests render synthetic scenes through the scanner's REAL factory rectification (tests/data/metroy is a
copy of the calibration read off our MetroY Ultra), so they exercise the same camera model the driver runs with.
The scene is a box on a floor lit by 17 laser sheets from a projector beside the cameras - with the occlusions and
laser shadows that make stereo-order matching slip by a stripe - and every triangulated point must land on the true
surface.
"""
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cloudclean.capture.metroy import laserfile, markers  # noqa: E402
from cloudclean.capture.metroy.stripes import Triangulator, smooth_tracks, stripe_centres  # noqa: E402

DATA = Path(__file__).parent / "data" / "metroy"


@pytest.fixture(scope="module")
def tri():
    return Triangulator.from_yaml(str(DATA / "camparam.yaml"), extra=str(DATA / "metroExtra.bin"))


# ----------------------------------------------------------------------------------------------- calibration file
def test_laser_file_layout():
    sets = laserfile.parse(str(DATA / "metroExtra.bin"))
    assert laserfile.calibration_date(str(DATA / "metroExtra.bin")) == "20260902100715"
    assert {i: (s.mode, s.lines) for i, s in sets.items()} == {0: (1, 17), 1: (1, 17), 2: (2, 15), 3: (0, 1)}
    for i in (0, 1, 2):
        s = sets[i]
        # left->right then right->left returns to the start, on points where each line actually is (its region)
        errs = []
        for k in range(s.lines):
            E = s.records[k, 12:]
            ys = np.linspace(min(E[1], E[3]) + 20, max(E[1], E[3]) - 20, 25)
            xs = E[0] + (E[2] - E[0]) * (ys - E[1]) / (E[3] - E[1])
            xr = laserfile.monomials(xs, ys) @ s.forward[k]
            back = laserfile.monomials(xr, ys) @ s.reverse[k]
            errs.append(back - xs)
        assert np.sqrt(np.mean(np.square(errs))) < 0.5


# ----------------------------------------------------------------------------------------------- synthetic scene
FLOOR, TOP = 380.0, 330.0                    # mm; the box is 50 mm tall
BOX_X, BOX_Y = (-25.0, 65.0), (-40.0, 20.0)
PROJECTOR = np.array([80.0, -10.0, -20.0])   # where the real projector sits relative to the left camera


def _cast(o, d):
    """Nearest hit of rays o + t d (t > 0) with the floor, the box top and the four box walls."""
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
    """Left<->right maps for synthetic laser sheets, fitted exactly as the factory ones are shaped."""
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


def _render(tri, normals, offsets, origin, cx, sigma_mm=0.5, seed=0):
    f, cy = tri.P1[0, 0], tri.P1[1, 2]
    v, u = np.mgrid[0:1200, 0:1600].astype(np.float64)
    d = np.c_[((u - cx) / f).ravel(), ((v - cy) / f).ravel(), np.ones(u.size)]
    t = _cast(origin, d)
    X = origin + t[:, None] * d
    lit = np.isfinite(t)
    t_proj = _cast(PROJECTOR, X - PROJECTOR)                   # shadow: does the laser reach this point?
    lit &= t_proj > 1 - 1e-6
    dist = X @ normals.T - offsets
    stripe = 210 * np.exp(-0.5 * (dist / sigma_mm) ** 2).max(1)
    img = 25 + np.where(lit, stripe, 0) + np.random.default_rng(seed).normal(0, 2, len(t))
    return np.clip(img, 0, 255).astype(np.uint8).reshape(1200, 1600)


def _true_depth_error(points):
    t = _cast(np.zeros(3), points / points[:, 2:3])
    return points[:, 2] - t          # the ray from the left camera through each point, Z = t for unit-z rays


def test_line_identity_triangulates_a_step_exactly(tri):
    normals, offsets = _laser_planes()
    line_set = _line_set(tri, normals, offsets)
    left = _render(tri, normals, offsets, np.zeros(3), tri.P1[0, 2])
    right = _render(tri, normals, offsets, np.array([tri.B, 0.0, 0.0]), tri.P2[0, 2], seed=1)
    cl = smooth_tracks(stripe_centres(left, tri.p), tri.p)
    cr = smooth_tracks(stripe_centres(right, tri.p), tri.p)

    saved = tri.line_sets
    tri.line_sets = [line_set]
    try:
        pts, _, _ = tri.triangulate_by_lines(cl, cr)
    finally:
        tri.line_sets = saved
    assert len(pts) > 5000
    err = np.abs(_true_depth_error(pts.astype(np.float64)))
    # no stripe paired with the wrong partner: a one-stripe slip lands 10-40 mm off. What remains are points on the
    # box's convex edges, where any stripe smoothing rounds the corner (0.2% of points, < 2 mm)
    assert err.max() < 3.0, f"worst point {err.max():.2f} mm off the surface"
    assert np.percentile(err, 99.5) < 0.1
    heights = FLOOR - pts[:, 2]
    assert ((np.abs(heights) < 0.2) | (np.abs(heights - 50) < 0.2) | ((heights > 0) & (heights < 50))).all()
    assert (np.abs(heights - 50) < 0.2).sum() > 500      # the box top is measured, and measured 50 mm up


# ----------------------------------------------------------------------------------------------- markers
def _marker_images(tri, world_pts, radius_mm=3.0):
    f, cy = tri.P1[0, 0], tri.P1[1, 2]
    imgs = []
    for origin_x, cx in ((0.0, tri.P1[0, 2]), (tri.B, tri.P2[0, 2])):
        img = np.full((1200, 1600), 25, np.uint8)
        for P in world_pts:
            x, y, r = f * (P[0] - origin_x) / P[2] + cx, f * P[1] / P[2] + cy, f * radius_mm / P[2]
            cv2.ellipse(img, (int(round(x * 16)), int(round(y * 16))), (int(round(r * 16)), int(round(r * 12))),
                        0, 0, 360, 255, -1, cv2.LINE_AA, shift=4)
        cv2.line(img, (0, 900), (1600, 100), 230, 6)                       # a laser stripe through the scene
        imgs.append(img)
    return imgs


def test_markers_are_detected_matched_and_triangulated(tri):
    rng = np.random.default_rng(3)
    truth = np.c_[rng.uniform(-40, 160, 12), rng.uniform(-90, 90, 12), rng.uniform(300, 380, 12)]
    # two markers on one rectified row could pair either way; like Revo Metro, stereo() drops such pairs, so the
    # scene keeps them apart
    truth[:, 1] = np.linspace(-90, 90, 12) + rng.uniform(-3, 3, 12)
    f = tri.P1[0, 0]
    xr = f * (truth[:, 0] - tri.B) / truth[:, 2] + tri.P2[0, 2]
    xl = f * truth[:, 0] / truth[:, 2] + tri.P1[0, 2]
    both = ((xr > 30) & (xr < 1570) & (xl > 30) & (xl < 1570)).sum()      # a marker needs both cameras
    left, right = _marker_images(tri, truth)
    pts = markers.stereo(markers.detect(left), markers.detect(right), tri.Q)
    assert len(pts) >= both - 1 and both >= 8
    d = np.linalg.norm(pts[:, None] - truth[None], axis=2).min(1)
    assert np.median(d) < 0.1 and d.max() < 0.3, d


def _pose(rx, ry, rz, t):
    R = cv2.Rodrigues(np.array([rx, ry, rz], float))[0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def test_marker_tracker_follows_a_sweep_and_recovers_after_a_jump():
    rng = np.random.default_rng(5)
    world = np.c_[rng.uniform(-200, 200, 60), rng.uniform(-150, 150, 60), rng.uniform(-5, 5, 60)]
    tracker = markers.MarkerTracker()
    errors, states = [], []
    for i in range(80):
        if i == 60:
            i += 25                                   # the scanner moves a lot while no frame arrives
        T_true = _pose(np.pi + 0.002 * i, 0.001 * i, 0.003 * i, [2.0 * i - 60, 0.5 * i, 330.0])  # sensor->world
        local = (world - T_true[:3, 3]) @ T_true[:3, :3]                                         # world->sensor
        visible = (np.abs(local[:, 0]) < 0.35 * local[:, 2]) & (np.abs(local[:, 1]) < 0.28 * local[:, 2])
        seen = local[visible] + rng.normal(0, 0.02, (visible.sum(), 3))
        if len(tracker.world) == 0:
            tracker.initial = T_true                   # anchor the map to the true world for the comparison
        res = tracker.track(seen)
        states.append(res.state)
        if res.pose is not None:
            errors.append(np.abs(res.pose[:3, 3] - T_true[:3, 3]).max())
    assert "relocalized" in states or states[60] == "tracked"
    assert states.count("lost") == 0
    assert max(errors) < 0.15, f"worst pose error {max(errors):.3f} mm"


# ----------------------------------------------------------------------------------------------- driver
def test_driver_explains_why_it_cannot_capture_here():
    import platform

    from cloudclean.capture.drivers import create_driver
    drv = create_driver("metroy_usb")
    ok, reason = drv.availability()
    if platform.system() != "Linux":
        assert not ok and ("Linux" in reason or "No MetroY" in reason)
    keys = [s["key"] for s in drv.settings_schema()]
    assert {"matching", "point_distance", "tracking"} <= set(keys)


def _plate_view(pitch_deg=28.5, spin_deg=0.0, rng=None):
    """Markers in a ring on a turntable plate (world z = 0, turning about Z), as a scanner 300 mm away pitched down
    by pitch_deg sees them: (sensor->world pose, markers in sensor coordinates)."""
    a = np.radians(np.arange(0, 360, 30) + spin_deg)
    world = np.c_[75 * np.cos(a), 75 * np.sin(a), np.zeros(len(a))]
    p = np.radians(pitch_deg)
    fwd = np.array([0.0, np.cos(p), -np.sin(p)])               # looking along +Y and down onto the plate
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(fwd, right)                                 # sensor y points down the image
    T = np.eye(4)
    T[:3, :3] = np.c_[right, down, fwd]
    T[:3, 3] = -300 * fwd
    local = (world - T[:3, 3]) @ T[:3, :3]
    if rng is not None:
        local = local + rng.normal(0, 0.03, local.shape)
    return T, local


def test_scan_stands_on_the_turntable_plate_from_the_first_marker_frame():
    """The user's first native turntable scan came out lying on its side: the map started on a frame that had
    markers but no laser points yet, so the scan stayed in sensor coordinates. The markers on the plate fix the
    level: Z along the plate's normal, the plate at z = 0, whatever the scanner's pitch."""
    from cloudclean.capture.drivers.metroy_usb import _initial_pose
    rng = np.random.default_rng(3)
    for pitch in (15.0, 28.5, 60.0):
        T_true, local = _plate_view(pitch, rng=rng)
        T = _initial_pose(np.zeros((0, 3)), local)
        M = T @ np.linalg.inv(T_true)                            # true world -> CloudClean's world
        assert np.degrees(np.arccos(np.clip(M[2, 2], -1, 1))) < 0.1, "the plate's normal is Z"
        assert abs(M[2, 3]) < 0.05, "the plate is at z = 0"
        part_top = markers.apply(M, np.array([[0.0, 0.0, 40.0]]))
        assert abs(part_top[0, 2] - 40.0) < 0.1                  # a part on the plate stands above it
    # the tracker builds its map in that frame, so every later pose (the table turned) is level too
    T_true, local = _plate_view()
    tracker = markers.MarkerTracker()
    tracker.initial = _initial_pose(np.zeros((0, 3)), local)
    assert tracker.track(local).state == "init"
    assert np.abs(tracker.world[:, 2]).max() < 1e-6


def test_markers_off_one_table_keep_the_sensor_up_frame():
    from cloudclean.capture.drivers.metroy_usb import _initial_pose
    sensor_up = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    _, local = _plate_view()
    on_part = local.copy()
    on_part[0] += np.array([0.0, -10.0, 0.0])                    # one marker stuck on the part, 10 mm up
    T = _initial_pose(np.zeros((0, 3)), on_part)
    assert np.allclose(T[:3, :3], sensor_up)
    assert np.allclose(markers.apply(T, on_part).mean(axis=0), 0, atol=1e-9)   # not left 300 mm away
    board = np.c_[np.linspace(-60, 60, 6), np.tile([-30.0, 30.0], 3), np.full(6, 300.0)]   # a wall facing the scanner
    assert np.allclose(_initial_pose(np.zeros((0, 3)), board)[:3, :3], sensor_up)
    pts = np.array([[0.0, 0.0, 250.0], [10.0, 0.0, 250.0]])
    assert np.allclose(markers.apply(_initial_pose(pts, None), pts).mean(axis=0), 0)   # untracked: as before


def test_regular_marker_grid_never_gives_a_wrong_pose():
    """Four markers of a regular grid fit in many places: recovering the pose from them alone must refuse (a wrong
    pose is what fuses a surface a second time at an offset), and a larger, unambiguous view must be accepted."""
    g = np.array([(x, y, 0.0) for x in range(0, 150, 30) for y in range(0, 150, 30)], float)
    T_true = _pose(np.pi, 0.0, 0.0, [60.0, 60.0, 300.0])
    local = (g - T_true[:3, 3]) @ T_true[:3, :3]
    tracker = markers.MarkerTracker()
    tracker.initial = T_true
    assert tracker.track(local).state == "init"
    tracker._T1 = tracker._T2 = None                     # tracking was lost: no prediction to lean on
    square = local[[0, 1, 5, 6]]                         # a 2x2 patch: identical anywhere on the grid
    res = tracker.track(square)
    assert res.pose is None or np.abs(res.pose[:3, 3] - T_true[:3, 3]).max() < 0.2
    # breaking the symmetry (an extra marker off the grid) makes it unambiguous again
    extra = np.array([[37.0, 81.0, 0.0]])
    tracker2 = markers.MarkerTracker()
    tracker2.initial = T_true
    tracker2.track(np.vstack([local, (extra - T_true[:3, 3]) @ T_true[:3, :3]]))
    tracker2._T1 = tracker2._T2 = None
    view = np.vstack([local[[5, 6, 10, 11, 12]], (extra - T_true[:3, 3]) @ T_true[:3, :3]])
    res2 = tracker2.track(view)
    assert res2.pose is not None and np.abs(res2.pose[:3, 3] - T_true[:3, 3]).max() < 0.2


def test_marker_map_first_then_frozen():
    rng = np.random.default_rng(7)
    world = np.c_[rng.uniform(-150, 150, 40), rng.uniform(-100, 100, 40), rng.uniform(-3, 3, 40)]
    tracker = markers.MarkerTracker()
    for i in range(120):                                  # the "map markers" sweep
        T_true = _pose(np.pi + 0.3 * np.sin(i / 20), 0.2 * np.cos(i / 25), 0.01 * i, [1.5 * i - 90, 20 * np.sin(i / 15), 320.0])
        local = (world - T_true[:3, 3]) @ T_true[:3, :3]
        vis = (np.abs(local[:, 0]) < 0.35 * local[:, 2]) & (np.abs(local[:, 1]) < 0.28 * local[:, 2])
        if len(tracker.world) == 0:
            tracker.initial = T_true
        tracker.track(local[vis] + rng.normal(0, 0.03, (vis.sum(), 3)))
    info = tracker.freeze()
    assert tracker.frozen and info["markers"] >= 30 and info["rms_mm"] < 0.1
    d = np.linalg.norm(tracker.world[:, None] - world[None], axis=2).min(1)
    assert np.median(d) < 0.05 and d.max() < 0.3
    before = tracker.world.copy()
    T_true = _pose(np.pi, 0.0, 0.0, [0.0, 0.0, 320.0])
    local = (world - T_true[:3, 3]) @ T_true[:3, :3]
    vis = (np.abs(local[:, 0]) < 0.35 * local[:, 2]) & (np.abs(local[:, 1]) < 0.28 * local[:, 2])
    res = tracker.track(local[vis] + 0.3)                 # a biased frame must not move a frozen map
    assert np.array_equal(before, tracker.world)
