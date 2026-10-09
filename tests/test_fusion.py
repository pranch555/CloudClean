"""Averaging fusion on parts of known shape: noisy laser-line frames with per-frame tracking jitter go through the
capture session; the averaged cloud must be thinner than the raw one and stay where the true surface is (no shrinkage,
no rounded edge, no step or wall pulled towards another surface)."""
import numpy as np
import pytest

from cloudclean.capture.drivers.base import Frame
from cloudclean.capture.fusion import SurfaceField, project
from cloudclean.capture.session import CaptureSession

POINT_NOISE = 0.012       # the MetroY's single-frame spec: 0.01-0.015 mm
JITTER_MM = 0.035         # per-frame tracking offset along each axis; the replayed bust: 0.038 mm per frame
JITTER_DEG = 0.004        # ~0.02 mm at 300 mm


class _Driver:
    id = name = "test"
    exhausted = False

    def connect(self, settings):
        return {"point_distance": 0.2}

    def capabilities(self):
        return {"range_mm": [200.0, 430.0]}

    def status(self):
        return {}

    def start(self):
        pass

    def stop(self):
        pass

    def disconnect(self):
        pass


def look_at(eye, target):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0] if abs(z[2]) < 0.95 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = eye
    return T


def jitter(rng):
    """A small rigid error, as marker tracking leaves on each frame's pose."""
    axis = rng.normal(size=3)
    a = np.radians(rng.normal(scale=JITTER_DEG)) * axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    E = np.eye(4)
    E[:3, :3] = np.eye(3) + K + K @ K / 2
    E[:3, 3] = rng.normal(scale=JITTER_MM, size=3)
    return E


def capture(hit, poses, fusion, seed=0, half=0.06, lines=17, step=3e-4, still=0):
    """Laser-line frames of the surface `hit(origin, dirs) -> distance (nan = miss)` seen from `poses`; the first
    `still` frames lay their stripes in the same place (a scanner and part standing still)."""
    rng = np.random.default_rng(seed)
    session = CaptureSession(_Driver(), {"fusion": fusion, "speed_limit": 5000})
    session.connect()
    v = np.arange(-half, half, step)
    for i, T in enumerate(poses):
        u = (np.arange(lines) + (0.37 if i < still else rng.uniform())) / lines * 2 * half - half
        d = np.stack([np.repeat(u, len(v)), np.tile(v, lines), np.ones(lines * len(v))], axis=1)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t = hit(T[:3, 3], d @ T[:3, :3].T)
        ok = np.isfinite(t)
        pts = d[ok] * (t[ok] + rng.normal(scale=POINT_NOISE, size=ok.sum()))[:, None]
        session.process_frame(Frame(points=pts, pose=T @ jitter(rng), timestamp=i / 45.0, coordinates="sensor"))
    cloud, density = session.build_cloud()
    assert len(density) == len(cloud.points)
    return np.asarray(cloud.points), session


def ring(target, distance, elevations, n):
    return [look_at(np.asarray(target) + distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]),
                    target)
            for e in np.radians(elevations) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]


def fit_sphere(p):
    A = np.c_[2 * p, np.ones(len(p))]
    x = np.linalg.lstsq(A, (p ** 2).sum(axis=1), rcond=None)[0]
    c, r = x[:3], np.sqrt(x[3] + x[:3] @ x[:3])
    for _ in range(5):                        # geometric refinement (Gauss-Newton)
        d = p - c
        dist = np.linalg.norm(d, axis=1)
        J = np.c_[-d / dist[:, None], -np.ones(len(p))]
        step = np.linalg.lstsq(J, -(dist - r), rcond=None)[0]
        c, r = c + step[:3], r + step[3]
    return c, r


def thickness(err):
    return 2 * np.percentile(np.abs(err), 95)


def test_sphere_gets_thinner_and_keeps_its_radius():
    R = 15.0
    centre = np.array([0.0, 0.0, 0.0])

    def hit(o, d):
        b = d @ (o - centre)
        disc = b * b - ((o - centre) @ (o - centre) - R * R)
        return np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.nan)

    poses = ring(centre, 300.0, (15, 45, 70), 40)
    out = {}
    for mode in ("raw", "average"):
        p, session = capture(hit, poses, mode)
        err = np.linalg.norm(p - centre, axis=1) - R
        c, r = fit_sphere(p)
        out[mode] = {"points": len(p), "thickness": thickness(err), "mean": err.mean(), "radius": r - R,
                     "centre": np.linalg.norm(c - centre), "stats": session.fusion_stats}
    print("\nsphere R=15:", {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}
                            for m, o in out.items()})
    raw, avg = out["raw"], out["average"]
    assert avg["points"] == raw["points"], "averaging moves points, it never adds or removes them"
    assert avg["thickness"] < 0.5 * raw["thickness"], (raw["thickness"], avg["thickness"])
    assert abs(avg["radius"]) < 0.01 and abs(avg["mean"]) < 0.01, avg      # no shrinkage
    assert avg["centre"] < 0.01
    assert avg["stats"]["moved_pct"] > 80


def test_step_and_plane_positions_stay_unbiased_and_the_edge_is_not_rounded():
    h = 0.5          # step height; the edge runs along y at x = 0

    def hit(o, d):
        t1 = (h - o[2]) / d[:, 2]
        x1, y1 = o[0] + t1 * d[:, 0], o[1] + t1 * d[:, 1]
        t0 = -o[2] / d[:, 2]
        x0, y0 = o[0] + t0 * d[:, 0], o[1] + t0 * d[:, 1]
        t = np.where(x1 >= 0, t1, np.where(x0 < 0, t0, np.nan))      # the wall itself: not measured
        x, y = np.where(x1 >= 0, x1, x0), np.where(x1 >= 0, y1, y0)
        return np.where((np.abs(x) < 12) & (np.abs(y) < 12), t, np.nan)

    def signed_distance(p):        # to the true step (both faces and the wall), + = above the material
        x, z = p[:, 0], p[:, 2]
        low = np.where(x <= 0, np.abs(z), np.hypot(x, z))
        wall = np.where((z >= 0) & (z <= h), np.abs(x), np.minimum(np.hypot(x, z), np.hypot(x, z - h)))
        high = np.where(x >= 0, np.abs(z - h), np.hypot(x, z - h))
        inside = np.where(x < 0, z < 0, z < h)
        return np.where(inside, -1, 1) * np.minimum(np.minimum(low, wall), high)

    poses = ring([0.0, 0.0, 0.0], 300.0, (55, 70, 85), 30)
    res = {}
    for mode in ("raw", "average"):
        p, session = capture(hit, poses, mode, half=0.05)
        err = signed_distance(p)
        far = np.abs(p[:, 0]) > 1.0
        near = np.abs(p[:, 0]) < 0.6
        res[mode] = {"low": err[far & (p[:, 0] < 0)].mean(), "high": err[far & (p[:, 0] > 0)].mean(),
                     "thickness": thickness(err[far]), "edge_mean": err[near].mean(),
                     "edge_thickness": thickness(err[near]), "stats": session.fusion_stats}
    print("\nstep 0.5 mm:", {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}
                            for m, o in res.items()})
    raw, avg = res["raw"], res["average"]
    assert abs(avg["low"]) < 0.01 and abs(avg["high"]) < 0.01, avg               # both planes where they are
    assert abs(avg["high"] - avg["low"]) < 0.01, "the step height must not change"
    assert avg["thickness"] < 0.5 * raw["thickness"], (raw, avg)
    # within 0.6 mm of the edge the points are averaged on their own face or left as measured: never pulled across
    # the step or rounded into it
    assert abs(avg["edge_mean"]) <= abs(raw["edge_mean"]) + 0.005, (raw, avg)
    assert avg["edge_thickness"] <= raw["edge_thickness"] + 0.005, (raw, avg)


@pytest.mark.parametrize("angle", [90, 135])
def test_a_sharp_edge_is_not_rounded(angle):
    """A ridge z = -s |x| (90 or 135 degrees between the faces) seen from above: averaging must not pull the points
    near the ridge inside."""
    s = np.tan(np.radians(90 - angle / 2))

    def hit(o, d):
        tl = (s * o[0] - o[2]) / (d[:, 2] - s * d[:, 0])          # face z = s x (x <= 0)
        tr = (-s * o[0] - o[2]) / (d[:, 2] + s * d[:, 0])         # face z = -s x (x >= 0)
        tl = np.where((tl > 0) & (o[0] + tl * d[:, 0] <= 0), tl, np.inf)
        tr = np.where((tr > 0) & (o[0] + tr * d[:, 0] >= 0), tr, np.inf)
        t = np.minimum(tl, tr)
        y = o[1] + t * d[:, 1]
        x = o[0] + t * d[:, 0]
        return np.where(np.isfinite(t) & (np.abs(y) < 10) & (np.abs(x) < 8), t, np.nan)

    def signed_distance(p):        # + = outside the material z <= -s |x|
        a, b = (p[:, 2] - s * p[:, 0]) / np.hypot(1, s), (p[:, 2] + s * p[:, 0]) / np.hypot(1, s)
        # between the two face normals above the ridge the nearest point is the ridge itself
        ridge = (p[:, 2] > 0) & (np.abs(p[:, 0]) <= s * p[:, 2])
        return np.where(ridge, np.hypot(p[:, 0], p[:, 2]), np.maximum(a, b))

    poses = ring([0.0, 0.0, -2.0], 300.0, (50, 65, 80), 30)
    res = {}
    for mode in ("raw", "average"):
        p, session = capture(hit, poses, mode, half=0.05)
        err = signed_distance(p)
        near, far = np.abs(p[:, 0]) < 0.3, np.abs(p[:, 0]) > 1.0
        # medians: near a convex ridge the distance of a point pushed inside is shorter than its push (it is close to
        # the other face too), so the mean of even the raw points reads 0.004-0.008 mm outside there
        res[mode] = {"ridge_median": np.median(err[near]), "ridge_p05": np.percentile(err[near], 5),
                     "ridge_p95": np.percentile(err[near], 95), "faces_median": np.median(err[far]),
                     "faces_thickness": thickness(err[far]), "stats": session.fusion_stats}
    print(f"\n{angle} degree ridge:", {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}
                                     for m, o in res.items()})
    raw, avg = res["raw"], res["average"]
    assert abs(avg["faces_median"]) < 0.01 and avg["faces_thickness"] < 0.5 * raw["faces_thickness"]
    # not rounded: the points within 0.3 mm of the ridge sit where the faces do, as the raw ones do (a plane fit alone
    # pulled them 0.031 mm inside at 90 degrees, 0.020 mm at 135), none is pulled deeper than the raw ones go, and none
    # is pushed out past the ridge (onto the other face's plane)
    shift = (avg["ridge_median"] - avg["faces_median"]) - (raw["ridge_median"] - raw["faces_median"])
    assert abs(shift) < 0.005, (shift, raw, avg)
    assert avg["ridge_p05"] >= raw["ridge_p05"] - 0.005, (raw, avg)
    assert avg["ridge_p95"] <= raw["ridge_p95"] + 0.005, (raw, avg)


def test_thin_wall_seen_from_both_sides_is_not_merged():
    """A 0.4 mm plate scanned from above and below: two surfaces 2 cells apart must stay 0.4 mm apart."""
    gap = 0.4

    def hit(o, d):
        z = 0.0 if o[2] > 0 else -gap
        t = (z - o[2]) / d[:, 2]
        x, y = o[0] + t * d[:, 0], o[1] + t * d[:, 1]
        return np.where((np.abs(x) < 10) & (np.abs(y) < 10), t, np.nan)

    poses = ring([0.0, 0.0, 0.0], 300.0, (60, 80), 20) + ring([0.0, 0.0, -gap], 300.0, (-60, -80), 20)
    res = {}
    for mode in ("raw", "average"):
        p, session = capture(hit, poses, mode, half=0.04)
        top = p[:, 2] > -gap / 2
        res[mode] = {"top": p[top, 2].mean(), "bottom": p[~top, 2].mean() + gap,
                     "thickness_top": thickness(p[top, 2]), "stats": session.fusion_stats}
    print("\nthin wall 0.4 mm:", {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}
                                 for m, o in res.items()})
    avg = res["average"]
    assert abs(avg["top"]) < 0.01 and abs(avg["bottom"]) < 0.01, avg
    assert avg["thickness_top"] <= res["raw"]["thickness_top"] + 0.005


def test_a_long_standstill_before_turning_is_still_averaged():
    """2026-10-09: the turntable stood still 65 s (3400 frames of the same 17 stripes) before it turned. Those stripes
    outweighed the turning frames in their blocks 100:1 and 3.6 % of the bust stayed raw along 17 lines. A sphere
    seen still for 400 frames, then all around, must be averaged there too and keep its radius."""
    R = 15.0

    def hit(o, d):
        b = d @ o
        disc = b * b - (o @ o - R * R)
        return np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.nan)

    still = [look_at(np.array([300.0, 0.0, 80.0]), [0.0, 0.0, 0.0])] * 400
    p, session = capture(hit, still + ring([0.0, 0.0, 0.0], 300.0, (15, 45, 70), 40), "average", seed=4,
                         still=len(still))
    err = np.linalg.norm(p, axis=1) - R
    st = session.fusion_stats
    print(f"\nstill then turning: thickness {thickness(err):.4f}, mean {err.mean():+.5f}, {st}")
    assert st["kept_single_stripe_pct"] < 2.0 and st["moved_pct"] > 95, st      # 6 % stayed raw before
    assert abs(err.mean()) < 0.005 and abs(fit_sphere(p)[1] - R) < 0.01


def test_raw_fusion_keeps_the_points_as_measured():
    """The old behaviour stays selectable: fusion "raw" saves exactly the capped raw points, nothing moved."""
    rng = np.random.default_rng(1)
    session = CaptureSession(_Driver(), {"fusion": "raw"})
    session.connect()
    pts = rng.normal(scale=5, size=(4000, 3)) + [0, 0, 300]
    session.process_frame(Frame(points=pts, pose=np.eye(4), timestamp=0.0, coordinates="sensor"))
    assert session.field is None and session.report()["fusion"]["method"] == "raw"
    cloud, _ = session.build_cloud()
    kept = np.concatenate([c[0] for c in session.chunks]).astype(np.float64)
    assert np.array_equal(np.asarray(cloud.points), kept) and session.fusion_stats is None


def test_field_moments_add_up_like_one_big_fit():
    """Adding in many small frames or at once gives the same field; the plane of a noisy plane is unbiased."""
    rng = np.random.default_rng(2)
    p = np.c_[rng.uniform(0, 4, 50_000), rng.uniform(0, 4, 50_000), 1.234 + rng.normal(scale=0.05, size=50_000)]
    a, b = SurfaceField(0.2), SurfaceField(0.2)
    a.add(p)
    for chunk in np.array_split(p, 37):
        b.add(chunk)
    q = p[:5000]
    pa, sa = project(a, q)
    pb, _ = project(b, q)
    assert np.allclose(pa, pb, atol=1e-9)
    inner = (q[:, :2] > 0.6).all(axis=1) & (q[:, :2] < 3.4).all(axis=1)
    assert abs(pa[inner, 2].mean() - 1.234) < 0.002 and pa[inner, 2].std() < 0.012
    assert sa["moved_pct"] > 95


@pytest.mark.parametrize("R", [1.0, 3.0])
def test_projection_does_not_shrink_a_small_radius(R):
    """Curvature: the plane of a 0.6 mm block lies inside a convex surface by about a^2 / (6 R) - 0.0056 mm on a 3 mm
    radius, 0.017 mm on 1 mm (measured without the correction). The correction must take it below 0.002 mm, inside
    and outside (a hole of the same radius)."""
    rng = np.random.default_rng(3)
    n = int(70_000 * R)
    th = rng.uniform(0, 2 * np.pi, n)
    r = R + rng.normal(scale=0.03, size=n)
    p = np.c_[r * np.cos(th), r * np.sin(th), rng.uniform(0, 4, n)]
    f = SurfaceField(0.2)
    f.add(p)
    q = p[(p[:, 2] > 1) & (p[:, 2] < 3)]
    out, stats = project(f, q)
    plain, _ = project(f, q, curvature=False)
    radial = np.linalg.norm(out[:, :2], axis=1) - R
    before = np.linalg.norm(plain[:, :2], axis=1).mean() - R
    print(f"\ncylinder R={R}: mean radial error {radial.mean():+.5f} mm (plane only {before:+.5f}), "
          f"std {radial.std():.4f}, {stats}")
    assert abs(radial.mean()) < 0.002 and radial.std() < 0.01 and stats["moved_pct"] > 95
