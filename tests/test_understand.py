"""Part understanding and measuring (Contracts 2 and 3): robust fits on synthetic shapes with known answers, the part
frame, regions (validation, masks, the shapes the browser evaluates, distances, edit ops), the summary with its
description and cache, the measurement functions and their endpoints."""
import math
import time

import numpy as np
import open3d as o3d
import pytest
from fastapi.testclient import TestClient

from cloudclean import fitting as F
from cloudclean import understand as U
from cloudclean.capture.drivers.simulated import part_mesh
from cloudclean.edit import apply_edits, validate_ops
from cloudclean.io import save
from cloudclean.regions import (check_region, part_frame, region_distance, region_mask, resolve_region,
                                vertex_weights)
from cloudclean.web.server import create_app
from cloudclean.web.workspace import Workspace
from tests.synthetic import random_rigid

QUIET = lambda msg: None  # noqa: E731


# --------------------------------------------------------------------------- synthetic parts
def cylinder_mesh(radius: float, height: float, z0: float = 0.0, resolution: int = 128) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius, height, resolution, 8)
    return mesh.translate((0, 0, z0 + height / 2))


def bolt_mesh() -> o3d.geometry.TriangleMesh:
    """A Ø20 x 70 mm shank with a Ø38 x 38 mm head on top (108 mm long)."""
    mesh = cylinder_mesh(10.0, 70.0) + cylinder_mesh(19.0, 38.0, 70.0)
    mesh.compute_vertex_normals()
    return mesh


def scan_of(mesh, n: int, noise: float = 0.02, seed: int = 0, transform=None) -> o3d.geometry.PointCloud:
    o3d.utility.random.seed(seed)
    pcd = mesh.sample_points_uniformly(n, use_triangle_normal=True)
    pts = np.asarray(pcd.points) + np.random.default_rng(seed).normal(scale=noise, size=(n, 3))
    pcd.points = o3d.utility.Vector3dVector(pts)
    if transform is not None:
        pcd.transform(transform)
    return pcd


def box_mesh(x=50.0, y=30.0, z=20.0) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_box(x, y, z)


# --------------------------------------------------------------------------- fits
def test_fits_recover_known_shapes_with_noise_and_outliers():
    rng = np.random.default_rng(1)
    # plane z' = tilted, 20 % outliers
    normal = np.array([0.2, -0.3, 0.93])
    normal /= np.linalg.norm(normal)
    u, v = F._basis(normal)
    uv = rng.uniform(-20, 20, (40_000, 2))
    plane = 5 * normal + uv[:, :1] * u + uv[:, 1:] * v + rng.normal(scale=0.02, size=(40_000, 1)) * normal
    fit = F.fit_plane(np.vstack([plane, rng.uniform(-30, 30, (8_000, 3))]))
    assert abs(abs(np.dot(fit["normal"], normal)) - 1) < 1e-6
    assert abs(np.dot(fit["point"], normal) - 5) < 0.005
    assert 0.017 < fit["rms"] < 0.023 and fit["flatness"] < 0.2 and fit["inliers"] >= 39_500

    # partial sphere (upper 60 %), outliers
    d = rng.normal(size=(30_000, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    d = d[d[:, 2] > -0.2]
    sphere = np.array([3.0, -2.0, 7.0]) + 12.5 * d + rng.normal(scale=0.02, size=d.shape)
    fit = F.fit_sphere(np.vstack([sphere, rng.uniform(-10, 20, (2_000, 3))]))
    assert np.allclose(fit["center"], [3, -2, 7], atol=0.005) and abs(fit["radius"] - 12.5) < 0.003
    assert 0.4 < fit["coverage"] < 0.8

    # half cylinder, tilted axis
    axis = np.array([0.3, 0.1, 1.0])
    axis /= np.linalg.norm(axis)
    cu, cv = F._basis(axis)
    th, z = rng.uniform(0, math.pi, 40_000), rng.uniform(-15, 15, 40_000)
    cyl = np.array([10, 20, -5]) + (9.95 * np.cos(th))[:, None] * cu + (9.95 * np.sin(th))[:, None] * cv \
        + z[:, None] * axis + rng.normal(scale=0.02, size=(40_000, 3))
    fit = F.fit_cylinder(cyl)
    assert abs(fit["radius"] - 9.95) < 0.003 and abs(abs(np.dot(fit["axis"], axis)) - 1) < 1e-5
    assert abs(fit["length"] - 30) < 0.1 and 170 <= fit["coverage_deg"] <= 200
    assert fit["points_used"] <= F.CYLINDER_SAMPLE and fit["inliers"] > 39_000

    # circles and a line
    th = rng.uniform(0, 2 * math.pi, 5_000)
    fit = F.fit_circle_2d(np.c_[3 + 4 * np.cos(th), -1 + 4 * np.sin(th)] + rng.normal(scale=0.01, size=(5_000, 2)))
    assert np.allclose(fit["center"], [3, -1], atol=0.002) and abs(fit["radius"] - 4) < 0.002
    ring = cyl[np.abs((cyl - np.array([10, 20, -5])) @ axis) < 0.2]
    fit = F.fit_circle_3d(ring)
    assert abs(fit["radius"] - 9.95) < 0.02 and abs(abs(np.dot(fit["normal"], axis)) - 1) < 1e-3
    line = np.array([1, 2, 3]) + np.linspace(-10, 10, 2_000)[:, None] * np.array([0.6, 0.8, 0.0])
    fit = F.fit_line(line + rng.normal(scale=0.01, size=line.shape))
    assert np.allclose(fit["direction"], [0.6, 0.8, 0], atol=1e-4) and abs(fit["length"] - 20) < 0.05


def test_fits_are_exact_on_exact_data_and_refuse_too_few_points():
    rng = np.random.default_rng(2)
    pts = np.c_[rng.uniform(-5, 5, (500, 2)), np.full(500, 2.5)]
    fit = F.fit_plane(pts)
    assert fit["rms"] < 1e-12 and fit["flatness"] < 1e-12 and np.allclose(np.abs(fit["normal"]), [0, 0, 1])
    th, z = rng.uniform(0, 2 * math.pi, 2_000), rng.uniform(0, 10, 2_000)
    fit = F.fit_cylinder(np.c_[4 * np.cos(th), 4 * np.sin(th), z])
    assert abs(fit["radius"] - 4) < 1e-9 and fit["rms"] < 1e-9
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    assert abs(F.fit_sphere(7 * d)["radius"] - 7) < 1e-9
    for fn, n in ((F.fit_plane, 2), (F.fit_sphere, 3), (F.fit_cylinder, 5), (F.fit_line, 1)):
        with pytest.raises(ValueError, match="at least"):
            fn(np.zeros((n, 3)))


# --------------------------------------------------------------------------- part frame
def test_part_frame_finds_the_natural_box_in_any_orientation():
    bracket = part_mesh("bracket")   # 70 x 40 x 45 mm L-bracket with a boss: PCA alone tilts its axes
    for seed in (1, 2, 3):
        mesh = o3d.geometry.TriangleMesh(bracket).transform(random_rigid(seed))
        frame = part_frame(mesh)
        np.testing.assert_allclose(frame.dims, [70, 45, 40], atol=1e-6)
        np.testing.assert_allclose(frame.dims_raw, [70, 45, 40], atol=1e-6)
        np.testing.assert_allclose(frame.axes @ frame.axes.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(frame.axes) == pytest.approx(1.0)
        coords = frame.coords(np.asarray(mesh.vertices))
        np.testing.assert_allclose(coords.min(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(coords.max(axis=0), [70, 45, 40], atol=1e-6)
    again = part_frame(mesh)
    np.testing.assert_array_equal(again.axes, frame.axes)           # deterministic
    np.testing.assert_array_equal(again.origin, frame.origin)
    info = frame.info()
    assert set(info) == {"dimensions", "dimensions_raw", "frame"} and set(info["frame"]) == {
        "origin", "length", "width", "height"}


def test_part_frame_ignores_stray_points_but_reports_them_in_raw():
    box = scan_of(box_mesh(), 200_000, noise=0.01, seed=3)
    pts = np.vstack([np.asarray(box.points), [[400.0, -300.0, 200.0], [-250.0, 90.0, 10.0]]])
    frame = part_frame(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
    np.testing.assert_allclose(frame.dims, [50, 30, 20], atol=0.1)   # extents include ~3 sigma of noise per side
    assert frame.dims_raw.max() > 400
    # meshes are trimmed by surface area: a coarse CAD box keeps its exact corners
    w = vertex_weights(box_mesh())
    assert w is not None and w.sum() == pytest.approx((50 * 30 + 50 * 20 + 30 * 20) * 2)


# --------------------------------------------------------------------------- regions
def test_region_validation_of_new_kinds():
    ok = check_region({"end": {"direction": "Length", "side": "max", "length": 10}, "invert": True})
    assert ok == {"end": {"direction": "length", "side": "max", "length": 10.0}, "invert": True}
    assert check_region({"slab": {"direction": "height", "from": 0, "to": 2}})["slab"]["frame"] == "part"
    assert check_region({"slab": {"direction": [0, 0, 2], "from": 0, "to": 2}})["slab"]["frame"] == "world"
    obb = check_region({"obb": {"center": [0, 0, 0], "axes": [[2, 0, 0], [0, 1, 0], [0, 0, 1]], "half": [1, 2, 3]}})
    assert obb["obb"]["axes"][0] == [1.0, 0.0, 0.0]
    cyl = check_region({"cylinder": {"point": [0, 0, 0], "axis": [0, 0, 5], "radius": 1, "half_length": 2}})
    assert cyl["cylinder"]["axis"] == [0.0, 0.0, 1.0]
    assert check_region({"all": True}) == {"all": True}
    bad = [
        ({"end": {"direction": "sideways", "side": "max", "length": 1}}, "must be one of x, y, z"),
        ({"end": {"direction": "x", "side": "top", "length": 1}}, "'min' or 'max'"),
        ({"end": {"direction": "x", "side": "min", "length": 0}}, "> 0"),
        ({"slab": {"direction": "x", "from": 3, "to": 1}}, "'from' must be <="),
        ({"slab": {"direction": "x", "from": 0, "to": 1, "frame": "camera"}}, "'world' or 'part'"),
        ({"obb": {"center": [0, 0, 0], "axes": [[1, 0, 0], [1, 1, 0], [0, 0, 1]], "half": [1, 1, 1]}}, "perpendicular"),
        ({"obb": {"center": [0, 0, 0], "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "half": [1, 0, 1]}}, "> 0"),
        ({"cylinder": {"point": [0, 0, 0], "axis": [0, 0, 0], "radius": 1, "half_length": 1}}, "zero vector"),
        ({"all": False}, "must be true"),
        ({"all": True, "invert": "yes"}, "true or false"),
        ({"all": True, "box": {"min": [0, 0, 0], "max": [1, 1, 1]}}, "exactly one of"),
        ({"cylinder": {"point": [0, 0, 0], "axis": [0, 0, 1], "radius": 1, "half_length": 1}, "r": 2}, "unexpected"),
    ]
    for region, message in bad:
        with pytest.raises(ValueError, match=message):
            check_region(region)
    with pytest.raises(ValueError, match="must be one of x, y, z"):
        validate_ops([{"op": "delete_region", "region": {"end": {"direction": "up", "side": "max", "length": 1}}}],
                     "pointcloud")


def _browser_eval(pts: np.ndarray, resolved: dict) -> np.ndarray:
    """What the viewer does with a resolved region: inside any shape (then invert)."""
    mask = np.zeros(len(pts), bool)
    for s in resolved["shapes"]:
        if s["type"] == "sphere":
            mask |= np.linalg.norm(pts - s["center"], axis=1) <= s["radius"] + 1e-9
        elif s["type"] == "obb":
            local = (pts - np.asarray(s["center"])) @ np.asarray(s["axes"]).T
            mask |= np.all(np.abs(local) <= np.asarray(s["half"]) + 1e-7, axis=1)
        elif s["type"] == "cylinder":
            rel = pts - np.asarray(s["point"])
            t = rel @ np.asarray(s["axis"])
            radial = np.linalg.norm(rel - np.outer(t, s["axis"]), axis=1)
            mask |= (np.abs(t) <= s["half_length"] + 1e-9) & (radial <= s["radius"] + 1e-9)
    return ~mask if resolved["invert"] else mask


def test_region_masks_match_numpy_and_the_resolved_shapes():
    rng = np.random.default_rng(4)
    pts = rng.uniform(-10, 10, (30_000, 3)) * [3.0, 1.5, 1.0]   # elongated along x: length = x
    geom = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    frame = part_frame(geom)
    c = frame.coords(pts)
    x = pts[:, 0]
    rot = np.asarray(o3d.geometry.get_rotation_matrix_from_xyz((0.3, -0.2, 0.9)))
    cases = {
        "box": ({"box": {"min": [-5, -5, -5], "max": [5, 6, 7]}},
                np.all((pts >= [-5, -5, -5]) & (pts <= [5, 6, 7]), axis=1)),
        "spheres": ({"spheres": [[0, 0, 0, 4], [12, 3, 1, 2.5]]},
                    (np.linalg.norm(pts, axis=1) <= 4) | (np.linalg.norm(pts - [12, 3, 1], axis=1) <= 2.5)),
        "obb": ({"obb": {"center": [2, 1, 0], "axes": rot.tolist(), "half": [6, 3, 2]}},
                np.all(np.abs((pts - [2, 1, 0]) @ rot.T) <= [6, 3, 2], axis=1)),
        "cylinder": ({"cylinder": {"point": [1, 0, 0], "axis": [0, 0, 1], "radius": 5, "half_length": 4}},
                     (np.hypot(pts[:, 0] - 1, pts[:, 1]) <= 5) & (np.abs(pts[:, 2]) <= 4)),
        "slab world": ({"slab": {"direction": "x", "from": -3, "to": 8}}, (x >= -3) & (x <= 8)),
        "slab part": ({"slab": {"direction": "length", "from": 10, "to": 20}}, (c[:, 0] >= 10) & (c[:, 0] <= 20)),
        "end part": ({"end": {"direction": "length", "side": "max", "length": 5}}, c[:, 0] >= frame.dims[0] - 5),
        "end world": ({"end": {"direction": "x", "side": "min", "length": 7}},
                      x <= np.percentile(x, 0.05) + 7),
        "all": ({"all": True}, np.ones(len(pts), bool)),
        "invert": ({"cylinder": {"point": [1, 0, 0], "axis": [0, 0, 1], "radius": 5, "half_length": 4},
                    "invert": True}, ~((np.hypot(pts[:, 0] - 1, pts[:, 1]) <= 5) & (np.abs(pts[:, 2]) <= 4))),
    }
    for name, (region, expected) in cases.items():
        mask = region_mask(geom, region, frame)
        assert 0 < expected.sum() and np.array_equal(mask, expected), name
        resolved = resolve_region(geom, region, frame)
        assert np.array_equal(_browser_eval(pts, resolved), expected), name
    assert region_mask(geom, None).all()
    # end regions are open-ended: stray points beyond the part's robust end are included
    assert cases["end part"][1][np.argmax(c[:, 0])]


def test_region_distances_for_new_kinds():
    pts = np.array([[0, 0, 0], [3, 0, 0], [0, 0, 7], [6, 0, 0], [0, 0, 0.5]], float)
    geom = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cyl = {"cylinder": {"point": [0, 0, 0], "axis": [0, 0, 1], "radius": 2, "half_length": 5}}
    np.testing.assert_allclose(region_distance(geom, cyl, 10), [0, 1, 2, 4, 0])
    obb = {"obb": {"center": [0, 0, 0], "axes": np.eye(3).tolist(), "half": [1, 1, 1]}}
    np.testing.assert_allclose(region_distance(geom, obb, 3), [0, 2, np.inf, np.inf, 0])
    slab = {"slab": {"direction": "z", "from": -1, "to": 1}}
    np.testing.assert_allclose(region_distance(geom, slab, 10), [0, 0, 6, 0, 0])
    inverted = region_distance(geom, {**cyl, "invert": True}, 10)
    assert inverted[3] == 0 and inverted[1] == 0 and inverted[0] == pytest.approx(3.0)


def test_region_edit_ops():
    cloud = scan_of(bolt_mesh(), 60_000, seed=5)
    pts = np.asarray(cloud.points)
    head = {"end": {"direction": "length", "side": "max", "length": 38}}
    out, report = apply_edits(cloud, [{"op": "delete_region", "region": head}], QUIET)
    entry = report["ops"][0]
    assert entry["op"] == "delete_region" and entry["selected"] > 0 and "last 38" in entry["region"]
    assert len(out.points) == len(pts) - entry["selected"] and np.asarray(out.points)[:, 2].max() < 70.5
    kept, _ = apply_edits(cloud, [{"op": "keep_region", "region": head}], QUIET)
    assert len(kept.points) == entry["selected"] and np.asarray(kept.points)[:, 2].min() > 69.5
    with pytest.raises(ValueError, match="contains no points"):
        apply_edits(cloud, [{"op": "delete_region", "region": {"box": {"min": [500, 500, 500], "max": [501, 501, 501]}}}],
                    QUIET)

    painted, report = apply_edits(cloud, [{"op": "paint", "rgb": [1, 0, 0], "region": head}], QUIET)
    colors = np.asarray(painted.colors)
    assert report["ops"][0]["painted"] == entry["selected"] and report["ops"][0]["unpainted_color"] == [0.75] * 3
    assert np.all(colors[pts[:, 2] > 71] == [1, 0, 0]) and np.all(colors[pts[:, 2] < 69] == 0.75)
    np.testing.assert_array_equal(np.asarray(painted.points), pts)          # geometry untouched

    mesh = bolt_mesh()
    trimmed, report = apply_edits(mesh, [{"op": "delete_region", "region": {"slab": {"direction": "z", "from": 80,
                                                                                    "to": 200}}}], QUIET)
    assert np.asarray(trimmed.vertices)[:, 2].max() < 80 and len(trimmed.triangles) < len(mesh.triangles)

    # local smoothing limited to the shank's first 20 mm: the head does not move at all
    smoothed, report = apply_edits(cloud, [{"op": "smooth", "region": {"end": {"direction": "length", "side": "min",
                                                                              "length": 20}}, "feather": 1.0}], QUIET)
    moved = np.linalg.norm(np.asarray(smoothed.points) - pts, axis=1)
    assert moved[pts[:, 2] > 25].max() == 0 and moved[pts[:, 2] < 15].max() > 0


# --------------------------------------------------------------------------- summary
def test_summary_describes_a_bolt():
    for geom in (bolt_mesh(), scan_of(bolt_mesh(), 300_000, seed=6, transform=random_rigid(4))):
        s = U.summarize(geom)
        d = s["dimensions"]
        assert d["length"] == pytest.approx(108, abs=0.1) and d["width"] == pytest.approx(38, abs=0.1)
        cyl = s["features"]["cylinders"]
        assert cyl and cyl[0]["axis_name"] == "length"
        assert cyl[0]["diameter"] == pytest.approx(20, abs=0.02) and cyl[0]["length"] == pytest.approx(70, abs=0.5)
        assert any(c["diameter"] == pytest.approx(38, abs=0.03) for c in cyl)
        text = s["description"]
        assert text.startswith("Elongated part 108.")
        assert "the length-max end" in text and "likely a head" in text and "Ø20.0 cylinder" in text
        assert len(s["profile"]["slices"]) == 24 and s["units"] == "mm"
        sizes = [max(x["width"], x["height"]) for x in s["profile"]["slices"]]
        assert sizes[2] == pytest.approx(20, abs=0.3) and sizes[-3] == pytest.approx(38, abs=0.3)


def test_summary_of_bracket_and_plate():
    s = U.summarize(part_mesh("bracket"))
    assert [s["dimensions"][k] for k in ("length", "width", "height")] == pytest.approx([70, 45, 40], abs=1e-6)
    planes = s["features"]["planes"]
    assert len(planes) >= 4 and all(p["flatness"] < 1e-6 for p in planes)
    assert s["description"].startswith("Compact part 70.0 × 45.0 × 40.0 mm")
    plate = U.summarize(scan_of(box_mesh(120, 80, 5), 250_000, seed=7))
    assert plate["description"].startswith("Flat part (plate-like, 5.")
    labels = [p["label"] for p in plate["features"]["planes"]]
    assert labels[0].startswith("height-") and labels[1].startswith("height-")
    assert plate["features"]["planes"][0]["area_mm2"] == pytest.approx(9600, rel=0.05)


def test_summary_cache_and_speed(tmp_path):
    ws = Workspace(tmp_path / "ws")
    meta = ws.add_geometry(scan_of(bolt_mesh(), 1_000_000, seed=8), "bolt", "import")
    started = time.perf_counter()
    first = U.get_summary(ws, meta["id"])
    elapsed = time.perf_counter() - started
    assert elapsed < 20, f"summary of 1 M points took {elapsed:.1f} s"   # ~2-4 s on an idle machine
    assert (ws.asset_dir(meta["id"]) / "summary.json").exists() and first["asset_id"] == meta["id"]
    assert first["dimensions"] == ws.get(meta["id"])["part"]["dimensions"]  # same frame as the asset meta
    assert U.get_summary(ws, meta["id"])["computed_at"] == first["computed_at"]
    assert U.load_cached_summary(ws, meta["id"]) is not None
    time.sleep(0.05)
    save(scan_of(cylinder_mesh(5, 30), 20_000, seed=9), ws.asset_dir(meta["id"]) / "data.ply")
    assert U.load_cached_summary(ws, meta["id"]) is None                   # data.ply changed: stale
    assert U.get_summary(ws, meta["id"])["dimensions"]["length"] == pytest.approx(30, abs=0.2)


# --------------------------------------------------------------------------- measurements
def test_measurements_on_known_shapes():
    box = scan_of(box_mesh(), 300_000, noise=0.02, seed=10, transform=random_rigid(6))
    frame = part_frame(box)
    ext = U.measure_extent(box, "length", frame=frame)
    assert ext["length"] > 50.1 and ext["robust_length"] == pytest.approx(50.1, abs=0.05)   # noise widens extents
    assert ext["points_used"] == 300_000 and np.linalg.norm(np.subtract(ext["b"], ext["a"])) == pytest.approx(
        ext["length"])
    cal = U.measure_caliper(box, "length", frame=frame)
    assert cal["distance"] == pytest.approx(50.0, abs=0.005) and cal["parallelism_deg"] < 0.1
    assert cal["face_a"]["kind"] == cal["face_b"]["kind"] == "plane" and not cal["warnings"]
    assert U.measure_caliper(box, "height", frame=frame)["distance"] == pytest.approx(20.0, abs=0.005)

    top = {"end": {"direction": "height", "side": "max", "length": 0.1}}
    side = {"end": {"direction": "length", "side": "max", "length": 0.1}}
    plane = U.measure_plane(box, top, frame)
    assert abs(abs(np.dot(plane["normal"], frame.axis("height"))) - 1) < 1e-4 and plane["flatness"] < 0.2
    angle = U.measure_angle(box, top, side, frame)
    assert angle["angle_deg"] == pytest.approx(90, abs=0.05) and angle["fit_a"]["type"] == "plane"

    bolt = scan_of(bolt_mesh(), 300_000, seed=11)
    bframe = part_frame(bolt)
    dia = U.measure_diameter(bolt, {"slab": {"direction": "length", "from": 10, "to": 60}}, frame=bframe)
    assert dia["kind"] == "cylinder" and dia["diameter"] == pytest.approx(20, abs=0.005) and dia["coverage_deg"] == 360
    ring = U.measure_diameter(bolt, {"slab": {"direction": "length", "from": 30, "to": 30.4}}, "length", bframe)
    assert ring["kind"] == "circle" and ring["diameter"] == pytest.approx(20, abs=0.02)
    head = U.measure_diameter(bolt, {"end": {"direction": "length", "side": "max", "length": 30}}, "length", bframe)
    assert head["diameter"] == pytest.approx(38, abs=0.01) and head["warnings"]   # the top face is in the region

    d = np.random.default_rng(12).normal(size=(40_000, 3))
    ball = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        [3, -2, 7] + 12.5 * d / np.linalg.norm(d, axis=1)[:, None]))
    sph = U.measure_sphere(ball, {"all": True})
    assert sph["diameter"] == pytest.approx(25, abs=1e-6) and np.allclose(sph["center"], [3, -2, 7], atol=1e-6)

    section = U.measure_section(bolt_mesh(), {"direction": "length", "at": 30})
    assert len(section["polylines"]) == 1 and section["closed"] == [True]
    assert section["width"] == pytest.approx(20, abs=0.01) and section["height"] == pytest.approx(20, abs=0.01)
    cloud_section = U.measure_section(bolt, {"point": [0, 0, 90], "normal": [0, 0, 1]})
    assert cloud_section["width"] == pytest.approx(38, abs=0.15) and cloud_section["points_used"] > 100
    with pytest.raises(ValueError, match="does not cut"):
        U.measure_section(bolt_mesh(), {"point": [0, 0, 500], "normal": [0, 0, 1]})
    items = U.overlay_items("caliper", cal)
    assert items[0]["unit"] == "mm" and len(items[0]["a"]) == 3 and items[0]["value"] == pytest.approx(50, abs=0.005)


# --------------------------------------------------------------------------- endpoints
def _wait(client, job, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        current = {j["id"]: j for j in client.get("/api/jobs").json()}[job["id"]]
        if current["status"] in ("done", "failed", "cancelled"):
            assert current["status"] == "done", "\n".join(current["logs"][-30:]) + f"\n{current['error']}"
            return current
        time.sleep(0.3)
    raise TimeoutError(job["title"])


def test_understand_endpoints(tmp_path):
    app = create_app(tmp_path / "ws")
    ws = app.state.workspace
    bolt = ws.add_geometry(scan_of(bolt_mesh(), 200_000, seed=13), "bolt", "import")["id"]
    boxed = ws.add_geometry(scan_of(box_mesh(), 150_000, seed=14), "box", "import")["id"]
    with TestClient(app) as client:
        s = client.get(f"/api/assets/{bolt}/summary").json()
        assert "Ø20.0 cylinder" in s["description"] and set(s["part_frame"]) == {"origin", "length", "width", "height"}
        assert client.get("/api/assets/0123456789ab/summary").status_code == 404

        end = {"end": {"direction": "length", "side": "max", "length": 38}}
        res = client.post("/api/regions/resolve", json={"asset_id": bolt, "region": end}).json()
        assert 0 < res["count"] < res["total"] == 200_000 and res["resolved"]["shapes"][0]["type"] == "obb"
        assert res["bbox"]["min"][2] > 69 and res["description"].startswith("last 38")
        bad = client.post("/api/regions/resolve", json={"asset_id": bolt, "region": {"end": {"direction": "up"}}})
        assert bad.status_code == 400

        ext = client.post("/api/measure/extent", json={"asset_id": bolt, "direction": "length"}).json()
        assert ext["length"] == pytest.approx(108.1, abs=0.15) and ext["frame"] == "part" and len(ext["a"]) == 3
        cal = client.post("/api/measure/caliper", json={"asset_id": boxed, "direction": "x"}).json()
        assert cal["distance"] == pytest.approx(50, abs=0.005) and cal["face_a"]["kind"] == "plane"
        dia = client.post("/api/measure/diameter", json={"asset_id": bolt, "axis_hint": "length",
                                                         "region": {"slab": {"direction": "z", "from": 5, "to": 65}}})
        assert dia.json()["diameter"] == pytest.approx(20, abs=0.005)
        top = {"end": {"direction": "z", "side": "max", "length": 0.1}}
        side = {"end": {"direction": "x", "side": "max", "length": 0.1}}
        plane = client.post("/api/measure/plane", json={"asset_id": boxed, "region": top}).json()
        assert plane["flatness"] < 0.2 and plane["points_used"] > 100
        angle = client.post("/api/measure/angle", json={"asset_id": boxed, "region_a": top, "region_b": side}).json()
        assert angle["angle_deg"] == pytest.approx(90, abs=0.05)
        section = client.post("/api/measure/section", json={"asset_id": boxed, "plane": {"direction": "x", "at": 25}})
        assert sorted([section.json()["width"], section.json()["height"]]) == pytest.approx([20, 30], abs=0.2)
        sphere = client.post("/api/measure/sphere", json={"asset_id": bolt, "region": {"box": {
            "min": [900, 900, 900], "max": [901, 901, 901]}}})
        assert sphere.status_code == 400 and "no points" in sphere.json()["detail"]
        assert client.post("/api/measure/extent", json={"asset_id": bolt, "direction": "up"}).status_code == 400
        assert client.post("/api/measure/plane", json={"asset_id": "0123456789ab", "region": top}).status_code == 404

        job = client.post("/api/assets/summaries", json={"asset_ids": [boxed]}).json()
        done = _wait(client, job)
        assert "Compact part 50." in done["output"]["summaries"][boxed]
        assert client.post("/api/assets/summaries", json={"asset_ids": ["0123456789ab"]}).status_code == 404
