"""Geometry editing: exact results for every op, validation, measuring and the edit/export/measure API."""
import math
import time

import numpy as np
import open3d as o3d
import pytest
from fastapi.testclient import TestClient

from cloudclean.edit import OP_CATALOGUE, Snapper, apply_edits, measure, points_in_polygon, validate_ops
from cloudclean.io import load, save
from cloudclean.web.server import create_app
from tests.synthetic import random_rigid, three_view_matrix


# --------------------------------------------------------------------------- fixtures / helpers
def grid_cloud(n=41, span=4.0, z=0.0, colors=True) -> o3d.geometry.PointCloud:
    """Regular grid in a z plane; coordinates chosen so no point lies on the test boundaries."""
    g = np.linspace(-span / 2, span / 2, n) + 0.0137
    xx, yy = np.meshgrid(g, g)
    pts = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, z)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(np.random.default_rng(0).random((len(pts), 3)))
    pcd.normals = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (len(pts), 1)))
    return pcd


def random_cloud(n=20000, seed=0) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(seed)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(-5, 5, (n, 3))))
    pcd.colors = o3d.utility.Vector3dVector(rng.random((n, 3)))
    return pcd


def colored_sphere(radius=1.0, resolution=30) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector((v - v.min(0)) / (v.max(0) - v.min(0)))
    return mesh


def signed_volume(mesh) -> float:
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    return float(np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6)


def outward_fraction(mesh) -> float:
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    centre = v.mean(0)
    return float(np.mean(np.einsum("ij,ij->i", n, (a + b + c) / 3 - centre) > 0))


def perspective_vp(eye, target, fov=50.0, aspect=1.5, near=0.1, far=100.0) -> list[float]:
    """three.js projectionMatrix * matrixWorldInverse as column-major elements."""
    view = np.asarray(three_view_matrix(eye, target, up=(0, 1, 0))).reshape(4, 4).T
    f = 1 / math.tan(math.radians(fov) / 2)
    proj = np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]])
    return (proj @ view).T.reshape(-1).tolist()


# --------------------------------------------------------------------------- validation
def test_catalogue_and_validation():
    for name, entry in OP_CATALOGUE.items():
        assert entry["description"] and entry["applies_to"]
        for arg in entry["args"].values():
            assert {"type", "default", "description"} <= set(arg)

    ops = validate_ops([{"op": "rotate", "axis": "z", "degrees": 90}], "pointcloud")
    assert ops == [{"op": "rotate", "axis": "z", "degrees": 90.0, "center": "centroid"}]
    assert validate_ops([{"op": "translate", "args": {"offset": [1, 2, 3]}}], "mesh")[0]["offset"] == [1.0, 2.0, 3.0]
    # the kind changes along the sequence
    validate_ops([{"op": "to_pointcloud"}, {"op": "downsample", "voxel": 0.1}], "mesh")

    bad = [
        ([{"op": "explode"}], "unknown op"),
        ([{"op": "translate"}], "missing required argument 'offset'"),
        ([{"op": "translate", "offset": [1, 2]}], "three numbers"),
        ([{"op": "rotate", "axis": "w", "degrees": 1}], '"x", "y", "z"'),
        ([{"op": "rotate", "axis": "x", "degrees": "90"}], "must be a number"),
        ([{"op": "crop_box", "min": [0, 0, 0], "max": [1, 1, 1], "invert": "yes"}], "true or false"),
        ([{"op": "crop_box", "min": [2, 0, 0], "max": [1, 1, 1]}], "<="),
        ([{"op": "simplify", "ratio": 0.5}], "only works on meshes"),
        ([{"op": "downsample", "voxel": 1}], "only works on point clouds"),
        ([{"op": "delete_sphere", "center": [0, 0, 0], "radius": 1, "colour": 2}], "unknown argument"),
        ([{"op": "scale", "factor": -1}], "> 0"),
        ([{"op": "transform", "matrix": [[1, 0, 0, 0]] * 4}], "last row"),
        ([], "at least one"),
    ]
    kinds = {"simplify": "pointcloud", "downsample": "mesh"}
    for ops, message in bad:
        kind = kinds.get(ops[0]["op"], "pointcloud") if ops else "pointcloud"
        with pytest.raises(ValueError, match=message.replace("(", r"\(")):
            validate_ops(ops, kind)


# --------------------------------------------------------------------------- transforms
def test_rotate_translate_scale_exact_and_input_untouched():
    pts = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.normals = o3d.utility.Vector3dVector(np.eye(3))
    out, report = apply_edits(pcd, [{"op": "rotate", "axis": "z", "degrees": 90, "center": "origin"}], log=lambda m: None)
    np.testing.assert_array_equal(np.asarray(out.points), [[0, 1, 0], [-2, 0, 0], [0, 0, 3]])
    np.testing.assert_array_equal(np.asarray(out.normals), [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(np.asarray(pcd.points), pts)  # input not modified
    np.testing.assert_array_equal(report["transform"], [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    ops = [{"op": "translate", "offset": [1, 2, 3]}, {"op": "scale", "factor": 2, "center": "origin"},
           {"op": "rotate", "axis": [0, 0, 1], "degrees": 30, "center": [1, 1, 0]}]
    out, report = apply_edits(pcd, ops, log=lambda m: None)
    T = np.asarray(report["transform"])
    expected = np.c_[pts, np.ones(3)] @ T.T
    np.testing.assert_allclose(np.asarray(out.points), expected[:, :3], atol=1e-12)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    p = (np.array([1.0, 0, 0]) + [1, 2, 3]) * 2 - [1, 1, 0]
    np.testing.assert_allclose(np.asarray(out.points)[0], [c * p[0] - s * p[1] + 1, s * p[0] + c * p[1] + 1, p[2]],
                               atol=1e-12)
    assert [e["op"] for e in report["ops"]] == ["translate", "scale", "rotate"]
    assert report["ops"][0]["before"] == {"points": 3} and report["ops"][0]["after"] == {"points": 3}

    none_moved = apply_edits(pcd, [{"op": "paint", "rgb": [1, 0, 0]}], log=lambda m: None)[1]
    assert none_moved["transform"] is None


def test_transform_matrix_and_center_modes():
    mesh = colored_sphere()
    m = random_rigid(3)
    out, report = apply_edits(mesh, [{"op": "transform", "matrix": m.tolist()}], log=lambda m: None)
    v = np.asarray(mesh.vertices)
    np.testing.assert_allclose(np.asarray(out.vertices), v @ m[:3, :3].T + m[:3, 3], atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.vertex_normals), np.asarray(mesh.vertex_normals) @ m[:3, :3].T, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(out.vertex_colors), np.asarray(mesh.vertex_colors))
    np.testing.assert_allclose(report["transform"], m)

    shifted, _ = apply_edits(mesh, [{"op": "translate", "offset": [10, -4, 7]}], log=lambda m: None)
    c, _ = apply_edits(shifted, [{"op": "center", "mode": "bbox"}], log=lambda m: None)
    v = np.asarray(c.vertices)
    np.testing.assert_allclose((v.min(0) + v.max(0)) / 2, 0, atol=1e-12)
    c, _ = apply_edits(shifted, [{"op": "center", "mode": "centroid"}], log=lambda m: None)
    np.testing.assert_allclose(np.asarray(c.vertices).mean(0), 0, atol=1e-12)
    c, _ = apply_edits(shifted, [{"op": "center", "mode": "bbox_bottom", "up": "y"}], log=lambda m: None)
    v = np.asarray(c.vertices)
    assert abs(v[:, 1].min()) < 1e-12 and abs(v[:, 0].min() + v[:, 0].max()) < 1e-12


def test_mirror_keeps_outward_normals():
    mesh = colored_sphere()
    mesh.translate([3.0, 0, 0])
    assert outward_fraction(mesh) == 1.0
    out, report = apply_edits(mesh, [{"op": "mirror", "axis": "x"}], log=lambda m: None)
    v0, v1 = np.asarray(mesh.vertices), np.asarray(out.vertices)
    np.testing.assert_allclose(v1[:, 0], 6.0 - v0[:, 0], atol=1e-12)  # mirrored about the bbox centre x=3
    np.testing.assert_array_equal(v1[:, 1:], v0[:, 1:])
    assert outward_fraction(out) == 1.0 and signed_volume(out) > 0
    n0, n1 = np.asarray(mesh.vertex_normals), np.asarray(out.vertex_normals)
    np.testing.assert_allclose(n1, n0 * [-1, 1, 1], atol=1e-12)
    assert np.linalg.det(np.asarray(report["transform"])[:3, :3]) == pytest.approx(-1)

    flipped, _ = apply_edits(mesh, [{"op": "flip_normals"}], log=lambda m: None)
    assert outward_fraction(flipped) == 0.0 and signed_volume(flipped) < 0


def test_align_principal():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(20000, 3)) * [10.0, 4.0, 1.0]
    m = random_rigid(5)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts @ m[:3, :3].T + m[:3, 3]))
    out, report = apply_edits(pcd, [{"op": "align_principal"}], log=lambda m: None)
    q = np.asarray(out.points)
    std = q.std(axis=0)
    assert std[0] > std[1] > std[2]
    np.testing.assert_allclose(std, pts.std(axis=0), rtol=0.02)
    np.testing.assert_allclose(q.mean(0), np.asarray(pcd.points).mean(0), atol=1e-9)  # rotated about centroid
    assert np.linalg.det(np.asarray(report["transform"])[:3, :3]) == pytest.approx(1)


@pytest.mark.parametrize("kind", ["mesh", "pointcloud"])
def test_align_floor_tilted_box(kind):
    box = o3d.geometry.TriangleMesh.create_box(4.0, 3.0, 1.0)
    T = random_rigid(11)
    box.transform(T)
    o3d.utility.random.seed(0)
    geom = box if kind == "mesh" else box.sample_points_uniformly(60000)
    out, report = apply_edits(geom, [{"op": "align_floor", "up": "z"}], log=lambda m: None)
    pts = np.asarray(out.vertices if kind == "mesh" else out.points)
    size = 4.0
    assert abs(pts[:, 2].min()) < 1e-3 * size  # bottom on the floor
    assert abs(pts[:, 2].max() - 1.0) < 1e-3 * size  # a 4 x 3 face is the floor, object above
    r = np.asarray(report["transform"])[:3, :3]
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)  # rigid: nothing rescaled
    assert np.linalg.det(r) == pytest.approx(1)
    if kind == "mesh":  # the 3 x 4 face lies flat: its diagonal is horizontal
        bottom = pts[np.abs(pts[:, 2]) < 1e-6]
        assert len(bottom) == 4 and np.max(np.linalg.norm(bottom[:, None, :2] - bottom[None, :, :2], axis=2)) == pytest.approx(5.0)

    out_y, _ = apply_edits(geom, [{"op": "align_floor", "up": "y"}], log=lambda m: None)
    pts = np.asarray(out_y.vertices if kind == "mesh" else out_y.points)
    assert abs(pts[:, 1].min()) < 1e-3 * size and abs(pts[:, 1].max() - 1.0) < 1e-3 * size


# --------------------------------------------------------------------------- selections
def test_crop_cut_sphere_counts_match_numpy():
    pcd = random_cloud()
    pts = np.asarray(pcd.points)
    lo, hi = np.array([-1.0, -2.0, -3.0]), np.array([2.0, 1.5, 4.0])
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    log = lambda m: None

    out, report = apply_edits(pcd, [{"op": "crop_box", "min": lo.tolist(), "max": hi.tolist()}], log)
    np.testing.assert_array_equal(np.asarray(out.points), pts[inside])  # not moved, order kept
    np.testing.assert_array_equal(np.asarray(out.colors), np.asarray(pcd.colors)[inside])
    assert report["ops"][0]["before"] == {"points": 20000} and report["ops"][0]["after"] == {"points": int(inside.sum())}
    out, _ = apply_edits(pcd, [{"op": "crop_box", "min": lo.tolist(), "max": hi.tolist(), "invert": True}], log)
    np.testing.assert_array_equal(np.asarray(out.points), pts[~inside])

    normal, point = np.array([1.0, 2.0, -0.5]), np.array([0.5, 0.0, 1.0])
    positive = (pts - point) @ normal >= 0
    out, _ = apply_edits(pcd, [{"op": "cut_plane", "point": point.tolist(), "normal": normal.tolist()}], log)
    np.testing.assert_array_equal(np.asarray(out.points), pts[positive])
    out, _ = apply_edits(pcd, [{"op": "cut_plane", "point": point.tolist(), "normal": normal.tolist(),
                                "keep": "negative"}], log)
    assert len(out.points) == int(((pts - point) @ normal <= 0).sum())

    far = np.linalg.norm(pts - [1, 1, 1], axis=1) > 2.5
    out, _ = apply_edits(pcd, [{"op": "delete_sphere", "center": [1, 1, 1], "radius": 2.5}], log)
    np.testing.assert_array_equal(np.asarray(out.points), pts[far])


def test_mesh_crop_drops_triangles_and_keeps_colours():
    mesh = colored_sphere()
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    keep_v = v[:, 2] <= 0.3
    keep_t = keep_v[t].all(axis=1)
    out, report = apply_edits(mesh, [{"op": "crop_box", "min": [-2, -2, -2], "max": [2, 2, 0.3]}], lambda m: None)
    assert len(out.triangles) == int(keep_t.sum())
    used = np.unique(t[keep_t])
    np.testing.assert_array_equal(np.asarray(out.vertices), v[used])
    np.testing.assert_array_equal(np.asarray(out.vertex_colors), np.asarray(mesh.vertex_colors)[used])
    assert report["ops"][0]["after"] == {"vertices": len(used), "triangles": int(keep_t.sum())}

    with pytest.raises(ValueError, match="remove everything"):
        apply_edits(mesh, [{"op": "delete_sphere", "center": [0, 0, 0], "radius": 5}], lambda m: None)


def test_points_in_polygon_even_odd_many_vertices():
    rng = np.random.default_rng(2)
    xy = rng.uniform(-1, 1, (50000, 2))
    # star with 3000 vertices: compare with an exact polar test
    k = 3000
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    rad = 0.6 + 0.3 * np.cos(7 * ang)
    poly = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    inside = points_in_polygon(xy, poly)
    r = np.hypot(xy[:, 0], xy[:, 1])
    a = np.arctan2(xy[:, 1], xy[:, 0])
    exact = r < 0.6 + 0.3 * np.cos(7 * a)
    assert np.mean(inside == exact) > 0.998  # differs only within chord error of the polygon
    # square with a square hole (even-odd): outer then inner ring
    ring = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5],
            [-0.5, -0.5]]
    test = np.array([[0.0, 0.0], [0.75, 0.0], [2.0, 0.0]])
    np.testing.assert_array_equal(points_in_polygon(test, ring), [False, True, False])


def test_select_screen_cloud_matches_camera_math():
    pcd = grid_cloud(n=81, span=6.0)
    pts = np.asarray(pcd.points)
    fov, aspect = 50.0, 1.5
    vp = perspective_vp((0, 0, 10), (0, 0, 0), fov, aspect)
    polygon = [[-0.4, -0.3], [0.5, -0.3], [0.5, 0.2], [-0.4, 0.2]]
    f = 1 / math.tan(math.radians(fov) / 2)
    ndc_x, ndc_y = f / aspect * pts[:, 0] / 10, f * pts[:, 1] / 10
    expected = (ndc_x > -0.4) & (ndc_x < 0.5) & (ndc_y > -0.3) & (ndc_y < 0.2)
    assert 50 < expected.sum() < len(pts)
    log = lambda m: None

    out, report = apply_edits(pcd, [{"op": "select_screen", "view_projection": vp, "polygon": polygon,
                                     "mode": "keep"}], log)
    np.testing.assert_array_equal(np.asarray(out.points), pts[expected])
    assert report["ops"][0]["selected"] == int(expected.sum())
    out, _ = apply_edits(pcd, [{"op": "select_screen", "view_projection": vp, "polygon": polygon}], log)
    np.testing.assert_array_equal(np.asarray(out.points), pts[~expected])

    # points behind the camera never get selected
    behind, _ = apply_edits(pcd, [{"op": "translate", "offset": [0, 0, 20]}], log)
    with pytest.raises(ValueError, match="remove everything"):
        apply_edits(behind, [{"op": "select_screen", "view_projection": vp, "polygon": polygon, "mode": "keep"}], log)


def test_select_screen_visible_only():
    log = lambda m: None
    vp = perspective_vp((0, 0, 10), (0, 0, 0), 50.0, 1.0)
    whole_screen = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    front = grid_cloud(n=121, span=6.0, z=0.0)
    back = grid_cloud(n=61, span=3.0, z=-2.0)  # entirely hidden behind the front sheet
    both = front + back
    n_front = len(front.points)
    out, _ = apply_edits(both, [{"op": "select_screen", "view_projection": vp, "polygon": whole_screen,
                                 "mode": "keep", "visible_only": True}], log)
    z = np.asarray(out.points)[:, 2]
    assert (z == 0).sum() >= 0.99 * n_front and (z < -1).sum() == 0
    out, _ = apply_edits(both, [{"op": "select_screen", "view_projection": vp, "polygon": whole_screen,
                                 "mode": "keep"}], log)
    assert len(out.points) == len(both.points)

    sphere = colored_sphere(radius=1.0, resolution=40)
    out, _ = apply_edits(sphere, [{"op": "select_screen", "view_projection": vp, "polygon": whole_screen,
                                   "mode": "keep", "visible_only": True}], log)
    v = np.asarray(out.vertices)
    assert v[:, 2].min() > -0.05  # nothing from the far side
    all_v = np.asarray(sphere.vertices)
    assert len(v) >= 0.95 * np.sum(all_v[:, 2] > 0.15)
    assert out.has_vertex_colors()

    # mesh delete: a triangle goes only when all of its vertices are selected
    polygon = [[-0.05, -1], [1, -1], [1, 1], [-0.05, 1]]
    out, report = apply_edits(sphere, [{"op": "select_screen", "view_projection": vp, "polygon": polygon}], log)
    t = np.asarray(sphere.triangles)
    sel = np.zeros(len(all_v), bool)
    ndc_x = (1 / math.tan(math.radians(25))) * all_v[:, 0] / (10 - all_v[:, 2])
    sel = ndc_x > -0.05
    assert len(out.triangles) == int((~sel[t].all(axis=1)).sum())
    assert report["ops"][0]["selected_triangles"] == int(sel[t].all(axis=1).sum())


# --------------------------------------------------------------------------- cloud / mesh processing
def test_cloud_processing_ops():
    log = lambda m: None
    pcd = random_cloud(5000)
    out, _ = apply_edits(pcd, [{"op": "downsample", "voxel": 2.0}], log)
    assert 0 < len(out.points) <= 6 ** 3 and out.has_colors()

    rng = np.random.default_rng(3)
    base = grid_cloud(n=60, span=3.0)
    outliers = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        np.c_[rng.uniform(-1.5, 1.5, (20, 2)), np.full(20, 5.0)]))
    out, _ = apply_edits(base + outliers, [{"op": "remove_outliers", "neighbors": 16, "std_ratio": 2.0}], log)
    assert np.asarray(out.points)[:, 2].max() < 1 and len(out.points) >= 0.97 * len(base.points)

    small = grid_cloud(n=8, span=0.3)
    small.translate([10, 0, 0])
    out, report = apply_edits(base + small, [{"op": "remove_small_components", "min_ratio": 0.1}], log)
    assert len(out.points) == len(base.points) and report["ops"][0]["removed_components"] == 1

    out, _ = apply_edits(base, [{"op": "recompute_normals"}, {"op": "flip_normals"}, {"op": "paint", "rgb": [0, 1, 0]}], log)
    assert np.allclose(np.abs(np.asarray(out.normals)[:, 2]), 1, atol=1e-6)
    assert np.all(np.asarray(out.colors) == [0, 1, 0])


def test_mesh_processing_ops_keep_colours():
    log = lambda m: None
    sphere = colored_sphere(resolution=30)
    n_tri = len(sphere.triangles)

    out, _ = apply_edits(sphere, [{"op": "simplify", "ratio": 0.25}], log)
    assert len(out.triangles) <= 0.26 * n_tri and out.has_vertex_colors()
    out, _ = apply_edits(sphere, [{"op": "simplify", "target_triangles": 500}], log)
    assert len(out.triangles) <= 500
    for method in ("taubin", "laplacian"):
        out, _ = apply_edits(sphere, [{"op": "smooth", "iterations": 3, "method": method}], log)
        assert len(out.triangles) == n_tri and out.has_vertex_colors()
    out, _ = apply_edits(sphere, [{"op": "subdivide", "iterations": 1}], log)
    assert len(out.triangles) == 4 * n_tri and out.has_vertex_colors()

    dup = sphere + sphere  # duplicated vertices and triangles
    out, report = apply_edits(dup, [{"op": "repair"}], log)
    assert len(out.triangles) == n_tri and report["ops"][0]["watertight"]

    big, little = colored_sphere(2.0), colored_sphere(0.1)
    little.translate([10, 0, 0])
    out, report = apply_edits(big + little, [{"op": "remove_small_components", "min_ratio": 0.05}], log)
    assert len(out.triangles) == len(big.triangles) and out.has_vertex_colors()

    cloud, _ = apply_edits(sphere, [{"op": "to_pointcloud"}], log)
    np.testing.assert_array_equal(np.asarray(cloud.points), np.asarray(sphere.vertices))
    assert cloud.has_colors() and cloud.has_normals()
    cloud, _ = apply_edits(sphere, [{"op": "to_pointcloud", "samples": 3000}, {"op": "downsample", "voxel": 0.5}], log)
    assert 0 < len(cloud.points) < 3000

    out, _ = apply_edits(sphere, [{"op": "recompute_normals"}, {"op": "paint", "rgb": [0.2, 0.4, 0.6]}], log)
    assert np.allclose(np.asarray(out.vertex_colors), [0.2, 0.4, 0.6])


def test_fill_holes_makes_sphere_watertight():
    sphere = colored_sphere(resolution=30)
    v = np.asarray(sphere.vertices)
    t = np.asarray(sphere.triangles)
    open_mesh = o3d.geometry.TriangleMesh(sphere)
    open_mesh.triangles = o3d.utility.Vector3iVector(t[~(v[t][:, :, 2] > 0.7).all(axis=1)])
    assert not open_mesh.is_watertight()

    small, _ = apply_edits(open_mesh, [{"op": "fill_holes", "max_hole_size": 0.05}], lambda m: None)
    assert not small.is_watertight()  # the cap hole (radius ~0.7) is larger than the limit
    out, report = apply_edits(open_mesh, [{"op": "fill_holes"}], lambda m: None)
    assert out.is_watertight()
    assert report["ops"][0]["added_triangles"] > 0
    np.testing.assert_array_equal(np.asarray(out.vertices), np.asarray(open_mesh.vertices))  # nothing moved
    np.testing.assert_array_equal(np.asarray(out.vertex_colors), np.asarray(open_mesh.vertex_colors))


def test_measure_snaps_exactly():
    pcd = grid_cloud(n=11, span=2.0, colors=False)
    pts = np.asarray(pcd.points)
    res = measure(Snapper(pcd), [pts[0] + [0.01, 0.01, 0.3], pts[10] + [0, 0, -0.2], pts[120]])
    np.testing.assert_array_equal(res["points"], pts[[0, 10, 120]])
    seg = np.linalg.norm(np.diff(pts[[0, 10, 120]], axis=0), axis=1)
    np.testing.assert_allclose(res["segments"], seg)
    assert res["total"] == pytest.approx(seg.sum())
    assert res["angle_deg"] == pytest.approx(90.0)

    box = o3d.geometry.TriangleMesh.create_box(2.0, 2.0, 2.0)
    box.translate([100.123, -50.0, 7.0])  # float32 ray casting must not cost precision
    res = measure(Snapper(box), [[101.3777, -49.1, 12.0], [100.5, -48.0, 8.0]])
    np.testing.assert_allclose(res["points"][0], [101.3777, -49.1, 9.0], atol=1e-9)
    np.testing.assert_allclose(res["points"][1], [100.5, -48.0, 8.0], atol=1e-9)
    assert res["segments"][0] == pytest.approx(math.dist(res["points"][0], res["points"][1]))
    assert "angle_deg" not in res


# --------------------------------------------------------------------------- API
def _wait(client, job, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        current = {j["id"]: j for j in client.get("/api/jobs").json()}[job["id"]]
        if current["status"] in ("done", "failed", "cancelled"):
            assert current["status"] == "done", "\n".join(current["logs"][-40:]) + f"\n{current['error']}"
            return current
        time.sleep(0.3)
    raise TimeoutError(job["title"])


def test_edit_api(tmp_path):
    mesh = colored_sphere(resolution=20)
    save(mesh, tmp_path / "ball.ply")
    with TestClient(create_app(tmp_path / "ws")) as client:
        ops = client.get("/api/edit/ops").json()
        assert set(ops) == set(OP_CATALOGUE) and "args" in ops["crop_box"]

        job = client.post("/api/import-paths", json={"paths": [str(tmp_path / "ball.ply")]}).json()["jobs"][0]
        asset = _wait(client, job)["result"][0]

        bad = client.post("/api/edit", json={"asset_id": asset, "ops": [{"op": "downsample", "voxel": 1}]})
        assert bad.status_code == 400 and "point clouds" in bad.json()["detail"]
        assert client.post("/api/edit", json={"asset_id": asset, "ops": [{"op": "nope"}]}).status_code == 400
        assert client.post("/api/edit", json={"asset_id": "0123456789ab", "ops": [{"op": "repair"}]}).status_code == 404

        edit_job = client.post("/api/edit", json={"asset_id": asset, "ops": [
            {"op": "rotate", "axis": "z", "degrees": 90, "center": "origin"},
            {"op": "crop_box", "min": [-2, -2, -2], "max": [2, 2, 0.5]}]}).json()
        done = _wait(client, edit_job)
        assert done["progress"]["fraction"] == 1.0
        new_id = done["result"][0]
        info = client.get(f"/api/assets/{new_id}").json()
        assert info["name"] == "ball · edit" and info["operation"] == "edit" and info["parents"] == [asset]
        assert info["params"]["ops"][0]["center"] == "origin"
        assert info["report"]["transform"][0][:2] == [0.0, -1.0]
        assert info["stats"]["bbox_max"][2] <= 0.5

        named = _wait(client, client.post("/api/edit", json={"asset_id": asset, "name": "cloud",
                                                             "ops": [{"op": "to_pointcloud"}]}).json())
        cloud_id = named["result"][0]
        assert client.get(f"/api/assets/{cloud_id}").json()["kind"] == "pointcloud"

        out_dir = tmp_path / "exports" / "nested"
        res = client.post("/api/export", json={"asset_id": new_id, "format": "stl", "folder": str(out_dir),
                                               "filename": "part.stl"})
        assert res.status_code == 200, res.text
        path = res.json()["path"]
        assert path.endswith("part.stl") and len(load(path).triangles) == len(
            load(tmp_path / "ws" / "assets" / new_id / "data.ply").triangles)
        again = client.post("/api/export", json={"asset_id": new_id, "format": "stl", "folder": str(out_dir),
                                                 "filename": "part"})
        assert again.status_code == 409
        assert client.post("/api/export", json={"asset_id": new_id, "format": "stl", "folder": str(out_dir),
                                                "filename": "part", "overwrite": True}).status_code == 200
        assert client.post("/api/export", json={"asset_id": cloud_id, "format": "stl",
                                                "folder": str(out_dir)}).status_code == 400
        assert client.post("/api/export", json={"asset_id": cloud_id, "format": "xyz",
                                                "folder": "relative/folder"}).status_code == 400
        res = client.post("/api/export", json={"asset_id": cloud_id, "format": "xyz", "folder": str(out_dir)})
        assert res.status_code == 200 and res.json()["path"].endswith("cloud.xyz")

        m = client.post("/api/measure", json={"asset_id": asset, "points": [[0, 0, 3], [0, 0, -3]]}).json()
        np.testing.assert_allclose(m["points"], [[0, 0, 1], [0, 0, -1]], atol=1e-9)
        assert m["total"] == pytest.approx(2.0)
        m = client.post("/api/measure", json={"asset_id": cloud_id, "points": [[3, 0, 0], [0, 3, 0], [0, 0, 3]]}).json()
        assert m["angle_deg"] == pytest.approx(60.0, abs=1e-6) and len(m["segments"]) == 2
        assert client.post("/api/measure", json={"asset_id": asset, "points": [[0, 0]]}).status_code == 400
