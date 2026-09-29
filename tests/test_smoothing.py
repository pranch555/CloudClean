"""Local smoothing, feature-preserving denoising and spike removal: accuracy, locality and reports."""
import math
import time

import numpy as np
import open3d as o3d
import pytest
from fastapi.testclient import TestClient

from cloudclean.edit import OP_CATALOGUE, apply_edits, validate_ops
from cloudclean.io import save
from cloudclean.web.server import create_app
from tests.synthetic import three_view_matrix

V3 = o3d.utility.Vector3dVector
quiet = lambda m: None


# --------------------------------------------------------------------------- fixtures / helpers
def icosphere(radius=20.0, subdivisions=5) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_icosahedron(1.0).subdivide_midpoint(subdivisions)
    v = np.asarray(mesh.vertices)
    mesh.vertices = V3(v / np.linalg.norm(v, axis=1, keepdims=True) * radius)
    mesh.compute_vertex_normals()
    return mesh


def with_noise(mesh, sigma, seed=0) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh(mesh)
    v = np.asarray(mesh.vertices)
    out.vertices = V3(v + np.random.default_rng(seed).normal(scale=sigma, size=v.shape))
    out.compute_vertex_normals()
    out.vertex_colors = V3(np.full((len(v), 3), 0.25))
    return out


def grid_mesh(n=201, span=20.0, height=None) -> o3d.geometry.TriangleMesh:
    """Regular triangulated grid in the z = height(x, y) plane, centred on the origin."""
    g = np.linspace(-span / 2, span / 2, n)
    xx, yy = np.meshgrid(g, g)
    zz = np.zeros_like(xx) if height is None else height(xx, yy)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    mesh = o3d.geometry.TriangleMesh(V3(np.c_[xx.ravel(), yy.ravel(), zz.ravel()]),
                                     o3d.utility.Vector3iVector(np.r_[np.c_[a, b, d], np.c_[a, d, c]]))
    mesh.compute_vertex_normals()
    return mesh


def dihedral_along(mesh, tris, pairs) -> np.ndarray:
    """Interior angle (degrees, 90 for a cube edge) between the faces of each (face0, face1) pair."""
    v = np.asarray(mesh.vertices)
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return 180 - np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n[pairs[:, 0]], n[pairs[:, 1]]), -1, 1)))


# --------------------------------------------------------------------------- (a) sphere
def test_denoise_sphere_reduces_error_without_shrinking():
    clean = icosphere(20.0, 5)
    noisy = with_noise(clean, 0.05)
    r0 = np.linalg.norm(np.asarray(noisy.vertices), axis=1)
    rms0 = np.sqrt(np.mean((r0 - 20) ** 2))
    out, report = apply_edits(noisy, [{"op": "denoise"}], quiet)
    r = np.linalg.norm(np.asarray(out.vertices), axis=1)
    rms = np.sqrt(np.mean((r - 20) ** 2))
    assert rms <= 0.5 * rms0, (rms0, rms)
    assert abs(r.mean() - r0.mean()) < 0.02
    entry = report["ops"][0]
    assert entry["after"] == entry["before"]
    np.testing.assert_array_equal(np.asarray(out.triangles), np.asarray(noisy.triangles))
    assert out.has_vertex_colors() and out.has_vertex_normals()
    moved = np.linalg.norm(np.asarray(out.vertices) - np.asarray(noisy.vertices), axis=1)
    assert entry["moved"]["max"] == pytest.approx(moved.max())
    assert entry["moved"]["mean"] == pytest.approx(moved.mean())
    assert entry["region_vertices"] == len(r) and entry["moved_vertices"] == int((moved > 0).sum())


# --------------------------------------------------------------------------- (b) cube
def test_denoise_keeps_cube_edges_sharp():
    box = o3d.geometry.TriangleMesh.create_box(10, 10, 10).subdivide_midpoint(6)
    box.remove_duplicated_vertices()
    true = np.asarray(box.vertices).copy()
    tris = np.asarray(box.triangles)
    noisy = with_noise(box, 0.02, seed=1)
    on_face = np.isclose(true, 0) | np.isclose(true, 10)
    on_edge = on_face.sum(axis=1) == 2

    # mesh edges lying on the cube edges, with their two faces
    e = np.sort(tris[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    key = e[:, 0] * len(true) + e[:, 1]
    order = np.argsort(key, kind="stable")
    first = np.flatnonzero(np.r_[True, key[order][1:] != key[order][:-1]])
    ends = e[order[first]]
    crease = on_edge[ends[:, 0]] & on_edge[ends[:, 1]] & ((on_face[ends[:, 0]] & on_face[ends[:, 1]]).sum(1) == 2)
    pairs = np.c_[order[first] // 3, order[first + 1] // 3][crease]

    def size_and_edge_error(mesh):
        p = np.asarray(mesh.vertices)
        inner = (true[:, 1] > 1) & (true[:, 1] < 9) & (true[:, 2] > 1) & (true[:, 2] < 9)
        size = p[inner & np.isclose(true[:, 0], 10), 0].mean() - p[inner & np.isclose(true[:, 0], 0), 0].mean()
        err = np.sqrt((((p - true) ** 2) * on_face)[on_edge].sum(axis=1))  # distance from the true edge line
        return size, float(err.mean())

    denoised, report = apply_edits(noisy, [{"op": "denoise"}], quiet)
    rounded, _ = apply_edits(noisy, [{"op": "smooth", "method": "taubin", "preserve_edges": False}], quiet)
    kept, kept_report = apply_edits(noisy, [{"op": "smooth", "method": "taubin", "preserve_edges": True}], quiet)

    size, err = size_and_edge_error(denoised)
    _, err_noisy = size_and_edge_error(noisy)
    assert abs(size - 10) < 0.05
    assert err < 0.5 * err_noisy
    angles = dihedral_along(denoised, tris, pairs)
    assert abs(np.median(angles) - 90) < 2 and np.percentile(np.abs(angles - 90), 90) < 10

    rounded_angles = dihedral_along(rounded, tris, pairs)
    assert np.median(rounded_angles) > 110  # plain Taubin bevels the edges
    assert size_and_edge_error(rounded)[1] > 2 * err
    kept_angles = dihedral_along(kept, tris, pairs)
    assert abs(np.median(kept_angles) - 90) < abs(np.median(rounded_angles) - 90) / 3
    assert kept_report["ops"][0]["feature_edges"] >= len(pairs)
    assert abs(size_and_edge_error(kept)[0] - 10) < 0.05
    assert report["ops"][0]["moved"]["p95"] < 0.1


# --------------------------------------------------------------------------- (c) regions
def test_region_smoothing_is_local_and_blends():
    rng = np.random.default_rng(2)
    mesh = grid_mesh(n=161, span=16.0)
    v = np.asarray(mesh.vertices)
    v[:, 2] = rng.normal(scale=0.02, size=len(v))
    mesh.vertices = V3(v)
    mesh.compute_vertex_normals()
    center, radius, feather = np.array([1.0, -0.5, 0.0]), 2.0, 1.5
    region = {"spheres": [[*center, radius]]}
    for op in ({"op": "smooth", "iterations": 10}, {"op": "denoise"}):
        out, report = apply_edits(mesh, [{**op, "region": region, "feather": feather}], quiet)
        p = np.asarray(out.vertices)
        moved = np.linalg.norm(p - v, axis=1)
        dist = np.maximum(np.linalg.norm(v - center, axis=1) - radius, 0)
        assert np.all(moved[dist >= feather] == 0)  # exactly untouched outside region + feather band
        np.testing.assert_array_equal(p[dist >= feather], v[dist >= feather])
        inside = dist == 0
        assert np.std(p[inside, 2]) < 0.5 * np.std(v[inside, 2])
        # no step: the displacement fades with the cosine ramp across the band, so the outer part hardly moves
        ramp = np.where(dist < feather, 0.5 * (1 + np.cos(np.pi * np.minimum(dist / feather, 1))), 0)
        assert np.all(moved <= 1.5 * ramp * moved[inside].max() + 1e-12)
        assert moved[(dist > 0.8 * feather) & (dist < feather)].max() < 0.15 * moved[inside].max()
        entry = report["ops"][0]
        assert entry["region_vertices"] == int((dist < feather).sum())
        assert 0 < entry["moved_vertices"] <= entry["region_vertices"]
        assert entry["moved"]["max"] == pytest.approx(moved.max())

    # the same on a cloud with a box region, and with a screen selection
    cloud = o3d.geometry.PointCloud(V3(v))
    cloud.colors = V3(rng.random((len(v), 3)))
    box = {"box": {"min": [-3, -3, -1], "max": [0, 3, 1]}}
    out, report = apply_edits(cloud, [{"op": "smooth_points", "region": box, "feather": 1.0}], quiet)
    p = np.asarray(out.points)
    outside = np.linalg.norm(np.maximum(np.maximum([-3, -3, -1] - v, v - [0, 3, 1]), 0), axis=1) >= 1.0
    np.testing.assert_array_equal(p[outside], v[outside])
    assert np.abs(p[~outside, 2]).mean() < np.abs(v[~outside, 2]).mean()
    np.testing.assert_array_equal(np.asarray(out.colors), np.asarray(cloud.colors))

    vm = np.asarray(three_view_matrix((0, 0, 30), (0, 0, 0), up=(0, 1, 0))).reshape(4, 4).T
    f = 1 / math.tan(math.radians(25))
    proj = np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, -1.002, -0.2002], [0, 0, -1, 0]])
    vp = (proj @ vm).T.reshape(-1).tolist()
    screen = {"view_projection": vp, "polygon": [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]}
    out, report = apply_edits(mesh, [{"op": "denoise", "region": screen, "feather": 0.5}], quiet)
    moved = np.linalg.norm(np.asarray(out.vertices) - v, axis=1)
    half = 0.05 * 30 / f  # world half-width of the selection square at z = 0
    far = (np.abs(v[:, 0]) > half + 0.5 + 1e-3) | (np.abs(v[:, 1]) > half + 0.5 + 1e-3)
    assert np.all(moved[far] == 0) and moved[~far].max() > 0
    assert report["ops"][0]["region_vertices"] < len(v) / 4


# --------------------------------------------------------------------------- (d) spikes
def bumps(xx, yy, centres, height=0.5, width=0.15):
    z = np.zeros_like(xx)
    for cx, cy in centres:
        z += height * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * width ** 2))
    return z


def test_remove_spikes_mesh_keeps_ridge():
    centres = [(-6, -6), (-3, 4), (2, -5), (6, 6), (-7, 1)]
    ridge = lambda xx, yy: 0.5 * np.exp(-((xx - 4) ** 2) / (2 * 0.15 ** 2))
    rng = np.random.default_rng(3)
    mesh = grid_mesh(n=201, span=20.0, height=lambda xx, yy: bumps(xx, yy, centres) + ridge(xx, yy))
    v = np.asarray(mesh.vertices)
    v[:, 2] += rng.normal(scale=0.003, size=len(v))
    mesh.vertices = V3(v)
    mesh.compute_vertex_normals()
    truth = ridge(v[:, 0], v[:, 1])  # the surface without the spikes

    out, report = apply_edits(mesh, [{"op": "remove_spikes", "max_size": 1.5}], quiet)
    p = np.asarray(out.vertices)
    residual = np.abs(p[:, 2] - truth)
    assert residual.max() < 0.05, residual.max()
    ridge_line = np.abs(v[:, 0] - 4) < 0.05
    assert p[ridge_line, 2].min() > 0.45  # the real feature is untouched
    assert np.abs(p[:, 2] - v[:, 2])[np.abs(v[:, 0] - 4) < 1].max() < 0.02
    entry = report["ops"][0]
    assert entry["removed_patches"] == len(centres) and entry["kept_features"] >= 1
    assert entry["moved"]["max"] == pytest.approx(0.5, abs=0.05)
    np.testing.assert_array_equal(np.asarray(out.triangles), np.asarray(mesh.triangles))

    inward, _ = apply_edits(mesh, [{"op": "remove_spikes", "max_size": 1.5, "direction": "inward"}], quiet)
    np.testing.assert_allclose(np.asarray(inward.vertices), v, atol=0.02)  # bumps point outward: kept


def test_remove_spikes_sphere_cloud():
    rng = np.random.default_rng(4)
    n = 120_000
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    spikes = dirs[rng.choice(n, 6, replace=False)]
    ang = np.arccos(np.clip(dirs @ spikes.T, -1, 1)) * 5  # arc distance (mm) to each spike centre
    radius = 5 + 0.5 * np.exp(-ang ** 2 / (2 * 0.08 ** 2)).sum(axis=1) + rng.normal(scale=0.003, size=n)
    cloud = o3d.geometry.PointCloud(V3(dirs * radius[:, None]))
    cloud.normals = V3(dirs)
    out, report = apply_edits(cloud, [{"op": "remove_spikes", "max_size": 0.8, "direction": "outward"}], quiet)
    r = np.linalg.norm(np.asarray(out.points), axis=1)
    assert len(r) == n
    assert np.abs(r - 5).max() < 0.05
    assert report["ops"][0]["removed_patches"] >= 6
    far = ang.min(axis=1) > 0.5  # the rest of the sphere is left alone
    assert np.abs(np.asarray(out.points) - np.asarray(cloud.points))[far].max() < 0.01


# --------------------------------------------------------------------------- (e) MLS
def test_smooth_points_plane_cloud():
    rng = np.random.default_rng(5)
    n = 60_000
    pts = np.c_[rng.uniform(-5, 5, (n, 2)), rng.normal(scale=0.01, size=n)]
    tilt = o3d.geometry.get_rotation_matrix_from_xyz([0.3, -0.2, 0.1])
    cloud = o3d.geometry.PointCloud(V3(pts @ tilt.T + [3, 1, 2]))
    cloud.normals = V3(np.tile(tilt[:, 2], (n, 1)))

    def plane_residual(p):
        c = p.mean(axis=0)
        normal = np.linalg.svd(p - c, full_matrices=False)[2][2]
        return np.sqrt(np.mean(((p - c) @ normal) ** 2))

    before = plane_residual(np.asarray(cloud.points))
    for order in (1, 2):
        out, report = apply_edits(cloud, [{"op": "smooth_points", "order": order}], quiet)
        assert len(out.points) == n
        after = plane_residual(np.asarray(out.points))
        assert after <= (0.4 if order == 1 else 0.6) * before, (order, before, after)
        assert np.all(np.abs(np.asarray(out.normals) @ tilt[:, 2]) > 0.9)  # normals refreshed and oriented
        assert report["ops"][0]["moved"]["mean"] > 0
    out, _ = apply_edits(cloud, [{"op": "smooth", "method": "laplacian"}, {"op": "denoise", "iterations": 2}], quiet)
    assert len(out.points) == n and plane_residual(np.asarray(out.points)) < 0.4 * before


# --------------------------------------------------------------------------- (f) validation
def test_smoothing_catalogue_and_validation():
    for name in ("smooth", "denoise", "smooth_points", "remove_spikes"):
        assert OP_CATALOGUE[name]["args"]["region"]["type"] == "region"
    assert OP_CATALOGUE["smooth"]["applies_to"] == ["pointcloud", "mesh"]
    ops = validate_ops([{"op": "smooth", "iterations": 3}], "mesh")  # old arguments keep working
    assert ops[0]["method"] == "taubin" and ops[0]["preserve_edges"] is True and ops[0]["region"] is None
    ok = validate_ops([{"op": "denoise", "region": {"spheres": [[0, 0, 0, 1], [1, 0, 0, 2]]}},
                       {"op": "remove_spikes", "region": {"box": {"min": [0, 0, 0], "max": [1, 1, 1]}}}], "mesh")
    assert ok[0]["region"] == {"spheres": [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0]]}
    vp = np.eye(4).reshape(-1).tolist()
    ok = validate_ops([{"op": "smooth_points", "region": {"view_projection": vp, "polygon": [[0, 0], [1, 0], [1, 1]]}}],
                      "pointcloud")
    assert ok[0]["region"]["visible_only"] is False

    bad = [
        ({"op": "denoise", "region": [0, 0, 0, 1]}, "must be an object"),
        ({"op": "denoise", "region": {}}, "exactly one of"),
        ({"op": "denoise", "region": {"spheres": [[0, 0, 0]]}}, r"\[x, y, z, radius\]"),
        ({"op": "denoise", "region": {"spheres": [[0, 0, 0, -1]]}}, "radius must be > 0"),
        ({"op": "denoise", "region": {"spheres": []}}, "non-empty"),
        ({"op": "denoise", "region": {"spheres": [[0, 0, 0, 1]], "box": {}}}, "exactly one of"),
        ({"op": "denoise", "region": {"box": {"min": [1, 0, 0], "max": [0, 1, 1]}}}, "<="),
        ({"op": "denoise", "region": {"view_projection": vp}}, "needs both"),
        ({"op": "denoise", "region": {"view_projection": [0] * 16, "polygon": [[0, 0], [1, 0], [1, 1]]}},
         "not invertible"),
        ({"op": "denoise", "region": {"spheres": [[0, 0, 0, 1]], "radius": 2}}, "unexpected key"),
        ({"op": "smooth", "strength": 1.5}, "strength"),
        ({"op": "smooth", "method": "gaussian"}, "must be one of"),
        ({"op": "denoise", "normal_sigma_deg": 0}, "normal_sigma_deg"),
        ({"op": "denoise", "feather": -1}, "feather"),
        ({"op": "smooth_points", "order": 3}, "order"),
        ({"op": "remove_spikes", "direction": "up"}, "must be one of"),
        ({"op": "remove_spikes", "max_size": 2, "radius": 1}, "radius"),
    ]
    for op, message in bad:
        with pytest.raises(ValueError, match=message):
            validate_ops([op], "pointcloud" if op["op"] == "smooth_points" else "mesh")
    with pytest.raises(ValueError, match="only works on point clouds"):
        validate_ops([{"op": "smooth_points"}], "mesh")
    with pytest.raises(ValueError, match="does not touch the object"):
        apply_edits(icosphere(1.0, 2), [{"op": "denoise", "region": {"spheres": [[50, 0, 0, 1]]}}], quiet)


# --------------------------------------------------------------------------- (g) API
def _wait(client, job, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        current = {j["id"]: j for j in client.get("/api/jobs").json()}[job["id"]]
        if current["status"] in ("done", "failed", "cancelled"):
            assert current["status"] == "done", "\n".join(current["logs"][-40:]) + f"\n{current['error']}"
            return current
        time.sleep(0.3)
    raise TimeoutError(job["title"])


def test_denoise_api(tmp_path):
    save(with_noise(icosphere(20.0, 4), 0.05), tmp_path / "head.ply")
    with TestClient(create_app(tmp_path / "ws")) as client:
        assert "region" in client.get("/api/edit/ops").json()["denoise"]["args"]
        job = client.post("/api/import-paths", json={"paths": [str(tmp_path / "head.ply")]}).json()["jobs"][0]
        asset = _wait(client, job)["result"][0]
        bad = client.post("/api/edit", json={"asset_id": asset, "ops": [
            {"op": "denoise", "region": {"spheres": [[0, 0, 20]]}}]})
        assert bad.status_code == 400 and "radius" in bad.json()["detail"]
        done = _wait(client, client.post("/api/edit", json={"asset_id": asset, "ops": [
            {"op": "denoise", "region": {"spheres": [[0, 0, 20, 4], [2, 0, 19.9, 4]]}, "strength": 0.8}]}).json())
        info = client.get(f"/api/assets/{done['result'][0]}").json()
        entry = info["report"]["ops"][0]
        assert entry["op"] == "denoise" and entry["moved"]["max"] > 0 and {"mean", "p95"} <= set(entry["moved"])
        assert 0 < entry["region_vertices"] < entry["before"]["vertices"]
        assert entry["after"] == entry["before"]
