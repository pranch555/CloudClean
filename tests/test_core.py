import json

import numpy as np
import open3d as o3d
import pytest

from cloudclean.clean import CleanParams, clean_point_cloud
from cloudclean.io import describe, load, save
from cloudclean.mesh import MeshParams, reconstruct_mesh
from cloudclean.register import MergeParams, merge_geometries, rigid_from_pairs
from cloudclean.texture import CameraView, TextureParams, colorize_mesh, save_textured
from tests.synthetic import (distance_to_gt, ground_truth_mesh, random_rigid, render_photo, simulate_scan,
                             surface_color)

GT_DIMS = np.asarray(describe(ground_truth_mesh())["dimensions"])


@pytest.fixture(scope="module")
def scans():
    T_b = random_rigid(7)
    a, _ = simulate_scan([0.2, 0.1, 1.0], seed=1, add_table=True)
    b, b_truth = simulate_scan([-0.1, -0.2, -1.0], seed=2, transform=T_b)
    return a, b, T_b, b_truth


def test_io_roundtrip_text_formats(tmp_path, scans):
    a = scans[0].select_by_index(np.arange(5000))
    for ext in (".ply", ".xyz", ".asc", ".csv"):
        path = save(a, tmp_path / f"scan{ext}")
        loaded = load(path)
        assert len(loaded.points) == 5000
        assert np.allclose(np.asarray(loaded.points), np.asarray(a.points), atol=1e-4)
    assert load(tmp_path / "scan.asc").has_colors()


def test_clean_removes_noise_table_and_debris(scans):
    a = scans[0]
    params = CleanParams(remove_plane=True)
    cleaned, report = clean_point_cloud(a, params, log=lambda m: None)
    d = distance_to_gt(np.asarray(cleaned.points))
    assert np.mean(d > 0.2) < 0.001, f"{np.mean(d > 0.2):.4%} of points are still far from the surface"
    assert report["output_points"] > 0.9 * 250_000 * 0.5  # did not eat the object
    assert cleaned.has_normals()


def test_rigid_from_pairs_recovers_transform():
    T = random_rigid(3)
    src = np.random.default_rng(0).uniform(-10, 10, (5, 3))
    tgt = src @ T[:3, :3].T + T[:3, 3]
    assert np.allclose(rigid_from_pairs(src, tgt), T, atol=1e-8)


def _rotation_error_deg(T):
    return np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))


def test_merge_with_manual_pairs(scans):
    """Fallback for symmetric parts: a few roughly clicked point pairs must lock in the right pose."""
    _, b, T_b, b_truth = scans
    a_small, _ = simulate_scan([0.2, 0.1, 1.0], seed=1, n_points=60_000, outlier_fraction=0)
    b_small = b.voxel_down_sample(0.3)
    rng = np.random.default_rng(5)
    picks = b_truth[rng.choice(len(b_truth), 4, replace=False)]
    source = picks @ T_b[:3, :3].T + T_b[:3, 3]
    target = picks + rng.normal(scale=0.4, size=picks.shape)  # imprecise clicks
    pairs = {1: {"source": source.tolist(), "target": target.tolist()}}
    _, transforms, report = merge_geometries([a_small, b_small], MergeParams(), pairs=pairs, log=lambda m: None)
    err = transforms[1] @ T_b
    assert report["scans"][1]["best_candidate"] == "manual pairs"
    assert _rotation_error_deg(err) < 0.3 and np.linalg.norm(err[:3, 3]) < 0.2


@pytest.fixture(scope="module")
def merged(scans):
    a, b, T_b, _ = scans
    quiet = lambda m: None
    ca, _ = clean_point_cloud(a, CleanParams(remove_plane=True), quiet)
    cb, _ = clean_point_cloud(b, CleanParams(), quiet)
    cloud, transforms, report = merge_geometries([ca, cb], MergeParams(), log=quiet)
    return cloud, transforms, report, T_b


def test_merge_aligns_unknown_pose(merged):
    cloud, transforms, report, T_b = merged
    # transforms[1] should undo T_b
    err = transforms[1] @ T_b
    angle = _rotation_error_deg(err)
    assert angle < 0.2, f"rotation error {angle:.3f} deg"
    assert np.linalg.norm(err[:3, 3]) < 0.1, f"translation error {np.linalg.norm(err[:3, 3]):.3f} mm"
    d = distance_to_gt(np.asarray(cloud.points))
    assert np.sqrt(np.mean(d ** 2)) < 0.05


def test_mesh_is_accurate_in_size_and_shape(merged):
    cloud = merged[0]
    mesh, report = reconstruct_mesh(cloud, MeshParams(), log=lambda m: None)
    dims = np.asarray(report["mesh"]["dimensions"])
    assert np.all(np.abs(dims - GT_DIMS) < 0.15), f"mesh dims {dims} vs truth {GT_DIMS}"
    d = distance_to_gt(np.asarray(mesh.vertices))
    assert np.mean(d) < 0.03 and np.percentile(d, 99) < 0.15
    assert report["deviation"]["p95"] < 0.1


def test_mesh_from_colourless_scan_has_no_black_vertex_colours(merged, tmp_path):
    """Poisson always emits a zero-filled colour per vertex. Kept, it reads back as a genuinely black scan and the
    viewer renders an unlit black blob, so meshing a colourless cloud must leave the mesh with no colours at all."""
    cloud = o3d.geometry.PointCloud(merged[0])
    cloud.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))  # a scan captured without texture
    mesh, report = reconstruct_mesh(cloud, MeshParams(), log=lambda m: None)
    assert not mesh.has_vertex_colors()
    assert report["mesh"]["has_colors"] is False

    # ...and a file that already carries the blank array is repaired when it is loaded.
    blank = o3d.geometry.TriangleMesh(mesh)
    blank.vertex_colors = o3d.utility.Vector3dVector(np.zeros((len(blank.vertices), 3)))
    path = tmp_path / "blank.ply"
    save(blank, path)
    assert describe(load(path))["has_colors"] is False


def test_real_colours_survive_loading(tmp_path):
    gt = ground_truth_mesh()
    gt.vertex_colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.0, 0.004], (len(gt.vertices), 1)))  # very dark, but not blank
    path = tmp_path / "dark.ply"
    save(gt, path)
    assert describe(load(path))["has_colors"] is True


def test_texture_projection_colors_match_photo(tmp_path):
    gt = ground_truth_mesh()
    views = []
    for i, eye in enumerate([(0, -160, 60), (150, 60, 40), (-140, 70, -60), (20, 30, -170)]):
        cam = render_photo(tmp_path / f"photo{i}.png", eye, fov=40)
        (tmp_path / f"cam{i}.json").write_text(json.dumps(cam))
        views.append(CameraView.from_dict(cam, image_path=tmp_path / f"photo{i}.png"))
    colored, report, textured = colorize_mesh(gt, views, TextureParams(texture_size=1024), log=lambda m: None)
    V = np.asarray(colored.vertices)
    err = np.abs(np.asarray(colored.vertex_colors) - surface_color(V)).max(axis=1)
    assert report["coverage"] > 0.7
    # stripe boundaries blur slightly; the vast majority must match exactly
    assert np.mean(err < 0.1) > 0.9, f"only {np.mean(err < 0.1):.1%} of vertices match the photo colour"
    assert textured is not None, report.get("texture_error")
    paths = save_textured(textured, tmp_path / "tex.glb") + save_textured(textured, tmp_path / "tex.obj")
    assert all(p.exists() for p in paths)

    # UV convention check: sample the texture at each vertex's UV and compare to the true colour
    tris = np.asarray(textured["mesh"].triangles)
    uvs = textured["uvs"].reshape(-1, 2)
    pos = np.asarray(textured["mesh"].vertices)[tris].reshape(-1, 3)
    img = textured["image"].astype(float) / 255
    H, W = img.shape[:2]
    px = np.clip((uvs[:, 0] * W).astype(int), 0, W - 1)
    py = np.clip(((1 - uvs[:, 1]) * H).astype(int), 0, H - 1)  # Open3D uv origin is bottom-left
    sampled = img[py, px]
    painted = sampled.sum(axis=1) > 0.05
    match = np.abs(sampled[painted] - surface_color(pos[painted])).max(axis=1) < 0.15
    assert match.mean() > 0.75, f"texture/uv mismatch: {match.mean():.1%}"

    # exported GLB must be valid glTF with the texture embedded
    import struct
    raw = (tmp_path / "tex.glb").read_bytes()
    magic, version, length = struct.unpack("<4sII", raw[:12])
    assert magic == b"glTF" and version == 2 and length == len(raw)
    doc = json.loads(raw[20:20 + struct.unpack("<I", raw[12:16])[0]])
    assert doc["images"][0]["mimeType"] == "image/png" and "TEXCOORD_0" in doc["meshes"][0]["primitives"][0]["attributes"]
