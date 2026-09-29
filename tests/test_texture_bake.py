"""Pure numpy UV texture baker (used where Open3D's project_images_to_albedo is unavailable, e.g. aarch64)."""
from __future__ import annotations

import numpy as np
import pytest

from cloudclean import texture as tx
from cloudclean.texture import CameraView, TextureParams, colorize_mesh
from tests import test_core
from tests.synthetic import ground_truth_mesh, render_photo, surface_color

EYES = [(0, -160, 60), (150, 60, 40), (-140, 70, -60), (20, 30, -170)]


@pytest.fixture
def numpy_baker(monkeypatch):
    monkeypatch.setenv("CLOUDCLEAN_TEXTURE_BAKER", "numpy")


def _views(tmp_path):
    views = []
    for i, eye in enumerate(EYES):
        cam = render_photo(tmp_path / f"photo{i}.png", eye, fov=40)
        views.append(CameraView.from_dict(cam, image_path=tmp_path / f"photo{i}.png"))
    return views


def test_core_texture_test_with_numpy_baker(tmp_path, numpy_baker):
    test_core.test_texture_projection_colors_match_photo(tmp_path)


def test_numpy_bake_agrees_with_vertex_colours(tmp_path, numpy_baker):
    gt = ground_truth_mesh()
    views = _views(tmp_path)
    colored, report, textured = colorize_mesh(gt, views, TextureParams(texture_size=1024), log=lambda m: None)
    assert textured is not None, report.get("texture_error")
    work, uvs, img = textured["mesh"], textured["uvs"], textured["image"]
    assert img.shape == (1024, 1024, 3) and img.dtype == np.uint8
    assert uvs.shape == (len(work.triangles), 3, 2)

    # sample the texture slightly inside each triangle corner (avoids chart-border texels)
    tris = np.asarray(work.triangles)
    centroid = uvs.mean(axis=1, keepdims=True)
    corner_uv = (uvs + 0.1 * (centroid - uvs)).reshape(-1, 2)
    H, W = img.shape[:2]
    px = np.clip((corner_uv[:, 0] * W).astype(int), 0, W - 1)
    py = np.clip(((1 - corner_uv[:, 1]) * H).astype(int), 0, H - 1)
    sampled = img[py, px].astype(float) / 255
    vid = tris.reshape(-1)
    # same vertex order (no decimation at this size), so compare with the vertex-colour path directly
    assert len(work.vertices) == len(colored.vertices)
    vcol = np.asarray(colored.vertex_colors)[vid]
    pos = np.asarray(work.vertices)
    corner_pos = (pos[tris] + 0.1 * (pos[tris].mean(axis=1, keepdims=True) - pos[tris])).reshape(-1, 3)
    diff = np.abs(sampled - vcol).max(axis=1)
    # away from stripe boundaries (where a vertex and a nearby texel legitimately differ) they must agree
    truth = surface_color(corner_pos)
    flat = np.abs(surface_color(pos[vid]) - truth).max(axis=1) < 1e-3
    # restrict to surface the photos see (vertex colour correct); unseen areas are filled by guesswork in both
    seen = flat & (np.abs(vcol - truth).max(axis=1) < 0.05)
    agree = np.mean(diff[seen] < 0.05)
    assert agree > 0.97, f"only {agree:.1%} of texture samples agree with vertex colours"
    assert np.median(diff[seen]) < 0.02
    assert np.mean(diff[flat] < 0.05) > 0.85  # including filled (unseen) areas
    truth_err = np.abs(sampled - truth).max(axis=1)
    assert np.mean(truth_err < 0.1) > 0.88


def test_grid_uv_fallback_bakes(tmp_path, numpy_baker, monkeypatch):
    """If UVAtlas is unavailable the per-triangle grid charts must still produce a correct texture."""
    import open3d as o3d

    def boom(self, *a, **k):
        raise RuntimeError("no uvatlas")

    monkeypatch.setattr(o3d.t.geometry.TriangleMesh, "compute_uvatlas", boom)
    gt = ground_truth_mesh()
    views = _views(tmp_path)
    colored, report, textured = colorize_mesh(gt, views, TextureParams(texture_size=2048), log=lambda m: None)
    assert textured is not None, report.get("texture_error")
    uvs = textured["uvs"]
    tris = np.asarray(textured["mesh"].triangles)
    centroid = uvs.mean(axis=1)
    img = textured["image"].astype(float) / 255
    H, W = img.shape[:2]
    px = np.clip((centroid[:, 0] * W).astype(int), 0, W - 1)
    py = np.clip(((1 - centroid[:, 1]) * H).astype(int), 0, H - 1)
    truth = surface_color(np.asarray(textured["mesh"].vertices)[tris].mean(axis=1))
    assert np.mean(np.abs(img[py, px] - truth).max(axis=1) < 0.15) > 0.8


def test_rasterize_covers_triangles_without_gaps():
    uvs = tx.grid_uvs(50, 256)
    texel, tri, bary = tx._rasterize_uv(uvs.astype(np.float64), 256)
    assert len(np.unique(texel)) == len(texel)
    assert np.allclose(bary.sum(1), 1, atol=1e-5) and (bary >= 0).all()
    # every triangle owns at least its interior texels
    assert len(np.unique(tri)) == 50
