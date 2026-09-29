"""Colour from photos (job colour_from_photos): the photo model lined up with the scan or mesh (a similarity), the
cameras moved into the model's frame, colouring point clouds, the job, route and assistant tool.

No Docker, GPU or network: the reconstruction is replaced by a folder of synthetic outputs - a photo model of a
flanged part (surface points under a known similarity, plus the table, clutter and stray points), cameras around it
and ray-traced photos."""
import asyncio
import json
import math
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from cloudclean.photo_align import (align_photo_model, camera_centres_axes, camera_to_model, find_part,
                                    find_part_by_camera_points, look_at_point, similarity, split_similarity)
from cloudclean.texture import CameraView, TextureParams, colour_points
from cloudclean.web import jobs_colour_photos as jcp
from cloudclean.web import jobs_photos3d as photos3d
from cloudclean.web import routes_colour_photos
from cloudclean.web.workspace import Workspace
from tests.synthetic import ground_truth_mesh, scan_surface

TRUE_SCALE = 0.037            # photo units per mm
SIZE = (480, 360)             # photo width, height (px)
K = np.array([[450.0, 0, 239.5], [0, 450.0, 179.5], [0, 0, 1]])
RED, BLUE, TABLE = (0.9, 0.1, 0.1), (0.1, 0.2, 0.9), (0.35, 0.6, 0.35)

# A flange with a shaft, a side lug and a boss: no rotation maps it onto itself. Cylinders along z: (x, y, r, z0, z1),
# boxes: (min, max). It stands on the table at z = -6.
PARTS = [("cyl", (0.0, 0.0, 6.0, 0.0, 40.0)), ("cyl", (0.0, 0.0, 18.0, -6.0, 0.0)),
         ("box", ((4.0, -5.0, 26.0), (24.0, 5.0, 34.0))), ("cyl", (-8.0, 10.0, 3.0, 0.0, 5.0))]


# --------------------------------------------------------------------------- synthetic part, photos and cameras
def _primitive(kind, p):
    if kind == "cyl":
        x, y, r, z0, z1 = p
        m = o3d.geometry.TriangleMesh.create_cylinder(r, z1 - z0, resolution=max(24, int(r * 8)), split=4)
        return m.translate([x, y, (z0 + z1) / 2])
    lo, hi = np.array(p[0]), np.array(p[1])
    return o3d.geometry.TriangleMesh.create_box(*(hi - lo)).subdivide_midpoint(2).translate(lo)


def part_mesh(parts=PARTS):
    out = o3d.geometry.TriangleMesh()
    for kind, p in parts:
        out += _primitive(kind, p)
    out.compute_vertex_normals()
    return out


def _inside(pts, kind, p, tol=1e-3):
    if kind == "cyl":
        x, y, r, z0, z1 = p
        return (np.hypot(pts[:, 0] - x, pts[:, 1] - y) < r + tol) & (pts[:, 2] > z0 - tol) & (pts[:, 2] < z1 + tol)
    return np.all((pts > np.array(p[0]) - tol) & (pts < np.array(p[1]) + tol), axis=1)


def surface(spacing, parts=PARTS, bottom=True):
    """Outer surface points and normals of the part (faces inside another primitive dropped)."""
    pts, nrm = [], []
    for k, (kind, p) in enumerate(parts):
        m = _primitive(kind, p)
        pc = m.sample_points_uniformly(max(100, int(m.get_surface_area() / spacing ** 2)), use_triangle_normal=True)
        x, n = np.asarray(pc.points), np.asarray(pc.normals)
        keep = np.ones(len(x), bool)
        for j, (kind2, p2) in enumerate(parts):
            if j != k:
                keep &= ~_inside(x, kind2, p2)
        pts.append(x[keep])
        nrm.append(n[keep])
    pts, nrm = np.vstack(pts), np.vstack(nrm)
    keep = np.ones(len(pts), bool) if bottom else pts[:, 2] > -6 + 1e-3
    return pts[keep], nrm[keep]


def scan_cloud(parts=PARTS, spacing=0.35, noise=0.02, seed=1):
    pts, nrm = surface(spacing, parts)
    pts = pts + np.random.default_rng(seed).normal(scale=noise, size=pts.shape)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cloud.normals = o3d.utility.Vector3dVector(nrm)
    return cloud


def look_at(eye, target, up=(0, 0, 1)):
    """OpenCV world_to_camera of a camera at `eye` looking at `target`."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    W = np.eye(4)
    W[:3, :3], W[:3, 3] = R, -R @ eye
    return W


def ring_cameras(n=12, distance=150.0, target=(0, 0, 15), seed=0):
    rng = np.random.default_rng(seed)
    cams = []
    for i in range(n):
        el, az = math.radians(30 if i % 2 == 0 else 55), 2 * math.pi * i / n
        eye = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)]) * distance
        cams.append(look_at(eye + target, np.asarray(target) + rng.normal(scale=3, size=3)))
    return cams


def random_similarity(seed, scale=TRUE_SCALE):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * rng.uniform(0.8, 2.8))
    return similarity(scale, R, rng.uniform(-3, 3, 3))


def photo_model(parts=PARTS, seed=0, cameras=None, dense_offset=0.0, table_drop=0.0, table_sigma=0.05):
    """A reconstruction in its own frame: {points, colours (the dense model: the part without its underside, the
    table it stands on, a box and a ball on the table, stray points), camera_points (COLMAP-like: exact points on the
    part and the table, a few strays), S (similarity mm -> photo frame), cameras (world_to_camera, photo frame),
    cameras_mm}. dense_offset scales the dense model about the cameras, as MapAnything's depth can be off by 2-3 %
    against the cameras. table_drop / table_sigma: the dense table sits that far below the exact one and is that
    thick (mm) - on real photos MapAnything's table was ~2 % of the camera distance low and ~0.7 % thick."""
    rng = np.random.default_rng(seed)
    part, _ = surface(0.5, parts, bottom=False)
    part = part + rng.normal(scale=0.08, size=part.shape)
    g = rng.uniform(-120, 120, size=(25_000, 2))
    g = g[np.hypot(g[:, 0], g[:, 1]) > 18]
    table = np.c_[g, np.full(len(g), -6.0 - table_drop) + rng.normal(scale=table_sigma, size=len(g))]
    box = o3d.geometry.TriangleMesh.create_box(20, 15, 10).translate([60, -60, -6])
    ball = o3d.geometry.TriangleMesh.create_sphere(12).translate([-60, 65, 6])
    clutter = np.vstack([np.asarray(box.sample_points_uniformly(3000).points),
                         np.asarray(ball.sample_points_uniformly(3000).points)])
    stray = rng.uniform([-120, -120, -6], [120, 120, 80], size=(500, 3))
    pts = np.vstack([part, table, clutter, stray])
    colours = np.vstack([np.where(part[:, :1] > 0, RED, BLUE), np.tile(TABLE, (len(table), 1)),
                         np.full((len(clutter) + len(stray), 3), 0.5)])
    cams_mm = cameras if cameras is not None else ring_cameras()
    middle = camera_centres_axes(cams_mm)[0].mean(axis=0)
    pts = middle + (pts - middle) * (1 + dense_offset)
    exact, _ = surface(1.0, parts, bottom=False)
    sheet = np.c_[rng.uniform(-80, 80, size=(3000, 2)), np.full(3000, -6.0)]
    sparse = np.vstack([exact[rng.choice(len(exact), 2500, replace=False)], sheet[np.hypot(*sheet[:, :2].T) > 18]])
    sparse = np.vstack([sparse + rng.normal(scale=0.03, size=sparse.shape),
                        rng.uniform([-80, -80, -6], [80, 80, 60], size=(40, 3))])
    S = random_similarity(seed + 100)
    return {"points": pts @ S[:3, :3].T + S[:3, 3], "colours": colours, "S": S,
            "camera_points": sparse @ S[:3, :3].T + S[:3, 3], "cameras": [camera_to_model(W, S) for W in cams_mm],
            "cameras_mm": cams_mm}


def red_or_blue(centres):
    return np.where(centres[:, :1] > 0, RED, BLUE)


def render(items, W, K=K, size=SIZE, background=(0.5, 0.5, 0.5)) -> np.ndarray:
    """Ray-trace [(mesh, colour_of(triangle centres))] through a pinhole camera; pixel centres at whole-number
    coordinates, as the colouring code samples them."""
    scene = o3d.t.geometry.RaycastingScene()
    colours = []
    for mesh, colour_of in items:
        V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        colours.append(np.asarray(colour_of(V[F].mean(axis=1)), float))
    w, h = size
    us, vs = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    d = (np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], 1) @ np.linalg.inv(K).T) @ W[:3, :3]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    centre = -W[:3, :3].T @ W[:3, 3]
    hit = scene.cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(centre, d.shape), d]).astype(np.float32)))
    geom, prim = hit["geometry_ids"].numpy(), hit["primitive_ids"].numpy()
    seen = np.isfinite(hit["t_hit"].numpy())
    img = np.tile(np.asarray(background, float), (len(d), 1))
    for g, cols in enumerate(colours):
        sel = seen & (geom == g)
        img[sel] = cols[prim[sel]]
    return np.round(img.reshape(h, w, 3) * 255).astype(np.uint8)


def table_mesh():
    return o3d.geometry.TriangleMesh.create_box(300, 300, 1).translate([-150, -150, -7])


def probe_error(S_found, S_true) -> float:
    """Largest miss (mm) of a few points of the part mapped mm -> photo (truth) -> mm (found)."""
    probe = np.array([[0, 0, 0], [0, 0, 40], [18, 0, -6], [24, 0, 30.0], [-8, 10, 5]])
    back = (probe @ S_true[:3, :3].T + S_true[:3, 3]) @ S_found[:3, :3].T + S_found[:3, 3]
    return float(np.abs(back - probe).max())


def rotation_error(S_found, S_true) -> float:
    R = split_similarity(S_found)[1] @ split_similarity(S_true)[1]
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


@pytest.fixture(scope="module")
def scene():
    return photo_model()


class Log:
    def __init__(self):
        self.messages, self.steps = [], []

    def __call__(self, msg):
        self.messages.append(msg)

    def progress(self, fraction, label=None):
        self.steps.append((fraction, label))


# --------------------------------------------------------------------------- cameras into the model's frame
def test_camera_moved_into_the_model_frame_sees_the_same_pixels():
    """W' = [Rw R^T | s tw - Rw R^T t] for x_model = s R x_photo + t: every point lands on the same pixel, at s x the
    depth, and the camera centre is the moved centre."""
    rng = np.random.default_rng(4)
    for seed in range(3):
        S = random_similarity(seed, scale=rng.uniform(0.02, 40))   # photo -> model
        W = look_at(rng.normal(size=3) * 5 + [0, 0, 20], rng.normal(size=3))
        pts = rng.normal(size=(50, 3)) * 2
        moved = pts @ S[:3, :3].T + S[:3, 3]
        Wm = camera_to_model(W, S)

        def pixels(W, P):
            c = P @ W[:3, :3].T + W[:3, 3]
            return (c @ K.T)[:, :2] / c[:, 2:], c[:, 2]

        (uv, z), (uv_m, z_m) = pixels(W, pts), pixels(Wm, moved)
        s = split_similarity(S)[0]
        assert np.allclose(uv, uv_m, atol=1e-7) and np.allclose(z_m, s * z)
        R = Wm[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
        centre = -W[:3, :3].T @ W[:3, 3]
        assert np.allclose(CameraView("x", K, Wm, *SIZE).center, S[:3, :3] @ centre + S[:3, 3])
    # the aim point of cameras around a part is the part
    centres, axes = camera_centres_axes(ring_cameras())
    assert np.linalg.norm(look_at_point(centres, axes) - [0, 0, 15]) < 3


# --------------------------------------------------------------------------- lining up
def test_alignment_recovers_the_scale_rotation_and_shift(scene):
    log = []
    res = align_photo_model(scene["points"], scene["cameras"], scan_cloud(), "flange scan", log=log.append,
                            camera_points=scene["camera_points"])
    truth = np.linalg.inv(scene["S"])   # photo -> mm
    S = res["transform"]
    assert rotation_error(S, scene["S"]) < 0.5
    assert split_similarity(S)[0] == pytest.approx(split_similarity(truth)[0], rel=0.005)
    assert probe_error(S, scene["S"]) < 0.2
    assert res["trusted"] and not res["ambiguous"] and not res["warnings"]
    assert res["fitted_on"] == "camera points" and res["photo_points"] > 1000   # the part's, not the table's
    assert res["fitness"] > 0.9 and res["coverage"] > 0.5 and res["rmse_mm"] < 0.2
    assert res["planes_removed"] >= 1   # the table
    assert any(m.startswith("Lined the photos up with flange scan: scale ×27") for m in log)


def test_a_thick_dense_table_off_the_exact_one_is_still_removed():
    """Real photos: the dense table sat 2 % of the camera distance below COLMAP's (exact) table and was 0.7 % thick,
    so a plane fitted to the dense points kept a ring of table around the part and the pose came out wrong. The table
    is now found in the exact points and the dense table measured against it."""
    scene = photo_model(seed=3, table_drop=3.0, table_sigma=1.0)
    centres, axes = camera_centres_axes(scene["cameras"])
    table_share = []
    for finder in (lambda: find_part_by_camera_points(scene["points"], scene["camera_points"], centres, axes,
                                                      lambda m: None),
                   lambda: find_part(scene["points"], centres, axes, lambda m: None)):
        idx, _ = finder()
        mm = (scene["points"][idx] - scene["S"][:3, 3]) @ scene["S"][:3, :3] / split_similarity(scene["S"])[0] ** 2
        table_share.append(float((mm[:, 2] < -4.5).mean()))   # the part stands on z = -6, the dense table lower
    assert table_share[0] < 0.02 < table_share[1]   # the dense-only finder keeps a ring of table
    log = []
    res = align_photo_model(scene["points"], scene["cameras"], scan_cloud(), "flange scan", log=log.append,
                            camera_points=scene["camera_points"])
    assert any("Found the part with the photos' own points" in m for m in log)
    assert rotation_error(res["transform"], scene["S"]) < 0.5 and probe_error(res["transform"], scene["S"]) < 0.2
    assert res["trusted"] and res["fitted_on"] == "camera points" and res["fitness"] > 0.9


def test_the_camera_points_correct_a_dense_model_off_in_scale():
    """The dense model 2.5 % too big against the cameras (seen on real photos): fitted on it alone, every camera
    would sit 2.5 % off; the final fit on the reconstruction's own points puts them back."""
    data = photo_model(seed=1, dense_offset=0.025)
    scan = scan_cloud()
    log = []
    res = align_photo_model(data["points"], data["cameras"], scan, "flange scan", log=log.append,
                            camera_points=data["camera_points"])
    assert split_similarity(res["transform"])[0] == pytest.approx(1 / TRUE_SCALE, rel=0.005)
    assert probe_error(res["transform"], data["S"]) < 0.2 and rotation_error(res["transform"], data["S"]) < 0.5
    assert res["dense_scale_offset_pct"] == pytest.approx(-2.44, abs=0.3) and res["trusted"]
    assert any("off in scale against the cameras - corrected" in m for m in log)

    dense_only = align_photo_model(data["points"], data["cameras"], scan, "flange scan", log=lambda m: None)
    assert abs(split_similarity(dense_only["transform"])[0] * TRUE_SCALE - 1) > 0.02
    assert dense_only["fitted_on"] == "dense model" and not dense_only["trusted"]
    assert any("missing" in w for w in dense_only["warnings"])


def test_a_different_part_is_refused(scene):
    pebble = scan_surface(ground_truth_mesh(), spacing=0.6, noise=0.02)
    with pytest.raises(ValueError, match="could not be lined up with pebble.*all the way round"):
        align_photo_model(scene["points"], scene["cameras"], pebble, "pebble", log=lambda m: None)


def test_round_part_is_lined_up_but_flagged():
    """A plain flange and shaft looks the same turned about its axis: warn that colours could land turned, not fail;
    and rough cameras are never trusted."""
    round_part = PARTS[:2]
    data = photo_model(round_part, seed=3)
    res = align_photo_model(data["points"], data["cameras"], scan_cloud(round_part), "round part", poses="mapanything",
                            log=lambda m: None)
    assert split_similarity(res["transform"])[0] == pytest.approx(1 / TRUE_SCALE, rel=0.005)
    assert res["ambiguous"] and res["alternative_angle"] > 15 and not res["trusted"]
    assert any("turned" in w for w in res["warnings"]) and any("roughly" in w for w in res["warnings"])


# --------------------------------------------------------------------------- colouring a point cloud
def plate(size, z, spacing=0.4):
    """Square plate facing +z (towards the cameras), centred on the z axis: mesh and points with normals."""
    mesh = o3d.geometry.TriangleMesh.create_box(size, size, 0.01).translate([-size / 2, -size / 2, z - 0.01])
    n = int(size / spacing)
    g = (np.stack(np.meshgrid(np.arange(n), np.arange(n)), -1).reshape(-1, 2) + 0.5) * size / n - size / 2
    return mesh, np.c_[g, np.full(len(g), z)]


def test_colour_points_takes_each_point_from_the_photos_that_see_it(tmp_path):
    """A small red plate 20 mm in front of a big blue one. The front photo sees red where the blue plate is hidden
    behind the red one: those blue points must stay blue (the oblique photo sees them)."""
    red_mesh, red_pts = plate(10, 20)
    blue_mesh, blue_pts = plate(60, 0)
    cams = [look_at([0, 0, 150], [0, 0, 0], up=(0, 1, 0)), look_at([110, 0, 100], [0, 0, 0], up=(0, 1, 0))]
    wide = np.array([[600.0, 0, 239.5], [0, 600.0, 179.5], [0, 0, 1]])
    views = []
    for k, W in enumerate(cams):
        img = render([(red_mesh, lambda c: np.tile(RED, (len(c), 1))),
                      (blue_mesh, lambda c: np.tile(BLUE, (len(c), 1)))], W, K=wide)
        Image.fromarray(img).save(tmp_path / f"{k}.png")
        views.append(CameraView(str(tmp_path / f"{k}.png"), wide, W, *SIZE))
    pts = np.vstack([red_pts, blue_pts])
    is_red = np.arange(len(pts)) < len(red_pts)
    hidden = ~is_red & (np.abs(pts[:, 0]) < 4) & (np.abs(pts[:, 1]) < 4)   # behind the red plate, seen from the front
    for with_normals in (True, False):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if with_normals:
            cloud.normals = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (len(pts), 1)))
        steps = []
        coloured, report = colour_points(cloud, views, TextureParams(), log=lambda m: None, progress=steps.append)
        c = np.asarray(coloured.colors)
        assert np.abs(c[is_red] - RED).max(axis=1).mean() < 0.02
        assert np.abs(c[~is_red] - BLUE).max(axis=1).mean() < 0.02
        assert np.all(np.abs(c[hidden] - BLUE).max(axis=1) < 0.05), "a hidden point took the colour in front of it"
        assert report["coverage"] > 0.8 and steps == [0.5, 1.0]   # the rest: near the plates' outer silhouettes
        assert [v["visible_points"] > 0 for v in report["views"]] == [True, True]
    assert np.allclose(np.asarray(coloured.points), pts)   # the points never move


# --------------------------------------------------------------------------- the job
def fake_reconstruction(monkeypatch, scene, poses="colmap", unplaced=()):
    """Replaces the container: writes the scene's photo model, its cameras (photo frame), ray-traced photos and, with
    COLMAP poses, the camera points (sparse.ply)."""
    calls = []
    given = poses   # the cameras this fake container returns (the job asks for poses=... below)
    items = [(part_mesh(), red_or_blue), (table_mesh(), lambda c: np.tile(TABLE, (len(c), 1)))]

    @contextmanager
    def fake(ws, photos, log, progress=None, lo=0.02, hi=0.92, dense=True, poses="auto"):
        calls.append({"photos": [m["id"] for m in photos], "lo": lo, "hi": hi, "poses": poses})
        poses = given
        (progress or (lambda *a: None))(lo, "Starting the reconstruction")
        work = Path(tempfile.mkdtemp(dir=ws.root))
        out = work / "out"
        (out / "images").mkdir(parents=True)
        try:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene["points"]))
            cloud.colors = o3d.utility.Vector3dVector(scene["colours"])
            o3d.io.write_point_cloud(str(out / "points.ply"), cloud)
            if poses == "colmap":
                sparse = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene["camera_points"]))
                o3d.io.write_point_cloud(str(out / "sparse.ply"), sparse)
            cams = []
            for k, (W, W_mm) in enumerate(zip(scene["cameras"][:len(photos)], scene["cameras_mm"])):
                if k in unplaced:
                    continue
                Image.fromarray(render(items, W_mm)).save(out / "images" / f"{k:03d}.png")
                cams.append({"image": f"{k:03d}.png", "file": f"images/{k:03d}.png", "K": K.tolist(),
                             "width": SIZE[0], "height": SIZE[1], "world_to_camera": W.tolist(),
                             "cam_to_world": np.linalg.inv(W).tolist()})
            data = {"poses": poses, "registered": len(cams), "photos": len(photos), "cameras": cams}
            (out / "cameras.json").write_text(json.dumps(data), encoding="utf-8")
            (progress or (lambda *a: None))(hi, "Done")
            yield out, {"photos": len(photos), "registered": len(cams), "points": len(scene["points"]),
                        "extent_mm": [1.9, 1.7, 1.8], "seconds": 30.0, "poses": poses, "model": "test"}, data
        finally:
            shutil.rmtree(work, ignore_errors=True)

    monkeypatch.setattr(jcp, "reconstruction", fake)
    return calls


def add_photos(ws, tmp_path, n, project=None):
    ids = []
    for k in range(n):
        src = tmp_path / f"{uuid.uuid4().hex[:8]}.png"
        Image.new("RGB", SIZE, (40 + 15 * k, 90, 160)).save(src)
        ids.append(ws.add_image(src, f"flange photo {k}", project=project)["id"])
    return ids


def share_right(colours, points, lo=2.0):
    """Share of the part's upper surface (clear of the red / blue border at x = 0) with the right colour."""
    right, left = (points[:, 0] > lo) & (points[:, 2] > -5.5), (points[:, 0] < -lo) & (points[:, 2] > -5.5)
    red = (colours[:, 0] > 0.6) & (colours[:, 2] < 0.4)
    blue = (colours[:, 2] > 0.6) & (colours[:, 0] < 0.4)
    return float(red[right].mean()), float(blue[left].mean())


def test_job_colours_a_mesh_and_bakes_a_texture(tmp_path, monkeypatch, scene):
    ws = Workspace(tmp_path / "ws")
    project = ws.create_project("Flange")
    # vertices spread over the faces, as on a mesh made from a scan (the primitives only have them on the edges)
    mesh = ws.add_geometry(part_mesh().subdivide_midpoint(1), "flange mesh", "import", project=project["id"])
    photo_ids = add_photos(ws, tmp_path, 12, project["id"])
    calls = fake_reconstruction(monkeypatch, scene, unplaced=(5,))
    log = Log()

    created = jcp.job_colour_from_photos(ws, {"asset_id": mesh["id"], "texture_size": 256}, log)

    assert len(calls) == 1 and sorted(calls[0]["photos"]) == sorted(photo_ids)   # every photo of the project
    assert (calls[0]["lo"], calls[0]["hi"]) == (0.02, 0.6)
    meta = ws.get(created[0])
    placed = [i for k, i in enumerate(calls[0]["photos"]) if k != 5]
    assert meta["kind"] == "mesh" and meta["operation"] == "texture" and meta["project"] == project["id"]
    assert meta["name"] == "flange mesh · colour from 11 photos" and meta["parents"] == [mesh["id"], *placed]
    assert meta["textured"] is True
    d = ws.asset_dir(created[0])
    assert (d / "textured.glb").stat().st_size > 0 and (d / "textured_obj" / "model.obj").is_file()

    coloured = ws.load_geometry(created[0])
    red, blue = share_right(np.asarray(coloured.vertex_colors), np.asarray(coloured.vertices))
    assert red > 0.9 and blue > 0.9, (red, blue)

    report = ws.report(created[0])
    al = report["alignment"]
    assert al["scale"] == pytest.approx(1 / TRUE_SCALE, rel=0.005) and al["trusted"] and not al["ambiguous"]
    assert al["fitted_on"] == "camera points" and al["fitness"] > 0.9
    assert probe_error(np.asarray(al["transform"]), scene["S"]) < 0.2
    assert report["photos"] == 12 and report["registered"] == 11 and report["used"] == 11
    assert report["poses"] == "colmap" and report["texture_size"] == 256 and 0.5 < report["coverage"] <= 1
    assert [c["image"] for c in report["cameras"]] == placed and [v["image"] for v in report["views"]] == placed
    for cam, W_mm in zip(report["cameras"], [W for k, W in enumerate(scene["cameras_mm"]) if k != 5]):
        assert np.abs(np.asarray(cam["world_to_camera"]) - W_mm).max() < 0.05   # back where the photos were taken
    assert str(tmp_path) not in json.dumps(report)   # no paths into the (deleted) work folder

    fractions = [f for f, _ in log.steps]
    assert fractions == sorted(fractions) and fractions[0] == pytest.approx(0.02) and fractions[-1] == 1.0
    assert any(0.6 < f < 0.75 for f in fractions) and any(0.75 < f < 0.98 for f in fractions)
    assert any(m.startswith("Lined the photos up with flange mesh: scale ×27") for m in log.messages)
    assert any("1 photo(s) could not be placed" in m for m in log.messages)


def test_job_refuses_photos_as_the_model_and_too_few_photos(tmp_path, monkeypatch, scene):
    ws = Workspace(tmp_path / "ws")
    project = ws.create_project("Flange")
    scan = ws.add_geometry(scan_cloud(), "flange scan", "import", project=project["id"])
    ids = add_photos(ws, tmp_path, 2, project["id"])
    calls = fake_reconstruction(monkeypatch, scene)
    with pytest.raises(ValueError, match="at least 3 photos"):
        jcp.job_colour_from_photos(ws, {"asset_id": scan["id"]}, Log())
    with pytest.raises(ValueError, match="is a photo"):
        jcp.job_colour_from_photos(ws, {"asset_id": ids[0]}, Log())
    with pytest.raises(ValueError, match="texture_size"):
        jcp.job_colour_from_photos(ws, {"asset_id": scan["id"], "photo_ids": ids * 2 + add_photos(ws, tmp_path, 1),
                                        "texture_size": 100}, Log())
    assert calls == []


def test_job_and_route_are_registered():
    from cloudclean.web.jobs import job_function
    from cloudclean.web.server import ROUTER_MODULES

    assert job_function("colour_from_photos") is jcp.job_colour_from_photos
    assert "cloudclean.web.routes_colour_photos" in ROUTER_MODULES


# --------------------------------------------------------------------------- route
class Jobs:
    """JobManager stand-in: run=True runs the job in this process, run=False only records it."""

    def __init__(self, ws, run=True):
        self.ws, self.run, self.jobs = ws, run, {}

    def submit(self, kind, title, payload):
        from cloudclean.web.jobs import job_function

        job = {"id": f"job{len(self.jobs)}", "kind": kind, "title": title, "payload": payload, "status": "queued",
               "logs": [], "progress": None, "result": [], "error": None, "output": {}}
        if self.run:
            log = Log()
            try:
                job.update(status="done", result=job_function(kind)(self.ws, payload, log))
            except Exception as exc:
                job.update(status="failed", error=str(exc))
            job["logs"] = log.messages
        self.jobs[job["id"]] = job
        return dict(job)

    def get(self, job_id):
        return dict(self.jobs[job_id])


def test_route(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "ws")
    project = ws.create_project("Flange")
    scan = ws.add_geometry(scan_cloud(), "flange scan", "import", project=project["id"])
    jobs = Jobs(ws, run=False)
    app = FastAPI()
    app.include_router(routes_colour_photos.create_router(ws, jobs))
    client = TestClient(app)
    monkeypatch.setattr(photos3d, "recon_available", lambda: (True, "ready"))

    assert client.post("/api/colour-from-photos", json={"asset_id": "0123456789ab"}).status_code == 404
    ids = add_photos(ws, tmp_path, 2, project["id"])
    res = client.post("/api/colour-from-photos", json={"asset_id": ids[0]})
    assert res.status_code == 400 and "is a photo" in res.json()["detail"]
    res = client.post("/api/colour-from-photos", json={"asset_id": scan["id"]})
    assert res.status_code == 400 and "at least 3 photos" in res.json()["detail"]
    ids += add_photos(ws, tmp_path, 1, project["id"])
    add_photos(ws, tmp_path, 2, ws.create_project("Other part")["id"])   # another part's photos do not count
    res = client.post("/api/colour-from-photos", json={"asset_id": scan["id"], "texture_size": 50})
    assert res.status_code == 400 and "texture_size" in res.json()["detail"]
    assert client.post("/api/colour-from-photos",
                       json={"asset_id": scan["id"], "photo_ids": [ids[0], "0123456789ab"]}).status_code == 404

    not_built = "The reconstruction image cloudclean-recon:latest is not built yet - see docs/photos-to-3d.md"
    monkeypatch.setattr(photos3d, "recon_available", lambda: (False, not_built))
    res = client.post("/api/colour-from-photos", json={"asset_id": scan["id"]})
    assert res.status_code == 503 and res.json()["detail"] == not_built
    assert not jobs.jobs

    monkeypatch.setattr(photos3d, "recon_available", lambda: (True, "ready"))
    res = client.post("/api/colour-from-photos", json={"asset_id": scan["id"], "name": "flange in colour"})
    assert res.status_code == 200, res.text
    job = res.json()
    assert job["kind"] == "colour_from_photos" and job["title"] == "Colour flange scan from 3 photos"
    # every photo of the project (photos added in the same millisecond have no fixed order)
    assert sorted(job["payload"]["photo_ids"]) == sorted(ids) and job["payload"]["name"] == "flange in colour"
    res = client.post("/api/colour-from-photos", json={"asset_id": scan["id"], "photo_ids": ids[::-1]})
    assert res.status_code == 200 and res.json()["payload"]["photo_ids"] == ids[::-1]


# --------------------------------------------------------------------------- assistant
def call(ws, args, ui, jobs=None):
    from cloudclean.assistant.tools import ToolContext, call_tool

    events = []
    ctx = ToolContext(ws, jobs, {}, ui, lambda e, d: events.append((e, d)), "conv", "call", poll_interval=0.01)
    return asyncio.run(call_tool(ctx, "colour_from_photos", args)), events


def test_assistant_tool_and_guide(tmp_path, monkeypatch):
    from cloudclean.assistant.guide import BY_ID, search
    from cloudclean.assistant.tools import TOOLS, ToolError

    names = list(TOOLS)
    assert names.index("colour_from_photos") == names.index("photos_to_3d") + 1
    schema = TOOLS["colour_from_photos"].schema()["function"]
    assert set(schema["parameters"]["properties"]) == {"model", "photo_ids"}
    assert "coloured or textured from photos" in schema["description"]
    assert search("colour my scan from photos")[0][1].id == "mesh.colour"
    assert search("make a 3D model from photos")[0][1].id == "photos-to-3d"
    assert BY_ID["mesh.colour"].name == "Colour from photos" and BY_ID["mesh.colour"].nav["step"] == "mesh"

    ws = Workspace(tmp_path / "ws")
    project = ws.create_project("Flange")
    scan = ws.add_geometry(scan_cloud(), "flange scan", "import", project=project["id"])
    ui = {"project_id": project["id"], "active_id": scan["id"]}
    with pytest.raises(ToolError, match="fewer than 3 photos"):
        call(ws, {}, ui)
    ids = add_photos(ws, tmp_path, 3, project["id"])
    with pytest.raises(ToolError, match="is a image"):
        call(ws, {"model": ids[0]}, ui)
    with pytest.raises(ToolError, match="model is required"):
        call(ws, {}, {"project_id": project["id"]})
    monkeypatch.setattr(photos3d, "recon_available", lambda: (False, "Docker is installed but not running"))
    with pytest.raises(ToolError, match="not running"):
        call(ws, {"model": "flange scan"}, ui)


def test_assistant_refuses_to_colour_from_rough_cameras(tmp_path, monkeypatch, scene):
    """Through the assistant, on a point cloud: rough (MapAnything) cameras painted the colours in the wrong place
    in tests, so colouring needs the photos placed exactly - the tool says so instead of colouring."""
    from cloudclean.assistant.tools import ToolError

    ws = Workspace(tmp_path / "ws")
    project = ws.create_project("Flange")
    scan = ws.add_geometry(scan_cloud(), "flange scan", "import", project=project["id"])
    add_photos(ws, tmp_path, 12, project["id"])
    fake_reconstruction(monkeypatch, scene, poses="mapanything")
    monkeypatch.setattr(photos3d, "recon_available", lambda: (True, "ready"))

    with pytest.raises(ToolError, match="could not be placed exactly"):
        call(ws, {"model": scan["id"]}, {"project_id": project["id"]}, Jobs(ws))
    assert [m["kind"] for m in ws.list()].count("pointcloud") == 1   # nothing was added


# --------------------------------------------------------------------------- the photo check
def test_photo_check_repairs_a_placement_the_geometry_left_turned(tmp_path):
    """With few photos COLMAP's points lie mostly on flat faces, which still fit turned; the photos themselves agree
    only near the right placement. A camera set left 5 degrees turned and 1 mm off is repaired; one placed right is
    left as it is."""
    from cloudclean.photo_check import check_placement

    items = [(part_mesh(), red_or_blue), (table_mesh(), lambda c: np.tile(TABLE, (len(c), 1)))]
    views = []
    for k, W in enumerate(ring_cameras(8)):
        Image.fromarray(render(items, W)).save(tmp_path / f"{k}.png")
        views.append(CameraView(str(tmp_path / f"{k}.png"), K, W, *SIZE))
    o3d.utility.random.seed(3)
    pts, nrm = surface(0.3, bottom=False)

    def centres(vs):
        return np.array([-np.asarray(v.world_to_camera)[:3, :3].T @ np.asarray(v.world_to_camera)[:3, 3] for v in vs])

    truth = centres(views)
    right = check_placement(pts, nrm, views, log=lambda m: None, spacing=0.3)
    assert right["moved_mm"] == 0 and np.allclose(centres(right["views"]), truth)
    assert right["agreement"] - right["typical_wrong"] > 0.2   # clearly better than wrong placements

    T = np.eye(4)
    T[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, math.radians(5)])
    T[:3, 3] = [1.0, 0.5, 0.0]
    turned = [CameraView(v.image_path, v.K, np.asarray(v.world_to_camera) @ T, v.width, v.height) for v in views]
    before = np.median(np.linalg.norm(centres(turned) - truth, axis=1))
    fixed = check_placement(pts, nrm, turned, log=lambda m: None, spacing=0.3)
    after = np.median(np.linalg.norm(centres(fixed["views"]) - truth, axis=1))
    assert after < 0.2 * before, (before, after)
    assert fixed["agreement"] > fixed["agreement_before"] + 0.02 and fixed["moved_deg"] > 3
