"""Synthetic scans of an object with exactly known geometry, for testing accuracy."""
from __future__ import annotations

import math

import numpy as np
import open3d as o3d
from PIL import Image


def ground_truth_mesh() -> o3d.geometry.TriangleMesh:
    """Closed pebble-like part (~80 x 50 x 30 mm): tapered ellipsoid with a bump, no mirror symmetry."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=120)
    v = np.asarray(mesh.vertices).copy()
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    shaped = np.stack([x * (1 + 0.2 * z), y * (1 + 0.15 * x), z], axis=1)
    bump_dir = np.array([0.75, 0.6, 0.12])  # near the equator: visible in top and bottom scans
    bump_dir /= np.linalg.norm(bump_dir)
    bump = 1.0 + 0.35 * np.exp(-np.sum((v - bump_dir) ** 2, axis=1) / 0.08)
    v = shaped * bump[:, None] * np.array([40.0, 25.0, 15.0])
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_vertex_normals()
    return mesh


def surface_color(points: np.ndarray) -> np.ndarray:
    """Colour pattern defined on 3D position (stripes along x, tint along z)."""
    stripe = (np.floor(points[:, 0] / 10.0) % 2).astype(float)
    return np.stack([0.2 + 0.7 * stripe, 0.3 + 0.01 * (points[:, 2] + 15), 0.8 - 0.6 * stripe], axis=1).clip(0, 1)


def random_rigid(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * rng.uniform(0.6, 2.5))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.uniform(-60, 60, 3)
    return T


def simulate_scan(view_dir, seed: int, n_points: int = 250_000, noise: float = 0.02,
                  outlier_fraction: float = 0.02, add_table: bool = False, transform=None):
    """Returns (scan cloud with defects, clean ground-truth points of that scan in the GT frame).
    The clean points are returned before `transform` is applied."""
    rng = np.random.default_rng(seed)
    gt = ground_truth_mesh()
    pcd = gt.sample_points_uniformly(n_points * 2, use_triangle_normal=True)
    pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
    d = np.asarray(view_dir, float)
    d /= np.linalg.norm(d)
    keep = nrm @ d > -0.45  # partial view: one side plus the flanks, overlapping the other scan
    pts, nrm = pts[keep], nrm[keep]
    clean_pts = pts.copy()
    pts = pts + rng.normal(scale=noise, size=pts.shape)
    colors = surface_color(clean_pts)

    extra_pts = []
    lo, hi = pts.min(0) - 20, pts.max(0) + 20
    n_out = int(len(pts) * outlier_fraction)
    extra_pts.append(rng.uniform(lo, hi, size=(n_out, 3)))  # scattered noise
    blob_center = hi + np.array([10.0, 0, 0])
    extra_pts.append(blob_center + rng.normal(scale=1.5, size=(3000, 3)))  # floating debris cluster
    if add_table:
        g = rng.uniform([-90, -70], [90, 70], size=(120_000, 2))
        z = np.full(len(g), gt.get_min_bound()[2] - 0.2) + rng.normal(scale=noise, size=len(g))
        table = np.c_[g, z]
        inside = (table[:, 0] / 42) ** 2 + (table[:, 1] / 27) ** 2 < 1  # hidden under the object
        extra_pts.append(table[~inside])
    extra = np.vstack(extra_pts)

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([pts, extra])))
    cloud.colors = o3d.utility.Vector3dVector(np.vstack([colors, np.full((len(extra), 3), 0.5)]))
    if transform is not None:
        cloud.transform(transform)
    return cloud, clean_pts


def distance_to_gt(points: np.ndarray, gt: o3d.geometry.TriangleMesh | None = None) -> np.ndarray:
    gt = gt or ground_truth_mesh()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(gt.vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(gt.triangles, dtype=np.uint32)))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()


def three_view_matrix(eye, target, up=(0, 0, 1)) -> list[float]:
    """Replicates THREE.Camera.matrixWorldInverse.elements (column major) for a lookAt camera."""
    eye, target, up = (np.asarray(x, float) for x in (eye, target, up))
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    world = np.eye(4)
    world[:3, 0], world[:3, 1], world[:3, 2], world[:3, 3] = x, y, z, eye
    return np.linalg.inv(world).T.reshape(-1).tolist()


def render_photo(path, eye, fov: float, width: int = 800, height: int = 600, target=(0, 0, 0)):
    """Ray-trace the GT object with its colour pattern as a 'photo' from a three.js style camera."""
    gt = ground_truth_mesh()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(gt.vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(gt.triangles, dtype=np.uint32)))
    vm = three_view_matrix(eye, target)
    cam_to_world = np.linalg.inv(np.asarray(vm).reshape(4, 4).T)
    f = 1 / math.tan(math.radians(fov) / 2)
    xs, ys = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    ndc_x = xs / width * 2 - 1
    ndc_y = 1 - ys / height * 2
    dirs_cam = np.stack([ndc_x * (width / height) / f, ndc_y / f, -np.ones_like(xs)], axis=-1).reshape(-1, 3)
    dirs = dirs_cam @ cam_to_world[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origin = np.broadcast_to(np.asarray(eye, float), dirs.shape)
    res = scene.cast_rays(o3d.core.Tensor(np.hstack([origin, dirs]).astype(np.float32)))
    t = res["t_hit"].numpy()
    img = np.full((len(dirs), 3), [0.95, 0.95, 0.9])  # light background
    hit = np.isfinite(t)
    img[hit] = surface_color(origin[hit] + dirs[hit] * t[hit, None])
    Image.fromarray((img.reshape(height, width, 3) * 255).astype(np.uint8)).save(path)
    return {"view_matrix": vm, "fov": fov, "aspect": width / height}


# --------------------------------------------------------------------------- reference parts (accuracy checks)
# Closed meshes of exactly known size, centred on the origin, fine enough (< 1 um chord error at the default
# resolution) that sampling them is the only source of error.
def cylinder_mesh(diameter: float = 20.0, length: float = 60.0, segments: int = 1440) -> o3d.geometry.TriangleMesh:
    """Closed cylinder along z (flat end caps). Chord error of the 1440-gon: r (1 - cos(pi / 1440)) = 0.02 um."""
    return o3d.geometry.TriangleMesh.create_cylinder(radius=diameter / 2, height=length, resolution=segments,
                                                     split=max(1, int(length / 2)))


def box_mesh(size=(40.0, 30.0, 20.0)) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_box(*size)
    mesh.translate(-np.asarray(size, float) / 2)
    return mesh.subdivide_midpoint(4)


def sphere_pair_mesh(diameter: float = 20.0, distance: float = 60.0, resolution: int = 200):
    """Two spheres (a ball bar without the bar) with centres at x = -distance/2 and +distance/2."""
    out = o3d.geometry.TriangleMesh()
    for x in (-distance / 2, distance / 2):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=diameter / 2, resolution=resolution)
        s.translate([x, 0.0, 0.0])
        out += s
    return out


def iso_thread_radius(u: np.ndarray, pitch: float, major: float) -> np.ndarray:
    """Basic 60 deg profile (ISO 68-1 / Unified): crest flat P/8 at the major diameter, root flat P/4 at
    major - 1.0825 P (crest centred on integer u = axial phase in pitches)."""
    H = math.sqrt(3) / 2 * pitch
    d = np.abs(u - np.round(u)) * pitch
    return np.where(d <= pitch / 16, major / 2,
                    np.where(d >= 3 * pitch / 8, major / 2 - 5 * H / 8, major / 2 - (d - pitch / 16) * math.sqrt(3)))


def thread_rod_mesh(major: float = 25.4, pitch: float = 25.4 / 8, length: float = 60.0, hand: int = 1,
                    rows_per_pitch: int = 64, cols: int = 1440) -> o3d.geometry.TriangleMesh:
    """External thread along z centred on the origin (open tube: the thread surface only). Pitch diameter =
    major - 0.6495 P, minor = major - 1.0825 P."""
    rows = int(round(length / pitch * rows_per_pitch)) + 1
    t = np.linspace(-length / 2, length / 2, rows)
    th = np.linspace(0, 2 * math.pi, cols, endpoint=False)
    T, TH = np.meshgrid(t, th, indexing="ij")
    R = iso_thread_radius((T - hand * pitch * TH / (2 * math.pi)) / pitch, pitch, major)
    V = np.stack([R * np.cos(TH), R * np.sin(TH), T], axis=-1).reshape(-1, 3)
    i, j = (x.ravel() for x in np.meshgrid(np.arange(rows - 1), np.arange(cols), indexing="ij"))
    jn = (j + 1) % cols
    a, b, c, d = i * cols + j, i * cols + jn, (i + 1) * cols + j, (i + 1) * cols + jn
    F = np.vstack([np.c_[a, b, c], np.c_[b, d, c]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    mesh.compute_vertex_normals()
    return mesh


def scan_surface(mesh: o3d.geometry.TriangleMesh, spacing: float = 0.15, noise: float = 0.03, seed: int = 0,
                 view=None, view_limit: float = -0.2, outlier_fraction: float = 0.0, debris: bool = False,
                 table_z: float | None = None, with_normals: bool = True) -> o3d.geometry.PointCloud:
    """Scanner-like cloud of a reference mesh: area-uniform samples at `spacing`, Gaussian noise (1 sigma, each
    axis), optional one-sided view (keeps n . view > view_limit), stray outliers in the bounding box, a floating
    debris cluster and a table plane at z = table_z. Normals are the exact surface normals (outward)."""
    rng = np.random.default_rng(seed)
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.triangles, np.int64)
    cross = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area2 = np.linalg.norm(cross, axis=1)
    n = int(area2.sum() / 2 / spacing ** 2)
    tri = rng.choice(len(F), size=n, p=area2 / area2.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    pts = V[F[tri, 0]] * (1 - r1)[:, None] + V[F[tri, 1]] * (r1 * (1 - r2))[:, None] + V[F[tri, 2]] * (r1 * r2)[:, None]
    nrm = cross[tri] / np.maximum(area2[tri], 1e-300)[:, None]
    if np.einsum("ij,ij->", pts - V.mean(axis=0), nrm) < 0:
        nrm = -nrm
    if view is not None:
        d = np.asarray(view, float) / np.linalg.norm(view)
        keep = nrm @ d > view_limit
        pts, nrm = pts[keep], nrm[keep]
    pts = pts + rng.normal(scale=noise, size=pts.shape)
    extra, extra_n = [], []
    lo, hi = pts.min(axis=0) - 5, pts.max(axis=0) + 5
    if outlier_fraction > 0:
        k = int(len(pts) * outlier_fraction)
        extra.append(rng.uniform(lo, hi, size=(k, 3)))
        extra_n.append(np.tile([0.0, 0.0, 1.0], (k, 1)))
    if debris:
        extra.append(hi + np.array([8.0, 0, 0]) + rng.normal(scale=1.0, size=(2000, 3)))
        extra_n.append(np.tile([0.0, 0.0, 1.0], (2000, 1)))
    if table_z is not None:
        k = int((hi[0] - lo[0] + 40) * (hi[1] - lo[1] + 40) / spacing ** 2)
        g = rng.uniform([lo[0] - 20, lo[1] - 20], [hi[0] + 20, hi[1] + 20], size=(k, 2))
        table = np.c_[g, np.full(k, table_z) + rng.normal(scale=noise, size=k)]
        from scipy.spatial import cKDTree

        foot = np.zeros(k, bool)  # the part hides the table under its footprint
        low = pts[pts[:, 2] < table_z + 3.0]
        if len(low):
            dist, _ = cKDTree(low[:, :2]).query(g, k=1)
            foot = dist < 3 * spacing
        extra.append(table[~foot])
        extra_n.append(np.tile([0.0, 0.0, 1.0], ((~foot).sum(), 1)))
    if extra:
        pts = np.vstack([pts] + extra)
        nrm = np.vstack([nrm] + extra_n)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if with_normals:
        cloud.normals = o3d.utility.Vector3dVector(nrm)
    return cloud
