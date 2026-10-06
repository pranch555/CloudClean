"""Synthetic golden models and scans with known faults, for tests/test_golden.py."""
from __future__ import annotations

import numpy as np
import open3d as o3d

# a stepped tube, as (radius, height) corners of its cross-section: Ø30 x 8 at the bottom, Ø20 up to 20, a Ø10 hole
STEPPED_TUBE = [(5.0, 0.0), (15.0, 0.0), (15.0, 8.0), (10.0, 8.0), (10.0, 20.0), (5.0, 20.0)]


def revolve(profile, segments: int = 160) -> o3d.geometry.TriangleMesh:
    """A closed, welded mesh made by turning a closed (radius, height) outline around Z - like a CAD tessellation of
    a turned part: flat rings, round walls."""
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = [np.c_[r * np.cos(a), r * np.sin(a), np.full(segments, z)] for r, z in profile]
    V = np.vstack(rings)
    idx = [np.arange(segments) + k * segments for k in range(len(profile))]
    F = []
    for k in range(len(profile)):
        lo, hi = idx[k], idx[(k + 1) % len(profile)]
        F += [np.c_[lo, np.roll(lo, -1), np.roll(hi, -1)], np.c_[lo, np.roll(hi, -1), hi]]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.vstack(F)))
    mesh.orient_triangles()
    mesh.compute_triangle_normals()
    return mesh


def stepped_tube() -> o3d.geometry.TriangleMesh:
    return revolve(STEPPED_TUBE)


# Ø30 x 20 with a Ø6 hole through it and a Ø16 counterbore 6 deep from the top (its floor: a recess floor at z = 14)
COUNTERBORED_TUBE = [(3.0, 0.0), (15.0, 0.0), (15.0, 20.0), (8.0, 20.0), (8.0, 14.0), (3.0, 14.0)]


def counterbored_tube() -> o3d.geometry.TriangleMesh:
    return revolve(COUNTERBORED_TUBE)


# a socket head screw standing on its tip: Ø12 shaft from z = 0 to 40, Ø20 head up to 52, a hex socket 8 across flats
# and 6 deep in the top (its bottom at z = 46)
SCREW = {"shaft_r": 6.0, "head_z": 40.0, "head_r": 10.0, "top_z": 52.0, "hex_af": 8.0, "hex_depth": 6.0}


def _ring(radius, z: float, angles: np.ndarray) -> np.ndarray:
    r = np.broadcast_to(np.asarray(radius, dtype=np.float64), angles.shape)
    return np.c_[r * np.cos(angles), r * np.sin(angles), np.full(len(angles), z)]


def _strip(lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
    """Triangles joining two rings of vertex indices (same count, same angles)."""
    return [np.c_[lo, np.roll(lo, -1), np.roll(hi, -1)], np.c_[lo, np.roll(hi, -1), hi]]


def screw(segments: int = 96) -> o3d.geometry.TriangleMesh:
    """A closed, welded mesh of SCREW. The hex socket's six walls are flat (every ring has `segments` points at the
    same angles; the hex's corners, at 30° + k 60°, fall on them)."""
    s = SCREW
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    flat = np.radians(60.0) * np.round(a / np.radians(60.0))          # the nearest flat's normal
    hex_r = (s["hex_af"] / 2) / np.cos(a - flat)
    floor_z = s["top_z"] - s["hex_depth"]
    rings = [_ring(s["shaft_r"], 0.0, a), _ring(s["shaft_r"], s["head_z"], a), _ring(s["head_r"], s["head_z"], a),
             _ring(s["head_r"], s["top_z"], a), _ring(hex_r, s["top_z"], a), _ring(hex_r, floor_z, a)]
    V = np.vstack([[[0.0, 0.0, 0.0]], *rings, [[0.0, 0.0, floor_z]]])
    idx = [1 + k * segments + np.arange(segments) for k in range(len(rings))]
    F = [np.c_[np.zeros(segments, dtype=int), np.roll(idx[0], -1), idx[0]]]        # the tip
    for lo, hi in zip(idx, idx[1:]):
        F += _strip(lo, hi)
    F.append(np.c_[np.full(segments, len(V) - 1), idx[-1], np.roll(idx[-1], -1)])  # the socket's bottom
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.vstack(F)))
    mesh.orient_triangles()
    mesh.compute_triangle_normals()
    return mesh


def hex_bolt(segments: int = 96, across_flats: float = 16.0, head_h: float = 6.0, shaft_r: float = 5.0,
             length: float = 40.0) -> o3d.geometry.TriangleMesh:
    """A hex head bolt standing on its tip: a Ø10 shaft `length` long under a hex head `across_flats` wide."""
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    flat = np.radians(60.0) * np.round(a / np.radians(60.0))
    hex_r = (across_flats / 2) / np.cos(a - flat)
    top = length + head_h
    rings = [_ring(shaft_r, 0.0, a), _ring(shaft_r, length, a), _ring(hex_r, length, a), _ring(hex_r, top, a)]
    V = np.vstack([[[0.0, 0.0, 0.0]], *rings, [[0.0, 0.0, top]]])
    idx = [1 + k * segments + np.arange(segments) for k in range(len(rings))]
    F = [np.c_[np.zeros(segments, dtype=int), np.roll(idx[0], -1), idx[0]]]
    for lo, hi in zip(idx, idx[1:]):
        F += _strip(lo, hi)
    F.append(np.c_[np.full(segments, len(V) - 1), idx[-1], np.roll(idx[-1], -1)])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.vstack(F)))
    mesh.orient_triangles()
    mesh.compute_triangle_normals()
    return mesh


def screw_scan(n: int = 300_000, noise: float = 0.01, seed: int = 2, floor_drop: float = 0.4,
               corner_gap: float = 1.6, pose: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    """A scan of screw() with known faults:
      the hex socket's bottom `floor_drop` deeper (it reads like a drill point),
      the corner under the head not scanned within `corner_gap` of it (the head shades it)."""
    s = SCREW
    rng = np.random.default_rng(seed)
    P = sample_surface(screw(), n, seed)
    floor_z = s["top_z"] - s["hex_depth"]
    r = np.hypot(P[:, 0], P[:, 1])
    floor = np.abs(P[:, 2] - floor_z) < 1e-6
    P[:, 2] += rng.normal(0, noise, len(P))
    P[floor, 2] -= floor_drop
    corner = np.hypot(np.maximum(r - s["shaft_r"], 0), s["head_z"] - P[:, 2]) < corner_gap
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[~corner]))
    if pose is not None:
        cloud.transform(pose)
    return cloud


# a block with no main axis (36 x 30 x 20, Z up) and a pocket 16 x 10 x 6 deep in its top
POCKET_BLOCK = {"size": (36.0, 30.0, 20.0), "pocket": ((10.0, 10.0), (26.0, 20.0)), "depth": 6.0}


def pocket_block() -> o3d.geometry.TriangleMesh:
    """A closed mesh of POCKET_BLOCK, drawn with a few big triangles like a CAD tessellation."""
    (X, Y, Z), ((x0, y0), (x1, y1)), d = POCKET_BLOCK["size"], POCKET_BLOCK["pocket"], POCKET_BLOCK["depth"]
    square = lambda a0, b0, a1, b1, z: [(a0, b0, z), (a1, b0, z), (a1, b1, z), (a0, b1, z)]
    V = np.array(square(0, 0, X, Y, 0) + square(0, 0, X, Y, Z) + square(x0, y0, x1, y1, Z)
                 + square(x0, y0, x1, y1, Z - d), dtype=np.float64)
    bottom, top, rim, floor = (np.arange(4) + 4 * k for k in range(4))
    F = [[0, 2, 1], [0, 3, 2]]
    for k in range(4):
        n_ = (k + 1) % 4
        F += [[bottom[k], bottom[n_], top[n_]], [bottom[k], top[n_], top[k]]]             # the sides
        F += [[top[k], top[n_], rim[n_]], [top[k], rim[n_], rim[k]]]                       # the top, round the pocket
        F += [[rim[k], rim[n_], floor[n_]], [rim[k], floor[n_], floor[k]]]                 # the pocket's walls
    F += [[floor[0], floor[1], floor[2]], [floor[0], floor[2], floor[3]]]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.array(F)))
    mesh.orient_triangles()
    mesh.compute_triangle_normals()
    return mesh


def pocket_block_scan(n: int = 250_000, noise: float = 0.01, seed: int = 3,
                      pose: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(seed)
    P = sample_surface(pocket_block(), n, seed) + rng.normal(0, noise, (n, 3))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    if pose is not None:
        cloud.transform(pose)
    return cloud


def sample_cloud(mesh: o3d.geometry.TriangleMesh, n: int, noise: float = 0.01, seed: int = 4) -> o3d.geometry.PointCloud:
    """A plain scan of a mesh: area-uniform points with a little noise."""
    P = sample_surface(mesh, n, seed) + np.random.default_rng(seed).normal(0, noise, (n, 3))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))


def sample_surface(mesh: o3d.geometry.TriangleMesh, n: int, seed: int = 0) -> np.ndarray:
    """Area-uniform points on the mesh."""
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    rng = np.random.default_rng(seed)
    t = rng.choice(len(F), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return V[F[t, 0]] * (1 - s)[:, None] + V[F[t, 1]] * (s * (1 - r2))[:, None] + V[F[t, 2]] * (s * r2)[:, None]


def stepped_tube_scan(n: int = 400_000, noise: float = 0.01, seed: int = 1, hole_grow: float = 0.08,
                      bump: float = 0.25, rough: float = 0.15, missing_bottom: bool = True,
                      pose: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    """A scan of stepped_tube() with known faults:
      the bottom face not scanned (missing_bottom),
      the Ø10 hole `hole_grow` bigger in radius (the hole reads 10 + 2 * hole_grow),
      a patch on the top face `bump` high (x > 6, |y| < 3),
      a rough patch on the Ø30 wall (points scatter by `rough`, around +Y, 2 < z < 6)."""
    rng = np.random.default_rng(seed)
    P = sample_surface(stepped_tube(), n, seed)
    r = np.hypot(P[:, 0], P[:, 1])
    ang = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    flat = (np.abs(P[:, 2]) < 1e-6) | (np.abs(P[:, 2] - 8) < 1e-6) | (np.abs(P[:, 2] - 20) < 1e-6)
    bottom = np.abs(P[:, 2]) < 1e-6
    top = np.abs(P[:, 2] - 20) < 1e-6
    hole = (np.abs(r - 5) < 0.01) & ~flat
    wall30 = (np.abs(r - 15) < 0.01) & ~flat
    radial = np.c_[P[:, 0] / r, P[:, 1] / r, np.zeros(len(P))]
    along = np.where(flat[:, None], [[0.0, 0.0, 1.0]], radial)
    P = P + along * rng.normal(0, noise, (len(P), 1))
    P[hole] += radial[hole] * hole_grow
    patch = top & (P[:, 0] > 6) & (np.abs(P[:, 1]) < 3)
    P[patch, 2] += bump
    scatter = wall30 & (np.abs(ang - 90) < 20) & (P[:, 2] > 2) & (P[:, 2] < 6)
    P[scatter] += radial[scatter] * rng.normal(0, rough, (scatter.sum(), 1))
    keep = ~bottom if missing_bottom else np.ones(len(P), dtype=bool)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[keep]))
    if pose is not None:
        cloud.transform(pose)
    return cloud
