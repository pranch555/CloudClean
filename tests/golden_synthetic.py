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
