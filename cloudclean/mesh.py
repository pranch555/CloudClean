"""Surface reconstruction with scale-preserving trimming and accuracy reporting."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .clean import ensure_normals
from .io import Geometry, describe, drop_blank_colors, estimate_spacing, to_cloud
from .params import ParamsMixin

Log = Callable[[str], None]


@dataclass
class MeshParams(ParamsMixin):
    method: str = "poisson"               # poisson (smooth, hole-filling) | bpa (ball pivoting, exact points)
    depth: int = 0                        # Poisson octree depth; 0 -> auto from point spacing
    max_auto_depth: int = 11              # cap for auto depth (12+ needs lots of RAM)
    linear_fit: bool = True               # Poisson linear interpolation (better surface position)
    watertight: bool = False              # keep Poisson's closed surface (fills unscanned holes)
    trim: str = "distance"                # distance | density | none (ignored when watertight)
    trim_distance_multiplier: float = 3.0 # remove surface farther than this x spacing from any scan point
    density_quantile: float = 0.02
    min_component_ratio: float = 0.02     # drop mesh islands smaller than this fraction of the largest
    smooth_iterations: int = 0            # Taubin smoothing (volume preserving)
    target_triangles: int = 0             # >0: decimate to this many triangles
    transfer_colors: bool = True          # copy scan colours to mesh vertices


def poisson_threads() -> int:
    """Open3D's multi-threaded Poisson solver fails ("Failed to close loop") or segfaults in the Linux aarch64
    builds (tested on DGX Spark, Open3D 0.19 main-devel), so ARM64 uses one thread: depth 11 on 1M points still
    takes ~30 s there. CLOUDCLEAN_POISSON_THREADS overrides (-1 = all cores)."""
    import os
    import platform

    env = os.environ.get("CLOUDCLEAN_POISSON_THREADS")
    if env:
        return int(env)
    return 1 if platform.system() == "Linux" and platform.machine().lower() in ("aarch64", "arm64") else -1


def auto_depth(extent: float, spacing: float, cap: int) -> int:
    if spacing <= 0:
        return 9
    return int(min(max(math.ceil(math.log2(extent * 1.1 / spacing)), 6), cap))


def transfer_colors(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, k: int = 4) -> None:
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    d, idx = cKDTree(pts).query(np.asarray(mesh.vertices), k=k, workers=-1)
    w = 1.0 / (d + 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.einsum("nk,nkc->nc", w, cols[idx]))


def surface_deviation(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud,
                      sample: int = 200_000) -> dict:
    """Distance from scan points to the mesh surface (how faithfully the mesh follows the scan)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.triangles, dtype=np.uint32)))
    pts = np.asarray(pcd.points)
    if len(pts) > sample:
        pts = pts[np.random.default_rng(0).choice(len(pts), sample, replace=False)]
    d = scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()
    return {"mean": float(d.mean()), "rms": float(np.sqrt((d ** 2).mean())),
            "p95": float(np.percentile(d, 95)), "max": float(d.max()), "samples": int(len(d))}


def cleanup_mesh(mesh: o3d.geometry.TriangleMesh, min_component_ratio: float, log: Log = print):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        return mesh
    clusters, _, areas = mesh.cluster_connected_triangles()
    clusters, areas = np.asarray(clusters), np.asarray(areas)
    small = areas < min_component_ratio * areas.max()
    if small.any():
        mesh.remove_triangles_by_mask(small[clusters])
        mesh.remove_unreferenced_vertices()
        log(f"  removed {int(small.sum())} small disconnected pieces (kept {int((~small).sum())})")
    return mesh


def reconstruct_mesh(geom: Geometry, params: MeshParams | None = None,
                     log: Log = print) -> tuple[o3d.geometry.TriangleMesh, dict]:
    p = params or MeshParams()
    pcd = o3d.geometry.PointCloud(to_cloud(geom))
    spacing = estimate_spacing(pcd.points)
    ensure_normals(pcd, spacing, log=log)
    extent = float(np.max(pcd.get_max_bound() - pcd.get_min_bound()))
    log(f"Meshing {len(pcd.points):,} points with {p.method} (spacing {spacing:.5f})")
    report: dict = {"method": p.method, "spacing": spacing}

    if p.method == "poisson":
        depth = p.depth or auto_depth(extent, spacing, p.max_auto_depth)
        grid = extent * 1.1 / 2 ** depth
        log(f"  Poisson depth {depth} (grid cell {grid:.5f})")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, width=0, scale=1.1, linear_fit=p.linear_fit, n_threads=poisson_threads())
        densities = np.asarray(densities)
        report.update(depth=depth, grid_cell=grid)
        trim = "none" if p.watertight else p.trim
        if trim == "distance":
            thr = max(p.trim_distance_multiplier * spacing, 1.5 * grid)
            d, _ = cKDTree(np.asarray(pcd.points)).query(np.asarray(mesh.vertices), k=1, workers=-1)
            mesh.remove_vertices_by_mask(d > thr)
            log(f"  trimmed surface farther than {thr:.5f} from the scan ({int((d > thr).sum()):,} vertices)")
        elif trim == "density":
            mesh.remove_vertices_by_mask(densities < np.quantile(densities, p.density_quantile))
        report["trim"] = trim
    elif p.method == "bpa":
        radii = [spacing * r for r in (1.5, 3.0, 6.0)]
        log(f"  ball radii {', '.join(f'{r:.5f}' for r in radii)}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
    else:
        raise ValueError(f"Unknown mesh method '{p.method}' (use poisson or bpa)")

    mesh = cleanup_mesh(mesh, p.min_component_ratio, log)
    if len(mesh.triangles) == 0:
        raise RuntimeError("Meshing produced no triangles - check that the cloud has valid normals")

    if p.smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=p.smooth_iterations)
        log(f"  Taubin smoothing x{p.smooth_iterations}")
    if p.target_triangles and len(mesh.triangles) > p.target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=p.target_triangles)
        mesh = cleanup_mesh(mesh, 0.0, log)
        log(f"  decimated to {len(mesh.triangles):,} triangles")

    mesh.compute_vertex_normals()
    if p.transfer_colors and pcd.has_colors():
        transfer_colors(mesh, pcd)
        log("  transferred scan colours to mesh")
    drop_blank_colors(mesh)

    report["deviation"] = surface_deviation(mesh, pcd)
    report["mesh"] = describe(mesh)
    report["cloud_dimensions"] = (pcd.get_max_bound() - pcd.get_min_bound()).round(4).tolist()
    report["params"] = p.to_dict()
    dev = report["deviation"]
    log(f"Mesh done: {len(mesh.triangles):,} triangles, deviation mean {dev['mean']:.5f} "
        f"/ p95 {dev['p95']:.5f}, size {report['mesh']['dimensions']}")
    return mesh, report
