"""Coverage, density and hole analysis of a live capture map, plus plain-language scanning hints.

How completeness is estimated (robust and cheap enough to run about once a second):

1. The fused points are counted on a surface-voxel grid of ~2.5 mm cells. A cell with >= 2 points is
   observed surface. Its density ratio = points per mm^2 (count / (cell^2 / 1.5), the mean area of a
   surface patch inside an occupied cube) divided by the target density 1 / point_distance^2.
2. The unobserved part of the closed surface is estimated with the **convex hull** of the observed cells:
   the hull is sampled at cell spacing and samples farther than 3 cells from any observed cell are candidate
   missing surface. Concave regions are handled with **free-space carving**: every fused frame marks coarse
   cells its rays passed through (stopping 3 cells before the measured point). A hull sample in space the
   sensor has seen through is not missing surface, it is air across a concavity.
3. completeness = dense area / (observed area + missing area).

Holes are clusters (connectivity within 3 cells) of under-sampled cells, open boundary cells (cells whose
neighbours lie on one side in the tangent plane) and missing hull samples. Each cluster gets a suggested
viewing direction: the area-weighted mean outward normal (surface normals are oriented towards the sensors
that saw them, hull normals point outwards; flipped when it points into the hull)."""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

from .grid import _lookup, key_centers, neighbor_keys, sorted_getter, voxel_keys

DENSE_RATIO = 0.7         # a cell counts as covered at 70 % of the target density
SPARSE_RATIO = 0.35       # clearly under-sampled
MIN_CELL_POINTS = 2
AREA_FACTOR = 1.5         # occupied cubes per unit of surface area / cell^2


def cell_area(cell: float) -> float:
    return cell * cell / AREA_FACTOR


def density_ratio(counts, cell: float, target_per_mm2: float) -> np.ndarray:
    return np.asarray(counts, dtype=np.float32) / np.float32(cell_area(cell) * max(target_per_mm2, 1e-9))


def smoothed_density(get_counts, keys, cell: float) -> np.ndarray:
    """Points per mm^2 around each cell: counts of the occupied 3x3x3 block / (occupied cells * cell area).

    A single cell is biased low where the surface only clips its corner; the block average is not."""
    keys = np.asarray(keys, np.int64)
    if len(keys) == 0:
        return np.zeros(0, np.float32)
    c = get_counts(neighbor_keys(keys).reshape(-1)).reshape(len(keys), 27)
    occupied = c >= MIN_CELL_POINTS
    total = np.where(occupied, c, 0).sum(axis=1)
    return (total / (np.maximum(occupied.sum(axis=1), 1) * cell_area(cell))).astype(np.float32)


def _normals(centers: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """PCA normals and in-plane offset of the neighbourhood centroid (large at open boundaries)."""
    k = min(12, len(centers))
    dist, idx = cKDTree(centers).query(centers, k=k, workers=-1)
    nb = centers[idx]
    close = (dist <= 3.0 * cell)[:, :, None]
    weight = close.sum(axis=1)
    mean = (nb * close).sum(axis=1) / np.maximum(weight, 1)
    X = (nb - mean[:, None, :]) * close
    cov = np.einsum("nki,nkj->nij", X, X) / np.maximum(weight[:, :, None], 1)
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]
    off = mean - centers
    tangent = off - np.einsum("ij,ij->i", off, normals)[:, None] * normals
    few = weight[:, 0] < 5
    tangent[few] = 0.0
    return normals, tangent


def _hull_samples(hull: ConvexHull, spacing: float, max_samples: int = 150_000):
    pts = hull.points
    tri = pts[hull.simplices]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    sample_area = max(spacing * spacing, float(area.sum()) / max_samples)
    n = np.maximum(np.round(area / sample_area).astype(int), 0)
    rng = np.random.default_rng(0)       # deterministic so holes do not flicker between updates
    face = np.repeat(np.arange(len(tri)), n)
    r1, r2 = rng.random(len(face)), rng.random(len(face))
    s = np.sqrt(r1)
    a, b, c = tri[face, 0], tri[face, 1], tri[face, 2]
    samples = a * (1 - s)[:, None] + b * (s * (1 - r2))[:, None] + c * (s * r2)[:, None]
    normals = hull.equations[face, :3]
    return samples, normals, sample_area


def _inside(hull: ConvexHull, points: np.ndarray) -> np.ndarray:
    return np.all(points @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-9, axis=1)


# --------------------------------------------------------------------------- hints
def _axis_name(d: np.ndarray) -> str:
    names = {0: ("+X side", "-X side"), 1: ("+Y side", "-Y side"), 2: ("top", "underside")}
    i = int(np.argmax(np.abs(d)))
    return names[i][0 if d[i] > 0 else 1]


def view_hint(direction, center, object_center, sensor_position=None, turntable: bool = False,
              kind: str = "missing") -> str:
    """Plain-language advice how to see a region whose outward normal is `direction` (world Z is up)."""
    d = np.asarray(direction, float)
    d = d / max(np.linalg.norm(d), 1e-9)
    if d[2] < -0.75:
        return "Flip the part over and start a second scan to capture the underside"
    parts = []
    horiz = math.hypot(d[0], d[1])
    if sensor_position is not None:
        s = np.asarray(sensor_position, float) - np.asarray(object_center, float)
        if horiz > 0.35:
            delta = math.degrees(math.atan2(d[1], d[0]) - math.atan2(s[1], s[0]))
            delta = (delta + 180.0) % 360.0 - 180.0
            if abs(delta) >= 25.0:
                amount = int(round(abs(delta) / 15.0) * 15)
                if turntable:  # rotating the part by -delta brings the region in front of the scanner
                    turn = "counter-clockwise" if -delta > 0 else "clockwise"
                    parts.append(f"rotate the part ~{amount}° {turn} (seen from above)")
                else:
                    parts.append(f"move the scanner ~{amount}° to your {'right' if delta > 0 else 'left'} "
                                 "around the part")
        el_s = math.degrees(math.asin(np.clip(s[2] / max(np.linalg.norm(s), 1e-9), -1, 1)))
        el_d = math.degrees(math.asin(np.clip(d[2], -1, 1)))
        if d[2] > 0.75 and el_s < 60:
            parts.append("scan more from above to capture the top")
        elif d[2] < -0.3:
            parts.append("tilt the part (or the scanner lower) to see under the edge")
        elif el_s - el_d > 30:
            parts.append("lower the scanner to look at it more from the side")
        elif el_d - el_s > 30:
            parts.append("raise the scanner to look down on it")
    else:
        parts.append(f"scan the {_axis_name(d)} of the part (facing {'+' if d[np.argmax(np.abs(d))] > 0 else '-'}"
                     f"{'XYZ'[int(np.argmax(np.abs(d)))]})")
    if not parts:
        parts.append("keep the scanner on this area a little longer" if kind == "sparse"
                     else "aim the scanner straight at this area")
    text = " and ".join(parts)
    return text[0].upper() + text[1:]


# --------------------------------------------------------------------------- analysis
def analyze(keys, counts, view_vectors, cell: float, target_per_mm2: float, free_keys=None,
            free_cell: float | None = None, sensor_position=None, turntable: bool = False,
            max_holes: int = 6) -> dict:
    """Coverage / density / hole report for the density grid (keys, counts, summed view vectors)."""
    keys = np.asarray(keys, np.int64)
    counts = np.asarray(counts)
    apc = cell_area(cell)
    surface = counts >= MIN_CELL_POINTS
    ratio_all = np.zeros(len(keys), np.float32)
    ratio_all[surface] = smoothed_density(sorted_getter(keys, counts), keys[surface], cell) / max(target_per_mm2, 1e-9)
    n_surface = int(surface.sum())
    dense = surface & (ratio_all >= DENSE_RATIO)
    observed_area = n_surface * apc
    dense_area = int(dense.sum()) * apc
    out = {"cell_mm": cell, "target_per_mm2": round(float(target_per_mm2), 3),
           "cells": n_surface, "dense_cells": int(dense.sum()),
           "median_ratio": round(float(np.median(ratio_all[surface])), 3) if n_surface else 0.0,
           "observed_area_mm2": round(observed_area, 1), "covered_area_mm2": round(dense_area, 1),
           "missing_area_mm2": 0.0, "estimated_area_mm2": round(observed_area, 1),
           "completeness": 0.0, "method": "surface cells + convex hull with free-space carving", "holes": []}
    if n_surface < 20:
        return out

    C = key_centers(keys[surface], cell)
    ratio = ratio_all[surface]
    object_center = C.mean(axis=0)
    normals, tangent = _normals(C, cell)
    if view_vectors is not None:
        view = np.asarray(view_vectors, float)[surface]
        has_view = np.linalg.norm(view, axis=1) > 1e-6
    else:
        view, has_view = np.zeros_like(C), np.zeros(len(C), bool)
    toward = np.where(has_view[:, None], view, C - object_center)
    flip = np.einsum("ij,ij->i", normals, toward) < 0
    normals[flip] *= -1

    # --- missing surface from the convex hull
    hull = None
    missing_pts = np.empty((0, 3))
    missing_nrm = np.empty((0, 3))
    sample_area = apc
    try:
        hull = ConvexHull(C)
    except Exception:
        hull = None
    if hull is not None:
        samples, s_normals, sample_area = _hull_samples(hull, cell)
        if len(samples):
            dist, _ = cKDTree(C).query(samples, k=1, distance_upper_bound=3.0 * cell, workers=-1)
            miss = ~np.isfinite(dist)
            if free_keys is not None and len(free_keys) and free_cell:
                _, carved = _lookup(np.asarray(free_keys, np.int64), voxel_keys(samples[miss], free_cell))
                idx = np.flatnonzero(miss)
                miss[idx[carved]] = False
            missing_pts, missing_nrm = samples[miss], s_normals[miss]
    missing_area = len(missing_pts) * sample_area
    estimated = observed_area + missing_area
    out.update(missing_area_mm2=round(missing_area, 1), estimated_area_mm2=round(estimated, 1),
               completeness=round(float(dense_area / estimated), 4) if estimated > 0 else 0.0)

    # --- hole clusters
    boundary = np.linalg.norm(tangent, axis=1) > 0.45 * cell
    sparse = ratio < SPARSE_RATIO
    b_dir = normals - tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    b_dir /= np.maximum(np.linalg.norm(b_dir, axis=1, keepdims=True), 1e-9)
    cand_pts = [C[sparse], C[boundary & ~sparse], missing_pts]
    cand_dir = [normals[sparse], b_dir[boundary & ~sparse], missing_nrm]
    cand_w = [np.full(int(sparse.sum()), apc), np.full(int((boundary & ~sparse).sum()), 0.5 * apc),
              np.full(len(missing_pts), sample_area)]
    cand_kind = [np.full(len(p), k) for p, k in zip(cand_pts, (0, 1, 2))]
    P = np.vstack(cand_pts)
    if len(P) == 0:
        return out
    D, W, K = np.vstack(cand_dir), np.concatenate(cand_w), np.concatenate(cand_kind)
    pairs = cKDTree(P).query_pairs(3.0 * cell, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    n_comp, labels = connected_components(graph, directed=False)
    area = np.bincount(labels, weights=W, minlength=n_comp)
    min_area = max(4.0 * cell * cell, 0.003 * estimated)
    holes = []
    for comp in np.argsort(-area):
        if area[comp] < min_area or len(holes) >= max_holes:
            break
        m = labels == comp
        w = W[m]
        center = (P[m] * w[:, None]).sum(axis=0) / w.sum()
        direction = (D[m] * w[:, None]).sum(axis=0)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0.2 * w.sum() else (center - object_center) / max(
            np.linalg.norm(center - object_center), 1e-9)
        if hull is not None:
            step = 2.0 * cell
            if _inside(hull, (center + direction * step)[None])[0] and \
                    not _inside(hull, (center - direction * step)[None])[0]:
                direction = -direction
        kind_area = np.bincount(K[m], weights=w, minlength=3)
        kind = ("sparse", "edge", "missing")[int(np.argmax(kind_area))]
        hint = view_hint(direction, center, object_center, sensor_position, turntable, kind)
        label = {"sparse": "Thin coverage", "edge": "Open edge", "missing": "Unscanned area"}[kind]
        holes.append({"id": len(holes), "kind": kind, "center": center.round(2).tolist(),
                      "direction": direction.round(3).tolist(), "area_mm2": round(float(area[comp]), 1),
                      "region": _axis_name(direction), "hint": hint,
                      "message": f"{label} (~{area[comp]:.0f} mm²) on the {_axis_name(direction)}: {hint}"})
    out["holes"] = holes
    out["object_center"] = object_center.round(2).tolist()
    return out
