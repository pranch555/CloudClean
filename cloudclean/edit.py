"""Geometry editing: crop, cut, screen selection, exact transforms, alignment, mesh repair and local
smoothing / denoising / spike removal.

Every operation works on a private copy of the input, validates its arguments up front and reports
before/after counts. Positional operations are applied in float64 with numpy (normals use the inverse
transpose, mirrored meshes get their winding fixed) so nothing moves by more than rounding. Smoothing ops keep
the topology, only touch their (feathered) region and report how far they moved the geometry."""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import regions
from .io import Geometry, estimate_spacing, is_cloud

Log = Callable[[str], None]
OnOp = Callable[[int, int, str], None]

PC, MESH = "pointcloud", "mesh"
_BOTH, _PC_ONLY, _MESH_ONLY = [PC, MESH], [PC], [MESH]


# --------------------------------------------------------------------------- catalogue
def _arg(type_: str, default: Any = None, description: str = "", choices: list | None = None,
         required: bool = False) -> dict:
    spec = {"type": type_, "default": default, "description": description, "required": required}
    if choices:
        spec["choices"] = choices
    return spec


_CENTER_HELP = '"origin", "centroid", "bbox" (bounding-box centre) or [x, y, z]'
_REGION_HELP = regions.REGION_HELP  # all region kinds: see cloudclean/regions.py
_FEATHER_HELP = ("Blend width outside the region in scan units: the effect fades smoothly to zero across it so "
                 "there is no step (0 = 3x point spacing)")
_STRENGTH_HELP = "How far to move towards the smoothed position per iteration (0..1]"
# Measured on parts of known size (docs/accuracy-investigation-2026-09-24.md): these filters change feature sizes.
DENOISE_CLOUD_WARNING = ("On point clouds it changes feature sizes: thread crests came out 0.13 mm lower and a 20 mm "
                         "ball 0.02 mm smaller in tests - do not use it on surfaces you are going to measure (smooth, "
                         "method taubin, changed them by 0.008 mm or less).")
LAPLACIAN_MESH_WARNING = ("Laplacian smoothing of a mesh shrinks features: thread crests came out 0.04 mm lower after "
                          "5 iterations in tests - use taubin on surfaces you are going to measure.")

# Argument types: number, integer, boolean, choice, vec3, axis ("x"|"y"|"z" or vec3),
# center (named point or vec3), matrix4 (4x4 row-major, nested or flat 16), matrix16 (flat, column-major),
# polygon ([[x, y], ...]), region (see _REGION_HELP; null = everything).
OP_CATALOGUE: dict[str, dict] = {
    "crop_box": {
        "description": "Keep only what lies inside an axis-aligned box (or delete it with invert). "
                       "Points are never moved; mesh triangles touching a removed vertex are dropped.",
        "applies_to": _BOTH,
        "args": {"min": _arg("vec3", None, "Box minimum corner [x, y, z] in scan units", required=True),
                 "max": _arg("vec3", None, "Box maximum corner [x, y, z] in scan units", required=True),
                 "invert": _arg("boolean", False, "Delete the inside of the box instead of keeping it")},
    },
    "cut_plane": {
        "description": "Cut with a plane and keep one side (points on the plane are kept).",
        "applies_to": _BOTH,
        "args": {"point": _arg("vec3", None, "Any point on the plane", required=True),
                 "normal": _arg("vec3", None, "Plane normal (need not be unit length)", required=True),
                 "keep": _arg("choice", "positive", "Side to keep relative to the normal",
                              ["positive", "negative"])},
    },
    "select_screen": {
        "description": "Delete or keep what falls inside a lasso/rectangle drawn on screen. "
                       "A mesh triangle is selected when all its vertices are.",
        "applies_to": _BOTH,
        "args": {"view_projection": _arg("matrix16", None, "three.js projectionMatrix * matrixWorldInverse "
                                         "(16 numbers, column-major)", required=True),
                 "polygon": _arg("polygon", None, "Selection outline [[x, y], ...] in NDC (-1..1, y up)",
                                 required=True),
                 "mode": _arg("choice", "delete", "Delete the selection or keep only the selection",
                              ["delete", "keep"]),
                 "visible_only": _arg("boolean", False, "Only select surfaces visible from the camera "
                                      "(not what is hidden behind them)")},
    },
    "delete_sphere": {
        "description": "Delete everything within a sphere.",
        "applies_to": _BOTH,
        "args": {"center": _arg("vec3", None, "Sphere centre [x, y, z]", required=True),
                 "radius": _arg("number", None, "Sphere radius in scan units", required=True)},
    },
    "delete_region": {
        "description": "Delete everything inside a region: a brush stroke, a world or oriented box, a cylinder, a "
                     "slab or end of the part along a world or part axis (e.g. the last 10 mm along the length), a "
                     "screen selection, or the inverse of any of these. Points are never moved; a mesh loses the "
                     "region's vertices and every triangle that uses one.",
        "applies_to": _BOTH,
        "args": {"region": _arg("region", None, f"What to delete: {_REGION_HELP}", required=True)},
    },
    "keep_region": {
        "description": "Keep only what is inside a region (any region kind, see delete_region) and delete the rest. "
                     "Points are never moved; a mesh keeps the triangles whose vertices are all inside.",
        "applies_to": _BOTH,
        "args": {"region": _arg("region", None, f"What to keep: {_REGION_HELP}", required=True)},
    },
    "transform": {
        "description": "Apply a 4x4 affine matrix (row-major, last row 0 0 0 1).",
        "applies_to": _BOTH,
        "args": {"matrix": _arg("matrix4", None, "4x4 matrix, row-major ([[...], ...] or 16 numbers)",
                                required=True)},
    },
    "translate": {
        "description": "Move by an offset.",
        "applies_to": _BOTH,
        "args": {"offset": _arg("vec3", None, "Offset [dx, dy, dz] in scan units", required=True)},
    },
    "rotate": {
        "description": "Rotate about an axis through a centre (right-hand rule).",
        "applies_to": _BOTH,
        "args": {"axis": _arg("axis", None, '"x", "y", "z" or a direction [x, y, z]', required=True),
                 "degrees": _arg("number", None, "Rotation angle in degrees", required=True),
                 "center": _arg("center", "centroid", f"Rotation centre: {_CENTER_HELP}")},
    },
    "scale": {
        "description": "Uniformly scale. Changes the size of the data - only use when explicitly requested "
                       "(e.g. unit conversion).",
        "applies_to": _BOTH,
        "args": {"factor": _arg("number", None, "Scale factor (> 0), e.g. 1000 for m -> mm", required=True),
                 "center": _arg("center", "origin", f"Fixed point of the scaling: {_CENTER_HELP}")},
    },
    "mirror": {
        "description": "Mirror across a plane perpendicular to an axis (mesh winding is fixed so normals "
                       "still point outward).",
        "applies_to": _BOTH,
        "args": {"axis": _arg("choice", None, "Axis that gets negated", ["x", "y", "z"], required=True),
                 "center": _arg("center", "bbox", f"Point on the mirror plane: {_CENTER_HELP}")},
    },
    "center": {
        "description": "Translate so that the object is centred at the origin.",
        "applies_to": _BOTH,
        "args": {"mode": _arg("choice", "bbox", "bbox centre, centroid, or bbox_bottom (centred horizontally, "
                              "resting on 0 along the up axis)", ["bbox", "centroid", "bbox_bottom"]),
                 "up": _arg("choice", "z", "Up axis (used by bbox_bottom)", ["z", "y"])},
    },
    "align_principal": {
        "description": "Rotate about the centroid so the principal axes line up with X (longest), Y, Z "
                       "(shortest). Meshes use exact area-weighted moments.",
        "applies_to": _BOTH,
        "args": {},
    },
    "align_floor": {
        "description": "Find the support plane (clouds: largest plane with the object on one side; meshes: "
                       "largest flat face with the object on one side), rotate it to be the floor and move it "
                       "to 0 on the up axis with the object above.",
        "applies_to": _BOTH,
        "args": {"up": _arg("choice", "z", "Up axis", ["z", "y"])},
    },
    "downsample": {
        "description": "Voxel downsampling (averages points per voxel).",
        "applies_to": _PC_ONLY,
        "args": {"voxel": _arg("number", None, "Voxel size in scan units (> 0)", required=True)},
    },
    "remove_outliers": {
        "description": "Statistical outlier removal.",
        "applies_to": _PC_ONLY,
        "args": {"neighbors": _arg("integer", 24, "Neighbours used for the mean distance"),
                 "std_ratio": _arg("number", 2.0, "Remove points farther than mean + std_ratio * std")},
    },
    "remove_small_components": {
        "description": "Remove disconnected pieces smaller than a fraction of the largest one "
                       "(clouds: DBSCAN point count, meshes: connected surface area).",
        "applies_to": _BOTH,
        "args": {"min_ratio": _arg("number", 0.02, "Keep pieces >= this fraction of the largest (0..1)"),
                 "eps": _arg("number", 0.0, "Clouds only: DBSCAN neighbour distance (0 = 8x point spacing)")},
    },
    "simplify": {
        "description": "Quadric decimation to a triangle budget.",
        "applies_to": _MESH_ONLY,
        "args": {"target_triangles": _arg("integer", 0, "Target triangle count (> 0), or use ratio"),
                 "ratio": _arg("number", 0.0, "Fraction of triangles to keep (0..1], used when "
                               "target_triangles is 0")},
    },
    "smooth": {
        "description": "Blend the surface smoother, on the whole object or only inside a region (any region kind: "
                       "brush, box, cylinder, part end, screen selection...) with a feathered edge. Meshes: Taubin "
                       "(keeps volume), Laplacian "
                       "(shrinks) or bilateral (feature-preserving, same as denoise); preserve_edges keeps "
                       "vertices from moving across sharp edges (screw-head rims, thread crests). Point clouds: "
                       "moving-least-squares projection (taubin = quadratic fit, laplacian = plane fit). Never "
                       "adds or removes points; reports how far the geometry moved. Warning: " + LAPLACIAN_MESH_WARNING
                       + " Bilateral on point clouds is the denoise filter: " + DENOISE_CLOUD_WARNING,
        "applies_to": _BOTH,
        "args": {"iterations": _arg("integer", 5, "Number of smoothing iterations"),
                 "method": _arg("choice", "taubin", "Smoothing filter. taubin keeps sizes; laplacian shrinks meshes and "
                                "bilateral changes feature sizes on point clouds - avoid both on surfaces you will "
                                "measure", ["taubin", "laplacian", "bilateral"]),
                 "strength": _arg("number", 1.0, _STRENGTH_HELP),
                 "preserve_edges": _arg("boolean", True, "Do not smooth across sharp feature edges"),
                 "edge_angle": _arg("number", 30.0, "Angle between surfaces (degrees) above which an edge counts "
                                    "as sharp"),
                 "region": _arg("region", None, f"Where to smooth: {_REGION_HELP}"),
                 "feather": _arg("number", 0.0, _FEATHER_HELP)},
    },
    "denoise": {
        "description": "Feature-preserving denoising of scanner roughness (e.g. rough patches on a screw head) "
                       "that keeps sharp edges. Meshes: bilateral normal filtering, then vertices are fitted to "
                       "the filtered normals. Point clouds: bilateral filtering along normals. Works on the whole "
                       "object or a feathered region; reports how far the geometry moved. Warning: "
                       + DENOISE_CLOUD_WARNING,
        "applies_to": _BOTH,
        "args": {"iterations": _arg("integer", 5, "Number of filter iterations"),
                 "normal_sigma_deg": _arg("number", 20.0, "Normal difference (degrees) that still counts as the "
                                          "same surface; smaller keeps more edges"),
                 "spatial_sigma": _arg("number", 0.0, "Neighbourhood size in scan units (0 = auto from edge "
                                       "length / point spacing)"),
                 "strength": _arg("number", 1.0, _STRENGTH_HELP),
                 "region": _arg("region", None, f"Where to denoise: {_REGION_HELP}"),
                 "feather": _arg("number", 0.0, _FEATHER_HELP)},
    },
    "smooth_points": {
        "description": "Moving-least-squares smoothing of a point cloud: each point moves towards a weighted "
                       "plane (order 1) or quadratic surface (order 2) fitted to its neighbours. Points are never "
                       "added or removed; reports how far they moved.",
        "applies_to": _PC_ONLY,
        "args": {"radius": _arg("number", 0.0, "Neighbourhood radius in scan units (0 = 6x point spacing)"),
                 "order": _arg("integer", 1, "1 = plane fit, 2 = quadratic fit (keeps curvature better)"),
                 "strength": _arg("number", 1.0, _STRENGTH_HELP),
                 "iterations": _arg("integer", 1, "Number of passes"),
                 "preserve_edges": _arg("boolean", True, "Ignore neighbours whose normal differs by more than "
                                        "edge_angle (keeps sharp edges)"),
                 "edge_angle": _arg("number", 30.0, "Normal difference in degrees for preserve_edges"),
                 "region": _arg("region", None, f"Where to smooth: {_REGION_HELP}"),
                 "feather": _arg("number", 0.0, _FEATHER_HELP)},
    },
    "remove_spikes": {
        "description": "Edit out protrusions and dents: fits a robust local reference surface, finds patches that "
                       "stick out of it by more than threshold and are no wider than max_size, and pulls them "
                       "(with a feathered rim) back onto the surface. Larger features such as thread crests are "
                       "kept. Works on the whole object or a region; reports how far the geometry moved.",
        "applies_to": _BOTH,
        "args": {"threshold": _arg("number", 0.0, "Distance from the reference surface (scan units) that counts "
                                   "as a spike (0 = auto: 6x the local noise sigma)"),
                 "max_size": _arg("number", 0.0, "Widest patch (scan units) that is removed; wider ones are "
                                  "treated as real features (0 = 25x point spacing)"),
                 "radius": _arg("number", 0.0, "Neighbourhood for the reference surface, must be >= max_size "
                                "(0 = 1.5x max_size)"),
                 "direction": _arg("choice", "both", "Remove bumps (outward), dents (inward) or both",
                                   ["both", "outward", "inward"]),
                 "iterations": _arg("integer", 2, "Detection passes"),
                 "region": _arg("region", None, f"Where to look: {_REGION_HELP}"),
                 "feather": _arg("number", 0.0, _FEATHER_HELP)},
    },
    "fill_holes": {
        "description": "Close boundary loops (holes) with new triangles; vertices are not moved.",
        "applies_to": _MESH_ONLY,
        "args": {"max_hole_size": _arg("number", 0.0, "Largest hole to fill (radius, scan units); 0 = all")},
    },
    "subdivide": {
        "description": "Loop subdivision (each iteration x4 triangles).",
        "applies_to": _MESH_ONLY,
        "args": {"iterations": _arg("integer", 1, "Number of subdivision iterations (1-4)")},
    },
    "repair": {
        "description": "Remove degenerate and duplicate triangles, duplicate vertices, non-manifold edges and "
                       "unreferenced vertices.",
        "applies_to": _MESH_ONLY,
        "args": {},
    },
    "flip_normals": {
        "description": "Reverse normals (and triangle winding for meshes).",
        "applies_to": _BOTH,
        "args": {},
    },
    "recompute_normals": {
        "description": "Recompute normals (clouds: estimated from neighbours and oriented outward).",
        "applies_to": _BOTH,
        "args": {},
    },
    "to_pointcloud": {
        "description": "Convert a mesh to a point cloud (its vertices, or uniformly sampled points).",
        "applies_to": _MESH_ONLY,
        "args": {"samples": _arg("integer", 0, "Number of surface samples; 0 = use the vertices")},
    },
    "paint": {
        "description": "Give every point / vertex one colour, or only those inside a region (the others keep their "
                       "colour; without colours they become neutral grey 0.75). Geometry is not changed.",
        "applies_to": _BOTH,
        "args": {"rgb": _arg("vec3", None, "Colour [r, g, b] in 0..1", required=True),
                 "region": _arg("region", None, f"Where to paint: {_REGION_HELP}")},
    },
}
UNPAINTED_GREY = 0.75


# --------------------------------------------------------------------------- validation
def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def _number(value, where: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    return float(value)


def _integer(value, where: str) -> int:
    num = _number(value, where)
    if num != int(num):
        raise ValueError(f"{where} must be a whole number, got {value!r}")
    return int(num)


def _vec3(value, where: str) -> list[float]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        raise ValueError(f"{where} must be a list of three numbers [x, y, z], got {value!r}")
    return [_number(v, where) for v in value]


def _check_arg(op: str, name: str, spec: dict, value):
    where = f"{op}: '{name}'"
    kind = spec["type"]
    if kind == "number":
        return _number(value, where)
    if kind == "integer":
        return _integer(value, where)
    if kind == "boolean":
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{where} must be true or false, got {value!r}")
        return bool(value)
    if kind == "choice":
        if value not in spec["choices"]:
            raise ValueError(f"{where} must be one of {', '.join(map(repr, spec['choices']))}, got {value!r}")
        return value
    if kind == "vec3":
        return _vec3(value, where)
    if kind == "axis":
        if isinstance(value, str):
            if value.lower() not in ("x", "y", "z"):
                raise ValueError(f'{where} must be "x", "y", "z" or a direction [x, y, z], got {value!r}')
            return value.lower()
        vec = _vec3(value, where)
        if np.linalg.norm(vec) == 0:
            raise ValueError(f"{where} direction must not be zero")
        return vec
    if kind == "center":
        if isinstance(value, str):
            if value not in ("origin", "centroid", "bbox"):
                raise ValueError(f"{where} must be {_CENTER_HELP}, got {value!r}")
            return value
        return _vec3(value, where)
    if kind == "matrix4":
        arr = np.asarray(value, dtype=object)
        if arr.size != 16 or arr.shape not in ((4, 4), (16,)):
            raise ValueError(f"{where} must be a 4x4 matrix (4 rows of 4 numbers, or 16 numbers row-major)")
        mat = np.array([_number(v, where) for v in arr.reshape(-1)]).reshape(4, 4)
        return mat.tolist()
    if kind == "matrix16":
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 16:
            raise ValueError(f"{where} must be a list of 16 numbers (column-major 4x4 matrix)")
        return [_number(v, where) for v in value]
    if kind == "polygon":
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
            raise ValueError(f"{where} must be a list of at least three [x, y] points")
        pts = []
        for p in value:
            if not isinstance(p, (list, tuple, np.ndarray)) or len(p) != 2:
                raise ValueError(f"{where} points must be [x, y] pairs, got {p!r}")
            pts.append([_number(p[0], where), _number(p[1], where)])
        return pts
    if kind == "region":
        return _check_region(value, where)
    raise ValueError(f"internal: unknown argument type {kind}")  # pragma: no cover


def _check_region(value, where: str) -> dict:
    return regions.check_region(value, where)


_LOCAL_OPS = ("smooth", "denoise", "smooth_points", "remove_spikes")


def _check_semantics(op: str, a: dict) -> None:
    """Checks that involve the value, not just its type."""
    def bad(msg: str):
        raise ValueError(f"{op}: {msg}")

    if op == "crop_box" and any(lo > hi for lo, hi in zip(a["min"], a["max"])):
        bad("every 'min' coordinate must be <= the matching 'max' coordinate")
    if op == "cut_plane" and np.linalg.norm(a["normal"]) == 0:
        bad("'normal' must not be zero")
    if op == "select_screen":
        if abs(np.linalg.det(np.asarray(a["view_projection"]).reshape(4, 4))) < 1e-300:
            bad("'view_projection' is not invertible")
    if op == "delete_sphere" and a["radius"] <= 0:
        bad("'radius' must be > 0")
    if op == "transform":
        m = np.asarray(a["matrix"])
        if not np.allclose(m[3], [0, 0, 0, 1]):
            bad("the last row of 'matrix' must be [0, 0, 0, 1] (affine transforms only)")
        if abs(np.linalg.det(m[:3, :3])) < 1e-300:
            bad("'matrix' is singular (it would flatten the geometry)")
    if op == "scale" and a["factor"] <= 0:
        bad("'factor' must be > 0 (use mirror to flip)")
    if op == "downsample" and a["voxel"] <= 0:
        bad("'voxel' must be > 0")
    if op == "remove_outliers":
        if a["neighbors"] < 1:
            bad("'neighbors' must be >= 1")
        if a["std_ratio"] <= 0:
            bad("'std_ratio' must be > 0")
    if op == "remove_small_components":
        if not 0 <= a["min_ratio"] <= 1:
            bad("'min_ratio' must be between 0 and 1")
        if a["eps"] < 0:
            bad("'eps' must be >= 0")
    if op == "simplify":
        if a["target_triangles"] < 0 or not 0 <= a["ratio"] <= 1:
            bad("'target_triangles' must be >= 0 and 'ratio' between 0 and 1")
        if a["target_triangles"] == 0 and a["ratio"] == 0:
            bad("give 'target_triangles' (> 0) or 'ratio' (0..1]")
    if op == "smooth" and a["iterations"] < 1:
        bad("'iterations' must be >= 1")
    if op in _LOCAL_OPS:
        if not 1 <= a["iterations"] <= 100:
            bad("'iterations' must be between 1 and 100")
        if "strength" in a and not 0 < a["strength"] <= 1:
            bad("'strength' must be > 0 and <= 1")
        if a["feather"] < 0:
            bad("'feather' must be >= 0 (0 = automatic)")
        if "edge_angle" in a and not 0 < a["edge_angle"] < 180:
            bad("'edge_angle' must be between 0 and 180 degrees")
        for name in ("spatial_sigma", "radius", "threshold", "max_size"):
            if name in a and a[name] < 0:
                bad(f"'{name}' must be >= 0 (0 = automatic)")
    if op == "denoise" and not 0 < a["normal_sigma_deg"] <= 90:
        bad("'normal_sigma_deg' must be > 0 and <= 90")
    if op == "smooth_points" and a["order"] not in (1, 2):
        bad("'order' must be 1 (plane) or 2 (quadratic)")
    if op == "remove_spikes" and a["radius"] > 0 and a["max_size"] > 0 and a["radius"] < a["max_size"]:
        bad("'radius' must be >= 'max_size' (the reference surface has to reach past the spike)")
    if op == "fill_holes" and a["max_hole_size"] < 0:
        bad("'max_hole_size' must be >= 0 (0 fills all holes)")
    if op == "subdivide" and not 1 <= a["iterations"] <= 4:
        bad("'iterations' must be between 1 and 4 (each one multiplies the triangle count by 4)")
    if op == "to_pointcloud" and a["samples"] < 0:
        bad("'samples' must be >= 0")
    if op == "paint" and any(not 0 <= c <= 1 for c in a["rgb"]):
        bad("'rgb' values must be between 0 and 1")


def validate_ops(ops, kind: str) -> list[dict]:
    """Check a list of ops for a geometry of `kind` ("pointcloud" | "mesh").

    Accepts `{"op": name, **args}` or `{"op": name, "args": {...}}` and returns normalised flat ops with
    defaults filled in. Raises ValueError with a human-readable message on the first problem."""
    if kind not in (PC, MESH):
        raise ValueError(f"Geometry kind must be '{PC}' or '{MESH}', got {kind!r}")
    if not isinstance(ops, (list, tuple)) or not ops:
        raise ValueError("Give at least one edit operation")
    out = []
    for i, raw in enumerate(ops, 1):
        if not isinstance(raw, dict) or "op" not in raw:
            raise ValueError(f"Operation {i} must be an object with an 'op' field")
        name = raw["op"]
        if name not in OP_CATALOGUE:
            raise ValueError(f"Operation {i}: unknown op {name!r}. Available: {', '.join(OP_CATALOGUE)}")
        entry = OP_CATALOGUE[name]
        if kind not in entry["applies_to"]:
            what = "point clouds" if entry["applies_to"] == _PC_ONLY else "meshes"
            here = "a point cloud" if kind == PC else "a mesh"
            raise ValueError(f"Operation {i}: '{name}' only works on {what}, but this is {here} at that step")
        given = {k: v for k, v in raw.items() if k != "op"}
        if isinstance(given.get("args"), dict) and "args" not in entry["args"]:
            given = {**{k: v for k, v in given.items() if k != "args"}, **given["args"]}
        unknown = sorted(set(given) - set(entry["args"]))
        if unknown:
            valid = ", ".join(entry["args"]) or "none"
            raise ValueError(f"Operation {i} ('{name}'): unknown argument(s) {', '.join(unknown)}. "
                             f"Valid arguments: {valid}")
        args = {}
        for arg, spec in entry["args"].items():
            if arg not in given or given[arg] is None:
                if spec["required"]:
                    raise ValueError(f"Operation {i} ('{name}'): missing required argument '{arg}' "
                                     f"({spec['description']})")
                args[arg] = spec["default"]
            else:
                args[arg] = _check_arg(name, arg, spec, given[arg])
        _check_semantics(name, args)
        out.append({"op": name, **args})
        if name == "to_pointcloud":
            kind = PC
    return out


# --------------------------------------------------------------------------- geometry helpers
V3 = o3d.utility.Vector3dVector
_AXES = {"x": 0, "y": 1, "z": 2}


def _copy(geom: Geometry) -> Geometry:
    return o3d.geometry.PointCloud(geom) if is_cloud(geom) else o3d.geometry.TriangleMesh(geom)


def _positions(geom: Geometry) -> np.ndarray:
    return np.asarray(geom.points if is_cloud(geom) else geom.vertices)


def _counts(geom: Geometry) -> dict:
    if is_cloud(geom):
        return {"points": len(geom.points)}
    return {"vertices": len(geom.vertices), "triangles": len(geom.triangles)}


def _is_empty(geom: Geometry) -> bool:
    return len(geom.points) == 0 if is_cloud(geom) else len(geom.triangles) == 0


def _diagonal(points: np.ndarray) -> float:
    return float(np.linalg.norm(points.max(0) - points.min(0))) if len(points) else 0.0


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _normalize_rows(n: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, length, out=np.zeros_like(n), where=length > 0)


def _translation(t) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def _linear(a: np.ndarray, center) -> np.ndarray:
    """4x4 for x -> A (x - c) + c."""
    m = np.eye(4)
    m[:3, :3] = a
    c = np.asarray(center, float)
    m[:3, 3] = c - a @ c
    return m


def _resolve_center(geom: Geometry, center) -> np.ndarray:
    if not isinstance(center, str):
        return np.asarray(center, float)
    pts = _positions(geom)
    if center == "origin":
        return np.zeros(3)
    if center == "centroid":
        return pts.mean(axis=0)
    return (pts.min(axis=0) + pts.max(axis=0)) / 2  # bbox


def _axis_rotation(axis, degrees: float) -> np.ndarray:
    k = np.eye(3)[_AXES[axis]] if isinstance(axis, str) else _unit(axis)
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    r = np.eye(3) + s * cross + (1 - c) * (cross @ cross)
    if degrees % 90 == 0:
        # cos(90 deg) is 6e-17 in floating point: snap so quarter turns are exact
        r = np.round(r, 12) + 0.0
    return r


def _rotation_between(a, b) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = _unit(a), _unit(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -1 + 1e-12:  # opposite: 180 deg about any perpendicular axis
        perp = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        return _axis_rotation(_unit(perp).tolist(), 180.0)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _flip_winding(mesh: o3d.geometry.TriangleMesh) -> None:
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, [0, 2, 1]])
    if mesh.has_triangle_uvs():
        uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)[:, [0, 2, 1]]
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs.reshape(-1, 2))


def apply_matrix(geom: Geometry, m: np.ndarray) -> Geometry:
    """Apply an affine 4x4 in place, in float64. Normals use the inverse transpose; a mirroring matrix
    (negative determinant) reverses mesh winding so faces keep pointing outward."""
    a, t = m[:3, :3], m[:3, 3]
    is_translation = np.array_equal(a, np.eye(3))
    normal_matrix = None if is_translation else np.linalg.inv(a).T

    def moved(arr) -> V3:
        pts = np.asarray(arr)
        return V3(pts + t if is_translation else pts @ a.T + t)

    def turned(arr) -> V3:
        return V3(_normalize_rows(np.asarray(arr) @ normal_matrix.T))

    if is_cloud(geom):
        geom.points = moved(geom.points)
        if normal_matrix is not None and geom.has_normals():
            geom.normals = turned(geom.normals)
        if normal_matrix is not None and geom.has_covariances():
            cov = np.asarray(geom.covariances)
            geom.covariances = o3d.utility.Matrix3dVector(np.einsum("ij,njk,lk->nil", a, cov, a))
    else:
        geom.vertices = moved(geom.vertices)
        if normal_matrix is not None:
            if geom.has_vertex_normals():
                geom.vertex_normals = turned(geom.vertex_normals)
            if geom.has_triangle_normals():
                geom.triangle_normals = turned(geom.triangle_normals)
            if np.linalg.det(a) < 0:
                _flip_winding(geom)
    return geom


def _cloud_subset(pcd: o3d.geometry.PointCloud, keep: np.ndarray) -> o3d.geometry.PointCloud:
    return pcd.select_by_index(np.flatnonzero(keep).tolist())


def _mesh_subset(mesh: o3d.geometry.TriangleMesh, tri_keep: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Keep the given triangles and the vertices they use; carries colours, normals, UVs and materials."""
    tris = np.asarray(mesh.triangles)
    kept = tris[tri_keep]
    used = np.zeros(len(mesh.vertices), bool)
    used[kept.reshape(-1)] = True
    remap = np.full(len(used), -1, np.int64)
    remap[used] = np.arange(int(used.sum()))
    out = o3d.geometry.TriangleMesh()
    out.vertices = V3(np.asarray(mesh.vertices)[used])
    if mesh.has_vertex_normals():
        out.vertex_normals = V3(np.asarray(mesh.vertex_normals)[used])
    if mesh.has_vertex_colors():
        out.vertex_colors = V3(np.asarray(mesh.vertex_colors)[used])
    out.triangles = o3d.utility.Vector3iVector(remap[kept].astype(np.int32).reshape(-1, 3))
    if mesh.has_triangle_normals():
        out.triangle_normals = V3(np.asarray(mesh.triangle_normals)[tri_keep])
    if mesh.has_triangle_uvs():
        uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)[tri_keep]
        out.triangle_uvs = o3d.utility.Vector2dVector(uvs.reshape(-1, 2))
    if mesh.has_triangle_material_ids():
        out.triangle_material_ids = o3d.utility.IntVector(np.asarray(mesh.triangle_material_ids)[tri_keep])
    if mesh.has_textures():
        out.textures = mesh.textures
    return out


def _keep_vertices(geom: Geometry, keep: np.ndarray) -> Geometry:
    """Clouds: keep masked points. Meshes: drop unmasked vertices together with their triangles."""
    if is_cloud(geom):
        return _cloud_subset(geom, keep)
    return _mesh_subset(geom, keep[np.asarray(geom.triangles)].all(axis=1))


def _restore_mesh_attributes(out: o3d.geometry.TriangleMesh, src: o3d.geometry.TriangleMesh):
    """Some Open3D filters drop vertex colours or normals: bring them back."""
    if src.has_vertex_colors() and not out.has_vertex_colors() and len(out.vertices):
        _, nn = cKDTree(np.asarray(src.vertices)).query(np.asarray(out.vertices), k=1, workers=-1)
        out.vertex_colors = V3(np.asarray(src.vertex_colors)[nn])
    if src.has_vertex_normals():
        out.compute_vertex_normals()
    return out


def _spacing(points: np.ndarray) -> float:
    s = estimate_spacing(points)
    return s if s > 0 else max(_diagonal(points) * 1e-4, 1e-12)


# --------------------------------------------------------------------------- screen selection
def points_in_polygon(xy: np.ndarray, polygon) -> np.ndarray:
    """Even-odd point-in-polygon, vectorised. Points are sorted by y so each edge only tests the points in
    its own y-span: cost ~ points x crossings, fine for lassos with thousands of vertices."""
    xy = np.asarray(xy, float)
    poly = np.asarray(polygon, float)
    inside = np.zeros(len(xy), bool)
    if len(xy) == 0:
        return inside
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    cand = np.flatnonzero((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]))
    if len(cand) == 0:
        return inside
    idx = cand[np.argsort(xy[cand, 1], kind="stable")]
    px, py = xy[idx, 0], xy[idx, 1]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    # half-open span [ymin, ymax) so a vertex shared by two edges is counted once
    starts = np.searchsorted(py, np.minimum(y1, y2), "left")
    ends = np.searchsorted(py, np.maximum(y1, y2), "left")
    parity = np.zeros(len(idx), bool)
    for e in np.flatnonzero(ends > starts):
        s, t = starts[e], ends[e]
        x_cross = x1[e] + (py[s:t] - y1[e]) * (x2[e] - x1[e]) / (y2[e] - y1[e])
        parity[s:t] ^= px[s:t] < x_cross
    inside[idx] = parity
    return inside


def _vp_matrix(view_projection) -> np.ndarray:
    return np.asarray(view_projection, float).reshape(4, 4).T  # three.js elements are column-major


def project(points: np.ndarray, vp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ndc xyz, clip w) for clip = VP * [x, y, z, 1]."""
    clip = points @ vp[:3, :3].T + vp[:3, 3]
    w = points @ vp[3, :3] + vp[3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip / w[:, None]
    return ndc, w


def _visible_cloud(points, ndc, w, in_frustum, vp) -> np.ndarray:
    """Coarse screen-space z-buffer: a point is visible when it is within a tolerance of the nearest point
    in its screen cell. Cells are square in pixels and sized for a few points each."""
    visible = np.zeros(len(points), bool)
    idx = np.flatnonzero(in_frustum)
    if len(idx) == 0:
        return visible
    sx, sy = np.linalg.norm(vp[0, :3]), np.linalg.norm(vp[1, :3])
    aspect = sy / sx  # width / height of the viewport
    perspective = np.linalg.norm(vp[3, :3]) > 1e-12
    if perspective:
        depth = w[idx] / np.linalg.norm(vp[3, :3])  # distance along the view axis (world units)
    else:
        depth = ndc[idx, 2] / np.linalg.norm(vp[2, :3])
    x, y = ndc[idx, 0], ndc[idx, 1]
    ux, uy = x.max() - x.min(), (y.max() - y.min()) / aspect  # extents in x-NDC units (square pixels)
    cell = math.sqrt(max(ux * uy, 1e-18) / max(len(idx) / 4, 1))
    cell = max(cell, max(ux, uy) / 1024, 1e-9)
    ix = np.floor((x - x.min()) / cell).astype(np.int64)
    iy = np.floor((y - y.min()) / (cell * aspect)).astype(np.int64)
    key = ix * (int(iy.max()) + 1) + iy
    uniq, inverse = np.unique(key, return_inverse=True)
    nearest = np.full(len(uniq), np.inf)
    np.minimum.at(nearest, inverse, depth)
    cell_world = cell * (depth if perspective else 1.0) / sx
    tol = 3 * _spacing(points[idx]) + 2 * cell_world
    visible[idx] = depth <= nearest[inverse] + tol
    return visible


def _visible_mesh(mesh: o3d.geometry.TriangleMesh, candidates: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Ray cast from the camera to each candidate vertex; visible when nothing is hit before it."""
    verts = np.asarray(mesh.vertices)
    visible = np.zeros(len(verts), bool)
    idx = np.flatnonzero(candidates)
    if len(idx) == 0:
        return visible
    offset = verts.mean(axis=0)  # float32 scene: keep coordinates small for precision
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor((verts - offset).astype(np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.triangles, dtype=np.uint32)))
    diag = _diagonal(verts)
    target = verts[idx] - offset
    h = np.linalg.inv(vp) @ np.array([0.0, 0.0, 1.0, 0.0])
    if abs(h[3]) > 1e-12 * np.linalg.norm(h[:3]):  # perspective: the camera centre maps to clip (0, 0, *, 0)
        eye = h[:3] / h[3] - offset
        dirs = target - eye
        dist = np.linalg.norm(dirs, axis=1)
        origins = np.broadcast_to(eye, target.shape)
    else:  # orthographic: parallel rays along the view direction, started well behind the object
        forward = _unit(h[:3])
        dist = np.full(len(idx), 2 * diag + 1.0)
        dirs = np.broadcast_to(forward, target.shape)
        origins = target - forward * dist[:, None]
    dirs = dirs / np.maximum(dist, 1e-30)[:, None]
    rays = np.hstack([origins, dirs]).astype(np.float32)
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    tol = 2e-4 * dist + 1e-5 * diag
    visible[idx] = t_hit >= dist - tol
    return visible


def screen_selection(geom: Geometry, view_projection, polygon, visible_only: bool = False) -> np.ndarray:
    """Boolean mask of points / vertices inside the screen polygon (NDC) and inside the view frustum."""
    pts = _positions(geom)
    vp = _vp_matrix(view_projection)
    ndc, w = project(pts, vp)
    in_frustum = (w > 0) & (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1)
    selected = np.zeros(len(pts), bool)
    idx = np.flatnonzero(in_frustum)
    selected[idx] = points_in_polygon(ndc[idx, :2], polygon)
    if visible_only and selected.any():
        if is_cloud(geom):
            selected &= _visible_cloud(pts, ndc, w, in_frustum, vp)
        else:
            selected &= _visible_mesh(geom, selected, vp)
    return selected


# --------------------------------------------------------------------------- ops: selection / removal
def _op_crop_box(g, log, min, max, invert):
    pts = _positions(g)
    inside = np.all((pts >= np.asarray(min)) & (pts <= np.asarray(max)), axis=1)
    return _keep_vertices(g, ~inside if invert else inside), {}


def _op_cut_plane(g, log, point, normal, keep):
    signed = (_positions(g) - np.asarray(point)) @ np.asarray(normal)
    return _keep_vertices(g, signed >= 0 if keep == "positive" else signed <= 0), {}


def _op_select_screen(g, log, view_projection, polygon, mode, visible_only):
    sel = screen_selection(g, view_projection, polygon, visible_only)
    info = {"selected": int(sel.sum())}
    if is_cloud(g):
        return _cloud_subset(g, sel if mode == "keep" else ~sel), info
    tri_sel = sel[np.asarray(g.triangles)].all(axis=1)
    info["selected_triangles"] = int(tri_sel.sum())
    return _mesh_subset(g, tri_sel if mode == "keep" else ~tri_sel), info


def _op_delete_sphere(g, log, center, radius):
    d2 = np.sum((_positions(g) - np.asarray(center)) ** 2, axis=1)
    return _keep_vertices(g, d2 > radius * radius), {}


def _region_selection(op: str, g, region) -> tuple[np.ndarray, dict]:
    mask = regions.region_mask(g, region)
    info = {"selected": int(mask.sum()), "region": regions.describe_region(region)}
    return mask, info


def _op_delete_region(g, log, region):
    mask, info = _region_selection("delete_region", g, region)
    if not mask.any():
        raise ValueError(f"delete_region: the region ({info['region']}) contains no points of this object - "
                         "nothing to delete")
    log(f"  delete_region: {info['selected']:,} {'points' if is_cloud(g) else 'vertices'} in {info['region']}")
    return _keep_vertices(g, ~mask), info


def _op_keep_region(g, log, region):
    mask, info = _region_selection("keep_region", g, region)
    log(f"  keep_region: {info['selected']:,} {'points' if is_cloud(g) else 'vertices'} in {info['region']}")
    return _keep_vertices(g, mask), info


def _op_downsample(g, log, voxel):
    return g.voxel_down_sample(voxel), {}


def _op_remove_outliers(g, log, neighbors, std_ratio):
    out, _ = g.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
    return out, {}


def _op_remove_small_components(g, log, min_ratio, eps):
    if not is_cloud(g):
        clusters, _, areas = g.cluster_connected_triangles()
        clusters, areas = np.asarray(clusters), np.asarray(areas)
        small = areas < min_ratio * areas.max()
        return _mesh_subset(g, ~small[clusters]), {"components": int(len(areas)), "removed_components": int(small.sum())}
    pts = np.asarray(g.points)
    eps = eps or 8 * _spacing(pts)
    if len(pts) > 1_500_000:
        # DBSCAN on millions of points is slow and memory hungry; label a voxel grid finer than eps
        # (every point is within eps of its voxel representative) and map labels back.
        coarse = g.voxel_down_sample(eps / 3)
        labels_c = np.asarray(coarse.cluster_dbscan(eps=eps, min_points=3))
        _, nn = cKDTree(np.asarray(coarse.points)).query(pts, k=1, workers=-1)
        labels = labels_c[nn]
    else:
        labels = np.asarray(g.cluster_dbscan(eps=eps, min_points=3))
    sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.zeros(0, int)
    largest = max(int(sizes.max()) if len(sizes) else 0, 1)
    keep_label = sizes >= min_ratio * largest
    keep = np.zeros(len(pts), bool)
    keep[labels >= 0] = keep_label[labels[labels >= 0]]
    keep[labels < 0] = 1 >= min_ratio * largest  # isolated points are components of size 1
    return _cloud_subset(g, keep), {"eps": eps, "components": int(len(sizes)),
                                    "removed_components": int((~keep_label).sum())}


# --------------------------------------------------------------------------- ops: transforms
def _op_transform(g, log, matrix):
    m = np.asarray(matrix, float)
    return apply_matrix(g, m), {"_matrix": m}


def _op_translate(g, log, offset):
    m = _translation(offset)
    return apply_matrix(g, m), {"_matrix": m}


def _op_rotate(g, log, axis, degrees, center):
    m = _linear(_axis_rotation(axis, degrees), _resolve_center(g, center))
    return apply_matrix(g, m), {"_matrix": m}


def _op_scale(g, log, factor, center):
    m = _linear(np.eye(3) * factor, _resolve_center(g, center))
    return apply_matrix(g, m), {"_matrix": m}


def _op_mirror(g, log, axis, center):
    a = np.eye(3)
    a[_AXES[axis], _AXES[axis]] = -1.0
    m = _linear(a, _resolve_center(g, center))
    return apply_matrix(g, m), {"_matrix": m}


def _op_center(g, log, mode, up):
    pts = _positions(g)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    if mode == "centroid":
        target = pts.mean(axis=0)
    else:
        target = (lo + hi) / 2
        if mode == "bbox_bottom":
            target[_AXES[up]] = lo[_AXES[up]]
    m = _translation(-target)
    return apply_matrix(g, m), {"_matrix": m}


def _moments(g) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and covariance. Meshes use exact surface integrals, so uneven tessellation (CAD) does not
    bias the axes."""
    if is_cloud(g) or len(g.triangles) == 0:
        pts = _positions(g)
        c = pts.mean(axis=0)
        x = pts - c
        return c, x.T @ x / len(x)
    v = np.asarray(g.vertices)
    a, b, cc = (v[np.asarray(g.triangles)[:, i]] for i in range(3))
    area = np.linalg.norm(np.cross(b - a, cc - a), axis=1) / 2
    total = area.sum()
    s = a + b + cc
    centroid = (area[:, None] * s).sum(axis=0) / (3 * total)
    # integral of x x^T over a triangle = A/12 (a a^T + b b^T + c c^T + s s^T)
    second = sum(np.einsum("n,ni,nj->ij", area, p, p) for p in (a, b, cc, s)) / 12
    return centroid, second / total - np.outer(centroid, centroid)


def _op_align_principal(g, log):
    c, cov = _moments(g)
    evals, evecs = np.linalg.eigh(cov)
    axes = evecs[:, ::-1]  # columns: longest, middle, shortest
    x = _positions(g) - c
    for k in range(3):  # deterministic signs: heavier tail on the positive side
        proj = x @ axes[:, k]
        skew = float(np.mean(proj ** 3))
        if skew < 0 or (skew == 0 and axes[np.argmax(np.abs(axes[:, k])), k] < 0):
            axes[:, k] *= -1
    r = axes.T
    if np.linalg.det(r) < 0:
        r[2] *= -1  # keep a proper rotation (no mirroring)
    m = _linear(r, c)
    return apply_matrix(g, m), {"_matrix": m, "extents_sqrt_variance": np.sqrt(np.maximum(evals[::-1], 0)).tolist()}


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    return vt[2], c


_NO_FLOOR = (" - a scan's underside is often missing: use the Floor button on the right of the 3D view and "
             "click a flat part of the model")


def _floor_plane_cloud(pcd: o3d.geometry.PointCloud, log: Log):
    pts = np.asarray(pcd.points)
    n_all = len(pts)
    thr = 3 * _spacing(pts)
    rng = np.random.default_rng(0)
    sample_idx = np.sort(rng.choice(n_all, 300_000, replace=False)) if n_all > 300_000 else np.arange(n_all)
    sample = pts[sample_idx]
    remaining = o3d.geometry.PointCloud(V3(sample))
    o3d.utility.random.seed(0)
    for attempt in range(8):
        if len(remaining.points) < max(0.02 * len(sample), 3):
            break
        model, inliers = remaining.segment_plane(distance_threshold=thr, ransac_n=3, num_iterations=3000)
        normal = _unit(model[:3])
        point = -float(model[3]) / float(np.linalg.norm(model[:3])) * normal
        # least-squares refit on full-resolution inliers with a shrinking band (removes edge bias)
        for band in (thr, thr / 2, thr / 4):
            near = np.abs((pts - point) @ normal) <= band
            if near.sum() < 3:
                break
            normal, point = _fit_plane(pts[near])
        signed = (sample - point) @ normal
        above, below = int((signed > thr).sum()), int((signed < -thr).sum())
        if min(above, below) <= 0.01 * len(sample):
            up = normal if above >= below else -normal
            log(f"  support plane with {int(near.sum()):,} points (attempt {attempt + 1})")
            return up, point, int(near.sum())
        remaining = remaining.select_by_index(inliers, invert=True)
    raise ValueError("align_floor: no plane with the whole object on one side of it was found" + _NO_FLOOR)


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _floor_plane_mesh(mesh: o3d.geometry.TriangleMesh, log: Log):
    v = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    normals = _normalize_rows(cross)
    centers = (a + b + c) / 3
    diag = _diagonal(v)
    tol = 1e-3 * diag
    bins = _fibonacci_sphere(4096)
    _, nearest = cKDTree(bins).query(normals, k=1)
    bin_area = np.bincount(nearest, weights=area, minlength=len(bins))
    best = None
    for k in np.argsort(bin_area)[::-1][:24]:
        if bin_area[k] <= 0:
            break
        direction = bins[k]
        for cos_lim in (math.cos(math.radians(5)), math.cos(math.radians(1))):
            sel = normals @ direction > cos_lim
            if not sel.any():
                break
            direction = _unit((normals[sel] * area[sel, None]).sum(axis=0))
        if not sel.any():
            continue
        offsets = centers[sel] @ direction
        # coplanar group: the offset with the most area (parallel faces at other heights are separate)
        width = max(tol, 1e-12)
        hist_keys = np.floor(offsets / width).astype(np.int64)
        uniq, inv = np.unique(hist_keys, return_inverse=True)
        peak = uniq[np.argmax(np.bincount(inv, weights=area[sel]))]
        members = np.flatnonzero(sel)[np.abs(hist_keys - peak) <= 1]
        w = area[members]
        normal = _unit((normals[members] * w[:, None]).sum(axis=0))
        point = (centers[members] * w[:, None]).sum(axis=0) / w.sum()
        face_area = float(w.sum())
        if best is not None and face_area <= best[2]:
            continue
        outside = ((v - point) @ normal > tol).sum()  # outward normal: object must be behind the face
        if outside <= 0.002 * len(v):
            best = (-normal, point, face_area)
    if best is None:
        raise ValueError("align_floor: no flat face with the whole object on one side of it was found" + _NO_FLOOR)
    log(f"  floor face area {best[2]:.5g}")
    return best


def _op_align_floor(g, log, up):
    if is_cloud(g):
        normal_up, point, support = _floor_plane_cloud(g, log)
        info = {"support_points": support}
    else:
        normal_up, point, area = _floor_plane_mesh(g, log)
        info = {"face_area": area}
    k = _AXES[up]
    r = _rotation_between(normal_up, np.eye(3)[k])
    m = _linear(r, point)  # rotate about a point on the plane, so it stays put horizontally
    m = _translation(-point[k] * np.eye(3)[k]) @ m  # then drop the plane onto 0
    info["plane_normal"] = normal_up.tolist()
    info["plane_point"] = point.tolist()
    return apply_matrix(g, m), {"_matrix": m, **info}


# --------------------------------------------------------------------------- ops: mesh processing
def _op_simplify(g, log, target_triangles, ratio):
    target = target_triangles or max(int(round(len(g.triangles) * ratio)), 1)
    if target >= len(g.triangles):
        log(f"  already has {len(g.triangles):,} triangles (target {target:,}) - unchanged")
        return g, {"target_triangles": target}
    out = g.simplify_quadric_decimation(target_number_of_triangles=target)
    return _restore_mesh_attributes(out, g), {"target_triangles": target}


def _op_fill_holes(g, log, max_hole_size):
    size = max_hole_size if max_hole_size > 0 else max(_diagonal(np.asarray(g.vertices)) * 1e3, 1e6)
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(g, vertex_dtype=o3d.core.float64)
    tris = tmesh.fill_holes(hole_size=size).triangle.indices.numpy()
    before = len(g.triangles)
    out = o3d.geometry.TriangleMesh(g)
    # fill_holes only adds triangles; reuse the original float64 vertices, colours and normals as they are
    out.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))
    if g.has_triangle_normals():
        out.compute_triangle_normals()
    if g.has_triangle_uvs():  # new triangles have no texture coordinates
        uvs = np.zeros((len(tris), 3, 2))
        uvs[:before] = np.asarray(g.triangle_uvs).reshape(-1, 3, 2)
        out.triangle_uvs = o3d.utility.Vector2dVector(uvs.reshape(-1, 2))
    if g.has_triangle_material_ids():
        ids = np.zeros(len(tris), np.int32)
        ids[:before] = np.asarray(g.triangle_material_ids)
        out.triangle_material_ids = o3d.utility.IntVector(ids)
    if g.has_vertex_normals():
        out.compute_vertex_normals()
    return out, {"added_triangles": int(len(tris) - before)}


def _op_subdivide(g, log, iterations):
    return _restore_mesh_attributes(g.subdivide_loop(number_of_iterations=iterations), g), {}


def _op_repair(g, log):
    g.remove_duplicated_vertices()  # first, so triangles that only differed by vertex copies become duplicates
    g.remove_degenerate_triangles()
    g.remove_duplicated_triangles()
    g.remove_non_manifold_edges()
    g.remove_unreferenced_vertices()
    return g, {"watertight": bool(g.is_watertight())}


def _op_flip_normals(g, log):
    if is_cloud(g):
        if not g.has_normals():
            raise ValueError("flip_normals: this point cloud has no normals (use recompute_normals first)")
        g.normals = V3(-np.asarray(g.normals))
        return g, {}
    _flip_winding(g)
    if g.has_vertex_normals():
        g.vertex_normals = V3(-np.asarray(g.vertex_normals))
    if g.has_triangle_normals():
        g.triangle_normals = V3(-np.asarray(g.triangle_normals))
    return g, {}


def _op_recompute_normals(g, log):
    if is_cloud(g):
        from .clean import ensure_normals

        ensure_normals(g, _spacing(np.asarray(g.points)), recompute=True, log=log)
    else:
        g.compute_triangle_normals()
        g.compute_vertex_normals()
    return g, {}


def _op_to_pointcloud(g, log, samples):
    if samples > 0:
        if not g.has_vertex_normals():
            g.compute_vertex_normals()
        return g.sample_points_uniformly(number_of_points=samples), {}
    pcd = o3d.geometry.PointCloud(V3(np.asarray(g.vertices)))
    if g.has_vertex_normals():
        pcd.normals = V3(np.asarray(g.vertex_normals))
    if g.has_vertex_colors():
        pcd.colors = V3(np.asarray(g.vertex_colors))
    return pcd, {}


def _op_paint(g, log, rgb, region=None):
    n = len(_positions(g))
    info = {}
    if region is None:
        colors = np.tile(np.asarray(rgb, float), (n, 1))
    else:
        mask, info = _region_selection("paint", g, region)
        if not mask.any():
            raise ValueError(f"paint: the region ({info['region']}) contains no points of this object - nothing "
                             "to paint")
        has = g.has_colors() if is_cloud(g) else g.has_vertex_colors()
        if has:
            colors = np.array(np.asarray(g.colors if is_cloud(g) else g.vertex_colors), dtype=float)
        else:
            colors = np.full((n, 3), UNPAINTED_GREY)
            info["unpainted_color"] = [UNPAINTED_GREY] * 3
        colors[mask] = np.asarray(rgb, float)
        info["painted"] = int(mask.sum())
    if is_cloud(g):
        g.colors = V3(colors)
    else:
        g.vertex_colors = V3(colors)
    return g, info


# --------------------------------------------------------------------------- local edits: regions and reports
_PAIR_CHUNK = 4_000_000  # face pairs per block in the bilateral filter
_POINT_CHUNK = 16_384  # points per block in neighbourhood fits (blocks run on a thread pool)


def _falloff(dist: np.ndarray, feather: float) -> np.ndarray:
    """1 inside the region (distance 0), a cosine ramp down to 0 across the feather band, 0 beyond.
    The ramp has zero slope at both ends, so edited and untouched surface meet without a step or a crease."""
    w = (dist <= 0).astype(float)
    band = (dist > 0) & (dist < feather)
    w[band] = 0.5 * (1 + np.cos(np.pi * dist[band] / feather))
    return w


def _region_distance(g: Geometry, region: dict | None, cutoff: float) -> np.ndarray:
    """Distance of every point / vertex to the region (0 inside); inf where it is farther than `cutoff`.
    Any region kind of cloudclean.regions (brush, boxes, cylinder, slab / part end, screen, inverted)."""
    return regions.region_distance(g, region, cutoff)


def _mesh_spacing(v: np.ndarray, tris: np.ndarray) -> float:
    """Median edge length."""
    t = tris
    if len(t) > 200_000:
        t = t[np.random.default_rng(0).choice(len(t), 200_000, replace=False)]
    lengths = np.linalg.norm(v[t[:, [1, 2, 0]]] - v[t], axis=2).ravel()
    lengths = lengths[lengths > 0]
    return float(np.median(lengths)) if len(lengths) else max(_diagonal(v) * 1e-4, 1e-12)


def _geometry_spacing(g: Geometry, pts: np.ndarray) -> float:
    return _spacing(pts) if is_cloud(g) else _mesh_spacing(pts, np.asarray(g.triangles))


def _local_setup(op: str, g: Geometry, region, feather: float, reach: float = 0.0, spacing: float | None = None):
    """float64 copy of the positions, spacing, feather, region distance (up to feather + reach) and weights."""
    pts = np.array(_positions(g), dtype=np.float64)
    spacing = spacing or _geometry_spacing(g, pts)
    feather = feather if feather > 0 else 3 * spacing
    dist = _region_distance(g, region, feather + reach)
    weights = _falloff(dist, feather)
    if not (weights > 0).any():
        raise ValueError(f"{op}: the region does not touch the object - nothing to edit")
    return pts, spacing, feather, dist, weights


def _moved_report(before: np.ndarray, after: np.ndarray, weights: np.ndarray) -> dict:
    """Displacement statistics over the region (including its feather band), in scan units."""
    d = np.linalg.norm(after - before, axis=1)
    inside = d[weights > 0]
    stats = {"mean": 0.0, "p95": 0.0, "max": 0.0}
    if len(inside):
        stats = {"mean": float(inside.mean()), "p95": float(np.percentile(inside, 95)), "max": float(d.max())}
    return {"moved": stats, "moved_vertices": int(np.count_nonzero(d)),
            "region_vertices": int(np.count_nonzero(weights > 0))}


def _support(points: np.ndarray, idx: np.ndarray, reach: float) -> np.ndarray:
    """Sorted indices of the points inside the bounding box of points[idx] grown by `reach`."""
    if len(idx) == len(points):
        return np.arange(len(points))
    lo, hi = points[idx].min(axis=0) - reach, points[idx].max(axis=0) + reach
    return np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))


def _pca_normals(points: np.ndarray, radius: float, max_nn: int = 24) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(V3(points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pcd.normals).copy()


def _store_positions(g: Geometry, new: np.ndarray, spacing: float) -> None:
    """Write moved positions back and refresh normals (only where points moved, for clouds)."""
    if not is_cloud(g):
        had_tn, had_vn = g.has_triangle_normals(), g.has_vertex_normals()
        g.vertices = V3(new)
        if had_tn:
            g.compute_triangle_normals()
        if had_vn:
            g.compute_vertex_normals()
        return
    old = np.asarray(g.points)
    changed = np.flatnonzero(np.any(new != old, axis=1))
    if g.has_normals() and len(changed):
        normals = np.asarray(g.normals).copy()
        sup = _support(new, changed, 6 * spacing)
        est = _pca_normals(new[sup], 6 * spacing, 48)[np.searchsorted(sup, changed)]
        flip = np.einsum("ij,ij->i", est, normals[changed]) < 0
        est[flip] *= -1
        normals[changed] = est
        g.normals = V3(normals)
    g.points = V3(new)


def _each_neighbourhood(points: np.ndarray, query_idx: np.ndarray, radius: float, k: int, fn) -> None:
    """Call fn(ids, distances (m, k), neighbour indices (m, k), valid mask) for blocks of query_idx on a thread
    pool, with the neighbours within `radius` (at most k, nearest first). Invalid slots point at the query itself
    with distance 0. fn must only write to rows of its own block."""
    sup = _support(points, query_idx, radius)
    tree = cKDTree(points[sup])
    k = int(min(k, len(sup)))

    def block(b):
        ids = query_idx[b[0]:b[1]]
        d, nb = tree.query(np.take(points, ids, axis=0), k=k, distance_upper_bound=radius, workers=1)
        d, nb = d.reshape(len(ids), k), nb.reshape(len(ids), k)
        valid = np.isfinite(d)
        nb = np.where(valid, np.take(sup, np.minimum(nb, len(sup) - 1)), ids[:, None])
        fn(ids, np.where(valid, d, 0.0), nb, valid)

    n = len(query_idx)
    _parallel(block, [(s, min(s + _POINT_CHUNK, n)) for s in range(0, n, _POINT_CHUNK)])


def _perpendicular_frame(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    e1 = _normalize_rows(np.cross(n, helper))
    return e1, np.cross(n, e1)


def _quadric_design(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([u * u, u * v, v * v, u, v, np.ones_like(u)], axis=-1)


def _weighted_quadric(a, h, w) -> np.ndarray:
    """Batched weighted least squares h ~ a u^2 + b uv + c v^2 + d u + e v + f for the design a = (m, k, 6) from
    _quadric_design; returns (m, 6)."""
    aw = a * w[..., None]
    ata = aw.transpose(0, 2, 1) @ a
    ridge = 1e-9 * np.maximum(w.sum(axis=1), 1e-30)
    ata += ridge[:, None, None] * np.eye(6)
    atb = np.einsum("mki,mk->mi", aw, h)
    return np.linalg.solve(ata, atb[..., None])[..., 0]


# --------------------------------------------------------------------------- local edits: mesh machinery
def _corners(x: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # np.take on contiguous index columns is several times faster than fancy indexing with a strided view
    return tuple(np.take(x, np.ascontiguousarray(tris[:, k]), axis=0) for k in range(3))


def _face_frames(x: np.ndarray, tris: np.ndarray):
    a, b, c = _corners(x, tris)
    cross = np.cross(b - a, c - a)
    return _normalize_rows(cross), (a + b + c) / 3, np.linalg.norm(cross, axis=1) / 2


def _vertex_normals(x: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a, b, c = x[tris[:, 0]], x[tris[:, 1]], x[tris[:, 2]]
    cross = np.repeat(np.cross(b - a, c - a), 3, axis=0)  # area weighted
    flat = tris.ravel()
    acc = np.stack([np.bincount(flat, weights=cross[:, k], minlength=len(x)) for k in range(3)], axis=1)
    return _normalize_rows(acc)


def _local_mesh(tris: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Working set for a region: faces within two rings of the active vertices, re-indexed.
    Returns (global index of each local vertex, local triangles). Everything when all vertices are active."""
    if active.all():
        return np.arange(len(active)), tris
    ring = np.zeros(len(active), bool)
    ring[tris[active[tris].any(axis=1)].ravel()] = True
    faces = tris[ring[tris].any(axis=1)]
    verts, inverse = np.unique(faces, return_inverse=True)
    return verts, inverse.reshape(-1, 3).astype(np.int64)


class _FaceFilter:
    """Face / vertex incidence of a mesh, for bilateral normal filtering and fitting vertices to normals."""

    def __init__(self, tris: np.ndarray, n_vertices: int):
        f = len(tris)
        self.tris = tris
        inc = sparse.csr_matrix((np.ones(3 * f, np.float32), (np.repeat(np.arange(f), 3), tris.ravel())),
                                shape=(f, n_vertices))
        adj = (inc @ inc.T).tocsr()  # faces sharing at least one vertex, each face with itself
        adj.sort_indices()
        self.indptr, self.cols = adj.indptr.astype(np.int64), adj.indices.astype(np.int32)
        self.rows = np.repeat(np.arange(f, dtype=np.int32), np.diff(self.indptr))
        cuts = np.searchsorted(self.indptr, np.arange(0, len(self.cols), _PAIR_CHUNK), side="right") - 1
        self.blocks = list(zip(np.r_[0, cuts[1:]].tolist(), np.r_[cuts[1:], f].tolist()))
        self.vertex_faces = inc.T.tocsr()
        self.valence = np.diff(self.vertex_faces.indptr).astype(float)

    def filter_normals(self, normals, centers, areas, sigma_s: float, sigma_r: float, passes: int) -> np.ndarray:
        """Bilateral filter of face normals (Zheng et al. 2011): area x spatial Gaussian x normal-difference
        Gaussian over the faces sharing a vertex."""
        c32 = (centers - centers.mean(axis=0)).astype(np.float32)
        a32 = areas.astype(np.float32)
        spatial = np.empty(len(self.cols), np.float32)

        def spatial_block(block):
            p0, p1 = self.indptr[block[0]], self.indptr[block[1]]
            cols = self.cols[p0:p1]
            d = np.take(c32, self.rows[p0:p1], axis=0) - np.take(c32, cols, axis=0)
            spatial[p0:p1] = np.take(a32, cols) * np.exp(-np.einsum("ij,ij->i", d, d) / (2 * sigma_s * sigma_s))

        _parallel(spatial_block, self.blocks)
        n = normals
        for _ in range(passes):
            nf = n.astype(np.float32)
            acc = np.empty_like(nf)

            def filter_block(block):
                p0, p1 = self.indptr[block[0]], self.indptr[block[1]]
                nc = np.take(nf, self.cols[p0:p1], axis=0)
                dot = np.einsum("ij,ij->i", np.take(nf, self.rows[p0:p1], axis=0), nc)
                # |ni - nj|^2 / (2 sigma^2) = (1 - dot) / sigma^2
                w = spatial[p0:p1] * np.exp((dot - 1) / (sigma_r * sigma_r))
                acc[block[0]:block[1]] = np.add.reduceat(nc * w[:, None], self.indptr[block[0]:block[1]] - p0, axis=0)

            _parallel(filter_block, self.blocks)
            n = _normalize_rows(acc.astype(np.float64))
        return n

    def fit_vertices(self, x: np.ndarray, face_normals: np.ndarray, step: np.ndarray, iterations: int) -> np.ndarray:
        """Vertex update of Sun et al. 2007: move each vertex towards the planes of its faces (through their
        centroids, with the filtered normals): dx = (sum n (n.c) - sum n n^T x) / valence. Only vertices with
        step > 0 move."""
        nf = face_normals
        scale = np.divide(step, self.valence, out=np.zeros(len(x)), where=self.valence > 0)
        moving = np.flatnonzero(scale > 0)
        outer = self.vertex_faces @ np.stack([nf[:, 0] * nf[:, 0], nf[:, 1] * nf[:, 1], nf[:, 2] * nf[:, 2],
                                              nf[:, 0] * nf[:, 1], nf[:, 0] * nf[:, 2], nf[:, 1] * nf[:, 2]], axis=1)
        full = len(moving) == len(x)
        pick = (lambda arr: arr) if full else (lambda arr: np.take(arr, moving, axis=0))
        a = pick(outer)[:, [0, 3, 4, 3, 1, 5, 4, 5, 2]].reshape(-1, 3, 3)
        rate = pick(scale)[:, None]
        for _ in range(iterations):
            a0, a1, a2 = _corners(x, self.tris)
            offsets = np.einsum("ij,ij->i", a0 + a1 + a2, nf) / 3
            b = pick(self.vertex_faces @ (nf * offsets[:, None]))
            xm = pick(x)
            updated = xm + rate * (b - (a @ xm[:, :, None])[:, :, 0])
            if full:
                x = updated
            else:
                x[moving] = updated
        return x


def _parallel(fn, blocks: list) -> list:
    """fn(block) for every block on a thread pool (the numpy / scipy work inside releases the GIL)."""
    workers = max(1, min(os.cpu_count() or 1, 16, len(blocks)))
    if workers == 1:
        return [fn(b) for b in blocks]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, blocks))


def _mesh_edges(tris: np.ndarray, n_vertices: int):
    """Unique edges (E, 2), number of faces on each, and the first two of those faces (-1 when absent)."""
    e = np.sort(tris[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1).astype(np.int64)
    key = e[:, 0] * n_vertices + e[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    counts = np.diff(np.r_[starts, len(ks)])
    second = np.minimum(starts + 1, len(ks) - 1)
    face1 = np.where(counts >= 2, order[second] // 3, -1)
    return e[order[starts]], counts, order[starts] // 3, face1


def _mesh_working_set(op, g, region, feather, strength):
    pts, spacing, feather, _, weights = _local_setup(op, g, region, feather)
    verts, tris = _local_mesh(np.asarray(g.triangles), weights > 0)
    return pts, spacing, weights, verts, tris, pts[verts].copy(), strength * (weights[verts] > 0)


def _blend(before: np.ndarray, filtered: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """before + weight x (filtered - before): the region weight scales the total displacement, so the result
    fades continuously into the untouched surface (which keeps its exact coordinates)."""
    new = before.copy()
    band = (weights > 0) & (weights < 1)
    new[weights >= 1] = filtered[weights >= 1]
    new[band] = before[band] + weights[band, None] * (filtered[band] - before[band])
    return new


def _finish(g, pts, verts, x, spacing, weights, info) -> tuple[Geometry, dict]:
    filtered = pts.copy()
    filtered[verts] = x
    new = _blend(pts, filtered, weights)
    _store_positions(g, new, spacing)
    return g, {**_moved_report(pts, new, weights), **info}


def _denoise_mesh(g, log, op, iterations, normal_sigma_deg, spatial_sigma, strength, region, feather):
    pts, spacing, weights, verts, tris, x, step = _mesh_working_set(op, g, region, feather, strength)
    sigma_s = spatial_sigma if spatial_sigma > 0 else spacing
    sigma_r = 2 * math.sin(math.radians(normal_sigma_deg) / 2)
    faces = _FaceFilter(tris, len(x))
    log(f"  {op}: {int((step > 0).sum()):,} vertices to filter ({len(tris):,} faces in the working set), "
        f"spatial sigma {sigma_s:.4g}, normal sigma {normal_sigma_deg:g} deg")
    for it in range(iterations):
        normals, centers, areas = _face_frames(x, tris)
        filtered = faces.filter_normals(normals, centers, areas, sigma_s, sigma_r, passes=2)
        x = faces.fit_vertices(x, filtered, step, 10)
        log(f"  {op}: iteration {it + 1}/{iterations}")
    return _finish(g, pts, verts, x, spacing, weights, {"spatial_sigma": sigma_s})


def _smooth_mesh(g, log, iterations, method, strength, preserve_edges, edge_angle, region, feather):
    pts, spacing, weights, verts, tris, x, step = _mesh_working_set("smooth", g, region, feather, strength)
    n = len(x)
    edges, counts, f0, f1 = _mesh_edges(tris, n)
    feature = np.zeros(len(edges), bool)
    if preserve_edges:
        # sharp edges from bilaterally pre-filtered face normals, so scanner noise does not look like an edge;
        # open boundaries and non-manifold edges count as feature edges too
        normals, centers, areas = _face_frames(x, tris)
        smoothed = _FaceFilter(tris, n).filter_normals(normals, centers, areas, spacing,
                                                       2 * math.sin(math.radians(edge_angle) / 2), passes=4)
        feature = counts != 2
        two = ~feature
        feature[two] = np.einsum("ij,ij->i", smoothed[f0[two]], smoothed[f1[two]]) < math.cos(math.radians(edge_angle))
    # umbrella operator: free vertices average all neighbours, crease vertices (two feature edges) only their
    # crease neighbours, corners and feature end points stay fixed
    fcount = np.bincount(edges[feature].ravel(), minlength=n)
    a, b = edges[:, 0], edges[:, 1]
    use_a = (fcount[a] == 0) | ((fcount[a] == 2) & feature)
    use_b = (fcount[b] == 0) | ((fcount[b] == 2) & feature)
    rows = np.concatenate([a[use_a], b[use_b]])
    cols = np.concatenate([b[use_a], a[use_b]])
    degree = np.bincount(rows, minlength=n).astype(float)
    umbrella = sparse.csr_matrix((1.0 / degree[rows], (rows, cols)), shape=(n, n))
    step = step * (degree > 0)
    moving = np.flatnonzero(step > 0)
    factors = (0.5, -0.53) if method == "taubin" else (0.5,)
    log(f"  smooth ({method}): {len(moving):,} vertices, {int(feature.sum()):,} feature edges kept sharp")
    if method == "laplacian":
        log(f"  WARNING: {LAPLACIAN_MESH_WARNING}")
    for _ in range(iterations):
        for f in factors:
            lap = umbrella @ x - x
            x[moving] += (f * step[moving])[:, None] * lap[moving]
    info = {"feature_edges": int(feature.sum())} if preserve_edges else {}
    if method == "laplacian":
        info["warning"] = LAPLACIAN_MESH_WARNING
    return _finish(g, pts, verts, x, spacing, weights, info)


# --------------------------------------------------------------------------- local edits: point clouds
def _denoise_cloud(g, log, op, iterations, normal_sigma_deg, spatial_sigma, strength, region, feather):
    """Bilateral filtering along normals: each point's normal is filtered with spatial x normal-difference
    weights, then the point moves along it to the weighted mean height of its (same-surface) neighbours."""
    pts, spacing, _, _, weights = _local_setup(op, g, region, feather)
    log(f"  WARNING ({op}): {DENOISE_CLOUD_WARNING}")
    sigma_s = spatial_sigma if spatial_sigma > 0 else 2 * spacing
    sigma_r = 2 * math.sin(math.radians(normal_sigma_deg) / 2)
    radius = 2.5 * sigma_s
    active = np.flatnonzero(weights > 0)
    log(f"  {op}: {len(active):,} points, spatial sigma {sigma_s:.4g}, normal sigma {normal_sigma_deg:g} deg")
    x = pts.copy()
    for it in range(iterations):
        sup = _support(x, active, radius + 3 * spacing)
        normals = np.zeros_like(x)
        normals[sup] = _pca_normals(x[sup], 5 * spacing)
        new = x.copy()

        def filter_block(ids, d, nb, valid, x=x, normals=normals, new=new):
            nj = np.take(normals, nb, axis=0)
            dot = np.einsum("mi,mki->mk", np.take(normals, ids, axis=0), nj)
            w = valid * np.exp(-d * d / (2 * sigma_s * sigma_s)) * np.exp((np.abs(dot) - 1) / (sigma_r * sigma_r))
            nf = _normalize_rows(np.einsum("mk,mki->mi", w * np.where(dot < 0, -1.0, 1.0), nj))
            p = np.take(x, ids, axis=0)
            h = np.einsum("mki,mi->mk", np.take(x, nb, axis=0) - p[:, None], nf)
            delta = (w * h).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-300)
            delta[valid.sum(axis=1) < 7] = 0.0  # too few neighbours for a trustworthy normal: leave it
            new[ids] = p + (strength * delta)[:, None] * nf

        _each_neighbourhood(x, active, radius, 48, filter_block)
        x = new
        log(f"  {op}: iteration {it + 1}/{iterations}")
    x = _blend(pts, x, weights)
    _store_positions(g, x, spacing)
    return g, {**_moved_report(pts, x, weights), "spatial_sigma": sigma_s, "warning": DENOISE_CLOUD_WARNING}


def _mls_cloud(g, log, op, iterations, radius, order, strength, preserve_edges, edge_angle, region, feather):
    """Moving least squares: project each point onto a Gaussian-weighted plane / quadratic of its neighbours."""
    pts, spacing, _, _, weights = _local_setup(op, g, region, feather)
    radius = radius if radius > 0 else 6 * spacing
    h = radius / 2
    cos_lim = math.cos(math.radians(edge_angle))
    active = np.flatnonzero(weights > 0)
    log(f"  {op}: {len(active):,} points, radius {radius:.4g}, {'plane' if order == 1 else 'quadratic'} fit")
    x = pts.copy()
    normals = None
    if preserve_edges:  # which neighbours lie on the same surface; estimated once from the input
        sup = _support(x, active, radius + 2 * spacing)
        normals = np.zeros_like(x)
        normals[sup] = _pca_normals(x[sup], max(0.75 * radius, 3 * spacing))
    for it in range(iterations):
        new = x.copy()

        def mls_block(ids, d, nb, valid, x=x, normals=normals, new=new):
            w = valid * np.exp(-(d / h) ** 2)
            if normals is not None:
                w *= np.abs(np.einsum("mi,mki->mk", np.take(normals, ids, axis=0), np.take(normals, nb, axis=0))) >= cos_lim
            p = np.take(x, ids, axis=0)
            target = _mls_target(p, np.take(x, nb, axis=0), w, order, radius)
            new[ids] = p + strength * (target - p)

        _each_neighbourhood(x, active, radius, 64, mls_block)
        x = new
        log(f"  {op}: pass {it + 1}/{iterations}")
    x = _blend(pts, x, weights)
    _store_positions(g, x, spacing)
    return g, {**_moved_report(pts, x, weights), "radius": radius}


def _mls_target(p: np.ndarray, q: np.ndarray, w: np.ndarray, order: int, scale: float) -> np.ndarray:
    count = (w > 1e-12).sum(axis=1)
    sw = np.maximum(w.sum(axis=1), 1e-300)
    c = np.einsum("mk,mki->mi", w, q) / sw[:, None]
    d = q - c[:, None]
    cov = (d * w[..., None]).transpose(0, 2, 1) @ d
    _, vecs = np.linalg.eigh(cov)
    nrm, e2, e1 = vecs[:, :, 0], vecs[:, :, 1], vecs[:, :, 2]
    rel = p - c
    hp = np.einsum("mi,mi->m", rel, nrm)
    shift = -hp
    if order == 2:
        u, v = np.einsum("mki,mi->mk", d, e1) / scale, np.einsum("mki,mi->mk", d, e2) / scale
        hq = np.einsum("mki,mi->mk", d, nrm)
        coef = _weighted_quadric(_quadric_design(u, v), hq, w)
        u0, v0 = np.einsum("mi,mi->m", rel, e1) / scale, np.einsum("mi,mi->m", rel, e2) / scale
        h0 = np.einsum("mi,mi->m", coef, _quadric_design(u0, v0))
        quad = h0 - hp
        good = (count >= 10) & np.isfinite(quad) & (np.abs(quad) <= scale)
        shift = np.where(good, quad, shift)
    shift = np.where(count >= 4, shift, 0.0)
    return p + shift[:, None] * nrm


# --------------------------------------------------------------------------- local edits: spikes
class _SpikeReference:
    """Robust local quadratic surfaces (Tukey-biweight IRLS) fitted around anchors spread over the surface.
    Patches that stick out of the surface get zero weight, so the fit follows the surface underneath."""

    def __init__(self, points: np.ndarray, normals: np.ndarray, radius: float, spacing: float, oriented: bool):
        self.radius = radius
        pcd = o3d.geometry.PointCloud(V3(points))
        pcd.normals = V3(normals)
        anchors = pcd.voxel_down_sample(radius / 3)
        self.anchors = np.asarray(anchors.points).copy()
        anchor_normals = np.asarray(anchors.normals).copy()
        # a random subsample of the raw points (~150 per neighbourhood) keeps the fits cheap and the noise real
        frac = min(1.0, 150 * spacing * spacing / (math.pi * radius * radius))
        keep = np.random.default_rng(0).random(len(points)) < frac if frac < 1 else np.ones(len(points), bool)
        sample, sample_normals = points[keep], normals[keep]
        tree = cKDTree(sample)
        k = int(min(192, len(sample)))
        m = len(self.anchors)
        self.frame = np.zeros((m, 3, 3))
        self.coef = np.zeros((m, 6))
        self.sigma = np.zeros(m)
        self.ok = np.zeros(m, bool)

        def fit_block(block):
            sl = slice(*block)
            a = self.anchors[sl]
            d, nb = tree.query(a, k=k, distance_upper_bound=radius, workers=1)
            d, nb = d.reshape(len(a), k), nb.reshape(len(a), k)
            valid = np.isfinite(d)
            nb = np.minimum(nb, len(sample) - 1)
            d = np.where(valid, d, radius)
            q = sample[nb] - a[:, None]
            w0 = valid * (1 - (d / radius) ** 2) ** 2
            rows = np.arange(len(a))
            floor = 1e-3 * spacing
            w = w0.copy()
            compat_checked = False
            # concentration steps on the better half first (least trimmed squares), so a spike or a nearby
            # feature cannot drag the start of the reweighting away from the majority surface; the frame comes
            # from a weighted PCA about the weighted centroid and is refined once the outliers are down-weighted
            for step in range(9):
                if step in (0, 3):
                    sw = np.maximum(w.sum(axis=1), 1e-300)
                    centred = q - (np.einsum("mk,mki->mi", w, q) / sw[:, None])[:, None]
                    e3 = np.linalg.eigh((centred * w[..., None]).transpose(0, 2, 1) @ centred)[1][:, :, 0]
                    if oriented:
                        e3[np.einsum("ij,ij->i", e3, anchor_normals[sl]) < 0] *= -1
                    if not compat_checked:  # same side of thin walls, same surface
                        compat = np.einsum("mki,mi->mk", sample_normals[nb], e3)
                        valid &= (compat > 0.5) if oriented else (np.abs(compat) > 0.5)
                        cnt = valid.sum(axis=1)
                        mid = ((cnt - 1) // 2).clip(0, k - 1)
                        w = w * valid
                        compat_checked = True
                    e1, e2 = _perpendicular_frame(e3)
                    u, v = np.einsum("mki,mi->mk", q, e1) / radius, np.einsum("mki,mi->mk", q, e2) / radius
                    h = np.einsum("mki,mi->mk", q, e3)
                    design = _quadric_design(u, v)
                coef = _weighted_quadric(design, h, w)
                res = h - np.einsum("mki,mi->mk", design, coef)
                ranked = np.sort(np.where(valid, np.abs(res), np.inf), axis=1)
                med = np.where(cnt > 0, ranked[rows, mid], 0.0)
                if step < 3:
                    w = w0 * valid * (np.abs(res) <= med[:, None])
                    continue
                s = np.maximum(1.4826 * med, floor)
                c = 4.685 * s
                w = w0 * valid * np.where(np.abs(res) < c[:, None], (1 - (res / c[:, None]) ** 2) ** 2, 0.0)
            inliers = (w > 0).sum(axis=1)
            self.frame[sl] = np.stack([e1, e2, e3], axis=1)
            self.coef[sl], self.sigma[sl] = coef, s
            self.ok[sl] = (cnt >= 15) & (inliers >= 0.5 * np.maximum(cnt, 1)) & np.isfinite(coef).all(axis=1)

        _parallel(fit_block, [(b, min(b + 1024, m)) for b in range(0, m, 1024)])
        self.good = np.flatnonzero(self.ok)
        if len(self.good):  # a neighbourhood that happens to look quieter than the scan must not lower the bar
            self.sigma = np.maximum(self.sigma, 0.75 * np.median(self.sigma[self.good]))
        self.tree = cKDTree(self.anchors[self.good]) if len(self.good) else None

    def evaluate(self, p: np.ndarray):
        """(foot point on the reference, signed distance along the frame normal, local noise sigma, ok)."""
        n = len(p)
        foot, resid, sigma, ok = p.copy(), np.zeros(n), np.zeros(n), np.zeros(n, bool)
        if self.tree is None:
            return foot, resid, sigma, ok
        k = int(min(4, len(self.good)))
        for start in range(0, n, 250_000):
            sl = slice(start, min(start + 250_000, n))
            q = p[sl]
            d, ai = self.tree.query(q, k=k, workers=-1)
            d, ai = d.reshape(len(q), k), self.good[ai.reshape(len(q), k)]
            wsum, rsum, ssum = np.zeros(len(q)), np.zeros(len(q)), np.zeros(len(q))
            fsum = np.zeros_like(q)
            ref_normal = self.frame[ai[:, 0], 2]
            for j in range(k):
                a = ai[:, j]
                fr = self.frame[a]
                rel = q - self.anchors[a]
                u = np.einsum("mi,mi->m", rel, fr[:, 0]) / self.radius
                v = np.einsum("mi,mi->m", rel, fr[:, 1]) / self.radius
                r = np.einsum("mi,mi->m", rel, fr[:, 2]) - np.einsum("mi,mi->m", self.coef[a], _quadric_design(u, v))
                wj = 1.0 / (d[:, j] + 1e-3 * self.radius) ** 2
                sign = np.where(np.einsum("mi,mi->m", fr[:, 2], ref_normal) < 0, -1.0, 1.0)
                fsum += wj[:, None] * (q - r[:, None] * fr[:, 2])
                rsum += wj * r * sign
                ssum += wj * self.sigma[a]
                wsum += wj
            foot[sl] = fsum / wsum[:, None]
            resid[sl], sigma[sl] = rsum / wsum, ssum / wsum
            ok[sl] = d[:, 0] <= self.radius / 2
        return foot, resid, sigma, ok


def _patch_labels(g: Geometry, idx: np.ndarray, lateral: np.ndarray, spacing: float) -> np.ndarray:
    """Connected patches among points / vertices idx: mesh edges for meshes; for clouds 4x spacing proximity of
    `lateral` (the points projected onto the reference surface, so steep spike flanks stay connected)."""
    m = len(idx)
    if is_cloud(g):
        pairs = cKDTree(lateral).query_pairs(4 * spacing, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
    else:
        n = len(g.vertices)
        flagged = np.zeros(n, bool)
        flagged[idx] = True
        e = np.asarray(g.triangles)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        e = e[flagged[e[:, 0]] & flagged[e[:, 1]]]
        remap = np.full(n, -1, np.int64)
        remap[idx] = np.arange(m)
        i, j = remap[e[:, 0]], remap[e[:, 1]]
    graph = sparse.coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(m, m))
    return connected_components(graph, directed=False)[1]


def _patch_sizes(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Bounding-box diagonal of each patch (labels 0..L-1)."""
    order = np.argsort(labels, kind="stable")
    lab = labels[order]
    starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    p = points[order]
    return np.linalg.norm(np.maximum.reduceat(p, starts, axis=0) - np.minimum.reduceat(p, starts, axis=0), axis=1)


def _op_remove_spikes(g, log, threshold, max_size, radius, direction, iterations, region, feather):
    cloud = is_cloud(g)
    pts0 = np.array(_positions(g), dtype=np.float64)
    spacing = _geometry_spacing(g, pts0)
    max_size = max_size if max_size > 0 else 25 * spacing
    radius = radius if radius > 0 else 1.5 * max_size
    pts, spacing, feather, dist, weights = _local_setup("remove_spikes", g, region, feather, radius, spacing)
    support = np.flatnonzero(np.isfinite(dist))
    active = np.flatnonzero(weights > 0)
    oriented = True
    normals = None
    if cloud:
        if g.has_normals():
            normals = _normalize_rows(np.asarray(g.normals).copy())
        elif direction == "both":
            oriented = False
            normals = np.zeros_like(pts)
            normals[support] = _pca_normals(pts[support], 3 * spacing)
        else:
            from .clean import ensure_normals

            tmp = o3d.geometry.PointCloud(V3(pts))
            ensure_normals(tmp, spacing, recompute=True, log=log)
            normals = np.asarray(tmp.normals).copy()
    log(f"  remove_spikes: {len(active):,} {'points' if cloud else 'vertices'}, max size {max_size:.4g}, "
        f"reference radius {radius:.4g}, direction {direction}")
    x = pts.copy()
    removed, kept, thresholds = 0, 0, []
    for it in range(iterations):
        if not cloud:
            normals = _vertex_normals(x, np.asarray(g.triangles))
        ref = _SpikeReference(x[support], normals[support], radius, spacing, oriented)
        foot, resid, sigma, ok = ref.evaluate(x[active])
        thr = np.full(len(active), threshold) if threshold > 0 else np.maximum(6 * sigma, 0.05 * spacing)
        thresholds.append(float(np.median(thr[ok])) if ok.any() else float(threshold))
        signed = {"outward": resid, "inward": -resid, "both": np.abs(resid)}[direction]
        # hysteresis: a patch is everything connected above half the threshold, and it is a spike when some
        # of it is above the threshold (a feature does not fall apart where it dips a little)
        weak = ok & (signed > 0.5 * thr)
        strong = ok & (signed > thr)
        idx = active[weak]
        if not strong.any():
            log(f"  pass {it + 1}: nothing sticks out")
            break
        labels = _patch_labels(g, idx, foot[weak], spacing)
        has_strong = np.bincount(labels, weights=strong[weak].astype(float)) > 0
        fits = _patch_sizes(foot[weak], labels) <= max_size  # lateral extent on the reference surface
        small = fits & has_strong
        remove = idx[small[labels]]
        kept = int((~fits & has_strong).sum())
        log(f"  pass {it + 1}: {int(small.sum()):,} spike patch(es) removed, {kept:,} larger feature(s) kept")
        if len(remove) == 0:
            break
        removed = max(removed, int(small.sum()))
        d, _ = cKDTree(x[remove]).query(x[active], k=1, distance_upper_bound=feather, workers=-1)
        w = _falloff(d, feather) * ok
        moving = np.flatnonzero(w > 0)
        target = active[moving]
        x[target] += w[moving, None] * (foot[moving] - x[target])
    x = _blend(pts, x, weights)
    _store_positions(g, x, spacing)
    return g, {**_moved_report(pts, x, weights), "removed_patches": removed, "kept_features": kept,
               "threshold": thresholds[-1] if thresholds else threshold, "max_size": max_size, "radius": radius}


# --------------------------------------------------------------------------- local edits: ops
def _op_smooth(g, log, iterations, method, strength, preserve_edges, edge_angle, region, feather):
    if method == "bilateral":
        denoise = _denoise_cloud if is_cloud(g) else _denoise_mesh
        return denoise(g, log, "smooth", iterations, 20.0, 0.0, strength, region, feather)
    if is_cloud(g):
        order = 2 if method == "taubin" else 1
        return _mls_cloud(g, log, "smooth", iterations, 0.0, order, strength, preserve_edges, edge_angle,
                          region, feather)
    return _smooth_mesh(g, log, iterations, method, strength, preserve_edges, edge_angle, region, feather)


def _op_denoise(g, log, iterations, normal_sigma_deg, spatial_sigma, strength, region, feather):
    denoise = _denoise_cloud if is_cloud(g) else _denoise_mesh
    return denoise(g, log, "denoise", iterations, normal_sigma_deg, spatial_sigma, strength, region, feather)


def _op_smooth_points(g, log, radius, order, strength, iterations, preserve_edges, edge_angle, region, feather):
    return _mls_cloud(g, log, "smooth_points", iterations, radius, order, strength, preserve_edges, edge_angle,
                      region, feather)


_OPS: dict[str, Callable] = {
    "crop_box": _op_crop_box, "cut_plane": _op_cut_plane, "select_screen": _op_select_screen,
    "delete_sphere": _op_delete_sphere, "delete_region": _op_delete_region, "keep_region": _op_keep_region,
    "transform": _op_transform, "translate": _op_translate,
    "rotate": _op_rotate, "scale": _op_scale, "mirror": _op_mirror, "center": _op_center,
    "align_principal": _op_align_principal, "align_floor": _op_align_floor, "downsample": _op_downsample,
    "remove_outliers": _op_remove_outliers, "remove_small_components": _op_remove_small_components,
    "simplify": _op_simplify, "smooth": _op_smooth, "fill_holes": _op_fill_holes, "subdivide": _op_subdivide,
    "repair": _op_repair, "flip_normals": _op_flip_normals, "recompute_normals": _op_recompute_normals,
    "to_pointcloud": _op_to_pointcloud, "paint": _op_paint, "denoise": _op_denoise,
    "smooth_points": _op_smooth_points, "remove_spikes": _op_remove_spikes,
}
assert set(_OPS) == set(OP_CATALOGUE)


# --------------------------------------------------------------------------- entry point
def apply_edits(geom: Geometry, ops: list[dict], log: Log = print,
                on_op: OnOp | None = None) -> tuple[Geometry, dict]:
    """Apply `ops` in order to a copy of `geom`. `on_op(index, total, op_name)` is called before each op.

    Report: {"ops": [{"op", "args", "before", "after", ...}], "transform": accumulated 4x4 (row-major) of
    all positional ops, or None when nothing was moved}."""
    ops = validate_ops(ops, PC if is_cloud(geom) else MESH)
    geom = _copy(geom)
    total, moved = np.eye(4), False
    entries = []
    for i, op in enumerate(ops):
        name = op["op"]
        args = {k: v for k, v in op.items() if k != "op"}
        if on_op is not None:
            on_op(i, len(ops), name)
        before = _counts(geom)
        geom, info = _OPS[name](geom, log, **args)
        if _is_empty(geom):
            raise ValueError(f"'{name}' would remove everything - nothing left to save")
        matrix = info.pop("_matrix", None)
        if matrix is not None:
            total, moved = matrix @ total, True
            info["matrix"] = matrix.tolist()
        after = _counts(geom)
        entries.append({"op": name, "args": args, "before": before, "after": after, **info})
        change = ", ".join(f"{k} {before.get(k, 0):,} -> {v:,}" for k, v in after.items())
        log(f"{name}: {change}")
    return geom, {"ops": entries, "transform": total.tolist() if moved else None}


# --------------------------------------------------------------------------- measuring
def _closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Exact (float64) closest point on triangle (a, b, c) to p, row-wise."""
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    safe = np.where(nn > 0, nn, 1.0)
    q = p - (np.einsum("ij,ij->i", p - a, n) / safe)[:, None] * n
    d00, d01, d11 = (np.einsum("ij,ij->i", x, y) for x, y in ((ab, ab), (ab, ac), (ac, ac)))
    aq = q - a
    d20, d21 = np.einsum("ij,ij->i", aq, ab), np.einsum("ij,ij->i", aq, ac)
    denom = d00 * d11 - d01 * d01
    denom_safe = np.where(denom > 0, denom, 1.0)
    v = (d11 * d20 - d01 * d21) / denom_safe
    w = (d00 * d21 - d01 * d20) / denom_safe
    inside = (nn > 0) & (denom > 0) & (v >= 0) & (w >= 0) & (v + w <= 1)
    best = q.copy()
    best_d = np.where(inside, np.linalg.norm(p - q, axis=1), np.inf)
    for e0, e1 in ((a, b), (b, c), (c, a)):
        seg = e1 - e0
        ll = np.einsum("ij,ij->i", seg, seg)
        t = np.clip(np.einsum("ij,ij->i", p - e0, seg) / np.where(ll > 0, ll, 1.0), 0, 1)
        cand = e0 + t[:, None] * seg
        d = np.linalg.norm(p - cand, axis=1)
        better = ~inside & (d < best_d)
        best[better], best_d[better] = cand[better], d[better]
    return best


class Snapper:
    """Snaps picked positions to the nearest full-resolution point (clouds) or surface point (meshes)."""

    def __init__(self, geom: Geometry):
        self.cloud = is_cloud(geom)
        if self.cloud:
            self.points = np.asarray(geom.points).copy()
            self.tree = cKDTree(self.points)
        else:
            self.vertices = np.asarray(geom.vertices).copy()
            self.triangles = np.asarray(geom.triangles).copy()
            self.offset = self.vertices.mean(axis=0)
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(o3d.core.Tensor((self.vertices - self.offset).astype(np.float32)),
                                     o3d.core.Tensor(self.triangles.astype(np.uint32)))

    def snap(self, points) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(points, float).reshape(-1, 3)
        if self.cloud:
            dist, idx = self.tree.query(p, k=1)
            return self.points[idx], dist
        res = self.scene.compute_closest_points(o3d.core.Tensor((p - self.offset).astype(np.float32)))
        tri = self.triangles[res["primitive_ids"].numpy().astype(np.int64)]
        snapped = _closest_on_triangles(p, self.vertices[tri[:, 0]], self.vertices[tri[:, 1]],
                                        self.vertices[tri[:, 2]])
        return snapped, np.linalg.norm(snapped - p, axis=1)


def measure(snapper: Snapper, points) -> dict:
    """Snap points and report segment lengths, total path length and the angle at the middle of 3 points."""
    snapped, dist = snapper.snap(points)
    segments = np.linalg.norm(np.diff(snapped, axis=0), axis=1)
    out = {"points": snapped.tolist(), "snap_distances": dist.tolist(), "segments": segments.tolist(),
           "total": float(segments.sum())}
    if len(snapped) == 3:
        u, v = snapped[0] - snapped[1], snapped[2] - snapped[1]
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu > 0 and lv > 0:
            out["angle_deg"] = float(np.degrees(np.arccos(np.clip(u @ v / (lu * lv), -1, 1))))
    return out
