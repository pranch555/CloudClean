"""Regions: name a part of a model by a shape, by the part's own axes or by a screen selection (Contract 2).

A region is a dict with exactly one kind (plus an optional ``"invert": true``)::

    {"spheres": [[x, y, z, r], ...]}                                    brush stroke (union of spheres)
    {"box": {"min": [3], "max": [3]}}                                   world axis-aligned box
    {"view_projection": [16], "polygon": [[x, y], ...], "visible_only": bool}    screen lasso / box (NDC)
    {"obb": {"center": [3], "axes": [[3], [3], [3]], "half": [3]}}      oriented box in world coordinates
    {"cylinder": {"point": [3], "axis": [3], "radius": r, "half_length": h}}    centred at point
    {"slab": {"direction": D, "from": a, "to": b, "frame": "world" | "part"}}  a <= coordinate along D <= b
    {"end": {"direction": D, "side": "min" | "max", "length": L}}      first / last L mm of the part along D
    {"all": true}                                                       everything

``D`` is ``[x, y, z]``, ``"x" | "y" | "z"`` (world axes) or ``"length" | "width" | "height"`` (the part frame axes).
A slab in the ``part`` frame (the default for part axes) measures ``from`` / ``to`` from the part's robust minimum
along D, so ``{"direction": "length", "from": 0, "to": 10}`` is the first 10 mm of the part; in the ``world`` frame
(the default for world axes and vectors) they are world coordinates along the unit vector D.

The **part frame** (shared with :mod:`cloudclean.understand`): principal axes of the asset (area-weighted moments
for meshes, robust to stray points for clouds), refined to the smallest robust bounding box when that is clearly
smaller (L-shaped brackets, square plates), sorted longest -> shortest and named length / width / height,
right-handed; the first two point towards their largest positive world component. The origin is the robust minimum
corner (0.05 % of the points trimmed at each end of each axis), so part coordinates along each axis run from 0 to
the robust dimension. Everything is deterministic: the same data gives the same frame in every process.

`resolve_region(geom, spec)` turns any region into world shapes the browser can evaluate; `region_mask` gives a
boolean per point / mesh vertex; `region_distance` is the distance of each point to the region (0 inside), used to
feather local edits. Nothing here moves or resamples data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .io import Geometry, is_cloud

TRIM = 0.0005                          # part frame: fraction trimmed at each end of each axis (0.05 %)
PART_AXES = ("length", "width", "height")
WORLD_AXES = ("x", "y", "z")
DIRECTION_NAMES = WORLD_AXES + PART_AXES
KINDS = ("spheres", "box", "view_projection", "obb", "cylinder", "slab", "end", "all")
FRAME_SAMPLE = 200_000                 # points used to choose the frame orientation (extents use every point)
REFINE_GAIN = 0.01                     # a refined box must be > 1 % smaller than the PCA box to replace it

REGION_HELP = (
    '{"spheres": [[x, y, z, r], ...]} brush; {"box": {"min": [3], "max": [3]}} world box; '
    '{"obb": {"center": [3], "axes": [[3], [3], [3]], "half": [3]}}; '
    '{"cylinder": {"point": [3], "axis": [3], "radius": r, "half_length": h}}; '
    '{"slab": {"direction": D, "from": a, "to": b, "frame": "world"|"part"}}; '
    '{"end": {"direction": D, "side": "min"|"max", "length": L}} (first/last L mm of the part); '
    '{"all": true}; {"view_projection": [16], "polygon": [[x, y], ...], "visible_only": false} (screen selection). '
    'D = "x"|"y"|"z"|"length"|"width"|"height" or [x, y, z]; add "invert": true for everything else. '
    'null = the whole object')


def positions(geom: Geometry) -> np.ndarray:
    return np.asarray(geom.points if is_cloud(geom) else geom.vertices)


# --------------------------------------------------------------------------- validation
def _num(value, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    return float(value)


def _vec3(value, where: str) -> list[float]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        raise ValueError(f"{where} must be a list of three numbers [x, y, z], got {value!r}")
    return [_num(v, where) for v in value]


def _nonzero_vec3(value, where: str) -> list[float]:
    v = _vec3(value, where)
    if np.linalg.norm(v) == 0:
        raise ValueError(f"{where} must not be a zero vector")
    return v


def check_direction(value, where: str = "direction"):
    """'x' | 'y' | 'z' | 'length' | 'width' | 'height' (case-insensitive) or a non-zero [x, y, z]."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in DIRECTION_NAMES:
            raise ValueError(f"{where} must be one of {', '.join(DIRECTION_NAMES)} or a vector [x, y, z], "
                             f"got {value!r}")
        return key
    return _nonzero_vec3(value, where)


def _bool(value, where: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{where} must be true or false, got {value!r}")
    return bool(value)


def region_kind(spec: dict) -> str:
    return next(k for k in KINDS if k in spec)


def check_region(value, where: str = "region") -> dict:
    """Validate a region and return its normalised form (raises ValueError with a sentence)."""
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object: {REGION_HELP}")
    kinds = [k for k in KINDS if k in value]
    if len(kinds) != 1:
        raise ValueError(f"{where} must contain exactly one of 'spheres', 'box', 'view_projection' (with 'polygon'), "
                         f"'obb', 'cylinder', 'slab', 'end' or 'all', got keys {sorted(value)}: {REGION_HELP}")
    kind = kinds[0]
    allowed = {"view_projection": {"view_projection", "polygon", "visible_only"}}.get(kind, {kind}) | {"invert"}
    extra = sorted(set(value) - allowed)
    if extra:
        raise ValueError(f"{where}: unexpected key(s) {', '.join(map(repr, extra))} in a '{kind}' region")
    out = _check_kind(kind, value, where)
    if "invert" in value and _bool(value["invert"], f"{where}: 'invert'"):
        out["invert"] = True
    return out


def _check_kind(kind: str, value: dict, where: str) -> dict:
    if kind == "spheres":
        spheres = value["spheres"]
        if not isinstance(spheres, (list, tuple, np.ndarray)) or len(spheres) == 0:
            raise ValueError(f"{where}: 'spheres' must be a non-empty list of [x, y, z, radius]")
        if len(spheres) > 500_000:
            raise ValueError(f"{where}: too many spheres ({len(spheres):,}; at most 500,000)")
        out = []
        for s in spheres:
            if not isinstance(s, (list, tuple, np.ndarray)) or len(s) != 4:
                raise ValueError(f"{where}: each sphere must be [x, y, z, radius], got {s!r}")
            item = [_num(v, where) for v in s]
            if item[3] <= 0:
                raise ValueError(f"{where}: sphere radius must be > 0, got {item[3]!r}")
            out.append(item)
        return {"spheres": out}
    if kind == "box":
        box = value["box"]
        if not isinstance(box, dict) or set(box) != {"min", "max"}:
            raise ValueError(f"{where}: 'box' must be {{\"min\": [x, y, z], \"max\": [x, y, z]}}")
        lo, hi = _vec3(box["min"], f"{where} box 'min'"), _vec3(box["max"], f"{where} box 'max'")
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError(f"{where}: every box 'min' coordinate must be <= the matching 'max' coordinate")
        return {"box": {"min": lo, "max": hi}}
    if kind == "view_projection":
        if "polygon" not in value:
            raise ValueError(f"{where}: a screen region needs both 'view_projection' and 'polygon'")
        vp = value["view_projection"]
        if not isinstance(vp, (list, tuple, np.ndarray)) or len(vp) != 16:
            raise ValueError(f"{where}: 'view_projection' must be a list of 16 numbers (column-major 4x4 matrix)")
        vp = [_num(v, f"{where}: 'view_projection'") for v in vp]
        if abs(np.linalg.det(np.asarray(vp).reshape(4, 4))) < 1e-300:
            raise ValueError(f"{where}: 'view_projection' is not invertible")
        poly = value["polygon"]
        if not isinstance(poly, (list, tuple, np.ndarray)) or len(poly) < 3:
            raise ValueError(f"{where}: 'polygon' must be a list of at least three [x, y] points")
        polygon = []
        for p in poly:
            if not isinstance(p, (list, tuple, np.ndarray)) or len(p) != 2:
                raise ValueError(f"{where}: polygon points must be [x, y] pairs, got {p!r}")
            polygon.append([_num(p[0], f"{where}: 'polygon'"), _num(p[1], f"{where}: 'polygon'")])
        visible = value.get("visible_only", False)
        return {"view_projection": vp, "polygon": polygon,
                "visible_only": _bool(visible, f"{where}: 'visible_only'")}
    if kind == "obb":
        obb = value["obb"]
        if not isinstance(obb, dict) or set(obb) != {"center", "axes", "half"}:
            raise ValueError(f"{where}: 'obb' must be {{\"center\": [3], \"axes\": [[3], [3], [3]], \"half\": [3]}}")
        center = _vec3(obb["center"], f"{where} obb 'center'")
        axes = obb["axes"]
        if not isinstance(axes, (list, tuple, np.ndarray)) or len(axes) != 3:
            raise ValueError(f"{where}: obb 'axes' must be three direction vectors [[x, y, z], [...], [...]]")
        a = np.array([_nonzero_vec3(v, f"{where} obb 'axes'") for v in axes])
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        gram = np.abs(a @ a.T - np.eye(3))
        if gram.max() > 2e-2:
            raise ValueError(f"{where}: obb 'axes' must be perpendicular to each other")
        # tiny float32 errors from the browser: re-orthonormalise, keeping the first axis and the plane of the first two
        a[1] -= (a[1] @ a[0]) * a[0]
        a[1] /= np.linalg.norm(a[1])
        a[2] = np.cross(a[0], a[1]) * np.sign(np.dot(np.cross(a[0], a[1]), a[2]) or 1.0)
        half = _vec3(obb["half"], f"{where} obb 'half'")
        if any(h <= 0 for h in half):
            raise ValueError(f"{where}: obb 'half' sizes must be > 0")
        return {"obb": {"center": center, "axes": a.tolist(), "half": half}}
    if kind == "cylinder":
        cyl = value["cylinder"]
        if isinstance(cyl, dict) and "point" not in cyl:
            raise ValueError(where + ': a cylinder region needs its centre too: {"cylinder": {"point": [x, y, z], '
                             '"axis": "z" or [x, y, z], "radius": r, "half_length": h}}. For a hole, holes '
                             "action=list gives each hole's diameter and centre directly")
        if not isinstance(cyl, dict) or set(cyl) != {"point", "axis", "radius", "half_length"}:
            raise ValueError(f"{where}: 'cylinder' must be {{\"point\": [3], \"axis\": [3], \"radius\": r, "
                             "\"half_length\": h}")
        named = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
        raw = cyl["axis"]
        if isinstance(raw, str) and raw.strip().lower().lstrip("+-") in named:   # "z", "-x": a world axis
            raw = [(-1 if raw.strip().startswith("-") else 1) * c for c in named[raw.strip().lower().lstrip("+-")]]
        axis = np.asarray(_nonzero_vec3(raw, f"{where} cylinder 'axis'"))
        radius = _num(cyl["radius"], f"{where} cylinder 'radius'")
        half = _num(cyl["half_length"], f"{where} cylinder 'half_length'")
        if radius <= 0 or half <= 0:
            raise ValueError(f"{where}: cylinder 'radius' and 'half_length' must be > 0")
        return {"cylinder": {"point": _vec3(cyl["point"], f"{where} cylinder 'point'"),
                             "axis": (axis / np.linalg.norm(axis)).tolist(), "radius": radius, "half_length": half}}
    if kind == "slab":
        slab = value["slab"]
        if not isinstance(slab, dict) or not {"direction", "from", "to"} <= set(slab) \
                or set(slab) - {"direction", "from", "to", "frame"}:
            raise ValueError(f"{where}: 'slab' must be {{\"direction\": D, \"from\": a, \"to\": b, "
                             "\"frame\": \"world\"|\"part\"}")
        direction = check_direction(slab["direction"], f"{where} slab 'direction'")
        lo, hi = _num(slab["from"], f"{where} slab 'from'"), _num(slab["to"], f"{where} slab 'to'")
        if lo > hi:
            raise ValueError(f"{where}: slab 'from' must be <= 'to'")
        frame = slab.get("frame") or ("part" if direction in PART_AXES else "world")
        if frame not in ("world", "part"):
            raise ValueError(f"{where}: slab 'frame' must be 'world' or 'part', got {frame!r}")
        return {"slab": {"direction": direction, "from": lo, "to": hi, "frame": frame}}
    if kind == "end":
        end = value["end"]
        if not isinstance(end, dict) or set(end) != {"direction", "side", "length"}:
            raise ValueError(f"{where}: 'end' must be {{\"direction\": D, \"side\": \"min\"|\"max\", \"length\": L}}")
        direction = check_direction(end["direction"], f"{where} end 'direction'")
        if end["side"] not in ("min", "max"):
            raise ValueError(f"{where}: end 'side' must be 'min' or 'max', got {end['side']!r}")
        length = _num(end["length"], f"{where} end 'length'")
        if length <= 0:
            raise ValueError(f"{where}: end 'length' must be > 0")
        return {"end": {"direction": direction, "side": end["side"], "length": length}}
    if value["all"] is not True:
        raise ValueError(f"{where}: 'all' must be true")
    return {"all": True}


# --------------------------------------------------------------------------- part frame
@dataclass
class PartFrame:
    origin: np.ndarray          # world position of the robust minimum corner
    axes: np.ndarray            # rows: length, width, height (unit, right-handed)
    dims: np.ndarray            # robust extents along the axes (part coordinates run 0 .. dims)
    dims_raw: np.ndarray        # full extents along the axes

    def coords(self, points: np.ndarray) -> np.ndarray:
        """Part coordinates (length, width, height) of world points."""
        return (np.asarray(points, float) - self.origin) @ self.axes.T

    def to_world(self, coords) -> np.ndarray:
        return self.origin + np.asarray(coords, float) @ self.axes

    def axis(self, name: str) -> np.ndarray:
        return self.axes[PART_AXES.index(name)]

    def to_dict(self) -> dict:
        return {"origin": _r(self.origin), **{name: _r(self.axes[k], 9) for k, name in enumerate(PART_AXES)}}

    def dims_dict(self, raw: bool = False) -> dict:
        d = self.dims_raw if raw else self.dims
        return {name: round(float(d[k]), 6) for k, name in enumerate(PART_AXES)}

    def info(self) -> dict:
        """What asset meta stores as `part`."""
        return {"dimensions": self.dims_dict(), "dimensions_raw": self.dims_dict(True), "frame": self.to_dict()}

    @classmethod
    def from_info(cls, info: dict) -> "PartFrame":
        f = info["frame"]
        axes = np.array([f[name] for name in PART_AXES], float)
        dims = np.array([info["dimensions"][name] for name in PART_AXES], float)
        raw = np.array([(info.get("dimensions_raw") or info["dimensions"])[name] for name in PART_AXES], float)
        return cls(np.asarray(f["origin"], float), axes, dims, raw)


def _r(values, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values]


def _moments(geom, pts: np.ndarray, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and covariance: exact area-weighted surface integrals for meshes (tessellation does not bias the
    axes), plain point moments for clouds (optionally only the `keep` points)."""
    if geom is not None and not is_cloud(geom) and len(geom.triangles):
        from .edit import _moments as mesh_moments

        return mesh_moments(geom)
    p = pts if keep is None else pts[keep]
    c = p.mean(axis=0)
    x = p - c
    return c, x.T @ x / len(x)


def _sample(n: int, size: int, seed: int = 0) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size, replace=False))


def _min_area_angle(xy: np.ndarray) -> float | None:
    """Rotation (radians, 0..pi/2) of the minimum-area bounding rectangle of 2D points (rotating calipers)."""
    from scipy.spatial import ConvexHull

    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:  # collinear / degenerate
        return None
    edges = np.roll(hull, -1, axis=0) - hull
    ang = np.unique(np.round(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), math.pi / 2), 12))
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x = c * hull[:, 0] + s * hull[:, 1]
    y = -s * hull[:, 0] + c * hull[:, 1]
    area = np.ptp(x, axis=1) * np.ptp(y, axis=1)
    return float(ang[int(np.argmin(area))])


def vertex_weights(geom) -> np.ndarray | None:
    """Meshes: each vertex's share of the surface area (a third of every triangle it belongs to), so trimming is
    by surface area and a coarse tessellation (a CAD box has 8 vertices) never loses a real corner.
    Clouds: None - every point counts once."""
    if geom is None or is_cloud(geom) or len(geom.triangles) == 0:
        return None
    v, t = np.asarray(geom.vertices), np.asarray(geom.triangles)
    area = np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1) / 6
    return np.bincount(t.ravel(), weights=np.repeat(area, 3), minlength=len(v))


def _weighted_low(values: np.ndarray, weights: np.ndarray, target: float) -> float:
    """Smallest value at which the cumulative weight (ascending) exceeds `target`."""
    n = len(values)
    k = min(n, max(2000, n // 50))
    idx = np.argpartition(values, k - 1)[:k] if k < n else np.arange(n)
    for attempt in (idx, np.arange(n)):
        order = attempt[np.argsort(values[attempt], kind="stable")]
        cum = np.cumsum(weights[order])
        if cum[-1] > target or len(attempt) == n:
            return float(values[order[min(int(np.searchsorted(cum, target, side="right")), len(order) - 1)]])
    return float(values.min())  # pragma: no cover


def trimmed_range(values: np.ndarray, weights: np.ndarray | None = None, trim: float = TRIM) -> tuple[float, float]:
    """Robust (min, max) of values: `trim` of the points (or of the surface area, with weights) cut at each end."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return 0.0, 0.0
    if weights is None or trim <= 0:
        if trim <= 0:
            return float(values.min()), float(values.max())
        lo, hi = np.percentile(values, [100 * trim, 100 * (1 - trim)])
        return float(lo), float(hi)
    total = float(weights.sum())
    if total <= 0:
        return float(values.min()), float(values.max())
    return _weighted_low(values, weights, trim * total), -_weighted_low(-values, weights, trim * total)


def _robust_extents(coords: np.ndarray, weights, trim: float) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        lo, hi = np.percentile(coords, [100 * trim, 100 * (1 - trim)], axis=0)
        return lo, hi
    ranges = [trimmed_range(coords[:, k], weights, trim) for k in range(coords.shape[1])]
    return np.array([r[0] for r in ranges]), np.array([r[1] for r in ranges])


def _box_volume(points: np.ndarray, axes: np.ndarray, trim: float) -> float:
    c = points @ axes.T
    if trim > 0:
        lo, hi = np.percentile(c, [100 * trim, 100 * (1 - trim)], axis=0)
    else:
        lo, hi = c.min(axis=0), c.max(axis=0)
    return float(np.prod(np.maximum(hi - lo, 1e-12)))


def _axes_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Largest angle between the axis lines of two frames, matching each axis of a to its closest axis of b."""
    closeness = np.abs(a @ b.T).max(axis=1)
    return float(np.degrees(np.arccos(np.clip(closeness.min(), -1.0, 1.0))))


def _refine_axes(core: np.ndarray, axes: np.ndarray, trim: float) -> np.ndarray:
    """PCA is not the natural box of every part (L-brackets, square plates, hexagon heads), and on sampled data its
    axes carry a small statistical tilt (~1/sqrt(n) rad) that inflates extents. Coordinate descent: rotate about each
    axis in turn to the minimum-area rectangle of the projection onto the other two, while the box keeps shrinking.
    The result replaces PCA when it is a small correction (<= 3 deg) that shrinks the box, or when its box is clearly
    (> 1 %) smaller - so smooth parts, whose best box is ill-defined, keep their stable PCA axes."""
    start = best = _box_volume(core, axes, trim)
    current = axes
    for _ in range(8):
        improved = False
        for k in range(3):
            i, j = [m for m in range(3) if m != k]
            theta = _min_area_angle(np.c_[core @ current[i], core @ current[j]])
            if theta is None or theta == 0.0:
                continue
            u = math.cos(theta) * current[i] + math.sin(theta) * current[j]
            v = -math.sin(theta) * current[i] + math.cos(theta) * current[j]
            candidate = current.copy()
            candidate[i], candidate[j] = u, v
            volume = _box_volume(core, candidate, trim)
            if volume < best * (1 - 1e-6):
                current, best, improved = candidate, volume, True
        if not improved:
            break
    if best < (1 - REFINE_GAIN) * start:
        return current
    if best < start * (1 - 1e-6) and _axes_angle_deg(current, axes) <= 3.0:
        return current
    return axes


def part_frame(geom_or_points, trim: float = TRIM) -> PartFrame:
    """The part frame of a geometry (or an (n, 3) array of points). See the module docstring."""
    if isinstance(geom_or_points, np.ndarray):
        geom, pts = None, np.asarray(geom_or_points, dtype=np.float64)
    else:
        geom, pts = geom_or_points, np.asarray(positions(geom_or_points), dtype=np.float64)
    n = len(pts)
    if n == 0:
        raise ValueError("The model has no points")
    cloud = geom is None or is_cloud(geom)
    weights = None if cloud else vertex_weights(geom)
    if n < 4:
        axes = np.eye(3)
    else:
        centroid, cov = _moments(geom, pts)
        axes = np.linalg.eigh(cov)[1][:, ::-1].T.copy()
        sample = pts[_sample(n, FRAME_SAMPLE // 4)]
        core = sample
        if cloud and n >= 50:
            # stray points (unremoved noise) must not tilt the axes: re-estimate without what lies outside the
            # 0.1 % trimmed box (plus a 2 % margin)
            c = (pts - centroid) @ axes.T
            lo, hi = np.percentile((sample - centroid) @ axes.T, [0.1, 99.9], axis=0)
            margin = 0.02 * (hi - lo) + 1e-12
            keep = np.all((c >= lo - margin) & (c <= hi + margin), axis=1)
            if 10 <= keep.sum() < n:
                centroid, cov = _moments(None, pts, keep)
                axes = np.linalg.eigh(cov)[1][:, ::-1].T.copy()
            c = sample @ axes.T
            lo, hi = np.percentile(c, [0.1, 99.9], axis=0)
            core = sample[np.all((c >= lo) & (c <= hi), axis=1)]
        if len(core) >= 10:
            axes = _refine_axes(core, axes, trim if cloud else 0.0)
    coords = pts @ axes.T
    lo, hi = _robust_extents(coords, weights, trim)
    rmin, rmax = coords.min(axis=0), coords.max(axis=0)
    order = sorted(range(3), key=lambda k: (-(hi[k] - lo[k]), k))
    axes, lo, hi, rmin, rmax = axes[order], lo[order], hi[order], rmin[order], rmax[order]
    for k in (0, 1):  # deterministic signs: the largest world component of length and width is positive
        if axes[k][int(np.argmax(np.abs(axes[k])))] < 0:
            axes[k] = -axes[k]
            lo[k], hi[k], rmin[k], rmax[k] = -hi[k], -lo[k], -rmax[k], -rmin[k]
    height = np.cross(axes[0], axes[1])
    if height @ axes[2] < 0:
        lo[2], hi[2], rmin[2], rmax[2] = -hi[2], -lo[2], -rmax[2], -rmin[2]
    axes[2] = height / np.linalg.norm(height)
    return PartFrame(origin=lo @ axes, axes=axes, dims=hi - lo, dims_raw=rmax - rmin)


# --------------------------------------------------------------------------- directions
def direction_vector(direction, frame: PartFrame | None = None, geom=None) -> tuple[np.ndarray, str]:
    """(unit world vector, label) for a direction D. Part axes need `frame` (computed from `geom` if missing)."""
    d = check_direction(direction)
    if isinstance(d, str):
        if d in WORLD_AXES:
            return np.eye(3)[WORLD_AXES.index(d)], d
        if frame is None:
            if geom is None:
                raise ValueError(f"'{d}' is a part axis: the part frame is needed")
            frame = part_frame(geom)
        return frame.axis(d).copy(), d
    v = np.asarray(d, float)
    v = v / np.linalg.norm(v)
    return v, "[" + ", ".join(f"{x:.4g}" for x in v) + "]"


def perpendicular_basis(d: np.ndarray, frame: PartFrame | None = None, name: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing d to a right-handed basis (the other part axes for a part axis)."""
    if frame is not None and name in PART_AXES:
        k = PART_AXES.index(name)
        return frame.axes[(k + 1) % 3].copy(), frame.axes[(k + 2) % 3].copy()
    helper = np.eye(3)[int(np.argmin(np.abs(d)))]
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def _needs_frame(spec: dict) -> bool:
    kind = region_kind(spec)
    return kind in ("slab", "end") and isinstance(spec[kind]["direction"], str) \
        and spec[kind]["direction"] in PART_AXES


def robust_range(values: np.ndarray, trim: float = TRIM, weights: np.ndarray | None = None) -> tuple[float, float]:
    return trimmed_range(values, weights, trim)


def _slab_bounds(pts: np.ndarray, spec: dict, frame: PartFrame | None, geom) -> tuple[np.ndarray, str, float, float]:
    """(unit direction, label, t_lo, t_hi): the region is t_lo <= p . d <= t_hi (world dot products)."""
    kind = region_kind(spec)
    body = spec[kind]
    d, label = direction_vector(body["direction"], frame, geom)
    name = body["direction"] if isinstance(body["direction"], str) else ""
    if name in PART_AXES and frame is not None:
        k = PART_AXES.index(name)
        base = float(frame.origin @ d)
        lo, hi = base, base + float(frame.dims[k])
    else:
        t = pts @ d
        lo, hi = robust_range(t, weights=vertex_weights(geom)) if len(t) else (0.0, 0.0)
    if kind == "slab":
        if body["frame"] == "world":
            return d, label, body["from"], body["to"]
        return d, label, lo + body["from"], lo + body["to"]
    t = pts @ d
    reach = float(np.abs(t).max()) + 1.0 if len(t) else 1.0  # beyond every point: the end region is open-ended
    if body["side"] == "min":
        return d, label, -reach, lo + body["length"]
    return d, label, hi - body["length"], reach


def _obb(center, axes, half) -> dict:
    return {"type": "obb", "center": _r(center, 9), "axes": [_r(a, 9) for a in axes], "half": _r(half, 9)}


def _cover_obb(pts: np.ndarray, d: np.ndarray, e1: np.ndarray, e2: np.ndarray, t_lo: float, t_hi: float) -> dict:
    """An oriented box spanning t_lo..t_hi along d and covering every point across it."""
    if len(pts):
        u, v = pts @ e1, pts @ e2
        diag = float(np.linalg.norm(np.ptp(pts, axis=0))) if len(pts) > 1 else 1.0
        margin = 0.01 * diag + 1e-6
        u0, u1, v0, v1 = u.min() - margin, u.max() + margin, v.min() - margin, v.max() + margin
        t_lo = max(t_lo, float((pts @ d).min()) - margin)
        t_hi = min(t_hi, float((pts @ d).max()) + margin)
        if t_hi < t_lo:
            t_hi = t_lo
    else:
        u0 = u1 = v0 = v1 = 0.0
        t_lo, t_hi = (t_lo, t_hi) if math.isfinite(t_lo) and math.isfinite(t_hi) else (0.0, 0.0)
    center = d * (t_lo + t_hi) / 2 + e1 * (u0 + u1) / 2 + e2 * (v0 + v1) / 2
    return _obb(center, [d, e1, e2], [(t_hi - t_lo) / 2, (u1 - u0) / 2, (v1 - v0) / 2])


# --------------------------------------------------------------------------- resolve / mask / distance
def resolve_region(geom: Geometry, spec: dict, frame: PartFrame | None = None) -> dict:
    """World shapes the browser can evaluate: {"kind", "shapes": [obb | sphere | cylinder | screen], "invert"}.

    Shapes: {"type": "obb", center, axes (rows), half} | {"type": "sphere", center, radius} |
    {"type": "cylinder", point, axis, radius, half_length} | {"type": "screen", view_projection, polygon,
    visible_only}. A point is in the region when it is inside any shape (inverted when "invert" is true)."""
    spec = check_region(spec)
    kind = region_kind(spec)
    pts = np.asarray(positions(geom), dtype=np.float64)
    if frame is None and _needs_frame(spec) and len(pts):
        frame = part_frame(geom)
    if kind == "spheres":
        shapes = [{"type": "sphere", "center": s[:3], "radius": s[3]} for s in spec["spheres"]]
    elif kind == "box":
        lo, hi = np.asarray(spec["box"]["min"]), np.asarray(spec["box"]["max"])
        shapes = [_obb((lo + hi) / 2, np.eye(3), (hi - lo) / 2)]
    elif kind == "view_projection":
        shapes = [{"type": "screen", "view_projection": spec["view_projection"], "polygon": spec["polygon"],
                   "visible_only": spec["visible_only"]}]
    elif kind == "obb":
        o = spec["obb"]
        shapes = [_obb(o["center"], o["axes"], o["half"])]
    elif kind == "cylinder":
        shapes = [{"type": "cylinder", **spec["cylinder"]}]
    elif kind in ("slab", "end"):
        d, _, t_lo, t_hi = _slab_bounds(pts, spec, frame, geom)
        body = spec[kind]
        e1, e2 = perpendicular_basis(d, frame, body["direction"] if isinstance(body["direction"], str) else "")
        shapes = [_cover_obb(pts, d, e1, e2, t_lo, t_hi)]
    else:  # all
        shapes = [_cover_obb(pts, np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], -np.inf, np.inf)]
    return {"kind": kind, "shapes": shapes, "invert": bool(spec.get("invert", False))}


def _positive_mask(geom: Geometry, pts: np.ndarray, spec: dict, frame: PartFrame | None) -> np.ndarray:
    """Mask of the region before 'invert', evaluated on the native definition (exact boundaries)."""
    kind = region_kind(spec)
    n = len(pts)
    if kind == "all":
        return np.ones(n, bool)
    if kind == "box":
        lo, hi = np.asarray(spec["box"]["min"]), np.asarray(spec["box"]["max"])
        return np.all((pts >= lo) & (pts <= hi), axis=1)
    if kind == "spheres":
        return _sphere_distance(pts, np.asarray(spec["spheres"], float), 0.0) <= 0
    if kind == "view_projection":
        from .edit import screen_selection

        return screen_selection(geom, spec["view_projection"], spec["polygon"], spec["visible_only"])
    if kind == "obb":
        o = spec["obb"]
        local = (pts - np.asarray(o["center"])) @ np.asarray(o["axes"]).T
        return np.all(np.abs(local) <= np.asarray(o["half"]), axis=1)
    if kind == "cylinder":
        c = spec["cylinder"]
        axial, radial = _cylinder_coords(pts, c)
        return (np.abs(axial) <= c["half_length"]) & (radial <= c["radius"])
    d, _, t_lo, t_hi = _slab_bounds(pts, spec, frame, geom)
    t = pts @ d
    return (t >= t_lo) & (t <= t_hi)


def _cylinder_coords(pts: np.ndarray, c: dict) -> tuple[np.ndarray, np.ndarray]:
    rel = pts - np.asarray(c["point"], float)
    a = np.asarray(c["axis"], float)
    axial = rel @ a
    return axial, np.linalg.norm(rel - axial[:, None] * a, axis=1)


def region_mask(geom: Geometry, spec: dict | None, frame: PartFrame | None = None) -> np.ndarray:
    """Boolean per point (clouds) or vertex (meshes); None = everything."""
    pts = np.asarray(positions(geom), dtype=np.float64)
    if spec is None:
        return np.ones(len(pts), bool)
    spec = check_region(spec)
    if frame is None and _needs_frame(spec) and len(pts):
        frame = part_frame(geom)
    mask = _positive_mask(geom, pts, spec, frame)
    return ~mask if spec.get("invert") else mask


def _sphere_distance(pts: np.ndarray, spheres: np.ndarray, cutoff: float) -> np.ndarray:
    """Distance to the union of spheres (0 inside), inf where farther than cutoff."""
    dist = np.full(len(pts), np.inf)
    centers, radii = spheres[:, :3], spheres[:, 3]
    reach = radii + cutoff
    lo, hi = (centers - reach[:, None]).min(axis=0), (centers + reach[:, None]).max(axis=0)
    cand = np.flatnonzero(np.all((pts >= lo) & (pts <= hi), axis=1))
    if len(cand) == 0:
        return dist
    tree = cKDTree(pts[cand])
    best = np.full(len(cand), np.inf)
    for start in range(0, len(spheres), 512):
        stop = min(start + 512, len(spheres))
        hits = tree.query_ball_point(centers[start:stop], reach[start:stop], workers=-1)
        lens = np.fromiter((len(h) for h in hits), np.int64, len(hits))
        if lens.sum() == 0:
            continue
        idx = np.concatenate([np.asarray(h, np.int64) for h in hits if len(h)])
        sid = np.repeat(np.arange(start, stop), lens)
        d = np.linalg.norm(pts[cand[idx]] - centers[sid], axis=1) - radii[sid]
        np.minimum.at(best, idx, d)
    best = np.maximum(best, 0.0)
    dist[cand] = np.where(best <= cutoff, best, np.inf)
    return dist


def _mask_distance(pts: np.ndarray, inside: np.ndarray, cutoff: float) -> np.ndarray:
    """0 for points in the mask, the distance to the nearest mask point for the others (inf beyond cutoff)."""
    dist = np.full(len(pts), np.inf)
    if not inside.any():
        return dist
    dist[inside] = 0.0
    if cutoff > 0:
        inner = pts[inside]
        lo, hi = inner.min(axis=0) - cutoff, inner.max(axis=0) + cutoff
        cand = np.flatnonzero(~inside & np.all((pts >= lo) & (pts <= hi), axis=1))
        if len(cand):
            d, _ = cKDTree(inner).query(pts[cand], k=1, distance_upper_bound=cutoff, workers=-1)
            dist[cand] = d
    return dist


def region_distance(geom: Geometry, spec: dict | None, cutoff: float, frame: PartFrame | None = None) -> np.ndarray:
    """Distance of every point / vertex to the region (0 inside); inf where it is farther than `cutoff`.

    Boxes, spheres, oriented boxes, cylinders, slabs and ends are measured to the shape itself; screen selections
    and inverted regions to the nearest point that is in the region."""
    pts = np.asarray(positions(geom), dtype=np.float64)
    n = len(pts)
    if spec is None:
        return np.zeros(n)
    spec = check_region(spec)
    kind = region_kind(spec)
    if frame is None and _needs_frame(spec) and n:
        frame = part_frame(geom)
    if spec.get("invert") or kind == "view_projection":
        return _mask_distance(pts, region_mask(geom, spec, frame), cutoff)
    if kind == "all":
        return np.zeros(n)
    if kind == "spheres":
        return _sphere_distance(pts, np.asarray(spec["spheres"], float), cutoff)
    if kind == "box":
        lo, hi = np.asarray(spec["box"]["min"]), np.asarray(spec["box"]["max"])
        d = np.linalg.norm(np.maximum(np.maximum(lo - pts, pts - hi), 0), axis=1)
    elif kind == "obb":
        o = spec["obb"]
        local = (pts - np.asarray(o["center"])) @ np.asarray(o["axes"]).T
        d = np.linalg.norm(np.maximum(np.abs(local) - np.asarray(o["half"]), 0), axis=1)
    elif kind == "cylinder":
        c = spec["cylinder"]
        axial, radial = _cylinder_coords(pts, c)
        d = np.hypot(np.maximum(np.abs(axial) - c["half_length"], 0), np.maximum(radial - c["radius"], 0))
    else:  # slab / end: only the distance along the direction counts
        dvec, _, t_lo, t_hi = _slab_bounds(pts, spec, frame, geom)
        t = pts @ dvec
        d = np.maximum(np.maximum(t_lo - t, t - t_hi), 0)
    return np.where(d <= cutoff, d, np.inf)


def describe_region(spec: dict) -> str:
    """Short human description of a (normalised) region, e.g. 'last 10 mm along length'."""
    spec = check_region(spec)
    kind = region_kind(spec)
    body = spec[kind]
    if kind == "spheres":
        text = f"{len(body)} brush sphere(s)"
    elif kind == "box":
        text = "world box " + " .. ".join("[" + ", ".join(f"{v:.4g}" for v in body[k]) + "]" for k in ("min", "max"))
    elif kind == "view_projection":
        text = f"screen selection ({len(spec['polygon'])}-point outline{', visible only' if spec['visible_only'] else ''})"
    elif kind == "obb":
        text = "oriented box " + " x ".join(f"{2 * h:.4g}" for h in body["half"])
    elif kind == "cylinder":
        text = f"cylinder dia {2 * body['radius']:.4g}, length {2 * body['half_length']:.4g}"
    elif kind == "slab":
        d = body["direction"] if isinstance(body["direction"], str) else "vector"
        text = f"slab {body['from']:.4g}..{body['to']:.4g} along {d} ({body['frame']} frame)"
    elif kind == "end":
        d = body["direction"] if isinstance(body["direction"], str) else "vector"
        text = f"{'first' if body['side'] == 'min' else 'last'} {body['length']:.4g} along {d}"
    else:
        text = "everything"
    return ("everything except " + text) if spec.get("invert") else text


def subset(geom: Geometry, mask: np.ndarray) -> Geometry:
    """The masked points of a cloud, or the masked vertices of a mesh with the triangles entirely inside."""
    import open3d as o3d

    if is_cloud(geom):
        return geom.select_by_index(np.flatnonzero(mask).tolist())
    from .edit import _mesh_subset

    tris = np.asarray(geom.triangles)
    if len(tris) == 0:
        return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(positions(geom)[mask]),
                                         o3d.utility.Vector3iVector(np.zeros((0, 3), np.int32)))
    return _mesh_subset(geom, mask[tris].all(axis=1))
