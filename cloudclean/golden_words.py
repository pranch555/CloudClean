"""Plain words for the parts of a golden model (docs/golden-model.md, "Names").

The golden check names its areas and measurements after the part, not after coordinates: "Floor of the recess in
the wide end", "Step 25.40 mm from the wide end", "Edge of the narrow end", "Inner corner at the step...",
"Ø6.00 hole (2 of 4)". CAD files use any axis as 'up' and the 3D view can be turned any way, so the words lean on
landmarks of the part itself:

* Turned and long parts have a main axis: the axis of their round faces, else their longest side when it is clearly
  the longest. Its two ends are "the wide end" and "the narrow end" when their cross-sections differ, else they are
  named from the 3D view's default front view ("top end", "left end").
* Other parts are named from that default front view (up = Y or Z, as set in Settings → 3D view).
* Ray casting on the golden surface tells outer faces from recesses (a floor with walls all round), steps and inner
  corners, and finds a direction to look at an area from that nothing else blocks ("Show me").
"""
from __future__ import annotations

import numpy as np
import open3d.core as o3c

COS5 = float(np.cos(np.radians(5.0)))
SIN5 = float(np.sin(np.radians(5.0)))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def view_words(up_axis: str = "y") -> dict[int, tuple[str, str]]:
    """(word for the - end, word for the + end) of each axis in the viewer's default front view
    (frontend/src/viewer/Viewer.ts: Y up -> the camera looks from +Z; Z up -> from -Y; X is to the right)."""
    if up_axis == "z":
        return {0: ("left", "right"), 1: ("front", "back"), 2: ("bottom", "top")}
    return {0: ("left", "right"), 1: ("bottom", "top"), 2: ("back", "front")}


def _mm(x: float) -> str:
    """An exact size from the golden model (a face's position)."""
    return f"{x:.2f}" if x < 100 else f"{x:.1f}"


def _approx(x: float) -> str:
    """Roughly where an area is: no false precision."""
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _perp_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(n, a))
    return u, np.cross(n, u)


def _fibonacci(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


class PartWords:
    """Names for the faces, areas and sizes of one golden model (ref: compare.ReferenceSurface, faces: find_faces)."""

    def __init__(self, ref, faces: list[dict], up_axis: str = "y"):
        self.ref, self.faces = ref, faces
        V = ref.vertices
        self.lo, self.hi = V.min(0), V.max(0)
        self.size = self.hi - self.lo
        self.mid = (self.lo + self.hi) / 2
        self.diag = float(np.linalg.norm(self.size))
        self.eps = max(1e-3 * self.diag, 0.02)
        self.view = view_words(up_axis)
        self.samples, self.sample_normals = ref.sample(20_000, seed=3)
        self.axis = self._main_axis()
        self.ends = self._end_names()
        self.info = [self._classify(f) for f in faces]
        self.labels = self._labels()

    # ------------------------------------------------------------------ ray casting
    def cast(self, origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the first hit of each ray on the golden surface (inf: none) and the triangle hit."""
        if len(origins) == 0:
            return np.zeros(0), np.zeros(0, dtype=np.int64)
        rel = np.asarray(origins, dtype=np.float64) - self.ref.center
        rays = np.hstack([rel, np.asarray(dirs, dtype=np.float64)]).astype(np.float32)
        ans = self.ref.scene.cast_rays(o3c.Tensor(rays))
        return ans["t_hit"].numpy().astype(np.float64), ans["primitive_ids"].numpy().astype(np.int64)

    def _face_points(self, f: dict, count: int = 24) -> np.ndarray:
        """Area-weighted points on a face (triangle centroids, nudged off the surface along its outward normal)."""
        t = f["tris"]
        if len(t) > count:
            p = self.ref.areas[t] / self.ref.areas[t].sum()
            t = np.random.default_rng(1).choice(t, count, replace=False, p=p)
        c = self.ref.vertices[self.ref.triangles[t]].mean(1)
        return c + self.ref.normals[t] * self.eps

    def _enclosed(self, pts: np.ndarray, n: np.ndarray) -> float:
        """Share of sideways rays (across n) from just above a face that run into the part: ~1 inside a recess."""
        u, v = _perp_basis(n)
        a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        dirs = np.cos(a)[:, None] * u + np.sin(a)[:, None] * v
        O = np.repeat(pts, len(dirs), axis=0)
        D = np.tile(dirs, (len(pts), 1))
        t, _ = self.cast(O, D)
        return float(np.mean(t < 0.75 * self.diag)) if len(t) else 0.0

    def _blocked(self, pts: np.ndarray, d: np.ndarray, limit: float) -> float:
        t, _ = self.cast(pts, np.tile(d, (len(pts), 1)))
        return float(np.mean(t < limit)) if len(t) else 0.0

    # ------------------------------------------------------------------ the part's frame
    def _main_axis(self) -> int | None:
        total = float(self.ref.areas.sum())
        by_axis = np.zeros(3)
        for f in self.faces:
            if f["type"] == "cylinder":
                k = int(np.argmax(np.abs(f["axis"])))
                if abs(f["axis"][k]) >= COS5:
                    by_axis[k] += f["area"]
        if by_axis.max() >= 0.15 * total:
            return int(np.argmax(by_axis))
        order = np.argsort(self.size)[::-1]
        if self.size[order[0]] >= 1.3 * self.size[order[1]]:
            return int(order[0])
        return None

    def _radial(self, P: np.ndarray) -> np.ndarray:
        k = self.axis
        R = P - self.mid
        R[:, k] = 0
        return np.linalg.norm(R, axis=1)

    def _end_names(self) -> tuple[str, str] | None:
        k = self.axis
        if k is None:
            return None
        L = self.size[k]
        P = self.samples
        r = self._radial(P.copy())
        near_lo, near_hi = P[:, k] <= self.lo[k] + 0.1 * L, P[:, k] >= self.hi[k] - 0.1 * L
        if near_lo.sum() >= 20 and near_hi.sum() >= 20:
            w_lo, w_hi = np.percentile(r[near_lo], 98), np.percentile(r[near_hi], 98)
            if max(w_lo, w_hi) >= 1.12 * max(min(w_lo, w_hi), 1e-9):
                return ("wide end", "narrow end") if w_lo > w_hi else ("narrow end", "wide end")
        return f"{self.view[k][0]} end", f"{self.view[k][1]} end"

    def end(self, sign: int) -> str:
        """The main axis' end on the + (sign 1) or - side."""
        assert self.ends is not None
        return self.ends[1 if sign > 0 else 0]

    def _nearest_end(self, x: float) -> tuple[int, float]:
        k = self.axis
        d_lo, d_hi = x - self.lo[k], self.hi[k] - x
        return (-1, float(d_lo)) if d_lo <= d_hi else (1, float(d_hi))

    def along(self, x: float, exact: bool = False) -> str:
        """'at the wide end' or '40 mm from the narrow end' for a position along the main axis."""
        sign, d = self._nearest_end(x)
        L = self.size[self.axis]
        if d <= max(0.03 * L, 1.0):
            return f"at the {self.end(sign)}"
        return f"{_mm(d) if exact else _approx(d)} mm from the {self.end(sign)}"

    def direction(self, v) -> str:
        """'up', 'to the left', 'up and to the right' in the default front view."""
        v = _unit(v)
        parts = []
        for k in np.argsort(-np.abs(v)):
            if abs(v[k]) < 0.38 or len(parts) == 2:
                break
            w = self.view[k][1 if v[k] > 0 else 0]
            parts.append({"top": "up", "bottom": "down", "left": "to the left", "right": "to the right",
                          "front": "to the front", "back": "to the back"}[w])
        return " and ".join(parts) if parts else "sideways"

    def where(self, p, skip: int | None = None) -> str:
        """'near the top left', '' when central (default front view words)."""
        words = []
        for k in (self.up_index(), 0, 3 - self.up_index()):
            if k == skip or self.size[k] <= 1e-9:
                continue
            t = (p[k] - self.lo[k]) / self.size[k]
            if t < 0.25:
                words.append(self.view[k][0])
            elif t > 0.75:
                words.append(self.view[k][1])
        return f"near the {' '.join(words[:2])}" if words else ""

    def up_index(self) -> int:
        return 2 if self.view[2][1] == "top" else 1

    # ------------------------------------------------------------------ faces
    def _classify(self, f: dict) -> dict:
        k = self.axis
        if f["type"] == "cylinder":
            info = {"kind": "hole" if f["hole"] else "round", "d": 2 * f["radius"]}
            if k is not None:
                if abs(f["axis"][k]) >= COS5:
                    off = self._radial(f["center"][None].copy())[0]
                    info["central"] = bool(off < 0.05 * max(np.delete(self.size, k).max(), 1e-9))
                    info["through"] = bool(f["length"] >= 0.95 * self.size[k])
                    info["opens"] = self._nearest_end(f["center"][k])[0]
                else:
                    info["cross"] = True
            return info
        n, p = f["normal"], f["point"]
        pts = self._face_points(f)
        if k is not None:
            a = n[k]
            if abs(a) >= COS5:                                   # faces one end of the part
                sign = 1 if a > 0 else -1
                end_x = self.hi[k] if sign > 0 else self.lo[k]
                if abs(p[k] - end_x) <= max(1e-3 * self.size[k], 0.01):
                    return {"kind": "end", "end": sign}
                if self._blocked(pts, n, 0.75 * self.diag) < 0.5 and self._enclosed(pts, n) >= 0.85:
                    return {"kind": "floor", "end": sign, "depth": float(abs(end_x - p[k]))}
                near, d = self._nearest_end(p[k])
                return {"kind": "step", "facing": sign, "near": near, "dist": d}
            if abs(a) <= SIN5:                                   # faces sideways
                width = float(np.delete(self.size, k).max())
                if self._blocked(pts, n, 1.05 * width) >= 0.6:
                    open_lo = self._blocked(pts, -np.eye(3)[k], 2 * self.diag) < 0.5
                    open_hi = self._blocked(pts, np.eye(3)[k], 2 * self.diag) < 0.5
                    opens = 0 if open_lo == open_hi else (1 if open_hi else -1)
                    return {"kind": "wall", "opens": opens, "through": bool(open_lo and open_hi), "x": float(p[k])}
                return {"kind": "side", "x": float(p[k])}
            near, d = self._nearest_end(p[k])
            return {"kind": "sloped", "near": near, "dist": d, "x": float(p[k])}
        # no main axis: faces named from the default front view
        s = int(np.argmax(np.abs(n)))
        if abs(n[s]) < COS5:
            return {"kind": "sloped", "dir": self.direction(n)}
        sign = 1 if n[s] > 0 else -1
        ext = self.hi[s] if sign > 0 else self.lo[s]
        word = self.view[s][1 if sign > 0 else 0]
        if abs(p[s] - ext) <= max(1e-3 * self.size[s], 0.01):
            return {"kind": "outer", "word": word}
        if self._blocked(pts, n, 0.75 * self.diag) < 0.5 and self._enclosed(pts, n) >= 0.85:
            return {"kind": "pocket", "word": word, "depth": float(abs(ext - p[s]))}
        if self._blocked(pts, n, 0.75 * self.diag) >= 0.6:
            return {"kind": "inner", "dir": self.direction(n)}
        return {"kind": "facing", "word": word, "dir": self.direction(n), "dist": float(abs(ext - p[s]))}

    def _label(self, i: int) -> str:
        f, info = self.faces[i], self.info[i]
        kind = info["kind"]
        if kind in ("hole", "round"):
            d = f"Ø{info['d']:.2f}"
            if kind == "hole":
                if info.get("cross"):
                    return f"{d} cross hole {self.along(f['center'][self.axis], exact=True)}"
                if "central" in info:
                    if info["central"]:
                        return f"{d} centre hole"
                    return f"{d} hole through the part" if info["through"] else f"{d} hole in the {self.end(info['opens'])}"
                return f"{d} hole"
            return f"{d} round face"
        if kind == "end":
            return f"{self.end(info['end'])} face"
        if kind == "floor":
            return f"floor of the recess in the {self.end(info['end'])}"
        if kind == "step":
            return f"step {_mm(info['dist'])} mm from the {self.end(info['near'])}"
        if kind == "wall":
            if info["through"]:
                return "wall of the opening through the part"
            return f"wall of the recess in the {self.end(info['opens'])}" if info["opens"] else "inner wall"
        if kind == "side":
            return f"flat side {self.along(info['x'])}"
        if kind == "sloped" and self.axis is not None:
            return f"sloped face {self.along(info['x'])}"
        if kind == "sloped":
            return f"sloped face facing {info['dir']}"
        if kind == "outer":
            return f"{info['word']} face"
        if kind == "pocket":
            return f"floor of the pocket in the {info['word']}"
        if kind == "inner":
            return f"inner face facing {info['dir']}"
        return f"face facing {info['dir']}, {_mm(info['dist'])} mm in from the {info['word']}"

    def _labels(self) -> list[str]:
        labels = [self._label(i) for i in range(len(self.faces))]
        groups: dict[str, list[int]] = {}
        for i, s in enumerate(labels):
            groups.setdefault(s, []).append(i)
        for s, idx in groups.items():
            if len(idx) < 2:
                continue
            # the same name for several faces: say which way they face when that differs, else number them
            facing = {i: self._facing_phrase(i) for i in idx}
            if len(set(facing.values())) == len(idx):
                for i in idx:
                    labels[i] = f"{s} ({facing[i]})"
                continue
            for n, i in enumerate(sorted(idx, key=self._order_key), 1):
                labels[i] = f"{s} ({n} of {len(idx)})"
        return labels

    def _facing_phrase(self, i: int) -> str:
        f = self.faces[i]
        n = f["normal"] if f["type"] == "plane" else f["axis"]
        if self.axis is not None and abs(n[self.axis]) >= COS5:
            return f"facing the {self.end(1 if n[self.axis] > 0 else -1)}"
        return f"facing {self.direction(n)}"

    def _order_key(self, i: int):
        f = self.faces[i]
        c = f["point"] if f["type"] == "plane" else f["center"]
        k = self.axis if self.axis is not None else self.up_index()
        d = c - self.mid
        others = [j for j in range(3) if j != k]
        return (round(float(-c[k]), 2), round(float(np.degrees(np.arctan2(d[others[1]], d[others[0]]))) % 360, 1))

    # ------------------------------------------------------------------ areas of the check
    def region(self, face_id: int | None, part: float, center, normal, anchor) -> dict:
        """{'name', 'face_kind', 'inner_of'} for a problem area: mostly on one face (face_id, covering `part` of it),
        or spread over curved surfaces."""
        if face_id is not None:
            f, info = self.faces[face_id], self.info[face_id]
            name = self.labels[face_id]
            if part < 0.6:
                if f["type"] == "cylinder":
                    name = f"part of the {name}"
                elif info["kind"] in ("end", "floor", "step", "outer", "pocket") or (
                        self.axis is None and info["kind"] == "facing"):
                    name += ", " + self._on_face(center, f)
                elif self.axis is not None:
                    name += ", " + self._along_face(center, f)
            return {"name": _cap(name), "face_kind": info["kind"], "inner_of": None}
        n = _unit(normal)
        flat_normal = np.linalg.norm(normal) >= 0.3
        t, tri = self.cast(np.asarray(anchor, dtype=np.float64)[None] + n * self.eps, n[None])
        hit_face = None
        if flat_normal and np.isfinite(t[0]) and t[0] < 0.25 * self.diag:
            hit_face = int(self._tri_face[tri[0]]) if self._tri_face is not None else -1
        if hit_face is not None:
            label = self.labels[hit_face] if hit_face >= 0 else None
            if label and self.info[hit_face]["kind"] in ("floor", "wall"):
                return {"name": _cap(f"inside the recess in the {self.end(self.info[hit_face].get('end') or self.info[hit_face].get('opens') or 1)}"
                                     if self.axis is not None and (self.info[hit_face].get("end") or self.info[hit_face].get("opens"))
                                     else f"inside corner of the {label}"), "face_kind": "inner", "inner_of": hit_face}
            if label:
                return {"name": _cap(f"inner corner at the {label}"), "face_kind": "inner", "inner_of": hit_face}
        if self.axis is not None:
            k = self.axis
            sign, d = self._nearest_end(center[k])
            L = self.size[k]
            near_end = d <= max(0.03 * L, 2.0)
            if not flat_normal:
                what = "curved area"
            elif hit_face is not None:
                what = "inner corner"
            elif near_end and n[k] * sign > 0.2:
                return {"name": _cap(f"edge of the {self.end(sign)}"), "face_kind": "edge", "inner_of": None}
            elif abs(n[k]) < 0.5:
                what = "side surface"
            else:
                what = "sloped surface"
            return {"name": _cap(f"{what} {self.along(center[k])}"), "face_kind": what, "inner_of": None}
        where = self.where(center)
        if not flat_normal:
            what = "curved area"
        elif hit_face is not None:
            what = "inner corner"
        else:
            what = f"surface facing {self.direction(n)}"
        return {"name": _cap(f"{what}, {where}" if where else f"{what}, in the middle"), "face_kind": what,
                "inner_of": None}

    _tri_face: np.ndarray | None = None

    def set_triangle_faces(self, tri_face: np.ndarray) -> None:
        self._tri_face = tri_face

    def _on_face(self, c, f) -> str:
        """Where on an end, step or floor face: 'centre', 'near the rim'."""
        if self.axis is None:
            s = int(np.argmax(np.abs(f["normal"])))
            return self.where(c, skip=s) or "in the middle"
        P = self.ref.vertices[np.unique(self.ref.triangles[f["tris"]])]
        r_face = float(self._radial(P.copy()).max())
        r = float(self._radial(np.asarray(c, dtype=np.float64)[None].copy())[0])
        if r <= 0.35 * r_face:
            return "centre"
        return "near the rim" if r >= 0.75 * r_face else "between the centre and the rim"

    def _along_face(self, c, f) -> str:
        k = self.axis
        P = self.ref.vertices[np.unique(self.ref.triangles[f["tris"]])]
        a, b = P[:, k].min(), P[:, k].max()
        if b - a <= 1e-6:
            return "part of it"
        t = (c[k] - a) / (b - a)
        if 0.3 <= t <= 0.7:
            return "middle part"
        end = 1 if t > 0.5 else -1
        return f"part nearest the {self.end(end)}"

    def is_end_like(self, normal, center) -> int:
        """+1/-1 when an area faces one end of the main axis (the scanner missed that end), else 0."""
        if self.axis is None:
            return 0
        a = normal[self.axis]
        return (1 if a > 0 else -1) if abs(a) >= 0.7 else 0

    # ------------------------------------------------------------------ sizes and measurements
    def size_name(self, k: int) -> str:
        if self.axis == k:
            return "Overall length (end to end)"
        w0, w1 = self.view[k]
        what = "height" if w1 == "top" else "depth" if w1 in ("front", "back") else "width"
        return f"Overall {what} ({w0} to {w1})"

    def end_word(self, k: int, sign: int) -> str:
        """The end (main axis) or side (others) of the part on that side of axis k, for 'was not scanned'."""
        if self.axis == k:
            return self.end(sign)
        return f"{self.view[k][1 if sign > 0 else 0]} side"

    def pair_name(self, kind: str, i: int, j: int) -> str:
        a, b = self.info[i], self.info[j]
        kinds = {a["kind"], b["kind"]}
        if kind == "step" and kinds == {"end", "floor"}:
            fl = a if a["kind"] == "floor" else b
            en = b if a["kind"] == "floor" else a
            if fl["end"] == en["end"]:
                return f"Depth of the recess in the {self.end(fl['end'])}"
        if kind == "step" and kinds == {"outer", "pocket"}:
            pk = a if a["kind"] == "pocket" else b
            return f"Depth of the pocket in the {pk['word']}"
        if kind == "gap" and a["kind"] == b["kind"] == "wall":
            base = self.labels[i].split(" (")[0].replace("wall of the ", "")
            return f"Width across the {base}"
        if kind == "thickness" and "floor" in kinds and len(kinds) == 2:
            fl, other = (i, j) if a["kind"] == "floor" else (j, i)
            return f"Thickness below the recess in the {self.end(self.info[fl]['end'])} (to the {self.labels[other]})"
        if kind == "thickness" and a["kind"] == b["kind"] == "side":
            return f"Width across the flat sides {self.along((a['x'] + b['x']) / 2)}"
        word = {"thickness": "Thickness", "gap": "Gap", "step": "Step"}[kind]
        return f"{word}: {self.labels[i]} to {self.labels[j]}"

    # ------------------------------------------------------------------ "Show me"
    def view_for(self, anchor, normal, size: float, prefer=None) -> dict:
        """Where to put the camera to see an area: along its normal (or `prefer`) when nothing blocks the view,
        else the nearest direction that is clear; far enough back to show the area with some of the part around it."""
        anchor = np.asarray(anchor, dtype=np.float64)
        n = _unit(normal) if np.linalg.norm(normal) > 1e-6 else _unit(anchor - self.mid)
        want = _unit(prefer) if prefer is not None else n
        radius = max(0.75 * size, 0.22 * self.diag, 2.0)
        dist = radius / np.sin(np.radians(21.0))      # the viewer's 45° field of view, with a margin
        cands = _fibonacci(160)
        cands = cands[cands @ n > 0.1]
        cands = cands[np.argsort(-(cands @ want))]
        dirs = np.vstack([want[None], cands])
        origin = anchor + n * self.eps
        t, _ = self.cast(np.repeat(origin[None], len(dirs), axis=0), dirs)
        clear = np.flatnonzero(~(t < dist))
        look = dirs[clear[0]] if len(clear) else want
        return {"target": np.round(anchor, 4).tolist(), "from": np.round(anchor + look * dist, 4).tolist(),
                "radius": round(float(radius), 3)}
