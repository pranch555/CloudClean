"""Plain words for the parts of a golden model (docs/golden-model.md, "Names").

The golden check names its areas and sizes the way a person would point at them on the part: "Bottom of the hex
socket", "Corner under the head", "Rim around the tip of the shaft", "Head height", "Length under the head",
"Ø6.00 hole through the middle". CAD files use any axis as 'up' and the 3D view can be turned any way, so the words
lean on landmarks of the part itself, recognised conservatively from the golden mesh. When a landmark is not sure,
the plainest true description is used instead.

* The part's kind. Turned and long parts have a main axis: the axis of their round faces, else their longest side
  when it is clearly the longest.
  - Head and shaft (screws, bolts, pins, rivets): a section at one end of the axis, at most 45 % of the length,
    steps down abruptly to a section at least 15 % narrower, and no bore runs along the axis (that would be a tube or
    a bushing). Its faces are the top of the head, the underside of the head and the tip of the shaft.
  - Other parts with a main axis name their ends "the wide end" and "the narrow end" when their cross-sections
    differ, else from the 3D view's default front view ("top end", "left end").
  - Parts with no main axis are named from that front view (up = Y or Z, as set in Settings -> 3D view).
* Features. A recess in the middle of an end whose walls are six flats 60° apart, all at one distance from a common
  centre, is a hex socket (four at 90°: a square socket); a round blind hole is named by its size. Flat sides all
  round a section at one distance from the axis are one feature: six are a hex, four a square, eight or more the flat
  sides of a polygon or, when they cover only part of the way round, the grooves or ridges of a knurled grip.
  Threads are found from the part's outline (outline_check.turned_profile: one helix of one pitch).
* Ray casting on the golden surface tells outer faces from recesses (a floor with walls all round), steps and inner
  corners, and finds a direction to look at an area from that nothing else blocks ("Show me").
"""
from __future__ import annotations

import numpy as np
import open3d.core as o3c
from scipy.spatial import cKDTree

COS5 = float(np.cos(np.radians(5.0)))
SIN5 = float(np.sin(np.radians(5.0)))
HEAD_MAX_SHARE = 0.45      # a head is at most this share of the part's length ...
HEAD_RATIO = 1.15          # ... and at least this much wider than the rest (the shaft)
KNURL_MIN = 20             # grooves (ridges) all round: this many or more are a knurl
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")


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


def _approx(x: float) -> str:
    """Roughly where an area is: no false precision."""
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _base(label: str) -> str:
    """A label without its '(2 of 6)' or '(facing up)'."""
    return label.split(" (")[0]


def _plural(label: str) -> str:
    """'groove in the knurled grip' -> 'grooves in the knurled grip' (the noun before 'of / in / on the')."""
    words = label.split(" ")
    at = next((n for n, w in enumerate(words) if w in ("of", "in", "on", "at") and n > 0), len(words)) - 1
    w = words[at]
    words[at] = w + ("es" if w.endswith(("s", "x", "ch", "sh")) else "s")
    return " ".join(words)


def _ordinal(n: int) -> str:
    return ORDINALS[n - 1] if n <= len(ORDINALS) else f"{n}th"


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
    """Names for the faces, areas and sizes of one golden model (ref: compare.ReferenceSurface, faces: find_faces).

    kind: 'head_shaft' | 'axial' | 'block'; labels: a lower-case name per face ('bottom of the hex socket');
    region(): the name and 'where' sentence of a problem area; size_words() / pair_words() / diameter_words() /
    position_words(): the name, 'what' sentence, comparative words and group of a measurement."""

    _tri_face: np.ndarray | None = None

    def __init__(self, ref, faces: list[dict], up_axis: str = "y"):
        self.ref, self.faces = ref, faces
        V = ref.vertices
        self.lo, self.hi = V.min(0), V.max(0)
        self.size = self.hi - self.lo
        self.mid = (self.lo + self.hi) / 2
        self.diag = float(np.linalg.norm(self.size))
        self.eps = max(1e-3 * self.diag, 0.02)
        self.up_axis = up_axis
        self.view = view_words(up_axis)
        self.samples, self.sample_normals = ref.sample(20_000, seed=3)
        self.axis = self._main_axis()
        self.info = [self._classify(f) for f in faces]
        self.profile = self._profile()
        self.texture = self._texture()
        self.head = self._head()
        self.kind = "head_shaft" if self.head else "axial" if self.axis is not None else "block"
        self.ends = self._end_names()
        self.recesses = self._recesses()
        self.grips = self._grips()
        self._pockets()
        self._roles()
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

    # ------------------------------------------------------------------ the part's frame and kind
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

    def _others(self) -> list[int]:
        return [j for j in range(3) if j != self.axis]

    def _profile(self) -> dict | None:
        """The part's outer radius along the main axis: rays cast from outside towards the axis every 10° round it
        (a round part's outline; a hex or a knurl gives its outermost corners)."""
        k = self.axis
        if k is None:
            return None
        L = float(self.size[k])
        n = int(np.clip(L / max(2e-3 * self.diag, 0.02), 60, 600))
        xs = self.lo[k] + (np.arange(n) + 0.5) * L / n
        o2 = self._others()
        reach = float(self._radial(self.ref.vertices.copy()).max()) + 1.0 + self.eps
        ang = np.radians(np.arange(0.0, 360.0, 10.0))
        D = np.zeros((len(ang), 3))
        D[:, o2[0]], D[:, o2[1]] = np.cos(ang), np.sin(ang)
        O = np.repeat(self.mid[None], n * len(ang), axis=0)
        O[:, k] = np.repeat(xs, len(ang))
        Dt = np.tile(D, (n, 1))
        t, _ = self.cast(O + reach * Dt, -Dt)
        r = np.where(np.isfinite(t), reach - t, 0.0).reshape(n, len(ang))
        return {"x": xs, "r": r.max(1), "step": L / n}

    def radius_at(self, x0: float, x1: float) -> float:
        """The part's outer radius between two positions along the main axis (the profile's median there)."""
        prof = self.profile
        sel = (prof["x"] >= min(x0, x1)) & (prof["x"] <= max(x0, x1))
        if not sel.any():
            return float(np.interp((x0 + x1) / 2, prof["x"], prof["r"]))
        return float(np.median(prof["r"][sel]))

    def _texture(self) -> dict:
        """Is the part round about its main axis, and where are its threads and knurls (outline_check.turned_profile,
        as ranges along the main axis)?"""
        out = {"round": False, "zones": []}
        if self.axis is None:
            return out
        try:
            # imported here: outline_check imports golden, which imports this module
            from .outline_check import turned_profile

            prof = turned_profile(self.ref, self.ref.scene, log=lambda m: None)
        except Exception:        # noqa: BLE001 - only words depend on it
            return out
        if prof is None:
            return out
        a = np.asarray(prof["a"], dtype=np.float64)
        k = self.axis
        if abs(a[k]) < COS5:
            return out
        s = 1.0 if a[k] > 0 else -1.0                         # along the axis t = p . a, so x = s * t
        out["round"] = True
        for z in prof["zones"]:
            x0, x1 = sorted((s * z["t0"], s * z["t1"]))
            out["zones"].append({"x0": x0, "x1": x1, "r": z["R"], "texture": z["texture"], "pitch": z.get("pitch")})
        return out

    @property
    def threads(self) -> list[dict]:
        return [z for z in self.texture["zones"] if z["texture"] == "thread"]

    def _threaded_at(self, x: float) -> bool:
        return any(z["x0"] - self.eps <= x <= z["x1"] + self.eps for z in self.threads)

    def _head(self) -> dict | None:
        """A head and a shaft (screw, bolt, pin, rivet): a short wide section at one end of the main axis that steps
        down abruptly to a clearly narrower, longer shaft, and no bore along the axis (a tube, a bushing)."""
        prof = self.profile
        if prof is None:
            return None
        k = self.axis
        L = float(self.size[k])
        o = self.mid.copy()
        o[k] = self.hi[k] + 1.0
        t, _ = self.cast(o[None], -np.eye(3)[k][None])
        if not np.isfinite(t[0]) or t[0] > L + 1.0 + self.eps:      # a ray down the middle goes straight through
            return None
        r, step = prof["r"], prof["step"]
        found = []
        for sign in (1, -1):
            seq = r[::-1] if sign > 0 else r                         # from that end inwards
            prefix = np.maximum.accumulate(seq)
            suffix = np.maximum.accumulate(seq[::-1])[::-1]          # the widest point from here on
            for i in range(2, len(seq)):
                if i * step > HEAD_MAX_SHARE * L:
                    break
                if suffix[i] > 0 and prefix[i - 1] >= HEAD_RATIO * suffix[i] and seq[i - 1] >= 1.1 * suffix[i]:
                    found.append((sign, i, float(prefix[i - 1]), float(suffix[i])))
                    break
        if len(found) != 1:
            return None
        sign, i, r_head, r_shaft = found[0]
        end_x = float(self.hi[k] if sign > 0 else self.lo[k])
        x = end_x - sign * i * step
        # the underside: a flat face at that step, facing the tip
        under = [j for j, inf in enumerate(self.info) if inf["kind"] == "step" and inf["facing"] == -sign
                 and abs(self.faces[j]["point"][k] - x) <= 1.5 * step + self.eps]
        underside = max(under, key=lambda j: self.faces[j]["area"]) if under else None
        if underside is not None:
            x = float(self.faces[underside]["point"][k])
        return {"sign": sign, "x": x, "end_x": end_x, "len": abs(end_x - x), "r": r_head, "shaft_r": r_shaft,
                "underside": underside}

    def section_of(self, x: float) -> str:
        """'head' or 'shaft' for a position along the axis of a head-and-shaft part."""
        hs = self.head
        return "head" if (x - hs["x"]) * hs["sign"] > -1e-9 else "shaft"

    def _end_names(self) -> tuple[str, str] | None:
        k = self.axis
        if k is None:
            return None
        if self.head is not None:
            return ("tip of the shaft", "head") if self.head["sign"] > 0 else ("head", "tip of the shaft")
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
        """The main axis' end on the + (sign 1) or - side: 'head', 'tip of the shaft', 'wide end', 'top end'."""
        assert self.ends is not None
        return self.ends[1 if sign > 0 else 0]

    def end_face(self, sign: int) -> str:
        """The flat face at an end: 'top of the head', 'tip of the shaft', 'flat face at the wide end'."""
        if self.head is not None:
            return "top of the head" if sign == self.head["sign"] else "tip of the shaft"
        return f"flat face at the {self.end(sign)}"

    def _reference_end(self) -> int:
        """The end sizes and steps are counted from: the head, the wide end, else the + end."""
        if self.head is not None:
            return self.head["sign"]
        if self.ends and self.ends[0] == "wide end":
            return -1
        return 1

    def _nearest_end(self, x: float) -> tuple[int, float]:
        k = self.axis
        d_lo, d_hi = x - self.lo[k], self.hi[k] - x
        return (-1, float(d_lo)) if d_lo <= d_hi else (1, float(d_hi))

    def along_words(self, x: float) -> str:
        """Roughly where along the main axis, without numbers: 'near the top' or 'near the underside' (of a head),
        'near the head', 'near the tip', 'near the wide end', 'in the middle'."""
        k = self.axis
        hs = self.head
        if hs is not None:
            if self.section_of(x) == "head":
                t = abs(hs["end_x"] - x) / max(hs["len"], 1e-9)
                return "near the top" if t < 0.3 else "near the underside" if t > 0.7 else "in the middle"
            t = abs(x - hs["x"]) / max(self.size[k] - hs["len"], 1e-9)
            return "near the head" if t < 0.25 else "near the tip" if t > 0.75 else "in the middle"
        t = (x - self.lo[k]) / max(self.size[k], 1e-9)
        if t < 0.25:
            return f"near the {self.end(-1)}"
        if t > 0.75:
            return f"near the {self.end(1)}"
        return "in the middle"

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
                    if f["hole"] and not info["through"]:      # through, past a wider hole (a counterbore)?
                        o2 = [j for j in range(3) if j != k]       # rays along the inside of its wall, end to end
                        O = np.repeat(f["center"][None], 4, axis=0)
                        O[:, k] = self.hi[k] + 1.0
                        ang = np.radians([0.0, 90.0, 180.0, 270.0])
                        O[:, o2[0]] += 0.9 * f["radius"] * np.cos(ang)
                        O[:, o2[1]] += 0.9 * f["radius"] * np.sin(ang)
                        t, _ = self.cast(O, np.tile(-np.eye(3)[k], (4, 1)))
                        info["through"] = bool(np.all(~np.isfinite(t) | (t > self.size[k] + 1.0 + self.eps)))
                    info["opens"] = self._nearest_end(f["center"][k])[0]
                    info["x0"] = float(f["center"][k] - f["length"] / 2)
                    info["x1"] = float(f["center"][k] + f["length"] / 2)
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

    def _polygon(self, ids: list[int], inward: bool) -> dict | None:
        """Six (or four) flat faces round the main axis, 60° (90°) apart and all at one distance from a common
        centre: a hex (square). inward: the walls of a socket, facing its centre; else flats facing out."""
        o2 = self._others()
        ids = [i for i in ids if abs(self.faces[i]["normal"][self.axis]) <= SIN5]
        if len(ids) < 4:
            return None
        N = np.array([_unit(self.faces[i]["normal"][o2]) for i in ids])
        P = np.array([self.faces[i]["point"][o2] for i in ids])
        th = np.degrees(np.arctan2(N[:, 1], N[:, 0]))
        sgn = 1.0 if inward else -1.0
        for m, shape in ((6, "hex"), (4, "square")):
            step = 360.0 / m
            for t0 in th:
                d = np.mod(th - t0, step)
                member = np.flatnonzero((d <= 2.0) | (d >= step - 2.0))
                dirs = np.round(np.mod(th[member] - t0, 360.0) / step).astype(int) % m
                if len(set(dirs.tolist())) != m:
                    continue
                # inward: n.(c - p) = a; outward: n.(p - c) = a
                A = np.c_[sgn * N[member], -np.ones(len(member))]
                b = sgn * np.einsum("ij,ij->i", N[member], P[member])
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                a = float(sol[2])
                if a > 0 and np.abs(A @ sol - b).max() <= max(0.01, 0.003 * a):
                    centre = self.mid.copy()
                    centre[o2] = sol[:2]
                    return {"shape": shape, "faces": [ids[j] for j in member], "apothem": a, "centre": centre}
        return None

    def _recesses(self) -> dict[int, dict]:
        """What is sunk into the middle of each end of the main axis: a hex or square socket, a round hole or a
        recess, with its bottom."""
        k = self.axis
        out: dict[int, dict] = {}
        if k is None:
            return out
        width = float(np.delete(self.size, k).max())
        hs = self.head
        for s in (1, -1):
            walls = [i for i, inf in enumerate(self.info) if inf["kind"] == "wall" and inf.get("opens") == s
                     and not inf.get("through")]
            holes = [i for i, inf in enumerate(self.info) if inf["kind"] == "hole" and inf.get("central")
                     and not inf.get("through") and inf.get("opens") == s]
            floors = [i for i, inf in enumerate(self.info) if inf["kind"] == "floor" and inf["end"] == s]
            poly = self._polygon(walls, inward=True)
            hole = max(holes, key=lambda i: self.faces[i]["radius"]) if holes else None
            if poly is not None:
                reach = poly["apothem"] / np.cos(np.radians(180.0 / (6 if poly["shape"] == "hex" else 4)))
                centre = poly["centre"]
            elif hole is not None:
                reach, centre = self.faces[hole]["radius"], self.faces[hole]["center"]
            else:
                reach, centre = 0.1 * width, self.mid
            mine = []
            for i in floors:
                P = self.ref.vertices[np.unique(self.ref.triangles[self.faces[i]["tris"]])]
                R = P - centre
                R[:, k] = 0
                if poly is None and hole is None:
                    c = R.mean(0)
                    if np.linalg.norm(c) <= reach:
                        mine.append(i)
                elif np.linalg.norm(R, axis=1).max() <= reach * 1.02 + self.eps:
                    mine.append(i)
            if poly is None and hole is None and not mine:
                continue
            shape = poly["shape"] if poly else ("round" if hole is not None else None)
            place = ("head" if s == hs["sign"] else "tip of the shaft") if hs else self.end(s)
            if shape in ("hex", "square"):
                noun = f"{shape} socket"
                name = noun if hs and s == hs["sign"] else f"{noun} in the {place}"
                explain = (" (the six-sided hole a hex key fits in)" if shape == "hex" else
                           " (the four-sided hole a square drive fits in)")
                short = "socket bottom"
            elif shape == "round":
                d = f"Ø{2 * self.faces[hole]['radius']:.2f}"
                noun = name = f"{d} hole in the {place}"
                explain, short = "", "hole bottom"
            else:
                noun, name, explain, short = "recess", f"recess in the {place}", "", "recess bottom"
            group = _cap(noun) if hs and s == hs["sign"] and shape in ("hex", "square") else _cap(name)
            out[s] = {"end": s, "shape": shape, "walls": poly["faces"] if poly else walls, "floors": mine,
                      "hole": hole, "across": 2 * poly["apothem"] if poly else None, "noun": noun, "name": name,
                      "place": place, "inside": f"Inside the {place}", "explain": explain, "short": short,
                      "group": group}
        return out

    def _grips(self) -> list[dict]:
        """Flat sides all round the main axis at one distance from it: a hex, a square, the flat sides of a polygon,
        or the grooves (ridges) of a knurled grip when they only cover part of the way round."""
        k = self.axis
        if k is None:
            return []
        o2 = self._others()
        rows = []
        for i, inf in enumerate(self.info):
            if inf["kind"] != "side":
                continue
            f = self.faces[i]
            n2 = _unit(f["normal"][o2])
            a = float((f["point"][o2] - self.mid[o2]) @ n2)          # the plane's distance from the axis
            if a <= 0:
                continue
            P = self.ref.vertices[np.unique(self.ref.triangles[f["tris"]])]
            tang = np.array([-n2[1], n2[0]])
            rows.append((i, a, float(P[:, k].min()), float(P[:, k].max()),
                         float(np.degrees(np.arctan2(n2[1], n2[0]))), float(np.ptp(P[:, o2] @ tang))))
        groups: list[list[tuple]] = []
        for row in sorted(rows, key=lambda r: r[1]):
            for g in groups:
                g0 = g[0]
                overlap = min(row[3], g0[3]) - max(row[2], g0[2])
                if abs(row[1] - g0[1]) <= max(0.01, 0.003 * g0[1]) and \
                        overlap >= 0.5 * min(row[3] - row[2], g0[3] - g0[2]):
                    g.append(row)
                    break
            else:
                groups.append([row])
        grips = []
        for g in groups:
            m = len(g)
            if m < 4:
                continue
            th = np.sort(np.mod([r[4] for r in g], 360.0))
            gaps = np.diff(np.r_[th, th[0] + 360.0])
            base = float(np.median(gaps))
            if base < 1.0 or np.abs(gaps / base - np.round(gaps / base)).max() > 0.15:
                continue
            count = round(360.0 / base)
            if abs(360.0 / count - base) > 0.15 * base or m < 0.75 * count:
                continue
            a = float(np.median([r[1] for r in g]))
            w = float(np.median([r[5] for r in g]))
            x0, x1 = min(r[2] for r in g), max(r[3] for r in g)
            relief = None
            if count in (4, 6) and m == count:
                shape = "hex" if count == 6 else "square"
            elif count >= 8:
                R = self.radius_at(x0, x1)
                chord = 2 * np.sqrt(max(R * R - a * a, 0.0))
                if count * w / (2 * np.pi * a) >= 0.85:
                    shape = "polygon"
                elif a >= R - max(0.02, 0.005 * R):
                    relief = "ridges"
                elif w < 0.7 * chord:
                    relief = "grooves"
                else:
                    shape = "flats"
                if relief:                    # many fine ones are a knurl; a few (splines, ribs) are just named
                    shape = "knurl" if count >= KNURL_MIN else "ribs"
            else:
                continue
            on = self.section_of((x0 + x1) / 2) if self.head else None
            grips.append({"shape": shape, "relief": relief, "faces": [r[0] for r in g], "count": count,
                          "apothem": a, "x0": x0, "x1": x1, "on": on})
        return grips

    def _pockets(self) -> None:
        """Parts with no main axis: the inner faces that are walls of a pocket."""
        if self.axis is not None:
            return
        for p, inf in enumerate(self.info):
            if inf["kind"] != "pocket":
                continue
            f = self.faces[p]
            n = f["normal"]
            u, v = _perp_basis(n)
            P = self.ref.vertices[np.unique(self.ref.triangles[f["tris"]])]
            uv = np.c_[(P - f["point"]) @ u, (P - f["point"]) @ v]
            lo, hi = uv.min(0) - self.eps, uv.max(0) + self.eps
            for i, w in enumerate(self.info):
                g = self.faces[i]
                if w["kind"] != "inner" or g["type"] != "plane" or abs(g["normal"] @ n) > SIN5:
                    continue
                Q = self.ref.vertices[np.unique(self.ref.triangles[g["tris"]])]
                h = (Q - f["point"]) @ n
                q = np.c_[(Q - f["point"]) @ u, (Q - f["point"]) @ v]
                if h.min() >= -self.eps and h.max() <= inf["depth"] + self.eps and \
                        np.all(q >= lo) and np.all(q <= hi):
                    w["pocket_of"] = p

    def _roles(self) -> None:
        hs = self.head
        for s, rec in self.recesses.items():
            for i in rec["walls"] + rec["floors"] + ([rec["hole"]] if rec["hole"] is not None else []):
                self.info[i]["recess"] = s
        for g, grip in enumerate(self.grips):
            for i in grip["faces"]:
                self.info[i]["grip"] = g
        for i, inf in enumerate(self.info):
            role = inf["kind"]
            if hs is not None and role == "end":
                role = "head_top" if inf["end"] == hs["sign"] else "tip"
            elif hs is not None and i == hs["underside"]:
                role = "underside"
            inf["role"] = role

    def _cyl_label(self, i: int) -> str:
        f, inf = self.faces[i], self.info[i]
        d = f"Ø{inf['d']:.2f}"
        hs = self.head
        k = self.axis
        if inf["kind"] == "hole":
            if inf.get("cross"):
                if hs is not None:
                    return f"{d} cross hole through the {self.section_of(f['center'][k])}"
                return f"{d} cross hole"
            if "central" in inf:
                if inf["through"]:
                    return f"{d} hole through the middle" if inf["central"] else f"{d} hole through the part"
                return f"{d} hole in the {('head' if inf['opens'] == hs['sign'] else 'tip of the shaft') if hs else self.end(inf['opens'])}"
            return self._loose_hole(i)
        if inf.get("central"):
            if hs is not None:
                if self.section_of(f["center"][k]) == "head":
                    return "side of the head"
                shafts = [j for j, g in enumerate(self.info) if g["kind"] == "round" and g.get("central")
                          and self.section_of(self.faces[j]["center"][k]) == "shaft"]
                if len(shafts) > 1:
                    return f"{d} part of the shaft"
                return "smooth part of the shaft" if self.threads and not self._threaded_at(f["center"][k]) \
                    else "side of the shaft"
            L = self.size[k]
            near = max(0.05 * L, 1.0)
            at_lo, at_hi = inf["x0"] - self.lo[k] <= near, self.hi[k] - inf["x1"] <= near
            if at_lo and at_hi:
                return "round outside"
            if at_lo or at_hi:
                return f"round side at the {self.end(1 if at_hi else -1)}"
            return f"{d} round side"
        return f"{d} round face"

    def _loose_hole(self, i: int) -> str:
        """A hole of a part with no main axis (or tilted): where it opens, from the default front view."""
        f = self.faces[i]
        d = f"Ø{2 * f['radius']:.2f}"
        a = _unit(f["axis"])
        s = int(np.argmax(np.abs(a)))
        if abs(a[s]) < COS5:
            return f"{d} hole"
        e = np.sign(a[s]) * a                    # pointing to the + side of axis s
        half = f["length"] / 2
        t, _ = self.cast(np.repeat(f["center"][None], 2, axis=0), np.vstack([e, -e]))
        open_plus, open_minus = (not np.isfinite(t[0]) or t[0] > half + 2 * self.eps), \
            (not np.isfinite(t[1]) or t[1] > half + 2 * self.eps)
        if open_plus and open_minus:
            return f"{d} hole through the part"
        if open_plus or open_minus:
            return f"{d} hole in the {self.view[s][1 if open_plus else 0]}"
        return f"{d} hole"

    def grip_noun(self, g: dict) -> str:
        """'knurled grip on the head', 'hex head', 'flat sides of the head'."""
        on = g["on"]
        if g["shape"] == "knurl":
            return "knurled grip" + (f" on the {on}" if on else "")
        if g["shape"] == "ribs":
            return f"{g['relief']} round the {on or 'part'}"
        if g["shape"] in ("hex", "square"):
            return f"{g['shape']} head" if on == "head" else f"{g['shape']} on the shaft" if on == "shaft" \
                else g["shape"]
        return f"flat sides of the {on}" if on else "flat sides"

    def _grip_member(self, g: dict) -> str:
        if g["shape"] == "knurl":
            return "groove in the knurled grip" if g["relief"] == "grooves" else "ridge of the knurled grip"
        if g["shape"] == "ribs":
            return f"{g['relief'][:-1]} round the {g['on'] or 'part'}"
        if g["shape"] in ("hex", "square"):
            return f"flat of the {self.grip_noun(g)}"
        return f"flat side of the {g['on']}" if g["on"] else "flat side"

    def _label(self, i: int) -> str:
        f, inf = self.faces[i], self.info[i]
        kind, role = inf["kind"], inf.get("role")
        hs = self.head
        k = self.axis
        if "grip" in inf:
            return self._grip_member(self.grips[inf["grip"]])
        if "recess" in inf and kind in ("floor", "wall"):
            rec = self.recesses[inf["recess"]]
            if kind == "floor":
                return f"bottom of the {rec['name']}"
            return f"flat side of the {rec['name']}" if rec["shape"] in ("hex", "square") else f"wall of the {rec['name']}"
        if kind in ("hole", "round"):
            return self._cyl_label(i)
        if kind == "end":
            return self.end_face(inf["end"])
        if kind == "floor":
            hole = self._hole_over(i)
            return f"bottom of the {self._cyl_label(hole)}" if hole is not None else \
                f"bottom of a recess in the {self.end(inf['end'])}"
        if kind == "step":
            if role == "underside":
                return "underside of the head"
            return f"step on the {self.section_of(f['point'][k])}" if hs else "step"
        if kind == "wall":
            if inf["through"]:
                return "wall of the opening through the part"
            if inf["opens"]:
                return f"wall of a recess in the {self.end(inf['opens'])}"
            return "inside wall"
        if kind == "side":
            if hs:
                return "flat on the side of the head" if self.section_of(inf["x"]) == "head" else "flat on the shaft"
            return f"flat side {self.along_words(inf['x'])}"
        if kind == "sloped" and k is not None:
            if hs:
                return f"sloped face on the {self.section_of(inf['x'])}"
            return f"sloped face {self.along_words(inf['x'])}"
        if kind == "sloped":
            return f"sloped face facing {inf['dir']}"
        if kind == "outer":
            return f"{inf['word']} face"
        if kind == "pocket":
            return f"bottom of the pocket in the {inf['word']}"
        if kind == "inner":
            if inf.get("pocket_of") is not None:
                return f"{self._pocket_side(i)} wall of the pocket in the {self.info[inf['pocket_of']]['word']}"
            return f"inside face facing {inf['dir']}"
        return f"step facing {inf['dir']}"

    def _pocket_side(self, i: int) -> str:
        """Which wall of its pocket a face is ('front': it faces the back, into the pocket)."""
        n = self.faces[i]["normal"]
        s_ = int(np.argmax(np.abs(n)))
        return self.view[s_][0 if n[s_] > 0 else 1]

    def _hole_over(self, i: int) -> int | None:
        """The hole whose bottom this floor is (a blind hole off the axis)."""
        f = self.faces[i]
        P = self.ref.vertices[np.unique(self.ref.triangles[f["tris"]])]
        for j, g in enumerate(self.faces):
            if g["type"] != "cylinder" or not g["hole"] or abs(g["axis"] @ f["normal"]) < COS5:
                continue
            R = P - g["center"]
            R -= np.outer(R @ g["axis"], g["axis"])
            if np.linalg.norm(R, axis=1).max() <= g["radius"] * 1.02 + self.eps:
                return j
        return None

    def _labels(self) -> list[str]:
        labels = [self._label(i) for i in range(len(self.faces))]
        groups: dict[str, list[int]] = {}
        for i, s in enumerate(labels):
            groups.setdefault(s, []).append(i)
        k = self.axis
        for s, idx in groups.items():
            if len(idx) < 2:
                continue
            # steps across the axis: counted from the head (the wide end): 'step nearest the head', 'second step...'
            if k is not None and all(self.faces[i]["type"] == "plane" and abs(self.faces[i]["normal"][k]) >= COS5
                                     for i in idx):
                ref = self._reference_end()
                end_x = self.hi[k] if ref > 0 else self.lo[k]
                order = sorted(idx, key=lambda i: abs(self.faces[i]["point"][k] - end_x))
                positions = [round(float(self.faces[i]["point"][k]), 3) for i in order]
                if len(set(positions)) == len(positions):
                    for n, i in enumerate(order, 1):
                        labels[i] = f"{s} nearest the {self.end(ref)}" if n == 1 else \
                            f"{_ordinal(n)} {s} from the {self.end(ref)}"
                    continue
            # else: say which way they face when that differs, else number them
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

    def short(self, i: int) -> str:
        """A short name for 'A to B' sizes: 'top of head', 'tip', 'socket bottom', 'wide end'."""
        inf = self.info[i]
        role = inf.get("role")
        if role == "head_top":
            return "top of head"
        if role == "tip":
            return "tip"
        if role == "underside":
            return "underside of head"
        if inf["kind"] == "floor" and "recess" in inf:
            return self.recesses[inf["recess"]]["short"]
        if inf["kind"] == "end":
            return self.end(inf["end"])
        if inf["kind"] == "pocket":
            return "pocket bottom"
        if inf.get("pocket_of") is not None:
            return f"pocket's {self._pocket_side(i)} wall"
        return self.labels[i]

    def recess_of(self, i: int | None) -> dict | None:
        if i is None or i < 0 or "recess" not in self.info[i]:
            return None
        return self.recesses[self.info[i]["recess"]]

    def _part_of(self, i: int) -> str | None:
        """'head' or 'shaft' for a face of a head-and-shaft part."""
        if self.head is None:
            return None
        role = self.info[i].get("role")
        if role in ("head_top", "underside"):
            return "head"
        if role == "tip":
            return "shaft"
        f = self.faces[i]
        return self.section_of((f["point"] if f["type"] == "plane" else f["center"])[self.axis])

    # ------------------------------------------------------------------ "where" sentences
    def _thing(self, i: int) -> tuple[str, str]:
        """(context, description) of a face: ('Inside the head', 'the flat bottom of the hex socket')."""
        f, inf = self.faces[i], self.info[i]
        kind, role = inf["kind"], inf.get("role")
        k = self.axis
        hs = self.head
        if "grip" in inf:
            g = self.grips[inf["grip"]]
            what = {"grooves": "narrow flat-bottomed grooves", "ridges": "narrow flat-topped ridges"}.get(
                g["relief"], "flat sides")
            if g["shape"] == "ribs":
                return "", f"the {g['count']} flat-{'bottomed' if g['relief'] == 'grooves' else 'topped'} {g['relief']} " \
                           f"all round the {g['on'] or 'part'}"
            return "", f"the {self.grip_noun(g)}: {g['count']} {what} all round it"
        rec = self.recess_of(i)
        if rec is not None:
            if kind == "floor":
                return rec["inside"], f"the flat bottom of the {rec['name']}{rec['explain']}"
            if kind == "wall":
                return rec["inside"], (f"one of the {len(rec['walls'])} flat sides of the {rec['name']}"
                                       if rec["shape"] in ("hex", "square") else f"a wall of the {rec['name']}")
            return rec["inside"], f"the round wall of the {rec['name']}"
        label = _base(self.labels[i])
        if role == "head_top":
            head_rec = self.recesses.get(hs["sign"])
            return "", "the flat top of the head" + (f", around the {head_rec['name']}" if head_rec else "")
        if role == "tip":
            return "", "the flat end of the shaft, farthest from the head"
        if role == "underside":
            return "", "the flat ring under the head, around the shaft"
        if kind == "end":
            return "", f"the flat face at the {self.end(inf['end'])} of the part"
        if kind == "step":
            change = self._step_change(i)
            on = f"the {self.section_of(f['point'][k])}" if hs else "the part"
            return "", (f"the flat ring where {on} steps down from {change}" if change else
                        f"a flat step on {on}, facing the {self.end(inf['facing'])}")
        if kind == "floor":
            return f"Inside the {self.end(inf['end'])}", f"the {label}"
        if kind == "hole":
            return "", f"the inside of the {label}"
        if kind == "round":
            if label == "side of the head":
                return "", "the round outside of the head"
            if label == "side of the shaft":
                return "", "the round side of the shaft"
            if label == "smooth part of the shaft":
                return "", "the smooth (not threaded) part of the shaft"
            if label == "round outside":
                return "", "the round outside of the part"
            if label.startswith("round side at the "):
                return "", f"the round outside of the part at the {label[len('round side at the '):]}"
            return "", f"the {label}"
        if kind == "wall":
            return "", f"an inside wall: the {label}"
        if kind == "side":
            if hs:
                return "", f"a flat face on the side of the {self.section_of(inf['x'])}"
            return "", f"a flat face on the side of the part, {self.along_words(inf['x'])}"
        if kind == "sloped":
            if hs:
                return "", f"a sloped flat face on the {self.section_of(inf['x'])}"
            if k is not None:
                return "", f"a sloped flat face {self.along_words(inf['x'])}"
            return "", f"a sloped face facing {inf['dir']}"
        if kind == "outer":
            return "", f"the {inf['word']} face of the part, as the 3D view first shows it"
        if kind == "pocket":
            return "", f"the bottom of the pocket in the {inf['word']} face"
        if kind == "inner":
            if inf.get("pocket_of") is not None:
                return "", f"the {self._pocket_side(i)} wall of the pocket in the {self.info[inf['pocket_of']]['word']} face"
            return "", f"an inside face, facing {inf['dir']}"
        return "", f"a flat step facing {inf['dir']}, {_approx(inf['dist'])} mm in from the {inf['word']} face"

    def _step_change(self, i: int) -> str:
        """'Ø30.0 to Ø20.0' across a step of a round part ('' when the part is not round there)."""
        if not self.texture["round"] or self.profile is None:
            return ""
        k = self.axis
        x = self.faces[i]["point"][k]
        d = 2 * self.profile["step"] + self.eps
        a, b = 2 * self.radius_at(x - 3 * d, x - d), 2 * self.radius_at(x + d, x + 3 * d)
        if abs(a - b) < 0.02 * max(a, b):
            return ""
        return f"Ø{max(a, b):.1f} to Ø{min(a, b):.1f}"

    def _sentence(self, context: str, desc: str) -> str:
        return f"{context}: {desc}." if context else f"{_cap(desc)}."

    # ------------------------------------------------------------------ areas of the check
    def region(self, face_id: int | None, part: float, center, normal, anchor, spots=None) -> dict:
        """{'name', 'where', 'face_kind', 'inner_of', 'shade', 'recess'} for a problem area: mostly on one face
        (face_id, covering `part` of it), or spread over curved surfaces, edges or corners. spots: the area's spot
        (points, normals, face ids), for corners."""
        if spots is not None and self.head is not None and self._under_head(spots[0]):
            under = self.head["underside"]
            return {**self._corner_region(under), "inner_of": under} if under is not None else \
                {"name": "Corner under the head",
                 "where": "Where the shaft meets the head: a tight inside corner that is hard for a scanner to see into.",
                 "face_kind": "inner", "inner_of": None, "shade": "the head", "recess": None}
        if face_id is not None:
            return self._face_region(face_id, part, center)
        if spots is not None:
            corner = self._two_face_corner(*spots)
            if corner is not None:
                return corner
        n = _unit(normal)
        flat_normal = np.linalg.norm(normal) >= 0.3
        t, tri = self.cast(np.asarray(anchor, dtype=np.float64)[None] + n * self.eps, n[None])
        hit = None
        if flat_normal and np.isfinite(t[0]) and t[0] < 0.25 * self.diag:
            hit = int(self._tri_face[tri[0]]) if self._tri_face is not None else -1
        if hit is not None and hit >= 0:
            return self._corner_region(hit)
        k = self.axis
        hs = self.head
        if k is not None:
            x = float(center[k])
            sign, d = self._nearest_end(x)
            near_end = d <= max(0.03 * self.size[k], 2.0)
            along = self.along_words(x)
            on = self.section_of(x) if hs else None
            if not flat_normal or hit is not None:
                where_on = f"the {on}" if on else "the part"
                what = "Inside corner" if hit is not None else "Curved area"
                return {"name": f"{what} on {where_on}, {along}" if on else f"{what} {along}",
                        "where": f"A {'corner' if hit is not None else 'curved area'} on {where_on}, {along}.",
                        "face_kind": "inner" if hit is not None else "curved area", "inner_of": None,
                        "shade": None, "recess": None}
            if near_end and n[k] * sign > 0.2:
                face = self.end_face(sign)
                if hs:
                    if sign == hs["sign"]:
                        where = "The edge around the top of the head, where the flat top meets the side."
                    else:
                        meets = "the thread" if self._threaded_at(x - sign * 0.1 * self.size[k]) else "the side"
                        where = f"The edge around the end of the shaft, where its flat tip meets {meets}."
                    name = f"Rim around the {face}"
                else:
                    name, where = f"Rim around the {self.end(sign)}", f"The edge around the {self.end(sign)} of the part."
                return {"name": name, "where": where, "face_kind": "edge", "inner_of": None, "shade": None,
                        "recess": None}
            if abs(n[k]) < 0.5:
                if hs and on == "head":
                    return {"name": f"Side of the head, {along}", "where": f"On the outside of the head, {along}.",
                            "face_kind": "side surface", "inner_of": None, "shade": None, "recess": None}
                if hs:
                    if self._threaded_at(x):
                        return {"name": f"Thread on the shaft, {along}",
                                "where": f"On the thread of the shaft, {along}.", "face_kind": "side surface",
                                "inner_of": None, "shade": None, "recess": None}
                    return {"name": f"Side of the shaft, {along}", "where": f"On the side of the shaft, {along}.",
                            "face_kind": "side surface", "inner_of": None, "shade": None, "recess": None}
                return {"name": f"Outside of the part, {along}", "where": f"On the outside of the part, {along}.",
                        "face_kind": "side surface", "inner_of": None, "shade": None, "recess": None}
            if hs:
                return {"name": f"Sloped area on the {on}", "where": f"A sloping area on the {on}, {along}.",
                        "face_kind": "sloped surface", "inner_of": None, "shade": None, "recess": None}
            return {"name": f"Sloped area {along}", "where": f"A sloping area on the part, {along}.",
                    "face_kind": "sloped surface", "inner_of": None, "shade": None, "recess": None}
        where = self.where(center) or "in the middle"
        if not flat_normal:
            what = "curved area"
        elif hit is not None:
            what = "inside corner"
        else:
            what = f"surface facing {self.direction(n)}"
        return {"name": _cap(f"{what}, {where}"), "where": _cap(f"a {what}, {where} of the part as the 3D view "
                                                                 "first shows it."),
                "face_kind": what, "inner_of": None, "shade": None, "recess": None}

    def _under_head(self, P: np.ndarray) -> bool:
        """An area in the corner where the shaft meets the underside of the head: on the shaft just under the head
        or on the underside right next to the shaft."""
        hs, k = self.head, self.axis
        below = (hs["x"] - P[:, k]) * hs["sign"]              # from the underside towards the tip
        band = max(0.08 * (self.size[k] - hs["len"]), 2.0, 3 * self.eps)
        reach = hs["shaft_r"] + 0.5 * (hs["r"] - hs["shaft_r"])
        return bool(below.min() >= -2 * self.eps and below.min() <= max(0.5, 2 * self.eps) and below.max() <= band
                    and self._radial(P.copy()).max() <= reach)

    def _two_face_corner(self, P: np.ndarray, N: np.ndarray, F: np.ndarray) -> dict | None:
        """An area spread over two faces that meet in an inside corner (each faces the other)."""
        ids, counts = np.unique(F[F >= 0], return_counts=True)
        order = np.argsort(-counts)[:2]
        big = [int(ids[o]) for o in order if counts[o] >= 0.25 * len(F)]
        if len(big) < 2:
            return None
        a, b = F == big[0], F == big[1]
        da = P[b][cKDTree(P[b]).query(P[a])[1]] - P[a]           # from each spot on one face to the other face
        db = P[a][cKDTree(P[a]).query(P[b])[1]] - P[b]
        if np.einsum("ij,ij->i", N[a], da).mean() <= 0 or np.einsum("ij,ij->i", N[b], db).mean() <= 0:
            return None
        across = [x for x in big if self.faces[x]["type"] == "plane"
                  and self.info[x]["kind"] in ("step", "floor", "end", "outer", "pocket", "facing")]
        return self._corner_region(across[0] if across else big[0])

    def _corner_region(self, hit: int) -> dict:
        """An area whose normal runs into another face close by: an inside corner, shaded by that face."""
        rec = self.recess_of(hit)
        inf = self.info[hit]
        if rec is None and inf["kind"] in ("floor", "wall") and self.axis is not None:
            end = inf.get("end") or inf.get("opens")
            if end:
                place = ("head" if end == self.head["sign"] else "tip of the shaft") if self.head else self.end(end)
                return {"name": f"Inside corner of the recess in the {place}",
                        "where": f"Inside the {place}: a corner deep in the recess, shaded by its walls.",
                        "face_kind": "inner", "inner_of": hit, "shade": "the walls of the recess",
                        "recess": {"noun": "recess"}}
        if rec is not None:
            return {"name": f"Inside corner of the {rec['name']}",
                    "where": f"{rec['inside']}: a corner deep in the {rec['name']}, shaded by its walls.",
                    "face_kind": "inner", "inner_of": hit, "shade": "its walls", "recess": rec}
        if inf.get("role") == "underside":
            return {"name": "Corner under the head",
                    "where": "Where the shaft meets the underside of the head: a tight inside corner that is hard "
                             "for a scanner to see into.",
                    "face_kind": "inner", "inner_of": hit, "shade": "the head", "recess": None}
        label = _base(self.labels[hit])
        return {"name": _cap(f"inside corner by the {label}"),
                "where": f"An inside corner next to the {label}, which shades it.",
                "face_kind": "inner", "inner_of": hit, "shade": f"the {label}", "recess": None}

    def _face_region(self, i: int, part: float, center) -> dict:
        f, inf = self.faces[i], self.info[i]
        kind = inf["kind"]
        k = self.axis
        label = self.grip_noun(self.grips[inf["grip"]]) if "grip" in inf else self.labels[i]
        context, desc = self._thing(i)
        name = _cap(label)
        if part < 0.6:
            desc = desc.split(" (the ")[0]
            across = kind in ("end", "floor", "step", "outer", "pocket") or (k is None and kind == "facing")
            if f["type"] == "cylinder" and k is not None and abs(f["axis"][k]) >= COS5 \
                    and f["length"] >= 0.4 * self.size[k] and " at the " not in label:
                along = self.along_words(float(center[k]))
                name, desc = f"{_cap(_base(label))}, {along}", f"{desc}, {along}"
            elif f["type"] == "cylinder":
                name, desc = f"Part of the {_base(label)}", f"part of {desc}"
            elif across and k is not None:
                pos = self._on_face(center, f)
                lead = {"centre": "middle", "near the rim": "outer edge"}.get(pos, "part")
                name, desc = f"{_cap(lead)} of the {_base(label)}", f"the {lead} of {desc}" if lead != "part" else \
                    f"part of {desc}"
            elif across:
                pos = self._on_face(center, f)
                name, desc = f"{_cap(_base(label))}, {pos}", f"{desc}, {pos}"
            elif k is not None:
                along = self.along_words(float(center[k]))
                name, desc = f"{_cap(_base(label))}, {along}", f"{desc}, {along}"
        return {"name": name, "where": self._sentence(context, desc), "face_kind": kind, "inner_of": None,
                "shade": None, "recess": self.recess_of(i)}

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

    def is_end_like(self, normal, center) -> int:
        """+1/-1 when an area faces one end of the main axis (the scanner missed that end), else 0."""
        if self.axis is None:
            return 0
        a = normal[self.axis]
        return (1 if a > 0 else -1) if abs(a) >= 0.7 else 0

    def end_face_id(self, sign: int) -> int | None:
        """The biggest flat face at that end of the main axis."""
        ids = [i for i, inf in enumerate(self.info) if inf["kind"] == "end" and inf["end"] == sign]
        return max(ids, key=lambda i: self.faces[i]["area"]) if ids else None

    def scan_finds(self, i: int, dev: float, amount: str, that: bool = False) -> str:
        """What follows 'The scan finds' for a face whose scanned surface sits dev from the golden one: 'the bottom of
        the hex socket 2.41 mm deeper than the golden model' (that: 'that bottom ...')."""
        f, inf = self.faces[i], self.info[i]
        label = _base(self.labels[i])
        if inf["kind"] in ("floor", "pocket"):
            subject = "that bottom" if that else f"the {label}"
            return f"{subject} {amount} mm {'deeper' if dev < 0 else 'shallower'} than the golden model"
        if f["type"] == "cylinder" and f["hole"]:
            subject = "that hole's wall" if that else f"the wall of the {label}"
            return (f"{subject} {amount} mm {'farther out' if dev < 0 else 'farther in'} than the golden model, so "
                    f"the hole is {'bigger' if dev < 0 else 'smaller'}")
        subject = ("that round side" if f["type"] == "cylinder" else "that face") if that else f"the {label}"
        return (f"{subject} {amount} mm {'outside' if dev > 0 else 'inside'} the golden model's surface "
                f"({'more' if dev > 0 else 'less'} material)")

    def near_end(self, P: np.ndarray, sign: int) -> bool:
        """Does an area reach that end of the main axis?"""
        k = self.axis
        end_x = self.hi[k] if sign > 0 else self.lo[k]
        return bool(np.abs(P[:, k] - end_x).min() <= max(0.03 * self.size[k], 2.0))

    def pronoun(self, i: int) -> str:
        f, inf = self.faces[i], self.info[i]
        if inf["kind"] in ("floor", "pocket"):
            return "that bottom"
        if f["type"] == "cylinder":
            return "that hole" if f["hole"] else "that round side"
        return "that face"

    def faces_phrase(self, ids: list[int]) -> str:
        """'bottom of the hex socket', 'two grooves in the knurled grip', 'top of the head and the tip of the shaft'."""
        bases = [_base(self.labels[i]) for i in ids]
        if len(ids) == 2 and bases[0] == bases[1]:
            return f"two {_plural(bases[0])}"
        return " and the ".join(self.labels[i] for i in ids)

    def summary(self) -> str:
        """One sentence on what the part is, from what was recognised."""
        hs = self.head
        holes = [i for i, inf in enumerate(self.info) if inf["kind"] == "hole" and "recess" not in inf]
        extras = []
        for s, rec in sorted(self.recesses.items(), key=lambda kv: -kv[0] * (hs["sign"] if hs else 1)):
            if rec["shape"] in ("hex", "square"):
                extras.append(f"a {rec['noun']} in the {rec['place']}")
            elif rec["shape"] == "round":
                extras.append(f"a {rec['name']}")
            else:
                extras.append(f"a recess in the {rec['place']}")
        if len(holes) == 1:
            extras.append(f"a {self.labels[holes[0]]}")
        elif holes:
            extras.append(f"{len(holes)} holes")
        if hs:
            head_grip = next((g for g in self.grips if g["on"] == "head"), None)
            head = {"knurl": "a knurled head", "hex": "a hex head", "square": "a square head",
                    "polygon": "a head with flat sides"}.get(head_grip["shape"] if head_grip else "", "a head")
            shaft = "a threaded shaft" if any(self.section_of((z["x0"] + z["x1"]) / 2) == "shaft"
                                              for z in self.threads) else "a shaft"
            text = f"A {'round ' if self.texture['round'] else ''}part with {head} and {shaft}"
        elif self.axis is not None:
            text = f"A {'round' if self.texture['round'] else 'long'} part"
            if self.ends and self.ends[0] in ("wide end", "narrow end"):
                text += ", wider at one end"
            if self.threads:
                extras.insert(0, "a thread")
            for g in self.grips:
                if g["shape"] in ("polygon", "flats"):
                    extras.append("flat sides")
                elif g["shape"] == "ribs":
                    extras.append(f"{g['count']} {g['relief']} round it")
                else:
                    extras.append(f"a {self.grip_noun(g)}")
        else:
            text = "A part with no main axis: its sides are named as the 3D view first shows it"
            pockets = [inf for inf in self.info if inf["kind"] == "pocket"]
            if pockets:
                extras.append(f"a pocket in the {pockets[0]['word']}" if len(pockets) == 1 else f"{len(pockets)} pockets")
        return text + ("; " + " and ".join(extras) if extras else "") + "."

    # ------------------------------------------------------------------ sizes and measurements
    def size_words(self, k: int) -> dict:
        """Words for the overall size along axis k."""
        if self.axis == k:
            if self.head is not None:
                what = "From the top of the head to the tip of the shaft."
            else:
                ref = self._reference_end()
                what = f"From the {self.end(ref)} to the {self.end(-ref)} of the part."
            return {"name": "Overall length", "what": what, "more": "longer", "less": "shorter",
                    "group": "Overall size"}
        w0, w1 = self.view[k]
        if self.head is not None:
            cylinder = any(self.info[i]["kind"] == "round" and self._cyl_label(i) == "side of the head"
                           for i in range(len(self.faces)))
            noun = "diameter" if self.texture["round"] and not cylinder else "width"
            return {"name": f"Head {noun} ({w0} to {w1})",
                    "what": f"The widest point across the head, from {w0} to {w1} as the 3D view first shows the "
                            "part.",
                    "more": "bigger" if noun == "diameter" else "wider",
                    "less": "smaller" if noun == "diameter" else "narrower", "group": "Head"}
        word = "height" if w1 == "top" else "depth" if w1 in ("front", "back") else "width"
        more, less = {"height": ("taller", "shorter"), "width": ("wider", "narrower"),
                      "depth": ("deeper", "shallower")}[word]
        return {"name": f"Overall {word} ({w0} to {w1})",
                "what": f"From the part's {w0} side to its {w1} side, as the 3D view first shows it.",
                "more": more, "less": less, "group": "Overall size"}

    def end_word(self, k: int, sign: int) -> str:
        """The end (main axis) or side (others) of the part on that side of axis k, for 'was not scanned'."""
        if self.axis == k:
            return self.end_face(sign) if self.head is not None else self.end(sign)
        return f"{self.view[k][1 if sign > 0 else 0]} side"

    def _ordered(self, i: int, j: int) -> tuple[int, int]:
        """The two faces of a size, the one nearer the head (wide end, top) first."""
        k = self.axis
        if k is None:                         # the outside first, then from the top down
            u = self.up_index()
            key = [(self.info[x]["kind"] != "outer",
                    -(self.faces[x]["point"] if self.faces[x]["type"] == "plane" else self.faces[x]["center"])[u])
                   for x in (i, j)]
        else:
            ref = self._reference_end()
            end_x = self.hi[k] if ref > 0 else self.lo[k]
            key = [abs((self.faces[x]["point"] if self.faces[x]["type"] == "plane" else self.faces[x]["center"])[k]
                       - end_x) for x in (i, j)]
        return (i, j) if key[0] <= key[1] else (j, i)

    def _group_of(self, i: int, j: int) -> str:
        for x in (i, j):
            rec = self.recess_of(x)
            if rec is not None:
                return rec["group"]
            inf = self.info[x]
            if inf["kind"] == "pocket" or inf.get("pocket_of") is not None:
                word = inf["word"] if inf["kind"] == "pocket" else self.info[inf["pocket_of"]]["word"]
                return f"Pocket in the {word}"
        if self.head is not None:
            parts = {self._part_of(i), self._part_of(j)}
            if parts == {"head"}:
                return "Head"
            if parts == {"shaft"}:
                return "Shaft"
        return "Other sizes"

    def _section_between(self, i: int, j: int) -> dict | None:
        """The length of one section of a turned part, between two neighbouring steps or ends."""
        k = self.axis
        if k is None or self.profile is None:
            return None
        levels = [x for x, inf in enumerate(self.info) if inf["kind"] in ("end", "step")]
        if i not in levels or j not in levels:
            return None
        pos = sorted({round(float(self.faces[x]["point"][k]), 4) for x in levels})
        xi, xj = sorted((float(self.faces[i]["point"][k]), float(self.faces[j]["point"][k])))
        if any(xi + self.eps < p < xj - self.eps for p in pos):
            return None
        R = self.radius_at(xi + self.eps, xj - self.eps)
        a, b = self._ordered(i, j)
        span = f"From the {self.labels[a]} to the {self.labels[b]}"
        if self.head is not None:
            if self.section_of((xi + xj) / 2) != "shaft" or not self.texture["round"]:
                return None
            name = f"Length of the Ø{2 * R:.2f} part of the shaft"
            return {"name": name, "what": f"{span}: the length of the shaft where it is Ø{2 * R:.2f} across.",
                    "more": "longer", "less": "shorter", "group": "Shaft"}
        if len(pos) == 3:                     # two sections: the wide part and the narrow part
            first = abs(xi - pos[0]) <= self.eps and abs(xj - pos[1]) <= self.eps
            other = (pos[1], pos[2]) if first else (pos[0], pos[1])
            R_other = self.radius_at(other[0] + self.eps, other[1] - self.eps)
            if abs(R - R_other) < 0.02 * max(R, R_other):
                return None
            which = "wide" if R > R_other else "narrow"
            return {"name": f"Length of the {which} part", "what": f"{span}: the length of the {which} part.",
                    "more": "longer", "less": "shorter", "group": "Outside"}
        if self.texture["round"]:
            return {"name": f"Length of the Ø{2 * R:.2f} part",
                    "what": f"{span}: the length of the part where it is Ø{2 * R:.2f} across.",
                    "more": "longer", "less": "shorter", "group": "Outside"}
        return None

    def _grip_words(self, g: dict) -> dict:
        on = g["on"]
        across = f"the {on}" if on else "the part"
        if g["shape"] == "knurl":
            if g["relief"] == "grooves":
                name = "Knurled grip, across the grooves"
                what = (f"Across {across} from the flat bottom of one groove of the knurled grip to the groove "
                        "opposite.")
            else:
                name = "Knurled grip, across the ridges"
                what = f"Across {across} from the flat top of one ridge of the knurled grip to the ridge opposite."
        elif g["shape"] == "ribs":
            one = g["relief"][:-1]
            name = f"{_cap(on) if on else 'Part'}, across the {g['relief']}"
            what = (f"Across {across} from the flat {'bottom' if one == 'groove' else 'top'} of one {one} to the "
                    f"{one} opposite.")
        elif g["shape"] in ("hex", "square"):
            noun = self.grip_noun(g)
            name = f"{_cap(noun)}, across flats"
            what = f"Across two opposite flats of the {noun}: the spanner (wrench) size."
        else:
            name = f"{_cap(on) if on else 'Flat sides'}, across flats"
            what = f"Across {across} between two opposite flat sides."
        group = "Head" if on == "head" else "Shaft" if on == "shaft" else "Outside"
        return {"name": name, "what": what, "more": "wider", "less": "narrower", "group": group, "series": name}

    def pair_words(self, kind: str, i: int, j: int) -> dict:
        """Words for a size between two flat faces: thickness (opposite faces, material between), gap (opposite
        faces, space between) or step (faces facing the same way)."""
        more, less = {"thickness": ("thicker", "thinner"), "gap": ("wider", "narrower"),
                      "step": ("longer", "shorter")}[kind]
        a, b = self.info[i], self.info[j]
        roles = {a.get("role"), b.get("role")}
        hs = self.head
        for x, y in ((i, j), (j, i)):
            fx, fy = self.info[x], self.info[y]
            # a recess: from the end face down to its bottom
            if kind == "step" and fy["kind"] == "floor" and "recess" in fy and fx["kind"] == "end" \
                    and fx["end"] == fy["end"]:
                rec = self.recesses[fy["recess"]]
                name = f"{_cap(rec['noun'])} depth" if hs and rec["shape"] in ("hex", "square") \
                    and rec["end"] == hs["sign"] else f"Depth of the {rec['name']}"
                return {"name": name, "what": f"From the {self.labels[x]} down to the bottom of the {rec['name']}.",
                        "more": "deeper", "less": "shallower", "group": rec["group"]}
            if kind == "step" and fy["kind"] == "pocket" and fx["kind"] == "outer" and fx["word"] == fy["word"]:
                return {"name": f"Depth of the pocket in the {fy['word']}",
                        "what": f"From the {fx['word']} face down to the bottom of the pocket.",
                        "more": "deeper", "less": "shallower", "group": f"Pocket in the {fy['word']}"}
            # what is left under a recess or a pocket (the faces overlap: material between them)
            if kind == "thickness" and not hs and fx["kind"] == "floor" and "recess" in fx:
                rec = self.recesses[fx["recess"]]
                return {"name": f"Thickness under the {rec['name']}",
                        "what": f"From the bottom of the {rec['name']} to the {self.labels[y]}: the material left "
                                "under it.", "more": more, "less": less, "group": rec["group"]}
            if kind == "thickness" and fx["kind"] == "pocket":
                word = "under" if fx["word"] == "top" else "behind"
                return {"name": f"Thickness {word} the pocket in the {fx['word']}",
                        "what": f"From the bottom of the pocket in the {fx['word']} to the {self.labels[y]}: the "
                                f"material left {word} it.", "more": more, "less": less,
                        "group": f"Pocket in the {fx['word']}"}
        if kind == "thickness" and roles == {"head_top", "underside"}:
            return {"name": "Head height", "what": "From the top of the head down to its underside.",
                    "more": "taller", "less": "shorter", "group": "Head"}
        if hs is not None and "underside" in roles and {a["kind"], b["kind"]} == {"floor", "step"}:
            rec = self.recess_of(i if a["kind"] == "floor" else j)
            if rec is not None and rec["end"] == hs["sign"]:
                what = rec["noun"] if rec["shape"] in ("hex", "square") else "recess"
                return {"name": f"{_cap(rec['short'])} to underside of head",
                        "what": f"From the bottom of the {rec['name']} down to the level of the head's underside: "
                                f"how much of the head's height is left under the {what}.",
                        "more": "longer", "less": "shorter", "group": rec["group"]}
        if kind == "step" and roles == {"tip", "underside"}:
            return {"name": "Length under the head", "what": "From the underside of the head to the tip of the shaft.",
                    "more": "longer", "less": "shorter", "group": "Shaft"}
        if kind == "gap" and a["kind"] == b["kind"] == "wall" and "recess" in a and a.get("recess") == b.get("recess"):
            rec = self.recesses[a["recess"]]
            if rec["shape"] in ("hex", "square"):
                name = f"{_cap(rec['name'])}, across flats"
                what = (f"Across two opposite flat sides of the {rec['name']}: the size of "
                        f"{'hex key' if rec['shape'] == 'hex' else 'square drive'} that fits.")
            else:
                name = f"Width across the {rec['name']}"
                what = f"Across the {rec['name']}, from one inside wall to the wall opposite."
            return {"name": name, "what": what, "more": "wider", "less": "narrower", "group": rec["group"],
                    "series": name}
        if kind == "gap" and a.get("pocket_of") is not None and a.get("pocket_of") == b.get("pocket_of"):
            word = self.info[a["pocket_of"]]["word"]
            s = int(np.argmax(np.abs(self.faces[i]["normal"])))
            w0, w1 = self.view[s]
            return {"name": f"Width of the pocket in the {word} ({w0} to {w1})",
                    "what": f"Across the pocket in the {word}, from its {w0} wall to its {w1} wall.",
                    "more": "wider", "less": "narrower", "group": f"Pocket in the {word}"}
        for x, y in ((i, j), (j, i)):
            fx, fy = self.info[x], self.info[y]
            if kind == "thickness" and fx["kind"] == "outer" and fy.get("pocket_of") is not None:
                side, word = self._pocket_side(y), self.info[fy["pocket_of"]]["word"]
                return {"name": f"{_cap(side)} wall thickness",
                        "what": f"From the part's {fx['word']} face to the {side} wall of the pocket in the {word}: "
                                "how thick the wall between them is.",
                        "more": more, "less": less, "group": f"Pocket in the {word}"}
        if kind == "thickness" and "grip" in a and a.get("grip") == b.get("grip"):
            return self._grip_words(self.grips[a["grip"]])
        section = self._section_between(i, j)
        if section is not None:
            return section
        first, second = self._ordered(i, j)
        more, less = ("wider", "narrower") if kind == "gap" else ("longer", "shorter")
        return {"name": f"{_cap(_base(self.short(first)))} to {_base(self.short(second))}",
                "what": f"From the {_base(self.labels[first])} to the {_base(self.labels[second])}.",
                "more": more, "less": less, "group": self._group_of(i, j)}

    def diameter_words(self, i: int) -> dict:
        """Words for the diameter of a round face (a hole or a shaft)."""
        inf = self.info[i]
        label = self.labels[i]
        if inf["kind"] == "hole":
            rec = self.recess_of(i)
            return {"name": f"{_cap(label)}: diameter", "what": f"Across the {_base(label)}.", "more": "bigger",
                    "less": "smaller", "group": rec["group"] if rec else "Holes"}
        base = _base(label)
        if self.head is not None and inf.get("central"):
            if base == "side of the head":
                return {"name": "Head diameter", "what": "Across the round outside of the head.", "more": "bigger",
                        "less": "smaller", "group": "Head"}
            name = "Shaft diameter" if base == "side of the shaft" else f"Diameter of the {base}"
            what = "Across the round shaft." if base == "side of the shaft" else f"Across the {base}."
            return {"name": name, "what": what, "more": "bigger", "less": "smaller", "group": "Shaft"}
        if self.axis is not None and inf.get("central"):
            if base == "round outside":
                name, what = "Outside diameter", "Across the round outside of the part."
            elif base.startswith("round side at the "):
                end = base[len("round side at the "):]
                name, what = f"Diameter at the {end}", f"Across the round outside of the part at the {end}."
            else:
                name, what = f"{_cap(label)}: diameter", f"Across the {base} of the part."
            return {"name": name, "what": what, "more": "bigger", "less": "smaller", "group": "Outside"}
        return {"name": f"{_cap(label)}: diameter", "what": f"Across the {base}.", "more": "bigger",
                "less": "smaller", "group": "Other sizes"}

    def position_words(self, i: int) -> dict:
        """Words for where a round face sits (its centre line against the golden model's, after the line-up)."""
        f = self.faces[i]
        d = self.diameter_words(i)
        subject = d["name"].rsplit(": diameter", 1)[0]
        subject = {"Head diameter": "Head", "Shaft diameter": "Shaft", "Outside diameter": "Round outside"}.get(
            subject, subject)
        if subject.startswith("Diameter at the "):
            subject = f"Round side at the {subject[len('Diameter at the '):]}"
        if f["hole"]:
            what = ("How far the hole's centre sits from where the golden model puts it, once the scan is lined up "
                    "(0: exactly in place).")
        else:
            what = ("How far the centre line of this round side sits from where the golden model puts it, once the "
                    "scan is lined up (0: exactly in line).")
        return {"name": f"{subject}: where it sits", "what": what, "group": d["group"]}

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
