"""Golden model check: does a scan match the golden model? (docs/golden-model.md)

The golden model is the part as it should be: a CAD model (STEP/IGES) or a trusted mesh (STL/OBJ/PLY). The check
builds on compare.compare_to_reference (a rigid best-fit line-up and the signed deviation; nothing is rescaled), then:

1. Scan quality over the golden surface. Every scan point is projected onto its closest golden point, and each small
   spot of the golden surface gets a verdict from the points that landed around it:
     good      enough points, within the tolerance
     missing   (almost) no points: not scanned
     thin      far fewer points than elsewhere: seen only at a glancing angle
     rough     the points scatter more than the tolerance allows (shiny or dark surface)
     off       the scanned surface sits beyond the tolerance, outside (+) or inside (-) the golden surface
   Spots on sharp edges are never called rough or off: every scanner rounds edges a little.
2. Regions. Neighbouring spots with the same problem become a region with a plain name ("Bottom of the hex
   socket", "Corner under the head"), a sentence saying where it is, its size and what to do about it (scan it again
   and how, or check it by hand). Names come from golden_words, which recognises what the part is built from.
3. Measurements. The golden model's flat faces and round faces (holes, shafts) are found on the golden mesh and
   measured again on the scan points that landed on them: overall size, thicknesses, gaps, steps, diameters and hole
   positions. Each gets the golden value, the scanned value, the difference and a verdict against the tolerance that
   allows for how precisely the scan pins the value down (ISO 14253-1 style: ok, off, or too close to call), plain
   words (what it is, from where to where), a dimension line on the golden surfaces and a caveat when one of its
   faces carries a problem area. A cut through the golden model (cut_section) is the drawing the sizes are shown on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .compare import CompareParams, ReferenceSurface, compare_to_reference
from .golden_words import PartWords
from .params import ParamsMixin

Log = Callable[[str], None]

GOOD, MISSING, THIN, ROUGH, OFF_OUT, OFF_IN = range(6)
LEGEND = [
    {"code": GOOD, "key": "good", "label": "Matches", "color": "#9cc5a1"},
    {"code": MISSING, "key": "missing", "label": "Not scanned", "color": "#8b5cf6"},
    {"code": THIN, "key": "thin", "label": "Too few points", "color": "#f5b301"},
    {"code": ROUGH, "key": "rough", "label": "Rough: the points scatter", "color": "#f97316"},
    {"code": OFF_OUT, "key": "off_out", "label": "Off: scan sits outside (bigger)", "color": "#e5484d"},
    {"code": OFF_IN, "key": "off_in", "label": "Off: scan sits inside (smaller)", "color": "#3b82f6"},
]
KIND = {MISSING: "missing", THIN: "thin", ROUGH: "rough", OFF_OUT: "off", OFF_IN: "off"}
AXES = "XYZ"
SIDE = {(0, 1): "right", (0, -1): "left", (1, 1): "back", (1, -1): "front", (2, 1): "top", (2, -1): "bottom"}
ENDS = (("left", "right"), ("front", "back"), ("bottom", "top"))

MAX_POINTS = 2_000_000     # scan points analysed (a random subset of bigger scans)
MAX_SPOTS = 150_000        # golden surface spots
N_EFF = 400                # scanner noise is correlated between neighbouring points: count at most this many as
                           # independent when judging how precisely a value is known
MISSING_RATIO = 0.12       # spots with fewer points than this share of the usual density: not scanned
THIN_RATIO = 0.35          # ... fewer than this share: too few points
CHECK_MESH_TRIANGLES = 520_000   # the coloured golden surface: fine enough to show every listed area, and under
                                 # io.preview's 600k so the viewer gets it as it is (decimation would erase colours)
MOSTLY_MATCHES_PCT = 95.0  # a check that found problems on less than 5 % of the surface "mostly matches"
REPORT_VERSION = 3         # 2: names from golden_words, "Show me" views with radius, pins, match_pct; 3: the part's
                           # vocabulary (head, shaft, hex socket...), where / number per area, what / group / series /
                           # ends / caveat per size, sizes_story, the section drawing, check_face
SECTION_BYTES = 60_000     # the section drawing's JSON stays under this


@dataclass
class GoldenParams(ParamsMixin):
    tolerance: float = 0.1      # +/- mm: how far the scan may be from the golden model and still match
    align: str = "auto"         # auto (search + fit) | icp (roughly lined up already) | none
    max_distance: float = 0.0   # scan points farther than this from the golden surface are not part of the part;
                                # 0 -> 3 % of the golden model's diagonal


# --------------------------------------------------------------------------- small helpers
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _side(normal, limit_deg: float = 25.0) -> tuple[int, int] | None:
    """(axis, sign) when the direction is within limit_deg of a golden axis."""
    n = _unit(normal)
    k = int(np.argmax(np.abs(n)))
    if abs(n[k]) < np.cos(np.radians(limit_deg)):
        return None
    return k, (1 if n[k] > 0 else -1)


def _facing(normal) -> str:
    s = _side(normal)
    if s is None:
        n = _unit(normal)
        return "at an angle (" + ", ".join(f"{AXES[k]} {n[k]:+.2f}" for k in range(3)) + ")"
    return f"{SIDE[s]} ({'+' if s[1] > 0 else '-'}{AXES[s[0]]})"


def _where(point, lo, hi, skip: int | None = None) -> str:
    """'near the left end', 'near the top back', or '' when central; skip: an axis not to mention."""
    words = []
    for k in (2, 0, 1):
        if k == skip or hi[k] - lo[k] <= 1e-9:
            continue
        t = (point[k] - lo[k]) / (hi[k] - lo[k])
        if t < 0.25:
            words.append(ENDS[k][0])
        elif t > 0.75:
            words.append(ENDS[k][1])
    return f"near the {' '.join(words[:2])}" if words else ""


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _groups(labels: np.ndarray, keep: np.ndarray) -> dict[int, np.ndarray]:
    """{label: indices} for the labels in keep."""
    order = np.argsort(labels, kind="stable")
    sl = labels[order]
    out = {}
    for lab in keep:
        i0, i1 = np.searchsorted(sl, lab), np.searchsorted(sl, lab, side="right")
        out[int(lab)] = order[i0:i1]
    return out


def _group_median(values: np.ndarray, groups: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Median of values per group (NaN for empty groups) and the group sizes."""
    counts = np.bincount(groups, minlength=n)
    med = np.full(n, np.nan)
    if len(values) == 0:
        return med, counts
    order = np.lexsort((values, groups))
    v = values[order]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    has = counts > 0
    lo = starts[has] + (counts[has] - 1) // 2
    hi = starts[has] + counts[has] // 2
    med[has] = (v[lo] + v[hi]) / 2
    return med, counts


def _robust_sigma(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values)))) if len(values) else 0.0


def _se_median(sigma: float, n: int) -> float:
    return 1.2533 * sigma / np.sqrt(max(1, min(n, N_EFF)))


def _judge(diff: float, u: float, tol: float) -> str:
    if abs(diff) + u <= tol:
        return "ok"
    if abs(diff) - u > tol:
        return "off"
    return "close"


def _fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Least-squares circle (algebraic start, geometric refinement): centre, radius, rms, covariance of
    (cx, cy, r) for unit noise."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-18)))
    J = np.c_[np.ones(len(x)), np.zeros(len(x)), -np.ones(len(x))]
    for _ in range(20):
        dx, dy = x - cx, y - cy
        ri = np.maximum(np.hypot(dx, dy), 1e-12)
        res = ri - r
        J = np.c_[-dx / ri, -dy / ri, -np.ones(len(x))]
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.linalg.norm(step) < 1e-10:
            break
    res = np.hypot(x - cx, y - cy) - r
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)
    return np.array([cx, cy]), float(abs(r)), float(np.sqrt(np.mean(res ** 2))), cov


def _arc_deg(angles_deg: np.ndarray) -> float:
    """How much of the full circle the angles cover (360 minus the biggest gap)."""
    if len(angles_deg) < 3:
        return 0.0
    a = np.sort(np.mod(angles_deg, 360.0))
    gaps = np.diff(np.r_[a, a[0] + 360.0])
    return float(360.0 - gaps.max())


def _plane_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = _unit(np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0]))
    return u, np.cross(axis, u)


# --------------------------------------------------------------------------- the golden model's faces
def _adjacent_pairs(F: np.ndarray):
    """Triangle pairs sharing an edge (a, b, the edge's two vertices), the boundary edges and their triangles."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    T = np.tile(np.arange(len(F)), 3)
    key = E[:, 0].astype(np.int64) * (int(F.max()) + 1) + E[:, 1]
    order = np.argsort(key, kind="stable")
    key, T, E = key[order], T[order], E[order]
    same = key[1:] == key[:-1]
    first = np.r_[True, ~same]
    last = np.r_[~same, True]
    return T[:-1][same], T[1:][same], E[:-1][same], E[first & last], T[first & last]


def find_faces(ref: ReferenceSurface, min_area: float) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Flat faces and round faces (holes, shafts) of the golden mesh.

    Returns (faces, face index per triangle (-1: other surfaces), sharp edges as vertex pairs). Flat faces: triangles
    joined across edges bent less than 2 degrees, flat within a few microns, and either bounded mostly by sharp edges
    or wide in both directions (the long thin flat facets of a tessellated round face are neither). Round faces:
    smoothly joined triangles whose normals all lie across one axis, on a circle around it; pieces of the same
    cylinder are joined."""
    V, F, Nt, At = ref.vertices, ref.triangles, ref.normals, ref.areas
    a, b, ev, boundary, btri = _adjacent_pairs(F)
    cos_ab = np.einsum("ij,ij->i", Nt[a], Nt[b])
    sharp = np.concatenate([ev[cos_ab < np.cos(np.radians(30.0))], boundary])
    edge_len = np.linalg.norm(V[ev[:, 0]] - V[ev[:, 1]], axis=1)
    bound_len = np.linalg.norm(V[boundary[:, 0]] - V[boundary[:, 1]], axis=1)
    tri_face = np.full(len(F), -1, dtype=np.int64)
    faces: list[dict] = []
    centroids = V[F].mean(1)

    flat = cos_ab > np.cos(np.radians(2.0))
    labels = _components(len(F), a[flat], b[flat])
    nl = int(labels.max()) + 1
    area = np.bincount(labels, weights=At, minlength=nl)
    # each flat patch's outline: how much of it is a sharp edge (bent 25 degrees or more) or the mesh's own border
    rim = labels[a] != labels[b]
    bent = rim & (cos_ab < np.cos(np.radians(25.0)))
    outline = sum(np.bincount(labels[x[rim]], weights=edge_len[rim], minlength=nl) for x in (a, b))
    outline += np.bincount(labels[btri], weights=bound_len, minlength=nl)
    sharp_outline = sum(np.bincount(labels[x[bent]], weights=edge_len[bent], minlength=nl) for x in (a, b))
    sharp_outline += np.bincount(labels[btri], weights=bound_len, minlength=nl)
    plane_tol = max(0.005, 2e-5 * ref.diagonal)
    for lab, t in _groups(labels, np.flatnonzero(area >= min_area)).items():
        w = At[t]
        normal = _unit((Nt[t] * w[:, None]).sum(0))
        point = (centroids[t] * w[:, None]).sum(0) / w.sum()
        P = V[np.unique(F[t])]
        if np.abs((P - point) @ normal).max() > plane_tol:
            continue
        u, v = _plane_frame(normal)
        uv = np.c_[(P - point) @ u, (P - point) @ v]
        ev2, evec2 = np.linalg.eigh(np.cov(uv.T))
        ext = np.ptp(uv @ evec2, axis=0)            # (narrow, long) size of the patch
        wide = ext[0] >= max(2.0, 0.2 * ext[1])
        if not wide and sharp_outline[lab] < 0.5 * max(outline[lab], 1e-12):
            continue                               # a facet strip of a round face
        tri_face[t] = len(faces)
        faces.append({"type": "plane", "normal": normal, "point": point, "offset": float(normal @ point),
                      "area": float(w.sum()), "tris": t})

    rest = tri_face < 0
    smooth = (cos_ab > np.cos(np.radians(30.0))) & rest[a] & rest[b]
    labels = _components(len(F), a[smooth], b[smooth])
    area = np.bincount(labels, weights=np.where(rest, At, 0.0))
    cands = []
    for lab, t in _groups(labels, np.flatnonzero(area >= min_area)).items():
        t = t[rest[t]]
        cyl = _fit_cylinder(V, F, Nt, At, centroids, t)
        if cyl is not None:
            cands.append(cyl)
    for cyl in _merge_coaxial(V, F, Nt, At, centroids, cands):
        if cyl["arc"] < 150.0:        # fillets and rounded corners: not measured as a diameter
            continue
        tri_face[cyl["tris"]] = len(faces)
        faces.append(cyl)
    return faces, tri_face, sharp


def _fit_cylinder(V, F, Nt, At, centroids, t) -> dict | None:
    if len(t) < 4:
        return None
    n, w = Nt[t], At[t]
    evals, evecs = np.linalg.eigh((n * w[:, None]).T @ n)
    if evals[0] > 0.01 * evals.sum() or evals[1] < 0.05 * evals.sum():   # not all across one axis, or all one way
        return None
    axis = evecs[:, 0]
    P = V[np.unique(F[t])]
    u, v = _plane_frame(axis)
    xy = np.c_[P @ u, P @ v]
    c2, r, rms, _ = _fit_circle(xy)
    if not np.isfinite(r) or r <= 0 or rms > max(0.01, 0.01 * r):
        return None
    along = P @ axis
    t0, t1 = float(along.min()), float(along.max())
    center = c2[0] * u + c2[1] * v + axis * (t0 + t1) / 2
    cen = centroids[t]
    radial = cen - center - np.outer((cen - center) @ axis, axis)
    hole = float(np.sum(np.einsum("ij,ij->i", radial, n) * w)) < 0
    arc = _arc_deg(np.degrees(np.arctan2(xy[:, 1] - c2[1], xy[:, 0] - c2[0])))
    return {"type": "cylinder", "axis": axis, "center": center, "radius": r, "hole": hole, "arc": arc,
            "length": t1 - t0, "area": float(w.sum()), "tris": t}


def _merge_coaxial(V, F, Nt, At, centroids, cands: list[dict]) -> list[dict]:
    """Pieces of one hole or shaft (CAD splits cylinders at seams and where other holes cross them)."""
    parent = list(range(len(cands)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, ci in enumerate(cands):
        for j in range(i + 1, len(cands)):
            cj = cands[j]
            if ci["hole"] != cj["hole"] or abs(ci["axis"] @ cj["axis"]) < np.cos(np.radians(0.5)):
                continue
            if abs(ci["radius"] - cj["radius"]) > max(0.005, 1e-3 * ci["radius"]):
                continue
            off = cj["center"] - ci["center"]
            if np.linalg.norm(off - (off @ ci["axis"]) * ci["axis"]) > max(0.01, 2e-3 * ci["radius"]):
                continue
            parent[root(j)] = root(i)
    merged: dict[int, list[int]] = {}
    for i in range(len(cands)):
        merged.setdefault(root(i), []).append(i)
    out = []
    for members in merged.values():
        if len(members) == 1:
            out.append(cands[members[0]])
            continue
        t = np.concatenate([cands[i]["tris"] for i in members])
        cyl = _fit_cylinder(V, F, Nt, At, centroids, t)
        out.append(cyl if cyl is not None else max((cands[i] for i in members), key=lambda c: c["area"]))
    return out


def _face_labels(faces: list[dict]) -> list[str]:
    """Plain names: 'top face', 'top face at Z 12.00', 'Ø6.00 hole', 'Ø20.00 round face'."""
    labels = []
    for f in faces:
        if f["type"] == "plane":
            s = _side(f["normal"], 5.0)
            if s is None:
                labels.append("angled face facing " + _facing(f["normal"]))
                continue
            same = [g for g in faces if g["type"] == "plane" and _side(g["normal"], 5.0) == s]
            name = f"{SIDE[s]} face"
            if len(same) > 1:
                name += f" at {AXES[s[0]]} {f['point'][s[0]]:.2f}"
            labels.append(name)
        else:
            d = 2 * f["radius"]
            name = f"Ø{d:.2f} {'hole' if f['hole'] else 'round face'}"
            same = [g for g in faces if g["type"] == "cylinder" and g["hole"] == f["hole"]
                    and abs(2 * g["radius"] - d) < 0.01]
            if len(same) > 1:
                s = _side(f["axis"], 5.0)
                ks = [k for k in range(3) if s is None or k != s[0]]
                name += " at " + ", ".join(f"{AXES[k]} {f['center'][k]:.1f}" for k in ks)
            labels.append(name)
    return labels


def refine_long_edges(mesh: o3d.geometry.TriangleMesh, max_edge: float,
                      max_triangles: int = CHECK_MESH_TRIANGLES) -> o3d.geometry.TriangleMesh:
    """Split edges longer than max_edge at their midpoints (longest first, up to max_triangles) without moving the
    surface. CAD tessellations draw a flat face as a few huge triangles, and colours painted per vertex cannot show a
    problem inside them: the corner vertices sit on the face's edges, which are never judged, so a whole off face
    looked like it matched. Neighbouring triangles split the same shared edges, so no cracks or T-junctions appear."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    N = np.asarray(mesh.vertex_normals, dtype=np.float64) if mesh.has_vertex_normals() else None
    for _ in range(16):
        T = len(F)
        budget = max_triangles - T
        if budget <= 1:
            break
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])           # edge k of triangle t at row k*T + t
        uniq, inv = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True)
        inv = inv.ravel()
        length = np.linalg.norm(V[uniq[:, 0]] - V[uniq[:, 1]], axis=1)
        mark = length > max_edge
        if not mark.any():
            break
        uses = np.bincount(inv, minlength=len(uniq))                          # 1 (open border) or 2 triangles
        if uses[mark].sum() > budget:                                       # each split edge adds one per user
            cand = np.flatnonzero(mark)
            cand = cand[np.argsort(-length[cand])]
            keep = cand[: int(np.searchsorted(np.cumsum(uses[cand]), budget, side="right"))]
            if not len(keep):
                break
            mark = np.zeros_like(mark)
            mark[keep] = True
        mid = np.full(len(uniq), -1, dtype=np.int64)
        split = np.flatnonzero(mark)
        mid[split] = len(V) + np.arange(len(split))
        V = np.vstack([V, (V[uniq[split, 0]] + V[uniq[split, 1]]) / 2])
        if N is not None:
            nm = N[uniq[split, 0]] + N[uniq[split, 1]]
            N = np.vstack([N, nm / np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-12)])
        M = mid[inv].reshape(3, T).T                                        # midpoint on edge k, or -1
        bits = M >= 0
        k = bits.sum(1)
        out = [F[k == 0]]
        # rotate each split triangle so its pattern is canonical: one split edge -> edge 0; two -> edges 0 and 1
        first = np.argmax(bits, axis=1)
        unsplit = np.argmin(bits, axis=1)
        rot = np.where(k == 1, first, np.where(k == 2, (unsplit + 1) % 3, 0))
        cols = (rot[:, None] + np.arange(3)[None]) % 3
        Fr = np.take_along_axis(F, cols, axis=1)
        Mr = np.take_along_axis(M, cols, axis=1)
        a, b, c = Fr[:, 0], Fr[:, 1], Fr[:, 2]
        m0, m1, m2 = Mr[:, 0], Mr[:, 1], Mr[:, 2]
        s = k == 1
        out += [np.c_[a[s], m0[s], c[s]], np.c_[m0[s], b[s], c[s]]]
        s = k == 2
        out += [np.c_[m0[s], b[s], m1[s]], np.c_[a[s], m0[s], m1[s]], np.c_[a[s], m1[s], c[s]]]
        s = k == 3
        out += [np.c_[a[s], m0[s], m2[s]], np.c_[m0[s], b[s], m1[s]], np.c_[m2[s], m1[s], c[s]],
                np.c_[m0[s], m1[s], m2[s]]]
        F = np.vstack(out)
    out_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    if N is not None:
        out_mesh.vertex_normals = o3d.utility.Vector3dVector(N)
    else:
        out_mesh.compute_vertex_normals()
    return out_mesh


def _edge_samples(V: np.ndarray, sharp: np.ndarray, step: float) -> np.ndarray:
    if len(sharp) == 0:
        return np.empty((0, 3))
    A, B = V[sharp[:, 0]], V[sharp[:, 1]]
    k = np.maximum(1, np.ceil(np.linalg.norm(B - A, axis=1) / step).astype(int))
    seg = np.repeat(np.arange(len(A)), k + 1)
    t = np.concatenate([np.linspace(0, 1, n + 1) for n in k])
    return A[seg] + (B[seg] - A[seg]) * t[:, None]


# --------------------------------------------------------------------------- the check
def check_against_golden(scan, golden: o3d.geometry.TriangleMesh, params: GoldenParams | None = None,
                         log: Log = print, progress=None, debug: dict | None = None, up_axis: str = "y") -> dict:
    """Compare a scan with the golden model. Returns {"aligned", "deviation" (as compare_to_reference),
    "check_mesh" (the golden surface coloured by verdict, in the golden frame), "vertex_status",
    "vertex_region", "vertex_deviation" (per vertex of check_mesh), "compare_report", "report"}.
    up_axis: the 3D view's up axis ('y' or 'z'), for the few names that need 'top' or 'left' (golden_words)."""
    p = params or GoldenParams()
    if p.tolerance <= 0:
        raise ValueError("The tolerance must be more than 0")
    progress = progress or (lambda fraction, label=None: None)
    tol = float(p.tolerance)
    cmp = compare_to_reference(scan, golden, CompareParams(align=p.align, tolerance=tol,
                                                           max_distance=p.max_distance), log)
    ref: ReferenceSurface = cmp["surface"]
    creport = cmp["report"]
    # a pose that fits almost as well only matters when it does not map the golden model onto itself
    symmetric = False
    delta = creport.get("alignment", {}).get("rival_delta")
    if delta is not None:
        D = np.asarray(delta)
        G, _ = ref.sample(4000, seed=11)
        symmetric = float(np.percentile(np.abs(ref.query(G @ D[:3, :3].T + D[:3, 3])[0]), 95)) <= 0.5 * tol
    spacing = float(creport["spacing"])
    progress(0.72, "checking the surface")

    inc = np.flatnonzero(np.isfinite(cmp["deviation"]))
    if len(inc) > MAX_POINTS:
        inc = np.sort(np.random.default_rng(7).choice(inc, MAX_POINTS, replace=False))
    d = cmp["deviation"][inc]
    q = cmp["closest"][inc]
    tri = cmp["triangle"][inc]
    qn = ref.normals[tri]

    total_area = float(ref.areas.sum())
    lo, hi = ref.vertices.min(0), ref.vertices.max(0)
    # spots about 8 point spacings across: ~16 points each even where the points lie at random (a regular scanner
    # grid gives more), enough for a median and a spread per spot
    h = max(8.0 * spacing, np.sqrt(total_area / MAX_SPOTS))
    n_spots = int(np.clip(total_area / h ** 2, 3_000, MAX_SPOTS))
    h = float(np.sqrt(total_area / n_spots))
    spot_area = total_area / n_spots
    min_face = max(5e-5 * total_area, 1.0, 4 * spot_area)
    faces, tri_face, sharp = find_faces(ref, min_face)
    words = PartWords(ref, faces, up_axis)
    words.set_triangle_faces(tri_face)
    log(f"Golden check: {sum(f['type'] == 'plane' for f in faces)} flat faces, "
        f"{sum(f['type'] == 'cylinder' and f['hole'] for f in faces)} holes, "
        f"{sum(f['type'] == 'cylinder' and not f['hole'] for f in faces)} round faces; {n_spots:,} surface spots "
        f"{h:.3f} apart")

    # sharp edges: scanners round them, so they are left out of 'rough', 'off' and the measurements
    edge_band = float(np.clip(max(4.0 * spacing, 0.6 * h), 0.1, 1.0))
    edge_pts = _edge_samples(ref.vertices, sharp, edge_band / 2)
    edge_tree = cKDTree(edge_pts) if len(edge_pts) else None

    def near_edge(x: np.ndarray) -> np.ndarray:
        if edge_tree is None:
            return np.zeros(len(x), dtype=bool)
        return edge_tree.query(x, k=1, distance_upper_bound=edge_band, workers=-1)[0] <= edge_band

    S, Sn, Stri = ref.sample(n_spots, seed=5, triangles=True)
    spot_tree = cKDTree(S)
    spot_edge = near_edge(S)
    point_edge = near_edge(q)

    # each scan point -> the nearest spot on the same side of the surface
    spot_of = np.full(len(q), -1, dtype=np.int64)
    for i0 in range(0, len(q), 400_000):
        sl = slice(i0, i0 + 400_000)
        _, idx = spot_tree.query(q[sl], k=4, workers=-1)
        agree = np.einsum("ijk,ik->ij", Sn[idx], qn[sl]) > 0.5
        pick = idx[np.arange(len(idx)), np.argmax(agree, axis=1)]
        spot_of[sl] = np.where(agree.any(1), pick, -1)
    ok = spot_of >= 0
    med, counts = _group_median(d[ok], spot_of[ok], n_spots)
    mad, _ = _group_median(np.abs(d[ok] - med[spot_of[ok]]), spot_of[ok], n_spots)
    sig = 1.4826 * np.nan_to_num(mad)

    # neighbours: spots within 2h facing the same way
    pairs = spot_tree.query_pairs(2.0 * h, output_type="ndarray")
    pairs = pairs[np.einsum("ij,ij->i", Sn[pairs[:, 0]], Sn[pairs[:, 1]]) > 0.5]
    A = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_spots, n_spots)).tocsr()
    A = A + A.T
    W = A + sparse.identity(n_spots, format="csr")
    density = (W @ counts.astype(float)) / (W @ np.ones(n_spots))
    base = float(np.median(density[density > 0])) if np.any(density > 0) else 0.0
    ratio = density / base if base > 0 else np.zeros(n_spots)
    wc = np.where(counts >= 3, counts, 0).astype(float)
    wsum = W @ wc
    dev_s = (W @ (wc * np.nan_to_num(med))) / np.maximum(wsum, 1e-12)
    spread_s = np.sqrt((W @ (wc * sig ** 2)) / np.maximum(wsum, 1e-12))
    judged = (wsum >= 10) & (ratio >= THIN_RATIO)
    typical = float(np.median(spread_s[judged & ~spot_edge])) if np.any(judged & ~spot_edge) else 0.0
    rough_limit = max(tol, 2.5 * typical)

    status = np.full(n_spots, GOOD, dtype=np.int64)
    status[ratio < THIN_RATIO] = THIN
    status[ratio < MISSING_RATIO] = MISSING
    free = judged & ~spot_edge
    status[free & (spread_s > rough_limit)] = ROUGH
    status[free & (dev_s > tol)] = OFF_OUT
    status[free & (dev_s < -tol)] = OFF_IN
    for _ in range(2):   # the thinly scanned rim around a hole in the scan belongs to that hole
        status[(status == THIN) & ((A @ (status == MISSING).astype(float)) > 0)] = MISSING
    votes = np.column_stack([W @ (status == k).astype(float) for k in range(6)])
    total = votes.sum(1)
    own = votes[np.arange(n_spots), status] / total
    flip = (own < 0.3) & (votes.max(1) / total >= 0.6)
    status[flip] = np.argmax(votes, axis=1)[flip]
    edge_dev = np.abs(dev_s[spot_edge & judged])
    if debug is not None:   # the per-spot numbers, for tests
        debug.update(spots=S, status=status, dev=dev_s, spread=spread_s, ratio=ratio, counts=counts, sigma=sig,
                     edge=spot_edge, rough_limit=rough_limit, typical=typical)
    progress(0.8, "naming the regions")

    # regions
    min_region = max(5 * spot_area, 5e-4 * total_area, 0.5)
    spot_face = tri_face[Stri]
    face_spots = np.bincount(spot_face[spot_face >= 0], minlength=len(faces))
    covered = np.bincount(spot_face[(spot_face >= 0) & (status != MISSING)], minlength=len(faces)) \
        / np.maximum(face_spots, 1)
    regions: list[dict] = []
    spot_region = np.full(n_spots, -1, dtype=np.int64)
    for code in (MISSING, THIN, ROUGH, OFF_OUT, OFF_IN):
        members = np.flatnonzero(status == code)
        if len(members) == 0:
            continue
        comp = connected_components(A[members][:, members], directed=False)[1]
        for c in np.unique(comp):
            idx = members[comp == c]
            area = len(idx) * spot_area
            if area < min_region:
                continue
            region = _describe_region(code, idx, S, Sn, spot_face, face_spots, faces, words, area,
                                      total_area, dev_s, spread_s, ratio, typical, tol, covered)
            region["_index"] = len(regions)
            spot_region[idx] = len(regions)
            regions.append(region)
    regions.sort(key=lambda r: ({"missing": 0, "thin": 1, "rough": 2, "off": 3}[r["kind"]], -r["area_mm2"]))
    lut = np.full(len(regions) + 1, -1, dtype=np.int64)
    for i, r in enumerate(regions):
        lut[r.pop("_index")] = i
        r["id"] = i
    spot_region = lut[spot_region]          # -1 stays -1 (the last entry of lut)
    # the numbers the list and the pins show: areas that differ first (they decide the verdict), the biggest
    # difference over the biggest area first, then what to scan again (frontend GoldenCheck, `ordered`)
    off_first = sorted((r for r in regions if r["kind"] == "off"), key=lambda r: -abs(r["deviation"]) * r["area_mm2"])
    for n, r in enumerate(off_first + [r for r in regions if r["rescan"]], 1):
        r["number"] = n
    progress(0.86, "measuring")

    # measurements
    pts = np.asarray(cmp["aligned"].points)[inc]
    measurements = _measure(faces, words, tri_face, tri[ok], d[ok], q[ok], qn[ok], pts[ok], point_edge[ok], status,
                            spot_face, face_spots, spot_region, regions, lo, hi, tol, h, ref)
    on_face = _regions_on_faces(regions, spot_face, spot_region, face_spots, len(faces))
    for m in measurements:
        _annotate(m, regions, words, on_face, covered)
    progress(0.93, "writing the report")

    # the golden surface coloured by verdict, for the viewer: split long CAD edges first so every listed area shows
    mesh = refine_long_edges(cmp["reference"], max(1.5 * h, 0.05))
    Vd, Nd = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    _, idx = spot_tree.query(Vd, k=6, workers=-1)
    agree = np.einsum("ijk,ik->ij", Sn[idx], Nd) > 0.3
    pick = np.where(agree.any(1), idx[np.arange(len(idx)), np.argmax(agree, axis=1)], idx[:, 0])
    v_status = status[pick]
    v_region = spot_region[pick]
    v_dev = np.where(judged, dev_s, np.nan)[pick]
    v_face = tri_face[ref.closest(Vd)[3]]           # the golden face each point lies on (its closest triangle's)
    palette = np.array([[int(e["color"][i:i + 2], 16) / 255 for i in (1, 3, 5)] for e in LEGEND])
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.vertex_colors = o3d.utility.Vector3dVector(palette[v_status])

    share = {LEGEND[k]["key"]: round(100.0 * float((status == k).mean()), 2) for k in range(6)}
    surface = {"scanned_pct": round(100.0 - share["missing"], 2), "shares_pct": share,
               "noise": round(typical, 4), "rough_limit": round(rough_limit, 4),
               "edges": {"band": round(edge_band, 3),
                         "rounded_by": round(float(np.percentile(edge_dev, 90)), 4) if len(edge_dev) else None},
               "spot_spacing": round(h, 4), "area_mm2": round(total_area, 2)}
    report = _summarise(regions, measurements, surface, creport, tol, faces, symmetric)
    used = sorted({f for m in measurements for f in (m.get("faces") or []) if f >= 0})
    try:
        section = cut_section(ref, words, tri_face, used)
    except Exception as exc:        # noqa: BLE001 - the drawing is an extra: never fail the check for it
        log(f"  no section drawing: {exc}")
        section = None
    report.update(golden_size=np.round(hi - lo, 4).tolist(), golden_min=np.round(lo, 4).tolist(),
                  golden_max=np.round(hi, 4).tolist(),
                  sizes_story=_sizes_story(measurements, regions, words, on_face), section=section,
                  part={"axis": words.axis, "ends": list(words.ends) if words.ends else None, "up_axis": up_axis,
                        "summary": words.summary(), "kind": words.kind})
    for line in [report["headline"], *report["summary"]]:
        log("  " + line)
    return {"aligned": cmp["aligned"], "deviation": cmp["deviation"], "check_mesh": mesh,
            "vertex_status": v_status.astype(np.float64), "vertex_region": v_region.astype(np.float64),
            "vertex_deviation": v_dev, "vertex_face": v_face.astype(np.float64), "compare_report": creport,
            "report": report}


def _amount(x: float) -> str:
    """A distance in an area's text: 2.41, 0.080."""
    return f"{x:.2f}" if x >= 1 else f"{x:.3f}"


def _describe_region(code, idx, S, Sn, spot_face, face_spots, faces, words: PartWords, area, total_area, dev_s,
                     spread_s, ratio, typical, tol, covered) -> dict:
    P, N = S[idx], Sn[idx]
    center = P.mean(0)
    anchor = P[np.argmin(np.linalg.norm(P - center, axis=1))]
    mean_n = N.mean(0)
    normal = _unit(mean_n) if np.linalg.norm(mean_n) > 1e-6 else _unit(Sn[idx[0]])
    fids = spot_face[idx]
    fids = fids[fids >= 0]
    face_id, face = None, None
    if len(fids):
        best = int(np.bincount(fids).argmax())
        if (fids == best).sum() >= 0.5 * len(idx):
            face_id, face = best, faces[best]
    kind = KIND[code]
    part = (fids == face_id).sum() / max(1, face_spots[face_id]) if face is not None else 1.0
    named = words.region(face_id, part, center, mean_n, anchor, spots=(P, N, spot_face[idx]))
    name = named["name"]
    fkind = named["face_kind"]
    shade = named["shade"]
    rec = named["recess"]

    dev = float(np.mean(dev_s[idx]))
    spread = float(np.median(spread_s[idx]))
    dens = float(np.median(ratio[idx]))
    hole = face is not None and face["type"] == "cylinder" and face["hole"]
    end_sign = words.is_end_like(normal, center)
    if kind == "missing":
        why = "The scan has no points here: the scanner never saw it."
        if hole:
            how = ("Point the scanner straight into the hole (tilt the part so the hole faces the scanner) and scan "
                   "it again.")
            if face["length"] > 2 * face["radius"]:
                how += (" Holes deeper than they are wide are hard for any scanner to see into: if it stays empty, "
                        "check the hole with a gauge or caliper instead.")
        elif rec is not None or fkind in ("floor", "wall", "pocket"):
            opening = rec["noun"] if rec is not None else "pocket" if fkind == "pocket" else "recess"
            opening = "hole" if opening.startswith("Ø") else opening
            how = (f"Point the scanner straight into the {opening} (tilt the part so its opening faces the scanner) "
                   "and scan it again. Deep, narrow openings are hard for any scanner to see into: if it stays empty, "
                   "check the depth with a depth gauge instead.")
        elif shade == "the head":
            how = ("A tight corner like this is hard to see into. Tilt the part so the scanner looks up under the "
                   "head, straight at the corner, and scan it again.")
        elif shade is not None:
            how = (f"{shade[:1].upper()}{shade[1:]} shades this corner. Tilt the part so the scanner looks straight "
                   "into the corner, and scan it again.")
        elif end_sign and words.axis is not None and words.near_end(P, end_sign):
            end_face = words.end_face(end_sign)
            fid = words.end_face_id(end_sign)
            if fid is None or covered[fid] < 0.5:
                how = (f"The scanner did not see the {end_face}: the part probably stood on it. Turn the part over so "
                       f"the {end_face} faces the scanner, scan it again and merge the two scans (Align step).")
            else:
                how = ("The scanner missed this edge. Tilt the part so the edge faces the scanner, scan it again and "
                       "merge the two scans (Align step).")
        else:
            how = ("Turn or tilt the part so this side faces the scanner, and scan it again. If the part stood on "
                   "this side, turn it over, scan it again and merge the two scans (Align step).")
    elif kind == "thin":
        why = (f"Only {dens * 100:.0f} % of the usual number of points landed here: the scanner saw it at a steep "
               "angle.")
        how = "Scan it again with the scanner facing it straight on."
    elif kind == "rough":
        why = f"The points scatter by ±{spread:.3f} mm here, against ±{typical:.3f} mm on the rest of the scan."
        how = ("Shiny, dark or see-through surfaces do this. Give it a light coat of scanning spray (or lower the "
               "exposure) and scan it again.")
    else:
        x = _amount(abs(dev))
        if fkind in ("floor", "pocket"):
            why = (f"The scan finds this bottom {x} mm {'deeper' if dev < 0 else 'shallower'} than the golden model. "
                   f"The tolerance is ±{tol:g} mm.")
        elif hole:
            why = (f"The hole's wall sits {x} mm {'farther in' if dev > 0 else 'farther out'} than designed: the hole "
                   f"is {'smaller' if dev > 0 else 'bigger'} here. The tolerance is ±{tol:g} mm.")
        else:
            why = (f"Here the scan sits {x} mm {'outside' if dev > 0 else 'inside'} the golden model's surface: there "
                   f"is {'more' if dev > 0 else 'less'} material than designed. The tolerance is ±{tol:g} mm.")
        how = ("Measure it with a caliper or gauge. If the part is right there, this area probably came from a "
               "separate scan that was merged slightly off: redo that line-up in the Align step.")
        if fkind == "floor" and dev < 0:
            how = ("Drilled holes and sockets usually end in a cone (the drill point) where the drawing shows a flat "
                   "bottom, so the scan reads deeper here. Check the depth with a depth gauge before you decide the "
                   "part is wrong.")
    size = float(np.linalg.norm(P.max(0) - P.min(0)))
    prefer = None
    if hole:
        axis = face["axis"] if face["axis"] @ (anchor - words.mid) >= 0 else -face["axis"]
        prefer = _unit(axis + 0.35 * normal)
    return {"kind": kind, "sign": (1 if code == OFF_OUT else -1) if kind == "off" else 0, "name": name,
            "where": named["where"], "why": why, "advice": how, "rescan": kind in ("missing", "thin", "rough"),
            "face_kind": fkind,
            "area_mm2": round(area, 2), "share_pct": round(100.0 * area / total_area, 2),
            "center": np.round(center, 4).tolist(), "normal": np.round(normal, 4).tolist(),
            "pin": np.round(anchor, 4).tolist(), "box": [np.round(P.min(0), 4).tolist(), np.round(P.max(0), 4).tolist()],
            "size": round(size, 3), "face": face_id,
            "deviation": round(dev, 4) if kind == "off" else None,
            "spread": round(spread, 4) if kind == "rough" else None,
            "density_pct": round(dens * 100, 1) if kind in ("thin", "missing") else None,
            "view": words.view_for(anchor, normal, size, prefer)}


def _measure(faces, words: PartWords, tri_face, tri, d, q, qn, pts, point_edge, status, spot_face, face_spots,
             spot_region, regions, lo, hi, tol, h, ref) -> list[dict]:
    """Golden value vs scanned value for the overall size, face-to-face distances, diameters, hole positions."""
    out: list[dict] = []
    nf = len(faces)
    pf = tri_face[tri]
    use = (pf >= 0) & ~point_edge
    med, cnt = _group_median(d[use], pf[use], nf)
    absdev, _ = _group_median(np.abs(d[use] - med[pf[use]]), pf[use], nf)
    sigma = 1.4826 * np.nan_to_num(absdev)
    scanned = np.bincount(spot_face[(spot_face >= 0) & (status != MISSING)], minlength=nf)
    covered = scanned / np.maximum(face_spots, 1)

    def face_ok(i: int) -> bool:
        return cnt[i] >= 30 and covered[i] >= 0.25

    def region_for(i: int) -> int | None:
        hits = spot_region[(spot_face == i) & (spot_region >= 0)]
        hits = hits[[regions[k]["rescan"] for k in hits]] if len(hits) else hits
        return int(np.bincount(hits).argmax()) if len(hits) else None

    def not_measured(entry: dict, missing_faces: list[int], reason: str) -> dict:
        region = next((region_for(i) for i in missing_faces if region_for(i) is not None), None)
        return {**entry, "scan": None, "difference": None, "uncertainty": None, "status": "not_measured",
                "reason": reason, "region": region}

    def result(entry: dict, golden: float, scan: float, u: float) -> dict:
        u = max(u, 0.001)
        return {**entry, "golden": round(golden, 4), "scan": round(scan, 4), "difference": round(scan - golden, 4),
                "uncertainty": round(u, 4), "status": _judge(scan - golden, u, tol)}

    # overall size along X, Y, Z: the golden model's extreme surfaces, measured where the scan has them
    extreme_pairs = set()
    band = max(2.0 * h, 0.01 * float(np.max(hi - lo)))
    for k in range(3):
        golden = float(hi[k] - lo[k])
        entry = {"kind": "size", **words.size_words(k), "golden": round(golden, 4), "axis": k,
                 "at": ((lo + hi) / 2).round(4).tolist()}
        end_faces = {1: [], -1: []}
        for i, f in enumerate(faces):
            s_ = _side(f["normal"], 1.0) if f["type"] == "plane" else None
            if s_ and s_[0] == k and abs(f["point"][k] - (hi[k] if s_[1] > 0 else lo[k])) < 1e-3 * max(golden, 1.0):
                end_faces[s_[1]].append(i)
        entry["faces"] = end_faces[-1] + end_faces[1]
        extreme_pairs.update((i, j) for i in end_faces[1] for j in end_faces[-1])
        extreme_pairs.update((j, i) for i in end_faces[1] for j in end_faces[-1])
        ends, missing = [], []
        for sign, level in ((1, hi[k]), (-1, lo[k])):
            sel = (np.abs(q[:, k] - level) <= band) & (sign * qn[:, k] > 0.7) & ~point_edge
            if sel.sum() < 20:
                missing.append(sign)
                continue
            shift = d[sel] * np.abs(qn[sel, k])     # how far the scan's surface sits beyond the golden end
            ends.append((float(np.median(shift)), _se_median(_robust_sigma(shift), int(sel.sum()))))
        if missing:
            ends_missing = " and the ".join(words.end_word(k, sg) for sg in missing)
            m = not_measured(entry, [i for sg in missing for i in end_faces[sg]],
                             f"The {ends_missing} {'were' if len(missing) > 1 else 'was'} not scanned, so this size "
                             "cannot be measured.")
            if m["region"] is None:
                m["region"] = _region_at_end(k, missing, regions, lo, hi)
            out.append(m)
            continue
        scan = golden + ends[0][0] + ends[1][0]
        out.append(result(entry, golden, scan, 2 * float(np.hypot(ends[0][1], ends[1][1]))))

    # flat faces: thickness / gap between opposite faces, steps between faces facing the same way
    planes = [i for i, f in enumerate(faces) if f["type"] == "plane"]

    corners = {i: ref.vertices[np.unique(ref.triangles[faces[i]["tris"]])] for i in planes}

    def footprint(i: int, n: np.ndarray):
        u, v = _plane_frame(n)
        uv = np.c_[corners[i] @ u, corners[i] @ v]
        return uv.min(0), uv.max(0)

    pairs_opp, pairs_same = {}, []
    for i in planes:
        best = None
        ni = faces[i]["normal"]
        for j in planes:
            if j == i or (i, j) in extreme_pairs:
                continue
            nj = faces[j]["normal"]
            if ni @ nj < -np.cos(np.radians(1.0)):
                g = faces[i]["offset"] + faces[j]["offset"]
                if abs(g) < 1e-6:
                    continue
                (a0, a1), (b0, b1) = footprint(i, ni), footprint(j, ni)
                if np.any(np.minimum(a1, b1) - np.maximum(a0, b0) <= 0):
                    continue
                if best is None or abs(g) < abs(best[1]):
                    best = (j, g)
        if best is not None:
            key = tuple(sorted((i, best[0])))
            pairs_opp[key] = best[1]
    groups: list[list[int]] = []
    for i in planes:
        for g in groups:
            if faces[g[0]]["normal"] @ faces[i]["normal"] > np.cos(np.radians(1.0)):
                g.append(i)
                break
        else:
            groups.append([i])
    for g in groups:
        g.sort(key=lambda i: faces[i]["offset"])
        for i, j in zip(g, g[1:]):
            if abs(faces[j]["offset"] - faces[i]["offset"]) > 0.01:
                pairs_same.append((i, j))

    def pair_entry(kind: str, i: int, j: int, golden: float) -> dict:
        at = (faces[i]["point"] + faces[j]["point"]) / 2
        return {"kind": kind, **words.pair_words(kind, i, j), "golden": round(golden, 4),
                "faces": [i, j], "at": np.round(at, 4).tolist()}

    def too_little(missing: list[int]) -> str:
        return f"Too little of the {words.faces_phrase(missing)} was scanned to measure this."

    face_pairs = sorted(pairs_opp.items(), key=lambda kv: -min(faces[kv[0][0]]["area"], faces[kv[0][1]]["area"]))[:12]
    for (i, j), g in face_pairs:
        kind = "thickness" if g > 0 else "gap"
        golden = abs(g)
        entry = pair_entry(kind, i, j, golden)
        missing = [f for f in (i, j) if not face_ok(f)]
        if missing:
            out.append(not_measured(entry, missing, too_little(missing)))
            continue
        # each face moved outward by its median deviation: a thickness grows by both, a gap shrinks by both
        scan = golden + med[i] + med[j] if g > 0 else golden - med[i] - med[j]
        u = 2 * float(np.hypot(_se_median(sigma[i], cnt[i]), _se_median(sigma[j], cnt[j])))
        out.append(result(entry, golden, scan, u))
    for i, j in sorted(pairs_same, key=lambda p: -min(faces[p[0]]["area"], faces[p[1]]["area"]))[:10]:
        golden = faces[j]["offset"] - faces[i]["offset"]
        entry = pair_entry("step", j, i, golden)
        missing = [f for f in (i, j) if not face_ok(f)]
        if missing:
            out.append(not_measured(entry, missing, too_little(missing)))
            continue
        scan = golden + med[j] - med[i]
        u = 2 * float(np.hypot(_se_median(sigma[i], cnt[i]), _se_median(sigma[j], cnt[j])))
        out.append(result(entry, golden, scan, u))

    # holes and shafts: diameter and position
    cyls = sorted((i for i, f in enumerate(faces) if f["type"] == "cylinder"), key=lambda i: -faces[i]["area"])[:15]
    for i in cyls:
        f = faces[i]
        u_ax, v_ax = _plane_frame(f["axis"])
        golden_d = 2 * f["radius"]
        what = words.labels[i].split(" (")[0]
        base = {"faces": [i], "at": np.round(f["center"], 4).tolist()}
        e_d = {**base, "kind": "diameter", **words.diameter_words(i), "golden": round(golden_d, 4)}
        e_p = {**base, "kind": "position", **words.position_words(i), "golden": 0.0}
        sel = (pf == i) & ~point_edge
        if sel.sum() < 30:
            reason = f"Too little of the {what} was scanned to measure it."
            out.append(not_measured(e_d, [i], reason))
            out.append(not_measured(e_p, [i], reason))
            continue
        xy = np.c_[pts[sel] @ u_ax, pts[sel] @ v_ax]    # the scan points, across the golden axis
        c_g = np.array([f["center"] @ u_ax, f["center"] @ v_ax])
        arc = _arc_deg(np.degrees(np.arctan2(xy[:, 1] - c_g[1], xy[:, 0] - c_g[0])))
        if arc < 120.0:
            reason = f"Only {arc:.0f}° of the way round the {what} was scanned; at least 120° is needed."
            out.append(not_measured(e_d, [i], reason))
            out.append(not_measured(e_p, [i], reason))
            continue
        keep = np.ones(len(xy), dtype=bool)
        for _ in range(3):
            c2, r, rms, cov = _fit_circle(xy[keep])
            res = np.hypot(xy[:, 0] - c2[0], xy[:, 1] - c2[1]) - r
            s = max(_robust_sigma(res[keep]), 1e-6)
            new = np.abs(res) <= 3 * s
            if new.sum() < 20 or np.array_equal(new, keep):
                break
            keep = new
        n_used = int(keep.sum())
        inflate = np.sqrt(n_used / min(n_used, N_EFF))
        se = np.sqrt(np.clip(np.diag(cov), 0, None)) * s * inflate
        out.append(result(e_d, golden_d, 2 * r, 2 * 2 * float(se[2])))
        shift = c2 - c_g
        off = float(np.linalg.norm(shift))
        direction = shift[0] * u_ax + shift[1] * v_ax
        e_p["offset"] = np.round(direction, 4).tolist()
        e_p["toward"] = words.direction(direction) if off > 1e-6 else None
        out.append(result(e_p, 0.0, off, 2 * float(np.hypot(se[0], se[1]))))
    # several sizes of one feature (the three widths across a hex socket): one series, shown as one row
    series: dict[str, list[dict]] = {}
    for m in out:
        key = m.pop("series", None)
        if key:
            series.setdefault(key, []).append(m)
    for key, ms in series.items():
        if len(ms) > 1:
            for n, m in enumerate(ms, 1):
                m.update(series=key, series_index=n, series_size=len(ms))
    # any other name twice: number them
    seen: dict[str, list[dict]] = {}
    for m in out:
        if "series" not in m:
            seen.setdefault(m["name"], []).append(m)
    for name, ms in seen.items():
        if len(ms) > 1:
            for n, m in enumerate(ms, 1):
                m["name"] = f"{name} ({n} of {len(ms)})"
    # the dimension line of each size, on the golden surfaces it is measured between
    for k, m in enumerate(out):
        m["ends"] = _dimension_line(m, faces, words, ref)
        m["id"] = k
    return out


# --------------------------------------------------------------------------- dimension lines
def _section_axes(words: PartWords) -> tuple[np.ndarray, np.ndarray]:
    """The drawing's x and y directions: along the main axis and the 3D view's up (its right when the axis is up);
    with no main axis, the view's right and up."""
    up, right = words.up_index(), 0
    if words.axis is None:
        ku, kv = right, up
    else:
        ku = words.axis
        kv = up if ku != up else right
    return np.eye(3)[ku], np.eye(3)[kv]


def _in_drawing(words: PartWords, P: np.ndarray) -> np.ndarray:
    """Points close to the drawing's plane (dimension lines there show on the drawing as well as on the model)."""
    u, v = _section_axes(words)
    return np.abs((P - words.mid) @ np.cross(u, v)) <= max(0.02 * words.diag, 3 * words.eps)


def _face_samples(ref: ReferenceSurface, f: dict, n: int, seed: int = 0) -> np.ndarray:
    """Area-uniform points on one golden face."""
    t = f["tris"]
    w = ref.areas[t]
    rng = np.random.default_rng(seed)
    pick = rng.choice(t, size=n, p=w / w.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = (ref.vertices[ref.triangles[pick, k]] for k in range(3))
    return a * (1 - s)[:, None] + b * (s * (1 - r2))[:, None] + c * (s * r2)[:, None]


def _dimension_line(m: dict, faces, words: PartWords, ref: ReferenceSurface) -> list | None:
    """Two golden-frame points on the surfaces a size is measured between (None for positions)."""
    try:
        if m["kind"] == "size":
            ends = _size_ends(words, m["axis"])
        elif m["kind"] in ("thickness", "gap", "step"):
            ends = _plane_ends(words, ref, faces, *m["faces"])
        elif m["kind"] == "diameter":
            ends = _circle_ends(words, ref, faces[m["faces"][0]])
        else:
            return None
    except (ValueError, IndexError):
        return None
    return None if ends is None else [np.round(np.asarray(e, dtype=np.float64), 4).tolist() for e in ends]


def _size_ends(words: PartWords, k: int):
    """The overall size along axis k: a line along k between the extreme surfaces, through the middle of where the
    part reaches both (rays cast from both sides on a grid across the part); when no straight line reaches both (the
    flat top of a head only outside a chamfered socket, the tip smaller), the two nearest points on them."""
    lo, hi = words.lo, words.hi
    o2 = [j for j in range(3) if j != k]
    n = 61
    A, B = np.meshgrid(np.linspace(lo[o2[0]], hi[o2[0]], n), np.linspace(lo[o2[1]], hi[o2[1]], n), indexing="ij")
    P = np.zeros((n * n, 3))
    P[:, o2[0]], P[:, o2[1]] = A.ravel(), B.ravel()
    e = np.eye(3)[k]
    margin = 1.0 + words.eps
    top, bottom = P.copy(), P.copy()
    top[:, k], bottom[:, k] = hi[k] + margin, lo[k] - margin
    t_hi, _ = words.cast(top, np.tile(-e, (len(P), 1)))
    t_lo, _ = words.cast(bottom, np.tile(e, (len(P), 1)))
    x_hi, x_lo = hi[k] + margin - t_hi, lo[k] - margin + t_lo
    tol = max(2e-3 * (hi[k] - lo[k]), 0.02)
    ext = np.where(np.isfinite(t_hi) & np.isfinite(t_lo), x_hi - x_lo, -np.inf)
    cand = np.flatnonzero(ext >= hi[k] - lo[k] - tol)
    flat = _in_drawing(words, P)
    if (flat[cand]).any():
        cand = cand[flat[cand]]
    if len(cand):
        c = P[cand][:, o2].mean(0)
        pick = cand[np.argmin(np.linalg.norm(P[cand][:, o2] - c, axis=1))]
        p0, p1 = P[pick].copy(), P[pick].copy()
        p0[k], p1[k] = x_lo[pick], x_hi[pick]
        return [p0, p1]
    top = np.flatnonzero(np.isfinite(t_hi) & (x_hi >= hi[k] - tol))
    bottom = np.flatnonzero(np.isfinite(t_lo) & (x_lo <= lo[k] + tol))
    if not len(top) or not len(bottom):
        return None
    if flat[top].any() and flat[bottom].any():
        top, bottom = top[flat[top]], bottom[flat[bottom]]
    fine = max(1e-4 * (hi[k] - lo[k]), 0.005)                 # the extreme surfaces themselves, not a chamfer
    top, bottom = top[x_hi[top] >= x_hi[top].max() - fine], bottom[x_lo[bottom] <= x_lo[bottom].min() + fine]
    dist, near = cKDTree(P[bottom][:, o2]).query(P[top][:, o2])
    a = int(np.argmin(dist))
    p0, p1 = P[bottom[near[a]]].copy(), P[top[a]].copy()
    p0[k], p1[k] = x_lo[bottom[near[a]]], x_hi[top[a]]
    return [p0, p1]


def _plane_ends(words: PartWords, ref: ReferenceSurface, faces, i: int, j: int):
    """Between two flat faces: a point inside the part of face i that lies over face j and its projection onto j;
    when they do not overlap, the two closest points, one on each face."""
    fi, fj = faces[i], faces[j]
    if fi["type"] != "plane" or fj["type"] != "plane":
        return None
    Pi, Pj = _face_samples(ref, fi, 400, seed=i), _face_samples(ref, fj, 400, seed=j)
    nj = fj["normal"]
    Q = Pi - ((Pi - fj["point"]) @ nj)[:, None] * nj
    on = np.abs(ref.query(Q)[0]) <= max(1e-3, 1e-5 * words.diag)
    if on.any():
        idx = np.flatnonzero(on)
        flat = _in_drawing(words, Pi[idx])
        if flat.any():
            idx = idx[flat]
        pick = idx[np.argmin(np.linalg.norm(Pi[idx] - Pi[idx].mean(0), axis=1))]
        return [Pi[pick], Q[pick]]
    fi_in, fj_in = _in_drawing(words, Pi), _in_drawing(words, Pj)
    if fi_in.any() and fj_in.any():
        Pi, Pj = Pi[fi_in], Pj[fj_in]
    u, v = _plane_frame(fi["normal"])
    dist, near = cKDTree(np.c_[Pj @ u, Pj @ v]).query(np.c_[Pi @ u, Pi @ v])
    a = int(np.argmin(dist))
    return [Pi[a], Pj[near[a]]]


def _circle_ends(words: PartWords, ref: ReferenceSurface, f: dict):
    """Across a round face at its middle: two opposite points on it, the line upright in the default view when it
    can be (the drawing's cut contains that direction), snapped onto the golden mesh."""
    a = _unit(f["axis"])
    cands = []
    for e in (*_section_axes(words)[::-1], np.eye(3)[words.up_index()], np.eye(3)[0]):
        d = e - (e @ a) * a
        if np.linalg.norm(d) > 0.3:
            cands.append(_unit(d))
    u, v = _plane_frame(a)
    cands += [np.cos(t) * u + np.sin(t) * v for t in np.radians(np.arange(0.0, 180.0, 15.0))]
    r = f["radius"]
    tol = max(0.02, 0.02 * r)
    for shift in (0.0, 0.25, -0.25):
        c = f["center"] + a * shift * f["length"]
        for d in cands:
            dist, near, _ = ref.query(np.array([c + r * d, c - r * d]))
            if np.abs(dist).max() <= tol:
                return [near[0], near[1]]
    return [f["center"] + r * cands[0], f["center"] - r * cands[0]]


# --------------------------------------------------------------------------- what to say about each size
def _regions_on_faces(regions, spot_face, spot_region, face_spots, nf: int) -> dict[int, list[int]]:
    """{face id: the listed areas on it}: at least 3 of its spots, and 5 % of the face or 30 % of the area."""
    nr = len(regions)
    if nr == 0:
        return {}
    sel = (spot_face >= 0) & (spot_region >= 0)
    count = np.zeros((nr, nf), dtype=np.int64)
    np.add.at(count, (spot_region[sel], spot_face[sel]), 1)
    total = np.bincount(spot_region[spot_region >= 0], minlength=nr)
    out: dict[int, list[int]] = {}
    for r, f in zip(*np.nonzero(count >= 3)):
        if count[r, f] >= 0.05 * face_spots[f] or count[r, f] >= 0.3 * total[r]:
            out.setdefault(int(f), []).append(int(r))
    return out


def _annotate(m: dict, regions: list[dict], words: PartWords, on_face: dict, covered: np.ndarray) -> None:
    """A size's listed areas (on its faces) and, when it deserves care, why (caveat)."""
    faces = [f for f in (m.get("faces") or []) if f >= 0]
    found = {r for f in faces for r in on_face.get(f, [])}
    m["regions"] = sorted(found, key=lambda r: regions[r]["number"])
    m["caveat"] = None
    if m["status"] == "not_measured":
        return
    label = {f: words.labels[f].split(" (")[0] for f in faces}
    pairs = [(f, regions[r]) for f in faces for r in on_face.get(f, [])]
    off = [(f, r) for f, r in pairs if r["kind"] == "off"]
    if off:
        f, r = max(off, key=lambda fr: abs(fr[1]["deviation"]) * fr[1]["area_mm2"])
        if m["status"] in ("off", "close"):
            m["caveat"] = (f"The scan finds {words.scan_finds(f, r['deviation'], _amount(abs(r['deviation'])))} "
                           f"(area {r['number']}).")
        else:
            m["caveat"] = f"The {label[f]} itself is off (area {r['number']}), but this size still matches."
        return
    rescan = [(f, r) for f, r in pairs if r["rescan"]]
    if rescan:
        f, r = max(rescan, key=lambda fr: fr[1]["area_mm2"])
        m["caveat"] = {
            "missing": f"Part of the {label[f]} was not scanned (area {r['number']}), so this size comes from the "
                       "rest of it.",
            "thin": f"Part of the {label[f]} has too few points (area {r['number']}), so this size leans on the "
                    "rest of it.",
            "rough": f"Part of the {label[f]} is rough in the scan (area {r['number']}), which makes this size "
                     "less certain."}[r["kind"]]
        return
    thin = [f for f in faces if covered[f] < 0.5]
    if thin:
        f = min(thin, key=lambda f: covered[f])
        m["caveat"] = (f"Only about {covered[f] * 100:.0f} % of the {label[f]} was scanned, so this size rests on "
                       "few points.")


def _sizes_story(measurements: list[dict], regions: list[dict], words: PartWords, on_face: dict) -> str | None:
    """1-3 sentences when the sizes that are off (or too close to call) share a cause: one face they are all
    measured from."""
    bad = [m for m in measurements if m["status"] in ("off", "close")]
    if len(bad) < 2 or any(not m.get("faces") for m in bad):
        return None
    common = set(bad[0]["faces"]).intersection(*(set(m["faces"]) for m in bad[1:]))
    if not common:
        return None

    def worst_off(f):
        off = [regions[r] for r in on_face.get(f, []) if regions[r]["kind"] == "off"]
        return max(off, key=lambda r: abs(r["deviation"]) * r["area_mm2"]) if off else None

    with_off = [f for f in common if worst_off(f) is not None]
    if with_off:
        f = max(with_off, key=lambda f: abs(worst_off(f)["deviation"]) * worst_off(f)["area_mm2"])
    else:
        f = min(common)
    label = words.labels[f].split(" (")[0]
    if all(m["status"] == "off" for m in bad):
        which = "sizes that are off"
    else:
        which = "sizes that are off or too close to call"
    count = "Both" if len(bad) == 2 else f"All {len(bad)}"
    lines = [f"{count} {which} are measured from the {label}."]
    r = worst_off(f)
    if r is None:
        lines.append(f"Check {words.pronoun(f)} first: measure it by hand and look at its colours on the model.")
        return " ".join(lines)
    amount = f"about {abs(r['deviation']):.1f}"
    lines.append(f"The scan finds {words.scan_finds(f, r['deviation'], amount, that=True)} (area {r['number']}).")
    rec = words.recess_of(f)
    if words.info[f]["kind"] == "floor" and r["deviation"] < 0:
        if rec is not None and rec["shape"] in ("hex", "square"):
            what = "sockets"
        elif rec is not None and rec["shape"] == "round":
            what = "holes"
        else:
            what = "holes and sockets"
        lines.append(f"Drilled {what} usually end in a cone that the drawing shows flat, so check the depth with a "
                     "depth gauge before deciding the part is wrong.")
    else:
        lines.append(f"Measure {words.pronoun(f)} by hand: if it is right, it probably came from a separate scan that "
                     "was merged slightly off (redo that line-up in the Align step).")
    return " ".join(lines)


# --------------------------------------------------------------------------- the drawing
def _simplify(P: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on an open polyline: the indices of the points kept."""
    n = len(P)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[[0, n - 1]] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, ab = P[i], P[j] - P[i]
        L2 = float(ab @ ab)
        Q = P[i + 1:j] - a
        t = np.clip(Q @ ab / L2, 0.0, 1.0) if L2 > 1e-24 else np.zeros(len(Q))
        dist = np.linalg.norm(Q - t[:, None] * ab, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            mid = i + 1 + k
            keep[mid] = True
            stack += [(i, mid), (mid, j)]
    return np.flatnonzero(keep)


def _simplify_loop(P: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on a closed loop (split at its first point and the point farthest from it)."""
    n = len(P)
    if n < 4:
        return np.arange(n)
    far = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
    if far == 0:
        return np.array([0])
    a = _simplify(P[:far + 1], tol)
    b = _simplify(np.vstack([P[far:], P[:1]]), tol) + far
    return np.r_[a, b[1:-1]]


def cut_section(ref: ReferenceSurface, words: PartWords, tri_face: np.ndarray, face_ids) -> dict | None:
    """A cut through the golden model for the drawing in Sizes. Along the main axis: u = the axis (+ towards its +
    end), v = the 3D view's up direction (its right when the axis is up); with no main axis, square to the view's
    depth: u = right, v = up. Through the middle of the bounding box. Returns origin, u, v, the closed outlines of the
    cut (simplified to 0.0015 x the diagonal, 0.01 mm), their bounds and, per face in face_ids, where it meets the
    cut ([x1, y1, x2, y2] segments). None when the golden mesh is not closed or the cut does not form loops."""
    if not ref.watertight:
        return None
    V, F = ref.vertices, ref.triangles
    u, v = _section_axes(words)
    o = words.mid
    dist = (V - o) @ np.cross(u, v)
    side = dist >= 0
    T, nV = len(F), len(V)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])        # edge e of triangle t at row e * T + t
    rows = np.flatnonzero(side[E[:, 0]] != side[E[:, 1]])
    if len(rows) < 3:
        return None
    Es = np.sort(E[rows], axis=1)
    uniq, node = np.unique(Es[:, 0] * nV + Es[:, 1], return_inverse=True)
    node = node.ravel()
    a, b = uniq // nV, uniq % nV
    t = dist[a] / (dist[a] - dist[b])                                     # each crossed edge's point, once
    P = V[a] + t[:, None] * (V[b] - V[a])
    xy = np.c_[(P - o) @ u, (P - o) @ v]
    tri = rows % T
    order = np.argsort(tri, kind="stable")
    tri_s, node_s = tri[order], node[order]
    if len(tri_s) % 2 or np.any(tri_s[0::2] != tri_s[1::2]):
        return None
    seg_a, seg_b, seg_face = node_s[0::2], node_s[1::2], tri_face[tri_s[0::2]]
    nn, ns = len(uniq), len(seg_a)
    # a closed mesh: every crossed edge is shared by two crossed triangles, so the segments chain into loops
    ends = np.r_[seg_a, seg_b]
    if np.any(np.bincount(ends, minlength=nn) != 2):
        return None
    o_ = np.argsort(ends, kind="stable")
    nb = np.r_[seg_b, seg_a][o_].reshape(nn, 2)
    via = np.r_[np.arange(ns), np.arange(ns)][o_].reshape(nn, 2)
    seen = np.zeros(nn, dtype=bool)
    loops = []
    for start in range(nn):
        if seen[start]:
            continue
        path, segs = [start], []
        seen[start] = True
        cur, came = start, -1
        while True:
            k = 0 if via[cur, 0] != came else 1
            came, nxt = via[cur, k], nb[cur, k]
            segs.append(came)
            if nxt == start:
                break
            if seen[nxt]:
                return None
            seen[nxt] = True
            path.append(nxt)
            cur = nxt
        loops.append((np.array(path), np.array(segs)))
    wanted = [int(f) for f in face_ids if f >= 0]
    tol = 0.0015 * words.diag
    for _ in range(6):
        out_loops = []
        for path, _segs in loops:
            pts = xy[path]
            keep = _simplify_loop(pts, tol)
            if len(keep) >= 3:
                out_loops.append(np.round(pts[keep], 2).ravel().tolist())
        if not out_loops:
            return None
        traces: dict[str, list] = {}
        for fid in wanted:
            segs_out = []
            for path, segs in loops:
                on = seg_face[segs] == fid
                if not on.any():
                    continue
                m = len(segs)
                if on.all():
                    runs, shift = [(0, m)], 0
                else:
                    shift = int(np.flatnonzero(~on)[0])
                    d = np.diff(np.r_[0, np.roll(on, -shift).astype(int), 0])
                    runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
                for r0, r1 in runs:
                    idx = (np.arange(r0, r1 + 1) + shift) % m       # segment n runs from path[n] to path[n + 1]
                    pts = xy[path[idx]]
                    keep = _simplify(pts, tol)
                    for p0, p1 in zip(pts[keep[:-1]], pts[keep[1:]]):
                        segs_out.append(np.round(np.r_[p0, p1], 2).tolist())
            if segs_out:
                segs_out.sort(key=lambda s_: -np.hypot(s_[2] - s_[0], s_[3] - s_[1]))
                traces[str(fid)] = segs_out[:12]
        allxy = np.concatenate([np.asarray(lp).reshape(-1, 2) for lp in out_loops])
        section = {"origin": np.round(o, 4).tolist(), "u": u.tolist(), "v": v.tolist(), "loops": out_loops,
                   "bounds": np.round(np.r_[allxy.min(0), allxy.max(0)], 2).tolist(), "faces": traces}
        if len(json.dumps(section)) <= SECTION_BYTES:
            return section
        tol *= 2
    return None


def _region_at_end(k: int, signs: list[int], regions: list[dict], lo, hi) -> int | None:
    """The biggest rescan region at that end (+1 / -1 along axis k) of the part."""
    best = None
    for r in regions:
        if not r["rescan"]:
            continue
        c = r["center"][k]
        t = (c - lo[k]) / max(hi[k] - lo[k], 1e-9)
        at_end = (t > 0.9 and 1 in signs) or (t < 0.1 and -1 in signs)
        if at_end and (best is None or r["area_mm2"] > regions[best]["area_mm2"]):
            best = r["id"]
    return best


def _summarise(regions, measurements, surface, creport, tol, faces, symmetric: bool = False) -> dict:
    counts = {s: sum(m["status"] == s for m in measurements) for s in ("ok", "off", "close", "not_measured")}
    counts["measured"] = len(measurements) - counts["not_measured"]
    rescan = [r for r in regions if r["rescan"]]
    off_regions = [r for r in regions if r["kind"] == "off"]
    if counts["off"] or off_regions:
        verdict = "differs"
        bits = []
        if counts["off"]:
            bits.append(f"{counts['off']} measurement{'s are' if counts['off'] > 1 else ' is'} off")
        if off_regions:
            bits.append(f"{len(off_regions)} area{'s' if len(off_regions) > 1 else ''} of the surface "
                        f"{'are' if len(off_regions) > 1 else 'is'} beyond ±{tol:g} mm")
        good = surface["shares_pct"]["good"]
        if good >= MOSTLY_MATCHES_PCT:
            headline = (f"The scan mostly matches the golden model ({good:.0f} % of the surface is within "
                        f"±{tol:g} mm), but " + " and ".join(bits) + ".")
        else:
            headline = "The scan does not match the golden model: " + " and ".join(bits) + "."
    elif rescan or counts["not_measured"]:
        verdict = "incomplete"
        n = len(rescan)
        headline = (f"Everything that was scanned matches within ±{tol:g} mm, but "
                    + (f"{n} area{'s need' if n > 1 else ' needs'} scanning again" if n else
                       f"{counts['not_measured']} measurement{'s' if counts['not_measured'] > 1 else ''} could not be taken")
                    + " before the check is complete.")
    else:
        verdict = "match"
        headline = f"The scan matches the golden model: every measurement is within ±{tol:g} mm."
    shares = surface["shares_pct"]
    summary = [f"{surface['scanned_pct']:.0f} % of the golden surface was scanned; {shares['good']:.0f} % "
               f"matches within ±{tol:g} mm."]
    if measurements:
        summary.append(f"Measurements: {counts['ok']} match, {counts['off']} off, {counts['close']} too close to "
                       f"call, {counts['not_measured']} not measured (of {len(measurements)}).")
    if surface["noise"] > 0.5 * tol:
        summary.append(f"The scan's points scatter by about ±{surface['noise']:.3f} mm, which is large next to the "
                       f"±{tol:g} mm tolerance: single spots cannot be checked that finely (the measurements, which "
                       "average many points, still can).")
    if surface["edges"]["rounded_by"] and surface["edges"]["rounded_by"] > tol:
        summary.append(f"Sharp edges are rounded by up to {surface['edges']['rounded_by']:.2f} mm in the scan: "
                       "normal for any scanner, and left out of the check.")
    alignment = creport.get("alignment", {})
    warnings = [w for w in creport.get("warnings", []) if not (symmetric and w == alignment.get("warning"))]
    se = creport.get("scale_estimate") or {}
    if se.get("percent") is not None and abs(se["percent"]) >= 0.05:
        summary.append(f"The whole scan is {se['percent']:+.2f} % the size of the golden model: if many "
                       "measurements are off the same way, check the scanner calibration.")
    if symmetric:
        summary.append("The golden model looks the same turned another way (e.g. round about its axis), so where a "
                       "problem sits around that turn ('front', 'left') is only relative to this scan.")
    elif alignment.get("ambiguous") or alignment.get("warning"):
        summary.append("The part looks almost the same turned another way, so the line-up may be wrong: check "
                       "the colours make sense before trusting the numbers.")
    steps = [f"{r['name']}: {r['advice']}" for r in regions if r["rescan"]]
    return {"version": REPORT_VERSION, "verdict": verdict, "headline": headline, "summary": summary,
            "match_pct": shares["good"], "tolerance": tol, "counts": counts,
            "rescan": steps, "regions": regions, "measurements": measurements, "surface": surface,
            "faces": {"flat": sum(f["type"] == "plane" for f in faces),
                      "holes": sum(f["type"] == "cylinder" and f["hole"] for f in faces),
                      "round": sum(f["type"] == "cylinder" and not f["hole"] for f in faces)},
            "legend": LEGEND,
            "alignment": {**{k: alignment.get(k) for k in ("method", "fitness", "rmse", "ambiguous", "warning")},
                          "symmetric": symmetric},
            "transform": creport.get("transform"), "scale_estimate": creport.get("scale_estimate"),
            "deviation": {k: creport["stats"][k] for k in ("mean", "std", "rms", "p05", "p95",
                                                             "within_tolerance_pct")},
            "warnings": warnings}
