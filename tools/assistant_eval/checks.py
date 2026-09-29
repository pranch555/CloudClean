"""Machine checks of a scenario's outcome, independent of the CloudClean code under test.

The flange checks look at the geometry itself: boundary loops (edges used by one triangle, grouped into connected
loops) and rays shot at every flaw and through every design hole (open3d RaycastingScene).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

SMALL_LIMIT = 15.0      # loops below this diameter (mm) are "small holes"; the design holes are 25 and 40 mm
COVER_TOL = 0.6         # a filled flaw: the first surface hit lies within this of the part's face (mm)


# --------------------------------------------------------------------------- geometry
def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.triangles, dtype=np.int64)


def boundary_loops(V: np.ndarray, F: np.ndarray) -> list[dict]:
    """Connected groups of boundary edges: [{center, diameter (perimeter / pi), perimeter, edges}], largest first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if not len(F):
        return []
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    lo, hi = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    key = lo * np.int64(len(V)) + hi
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    b = np.c_[lo, hi][counts[inv] == 1]
    if not len(b):
        return []
    verts, local = np.unique(b.ravel(), return_inverse=True)
    local = local.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(local)), (local[:, 0], local[:, 1])), shape=(len(verts), len(verts)))
    n, label = connected_components(graph, directed=False)
    edge_label = label[local[:, 0]]
    length = np.linalg.norm(V[b[:, 0]] - V[b[:, 1]], axis=1)
    mid = (V[b[:, 0]] + V[b[:, 1]]) / 2
    loops = []
    for k in range(n):
        sel = edge_label == k
        per = float(length[sel].sum())
        w = length[sel] / max(per, 1e-12)
        loops.append({"center": (mid[sel] * w[:, None]).sum(axis=0).round(3).tolist(), "perimeter": round(per, 3),
                      "diameter": round(per / math.pi, 3), "edges": int(sel.sum())})
    return sorted(loops, key=lambda h: -h["perimeter"])


def first_hits(V: np.ndarray, F: np.ndarray, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Distance to the first triangle hit along each ray (inf: none)."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    rays = np.c_[origins, directions].astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)


def _feature_ray(f: dict) -> tuple[list, list]:
    x, y, z = f["center"]
    if f["face"] == "top":
        return [x, y, z + 18.0], [0.0, 0.0, -1.0]
    if f["face"] == "bottom":
        return [x, y, z - 18.0], [0.0, 0.0, 1.0]
    r = math.hypot(x, y)
    u = [x / r, y / r, 0.0]
    return [x + 18.0 * u[0], y + 18.0 * u[1], z], [-u[0], -u[1], 0.0]


def flange_check(V: np.ndarray, F: np.ndarray, truth: dict) -> dict:
    """Only the design holes (bore + 4 lightening holes) may stay open; every sticker hole and gap must be closed.

    success = no boundary loop below SMALL_LIMIT, no other unexplained loop, each design hole still open (5 vertical
    rays through it hit nothing between z = -1 and 13) and every flaw covered (a ray at it hits the part's face).
    When the files were posed in a scanner frame (truth["pose"]), the mesh is first mapped back to the part's axes."""
    pose = truth.get("pose")
    if pose:
        V = (V - np.asarray(pose["translation"])) @ np.asarray(pose["rotation"])
    loops = boundary_loops(V, F)
    design = truth["design_holes"]
    kinds = {"design": 0, "small": 0, "other": 0}
    small_examples, other_examples = [], []
    for h in loops:
        cx, cy, _ = h["center"]
        near = [d for d in design if math.hypot(cx - d["center"][0], cy - d["center"][1]) < 3.0
                and h["diameter"] >= 0.5 * d["diameter"]]
        if near:
            kinds["design"] += 1
        elif h["diameter"] < SMALL_LIMIT:
            kinds["small"] += 1
            if len(small_examples) < 5:
                small_examples.append({"d": h["diameter"], "at": h["center"]})
        else:
            kinds["other"] += 1
            if len(other_examples) < 5:
                other_examples.append({"d": h["diameter"], "at": h["center"]})
    # design holes: 5 rays each (axis + 4 at 0.6 r) must pass without touching the part
    origins, dirs, owner = [], [], []
    for i, d in enumerate(design):
        r = d["diameter"] / 2
        for dx, dy in ((0, 0), (0.6, 0), (-0.6, 0), (0, 0.6), (0, -0.6)):
            origins.append([d["center"][0] + dx * r, d["center"][1] + dy * r, 40.0])
            dirs.append([0.0, 0.0, -1.0])
            owner.append(i)
    t = first_hits(V, F, np.asarray(origins), np.asarray(dirs))
    z_hit = 40.0 - t
    blocked = np.isfinite(t) & (z_hit > -1.0) & (z_hit < truth["thickness"] + 1.0)
    closed = sorted({design[owner[k]]["name"] for k in np.flatnonzero(blocked)})
    # flaws: the first hit must be the face they sit in
    feats = truth["small_features"]
    rays = [_feature_ray(f) for f in feats]
    t = first_hits(V, F, np.asarray([r[0] for r in rays]), np.asarray([r[1] for r in rays]))
    covered = np.isfinite(t) & (np.abs(t - 18.0) < COVER_TOL)
    open_feats = [f for f, c in zip(feats, covered) if not c]
    by_kind = {}
    for f in open_feats:
        k = f"{f['kind']}s ({f['face']})"
        by_kind[k] = by_kind.get(k, 0) + 1
    problems = []
    if kinds["small"]:
        problems.append(f"{kinds['small']} small hole(s) still open")
    if open_feats:
        problems.append("not filled: " + ", ".join(f"{v} {k}" for k, v in sorted(by_kind.items())))
    if closed:
        problems.append("design hole(s) closed: " + ", ".join(closed))
    if kinds["other"]:
        problems.append(f"{kinds['other']} unexpected large opening(s)")
    return {"success": not problems, "problems": problems, "loops": len(loops), "loops_by_kind": kinds,
            "design_open": len(design) - len(closed), "design_total": len(design),
            "flaws_filled": int(covered.sum()), "flaws_total": len(feats),
            "small_loop_examples": small_examples, "other_loop_examples": other_examples,
            "triangles": int(len(F))}


# --------------------------------------------------------------------------- answers
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])(\d{1,5}(?:[.,]\d+)?)")   # "Ø40.0" counts: only ASCII letters block a number


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUMBER_RE.finditer(text or ""):
        try:
            out.append(float(m.group(1).replace(",", ".")))
        except ValueError:
            pass
    return out


def mentions(text: str, phrases: list[str]) -> dict:
    low = " ".join((text or "").lower().replace("-", " ").split())
    return {p: p.lower() in low for p in phrases}
