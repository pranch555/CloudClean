"""Hole detection and selective hole filling for triangle meshes (Revo Metro's "Hole Detection" / "Select Hole").

A hole is a closed loop of boundary edges (edges used by exactly one triangle). For an open scan the largest loop is
usually the scan's outer edge rather than a hole; it is reported with `outer: True` and never filled by default.

Filling adds new surface that was not scanned. It is done per loop with a fan around the loop centroid (one new
vertex per hole) oriented like the neighbouring triangles, so the rest of the mesh is not touched; the report says
how many holes and how much area were filled.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import open3d as o3d

Log = Callable[[str], None]


def _boundary_edges(tri: np.ndarray, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """Directed boundary edges (a -> b as they appear in their single triangle) and the non-manifold edge count."""
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]).astype(np.int64)
    lo, hi = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    key = lo * np.int64(n_vertices) + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    counts = np.diff(np.r_[starts, len(ks)])
    single = order[starts[counts == 1]]
    return e[single], int((counts > 2).sum())


def _loops(edges: np.ndarray) -> list[list[int]]:
    """Chain directed boundary edges into closed vertex loops (ordered as the edges run)."""
    nxt: dict[int, list[int]] = {}
    for a, b in edges.tolist():
        nxt.setdefault(a, []).append(b)
    loops: list[list[int]] = []
    used: set[tuple[int, int]] = set()
    for a0, b0 in edges.tolist():
        if (a0, b0) in used:
            continue
        loop = [a0]
        a, b = a0, b0
        ok = False
        for _ in range(len(edges) + 1):
            used.add((a, b))
            if b == a0:
                ok = True
                break
            loop.append(b)
            cands = [c for c in nxt.get(b, []) if (b, c) not in used]
            if not cands:
                break
            a, b = b, cands[0]
        if ok and len(loop) >= 3:
            loops.append(loop)
    return loops


def find_holes(mesh: o3d.geometry.TriangleMesh) -> dict:
    """All boundary loops with size, position and direction. Units are the mesh's (mm for scans)."""
    v = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)
    if not len(tri):
        raise ValueError("the mesh has no triangles")
    edges, nonmanifold = _boundary_edges(tri, len(v))
    loops = _loops(edges)
    holes = []
    for loop in loops:
        p = v[loop]
        seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        perimeter = float(seg.sum())
        center = p.mean(axis=0)
        # Newell's method: area vector of the (possibly non-planar) loop; its direction is the loop's normal
        area_vec = 0.5 * np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
        area = float(np.linalg.norm(area_vec))
        normal = area_vec / area if area > 0 else np.zeros(3)
        # diameter of the circle with the same area: a mesh's hole edge zig-zags along triangle edges, which makes
        # perimeter / pi read far too big (a 40 mm hole on a fine grid came out at 55 mm)
        diameter = 2.0 * float(np.sqrt(area / np.pi)) if area > 0 else perimeter / np.pi
        radii = np.linalg.norm(p - center, axis=1)
        spread = float(radii.std() / radii.mean()) if radii.mean() > 0 else 1.0
        holes.append({"loop": loop, "vertices": len(loop), "perimeter": perimeter,
                      "diameter": diameter, "area": area, "spread": spread,
                      "center": center.tolist(), "normal": normal.tolist()})
    holes.sort(key=lambda h: h["perimeter"], reverse=True)
    # an open scan's outer rim is the largest loop by far: report it, but it is not a hole to fill
    outer = bool(holes) and (len(holes) == 1 or holes[0]["perimeter"] > 3 * holes[1]["perimeter"]) \
        and holes[0]["perimeter"] > 0.25 * float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))) * np.pi
    for i, h in enumerate(holes):
        h["id"] = i
        h["outer"] = bool(outer and i == 0)
    return {"holes": holes, "boundary_edges": int(len(edges)), "nonmanifold_edges": nonmanifold,
            "watertight": len(edges) == 0}


def is_round(h: dict) -> bool:
    """Circular: its edge stays at about the same distance from its centre (zig-zag along triangle edges allowed)."""
    return h.get("spread", 1.0) < 0.12


def through_pairs(holes: list[dict]) -> list[tuple[int, int]]:
    """Loops that are the two ends of one through hole: same size, facing opposite ways, one behind the other."""
    inner = [h for h in holes if not h["outer"]]
    used: set[int] = set()
    pairs = []
    for i, h in enumerate(inner):
        if h["id"] in used:
            continue
        c1, n1 = np.asarray(h["center"]), np.asarray(h["normal"])
        best = None
        for g in inner[i + 1:]:
            if g["id"] in used or abs(g["diameter"] - h["diameter"]) > 0.08 * h["diameter"]:
                continue
            n2 = np.asarray(g["normal"])
            if float(n1 @ n2) > -0.9:
                continue
            off = np.asarray(g["center"]) - c1
            lateral = np.linalg.norm(off - (off @ n1) * n1)
            if lateral < 0.25 * h["diameter"] and (best is None or np.linalg.norm(off) < best[1]):
                best = (g["id"], float(np.linalg.norm(off)))
        if best is not None:
            used.update((h["id"], best[0]))
            pairs.append((h["id"], best[0]))
    return pairs


def hole_groups(holes: list[dict], ratio: float = 2.0) -> list[dict]:
    """The holes to fill (not the outer rim) grouped by size, biggest group first: sorted by diameter and split where
    the next hole is more than `ratio` times bigger (a part's design holes, marker-sticker spots and tiny gaps are
    usually far apart in size). A through hole shows as two loops, one per face: counted as one `through` hole.

    Each group: {diameter_min, diameter_max, count (loops), through (through holes), round, kind, ids, note?}
    kind: "sticker spots", "tiny gaps" or "holes"."""
    inner = sorted((h for h in holes if not h["outer"]), key=lambda h: h["diameter"])
    pair_of = {i: j for p in through_pairs(holes) for i, j in (p, p[::-1])}
    runs: list[list[dict]] = []
    for h in inner:
        if not runs or (h["diameter"] > ratio * runs[-1][-1]["diameter"]
                        and h["diameter"] - runs[-1][-1]["diameter"] > 0.5):
            runs.append([])
        runs[-1].append(h)
    out = []
    for g in reversed(runs):
        d = [h["diameter"] for h in g]
        ids = {h["id"] for h in g}
        through = sum(1 for h in g if h["id"] in pair_of and pair_of[h["id"]] in ids) // 2
        rnd = sum(1 for h in g if is_round(h))
        median = float(np.median(d))
        group = {"diameter_min": round(min(d), 2), "diameter_max": round(max(d), 2), "count": len(g),
                 "through": through, "round": rnd, "kind": "holes", "ids": [h["id"] for h in g]}
        if len(g) >= 6 and rnd >= 0.7 * len(g) and max(d) <= 1.3 * min(d) and 3 <= median <= 12 and through == 0:
            group["kind"] = "sticker spots"
            group["note"] = "many round holes of one size, not through: typical of spots left by marker stickers"
        elif max(d) < 3:
            group["kind"] = "tiny gaps"
        out.append(group)
    return out


def describe_group(g: dict) -> str:
    """'5 through holes Ø24.9-40.1 (10 loops)', '30 sticker spots Ø5.9-6.1', '20 tiny gaps Ø0.5-1.7'."""
    size = f"Ø{g['diameter_min']}-{g['diameter_max']}"
    if g["kind"] == "holes" and g["through"]:
        single = g["count"] - 2 * g["through"]
        return (f"{g['through']} through hole{'s' if g['through'] > 1 else ''} {size} ({g['count']} loops"
                + (f", {single} of them one-sided" if single else "") + ")")
    return f"{g['count']} {g['kind'] if g['kind'] != 'holes' else 'holes'} {size}"


def fill_splits(groups: list[dict]) -> list[dict]:
    """For each gap between size groups: the max_diameter that fills every smaller hole and keeps the bigger ones,
    in words. The one that fills the sticker spots (and anything smaller) comes first."""
    splits = []
    for i in range(len(groups) - 1):
        big, small = groups[i], groups[i + 1]
        cut = float(np.sqrt(big["diameter_min"] * small["diameter_max"]))
        kept, filled = groups[: i + 1], groups[i + 1:]
        splits.append({"max_diameter": round(cut, 1), "fills": sum(g["count"] for g in filled),
                       "keeps": sum(g["count"] for g in kept), "keeps_from_diameter": big["diameter_min"],
                       "fills_what": " + ".join(describe_group(g) for g in filled),
                       "keeps_what": " + ".join(describe_group(g) for g in kept),
                       "stickers": any(g["kind"] == "sticker spots" for g in filled)})
    # sticker options first, the one keeping everything bigger than the spots first among them
    splits.sort(key=lambda s: (not s["stickers"], s["fills"] if s["stickers"] else -s["fills"]))
    return splits


def public_holes(result: dict, limit: int = 200) -> dict:
    """The find_holes result without the vertex loops (for the API)."""
    out = dict(result)
    out["holes"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items() if k != "loop"}
                    for h in result["holes"][:limit]]
    out["count"] = len(result["holes"])
    out["holes_to_fill"] = sum(1 for h in result["holes"] if not h["outer"])
    return out


def fill_holes(mesh: o3d.geometry.TriangleMesh, hole_ids: list[int] | None = None, max_diameter: float = 0.0,
               log: Log = print, min_diameter: float = 0.0,
               except_ids: list[int] | None = None) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Fill the chosen holes (ids from find_holes), or every non-outer hole from `min_diameter` up to `max_diameter`
    (0 = no limit) apart from `except_ids`.

    Returns a new mesh and a report {filled, skipped, area_added, triangles_added, holes: [...]}.
    """
    found = find_holes(mesh)
    holes = found["holes"]
    if hole_ids is not None:
        wanted = set(int(i) for i in hole_ids)
        unknown = wanted - {h["id"] for h in holes}
        if unknown:
            raise ValueError(f"unknown hole id(s) {sorted(unknown)}; the mesh has {len(holes)} boundary loops")
        chosen = [h for h in holes if h["id"] in wanted]
    else:
        keep = set(int(i) for i in except_ids or [])
        chosen = [h for h in holes if not h["outer"] and h["id"] not in keep and h["diameter"] >= min_diameter
                  and (max_diameter <= 0 or h["diameter"] <= max_diameter)]
    v = np.asarray(mesh.vertices).copy()
    tri = np.asarray(mesh.triangles).copy()
    new_v, new_t, filled, area_added = [], [], [], 0.0
    base = len(v)
    for h in chosen:
        loop = h["loop"]
        c = base + len(new_v)
        new_v.append(h["center"])
        # the loop runs along the boundary edges in the direction they have in their triangles; the fill triangles
        # use each edge reversed so their winding (and normals) agree with the neighbours
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            new_t.append([b, a, c])
        area_added += h["area"]
        filled.append({"id": h["id"], "diameter": round(h["diameter"], 4), "vertices": h["vertices"]})
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.vstack([v, np.asarray(new_v).reshape(-1, 3)]) if new_v else v)
    out.triangles = o3d.utility.Vector3iVector(np.vstack([tri, np.asarray(new_t, dtype=np.int64).reshape(-1, 3)]) if new_t else tri)
    if mesh.has_vertex_colors():
        col = np.asarray(mesh.vertex_colors)
        extra = np.asarray([col[h["loop"]].mean(axis=0) for h in chosen]).reshape(-1, 3)
        out.vertex_colors = o3d.utility.Vector3dVector(np.vstack([col, extra]) if len(extra) else col)
    out.compute_vertex_normals()
    log(f"Filled {len(filled)} hole(s), {len(new_t):,} new triangles, {area_added:.2f} mm² of surface that was not "
        "scanned")
    report = {"filled": len(filled), "skipped": len(holes) - len(filled), "triangles_added": len(new_t),
              "area_added": round(area_added, 4), "holes": filled,
              "note": "Filled surface was not scanned; measurements across it are estimates."}
    return out, report
