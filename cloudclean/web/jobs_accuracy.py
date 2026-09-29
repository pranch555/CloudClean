"""Job kinds for accuracy checks (Contract 5, docs/v3-plan.md, docs/accuracy.md).

`compare_scans` payload `{a_id, b_id, transform?, tolerance?, save_aligned?, name?}`: aligns scan B to scan A (two
scans of the same part), fits a relative scale / surface offset and measures the separation where they overlap -
the scanner's repeatability, independent of any processing. Output `comparison` (see accuracy.compare_scans).
With save_aligned (default) it also saves scan B moved rigidly into A's coordinates, operation `compare_scans`,
parents [a, b], with a per-point scalar `separation` (mm, + = B outside A, NaN = no common surface)."""
from __future__ import annotations

import numpy as np

from .. import accuracy as acc
from ..io import is_cloud
from .workspace import Workspace

SCAN_KINDS = ("pointcloud", "mesh")
_SUMMARY_KEYS = ("verdict", "sentence", "scale_ppm", "scale_se_ppm", "scale_with_offset_ppm", "offset_mm",
                 "axis_fit_offset_mm", "offset_se_mm", "axis_scale_ppm", "length_difference_from_scale_mm",
                 "separation", "overlap_fraction", "noise", "extents", "extents_common", "tolerance", "spacing")


def summary(result: dict) -> dict:
    """The comparison without bulky parts (histogram, pose alternatives' matrices), for reports and messages."""
    out = {k: result[k] for k in _SUMMARY_KEYS if k in result}
    al = result.get("alignment") or {}
    out["alignment"] = {k: al[k] for k in ("best_candidate", "ambiguous", "fine_score", "median_separation", "score")
                        if k in al}
    if al.get("alternatives"):
        best = al["alternatives"][0]
        out["alignment"]["best_alternative"] = {k: best[k] for k in ("candidate", "fine_score", "angle_from_best",
                                                                     "shift_from_best") if k in best}
    return acc.jsonable(out)


def job_compare_scans(ws: Workspace, payload: dict, log) -> list[str]:
    progress = getattr(log, "progress", None) or (lambda fraction, label=None: None)
    a_id, b_id = payload["a_id"], payload["b_id"]
    if a_id == b_id:
        raise ValueError("Pick two different scans of the same part")
    ma, mb = ws.get(a_id), ws.get(b_id)
    for m in (ma, mb):
        if m["kind"] not in SCAN_KINDS:
            raise ValueError(f"'{m['name']}' is a {m['kind']}, not a scan")
    progress(0.02, "loading")
    a, b = ws.load_geometry(a_id), ws.load_geometry(b_id)
    log(f"Scan A: {ma['name']}  |  scan B: {mb['name']}")
    result = acc.compare_scans(a, b, transform=payload.get("transform"),
                               tolerance=float(payload.get("tolerance") or 0.02), log=log,
                               progress=lambda f, label=None: progress(0.03 + 0.85 * f, label))
    result["a"] = {"id": a_id, "name": ma["name"]}
    result["b"] = {"id": b_id, "name": mb["name"]}
    created: list[str] = []
    if payload.get("save_aligned", True):
        progress(0.9, "saving the aligned scan")
        T = np.asarray(result["transform"], float)
        moved = acc.transformed_geometry(b, T)
        report = summary(result)
        report.update(a=result["a"], b=result["b"], transform=np.asarray(T).round(10).tolist(),
                      histogram=acc.jsonable(result["separation_histogram"]))
        name = payload.get("name") or f"{mb['name']} vs {ma['name']} · repeatability"
        params = {"tolerance": result["tolerance"], "transform_given": payload.get("transform") is not None}
        meta = ws.add_geometry(moved, name, "compare_scans", [a_id, b_id], params, report)
        pts = acc.geometry_points(moved)
        normals = None
        if is_cloud(moved) and moved.has_normals():
            normals = np.asarray(moved.normals)
        elif not is_cloud(moved):
            moved.compute_vertex_normals()
            normals = np.asarray(moved.vertex_normals)
        d, _, gap, ok = acc.SurfaceModel(a).query(pts, normals)
        reach = 3 * result["spacing"] + 5 * result["noise"]["combined"]
        values = np.where(ok & (gap <= reach), d, np.nan)
        ws.add_scalars(meta["id"], "separation", values, unit="mm",
                       description=f"Signed distance to '{ma['name']}' (+ = this scan outside it; NaN = no common "
                                   "surface)")
        created.append(meta["id"])
        result["aligned_id"] = meta["id"]
    output = getattr(log, "output", None)
    if output:
        full = summary(result)
        full.update(a=result["a"], b=result["b"], aligned_id=result.get("aligned_id"),
                    transform=np.asarray(result["transform"]).round(10).tolist(),
                    histogram=acc.jsonable(result["separation_histogram"]))
        output(comparison=full)
    progress(1.0, "done")
    return created


JOBS = {"compare_scans": job_compare_scans}
