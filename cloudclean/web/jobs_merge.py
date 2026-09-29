"""Job `merge_assess`: check whether scans should be merged before merging them (no asset is created).

Payload `{asset_ids, params?}` (params = MergeParams overrides). Output `assessment` = `register.assess_merge`
result plus `names` / `asset_ids`; pass its `transforms` to the `merge` job to merge exactly what was approved."""
from __future__ import annotations

from ..register import MergeParams, assess_merge
from .workspace import Workspace


def run_assessment(ws: Workspace, asset_ids: list[str], params: dict | None = None, log=print) -> dict:
    """Load the assets and assess them (shared by the merge_assess and autopilot jobs)."""
    metas = [ws.get(i) for i in asset_ids]
    bad = [m["name"] for m in metas if m["kind"] not in ("pointcloud", "mesh")]
    if bad:
        raise ValueError(f"Only scans can be merged: {', '.join(bad)} is not a point cloud or mesh")
    geoms = [ws.load_geometry(i) for i in asset_ids]
    assessment = assess_merge(geoms, MergeParams.from_dict(params or {}), log)
    assessment["asset_ids"] = list(asset_ids)
    assessment["names"] = [m["name"] for m in metas]
    for s in assessment["scans"]:
        s["name"] = metas[s["index"]]["name"]
    return assessment


def job_merge_assess(ws: Workspace, payload: dict, log) -> list[str]:
    ids = list(payload.get("asset_ids") or [])
    if len(ids) < 2:
        raise ValueError("Select at least two scans to assess a merge")
    progress = getattr(log, "progress", None)
    if progress:
        progress(0.05, "Assessing scans")
    assessment = run_assessment(ws, ids, payload.get("params"), log)
    for reason in assessment["reasons"]:
        log(f"  {reason}")
    output = getattr(log, "output", None)
    if output:
        output(assessment=assessment)
    if progress:
        progress(1.0, "Done")
    return []


JOBS = {"merge_assess": job_merge_assess}
