"""Merge assessment API.

POST /api/merge/assess {asset_ids, params?} -> job `merge_assess`. When it is done, `job.output.assessment` is
{recommendation: single|merge|use_best|ask, reasons: [sentences], best_index, spacing, params, asset_ids, names,
 transforms: [4x4 per scan, scan -> scan 0],
 scans: [{index, name, points, noise_mm, (index >= 1:) fitness, rmse, rmse_over_spacing, ambiguous,
          alternative_angle?, already_aligned, overlap, coverage_gain, coverage_gain_pct, coverage_gain_mm2,
          reference_area_mm2, layering_mm, layering_p90_mm, separation_ratio, combined_noise_mm, doubled_surface,
          confident, adds_coverage, recommendation, compared_to}]}.
To merge what the user approved: POST /api/merge {asset_ids, params, transforms: assessment.transforms}.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..register import MergeParams


class AssessReq(BaseModel):
    asset_ids: list[str]
    params: dict = {}


def create_router(workspace, jobs) -> APIRouter:
    ws = workspace
    router = APIRouter()

    @router.post("/api/merge/assess")
    def assess(req: AssessReq):
        if len(req.asset_ids) < 2:
            raise HTTPException(400, "Select at least two scans to check a merge")
        if len(set(req.asset_ids)) != len(req.asset_ids):
            raise HTTPException(400, "The same scan is selected twice")
        metas = []
        for asset_id in req.asset_ids:
            try:
                metas.append(ws.get(asset_id))
            except KeyError:
                raise HTTPException(404, f"Asset {asset_id} not found")
        bad = [m["name"] for m in metas if m["kind"] not in ("pointcloud", "mesh")]
        if bad:
            raise HTTPException(400, f"Only scans can be merged - '{bad[0]}' is a photo")
        try:
            MergeParams.from_dict(req.params)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc))
        names = [m["name"] for m in metas]
        title = ", ".join(names[:2]) + (f" +{len(names) - 2}" if len(names) > 2 else "")
        return jobs.submit("merge_assess", f"Check merge {title}", req.model_dump())

    return router
