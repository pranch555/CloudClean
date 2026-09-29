"""Pictures of merge options (job `merge_options`, cloudclean/merge_views.py).

POST /api/merge/options {asset_ids: [reference, moving], params?} -> the job
GET  /api/merge/options/{key}/{option}.jpg -> the picture of one option
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


class OptionsReq(BaseModel):
    asset_ids: list[str]
    params: dict = {}


def create_router(workspace, jobs) -> APIRouter:
    router = APIRouter()

    @router.post("/api/merge/options")
    def start(req: OptionsReq):
        if len(req.asset_ids) != 2:
            raise HTTPException(400, "Pick exactly two scans: the reference first")
        for asset_id in req.asset_ids:
            try:
                workspace.get(asset_id)
            except KeyError:
                raise HTTPException(404, f"No asset {asset_id}")
        names = " + ".join(workspace.get(i)["name"] for i in req.asset_ids)
        return jobs.submit("merge_options", f"Merge options: {names}", req.model_dump())

    @router.get("/api/merge/options/{key}/{name}")
    def picture(key: str, name: str):
        if not re.fullmatch(r"[0-9a-f]{12}", key) or not re.fullmatch(r"[A-Z]\.jpg", name):
            raise HTTPException(404, "Not Found")
        path = workspace.root / "renders" / key / name
        if not path.is_file():
            raise HTTPException(404, "That picture is gone - compare the options again")
        return FileResponse(path, media_type="image/jpeg")

    return router
