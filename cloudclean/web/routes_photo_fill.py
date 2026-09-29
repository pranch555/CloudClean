"""Fill the parts a scan missed from photos of the part (cloudclean/photo_fill.py).

POST /api/fill-from-photos {asset_id, photo_ids?, gap?}   -> job "fill_from_photos"
     photo_ids default to every photo of the scan's project
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .jobs_colour_photos import MIN_COLOUR_PHOTOS, TOO_FEW
from .jobs_photos3d import photos_of


class FillReq(BaseModel):
    asset_id: str
    photo_ids: list[str] = []
    gap: float | None = None
    name: str | None = None


def create_router(workspace, jobs) -> APIRouter:
    router = APIRouter()
    ws = workspace

    @router.post("/api/fill-from-photos")
    def fill_from_photos(req: FillReq):
        try:
            meta = ws.get(req.asset_id)
        except KeyError:
            raise HTTPException(400, f"The scan '{req.asset_id}' does not exist")
        if meta["kind"] not in ("pointcloud", "mesh"):
            raise HTTPException(400, f"'{meta['name']}' is a photo - pick the scan to fill")
        try:
            photos = photos_of(ws, req.photo_ids, meta.get("project"))
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        if len(photos) < MIN_COLOUR_PHOTOS:
            raise HTTPException(400, TOO_FEW)
        params = {}
        if req.gap is not None:
            if req.gap < 0:
                raise HTTPException(400, "gap cannot be negative")
            params["gap"] = req.gap
        payload = {"asset_id": req.asset_id, "photo_ids": [p["id"] for p in photos], "params": params,
                   "name": req.name}
        return jobs.submit("fill_from_photos", f"Fill {meta['name']} from {len(photos)} photos", payload)

    return router
