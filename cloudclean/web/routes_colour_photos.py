"""Colour a scan or mesh from photos of the part (job `colour_from_photos`, docs/colour-from-photos.md).

POST /api/colour-from-photos {asset_id, photo_ids?, project_id?, name?, texture_size?}   -> the job
(is the reconstruction container ready: GET /api/photos-to-3d/status)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import jobs_photos3d
from .jobs_colour_photos import MIN_COLOUR_PHOTOS, TOO_FEW, check_texture_size
from .jobs_photos3d import photos_of


def recon_available() -> tuple[bool, str]:
    return jobs_photos3d.recon_available()


class ColourReq(BaseModel):
    asset_id: str
    photo_ids: list[str] | None = None
    project_id: str | None = None
    name: str | None = None
    texture_size: int | None = None


def create_router(workspace, jobs) -> APIRouter:
    router = APIRouter()

    @router.post("/api/colour-from-photos")
    def start(req: ColourReq):
        try:
            meta = workspace.get(req.asset_id)
        except KeyError:
            raise HTTPException(404, f"No asset {req.asset_id}")
        if meta.get("kind") == "image":
            raise HTTPException(400, f"'{meta['name']}' is a photo - pick the scan or mesh to colour")
        try:
            photos = photos_of(workspace, req.photo_ids, req.project_id or meta.get("project"))
            check_texture_size(req.texture_size)
        except KeyError as exc:
            raise HTTPException(404, f"No asset {exc}")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        photos = list({m["id"]: m for m in photos}.values())
        if len(photos) < MIN_COLOUR_PHOTOS:
            raise HTTPException(400, TOO_FEW)
        ok, reason = recon_available()
        if not ok:
            raise HTTPException(503, reason)
        payload = {**req.model_dump(), "photo_ids": [m["id"] for m in photos]}
        return jobs.submit("colour_from_photos", f"Colour {meta['name']} from {len(photos)} photos", payload)

    return router
