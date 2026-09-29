"""Make a 3D model from photos (job `photos_to_3d`, docs/photos-to-3d.md).

GET  /api/photos-to-3d/status   {available, reason}
POST /api/photos-to-3d {photo_ids?, project_id?, name?, ruler_mm?}   -> the job
GET  /api/photos/scale-sheet?paper=a4|letter&format=pdf|png   the printable scale sheet (true size from photos)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from .. import scale_sheet
from .jobs_photos3d import MIN_PHOTOS, check_ruler, photos_of, recon_available


class PhotosReq(BaseModel):
    photo_ids: list[str] | None = None
    project_id: str | None = None
    name: str | None = None
    ruler_mm: float | None = None


def create_router(workspace, jobs) -> APIRouter:
    router = APIRouter()

    @router.get("/api/photos-to-3d/status")
    def status():
        ok, reason = recon_available()
        return {"available": ok, "reason": reason}

    @router.post("/api/photos-to-3d")
    def start(req: PhotosReq):
        try:
            photos = photos_of(workspace, req.photo_ids, req.project_id)
            check_ruler(req.ruler_mm)
        except KeyError as exc:
            raise HTTPException(404, f"No asset {exc}")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if len(photos) < MIN_PHOTOS:
            raise HTTPException(400, "Add at least 2 photos of the part - 12 or more, about every 30 degrees all the way round, work best")
        ok, reason = recon_available()
        if not ok:
            raise HTTPException(503, reason)
        payload = {**req.model_dump(), "photo_ids": [m["id"] for m in photos]}
        return jobs.submit("photos_to_3d", f"3D model from {len(photos)} photos", payload)

    @router.get("/api/photos/scale-sheet")
    def sheet(paper: str = "a4", fmt: str = Query("pdf", alias="format")):
        """The scale sheet to print: put the part on it, and the photo model comes out at true size."""
        paper, fmt = paper.lower(), fmt.lower()
        if fmt not in ("pdf", "png"):
            raise HTTPException(400, "format must be pdf or png")
        try:
            data = scale_sheet.pdf(paper) if fmt == "pdf" else scale_sheet.png(paper)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        name = f"cloudclean-scale-sheet-{paper}.{fmt}"
        return Response(data, media_type="application/pdf" if fmt == "pdf" else "image/png",
                        headers={"Content-Disposition": f'inline; filename="{name}"',
                                 "Cache-Control": "public, max-age=86400"})

    return router
