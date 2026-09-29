"""Projects: group assets by part (Contract 1 in docs/v3-plan.md, details in docs/v3-backend.md).

GET    /api/projects                      -> [{id, name, created, updated, description, cover_asset_id,
                                               golden_asset_id, counts: {scans, meshes, results, photos, total}}]
                                               newest first
POST   /api/projects {name, description?} -> project (with counts)
GET    /api/projects/{id}                 -> project (with counts)
PATCH  /api/projects/{id} {name?, description?, cover_asset_id?, golden_asset_id?}
                                          (cover / golden model: an asset of the project, or null; the golden
                                          model must be a mesh - docs/golden-model.md)
DELETE /api/projects/{id}?delete_assets=false  -> {deleted, deleted_assets}; 409 while it still holds assets
POST   /api/assets/{id}/move {project_id} -> asset meta
PUT    /api/assets/{id}/thumbnail         body = PNG bytes (<= 2 MB) -> asset meta (has_thumbnail: true)
GET    /api/assets/{id}/thumbnail         -> image/png (404 when none)

On start-up every asset without a project is filed into "My scans" (created once).
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .workspace import MAX_THUMBNAIL, ProjectNotEmpty


class CreateReq(BaseModel):
    name: str
    description: str | None = None


class MoveReq(BaseModel):
    project_id: str


def create_router(workspace, jobs) -> APIRouter:
    ws = workspace
    router = APIRouter()
    ws.migrate_projects()

    def asset(asset_id: str) -> dict:
        try:
            return ws.get(asset_id)
        except KeyError:
            raise HTTPException(404, "Asset not found")

    def summary(project_id: str) -> dict:
        for p in ws.project_summaries():
            if p["id"] == project_id:
                return p
        raise HTTPException(404, "Project not found")

    @router.get("/api/projects")
    def list_projects():
        return ws.project_summaries()

    @router.post("/api/projects")
    def create_project(req: CreateReq):
        try:
            project = ws.create_project(req.name, req.description or "")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return summary(project["id"])

    @router.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        return summary(project_id)

    @router.patch("/api/projects/{project_id}")
    def patch_project(project_id: str, values: dict):
        summary(project_id)
        editable = ("name", "description", "cover_asset_id", "golden_asset_id")
        fields = {k: values[k] for k in editable if k in values}
        unknown = sorted(set(values) - set(editable))
        if unknown:
            raise HTTPException(400, f"Unknown project field(s): {', '.join(unknown)}. "
                                     f"Editable: {', '.join(editable)}")
        try:
            ws.update_project(project_id, **fields)
        except KeyError:
            raise HTTPException(404, "Project not found")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return summary(project_id)

    @router.delete("/api/projects/{project_id}")
    def delete_project(project_id: str, delete_assets: bool = False):
        summary(project_id)
        try:
            return ws.delete_project(project_id, delete_assets=delete_assets)
        except ProjectNotEmpty as exc:
            raise HTTPException(409, str(exc))
        except KeyError:
            raise HTTPException(404, "Project not found")

    @router.post("/api/assets/{asset_id}/move")
    def move_asset(asset_id: str, req: MoveReq):
        asset(asset_id)
        try:
            return ws.move_asset(asset_id, req.project_id)
        except KeyError:
            raise HTTPException(404, "Project not found")

    @router.put("/api/assets/{asset_id}/thumbnail")
    async def put_thumbnail(asset_id: str, request: Request):
        asset(asset_id)
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_THUMBNAIL:
            raise HTTPException(413, "The thumbnail is too large (at most 2 MB)")
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > MAX_THUMBNAIL:
                raise HTTPException(413, "The thumbnail is too large (at most 2 MB)")
        try:
            return await asyncio.to_thread(ws.set_thumbnail, asset_id, bytes(data))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/api/assets/{asset_id}/thumbnail")
    def get_thumbnail(asset_id: str):
        asset(asset_id)
        try:
            path = ws.thumbnail_path(asset_id)
        except KeyError:
            raise HTTPException(404, "This asset has no thumbnail")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})

    return router
