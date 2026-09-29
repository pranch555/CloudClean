"""FastAPI app behind the CloudClean web interface."""
from __future__ import annotations

import io
import json
import threading
import uuid
import webbrowser
import zipfile
from dataclasses import fields
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..clean import CleanParams
from ..compare import CompareParams
from ..io import IMAGE_EXTS, SUPPORTED_EXTS, save, to_jpeg
from ..mesh import MeshParams
from ..register import MergeParams
from ..texture import TextureParams
from .auth import COOKIE, KEY_HEADER, AuthError, AuthStore, current_user, set_session_cookie
from .jobs import JobManager
from .workspace import Workspace, safe_name

STATIC_DIR = Path(__file__).parent / "static"
ROUTER_MODULES = ["cloudclean.web.routes_edit", "cloudclean.web.routes_compare", "cloudclean.web.routes_assistant",
                  "cloudclean.web.routes_capture", "cloudclean.web.routes_autopilot", "cloudclean.web.routes_merge",
                  "cloudclean.web.routes_measure", "cloudclean.web.routes_projects", "cloudclean.web.routes_understand",
                  "cloudclean.web.routes_turntable", "cloudclean.web.routes_accuracy", "cloudclean.web.routes_holes",
                  "cloudclean.web.routes_system", "cloudclean.web.routes_merge_options",
                  "cloudclean.web.routes_markers", "cloudclean.web.routes_photos3d",
                  "cloudclean.web.routes_colour_photos", "cloudclean.web.routes_auth",
                  "cloudclean.web.routes_golden", "cloudclean.web.routes_photo_fill"]
EXPORT_FORMATS = {"pointcloud": ["ply", "pcd", "xyz", "asc", "csv"], "mesh": ["ply", "stl", "obj", "3mf", "glb", "off"]}


class PathsReq(BaseModel):
    paths: list[str]
    project_id: str | None = None


class RenameReq(BaseModel):
    name: str


class CleanReq(BaseModel):
    asset_ids: list[str]
    preset: str = "standard"
    params: dict = {}


class MergeReq(BaseModel):
    asset_ids: list[str]
    params: dict = {}
    pairs: dict | None = None
    name: str | None = None
    transforms: list[list[list[float]] | None] | None = None  # approved poses from /api/merge/assess
    assessment: dict | None = None  # the pre-merge check the user saw (kept on the merged asset for caveats)


class MeshReq(BaseModel):
    asset_id: str
    params: dict = {}


class ViewReq(BaseModel):
    image_id: str
    view_matrix: list[float]
    fov: float
    aspect: float | None = None


class TextureReq(BaseModel):
    asset_id: str
    views: list[ViewReq]
    params: dict = {}


class PipelineReq(BaseModel):
    asset_ids: list[str]
    preset: str = "standard"
    skip_clean: bool = False
    clean: dict = {}
    merge: dict = {}
    mesh: dict = {}
    pairs: dict | None = None


def _schema(cls) -> dict:
    obj = cls()
    return {f.name: {"default": getattr(obj, f.name), "type": type(getattr(obj, f.name)).__name__}
            for f in fields(obj)}


def _validate(cls, data: dict, preset: str | None = None) -> None:
    try:
        p = cls.preset(preset) if preset else cls()
        p.update(data or {})
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))


def create_app(workspace="workspace", auth: bool = False) -> FastAPI:
    """auth: everyone signs in (cloudclean serve turns it on; docs/accounts.md)."""
    ws = Workspace(workspace)
    jobs = JobManager(ws)
    app = FastAPI(title="CloudClean")
    app.state.workspace, app.state.jobs = ws, jobs
    app.state.auth = AuthStore(ws.root) if auth else None
    if app.state.auth is not None:
        store = app.state.auth
        jobs.on_submit = store.note_job
        public = {"/api/auth/status", "/api/auth/login", "/api/auth/register", "/api/auth/logout"}
        api_docs = {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}

        @app.exception_handler(AuthError)
        async def auth_error(request, exc: AuthError):
            return JSONResponse({"detail": str(exc)}, status_code=exc.status)

        @app.middleware("http")
        async def require_sign_in(request, call_next):
            """Every API call (and the API's docs) needs a signed-in account, or the machine key: the bridge,
            scripts. The pages themselves load without: the app then shows the sign-in page."""
            path = request.scope["path"]   # as routed: request.url.path would drop an encoded '#...'
            if not (path.startswith("/api/") or path in api_docs) or path in public:
                return await call_next(request)
            token = request.cookies.get(COOKIE)
            try:
                user, renewed = await run_in_threadpool(store.identify, token, request.headers.get(KEY_HEADER))
            except AuthError as exc:   # the accounts file is damaged
                return JSONResponse({"detail": str(exc)}, status_code=exc.status)
            if user is None:
                return JSONResponse({"detail": "Sign in first"}, status_code=401)
            request.state.user = user
            reset = current_user.set(user["username"])
            try:
                response = await call_next(request)
            finally:
                current_user.reset(reset)
            if renewed:   # the session now ends 30 days from now: the browser should keep the cookie as long
                set_session_cookie(response, token, request.url.scheme == "https")
            return response

    def meta(asset_id: str) -> dict:
        try:
            return ws.get(asset_id)
        except KeyError:
            raise HTTPException(404, "Asset not found")

    def names(ids) -> str:
        n = [meta(i)["name"] for i in ids]
        return ", ".join(n[:2]) + (f" +{len(n) - 2}" if len(n) > 2 else "")

    def target_project(project_id: str | None) -> str:
        """The project new imports go to: the given one, else the most recently updated project."""
        if project_id:
            try:
                return ws.get_project(project_id)["id"]
            except KeyError:
                raise HTTPException(400, f"Project {project_id} not found")
        return ws.default_project_id()

    # ------------------------------------------------------------------ info
    @app.get("/api/params")
    def params():
        return {"schema": {"clean": _schema(CleanParams), "merge": _schema(MergeParams),
                           "mesh": _schema(MeshParams), "texture": _schema(TextureParams),
                           "compare": _schema(CompareParams)},
                "presets": CleanParams.PRESETS, "formats": EXPORT_FORMATS, "workspace": str(ws.root)}

    @app.get("/api/assets")
    def list_assets(project: str | None = None):
        metas = ws.list()
        return [m for m in metas if m.get("project") == project] if project else metas

    @app.get("/api/assets/{asset_id}")
    def get_asset(asset_id: str):
        return {**meta(asset_id), "report": ws.report(asset_id)}

    @app.patch("/api/assets/{asset_id}")
    def rename_asset(asset_id: str, req: RenameReq):
        meta(asset_id)
        return ws.update(asset_id, name=req.name.strip() or "asset")

    @app.delete("/api/assets/{asset_id}")
    def delete_asset(asset_id: str):
        meta(asset_id)
        ws.delete(asset_id)
        return {"deleted": asset_id}

    @app.get("/api/assets/{asset_id}/preview")
    def preview_file(asset_id: str):
        meta(asset_id)
        # no-cache = revalidate via ETag each time (cheap 304), so a rebuilt preview is always picked up
        return FileResponse(ws.asset_dir(asset_id) / "preview.ply", media_type="application/octet-stream",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/api/assets/{asset_id}/scalars/{name}")
    def scalars_file(asset_id: str, name: str):
        """Float32 little-endian values aligned with preview.ply (NaN = no value)."""
        meta(asset_id)
        try:
            path = ws.scalars_path(asset_id, name)
        except KeyError:
            raise HTTPException(404, f"No scalar field '{name}'")
        return FileResponse(path, media_type="application/octet-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/assets/{asset_id}/image")
    def image_file(asset_id: str):
        meta(asset_id)
        return FileResponse(ws.image_path(asset_id))

    @app.get("/api/assets/{asset_id}/download")
    def download(asset_id: str, format: str = "ply"):
        m = meta(asset_id)
        base = safe_name(m["name"])
        d = ws.asset_dir(asset_id)
        if format == "textured_glb":
            if not (d / "textured.glb").exists():
                raise HTTPException(404, "This asset has no UV texture")
            return FileResponse(d / "textured.glb", filename=f"{base}_textured.glb")
        if format == "textured_obj":
            folder = d / "textured_obj"
            if not folder.exists():
                raise HTTPException(404, "This asset has no UV texture")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in folder.iterdir():
                    zf.write(f, f.name)
            return Response(buf.getvalue(), media_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{base}_textured_obj.zip"'})
        if format == "report":
            return Response(json.dumps({**m, "report": ws.report(asset_id)}, indent=2, default=str),
                            media_type="application/json",
                            headers={"Content-Disposition": f'attachment; filename="{base}_report.json"'})
        if m["kind"] == "image" or format not in EXPORT_FORMATS[m["kind"]]:
            raise HTTPException(400, f"Cannot export {m['kind']} as {format}")
        out = d / "exports" / f"export.{format}"
        if not out.exists():
            save(ws.load_geometry(asset_id), out)
        return FileResponse(out, filename=f"{base}.{format}")

    # ------------------------------------------------------------------ import
    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...), project_id: str | None = Form(None)):
        started, images = [], []
        project = target_project(project_id)
        for f in files:
            name = Path(f.filename or "upload").name
            ext = Path(name).suffix.lower()
            if ext not in IMAGE_EXTS and ext not in SUPPORTED_EXTS:
                raise HTTPException(400, f"Unsupported file type '{ext}' ({name})")
            tmp = ws.uploads_dir / f"{uuid.uuid4().hex}{ext}"
            with open(tmp, "wb") as out:
                while chunk := await f.read(4 << 20):
                    out.write(chunk)
            if ext in IMAGE_EXTS:
                try:
                    images.append(ws.add_image(tmp, Path(name).stem, project=project))
                except ValueError as exc:
                    raise HTTPException(400, f"{name}: {exc}")
                finally:
                    tmp.unlink(missing_ok=True)
            else:
                started.append(jobs.submit("import", f"Import {name}", {
                    "path": str(tmp), "name": Path(name).stem, "temporary": True, "source": name,
                    "project_id": project}))
        return {"jobs": started, "images": images}

    @app.post("/api/images/jpeg")
    async def image_as_jpeg(file: UploadFile = File(...), max_side: int = Form(1600)):
        """A photo the browser cannot show (HEIC from a phone) as an upright JPEG, for previews and the chat."""
        ext = Path(file.filename or "photo.jpg").suffix.lower()
        if ext not in IMAGE_EXTS:
            raise HTTPException(400, f"Not a photo: {file.filename}")
        tmp = ws.uploads_dir / f"{uuid.uuid4().hex}{ext}"
        out = tmp.with_name(tmp.stem + "-view.jpg")
        try:
            tmp.write_bytes(await file.read())
            to_jpeg(tmp, out, max_side=max(16, min(max_side, 4096)), quality=88)
            return Response(out.read_bytes(), media_type="image/jpeg")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except OSError as exc:
            raise HTTPException(400, f"Cannot read {file.filename}: {exc}")
        finally:
            tmp.unlink(missing_ok=True)
            out.unlink(missing_ok=True)

    @app.post("/api/import-paths")
    def import_paths(req: PathsReq):
        started, images = [], []
        project = target_project(req.project_id)
        for raw in req.paths:
            path = Path(raw.strip().strip('"'))
            if not path.is_file():
                raise HTTPException(400, f"File not found: {path}")
            ext = path.suffix.lower()
            if ext in IMAGE_EXTS:
                try:
                    images.append(ws.add_image(path, path.stem, project=project))
                except ValueError as exc:
                    raise HTTPException(400, f"{path.name}: {exc}")
            elif ext in SUPPORTED_EXTS:
                started.append(jobs.submit("import", f"Import {path.name}",
                                           {"path": str(path), "name": path.stem, "source": str(path),
                                            "project_id": project}))
            else:
                raise HTTPException(400, f"Unsupported file type '{ext}' ({path.name})")
        return {"jobs": started, "images": images}

    # ------------------------------------------------------------------ operations
    @app.post("/api/clean")
    def clean(req: CleanReq):
        if not req.asset_ids:
            raise HTTPException(400, "Select at least one scan")
        _validate(CleanParams, req.params, req.preset)
        return jobs.submit("clean", f"Clean {names(req.asset_ids)}", req.model_dump())

    @app.post("/api/merge")
    def merge(req: MergeReq):
        if len(req.asset_ids) < 2:
            raise HTTPException(400, "Select at least two scans to merge")
        _validate(MergeParams, req.params)
        return jobs.submit("merge", f"Merge {names(req.asset_ids)}", req.model_dump())

    @app.post("/api/mesh")
    def mesh(req: MeshReq):
        _validate(MeshParams, req.params)
        return jobs.submit("mesh", f"Mesh {names([req.asset_id])}", req.model_dump())

    @app.post("/api/texture")
    def texture(req: TextureReq):
        if not req.views:
            raise HTTPException(400, "Save at least one aligned photo view first")
        _validate(TextureParams, req.params)
        for v in req.views:
            meta(v.image_id)
        return jobs.submit("texture", f"Colour {names([req.asset_id])}", req.model_dump())

    @app.post("/api/pipeline")
    def pipeline(req: PipelineReq):
        if not req.asset_ids:
            raise HTTPException(400, "Select at least one scan")
        _validate(CleanParams, req.clean, req.preset)
        _validate(MergeParams, req.merge)
        _validate(MeshParams, req.mesh)
        return jobs.submit("pipeline", f"Pipeline {names(req.asset_ids)}", req.model_dump())

    # ------------------------------------------------------------------ jobs
    @app.get("/api/jobs")
    def list_jobs():
        return jobs.list()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            jobs.cancel(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")
        return {"cancelled": job_id}

    # Feature modules expose create_router(workspace, jobs) -> APIRouter
    import importlib

    for name in ROUTER_MODULES:
        app.include_router(importlib.import_module(name).create_router(ws, jobs))

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8765, workspace="workspace", open_browser: bool = True,
          accounts: bool = True) -> None:
    import uvicorn

    app = create_app(workspace, auth=accounts)
    url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}"
    print(f"CloudClean is running at {url}")
    print(f"Workspace: {Path(workspace).resolve()}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    # open chat streams would otherwise hold a restart for systemd's 90 s stop timeout
    uvicorn.run(app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=5)
