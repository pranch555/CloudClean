"""On-disk workspace: every scan, intermediate result and photo is an 'asset' folder; assets are grouped into
projects (one physical part each).

Projects live in `<workspace>/projects.json` = [{id, name, created, updated, description, cover_asset_id,
golden_asset_id}].
Every asset meta has `project`: new assets take an explicit project, else the project of their first parent (so
every job keeps the project of its input without knowing about projects), else the most recently updated project
("My scans" is created when there is none). `migrate_projects()` puts assets without a (valid) project into
"My scans" (created once, found again by name). A project's `updated` is the later of its own edits and its
newest asset, computed on read, so worker processes never write projects.json while adding assets.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from PIL import Image

from ..io import HEIF_EXTS, describe, is_cloud, load, preview, save, to_jpeg
from ..texture import image_size

DEFAULT_PROJECT = "My scans"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_THUMBNAIL = 2 * 1024 * 1024
RAW_OPERATIONS = ("import", "capture")   # assets made by these are counted as scans


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _touch_time(projects: list[dict]) -> str:
    """Now, but strictly later than every project's `updated`, so the project touched last sorts first even when
    two changes land in the same millisecond."""
    latest = max((p.get("updated") or "" for p in projects), default="")
    t = _now()
    while t <= latest:
        time.sleep(0.001)
        t = _now()
    return t


@contextmanager
def _file_lock(path: Path, timeout: float = 15.0):
    """Cross-process lock (a lock file created exclusively). A lock older than 30 s is considered stale."""
    lock = path.with_name(path.name + ".lock")
    start = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 30:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"{path.name} is locked by another process")
            time.sleep(0.01)
    try:
        yield
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


def _focal_35mm(path: Path) -> int | None:
    try:
        with Image.open(path) as im:
            value = im.getexif().get_ifd(0x8769).get(0xA405)  # Exif FocalLengthIn35mmFilm
        return int(value) if value else None
    except Exception:
        return None


def safe_name(name: str) -> str:
    return re.sub(r"[^\w\-. ]+", "_", name).strip() or "asset"


class ProjectNotEmpty(ValueError):
    """Deleting a project that still holds assets (without delete_assets)."""


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.assets_dir = self.root / "assets"
        self.uploads_dir = self.root / "uploads"
        self.projects_file = self.root / "projects.json"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._meta_lock = threading.RLock()   # read-modify-write of meta.json within this process

    def asset_dir(self, asset_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", asset_id):
            raise KeyError(asset_id)
        return self.assets_dir / asset_id

    # ----------------------------------------------------------------- metadata
    def _write_meta(self, meta: dict) -> None:
        d = self.asset_dir(meta["id"])
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        os.replace(tmp, d / "meta.json")  # atomic: half-written assets never show up

    def get(self, asset_id: str) -> dict:
        path = self.asset_dir(asset_id) / "meta.json"
        if not path.exists():
            raise KeyError(asset_id)
        return json.loads(path.read_text())

    def list(self) -> list[dict]:
        metas = []
        for p in self.assets_dir.glob("*/meta.json"):
            try:
                metas.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(metas, key=lambda m: m["created"])

    def update(self, asset_id: str, **fields) -> dict:
        with self._meta_lock:
            meta = self.get(asset_id)
            meta.update(fields)
            self._write_meta(meta)
            return meta

    def report(self, asset_id: str) -> dict | None:
        path = self.asset_dir(asset_id) / "report.json"
        return json.loads(path.read_text()) if path.exists() else None

    def delete(self, asset_id: str) -> None:
        shutil.rmtree(self.asset_dir(asset_id), ignore_errors=True)

    # ----------------------------------------------------------------- creation
    def _new(self) -> tuple[str, Path]:
        asset_id = uuid.uuid4().hex[:12]
        d = self.asset_dir(asset_id)
        d.mkdir(parents=True)
        return asset_id, d

    def add_geometry(self, geom, name: str, operation: str, parents=(), params=None, report=None,
                     project: str | None = None) -> dict:
        """Store a new asset. `project` = explicit project id; else the first parent's project; else the most
        recently updated project. Meta `part` holds the part frame and dimensions (cloudclean.regions)."""
        project_id = self._project_for_new(project, parents)
        asset_id, d = self._new()
        save(geom, d / "data.ply")
        save(preview(geom), d / "preview.ply")
        if report is not None:
            (d / "report.json").write_text(json.dumps(report, indent=2, default=str))
        meta = {"id": asset_id, "name": name, "kind": "pointcloud" if is_cloud(geom) else "mesh",
                "created": _now(), "operation": operation, "parents": list(parents), "params": params or {},
                "stats": describe(geom), "textured": False, "project": project_id}
        try:  # part frame + dimensions along the part's own axes: cheap, and never allowed to fail an import
            from ..regions import part_frame

            st = (d / "data.ply").stat()
            meta["part"] = {**part_frame(geom).info(), "source": {"mtime_ns": st.st_mtime_ns, "size": st.st_size}}
        except Exception:
            pass
        self._write_meta(meta)
        return meta

    def add_image(self, src: Path, name: str, project: str | None = None) -> dict:
        converted = None
        if src.suffix.lower() in HEIF_EXTS:  # phone photos: stored as JPEG so every browser can show them
            converted = src = to_jpeg(src, self.uploads_dir / f"{uuid.uuid4().hex}.jpg")
        asset_id, d = self._new()
        dst = d / f"image{src.suffix.lower()}"
        shutil.copyfile(src, dst)
        if converted:
            converted.unlink(missing_ok=True)
        w, h = image_size(dst)
        stats = {"width": w, "height": h}
        focal = _focal_35mm(dst)
        if focal:
            stats["focal_35mm"] = focal  # lets the alignment view start with the right lens
        meta = {"id": asset_id, "name": name, "kind": "image", "created": _now(), "operation": "import",
                "parents": [], "params": {}, "stats": stats, "file": dst.name,
                "project": self._project_for_new(project, ())}
        self._write_meta(meta)
        return meta

    def load_geometry(self, asset_id: str):
        meta = self.get(asset_id)
        if meta["kind"] == "image":
            raise ValueError(f"'{meta['name']}' is a photo, not a scan")
        return load(self.asset_dir(asset_id) / "data.ply")

    # ----------------------------------------------------------------- per-point scalars
    def add_scalars(self, asset_id: str, name: str, values, unit: str = "", description: str = "") -> dict:
        """Store one value per point (clouds) or vertex (meshes) of data.ply, e.g. deviation from CAD.

        Also writes a float32 copy aligned with preview.ply so the browser can colour the preview.
        NaN marks points without a value."""
        import numpy as np
        from scipy.spatial import cKDTree

        if not re.fullmatch(r"[a-z0-9_]{1,40}", name):
            raise ValueError("scalar name must be lowercase letters, digits or _")
        d = self.asset_dir(asset_id)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        full = load(d / "data.ply")
        full_pts = np.asarray(full.points if is_cloud(full) else full.vertices)
        if len(values) != len(full_pts):
            raise ValueError(f"{len(values)} values for {len(full_pts)} points")
        prev = load(d / "preview.ply")
        prev_pts = np.asarray(prev.points if is_cloud(prev) else prev.vertices)
        if len(prev_pts) == len(full_pts):
            preview_values = values
        else:  # preview is a subsample / decimation: take the value of the nearest full-resolution point
            _, nn = cKDTree(full_pts).query(prev_pts, k=1, workers=-1)
            preview_values = values[nn]
        np.save(d / f"scalars_{name}.npy", values)
        np.ascontiguousarray(preview_values, dtype="<f4").tofile(d / f"scalars_{name}.preview.bin")
        finite = values[np.isfinite(values)]
        info = {"name": name, "unit": unit, "description": description,
                "min": float(finite.min()) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None}
        meta = self.get(asset_id)
        scalars = [x for x in meta.get("scalars", []) if x["name"] != name] + [info]
        self.update(asset_id, scalars=scalars)
        return info

    def scalars_path(self, asset_id: str, name: str, preview: bool = True) -> Path:
        if not re.fullmatch(r"[a-z0-9_]{1,40}", name):
            raise KeyError(name)
        d = self.asset_dir(asset_id)
        path = d / (f"scalars_{name}.preview.bin" if preview else f"scalars_{name}.npy")
        if not path.exists():
            raise KeyError(name)
        return path

    def image_path(self, asset_id: str) -> Path:
        meta = self.get(asset_id)
        if meta["kind"] != "image":
            raise ValueError(f"'{meta['name']}' is not a photo")
        return self.asset_dir(asset_id) / meta["file"]

    # ----------------------------------------------------------------- thumbnails
    def set_thumbnail(self, asset_id: str, png: bytes) -> dict:
        """Store a browser-rendered PNG thumbnail (<= 2 MB); meta gets has_thumbnail = true."""
        self.get(asset_id)
        if not png.startswith(PNG_MAGIC):
            raise ValueError("The thumbnail must be a PNG image")
        if len(png) > MAX_THUMBNAIL:
            raise ValueError(f"The thumbnail is too large ({len(png):,} bytes; at most 2 MB)")
        d = self.asset_dir(asset_id)
        tmp = d / "thumbnail.png.tmp"
        tmp.write_bytes(png)
        os.replace(tmp, d / "thumbnail.png")
        return self.update(asset_id, has_thumbnail=True)

    def thumbnail_path(self, asset_id: str) -> Path:
        path = self.asset_dir(asset_id) / "thumbnail.png"
        if not path.exists():
            raise KeyError(asset_id)
        return path

    # ----------------------------------------------------------------- projects
    def _read_projects(self) -> list[dict]:
        try:
            data = json.loads(self.projects_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [p for p in data if isinstance(p, dict) and isinstance(p.get("id"), str)] if isinstance(data, list) \
            else []

    def _write_projects(self, projects: list[dict]) -> None:
        tmp = self.projects_file.with_name(f"projects.json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(projects, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.projects_file)

    @contextmanager
    def _projects_txn(self):
        """Read-modify-write of projects.json under a cross-process lock; yields the list to modify in place."""
        with self._meta_lock, _file_lock(self.projects_file):
            projects = self._read_projects()
            yield projects
            self._write_projects(projects)

    def projects(self) -> list[dict]:
        """Stored projects (without counts), in creation order."""
        return self._read_projects()

    def get_project(self, project_id: str) -> dict:
        for p in self._read_projects():
            if p["id"] == project_id:
                return p
        raise KeyError(project_id)

    def create_project(self, name: str, description: str = "") -> dict:
        name = " ".join(str(name or "").split())[:120]
        if not name:
            raise ValueError("Give the project a name")
        now = _now()
        project = {"id": uuid.uuid4().hex[:12], "name": name, "created": now, "updated": now,
                   "description": str(description or "")[:2000], "cover_asset_id": None, "golden_asset_id": None}
        with self._projects_txn() as projects:
            projects.append(project)
        return project

    def update_project(self, project_id: str, **fields) -> dict:
        """Change name / description / cover_asset_id / golden_asset_id (an asset of this project, or None; the
        golden model must be a mesh)."""
        unknown = set(fields) - {"name", "description", "cover_asset_id", "golden_asset_id"}
        if unknown:
            raise ValueError(f"Unknown project field(s): {', '.join(sorted(unknown))}")
        if "name" in fields:
            fields["name"] = " ".join(str(fields["name"] or "").split())[:120]
            if not fields["name"]:
                raise ValueError("The project name must not be empty")
        if "description" in fields:
            fields["description"] = str(fields["description"] or "")[:2000]
        for key in ("cover_asset_id", "golden_asset_id"):
            asset = fields.get(key)
            if asset is None:
                continue
            try:
                meta = self.get(str(asset))
            except KeyError:
                raise ValueError(f"No asset {asset}")
            if meta.get("project") != project_id:
                raise ValueError(f"'{meta['name']}' is not in this project")
            if key == "golden_asset_id" and meta.get("kind") != "mesh":
                raise ValueError(f"'{meta['name']}' is not a mesh: the golden model must be a CAD model or a mesh")
        with self._projects_txn() as projects:
            for p in projects:
                if p["id"] == project_id:
                    p.update(fields)
                    p["updated"] = _touch_time(projects)
                    return dict(p)
        raise KeyError(project_id)

    def delete_project(self, project_id: str, delete_assets: bool = False) -> dict:
        """Remove a project; refuses (ProjectNotEmpty) while it holds assets unless delete_assets is true."""
        self.get_project(project_id)
        assets = [m for m in self.list() if m.get("project") == project_id]
        if assets and not delete_assets:
            raise ProjectNotEmpty(f"The project still holds {len(assets)} asset(s) - move or delete them first")
        for m in assets:
            self.delete(m["id"])
        with self._projects_txn() as projects:
            projects[:] = [p for p in projects if p["id"] != project_id]
        return {"deleted": project_id, "deleted_assets": [m["id"] for m in assets]}

    def move_asset(self, asset_id: str, project_id: str) -> dict:
        self.get_project(project_id)
        meta = self.update(asset_id, project=project_id)
        with self._projects_txn() as projects:
            stamp = _touch_time(projects)
            for p in projects:
                if p["id"] == project_id:
                    p["updated"] = stamp
        return meta

    @staticmethod
    def asset_class(meta: dict) -> str:
        """photos (images), scans (anything imported or captured, meshes included), meshes (made in CloudClean) or
        results (other derived point clouds)."""
        if meta.get("kind") == "image":
            return "photos"
        if meta.get("operation") in RAW_OPERATIONS:
            return "scans"
        return "meshes" if meta.get("kind") == "mesh" else "results"

    def project_summaries(self, metas: list[dict] | None = None) -> list[dict]:
        """Projects with counts {scans, meshes, results, photos, total}; `updated` includes the newest asset;
        a cover that is no longer in the project reads as null. Newest first."""
        metas = self.list() if metas is None else metas
        out = []
        for p in self._read_projects():
            assets = [m for m in metas if m.get("project") == p["id"]]
            counts = {"scans": 0, "meshes": 0, "results": 0, "photos": 0, "total": len(assets)}
            for m in assets:
                counts[self.asset_class(m)] += 1
            updated = max([p.get("updated") or p.get("created") or ""] + [m.get("created") or "" for m in assets])
            cover, golden = p.get("cover_asset_id"), p.get("golden_asset_id")
            if cover and not any(m["id"] == cover for m in assets):
                cover = None
            if golden and not any(m["id"] == golden for m in assets):
                golden = None
            out.append({"id": p["id"], "name": p.get("name", ""), "created": p.get("created"), "updated": updated,
                        "description": p.get("description", ""), "cover_asset_id": cover,
                        "golden_asset_id": golden, "counts": counts})
        return sorted(out, key=lambda p: p["updated"] or "", reverse=True)

    def default_project_id(self, metas: list[dict] | None = None) -> str:
        """The most recently updated project; "My scans" is created when there is none."""
        summaries = self.project_summaries(metas)
        if summaries:
            return summaries[0]["id"]
        return self._ensure_default_project()

    def _ensure_default_project(self) -> str:
        with self._projects_txn() as projects:
            for p in projects:
                if p.get("name") == DEFAULT_PROJECT:
                    return p["id"]
            now = _now()
            project = {"id": uuid.uuid4().hex[:12], "name": DEFAULT_PROJECT, "created": now, "updated": now,
                       "description": "Scans from before projects existed, and anything not filed elsewhere.",
                       "cover_asset_id": None, "golden_asset_id": None}
            projects.append(project)
            return project["id"]

    def _project_for_new(self, project: str | None, parents) -> str:
        known = {p["id"] for p in self._read_projects()}
        if project and project in known:
            return project
        for parent in parents or ():
            try:
                pid = self.get(parent).get("project")
            except KeyError:
                continue
            if pid in known:
                return pid
        return self.default_project_id()

    def migrate_projects(self) -> dict:
        """Put every asset without a (known) project into "My scans". Returns {assigned, project_id}."""
        metas = self.list()
        known = {p["id"] for p in self._read_projects()}
        orphans = [m for m in metas if m.get("project") not in known]
        if not orphans:
            return {"assigned": 0, "project_id": None}
        pid = self._ensure_default_project()
        for m in orphans:
            try:
                self.update(m["id"], project=pid)
            except (KeyError, OSError):
                continue
        return {"assigned": len(orphans), "project_id": pid}
