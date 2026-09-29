"""Background jobs. Each job runs in its own worker process so the web server stays responsive
while Open3D is busy, and a running job can be cancelled."""
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path

import open3d as o3d

from ..clean import CleanParams, clean_point_cloud
from ..io import is_cloud, load, to_cloud
from ..mesh import MeshParams, reconstruct_mesh
from ..register import MergeParams, assessment_summary, merge_caveat, merge_geometries, transform_meshes
from ..texture import CameraView, TextureParams, colorize_mesh, save_textured
from .workspace import Workspace


# --------------------------------------------------------------------------- job bodies
def job_import(ws: Workspace, payload: dict, log) -> list[str]:
    path = Path(payload["path"])
    name = payload.get("name") or path.stem
    try:
        log(f"Reading {path.name}")
        geom = load(path)
        meta = ws.add_geometry(geom, name, "import", params={"source": payload.get("source", str(path))},
                               project=payload.get("project_id"))
        s = meta["stats"]
        log(f"Imported {meta['kind']} '{name}': {s.get('points', s.get('vertices')):,} points, "
            f"size {s['dimensions']}, spacing {s['spacing']}")
        return [meta["id"]]
    finally:
        if payload.get("temporary"):
            path.unlink(missing_ok=True)


def job_clean(ws: Workspace, payload: dict, log) -> list[str]:
    out = []
    for asset_id in payload["asset_ids"]:
        meta = ws.get(asset_id)
        geom = ws.load_geometry(asset_id)
        if not is_cloud(geom):
            log(f"'{meta['name']}' is a mesh - cleaning its vertices as points")
            geom = to_cloud(geom)
        params = CleanParams.preset(payload.get("preset", "standard"))
        params.update(payload.get("params", {}))
        cleaned, report = clean_point_cloud(geom, params, log)
        res = ws.add_geometry(cleaned, f"{meta['name']} · clean", "clean", [asset_id], params.to_dict(), report)
        out.append(res["id"])
    return out


def job_merge(ws: Workspace, payload: dict, log) -> list[str]:
    ids = payload["asset_ids"]
    metas = [ws.get(i) for i in ids]
    geoms = [ws.load_geometry(i) for i in ids]
    params = MergeParams.from_dict(payload.get("params", {}))
    # transforms: poses the user approved after a merge assessment - used as they are, never re-aligned
    merged, transforms, report = merge_geometries(geoms, params, payload.get("pairs"), log,
                                                  transforms=payload.get("transforms"))
    if payload.get("chosen_option"):  # the pose was picked from drawn options (compare_merge_options, photos)
        report["chosen_option"] = payload["chosen_option"]
    if payload.get("assessment"):  # the pre-merge check the user saw, kept so later measurements can warn
        report["assessment"] = assessment_summary(payload["assessment"], [m["name"] for m in metas])
    caveat = merge_caveat(report)
    if caveat:
        report["caveat"] = caveat
        log("Note: " + caveat["sentence"])
    name = payload.get("name") or f"merge of {len(ids)} scans"
    res = [ws.add_geometry(merged, name, "merge", ids, params.to_dict(), report)["id"]]
    if any(not is_cloud(g) for g in geoms):
        combined = transform_meshes(geoms, transforms)
        res.append(ws.add_geometry(combined, f"{name} · aligned meshes", "merge", ids, params.to_dict(), report)["id"])
        log("Also saved the aligned input meshes combined into one (not remeshed)")
    for m, info in zip(metas, report["scans"]):
        if "fitness" in info:
            log(f"  {m['name']}: fitness {info['fitness']:.3f}, rmse {info['rmse']:.5f}")
    return res


def job_mesh(ws: Workspace, payload: dict, log) -> list[str]:
    asset_id = payload["asset_id"]
    meta = ws.get(asset_id)
    params = MeshParams.from_dict(payload.get("params", {}))
    mesh, report = reconstruct_mesh(ws.load_geometry(asset_id), params, log)
    return [ws.add_geometry(mesh, f"{meta['name']} · mesh", "mesh", [asset_id], params.to_dict(), report)["id"]]


def job_texture(ws: Workspace, payload: dict, log) -> list[str]:
    asset_id = payload["asset_id"]
    meta = ws.get(asset_id)
    mesh = ws.load_geometry(asset_id)
    if is_cloud(mesh):
        raise ValueError("Colouring from photos needs a mesh - build one first")
    views = [CameraView.from_threejs(ws.image_path(v["image_id"]), v["view_matrix"], v["fov"], v.get("aspect"))
             for v in payload["views"]]
    params = TextureParams.from_dict(payload.get("params", {}))
    colored, report, textured = colorize_mesh(mesh, views, params, log)
    report["cameras"] = [v.to_dict() for v in views]
    image_ids = [v["image_id"] for v in payload["views"]]
    res = ws.add_geometry(colored, f"{meta['name']} · colour", "texture", [asset_id, *image_ids],
                          params.to_dict(), report)
    if textured is not None:
        d = ws.asset_dir(res["id"])
        save_textured(textured, d / "textured.glb")
        tex_dir = d / "textured_obj"
        save_textured(textured, tex_dir / "model.obj")
        ws.update(res["id"], textured=True)
        log("Saved UV-textured GLB and OBJ")
    return [res["id"]]


def job_pipeline(ws: Workspace, payload: dict, log) -> list[str]:
    ids = list(payload["asset_ids"])
    created = []
    if not payload.get("skip_clean"):
        cleaned = []
        for asset_id in ids:
            if ws.get(asset_id)["kind"] == "pointcloud":
                new = job_clean(ws, {"asset_ids": [asset_id], "preset": payload.get("preset", "standard"),
                                     "params": payload.get("clean", {})}, log)
                created += new
                cleaned += new
            else:
                cleaned.append(asset_id)
        ids = cleaned
    if len(ids) > 1:
        merged = job_merge(ws, {"asset_ids": ids, "params": payload.get("merge", {}),
                                "pairs": payload.get("pairs")}, log)
        created += merged
        ids = [merged[0]]
    created += job_mesh(ws, {"asset_id": ids[0], "params": payload.get("mesh", {})}, log)
    return created


JOBS = {"import": job_import, "clean": job_clean, "merge": job_merge, "mesh": job_mesh,
        "texture": job_texture, "pipeline": job_pipeline}

# Feature modules add job kinds by exposing a JOBS dict. They are imported lazily so the
# spawned worker process finds them too.
JOB_MODULES = ["cloudclean.web.jobs_edit", "cloudclean.web.jobs_compare", "cloudclean.web.jobs_autopilot",
               "cloudclean.web.jobs_merge", "cloudclean.web.jobs_understand", "cloudclean.web.jobs_accuracy",
               "cloudclean.web.jobs_holes", "cloudclean.web.jobs_merge_options", "cloudclean.web.jobs_photos3d",
               "cloudclean.web.jobs_colour_photos", "cloudclean.web.jobs_golden",
               "cloudclean.web.jobs_photo_fill"]


def job_function(kind: str):
    if kind in JOBS:
        return JOBS[kind]
    import importlib

    for name in JOB_MODULES:
        module = importlib.import_module(name)
        if kind in getattr(module, "JOBS", {}):
            return module.JOBS[kind]
    raise KeyError(f"Unknown job kind '{kind}'")


class JobLog:
    """Callable logger handed to job bodies. Also reports progress (0..1) and extra outputs."""

    def __init__(self, q):
        self.q = q

    def __call__(self, msg: str) -> None:
        self.q.put(("log", msg))

    def progress(self, fraction: float, label: str | None = None) -> None:
        self.q.put(("progress", {"fraction": max(0.0, min(1.0, float(fraction))), "label": label}))

    def output(self, **values) -> None:
        self.q.put(("output", values))


def _worker(root: str, kind: str, payload: dict, q) -> None:
    log = JobLog(q)
    try:
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            ids = job_function(kind)(Workspace(root), payload, log)
        q.put(("done", ids))
    except Exception as exc:
        q.put(("log", traceback.format_exc()))
        q.put(("error", str(exc)))


# --------------------------------------------------------------------------- manager
class JobManager:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.jobs: dict[str, dict] = {}
        self.pending: deque[str] = deque()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.process = None
        self.running_id = None
        self.on_submit = None   # (username, title) -> None: accounts count who started what (web.auth)
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, kind: str, title: str, payload: dict) -> dict:
        from .auth import current_user

        user = current_user.get()
        job = {"id": uuid.uuid4().hex[:10], "kind": kind, "title": title, "status": "queued",
               "created": time.time(), "started": None, "finished": None, "logs": [],
               "result": [], "error": None, "payload": payload, "progress": None, "output": {}, "user": user}
        if user and self.on_submit is not None:
            try:
                self.on_submit(user, title)
            except Exception:   # counting never stops a job
                pass
        with self.lock:
            self.jobs[job["id"]] = job
            self.pending.append(job["id"])
        self.wake.set()
        return job

    def list(self) -> list[dict]:
        with self.lock:
            return sorted(({**j, "logs": j["logs"][-400:]} for j in self.jobs.values()),
                          key=lambda j: j["created"], reverse=True)

    def get(self, job_id: str) -> dict:
        with self.lock:
            return dict(self.jobs[job_id])

    def wait(self, job_id: str, timeout: float | None = None, poll: float = 0.25) -> dict:
        """Block until a job finishes (used by the assistant and autopilot, never by request handlers)."""
        start = time.time()
        while True:
            job = self.get(job_id)
            if job["status"] in ("done", "failed", "cancelled"):
                return job
            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError(job["title"])
            time.sleep(poll)

    def cancel(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] == "queued":
                self.pending.remove(job_id)
                job["status"] = "cancelled"
                return
        if job["status"] == "running" and self.running_id == job_id and self.process is not None:
            job["cancel_requested"] = True
            self.process.terminate()

    def _log(self, job: dict, msg: str) -> None:
        elapsed = time.time() - (job["started"] or time.time())
        for line in str(msg).rstrip().splitlines():
            job["logs"].append(f"[{elapsed:6.1f}s] {line}")

    def _loop(self) -> None:
        ctx = mp.get_context("spawn")
        while True:
            with self.lock:
                job_id = self.pending.popleft() if self.pending else None
            if job_id is None:
                self.wake.wait(timeout=1.0)
                self.wake.clear()
                continue
            job = self.jobs[job_id]
            job["status"], job["started"] = "running", time.time()
            q = ctx.Queue()
            proc = ctx.Process(target=_worker, args=(str(self.workspace.root), job["kind"], job["payload"], q),
                               daemon=True)
            self.process, self.running_id = proc, job_id
            proc.start()
            finished = False
            while not finished:
                try:
                    kind, value = q.get(timeout=0.3)
                except queue.Empty:
                    if not proc.is_alive():
                        if job.get("cancel_requested"):
                            job["status"] = "cancelled"
                            self._log(job, "Cancelled")
                        else:
                            job["status"], job["error"] = "failed", f"worker exited (code {proc.exitcode})"
                        finished = True
                    continue
                if kind == "log":
                    self._log(job, value)
                elif kind == "progress":
                    job["progress"] = value
                elif kind == "output":
                    job["output"].update(value)
                elif kind == "done":
                    job["status"], job["result"] = "done", value
                    finished = True
                elif kind == "error":
                    job["status"], job["error"] = "failed", value
                    finished = True
            proc.join(timeout=5)
            job["finished"] = time.time()
            self.process, self.running_id = None, None
