"""Make a 3D model from photos (job photos_to_3d, routes, assistant tool) with a fake `docker run`: no Docker, GPU or
network needed. The fake reads the photos from the /in mount and writes /out like tools/recon/photos_to_3d.py."""
import asyncio
import json
import subprocess
import uuid
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from cloudclean.web import jobs_photos3d as photos3d
from cloudclean.web import routes_photos3d
from cloudclean.web.workspace import Workspace

MODEL = "facebook/map-anything-apache"
SIZE_MM = np.array([40.0, 30.0, 20.0])
NOT_BUILT = "The reconstruction image cloudclean-recon:latest is not built yet - see docs/photos-to-3d.md"


# --------------------------------------------------------------------------- helpers
def add_photos(ws, tmp_path, n, project=None, ext=".png"):
    ids = []
    for k in range(n):
        src = tmp_path / f"{uuid.uuid4().hex[:8]}{ext}"
        Image.new("RGB", (64, 48), (30 + 50 * k % 220, 90, 160)).save(src)
        ids.append(ws.add_image(src, f"bracket {k}", project=project)["id"])
    return ids


def cloud(n=300, seed=0):
    pts = np.random.default_rng(seed).uniform(0, 10, (n, 3))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))


def volumes(cmd) -> dict[str, Path]:
    """{container folder: host folder} of the `-v host:container[:ro]` options (a host path may hold a drive colon)."""
    out = {}
    for flag, spec in zip(cmd, cmd[1:]):
        if flag == "-v":
            host, _, target = spec.rpartition(":")
            if target in ("ro", "rw"):
                host, _, target = host.rpartition(":")
            out[target] = Path(host)
    return out


def write_outputs(out: Path, names: list[str], n: int) -> None:
    rng = np.random.default_rng(3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3)) * SIZE_MM))
    pcd.colors = o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3)))
    o3d.io.write_point_cloud(str(out / "points.ply"), pcd)
    cameras = [{"image": name, "cam_to_world_mm": np.eye(4).tolist(),
                "intrinsics": [[480.0, 0.0, 256.0], [0.0, 480.0, 192.0], [0.0, 0.0, 1.0]], "size": [512, 384],
                "points": n // len(names), "scale": 1.0} for name in names]
    (out / "cameras.json").write_text(json.dumps({"model": MODEL, "cameras": cameras}), encoding="utf-8")


def ok_lines(n_photos, n_points=400):
    return ["some docker noise",
            "PROGRESS 0.050 Loading the reconstruction model",
            f"PROGRESS 0.300 Reconstructing from {n_photos} photos",
            "PROGRESS 0.850 Collecting the points",
            "PROGRESS 1.000 Done",
            "RESULT " + json.dumps({"photos": n_photos, "points": n_points, "extent_mm": [39.2, 29.4, 19.6],
                                    "seconds": 42.0, "model": MODEL, "poses": "colmap", "registered": n_photos})]


def fake_docker(monkeypatch, lines, code=0, n_points=400) -> dict:
    """Replaces subprocess.Popen for the job; returns what the fake container saw (cmd, mounts, photos)."""
    seen = {"runs": 0}

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            mounts = volumes(cmd)
            photos = sorted(mounts["/in"].iterdir())
            seen.update(runs=seen["runs"] + 1, cmd=list(cmd), kwargs=kwargs, mounts=mounts,
                        photos=[p.name for p in photos], photo_bytes=[p.read_bytes() for p in photos])
            if code == 0:
                write_outputs(mounts["/out"], seen["photos"], n_points)
            self.stdout = iter([line + "\n" for line in lines])

        def wait(self, timeout=None):
            return code

        def poll(self):
            return code

    monkeypatch.setattr(photos3d.subprocess, "Popen", FakePopen)
    return seen


class Log:
    def __init__(self):
        self.messages, self.steps = [], []

    def __call__(self, msg):
        self.messages.append(msg)

    def progress(self, fraction, label=None):
        self.steps.append((fraction, label))


class Jobs:
    """JobManager stand-in. run=True runs the job right away in this process (the real manager spawns a worker
    process, where the monkeypatched Popen would not apply); run=False only records what was submitted."""

    def __init__(self, ws, run=True):
        self.ws, self.run, self.jobs = ws, run, {}

    def submit(self, kind, title, payload):
        from cloudclean.web.jobs import job_function

        job = {"id": f"job{len(self.jobs)}", "kind": kind, "title": title, "payload": payload, "status": "queued",
               "logs": [], "progress": None, "result": [], "error": None, "output": {}}
        if self.run:
            log = Log()
            try:
                job.update(status="done", result=job_function(kind)(self.ws, payload, log))
            except Exception as exc:
                job.update(status="failed", error=str(exc))
            job["logs"] = log.messages
            job["progress"] = {"fraction": log.steps[-1][0], "label": log.steps[-1][1]} if log.steps else None
        self.jobs[job["id"]] = job
        return dict(job)

    def get(self, job_id):
        return dict(self.jobs[job_id])


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws")


@pytest.fixture
def recon_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(photos3d, "recon_available", lambda: (True, "ready"))
    monkeypatch.setattr(photos3d, "RECON_CACHE", tmp_path / "hf")   # not the real ~/cloudclean-deploy
    removed = []   # docker rm -f of leftover / cancelled containers: recorded, never run
    monkeypatch.setattr(photos3d, "_remove_leftovers", lambda: removed.append("leftovers"))
    monkeypatch.setattr(photos3d, "_remove_container", removed.append)
    return removed


# --------------------------------------------------------------------------- recon_available
def test_recon_available_needs_docker_and_the_image(monkeypatch):
    def no_run(*a, **k):
        raise AssertionError("docker must not be asked when it is not installed")

    monkeypatch.setattr(photos3d.shutil, "which", lambda name: None)
    monkeypatch.setattr(photos3d.subprocess, "run", no_run)
    ok, why = photos3d.recon_available()
    assert ok is False and "DGX Spark" in why

    calls = []
    monkeypatch.setattr(photos3d.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(photos3d.subprocess, "run",
                        lambda cmd, **k: calls.append(cmd) or subprocess.CompletedProcess(cmd, 1, b"", b"No such image"))
    ok, why = photos3d.recon_available()
    assert ok is False and photos3d.RECON_IMAGE in why and "docs/photos-to-3d.md" in why
    assert calls == [["docker", "image", "inspect", photos3d.RECON_IMAGE]]

    def hangs(cmd, **k):
        raise subprocess.TimeoutExpired(cmd, 20)

    monkeypatch.setattr(photos3d.subprocess, "run", hangs)
    ok, why = photos3d.recon_available()
    assert ok is False and "Docker did not answer" in why

    monkeypatch.setattr(photos3d.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, b"[]", b""))
    assert photos3d.recon_available() == (True, "ready")


# --------------------------------------------------------------------------- photos_of
def test_photos_of_takes_the_project_photos_or_the_given_ids(ws, tmp_path):
    bracket, other = ws.create_project("Bracket"), ws.create_project("Other part")
    mine = add_photos(ws, tmp_path, 2, bracket["id"])
    theirs = add_photos(ws, tmp_path, 1, other["id"])
    scan = ws.add_geometry(cloud(), "bracket scan", "import", project=bracket["id"])
    cad = ws.add_geometry(o3d.geometry.TriangleMesh.create_box(10, 20, 30), "bracket cad", "import",
                          project=bracket["id"])

    assert {m["id"] for m in photos3d.photos_of(ws, None, bracket["id"])} == set(mine)   # scans are not photos
    assert {m["id"] for m in photos3d.photos_of(ws, [], bracket["id"])} == set(mine)
    with pytest.raises(ValueError, match="several projects"):   # no project: photos of two parts are not mixed
        photos3d.photos_of(ws, None, None)
    chosen = photos3d.photos_of(ws, [theirs[0], mine[1]], bracket["id"])   # explicit ids win, in their order
    assert [m["id"] for m in chosen] == [theirs[0], mine[1]]

    with pytest.raises(ValueError, match="Not photos: bracket scan"):
        photos3d.photos_of(ws, [mine[0], scan["id"]], None)
    with pytest.raises(ValueError, match="bracket cad"):
        photos3d.photos_of(ws, [cad["id"], mine[0]], None)
    with pytest.raises(KeyError):
        photos3d.photos_of(ws, [mine[0], "0123456789ab"], None)


# --------------------------------------------------------------------------- the job
def test_job_makes_a_coloured_model_from_the_photos(ws, tmp_path, monkeypatch, recon_ready):
    project = ws.create_project("Bracket")
    ids = add_photos(ws, tmp_path, 2, project["id"]) + add_photos(ws, tmp_path, 1, project["id"], ext=".jpg")
    ids = [ids[2], ids[0], ids[1]]   # the given order is kept
    seen = fake_docker(monkeypatch, ok_lines(3))
    log = Log()

    created = photos3d.job_photos_to_3d(ws, {"photo_ids": ids}, log)

    # what ran: the recon image, GPU, photos read-only at /in, the script from tools/recon at /code
    cmd = seen["cmd"]
    assert seen["runs"] == 1 and cmd[:2] == ["docker", "run"] and "--gpus" in cmd
    assert photos3d.RECON_IMAGE in cmd
    assert cmd[-7:] == ["python3", "/code/photos_to_3d.py", "--images", "/in", "--out", "/out", "--scale-sheet"]
    assert any(spec.endswith(":/in:ro") for spec in cmd)
    assert seen["mounts"]["/code"] == photos3d.SCRIPT_DIR and (photos3d.SCRIPT_DIR / "photos_to_3d.py").is_file()
    assert seen["mounts"]["/hf"] == tmp_path / "hf" and (tmp_path / "hf").is_dir()
    assert seen["kwargs"]["stdout"] == subprocess.PIPE and seen["kwargs"]["stderr"] == subprocess.STDOUT
    # every photo was copied, in order, unchanged
    assert seen["photos"] == ["000.jpg", "001.png", "002.png"]
    assert seen["photo_bytes"] == [ws.image_path(i).read_bytes() for i in ids]

    # the new model: a point cloud whose parents are the photos, in the photos' project
    assert len(created) == 1
    meta = ws.get(created[0])
    assert meta["kind"] == "pointcloud" and meta["operation"] == "photos"
    assert meta["parents"] == ids and meta["project"] == project["id"]
    assert meta["name"] == f"{ws.get(ids[0])['name']} · from 3 photos"
    assert meta["params"] == {"photos": 3, "model": MODEL, "poses": "colmap"}
    model = ws.load_geometry(created[0])
    pts = np.asarray(model.points)
    assert len(pts) == 400 and model.has_colors()
    assert np.all(pts.min(axis=0) >= 0) and np.all(pts.max(axis=0) <= SIZE_MM + 1e-3)   # mm, not rescaled
    assert np.all(pts.max(axis=0) > 0.9 * SIZE_MM)

    report = ws.report(created[0])
    assert report["photos"] == 3 and report["points"] == 400 and report["model"] == MODEL
    assert report["extent_mm"] == [39.2, 29.4, 19.6] and report["seconds"] == 42.0
    assert report["scale"].startswith("Size unknown: no scale sheet found")
    assert "unknown" in report["scale"] and "known length" in report["scale"] and report["poses"] == "colmap"
    assert [c["image"] for c in report["cameras"]] == seen["photos"]

    # progress only goes up, from the start to done, with the script's labels
    fractions = [f for f, _ in log.steps]
    assert fractions == sorted(fractions) and len(set(fractions)) == len(fractions)
    assert fractions[0] == pytest.approx(0.02) and fractions[-1] == 1.0
    assert "Reconstructing from 3 photos" in [label for _, label in log.steps]
    assert any("400 points from 3 of 3 photos" in m and "COLMAP" in m for m in log.messages)

    # the work folder is gone
    assert list((ws.root / "recon").iterdir()) == []
    assert not seen["mounts"]["/in"].exists() and not seen["mounts"]["/out"].exists()


def test_job_takes_every_photo_of_the_project_and_a_name(ws, tmp_path, monkeypatch, recon_ready):
    bracket, other = ws.create_project("Bracket"), ws.create_project("Other part")
    ids = add_photos(ws, tmp_path, 4, bracket["id"])
    add_photos(ws, tmp_path, 2, other["id"])
    seen = fake_docker(monkeypatch, ok_lines(4))
    created = photos3d.job_photos_to_3d(ws, {"project_id": bracket["id"], "name": "Bracket from photos"}, Log())
    meta = ws.get(created[0])
    assert len(seen["photos"]) == 4 and sorted(meta["parents"]) == sorted(ids)
    assert meta["name"] == "Bracket from photos" and meta["project"] == bracket["id"]


def test_job_reports_the_container_error_and_cleans_up(ws, tmp_path, monkeypatch, recon_ready):
    ids = add_photos(ws, tmp_path, 3)
    seen = fake_docker(monkeypatch, ["PROGRESS 0.050 Loading the reconstruction model", "ERROR CUDA out of memory"],
                       code=1)
    log = Log()
    with pytest.raises(ValueError, match="CUDA out of memory") as err:
        photos3d.job_photos_to_3d(ws, {"photo_ids": ids}, log)
    assert "exit 1" in str(err.value)
    assert "CUDA out of memory" in log.messages
    assert list((ws.root / "recon").iterdir()) == [] and not seen["mounts"]["/in"].exists()
    assert [m["kind"] for m in ws.list()] == ["image"] * 3   # no model was added


def test_job_without_an_error_line_shows_the_last_output(ws, tmp_path, monkeypatch, recon_ready):
    ids = add_photos(ws, tmp_path, 2)
    fake_docker(monkeypatch, ["Traceback (most recent call last):", '  File "/code/photos_to_3d.py", line 63',
                              "RuntimeError: boom"], code=1)
    with pytest.raises(ValueError, match="RuntimeError: boom"):
        photos3d.job_photos_to_3d(ws, {"photo_ids": ids}, Log())
    fake_docker(monkeypatch, ["PROGRESS 1.000 Done"], code=0, n_points=10)   # exit 0 but no RESULT line
    with pytest.raises(ValueError, match="did not finish"):
        photos3d.job_photos_to_3d(ws, {"photo_ids": ids}, Log())
    assert list((ws.root / "recon").iterdir()) == []


def test_job_needs_two_photos_and_the_container(ws, tmp_path, monkeypatch):
    def no_docker(*a, **k):
        raise AssertionError("docker must not run")

    monkeypatch.setattr(photos3d.subprocess, "Popen", no_docker)
    monkeypatch.setattr(photos3d, "recon_available", lambda: (True, "ready"))
    project = ws.create_project("Bracket")
    with pytest.raises(ValueError, match="at least 2 photos"):
        photos3d.job_photos_to_3d(ws, {"project_id": project["id"]}, Log())
    one = add_photos(ws, tmp_path, 1, project["id"])
    with pytest.raises(ValueError, match="at least 2 photos"):
        photos3d.job_photos_to_3d(ws, {"photo_ids": one}, Log())

    ids = one + add_photos(ws, tmp_path, 1, project["id"])
    monkeypatch.setattr(photos3d, "recon_available", lambda: (False, NOT_BUILT))
    with pytest.raises(ValueError, match="not built yet"):
        photos3d.job_photos_to_3d(ws, {"photo_ids": ids}, Log())
    assert not (ws.root / "recon").exists()


SHEET_FOUND = {"found": True, "dictionary": "DICT_4X4_50", "marker_mm": 20.0, "ruler_mm": 100.2, "markers": 14,
               "views": 24, "reprojection_px": 0.21, "residual_mm": 0.06, "max_residual_mm": 0.12,
               "uncertainty_pct": 0.034, "mm_per_unit": 57.3, "summary": "True size from the scale sheet (±0.034 %)"}


def test_job_on_the_scale_sheet_is_true_size(ws, tmp_path, monkeypatch, recon_ready):
    """With the scale sheet in the photos the script works in true mm: the job says so, with the uncertainty, and
    passes on what the 100 mm bar measured."""
    ids = add_photos(ws, tmp_path, 3)
    lines = ok_lines(3)
    result = json.loads(lines[-1][len("RESULT "):])
    lines[-1] = "RESULT " + json.dumps({**result, "scale_sheet": SHEET_FOUND})
    seen = fake_docker(monkeypatch, ["LOG True size from the scale sheet (±0.034 %): 14 markers in 24 photos"] + lines)
    log = Log()
    created = photos3d.job_photos_to_3d(ws, {"photo_ids": ids, "ruler_mm": 100.2}, log)
    assert seen["cmd"][-3:] == ["--scale-sheet", "--ruler-mm", "100.2"]
    meta = ws.get(created[0])
    assert meta["params"] == {"photos": 3, "model": MODEL, "poses": "colmap", "size": "scale sheet",
                              "size_uncertainty_pct": 0.034, "ruler_mm": 100.2}
    report = ws.report(created[0])
    assert report["scale"] == "True size from the scale sheet (±0.034 %)"
    assert report["scale_sheet"]["markers"] == 14 and report["scale_sheet"]["residual_mm"] == 0.06
    assert "True size from the scale sheet (±0.034 %)" in log.messages

    # found, but the print was not checked: said so (the photos cannot see a printer that scaled the page)
    unchecked = {**SHEET_FOUND, "ruler_mm": None}
    assert photos3d.size_sentence(unchecked) == ("True size from the scale sheet (±0.034 %); the print was not "
                                                 "checked with its 100 mm bar")

    # the sheet looked for but not found: the size stays unknown, with the reason
    lines[-1] = "RESULT " + json.dumps({**result, "scale_sheet": {"found": False, "reason": "no scale sheet markers "
                                                                                          "in the photos"}})
    fake_docker(monkeypatch, lines)
    created = photos3d.job_photos_to_3d(ws, {"photo_ids": ids}, Log())
    report = ws.report(created[0])
    assert report["scale"].startswith("Size unknown: no scale sheet found (no scale sheet markers in the photos)")
    assert ws.get(created[0])["params"] == {"photos": 3, "model": MODEL, "poses": "colmap"}


def test_a_bar_that_cannot_be_100_mm_is_refused(ws, tmp_path, monkeypatch, recon_ready):
    def no_docker(*a, **k):
        raise AssertionError("docker must not run")

    monkeypatch.setattr(photos3d.subprocess, "Popen", no_docker)
    ids = add_photos(ws, tmp_path, 2)
    for bad in (10.0, 1000, "a lot"):
        with pytest.raises(ValueError, match="100 mm bar"):
            photos3d.job_photos_to_3d(ws, {"photo_ids": ids, "ruler_mm": bad}, Log())
    assert photos3d.check_ruler(None) is None and photos3d.check_ruler(0) is None
    assert photos3d.check_ruler("99.8") == 99.8

    app = FastAPI()
    app.include_router(routes_photos3d.create_router(ws, Jobs(ws, run=False)))
    res = TestClient(app).post("/api/photos-to-3d", json={"photo_ids": ids, "ruler_mm": 50})
    assert res.status_code == 400 and "100 mm bar" in res.json()["detail"]


def test_job_and_routes_are_registered():
    from cloudclean.web.jobs import job_function
    from cloudclean.web.server import ROUTER_MODULES

    assert job_function("photos_to_3d") is photos3d.job_photos_to_3d
    assert "cloudclean.web.routes_photos3d" in ROUTER_MODULES


# --------------------------------------------------------------------------- routes
def test_routes(ws, tmp_path, monkeypatch):
    bracket, other = ws.create_project("Bracket"), ws.create_project("Other part")
    jobs = Jobs(ws, run=False)
    app = FastAPI()
    app.include_router(routes_photos3d.create_router(ws, jobs))
    client = TestClient(app)
    start = {"project_id": bracket["id"]}

    monkeypatch.setattr(photos3d.shutil, "which", lambda name: None)   # no Docker on this machine
    status = client.get("/api/photos-to-3d/status").json()
    assert status["available"] is False and "DGX Spark" in status["reason"]

    res = client.post("/api/photos-to-3d", json=start)
    assert res.status_code == 400 and "at least 2 photos" in res.json()["detail"]
    ids = add_photos(ws, tmp_path, 1, bracket["id"])
    add_photos(ws, tmp_path, 3, other["id"])   # another part's photos do not count
    assert client.post("/api/photos-to-3d", json=start).status_code == 400
    ids += add_photos(ws, tmp_path, 1, bracket["id"])
    scan = ws.add_geometry(cloud(), "bracket scan", "import", project=bracket["id"])
    res = client.post("/api/photos-to-3d", json={"photo_ids": [ids[0], scan["id"]]})
    assert res.status_code == 400 and "bracket scan" in res.json()["detail"]
    assert client.post("/api/photos-to-3d", json={"photo_ids": [ids[0], "0123456789ab"]}).status_code == 404

    monkeypatch.setattr(routes_photos3d, "recon_available", lambda: (False, NOT_BUILT))
    res = client.post("/api/photos-to-3d", json=start)
    assert res.status_code == 503 and res.json()["detail"] == NOT_BUILT
    assert not jobs.jobs

    monkeypatch.setattr(routes_photos3d, "recon_available", lambda: (True, "ready"))
    assert client.get("/api/photos-to-3d/status").json() == {"available": True, "reason": "ready"}
    res = client.post("/api/photos-to-3d", json={**start, "name": "Bracket from photos"})
    assert res.status_code == 200, res.text
    job = res.json()
    assert job["kind"] == "photos_to_3d" and job["title"] == "3D model from 2 photos" and job["status"] == "queued"
    assert sorted(job["payload"]["photo_ids"]) == sorted(ids)
    assert job["payload"]["project_id"] == bracket["id"] and job["payload"]["name"] == "Bracket from photos"
    res = client.post("/api/photos-to-3d", json={"photo_ids": ids[::-1]})
    assert res.status_code == 200 and res.json()["payload"]["photo_ids"] == ids[::-1]


# --------------------------------------------------------------------------- assistant
def call(ws, args, ui, jobs=None):
    from cloudclean.assistant.tools import ToolContext, call_tool

    events = []
    ctx = ToolContext(ws, jobs, {}, ui, lambda e, d: events.append((e, d)), "conv", "call", poll_interval=0.01)
    return asyncio.run(call_tool(ctx, "photos_to_3d", args)), events


def test_assistant_tool_and_guide(ws, tmp_path, monkeypatch):
    from cloudclean.assistant.guide import BY_ID, search
    from cloudclean.assistant.tools import TOOLS, ToolError

    assert set(TOOLS["photos_to_3d"].schema()["function"]["parameters"]["properties"]) == {"photo_ids", "name", "ruler_mm"}
    assert "photos-to-3d" in [f.id for _, f in search("photos 3d")]
    assert search("make a 3D model from photos")[0][1].id == "photos-to-3d"
    assert search("photogrammetry")[0][1].id == "photos-to-3d"
    assert BY_ID["photos-to-3d"].nav["step"] == "capture"

    project = ws.create_project("Bracket")
    ui = {"project_id": project["id"]}
    with pytest.raises(ToolError, match="fewer than 2 photos"):
        call(ws, {}, ui)
    ids = add_photos(ws, tmp_path, 1, project["id"])
    with pytest.raises(ToolError, match="fewer than 2 photos"):
        call(ws, {}, ui)
    ids += add_photos(ws, tmp_path, 1, project["id"])
    scan = ws.add_geometry(cloud(), "bracket scan", "import", project=project["id"])
    with pytest.raises(ToolError, match="Not photos: bracket scan"):
        call(ws, {"photo_ids": [ids[0], scan["id"]]}, ui)

    monkeypatch.setattr(photos3d, "recon_available", lambda: (False, NOT_BUILT))
    with pytest.raises(ToolError, match="not built yet"):
        call(ws, {}, ui)


def test_assistant_tool_makes_the_model_and_shows_it(ws, tmp_path, monkeypatch, recon_ready):
    from cloudclean.assistant.tools import ToolError

    project = ws.create_project("Bracket")
    ids = add_photos(ws, tmp_path, 3, project["id"])
    fake_docker(monkeypatch, ok_lines(3))
    result, events = call(ws, {"name": "Bracket from photos"}, {"project_id": project["id"]}, Jobs(ws))
    assert len(result.asset_ids) == 1
    meta = ws.get(result.asset_ids[0])
    assert meta["name"] == "Bracket from photos" and sorted(meta["parents"]) == sorted(ids)
    assert result.data["created"][0]["kind"] == "pointcloud" and "true size" in result.data["hint"]
    assert ("ui", {"action": "show", "asset_ids": result.asset_ids, "exclusive": True}) in events
    assert any(e == "tool_progress" for e, _ in events)

    fake_docker(monkeypatch, ["ERROR CUDA out of memory"], code=1)   # a failed job becomes a ToolError
    with pytest.raises(ToolError, match="CUDA out of memory"):
        call(ws, {}, {"project_id": project["id"]}, Jobs(ws))


# --------------------------------------------------------------------------- cancel
def test_cancel_stops_the_container_and_cleans_up(ws, tmp_path, monkeypatch, recon_ready):
    """Cancel terminates the worker (SIGTERM -> SystemExit): the running container is removed, the work folder too."""
    ids = add_photos(ws, tmp_path, 2)
    state = {}

    class Running:
        def __init__(self, cmd, **kwargs):
            state["cmd"] = list(cmd)
            state["killed"] = False

            def lines():
                yield "PROGRESS 0.300 Reconstructing from 2 photos\n"
                raise SystemExit("cancelled")   # what the SIGTERM handler raises

            self.stdout = lines()

        def poll(self):
            return None   # still running

        def kill(self):
            state["killed"] = True

    monkeypatch.setattr(photos3d.subprocess, "Popen", Running)
    with pytest.raises(SystemExit):
        photos3d.job_photos_to_3d(ws, {"photo_ids": ids}, Log())
    name = state["cmd"][state["cmd"].index("--name") + 1]
    assert name.startswith(photos3d.CONTAINER_PREFIX) and name in recon_ready and state["killed"]
    assert "leftovers" in recon_ready   # stale containers of a crashed run are cleared first
    assert list((ws.root / "recon").iterdir()) == []
    assert "HF_TOKEN" in state["cmd"] and "TORCH_HOME=/hf/torch" in state["cmd"]


# --------------------------------------------------------------------------- phone photos
def rotated_jpeg(path, size=(64, 48)):
    """A JPEG whose EXIF says 'rotate 90° clockwise to view' (orientation 6), as phones write portrait shots."""
    im = Image.new("RGB", size, (200, 30, 30))
    exif = im.getexif()
    exif[0x0112] = 6
    im.save(path, "JPEG", exif=exif.tobytes())
    return path


def test_rotated_and_heic_photos_become_upright_jpegs(ws, tmp_path):
    from cloudclean.io import to_jpeg

    out = to_jpeg(rotated_jpeg(tmp_path / "portrait.jpg"), tmp_path / "upright.jpg")
    with Image.open(out) as im:
        assert im.size == (48, 64) and im.getexif().get(0x0112, 1) == 1
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    heic = tmp_path / "IMG_0001.HEIC"
    Image.new("RGB", (80, 60), (20, 120, 200)).save(heic, format="HEIF")
    meta = ws.add_image(heic, "IMG_0001")
    assert meta["file"] == "image.jpg" and meta["stats"]["width"] == 80 and meta["stats"]["height"] == 60
    with Image.open(ws.image_path(meta["id"])) as im:
        assert im.format == "JPEG"
    assert not list(ws.uploads_dir.glob("*.jpg"))   # the temporary conversion is gone


def test_photo_preview_route_turns_photos_into_jpeg(tmp_path):
    from cloudclean.web.server import create_app

    src = rotated_jpeg(tmp_path / "portrait.jpg")
    with TestClient(create_app(tmp_path / "ws")) as client:
        res = client.post("/api/images/jpeg", files={"file": ("portrait.jpg", src.read_bytes(), "image/jpeg")},
                          data={"max_side": "32"})
        bad = client.post("/api/images/jpeg", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    (tmp_path / "view.jpg").write_bytes(res.content)
    with Image.open(tmp_path / "view.jpg") as im:
        assert im.size == (24, 32)   # upright and shrunk
    assert bad.status_code == 400


def test_job_passes_on_the_advice_about_the_photos(ws, tmp_path, monkeypatch, recon_ready):
    """Exit 3 = the photos are the problem (too few placed, no points): the job says what to do, no exit code."""
    ids = add_photos(ws, tmp_path, 3)
    advice = "Fewer than 3 of the photos could be placed exactly: neighbouring photos must overlap"
    fake_docker(monkeypatch, ["PROGRESS 0.040 Finding features in 3 photos", "ERROR " + advice], code=3)
    with pytest.raises(ValueError) as err:
        photos3d.job_photos_to_3d(ws, {"photo_ids": ids}, Log())
    assert str(err.value) == advice
    assert list((ws.root / "recon").iterdir()) == []
