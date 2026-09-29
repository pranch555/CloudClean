"""Projects (Contract 1): storage, migration into "My scans", inheritance through jobs, counts, moves, deletion,
thumbnails, and the project_id of the upload / import / capture / autopilot endpoints."""
import io
import json
import time

import numpy as np
import open3d as o3d
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cloudclean.io import save
from cloudclean.web.jobs import job_import
from cloudclean.web.server import create_app
from cloudclean.web.workspace import DEFAULT_PROJECT, ProjectNotEmpty, Workspace


def cloud(n=400, seed=0, scale=10.0) -> o3d.geometry.PointCloud:
    pts = np.random.default_rng(seed).uniform(0, scale, (n, 3))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))


def png_bytes(color=(200, 30, 30), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def wait(client, job, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        current = {j["id"]: j for j in client.get("/api/jobs").json()}[job["id"]]
        if current["status"] in ("done", "failed", "cancelled"):
            assert current["status"] == "done", "\n".join(current["logs"][-40:]) + f"\n{current['error']}"
            return current
        time.sleep(0.3)
    raise TimeoutError(job["title"])


# --------------------------------------------------------------------------- workspace level
def test_migration_files_legacy_assets_into_my_scans_once(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ids = [ws.add_geometry(cloud(seed=k), f"scan {k}", "import")["id"] for k in range(3)]
    # simulate a workspace from before projects: no projects.json, no project in meta
    ws.projects_file.unlink()
    for asset_id in ids:
        meta = ws.get(asset_id)
        meta.pop("project")
        ws._write_meta(meta)
    assert ws.projects() == []
    result = ws.migrate_projects()
    assert result["assigned"] == 3
    projects = ws.projects()
    assert [p["name"] for p in projects] == [DEFAULT_PROJECT]
    assert {ws.get(i)["project"] for i in ids} == {projects[0]["id"]}
    assert ws.migrate_projects() == {"assigned": 0, "project_id": None}   # created once, nothing left to do
    # an asset whose project disappeared is filed into the same "My scans"
    other = ws.create_project("Bracket")
    ws.move_asset(ids[0], other["id"])
    with ws._projects_txn() as stored:
        stored[:] = [p for p in stored if p["id"] != other["id"]]
    assert ws.migrate_projects()["project_id"] == projects[0]["id"]
    assert len(ws.projects()) == 1


def test_new_assets_inherit_the_first_parents_project(tmp_path):
    ws = Workspace(tmp_path / "ws")
    a = ws.create_project("Part A")
    b = ws.create_project("Part B")
    scan_a = ws.add_geometry(cloud(), "scan a", "import", project=a["id"])
    scan_b = ws.add_geometry(cloud(seed=1), "scan b", "import", project=b["id"])
    assert scan_a["project"] == a["id"] and scan_b["project"] == b["id"]
    child = ws.add_geometry(cloud(seed=2), "merged", "merge", parents=[scan_b["id"], scan_a["id"]])
    assert child["project"] == b["id"]                                   # first parent wins
    explicit = ws.add_geometry(cloud(seed=3), "x", "edit", parents=[scan_b["id"]], project=a["id"])
    assert explicit["project"] == a["id"]                                # an explicit project beats inheritance
    # no parent, no project: the most recently updated project (here: the one that just received an asset)
    time.sleep(0.01)
    ws.add_geometry(cloud(seed=4), "late", "import", project=b["id"])
    assert ws.add_geometry(cloud(seed=5), "loose", "import")["project"] == b["id"]
    # an unknown explicit project falls back instead of failing the import
    assert ws.add_geometry(cloud(seed=6), "lost", "import", project="0123456789ab")["project"] in {a["id"], b["id"]}
    # the part frame is stored with every new asset
    assert set(scan_a["part"]["dimensions"]) == {"length", "width", "height"}


def test_default_project_is_created_when_there_is_none(tmp_path):
    ws = Workspace(tmp_path / "ws")
    meta = ws.add_geometry(cloud(), "first", "import")
    assert [p["name"] for p in ws.projects()] == [DEFAULT_PROJECT] and meta["project"] == ws.projects()[0]["id"]
    image = tmp_path / "photo.png"
    image.write_bytes(png_bytes())
    photo = ws.add_image(image, "photo")
    assert photo["project"] == meta["project"]


def test_counts_update_move_delete_and_cover(tmp_path):
    ws = Workspace(tmp_path / "ws")
    p = ws.create_project("Housing", "left half")
    scan = ws.add_geometry(cloud(), "scan", "import", project=p["id"])
    ws.add_geometry(cloud(seed=1), "scan · clean", "clean", parents=[scan["id"]])
    mesh = o3d.geometry.TriangleMesh.create_box(1, 2, 3)
    ws.add_geometry(mesh, "mesh", "mesh", parents=[scan["id"]])
    ws.add_geometry(o3d.geometry.TriangleMesh.create_box(1, 1, 1), "cad.step", "import", project=p["id"])
    image = tmp_path / "photo.png"
    image.write_bytes(png_bytes())
    ws.add_image(image, "photo", project=p["id"])
    summary = {s["id"]: s for s in ws.project_summaries()}[p["id"]]
    assert summary["counts"] == {"scans": 2, "meshes": 1, "results": 1, "photos": 1, "total": 5}   # imports are scans
    assert summary["description"] == "left half" and summary["cover_asset_id"] is None

    with pytest.raises(ValueError, match="empty"):
        ws.update_project(p["id"], name="  ")
    other = ws.create_project("Other")
    stranger = ws.add_geometry(cloud(seed=9), "stranger", "import", project=other["id"])
    with pytest.raises(ValueError, match="not in this project"):
        ws.update_project(p["id"], cover_asset_id=stranger["id"])
    updated = ws.update_project(p["id"], name="Housing v2", cover_asset_id=scan["id"])
    assert updated["name"] == "Housing v2" and updated["cover_asset_id"] == scan["id"]

    ws.move_asset(scan["id"], other["id"])
    summaries = {s["id"]: s for s in ws.project_summaries()}
    assert summaries[p["id"]]["counts"]["total"] == 4 and summaries[other["id"]]["counts"]["total"] == 2
    assert summaries[p["id"]]["cover_asset_id"] is None          # the cover moved away
    assert ws.project_summaries()[0]["id"] == other["id"]         # newest first: the move touched it

    with pytest.raises(ProjectNotEmpty):
        ws.delete_project(p["id"])
    result = ws.delete_project(p["id"], delete_assets=True)
    assert len(result["deleted_assets"]) == 4 and p["id"] not in {x["id"] for x in ws.projects()}
    assert all(m["project"] == other["id"] for m in ws.list())


def test_thumbnails(tmp_path):
    ws = Workspace(tmp_path / "ws")
    meta = ws.add_geometry(cloud(), "scan", "import")
    with pytest.raises(KeyError):
        ws.thumbnail_path(meta["id"])
    with pytest.raises(ValueError, match="PNG"):
        ws.set_thumbnail(meta["id"], b"GIF89a....")
    with pytest.raises(ValueError, match="too large"):
        ws.set_thumbnail(meta["id"], png_bytes()[:8] + b"\0" * (2 * 1024 * 1024))
    assert ws.set_thumbnail(meta["id"], png_bytes())["has_thumbnail"] is True
    assert ws.thumbnail_path(meta["id"]).read_bytes() == png_bytes()


def test_job_import_uses_payload_project(tmp_path):
    ws = Workspace(tmp_path / "ws")
    target = ws.create_project("Target")
    ws.create_project("Newer")
    save(cloud(), tmp_path / "scan.ply")
    ids = job_import(ws, {"path": str(tmp_path / "scan.ply"), "name": "scan", "project_id": target["id"]},
                     lambda msg: None)
    assert ws.get(ids[0])["project"] == target["id"]


# --------------------------------------------------------------------------- API
def test_projects_api(tmp_path):
    save(cloud(3000), tmp_path / "scan.ply")
    with TestClient(create_app(tmp_path / "ws")) as client:
        assert client.get("/api/projects").json() == []
        assert client.post("/api/projects", json={"name": "  "}).status_code == 400
        p = client.post("/api/projects", json={"name": "Screw M8", "description": "zinc"}).json()
        assert p["name"] == "Screw M8" and p["counts"]["total"] == 0 and len(p["id"]) == 12
        q = client.post("/api/projects", json={"name": "Bracket"}).json()

        with open(tmp_path / "scan.ply", "rb") as f:
            up = client.post("/api/upload", data={"project_id": p["id"]},
                             files=[("files", ("scan.ply", f, "application/octet-stream")),
                                    ("files", ("photo.png", png_bytes(), "image/png"))]).json()
        assert up["images"][0]["project"] == p["id"]
        asset_id = wait(client, up["jobs"][0])["result"][0]
        assert client.get(f"/api/assets/{asset_id}").json()["project"] == p["id"]
        with open(tmp_path / "scan.ply", "rb") as f:
            bad = client.post("/api/upload", data={"project_id": "0123456789ab"},
                              files=[("files", ("scan.ply", f, "application/octet-stream"))])
        assert bad.status_code == 400 and "not found" in bad.json()["detail"]
        by_path = client.post("/api/import-paths", json={"paths": [str(tmp_path / "scan.ply")], "project_id": q["id"]})
        second = wait(client, by_path.json()["jobs"][0])["result"][0]
        assert client.get(f"/api/assets/{second}").json()["project"] == q["id"]

        # every job keeps the project of its input
        edit = client.post("/api/edit", json={"asset_id": asset_id, "ops": [{"op": "translate", "offset": [1, 0, 0]}]})
        edited = wait(client, edit.json())["result"][0]
        assert client.get(f"/api/assets/{edited}").json()["project"] == p["id"]

        listed = client.get("/api/projects").json()
        counts = {x["id"]: x["counts"] for x in listed}
        assert counts[p["id"]] == {"scans": 1, "meshes": 0, "results": 1, "photos": 1, "total": 3}
        assert {m["id"] for m in client.get(f"/api/assets?project={q['id']}").json()} == {second}
        assert len(client.get("/api/assets").json()) == 4

        patched = client.patch(f"/api/projects/{p['id']}", json={"name": "Screw M8x40", "cover_asset_id": edited})
        assert patched.status_code == 200 and patched.json()["name"] == "Screw M8x40"
        assert patched.json()["cover_asset_id"] == edited
        assert client.patch(f"/api/projects/{p['id']}", json={"colour": "red"}).status_code == 400
        assert client.patch(f"/api/projects/{p['id']}", json={"cover_asset_id": second}).status_code == 400
        assert client.patch("/api/projects/0123456789ab", json={"name": "x"}).status_code == 404
        assert client.get(f"/api/projects/{q['id']}").json()["counts"]["total"] == 1

        moved = client.post(f"/api/assets/{second}/move", json={"project_id": p["id"]})
        assert moved.status_code == 200 and moved.json()["project"] == p["id"]
        assert client.post(f"/api/assets/{second}/move", json={"project_id": "0123456789ab"}).status_code == 404
        assert client.post("/api/assets/0123456789ab/move", json={"project_id": p["id"]}).status_code == 404

        assert client.get(f"/api/assets/{asset_id}/thumbnail").status_code == 404
        res = client.put(f"/api/assets/{asset_id}/thumbnail", content=png_bytes(), headers={"Content-Type": "image/png"})
        assert res.status_code == 200 and res.json()["has_thumbnail"] is True
        got = client.get(f"/api/assets/{asset_id}/thumbnail")
        assert got.status_code == 200 and got.headers["content-type"] == "image/png" and got.content == png_bytes()
        assert client.put(f"/api/assets/{asset_id}/thumbnail", content=b"not a png").status_code == 400
        too_big = client.put(f"/api/assets/{asset_id}/thumbnail", content=b"\x89PNG\r\n\x1a\n" + b"\0" * (2 << 20))
        assert too_big.status_code == 413

        refused = client.delete(f"/api/projects/{p['id']}")
        assert refused.status_code == 409 and "asset" in refused.json()["detail"]
        empty = client.delete(f"/api/projects/{q['id']}")
        assert empty.status_code == 200 and empty.json()["deleted_assets"] == []
        gone = client.delete(f"/api/projects/{p['id']}?delete_assets=true").json()
        assert set(gone["deleted_assets"]) == {asset_id, edited, second, up["images"][0]["id"]}
        assert client.get("/api/assets").json() == [] and client.get("/api/projects").json() == []


def test_startup_migration_through_the_app(tmp_path):
    ws = Workspace(tmp_path / "ws")
    meta = ws.add_geometry(cloud(), "legacy", "import")
    ws.projects_file.unlink()
    raw = ws.get(meta["id"])
    raw.pop("project")
    ws._write_meta(raw)
    with TestClient(create_app(tmp_path / "ws")) as client:
        projects = client.get("/api/projects").json()
        assert [p["name"] for p in projects] == [DEFAULT_PROJECT] and projects[0]["counts"]["scans"] == 1
        assert client.get(f"/api/assets/{meta['id']}").json()["project"] == projects[0]["id"]


def test_capture_save_and_bridge_take_a_project(tmp_path):
    with TestClient(create_app(tmp_path / "ws")) as client:
        p = client.post("/api/projects", json={"name": "Captured part"}).json()
        client.post("/api/projects", json={"name": "Newer project"})
        r = client.post("/api/capture/connect", json={"driver": "simulated",
                                                      "settings": {"realtime": False, "max_frames": 12,
                                                                   "speed_limit": 400}})
        assert r.status_code == 200, r.text
        client.post("/api/capture/start")
        start = time.time()
        while client.get("/api/capture/status").json()["state"] != "finished":
            assert time.time() - start < 120
            time.sleep(0.2)
        bad = client.post("/api/capture/save", json={"auto_process": False, "project_id": "0123456789ab"})
        assert bad.status_code == 400
        saved = client.post("/api/capture/save", json={"name": "cap", "auto_process": False, "project_id": p["id"]})
        assert saved.status_code == 200, saved.text
        assert saved.json()["asset"]["project"] == p["id"]
        assert client.post("/api/capture/discard").json()["active"] is False
        res = client.post("/api/capture/bridge/upload", data={"project_id": "0123456789ab"},
                          files=[("files", ("export.ply", b"ply", "application/octet-stream"))])
        assert res.status_code == 400


def test_autopilot_endpoints_take_a_project(tmp_path):
    from tests.synthetic import simulate_scan

    scan, _ = simulate_scan([0.2, 0.1, 1.0], seed=5, n_points=20_000)
    save(scan, tmp_path / "scan.ply")
    with TestClient(create_app(tmp_path / "ws")) as client:
        client.put("/api/autopilot/settings", json={"output_folder": str(tmp_path / "out"), "formats": ["ply"]})
        p = client.post("/api/projects", json={"name": "Auto"}).json()
        client.post("/api/projects", json={"name": "Newer"})
        with open(tmp_path / "scan.ply", "rb") as f:
            bad = client.post("/api/autopilot/upload", data={"project_id": "0123456789ab"},
                              files=[("files", ("scan.ply", f, "application/octet-stream"))])
        assert bad.status_code == 400
        with open(tmp_path / "scan.ply", "rb") as f:
            job = client.post("/api/autopilot/upload", data={"project_id": p["id"]},
                              files=[("files", ("scan.ply", f, "application/octet-stream"))]).json()
        assert job["payload"]["project_id"] == p["id"]
        done = wait(client, job, timeout=600)
        created = done["result"]
        assert len(created) >= 3   # import, clean, mesh
        assert {client.get(f"/api/assets/{i}").json()["project"] for i in created} == {p["id"]}
        run = client.post("/api/autopilot/run", json={"asset_ids": ["0123456789ab"], "project_id": p["id"]})
        assert run.status_code == 404
