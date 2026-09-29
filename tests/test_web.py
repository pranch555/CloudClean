"""End-to-end test of the web API: upload -> clean -> merge -> mesh -> photo colour -> download."""
import time

import numpy as np
from fastapi.testclient import TestClient

from cloudclean.io import save
from cloudclean.web.server import create_app
from tests.synthetic import random_rigid, render_photo, simulate_scan


def wait(client, job, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        current = {j["id"]: j for j in client.get("/api/jobs").json()}[job["id"]]
        if current["status"] in ("done", "failed", "cancelled"):
            assert current["status"] == "done", "\n".join(current["logs"][-40:]) + f"\n{current['error']}"
            return current
        time.sleep(0.5)
    raise TimeoutError(job["title"])


def test_web_flow(tmp_path):
    a, _ = simulate_scan([0.2, 0.1, 1.0], seed=1, n_points=60_000)
    b, _ = simulate_scan([-0.1, -0.2, -1.0], seed=2, n_points=60_000, transform=random_rigid(7))
    save(a, tmp_path / "scan_a.ply")
    save(b, tmp_path / "scan_b.ply")
    cam = render_photo(tmp_path / "photo.png", (0, -160, 60), fov=40)

    with TestClient(create_app(tmp_path / "ws")) as client:
        assert "CloudClean" in client.get("/").text
        assert "clean" in client.get("/api/params").json()["schema"]

        with open(tmp_path / "scan_a.ply", "rb") as f:
            up = client.post("/api/upload", files=[("files", ("scan_a.ply", f, "application/octet-stream"))]).json()
        by_path = client.post("/api/import-paths", json={"paths": [str(tmp_path / "scan_b.ply")]}).json()
        ids = [wait(client, up["jobs"][0])["result"][0], wait(client, by_path["jobs"][0])["result"][0]]
        assert client.get(f"/api/assets/{ids[0]}/preview").status_code == 200

        bad = client.post("/api/clean", json={"asset_ids": ids, "params": {"not_a_param": 1}})
        assert bad.status_code == 400

        cleaned = wait(client, client.post("/api/clean", json={"asset_ids": ids, "preset": "standard"}).json())["result"]
        merged = wait(client, client.post("/api/merge", json={"asset_ids": cleaned}).json())["result"][0]
        report = client.get(f"/api/assets/{merged}").json()["report"]
        assert report["scans"][1]["fitness"] > 0.2  # top + flipped scan only share the flanks

        mesh_id = wait(client, client.post("/api/mesh", json={"asset_id": merged, "params": {"depth": 9}}).json())["result"][0]
        mesh = client.get(f"/api/assets/{mesh_id}").json()
        assert mesh["kind"] == "mesh" and mesh["report"]["deviation"]["p95"] < 0.2

        with open(tmp_path / "photo.png", "rb") as f:
            photo = client.post("/api/upload", files=[("files", ("photo.png", f, "image/png"))]).json()["images"][0]
        view = {"image_id": photo["id"], **cam}
        colored = wait(client, client.post("/api/texture", json={"asset_id": mesh_id, "views": [view]}).json())["result"][0]
        assert client.get(f"/api/assets/{colored}").json()["report"]["coverage"] > 0.2  # one front photo

        for fmt in ("stl", "obj", "glb", "ply"):
            res = client.get(f"/api/assets/{mesh_id}/download?format={fmt}")
            assert res.status_code == 200 and len(res.content) > 1000, fmt
        assert client.get(f"/api/assets/{merged}/download?format=xyz").status_code == 200

        assert client.delete(f"/api/assets/{ids[0]}").status_code == 200
        assert all(x["id"] != ids[0] for x in client.get("/api/assets").json())
