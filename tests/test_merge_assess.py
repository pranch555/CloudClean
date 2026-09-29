"""Merge assessment: check first whether scans should be merged (coverage gain, alignment confidence, doubled
surfaces), merging with approved transforms, autopilot merge modes + decisions, capture pending scans."""
import json
import time

import numpy as np
import open3d as o3d
import pytest

from cloudclean import register
from cloudclean.clean import CleanParams, clean_point_cloud
from cloudclean.register import MergeParams, assess_merge, assessment_summary, estimate_noise, merge_geometries
from tests.synthetic import random_rigid, simulate_scan

QUIET = lambda m: None  # noqa: E731
TOP, BOTTOM = [0.2, 0.1, 1.0], [-0.1, -0.2, -1.0]


def scan(view, seed, transform=None, scale=None, n=60_000):
    cloud, _ = simulate_scan(view, seed=seed, n_points=n, outlier_fraction=0)
    cloud, _ = clean_point_cloud(cloud, CleanParams(), QUIET)   # drops the synthetic debris blob
    if scale:
        cloud.scale(scale, center=np.zeros(3))
    if transform is not None:
        cloud.transform(transform)
    return cloud


def box_scan(view, seed, transform=None, n=60_000):
    """Partial scan of a plain box: symmetric, so a flipped pose fits as well as the right one."""
    mesh = o3d.geometry.TriangleMesh.create_box(60.0, 40.0, 20.0)
    mesh.translate(-mesh.get_center())
    o3d.utility.random.seed(seed)
    pcd = mesh.sample_points_uniformly(n * 2, use_triangle_normal=True)
    pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
    d = np.asarray(view, float) / np.linalg.norm(view)
    pts = pts[nrm @ d > -0.45]
    pts = pts + np.random.default_rng(seed).normal(scale=0.02, size=pts.shape)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if transform is not None:
        cloud.transform(transform)
    return cloud


def by_index(assessment, i):
    return next(s for s in assessment["scans"] if s["index"] == i)


@pytest.fixture(scope="module")
def complementary():
    T_b = random_rigid(7)
    geoms = [scan(TOP, 1), scan(BOTTOM, 2, transform=T_b)]
    return geoms, T_b, assess_merge(geoms, log=QUIET)


def test_noise_estimate_matches_simulated_noise():
    cloud, _ = simulate_scan(TOP, seed=4, n_points=60_000, noise=0.05, outlier_fraction=0)
    assert 0.035 < estimate_noise(cloud.points) < 0.065


def test_complementary_sides_merge(complementary):
    geoms, T_b, a = complementary
    assert a["recommendation"] == "merge", a["reasons"]
    s = by_index(a, 1)
    assert s["coverage_gain"] >= 0.3 and s["coverage_gain_mm2"] > 100
    assert not s["doubled_surface"] and not s["ambiguous"] and s["fitness"] > 0.2
    assert s["separation_ratio"] < 1.2                 # overlapping flanks coincide within noise
    assert 0.01 < s["noise_mm"] < 0.03 and s["points"] == len(geoms[1].points)
    assert not s["already_aligned"]
    err = np.asarray(a["transforms"][1]) @ T_b         # the transform undoes the unknown pose
    assert np.degrees(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))) < 0.3
    assert np.linalg.norm(err[:3, 3]) < 0.15
    assert json.dumps(a) and a["reasons"]
    summary = assessment_summary(a, names=["top", "bottom"])
    assert "transforms" not in summary and summary["scans"][1]["name"] == "bottom"


def test_same_side_twice_uses_best():
    geoms = [scan(TOP, 1), scan(TOP, 2, transform=random_rigid(3), n=80_000)]
    a = assess_merge(geoms, log=QUIET)
    assert a["recommendation"] == "use_best", a["reasons"]
    s = by_index(a, 1)
    assert s["coverage_gain"] < MergeParams().min_gain and s["overlap"] > 0.9
    assert a["best_index"] == 1                          # same noise level -> the scan with more points
    assert any("only" in r and "new surface" in r for r in a["reasons"])


def test_same_side_already_aligned_is_detected():
    geoms = [scan(TOP, 1), scan(TOP, 5)]
    s = by_index(assess_merge(geoms, MergeParams(method="icp"), log=QUIET), 1)
    assert s["already_aligned"] and s["coverage_gain"] < 0.03


def test_scaled_copy_is_a_doubled_surface():
    # the other side, 0.3 % larger: ICP cannot make the overlapping flanks coincide -> layered skin
    geoms = [scan(TOP, 1), scan(BOTTOM, 2, scale=1.003)]
    a = assess_merge(geoms, log=QUIET)
    s = by_index(a, 1)
    assert s["doubled_surface"] and s["separation_ratio"] > 1.5, s
    assert a["recommendation"] == "ask"
    assert any("doubled" in r for r in a["reasons"])

    same_side = assess_merge([scan(TOP, 1), scan(TOP, 2, scale=1.003)], log=QUIET)
    assert by_index(same_side, 1)["doubled_surface"]


def test_symmetric_part_is_ambiguous():
    geoms = [box_scan([0, 0, 1], 1), box_scan([0, 0, -1], 2, transform=random_rigid(11))]
    a = assess_merge(geoms, log=QUIET)
    s = by_index(a, 1)
    assert s["ambiguous"] and not s["confident"], s
    assert a["recommendation"] == "ask"
    assert any("symmetric" in r for r in a["reasons"])


def test_single_scan():
    a = assess_merge([scan(TOP, 1)], log=QUIET)
    assert a["recommendation"] == "single" and a["best_index"] == 0 and len(a["transforms"]) == 1


def test_merge_with_approved_transforms_skips_registration(complementary, monkeypatch):
    geoms, _, a = complementary
    reference, ref_transforms, _ = merge_geometries(geoms, MergeParams(), log=QUIET)

    def forbidden(*args, **kwargs):
        raise AssertionError("global registration must not run for approved transforms")

    monkeypatch.setattr(register, "global_candidates", forbidden)
    monkeypatch.setattr(register, "align_pair", forbidden)
    logs = []
    merged, transforms, report = merge_geometries(geoms, MergeParams(), log=logs.append, transforms=a["transforms"])
    assert np.allclose(transforms[1], np.asarray(a["transforms"][1]))   # exactly the approved pose
    assert report["scans"][1]["method"] == "approved" and report["scans"][1]["fitness"] > 0.2
    assert not any("candidate" in line for line in logs)
    err = transforms[1] @ np.linalg.inv(ref_transforms[1])
    assert np.degrees(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))) < 0.1
    assert np.linalg.norm(err[:3, 3]) < 0.05
    assert abs(len(merged.points) - len(reference.points)) < 0.01 * len(reference.points)

    with pytest.raises(ValueError):
        merge_geometries(geoms, MergeParams(), log=QUIET, transforms=[np.eye(4).tolist()])
    bad = np.eye(4)
    bad[0, 0] = 2.0
    with pytest.raises(ValueError):
        merge_geometries(geoms, MergeParams(), log=QUIET, transforms=[None, bad.tolist()])


# --------------------------------------------------------------------------- assistant guard
def test_assistant_merge_requires_confirmation(tmp_path):
    import asyncio

    from cloudclean.assistant.agent import SYSTEM_PROMPT
    from cloudclean.assistant.tools import TOOLS, ToolContext, ToolError, call_tool
    from cloudclean.web.workspace import Workspace

    ws = Workspace(tmp_path / "ws")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.default_rng(0).uniform(0, 10, (500, 3))))
    ids = [ws.add_geometry(cloud, f"scan {k}", "import")["id"] for k in (1, 2)]
    ctx = ToolContext(ws, None, {}, {}, lambda event, data: None, "conv")
    for name, args in (("merge", {"asset_ids": ids}),
                       ("run_pipeline", {"asset_ids": ids, "merge_mode": "always"})):
        with pytest.raises(ToolError, match="assess_merge"):
            asyncio.run(call_tool(ctx, name, args))
    assert "assess_merge" in TOOLS and "merge_mode" in TOOLS["run_pipeline"].parameters["properties"]
    assert "assess_merge" in SYSTEM_PROMPT and "user_confirmed" in SYSTEM_PROMPT


# --------------------------------------------------------------------------- CLI pipeline
def test_run_pipeline_auto_meshes_best_scan_of_duplicates(tmp_path):
    from cloudclean.io import save
    from cloudclean.pipeline import run_pipeline

    save(scan(TOP, 1), tmp_path / "part_1.ply")
    save(scan(TOP, 2, transform=random_rigid(3), n=80_000), tmp_path / "part_2.ply")
    out = tmp_path / "out"
    inputs = [tmp_path / "part_1.ply", tmp_path / "part_2.ply"]
    report = run_pipeline(inputs, out, skip_clean=True, formats=["stl"], log=QUIET)
    assert report["merge_assessment"]["recommendation"] == "use_best"
    assert report["merge"] == {"skipped": "use_best", "best_index": 1, "best_input": str(inputs[1])}
    assert report["outputs"]["mesh"] == [str(out / "mesh.stl")] and "merged" not in report["outputs"]
    assert report["notes"] and "part_2.ply" in report["notes"][0]

    never = run_pipeline(inputs, tmp_path / "never", skip_clean=True, formats=["stl"], merge_mode="never", log=QUIET)
    assert sorted(never["outputs"]["mesh"]) == sorted(str(tmp_path / "never" / f"mesh_part_{k}.stl") for k in (1, 2))
    assert "merge_assessment" not in never and len(never["meshes"]) == 2


# --------------------------------------------------------------------------- autopilot (web)
def _wait_job(client, job_id, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        job = {j["id"]: j for j in client.get("/api/jobs").json()}[job_id]
        if job["status"] in ("done", "failed", "cancelled"):
            assert job["status"] == "done", "\n".join(job["logs"][-60:]) + f"\n{job['error']}"
            return job
        time.sleep(0.5)
    raise TimeoutError(job_id)


def _upload(client, files):
    handles = [open(f, "rb") for f in files]
    try:
        res = client.post("/api/autopilot/upload",
                          files=[("files", (f.name, h, "application/octet-stream")) for f, h in zip(files, handles)])
    finally:
        for h in handles:
            h.close()
    assert res.status_code == 200, res.text
    return res.json()


def test_autopilot_auto_uses_best_scan_for_duplicates(tmp_path):
    from fastapi.testclient import TestClient

    from cloudclean.io import load, save
    from cloudclean.web.server import create_app

    a, _ = simulate_scan(TOP, seed=1, n_points=60_000)
    b, _ = simulate_scan(TOP, seed=2, n_points=60_000, transform=random_rigid(3))
    save(a, tmp_path / "screw_1.ply")
    save(b, tmp_path / "screw_2.ply")
    out = tmp_path / "out"
    with TestClient(create_app(tmp_path / "ws")) as client:
        settings = client.put("/api/autopilot/settings", json={"output_folder": str(out), "formats": ["stl"]}).json()
        assert settings["merge_mode"] == "auto"
        assert client.put("/api/autopilot/settings", json={"merge_mode": "sometimes"}).status_code == 400
        job = _wait_job(client, _upload(client, [tmp_path / "screw_1.ply", tmp_path / "screw_2.ply"])["id"])
        assert job["output"]["notes"] and "not merged" in job["output"]["notes"][0]
        assert sorted(p.name for p in (out / "screw").iterdir()) == ["report.json", "screw.stl"]
        report = json.loads((out / "screw" / "report.json").read_text())
        assert report["stages"]["assessment"]["recommendation"] == "use_best"
        assert report["stages"]["merge"]["skipped"] == "use_best" and report["notes"]
        assert not [m for m in client.get("/api/assets").json() if m["operation"] == "merge"]
        assert len(load(out / "screw" / "screw.stl").triangles) > 1000
        history = client.get("/api/autopilot/status").json()["history"]
        assert history[0]["status"] == "done" and history[0]["notes"]


def test_autopilot_ask_then_decide_merge(tmp_path):
    from fastapi.testclient import TestClient

    from cloudclean.io import describe, load, save
    from cloudclean.web.server import create_app
    from tests.synthetic import ground_truth_mesh

    a, _ = simulate_scan(TOP, seed=1, n_points=60_000)
    b, _ = simulate_scan(BOTTOM, seed=2, n_points=60_000, transform=random_rigid(7))
    save(a, tmp_path / "gear_1.ply")
    save(b, tmp_path / "gear_2.ply")
    out = tmp_path / "out"
    with TestClient(create_app(tmp_path / "ws")) as client:
        assert client.put("/api/autopilot/settings", json={"output_folder": str(out), "merge_mode": "ask",
                                                           "formats": ["stl"]}).status_code == 200
        first = _wait_job(client, _upload(client, [tmp_path / "gear_1.ply", tmp_path / "gear_2.ply"])["id"])
        assert first["output"]["decision_needed"] is True and len(first["output"]["cleaned_ids"]) == 2
        assert first["output"]["assessment"]["recommendation"] == "merge"
        assert not out.exists() or not any(out.iterdir())          # nothing meshed or exported yet

        entry = client.get("/api/autopilot/status").json()["history"][0]
        assert entry["status"] == "needs_decision" and entry["assessment"]["recommendation"] == "merge"
        assert "transforms" not in entry["assessment"] and entry["decision"]["transforms"]

        def decide(body):
            return client.post("/api/autopilot/decide", json={"job_id": first["id"], **body})

        assert decide({"decision": "maybe"}).status_code == 400
        assert decide({"decision": "best", "best_index": 5}).status_code == 400
        assert client.post("/api/autopilot/decide", json={"job_id": "nope", "decision": "merge"}).status_code == 404
        res = decide({"decision": "merge"})
        assert res.status_code == 200, res.text
        assert res.json()["entry"]["status"] == "decided"
        assert decide({"decision": "merge"}).status_code == 409   # decided once only

        job = _wait_job(client, res.json()["job"]["id"])
        stl = out / "gear" / "gear.stl"
        assert str(stl) in job["output"]["exports"]
        dims = np.asarray(describe(load(stl))["dimensions"])
        truth = np.asarray(describe(ground_truth_mesh())["dimensions"])
        assert np.all(np.abs(dims - truth) < 0.3), (dims, truth)
        report = json.loads((out / "gear" / "report.json").read_text())
        assert report["stages"]["merge"]["approved_transforms"] is True
        assert report["stages"]["merge"]["scans"][1]["best_candidate"] == "approved transform"
        assert report["stages"]["clean"]["skipped"]
        history = client.get("/api/autopilot/status").json()["history"]
        assert history[0]["decision_for"] == first["id"] and history[0]["status"] == "done"
        assert history[1]["decided"]["decision"] == "merge" and history[1]["decided"]["job_id"] == job["id"]


# --------------------------------------------------------------------------- capture
def test_capture_bridge_keeps_further_scans_pending(tmp_path):
    from fastapi.testclient import TestClient

    from cloudclean.io import save
    from cloudclean.web.routes_capture import manager_for
    from cloudclean.web.server import create_app
    from tests.test_capture import capture_threads, wait_for, world_scan

    first = world_scan(1, 0.0)
    save(first, tmp_path / "export_1.ply")
    save(first, tmp_path / "export_1_again.ply")
    save(world_scan(1, 6.0), tmp_path / "export_2.ply")

    with TestClient(create_app(tmp_path / "ws")) as client:
        def upload(name):
            with open(tmp_path / name, "rb") as f:
                res = client.post("/api/capture/bridge/upload", files=[("files", (name, f))],
                                  data={"import_asset": "false"})
            assert res.status_code == 200, res.text

        def status():
            return client.get("/api/capture/status").json()

        upload("export_1.ply")
        wait_for(lambda: status()["scans"] >= 1)
        manager = manager_for(tmp_path / "ws")
        points = status()["points"]
        cursor: dict = {}
        manager.stream(cursor)

        upload("export_1_again.ply")                      # the same scan again
        wait_for(lambda: status()["scans"] >= 2)
        s = status()
        assert s["points"] == points and s["pending_scans"] == 1
        assert s["guidance"]["status"]["code"] == "pending_scan"
        pending = client.get("/api/capture/pending").json()["pending"]
        assessment = pending[0]["assessment"]
        assert assessment["recommendation"] == "use_best" and assessment["coverage_gain_pct"] < 3
        messages = [m for m in manager.stream(cursor) if isinstance(m, dict) and m["type"] == "pending_scan"]
        assert messages and messages[0]["scan_id"] == pending[0]["scan_id"]
        assert messages[0]["assessment"]["reasons"]

        url = f"/api/capture/pending/{pending[0]['scan_id']}"
        assert client.post(url, json={"decision": "maybe"}).status_code == 400
        res = client.post(url, json={"decision": "discard"})
        assert res.status_code == 200, res.text
        assert res.json()["status"]["points"] == points and res.json()["status"]["pending_scans"] == 0
        assert client.post(url, json={"decision": "fuse"}).status_code == 404
        assert any(m.get("type") == "pending_resolved" for m in manager.stream(cursor) if isinstance(m, dict))

        upload("export_2.ply")                             # another view: kept aside too in "ask" mode
        wait_for(lambda: status()["scans"] >= 3)
        pending = client.get("/api/capture/pending").json()["pending"]
        assert len(pending) == 1 and pending[0]["assessment"]["coverage_gain_pct"] > 3
        res = client.post(f"/api/capture/pending/{pending[0]['scan_id']}", json={"decision": "fuse"}).json()
        assert res["status"]["points"] > points and res["status"]["pending_scans"] == 0
        assert [x.get("status") for x in manager.session.report()["scans"]] == ["fused", "discarded", "fused"]

        upload("export_1_again.ply")
        wait_for(lambda: status()["scans"] >= 4)
        pending = client.get("/api/capture/pending").json()["pending"]
        kept = client.post(f"/api/capture/pending/{pending[0]['scan_id']}", json={"decision": "keep_separate"}).json()
        assert kept["asset"]["name"] == "export_1_again" and kept["asset"]["kind"] == "pointcloud"
        client.post("/api/capture/stop")
    assert not capture_threads()


def test_capture_auto_fuse_mode(tmp_path):
    from cloudclean.capture.drivers import create_driver
    from cloudclean.capture.drivers.base import Frame
    from cloudclean.capture.session import CaptureSession
    from tests.test_capture import world_scan

    session = CaptureSession(create_driver("folder"), {"fuse_mode": "auto", "folder": str(tmp_path)})
    session.connect()
    first, other = world_scan(1, 0.0), world_scan(1, 6.0)

    def frame(cloud, name):
        return Frame(points=np.asarray(cloud.points), coordinates="scan", meta={"name": name})

    session.process_frame(frame(first, "a"))
    n = session.n_points
    dup = session.process_frame(frame(first, "a again"))
    assert dup["status"] == "pending" and session.n_points == n
    new = session.process_frame(frame(other, "b"))
    assert new["assessment"]["recommendation"] == "merge", new["assessment"]
    assert new["status"] == "fused" and session.n_points > n
    assert len(session.pending_scans()) == 1
