"""Autopilot: grouping/naming, watcher (settle, group, submit once, restart), web job end-to-end, CLI runner."""
import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from cloudclean.autopilot import (AutopilotSettings, AutopilotState, Watcher, group_files, run_autopilot_cli,
                                  strip_scan_suffix, suggest_name)
from cloudclean.io import describe, load, save
from cloudclean.web.server import create_app
from tests.synthetic import ground_truth_mesh, random_rigid, simulate_scan

GT_DIMS = np.asarray(describe(ground_truth_mesh())["dimensions"])


def wait_job(client, job_id, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        current = {j["id"]: j for j in client.get("/api/jobs").json()}[job_id]
        if current["status"] in ("done", "failed", "cancelled"):
            assert current["status"] == "done", "\n".join(current["logs"][-60:]) + f"\n{current['error']}"
            return current
        time.sleep(0.5)
    raise TimeoutError(job_id)


def wait_until(predicate, timeout=15.0):
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------- grouping / naming
def test_scan_suffixes_and_names():
    assert strip_scan_suffix("gear_1") == "gear"
    assert strip_scan_suffix("gear-scan2") == "gear"
    assert strip_scan_suffix("gear (3)") == "gear"
    assert strip_scan_suffix("Housing Scan 3") == "Housing"
    assert strip_scan_suffix("bracket2024") == "bracket2024"   # no separator: part of the name
    assert strip_scan_suffix("scan_1") == "scan_1"              # never empty
    assert suggest_name(["x/part_top.ply", "x/part_bottom.ply"]) == "part"
    assert suggest_name(["gear_1.ply", "gear_2.xyz"]) == "gear"


def test_group_files(tmp_path):
    root, out = tmp_path / "watch", tmp_path / "watch" / "out"
    files = [(root / "gear_1.ply", 0), (root / "gear_2.ply", 20), (root / "gear_3.ply", 45),   # chain < 30 s apart
             (root / "lid.xyz", 200),
             (root / "housing" / "a.ply", 10), (root / "housing" / "deep" / "b.ply", 900),     # one sub-folder item
             (root / "draft.ply.tmp", 1), (root / "~$lock.ply", 1), (root / "big.ply.part", 1),
             (root / "model.step", 1), (root / "photo.jpg", 1), (out / "gear" / "gear.ply", 50),
             (root / ".hidden" / "x.ply", 1)]
    groups = group_files(files, 30, root=root, exclude=[out])
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"gear", "lid", "housing"}
    assert [Path(f).name for f in by_name["gear"]["files"]] == ["gear_1.ply", "gear_2.ply", "gear_3.ply"]
    assert len(by_name["housing"]["files"]) == 2 and by_name["housing"]["folder"] == "housing"
    assert len(group_files(files, 10, root=root, exclude=[out])) == 5  # smaller gap splits gear into 3


def test_settings_validation():
    assert AutopilotSettings().validate().formats == ["stl", "ply"]
    for bad in ({"formats": ["xyz"]}, {"formats": []}, {"preset": "nope"}, {"merge_method": "magic"},
                {"mesh": {"method": "marching"}}):
        try:
            AutopilotSettings.from_dict(bad).validate()
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    assert AutopilotSettings.load({"unknown_future_key": 1}).preset == "standard"


# --------------------------------------------------------------------------- watcher
def test_watcher_submits_each_item_once_and_survives_restart(tmp_path):
    watch, out = tmp_path / "watch", tmp_path / "watch" / "exports"
    watch.mkdir()
    settings = AutopilotSettings(watch_enabled=True, watch_folder=str(watch), output_folder=str(out),
                                 settle_seconds=0.3, group_seconds=0.8)
    state_path = tmp_path / "ws" / "autopilot_state.json"
    calls = []

    def submit(name, files):
        calls.append((name, sorted(Path(f).name for f in files)))
        return {"id": f"job{len(calls)}", "status": "queued"}

    jobs = {}
    watcher = Watcher(settings, AutopilotState(state_path), submit, get_job=lambda i: jobs[i], poll_seconds=0.05)
    watcher.start()
    try:
        (watch / "bolt_1.ply").write_bytes(b"x" * 100)
        (watch / "bolt_2.ply.part").write_bytes(b"x" * 10)            # still downloading
        (watch / "notes.txt.tmp").write_bytes(b"x")
        assert wait_until(lambda: watcher.status()["pending_items"])
        time.sleep(0.3)
        os.replace(watch / "bolt_2.ply.part", watch / "bolt_2.ply")    # download finished: joins the same item
        (out / "old").mkdir(parents=True)
        (out / "old" / "old.stl").write_bytes(b"x" * 100)             # our own output: ignored
        assert wait_until(lambda: len(calls) == 1)
        assert calls[0] == ("bolt", ["bolt_1.ply", "bolt_2.ply"])

        (watch / "cover").mkdir()
        (watch / "cover" / "top.ply").write_bytes(b"y" * 100)
        (watch / "cover" / "bottom.ply").write_bytes(b"y" * 100)
        assert wait_until(lambda: len(calls) == 2)
        assert calls[1] == ("cover", ["bottom.ply", "top.ply"])
        time.sleep(1.5)
        assert len(calls) == 2  # nothing is submitted twice

        jobs["job1"] = {"id": "job1", "status": "done", "finished": time.time(), "error": None,
                        "output": {"exports": ["bolt.stl"], "warnings": ["low overlap"]}}
        jobs["job2"] = {"id": "job2", "status": "running", "finished": None, "error": None, "output": {}}
        history = watcher.status()["history"]
        assert [h["item"] for h in history] == ["cover", "bolt"]
        assert history[1]["status"] == "done" and history[1]["warnings"] == ["low overlap"]
        assert history[0]["status"] == "running"
    finally:
        watcher.stop()
    assert not watcher.watching

    restarted = Watcher(settings, AutopilotState(state_path), submit, get_job=lambda i: {}[i])  # new job manager
    assert restarted.poll_once(force=True) == []
    assert len(calls) == 2
    assert restarted.status()["history"][0]["status"] == "interrupted"  # job2 unknown after restart

    (watch / "cover" / "top.ply").write_bytes(b"z" * 150)  # re-exported scan: processed again
    assert len(restarted.poll_once(force=True)) == 1 and calls[-1][0] == "cover"


# --------------------------------------------------------------------------- web
def test_autopilot_settings_api(tmp_path):
    watch = tmp_path / "watch"
    watch.mkdir()
    with TestClient(create_app(tmp_path / "ws")) as client:
        defaults = client.get("/api/autopilot/settings").json()
        assert defaults["formats"] == ["stl", "ply"] and defaults["watch_enabled"] is False

        def put(body):
            return client.put("/api/autopilot/settings", json=body)

        assert put({"watch_enabled": True, "watch_folder": str(tmp_path / "missing")}).status_code == 400
        assert put({"watch_enabled": True, "watch_folder": ""}).status_code == 400
        assert put({"formats": ["docx"]}).status_code == 400
        assert put({"not_a_setting": 1}).status_code == 400
        assert put({"watch_folder": str(watch), "output_folder": str(watch)}).status_code == 400
        assert put({"reference_id": "0123456789ab"}).status_code == 400
        assert "detail" in put({"preset": "extreme"}).json()

        res = put({"watch_enabled": True, "watch_folder": str(watch), "output_folder": str(tmp_path / "out" / "x"),
                   "settle_seconds": 1, "mesh": {"watertight": True}})
        assert res.status_code == 200, res.text
        assert res.json()["mesh"]["watertight"] is True and res.json()["mesh"]["method"] == "poisson"
        assert (tmp_path / "out" / "x").is_dir()
        status = client.get("/api/autopilot/status").json()
        assert status["watching"] is True and status["enabled"] is True

        assert put({"watch_enabled": False}).json()["watch_enabled"] is False
        assert client.get("/api/autopilot/status").json()["watching"] is False

        assert client.post("/api/autopilot/run", json={"asset_ids": []}).status_code == 400
        assert client.post("/api/autopilot/run", json={"asset_ids": ["0123456789ab"]}).status_code == 404
        bad = client.post("/api/autopilot/upload", files=[("files", ("part.step", b"x", "application/octet-stream"))])
        assert bad.status_code == 400


def test_autopilot_upload_end_to_end(tmp_path):
    a, _ = simulate_scan([0.2, 0.1, 1.0], seed=1, n_points=60_000)
    b, _ = simulate_scan([-0.1, -0.2, -1.0], seed=2, n_points=60_000, transform=random_rigid(7))
    save(a, tmp_path / "gear_1.ply")
    save(b, tmp_path / "gear_2.ply")
    out = tmp_path / "out"

    with TestClient(create_app(tmp_path / "ws")) as client:
        assert client.put("/api/autopilot/settings", json={"output_folder": str(out)}).status_code == 200
        with open(tmp_path / "gear_1.ply", "rb") as fa, open(tmp_path / "gear_2.ply", "rb") as fb:
            job = client.post("/api/autopilot/upload", files=[("files", ("gear_1.ply", fa, "application/octet-stream")),
                                                              ("files", ("gear_2.ply", fb, "application/octet-stream"))])
        assert job.status_code == 200, job.text
        job = wait_job(client, job.json()["id"])

        stl, ply, report_path = out / "gear" / "gear.stl", out / "gear" / "gear.ply", out / "gear" / "report.json"
        assert stl.exists() and ply.exists() and report_path.exists()
        assert sorted(job["output"]["exports"]) == sorted(str(p) for p in (stl, ply, report_path))
        assert not list((tmp_path / "ws" / "uploads").iterdir())  # temporary uploads removed

        dims = np.asarray(describe(load(stl))["dimensions"])
        assert np.all(np.abs(dims - GT_DIMS) < 0.3), f"STL dims {dims} vs truth {GT_DIMS}"

        report = json.loads(report_path.read_text())
        assert report["stages"]["merge"]["scans"][1]["fitness"] > 0.2
        assert report["stages"]["clean"]["scans"] and report["stages"]["mesh"]["deviation"]["p95"] < 0.2
        assert report["final"]["dimensions"] and isinstance(report["warnings"], list)
        assert report["mesh_id"] == job["output"]["mesh_id"]
        mesh = client.get(f"/api/assets/{report['mesh_id']}").json()
        assert mesh["kind"] == "mesh" and mesh["name"] == "gear"

        history = client.get("/api/autopilot/status").json()["history"]
        assert history[0]["item"] == "gear" and history[0]["status"] == "done"
        assert history[0]["files"] == ["gear_1.ply", "gear_2.ply"]


# --------------------------------------------------------------------------- terminal runner
def test_cli_runner_once(tmp_path, capsys):
    watch, out = tmp_path / "incoming", tmp_path / "results"
    watch.mkdir()
    scan, _ = simulate_scan([0.2, 0.1, 1.0], seed=3, n_points=40_000, add_table=True)
    save(scan, watch / "widget_1.ply")
    (watch / "widget_2.ply.crdownload").write_bytes(b"partial")

    entries = run_autopilot_cli(watch, out, AutopilotSettings(formats=["stl", "obj"]), once=True)
    assert len(entries) == 1 and entries[0]["status"] == "done", entries
    assert (out / "widget" / "widget.stl").exists() and (out / "widget" / "widget.obj").exists()
    report = json.loads((out / "widget" / "report.json").read_text())
    assert report["item"] == "widget" and "warnings" in report
    assert "widget" in capsys.readouterr().out

    assert run_autopilot_cli(watch, out, AutopilotSettings(), once=True) == []  # already processed
