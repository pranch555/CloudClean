"""Live capture: simulated scanner, tracking, fusion accuracy, guidance, bridge upload, stream and bridge CLI."""
import struct
import threading
import time

import numpy as np
import open3d as o3d
import pytest
from fastapi.testclient import TestClient

from cloudclean.capture import bridge
from cloudclean.capture.drivers import available_drivers, create_driver
from cloudclean.capture.drivers.simulated import part_mesh
from cloudclean.capture.session import MAGIC, CaptureSession
from cloudclean.io import save
from cloudclean.web.routes_capture import manager_for
from cloudclean.web.server import create_app


def distance_to(mesh, points):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()


def capture_threads():
    return [t for t in threading.enumerate() if t.name.startswith("capture-") and t.is_alive()]


def run_frames(session, n, every=None):
    guidance = []
    for i in range(n):
        frame = session.driver.read()
        session.process_frame(frame)
        if every and (i + 1) % every == 0:
            guidance.append((frame.meta["episode"], session.update_guidance()))
    return guidance


def test_driver_registry():
    drivers = {d["id"]: d for d in available_drivers()}
    assert {"simulated", "revo_bridge", "folder", "revopoint_sdk", "realsense", "orbbec"} <= set(drivers)
    assert drivers["simulated"]["available"]
    sdk = drivers["revopoint_sdk"]
    assert not sdk["available"] and "customer@revopoint3d.com" in sdk["reason"]
    keys = {s["key"] for s in drivers["simulated"]["settings"]}
    assert {"scan_mode", "point_distance", "exposure_mode", "laser_brightness", "tracking", "surface",
            "turntable", "provide_pose"} <= keys
    with pytest.raises(ValueError):
        create_driver("simulated").connect({"scan_mode": "nonsense"})


def test_point_budget_caps_the_saved_cloud_without_changing_its_size():
    """The budget must thin the cloud evenly: fewer points, same measured part."""
    full = CaptureSession(create_driver("simulated"), {"realtime": False, "turntable": True})
    full.connect()
    full.driver.start()
    run_frames(full, 60)
    cloud, density = full.build_cloud()
    n = len(cloud.points)
    assert n > 20_000, f"need a reasonably dense cloud to test a budget, got {n}"

    budget = n // 4
    capped = CaptureSession(create_driver("simulated"),
                            {"realtime": False, "turntable": True, "max_points": budget})
    capped.connect()
    capped.driver.start()
    run_frames(capped, 60)
    small, small_density = capped.build_cloud()

    assert len(small.points) == budget
    assert len(small_density) == budget, "density must stay aligned with the points"
    big = np.asarray(cloud.get_max_bound()) - np.asarray(cloud.get_min_bound())
    cut = np.asarray(small.get_max_bound()) - np.asarray(small.get_min_bound())
    assert np.all(np.abs(big - cut) < 1e-6), f"budget changed the measured size: {big} vs {cut}"


def test_simulated_capture_with_pose():
    mesh = part_mesh("pebble")
    session = CaptureSession(create_driver("simulated"), {"realtime": False, "turntable": True})
    session.connect()
    session.driver.start()

    run_frames(session, 40)
    early = session.update_guidance(full=True)["coverage"]["completeness"]
    cursor = {}
    first = session.stream(cursor)
    assert first[0]["type"] == "reset" and cursor["sent"] == session.disp_n > 0
    # frames 60-75 contain the simulated fast-motion burst (t = 6.0-7.2 s at 10 fps)
    burst = run_frames(session, 40, every=1)
    run_frames(session, 120)
    final = session.update_guidance(full=True)

    updates = [m for m in session.stream(cursor) if isinstance(m, bytes) and struct.unpack_from("<I", m, 12)[0] & 4]
    assert updates, "density changes of already streamed points must be sent as indexed updates"
    count = struct.unpack_from("<I", updates[0], 8)[0]
    assert len(updates[0]) == 16 + count * (12 + 4 + 4)
    assert np.frombuffer(updates[0], "<u4", count, 16 + count * 16).max() < session.disp_n

    fast = [g for episode, g in burst if episode == "fast"]
    assert fast and any(g["speed"]["too_fast"] and any(m["code"] == "too_fast" for m in g["messages"]) for g in fast)
    assert session.stats["dropped"]["too_fast"] > 0
    assert not any(g["speed"]["too_fast"] for episode, g in burst[:15])

    cloud, density = session.build_cloud()
    err = distance_to(mesh, np.asarray(cloud.points))
    assert np.percentile(err, 95) < 0.1, np.percentile(err, [50, 95, 99])
    assert len(density) == len(cloud.points) and np.nanmax(density) > 0

    late = final["coverage"]["completeness"]
    assert late > early + 0.1 and late > 0.6, (early, late)
    # the part rests on the turntable: its underside is never seen
    underside = [h for h in final["holes"] if h["direction"][2] < -0.8]
    assert underside, final["holes"]
    assert "underside" in underside[0]["hint"] or "Flip" in underside[0]["hint"]


def test_tracking_without_pose():
    session = CaptureSession(create_driver("simulated"),
                             {"realtime": False, "turntable": False, "provide_pose": False,
                              "simulate_mistakes": False, "seed": 3})
    session.connect()
    session.driver.start()
    run_frames(session, 90)
    tracked = [t for t in session.trajectory if t["pose"] is not None and t["fused"]]
    assert len(tracked) >= 85
    assert session.stats["dropped"]["tracking_lost"] <= 2
    T0, G0 = tracked[0]["pose"], np.asarray(tracked[0]["true_pose"])
    trans, rot = [], []
    for t in tracked:
        E = np.linalg.inv(np.linalg.inv(G0) @ np.asarray(t["true_pose"])) @ (np.linalg.inv(T0) @ t["pose"])
        trans.append(np.linalg.norm(E[:3, 3]))
        rot.append(np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))))
    assert max(trans) < 1.5 and max(rot) < 0.6, (max(trans), max(rot))


def wait_for(fn, timeout=60.0, poll=0.1):
    start = time.time()
    while time.time() - start < timeout:
        value = fn()
        if value:
            return value
        time.sleep(poll)
    raise TimeoutError


def world_scan(seed: int, t: float):
    """A partial scan exported in the part's coordinates, like a Revo Metro export."""
    drv = create_driver("simulated")
    drv.connect({"realtime": False, "simulate_mistakes": False, "seed": seed})
    frames = [drv.render(t + k * 0.1) for k in range(5)]
    pts = np.vstack([(f.pose[:3, :3] @ f.points.T).T + f.pose[:3, 3] for f in frames])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(np.vstack([f.colors for f in frames]) / 255.0)
    return cloud


def test_bridge_upload_creates_session_and_asset(tmp_path):
    save(world_scan(1, 0.0), tmp_path / "export_1.ply")
    save(world_scan(1, 6.0), tmp_path / "export_2.ply")
    with TestClient(create_app(tmp_path / "ws")) as client:
        assert client.get("/api/capture/status").json()["active"] is False
        with open(tmp_path / "export_1.ply", "rb") as f:
            res = client.post("/api/capture/bridge/upload", files=[("files", ("export_1.ply", f))]).json()
        assert res["results"][0]["queued"] and res["results"][0]["import_job"]
        assert res["status"]["driver"] == "revo_bridge"
        with open(tmp_path / "export_2.ply", "rb") as f:
            client.post("/api/capture/bridge/upload", files=[("files", ("export_2.ply", f))],
                        data={"import_asset": "false"})
        status = wait_for(lambda: (s := client.get("/api/capture/status").json())["scans"] >= 2 and s)
        assert status["state"] == "running" and status["points"] > 10_000
        # further exports are checked and kept aside until the user decides (default fuse_mode "ask")
        assert status["pending_scans"] == 1 and status["unsaved"]
        pending = client.get("/api/capture/pending").json()["pending"]
        assert pending[0]["name"] == "export_2" and pending[0]["assessment"]["reasons"]
        fused = client.post(f"/api/capture/pending/{pending[0]['scan_id']}", json={"decision": "fuse"})
        assert fused.status_code == 200, fused.text
        status = fused.json()["status"]
        assert status["pending_scans"] == 0
        assert status["guidance"]["coverage"]["observed_area_mm2"] > 1000

        bad = client.post("/api/capture/bridge/upload", files=[("files", ("notes.docx", b"x"))])
        assert bad.status_code == 400

        saved = client.post("/api/capture/save", json={"name": "bridge capture", "auto_process": False}).json()
        asset = client.get(f"/api/assets/{saved['asset']['id']}").json()
        assert saved["job"] is None
        assert asset["operation"] == "capture" and asset["params"]["driver"] == "revo_bridge"
        assert asset["report"]["scans"][1]["fitness"] > 0.5
        assert [s["name"] for s in asset["scalars"]] == ["density"]
        assert client.get(f"/api/assets/{asset['id']}/scalars/density").status_code == 200

        job = client.get("/api/jobs").json()
        imported = [j for j in job if j["kind"] == "import"]
        assert len(imported) == 1
        # connecting another driver is refused only while the capture runs
        assert client.post("/api/capture/connect", json={"driver": "simulated"}).status_code == 409
        client.post("/api/capture/stop")
        assert client.post("/api/capture/connect", json={"driver": "simulated", "settings": {"realtime": False}}
                           ).status_code == 200
        assert client.post("/api/capture/connect", json={"driver": "nope"}).status_code == 400
    assert not capture_threads()


def test_websocket_stream(tmp_path):
    with TestClient(create_app(tmp_path / "ws")) as client:
        r = client.post("/api/capture/connect", json={"driver": "simulated",
                                                      "settings": {"realtime": False, "max_frames": 25,
                                                                   "speed_limit": 400}})
        assert r.status_code == 200, r.text
        assert client.post("/api/capture/start").json()["state"] == "running"
        got = {"binary": None, "live": None, "status": None, "guidance": None, "reset": None, "live_view": None}
        with client.websocket_connect("/api/capture/stream") as ws:
            for _ in range(600):
                msg = ws.receive()
                if msg.get("bytes"):
                    flags = struct.unpack_from("<IIII", msg["bytes"])[3]
                    key = "live" if flags & 8 else "binary"          # the current frame vs model chunks
                    got[key] = got[key] or msg["bytes"]
                elif msg.get("text"):
                    import json

                    data = json.loads(msg["text"])
                    if data["type"] in got:
                        got[data["type"]] = got[data["type"]] or data
                if all(got.values()):
                    break
        assert all(got.values()), {k: v is not None for k, v in got.items()}
        data = got["binary"]
        magic, version, count, flags = struct.unpack_from("<IIII", data)
        assert magic == MAGIC == 0x43435054 and version == 1 and count > 0
        assert flags & 1 and flags & 2 and not flags & 4
        assert len(data) == 16 + count * (12 + 3 + 4)
        xyz = np.frombuffer(data, "<f4", count * 3, 16).reshape(-1, 3)
        assert np.all(np.isfinite(xyz)) and np.abs(xyz).max() < 200
        density = np.frombuffer(data, "<f4", count, 16 + count * 15)
        assert np.all(density >= 0)
        live_n = struct.unpack_from("<IIII", got["live"])[2]
        assert live_n > 0 and len(got["live"]) == 16 + live_n * 12      # the live frame: xyz only
        assert len(got["live_view"]["pose"]) == 16
        status = got["status"]
        assert status["active"] and status["driver"] == "simulated" and "frames" in status
        assert got["guidance"]["status"]["message"]

        wait_for(lambda: client.get("/api/capture/status").json()["state"] == "finished")
        assert client.post("/api/capture/discard").json()["active"] is False
    assert not capture_threads()


def test_bridge_cli_once(tmp_path, capsys):
    watch = tmp_path / "exports"
    watch.mkdir()
    save(world_scan(2, 0.0), watch / "part_scan.ply")
    (watch / "readme.txt.bak").write_text("ignored")
    state = tmp_path / "state.json"
    with TestClient(create_app(tmp_path / "ws")) as client:
        args = ["--server", "http://testserver", "--watch", str(watch), "--once", "--settle", "0",
                "--no-import", "--state", str(state)]
        assert bridge.main(args, transport=client._transport) == 0
        out = capsys.readouterr().out
        assert "Uploading part_scan.ply" in out and "Uploaded 1 file" in out
        manager = manager_for(tmp_path / "ws")
        assert manager.session is not None and manager.session.driver.received == 1

        assert bridge.main(args, transport=client._transport) == 0
        out = capsys.readouterr().out
        assert "Uploading" not in out and "Uploaded 0 file" in out
        assert manager.session.driver.received == 1
        assert not [j for j in client.get("/api/jobs").json() if j["kind"] == "import"]
    assert not capture_threads()


def test_driver_commands_route(tmp_path):
    """Driver-specific actions (the MetroY's marker map) go through one route; a driver without them says so."""
    with TestClient(create_app(tmp_path / "ws")) as client:
        assert client.post("/api/capture/driver/map_markers").status_code == 409        # no session yet
        client.post("/api/capture/connect", json={"driver": "simulated", "settings": {"realtime": False}})
        r = client.post("/api/capture/driver/map_markers")
        assert r.status_code == 400 and "no command" in r.json()["detail"]
        assert client.get("/api/capture/status").json()["device"] == {}
        client.post("/api/capture/discard")


def test_status_survives_infinite_numbers(tmp_path):
    """No fit yet gives an infinite rmse (and a driver may report inf / NaN): the status still goes out, as null."""
    with TestClient(create_app(tmp_path / "ws")) as client:
        client.post("/api/capture/connect", json={"driver": "simulated", "settings": {"realtime": False}})
        session = manager_for(tmp_path / "ws").session
        session.tracking = {"state": "lost", "fitness": 0.0, "rmse_mm": float("inf")}
        session.guidance = {"type": "guidance", "tracking": dict(session.tracking), "speed": {"value_mm_s": float("nan")}}
        session.driver.status = lambda: {"distance_mm": float("inf")}
        r = client.get("/api/capture/status")
        assert r.status_code == 200
        assert r.json()["device"] == {"distance_mm": None} and r.json()["guidance"]["tracking"]["rmse_mm"] is None
        assert client.post("/api/capture/stop").status_code == 200
        client.post("/api/capture/discard")
