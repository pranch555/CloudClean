"""Scan vs CAD inspection: accuracy with known answers, STEP loading and the web API."""
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from fastapi.testclient import TestClient

from cloudclean.cad import cad_backend, load_cad
from cloudclean.compare import CompareParams, ReferenceSurface, compare_to_reference, rotation_angle
from cloudclean.io import load, save
from cloudclean.web.server import create_app
from tests.synthetic import ground_truth_mesh, random_rigid, simulate_scan

DATA = Path(__file__).parent / "data"
NOISE = 0.02
quiet = lambda msg: None  # noqa: E731


@pytest.fixture(scope="module")
def gt():
    return ground_truth_mesh()


def test_recovers_pose_and_deviation(gt):
    pose = random_rigid(11)
    scan, clean = simulate_scan([0.3, -0.2, 1.0], seed=3, n_points=80_000, noise=NOISE, transform=pose)
    res = compare_to_reference(scan, gt, CompareParams(), log=quiet)
    report = res["report"]
    T = np.asarray(report["transform"])
    err = T @ pose                                  # identity when the scan pose was undone exactly
    assert rotation_angle(err[:3, :3]) < 0.2
    assert np.linalg.norm(clean @ err[:3, :3].T + err[:3, 3] - clean, axis=1).max() < 0.05

    dev = res["deviation"]
    assert len(dev) == len(res["aligned"].points)
    s = report["stats"]
    assert s["excluded"] > 0 and np.isnan(dev).sum() == s["excluded"]   # scattered noise / debris
    assert abs(np.median(dev[np.isfinite(dev)])) < 0.005
    assert s["p05"] > -3 * NOISE and s["p95"] < 3 * NOISE
    assert np.median(np.abs(dev[np.isfinite(dev)])) < 1.5 * NOISE     # |N(0, s)| median = 0.67 s
    assert s["within_tolerance_pct"] >= 95
    assert abs(report["scale_estimate"]["percent"]) < 0.05
    assert not report["alignment"].get("ambiguous")
    hist = report["histogram"]
    assert len(hist["counts"]) == 41 and len(hist["edges"]) == 42
    assert sum(hist["counts"]) + hist["underflow"] + hist["overflow"] == s["points"]
    assert 40 < report["coverage"]["covered_area_pct"] < 80                # one-sided scan
    assert len(res["reference_distance"]) == len(res["reference"].vertices)
    dims = report["dimensions"]
    assert abs(dims["difference"][0]) < 0.3                                  # the top view spans full length


def test_scale_is_reported_not_applied(gt):
    scan, _ = simulate_scan([0.3, -0.2, 1.0], seed=4, n_points=80_000, noise=NOISE)
    scan.scale(1.005, center=np.zeros(3))
    scan.transform(random_rigid(5))
    res = compare_to_reference(scan, gt, CompareParams(), log=quiet)
    report = res["report"]
    assert abs(report["scale_estimate"]["percent"] - 0.5) < 0.1
    assert report["scale_estimate"]["applied"] is False
    T = np.asarray(report["transform"])
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-9)    # rigid only
    assert report["stats"]["mean"] > 0.02 and report["stats"]["p95"] > 0.15   # too big = material outside
    assert any("larger" in w for w in report["warnings"])


def test_bump_shows_positive_deviation(gt):
    scan, clean = simulate_scan([0.3, -0.2, 1.0], seed=5, n_points=80_000, noise=NOISE, outlier_fraction=0.0)
    pts = np.asarray(scan.points)
    _, _, normals = ReferenceSurface(gt, quiet).query(pts)
    centre = clean[np.argmax(clean[:, 2])]
    patch = np.linalg.norm(pts - centre, axis=1) < 6.0
    pts[patch] += 0.5 * normals[patch]
    scan.points = o3d.utility.Vector3dVector(pts)
    scan.transform(random_rigid(9))
    dev = compare_to_reference(scan, gt, CompareParams(), log=quiet)["deviation"]
    assert patch.sum() > 500
    assert abs(np.nanmedian(dev[patch]) - 0.5) < 0.03
    assert abs(np.nanmedian(dev[~patch])) < 0.005                           # the bump does not tilt the fit


@pytest.mark.skipif(cad_backend() is None, reason="no CAD kernel installed")
def test_step_box_loads_in_millimetres():
    mesh = load_cad(DATA / "box_40x30x20.step")
    assert mesh.is_watertight() and len(mesh.triangles) == 12
    assert np.allclose(mesh.get_min_bound(), [5, -10, 2], atol=1e-4)
    assert np.allclose(mesh.get_max_bound(), [45, 20, 22], atol=1e-4)
    assert abs(mesh.get_volume() - 24000) < 0.01
    via_io = load(DATA / "box_40x30x20.step")
    assert np.allclose(via_io.get_max_bound() - via_io.get_min_bound(), [40, 30, 20], atol=1e-4)

    cyl = load_cad(DATA / "cylinder_r10_h25.step")          # curved faces: default deflection is metrology grade
    v = np.asarray(cyl.vertices)
    assert np.allclose(np.hypot(v[:, 0], v[:, 1]), 10, atol=1e-4) and np.isclose(v[:, 2].max(), 25, atol=1e-4)
    samples = np.asarray(cyl.sample_points_uniformly(20_000).points)
    side = (samples[:, 2] > 0.01) & (samples[:, 2] < 24.99)
    assert 10 - np.hypot(samples[side, 0], samples[side, 1]).min() < 0.01


@pytest.mark.skipif(cad_backend() is None, reason="no CAD kernel installed")
def test_box_scan_against_step_flags_symmetry():
    box = load_cad(DATA / "box_40x30x20.step")
    dense = box.sample_points_uniformly(120_000, use_triangle_normal=True)
    pts, nrm = np.asarray(dense.points), np.asarray(dense.normals)
    keep = nrm[:, 2] > -0.5                                   # no underside, like a scan on a table
    rng = np.random.default_rng(0)
    scan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep] + rng.normal(scale=NOISE, size=(keep.sum(), 3))))
    scan.transform(random_rigid(3))
    report = compare_to_reference(scan, box, CompareParams(tolerance=0.05), log=quiet)["report"]
    assert report["stats"]["within_tolerance_pct"] > 95
    assert report["alignment"].get("ambiguous")               # 180 deg about the vertical axis fits equally well
    assert np.allclose(report["dimensions"]["scan"][:2], [40, 30], atol=0.2)


def _wait(client, job, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        current = {j["id"]: j for j in client.get("/api/jobs").json()}[job["id"]]
        if current["status"] in ("done", "failed", "cancelled"):
            assert current["status"] == "done", "\n".join(current["logs"][-40:]) + f"\n{current['error']}"
            return current
        time.sleep(0.5)
    raise TimeoutError(job["title"])


def test_compare_api(tmp_path, gt):
    save(gt, tmp_path / "part_cad.stl")
    scan, _ = simulate_scan([0.2, 0.1, 1.0], seed=8, n_points=60_000, transform=random_rigid(4))
    save(scan, tmp_path / "part_scan.ply")

    with TestClient(create_app(tmp_path / "ws")) as client:
        started = client.post("/api/import-paths", json={"paths": [str(tmp_path / "part_cad.stl"),
                                                                   str(tmp_path / "part_scan.ply")]}).json()["jobs"]
        ref_id, scan_id = (_wait(client, j)["result"][0] for j in started)

        assert client.post("/api/compare", json={"scan_id": scan_id, "reference_id": ref_id,
                                                 "params": {"nope": 1}}).status_code == 400
        assert client.post("/api/compare", json={"scan_id": ref_id, "reference_id": scan_id}).status_code == 400
        assert client.post("/api/compare", json={"scan_id": scan_id, "reference_id": "0" * 12}).status_code == 400

        job = client.post("/api/compare", json={"scan_id": scan_id, "reference_id": ref_id,
                                                "params": {"tolerance": 0.1}}).json()
        done = _wait(client, job)
        compare_id, coverage_id = done["result"]
        assert done["output"]["compare_id"] == compare_id and done["progress"]["fraction"] == 1.0

        cmp_meta = client.get(f"/api/assets/{compare_id}").json()
        assert cmp_meta["kind"] == "pointcloud" and cmp_meta["parents"] == [scan_id, ref_id]
        assert [s["name"] for s in cmp_meta["scalars"]] == ["deviation"]
        assert cmp_meta["report"]["stats"]["within_tolerance_pct"] > 95
        cov_meta = client.get(f"/api/assets/{coverage_id}").json()
        assert cov_meta["kind"] == "mesh" and [s["name"] for s in cov_meta["scalars"]] == ["reference_distance"]

        preview_points = len(load_preview(client, compare_id).points)
        raw = client.get(f"/api/assets/{compare_id}/scalars/deviation").content
        assert len(raw) == preview_points * 4
        values = np.frombuffer(raw, dtype="<f4")
        assert np.isnan(values).any() and np.nanmax(np.abs(values)) < 3.5

        page = client.get(f"/api/assets/{compare_id}/inspection-report")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        text = page.text
        within = cmp_meta["report"]["stats"]["within_tolerance_pct"]
        assert "PASS" in text and f"{within:.1f} %" in text and "<svg" in text and "part_scan" in text
        assert "http://" not in text.replace("http://www.w3.org/2000/svg", "")   # self-contained
        assert "FAIL" in client.get(f"/api/assets/{coverage_id}/inspection-report?pass_pct=100.1").text
        assert client.get(f"/api/assets/{scan_id}/inspection-report").status_code == 400


def load_preview(client, asset_id):
    path = Path(client.app.state.workspace.asset_dir(asset_id)) / "preview.ply"
    return o3d.io.read_point_cloud(str(path))
