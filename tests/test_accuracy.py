"""Accuracy checks (cloudclean/accuracy.py, Contract 5) on synthetic parts of exactly known geometry: robust fits,
robust dimensions vs the axis-aligned box, operation drift (clean / scale / rigid move / mesh), scan-vs-scan
comparison (relative scale, surface offset), reference artefacts, and the API."""
import math
import time

import numpy as np
import open3d as o3d
import pytest
from fastapi.testclient import TestClient

from cloudclean import accuracy as acc
from cloudclean.clean import CleanParams, clean_point_cloud
from cloudclean.io import save
from cloudclean.mesh import MeshParams, reconstruct_mesh
from cloudclean.web.server import create_app
from tests import synthetic as syn

QUIET = lambda m: None  # noqa: E731
UNC_PITCH = 25.4 / 8


def rotation(seed):
    return syn.random_rigid(seed)


# --------------------------------------------------------------------------- fits
def test_fits_recover_known_geometry():
    rng = np.random.default_rng(0)
    # sphere cap with stray points
    u = rng.normal(size=(20_000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    cap = u[u[:, 2] > -0.3] * 10 + [1, 2, 3] + rng.normal(scale=0.03, size=(int((u[:, 2] > -0.3).sum()), 3))
    s = acc.fit_sphere(np.vstack([cap, rng.uniform(-10, 15, size=(300, 3))]))
    assert s["radius"] == pytest.approx(10, abs=0.002) and np.allclose(s["center"], [1, 2, 3], atol=0.003)
    # 250 deg of a tilted cylinder
    th, z = rng.uniform(0, math.radians(250), 30_000), rng.uniform(-20, 20, 30_000)
    C = np.c_[6 * np.cos(th), 6 * np.sin(th), z] + rng.normal(scale=0.03, size=(30_000, 3))
    T = rotation(3)
    c = acc.fit_cylinder(C @ T[:3, :3].T + T[:3, 3])
    assert c["radius"] == pytest.approx(6, abs=0.002)
    assert math.degrees(math.acos(abs(c["axis"] @ T[:3, 2]))) < 0.02
    assert 240 <= c["coverage_deg"] <= 270 and c["length"] == pytest.approx(40, abs=0.3)
    # plane and circle
    P = np.c_[rng.uniform(-10, 10, (5000, 2)), rng.normal(scale=0.02, size=5000)]
    f = acc.fit_plane(np.vstack([P, [[0, 0, 5.0]] * 20]))
    assert abs(f["normal"][2]) > 0.99999 and f["rms"] == pytest.approx(0.02, rel=0.1)
    ang = rng.uniform(0, 2 * math.pi, 2000)
    circ = acc.fit_circle(np.c_[3 + 4 * np.cos(ang), -1 + 4 * np.sin(ang)] + rng.normal(scale=0.01, size=(2000, 2)))
    assert circ["radius"] == pytest.approx(4, abs=0.002)


def test_thread_fit_pitch_and_diameters():
    mesh = syn.thread_rod_mesh(25.4, UNC_PITCH, 30.0, rows_per_pitch=48, cols=1080)
    full = syn.scan_surface(mesh, 0.15, 0.02, 1)
    th = acc.fit_thread(np.asarray(full.points), np.asarray(full.normals))
    assert abs(th["pitch"] / UNC_PITCH - 1) < 60e-6            # 60 ppm
    assert th["handedness"] == "right" and th["kind"] == "external" and th["both_flanks"]
    assert th["major_diameter"] == pytest.approx(25.4, abs=0.01)
    assert th["minor_diameter"] == pytest.approx(25.4 - 1.0825 * UNC_PITCH, abs=0.01)
    assert th["pitch_diameter"] == pytest.approx(25.4 - 0.6495 * UNC_PITCH, abs=0.01)
    assert th["lead_residual_pp"] < 0.005
    # seen from one end: one flank is hidden under the crests -> pitch still exact, no pitch diameter
    one = syn.scan_surface(mesh, 0.15, 0.02, 2, view=[0, 0, 1], view_limit=-0.2)
    th1 = acc.fit_thread(np.asarray(one.points), np.asarray(one.normals))
    assert abs(th1["pitch"] / UNC_PITCH - 1) < 100e-6
    assert th1["pitch_diameter"] is None and not th1["both_flanks"] and th1["warnings"]
    assert th1["major_diameter"] == pytest.approx(25.4, abs=0.015)
    # thread_offsets of the same scan against its own model: no offset
    edges = np.linspace(-12, 12, 5)
    off = acc.thread_offsets(th["model"], np.asarray(full.points), edges)
    assert all(abs(o["axial_offset"]) < 0.005 and abs(o["radial_offset"]) < 0.005 for o in off)


def test_smooth_rod_is_not_a_thread():
    rod = syn.scan_surface(syn.cylinder_mesh(10.0, 20.0, 360), 0.15, 0.02, 3)
    pts = np.asarray(rod.points)
    with pytest.raises(ValueError, match="No thread"):
        acc.fit_thread(pts[np.abs(pts[:, 2]) < 9])


# --------------------------------------------------------------------------- dimensions
def test_robust_dimensions_vs_axis_aligned_box():
    box = syn.scan_surface(syn.box_mesh((40.0, 30.0, 20.0)), 0.3, 0.01, 4)
    T = rotation(5)
    box.transform(T)
    stray = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(acc.apply_transform(T, [[60.0, 0, 0], [0, 40.0, 0]])))
    d = acc.robust_dimensions(box + stray)
    # the trimmed extents along the principal axes are the part size; the axis-aligned box is not
    assert d["length"] == pytest.approx(40, abs=0.1) and d["width"] == pytest.approx(30, abs=0.1)
    assert d["height"] == pytest.approx(20, abs=0.1)
    assert d["raw"]["length"] > 45                     # two stray points inflate the raw extent
    assert np.abs(np.sort(d["aabb"]["size"]) - [20, 30, 40]).max() > 1   # tilted: the axis-aligned box is not the part
    # measured in a given frame
    d2 = acc.robust_dimensions(box, frame=(d["frame"]["center"], d["frame"]["axes"]))
    assert d2["length"] == pytest.approx(d["length"], abs=0.02)


def test_measure_length_like_calipers():
    block = syn.scan_surface(syn.box_mesh((50.0, 20.0, 9.0)), 0.2, 0.02, 6)
    T = rotation(7)
    block.transform(T)
    m = acc.measure_length(block, "length")
    assert m["method"] == "faces" and m["measured"] == pytest.approx(50.0, abs=0.004)
    assert m["parallelism_deg"] < 0.05 and m["uncertainty"] < 0.002
    assert acc.resolve_direction("x") @ np.array([1, 0, 0]) == 1
    with pytest.raises(ValueError):
        acc.resolve_direction("sideways")


# --------------------------------------------------------------------------- drift
def test_drift_clean_moves_nothing():
    raw = syn.scan_surface(syn.cylinder_mesh(20.0, 30.0, 360), 0.3, 0.02, 8, outlier_fraction=0.01, debris=True)
    cleaned, _ = clean_point_cloud(raw, CleanParams(), QUIET)
    r = acc.operation_drift(raw, cleaned, "clean")
    assert r["identical_fraction"] == 1.0 and r["displacement"]["max"] == 0.0
    assert r["verdict"] == "unchanged" and "none was moved" in r["sentence"]


def test_drift_detects_resize_rigid_move_and_mesh():
    base = syn.scan_surface(syn.box_mesh((40.0, 30.0, 20.0)), 0.3, 0.01, 9)
    # a 1000 ppm scale: every dimension grows, the transform is flagged as not rigid
    S = np.diag([1.001, 1.001, 1.001, 1.0])
    scaled = acc.transformed_geometry(base, S)
    r = acc.operation_drift(base, scaled, "edit")
    assert r["verdict"] == "changed"
    assert r["dimension_change"]["length"] == pytest.approx(0.040, abs=0.004)
    rs = acc.operation_drift(base, scaled, "edit", transform=S)
    assert not rs["transform_check"]["rigid"] and rs["transform_check"]["scale_ppm"] == pytest.approx(1000, abs=1)
    assert rs["verdict"] == "changed" and "not rigid" in rs["sentence"]
    # a rigid move with its transform: "moved", nothing else changed
    T = rotation(10)
    moved = acc.transformed_geometry(base, T)
    rm = acc.operation_drift(base, moved, "edit", transform=T)
    assert rm["verdict"] == "moved" and rm["identical_fraction"] == pytest.approx(1.0)
    assert rm["transform_check"]["rigid"] and max(abs(v) for v in rm["dimension_change"].values()) < 1e-6
    # a mesh follows the cloud: small signed mean, reverse direction measured too
    mesh, _ = reconstruct_mesh(base, MeshParams(depth=8), QUIET)
    rmesh = acc.operation_drift(base, mesh, "mesh", max_points=40_000)
    assert abs(rmesh["displacement"]["signed_mean"]) < 0.01 and rmesh["reverse"] is not None
    assert abs(rmesh["reverse"]["signed_mean"]) < 0.02 and rmesh["identical_fraction"] is None


def test_transform_check():
    assert acc.transform_check(np.eye(4))["identity"]
    M = rotation(1)
    M[:3, :3] = M[:3, :3] @ np.diag([1, 1, -1])
    c = acc.transform_check(M)
    assert c["mirrored"] and not c["rigid"]
    with pytest.raises(ValueError):
        acc.transform_check(np.eye(3))


# --------------------------------------------------------------------------- scan vs scan
@pytest.fixture(scope="module")
def pebble_scans():
    gt = syn.ground_truth_mesh()
    a = syn.scan_surface(gt, 0.3, 0.02, 21, view=[0.2, 0.1, 1.0], view_limit=-0.45)
    b = syn.scan_surface(gt, 0.3, 0.02, 22, view=[-0.1, -0.2, 1.0], view_limit=-0.45)
    return a, b


def test_compare_scans_recovers_relative_scale(pebble_scans):
    a, b = pebble_scans
    T = rotation(23)
    S = np.diag([1.001, 1.001, 1.001, 1.0])          # scan B reads 1000 ppm large
    b_scaled = acc.transformed_geometry(b, T @ S)
    r = acc.compare_scans(a, b_scaled, log=QUIET)
    assert r["alignment"]["ambiguous"] is False
    assert r["scale_ppm"] == pytest.approx(1000, abs=60)
    assert r["separation"]["median_abs"] < 0.03 and r["overlap_fraction"] > 0.5
    assert r["verdict"] == "differ" and "ppm" in r["sentence"]
    # the same scans without a scale error agree
    same = acc.compare_scans(a, acc.transformed_geometry(b, T), transform=np.linalg.inv(T), log=QUIET)
    assert abs(same["scale_ppm"]) < 60 and same["verdict"] == "consistent"


def test_compare_scans_separates_offset_from_scale():
    # all-round scans: a one-sided view could hide an offset in a translation
    gt = syn.ground_truth_mesh()
    a = syn.scan_surface(gt, 0.35, 0.02, 24)
    b = syn.scan_surface(gt, 0.35, 0.02, 25)
    thick = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(b.points) + 0.05 * np.asarray(b.normals)))
    thick.normals = b.normals
    r = acc.compare_scans(a, thick, transform=np.eye(4), log=QUIET)
    assert r["offset_mm"] == pytest.approx(0.05, abs=0.01)
    assert abs(r["scale_with_offset_ppm"]) < 150
    assert r["separation"]["median"] == pytest.approx(0.05, abs=0.01)


# --------------------------------------------------------------------------- references
def test_reference_checks():
    bar = syn.scan_surface(syn.sphere_pair_mesh(20.0, 60.0, 120), 0.3, 0.02, 30)
    r = acc.reference_check(bar, {"type": "sphere_pair", "nominal": 60.0,
                                  "region_a": {"spheres": [[-30, 0, 0, 12]]}, "region_b": {"spheres": [[30, 0, 0, 12]]}})
    assert r["verdict"] == "ok" and abs(r["error"]) < 0.005 and r["type"] == "sphere_pair"
    off = acc.reference_check(bar, {"type": "sphere_pair", "nominal": 59.9,
                                    "region_a": {"spheres": [[-30, 0, 0, 12]]},
                                    "region_b": {"spheres": [[30, 0, 0, 12]]}})
    assert off["verdict"] == "off" and off["error_ppm"] == pytest.approx(0.1 / 59.9 * 1e6, rel=0.05)
    block = syn.scan_surface(syn.box_mesh((25.0, 10.0, 8.0)), 0.2, 0.02, 31)
    g = acc.reference_check(block, {"type": "known_length", "direction": "x", "nominal": 25.0})
    assert g["verdict"] == "ok" and abs(g["error"]) < 0.005 and g["details"]["method"] == "faces"
    pin = syn.scan_surface(syn.cylinder_mesh(8.0, 30.0, 720), 0.2, 0.02, 32)
    d = acc.reference_check(pin, {"type": "diameter", "nominal": 8.0,
                                  "region": {"box": {"min": [-5, -5, -12], "max": [5, 5, 12]}}})
    assert d["verdict"] == "ok" and abs(d["error"]) < 0.005
    thread = syn.scan_surface(syn.thread_rod_mesh(25.4, UNC_PITCH, 20.0, rows_per_pitch=40, cols=900), 0.15, 0.02, 33)
    t = acc.reference_check(thread, {"type": "thread_pitch"})
    assert t["nominal"] == pytest.approx(UNC_PITCH) and t["details"]["standard"].startswith("1-8")
    assert t["verdict"] == "ok" and abs(t["error_ppm"]) < 150
    for bad in ({"type": "gauge"}, {"type": "known_length", "nominal": 25.0}, {"type": "diameter", "nominal": -1},
                {"type": "sphere_pair", "nominal": 60}, "x"):
        with pytest.raises(ValueError):
            acc.reference_check(block, bad)


# --------------------------------------------------------------------------- API
def _wait(client, job, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        current = {j["id"]: j for j in client.get("/api/jobs").json()}[job["id"]]
        if current["status"] in ("done", "failed", "cancelled"):
            assert current["status"] == "done", "\n".join(current["logs"][-40:]) + f"\n{current['error']}"
            return current
        time.sleep(0.3)
    raise TimeoutError(job["title"])


def test_drift_clean_thins_free_scan_edges():
    """Outlier removal takes points off the free border of a one-sided scan: an extent that ends at such a border
    (not at a real face) shrinks a little, and drift reports it."""
    gt = syn.ground_truth_mesh()
    a = syn.scan_surface(gt, 0.3, 0.02, 21, view=[0.2, 0.1, 1.0], view_limit=-0.45)
    cleaned, _ = clean_point_cloud(a, CleanParams(), QUIET)
    r = acc.operation_drift(a, cleaned, "clean")
    assert r["identical_fraction"] == 1.0 and r["removed_surface_points"] > 0
    assert -0.1 < min(r["dimension_change"].values()) <= 0.0


def test_accuracy_api(tmp_path, pebble_scans):
    _, b = pebble_scans
    debris = syn.scan_surface(syn.box_mesh((2.0, 2.0, 2.0)), 0.3, 0.0, 1).translate([0, 0, 45.0])
    raw = syn.scan_surface(syn.ground_truth_mesh(), 0.3, 0.02, 20) + debris   # all-round scan
    save(raw, tmp_path / "a.ply")
    T = rotation(40)
    save(acc.transformed_geometry(b, T), tmp_path / "b.ply")
    block = syn.scan_surface(syn.box_mesh((25.0, 10.0, 8.0)), 0.25, 0.02, 41)
    save(block, tmp_path / "block.ply")
    with TestClient(create_app(tmp_path / "ws")) as client:
        jobs = client.post("/api/import-paths", json={"paths": [str(tmp_path / n) for n in
                                                                 ("a.ply", "b.ply", "block.ply")]}).json()["jobs"]
        a_id, b_id, block_id = (_wait(client, j)["result"][0] for j in jobs)

        # drift: an imported scan has no parent; a cleaned one moved nothing
        assert client.post("/api/accuracy/drift", json={"asset_id": a_id}).status_code == 400
        clean_id = _wait(client, client.post("/api/clean", json={"asset_ids": [a_id]}).json())["result"][0]
        res = client.post("/api/accuracy/drift", json={"asset_id": clean_id})
        assert res.status_code == 200, res.text
        drift = res.json()
        assert drift["verdict"] == "unchanged" and drift["parent_id"] == a_id and drift["operation"] == "clean"
        assert set(drift) >= {"displacement", "dimensions_before", "dimensions_after", "dimension_change", "sentence",
                              "parents"}
        assert drift["displacement"]["max"] == 0 and drift["identical_fraction"] == 1.0
        assert client.post("/api/accuracy/drift", json={"asset_id": "0123456789ab"}).status_code == 404

        # reference check on the block (a 25 mm gauge block)
        res = client.post("/api/accuracy/reference", json={
            "asset_id": block_id, "reference": {"type": "known_length", "direction": "length", "nominal": 25.0}})
        assert res.status_code == 200, res.text
        ref = res.json()
        assert ref["verdict"] == "ok" and abs(ref["error"]) < 0.005 and "sentence" in ref
        bad = client.post("/api/accuracy/reference", json={"asset_id": block_id, "reference": {"type": "nope"}})
        assert bad.status_code == 400

        dims = client.post("/api/accuracy/dimensions", json={"asset_id": block_id}).json()
        assert dims["dimensions"]["length"] == pytest.approx(25.0, abs=0.2)

        # scan vs scan: job, output and the aligned copy with a separation scalar
        assert client.post("/api/accuracy/compare-scans", json={"a_id": a_id, "b_id": a_id}).status_code == 400
        job = client.post("/api/accuracy/compare-scans", json={"a_id": clean_id, "b_id": b_id})
        assert job.status_code == 200, job.text
        done = _wait(client, job.json())
        comp = done["output"]["comparison"]
        assert abs(comp["scale_ppm"]) < 100 and comp["verdict"] == "consistent"
        aligned = client.get(f"/api/assets/{done['result'][0]}").json()
        assert aligned["operation"] == "compare_scans" and aligned["parents"] == [clean_id, b_id]
        assert any(s["name"] == "separation" for s in aligned["scalars"])
        # the aligned copy was moved rigidly: drift says so
        moved = client.post("/api/accuracy/drift", json={"asset_id": done["result"][0], "parent_id": b_id}).json()
        assert moved["verdict"] == "moved" and moved["transform_check"]["rigid"]


# --------------------------------------------------------------------------- fixes that came out of the investigation
def test_merge_check_ignores_complementary_thread_flanks():
    """Scans of a thread from its two ends see opposite flanks; that is not a doubled skin (fix 3). A scan that
    really sits 0.06 mm off still is."""
    from cloudclean.register import MergeParams, assess_pair

    mesh = syn.thread_rod_mesh(25.4, UNC_PITCH, 16.0, rows_per_pitch=40, cols=720)
    a = syn.scan_surface(mesh, 0.3, 0.01, 1, view=[0, 0, 1], view_limit=-0.2)
    b = syn.scan_surface(mesh, 0.3, 0.01, 2, view=[0, 0, -1], view_limit=-0.2)
    spacing = acc.estimate_spacing(np.asarray(a.points))
    _, info = assess_pair(b, a, MergeParams(method="none"), spacing, 2.0, log=QUIET)
    assert not info["doubled_surface"] and info["separation_ratio"] < 1.5
    assert 0.1 < info["layering_facing_fraction"] < 0.9 and {"layering_mm", "layering_p90_mm"} <= set(info)
    thick = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(b.points) + 0.06 * np.asarray(b.normals)))
    _, info = assess_pair(thick, a, MergeParams(method="none"), spacing, 2.0, log=QUIET)
    assert info["doubled_surface"] and info["layering_mm"] == pytest.approx(0.06, abs=0.015)


def test_thread_tool_falls_back_to_a_helix_fit():
    """One-flank views have no complete crest: the app's thread tool answers with the helix fit (fix 4), with the
    same keys, no pitch diameter and a warning that names the hidden flank."""
    from cloudclean.analysis import thread_analysis

    mesh = syn.thread_rod_mesh(25.4, UNC_PITCH, 24.0, rows_per_pitch=40, cols=900)
    one = syn.scan_surface(mesh, 0.15, 0.02, 3, view=[0, 0, 1], view_limit=-0.2)
    r = thread_analysis(one, log=QUIET)
    assert r["method"] == "helix fit" and not r["both_flanks"] and r["missing_flank"] == "-axis"
    assert abs(r["pitch"] / UNC_PITCH - 1) < 100e-6
    assert r["major_diameter"] == pytest.approx(25.4, abs=0.01)
    assert r["minor_diameter"] == pytest.approx(25.4 - 1.0825 * UNC_PITCH, abs=0.015)
    assert r["pitch_diameter"] is None and r["flank_angle_deg"] is None
    assert any("Only one flank" in w for w in r["warnings"]) and r["confidence"] in ("medium", "low")
    assert isinstance(r["axis_uncertainty_deg"], float) and r["crest_count"] >= 5
    assert len(r["crest_points"]) == r["crest_count"] and len(r["per_crest_pitches"]) == r["crest_count"] - 1
    assert np.allclose(r["per_crest_pitches"], UNC_PITCH, atol=0.01)
    assert {"kind", "handedness", "tpi", "pitch_se", "center", "axis", "length", "angular_coverage_deg", "points_used",
            "noise_rms", "profile", "crests", "notes", "thread_depth"} <= set(r)
    # both flanks seen: the crest-by-crest path answers as before
    full = syn.scan_surface(mesh, 0.15, 0.02, 4)
    r = thread_analysis(full, log=QUIET)
    assert r["method"] == "crest by crest" and r["both_flanks"] and r["pitch_diameter"] is not None


def test_smoothing_warnings():
    """Cloud denoise and mesh Laplacian smoothing change feature sizes: the catalogue and the run say so (fix 5)."""
    from cloudclean.edit import OP_CATALOGUE, apply_edits

    assert "thread crests" in OP_CATALOGUE["denoise"]["description"]
    assert "laplacian" in OP_CATALOGUE["smooth"]["args"]["method"]["description"]
    cloud = syn.scan_surface(syn.cylinder_mesh(10.0, 10.0, 180), 0.3, 0.02, 5)
    logs = []
    _, rep = apply_edits(cloud, [{"op": "denoise", "iterations": 1}], logs.append)
    assert any("WARNING" in m for m in logs) and "warning" in rep["ops"][0]
    mesh = syn.cylinder_mesh(10.0, 10.0, 90)
    logs = []
    _, rep = apply_edits(mesh, [{"op": "smooth", "method": "laplacian", "iterations": 1}], logs.append)
    assert any("WARNING" in m for m in logs) and "warning" in rep["ops"][0]
    logs = []
    _, rep = apply_edits(mesh, [{"op": "smooth", "iterations": 1}], logs.append)
    assert not any("WARNING" in m for m in logs) and "warning" not in rep["ops"][0]


def test_snap_projects_onto_a_local_surface_fit(tmp_path):
    """POST /api/measure/snap (fix 7): picked points land on a quadric of their neighbours, with an uncertainty."""
    cyl = syn.scan_surface(syn.cylinder_mesh(20.0, 30.0, 720), 0.2, 0.03, 6)
    save(cyl, tmp_path / "cyl.ply")
    save(syn.cylinder_mesh(20.0, 30.0, 360), tmp_path / "cyl_mesh.ply")
    rng = np.random.default_rng(2)
    th = rng.uniform(0, 2 * math.pi, 40)
    clicks = np.c_[10.2 * np.cos(th), 10.2 * np.sin(th), rng.uniform(-10, 10, 40)]
    with TestClient(create_app(tmp_path / "ws")) as client:
        jobs = client.post("/api/import-paths", json={"paths": [str(tmp_path / "cyl.ply"),
                                                                 str(tmp_path / "cyl_mesh.ply")]}).json()["jobs"]
        cloud_id, mesh_id = (_wait(client, j)["result"][0] for j in jobs)
        res = client.post("/api/measure/snap", json={"asset_id": cloud_id, "points": clicks.tolist()})
        assert res.status_code == 200, res.text
        out = res.json()
        assert set(out) >= {"points", "uncertainty_mm", "normals", "on_surface_fit", "moved_mm", "method"}
        snapped = np.asarray(out["points"])
        err = np.hypot(snapped[:, 0], snapped[:, 1]) - 10.0
        raw = np.asarray(cyl.points)[acc.cKDTree(np.asarray(cyl.points)).query(clicks)[1]]
        raw_err = np.hypot(raw[:, 0], raw[:, 1]) - 10.0
        assert np.abs(err).mean() < 0.6 * np.abs(raw_err).mean() and all(out["on_surface_fit"])
        unc = np.asarray(out["uncertainty_mm"])
        assert np.all((unc > 0) & (unc < 0.03))
        mesh_out = client.post("/api/measure/snap", json={"asset_id": mesh_id, "points": clicks[:3].tolist()}).json()
        assert mesh_out["uncertainty_mm"] == [None] * 3 and mesh_out["method"] == "closest point on the mesh"
        assert client.post("/api/measure/snap", json={"asset_id": cloud_id, "points": [[0, 0]]}).status_code == 400
        assert client.post("/api/measure/snap", json={"asset_id": "0123456789ab", "points": [[0, 0, 0]]}).status_code == 404
