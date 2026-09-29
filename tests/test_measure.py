"""Measuring: thread analysis on synthetic ISO threads of exactly known geometry, standard matching, distances
and the measure API."""
import math
import time

import numpy as np
import open3d as o3d
import pytest
from fastapi.testclient import TestClient

from cloudclean.analysis import nearest_standard_thread, point_distance, region_mask, thread_analysis
from cloudclean.io import save
from cloudclean.web.server import create_app
from tests.synthetic import random_rigid


# --------------------------------------------------------------------------- synthetic threads
def iso_radius(u: np.ndarray, pitch: float, major: float) -> np.ndarray:
    """ISO 68-1 basic profile radius at thread phase u (crest centred on integer u): crest flat P/8 at the major
    diameter, 60 deg flanks, root flat P/4 at the basic minor diameter D - 1.0825 P."""
    H = math.sqrt(3) / 2 * pitch
    d = np.abs(u - np.round(u)) * pitch
    return np.where(d <= pitch / 16, major / 2,
                    np.where(d >= 3 * pitch / 8, major / 2 - 5 * H / 8, major / 2 - (d - pitch / 16) * math.sqrt(3)))


def thread_mesh(major: float, pitch: float | None, length: float = 12.0, hand: int = 1, internal: bool = False,
                rows_per_pitch: int = 48, cols: int = 1440) -> o3d.geometry.TriangleMesh:
    """Helical thread surface along +z centred on the origin (pitch None: a smooth tube)."""
    rows = int(round(length / (pitch or 1.0) * rows_per_pitch)) + 1
    t = np.linspace(-length / 2, length / 2, rows)
    th = np.linspace(0, 2 * math.pi, cols, endpoint=False)
    T, TH = np.meshgrid(t, th, indexing="ij")
    R = iso_radius((T - hand * pitch * TH / (2 * math.pi)) / pitch, pitch, major) if pitch else np.full_like(T, major / 2)
    V = np.stack([R * np.cos(TH), R * np.sin(TH), T], axis=-1).reshape(-1, 3)
    i, j = (x.ravel() for x in np.meshgrid(np.arange(rows - 1), np.arange(cols), indexing="ij"))
    jn = (j + 1) % cols
    a, b, c, d = i * cols + j, i * cols + jn, (i + 1) * cols + j, (i + 1) * cols + jn
    F = np.vstack([np.c_[a, b, c], np.c_[b, d, c]])  # outward winding
    if internal:
        F = F[:, [0, 2, 1]]  # a nut: the surface faces the axis
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    mesh.compute_vertex_normals()
    return mesh


def scan_of(mesh, seed: int, spacing: float = 0.06, noise: float = 0.02, view=None):
    """Scanner-like cloud: ~spacing point density, Gaussian noise, optional one-sided view, random rigid pose."""
    rng = np.random.default_rng(seed)
    pcd = mesh.sample_points_uniformly(int(mesh.get_surface_area() / spacing ** 2), use_triangle_normal=True)
    pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
    if view is not None:
        keep = nrm @ np.asarray(view, float) > -0.2
        pts, nrm = pts[keep], nrm[keep]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts + rng.normal(scale=noise, size=pts.shape)))
    cloud.normals = o3d.utility.Vector3dVector(nrm)
    T = random_rigid(seed)
    cloud.transform(T)
    return cloud, T


def axis_error_deg(T, axis) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(T[:3, 2] @ np.asarray(axis))))))


# --------------------------------------------------------------------------- thread analysis
@pytest.mark.parametrize("major, pitch, hand, view, designation, seed", [
    (8.0, 1.25, 1, None, "M8x1.25", 3),
    (12.0, 1.75, 1, [1.0, 0.0, 0.0], "M12x1.75", 4),   # one-sided scan (~240 deg of the circumference)
    (10.0, 1.0, -1, None, "M10x1", 5),                  # fine pitch, left-hand
])
def test_thread_analysis_iso(major, pitch, hand, view, designation, seed):
    cloud, T = scan_of(thread_mesh(major, pitch, hand=hand), seed, view=view)
    res = thread_analysis(cloud, log=lambda m: None)
    axis_err = axis_error_deg(T, res["axis"])
    minor = major - 1.0825 * pitch
    print(f"\n{designation}: pitch {res['pitch'] - pitch:+.5f} (se {res['pitch_se']:.5f}, helix "
          f"{res['pitch_helix_fit'] - pitch:+.5f}), major {res['major_diameter'] - major:+.4f}, minor "
          f"{res['minor_diameter'] - minor:+.4f}, pitch dia {res['pitch_diameter'] - (major - 0.6495 * pitch):+.4f}, "
          f"axis {axis_err:.4f} deg, flank {res['flank_angle_deg']:.2f}, crests {res['crest_count']}")
    assert abs(res["pitch"] - pitch) <= 0.005
    assert abs(res["pitch_helix_fit"] - pitch) <= 0.005
    assert abs(res["major_diameter"] - major) <= 0.05
    assert abs(res["minor_diameter"] - minor) <= 0.05
    assert abs(res["pitch_diameter"] - (major - 0.6495 * pitch)) <= 0.05
    assert axis_err <= 0.5
    assert abs(res["flank_angle_deg"] - 60) <= 2
    assert res["handedness"] == ("right" if hand > 0 else "left")
    assert res["kind"] == "external"
    assert res["confidence"] == "high" and res["crest_count"] >= 5
    assert nearest_standard_thread(res["major_diameter"], res["pitch"])[0]["designation"] == designation

    # the returned geometry is consistent: centre on the true axis, crest markers on the crests
    centre_local = np.linalg.inv(T) @ np.r_[res["center"], 1.0]
    assert np.hypot(*centre_local[:2]) < 0.02
    crest_local = (np.linalg.inv(T) @ np.c_[res["crest_points"], np.ones(len(res["crest_points"]))].T).T[:, :3]
    assert np.allclose(np.hypot(crest_local[:, 0], crest_local[:, 1]), major / 2, atol=0.05)
    assert len(res["profile"]["t"]) <= 600 and len(res["profile"]["t"]) == len(res["profile"]["r_max"])
    assert np.diff(res["crests"]) == pytest.approx(np.full(len(res["crests"]) - 1, pitch), abs=0.02)


def test_internal_thread():
    cloud, T = scan_of(thread_mesh(10.0, 1.5, internal=True), 6)
    res = thread_analysis(cloud, log=lambda m: None)
    print(f"\nM10x1.5 nut: pitch {res['pitch'] - 1.5:+.5f}, major {res['major_diameter'] - 10:+.4f}")
    assert res["kind"] == "internal" and res["handedness"] == "right"
    assert abs(res["pitch"] - 1.5) <= 0.005 and abs(res["major_diameter"] - 10.0) <= 0.05
    best = nearest_standard_thread(res["major_diameter"], res["pitch"], kind=res["kind"])[0]
    assert best["designation"] == "M10x1.5" and set(best["tolerance"]) == {"6H"}


def test_short_selection_of_four_threads():
    cloud, T = scan_of(thread_mesh(8.0, 1.25), 7)
    centre = T[:3, 3]
    # a ball on the axis that cuts ~5 mm of thread out of the 12 mm screw
    res = thread_analysis(cloud, {"spheres": [[*centre, math.hypot(4.0, 2.5)]]}, log=lambda m: None)
    print(f"\n4-thread ball: pitch {res['pitch'] - 1.25:+.5f}, crests {res['crest_count']}, length {res['length']:.2f}")
    assert res["length"] < 7.5
    assert abs(res["pitch"] - 1.25) <= 0.01 and abs(res["major_diameter"] - 8.0) <= 0.05
    # a patch on one side of the screw (~100 deg of the circumference, ~4 threads)
    side = centre + 4.0 * T[:3, 0]
    res = thread_analysis(cloud, {"spheres": [[*side, 3.0]]}, log=lambda m: None)
    print(f"side patch: pitch {res['pitch'] - 1.25:+.5f}, crests {res['crest_count']}, axis "
          f"{axis_error_deg(T, res['axis']):.3f} deg, confidence {res['confidence']}")
    assert abs(res["pitch"] - 1.25) <= 0.01 and axis_error_deg(T, res["axis"]) <= 0.5


def test_smooth_cylinder_is_not_a_thread():
    cloud, _ = scan_of(thread_mesh(8.0, None), 8)
    with pytest.raises(ValueError, match="No thread found"):
        thread_analysis(cloud, log=lambda m: None)
    with pytest.raises(ValueError, match="only"):
        thread_analysis(cloud, {"spheres": [[1e6, 0, 0, 1.0]]}, log=lambda m: None)


def test_region_validation():
    cloud, _ = scan_of(thread_mesh(8.0, 1.25, length=3.0, cols=180, rows_per_pitch=8), 9, spacing=0.3)
    n = len(cloud.points)
    assert region_mask(cloud, None).sum() == n
    box = {"box": {"min": (cloud.get_min_bound() - 1).tolist(), "max": (cloud.get_max_bound() + 1).tolist()}}
    assert region_mask(cloud, box).sum() == n
    for bad in ({"box": {"min": [0, 0], "max": [1, 1, 1]}}, {"spheres": [[0, 0, 0, -1]]},
                {"view_projection": [1] * 15, "polygon": [[0, 0], [1, 0], [0, 1]]}, {"nope": 1}, [1, 2]):
        with pytest.raises(ValueError):
            region_mask(cloud, bad)


# --------------------------------------------------------------------------- standards / distances
def test_nearest_standard_thread():
    m8 = nearest_standard_thread(7.9, 1.25)[0]
    assert m8["designation"] == "M8x1.25" and m8["series"] == "ISO metric coarse"
    assert m8["pitch_deviation"] == pytest.approx(0.0) and m8["major_diameter_deviation"] == pytest.approx(-0.1)
    assert m8["tolerance"]["6g"] == {"major_min": 7.76, "major_max": 7.972, "within": True}  # ISO 965-2 values
    assert m8["tolerance"]["6H"]["within"] is False
    assert nearest_standard_thread(7.95, 1.0)[0]["designation"] == "M8x1"
    assert nearest_standard_thread(6.30, 25.4 / 20)[0]["designation"] == "1/4-20 UNC"
    assert nearest_standard_thread(6.33, 25.4 / 28)[0]["designation"] == "1/4-28 UNF"
    g = nearest_standard_thread(20.9, 25.4 / 14)[0]
    assert g["designation"] == "G1/2 (BSPP)" and "tolerance" not in g
    with pytest.raises(ValueError):
        nearest_standard_thread(-1, 1)


def test_point_distance():
    res = point_distance([[0, 0, 0], [3, 4, 12]], axis="z")
    assert res["distance"] == pytest.approx(13) and (res["dx"], res["dy"], res["dz"]) == (3, 4, 12)
    assert res["along_axis"] == pytest.approx(12) and res["perpendicular"] == pytest.approx(5)
    res = point_distance([[1, 1, 1], [0, 0, 0]], axis=[2, 2, 2])
    assert res["along_axis"] == pytest.approx(math.sqrt(3)) and res["along_axis_signed"] < 0
    assert res["perpendicular"] == pytest.approx(0, abs=1e-12)
    for bad in (([[0, 0, 0]], None), ([[0, 0, 0], [1, 1, 1]], "w"), ([[0, 0, 0], [1, 1, 1]], [0, 0, 0])):
        with pytest.raises(ValueError):
            point_distance(*bad)


# --------------------------------------------------------------------------- API
def _wait(client, job, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        current = {j["id"]: j for j in client.get("/api/jobs").json()}[job["id"]]
        if current["status"] in ("done", "failed", "cancelled"):
            assert current["status"] == "done", "\n".join(current["logs"][-40:]) + f"\n{current['error']}"
            return current
        time.sleep(0.3)
    raise TimeoutError(job["title"])


def test_measure_api(tmp_path):
    T = random_rigid(11)
    mesh = thread_mesh(8.0, 1.25, rows_per_pitch=32, cols=720)
    mesh.transform(T)
    save(mesh, tmp_path / "screw.ply")
    tube, _ = scan_of(thread_mesh(8.0, None, length=6.0, cols=360, rows_per_pitch=16), 12, spacing=0.1)
    save(tube, tmp_path / "rod.ply")
    with TestClient(create_app(tmp_path / "ws")) as client:
        jobs = client.post("/api/import-paths", json={"paths": [str(tmp_path / "screw.ply"),
                                                                 str(tmp_path / "rod.ply")]}).json()["jobs"]
        screw, rod = (_wait(client, job)["result"][0] for job in jobs)

        res = client.post("/api/measure/thread", json={"asset_id": screw})
        assert res.status_code == 200, res.text
        thread = res.json()
        print(f"\nAPI mesh M8x1.25: pitch {thread['pitch'] - 1.25:+.5f}, major {thread['major_diameter'] - 8:+.4f}")
        assert abs(thread["pitch"] - 1.25) <= 0.005 and abs(thread["major_diameter"] - 8.0) <= 0.05
        assert axis_error_deg(T, thread["axis"]) <= 0.5
        assert [s["designation"] for s in thread["standards"]][0] == "M8x1.25" and len(thread["standards"]) == 3
        assert thread["handedness"] == "right" and thread["kind"] == "external"

        # crest-to-crest over five pitches, measured along the thread axis
        crests = thread["crest_points"]
        assert len(crests) >= 6
        res = client.post("/api/measure/distance", json={"asset_id": screw, "points": [crests[0], crests[5]],
                                                         "axis": thread["axis"]})
        assert res.status_code == 200, res.text
        dist = res.json()
        print(f"API crest-to-crest x5: along axis {dist['along_axis'] - 5 * 1.25:+.5f}")
        assert dist["along_axis"] == pytest.approx(5 * 1.25, abs=0.005)
        assert dist["perpendicular"] < 0.05 and max(dist["snap_distances"]) < 0.02
        assert set(dist) >= {"points", "distance", "dx", "dy", "dz", "delta", "axis", "along_axis"}

        plain = client.post("/api/measure/distance", json={"asset_id": screw, "points": [crests[0], crests[1]],
                                                           "axis": "z"}).json()
        assert plain["along_axis"] == pytest.approx(plain["dz"]) and plain["axis"] == [0.0, 0.0, 1.0]
        assert "along_axis" not in client.post("/api/measure/distance", json={
            "asset_id": screw, "points": [crests[0], crests[1]]}).json()

        bad = client.post("/api/measure/distance", json={"asset_id": screw, "points": [[0, 0, 0]]})
        assert bad.status_code == 400 and "two points" in bad.json()["detail"]
        bad = client.post("/api/measure/distance", json={"asset_id": screw, "points": [[0, 0, 0], [1, 1, 1]],
                                                         "axis": "q"})
        assert bad.status_code == 400
        assert client.post("/api/measure/thread", json={"asset_id": "0123456789ab"}).status_code == 404

        none = client.post("/api/measure/thread", json={"asset_id": rod})
        assert none.status_code == 400 and "No thread found" in none.json()["detail"]
        bad = client.post("/api/measure/thread", json={"asset_id": screw, "region": {"spheres": "x"}})
        assert bad.status_code == 400
