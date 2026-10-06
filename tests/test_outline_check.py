"""The outline check: a part against its golden model from one photo (cloudclean/outline_check.py, outline_camera.py,
outline_synthetic.py, tools/outline_check/, docs/outline-check.md).

Small synthetic photos (the check sheet backlit, the part a dark silhouette, anti-aliased by jittered supersampling,
blur, noise, sRGB, JPEG) with known size errors are measured and compared with the truth. The full-size accuracy
table is tools/outline_check/validate.py."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cloudclean import outline_camera as oc
from cloudclean import outline_check as ock
from cloudclean import outline_synthetic as syn
from cloudclean import outline_thread as oth
from cloudclean import scale_sheet

ROOT = Path(__file__).resolve().parents[1]
QUIET = lambda m: None  # noqa: E731
SIZE = (1600, 1200)
HFOV = 55.0
DIST = (0.015, -0.02, 0.0002, -0.0001, 0.005)
BOLT_ERR = {"head_h": 0.05, "under": -0.10, "shank_d": 0.03}


def camera():
    return syn.phone_camera(*SIZE, HFOV, DIST, (6.0, -4.0))


def shoot(tmp: Path, name: str, part=None, tilt=0.0, az=30.0, seed=0, ss=3, **kw) -> tuple[Path, np.ndarray,
                                                                                            np.ndarray]:
    cam = camera()
    R, t = syn.view(280.0, tilt, az, 2.0, (3.0, -2.0, 0.0))
    img = syn.render(cam, R, t, part, ss=ss, seed=seed, **kw)
    path = tmp / name
    syn.save_photo(path, img, meta=syn.camera_meta(cam))
    return path, R, t


@pytest.fixture(scope="module")
def bolt_golden():
    return ock.GoldenPart(syn.bolt(segments=180), QUIET)


@pytest.fixture(scope="module")
def bolt_check(tmp_path_factory, bolt_golden):
    tmp = tmp_path_factory.mktemp("bolt")
    _, true = syn.part_params("bolt", BOLT_ERR)
    part = syn.place(syn.bolt(**true, segments=720), (1, 0, 0), 4.0, -6.0, 23.0)
    path, _, _ = shoot(tmp, "bolt.jpg", part, seed=3)
    rep = ock.check_photos(bolt_golden, [path], camera(), ock.OutlineParams(bar_mm=100.0, mc_samples=8),
                           out_dir=tmp / "out", log=QUIET)
    return rep, syn.expected("bolt", *syn.part_params("bolt", BOLT_ERR)), tmp


# --------------------------------------------------------------------------- camera and photos
def test_camera_projection_round_trip_and_json(tmp_path):
    cam = camera()
    uv = np.random.default_rng(0).uniform([0, 0], SIZE, (500, 2))
    back = cam.distort_normalized(cam.normalized(uv))
    assert np.abs(back - uv).max() < 1e-7
    cam.cov = np.diag([1.0, 0.5, 0.5, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6])
    path = cam.save(tmp_path / "cam.json")
    again = oc.Camera.load(path)
    assert np.allclose(again.K, cam.K) and np.allclose(again.dist, cam.dist) and np.allclose(again.cov, cam.cov)
    assert again.f_rel_sigma() == pytest.approx(1.0 / cam.f)
    # the same camera for a photo stored at half the size: pixel centres stay consistent
    half = cam.resized((800, 600))
    X = np.array([[10.0, -20.0, 300.0], [-50.0, 40.0, 280.0]])
    assert np.allclose((cam.project(X) + 0.5) / 2 - 0.5, half.project(X), atol=1e-9)
    assert cam.matches({"make": "Other", "width": 1600, "height": 1200})


def test_metroy_camera_from_camparam():
    cam = oc.from_metroy(ROOT / "tools" / "metroy" / "camparam.yaml")
    assert cam.size == (1600, 1200)
    assert cam.f == pytest.approx(1816.744, abs=0.01)
    assert cam.dist[0] == pytest.approx(-0.0610967, abs=1e-6)
    assert cam.f_rel_sigma() == pytest.approx(0.001)       # assumed: the file has no uncertainties
    assert oc.from_metroy(ROOT / "tools" / "metroy" / "camparam.yaml", "R").f == pytest.approx(1810.469, abs=0.01)


@pytest.mark.parametrize("ext", [".jpg", ".png", ".heic"])
def test_photos_are_read_in_the_sensor_layout(tmp_path, ext):
    if ext == ".heic":
        pillow_heif = pytest.importorskip("pillow_heif")
        if not pillow_heif.libheif_info().get("encoders", {}).get("x265"):
            pytest.skip("no HEIF encoder")
    lin = np.zeros((60, 80), np.float32)
    lin[10:20, 5:30] = 0.8                     # a bright bar near the sensor's top left
    path = syn.save_photo(tmp_path / f"p{ext}", lin, orientation=6, quality=100)
    photo = oc.load_photo(path)
    assert photo.size == (80, 60) and photo.orientation == 6
    assert photo.linear[15, 15] > 0.5 and photo.linear[45, 60] < 0.05
    assert abs(float(photo.linear[15, 15]) - 0.8) < 0.08          # linear light again (sRGB undone)


# --------------------------------------------------------------------------- edges, markers, sheet
def test_edge_finder_is_unbiased_to_a_few_hundredths_of_a_pixel():
    """A blurred straight dark/bright edge at known sub-pixel positions and angles, integrated over pixels."""
    rng = np.random.default_rng(1)
    errs = []
    for ang in np.radians([3.0, 17.0, 41.0, 88.0]):
        n = np.array([math.cos(ang), math.sin(ang)])
        c0 = 40.0 + rng.uniform(-0.5, 0.5, 2)
        yy, xx = np.mgrid[0:80, 0:80]
        # pixel-integrated, blurred step (sigma 0.9 px): dark on the minus side of n
        ss = 8
        off = (np.arange(ss) + 0.5) / ss - 0.5
        acc = np.zeros((80, 80))
        for dy in off:
            for dx in off:
                s = (xx + dx - c0[0]) * n[0] + (yy + dy - c0[1]) * n[1]
                acc += s > 0
        import cv2

        img = cv2.GaussianBlur((0.03 + 0.77 * acc / ss ** 2).astype(np.float32), (0, 0), 0.9)
        t = np.array([-n[1], n[0]])
        base = c0 + np.outer(np.linspace(-15, 15, 31), t) + 0.37 * n
        o, ok, _ = oc.edge_offsets(img, base, n, 8.0)
        assert ok.all()
        errs.append(np.median(o + 0.37))
    assert np.max(np.abs(errs)) < 0.02


def test_markers_and_sheet_pose(tmp_path):
    path, R, t = shoot(tmp_path, "sheet.jpg", None, tilt=12.0, seed=1, ss=2)
    cam = camera()
    photo = oc.load_photo(path)
    markers = oc.find_markers(photo, cam)
    assert len(markers) == 16
    truth = {i: cam.project(c @ R.T + t) for i, c in scale_sheet.marker_corners().items()}
    err = np.concatenate([markers[i]["corners"] - truth[i] for i in markers])
    assert np.sqrt(np.mean(np.sum(err ** 2, 1))) < 0.06               # sub-pixel corners from the edges
    sp = oc.sheet_pose(cam, markers, (1.0, 1.0))
    assert np.abs(sp.R - R).max() < 2e-4 and np.abs(sp.t - t).max() < 0.05
    assert abs(sp.dilation_mm) < 0.01 and sp.rms_px < 0.1
    assert sp.tilt_deg == pytest.approx(12.0, abs=0.05)
    # a tilted photo shows its own focal length
    rel, sig = oc.focal_from_photo(cam.scaled_f(1.01), markers, (1.0, 1.0),
                                   oc.sheet_pose(cam.scaled_f(1.01), markers, (1.0, 1.0)))
    assert abs((1.01 * (1 + rel)) - 1) < max(4 * sig, 0.002)


def test_print_scale_from_the_caliper_readings():
    assert oc.sheet_scale(101.0) == (1.01, 1.01)
    sx, sy = oc.sheet_scale(101.0, 120.6)
    assert sx == pytest.approx(1.01) and sy == pytest.approx(1.005)
    assert oc.sheet_scale(76.2, 91.44) == (pytest.approx(0.762), pytest.approx(0.762))   # a measured odd print
    with pytest.raises(ValueError):
        oc.sheet_scale(40.0)
    with pytest.raises(ValueError):
        oc.sheet_scale(100.0, 100.0)                          # bar and height disagree: a reading is wrong
    L = oc.layout([15, 13], 1.0, 1.0)
    assert L[:, 1].max() - L[:, 1].min() == pytest.approx(oc.HEIGHT_NOMINAL)   # marker 15 top to marker 13 bottom


def test_calibration_from_tilted_sheet_photos(tmp_path):
    cam = syn.phone_camera(1200, 900, HFOV, DIST, (5.0, -3.0))
    rng = np.random.default_rng(4)
    paths = []
    for k in range(7):
        R, t = syn.view(rng.uniform(270, 310), rng.uniform(18, 35), 360.0 * k / 7, rng.uniform(-8, 8))
        img = syn.render(cam, R, t, None, ss=2, seed=k)
        paths.append(syn.save_photo(tmp_path / f"cal{k}.jpg", img, meta=syn.camera_meta(cam)))
    got = oc.calibrate(paths, log=QUIET)
    assert got.rms_px < 0.15
    assert abs(got.f / cam.f - 1) < 3 * got.f_rel_sigma() + 5e-4
    assert got.f_rel_sigma() < 0.003
    assert abs(got.K[0, 2] - cam.K[0, 2]) < 3 and abs(got.K[1, 2] - cam.K[1, 2]) < 3
    assert got.key["make"] == "Synthetic"


# --------------------------------------------------------------------------- the golden model
def test_resting_poses_of_a_bolt(bolt_golden):
    """A bolt is a round part: it lies on its head's rim and its shank's end (tilted atan(6 / 81.85)), or stands on
    either end; rolled about its axis it is the same pose; its values get plain names."""
    rests = bolt_golden.rests
    assert len(rests) == 3
    lying = rests[0]
    tilt = math.degrees(math.asin(abs(lying["normal"][2])))
    assert tilt == pytest.approx(math.degrees(math.atan(6.0 / 81.85)), abs=0.01)
    plan = {m["name"]: m["golden"] for m in ock.plan_for(bolt_golden)}
    assert plan == pytest.approx({"Overall length": 107.1, "Head height": 25.25, "Length under head": 81.85,
                                  "Head diameter": 36.0, "Shank diameter": 24.0})
    # settled at any roll it still touches the sheet at its head's rim and its shank's end
    T = bolt_golden.rest_frame(lying, np.array([0, 0, 0, 0, 0, 1.0]))
    Z = (bolt_golden.V @ T[:3, :3].T + T[:3, 3])[:, 2]
    assert Z.min() == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(math.asin(abs((T[:3, :3] @ bolt_golden.profile["a"])[2]))) == pytest.approx(tilt, abs=0.05)


def test_feature_basis_keeps_rim_points_on_both_features(bolt_golden):
    """A point on the rim between the shank's end face and the shank moves with both."""
    g = bolt_golden
    end = g.labels.index("shank end")
    shank = g.labels.index("shank")
    X = np.array([[12.0, 0.0, 107.1]])
    B = g.basis(X, [(end, shank)])
    d = np.zeros(len(g.features))
    d[end], d[shank] = 0.2, 0.05
    assert np.allclose((B @ d)[0], [0.05, 0.0, 0.2])


@pytest.fixture(scope="module")
def screw_golden():
    return ock.GoldenPart(syn.screw(segments=360, rows_per_pitch=16), QUIET)


def test_a_screws_profile_thread_and_knurl(screw_golden):
    """The procedural screw is found round, with a right-hand thread (its pitch and minor diameter from the mesh),
    a knurled head and three shoulders; the values it offers carry plain names."""
    pr = screw_golden.profile
    thread = next(z for z in pr["zones"] if z["texture"] == "thread")
    head = next(z for z in pr["zones"] if z["texture"] == "knurl")
    assert thread["pitch"] == pytest.approx(3.0, abs=2e-3) and thread["hand"] == "right"
    assert 2 * thread["R"] == pytest.approx(20.0, abs=1e-3) and 2 * head["R"] == pytest.approx(30.0, abs=1e-3)
    assert 2 * thread["minor"] == pytest.approx(20.0 - 2 * 0.54 * 3.0, abs=0.02)
    shoulders = sorted(f["label"] for f in screw_golden.features if f["kind"] == "shoulder")
    assert shoulders == ["head top", "head underside", "thread end"]
    names = [m["name"] for m in ock.plan_for(screw_golden)]
    assert names == ["Overall length", "Head height", "Length under head", "Thread major diameter",
                     "Thread minor diameter", "Thread pitch", "Head diameter (knurl crests)"]
    # a hexagon is not round
    hexa = syn.revolve_solid([(0, 0), (10, 0), (10, 5), (0, 5)], segments=6)
    assert ock.GoldenPart(hexa, QUIET).profile is None


def test_turned_errors_move_the_right_vertices():
    mesh = syn.screw(segments=120, rows_per_pitch=8)
    moved, pr = syn.turned_errors(mesh, head_top=0.05, end=-0.10, radial=0.03)
    V0, V1 = np.asarray(mesh.vertices), np.asarray(moved.vertices)
    z0, z1 = V0[:, 2], V1[:, 2]
    assert z1.max() - z0.max() == pytest.approx(0.05) and z1.min() - z0.min() == pytest.approx(0.10)
    mid = (z0 > 10) & (z0 < 40)
    assert np.allclose(np.hypot(V1[mid, 0], V1[mid, 1]) - np.hypot(V0[mid, 0], V0[mid, 1]), 0.03)


def test_model_edge_is_zero_on_straight_edges_and_inside_a_narrow_crest(screw_golden):
    """Where blur puts the model's own edge: nothing on the flat end face, inward on a thread's narrow crests."""
    cam = camera()
    R, t = syn.view(280.0, 0.0, 30.0, 2.0)
    sheet_corners = {i: cam.project(c @ R.T + t) for i, c in ock_marker_corners().items()}
    sheet = oc.sheet_pose(cam, sheet_corners, (1.0, 1.0))
    g = screw_golden
    rest = g.rests[0]
    T = g.rest_frame(rest, np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0]))
    S = ock.outline_samples(g, T, cam, sheet, 3.0, 5.0)
    C = ock.classify(g, S, T, cam, sheet)
    pred = ock.model_edge(g, T, cam, sheet, S["x"], S["n2"], 5.0, 1.0)
    end = C["feat"] == g.labels.index("thread end")
    crest = C["feat"] == g.labels.index("thread crests")
    assert end.sum() > 5 and crest.sum() > 20
    assert abs(np.median(pred[end])) < 0.01
    assert np.median(pred[crest]) < -0.03


def ock_marker_corners():
    from cloudclean import scale_sheet as ss_

    return ss_.marker_corners()


# --------------------------------------------------------------------------- the check
def test_bolt_with_known_errors(bolt_check):
    rep, expected, tmp = bolt_check
    got = {m["name"]: m for m in rep["combined"]}
    assert len(got) == 5 and all(m["status"] != "not_measured" for m in got.values())
    for m in got.values():
        true = syn.true_value(expected, m["kind"], m["golden"])
        assert abs(m["photo"] - true) < 0.015, (m["name"], m["photo"], true)
        assert 0.01 < m["uncertainty"] < 0.15
    # the length under the head is 0.10 short, exactly on the ±0.1 limit: too close to call
    step = next(m for m in got.values() if m["kind"] == "step")
    assert step["status"] == "close"
    assert rep["photos"][0]["part"]["iou"] > 0.9
    assert (tmp / "out" / "report.json").exists() and Path(rep["photos"][0]["overlay"]).exists()
    saved = json.loads((tmp / "out" / "report.json").read_text())
    assert saved["validated"] is False and saved["combined"][0]["name"] == rep["combined"][0]["name"]


@pytest.fixture(scope="module")
def screw_check(tmp_path_factory, screw_golden):
    """The procedural screw 0.05 taller in the head, 0.10 shorter under it, its thread 0.06 bigger, lying at a roll
    of its own, photographed at 7 px/mm."""
    tmp = tmp_path_factory.mktemp("screw")
    gp, tp = syn.part_params("screw", {"head_h": 0.05, "under": -0.10, "major_d": 0.06})
    part = syn.place(syn.screw(**tp, segments=360, rows_per_pitch=16), (1, 0, 0), 3.0, -2.0, 67.0)
    cam = syn.phone_camera(2000, 1500, HFOV, DIST, (6.0, -4.0))
    R, t = syn.view(280.0, 0.0, 30.0, 2.0, (3.0, -2.0, 0.0))
    path = syn.save_photo(tmp / "screw.jpg", syn.render(cam, R, t, part, ss=2, seed=3), meta=syn.camera_meta(cam))
    rep = ock.check_photos(screw_golden, [path], cam, ock.OutlineParams(bar_mm=100.0, mc_samples=6), log=QUIET)
    return rep, syn.expected("screw", gp, tp)


def test_screw_with_known_errors(screw_check, tmp_path):
    rep, expected = screw_check
    got = {m["name"]: m for m in rep["combined"]}
    for name in ("Overall length", "Head height", "Length under head", "Thread major diameter", "Thread pitch",
                 "Head diameter (knurl crests)"):
        m = got[name]
        assert m["status"] != "not_measured", m
        true = syn.true_value(expected, m["kind"], m["golden"])
        assert abs(m["photo"] - true) < (0.004 if m["kind"] == "pitch" else 0.03), (name, m["photo"], true)
        assert abs(m["photo"] - true) <= m["uncertainty"], (name, m["photo"], true, m["uncertainty"])
    assert rep["photos"][0]["part"]["roll_deg"] is not None
    assert "Thread flanks" not in got                      # a nuisance, not reported
    # groove by groove: the thread is 0.06 bigger all over (crests +0.03 on each side), its pitch and depth as drawn
    th = rep["photos"][0]["thread"]
    assert th["measurable"] and th["summary"]["teeth_placed"] >= 8, th.get("reason")
    teeth = [tt for sd in th["sides"] for tt in sd["teeth"] if tt.get("status") == "measured"]
    assert all(abs(tt["pitch_error"]) < 0.03 and tt["pitch_status"] != "off" for tt in teeth if "pitch_error" in tt)
    assert all(abs(tt["lead_error"]) < 0.04 for tt in teeth if "lead_error" in tt)
    crests = [tt["crest_error"] for tt in teeth if "crest_error" in tt]
    assert len(crests) >= 8 and np.median(crests) == pytest.approx(0.03, abs=0.02)
    assert got["Thread pitch error, worst tooth"]["status"] != "off"
    for g in (g for sd in th["sides"] for g in sd["grooves"] if g.get("status") == "measured"):
        assert abs(g["depth_error"]) < 0.05 and g["depth_status"] != "off", g
    assert oth.thread_chart(th, tmp_path / "chart.png").exists()


# --------------------------------------------------------------------------- a thread, groove by groove
@pytest.fixture(scope="module")
def thread_scene(screw_golden):
    """The procedural screw lying across the test camera's view, straight down: golden -> sheet, camera, sheet."""
    mesh = syn.screw(segments=360, rows_per_pitch=16)
    placed = syn.place(mesh, (1, 0, 0), 3.0, -2.0, 0.0)
    Rg, tg = syn.rigid_between(np.asarray(mesh.vertices), np.asarray(placed.vertices))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rg, tg
    cam = syn.phone_camera(2000, 1500, HFOV, DIST, (6.0, -4.0))
    R, t = syn.view(280.0, 0.0, 30.0, 2.0, (3.0, -2.0, 0.0))
    sheet = oc.SheetPose(R, t, list(range(16)), (1.0, 1.0), 0.0, 0.0, 0.0, 0.0, np.zeros((6, 6)), cam.f / 280.0)
    return screw_golden, mesh, T, cam, sheet


def _exact_thread(scene, mesh):
    g, _, T, cam, sheet = scene
    part = syn._mesh(np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3], np.asarray(mesh.triangles))
    return oth.inspect_thread(g, None, None, None, cam, sheet, 0.0, 0.0, ock.OutlineParams(), 1e-4,
                              edges=syn.silhouette_edges(part, cam, sheet), T=T)


def test_thread_groove_by_groove_on_the_exact_outline(thread_scene):
    """On the exact silhouette (no blur, no noise: the method alone) the perfect screw shows no error, and one
    groove filled 0.10 and one tooth moved 0.05 along the axis show where they are, on both sides."""
    g, mesh, T, cam, sheet = thread_scene
    perfect = _exact_thread(thread_scene, mesh)
    assert perfect["measurable"] and perfect["view_angle_deg"] > 80
    s = perfect["summary"]
    assert s["teeth_placed"] >= 16 and s["grooves_measured"] >= 12 and s["teeth_excluded"] >= 2
    for sd in perfect["sides"]:
        for tt in sd["teeth"]:
            for k in ("pitch_error", "lead_error", "crest_error"):
                assert abs(tt.get(k, 0.0)) < 0.006, (k, tt)
        for gr in sd["grooves"]:
            assert abs(gr.get("depth_error", 0.0)) < 0.006, gr
    # the errors: their helix turn's seam faces the camera, so both sides of the outline see them
    C = -sheet.R.T @ sheet.t
    rel = T[:3, :3].T @ (C - T[:3, 3]) - g.profile["o"]
    seam = math.degrees(math.atan2(rel @ g.profile["v"], rel @ g.profile["u"]))
    bad, _ = syn.thread_errors(mesh, g.profile, shallow=(2, 0.10), shift=(-3, 0.05), seam_deg=seam)
    th = _exact_thread(thread_scene, bad)
    for sd in th["sides"]:
        depths = sorted(gr["depth_error"] for gr in sd["grooves"] if "depth_error" in gr)
        assert depths[0] == pytest.approx(-0.10, abs=0.012) and abs(depths[1]) < 0.01, depths
        pitches = [tt["pitch_error"] for tt in sd["teeth"] if "pitch_error" in tt]
        k = int(np.argmax(pitches))
        assert pitches[k] == pytest.approx(0.05, abs=0.008) and pitches[k + 1] == pytest.approx(-0.05, abs=0.008)
        # (the filled groove also trims the foot of its neighbours' flanks a little)
        assert max(abs(v) for i, v in enumerate(pitches) if i not in (k, k + 1)) < 0.01
    rows = {r["name"]: r for r in oth.summary_rows(th, 0.1)}
    assert rows["Groove depth error, worst groove"]["difference"] == pytest.approx(-0.10, abs=0.012)
    assert rows["Thread pitch error, worst tooth"]["status"] == "ok"           # 0.05 of a 0.1 tolerance


def test_thread_pointing_at_the_camera_is_not_measured_and_says_why(thread_scene):
    """Seen 35 degrees down along its axis (as the MetroY saw the real screw), the outline is the crests' envelope:
    the check measures nothing groove by groove and says how to lay the screw."""
    g, mesh, T, cam, _ = thread_scene
    a = T[:3, :3] @ g.profile["a"]
    R, t = syn.view(280.0, 35.0, math.degrees(math.atan2(a[1], a[0])), 0.0, (3.0, -2.0, 0.0))
    sheet = oc.SheetPose(R, t, list(range(16)), (1.0, 1.0), 0.0, 0.0, 0.0, 0.0, np.zeros((6, 6)), cam.f / 280.0)
    th = _exact_thread((g, mesh, T, cam, sheet), mesh)
    assert not th["measurable"] and th["view_angle_deg"] < 70 and th["visible_depth"] < 0.4
    assert "Lay the screw across the view" in th["reason"]
    rows = oth.summary_rows(th, 0.1)
    assert len(rows) == 5 and all(r["status"] == "not_measured" for r in rows)


def test_combine_keeps_the_worst_photo_for_worst_rows():
    def rep(v, status):
        return {"measurements": [{"id": 0, "kind": "pitch_error", "name": "Thread pitch error, worst tooth",
                                  "golden": 0.0, "photo": v, "difference": v, "uncertainty": 0.02, "status": status,
                                  "combine": "worst", "where": f"tooth {v}", "budget": {
                                      "statistical": 0.01, "placement": 0.0, "camera_and_scale": 0.0,
                                      "toner_spread_assumed": 0.0, "edge_assumed": 0.0}}]}
    out = ock.combine([rep(0.01, "ok"), rep(-0.09, "close"), rep(0.03, "ok")], 0.1)[0]
    assert out["difference"] == -0.09 and out["status"] == "close" and out["photos"] == 3


def test_plate_as_drawn_matches_and_says_what_it_cannot_see(tmp_path):
    golden = ock.GoldenPart(syn.plate(), QUIET)
    part = syn.place(syn.plate(segments=512), (0, 0, -1), 2.0, 4.0, -32.0)
    path, _, _ = shoot(tmp_path, "plate.jpg", part, seed=5)
    rep = ock.check_photos(golden, [path], camera(), ock.OutlineParams(bar_mm=100.0, mc_samples=8), log=QUIET)
    by = {m["name"]: m for m in rep["combined"]}
    z = by["Overall size along Z"]
    assert z["status"] == "not_measured" and "not on the part's outline" in z["reason"]
    measured = [m for m in rep["combined"] if m["status"] != "not_measured"]
    assert len(measured) == 7
    for m in measured:
        assert abs(m["difference"]) < 0.015, m
        assert m["status"] != "off", m
    # at this small test size (5.7 px/mm) the assumed edge term alone is ±0.05 mm on a width, so widths may be too
    # close to call; steps (both edges facing the same way) do not carry it
    assert all(m["status"] == "ok" for m in measured if m["kind"] == "step")
    assert rep["verdict"] in ("match", "close")
    # without golden.py's faces, the outline's own straight runs and arcs measure the same widths, steps and hole
    rep2 = ock.check_photos(golden, [path], camera(), ock.OutlineParams(bar_mm=100.0, mc_samples=2,
                                                                         use_faces=False), log=QUIET)
    got = sorted(round(m["golden"], 1) for m in rep2["combined"] if m["status"] != "not_measured")
    assert got == [12.0, 14.0, 25.0, 28.0, 40.0, 45.0, 70.0]
    for m in rep2["combined"]:
        assert abs(m["difference"]) < 0.015, m


def test_combine_weights_photos_and_inflates_when_they_disagree():
    def rep(v, stat):
        return {"measurements": [{"id": 0, "kind": "size", "name": "L", "golden": 10.0, "faces": [0, 1],
                                  "photo": v, "status": "ok", "budget": {
                                      "statistical": stat, "placement": 0.0, "camera_and_scale": 0.01,
                                      "toner_spread_assumed": 0.0, "edge_assumed": 0.0}}]}
    agree = ock.combine([rep(10.012, 0.005), rep(10.018, 0.005)], 0.1)[0]
    assert agree["photo"] == pytest.approx(10.015) and agree["photos"] == 2 and agree["consistency"] == 1.0
    # the photos' own parts average down, the common part (camera, caliper) does not
    assert agree["uncertainty"] == pytest.approx(2 * math.hypot(0.005 / math.sqrt(2), 0.01), abs=1e-4)
    disagree = ock.combine([rep(10.00, 0.005), rep(10.08, 0.005)], 0.1)[0]
    assert disagree["uncertainty"] > 2 * agree["uncertainty"] and disagree["consistency"] > 5


def test_cli_render_and_check(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("outline_cli", ROOT / "tools" / "outline_check" / "outline_check.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    out = tmp_path / "set"
    assert cli.main(["render", "plate", str(out), "--calibration", "0", "--photos", "1", "--width", "1200",
                     "--height", "900", "--hfov", "55", "--ss", "2", "--errors", "hole_d=0.05"]) == 0
    truth = json.loads((out / "truth.json").read_text())
    oc.Camera.from_json(truth["camera"]).save(tmp_path / "camera.json")
    assert cli.main(["check", str(out / "golden.ply"), str(out / "photos"), "--camera", str(tmp_path / "camera.json"),
                     "--bar-mm", "100.0", "--mc-samples", "4", "--out", str(tmp_path / "result")]) == 0
    res = json.loads((tmp_path / "result" / "report.json").read_text())
    hole = next(m for m in res["combined"] if m["kind"] == "diameter")
    assert hole["difference"] == pytest.approx(0.05, abs=0.03)
