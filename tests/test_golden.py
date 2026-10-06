"""Golden model check (cloudclean/golden.py, web/jobs_golden.py, web/routes_golden.py, docs/golden-model.md).

A stepped tube (Ø30 x 8 at the bottom, Ø20 up to 20, a Ø10 hole) is the golden model. Its scan has known faults:
the bottom face not scanned, the hole 0.16 mm too big, a raised patch on top and a rough patch on the Ø30 wall -
and it is turned and moved, so the check has to line it up first. A screw (round head, shaft, hex socket) and a block
with a pocket check the words (golden_words): names after the part, 'where' and 'what' sentences, series, dimension
lines on the golden surface, the section drawing and the per-vertex golden face."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cloudclean.compare import ReferenceSurface
from cloudclean.golden import GoldenParams, _summarise, check_against_golden, find_faces, refine_long_edges
from cloudclean.golden_words import PartWords
from cloudclean.io import save
from cloudclean.web.server import create_app
from tests.golden_synthetic import (SCREW, counterbored_tube, pocket_block, pocket_block_scan, screw, screw_scan,
                                    stepped_tube, stepped_tube_scan)
from tests.synthetic import random_rigid

TOL = 0.05


@pytest.fixture(scope="module")
def faulty():
    scan = stepped_tube_scan(n=250_000, pose=random_rigid(4))
    return check_against_golden(scan, stepped_tube(), GoldenParams(tolerance=TOL), log=lambda m: None)["report"]


def by_name(items, text):
    found = [i for i in items if text in i["name"]]
    assert found, (text, [i["name"] for i in items])
    return found


def test_the_golden_models_faces():
    faces, tri_face, _ = find_faces(ReferenceSurface(stepped_tube(), log=lambda m: None), min_area=1.0)
    planes = sorted(round(f["point"][2], 3) for f in faces if f["type"] == "plane")
    cyl = sorted((round(2 * f["radius"], 3), f["hole"]) for f in faces if f["type"] == "cylinder")
    assert planes == [0.0, 8.0, 20.0]                                       # bottom, shoulder, top
    assert cyl == [(10.0, True), (20.0, False), (30.0, False)]              # the hole and the two round faces
    assert (tri_face >= 0).all()                                            # every triangle belongs to a face


def test_it_finds_what_was_not_scanned(faulty):
    missing = [r for r in faulty["regions"] if r["kind"] == "missing"]
    assert len(missing) == 1 and missing[0]["name"] == "Flat face at the wide end" and missing[0]["rescan"]
    assert missing[0]["where"] == "The flat face at the wide end of the part."
    assert missing[0]["area_mm2"] == pytest.approx(np.pi * (15 ** 2 - 5 ** 2), rel=0.08)
    assert "Turn the part over" in missing[0]["advice"] and "stood on it" in missing[0]["advice"]
    assert faulty["surface"]["scanned_pct"] == pytest.approx(100 * (1 - 628.3 / 3393.0), abs=2.5)
    assert faulty["rescan"] and faulty["rescan"][0].startswith("Flat face at the wide end: ")


def test_it_finds_rough_and_off_areas(faulty):
    rough = [r for r in faulty["regions"] if r["kind"] == "rough"]
    assert len(rough) == 1 and rough[0]["name"] == "Part of the round side at the wide end"
    assert rough[0]["spread"] == pytest.approx(0.15, abs=0.04)
    off = {r["name"]: r for r in faulty["regions"] if r["kind"] == "off"}
    hole = off["Ø10.00 hole through the middle"]
    assert hole["deviation"] == pytest.approx(-0.08, abs=0.01) and hole["sign"] == -1
    assert "the hole is bigger here" in hole["why"]
    bump = next(r for name, r in off.items() if name.endswith("flat face at the narrow end"))
    assert bump["deviation"] > 0.15 and bump["sign"] == 1 and 15 < bump["area_mm2"] < 40
    assert not any(r["kind"] == "thin" for r in faulty["regions"])       # the rims of the missing face are folded in


def test_the_measurements(faulty):
    m = {x["name"]: x for x in faulty["measurements"]}
    hole = m["Ø10.00 hole through the middle: diameter"]
    assert hole["status"] == "off" and hole["scan"] == pytest.approx(10.16, abs=0.01) and hole["group"] == "Holes"
    for name, golden in (("Diameter at the wide end", 30), ("Diameter at the narrow end", 20),
                         ("Overall width (left to right)", 30), ("Overall height (bottom to top)", 30)):
        assert m[name]["status"] == "ok" and m[name]["scan"] == pytest.approx(golden, abs=0.01), name
    step = m["Length of the narrow part"]
    assert step["golden"] == pytest.approx(12) and step["status"] == "ok" and step["kind"] == "step"
    assert step["what"] == "From the step to the flat face at the narrow end: the length of the narrow part."
    assert m["Length of the wide part"]["golden"] == pytest.approx(8)
    z = m["Overall length"]                                               # needs the wide end, which is missing
    assert z["status"] == "not_measured" and "wide end" in z["reason"]
    assert faulty["regions"][z["region"]]["name"] == "Flat face at the wide end"
    for p in by_name(faulty["measurements"], ": where it sits"):
        assert p["status"] == "ok" and p["scan"] < 0.01
    assert all(x["status"] in ("ok", "off", "close", "not_measured") for x in faulty["measurements"])


def test_the_verdict(faulty):
    assert faulty["verdict"] == "differs"
    assert faulty["headline"].startswith("The scan does not match the golden model: 1 measurement is off")
    assert faulty["counts"]["off"] == 1 and faulty["counts"]["not_measured"] >= 1
    assert faulty["alignment"]["symmetric"] is True                        # round: turning it about Z changes nothing
    assert not any("symmetric" in w for w in faulty["warnings"])
    assert faulty["faces"] == {"flat": 3, "holes": 1, "round": 2}


def test_a_good_scan_matches():
    scan = stepped_tube_scan(n=250_000, hole_grow=0.0, bump=0.0, rough=0.0, missing_bottom=False,
                             pose=random_rigid(9))
    report = check_against_golden(scan, stepped_tube(), GoldenParams(tolerance=TOL), log=lambda m: None)["report"]
    assert report["verdict"] == "match", report["headline"]
    assert report["regions"] == [] and report["rescan"] == []
    assert all(m["status"] == "ok" for m in report["measurements"]), [
        (m["name"], m["status"], m.get("difference")) for m in report["measurements"] if m["status"] != "ok"]
    assert report["surface"]["scanned_pct"] > 99


def _wait(client, job, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        current = {j["id"]: j for j in client.get("/api/jobs").json()}[job["id"]]
        if current["status"] in ("done", "failed", "cancelled"):
            return current
        time.sleep(0.5)
    raise TimeoutError(job["title"])


def test_the_golden_check_api(tmp_path):
    save(stepped_tube(), tmp_path / "golden.ply")
    save(stepped_tube_scan(n=120_000, pose=random_rigid(4)), tmp_path / "scan.ply")
    with TestClient(create_app(tmp_path / "ws")) as client:
        started = client.post("/api/import-paths", json={"paths": [str(tmp_path / "golden.ply"),
                                                                   str(tmp_path / "scan.ply")]}).json()["jobs"]
        golden_id, scan_id = (_wait(client, j)["result"][0] for j in started)
        project = client.get(f"/api/assets/{scan_id}").json()["project"]

        r = client.post("/api/golden-check", json={"scan_id": scan_id})
        assert r.status_code == 400 and "no golden model yet" in r.json()["detail"]
        r = client.patch(f"/api/projects/{project}", json={"golden_asset_id": scan_id})
        assert r.status_code == 400 and "not a mesh" in r.json()["detail"]
        assert client.post("/api/golden-check", json={"scan_id": scan_id, "golden_id": scan_id}).status_code == 400

        job = client.post("/api/golden-check", json={"scan_id": scan_id, "golden_id": golden_id, "remember": True,
                                                    "tolerance": TOL}).json()
        assert client.get(f"/api/projects/{project}").json()["golden_asset_id"] == golden_id
        done = _wait(client, job)
        assert done["status"] == "done", done.get("error")
        check_id, compare_id = done["result"]
        out = done["output"]
        assert out["verdict"] == "differs" and out["measurements_off"] == 1 and out["rescan"] >= 1
        check = client.get(f"/api/assets/{check_id}").json()
        assert check["operation"] == "golden_check" and check["kind"] == "mesh"
        assert client.get(f"/api/assets/{compare_id}").json()["operation"] == "compare"

        again = _wait(client, client.post("/api/golden-check", json={"scan_id": scan_id}).json())
        assert again["status"] == "done"                                   # the project's golden model is used

        page = client.get(f"/api/assets/{check_id}/golden-report")
        assert page.status_code == 200 and "Golden model check" in page.text and "Flat face at the wide end" in page.text
        assert "Across the Ø10.00 hole through the middle." in page.text           # what each size is
        assert "check_face" in {x["name"] for x in check["scalars"]}
        assert client.get(f"/api/assets/{compare_id}/golden-report").status_code == 400


# ---------------------------------------------------------------- names, "Show me" and the coloured surface (v2, v3)
def test_areas_are_named_after_the_part(faulty):
    assert faulty["version"] == 3 and faulty["part"]["ends"] == ["wide end", "narrow end"]
    assert faulty["part"]["kind"] == "axial"                                 # a bore runs through: not a head and shaft
    assert faulty["part"]["summary"] == "A round part, wider at one end; a Ø10.00 hole through the middle."
    assert sorted(r["number"] for r in faulty["regions"]) == list(range(1, len(faulty["regions"]) + 1))
    for r in faulty["regions"]:
        assert not any(c in r["name"] for c in ("X ", "Y ", "Z ", "(+", "(-")), r["name"]   # no coordinates
        v = r["view"]
        assert v["radius"] > 0 and np.linalg.norm(np.subtract(v["from"], v["target"])) > 2 * v["radius"]
        assert len(r["pin"]) == 3 and len(r["box"]) == 2
    for m in faulty["measurements"]:
        assert " at X " not in m["name"] and " at Z " not in m["name"], m["name"]
        assert m["what"].endswith(".") and m["group"], m["name"]
        assert ("more" in m) == (m["kind"] != "position"), m["name"]


def test_a_recess_is_named_and_measured():
    """A tube with a counterbore: the counterbore's floor is a recess floor, its depth a named measurement."""
    ref = ReferenceSurface(counterbored_tube(), log=lambda m: None)
    faces, tri_face, _ = find_faces(ref, min_area=1.0)
    words = PartWords(ref, faces, up_axis="z")
    named = dict(zip(words.labels, faces))
    assert words.ends == ("bottom end", "top end")                  # both ends equally wide: named from the view
    floor = named["bottom of the Ø16.00 hole in the top end"]
    assert floor["point"][2] == pytest.approx(14.0)
    assert {"flat face at the bottom end", "flat face at the top end", "round outside",
            "Ø6.00 hole through the middle"} <= set(named)       # the Ø6 hole opens into the Ø16 one: through
    end = next(i for i, f in enumerate(faces) if words.labels[i] == "flat face at the top end")
    fl = next(i for i, f in enumerate(faces) if words.labels[i] == "bottom of the Ø16.00 hole in the top end")
    depth = words.pair_words("step", end, fl)
    assert depth["name"] == "Depth of the Ø16.00 hole in the top end" and depth["more"] == "deeper"
    assert depth["what"] == "From the flat face at the top end down to the bottom of the Ø16.00 hole in the top end."
    assert words.kind == "axial" and words.summary() == ("A round part; a Ø16.00 hole in the top end and a Ø6.00 hole "
                                                         "through the middle.")
    # looking into the recess: the camera goes up the axis, not through the tube's wall
    v = words.view_for(floor["point"] + [5.5, 0, 0], [0, 0, 1], 4.0)
    look = np.subtract(v["from"], v["target"])
    assert look[2] / np.linalg.norm(look) > 0.9


def test_refining_keeps_the_surface_and_splits_long_edges():
    import open3d as o3d
    box = o3d.geometry.TriangleMesh.create_box(20, 10, 5)            # 12 huge triangles
    fine = refine_long_edges(box, max_edge=1.0, max_triangles=200_000)
    V, F = np.asarray(fine.vertices), np.asarray(fine.triangles)
    assert np.linalg.norm(V[F] - V[F[:, [1, 2, 0]]], axis=2).max() <= 1.0 + 1e-9
    assert fine.get_surface_area() == pytest.approx(box.get_surface_area())
    welded = o3d.geometry.TriangleMesh(fine).remove_duplicated_vertices()
    assert welded.is_watertight()                                    # no cracks: shared edges split the same way
    capped = refine_long_edges(box, max_edge=0.1, max_triangles=500)
    assert len(capped.triangles) <= 500


def test_a_scan_that_mostly_matches_says_so():
    surface = {"scanned_pct": 99.0, "shares_pct": {"good": 97.3}, "noise": 0.01, "edges": {"rounded_by": None}}
    creport = {"alignment": {}, "warnings": [], "stats": {k: 0.0 for k in ("mean", "std", "rms", "p05", "p95",
                                                                          "within_tolerance_pct")}}
    off = [{"status": "off"}]
    report = _summarise([], off, surface, creport, 1.0, [])
    assert report["verdict"] == "differs" and report["match_pct"] == 97.3
    assert report["headline"].startswith("The scan mostly matches the golden model (97 % of the surface")
    surface["shares_pct"]["good"] = 80.0
    assert _summarise([], off, surface, creport, 1.0, [])["headline"].startswith("The scan does not match")


# ---------------------------------------------------------------- a screw: head, shaft, hex socket (v3)
@pytest.fixture(scope="module")
def screw_check():
    return check_against_golden(screw_scan(pose=random_rigid(5)), screw(), GoldenParams(tolerance=TOL),
                                log=lambda m: None)


def test_a_screw_is_named_after_its_head_and_shaft(screw_check):
    rep = screw_check["report"]
    assert rep["part"]["kind"] == "head_shaft" and rep["part"]["ends"] == ["tip of the shaft", "head"]
    assert rep["part"]["summary"] == "A round part with a head and a shaft; a hex socket in the head."
    m = {x["name"]: x for x in rep["measurements"]}
    s = SCREW
    for name, golden, group, more in (
            ("Overall length", s["top_z"], "Overall size", "longer"),
            ("Head height", s["top_z"] - s["head_z"], "Head", "taller"),
            ("Length under the head", s["head_z"], "Shaft", "longer"),
            ("Hex socket depth", s["hex_depth"], "Hex socket", "deeper"),
            ("Socket bottom to underside of head", s["top_z"] - s["head_z"] - s["hex_depth"], "Hex socket", "longer"),
            ("Socket bottom to tip", s["top_z"] - s["hex_depth"], "Hex socket", "longer"),
            ("Shaft diameter", 2 * s["shaft_r"], "Shaft", "bigger"),
            ("Head diameter", 2 * s["head_r"], "Head", "bigger"),
            ("Head width (left to right)", 2 * s["head_r"], "Head", "wider")):
        assert m[name]["golden"] == pytest.approx(golden) and m[name]["group"] == group, name
        assert m[name]["more"] == more and m[name]["what"].endswith("."), name
    assert m["Hex socket depth"]["what"] == "From the top of the head down to the bottom of the hex socket."
    assert m["Length under the head"]["what"] == "From the underside of the head to the tip of the shaft."
    assert m["Shaft: where it sits"]["group"] == "Shaft" and "more" not in m["Shaft: where it sits"]
    # the three widths across the hex socket: one series, shown as one row
    across = [x for x in rep["measurements"] if x.get("series")]
    assert {x["name"] for x in across} == {"Hex socket, across flats"}
    assert sorted(x["series_index"] for x in across) == [1, 2, 3] and {x["series_size"] for x in across} == {3}
    assert all(x["golden"] == pytest.approx(s["hex_af"]) and x["status"] == "ok" for x in across)
    assert "the size of hex key that fits" in across[0]["what"]
    for x in rep["measurements"] + rep["regions"]:
        assert not any(w in x["name"] for w in ("wide end", "narrow end", " mm from", "(1 of", "X ", "Z ")), x["name"]


def test_a_screws_areas_and_sizes_say_why(screw_check):
    rep = screw_check["report"]
    regions = sorted(rep["regions"], key=lambda r: r["number"])
    assert [r["number"] for r in regions] == list(range(1, len(regions) + 1))
    first = regions[0]                                         # areas that differ come first
    assert first["kind"] == "off" and first["name"] == "Bottom of the hex socket" and first["deviation"] < -0.3
    assert first["where"].startswith("Inside the head: the flat bottom of the hex socket")
    assert "deeper" in first["why"] and "cone" in first["advice"]
    corners = [r for r in regions if r["kind"] == "missing"]
    assert corners and {r["name"] for r in corners} == {"Corner under the head"}
    assert all("up under the head" in r["advice"] for r in corners)
    off = [x for x in rep["measurements"] if x["status"] == "off"]
    assert {x["name"] for x in off} == {"Hex socket depth", "Socket bottom to underside of head",
                                        "Socket bottom to tip"}
    for x in off:
        assert x["caveat"].startswith("The scan finds the bottom of the hex socket 0.4")
        assert x["caveat"].endswith(f"mm deeper than the golden model (area {first['number']}).")
        assert first["id"] in x["regions"]
    assert rep["sizes_story"] == (
        "All 3 sizes that are off are measured from the bottom of the hex socket. The scan finds that bottom about "
        f"0.4 mm deeper than the golden model (area {first['number']}). Drilled sockets usually end in a cone that "
        "the drawing shows flat, so check the depth with a depth gauge before deciding the part is wrong.")
    height = next(x for x in rep["measurements"] if x["name"] == "Head height")
    assert height["caveat"].startswith("Part of the underside of the head was not scanned (area ")


def test_a_screws_dimension_lines_lie_on_its_surfaces(screw_check):
    ref = ReferenceSurface(screw(), log=lambda m: None)
    for m in screw_check["report"]["measurements"]:
        if m["kind"] == "position":
            assert m["ends"] is None
            continue
        ends = np.asarray(m["ends"], dtype=np.float64)
        assert ends.shape == (2, 3), m["name"]
        assert np.abs(ref.query(ends)[0]).max() < 0.05, m["name"]
        # across the hex socket and round faces: the full length; else along the axis (faces that do not overlap,
        # like the socket's bottom and the underside, get a slanted line between their nearest points)
        span = np.linalg.norm(ends[1] - ends[0]) if m["kind"] in ("gap", "diameter") or m.get("axis") in (0, 1) \
            else abs(ends[1][2] - ends[0][2])
        assert span == pytest.approx(m["golden"], abs=0.05), m["name"]


def test_a_screws_section_and_golden_faces(screw_check):
    rep, s = screw_check["report"], SCREW
    sec = rep["section"]
    assert sec["u"] == [0.0, 0.0, 1.0] and sec["v"] == [0.0, 1.0, 0.0]      # along the axis; the view's up
    assert len(json.dumps(sec)) < 60_000
    x0, y0, x1, y1 = sec["bounds"]
    assert (x0, x1) == pytest.approx((-s["top_z"] / 2, s["top_z"] / 2), abs=0.02)
    assert (y0, y1) == pytest.approx((-s["head_r"], s["head_r"]), abs=0.02)
    area = 0.0
    for loop in sec["loops"]:
        P = np.asarray(loop).reshape(-1, 2)
        assert len(P) >= 3 and (P >= [x0, y0]).all() and (P <= [x1, y1]).all()
        area += 0.5 * (P[:, 0] @ np.roll(P[:, 1], -1) - P[:, 1] @ np.roll(P[:, 0], -1))
    socket = s["hex_depth"] * s["hex_af"] / np.cos(np.radians(30.0))     # cut through two corners of the hex
    full = 2 * s["shaft_r"] * s["head_z"] + 2 * s["head_r"] * (s["top_z"] - s["head_z"]) - socket
    assert abs(area) == pytest.approx(full, rel=0.01)
    floor = next(r for r in rep["regions"] if r["name"] == "Bottom of the hex socket")["face"]
    depth = next(m for m in rep["measurements"] if m["name"] == "Hex socket depth")
    assert floor in depth["faces"] and str(floor) in sec["faces"]
    (xa, ya, xb, yb), = sec["faces"][str(floor)]                          # the socket's bottom: one straight trace
    assert xa == pytest.approx(xb, abs=0.01) and abs(ya - yb) == pytest.approx(s["hex_af"] / np.cos(np.radians(30)),
                                                                              abs=0.05)
    # the coloured golden surface: each vertex's golden face
    mesh, vf = screw_check["check_mesh"], screw_check["vertex_face"]
    V = np.asarray(mesh.vertices)
    assert len(vf) == len(V)
    on_floor = (np.abs(V[:, 2] - (s["top_z"] - s["hex_depth"])) < 1e-6) & (np.hypot(V[:, 0], V[:, 1]) < 3.0)
    assert on_floor.any() and (vf[on_floor] == floor).all()


# ---------------------------------------------------------------- a block: no main axis, a pocket (v3)
def test_a_block_is_named_from_the_view():
    res = check_against_golden(pocket_block_scan(pose=random_rigid(6)), pocket_block(), GoldenParams(tolerance=TOL),
                               log=lambda m: None, up_axis="z")
    rep = res["report"]
    assert rep["verdict"] == "match" and rep["part"]["kind"] == "block"
    m = {x["name"]: x for x in rep["measurements"]}
    assert {"Overall width (left to right)", "Overall depth (front to back)", "Overall height (bottom to top)"} <= set(m)
    depth = m["Depth of the pocket in the top"]
    assert depth["golden"] == pytest.approx(6) and depth["more"] == "deeper" and depth["group"] == "Pocket in the top"
    assert depth["what"] == "From the top face down to the bottom of the pocket."
    assert m["Thickness under the pocket in the top"]["golden"] == pytest.approx(14)
    assert m["Front wall thickness"]["golden"] == pytest.approx(10)
    assert m["Front face to pocket's back wall"]["golden"] == pytest.approx(20)
    sec = rep["section"]                                                   # no axis: right x up, square to the view
    assert sec["u"] == [1.0, 0.0, 0.0] and sec["v"] == [0.0, 0.0, 1.0]
    P = np.asarray(sec["loops"][0]).reshape(-1, 2)
    area = 0.5 * abs(P[:, 0] @ np.roll(P[:, 1], -1) - P[:, 1] @ np.roll(P[:, 0], -1))
    assert len(sec["loops"]) == 1 and area == pytest.approx(36 * 20 - 16 * 6, rel=0.01)
