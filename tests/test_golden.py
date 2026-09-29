"""Golden model check (cloudclean/golden.py, web/jobs_golden.py, web/routes_golden.py, docs/golden-model.md).

A stepped tube (Ø30 x 8 at the bottom, Ø20 up to 20, a Ø10 hole) is the golden model. Its scan has known faults:
the bottom face not scanned, the hole 0.16 mm too big, a raised patch on top and a rough patch on the Ø30 wall -
and it is turned and moved, so the check has to line it up first."""
from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cloudclean.compare import ReferenceSurface
from cloudclean.golden import GoldenParams, check_against_golden, find_faces
from cloudclean.io import save
from cloudclean.web.server import create_app
from tests.golden_synthetic import stepped_tube, stepped_tube_scan
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
    assert len(missing) == 1 and missing[0]["name"] == "Bottom face" and missing[0]["rescan"]
    assert missing[0]["area_mm2"] == pytest.approx(np.pi * (15 ** 2 - 5 ** 2), rel=0.08)
    assert "Turn the part over" in missing[0]["advice"]
    assert faulty["surface"]["scanned_pct"] == pytest.approx(100 * (1 - 628.3 / 3393.0), abs=2.5)
    assert faulty["rescan"] and faulty["rescan"][0].startswith("Bottom face: ")


def test_it_finds_rough_and_off_areas(faulty):
    rough = [r for r in faulty["regions"] if r["kind"] == "rough"]
    assert len(rough) == 1 and "Ø30.00 round face" in rough[0]["name"]
    assert rough[0]["spread"] == pytest.approx(0.15, abs=0.04)
    off = {r["name"]: r for r in faulty["regions"] if r["kind"] == "off"}
    hole = off["Ø10.00 hole"]
    assert hole["deviation"] == pytest.approx(-0.08, abs=0.01) and hole["sign"] == -1
    assert "the hole is bigger here" in hole["why"]
    bump = next(r for name, r in off.items() if name.startswith("Top face at Z 20.00"))
    assert bump["deviation"] > 0.15 and bump["sign"] == 1 and 15 < bump["area_mm2"] < 40
    assert not any(r["kind"] == "thin" for r in faulty["regions"])       # the rims of the missing face are folded in


def test_the_measurements(faulty):
    m = {x["name"]: x for x in faulty["measurements"]}
    hole = m["Diameter: Ø10.00 hole"]
    assert hole["status"] == "off" and hole["scan"] == pytest.approx(10.16, abs=0.01)
    for name, golden in (("Diameter: Ø30.00 round face", 30), ("Diameter: Ø20.00 round face", 20),
                         ("Overall size along X", 30), ("Overall size along Y", 30)):
        assert m[name]["status"] == "ok" and m[name]["scan"] == pytest.approx(golden, abs=0.01), name
    step = by_name(faulty["measurements"], "Step: ")[0]
    assert step["golden"] == pytest.approx(12) and step["status"] == "ok"
    z = m["Overall size along Z"]                                          # needs the bottom, which is missing
    assert z["status"] == "not_measured" and "bottom" in z["reason"]
    assert faulty["regions"][z["region"]]["name"] == "Bottom face"
    for p in by_name(faulty["measurements"], "Position: "):
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
        assert page.status_code == 200 and "Golden model check" in page.text and "Bottom face" in page.text
        assert client.get(f"/api/assets/{compare_id}/golden-report").status_code == 400
