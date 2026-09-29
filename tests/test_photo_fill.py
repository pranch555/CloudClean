"""Filling the parts a scan missed with points from photos (cloudclean/photo_fill.py).

The scan of the stepped tube misses its bottom face. The "photo model" shows the whole part, but like a real one it
is noisy (±0.4 mm), 2 % too big, turned and moved, and has a table below the part and a clump of clutter beside it."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cloudclean.io import estimate_spacing
from cloudclean.photo_fill import fill_gaps, refine_onto_scan
from tests.golden_synthetic import sample_surface, stepped_tube, stepped_tube_scan


def photo_model(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    part = sample_surface(stepped_tube(), 60_000, seed=seed)
    part += rng.normal(0, 0.4, part.shape)
    table = np.c_[rng.uniform(-40, 40, (8000, 2)), np.full(8000, -3.0)]          # the part stands on a pedestal
    clutter = rng.normal([40, 0, 10], 1.0, (400, 3))
    return np.vstack([part, table, clutter])


def test_the_photo_points_are_fitted_onto_the_scan():
    scan = np.asarray(stepped_tube_scan(n=150_000, hole_grow=0, bump=0, rough=0).points)
    truth = photo_model()[:60_000]
    R = Rotation.from_euler("xyz", [2, -3, 1], degrees=True).as_matrix()
    moved = 1.02 * truth @ R.T + [1.5, -2.0, 0.8]                                  # the photos' own frame and size
    T, info = refine_onto_scan(moved, scan, estimate_spacing(scan), log=lambda m: None)
    back = moved @ T[:3, :3].T + T[:3, 3]
    assert info["scale"] == pytest.approx(1 / 1.02, abs=0.004)
    assert np.median(np.linalg.norm(back - truth, axis=1)) < 0.3


def test_it_fills_what_the_scan_missed_and_nothing_else():
    scan = np.asarray(stepped_tube_scan(n=150_000, hole_grow=0, bump=0, rough=0).points)   # no bottom face
    out = fill_gaps(scan, photo_model(), estimate_spacing(scan), log=lambda m: None)
    added = out["points"][out["source"] == 1]
    rep = out["report"]
    assert len(out["points"]) == len(scan) + len(added) and (out["source"][:len(scan)] == 0).all()
    r = np.hypot(added[:, 0], added[:, 1])
    on_bottom = (np.abs(added[:, 2]) < 1.5) & (r > 4) & (r < 16)
    assert on_bottom.mean() > 0.95, "only the missing bottom face is filled"
    assert rep["filled_area_mm2"] == pytest.approx(np.pi * (15 ** 2 - 5 ** 2), rel=0.35)
    assert not (added[:, 2] < -2).any() and not (added[:, 0] > 30).any()          # no table, no clutter
    assert rep["left_out"]["table"] + rep["left_out"]["outside_the_part"] > 1000
    assert 0.2 < rep["photo_vs_scan_mm"]["median"] < 0.6
    assert rep["gap_mm"] >= 0.5


def test_nothing_to_fill():
    scan = np.asarray(stepped_tube_scan(n=150_000, hole_grow=0, bump=0, rough=0, missing_bottom=False).points)
    part = sample_surface(stepped_tube(), 40_000, seed=5) + np.random.default_rng(5).normal(0, 0.2, (40_000, 3))
    out = fill_gaps(scan, part, estimate_spacing(scan), log=lambda m: None)
    assert out["report"]["added"] < 40
    assert len(out["points"]) == len(scan) + out["report"]["added"]


def test_the_table_under_the_part_is_never_added():
    """The part stands on the table: the scan misses the bottom face and the table touches it. The table must not
    turn into a fake bottom face."""
    rng = np.random.default_rng(7)
    scan = np.asarray(stepped_tube_scan(n=150_000, hole_grow=0, bump=0, rough=0).points)
    part = sample_surface(stepped_tube(), 60_000, seed=7) + rng.normal(0, 0.4, (60_000, 3))
    part = part[part[:, 2] > 0.5]                                                # the camera never sees the bottom
    table = np.c_[rng.uniform(-45, 45, (20_000, 2)), rng.normal(0, 0.3, 20_000)]
    table = table[np.hypot(table[:, 0], table[:, 1]) > 15]                      # hidden under the part
    out = fill_gaps(scan, np.vstack([part, table]), estimate_spacing(scan), log=lambda m: None)
    added = out["points"][out["source"] == 1]
    assert len(added) < 50, "table points were added"
    assert out["report"]["left_out"]["table"] > 5000


# --------------------------------------------------------------------------- the job, with the colour tests' fake photos
from tests.test_colour_photos import Log, add_photos, fake_reconstruction, scan_cloud, scene  # noqa: E402,F401

import cloudclean.web.jobs_colour_photos as jcp  # noqa: E402
import cloudclean.web.jobs_photo_fill as jpf  # noqa: E402
from cloudclean.web.workspace import Workspace  # noqa: E402

LUG = (np.array([4.0, -5.0, 26.0]), np.array([24.0, 5.0, 34.0]))       # the side lug of the test flange


def test_the_job_fills_what_the_scanner_missed(tmp_path, monkeypatch, scene):
    """The scanner missed the flange's side lug; 12 photos show it. The job lines the photos up, adds the lug's photo
    points (marked as from photos) and nothing of the table."""
    ws = Workspace(tmp_path / "ws")
    project = ws.create_project("Flange")
    full = scan_cloud()
    P = np.asarray(full.points)
    lug = np.all((P > LUG[0] - 0.5) & (P < LUG[1] + 0.5), axis=1) & (P[:, 0] > 6.5)
    scan = ws.add_geometry(full.select_by_index(np.flatnonzero(~lug)), "flange scan", "import", project=project["id"])
    add_photos(ws, tmp_path, 12, project["id"])
    fake_reconstruction(monkeypatch, scene)
    monkeypatch.setattr(jpf, "reconstruction", jcp.reconstruction)

    created = jpf.job_fill_from_photos(ws, {"asset_id": scan["id"]}, Log())

    meta, report = ws.get(created[0]), ws.report(created[0])
    assert meta["operation"] == "photo_fill" and meta["parents"] == [scan["id"]]
    assert meta["name"] == "flange scan + photo fill"
    source = np.load(ws.scalars_path(created[0], "source", preview=False))
    pts = np.asarray(ws.load_geometry(created[0]).points)
    assert len(source) == len(pts) and (source[: int((~lug).sum())] == 0).all()
    added = pts[source > 0.5]
    assert report["added"] == len(added) > 200, (report["left_out"], report["warnings"])
    near_lug = np.all((added > LUG[0] - 2.0) & (added < LUG[1] + 2.0), axis=1)
    assert near_lug.mean() > 0.9, near_lug.mean()                           # the lug, not the table
    assert not (added[:, 2] < -5.0).any()
    assert report["photos"] == 12 and report["line_up"]["fit_rmse_mm"] < 0.5
    # without its lug the flange nearly looks the same turned 144 deg: the fill is right, but the report says to check
    if not report["line_up"]["trusted"]:
        assert any("check that the filled areas sit where they belong" in w for w in report["warnings"])
