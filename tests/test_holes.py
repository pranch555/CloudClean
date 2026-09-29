"""Hole detection and selective hole filling (cloudclean/holes.py, routes_holes.py, jobs_holes.py)."""
from __future__ import annotations

import time

import numpy as np
import open3d as o3d
from fastapi.testclient import TestClient

from cloudclean.holes import fill_holes, find_holes, public_holes
from cloudclean.io import save
from cloudclean.web.server import create_app


def _sphere_with_holes(radius: float = 10.0, cut: list[np.ndarray] | None = None) -> o3d.geometry.TriangleMesh:
    """A closed sphere with every triangle whose centroid is within 2.5 mm of a cut point removed."""
    m = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=40)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    keep = np.ones(len(t), bool)
    for c in cut or []:
        keep &= np.linalg.norm(v[t].mean(axis=1) - c, axis=1) > 2.5
    out = o3d.geometry.TriangleMesh(m.vertices, o3d.utility.Vector3iVector(t[keep]))
    out.remove_unreferenced_vertices()
    return out


def test_closed_mesh_has_no_holes():
    r = find_holes(_sphere_with_holes())
    assert r["watertight"] and r["holes"] == [] and r["boundary_edges"] == 0


def test_finds_each_hole_with_its_size_and_place():
    cuts = [np.array([10.0, 0, 0]), np.array([0, 0, -10.0])]
    r = find_holes(_sphere_with_holes(cut=cuts))
    assert len(r["holes"]) == 2 and not r["watertight"]
    for h in r["holes"]:
        # a ~2.5 mm radius cap: loop diameter from the perimeter is about 5 mm (triangles make it ragged)
        assert 3.5 < h["diameter"] < 7.5, h["diameter"]
        assert not h["outer"]
        assert min(np.linalg.norm(np.array(h["center"]) - c) for c in cuts) < 1.0
        # the loop normal points along the radius (either way round)
        radial = np.array(h["center"]) / np.linalg.norm(h["center"])
        assert abs(abs(float(np.dot(h["normal"], radial))) - 1) < 0.05
    pub = public_holes(r)
    assert pub["count"] == 2 and pub["holes_to_fill"] == 2 and "loop" not in pub["holes"][0]


def test_open_scan_rim_is_reported_as_outer_edge_not_a_hole():
    m = o3d.geometry.TriangleMesh.create_sphere(radius=10.0, resolution=40)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    t = t[v[t].mean(axis=1)[:, 2] > -2.0]          # an unscanned bottom: a big open rim
    t = t[np.linalg.norm(v[t].mean(axis=1) - [10.0, 0, 0], axis=1) > 2.5]   # plus one real hole
    open_mesh = o3d.geometry.TriangleMesh(m.vertices, o3d.utility.Vector3iVector(t))
    open_mesh.remove_unreferenced_vertices()
    holes = find_holes(open_mesh)["holes"]
    assert len(holes) == 2 and holes[0]["outer"] and not holes[1]["outer"]
    filled, report = fill_holes(open_mesh)                 # default: never the outer rim
    assert report["filled"] == 1 and report["skipped"] == 1
    assert len(find_holes(filled)["holes"]) == 1


def test_fill_makes_the_mesh_watertight_with_consistent_winding():
    mesh = _sphere_with_holes(cut=[np.array([10.0, 0, 0]), np.array([0, 10.0, 0])])
    filled, report = fill_holes(mesh, log=lambda m: None)
    assert report["filled"] == 2 and report["triangles_added"] > 0 and report["area_added"] > 0
    assert filled.is_watertight() and filled.is_orientable()
    # the original surface is untouched: its vertices are all still there, unmoved
    np.testing.assert_array_equal(np.asarray(filled.vertices)[: len(mesh.vertices)], np.asarray(mesh.vertices))
    # outward normals stay outward on the new triangles (winding agrees with the neighbours)
    filled.compute_triangle_normals()
    n = np.asarray(filled.triangle_normals)[len(mesh.triangles):]
    c = np.asarray(filled.vertices)[np.asarray(filled.triangles)[len(mesh.triangles):]].mean(axis=1)
    assert (np.einsum("ij,ij->i", n, c) > 0).all()


def test_fill_only_chosen_or_small_holes():
    mesh = _sphere_with_holes(cut=[np.array([10.0, 0, 0]), np.array([0, 0, 10.0])])
    ids = [h["id"] for h in find_holes(mesh)["holes"]]
    one, rep = fill_holes(mesh, hole_ids=[ids[0]], log=lambda m: None)
    assert rep["filled"] == 1 and len(find_holes(one)["holes"]) == 1
    none, rep = fill_holes(mesh, max_diameter=1.0, log=lambda m: None)
    assert rep["filled"] == 0
    try:
        fill_holes(mesh, hole_ids=[99])
    except ValueError as exc:
        assert "unknown hole" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown hole id accepted")


def _wait(client, job, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        cur = {j["id"]: j for j in client.get("/api/jobs").json()}[job["id"]]
        if cur["status"] in ("done", "failed", "cancelled"):
            return cur
        time.sleep(0.3)
    raise TimeoutError(job["title"])


def test_holes_api(tmp_path):
    save(_sphere_with_holes(cut=[np.array([10.0, 0, 0])]), tmp_path / "ball.ply")
    with TestClient(create_app(tmp_path / "ws")) as client:
        job = client.post("/api/import-paths", json={"paths": [str(tmp_path / "ball.ply")]}).json()["jobs"][0]
        asset = _wait(client, job)["result"][0]
        r = client.get(f"/api/assets/{asset}/holes")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["count"] == 1 and body["holes_to_fill"] == 1 and body["holes"][0]["diameter"] > 3
        assert client.post("/api/holes/fill", json={"asset_id": asset, "hole_ids": []}).status_code == 400
        done = _wait(client, client.post("/api/holes/fill", json={"asset_id": asset}).json())
        assert done["status"] == "done", done.get("error")
        new_id = done["result"][0]
        info = client.get(f"/api/assets/{new_id}").json()
        assert info["parents"] == [asset] and "filled" in info["name"]
        assert client.get(f"/api/assets/{new_id}/holes").json()["watertight"]
