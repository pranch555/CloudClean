"""3MF export (cloudclean.io._save_3mf): valid package, millimetres, coordinates kept to 1e-6."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import open3d as o3d

from cloudclean.io import save

NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


def test_3mf_round_trip(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=12.345678, resolution=12)
    mesh.translate((100.1234567, -50.5, 7.25))
    path = save(mesh, tmp_path / "part.3mf")
    with zipfile.ZipFile(path) as z:
        assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"} <= set(z.namelist())
        root = ET.fromstring(z.read("3D/3dmodel.model"))
    assert root.get("unit") == "millimeter"
    verts = np.array([[float(e.get(k)) for k in "xyz"] for e in root.iterfind(".//m:vertex", NS)])
    tris = np.array([[int(e.get(k)) for k in ("v1", "v2", "v3")] for e in root.iterfind(".//m:triangle", NS)])
    np.testing.assert_allclose(verts, np.asarray(mesh.vertices), atol=1e-6)
    np.testing.assert_array_equal(tris, np.asarray(mesh.triangles))
    assert root.find(".//m:build/m:item", NS).get("objectid") == "1"
