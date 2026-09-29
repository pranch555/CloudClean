"""Tessellating CAD models (STEP / IGES / BREP) into triangle meshes in millimetres."""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d

Log = Callable[[str], None]

DEFAULT_TOLERANCE = 0.01      # mm linear deflection: chord error ~0.005 mm, well below scanner accuracy
DEFAULT_ANGULAR = 0.2         # radians angular deflection

_GLTF_COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_GLTF_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def cad_backend() -> str | None:
    """Name of the available STEP/IGES tessellation backend ("cascadio", "ocp") or None."""
    try:
        import cascadio  # noqa: F401

        return "cascadio"
    except ImportError:
        pass
    try:
        import OCP  # noqa: F401

        return "ocp"
    except ImportError:
        return None


def step_length_unit(path) -> str | None:
    """Length unit declared in a STEP file (e.g. "mm", "m", "inch"), for reporting."""
    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read(4_000_000)
    except OSError:
        return None
    m = re.search(r"LENGTH_UNIT\s*\(\s*\)[^;]*?SI_UNIT\s*\(\s*(\.\w+\.|\$)\s*,\s*\.METRE\.", text, re.S) or \
        re.search(r"SI_UNIT\s*\(\s*(\.\w+\.|\$)\s*,\s*\.METRE\.\s*\)[^;]*?LENGTH_UNIT", text, re.S)
    if m:
        prefix = m.group(1).strip(".").upper()
        return {"MILLI": "mm", "CENTI": "cm", "MICRO": "um", "$": "m", "KILO": "km"}.get(prefix, prefix.lower() + "m")
    m = re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'([^']+)'", text)
    return m.group(1).lower() if m else None


# --------------------------------------------------------------------------- glTF parsing
def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], float).reshape(4, 4).T  # glTF is column major
    T = np.eye(4)
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        T[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion([w, x, y, z])
    if "scale" in node:
        T[:3, :3] = T[:3, :3] @ np.diag(node["scale"])
    if "translation" in node:
        T[:3, 3] = node["translation"]
    return T


def glb_to_mesh(glb: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Flatten all triangle primitives of a binary glTF (node transforms applied). Returns (vertices, triangles)."""
    magic, _, _ = struct.unpack_from("<4sII", glb, 0)
    if magic != b"glTF":
        raise ValueError("Tessellation did not produce a valid GLB")
    offset, doc, binary = 12, None, b""
    while offset < len(glb):
        length, kind = struct.unpack_from("<II", glb, offset)
        chunk = glb[offset + 8: offset + 8 + length]
        if kind == 0x4E4F534A:
            doc = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8 + length
    if doc is None:
        raise ValueError("GLB has no JSON chunk")

    def accessor(index: int) -> np.ndarray:
        acc = doc["accessors"][index]
        view = doc["bufferViews"][acc["bufferView"]]
        dtype = np.dtype(_GLTF_COMPONENTS[acc["componentType"]])
        width = _GLTF_WIDTH[acc["type"]]
        stride = view.get("byteStride") or dtype.itemsize * width
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        count = acc["count"]
        raw = np.frombuffer(binary, dtype=np.uint8, count=stride * (count - 1) + dtype.itemsize * width, offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dtype.itemsize * width), strides=(stride, 1))
        return np.frombuffer(rows.copy().tobytes(), dtype=dtype).reshape(count, width)

    verts, tris, base = [], [], 0

    def visit(node_index: int, parent: np.ndarray) -> None:
        nonlocal base
        node = doc["nodes"][node_index]
        T = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4 or "POSITION" not in prim["attributes"]:
                    continue
                v = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
                idx = accessor(prim["indices"]).reshape(-1) if "indices" in prim else np.arange(len(v))
                v = v @ T[:3, :3].T + T[:3, 3]
                t = idx.astype(np.int64).reshape(-1, 3)
                if np.linalg.det(T[:3, :3]) < 0:
                    t = t[:, ::-1]
                verts.append(v)
                tris.append(t + base)
                base += len(v)
        for child in node.get("children", []):
            visit(child, T)

    scene = doc.get("scenes", [{}])[doc.get("scene", 0)] if doc.get("scenes") else {}
    roots = scene.get("nodes", list(range(len(doc.get("nodes", [])))))
    for r in roots:
        visit(r, np.eye(4))
    if not verts:
        raise ValueError("The CAD file contains no surfaces")
    return np.vstack(verts), np.vstack(tris)


# --------------------------------------------------------------------------- backends
def _tessellate_cascadio(path: Path, tolerance: float, angular: float) -> tuple[np.ndarray, np.ndarray]:
    import cascadio

    ext = path.suffix.lower()
    if ext not in {".step", ".stp", ".iges", ".igs"}:
        raise ValueError(f"cascadio cannot read '{ext}' files - install cadquery-ocp-novtk for BREP support")
    glb = cascadio.load(path.read_bytes(), file_type="iges" if ext in {".iges", ".igs"} else "step",
                        tol_linear=tolerance, tol_angular=angular, tol_relative=False, merge_primitives=True,
                        use_parallel=True)
    if not glb:
        raise ValueError(f"Could not read {path.name} - is it a valid {ext[1:].upper()} file?")
    v, t = glb_to_mesh(glb)
    # OpenCASCADE's glTF writer always outputs metres (glTF convention) - convert back to millimetres.
    return v * 1000.0, t


def _tessellate_ocp(path: Path, tolerance: float, angular: float) -> tuple[np.ndarray, np.ndarray]:
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESControl import IGESControl_Reader
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Shape

    ext = path.suffix.lower()
    if ext in {".step", ".stp", ".iges", ".igs"}:
        reader = STEPControl_Reader() if ext in {".step", ".stp"} else IGESControl_Reader()
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            raise ValueError(f"Could not read {path.name}")
        reader.TransferRoots()
        shape = reader.OneShape()  # OCCT converts to millimetres by default
    else:
        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
            raise ValueError(f"Could not read {path.name}")
    BRepMesh_IncrementalMesh(shape, tolerance, False, angular, True)
    verts, tris, base = [], [], 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        explorer.Next()
        if tri is None:
            continue
        trsf = loc.Transformation()
        v = np.array([[p.X(), p.Y(), p.Z()] for p in
                      (tri.Node(i).Transformed(trsf) for i in range(1, tri.NbNodes() + 1))])
        t = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
        if face.Orientation() == TopAbs_REVERSED:
            t = t[:, ::-1]
        verts.append(v)
        tris.append(t + base)
        base += len(v)
    if not verts:
        raise ValueError(f"{path.name} contains no surfaces")
    return np.vstack(verts), np.vstack(tris)



def load_cad(path, tolerance: float | None = None, angular: float = DEFAULT_ANGULAR,
             log: Log | None = None) -> o3d.geometry.TriangleMesh:
    """Tessellate a STEP / IGES / BREP model into one welded triangle mesh in millimetres.

    tolerance: maximum linear deflection (chord error bound) in mm; default 0.01 mm.
    Nothing is rescaled beyond the unit conversion to millimetres done by the CAD kernel."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    tol = float(tolerance) if tolerance and tolerance > 0 else DEFAULT_TOLERANCE
    backend = cad_backend()
    if backend is None:
        raise RuntimeError("Reading STEP/IGES files needs a CAD kernel. Install one with "
                           "'pip install cascadio' (Windows x64, Linux x64/aarch64, macOS) or "
                           "'pip install cadquery-ocp-novtk', or export the CAD model as STL.")
    if backend == "cascadio" and path.suffix.lower() == ".brep":
        try:
            import OCP  # noqa: F401

            backend = "ocp"
        except ImportError:
            raise RuntimeError("BREP files need cadquery-ocp-novtk ('pip install cadquery-ocp-novtk'); "
                               "export the model as STEP instead.")
    tessellate = _tessellate_cascadio if backend == "cascadio" else _tessellate_ocp
    v, t = tessellate(path, tol, angular)

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(t.astype(np.int32)))
    # Faces are tessellated separately but share exactly the same edge nodes: weld them into one surface.
    mesh.merge_close_vertices(1e-6)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise ValueError(f"{path.name} produced no triangles")
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    if log:
        from .compare import is_closed

        unit = step_length_unit(path) if path.suffix.lower() in {".step", ".stp"} else None
        dims = np.asarray(mesh.get_max_bound() - mesh.get_min_bound())
        log(f"Tessellated {path.name} with {backend} (deflection {tol} mm"
            f"{', file unit ' + unit if unit else ''}): {len(mesh.triangles):,} triangles, "
            f"size {dims.round(4).tolist()} mm, closed {is_closed(np.asarray(mesh.triangles))}")
    return mesh
