"""Loading, saving and describing point clouds and meshes."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

Geometry = Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]

TEXT_CLOUD_EXTS = {".pts", ".asc", ".txt", ".csv"}
CLOUD_EXTS = {".pcd", ".xyz", ".xyzn", ".xyzrgb"} | TEXT_CLOUD_EXTS
MESH_EXTS = {".obj", ".stl", ".off", ".gltf", ".glb"}
HEIF_EXTS = {".heic", ".heif"}   # phone photos (iPhone default): turned into JPEG on the way in, see to_jpeg
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"} | HEIF_EXTS
CAD_EXTS = {".step", ".stp", ".iges", ".igs", ".brep"}   # tessellated by cloudclean.cad
SUPPORTED_EXTS = CLOUD_EXTS | MESH_EXTS | CAD_EXTS | {".ply"}


def is_image(path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def to_jpeg(src, dst, max_side: int | None = None, quality: int = 95) -> Path:
    """A photo as an upright JPEG: EXIF rotation applied, the rest of the EXIF kept (focal length). Reads HEIC/HEIF
    phone photos too (pillow-heif), which browsers other than Safari cannot show."""
    from PIL import Image, ImageOps

    if Path(src).suffix.lower() in HEIF_EXTS:
        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            raise ValueError("HEIC photos need the pillow-heif package (pip install pillow-heif) - or save the "
                             "photos as JPEG") from None
        register_heif_opener()
    with Image.open(src) as im:
        exif = im.getexif()
        up = ImageOps.exif_transpose(im).convert("RGB")
    if max_side and max(up.size) > max_side:
        up.thumbnail((max_side, max_side))
    if 0x0112 in exif:
        exif[0x0112] = 1  # the rotation is applied to the pixels now
    up.save(dst, "JPEG", quality=quality, exif=exif.tobytes() if len(exif) else b"")
    return Path(dst)


def is_cloud(geom) -> bool:
    return isinstance(geom, o3d.geometry.PointCloud)


# --------------------------------------------------------------------------- loading
def _ply_face_count(path: Path) -> int:
    with open(path, "rb") as f:
        for _ in range(300):
            line = f.readline()
            if not line:
                break
            text = line.decode("ascii", errors="ignore").strip()
            if text.startswith("element face"):
                return int(text.split()[-1])
            if text == "end_header":
                break
    return 0


def _looks_like_normals(block: np.ndarray) -> bool:
    if block.shape[1] != 3 or len(block) == 0:
        return False
    if np.abs(block).max() > 1.0001:
        return False
    return float(np.median(np.abs(np.linalg.norm(block, axis=1) - 1.0))) < 0.05


def _as_colors(block: np.ndarray) -> np.ndarray:
    block = np.clip(block, 0, None)
    return block / 255.0 if block.max() > 1.0 else block


def _load_text_cloud(path: Path) -> o3d.geometry.PointCloud:
    """ASCII clouds: x y z [r g b] [nx ny nz], PTS (count line + x y z i r g b), CSV."""
    first_row, ncols, delimiter = None, 0, None
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > 100:
                break
            text = line.strip()
            if not text or text.startswith(("#", "//")):
                continue
            parts = [p for p in re.split(r"[,;\s]+", text) if p]
            try:
                [float(p) for p in parts]
            except ValueError:
                continue
            if len(parts) < 3:  # e.g. the point count line of a .pts file
                continue
            first_row, ncols = i, len(parts)
            delimiter = "," if "," in text else (";" if ";" in text else None)
            break
    if first_row is None:
        raise ValueError(f"No numeric x y z rows found in {path.name}")

    data = np.loadtxt(path, delimiter=delimiter, skiprows=first_row, usecols=range(ncols),
                      comments=("#", "//"), ndmin=2, dtype=np.float64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data[:, :3]))
    extra = data[:, 3:]
    colors = normals = None
    if extra.shape[1] == 3:
        if _looks_like_normals(extra):
            normals = extra
        else:
            colors = _as_colors(extra)
    elif extra.shape[1] == 4:  # PTS: intensity r g b
        colors = _as_colors(extra[:, 1:4])
    elif extra.shape[1] >= 6:
        a, b = extra[:, :3], extra[:, 3:6]
        if _looks_like_normals(a):
            normals, colors = a, _as_colors(b)
        elif _looks_like_normals(b):
            colors, normals = _as_colors(a), b
        else:
            colors = _as_colors(a)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def load(path, *, repair: bool = True) -> Geometry:
    """Load a point cloud or mesh. PLY files are detected by whether they contain faces.

    `repair=False` returns the file exactly as stored, without dropping an all-black colour array; only maintenance
    tools that need to see what is on disk should ask for that.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext in CAD_EXTS:
        from .cad import load_cad

        geom = load_cad(path)
    elif ext in TEXT_CLOUD_EXTS:
        geom = _load_text_cloud(path)
    elif ext == ".ply":
        if _ply_face_count(path) > 0:
            geom = o3d.io.read_triangle_mesh(str(path))
        else:
            geom = o3d.io.read_point_cloud(str(path))
    elif ext in CLOUD_EXTS:
        geom = o3d.io.read_point_cloud(str(path))
    elif ext in MESH_EXTS:
        geom = o3d.io.read_triangle_mesh(str(path), enable_post_processing=ext in {".gltf", ".glb"})
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTS))}")

    if is_cloud(geom):
        if len(geom.points) == 0:
            raise ValueError(f"{path.name} contains no points")
    else:
        if len(geom.vertices) == 0:
            raise ValueError(f"{path.name} contains no vertices")
        if len(geom.triangles) == 0:  # vertex-only file: treat as a point cloud
            cloud = o3d.geometry.PointCloud(geom.vertices)
            if geom.has_vertex_colors():
                cloud.colors = geom.vertex_colors
            if geom.has_vertex_normals():
                cloud.normals = geom.vertex_normals
            geom = cloud
    if repair:
        drop_blank_colors(geom)
    return geom


def has_blank_colors(geom: Geometry) -> bool:
    """True when the geometry carries a colour array that is entirely black, so it encodes nothing."""
    cloud = is_cloud(geom)
    if not (geom.has_colors() if cloud else geom.has_vertex_colors()):
        return False
    cols = np.asarray(geom.colors if cloud else geom.vertex_colors)
    return not (cols.size and cols.max() > 1e-6)


def drop_blank_colors(geom: Geometry) -> bool:
    """Discard an all-black colour array, which carries no information but hides the shaded surface.

    Poisson reconstruction (and several exporters) always write a colour per vertex, zero-filled when the source had
    no colour. Loaded back it reads as a genuinely black scan, so the viewer renders an unlit black blob. Files whose
    colours are truly all black lose nothing by falling back to the shaded material. Returns True if removed.
    """
    if not has_blank_colors(geom):
        return False
    cloud = is_cloud(geom)
    blank = o3d.utility.Vector3dVector(np.empty((0, 3)))
    if cloud:
        geom.colors = blank
    else:
        geom.vertex_colors = blank
    return True


# --------------------------------------------------------------------------- saving
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def write_glb(path, positions, indices, normals=None, colors_rgba8=None, uvs=None, texture_png: bytes | None = None):
    """Minimal binary glTF 2.0 writer (much faster than generic exporters for multi-million triangle meshes)."""
    import json
    import struct

    buf = bytearray()
    views, accessors = [], []

    def add_view(data: bytes, target=None) -> int:
        while len(buf) % 4:
            buf.append(0)
        view = {"buffer": 0, "byteOffset": len(buf), "byteLength": len(data)}
        if target:
            view["target"] = target
        buf.extend(data)
        views.append(view)
        return len(views) - 1

    def add_accessor(arr, target, component, kind, normalized=False, bounds=False) -> int:
        arr = np.ascontiguousarray(arr)
        acc = {"bufferView": add_view(arr.tobytes(), target), "componentType": component,
               "count": int(len(arr)), "type": kind}
        if normalized:
            acc["normalized"] = True
        if bounds:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    attributes = {"POSITION": add_accessor(np.asarray(positions, np.float32), 34962, 5126, "VEC3", bounds=True)}
    if normals is not None:
        attributes["NORMAL"] = add_accessor(np.asarray(normals, np.float32), 34962, 5126, "VEC3")
    if colors_rgba8 is not None:
        attributes["COLOR_0"] = add_accessor(np.asarray(colors_rgba8, np.uint8), 34962, 5121, "VEC4", normalized=True)
    if uvs is not None:
        attributes["TEXCOORD_0"] = add_accessor(np.asarray(uvs, np.float32), 34962, 5126, "VEC2")
    index_acc = add_accessor(np.asarray(indices, np.uint32).reshape(-1), 34963, 5125, "SCALAR")

    material = {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}, "doubleSided": True}
    gltf = {"asset": {"version": "2.0", "generator": "cloudclean"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}], "materials": [material],
            "meshes": [{"primitives": [{"attributes": attributes, "indices": index_acc, "material": 0}]}]}
    if texture_png is not None:
        gltf["images"] = [{"bufferView": add_view(texture_png), "mimeType": "image/png"}]
        gltf["samplers"] = [{}]
        gltf["textures"] = [{"source": 0, "sampler": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    while len(buf) % 4:
        buf.append(0)
    gltf["accessors"], gltf["bufferViews"] = accessors, views
    gltf["buffers"] = [{"byteLength": len(buf)}]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(buf)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(buf), 0x004E4942))
        f.write(buf)


def _save_glb(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    colors = None
    if mesh.has_vertex_colors():
        # glTF COLOR_0 is linear; our colors are sRGB (photos / scanner).
        lin = _srgb_to_linear(np.clip(np.asarray(mesh.vertex_colors), 0, 1))
        colors = np.hstack([np.round(lin * 255), np.full((len(lin), 1), 255)]).astype(np.uint8)
    write_glb(path, np.asarray(mesh.vertices), np.asarray(mesh.triangles),
              normals=np.asarray(mesh.vertex_normals), colors_rgba8=colors)


def _save_text_cloud(pcd: o3d.geometry.PointCloud, path: Path) -> None:
    cols = [np.asarray(pcd.points)]
    fmt = ["%.6f"] * 3
    if pcd.has_colors():
        cols.append(np.round(np.asarray(pcd.colors) * 255))
        fmt += ["%d"] * 3
    if pcd.has_normals():
        cols.append(np.asarray(pcd.normals))
        fmt += ["%.5f"] * 3
    delimiter = "," if path.suffix.lower() == ".csv" else " "
    np.savetxt(path, np.hstack(cols), fmt=fmt, delimiter=delimiter)


def save(geom: Geometry, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    drop_blank_colors(geom)  # never store an all-black array: it hides the surface in every viewer
    if is_cloud(geom):
        if ext in {".asc", ".txt", ".csv"}:
            _save_text_cloud(geom, path)
            return path
        if ext not in {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}:
            raise ValueError(f"A point cloud cannot be saved as '{ext}' - build a mesh first")
        ok = o3d.io.write_point_cloud(str(path), geom, write_ascii=False, compressed=False)
    else:
        if ext == ".glb":
            _save_glb(geom, path)
            return path
        if ext == ".3mf":
            _save_3mf(geom, path)
            return path
        if ext not in {".ply", ".obj", ".stl", ".off"}:
            raise ValueError(f"A mesh cannot be saved as '{ext}'")
        if not geom.has_vertex_normals():
            geom.compute_vertex_normals()
        if ext == ".stl":
            geom.compute_triangle_normals()
        ok = o3d.io.write_triangle_mesh(str(path), geom, write_ascii=False, compressed=False,
                                        write_vertex_normals=True, write_vertex_colors=True)
    if not ok:
        raise IOError(f"Failed to write {path}")
    return path


_3MF_TYPES = ('<?xml version="1.0" encoding="UTF-8"?>\n'
              '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
              '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
              '</Types>')
_3MF_RELS = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
             'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')


def _save_3mf(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    """3MF (3D Manufacturing Format) for 3D printing: a zip with one XML mesh in millimetres. Coordinates are written
    with 6 decimals (1 nm), so the export adds no measurable error. No dependency beyond the standard library."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    t = np.asarray(mesh.triangles, dtype=np.int64)
    if not len(t):
        raise ValueError("The mesh has no triangles to save as 3MF")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", _3MF_TYPES)
        z.writestr("_rels/.rels", _3MF_RELS)
        with z.open("3D/3dmodel.model", "w", force_zip64=True) as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
                    b'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
                    b'<metadata name="Application">CloudClean</metadata><resources>'
                    b'<object id="1" type="model"><mesh><vertices>')
            for i in range(0, len(v), 200_000):
                f.write("".join(f'<vertex x="{x:.6f}" y="{y:.6f}" z="{w:.6f}"/>'
                                for x, y, w in v[i:i + 200_000].tolist()).encode())
            f.write(b"</vertices><triangles>")
            for i in range(0, len(t), 200_000):
                f.write("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>'
                                for a, b, c in t[i:i + 200_000].tolist()).encode())
            f.write(b'</triangles></mesh></object></resources><build><item objectid="1"/></build></model>')


# --------------------------------------------------------------------------- analysis
def estimate_spacing(points, sample: int = 20000, seed: int = 0) -> float:
    """Median nearest-neighbour distance (the scan's point spacing / resolution)."""
    pts = np.asarray(points)
    if len(pts) < 2:
        return 0.0
    tree = cKDTree(pts)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=min(sample, len(pts)), replace=False)
    dist, _ = tree.query(pts[idx], k=2, workers=-1)
    d = dist[:, 1]
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.0


def to_cloud(geom: Geometry, min_points: int = 100_000) -> o3d.geometry.PointCloud:
    """Point cloud view of any geometry (mesh vertices, densified by sampling if sparse)."""
    if is_cloud(geom):
        return geom
    mesh = geom
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if len(mesh.vertices) >= min_points or len(mesh.triangles) == 0:
        pcd = o3d.geometry.PointCloud(mesh.vertices)
        pcd.normals = mesh.vertex_normals
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors
        return pcd
    return mesh.sample_points_uniformly(number_of_points=min_points)


def closed_surface(triangles) -> bool:
    """True when every undirected edge is used by exactly two triangles (closed 2-manifold edges)."""
    tris = np.asarray(triangles, dtype=np.int64)
    if len(tris) == 0:
        return False
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges[:, 0] * (int(tris.max()) + 1) + edges[:, 1], return_counts=True)
    return bool(np.all(counts == 2))


def describe(geom: Geometry) -> dict:
    if is_cloud(geom):
        pts = np.asarray(geom.points)
        info = {"kind": "pointcloud", "points": int(len(pts)),
                "has_colors": bool(geom.has_colors()), "has_normals": bool(geom.has_normals())}
    else:
        pts = np.asarray(geom.vertices)
        info = {"kind": "mesh", "vertices": int(len(pts)), "triangles": int(len(geom.triangles)),
                "has_colors": bool(geom.has_vertex_colors()), "has_normals": bool(geom.has_vertex_normals())}
        try:
            info["surface_area"] = float(geom.get_surface_area())
            # Open3D's is_watertight() includes a self-intersection test that takes minutes on large meshes.
            # A closed surface has every edge shared by exactly two triangles; volume is the divergence sum.
            tris = np.asarray(geom.triangles)
            watertight = closed_surface(tris)
            info["watertight"] = watertight
            if watertight:
                v = pts[tris]
                info["volume"] = float(abs(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum()) / 6.0)
        except Exception:
            pass
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    info["bbox_min"] = mn.round(4).tolist()
    info["bbox_max"] = mx.round(4).tolist()
    info["dimensions"] = (mx - mn).round(4).tolist()
    info["diagonal"] = round(float(np.linalg.norm(mx - mn)), 4)
    try:
        rng = np.random.default_rng(0)
        sample = pts if len(pts) <= 200_000 else pts[rng.choice(len(pts), 200_000, replace=False)]
        # smallest box around all points in the part's own axes (Open3D's oriented box leans by degrees on long
        # round parts, which read 0.5-1 mm too long and several mm too wide; docs/accuracy-investigation-2026-09-24.md)
        from .accuracy import principal_frame, refine_axes
        centre, axes, _ = principal_frame(sample)
        axes = refine_axes(sample, centre, axes, trim=0.0)
        coords = (pts - centre) @ axes.T
        info["oriented_dimensions"] = sorted((coords.max(axis=0) - coords.min(axis=0)).round(4).tolist(), reverse=True)
    except Exception:
        pass
    info["spacing"] = round(estimate_spacing(pts), 6)
    return info


def preview(geom: Geometry, max_points: int = 1_500_000, max_triangles: int = 600_000) -> Geometry:
    """Lightweight copy for the browser viewer."""
    if is_cloud(geom):
        n = len(geom.points)
        if n <= max_points:
            return geom
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
        return geom.select_by_index(np.sort(idx))
    if len(geom.triangles) <= max_triangles:
        out = geom
    else:
        out = geom.simplify_quadric_decimation(target_number_of_triangles=max_triangles)
    if not out.has_vertex_normals():
        out.compute_vertex_normals()
    return out
