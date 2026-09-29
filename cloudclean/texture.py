"""Colouring a mesh from photos of the item (vertex colours and optional UV texture)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
from PIL import Image, ImageOps
from scipy.spatial import cKDTree

from .mesh import cleanup_mesh
from .params import ParamsMixin

Log = Callable[[str], None]


@dataclass
class TextureParams(ParamsMixin):
    blend_power: float = 4.0          # 1 = soft average of photos, higher = prefer the most head-on photo
    min_cos: float = 0.15             # ignore surfaces seen at grazing angles
    edge_margin_px: int = 6           # ignore surface near silhouettes (avoids background bleeding)
    visibility_tolerance: float = 0.003   # occlusion test tolerance, fraction of object size
    unseen: str = "diffuse"           # colour for areas no photo sees: diffuse | scan | gray
    texture_size: int = 0             # >0: also bake a UV texture image of this size (e.g. 4096)
    texture_max_triangles: int = 300_000  # mesh is decimated to this before UV unwrapping


# --------------------------------------------------------------------------- cameras
def image_size(path) -> tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(0x0112) in (5, 6, 7, 8):  # EXIF rotation by 90 degrees
            w, h = h, w
    return w, h


def load_image(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float32) / 255.0


@dataclass
class CameraView:
    """Pinhole camera for one photo. world_to_camera uses the OpenCV convention (x right, y down, z forward)."""
    image_path: str
    K: np.ndarray
    world_to_camera: np.ndarray
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        R, t = self.world_to_camera[:3, :3], self.world_to_camera[:3, 3]
        return -R.T @ t

    @classmethod
    def from_threejs(cls, image_path, view_matrix, fov: float, aspect: float | None = None,
                     width: int | None = None, height: int | None = None) -> "CameraView":
        """From a three.js PerspectiveCamera: matrixWorldInverse.elements (column major) and vertical fov."""
        if width is None or height is None:
            width, height = image_size(image_path)
        M = np.asarray(view_matrix, dtype=float).reshape(4, 4).T
        T = np.diag([1.0, -1.0, -1.0, 1.0]) @ M
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        aspect = aspect or width / height
        K = np.array([[f * width / (2 * aspect), 0, width / 2.0],
                      [0, f * height / 2.0, height / 2.0],
                      [0, 0, 1]])
        return cls(str(image_path), K, T, width, height)

    @classmethod
    def from_dict(cls, data: dict, image_path=None, base_dir=None) -> "CameraView":
        img = image_path or data.get("image")
        if img is None:
            raise ValueError("Camera needs an image path")
        img = Path(img)
        if not img.is_absolute() and base_dir is not None and not img.exists():
            img = Path(base_dir) / img
        W, H = image_size(img)
        if "view_matrix" in data:
            return cls.from_threejs(img, data["view_matrix"], float(data["fov"]), data.get("aspect"), W, H)
        if "K" in data:
            K = np.asarray(data["K"], dtype=float)
        else:
            K = np.array([[data["fx"], 0, data["cx"]], [0, data["fy"], data["cy"]], [0, 0, 1]], dtype=float)
        K[0] *= W / float(data.get("width", W))
        K[1] *= H / float(data.get("height", H))
        return cls(str(img), K, np.asarray(data["world_to_camera"], dtype=float), W, H)

    def to_dict(self) -> dict:
        return {"image": self.image_path, "width": self.width, "height": self.height,
                "K": self.K.tolist(), "world_to_camera": self.world_to_camera.tolist()}


# --------------------------------------------------------------------------- vertex colours
def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    x0 = np.clip(np.floor(u).astype(int), 0, W - 1)
    y0 = np.clip(np.floor(v).astype(int), 0, H - 1)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def _cast(scene, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    rays = np.hstack([np.broadcast_to(origin, dirs.shape), dirs]).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()


def _diffuse_fill(tris: np.ndarray, colors: np.ndarray, known: np.ndarray, iterations: int = 40):
    """Grow known colours into unknown vertices along mesh edges, then nearest-neighbour the rest."""
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    n = len(colors)
    for _ in range(iterations):
        ee = e[known[e[:, 0]] & ~known[e[:, 1]]]
        if len(ee) == 0:
            break
        cnt = np.bincount(ee[:, 1], minlength=n)
        target = cnt > 0
        for ch in range(3):
            s = np.bincount(ee[:, 1], weights=colors[ee[:, 0], ch], minlength=n)
            colors[target, ch] = s[target] / cnt[target]
        known = known | target
    return colors, known


def project_vertex_colors(mesh: o3d.geometry.TriangleMesh, views: list[CameraView],
                          params: TextureParams | None = None, log: Log = print):
    p = params or TextureParams()
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    tris = np.asarray(mesh.triangles)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    tol = max(p.visibility_tolerance * diag, 1e-9)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(tris.astype(np.uint32)))

    acc = np.zeros((len(V), 3))
    wsum = np.zeros(len(V))
    per_view = []
    for i, view in enumerate(views):
        img = load_image(view.image_path)
        H, W = img.shape[:2]
        K = view.K.copy()
        K[0] *= W / view.width
        K[1] *= H / view.height
        R, t = view.world_to_camera[:3, :3], view.world_to_camera[:3, 3]
        C = view.center
        Pc = V @ R.T + t
        z = Pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * Pc[:, 0] / z + K[0, 1] * Pc[:, 1] / z + K[0, 2]
            v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
        to_cam = C - V
        dist = np.linalg.norm(to_cam, axis=1)
        cos = np.einsum("ij,ij->i", N, to_cam / dist[:, None])
        m = p.edge_margin_px
        cand = (z > 0) & (u >= m) & (u <= W - 1 - m) & (v >= m) & (v <= H - 1 - m) & (cos > p.min_cos)
        idx = np.flatnonzero(cand)
        if len(idx):
            dirs = -(to_cam[idx] / dist[idx, None])
            idx = idx[_cast(scene, C, dirs) >= dist[idx] - tol]
        if len(idx) and m > 0:  # reject vertices near depth discontinuities (silhouettes)
            Kinv = np.linalg.inv(K)
            ok = np.ones(len(idx), bool)
            for du, dv in ((m, 0), (-m, 0), (0, m), (0, -m)):
                pix = np.stack([u[idx] + du, v[idx] + dv, np.ones(len(idx))], axis=1)
                d = (pix @ Kinv.T) @ R  # camera ray -> world direction
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                hit = _cast(scene, C, d)
                ok &= np.isfinite(hit) & (np.abs(hit - dist[idx]) < 0.05 * diag)
            idx = idx[ok]
        if len(idx):
            w = np.clip(cos[idx], 0, 1) ** p.blend_power
            acc[idx] += _bilinear(img, u[idx], v[idx]) * w[:, None]
            wsum[idx] += w
        per_view.append({"image": Path(view.image_path).name, "visible_vertices": int(len(idx))})
        log(f"  photo {i + 1} ({Path(view.image_path).name}): colours {len(idx):,} vertices")

    seen = wsum > 0
    colors = np.zeros((len(V), 3))
    colors[seen] = acc[seen] / wsum[seen, None]
    coverage = float(seen.mean())
    unseen = ~seen
    if unseen.any():
        if p.unseen == "scan" and mesh.has_vertex_colors():
            colors[unseen] = np.asarray(mesh.vertex_colors)[unseen]
        elif p.unseen == "diffuse" and seen.any():
            colors, known = _diffuse_fill(tris, colors, seen.copy())
            if (~known).any():
                _, nn = cKDTree(V[known]).query(V[~known], k=1, workers=-1)
                colors[~known] = colors[known][nn]
        else:
            colors[unseen] = 0.6
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    log(f"  photo coverage: {coverage:.1%} of vertices seen directly")
    return mesh, {"coverage": coverage, "views": per_view}


# --------------------------------------------------------------------------- point clouds
def colour_points(cloud: o3d.geometry.PointCloud, views: list[CameraView], params: TextureParams | None = None,
                  log: Log = print, progress: Callable | None = None):
    """Colour a point cloud from photos. Every photo that sees a point adds its colour there, weighted by how
    head-on it sees the surface (cos^blend_power). A point counts as seen when no nearer point covers it in a depth
    map made from the points themselves (splatted at a resolution where neighbouring points touch) and no much
    farther surface (or nothing) lies within edge_margin_px of it (silhouettes). Points no photo sees take the
    colour of their nearest coloured neighbours. Returns (coloured copy, report)."""
    from scipy.ndimage import maximum_filter, minimum_filter

    from .io import estimate_spacing

    p = params or TextureParams()
    if not views:
        raise ValueError("At least one photo with a camera pose is required")
    out = o3d.geometry.PointCloud(cloud)
    V = np.asarray(out.points, dtype=np.float64)
    n = len(V)
    spacing = estimate_spacing(V)
    signed = out.has_normals()
    if signed:
        N = np.asarray(out.normals)
    else:  # unoriented: which side faces a photo is left to the depth test
        tmp = o3d.geometry.PointCloud(out.points)
        tmp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * spacing, max_nn=30))
        N = np.asarray(tmp.normals)
    lo, hi = np.percentile(V, [1, 99], axis=0)
    jump = 0.05 * float(np.linalg.norm(hi - lo))   # a silhouette: this much farther behind (as for meshes)
    log(f"Colouring {n:,} points from {len(views)} photo(s)")

    acc = np.zeros((n, 3), dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)
    per_view = []
    m = p.edge_margin_px
    for i, view in enumerate(views):
        with Image.open(view.image_path) as im:
            img = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        H, W = img.shape[:2]
        K = _scaled_K(view, W, H)
        R, t = view.world_to_camera[:3, :3], view.world_to_camera[:3, 3]
        z = V @ R[2] + t[2]
        idx = np.flatnonzero(z > 0)
        x, y, z = V[idx] @ R[0] + t[0], V[idx] @ R[1] + t[1], z[idx]
        u = (K[0, 0] * x + K[0, 1] * y) / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
        inside = (u > -0.5) & (u < W - 0.5) & (v > -0.5) & (v < H - 0.5)
        idx, u, v, z = idx[inside], u[inside], v[inside], z[inside]
        used = 0
        if len(idx):
            # depth map pixels ~1.5 point spacings wide so the splats touch; a 3x3 minimum closes the pinholes
            f = int(np.clip(np.ceil(1.5 * spacing * K[0, 0] / np.median(z)), 2, 64))
            dw, dh = -(-W // f), -(-H // f)
            px = np.clip(np.rint(u).astype(np.int64) // f, 0, dw - 1)
            py = np.clip(np.rint(v).astype(np.int64) // f, 0, dh - 1)
            depth = np.full(dw * dh, np.inf, dtype=np.float32)
            np.minimum.at(depth, py * dw + px, z.astype(np.float32))
            depth = minimum_filter(depth.reshape(dh, dw), size=3)
            to_cam = view.center - V[idx]
            cos = np.einsum("ij,ij->i", N[idx], to_cam) / np.linalg.norm(to_cam, axis=1)
            if not signed:
                cos = np.abs(cos)
            # a slanted surface's depth changes across the filtered footprint: allow for it, plus the scan noise
            tan = np.sqrt(np.clip(1 - cos ** 2, 0, 1)) / np.maximum(cos, p.min_cos)
            tol = 3 * spacing + 2 * (f * z / K[0, 0]) * tan
            ok = (cos > p.min_cos) & (z <= depth[py, px] + tol)
            ok &= (u >= m) & (u <= W - 1 - m) & (v >= m) & (v <= H - 1 - m)
            if m > 0:
                r = max(1, math.ceil(m / f))
                far = maximum_filter(np.where(np.isfinite(depth), depth, np.float32(3e38)), size=2 * r + 1)
                ok &= far[py, px] - z <= jump
            sel = idx[ok]
            used = len(sel)
            if used:
                w = (np.clip(cos[ok], 0, 1) ** p.blend_power).astype(np.float32)
                acc[sel] += (_bilinear(img, u[ok], v[ok]) / 255.0 * w[:, None]).astype(np.float32)
                wsum[sel] += w
        per_view.append({"image": Path(view.image_path).name, "visible_points": int(used)})
        log(f"  photo {i + 1} ({Path(view.image_path).name}): colours {used:,} points")
        if progress:
            progress((i + 1) / len(views))

    seen = wsum > 0
    if not seen.any():
        raise ValueError("No photo sees the model - the cameras do not look at it")
    colors = np.zeros((n, 3))
    colors[seen] = acc[seen] / wsum[seen, None]
    unseen = np.flatnonzero(~seen)
    if len(unseen):
        # next to seen points: their inverse-distance mean; deep inside unseen areas (the underside): the nearest
        # colour of a coarse grid - a far search in the full cloud takes minutes
        known = np.flatnonzero(seen)
        k = min(4, len(known))
        d, nb = cKDTree(V[known]).query(V[unseen], k=k, distance_upper_bound=4 * spacing, workers=-1)
        d, nb = d.reshape(len(unseen), k), nb.reshape(len(unseen), k)
        near = np.isfinite(d[:, 0])
        w = np.where(np.isfinite(d), 1.0 / np.maximum(d, 1e-3 * spacing + 1e-12), 0.0)[near]
        colors[unseen[near]] = np.einsum("nk,nkc->nc", w, colors[known][np.minimum(nb[near], len(known) - 1)]) \
            / w.sum(1, keepdims=True)
        far = unseen[~near]
        if len(far):
            grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V[known]))
            grid.colors = o3d.utility.Vector3dVector(colors[known])
            grid = grid.voxel_down_sample(max(20 * spacing, 1e-9))
            _, nb = cKDTree(np.asarray(grid.points)).query(V[far], k=1, workers=-1)
            colors[far] = np.asarray(grid.colors)[nb]
    out.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    coverage = float(seen.mean())
    log(f"  photo coverage: {coverage:.1%} of points seen directly")
    return out, {"coverage": coverage, "views": per_view, "spacing": spacing, "params": p.to_dict()}


# --------------------------------------------------------------------------- UV texture
def _scaled_K(view: CameraView, W: int, H: int) -> np.ndarray:
    K = view.K.copy()
    K[0] *= W / view.width
    K[1] *= H / view.height
    return K


def grid_uvs(n_triangles: int, size: int) -> np.ndarray:
    """Fallback UV parameterisation: every triangle gets its own cell in a square grid (bottom-left uv origin).
    Low texel density and no shape preservation - only used when UVAtlas is unavailable."""
    cells = max(1, math.ceil(math.sqrt(n_triangles)))
    cell = size / cells
    pad = min(1.0, cell * 0.2)  # texels of gap so conservative rasterisation does not leak between cells
    i = np.arange(n_triangles)
    cx, cy = (i % cells) * cell, (i // cells) * cell  # texel coords, top-left origin
    lo, hi = pad, cell - pad
    tri = np.array([[lo, lo], [hi, lo], [lo, hi]])  # right triangle in the cell
    px = cx[:, None] + tri[None, :, 0]
    py = cy[:, None] + tri[None, :, 1]
    return np.stack([px / size, 1.0 - py / size], axis=-1).astype(np.float32)


def _edge_functions(uvs: np.ndarray, size: int):
    """Per triangle, affine functions of texel-space (x, y) (top-left origin): barycentrics (T,3,3) as
    b_i = c[i,0] x + c[i,1] y + c[i,2], plus the same scaled to signed distance to the opposite edge in texels."""
    P = np.empty(uvs.shape, dtype=np.float64)
    P[..., 0] = uvs[..., 0] * size
    P[..., 1] = (1.0 - uvs[..., 1]) * size  # texel (x, y) has its centre at (x + 0.5, y + 0.5)
    bc = np.zeros((len(P), 3, 3))
    dc = np.zeros((len(P), 3, 3))
    area2 = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
    good = np.abs(area2) > 1e-10
    for i in range(3):
        a, b = P[:, (i + 1) % 3], P[:, (i + 2) % 3]
        e = b - a
        coef = np.stack([-e[:, 1], e[:, 0], e[:, 1] * a[:, 0] - e[:, 0] * a[:, 1]], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bc[:, i] = coef / area2[:, None]
            dc[:, i] = coef / (np.linalg.norm(e, axis=1) * np.sign(area2))[:, None]
    bc[~good] = 0
    dc[~good] = 0
    return P, bc, dc, good


def _rasterize_uv(uvs: np.ndarray, size: int, dilate_px: float = 1.0, budget: int = 8_000_000):
    """Assign every texel touched by a triangle (conservatively, within `dilate_px` texels) to the triangle that
    covers it best (largest signed distance inside). Returns (texel flat index, triangle id, clamped barycentrics)."""
    S = size
    P, bc, dc, good = _edge_functions(uvs, S)
    xmin = np.clip(np.floor(P[..., 0].min(1) - dilate_px - 0.5), 0, S - 1).astype(np.int64)
    xmax = np.clip(np.ceil(P[..., 0].max(1) + dilate_px - 0.5), 0, S - 1).astype(np.int64)
    ymin = np.clip(np.floor(P[..., 1].min(1) - dilate_px - 0.5), 0, S - 1).astype(np.int64)
    ymax = np.clip(np.ceil(P[..., 1].max(1) + dilate_px - 0.5), 0, S - 1).astype(np.int64)
    w, h = xmax - xmin + 1, ymax - ymin + 1
    n = np.where(good, w * h, 0)
    dc32 = dc.astype(np.float32)

    best = np.full(S * S, -np.inf, dtype=np.float32)
    owner = np.full(S * S, -1, dtype=np.int32)
    ids = np.flatnonzero(n)
    csum = np.cumsum(n[ids])
    start = 0
    while start < len(ids):
        base = csum[start - 1] if start else 0
        stop = max(int(np.searchsorted(csum, base + budget, side="right")), start + 1)
        t = ids[start:stop]
        cnt = n[t]
        local = np.repeat(np.arange(len(t), dtype=np.int32), cnt)
        k = np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        dy, dx = np.divmod(k, np.repeat(w[t], cnt))
        x = (np.repeat(xmin[t], cnt) + dx).astype(np.float32) + 0.5
        y = (np.repeat(ymin[t], cnt) + dy).astype(np.float32) + 0.5
        del k, dx, dy
        c = dc32[t][local]  # (m,3,3)
        score = np.minimum(np.minimum(c[:, 0, 0] * x + c[:, 0, 1] * y + c[:, 0, 2],
                                      c[:, 1, 0] * x + c[:, 1, 1] * y + c[:, 1, 2]),
                           c[:, 2, 0] * x + c[:, 2, 1] * y + c[:, 2, 2])
        del c
        keep = np.flatnonzero(score >= -dilate_px)
        score = score[keep]
        flat = (y[keep].astype(np.int64)) * S + x[keep].astype(np.int64)
        tri = t[local[keep]].astype(np.int32)
        np.maximum.at(best, flat, score)
        win = score >= best[flat]  # this candidate holds the best score for its texel (so far)
        owner[flat[win]] = tri[win]
        start = stop

    texel = np.flatnonzero(owner >= 0)
    tri = owner[texel]
    del best, owner
    bary = np.empty((len(texel), 3), dtype=np.float32)
    for s in range(0, len(texel), budget):
        sl = slice(s, s + budget)
        ty, tx = np.divmod(texel[sl], S)
        xc, yc = tx + 0.5, ty + 0.5
        c = bc[tri[sl]]
        b = c[:, :, 0] * xc[:, None] + c[:, :, 1] * yc[:, None] + c[:, :, 2]
        b = np.clip(b, 0, None)  # texels just outside the triangle take the nearest surface point
        b /= np.maximum(b.sum(1, keepdims=True), 1e-12)
        bary[sl] = b
    return texel, tri, bary


def _depth_map(scene, K: np.ndarray, R: np.ndarray, C: np.ndarray, W: int, H: int, max_pixels: int = 2_000_000):
    """Ray-hit distance for a (possibly strided) grid of photo pixels. Returns (depth, stride)."""
    stride = max(1, math.ceil(math.sqrt(W * H / max_pixels)))
    xs = np.arange(0, W, stride, dtype=np.float64)
    ys = np.arange(0, H, stride, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    pix = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=1)
    d = (pix @ np.linalg.inv(K).T) @ R
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return _cast(scene, C, d).reshape(len(ys), len(xs)), stride


def bake_albedo_numpy(V: np.ndarray, tris: np.ndarray, N: np.ndarray, uvs: np.ndarray, views: list[CameraView],
                      params: TextureParams, size: int, log: Log = print,
                      scan_colors: np.ndarray | None = None) -> np.ndarray:
    """CPU (numpy/scipy + Open3D raycasting) albedo bake; same visibility and weighting rules as
    project_vertex_colors. Returns a (size, size, 3) uint8 image, row 0 = top (uv v = 1)."""
    from scipy.ndimage import distance_transform_edt

    import time
    p = params
    t0 = time.perf_counter()
    V = np.asarray(V, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    texel, tri, bary = _rasterize_uv(np.asarray(uvs, dtype=np.float64), size)
    log(f"  rasterised UV atlas: {len(texel):,} texels ({time.perf_counter() - t0:.1f}s)")
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    tol = max(p.visibility_tolerance * diag, 1e-9)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(tris.astype(np.uint32)))

    n = len(texel)
    chunk = 4_000_000
    # surface points and interpolated normals per texel
    pos = np.empty((n, 3), dtype=np.float32)
    nrm = np.empty((n, 3), dtype=np.float32)
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        tv = tris[tri[sl]]
        b = bary[sl].astype(np.float64)
        pos[sl] = np.einsum("nk,nkj->nj", b, V[tv])
        nn = np.einsum("nk,nkj->nj", b, N[tv])
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        nrm[sl] = nn

    acc = np.zeros((n, 3), dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)
    m = p.edge_margin_px
    for i, view in enumerate(views):
        t1 = time.perf_counter()
        img = load_image(view.image_path)
        H, W = img.shape[:2]
        K = _scaled_K(view, W, H)
        R, t = view.world_to_camera[:3, :3], view.world_to_camera[:3, 3]
        C = view.center
        depth, stride = _depth_map(scene, K, R, C, W, H) if m > 0 else (None, 1)
        dh, dw = depth.shape if depth is not None else (0, 0)
        used = 0
        for s in range(0, n, chunk):
            P = pos[s:s + chunk].astype(np.float64)
            Pc = P @ R.T + t
            z = Pc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = K[0, 0] * Pc[:, 0] / z + K[0, 1] * Pc[:, 1] / z + K[0, 2]
                v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
            to_cam = C - P
            dist = np.linalg.norm(to_cam, axis=1)
            cos = np.einsum("ij,ij->i", nrm[s:s + chunk], to_cam / dist[:, None])
            cand = (z > 0) & (u >= m) & (u <= W - 1 - m) & (v >= m) & (v <= H - 1 - m) & (cos > p.min_cos)
            idx = np.flatnonzero(cand)
            if len(idx):
                dirs = -(to_cam[idx] / dist[idx, None])
                idx = idx[_cast(scene, C, dirs) >= dist[idx] - tol]
            if len(idx) and depth is not None:  # reject texels near depth discontinuities (silhouettes)
                ok = np.ones(len(idx), bool)
                for du, dv in ((m, 0), (-m, 0), (0, m), (0, -m)):
                    gx = np.clip(np.rint((u[idx] + du) / stride).astype(np.int64), 0, dw - 1)
                    gy = np.clip(np.rint((v[idx] + dv) / stride).astype(np.int64), 0, dh - 1)
                    hit = depth[gy, gx]
                    ok &= np.isfinite(hit) & (np.abs(hit - dist[idx]) < 0.05 * diag)
                idx = idx[ok]
            if len(idx):
                w = np.clip(cos[idx], 0, 1) ** p.blend_power
                acc[s + idx] += (_bilinear(img, u[idx], v[idx]) * w[:, None]).astype(np.float32)
                wsum[s + idx] += w.astype(np.float32)
                used += len(idx)
        log(f"  photo {i + 1} ({Path(view.image_path).name}): colours {used:,} texels "
            f"({time.perf_counter() - t1:.1f}s)")
    del pos, nrm

    seen = wsum > 0
    colors = np.zeros((n, 3), dtype=np.float32)
    colors[seen] = acc[seen] / wsum[seen, None]
    del acc, wsum
    log(f"  texture coverage: {seen.mean():.1%} of texels seen directly")
    unseen = ~seen
    if unseen.any():
        nv = len(V)
        vcol = None
        if p.unseen == "scan" and scan_colors is not None and len(scan_colors) == nv:
            vcol = np.asarray(scan_colors, dtype=np.float64)
        elif p.unseen == "diffuse" and seen.any():
            # per-vertex colour from the texels seen around it, grown over the mesh, then interpolated back
            st = tri[seen]
            sb = bary[seen].astype(np.float64)
            vid = tris[st].ravel()
            wv = sb.ravel()
            vw = np.bincount(vid, weights=wv, minlength=nv)
            vcol = np.zeros((nv, 3))
            for ch in range(3):
                vcol[:, ch] = np.bincount(vid, weights=wv * np.repeat(colors[seen, ch], 3), minlength=nv)
            known = vw > 0
            vcol[known] /= vw[known, None]
            vcol, known = _diffuse_fill(tris, vcol, known)
            if (~known).any():
                _, nnb = cKDTree(V[known]).query(V[~known], k=1, workers=-1)
                vcol[~known] = vcol[known][nnb]
        ui = np.flatnonzero(unseen)
        if vcol is None:
            colors[ui] = 0.6
        else:
            colors[ui] = np.einsum("nk,nkj->nj", bary[ui].astype(np.float64), vcol[tris[tri[ui]]])

    S = size
    tex = np.zeros((S, S, 3), dtype=np.uint8)
    flat = tex.reshape(-1, 3)
    flat[texel] = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    # dilate into empty atlas space so mip-mapping / bilinear lookups at chart borders don't pick up black
    empty = np.ones(S * S, dtype=bool)
    empty[texel] = False
    empty = empty.reshape(S, S)
    if empty.any() and len(texel):
        radius = max(2, S // 512)
        dist, (iy, ix) = distance_transform_edt(empty, return_indices=True)
        fill = empty & (dist <= radius)
        tex[fill] = tex[iy[fill], ix[fill]]
    log(f"  numpy texture bake done ({time.perf_counter() - t0:.1f}s)")
    return tex


def _texture_baker() -> str:
    import os
    b = os.environ.get("CLOUDCLEAN_TEXTURE_BAKER", "auto").strip().lower()
    return b if b in ("native", "numpy") else "auto"


def bake_texture(mesh: o3d.geometry.TriangleMesh, views: list[CameraView],
                 params: TextureParams, log: Log = print) -> dict:
    """UV-unwrap the mesh and bake the photos into an albedo texture. Uses Open3D's native
    project_images_to_albedo when available (x86_64) and the numpy baker otherwise; the env var
    CLOUDCLEAN_TEXTURE_BAKER=native|numpy forces one."""
    work = o3d.geometry.TriangleMesh(mesh)
    if len(work.triangles) > params.texture_max_triangles:
        work = work.simplify_quadric_decimation(target_number_of_triangles=params.texture_max_triangles)
        work = cleanup_mesh(work, 0.0, log)
        log(f"  decimated to {len(work.triangles):,} triangles for UV unwrapping")
    work.compute_vertex_normals()
    tm = o3d.t.geometry.TriangleMesh.from_legacy(work)
    baker = _texture_baker()
    log(f"  computing UV atlas ({params.texture_size}px) - this can take a while")
    try:
        tm.compute_uvatlas(size=params.texture_size, parallel_partitions=4)
        uvs = tm.triangle.texture_uvs.numpy()
    except Exception as exc:
        if baker == "native":
            raise
        log(f"  UV atlas unavailable ({exc}); using per-triangle grid charts (lower texel density)")
        uvs = grid_uvs(len(work.triangles), params.texture_size)
        baker = "numpy"

    tex = None
    if baker in ("auto", "native"):
        try:
            images, Ks, Ts = [], [], []
            for view in views:
                img = load_image(view.image_path)
                H, W = img.shape[:2]
                images.append(o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray((img * 255).astype(np.uint8)))))
                Ks.append(o3d.core.Tensor(_scaled_K(view, W, H)))
                Ts.append(o3d.core.Tensor(view.world_to_camera))
            log("  projecting photos into texture")
            tex = tm.project_images_to_albedo(images, Ks, Ts, tex_size=params.texture_size, update_material=False)
            tex = np.asarray(tex.as_tensor().numpy())
            if tex.dtype != np.uint8:
                tex = (np.clip(tex, 0, 1) * 255).astype(np.uint8) if tex.max() <= 1.0 else tex.astype(np.uint8)
            tex = tex[..., :3]
        except Exception as exc:
            if baker == "native":
                raise
            log(f"  native texture projection unavailable ({exc}); using the numpy baker")
            tex = None
    if tex is None:
        log("  projecting photos into texture (numpy baker)")
        scan = np.asarray(work.vertex_colors) if work.has_vertex_colors() else None
        tex = bake_albedo_numpy(np.asarray(work.vertices), np.asarray(work.triangles),
                                np.asarray(work.vertex_normals), uvs, views, params, params.texture_size,
                                log, scan_colors=scan)
    return {"mesh": work, "uvs": np.asarray(uvs, dtype=np.float32), "image": np.ascontiguousarray(tex)}


def save_textured(textured: dict, path) -> list[Path]:
    """Write a UV-textured mesh as .glb (single file) or .obj (+ .mtl + .png)."""
    import io as _io

    from .io import write_glb

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh, uvs, image = textured["mesh"], textured["uvs"], textured["image"]
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)[tris].reshape(-1, 3)
    normals = np.asarray(mesh.vertex_normals)[tris].reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    uv = uvs.reshape(-1, 2)  # Open3D uv origin is bottom-left (same as OBJ)
    if path.suffix.lower() == ".obj":
        png, mtl = path.with_suffix(".png"), path.with_suffix(".mtl")
        Image.fromarray(image).save(png)
        mtl.write_text(f"newmtl material0\nKa 0 0 0\nKd 1 1 1\nKs 0 0 0\nmap_Kd {png.name}\n")
        with open(path, "w") as f:
            f.write(f"mtllib {mtl.name}\nusemtl material0\n")
            np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, uv, fmt="vt %.6f %.6f")
            np.savetxt(f, normals, fmt="vn %.5f %.5f %.5f")
            np.savetxt(f, np.repeat(faces + 1, 3, axis=1), fmt="f %d/%d/%d %d/%d/%d %d/%d/%d")
        return [path, mtl, png]
    png_bytes = _io.BytesIO()
    Image.fromarray(image).save(png_bytes, "PNG")
    uv_gltf = np.c_[uv[:, 0], 1.0 - uv[:, 1]]  # glTF uv origin is top-left
    write_glb(path, verts, faces, normals=normals, uvs=uv_gltf, texture_png=png_bytes.getvalue())
    return [path]


def colorize_mesh(mesh: o3d.geometry.TriangleMesh, views: list[CameraView],
                  params: TextureParams | None = None, log: Log = print):
    """Returns (vertex_coloured_mesh, report, textured_or_None)."""
    p = params or TextureParams()
    if not views:
        raise ValueError("At least one photo with a camera pose is required")
    log(f"Colouring mesh ({len(mesh.vertices):,} vertices) from {len(views)} photo(s)")
    colored, report = project_vertex_colors(mesh, views, p, log)
    textured = None
    if p.texture_size > 0:
        try:
            textured = bake_texture(mesh, views, p, log)
            report["texture_size"] = p.texture_size
        except Exception as exc:
            report["texture_error"] = str(exc)
            log(f"  UV texture baking failed ({exc}); vertex colours are still available")
    report["params"] = p.to_dict()
    return colored, report, textured
