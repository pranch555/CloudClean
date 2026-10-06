"""Synthetic test photos for the outline check (cloudclean/outline_check.py, docs/outline-check.md).

A part lying on the printed photo check sheet (scale_sheet style "check"), backlit on a light pad, photographed by a
phone-like camera: the paper glows, the black markers stay dark, the part is a dark silhouette. Everything that
decides where an edge lands in the photo is modelled exactly, because sub-pixel edge bias is what the check lives
on:

- the camera: OpenCV's pinhole model with distortion (the same model the check uses), pixel centres at whole numbers;
- the print: scale_sheet.raster at 20 px/mm (marker edges on raster pixel edges), scaled by the printer (sx, sy);
- anti-aliasing: every pixel is the mean of ss x ss jittered samples (stratified), each sample traced exactly: the
  paper by its homography through the lens, the part by ray casting (Open3D);
- optics and sensor: a Gaussian blur (symmetric), vignetting and an uneven light pad, shot and read noise, all in
  linear light, then the sRGB curve, 8 bits and JPEG, with EXIF (make, model, lens, focal length, orientation).

Procedural parts: bolt() (a revolved head and shank, like an M24 socket head bolt without the socket and thread),
plate() (an L-shaped plate with a through hole) and screw() (a knurled, chamfered head on a right-hand threaded
shank), all closed, welded meshes like a CAD tessellation. turned_errors() puts known errors into any round part's
mesh (a real golden screw) by moving its vertices along its own axis: the head's end, the other end, the radius
below the head.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import open3d as o3d

from . import scale_sheet
from .outline_camera import Camera, _undistort, linear_to_srgb

RASTER_PPM = 20        # the print's raster (px per mm): whole numbers keep the marker edges on pixel edges
BOLT = {"head_d": 36.0, "head_h": 25.25, "shank_d": 24.0, "under": 81.85}
# the L-shaped plate: length x width, narrowed to step_y wide beyond x = step_x, a round hole, thickness
PLATE = {"length": 70.0, "width": 40.0, "step_x": 45.0, "step_y": 28.0, "hole_x": 20.0, "hole_y": 18.0,
         "hole_d": 14.0, "thickness": 12.0}


# --------------------------------------------------------------------------- parts
def _mesh(V: np.ndarray, F: np.ndarray) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(V, np.float64)),
                                  o3d.utility.Vector3iVector(np.asarray(F, np.int32)))
    m.compute_triangle_normals()
    return m


def revolve_solid(profile, segments: int = 360) -> o3d.geometry.TriangleMesh:
    """A closed solid made by turning a (radius, z) outline around Z. The outline starts and ends on the axis
    (radius 0); in between, every corner becomes a ring of `segments` vertices."""
    prof = [(float(r), float(z)) for r, z in profile]
    assert prof[0][0] == 0 and prof[-1][0] == 0
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = prof[1:-1]
    V = [np.c_[r * np.cos(a), r * np.sin(a), np.full(segments, z)] for r, z in rings]
    V = np.vstack(V + [np.array([[0, 0, prof[0][1]], [0, 0, prof[-1][1]]])])
    bottom, top = len(V) - 2, len(V) - 1
    idx = [np.arange(segments) + k * segments for k in range(len(rings))]
    F = [np.c_[np.full(segments, bottom), np.roll(idx[0], -1), idx[0]]]
    for k in range(len(rings) - 1):
        lo, hi = idx[k], idx[k + 1]
        F += [np.c_[lo, np.roll(lo, -1), np.roll(hi, -1)], np.c_[lo, np.roll(hi, -1), hi]]
    F.append(np.c_[np.full(segments, top), idx[-1], np.roll(idx[-1], -1)])
    F = np.vstack(F)
    # outward normals: the profile runs bottom -> out -> up; check the signed volume
    tri = V[F]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    if vol < 0:
        F = F[:, ::-1]
    return _mesh(V, F)


def bolt(head_d: float = BOLT["head_d"], head_h: float = BOLT["head_h"], shank_d: float = BOLT["shank_d"],
         under: float = BOLT["under"], segments: int = 360) -> o3d.geometry.TriangleMesh:
    """A bolt without thread: head (Ø head_d x head_h, its top at z = 0) and shank (Ø shank_d, `under` long below
    the head), along +Z."""
    rh, rs = head_d / 2, shank_d / 2
    return revolve_solid([(0, 0), (rh, 0), (rh, head_h), (rs, head_h), (rs, head_h + under), (0, head_h + under)],
                         segments)


def _polygon_area(P: np.ndarray) -> float:
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def plate(length: float = PLATE["length"], width: float = PLATE["width"], step_x: float = PLATE["step_x"],
          step_y: float = PLATE["step_y"], hole_x: float = PLATE["hole_x"], hole_y: float = PLATE["hole_y"],
          hole_d: float = PLATE["hole_d"], thickness: float = PLATE["thickness"],
          segments: int = 256) -> o3d.geometry.TriangleMesh:
    """An L-shaped plate lying flat: length x width, only step_y wide beyond x = step_x, with a round through hole
    (hole_x, hole_y, Ø hole_d), thickness high."""
    outline = [(0.0, 0.0), (length, 0.0), (length, step_y), (step_x, step_y), (step_x, width), (0.0, width)]
    return extrude_with_hole(outline, (hole_x, hole_y, hole_d / 2), thickness, segments)


def extrude_with_hole(outline, hole, thickness: float, segments: int = 256) -> o3d.geometry.TriangleMesh:
    """A flat part: the outline (x, y corners) extruded `thickness` up from z = 0, with a round through hole
    (cx, cy, radius). The outline must be star-shaped from the hole's centre (every corner visible from it): the
    caps are made of strips between the hole's circle and the outline, ray by ray from the centre."""
    P = np.asarray(outline, np.float64)
    if _polygon_area(P) < 0:
        P = P[::-1]
    cx, cy, r = hole
    c = np.array([cx, cy])
    # angles of the outline's corners seen from the hole centre, plus a regular fan
    ang_c = np.mod(np.arctan2(P[:, 1] - cy, P[:, 0] - cx), 2 * np.pi)
    ang = np.unique(np.round(np.r_[np.linspace(0, 2 * np.pi, segments, endpoint=False), ang_c], 12))
    ang = np.sort(np.mod(ang, 2 * np.pi))
    keep = np.r_[True, np.diff(ang) > 1e-9]
    ang = ang[keep]
    d = np.c_[np.cos(ang), np.sin(ang)]
    outer = []
    for di in d:   # where the ray from the centre leaves the outline
        best = np.inf
        for k in range(len(P)):
            a, b = P[k], P[(k + 1) % len(P)]
            M = np.array([di, a - b]).T
            try:
                s, u = np.linalg.solve(M, a - c)
            except np.linalg.LinAlgError:
                continue
            if s > 1e-9 and -1e-9 <= u <= 1 + 1e-9:
                best = min(best, s)
        outer.append(c + best * di)
    outer = np.array(outer)
    inner = c + r * d
    n = len(ang)
    V = np.vstack([np.c_[outer, np.zeros(n)], np.c_[inner, np.zeros(n)],
                   np.c_[outer, np.full(n, thickness)], np.c_[inner, np.full(n, thickness)]])
    o0, i0, o1, i1 = 0, n, 2 * n, 3 * n
    k = np.arange(n)
    k1 = np.roll(k, -1)
    F = [np.c_[o0 + k, i0 + k1, o0 + k1], np.c_[o0 + k, i0 + k, i0 + k1],        # bottom (facing -z)
         np.c_[o1 + k, o1 + k1, i1 + k1], np.c_[o1 + k, i1 + k1, i1 + k],        # top
         np.c_[o0 + k, o0 + k1, o1 + k1], np.c_[o0 + k, o1 + k1, o1 + k],        # outer wall
         np.c_[i0 + k, i1 + k1, i0 + k1], np.c_[i0 + k, i1 + k, i1 + k1]]        # hole wall
    mesh = _mesh(V, np.vstack(F))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_triangle_normals()
    return mesh


def place(mesh: o3d.geometry.TriangleMesh, down, x: float, y: float, yaw_deg: float) -> o3d.geometry.TriangleMesh:
    """The part lying on the sheet as it would physically rest: on the convex-hull face (with the centre of mass
    above it) whose outward normal is closest to `down` (a direction in the part's frame), turned yaw_deg about the
    vertical, the middle of its footprint at (x, y), its lowest point at z = 0."""
    from scipy.spatial import ConvexHull

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    com = ((a + b + c) / 4.0 * vol[:, None]).sum(0) / vol.sum()
    hull = ConvexHull(V)
    down = np.asarray(down, np.float64) / np.linalg.norm(down)
    eq = hull.equations
    best, best_dot = None, -2.0
    for k in np.argsort(-(eq[:, :3] @ down)):
        n, d = eq[k, :3], eq[k, 3]
        same = np.flatnonzero((np.abs(eq[:, :3] - n).max(1) < 1e-7) & (np.abs(eq[:, 3] - d) < 1e-6))
        P = V[np.unique(hull.simplices[same])]
        q = com - (n @ com + d) * n
        u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        try:
            h2 = ConvexHull(np.c_[P @ u, P @ v])
        except Exception:
            continue
        if np.max(h2.equations[:, :2] @ [q @ u, q @ v] + h2.equations[:, 2]) < 0:     # centre of mass above it
            best = n
            break
    if best is None:
        raise ValueError("no stable resting face")
    n = best
    target = np.array([0, 0, -1.0])
    v = np.cross(n, target)
    s, c = np.linalg.norm(v), float(n @ target)
    if s < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / s ** 2
    a = math.radians(yaw_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    W = V @ (Rz @ R).T
    W[:, 2] -= W[:, 2].min()
    W[:, :2] += np.array([x, y]) - (W[:, :2].max(0) + W[:, :2].min(0)) / 2
    out = o3d.geometry.TriangleMesh(mesh)
    out.vertices = o3d.utility.Vector3dVector(W)
    out.compute_triangle_normals()
    return out


# --------------------------------------------------------------------------- cameras
def phone_camera(width: int = 4032, height: int = 3024, hfov_deg: float = 69.0, dist=(0.0, 0.0, 0.0, 0.0, 0.0),
                 principal_offset=(0.0, 0.0)) -> Camera:
    """A phone's main camera (sensor layout, landscape): horizontal field of view, distortion, principal point
    offset from the middle (px)."""
    f = width / 2 / math.tan(math.radians(hfov_deg) / 2)
    K = np.array([[f, 0, (width - 1) / 2 + principal_offset[0]], [0, f, (height - 1) / 2 + principal_offset[1]],
                  [0, 0, 1.0]])
    return Camera(K, np.asarray(dist, np.float64), (width, height),
                  {"make": "Synthetic", "model": "iPhone-like 1x", "lens": "main camera 6.8 mm"}, "synthetic")


def view(distance: float = 280.0, tilt_deg: float = 0.0, azimuth_deg: float = 0.0, roll_deg: float = 0.0,
         target=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """A camera `distance` mm from `target` on the sheet, its axis tilted tilt_deg from straight down (coming from
    the side azimuth_deg: 0 = from +x), held so the photo's long side runs along the sheet's y (a phone held upright
    over the page), then rolled roll_deg. Returns (R, t): sheet -> camera."""
    T = np.asarray(target, np.float64)
    th, az = math.radians(tilt_deg), math.radians(azimuth_deg)
    C = T + distance * np.array([math.sin(th) * math.cos(az), math.sin(th) * math.sin(az), math.cos(th)])
    z = (T - C) / np.linalg.norm(T - C)
    up = np.array([0.0, 1.0, 0.0])
    x = up - (up @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = math.radians(roll_deg)
    x, y = math.cos(r) * x + math.sin(r) * y, -math.sin(r) * x + math.cos(r) * y
    R = np.stack([x, y, z])
    return R, -R @ C


# --------------------------------------------------------------------------- rendering
class _Print:
    """The printed sheet: transmission (0.03 black toner .. 1 paper) of each point of the page, and the light pad
    under it."""

    def __init__(self, paper: str = "a4", scale=(1.0, 1.0)):
        self.w, self.h, _ = scale_sheet.PAPERS[paper]
        img = scale_sheet.raster(paper, RASTER_PPM, style="check").astype(np.float32) / 255.0
        self.T = (0.03 + 0.97 * img).astype(np.float32)
        self.sx, self.sy = scale

    def transmission(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        import cv2

        shape = X.shape
        X2 = X.reshape(shape[0] * shape[1], -1) if X.ndim > 2 else X.reshape(len(X), -1)
        Y2 = Y.reshape(X2.shape)
        u, v = X2 / self.sx, Y2 / self.sy
        col = ((u + self.w / 2) * RASTER_PPM - 0.5).astype(np.float32)
        row = ((self.h / 2 - v) * RASTER_PPM - 0.5).astype(np.float32)
        out = cv2.remap(self.T, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=-1.0).reshape(shape)
        on_paper = out >= 0
        pad = (np.abs(X) < 190) & (np.abs(Y) < 250)
        return np.where(on_paper, out, np.where(pad, 1.35, 0.04))


class _IdealGrid:
    """Ideal (undistorted) normalized coordinates of a camera's pixels: computed exactly every `step` px and
    interpolated bilinearly in between (the lens distortion is smooth: the interpolation error is ~1e-4 px)."""

    def __init__(self, camera: Camera, step: int = 4):
        W, H = camera.size
        self.step = step
        self.u = np.arange(-2 * step, W + 2 * step + 1, step, dtype=np.float64)
        self.v = np.arange(-2 * step, H + 2 * step + 1, step, dtype=np.float64)
        uu, vv = np.meshgrid(self.u, self.v)
        xyd = np.stack([(uu - camera.K[0, 2]) / camera.K[0, 0], (vv - camera.K[1, 2]) / camera.K[1, 1]], -1)
        self.xy = _undistort(xyd, camera.dist)

    def __call__(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fu = (u - self.u[0]) / self.step
        fv = (v - self.v[0]) / self.step
        iu = np.clip(np.floor(fu).astype(np.int64), 0, len(self.u) - 2)
        iv = np.clip(np.floor(fv).astype(np.int64), 0, len(self.v) - 2)
        au, av = fu - iu, fv - iv
        g = self.xy
        out = []
        for k in range(2):
            c00, c01 = g[iv, iu, k], g[iv, iu + 1, k]
            c10, c11 = g[iv + 1, iu, k], g[iv + 1, iu + 1, k]
            out.append((c00 * (1 - au) + c01 * au) * (1 - av) + (c10 * (1 - au) + c11 * au) * av)
        return out[0], out[1]


def render(camera: Camera, R: np.ndarray, t: np.ndarray, part: o3d.geometry.TriangleMesh | None = None,
           paper: str = "a4", printed=(1.0, 1.0), exposure: float = 0.78, part_level: float = 0.02,
           blur_px: float = 0.8, ss: int = 4, noise: float = 1.0, pad_falloff: float = 0.06,
           vignette: float = 0.12, seed: int = 0, band_rows: int = 64, light=None,
           ambient: float = 0.12) -> np.ndarray:
    """One photo as linear light (float32, H x W, sensor layout). exposure: the paper's level in the middle of the
    page (1 = the sensor's full scale: above 1 clips). part_level: the part's brightness relative to the paper
    (dark silhouette). noise: 1 = a phone sensor in good light (shot + read noise), 0 = none. Every pixel is the
    mean of ss x ss jittered samples.

    light: None - backlit (a light pad under the paper); or a point light's position (sheet frame, mm) - front
    lighting, like the MetroY's fill light beside its cameras: the paper and its print reflect it (falling off with
    distance and angle), the part (part_level is then its albedo) is shaded by it, and the part casts a shadow on
    the paper; `ambient` is the share of light from the room."""
    W, H = camera.size
    rng = np.random.default_rng(seed)
    prn = _Print(paper, printed)
    ideal = _IdealGrid(camera)
    C = -R.T @ t
    Lp = None if light is None else np.asarray(light, np.float64)
    if Lp is not None:
        lit0 = Lp[2] / np.linalg.norm(Lp - [0.0, 0.0, 0.0]) ** 3     # the paper's direct light at the page middle
    # normalized ideal (x, y, 1) -> sheet (X, Y, 1): the inverse of the sheet's homography [r1 r2 t]
    Hn = np.linalg.inv(np.c_[R[:, 0], R[:, 1], t])
    scene = None
    if part is not None:
        V = np.asarray(part.vertices)
        shift = V.mean(0)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor((V - shift).astype(np.float32)),
                            o3d.core.Tensor(np.asarray(part.triangles).astype(np.uint32)))
        uv = camera.project(V @ R.T + t)
        u0, v0 = np.floor(uv.min(0)) - 4
        u1, v1 = np.ceil(uv.max(0)) + 4
    out = np.zeros((H, W), np.float32)
    grid = np.arange(ss) / ss - 0.5
    for r0 in range(0, H, band_rows):
        r1 = min(H, r0 + band_rows)
        n = r1 - r0
        # stratified jittered samples: (rows, ss, cols, ss)
        vv = (np.arange(r0, r1)[:, None, None, None] + grid[None, :, None, None]
              + rng.random((n, ss, W, ss), dtype=np.float32) / ss)
        uu = (np.arange(W)[None, None, :, None] + grid[None, None, None, :]
              + rng.random((n, ss, W, ss), dtype=np.float32) / ss)
        x, y = ideal(uu, vv)
        den = Hn[2, 0] * x + Hn[2, 1] * y + Hn[2, 2]
        X = (Hn[0, 0] * x + Hn[0, 1] * y + Hn[0, 2]) / den
        Y = (Hn[1, 0] * x + Hn[1, 1] * y + Hn[1, 2]) / den
        L = prn.transmission(X.astype(np.float32), Y.astype(np.float32)).astype(np.float32)
        if Lp is None:
            L *= (1.0 - pad_falloff * ((X / 120.0) ** 2 + (Y / 150.0) ** 2)).astype(np.float32)
        else:                                              # front light: the paper reflects it
            dx, dy = Lp[0] - X, Lp[1] - Y
            direct = Lp[2] / np.power(dx * dx + dy * dy + Lp[2] ** 2, 1.5) / lit0
            L *= (ambient + (1 - ambient) * direct).astype(np.float32)
        if scene is not None and r1 > v0 and r0 < v1:
            sel = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)
            if sel.any():
                dc = np.stack([x[sel], y[sel], np.ones(int(sel.sum()))], 1)
                dd = dc @ R                                    # camera -> sheet frame: R^T d
                dd /= np.linalg.norm(dd, axis=1, keepdims=True)
                rays = np.concatenate([np.broadcast_to(C - shift, dd.shape), dd], 1).astype(np.float32)
                res = scene.cast_rays(o3d.core.Tensor(rays))
                th = res["t_hit"].numpy()
                hit = np.isfinite(th)
                Ls = L[sel]
                if Lp is None:
                    Ls[hit] = part_level
                else:
                    # the part, shaded by the front light (Lambert, its normal turned towards the camera)
                    P = C + dd[hit] * th[hit, None]
                    nrm = res["primitive_normals"].numpy()[hit].astype(np.float64)
                    nrm[np.einsum("ij,ij->i", nrm, dd[hit]) > 0] *= -1
                    to_l = Lp - P
                    dist = np.linalg.norm(to_l, axis=1)
                    cosl = np.clip(np.einsum("ij,ij->i", nrm, to_l) / dist, 0, None)
                    Ls[hit] = part_level * (ambient + (1 - ambient) * cosl / dist ** 2 / lit0)
                    # the paper in the part's shadow: rays from the paper towards the light that hit the part
                    paper = ~hit
                    if paper.any():
                        Pp = C + dd[paper] * (-C[2] / dd[paper, 2])[:, None]
                        to = Lp - Pp
                        to /= np.linalg.norm(to, axis=1, keepdims=True)
                        srays = np.concatenate([Pp + to * 0.01 - shift, to], 1).astype(np.float32)
                        shade = np.isfinite(scene.cast_rays(o3d.core.Tensor(srays))["t_hit"].numpy())
                        lp = Ls[paper]
                        dxs, dys = Lp[0] - Pp[shade, 0], Lp[1] - Pp[shade, 1]
                        direct_s = Lp[2] / np.power(dxs * dxs + dys * dys + Lp[2] ** 2, 1.5) / lit0
                        # only the room's light reaches it
                        lp[shade] = lp[shade] * ambient / (ambient + (1 - ambient) * direct_s)
                        Ls[paper] = lp
                L[sel] = Ls
        L *= (1.0 - vignette * (x * x + y * y) / 0.5).astype(np.float32)
        out[r0:r1] = L.mean(axis=(1, 3))
    img = out * exposure
    if blur_px > 0:
        import cv2

        img = cv2.GaussianBlur(img, (0, 0), blur_px, borderType=cv2.BORDER_REFLECT)
    if noise > 0:
        full_well = 2500.0 / noise ** 2          # electrons at full scale
        sig = np.sqrt(np.clip(img, 0, None) / full_well + (0.0025 * noise) ** 2)
        img = img + rng.normal(size=img.shape).astype(np.float32) * sig.astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def save_photo(path, linear: np.ndarray, orientation: int = 8, meta: dict | None = None, quality: int = 95) -> Path:
    """Linear light -> sRGB 8-bit RGB (paper slightly warm) -> JPEG / PNG / HEIC with EXIF. The pixels are stored
    in the sensor layout; orientation says how to show them (8: the phone was held upright over the page)."""
    from PIL import Image

    path = Path(path)
    rgb = np.stack([linear * 1.0, linear * 0.985, linear * 0.95], -1)
    enc = np.clip(np.round(linear_to_srgb(rgb) * 255), 0, 255).astype(np.uint8)
    im = Image.fromarray(enc, "RGB")
    exif = Image.Exif()
    meta = meta or {}
    exif[0x010F] = meta.get("make", "Synthetic")
    exif[0x0110] = meta.get("model", "iPhone-like 1x")
    exif[0x0112] = orientation
    sub = exif.get_ifd(0x8769)
    sub[0xA434] = meta.get("lens", "main camera 6.8 mm")
    if meta.get("focal_mm"):
        sub[0x920A] = float(meta["focal_mm"])
    if meta.get("focal35_mm"):
        sub[0xA405] = int(round(meta["focal35_mm"]))
    ext = path.suffix.lower()
    if ext in (".heic", ".heif"):
        from pillow_heif import register_heif_opener

        register_heif_opener()
        im.save(path, format="HEIF", quality=quality, exif=exif.tobytes())
    elif ext == ".png":
        im.save(path, exif=exif.tobytes())
    else:
        im.save(path, quality=quality, subsampling=0, exif=exif.tobytes())
    return path


def camera_meta(camera: Camera) -> dict:
    """EXIF fields a phone would write for this camera (35 mm equivalent from the diagonal)."""
    W, H = camera.size
    f35 = camera.f * math.hypot(36, 24) / math.hypot(W, H)
    return {**camera.key, "focal_mm": 6.86, "focal35_mm": f35}


# --------------------------------------------------------------------------- a screw with a knurl and a thread
SCREW = {"head_d": 30.0, "head_h": 15.0, "major_d": 20.0, "pitch": 3.0, "under": 45.0, "end_chamfer": 1.5,
         "head_chamfer": 1.0, "knurl": 40, "knurl_depth": 0.25}


def _tooth(u: np.ndarray) -> np.ndarray:
    """A thread's tooth across one pitch (u in 0..1): root flat, rising flank, crest flat (1/8), falling flank."""
    u = np.mod(u, 1.0)
    root, crest = 0.25, 0.125
    flank = (1.0 - root - crest) / 2
    return np.clip(np.minimum((u - root) / flank, (1.0 - u) / flank), 0.0, 1.0)


def screw(head_d: float = SCREW["head_d"], head_h: float = SCREW["head_h"], major_d: float = SCREW["major_d"],
          pitch: float = SCREW["pitch"], under: float = SCREW["under"], end_chamfer: float = SCREW["end_chamfer"],
          head_chamfer: float = SCREW["head_chamfer"], knurl: int = SCREW["knurl"],
          knurl_depth: float = SCREW["knurl_depth"], segments: int = 240, rows_per_pitch: int = 12,
          thread_depth: float | None = None) -> o3d.geometry.TriangleMesh:
    """A socket-head-like screw along +Z: a right-hand thread (trapezoid teeth, depth 0.54 pitch unless given, crest
    flat 1/8 pitch) from its chamfered end (z = 0) to the head's underside (z = under), and a head (Ø head_d x
    head_h) with a straight knurl (`knurl` grooves) and a chamfered top. A closed mesh; the radius is a function of
    angle and height."""
    Rm, Rh = major_d / 2, head_d / 2
    depth = 0.54 * pitch if thread_depth is None else thread_depth
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    t_sh = np.linspace(0.0, under, int(round(under / pitch * rows_per_pitch)) + 1)
    t_hd = np.linspace(under, under + head_h, int(round(head_h / 0.5)) + 1)

    def shank(t):
        T, A = np.meshgrid(t, th, indexing="ij")
        r = Rm - depth + depth * _tooth((T - pitch * A / (2 * np.pi)) / pitch)
        return np.minimum(r, Rm - end_chamfer + T)             # the 45 degree chamfer at the end

    def head(t):
        T, A = np.meshgrid(t, th, indexing="ij")
        g = np.abs(np.mod(A * knurl / (2 * np.pi), 1.0) - 0.5)   # 0.5 on a crest's middle, 0 in a groove
        r = Rh - knurl_depth * np.clip(1 - g / 0.3, 0, 1)
        top = under + head_h
        return np.minimum(r, Rh - head_chamfer + (top - T))

    rows = [(t, r) for t, r in zip(t_sh, shank(t_sh))] + [(t, r) for t, r in zip(t_hd, head(t_hd))]
    V = [np.c_[r * np.cos(th), r * np.sin(th), np.full(segments, t)] for t, r in rows]
    V = np.vstack(V + [np.array([[0, 0, 0.0], [0, 0, under + head_h]])])
    bottom, top_c = len(V) - 2, len(V) - 1
    idx = [np.arange(segments) + k * segments for k in range(len(rows))]
    F = [np.c_[np.full(segments, bottom), np.roll(idx[0], -1), idx[0]]]
    for k in range(len(rows) - 1):
        lo, hi = idx[k], idx[k + 1]
        F += [np.c_[lo, np.roll(lo, -1), np.roll(hi, -1)], np.c_[lo, np.roll(hi, -1), hi]]
    F.append(np.c_[np.full(segments, top_c), idx[-1], np.roll(idx[-1], -1)])
    F = np.vstack(F)
    tri = V[F]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        F = F[:, ::-1]
    return _mesh(V, F)


def turned_errors(mesh: o3d.geometry.TriangleMesh, head_top: float = 0.0, end: float = 0.0,
                  radial: float = 0.0) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """A round part (a screw) with known errors made by moving its vertices along its own axis (found as the outline
    check finds it): head_top: the head's end region (its top face and top fillet / chamfer) moved out along the
    axis (the head is that much taller); end: the other end region (its end face and chamfer) moved out (negative:
    the part is shorter); radial: everything below the head's underside moved out radially (a thread's major and
    minor diameters grow by twice that). Returns (mesh, the profile it was cut along)."""
    from .compare import ReferenceSurface
    from .outline_check import turned_profile

    ref = ReferenceSurface(mesh, log=lambda m: None)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor((ref.vertices - ref.center).astype(np.float32)),
                        o3d.core.Tensor(ref.triangles.astype(np.uint32)))
    pr = turned_profile(ref, scene, log=lambda m: None)
    if pr is None or pr["head"] is None:
        raise ValueError("turned_errors needs a round part with a head")
    a, o = pr["a"], pr["o"]
    zones = pr["zones"]
    hz = zones[pr["head"]]
    up = hz["tc"] > (pr["t0"] + pr["t1"]) / 2                 # the head is at the +t end
    V = np.asarray(mesh.vertices).copy()
    t = (V - o) @ a
    rel = V - o - np.outer(t, a)
    r = np.linalg.norm(rel, axis=1)
    other = zones[0] if up else zones[-1]
    s = 1.0 if up else -1.0
    head_region = t > hz["t1"] + 1e-6 if up else t < hz["t0"] - 1e-6
    end_region = t < other["t0"] - 1e-6 if up else t > other["t1"] + 1e-6
    under = hz["t0"] if up else hz["t1"]
    shank = t < under - 0.2 if up else t > under + 0.2
    V[head_region] += head_top * s * a
    V[end_region] += end * (-s) * a
    V[shank] += radial * rel[shank] / np.maximum(r[shank], 1e-9)[:, None]
    out = o3d.geometry.TriangleMesh(mesh)
    out.vertices = o3d.utility.Vector3dVector(V)
    out.compute_triangle_normals()
    return out, pr


def _smooth(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def thread_errors(mesh: o3d.geometry.TriangleMesh, profile: dict | None = None, shallow=None, shift=None,
                  stretch: float = 0.0, nick=None, seam_deg: float = 0.0,
                  ramp_deg: float = 25.0) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Known errors in one thread's teeth, made by moving the vertices of a round part's mesh (a real golden screw)
    in its helix's own coordinates (found as the outline check finds them: turned_profile; or `profile`). A helix
    turn runs from the angle seam_deg about the axis once round; each error fades in and out over ramp_deg at the
    seam (put the seam where the camera does not see the outline, e.g. facing the camera).

    shallow = (k, mm): groove k (its root at s = k, s = (t - root_t0) / pitch - hand * angle / 2 pi) filled in mm
    above the minor radius (that much shallower).
    shift = (j, mm): tooth j (between roots j and j + 1) moved mm along the axis (+ = towards +t), its flanks and
    crest; the root flats stay.
    stretch: the thread side of the head's underside stretched by this share along the axis about the underside
    (0.001: a 0.1 % lead error, the far end that much further out).
    nick = (j, angle_deg, mm, half_width_deg): tooth j's crest pushed in by mm within half_width_deg of angle_deg,
    fading to nothing 0.3 mm down the flanks.
    Returns (mesh, info: the profile's thread and what was done)."""
    from .compare import ReferenceSurface
    from .outline_check import turned_profile

    if profile is None:
        ref = ReferenceSurface(mesh, log=lambda m: None)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor((ref.vertices - ref.center).astype(np.float32)),
                            o3d.core.Tensor(ref.triangles.astype(np.uint32)))
        profile = turned_profile(ref, scene, log=lambda m: None)
    if profile is None:
        raise ValueError("thread_errors needs a round part with a thread")
    z = next((zz for zz in profile["zones"] if zz["texture"] == "thread"), None)
    if z is None:
        raise ValueError("thread_errors needs a round part with a thread")
    a, o, u, v = profile["a"], profile["o"], profile["u"], profile["v"]
    p, R, minor, hand = z["pitch"], z["R"], z["minor"], z["hand_sign"]
    V = np.asarray(mesh.vertices).copy()
    t = (V - o) @ a
    rel = V - o - np.outer(t, a)
    rho = np.linalg.norm(rel, axis=1)
    rhat = rel / np.maximum(rho, 1e-12)[:, None]
    th = np.arctan2(rhat @ v, rhat @ u)
    seam = math.radians(seam_deg)
    rep = seam + np.mod(th - seam, 2 * np.pi)                       # each angle once, from the seam round
    s = (t - z["root_t0"]) / p - hand * rep / (2 * np.pi)
    ramp = math.radians(ramp_deg)
    w_seam = _smooth(np.minimum(rep - seam, seam + 2 * np.pi - rep) / ramp)
    zone = (t >= z["t0"]) & (t <= z["t1"])
    info = {"pitch": p, "root_t0": z["root_t0"], "hand_sign": hand, "R": R, "minor": minor, "seam_deg": seam_deg,
            "axis": a.tolist(), "o": o.tolist(), "u": u.tolist(), "v": v.tolist(), "errors": []}
    if shallow is not None:
        k, mm = shallow
        sel = zone & (np.abs(s - k) < 0.5) & (rho < minor + mm)
        new = rho[sel] + w_seam[sel] * (minor + mm - rho[sel])
        V[sel] = o + np.outer(t[sel], a) + new[:, None] * rhat[sel]
        info["errors"].append({"kind": "shallow", "groove": k, "mm": mm, "vertices": int(sel.sum())})
    if shift is not None:
        j, mm = shift
        sel = zone & (s > j + 0.06) & (s < j + 0.94) & (rho > minor + 0.03)
        V[sel] += np.outer(w_seam[sel] * mm, a)
        info["errors"].append({"kind": "shift", "tooth": j, "mm": mm, "vertices": int(sel.sum())})
    if nick is not None:
        j, ang, mm, hw = nick
        d_ang = np.abs(np.angle(np.exp(1j * (th - math.radians(ang)))))
        w_ang = _smooth((math.radians(hw) + ramp - d_ang) / ramp)
        top = np.clip((rho - (R - 0.3)) / 0.3, 0.0, 1.0)
        sel = zone & (s > j) & (s < j + 1) & (top > 0) & (w_ang > 0)
        new = rho[sel] - mm * w_ang[sel] * top[sel]
        V[sel] = o + np.outer(t[sel], a) + new[:, None] * rhat[sel]
        info["errors"].append({"kind": "nick", "tooth": j, "angle_deg": ang, "mm": mm, "half_width_deg": hw,
                               "vertices": int(sel.sum())})
    if stretch:
        hz = profile["zones"][profile["head"]] if profile["head"] is not None else None
        if hz is None:
            raise ValueError("stretch needs a head (it is about the head's underside)")
        up = hz["tc"] > z["tc"]
        under = hz["t0"] if up else hz["t1"]
        tv = (V - o) @ a
        sel = tv < under - 1e-6 if up else tv > under + 1e-6
        V[sel] += np.outer((tv[sel] - under) * stretch, a)
        info["errors"].append({"kind": "stretch", "share": stretch, "about_t": under, "vertices": int(sel.sum())})
    out = o3d.geometry.TriangleMesh(mesh)
    out.vertices = o3d.utility.Vector3dVector(V)
    out.compute_triangle_normals()
    return out, info


def silhouette_edges(part: o3d.geometry.TriangleMesh, camera: Camera, sheet):
    """The exact (unblurred) silhouette of a part lying on the sheet (mesh in the sheet frame) as seen by `camera` at
    the sheet pose `sheet` (R, t: sheet -> camera), along lines through pixels: a function (x, n2, half) -> (offset
    px along n2 where the silhouette ends, valid) - what a perfect edge finder would give on a perfect photo (the
    outline_thread validation's reference)."""
    V = np.asarray(part.vertices)
    shift = V.mean(0)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor((V - shift).astype(np.float32)),
                        o3d.core.Tensor(np.asarray(part.triangles).astype(np.uint32)))
    C = -sheet.R.T @ sheet.t

    def edges(x, n2, half):
        reach = half + 2.0

        def solid(v):
            d = camera.rays(x + v[:, None] * n2) @ sheet.R
            rays = np.concatenate([np.broadcast_to(C - shift, d.shape), d], 1).astype(np.float32)
            return np.isfinite(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy())

        lo, hi = np.full(len(x), -reach), np.full(len(x), reach)
        valid = solid(lo) & ~solid(hi)
        for _ in range(26):
            mid = (lo + hi) / 2
            s = solid(mid)
            lo = np.where(s, mid, lo)
            hi = np.where(s, hi, mid)
        return (lo + hi) / 2, valid

    return edges


def rigid_between(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation and shift with B = A @ R.T + t (the same mesh's vertices before and after place())."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


# --------------------------------------------------------------------------- test parts with known errors
PARTS = {"bolt": (bolt, BOLT, (1.0, 0.0, 0.0)), "plate": (plate, PLATE, (0.0, 0.0, -1.0)),
         "screw": (screw, SCREW, (1.0, 0.0, 0.0))}


def part_params(kind: str, errors: dict | None = None) -> tuple[dict, dict]:
    """(golden parameters, true parameters = golden + errors) of a test part; errors: {name: mm}."""
    if kind not in PARTS:
        raise ValueError(f"Unknown test part {kind!r}: use {' or '.join(PARTS)}")
    golden = dict(PARTS[kind][1])
    errors = errors or {}
    bad = set(errors) - set(golden)
    if bad:
        raise ValueError(f"{kind} has no {', '.join(sorted(bad))} (its dimensions: {', '.join(golden)})")
    return golden, {k: v + errors.get(k, 0.0) for k, v in golden.items()}


def expected(kind: str, golden: dict, true: dict) -> list[dict]:
    """What the outline check should find on a test part: {kind, golden, true} for each value its golden model
    offers (the check's plan for these parts), so results can be matched by kind and golden value."""
    def rows(p):
        if kind == "bolt":
            return [("size", p["head_h"] + p["under"]), ("thickness", p["head_h"]), ("step", p["under"]),
                    ("diameter", p["head_d"]), ("diameter", p["shank_d"])]
        if kind == "screw":
            return [("size", p["head_h"] + p["under"]), ("thickness", p["head_h"]), ("step", p["under"]),
                    ("diameter", p["head_d"]), ("diameter", p["major_d"]),
                    ("diameter", p["major_d"] - 2 * 0.54 * p["pitch"]), ("pitch", p["pitch"])]
        return [("size", p["length"]), ("size", p["width"]), ("size", p["thickness"]),
                ("thickness", p["step_x"]), ("thickness", p["step_y"]),
                ("step", p["length"] - p["step_x"]), ("step", p["width"] - p["step_y"]), ("diameter", p["hole_d"])]
    return [{"kind": k, "golden": round(g, 6), "true": round(t, 6)} for (k, g), (_, t) in zip(rows(golden),
                                                                                              rows(true))]


def make_part(kind: str, params: dict, segments: int | None = None) -> o3d.geometry.TriangleMesh:
    fn = PARTS[kind][0]
    return fn(**params) if segments is None else fn(**params, segments=segments)


def true_value(rows: list[dict], kind: str, golden: float) -> float | None:
    """The true value of a measurement (matched by kind and golden value) from expected()."""
    for r in rows:
        if r["kind"] == kind and abs(r["golden"] - golden) < 1e-3:
            return r["true"]
    return None
