"""How accurate is a photo model at true size from the scale sheet? (docs/photos-to-3d.md, Accuracy with the sheet)

A CAD flange of known dimensions lying on the printed scale sheet (cloudclean/scale_sheet.py) is rendered from
cameras round it: ray traced (Open3D), textured (a sprayed-looking speckle on the part, the sheet's own print on the
paper, a wooden table), lit with shadows, anti-aliased, with sensor noise and JPEG. The photos then go through
tools/recon/photos_to_3d.py --scale-sheet, and the result is compared with the truth, with no alignment of any kind:

  (a) scale: the recovered cameras against the true ones (size error in %, and where the sheet's frame landed)
  (b) dimensions measured on the recovered points (sparse and dense apart) by fitting circles and planes: outer
      diameter, plate thickness and overall height (above the sheet), hub and bore diameters, bolt-hole diameters,
      positions and pitch circle
  (c) surface error: distance of the recovered points to the true mesh (median, 90 %) and how much of it is covered

    python sheet_check.py flange FLANGE.ply                                 (the CAD flange; needs manifold3d)
    python sheet_check.py render FLANGE.ply OUT [--per-ring 12] [--rings 30,55] [--blur 1.5] [--curl 3]
                                                [--printed 1.01] [--k1 0] [--distance 250]
    python sheet_check.py evaluate FLANGE.ply OUT/truth.json RECON_OUT [--label NAME]

Run from the repository root with CloudClean's venv (render imports cloudclean.scale_sheet for the sheet's print).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# the flange (mm): a plate with a hub, a bore and four bolt holes, lying on the sheet with its bottom at z = 0
FLANGE = {"od": 97.0, "plate": 8.0, "height": 22.0, "hub_od": 50.0, "bore": 28.0, "holes": 4, "hole_d": 11.0,
          "pcd": 72.0, "hole_angle0": 45.0}
POSE = {"x": 3.0, "y": -4.0, "angle_deg": 20.0}   # where it lies on the sheet: centre and turn about z
IMAGE = (2400, 1800)                                 # photo size (px): a phone photo as the pipeline sees it
HFOV = 55.0                                          # a phone's main camera


# --------------------------------------------------------------------------- the CAD flange
def build_flange(path: Path) -> None:
    from manifold3d import Manifold

    f = FLANGE
    body = Manifold.cylinder(f["plate"], f["od"] / 2, circular_segments=1440)
    body = body + Manifold.cylinder(f["height"] - f["plate"] + 1.0, f["hub_od"] / 2,
                                    circular_segments=720).translate((0, 0, f["plate"] - 1.0))
    body = body - Manifold.cylinder(f["height"] + 2.0, f["bore"] / 2, circular_segments=720).translate((0, 0, -1.0))
    for k in range(f["holes"]):
        a = math.radians(f["hole_angle0"] + 360.0 * k / f["holes"])
        hole = Manifold.cylinder(f["plate"] + 2.0, f["hole_d"] / 2, circular_segments=360)
        body = body - hole.translate((f["pcd"] / 2 * math.cos(a), f["pcd"] / 2 * math.sin(a), -1.0))
    mesh = body.to_mesh()
    V = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    F = np.asarray(mesh.tri_verts).astype(np.int32)
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty double x\nproperty double y\n"
              f"property double z\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    faces = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)])
    faces["n"], faces["i"] = 3, F
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(V.astype("<f8").tobytes())
        fh.write(faces.tobytes())
    print(f"{len(V)} vertices, {len(F)} triangles -> {path}")


def pose_matrix() -> np.ndarray:
    a = math.radians(POSE["angle_deg"])
    T = np.eye(4)
    T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = [POSE["x"], POSE["y"], 0.0]
    return T


def load_flange(path: Path):
    """The flange mesh in the sheet's frame (Open3D legacy mesh)."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.transform(pose_matrix())
    mesh.compute_triangle_normals()
    return mesh


def hole_centres() -> np.ndarray:
    T = pose_matrix()
    out = []
    for k in range(FLANGE["holes"]):
        a = math.radians(FLANGE["hole_angle0"] + 360.0 * k / FLANGE["holes"])
        out.append(T[:2, :2] @ [FLANGE["pcd"] / 2 * math.cos(a), FLANGE["pcd"] / 2 * math.sin(a)] + T[:2, 3])
    return np.array(out)


# --------------------------------------------------------------------------- textures
def value_noise(X: np.ndarray, cell: float, seed: int) -> np.ndarray:
    """Smooth random field in [0, 1): random values on a grid of `cell` mm, interpolated (smoothstep)."""
    table_seed = int(np.random.default_rng(seed).integers(1, 1 << 30))
    g = X / cell
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    f = f * f * (3 - 2 * f)
    acc = np.zeros(len(X))
    dims = X.shape[1]
    for corner in range(1 << dims):
        w = np.ones(len(X))
        k = np.zeros(len(X), np.int64)
        for d in range(dims):
            bit = (corner >> d) & 1
            w *= f[:, d] if bit else 1 - f[:, d]
            k ^= (i0[:, d] + bit + (table_seed if d == 0 else 0)) * (73856093, 19349663, 83492791)[d]
        acc += w * ((np.abs(k) % 1000003) / 1000003.0)
    return acc


def part_albedo(P_local: np.ndarray) -> np.ndarray:
    """A sprayed (matte, speckled) part: grey-beige with speckles at several scales."""
    t = 0.5 * value_noise(P_local, 4.0, 1) + 0.3 * value_noise(P_local, 1.3, 2) + 0.2 * value_noise(P_local, 0.5, 3)
    base = np.stack([0.62 + 0.35 * (t - 0.5), 0.58 + 0.35 * (t - 0.5), 0.52 + 0.35 * (t - 0.5)], axis=1)
    base[value_noise(P_local, 0.35, 4) > 0.82] *= 0.35
    return base


def table_albedo(P: np.ndarray) -> np.ndarray:
    """A wooden table top: brown with grain along x."""
    g = 0.6 * value_noise(P[:, :2] * [1 / 30.0, 1 / 1.2], 1.0, 7) + 0.4 * value_noise(P[:, :2], 9.0, 8)
    return np.array([0.50, 0.36, 0.24]) * (0.75 + 0.5 * g)[:, None]


class Sheet:
    """The printed sheet: its print (20 px/mm), how big the printer made it, and how it lies (flat, or its sides
    curling up by `curl` mm at the short edges, flat under the part)."""

    def __init__(self, printed: float = 1.0, curl: float = 0.0, paper: str = "a4"):
        from cloudclean import scale_sheet

        self.w, self.h, _ = scale_sheet.PAPERS[paper]
        self.ppm = 20
        self.img = scale_sheet.raster(paper, self.ppm).astype(np.float32) / 255.0
        self.printed, self.curl = printed, curl
        self.flat = 60.0   # mm either side of the middle that stay flat (under the part)
        # the curled paper keeps its length: parametrise by arc length u along x, height z(u)
        u = np.linspace(0, self.w / 2 * printed, 2001)
        excess = np.clip(u - self.flat, 0, None)
        c = curl / max((self.w / 2 * printed - self.flat) ** 2, 1e-9)
        slope = 2 * c * excess
        theta = np.arctan(slope)
        du = np.diff(u)
        self.u = u
        self.x = np.r_[0, np.cumsum(np.cos(theta[:-1]) * du)]
        self.z = np.r_[0, np.cumsum(np.sin(theta[:-1]) * du)]

    def mesh(self):
        import open3d as o3d

        hw, hh = self.w / 2 * self.printed, self.h / 2 * self.printed
        if self.curl <= 0:
            V = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]], float)
            F = np.array([[0, 1, 2], [0, 2, 3]])
        else:
            xs = np.r_[-self.x[::-10][:-1], self.x[::10]]
            zs = np.r_[self.z[::-10][:-1], self.z[::10]]
            V = np.concatenate([np.c_[xs, np.full_like(xs, -hh), zs], np.c_[xs, np.full_like(xs, hh), zs]])
            n = len(xs)
            F = np.array([[k, k + 1, n + k + 1] for k in range(n - 1)] + [[k, n + k + 1, n + k] for k in range(n - 1)])
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
        m.compute_triangle_normals()
        return m

    def albedo(self, P: np.ndarray) -> np.ndarray:
        """Paper white / ink black from the print at the paper position of each 3D point."""
        u = np.sign(P[:, 0]) * np.interp(np.abs(P[:, 0]), self.x, self.u) if self.curl > 0 else P[:, 0]
        u, v = u / self.printed, P[:, 1] / self.printed          # position on the design (mm)
        col = (u + self.w / 2) * self.ppm - 0.5
        row = (self.h / 2 - v) * self.ppm - 0.5
        H, W = self.img.shape
        c0 = np.clip(np.floor(col).astype(int), 0, W - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, H - 2)
        fc, fr = np.clip(col - c0, 0, 1), np.clip(row - r0, 0, 1)
        im = self.img
        g = ((im[r0, c0] * (1 - fc) + im[r0, c0 + 1] * fc) * (1 - fr)
             + (im[r0 + 1, c0] * (1 - fc) + im[r0 + 1, c0 + 1] * fc) * fr)
        a = 0.05 + (0.86 - 0.05) * g     # ink and paper reflectance
        return np.repeat(a[:, None], 3, axis=1) * [1.0, 0.99, 0.96]

    def marker_points(self) -> dict:
        """Where the markers' corners really are in 3D (curled paper lifts them)."""
        from cloudclean import scale_sheet

        out = {}
        for i, c in scale_sheet.marker_corners().items():
            u = c[:, 0] * self.printed
            x = np.sign(u) * np.interp(np.abs(u), self.u, self.x)
            z = np.interp(np.abs(u), self.u, self.z)
            out[i] = np.c_[x, c[:, 1] * self.printed, z]
        return out


# --------------------------------------------------------------------------- cameras and rendering
def look_at(C: np.ndarray, target: np.ndarray, roll_deg: float = 0.0) -> np.ndarray:
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = math.radians(roll_deg)
    x, y = math.cos(r) * x + math.sin(r) * y, -math.sin(r) * x + math.cos(r) * y
    W = np.eye(4)
    W[:3, :3] = np.stack([x, y, z])
    W[:3, 3] = -W[:3, :3] @ C
    return W


def cameras(per_ring: int, rings: list[float], distance: float, seed: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for k, elev in enumerate(rings):
        for az in np.arange(per_ring) * 360.0 / per_ring + k * 180.0 / per_ring + 10.0:
            a, e = math.radians(az), math.radians(elev)
            d = distance + rng.normal(0, 8)
            C = d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
            target = np.array([rng.normal(0, 4), rng.normal(0, 4), 8.0 + rng.normal(0, 2)])
            out.append(look_at(C, target, rng.normal(0, 3)))
    return out


def render(scene, sheet: Sheet, flange_inv: np.ndarray, W: np.ndarray, K: np.ndarray, size, k1: float = 0.0,
           ss: int = 2, light=(0.35, 0.25, 1.0), seed: int = 0) -> np.ndarray:
    """One photo (float RGB 0..1): ray traced with ss x ss samples per pixel, Lambert shading and shadows."""
    import open3d as o3d

    width, height = size
    L = np.array(light, float)
    L /= np.linalg.norm(L)
    R, C = W[:3, :3], np.linalg.inv(W)[:3, 3]
    img = np.zeros((height, width, 3), np.float32)
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    rows_per = max(1, 1_500_000 // (width * ss * ss))
    for r0 in range(0, height, rows_per):
        r1 = min(height, r0 + rows_per)
        v, u, dv, du = np.meshgrid(np.arange(r0, r1), np.arange(width), offs, offs, indexing="ij")
        xd = (u + du - K[0, 2]) / K[0, 0]
        yd = (v + dv - K[1, 2]) / K[1, 1]
        xu, yu = xd.copy(), yd.copy()
        if k1:
            for _ in range(8):   # distorted -> ideal (SIMPLE_RADIAL: x_d = x_u (1 + k1 r_u^2))
                f = 1 + k1 * (xu ** 2 + yu ** 2)
                xu, yu = xd / f, yd / f
        d = np.stack([xu.ravel(), yu.ravel(), np.ones(xu.size)], axis=1) @ R   # camera -> world (R^T d)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        rays = np.concatenate([np.broadcast_to(C, d.shape), d], axis=1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        t = hit["t_hit"].numpy()
        gid = hit["geometry_ids"].numpy()
        n = hit["primitive_normals"].numpy().astype(np.float64)
        col = np.zeros((len(d), 3))
        miss = ~np.isfinite(t)
        col[miss] = 0.55 + 0.1 * (d[miss, 2:3] > 0)   # the room behind: a grey wall
        ok = ~miss
        P = C + d[ok] * t[ok, None]
        N = n[ok]
        N[(N * d[ok]).sum(1) > 0] *= -1
        g = gid[ok]
        alb = np.zeros((len(P), 3))
        for k, fn in ((0, lambda Q: part_albedo(Q @ flange_inv[:3, :3].T + flange_inv[:3, 3])),
                      (1, sheet.albedo), (2, table_albedo)):
            sel = g == k
            if sel.any():
                alb[sel] = fn(P[sel])
        lit = np.clip(N @ L, 0, None)
        shadow_rays = np.concatenate([P + N * 0.02, np.broadcast_to(L, P.shape)], axis=1).astype(np.float32)
        blocked = np.isfinite(scene.cast_rays(o3d.core.Tensor(shadow_rays))["t_hit"].numpy())
        col[ok] = alb * (0.38 + 0.72 * lit * ~blocked)[:, None]
        img[r0:r1] = col.reshape(r1 - r0, width, ss * ss, 3).mean(axis=2)
    return img


def cmd_render(args) -> None:
    import open3d as o3d
    from PIL import Image, ImageFilter

    out = Path(args.out)
    (out / "photos").mkdir(parents=True, exist_ok=True)
    flange = load_flange(Path(args.flange))
    sheet = Sheet(args.printed, args.curl)
    table = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[-800, -800, -0.1], [800, -800, -0.1],
                                                                   [800, 800, -0.1], [-800, 800, -0.1]]),
                                      o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
    scene = o3d.t.geometry.RaycastingScene()
    for m in (flange, sheet.mesh(), table):   # geometry ids 0, 1, 2
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    width, height = IMAGE
    f = width / 2 / math.tan(math.radians(HFOV / 2))
    K = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])
    rings = [float(x) for x in args.rings.split(",")]
    cams = cameras(args.per_ring, rings, args.distance)
    flange_inv = np.linalg.inv(pose_matrix())
    rng = np.random.default_rng(5)
    started = time.time()
    truth = []
    for k, W in enumerate(cams):
        img = render(scene, sheet, flange_inv, W, K, IMAGE, args.k1)
        img = np.clip(img * 255 * 1.05 + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        im = Image.fromarray(img)
        if args.blur > 0:
            im = im.filter(ImageFilter.GaussianBlur(args.blur))
        name = f"{k:03d}.jpg"
        im.save(out / "photos" / name, quality=92)
        truth.append({"image": name, "world_to_camera": W.tolist(), "cam_to_world": np.linalg.inv(W).tolist()})
        print(f"{name} {time.time() - started:.0f} s", flush=True)
    (out / "truth.json").write_text(json.dumps({
        "flange": FLANGE, "pose": POSE, "K": K.tolist(), "size": IMAGE, "k1": args.k1, "blur": args.blur,
        "curl_mm": args.curl, "printed": args.printed, "rings": rings, "per_ring": args.per_ring,
        "distance": args.distance, "cameras": truth,
        "marker_corners": {str(i): c.tolist() for i, c in sheet.marker_points().items()}}, indent=1))


# --------------------------------------------------------------------------- measuring
MIN_CIRCLE_POINTS, MIN_CIRCLE_ARC = 20, 180.0   # a circle counts only with this many points round this much arc


def arc_deg(xy: np.ndarray, centre: np.ndarray) -> float:
    """How much of the circle the points go round (360 minus the largest empty gap), degrees."""
    a = np.sort(np.degrees(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])))
    gaps = np.diff(np.r_[a, a[0] + 360.0])
    return float(360.0 - gaps.max())


def fit_circle(xy: np.ndarray, iters: int = 30):
    """Robust circle fit: algebraic start, then geometric Gauss-Newton with Tukey weights (c = 4.685 sigma).
    Returns (centre, radius, rms of inliers, inliers) or None when there are too few points or they cover too
    little of the circle (MIN_CIRCLE_POINTS, MIN_CIRCLE_ARC)."""
    if len(xy) < MIN_CIRCLE_POINTS:
        return None
    A = np.c_[2 * xy, np.ones(len(xy))]
    sol = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)[0]
    c = sol[:2]
    r = math.sqrt(max(sol[2] + c @ c, 1e-12))
    w = np.ones(len(xy))
    for _ in range(iters):
        d = xy - c
        dist = np.linalg.norm(d, axis=1)
        res = dist - r
        s = max(1.4826 * np.median(np.abs(res)), 1e-4)
        u = res / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        J = np.c_[-d / np.maximum(dist, 1e-12)[:, None], -np.ones(len(xy))]
        H = (J * w[:, None]).T @ J
        g = (J * w[:, None]).T @ res
        try:
            step = np.linalg.solve(H + 1e-9 * np.eye(3), -g)
        except np.linalg.LinAlgError:
            return None
        c, r = c + step[:2], r + step[2]
        if np.linalg.norm(step) < 1e-7:
            break
    inl = w > 0
    if inl.sum() < MIN_CIRCLE_POINTS or arc_deg(xy[inl], c) < MIN_CIRCLE_ARC:
        return None
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(xy[inl] - c, axis=1) - r) ** 2))), int(inl.sum())


def fit_plane_height(P: np.ndarray, at: np.ndarray, iters: int = 20):
    """Robust plane z = a x + b y + c (Tukey); its height at `at` (x, y). Returns (height, tilt deg, n) or None."""
    if len(P) < 10:
        return None
    w = np.ones(len(P))
    A = np.c_[P[:, 0] - at[0], P[:, 1] - at[1], np.ones(len(P))]
    for _ in range(iters):
        sol = np.linalg.lstsq(A * np.sqrt(w)[:, None], P[:, 2] * np.sqrt(w), rcond=None)[0]
        res = P[:, 2] - A @ sol
        s = max(1.4826 * np.median(np.abs(res)), 1e-4)
        u = res / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    return float(sol[2]), float(np.degrees(np.arctan(np.hypot(sol[0], sol[1])))), int((w > 0).sum())


def measure(P: np.ndarray) -> dict:
    """The flange's dimensions from points in the sheet's frame (mm), each from the points in a band round where
    the feature is on the drawing (as a person picking the feature would), fitted robustly."""
    f = FLANGE
    centre = np.array([POSE["x"], POSE["y"]])
    out = {}

    def band(r_lo, r_hi, z_lo, z_hi, about):
        r = np.linalg.norm(P[:, :2] - about, axis=1)
        return P[(r >= r_lo) & (r <= r_hi) & (P[:, 2] >= z_lo) & (P[:, 2] <= z_hi)]

    circles = {"outer_diameter": (f["od"] / 2 - 3, f["od"] / 2 + 3, 1.5, f["plate"] - 1.5, f["od"]),
               "hub_diameter": (f["hub_od"] / 2 - 3, f["hub_od"] / 2 + 3, f["plate"] + 2, f["height"] - 2,
                                f["hub_od"]),
               "bore_diameter": (f["bore"] / 2 - 3, f["bore"] / 2 + 3, 2.0, f["height"] - 2, f["bore"])}
    axis = None   # the flange's axis, measured: from the outside of the plate, else the hub, else the bore
    for key in circles:
        r_lo, r_hi, z_lo, z_hi, _ = circles[key]
        about = centre
        for _ in range(3):
            got = fit_circle(band(r_lo, r_hi, z_lo, z_hi, about)[:, :2])
            if got is None:
                break
            about = got[0]
        if got is not None:
            axis = centre = about
            break
    for key, (r_lo, r_hi, z_lo, z_hi, nominal) in circles.items():
        got = fit_circle(band(r_lo, r_hi, z_lo, z_hi, centre)[:, :2])
        out[key] = None if got is None else {"value": 2 * got[1], "error": 2 * got[1] - nominal, "points": got[3],
                                             "rms": got[2]}
    holes = hole_centres() - np.array([POSE["x"], POSE["y"]]) + centre
    plate_top = band(f["hub_od"] / 2 + 2.5, f["od"] / 2 - 2.5, f["plate"] - 2.0, f["plate"] + 2.0, centre)
    for h in holes:
        plate_top = plate_top[np.linalg.norm(plate_top[:, :2] - h, axis=1) > f["hole_d"] / 2 + 2.0]
    for key, pts, nominal in (("plate_thickness", plate_top, f["plate"]),
                              ("overall_height", band(f["bore"] / 2 + 2, f["hub_od"] / 2 - 2, f["height"] - 2.5,
                                                      f["height"] + 2.5, centre), f["height"])):
        got = fit_plane_height(pts, centre)
        out[key] = None if got is None else {"value": got[0], "error": got[0] - nominal, "points": got[2],
                                             "tilt_deg": got[1]}
    true_holes = hole_centres()
    hole_rows = []
    for h, th in zip(holes, true_holes):
        pts = band(2.0, f["hole_d"] / 2 + 3, 1.0, f["plate"] - 1.0, h)
        pts = pts[np.linalg.norm(pts[:, :2] - centre, axis=1) < f["od"] / 2 - 2]
        got = fit_circle(pts[:, :2])
        if got is None:
            hole_rows.append(None)
            continue
        hole_rows.append({"diameter_error": 2 * got[1] - f["hole_d"], "points": got[3],
                          "pcd_error": None if axis is None else 2 * np.linalg.norm(got[0] - axis) - f["pcd"],
                          "position_error": float(np.linalg.norm(got[0] - th)),   # absolute, in the sheet's frame
                          "centre": got[0].tolist()})
    out["holes"] = hole_rows
    out["axis_position_error"] = None if axis is None else float(np.linalg.norm(axis - [POSE["x"], POSE["y"]]))
    return out


# --------------------------------------------------------------------------- evaluation
def read_ply_points(path: Path) -> np.ndarray:
    import open3d as o3d

    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / ((xs ** 2).sum() / len(src)))
    return s, R, mu_d - s * R @ mu_s


def surface(P: np.ndarray, mesh_t, samples: np.ndarray) -> dict:
    """Distance of the points to the true surface, and the share of the true surface they cover."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    if not len(P):
        return {"points": 0}
    d = mesh_t.compute_distance(o3d.core.Tensor(P.astype(np.float32))).numpy()
    cover = cKDTree(P).query(samples, distance_upper_bound=1.0)[0]
    return {"points": int(len(P)), "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
            "mean": float(d.mean()), "within_0.5": float((d < 0.5).mean()),
            "covered_1mm": float(np.isfinite(cover).mean())}


def cmd_evaluate(args) -> dict:
    import open3d as o3d

    truth = json.loads(Path(args.truth).read_text())
    rec = Path(args.recon)
    cams = json.loads((rec / "cameras.json").read_text())
    sheet = cams.get("scale_sheet") or {}
    report = {"label": args.label, "photos": len(truth["cameras"]), "placed": cams.get("registered"),
              "sheet": {k: sheet.get(k) for k in ("found", "markers", "views", "reprojection_px", "residual_mm",
                                                   "max_residual_mm", "uncertainty_pct", "ruler_mm", "reason",
                                                   "warning", "level")}}
    true_c = {c["image"]: np.array(c["cam_to_world"])[:3, 3] for c in truth["cameras"]}
    got_c = {c["image"]: np.array(c["cam_to_world"])[:3, 3] for c in cams["cameras"]}
    names = sorted(set(true_c) & set(got_c))
    A, B = np.array([got_c[n] for n in names]), np.array([true_c[n] for n in names])
    s, R, t = umeyama(A, B)
    report["scale"] = {"size_error_pct": (1 / s - 1) * 100,     # + = the model came out too big
                       "frame_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                       "camera_error_mm": float(np.median(np.linalg.norm(A - B, axis=1))),   # no alignment at all
                       "camera_error_after_similarity_mm": float(np.median(np.linalg.norm((s * (R @ A.T)).T + t - B,
                                                                                          axis=1)))}
    mesh = load_flange(Path(args.flange))
    mesh_t = o3d.t.geometry.RaycastingScene()
    mesh_t.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    o3d.utility.random.seed(1)
    samples = np.asarray(mesh.sample_points_uniformly(200_000).points)
    samples = samples[samples[:, 2] > 0.05]   # the bottom face lies on the sheet: never seen
    centre = np.array([POSE["x"], POSE["y"]])
    roi = lambda P: P[(np.linalg.norm(P[:, :2] - centre, axis=1) < FLANGE["od"] / 2 + 6) & (P[:, 2] < FLANGE["height"] + 6)]
    report["truth_check"] = measure(samples)   # the measuring itself, on perfect points
    for kind in ("sparse", "points"):
        path = rec / f"{kind}.ply"
        if not path.exists():
            continue
        P = read_ply_points(path)
        Pr = roi(P)
        report[kind] = {"all_points": int(len(P)), "surface": surface(Pr, mesh_t, samples), "dims": measure(Pr)}
    if args.save:
        Path(args.save).write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("flange")
    p.add_argument("out")
    p = sub.add_parser("render")
    p.add_argument("flange")
    p.add_argument("out")
    p.add_argument("--per-ring", type=int, default=12)
    p.add_argument("--rings", default="30,55")
    p.add_argument("--distance", type=float, default=250.0)
    p.add_argument("--blur", type=float, default=0.0, help="Gaussian blur (px) of each photo")
    p.add_argument("--curl", type=float, default=0.0, help="the sheet's short edges curl up by this (mm)")
    p.add_argument("--printed", type=float, default=1.0, help="the printer's scale (1.01 = 1 %% too big)")
    p.add_argument("--k1", type=float, default=0.0, help="radial lens distortion")
    p = sub.add_parser("evaluate")
    p.add_argument("flange")
    p.add_argument("truth")
    p.add_argument("recon")
    p.add_argument("--label", default="")
    p.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.cmd == "flange":
        build_flange(Path(args.out))
    elif args.cmd == "render":
        cmd_render(args)
    else:
        cmd_evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
