"""Synthetic assets for the assistant evaluation (tools/assistant_eval/run_eval.py).

Everything is in millimetres and deterministic (fixed seeds), so a BEFORE and an AFTER run see identical files.

flange.ply          mesh of a flange: disc 150 mm across, 12 mm thick, central bore 40 mm, 4 lightening holes 25 mm
                    on a 95 mm circle. Like a scan mesh, the design holes are open: only the outer 4 mm of their walls
                    were scanned from the top and from the bottom, so each design hole shows two boundary loops (at
                    z = 8 and z = 4). Watertight otherwise, except for the flaws to fill: 30 round sticker holes
                    6 mm across (15 on the top face, 15 on the bottom face, only in the surface) and 20 tiny irregular
                    gaps 0.5-2 mm across (8 top, 8 bottom, 4 on the rim). 60 boundary loops in all. Axis-aligned
                    at the origin by default; build_assets(posed=True) stores it in a scanner-frame pose instead.
flange scan.ply     point cloud (with normals) sampled from that mesh - the "scan" of scenario B (same pose).
bracket scan.ply    small scan: a 40 x 25 x 10 block with a 12 mm boss, bottom not scanned (scenario C).
tube CAD.stl        golden model: tests/golden_synthetic.py stepped_tube() (scenario H).
tube scan.ply       tests/golden_synthetic.py stepped_tube_scan() (scenario H).
flange view.png     a rendered "screenshot of the 3D view" of the flange (scenario F).
truth.json          where the design holes and the flaws are (for the checks).
real/               scenario I (build_real_assets): a scan-like flange whose 15 largest holes hide every sticker
                    hole, the other assets of a busy project, its screenshot and truth real.json.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ASSET_VERSION = "v1"   # bump when any generator changes (cached asset folders are keyed by it)
REPO = Path(__file__).resolve().parents[2]

R_OUT = 75.0           # flange radius
THICK = 12.0           # flange thickness (z from 0 to 12)
WALL_SCANNED = 4.0     # mm of every design-hole wall scanned from each side
N_RIM = 360            # points around the outer rim
RIM_ROWS = 8           # rows of the rim band (1.5 mm each)
DESIGN = [("centre bore", (0.0, 0.0), 20.0, 160)] + [
    (f"lightening hole at {a} deg", (round(47.5 * math.cos(math.radians(a)), 6), round(47.5 * math.sin(math.radians(a)), 6)),
     12.5, 100) for a in (0, 90, 180, 270)]
STICKER_R = 3.0
N_STICKER_PTS = 24
STICKERS_PER_FACE = 15
GAPS_PER_FACE = 8
RIM_GAPS = 4
SPACING = 1.5          # interior point spacing of the faces
MARGIN = 1.2           # interior points keep this far from every boundary


# --------------------------------------------------------------------------- planar triangulation with holes
def _circle(center, r, n, phase=0.0) -> np.ndarray:
    a = phase + 2 * np.pi * np.arange(n) / n
    return np.c_[center[0] + r * np.cos(a), center[1] + r * np.sin(a)]


def _inside_convex(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Points strictly inside a CCW convex polygon."""
    inside = np.ones(len(pts), dtype=bool)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        cross = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
        inside &= cross > 0
    return inside


def _face(holes: list[dict], rng) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Triangulate the disc minus `holes` ({'poly': CCW points, 'center', 'r_max', 'round': bool}).

    The outer circle's points come first (N_RIM), then every hole's points in the given order, then interior points.
    Returns (points 2D, CCW triangles, index array of every loop: [outer, *holes])."""
    from scipy.spatial import Delaunay

    outer = _circle((0.0, 0.0), R_OUT, N_RIM)
    loops, pts, start = [], [outer], N_RIM
    loops.append(np.arange(N_RIM))
    for h in holes:
        loops.append(np.arange(start, start + len(h["poly"])))
        pts.append(h["poly"])
        start += len(h["poly"])
    g = np.arange(-R_OUT, R_OUT + SPACING, SPACING)
    xx, yy = np.meshgrid(g, g)
    grid = np.c_[xx.ravel(), yy.ravel()] + rng.uniform(-0.3, 0.3, (xx.size, 2)) * SPACING
    keep = np.hypot(grid[:, 0], grid[:, 1]) < R_OUT - MARGIN
    for h in holes:
        keep &= np.hypot(grid[:, 0] - h["center"][0], grid[:, 1] - h["center"][1]) > h["r_max"] + MARGIN
    pts.append(grid[keep])
    P = np.vstack(pts)
    tri = Delaunay(P).simplices.astype(np.int64)
    c = P[tri].mean(axis=1)
    ok = np.hypot(c[:, 0], c[:, 1]) < R_OUT
    for h in holes:
        if h["round"]:
            ok &= np.hypot(c[:, 0] - h["center"][0], c[:, 1] - h["center"][1]) > h["r_min"]
        else:
            ok &= ~_inside_convex(c, h["poly"])
    tri = tri[ok]
    a, b, d = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    area = (b[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0])
    tri[area < 0] = tri[area < 0][:, [0, 2, 1]]
    edges = {(int(min(u, v)), int(max(u, v))) for t in tri for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    for lp in loops:
        for i in range(len(lp)):
            u, v = int(lp[i]), int(lp[(i + 1) % len(lp)])
            if (min(u, v), max(u, v)) not in edges:
                raise RuntimeError("a hole edge is missing from the triangulation")
    return P, tri, loops


def _place(rng, n, radius, clear_from: list[tuple], r_limit, min_gap) -> list[tuple]:
    """n random centres on the disc, each `min_gap` clear of the keep-outs (x, y, r) and of each other."""
    out = []
    for _ in range(20000):
        if len(out) == n:
            break
        x, y = rng.uniform(-r_limit, r_limit, 2)
        if math.hypot(x, y) > r_limit:
            continue
        if any(math.hypot(x - cx, y - cy) < cr + radius + min_gap for cx, cy, cr in clear_from + out):
            continue
        out.append((float(x), float(y), radius))
    if len(out) < n:
        raise RuntimeError("could not place the features")
    return out


def _gap_poly(rng, center, diameter, spacing: float | None = None, max_e: float = 0.3) -> np.ndarray:
    """A convex irregular polygon (points on a random ellipse); spacing = vertex spacing in mm (default 5-8 points)."""
    k = int(rng.integers(5, 9)) if spacing is None else max(8, int(math.pi * diameter / spacing))
    e = rng.uniform(0.0, max_e)
    a0 = rng.uniform(0, 2 * np.pi)
    ang = 2 * np.pi * np.arange(k) / k + rng.uniform(-0.2, 0.2, k) * (2 * np.pi / k)
    ang = np.sort(ang)
    ra, rb = diameter / 2 * (1 + e), diameter / 2 * (1 - e)
    x, y = ra * np.cos(ang), rb * np.sin(ang)
    return np.c_[center[0] + x * np.cos(a0) - y * np.sin(a0), center[1] + x * np.sin(a0) + y * np.cos(a0)]


# --------------------------------------------------------------------------- the flange
def make_flange(seed: int = 7, realistic: bool = False):
    """(open3d TriangleMesh, truth dict). See the module docstring.

    realistic=True adds what a real scan mesh has besides sticker holes: medium unscanned patches, 2 per face
    (8-14 mm across) and 4 strips on the rim (about 10-20 x 3-6 mm). They are bigger than the sticker holes, so the
    15 largest boundary loops - all the `holes` tool lists - are the 10 design-hole loops and 5 of these patches: no
    sticker hole is visible in the list. The extra features use their own random stream, so the default flange is
    unchanged."""
    import open3d as o3d

    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1000)
    design_keepout = [(c[0], c[1], r) for _, c, r, _ in DESIGN]
    faces = {}
    truth_small = []
    stickers_all: list[tuple] = []
    for face in ("top", "bottom"):
        stickers = _place(rng, STICKERS_PER_FACE, STICKER_R, design_keepout + stickers_all, R_OUT - STICKER_R - 5, 5.0)
        stickers_all += stickers
        gap_sizes = rng.uniform(0.5, 2.0, GAPS_PER_FACE)
        gap_centres = _place(rng, GAPS_PER_FACE, 1.2, design_keepout + stickers, R_OUT - 6, 3.0)
        holes = []
        for name, c, r, n in DESIGN:
            holes.append({"poly": _circle(c, r, n), "center": c, "r_max": r, "r_min": r, "round": True})
        for x, y, r in stickers:
            holes.append({"poly": _circle((x, y), r, N_STICKER_PTS), "center": (x, y), "r_max": r, "r_min": r,
                          "round": True})
            truth_small.append({"kind": "sticker", "face": face, "center": [x, y, THICK if face == "top" else 0.0],
                                "diameter": 2 * r})
        for (x, y, _), d in zip(gap_centres, gap_sizes):
            poly = _gap_poly(rng, (x, y), d)
            cx, cy = poly.mean(axis=0)
            holes.append({"poly": poly, "center": (cx, cy), "r_max": float(np.max(np.hypot(*(poly - [cx, cy]).T))),
                          "r_min": 0.0, "round": False})
            per = float(np.sum(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)))
            truth_small.append({"kind": "gap", "face": face, "center": [float(cx), float(cy),
                                                                         THICK if face == "top" else 0.0],
                                "diameter": round(per / np.pi, 3)})
        if realistic:
            taken = design_keepout + stickers + [(x, y, 1.5) for x, y, _ in gap_centres]
            for x, y, _ in _place(rng2, 2, 7.5, taken, R_OUT - 11, 3.0):
                poly = _gap_poly(rng2, (x, y), rng2.uniform(8.0, 14.0), spacing=1.0, max_e=0.2)
                cx, cy = poly.mean(axis=0)
                holes.append({"poly": poly, "center": (cx, cy),
                              "r_max": float(np.max(np.hypot(*(poly - [cx, cy]).T))), "r_min": 0.0, "round": False})
                per = float(np.sum(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)))
                truth_small.append({"kind": "patch", "face": face, "center": [float(cx), float(cy),
                                                                               THICK if face == "top" else 0.0],
                                    "diameter": round(per / np.pi, 3)})
        for _ in range(50):
            try:
                faces[face] = _face(holes, rng)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("could not triangulate the flange face")

    P_top, T_top, L_top = faces["top"]
    P_bot, T_bot, L_bot = faces["bottom"]
    V = [np.c_[P_top, np.full(len(P_top), THICK)], np.c_[P_bot, np.zeros(len(P_bot))]]
    n_top = len(P_top)
    F = [T_top, T_bot[:, [0, 2, 1]] + n_top]
    count = n_top + len(P_bot)

    def ring(xy: np.ndarray, z: float) -> np.ndarray:
        nonlocal count
        V.append(np.c_[xy, np.full(len(xy), z)])
        idx = np.arange(count, count + len(xy))
        count += len(xy)
        return idx

    def band(rings: list[np.ndarray], outward: bool) -> np.ndarray:
        tris = []
        for up, lo in zip(rings[:-1], rings[1:]):
            u1, l1 = np.roll(up, -1), np.roll(lo, -1)
            if outward:
                tris.append(np.stack([np.c_[up, lo, l1], np.c_[up, l1, u1]], axis=1).reshape(-1, 3))
            else:
                tris.append(np.stack([np.c_[up, l1, lo], np.c_[up, u1, l1]], axis=1).reshape(-1, 3))
        return np.vstack(tris)

    # rim band: row k runs between ring k and k+1 (z from 12 down to 0); cell (k, j) = triangles 2*(k*N+j) and +1
    outer_xy = P_top[L_top[0]]
    rim_rings = [L_top[0]] + [ring(outer_xy, THICK - k * THICK / RIM_ROWS) for k in range(1, RIM_ROWS)] + \
                [L_bot[0] + n_top]
    rim = band(rim_rings, outward=True)
    remove = set()
    cols = (np.array([40, 130, 220, 310]) + rng.integers(0, 30, 4)) % N_RIM
    for g, (k, j) in enumerate(zip((2, 3, 4, 5), cols)):
        cell = 2 * (k * N_RIM + int(j))
        cut = [cell] if g % 2 == 0 else [cell, cell + 1]
        remove.update(cut)
        verts = np.unique(rim[cut].ravel())
        pts = np.vstack(V)[verts]
        ctr = pts.mean(axis=0)
        theta = math.atan2(ctr[1], ctr[0])
        sides = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        truth_small.append({"kind": "gap", "face": "rim", "center": [R_OUT * math.cos(theta), R_OUT * math.sin(theta),
                                                                     float(ctr[2])],
                            "diameter": round(float(sides.max()), 3)})
    if realistic:  # unscanned strips on the rim, well away from the tiny rim gaps (columns 40-70, 130-160, ...)
        allv = np.vstack(V)
        for j0 in (80, 170, 260, 350):
            width, height = int(rng2.integers(8, 16)), int(rng2.integers(2, 5))
            j0 = (j0 + int(rng2.integers(0, 12))) % N_RIM
            k0 = int(rng2.integers(1, RIM_ROWS - height))
            cells = [2 * (k * N_RIM + (j0 + dj) % N_RIM) + t for k in range(k0, k0 + height)
                     for dj in range(width) for t in (0, 1)]
            remove.update(cells)
            ctr = allv[np.unique(rim[cells].ravel())].mean(axis=0)
            theta = math.atan2(ctr[1], ctr[0])
            length = width * 2 * math.pi * R_OUT / N_RIM
            truth_small.append({"kind": "patch", "face": "rim",
                                "center": [R_OUT * math.cos(theta), R_OUT * math.sin(theta), float(ctr[2])],
                                "diameter": round(2 * (length + height * THICK / RIM_ROWS) / math.pi, 3)})
    rim = np.delete(rim, sorted(remove), axis=0)
    F.append(rim)
    # design-hole walls: the outer WALL_SCANNED mm from each face; the middle stays open
    for h in range(len(DESIGN)):
        top_loop, bot_loop = L_top[1 + h], L_bot[1 + h] + n_top
        xy = P_top[L_top[1 + h]]
        steps = [THICK - WALL_SCANNED / 2, THICK - WALL_SCANNED]
        F.append(band([top_loop] + [ring(xy, z) for z in steps], outward=False))
        steps = [WALL_SCANNED, WALL_SCANNED / 2]
        F.append(band([ring(xy, z) for z in steps] + [bot_loop], outward=False))
    V = np.vstack(V)
    F = np.vstack(F)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    truth = {
        "kind": "flange (scan-like)" if realistic else "flange", "units": "mm", "outer_diameter": 2 * R_OUT,
        "thickness": THICK,
        "design_holes": [{"name": name, "center": [c[0], c[1]], "diameter": 2 * r,
                          "loop_z": [THICK - WALL_SCANNED, WALL_SCANNED]} for name, c, r, _ in DESIGN],
        "small_features": truth_small,
        "expected_loops": 2 * len(DESIGN) + len(truth_small),
    }
    return mesh, truth


def check_orientation(F: np.ndarray) -> int:
    """Number of directed edges used twice (0 = consistently oriented)."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts > 1).sum())


# --------------------------------------------------------------------------- point clouds
def sample_mesh(mesh, n: int, seed: int, noise: float = 0.015, voxel: float = 0.0):
    """Area-uniform points with the triangle normals, a little noise along the normal (open3d PointCloud)."""
    import open3d as o3d

    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    rng = np.random.default_rng(seed)
    cross = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(area, 1e-12)[:, None]
    t = rng.choice(len(F), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    P = V[F[t, 0]] * (1 - s)[:, None] + V[F[t, 1]] * (s * (1 - r2))[:, None] + V[F[t, 2]] * (s * r2)[:, None]
    N = normals[t]
    P = P + N * rng.normal(0, noise, (n, 1))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    cloud.normals = o3d.utility.Vector3dVector(N)
    if voxel:
        cloud = cloud.voxel_down_sample(voxel)
        cloud.normalize_normals()
    return cloud


def make_bracket_scan(seed: int = 3):
    """A 40 x 25 x 10 block with a 12 mm boss (6 mm high) on top; the bottom face was not scanned."""
    import open3d as o3d

    box = o3d.geometry.TriangleMesh.create_box(40.0, 25.0, 10.0)
    boss = o3d.geometry.TriangleMesh.create_cylinder(radius=6.0, height=6.0, resolution=96, split=6)
    boss.translate((28.0, 12.5, 13.0))
    parts = []
    for m, n in ((box, 90_000), (boss, 12_000)):
        m.compute_triangle_normals()
        parts.append(sample_mesh(m, n, seed, noise=0.015))
        seed += 1
    P0, N0 = np.asarray(parts[0].points), np.asarray(parts[0].normals)
    keep = (N0[:, 2] > -0.5) & ~((N0[:, 2] > 0.5) & (np.hypot(P0[:, 0] - 28.0, P0[:, 1] - 12.5) < 6.0))
    P1, N1 = np.asarray(parts[1].points), np.asarray(parts[1].normals)
    keep1 = N1[:, 2] > -0.5
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([P0[keep], P1[keep1]])))
    cloud.normals = o3d.utility.Vector3dVector(np.vstack([N0[keep], N1[keep1]]))
    return cloud


def _golden_module():
    path = REPO / "tests" / "golden_synthetic.py"
    spec = importlib.util.spec_from_file_location("eval_golden_synthetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- a viewer-like screenshot
def render_png(mesh, path: Path, width: int = 1400, height: int = 900, eye_dir=(0.55, -0.8, 0.7),
               target=(0.0, 0.0, 4.0), dist: float = 265.0, fov_deg: float = 34.0) -> None:
    """Shade the mesh like the app's 3D view (dark theme) with a painter's algorithm; supersampled 2x."""
    from PIL import Image, ImageDraw

    ss = 2
    W, H = width * ss, height * ss
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    d = np.asarray(eye_dir, float) / np.linalg.norm(eye_dir)
    tgt = np.asarray(target, float)
    eye = tgt + dist * d
    fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = V - eye
    x, y, z = rel @ right, rel @ up, rel @ fwd
    f = (H / 2) / math.tan(math.radians(fov_deg) / 2)
    sx, sy = W / 2 + f * x / z, H / 2 - f * y / z
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    centre = V[F].mean(axis=1)
    to_eye = eye - centre
    front = np.einsum("ij,ij->i", n, to_eye) > 0
    light = d + np.array([0.3, 0.2, 0.5])
    light /= np.linalg.norm(light)
    lam = np.abs(n @ light)
    depth = z[F].mean(axis=1)
    img = Image.new("RGB", (W, H))
    top, bottom = np.array([38, 41, 46]), np.array([17, 18, 21])
    grad = (top[None, :] * (1 - np.linspace(0, 1, H))[:, None] + bottom[None, :] * np.linspace(0, 1, H)[:, None])
    img.paste(Image.fromarray(np.repeat(grad[:, None, :], W, axis=1).astype(np.uint8)))
    draw = ImageDraw.Draw(img)
    base = np.array([196, 201, 208])
    for i in np.argsort(-depth):
        shade = (0.28 + 0.72 * lam[i]) if front[i] else 0.18 + 0.2 * lam[i]
        col = tuple(int(c) for c in np.clip(base * shade, 0, 255))
        tri = F[i]
        draw.polygon([(sx[tri[0]], sy[tri[0]]), (sx[tri[1]], sy[tri[1]]), (sx[tri[2]], sy[tri[2]])], fill=col)
    img = img.resize((width, height), Image.LANCZOS)
    img.save(path, "PNG", optimize=True)


# --------------------------------------------------------------------------- scanner frame (optional)
# Real scans rarely sit at the origin, axis-aligned. With posed=True (run_eval.py --posed-flange) the flange files are
# placed like a scan in the scanner's frame (turned, tilted a little, away from the origin), so tool results carry
# untidy coordinates. The checks map every result back to the part's axes with the inverse (truth.json "pose").
# Measured: this barely changes the size of a holes list result (centres are rounded to 2 decimals).
POSE_DEG = (-6.0, 9.0, 37.0)        # rotations about x, then y, then z
POSE_T = (112.43, -63.71, 41.87)    # mm


def pose() -> tuple[np.ndarray, np.ndarray]:
    ax, ay, az = np.radians(POSE_DEG)
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx, np.asarray(POSE_T, dtype=float)


# --------------------------------------------------------------------------- all of it
def asset_version(posed: bool = False) -> str:
    return ASSET_VERSION + ("-posed" if posed else "")


def build_assets(folder: Path, log=print, posed: bool = False) -> dict:
    """Write every asset (once; cached by asset_version()) and return {name: path, 'truth': dict}."""
    import open3d as o3d

    folder = Path(folder)
    version = asset_version(posed)
    done = folder / "done.json"
    names = {"flange": "flange.ply", "flange_scan": "flange scan.ply", "bracket_scan": "bracket scan.ply",
             "tube_cad": "tube CAD.stl", "tube_scan": "tube scan.ply", "flange_png": "flange view.png",
             "truth": "truth.json"}
    paths = {k: folder / v for k, v in names.items()}
    if done.exists() and json.loads(done.read_text()).get("version") == version and \
            all(p.exists() for p in paths.values()):
        out = {k: str(p) for k, p in paths.items()}
        out["truth"] = json.loads(paths["truth"].read_text())
        return out
    folder.mkdir(parents=True, exist_ok=True)
    log("building the synthetic assets (once) ...")
    flange, truth = make_flange()
    bad = check_orientation(np.asarray(flange.triangles))
    if bad:
        raise RuntimeError(f"flange mesh is not consistently oriented ({bad} doubled edges)")
    scan = sample_mesh(flange, 160_000, seed=11, noise=0.015, voxel=0.5)
    stored = flange
    if posed:
        R, t = pose()
        stored = o3d.geometry.TriangleMesh(flange)
        stored.rotate(R, center=(0.0, 0.0, 0.0))
        stored.translate(t)
        stored.compute_vertex_normals()
        scan.rotate(R, center=(0.0, 0.0, 0.0))        # points and normals
        scan.translate(t)
        truth["pose"] = {"rotation": R.tolist(), "translation": t.tolist(),
                         "note": "flange.ply / flange scan.ply = rotation @ part + translation; truth in part axes"}
    o3d.io.write_triangle_mesh(str(paths["flange"]), stored, write_ascii=False)
    o3d.io.write_point_cloud(str(paths["flange_scan"]), scan, write_ascii=False)
    o3d.io.write_point_cloud(str(paths["bracket_scan"]), make_bracket_scan(), write_ascii=False)
    gs = _golden_module()
    tube = gs.stepped_tube()
    tube.compute_triangle_normals()
    o3d.io.write_triangle_mesh(str(paths["tube_cad"]), tube)
    o3d.io.write_point_cloud(str(paths["tube_scan"]), gs.stepped_tube_scan(), write_ascii=False)
    render_png(flange, paths["flange_png"])
    truth["flange_scan_points"] = len(scan.points)
    paths["truth"].write_text(json.dumps(truth, indent=1))
    done.write_text(json.dumps({"version": version}))
    out = {k: str(p) for k, p in paths.items()}
    out["truth"] = truth
    return out


REAL_VERSION = "r1"


def build_real_assets(folder: Path, flange_scan: str, log=print) -> dict:
    """Scenario I's assets, in <folder>/real (cached by REAL_VERSION): the scan-like flange (make_flange(realistic=
    True)) named like a pipeline result, 'merge of 4 scans · mesh'; the rest of a busy project - 4 partial scans,
    their clean copies and the merge, cut from `flange_scan` (all imported, so they only fill the project's asset
    list); and a screenshot."""
    import open3d as o3d

    real = Path(folder) / "real"
    files = {"flange_real": "merge of 4 scans · mesh.ply", "busy_merge": "merge of 4 scans.ply",
             "flange_real_png": "flange view 2.png", "truth_real": "truth real.json"}
    for k in range(1, 5):
        files[f"busy_scan_{k}"] = f"flange scan {k}.ply"
        files[f"busy_clean_{k}"] = f"flange scan {k} · clean.ply"
    paths = {k: real / v for k, v in files.items()}
    done = real / "done.json"
    if not (done.exists() and json.loads(done.read_text()).get("version") == REAL_VERSION and
            all(p.exists() for p in paths.values())):
        real.mkdir(parents=True, exist_ok=True)
        log("building the scan-like flange assets (once) ...")
        mesh, truth = make_flange(realistic=True)
        bad = check_orientation(np.asarray(mesh.triangles))
        if bad:
            raise RuntimeError(f"scan-like flange is not consistently oriented ({bad} doubled edges)")
        o3d.io.write_triangle_mesh(str(paths["flange_real"]), mesh, write_ascii=False)
        cloud = o3d.io.read_point_cloud(str(flange_scan))
        P = np.asarray(cloud.points)
        ang = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360
        rng = np.random.default_rng(21)
        for k in range(1, 5):
            part = cloud.select_by_index(np.flatnonzero((ang - 90 * (k - 1)) % 360 < 150))
            o3d.io.write_point_cloud(str(paths[f"busy_scan_{k}"]), part, write_ascii=False)
            kept = part.select_by_index(np.flatnonzero(rng.random(len(part.points)) > 0.04))
            o3d.io.write_point_cloud(str(paths[f"busy_clean_{k}"]), kept, write_ascii=False)
        o3d.io.write_point_cloud(str(paths["busy_merge"]), cloud, write_ascii=False)
        render_png(mesh, paths["flange_real_png"])
        paths["truth_real"].write_text(json.dumps(truth, indent=1))
        done.write_text(json.dumps({"version": REAL_VERSION}))
    out = {k: str(p) for k, p in paths.items()}
    out["truth_real"] = json.loads(paths["truth_real"].read_text())
    return out


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "eval_assets"
    info = build_assets(target)
    print(json.dumps({k: v for k, v in info.items() if k != "truth"}, indent=1))
