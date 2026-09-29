"""Merge options you can look at: every plausible way two scans line up, refined and rendered to pictures.

A symmetric part (a 12-sided head, a thread, a plain cylinder) can line up in more than one pose that fits the
surface almost equally well; numbers alone cannot choose. This module collects those poses - the best fit (the
stickers' pose when both scans share 3+ marker stickers and it fits the surface) and the alternatives the pre-merge
alignment found - refines each with ICP, and draws each merged result from a few directions. The assistant compares the pictures with the
user's reference photos of the real part (tool compare_merge_options) and picks the option that matches.

Photos can tell apart options that look different (a flipped part, the wrong end joined, a head that ends up on the
wrong side); they cannot resolve sub-millimetre ambiguities, which the stickers do.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import open3d as o3d

from .io import Geometry, estimate_spacing, to_cloud

Log = Callable[[str], None]

RENDER_POINTS = 350_000     # per picture; plenty for a 480 px panel
PANEL = (480, 400)          # width, height of one view
BACKGROUND = (241, 238, 232)
REF_COLOUR = np.array([0.93, 0.52, 0.30])    # scan 1 (the reference) in the two-colour view
MOVED_COLOUR = np.array([0.30, 0.52, 0.90])  # scan 2
GREY = np.array([0.80, 0.79, 0.76])


def _rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def _same_pose(a: np.ndarray, b: np.ndarray, centre: np.ndarray, angle_deg: float = 2.0, shift_mm: float = 0.5) -> bool:
    """Two poses that move the part's centre and orientation (almost) identically."""
    return (_rotation_angle_deg(a[:3, :3].T @ b[:3, :3]) < angle_deg
            and float(np.linalg.norm((a[:3, :3] @ centre + a[:3, 3]) - (b[:3, :3] @ centre + b[:3, 3]))) < shift_mm)


def merge_options(ref: Geometry, moving: Geometry, params: dict | None = None, log: Log = print,
                  max_options: int = 4) -> list[dict]:
    """The distinct poses of `moving` on `ref` worth looking at, best fit first.

    Each: {name, source ("best fit" | "alternative" | "stickers"), T (4x4), fitness (overlap 0..1), rmse_mm,
    angle_from_best_deg, shift_from_best_mm}."""
    from .register import MergeParams, align_pair, refine_icp

    p = MergeParams.from_dict(params or {})
    tgt = o3d.geometry.PointCloud(to_cloud(ref))
    src = o3d.geometry.PointCloud(to_cloud(moving))
    spacing = float(np.median([estimate_spacing(tgt.points), estimate_spacing(src.points)]))
    diag = max(float(np.linalg.norm(c.get_max_bound() - c.get_min_bound())) for c in (tgt, src))
    voxel = p.feature_voxel if p.feature_voxel > 0 else max(diag / 60.0, spacing * 4)
    log("Finding the ways the scans can line up")
    best, info = align_pair(src, tgt, p, spacing, voxel, log=log)
    st = info.get("stickers") or {}
    first = (f"lined up on {st['common']} stickers", "stickers") if info.get("best_candidate") == "stickers" \
        else ("best fit", "best fit")
    candidates: list[tuple[str, str, np.ndarray]] = [(first[0], first[1], np.asarray(best))]
    for alt in info.get("alternatives") or []:
        angle = alt.get("angle_from_best")
        candidates.append((f"turned {angle:.0f}° from the best fit" if isinstance(angle, (int, float)) else
                           "an alternative fit", "alternative", np.asarray(alt["T"], float)))
    fine = spacing * p.icp_threshold_multiplier
    cache: dict = {}
    centre = np.asarray(src.get_center())
    options: list[dict] = []
    for name, source, T0 in candidates:
        if source in ("best fit", "stickers") and T0 is candidates[0][2]:
            T, fit, rmse = T0, float(info.get("fitness", 0.0)), float(info.get("rmse", 0.0))
        else:
            T, fit, rmse = refine_icp(src, tgt, T0, [fine * 2, fine], cache)
        if any(_same_pose(T, o["T"], centre) for o in options):
            log(f"  {name}: same pose as an option already found")
            continue
        options.append({"name": name, "source": source, "T": np.asarray(T, float), "fitness": float(fit),
                        "rmse_mm": float(rmse)})
        log(f"  option: {name} - overlap {fit:.2f}, gap {rmse:.3f} mm")
        if len(options) >= max_options:
            break
    first = options[0]
    for o in options:
        o["angle_from_best_deg"] = round(_rotation_angle_deg(first["T"][:3, :3].T @ o["T"][:3, :3]), 1)
        o["shift_from_best_mm"] = round(float(np.linalg.norm((o["T"][:3, :3] @ centre + o["T"][:3, 3])
                                                             - (first["T"][:3, :3] @ centre + first["T"][:3, 3]))), 3)
    return options


# --------------------------------------------------------------------------- pictures
def _points_normals(geom: Geometry, T: np.ndarray | None, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.geometry.PointCloud(to_cloud(geom))
    n = len(pcd.points)
    if n > limit:
        idx = np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))
        pcd = pcd.select_by_index(idx)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
    if T is not None:
        pcd.transform(T)
    return np.asarray(pcd.points), np.asarray(pcd.normals)


def _frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre and principal axes (rows: longest first) of the merged part: views follow the part, not the scanner."""
    c = points.mean(axis=0)
    sample = points[:: max(1, len(points) // 100_000)] - c
    _, _, axes = np.linalg.svd(sample, full_matrices=False)
    axes[2] = np.cross(axes[0], axes[1])
    return c, axes


def _views(axes: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(label, right, up) of the views: from the side (length across the picture), from one end, and an angle."""
    e1, e2, e3 = axes
    iso_fwd = (e1 + e2 + e3) / np.linalg.norm(e1 + e2 + e3)
    iso_right = np.cross(e3, iso_fwd)
    iso_right /= np.linalg.norm(iso_right)
    return [("side", e1, e2), ("end", e2, e3), ("angle", iso_right, np.cross(iso_fwd, iso_right))]


def _render(points: np.ndarray, normals: np.ndarray, colours: np.ndarray, centre: np.ndarray, right: np.ndarray,
            up: np.ndarray, scale: float, size=PANEL) -> np.ndarray:
    """Orthographic splat render with depth order (nearer points drawn last) and simple two-sided shading."""
    w, h = size
    fwd = np.cross(up, right)  # into the screen
    d = points - centre
    x, y, z = d @ right, d @ up, d @ fwd
    px = np.round(x * scale + w / 2).astype(int)
    py = np.round(-y * scale + h / 2).astype(int)
    shade = 0.28 + 0.72 * np.abs(normals @ fwd)
    rgb = np.clip(colours * shade[:, None] * 255, 0, 255).astype(np.uint8)
    order = np.argsort(-z)  # far first
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BACKGROUND
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        qx, qy = px[order] + dx, py[order] + dy
        ok = (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h)
        img[qy[ok], qx[ok]] = rgb[order][ok]
    return img


def render_option(ref: Geometry, moving: Geometry, T: np.ndarray, title: str) -> "Image.Image":
    """One picture of a merge option: three views of the part in plain grey (to compare with photos of the real part)
    above the same three views coloured by scan (orange = scan 1, blue = scan 2) to show how they overlap."""
    from PIL import Image, ImageDraw

    pa, na = _points_normals(ref, None, RENDER_POINTS // 2, 0)
    pb, nb = _points_normals(moving, T, RENDER_POINTS // 2, 1)
    pts, nrm = np.vstack([pa, pb]), np.vstack([na, nb])
    centre, axes = _frame(pts)
    extent = np.ptp((pts - centre) @ axes.T, axis=0)
    scale = 0.86 * min(PANEL[0] / max(extent[0], 1e-6), PANEL[1] / max(extent[0], 1e-6))
    grey = np.tile(GREY, (len(pts), 1))
    two = np.vstack([np.tile(REF_COLOUR, (len(pa), 1)), np.tile(MOVED_COLOUR, (len(pb), 1))])
    header = 34
    sheet = Image.new("RGB", (PANEL[0] * 3, PANEL[1] * 2 + header), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill=(30, 30, 34))
    for k, (label, right, up) in enumerate(_views(axes)):
        for row, colours in enumerate((grey, two)):
            panel = Image.fromarray(_render(pts, nrm, colours, centre, right, up, scale))
            ImageDraw.Draw(panel).text((10, 8), f"{label}{' - by scan' if row else ''}", fill=(90, 88, 84))
            sheet.paste(panel, (PANEL[0] * k, header + PANEL[1] * row))
    return sheet
