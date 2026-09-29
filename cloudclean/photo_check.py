"""Do the photos agree? A check - and a repair - of where photos were placed on a model, using the photos themselves.

Placed right, every point of the model looks the same in each photo that sees it. Placed wrong - turned a little,
shifted, the wrong size - points near holes and edges land on the table or background, differently in each photo.
The score (photo agreement) is 1 - the spread of each point's colour across the photos that see it, divided by the
spread of all the colours sampled: ~0.8-0.9 placed right on the test flange, ~0.4 placed clearly wrong.

Why it is needed: the geometric fit (cloudclean.photo_align) rests on the few exact points the reconstruction has on
the part. With few photos those lie mostly on flat faces, which still fit after a turn - 6 photos fitted "100 %"
while the cameras were 6 deg and 16 mm off. The photos' colours pin the pose down near holes and edges.

The repair turns and shifts the model about its centre, which is the same as moving the whole set of cameras the
other way; the cameras stay fixed relative to each other (COLMAP placed them exactly). It never changes the size:
a smaller model agrees better for the wrong reason (its edge points move in, away from the background), and a free
search even shrank the model onto one feature. It stays within 20 deg of where it started. A placement that is badly
wrong (e.g. the size off, from photos of one side only) is not repaired but refused: the agreement stays close to
that of wrong placements.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import open3d as o3d
from PIL import Image

from .texture import CameraView

Log = Callable[[str], None]

MAX_SIDE = 960           # photos are read at up to this size (colour detail near holes survives, speed matters)
ZBUF_REDUCE = 2          # the visibility buffer is this much coarser than the photos as read
ZBUF_POINTS = 120_000    # model points that decide what hides what (fewer leave gaps that hidden points show through)
SCORE_POINTS = 12_000    # model points scored
MAX_VIEWS = 12           # photos used while repairing (spread over the set); the final score uses them all
TURN_STEP = 30.0         # the part is also tried turned about each of its axes in steps of this many degrees
LEVELS = (4.0, 2.0, 1.0, 0.5, 0.25, 0.1)   # repair steps: degrees, % of the part size
MAX_TURN = 20.0          # the repair stays within this many degrees of where it started
TURNED_MARGIN = 0.05     # a turned start must agree this much better than the placement as found to replace it
MIN_GAIN = 0.02          # a repair must raise the agreement by at least this much (else the placement stays)


class _View:
    __slots__ = ("img", "K", "W", "C", "H", "Wd")

    def __init__(self, img: np.ndarray, K: np.ndarray, W: np.ndarray):
        self.img, self.K, self.W = img, K, W
        self.H, self.Wd = img.shape[:2]
        self.C = -W[:3, :3].T @ W[:3, 3]


def load_views(views: list[CameraView], max_side: int = MAX_SIDE) -> list[_View]:
    """The photos, read small, with intrinsics to match (cameras in the model's frame)."""
    out = []
    for v in views:
        with Image.open(v.image_path) as im:
            im = im.convert("RGB")
            f = min(1.0, max_side / max(im.size))
            small = im.resize((max(1, round(im.size[0] * f)), max(1, round(im.size[1] * f))), Image.BILINEAR)
        K = np.asarray(v.K, float).copy()
        K[0] *= small.size[0] / v.width
        K[1] *= small.size[1] / v.height
        out.append(_View(np.asarray(small, np.float32) / 255.0, K, np.asarray(v.world_to_camera, float)))
    return out


def _spread(views: list[_View], n: int) -> list[_View]:
    """n photos spread over the set (by viewing direction)."""
    if len(views) <= n:
        return views
    dirs = np.array([v.W[2, :3] for v in views])
    chosen = [0]
    dist = 1 - dirs @ dirs[0]
    while len(chosen) < n:
        k = int(np.argmax(dist))
        chosen.append(k)
        dist = np.minimum(dist, 1 - dirs @ dirs[k])
    return [views[k] for k in sorted(chosen)]


def _zbuffer(Z: np.ndarray, v: _View) -> tuple[np.ndarray, int]:
    Zc = Z @ v.W[:3, :3].T + v.W[:3, 3]
    z = Zc[:, 2]
    ok = z > 1e-9
    u = (v.K[0, 0] * Zc[ok, 0] / z[ok] + v.K[0, 2]) / ZBUF_REDUCE
    w = (v.K[1, 1] * Zc[ok, 1] / z[ok] + v.K[1, 2]) / ZBUF_REDUCE
    hz, wz = v.H // ZBUF_REDUCE + 1, v.Wd // ZBUF_REDUCE + 1
    ui, vi = np.floor(u).astype(np.int64), np.floor(w).astype(np.int64)
    inb = (ui >= 0) & (ui < wz) & (vi >= 0) & (vi < hz)
    pix, depth = vi[inb] * wz + ui[inb], z[ok][inb]
    buf = np.full(hz * wz, np.inf)
    if len(pix):
        order = np.argsort(pix, kind="stable")
        pix, depth = pix[order], depth[order]
        starts = np.flatnonzero(np.r_[True, pix[1:] != pix[:-1]])
        buf[pix[starts]] = np.minimum.reduceat(depth, starts)
    return buf, wz


def visible(points: np.ndarray, normals: np.ndarray, views: list[_View], zbuf_points: np.ndarray, tol: float
            ) -> list[np.ndarray]:
    """Per photo, the indices of the points it sees: in front, inside the photo, not edge-on, not hidden."""
    out = []
    for v in views:
        buf, wz = _zbuffer(zbuf_points, v)
        Pc = points @ v.W[:3, :3].T + v.W[:3, 3]
        z = Pc[:, 2]
        to_cam = v.C - points
        facing = np.abs(np.einsum("ij,ij->i", normals, to_cam)) > 0.15 * np.linalg.norm(to_cam, axis=1)
        idx = np.flatnonzero((z > 1e-9) & facing)
        u = v.K[0, 0] * Pc[idx, 0] / z[idx] + v.K[0, 2]
        w = v.K[1, 1] * Pc[idx, 1] / z[idx] + v.K[1, 2]
        inside = (u >= 1) & (u < v.Wd - 2) & (w >= 1) & (w < v.H - 2)
        idx, u, w = idx[inside], u[inside], w[inside]
        zi = buf[np.floor(w / ZBUF_REDUCE).astype(np.int64) * wz + np.floor(u / ZBUF_REDUCE).astype(np.int64)]
        out.append(idx[z[idx] <= zi + tol])
    return out


def agreement(points: np.ndarray, views: list[_View], seen: list[np.ndarray]) -> tuple[float, int]:
    """(photo agreement, points seen by at least 2 photos), for the points each photo sees (`visible`). Each photo's
    brightness is evened out first (phones change the exposure from shot to shot)."""
    n = len(points)
    samples = []
    for v, idx in zip(views, seen):
        if len(idx) < 20:
            continue
        Pc = points[idx] @ v.W[:3, :3].T + v.W[:3, 3]
        z = np.maximum(Pc[:, 2], 1e-9)
        u = v.K[0, 0] * Pc[:, 0] / z + v.K[0, 2]
        w = v.K[1, 1] * Pc[:, 1] / z + v.K[1, 2]
        ok = (u >= 0) & (u <= v.Wd - 1) & (w >= 0) & (w <= v.H - 1)
        col = v.img[np.round(w[ok]).astype(np.int64), np.round(u[ok]).astype(np.int64)]
        samples.append((idx[ok], col))
    if len(samples) < 2:
        return 0.0, 0
    levels = [float(np.median(col @ [0.299, 0.587, 0.114])) for _, col in samples]
    ref = float(np.median(levels))
    s1, s2, cnt = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
    for (idx, col), level in zip(samples, levels):
        col = col * (ref / max(level, 1e-6))
        np.add.at(s1, idx, col)
        np.add.at(s2, idx, col * col)
        np.add.at(cnt, idx, 1)
    seen2 = cnt >= 2
    if seen2.sum() < 50:
        return 0.0, int(seen2.sum())
    mean = s1[seen2] / cnt[seen2, None]
    var_point = np.clip(s2[seen2] / cnt[seen2, None] - mean ** 2, 0, None).mean(1)
    total = cnt[seen2].sum()
    all_mean = s1[seen2].sum(0) / total
    var_all = float(np.clip(s2[seen2].sum(0) / total - all_mean ** 2, 0, None).mean())
    return float(1.0 - var_point.mean() / max(var_all, 1e-12)), int(seen2.sum())


# --------------------------------------------------------------------------- moving the model (= the cameras)
def _move(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (points - c) @ (s * R).T + c + t


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    return o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray(axis, float) * math.radians(deg))


def _angle(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def moved_cameras(views: list[CameraView], s: float, R: np.ndarray, t: np.ndarray, c: np.ndarray) -> list[CameraView]:
    """The cameras that see the model where the moved model was seen: x' = s R (x - c) + c + t, so a camera
    [Rw|tw] becomes [Rw R | Rw (c + t - s R c) + tw] / s (pixels do not change when a camera frame is scaled)."""
    out = []
    for v in views:
        W = np.asarray(v.world_to_camera, float)
        Wn = np.eye(4)
        Wn[:3, :3] = W[:3, :3] @ R
        Wn[:3, 3] = (W[:3, :3] @ (c + t - s * R @ c) + W[:3, 3]) / s
        out.append(CameraView(v.image_path, v.K, Wn, v.width, v.height))
    return out


def _centre(W) -> np.ndarray:
    W = np.asarray(W, float)
    return -W[:3, :3].T @ W[:3, 3]


def check_placement(points: np.ndarray, normals: np.ndarray, views: list[CameraView], log: Log = print,
                    progress=None, spacing: float | None = None, seed: int = 0,
                    camera_points: np.ndarray | None = None, camera_tol: float | None = None) -> dict:
    """Checks, and repairs, where the photos were placed on the model (cameras in the model's frame).

    Returns {views (the repaired cameras), agreement (after the repair), agreement_before, typical_wrong (the median
    score of clearly different placements: what "wrong" looks like for this part and these photos), margin
    (agreement - typical_wrong), moved_mm, moved_deg, scale_change_pct (always 0: the size is COLMAP's), rival_deg
    (a turn > 15 deg away agreeing
    almost as well: the part looks alike from several sides), geometry_fit / geometry_fit_before (share of the
    reconstruction's exact points on the part that lie on the model, when camera_points are given)}.

    camera_points: the reconstruction's own points on the part, in the model's frame (as placed). A placement the
    photos prefer must keep them on the model too: one-side photos "agreed" best with the part upside down."""
    progress = progress or (lambda *a: None)
    rng = np.random.default_rng(seed)
    P = np.asarray(points, float)
    N = np.asarray(normals, float)
    c = P.mean(axis=0)
    _, _, axes = np.linalg.svd((P - c)[:: max(1, len(P) // 50_000)], full_matrices=False)
    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    tol = max(3 * (spacing or 0.0), 0.004 * diag)
    zb = P[rng.choice(len(P), min(len(P), ZBUF_POINTS), replace=False)]
    pick = rng.choice(len(P), min(len(P), SCORE_POINTS), replace=False)
    Ps, Ns = P[pick], N[pick]
    all_views = load_views(views)
    search = _spread(all_views, MAX_VIEWS)
    geo = None
    if camera_points is not None and len(camera_points) >= 30:
        from scipy.spatial import cKDTree

        tree = cKDTree(P[:: max(1, len(P) // 400_000)])
        cp = np.asarray(camera_points, float)
        cp = cp[rng.choice(len(cp), min(len(cp), 5_000), replace=False)]
        gtol = camera_tol or tol

        def geo(s, R, t):   # the exact points moved with the cameras: x = R^T (x' - c - t) / s + c
            d, _ = tree.query((cp - c - t) @ R / s + c, k=1, distance_upper_bound=gtol)
            return float(np.isfinite(d).mean())

    def seen_at(s, R, t, vs):
        return visible(_move(Ps, s, R, t, c), Ns @ R.T, vs, _move(zb, s, R, t, c), tol)

    def score(s, R, t, vs, seen=None):
        return agreement(_move(Ps, s, R, t, c), vs, seen if seen is not None else seen_at(s, R, t, vs))[0]

    before = score(1.0, np.eye(3), np.zeros(3), all_views)
    # 1. the placement as found, and the part turned about each of its axes: a hole pattern or a round or flat part
    #    can be placed turned. The median score of the turns is what a wrong placement looks like.
    turns = [(score(1.0, _rot(axes[k], a), np.zeros(3), search), _rot(axes[k], a))
             for k in range(3) for a in np.arange(TURN_STEP, 360.0, TURN_STEP)]
    progress(0.35)
    typical_wrong = float(np.median([v for v, _ in turns]))
    turns.sort(key=lambda x: -x[0])
    found = score(1.0, np.eye(3), np.zeros(3), search)
    geo_before = geo(1.0, np.eye(3), np.zeros(3)) if geo else None
    starts = [np.eye(3)] + [R for v, R in turns[:2] if v > found and (geo is None or geo(1.0, R, np.zeros(3))
                                                                      >= 0.9 * geo_before)]

    # 2. the repair: turn, shift and size in shrinking steps while the agreement improves, staying near the start.
    #    What each photo sees is worked out once per step size (a trial move barely changes it).
    results = []
    for i, R0 in enumerate(starts):
        s, R, t = 1.0, R0, np.zeros(3)
        for level in LEVELS:
            seen = seen_at(s, R, t, search)
            best = score(s, R, t, search, seen)
            improved = True
            while improved:
                improved = False
                for k in range(6):
                    for sign in (1.0, -1.0):
                        if k < 3:
                            cand = (s, _rot(axes[k], sign * level) @ R, t)
                        else:
                            cand = (s, R, t + sign * level * 0.01 * diag * axes[k - 3])
                        if _angle(cand[1] @ R0.T) > MAX_TURN or np.linalg.norm(cand[2]) > 0.25 * diag:
                            continue
                        val = score(*cand, search, seen)
                        if val > best + 1e-4:
                            best, (s, R, t), improved = val, cand, True
                            break
                    if improved:
                        break
        fit = geo(s, R, t) if geo else None
        if fit is not None and fit < 0.9 * geo_before and i > 0:
            continue   # the photos agree, the exact points do not: not this one
        results.append({"s": s, "R": R, "t": t, "agreement": score(s, R, t, all_views), "geo": fit})
        progress(0.35 + 0.6 * (i + 1) / len(starts))
    # the placement as found (repaired) stays unless a turned one agrees clearly better
    found = results[0]
    results.sort(key=lambda r: -r["agreement"])
    best = results[0]
    if best is not found and best["agreement"] < found["agreement"] + TURNED_MARGIN:
        best = found
    rival = next((round(_angle(r["R"] @ best["R"].T), 1) for r in results[1:]
                  if _angle(r["R"] @ best["R"].T) > 15 and r["agreement"] >= best["agreement"] - 0.02), None)
    if best["agreement"] < before + MIN_GAIN:
        # no clear gain: keep the placement as found (COLMAP's exact points place a well-seen part to well under a
        # pixel; the photos, read small, would only move it by pixel steps)
        best = {"s": 1.0, "R": np.eye(3), "t": np.zeros(3), "agreement": before, "geo": geo_before}
    s, R, t = best["s"], best["R"], best["t"]
    moved = moved_cameras(views, s, R, t, c)
    shift = [float(np.linalg.norm(_centre(a.world_to_camera) - _centre(b.world_to_camera))) for a, b in zip(moved, views)]
    report = {"views": moved, "agreement": round(best["agreement"], 4), "agreement_before": round(before, 4),
              "typical_wrong": round(typical_wrong, 4), "margin": round(best["agreement"] - typical_wrong, 4),
              "moved_deg": round(_angle(R), 3), "moved_mm": round(float(np.median(shift)), 3),
              "scale_change_pct": round((s - 1) * 100, 3), "rival_deg": rival,
              "geometry_fit": None if best["geo"] is None else round(best["geo"], 4),
              "geometry_fit_before": None if geo_before is None else round(geo_before, 4)}
    log(f"Checked the placement against the photos themselves: agreement {before:.2f} -> {best['agreement']:.2f} "
        f"(a clearly wrong placement scores ~{typical_wrong:.2f}); cameras moved {report['moved_mm']:.2f} mm, "
        f"turned {report['moved_deg']:.2f} deg, size {report['scale_change_pct']:+.2f} %")
    progress(1.0)
    return report
