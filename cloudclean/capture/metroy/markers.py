"""Retro-reflective markers: detection in the rectified IR views, stereo matching and 3D, and marker-map tracking.

In laser mode Revo Metro tracks the scanner by markers only (docs/revo-metro-internals.md, section 7): a laser frame
holds ~17 stripes, far too sparse for geometric (ICP) tracking to hold, but a handful of markers stuck on or around
the part pin the pose exactly. Markers are lit by the scanner's IR fill light (register 0xb07) and return far more
light than any diffuse surface, so they show as bright, filled ellipses.

Detection (per rectified view)
    threshold well above the background, open with a disk that is wider than a laser stripe (so stripes vanish and
    marker blobs stay), keep components that fill their bounding ellipse and are not too elongated, and locate each
    centre as the intensity-weighted centroid over the blob.
Stereo
    a marker's partner sits on the same rectified row (within `epipolar_px`) with a similar size, at a disparity that
    means a depth in range; only pairs that are unique both ways are kept, then triangulated through Q.
Tracking (MarkerTracker)
    a global marker map in world coordinates. Each frame is registered to it by nearest-marker association from the
    motion-predicted pose and a least-squares rigid fit (Kabsch); when that fails, by matching triangles of
    inter-marker distances (which do not depend on the pose) and a RANSAC fit, as Revo Metro's markerAlignByTriangle
    does. Registered markers refine the map; new ones join it once seen consistently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np


@dataclass
class MarkerParams:
    threshold_above: float = 40.0  # a marker is this much brighter than the image's median (the background). At
                                   # Revo Metro's 200 us exposure lit markers read ~100-130 over a background of 16;
                                   # a fixed level misses them when exposure changes
    threshold_min: float = 45.0
    stripe_width_px: int = 9      # laser stripes are 5-8 px wide: an opening this wide removes them
    min_area_px: int = 40
    max_area_px: int = 6000
    min_fill: float = 0.6         # blob area / area of its fitted ellipse (Revo: saturation)
    min_axis_ratio: float = 0.4   # minor / major axis (Revo: ellipse_ab / minAb)
    epipolar_px: float = 1.5      # rectified row difference of a stereo pair (Revo: 1.0..1.5)
    size_ratio: float = 1.6       # left/right blob size may differ by this factor at most
    z_min: float = 150.0
    z_max: float = 500.0


@dataclass
class Blob:
    x: float
    y: float
    radius: float                 # equivalent radius, px
    axis_ratio: float


def detect(img: np.ndarray, p: MarkerParams | None = None) -> list[Blob]:
    p = p or MarkerParams()
    threshold = max(p.threshold_min, float(np.median(img[::8, ::8])) + p.threshold_above)
    bright = (img >= threshold).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.stripe_width_px, p.stripe_width_px))
    core = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    out = []
    img_f = img.astype(np.float32)
    for i in range(1, n):
        x0, y0, w, h, area = st[i]
        if not (p.min_area_px <= area <= p.max_area_px):
            continue
        mask = (lab[y0:y0 + h, x0:x0 + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts or len(cnts[0]) < 6:
            continue
        (_, _), (a, b), _ = cv2.fitEllipse(cnts[0])
        major, minor = max(a, b), min(a, b)
        if major <= 0 or minor / major < p.min_axis_ratio:
            continue
        if area / (np.pi * a * b / 4.0) < p.min_fill:
            continue
        # intensity-weighted centroid over the blob grown by a pixel (captures the soft edge symmetrically)
        grown = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        pad = 1
        ya, yb = max(0, y0 - pad), min(img.shape[0], y0 + h + pad)
        xa, xb = max(0, x0 - pad), min(img.shape[1], x0 + w + pad)
        m = np.zeros((yb - ya, xb - xa), np.float32)
        m[y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w] = grown[:, :]
        wts = np.clip(img_f[ya:yb, xa:xb] - threshold * 0.5, 0, None) * m
        s = wts.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        out.append(Blob(float((wts * xx).sum() / s), float((wts * yy).sum() / s),
                        float(np.sqrt(area / np.pi)), float(minor / major)))
    return out


def mask(shape, blobs: list[Blob], grow: float = 1.8, pad_px: float = 4.0) -> np.ndarray:
    """Pixels covered by markers (a little beyond their rim): a marker's bright disc must never be taken for
    laser stripe, or every marker turns into a dense clump of false surface points."""
    m = np.zeros(shape, np.uint8)
    for b in blobs:
        cv2.circle(m, (int(round(b.x)), int(round(b.y))), int(np.ceil(b.radius * grow + pad_px)), 1, -1)
    return m.astype(bool)


def stereo(left: list[Blob], right: list[Blob], Q: np.ndarray, p: MarkerParams | None = None) -> np.ndarray:
    """(n, 3) marker positions in mm (rectified left camera frame) from unique left/right pairs."""
    p = p or MarkerParams()
    if not left or not right:
        return np.zeros((0, 3))
    L = np.array([(b.x, b.y, b.radius) for b in left])
    R = np.array([(b.x, b.y, b.radius) for b in right])
    dy = np.abs(L[:, None, 1] - R[None, :, 1])
    ratio = np.maximum(L[:, None, 2], R[None, :, 2]) / np.maximum(1e-6, np.minimum(L[:, None, 2], R[None, :, 2]))
    d = L[:, None, 0] - R[None, :, 0]
    w = d * Q[3, 2] + Q[3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(w != 0, Q[2, 3] / w, 0.0)
    ok = (dy < p.epipolar_px) & (ratio < p.size_ratio) & (z > p.z_min) & (z < p.z_max)
    pairs = [(i, int(np.flatnonzero(ok[i])[0])) for i in range(len(L))
             if ok[i].sum() == 1 and ok[:, np.flatnonzero(ok[i])[0]].sum() == 1]
    if not pairs:
        return np.zeros((0, 3))
    i, j = np.array(pairs).T
    y = 0.5 * (L[i, 1] + R[j, 1])
    h = np.c_[L[i, 0], y, L[i, 0] - R[j, 0], np.ones(len(i))] @ Q.T
    return h[:, :3] / h[:, 3:4]


# -- tracking ----------------------------------------------------------------------------------------------------

def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid 4x4 T minimising |T(src) - dst| (no scale: the markers are metric)."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cd - R @ cs
    return T


def apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


@dataclass
class TrackResult:
    pose: np.ndarray | None       # sensor -> world, None when the frame could not be registered
    state: str                    # "init", "tracked", "relocalized", "lost", "no_markers"
    markers: int                  # markers seen in the frame
    inliers: int                  # of them, registered to the map
    rmse_mm: float | None


@dataclass
class MarkerTracker:
    """Registers each frame's markers to a world marker map.

    Two ways to use it, mirroring Revo Metro:

    * map as you go: the map grows while scanning (new markers join after `confirm` consistent sightings);
    * map first ("marker scan"): sweep once over all markers without fusing surface (`frozen` False), then
      `refine()` the whole map at once and freeze it; the surface scan then tracks against a rigid map that a bad
      frame can never corrupt.

    Wrong registrations are what fuse a surface twice at an offset, so a pose is only accepted with evidence: at
    least `min_inliers` markers within `inlier_mm`, most of the frame's markers agreeing, and - when the pose is
    recovered from scratch - no second, different pose explaining the markers nearly as well (markers laid out in a
    regular grid fit in several places).
    """
    gate_mm: float = 5.0          # association radius after prediction (hand-held motion reaches ~3 mm per frame;
                                  # markers sit 20-40 mm apart)
    inlier_mm: float = 0.5        # a registered marker must land this close to its map position
    min_inliers: int = 3          # while tracking continuously from the predicted pose
    min_inliers_reloc: int = 4    # when recovering the pose from scratch
    min_agree: float = 0.5        # fraction of the frame's markers that must register
    side_tol_mm: float = 0.6      # triangle side lengths must agree this well to be a candidate match
    min_side_mm: float = 8.0      # triangles with shorter sides pin nothing down
    confirm: int = 3              # sightings before a new marker joins the map
    frozen: bool = False          # a finished marker map: registered markers no longer move it, none are added
    initial: np.ndarray | None = None
    world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    seen: np.ndarray = field(default_factory=lambda: np.zeros(0))
    version: int = 0              # bumps whenever the map changes (for display)
    _candidates: list = field(default_factory=list)
    _obs: list = field(default_factory=list)        # (frame markers, frame->map index) for refine()
    _T1: np.ndarray | None = None
    _T2: np.ndarray | None = None
    _tri_cache: tuple | None = None

    def reset(self) -> None:
        self.world = np.zeros((0, 3))
        self.seen = np.zeros(0)
        self._candidates, self._obs = [], []
        self._T1 = self._T2 = None
        self._tri_cache = None
        self.frozen = False
        self.version += 1

    def track(self, markers: np.ndarray) -> TrackResult:
        m = len(markers)
        if len(self.world) == 0:
            if m < self.min_inliers_reloc or _degenerate(markers):
                return TrackResult(None, "no_markers", m, 0, None)
            T = self.initial if self.initial is not None else np.eye(4)
            self.world = apply(T, markers)
            self.seen = np.ones(m)
            self._obs.append((markers.copy(), np.arange(m)))
            self._T2, self._T1 = self._T1, T
            self.version += 1
            return TrackResult(T, "init", m, m, 0.0)
        if m < self.min_inliers:
            return TrackResult(None, "no_markers", m, 0, None)

        state, T, idx = "tracked", None, None
        for guess in self._predictions():
            T, idx = self._refine(markers, guess, self.min_inliers)
            if T is not None:
                break
        if T is None:
            state = "relocalized"
            T = self._relocalize(markers)
            if T is not None:
                T, idx = self._refine(markers, T, self.min_inliers_reloc)
        if T is None:
            return TrackResult(None, "lost", m, 0, None)
        good = idx >= 0
        W = apply(T, markers)
        rmse = float(np.sqrt(np.mean(np.sum((W[good] - self.world[idx[good]]) ** 2, 1))))
        if not self.frozen:
            self._update_map(W, idx, well_posed=good.sum() >= self.min_inliers_reloc and rmse < 0.3)
            if len(self._obs) < 4000:
                self._obs.append((markers.copy(), idx.copy()))
        self._T2, self._T1 = self._T1, T
        return TrackResult(T, state, m, int(good.sum()), rmse)

    def refine(self, iterations: int = 10) -> float:
        """Whole-map adjustment over every frame recorded while mapping: alternately re-solve each frame's pose
        from its markers and each marker from all its sightings. Spreads what incremental mapping accumulated
        (drift) over the map instead of leaving it where the sweep ended. Returns the final RMS residual, mm."""
        obs = [(M, idx) for M, idx in self._obs if (idx >= 0).sum() >= self.min_inliers_reloc]
        rms = float("nan")
        if not obs or len(self.world) < 3:
            return rms
        for _ in range(iterations):
            sums = np.zeros_like(self.world)
            counts = np.zeros(len(self.world))
            poses = []
            for M, idx in obs:
                good = idx >= 0
                T = kabsch(M[good], self.world[idx[good]])
                poses.append(T)
                np.add.at(sums, idx[good], apply(T, M[good]))
                np.add.at(counts, idx[good], 1)
            seen = counts > 0
            self.world[seen] = sums[seen] / counts[seen, None]
            res = [apply(T, M[idx >= 0]) - self.world[idx[idx >= 0]] for (M, idx), T in zip(obs, poses)]
            rms = float(np.sqrt(np.mean(np.concatenate(res) ** 2) * 3))
        self._tri_cache = None
        self.version += 1
        return rms

    def freeze(self) -> dict:
        rms = self.refine()
        self.frozen = True
        self._candidates = []
        return {"markers": int(len(self.world)), "frames": len(self._obs), "rms_mm": rms}

    # -- internals
    def _predictions(self):
        if self._T1 is None:
            return []
        if self._T2 is None:
            return [self._T1]
        motion = self._T1 @ np.linalg.inv(self._T2)
        return [motion @ self._T1, self._T1]

    def _associate(self, W: np.ndarray, radius: float) -> np.ndarray:
        d = np.linalg.norm(W[:, None, :] - self.world[None, :, :], axis=2)
        idx = np.argmin(d, 1)
        idx[d[np.arange(len(W)), idx] > radius] = -1
        # one map marker per frame marker: drop the farther claims
        for j in np.unique(idx[idx >= 0]):
            claim = np.flatnonzero(idx == j)
            if len(claim) > 1:
                keep = claim[np.argmin(d[claim, j])]
                idx[claim[claim != keep]] = -1
        return idx

    def _accept(self, markers: np.ndarray, idx: np.ndarray, need: int) -> bool:
        good = idx >= 0
        return good.sum() >= need and good.sum() >= self.min_agree * len(markers) and not _degenerate(markers[good])

    def _refine(self, markers: np.ndarray, T: np.ndarray, need: int):
        radius = self.gate_mm
        for _ in range(4):
            idx = self._associate(apply(T, markers), radius)
            good = idx >= 0
            if good.sum() < need or _degenerate(markers[good]):
                return None, idx
            T = kabsch(markers[good], self.world[idx[good]])
            radius = max(self.inlier_mm * 2, radius * 0.5)
        idx = self._associate(apply(T, markers), self.inlier_mm)
        if not self._accept(markers, idx, need):
            return None, idx
        good = idx >= 0
        return kabsch(markers[good], self.world[idx[good]]), idx

    def _map_triangles(self, max_side: float):
        key = (self.version, len(self.world), round(max_side))
        if self._tri_cache is None or self._tri_cache[0] != key:
            self._tri_cache = (key, _triangles(self.world, self.min_side_mm, max_side=max_side))
        return self._tri_cache[1]

    def _relocalize(self, markers: np.ndarray) -> np.ndarray | None:
        """Pose from triangles of inter-marker distances (which the pose does not change), accepted only when it
        is clearly better than any other pose - on a regular marker grid several poses fit a few markers."""
        if len(self.world) < 3 or len(markers) < self.min_inliers_reloc:
            return None
        extent = float(np.max(np.linalg.norm(markers[:, None] - markers[None], axis=2)))
        keys_map, tri_map = self._map_triangles(extent + 2 * self.side_tol_mm)
        if not len(keys_map):
            return None
        ft_keys, ft = _triangles(markers, self.min_side_mm)
        cands = []
        for key, tri in zip(ft_keys, ft):
            close = np.flatnonzero(np.all(np.abs(keys_map - key) < self.side_tol_mm, axis=1))
            for c in close[:60]:
                T = kabsch(markers[list(tri)], self.world[list(tri_map[c])])
                idx = self._associate(apply(T, markers), self.inlier_mm * 2)
                n = int((idx >= 0).sum())
                if n >= self.min_inliers_reloc:
                    # compare poses fitted to ALL their inliers: from one small triangle, noise alone moves a pose
                    # by millimetres, which would make one true pose look like two
                    for _ in range(3):              # to convergence: the inlier set settles in 1-2 rounds
                        good = idx >= 0
                        T = kabsch(markers[good], self.world[idx[good]])
                        idx = self._associate(apply(T, markers), self.inlier_mm * 2)
                    cands.append((int((idx >= 0).sum()), T))
        if not cands:
            return None
        cands.sort(key=lambda c: -c[0])
        best_n, best = cands[0]
        prediction = self._T1
        full = best_n == len(markers)             # explains every marker in view
        for n, T in cands[1:]:
            if n < best_n - (0 if full else 1):     # a full explanation beats any that leaves a marker out
                break
            if _pose_distance(T, best) > (2.0, 1.0):                    # a genuinely different pose fits as well
                if prediction is not None and _pose_distance(best, prediction) < (20.0, 10.0) \
                        and not _pose_distance(T, prediction) < (20.0, 10.0):
                    continue                                              # the motion so far settles it
                return None
        return best

    def _update_map(self, W: np.ndarray, idx: np.ndarray, well_posed: bool) -> None:
        good = idx >= 0
        j = idx[good]
        w = self.seen[j]
        self.world[j] = (self.world[j] * w[:, None] + W[good]) / (w + 1)[:, None]
        self.seen[j] = np.minimum(w + 1, 50)       # cap: the map keeps adapting slowly
        if not well_posed:
            return                                  # only a firmly registered frame may introduce markers
        fresh = W[~good]
        kept = []
        for c in self._candidates:
            d = np.linalg.norm(fresh - c[0], axis=1) if len(fresh) else np.zeros(0)
            if len(d) and d.min() < self.inlier_mm * 2:
                k = int(np.argmin(d))
                c = ((c[0] * c[1] + fresh[k]) / (c[1] + 1), c[1] + 1)
                fresh = np.delete(fresh, k, axis=0)
            kept.append(c)
        kept += [(f, 1) for f in fresh]
        promote = [c for c in kept if c[1] >= self.confirm]
        if promote:
            self.world = np.vstack([self.world, [c[0] for c in promote]])
            self.seen = np.r_[self.seen, [c[1] for c in promote]]
            self.version += 1
        self._candidates = [c for c in kept if c[1] < self.confirm][-200:]


class _Dist(tuple):
    def __gt__(self, other):          # "farther than" in either translation (mm) or rotation (deg)
        return self[0] > other[0] or self[1] > other[1]

    def __lt__(self, other):
        return self[0] < other[0] and self[1] < other[1]


def _pose_distance(A: np.ndarray, B: np.ndarray) -> "_Dist":
    dR = A[:3, :3].T @ B[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return _Dist((float(np.linalg.norm(A[:3, 3] - B[:3, 3])), float(angle)))


def _degenerate(P: np.ndarray) -> bool:
    """Three or more markers on (nearly) one line leave a rotation free."""
    if len(P) < 3:
        return True
    s = np.linalg.svd(P - P.mean(0), compute_uv=False)
    return s[1] < 2.0          # mm: the second extent of the set must be real


def _triangles(P: np.ndarray, min_side: float, limit: int = 40000, max_side: float = np.inf):
    keys, tris = [], []
    for tri in combinations(range(len(P)), 3):
        a, b, c = P[list(tri)]
        sides = np.array([np.linalg.norm(b - c), np.linalg.norm(a - c), np.linalg.norm(a - b)])
        if sides.min() < min_side or sides.max() > max_side:
            continue
        o = np.argsort(sides)                    # canonical order: vertices opposite the sorted sides
        keys.append(sides[o])
        tris.append(tuple(np.array(tri)[o]))
        if len(keys) >= limit:
            break
    return np.array(keys).reshape(-1, 3), tris
