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
    threshold_above: float = 25.0  # a marker is this much brighter than the image's median (the background). At
                                   # Revo Metro's 200 us exposure near markers read ~100-250 over a background of 16,
                                   # but markers further away and seen at a low angle (the far side of a turntable
                                   # plate) return far less light (retro-reflection falls off with the angle) and
                                   # read ~60-110. On the user's plate + bust recording (2026-10-06): 40 paired 10.2
                                   # markers a frame, 25 paired 17.8 (map 13 -> 23, pose jitter at the surface
                                   # 0.141 -> 0.058 mm RMS, all on the plate within 0.22 mm, p95); 20 began to take
                                   # laser-lit spots on the bust (32 of 700 off the plate)
    threshold_min: float = 40.0
    stripe_width_px: int = 9      # laser stripes are 5-8 px wide: an opening this wide removes them
    min_area_px: int = 40
    max_area_px: int = 6000
    min_fill: float = 0.6         # blob area / area of its fitted ellipse (Revo: saturation)
    min_axis_ratio: float = 0.25  # minor / major axis (Revo: ellipse_ab / minAb). A round marker seen 75 deg off its
                                  # normal is 0.26: the far markers of a plate viewed from the side are that flat
    epipolar_px: float = 1.5      # rectified row difference of a stereo pair (Revo: 1.0..1.5)
    size_ratio: float = 1.6       # left/right blob size may differ by this factor at most
    min_diameter_mm: float = 2.5  # a paired blob's size in mm (area-equivalent, at its triangulated depth). The
                                  # MetroY takes 6 mm markers only; seen at an angle they measure 3.3-5.5 mm
                                  # (p1-p99, the user's plate, 2026-10-09), laser spots on a white part ~1.4 mm
                                  # (median): 2.5 keeps every plate marker and drops 92 % of the spots
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
    return stereo_pairs(left, right, Q, p)[0]


def stereo_pairs(left: list[Blob], right: list[Blob], Q: np.ndarray, p: MarkerParams | None = None):
    """stereo(), plus which blobs made the pairs: (points (n, 3), left indices (n,), right indices (n,)). A blob in
    neither list was found in one picture but not matched in the other - the camera view shows it apart."""
    p = p or MarkerParams()
    none = np.zeros(0, int)
    if not left or not right:
        return np.zeros((0, 3)), none, none
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
        return np.zeros((0, 3)), none, none
    i, j = np.array(pairs).T
    y = 0.5 * (L[i, 1] + R[j, 1])
    h = np.c_[L[i, 0], y, L[i, 0] - R[j, 0], np.ones(len(i))] @ Q.T
    P = h[:, :3] / h[:, 3:4]
    # a marker has a known physical size: bright laser spots on a light part pair across the cameras too (they are
    # real surface points), but they are a millimetre or so across, and a marker map that takes them in follows the
    # laser instead of the part
    big = 2.0 * L[i, 2] * P[:, 2] / Q[2, 3] >= p.min_diameter_mm      # Q[2, 3] = f
    return P[big], i[big], j[big]


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
    forget_after: int = 10        # lost frames in a row after which the motion prediction is no longer trusted
    frozen: bool = False          # a finished marker map: registered markers no longer move it, none are added
    initial: np.ndarray | None = None
    world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    seen: np.ndarray = field(default_factory=lambda: np.zeros(0))
    version: int = 0              # bumps whenever the map changes (for display)
    _candidates: list = field(default_factory=list)
    _obs: list = field(default_factory=list)        # (frame markers, frame->map index) for refine()
    _T1: np.ndarray | None = None
    _T2: np.ndarray | None = None
    _lost_run: int = 0
    _tri_cache: tuple | None = None

    def reset(self) -> None:
        self.world = np.zeros((0, 3))
        self.seen = np.zeros(0)
        self._candidates, self._obs = [], []
        self._T1 = self._T2 = None
        self._lost_run = 0
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
            self._lost_run += 1
            if self._lost_run >= self.forget_after:
                self._T1 = self._T2 = None           # the motion since is unknown: recover from the markers alone
            return TrackResult(None, "lost", m, 0, None)
        self._lost_run = 0
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

    def _accept(self, markers: np.ndarray, idx: np.ndarray, need: int, T: np.ndarray) -> bool:
        """Enough markers register, they pin the pose down, and most markers that could have registered did.

        A frame marker far from every map marker is simply not mapped yet: it is no evidence against the pose. Only
        one that lands within the association gate of a map marker without matching it contradicts the pose (two
        real markers never sit 0.5-5 mm apart). Counting every unregistered marker against the pose, as before,
        deadlocked a map started on a frame with few markers: a 4-marker map could never register half of a
        10-marker frame, and only registered frames may add markers (the user's turntable scan, 2026-10-09)."""
        good = idx >= 0
        n = int(good.sum())
        if n < need or _degenerate(markers[good]):
            return False
        if self.frozen or n >= self.min_agree * len(markers):
            # a finished map holds every marker there is: whatever does not register counts against the pose
            return n >= self.min_agree * len(markers)
        rest = apply(T, markers[~good])
        against = int((np.linalg.norm(rest[:, None] - self.world[None], axis=2).min(1) < self.gate_mm).sum()) \
            if len(rest) else 0
        if n < self.min_agree * (n + against) or n < self.min_inliers_reloc:
            return False
        # most of the frame is not explained (new markers coming into view of a small map): take the pose only if
        # the layout pins it down. On a grid or a ring a pose one spacing off registers as many markers and puts the
        # rest in empty map space (review 2026-10-09: 30 mm and 131-228 mm wrong poses accepted from a stale
        # prediction); the triangle search finds the pose that explains most, and refuses when two different ones
        # do equally well
        alt = self._relocalize(markers, use_prediction=False)   # the pose under test came from the prediction
        return alt is not None and _pose_distance(alt, T) < (2.0, 1.0)

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
        if not self._accept(markers, idx, need, T):
            return None, idx
        good = idx >= 0
        return kabsch(markers[good], self.world[idx[good]]), idx

    def _map_triangles(self, max_side: float):
        """The map's triangles, sorted by their shortest side (a frame triangle's partners are a binary search
        away instead of a scan of every map triangle)."""
        key = (self.version, len(self.world), round(max_side))
        if self._tri_cache is None or self._tri_cache[0] != key:
            keys, tris = _triangles(self.world, self.min_side_mm, max_side=max_side)
            o = np.argsort(keys[:, 0], kind="stable")
            self._tri_cache = (key, (keys[o], [tris[i] for i in o]))
        return self._tri_cache[1]

    RELOC_TRIANGLES = 200   # frame triangles tried, the largest first: a few already find a random layout's pose,
                            # and on a symmetric one (ring, grid) any of them meets every rotated / shifted copy

    def _relocalize(self, markers: np.ndarray, use_prediction: bool = True) -> np.ndarray | None:
        """Pose from triangles of inter-marker distances (which the pose does not change), accepted only when it
        is clearly better than any other pose - on a regular marker grid several poses fit a few markers.

        Every triangle match used to be fitted and associated, so a 25-marker frame against a 38-marker map took
        1.5 s - on the capture thread, for every lost frame. Now the map's triangles are looked up in a sorted
        list, only the largest frame triangles are tried, and a triangle pose that repeats a pose already found
        (within 10 mm / 3 deg - the copies a symmetric layout produces differ by a marker spacing or a turn) is
        not fitted again."""
        if len(self.world) < 3 or len(markers) < self.min_inliers_reloc:
            return None
        extent = float(np.max(np.linalg.norm(markers[:, None] - markers[None], axis=2)))
        keys_map, tri_map = self._map_triangles(extent + 2 * self.side_tol_mm)
        if not len(keys_map):
            return None
        ft_keys, ft = _triangles(markers, self.min_side_mm)
        if not len(ft_keys):
            return None
        order = np.argsort(-ft_keys[:, 0], kind="stable")[:self.RELOC_TRIANGLES]
        first = keys_map[:, 0]
        cands = []
        for key, tri in zip(ft_keys[order], [ft[i] for i in order]):
            lo = np.searchsorted(first, key[0] - self.side_tol_mm, "left")
            hi = np.searchsorted(first, key[0] + self.side_tol_mm, "right")
            close = lo + np.flatnonzero(np.all(np.abs(keys_map[lo:hi] - key) < self.side_tol_mm, axis=1))
            for c in close[:60]:
                T = kabsch(markers[list(tri)], self.world[list(tri_map[c])])
                if any(_pose_distance(T, Tc) < (10.0, 3.0) for _, Tc in cands):
                    continue
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
        prediction = self._T1 if use_prediction else None
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
        if len(fresh) and len(self.world):
            # within the gate of a map marker it IS that marker, a little off (two markers never sit 5 mm apart):
            # re-adding it as new made a copy of every plate marker on each turn of the table (444 markers for ~30)
            near = np.linalg.norm(fresh[:, None] - self.world[None], axis=2).min(1) < self.gate_mm
            fresh = fresh[~near]
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
    """Every triangle of markers with sides in [min_side, max_side]: its sides sorted (the pose-free key) and its
    vertices opposite those sides, in combination order, at most `limit`. Vectorised: the loop it replaces spent
    ~60 ms on 25 markers, most of a relocalization."""
    n = len(P)
    if n < 3:
        return np.zeros((0, 3)), []
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    abc = np.fromiter(combinations(range(n), 3), dtype=np.dtype((np.int64, 3)))
    a, b, c = abc.T
    sides = np.c_[D[b, c], D[a, c], D[a, b]]                     # side k is opposite vertex k
    ok = (sides.min(1) >= min_side) & (sides.max(1) <= max_side)
    sides, abc = sides[ok][:limit], abc[ok][:limit]
    o = np.argsort(sides, axis=1, kind="stable")                  # canonical order: vertices opposite sorted sides
    keys = np.take_along_axis(sides, o, axis=1)
    verts = np.take_along_axis(abc, o, axis=1)
    return keys.reshape(-1, 3), [tuple(v) for v in verts.tolist()]
