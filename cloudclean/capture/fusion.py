"""Averaging fusion: every fused point of every frame is summed into a sparse field of surface moments, and the saved
points are moved onto the surface those moments average out.

Why: keeping raw points (at most a few per 0.2 mm cell) keeps every frame's noise. On the MetroY that noise is mostly
tracking jitter - replaying the user's bust, each frame sat 0.038 mm (std) off the averaged surface as a whole, with
0.032 mm more within the frame; the scanner's own single-frame noise is 0.01-0.015 mm - and the bust came out 0.24 mm
thick (2 x p95 offset from a local plane over 0.6 mm). Every vendor averages instead (Creaform's vector field, Curless
& Levoy's weighted signed distance, docs/laser-scanning-research.md section 5).

How: per 0.2 mm cell (sparse, int64 keys as in grid.py) the field keeps the count and the first and second moments of
all points that fell in it, about the cell's centre (10 numbers, additive: adding a frame is a scatter-add, nothing is
resampled; 0.2 ms per 1.2k-point frame, 0.43 ms per 5k on the Spark, 22 MB for the bust). At save time the moments of
each cell's 3x3x3 block give the least-squares plane of everything measured there, corrected for curvature (a plane
lies inside a convex surface by about a^2 / 6R: 0.017 mm on a 1 mm radius, 0.0004 mm left after the correction), and
each kept point moves along the normal onto it. The points keep their number, density and place along the surface;
only the noise across the surface is averaged away.

Every point counts the same. Weighting by incidence angle (Curless & Levoy) was measured and left out: on the bust the
7 % of points seen at grazing angles (cos < 0.4) read 0.01-0.02 mm further from the scanner than the average and are a
little noisier (0.055-0.061 vs 0.048-0.052 mm RMS), so weighting them down would move the surface by under 0.002 mm.

Where the block is not one clean, smooth surface - a sharp edge or corner, two sides of a thin wall, a single stripe,
too few samples - the points stay exactly as measured, so averaging does not round an edge or merge two surfaces;
`project` counts what moved and what did not, and the capture report carries it.

Measured on the replayed bust (recording 20261009-114240, 6237 frames, 7.2M measured points, the same 630,369 points
saved; thickness and RMS over 0.6 mm patches as the user measured them; "halves": the even and the odd frames fused
apart, median / p95 distance of one to the other's local plane - independent noise, so it measures precision):

    fusion                               thickness med / p90   RMS     halves med / p95   points/mm2
    raw, 3 per 0.2 mm cell (old)         0.243 / 0.355         0.065   0.037 / 0.106      132
    voxel mean 0.05 mm (one per voxel)   0.226 / 0.318         0.059   0.034 / 0.105      527
    voxel mean 0.1 mm                    0.240 / 0.341         0.064   0.037 / 0.106      210
    this, fed only the kept points       0.096 / 0.202         0.025   0.013 / 0.054      134
    this, 0.15 mm cells                  0.094 / 0.244         0.025   0.012 / 0.060      134
    this, 0.2 mm cells (default)         0.075 / 0.162         0.020   0.0097 / 0.042     134
    this, 0.25 mm cells                  0.061 / 0.139         0.016   0.0085 / 0.034     134

Every point keeps a twin: 99.8 % of the old points have a new one within 0.2 mm and all new ones an old one within
0.3 mm (median move 0.041 mm, largest 0.39); the local surface at the bounding box's extremes stays within 0.001-0.009
mm of the old cloud's local mean (the old box was 0.07 mm taller in Y only because one noise spike stuck out 0.21 mm).

A voxel mean barely helps: a 0.05-0.1 mm voxel is thinner than the noise, so the noise just spreads over more voxels.
Fed only the points the cell cap keeps, the fit loses the frames the cap turned away. 0.25 mm cells average more but
widen the block to 0.75 mm (more curvature to correct, more of an edge left raw); 0.2 mm matches the MetroY's point
distance and is what the known-shape tests (tests/test_fusion.py) check."""
from __future__ import annotations

import numpy as np

from .grid import NEIGHBOR_OFFSETS, key_cells, key_centers, voxel_keys

FUSION_CELL_MM = 0.2      # see the table above: 0.6 mm blocks, the MetroY's point distance (coarser point distances
                          # use their own: the field then has no more cells than the session's point grid)

_E = np.empty(0, np.int64)


class _KeyIndex:
    """A stable slot number per voxel key: sorted main + small sorted delta (as grid.VoxelCounter), so a frame costs
    O(frame log map) and the moment arrays are append-only (never re-sorted or copied by an insert)."""

    def __init__(self):
        self.main = (_E, _E)
        self.delta = (_E, _E)
        self.n = 0

    def find(self, keys: np.ndarray) -> np.ndarray:
        out = np.full(len(keys), -1, np.int64)
        for k, s in (self.main, self.delta):
            if len(k):
                pos = np.minimum(np.searchsorted(k, keys), len(k) - 1)
                hit = k[pos] == keys
                out[hit] = s[pos[hit]]
        return out

    def slots(self, uniq: np.ndarray) -> np.ndarray:
        """Slots of sorted unique keys, creating the missing ones."""
        out = self.find(uniq)
        new = out < 0
        if new.any():
            nk = uniq[new]
            ns = np.arange(self.n, self.n + len(nk), dtype=np.int64)
            out[new] = ns
            self.n += len(nk)
            dk, ds = self.delta
            pos = np.searchsorted(dk, nk)
            self.delta = (np.insert(dk, pos, nk), np.insert(ds, pos, ns))
            if len(self.delta[0]) > max(50_000, len(self.main[0]) // 8):
                self.merge()
        return out

    def merge(self) -> None:
        (mk, ms), (dk, ds) = self.main, self.delta
        if len(dk):
            pos = np.searchsorted(mk, dk)
            self.main = (np.insert(mk, pos, dk), np.insert(ms, pos, ds))
            self.delta = (_E, _E)


# moments columns: w, sum x, y, z, sum xx, xy, xz, yy, yz, zz (relative to the cell's centre)
_W, _S1, _S2 = 0, slice(1, 4), slice(4, 10)
_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_D = np.array([-1.0, 0.0, 1.0])
_OFFSETS = np.stack(np.meshgrid(_D, _D, _D, indexing="ij"), axis=-1).reshape(-1, 3)   # cells, as NEIGHBOR_OFFSETS


class SurfaceField:
    def __init__(self, cell: float):
        self.cell = float(cell)
        self._index = _KeyIndex()
        self._m = np.zeros((0, 10), np.float64)
        self.points = 0

    def __len__(self) -> int:
        return self._index.n

    def nbytes(self) -> int:
        return self._m[:self._index.n].nbytes + 16 * self._index.n

    def add(self, points: np.ndarray) -> None:
        """Sum a frame's points (world mm) into their cells."""
        p = np.asarray(points, np.float64)
        if not len(p):
            return
        keys = voxel_keys(p, self.cell)
        local = p - (np.floor(p / self.cell) + 0.5) * self.cell
        cols = np.empty((len(p), 10))
        cols[:, _W] = 1.0
        cols[:, _S1] = local
        for j, (a, b) in enumerate(_PAIRS):
            cols[:, 4 + j] = cols[:, 1 + a] * local[:, b]
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        start = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
        sums = np.add.reduceat(cols[order], start, axis=0)
        slots = self._index.slots(sk[start])
        if self._index.n > len(self._m):
            grown = np.zeros((max(self._index.n, 2 * len(self._m), 65_536), 10))
            grown[:len(self._m)] = self._m
            self._m = grown
        self._m[slots] += sums
        self.points += len(p)

    def copy(self) -> "SurfaceField":
        """A frozen snapshot (merged index), cheap enough to take under the session lock."""
        out = SurfaceField(self.cell)
        self._index.merge()
        out._index.main = (self._index.main[0].copy(), self._index.main[1].copy())
        out._index.n = self._index.n
        out._m = self._m[:self._index.n].copy()
        out.points = self.points
        return out

    def _gather(self, keys: np.ndarray) -> np.ndarray:
        """(n, 27, 10) moments of the 3x3x3 block around each key, each about the key's cell centre."""
        out = np.zeros((len(keys), 27, 10))
        for j, (o, d) in enumerate(zip(NEIGHBOR_OFFSETS, _OFFSETS * self.cell)):
            slot = self._index.find(keys + o)
            hit = slot >= 0
            out[hit, j] = _shift(self._m[slot[hit]], d)
        return out

    def coarsen(self) -> "SurfaceField":
        """The same moments in cells twice the size (2 x 2 x 2 cells summed - exact, nothing resampled)."""
        self._index.merge()
        keys, slots = self._index.main
        cells = key_cells(keys)
        parent = np.floor_divide(cells, 2)
        m = _shift(self._m[slots], (cells + 0.5) * self.cell - (parent + 0.5) * 2 * self.cell)
        pkeys = voxel_keys((parent + 0.5) * 2 * self.cell, 2 * self.cell)
        order = np.argsort(pkeys, kind="stable")
        sk = pkeys[order]
        start = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
        out = SurfaceField(2 * self.cell)
        out._m = np.add.reduceat(m[order], start, axis=0)
        out._index.main = (sk[start], np.arange(len(start), dtype=np.int64))
        out._index.n = len(start)
        out.points = self.points
        return out

    def fit(self, min_spread: float = 0.5, batch: int = 20_000) -> dict:
        """Least-squares plane of every cell's 3x3x3 block, by slot: weight, mean (world), axes (rows: the two
        tangent axes, then the normal), the standard deviations along them, the spread over the surface that decides
        whether the block is more than one stripe (`spread`), and the tangent covariance (s11, s12, s22).

        Robust to a second surface at the block's edge (the other side of a thin wall, the other face of a step):
        whole neighbour cells whose own mean lies further from the first plane than 3 sigma (at least one cell) are
        dropped and the plane fitted again. A 0.4 mm wall scanned from both sides was pulled 0.02 mm towards its
        other side without this; the own surface's noise never reaches a cell that far (> 4 sigma).

        Where one stripe outweighs everything else in the block, the normal comes from the block with each cell's
        weight capped at the block's lower quartile: the user's turntable stood still for 65 s before turning, 3400
        frames laid the same 17 stripes, and 3.6 % of the bust stayed raw along those lines (0.3 % with the cap)
        because their blocks looked like a single stripe although the turning frames covered them all around. The
        mean stays the plain weighted mean (a capped weight per cell would also reweight the noise across the surface
        and shift it)."""
        self._index.merge()
        keys, slots = self._index.main
        n = self._index.n
        res = {"weight": np.zeros(n), "mean": np.zeros((n, 3)), "axes": np.zeros((n, 3, 3)),
               "sigma": np.zeros((n, 3)), "spread": np.zeros(n), "tcov": np.zeros((n, 3))}
        for b in range(0, len(keys), batch):
            kb, sb = keys[b:b + batch], slots[b:b + batch]
            M = self._gather(kb)
            W, mu, C = _moments(M.sum(axis=1))
            sig, axes = _axes(C)
            Wj = M[:, :, _W]
            cj = M[:, :, _S1] / np.maximum(Wj, 1e-12)[:, :, None]
            dist = np.abs(np.einsum("njk,nk->nj", cj - mu[:, None], axes[:, 2]))
            far = (Wj > 0) & (dist > np.maximum(3.0 * sig[:, 2:3], self.cell))
            redo = far.any(axis=1)
            if redo.any():
                M[redo] *= ~far[redo][:, :, None]
                W[redo], mu[redo], C[redo] = _moments(M[redo].sum(axis=1))
                sig[redo], axes[redo] = _axes(C[redo])
            spread = sig[:, 1].copy()
            one = np.flatnonzero((spread < min_spread * self.cell) & (W > 0))
            if len(one):
                Wo = M[one, :, _W]
                cap = np.nanpercentile(np.where(Wo > 0, Wo, np.nan), 25, axis=1)
                scale = np.minimum(1.0, cap[:, None] / np.maximum(Wo, 1e-12))
                sc, ac = _axes(_moments((M[one] * scale[:, :, None]).sum(axis=1))[2])
                fixed = sc[:, 1] >= min_spread * self.cell
                axes[one[fixed]], spread[one[fixed]] = ac[fixed], sc[fixed, 1]
            L = np.einsum("nik,nkl,njl->nij", axes, C, axes)            # weighted covariance in the block's axes
            res["weight"][sb] = W
            res["mean"][sb] = mu + key_centers(kb, self.cell)
            res["axes"][sb] = axes
            res["sigma"][sb] = np.sqrt(np.maximum(np.diagonal(L, axis1=1, axis2=2), 0.0))
            res["spread"][sb] = spread
            res["tcov"][sb] = np.stack([L[:, 0, 0], L[:, 0, 1], L[:, 1, 1]], axis=1)
        return res


def _shift(m: np.ndarray, d) -> np.ndarray:
    """Moments about a cell centre re-expressed about a point -d from it (parallel-axis theorem); d (3,) or (n, 3)."""
    d = np.broadcast_to(np.asarray(d, np.float64), (len(m), 3))
    w, s1 = m[:, _W], m[:, _S1]
    out = m.copy()
    out[:, _S1] += w[:, None] * d
    for c, (a, b) in enumerate(_PAIRS):
        out[:, 4 + c] += s1[:, a] * d[:, b] + s1[:, b] * d[:, a] + w * d[:, a] * d[:, b]
    return out


def _moments(acc: np.ndarray):
    """Weight, mean and covariance (n, 3, 3) from summed moments."""
    W = acc[:, _W]
    safe = np.maximum(W, 1e-12)
    mu = acc[:, _S1] / safe[:, None]
    C = np.empty((len(acc), 3, 3))
    for c, (a, b) in enumerate(_PAIRS):
        C[:, a, b] = C[:, b, a] = acc[:, 4 + c] / safe - mu[:, a] * mu[:, b]
    return W, mu, C


def _axes(C: np.ndarray):
    """Principal standard deviations (descending) and axes (rows: largest spread, middle, normal)."""
    lam, vec = np.linalg.eigh(C)                                   # ascending
    return np.sqrt(np.maximum(lam[:, ::-1], 0.0)), np.swapaxes(vec[:, :, ::-1], 1, 2)


def _curvature(field: SurfaceField, fits: dict, keys: np.ndarray, slots: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Second fundamental form (k11, k12, k22 in each block's own tangent axes) from how the neighbouring blocks'
    normals turn: a tangent offset u tilts the normal by -K u. (n, 3); NaN where too few neighbours fit."""
    axes = fits["axes"][slots]
    e1, e2, nq = axes[:, 0], axes[:, 1], axes[:, 2]
    mq = fits["mean"][slots]
    A = np.zeros((len(keys), 3, 3))
    rhs = np.zeros((len(keys), 3))
    count = np.zeros(len(keys))
    for o in NEIGHBOR_OFFSETS:
        if o == 0:
            continue
        sj = field._index.find(keys + o)
        ok = (sj >= 0) & good
        ok[ok] &= fits["good"][sj[ok]]
        if not ok.any():
            continue
        s = sj[ok]
        nj = fits["axes"][s, 2]
        nj *= np.sign(np.einsum("ij,ij->i", nj, nq[ok]))[:, None]
        d = fits["mean"][s] - mq[ok]
        u1, u2 = np.einsum("ij,ij->i", d, e1[ok]), np.einsum("ij,ij->i", d, e2[ok])
        t1, t2 = np.einsum("ij,ij->i", nj, e1[ok]), np.einsum("ij,ij->i", nj, e2[ok])
        z = np.zeros_like(u1)
        r1, r2 = np.stack([u1, u2, z], 1), np.stack([z, u1, u2], 1)
        A[ok] += r1[:, :, None] * r1[:, None, :] + r2[:, :, None] * r2[:, None, :]
        rhs[ok] -= t1[:, None] * r1 + t2[:, None] * r2
        count[ok] += 1
    K = np.full((len(keys), 3), np.nan)
    # neighbours on (nearly) one line fix only the curvature along it
    lam = np.linalg.eigvalsh(A)
    solvable = (count >= 4) & (lam[:, 0] > 0.05 * lam[:, 2])
    if solvable.any():
        K[solvable] = np.linalg.solve(A[solvable], rhs[solvable][:, :, None])[:, :, 0]
    return K


def _classify(fits: dict, cell: float, min_weight: float, min_spread: float, max_sigma: float):
    """One surface: enough samples, spread over the surface in both directions (not one stripe), thin across it.
    Sets fits["good"]; returns the (few, stripe, thick) masks by slot."""
    few = fits["weight"] < min_weight
    stripe = ~few & (fits["spread"] < min_spread * cell)
    thick = ~few & ~stripe & (fits["sigma"][:, 2] > max_sigma)
    fits["good"] = ~(few | stripe | thick)
    return few, stripe, thick


def _to_world(K: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """(n, 3, 3) shape tensors from (k11, k12, k22) in the blocks' tangent axes (rows 0, 1 of `axes`)."""
    e1, e2 = axes[:, 0], axes[:, 1]
    o = lambda a, b: a[:, :, None] * b[:, None, :]  # noqa: E731
    return (K[:, 0, None, None] * o(e1, e1) + K[:, 1, None, None] * (o(e1, e2) + o(e2, e1))
            + K[:, 2, None, None] * o(e2, e2))


def _to_block(S: np.ndarray, axes: np.ndarray) -> np.ndarray:
    e1, e2 = axes[:, 0], axes[:, 1]
    return np.stack([np.einsum("ni,nij,nj->n", e1, S, e1), np.einsum("ni,nij,nj->n", e1, S, e2),
                     np.einsum("ni,nij,nj->n", e2, S, e2)], axis=1)


def _block_curvature(field: SurfaceField, fits: dict, gate: dict):
    """Curvature (k11, k12, k22 in each block's own tangent axes, NaN where unknown) of every block, by slot:
    (raw, steady). `raw` from the block's own neighbours: sharp but noisy, it finds edges. `steady`, for the
    correction, is measured one level coarser (1.2 mm blocks, normals 0.4 mm apart).

    The normals of 0.6 mm blocks are tilted by the frames' tracking jitter enough that correcting with `raw` added
    0.007 mm of noise to a 15 mm sphere (thickness 0.054 -> 0.060) and to the replayed bust (two halves agreed to
    0.0106 instead of 0.0096 mm); `raw` averaged over the 3x3x3 neighbouring blocks still added 0.003 (sphere 0.057,
    bust 0.0103). The coarse one adds nothing measurable (sphere 0.055, bust 0.0098)."""
    keys, slots = field._index.main
    raw = np.full((field._index.n, 3), np.nan)
    raw[slots] = _curvature(field, fits, keys, slots, fits["good"][slots])
    coarse = field.coarsen()
    cf = coarse.fit(gate["min_spread"])
    _classify(cf, coarse.cell, **gate)
    ck, cs = coarse._index.main
    shape = np.full((coarse._index.n, 3, 3), np.nan)
    shape[cs] = _to_world(_curvature(coarse, cf, ck, cs, cf["good"][cs]), cf["axes"][cs])
    parent = coarse._index.find(voxel_keys(key_centers(keys, field.cell), coarse.cell))
    steady = np.full((field._index.n, 3), np.nan)
    have = parent >= 0
    A = fits["axes"][slots[have]]
    # the curvature belongs to the side the normal points to: flip with the normal
    sign = np.sign(np.einsum("ij,ij->i", A[:, 2], cf["axes"][parent[have], 2]))[:, None]
    steady[slots[have]] = sign * _to_block(shape[parent[have]], A)
    return raw, steady


def _kmax(K: np.ndarray) -> np.ndarray:
    """Largest principal curvature (absolute) of (k11, k12, k22); 0 where unknown."""
    K = np.nan_to_num(K)
    return np.abs(0.5 * (K[:, 0] + K[:, 2])) + np.sqrt(0.25 * (K[:, 0] - K[:, 2]) ** 2 + K[:, 1] ** 2)


def project(field: SurfaceField, points: np.ndarray, min_weight: float = 10.0, min_spread: float = 0.5,
            max_sigma: float = 0.5, residual_k: float = 6.0, max_curvature: float = 0.3,
            curvature: bool = True) -> tuple[np.ndarray, dict]:
    """Move each point along the normal onto the averaged surface of its cell's block; points with no clean, smooth
    block stay exactly as measured. Returns (points float64, stats).

    A block is clean with at least `min_weight` samples, spread at least `min_spread` cells (std) along both tangents
    (else it is one stripe and has no normal), and at most `max_sigma` cells (std) across its plane: in 0.2 mm cells
    clean blocks on the bust have 0.048 mm, two sides of a 0.25 mm wall or a 0.25 mm step in one block exceed 0.1. It
    is smooth when its curvature stays below `max_curvature` per cell (1.5 per mm in 0.2 mm cells). A point further
    than `residual_k` sigma from its block's surface stays raw (something else); at 4 sigma 0.5 % of the bust stayed
    raw - the long tails of poorly tracked frames - and the p90 thickness was 0.183 mm instead of 0.164. With these
    settings 99.0 % of the bust's points moved: 0.28 % stayed on single stripes, 0.51 % at edges, 0.13 % as outliers."""
    p = np.asarray(points, np.float64)
    out = p.copy()
    if not len(p) or not len(field):
        return out, {"points": len(p), "moved": 0, "moved_pct": 0.0, "median_move_mm": 0.0}
    cell = field.cell
    gate = {"min_weight": min_weight, "min_spread": min_spread, "max_sigma": max_sigma * cell}
    fits = field.fit(min_spread)
    few, stripe, thick = _classify(fits, cell, **gate)
    sig, tc = np.maximum(fits["sigma"], 1e-6 * cell), fits["tcov"]     # noiseless (CAD) input: no division by 0
    good = fits["good"]
    K = np.zeros((len(good), 3))
    bent = np.zeros(len(good), bool)
    if curvature:
        raw, steady = _block_curvature(field, fits, gate)
        # a correction beyond the edge limit is not a curvature (too few or lined-up neighbours): unknown
        steady[_kmax(steady) > max_curvature / cell] = np.nan
        K = np.nan_to_num(steady)
        # tighter than a 0.67 mm radius a 0.6 mm block is no longer one smooth surface: an edge, a corner (a 90 and
        # a 135 degree ridge both read sharper than that; a 1 mm radius is averaged and corrected exactly). Below
        # about 0.9 mm the 1.2 mm block that measures the correction is no longer one surface either: a plane alone
        # put a 0.7 mm radius 0.017 mm inside and 0.8 mm 0.010 inside, so where it has no value and the block's own
        # neighbours read tighter than 1 mm the points stay as measured.
        bent = good & ((_kmax(raw) > max_curvature / cell)
                       | (np.isnan(steady[:, 0]) & (_kmax(raw) > 0.2 / cell)))
    usable = good & ~bent

    def offset(s, q):
        """Signed distance of q above block s's surface along its normal, and the normal. The plane of a curved block
        lies inside the surface by tr(K S) / 2 (S: the block's tangent covariance), so the surface at q is
        h = (u^T K u - tr(K S)) / 2 above the plane."""
        axes = fits["axes"][s]
        d = q - fits["mean"][s]
        u1, u2 = np.einsum("ij,ij->i", d, axes[:, 0]), np.einsum("ij,ij->i", d, axes[:, 1])
        k = K[s]
        h = 0.5 * (k[:, 0] * (u1 ** 2 - tc[s, 0]) + 2 * k[:, 1] * (u1 * u2 - tc[s, 1]) + k[:, 2] * (u2 ** 2 - tc[s, 2]))
        return np.einsum("ij,ij->i", d, axes[:, 2]) - h, axes[:, 2]

    keys = voxel_keys(p, cell)
    s0 = field._index.find(keys)
    e = np.zeros(len(p))
    n = np.zeros((len(p), 3))
    chosen = s0.copy()
    own = np.flatnonzero(s0 >= 0)
    e[own], n[own] = offset(s0[own], p[own])
    move = np.zeros(len(p), bool)
    move[own] = usable[s0[own]] & (np.abs(e[own]) <= residual_k * sig[s0[own], 2])
    # A point whose noise carried it across a cell boundary lands in the next cell out from the surface; where that
    # cell's block is not clean (it reaches the other side of a thin wall) the point would stay raw while its twin on
    # the near side is averaged - and keeping only one side of the noise biased a 0.4 mm wall by 0.015 mm. Such points
    # use the clean block of the neighbour across the surface (never one beside it: that would extend a face over a
    # fillet). At a ridge this puts the points next to it on their own face's plane: within 0.3 mm of a 90 degree
    # ridge 5 % of the points went 0.030 mm inside (0.065 without, 0.072 raw) and 5 % 0.059 mm outside (0.073, 0.080).
    rest = np.flatnonzero(~move)
    best = np.full(len(rest), np.inf)
    for o, off in zip(NEIGHBOR_OFFSETS, _OFFSETS):
        if o == 0 or not len(rest):
            continue
        sj = field._index.find(keys[rest] + o)
        idx = np.flatnonzero(sj >= 0)
        idx = idx[usable[sj[idx]]]
        if not len(idx):
            continue
        s = sj[idx]
        ej, nj = offset(s, p[rest[idx]])
        score = np.abs(ej) / sig[s, 2]
        across = np.abs(nj @ (off / np.linalg.norm(off))) > 0.57          # within 55 degrees of the normal
        take = across & (score <= residual_k) & (score < best[idx])
        best[idx[take]] = score[take]
        e[rest[idx[take]]], n[rest[idx[take]]], chosen[rest[idx[take]]] = ej[take], nj[take], s[take]
    via_neighbour = rest[np.isfinite(best)]
    move[via_neighbour] = True
    out[move] -= e[move, None] * n[move]

    def pct(count):
        return round(100.0 * float(count) / len(p), 2)

    kept = ~move
    sk = np.where(s0 >= 0, s0, 0)
    stats = {"points": len(p), "moved": int(move.sum()), "moved_pct": pct(move.sum()),
             "moved_via_neighbour_pct": pct(len(via_neighbour)),
             "median_move_mm": round(float(np.median(np.abs(e[move]))), 4) if move.any() else 0.0,
             "kept_few_samples_pct": pct((kept & ((s0 < 0) | few[sk])).sum()),
             "kept_single_stripe_pct": pct((kept & (s0 >= 0) & stripe[sk]).sum()),
             "kept_not_one_surface_pct": pct((kept & (s0 >= 0) & thick[sk]).sum()),
             "kept_edge_pct": pct((kept & (s0 >= 0) & bent[sk]).sum()),
             "kept_outlier_pct": pct((kept & (s0 >= 0) & usable[sk]).sum()),
             "median_sigma_mm": round(float(np.median(sig[chosen[move], 2])), 4) if move.any() else None}
    return out, stats


