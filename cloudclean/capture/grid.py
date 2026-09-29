"""Sparse voxel bookkeeping for live fusion, built on sorted int64 keys (numpy only, no Python loops).

A voxel key packs the integer cell coordinates (21 bits each, +-1M cells per axis) into one int64.
`VoxelCounter` keeps per-cell counts (and optional summed vectors) in two sorted levels: a large main
array and a small delta that new cells are inserted into; the delta is merged into the main array once it
grows past a fraction of it, so adding a frame costs O(frame log frame + delta) instead of O(map)."""
from __future__ import annotations

import numpy as np

_BITS = 21
_MASK = (1 << _BITS) - 1
_OFF = 1 << (_BITS - 1)


def voxel_keys(points: np.ndarray, size: float) -> np.ndarray:
    ijk = np.floor(np.asarray(points, dtype=np.float64) / size).astype(np.int64) + _OFF
    np.clip(ijk, 0, _MASK, out=ijk)
    return (ijk[:, 0] << (2 * _BITS)) | (ijk[:, 1] << _BITS) | ijk[:, 2]


def key_cells(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    return np.stack([(keys >> (2 * _BITS)) & _MASK, (keys >> _BITS) & _MASK, keys & _MASK], axis=1) - _OFF


def key_centers(keys: np.ndarray, size: float) -> np.ndarray:
    return (key_cells(keys) + 0.5) * size


def _lookup(sorted_keys: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.searchsorted(sorted_keys, query)
    if len(sorted_keys) == 0:
        return pos, np.zeros(len(query), bool)
    found = sorted_keys[np.minimum(pos, len(sorted_keys) - 1)] == query
    return pos, found


class VoxelCounter:
    def __init__(self, vectors: bool = False):
        self.with_vectors = vectors
        self._levels = [self._empty(), self._empty()]   # [main, delta]

    def _empty(self) -> dict:
        lvl = {"keys": np.empty(0, np.int64), "counts": np.empty(0, np.int32)}
        if self.with_vectors:
            lvl["vecs"] = np.empty((0, 3), np.float32)
        return lvl

    def __len__(self) -> int:
        return sum(len(lvl["keys"]) for lvl in self._levels)

    def nbytes(self) -> int:
        return sum(a.nbytes for lvl in self._levels for a in lvl.values())

    def get(self, keys: np.ndarray) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.int64)
        out = np.zeros(len(keys), np.int32)
        for lvl in self._levels:
            pos, found = _lookup(lvl["keys"], keys)
            out[found] = lvl["counts"][pos[found]]
        return out

    def add(self, keys: np.ndarray, cap: int | None = None, vectors: np.ndarray | None = None):
        """Count points into their cells, accepting at most `cap` points per cell over the whole session.

        Returns (accepted mask per input key, unique keys, new-cell mask per unique key, count after per unique key,
        number accepted per unique key)."""
        keys = np.asarray(keys, dtype=np.int64)
        if len(keys) == 0:
            e = np.empty(0, np.int64)
            return np.zeros(0, bool), e, np.zeros(0, bool), np.zeros(0, np.int32), np.zeros(0, np.int32)
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        uniq, start, n_in = np.unique(sk, return_index=True, return_counts=True)
        before = self.get(uniq)
        if cap is None:
            acc_sorted = np.ones(len(sk), bool)
        else:
            rank = np.arange(len(sk)) - np.repeat(start, n_in)
            acc_sorted = np.repeat(before, n_in) + rank < cap
        accepted = np.empty(len(keys), bool)
        accepted[order] = acc_sorted
        added = np.add.reduceat(acc_sorted.astype(np.int32), start)
        after = before + added
        vec_add = None
        if self.with_vectors and vectors is not None:
            v = np.asarray(vectors, np.float32)[order] * acc_sorted[:, None]
            vec_add = np.add.reduceat(v, start, axis=0)

        remaining = added > 0
        for lvl in self._levels:
            pos, found = _lookup(lvl["keys"], uniq)
            hit = found & remaining
            if hit.any():
                lvl["counts"][pos[hit]] += added[hit]
                if vec_add is not None:
                    lvl["vecs"][pos[hit]] += vec_add[hit]
                remaining &= ~found
        new = remaining
        if new.any():
            delta = self._levels[1]
            pos = np.searchsorted(delta["keys"], uniq[new])
            delta["keys"] = np.insert(delta["keys"], pos, uniq[new])
            delta["counts"] = np.insert(delta["counts"], pos, added[new])
            if self.with_vectors:
                vals = vec_add[new] if vec_add is not None else np.zeros((int(new.sum()), 3), np.float32)
                delta["vecs"] = np.insert(delta["vecs"], pos, vals, axis=0)
            if len(delta["keys"]) > max(50_000, len(self._levels[0]["keys"]) // 8):
                self._merge()
        return accepted, uniq, new, after, added

    def _merge(self) -> None:
        main, delta = self._levels
        pos = np.searchsorted(main["keys"], delta["keys"])
        merged = {"keys": np.insert(main["keys"], pos, delta["keys"]),
                  "counts": np.insert(main["counts"], pos, delta["counts"])}
        if self.with_vectors:
            merged["vecs"] = np.insert(main["vecs"], pos, delta["vecs"], axis=0)
        self._levels = [merged, self._empty()]

    def items(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """All cells (sorted keys, counts, summed vectors or None). Returned arrays are copies."""
        self._merge()
        main = self._levels[0]
        return main["keys"].copy(), main["counts"].copy(), main["vecs"].copy() if self.with_vectors else None


_D = np.array([-1, 0, 1], np.int64)
NEIGHBOR_OFFSETS = ((_D[:, None, None] << (2 * _BITS)) + (_D[None, :, None] << _BITS) + _D[None, None, :]).reshape(-1)


def neighbor_keys(keys: np.ndarray) -> np.ndarray:
    """(n, 27) keys of the 3x3x3 block around each cell."""
    return np.asarray(keys, np.int64)[:, None] + NEIGHBOR_OFFSETS[None, :]


def sorted_getter(sorted_keys: np.ndarray, counts: np.ndarray):
    """Count lookup function over sorted key / count arrays (absent keys -> 0)."""
    def get(query):
        pos, found = _lookup(sorted_keys, query)
        out = np.zeros(len(query), np.int64)
        out[found] = counts[pos[found]]
        return out
    return get
