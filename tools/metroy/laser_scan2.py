"""Laser triangulation, second pass: stricter stripe detection and a neighbourhood consistency filter.

Improvements over laser_scan.py, all aimed at dark or low-return surfaces where the first version latched onto
noise:

* a peak must clear an absolute floor *and* stand proud of the local background (prominence), so faint noise ripples
  are ignored;
* a peak must be narrow. The board's retro-reflective markers are wide saturated blobs, not stripes, and were being
  matched as if they were;
* after triangulation, a point must agree with its neighbours in the image. A mis-paired stripe lands at a wildly
  different depth from everything around it, which no real surface does.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from laser_scan import Z_MAX, Z_MIN, match_row, rectifiers
from rectify_check import load_calib, split


def peaks(row: np.ndarray, floor: float, prominence: float, max_width: int = 14) -> list[float]:
    out = []
    n = len(row) - 1
    i = 1
    while i < n:
        if row[i] >= floor and row[i] >= row[i - 1] and row[i] > row[i + 1]:
            lo = i
            while lo > 0 and row[lo - 1] < row[lo]:
                lo -= 1
            hi = i
            while hi < n and row[hi + 1] < row[hi]:
                hi += 1
            width = hi - lo
            base = max(row[lo], row[hi])
            if width <= max_width and row[i] - base >= prominence:
                a, b, c = float(row[i - 1]), float(row[i]), float(row[i + 1])
                den = a - 2 * b + c
                shift = 0.5 * (a - c) / den if den else 0.0
                out.append(i + (shift if abs(shift) <= 1 else 0.0))
            i = hi
        i += 1
    return out


def scan2(raw: str, calib_path: str, floor: float = 45.0, prominence: float = 18.0, agree_mm: float = 2.0):
    c = load_calib(calib_path)
    m1, m2, P1, P2, Q = rectifiers(c, calib_path)
    left, right = split(raw, c["size"])
    rl = cv2.remap(left, *m1, cv2.INTER_LINEAR)
    rr = cv2.remap(right, *m2, cv2.INTER_LINEAR)
    f, B, off = P1[0, 0], -P2[0, 3] / P2[0, 0], P1[0, 2] - P2[0, 2]

    xs, ys, pts = [], [], []
    for y in range(rl.shape[0]):
        pa = peaks(rl[y].astype(np.float32), floor, prominence)
        pb = peaks(rr[y].astype(np.float32), floor, prominence)
        for xl, xr in match_row(pa, pb, f, B, off):
            d = xl - xr
            X = np.array([xl, y, d, 1.0]) @ Q.T
            pts.append(X[:3] / X[3])
            xs.append(xl)
            ys.append(y)
    if not pts:
        return np.zeros((0, 3)), rl, rr, 0
    pts = np.array(pts)
    xs = np.array(xs)
    ys = np.array(ys)

    # neighbourhood agreement: bin by image position, compare each depth with the local median
    raw_n = len(pts)
    cell = 24
    keys = (ys // cell).astype(np.int64) * 10_000 + (xs // cell).astype(np.int64)
    order = np.argsort(keys)
    keep = np.zeros(len(pts), bool)
    z = pts[:, 2]
    start = 0
    ks = keys[order]
    for i in range(1, len(ks) + 1):
        if i == len(ks) or ks[i] != ks[start]:
            idx = order[start:i]
            if len(idx) >= 4:
                med = np.median(z[idx])
                keep[idx] = np.abs(z[idx] - med) < agree_mm
            start = i
    return pts[keep], rl, rr, raw_n


if __name__ == "__main__":
    pts, rl, rr, raw_n = scan2(sys.argv[1], sys.argv[2])
    print(f"stripe points {raw_n} -> {len(pts)} after the agreement filter")
    if len(pts):
        print(f"depth median {np.median(pts[:, 2]):.1f} mm")
        if len(sys.argv) > 3:
            from laser_scan import write_ply
            write_ply(sys.argv[3], pts)
            print("wrote", sys.argv[3])
