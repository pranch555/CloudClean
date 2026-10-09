"""Stereo row offset measured by the markers: dy = y_left - y_right of every marker pair (rectified views).

A rectified pair has dy = 0 everywhere. A constant dy is a vertical offset of the two rectified views, a dy that
grows across the image a small roll between them (docs/metroy-handoff.md, "What is missing" 2: a ~0.2-0.25 px offset
was suspected from the two laser line families' disagreement). REPORT ONLY - nothing here changes the rectification.

Input: the per-frame results marker_precision.py saves (--save). Only pairs whose centres are rim fits in BOTH views
count (a centroid, the fallback, is dragged by the stripes), and markers are binned over the image so that a few
markers seen in many frames do not outweigh the rest.

    python marker_dy.py run1.pkl [run2.pkl ...]
"""
import pickle
import sys

import numpy as np


def main():
    rows = []
    for path in sys.argv[1:]:
        for f in pickle.load(open(path, "rb")):
            L, R = f["L"], f["R"]
            ok = np.isfinite(L[:, 4]) & np.isfinite(R[:, 4])
            rows.append(np.c_[L[ok, 0], L[ok, 1], L[ok, 1] - R[ok, 1], f["P"][ok, 2], L[ok, 0] - R[ok, 0]])
    X = np.concatenate(rows)
    x, y, dy, z = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    print(f"{len(X)} pairs (rim fits in both views): dy mean {dy.mean():+.3f} px, median {np.median(dy):+.3f}, "
          f"std {dy.std():.3f}, robust std {1.4826 * np.median(np.abs(dy - np.median(dy))):.3f}")
    # 100 px cells: the median of each cell, then a robust plane through the cells
    cx, cy = (x // 100).astype(int), (y // 100).astype(int)
    cells = {}
    for i, key in enumerate(zip(cx, cy)):
        cells.setdefault(key, []).append(i)
    C = np.array([(np.mean(x[v]), np.mean(y[v]), np.median(dy[v]), len(v)) for v in cells.values() if len(v) >= 5])
    A = np.c_[np.ones(len(C)), (C[:, 0] - 800) / 1000, (C[:, 1] - 600) / 1000]
    w = np.ones(len(C))
    for _ in range(5):                                           # Huber-weighted plane
        coef = np.linalg.lstsq(A * w[:, None], C[:, 2] * w, rcond=None)[0]
        res = C[:, 2] - A @ coef
        s = 1.4826 * np.median(np.abs(res))
        w = np.sqrt(np.minimum(1.0, 1.5 * s / np.maximum(np.abs(res), 1e-9)))
    cov = np.linalg.inv((A * w[:, None] ** 2).T @ A) * s ** 2
    se = np.sqrt(np.diag(cov))
    print(f"{len(C)} cells of 100 px: dy = {coef[0]:+.3f} (+-{se[0]:.3f}) {coef[1]:+.3f} (+-{se[1]:.3f}) "
          f"* (x - 800)/1000 "
          f"{coef[2]:+.3f} (+-{se[2]:.3f}) * (y - 600)/1000 px; cell scatter about it {s:.3f} px")
    for lo, hi in ((0, 400), (400, 800), (800, 1200), (1200, 1600)):
        m = (C[:, 0] >= lo) & (C[:, 0] < hi)
        if m.any():
            print(f"  x {lo:4d}-{hi:4d}: dy median {np.median(C[m, 2]):+.3f} px over {m.sum()} cells "
                  f"({int(C[m, 3].sum())} pairs)")
    for lo, hi in ((0, 400), (400, 800), (800, 1200)):
        m = (C[:, 1] >= lo) & (C[:, 1] < hi)
        if m.any():
            print(f"  y {lo:4d}-{hi:4d}: dy median {np.median(C[m, 2]):+.3f} px over {m.sum()} cells")
    zb = np.percentile(z, [0, 33, 67, 100])
    for lo, hi in zip(zb[:-1], zb[1:]):
        m = (z >= lo) & (z <= hi)
        print(f"  depth {lo:5.0f}-{hi:5.0f} mm: dy median {np.median(dy[m]):+.3f} px ({m.sum()} pairs)")


if __name__ == "__main__":
    main()
