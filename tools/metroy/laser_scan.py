"""Turn one MetroY laser frame into 3D points, the way the scanner itself measures.

Rectify both views with the factory calibration, find the laser stripes to sub-pixel precision on every row, pair
them left-to-right, and triangulate. Stripes keep their order in both views (no crossing), so an order-preserving
match is correct whenever the same stripes are visible in both; rows where the counts disagree are matched by
dynamic programming that may skip a stripe on either side rather than guess.

    python laser_scan.py frame.raw camparam.yaml [out.ply]
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from rectify_check import load_calib, split

Z_MIN, Z_MAX = 120.0, 800.0      # plausible working range in mm; everything else is a mismatch


def rectifiers(c, calib_path):
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    P1, P2, Q = fs.getNode("pL").mat(), fs.getNode("pR").mat(), fs.getNode("Q").mat()
    # flags=0 (no CALIB_ZERO_DISPARITY) is what reproduces Revopoint's own pL/pR
    R1, R2, *_ = cv2.stereoRectify(c["KL"], c["DL"], c["KR"], c["DR"], c["size"], c["R"], c["T"], flags=0)
    m1 = cv2.initUndistortRectifyMap(c["KL"], c["DL"], R1, P1, c["size"], cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(c["KR"], c["DR"], R2, P2, c["size"], cv2.CV_32FC1)
    return m1, m2, P1, P2, Q


def peaks_in_row(row: np.ndarray, threshold: float, min_gap: int = 4) -> list[float]:
    """Sub-pixel stripe centres: local maxima refined by a parabola through the neighbouring intensities."""
    out = []
    i = 1
    n = len(row) - 1
    while i < n:
        v = row[i]
        if v >= threshold and v >= row[i - 1] and v >= row[i + 1]:
            a, b, cc = float(row[i - 1]), float(v), float(row[i + 1])
            denom = a - 2 * b + cc
            shift = 0.5 * (a - cc) / denom if denom != 0 else 0.0
            if abs(shift) <= 1:
                if out and i + shift - out[-1] < min_gap:
                    if b > row[int(round(out[-1]))]:
                        out[-1] = i + shift
                else:
                    out.append(i + shift)
            i += min_gap - 1
        i += 1
    return out


def match_row(pl: list[float], pr: list[float], f: float, B: float, off: float) -> list[tuple[float, float]]:
    """Order-preserving match (a stripe cannot overtake another) with skips allowed on either side."""
    if not pl or not pr:
        return []
    n, m = len(pl), len(pr)
    NEG = -1e9
    score = np.full((n + 1, m + 1), NEG)
    back = np.zeros((n + 1, m + 1), np.int8)
    score[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if score[i, j] == NEG:
                continue
            if i < n and j < m:
                d = pl[i] - pr[j]
                z = f * B / (d - off) if d != off else 0.0
                if Z_MIN < z < Z_MAX:
                    s = score[i, j] + 1.0
                    if s > score[i + 1, j + 1]:
                        score[i + 1, j + 1], back[i + 1, j + 1] = s, 1
            if i < n and score[i, j] - 0.2 > score[i + 1, j]:
                score[i + 1, j], back[i + 1, j] = score[i, j] - 0.2, 2
            if j < m and score[i, j] - 0.2 > score[i, j + 1]:
                score[i, j + 1], back[i, j + 1] = score[i, j] - 0.2, 3
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        b = back[i, j]
        if b == 1:
            pairs.append((pl[i - 1], pr[j - 1])); i, j = i - 1, j - 1
        elif b == 2:
            i -= 1
        elif b == 3:
            j -= 1
        else:
            break
    return pairs[::-1]


def scan(raw: str, calib_path: str):
    c = load_calib(calib_path)
    m1, m2, P1, P2, Q = rectifiers(c, calib_path)
    left, right = split(raw, c["size"])
    rl = cv2.remap(left, *m1, cv2.INTER_LINEAR)
    rr = cv2.remap(right, *m2, cv2.INTER_LINEAR)
    f, B, off = P1[0, 0], -P2[0, 3] / P2[0, 0], P1[0, 2] - P2[0, 2]

    # the stripes are far brighter than the surface; threshold per row, relative to that row
    pts, rows_used = [], 0
    for y in range(rl.shape[0]):
        a, b = rl[y].astype(np.float32), rr[y].astype(np.float32)
        ta, tb = max(40.0, a.max() * 0.45), max(40.0, b.max() * 0.45)
        pa, pb = peaks_in_row(a, ta), peaks_in_row(b, tb)
        pairs = match_row(pa, pb, f, B, off)
        if pairs:
            rows_used += 1
        for xl, xr in pairs:
            d = xl - xr
            z = f * B / (d - off)
            X = np.array([xl, y, d, 1.0]) @ Q.T
            pts.append(X[:3] / X[3])
    pts = np.array(pts) if pts else np.zeros((0, 3))
    return pts, rl, rr, rows_used


def write_ply(path, pts):
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(fh, pts, fmt="%.4f")


if __name__ == "__main__":
    raw, calib = sys.argv[1], sys.argv[2]
    pts, rl, rr, rows = scan(raw, calib)
    print(f"rows with matches: {rows}/{rl.shape[0]}, points: {len(pts)}")
    if len(pts):
        z = pts[:, 2]
        print(f"depth: median {np.median(z):.1f} mm, 5-95% {np.percentile(z, 5):.1f}..{np.percentile(z, 95):.1f} mm")
        if len(sys.argv) > 3:
            write_ply(sys.argv[3], pts)
            print("wrote", sys.argv[3])
