"""Measure a box's top face from MetroY laser frames: size, flatness and height above the surface it sits on."""
import sys

import cv2
import numpy as np

from stripes import frame_points


def fit_plane(pts, tol, iters=4000, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - s[0]) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    P = pts[best]
    centroid = P.mean(0)
    _, _, vt = np.linalg.svd(P - centroid)
    return best, centroid, vt[2], vt


def rect_of(P, centroid, basis):
    u, v = basis[0], basis[1]
    uv = np.c_[(P - centroid) @ u, (P - centroid) @ v].astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(uv)
    return max(w, h), min(w, h)


for raw in sys.argv[1:-1]:
    pts = frame_points(raw, sys.argv[-1])
    if len(pts) < 300:
        print(f"{raw}: only {len(pts)} points"); continue
    inl1, c1, n1, b1 = fit_plane(pts, 0.6)
    rest = pts[~inl1]
    inl2, c2, n2, _ = fit_plane(rest, 0.6) if len(rest) > 200 else (None, None, None, None)
    # the top face is the plane nearer the scanner
    top_first = c2 is None or np.linalg.norm(c1) < np.linalg.norm(c2)
    P = pts[inl1] if top_first else rest[inl2]
    C, N, B = (c1, n1, b1) if top_first else (c2, n2, np.linalg.svd(rest[inl2] - c2)[2])
    long_mm, short_mm = rect_of(P, C, B)
    res = (P - C) @ N
    line = (f"{raw.split('/')[-1]}: top face {long_mm:6.1f} x {short_mm:5.1f} mm  "
            f"({len(P)} pts, flatness RMS {res.std() * 1000:3.0f} um, distance {np.linalg.norm(C):.0f} mm)")
    if c2 is not None:
        other = c2 if top_first else c1
        line += f"  height above surface {abs((other - C) @ N):5.1f} mm"
    print(line)
