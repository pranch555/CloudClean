"""Heights above the surface the object sits on: reveals steps such as a lid rim around a recessed panel."""
import sys

import cv2
import numpy as np

from box_measure import fit_plane
from stripes import frame_points

pts = frame_points(sys.argv[1], sys.argv[-1])
print(f"points: {len(pts)}")

# the supporting surface is the plane with the most points once the object is a minority of the scene
base_inl, C, N, _ = fit_plane(pts, 0.8)
if N @ C > 0:            # point the normal towards the scanner so "up" is positive
    N = -N
h = (pts - C) @ N
print(f"base plane: {base_inl.sum()} points at {np.linalg.norm(C):.0f} mm")

hist, edges = np.histogram(h[(h > -5) & (h < 120)], bins=60)
for cnt, e in zip(hist, edges):
    if cnt > max(3, hist.max() * 0.02):
        print(f"   {e:6.1f} mm | {'#' * int(50 * cnt / hist.max())} {cnt}")

for lo, hi, label in [(float(sys.argv[2]), float(sys.argv[3]), "band")] if len(sys.argv) > 4 else []:
    band = pts[(h > lo) & (h < hi)]
    if len(band) < 30:
        print(f"{label} {lo}-{hi} mm: only {len(band)} points")
        continue
    c = band.mean(0)
    _, _, vt = np.linalg.svd(band - c)
    uv = np.c_[(band - c) @ vt[0], (band - c) @ vt[1]].astype(np.float32)
    (_, _), (w, hgt), _ = cv2.minAreaRect(uv)
    print(f"{label} {lo}-{hi} mm: {len(band)} points, rectangle {max(w, hgt):.1f} x {min(w, hgt):.1f} mm, "
          f"mean height {h[(h > lo) & (h < hi)].mean():.1f} mm")
