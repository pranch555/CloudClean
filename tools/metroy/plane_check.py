"""Fit a plane to a laser scan of a flat surface: how flat does flat measure?"""
import sys

import numpy as np

from laser_scan import scan

pts, rl, rr, rows = scan(sys.argv[1], sys.argv[2])
print(f"points: {len(pts)}")

best, rng = None, np.random.default_rng(0)
for _ in range(2000):
    s = pts[rng.choice(len(pts), 3, replace=False)]
    n = np.cross(s[1] - s[0], s[2] - s[0])
    if np.linalg.norm(n) < 1e-9:
        continue
    n /= np.linalg.norm(n)
    inl = np.abs((pts - s[0]) @ n) < 1.0
    if best is None or inl.sum() > best.sum():
        best = inl

P = pts[best]
centroid = P.mean(0)
_, _, vt = np.linalg.svd(P - centroid)
normal = vt[2]
res = (P - centroid) @ normal
tilt = np.degrees(np.arccos(abs(normal @ [0, 0, 1])))

print(f"inliers: {best.sum()}/{len(pts)} ({best.mean():.1%}) within 1 mm of a plane")
print(f"distance to surface: {np.linalg.norm(centroid):.1f} mm, surface tilted {tilt:.1f} deg from face-on")
print(f"FLATNESS: RMS {res.std() * 1000:.0f} um, 95% within {np.percentile(np.abs(res), 95) * 1000:.0f} um, "
      f"max {np.abs(res).max() * 1000:.0f} um")

# extent of the measured patch, in the plane
u = vt[0] / np.linalg.norm(vt[0])
v = np.cross(normal, u)
uv = np.c_[(P - centroid) @ u, (P - centroid) @ v]
print(f"patch size: {np.ptp(uv[:, 0]):.1f} x {np.ptp(uv[:, 1]):.1f} mm")
