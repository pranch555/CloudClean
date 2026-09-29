"""Find the dominant planes in a laser scan and report each one's size, tilt and flatness."""
import sys

import numpy as np

from laser_scan import scan

pts, rl, rr, rows = scan(sys.argv[1], sys.argv[2])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
print(f"points: {len(pts)}")
rng = np.random.default_rng(0)
remaining = pts.copy()
planes = []
for k in range(3):
    if len(remaining) < 200:
        break
    best = None
    for _ in range(3000):
        s = remaining[rng.choice(len(remaining), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((remaining - s[0]) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 150:
        break
    P = remaining[best]
    centroid = P.mean(0)
    _, _, vt = np.linalg.svd(P - centroid)
    normal, u = vt[2], vt[0]
    v = np.cross(normal, u)
    uv = np.c_[(P - centroid) @ u, (P - centroid) @ v]
    res = (P - centroid) @ normal
    planes.append((len(P), centroid, normal, res, uv))
    print(f"\nplane {k + 1}: {len(P)} points ({len(P) / len(pts):.0%})")
    print(f"   distance {np.linalg.norm(centroid):7.1f} mm   tilt {np.degrees(np.arccos(abs(normal @ [0, 0, 1]))):.1f} deg")
    print(f"   extent   {np.ptp(uv[:, 0]):7.1f} x {np.ptp(uv[:, 1]):.1f} mm")
    print(f"   flatness RMS {res.std() * 1000:.0f} um, 95% within {np.percentile(np.abs(res), 95) * 1000:.0f} um")
    remaining = remaining[~best]

if len(planes) >= 2:
    (_, c1, n1, *_), (_, c2, *_) = planes[0], planes[1]
    print(f"\nseparation between plane 1 and 2 along plane 1's normal: {abs((c2 - c1) @ n1):.2f} mm")
