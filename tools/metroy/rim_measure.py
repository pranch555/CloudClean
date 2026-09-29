"""Lid rim of the test box: height of the rim top above the paper, and the outer rectangle of the rim points.

Calipers: lid 120.65 mm long, rim top 62 mm above the table. Several frames are fused in the scanner frame (the
scene and the scanner are static), which fills in the ~12 mm gaps between the stripes of one frame.
    python rim_measure.py frame.raw [frame.raw ...] camparam.yaml
"""
import sys
import cv2
import numpy as np
import stripes as S
from flatness import fit_plane

*frames, calib = sys.argv[1:]
pts = np.concatenate([S.frame_points(f, calib) for f in frames])
c, vt, r, inl = fit_plane(pts, gate=1.0)
n = vt[2] if vt[2] @ c < 0 else -vt[2]
base = pts[np.abs((pts - c) @ n) < 1.0]
c = base.mean(0)                                   # refit the paper plane on its own points
_, _, v2 = np.linalg.svd(base - c, full_matrices=False)
n = v2[2] if v2[2] @ c < 0 else -v2[2]
h = (pts - c) @ n
print(f"{len(frames)} frames, {len(pts)} points; paper plane RMS {((base - c) @ n).std() * 1000:.0f} um")
hist, edges = np.histogram(h[h > 50], bins=np.arange(50, 66, 0.5))
print("heights above paper (mm):", " ".join(f"{e:.1f}:{v}" for e, v in zip(edges, hist) if v > 20))
top = pts[(h > 58.5) & (h < 64)]
u = v2[0]; w = np.cross(n, u)
uv = np.c_[(top - c) @ u, (top - c) @ w].astype(np.float32)
# the rim is a ring: drop isolated stragglers before taking the outer rectangle
keep = np.ones(len(uv), bool)
for _ in range(2):
    ctr = np.median(uv[keep], 0)
    d = np.abs(uv - ctr)
    keep = (d[:, 0] < np.percentile(d[keep, 0], 99.5) + 1) & (d[:, 1] < np.percentile(d[keep, 1], 99.5) + 1)
(cx, cy), (a, b), ang = cv2.minAreaRect(uv[keep])
print(f"rim top: {keep.sum()} points, height median {np.median(h[(h > 58.5) & (h < 64)]):.2f} mm, "
      f"95th pct {np.percentile(h[(h > 58.5) & (h < 64)], 95):.2f} mm; outer rectangle {max(a, b):.2f} x {min(a, b):.2f} mm")
