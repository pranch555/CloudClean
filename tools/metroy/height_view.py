"""Colour each triangulated point by its height above the dominant plane (the surface things sit on).

A mis-indexed stripe shows up as a whole stripe at a height the surface around it does not have.
    python height_view.py frame.raw camparam.yaml out.png [method]
"""
import sys
import cv2
import numpy as np
import stripes as S
from flatness import fit_plane

raw, calib, out = sys.argv[1:4]
method = sys.argv[4] if len(sys.argv) > 4 else "lines"
tri = S.Triangulator.from_yaml(calib, extra=S.default_extra(calib))
frame = S.load_raw(raw)
rl, _ = tri.rectify(frame)
pts, xs, ys = tri.points(frame, with_pixels=True, matching="lines" if method == "lines" else "order")
c, vt, r, inl = fit_plane(pts, gate=1.0)
n = vt[2] if vt[2] @ c < 0 else -vt[2]          # normal towards the scanner = up
h = (pts - c) @ n
col = cv2.applyColorMap(np.clip(h / 70 * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
img = cv2.cvtColor((rl // 3).astype(np.uint8), cv2.COLOR_GRAY2BGR)
for (x, y), cc, hh in zip(np.c_[xs, ys].astype(int), col, h):
    cv2.circle(img, (x, y), 2, tuple(int(v) for v in cc) if hh > -5 else (255, 0, 255), -1)
cv2.putText(img, f"{method}: height 0 (blue) .. 70 mm (red), below -5 magenta", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
cv2.imwrite(out, cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA))
hist, edges = np.histogram(h, bins=np.arange(-60, 80, 4))
print(method, len(pts), "pts;", " ".join(f"{e:.0f}:{v}" for e, v in zip(edges, hist) if v > len(h) * 0.01))
