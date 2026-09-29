"""Render the triangulated points of a frame as a depth-coloured image over the rectified left view."""
import sys
import cv2
import numpy as np
from stripes import Triangulator, load_raw
tri = Triangulator.from_yaml(sys.argv[2])
frame = load_raw(sys.argv[1])
rl, _ = tri.rectify(frame)
pts, xs, ys = tri.points(frame, with_pixels=True)
z = pts[:, 2]
lo, hi = np.percentile(z, 2), np.percentile(z, 98)
col = cv2.applyColorMap(np.clip((z - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
img = cv2.cvtColor((rl // 3).astype(np.uint8), cv2.COLOR_GRAY2BGR)
for (x, y), c in zip(np.c_[xs, ys].astype(int), col):
    cv2.circle(img, (x, y), 2, tuple(int(v) for v in c), -1)
cv2.putText(img, f"z {lo:.0f} (blue) .. {hi:.0f} mm (red)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
cv2.imwrite(sys.argv[3], cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA))
