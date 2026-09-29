"""Render the rectified left view plus a depth-coloured overlay of the triangulated laser points."""
import sys

import cv2
import numpy as np

from laser_scan import scan

pts, rl, rr, rows = scan(sys.argv[1], sys.argv[2])
z = pts[:, 2]
print(f"points {len(pts)}; depth percentiles "
      f"{np.round(np.percentile(z, [1, 5, 25, 50, 75, 95, 99]), 1)}")
hist, edges = np.histogram(z, bins=60)
for h, e in zip(hist, edges):
    if h > len(z) * 0.01:
        print(f"   {e:7.1f} mm | {'#' * int(60 * h / hist.max())} {h}")
vis = cv2.cvtColor(rl, cv2.COLOR_GRAY2BGR)
cv2.imwrite("/tmp/scene_left.png", cv2.resize(vis, (800, 600)))
