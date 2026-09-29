"""Overlay marker-detector blobs on the rectified left view of a saved frame (.npy or .raw)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from cloudclean.capture.metroy import markers
from cloudclean.capture.metroy.stripes import Triangulator
HERE = Path(__file__).resolve().parent
tri = Triangulator.from_yaml(str(HERE / "camparam.yaml"))
src = sys.argv[1]
img = np.load(src) if src.endswith(".npy") else np.fromfile(src, np.uint8).reshape(2400, 1600)
rl, rr = tri.rectify(img)
bl = markers.detect(rl)
vis = cv2.cvtColor(rl, cv2.COLOR_GRAY2BGR)
for b in bl:
    cv2.circle(vis, (int(b.x), int(b.y)), int(b.radius) + 6, (0, 0, 255), 2)
print(len(bl), "blobs; radii", sorted(round(b.radius, 1) for b in bl)[:20])
x0, y0, w, h = (int(v) for v in sys.argv[3:7]) if len(sys.argv) > 6 else (0, 0, 1600, 1200)
cv2.imwrite(sys.argv[2], cv2.resize(vis[y0:y0 + h, x0:x0 + w], (min(1600, w * 2) if w < 800 else 800,
                                                                  (min(1600, w * 2) if w < 800 else 800) * h // w)))
