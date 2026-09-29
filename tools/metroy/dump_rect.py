"""Write rectified left|right of a frame as a PNG (contrast-stretched) for eyeballing."""
import sys
import cv2
import numpy as np
from laser_scan import rectifiers
from rectify_check import load_calib, split
raw, calib, out = sys.argv[1:4]
c = load_calib(calib)
m1, m2, *_ = rectifiers(c, calib)
L, R = split(raw, c["size"])
rl, rr = cv2.remap(L, *m1, cv2.INTER_LINEAR), cv2.remap(R, *m2, cv2.INTER_LINEAR)
both = np.concatenate([rl, rr], 1).astype(np.float32)
both = np.clip(both * 2.0, 0, 255).astype(np.uint8)
cv2.imwrite(out, cv2.resize(both, (1600, 600), interpolation=cv2.INTER_AREA))
