"""Save the factory-rectified stereo pair side by side with row guides, to eyeball the epipolar alignment."""
import sys

import cv2
import numpy as np

from rectify_check import load_calib, split

raw, calib_path = sys.argv[1], sys.argv[2]
c = load_calib(calib_path)
fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
P1, P2 = fs.getNode("pL").mat(), fs.getNode("pR").mat()
R1, R2, *_ = cv2.stereoRectify(c["KL"], c["DL"], c["KR"], c["DR"], c["size"], c["R"], c["T"], flags=0)
left, right = split(raw, c["size"])
m1 = cv2.initUndistortRectifyMap(c["KL"], c["DL"], R1, P1, c["size"], cv2.CV_32FC1)
m2 = cv2.initUndistortRectifyMap(c["KR"], c["DR"], R2, P2, c["size"], cv2.CV_32FC1)
rl, rr = cv2.remap(left, *m1, cv2.INTER_LINEAR), cv2.remap(right, *m2, cv2.INTER_LINEAR)
print("nonblack fraction L/R:", round(float((rl > 3).mean()), 2), round(float((rr > 3).mean()), 2))
both = cv2.cvtColor(np.concatenate([rl, np.full((rl.shape[0], 12), 80, np.uint8), rr], 1), cv2.COLOR_GRAY2BGR)
for y in range(40, both.shape[0], 60):
    both[y] = (0, 200, 255)
cv2.imwrite("/tmp/rect_pair.png", cv2.resize(both, (both.shape[1] // 2, both.shape[0] // 2)))
