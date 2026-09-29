"""Which SGBM settings survive the strong foreshortening between the two MetroY views?"""
import sys

import cv2
import numpy as np

from rectify_check import load_calib, split

raw, calib_path = sys.argv[1], sys.argv[2]
c = load_calib(calib_path)
fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
P1, P2, Q = fs.getNode("pL").mat(), fs.getNode("pR").mat(), fs.getNode("Q").mat()
R1, R2, *_ = cv2.stereoRectify(c["KL"], c["DL"], c["KR"], c["DR"], c["size"], c["R"], c["T"], flags=0)
left, right = split(raw, c["size"])
m1 = cv2.initUndistortRectifyMap(c["KL"], c["DL"], R1, P1, c["size"], cv2.CV_32FC1)
m2 = cv2.initUndistortRectifyMap(c["KR"], c["DR"], R2, P2, c["size"], cv2.CV_32FC1)
rl, rr = cv2.remap(left, *m1, cv2.INTER_LINEAR), cv2.remap(right, *m2, cv2.INTER_LINEAR)
clahe = cv2.createCLAHE(3.0, (8, 8))
rl, rr = clahe.apply(rl), clahe.apply(rr)
roi = rl > 40                                    # the keyboard, not the black desk

f, B = P1[0, 0], -P2[0, 3] / P2[0, 0]
off = P1[0, 2] - P2[0, 2]
for scale in (1.0, 0.5):
    L = cv2.resize(rl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    R = cv2.resize(rr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(roi.astype(np.uint8), (L.shape[1], L.shape[0])) > 0
    lo = int(np.floor((f * B / 600 + off) * scale / 16) * 16)
    num = int(np.ceil(((f * B / 140 + off) * scale - lo) / 16) * 16)
    for block in (7, 11, 15, 21):
        s = cv2.StereoSGBM_create(minDisparity=lo, numDisparities=num, blockSize=block, P1=8 * block * block,
                                  P2=32 * block * block, uniquenessRatio=5, speckleWindowSize=200, speckleRange=4,
                                  disp12MaxDiff=2, mode=cv2.STEREO_SGBM_MODE_HH)
        d = s.compute(L, R).astype(np.float32) / 16.0
        ok = (d > lo) & mask
        z = f * B / ((d[ok] / scale) - off) if ok.any() else np.array([np.nan])
        print(f"scale {scale:.1f} block {block:2d}: valid {ok.sum() / mask.sum():5.1%} of keyboard, "
              f"depth median {np.median(z):6.1f} mm, IQR {np.subtract(*np.percentile(z, [75, 25])):5.1f} mm")
        if scale == 0.5 and block == 15:
            np.save("/tmp/disp_half.npy", d)
