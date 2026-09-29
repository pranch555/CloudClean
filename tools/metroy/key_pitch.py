"""Metric accuracy check: measure a keyboard's key pitch (standard 19.05 mm) from one MetroY stereo frame.

Rectifies with the factory calibration exactly as Revopoint does (no zero-disparity; the factory's own pL/pR/Q),
computes dense SGBM disparity, finds the light keycaps in the left image, triangulates each keycap's centre, and
reports the 3D distances between neighbouring keys.

    python key_pitch.py frame.raw camparam.yaml
"""
import sys

import cv2
import numpy as np

from rectify_check import load_calib, split

PITCH_MM = 19.05

raw, calib_path = sys.argv[1], sys.argv[2]
c = load_calib(calib_path)
fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
P1, P2, Q = fs.getNode("pL").mat(), fs.getNode("pR").mat(), fs.getNode("Q").mat()
R1, R2, *_ = cv2.stereoRectify(c["KL"], c["DL"], c["KR"], c["DR"], c["size"], c["R"], c["T"], flags=0)
left, right = split(raw, c["size"])
m1 = cv2.initUndistortRectifyMap(c["KL"], c["DL"], R1, P1, c["size"], cv2.CV_32FC1)
m2 = cv2.initUndistortRectifyMap(c["KR"], c["DR"], R2, P2, c["size"], cv2.CV_32FC1)
rl, rr = cv2.remap(left, *m1, cv2.INTER_LINEAR), cv2.remap(right, *m2, cv2.INTER_LINEAR)

# disparity window from the rectified geometry for 150-600 mm: d = f*B/Z + (cxL - cxR)
f, B = P1[0, 0], -P2[0, 3] / P2[0, 0]
off = P1[0, 2] - P2[0, 2]
dmin, dmax = f * B / 600 + off, f * B / 150 + off
min_disp = int(np.floor(dmin / 16) * 16)
num = int(np.ceil((dmax - min_disp) / 16) * 16)
sgbm = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num, blockSize=7, P1=8 * 49, P2=32 * 49,
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                             mode=cv2.STEREO_SGBM_MODE_SGBM)
disp = sgbm.compute(rl, rr).astype(np.float32) / 16.0
valid = disp > min_disp
xyz = cv2.reprojectImageTo3D(disp, Q)
print(f"baseline {B:.2f} mm, f {f:.1f} px, disparity window {min_disp}..{min_disp + num}, "
      f"valid disparity on {valid.mean():.0%} of the left image")

# keycaps: bright blobs separated by dark gaps
blur = cv2.GaussianBlur(rl, (5, 5), 0)
_, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # fill the printed legends
bright = cv2.erode(bright, np.ones((7, 7), np.uint8))                            # split touching keys
n, lab, stats, cents = cv2.connectedComponentsWithStats(bright, 8)
areas = stats[1:, cv2.CC_STAT_AREA]
typical = np.median(areas[areas > 1500]) if (areas > 1500).any() else 0
keys = []
for i in range(1, n):
    a = stats[i, cv2.CC_STAT_AREA]
    if not (0.5 * typical < a < 1.6 * typical):          # 1u keys only; wide keys have other pitches
        continue
    region = (lab == i) & valid
    if region.sum() < 0.3 * a:
        continue
    keys.append(np.median(xyz[region], axis=0))
keys = np.array(keys)
print(f"1u keycaps found with depth: {len(keys)} (typical area {typical:.0f} px)")
if len(keys) < 4:
    sys.exit("too few keys to judge")

print(f"keyboard distance: median {np.median(np.linalg.norm(keys, axis=1)):.1f} mm")
D = np.linalg.norm(keys[:, None] - keys[None], axis=2)
np.fill_diagonal(D, np.inf)
nn = D.min(1)
on_grid = nn[np.abs(nn - PITCH_MM) < 4]
print(f"nearest-neighbour distances (mm): {np.round(np.sort(nn), 2)}")
if len(on_grid):
    err = on_grid.mean() - PITCH_MM
    print(f"\nKEY PITCH: {on_grid.mean():.3f} mm  (standard {PITCH_MM}), error {err:+.3f} mm "
          f"({100 * err / PITCH_MM:+.2f}%), spread {on_grid.std():.3f} mm over {len(on_grid)} keys")

vis = cv2.cvtColor(rl, cv2.COLOR_GRAY2BGR)
vis[bright > 0] = (0.6 * vis[bright > 0] + (0, 90, 0)).astype(np.uint8)
cv2.imwrite("/tmp/keys_left.png", cv2.resize(vis, (800, 600)))
dv = np.where(valid, (disp - min_disp) / num * 255, 0).astype(np.uint8)
cv2.imwrite("/tmp/keys_disp.png", cv2.resize(cv2.applyColorMap(dv, cv2.COLORMAP_TURBO), (800, 600)))
