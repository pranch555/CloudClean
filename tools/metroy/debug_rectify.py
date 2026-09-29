"""Where does matching fail: raw images, or after rectification? Compare my rectification with the factory's own."""
import sys

import cv2
import numpy as np

from rectify_check import load_calib, split

raw, calib_path = sys.argv[1], sys.argv[2]
c = load_calib(calib_path)
top, bottom = split(raw, c["size"])
sift = cv2.SIFT_create(4000)


def count(a, b, label):
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        print(f"{label}: keypoints {len(ka)}/{len(kb)}, no descriptors")
        return None
    m = [x for x, y in (p for p in cv2.BFMatcher().knnMatch(da, db, k=2) if len(p) == 2) if x.distance < 0.75 * y.distance]
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    dy = (pb[:, 1] - pa[:, 1]) if len(m) else np.array([np.nan])
    print(f"{label}: keypoints {len(ka)}/{len(kb)}, ratio-test matches {len(m)}, "
          f"dy median {np.median(dy):+.2f} px, |dy|<2px: {(np.abs(dy - np.median(dy)) < 2).sum()}")
    return pa, pb


count(top, bottom, "raw top vs bottom")

fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
P1f, P2f = fs.getNode("pL").mat(), fs.getNode("pR").mat()
for alpha in (0, 1, -1):
    R1, R2, P1, P2, Q, r1, r2 = cv2.stereoRectify(c["KL"], c["DL"], c["KR"], c["DR"], c["size"], c["R"], c["T"],
                                                  flags=cv2.CALIB_ZERO_DISPARITY if alpha >= 0 else 0,
                                                  alpha=max(alpha, 0))
    print(f"\nalpha={alpha}: my P1 f={P1[0,0]:.1f} cx={P1[0,2]:.1f} | P2 cx={P2[0,2]:.1f} Tx={P2[0,3]:.1f}  "
          f"valid ROI L={r1} R={r2}")
    for name, (L, R) in {"top=L": (top, bottom), "top=R": (bottom, top)}.items():
        m1 = cv2.initUndistortRectifyMap(c["KL"], c["DL"], R1, P1, c["size"], cv2.CV_32FC1)
        m2 = cv2.initUndistortRectifyMap(c["KR"], c["DR"], R2, P2, c["size"], cv2.CV_32FC1)
        rl, rr = cv2.remap(L, *m1, cv2.INTER_LINEAR), cv2.remap(R, *m2, cv2.INTER_LINEAR)
        print(f"   {name}: nonblack L {(rl > 5).mean():.2f}, R {(rr > 5).mean():.2f}", end="  ")
        count(rl, rr, "rectified")
print(f"\nfactory pL cx={P1f[0,2]:.1f}, pR cx={P2f[0,2]:.1f}, pR Tx={P2f[0,3]:.1f}")
