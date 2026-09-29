"""Scan a vertical offset of the rectified right image and watch the two cross-line families' disagreement.

If the factory rectification leaves the right image dy rows off, a stripe of slope s gets a disparity error s*dy;
the families have opposite slopes, so they disagree by (s1 - s0)*dy. The dy that makes them agree is measured,
not assumed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from scipy.spatial import cKDTree
import stripes as S
from flatness import fit_plane

*frames, calib = sys.argv[1:]
tri = S.Triangulator.from_yaml(calib, extra=S.default_extra(calib))
imgs = [np.load(f) if f.endswith(".npy") else S.load_raw(f) for f in frames]
# float maps so the right view can be shifted by a fraction of a row
calib_d, P1, P2, Q = S.read_calibration(calib)
R1, R2, *_ = cv2.stereoRectify(calib_d["KL"], calib_d["DL"], calib_d["KR"], calib_d["DR"], calib_d["size"],
                               calib_d["R"], calib_d["T"], flags=0)
mx, my = cv2.initUndistortRectifyMap(calib_d["KR"], calib_d["DR"], R2, P2, calib_d["size"], cv2.CV_32FC1)
lefts = [tri.rectify(im)[0] for im in imgs]
for dy in (-0.6, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4):
    # rectified right pixel (x, y) now samples where the factory map puts (x, y + dy)
    ys = np.arange(my.shape[0], dtype=np.float32)[:, None]
    mx2 = cv2.remap(mx, np.tile(np.arange(mx.shape[1], dtype=np.float32), (mx.shape[0], 1)), np.broadcast_to(ys + dy, mx.shape).astype(np.float32), cv2.INTER_LINEAR)
    my2 = cv2.remap(my, np.tile(np.arange(my.shape[1], dtype=np.float32), (my.shape[0], 1)), np.broadcast_to(ys + dy, my.shape).astype(np.float32), cv2.INTER_LINEAR)
    fam = {0: [], 1: []}
    for im, rl in zip(imgs, lefts):
        rr = cv2.remap(im[1200:], mx2, my2, cv2.INTER_LINEAR)
        pts = tri.points_rectified(rl, rr)
        fam[tri.last["family"]].append(pts)
    A, B = np.concatenate(fam[0]), np.concatenate(fam[1])
    c, vt, r, inl = fit_plane(np.concatenate([A, B]), gate=1.0)
    n = vt[2]
    A = A[np.abs((A - c) @ n) < 1.5]; B = B[np.abs((B - c) @ n) < 1.5]
    dd, ii = cKDTree(A[:, :2]).query(B[:, :2], distance_upper_bound=0.3)
    ok = np.isfinite(dd)
    dz = (B[ok] - A[ii[ok]]) @ n
    print(f"dy {dy:+.2f} px: crossings {ok.sum():5d}, family offset {np.median(dz) * 1000:+5.0f} um, spread {1.4826 * np.median(np.abs(dz - np.median(dz))) * 1000:4.0f} um")
