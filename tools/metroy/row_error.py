"""Vertical misalignment of the rectified pair, measured directly from point features (SIFT) in a fill-lit, laser-off
frame, with exactly the pipeline's rectification. dy = y_right - y_left for matched features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
import stripes as S

src, calib = sys.argv[1:3]
tri = S.Triangulator.from_yaml(calib)
img = np.load(src) if src.endswith(".npy") else S.load_raw(src)
rl, rr = tri.rectify(img)
clahe = cv2.createCLAHE(3.0, (8, 8))
a, b = clahe.apply(rl), clahe.apply(rr)
sift = cv2.SIFT_create(8000)
ka, da = sift.detectAndCompute(a, None)
kb, db = sift.detectAndCompute(b, None)
m = cv2.BFMatcher().knnMatch(da, db, k=2)
good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
dy = pb[:, 1] - pa[:, 1]
d = pa[:, 0] - pb[:, 0]
ok = (np.abs(dy - np.median(dy)) < 2) & (d > -400) & (d < 100)
print(f"{len(good)} matches, {ok.sum()} consistent; dy median {np.median(dy[ok]):+.3f} px, mean {dy[ok].mean():+.3f}, "
      f"robust sd {1.4826 * np.median(np.abs(dy[ok] - np.median(dy[ok]))):.3f} px, sem {dy[ok].std() / np.sqrt(ok.sum()):.3f}")
# does it vary across the image? (a rotation/scale residual would)
for name, sel in (("left half", pa[:, 0] < 800), ("right half", pa[:, 0] >= 800), ("top", pa[:, 1] < 600), ("bottom", pa[:, 1] >= 600)):
    s = ok & sel
    if s.sum() > 20:
        print(f"   {name:10s}: n {s.sum():4d}, dy median {np.median(dy[s]):+.3f}")
