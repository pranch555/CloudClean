"""Full-resolution crop of the rectified left view, and pixel statistics of the brightest blobs (marker hunting)."""
import sys
import cv2
import numpy as np
import stripes as S
raw, calib, out = sys.argv[1:4]
x0, y0, w, h = (int(v) for v in sys.argv[4:8])
tri = S.Triangulator.from_yaml(calib)
rl, rr = tri.rectify(S.load_raw(raw))
crop = rl[y0:y0 + h, x0:x0 + w]
cv2.imwrite(out, cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST))
# connected bright components: size, fill, aspect
_, bw = cv2.threshold(rl, 200, 255, cv2.THRESH_BINARY)
n, lab, st, cen = cv2.connectedComponentsWithStats(bw)
big = [(i, st[i]) for i in range(1, n) if st[i][4] >= 20]
ars = []
for i, s in big:
    bw_, bh_ = s[2], s[3]
    fill = s[4] / (bw_ * bh_)
    ars.append((fill, max(bw_, bh_) / max(1, min(bw_, bh_)), s[4], bw_, bh_, cen[i]))
blobby = [a for a in ars if a[0] > 0.5 and a[1] < 2.5]
print(f"{len(big)} bright components >= 20 px; blob-like (fill>0.5, aspect<2.5): {len(blobby)}")
for a in sorted(blobby, key=lambda a: -a[2])[:15]:
    print(f"   area {a[2]:4d} px  box {a[3]}x{a[4]}  fill {a[0]:.2f}  at ({a[5][0]:.0f},{a[5][1]:.0f})")
