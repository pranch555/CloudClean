"""Characterise the laser stripes in a frame: orientation, width, peak level, saturation."""
import sys

import cv2
import numpy as np

from laser_scan import rectifiers
from laser_scan2 import peaks
from rectify_check import load_calib, split

raw, calib = sys.argv[1], sys.argv[2]
c = load_calib(calib)
m1, m2, *_ = rectifiers(c, calib)
L, R = split(raw, c["size"])
for name, img, m in (("left", L, m1), ("right", R, m2)):
    rimg = cv2.remap(img, *m, cv2.INTER_LINEAR)
    for tag, im in (("raw", img), ("rect", rimg)):
        widths, tops, sat = [], [], 0
        for y in range(300, 900, 10):
            row = im[y].astype(np.float32)
            for p in peaks(row, 45, 18):
                i = int(round(p))
                top = row[i]
                half = top / 2
                lo = i
                while lo > 0 and row[lo] > half: lo -= 1
                hi = i
                while hi < len(row) - 1 and row[hi] > half: hi += 1
                widths.append(hi - lo)
                tops.append(top)
                sat += top >= 254
        widths, tops = np.array(widths), np.array(tops)
        print(f"{name:5s} {tag:4s}: peaks/row {len(widths)/60:.1f}  FWHM px median {np.median(widths):.1f} "
              f"(5-95% {np.percentile(widths,5):.0f}-{np.percentile(widths,95):.0f})  peak median {np.median(tops):.0f} "
              f"saturated {sat/len(tops):.0%}")
    # print one profile around a peak
    row = rimg[600].astype(np.float32)
    p = peaks(row, 45, 18)
    if p:
        i = int(round(p[len(p)//2]))
        print(f"  {name} row 600 profile around x={i}:", row[i-6:i+7].astype(int).tolist())
