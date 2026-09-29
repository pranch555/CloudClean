"""Is register 0x903 the gain (0x10 = 1x, inferred 0x20 = 2x)? Laser on, fill off, exposure fixed; compare stripe
peak brightness at 0x10 and 0x20, then restore 0x10."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from cloudclean.capture.metroy.hid import MetroyHid
from cloudclean.capture.metroy.v4l2 import Stream

s = Stream()
try:
    with MetroyHid() as hid:
        hid.laser(True)
        hid.shell("echo s 0x910 8000 > /dev/rk_preisp; echo s 0x911 1000 > /dev/rk_preisp")
        time.sleep(1.5)
        for gain in ("0x10", "0x20", "0x10", "0x20", "0x10"):
            hid.shell(f"echo s 0x903 {gain} > /dev/rk_preisp")
            time.sleep(1.5)
            for _ in range(8):
                s.read(1.0)            # drain frames exposed before the change
            imgs = []
            for _ in range(6):
                f = s.read(2.0)
                if f is not None:
                    imgs.append(f.image[:1200].astype(np.float32))
            g = cv2.GaussianBlur(np.mean(imgs, 0), (0, 0), 2)
            print(f"gain {gain}: frames {len(imgs)}/6, background median {np.median(g):5.1f}, 99th pct {np.percentile(g, 99):5.1f}, "
                  f"99.9th {np.percentile(g, 99.9):5.1f}")
        hid.shell("echo s 0x910 8000 > /dev/rk_preisp; echo s 0x911 200 > /dev/rk_preisp")
        hid.laser(False)
finally:
    s.close()
