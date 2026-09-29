"""Does register 0xb07 drive the IR fill light (which lights retro-reflective markers)? Laser off throughout."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from cloudclean.capture.metroy.hid import MetroyHid
from cloudclean.capture.metroy.v4l2 import Stream


def grab(s, n=4):
    f = None
    for _ in range(n):
        f = s.read(1.0) or f
    return f.image


s = Stream()
try:
    with MetroyHid() as hid:
        hid.laser(False)
        time.sleep(0.4)
        for label, value in (("fill 0", 0), ("fill 30", 30), ("fill 60", 60), ("fill 0 again", 0)):
            hid.preisp(0xB07, value)
            time.sleep(0.5)
            img = grab(s)
            top = img[:1200]
            print(f"{label:13s} mean {top.mean():6.1f}  p99 {np.percentile(top, 99):5.0f}  saturated {100 * (top >= 250).mean():.3f}%")
            np.save(f"/tmp/fill_{value}.npy", img)
        hid.preisp(0xB07, 0)
finally:
    s.close()
