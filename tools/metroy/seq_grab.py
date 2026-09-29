"""Grab a run of consecutive laser frames and report, per frame, which line family is lit and the timing."""
import sys
import time

import numpy as np

from laser_test import Stream
from metroy_hid import MetroyHid
import stripes as S

n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/seq"
tri = S.Triangulator.from_yaml(str(S.HERE / "camparam.yaml"))
s = Stream()
try:
    with MetroyHid() as hid:
        hid.preisp(0xB04, 1)
    time.sleep(0.5)
    s.latest(3)
    frames, stamps = [], []
    for i in range(n):
        frames.append(s.frame())
        stamps.append(time.monotonic())
    with MetroyHid() as hid:
        hid.preisp(0xB04, 0)
finally:
    s.close()
dt = np.diff(stamps)
print(f"{n} frames, interval median {1000 * np.median(dt):.0f} ms (min {1000 * dt.min():.0f}, max {1000 * dt.max():.0f})")
for i, f in enumerate(frames):
    f.tofile(f"{out}_{i:02d}.raw")
    rl, _ = tri.rectify(f)
    c = S.stripe_centres(rl, tri.p)
    sl = np.median(c.slope) if len(c.slope) else np.nan
    print(f"frame {i:2d}: mean {f.mean():5.1f}  lit {100 * (f > 200).mean():4.2f}%  centres {len(c.x):6d}  stripe slope dx/dy {sl:+.2f}")
