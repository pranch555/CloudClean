"""Revo Metro's acquisition exposure (frame time 0x910 = 8000 us, exposure 0x911 = 200 us), written as ONE shell
command so 0x910 is never left alone. Reports stream health, background level, stripes and marker blobs."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from cloudclean.capture.metroy import markers
from cloudclean.capture.metroy.hid import REG_FILL_LIGHT, MetroyHid
from cloudclean.capture.metroy.stripes import Triangulator
from cloudclean.capture.metroy.v4l2 import Stream

HERE = Path(__file__).resolve().parent
tri = Triangulator.from_yaml(str(HERE / "camparam.yaml"), extra=str(HERE / "metroExtra.bin"))
frame_us, exposure_us = (int(v) for v in sys.argv[1:3]) if len(sys.argv) > 2 else (8000, 200)


def measure(s, label):
    for _ in range(4):
        s.read(1.0)
    got, counts, mk, bg = 0, [], [], []
    t0 = time.monotonic()
    for i in range(8):
        f = s.read(2.0)
        if f is None:
            continue
        got += 1
        rl, rr = tri.rectify(f.image)
        counts.append(len(tri.points_rectified(rl, rr)))
        bl, br = markers.detect(rl), markers.detect(rr)
        mk.append((len(bl), len(br)))
        bg.append(np.median(rl))
        if i == 0:
            np.save("/tmp/exp_" + "".join(c if c.isalnum() else "_" for c in label) + ".npy", f.image)
    print(f"{label:24s} frames {got}/8 in {time.monotonic() - t0:.1f}s  background median {np.median(bg):5.1f}  "
          f"points/frame {int(np.median(counts)) if counts else 0}  marker blobs L/R {mk[:4]}")


s = Stream()
try:
    with MetroyHid() as hid:
        hid.laser(True)
        hid.preisp(REG_FILL_LIGHT, 0)
        measure(s, "boot exposure, fill 0")
        hid.shell(f"echo s 0x910 {frame_us} > /dev/rk_preisp; echo s 0x911 {exposure_us} > /dev/rk_preisp")
        time.sleep(0.5)
        measure(s, f"{frame_us}/{exposure_us} us, fill 0")
        hid.preisp(REG_FILL_LIGHT, 30)
        time.sleep(0.4)
        measure(s, f"{frame_us}/{exposure_us} us, fill 30")
        hid.preisp(REG_FILL_LIGHT, 0)
        hid.laser(False)
finally:
    s.close()
