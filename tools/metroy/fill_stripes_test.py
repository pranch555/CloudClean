"""With the fill light on (as marker tracking needs), are stripes still triangulated as well, and are there false
markers? Grabs a few frames at fill 0 and fill 30 with the laser on and compares."""
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
s = Stream()
try:
    with MetroyHid() as hid:
        hid.laser(True)
        for fill in (0, 30, 0):
            hid.preisp(REG_FILL_LIGHT, fill)
            time.sleep(0.5)
            for _ in range(4):
                s.read(1.0)
            counts, mk, paired = [], [], []
            for i in range(6):
                f = s.read(1.0)
                rl, rr = tri.rectify(f.image)
                pts = tri.points_rectified(rl, rr)
                counts.append(len(pts)); paired.append(tri.last.get("paired", 0))
                bl, br = markers.detect(rl), markers.detect(rr)
                mk.append((len(bl), len(br), len(markers.stereo(bl, br, tri.Q))))
                if i == 0:
                    np.save(f"/tmp/fill{fill}_frame.npy", f.image)
            print(f"fill {fill:3d}: points/frame {int(np.median(counts))}, paired {np.median(paired):.0%}, "
                  f"marker blobs L/R/3D {mk}")
        hid.laser(False)
        hid.preisp(REG_FILL_LIGHT, 0)
finally:
    s.close()
