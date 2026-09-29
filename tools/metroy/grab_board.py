"""Laser off, then grab the board at a few exposures (register 0x910), laser back on at the end."""
import sys
import time

import numpy as np

from laser_test import Stream
from metroy_hid import MetroyHid

exposures = [int(e) for e in sys.argv[1:]] or [8000, 16000, 30000]
s = Stream()
try:
    with MetroyHid() as hid:
        hid.preisp(0xB04, 0)
        for e in exposures:
            hid.preisp(0x910, e)
            time.sleep(0.8)
            img = s.latest(4)
            np.save(f"/tmp/board_off_{e}.npy", img)
            print(f"exposure {e:6d}: mean {img.mean():5.1f}  p99 {np.percentile(img, 99):5.1f}  "
                  f"saturated {100 * (img >= 250).mean():.2f}%")
        hid.preisp(0x910, 8000)
        hid.preisp(0xB04, 1)
finally:
    s.close()
