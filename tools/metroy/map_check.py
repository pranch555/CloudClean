"""Build a marker map for a few seconds (static scanner) and check it for duplicate markers."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import numpy as np
    from cloudclean.capture.drivers import create_driver
    drv = create_driver("metroy_usb")
    drv.connect({"tracking": "markers"})
    drv.command("map_markers")
    drv.start()
    t0, counts = time.monotonic(), []
    while time.monotonic() - t0 < 6:
        f = drv.read(0.2)
        if f is not None:
            counts.append(f.meta["markers"])
    info = drv.command("finish_map")
    W = drv._tracker.world
    d = np.linalg.norm(W[:, None] - W[None], axis=2) + np.eye(len(W)) * 1e9
    print(f"markers per frame median {np.median(counts):.0f} (max {max(counts)}); map {len(W)}; refine {info}")
    print("nearest-neighbour distances in the map (mm):", np.sort(d.min(1)).round(2))
    drv.stop(); drv.disconnect()


if __name__ == "__main__":
    main()
