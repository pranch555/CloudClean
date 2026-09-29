"""Run the MetroY capture engine live for a few seconds and report throughput and per-frame results."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from cloudclean.capture.metroy.scanner import MetroyScanner


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 6
    matching = sys.argv[2] if len(sys.argv) > 2 else "lines"
    t_open = time.perf_counter()
    with MetroyScanner(cache_dir=Path("/tmp/metroy-cache"), matching=matching) as sc:
        print(f"opened in {time.perf_counter() - t_open:.1f} s: serial {sc.files.serial}, calibration "
              f"{len(sc.files.calibration)} B, laser file {len(sc.files.laser or b'')} B, {sc.workers} workers")
        sc.start()
        t0, got, prev = time.monotonic(), [], None
        while time.monotonic() - t0 < secs:
            f = sc.next(0.2)
            if f is None:
                continue
            got.append(f)
        stats = dict(sc.stats)
    dts = np.diff([f.timestamp for f in got])
    print(f"delivered {len(got)} frames in {secs:.0f} s ({len(got) / secs:.1f} fps); stats {stats}")
    if len(got):
        print(f"frame gap median {1000 * np.median(dts):.0f} ms, max {1000 * dts.max():.0f} ms; "
              f"processing median {1000 * np.median([f.process_s for f in got]):.0f} ms; "
              f"points median {int(np.median([len(f.points) for f in got]))}, depth median "
              f"{np.median([np.median(f.points[:, 2]) for f in got if len(f.points)]):.0f} mm")


if __name__ == "__main__":     # required: the worker processes re-import this module
    main()
