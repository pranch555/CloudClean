"""Command-line access to the MetroY triangulation, for validation scripts.

The implementation lives in `cloudclean.capture.metroy.stripes` - the exact code the capture driver runs - so every
number these scripts report is a number the driver would produce. It needs only numpy and OpenCV.

    python stripes.py frame.raw camparam.yaml [metroExtra.bin] [out.ply]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloudclean.capture.metroy.stripes import (  # noqa: E402,F401
    Centres, Params, Triangulator, agreement, load_raw, match_row, read_calibration, smooth_tracks, stripe_centres)

HERE = Path(__file__).resolve().parent


def default_extra(calib_path: str) -> str | None:
    for candidate in (Path(calib_path).resolve().parent / "metroExtra.bin", HERE / "metroExtra.bin"):
        if candidate.exists():
            return str(candidate)
    return None


def frame_points(raw_path: str, calib_path: str, method: str | None = None, extra: str | None = None):
    """Points of one saved frame by any method, for side-by-side validation.

    method: 'lines' (laser line identity; the default), 'order' (stereo order), 'steger-notrack', 'v2', 'v1';
    the METROY_METHOD environment variable sets it for scripts that do not pass one."""
    method = method or os.environ.get("METROY_METHOD", "lines")
    if method in ("steger", "steger+track"):
        method = "order"
    if method == "v1":
        from laser_scan import scan
        return scan(raw_path, calib_path)[0]
    if method == "v2":
        from laser_scan2 import scan2
        return scan2(raw_path, calib_path)[0]
    tri = _cached(calib_path, (extra or default_extra(calib_path)) if method == "lines" else None)
    return tri.points(load_raw(raw_path, tri.size), track=method != "steger-notrack",
                      matching="lines" if method == "lines" else "order")


_TRI: dict[tuple, Triangulator] = {}


def _cached(calib_path: str, extra: str | None) -> Triangulator:
    if (calib_path, extra) not in _TRI:
        _TRI[calib_path, extra] = Triangulator.from_yaml(calib_path, extra=extra)
    return _TRI[calib_path, extra]


if __name__ == "__main__":
    import time

    import numpy as np
    args = sys.argv[1:]
    extra = next((a for a in args if a.endswith(".bin")), None) or default_extra(args[1])
    out = next((a for a in args[2:] if a.endswith(".ply")), None)
    tri = Triangulator.from_yaml(args[1], extra=extra)
    frame = load_raw(args[0])
    t0 = time.perf_counter()
    pts = tri.points(frame)
    ms = 1000 * (time.perf_counter() - t0)
    print(f"{len(pts)} points in {ms:.0f} ms" + (f", depth median {np.median(pts[:, 2]):.1f} mm" if len(pts) else ""),
          tri.last)
    if out:
        from laser_scan import write_ply
        write_ply(out, pts)
