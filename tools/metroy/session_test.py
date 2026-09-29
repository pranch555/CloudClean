"""Drive CloudClean's real CaptureSession with the native MetroY driver for a few seconds, headless.

    python tools/metroy/session_test.py [seconds] [markers|geometry] [general|dark|reflective]
Reports what the session did with the frames and, with the scanner held still, how still the solved pose is."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import numpy as np
    from cloudclean.capture.drivers import create_driver
    from cloudclean.capture.session import CaptureSession

    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 8
    tracking = sys.argv[2] if len(sys.argv) > 2 else "markers"
    surface = sys.argv[3] if len(sys.argv) > 3 else "general"
    drv = create_driver("metroy_usb")
    print("available:", drv.availability())
    s = CaptureSession(drv, {"tracking": tracking, "surface": surface})
    s.connect()
    s.start()
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < secs:
            time.sleep(1.0)
            st = s.status()
            g = s.update_guidance()
            print({k: st.get(k) for k in ("frames", "fused_frames", "tracking", "points")},
                  {k: drv.status().get(k) for k in ("processed", "errors")}, "|", g["status"]["message"][:100])
    finally:
        s.stop()
    posed = [t for t in s.trajectory if t["pose"] is not None and t["tracking"] != "lost"]
    if posed:
        P = np.array([t["pose"][:3, 3] for t in posed])
        print(f"posed frames {len(P)} of {len(s.trajectory)}; pose position spread (sd, mm) {np.round(P.std(0), 4)}")
    if drv._tracker is not None:
        print("marker map:", len(drv._tracker.world), "markers")


if __name__ == "__main__":     # required: the worker processes re-import this module
    main()
