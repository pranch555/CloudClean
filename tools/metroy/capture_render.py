"""Capture a few seconds (static scanner) with the real session and render the fused cloud from the scanner's
viewpoint and from above, to eyeball against Revo Metro's view."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import cv2
    import numpy as np
    from cloudclean.capture.drivers import create_driver
    from cloudclean.capture.session import CaptureSession
    surface = sys.argv[1] if len(sys.argv) > 1 else "general"
    drv = create_driver("metroy_usb")
    s = CaptureSession(drv, {"tracking": "markers", "surface": surface})
    s.connect(); s.start(); time.sleep(6); s.stop()
    P = s.disp_xyz[:s.disp_n].astype(float)
    T = [t["pose"] for t in s.trajectory if t["pose"] is not None][-1]
    local = (P - T[:3, 3]) @ T[:3, :3]                      # back into the scanner frame
    print(f"frames {s.stats['frames']}, fused {s.stats['fused_frames']}, display points {len(P)}")
    f, cx, cy = 1813.6, 383.6, 600
    img = np.zeros((1200, 1600), np.uint8)
    ok = local[:, 2] > 50
    u = (f * local[ok, 0] / local[ok, 2] + cx).astype(int); v = (f * local[ok, 1] / local[ok, 2] + cy).astype(int)
    inb = (u >= 0) & (u < 1600) & (v >= 0) & (v < 1200)
    img[v[inb], u[inb]] = 255
    cv2.imwrite("/tmp/cap_scanner_view.png", cv2.resize(cv2.dilate(img, np.ones((3, 3))), (800, 600)))
    z = local[:, 2]
    top = np.zeros((600, 800, 3), np.uint8)
    lo, hi = np.percentile(local[:, :2], 1, 0), np.percentile(local[:, :2], 99, 0)
    uu = ((local[:, 0] - lo[0]) / (hi[0] - lo[0]) * 799).astype(int); vv = ((local[:, 1] - lo[1]) / (hi[1] - lo[1]) * 599).astype(int)
    col = cv2.applyColorMap(np.clip((z - np.percentile(z, 2)) / (np.ptp(np.percentile(z, [2, 98])) + 1e-6) * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    m = (uu >= 0) & (uu < 800) & (vv >= 0) & (vv < 600)
    top[vv[m], uu[m]] = col[m]
    cv2.imwrite("/tmp/cap_depth.png", top)


if __name__ == "__main__":
    main()
