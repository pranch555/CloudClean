"""Validate the MetroY factory calibration against a live frame.

A YUYV 1600x1200 frame from the IR node is really 8-bit 1600x2400: one camera stacked on the other. Rectifying with
the calibration read off the scanner must put every matched feature on the same row (vertical disparity ~ 0) and
give positive horizontal disparity; both hold only with the right calibration *and* the right camera assignment, so
the test tries both assignments and reports the residuals.

    python rectify_check.py frame.raw camparam.yaml
"""
import sys

import cv2
import numpy as np


def load_calib(path: str) -> dict:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    get = lambda k: fs.getNode(k).mat()
    return {"KL": get("cameraMatrixL"), "DL": get("distCoeffL"), "KR": get("cameraMatrixR"),
            "DR": get("distCoeffR"), "R": get("steroRotation"), "T": get("steroTranslation"),
            "size": (int(fs.getNode("width").string()), int(fs.getNode("height").string()))}


def split(raw_path: str, size=(1600, 1200)):
    w, h = size
    raw = np.fromfile(raw_path, np.uint8).reshape(2 * h, w)
    return raw[:h], raw[h:]


def residuals(left, right, c):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(c["KL"], c["DL"], c["KR"], c["DR"], c["size"], c["R"], c["T"],
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1 = cv2.initUndistortRectifyMap(c["KL"], c["DL"], R1, P1, c["size"], cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(c["KR"], c["DR"], R2, P2, c["size"], cv2.CV_32FC1)
    rl, rr = cv2.remap(left, *m1, cv2.INTER_LINEAR), cv2.remap(right, *m2, cv2.INTER_LINEAR)

    sift = cv2.SIFT_create(4000)
    kl, dl = sift.detectAndCompute(rl, None)
    kr, dr = sift.detectAndCompute(rr, None)
    if dl is None or dr is None:
        return None
    matches = cv2.BFMatcher().knnMatch(dl, dr, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.7 * n.distance]
    if len(good) < 20:
        return None
    pl = np.float32([kl[m.queryIdx].pt for m in good])
    pr = np.float32([kr[m.trainIdx].pt for m in good])
    dy, dx = pr[:, 1] - pl[:, 1], pl[:, 0] - pr[:, 0]
    # keep the geometric consensus; stray matches on repeated laser stripes are expected
    keep = np.abs(dy - np.median(dy)) < 5
    return {"matches": len(good), "consensus": int(keep.sum()), "dy_median": float(np.median(dy[keep])),
            "dy_p95_abs": float(np.percentile(np.abs(dy[keep]), 95)), "dx_median": float(np.median(dx[keep])),
            "Q": Q, "rectified": (rl, rr)}


if __name__ == "__main__":
    frame, calib = sys.argv[1], load_calib(sys.argv[2])
    top, bottom = split(frame, calib["size"])
    for name, (L, R) in {"top=left, bottom=right": (top, bottom), "top=right, bottom=left": (bottom, top)}.items():
        r = residuals(L, R, calib)
        if r is None:
            print(f"{name}: too few matches")
            continue
        print(f"{name}: {r['matches']} matches, {r['consensus']} in consensus, vertical error median "
              f"{r['dy_median']:+.3f} px, 95% within {r['dy_p95_abs']:.3f} px, disparity median {r['dx_median']:+.1f} px")
        if r["dx_median"] > 0 and r["dy_p95_abs"] < 2:
            Z = r["Q"][2, 3] / (r["dx_median"] * r["Q"][3, 2] + r["Q"][3, 3])
            print(f"    -> consistent. Median disparity implies a scene distance of about {abs(Z):.0f} mm")
            rl, rr = r["rectified"]
            both = np.concatenate([rl, rr], axis=1)
            for y in range(0, both.shape[0], 80):
                both[y, :] = 255
            cv2.imwrite("/tmp/rectified_pair.png", cv2.resize(both, (1600, 600)))
