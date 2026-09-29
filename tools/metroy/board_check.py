"""Measure the Revopoint calibration board with the factory calibration: an accuracy check of the DGX pipeline.

The board's dots are small dark blobs, so a black-hat filter isolates them from the bright laser stripes. Their
centres are rectified, paired row-by-row, triangulated, and then judged against two physical facts: the board is
flat (plane-fit residual), and the dots sit on a regular grid (nearest-neighbour spacing).

    python board_check.py frame.raw camparam.yaml [top-is-left: 1|0]
"""
import sys

import cv2
import numpy as np

from rectify_check import load_calib, split


def dots(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    keep = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 6 <= area <= 220 and 0.5 <= w / max(h, 1) <= 2.0:
            keep.append(cents[i])
    return np.float32(keep).reshape(-1, 2)


def main(frame: str, calib_path: str, top_is_left: bool = True):
    c = load_calib(calib_path)
    top, bottom = split(frame, c["size"])
    L, R = (top, bottom) if top_is_left else (bottom, top)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(c["KL"], c["DL"], c["KR"], c["DR"], c["size"], c["R"], c["T"],
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    dl, dr = dots(L), dots(R)
    print(f"dots found: left {len(dl)}, right {len(dr)}")
    ul = cv2.undistortPoints(dl.reshape(-1, 1, 2), c["KL"], c["DL"], R=R1, P=P1).reshape(-1, 2)
    ur = cv2.undistortPoints(dr.reshape(-1, 1, 2), c["KR"], c["DR"], R=R2, P=P2).reshape(-1, 2)

    # disparity window for 120-800 mm, from the rectified geometry
    f, B = P1[0, 0], -P2[0, 3] / P2[0, 0]
    dmin, dmax = f * B / 800.0, f * B / 120.0
    pairs = []
    for i, (x, y) in enumerate(ul):
        cand = np.where((np.abs(ur[:, 1] - y) < 1.5) & (x - ur[:, 0] > dmin) & (x - ur[:, 0] < dmax))[0]
        if len(cand) == 1:                     # unambiguous only: repeated rows are skipped, not guessed
            pairs.append((i, cand[0]))
    if len(pairs) < 12:
        print(f"only {len(pairs)} unambiguous pairs - not enough to judge")
        return
    pl = np.float32([ul[i] for i, _ in pairs])
    pr = np.float32([ur[j] for _, j in pairs])
    X = cv2.triangulatePoints(P1, P2, pl.T, pr.T)
    X = (X[:3] / X[3]).T
    dy = pr[:, 1] - pl[:, 1]

    # robust plane (the board) - a few bad pairs on repeating rows are rejected, and reported
    best = None
    rng = np.random.default_rng(0)
    for _ in range(500):
        s = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = np.abs((X - s[0]) @ n)
        inl = d < 0.5
        if best is None or inl.sum() > best[0].sum():
            best = (inl, n, s[0])
    inl = best[0]
    P = X[inl]
    centroid = P.mean(0)
    _, _, vt = np.linalg.svd(P - centroid)
    normal = vt[2]
    res = (P - centroid) @ normal

    # grid spacing: nearest-neighbour distances between inlier dots
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    pitch = np.median(nn)
    close = nn[np.abs(nn - pitch) < 0.2 * pitch]

    print(f"unambiguous pairs: {len(pairs)}   vertical row error: median {np.median(dy):+.3f} px, "
          f"95% within {np.percentile(np.abs(dy), 95):.3f} px")
    print(f"distance to board: {np.linalg.norm(centroid):.1f} mm (centroid z {centroid[2]:.1f} mm)")
    print(f"board flatness: {inl.sum()}/{len(X)} dots on the plane, RMS {np.sqrt((res ** 2).mean()) * 1000:.1f} um, "
          f"max {np.abs(res).max() * 1000:.1f} um")
    print(f"dot pitch: median {pitch:.3f} mm, spread (std of on-grid neighbours) {close.std() * 1000:.1f} um "
          f"over {len(close)} dots")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], len(sys.argv) < 4 or sys.argv[3] == "1")
