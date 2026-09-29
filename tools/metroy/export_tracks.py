"""Export matched stripe points of a frame with every coordinate a light-plane model could be written in."""
import sys
import cv2
import numpy as np
import stripes as S

calib_path, out = sys.argv[-2], sys.argv[-1]
tri = S.Triangulator.from_yaml(calib_path)
fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
KL, DL, KR, DR = (fs.getNode(k).mat() for k in ("cameraMatrixL", "distCoeffL", "cameraMatrixR", "distCoeffR"))
Rs, Ts = fs.getNode("steroRotation").mat(), fs.getNode("steroTranslation").mat()
R1, R2, *_ = cv2.stereoRectify(KL, DL, KR, DR, tri.size, Rs, Ts, flags=0)
rows = []
for fi, raw in enumerate(sys.argv[1:-2]):
    frame = S.load_raw(raw, tri.size)
    rl, rr = tri.rectify(frame)
    cl = S.smooth_tracks(S.stripe_centres(rl, tri.p), tri.p)
    cr = S.smooth_tracks(S.stripe_centres(rr, tri.p), tri.p)
    tl = {(int(y), round(x, 4)): int(t) for x, y, t in zip(cl.x, cl.y, cl.track)}
    tr = {(int(y), round(x, 4)): int(t) for x, y, t in zip(cr.x, cr.y, cr.track)}
    pts, xs, ys = tri.triangulate(cl, cr)
    keep = S.agreement(pts, xs, ys, tri.p.agree_mm, tri.p.agree_cell)
    # recover xr from disparity through Q: redo the pairs
    pairs = []
    for y, pa in cl.by_row().items():
        pb = cr.by_row().get(y)
        if pb:
            pairs += [(y, a, b) for a, b in S.match_row(pa, pb, tri.f, tri.B, tri.off)]
    pairs = np.array(pairs)[keep]
    pts = pts[keep]
    y, xl, xr = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    # rectified -> raw pixels, per camera
    def to_raw(x, yy, P, R, K, D):
        n = np.c_[(x - P[0, 2]) / P[0, 0], (yy - P[1, 2]) / P[1, 1], np.ones_like(x)]
        cam = n @ R          # R maps camera -> rectified, so R^T maps back: row-vector form
        uv, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        return uv.reshape(-1, 2)
    rawL = to_raw(xl, y, tri.P1, R1, KL, DL)
    rawR = to_raw(xr, y, tri.P2, R2, KR, DR)
    XL = pts @ R1              # rectified-left frame -> raw-left camera frame
    XR = (XL @ Rs.T) + Ts.ravel()
    track = np.array([tl.get((int(a), round(b, 4)), -1) for a, b in zip(y, xl)])
    trackR = np.array([tr.get((int(a), round(b, 4)), -1) for a, b in zip(y, xr)])
    rows.append(np.c_[np.full(len(y), fi), track, trackR, xl, y, xr, rawL, rawR, pts, XL, XR])
np.save(out, np.concatenate(rows))
print("saved", sum(len(r) for r in rows), "points")
