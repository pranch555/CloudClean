"""Reliable stereo correspondences from board frames: stripes lying on the board plane cannot be mis-indexed
(a one-stripe error lands ~40 mm off it). Saves xl, y, xr, 3D (rectified-left) and the left track id per point."""
import sys
import numpy as np
import stripes as S
from flatness import fit_plane

calib, out = sys.argv[-2], sys.argv[-1]
tri = S.Triangulator.from_yaml(calib)
rows = []
for fi, raw in enumerate(sys.argv[1:-2]):
    frame = S.load_raw(raw)
    rl, rr = tri.rectify(frame)
    cl = S.smooth_tracks(S.stripe_centres(rl, tri.p), tri.p)
    cr = S.smooth_tracks(S.stripe_centres(rr, tri.p), tri.p)
    tl = {(int(y), float(x)): int(t) for x, y, t in zip(cl.x, cl.y, cl.track)}
    pairs = []
    rows_r = cr.by_row()
    for y, pa in cl.by_row().items():
        pb = rows_r.get(y)
        if pb:
            pairs += [(y, a, b) for a, b in S.match_row(pa, pb, tri.f, tri.B, tri.off)]
    pairs = np.array(pairs)
    y, xl, xr = pairs.T
    h = np.c_[xl, y, xl - xr, np.ones_like(xl)] @ tri.Q.T
    P = h[:, :3] / h[:, 3:4]
    c, vt, r, inl = fit_plane(P, gate=0.5)
    on = np.abs(r) < 0.3
    slope = np.median(cl.slope)
    track = np.array([tl.get((int(a), float(b)), -1) for a, b in zip(y, xl)])
    print(f"{raw}: {on.sum()} of {len(P)} matched points on the board plane, family slope {slope:+.2f}")
    rows.append(np.c_[np.full(on.sum(), fi), np.full(on.sum(), slope), track[on], xl[on], y[on], xr[on], P[on]])
np.save(out, np.concatenate(rows))
