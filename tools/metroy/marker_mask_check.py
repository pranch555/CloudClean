"""Laser points kept out by the marker mask: every detected blob (before) vs only the blobs that may be markers
(markers.masks(), after). A bright laser spot on a light part is laser line: masking it cut a hole into the line.

Per frame of a recording: laser points before/after, how many blobs each mask covers, how many extra points lie
within a masked-before spot's circle, and how far they lie from the nearest point kept before (the hole they fill,
mm). With --static, whether they lie on the surface the other points make.

    python marker_mask_check.py RECORDING [--every N] [--t0 S] [--t1 S] [--static]
"""
import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

G: dict = {}


def _init(calib, laser):
    import cv2
    cv2.setNumThreads(1)
    from cloudclean.capture.metroy import markers, stripes
    G["tri"], G["m"] = stripes.Triangulator.from_yaml(calib, None, laser), markers


def _frame(path):
    from scipy.spatial import cKDTree
    tri, m = G["tri"], G["m"]
    rl, rr = tri.rectify(np.load(path))
    bl, br = m.detect(rl), m.detect(rr)
    before = (m.mask(rl.shape, bl), m.mask(rr.shape, br))
    after = m.masks(rl.shape, bl, br, tri.Q)
    pb, xb, yb = tri.points_rectified(rl, rr, with_pixels=True, masks=before)
    pa, xa, ya = tri.points_rectified(rl, rr, with_pixels=True, masks=after)
    row = np.clip(np.round(ya).astype(int), 0, rl.shape[0] - 1)
    extra = before[0][row, np.clip(np.round(xa).astype(int), 0, rl.shape[1] - 1)]       # where the old mask was
    d = cKDTree(pb).query(pa[extra], k=1)[0] if len(pb) and extra.any() else np.zeros(0)
    keep_l, keep_r = m.plausible(bl, br, tri.Q)
    return {"before": len(pb), "after": len(pa), "extra": int(extra.sum()), "extra_d": d, "pa": pa, "is_extra": extra,
            "family": tri.last.get("family", -1),
            "blobs": (len(bl), len(br)), "masked_after": (int(keep_l.sum()), int(keep_r.sum())),
            "paired": len(m.stereo_pairs(bl, br, tri.Q)[0]), "t": float(Path(path).stem.split("_")[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--static", action="store_true", help="scene and scanner still: check the recovered points")
    a = ap.parse_args()
    rec = Path(a.recording).expanduser()
    meta = pickle.load(open(rec / "frames.pkl", "rb"))
    paths = sorted(rec.glob("raw_*.npy"), key=lambda q: int(q.stem.split("_")[1]))
    t_first = float(paths[0].stem.split("_")[2])
    paths = [q for q in paths if a.t0 <= float(q.stem.split("_")[2]) - t_first <= a.t1][::a.every]
    with Pool(a.workers, _init, (meta["calibration"], meta.get("laser"))) as pool:
        out = pool.map(_frame, [str(q) for q in paths], chunksize=2)
    b = np.array([f["before"] for f in out])
    af = np.array([f["after"] for f in out])
    ex = np.array([f["extra"] for f in out])
    d = np.concatenate([f["extra_d"] for f in out])
    nb = np.array([sum(f["blobs"]) for f in out])
    nm = np.array([sum(f["masked_after"]) for f in out])
    print(f"{rec.name}: {len(out)} frames; laser points per frame {b.mean():.0f} -> {af.mean():.0f} "
          f"({100 * (af.mean() / b.mean() - 1):+.2f} %, median per frame {np.median(af - b):+.0f}); inside spots "
          f"masked before: {ex.mean():.1f} per frame; blobs masked {nb.mean():.1f} -> {nm.mean():.1f} per frame "
          f"(both views); markers paired {np.mean([f['paired'] for f in out]):.2f}")
    if len(d):
        print(f"  recovered points lie from the nearest point kept before: median {np.median(d):.3f} mm, p95 "
              f"{np.percentile(d, 95):.3f}, max {d.max():.3f} ({len(d)} points)")
    if a.static:
        _surface_check(out)


def _surface_check(out):
    """Static scene: are the recovered points on the surface? The surface around a point is every frame's points of
    both line families (a crossing pattern) except the recovered ones: a quadric through those within 4 mm, where
    they span two directions (one stripe alone leaves the surface's tilt free). Ordinary points of the same frames get
    the same test, with the same 2 mm hole cut around them, as the yardstick."""
    from scipy.spatial import cKDTree
    ref = np.concatenate([f["pa"][~f["is_extra"]] for f in out])
    tree = cKDTree(ref)
    rng = np.random.default_rng(0)

    def dist(points, skip_self):
        res = []
        for q in points:
            idx = tree.query_ball_point(q, 4.0)
            N = ref[idx]
            if skip_self:
                N = N[np.linalg.norm(N - q, axis=1) > 2.0]   # the same hole a recovered point sits in (median 2 mm)
            if len(N) < 30:
                continue
            c = N.mean(0)
            _, s, vt = np.linalg.svd(N - c, full_matrices=False)
            if s[1] < 0.35 * s[0]:
                continue
            u, v, w = ((N - c) @ vt[0], (N - c) @ vt[1], (N - c) @ vt[2])
            A = np.c_[u * u, u * v, v * v, u, v, np.ones_like(u)]
            k = np.linalg.lstsq(A, w, rcond=None)[0]
            qu, qv, qw = (q - c) @ vt[0], (q - c) @ vt[1], (q - c) @ vt[2]
            res.append(abs(qw - np.array([qu * qu, qu * qv, qv * qv, qu, qv, 1.0]) @ k))
        return np.array(res)
    rec, ordinary = [], []
    for f in out:
        rec.append(dist(f["pa"][f["is_extra"]], False))
        pick = rng.choice(len(f["pa"]), min(30, len(f["pa"])), replace=False) if len(f["pa"]) else []
        ordinary.append(dist(f["pa"][pick], True))
    r, o = np.concatenate(rec), np.concatenate(ordinary)
    if len(r) and len(o):
        print(f"  off the surface the other points make: recovered points median {np.median(r):.3f} mm, p95 "
              f"{np.percentile(r, 95):.3f} ({len(r)} tested); ordinary points median {np.median(o):.3f}, p95 "
              f"{np.percentile(o, 95):.3f} ({len(o)} tested)")


if __name__ == "__main__":
    main()
