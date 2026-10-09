"""How precisely are the markers located? Measured on a recording (raw_<seq>_<t>.npy + frames.pkl, the driver's
"record" option).

On a STATIC scene (scanner and part still) every frame must give the same markers, so their spread over the frames
is the centre noise:
  2D       per marker, std of its centre in the left and right rectified image (px, per axis)
  3D       per marker, RMS distance of its triangulated position from its mean (mm)
  pairs    std of the distance between two markers (mm): does not depend on the pose, the cleanest figure
  pose     a point 220 mm in front of the scanner (where the surface is), mapped into the world by each frame's
           pose: RMS distance from its mean (mm). "tracker" = MarkerTracker as the scan runs it (map as you go),
           "fixed map" = each frame registered (Kabsch) to the mean marker positions
  family   the stripes move between the two laser line families (alternate frames); a centre the stripe drags
           lands in two places, so the shift between the families' mean centres is reported too
  dy       y_left - y_right of each pair: a rectified pair has dy = 0; a constant offset is a rectification error
           (docs/metroy-handoff.md, "What is missing" 2). Reported, never applied.
Any recording: markers paired per frame and the time detect() takes for the two views of a frame.

    python marker_precision.py RECORDING [--t0 S] [--t1 S] [--centre rim|centroid[,param=value...]] [--static]
                               [--save out.pkl | --load out.pkl] [--timing]
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

G: dict = {}
SURFACE = np.array([0.0, 0.0, 220.0])        # sensor frame, mm: a point on the optical axis where the part is


def _init(calib: bytes, laser: bytes | None, centre: str | None) -> None:
    import cv2
    cv2.setNumThreads(1)                     # as in the scan's worker processes
    from cloudclean.capture.metroy import markers, stripes
    G["tri"] = stripes.Triangulator.from_yaml(calib, None, laser)
    G["p"] = params(centre)
    G["m"] = markers


def params(spec: str | None):
    """MarkerParams from "centroid" or "rim,name=value,...": the default for None."""
    from cloudclean.capture.metroy import markers
    p = markers.MarkerParams()
    for item in (spec or "").split(","):
        if "=" in item:
            k, v = item.split("=")
            cur = getattr(p, k)
            setattr(p, k, (v not in ("0", "false", "False")) if isinstance(cur, bool) else type(cur)(v))
        elif item:
            p.centre = item
    return p


def _frame(path: str) -> dict:
    tri, p, m = G["tri"], G["p"], G["m"]
    rl, rr = tri.rectify(np.load(path))
    t0 = time.perf_counter()
    bl, br = m.detect(rl, p), m.detect(rr, p)
    dt = time.perf_counter() - t0
    P, il, ir = m.stereo_pairs(bl, br, tri.Q, p)
    row = lambda b: (b.x, b.y, b.radius, b.axis_ratio, getattr(b, "rim_rms", np.nan))   # noqa: E731
    name = Path(path).stem.split("_")
    return {"seq": int(name[1]), "t": float(name[2]), "P": P, "L": np.array([row(bl[i]) for i in il]).reshape(-1, 5),
            "R": np.array([row(br[j]) for j in ir]).reshape(-1, 5), "nl": len(bl), "nr": len(br), "dt": dt}


def run(rec: Path, centre: str | None, workers: int) -> list[dict]:
    meta = pickle.load(open(rec / "frames.pkl", "rb"))
    family = {f["sequence"]: f["meta"].get("family", -1) for f in meta["frames"]}
    paths = sorted(rec.glob("raw_*.npy"), key=lambda q: int(q.stem.split("_")[1]))
    with Pool(workers, _init, (meta["calibration"], meta.get("laser"), centre)) as pool:
        out = pool.map(_frame, [str(q) for q in paths], chunksize=4)
    for f in out:
        f["family"] = family.get(f["seq"], -1)
    return out


def timing(rec: Path, centre: str | None, n: int = 40) -> float:
    """detect() on both views of a frame, one thread, nothing else running in this process: median ms."""
    meta = pickle.load(open(rec / "frames.pkl", "rb"))
    _init(meta["calibration"], meta.get("laser"), centre)
    paths = sorted(rec.glob("raw_*.npy"))[:n]
    views = [G["tri"].rectify(np.load(q)) for q in paths]
    ts = []
    for rl, rr in views * 2:
        t0 = time.perf_counter()
        G["m"].detect(rl, G["p"]), G["m"].detect(rr, G["p"])
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts[len(views):]))


def tracks(frames: list[dict], gate_mm: float = 1.5, min_frac: float = 0.5):
    """Associate each frame's markers with the same physical marker (static scene: same place in the sensor frame).
    Returns idx (per frame, the track of each marker or -1) and the tracks' mean positions."""
    centres: list[np.ndarray] = []
    for f in frames:
        for q in f["P"]:
            if not centres or np.linalg.norm(np.array(centres) - q, axis=1).min() > gate_mm:
                centres.append(q)
    C = np.array(centres).reshape(-1, 3)
    for _ in range(2):                                  # refine the tracks, dropping those seen too rarely
        idx = _associate(frames, C, gate_mm)
        sums = np.zeros_like(C)
        cnt = np.zeros(len(C))
        for f, k in zip(frames, idx):
            np.add.at(sums, k[k >= 0], f["P"][k >= 0])
            np.add.at(cnt, k[k >= 0], 1)
        keep = cnt >= min_frac * len(frames)
        C = sums[keep] / cnt[keep, None]
    return _associate(frames, C, gate_mm), C           # indices into the final tracks


def _associate(frames: list[dict], C: np.ndarray, gate_mm: float) -> list[np.ndarray]:
    idx = []
    for f in frames:
        if not len(f["P"]) or not len(C):
            idx.append(np.full(len(f["P"]), -1))
            continue
        d = np.linalg.norm(f["P"][:, None] - C[None], axis=2)
        k = d.argmin(1)
        k = np.where(d[np.arange(len(k)), k] < gate_mm, k, -1)
        for j in np.unique(k[k >= 0]):                  # one marker per track and frame
            c = np.flatnonzero(k == j)
            if len(c) > 1:
                k[c] = -1
        idx.append(k)
    return idx


def static_report(frames: list[dict]) -> dict:
    from cloudclean.capture.metroy import markers as m
    idx, C = tracks(frames)
    n, T = len(frames), len(C)
    L = np.full((n, T, 2), np.nan)
    R = np.full((n, T, 2), np.nan)
    P = np.full((n, T, 3), np.nan)
    fam = np.array([f["family"] for f in frames])
    for a, (f, k) in enumerate(zip(frames, idx)):
        ok = k >= 0
        L[a, k[ok]], R[a, k[ok]], P[a, k[ok]] = f["L"][ok, :2], f["R"][ok, :2], f["P"][ok]
    rep = {"frames": n, "tracks": T, "seen": float(np.mean(np.isfinite(P[..., 0])))}
    rep["L_std_px"] = np.sqrt(np.nanmean(np.nanvar(L, 0), 1))              # per track, per-axis RMS
    rep["R_std_px"] = np.sqrt(np.nanmean(np.nanvar(R, 0), 1))
    rep["L_std_xy"] = np.sqrt(np.nanvar(L, 0))                               # (T, 2)
    rep["P_std_mm"] = np.sqrt(np.nansum(np.nanvar(P, 0), 1))                 # per track, 3D RMS
    D = np.linalg.norm(P[:, :, None] - P[:, None, :], axis=3)                # (n, T, T)
    iu = np.triu_indices(T, 1)
    rep["pair_std_mm"] = np.nanstd(D[:, iu[0], iu[1]], 0)
    rep["pair_len_mm"] = np.nanmean(D[:, iu[0], iu[1]], 0)
    # the two laser families
    if (fam == 0).sum() > 3 and (fam == 1).sum() > 3:
        rep["family_shift_L_px"] = np.linalg.norm(np.nanmean(L[fam == 0], 0) - np.nanmean(L[fam == 1], 0), axis=1)
        rep["family_shift_mm"] = np.linalg.norm(np.nanmean(P[fam == 0], 0) - np.nanmean(P[fam == 1], 0), axis=1)
        rep["within_family_L_px"] = np.sqrt(0.5 * (np.nanmean(np.nanvar(L[fam == 0], 0), 1)
                                                   + np.nanmean(np.nanvar(L[fam == 1], 0), 1)))
    # pose jitter of a surface point
    tr = m.MarkerTracker()
    w_tr, w_fix = [], []
    mean = np.nanmean(P, 0)
    for a, f in enumerate(frames):
        res = tr.track(f["P"])
        if res.pose is not None and a >= 10:
            w_tr.append(m.apply(res.pose, SURFACE[None])[0])
        k = idx[a]
        if (k >= 0).sum() >= 4:
            Tk = m.kabsch(f["P"][k >= 0], mean[k[k >= 0]])
            w_fix.append(m.apply(Tk, SURFACE[None])[0])
    spread = lambda W: float(np.sqrt(np.mean(np.sum((np.array(W) - np.mean(W, 0)) ** 2, 1)))) if len(W) > 2 \
        else float("nan")   # noqa: E731
    rep["pose_tracker_mm"], rep["pose_fixed_mm"] = spread(w_tr), spread(w_fix)
    rep["pose_tracker_frames"], rep["pose_fixed_frames"] = len(w_tr), len(w_fix)
    if len(w_fix) > 4:
        Wf = np.array(w_fix)
        fam_fix = np.array([f["family"] for a, f in enumerate(frames) if (idx[a] >= 0).sum() >= 4])
        if (fam_fix == 0).sum() > 2 and (fam_fix == 1).sum() > 2:
            rep["pose_family_shift_mm"] = float(np.linalg.norm(Wf[fam_fix == 0].mean(0) - Wf[fam_fix == 1].mean(0)))
            rep["pose_within_family_mm"] = float(np.sqrt(0.5 * sum(
                np.mean(np.sum((Wf[fam_fix == g] - Wf[fam_fix == g].mean(0)) ** 2, 1)) for g in (0, 1))))
    # stereo row offset
    dy = L[..., 1] - R[..., 1]                                               # (n, T)
    rep["dy_track_mean"] = np.nanmean(dy, 0)
    rep["dy_track_std"] = np.nanstd(dy, 0)
    rep["dy_track_n"] = np.sum(np.isfinite(dy), 0)
    rep["track_xy"] = np.nanmean(L, 0)
    rep["track_z"] = mean[:, 2]
    if (fam == 0).sum() > 3 and (fam == 1).sum() > 3:
        rep["dy_family"] = (float(np.nanmean(dy[fam == 0])), float(np.nanmean(dy[fam == 1])))
    return rep


def _fmt(rep: dict) -> str:
    med = lambda k: float(np.nanmedian(rep[k]))        # noqa: E731
    rms = lambda k: float(np.sqrt(np.nanmean(np.square(rep[k]))))   # noqa: E731
    s = [f"  {rep['frames']} frames, {rep['tracks']} markers tracked (present {100 * rep['seen']:.0f} % of the time)",
         f"  2D std  left  median {med('L_std_px'):.3f} px  RMS {rms('L_std_px'):.3f} | right median "
         f"{med('R_std_px'):.3f} px  RMS {rms('R_std_px'):.3f}   (x {np.nanmedian(rep['L_std_xy'][:, 0]):.3f}, "
         f"y {np.nanmedian(rep['L_std_xy'][:, 1]):.3f} left)",
         f"  3D std  median {med('P_std_mm'):.4f} mm  RMS {rms('P_std_mm'):.4f}",
         f"  inter-marker distance std  median {med('pair_std_mm'):.4f} mm  RMS {rms('pair_std_mm'):.4f}  "
         f"({len(rep['pair_std_mm'])} pairs)",
         f"  pose jitter at 220 mm  tracker {rep['pose_tracker_mm']:.4f} mm ({rep['pose_tracker_frames']} frames), "
         f"fixed map {rep['pose_fixed_mm']:.4f} mm"]
    if "family_shift_L_px" in rep:
        s.append(f"  families: centre shift left median {med('family_shift_L_px'):.3f} px, 3D median "
                 f"{med('family_shift_mm'):.4f} mm; within one family 2D std median {med('within_family_L_px'):.3f} px;"
                 f" pose shift {rep.get('pose_family_shift_mm', np.nan):.4f} mm, within one family "
                 f"{rep.get('pose_within_family_mm', np.nan):.4f} mm")
    good = rep["dy_track_n"] >= 10
    dm = rep["dy_track_mean"][good]
    se = rep["dy_track_std"][good] / np.sqrt(rep["dy_track_n"][good])
    xy = rep["track_xy"][good]
    A = np.c_[np.ones(good.sum()), (xy[:, 0] - 800) / 1000, (xy[:, 1] - 600) / 1000]
    coef, *_ = np.linalg.lstsq(A, dm, rcond=None)
    res = dm - A @ coef
    s.append(f"  dy = yL - yR over {good.sum()} markers: mean {dm.mean():+.3f} px, median {np.median(dm):+.3f}, "
             f"spread of the per-marker means {dm.std():.3f} px (each known to {np.median(se):.3f} px); "
             f"per-frame std median {np.median(rep['dy_track_std'][good]):.3f} px")
    s.append(f"  dy ~ {coef[0]:+.3f} {coef[1]:+.3f}*(x-800)/1000 {coef[2]:+.3f}*(y-600)/1000 px, residual "
             f"{res.std():.3f} px" + (f"; family 0 {rep['dy_family'][0]:+.3f}, family 1 {rep['dy_family'][1]:+.3f}"
                                       if "dy_family" in rep else ""))
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--t0", type=float, default=0.0, help="s from the first frame")
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--centre", default=None)
    ap.add_argument("--static", action="store_true", help="the scene and the scanner were still: precision report")
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--save")
    ap.add_argument("--load")
    ap.add_argument("--timing", action="store_true", help="also time detect() alone in this process, one thread")
    a = ap.parse_args()
    rec = Path(a.recording).expanduser()
    if a.timing:
        print(f"detect() alone: {timing(rec, a.centre):.2f} ms per frame (both views, median of 40 frames x 2)")
    frames = pickle.load(open(a.load, "rb")) if a.load else run(rec, a.centre, a.workers)
    if a.save:
        pickle.dump(frames, open(a.save, "wb"))
    t_first = min(f["t"] for f in frames)
    sel = [f for f in frames if a.t0 <= f["t"] - t_first <= a.t1]
    npair = np.array([len(f["P"]) for f in sel])
    print(f"{rec.name} [{a.t0:g}, {min(a.t1, 1e6):g}] s, centre {a.centre or 'default'}: {len(sel)} frames; markers "
          f"paired per frame mean {npair.mean():.2f} (min {npair.min()}, p10 {np.percentile(npair, 10):.0f}), detected "
          f"left {np.mean([f['nl'] for f in sel]):.1f} right {np.mean([f['nr'] for f in sel]):.1f}; detect() "
          f"{1000 * np.median([f['dt'] for f in sel]):.1f} ms per frame (median, {a.workers} workers busy)")
    if a.static:
        print(_fmt(static_report(sel)))


if __name__ == "__main__":
    main()
