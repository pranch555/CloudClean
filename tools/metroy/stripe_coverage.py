"""Where do the laser stripe centres ON THE PART get lost? Stage by stage over a recording, checked against a scan.

    python tools/metroy/stripe_coverage.py RECORDING_DIR REFERENCE.ply [--every N] [--workers N] [--crops DIR]
           [--holdout cache.npz] [--params key=value ...] [--save run.npy] [--compare baseline.npy] [--dump pts.npz]

On the DGX (the recording of 2026-10-09 and the scan made from it):

    PYTHONPATH=. LD_LIBRARY_PATH=.venv/lib/extra .venv/bin/python tools/metroy/stripe_coverage.py \\
        ~/.cache/cloudclean/metroy/recordings/20261009-114240 ~/cloudclean-workspace/assets/c0defb931798/data.ply \\
        --holdout /tmp/holdout.npz --workers 10 --save /tmp/run.npy

RECORDING_DIR is a "Record for diagnosis" folder (raw_<seq>_<t>.npy every 10th frame + frames.pkl with every frame's
points and pose); REFERENCE.ply a fused scan of the same session (same world frame as the poses). Per frame:

* the part's region in the left rectified image: the reference projected through the frame's pose and closed over its
  holes (so stripes where the scan has none still count), plus the pixels of the frame's own on-part points
  (drivers.metroy_usb._part_on_table, as the driver does live);
* every left stripe centre in that region gets the stage that dropped it, from the real code path
  (Triangulator.trace; the traced run must reproduce points_rectified exactly);
* ground truth per centre: which right-view centre, if any, triangulates onto the reference (< GT_MM) without being
  hidden behind it. A dropped centre with such a partner was measurable; one without is a stripe the right camera does
  not see there, or one in a hole of the reference;
* accuracy of the points, in the world frame through the pose, against two references:
  - REFERENCE.ply, leaving out its copies of this frame's own points;
  - the hold-out: every OTHER frame of the recording seen from a clearly different viewpoint (table angle more than
    HOLDOUT_DEG away). Neighbouring frames repeat a frame's stripes almost exactly - and with them any systematic
    error - so only the hold-out says whether a point agrees with an independent measurement. "plane" is the distance
    to a plane through the 16 nearest hold-out points (all within 1 mm; else the point has no surface near it).
* with --compare (a --save of the baseline run): the points this run adds ("recovered") and drops, by the same tests.

Writes a JSON summary per frame (--out) and, with --crops, pictures of the part with the centres coloured by fate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cloudclean.capture.metroy import markers as M          # noqa: E402
from cloudclean.capture.metroy import stripes as S          # noqa: E402

GT_MM = 0.4          # a pairing whose 3D point lies this close to the reference is the true one
HIDDEN_MM = 1.0      # ... unless it lies this far behind the reference's front surface along the ray
HOLDOUT_DEG = 15.0   # hold-out reference: frames whose table angle differs by more than this
FATE_COLOURS = {"accepted": (0, 200, 0), "marker mask": (0, 255, 255), "short track": (0, 0, 255),
                "smooth reject": (0, 128, 255), "trimmed ends": (255, 128, 0), "short after trim": (255, 0, 0),
                "low support": (0, 0, 160), "margin": (160, 0, 160), "unconfirmed point": (128, 128, 255),
                "reverse check": (128, 255, 128), "right claimed": (255, 255, 0), "depth window": (64, 64, 64),
                "agreement": (255, 255, 255)}
STAGES = list(FATE_COLOURS)


def read_ply(path: str) -> np.ndarray:
    raw = open(path, "rb").read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("ascii", "replace")
    n = int([ln for ln in head.splitlines() if ln.startswith("element vertex")][0].split()[-1])
    props = [ln.split() for ln in head.splitlines() if ln.startswith("property")]
    types = {"double": "<f8", "float": "<f4", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    dt = np.dtype([(p[2], types[p[1]]) for p in props])
    v = np.frombuffer(raw[end:end + n * dt.itemsize], dt)
    return np.c_[v["x"], v["y"], v["z"]].astype(np.float64)


def yaw_of(pose: np.ndarray) -> float:
    """Table angle: the sensor's forward axis laid in the plate plane (world XY), degrees."""
    return float(np.degrees(np.arctan2(pose[1, 2], pose[0, 2])))


# -- per worker state --------------------------------------------------------------------------------------------
W: dict = {}


def init(calib: bytes, laser: bytes, ref_path: str, params: dict, holdout_path: str | None):
    cv2.setNumThreads(1)
    from scipy.spatial import cKDTree
    W["tri"] = S.Triangulator.from_yaml(calib, S.Params(**params), laser)
    W["ref"] = read_ply(ref_path)
    W["tree"] = cKDTree(W["ref"])
    W["hold"] = []
    if holdout_path:
        h = np.load(holdout_path)
        P, yaw = h["P"].astype(np.float64), h["yaw"]
        for lo in np.arange(-180.0, 180.0, HOLDOUT_DEG):
            sel = (yaw >= lo) & (yaw < lo + HOLDOUT_DEG)
            if sel.any():
                W["hold"].append((lo + HOLDOUT_DEG / 2, cKDTree(P[sel]), P[sel]))


def project(tri, P: np.ndarray):
    f, cx, cy = tri.P1[0, 0], tri.P1[0, 2], tri.P1[1, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return f * P[:, 0] / P[:, 2] + cx, f * P[:, 1] / P[:, 2] + cy


def part_region(tri, pose: np.ndarray, onpart: np.ndarray, shape):
    """(region mask, front depth of the reference per pixel) in the left rectified image."""
    h, w = shape
    R, t = pose[:3, :3], pose[:3, 3]
    ref = (W["ref"] - t) @ R
    ref = ref[(ref[:, 2] > 100) & (ref[:, 2] < 600)]
    u, v = project(tri, ref)
    ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    zbuf = np.full(h * w, np.inf, np.float32)
    np.minimum.at(zbuf, vi[ok] * w + ui[ok], ref[ok, 2].astype(np.float32))
    zbuf = zbuf.reshape(h, w)
    front = cv2.erode(zbuf, np.ones((5, 5), np.uint8))       # min over 5x5: the surface the camera sees
    seen = np.isfinite(front).astype(np.uint8)
    region = cv2.morphologyEx(seen, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    if len(onpart):
        u, v = project(tri, onpart)
        ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
        ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        m = np.zeros((h, w), np.uint8)
        m[vi[ok], ui[ok]] = 1
        region |= cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return region.astype(bool), front


def to_world(pose, P):
    return np.asarray(P, np.float64) @ pose[:3, :3].T + pose[:3, 3]


def ground_truth(tri, pose, front, xl, yl, cr: S.Centres):
    """Per left centre: the right-view x of its true partner (NaN if none sees it) and that pairing's distance."""
    n = len(xl)
    gt_xr, gt_d = np.full(n, np.nan), np.full(n, np.inf)
    if not n or not len(cr.x):
        return gt_xr, gt_d
    rows = cr.by_row()
    width = max(len(v) for v in rows.values())
    R = np.full((tri.size[1], width), np.nan)
    for y, xs in rows.items():
        R[y, :len(xs)] = xs
    cand = R[yl.astype(int)]                                    # (n, width)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = tri.f * tri.B / (xl[:, None] - cand - tri.off)
    ok = np.isfinite(cand) & (z > 150) & (z < 500)
    i, j = np.nonzero(ok)
    if not len(i):
        return gt_xr, gt_d
    xr = cand[i, j]
    hq = np.c_[xl[i], yl[i], xl[i] - xr, np.ones(len(i))] @ tri.Q.T
    P = hq[:, :3] / hq[:, 3:4]
    d, _ = W["tree"].query(to_world(pose, P), distance_upper_bound=5.0)
    u, v = np.rint(xl[i]).astype(int).clip(0, front.shape[1] - 1), yl[i].astype(int)
    fz = front[v, u]
    d = np.where(np.isfinite(fz) & (P[:, 2] > fz + HIDDEN_MM), np.inf, d)     # hidden behind the reference
    best = np.full(n, np.inf)
    np.minimum.at(best, i, d)
    win = (d == best[i]) & (best[i] < GT_MM)
    gt_xr[i[win]] = xr[win]
    return gt_xr, best


def reference_distance(pose, P, own=None):
    """Distance of sensor-frame points P to REFERENCE.ply, leaving out its copies of this frame's own recorded points
    (`own`, sensor frame): the fused scan contains them, and a point is not checked against itself."""
    if not len(P):
        return np.zeros(0)
    tree, n = W["tree"], len(W["ref"])
    d, j = tree.query(to_world(pose, P), k=24, distance_upper_bound=50.0)
    if own is not None and len(own):
        do, jo = tree.query(to_world(pose, own), k=1)
        mine = np.zeros(n + 1, bool)
        mine[jo[do < 1e-3]] = True
        d = np.where(mine[np.minimum(j, n)], np.inf, d)
    return d.min(1)


def holdout_distance(pose, P, k: int = 16, radius: float = 1.0):
    """(nearest, plane) distance of sensor-frame points P to the frames seen from another table angle. plane is NaN
    where fewer than k hold-out points lie within `radius` (no independent surface there)."""
    if not len(P) or not W["hold"]:
        return np.full(len(P), np.nan), np.full(len(P), np.nan)
    Wp = to_world(pose, P)
    yaw = yaw_of(pose)
    ds, nb = [], []
    for centre, tree, pts in W["hold"]:
        if abs((centre - yaw + 180) % 360 - 180) < HOLDOUT_DEG + HOLDOUT_DEG / 2:
            continue
        d, j = tree.query(Wp, k=k, distance_upper_bound=radius)
        ds.append(d)
        nb.append(np.where(np.isfinite(d)[..., None], pts[np.minimum(j, len(pts) - 1)], np.nan))
    d = np.concatenate(ds, 1)
    nbp = np.concatenate(nb, 1)
    order = np.argsort(d, 1)[:, :k]
    dk = np.take_along_axis(d, order, 1)
    Q = np.take_along_axis(nbp, order[..., None], 1)
    nearest = dk[:, 0]
    full = np.isfinite(dk[:, -1])
    plane = np.full(len(P), np.nan)
    if full.any():
        Qf = Q[full]
        c = Qf.mean(1)
        cov = np.einsum("nki,nkj->nij", Qf - c[:, None], Qf - c[:, None])
        _, vec = np.linalg.eigh(cov)
        plane[full] = np.abs(np.einsum("ni,ni->n", Wp[full] - c, vec[:, :, 0]))
    if not np.isfinite(nearest).all():                    # beyond the radius: the true nearest, for the record
        far = ~np.isfinite(nearest)
        nearest[far] = np.min([tree.query(Wp[far], k=1)[0] for centre, tree, _ in W["hold"]
                               if abs((centre - yaw + 180) % 360 - 180) >= 1.5 * HOLDOUT_DEG], axis=0)
    return nearest, plane


def fates(p, tr) -> np.ndarray:
    """The stage that dropped each raw left centre (or "accepted"), from Triangulator.trace's record."""
    raw = tr["left_raw"]
    fate = np.full(len(raw.x), "marker mask", object)
    um = tr.get("left_unmasked", raw)
    base = um.src if um is not raw else np.arange(len(raw.x))
    f2 = np.full(len(um.x), "short track", object)
    for t in S.link_tracks(um, p):
        k = np.arange(len(t))
        trimmed = (k < p.trim_ends) | (k >= len(t) - p.trim_ends)
        f2[t[trimmed]] = "trimmed ends"
        f2[t[~trimmed]] = "smooth reject"
    sm = tr["left"]
    f2[sm.src] = np.array(S.FATES, object)[tr["why"]]
    fate[base] = f2
    return fate


def final_raw(tr) -> np.ndarray:
    """Raw left centre index of each final point, in the order points_rectified returns them."""
    i = tr["left"].src[tr["final"]]
    return tr["left_unmasked"].src[i] if "left_unmasked" in tr else i


def trace_frame(task):
    path, pose, onpart, world_markers, crops_dir, base = task
    from cloudclean.capture.drivers.metroy_usb import _part_on_table
    tri = W["tri"]
    frame = np.load(path)
    rl, rr = tri.rectify(frame)
    t0 = time.perf_counter()                           # what a capture worker does per frame (scanner._triangulate)
    bl, br = M.detect(rl), M.detect(rr)
    ml, mr = M.mask(rl.shape, bl), M.mask(rr.shape, br)
    pts, xs, ys = tri.points_rectified(rl, rr, with_pixels=True, masks=(ml, mr))
    t_prod = time.perf_counter() - t0
    info = dict(tri.last)

    tri.trace = True                                   # the same pipeline again, keeping every stage
    try:
        _, xs2, ys2 = tri.points_rectified(rl, rr, with_pixels=True, masks=(ml, mr))
        tr = dict(tri.traced)
    finally:
        tri.trace = False
    if not (np.array_equal(xs2, xs) and np.array_equal(ys2, ys)):
        raise AssertionError(f"{path}: tracing changed the result")
    cl0, fate = tr["left_raw"], fates(tri.p, tr)
    fin = final_raw(tr)
    if (fate == "accepted").sum() != len(xs) or not np.array_equal(cl0.y[fin], ys):
        raise AssertionError(f"{path}: the fates do not account for the {len(xs)} points")
    region, front = part_region(tri, pose, onpart, rl.shape)
    inr = region[cl0.y.astype(int), np.clip(np.rint(cl0.x).astype(int), 0, rl.shape[1] - 1)]
    gt_xr, _ = ground_truth(tri, pose, front, cl0.x, cl0.y, tr["right_raw"])
    has_gt = np.isfinite(gt_xr)
    sat = rl[cl0.y.astype(int), np.clip(np.rint(cl0.x).astype(int), 0, rl.shape[1] - 1)] >= 250

    # the points the driver keeps: on the table, inside the markers' area (and in the image region)
    keep = _part_on_table(pts, pose, world_markers) if len(pts) else np.zeros(0, bool)
    on = (np.zeros(len(pts), bool) if keep is None else keep) & inr[fin]
    acc = {}

    def measure(name, sel):
        P = pts[sel]
        hn, hp = holdout_distance(pose, P)
        acc[name] = {"ref": reference_distance(pose, P, onpart).tolist(), "hold": hn.tolist(),
                     "plane": np.nan_to_num(hp, nan=-1.0).tolist()}

    measure("accepted", on)
    lost = 0
    if base is not None:
        new = on & ~np.isin(fin, base["acc"])
        measure("recovered", new)
        was = np.asarray(STAGES)[base["fate"][fin]]       # the stage that dropped each point in the baseline
        for stage in STAGES[1:]:
            if (new & (was == stage)).any():
                measure(f"recovered: {stage}", new & (was == stage))
        lost = int(np.sum(inr[base["acc"]] & ~np.isin(base["acc"], fin)))
    out = {"path": os.path.basename(path), "ms": 1000 * t_prod, "yaw": yaw_of(pose),
           "info": {k: v for k, v in info.items() if not isinstance(v, np.ndarray)},
           "fates": Counter(fate[inr].tolist()), "fates_gt": Counter(fate[inr & has_gt].tolist()),
           "fates_sat": Counter(fate[inr & sat].tolist()), "in_region": int(inr.sum()),
           "acc": acc, "lost": lost, "on_part": int(on.sum()),
           "save": {"acc": fin[on], "fate": np.array([STAGES.index(f) for f in fate], np.int8)},
           "world": to_world(pose, pts[on]).astype(np.float32)}
    if crops_dir:
        draw(crops_dir, path, rl, cl0, fate, inr, has_gt)
    return out


def draw(crops_dir, path, rl, cl0, fate, inr, has_gt):
    ys, xs = cl0.y[inr].astype(int), cl0.x[inr]
    if not len(xs):
        return
    s = 3
    x0, x1 = max(int(np.percentile(xs, 1)) - 30, 0), min(int(np.percentile(xs, 99)) + 30, rl.shape[1])
    y0, y1 = max(int(np.percentile(ys, 1)) - 30, 0), min(int(np.percentile(ys, 99)) + 30, rl.shape[0])
    img = cv2.cvtColor(np.clip(rl.astype(np.float32) * 1.5, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    big = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    # each centre: a short dash to its right in its fate's colour; a gap between stripe and dash where the right
    # view has no true partner for it
    for f, col in FATE_COLOURS.items():
        sel = inr & (fate == f)
        for x, y, g in zip(cl0.x[sel], cl0.y[sel], has_gt[sel]):
            px, py = int(round((x - x0) * s)), int(round((y - y0) * s)) + 1
            if 0 <= py < big.shape[0]:
                big[py:py + 2, min(px + (3 if g else 6), big.shape[1]):min(px + 12, big.shape[1])] = col
    cv2.imwrite(str(Path(crops_dir) / (Path(path).stem + "_left.png")), big)


def build_holdout(frames, out_path: str):
    """Every frame's on-part points in the world frame, with the frame's table angle."""
    from cloudclean.capture.drivers.metroy_usb import _part_on_table
    P, yaw = [], []
    for f in frames:
        if f["pose"] is None or not len(f["points"]):
            continue
        pts = np.asarray(f["points"], np.float64)
        keep = _part_on_table(pts, f["pose"], np.asarray(f["meta"]["map_points"]))
        if keep is None or not keep.any():
            continue
        P.append(to_world(f["pose"], pts[keep]).astype(np.float32))
        yaw.append(np.full(int(keep.sum()), yaw_of(f["pose"]), np.float32))
    np.savez(out_path, P=np.concatenate(P), yaw=np.concatenate(yaw))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("reference")
    ap.add_argument("--every", type=int, default=1, help="use every Nth raw file")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crops", default="", help="folder for pictures of the part with centres coloured by fate")
    ap.add_argument("--params", nargs="*", default=[], help="Params overrides, key=value")
    ap.add_argument("--save", default="", help="save each frame's accepted centres (for a later --compare)")
    ap.add_argument("--compare", default="", help="a --save of the baseline run")
    ap.add_argument("--holdout", default="", help="hold-out cache (.npz; built from frames.pkl if missing)")
    ap.add_argument("--dump", default="", help="write the on-part points (world frame) of all frames to this .npz")
    ap.add_argument("--out", default="stripe_coverage.json")
    a = ap.parse_args()

    from cloudclean.capture.drivers.metroy_usb import _part_on_table
    rec = pickle.load(open(Path(a.recording) / "frames.pkl", "rb"))
    if a.holdout and not os.path.exists(a.holdout):
        build_holdout(rec["frames"], a.holdout)
    types = {f.name: f.type for f in fields(S.Params)}
    params = {}
    for kv in a.params:
        k, v = kv.split("=", 1)
        params[k] = (v.lower() in ("1", "true", "yes")) if "bool" in str(types[k]) else type(getattr(S.Params(), k))(v)
    base = dict(np.load(a.compare, allow_pickle=True).item()) if a.compare else {}
    by_seq = {f["sequence"]: f for f in rec["frames"]}
    raws = sorted(glob.glob(str(Path(a.recording) / "raw_*.npy")))[::a.every]
    tasks = []
    for r in raws:
        f = by_seq.get(int(Path(r).name.split("_")[1]))
        if f is None or f["pose"] is None:
            continue
        pts = np.asarray(f["points"])
        world = np.asarray(f["meta"]["map_points"])
        keep = _part_on_table(pts, f["pose"], world) if len(pts) else None
        b = base.get(Path(r).name) if a.compare else None
        tasks.append((r, f["pose"], pts[keep] if keep is not None else pts[:0], world, a.crops or None, b))
    if a.crops:
        os.makedirs(a.crops, exist_ok=True)
    print(f"{len(tasks)} frames, params {params or 'default'}", flush=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers, init, (rec["calibration"], rec["laser"], a.reference, params,
                                                        a.holdout or None)) as pool:
        res = pool.map(trace_frame, tasks, chunksize=2)
    saved = {r["path"]: r.pop("save") for r in res}
    if a.save:
        np.save(a.save, saved, allow_pickle=True)
    world = [r.pop("world") for r in res]
    if a.dump:                                         # the run's on-part points in the world frame, for a fusion
        np.savez(a.dump, P=np.concatenate(world), frame=np.repeat(np.arange(len(world)), [len(w) for w in world]))
    summarise(res)
    json.dump(res, open(a.out, "w"))


def dist_stats(d: np.ndarray) -> str:
    d = d[np.isfinite(d)]
    if not len(d):
        return "n 0"
    return (f"n {len(d):7d}  median {np.median(d):.3f}  p95 {np.percentile(d, 95):.3f}"
            f"  > 0.3 mm {np.mean(d > 0.3):6.2%}  > 1 mm {np.mean(d > 1):6.2%}  > 3 mm {np.mean(d > 3):6.3%}")


def summarise(res):
    tot, gt, sat = Counter(), Counter(), Counter()
    for r in res:
        tot.update(r["fates"]); gt.update(r["fates_gt"]); sat.update(r["fates_sat"])
    n = sum(tot.values())
    print(f"{len(res)} frames, {n} left centres on the part ({n / max(len(res), 1):.0f} per frame), "
          f"{sum(gt.values()) / max(n, 1):.0%} with a true partner in the right view; CPU ms/frame median "
          f"{np.median([r['ms'] for r in res]):.0f}, mean {np.mean([r['ms'] for r in res]):.0f}")
    print(f"{'stage':20s} {'centres':>8s} {'share':>7s} {'true partner':>13s} {'saturated':>10s}")
    for k in STAGES:
        if tot[k]:
            print(f"{k:20s} {tot[k]:8d} {tot[k] / n:7.1%} {gt[k]:13d} {sat[k]:10d}")
    print(f"points kept on the part: {sum(r['on_part'] for r in res)}; baseline points lost: "
          f"{sum(r['lost'] for r in res)}")
    for key in ["accepted", "recovered"] + [f"recovered: {k}" for k in STAGES[1:]]:
        if not any(key in r["acc"] for r in res):
            continue
        cat = lambda f: np.concatenate([np.asarray(r["acc"][key][f], float) for r in res if key in r["acc"]])  # noqa
        ref, hold, plane = cat("ref"), cat("hold"), cat("plane")
        print(f"{key}:")
        print(f"  vs REFERENCE.ply (own copies left out)  {dist_stats(ref)}")
        print(f"  vs hold-out, nearest                    {dist_stats(hold)}")
        print(f"  vs hold-out, plane (16 within 1 mm)     {dist_stats(plane[plane >= 0])}   "
              f"no hold-out surface within 1 mm: {np.mean(plane < 0):.1%}")


if __name__ == "__main__":
    main()
