"""Accuracy investigation on the real bolt scans (2026-09-24): scanner or software?

Reads a CloudClean workspace READ ONLY (a local copy of the asset folders is enough: <root>/<asset id>/data.ply,
meta.json, report.json) and measures, without changing anything:

* per asset: what the viewer shows (axis-aligned box, "fitted" Open3D box), robust part dimensions, and bolt
  features measured in the bolt's own frame (thread helix fit: pitch, diameters, lead; head across-flats; axial
  faces: head top, shoulder, tip -> head height, length under the head, overall length)
* per processing step: operation drift of each child against its parent (clean = subset? merge transform rigid?
  mesh signed deviation), and what the step did to the bolt features
* scanner side: scan 05 vs scan 06 (the same bolt scanned twice): alignment candidates (the part is nearly
  symmetric), similarity fit (relative scale), surface offset, separation by region, thread offsets per segment

    python tools/accuracy/real_bolt.py <assets dir> [--out result.json]

The asset ids default to the 2026-09-17 bolt workspace (docs/accuracy-investigation-2026-09-24.md)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloudclean import accuracy as acc  # noqa: E402
from cloudclean.io import estimate_spacing, is_cloud, load  # noqa: E402
from cloudclean.register import estimate_noise  # noqa: E402

SCAN_A, SCAN_B = "03264af7e9f5", "ecce23d3a40c"
MERGE, CLEAN, MESH = "36a13a3e43aa", "d225e85ddfba", "0a2f2517ca3f"
ORDER = [SCAN_A, SCAN_B, MERGE, CLEAN, MESH]
UNC_1_8 = 25.4 / 8          # 1"-8 UNC pitch, mm
UNC_1_MAJOR = 25.4


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- bolt frame and features
def bolt_frame(pts, nrm):
    """Thread axis (helix fit), oriented head -> tip, and the thread fit itself. Two passes: a rough selection from
    the principal axis, then the head and thread are re-found around the fitted thread axis (robust to stray
    points, which a principal axis is not)."""
    c, axes, _ = acc.principal_frame(pts)
    origin, axis = c, axes[0]
    for _ in range(2):
        t = (pts - origin) @ axis
        rad = np.linalg.norm((pts - origin) - np.outer(t, axis), axis=1)
        head = rad > 15.5
        head_side = np.sign(np.median(t[head]))
        tt = -t * head_side                                  # head at negative tt, tip at positive tt
        head_lo, head_hi = np.percentile(tt[head], [1, 99])
        tip = np.percentile(tt[rad < 14.5], 99.9)
        sel = (rad < 14.5) & (tt > head_hi + 6) & (tt < tip - 6)
        th = acc.fit_thread(pts[sel], None if nrm is None else nrm[sel])
        origin, axis = th["model"]["origin"], th["model"]["axis"]
        axis = axis if (axis @ np.median(pts[head] - origin, axis=0)) < 0 else -axis   # head behind the origin
    return {"origin": origin, "axis": axis, "thread": th, "thread_mask": sel}


def coords(frame, pts):
    X = pts - frame["origin"]
    t = X @ frame["axis"]
    q = X - np.outer(t, frame["axis"])
    return t, np.linalg.norm(q, axis=1), q


def axial_faces(frame, pts, nrm):
    """Axial positions (t, head -> tip) of the head top, the shoulder under the head and the tip face: robust
    plane fits to points whose normal is within ~18 deg of the axis."""
    t, r, _ = coords(frame, pts)
    na = nrm @ frame["axis"]
    out = {}
    head = r > 15.5
    t_head_lo, t_head_hi = np.percentile(t[head], [1, 99])
    t_tip = np.percentile(t[r < 14.5], 99.9)
    specs = {
        "head_top": (na < -0.95) & (t < t_head_lo + 3) & (r < 18.5),
        "shoulder": (na > 0.95) & (np.abs(t - t_head_hi) < 3) & (r > 13.8) & (r < 18.0),
        "tip": (na > 0.95) & (t > t_tip - 3) & (r < 11.0),
    }
    for name, m in specs.items():
        if m.sum() < 200:
            out[name] = None
            continue
        f = acc.fit_plane(pts[m])
        tilt = math.degrees(math.acos(min(1.0, abs(f["normal"] @ frame["axis"]))))
        out[name] = {"t": float((f["point"] - frame["origin"]) @ frame["axis"]), "points": int(f["points_used"]),
                     "rms": f["rms"], "tilt_deg": tilt}
    lengths = {}
    if out["head_top"] and out["shoulder"]:
        lengths["head_height"] = out["shoulder"]["t"] - out["head_top"]["t"]
    if out["shoulder"] and out["tip"]:
        lengths["under_head_length"] = out["tip"]["t"] - out["shoulder"]["t"]
    if out["head_top"] and out["tip"]:
        lengths["overall_length"] = out["tip"]["t"] - out["head_top"]["t"]
    return out, lengths


def head_widths(frame, pts, step_deg=0.5):
    """Caliper widths of the head across directions perpendicular to the axis: minima = across flats,
    maxima = across corners (the head has 12 flats)."""
    t, r, q = coords(frame, pts)
    head = r > 15.5
    lo, hi = np.percentile(t[head], [1, 99])
    m = head & (t > lo + 2.0) & (t < hi - 2.0)
    Q = q[m]
    u = np.cross(frame["axis"], np.eye(3)[int(np.argmin(np.abs(frame["axis"])))])
    u /= np.linalg.norm(u)
    v = np.cross(frame["axis"], u)
    xy = np.c_[Q @ u, Q @ v]
    angles = np.radians(np.arange(0, 180, step_deg))
    widths = []
    for ang in angles:
        s = xy @ np.array([math.cos(ang), math.sin(ang)])
        lo_s, hi_s = np.quantile(s, [0.001, 0.999])
        widths.append(hi_s - lo_s)
    w = np.array(widths)
    n = len(w)
    mins = [i for i in range(n) if w[i] <= w[(i - 1) % n] and w[i] <= w[(i + 1) % n]
            and w[i] == w[max(0, i - 8):i + 9].min()]
    maxs = [i for i in range(n) if w[i] >= w[(i - 1) % n] and w[i] >= w[(i + 1) % n]
            and w[i] == w[max(0, i - 8):i + 9].max()]
    af = np.sort(w[mins])[:6] if len(mins) >= 6 else w[mins]
    return {"across_flats_mean": float(np.mean(af)), "across_flats_min": float(np.min(af)),
            "across_flats_max": float(np.max(af)), "flats_found": int(len(mins)),
            "across_corners_mean": float(np.mean(w[maxs])) if maxs else None, "points": int(m.sum())}


def crest_root(frame, pts, nrm):
    """Major / minor diameter from the crest and root flats: thread points whose normal is radial (|n.axis| < 0.25),
    above / below the middle radius. Independent of any profile model; both scans see crests and roots."""
    t, r, q = coords(frame, pts)
    m = frame["thread_mask"]
    rn = np.einsum("ij,ij->i", nrm, q / np.maximum(r, 1e-12)[:, None])
    na = np.abs(nrm @ frame["axis"])
    mid = 0.5 * (np.percentile(r[m], 2) + np.percentile(r[m], 98))
    crest = m & (na < 0.25) & (rn > 0.9) & (r > mid)
    root = m & (na < 0.25) & (rn > 0.9) & (r < mid)
    flank_tip = m & (nrm @ frame["axis"] > 0.3)
    flank_head = m & (nrm @ frame["axis"] < -0.3)
    return {"major_from_crests": 2 * float(np.median(r[crest])) if crest.sum() > 100 else None,
            "minor_from_roots": 2 * float(np.median(r[root])) if root.sum() > 100 else None,
            "crest_points": int(crest.sum()), "root_points": int(root.sum()),
            "flank_points_facing_tip": int(flank_tip.sum()), "flank_points_facing_head": int(flank_head.sum())}


def thread_halves(pts, nrm, frame):
    t, r, _ = coords(frame, pts)
    m = frame["thread_mask"]
    mid = np.median(t[m])
    out = {}
    for name, sel in (("head_half", m & (t < mid)), ("tip_half", m & (t >= mid))):
        th = acc.fit_thread(pts[sel], None if nrm is None else nrm[sel])
        out[name] = {"pitch": th["pitch"], "ppm_vs_nominal": (th["pitch"] / UNC_1_8 - 1) * 1e6,
                     "pitch_se": th["pitch_se"], "major": th["major_diameter"], "pitch_diameter": th["pitch_diameter"]}
    return out


def app_thread_analysis(pts, nrm, mask):
    """CloudClean's own thread tool (cloudclean.analysis.thread_analysis) on the same selection."""
    import open3d as o3d

    from cloudclean.analysis import thread_analysis

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[mask]))
    if nrm is not None:
        pc.normals = o3d.utility.Vector3dVector(nrm[mask])
    try:
        res = thread_analysis(pc, log=lambda m: None)
        return {"ok": True, "pitch": res["pitch"], "major_diameter": res["major_diameter"],
                "minor_diameter": res["minor_diameter"], "crest_count": res["crest_count"],
                "confidence": res["confidence"]}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


def measure_asset(root: Path, aid: str) -> dict:
    t0 = time.time()
    meta = json.loads((root / aid / "meta.json").read_text())
    geom = load(root / aid / "data.ply")
    pts, nrm = acc.surface_points(geom, 2_500_000)
    out = {"id": aid, "name": meta["name"], "kind": meta["kind"], "operation": meta.get("operation"),
           "count": int(len(acc.geometry_points(geom))),
           "displayed": {"aabb_xyz": meta["stats"]["dimensions"], "fitted_obb": meta["stats"].get("oriented_dimensions")}}
    out["spacing"] = float(estimate_spacing(pts))
    out["noise"] = float(estimate_noise(pts))
    rd = acc.robust_dimensions(pts)
    out["robust_dimensions"] = {n: rd[n] for n in acc.AXIS_NAMES}
    out["raw_principal_extents"] = rd["raw"]
    frame = bolt_frame(pts, nrm)
    th = frame["thread"]
    out["thread"] = {k: th[k] for k in ("pitch", "pitch_se", "handedness", "hand_ratio", "major_diameter",
                                        "pitch_diameter", "minor_diameter", "profile_coverage", "both_flanks", "depth",
                                        "noise", "coverage_deg", "length", "points_used", "lead_slope_ppm",
                                        "lead_residual_pp")}
    if nrm is not None:
        out["thread_crest_root"] = crest_root(frame, pts, nrm)
    out["thread"]["ppm_vs_1_8_unc"] = (th["pitch"] / UNC_1_8 - 1) * 1e6
    out["thread"]["lead"] = th["lead"]
    out["thread_halves"] = thread_halves(pts, nrm, frame)
    cyl = acc.fit_cylinder(pts[frame["thread_mask"]], None if nrm is None else nrm[frame["thread_mask"]])
    out["thread_cylinder_fit"] = {"diameter": cyl["diameter"], "rms": cyl["rms"]}
    if nrm is not None:
        faces, lengths = axial_faces(frame, pts, nrm)
        out["axial_faces"], out["lengths"] = faces, lengths
    out["head"] = head_widths(frame, pts)
    out["app_thread_tool"] = app_thread_analysis(pts, nrm, frame["thread_mask"]) if is_cloud(geom) else None
    out["frame"] = {"origin": frame["origin"], "axis": frame["axis"]}
    out["seconds"] = time.time() - t0
    pd = th["pitch_diameter"]
    cr = out.get("thread_crest_root", {})
    log(f"  {meta['name']}: pitch {th['pitch']:.5f} ({out['thread']['ppm_vs_1_8_unc']:+.0f} ppm), major "
        f"{th['major_diameter']:.4f} (crests {cr.get('major_from_crests') or 0:.4f}), pitch dia "
        f"{'n/a' if pd is None else f'{pd:.4f}'} (profile {100 * th['profile_coverage']:.0f} %), minor "
        f"{th['minor_diameter']:.4f} (roots {cr.get('minor_from_roots') or 0:.4f}), head AF "
        f"{out['head']['across_flats_mean']:.4f}, "
        f"lengths {({k: round(v, 4) for k, v in out.get('lengths', {}).items()})}, noise {out['noise']:.4f} "
        f"({out['seconds']:.0f}s)")
    return out


# --------------------------------------------------------------------------- steps and scans
def drift_steps(root: Path) -> dict:
    rep_merge = json.loads((root / MERGE / "report.json").read_text())
    T_b = np.array(rep_merge["scans"][1]["transform"])
    geoms = {aid: load(root / aid / "data.ply") for aid in (SCAN_A, SCAN_B, MERGE, CLEAN, MESH)}
    out = {}
    log("  merge vs scan 05 (reference, identity)")
    out["merge_vs_05"] = acc.operation_drift(geoms[SCAN_A], geoms[MERGE], "merge", np.eye(4), log=log)
    log("  merge vs scan 06 (approved transform)")
    out["merge_vs_06"] = acc.operation_drift(geoms[SCAN_B], geoms[MERGE], "merge", T_b, log=log)
    log("  clean vs merge")
    out["clean_vs_merge"] = acc.operation_drift(geoms[MERGE], geoms[CLEAN], "clean", log=log)
    log("  mesh vs clean")
    out["mesh_vs_clean"] = acc.operation_drift(geoms[CLEAN], geoms[MESH], "mesh", log=log)
    return out, T_b


def compare(root: Path, T_merge: np.ndarray, T_align: np.ndarray | None) -> dict:
    A = load(root / SCAN_A / "data.ply")
    B = load(root / SCAN_B / "data.ply")
    out = {}
    if T_align is None:
        T_align, info = acc.align_scans(A, B, log=log)
        out["alignment"] = info
    out["T_align"] = T_align
    out["pose_difference_to_merge_deg"] = acc.rotation_angle_deg(T_align[:3, :3].T @ T_merge[:3, :3])
    for name, T in (("own_alignment", T_align), ("merge_pose", T_merge)):
        log(f"  compare scans at {name}")
        res = acc.compare_scans(A, B, transform=T, log=log)
        out[name] = res
        out[name + "_regions"] = region_breakdown(A, B, res["transform"])
        out[name + "_thread_only"] = thread_only(A, B, T)
        out[name + "_lengths"] = pose_lengths(A, B, T)
    return out


def region_breakdown(A, B, T) -> dict:
    """Separation of B (moved by T) from A's surface by bolt region, and thread offsets per axial segment."""
    pa, na = np.asarray(A.points), np.asarray(A.normals)
    pb, nb = np.asarray(B.points), np.asarray(B.normals)
    frame = bolt_frame(pa, na)
    model = frame["thread"]["model"]
    Pb = acc.apply_transform(T, pb)
    Nb = nb @ T[:3, :3].T
    surface = acc.SurfaceModel(A)
    d, _, gap, ok = surface.query(Pb, Nb, 30.0)
    good = ok & (gap <= 1.5 * 0.15) & (np.abs(d) < 1.0)
    t, r, _ = coords(frame, Pb)
    ta, ra, _ = coords(frame, pa)
    t_head_hi = np.percentile(ta[ra > 15.5], 99)
    na_ax = Nb @ frame["axis"]
    regions = {"head_side": (r > 14) & (np.abs(na_ax) < 0.5),
               "head_axial_faces": (r > 13) & (np.abs(na_ax) >= 0.5),
               "thread_flanks_facing_tip": (r < 13.5) & (t > t_head_hi + 3) & (na_ax > 0.3),
               "thread_flanks_facing_head": (r < 13.5) & (t > t_head_hi + 3) & (na_ax < -0.3),
               "thread_crests_roots": (r < 13.5) & (t > t_head_hi + 3) & (np.abs(na_ax) <= 0.3)}
    out = {"regions": {}}
    for name, m in regions.items():
        g = good & m
        out["regions"][name] = acc._stats(d[g]) if g.sum() >= 50 else None
    sgn = 1.0 if model["axis"] @ frame["axis"] > 0 else -1.0
    tm = ta[frame["thread_mask"]]
    edges = np.linspace(tm.min(), tm.max(), 11)
    edges_m = np.sort(edges * sgn)
    thr_b = (r < 13.5) & (t > t_head_hi + 3)
    off_a = acc.thread_offsets(model, pa[frame["thread_mask"]], edges_m)
    off_b = acc.thread_offsets(model, Pb[thr_b], edges_m)
    seg = []
    for sa, sb in zip(off_a, off_b):
        if sa["axial_offset"] is None or sb["axial_offset"] is None:
            continue
        seg.append({"t_head_to_tip": sa["t"] * sgn,
                    "axial_b_minus_a_towards_tip": (sb["axial_offset"] - sa["axial_offset"]) * sgn,
                    "radial_b_minus_a": sb["radial_offset"] - sa["radial_offset"]})
    seg.sort(key=lambda s: s["t_head_to_tip"])
    out["thread_segments"] = seg
    if len(seg) >= 3:
        tt = np.array([s["t_head_to_tip"] for s in seg])
        ax = np.array([s["axial_b_minus_a_towards_tip"] for s in seg])
        k1, k0 = np.polyfit(tt, ax, 1)
        out["thread_axial_trend_ppm"] = float(k1 * 1e6)
        out["thread_axial_mean"] = float(ax.mean())
        out["thread_radial_mean"] = float(np.mean([s["radial_b_minus_a"] for s in seg]))
    return out


def pose_lengths(A, B, T) -> dict:
    """Head height and overall length of the merged bolt when scan B is placed by T: the head top is only in scan 06,
    the shoulder and the tip only in scan 05, so both depend on where the pose puts scan 06 along the axis."""
    pa, na = np.asarray(A.points), np.asarray(A.normals)
    frame = bolt_frame(pa, na)
    faces_a, _ = axial_faces(frame, pa, na)
    Pb = acc.apply_transform(T, np.asarray(B.points))
    Nb = np.asarray(B.normals) @ T[:3, :3].T
    faces_b, _ = axial_faces(frame, Pb, Nb)
    top = faces_b["head_top"]["t"]
    return {"head_top_t": top, "head_height": faces_a["shoulder"]["t"] - top,
            "overall_length": faces_a["tip"]["t"] - top, "under_head_length": faces_a["tip"]["t"] - faces_a["shoulder"]["t"]}


def thread_only(A, B, T) -> dict:
    """Scan B vs scan A on the threaded section only (the head's dimples differ between the scans by up to 0.5 mm
    and would dominate a whole-part fit): scale along the axis, across it, and a thickness offset."""
    pa, na = np.asarray(A.points), np.asarray(A.normals)
    pb, nb = np.asarray(B.points), np.asarray(B.normals)
    frame = bolt_frame(pa, na)
    t_a, r_a, _ = coords(frame, pa)
    thr_a = frame["thread_mask"]
    Pb = acc.apply_transform(T, pb)
    Nb = nb @ T[:3, :3].T
    t_b, r_b, _ = coords(frame, Pb)
    lo, hi = t_a[thr_a].min(), t_a[thr_a].max()
    thr_b = (r_b < 14.5) & (t_b > lo) & (t_b < hi)
    import open3d as o3d

    cloud_a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pa[thr_a]))
    cloud_a.normals = o3d.utility.Vector3dVector(na[thr_a])
    surface = acc.SurfaceModel(cloud_a)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(thr_b)
    idx = idx if len(idx) <= 80_000 else np.sort(rng.choice(idx, 80_000, replace=False))
    # frame axes: bolt axis first, then two across it
    a = frame["axis"]
    u = np.cross(a, np.eye(3)[int(np.argmin(np.abs(a)))])
    u /= np.linalg.norm(u)
    F = np.array([a, u, np.cross(a, u)])
    out = {}
    for mode in ("rigid", "axes_offset", "similarity_offset"):
        _, info = acc.fit_pose(surface, Pb[idx], np.eye(4), mode, frame_axes=F, normals=Nb[idx])
        out[mode] = {k: info.get(k) for k in ("rms", "overlap", "scale", "scale_se", "axis_scales", "axis_scales_se",
                                                 "offset", "offset_se", "iterations", "converged")}
    ax = out["axes_offset"]["axis_scales"]
    out["axial_ppm"] = (1 / ax[0] - 1) * 1e6
    out["radial_ppm"] = [(1 / ax[1] - 1) * 1e6, (1 / ax[2] - 1) * 1e6]
    out["offset_mm"] = out["axes_offset"]["offset"]
    d, _, gap, ok = surface.query(Pb[idx], Nb[idx], 30.0)
    good = ok & (gap <= 0.25)
    out["separation"] = acc._stats(d[good])
    out["overlap_fraction"] = float(good.mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assets", type=Path, help="folder holding the asset folders (data.ply, meta.json, report.json)")
    ap.add_argument("--out", type=Path, default=Path("real_bolt_result.json"))
    ap.add_argument("--transform", type=Path, default=None, help="optional .npy pose of scan 06 in scan 05 (skips "
                                                                  "the multi-start alignment)")
    ap.add_argument("--skip", default="", help="comma list of sections to skip: assets,drift,compare")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    if "assets" not in skip:
        log("Per asset measurements")
        result["assets"] = {aid: measure_asset(args.assets, aid) for aid in ORDER}
        args.out.write_text(json.dumps(acc.jsonable(result), indent=1))
    T_merge = np.array(json.loads((args.assets / MERGE / "report.json").read_text())["scans"][1]["transform"])
    if "drift" not in skip:
        log("Operation drift")
        drift, _ = drift_steps(args.assets)
        result["drift"] = drift
        args.out.write_text(json.dumps(acc.jsonable(result), indent=1))
    if "compare" not in skip:
        log("Scan 05 vs scan 06")
        T_align = np.load(args.transform) if args.transform else None
        result["compare"] = compare(args.assets, T_merge, T_align)
        args.out.write_text(json.dumps(acc.jsonable(result), indent=1))
    log(f"Saved {args.out}")


if __name__ == "__main__":
    main()
