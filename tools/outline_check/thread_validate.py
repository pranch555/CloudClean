"""The thread's groove-by-groove check on synthetic photos (docs/outline-check.md, Threads groove by groove).

A real golden screw (its mesh) gets known thread errors by moving its vertices in its helix's coordinates
(outline_synthetic.thread_errors): one groove 0.10 shallower, one tooth moved 0.05 along the axis, the thread
stretched 0.1 % (a progressive lead error) and one crest nicked 0.10 on one side. It is rendered lying across the
view, together with the perfect screw, for each camera:

  iphone48           8064 x 6048 iPhone-like, backlit on a light pad, 280 mm, 2 degrees from straight down
  scanner35_coax     the MetroY's left camera (its camparam.yaml), 1600 x 1200, 270 mm, tilted 35 degrees (the tilt
                     along the photo's short side, as the scanner stands), front-lit by a light at the lens; the screw
                     lies across the tilt
  scanner_down_coax  the same camera 215 mm straight down, front-lit the same way (6 markers in view)
  scanner35, scanner_down  the same two with the light 5 mm beside the lens (its shadow shades the grooves)
  scanner35_lamp     scanner35 with the light 30 mm beside the lens: what a lamp's shadow does

and checked (with the true camera; the bar exactly 100 mm). Per error the table gives what was injected, what the
exact silhouette of the part shows at the fitted pose (an ideal photo: the method alone), what the photo gave, its U
(k = 2) and verdict; and over all the other teeth and grooves, the largest error and how often |error| <= U.

    python tools/outline_check/thread_validate.py OUTDIR --screw GOLDEN.ply [--camparam camparam.yaml]
                                                 [--cases iphone48,scanner35_coax,...] [--parts errors,perfect]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloudclean import outline_camera as oc  # noqa: E402
from cloudclean import outline_check as ock  # noqa: E402
from cloudclean import outline_synthetic as syn  # noqa: E402
from cloudclean import outline_thread as oth  # noqa: E402

TRY5 = Path(r"C:\Users\user\Desktop\CloudClean photo check\metroy\try5\camparam.yaml")
CASES = {
    "iphone48": {"camera": "phone48", "distance": 280.0, "tilt": 2.0, "azimuth": 40.0, "roll": 0.0, "light": None,
                 "blur": 1.2, "ss": 3, "exposure": 0.78, "part_level": 0.02, "noise": 1.0, "axis": "y",
                 "format": "jpg"},
    "scanner35": {"camera": "metroy", "distance": 270.0, "tilt": 35.0, "azimuth": 270.0, "roll": 90.0,
                  "light": (5.0, 0.0, 0.0), "blur": 1.3, "ss": 3, "exposure": 0.72, "part_level": 0.06,
                  "noise": 0.3, "axis": "x", "format": "png"},
    "scanner_down": {"camera": "metroy", "distance": 215.0, "tilt": 0.0, "azimuth": 0.0, "roll": 90.0,
                     "light": (5.0, 0.0, 0.0), "blur": 1.3, "ss": 3, "exposure": 0.72, "part_level": 0.06,
                     "noise": 0.3, "axis": "x", "format": "png"},
    # the light exactly at the lens (no shadow at all next to the outline: the method alone, front-lit)
    "scanner35_coax": {"camera": "metroy", "distance": 270.0, "tilt": 35.0, "azimuth": 270.0, "roll": 90.0,
                       "light": (0.0, 0.0, 0.0), "blur": 1.3, "ss": 3, "exposure": 0.72, "part_level": 0.06,
                       "noise": 0.3, "axis": "x", "format": "png"},
    "scanner_down_coax": {"camera": "metroy", "distance": 215.0, "tilt": 0.0, "azimuth": 0.0, "roll": 90.0,
                          "light": (0.0, 0.0, 0.0), "blur": 1.3, "ss": 3, "exposure": 0.72, "part_level": 0.06,
                          "noise": 0.3, "axis": "x", "format": "png"},
    # the same as scanner35 with the light 30 mm off the lens (a desk lamp, or a fill light far from the camera):
    # the part's shadow on the paper joins its silhouette where the paper behind it is in view
    "scanner35_lamp": {"camera": "metroy", "distance": 270.0, "tilt": 35.0, "azimuth": 270.0, "roll": 90.0,
                       "light": (30.0, 0.0, 0.0), "blur": 1.3, "ss": 3, "exposure": 0.72, "part_level": 0.06,
                       "noise": 0.3, "axis": "x", "format": "png"},
}
SHALLOW, SHIFT, STRETCH, NICK = 0.10, 0.05, 0.001, 0.10


def lay(mesh, pr, axis: str):
    """The golden mesh lying on the sheet with its axis along the sheet's x or y, its middle at (0, 5): returns
    (R, t) golden -> sheet."""
    down = pr["u"]
    V0 = np.asarray(mesh.vertices)
    P = syn.place(mesh, down, 0.0, 5.0, 0.0)
    R, t = syn.rigid_between(V0, np.asarray(P.vertices))
    a = R @ pr["a"]
    ang = math.degrees(math.atan2(a[1], a[0]))
    yaw = (0.0 if axis == "x" else 90.0) - ang
    P = syn.place(mesh, down, 0.0, 5.0, yaw)
    return syn.rigid_between(V0, np.asarray(P.vertices))


def camera_for(case: dict, camparam: Path):
    if case["camera"] == "phone48":
        cam = syn.phone_camera(8064, 6048, 69.0, (0.02, -0.03, 0.0002, -0.0001, 0.01), (12.0, -8.0))
        cam.f_sigma_pct = 0.1
        return cam
    return oc.from_metroy(camparam, "L")


def expected_rows(thread: dict, info: dict, sgn: float, nick_angle: float) -> dict:
    """What each tooth and groove of the inspection should show for the injected errors (None for the perfect part):
    {(side, 'tooth'|'groove', n): {field: value}}."""
    out = {}
    p, root_t0, hand, seam = info["pitch"], info["root_t0"], info["hand_sign"], math.radians(info["seam_deg"])
    errs = {e["kind"]: e for e in info["errors"]}
    for sd in thread["sides"]:
        th = math.radians(sd["angle_deg"])
        rep = seam + (th - seam) % (2 * math.pi)
        off = hand * rep / (2 * math.pi)
        teeth = [tt for tt in sd["teeth"] if tt.get("n")]
        first = next((tt for tt in teeth if "axial" in tt), None)

        def disp(tt):           # the tooth's true shift along the thread from its start (mm)
            d = 0.0
            if "stretch" in errs:
                d += errs["stretch"]["share"] * sgn * (tt["t"] - errs["stretch"]["about_t"])
            if "shift" in errs:
                j = errs["shift"]["tooth"]
                if abs(tt["t"] - (root_t0 + p * (j + 0.5 + off))) < p / 4:
                    d += sgn * errs["shift"]["mm"]
            return d

        for k, tt in enumerate(teeth):
            e = {"lead_error": disp(tt) - disp(first) if first else None}
            nxt = teeth[k + 1] if k + 1 < len(teeth) else None
            if nxt is not None:
                e["pitch_error"] = disp(nxt) - disp(tt)
            e["crest_error"] = 0.0
            if "nick" in errs:
                j = errs["nick"]["tooth"]
                near_side = abs(math.remainder(th - math.radians(nick_angle), 2 * math.pi)) < math.radians(40)
                if near_side and abs(tt["t"] - (root_t0 + p * (j + 0.5 + off))) < p / 4:
                    e["crest_error"] = -errs["nick"]["mm"]
            out[(sd["side"], "tooth", tt["n"])] = e
        for g in sd["grooves"]:
            if not g.get("n"):
                continue
            e = {"depth_error": 0.0}
            if "shallow" in errs:
                k = errs["shallow"]["groove"]
                if abs(g["t"] - (root_t0 + p * (k + off))) < p / 4:
                    e["depth_error"] = -errs["shallow"]["mm"]
            out[(sd["side"], "groove", g["n"])] = e
    return out


def compare(thread: dict, exact: dict | None, expect: dict) -> dict:
    """Per injected error: injected, exact silhouette, photo ± U, verdict; over the rest: worst |error| and the share
    within U."""
    rows, rest_err, rest_in = [], [], []
    p = thread["pitch"]
    ex: dict = {}               # the exact silhouette's teeth and grooves by side, matched to the photo's by position
    if exact is not None:
        for sd in exact["sides"]:
            ph = next((s for s in thread["sides"] if s["side"] == sd["side"]), {"teeth": []})
            ref = next((tt["t"] for tt in ph["teeth"] if tt.get("lead_status") == "reference"), None)
            base = next((tt.get("lead_error") for tt in sd["teeth"] if ref is not None and "lead_error" in tt
                         and abs(tt["t"] - ref) < p / 4), None)
            for tt in sd["teeth"]:
                tt = dict(tt)
                if base is not None and tt.get("lead_error") is not None:
                    tt["lead_error"] = tt["lead_error"] - base     # from the same first tooth as the photo's
                ex.setdefault((sd["side"], "tooth"), []).append(tt)
            ex[(sd["side"], "groove")] = list(sd["grooves"])

    def exact_of(side, what, t):
        return next((it for it in ex.get((side, what), []) if abs(it["t"] - t) < p / 4), {})

    for sd in thread["sides"]:
        items = [("tooth", tt) for tt in sd["teeth"] if tt.get("n")] + [("groove", g) for g in sd["grooves"]
                                                                          if g.get("n")]
        for what, it in items:
            key = (sd["side"], what, it["n"])
            e = expect.get(key, {})
            for fld in ("pitch_error", "lead_error", "crest_error", "depth_error"):
                if fld not in it or e.get(fld) is None:
                    continue
                inj = e[fld]
                val, U = it[fld], it[fld.replace("_error", "_U")]
                status = it[fld.replace("_error", "_status")]
                xv = exact_of(sd["side"], what, it["t"]).get(fld)
                injected = abs(inj) >= 0.02 or (fld == "lead_error" and abs(inj) >= 0.02)
                if injected:
                    rows.append({"side": sd["side"], what: it["n"], "position": it["position"], "value": fld,
                                 "injected": round(inj, 4), "exact": None if xv is None else round(xv, 4),
                                 "photo": val, "U": U, "status": status, "error": round(val - inj, 4)})
                elif status != "reference":
                    rest_err.append(abs(val - inj))
                    rest_in.append(abs(val - inj) <= U)
    return {"injected": rows, "others": {"n": len(rest_err), "max_abs_error": round(max(rest_err), 4) if rest_err
                                         else None, "within_U": round(float(np.mean(rest_in)), 3) if rest_in
                                         else None}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--screw", required=True, help="the real golden screw (PLY / STL / STEP)")
    ap.add_argument("--camparam", default=str(TRY5), help="a MetroY camparam.yaml for the scanner cases")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--parts", default="errors,perfect")
    args = ap.parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    quiet = lambda m: None  # noqa: E731
    t0 = time.time()
    mesh = ock.load_golden(args.screw, quiet)
    golden = ock.GoldenPart(mesh, quiet)
    print(f"golden model ready ({time.time() - t0:.0f} s)", flush=True)
    pr = golden.profile
    z = oth.thread_zone(golden)
    sgn = oth._tip_sign(golden, z)
    results = []
    for name in args.cases.split(","):
        case = CASES[name]
        cam = camera_for(case, Path(args.camparam))
        Rg, tg = lay(mesh, pr, case["axis"])
        R, t = syn.view(case["distance"], case["tilt"], case["azimuth"], case["roll"], (0.0, 5.0, 0.0))
        C = -R.T @ t
        Cg = Rg.T @ (C - tg)                                   # the camera in the golden frame
        rel = Cg - pr["o"]
        th_cam = math.degrees(math.atan2(rel @ pr["v"], rel @ pr["u"]))
        light = None if case["light"] is None else C + np.asarray(case["light"])
        for part in args.parts.split(","):
            d = root / name / part
            d.mkdir(parents=True, exist_ok=True)
            info = {"errors": [], "pitch": z["pitch"], "root_t0": z["root_t0"], "hand_sign": z["hand_sign"],
                    "seam_deg": th_cam}
            m = mesh
            nick_angle = th_cam + 90.0
            if part == "errors":
                m, info = syn.thread_errors(mesh, pr, shallow=(3, SHALLOW), shift=(0, SHIFT), stretch=STRETCH,
                                            nick=(5, nick_angle, NICK, 20.0), seam_deg=th_cam)
            photo_path = d / f"photo.{case['format']}"
            placed = syn._mesh(np.asarray(m.vertices) @ Rg.T + tg, np.asarray(m.triangles))
            if not photo_path.exists():
                ts = time.time()
                img = syn.render(cam, R, t, placed, "a4", (1.0, 1.0), case["exposure"], part_level=case["part_level"],
                                 blur_px=case["blur"], ss=case["ss"], noise=case["noise"], seed=7, light=light)
                syn.save_photo(photo_path, img, orientation=8 if case["camera"] == "phone48" else 1,
                               meta=syn.camera_meta(cam) if case["camera"] == "phone48" else {"make": "Synthetic",
                                                                                               "model": "MetroY-like",
                                                                                               "lens": "scanner"})
                print(f"{name}/{part}: rendered ({time.time() - ts:.0f} s)", flush=True)
            par = ock.OutlineParams(bar_mm=100.0, paper="a4", mc_samples=16)
            ts = time.time()
            photo = oc.load_photo(photo_path)
            rep = ock.check_photo(golden, photo, cam, par, log=quiet)
            secs = time.time() - ts
            thread = rep.get("_thread")
            pose = rep["_pose"]
            exact = None
            if thread and thread.get("measurable"):
                edges = syn.silhouette_edges(placed, pose["camera"], pose["sheet"])
                exact = oth.inspect_thread(golden, pose["rest"], pose["fit"], None, pose["camera"], pose["sheet"],
                                           0.0, 0.0, par, oth_rel_u(rep), edges=edges)
                oth.thread_overlay(photo, thread, d / "thread.png")
                oth.thread_chart(thread, d / "thread-chart.png")
                ock.overlay(photo, rep, par.tolerance, d / "outline.png")
            expect = expected_rows(thread, info, sgn, nick_angle) if thread and thread.get("measurable") else {}
            cmp_ = compare(thread, exact, expect) if thread and thread.get("measurable") else None
            res = {"case": name, "part": part, "seconds": round(secs, 1), "injected": info["errors"],
                   "view_angle_deg": thread and thread.get("view_angle_deg"),
                   "px_per_mm": thread and thread.get("px_per_mm"), "blur_px": rep["part"]["edge_blur_px"],
                   "fit_sigma_px": rep["part"]["fit_sigma_px"],
                   "thread": rep.get("thread"), "exact_summary": exact and exact.get("summary"),
                   "comparison": cmp_,
                   "measurements": [{k: mm.get(k) for k in ("name", "golden", "photo", "difference", "uncertainty",
                                                             "status", "where", "reason")}
                                    for mm in rep["measurements"]], "warnings": rep["warnings"]}
            (d / "result.json").write_text(json.dumps(res, indent=1, default=ock._json_default), encoding="utf-8")
            results.append(res)
            print(f"{name}/{part}: checked in {secs:.0f} s", flush=True)
            oth.print_thread(rep.get("thread"), print, f"{name}/{part}")
            if cmp_:
                for r in cmp_["injected"]:
                    print("   injected:", r, flush=True)
                print("   others:", cmp_["others"], flush=True)
    lines = ["| Camera | Part | Value | Where | Injected | Exact silhouette | Photo | U (k=2) | Verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["comparison"]
        if c is None:
            reason = (r["thread"] or {}).get("reason", "")
            lines.append(f"| {r['case']} | {r['part']} | thread | | | | not measured: {reason} | | |")
            continue
        for row in c["injected"]:
            item = f"tooth {row['tooth']}" if "tooth" in row else f"groove {row['groove']}"
            where = f"side {row['side']} {item} at {row['position']:.1f} mm"
            ex = "" if row["exact"] is None else f"{row['exact']:+.4f}"
            lines.append(f"| {r['case']} | {r['part']} | {row['value']} | {where} | {row['injected']:+.4f} | {ex} | "
                         f"{row['photo']:+.4f} | {row['U']:.4f} | {row['status']} |")
        o = c["others"]
        lines.append(f"| {r['case']} | {r['part']} | all other teeth / grooves ({o['n']} values) | | 0 | | "
                     f"worst {o['max_abs_error'] if o['max_abs_error'] is not None else '-'} | within U: "
                     f"{o['within_U']} | |")
    (root / "thread_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "thread_validation.json").write_text(json.dumps(results, indent=1, default=ock._json_default),
                                                 encoding="utf-8")
    print(f"Wrote {root / 'thread_validation.md'}")
    return 0


def oth_rel_u(rep: dict) -> float:
    return ock._length_rel_u(rep["measurements"])


if __name__ == "__main__":
    sys.exit(main())
