"""The outline check's synthetic accuracy table (docs/outline-check.md, Accuracy on synthetic photos).

Renders test sets with outline_check.py render (unless they exist), calibrates the camera from the first set's
sheet-only photos, runs the check on every case and writes VALDIR/validation.json and VALDIR/validation.md: injected
error, recovered error, its uncertainty U (k = 2) and the verdict for every value.

    python tools/outline_check/validate.py VALDIR [--screw GOLDEN.ply] [--only bolt|screw] [--jobs 3] [--quick]

Cases (the camera calibrated from 10 tilted sheet-only renders, the bar "measured" exactly):
  straight      bolt with head_h +0.05, under -0.10, shank_d +0.03; one photo 0-3 degrees from straight down
  tilted20      the same bolt, one photo tilted 20 degrees (it also measures its own focal length)
  repeat3       the straight set's three photos, combined
  printed       printed 1 % big in x and 0.5 % in y; bar and height measured (101.0, 120.6 mm)
  printed_nobar the same photo, the bar not measured (100 % assumed): what an unmeasured print does
  f+1%          the straight photo checked with the calibrated focal length 1 % too long
  f+1%_tilted   the tilted photo, same wrong focal length (the photo measures its own)
  perfect       the bolt as drawn
  overexposed   the bolt with errors, paper clipped white (exposure 1.5), markers' edge correction on / off
  plate         an L-plate with a through hole: length +0.03, step_x -0.06, hole_d +0.04, thickness +0.05
  plate_perfect the plate as drawn
  48mp          the straight bolt at 8064 x 6048 (48 MP), checked with the true camera
With --screw (a real golden screw mesh; its errors made by moving its vertices: head top +0.05, thread end -0.10
shorter, thread radially +0.03):
  screw_perfect, screw_straight, screw_repeat3, screw_tilted20, screw_printed76 (printed at 76.2 %, bar 76.2 and
  height 91.44 measured), screw_48mp (8064 x 6048, true camera; also timed)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloudclean import outline_camera as oc  # noqa: E402
from cloudclean import outline_check as ock  # noqa: E402
from cloudclean import outline_synthetic as syn  # noqa: E402

CLI = Path(__file__).with_name("outline_check.py")
BOLT_ERR = "head_h=0.05,under=-0.10,shank_d=0.03"
PLATE_ERR = "length=0.03,step_x=-0.06,hole_d=0.04,thickness=0.05"
SCREW_ERR = "head_top=0.05,end=-0.10,radial=0.03"
SETS = {   # name: render arguments
    "bolt_errors": ["bolt", "--calibration", "10", "--photos", "3", "--errors", BOLT_ERR, "--seed", "1"],
    "bolt_tilted": ["bolt", "--calibration", "0", "--photos", "1", "--tilt", "20", "--errors", BOLT_ERR,
                    "--seed", "2"],
    "bolt_printed": ["bolt", "--calibration", "0", "--photos", "1", "--printed", "1.01,1.005", "--errors", BOLT_ERR,
                     "--seed", "3"],
    "bolt_perfect": ["bolt", "--calibration", "0", "--photos", "1", "--seed", "4"],
    "bolt_overexposed": ["bolt", "--calibration", "0", "--photos", "1", "--exposure", "1.5", "--errors", BOLT_ERR,
                         "--seed", "5"],
    "plate_errors": ["plate", "--calibration", "0", "--photos", "1", "--errors", PLATE_ERR, "--pose=3,-4,28",
                     "--seed", "6"],
    "plate_perfect": ["plate", "--calibration", "0", "--photos", "1", "--pose=-2,5,-35", "--seed", "7"],
    "bolt_48mp": ["bolt", "--calibration", "0", "--photos", "1", "--errors", BOLT_ERR, "--width", "8064",
                  "--height", "6048", "--ss", "3", "--blur", "1.2", "--seed", "8"],
}


def screw_sets(mesh: str) -> dict:
    m = ["mesh", "--mesh", mesh, "--calibration", "0"]
    e = ["--turned-errors", SCREW_ERR]
    return {"screw_perfect": m + ["--photos", "1", "--seed", "11"],
            "screw_errors": m + ["--photos", "3", "--seed", "12"] + e,
            "screw_tilted": m + ["--photos", "1", "--tilt", "20", "--seed", "13"] + e,
            "screw_printed": m + ["--photos", "1", "--printed", "0.762", "--pose=0,0,80", "--seed", "14"] + e,
            "screw_48mp": m + ["--photos", "1", "--width", "8064", "--height", "6048", "--ss", "3", "--blur", "1.2",
                               "--seed", "15"] + e}


def render_sets(root: Path, sets: dict, names: list[str], jobs: int) -> None:
    todo = [n for n in names if not (root / n / "truth.json").exists()]

    def one(n):
        log = open(root / f"{n}.log", "w")
        subprocess.run([sys.executable, str(CLI), "render", sets[n][0], str(root / n), *sets[n][1:]], stdout=log,
                       stderr=subprocess.STDOUT, check=True)
        return n

    with ThreadPoolExecutor(jobs) as ex:
        for n in ex.map(one, todo):
            print(f"rendered {n}", flush=True)


def rows_for(case: str, result: dict, truth: dict) -> list[dict]:
    out = []
    for m in result["combined"]:
        tv = syn.true_value(truth["expected"], m["kind"], m["golden"])
        row = {"case": case, "name": m["name"], "kind": m["kind"], "golden": m["golden"],
               "injected": None if tv is None else round(tv - m["golden"], 4), "status": m["status"]}
        if m["status"] != "not_measured":
            row.update(recovered=m["difference"], error=None if tv is None else round(m["photo"] - tv, 4),
                       U=m["uncertainty"])
        else:
            row["reason"] = m.get("reason")
        out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--screw", help="a real golden screw (PLY / STL / STEP) for the round-part cases")
    ap.add_argument("--only", choices=["bolt", "screw"], help="only the bolt / plate cases, or only the screw's")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="skip the 48 MP renders")
    args = ap.parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    sets = dict(SETS)
    if args.screw:
        sets.update(screw_sets(args.screw))
    names = ["bolt_errors"] + [n for n in sets if n != "bolt_errors" and not (args.quick and n.endswith("48mp"))
                               and not (args.only == "bolt" and n.startswith("screw"))
                               and not (args.only == "screw" and not n.startswith("screw"))]
    render_sets(root, sets, names, args.jobs)
    quiet = lambda m: None  # noqa: E731
    truth = {n: json.loads((root / n / "truth.json").read_text()) for n in names}

    # the camera, from the first set's sheet-only photos
    t0 = time.time()
    cam = oc.calibrate(sorted((root / "bolt_errors" / "calibration").iterdir()), log=quiet)
    cam.save(root / "camera.json")
    true_cam = oc.Camera.from_json(truth["bolt_errors"]["camera"])
    fs = cam.f_rel_sigma()
    calib = {"views": cam.info["used"], "rms_px": round(cam.rms_px, 4), "f_px": round(cam.f, 3),
             "f_true_px": round(true_cam.f, 3), "f_error_pct": round(100 * (cam.f / true_cam.f - 1), 4),
             "f_sigma_pct": round(100 * fs, 4),
             "cx_error_px": round(float(cam.K[0, 2] - true_cam.K[0, 2]), 3),
             "cy_error_px": round(float(cam.K[1, 2] - true_cam.K[1, 2]), 3),
             "k1": [round(float(cam.dist[0]), 5), round(float(true_cam.dist[0]), 5)],
             "seconds": round(time.time() - t0, 1)}
    print("calibration", calib, flush=True)
    golden: dict = {}

    def golden_for(n):
        path = root / n / "golden.ply"
        key = truth[n]["part"] + (truth[n]["golden_params"].get("mesh", "") if truth[n]["part"] == "mesh" else "")
        if key not in golden:
            t = time.time()
            golden[key] = ock.GoldenPart(ock.load_golden(path), quiet)
            golden[key + "_seconds"] = round(time.time() - t, 1)
        return golden[key]

    def photos(n, k=None):
        ps = sorted((root / n / "photos").iterdir())
        return ps if k is None else ps[k:k + 1]

    def run(case, n, camera, k=0, height=False, **kw):
        tr = truth[n]
        par = ock.OutlineParams(**{"bar_mm": tr["bar_mm"], "height_mm": tr["height_mm"] if height else 0.0, **kw})
        g = golden_for(n)
        t = time.time()
        res = ock.check_photos(g, photos(n, k), camera, par, out_dir=root / "out" / case, log=quiet)
        secs = time.time() - t
        rows = rows_for(case, res, tr)
        print(f"{case}: {res['verdict']} ({secs:.0f} s)", flush=True)
        for r in rows:
            print("   ", r, flush=True)
        return {"case": case, "verdict": res["verdict"], "headline": res["headline"], "rows": rows,
                "seconds": round(secs, 1), "warnings": res["warnings"],
                "photos": [{k2: p[k2] for k2 in ("photo", "sheet", "part", "camera")} for p in res["photos"]]}

    cases = []
    if args.only != "screw":
        cases += [run("straight", "bolt_errors", cam),
                  run("tilted20", "bolt_tilted", cam),
                  run("repeat3", "bolt_errors", cam, k=None),
                  run("printed", "bolt_printed", cam, height=True),
                  run("printed_nobar", "bolt_printed", cam, bar_mm=0.0),
                  run("f+1%", "bolt_errors", cam.scaled_f(1.01)),
                  run("f+1%_tilted", "bolt_tilted", cam.scaled_f(1.01)),
                  run("perfect", "bolt_perfect", cam),
                  run("overexposed", "bolt_overexposed", cam),
                  run("overexposed_nocorr", "bolt_overexposed", cam, edge_correction="none"),
                  run("plate", "plate_errors", cam),
                  run("plate_perfect", "plate_perfect", cam)]
        if "bolt_48mp" in names:
            cases.append(run("48mp", "bolt_48mp", oc.Camera.from_json(truth["bolt_48mp"]["camera"])))
            cases.append(run("12mp_true_camera", "bolt_errors", true_cam))
    if args.screw and args.only != "bolt":
        cases += [run("screw_perfect", "screw_perfect", cam),
                  run("screw_straight", "screw_errors", cam),
                  run("screw_repeat3", "screw_errors", cam, k=None),
                  run("screw_tilted20", "screw_tilted", cam),
                  run("screw_printed76", "screw_printed", cam, height=True)]
        if "screw_48mp" in names:
            cases.append(run("screw_48mp", "screw_48mp", oc.Camera.from_json(truth["screw_48mp"]["camera"])))
    out = {"calibration": calib, "cases": cases,
           "golden_seconds": {k: v for k, v in golden.items() if k.endswith("_seconds")}}
    stem = "validation" if not args.only else f"validation_{args.only}"
    (root / f"{stem}.json").write_text(json.dumps(out, indent=1, default=ock._json_default), encoding="utf-8")
    lines = ["| Case | Value | Golden | Injected | Recovered | Error | U (k=2) | Verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for c in cases:
        for r in c["rows"]:
            inj = "" if r["injected"] is None else f"{r['injected']:+.3f}"
            if r["status"] == "not_measured":
                lines.append(f"| {c['case']} | {r['name']} | {r['golden']:.3f} | {inj} | not measured | | | |")
                continue
            err = "" if r["error"] is None else f"{r['error']:+.4f}"
            lines.append(f"| {c['case']} | {r['name']} | {r['golden']:.3f} | {inj} | {r['recovered']:+.4f} | {err} | "
                         f"{r['U']:.3f} | {r['status']} |")
    (root / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {root / (stem + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
