"""Check a part against its golden model from a photo of it lying on the photo check sheet (docs/outline-check.md).

    python tools/outline_check/outline_check.py calibrate PHOTOS... --out camera.json [--bar-mm 100.02]
                                                          [--height-mm 120.01]
    python tools/outline_check/outline_check.py check GOLDEN PHOTOS... (--camera camera.json |
                                                     --metroy-camparam camparam.yaml) --bar-mm 100.02
                                                     [--height-mm 120.01] [--tolerance 0.1] [--paper a4|letter]
                                                     [--out DIR] [--edge-correction markers|none]
    python tools/outline_check/outline_check.py render bolt|plate|screw OUT [--calibration 10] [--photos 1]
                                                     [--tilt 0] [--errors head_h=0.05,under=-0.10,shank_d=0.03]
                                                     [--printed 1.01,1.005] [--width 4032 --height 3024]
    python tools/outline_check/outline_check.py render mesh OUT --mesh GOLDEN.ply
                                                     [--turned-errors head_top=0.05,end=-0.10,radial=0.03]

calibrate: 8-15 photos of the sheet alone (no part), tilted 15-40 degrees in different directions, from about the
           distance you photograph the parts, same camera, lens (1x) and resolution. Writes the camera file.
check:     GOLDEN is the golden model (STEP / IGES / STL / PLY / OBJ). PHOTOS are photos of the part lying in the
           blank middle of the sheet, backlit on a light pad, roughly straight down (files or folders). Writes
           report.json and one overlay picture per photo into --out, and prints the measurements. A threaded
           golden model's thread is also checked groove by groove (pitch and lead per tooth, crest, root and groove
           depth): the per-groove table is printed and written to report.json, with a thread close-up
           (NAME-thread.png) and a chart (NAME-thread-chart.png); --set thread=off skips it. The teeth show on the
           outline only when the screw lies across the view (its axis square to the line of sight).
render:    synthetic photos for testing: the golden model (golden.ply), a calibration set (calibration/), photos of
           the part with the given size errors (photos/) and the truth (truth.json). Parts: bolt, plate, screw
           (procedural), or mesh: any golden mesh; a round part (a screw) can get errors made by moving its
           vertices: head_top (its head end out along the axis), end (its other end), radial (below the head).

Photos: HEIC, JPEG, PNG or TIFF, used in the sensor's pixel layout (the phone's orientation does not matter).
Run from the repository root with CloudClean's venv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloudclean import outline_camera as oc  # noqa: E402
from cloudclean import outline_check as ock  # noqa: E402
from cloudclean.io import IMAGE_EXTS  # noqa: E402


def photo_list(items: list[str]) -> list[Path]:
    """Photos from files and folders (sorted by name)."""
    out: list[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.exists():
            out.append(p)
        else:
            raise SystemExit(f"No such photo or folder: {p}")
    if not out:
        raise SystemExit("No photos given")
    return out


# --------------------------------------------------------------------------- calibrate
def cmd_calibrate(args) -> int:
    photos = photo_list(args.photos)
    print(f"Calibrating from {len(photos)} photos of the sheet")
    cam = oc.calibrate(photos, args.bar_mm, args.height_mm)
    cam.save(args.out)
    fs = cam.f_rel_sigma()
    print(f"Camera: {cam.size[0]} x {cam.size[1]} px, f {cam.f:.2f} px (±{100 * (fs or 0):.3f} %), "
          f"reprojection {cam.rms_px:.3f} px, key {cam.key}")
    for v in cam.info["views"]:
        print(f"  {v['photo']}: {v['markers']} markers, {v['rms_px']:.3f} px, tilted {v['tilt_deg']:.0f}°, "
              f"{v['distance_mm']:.0f} mm away")
    print(f"Wrote {args.out}")
    return 0


# --------------------------------------------------------------------------- check
def cmd_check(args) -> int:
    if bool(args.camera) == bool(args.metroy_camparam):
        raise SystemExit("Give the camera: --camera camera.json (from calibrate) or --metroy-camparam camparam.yaml")
    camera = oc.Camera.load(args.camera) if args.camera else oc.from_metroy(args.metroy_camparam, args.metroy_camera)
    photos = photo_list(args.photos)
    params = ock.OutlineParams(tolerance=args.tolerance, paper=args.paper, bar_mm=args.bar_mm or 0.0,
                               height_mm=args.height_mm or 0.0, bar_u=args.bar_u,
                               edge_correction=args.edge_correction, mc_samples=args.mc_samples)
    for kv in args.set or []:
        params.update({kv.split("=", 1)[0]: kv.split("=", 1)[1]})
    out = Path(args.out)
    started = time.time()
    golden = ock.GoldenPart(ock.load_golden(args.golden), max_poses=params.max_poses)
    result = ock.check_photos(golden, photos, camera, params, out_dir=out)
    print()
    print(f"{'measurement':58s} {'golden':>9s} {'photo':>9s} {'diff':>8s} {'U (k=2)':>8s}  result")
    for m in result["combined"]:
        if m["status"] == "not_measured":
            print(f"{m['name'][:58]:58s} {m['golden']:9.3f} {'':>9s} {'':>8s} {'':>8s}  not measured: "
                  f"{m.get('reason', '')[29:]}")
        else:
            print(f"{m['name'][:58]:58s} {m['golden']:9.3f} {m['photo']:9.4f} {m['difference']:+8.4f} "
                  f"{m['uncertainty']:8.4f}  {m['status']}" + (f" ({m['where']})" if m.get("where") else ""))
    print(result["headline"])
    for w in result["warnings"]:
        print("WARNING: " + w)
    print(f"Report: {out / 'report.json'} ({time.time() - started:.0f} s)")
    return 0


# --------------------------------------------------------------------------- render
def parse_errors(text: str | None) -> dict:
    out = {}
    for kv in (text or "").split(","):
        if kv.strip():
            k, v = kv.split("=")
            out[k.strip()] = float(v)
    return out


def mesh_expected(golden, errs: dict) -> list[dict]:
    """The true values of a round part made with turned_errors: {kind, golden, true} for each value its plan has."""
    g = ock.GoldenPart(golden, log=lambda m: None)
    ht, end, rad = errs.get("head_top", 0.0), errs.get("end", 0.0), errs.get("radial", 0.0)
    out = []
    for m in ock.plan_for(g):
        delta = {"Overall length": ht + end, "Head height": ht, "Length under head": end,
                 "Thread major diameter": 2 * rad, "Thread minor diameter": 2 * rad, "Shank diameter": 2 * rad,
                 "Thread pitch": 0.0}.get(m["name"], 0.0 if m["name"].startswith("Head diameter") else None)
        if delta is not None:
            out.append({"kind": m["kind"], "golden": round(m["golden"], 6), "true": round(m["golden"] + delta, 6),
                        "name": m["name"]})
    return out


def cmd_render(args) -> int:
    from cloudclean import outline_synthetic as syn
    from cloudclean import scale_sheet

    out = Path(args.out)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    (out / "photos").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    import open3d as o3d

    if args.part == "mesh":
        if not args.mesh:
            raise SystemExit("render mesh needs --mesh GOLDEN")
        golden = ock.load_golden(args.mesh)
        errs = parse_errors(args.turned_errors)
        truth_mesh = syn.turned_errors(golden, **errs)[0] if errs else golden
        golden_p, true_p = {"mesh": str(args.mesh)}, {"turned_errors": errs}
        expected = mesh_expected(golden, errs)
        down = tuple(float(v) for v in args.down.split(","))
    else:
        golden_p, true_p = syn.part_params(args.part, parse_errors(args.errors))
        fine = {"bolt": (180, 720), "plate": (256, 1024), "screw": (360, 360)}[args.part]
        golden = syn.make_part(args.part, golden_p, fine[0])
        truth_mesh = syn.make_part(args.part, true_p, fine[1])
        expected = syn.expected(args.part, golden_p, true_p)
        down = syn.PARTS[args.part][2]
    o3d.io.write_triangle_mesh(str(out / "golden.ply"), golden)
    x, y, yaw = (float(v) for v in args.pose.split(","))
    part = syn.place(truth_mesh, down, x, y, yaw)
    printed = tuple(float(v) for v in args.printed.split(",")) if "," in args.printed else (float(args.printed),) * 2
    dist = tuple(float(v) for v in args.dist.split(","))
    cam = syn.phone_camera(args.width, args.height, args.hfov, dist, (12.0, -8.0))
    meta = syn.camera_meta(cam)
    truth = {"part": args.part, "golden_params": golden_p, "true_params": true_p,
             "expected": expected, "printed": list(printed),
             "bar_mm": round(scale_sheet.BAR_MM * printed[0], 4), "height_mm": round(120.0 * printed[1], 4),
             "pose": [x, y, yaw], "camera": cam.to_json(), "calibration": [], "photos": []}
    started = time.time()

    def in_view(R, t) -> int:
        n = 0
        for c in scale_sheet.marker_corners().values():
            uv = cam.project((c * [printed[0], printed[1], 1]) @ R.T + t)
            n += bool(np.all((uv > 20) & (uv < np.array(cam.size) - 20)))
        return n

    k = 0
    while k < args.calibration:
        tilt = rng.uniform(15, 40)
        az = 360.0 * k / max(args.calibration, 1) + rng.uniform(-15, 15)
        R, t = syn.view(rng.uniform(260, 320), tilt, az, rng.uniform(-10, 10),
                        (rng.uniform(-10, 10), rng.uniform(-10, 10), 0))
        if in_view(R, t) < 10:
            continue
        img = syn.render(cam, R, t, None, args.paper, printed, args.exposure, blur_px=args.blur, ss=args.ss,
                         seed=int(rng.integers(1 << 30)))
        name = f"cal_{k:02d}.{args.format}"
        syn.save_photo(out / "calibration" / name, img, meta=meta)
        truth["calibration"].append({"image": name, "R": R.tolist(), "t": t.tolist(), "tilt_deg": tilt})
        print(f"calibration/{name} ({time.time() - started:.0f} s)", flush=True)
        k += 1
    for k in range(args.photos):
        tilt = args.tilt if args.tilt else rng.uniform(0, 3)
        R, t = syn.view(args.distance + rng.uniform(-10, 10), tilt, rng.uniform(0, 360), rng.uniform(-5, 5),
                        (x + rng.uniform(-8, 8), y + rng.uniform(-8, 8), 0))
        img = syn.render(cam, R, t, part, args.paper, printed, args.exposure, part_level=args.part_level,
                         blur_px=args.blur, ss=args.ss, seed=int(rng.integers(1 << 30)))
        name = f"photo_{k:02d}.{args.format}"
        syn.save_photo(out / "photos" / name, img, meta=meta)
        truth["photos"].append({"image": name, "R": R.tolist(), "t": t.tolist(), "tilt_deg": tilt})
        print(f"photos/{name} ({time.time() - started:.0f} s)", flush=True)
    (out / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(f"Wrote {out} (bar {truth['bar_mm']} mm, height {truth['height_mm']} mm)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate", help="calibrate a camera from photos of the sheet alone")
    p.add_argument("photos", nargs="+")
    p.add_argument("--out", required=True, help="camera file to write (JSON)")
    p.add_argument("--bar-mm", type=float, default=None, help="the sheet's 100 mm bar as the caliper reads it")
    p.add_argument("--height-mm", type=float, default=None,
                   help="top edge of marker 15 to bottom edge of marker 13 (left column), 120 mm drawn")
    p = sub.add_parser("check", help="check a part against its golden model")
    p.add_argument("golden")
    p.add_argument("photos", nargs="+")
    p.add_argument("--camera", help="camera file from calibrate")
    p.add_argument("--metroy-camparam", help="the MetroY's camparam.yaml (its left camera, raw images)")
    p.add_argument("--metroy-camera", default="L", choices=["L", "R"])
    p.add_argument("--paper", default="a4", choices=["a4", "letter"])
    p.add_argument("--bar-mm", type=float, default=None, help="the sheet's 100 mm bar as the caliper reads it")
    p.add_argument("--height-mm", type=float, default=None,
                   help="top edge of marker 15 to bottom edge of marker 13 (left column), 120 mm drawn")
    p.add_argument("--bar-u", type=float, default=0.05, help="± limits of each caliper reading (mm)")
    p.add_argument("--tolerance", type=float, default=0.1, help="± tolerance (mm)")
    p.add_argument("--edge-correction", default="markers", choices=["markers", "none"])
    p.add_argument("--mc-samples", type=int, default=32)
    p.add_argument("--set", action="append", help="any other OutlineParams field, key=value")
    p.add_argument("--out", default="outline-check")
    p = sub.add_parser("render", help="synthetic test photos")
    p.add_argument("part", choices=["bolt", "plate", "screw", "mesh"])
    p.add_argument("out")
    p.add_argument("--mesh", help="part mesh: the golden model file (render mesh)")
    p.add_argument("--turned-errors", default="", help="errors of a round mesh part: head_top=,end=,radial= (mm)")
    p.add_argument("--down", default="1,0,0", help="which way the mesh part lies (a direction in its frame)")
    p.add_argument("--calibration", type=int, default=10, help="sheet-only calibration photos")
    p.add_argument("--photos", type=int, default=1, help="photos of the part")
    p.add_argument("--tilt", type=float, default=0.0, help="camera tilt of the part photos (0: 0-3 degrees)")
    p.add_argument("--errors", default="", help="size errors of the part, e.g. head_h=0.05,under=-0.1,shank_d=0.03")
    p.add_argument("--pose", default="4,-6,23", help="where the part lies: x,y,turn (mm, mm, degrees)")
    p.add_argument("--printed", default="1.0", help="the printer's scale: s or sx,sy")
    p.add_argument("--paper", default="a4", choices=["a4", "letter"])
    p.add_argument("--width", type=int, default=4032)
    p.add_argument("--height", type=int, default=3024)
    p.add_argument("--hfov", type=float, default=69.0)
    p.add_argument("--dist", default="0.02,-0.03,0.0002,-0.0001,0.01", help="k1,k2,p1,p2,k3")
    p.add_argument("--distance", type=float, default=280.0)
    p.add_argument("--exposure", type=float, default=0.78, help="paper brightness (1 = full scale; more clips)")
    p.add_argument("--part-level", type=float, default=0.02, help="the part's brightness relative to the paper")
    p.add_argument("--blur", type=float, default=0.8, help="optical blur (Gaussian sigma, px)")
    p.add_argument("--ss", type=int, default=4, help="samples per pixel along each side (anti-aliasing)")
    p.add_argument("--format", default="jpg", choices=["jpg", "png", "heic"])
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return {"calibrate": cmd_calibrate, "check": cmd_check, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
