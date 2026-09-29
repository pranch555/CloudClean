"""Stress benchmark: large synthetic scans through every CloudClean stage, with timings, peak memory and accuracy.

    python tools/benchmark.py --points 3000000 --out bench.json

The part has exactly known geometry (tests/synthetic.py), so every stage also reports how accurate it stayed.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

RESULTS: list[dict] = []


def peak_rss_gb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024 ** 3) if platform.system() == "Darwin" else kb / (1024 ** 2)


def stage(name: str, fn, **info):
    print(f"-> {name} ...", flush=True)
    t = time.perf_counter()
    try:
        result = fn()
        ok, err = True, None
    except Exception as exc:  # keep benchmarking the remaining stages
        result, ok, err = None, False, f"{type(exc).__name__}: {exc}"
    dt = time.perf_counter() - t
    row = {"stage": name, "seconds": round(dt, 2), "ok": ok, "peak_rss_gb": round(peak_rss_gb(), 2), **info}
    if err:
        row["error"] = err
    RESULTS.append(row)
    print(f"   {'ok' if ok else 'FAILED'} {dt:8.2f}s  peak {row['peak_rss_gb']:.1f} GB {err or ''}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=3_000_000, help="points per synthetic scan")
    ap.add_argument("--thread-points", type=int, default=2_000_000)
    ap.add_argument("--depth", type=int, default=11, help="Poisson depth for the big mesh")
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()
    quiet = lambda m: None  # noqa: E731

    from cloudclean.analysis import thread_analysis
    from cloudclean.clean import CleanParams, clean_point_cloud
    from cloudclean.compare import CompareParams, compare_to_reference
    from cloudclean.edit import apply_edits
    from cloudclean.io import describe, load, save
    from cloudclean.mesh import MeshParams, poisson_threads, reconstruct_mesh
    from cloudclean.register import MergeParams, assess_merge, merge_geometries
    from tests.synthetic import ground_truth_mesh, random_rigid, simulate_scan

    gt = ground_truth_mesh()
    gt_dims = np.asarray(describe(gt)["dimensions"])
    print(f"platform {platform.machine()} · open3d {o3d.__version__} · poisson threads {poisson_threads()}")

    a = stage("simulate scan A", lambda: simulate_scan([0.2, 0.1, 1.0], seed=1, n_points=args.points, add_table=True)[0], points=args.points)
    b = stage("simulate scan B (flipped, random pose)", lambda: simulate_scan([-0.1, -0.2, -1.0], seed=2, n_points=args.points, transform=random_rigid(7))[0], points=args.points)
    tmp = Path(args.out).with_suffix("")
    tmp.mkdir(exist_ok=True)
    stage("save + load PLY (A)", lambda: load(save(a, tmp / "a.ply")))

    ca = stage("clean A (remove table)", lambda: clean_point_cloud(a, CleanParams(remove_plane=True), quiet)[0])
    cb = stage("clean B", lambda: clean_point_cloud(b, CleanParams(), quiet)[0])
    assessment = stage("merge check (assess_merge)", lambda: assess_merge([ca, cb], MergeParams(), quiet))
    if assessment:
        RESULTS[-1]["recommendation"] = assessment["recommendation"]
    merged = stage("merge with approved transforms", lambda: merge_geometries([ca, cb], MergeParams(), None, quiet,
                                                                                 transforms=assessment["transforms"])[0])
    if merged is not None:
        RESULTS[-1]["points_out"] = len(merged.points)

    mesh = stage(f"Poisson mesh depth {args.depth}", lambda: reconstruct_mesh(merged, MeshParams(depth=args.depth), quiet))
    if mesh:
        mesh, rep = mesh
        dims = np.asarray(rep["mesh"]["dimensions"])
        RESULTS[-1].update(triangles=len(mesh.triangles), deviation_p95=round(rep["deviation"]["p95"], 4),
                           dim_error_max_mm=round(float(np.abs(dims - gt_dims).max()), 4))
        stage("denoise whole mesh (5 passes)", lambda: apply_edits(mesh, [{"op": "denoise", "iterations": 5}], quiet))
        c = np.asarray(mesh.vertices)[len(mesh.vertices) // 2]  # a point on the surface (the centre is inside the part)
        stage("denoise brush region (r = 10 mm)", lambda: apply_edits(mesh, [{"op": "denoise", "iterations": 5,
                                                                              "region": {"spheres": [[*c, 10.0]]}}], quiet))

    cmp = stage("compare merged scan vs CAD", lambda: compare_to_reference(merged, gt, CompareParams(tolerance=0.1), quiet))
    if cmp:
        st = cmp["report"]["stats"]
        RESULTS[-1].update(within_tolerance_pct=round(st["within_tolerance_pct"], 2), mean=round(st["mean"], 4), rms=round(st["rms"], 4))

    # threads
    from tests.test_measure import scan_of, thread_mesh

    screw = stage("simulate M12x1.75 thread scan", lambda: scan_of(thread_mesh(12.0, 1.75, length=30.0), seed=5,
                                                                  spacing=float(np.sqrt(np.pi * 12 * 30 / args.thread_points)))[0])
    if screw is not None:
        RESULTS[-1]["points"] = len(screw.points)
        th = stage("thread analysis", lambda: thread_analysis(screw, None, quiet))
        if th:
            RESULTS[-1].update(pitch=round(th["pitch"], 5), pitch_error=round(th["pitch"] - 1.75, 5),
                               major_error=round(th["major_diameter"] - 12.0, 4), crests=th["crest_count"])

    report = {"platform": platform.platform(), "machine": platform.machine(), "open3d": o3d.__version__,
              "points_per_scan": args.points, "stages": RESULTS}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n" + "\n".join(f"{r['stage']:<42} {r['seconds']:>8.1f}s  {'ok' if r['ok'] else 'FAIL'}" for r in RESULTS))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
