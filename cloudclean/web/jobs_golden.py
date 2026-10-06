"""Job body for the golden model check (cloudclean/golden.py, docs/golden-model.md)."""
from __future__ import annotations

import numpy as np

from ..golden import GoldenParams, check_against_golden
from ..io import is_cloud
from .workspace import Workspace

# Engine log lines that mark the start of a stage -> (progress fraction, label)
_STAGES = [("Comparing", 0.08, "lining up"), ("  global feature", 0.15, "trying poses"),
           ("    candidate", 0.3, "refining the line-up"), ("  aligned:", 0.55, "measuring the deviation"),
           ("  scale estimate", 0.62, "coverage"), ("Golden check:", 0.74, "checking the surface")]


def job_golden_check(ws: Workspace, payload: dict, log) -> list[str]:
    progress = getattr(log, "progress", None) or (lambda fraction, label=None: None)
    scan_id, golden_id = payload["scan_id"], payload["golden_id"]
    scan_meta, golden_meta = ws.get(scan_id), ws.get(golden_id)
    params = GoldenParams.from_dict(payload.get("params", {}))

    progress(0.02, "loading")
    scan = ws.load_geometry(scan_id)
    golden = ws.load_geometry(golden_id)
    if is_cloud(golden):
        raise ValueError(f"'{golden_meta['name']}' is a point cloud. The golden model must be a CAD model or a "
                         "mesh: import the CAD file, or make a mesh of it first (Mesh step).")
    # points filled in from photos are only good to a millimetre or two: never let them pass (or fail) the check
    from_photos = 0
    try:
        source = np.load(ws.scalars_path(scan_id, "source", preview=False))
    except (KeyError, OSError, ValueError):
        source = None
    if source is not None and is_cloud(scan) and len(source) == len(scan.points) and (source > 0.5).any():
        keep = np.flatnonzero(source < 0.5)
        from_photos = len(source) - len(keep)
        scan = scan.select_by_index(keep)
        log(f"Leaving out {from_photos:,} points filled in from photos")

    def staged_log(msg: str) -> None:
        for prefix, fraction, label in _STAGES:
            if msg.startswith(prefix):
                progress(fraction, label)
                break
        log(msg)

    result = check_against_golden(scan, golden, params, staged_log, progress,
                                  up_axis=payload.get("up_axis") or "y")
    report, creport = result["report"], result["compare_report"]
    if from_photos:
        report["photo_points_left_out"] = from_photos
        report["summary"].append(f"{from_photos:,} points of this scan were filled in from photos. Photos are only "
                                 "good to a millimetre or two, so the check leaves them out: those areas show as "
                                 "not scanned.")
    scan_ref = {"id": scan_id, "name": scan_meta["name"]}
    golden_ref = {"id": golden_id, "name": golden_meta["name"]}
    creport.update(scan_asset=scan_ref, reference_asset=golden_ref)
    report.update(scan_asset=scan_ref, golden_asset=golden_ref)

    progress(0.95, "saving")
    # the scan lined up with the golden model, coloured by deviation (the same result as Measure -> Compare)
    name = payload.get("name") or f"{scan_meta['name']} vs {golden_meta['name']}"
    compared = ws.add_geometry(result["aligned"], name, "compare", [scan_id, golden_id], params.to_dict(), creport)
    ws.add_scalars(compared["id"], "deviation", result["deviation"], unit="mm",
                   description="Signed distance to the golden surface (+ = material outside, NaN = excluded)")
    # the golden model coloured by what the check found (colours baked in; the numbers as scalars)
    report["compare_asset"] = {"id": compared["id"], "name": name}
    check = ws.add_geometry(result["check_mesh"], f"{golden_meta['name']} · check of {scan_meta['name']}",
                            "golden_check", [scan_id, golden_id], params.to_dict(), report)
    ws.add_scalars(check["id"], "golden_deviation", result["vertex_deviation"], unit="mm",
                   description="How far the scanned surface sits from the golden surface here (+ = outside, "
                               "NaN = not judged: not scanned or too few points)")
    ws.add_scalars(check["id"], "check_status", result["vertex_status"], unit="",
                   description="0 matches, 1 not scanned, 2 too few points, 3 rough, 4 off outside, 5 off inside")
    ws.add_scalars(check["id"], "check_region", result["vertex_region"], unit="",
                   description="The listed problem region this point belongs to (-1: none)")
    ws.add_scalars(check["id"], "check_face", result["vertex_face"], unit="",
                   description="The golden face (flat or round) this point belongs to (-1: none)")

    counts = report["counts"]
    if hasattr(log, "output"):
        log.output(check_id=check["id"], compare_id=compared["id"], verdict=report["verdict"],
                   headline=report["headline"], scanned_pct=report["surface"]["scanned_pct"],
                   rescan=len(report["rescan"]), measurements_off=counts["off"],
                   measurements_ok=counts["ok"], not_measured=counts["not_measured"])
    progress(1.0, "done")
    return [check["id"], compared["id"]]


JOBS = {"golden_check": job_golden_check}
