"""Job body for scan-vs-CAD inspection."""
from __future__ import annotations

from ..compare import CompareParams, compare_to_reference
from ..io import is_cloud
from .workspace import Workspace

# Engine log lines that mark the start of a stage -> (progress fraction, label)
_STAGES = [("Comparing", 0.12, "aligning"), ("  global feature", 0.2, "global search"),
           ("    candidate", 0.35, "refining candidates"), ("  aligned:", 0.6, "measuring deviation"),
           ("  scale estimate", 0.68, "coverage"), ("Deviation:", 0.75, "saving results")]


def job_compare(ws: Workspace, payload: dict, log) -> list[str]:
    progress = getattr(log, "progress", None) or (lambda fraction, label=None: None)
    scan_id, ref_id = payload["scan_id"], payload["reference_id"]
    scan_meta, ref_meta = ws.get(scan_id), ws.get(ref_id)
    params = CompareParams.from_dict(payload.get("params", {}))

    progress(0.02, "loading")
    scan = ws.load_geometry(scan_id)
    reference = ws.load_geometry(ref_id)
    if is_cloud(reference):
        raise ValueError(f"'{ref_meta['name']}' is a point cloud - the reference must be a CAD model or mesh")

    def staged_log(msg: str) -> None:
        for prefix, fraction, label in _STAGES:
            if msg.startswith(prefix):
                progress(fraction, label)
                break
        log(msg)

    result = compare_to_reference(scan, reference, params, staged_log)
    report = result["report"]
    report["scan_asset"] = {"id": scan_id, "name": scan_meta["name"]}
    report["reference_asset"] = {"id": ref_id, "name": ref_meta["name"]}

    progress(0.8, "saving compared scan")
    name = payload.get("name") or f"{scan_meta['name']} vs {ref_meta['name']}"
    compared = ws.add_geometry(result["aligned"], name, "compare", [scan_id, ref_id], params.to_dict(), report)
    ws.add_scalars(compared["id"], "deviation", result["deviation"], unit="mm",
                   description="Signed distance to the reference surface (+ = material outside, NaN = excluded)")

    progress(0.92, "saving coverage")
    coverage_report = {k: report[k] for k in ("coverage", "tolerance", "transform", "scan_asset", "reference_asset")}
    coverage_report["compare_asset"] = {"id": compared["id"], "name": name}
    covered = ws.add_geometry(result["reference"], f"{ref_meta['name']} · coverage", "compare", [scan_id, ref_id],
                              params.to_dict(), coverage_report)
    ws.add_scalars(covered["id"], "reference_distance", result["reference_distance"], unit="mm",
                   description="Distance from the reference surface to the nearest scan point (coverage)")

    stats = report["stats"]
    if hasattr(log, "output"):
        log.output(compare_id=compared["id"], coverage_id=covered["id"],
                   within_tolerance_pct=round(stats["within_tolerance_pct"], 2), rms=stats["rms"],
                   covered_area_pct=round(report["coverage"]["covered_area_pct"], 2))
    progress(1.0, "done")
    return [compared["id"], covered["id"]]


JOBS = {"compare": job_compare}
