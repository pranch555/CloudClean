"""Job `autopilot`: import -> clean -> check merge -> merge (>= 2, when it helps) -> mesh -> compare (optional) ->
export + report.json.

Merging is decided by `settings.merge_mode` (see `autopilot.MERGE_MODES`) or, when the user already decided, by
`payload.merge_decision` ("merge" | "best" | "separate", with `transforms` / `best_index` from the assessment).
When a decision is needed the job stops after cleaning (no mesh, no export) with
`log.output(decision_needed=True, assessment=..., cleaned_ids=[...])`; `POST /api/autopilot/decide` continues."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..autopilot import AutopilotSettings, collect_warnings, safe_item_name, unique_folder
from ..io import save
from ..register import assessment_summary
from .jobs import job_clean, job_import, job_merge, job_mesh
from .jobs_merge import run_assessment
from .settings import Settings
from .workspace import Workspace

MERGE_DECISIONS = ("merge", "best", "separate")


def _progress(log, fraction: float, label: str) -> None:
    progress = getattr(log, "progress", None)
    if progress:
        progress(fraction, label)


def load_settings(ws: Workspace, overrides: dict | None = None) -> AutopilotSettings:
    """Saved `autopilot` settings, optionally overridden, validated."""
    s = AutopilotSettings.load(Settings(ws.root).get("autopilot", AutopilotSettings().to_dict()))
    s.update(overrides or {})
    return s.validate()


def export_root(ws: Workspace, settings: AutopilotSettings) -> Path:
    return Path(settings.output_folder) if settings.output_folder else ws.root / "exports"


def _summary(meta: dict) -> dict:
    s = meta.get("stats", {})
    return {"id": meta["id"], "name": meta["name"], "kind": meta["kind"],
            "points": s.get("points", s.get("vertices")), "dimensions": s.get("dimensions"), "spacing": s.get("spacing")}


def job_autopilot(ws: Workspace, payload: dict, log) -> list[str]:
    settings = load_settings(ws, payload.get("settings"))
    sources = payload.get("sources") or {}  # original filenames of temporary uploads
    paths = [str(p) for p in payload.get("paths") or []]
    asset_ids = list(payload.get("asset_ids") or [])
    if not paths and not asset_ids:
        raise ValueError("Autopilot needs at least one scan (asset_ids or paths)")
    decision = payload.get("merge_decision")
    if decision is not None and decision not in MERGE_DECISIONS:
        raise ValueError(f"Unknown merge decision '{decision}'. Choose from: {', '.join(MERGE_DECISIONS)}")
    name = (payload.get("name") or "").strip() or "item"
    temporary = bool(payload.get("temporary"))
    created: list[str] = []
    report: dict = {"item": name, "started": datetime.now().isoformat(timespec="seconds"),
                    "settings": settings.to_dict(), "stages": {}}
    warnings: list[str] = []
    notes: list[str] = []
    output = getattr(log, "output", None)

    def warn(msg: str) -> None:
        warnings.append(msg)
        log(f"WARNING: {msg}")

    def note(msg: str) -> None:
        notes.append(msg)
        log(f"NOTE: {msg}")

    log(f"Autopilot '{name}': {len(paths)} file(s), {len(asset_ids)} existing asset(s)")
    # ------------------------------------------------------------------ import
    try:
        for i, path in enumerate(paths):
            _progress(log, 0.15 * i / max(len(paths), 1), f"Importing {Path(path).name}")
            source = sources.get(path, path)
            new = job_import(ws, {"path": path, "name": Path(source).stem, "source": source,
                                  "temporary": temporary, "project_id": payload.get("project_id")}, log)
            created += new
            asset_ids += new
    finally:
        if temporary:  # job_import removes each file it reached; also remove the rest after a failure
            for path in paths:
                Path(path).unlink(missing_ok=True)

    metas = [ws.get(i) for i in asset_ids]
    skipped = [m["name"] for m in metas if m["kind"] == "image"]
    if skipped:
        log(f"Ignoring photos: {', '.join(skipped)}")
    metas = [m for m in metas if m["kind"] != "image"]
    if not metas:
        raise ValueError("None of the inputs is a scan (only photos were given)")
    report["stages"]["import"] = {"inputs": [_summary(m) for m in metas]}

    # ------------------------------------------------------------------ clean
    if payload.get("skip_clean"):
        ids = [m["id"] for m in metas]
        report["stages"]["clean"] = {"skipped": "inputs are already cleaned"}
    else:
        ids, clean_reports = [], []
        for i, m in enumerate(metas):
            _progress(log, 0.15 + 0.2 * i / len(metas), f"Cleaning {m['name']}")
            if m["kind"] != "pointcloud":
                log(f"'{m['name']}' is a mesh - not cleaned")
                ids.append(m["id"])
                continue
            new = job_clean(ws, {"asset_ids": [m["id"]], "preset": settings.preset,
                                 "params": {"remove_plane": settings.remove_plane}}, log)
            created += new
            ids += new
            rep = ws.report(new[0]) or {}
            clean_reports.append({"input": m["name"], "asset_id": new[0],
                                  **{k: v for k, v in rep.items() if k != "params"}})
        report["stages"]["clean"] = {"preset": settings.preset, "remove_plane": settings.remove_plane,
                                     "scans": clean_reports}

    # ------------------------------------------------------------------ merge decision
    names = [m["name"] for m in metas]
    targets = [(ids[0], name, None)]      # (asset to mesh, mesh name, export stem suffix)
    if len(ids) >= 2:
        transforms, best_index = payload.get("transforms"), payload.get("best_index")
        checked = None  # assessment whose poses the merge uses (kept on the merged asset for measurement caveats)
        if decision:
            choice = decision
            report["stages"]["decision"] = {"decision": decision, "decided_for": payload.get("decision_for")}
            log(f"Merge decision by the user: {decision}")
        elif settings.merge_mode == "always":
            choice, transforms = "merge", None
        elif settings.merge_mode == "never":
            choice = "separate"
        else:
            _progress(log, 0.35, "Checking whether to merge")
            assessment = run_assessment(ws, ids, {"method": settings.merge_method}, log)
            summary = assessment_summary(assessment, names)
            report["stages"]["assessment"] = summary
            for reason in assessment["reasons"]:
                log(f"  {reason}")
            rec = assessment["recommendation"]
            if settings.merge_mode == "ask" or rec == "ask":
                assessment["names"] = names
                log("Waiting for a decision: merge, use the best scan, keep the scans separate or cancel "
                    "(nothing was meshed or exported yet)")
                if output:
                    output(decision_needed=True, assessment=assessment, cleaned_ids=ids, item=name, warnings=warnings)
                _progress(log, 1.0, "Needs a decision")
                return created
            choice = "merge" if rec == "merge" else "best"
            transforms, best_index = assessment["transforms"], assessment["best_index"]
            checked = assessment
        if choice == "merge":
            _progress(log, 0.4, f"Merging {len(ids)} scans")
            merged = job_merge(ws, {"asset_ids": ids, "params": {"method": settings.merge_method},
                                    "name": f"{name} · merged", "transforms": transforms, "assessment": checked}, log)
            created += merged
            merge_rep = ws.report(merged[0]) or {}
            for w in collect_warnings(merge_rep, names=names):
                warn(w)
            report["stages"]["merge"] = {
                "asset_id": merged[0], "method": settings.merge_method, "approved_transforms": transforms is not None,
                "output_points": merge_rep.get("output_points"), "combined_points": merge_rep.get("combined_points"),
                "spacing": merge_rep.get("spacing"),
                "scans": [{"name": names[s.get("index", 0)] if s.get("index", 0) < len(names) else None,
                           **{k: s.get(k) for k in ("index", "reference", "points", "fitness", "rmse", "best_candidate",
                                                    "ambiguous", "warning", "transform") if k in s}}
                          for s in merge_rep.get("scans", [])]}
            targets = [(merged[0], name, None)]
        elif choice == "best":
            best = int(best_index or 0)
            if not 0 <= best < len(ids):
                raise ValueError(f"best_index {best} is out of range (0..{len(ids) - 1})")
            note(f"The scans were not merged: they cover the same surface, so merging would only add noise. "
                 f"Only '{names[best]}' (the best scan) was meshed.")
            report["stages"]["merge"] = {"skipped": "use_best", "best_index": best, "best_name": names[best],
                                         "asset_id": ids[best]}
            targets = [(ids[best], name, None)]
        else:
            note("The scans were not merged; each scan was meshed and exported separately.")
            report["stages"]["merge"] = {"skipped": "separate"}
            targets = [(ids[k], f"{name} · {names[k]}", names[k]) for k in range(len(ids))]
    else:
        report["stages"]["merge"] = {"skipped": "single scan"}

    # ------------------------------------------------------------------ mesh / compare / export
    folder = unique_folder(export_root(ws, settings), safe_item_name(name))
    exports, meshes = [], []
    for k, (source_id, mesh_name, suffix) in enumerate(targets):
        span = 0.45 / len(targets)
        base = 0.5 + span * k
        stem = safe_item_name(name if suffix is None else f"{name}_{suffix}")
        result = _mesh_compare_export(ws, settings, source_id, mesh_name, stem, folder, log, warn, base, span)
        created += result.pop("created")
        exports += result["exports"]
        meshes.append(result)

    first = meshes[0]
    if len(meshes) == 1:
        report["stages"]["mesh"] = first["mesh_stage"]
        if first["compare_stage"] is not None:
            report["stages"]["compare"] = first["compare_stage"]
        report["final"] = first["final"]
    else:
        report["stages"]["meshes"] = [{"name": m["name"], "exports": m["exports"], **m["mesh_stage"]} for m in meshes]
        compares = [m["compare_stage"] for m in meshes if m["compare_stage"] is not None]
        if compares:
            report["stages"]["compares"] = compares
        report["final"] = [{"name": m["name"], **m["final"]} for m in meshes]
    report_path = folder / "report.json"
    exports_all = exports + [str(report_path)]
    mesh_ids = [m["mesh_id"] for m in meshes]
    compare_id = first["compare_id"]
    report.update(warnings=warnings, notes=notes, exports=exports, mesh_id=mesh_ids[0], mesh_ids=mesh_ids,
                  compare_id=compare_id, finished=datetime.now().isoformat(timespec="seconds"))
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"Report written to {report_path}")
    if warnings:
        log(f"Finished with {len(warnings)} warning(s) - check them before using the result")
    if output:
        output(exports=exports_all, folder=str(folder), mesh_id=mesh_ids[0], mesh_ids=mesh_ids, compare_id=compare_id,
               warnings=warnings, notes=notes)
    _progress(log, 1.0, "Done")
    return created


def _mesh_compare_export(ws: Workspace, settings: AutopilotSettings, source_id: str, mesh_name: str, stem: str,
                         folder: Path, log, warn, base: float, span: float) -> dict:
    created: list[str] = []
    _progress(log, base, f"Meshing {mesh_name}")
    mesh_id = job_mesh(ws, {"asset_id": source_id, "params": settings.mesh}, log)[0]
    created.append(mesh_id)
    ws.update(mesh_id, name=mesh_name)
    mesh_meta, mesh_rep = ws.get(mesh_id), ws.report(mesh_id) or {}
    for w in collect_warnings(mesh_report=mesh_rep):
        warn(w)
    mesh_stage = {"asset_id": mesh_id, "method": mesh_rep.get("method"), "depth": mesh_rep.get("depth"),
                  "spacing": mesh_rep.get("spacing"), "deviation": mesh_rep.get("deviation"),
                  "cloud_dimensions": mesh_rep.get("cloud_dimensions"), "params": mesh_rep.get("params")}

    compare_id, compare_stage = None, None
    if settings.reference_id:
        _progress(log, base + 0.6 * span, "Comparing with CAD")
        compare = None
        try:
            from .jobs_compare import JOBS as COMPARE_JOBS

            compare = COMPARE_JOBS.get("compare")
        except ImportError:
            pass
        if compare is None:
            warn("A CAD reference is set but CAD comparison is not available in this installation - skipped")
            compare_stage = {"skipped": "compare job not available"}
        else:
            try:
                ws.get(settings.reference_id)
            except KeyError:
                raise ValueError(f"CAD reference asset {settings.reference_id} no longer exists - "
                                 "choose another one in the autopilot settings")
            # compare the measured points (merged/cleaned cloud), not the reconstructed surface
            compare_ids = compare(ws, {"scan_id": source_id, "reference_id": settings.reference_id,
                                       "params": settings.compare}, log)
            created += compare_ids
            compare_id = compare_ids[0] if compare_ids else None
            cmp_rep = (ws.report(compare_id) or {}) if compare_id else {}
            for w in collect_warnings(compare_report=cmp_rep):
                warn(w)
            compare_stage = {"asset_id": compare_id, "reference_id": settings.reference_id,
                             **{k: cmp_rep.get(k) for k in ("alignment", "stats", "dimensions", "coverage",
                                                            "scale_estimate", "tolerance") if k in cmp_rep}}

    _progress(log, base + 0.9 * span, "Exporting")
    folder.mkdir(parents=True, exist_ok=True)
    mesh = ws.load_geometry(mesh_id)
    exports = []
    for fmt in settings.formats:
        path = save(mesh, folder / f"{stem}.{fmt}")
        exports.append(str(path))
        log(f"Exported {path}")
    log(f"Final mesh size of '{mesh_name}' (x, y, z): {mesh_meta['stats'].get('dimensions')}")
    return {"created": created, "name": mesh_name, "mesh_id": mesh_id, "compare_id": compare_id,
            "exports": exports, "mesh_stage": mesh_stage, "compare_stage": compare_stage, "final": mesh_meta["stats"]}


JOBS = {"autopilot": job_autopilot}
