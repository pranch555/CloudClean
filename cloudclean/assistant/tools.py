"""Function tools the assistant can call. They use the same workspace / jobs APIs as the web UI.

A tool is a `Tool(name, description, parameters, handler)`. Handlers take `(ctx: ToolContext, args: dict)` and
return a `ToolResult` (or raise `ToolError` with a sentence for the model). Handlers may be plain functions
(run in a worker thread) or coroutines (job tools, which poll the job and stream progress).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

RESULT_LIMIT = 2000  # bytes of JSON handed back to the model per tool call
UNITS = "scan units (millimetres for Revopoint scans)"


class ToolError(Exception):
    """Invalid arguments or a failed operation, phrased for the model."""


class ToolCancelled(Exception):
    """The user stopped the assistant while the tool was running."""


@dataclass
class ToolResult:
    data: Any
    summary: str = ""
    asset_ids: list[str] = field(default_factory=list)
    # images for the model to look at ({url: data URL, caption}); the agent shows them in its next request of the
    # turn (never stored in the conversation)
    images: list[dict] = field(default_factory=list)


@dataclass
class ToolContext:
    workspace: Any
    jobs: Any
    settings: dict
    ui: dict
    emit: Callable[[str, dict], None]
    conversation_id: str
    call_id: str = ""
    stop: threading.Event = field(default_factory=threading.Event)
    poll_interval: float = 0.4

    @property
    def stopped(self) -> bool:
        return self.stop.is_set()


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


# --------------------------------------------------------------------------- formatting helpers
def num(x, digits: int = 4):
    """Round for display without losing metrology-relevant precision."""
    if x is None or isinstance(x, bool) or not isinstance(x, (int, float)):
        return x
    if isinstance(x, int):
        return x
    if x != x or x in (float("inf"), float("-inf")):
        return None
    a = abs(x)
    if a == 0:
        return 0.0
    if a >= 100:
        return round(x, 2)
    if a >= 1:
        return round(x, 3)
    return float(f"{x:.{digits - 1}g}")


def nums(values):
    return [num(v) for v in values] if isinstance(values, (list, tuple)) else values


def dims_text(stats: dict) -> str:
    d = stats.get("dimensions")
    return "x".join(f"{num(v):g}" for v in d) if d else ""


def shrink(obj, limit: int = RESULT_LIMIT) -> str:
    """JSON text of obj, cut down (long lists / strings first) so it stays under `limit` bytes."""
    text = json.dumps(obj, separators=(",", ":"), default=str, ensure_ascii=False)
    if len(text.encode()) <= limit:
        return text

    def cut(o, n_items: int, n_chars: int):
        if isinstance(o, dict):
            return {k: cut(v, n_items, n_chars) for k, v in o.items()}
        if isinstance(o, list):
            head = [cut(v, n_items, n_chars) for v in o[:n_items]]
            return head + [f"... {len(o) - n_items} more"] if len(o) > n_items else head
        if isinstance(o, str) and len(o) > n_chars:
            return o[:n_chars] + "..."
        return o

    for n_items, n_chars in ((20, 400), (10, 200), (5, 120), (3, 80), (2, 60), (1, 40)):
        text = json.dumps(cut(obj, n_items, n_chars), separators=(",", ":"), default=str, ensure_ascii=False)
        if len(text.encode()) <= limit:
            return text
    return text.encode()[: limit - 40].decode("utf-8", "ignore") + " ...(truncated)"


def count_text(meta: dict) -> str:
    s = meta.get("stats", {})
    if meta["kind"] == "pointcloud":
        return f"{s.get('points', 0):,} pts"
    if meta["kind"] == "mesh":
        return f"{s.get('triangles', 0):,} tris"
    return f"{s.get('width', '?')}x{s.get('height', '?')} px"


def prompt_safe(text, limit: int = 80) -> str:
    """Asset names come from uploaded file names: collapse whitespace and cap length before they reach the model,
    so a crafted name cannot fake prompt structure (new lines, table rows, headings)."""
    clean = " ".join(str(text).split()).replace("|", "/")
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def workspace_table(ws, limit: int = 60, kinds=None, query: str = "") -> tuple[str, int]:
    """Compact pipe table of assets (newest last). Returns (text, number of rows omitted)."""
    metas = [m for m in ws.list() if (not kinds or m["kind"] in kinds)
             and (not query or query.lower() in m["name"].lower() or m["id"].startswith(query.lower()))]
    omitted = max(0, len(metas) - limit)
    rows = ["id | name | kind | size | dims | op | parents"]
    for m in metas[omitted:]:
        rows.append(" | ".join([m["id"], prompt_safe(m["name"]), m["kind"], count_text(m), dims_text(m.get("stats", {})),
                                m.get("operation", ""), ",".join(m.get("parents", [])) or "-"]))
    return "\n".join(rows), omitted


def resolve_asset(ws, ref, kinds=None) -> dict:
    """Find an asset by id, unique name or unique id prefix."""
    ref = str(ref or "").strip().strip("'\"")
    if not ref:
        raise ToolError("asset id is required")
    try:
        meta = ws.get(ref)
    except KeyError:
        metas = ws.list()
        matches = [m for m in metas if m["name"].casefold() == ref.casefold()]
        if not matches and len(ref) >= 4:
            matches = [m for m in metas if m["id"].startswith(ref.lower())]
        if not matches:
            raise ToolError(f"No asset '{ref}'. Call list_assets to see the ids.")
        if len(matches) > 1:
            raise ToolError(f"'{ref}' matches several assets: "
                            + ", ".join(f"{m['id']} ({m['name']})" for m in matches[:6]) + ". Use the id.")
        meta = matches[0]
    if kinds and meta["kind"] not in kinds:
        raise ToolError(f"'{meta['name']}' ({meta['id']}) is a {meta['kind']}; this needs a {' or '.join(kinds)}.")
    return meta


def report_summary(meta: dict, report: dict | None) -> dict:
    """Key numbers of an operation report."""
    if not report:
        return {}
    op = meta.get("operation")
    try:
        if op == "clean":
            steps = {s["step"]: s["removed"] for s in report.get("steps", []) if s.get("removed")}
            return {"input_points": report.get("input_points"), "output_points": report.get("output_points"),
                    "removed_pct": num(100 * report.get("removed_fraction", 0)), "spacing": num(report.get("spacing")),
                    "removed_by_step": steps}
        if op == "merge" and "scans" in report:
            scans = []
            for s in report["scans"]:
                item = {"scan": s.get("index")}
                if s.get("reference"):
                    item["reference"] = True
                for k in ("fitness", "rmse", "best_candidate"):
                    if k in s:
                        item[k] = num(s[k])
                if s.get("warning"):
                    item["warning"] = s["warning"]
                if s.get("ambiguous"):
                    item["ambiguous"] = "a rotated pose fits almost as well (symmetric part?)"
                scans.append(item)
            return {"scans": scans, "combined_points": report.get("combined_points"),
                    "output_points": report.get("output_points"), "spacing": num(report.get("spacing"))}
        if op == "mesh" and "deviation" in report:
            m = report.get("mesh", {})
            dev = report["deviation"]
            return {"method": report.get("method"), "depth": report.get("depth"), "trim": report.get("trim"),
                    "deviation_to_scan": {k: num(dev.get(k)) for k in ("mean", "rms", "p95", "max")},
                    "triangles": m.get("triangles"), "watertight": m.get("watertight"),
                    "dimensions": nums(m.get("dimensions")), "spacing": num(report.get("spacing"))}
        if op == "golden_check" and "verdict" in report:
            return golden_brief(report)
        if op == "compare" and "stats" in report:
            st = report["stats"]
            out = {"tolerance": report.get("tolerance"),
                   "stats": {k: num(st.get(k)) for k in ("points", "excluded", "mean", "std", "rms", "min", "max",
                                                         "p05", "p95", "within_tolerance_pct") if k in st}}
            al = report.get("alignment") or {}
            out["alignment"] = {k: num(al.get(k)) for k in ("method", "fitness", "rmse") if k in al}
            dims = report.get("dimensions") or {}
            if dims:
                out["dimensions"] = {k: nums(v) for k, v in dims.items()}
            if report.get("coverage"):
                out["coverage_pct"] = num(report["coverage"].get("covered_area_pct"))
            if report.get("scale_estimate"):
                out["scale_estimate_pct"] = num(report["scale_estimate"].get("percent"))
            return out
        if op == "edit" and "ops" in report:
            return {"ops": [{k: v for k, v in o.items() if k in ("op", "before", "after", "removed", "warning",
                                                                  "filled", "skipped", "area_added", "note")}
                            for o in report["ops"][:12]], "transformed": report.get("transform") is not None}
    except (TypeError, AttributeError, KeyError):
        pass
    return {k: num(v) for k, v in list(report.items())[:15] if isinstance(v, (int, float, str, bool)) or v is None}


def asset_brief(meta: dict) -> dict:
    s = meta.get("stats", {})
    out = {"id": meta["id"], "name": meta["name"], "kind": meta["kind"]}
    if meta["kind"] == "pointcloud":
        out["points"] = s.get("points")
    elif meta["kind"] == "mesh":
        out.update(triangles=s.get("triangles"), watertight=s.get("watertight"))
    if s.get("dimensions"):
        out["dimensions"] = nums(s["dimensions"])
    return out


# --------------------------------------------------------------------------- jobs
async def run_job(ctx: ToolContext, kind: str, title: str, payload: dict) -> dict:
    """Submit a job, stream tool_progress events until it finishes, raise ToolError when it fails."""
    job = ctx.jobs.submit(kind, title, payload)
    last = None
    while True:
        cur = ctx.jobs.get(job["id"])
        logs = cur.get("logs") or []
        prog = cur.get("progress") or {}
        signature = (cur["status"], prog.get("fraction"), prog.get("label"), len(logs))
        if signature != last:
            last = signature
            ctx.emit("tool_progress", {"call_id": ctx.call_id, "job_id": job["id"], "status": cur["status"],
                                       "progress": prog.get("fraction"), "label": prog.get("label"),
                                       "last_log": logs[-1] if logs else None})
        if cur["status"] in ("done", "failed", "cancelled"):
            break
        if ctx.stopped:
            try:
                ctx.jobs.cancel(job["id"])
            except KeyError:
                pass
            raise ToolCancelled(f"Stopped by the user; job {job['id']} was cancelled")
        await asyncio.sleep(ctx.poll_interval)
    if cur["status"] == "failed":
        tail = " / ".join(line.split("] ", 1)[-1] for line in (cur.get("logs") or [])[-4:]
                          if "Traceback" not in line and not line.strip().startswith("File "))
        raise ToolError(f"{title} failed: {cur.get('error')}" + (f" (last log: {tail[-500:]})" if tail else ""))
    if cur["status"] == "cancelled":
        raise ToolError(f"{title} was cancelled")
    return cur


def created_summary(ws, ids: list[str]) -> list[dict]:
    out = []
    for asset_id in ids:
        try:
            meta = ws.get(asset_id)
        except KeyError:
            continue
        out.append({**asset_brief(meta), "operation": meta.get("operation"),
                    "report": report_summary(meta, ws.report(asset_id))})
    return out


def _job_result(ctx: ToolContext, job: dict, verb: str, hint: str = "") -> ToolResult:
    ids = list(job.get("result") or [])
    created = created_summary(ctx.workspace, ids)
    data = {"ok": True, "job_id": job["id"], "created": created}
    if job.get("output"):
        data["output"] = job["output"]
    if hint:
        data["hint"] = hint
    names = ", ".join(f"{c['name']}" for c in created) or "nothing"
    return ToolResult(data, f"{verb}: {names}", ids)


def _names(metas) -> str:
    n = [m["name"] for m in metas]
    return ", ".join(n[:2]) + (f" +{len(n) - 2}" if len(n) > 2 else "")


def _params_comments(cls) -> dict[str, str]:
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}
    out = {}
    for line in source.splitlines():
        m = re.match(r"^\s+(\w+)\s*:\s*[^=#]+=\s*[^#]*?(?:#\s*(.*))?$", line)
        if m and m.group(2):
            out[m.group(1)] = m.group(2).strip()
    return out


def validate_params(cls, data, preset: str | None = None):
    if data is not None and not isinstance(data, dict):
        raise ToolError(f"params must be an object of {cls.__name__} fields")
    try:
        p = cls.preset(preset) if preset else cls()
        p.update(data or {})
        return p
    except (ValueError, TypeError) as exc:
        raise ToolError(str(exc))


def _ids(ctx, refs, kinds=None, minimum: int = 1) -> list[dict]:
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list) or len(refs) < minimum:
        raise ToolError(f"asset_ids must be a list with at least {minimum} id(s)")
    metas = [resolve_asset(ctx.workspace, r, kinds) for r in refs]
    if len({m["id"] for m in metas}) != len(metas):
        raise ToolError("asset_ids contains the same asset twice")
    return metas


def _guard_scale(clean_params: dict | None, args: dict) -> None:
    scale = (clean_params or {}).get("scale", 1.0)
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return
    if scale != 1.0 and args.get("user_requested_scaling") is not True:
        raise ToolError("Refusing to rescale: scale != 1 changes every dimension. Only do this when the user "
                        "explicitly asked for a unit conversion, then pass user_requested_scaling=true.")


# --------------------------------------------------------------------------- handlers
def h_list_assets(ctx: ToolContext, args: dict) -> ToolResult:
    """Legacy name: workspace_overview (projects and assets with part dimensions)."""
    from .tools_v3 import h_workspace_overview

    return h_workspace_overview(ctx, args)


def h_get_asset(ctx: ToolContext, args: dict) -> ToolResult:
    ws = ctx.workspace
    meta = resolve_asset(ws, args.get("asset_id"))
    s = meta.get("stats", {})
    stats = {k: (nums(v) if isinstance(v, list) else num(v)) for k, v in s.items()
             if k in ("points", "vertices", "triangles", "has_colors", "has_normals", "surface_area", "watertight",
                      "volume", "bbox_min", "bbox_max", "dimensions", "oriented_dimensions", "diagonal", "spacing",
                      "width", "height")}
    children = [m["id"] for m in ws.list() if meta["id"] in m.get("parents", [])]
    data = {"id": meta["id"], "name": meta["name"], "kind": meta["kind"], "operation": meta.get("operation"),
            "parents": meta.get("parents", []), "children": children[:10], "created": meta.get("created"),
            "stats": stats, "report": report_summary(meta, ws.report(meta["id"]))}
    if meta.get("scalars"):
        data["scalars"] = [{"name": x["name"], "unit": x.get("unit"), "min": num(x.get("min")),
                            "max": num(x.get("max"))} for x in meta["scalars"]]
    if meta.get("textured"):
        data["textured"] = True
    source = (meta.get("params") or {}).get("source")
    if source:
        data["source"] = source
    return ToolResult(data, f"{meta['name']}", [meta["id"]])


def h_describe_parameters(ctx: ToolContext, args: dict) -> ToolResult:
    op = str(args.get("operation") or "").lower()
    if op in ("clean", "merge", "mesh", "compare"):
        if op == "clean":
            from ..clean import CleanParams as cls
        elif op == "merge":
            from ..register import MergeParams as cls
        elif op == "mesh":
            from ..mesh import MeshParams as cls
        else:
            try:
                from ..compare import CompareParams as cls
            except ImportError:
                raise ToolError("Scan-vs-CAD comparison is not installed in this build")
        comments = _params_comments(cls)
        obj = cls()
        params = {f.name: f"{json.dumps(getattr(obj, f.name))}" + (f" - {comments[f.name]}" if f.name in comments
                                                                     else "") for f in fields(obj)}
        data = {"operation": op, "params": params, "units": UNITS}
        if op == "clean":
            data["presets"] = cls.PRESETS
            data["note"] = "'multiplier' distances are relative to the measured point spacing"
        return ToolResult(data, f"{op} parameters")
    if op == "edit":
        try:
            from ..edit import OP_CATALOGUE
        except ImportError:
            raise ToolError("Geometry editing is not installed in this build")
        name = args.get("op")
        if name:
            if name not in OP_CATALOGUE:
                raise ToolError(f"Unknown edit op '{name}'. Known: {', '.join(OP_CATALOGUE)}")
            return ToolResult({"op": name, **OP_CATALOGUE[name], "units": UNITS}, f"edit op {name}")
        compact = {}
        for key, spec in OP_CATALOGUE.items():
            arg_specs = spec.get("args", {}) if isinstance(spec, dict) else {}
            sig = ", ".join(f"{a}={json.dumps(v.get('default')) if isinstance(v, dict) else ''}"
                            for a, v in arg_specs.items())
            applies = spec.get("applies_to", []) if isinstance(spec, dict) else []
            compact[key] = f"({sig}) {'/'.join(applies)}"
        return ToolResult({"ops": compact, "detail": "pass op=<name> for argument descriptions"}, "edit ops")
    if op == "pipeline":
        return ToolResult({"operation": "pipeline",
                           "args": "asset_ids, preset, skip_clean, clean{CleanParams}, merge{MergeParams}, "
                                   "mesh{MeshParams}, merge_mode ask|auto|always|never, user_confirmed",
                           "steps": "clean each point cloud -> check the merge (2+) -> merge only when allowed -> "
                                    "mesh",
                           "detail": "call describe_parameters for clean, merge and mesh"}, "pipeline parameters")
    raise ToolError("operation must be one of clean, merge, mesh, pipeline, edit, compare")


async def h_clean(ctx: ToolContext, args: dict) -> ToolResult:
    from ..clean import CleanParams

    metas = _ids(ctx, args.get("asset_ids"), ("pointcloud", "mesh"))
    preset = args.get("preset") or "standard"
    params = args.get("params") or {}
    validate_params(CleanParams, params, preset)
    _guard_scale(params, args)
    payload = {"asset_ids": [m["id"] for m in metas], "preset": preset, "params": params}
    job = await run_job(ctx, "clean", f"Clean {_names(metas)}", payload)
    return _job_result(ctx, job, "Cleaned")


MERGE_CONFIRM_HINT = ("Check first: call assess_merge, explain its recommendation and reasons to the user and ask "
                      "whether to merge, use the best scan or keep the scans separate. Pass user_confirmed=true only "
                      "when the user explicitly asked to merge these scans in this conversation or approved it.")


def assessment_for_model(job_id: str, assessment: dict, metas: list[dict]) -> dict:
    """Compact merge assessment (no transforms) for the model."""
    scans = []
    for s in assessment.get("scans", []):
        i = s.get("index", 0)
        item = {"scan": i, "id": metas[i]["id"] if i < len(metas) else None,
                "name": prompt_safe(metas[i]["name"], 40) if i < len(metas) else None,
                "points": s.get("points"), "noise_mm": num(s.get("noise_mm"))}
        if i:
            item.update(fitness=num(s.get("fitness")), overlap=num(s.get("overlap")),
                        new_surface_pct=num(s.get("coverage_gain_pct")),
                        new_surface_mm2=num(s.get("coverage_gain_mm2")),
                        layering_mm=num(s.get("layering_mm")), layering_x_noise=num(s.get("separation_ratio")),
                        doubled_surface=s.get("doubled_surface"), ambiguous=s.get("ambiguous"),
                        already_aligned=s.get("already_aligned"))
        scans.append(item)
    best = assessment.get("best_index") or 0
    return {"assessment_job_id": job_id, "recommendation": assessment.get("recommendation"),
            "best_scan": {"scan": best, "id": metas[best]["id"], "name": prompt_safe(metas[best]["name"], 40)},
            "reasons": assessment.get("reasons", []), "scans": scans}


async def _assess(ctx: ToolContext, metas: list[dict], params: dict) -> tuple[dict, dict]:
    job = await run_job(ctx, "merge_assess", f"Check merge {_names(metas)}",
                        {"asset_ids": [m["id"] for m in metas], "params": params})
    assessment = (job.get("output") or {}).get("assessment")
    if not assessment:
        raise ToolError("The merge assessment returned no result")
    return job, assessment


async def h_assess_merge(ctx: ToolContext, args: dict) -> ToolResult:
    from ..register import MergeParams

    metas = _ids(ctx, args.get("asset_ids"), ("pointcloud", "mesh"), minimum=2)
    params = args.get("params") or {}
    validate_params(MergeParams, params)
    job, assessment = await _assess(ctx, metas, params)
    data = assessment_for_model(job["id"], assessment, metas)
    if assessment.get("recommendation") == "merge":
        data["next"] = ("Merging is recommended. If the user already asked to merge in this conversation, merge with "
                        "assessment_job_id and user_confirmed=true; otherwise report the reasons and ask first.")
    else:
        # Enforced stop: the user asked to be consulted whenever merging is not clearly right.
        data["next"] = ("STOP HERE. Do not call any more tools in this reply (no merge, no mesh, no clean). Explain the "
                        "recommendation and the reasons in plain words, then ask the user to choose: merge anyway, use "
                        "only the best scan (name it), or keep the scans separate. Wait for their answer.")
    return ToolResult(data, f"Merge check: {assessment.get('recommendation')}", [m["id"] for m in metas])


async def h_merge(ctx: ToolContext, args: dict) -> ToolResult:
    from ..register import MergeParams

    metas = _ids(ctx, args.get("asset_ids"), ("pointcloud", "mesh"), minimum=2)
    params = args.get("params") or {}
    validate_params(MergeParams, params)
    if args.get("user_confirmed") is not True:
        raise ToolError(f"Not merged. {MERGE_CONFIRM_HINT}")
    transforms = None
    choice = None
    if args.get("options_job_id"):
        try:
            prior = ctx.jobs.get(str(args["options_job_id"]))
        except KeyError:
            raise ToolError("Unknown options_job_id - call compare_merge_options again")
        out = prior.get("output") or {}
        if prior.get("kind") != "merge_options" or not out.get("options"):
            raise ToolError("options_job_id is not a finished compare_merge_options job")
        if out.get("asset_ids") != [m["id"] for m in metas]:
            raise ToolError("The options were drawn for other scans or another order - pass the same asset_ids")
        chosen = next((o for o in out["options"] if o["id"] == str(args.get("option") or "").strip().upper()), None)
        if chosen is None:
            raise ToolError(f"option must be one of {', '.join(o['id'] for o in out['options'])}")
        transforms = [None, chosen["transform"]]
        choice = {"options_job_id": prior["id"], "option": chosen["id"], "name": chosen["name"],
                  "of": len(out["options"])}
    elif args.get("assessment_job_id"):
        try:
            prior = ctx.jobs.get(str(args["assessment_job_id"]))
        except KeyError:
            raise ToolError("Unknown assessment_job_id - call assess_merge again")
        assessment = (prior.get("output") or {}).get("assessment") or {}
        if prior.get("kind") != "merge_assess" or not assessment:
            raise ToolError("assessment_job_id is not a finished assess_merge job")
        if assessment.get("asset_ids") != [m["id"] for m in metas]:
            raise ToolError("The assessment was made for other scans or another order - pass the same asset_ids "
                            "in the same order, or call assess_merge again")
        transforms = assessment.get("transforms")
    payload = {"asset_ids": [m["id"] for m in metas], "params": params, "pairs": None,
               "name": (args.get("name") or None), "transforms": transforms}
    if choice:  # recorded on the merged asset: which drawn option (compare_merge_options) the pose came from
        payload["chosen_option"] = choice
    job = await run_job(ctx, "merge", f"Merge {_names(metas)}", payload)
    return _job_result(ctx, job, "Merged", "fitness = overlap fraction after alignment (>0.3 good for partial "
                                           "views); rmse in scan units. Warnings/ambiguous mean check visually.")


async def h_mesh(ctx: ToolContext, args: dict) -> ToolResult:
    from ..mesh import MeshParams

    meta = resolve_asset(ctx.workspace, args.get("asset_id"), ("pointcloud", "mesh"))
    params = dict(args.get("params") or {})
    # small models often put settings next to asset_id ({"watertight": true}) instead of in params: take them too
    fields = set(MeshParams.__dataclass_fields__)
    params.update({k: v for k, v in args.items() if k in fields and k not in params})
    validate_params(MeshParams, params)
    job = await run_job(ctx, "mesh", f"Mesh {meta['name']}", {"asset_id": meta["id"], "params": params})
    return _job_result(ctx, job, "Meshed", "deviation_to_scan = distance from scan points to the mesh surface; "
                                           "compare p95 with the point spacing")


async def h_run_pipeline(ctx: ToolContext, args: dict) -> ToolResult:
    from ..clean import CleanParams
    from ..mesh import MeshParams
    from ..register import MergeParams

    metas = _ids(ctx, args.get("asset_ids"), ("pointcloud", "mesh"))
    preset = args.get("preset") or "standard"
    merge_mode = args.get("merge_mode") or "ask"
    if merge_mode not in PIPELINE_MERGE_MODES:
        raise ToolError(f"merge_mode must be one of {', '.join(PIPELINE_MERGE_MODES)}")
    payload = {"asset_ids": [m["id"] for m in metas], "preset": preset,
               "skip_clean": bool(args.get("skip_clean", False)), "clean": args.get("clean") or {},
               "merge": args.get("merge") or {}, "mesh": args.get("mesh") or {}, "pairs": None}
    validate_params(CleanParams, payload["clean"], preset)
    validate_params(MergeParams, payload["merge"])
    validate_params(MeshParams, payload["mesh"])
    _guard_scale(payload["clean"], args)
    if len(metas) == 1 or merge_mode == "always":
        if len(metas) > 1 and args.get("user_confirmed") is not True:
            raise ToolError(f"Not run: merge_mode=always merges without checking. {MERGE_CONFIRM_HINT}")
        job = await run_job(ctx, "pipeline", f"Pipeline {_names(metas)}", payload)
        return _job_result(ctx, job, "Pipeline created")

    # 2+ scans: clean, then check before merging (or mesh each scan separately)
    ws = ctx.workspace
    created: list[str] = []
    ids = [m["id"] for m in metas]
    if not payload["skip_clean"]:
        clouds = [m["id"] for m in metas if m["kind"] == "pointcloud"]
        if clouds:
            job = await run_job(ctx, "clean", f"Clean {_names(metas)}",
                                {"asset_ids": clouds, "preset": preset, "params": payload["clean"]})
            cleaned = dict(zip(clouds, job.get("result") or []))
            created += list(job.get("result") or [])
            ids = [cleaned.get(i, i) for i in ids]
    work = [ws.get(i) for i in ids]
    if merge_mode == "never":
        for m in work:
            job = await run_job(ctx, "mesh", f"Mesh {m['name']}", {"asset_id": m["id"], "params": payload["mesh"]})
            created += list(job.get("result") or [])
        return ToolResult({"ok": True, "merge": "not merged (merge_mode=never)",
                           "created": created_summary(ws, created)}, "Meshed each scan separately", created)

    job, assessment = await _assess(ctx, work, payload["merge"])
    summary = assessment_for_model(job["id"], assessment, work)
    if merge_mode == "auto" and assessment.get("recommendation") == "merge":
        merged = await run_job(ctx, "merge", f"Merge {_names(work)}",
                               {"asset_ids": ids, "params": payload["merge"], "pairs": None,
                                "transforms": assessment.get("transforms")})
        created += list(merged.get("result") or [])
        mesh = await run_job(ctx, "mesh", f"Mesh {_names(work)}",
                             {"asset_id": merged["result"][0], "params": payload["mesh"]})
        created += list(mesh.get("result") or [])
        return ToolResult({"ok": True, "assessment": {k: summary[k] for k in ("recommendation", "reasons")},
                           "created": created_summary(ws, created)}, "Checked, merged and meshed", created)
    data = {"ok": True, "stopped": "decision needed before merging - nothing was merged or meshed",
            "cleaned_ids": ids, **summary,
            "next": "Explain the reasons and ask the user: merge (merge with these cleaned_ids, assessment_job_id "
                    "and user_confirmed=true, then mesh), mesh only the best scan, or mesh each scan separately."}
    return ToolResult(data, f"Cleaned; merge check: {assessment.get('recommendation')} - asking the user",
                      created or ids)


async def h_edit(ctx: ToolContext, args: dict) -> ToolResult:
    try:
        from ..edit import validate_ops
    except ImportError:
        raise ToolError("Geometry editing is not installed in this build")
    ops = args.get("ops") or []
    if isinstance(ops, dict):
        ops = [ops]
    if not isinstance(ops, list) or not all(isinstance(o, dict) for o in ops):
        raise ToolError("ops must be a list of objects like {\"op\": \"translate\", \"offset\": [0, 0, 5]}")
    ops = [dict(o) for o in ops]
    asset_ref = args.get("asset_id")
    if args.get("use_screen_selection"):
        sel = (ctx.ui or {}).get("selection") or {}
        if not sel.get("polygon") or not sel.get("view_projection"):
            raise ToolError("The user has no screen selection. Ask them to draw one with the lasso/box tool first.")
        if not asset_ref:  # Contract 7 sends asset_ids, older clients asset_id
            asset_ref = sel.get("asset_id") or (sel.get("asset_ids") or [None])[0] or (ctx.ui or {}).get("active_id")
        filled = False
        for o in ops:
            if o.get("op") == "select_screen":
                o["view_projection"], o["polygon"] = sel["view_projection"], sel["polygon"]
                filled = True
        if not filled:
            ops.insert(0, {"op": "select_screen", "view_projection": sel["view_projection"],
                           "polygon": sel["polygon"], "mode": args.get("selection_mode") or "delete",
                           "visible_only": bool(args.get("visible_only", False))})
    if not ops:
        raise ToolError("ops is empty")
    meta = resolve_asset(ctx.workspace, asset_ref, ("pointcloud", "mesh"))
    if any(o.get("op") == "scale" for o in ops) and args.get("user_requested_scaling") is not True:
        raise ToolError("Refusing to scale: only when the user explicitly asked; then pass "
                        "user_requested_scaling=true.")
    try:
        ops = validate_ops(ops, meta["kind"])
    except (ValueError, TypeError) as exc:
        raise ToolError(f"Invalid edit ops: {exc}")
    payload = {"asset_id": meta["id"], "ops": ops, "name": args.get("name") or None}
    title = f"Edit {meta['name']} ({', '.join(o['op'] for o in ops[:3])}{'...' if len(ops) > 3 else ''})"
    job = await run_job(ctx, "edit", title, payload)
    return _job_result(ctx, job, "Edited")


async def h_compare(ctx: ToolContext, args: dict) -> ToolResult:
    try:
        from ..compare import CompareParams
    except ImportError:
        raise ToolError("Scan-vs-CAD comparison is not installed in this build")
    scan = resolve_asset(ctx.workspace, args.get("scan_id"), ("pointcloud", "mesh"))
    ref = resolve_asset(ctx.workspace, args.get("reference_id"), ("mesh",))
    if scan["id"] == ref["id"]:
        raise ToolError("scan_id and reference_id are the same asset")
    params = dict(args.get("params") or {})
    if args.get("tolerance") is not None:
        params["tolerance"] = args["tolerance"]
    validate_params(CompareParams, params)
    payload = {"scan_id": scan["id"], "reference_id": ref["id"], "params": params}
    job = await run_job(ctx, "compare", f"Compare {scan['name']} vs {ref['name']}", payload)
    return _job_result(ctx, job, "Compared", "signed deviation in scan units (+ = material outside CAD). Show "
                                             "it with ui display color_mode=scalar scalar=deviation.")


def golden_brief(report: dict) -> dict:
    """What a golden model check found, short enough for the model to explain (docs/golden-model.md)."""
    def meas(m: dict) -> dict:
        out = {k: m.get(k) for k in ("name", "golden", "scan", "difference", "uncertainty", "status")}
        if m.get("reason"):
            out["reason"] = m["reason"]
        if m.get("toward"):
            out["moved_toward"] = m["toward"]
        return out

    measurements = report.get("measurements", [])
    ranked = sorted(measurements, key=lambda m: {"off": 0, "close": 1, "not_measured": 2, "ok": 3}.get(m["status"], 4))
    return {"verdict": report.get("verdict"), "headline": report.get("headline"), "summary": report.get("summary"),
            "tolerance": report.get("tolerance"), "scan_again": report.get("rescan", [])[:8],
            "areas": [{k: r.get(k) for k in ("id", "kind", "name", "area_mm2", "share_pct", "deviation", "spread")
                       if r.get(k) is not None} for r in report.get("regions", [])[:10]],
            "measurements": [meas(m) for m in ranked[:20]], "counts": report.get("counts"),
            "surface_scanned_pct": (report.get("surface") or {}).get("scanned_pct"),
            "warnings": report.get("warnings", [])}


async def h_golden(ctx: ToolContext, args: dict) -> ToolResult:
    from ..golden import GoldenParams

    ws = ctx.workspace
    scan = resolve_asset(ws, args.get("scan_id"), ("pointcloud", "mesh"))
    golden_id = args.get("golden_id")
    if not golden_id:
        try:
            golden_id = ws.get_project(scan["project"]).get("golden_asset_id") if scan.get("project") else None
            if golden_id:
                ws.get(golden_id)
        except KeyError:
            golden_id = None
        if not golden_id:
            raise ToolError("This project has no golden model yet. Ask the user which CAD model or mesh is the golden "
                            "model (or to import the CAD file), then pass its id as golden_id.")
    golden = resolve_asset(ws, golden_id, ("mesh",))
    if scan["id"] == golden["id"]:
        raise ToolError("The scan cannot be its own golden model: pass the CAD model or mesh as golden_id")
    params = {}
    if args.get("tolerance") is not None:
        params["tolerance"] = args["tolerance"]
    if args.get("align"):
        params["align"] = args["align"]
    validate_params(GoldenParams, params)
    if args.get("remember", True) and scan.get("project"):
        try:
            ws.update_project(scan["project"], golden_asset_id=golden["id"])
        except (KeyError, ValueError):
            pass   # a golden model from another project: used this once, not remembered
    payload = {"scan_id": scan["id"], "golden_id": golden["id"], "params": params}
    job = await run_job(ctx, "golden_check", f"Check {scan['name']} against {golden['name']}", payload)
    result = _job_result(ctx, job, "Checked", "Explain the headline, then what to scan again (and how), then the "
                                              "measurements that are off or not measured, in plain words. The user "
                                              "sees it all in Measure -> Golden model.")
    ids = list(job.get("result") or [])
    report = ws.report(ids[0]) if ids else None
    if report:
        result.data["check"] = golden_brief(report)
    return result


async def h_fill_photos(ctx: ToolContext, args: dict) -> ToolResult:
    meta = resolve_asset(ctx.workspace, args.get("asset_id"), ("pointcloud", "mesh"))
    payload = {"asset_id": meta["id"], "photo_ids": list(args.get("photo_ids") or []), "params": {}}
    job = await run_job(ctx, "fill_from_photos", f"Fill {meta['name']} from photos", payload)
    return _job_result(ctx, job, "Filled", "Say how much was filled (output.added points, filled_area_mm2) and how "
                                           "far the photo surface sat from the scan (photo_vs_scan_mm): filled areas "
                                           "complete the model but are not for measuring, and the golden model "
                                           "check leaves them out.")


def h_export(ctx: ToolContext, args: dict) -> ToolResult:
    from ..io import save
    from ..web.workspace import safe_name

    try:
        from ..web.server import EXPORT_FORMATS
    except ImportError:  # pragma: no cover
        EXPORT_FORMATS = {"pointcloud": ["ply", "pcd", "xyz", "asc", "csv"], "mesh": ["ply", "stl", "obj", "glb", "off"]}
    ws = ctx.workspace
    meta = resolve_asset(ws, args.get("asset_id"), ("pointcloud", "mesh"))
    fmt = str(args.get("format") or "").lower().lstrip(".")
    allowed = EXPORT_FORMATS[meta["kind"]]
    if fmt not in allowed:
        raise ToolError(f"A {meta['kind']} can be exported as {', '.join(allowed)}, not '{fmt}'")
    folder = Path(str(args.get("folder") or "").strip().strip('"')).expanduser() if args.get("folder") \
        else ws.root / "exports"
    if not folder.is_absolute():
        folder = ws.root / folder
    try:
        from ..web.jobs_edit import export_asset
    except ImportError:
        export_asset = None
    if export_asset is not None:  # same code path as POST /api/export
        try:
            # Guardrail: the assistant never overwrites files (an instruction hidden in a file or asset name must
            # not be able to replace existing exports). The user can still overwrite from the Info page.
            path = export_asset(ws, meta["id"], fmt, str(folder), args.get("filename") or None, False)
        except FileExistsError as exc:
            raise ToolError(f"{exc}. The assistant never overwrites files: pick a new filename, or tell the user to "
                            "overwrite it from the Info page.")
        except (ValueError, OSError) as exc:
            raise ToolError(f"Export failed: {exc}")
        path = Path(path)
        return ToolResult({"ok": True, "path": str(path), "bytes": path.stat().st_size if path.exists() else 0,
                           "format": fmt}, f"Exported {meta['name']} to {path}", [meta["id"]])
    stem = safe_name(Path(str(args.get("filename") or meta["name"])).stem)
    path = folder / f"{stem}.{fmt}"
    if path.exists():
        raise ToolError(f"{path} already exists. The assistant never overwrites files - pick a new "
                        "filename.")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        save(ws.load_geometry(meta["id"]), path)
    except OSError as exc:
        raise ToolError(f"Could not write {path}: {exc}")
    size = path.stat().st_size if path.exists() else 0
    return ToolResult({"ok": True, "path": str(path), "bytes": size, "format": fmt},
                      f"Exported {meta['name']} to {path}", [meta["id"]])


def h_rename(ctx: ToolContext, args: dict) -> ToolResult:
    meta = resolve_asset(ctx.workspace, args.get("asset_id"))
    name = str(args.get("name") or "").strip()
    if not name:
        raise ToolError("name must not be empty")
    ctx.workspace.update(meta["id"], name=name[:120])
    return ToolResult({"ok": True, "id": meta["id"], "old_name": meta["name"], "name": name[:120]},
                      f"Renamed '{meta['name']}' to '{name[:120]}'", [meta["id"]])


def h_measure_thread(ctx: ToolContext, args: dict) -> ToolResult:
    """Legacy name: measure kind=thread."""
    from .tools_v3 import h_measure

    return h_measure(ctx, {**args, "kind": "thread"})


def thread_result(geom) -> tuple[dict, str, dict]:
    """Thread analysis of a geometry for the model: (compact result, summary sentence, raw analysis)."""
    from ..analysis import thread_analysis

    try:
        t = thread_analysis(geom, None, lambda m: None)
    except ValueError as exc:
        raise ToolError(f"No thread measured: {exc}")
    from ..analysis import nearest_standard_thread

    standards = nearest_standard_thread(t["major_diameter"], t["pitch"])[:3]
    result = {k: (num(t[k]) if isinstance(t.get(k), float) else t.get(k)) for k in
              ("kind", "handedness", "pitch", "pitch_se", "pitch_max_deviation", "crest_count", "major_diameter",
               "minor_diameter", "pitch_diameter", "flank_angle_deg", "confidence", "warnings")}
    # Tolerance interpretation is done here, not by the model: 6g applies to external (bolt) threads, 6H to internal
    # (nut) threads. Only the class matching the measured kind is passed on, with a computed verdict sentence.
    tol_class = "6g" if t.get("kind") == "external" else "6H" if t.get("kind") == "internal" else None
    result["standards"] = []
    for s in standards:
        item = {k: (num(s[k]) if isinstance(s.get(k), float) else s.get(k)) for k in
                ("designation", "pitch", "major_diameter", "pitch_deviation", "major_diameter_deviation", "match")}
        tol = (s.get("tolerance") or {}).get(tol_class) if tol_class else None
        if tol:
            item["tolerance_class"] = tol_class
            item["major_diameter_limits"] = [num(tol.get("major_min")), num(tol.get("major_max"))]
        result["standards"].append(item)
    best = standards[0] if standards else None
    tol = ((best or {}).get("tolerance") or {}).get(tol_class) if tol_class else None
    if best and tol:
        d, lo, hi = t["major_diameter"], tol.get("major_min"), tol.get("major_max")
        if hi is not None and d > hi:
            where = f"{d - hi:.3f} mm above the ISO {tol_class} maximum {hi:.3f} mm"
        elif lo is not None and d < lo:
            where = f"{lo - d:.3f} mm below the ISO {tol_class} minimum {lo:.3f} mm"
        else:
            where = f"within ISO {tol_class} ({lo if lo is not None else '-'} to {hi if hi is not None else '-'} mm)"
        result["verdict"] = (f"{t['kind'].capitalize()} thread, closest standard {best['designation']}: pitch "
                             f"{best['pitch_deviation']:+.4f} mm from nominal; major diameter {d:.3f} mm is {where}. "
                             f"{tol_class} is the class that applies to {t['kind']} threads. A scan of an ideal CAD "
                             "profile sits at the basic size, which is above 6g; real bolts are made slightly under it.")
    best = standards[0]["designation"] if standards else "no standard"
    return result, f"pitch {t['pitch']:.4f}, major Ø {t['major_diameter']:.3f} ({best})", t


PIPELINE_MERGE_MODES = ["ask", "auto", "always", "never"]
def h_request_delete(ctx: ToolContext, args: dict) -> ToolResult:
    metas = _ids(ctx, args.get("asset_ids"))
    names = ", ".join(f"'{m['name']}'" for m in metas[:5]) + (f" and {len(metas) - 5} more" if len(metas) > 5 else "")
    message = f"Delete {names}? This cannot be undone."
    if args.get("reason"):
        message += f" Reason: {str(args['reason'])[:200]}"
    ids = [m["id"] for m in metas]
    ctx.emit("confirm", {"call_id": ctx.call_id, "action": "delete", "asset_ids": ids, "message": message})
    return ToolResult("Asked the user to confirm — do not assume it was deleted", "Asked to confirm deletion", ids)


# --------------------------------------------------------------------------- schemas
def _obj(props: dict, required=()) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


_IDS = {"type": "array", "items": {"type": "string"}, "description": "asset ids (names also accepted)"}
_ID = {"type": "string", "description": "asset id (a unique name also works)"}
_SCALE_FLAG = {"type": "boolean", "description": "true only if the user explicitly asked to rescale / convert units"}

EDIT_OPS_DOC = (
    "crop_box(min[3],max[3],invert=false); cut_plane(point[3],normal[3],keep=positive|negative); "
    "(stray surface under the base: describe_part gives the base face's point and outward normal n; cut_plane "
    "point=base point + 0.2*n, normal=n, keep=negative); "
    "delete_region(region); keep_region(region); "
    "select_screen(view_projection[16],polygon[[x,y]..] NDC,mode=delete|keep,visible_only=false); "
    "delete_sphere(center[3],radius); transform(matrix 4x4 row-major); translate(offset[3]); "
    "rotate(axis x|y|z|[3],degrees,center=centroid|origin|[3]); scale(factor,center) ONLY if user asked; "
    "mirror(axis,center); center(mode=bbox|centroid|bbox_bottom,up=z|y); align_principal(); align_floor(up=z|y); "
    "downsample(voxel) cloud; remove_outliers(neighbors=24,std_ratio=2.0) cloud; "
    "remove_small_components(min_ratio=0.02); simplify(target_triangles|ratio) mesh; "
    "smooth(iterations=5,method=taubin|laplacian|bilateral,region,feather); denoise(region,feather); "
    "remove_spikes(region,feather); fill_holes(max_hole_size, 0=all) mesh; subdivide(iterations=1) mesh; "
    "repair() mesh; flip_normals(); recompute_normals(); to_pointcloud(samples=0) mesh; paint(rgb[3] 0..1,region). "
    "region = {spheres|box|obb|cylinder|slab|end|all} (see select_region), e.g. {\"end\": {\"direction\": "
    "\"length\", \"side\": \"max\", \"length\": 10}} = last 10 mm along the part's length")


def build_tools() -> dict[str, Tool]:
    """The advertised tools, in a fixed order (the tool list is part of the cached prompt prefix)."""
    from .tools_v3 import v3_tools

    v3 = {t.name: t for t in v3_tools()}
    tools = [
        v3["workspace_overview"],
        Tool("get_asset", "Details of one asset: stats (points/triangles, bbox, dimensions, spacing, watertight, "
             "volume, area), parents/children, scalar fields and the key numbers of the report of the operation "
             "that created it (clean removed %, merge fitness/rmse, mesh deviation, compare stats).",
             _obj({"asset_id": _ID}, ["asset_id"]), h_get_asset),
        v3["describe_part"],
        v3["look_at_photos"],
        v3["app_guide"],
        v3["compare_merge_options"],
        v3["photos_to_3d"],
        v3["colour_from_photos"],
        v3["select_region"],
        v3["measure"],
        Tool("describe_parameters", "Parameters with defaults and explanations for an operation. Call before "
             "using non-default params; never invent parameter names.",
             _obj({"operation": {"type": "string", "enum": ["clean", "merge", "mesh", "pipeline", "edit", "compare"]},
                   "op": {"type": "string", "description": "for operation=edit: one op name for full details"}},
                  ["operation"]), h_describe_parameters),
        Tool("clean", "Clean point cloud scans (duplicates, statistical outliers, sparse points, floating "
             "clusters, optional support-plane/turntable removal, normals). Creates one new '<name> · clean' asset "
             "per input; inputs are kept. Presets: light, standard, aggressive. Waits for the job and returns "
             "points removed. Never set scale unless the user asked.",
             _obj({"asset_ids": _IDS, "preset": {"type": "string", "enum": ["light", "standard", "aggressive"]},
                   "params": {"type": "object", "description": "CleanParams overrides, e.g. {\"remove_plane\": true}"},
                   "user_requested_scaling": _SCALE_FLAG}, ["asset_ids"]), h_clean),
        Tool("assess_merge", "Check BEFORE merging whether 2+ scans should be merged (no asset is created): "
             "aligns each scan and measures new surface it adds, alignment confidence (fitness, symmetric "
             "ambiguity) and doubled/layered surfaces where scans overlap. Returns recommendation merge | use_best "
             "(scans repeat the same surface: merging only adds noise, use the best scan) | ask (uncertain) with "
             "plain-English reasons, the best scan and assessment_job_id.",
             _obj({"asset_ids": _IDS, "params": {"type": "object", "description": "MergeParams overrides"}},
                  ["asset_ids"]), h_assess_merge),
        Tool("merge", "Align (global feature matching + ICP) and merge 2+ scans of the same object into one "
             "point cloud; the first asset is the reference frame. Only after assess_merge and the user's "
             "approval (or an explicit request to merge): user_confirmed=true is required. Pass "
             "assessment_job_id to merge with exactly the alignment the user approved. Returns per-scan fitness "
             "(overlap 0..1) and rmse (scan units) plus warnings about low overlap or symmetric ambiguity.",
             _obj({"asset_ids": _IDS, "params": {"type": "object", "description": "MergeParams overrides, e.g. "
                   "{\"method\": \"icp\"} when scans are already roughly aligned, {\"method\": \"markers\"} to line "
                   "them up only on the marker stickers they share (auto already tries stickers first)"},
                   "name": {"type": "string"},
                   "options_job_id": {"type": "string", "description": "compare_merge_options result for the "
                                      "same asset_ids; merge with the pose of `option`"},
                   "option": {"type": "string", "description": "the option letter chosen from compare_merge_options"},
                   "assessment_job_id": {"type": "string", "description": "assess_merge result for the same "
                                         "asset_ids (same order): reuse its alignment"},
                   "user_confirmed": {"type": "boolean", "description": "true only if the user explicitly asked "
                                      "to merge these scans in this conversation or approved it after the check"}},
                  ["asset_ids"]), h_merge),
        Tool("mesh", "Reconstruct a triangle mesh from a point cloud (poisson: smooth, trimmed to the scan unless "
             "watertight=true; bpa: exact points). Returns triangles, dimensions and deviation of the scan points "
             "from the mesh (mean/p95/max, scan units).",
             _obj({"asset_id": _ID, "params": {"type": "object", "description": "MeshParams overrides, e.g. "
                   "{\"watertight\": true, \"target_triangles\": 200000}"}}, ["asset_id"]), h_mesh),
        Tool("run_pipeline", "Full pipeline: clean each point cloud -> (2+ scans) check the merge -> merge -> "
             "mesh. Use when the user wants a finished mesh from raw scans. merge_mode: ask (default: clean, check, "
             "then stop and return the assessment so you can ask the user), auto (merge only when the check "
             "recommends it, otherwise stop), always (merge without checking; needs user_confirmed=true), never "
             "(mesh every scan separately).",
             _obj({"asset_ids": _IDS, "preset": {"type": "string", "enum": ["light", "standard", "aggressive"]},
                   "skip_clean": {"type": "boolean"}, "clean": {"type": "object"}, "merge": {"type": "object"},
                   "mesh": {"type": "object"}, "merge_mode": {"type": "string", "enum": PIPELINE_MERGE_MODES},
                   "user_confirmed": {"type": "boolean", "description": "true only if the user explicitly asked "
                                      "to merge these scans"},
                   "user_requested_scaling": _SCALE_FLAG}, ["asset_ids"]),
             h_run_pipeline),
        Tool("edit", "Apply geometry edits in order to one asset, creating a new asset. Coordinates/distances in "
             "scan units (mm), transforms rigid unless scale was requested. To delete or keep what the user "
             "selected on screen, pass use_screen_selection=true (with selection_mode delete|keep) instead of "
             "writing select_screen yourself. Ops: " + EDIT_OPS_DOC,
             _obj({"asset_id": _ID,
                   "ops": {"type": "array", "items": {"type": "object", "properties": {"op": {"type": "string"}},
                                                      "required": ["op"]},
                           "description": "e.g. [{\"op\": \"rotate\", \"axis\": \"z\", \"degrees\": 90}]"},
                   "name": {"type": "string", "description": "name of the new asset"},
                   "use_screen_selection": {"type": "boolean"},
                   "selection_mode": {"type": "string", "enum": ["delete", "keep"]},
                   "visible_only": {"type": "boolean"},
                   "user_requested_scaling": _SCALE_FLAG}, ["ops"]), h_edit),
        Tool("compare_to_reference", "Inspect a scan against a CAD/reference mesh: aligns (auto/icp/none), "
             "computes signed deviation per point (+ = outside material, scan units), stats, % within tolerance, "
             "dimension differences, coverage and a scale estimate (reported, never applied). Creates a deviation-"
             "coloured result asset and a coverage mesh.",
             _obj({"scan_id": _ID, "reference_id": {"type": "string", "description": "CAD / reference mesh id"},
                   "tolerance": {"type": "number", "description": "± tolerance in scan units (mm), default 0.1"},
                   "params": {"type": "object", "description": "CompareParams overrides"}},
                  ["scan_id", "reference_id"]), h_compare),
        Tool("check_against_golden", "Check a scan against the golden model (the part as it should be: its CAD "
             "model or a trusted mesh). Lines them up, then finds what was not scanned, too thinly scanned, rough "
             "or off (as named areas with what to do), and measures every flat face distance, diameter and hole "
             "position of the golden model on the scan (golden vs scan vs difference, ok / off / too close to "
             "call / not measured). golden_id defaults to the project's golden model; the one used becomes the "
             "project's golden model. Use this when the user asks how the scan compares with the golden model or "
             "CAD, whether the measurements match, what is missing or what to rescan.",
             _obj({"scan_id": _ID,
                   "golden_id": {"type": "string", "description": "the golden model (CAD / mesh) asset id; "
                                                                  "default: the project's golden model"},
                   "tolerance": {"type": "number", "description": "± tolerance in mm, default 0.1"},
                   "align": {"type": "string", "enum": ["auto", "icp", "none"]},
                   "remember": {"type": "boolean", "description": "make it the project's golden model (default "
                                                                  "true)"}},
                  ["scan_id"]), h_golden),
        Tool("fill_from_photos", "Fill the areas a scan missed with points from photos of the part (the project's "
             "photos by default, 3 or more, best 12+ all the way round). Lines the photos up with the scan, adds "
             "only photo points where the scan has nothing (never the table), marks them as from photos, and makes "
             "a new asset '<scan> + photo fill'. Photos are only good to about 1-2 mm: use it to complete a model "
             "when the part cannot be scanned again, not for measuring.",
             _obj({"asset_id": _ID, "photo_ids": {"type": "array", "items": {"type": "string"},
                                                  "description": "photos to use (default: all of the project)"}},
                  ["asset_id"]), h_fill_photos),
        Tool("export_asset", "Write an asset to a file on the server disk. Point clouds: ply pcd xyz asc csv; "
             "meshes: ply stl obj glb off. Default folder <workspace>/exports. Never overwrites an existing file.",
             _obj({"asset_id": _ID, "format": {"type": "string"}, "folder": {"type": "string"},
                   "filename": {"type": "string"}}, ["asset_id", "format"]),
             h_export),
        Tool("rename_asset", "Rename an asset.", _obj({"asset_id": _ID, "name": {"type": "string"}},
                                                      ["asset_id", "name"]), h_rename),
        v3["holes"],
        v3["project"],
        v3["capture"],
        v3["turntable"],
        v3["ui"],
        Tool("request_delete", "Ask the user to confirm deleting assets. You cannot delete directly; the user "
             "decides in the UI.", _obj({"asset_ids": _IDS, "reason": {"type": "string"}}, ["asset_ids"]),
             h_request_delete),
    ]
    return {t.name: t for t in tools}


# Names older conversations (and models) may still use; not advertised, same behaviour
ALIASES = {"list_assets": h_list_assets, "measure_thread": h_measure_thread}


TOOLS = build_tools()


_TRUE = {"true", "yes", "1", "on"}
_FALSE = {"false", "no", "0", "off", "", "none", "null"}


def coerce(value, schema: dict | None):
    """A tool argument brought to the type its schema asks for. Tool calls parsed from text (servers without a tool
    parser) carry booleans as "True" / "False" and numbers as strings - and a guard like `if not
    args.get("user_confirmed")` would take the string "False" as a yes."""
    if not isinstance(schema, dict):
        return value
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    if kind == "boolean":
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
            return bool(value)
        return value
    if kind in ("integer", "number") and isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return value
        return int(number) if kind == "integer" and number.is_integer() else number
    if kind == "array":
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                value = json.loads(value)
            except ValueError:
                return value
        if isinstance(value, list) and isinstance(schema.get("items"), dict):
            return [coerce(v, schema["items"]) for v in value]
        return value
    if kind == "object" and isinstance(value, dict):
        props = schema.get("properties") or {}
        return {k: coerce(v, props.get(k)) for k, v in value.items()}
    return value


async def call_tool(ctx: ToolContext, name: str, args: dict) -> ToolResult:
    tool = TOOLS.get(name)
    handler = tool.handler if tool is not None else ALIASES.get(name)
    if handler is None:
        raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}")
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    if tool is not None:
        args = coerce(args, tool.parameters)
    if inspect.iscoroutinefunction(handler):
        return await handler(ctx, args)
    return await asyncio.to_thread(handler, ctx, args)
