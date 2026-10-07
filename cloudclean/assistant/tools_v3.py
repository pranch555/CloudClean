"""Assistant tools of CloudClean v3 (Contract 6): projects, part understanding, regions, the grouped measure tool,
live capture, the turntable and the full viewer control of Contract 7.

Registered by cloudclean.assistant.tools.build_tools(). Tools are grouped behind an `action` / `kind` enum so the
tool list stays short: the production model runs at a few tokens per second and every tool description is part of
every prompt (it is cached as a prefix, but it still has to be read once and it competes for attention).
"""
from __future__ import annotations

import asyncio
import math
import time

from .tools import (RESULT_LIMIT, Tool, ToolCancelled, ToolContext, ToolError, ToolResult, _ids, _obj,
                    asset_brief, num, nums, prompt_safe, resolve_asset, shrink)

DIRECTION_HELP = '"x"|"y"|"z"|"length"|"width"|"height" or [x,y,z]'
MEASURE_KINDS = ["points", "distance", "extent", "caliper", "faces", "diameter", "sphere", "plane", "angle",
                 "section", "thread"]
CAPTURE_ACTIONS = ["status", "drivers", "connect", "start", "pause", "resume", "stop", "discard", "save",
                   "marker_map", "follow", "pending", "decide"]
TURNTABLE_ACTIONS = ["status", "devices", "connect", "disconnect", "rotate", "tilt", "stop", "speed", "program",
                     "stop_program"]
PROJECT_ACTIONS = ["list", "create", "open", "rename", "move_asset"]
MARKER_COMMANDS = ["map_markers", "finish_map", "clear_map"]
PENDING_DECISIONS = ["fuse", "discard", "keep_separate"]
MOVE_TIMEOUT_S = 240.0


# --------------------------------------------------------------------------- shared helpers
def _fmt(x) -> str:
    return f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"


def part_dims(ws, meta: dict) -> str:
    """'L×W×H' along the part axes (robust, mm); '~' marks an approximation (older assets without a part frame)."""
    part = meta.get("part") or {}
    dims = part.get("dimensions")
    if isinstance(dims, dict) and all(k in dims for k in ("length", "width", "height")):
        return "×".join(_fmt(dims[k]) for k in ("length", "width", "height"))
    try:
        from ..understand import load_cached_summary

        cached = load_cached_summary(ws, meta["id"]) if meta.get("kind") != "image" else None
    except Exception:
        cached = None
    if cached:
        d = cached["dimensions"]
        return "×".join(_fmt(d[k]) for k in ("length", "width", "height"))
    stats = meta.get("stats") or {}
    approx = stats.get("oriented_dimensions") or stats.get("dimensions")
    if approx:
        return "~" + "×".join(_fmt(v) for v in sorted(approx, reverse=True))
    return ""


def resolve_project(ws, ref) -> dict:
    """A project by id, exact name (case-insensitive) or unique id prefix."""
    ref = str(ref or "").strip().strip("'\"")
    if not ref:
        raise ToolError("project is required (id or name)")
    projects = ws.projects()
    for p in projects:
        if p["id"] == ref:
            return p
    matches = [p for p in projects if p.get("name", "").casefold() == ref.casefold()]
    if not matches and len(ref) >= 4:
        matches = [p for p in projects if p["id"].startswith(ref.lower())]
    if not matches:
        raise ToolError(f"No project '{ref}'. Call project action=list to see them.")
    if len(matches) > 1:
        raise ToolError(f"'{ref}' matches several projects: "
                        + ", ".join(f"{p['id']} ({prompt_safe(p['name'], 40)})" for p in matches[:6]))
    return matches[0]


def _target_asset(ctx: ToolContext, args: dict, kinds=("pointcloud", "mesh")) -> dict:
    """asset_id from the arguments, else the asset of the screen selection (when it is used), else the active one."""
    ref = args.get("asset_id")
    if not ref and args.get("use_screen_selection"):
        sel = (ctx.ui or {}).get("selection") or {}
        ref = sel.get("asset_id") or (sel.get("asset_ids") or [None])[0]
    ref = ref or (ctx.ui or {}).get("active_id")
    if not ref:
        raise ToolError("asset_id is required (no asset is active in the viewer)")
    return resolve_asset(ctx.workspace, ref, kinds)


def _selection_region(ctx: ToolContext) -> dict:
    sel = (ctx.ui or {}).get("selection") or {}
    if not sel.get("polygon") or not sel.get("view_projection"):
        raise ToolError("The user has no screen selection. Ask them to draw one with the lasso/box tool first.")
    return {"view_projection": sel["view_projection"], "polygon": sel["polygon"],
            "visible_only": bool(sel.get("visible_only", False))}


def _region(ctx: ToolContext, args: dict, key: str = "region", required: bool = False):
    from ..regions import check_region

    if key == "region" and args.get("use_screen_selection"):
        return _selection_region(ctx)
    value = args.get(key)
    if value is None:
        if required:
            raise ToolError(f"{key} is required: a region object, e.g. {{\"end\": {{\"direction\": \"length\", "
                            "\"side\": \"max\", \"length\": 10}}")
        return None
    try:
        return check_region(value, key)
    except ValueError as exc:
        raise ToolError(str(exc))


def _geometry(ctx: ToolContext, meta: dict):
    from .. import understand as U

    try:
        geom = U.cached_geometry(ctx.workspace, meta["id"])
    except (KeyError, ValueError) as exc:
        raise ToolError(f"Cannot load '{meta['name']}': {exc}")
    return geom, U.frame_for(ctx.workspace, meta["id"], geom)


def _overlay(ctx: ToolContext, kind: str, result: dict, asset_id: str) -> None:
    from ..understand import overlay_items

    items = [{**item, "asset_id": asset_id} for item in overlay_items(kind, result)]
    if items:
        ctx.emit("ui", {"action": "measure_overlay", "items": items, "replace": False, "asset_id": asset_id})


def _rounded(obj):
    if isinstance(obj, float):
        return num(obj)
    if isinstance(obj, list):
        return [_rounded(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------- compare_merge_options
MAX_OPTION_PHOTOS = 2


async def h_compare_merge_options(ctx: ToolContext, args: dict) -> ToolResult:
    """Every way two scans can line up, drawn, next to the user's reference photos: the model picks the option that
    looks like the real part, then merge(options_job_id, option) uses exactly that pose."""
    from .tools import run_job

    metas = _ids(ctx, args.get("asset_ids"), ("pointcloud", "mesh"), minimum=2)
    if len(metas) != 2:
        raise ToolError("Compare the options of two scans at a time: the reference first, then the scan that moves")
    if not ctx.settings.get("vision"):
        raise ToolError("Vision is switched off in the assistant settings (Settings > Assistant), so you cannot see "
                        "the options - use assess_merge instead and tell the user")
    job = await run_job(ctx, "merge_options", f"Merge options: {metas[0]['name']} + {metas[1]['name']}",
                        {"asset_ids": [m["id"] for m in metas], "params": args.get("params") or {}})
    out = job.get("output") or {}
    options, key = out.get("options") or [], out.get("renders_key")
    if not options or not key:
        raise ToolError("No merge option was found - the scans may not overlap")
    folder = ctx.workspace.root / "renders" / key
    images = [{"url": photo_data_url(folder / f"{o['id']}.jpg", 1440), "caption": f"Option {o['id']}: {o['name']}"}
              for o in options]
    # the user's reference photos of the real part, to compare with
    wanted = [str(i) for i in (args.get("photo_ids") or []) if i]
    use_photos = args.get("use_photos", True)
    if isinstance(use_photos, str):  # some models send booleans as text
        use_photos = use_photos.strip().lower() not in ("false", "0", "no")
    if not wanted and use_photos:
        project = metas[0].get("project") or (ctx.ui or {}).get("project_id")
        photos = [m for m in ctx.workspace.list() if m.get("kind") == "image" and m.get("project") == project]
        photos.sort(key=lambda m: m.get("created") or "", reverse=True)
        wanted = [m["id"] for m in photos[:MAX_OPTION_PHOTOS]]
    for pid in wanted[:MAX_OPTION_PHOTOS]:
        try:
            meta = ctx.workspace.get(pid)
        except KeyError:
            raise ToolError(f"No photo {pid}")
        if meta.get("kind") != "image":
            raise ToolError(f"'{meta['name']}' ({pid}) is not a photo")
        images.append({"url": photo_data_url(ctx.workspace.image_path(pid)),
                       "caption": f"reference photo '{prompt_safe(meta['name'], 60)}' ({pid})"})
    ctx.emit("ui", {"action": "images", "title": "Merge options compared",
                    "images": [{"url": f"/api/merge/options/{key}/{o['id']}.jpg",
                                "caption": f"Option {o['id']}: {o['name']}"} for o in options]})
    rows = [{"option": o["id"], "what": o["name"], "overlap": num(o["fitness"]), "gap_mm": num(o["rmse_mm"]),
             "turned_deg": o["angle_from_best_deg"], "shifted_mm": o["shift_from_best_mm"], "from": o["source"]}
            for o in options]
    photos_seen = len(images) - len(options)
    note = ("Each option picture: top row plain grey (compare with the photos of the real part), bottom row coloured "
            "by scan (orange = reference, blue = the moved scan; mixed colours = the scans overlap there). Pick the "
            "option that looks like the real part: a wrong option shows e.g. two heads, a head at the wrong end, or "
            "surfaces that do not join. Options that look the same cannot be told apart by photos - say so and "
            "suggest stickers (Align > Line up on stickers) or a caliper check. Merge the chosen one with merge("
            "asset_ids, options_job_id, option) once the user agrees.")
    if not photos_seen:
        note += " There are no reference photos: judge from the shape alone, or ask the user for a photo (Photos)."
    summary = f"{len(options)} merge options drawn" + (f", {photos_seen} photo(s) to compare" if photos_seen else "")
    return ToolResult({"options_job_id": job["id"], "options": rows, "reference_photos": photos_seen, "note": note},
                      summary, [m["id"] for m in metas], images=images)


# --------------------------------------------------------------------------- photos_to_3d
async def h_photos_to_3d(ctx: ToolContext, args: dict) -> ToolResult:
    """A 3D model (coloured point cloud) from photos of the part (job photos_to_3d, MapAnything)."""
    from ..web.jobs_photos3d import MIN_PHOTOS, photos_of, recon_available
    from .tools import _job_result, run_job

    project = (ctx.ui or {}).get("project_id")
    try:
        photos = photos_of(ctx.workspace, [str(i) for i in args.get("photo_ids") or []], project)
    except KeyError as exc:
        raise ToolError(f"No asset {exc}")
    except ValueError as exc:
        raise ToolError(str(exc))
    if len(photos) < MIN_PHOTOS:
        raise ToolError("There are fewer than 2 photos of the part in this project. Ask the user to add 12 or more "
                        "photos taken all the way round (Photos button in the chat, or Scan > Make a 3D model from "
                        "photos)")
    ok, why = await asyncio.to_thread(recon_available)
    if not ok:
        raise ToolError(why)
    payload = {"photo_ids": [m["id"] for m in photos], "name": args.get("name") or None}
    if args.get("ruler_mm") is not None:
        payload["ruler_mm"] = float(args["ruler_mm"])
    job = await run_job(ctx, "photos_to_3d", f"3D model from {len(photos)} photos", payload)
    ids = list(job.get("result") or [])
    sheet = ((ctx.workspace.report(ids[0]) or {}).get("scale_sheet") or {}) if ids else {}
    if sheet.get("found"):
        hint = (f"Made on the printed scale sheet: the model is already true size (±{sheet.get('uncertainty_pct', 0):.2g} "
                "%), with Z = 0 on the sheet. Do not rescale it. Its surface is only about 0.5-1 mm accurate, so for "
                "holes, diameters and tight sizes the user should scan the part." +
                ("" if sheet.get("ruler_mm") else " The print was not checked: ask the user to measure the sheet's "
                 "100 mm bar with a caliper and give you the length (ruler_mm) next time."))
    else:
        hint = ("Photos do not show the true size: the model's size is a guess and can be many times off. Tell the "
                "user, ask for one known dimension, measure it on the model and scale with edit op scale (factor = "
                "true / measured, user_requested_scaling=true), or point them to Clean → Set the true size. Next "
                "time, photographing the part on the printed scale sheet (Scan → Make a 3D model from photos → "
                "Print the scale sheet) gives the true size by itself.")
    result = _job_result(ctx, job, "Made from photos", hint)
    if result.asset_ids:
        ctx.emit("ui", {"action": "show", "asset_ids": result.asset_ids[:1], "exclusive": True})
    return result


# --------------------------------------------------------------------------- colour_from_photos
async def h_colour_from_photos(ctx: ToolContext, args: dict) -> ToolResult:
    """The user's scan or mesh coloured from photos of the part (job colour_from_photos): the photos are lined up
    with the model automatically, the model's shape and size stay as they are."""
    from ..web.jobs_colour_photos import MIN_COLOUR_PHOTOS
    from ..web.jobs_photos3d import photos_of, recon_available
    from .tools import _job_result, run_job

    ref = args.get("model") or args.get("asset_id") or (ctx.ui or {}).get("active_id")
    if not ref:
        raise ToolError("model is required (no model is active in the viewer)")
    meta = resolve_asset(ctx.workspace, ref, ("pointcloud", "mesh"))
    try:
        photos = photos_of(ctx.workspace, [str(i) for i in args.get("photo_ids") or []],
                           meta.get("project") or (ctx.ui or {}).get("project_id"))
    except KeyError as exc:
        raise ToolError(f"No asset {exc}")
    except ValueError as exc:
        raise ToolError(str(exc))
    photos = list({m["id"]: m for m in photos}.values())
    if len(photos) < MIN_COLOUR_PHOTOS:
        raise ToolError(f"There are fewer than {MIN_COLOUR_PHOTOS} photos of the part in the project of "
                        f"'{prompt_safe(meta['name'], 60)}'. Ask the user to add 12 or more photos taken about every "
                        "30 degrees all the way round (Photos button in the chat, or Mesh → Colour from photos)")
    ok, why = await asyncio.to_thread(recon_available)
    if not ok:
        raise ToolError(why)
    job = await run_job(ctx, "colour_from_photos", f"Colour {meta['name']} from {len(photos)} photos",
                        {"asset_id": meta["id"], "photo_ids": [m["id"] for m in photos]})
    result = _job_result(ctx, job, "Coloured from photos",
                         "Same shape and size as the model, now with the colours of the photos. Pass on any warning "
                         "(a part that looks alike from several sides can get its colours turned). If the colours look "
                         "wrong, the user can line up one photo by hand: Mesh → Colour from photos.")
    if result.asset_ids:
        report = ctx.workspace.report(result.asset_ids[0]) or {}
        al = report.get("alignment") or {}
        result.data["alignment"] = {k: _rounded(al.get(k)) for k in
                                    ("scale", "fitness", "rmse_mm", "trusted", "ambiguous", "alternative_angle")}
        result.data["photos_used"] = report.get("used")
        result.data["coverage"] = num(report.get("coverage"))
        if report.get("warnings"):
            result.data["warnings"] = report["warnings"]
        ctx.emit("ui", {"action": "show", "asset_ids": result.asset_ids[:1], "exclusive": True})
    return result


# --------------------------------------------------------------------------- app_guide
def h_app_guide(ctx: ToolContext, args: dict) -> ToolResult:
    """Where a feature is and what it does (cloudclean/assistant/guide.py); opens it in the app and rings it."""
    from .guide import BY_ID, search

    wanted = str(args.get("feature") or "").strip()
    question = str(args.get("question") or "").strip()
    show = args.get("open", True) is not False
    if wanted:
        chosen = BY_ID.get(wanted)
        if chosen is None:
            hits = search(wanted, 1)
            if not hits:
                raise ToolError(f"No feature '{wanted}' in the app guide - pass the user's question instead")
            chosen = hits[0][1]
        matches = [chosen]
    else:
        if not question:
            raise ToolError("Pass the user's question (question) or a feature id (feature)")
        hits = search(question, 5)
        if not hits:
            return ToolResult({"matches": [], "note": "Nothing in the app guide matches. Ask the user what they want "
                               "to do, or do it for them with the other tools."}, "guide: no match")
        matches = [f for _, f in hits]
        top, second = hits[0][0], (hits[1][0] if len(hits) > 1 else 0.0)
        chosen = matches[0] if top >= 2.0 and top >= 1.3 * second else None
    if chosen and show:
        ctx.emit("ui", {"action": "guide", "feature": chosen.id, "label": chosen.name, "where": chosen.where,
                        "nav": chosen.nav})
    note = "Tell the user where it is in words (the 'where' path) and in a sentence how to use it."
    if not chosen:
        note += " Several features match: pick the one the user means and call again with feature=<id>, or ask."
    return ToolResult({"matches": [m.brief() for m in matches], "opened": chosen.id if chosen and show else None,
                       "note": note}, f"guide: {chosen.name}" if chosen else f"guide: {len(matches)} matches")


# --------------------------------------------------------------------------- look_at_photos
MAX_LOOK_PHOTOS = 4
PHOTO_SIDE = 1280


def photo_data_url(path, side: int = PHOTO_SIDE) -> str:
    """A photo as a JPEG data URL no larger than `side` px (phone photos are far bigger than a vision model needs),
    upright according to its EXIF orientation."""
    import base64
    from io import BytesIO

    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((side, side))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def h_look_at_photos(ctx: ToolContext, args: dict) -> ToolResult:
    if not ctx.settings.get("vision"):
        raise ToolError("Vision is switched off in the assistant settings (Settings > Assistant), so photos cannot "
                        "be shown to you - tell the user")
    ws = ctx.workspace
    ids = [str(i) for i in (args.get("asset_ids") or []) if i]
    if not ids:
        current = (ctx.ui or {}).get("project_id")
        choice = args.get("project")
        pid = resolve_project(ws, choice)["id"] if choice else current
        photos = [m for m in ws.list() if m.get("kind") == "image" and (not pid or m.get("project") == pid)]
        photos.sort(key=lambda m: m.get("created") or "", reverse=True)
        ids = [m["id"] for m in photos[:MAX_LOOK_PHOTOS]]
        if not ids:
            raise ToolError("There are no photos in this project yet. The user can attach photos of the part in the "
                            "chat (they are kept in the project) or add them with Add > Open files")
    if len(ids) > MAX_LOOK_PHOTOS:
        raise ToolError(f"Look at {MAX_LOOK_PHOTOS} photos at most at a time")
    images, shown = [], []
    for aid in ids:
        try:
            meta = ws.get(aid)
        except KeyError:
            raise ToolError(f"No asset {aid}")
        if meta.get("kind") != "image":
            raise ToolError(f"'{meta['name']}' ({aid}) is not a photo")
        name = prompt_safe(meta["name"], 60)
        images.append({"url": photo_data_url(ws.image_path(aid)), "caption": f"photo '{name}' ({aid})"})
        shown.append({"id": aid, "name": name, "size_px": [meta["stats"].get("width"), meta["stats"].get("height")]})
    return ToolResult({"photos": shown, "note": "The photos follow as images in the next message; compare them with "
                       "the scans as asked. Photos show shape and features, not calibrated sizes."},
                      f"looking at {len(images)} photo{'s' if len(images) > 1 else ''}", ids, images=images)


# --------------------------------------------------------------------------- workspace_overview / describe_part
def h_workspace_overview(ctx: ToolContext, args: dict) -> ToolResult:
    ws = ctx.workspace
    metas = ws.list()
    summaries = ws.project_summaries(metas)
    current = (ctx.ui or {}).get("project_id")
    known = {p["id"] for p in summaries}
    choice = args.get("project")
    if choice == "all":
        pid = None
    elif choice:
        pid = resolve_project(ws, choice)["id"]
    else:
        pid = current if current in known else None
    kinds = [args["kind"]] if args.get("kind") else None
    query = str(args.get("name_contains") or "").lower()
    rows = [m for m in metas if (pid is None or m.get("project") == pid) and (not kinds or m["kind"] in kinds)
            and (not query or query in m["name"].lower() or m["id"].startswith(query))]
    try:
        limit = max(1, min(int(args.get("limit") or 40), 200))
    except (TypeError, ValueError):
        raise ToolError("limit must be a whole number")
    omitted = max(0, len(rows) - limit)
    names = {p["id"]: p["name"] for p in summaries}
    lines = ["id | name | kind | size | part L×W×H mm | op<-parents" + (" | project" if pid is None else "")]
    from .tools import count_text

    for m in rows[omitted:]:
        cells = [m["id"], prompt_safe(m["name"], 60), m["kind"], count_text(m),
                 part_dims(ws, m) if m["kind"] != "image" else "",
                 (m.get("operation") or "") + ("<-" + ",".join(p[:6] for p in m.get("parents", [])) if m.get("parents")
                                               else "")]
        if pid is None:
            cells.append(prompt_safe(names.get(m.get("project"), "?"), 30))
        lines.append(" | ".join(cells))
    table = "\n".join(lines)
    projects = [{"id": p["id"], "name": prompt_safe(p["name"], 40), "assets": p["counts"]["total"],
                 **({"current": True} if p["id"] == current else {})} for p in summaries[:12]]
    data = {"projects": projects, "showing": names.get(pid, "all projects") if pid else "all projects",
            "assets": table, "units": "mm"}
    while len(shrink(data).encode()) > RESULT_LIMIT - 60 and table.count("\n") > 1:
        head, _, rest = table.partition("\n")
        table = head + "\n" + rest.partition("\n")[2]
        omitted += 1
        data["assets"] = table
    if omitted:
        data["note"] = f"{omitted} older asset(s) not shown: filter with project, kind or name_contains"
    return ToolResult(data, f"{table.count(chr(10))} asset(s) in {data['showing']}")


def merge_caveat_for(ws, asset_id: str) -> str | None:
    """The caveat of the nearest merge this asset was made from (itself included), if its pre-merge check flagged
    an ambiguous pose or a doubled skin (register.merge_caveat)."""
    seen, queue = set(), [asset_id]
    while queue:
        aid = queue.pop(0)
        if aid in seen:
            continue
        seen.add(aid)
        try:
            meta = ws.get(aid)
        except Exception:
            continue
        if meta.get("operation") == "merge":
            caveat = (ws.report(aid) or {}).get("caveat") or {}
            return caveat.get("sentence")
        queue += list(meta.get("parents") or [])
    return None


def h_describe_part(ctx: ToolContext, args: dict) -> ToolResult:
    from .. import understand as U

    meta = _target_asset(ctx, args)
    try:
        s = U.get_summary(ctx.workspace, meta["id"])
    except ValueError as exc:
        raise ToolError(str(exc))
    frame = s["part_frame"]
    axes = {}
    for name in ("length", "width", "height"):
        hint = U.world_hint(frame[name])
        axes[name] = hint or [round(v, 3) for v in frame[name]]
    data = {"id": meta["id"], "name": prompt_safe(meta["name"], 60), "kind": s["kind"], "units": "mm",
            "description": s["description"], "dimensions": _rounded(s["dimensions"]),
            "dimensions_raw": _rounded(s["dimensions_raw"]), "axes": axes, "points": s["count"],
            "spacing": num(s["spacing"]), "noise": num(s.get("noise")),
            "cylinders": [{"diameter": num(c["diameter"]), "length": num(c["length"]), "along": c["axis_name"],
                           "span_mm": [num(c["span"]["from"]), num(c["span"]["to"])], "rms": num(c["rms"]),
                           "coverage_deg": c["coverage_deg"]} for c in s["features"]["cylinders"]],
            "flat_faces": [{"label": p["label"], "area_mm2": round(p["area_mm2"]), "flatness": num(p["flatness"])}
                           for p in s["features"]["planes"]],
            "note": "dimensions = robust extents along the part axes (0.05 % trimmed); raw = full extents. "
                    "Face-to-face sizes: measure kind=caliper."}
    caveat = merge_caveat_for(ctx.workspace, meta["id"])
    if caveat:
        data["measurement_caveat"] = caveat
    if args.get("profile"):
        data["profile_along_length"] = "; ".join(
            f"{x['from']:.1f}-{x['to']:.1f}: {_fmt(x['width'])}x{_fmt(x['height'])}"
            for x in s["profile"]["slices"] if x["width"] is not None)
    return ToolResult(data, s["description"], [meta["id"]])


# --------------------------------------------------------------------------- select_region
def h_select_region(ctx: ToolContext, args: dict) -> ToolResult:
    import numpy as np

    from ..regions import describe_region, positions, region_mask, resolve_region

    meta = _target_asset(ctx, args)
    region = _region(ctx, args, required=True)
    geom, frame = _geometry(ctx, meta)
    try:
        resolved = resolve_region(geom, region, frame)
        mask = region_mask(geom, region, frame)
    except ValueError as exc:
        raise ToolError(str(exc))
    count, total = int(mask.sum()), int(len(mask))
    event = {"action": "highlight", "asset_id": meta["id"], "resolved": resolved, "region": region,
             "label": prompt_safe(args.get("label") or describe_region(region), 60)}
    if args.get("color"):
        event["color"] = str(args["color"])[:20]
    ctx.emit("ui", event)
    pts = np.asarray(positions(geom))[mask]
    data = {"asset_id": meta["id"], "region": describe_region(region), "count": count, "total": total,
            "percent": num(100.0 * count / max(total, 1)),
            "bbox": {"min": nums(pts.min(axis=0).tolist()), "max": nums(pts.max(axis=0).tolist())} if count else None,
            "shown": "highlighted in the viewer"}
    if count == 0:
        data["warning"] = "The region contains no points of this model - check its position and size"
    else:
        data["next"] = "Use the same region object in edit ops (delete_region, keep_region, smooth, paint...) or measure"
    return ToolResult(data, f"Region on {meta['name']}: {count:,} of {total:,} points", [meta["id"]])


# --------------------------------------------------------------------------- measure
def _measure_points(ctx, meta, args) -> tuple[dict, str]:
    from ..edit import Snapper, measure

    pts = args.get("points")
    if not isinstance(pts, list) or not 1 <= len(pts) <= 20 or any(not isinstance(p, list) or len(p) != 3 for p in pts):
        raise ToolError("points must be a list of 1-20 [x, y, z] coordinates")
    geom, _ = _geometry(ctx, meta)
    try:
        raw = measure(Snapper(geom), pts)
    except (ValueError, TypeError) as exc:
        raise ToolError(f"Measure failed: {exc}")
    result = {"snapped": [nums(p) for p in raw.get("points", [])], "snap_distance": nums(raw.get("snap_distances", [])),
              "segments": nums(raw.get("segments", [])), "total": num(raw.get("total"))}
    if len(result["snapped"]) == 2:
        result["delta_xyz"] = nums([b - a for a, b in zip(raw["points"][0], raw["points"][1])])
    if "angle_deg" in raw:
        result["angle_deg"] = num(raw["angle_deg"])
    if len(raw.get("points", [])) >= 2:
        _overlay(ctx, "points", {"points": raw["points"], "total": raw["total"]}, meta["id"])
    return result, f"total {result['total']} mm"


def _measure_distance(ctx, meta, args) -> tuple[dict, str]:
    from ..analysis import point_distance
    from ..edit import Snapper

    pts = args.get("points")
    if not isinstance(pts, list) or len(pts) != 2 or any(not isinstance(p, list) or len(p) != 3 for p in pts):
        raise ToolError("kind=distance needs points: exactly two [x, y, z]")
    geom, frame = _geometry(ctx, meta)
    axis = args.get("direction")
    if isinstance(axis, str) and axis.lower() in ("length", "width", "height"):
        axis = frame.axis(axis.lower()).tolist()
    snapped, dist = Snapper(geom).snap(pts)
    try:
        out = point_distance(snapped, axis)
    except ValueError as exc:
        raise ToolError(str(exc))
    out["points"] = snapped.tolist()
    _overlay(ctx, "distance", out, meta["id"])
    result = _rounded({k: out[k] for k in ("distance", "delta", "dx", "dy", "dz", "along_axis", "perpendicular")
                       if k in out})
    result["snap_distances"] = nums(dist.tolist())
    return result, f"distance {num(out['distance'])} mm"


def _measure_thread(ctx, meta, args) -> tuple[dict, str]:
    from .tools import thread_result

    region = _region(ctx, args)
    geom, frame = _geometry(ctx, meta)
    if region is not None:
        from ..regions import region_mask, subset

        try:
            geom = subset(geom, region_mask(geom, region, frame))
        except ValueError as exc:
            raise ToolError(str(exc))
    result, summary, raw = thread_result(geom)
    crests = raw.get("crest_points") or []
    if len(crests) >= 2:
        ctx.emit("ui", {"action": "measure_overlay", "replace": False, "asset_id": meta["id"],
                        "items": [{"label": "pitch", "kind": "thread", "value": round(raw["pitch"], 6), "unit": "mm",
                                   "a": crests[0], "b": crests[1], "asset_id": meta["id"]}]})
    ctx.emit("ui", {"action": "navigate", "step": "measure"})
    return result, summary


def h_measure(ctx: ToolContext, args: dict) -> ToolResult:
    from .. import understand as U

    kind = args.get("kind") or ("points" if args.get("points") else None)
    if kind not in MEASURE_KINDS:
        raise ToolError(f"kind must be one of {', '.join(MEASURE_KINDS)}")
    meta = _target_asset(ctx, args)
    if kind == "points":
        result, summary = _measure_points(ctx, meta, args)
    elif kind == "distance":
        result, summary = _measure_distance(ctx, meta, args)
    elif kind == "thread":
        result, summary = _measure_thread(ctx, meta, args)
    else:
        geom, frame = _geometry(ctx, meta)
        spacing = (meta.get("stats") or {}).get("spacing")
        try:
            if kind in ("extent", "caliper"):
                if args.get("direction") is None:
                    raise ToolError(f"kind={kind} needs direction: {DIRECTION_HELP}")
                region = _region(ctx, args)
                raw = (U.measure_extent(geom, args["direction"], region, frame) if kind == "extent"
                       else U.measure_caliper(geom, args["direction"], region, frame, spacing))
            elif kind == "faces":
                raw = U.measure_faces(geom, args.get("direction") or "length", _region(ctx, args), frame, spacing)
            elif kind == "diameter":
                raw = U.measure_diameter(geom, _region(ctx, args), args.get("direction"), frame)
            elif kind == "sphere":
                raw = U.measure_sphere(geom, _region(ctx, args, required=True), frame)
            elif kind == "plane":
                raw = U.measure_plane(geom, _region(ctx, args, required=True), frame)
            elif kind == "angle":
                raw = U.measure_angle(geom, _region(ctx, args, required=True),
                                      _region(ctx, args, "region_b", required=True), frame)
            else:  # section
                plane = args.get("plane")
                if plane is None:
                    if args.get("direction") is None or args.get("at") is None:
                        raise ToolError("kind=section needs plane {point, normal} or direction + at (mm from the "
                                        "part's start along that direction)")
                    plane = {"direction": args["direction"], "at": args["at"]}
                raw = U.measure_section(geom, plane, frame, spacing)
        except ValueError as exc:
            raise ToolError(str(exc))
        _overlay(ctx, kind, raw, meta["id"])
        result, summary = _compact_measure(kind, raw)
    result = {"asset": meta["id"], "measure": kind, **result, "units": "mm"}
    return ToolResult(result, f"{kind} on {meta['name']}: {summary}", [meta["id"]])


def _compact_measure(kind: str, r: dict) -> tuple[dict, str]:
    """The numbers the model needs (the full result is drawn in the viewer)."""
    warn = {"warnings": r["warnings"]} if r.get("warnings") else {}
    if kind == "extent":
        return _rounded({"length": r["length"], "robust_length": r["robust_length"], "min": r["min"],
                         "max": r["max"], "direction": r["direction_name"], "frame": r["frame"],
                         "points_used": r["points_used"]}), f"{num(r['length'])} mm along {r['direction_name']}"
    if kind == "faces":
        faces = [{"face": k + 1, "at": f["position"], "is": f.get("label"), "tilt_deg": f["tilt_deg"],
                  "rms": f["rms"], "ring_diameter": [2 * f["radius_range"][0], 2 * f["radius_range"][1]]}
                 for k, f in enumerate(r["faces"])]
        key = [{"what": k["what"], "faces": [k["from"] + 1, k["to"] + 1], "distance": k["distance"]}
               for k in r.get("key_distances", [])]
        return _rounded({"direction": r["direction_name"], "faces": faces, "key_distances": key, **warn}), \
            "; ".join(f"{k['what'].split(' (')[0]} {num(k['distance'])} mm" for k in key) or f"{len(faces)} face"
    if kind == "caliper":
        faces = [{"kind": f["kind"], "rms": num(f.get("rms")), "points": f.get("points")}
                 for f in (r["face_a"], r["face_b"])]
        return _rounded({"distance": r["distance"], "parallelism_deg": r["parallelism_deg"], "faces": faces,
                         "extent": r["extent"], "direction": r["direction_name"], "points_used": r["points_used"],
                         **warn}), f"{num(r['distance'])} mm across {r['direction_name']}"
    if kind == "diameter":
        return _rounded({"fit": r["kind"], "diameter": r["diameter"], "radius": r["radius"], "axis": r["axis"],
                         "center": r["center"], "length": r["length"], "rms": r["rms"],
                         "coverage_deg": r["coverage_deg"], "points_used": r["points_used"], "inliers": r["inliers"],
                         **warn}), f"Ø {num(r['diameter'])} mm ({r['kind']})"
    if kind == "sphere":
        return _rounded({"diameter": r["diameter"], "radius": r["radius"], "center": r["center"], "rms": r["rms"],
                         "coverage": r["coverage"], "points_used": r["points_used"], **warn}), \
            f"sphere Ø {num(r['diameter'])} mm"
    if kind == "plane":
        return _rounded({"normal": r["normal"], "point": r["point"], "flatness": r["flatness"], "rms": r["rms"],
                         "points_used": r["points_used"], **warn}), f"flatness {num(r['flatness'])} mm"
    if kind == "angle":
        out = {"angle_deg": r["angle_deg"], "supplement_deg": r["supplement_deg"],
               "fit_a": {k: r["fit_a"][k] for k in ("type", "rms", "points_used")},
               "fit_b": {k: r["fit_b"][k] for k in ("type", "rms", "points_used")}}
        if "normals_angle_deg" in r:
            out["normals_angle_deg"] = r["normals_angle_deg"]
        return _rounded(out), f"{num(r['angle_deg'])} deg"
    return _rounded({"width": r["width"], "height": r["height"], "plane": r["plane_label"],
                     "curves": len(r["polylines"]), "closed": r["closed"][:10], "curve_length": r["length"],
                     "points_used": r["points_used"]}), f"section {num(r['width'])} x {num(r['height'])} mm"


# --------------------------------------------------------------------------- projects
def h_project(ctx: ToolContext, args: dict) -> ToolResult:
    ws = ctx.workspace
    action = args.get("action")
    if action not in PROJECT_ACTIONS:
        raise ToolError(f"action must be one of {', '.join(PROJECT_ACTIONS)}")
    current = (ctx.ui or {}).get("project_id")
    if action == "list":
        rows = [{"id": p["id"], "name": prompt_safe(p["name"], 60), **p["counts"], "updated": p["updated"][:16],
                 **({"current": True} if p["id"] == current else {})} for p in ws.project_summaries()]
        return ToolResult({"projects": rows}, f"{len(rows)} project(s)")
    if action == "create":
        try:
            p = ws.create_project(str(args.get("name") or ""), str(args.get("description") or ""))
        except ValueError as exc:
            raise ToolError(str(exc))
        return ToolResult({"ok": True, "project": {"id": p["id"], "name": p["name"]},
                           "next": "Open it with project action=open if the user wants to work in it"},
                          f"Created project '{p['name']}'")
    project = resolve_project(ws, args.get("project"))
    if action == "open":
        ctx.emit("ui", {"action": "project", "project_id": project["id"]})
        return ToolResult({"ok": True, "opened": {"id": project["id"], "name": project["name"]}},
                          f"Opened project '{project['name']}'")
    if action == "rename":
        fields = {k: args[k] for k in ("name", "description", "cover_asset_id") if args.get(k) is not None}
        if not fields:
            raise ToolError("rename needs name, description or cover_asset_id")
        if "cover_asset_id" in fields:
            fields["cover_asset_id"] = resolve_asset(ws, fields["cover_asset_id"])["id"]
        try:
            p = ws.update_project(project["id"], **fields)
        except (ValueError, KeyError) as exc:
            raise ToolError(str(exc))
        return ToolResult({"ok": True, "project": {"id": p["id"], "name": p["name"]}}, f"Updated '{p['name']}'")
    metas = _ids(ctx, args.get("asset_ids"))
    for m in metas:
        ws.move_asset(m["id"], project["id"])
    return ToolResult({"ok": True, "moved": [m["id"] for m in metas], "to": project["name"]},
                      f"Moved {len(metas)} asset(s) to '{project['name']}'", [m["id"] for m in metas])


# --------------------------------------------------------------------------- capture
def _capture_manager(ctx: ToolContext):
    from ..web.routes_capture import manager_for

    manager = manager_for(ctx.workspace.root)
    if manager is None:
        raise ToolError("Live capture is not available in this server")
    return manager


def capture_running(ws) -> bool:
    try:
        from ..web.routes_capture import manager_for

        manager = manager_for(ws.root)
    except Exception:
        return False
    session = getattr(manager, "session", None) if manager is not None else None
    return bool(session is not None and session.state == "running")


def capture_brief(status: dict) -> dict:
    """The capture status in a few numbers (guidance messages, holes and coverage included)."""
    if not status.get("active"):
        return {"state": status.get("state", "idle"), "session": None}
    out = {k: status.get(k) for k in ("state", "driver", "driver_name", "points", "fps", "elapsed_s", "tracking",
                                      "pending_scans", "unsaved", "saved_asset_id", "error") if status.get(k) is not None}
    g = status.get("guidance") or {}
    if g:
        out["guidance"] = (g.get("status") or {}).get("message")
        codes = [m.get("code") for m in g.get("messages") or [] if m.get("severity") in ("warning", "error")]
        if codes:
            out["issues"] = codes[:6]
        cov = g.get("coverage") or {}
        if cov.get("completeness") is not None:
            out["completeness_pct"] = num(100 * cov["completeness"])
        holes = g.get("holes") or []
        if holes:
            out["holes"] = [{"hint": prompt_safe(h.get("hint") or h.get("message") or h.get("kind"), 80),
                             "area_mm2": num(h.get("area_mm2")), "look_from": nums(h.get("direction") or [])}
                            for h in holes[:3]]
        dist = g.get("distance") or {}
        if dist.get("state") and dist.get("state") != "ok":
            out["distance"] = dist["state"]
    device = status.get("device") or {}
    if device:
        out["device"] = {k: v for k, v in list(device.items())[:6] if isinstance(v, (int, float, str, bool))}
    return out


def h_capture(ctx: ToolContext, args: dict) -> ToolResult:
    from ..capture.manager import CaptureError

    action = args.get("action")
    if action not in CAPTURE_ACTIONS:
        raise ToolError(f"action must be one of {', '.join(CAPTURE_ACTIONS)}")
    if action == "follow":
        on = args.get("on", True)
        if not isinstance(on, bool):
            raise ToolError("on must be true or false")
        ctx.emit("ui", {"action": "follow_scanner", "on": on})
        return ToolResult({"ok": True, "follow_scanner": on}, f"Follow scanner {'on' if on else 'off'}")
    manager = _capture_manager(ctx)
    try:
        if action == "status":
            data = capture_brief(manager.status())
            tt = turntable_brief(ctx.workspace)
            if tt:
                data["turntable"] = tt
            return ToolResult(data, f"Capture: {data.get('state')}")
        if action == "drivers":
            info = manager.drivers()
            rows = [{"id": d["id"], "name": d["name"], "available": d["available"],
                     **({"reason": prompt_safe(d["reason"], 120)} if not d["available"] and d.get("reason") else {})}
                    for d in info["drivers"]]
            return ToolResult({"drivers": rows, "last_driver": info.get("last_driver")}, f"{len(rows)} drivers")
        if action == "connect":
            driver = str(args.get("driver") or "").strip()
            if not driver:
                raise ToolError("connect needs driver (call capture action=drivers)")
            settings = args.get("settings") or {}
            if not isinstance(settings, dict):
                raise ToolError("settings must be an object")
            status = manager.connect(driver, settings)
            return ToolResult(capture_brief(status), f"Connected {driver}")
        if action in ("start", "pause", "resume", "stop"):
            status = getattr(manager, action)()
            return ToolResult(capture_brief(status), f"Capture {action}")
        if action == "discard":
            status = manager.status()
            if status.get("unsaved") and args.get("user_confirmed") is not True:
                raise ToolError("The capture holds unsaved data that discarding would lose. Ask the user first; call "
                                "again with user_confirmed=true only if they agree.")
            return ToolResult(capture_brief(manager.discard()), "Capture discarded")
        if action == "save":
            project = args.get("project")
            pid = resolve_project(ctx.workspace, project)["id"] if project else None
            if pid is None:
                current = (ctx.ui or {}).get("project_id")
                if current and any(p["id"] == current for p in ctx.workspace.projects()):
                    pid = current
            auto = args.get("auto_process")
            res = manager.save(args.get("name"), auto if isinstance(auto, bool) else None, pid)
            asset = res["asset"]
            data = {"ok": True, "asset": asset_brief(asset), "project": asset.get("project"),
                    "autopilot_job": (res.get("job") or {}).get("id")}
            return ToolResult(data, f"Saved capture '{asset['name']}'", [asset["id"]])
        if action == "marker_map":
            command = args.get("command")
            if command not in MARKER_COMMANDS:
                raise ToolError(f"command must be one of {', '.join(MARKER_COMMANDS)}")
            res = manager.driver_command(command, {})
            return ToolResult({"result": res.get("result"), "capture": capture_brief(res.get("status") or {})},
                              f"Marker map: {command}")
        if action == "pending":
            info = manager.pending()
            rows = [{"scan_id": p["scan_id"], "name": prompt_safe(p["name"], 60), "points": p.get("points"),
                     "recommendation": (p.get("assessment") or {}).get("recommendation"),
                     "headline": prompt_safe((p.get("assessment") or {}).get("headline") or "", 160)}
                    for p in info.get("pending", [])]
            return ToolResult({"pending": rows}, f"{len(rows)} pending scan(s)")
        # decide
        decision = args.get("decision")
        if decision not in PENDING_DECISIONS:
            raise ToolError(f"decision must be one of {', '.join(PENDING_DECISIONS)}")
        res = manager.decide_pending(str(args.get("scan_id") or ""), decision)
        ids = [res["asset"]["id"]] if res.get("asset") else []
        return ToolResult({"ok": True, "decision": decision, "asset": ids[0] if ids else None,
                           "capture": capture_brief(res.get("status") or {})}, f"Pending scan: {decision}", ids)
    except CaptureError as exc:
        raise ToolError(str(exc))
    except ValueError as exc:
        raise ToolError(str(exc))


# --------------------------------------------------------------------------- turntable
def _turntable_manager(ws):
    """The workspace's turntable manager (Contract 4, workstream T). Imported lazily: the package is optional."""
    try:
        from ..capture.turntable.manager import get_turntable_manager
    except ImportError as exc:
        raise ToolError(f"Turntable control is not installed in this build ({exc})")
    return get_turntable_manager(ws)


def turntable_brief(ws, status: dict | None = None) -> dict | None:
    """Compact turntable status, or None when turntable support is not available."""
    if status is None:
        try:
            status = _turntable_manager(ws).status()
        except Exception:
            return None
    out = {k: status.get(k) for k in ("connected", "kind", "name", "angle_deg", "tilt_deg", "moving",
                                      "speed_s_per_rev", "direction", "validated", "error")
           if status.get(k) is not None}
    for k in ("angle_deg", "tilt_deg", "speed_s_per_rev"):
        if k in out:
            out[k] = num(float(out[k]))
    program = status.get("program")
    if isinstance(program, dict) and program:
        out["program"] = {k: program[k] for k in ("state", "rotation", "rotations", "stop", "stops_per_rotation",
                                                  "progress", "tilt_deg", "error") if program.get(k) is not None}
        extra = [prompt_safe(m, 160) for m in (program.get("warnings") or []) + (program.get("notes") or [])]
        if extra:
            out["program"]["notes"] = extra[:4]
    move = status.get("move") or status.get("last_move") or {}
    if isinstance(move, dict) and (move.get("note") or move.get("error")):
        out["move"] = {k: prompt_safe(move[k], 160) for k in ("state", "note", "error") if move.get(k)}
    caps = status.get("capabilities") or {}
    if caps:
        out["capabilities"] = {k: caps[k] for k in ("tilt", "tilt_range", "speed_range", "interval_range") if k in caps}
    return out


def _number(args: dict, key: str, required: bool = True):
    value = args.get(key)
    if value is None:
        if required:
            raise ToolError(f"{key} is required (a number)")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolError(f"{key} must be a number")
    return float(value)


async def _wait_motion(ctx: ToolContext, manager, label: str) -> dict:
    """Poll until the turntable stops moving; the user's stop button stops the turntable too."""
    start = time.monotonic()
    status = await asyncio.to_thread(manager.status)
    while status.get("moving"):
        if ctx.stopped:
            await asyncio.to_thread(manager.stop)
            raise ToolCancelled("Stopped by the user; the turntable was stopped")
        if time.monotonic() - start > MOVE_TIMEOUT_S:
            raise ToolError(f"The turntable is still moving after {MOVE_TIMEOUT_S:.0f} s - check it")
        ctx.emit("tool_progress", {"call_id": ctx.call_id, "job_id": None, "status": "running", "progress": None,
                                   "label": f"{label}: angle {num(float(status.get('angle_deg') or 0))} deg, tilt "
                                            f"{num(float(status.get('tilt_deg') or 0))} deg", "last_log": None})
        await asyncio.sleep(max(ctx.poll_interval, 0.1))
        status = await asyncio.to_thread(manager.status)
    return status


def _tilt_guard(ctx: ToolContext, args: dict, what: str) -> None:
    if capture_running(ctx.workspace) and args.get("user_confirmed") is not True:
        raise ToolError(f"A scan is running: {what} tilts the part relative to the scanner mid-scan, which can break "
                        "tracking. Ask the user first; call again with user_confirmed=true only if they agree (or pause "
                        "the capture first).")


async def h_turntable(ctx: ToolContext, args: dict) -> ToolResult:
    action = args.get("action")
    if action not in TURNTABLE_ACTIONS:
        raise ToolError(f"action must be one of {', '.join(TURNTABLE_ACTIONS)}")
    manager = _turntable_manager(ctx.workspace)
    call = lambda fn, *a, **k: asyncio.to_thread(fn, *a, **k)  # noqa: E731 - device I/O off the event loop
    try:
        if action == "status":
            status = await call(manager.status)
        elif action == "devices":
            seconds = _number(args, "scan_seconds", required=False) or 4.0
            devices = await call(manager.devices, min(max(seconds, 1.0), 15.0))
            rows = [{k: d.get(k) for k in ("id", "name", "kind", "rssi")} for d in devices[:12]]
            return ToolResult({"devices": rows}, f"{len(rows)} turntable(s) found")
        elif action == "connect":
            kind = args.get("kind") or "auto"
            status = await call(manager.connect, args.get("device") or None, kind)
        elif action == "disconnect":
            status = await call(manager.disconnect)
        elif action == "stop":
            status = await call(manager.stop)
        elif action == "speed":
            status = await call(manager.set_speed, _number(args, "speed_s_per_rev"))
        elif action == "rotate":
            degrees = _number(args, "degrees")
            speed = _number(args, "speed_s_per_rev", required=False)
            await call(manager.rotate, degrees, speed, False)
            status = await _wait_motion(ctx, manager, f"rotating {num(degrees)} deg")
        elif action == "tilt":
            degrees = _number(args, "degrees")
            _tilt_guard(ctx, args, "tilting the turntable")
            await call(manager.tilt, degrees, False)
            status = await _wait_motion(ctx, manager, f"tilting to {num(degrees)} deg")
        elif action == "program":
            program = {"interval_deg": _number(args, "interval_deg")}
            for key in ("frames_per_stop", "speed_s_per_rev"):
                value = _number(args, key, required=False)
                if value is not None:
                    program[key] = int(value) if key == "frames_per_stop" else value
            if args.get("direction") is not None:
                if args["direction"] not in ("cw", "ccw"):
                    raise ToolError("direction must be cw or ccw")
                program["direction"] = args["direction"]
            rotations = args.get("rotations")
            if rotations is not None:
                if not isinstance(rotations, list) or not 1 <= len(rotations) <= 5 or not all(
                        isinstance(r, dict) and isinstance(r.get("tilt_deg", 0), (int, float)) for r in rotations):
                    raise ToolError("rotations must be 1-5 objects like {\"tilt_deg\": 15}")
                program["rotations"] = [{"tilt_deg": float(r.get("tilt_deg", 0))} for r in rotations]
                if any(r["tilt_deg"] for r in program["rotations"]):
                    _tilt_guard(ctx, args, "a program with tilted rotations")
            if args.get("sync_scan") is not None:
                program["sync_scan"] = bool(args["sync_scan"])
            status = await call(manager.start_program, program)
        else:  # stop_program
            status = await call(manager.stop_program)
    except (ToolError, ToolCancelled):
        raise
    except Exception as exc:  # the device layer raises its own errors: pass the sentence on
        raise ToolError(f"Turntable: {exc}")
    brief = turntable_brief(ctx.workspace, status or {})
    if brief is not None and brief.get("validated") is False:
        brief["note"] = "the Bluetooth protocol is not yet confirmed on the real device"
    return ToolResult(brief or {}, f"Turntable {action}")


# --------------------------------------------------------------------------- mesh holes (cloudclean.holes)
async def h_holes(ctx: ToolContext, args: dict) -> ToolResult:
    """list: the holes of a mesh (id, diameter, area, centre; the open scan's outer rim is marked outer); fill: job
    fill_holes_selected (hole_ids, or every non-outer hole up to max_diameter; creates a new mesh)."""
    from .tools import _job_result, run_job

    action = args.get("action") or "list"
    if action not in ("list", "fill"):
        raise ToolError("action must be list or fill")
    meta = _target_asset(ctx, args, ("mesh",))
    if action == "list":
        from ..holes import describe_group, fill_splits, find_holes, hole_groups

        geom, _ = _geometry(ctx, meta)
        try:
            found = find_holes(geom)
        except ValueError as exc:
            raise ToolError(str(exc))
        holes = found["holes"]
        groups = hole_groups(holes)
        lo = _number(args, "min_diameter") if args.get("min_diameter") is not None else 0.0
        hi = _number(args, "max_diameter") if args.get("max_diameter") else float("inf")
        offset = max(0, int(args.get("offset") or 0))
        pick = [h for h in holes if lo <= h["diameter"] <= hi]        # biggest first
        rows = [{"id": h["id"], "diameter": num(h["diameter"]), "center": nums([round(c, 1) for c in h["center"]]),
                 **({"outer": True} if h["outer"] else {})} for h in pick[offset:offset + 12]]
        inner = [h for h in holes if not h["outer"]]
        data = {"asset": meta["id"], "units": "mm", "holes_to_fill": len(inner),
                "outer_rim": any(h["outer"] for h in holes), "watertight": found["watertight"],
                "size_groups": [{"what": describe_group(g), "loops": g["count"], "round": g["round"],
                                 **({"note": g["note"]} if g.get("note") else {})} for g in groups],
                "fill_options": [{"call": f"holes action=fill max_diameter={s['max_diameter']}",
                                  "fills": s["fills_what"], "keeps": s["keeps_what"]}
                                 for s in fill_splits(groups)],
                "holes": rows, "shown": f"{len(rows)} of {len(pick)} (biggest first; offset/min_diameter/"
                                        "max_diameter page through them)"}
        data["note"] = ("Pick the fill_option whose 'keeps' matches what the user wants to keep (a through hole "
                        "has 2 loops, one per face; the user counts it once). It fills everything smaller in one "
                        "call - you do not need hole ids.")
        summary = f"{len(inner)} hole loop(s) in {meta['name']}" + (
            ": " + ", ".join(describe_group(g) for g in groups[:4]) if groups else "")
        return ToolResult(data, summary, [meta["id"]])
    payload = {"asset_id": meta["id"], "name": args.get("name") or None}
    if args.get("hole_ids") is not None:
        ids = args["hole_ids"]
        if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
            raise ToolError("hole_ids must be a list of hole ids from holes action=list")
        payload["hole_ids"] = ids
    if args.get("max_diameter") is not None:
        payload["max_diameter"] = _number(args, "max_diameter")
    if args.get("min_diameter") is not None:
        payload["min_diameter"] = _number(args, "min_diameter")
    if args.get("except_ids") is not None:
        ids = args["except_ids"]
        if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
            raise ToolError("except_ids must be a list of hole ids from holes action=list")
        payload["except_ids"] = ids
    job = await run_job(ctx, "fill_holes_selected", f"Fill holes of {meta['name']}", payload)
    return _job_result(ctx, job, "Filled holes", "new triangles close the holes; existing vertices are not moved")


# --------------------------------------------------------------------------- ui (Contract 7)
UI_ACTIONS = ["show", "focus", "select", "display", "camera", "navigate", "open", "tool", "section",
              "clear_highlight", "measure_overlay", "annotate", "clear_annotations", "follow_scanner", "layout",
              "project"]
COLOR_MODES = ["original", "solid", "asset", "normal", "scalar", "height"]
VIEWS = ["front", "back", "left", "right", "top", "bottom", "iso", "fit"]
PANELS = ["process", "inspect", "capture", "autopilot"]
SCREENS = ["home", "workspace"]
STEPS = ["capture", "clean", "align", "mesh", "measure", "export"]
TOOLS_UI = ["navigate", "box", "lasso", "measure", "brush", "pivot"]
PROJECTIONS = ["perspective", "orthographic"]
THEMES = ["paper", "carbon", "system"]           # the app's themes: paper (light), carbon (dark)
THEME_ALIASES = {"light": "paper", "dark": "carbon"}


def _vec3(value, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ToolError(f"{name} must be [x, y, z]")
    return [float(v) for v in value]


def _bool(args: dict, key: str, event: dict) -> None:
    if args.get(key) is not None:
        if not isinstance(args[key], bool):
            raise ToolError(f"{key} must be true or false")
        event[key] = args[key]


def _choice(args: dict, key: str, choices: list, event: dict) -> None:
    if args.get(key) is not None:
        if args[key] not in choices:
            raise ToolError(f"{key} must be one of {', '.join(choices)}")
        event[key] = args[key]


def h_ui(ctx: ToolContext, args: dict) -> ToolResult:
    action = args.get("action")
    if action not in UI_ACTIONS:
        raise ToolError(f"action must be one of {', '.join(UI_ACTIONS)}")
    event: dict = {"action": action}
    ws = ctx.workspace
    if action in ("show", "select"):
        metas = _ids(ctx, args.get("asset_ids"))
        event["asset_ids"] = [m["id"] for m in metas]
        if action == "show":
            event["exclusive"] = bool(args.get("exclusive", True))
    elif action == "focus":
        event["asset_id"] = resolve_asset(ws, args.get("asset_id"))["id"]
    elif action == "display":
        _choice(args, "color_mode", COLOR_MODES, event)
        if event.get("color_mode") == "scalar":
            if not args.get("scalar"):
                raise ToolError("color_mode=scalar needs scalar (e.g. 'deviation')")
            event["scalar"] = str(args["scalar"])
        if args.get("point_scale") is not None:
            scale = _number(args, "point_scale")
            if scale <= 0:
                raise ToolError("point_scale must be > 0")
            event["point_scale"] = scale
        for key in ("wireframe", "show_grid", "show_box"):
            _bool(args, key, event)
        _choice(args, "projection", PROJECTIONS, event)
        if args.get("theme") is not None:
            _choice({"theme": THEME_ALIASES.get(args["theme"], args["theme"])}, "theme", THEMES, event)
        if len(event) == 1:
            raise ToolError("display needs color_mode, point_scale, wireframe, show_grid, show_box, projection or "
                            "theme")
    elif action == "camera":
        _choice(args, "view", VIEWS, event)
        if args.get("orbit") is not None:
            orbit = args["orbit"]
            if not isinstance(orbit, dict) or not set(orbit) <= {"yaw_deg", "pitch_deg"} or not orbit:
                raise ToolError("orbit must be {yaw_deg, pitch_deg}")
            event["orbit"] = {k: _number(orbit, k) for k in orbit}
        if args.get("zoom") is not None:
            zoom = _number(args, "zoom")
            if zoom <= 0:
                raise ToolError("zoom must be > 0 (>1 = closer)")
            event["zoom"] = zoom
        if args.get("look_at") is not None:
            event["look_at"] = _vec3(args["look_at"], "look_at")
        if args.get("asset_ids"):
            event["asset_ids"] = [m["id"] for m in _ids(ctx, args["asset_ids"])]
        if len(event) == 1:
            raise ToolError("camera needs view, orbit, zoom, look_at or asset_ids")
    elif action == "navigate":
        _choice(args, "screen", SCREENS, event)
        _choice(args, "step", STEPS, event)
        if args.get("panel"):
            event["panel"] = str(args["panel"])[:40]
        if len(event) == 1:
            raise ToolError("navigate needs screen, step or panel")
    elif action == "open":
        if args.get("panel") not in PANELS:
            raise ToolError(f"panel must be one of {', '.join(PANELS)}")
        event["panel"] = args["panel"]
    elif action == "tool":
        if args.get("tool") not in TOOLS_UI:
            raise ToolError(f"tool must be one of {', '.join(TOOLS_UI)}")
        event["tool"] = args["tool"]
    elif action == "section":
        enabled = args.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ToolError("enabled must be true or false")
        event["enabled"] = enabled
        if enabled:
            if args.get("axis") not in ("x", "y", "z"):
                raise ToolError("axis must be x, y or z")
            event["axis"] = args["axis"]
            event["position"] = _number(args, "position")
            event["flip"] = bool(args.get("flip", False))
    elif action == "measure_overlay":
        items = args.get("items")
        if not isinstance(items, list) or len(items) > 50:
            raise ToolError("items must be a list (at most 50) of {label, value, unit, a, b}")
        clean = []
        for it in items:
            if not isinstance(it, dict):
                raise ToolError("each item must be {label, value, unit, a, b}")
            clean.append({"label": prompt_safe(it.get("label") or "", 60), "kind": str(it.get("kind") or "custom")[:20],
                          "value": _number(it, "value"), "unit": str(it.get("unit") or "mm")[:8],
                          "a": _vec3(it.get("a"), "a"), "b": _vec3(it.get("b"), "b")})
        event.update(items=clean, replace=bool(args.get("replace", False)))
    elif action == "annotate":
        items = args.get("items")
        if not isinstance(items, list) or not 1 <= len(items) <= 50:
            raise ToolError("items must be a list (1-50) of {position: [x, y, z], text}")
        event["items"] = [{"position": _vec3(it.get("position") if isinstance(it, dict) else None, "position"),
                           "text": str(it.get("text") or "")[:200]} for it in items]
    elif action == "follow_scanner":
        on = args.get("on", True)
        if not isinstance(on, bool):
            raise ToolError("on must be true or false")
        event["on"] = on
    elif action == "layout":
        for key in ("assets_open", "panel_open", "assistant_open"):
            _bool(args, key, event)
        if len(event) == 1:
            raise ToolError("layout needs assets_open, panel_open or assistant_open")
    elif action == "project":
        event["project_id"] = resolve_project(ws, args.get("project"))["id"]
    ctx.emit("ui", event)
    ids = event.get("asset_ids") or ([event["asset_id"]] if "asset_id" in event else [])
    return ToolResult("done", f"Viewer: {action}", ids)


# --------------------------------------------------------------------------- schemas
_ID = {"type": "string", "description": "default: the active asset"}
_IDS = {"type": "array", "items": {"type": "string"}}
_REGION_FULL = {"type": "object", "description": "one of {spheres:[[x,y,z,r]..]} {box:{min,max}} "
                                                  "{obb:{center,axes,half}} {cylinder:{point,axis,radius,half_length}} "
                                                  "{slab:{direction,from,to,frame:world|part}} "
                                                  "{end:{direction,side:min|max,length}} {all:true}, +invert:true. "
                                                  "direction: " + DIRECTION_HELP}
_REGION = {"type": "object", "description": "region object (as in select_region)"}
_VEC3 = {"type": "array", "items": {"type": "number"}}
UI_ADVERTISED = [a for a in UI_ACTIONS if a != "open"]   # "open" (legacy panels) still works


def v3_tools() -> list[Tool]:
    from .guide import feature_index

    return [
        Tool("workspace_overview", "Projects and assets with part dimensions L×W×H (mm, along the part's own "
             "axes). Default: current project; project=\"all\" for all.",
             _obj({"project": {"type": "string"}, "kind": {"type": "string", "enum": ["pointcloud", "mesh", "image"]},
                   "name_contains": {"type": "string"}, "limit": {"type": "integer"}}), h_workspace_overview),
        Tool("photos_to_3d", "Make a 3D model (coloured point cloud) from photos of the part saved in the project "
             "(default: all photos of the current project). Photos needed (measured): 12+ about every 30 degrees "
             "all the way round at one height; 24-36 at 2-3 heights rebuild the most; 3-4 give only a rough model. "
             "Takes about 1-2 minutes. Good for the shape, not for measuring holes or diameters (surface about "
             "0.5-1 mm off). Photographed on the printed scale sheet (markers + a 100 mm bar) the model comes out at "
             "true size by itself; pass ruler_mm = the bar's length as the user measured it with a caliper, when "
             "given. Without the sheet the size is unknown until the user gives one known length. If the user "
             "already has a scan of the part and wants it coloured, use colour_from_photos instead.",
             _obj({"photo_ids": {"type": "array", "items": {"type": "string"}}, "name": {"type": "string"},
                   "ruler_mm": {"type": "number", "description": "the printed sheet's 100 mm bar as measured "
                                                                 "(90-110), if the user gave it"}}),
             h_photos_to_3d),
        Tool("colour_from_photos", "Colour the user's scan or mesh with the real colours of the part, from photos of "
             "it saved in the project. Use when the user wants their scan/mesh coloured or textured from photos of "
             "the part. The photos are lined up with the model automatically; its shape and size do not change. "
             "Creates a new coloured model (a mesh also gets a UV texture). Default: all photos of the model's "
             "project. Photos needed (measured): 12+ about every 30 degrees all the way round at one height; 6 every "
             "60 degrees is the fewest that worked; 3-4 photos, big jumps in angle or height, or one side only are "
             "refused. "
             "Takes 1-4 minutes.",
             _obj({"model": {"type": "string", "description": "asset id or name; default: the active model"},
                   "photo_ids": {"type": "array", "items": {"type": "string"}}}),
             h_colour_from_photos),
        Tool("compare_merge_options", "When scans could line up in more than one way (assess_merge says ambiguous "
             "or the user doubts a merge), draw every option of scan 2 on scan 1 and look at them next to the user's "
             "reference photos of the part (use_photos, default true; photo_ids to choose). Two scans: the "
             "reference first. Then merge with options_job_id + option.",
             _obj({"asset_ids": _IDS, "photo_ids": {"type": "array", "items": {"type": "string"}},
                   "use_photos": {"type": "boolean"}, "params": {"type": "object"}}, ["asset_ids"]),
             h_compare_merge_options),
        Tool("app_guide", "The app guide: where a feature is in CloudClean, what it does and how to use it. question = "
             "the user's words (e.g. 'where do I measure the head height'), or feature = an id from this list: "
             + feature_index() + ". Opens that place in the app and highlights it (open=false only looks it up). "
             "In your reply always say the path in words too.",
             _obj({"question": {"type": "string"}, "feature": {"type": "string"}, "open": {"type": "boolean"}}),
             h_app_guide),
        Tool("look_at_photos", "See photos of the real part that are saved in the project (reference photos the "
             "user attached, or imported photos): no asset_ids = the newest photos of the current project (max 4). "
             "They are shown to you as images right after this call.",
             _obj({"asset_ids": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_LOOK_PHOTOS},
                   "project": {"type": "string"}}), h_look_at_photos),
        Tool("describe_part", "Understand a model: description, dimensions, part axes, cylinders and flat faces; "
             "profile=true adds cross-section sizes along the length.",
             _obj({"asset_id": _ID, "profile": {"type": "boolean"}}), h_describe_part),
        Tool("select_region", "Count and highlight a region of a model (show the user what you mean before editing). "
             "Region objects also work in edit ops and measure.",
             _obj({"asset_id": _ID, "region": _REGION_FULL, "use_screen_selection": {"type": "boolean"},
                   "label": {"type": "string"}, "color": {"type": "string"}}), h_select_region),
        Tool("measure", "Measure; drawn in the viewer. kind: points (snap 1-20 points), distance (2 points, "
             "direction?), extent (size along direction), caliper (face-to-face across direction), diameter "
             "(cylinder/circle; direction = axis hint), faces (every flat face across direction, e.g. head "
             "top, shoulder, tip, with the distances between them: head heights, steps, length under a head), "
             "sphere, plane (flatness), angle (region vs region_b), "
             "section (plane, or direction + at mm), thread. region limits it to part of the model.",
             _obj({"kind": {"type": "string", "enum": MEASURE_KINDS}, "asset_id": _ID,
                   "points": {"type": "array", "items": _VEC3}, "direction": {"description": DIRECTION_HELP},
                   "region": _REGION, "region_b": _REGION, "use_screen_selection": {"type": "boolean"},
                   "plane": {"type": "object", "description": "{point, normal}"}, "at": {"type": "number"}},
                  ["kind"]), h_measure),
        Tool("holes", "Holes of a mesh (mm). list: size groups (count, Ø range, round ones, e.g. marker-sticker "
             "spots) with ready fill_options, and the biggest holes (id, Ø, centre; the open scan's rim is 'outer'). "
             "fill: every hole up to max_diameter (0 = all; keeps bigger ones), optionally from min_diameter, "
             "except except_ids, or only hole_ids - makes a new mesh; only when the user asks. List once, then act.",
             _obj({"action": {"type": "string", "enum": ["list", "fill"]}, "asset_id": _ID,
                   "hole_ids": {"type": "array", "items": {"type": "integer"}}, "max_diameter": {"type": "number"},
                   "min_diameter": {"type": "number"}, "except_ids": {"type": "array", "items": {"type": "integer"}},
                   "offset": {"type": "integer"}, "name": {"type": "string"}}, ["action"]), h_holes),
        Tool("project", "Projects group one physical part's assets. list; create{name, description}; open{project}; "
             "rename{project, name?, description?, cover_asset_id?}; move_asset{asset_ids, project}.",
             _obj({"action": {"type": "string", "enum": PROJECT_ACTIONS}, "project": {"type": "string"},
                   "name": {"type": "string"}, "description": {"type": "string"}, "cover_asset_id": {"type": "string"},
                   "asset_ids": _IDS}, ["action"]), h_project),
        Tool("capture", "Live scanning. status (guidance, holes, coverage, turntable); drivers; connect{driver, "
             "settings}; start; pause; resume; stop; discard; save{name, auto_process, project}; "
             "marker_map{command}; follow{on}; pending; decide{scan_id, decision}.",
             _obj({"action": {"type": "string", "enum": CAPTURE_ACTIONS}, "driver": {"type": "string"},
                   "settings": {"type": "object"}, "name": {"type": "string"}, "auto_process": {"type": "boolean"},
                   "project": {"type": "string"}, "command": {"type": "string", "enum": MARKER_COMMANDS},
                   "on": {"type": "boolean"}, "scan_id": {"type": "string"},
                   "decision": {"type": "string", "enum": PENDING_DECISIONS},
                   "user_confirmed": {"type": "boolean"}}, ["action"]), h_capture),
        Tool("turntable", "Turntable. status; devices; connect{device?, kind?}; disconnect; rotate{degrees relative, "
             "+ = clockwise from above, speed_s_per_rev?} (waits); tilt{degrees absolute}; stop; speed{speed_s_per_rev}; "
             "program{interval_deg, "
             "frames_per_stop, direction, speed_s_per_rev, rotations [{tilt_deg}] (max 5), sync_scan}; stop_program.",
             _obj({"action": {"type": "string", "enum": TURNTABLE_ACTIONS}, "degrees": {"type": "number"},
                   "speed_s_per_rev": {"type": "number"}, "device": {"type": "string"}, "kind": {"type": "string"},
                   "scan_seconds": {"type": "number"}, "interval_deg": {"type": "number"},
                   "frames_per_stop": {"type": "integer"}, "direction": {"type": "string", "enum": ["cw", "ccw"]},
                   "rotations": {"type": "array", "items": {"type": "object"}}, "sync_scan": {"type": "boolean"},
                   "user_confirmed": {"type": "boolean"}}, ["action"]), h_turntable),
        Tool("ui", "Viewer. show{asset_ids, exclusive}; focus{asset_id}; select{asset_ids}; display{color_mode, "
             "scalar, point_scale, wireframe, show_grid, show_box, projection, theme}; camera{view, orbit{yaw_deg, "
             "pitch_deg}, zoom (>1 closer), look_at, asset_ids}; navigate{screen, step, panel}; tool{tool}; "
             "section{enabled, axis, position, flip}; clear_highlight; measure_overlay{items [{label,value,unit,a,b}], "
             "replace}; annotate{items [{position,text}]}; clear_annotations; follow_scanner{on}; layout{assets_open, "
             "panel_open, assistant_open}; project{project}.",
             _obj({"action": {"type": "string", "enum": UI_ADVERTISED}, "asset_ids": _IDS,
                   "asset_id": {"type": "string"}, "exclusive": {"type": "boolean"},
                   "color_mode": {"type": "string", "enum": COLOR_MODES}, "scalar": {"type": "string"},
                   "point_scale": {"type": "number"}, "wireframe": {"type": "boolean"}, "show_grid": {"type": "boolean"},
                   "show_box": {"type": "boolean"}, "projection": {"type": "string", "enum": PROJECTIONS},
                   "theme": {"type": "string", "enum": THEMES}, "view": {"type": "string", "enum": VIEWS},
                   "orbit": {"type": "object"}, "zoom": {"type": "number"}, "look_at": _VEC3,
                   "screen": {"type": "string", "enum": SCREENS}, "step": {"type": "string", "enum": STEPS},
                   "panel": {"type": "string"}, "tool": {"type": "string", "enum": TOOLS_UI},
                   "enabled": {"type": "boolean"}, "axis": {"type": "string", "enum": ["x", "y", "z"]},
                   "position": {"type": "number"}, "flip": {"type": "boolean"},
                   "items": {"type": "array", "items": {"type": "object"}}, "replace": {"type": "boolean"},
                   "on": {"type": "boolean"}, "assets_open": {"type": "boolean"}, "panel_open": {"type": "boolean"},
                   "assistant_open": {"type": "boolean"}, "project": {"type": "string"}}, ["action"]), h_ui),
    ]
