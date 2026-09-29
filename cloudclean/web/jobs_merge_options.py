"""Job `merge_options`: the ways two scans can line up, each refined and drawn to a picture (cloudclean/merge_views.py).

Payload {asset_ids: [reference, moving], params?}. Output {options: [{id: "A", name, source, fitness, rmse_mm,
angle_from_best_deg, shift_from_best_mm, transform}], renders_key, asset_ids}; the pictures are
workspace/renders/<renders_key>/<id>.jpg, served by GET /api/merge/options/<renders_key>/<id>.jpg."""
from __future__ import annotations

import uuid

from ..merge_views import merge_options, render_option
from .workspace import Workspace


def job_merge_options(ws: Workspace, payload: dict, log) -> list[str]:
    ids = list(payload.get("asset_ids") or [])
    if len(ids) != 2:
        raise ValueError("Pick exactly two scans: the reference first, then the scan that moves onto it")
    progress = getattr(log, "progress", None) or (lambda *a: None)
    metas = [ws.get(i) for i in ids]
    if any(m["kind"] == "image" for m in metas):
        raise ValueError("Pick two scans, not photos")
    progress(0.05, "Finding the ways the scans can line up")
    ref, moving = ws.load_geometry(ids[0]), ws.load_geometry(ids[1])
    options = merge_options(ref, moving, payload.get("params"), log)
    key = uuid.uuid4().hex[:12]
    folder = ws.root / "renders" / key
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for k, o in enumerate(options):
        letter = chr(ord("A") + k)
        progress(0.7 + 0.3 * k / max(len(options), 1), f"Drawing option {letter}")
        title = (f"Option {letter}: {o['name']} - overlap {o['fitness']:.0%}, gap {o['rmse_mm']:.3f} mm "
                 f"({metas[0]['name']} orange, {metas[1]['name']} blue)")
        render_option(ref, moving, o["T"], title).save(folder / f"{letter}.jpg", quality=85)
        out.append({"id": letter, **{k2: v for k2, v in o.items() if k2 != "T"},
                    "transform": o["T"].round(10).tolist()})
    output = getattr(log, "output", None)
    if output:
        output(options=out, renders_key=key, asset_ids=ids)
    progress(1.0, f"{len(out)} option{'s' if len(out) != 1 else ''}")
    return []


JOBS = {"merge_options": job_merge_options}
