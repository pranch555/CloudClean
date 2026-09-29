"""Job kinds for part understanding (see docs/v3-backend.md).

summary  payload {asset_ids?: [...] (default: every scan / mesh without a valid cached summary), refresh?: bool}
         computes and caches the Contract 3 summary of each asset (summary.json in the asset folder) so the UI and
         the assistant find them ready. Creates no asset; log.output(summaries={id: description}).
"""
from __future__ import annotations

from .. import understand as U
from .workspace import Workspace


def job_summary(ws: Workspace, payload: dict, log) -> list[str]:
    refresh = bool(payload.get("refresh"))
    ids = list(payload.get("asset_ids") or [])
    if not ids:
        ids = [m["id"] for m in ws.list() if m["kind"] != "image"
               and (refresh or U.load_cached_summary(ws, m["id"]) is None)]
    progress = getattr(log, "progress", None)
    done = {}
    for i, asset_id in enumerate(ids):
        if progress:
            progress(i / max(len(ids), 1), f"Summarising {i + 1}/{len(ids)}")
        try:
            meta = ws.get(asset_id)
        except KeyError:
            log(f"{asset_id}: not found - skipped")
            continue
        if meta["kind"] == "image":
            continue
        summary = U.get_summary(ws, asset_id, refresh=refresh, store_part=False)
        done[asset_id] = summary["description"]
        log(f"{meta['name']}: {summary['description']} ({summary.get('seconds', 0):.1f} s)")
    if hasattr(log, "output"):
        log.output(summaries=done)
    if progress:
        progress(1.0, "done")
    return []


JOBS = {"summary": job_summary}
