"""Job kind `fill_holes_selected`: fill chosen holes of a mesh (see cloudclean/holes.py)."""
from __future__ import annotations

from ..holes import fill_holes
from ..io import is_cloud
from .workspace import Workspace


def job_fill_holes(ws: Workspace, payload: dict, log) -> list[str]:
    asset_id = payload["asset_id"]
    meta = ws.get(asset_id)
    geom = ws.load_geometry(asset_id)
    if is_cloud(geom):
        raise ValueError(f"'{meta['name']}' is a point cloud; holes can only be filled in a mesh")
    ids = payload.get("hole_ids")
    filled, report = fill_holes(geom, [int(i) for i in ids] if ids is not None else None,
                                float(payload.get("max_diameter") or 0.0), log,
                                min_diameter=float(payload.get("min_diameter") or 0.0),
                                except_ids=[int(i) for i in payload.get("except_ids") or []])
    if not report["filled"]:
        raise ValueError("No hole matched: nothing to fill")
    name = payload.get("name") or f"{meta['name']} · {report['filled']} hole{'s' if report['filled'] > 1 else ''} filled"
    params = {k: payload.get(k) for k in ("hole_ids", "max_diameter", "min_diameter", "except_ids")
              if payload.get(k) is not None}
    return [ws.add_geometry(filled, name, "edit", [asset_id], params, {"ops": [{"op": "fill_holes_selected", **report}]})["id"]]


JOBS = {"fill_holes_selected": job_fill_holes}
