"""Marker stickers (cloudclean/markers.py) for the web app.

GET  /api/assets/{asset_id}/markers   the sticker holes of a model (cached next to it, per data file)
POST /api/merge/markers-preview {asset_ids}   each scan after the first lined up on it by the stickers they share
"""
from __future__ import annotations

import json
import threading

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..markers import align_by_markers, find_markers


class PreviewReq(BaseModel):
    asset_ids: list[str]


def create_router(workspace, jobs) -> APIRouter:
    router = APIRouter()
    lock = threading.Lock()

    def meta_of(asset_id: str) -> dict:
        try:
            meta = workspace.get(asset_id)
        except KeyError:
            raise HTTPException(404, f"No asset {asset_id}")
        if meta["kind"] == "image":
            raise HTTPException(400, f"'{meta['name']}' is a photo, not a scan")
        return meta

    def source(asset_id: str) -> dict:
        st = (workspace.asset_dir(asset_id) / "data.ply").stat()
        return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}

    def markers_of(asset_id: str) -> dict:
        path = workspace.asset_dir(asset_id) / "markers.json"
        src = source(asset_id)
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("source") == src:
                return cached
        except (OSError, ValueError):
            pass
        found = find_markers(workspace.load_geometry(asset_id))
        result = {"asset_id": asset_id, "markers": found["markers"], "count": len(found["markers"]), "source": src}
        path.write_text(json.dumps(result), encoding="utf-8")
        return result

    @router.get("/api/assets/{asset_id}/markers")
    def markers(asset_id: str):
        meta_of(asset_id)
        with lock:
            return markers_of(asset_id)

    @router.post("/api/merge/markers-preview")
    def preview(req: PreviewReq):
        if len(req.asset_ids) < 2:
            raise HTTPException(400, "Pick two or more scans: the first is the reference")
        metas = [meta_of(i) for i in req.asset_ids]
        with lock:
            ref = workspace.load_geometry(metas[0]["id"])
            out = []
            for k, meta in enumerate(metas[1:], start=1):
                try:
                    T, info = align_by_markers(workspace.load_geometry(meta["id"]), ref, log=lambda *_: None)
                    out.append({"index": k, "asset_id": meta["id"], "name": meta["name"], "ok": True,
                                "transform": np.asarray(T).round(10).tolist(),
                                **{key: info[key] for key in ("common", "marker_rms_mm", "markers_moving",
                                                              "markers_reference", "ambiguous", "fitness", "rmse",
                                                              "refine_shift_mm", "refine_turn_deg") if key in info},
                                **({"warning": info["warning"]} if info.get("warning") else {})})
                except ValueError as exc:
                    out.append({"index": k, "asset_id": meta["id"], "name": meta["name"], "ok": False,
                                "error": str(exc)})
        return {"reference": metas[0]["id"], "scans": out, "ok": all(s["ok"] for s in out)}

    return router
