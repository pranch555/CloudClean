"""Job body: fill the parts a scan missed from photos of the part (cloudclean/photo_fill.py)."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ..io import estimate_spacing, to_cloud
from ..photo_align import align_photo_model
from ..photo_fill import FillParams, fill_gaps, refine_onto_scan
from .jobs_colour_photos import MIN_COLOUR_PHOTOS, TOO_FEW, TOO_FEW_PLACED, read_cameras
from .jobs_photos3d import photos_of, reconstruction
from .workspace import Workspace


def job_fill_from_photos(ws: Workspace, payload: dict, log) -> list[str]:
    progress = getattr(log, "progress", None) or (lambda *a: None)
    asset_id = str(payload["asset_id"])
    meta = ws.get(asset_id)
    if meta.get("kind") == "image":
        raise ValueError(f"'{meta['name']}' is a photo - pick the scan to fill")
    photos = photos_of(ws, [str(i) for i in payload.get("photo_ids") or []],
                       payload.get("project_id") or meta.get("project"))
    photos = list({m["id"]: m for m in photos}.values())
    if len(photos) < MIN_COLOUR_PHOTOS:
        raise ValueError(TOO_FEW)
    params = FillParams.from_dict(payload.get("params", {}))
    geom = ws.load_geometry(asset_id)
    scan = o3d.geometry.PointCloud(to_cloud(geom))           # a mesh: points on its surface
    scan_pts = np.asarray(scan.points)
    spacing = estimate_spacing(scan_pts)

    with reconstruction(ws, photos, log, progress, 0.02, 0.6, poses="colmap") as (out, result, cameras):
        progress(0.6, f"Lining the photos up with {meta['name']}")
        placed = read_cameras(cameras, out, photos, ws)
        if len(placed) < 2:
            raise ValueError(f"Only {len(placed)} of {len(photos)} photos could be placed, so they cannot be lined up "
                             f"with {meta['name']}. Take more photos all the way round, with overlap between them")
        poses = cameras.get("poses") or result.get("poses") or "mapanything"
        if poses != "colmap":
            raise ValueError(f"The photos could not be placed exactly, so they cannot fill {meta['name']}. "
                             f"{TOO_FEW_PLACED}")
        dense = o3d.io.read_point_cloud(str(out / "points.ply"))
        sparse = out / "sparse.ply"
        camera_points = np.asarray(o3d.io.read_point_cloud(str(sparse)).points) if sparse.is_file() else None
        align = align_photo_model(np.asarray(dense.points), [c["world_to_camera"] for c in placed], geom,
                                  meta["name"], poses, log,
                                  lambda f, *_: progress(0.6 + 0.1 * f, "Lining the photos up"), camera_points,
                                  partial=True)   # the photos show more of the part than the scan: that is the point
    S = align["transform"]
    photo_pts = np.asarray(dense.points) @ S[:3, :3].T + S[:3, 3]
    progress(0.75, "Fitting the photo points onto the scan")
    T, fit = refine_onto_scan(photo_pts, scan_pts, spacing, log)
    photo_pts = photo_pts @ T[:3, :3].T + T[:3, 3]

    progress(0.85, "Filling the gaps")
    res = fill_gaps(scan_pts, photo_pts, spacing, params, log)
    idx = res["added_index"]
    merged = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(res["points"]))
    if scan.has_colors() and dense.has_colors():
        merged.colors = o3d.utility.Vector3dVector(np.vstack([np.asarray(scan.colors), np.asarray(dense.colors)[idx]]))
    if scan.has_normals() and len(idx):
        added = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(res["points"][len(scan_pts):]))
        added.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
        n_add = np.asarray(added.normals)
        n_scan = np.asarray(scan.normals)
        _, near = cKDTree(scan_pts).query(res["points"][len(scan_pts):], k=1, workers=-1)
        n_add *= np.where(np.einsum("ij,ij->i", n_add, n_scan[near]) < 0, -1.0, 1.0)[:, None]  # the scan's side out
        merged.normals = o3d.utility.Vector3dVector(np.vstack([n_scan, n_add]))

    report = {**res["report"], "photos": len(photos), "photos_placed": len(placed),
              "line_up": {"scale": round(float(align["scale"]) * fit["scale"], 6), "trusted": bool(align["trusted"]),
                          "fit_rmse_mm": round(fit["rmse_mm"], 4), "fit_share": round(fit["fitness"], 4)},
              "warnings": list(align["warnings"]) + res["report"]["warnings"]}
    if not align["trusted"]:
        report["warnings"].append("The photos did not line up with the scan with full confidence: check that the "
                                  "filled areas sit where they belong.")
    progress(0.95, "Saving")
    name = payload.get("name") or f"{meta['name']} + photo fill"
    filled = ws.add_geometry(merged, name, "photo_fill", [asset_id], params.to_dict(), report)
    ws.add_scalars(filled["id"], "source", res["source"], unit="",
                   description="0 = scanned, 1 = filled from photos (approximate: left out of the golden model check)")
    if hasattr(log, "output"):
        log.output(asset_id=filled["id"], added=report["added"], filled_area_mm2=report["filled_area_mm2"],
                   photo_vs_scan_mm=report["photo_vs_scan_mm"])
    progress(1.0, "done")
    return [filled["id"]]


JOBS = {"fill_from_photos": job_fill_from_photos}
