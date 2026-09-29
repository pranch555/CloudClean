"""Job `colour_from_photos`: colour the owner's scan or mesh from photos of the part, with no lining up by hand
(docs/colour-from-photos.md).

1. The photos go through the reconstruction container (jobs_photos3d.reconstruction): where each photo was taken,
   a dense photo model (points.ply) and COLMAP's own points (sparse.ply), in the reconstruction's frame and scale.
2. The photo model is lined up with the scan or mesh (cloudclean.photo_align: scale, rotation, shift): the dense
   model finds the part and the pose, the final scale and pose come from sparse.ply, whose points agree exactly with
   the cameras (the dense model can be ~2 % off against them). No sparse.ply (rough MapAnything cameras): the dense
   fit is used and the result is not marked trusted.
3. The cameras move into the model's frame and the photos are projected onto it: vertex colours plus a UV texture
   for a mesh (texture.colorize_mesh), per-point colours for a point cloud (texture.colour_points).

Payload {asset_id, photo_ids?, project_id?, name?, texture_size?}: no photo_ids = every photo of the model's
project. The model is never moved or reshaped: the result is a new asset with the same geometry, in colour."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ..io import estimate_spacing, is_cloud
from ..photo_align import align_photo_model, camera_to_model
from ..photo_check import check_placement
from ..texture import CameraView, TextureParams, colorize_mesh, colour_points, image_size, save_textured
from .jobs_photos3d import photos_of, reconstruction
from .workspace import Workspace

MIN_COLOUR_PHOTOS = 3
# photo check (cloudclean.photo_check): confidence = how far the agreement is from that of wrong placements, 0-1
CONFIDENCE_FAIL = 0.3    # below: the photos do not agree placed on the model - refused
CONFIDENCE_TRUST = 0.55  # at or above (and the geometry fits): trusted. Measured: 0.72 right, 0.1-0.5 wrong
DEFAULT_TEXTURE_SIZE = 4096
TOO_FEW = "Add at least 3 photos of the part - 12 or more, taken all the way round with overlap, work best"
TOO_FEW_PLACED = ("Neighbouring photos must overlap: take them about 30 degrees apart all the way round the part, "
                  "keeping the height the same from one photo to the next, with the part filling most of the frame")


def check_texture_size(value) -> int:
    size = DEFAULT_TEXTURE_SIZE if value is None else int(value)
    if size != 0 and not 256 <= size <= 8192:
        raise ValueError("texture_size must be 0 (colours only) or 256 to 8192 pixels")
    return size


def read_cameras(cameras: dict, out: Path, photos: list[dict], ws: Workspace) -> list[dict]:
    """The placed photos' cameras: {photo, file, K, width, height, world_to_camera}. `file` is the undistorted photo
    the container wrote (K is for it), else the original photo."""
    placed, root = [], Path(out).resolve()
    for k, cam in enumerate(cameras.get("cameras") or []):
        digits = re.match(r"\d+", Path(str(cam.get("image") or "")).stem)
        index = int(digits.group(0)) if digits else k   # photos were copied in as NNN.<ext>, in order
        if not 0 <= index < len(photos):
            continue
        if cam.get("world_to_camera") is not None:
            W = np.asarray(cam["world_to_camera"], float)
        elif cam.get("cam_to_world") is not None or cam.get("cam_to_world_mm") is not None:
            W = np.linalg.inv(np.asarray(cam.get("cam_to_world", cam.get("cam_to_world_mm")), float))
        else:
            continue
        K = np.asarray(cam.get("K", cam.get("intrinsics")), float)
        size = cam.get("size") or [cam.get("width"), cam.get("height")]
        file = (root / str(cam["file"])).resolve() if cam.get("file") else None
        if file is None or root not in file.parents or not file.is_file():
            file = ws.image_path(photos[index]["id"])
        if not all(size):
            size = image_size(file)
        if W.shape != (4, 4) or K.shape != (3, 3) or not (np.all(np.isfinite(W)) and np.all(np.isfinite(K))):
            continue
        placed.append({"photo": photos[index], "file": file, "K": K, "width": int(size[0]),
                       "height": int(size[1]), "world_to_camera": W})
    return placed


def _check_points(geom, mesh: bool, count: int = 200_000):
    """Points with normals on the model for the photo check, and their spacing."""
    if mesh:
        m = o3d.geometry.TriangleMesh(geom)
        m.compute_triangle_normals()
        pc = m.sample_points_uniformly(count, use_triangle_normal=True)
    else:
        pc = geom if len(geom.points) <= count else geom.random_down_sample(count / len(geom.points))
        pc = o3d.geometry.PointCloud(pc)
        if not pc.has_normals():
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    pts = np.asarray(pc.points)
    return pts, np.asarray(pc.normals), estimate_spacing(pts)


class _PhotoSteps:
    """Log wrapper that moves the progress bar on every "  photo i ..." line the colouring code writes."""

    def __init__(self, log, progress, lo: float, hi: float, steps: int):
        self.log, self.progress, self.lo, self.hi, self.steps, self.done = log, progress, lo, hi, max(1, steps), 0

    def __call__(self, msg: str) -> None:
        self.log(msg)
        if msg.startswith("  photo ") and "coverage" not in msg:
            self.done += 1
            self.progress(self.lo + (self.hi - self.lo) * min(1.0, self.done / self.steps),
                          "Colouring from the photos")


def job_colour_from_photos(ws: Workspace, payload: dict, log) -> list[str]:
    progress = getattr(log, "progress", None) or (lambda *a: None)
    asset_id = str(payload["asset_id"])
    meta = ws.get(asset_id)
    if meta.get("kind") == "image":
        raise ValueError(f"'{meta['name']}' is a photo - pick the scan or mesh to colour")
    photos = photos_of(ws, [str(i) for i in payload.get("photo_ids") or []],
                       payload.get("project_id") or meta.get("project"))
    photos = list({m["id"]: m for m in photos}.values())
    if len(photos) < MIN_COLOUR_PHOTOS:
        raise ValueError(TOO_FEW)
    size = check_texture_size(payload.get("texture_size"))
    geom = ws.load_geometry(asset_id)
    mesh = not is_cloud(geom) and len(geom.triangles) > 0
    if not is_cloud(geom) and not mesh:
        geom = o3d.geometry.PointCloud(geom.vertices)

    with reconstruction(ws, photos, log, progress, 0.02, 0.6, poses="colmap") as (out, result, cameras):
        progress(0.6, f"Lining the photos up with {meta['name']}")
        placed = read_cameras(cameras, out, photos, ws)
        if len(placed) < 2:
            raise ValueError(f"Only {len(placed)} of {len(photos)} photos could be placed, so they cannot be lined up "
                             f"with {meta['name']}. Take more photos all the way round, with overlap between them")
        if len(placed) < len(photos):
            log(f"{len(photos) - len(placed)} photo(s) could not be placed and were left out")
        poses = cameras.get("poses") or result.get("poses") or "mapanything"
        if poses != "colmap":   # rough cameras painted every colour in the wrong place in tests: never used here
            raise ValueError(f"The photos could not be placed exactly, so their colours cannot be put on "
                             f"{meta['name']}. {TOO_FEW_PLACED}")
        photo_model = o3d.io.read_point_cloud(str(out / "points.ply"))
        # COLMAP's own points agree exactly with its cameras; the dense points can be ~2 % off (see the top)
        sparse = out / "sparse.ply"
        camera_points = np.asarray(o3d.io.read_point_cloud(str(sparse)).points) if sparse.is_file() else None
        align = align_photo_model(np.asarray(photo_model.points), [c["world_to_camera"] for c in placed], geom,
                                  meta["name"], poses, log,
                                  lambda f, *_: progress(0.6 + 0.1 * f, "Lining the photos up"), camera_points)
        S = align["transform"]
        views = [CameraView(str(c["file"]), c["K"], camera_to_model(c["world_to_camera"], S), c["width"], c["height"])
                 for c in placed]

        # the photos themselves check (and repair) where they were placed: with few photos the exact points lie
        # mostly on flat faces, which fit even when turned; near holes and edges the photos disagree unless placed
        # right (6 photos: 16 mm / 6 deg off -> 0.7 mm / 0.14 deg)
        progress(0.7, "Checking the placement against the photos")
        pts, nrm, spacing = _check_points(geom, mesh)
        near = None
        if camera_points is not None and len(camera_points):
            cp = np.asarray(camera_points, float) @ S[:3, :3].T + S[:3, 3]
            reach = max(3 * align["within_mm"], 0.01 * float(np.linalg.norm(pts.max(0) - pts.min(0))))
            d, _ = cKDTree(pts).query(cp, k=1, distance_upper_bound=reach)
            near = cp[np.isfinite(d)]
        check = check_placement(pts, nrm, views, log, lambda f: progress(0.7 + 0.05 * f,
                                                                       "Checking the placement against the photos"),
                                spacing, camera_points=near, camera_tol=align["within_mm"])
        confidence = (check["agreement"] - check["typical_wrong"]) / max(1e-6, 1 - check["typical_wrong"])
        geometry_kept = check["geometry_fit"] is None or \
            check["geometry_fit"] >= 0.8 * (check["geometry_fit_before"] or 0.0)
        if confidence < CONFIDENCE_FAIL or not geometry_kept:
            raise ValueError(
                f"The photos could not be lined up with {meta['name']} reliably: placed on it, they do not agree on "
                f"its colours (agreement {check['agreement']:.2f}; a wrong placement scores "
                f"{check['typical_wrong']:.2f}). Take photos all the way round the part, about 30 degrees apart at "
                "the same height, with the whole part in view")
        views = check["views"]
        warnings = list(align["warnings"])
        photo_ok = confidence >= CONFIDENCE_TRUST and check["rival_deg"] is None
        if photo_ok:   # the photos settle the "a pose turned ... fits almost as well" doubt of the geometry
            warnings = [w for w in warnings if not w.startswith("A pose turned")]
        else:
            if check["rival_deg"] is not None:
                warnings.append(f"Turned {check['rival_deg']:.0f} degrees, the photos agree almost as well - the "
                                "part looks alike from several sides, so the colours could land turned. Check them.")
            if confidence < CONFIDENCE_TRUST:
                warnings.append("The photos agree only partly on the colours once placed on the model, so some "
                                "colours may be misplaced. Check them on the model; more photos all the way round "
                                "help.")
            for w in warnings[len(align["warnings"]):]:
                log("WARNING: " + w)
        trusted = bool(align["geometry_ok"] and photo_ok)

        progress(0.75, "Colouring from the photos")
        textured = None
        if mesh:
            params = TextureParams(texture_size=size)
            steps = _PhotoSteps(log, progress, 0.75, 0.98, len(views) * (2 if size else 1))
            coloured, report, textured = colorize_mesh(geom, views, params, steps)
        else:
            params = TextureParams()
            coloured, report = colour_points(geom, views, params, log,
                                             lambda f: progress(0.75 + 0.23 * f, "Colouring from the photos"))
        for entry, c in zip(report.get("views", []), placed):   # names in the work folder mean nothing later
            entry.update(image=c["photo"]["id"], name=c["photo"]["name"])
        report.update(
            photos=len(photos), registered=int(result.get("registered", len(placed))), used=len(views), poses=poses,
            model=result.get("model"), reconstruction_seconds=result.get("seconds"),
            alignment={"scale": round(align["scale"], 8), "fitted_on": align["fitted_on"],
                       "fitness": round(align["fitness"], 4), "rmse_mm": round(align["rmse_mm"], 5),
                       "within_mm": round(align["within_mm"], 5), "photo_points": align["photo_points"],
                       "coverage": round(align["coverage"], 4), "dense_fitness": round(align["dense_fitness"], 4),
                       "dense_scale_offset_pct": round(align["dense_scale_offset_pct"], 3),
                       "trusted": trusted, "ambiguous": check["rival_deg"] is not None,
                       "photo_agreement": check["agreement"], "photo_agreement_before": check["agreement_before"],
                       "photo_agreement_wrong": check["typical_wrong"], "photo_confidence": round(confidence, 4),
                       "repair_mm": check["moved_mm"], "repair_deg": check["moved_deg"],
                       "repair_scale_pct": check["scale_change_pct"],
                       "alternative_angle": align["alternative_angle"], "part_points": align["part_points"],
                       "transform": np.asarray(S).round(10).tolist()},
            warnings=warnings,
            cameras=[{**v.to_dict(), "image": c["photo"]["id"], "name": c["photo"]["name"]}
                     for v, c in zip(views, placed)])
        report["coverage"] = round(float(report["coverage"]), 4)
        used_ids = [c["photo"]["id"] for c in placed]
        name = payload.get("name") or f"{meta['name']} · colour from {len(views)} photos"
        res = ws.add_geometry(coloured, name, "texture", [asset_id, *dict.fromkeys(used_ids)],
                              {"photos": len(photos), "texture_size": size if mesh else 0, "poses": poses}, report)
        log(f"Coloured {meta['name']} from {len(views)} photos: {report['coverage']:.0%} of the surface seen directly, "
            "the rest filled in from around it")
        if textured is not None:
            d = ws.asset_dir(res["id"])
            save_textured(textured, d / "textured.glb")
            save_textured(textured, d / "textured_obj" / "model.obj")
            ws.update(res["id"], textured=True)
            log("Saved UV-textured GLB and OBJ")
    progress(1.0, "Done")
    return [res["id"]]


JOBS = {"colour_from_photos": job_colour_from_photos}
