"""Job `photos_to_3d`: a 3D model (coloured point cloud) from photos of the part, with MapAnything.

The model runs in the `cloudclean-recon` Docker image (docs/photos-to-3d.md), so CloudClean itself needs no GPU
libraries: the job copies the photos to a work folder, runs tools/recon/photos_to_3d.py in the container, reads its
PROGRESS / RESULT lines, and adds the points as a new model whose parents are the photos.

Payload {photo_ids?: [...], project_id?, name?, ruler_mm?}: no photo_ids = every photo of the project. The photos
are searched for the printed scale sheet (cloudclean/scale_sheet.py): found, the model comes out at true size in mm
with the sheet as the ground (Z = 0); ruler_mm = what its 100 mm bar measured on the print."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import open3d as o3d

from .workspace import Workspace

RECON_IMAGE = os.environ.get("CLOUDCLEAN_RECON_IMAGE", "cloudclean-recon:latest")
RECON_CACHE = Path(os.environ.get("CLOUDCLEAN_RECON_CACHE", str(Path.home() / "cloudclean-deploy" / "recon" / "hf")))
SCRIPT_DIR = Path(__file__).resolve().parents[2] / "tools" / "recon"
MIN_PHOTOS = 2
CONTAINER_PREFIX = "cloudclean-recon-"


def recon_available() -> tuple[bool, str]:
    """Whether photos can be turned into 3D here (Docker, the GPU image), with the reason when not."""
    if not shutil.which("docker"):
        return False, "Making 3D models from photos needs the reconstruction container, which runs on the DGX Spark"
    try:
        found = subprocess.run(["docker", "image", "inspect", RECON_IMAGE], capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Docker did not answer: {exc}"
    if found.returncode != 0:
        err = (found.stderr or b"").decode(errors="replace").strip()
        low = err.lower()
        if "no such image" in low or "no such object" in low:
            return False, f"The reconstruction image {RECON_IMAGE} is not built yet - see docs/photos-to-3d.md"
        if "permission denied" in low:
            return False, ("CloudClean is not allowed to use Docker here: add its user to the docker group "
                           f"(docs/photos-to-3d.md). Docker said: {err[:200]}")
        if "daemon" in low or "connect" in low or "pipe" in low:
            return False, ("Docker is installed but not running, so photos cannot be turned into 3D on this computer. "
                           "Use CloudClean on the DGX Spark, or start Docker.")
        return False, f"Docker could not be used: {err[:300] or 'no reason given'}"
    return True, "ready"


def check_ruler(value) -> float | None:
    """What the scale sheet's 100 mm bar measured on the print (mm), or None when not given. A printer that scales
    the page is off by a few tenths of a percent, 'fit to page' by a few percent: beyond 90-110 mm it is a typo."""
    if value in (None, "", 0):
        return None
    try:
        mm = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"The length of the 100 mm bar must be a number of mm, not {value!r}") from None
    if not 90.0 <= mm <= 110.0:
        raise ValueError(f"The 100 mm bar on the scale sheet cannot measure {mm:g} mm - measure it again, or print "
                         "the sheet again at 100 % (actual size)")
    return mm


def photos_of(ws: Workspace, photo_ids: list[str] | None, project_id: str | None) -> list[dict]:
    if photo_ids:
        metas = [ws.get(i) for i in photo_ids]
        bad = [m["name"] for m in metas if m.get("kind") != "image"]
        if bad:
            raise ValueError(f"Not photos: {', '.join(bad[:3])}")
        return metas
    photos = [m for m in ws.list() if m.get("kind") == "image" and (not project_id or m.get("project") == project_id)]
    if not project_id and len({m.get("project") for m in photos}) > 1:
        raise ValueError("These photos belong to several projects - open the project of the part first, or pick "
                         "the photos to use")
    return photos


def _remove_container(name: str) -> None:
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _remove_leftovers() -> None:
    """Containers of an earlier run that never finished (server restarted mid-job): one job runs at a time, so any
    cloudclean-recon-* container still there is stale and holds GPU memory."""
    try:
        found = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^{CONTAINER_PREFIX}"], capture_output=True,
                               text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return
    for cid in found.stdout.split():
        _remove_container(cid)


@contextmanager
def reconstruction(ws: Workspace, photos: list[dict], log, progress=None, lo: float = 0.02, hi: float = 0.92,
                   dense: bool = True, poses: str = "auto", scale_sheet: bool = False, ruler_mm: float | None = None):
    """Run tools/recon/photos_to_3d.py in the container on these photos and yield (out, result, cameras):

    out/points.ply     dense points in the reconstruction's frame (mm by the model's size estimate; with COLMAP
                       cameras the frame and scale are COLMAP's, scaled by that estimate), with colours. With
                       scale_sheet and the sheet found (result["scale_sheet"]["found"]): true mm in the sheet's
                       frame (Z up from the sheet), the sheet and the table cropped away
    out/sparse.ply     with COLMAP cameras: COLMAP's triangulated points, same frame and units - exact relative to
                       the cameras (the dense points can be ~2 % off in depth), so the one to trust for alignment
    out/cameras.json   {"poses": "colmap" | "mapanything", "registered", "photos", "cameras": [{"image", "file",
                       "K", "width", "height", "world_to_camera", "cam_to_world"}]}: per placed photo, a pinhole
                       camera (OpenCV convention, same frame and units as the points) for the undistorted image
                       out/<file>; photos that could not be placed are missing from the list
    result             the RESULT line (photos, registered, points, extent_mm, seconds, poses, model)

    poses: "auto" = COLMAP, or MapAnything's rough cameras when COLMAP places fewer than half of the photos;
    "colmap" = only the photos COLMAP placed exactly (at least 3, else the job fails with advice).

    Photos are copied as NNN.<ext> in the given order; cameras name them the same way. The work folder is removed
    afterwards, also when the job is cancelled (the container is stopped)."""
    progress = progress or (lambda *a: None)
    ok, why = recon_available()
    if not ok:
        raise ValueError(why)
    work = ws.root / "recon" / uuid.uuid4().hex[:12]
    images, out = work / "images", work / "out"
    images.mkdir(parents=True)
    out.mkdir()
    RECON_CACHE.mkdir(parents=True, exist_ok=True)
    container = CONTAINER_PREFIX + work.name
    proc = None
    # Cancel terminates this worker process: turn that into an exception so the finally below stops the container
    old_handler = None
    if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGTERM"):
        old_handler = signal.signal(signal.SIGTERM, _cancelled)
    _remove_leftovers()
    try:
        for k, m in enumerate(photos):
            src = ws.image_path(m["id"])
            shutil.copyfile(src, images / f"{k:03d}{src.suffix.lower()}")
        cmd = ["docker", "run", "--rm", "--name", container, "--gpus", "all", "--ipc", "host",
               "-e", "HF_TOKEN", "-e", "TORCH_HOME=/hf/torch",
               "-v", f"{images}:/in:ro", "-v", f"{out}:/out", "-v", f"{RECON_CACHE}:/hf", "-v", f"{SCRIPT_DIR}:/code:ro",
               RECON_IMAGE, "python3", "/code/photos_to_3d.py", "--images", "/in", "--out", "/out"]
        if not dense:
            cmd.append("--no-dense")
        if poses != "auto":
            cmd += ["--poses", poses]
        if scale_sheet:
            cmd.append("--scale-sheet")
            if ruler_mm:
                cmd += ["--ruler-mm", f"{ruler_mm:g}"]
        log(f"Finding where {len(photos)} photos were taken and reconstructing the part "
            "(the first run downloads the model, ~5 GB)")
        progress(lo, "Starting the reconstruction")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        result, errors, tail = None, [], deque(maxlen=12)
        for line in proc.stdout or []:
            line = line.rstrip()
            if line.startswith("PROGRESS "):
                _, fraction, *label = line.split(" ", 2)
                try:
                    progress(lo + (hi - lo) * float(fraction), label[0] if label else None)
                except ValueError:
                    tail.append(line)
            elif line.startswith("RESULT "):
                try:
                    result = json.loads(line[len("RESULT "):])
                except ValueError:
                    tail.append(line)
            elif line.startswith("ERROR "):
                errors.append(line[len("ERROR "):])
                log(line[len("ERROR "):])
            elif line.startswith("LOG "):
                log(line[len("LOG "):])
            elif line:
                tail.append(line)
        code = proc.wait()
        if code == 3 and errors:   # the photos themselves are the problem: the script says what to do
            raise ValueError(errors[-1])
        if code != 0 or result is None:
            detail = errors[-1] if errors else " / ".join(list(tail)[-3:])
            raise ValueError(f"The reconstruction did not finish (exit {code}): {detail}")
        cameras = json.loads((out / "cameras.json").read_text(encoding="utf-8"))
        yield out, result, cameras
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            _remove_container(container)
        if old_handler is not None:
            signal.signal(signal.SIGTERM, old_handler)
        shutil.rmtree(work, ignore_errors=True)


def size_sentence(sheet: dict | None) -> str:
    """What the job says about the model's size: from the scale sheet (with its uncertainty from the photos), or
    unknown. The photos cannot see a printer that scaled the whole page: only the measured 100 mm bar can."""
    if sheet and sheet.get("found"):
        return (f"True size from the scale sheet (±{sheet['uncertainty_pct']:.2g} %)"
                + ("" if sheet.get("ruler_mm") else "; the print was not checked with its 100 mm bar"))
    why = (sheet or {}).get("reason")
    return ("Size unknown: no scale sheet found" + (f" ({why})" if why else "") + ". The photos alone do not show "
            "the true size; set it from one known length, or photograph the part on the printed scale sheet")


def job_photos_to_3d(ws: Workspace, payload: dict, log) -> list[str]:
    progress = getattr(log, "progress", None) or (lambda *a: None)
    photos = photos_of(ws, [str(i) for i in payload.get("photo_ids") or []], payload.get("project_id"))
    if len(photos) < MIN_PHOTOS:
        raise ValueError("Add at least 2 photos of the part - 12 or more, about every 30 degrees all the way round, work "
                         "best")
    ruler_mm = check_ruler(payload.get("ruler_mm"))
    with reconstruction(ws, photos, log, progress, scale_sheet=True, ruler_mm=ruler_mm) as (out, result, cameras):
        progress(0.95, "Saving the model")
        cloud = o3d.io.read_point_cloud(str(out / "points.ply"))
        poses = result.get("poses", "mapanything")
        sheet = result.get("scale_sheet")
        on_sheet = bool(sheet and sheet.get("found"))
        report = {"photos": len(photos), "model": result["model"], "points": result["points"],
                  "seconds": result["seconds"], "extent_mm": result["extent_mm"], "poses": poses,
                  "registered": result.get("registered", len(photos)), "scale": size_sentence(sheet),
                  "cameras": cameras.get("cameras", [])}
        if sheet is not None:
            report["scale_sheet"] = sheet
        placed = result.get("registered", len(photos))
        log(f"{result['points']:,} points from {placed} of {len(photos)} photos in {result['seconds']:.0f} s "
            f"(cameras by {'COLMAP' if poses == 'colmap' else 'MapAnything alone - rough'})")
        log(report["scale"])
        if poses != "colmap":
            report["warning"] = ("The camera positions could only be estimated roughly, so the model may be warped "
                                 "or doubled. More photos with more overlap and a patterned sheet under the part help.")
            log("WARNING: " + report["warning"])
        if placed < len(photos):
            log(f"{len(photos) - placed} photo(s) could not be placed and were left out")
        name = payload.get("name") or f"{photos[0]['name']} · from {len(photos)} photos"
        params = {"photos": len(photos), "model": result["model"], "poses": poses}
        if on_sheet:   # the UI's "True size" block reads these
            params.update(size="scale sheet", size_uncertainty_pct=sheet["uncertainty_pct"])
            if ruler_mm:
                params["ruler_mm"] = ruler_mm
        created = ws.add_geometry(cloud, name, "photos", [m["id"] for m in photos], params, report)
        progress(1.0, "Done")
        return [created["id"]]


def _cancelled(*_):
    raise SystemExit("cancelled")


JOBS = {"photos_to_3d": job_photos_to_3d}
