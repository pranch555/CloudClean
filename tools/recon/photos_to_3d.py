"""Photos -> cameras + dense 3D points (runs inside the cloudclean-recon container, see docs/photos-to-3d.md).

    python3 photos_to_3d.py --images DIR --out DIR [--poses auto|colmap|mapanything] [--no-dense] [--keep 0.8]
                            [--scale-sheet [--ruler-mm 100.0] [--sheet-crop 0.5]]

1. Where was each photo taken? COLMAP (classical structure from motion, CPU): SIFT features, matching, mapping.
   Accurate (on real DTU photos: camera positions to ~0.3 % of the camera spread, rotations < 1 deg) but it needs
   texture and overlap. When it places fewer than half of the photos, MapAnything's own cameras are used instead
   ("poses": "mapanything"): rough (several degrees), so the model may come out warped or doubled.
2. True size (--scale-sheet): the printed scale sheet (cloudclean/scale_sheet.py) is looked for in the undistorted
   photos. Its ArUco markers' corners are triangulated with COLMAP's cameras and the sheet's known layout is fitted
   to them (similarity, outliers rejected): that gives the scale, the sheet's plane and a frame in mm with Z up from
   the sheet. Every output is moved into that frame and the sheet and table (points less than --sheet-crop mm above
   the sheet, or below it) are cropped away. Not found well enough (fewer than 4 markers or 3 photos): as without.
3. Dense points: MapAnything (facebook/map-anything-apache) given the undistorted photos, their intrinsics and the
   COLMAP cameras, so its depth maps agree with each other.

Writes
  OUT/points.ply    x y z with colours: mm in the scale sheet's frame when it was found, else MapAnything's size
                    estimate (metres x 1000 - the true size is unknown)
  OUT/sparse.ply    COLMAP's triangulated points, same frame and units
  OUT/cameras.json  {"model", "poses", "photos", "registered", "scale_sheet", "cameras": [{"image", "file", "K",
                    "width", "height", "world_to_camera", "cam_to_world"}]} - pinhole cameras (OpenCV: x right,
                    y down, z forward) in the frame and units of points.ply, for the undistorted photo OUT/<file>
  OUT/images/       the undistorted photos the cameras describe
stdout: "PROGRESS <fraction> <label>", "LOG <text>", one "RESULT <json>" line, or "ERROR <message>".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}
COLMAP_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MODEL = "facebook/map-anything-apache"

# The printed scale sheet (cloudclean/scale_sheet.py - the container cannot import cloudclean, so this is a copy;
# tests/test_scale_sheet.py keeps the two the same): ArUco DICT_4X4_50 markers of SHEET_MARKER_MM, id -> centre in
# mm in the sheet's frame (origin in the middle of the page, x right, y to the top edge, z up out of the paper).
SHEET_MARKER_MM = 20.0
SHEET_BAR_MM = 100.0
SHEET_MARKERS = {0: (-80.0, 100.0), 1: (-40.0, 100.0), 2: (0.0, 100.0), 3: (40.0, 100.0), 4: (80.0, 100.0),
                 5: (80.0, 50.0), 6: (80.0, 0.0), 7: (80.0, -50.0),
                 8: (80.0, -100.0), 9: (40.0, -100.0), 10: (0.0, -100.0), 11: (-40.0, -100.0), 12: (-80.0, -100.0),
                 13: (-80.0, -50.0), 14: (-80.0, 0.0), 15: (-80.0, 50.0)}
SHEET_PAGE_MM = (215.9, 297.0)     # A4 and US Letter together: the crop keeps SHEET_KEEP_MM round this
SHEET_KEEP_MM = 150.0
SHEET_DOTS_MM = (132.0, 172.0)     # the printed dot pattern in the middle, where the part stands
MIN_SHEET_MARKERS, MIN_SHEET_VIEWS = 4, 3


def progress(fraction: float, label: str) -> None:
    print(f"PROGRESS {fraction:.3f} {label}", flush=True)


def say(text: str) -> None:
    print(f"LOG {text}", flush=True)


def write_ply(path: Path, points: np.ndarray, colours: np.ndarray) -> None:
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


# --------------------------------------------------------------------------- COLMAP
def colmap(*args: str, cwd: Path) -> str:
    done = subprocess.run(["colmap", *args], cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"colmap {args[0]} failed: {(done.stderr or done.stdout).strip()[-400:]}")
    return done.stdout + done.stderr


def quat_to_R(qw, qx, qy, qz) -> np.ndarray:
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def read_text_model(folder: Path) -> tuple[dict, dict]:
    """cameras.txt + images.txt -> ({camera_id: K 3x3, width, height}, {image name: (world_to_camera 4x4, cam id)})."""
    cams = {}
    for line in (folder / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        v = line.split()
        cid, model, w, h, p = int(v[0]), v[1], int(v[2]), int(v[3]), [float(x) for x in v[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = p[:4]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        else:
            fx, fy, cx, cy = p[:4]
        # COLMAP puts pixel centres at +0.5, OpenCV (texture.py, MapAnything) at whole numbers
        cams[cid] = (np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1.0]]), w, h)
    images = {}
    lines = [l for l in (folder / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    for line in lines[::2]:
        v = line.split()
        W = np.eye(4)
        W[:3, :3] = quat_to_R(*map(float, v[1:5]))
        W[:3, 3] = [float(x) for x in v[5:8]]
        images[v[9]] = (W, int(v[8]))
    return cams, images


def read_sparse(folder: Path) -> dict:
    """COLMAP's triangulated points: xyz, rgb, reprojection error and, per image name, the points it sees."""
    ids = {}
    lines = [l for l in (folder / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    for line in lines[::2]:
        v = line.split()
        ids[int(v[0])] = v[9]
    xyz, rgb, err, seen = [], [], [], {}
    for line in (folder / "points3D.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        v = line.split()
        k = len(xyz)
        xyz.append([float(x) for x in v[1:4]])
        rgb.append([int(x) for x in v[4:7]])
        err.append(float(v[7]))
        track = v[8:]
        for image_id in {int(i) for i in track[0::2]}:
            seen.setdefault(ids.get(image_id), []).append(k)
    return {"xyz": np.array(xyz, float).reshape(-1, 3), "rgb": np.array(rgb, np.uint8).reshape(-1, 3),
            "error": np.array(err), "seen": {n: np.array(v) for n, v in seen.items()}}


def colmap_has_cuda() -> bool:
    try:
        return "with CUDA" in subprocess.run(["colmap", "-h"], capture_output=True, text=True).stdout
    except OSError:
        return False


def colmap_cameras(images_dir: Path, names: list[str], work: Path, min_share: float = 0.5):
    """Undistorted photos + pinhole cameras from COLMAP (those it could place), or None when it places fewer than
    3, or fewer than min_share of the photos."""
    from PIL import Image

    sizes = {Image.open(images_dir / n).size for n in names}
    db = work / "db.db"

    def features(gpu: str) -> None:
        progress(0.04, f"Finding features in {len(names)} photos")
        colmap("feature_extractor", "--database_path", str(db), "--image_path", str(images_dir),
               "--ImageReader.camera_model", "SIMPLE_RADIAL",
               "--ImageReader.single_camera", "1" if len(sizes) == 1 else "0",
               "--SiftExtraction.use_gpu", gpu, "--SiftExtraction.max_image_size", "1600", cwd=work)
        progress(0.2, "Matching the photos with each other")
        if len(names) <= 120:
            colmap("exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", gpu, cwd=work)
        else:
            colmap("sequential_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", gpu,
                   "--SequentialMatching.overlap", "20", cwd=work)

    if colmap_has_cuda():   # the CUDA build finds and matches features ~20x faster
        try:
            features("1")
        except RuntimeError as exc:   # the GPU is shared: when it is full, the CPU does it (slower, same result)
            say(f"COLMAP could not use the GPU ({(str(exc).splitlines() or [''])[0][:100]}): features on the CPU "
                "instead")
            for leftover in work.glob("db.db*"):
                leftover.unlink(missing_ok=True)
            features("0")
    else:
        features("0")
    progress(0.4, "Working out where each photo was taken")
    sparse = work / "sparse"
    sparse.mkdir()
    colmap("mapper", "--database_path", str(db), "--image_path", str(images_dir), "--output_path", str(sparse), cwd=work)
    best, best_n = None, 0
    for model in sorted(p for p in sparse.iterdir() if p.is_dir()):
        txt = work / f"txt_{model.name}"
        txt.mkdir()
        colmap("model_converter", "--input_path", str(model), "--output_path", str(txt), "--output_type", "TXT",
               cwd=work)
        n = len(read_text_model(txt)[1])
        if n > best_n:
            best, best_n = model, n
    if best is None or best_n < max(3, math.ceil(min_share * len(names))):
        say(f"COLMAP placed {best_n} of {len(names)} photos - not enough")
        return None
    stats = colmap("model_analyzer", "--path", str(best), cwd=work)
    reproj = next((l.split(":")[-1].strip() for l in stats.splitlines() if "reprojection error" in l.lower()), "?")
    say(f"COLMAP placed {best_n} of {len(names)} photos (mean reprojection error {reproj})")
    progress(0.5, "Removing lens distortion")
    undist = work / "undistorted"
    colmap("image_undistorter", "--image_path", str(images_dir), "--input_path", str(best), "--output_path",
           str(undist), "--output_type", "COLMAP", "--max_image_size", "2400", cwd=work)
    txt = work / "txt_undistorted"
    txt.mkdir()
    colmap("model_converter", "--input_path", str(undist / "sparse"), "--output_path", str(txt), "--output_type",
           "TXT", cwd=work)
    cams, placed = read_text_model(txt)
    out = []
    for name in names:
        if name not in placed:
            continue
        W, cid = placed[name]
        K, w, h = cams[cid]
        out.append({"image": name, "path": undist / "images" / name, "K": K, "width": w, "height": h,
                    "world_to_camera": W})
    return out


# --------------------------------------------------------------------------- MapAnything
def load_model():
    """MapAnything: the same model as MapAnything.from_pretrained(MODEL).to(device), loaded with one copy of its
    4.9 GB of weights at a time instead of two or three. On the Spark the GPU shares its memory with the other
    projects' model servers, and the usual way ran out of memory (2026-09-25, 11-16 GB free):

    - the GPU context is made first, while memory is free (made after the model was built on the CPU, moving the
      model to the GPU failed with 'CUDA error: out of memory' even with 11 GB free; made first, it works);
    - its DINOv2 encoder is built without first fetching DINOv2's own pretrained weights (torch_hub_pretrained):
      the checkpoint holds every encoder weight and overwrites them anyway (checked: the 76 tensors it does not
      hold are all in the DPT heads, left at their initial values either way);
    - the checkpoint is copied in one tensor at a time instead of as a whole second copy, then dropped from the
      page cache; the model is built on the CPU as usual (the same initial values) and moved to the GPU.
    Falls back to the usual loading if the lean one fails."""
    import torch
    from mapanything.models import MapAnything

    progress(0.6, "Loading the reconstruction model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":   # the GPU context first (see above)
        torch.zeros(1, device=device)

    def lean(cls, model, model_file, map_location, strict):
        from safetensors import safe_open

        state = model.state_dict()
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():   # strict=False, as from_pretrained: keys the model lacks are skipped
                if key in state:
                    state[key].copy_(f.get_tensor(key))
        uncache(model_file)
        return model.eval()

    had = "_load_as_safetensor" in MapAnything.__dict__
    usual = MapAnything.__dict__.get("_load_as_safetensor")

    def restore():
        if had:
            MapAnything._load_as_safetensor = usual
        elif "_load_as_safetensor" in MapAnything.__dict__:
            delattr(MapAnything, "_load_as_safetensor")

    try:
        from huggingface_hub import hf_hub_download

        config = json.loads(Path(hf_hub_download(MODEL, "config.json")).read_text(encoding="utf-8"))
        extra = {}
        encoder = config.get("encoder_config")
        if isinstance(encoder, dict) and encoder.get("uses_torch_hub"):
            extra["encoder_config"] = {**encoder, "torch_hub_pretrained": False}
        MapAnything._load_as_safetensor = classmethod(lean)
        model = MapAnything.from_pretrained(MODEL, **extra).to(device).eval()
        release_freed_memory()   # the CPU copy of the weights (~2 GB stayed in the heap)
        return model
    except Exception as exc:   # e.g. another huggingface_hub or MapAnything version: the usual way
        restore()
        say(f"Loading the model the usual way ({type(exc).__name__}: {str(exc)[:120]})")
        model = MapAnything.from_pretrained(MODEL)
        try:
            from huggingface_hub import hf_hub_download

            uncache(hf_hub_download(MODEL, "model.safetensors"))
        except Exception:
            pass
        return model.to(device).eval()
    finally:
        restore()


def release_freed_memory() -> None:
    """Give memory Python and torch have freed back to the system (glibc keeps freed blocks of up to 32 MB in its
    heap: the model's CPU copy, once on the GPU, stayed there). The GPU on the Spark allocates from the same memory."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):   # not glibc
        pass


def uncache(path) -> None:
    """Drop a file's pages from the page cache (the 4.9 GB of weights, read once): on the Spark the GPU allocates
    from the same memory, and cached pages count as 'available', not 'free'."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):   # not on Linux: nothing to do
        pass


def look_at_point(cams: list[dict]) -> np.ndarray:
    """The point all the cameras look at: closest (least squares) to every optical axis."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c in cams:
        T = np.linalg.inv(c["world_to_camera"])
        C, d = T[:3, 3], T[:3, 2] / np.linalg.norm(T[:3, 2])
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ C
    return np.linalg.solve(A, b)


def above_the_table(points: np.ndarray, centres: np.ndarray, distance: float, seed: int = 0) -> np.ndarray:
    """Without the surface the part stands on: the plane holding most points (RANSAC) is removed, with everything
    on its far side from the cameras. Returns the points unchanged when no such plane dominates."""
    if len(points) < 100:
        return points
    rng = np.random.default_rng(seed)
    tol = 0.01 * distance
    best, best_n = None, 0
    for _ in range(300):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        count = int((np.abs((points - a) @ n) < tol).sum())
        if count > best_n:
            best, best_n = (a, n), count
    if best is None or best_n < 0.3 * len(points):
        return points
    a, n = best
    if ((centres - a) @ n).mean() < 0:   # the cameras are on the positive side
        n = -n
    return points[(points - a) @ n > 2 * tol]


def part_boxes(cams: list[dict], sparse: dict, margin: float = 0.12):
    """Per photo, the pixel box around the part: COLMAP's points near the point the cameras look at, projected.
    Meant to let the dense model (~518 px) see the part larger; on the flange test it made the surface worse
    (median 8 mm vs 1.8 mm), so it is off unless --crop. None = no crop."""
    L = look_at_point(cams)
    centres = np.array([np.linalg.inv(c["world_to_camera"])[:3, 3] for c in cams])
    reach = 0.45 * float(np.median(np.linalg.norm(centres - L, axis=1)))
    near = sparse["xyz"][(np.linalg.norm(sparse["xyz"] - L, axis=1) < reach) & (sparse["error"] < 2.0)]
    near = above_the_table(near, centres, reach / 0.45)
    boxes = []
    for c in cams:
        box = None
        if len(near) >= 30:
            W, K = c["world_to_camera"], c["K"]
            Xc = near @ W[:3, :3].T + W[:3, 3]
            front = Xc[:, 2] > 0
            u = K[0, 0] * Xc[front, 0] / Xc[front, 2] + K[0, 2]
            v = K[1, 1] * Xc[front, 1] / Xc[front, 2] + K[1, 2]
            if len(u) >= 30:
                x0, x1 = np.percentile(u, [1, 99])
                y0, y1 = np.percentile(v, [1, 99])
                mx, my = margin * (x1 - x0), margin * (y1 - y0)
                x0, x1 = int(max(0, x0 - mx)), int(min(c["width"], x1 + mx))
                y0, y1 = int(max(0, y0 - my)), int(min(c["height"], y1 + my))
                if (x1 - x0) * (y1 - y0) < 0.7 * c["width"] * c["height"] and min(x1 - x0, y1 - y0) >= 128:
                    box = (x0, y0, x1, y1)
        boxes.append(box)
    return boxes


def dense_from_cameras(model, cams: list[dict], boxes=None):
    """MapAnything depth for each photo, given its intrinsics and COLMAP pose (arbitrary scale), cropped to the part
    where a box is given (the intrinsics shift with the crop; the pose stays)."""
    import torch
    from PIL import Image
    from mapanything.utils.image import preprocess_inputs

    views = []
    for c, box in zip(cams, boxes or [None] * len(cams)):
        img = np.asarray(Image.open(c["path"]).convert("RGB"))
        K = c["K"].astype(np.float32).copy()
        if box is not None:
            x0, y0, x1, y1 = box
            img = np.ascontiguousarray(img[y0:y1, x0:x1])
            K[0, 2] -= x0
            K[1, 2] -= y0
        cam_to_world = np.linalg.inv(c["world_to_camera"]).astype(np.float32)
        views.append({"img": torch.from_numpy(img), "intrinsics": torch.from_numpy(K),
                      "camera_poses": torch.from_numpy(cam_to_world), "is_metric_scale": torch.tensor([False])})
    progress(0.7, f"Building the surface from {len(views)} photos")
    with torch.no_grad():
        return model.infer(preprocess_inputs(views), memory_efficient_inference=True, use_amp=True,
                           amp_dtype="bf16", apply_mask=True, mask_edges=True, apply_confidence_mask=False)


def dense_images_only(model, paths: list[Path]):
    import torch
    from mapanything.utils.image import load_images

    progress(0.7, f"Reconstructing from {len(paths)} photos")
    with torch.no_grad():
        return model.infer(load_images([str(p) for p in paths]), memory_efficient_inference=True, use_amp=True,
                           amp_dtype="bf16", apply_mask=True, mask_edges=True, apply_confidence_mask=False)


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity dst = s R src + t (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / max((xs ** 2).sum() / len(src), 1e-12))
    return s, R, mu_d - s * R @ mu_s


def refine_dense(preds, cams: list[dict], sim, sparse: dict, neighbours: int = 4, tol: float = 0.01,
                 rescale: bool = False):
    """MapAnything's depth maps made to agree with COLMAP: each photo's depth is rescaled so COLMAP's exact points
    in that photo land at their depth (removes the few-% size bias of learned depth), then a point is kept only
    where at least two neighbouring photos see the surface at the same depth (removes floaters and noise).
    Returns per-photo lists of points (COLMAP frame), colours and confidences."""
    s, R, t = sim
    grids, depth, poses, Ks, colours, confs, masks = [], [], [], [], [], [], []
    for p, c in zip(preds, cams):
        X = p["pts3d"][0].float().cpu().numpy()
        X = (s * (X.reshape(-1, 3) @ R.T) + t).reshape(X.shape)          # COLMAP frame
        W = c["world_to_camera"]
        Xc = X @ W[:3, :3].T + W[:3, 3]
        mask = p["mask"][0].squeeze(-1).bool().cpu().numpy() & np.isfinite(X).all(-1) & (Xc[..., 2] > 0)
        # the model's grid intrinsics, for projecting into this photo's depth map
        Kg = p["intrinsics"][0].float().cpu().numpy().astype(float)
        grids.append(X)
        depth.append(np.where(mask, Xc[..., 2], np.nan))
        poses.append(W)
        Ks.append(Kg)
        colours.append(p["img_no_norm"][0].float().cpu().numpy())
        confs.append(p["conf"][0].float().cpu().numpy())
        masks.append(mask)

    def project(Xw, i):
        W = poses[i]
        Xc = Xw @ W[:3, :3].T + W[:3, 3]
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = Ks[i][0, 0] * Xc[:, 0] / z + Ks[i][0, 2]
            v = Ks[i][1, 1] * Xc[:, 1] / z + Ks[i][1, 2]
        return u, v, z

    def sample(i, u, v):
        H, Wd = depth[i].shape
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = (ui >= 0) & (ui < Wd) & (vi >= 0) & (vi < H)
        out = np.full(len(u), np.nan)
        out[ok] = depth[i][vi[ok], ui[ok]]
        return out

    # 1. per-photo depth scale from COLMAP's points (reprojection error < 2 px). Off by default: on test photos
    #    the depths already agreed with the cameras and this moved them ~2 % (sampling bias on tilted surfaces)
    ratios = []
    good = sparse["error"] < 2.0
    for i, c in enumerate(cams):
        idx = sparse["seen"].get(c["image"], np.array([], int))
        idx = idx[good[idx]] if len(idx) else idx
        r = 1.0
        if len(idx) >= 20:
            u, v, z = project(sparse["xyz"][idx], i)
            zp = sample(i, u, v)
            ok = np.isfinite(zp) & (zp > 0) & (z > 0)
            if ok.sum() >= 20:
                r = float(np.median(z[ok] / zp[ok]))
        if not rescale or not 0.7 < r < 1.4:   # implausible: keep the depth as it is
            r = 1.0
        ratios.append(r)
        C = np.linalg.inv(poses[i])[:3, 3]
        grids[i] = C + r * (grids[i] - C)
        depth[i] = depth[i] * r
    ratios = np.array(ratios)
    if rescale:
        say(f"Depth rescaled to COLMAP's points: median x{np.median(ratios):.4f} (photos x{ratios.min():.3f} to "
            f"x{ratios.max():.3f})")

    # 2. multi-view consistency against the nearest photos (by viewing direction)
    axes = np.array([np.linalg.inv(W)[:3, 2] for W in poses])
    out_p, out_c, out_w, kept, total = [], [], [], 0, 0
    for i in range(len(cams)):
        order = np.argsort(-(axes @ axes[i]))
        near = [j for j in order if j != i][:neighbours]
        m = masks[i]
        X = grids[i][m]
        agree = np.zeros(len(X), int)
        for j in near:
            u, v, z = project(X, j)
            zj = sample(j, u, v)
            agree += (np.abs(zj - z) <= tol * z)
        keep = agree >= min(2, len(near))
        total += len(X)
        kept += int(keep.sum())
        out_p.append(X[keep])
        out_c.append(colours[i][m][keep])
        out_w.append(confs[i][m][keep])
    say(f"Kept {kept:,} of {total:,} points that at least two other photos agree with (within {tol * 100:.0f} %)")
    return out_p, out_c, out_w


# --------------------------------------------------------------------------- scale sheet
def sheet_corners(ruler_mm: float | None = None) -> dict[int, np.ndarray]:
    """{marker id: 4x3 corners (mm, z = 0)} in ArUco's order (top left, top right, bottom right, bottom left as
    printed), stretched by what the 100 mm bar measured on the print."""
    k = (ruler_mm / SHEET_BAR_MM) if ruler_mm else 1.0
    h = SHEET_MARKER_MM / 2
    return {i: k * np.array([[x - h, y + h, 0.0], [x + h, y + h, 0.0], [x + h, y - h, 0.0], [x - h, y - h, 0.0]])
            for i, (x, y) in SHEET_MARKERS.items()}


def detect_markers(cams: list[dict]) -> list[dict]:
    """Per photo, {marker id: 4x2 corners (px, OpenCV convention: pixel centres at whole numbers, like K)} of the
    scale sheet's markers found in its undistorted image, corners refined to sub-pixel."""
    import cv2

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 60
    params.cornerRefinementMinAccuracy = 0.005
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
    found = []
    for c in cams:
        grey = cv2.imread(str(c["path"]), cv2.IMREAD_GRAYSCALE)
        seen = {}
        if grey is not None:
            corners, ids, _ = detector.detectMarkers(grey)
            ids = [] if ids is None else [int(i) for i in np.ravel(ids)]
            for q, i in zip(corners, ids):
                if i in SHEET_MARKERS and ids.count(i) == 1:   # found twice in one photo: trust neither
                    seen[i] = np.asarray(q, float).reshape(4, 2)
        found.append(seen)
    return found


def triangulate(uv: np.ndarray, Ws: list[np.ndarray], Ks: list[np.ndarray], huber: float = 1.0):
    """One point seen at pixels uv (n x 2) by cameras (world_to_camera 4x4, K 3x3): linear triangulation (DLT, in
    normalised coordinates), then Gauss-Newton on the reprojection error with Huber weights; views more than 2 px
    (and 3x the median) off are dropped and it is refined again. Returns (X, errors px, kept mask) or None."""
    n = len(uv)
    rows = []
    for (u, v), W, K in zip(uv, Ws, Ks):
        x = np.linalg.solve(K, [u, v, 1.0])
        rows += [x[0] / x[2] * W[2] - W[0], x[1] / x[2] * W[2] - W[1]]
    X = np.linalg.svd(np.array(rows))[2][-1]
    if abs(X[3]) < 1e-12:
        return None
    X = X[:3] / X[3]
    keep = np.ones(n, bool)

    def residuals(X):
        err, J = np.zeros((n, 2)), np.zeros((n, 2, 3))
        for k, ((u, v), W, K) in enumerate(zip(uv, Ws, Ks)):
            Xc = W[:3, :3] @ X + W[:3, 3]
            z = Xc[2] if abs(Xc[2]) > 1e-12 else 1e-12
            err[k] = [K[0, 0] * Xc[0] / z + K[0, 2] - u, K[1, 1] * Xc[1] / z + K[1, 2] - v]
            J[k] = np.array([[K[0, 0] / z, 0, -K[0, 0] * Xc[0] / z ** 2],
                             [0, K[1, 1] / z, -K[1, 1] * Xc[1] / z ** 2]]) @ W[:3, :3]
        return err, J

    for _ in range(3):
        for _ in range(20):
            err, J = residuals(X)
            norm = np.linalg.norm(err, axis=1)
            w = np.where(norm <= huber, 1.0, huber / np.maximum(norm, 1e-12)) * keep
            H = np.einsum("k,kij,kil->jl", w, J, J)
            g = np.einsum("k,kij,ki->j", w, J, err)
            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                return None
            X = X + step
            if np.linalg.norm(step) < 1e-10 * (1 + np.linalg.norm(X)):
                break
        err = np.linalg.norm(residuals(X)[0], axis=1)
        bad = keep & (err > max(2.0, 3 * float(np.median(err[keep]))))
        if not bad.any() or keep.sum() - bad.sum() < 2:
            break
        keep &= ~bad
    depth = np.array([(W[:3, :3] @ X + W[:3, 3])[2] for W in Ws])
    if keep.sum() < 2 or np.any(depth[keep] <= 0):
        return None
    rays = np.array([np.linalg.inv(W)[:3, 3] - X for W in Ws])[keep]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    if np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1.0, 1.0))) < 2.0:   # too narrow a baseline
        return None
    return X, err, keep


def _marker_residuals(ref: dict, tri: dict, ids: list, s: float, R: np.ndarray, t: np.ndarray) -> dict:
    """Per marker, the RMS distance (mm, on the sheet's scale) of its triangulated corners from the layout."""
    out = {}
    for i in ids:
        back = ((tri[i] - t) @ R) / s
        out[i] = float(np.sqrt(((back - ref[i]) ** 2).sum(axis=1).mean()))
    return out


def fit_sheet(tri: dict[int, np.ndarray], ref: dict[int, np.ndarray]):
    """Similarity from the sheet's layout (mm) to the triangulated marker corners: tri = s R ref + t. A consensus
    over pairs of markers first (a wrongly read marker cannot pull the fit), then refits without the markers that
    sit off the others. Returns (s, R, t, inlier ids, residuals per marker in mm) or None."""
    ids = sorted(tri)
    if len(ids) < 2:
        return None
    best, best_key = None, None
    for a, b in combinations(ids, 2):
        s, R, t = umeyama(np.r_[ref[a], ref[b]], np.r_[tri[a], tri[b]])
        if not s > 0:
            continue
        res = _marker_residuals(ref, tri, ids, s, R, t)
        inl = [i for i in ids if res[i] < 3.0]
        if not inl:
            continue
        key = (len(inl), -float(np.median([res[i] for i in inl])))
        if best_key is None or key > best_key:
            best, best_key = inl, key
    if best is None:
        return None

    def refit(chosen):
        s, R, t = umeyama(np.concatenate([ref[i] for i in chosen]), np.concatenate([tri[i] for i in chosen]))
        return s, R, t, _marker_residuals(ref, tri, ids, s, R, t)

    inliers = best
    for _ in range(10):
        if len(inliers) < 2:
            return None
        s, R, t, res = refit(inliers)
        limit = max(1.0, 3.0 * 1.4826 * float(np.median([res[i] for i in inliers])))
        new = [i for i in ids if res[i] <= limit]
        if new == inliers:
            break
        inliers = new
    if len(inliers) < 2:
        return None
    s, R, t, res = refit(inliers)
    return s, R, t, inliers, res


def find_sheet(cams: list[dict], ruler_mm: float | None = None, seed: int = 0) -> dict:
    """The scale sheet in the placed photos: {"found": True, "transform": (a, Q, b) with x_sheet = a Q x + b (COLMAP
    frame -> mm, Z up from the sheet), quality numbers...} or {"found": False, "reason"}."""
    detections = detect_markers(cams)
    ref = sheet_corners(ruler_mm)
    views_with = sum(1 for d in detections if d)
    seen_ids = sorted({i for d in detections for i in d})
    base = {"found": False, "dictionary": "DICT_4X4_50", "marker_mm": SHEET_MARKER_MM, "ruler_mm": ruler_mm,
            "photos_with_markers": views_with, "markers_seen": len(seen_ids)}
    if not seen_ids:
        return {**base, "reason": "no scale sheet markers in the photos"}
    Ws = [c["world_to_camera"] for c in cams]
    Ks = [c["K"] for c in cams]
    tri, obs_err = {}, []
    for i in seen_ids:
        where = [k for k, d in enumerate(detections) if i in d]
        if len(where) < 2:
            continue
        corners, errs, used = [], [], set()
        for corner in range(4):
            uv = np.array([detections[k][i][corner] for k in where])
            got = triangulate(uv, [Ws[k] for k in where], [Ks[k] for k in where])
            if got is None:
                break
            X, err, keep = got
            corners.append(X)
            errs += err[keep].tolist()
            used |= {where[k] for k in np.flatnonzero(keep)}
        if len(corners) == 4:
            tri[i] = np.array(corners)
            obs_err.append((i, errs, used))
    if len(tri) < MIN_SHEET_MARKERS:
        return {**base, "markers": len(tri),
                "reason": f"only {len(tri)} of the sheet's markers were seen in two or more photos (at least "
                          f"{MIN_SHEET_MARKERS} are needed)"}
    fit = fit_sheet(tri, ref)
    if fit is None:
        return {**base, "reason": "the markers found do not fit the sheet's layout"}
    s, R, t, inliers, res = fit
    used_views = set().union(*(u for i, _, u in obs_err if i in inliers))
    if len(inliers) < MIN_SHEET_MARKERS or len(used_views) < MIN_SHEET_VIEWS:
        return {**base, "markers": len(inliers), "views": len(used_views),
                "reason": f"{len(inliers)} markers in {len(used_views)} photos agree with the sheet's layout (at "
                          f"least {MIN_SHEET_MARKERS} markers in {MIN_SHEET_VIEWS} photos are needed)"}
    a, Q = 1.0 / s, R.T
    b = -a * (Q @ t)
    centres = np.array([np.linalg.inv(W)[:3, 3] for W in Ws])
    heights = (a * (centres @ Q.T) + b)[:, 2]
    if np.median(heights) <= 0:
        return {**base, "reason": "the cameras came out below the sheet - the markers were misread"}
    # quality: corner reprojection error, fit residual, scale uncertainty (bootstrap over markers, and from the
    # residuals: whichever is larger), each marker's residual
    reproj = np.concatenate([e for i, e, _ in obs_err if i in inliers])
    src = np.concatenate([ref[i] for i in inliers])
    dst = np.concatenate([tri[i] for i in inliers])
    back = ((dst - t) @ R) / s
    d = back - src
    dof = max(3 * len(src) - 7, 1)
    sigma = float(np.sqrt((d ** 2).sum() / dof))
    spread = float(np.sqrt(((src - src.mean(0)) ** 2).sum()))
    rel_analytic = sigma / max(spread, 1e-9)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(300):
        pick = rng.choice(inliers, len(inliers), replace=True)
        if len(set(pick)) < 2:
            continue
        sb = umeyama(np.concatenate([ref[i] for i in pick]), np.concatenate([tri[i] for i in pick]))[0]
        boots.append(sb / s)
    rel_boot = float(np.std(boots)) if len(boots) > 10 else rel_analytic
    uncertainty = 2.0 * max(rel_analytic, rel_boot) * 100   # ~95 %
    flat = float(np.abs(back[:, 2]).max())
    out = {**base, "found": True, "markers": len(inliers), "views": len(used_views),
           "markers_rejected": sorted(set(tri) - set(inliers)),
           "reprojection_px": round(float(np.sqrt((reproj ** 2).mean())), 3),
           "residual_mm": round(float(np.sqrt((d ** 2).sum(axis=1).mean())), 3),
           "max_residual_mm": round(float(np.linalg.norm(d, axis=1).max()), 3),
           "off_plane_mm": round(flat, 3),
           "uncertainty_pct": round(uncertainty, 3),
           "mm_per_unit": round(a, 8),
           "camera_height_mm": round(float(np.median(heights)), 1),
           "marker_residual_mm": {str(i): round(res[i], 3) for i in sorted(res)},
           "transform": (a, Q, b)}
    if out["max_residual_mm"] > 0.5:
        out["warning"] = (f"Some markers sit up to {out['max_residual_mm']:.1f} mm off the printed layout: the sheet "
                          "may not lie flat (tape it down on a flat board) or it was printed stretched")
    return out


def level_on_dots(sheet: dict, xyz: np.ndarray, part_above_mm: float = 3.0) -> dict | None:
    """The ground right where the part stands: COLMAP's exact points on the sheet's printed dots round the part.
    The markers are at the sheet's edges, where paper curls up, so the plane through them can sit above the middle
    of the sheet; the frame is tilted and lifted onto the dots' plane (xyz: COLMAP's points, COLMAP's frame). Changes
    sheet["transform"] and returns what was corrected, or None when too few dots were found or the correction is
    implausible (then the markers' plane stays)."""
    a, Q, b = sheet["transform"]
    P = a * (xyz @ Q.T) + b
    # leave out the part and a margin round it: its lowest points are near the sheet too
    part = P[(P[:, 2] > part_above_mm) & (np.abs(P[:, 0]) < SHEET_DOTS_MM[0]) & (np.abs(P[:, 1]) < SHEET_DOTS_MM[1])]
    near = P[(np.abs(P[:, 0]) < SHEET_DOTS_MM[0] / 2) & (np.abs(P[:, 1]) < SHEET_DOTS_MM[1] / 2) & (np.abs(P[:, 2]) < 2.0)]
    if len(part):
        cells = {tuple(c) for c in np.floor(part[:, :2] / 2.0).astype(int)}
        cells = {(x + dx, y + dy) for x, y in cells for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2)}
        mine = np.floor(near[:, :2] / 2.0).astype(int)
        near = near[[tuple(c) not in cells for c in mine]]
    if len(near) < 100:
        return None
    A = np.c_[near[:, 0], near[:, 1], np.ones(len(near))]
    w = np.ones(len(near))
    for _ in range(20):   # robust plane z = alpha x + beta y + gamma (Tukey)
        sol = np.linalg.lstsq(A * np.sqrt(w)[:, None], near[:, 2] * np.sqrt(w), rcond=None)[0]
        res = near[:, 2] - A @ sol
        s = max(1.4826 * float(np.median(np.abs(res))), 1e-4)
        u = res / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    inl = w > 0
    alpha, beta, gamma = (float(v) for v in sol)
    tilt = math.degrees(math.atan(math.hypot(alpha, beta)))
    if inl.sum() < 100 or tilt > 2.0 or abs(gamma) > 3.0:
        return None
    n = np.array([-alpha, -beta, 1.0])
    n /= np.linalg.norm(n)
    axis = np.cross(n, [0.0, 0.0, 1.0])
    sin, cos = np.linalg.norm(axis), float(n[2])
    Rl = np.eye(3)
    if sin > 1e-12:   # Rodrigues: the plane's normal onto z
        k = axis / sin
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rl = np.eye(3) + sin * Kx + (1 - cos) * Kx @ Kx
    g = np.array([0.0, 0.0, gamma])
    sheet["transform"] = (a, Rl @ Q, Rl @ (b - g))
    return {"points": int(inl.sum()), "tilt_deg": round(tilt, 4), "lift_mm": round(gamma, 3),
            "rms_mm": round(float(np.sqrt(np.mean(res[inl] ** 2))), 3)}


def sheet_summary(sheet: dict) -> str:
    if not sheet.get("found"):
        return f"Size unknown: no scale sheet found ({sheet.get('reason', 'not looked for')})"
    return (f"True size from the scale sheet (±{sheet['uncertainty_pct']:.2g} %): {sheet['markers']} markers in "
            f"{sheet['views']} photos, corners to {sheet['reprojection_px']:.2f} px, the layout fits to "
            f"{sheet['residual_mm']:.2f} mm" + (f", printed bar {sheet['ruler_mm']:g} mm" if sheet.get("ruler_mm")
                                               else ""))


def crop_to_sheet(P: np.ndarray, above_mm: float) -> np.ndarray:
    """Mask of the points that are not the sheet or the table (more than above_mm above the sheet) and not far
    background (within SHEET_KEEP_MM of the page, sideways)."""
    return ((P[:, 2] > above_mm) & (np.abs(P[:, 0]) < SHEET_PAGE_MM[0] / 2 + SHEET_KEEP_MM)
            & (np.abs(P[:, 1]) < SHEET_PAGE_MM[1] / 2 + SHEET_KEEP_MM))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poses", default="auto", choices=["auto", "colmap", "mapanything"])
    ap.add_argument("--no-dense", action="store_true", help="cameras only (points = COLMAP's sparse points)")
    ap.add_argument("--keep", type=float, default=0.8, help="share of the dense points kept, most confident first")
    ap.add_argument("--crop", action="store_true",
                    help="experiment: crop each photo to the part for the dense model (worse on tests: 8 vs 1.8 mm)")
    ap.add_argument("--scale-sheet", action="store_true",
                    help="look for the printed scale sheet: true size in mm, Z up from the sheet, sheet cropped away")
    ap.add_argument("--ruler-mm", type=float, default=None,
                    help="what the sheet's 100 mm bar measured on the print (corrects a printer's scaling)")
    ap.add_argument("--sheet-crop", type=float, default=0.5,
                    help="points less than this (mm) above the scale sheet are the sheet or the table: removed")
    ap.add_argument("--no-level", action="store_true",
                    help="keep the markers' plane as the ground (do not level on the dots round the part)")
    args = ap.parse_args()
    if args.ruler_mm is not None and not 90.0 <= args.ruler_mm <= 110.0:
        print(f"ERROR The 100 mm bar cannot measure {args.ruler_mm:g} mm - check the value (or print the sheet "
              "again at 100 %)", flush=True)
        return 3
    started = time.time()
    src = Path(args.images)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp())
    photos = work / "photos"
    photos.mkdir()
    # COLMAP reads JPEG/PNG/TIFF/BMP: anything else is converted (upright, EXIF rotation applied)
    from PIL import Image, ImageOps

    names = []
    for p in sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        if p.suffix.lower() in COLMAP_EXTS:
            with Image.open(p) as im:
                rotated = im.getexif().get(0x0112, 1) != 1
            if not rotated:
                shutil.copyfile(p, photos / p.name)
                names.append(p.name)
                continue
        if p.suffix.lower() in (".heic", ".heif"):
            from pillow_heif import register_heif_opener
            register_heif_opener()
        with Image.open(p) as im:
            ImageOps.exif_transpose(im).convert("RGB").save(photos / (p.stem + ".jpg"), quality=95)
        names.append(p.stem + ".jpg")
    if len(names) < 2:
        print("ERROR Need at least 2 photos", flush=True)
        return 2

    cams = None
    if args.poses in ("auto", "colmap") and len(names) >= 3:
        try:
            # colmap: only exactly placed photos, however few (colouring needs exact cameras: rough ones painted
            # the colours in the wrong places); auto: at least half of them, else MapAnything's rough cameras
            cams = colmap_cameras(photos, names, work, 0.0 if args.poses == "colmap" else 0.5)
        except RuntimeError as exc:
            say(str(exc))
            cams = None
        if cams is None and args.poses == "colmap":
            print("ERROR Fewer than 3 of the photos could be placed exactly: neighbouring photos must overlap. Take "
                  "them about 30 degrees apart all the way round the part, keeping the height the same from one "
                  "photo to the next, with the part filling most of the frame on a patterned sheet", flush=True)
            return 3
    poses = "colmap" if cams else "mapanything"

    sheet = None
    if args.scale_sheet:
        if cams:
            progress(0.55, "Looking for the scale sheet")
            try:
                sheet = find_sheet(cams, args.ruler_mm)
                if sheet.get("found") and not args.no_level:
                    sp = read_sparse(work / "txt_undistorted")
                    sheet["level"] = level_on_dots(sheet, sp["xyz"][sp["error"] < 2.0])
                    if sheet["level"]:
                        lv = sheet["level"]
                        say(f"Levelled on the sheet's dots round the part ({lv['points']} points): tilted "
                            f"{lv['tilt_deg']:.3f} deg, lifted {lv['lift_mm']:+.3f} mm")
            except Exception as exc:   # the model is still worth having without its true size
                sheet = {"found": False, "reason": f"the scale sheet could not be measured ({exc})"}
        else:
            sheet = {"found": False, "reason": "the photos could only be placed roughly, which is not exact enough "
                                               "to measure the scale sheet"}
        say(sheet_summary(sheet))
        if sheet.get("warning"):
            say("WARNING: " + sheet["warning"])

    points, colours, confidence, cameras_out = [], [], [], []
    if cams and args.no_dense:
        # the sparse points of COLMAP (in its units) are the only geometry
        say("Cameras only: no dense points")
        W_scale = 1.0
    else:
        model = load_model()
        if cams:
            boxes = part_boxes(cams, read_sparse(work / "txt_undistorted")) if args.crop else None
            if boxes and any(b is not None for b in boxes):
                say(f"Cropped {sum(b is not None for b in boxes)} of {len(boxes)} photos to the part")
            preds = dense_from_cameras(model, cams, boxes)
        else:
            if args.poses == "auto":
                say("Using MapAnything's own camera estimate (rough)")
            preds = dense_images_only(model, [photos / n for n in names])
        progress(0.9, "Collecting the points")
        for i, p in enumerate(preds):
            pts = p["pts3d"][0].float().cpu().numpy()
            mask = p["mask"][0].squeeze(-1).bool().cpu().numpy()
            conf = p["conf"][0].float().cpu().numpy()
            img = p["img_no_norm"][0].float().cpu().numpy()
            ok = mask & np.isfinite(pts).all(axis=-1)
            points.append(pts[ok])
            colours.append(img[ok])
            confidence.append(conf[ok])
        W_scale = 1000.0
        if cams:
            # the dense points come in MapAnything's frame (metres by its size estimate): move them into COLMAP's
            # frame, then express both in mm by that estimate (x_colmap = s R x_pred + t)
            pred_centres = np.array([p["camera_poses"][0].float().cpu().numpy()[:3, 3] for p in preds])
            in_centres = np.array([np.linalg.inv(c["world_to_camera"])[:3, 3] for c in cams])
            s, R, t = umeyama(pred_centres, in_centres)
            gap = np.linalg.norm((s * (R @ pred_centres.T)).T + t - in_centres, axis=1)
            spread = np.linalg.norm(in_centres - in_centres.mean(0), axis=1).mean()
            say(f"Dense cameras agree with COLMAP's to {np.median(gap) / max(spread, 1e-12) * 100:.2f} % of the "
                "camera spread")
            W_scale = 1000.0 / s                                  # COLMAP units -> mm (MapAnything's estimate)
            points, colours, confidence = refine_dense(preds, cams, (s, R, t), read_sparse(work / "txt_undistorted"))

    # the output frame: x_out = a Q x + b from COLMAP's frame - the scale sheet's frame (true mm, Z up from the
    # sheet) when it was found, else COLMAP's frame in mm by MapAnything's size estimate
    on_sheet = bool(sheet and sheet.get("found"))
    a, Q, b = sheet["transform"] if on_sheet else (W_scale, np.eye(3), np.zeros(3))
    to_out = lambda X: a * (np.asarray(X, float) @ Q.T) + b
    if cams:
        for c in cams:
            W = c["world_to_camera"].copy()
            W[:3, :3] = c["world_to_camera"][:3, :3] @ Q.T
            W[:3, 3] = a * c["world_to_camera"][:3, 3] - W[:3, :3] @ b
            dst = out / "images" / (Path(c["image"]).stem + ".jpg")
            Image.open(c["path"]).convert("RGB").save(dst, quality=92)
            cameras_out.append({"image": c["image"], "file": f"images/{dst.name}", "K": c["K"].round(4).tolist(),
                                "width": c["width"], "height": c["height"], "world_to_camera": W.round(8).tolist(),
                                "cam_to_world": np.linalg.inv(W).round(8).tolist()})
    else:
        for name, p in zip(names, preds):
            pose = p["camera_poses"][0].float().cpu().numpy().astype(float)
            pose[:3, 3] *= 1000.0
            img = p["img_no_norm"][0].float().cpu().numpy()
            img = np.clip(img * 255.0 if img.max() <= 1.5 else img, 0, 255).astype(np.uint8)
            dst = out / "images" / (Path(name).stem + ".jpg")
            Image.fromarray(img).save(dst, quality=92)
            h, w = img.shape[:2]
            cameras_out.append({"image": name, "file": f"images/{dst.name}",
                                "K": p["intrinsics"][0].float().cpu().numpy().round(4).tolist(), "width": w,
                                "height": h, "world_to_camera": np.linalg.inv(pose).round(8).tolist(),
                                "cam_to_world": pose.round(8).tolist()})

    if points:
        P = np.concatenate(points)
        C = np.concatenate(colours)
        Wc = np.concatenate(confidence)
        if not len(P):
            print("ERROR The photos gave no 3D points - use more photos with overlap all the way round", flush=True)
            return 3
        keep = Wc >= np.quantile(Wc, 1.0 - min(max(args.keep, 0.05), 1.0))
        P, C = to_out(P[keep]), C[keep]
        C = np.clip(C * 255.0 if C.max() <= 1.5 else C, 0, 255).astype(np.uint8)
    else:
        P, C = sparse_points(work)
        P = to_out(P)
    if on_sheet:   # without the sheet and the table under it, and far background
        above = crop_to_sheet(P, args.sheet_crop)
        sheet["cropped_points"] = int((~above).sum())
        say(f"Cropped the sheet and the table away: {int(above.sum()):,} of {len(P):,} points stand on the sheet")
        P, C = P[above], C[above]
        if not len(P):
            print("ERROR Nothing was found standing on the scale sheet - put the part in the middle of the sheet "
                  "and photograph it all the way round", flush=True)
            return 3
    write_ply(out / "points.ply", P.astype(np.float32), C)
    if cams:   # exact (triangulated) points in the cameras' frame: what lining up with a scan should trust
        sp = read_sparse(work / "txt_undistorted")
        ok = sp["error"] < 2.0
        S, SC = to_out(sp["xyz"][ok]), sp["rgb"][ok]
        if on_sheet:
            above = crop_to_sheet(S, args.sheet_crop)
            S, SC = S[above], SC[above]
        write_ply(out / "sparse.ply", S.astype(np.float32), SC)
    info = None
    if sheet is not None:
        info = {k: v for k, v in sheet.items() if k != "transform"}
        info["summary"] = sheet_summary(sheet)
    (out / "cameras.json").write_text(json.dumps({"model": MODEL, "poses": poses, "photos": len(names),
                                                  "registered": len(cameras_out),
                                                  "units": "mm, scale sheet" if on_sheet else "mm, estimated",
                                                  "scale_sheet": info, "cameras": cameras_out}),
                                      encoding="utf-8")
    extent = (np.percentile(P, 99, axis=0) - np.percentile(P, 1, axis=0)).round(2).tolist() if len(P) else [0, 0, 0]
    progress(1.0, "Done")
    result = {"photos": len(names), "registered": len(cameras_out), "points": int(len(P)), "extent_mm": extent,
              "seconds": round(time.time() - started, 1), "poses": poses, "model": MODEL}
    if info is not None:
        result["scale_sheet"] = info
    print("RESULT " + json.dumps(result), flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return 0


def sparse_points(work: Path):
    txt = work / "txt_undistorted"
    P, C = [], []
    for line in (txt / "points3D.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        v = line.split()
        P.append([float(x) for x in v[1:4]])
        C.append([int(x) for x in v[4:7]])
    return np.array(P, float).reshape(-1, 3), np.array(C, np.uint8).reshape(-1, 3)


def give_back(folder: Path) -> None:
    """The container runs as root: make what it wrote belong to the owner of the output folder, so CloudClean
    (another user) can delete it."""
    try:
        st = folder.stat()
        for p in [folder, *folder.rglob("*")]:
            os.chown(p, st.st_uid, st.st_gid)
    except OSError:
        pass


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception as exc:   # one plain line for CloudClean, the traceback for the log
        import traceback

        traceback.print_exc()
        print(f"ERROR {exc}", flush=True)
    finally:
        out_arg = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--out"), None)
        if out_arg:
            give_back(Path(out_arg))
    sys.exit(code)
