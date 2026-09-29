"""How well does MapAnything find the cameras of real photos? A check on DTU scans (real photos of small objects on
a table, with the true camera poses in millimetres), run inside the cloudclean-recon container:

    docker run --rm --gpus all --ipc host -v <dtu>:/dtu:ro -v <hf>:/hf -v <repo>/tools/recon:/code:ro \
        cloudclean-recon:latest python3 /code/dtu_check.py /dtu/scan37 [--views 49]

A scan folder holds image/000000.png... and cameras.npz (NeuS format: world_mat_i = K [R|t], world units mm).
For each mode (photos only; photos + the true focal length, as a phone's EXIF would give) it prints the metric size
error and, after a similarity fit of the camera centres, the camera position (mm) and rotation (deg) errors.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def rq(M):
    """M = K R with K upper triangular (positive diagonal) and R a rotation."""
    Q, Rm = np.linalg.qr(np.flipud(M).T)
    K = np.flipud(np.fliplr(Rm.T))
    R = np.flipud(Q.T)
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    if np.linalg.det(R) < 0:
        R = -R
    return K / K[2, 2], R


def truth(scan: Path, n: int):
    cams = np.load(scan / "cameras.npz")
    out = []
    for i in range(n):
        P = cams[f"world_mat_{i}"][:3, :4]
        K, R = rq(P[:, :3])
        t = np.linalg.solve(K, P[:, 3])
        C = -R.T @ t
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R.T, C
        out.append((K, pose))
    return out


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((xs ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def score(poses, gt):
    A = np.array([p[:3, 3] for p in poses]) * 1000.0     # m -> mm
    B = np.array([g[1][:3, 3] for g in gt])
    s, R, t = umeyama(A, B)
    centre = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    rot = []
    for p, g in zip(poses, gt):
        dR = g[1][:3, :3].T @ (R @ p[:3, :3])
        rot.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    rot = np.array(rot)
    spread = np.linalg.norm(B - B.mean(0), axis=1).mean()
    return {"size_estimate_vs_true": round(float(1 / s), 4),
            "camera_centre_mm": {"median": round(float(np.median(centre)), 2), "p90": round(float(np.percentile(centre, 90)), 2)},
            "camera_centre_rel_%": round(float(np.median(centre) / spread * 100), 3),
            "rotation_deg": {"median": round(float(np.median(rot)), 3), "p90": round(float(np.percentile(rot, 90)), 3)}}


def colmap_poses(files, work: Path):
    """Camera poses from classical structure from motion (COLMAP, CPU): cam-to-world 4x4 per photo (None when
    a photo was not registered), in COLMAP's arbitrary units."""
    import shutil
    import subprocess

    images = work / "images"
    images.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(f, images / f.name)
    db, sparse = work / "db.db", work / "sparse"
    sparse.mkdir(exist_ok=True)
    run = lambda *a: subprocess.run(["colmap", *a], check=True, capture_output=True, text=True)
    run("feature_extractor", "--database_path", str(db), "--image_path", str(images),
        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu", "0", "--SiftExtraction.max_image_size", "1600")
    run("exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", "0")
    run("mapper", "--database_path", str(db), "--image_path", str(images), "--output_path", str(sparse))
    models = sorted(sparse.iterdir(), key=lambda d: -len(list(d.iterdir())))
    txt = work / "txt"
    txt.mkdir(exist_ok=True)
    run("model_converter", "--input_path", str(models[0]), "--output_path", str(txt), "--output_type", "TXT")
    poses = {}
    lines = [l for l in (txt / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    for line in lines[::2]:
        v = line.split()
        qw, qx, qy, qz, tx, ty, tz = map(float, v[1:8])
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                      [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R.T, -R.T @ np.array([tx, ty, tz])
        poses[v[9]] = pose
    return [poses.get(f.name) for f in files], len(models)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--views", type=int, default=49)
    ap.add_argument("--modes", default="images,intrinsics")
    args = ap.parse_args()
    scan = Path(args.scan)
    files = sorted((scan / "image").glob("*.png"))[: args.views]
    gt = truth(scan, len(files))

    import torch
    from PIL import Image
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images, preprocess_inputs

    model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
    report = {"scan": scan.name, "views": len(files)}
    for mode in args.modes.split(","):
        if mode == "colmap":
            import tempfile

            t0 = time.time()
            poses, n_models = colmap_poses(files, Path(tempfile.mkdtemp()))
            seconds = time.time() - t0
            found = [i for i, p in enumerate(poses) if p is not None]
            report[mode] = {**score([poses[i] for i in found], [gt[i] for i in found]), "registered": len(found),
                            "models": n_models, "seconds": round(seconds, 1)}
            print(mode, json.dumps(report[mode]), flush=True)
            continue
        if mode == "images":
            views = load_images([str(f) for f in files])
        else:
            views = preprocess_inputs([{"img": torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))),
                                        "intrinsics": torch.from_numpy(K.astype(np.float32))}
                                       for f, (K, _) in zip(files, gt)])
        t0 = time.time()
        with torch.no_grad():
            preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                                apply_mask=True, mask_edges=True, apply_confidence_mask=False)
        seconds = time.time() - t0
        poses = [p["camera_poses"][0].float().cpu().numpy().astype(float) for p in preds]
        report[mode] = {**score(poses, gt), "seconds": round(seconds, 1)}
        print(mode, json.dumps(report[mode]), flush=True)
    print("RESULT " + json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
