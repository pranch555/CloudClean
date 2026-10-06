"""Still photos from the MetroY's own stereo cameras for the outline check (docs/outline-check.md).

Runs on the machine the scanner is plugged into (Linux: the metroy_usb driver's HID and V4L2 access). Revo Metro and
CloudClean's capture must not be using the scanner. The laser is switched off, the exposure is set so the backlit
paper of the check sheet sits at about 70 % of full scale (never clipped), and N frames are averaged per camera.

    python tools/outline_check/metroy_photo.py OUT [--name screw] [--frames 32] [--exposure-us auto] [--gain 1]
                                                   [--fill 0] [--look]

Writes into OUT:
  NAME-L.png, NAME-R.png   the left / right camera, raw unrectified 1600 x 1200, averaged, as 16-bit PNG. The raw
                           pixels are linear; they are stored sRGB-encoded because the outline check reads 16-bit
                           PNG as sRGB (cloudclean/outline_camera.load_photo), so it gets the linear values back.
  NAME-preview.jpg         both cameras side by side, small, with the check sheet's markers outlined
  camparam.yaml            the factory calibration, read off this scanner now (the check's --metroy-camparam)
  NAME.json                settings, levels, frames used, markers seen per camera
--look: one quick exposure search and preview, no averaged photo (for aiming the scanner at the sheet).

Afterwards the scanner is left as Revo Metro's start-up leaves it (laser on, general preset).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from cloudclean.capture.metroy import hid as hidmod  # noqa: E402
from cloudclean.capture.metroy.scanner import read_device_files  # noqa: E402
from cloudclean.capture.metroy.v4l2 import HEIGHT, Stream  # noqa: E402

TARGET = 0.70            # the paper's bright level (99.5th percentile) as a share of full scale
CLIP = 250               # 8-bit values at or above this count as clipped
# (frame time, exposure) in microseconds: frame time must exceed exposure; 0x910 and 0x911 always go together
LADDER = [(8000, 500), (8000, 1000), (8000, 2000), (8000, 4000), (8000, 7000), (16000, 14000), (33000, 30000)]


def srgb_encode(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def grab(stream: Stream, n: int, settle_s: float = 0.6) -> tuple[list[np.ndarray], dict]:
    """n frames after settle_s of discarded ones; drops frames whose brightness is off the median (a frame type
    that differs) and counts gaps in the driver's sequence."""
    t_end = time.time() + settle_s
    while time.time() < t_end:
        stream.read(0.5)
    frames, seqs = [], []
    deadline = time.time() + 10 + n * 0.1
    while len(frames) < n and time.time() < deadline:
        f = stream.read(1.0)
        if f is not None:
            frames.append(f.image)
            seqs.append(f.sequence)
    if not frames:
        raise RuntimeError("no frames from the scanner (is it streaming? is something else using it?)")
    means = np.array([fr.mean() for fr in frames])
    med = float(np.median(means))
    keep = [fr for fr, m in zip(frames, means) if abs(m - med) <= 0.03 * max(med, 1.0)]
    gaps = int(sum(max(0, b - a - 1) for a, b in zip(seqs, seqs[1:])))
    return keep, {"grabbed": len(frames), "kept": len(keep), "dropped_by_brightness": len(frames) - len(keep),
                  "sequence_gaps": gaps}


def levels(img: np.ndarray) -> dict:
    out = {}
    for cam, part in (("L", img[:HEIGHT]), ("R", img[HEIGHT:])):
        out[cam] = {"p995": float(np.percentile(part, 99.5)), "clipped_pct": float(100 * (part >= CLIP).mean()),
                    "mean": float(part.mean())}
    return out


def choose_exposure(hid, stream, gain: float, fill: int, log) -> tuple[int, int, list[dict]]:
    """The longest ladder step whose paper stays below the target (neither camera clipped), then scaled toward the
    target (the raw sensor is linear) and checked once more."""
    tried, best = [], None
    for ft, ex in LADDER:
        hid.acquisition(ex, gain, fill, ft)
        frames, _ = grab(stream, 3)
        lv = levels(np.mean(frames, axis=0))
        hi = max(lv["L"]["p995"], lv["R"]["p995"]) / 255.0
        clipped = max(lv["L"]["clipped_pct"], lv["R"]["clipped_pct"])
        tried.append({"frame_us": ft, "exposure_us": ex, "paper_level": round(hi, 3), "clipped_pct": round(clipped, 3)})
        log(f"  exposure {ex:6d} us: paper at {100 * hi:5.1f} % of full scale, clipped {clipped:.2f} %")
        if hi < 0.97 and clipped < 0.2:
            best = (ft, ex, hi)
        if hi >= TARGET:
            break
    if best is None:
        raise RuntimeError("even the shortest exposure clips the paper: turn the light pad down")
    ft, ex, hi = best
    want = int(min(30000, max(100, ex * TARGET / max(hi, 1e-3))))
    ft = max(ft, 8000 if want <= 7000 else 16000 if want <= 14000 else 33000)
    if want != ex:
        hid.acquisition(want, gain, fill, ft)
        frames, _ = grab(stream, 3)
        lv = levels(np.mean(frames, axis=0))
        hi = max(lv["L"]["p995"], lv["R"]["p995"]) / 255.0
        clipped = max(lv["L"]["clipped_pct"], lv["R"]["clipped_pct"])
        tried.append({"frame_us": ft, "exposure_us": want, "paper_level": round(hi, 3), "clipped_pct": round(clipped, 3)})
        log(f"  exposure {want:6d} us: paper at {100 * hi:5.1f} % of full scale, clipped {clipped:.2f} %  (chosen)")
        if clipped >= 0.2:
            want = int(want * 0.8)
            hid.acquisition(want, gain, fill, ft)
            log(f"  clipped: backed off to {want} us")
        ex = want
    if hi < 0.25:
        log("  the paper is still dark at the longest exposure: the cameras may not see the light pad's light well; "
            "try --gain 2 or 4, or a brighter pad")
    return ft, ex, tried


def markers(gray8: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    try:
        import cv2
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                      cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray8)
    except Exception:
        return [], []
    if ids is None:
        return [], []
    keep = [(int(i), c) for i, c in zip(ids.ravel(), corners) if 0 <= int(i) <= 15]
    return sorted(i for i, _ in keep), [c for _, c in keep]


def preview(avg: np.ndarray, path: Path) -> dict:
    """Both cameras side by side (sRGB-encoded for viewing), markers outlined; returns the marker ids per camera."""
    from PIL import Image, ImageDraw

    seen = {}
    tiles = []
    for cam, part in (("L", avg[:HEIGHT]), ("R", avg[HEIGHT:])):
        g8 = np.clip(np.round(srgb_encode(part / 255.0) * 255), 0, 255).astype(np.uint8)
        ids, corners = markers(g8)
        seen[cam] = ids
        im = Image.fromarray(g8).convert("RGB")
        d = ImageDraw.Draw(im)
        for c in corners:
            pts = [tuple(map(float, p)) for p in c.reshape(4, 2)]
            d.line(pts + [pts[0]], fill=(255, 60, 0), width=5)
        d.text((20, 20), f"{cam}: {len(ids)} of 16 markers", fill=(255, 60, 0))
        tiles.append(im.resize((800, 600)))
    both = Image.new("RGB", (1600, 600))
    both.paste(tiles[0], (0, 0))
    both.paste(tiles[1], (800, 0))
    both.save(path, quality=90)
    return seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--name", default="metroy")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--exposure-us", default="auto")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--fill", type=int, default=0, help="the scanner's fill light, 0-255 (0: backlight only)")
    ap.add_argument("--look", action="store_true", help="exposure search and preview only")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    meta: dict = {"name": args.name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gain": args.gain, "fill": args.fill}
    stream = Stream()
    try:
        with hidmod.MetroyHid() as hid:
            files = read_device_files(hid)
            (out / "camparam.yaml").write_bytes(files.calibration)
            meta.update(serial=files.serial, calibration_fingerprint=files.fingerprint)
            log(f"scanner {files.serial}: calibration read off the device -> {out / 'camparam.yaml'}")
            try:
                hid.startup()
                hid.laser(False)
                time.sleep(0.3)
                if args.exposure_us == "auto":
                    log("choosing the exposure (laser off):")
                    ft, ex, tried = choose_exposure(hid, stream, args.gain, args.fill, log)
                    meta["exposure_search"] = tried
                else:
                    ex = int(args.exposure_us)
                    ft = max(8000, ex + 1000)
                    hid.acquisition(ex, args.gain, args.fill, ft)
                meta.update(frame_us=ft, exposure_us=ex)
                n = 4 if args.look else args.frames
                frames, info = grab(stream, n, settle_s=0.8)
                meta["frames"] = info
                avg = np.mean(np.stack(frames).astype(np.float64), axis=0)
                meta["levels"] = levels(avg)
                seen = preview(avg, out / f"{args.name}-preview.jpg")
                meta["markers"] = {cam: {"count": len(ids), "ids": ids} for cam, ids in seen.items()}
                log(f"markers seen: left {len(seen['L'])} of 16, right {len(seen['R'])} of 16 "
                    f"(preview: {out / (args.name + '-preview.jpg')})")
                if not args.look:
                    from PIL import Image
                    for cam, part in (("L", avg[:HEIGHT]), ("R", avg[HEIGHT:])):
                        v16 = np.round(srgb_encode(part / 255.0) * 65535).astype(np.uint16)
                        Image.fromarray(v16).save(out / f"{args.name}-{cam}.png")
                    log(f"saved {args.name}-L.png and {args.name}-R.png ({info['kept']} frames averaged, "
                        f"{info['sequence_gaps']} dropped by the driver)")
            finally:
                # leave the scanner as Revo Metro's start-up does: general preset, laser on
                p = hidmod.SURFACE_PRESETS["general"]
                hid.acquisition(p["exposure_us"], p["gain"], p["fill_light"], hidmod.FRAME_TIME_US)
                hid.laser(True)
    finally:
        stream.close()
    (out / f"{args.name}.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
