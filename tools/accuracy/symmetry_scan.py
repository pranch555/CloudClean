"""How well is the relative pose of two bolt scans determined? (accuracy investigation 2026-09-24)

A thread is invariant under a screw motion (rotate by phi about its axis and advance pitch * phi / 360 along it),
so between two scans of a bolt only the head can fix that motion. This scans the screw motion over a full turn
from a given pose, plus a pure axial shift, and reports how well each region of scan B agrees with scan A:
the fraction of B's points within 3 sigma of A's surface (both facing the same way) and the median separation.

    python tools/accuracy/symmetry_scan.py <assets dir> --pose pose.npy [--out symmetry.json]

Asset ids default to the 2026-09-17 bolt workspace (scan 05 = A, scan 06 = B)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cloudclean import accuracy as acc  # noqa: E402
from cloudclean.io import load  # noqa: E402
from cloudclean.register import estimate_noise  # noqa: E402
from real_bolt import SCAN_A, SCAN_B, bolt_frame, coords  # noqa: E402


def screw(axis, origin, phi, advance):
    """4x4: rotate by phi (rad) about the line (origin, axis) and move `advance` along it."""
    R = acc._rotation(axis * phi)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin - R @ origin + advance * axis
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assets", type=Path)
    ap.add_argument("--pose", type=Path, required=True, help=".npy 4x4 pose of scan B in scan A")
    ap.add_argument("--out", type=Path, default=Path("symmetry.json"))
    ap.add_argument("--step", type=float, default=1.0, help="rotation step in degrees")
    args = ap.parse_args()
    A = load(args.assets / SCAN_A / "data.ply")
    B = load(args.assets / SCAN_B / "data.ply")
    pa, na = np.asarray(A.points), np.asarray(A.normals)
    pb, nb = np.asarray(B.points), np.asarray(B.normals)
    T0 = np.load(args.pose)
    frame = bolt_frame(pa, na)
    model = frame["thread"]["model"]
    axis, origin, pitch, hand = model["axis"], model["origin"], model["pitch"], model["hand"]
    surface = acc.SurfaceModel(A)
    sigma = math.hypot(estimate_noise(pa), estimate_noise(pb))
    within = 3 * sigma
    Pb0 = acc.apply_transform(T0, pb)
    Nb0 = nb @ T0[:3, :3].T
    t, r, _ = coords(frame, Pb0)
    rng = np.random.default_rng(0)
    regions = {"head": np.flatnonzero(r > 15.5), "thread": np.flatnonzero((r < 13.5) & (t > -30))}
    for k, idx in regions.items():
        if len(idx) > 40_000:
            regions[k] = np.sort(rng.choice(idx, 40_000, replace=False))

    def score(T):
        out = {}
        for k, idx in regions.items():
            P = acc.apply_transform(T, Pb0[idx])
            N = Nb0[idx] @ T[:3, :3].T
            d, _, gap, ok = surface.query(P, N, 30.0)
            good = ok & (gap <= 0.3)
            out[k] = {"within": float(np.mean(good & (np.abs(d) <= within))),
                      "median_abs": float(np.median(np.abs(d[good]))) if good.any() else None,
                      "signed_median": float(np.median(d[good])) if good.any() else None}
        return out

    result = {"sigma_combined": sigma, "within_mm": within, "pitch": pitch, "hand": hand, "screw": [], "axial": []}
    for deg in np.arange(-180, 180 + 1e-9, args.step):
        phi = math.radians(deg)
        T = screw(axis, origin, phi, hand * pitch * phi / (2 * math.pi))
        s = score(T)
        result["screw"].append({"deg": float(deg), **{f"{k}_{m}": v[m] for k, v in s.items() for m in v}})
        if abs(deg % 15) < 1e-9:
            print(f"screw {deg:+7.1f} deg: head within {100 * s['head']['within']:5.1f} % (median {s['head']['median_abs']:.4f}),"
                  f" thread within {100 * s['thread']['within']:5.1f} % (median {s['thread']['median_abs']:.4f})", flush=True)
    for shift in np.arange(-0.5, 0.5 + 1e-9, 0.025):
        T = screw(axis, origin, 0.0, shift)
        s = score(T)
        result["axial"].append({"shift": float(shift), **{f"{k}_{m}": v[m] for k, v in s.items() for m in v}})
        print(f"axial {shift:+.3f} mm: head within {100 * s['head']['within']:5.1f} % (median {s['head']['median_abs']:.4f}), "
              f"thread within {100 * s['thread']['within']:5.1f} % (median {s['thread']['median_abs']:.4f})", flush=True)
    args.out.write_text(json.dumps(acc.jsonable(result), indent=1))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
