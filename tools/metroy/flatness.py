"""How flat does a flat surface measure? Split the error into noise and structure.

A plain RMS about a fitted plane mixes two very different things: random scatter from point to point (stripe
localisation noise, laser speckle) and smooth structured error across the patch (calibration, bias that depends on
where a stripe falls). Fusing many frames averages the first away and does nothing to the second, so they are
reported separately:

* random     - RMS of each residual about the median residual of its 4 mm cell
* systematic - RMS of the cell medians themselves

    python flatness.py frame.raw [frame.raw ...] camparam.yaml [--method lines|order|steger-notrack|v2|v1] [--gate mm]
"""
from __future__ import annotations

import sys

import numpy as np


def fit_plane(pts: np.ndarray, gate: float = 1.0, iters: int = 1500, seed: int = 0):
    """RANSAC with a coarse gate, then least squares refits that tighten the gate to 4 robust sigma."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        inl = np.abs((pts - s[0]) @ (n / nn)) < gate
        if best is None or inl.sum() > best.sum():
            best = inl
    inl = best
    for _ in range(5):
        c = pts[inl].mean(0)
        _, _, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
        normal = vt[2]
        r = (pts - c) @ normal
        sig = 1.4826 * np.median(np.abs(r[inl] - np.median(r[inl])))
        inl = np.abs(r) < max(4 * sig, 0.05)
    return c, vt, r, inl


def evaluate(pts: np.ndarray, cell: float = 4.0, gate: float = 1.0) -> dict:
    c, vt, r, inl = fit_plane(pts, gate)
    P, res = pts[inl], r[inl]
    uv = np.c_[(P - c) @ vt[0], (P - c) @ vt[1]]
    keys = np.floor(uv / cell).astype(np.int64)
    keys = keys[:, 0] * 100_003 + keys[:, 1]
    order = np.argsort(keys)
    ks, rs = keys[order], res[order]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    meds, noise = [], np.full(len(rs), np.nan)
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(ks)]):
        if b - a >= 8:
            m = np.median(rs[a:b])
            meds.append((m, b - a))
            noise[a:b] = rs[a:b] - m
    meds = np.array(meds)
    sysrms = np.sqrt(np.average(meds[:, 0] ** 2, weights=meds[:, 1])) if len(meds) else np.nan
    return {
        "points": len(pts),
        "inliers": float(inl.mean()),
        "rms": float(res.std()),
        "robust_sigma": float(1.4826 * np.median(np.abs(res - np.median(res)))),
        "random": float(np.sqrt(np.nanmean(noise ** 2))),
        "systematic": float(sysrms),
        "p95": float(np.percentile(np.abs(res), 95)),
        "patch": (float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1]))),
        "tilt_deg": float(np.degrees(np.arccos(abs(vt[2] @ [0, 0, 1])))),
        "distance": float(np.linalg.norm(c)),
    }


def line(name: str, m: dict) -> str:
    um = lambda v: f"{v * 1000:4.0f}"
    return (f"{name:14s} pts {m['points']:6d}  inl {m['inliers']:5.1%}  RMS {um(m['rms'])} um  "
            f"robust {um(m['robust_sigma'])}  random {um(m['random'])}  systematic {um(m['systematic'])}  "
            f"p95 {um(m['p95'])}  patch {m['patch'][0]:.0f}x{m['patch'][1]:.0f} mm @ {m['distance']:.0f} mm")


if __name__ == "__main__":
    args = sys.argv[1:]
    method, gate = "lines", 1.0
    if "--gate" in args:
        i = args.index("--gate")
        gate = float(args[i + 1])
        del args[i:i + 2]
    if "--method" in args:
        i = args.index("--method")
        method = args[i + 1]
        del args[i:i + 2]
    *frames, calib = args
    from stripes import frame_points
    for f in frames:
        pts = frame_points(f, calib, "steger" if method == "steger+track" else method)
        print(line(f"{f.rsplit('/', 1)[-1]}", evaluate(pts, gate=gate)))
