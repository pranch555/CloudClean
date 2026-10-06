"""The outline check: a part against its golden model from one photo (docs/outline-check.md).

An optical comparator (shadowgraph) with a phone: the part lies in the blank middle of the printed photo check sheet
(scale_sheet style "check") on a light pad, so the paper glows, the sheet's black markers stay dark and the part is
a dark silhouette. One photo, roughly straight down, instead of a 20-30 minute scan. Nothing is reconstructed: the
golden model itself is placed where the part lies and its outline is compared with the edge in the photo.

Per photo:

1. The camera (outline_camera.Camera: calibrated once from photos of the sheet, or the MetroY's own calibration)
   and the sheet's pose from its markers (outline_camera.sheet_pose): where every pixel looks, in millimetres on the
   sheet. The print scale comes from the caliper reading of the sheet's 100 mm bar (and optionally of the 120 mm
   marker column: printers scale x and y differently; a sheet printed at another scale works when it is measured).
2. The part's silhouette in the blank middle (a top-down map of the sheet, dark = part).
3. The part's pose. It rests ON the sheet: one view cannot tell "bigger" from "closer to the camera", so the depth is
   never free. The golden mesh's stable resting poses (convex-hull faces with the centre of mass above them) are
   tried; for each, the position and turn on the sheet whose shadow (from the camera centre) best overlaps the
   silhouette; the best is refined on the edges themselves.
4. The outline. The golden mesh is projected (with distortion) and its occluding contour taken: edges between
   triangles facing the camera and triangles facing away, kept only where they really are the silhouette's edge
   (ray casts on both sides over the whole search window: no corners, no hidden edges). For each sample the image
   edge is searched along the outline's normal: the 50 % crossing between the paper and the part, on linear light,
   to a few hundredths of a pixel.
5. What the outline measures: features of the golden model, each with an outward shift d.
   - Round (turned) parts - bolts, screws, pins, shafts - are recognised from the golden mesh (an axis the part is
     round about, knurls and threads included) and described by their profile along it, the way a shadowgraph reads
     them: shoulders (flat faces across the axis: the ends, a head's underside), zones of one outer diameter (a head,
     a shank; knurl crests), and threads (crests, roots, flanks and the pitch, found from the profile's period and
     helix). The part may lie turned any way about its own axis: that roll is fitted too (a thread's outline shifts
     along the axis with it).
   - Other parts: golden.find_faces' flat and round faces, as in the golden model check.
   Samples on corners, chamfers and fillets measure nothing (they still show in the picture, grey).
6. Pose and shifts together. A sample on a feature moves with its shift (a sample on the edge between two faces with
   both: the smallest move that keeps it on both), and so do the points the part rests on (its lowest points and
   the resting face, which set its height and tilt). One robust least-squares fit finds the part's position, turn,
   roll, small tilts (only with fit_tilt) and all the shifts at once. Fitting the pose alone would let it "explain" a
   longer part by tilting it (perspective) or by sliding it.
7. Measurements from the shifts: overall length, head height, length under head, diameters (knurl crests, thread
   major / minor), thread pitch for round parts; overall size, thickness / gap, steps and diameters (as the golden
   model check) for others. Features not on the outline: "Not measured from this photo".
8. Uncertainty, per value, as standard uncertainties combined and doubled (k = 2, like golden.py): statistical (from
   the fit, at most N_EFF samples per feature counted as independent), camera and scale (a Monte Carlo redoing the
   sheet pose and the fit on the same image edges with the intrinsics drawn from the calibration's covariance, an
   assumed focus change and the caliper readings within their limits), placement (marker corner noise, the tilt,
   the paper's height under the part, and the features that are not on the outline, each within assumed limits),
   the printer's toner spread and the part's edge against the markers' edge (both assumed, marked so). Verdicts
   follow ISO 14253-1 (golden._judge).

The markers also calibrate the edge finder: their black squares' edges in the same photo have the same bias as the
part's dark silhouette (blur on a tone curve, over-exposure and sharpening shift a dark-on-bright edge the same way),
so the markers' dilation (outline_camera.sheet_pose) is taken off the part's edges (edge_correction "markers").

check_photos(golden, photos, camera, params) -> {"photos": [...], "combined": [...], ...}; overlays per photo.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

from .compare import ReferenceSurface
from .golden import AXES, N_EFF, _adjacent_pairs, _face_labels, _judge, _plane_frame, _robust_sigma, find_faces
from .outline_camera import Camera, Photo, SheetPose, edge_blur, edge_offsets, find_markers, focal_from_photo, \
    load_photo, lm, sheet_pose, sheet_scale
from .params import ParamsMixin

Log = Callable[[str], None]

CLEAR = (-70.0, 70.0, -90.0, 90.0)     # the blank middle between the markers as drawn (x0, x1, y0, y1), mm
MAP_PPM = 3.0                           # the top-down silhouette map (px per mm)
MIN_SENSITIVITY = 0.6                   # samples where the outline moves less than this share of a pixel-per-mm
                                        # when their feature moves (not seen edge-on there) measure nothing
FLANK_SENSITIVITY = 0.25                # the same for thread flanks (seen at the flank angle)
D_SIGMA = 1.0                           # weak prior on each shift (mm): only settles what the outline cannot tell
                                        # apart (sliding the part vs. moving both its ends)
K_U = 2.0                               # coverage factor of the expanded uncertainty
POSE_STEPS = np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-5])   # x, y (mm), turn, tilts, roll (rad)
MC_MAX_SAMPLES = 3000                   # outline samples used in each Monte Carlo refit


@dataclass
class OutlineParams(ParamsMixin):
    tolerance: float = 0.1       # +/- mm
    paper: str = "a4"            # a4 | letter (only for pictures; the markers sit in the same place on both)
    bar_mm: float = 0.0          # the 100 mm bar as the caliper reads it; 0: not measured (then ±1 % is assumed)
    height_mm: float = 0.0       # top edge of marker 15 to bottom edge of marker 13 (120 mm drawn); 0: not measured
    bar_u: float = 0.05          # ± limits of each caliper reading (mm, rectangular)
    dot_gain_u: float = 0.02     # ± limits of the printer's toner spread per edge (mm, rectangular; assumed)
    edge_u_px: float = 0.15      # standard uncertainty of the part's edge against the markers' edge (px; assumed)
    f_extra_pct: float = 0.3     # ± limits of a focal-length change between calibration and check (focus; assumed)
    tilt_sigma_deg: float = 0.05  # how far the part may tilt off its resting face (degrees; dirt, burrs): in the
                                  # uncertainty, and the prior when fit_tilt
    fit_tilt: bool = False       # fit the tilt too (one view barely sees it: small edge errors would turn into
                                 # length errors through perspective, so by default the part rests on its face)
    flatness_mm: float = 0.05    # ± limits of the paper's height under the part against the markers' plane (assumed)
    unseen_u: float = -1.0       # ± limits of features not on the outline (mm, assumed); < 0: the tolerance
    edge_correction: str = "markers"   # markers (take the markers' dilation off the part's edges) | none
    round_parts: str = "auto"    # auto: measure parts round about an axis by their profile | off: always by faces
    use_faces: bool = True       # other parts: golden.find_faces' faces, plus the outline's own straight runs and
                                 # arcs where no face is; False: the outline's runs and arcs only
    thread: str = "auto"         # auto: a thread is also inspected groove by groove (outline_thread) | off
    sample_px: float = 3.0       # outline samples this far apart (px)
    window_px: float = 8.0       # the edge is searched this far either side of the model's outline (px), at the end
    mc_samples: int = 32         # Monte Carlo draws (each of two sets) for the camera / scale / placement uncertainty
    max_poses: int = 8           # resting poses tried
    seed: int = 0


# --------------------------------------------------------------------------- the golden model
def load_golden(path, log: Log = print) -> o3d.geometry.TriangleMesh:
    """A golden model file: STEP / IGES (tessellated finely: diameters come from the outline) or a mesh (mm)."""
    from .io import CAD_EXTS

    path = Path(path)
    if path.suffix.lower() in CAD_EXTS:
        from .cad import load_cad

        return load_cad(path, tolerance=0.002, log=log)
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"{path.name} has no triangles: the golden model must be a mesh or a CAD file")
    return mesh


def _centre_of_mass(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    if abs(vol.sum()) < 1e-9:
        return V.mean(0)
    return ((a + b + c) / 4.0 * vol[:, None]).sum(0) / vol.sum()


def _rotation_to(n: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector n onto target."""
    v = np.cross(n, target)
    s, c = np.linalg.norm(v), float(n @ target)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 - c) / s ** 2


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _solid_angles(tri: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Solid angle of each triangle (n, 3, 3) seen from p (Van Oosterom & Strackee)."""
    a, b, c = tri[:, 0] - p, tri[:, 1] - p, tri[:, 2] - p
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    num = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    den = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", a, c) * lb \
        + np.einsum("ij,ij->i", b, c) * la
    return 2 * np.arctan2(num, den)


def stable_poses(V: np.ndarray, com: np.ndarray) -> list[dict]:
    """Ways the part can rest on a flat table: convex-hull faces with the centre of mass above them. Each: the face's
    outward normal and vertices, the rotation R taking that normal to -z, the chance of landing on it (solid angle
    from the centre of mass / 4 pi) and how far the centre of mass is inside the face (margin, mm); likeliest first."""
    hull = ConvexHull(V)
    eq, S = hull.equations, hull.simplices
    n = eq[:, :3] / np.linalg.norm(eq[:, :3], axis=1, keepdims=True)
    # the centre of mass dropped onto each hull triangle's plane: inside that triangle?
    A, B, C = V[S[:, 0]], V[S[:, 1]], V[S[:, 2]]
    p = com - ((n @ com) + eq[:, 3])[:, None] * n
    v0, v1, v2 = B - A, C - A, p - A
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-300, d00 * d11 - d01 * d01, 1e-300)
    bv = (d11 * d20 - d01 * d21) / den
    bw = (d00 * d21 - d01 * d20) / den
    inside = (bv >= -1e-9) & (bw >= -1e-9) & (1 - bv - bw >= -1e-9)
    key = np.c_[np.round(eq[:, :3], 5), np.round(eq[:, 3], 3)]
    _, group = np.unique(key, axis=0, return_inverse=True)
    group = group.ravel()
    solid = np.bincount(group, weights=_solid_angles(V[S], com))
    out = []
    for g in np.unique(group[inside]):
        idx = np.flatnonzero(group == g)
        nn, d = n[idx[0]], eq[idx[0], 3] / np.linalg.norm(eq[idx[0], :3])
        verts = np.unique(S[idx])
        u, v = _plane_frame(nn)
        P = V[verts]
        q = com - (nn @ com + d) * nn
        try:
            h2 = ConvexHull(np.c_[P @ u, P @ v])
            margin = float(-np.max(h2.equations[:, :2] @ [q @ u, q @ v] + h2.equations[:, 2]))
        except Exception:
            continue
        if margin <= 1e-6 * np.ptp(V, axis=0).max():
            continue
        out.append({"normal": nn, "verts": verts, "R": _rotation_to(nn, np.array([0, 0, -1.0])),
                    "prob": float(solid[g] / (4 * np.pi)), "margin": margin, "facets": len(idx)})
    out.sort(key=lambda r: -r["prob"])
    return out


# --------------------------------------------------------------------------- round (turned) parts
def _meridians(scene, center: np.ndarray, o: np.ndarray, a: np.ndarray, u: np.ndarray, v: np.ndarray,
               t: np.ndarray, angles: np.ndarray, rmax: float) -> np.ndarray:
    """The part's outer radius along meridians (angles about the axis) at axial positions t: rays cast from outside
    towards the axis (0 where nothing is hit)."""
    out = np.zeros((len(angles), len(t)))
    for k, th in enumerate(angles):
        d = math.cos(th) * u + math.sin(th) * v
        P = o + np.outer(t, a) + (rmax + 5.0) * d
        rays = np.concatenate([P - center, np.broadcast_to(-d, P.shape)], 1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        out[k] = np.where(np.isfinite(hit), rmax + 5.0 - hit, 0.0)
    return out


def turned_profile(ref: ReferenceSurface, scene, log: Log = print) -> dict | None:
    """Is the part round about an axis (turned: plain, knurled or threaded), and its profile along that axis? None
    for other parts. The profile: the axis (a, through o), shoulders (flat faces square to the axis that show on the
    outline: the ends, a head's underside), zones of one outer radius (with their texture: a thread - pitch, minor
    radius, hand - or a knurl), and which zone is a head."""
    from scipy.ndimage import maximum_filter, maximum_filter1d

    V, F, N, A = ref.vertices, ref.triangles, ref.normals, ref.areas
    c = V.mean(0)
    w, ev = np.linalg.eigh(np.cov((V - c).T))
    if (w[2] - w[1]) >= (w[1] - w[0]):
        a, pair = ev[:, 2], (w[0], w[1])
    else:
        a, pair = ev[:, 0], (w[1], w[2])
    if abs(pair[1] - pair[0]) > 0.08 * max(pair):         # not a round cross-section
        return None
    for _ in range(3):                                      # CAD end faces are exactly square to the axis
        ax = N @ a
        sel = np.abs(ax) > 0.995
        if A[sel].sum() < 1e-3 * A.sum():
            break
        nrm = (N[sel] * (np.sign(ax[sel]) * A[sel])[:, None]).sum(0)
        a = nrm / np.linalg.norm(nrm)
    if a[np.argmax(np.abs(a))] < 0:
        a = -a
    u, v = _plane_frame(a)
    uv = np.c_[V @ u, V @ v]
    cuv = (uv.min(0) + uv.max(0)) / 2
    o = cuv[0] * u + cuv[1] * v                              # the axis: o + s a
    t = V @ a
    t0, t1 = float(t.min()), float(t.max())
    L = t1 - t0
    rho = np.hypot(uv[:, 0] - cuv[0], uv[:, 1] - cuv[1])
    rmax = float(rho.max())
    step = float(np.clip(L / 6000, 0.004, 0.05))
    tt = np.arange(t0 + step / 2, t1, step)
    angles = np.radians(np.arange(0.0, 360.0, 5.0))
    prof = _meridians(scene, ref.center, o, a, u, v, tt, angles, rmax)
    env = prof.max(0)
    # round: every meridian reaches the envelope within a short stretch (a helix, a knurl) and within +-10 degrees
    wt = max(3, int(round(max(2.0, 0.05 * L) / step)))
    rm = maximum_filter(prof, size=(5, wt), mode=("wrap", "nearest"))
    env_rm = maximum_filter1d(env, wt, mode="nearest")
    tol = 0.02 + 0.005 * env_rm
    solid = env_rm > 0
    frac = float((np.abs(rm - env_rm[None, :]) <= tol[None, :])[:, solid].mean()) if solid.any() else 0.0
    if frac < 0.95:
        return None
    low = np.where(prof > 0, prof, np.inf).min(0)
    low[~np.isfinite(low)] = 0.0

    def env_at(x):
        return float(np.interp(x, tt, env, left=0.0, right=0.0))

    # zones of one outer radius
    zones = []
    i, n = 0, len(tt)
    while i < n:
        if env[i] <= 0:
            i += 1
            continue
        tolz = 0.004 + 0.0005 * env[i]
        lo_, hi_ = env[i], env[i]
        j = i
        while j + 1 < n and max(hi_, env[j + 1]) - min(lo_, env[j + 1]) <= 2 * tolz:
            j += 1
            lo_, hi_ = min(lo_, env[j]), max(hi_, env[j])
        if (j - i + 1) * step >= max(1.5, 0.03 * L):
            zones.append({"t0": float(tt[i] - step / 2), "t1": float(tt[j] + step / 2)})
        i = j + 1
    for z in zones:
        core = (tt >= z["t0"] + 0.1) & (tt <= z["t1"] - 0.1)
        vz = (t >= z["t0"] + 0.05) & (t <= z["t1"] - 0.05)
        z["R"] = float(rho[vz].max()) if vz.any() else float(np.median(env[core]))
        z["tc"] = (z["t0"] + z["t1"]) / 2
        z["depth"] = float(z["R"] - np.median(low[core])) if core.any() else 0.0
        z["texture"], z["pitch"], z["minor"], z["hand"] = None, None, None, None
        z["ripple"] = 0.0
        if z["depth"] <= 0.05:
            continue
        z["texture"] = "knurl"
        # how much the silhouette's edge moves as the part rolls (crest or gap at the edge)
        th = np.radians(np.arange(0.0, 360.0, 0.2))
        ring = _meridians(scene, ref.center, o, a, u, v, np.array([z["tc"]]), th, rmax)[:, 0]
        psi = np.radians(np.arange(0.0, 360.0, 0.5))
        half_w = (ring[None, :] * np.cos(th[None, :] - psi[:, None])).max(1)
        z["ripple"] = float(half_w.max() - half_w.min())
        m0 = prof[0, core]
        x = m0 - m0.mean()
        if len(x) < 40 or x @ x <= 0:
            continue
        spec = np.fft.rfft(x, 2 * len(x))
        ac = np.fft.irfft(spec * np.conj(spec))[:len(x)] / (x @ x)
        lag0 = max(2, int(0.2 / step))
        if lag0 >= len(ac) // 2:
            continue
        k = lag0 + int(np.argmax(ac[lag0:len(ac) // 2]))
        if ac[k] < 0.6:
            continue
        p_est = k * step
        q = int(round(len(angles) / 4))                    # the meridian 90 degrees round: a helix is p / 4 on
        x9 = prof[q, core] - prof[q, core].mean()
        cc = np.fft.irfft(np.fft.rfft(x9, 2 * len(x)) * np.conj(np.fft.rfft(x, 2 * len(x))))
        lags = np.r_[np.arange(len(x)), np.arange(-len(x), 0)] * step
        near = np.abs(lags) <= 0.6 * p_est
        shift = float(lags[near][np.argmax(cc[near])])
        if abs(abs(shift) - p_est / 4) > 0.3 * p_est / 4:
            continue                                        # rings (grooves), not a helix
        # the pitch from the flanks' mid-depth crossings along one meridian
        level = (z["R"] + np.median(low[core])) / 2
        tc_ = tt[core]
        up = np.flatnonzero((m0[:-1] < level) & (m0[1:] >= level))
        cross = tc_[up] + (level - m0[up]) / (m0[up + 1] - m0[up]) * step
        if len(cross) >= 4:
            kk = np.round((cross - cross[0]) / p_est)
            pitch = float(np.polyfit(kk, cross, 1)[0])
        else:
            pitch = p_est
        minor = float(np.median(low[core]))
        # the full thread: where the grooves reach the minor radius all round (not the chamfered start, not the
        # runout under a head)
        zone_t = (tt >= z["t0"]) & (tt <= z["t1"])
        deep = zone_t & (low <= minor + 0.03 * (z["R"] - minor) + 0.005)
        full = (float(tt[deep].min()), float(tt[deep].max())) if deep.any() else (z["t0"], z["t1"])
        # the helix's phase: where a root of the meridian at angle 0 lies (a root at angle th is hand * p * th / 2pi
        # further along)
        down = np.flatnonzero((m0[:-1] >= level) & (m0[1:] < level))
        dcross = tc_[down] + (level - m0[down]) / (m0[down + 1] - m0[down]) * step
        roots = [(dc + cross[cross > dc][0]) / 2 for dc in dcross if np.any(cross > dc)]
        z.update(texture="thread", pitch=pitch, minor=minor, hand="right" if shift > 0 else "left",
                 hand_sign=1.0 if shift > 0 else -1.0, full=full,
                 root_t0=float(roots[len(roots) // 2]) if roots else float(z["tc"]))
    # shoulders: flat faces square to the axis that stand out of the outline
    ax = N @ a
    sel = np.flatnonzero(np.abs(ax) > 0.9999)
    tc = ref.vertices[ref.triangles[sel]].mean(1) @ a
    order = np.argsort(tc)
    sel, tc = sel[order], tc[order]
    shoulders = []
    if len(sel):
        cut = np.flatnonzero(np.diff(tc) > 2e-3) + 1
        for grp_sel, grp_t in zip(np.split(sel, cut), np.split(tc, cut)):
            for sign in (1.0, -1.0):
                g = grp_sel[np.sign(ax[grp_sel]) == sign]
                if len(g) == 0 or A[g].sum() < 0.3:
                    continue
                tk = float(np.average(grp_t[np.sign(ax[grp_sel]) == sign], weights=A[g]))
                P = V[np.unique(F[g])]
                r = np.linalg.norm((P - o) - np.outer((P - o) @ a, a), axis=1)
                inner = env_at(tk - sign * 0.05)
                beyond = env_at(tk + sign * 0.05)
                if r.max() < 0.9 * inner or beyond > r.max() - 0.05:
                    continue                                 # inside the part (a socket's floor) or hidden
                shoulders.append({"t": tk, "sign": sign, "rho": (float(r.min()), float(r.max())),
                                  "vis": (float(max(r.min(), beyond)), float(r.max())), "area": float(A[g].sum()),
                                  "end": bool(abs(tk - (t1 if sign > 0 else t0)) < 1e-3 * max(L, 1.0))})
    # each shoulder's band: from its plane to the next zone on the material side (the fillet or chamfer there): the
    # outline at an end is often that fillet's edge, just short of the plane, and it moves with the end
    for s in shoulders:
        # the zone on the material side next to the shoulder (its rim is that zone's edge)
        side = [k for k, z in enumerate(zones) if (z["t0"] - 0.05 <= s["t"] - s["sign"] * 0.1 <= z["t1"] + 0.05)
                or (s["sign"] > 0 and z["t1"] <= s["t"] + 1e-6 and s["t"] - z["t1"] <= 3.0)
                or (s["sign"] < 0 and z["t0"] >= s["t"] - 1e-6 and z["t0"] - s["t"] <= 3.0)]
        s["zone"] = (min(side, key=lambda k: min(abs(zones[k]["t0"] - s["t"]), abs(zones[k]["t1"] - s["t"])))
                     if side else None)
        if s["sign"] > 0:
            below = [z["t1"] for z in zones if z["t1"] <= s["t"] + 1e-6]
            s["band"] = (max(below) if below else s["t"] - 0.5, s["t"])
        else:
            above = [z["t0"] for z in zones if z["t0"] >= s["t"] - 1e-6]
            s["band"] = (s["t"], min(above) if above else s["t"] + 0.5)
        s["band"] = (min(s["band"]), max(s["band"]))
        if s["band"][1] - s["band"][0] > 3.0:          # no zone close by: a narrow band only
            s["band"] = (s["t"] - 0.5, s["t"]) if s["sign"] > 0 else (s["t"], s["t"] + 0.5)
    # a head: a zone at one end of the part clearly bigger than every other zone
    head = None
    if len(zones) >= 2:
        for k in (0, len(zones) - 1):
            others = [z["R"] for j, z in enumerate(zones) if j != k]
            gap = zones[k]["t0"] - t0 if k == 0 else t1 - zones[k]["t1"]
            if zones[k]["R"] >= 1.15 * max(others) and gap <= 0.15 * L:
                head = k
    prof_info = {"a": a, "o": o, "u": u, "v": v, "t0": t0, "t1": t1, "length": L, "zones": zones,
                 "shoulders": shoulders, "head": head, "roundness": frac}
    log("  round part: " + ", ".join(
        f"Ø{2 * z['R']:.3f} from {z['t0']:.2f} to {z['t1']:.2f}"
        + (f" (thread, pitch {z['pitch']:.4f}, minor Ø{2 * z['minor']:.3f})" if z["texture"] == "thread" else
           " (knurl)" if z["texture"] == "knurl" else "") for z in zones)
        + f"; {len(shoulders)} shoulders")
    return prof_info


def _turned_features(prof: dict) -> list[dict]:
    """The features a round part's outline can measure, with plain labels."""
    zones, shoulders, head = prof["zones"], prof["shoulders"], prof["head"]
    feats: list[dict] = []
    head_side = None
    if head is not None:
        head_side = 1.0 if zones[head]["tc"] > (prof["t0"] + prof["t1"]) / 2 else -1.0
    names = {}
    for k, s in enumerate(shoulders):
        if s["end"] and head_side is not None and s["sign"] == head_side:
            names[k] = "head top"
        elif s["end"]:
            other = zones[0] if s["sign"] < 0 else zones[-1]
            names[k] = ("thread end" if other["texture"] == "thread" else "shank end") if head is not None \
                else f"{'upper' if s['sign'] > 0 else 'lower'} end"
        else:
            names[k] = f"shoulder at {s['t']:.2f}"
    if head is not None:        # the underside: the shoulder facing away from the head next to the head zone
        hz = zones[head]
        cand = [k for k, s in enumerate(shoulders) if not s["end"] and s["sign"] == -head_side
                and (hz["t0"] - 3.0 <= s["t"] <= hz["t0"] + 0.5 if head_side > 0 else
                     hz["t1"] - 0.5 <= s["t"] <= hz["t1"] + 3.0)]
        if cand:
            names[max(cand, key=lambda k: shoulders[k]["area"])] = "head underside"
    for k, s in enumerate(shoulders):
        feats.append({"kind": "shoulder", "label": names[k], **s})
    for zi, z in enumerate(zones):
        if zi == head:
            label = "head" + (" (knurl crests)" if z["texture"] == "knurl" else "")
        elif z["texture"] == "thread":
            label = "thread crests"
        elif head is not None and abs(zi - head) == 1:
            label = "shank"
        else:
            label = f"Ø{2 * z['R']:.2f} round face"
        feats.append({"kind": "crest", "zone": zi, "label": label, **z})
        if z["texture"] == "thread":
            feats.append({"kind": "root", "zone": zi, "label": "thread roots", **z})
            feats.append({"kind": "flank", "zone": zi, "label": "thread flanks", **z})
            feats.append({"kind": "stretch", "zone": zi, "label": "thread pitch", **z})
    return feats


class GoldenPart:
    """The golden mesh prepared for the outline check: outward normals (compare.ReferenceSurface), its measurable
    features (a round part's profile, else golden.find_faces' faces), centre of mass, resting poses, how points move
    when features move, and a ray caster in its own frame."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh, log: Log = print, max_poses: int = 8,
                 round_parts: str = "auto"):
        ref = ReferenceSurface(mesh, log=lambda m: None)
        self.ref = ref
        self.V, self.F = ref.vertices, ref.triangles
        self.N, self.areas = ref.normals, ref.areas
        self.centroids = self.V[self.F].mean(1)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor((self.V - ref.center).astype(np.float32)),
                                 o3d.core.Tensor(self.F.astype(np.uint32)))
        self.profile = turned_profile(ref, self.scene, log) if round_parts != "off" else None
        if self.profile is not None:
            self.faces, self.tri_face = [], np.full(len(self.F), -1, dtype=np.int64)
            self.features = _turned_features(self.profile)
        else:
            total = float(self.areas.sum())
            self.faces, self.tri_face, _ = find_faces(ref, max(5e-5 * total, 1.0))
            labels = _face_labels(self.faces)
            self.features = [{**f, "kind": f["type"], "label": lab} for f, lab in zip(self.faces, labels)]
        self.labels = [f["label"] for f in self.features]
        self.edge_a, self.edge_b, self.edge_v, self.boundary, self.btri = _adjacent_pairs(self.F)
        self.com = _centre_of_mass(self.V, self.F)
        self.hull_idx = ConvexHull(self.V).vertices
        self.hull_pts = self.V[self.hull_idx]
        self.lo, self.hi = self.V.min(0), self.V.max(0)
        self.diagonal = float(np.linalg.norm(self.hi - self.lo))
        self.vertex_feats = self._vertex_features()
        self.hull_B = self.basis(self.hull_pts, [self.vertex_feats[i] for i in self.hull_idx])
        self.coarse = self._decimated(4000)
        self.rests = self._distinct_rests(stable_poses(self.V, self.com), max_poses)
        for r in self.rests:
            r["B"] = self.basis(self.V[r["verts"]], [self.vertex_feats[i] for i in r["verts"]])
        if not ref.watertight:
            log("  WARNING: the golden mesh is not closed; its resting poses and outline may be wrong")
        kinds = [f["kind"] for f in self.features]
        what = (f"round part, {kinds.count('shoulder')} shoulders, {kinds.count('crest')} diameters"
                + (", a thread" if "stretch" in kinds else "") if self.profile is not None else
                f"{kinds.count('plane')} flat and {kinds.count('cylinder')} round faces")
        log(f"Golden model: {len(self.F):,} triangles, {what}, {len(self.rests)} distinct ways to lie on the sheet")

    @property
    def textured(self) -> bool:
        """A thread or knurl whose outline turns with the part's roll about its axis."""
        return self.profile is not None and any(z["texture"] for z in self.profile["zones"])

    def _decimated(self, target: int) -> tuple[np.ndarray, np.ndarray]:
        if len(self.F) <= target:
            return self.V, self.F
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.V), o3d.utility.Vector3iVector(self.F))
        m = m.simplify_quadric_decimation(target)
        return np.asarray(m.vertices), np.asarray(m.triangles)

    # ---- geometry along a round part's axis
    def axial(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t along the axis, radius, radial unit vector) of points (golden frame)."""
        pr = self.profile
        rel = X - pr["o"]
        t = rel @ pr["a"]
        radial = rel - np.outer(t, pr["a"])
        r = np.linalg.norm(radial, axis=1)
        return t, r, radial / np.maximum(r, 1e-12)[:, None]

    def _vertex_features(self) -> list[tuple[int, ...]]:
        """The features each vertex lies on (for the points the part rests on)."""
        if self.profile is None:
            pairs = np.c_[self.F.ravel(), np.repeat(self.tri_face, 3)]
            pairs = np.unique(pairs[pairs[:, 1] >= 0], axis=0)
            out: list[list[int]] = [[] for _ in range(len(self.V))]
            for vtx, f in pairs:
                out[vtx].append(int(f))
            return [tuple(x) for x in out]
        t, r, _ = self.axial(self.V)
        out = [[] for _ in range(len(self.V))]
        for fi, f in enumerate(self.features):
            if f["kind"] == "shoulder":
                sel = (np.abs(t - f["t"]) < 1e-3) & (r >= f["rho"][0] - 1e-3) & (r <= f["rho"][1] + 1e-3)
            elif f["kind"] == "crest":
                sel = (t >= f["t0"]) & (t <= f["t1"]) & (r >= f["R"] - _crest_tol(f))
            else:
                continue
            for vtx in np.flatnonzero(sel):
                out[vtx].append(fi)
        return [tuple(x) for x in out]

    def feature_move(self, fi: int, X: np.ndarray) -> np.ndarray:
        """How points of feature fi move (golden frame) per unit of its shift: outward by 1 mm (a face, a shoulder, a
        zone's radius), or, for a thread's pitch, stretched along the axis about the zone's middle by 1 (strain)."""
        f = self.features[fi]
        kind = f["kind"]
        if kind == "plane":
            return np.broadcast_to(f["normal"], X.shape).copy()
        if kind == "cylinder":
            rel = X - f["center"]
            rad = rel - np.outer(rel @ f["axis"], f["axis"])
            rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
            return -rad if f["hole"] else rad
        a = self.profile["a"]
        if kind == "shoulder":
            return np.broadcast_to(f["sign"] * a, X.shape).copy()
        t, _, rhat = self.axial(X)
        if kind == "stretch":
            return np.outer(t - f["tc"], a)
        return rhat

    def basis(self, X: np.ndarray, inc: list[tuple[int, ...]], extra: np.ndarray | None = None) -> np.ndarray:
        """(n, 3, features): how each point moves when the features it lies on (inc) shift by 1 each: the smallest
        move that keeps it on all of them. extra: one more feature per point (-1: none) whose move is added (a
        thread flank also moves with the pitch)."""
        B = np.zeros((len(X), 3, len(self.features)))
        groups: dict[tuple, list[int]] = {}
        for i, fs in enumerate(inc):
            if fs:
                groups.setdefault(tuple(fs), []).append(i)
        for fs, idx in groups.items():
            idx = np.asarray(idx)
            M = np.stack([self.feature_move(f, X[idx]) for f in fs], axis=1)      # (m, k, 3)
            P = np.linalg.pinv(M)                                                  # (m, 3, k)
            for j, f in enumerate(fs):
                B[idx, :, f] = P[:, :, j]
        if extra is not None:
            for f in np.unique(extra[extra >= 0]):
                idx = np.flatnonzero(extra == f)
                B[idx, :, f] += self.feature_move(int(f), X[idx])
        return B

    def rest_frame(self, rest: dict, p: np.ndarray, d: np.ndarray | None = None, dz: float = 0.0) -> np.ndarray:
        """4x4 golden -> sheet: resting pose `rest` of the part with its features shifted by d (mm; None: the golden
        model), moved by p = (x, y, turn, tilt about x, tilt about y, roll about a round part's own axis) (mm,
        radians): the centre of mass above (x, y), the lowest point on the sheet (z = dz, normally 0). With shifts,
        the resting face is the plane through its shifted corners and the lowest point is found among the shifted
        hull points."""
        x, y, yaw, ax, ay = p[:5]
        roll = p[5] if len(p) > 5 else 0.0
        R0, hull = rest["R"], self.hull_pts
        lying = self.profile is not None and abs(rest["normal"] @ self.profile["a"]) < 0.5
        moved = d is not None and np.any(d)
        if moved:
            hull = hull + self.hull_B @ d
        if moved and not lying:
            P = self.V[rest["verts"]] + rest["B"] @ d
            _, _, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
            n = vt[2] if vt[2] @ rest["normal"] > 0 else -vt[2]
            R0 = _rotation_to(n, np.array([0, 0, -1.0]))
        if roll and self.profile is not None:
            R0 = _axis_rotation(R0 @ self.profile["a"], roll) @ R0
        if lying:
            R0 = self._settle(R0, hull)
        ca, sa = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
        Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
        Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
        R = Rz @ Rx @ Ry @ R0
        z = -float(((hull - self.com) @ R[2]).min())
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array([x, y, z + dz]) - R @ self.com
        return T

    def _settle(self, R: np.ndarray, hull: np.ndarray) -> np.ndarray:
        """A round part lying on its side rocks onto the line of its hull under its centre of mass: which line that is
        changes as it rolls (a thread's last crest lies further along at another roll), and so does its tilt."""
        H = (hull - self.com) @ R.T
        ax = R @ self.profile["a"]
        e = np.array([ax[0], ax[1], 0.0])
        e /= max(np.linalg.norm(e), 1e-12)
        s, z = H @ e, H[:, 2]
        left, right = np.flatnonzero(s < 0), np.flatnonzero(s > 0)
        if len(left) == 0 or len(right) == 0:
            return R
        left = left[np.argsort(z[left])[:200]]
        right = right[np.argsort(z[right])[:200]]
        si, zi = s[left][:, None], z[left][:, None]
        sj, zj = s[right][None, :], z[right][None, :]
        z0 = zi + (zj - zi) * (-si) / (sj - si)            # each chord's height under the centre of mass
        i, j = np.unravel_index(np.argmin(z0), z0.shape)
        slope = float((zj[0, j] - zi[i, 0]) / (sj[0, j] - si[i, 0]))
        return _axis_rotation(np.cross([0.0, 0.0, 1.0], e), math.atan(slope)) @ R

    def _distinct_rests(self, rests: list[dict], keep: int) -> list[dict]:
        """Resting poses that give different silhouettes (a round part rolled about its axis is one pose): the
        top-down silhouettes turned onto their principal axes are compared (and turned half round), with the height
        of the centre of mass."""
        import cv2

        V, F = self.coarse
        span = self.diagonal
        res, N = 480.0 / span, 640

        def raster(xy):
            return _fill_triangles(np.zeros((N, N), np.uint8), xy[F])

        out, sigs = [], []
        if self.profile is not None:     # a round part: lying the same way to its axis is the same pose
            for r in rests:
                cos = abs(float(r["normal"] @ self.profile["a"]))
                key = "lying" if cos < 0.5 else (round(float(r["normal"] @ self.profile["a"]), 2),
                                                 round(float(-((self.hull_pts - self.com) @ r["R"][2]).min()), 1))
                same = next((k for k, s in enumerate(sigs) if s == key), None)
                if same is None:
                    sigs.append(key)
                    out.append(dict(r))
                else:
                    out[same]["prob"] += r["prob"]
            out.sort(key=lambda r: -r["prob"])
            for k, r in enumerate(out):
                r["index"] = k
            return out[:keep]
        for r in rests[:64]:
            Q = (V - self.com) @ r["R"].T
            zc = float(-Q[:, 2].min())               # centre of mass above the table
            xy = Q[:, :2] * res + N / 2
            m = cv2.moments(raster(xy), True)
            if m["m00"] <= 0:
                continue
            c = np.array([m["m10"], m["m01"]]) / m["m00"]
            ang = 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])
            rot = np.array([[math.cos(-ang), -math.sin(-ang)], [math.sin(-ang), math.cos(-ang)]])
            mask = raster((xy - c) @ rot.T + N / 2).astype(bool)
            same = None
            for k, (z0, m0) in enumerate(sigs):
                if abs(z0 - zc) < 0.005 * span and max(_iou(m0, mask), _iou(m0, mask[::-1, ::-1])) > 0.96:
                    same = k
                    break
            if same is None:
                sigs.append((zc, mask))
                out.append(dict(r))
            else:
                out[same]["prob"] += r["prob"]
        out.sort(key=lambda r: -r["prob"])
        for k, r in enumerate(out):
            r["index"] = k
        return out[:keep]

    def cast(self, origin_g: np.ndarray, dirs_g: np.ndarray) -> np.ndarray:
        """Do rays (golden frame) hit the part?"""
        rays = np.concatenate([np.broadcast_to(origin_g - self.ref.center, dirs_g.shape), dirs_g], 1)
        return np.isfinite(self.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy())


def _crest_tol(z: dict) -> float:
    """How far below a zone's outer radius a sample still counts as on its crest (mm)."""
    return 0.01 + 0.05 * z["depth"]


# --------------------------------------------------------------------------- the silhouette on the sheet
def _to_camera(T: np.ndarray, sheet: SheetPose) -> tuple[np.ndarray, np.ndarray]:
    """golden -> camera (Rc, tc) for a part pose T (golden -> sheet)."""
    return sheet.R @ T[:3, :3], sheet.R @ T[:3, 3] + sheet.t


def silhouette_map(photo: Photo, camera: Camera, sheet: SheetPose, ppm: float = MAP_PPM) -> dict:
    """The blank middle (as printed: the drawn one scaled like the print) seen from straight above (ppm px per mm,
    row 0 at the top of the page), and the part's silhouette in it: dark against the paper's local brightness."""
    import cv2
    from scipy.ndimage import map_coordinates

    sx, sy = sheet.scale
    x0, x1, y0, y1 = CLEAR[0] * sx, CLEAR[1] * sx, CLEAR[2] * sy, CLEAR[3] * sy
    xs = x0 + (np.arange(int((x1 - x0) * ppm)) + 0.5) / ppm
    ys = y1 - (np.arange(int((y1 - y0) * ppm)) + 0.5) / ppm
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3)
    uv = camera.project(sheet.to_camera(P))
    M = map_coordinates(photo.linear, [uv[:, 1], uv[:, 0]], order=1, mode="nearest").reshape(X.shape)
    M = M.astype(np.float32)
    k = int(2 * round(22 * ppm) + 1)
    paper = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    paper = cv2.GaussianBlur(paper, (0, 0), 8 * ppm)
    mask = (M < 0.5 * paper).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise ValueError("No part found in the blank middle of the sheet: it must lie between the markers, dark "
                         "against the lit paper (light pad under the sheet)")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    part = (lab == big)
    border = part[0].any() or part[-1].any() or part[:, 0].any() or part[:, -1].any()
    around = cv2.dilate(part.astype(np.uint8), np.ones((int(6 * ppm), int(6 * ppm)), np.uint8)).astype(bool) & ~part
    clipped = float(np.mean(M[around] >= 0.98)) if around.any() else 0.0
    return {"mask": part, "ppm": ppm, "clear": (x0, x1, y0, y1), "touches_border": bool(border),
            "area_mm2": float(part.sum() / ppm ** 2), "paper_level": float(np.median(paper)), "clipped": clipped}


def _shadow(golden: GoldenPart, T: np.ndarray, C: np.ndarray, sil: dict) -> np.ndarray:
    """The part's shadow on the sheet from the camera centre C (sheet frame): its silhouette as the photo sees it,
    in the silhouette map's pixels."""
    V, F = golden.coarse
    X = V @ T[:3, :3].T + T[:3, 3]
    lam = C[2] / np.maximum(C[2] - X[:, 2], 1e-6)
    P = C[:2] + (X[:, :2] - C[:2]) * lam[:, None]
    ppm = sil["ppm"]
    x0, _, _, y1 = sil["clear"]
    col = (P[:, 0] - x0) * ppm - 0.5
    row = (y1 - P[:, 1]) * ppm - 0.5
    return _fill_triangles(np.zeros(sil["mask"].shape, np.uint8), np.stack([col, row], -1)[F]).astype(bool)


def _fill_triangles(mask: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Fill triangles (n, 3, 2 pixel coordinates) into mask, one by one: their union (cv2.fillPoly with many
    polygons fills even-odd, so overlapping front and back faces would cancel)."""
    import cv2

    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    keep = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-9
    for t in np.round(tri[keep] * 16).astype(np.int32):
        cv2.fillConvexPoly(mask, t, 1, lineType=cv2.LINE_8, shift=4)
    return mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 0.0


def _principal(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    r, c = np.nonzero(mask)
    P = np.c_[c, -r].astype(np.float64)
    m = P.mean(0)
    w, v = np.linalg.eigh(np.cov((P - m).T))
    return m, float(math.atan2(v[1, 1], v[0, 1])), float(w[1] / max(w[0], 1e-12))


def coarse_pose(golden: GoldenPart, sil: dict, sheet: SheetPose, log: Log = print) -> tuple[dict, np.ndarray, float]:
    """The resting pose, position and turn whose shadow best overlaps the silhouette (IoU)."""
    C = sheet.centre
    obs = sil["mask"]
    ppm = sil["ppm"]
    x0, _, _, y1 = sil["clear"]
    m_obs, a_obs, elong = _principal(obs)

    def to_mm(m):
        return np.array([x0 + (m[0] + 0.5) / ppm, y1 + (m[1] - 0.5) / ppm])

    c_obs = to_mm(m_obs)
    best = (None, None, -1.0)
    for rest in golden.rests:
        p = np.array([c_obs[0], c_obs[1], 0.0, 0.0, 0.0, 0.0])
        sh = _shadow(golden, golden.rest_frame(rest, p), C, sil)
        if not sh.any():
            continue
        _, a_mod, el_mod = _principal(sh)
        if min(elong, el_mod) > 1.3:
            yaws = [a_obs - a_mod, a_obs - a_mod + math.pi]
        elif min(elong, el_mod) > 1.08:
            yaws = [a_obs - a_mod + k * math.pi / 2 for k in range(4)]
        else:
            yaws = list(np.radians(np.arange(0, 360, 15)))
        for yaw in yaws:
            q = np.array([c_obs[0], c_obs[1], yaw, 0.0, 0.0, 0.0])
            for _ in range(3):   # line the shadow's centroid up with the silhouette's
                sh = _shadow(golden, golden.rest_frame(rest, q), C, sil)
                if not sh.any():
                    break
                q[:2] += c_obs - to_mm(_principal(sh)[0])
            s = _iou(obs, _shadow(golden, golden.rest_frame(rest, q), C, sil))
            if s > best[2]:
                best = (rest, q, s)
    rest, q, score = best
    if rest is None:
        raise ValueError("The golden model's outline could not be matched to the silhouette in the photo")
    steps = np.array([1.0, 1.0, math.radians(2.0)])      # pattern search on (x, y, turn)
    for _ in range(60):
        improved = False
        for k in range(3):
            for sgn in (1, -1):
                t = q.copy()
                t[k] += sgn * steps[k]
                s = _iou(obs, _shadow(golden, golden.rest_frame(rest, t), C, sil))
                if s > score:
                    q, score, improved = t, s, True
        if not improved:
            steps /= 2
            if steps[0] < 0.1 / ppm:
                break
    log(f"  silhouette: {sil['area_mm2']:.0f} mm², best overlap {score:.3f} lying on resting pose {rest['index']}")
    return rest, q, score


# --------------------------------------------------------------------------- the outline
def _snap_cylinder(X: np.ndarray, cyl: dict, C_g: np.ndarray, along_axis: np.ndarray) -> np.ndarray:
    """Samples on a tessellated round face moved onto the exact cylinder: along the axis (a tangent line of the
    outline) to the exact tangent point seen from the camera centre C_g, across it (a rim) radially."""
    a, o, R = cyl["axis"], cyl["center"], cyl["radius"]
    u, v = _plane_frame(a)
    rel = X - o
    t = rel @ a
    phi = np.arctan2(rel @ v, rel @ u)
    c = np.array([(C_g - o) @ u, (C_g - o) @ v])
    rc = float(np.linalg.norm(c))
    if rc > R:
        base, half = math.atan2(c[1], c[0]), math.acos(R / rc)
        cands = np.stack([np.full_like(phi, base + half), np.full_like(phi, base - half)], 1)
        diff = np.abs(np.angle(np.exp(1j * (cands - phi[:, None]))))
        tang = cands[np.arange(len(phi)), np.argmin(diff, 1)]
        phi = np.where(along_axis, tang, phi)
    rho = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v
    return o + t[:, None] * a + R * rho


def outline_samples(golden: GoldenPart, T: np.ndarray, camera: Camera, sheet: SheetPose, step_px: float,
                    half_px: float) -> dict:
    """The model's visible outline at pose T: samples every step_px pixels along the occluding contour, each with
    its 3D point (golden frame), pixel, outward 2D normal and the golden faces its edge lies on. Only samples where
    the model is solid on the inside and empty on the outside over the whole search window (half_px either way, and
    half that along the outline) are kept: no corners, no hidden edges. Samples on round faces (a golden face, or a
    plain zone of a round part) are moved onto the exact cylinder."""
    Rc, tc = _to_camera(T, sheet)
    Cg = -Rc.T @ tc
    front = np.einsum("ij,ij->i", golden.N, Cg - golden.centroids) > 0
    a, b, ev = golden.edge_a, golden.edge_b, golden.edge_v
    sil = front[a] != front[b]
    ta, tb, E = a[sil], b[sil], ev[sil]
    if len(golden.boundary):
        ta = np.r_[ta, golden.btri]
        tb = np.r_[tb, golden.btri]
        E = np.r_[E, golden.boundary]
    A, B = golden.V[E[:, 0]], golden.V[E[:, 1]]
    pA, pB = camera.project(A @ Rc.T + tc), camera.project(B @ Rc.T + tc)
    ln = np.linalg.norm(pB - pA, axis=1)
    n = np.maximum(1, np.ceil(ln / step_px).astype(int))
    eid = np.repeat(np.arange(len(E)), n)
    s = (np.concatenate([np.arange(k) for k in n]) + 0.5) / n[eid]
    X = A[eid] + (B - A)[eid] * s[:, None]
    ta_, tb_ = ta[eid], tb[eid]
    fa, fb = golden.tri_face[ta_], golden.tri_face[tb_]
    edge_dir = (B - A)[eid]
    edge_dir /= np.maximum(np.linalg.norm(edge_dir, axis=1, keepdims=True), 1e-12)
    snapped = np.zeros(len(X), bool)
    for fi, f in enumerate(golden.faces):                 # round golden faces: onto the exact cylinder
        if f["type"] != "cylinder":
            continue
        on = (fa == fi) | (fb == fi)
        if on.any():
            X[on] = _snap_cylinder(X[on], f, Cg, np.abs(edge_dir[on] @ f["axis"]) > 0.95)
            snapped |= on
    if golden.profile is not None:                         # plain zones of a round part: the same
        pr = golden.profile
        t, r, _ = golden.axial(X)
        for z in pr["zones"]:
            if z["texture"] is None:
                on = (t >= z["t0"]) & (t <= z["t1"]) & (np.abs(r - z["R"]) < 0.02)
                if on.any():
                    cyl = {"axis": pr["a"], "center": pr["o"], "radius": z["R"]}
                    X[on] = _snap_cylinder(X[on], cyl, Cg, np.abs(edge_dir[on] @ pr["a"]) > 0.95)
                    snapped |= on
    # the golden face seen most edge-on there
    view = X - Cg
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    ca = np.abs(np.einsum("ij,ij->i", golden.N[ta_], view))
    cb = np.abs(np.einsum("ij,ij->i", golden.N[tb_], view))
    tri = np.where(ca <= cb, ta_, tb_)
    face = golden.tri_face[tri]
    Xc = X @ Rc.T + tc
    x = camera.project(Xc)
    t2 = pB[eid] - pA[eid]
    t2 /= np.maximum(np.linalg.norm(t2, axis=1, keepdims=True), 1e-12)
    n2 = np.c_[t2[:, 1], -t2[:, 0]]
    tf = np.where(front[ta_], ta_, tb_)
    third = camera.project(golden.centroids[tf] @ Rc.T + tc)
    n2[np.einsum("ij,ij->i", third - x, n2) > 0] *= -1
    h = half_px
    # +-0.4 px: on the silhouette's boundary itself (a knurl or thread also has contour edges a pixel inside it)
    offs = [(-0.4, 0), (-1.5, 0), (-h / 2, 0), (-h, 0), (-h / 2, h / 2), (-h / 2, -h / 2),
            (0.4, 0), (1.5, 0), (h / 2, 0), (h, 0), (h / 2, h / 2), (h / 2, -h / 2)]
    keep = np.ones(len(X), bool)
    for on_, ot in offs:
        d = camera.rays(x + on_ * n2 + ot * t2) @ Rc                  # camera -> golden frame
        hit = golden.cast(Cg, d)
        keep &= hit if on_ < 0 else ~hit
    inc = [tuple(sorted({int(q) for q in pair if q >= 0})) for pair in zip(fa[keep], fb[keep])]
    return {"X": X[keep], "x": x[keep], "n2": n2[keep], "t2": t2[keep], "face": face[keep], "inc": inc,
            "depth": Xc[keep, 2], "snapped": snapped[keep], "candidates": int(len(X)), "step_px": step_px}


def _project_samples(golden_X: np.ndarray, T: np.ndarray, camera: Camera, sheet: SheetPose) -> np.ndarray:
    Rc, tc = _to_camera(T, sheet)
    return camera.project(golden_X @ Rc.T + tc)


def _sensitivity(X: np.ndarray, M: np.ndarray, n2: np.ndarray, T: np.ndarray, camera: Camera,
                 sheet: SheetPose, eps: float = 0.01) -> np.ndarray:
    """How many pixels the outline moves outward (along n2) per mm the point moves along M (golden frame)."""
    x0 = _project_samples(X, T, camera, sheet)
    x1 = _project_samples(X + eps * M, T, camera, sheet)
    return np.einsum("ij,ij->i", x1 - x0, n2) / eps


def classify(golden: GoldenPart, S: dict, T: np.ndarray, camera: Camera, sheet: SheetPose) -> dict:
    """Which feature each outline sample measures, how many pixels its outline moves per mm of that feature's shift
    (sens), the lowest share of the full pixels-per-mm it must reach to measure (smin), how the samples move with
    the features (B), and for round parts the samples' axial position and side of the axis."""
    X, n2 = S["X"], S["n2"]
    n = len(X)
    feat = np.full(n, -1, dtype=np.int64)
    sens = np.zeros(n)
    smin = np.full(n, MIN_SENSITIVITY)
    out = {"feat": feat, "sens": sens, "smin": smin}
    if golden.profile is None:
        face = S["face"]
        for fi in np.unique(face[face >= 0]):
            sel = face == fi
            sens[sel] = _sensitivity(X[sel], golden.feature_move(int(fi), X[sel]), n2[sel], T, camera, sheet)
        feat[:] = face
        inc = list(S["inc"])
        segs = [i for i, f in enumerate(golden.features) if f.get("segment")]
        free = np.flatnonzero(feat < 0)
        if segs and len(free):                     # this photo's own outline runs and arcs (outline_segments)
            R = T[:3, :3]
            Xk = X[free]
            sx = _sensitivity(Xk, np.broadcast_to(R.T @ [1.0, 0, 0], Xk.shape), n2[free], T, camera, sheet)
            sy = _sensitivity(Xk, np.broadcast_to(R.T @ [0, 1.0, 0], Xk.shape), n2[free], T, camera, sheet)
            m2 = np.c_[sx, sy] / np.maximum(np.hypot(sx, sy), 1e-12)[:, None]
            Pk = (Xk @ R.T + T[:3, 3])[:, :2]
            for fi in segs:
                f = golden.features[fi]
                open_ = feat[free] < 0
                if f["kind"] == "plane":
                    c2 = (R @ f["point"] + T[:3, 3])[:2]
                    nd = (R @ f["normal"])[:2]
                    nd /= max(np.linalg.norm(nd), 1e-12)
                    a_ = Xk @ f["line"]
                    sel = (open_ & (np.abs((Pk - c2) @ nd) < 0.1) & (a_ >= f["span_lo"] - 1.0)
                           & (a_ <= f["span_hi"] + 1.0) & (m2 @ nd > 0.9))
                else:
                    cc = (R @ f["center"] + T[:3, 3])[:2]
                    rr = np.linalg.norm(Pk - cc, axis=1)
                    radial = (Pk - cc) / np.maximum(rr, 1e-12)[:, None]
                    sel = (open_ & (np.abs(rr - f["radius"]) < 0.1)
                           & ((-1 if f["hole"] else 1) * np.einsum("ij,ij->i", radial, m2) > 0.9))
                k = free[sel]
                feat[k] = fi
                sens[k] = _sensitivity(X[k], golden.feature_move(fi, X[k]), n2[k], T, camera, sheet)
                for i in k:
                    inc[i] = (fi,)
        out["B"] = golden.basis(X, inc)
        return out
    pr = golden.profile
    t, r, rhat = golden.axial(X)
    s_rad = _sensitivity(X, rhat, n2, T, camera, sheet)
    s_ax = _sensitivity(X, np.broadcast_to(pr["a"], X.shape), n2, T, camera, sheet)
    extra = np.full(n, -1, dtype=np.int64)
    also = np.full(n, -1, dtype=np.int64)                  # a shoulder's rim also moves with its zone's radius
    crest_of = {f["zone"]: k for k, f in enumerate(golden.features) if f["kind"] == "crest"}
    along = np.abs(s_rad) >= 1.5 * np.abs(s_ax)             # the outline runs along the axis there
    across = np.abs(s_ax) >= 1.5 * np.abs(s_rad)            # ... across it
    for fi, f in enumerate(golden.features):
        kind = f["kind"]
        if kind == "shoulder":
            # the flat face's rim, or the fillet / chamfer between it and the next zone (they move with it)
            sel = ((t >= f["band"][0] - 0.005) & (t <= f["band"][1] + 0.005) & (r >= f["vis"][0] - 0.05)
                   & across & (feat < 0))
            feat[sel] = fi
            sens[sel] = f["sign"] * s_ax[sel]
            if f.get("zone") is not None and f["zone"] in crest_of:
                also[sel] = crest_of[f["zone"]]
            continue
        if kind in ("flank", "stretch"):
            continue
        zone = (t >= f["t0"] + 0.3) & (t <= f["t1"] - 0.3) & (feat < 0)
        if kind == "crest":
            sel = zone & (r >= f["R"] - _crest_tol(f)) & along
        else:                                              # root
            sel = zone & (r <= f["minor"] + _crest_tol(f)) & along
        feat[sel] = fi
        sens[sel] = s_rad[sel]
    for fi, f in enumerate(golden.features):               # thread flanks: the rest of a thread zone
        if f["kind"] != "flank":
            continue
        stretch = next(k for k, g in enumerate(golden.features) if g["kind"] == "stretch" and g["zone"] == f["zone"])
        sel = ((t >= f["t0"] + 0.3) & (t <= f["t1"] - 0.3) & (feat < 0)
               & (r > f["minor"] + _crest_tol(f)) & (r < f["R"] - _crest_tol(f)))
        feat[sel] = fi
        sens[sel] = s_rad[sel]
        smin[sel] = FLANK_SENSITIVITY
        extra[sel] = stretch
    out["B"] = golden.basis(X, [tuple(sorted({int(f), int(g)} - {-1})) for f, g in zip(feat, also)], extra)
    # the side of the axis each sample is on, in the photo
    P = np.array([pr["o"] + pr["a"] * pr["t0"], pr["o"] + pr["a"] * pr["t1"]])
    q = _project_samples(P, T, camera, sheet)
    dirn = (q[1] - q[0]) / max(np.linalg.norm(q[1] - q[0]), 1e-9)
    rel = S["x"] - q[0]
    out.update(t=t, r=r, side=np.sign(dirn[0] * rel[:, 1] - dirn[1] * rel[:, 0]), extra=extra)
    return out


# --------------------------------------------------------------------------- the fit
@dataclass
class Fit:
    p: np.ndarray            # x, y, turn, tilt about x, tilt about y, roll
    d: np.ndarray            # shift of every feature (mm; strain for a pitch; 0 for features not fitted)
    free: np.ndarray         # features whose shift was fitted
    weights: np.ndarray      # robust weight of each sample
    sigma: float             # robust spread of the residuals (px)
    cov: np.ndarray          # covariance of (the fitted pose values, d[free])
    pose_free: tuple = (0, 1, 2)
    tilt_fitted: bool = False
    extra: dict = field(default_factory=dict)


def fit_part(golden: GoldenPart, rest: dict, p0: np.ndarray, d0: np.ndarray, X: np.ndarray, B: np.ndarray,
             e: np.ndarray, n2: np.ndarray, free: np.ndarray, camera: Camera, sheet: SheetPose, tilt_sigma: float,
             weights: np.ndarray | None = None, irls: int = 4, pose_free=(0, 1, 2), dz: float = 0.0,
             sigma_floor: float = 0.05, groups: np.ndarray | None = None) -> Fit:
    """Robust least squares of the part's pose (the entries of p in pose_free) and the free features' shifts: the
    outline samples X (golden frame; B: how they move with the features) must land on the image edges e (distance
    along n2). Fitted tilts get a prior keeping them near the resting face, the shifts a weak one that settles what
    the outline cannot tell apart. The lowest point touches z = dz. weights given: no reweighting (Monte Carlo).
    sigma_floor: the least robust scale (px; larger while the search window is wide). groups: each sample's
    feature (-1: none): outliers are judged within a feature, so a feature whose shift is still settling is never
    thrown out as a whole."""
    free = np.asarray(free, int)
    pf = np.asarray(pose_free, int)
    npose = len(pf)
    w = np.ones(len(X)) if weights is None else weights.copy()
    reweight = weights is None
    p, d = np.asarray(p0, float).copy(), d0.copy()
    sigma = 1.0
    cov = np.full((npose + len(free),) * 2, np.nan)
    steps = np.r_[POSE_STEPS[pf], np.full(len(free), 1e-4)]
    tilts = [k for k, i in enumerate(pf) if i in (3, 4)]
    # a round part's shoulders all shifting outward along the axis by the same amount is the part sliding along its
    # axis (with a thread: and rolling) - no length changes; pin that to zero, the pose takes it. With one shoulder
    # on the outline (the other end hidden) that shoulder is held: it alone tells where along its axis the part lies
    # (left free, a threaded part could slide a whole pitch with its thread, the shoulder's shift taking it up)
    slide = np.zeros(len(free))
    if golden.profile is not None:
        for k, f in enumerate(free):
            if golden.features[f]["kind"] == "shoulder":
                slide[k] = golden.features[f]["sign"]

    def unpack(th):
        q = p0.copy()
        q[pf] = th[:npose]
        dd = d0.copy()
        dd[free] = th[npose:]
        return q, dd

    def residuals(q, dd):
        T = golden.rest_frame(rest, q, dd, dz)
        return np.einsum("ij,ij->i", e - _project_samples(X + B @ dd, T, camera, sheet), n2)

    for _ in range(irls if reweight else 1):
        sw = np.sqrt(w)
        ps = max(sigma, 0.05)

        def fun(th):
            q, dd = unpack(th)
            return np.concatenate([sw * residuals(q, dd), ps * th[tilts] / tilt_sigma, ps * th[npose:] / D_SIGMA,
                                   [ps * float(slide @ th[npose:]) / 1e-3]])

        th, J, _ = lm(fun, np.r_[p[pf], d[free]], steps, iters=20)
        p, d = unpack(th)
        r = residuals(p, d)
        if reweight:
            rc = r.copy()
            if groups is not None:
                for gid in np.unique(groups[groups >= 0]):
                    sel = groups == gid
                    rc[sel] -= np.median(r[sel])
            core = rc[w > 0] if np.any(w > 0) else rc
            sigma = max(_robust_sigma(core - np.median(core)), sigma_floor)
            c = 4.685 * sigma
            w = np.where(np.abs(rc) < c, (1 - (rc / c) ** 2) ** 2, 0.0)
        try:
            dof = max(1, int((w > 0).sum()) - len(th))
            cov = np.linalg.pinv(J.T @ J) * float(np.sum(w * r * r)) / dof
        except np.linalg.LinAlgError:
            pass
    return Fit(p, d, free, w, sigma, cov, tuple(int(i) for i in pf), bool(set(pf) & {3, 4}))


# --------------------------------------------------------------------------- measuring
def _plan_faces(golden: GoldenPart) -> list[dict]:
    """What a part's golden faces offer to measure (as golden.py): overall sizes between the extreme flat faces,
    thickness / gap between the nearest overlapping opposite faces, steps between faces facing the same way,
    diameters of round faces and holes. Each: kind, name, golden value, feature groups and their weights w (value =
    golden + sum of w times the group's shift), bias (how an equal outward shift of every edge moves the value) and
    the direction it runs along."""
    faces, labels = golden.faces, golden.labels
    lo, hi = golden.lo, golden.hi
    planes = [i for i, f in enumerate(faces) if f["type"] == "plane"]
    plan: list[dict] = []
    extreme = set()
    for k in range(3):
        ends = {1: [], -1: []}
        for i in planes:
            n = faces[i]["normal"]
            if abs(abs(n[k]) - 1) < 1.5e-4:
                sg = 1 if n[k] > 0 else -1
                level = hi[k] if sg > 0 else lo[k]
                if abs(faces[i]["point"][k] - level) < 1e-3 * max(hi[k] - lo[k], 1.0):
                    ends[sg].append(i)
        if ends[1] and ends[-1]:
            extreme.update((i, j) for i in ends[1] for j in ends[-1])
            extreme.update((j, i) for i in ends[1] for j in ends[-1])
            axis = np.eye(3)[k]
            plan.append({"kind": "size", "name": f"Overall size along {AXES[k]}", "golden": float(hi[k] - lo[k]),
                         "groups": [ends[1], ends[-1]], "w": [1.0, 1.0], "dir": ("along", axis)})
    corners = {i: golden.V[np.unique(golden.F[faces[i]["tris"]])] for i in planes}

    def footprint(i, n):
        u, v = _plane_frame(n)
        uv = np.c_[corners[i] @ u, corners[i] @ v]
        return uv.min(0), uv.max(0)

    pairs = {}
    for i in planes:
        best = None
        ni = faces[i]["normal"]
        for j in planes:
            if j == i or (i, j) in extreme or ni @ faces[j]["normal"] > -math.cos(math.radians(1.0)):
                continue
            g = faces[i]["offset"] + faces[j]["offset"]
            if abs(g) < 1e-6:
                continue
            (a0, a1), (b0, b1) = footprint(i, ni), footprint(j, ni)
            if np.any(np.minimum(a1, b1) - np.maximum(a0, b0) <= 0):
                continue
            if best is None or abs(g) < abs(best[1]):
                best = (j, g)
        if best is not None:
            pairs[tuple(sorted((i, best[0])))] = best[1]
    for (i, j), g in sorted(pairs.items(), key=lambda kv: -min(faces[kv[0][0]]["area"], faces[kv[0][1]]["area"])):
        word, sgn = ("Thickness", 1.0) if g > 0 else ("Gap", -1.0)
        plan.append({"kind": word.lower(), "name": f"{word}: {labels[i]} to {labels[j]}", "golden": abs(g),
                     "groups": [[i], [j]], "w": [sgn, sgn], "dir": ("along", faces[i]["normal"])})
    groups: list[list[int]] = []
    for i in planes:
        for g in groups:
            if faces[g[0]]["normal"] @ faces[i]["normal"] > math.cos(math.radians(1.0)):
                g.append(i)
                break
        else:
            groups.append([i])
    for g in groups:
        g.sort(key=lambda i: faces[i]["offset"])
        for i, j in zip(g, g[1:]):
            if abs(faces[j]["offset"] - faces[i]["offset"]) > 0.01:
                plan.append({"kind": "step", "name": f"Step: {labels[j]} to {labels[i]}",
                             "golden": float(faces[j]["offset"] - faces[i]["offset"]), "groups": [[j], [i]],
                             "w": [1.0, -1.0], "dir": ("along", faces[i]["normal"])})
    for i, f in enumerate(faces):
        if f["type"] == "cylinder":
            plan.append({"kind": "diameter", "name": f"Diameter: {labels[i]}", "golden": 2 * f["radius"],
                         "groups": [[i]], "w": [-2.0 if f["hole"] else 2.0], "dir": ("across", f["axis"])})
    return plan


def _plan_turned(golden: GoldenPart) -> list[dict]:
    """What a round part's profile offers to measure, with plain names: overall length (end to end), head height
    and length under head (with a head; otherwise the lengths between shoulders), each zone's diameter (a knurled
    head's crests, a thread's major diameter), a thread's minor diameter and pitch."""
    feats = golden.features
    pr = golden.profile
    a = pr["a"]
    sh = [i for i, f in enumerate(feats) if f["kind"] == "shoulder"]
    plan: list[dict] = []
    by_label = {feats[i]["label"]: i for i in sh}
    ends = [i for i in sh if feats[i]["end"]]
    top = [i for i in ends if feats[i]["sign"] > 0]
    bottom = [i for i in ends if feats[i]["sign"] < 0]
    done = set()
    if top and bottom:
        i, j = top[0], bottom[0]
        plan.append({"kind": "size", "name": "Overall length", "golden": feats[i]["t"] - feats[j]["t"],
                     "groups": [[i], [j]], "w": [1.0, 1.0], "dir": ("along", a)})
        done.add(tuple(sorted((i, j))))
    head_top, under = by_label.get("head top"), by_label.get("head underside")
    other_end = by_label.get("thread end", by_label.get("shank end"))
    if head_top is not None and under is not None:
        plan.append({"kind": "thickness", "name": "Head height",
                     "golden": abs(feats[head_top]["t"] - feats[under]["t"]), "groups": [[head_top], [under]],
                     "w": [1.0, 1.0], "dir": ("along", a)})
        done.add(tuple(sorted((head_top, under))))
    if under is not None and other_end is not None:
        plan.append({"kind": "step", "name": "Length under head",
                     "golden": abs(feats[under]["t"] - feats[other_end]["t"]), "groups": [[other_end], [under]],
                     "w": [1.0, -1.0], "dir": ("along", a)})
        done.add(tuple(sorted((under, other_end))))
    # any other shoulders: the nearest opposite one (a collar's width), the next one facing the same way (a step)
    for i in sh:
        fi = feats[i]
        opp = [j for j in sh if feats[j]["sign"] != fi["sign"] and tuple(sorted((i, j))) not in done
               and min(fi["rho"][1], feats[j]["rho"][1]) > max(fi["rho"][0], feats[j]["rho"][0])]
        if opp:
            j = min(opp, key=lambda j: abs(feats[j]["t"] - fi["t"]))
            key = tuple(sorted((i, j)))
            if key not in done:
                done.add(key)
                thick = (fi["t"] - feats[j]["t"]) * fi["sign"] > 0
                plan.append({"kind": "thickness" if thick else "gap",
                             "name": f"{'Length' if thick else 'Gap'}: {fi['label']} to {feats[j]['label']}",
                             "golden": abs(fi["t"] - feats[j]["t"]), "groups": [[i], [j]],
                             "w": [1.0, 1.0] if thick else [-1.0, -1.0], "dir": ("along", a)})
    for sign in (1.0, -1.0):
        same = sorted((i for i in sh if feats[i]["sign"] == sign), key=lambda i: feats[i]["t"])
        for i, j in zip(same, same[1:]):
            key = tuple(sorted((i, j)))
            if key in done:
                continue
            done.add(key)
            hi_, lo_ = (j, i) if sign > 0 else (i, j)
            plan.append({"kind": "step", "name": f"Step: {feats[hi_]['label']} to {feats[lo_]['label']}",
                         "golden": abs(feats[j]["t"] - feats[i]["t"]), "groups": [[hi_], [lo_]], "w": [1.0, -1.0],
                         "dir": ("along", a)})
    for i, f in enumerate(feats):
        if f["kind"] == "crest":
            label = f["label"]
            name = ("Head diameter" + (" (knurl crests)" if f["texture"] == "knurl" else "")
                    if label.startswith("head") else "Thread major diameter" if f["texture"] == "thread"
                    else "Shank diameter" if label == "shank" else f"Diameter: {label}")
            plan.append({"kind": "diameter", "name": name, "golden": 2 * f["R"], "groups": [[i]], "w": [2.0],
                         "dir": ("across", a), "ripple": f["ripple"] if f["texture"] == "knurl" else 0.0})
        elif f["kind"] == "root":
            plan.append({"kind": "diameter", "name": "Thread minor diameter", "golden": 2 * f["minor"],
                         "groups": [[i]], "w": [2.0], "dir": ("across", a)})
        elif f["kind"] == "stretch":
            plan.append({"kind": "pitch", "name": "Thread pitch", "golden": f["pitch"], "groups": [[i]],
                         "w": [f["pitch"]], "bias": 0.0, "dir": ("along", a)})
    return plan


def _plan_segments(golden: GoldenPart, first: int) -> list[dict]:
    """What the outline's own straight runs and arcs (features first.., see outline_segments) offer: the width
    between opposite parallel runs whose extents overlap, the step between runs facing the same way, arcs'
    diameters."""
    feats = golden.features
    segs = [i for i in range(first, len(feats)) if feats[i]["kind"] == "plane"]
    plan: list[dict] = []
    done = set()
    def span_along(j, line):
        lo_, hi_ = feats[j]["span_lo"], feats[j]["span_hi"]
        return (lo_, hi_) if feats[j]["line"] @ line > 0 else (-hi_, -lo_)

    for i in segs:
        fi = feats[i]
        opp = []
        for j in segs:
            if j == i or fi["normal"] @ feats[j]["normal"] > -math.cos(math.radians(2.0)):
                continue
            (a0, a1), (b0, b1) = span_along(i, fi["line"]), span_along(j, fi["line"])
            if min(a1, b1) > max(a0, b0):          # facing each other across the part
                opp.append(j)
        if not opp:
            continue
        j = min(opp, key=lambda j: abs(fi["offset"] + feats[j]["offset"]))
        key = tuple(sorted((i, j)))
        if key in done:
            continue
        done.add(key)
        width = fi["offset"] + feats[j]["offset"]
        word, sgn = ("Width", 1.0) if width > 0 else ("Gap", -1.0)
        plan.append({"kind": "thickness" if width > 0 else "gap", "name": f"{word}: {fi['label']} to "
                     f"{feats[j]['label']}", "golden": abs(width), "groups": [[i], [j]], "w": [sgn, sgn],
                     "dir": ("along", fi["normal"])})
    for i in segs:
        same = sorted((j for j in segs if feats[i]["normal"] @ feats[j]["normal"] > math.cos(math.radians(2.0))),
                      key=lambda j: feats[j]["offset"])
        for a_, b_ in zip(same, same[1:]):
            key = ("step",) + tuple(sorted((a_, b_)))
            if key in done or abs(feats[b_]["offset"] - feats[a_]["offset"]) < 0.01:
                continue
            done.add(key)
            plan.append({"kind": "step", "name": f"Step: {feats[b_]['label']} to {feats[a_]['label']}",
                         "golden": float(feats[b_]["offset"] - feats[a_]["offset"]), "groups": [[b_], [a_]],
                         "w": [1.0, -1.0], "dir": ("along", feats[a_]["normal"])})
    for i in range(first, len(feats)):
        f = feats[i]
        if f["kind"] == "cylinder":
            plan.append({"kind": "diameter", "name": f"Diameter: {f['label']}", "golden": 2 * f["radius"],
                         "groups": [[i]], "w": [-2.0 if f["hole"] else 2.0], "dir": ("across", f["axis"])})
    return plan


def outline_segments(S: dict, C: dict, T: np.ndarray, camera: Camera, sheet: SheetPose,
                     use: np.ndarray) -> list[dict]:
    """The outline's own straight runs and circular arcs where no golden face is (a drafted or chamfered wall, a
    face golden.find_faces does not recognise): each a feature of its own, like a face seen edge-on. A run moves
    outward along the sheet (its outward direction on the sheet, in the golden frame); an arc of at least 150
    degrees is a round face with a vertical axis. Returns pseudo faces (kind "plane" / "cylinder")."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from .golden import _facing, _fit_circle

    idx = np.flatnonzero(use & (C["feat"] < 0))
    if len(idx) < 15:
        return []
    X, x = S["X"][idx], S["x"][idx]
    R = T[:3, :3]
    ex, ey, ez = R.T @ [1.0, 0, 0], R.T @ [0, 1.0, 0], R.T @ [0, 0, 1.0]
    sx = _sensitivity(X, np.broadcast_to(ex, X.shape), S["n2"][idx], T, camera, sheet)
    sy = _sensitivity(X, np.broadcast_to(ey, X.shape), S["n2"][idx], T, camera, sheet)
    m2 = np.c_[sx, sy] / np.maximum(np.hypot(sx, sy), 1e-12)[:, None]      # outward direction on the sheet
    P = (X @ R.T + T[:3, 3])[:, :2]                                          # positions on the sheet (mm)
    pairs = cKDTree(x).query_pairs(1.7 * S["step_px"], output_type="ndarray")
    if len(pairs):
        ok = np.einsum("ij,ij->i", m2[pairs[:, 0]], m2[pairs[:, 1]]) > 0.9
        pairs = pairs[ok]
    from scipy import sparse

    g = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else
                          (np.zeros(0), (np.zeros(0, int), np.zeros(0, int))), shape=(len(idx), len(idx)))
    _, lab = connected_components(g, directed=False)
    out: list[dict] = []
    for c in np.unique(lab):
        k = np.flatnonzero(lab == c)
        if len(k) < 15:
            continue
        Q = P[k]
        cq = Q.mean(0)
        _, sv, vt = np.linalg.svd(Q - cq, full_matrices=False)
        dirn = vt[0]
        nrm2 = np.array([-dirn[1], dirn[0]])
        if nrm2 @ m2[k].mean(0) < 0:
            nrm2 = -nrm2
        res = (Q - cq) @ nrm2
        along = (Q - cq) @ dirn
        if float(np.sqrt(np.mean(res ** 2))) < 0.01 and np.ptp(along) >= 2.0:
            n3 = nrm2[0] * ex + nrm2[1] * ey
            c3 = X[k].mean(0)
            line = dirn[0] * ex + dirn[1] * ey
            out.append({"kind": "plane", "type": "plane", "normal": n3, "point": c3, "offset": float(n3 @ c3),
                        "line": line, "span_lo": float((X[k] @ line).min()), "span_hi": float((X[k] @ line).max()),
                        "area": float(np.ptp(along)), "tris": np.zeros(0, int),
                        "label": f"outline edge facing {_facing(n3)}", "segment": True})
            continue
        c2, r, rms, _ = _fit_circle(Q)
        ang = np.degrees(np.arctan2(Q[:, 1] - c2[1], Q[:, 0] - c2[0]))
        if rms < 0.01 and r < 200.0 and _arc(np.c_[np.cos(np.radians(ang)), np.sin(np.radians(ang))]) >= 150.0:
            radial = (Q - c2) / np.linalg.norm(Q - c2, axis=1, keepdims=True)
            hole = float(np.mean(np.einsum("ij,ij->i", radial, m2[k]))) < 0
            height = float(np.mean(X[k] @ ez))
            centre = R.T @ (np.array([c2[0], c2[1], 0.0]) - T[:3, 3])
            centre = centre + ez * (height - centre @ ez)
            out.append({"kind": "cylinder", "type": "cylinder", "axis": ez, "center": centre, "radius": float(r),
                        "hole": hole, "arc": 360.0, "length": 0.0, "area": float(r), "tris": np.zeros(0, int),
                        "label": f"outline arc Ø{2 * r:.2f} ({'hole' if hole else 'round'})", "segment": True})
    return out


def _faceless(golden: GoldenPart) -> GoldenPart:
    """A copy of a (not round) golden part without its faces: the outline's own runs and arcs measure it."""
    import copy

    ext = copy.copy(golden)
    ext.faces, ext.features, ext.labels = [], [], []
    ext.tri_face = np.full(len(golden.F), -1, dtype=np.int64)
    ext.vertex_feats = [()] * len(golden.V)
    ext.hull_B = np.zeros(golden.hull_B.shape[:2] + (0,))
    ext.rests = [dict(r, B=np.zeros(r["B"].shape[:2] + (0,))) for r in golden.rests]
    return ext


def _with_segments(golden: GoldenPart, segs: list[dict]) -> GoldenPart:
    """A copy of the golden part that also has this photo's outline runs and arcs as features."""
    import copy

    ext = copy.copy(golden)
    nf = len(golden.features)
    ext.features = golden.features + segs
    ext.labels = golden.labels + [s["label"] for s in segs]
    pad = len(segs)
    ext.hull_B = np.concatenate([golden.hull_B, np.zeros(golden.hull_B.shape[:2] + (pad,))], 2)
    ext.rests = [dict(r, B=np.concatenate([r["B"], np.zeros(r["B"].shape[:2] + (pad,))], 2)) for r in golden.rests]
    ext.segment_first = nf
    return ext


def plan_for(golden: GoldenPart) -> list[dict]:
    return _plan_turned(golden) if golden.profile is not None else _plan_faces(golden)


def _arc(ruv: np.ndarray) -> float:
    a = np.sort(np.degrees(np.arctan2(ruv[:, 1], ruv[:, 0])))
    if len(a) < 3:
        return 0.0
    gaps = np.diff(np.r_[a, a[0] + 360.0])
    return float(360.0 - gaps.max())


def _feature_samples(golden: GoldenPart, S: dict, C: dict, use: np.ndarray) -> dict:
    """Per feature: its measuring samples (count, effective count, span, px per mm) and whether it is seen well
    enough to measure, with the reason when not."""
    out = {}
    feat, sens = C["feat"], C["sens"]
    for fi, f in enumerate(golden.features):
        kind = f["kind"]
        member = C["extra"] == fi if kind == "stretch" else feat == fi
        sel = np.flatnonzero(use & member)
        label = golden.labels[fi]
        entry = {"n": int(len(sel)), "seen": False}
        out[fi] = entry
        if len(sel) == 0:
            entry["reason"] = (f"the {label} {'are' if label.endswith('s') else 'is'} not on the part's outline in "
                               "this photo")
            if kind == "root":
                entry["reason"] += " (the gaps between the thread's teeth are too narrow to see into at this " \
                                   "resolution)"
            continue
        X = S["X"][sel]
        span = float(np.ptp(X, axis=0).max()) if len(sel) > 1 else 0.0
        entry.update(span_mm=span, n_eff=int(max(1, min(len(sel), len(sel) * S["step_px"] / 4.0, N_EFF))),
                     s_mean=float(np.mean(np.abs(sens[sel]))) if kind != "stretch" else 1.0)
        need_n, need_span = {"plane": (12, 2.0), "cylinder": (12, 2.0), "shoulder": (6, 1.0), "crest": (12, 2.0),
                             "root": (8, 2.0), "flank": (20, 2.0)}.get(kind, (20, 0.0))
        if kind == "stretch":
            need_span = 3.0 * f["pitch"]
        if len(sel) < need_n or span < need_span:
            entry["reason"] = f"only {span:.1f} mm of the {label} is on the outline in this photo"
            continue
        if kind == "cylinder":
            u, v = _plane_frame(f["axis"])
            rad = golden.feature_move(fi, X) * (-1 if f["hole"] else 1)
            ruv = np.c_[rad @ u, rad @ v]
            ev, evec = np.linalg.eigh(ruv.T @ ruv / len(ruv))
            proj = ruv @ evec[:, 1]
            both = min((proj > 0.3).mean(), (proj < -0.3).mean())
            if both < 0.15 and _arc(ruv) < 150:
                entry["reason"] = f"only one side of the {label} is on the outline in this photo"
                continue
        if kind in ("crest", "root"):
            side = C["side"][sel]
            if min((side > 0).mean(), (side < 0).mean()) < 0.1:
                entry["reason"] = f"only one side of the {label} is on the outline in this photo"
                continue
        entry["seen"] = True
    return out


def _coefficients(m: dict, feats: dict) -> tuple[np.ndarray | None, str]:
    """The measurement as golden + coef . d (coef over all features), or None and why it cannot be measured."""
    coef = np.zeros(len(feats))
    for grp, w in zip(m["groups"], m["w"]):
        seen = [f for f in grp if feats[f]["seen"]]
        if not seen:
            return None, feats[grp[0]].get("reason", "not on the outline")
        n = np.array([feats[f]["n"] for f in seen], float)
        for f, share in zip(seen, n / n.sum()):
            coef[f] += w * share
    return coef, ""


def _values(plan: list[dict], fit: Fit, feats: dict) -> list[tuple[float | None, float, str]]:
    """(value, statistical standard uncertainty, reason if not measured) for each planned measurement."""
    free = list(fit.free)
    npose = fit.cov.shape[0] - len(free)
    cd = fit.cov[npose:, npose:] if len(free) else np.zeros((0, 0))
    infl = np.array([math.sqrt(feats[f]["n"] / feats[f]["n_eff"]) if feats[f].get("n_eff") else 1.0
                     for f in free]) if free else np.zeros(0)
    cd = cd * np.outer(infl, infl)
    out = []
    for m in plan:
        coef, reason = _coefficients(m, feats)
        if coef is None:
            out.append((None, 0.0, reason))
            continue
        a = coef[free]
        var = float(a @ cd @ a) if len(free) else 0.0
        out.append((float(m["golden"] + coef @ fit.d), math.sqrt(max(var, 0.0)), ""))
    return out


def _direction(m: dict, T: np.ndarray) -> np.ndarray:
    """The direction on the sheet (unit x, y) a measurement runs along, for the print's x / y scale."""
    how, vec = m["dir"]
    d = (T[:3, :3] @ vec)[:2]
    if how == "across":
        d = np.array([-d[1], d[0]])
    return d / max(np.linalg.norm(d), 1e-12)


# --------------------------------------------------------------------------- one photo
def model_edge(golden: GoldenPart, T: np.ndarray, camera: Camera, sheet: SheetPose, x: np.ndarray, n2: np.ndarray,
               half: float, sigma: float, chunk: int = 400, blur_only: np.ndarray | None = None,
               return_found: bool = False, return_profile: bool = False):
    """Where the edge finder would put the model's own edge at each outline sample (px along n2 from the sample): the
    model's silhouette around the sample (ray cast on a fine grid), blurred like the photo (Gaussian, sigma px),
    through the same 50 % estimator. 0 on a long straight edge; inside a narrow crest, a small hole or a tight
    curve, where blur pulls the 50 % line in. The photo's edge is compared with this, not with the bare outline.
    blur_only: samples already moved onto an exact cylinder: only the blur's part (from the tessellated mesh's own
    edge there), so the tessellation's chords do not come back. return_found: also whether the model's own blurred
    edge is there at all (a groove filled in by blur has none). return_profile: also the model's blurred profile
    along each line (share of open paper, 0 .. 1, at the edge finder's offsets -half .. half every 0.25 px)."""
    from scipy.special import ndtr

    from .outline_camera import profile_crossing

    sigma = max(float(sigma), 0.3)
    reach = half + 3.0 * sigma
    u = np.linspace(-3.0 * sigma, 3.0 * sigma, 25)
    gu = np.exp(-0.5 * (u / sigma) ** 2)
    gu /= gu.sum()
    t = np.arange(-half, half + 1e-9, 0.25)              # the edge finder's own sampling
    Rc, tc = _to_camera(T, sheet)
    Cg = -Rc.T @ tc
    t2 = np.c_[-n2[:, 1], n2[:, 0]]
    out = np.zeros(len(x))
    ok = np.zeros(len(x), bool)
    prof = np.zeros((len(x), len(t))) if return_profile else None
    for i0 in range(0, len(x), chunk):
        xs, ns, ts = x[i0:i0 + chunk], n2[i0:i0 + chunk], t2[i0:i0 + chunk]
        m = len(xs)
        base = xs[:, None, :] + u[None, :, None] * ts[:, None, :]            # (m, nu, 2): lines across the edge

        def solid(vv):
            d = camera.rays((base + vv[..., None] * ns[:, None, :]).reshape(-1, 2)) @ Rc
            return golden.cast(Cg, d).reshape(m, len(u))

        lo = np.full((m, len(u)), -reach)
        hi = np.full((m, len(u)), reach)
        in_lo, in_hi = solid(lo), solid(hi)
        for _ in range(22):                                  # each line's edge, to ~1e-5 px
            mid = (lo + hi) / 2
            s = solid(mid)
            lo = np.where(s, mid, lo)
            hi = np.where(s, hi, mid)
        vb = (lo + hi) / 2
        vb = np.where(~in_lo & ~in_hi, -np.inf, np.where(in_lo & in_hi, np.inf, vb))
        # a straight step at vb blurred by a Gaussian: the part covers ndtr((vb - v) / sigma) of the light
        cover = (gu[None, :, None] * ndtr((vb[:, :, None] - t[None, None, :]) / sigma)).sum(1)
        off, found, _, _ = profile_crossing(1.0 - cover, t, half)
        if blur_only is not None:
            mid = vb[:, len(u) // 2]
            bo = blur_only[i0:i0 + chunk] & np.isfinite(mid)
            off = np.where(bo, off - np.where(np.isfinite(mid), mid, 0.0), off)
        out[i0:i0 + chunk] = np.where(found, off, 0.0)
        ok[i0:i0 + chunk] = found
        if prof is not None:
            prof[i0:i0 + chunk] = 1.0 - cover
    if return_profile:
        return out, ok, prof
    return (out, ok) if return_found else out


def _fit_loop(golden: GoldenPart, rest: dict, fit: Fit, photo: Photo, cam: Camera, sheet: SheetPose, D_px: float,
              p: OutlineParams, half: float, last_half: float, pose_free, tilt_sigma: float,
              kinds: tuple | None = None) -> dict:
    """Outline, edges and fit, again and again with the search window shrinking from half to last_half px, until the
    pose settles. kinds: fit only samples of these feature kinds (the rest of the outline is left out; with
    "untextured" among them, a thread's crests too: where they show along the axis depends on the roll)."""
    nf = len(golden.features)
    st: dict = {}
    level = 0
    for it in range(20):
        final = half <= last_half + 1e-9
        T = golden.rest_frame(rest, fit.p, fit.d)
        S = outline_samples(golden, T, cam, sheet, p.sample_px, half)
        if len(S["X"]) < 20:
            raise ValueError("Too little of the model's outline is visible to fit it to the photo")
        C = classify(golden, S, T, cam, sheet)
        B = C["B"]
        x_as = _project_samples(S["X"] + B @ fit.d, T, cam, sheet)       # the outline as measured so far
        r, valid, _ = edge_offsets(photo.linear, x_as, S["n2"], half)
        blur = edge_blur(photo.linear, x_as + r[:, None] * S["n2"], S["n2"], half)
        ok_b = valid & np.isfinite(blur)
        sigma_blur = float(np.median(blur[ok_b])) if np.any(ok_b) else 0.0
        # where blur puts the model's own edge (narrow crests, tight curves): the photo is compared with that
        pred = model_edge(golden, T, cam, sheet, S["x"], S["n2"], half, sigma_blur, blur_only=S["snapped"]) \
            if half <= 1.6 * last_half else np.zeros(len(r))
        e = x_as + (r - pred - D_px)[:, None] * S["n2"]                   # the edge, biases taken off
        full = cam.f / S["depth"]                                         # px per mm across the view there
        kind_ok = C["feat"] >= 0
        if kinds is not None:
            kind_ok &= np.array([golden.features[f]["kind"] in kinds and not (
                "untextured" in kinds and golden.features[f].get("texture") == "thread") if f >= 0 else False
                for f in C["feat"]], bool)
        good = valid & kind_ok & (np.abs(C["sens"]) > C["smin"] * full)
        feats = _feature_samples(golden, S, C, good)
        free = np.array([f for f in feats if feats[f]["seen"]], int)
        fit_use = valid & (kind_ok if (golden.profile is not None or kinds is not None) else True)
        old = fit.p.copy()
        fit = fit_part(golden, rest, fit.p, np.where(np.isin(np.arange(nf), free), fit.d, 0.0),
                       S["X"][fit_use], B[fit_use], e[fit_use], S["n2"][fit_use], free, cam, sheet, tilt_sigma,
                       pose_free=pose_free, sigma_floor=max(0.05, half / 16 if not final else 0.05),
                       groups=C["feat"][fit_use])
        change = np.abs(fit.p - old)
        st = {"S": S, "C": C, "B": B, "e": e, "valid": valid, "good": good, "feats": feats, "fit_use": fit_use,
              "fit": fit, "sigma_blur": sigma_blur, "full": full}
        if final and change[0] < 2e-4 and change[1] < 2e-4 and change[2] < 2e-6 and change[5] < 2e-5:
            break
        # the window shrinks once the pose has settled in it (a far-off start must first be pulled in while the
        # ends are still inside the window)
        moved_px = math.hypot(change[0], change[1]) * float(np.median(full))
        level += 1
        if final or moved_px < 0.2 * half or level >= 4:
            half, level = max(last_half, half * 0.55), 0
    return st


def _roll_scan(golden: GoldenPart, rest: dict, fit: Fit, photo: Photo, cam: Camera, sheet: SheetPose,
               p: OutlineParams, half: float, log: Log) -> Fit:
    """A thread's outline shifts along the axis as the part rolls about it: with the part's slide along its axis
    pinned by its shoulders, try rolls all round and keep the one whose thread edges lie closest to the model's
    (median distance of the thread samples)."""
    best, best_cost = fit.p.copy(), np.inf
    if not any(f["kind"] == "flank" for f in golden.features):
        return fit
    for roll in np.radians(np.arange(0.0, 360.0, 10.0)):
        q = fit.p.copy()
        q[5] = roll
        T = golden.rest_frame(rest, q, fit.d)
        S = outline_samples(golden, T, cam, sheet, p.sample_px, half)
        if len(S["X"]) < 20:
            continue
        C = classify(golden, S, T, cam, sheet)
        use = np.array([f >= 0 and golden.features[f]["kind"] in ("flank", "root") for f in C["feat"]], bool)
        if use.sum() < 10:
            continue
        x_as = _project_samples(S["X"][use] + C["B"][use] @ fit.d, T, cam, sheet)
        r, valid, _ = edge_offsets(photo.linear, x_as, S["n2"][use], half)
        cost = float(np.median(np.where(valid, np.abs(r), half)))
        if cost < best_cost:
            best, best_cost = q, cost
    extra = {}
    if not np.isfinite(best_cost):
        from .outline_thread import VISIBLE_SHARE, geometry_advice, thread_zone, view_angle, visible_depth

        z = thread_zone(golden)
        T = golden.rest_frame(rest, fit.p, fit.d)
        vis = visible_depth(golden, T, cam, sheet, z) if z is not None else 0.0
        if z is not None and vis < VISIBLE_SHARE * (z["R"] - z["minor"]):
            why = geometry_advice(view_angle(golden, T, sheet), vis, z["R"] - z["minor"])
            extra["thread_hidden"] = why
            extra["roll_warning"] = ("the thread's roll about its axis is not known (its pitch, minor diameter and "
                                     "teeth are not measured): " + why)
        else:
            extra["roll_warning"] = ("the thread's flanks were not found on the outline (too few clean outline "
                                     "points on them at this resolution): its roll about the axis is not known")
        log("  WARNING: " + extra["roll_warning"])
    else:
        log(f"  roll about the axis: {math.degrees(best[5]):.0f}° (thread edges within {best_cost:.2f} px)")
    return Fit(best, fit.d.copy(), fit.free, fit.weights, fit.sigma, fit.cov, fit.pose_free, fit.tilt_fitted, extra)


def check_photo(golden: GoldenPart, photo: Photo | str, camera: Camera, params: OutlineParams | None = None,
                log: Log = print) -> dict:
    """The outline check of one photo. Returns a report dict; "_samples" holds the outline samples and their
    deviations from the golden outline (for the overlay)."""
    p = params or OutlineParams()
    photo = load_photo(photo) if isinstance(photo, (str, Path)) else photo
    warnings: list[str] = []
    name = Path(photo.path).name
    log(f"{name}: {photo.size[0]} x {photo.size[1]} px")
    cam = camera
    why = camera.matches(photo.meta)
    if photo.size != camera.size:
        cam = camera.resized(photo.size)
        warnings.append(f"the photo is {photo.size[0]} x {photo.size[1]} px, the camera was calibrated at "
                        f"{camera.size[0]} x {camera.size[1]}: its calibration was scaled (take the check photos "
                        "at the calibration's resolution)")
        why = [w for w in why if "px but" not in w]
    warnings += [f"{w}: is this the calibrated camera?" for w in why]
    bar = p.bar_mm or 100.0
    bar_u = p.bar_u if p.bar_mm else 1.0
    if not p.bar_mm:
        warnings.append("the sheet's 100 mm bar was not measured: its print scale is taken as 100 % ± 1 %, which "
                        "is far too uncertain for a ±0.1 mm check; measure the bar with a caliper (--bar-mm)")
    scale = sheet_scale(bar, p.height_mm or None)
    if abs(scale[0] - 1) > 0.03 or abs(scale[1] - 1) > 0.03:
        warnings.append(f"the sheet was printed at {100 * scale[0]:.1f} % x {100 * scale[1]:.1f} %, not 100 %: "
                        "measured, so the check allows for it, but the blank middle is only "
                        f"{140 * scale[0]:.0f} x {180 * scale[1]:.0f} mm and the markers are smaller")
    tilt_sigma = math.radians(p.tilt_sigma_deg)

    # 1. the sheet
    markers = find_markers(photo, cam)
    sheet = sheet_pose(cam, markers, scale)
    warnings += sheet.warnings
    log(f"  sheet: {len(sheet.ids)} markers, {sheet.rms_px:.3f} px, camera {sheet.centre[2]:.0f} mm above it, "
        f"tilted {sheet.tilt_deg:.1f}°, markers dilated {sheet.dilation_mm * 1000:+.1f} µm "
        f"({sheet.dilation_px:+.3f} px)")
    focal, f_own = None, None
    if sheet.tilt_deg >= 10:
        # a tilted photo of the flat sheet shows its own focal length (perspective): used instead of the
        # calibration's when it is known better than the assumed focus change
        got = focal_from_photo(cam, markers, scale, sheet)
        if got is not None:
            used = got[1] < p.f_extra_pct / 100 / math.sqrt(3) and abs(got[0]) < 0.03
            focal = {"relative_pct": round(100 * got[0], 4), "sigma_pct": round(100 * got[1], 4), "used": used}
            if abs(got[0]) > max(3 * math.hypot(got[1], cam.f_rel_sigma() or 0.0), 0.003):
                warnings.append(f"this photo's markers imply a focal length {100 * got[0]:+.2f} % (±"
                                f"{100 * got[1]:.2f} %) from the calibration's: the focus changed (lock the focus, "
                                "keep the distance)" + ("; this photo's own focal length is used" if used else ""))
            if used:
                f_own = got[1]
                cam = cam.scaled_f(1.0 + got[0])
                sheet = sheet_pose(cam, markers, scale, True, sheet.R, sheet.t)
    D_px = sheet.dilation_px if p.edge_correction == "markers" else 0.0
    if abs(sheet.dilation_px) > 0.25:
        warnings.append(f"the markers' black squares look {sheet.dilation_mm * 1000:+.0f} µm "
                        f"({sheet.dilation_px:+.2f} px) {'bigger' if sheet.dilation_px > 0 else 'smaller'} than "
                        "printed: an over- or under-exposed photo, heavy sharpening or a smeared print. The part's "
                        "edges are corrected by the same amount, but that correction is only as good as the part's "
                        "edge is like the markers' edge: fix the exposure (paper light grey, not white-clipped)")

    # 2-3. silhouette and starting pose
    sil = silhouette_map(photo, cam, sheet)
    if sil["touches_border"]:
        warnings.append("the part's silhouette reaches the edge of the blank middle: it must lie fully between the "
                        "markers")
    if sil["clipped"] > 0.05:
        warnings.append(f"{100 * sil['clipped']:.0f} % of the paper around the part is white-clipped (over-exposed): "
                        "edges shift when the light is clipped; lower the exposure until the paper is light grey")
    rest, p0, iou = coarse_pose(golden, sil, sheet, log)
    ppm_part = cam.f / max(sheet.centre[2] - golden.diagonal / 4, 1.0)
    # the first search window: the silhouette's pose is good to about a map pixel, but a long part's ends can be
    # 1-1.5 mm off along it (seen on a 108 mm screw at 48 MP), and its shoulders must be inside the window
    half = float(np.clip(max(1.5 * ppm_part / MAP_PPM * 1.5, 1.6 * ppm_part), 3 * p.window_px, 60.0))
    win = p.window_px
    pitches = [f["pitch"] for f in golden.features if f["kind"] == "stretch"]
    if pitches:           # the search window must stay within a thread's tooth: a third of the pitch at most
        win = float(np.clip(0.28 * min(pitches) * ppm_part, 4.0, p.window_px))
        if 0.28 * min(pitches) * ppm_part < 4.0:
            warnings.append(f"the thread's pitch is only {min(pitches) * ppm_part:.0f} px in this photo: too fine to "
                            "measure its flanks well; come closer or use the 48 MP setting")
    pose_free = (0, 1, 2) + ((3, 4) if p.fit_tilt else ()) + ((5,) if golden.textured else ())
    thread_hidden = None
    nf = len(golden.features)
    fit = Fit(p0.copy(), np.zeros(nf), np.zeros(0, int), np.zeros(0), 1.0, np.zeros((0, 0)), pose_free)
    if golden.textured:
        # a thread's outline matches at any slide along the axis if the part rolls with it: first pin the slide on
        # the shoulders and crests (roll kept), then find the roll, then fit everything
        st = _fit_loop(golden, rest, fit, photo, cam, sheet, D_px, p, half, 2 * p.window_px, (0, 1, 2),
                       tilt_sigma, kinds=("shoulder", "crest", "cylinder", "plane", "untextured"))
        fit = _roll_scan(golden, rest, st["fit"], photo, cam, sheet, p, win, log)
        if fit.extra.get("roll_warning"):
            warnings.append(fit.extra["roll_warning"])
        thread_hidden = fit.extra.get("thread_hidden")
        half = 1.5 * win

    # 4-6. outline, edges, pose and feature shifts, the search window shrinking as the fit settles
    base = golden
    if golden.profile is None and not p.use_faces:
        golden = _faceless(golden)
        rest = golden.rests[rest["index"]]
        fit = Fit(fit.p, np.zeros(0), np.zeros(0, int), fit.weights, fit.sigma, fit.cov, pose_free)
    st = _fit_loop(golden, rest, fit, photo, cam, sheet, D_px, p, half, win, pose_free, tilt_sigma)
    if golden.profile is None:
        # the outline's own straight runs and arcs where no golden face is: features too, then fit again
        T = golden.rest_frame(rest, st["fit"].p, st["fit"].d)
        segs = outline_segments(st["S"], st["C"], T, cam, sheet, st["valid"])
        if segs:
            golden = _with_segments(golden, segs)
            rest = golden.rests[rest["index"]]
            f0 = st["fit"]
            fit = Fit(f0.p, np.r_[f0.d, np.zeros(len(segs))], f0.free, f0.weights, f0.sigma, f0.cov, pose_free)
            st = _fit_loop(golden, rest, fit, photo, cam, sheet, D_px, p, win, win, pose_free, tilt_sigma)
            log(f"  {len(segs)} more outline runs and arcs where the golden model has no face")
    S, C, B, e, valid, good, feats, fit_use, fit = (st[k] for k in ("S", "C", "B", "e", "valid", "good", "feats",
                                                                   "fit_use", "fit"))
    sigma_blur, full = st["sigma_blur"], st["full"]
    T = golden.rest_frame(rest, fit.p, fit.d)
    w_all = np.zeros(len(valid))
    w_all[fit_use] = fit.weights
    dev = np.einsum("ij,ij->i", e - _project_samples(S["X"], T, cam, sheet), S["n2"]) / np.where(
        np.abs(C["sens"]) > 1e-9, C["sens"], np.nan)                    # edge beyond the golden outline (mm)
    dev[C["feat"] < 0] = np.nan
    log(f"  fit: {len(S['X'])} outline samples ({int(valid.sum())} with an edge, {int(good.sum())} measuring), "
        f"{fit.sigma:.3f} px, turn {math.degrees(fit.p[2]):.2f}°"
        + (f", roll {math.degrees(fit.p[5]):.1f}°" if 5 in pose_free else "")
        + (f", tilts {math.degrees(fit.p[3]):+.3f}° {math.degrees(fit.p[4]):+.3f}°" if fit.tilt_fitted else "")
        + f", {len(fit.free)} features on the outline")

    # 7-8. measurements and their uncertainty
    plan = plan_for(base) if (base.profile is not None or p.use_faces) else []
    if getattr(golden, "segment_first", None) is not None:
        plan += _plan_segments(golden, golden.segment_first)
    vals = _values(plan, fit, feats)
    mc_common, mc_photo = _monte_carlo(golden, rest, fit, S, B, e, fit_use, feats, plan, markers, cam, sheet,
                                       scale, bar, bar_u, p, f_own)
    measurements = []
    for k, (m, (val, u_stat, reason)) in enumerate(zip(plan, vals)):
        entry = {"id": k, "kind": m["kind"], "name": m["name"], "golden": round(m["golden"], 4),
                 "features": [golden.labels[i] for g in m["groups"] for i in g]}
        if val is None:
            measurements.append({**entry, "photo": None, "difference": None, "uncertainty": None,
                                 "status": "not_measured", "reason": "Not measured from this photo: " + reason})
            continue
        bias = float(m.get("bias", np.sum(m["w"])))
        dirn = _direction(m, T)
        inv_b = dirn[0] ** 2 / 100.0 + dirn[1] ** 2 / (120.0 if p.height_mm else 100.0)
        used = [i for g in m["groups"] for i in g if feats[i]["seen"]]
        s_mean = float(np.mean([feats[i]["s_mean"] for i in used]))
        u_g = p.dot_gain_u / math.sqrt(3) * abs((-bias if p.edge_correction == "markers" else 0.0)
                                                + 2 * m["golden"] * inv_b)
        e_px = p.edge_u_px if p.edge_correction == "markers" else math.hypot(p.edge_u_px,
                                                                              sheet.dilation_px / math.sqrt(3))
        u_e = e_px / s_mean * abs(bias)
        u_c = float(mc_common[k]) if np.isfinite(mc_common[k]) else 0.0
        u_p = float(mc_photo[k]) if np.isfinite(mc_photo[k]) else 0.0
        # a knurl's edge in the photo depends on how the part is rolled; the roll comes from the thread, whose start
        # on the real part need not be where the golden model has it: each side within the knurl's ripple
        u_k = math.sqrt(2) * m.get("ripple", 0.0) / math.sqrt(3)
        u = max(K_U * math.sqrt(u_stat ** 2 + u_c ** 2 + u_p ** 2 + u_g ** 2 + u_e ** 2 + u_k ** 2), 0.0005)
        diff = val - m["golden"]
        budget = {"statistical": round(u_stat, 5), "camera_and_scale": round(u_c, 5), "placement": round(u_p, 5),
                  "toner_spread_assumed": round(float(u_g), 5), "edge_assumed": round(float(u_e), 5), "k": K_U}
        if u_k:
            budget["knurl_roll"] = round(u_k, 5)
        measurements.append({**entry, "photo": round(val, 4), "difference": round(diff, 4),
                             "uncertainty": round(u, 4), "status": _judge(diff, u, p.tolerance), "budget": budget,
                             "samples": int(sum(feats[i]["n"] for i in used))})
    if golden.textured and thread_hidden:      # say why the thread's pitch and roots are not seen: the view
        for m in measurements:
            if m["status"] == "not_measured" and {"thread roots", "thread pitch"} & set(m["features"]):
                m["reason"] = "Not measured from this photo: " + thread_hidden
    # 9. a thread, groove by groove
    thread = None
    if p.thread != "off" and golden.profile is not None:
        from .outline_thread import inspect_thread, summary_rows, thread_zone

        if thread_zone(golden) is not None:
            thread = inspect_thread(golden, rest, fit, photo, cam, sheet, D_px, sigma_blur, p,
                                    _length_rel_u(measurements))
            for row in summary_rows(thread, p.tolerance):
                measurements.append({"id": len(measurements), **row})
            s = thread.get("summary", {}) if thread.get("measurable") else {}
            log(f"  thread: " + (f"{s.get('teeth_placed', 0)} teeth placed, {s.get('grooves_measured', 0)} of "
                                 f"{s.get('grooves', 0)} groove depths measured" if thread.get("measurable") else
                                 "not measured groove by groove: " + thread.get("reason", "")))
    report = {
        "photo": name, "path": str(photo.path), "size": list(photo.size), "orientation": photo.orientation,
        "exif": {k: photo.meta.get(k) for k in ("make", "model", "lens", "focal_mm", "focal35_mm")},
        "camera": {"source": cam.source, "f_px": round(cam.f, 3),
                   "f_sigma_pct": None if cam.f_rel_sigma() is None else round(100 * cam.f_rel_sigma(), 4),
                   "focal_check": focal},
        "sheet": sheet.report(),
        "part": {"resting_pose": int(rest["index"]), "iou": round(float(iou), 4), "x_mm": round(float(fit.p[0]), 3),
                 "y_mm": round(float(fit.p[1]), 3),
                 "turn_deg": round(float((math.degrees(fit.p[2]) + 180) % 360 - 180), 3),
                 "roll_deg": round(float(math.degrees(fit.p[5]) % 360), 3) if 5 in pose_free else None,
                 "tilt_deg": [round(math.degrees(fit.p[3]), 4), round(math.degrees(fit.p[4]), 4)],
                 "tilt_fitted": fit.tilt_fitted, "fit_sigma_px": round(float(fit.sigma), 4),
                 "outline_samples": int(len(S["X"])), "with_edge": int(valid.sum()),
                 "measuring": int(good.sum()), "rejected": int((fit.weights <= 0).sum()),
                 "px_per_mm": round(float(np.median(full)), 3), "edge_correction_px": round(float(D_px), 4),
                 "edge_blur_px": round(sigma_blur, 3),
                 "shifts_mm": {golden.labels[f]: round(float(fit.d[f]), 5) for f in fit.free}},
        "features_seen": [golden.labels[f] for f in fit.free], "measurements": measurements, "warnings": warnings,
        "_samples": {"x": S["x"], "e": e, "n2": S["n2"], "d": dev, "good": good & (w_all > 0), "feat": C["feat"]},
        "_pose": {"T": T, "fit": fit, "rest": rest, "sheet": sheet, "camera": cam, "D_px": D_px,
                  "sigma_blur": sigma_blur},
    }
    if thread is not None:
        report["thread"] = {k: v for k, v in thread.items() if k != "sides"}
        report["thread"]["sides"] = [{k: v for k, v in sd.items() if k != "px"} for sd in thread["sides"]]
        report["_thread"] = thread
    return report


def _length_rel_u(measurements: list[dict]) -> float:
    """Relative standard uncertainty of a length along the part from this photo's own budget (camera, scale and
    placement parts of its pitch or overall length): what a thread's lead error over a distance carries."""
    rel = [math.hypot(m["budget"]["camera_and_scale"], m["budget"]["placement"]) / m["golden"]
           for m in measurements if m.get("budget") and m["kind"] in ("pitch", "size") and m["golden"] > 1.0]
    return max(rel + [2e-4]) if rel else 1e-3


def _monte_carlo(golden, rest, fit: Fit, S, B, e, fit_use, feats, plan, markers, cam, sheet, scale, bar, bar_u,
                 p: OutlineParams, f_own: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviations of every planned value when what the photo cannot pin down is redrawn and the fit redone
    on the same image edges. Common to all photos of this camera and print: the intrinsics (the calibration's
    covariance), a focus change since the calibration (assumed limits; or, f_own, the standard uncertainty of the
    focal length this tilted photo measured itself), the caliper readings (their limits). This photo only: noisy
    marker corners, the part's tilt off its resting face (from the fit and its prior), the paper's height under the
    part and the shifts of features that are not on the outline (assumed limits). At most MC_MAX_SAMPLES outline
    samples are refitted each time (these are shifts of the whole chain, not noise)."""
    rng = np.random.default_rng(p.seed)
    n = p.mc_samples
    tilt_sigma = math.radians(p.tilt_sigma_deg)
    idx = np.flatnonzero(np.ones(int(fit_use.sum()), bool))
    if len(idx) > MC_MAX_SAMPLES:
        idx = np.sort(rng.choice(len(idx), MC_MAX_SAMPLES, replace=False))
    X, n2, Bv = S["X"][fit_use][idx], S["n2"][fit_use][idx], B[fit_use][idx]
    wts = fit.weights[idx]
    raw = e[fit_use][idx] + (sheet.dilation_px if p.edge_correction == "markers" else 0.0) * n2  # before the bias
    unseen = np.array([f for f in range(len(golden.features)) if not feats[f]["seen"]
                       and golden.features[f]["kind"] != "stretch"], int)
    lim = p.tolerance if p.unseen_u < 0 else p.unseen_u
    base_free = tuple(i for i in fit.pose_free if i not in (3, 4))

    def run(cam_s, sheet_s, tilt=None, dz=0.0, d_unseen=None):
        p0 = fit.p.copy()
        pf = fit.pose_free
        if tilt is not None:
            p0[3:5] = tilt
            pf = base_free
        d0 = fit.d.copy()
        if d_unseen is not None:
            d0[unseen] = d_unseen
        D = sheet_s.dilation_px if p.edge_correction == "markers" else 0.0
        f = fit_part(golden, rest, p0, d0, X, Bv, raw - D * n2, n2, fit.free, cam_s, sheet_s, tilt_sigma,
                     weights=wts, pose_free=pf, dz=dz)
        return np.array([v if v is not None else np.nan for v, _, _ in _values(plan, f, feats)])

    base = run(cam, sheet)
    common, photo_only = [], []
    theta = cam.params()
    fsig = cam.f_rel_sigma() or 0.0
    for _ in range(n):
        if cam.cov is not None:
            th = rng.multivariate_normal(theta, cam.cov)
        else:
            th = theta.copy()
            th[0] *= 1 + rng.normal(0, fsig)
        if f_own is not None:      # this photo measured its own focal length: that replaces the calibration's
            th[0] = theta[0] * (1 + rng.normal(0, f_own))
        else:
            th[0] *= 1 + rng.uniform(-1, 1) * p.f_extra_pct / 100
        cam_s = cam.with_params(th)
        b = bar + rng.uniform(-bar_u, bar_u)
        hgt = (p.height_mm + rng.uniform(-p.bar_u, p.bar_u)) if p.height_mm else None
        try:
            sh = sheet_pose(cam_s, markers, sheet_scale(b, hgt), True, sheet.R, sheet.t)
            common.append(run(cam_s, sh) - base)
        except (ValueError, np.linalg.LinAlgError):
            continue
    tc = np.full((2, 2), np.nan)
    if fit.tilt_fitted:
        k3 = [k for k, i in enumerate(fit.pose_free) if i in (3, 4)]
        tc = fit.cov[np.ix_(k3, k3)]
    if not np.all(np.isfinite(tc)):
        tc = np.eye(2) * tilt_sigma ** 2        # the allowed tilt off the resting face
    for _ in range(n):
        corner = rng.normal(0, sheet.rms_px, (len(sheet.ids) * 4, 2))
        tilt = rng.multivariate_normal(fit.p[3:5], tc)
        dz = rng.uniform(-1, 1) * p.flatness_mm
        du = rng.uniform(-lim, lim, len(unseen)) if len(unseen) else None
        try:
            sh = sheet_pose(cam, markers, scale, True, sheet.R, sheet.t, corner_noise=corner)
            photo_only.append(run(cam, sh, tilt, dz, du) - base)
        except (ValueError, np.linalg.LinAlgError):
            continue

    def spread(a):
        if not a:
            return np.full(len(plan), np.nan)
        a = np.asarray(a)
        ok = np.isfinite(a)
        n_ok = ok.sum(0)
        return np.where(n_ok > 0, np.sqrt(np.where(ok, a * a, 0.0).sum(0) / np.maximum(n_ok, 1)), np.nan)

    return spread(common), spread(photo_only)


# --------------------------------------------------------------------------- several photos
def combine(reports: list[dict], tol: float) -> list[dict]:
    """One value per measurement from all photos that measured it: the weighted mean (weights from each photo's own
    uncertainty: statistical and placement parts), its uncertainty from those parts (inflated when the photos
    disagree more than they should), plus the parts common to all photos (camera, caliper, focus, toner, edge)."""
    if not reports:
        return []
    out = []
    for k, m0 in enumerate(reports[0]["measurements"]):
        rows = [r["measurements"][k] for r in reports if r["measurements"][k]["status"] != "not_measured"]
        entry = {key: m0[key] for key in ("id", "kind", "name", "golden") if key in m0}
        entry["features"] = m0.get("features", m0.get("faces"))
        if not rows:
            reason = next((r["measurements"][k].get("reason") for r in reports), "Not measured from these photos")
            out.append({**entry, "photo": None, "difference": None, "uncertainty": None, "status": "not_measured",
                        "photos": 0, "reason": reason})
            continue
        if m0.get("combine") == "worst":    # a thread's worst tooth or groove: the worst photo's, not a mean
            rank = {"off": 3, "close": 2, "ok": 1}
            w = max(rows, key=lambda m: (rank.get(m["status"], 0), abs(m["difference"])))
            out.append({**entry, **{key: w[key] for key in ("photo", "difference", "uncertainty", "status")},
                        "where": w.get("where", ""), "photos": len(rows), "combined_as": "worst photo"})
            continue
        v = np.array([m["photo"] for m in rows])
        ind = np.array([math.hypot(m["budget"]["statistical"], m["budget"]["placement"],
                                   m["budget"].get("knurl_roll", 0.0)) for m in rows])
        ind = np.maximum(ind, 1e-5)
        common = np.array([math.sqrt(m["budget"]["camera_and_scale"] ** 2 + m["budget"]["toner_spread_assumed"] ** 2
                                     + m["budget"]["edge_assumed"] ** 2) for m in rows])
        wts = 1 / ind ** 2
        mean = float(np.sum(wts * v) / wts.sum())
        u_ind = float(1 / math.sqrt(wts.sum()))
        birge = 1.0
        if len(v) > 1:
            chi2 = float(np.sum(wts * (v - mean) ** 2)) / (len(v) - 1)
            birge = max(1.0, math.sqrt(chi2))
        u = K_U * math.sqrt((u_ind * birge) ** 2 + float(np.mean(common ** 2)))
        diff = mean - m0["golden"]
        out.append({**entry, "photo": round(mean, 4), "difference": round(diff, 4), "uncertainty": round(u, 4),
                    "status": _judge(diff, u, tol), "photos": len(rows),
                    "spread": round(float(np.ptp(v)), 4) if len(v) > 1 else 0.0,
                    "consistency": round(birge, 3)})
    return out


def _headline(measurements: list[dict], tol: float) -> tuple[str, str]:
    counts = {s: sum(m["status"] == s for m in measurements) for s in ("ok", "off", "close", "not_measured")}
    measured = len(measurements) - counts["not_measured"]
    if counts["off"]:
        return "differs", (f"The part does not match the golden model: {counts['off']} measurement"
                           f"{'s are' if counts['off'] > 1 else ' is'} off by more than ±{tol:g} mm.")
    if counts["close"]:
        return "close", (f"{counts['close']} measurement{'s are' if counts['close'] > 1 else ' is'} too close to the "
                         f"±{tol:g} mm limit to call from the photo: measure {'them' if counts['close'] > 1 else 'it'}"
                         " by hand.")
    if measured == 0:
        return "not_measured", "Nothing on the golden model could be measured from the outline in this photo."
    tail = (f"; {counts['not_measured']} could not be seen on the outline (not measured)"
            if counts["not_measured"] else "")
    return "match", f"Everything measured matches the golden model within ±{tol:g} mm ({measured} values){tail}."


def check_photos(golden: GoldenPart | o3d.geometry.TriangleMesh, photos: list, camera: Camera,
                 params: OutlineParams | None = None, out_dir=None, log: Log = print) -> dict:
    """The outline check of one or more photos of the same part (each lying however it lies). Writes report.json
    and an overlay picture per photo into out_dir when given."""
    p = params or OutlineParams()
    if p.tolerance <= 0:
        raise ValueError("The tolerance must be more than 0")
    if p.edge_correction not in ("markers", "none"):
        raise ValueError("edge_correction must be 'markers' or 'none'")
    if not isinstance(golden, GoldenPart):
        golden = GoldenPart(golden, log, p.max_poses, p.round_parts)
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for ph in photos:
        photo = load_photo(ph) if isinstance(ph, (str, Path)) else ph
        rep = check_photo(golden, photo, camera, p, log)
        if out_dir:
            path = out_dir / (Path(photo.path).stem + "-outline.png")
            overlay(photo, rep, p.tolerance, path)
            rep["overlay"] = str(path)
            if rep.get("_thread", {}).get("measurable"):
                from .outline_thread import thread_chart, thread_overlay

                stem = Path(photo.path).stem
                if thread_overlay(photo, rep["_thread"], out_dir / f"{stem}-thread.png"):
                    rep["thread_overlay"] = str(out_dir / f"{stem}-thread.png")
                if thread_chart(rep["_thread"], out_dir / f"{stem}-thread-chart.png"):
                    rep["thread_chart"] = str(out_dir / f"{stem}-thread-chart.png")
        reports.append(rep)
        for m in rep["measurements"]:
            if m["status"] != "not_measured":
                log(f"    {m['name']}: {m['photo']:.4f} ({m['difference']:+.4f} ± {m['uncertainty']:.4f}) "
                    f"{m['status']}" + (f" ({m['where']})" if m.get("where") else ""))
        if rep.get("thread") is not None:
            from .outline_thread import print_thread

            print_thread(rep["thread"], log, rep["photo"])
        del photo
    combined = combine(reports, p.tolerance)
    verdict, headline = _headline(combined, p.tolerance)
    warnings = sorted({w for r in reports for w in r["warnings"]})
    cam_json = camera.to_json()
    cam_json["covariance"] = None
    prof = None
    if golden.profile is not None:
        pr = golden.profile
        prof = {"axis": np.round(pr["a"], 6).tolist(), "length": round(pr["length"], 4),
                "zones": [{"from": round(z["t0"], 3), "to": round(z["t1"], 3), "diameter": round(2 * z["R"], 4),
                           "texture": z["texture"], "pitch": None if z["pitch"] is None else round(z["pitch"], 5),
                           "minor_diameter": None if z["minor"] is None else round(2 * z["minor"], 4),
                           "hand": z["hand"]} for z in pr["zones"]],
                "shoulders": [{"at": round(f["t"], 4), "label": f["label"]} for f in golden.features
                              if f["kind"] == "shoulder"]}
    result = {
        "kind": "outline_check", "version": 1, "tolerance": p.tolerance, "verdict": verdict, "headline": headline,
        "validated": False,
        "note": ("validated on synthetic photos only (docs/outline-check.md); the terms marked 'assumed' in each "
                 "budget have not been measured on real photos yet"),
        "params": p.to_dict(), "camera": cam_json,
        "golden": {"size": np.round(golden.hi - golden.lo, 4).tolist(), "features": len(golden.features),
                   "round_part": prof, "resting_poses": len(golden.rests)},
        "photos": [{k: v for k, v in r.items() if not k.startswith("_")} for r in reports],
        "combined": combined, "warnings": warnings,
    }
    log(headline)
    if out_dir:
        (out_dir / "report.json").write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    return result


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# --------------------------------------------------------------------------- pictures
def _colour(d: np.ndarray, tol: float) -> np.ndarray:
    """Blue (smaller / less material) - green (as the golden model) - red (bigger), saturating at ±tol."""
    t = np.clip(np.nan_to_num(d) / tol, -1, 1)
    green = np.array([60, 190, 90.0])
    red = np.array([230, 60, 60.0])
    blue = np.array([50, 110, 240.0])
    c = np.where(t[:, None] >= 0, green + (red - green) * t[:, None], green + (blue - green) * (-t)[:, None])
    return c.astype(np.uint8)


def overlay(photo: Photo, report: dict, tol: float, path, max_side: int = 2400) -> Path:
    """The photo around the part with the golden model's outline at the fitted pose, each outline sample coloured
    by how far the photo's edge is outside (red: more material) or inside (blue: less) the golden outline,
    saturating at the tolerance; grey: samples that measure nothing (corners, chamfers, rejected). Upright."""
    from PIL import Image, ImageDraw

    from .outline_camera import upright

    S = report["_samples"]
    x = S["x"]
    lo, hi = x.min(0), x.max(0)
    pad = 0.12 * (hi - lo).max() + 40
    W, H = photo.size
    c0, r0 = int(max(0, lo[0] - pad)), int(max(0, lo[1] - pad))
    c1, r1 = int(min(W, hi[0] + pad)), int(min(H, hi[1] + pad))
    crop = photo.grey8[r0:r1, c0:c1]
    k = min(1.0, max_side / max(crop.shape))
    im = Image.fromarray(crop).convert("RGB")
    if k < 1:
        im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    pts = (x - np.array([c0, r0])) * k
    col = _colour(S["d"], tol)
    rad = max(1.5, 4.0 * k)
    for (u, v), g, c in zip(pts, S["good"], col):
        fill = tuple(int(q) for q in c) if g else (150, 150, 150)
        draw.ellipse([u - rad, v - rad, u + rad, v + rad], fill=fill)
    im = upright(im, photo.orientation)
    out = Image.new("RGB", (im.width, im.height + 46), (250, 250, 248))
    out.paste(im, (0, 0))
    d2 = ImageDraw.Draw(out)
    bar_w = min(360, im.width - 40)
    y0 = im.height + 10
    for i in range(bar_w):
        t = (i / (bar_w - 1)) * 2 - 1
        c = tuple(int(q) for q in _colour(np.array([t * tol]), tol)[0])
        d2.line([(20 + i, y0), (20 + i, y0 + 12)], fill=c)
    d2.text((20, y0 + 16), f"-{tol:g} mm (less material)", fill=(40, 40, 40))
    d2.text((20 + bar_w - 130, y0 + 16), f"+{tol:g} mm (more material)", fill=(40, 40, 40))
    counts = {s: sum(m["status"] == s for m in report["measurements"]) for s in ("ok", "off", "close")}
    d2.text((40 + bar_w, y0), f"{report['photo']}: {counts['ok']} ok, {counts['off']} off, {counts['close']} too "
                              f"close to call", fill=(40, 40, 40))
    path = Path(path)
    out.save(path)
    return path
