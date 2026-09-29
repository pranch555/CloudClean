"""Software-side accuracy bench (investigation 2026-09-24): what does each CloudClean processing step do to the
dimensions of a part whose geometry is known exactly?

Reference parts (tests/synthetic.py): cylinder D20 x 60, box 40 x 30 x 20, two D20 spheres 60 mm apart, a
1"-8 UNC thread (basic profile, 60 mm). Each is "scanned" at 0.15 mm spacing with 0.03 mm Gaussian noise,
1 % stray outliers and a debris cluster, then run through the paths the user has: cleaning presets (with / without
support-plane removal), merging two partial views (automatic and with the exact approved transform), meshing
(Poisson open / watertight / + Taubin smoothing, ball pivoting), the local smoothing ops, two-point measuring with
the snapper, and the thread tools. Every result is measured with robust fits (cylinder / sphere / plane / helix)
and compared with the truth.

    python tools/accuracy/synthetic_bench.py [--out bench.json] [--parts cylinder,box,spheres,thread] [--quick]

Prints one line per measurement (error in micrometres) and writes all records to JSON."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SPACING, NOISE = 0.15, 0.03
UNC_PITCH, UNC_MAJOR = 25.4 / 8, 25.4


def quiet(msg):
    pass


# --------------------------------------------------------------------------- measuring a result against the truth
def measures(part: str, geom) -> dict:
    """Robust measurements of `geom` (cloud or mesh) of a reference part in its true frame (the parts are centred
    on the origin; nothing is moved by the processing except merges, which are measured after alignment)."""
    from cloudclean import accuracy as acc

    pts, nrm = acc.surface_points(geom, 3_000_000)
    out = {}
    if part == "cylinder":
        side = (np.abs(pts[:, 2]) < 26) & (np.hypot(pts[:, 0], pts[:, 1]) > 7)
        if nrm is not None:
            side &= np.abs(nrm[:, 2]) < 0.3
        c = acc.fit_cylinder(pts[side], None if nrm is None else nrm[side], axis_hint=[0, 0, 1])
        out["diameter"] = (c["diameter"], 20.0)
        L = acc.measure_length(geom, "z", {"box": {"min": [-8, -8, -40], "max": [8, 8, 40]}})
        out["length"] = (L["measured"], 60.0)
        rd = acc.robust_dimensions(pts)
        out["robust_extent_length"] = (rd["length"], 60.0)
        out["robust_extent_diameter"] = (0.5 * (rd["width"] + rd["height"]), 20.0)
    elif part == "box":
        for axis, true in (("x", 40.0), ("y", 30.0), ("z", 20.0)):
            out[f"length_{axis}"] = (acc.measure_length(geom, axis)["measured"], true)
        rd = acc.robust_dimensions(pts)
        out["robust_extents"] = ((rd["length"] + rd["width"] + rd["height"]) / 3, 30.0)
    elif part == "spheres":
        s1 = acc.fit_sphere(pts[(pts[:, 0] < 0) & (np.linalg.norm(pts - [-30, 0, 0], axis=1) < 13)])
        s2 = acc.fit_sphere(pts[(pts[:, 0] > 0) & (np.linalg.norm(pts - [30, 0, 0], axis=1) < 13)])
        out["diameter_a"] = (s1["diameter"], 20.0)
        out["diameter_b"] = (s2["diameter"], 20.0)
        out["centre_distance"] = (float(np.linalg.norm(s2["center"] - s1["center"])), 60.0)
    elif part in ("thread", "thread_view"):
        th = acc.fit_thread(pts, nrm)
        out["pitch_ppm"] = ((th["pitch"] / UNC_PITCH - 1) * 1e6, 0.0)
        out["major"] = (th["major_diameter"], UNC_MAJOR)
        out["minor"] = (th["minor_diameter"], UNC_MAJOR - 1.0825 * UNC_PITCH)
        if th["pitch_diameter"] is not None:
            out["pitch_diameter"] = (th["pitch_diameter"], UNC_MAJOR - 0.6495 * UNC_PITCH)
    return out


def app_thread_tool(geom) -> dict:
    """CloudClean's own thread tool (cloudclean.analysis.thread_analysis)."""
    from cloudclean.analysis import thread_analysis

    try:
        r = thread_analysis(geom, log=quiet)
        return {"pitch_ppm": ((r["pitch"] / UNC_PITCH - 1) * 1e6, 0.0), "major": (r["major_diameter"], UNC_MAJOR),
                "minor": (r["minor_diameter"], UNC_MAJOR - 1.0825 * UNC_PITCH)}
    except ValueError as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------- parts and scans
def part_mesh(part: str):
    from tests import synthetic as syn

    if part == "cylinder":
        return syn.cylinder_mesh(20.0, 60.0)
    if part == "box":
        return syn.box_mesh((40.0, 30.0, 20.0))
    if part == "spheres":
        return syn.sphere_pair_mesh(20.0, 60.0)
    return syn.thread_rod_mesh(UNC_MAJOR, UNC_PITCH, 60.0)


def full_scan(part, seed=1, outliers=0.01, debris=True, noise=NOISE):
    from tests import synthetic as syn

    return syn.scan_surface(part_mesh(part), SPACING, noise, seed, outlier_fraction=outliers, debris=debris)


def revo_like_scan(part, seed=1):
    """Dense samples averaged on a 0.15 mm voxel grid: the quasi-regular sampling of a Revo Metro export."""
    from tests import synthetic as syn

    dense = syn.scan_surface(part_mesh(part), SPACING / 3, NOISE * 1.7, seed)
    return dense.voxel_down_sample(SPACING)


# --------------------------------------------------------------------------- experiments
def run_part(part: str, quick: bool = False) -> list[dict]:
    import open3d as o3d

    from cloudclean import accuracy as acc
    from cloudclean.clean import CleanParams, clean_point_cloud
    from cloudclean.edit import Snapper, apply_edits
    from cloudclean.mesh import MeshParams, reconstruct_mesh
    from cloudclean.register import MergeParams, merge_geometries
    from tests import synthetic as syn

    records: list[dict] = []
    t_start = time.time()

    def record(step, geom=None, values=None, note=""):
        vals = values if values is not None else measures(part, geom)
        if "error" in vals:
            records.append({"part": part, "step": step, "measure": "error", "note": vals["error"]})
            print(f"{part:9s} {step:42s} ERROR {vals['error'][:90]}", flush=True)
            return
        for k, (value, truth) in vals.items():
            err = value - truth if not k.endswith("_ppm") else value
            records.append({"part": part, "step": step, "measure": k, "value": value, "truth": truth,
                            "error_um": err * 1000 if not k.endswith("_ppm") else None,
                            "error_ppm": value if k.endswith("_ppm") else err / truth * 1e6, "note": note})
        line = "  ".join(f"{k} {((v - t) * 1000 if not k.endswith('_ppm') else v):+8.1f}{'ppm' if k.endswith('_ppm') else 'um'}"
                         for k, (v, t) in vals.items())
        print(f"{part:9s} {step:42s} {line}", flush=True)

    raw = full_scan(part)
    clean_cloud = {}
    record("raw scan (outliers included)", raw)
    for preset in ("light", "standard", "aggressive"):
        c, rep = clean_point_cloud(raw, CleanParams.preset(preset), quiet)
        clean_cloud[preset] = c
        drift = acc.operation_drift(raw, c, "clean", max_points=60_000)
        record(f"clean {preset}", c, note=f"subset={drift['identical_fraction']:.4f} removed={rep['removed_fraction']:.3%}")
    base = clean_cloud["standard"]

    if part in ("box", "cylinder"):
        # part lying on a table, scanned from above: no bottom face, a table plane touching the part
        mesh = part_mesh(part)
        if part == "cylinder":
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([math.pi / 2, 0, 0])
            mesh.rotate(R, center=(0, 0, 0))
        zmin = mesh.get_min_bound()[2]
        on_table = syn.scan_surface(mesh, SPACING, NOISE, 3, view=[0, 0, 1], view_limit=-0.05, table_z=zmin,
                                    outlier_fraction=0.005)
        for remove_plane in (False, True):
            p = CleanParams.preset("standard")
            p.remove_plane = remove_plane
            c, _ = clean_point_cloud(on_table, p, quiet)
            pts = np.asarray(c.points)
            top = float(np.quantile(pts[:, 2], 0.9995))
            if not remove_plane:
                # the table is still in the cloud: height of the part above a plane fitted to the table
                table = pts[(np.abs(pts[:, 2] - zmin) < 0.2) & (np.abs(pts[:, 0]) > 25)]
                f = acc.fit_plane(table)
                vals = {"height_above_table": (top - float(f["point"][2]), 20.0)}
                note = "part on a table, bottom hidden; table kept, height measured from a plane fitted to it"
            else:
                # the table is gone: what the part dimensions (trimmed extents) show for the height
                vals = {"height_extent_after_plane_removal": (top - float(np.quantile(pts[:, 2], 0.0005)), 20.0)}
                note = "part on a table, bottom hidden; support plane removed (points within 3x spacing of it go too)"
            if part == "cylinder":
                sel = (np.abs(pts[:, 1]) < 26) & (pts[:, 2] > zmin + 0.2)
                vals["diameter_fit"] = (acc.fit_cylinder(pts[sel], axis_hint=[0, 1, 0])["diameter"], 20.0)
            record(f"on table: clean standard{' + remove_plane' if remove_plane else ''}", values=vals, note=note)

    # merge two partial views with a known rigid transform
    mesh = part_mesh(part)
    if part == "thread":      # like the real bolt scans: each view sees the crests, the roots and ONE flank
        view_a, view_b, limit = [0, 0, 1.0], [0, 0, -1.0], -0.2
    elif part == "cylinder":  # two sides, both end caps in both views
        view_a, view_b, limit = [1.0, 0.2, 0.0], [-1.0, -0.2, 0.0], -0.2
    else:
        view_a, view_b, limit = [0.3, 0.2, 1.0], [-0.3, -0.2, -1.0], -0.25
    a = syn.scan_surface(mesh, SPACING, NOISE, 11, view=view_a, view_limit=limit)
    b = syn.scan_surface(mesh, SPACING, NOISE, 12, view=view_b, view_limit=limit)
    if part == "thread":
        record("single view (one flank)", a)
    T = syn.random_rigid(7)
    T_inv = np.linalg.inv(T)
    b_moved = o3d.geometry.PointCloud(b).transform(T)
    merged_ok = None
    for mode in ("auto", "approved"):
        merged, transforms, rep = merge_geometries(
            [a, b_moved], MergeParams(), log=quiet, transforms=[np.eye(4), T_inv] if mode == "approved" else None)
        dT = transforms[1] @ T
        rot = acc.rotation_angle_deg(dT[:3, :3])
        shift = float(np.linalg.norm(dT[:3, 3]))
        amb = any(s.get("ambiguous") for s in rep["scans"])
        record(f"merge 2 views ({mode})", merged,
               note=f"pose error {rot:.4f} deg, {shift * 1000:.1f} um{'; flagged ambiguous' if amb else ''}")
        if mode == "approved":
            merged_ok = merged
    if part == "thread":
        # the relative axial position of the two views decides the pitch diameter (each flank comes from one view)
        shifted = o3d.geometry.PointCloud(b).translate([0, 0, 0.05]).transform(T)
        m_shift, _, _ = merge_geometries([a, shifted], MergeParams(), log=quiet, transforms=[np.eye(4), T_inv])
        record("merge, view B placed 0.05 mm off along the axis", m_shift)
    # a scanner that measures one view 0.05 mm "thicker" (as on the real scans' head): a doubled skin
    nb = np.asarray(b.normals)
    b_thick = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(b.points) + 0.05 * nb))
    b_thick.normals = b.normals
    merged_thick, _, _ = merge_geometries([a, b_thick.transform(T)], MergeParams(), log=quiet,
                                          transforms=[np.eye(4), T_inv])
    record("merge, view B 0.05 mm thicker (doubled skin)", merged_thick)
    if not quick:
        mesh_thick, _ = reconstruct_mesh(merged_thick, MeshParams(), quiet)
        record("  -> poisson mesh of the doubled skin", mesh_thick)

    # meshing the cleaned single scan
    meshes = {}
    variants = [("poisson (trim)", dict()), ("poisson watertight", dict(watertight=True)),
                ("poisson + taubin 5", dict(smooth_iterations=5)), ("poisson + taubin 20", dict(smooth_iterations=20)),
                ("bpa", dict(method="bpa"))]
    if quick:
        variants = variants[:2]
    for name, params in variants:
        try:
            m, rep = reconstruct_mesh(base, MeshParams.from_dict(params), quiet)
        except Exception as exc:  # noqa: BLE001 - report and continue
            record(f"mesh {name}", values={"error": str(exc)})
            continue
        meshes[name] = m
        record(f"mesh {name}", m, note=f"deviation mean {rep['deviation']['mean'] * 1000:.1f} um")
    m, _ = reconstruct_mesh(merged_ok, MeshParams(), quiet)
    record("mesh poisson of the approved merge", m)

    # local smoothing ops on the cleaned cloud and on the Poisson mesh
    cloud_ops = [("denoise", {"op": "denoise"}), ("smooth (MLS quadratic)", {"op": "smooth"}),
                 ("smooth_points (plane MLS)", {"op": "smooth_points"}), ("remove_spikes", {"op": "remove_spikes"})]
    mesh_ops = [("smooth taubin x5", {"op": "smooth", "iterations": 5}),
                ("smooth taubin x20", {"op": "smooth", "iterations": 20}),
                ("smooth laplacian x5", {"op": "smooth", "method": "laplacian", "iterations": 5}),
                ("denoise x5", {"op": "denoise"}), ("remove_spikes", {"op": "remove_spikes"})]
    if quick:
        cloud_ops, mesh_ops = cloud_ops[:2], mesh_ops[:2]
    for name, op in cloud_ops:
        g, rep = apply_edits(base, [op], quiet)
        moved = rep["ops"][0].get("moved", {})
        record(f"cloud {name}", g, note=f"moved mean {moved.get('mean', 0) * 1000:.1f} um")
    if "poisson (trim)" in meshes:
        for name, op in mesh_ops:
            g, rep = apply_edits(meshes["poisson (trim)"], [op], quiet)
            moved = rep["ops"][0].get("moved", {})
            record(f"mesh {name}", g, note=f"moved mean {moved.get('mean', 0) * 1000:.1f} um")

    # two-point measuring with the snapper (what a user clicking two points gets)
    if part in ("cylinder", "box"):
        rng = np.random.default_rng(5)
        targets = {"cloud": base}
        if "poisson (trim)" in meshes:
            targets["mesh"] = meshes["poisson (trim)"]
        for tname, geom in targets.items():
            snap = Snapper(geom)
            errs = []
            for _ in range(400):
                if part == "cylinder":
                    th, z = rng.uniform(0, 2 * math.pi), rng.uniform(-20, 20)
                    p1 = [10 * math.cos(th), 10 * math.sin(th), z]
                    p2 = [-10 * math.cos(th), -10 * math.sin(th), z]
                    truth = 20.0
                else:
                    y, z = rng.uniform(-12, 12), rng.uniform(-7, 7)
                    p1, p2, truth = [-20, y, z], [20, y, z], 40.0
                jitter = rng.normal(scale=0.2, size=(2, 3))   # the click lands near, not on, the true point
                snapped, _ = snap.snap(np.array([p1, p2]) + jitter)
                errs.append(float(np.linalg.norm(snapped[1] - snapped[0])) - truth)
            errs = np.array(errs)
            records.append({"part": part, "step": f"two-point measure on the {tname}", "measure": "distance",
                            "error_um": float(errs.mean() * 1000), "spread_um": float(errs.std() * 1000),
                            "worst_um": float(np.abs(errs).max() * 1000), "note": "400 random opposite-point pairs"})
            print(f"{part:9s} {'two-point measure on the ' + tname:42s} mean {errs.mean() * 1000:+8.1f}um  std "
                  f"{errs.std() * 1000:6.1f}um  worst {np.abs(errs).max() * 1000:6.1f}um", flush=True)

    if part == "thread":
        # the app's own thread tool on random, Revo-like (voxel grid) and one-sided scans
        record("app thread tool: random sampling", values=app_thread_tool(base))
        revo = revo_like_scan("thread")
        record("app thread tool: Revo-like voxel sampling", values=app_thread_tool(revo))
        record("helix fit (accuracy.fit_thread): Revo-like sampling", revo)
        one_side = syn.scan_surface(part_mesh("thread"), SPACING, NOISE, 21, view=[0, 0, 1], view_limit=-0.2)
        record("app thread tool: one flank visible", values=app_thread_tool(one_side))
        record("helix fit: one flank visible", one_side)
    print(f"{part}: done in {time.time() - t_start:.0f}s", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("synthetic_bench.json"))
    ap.add_argument("--parts", default="cylinder,box,spheres,thread")
    ap.add_argument("--quick", action="store_true", help="fewer variants (for a smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    parts = [p for p in args.parts.split(",") if p]
    records = []
    if args.workers > 1 and len(parts) > 1:
        with ProcessPoolExecutor(min(args.workers, len(parts))) as pool:
            for rec in pool.map(run_part, parts, [args.quick] * len(parts)):
                records += rec
    else:
        for p in parts:
            records += run_part(p, args.quick)
    args.out.write_text(json.dumps(records, indent=1, default=float))
    print(f"Saved {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
