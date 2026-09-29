"""Before / after numbers on the real bolt for the merge check layering and the thread tool (fixes 3 and 4).

    python tools/accuracy/fixes_before_after.py <assets dir> <tag> <out.json>
"""
import sys, json, time, numpy as np, open3d as o3d
from pathlib import Path
sys.path.insert(0, "tools/accuracy")
from cloudclean.io import load
from cloudclean.register import assess_merge
from cloudclean.analysis import thread_analysis
import real_bolt as rb
D = Path(sys.argv[1]); tag = sys.argv[2]; out = {}
A = load(D / rb.SCAN_A / "data.ply"); B = load(D / rb.SCAN_B / "data.ply")
t = time.time()
res = assess_merge([A, B], log=lambda m: None)
s = res["scans"][1]
out["assess"] = {k: s.get(k) for k in ("layering_mm", "layering_p90_mm", "separation_ratio", "doubled_surface", "ambiguous", "fitness", "overlap", "combined_noise_mm", "recommendation")}
out["assess"]["reasons"] = res["reasons"]; out["assess"]["seconds"] = time.time() - t
print(tag, "assess", json.dumps(out["assess"], indent=1), flush=True)
for aid in (rb.SCAN_A, rb.SCAN_B, rb.MERGE):
    g = load(D / aid / "data.ply"); pts, nrm = np.asarray(g.points), np.asarray(g.normals)
    fr = rb.bolt_frame(pts, nrm); m = fr["thread_mask"]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[m])); pc.normals = o3d.utility.Vector3dVector(nrm[m])
    try:
        r = thread_analysis(pc, log=lambda msg: None)
        out[aid] = {k: r.get(k) for k in ("pitch", "pitch_se", "major_diameter", "minor_diameter", "pitch_diameter", "flank_angle_deg", "crest_count", "handedness", "confidence", "warnings", "axis_uncertainty_deg")}
    except ValueError as exc:
        out[aid] = {"error": str(exc)}
    print(tag, aid, json.dumps(out[aid]), flush=True)
json.dump(out, open(Path(sys.argv[3]), "w"), indent=1, default=float)
