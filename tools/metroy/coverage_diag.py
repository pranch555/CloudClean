"""Why do left tracks go unpaired? Per unassigned track: length, best/runner-up support, predicted depth."""
import sys
import numpy as np
import stripes as S
from cloudclean.capture.metroy.laserfile import monomials

raw, calib = sys.argv[1], sys.argv[2]
tri = S.Triangulator.from_yaml(calib, extra=S.default_extra(calib))
frame = S.load_raw(raw)
rl, rr = tri.rectify(frame)
cl = S.smooth_tracks(S.stripe_centres(rl, tri.p), tri.p)
cr = S.smooth_tracks(S.stripe_centres(rr, tri.p), tri.p)
tri.triangulate_by_lines(cl, cr)
fam, bias = tri.last["family"], tri.last["bias_px"]
ls = tri.line_sets[fam]
rows = cr.by_row()
pred = monomials(cl.x, cl.y) @ ls.forward.T + bias
reasons = {}
for t in np.unique(cl.track):
    idx = np.flatnonzero(cl.track == t)
    if len(idx) < tri.p.min_track:
        reasons["short"] = reasons.get("short", 0) + len(idx); continue
    sup = np.zeros(ls.lines)
    for i in idx:
        row = rows.get(int(cl.y[i]))
        if row is None: continue
        d = np.min(np.abs(np.asarray(row)[:, None] - pred[i][None, :]), 0)
        sup += d < tri.p.line_tol
    o = np.argsort(sup)[::-1]
    f1, f2 = sup[o[0]] / len(idx), sup[o[1]] / len(idx)
    k = o[0]
    xr = pred[idx, k]
    Z = tri.f * tri.B / (cl.x[idx] - xr - tri.off)
    ok = f1 >= tri.p.line_min_support and f1 - f2 >= tri.p.line_margin
    key = "assigned" if ok else ("no right partner (<10%)" if f1 < 0.1 else ("partial right support" if f1 < tri.p.line_min_support else "ambiguous runner-up"))
    reasons[key] = reasons.get(key, 0) + len(idx)
    if not ok and len(idx) > 60:
        print(f"track {t:4d} n {len(idx):4d} x~{np.median(cl.x[idx]):6.0f} y {cl.y[idx].min():.0f}-{cl.y[idx].max():.0f}: best line {k} {f1:.0%}, runner {o[1]} {f2:.0%}, Z~{np.median(Z):.0f}")
tot = sum(reasons.values())
print({k: f"{v} ({v / tot:.0%})" for k, v in reasons.items()}, "family", fam, "bias", round(bias, 2))
