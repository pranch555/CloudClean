"""API routes for scan-vs-CAD inspection: submit a comparison and render a printable report."""
from __future__ import annotations

import html
import math
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..compare import CompareParams


class CompareReq(BaseModel):
    scan_id: str
    reference_id: str
    params: dict = {}
    name: str | None = None


def _num(value, digits: int = 4, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _histogram_svg(hist: dict, tolerance: float) -> str:
    edges, counts = hist["edges"], hist["counts"]
    if not counts:
        return ""
    W, H, left, right, top, bottom = 720, 260, 56, 16, 14, 42
    pw, ph = W - left - right, H - top - bottom
    lo, hi = edges[0], edges[-1]
    peak = max(max(counts), 1)
    x = lambda v: left + (v - lo) / (hi - lo) * pw  # noqa: E731
    y = lambda c: top + ph - c / peak * ph  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Deviation histogram" '
             f'xmlns="http://www.w3.org/2000/svg" font-family="inherit" font-size="11">']
    band0, band1 = max(-tolerance, lo), min(tolerance, hi)
    parts.append(f'<rect x="{x(band0):.1f}" y="{top}" width="{x(band1) - x(band0):.1f}" height="{ph}" '
                 f'fill="#16a34a" fill-opacity="0.10"/>')
    for t in (-tolerance, tolerance):
        if lo <= t <= hi:
            parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{top + ph}" stroke="#16a34a" '
                         f'stroke-dasharray="4 3"/>')
    for i, c in enumerate(counts):
        a, b = edges[i], edges[i + 1]
        mid = (a + b) / 2
        color = "#2563eb" if abs(mid) <= tolerance else ("#dc2626" if mid > 0 else "#d97706")
        parts.append(f'<rect x="{x(a) + 0.5:.1f}" y="{y(c):.1f}" width="{max(x(b) - x(a) - 1, 0.5):.1f}" '
                     f'height="{top + ph - y(c):.1f}" fill="{color}"><title>{a:+.4f} … {b:+.4f}: {c}</title></rect>')
    parts.append(f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" stroke="#475569"/>')
    for k in range(-4, 5):
        v = k * (hi - lo) / 8
        parts.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top + ph}" y2="{top + ph + 4}" stroke="#475569"/>'
                     f'<text x="{x(v):.1f}" y="{top + ph + 16}" text-anchor="middle" fill="#334155">{v:+.3g}</text>')
    for frac in (0.5, 1.0):
        parts.append(f'<text x="{left - 6}" y="{y(peak * frac) + 4:.1f}" text-anchor="end" fill="#64748b">'
                     f'{int(peak * frac):,}</text>'
                     f'<line x1="{left}" x2="{left + pw}" y1="{y(peak * frac):.1f}" y2="{y(peak * frac):.1f}" '
                     f'stroke="#e2e8f0"/>')
    parts.append(f'<text x="{left + pw / 2}" y="{H - 6}" text-anchor="middle" fill="#334155">signed deviation (mm), '
                 f'shaded = ±{tolerance:g} tolerance</text>')
    if hist.get("underflow") or hist.get("overflow"):
        parts.append(f'<text x="{left + 4}" y="{top + 10}" fill="#64748b">below range: {hist.get("underflow", 0):,}'
                     f'</text><text x="{left + pw - 4}" y="{top + 10}" text-anchor="end" fill="#64748b">above range: '
                     f'{hist.get("overflow", 0):,}</text>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f1f5f9; color: #0f172a; font: 14px/1.45 "Segoe UI", system-ui, -apple-system, Roboto,
       "Helvetica Neue", Arial, sans-serif; }
.page { max-width: 900px; margin: 24px auto; background: #fff; padding: 36px 40px; border-radius: 10px;
        box-shadow: 0 1px 3px rgba(15,23,42,.08), 0 8px 24px rgba(15,23,42,.06); }
header { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px;
         border-bottom: 2px solid #0f172a; padding-bottom: 16px; margin-bottom: 20px; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -.01em; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: #475569; margin: 28px 0 10px;
     border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
.sub { color: #475569; font-size: 13px; }
.sub b { color: #0f172a; font-weight: 600; }
.badge { font-size: 20px; font-weight: 700; letter-spacing: .06em; padding: 10px 20px; border-radius: 8px;
         border: 2px solid; text-align: center; min-width: 120px; }
.badge small { display: block; font-size: 11px; font-weight: 500; letter-spacing: 0; }
.pass { color: #15803d; border-color: #16a34a; background: #f0fdf4; }
.fail { color: #b91c1c; border-color: #dc2626; background: #fef2f2; }
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; }
.card .k { color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
.card .v { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; border-bottom: 1px solid #e2e8f0; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { font-size: 12px; color: #475569; font-weight: 600; background: #f8fafc; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.warn { background: #fffbeb; border: 1px solid #fcd34d; color: #78350f; border-radius: 8px; padding: 10px 14px; }
.warn li { margin: 2px 0; }
.muted { color: #64748b; font-size: 12px; }
code { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
footer { margin-top: 28px; color: #94a3b8; font-size: 11px; border-top: 1px solid #e2e8f0; padding-top: 8px; }
@media (max-width: 700px) { .cards { grid-template-columns: repeat(2, 1fr); } .two { grid-template-columns: 1fr; }
  .page { padding: 20px 16px; margin: 0; border-radius: 0; } header { flex-direction: column; } }
@media print { body { background: #fff; } .page { box-shadow: none; margin: 0; max-width: none; padding: 0; }
  h2 { break-after: avoid; } table, svg, .cards { break-inside: avoid; } @page { margin: 14mm; } }
"""


def render_report(meta: dict, report: dict, scan_name: str, reference_name: str, pass_pct: float = 95.0) -> str:
    esc = html.escape
    s, tol = report["stats"], report["tolerance"]
    passed = s["within_tolerance_pct"] >= pass_pct
    al, cov, scale = report.get("alignment", {}), report.get("coverage", {}), report.get("scale_estimate")
    dims = report.get("dimensions", {})
    warnings = list(report.get("warnings", []))

    cards = [("Within ±" + f"{tol:g}", f"{s['within_tolerance_pct']:.1f} %"), ("Mean", _num(s["mean"], 4, True)),
             ("Std dev", _num(s["std"])), ("RMS", _num(s["rms"])),
             ("P05 / P95", f"{_num(s['p05'], 3, True)} / {_num(s['p95'], 3, True)}"),
             ("Min / Max", f"{_num(s['min'], 3, True)} / {_num(s['max'], 3, True)}"),
             ("Mean |dev|", _num(s["abs_mean"])),
             ("Coverage", f"{cov.get('covered_area_pct', 0):.1f} %")]
    cards_html = "".join(f'<div class="card"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
                         for k, v in cards)

    dim_rows = ""
    if dims:
        for i, axis in enumerate("XYZ"):
            dim_rows += (f"<tr><td>{axis}</td><td>{dims['reference'][i]:.4f}</td><td>{dims['scan'][i]:.4f}</td>"
                         f"<td>{dims['difference'][i]:+.4f}</td></tr>")

    stat_rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in [
        ("Points measured", f"{s['points']:,}"), ("Points excluded (farther than max distance)", f"{s['excluded']:,}"),
        ("Max distance", _num(report.get("max_distance"))), ("Mean deviation", _num(s["mean"], 4, True)),
        ("Standard deviation", _num(s["std"])), ("RMS", _num(s["rms"])), ("Mean |deviation|", _num(s["abs_mean"])),
        ("Minimum", _num(s["min"], 4, True)), ("Maximum", _num(s["max"], 4, True)),
        ("5th percentile", _num(s["p05"], 4, True)), ("95th percentile", _num(s["p95"], 4, True)),
        ("Within tolerance", f"{s['within_tolerance_pct']:.2f} % (±{tol:g})")])

    align_rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(str(v))}</td></tr>" for k, v in [
        ("Method", al.get("method", "–")), ("Best start", al.get("best_candidate", "–")),
        ("Fitness (share of points close to the surface)", _num(al.get("fitness"), 3)),
        ("RMS of included points", _num(al.get("rmse"))),
        ("Scan point spacing", _num(report.get("spacing"), 5)),
        ("Scale estimate (not applied)",
         f"{scale['factor']:.5f} ({scale['percent']:+.3f} %)" if scale else "not estimated"),
        ("Coverage threshold", _num(cov.get("threshold"))),
        ("Symmetry ambiguity", "yes – check visually" if al.get("ambiguous") else "none detected")])
    cand_rows = "".join(f"<tr><td>{esc(c['name'])}</td><td>{c['score']:.3f}</td><td>{_num(c['rmse'], 4)}</td>"
                        f"<td>{c['angle_from_best']:.1f}°</td></tr>" for c in al.get("candidates", []))
    matrix = "<br>".join(" ".join(f"{v:+.6f}" for v in row) for row in report.get("transform", []))
    warn_html = ('<div class="warn"><b>Warnings</b><ul>' + "".join(f"<li>{esc(w)}</li>" for w in warnings) +
                 "</ul></div>") if warnings else '<p class="muted">No warnings.</p>'
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = f"Inspection report – {scan_name} vs {reference_name}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head>
<body><div class="page">
<header><div><h1>Scan inspection report</h1>
<div class="sub">Scan <b>{esc(scan_name)}</b> compared with reference <b>{esc(reference_name)}</b></div>
<div class="sub">Result asset <b>{esc(meta.get('name', ''))}</b> · computed {esc(str(meta.get('created', ''))[:16])}
· generated {generated} · units mm</div></div>
<div class="badge {'pass' if passed else 'fail'}">{'PASS' if passed else 'FAIL'}
<small>{s['within_tolerance_pct']:.1f} % within ±{tol:g} (≥ {pass_pct:g} % required)</small></div></header>
<div class="cards">{cards_html}</div>
<h2>Deviation distribution</h2>{_histogram_svg(report['histogram'], tol)}
<p class="muted">Positive = scan material outside the CAD surface, negative = inside. Points farther than
{_num(report.get('max_distance'), 3)} are treated as not belonging to the part and are excluded. The scan was only
moved rigidly; no scaling was applied.</p>
<div class="two"><div><h2>Statistics</h2><table>{stat_rows}</table></div>
<div><h2>Alignment</h2><table>{align_rows}</table></div></div>
<h2>Dimensions (axis-aligned in the CAD frame)</h2>
<table><tr><th>Axis</th><th>Reference</th><th>Scan</th><th>Difference</th></tr>{dim_rows}</table>
<p class="muted">Scan extents only cover the scanned surface ({cov.get('covered_area_pct', 0):.1f} % of the CAD
surface); unscanned sides make an extent shorter than the part.</p>
<h2>Warnings</h2>{warn_html}
<h2>Alignment candidates</h2>
<table><tr><th>Start</th><th>Share within 2× spacing</th><th>RMS</th><th>Rotation from best</th></tr>{cand_rows}</table>
<h2>Transform (scan → CAD)</h2><p><code>{matrix}</code></p>
<footer>CloudClean inspection · {esc(title)}</footer>
</div></body></html>"""


def create_router(workspace, jobs) -> APIRouter:
    router = APIRouter()
    ws = workspace

    def meta_or_400(asset_id: str, role: str) -> dict:
        try:
            return ws.get(asset_id)
        except KeyError:
            raise HTTPException(400, f"The {role} asset '{asset_id}' does not exist")

    @router.post("/api/compare")
    def compare(req: CompareReq):
        try:
            params = CompareParams.from_dict(req.params)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc))
        if params.align not in ("auto", "icp", "none"):
            raise HTTPException(400, "align must be auto, icp or none")
        if params.tolerance <= 0:
            raise HTTPException(400, "The tolerance must be greater than zero")
        if params.max_distance < 0 or params.coverage_threshold < 0 or params.sample_points < 0:
            raise HTTPException(400, "max_distance, coverage_threshold and sample_points cannot be negative")
        if req.scan_id == req.reference_id:
            raise HTTPException(400, "Choose a different asset as the reference")
        scan = meta_or_400(req.scan_id, "scan")
        ref = meta_or_400(req.reference_id, "reference")
        if scan["kind"] not in ("pointcloud", "mesh"):
            raise HTTPException(400, f"'{scan['name']}' is a photo, not a scan")
        if ref["kind"] != "mesh":
            raise HTTPException(400, f"'{ref['name']}' is not a mesh - the reference must be a CAD model (STEP) or "
                                     "a mesh (STL)")
        payload = {"scan_id": req.scan_id, "reference_id": req.reference_id, "params": params.to_dict(),
                   "name": req.name}
        return jobs.submit("compare", f"Compare {scan['name']} with {ref['name']}", payload)

    @router.get("/api/assets/{asset_id}/inspection-report", response_class=HTMLResponse)
    def inspection_report(asset_id: str, pass_pct: float = 95.0):
        try:
            meta = ws.get(asset_id)
        except KeyError:
            raise HTTPException(404, "Asset not found")
        report = ws.report(asset_id) or {}
        if "compare_asset" in report:  # the coverage mesh points at the compared scan
            try:
                meta, report = ws.get(report["compare_asset"]["id"]), ws.report(report["compare_asset"]["id"]) or {}
            except KeyError:
                raise HTTPException(404, "The compared scan of this coverage mesh was deleted")
        if "stats" not in report or "histogram" not in report:
            raise HTTPException(400, f"'{meta['name']}' is not a comparison result")

        def name(key: str) -> str:
            info = report.get(key) or {}
            try:
                return ws.get(info["id"])["name"]
            except (KeyError, TypeError):
                return info.get("name", "(deleted)") if isinstance(info, dict) else "(deleted)"

        page = render_report(meta, report, name("scan_asset"), name("reference_asset"), pass_pct)
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    return router
