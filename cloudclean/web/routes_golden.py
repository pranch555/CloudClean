"""Golden model check (cloudclean/golden.py, docs/golden-model.md).

POST /api/golden-check {scan_id, golden_id?, tolerance?, align?, remember?}
     golden_id defaults to the project's golden model; remember: make golden_id the project's golden model
GET  /api/assets/{id}/golden-report   a printable page of a check (the check's mesh or its compared scan)
"""
from __future__ import annotations

import html
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..golden import LEGEND, GoldenParams


class GoldenReq(BaseModel):
    scan_id: str
    golden_id: str | None = None
    tolerance: float | None = None
    align: str | None = None
    remember: bool = False
    name: str | None = None
    up_axis: str | None = None      # the 3D view's up axis ('y' or 'z'): only for the few names that say 'top'


STATUS_TEXT = {"ok": "Matches", "off": "Off", "close": "Too close to call", "not_measured": "Not measured"}
KIND_TEXT = {"missing": "Not scanned", "thin": "Too few points", "rough": "Rough", "off": "Off"}


def _mm(v, signed: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def render_golden_report(meta: dict, report: dict, scan_name: str, golden_name: str) -> str:
    e = html.escape
    tol = report.get("tolerance", 0)
    verdict = report.get("verdict", "")
    tone = {"match": "#15803d", "incomplete": "#b45309", "differs": "#b91c1c"}.get(verdict, "#334155")
    rows = []
    for m in report.get("measurements", []):
        st = m.get("status")
        colour = {"ok": "#15803d", "off": "#b91c1c", "close": "#b45309"}.get(st, "#64748b")
        note = m.get("reason") or ""
        if m.get("kind") == "position" and m.get("toward"):
            note = f"moved toward {m['toward']}"
        rows.append(f"<tr><td>{e(m['name'])}</td><td class=n>{_mm(m.get('golden'))}</td><td class=n>{_mm(m.get('scan'))}</td>"
                    f"<td class=n>{_mm(m.get('difference'), True)}</td><td class=n>{'±' + _mm(m['uncertainty']) if m.get('uncertainty') else '–'}</td>"
                    f"<td style='color:{colour};font-weight:600'>{STATUS_TEXT.get(st, st)}</td><td class=small>{e(note)}</td></tr>")
    regions = []
    for r in report.get("regions", []):
        detail = {"off": f"{_mm(r.get('deviation'), True)} mm", "rough": f"±{_mm(r.get('spread'))} mm",
                  "thin": f"{r.get('density_pct') or 0:.0f} % of the usual points"}.get(r["kind"], "")
        regions.append(f"<tr><td>{r['id'] + 1}</td><td><b>{e(r['name'])}</b><div class=small>{e(r['why'])}</div></td>"
                       f"<td>{KIND_TEXT.get(r['kind'], r['kind'])}</td><td class=n>{r['area_mm2']:.0f} mm² "
                       f"({r['share_pct']:.1f} %)</td><td class=n>{e(detail)}</td><td class=small>{e(r['advice'])}</td></tr>")
    surface = report.get("surface", {})
    shares = surface.get("shares_pct", {})
    legend = "".join(f"<span class=chip><i style='background:{g['color']}'></i>{e(g['label'])} "
                     f"{shares.get(g['key'], 0):.1f} %</span>" for g in LEGEND)
    summary = "".join(f"<li>{e(s)}</li>" for s in report.get("summary", []))
    rescan = "".join(f"<li>{e(s)}</li>" for s in report.get("rescan", [])) or "<li>Nothing: every area was scanned well enough.</li>"
    warnings = "".join(f"<li>{e(w)}</li>" for w in report.get("warnings", []))
    dev = report.get("deviation", {})
    align = report.get("alignment", {})
    when = e(meta.get("created", "")[:19].replace("T", " ")) or datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Golden model check - {e(scan_name)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#1e293b;margin:32px auto;max-width:980px;padding:0 16px;line-height:1.45}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} .meta{{color:#64748b;font-size:13px}}
.verdict{{border-left:5px solid {tone};background:#f8fafc;padding:12px 16px;margin:18px 0;font-size:17px;font-weight:600;color:{tone}}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left;vertical-align:top}}
th{{font-size:12px;color:#475569;text-transform:uppercase;letter-spacing:.03em}} td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.small{{font-size:12px;color:#475569}} .chip{{display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;font-size:13px}}
.chip i{{width:12px;height:12px;border-radius:3px;display:inline-block}} ul{{margin:6px 0;padding-left:20px}}
@media print{{body{{margin:0}} .noprint{{display:none}}}}
</style></head><body>
<h1>Golden model check</h1>
<div class=meta>Scan <b>{e(scan_name)}</b> against the golden model <b>{e(golden_name)}</b> · tolerance ±{tol:g} mm · {when}</div>
<div class=verdict>{e(report.get('headline', ''))}</div>
<ul>{summary}</ul>
<h2>What to scan again</h2><ol>{rescan}</ol>
<h2>Measurements (mm)</h2>
<table><thead><tr><th>Measurement</th><th>Golden</th><th>Scan</th><th>Difference</th><th>Certainty</th><th>Result</th><th></th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan=7>No flat or round faces were found to measure.</td></tr>'}</tbody></table>
<p class=small>Certainty: how precisely the scan pins the value down (twice the standard uncertainty). A value only
counts as matching when it is inside ±{tol:g} mm even allowing for that; "too close to call" means it is within
that margin of the limit.</p>
<h2>The surface</h2><div>{legend}</div>
<table><thead><tr><th>#</th><th>Area</th><th>Problem</th><th>Size</th><th>By</th><th>What to do</th></tr></thead>
<tbody>{''.join(regions) or '<tr><td colspan=6>No problem areas.</td></tr>'}</tbody></table>
<h2>Details</h2>
<table><tbody>
<tr><td>Surface scanned</td><td class=n>{surface.get('scanned_pct', 0):.1f} %</td></tr>
<tr><td>Deviation of the scanned points (mean / std / 5-95 %)</td><td class=n>{_mm(dev.get('mean'), True)} / {_mm(dev.get('std'))} / {_mm(dev.get('p05'), True)} … {_mm(dev.get('p95'), True)}</td></tr>
<tr><td>Scanner noise (typical scatter)</td><td class=n>±{_mm(surface.get('noise'))}</td></tr>
<tr><td>Line-up: points on the golden surface / rms</td><td class=n>{(align.get('fitness') or 0) * 100:.1f} % / {_mm(align.get('rmse'))}</td></tr>
</tbody></table>
{f'<h2>Warnings</h2><ul>{warnings}</ul>' if warnings else ''}
<p class="small noprint"><button onclick="print()">Print</button></p>
</body></html>"""


def create_router(workspace, jobs) -> APIRouter:
    router = APIRouter()
    ws = workspace

    def meta_or_400(asset_id: str, role: str) -> dict:
        try:
            return ws.get(asset_id)
        except KeyError:
            raise HTTPException(400, f"The {role} '{asset_id}' does not exist")

    @router.post("/api/golden-check")
    def golden_check(req: GoldenReq):
        scan = meta_or_400(req.scan_id, "scan")
        if scan["kind"] not in ("pointcloud", "mesh"):
            raise HTTPException(400, f"'{scan['name']}' is a photo, not a scan")
        golden_id = req.golden_id
        if not golden_id:
            try:
                golden_id = ws.get_project(scan["project"]).get("golden_asset_id") if scan.get("project") else None
                if golden_id:
                    ws.get(golden_id)             # still there?
            except KeyError:
                golden_id = None
            if not golden_id:
                raise HTTPException(400, "This project has no golden model yet: import the CAD file or mesh of the "
                                         "part and pick it as the golden model")
        golden = meta_or_400(golden_id, "golden model")
        if golden_id == req.scan_id:
            raise HTTPException(400, "The scan cannot be its own golden model: pick the CAD model or mesh")
        if golden["kind"] != "mesh":
            raise HTTPException(400, f"'{golden['name']}' is not a mesh. The golden model must be a CAD model (STEP) "
                                     "or a mesh (STL/OBJ/PLY): import the CAD file, or make a mesh first")
        params = GoldenParams()
        if req.tolerance is not None:
            if not req.tolerance > 0:
                raise HTTPException(400, "The tolerance must be more than 0")
            params.tolerance = float(req.tolerance)
        if req.align is not None:
            if req.align not in ("auto", "icp", "none"):
                raise HTTPException(400, "align must be auto, icp or none")
            params.align = req.align
        if req.remember and scan.get("project"):
            try:
                ws.update_project(scan["project"], golden_asset_id=golden_id)
            except (KeyError, ValueError) as exc:
                raise HTTPException(400, str(exc))
        payload = {"scan_id": req.scan_id, "golden_id": golden_id, "params": params.to_dict(), "name": req.name,
                   "up_axis": req.up_axis if req.up_axis in ("y", "z") else "y"}
        return jobs.submit("golden_check", f"Check {scan['name']} against {golden['name']}", payload)

    @router.get("/api/assets/{asset_id}/golden-report", response_class=HTMLResponse)
    def golden_report(asset_id: str):
        try:
            meta = ws.get(asset_id)
        except KeyError:
            raise HTTPException(404, "Asset not found")
        report = ws.report(asset_id) or {}
        if "verdict" not in report:
            raise HTTPException(400, f"'{meta['name']}' is not a golden model check")

        def name(key: str) -> str:
            info = report.get(key) or {}
            try:
                return ws.get(info["id"])["name"]
            except (KeyError, TypeError):
                return info.get("name", "(deleted)") if isinstance(info, dict) else "(deleted)"

        page = render_golden_report(meta, report, name("scan_asset"), name("golden_asset"))
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    return router
