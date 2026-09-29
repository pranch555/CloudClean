import type { Asset } from './types';
import type { PartSummary } from './summary';
import { partDims } from './summary';
import { fmtLen } from './format';
import { ROLE_LABEL, roleOf } from './journey';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';

const esc = (s: unknown) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);

/**
 * A self-contained, printable HTML measurement report of one model: a picture of the current view, the part size,
 * every measurement on screen, the thread analysis and where the model came from. Print it to PDF from the browser.
 */
export function measurementReport(asset: Asset, summary: PartSummary | null): string {
  const st = useStore.getState();
  const units = st.display.units;
  const image = getViewer()?.screenshot(1600) ?? '';
  const { dims, robust } = partDims(asset, summary);
  const parents = asset.parents.map(id => st.byId.get(id)?.name).filter(Boolean);
  const measurements = st.measurements.filter(m => m.b && m.result && (!m.assetId || m.assetId === asset.id));
  const lines = st.dims.filter(d => !d.assetId || d.assetId === asset.id);
  const thread = st.thread && st.thread.asset_id === asset.id ? st.thread : null;
  const project = st.projects.find(p => p.id === st.projectId)?.name ?? '';
  const when = new Date();

  const rows = [
    ...lines.map(d => [d.label, d.kind, d.value == null ? '—' : `${fmtLen(d.value, 3)} ${d.kind === 'angle' ? '°' : d.unit}`]),
    ...measurements.map(m => [m.label, 'point to point', `${fmtLen(m.result!.distance, 4)} ${units}`]),
  ];

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>${esc(asset.name)} — measurement report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --ink:#17161a; --ink2:#4f4c45; --ink3:#6e6a61; --line:#e6e2d9; --paper:#f4f2ed; --signal:#ee4b1f; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 "Instrument Sans", system-ui, "Segoe UI", sans-serif; }
  .sheet { max-width: 900px; margin: 32px auto; background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 40px 48px; }
  header { display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 2px solid var(--ink); padding-bottom: 14px; margin-bottom: 24px; }
  h1 { font: 400 40px/1.05 "Instrument Serif", Georgia, serif; margin: 0; }
  .kicker { font-size: 13px; color: var(--ink3); }
  .mono, td.v { font-family: "Geist Mono", ui-monospace, Consolas, monospace; font-variant-numeric: tabular-nums; }
  .size { display: flex; gap: 28px; margin: 18px 0 8px; }
  .size div { display: grid; }
  .size b { font: 500 30px/1.1 "Geist Mono", ui-monospace, Consolas, monospace; }
  .size span { font-size: 12px; color: var(--ink3); font-weight: 600; }
  img { width: 100%; border-radius: 12px; border: 1px solid var(--line); margin: 12px 0 4px; }
  h2 { font-size: 17px; margin: 28px 0 8px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--line); }
  th { font-size: 12px; color: var(--ink3); font-weight: 600; }
  td.v { text-align: right; }
  .note { font-size: 13px; color: var(--ink2); }
  .chip { display: inline-block; background: var(--signal); color: var(--ink); border-radius: 999px; padding: 0 8px; font-size: 12px; font-weight: 700; }
  footer { margin-top: 28px; font-size: 12px; color: var(--ink3); border-top: 1px solid var(--line); padding-top: 12px; }
  @media print { body { background: #fff; } .sheet { border: 0; margin: 0; max-width: none; padding: 0; } }
</style></head>
<body><div class="sheet">
<header>
  <div><div class="kicker">${esc(project)} · ${esc(ROLE_LABEL[roleOf(asset)])}</div><h1>${esc(asset.name)}</h1></div>
  <div class="kicker mono">${esc(when.toLocaleString())}</div>
</header>
<div class="kicker">${robust ? 'Part size along its own axes (robust: stray points ignored)' : 'Fitted box along the part’s axes'}</div>
<div class="size">
  <div><span>LENGTH</span><b>${fmtLen(dims[0], 3)}</b></div>
  <div><span>WIDTH</span><b>${fmtLen(dims[1], 3)}</b></div>
  <div><span>HEIGHT</span><b>${fmtLen(dims[2], 3)}</b></div>
  <div><span>UNIT</span><b style="font-size:18px">${esc(units)}</b></div>
</div>
${summary?.description ? `<p class="note">${esc(summary.description)}</p>` : ''}
${image ? `<img src="${image}" alt="The model as shown in CloudClean">` : ''}
<h2>Measurements</h2>
${rows.length ? `<table><thead><tr><th>Label</th><th>Kind</th><th style="text-align:right">Value</th></tr></thead><tbody>${rows.map(r => `<tr><td><span class="chip">${esc(r[0])}</span></td><td>${esc(r[1])}</td><td class="v">${esc(r[2])}</td></tr>`).join('')}</tbody></table>` : '<p class="note">No measurements were taken on this model.</p>'}
${thread ? `<h2>Thread</h2><table><tbody>
  <tr><td>Nearest standard</td><td class="v">${esc(thread.standards?.[0]?.designation ?? '—')}</td></tr>
  <tr><td>Pitch</td><td class="v">${fmtLen(thread.pitch, 4)} ± ${fmtLen(thread.pitch_se, 2)} ${esc(units)}</td></tr>
  <tr><td>Major diameter</td><td class="v">${fmtLen(thread.major_diameter, 3)} ${esc(units)}</td></tr>
  <tr><td>Minor diameter</td><td class="v">${fmtLen(thread.minor_diameter, 3)} ${esc(units)}</td></tr>
  <tr><td>Pitch diameter</td><td class="v">${fmtLen(thread.pitch_diameter, 3)} ${esc(units)}</td></tr>
  <tr><td>Crests · handedness · confidence</td><td class="v">${thread.crest_count} · ${esc(thread.handedness)} · ${esc(thread.confidence)}</td></tr>
</tbody></table>` : ''}
<h2>Model</h2>
<table><tbody>
  <tr><td>${asset.kind === 'mesh' ? 'Triangles' : 'Points'}</td><td class="v">${(asset.kind === 'mesh' ? asset.stats.triangles : asset.stats.points)?.toLocaleString() ?? '—'}</td></tr>
  <tr><td>Point spacing</td><td class="v">${fmtLen(asset.stats.spacing, 3)} ${esc(units)}</td></tr>
  <tr><td>Scanner box (X × Y × Z)</td><td class="v">${asset.stats.dimensions.map(v => fmtLen(v, 2)).join(' × ')} ${esc(units)}</td></tr>
  ${parents.length ? `<tr><td>Made from</td><td class="v">${esc(parents.join(', '))}</td></tr>` : ''}
</tbody></table>
<footer>Made with CloudClean. Every value is computed from the full-resolution data; nothing was rescaled. Print this page to save it as a PDF.</footer>
</div></body></html>`;
}

export function downloadReport(asset: Asset, summary: PartSummary | null) {
  const html = measurementReport(asset, summary);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([html], { type: 'text/html' }));
  a.download = `${asset.name.replace(/[^\w\-. ]+/g, '_')} - measurement report.html`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}
