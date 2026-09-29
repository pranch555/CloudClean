import { useState } from 'react';
import { ChevronDown, Cpu, Download, FileText, FolderDown } from 'lucide-react';
import { api, ApiError } from '../lib/api';
import type { Asset } from '../lib/types';
import { fmtCount, fmtDims, fmtLen, OPERATION_LABEL } from '../lib/format';
import { csvOfMeasurements, isSizeLine } from '../lib/measure';
import { ROLE_LABEL, roleOf } from '../lib/journey';
import { partDims, useSummary } from '../lib/summary';
import { downloadReport } from '../lib/report';
import { useStore, useTarget } from '../store';
import { ExportGlyph } from '../ui/icons';
import { Badge, Button, Metric, Select, TextInput } from '../ui/primitives';
import { ReportView } from './ReportView';
import { Block, StepFrame, TargetCard } from './StepFrame';
import { useReport } from './useReport';

const FORMAT_INFO: Record<string, string> = {
  stl: '3D printing, CAD',
  '3mf': '3D printing, keeps millimetres',
  obj: 'CAD, rendering',
  ply: 'keeps every point and colour',
  glb: 'web and AR viewers',
  off: 'geometry tools',
  pcd: 'point cloud libraries',
  xyz: 'plain text points',
  asc: 'plain text points',
  csv: 'spreadsheets',
};

export function ExportStep() {
  const target = useTarget();
  const projectId = useStore(s => s.projectId);
  return (
    <StepFrame
      step="export"
      purpose="Save the part in the format you need, with a report of its measurements. Files are made from the full-resolution data."
      footer={
        <Button variant="ghost" block icon={<Cpu size={16} />} onClick={() => useStore.getState().set({ settingsOpen: 'automations' })}>
          Automate this for every new scan
        </Button>
      }
    >
      <TargetCard asset={target} label="Exporting" empty="Click the model you want to save in the list on the left." />
      {target && <Downloads asset={target} onDone={() => projectId && useStore.setState(s => ({ exported: { ...s.exported, [projectId]: true } }))} />}
      {target && <Details asset={target} />}
    </StepFrame>
  );
}

function Downloads({ asset: a, onDone }: { asset: Asset; onDone: () => void }) {
  const formats = useStore(s => s.params?.formats);
  const summary = useSummary(a);
  const measurements = useStore(s => s.measurements.filter(m => m.b).length + s.dims.filter(d => !isSizeLine(d.id)).length);
  const units = useStore(s => s.display.units);
  const [folder, setFolder] = useState(() => localStorage.getItem('cloudclean.exportFolder') ?? '');
  const list = formats?.[a.kind as 'mesh' | 'pointcloud'] ?? [];
  const [fmt, setFmt] = useState(a.kind === 'mesh' ? 'stl' : 'ply');

  const exportToServer = async (overwrite = false) => {
    try {
      localStorage.setItem('cloudclean.exportFolder', folder);
      const res = await api.post<{ path: string }>('/api/export', { asset_id: a.id, format: fmt, folder, overwrite });
      useStore.getState().toast({ kind: 'ok', title: 'Saved', body: res.path });
      onDone();
    } catch (err) {
      if (err instanceof ApiError && err.status === 409 && confirm(`${err.message}\n\nReplace it?`)) return exportToServer(true);
      useStore.getState().toast({ kind: 'error', title: 'Could not save', body: (err as Error).message });
    }
  };

  const csv = () => {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([csvOfMeasurements(units)], { type: 'text/csv' }));
    link.download = `${a.name}-measurements.csv`;
    link.click();
  };

  return (
    <>
      <Block title="Download to this computer" guide="export.download">
        <div className="format-grid">
          {list.map(f => (
            <a key={f} className="format-card" href={`/api/assets/${a.id}/download?format=${f}`} download onClick={onDone}>
              <span className="format-ext mono">.{f}</span>
              <span className="format-use">{FORMAT_INFO[f] ?? ''}</span>
              <Download size={15} className="format-dl" aria-hidden />
            </a>
          ))}
          {a.textured && (
            <>
              <a className="format-card" href={`/api/assets/${a.id}/download?format=textured_glb`} download onClick={onDone}>
                <span className="format-ext mono">.glb</span><span className="format-use">with photo texture</span><Download size={15} className="format-dl" aria-hidden />
              </a>
              <a className="format-card" href={`/api/assets/${a.id}/download?format=textured_obj`} download onClick={onDone}>
                <span className="format-ext mono">.obj</span><span className="format-use">with photo texture</span><Download size={15} className="format-dl" aria-hidden />
              </a>
            </>
          )}
        </div>
      </Block>

      <Block title="Reports" guide="export.reports">
        <Button variant="primary" icon={<FileText size={15} />} onClick={() => downloadReport(a, summary)}>Measurement report (print or save as PDF)</Button>
        <div className="chip-row">
          <a className="btn btn-secondary btn-sm" href={`/api/assets/${a.id}/download?format=report`} download><FileText size={14} aria-hidden /><span className="btn-label">Processing report (JSON)</span></a>
          {a.operation === 'compare' && <a className="btn btn-secondary btn-sm" href={`/api/assets/${a.id}/inspection-report`} target="_blank" rel="noreferrer"><FileText size={14} aria-hidden /><span className="btn-label">Inspection report</span></a>}
          <Button size="sm" icon={<FileText size={14} />} disabled={!measurements} onClick={csv}>Measurements (CSV{measurements ? ` · ${measurements}` : ''})</Button>
        </div>
      </Block>

      <details className="disclosure" data-guide="export.server">
        <summary><FolderDown size={16} aria-hidden /> Save on the server <span className="sub">a folder or network share</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
        <div className="disclosure-body">
          <p className="hint-text">Writes straight to disk on the machine running CloudClean (for example the DGX Spark) — useful for big files and shared folders.</p>
          <TextInput mono value={folder} placeholder="/mnt/share/exports" onChange={setFolder} />
          <div className="row">
            <div className="grow"><Select value={fmt} onChange={setFmt} options={list.map(f => ({ value: f, label: f.toUpperCase() }))} /></div>
            <Button variant="primary" icon={<ExportGlyph size={16} />} disabled={!folder.trim()} onClick={() => exportToServer()}>Save</Button>
          </div>
        </div>
      </details>
    </>
  );
}

function Details({ asset: a }: { asset: Asset }) {
  const units = useStore(s => s.display.units);
  const byId = useStore(s => s.byId);
  const report = useReport(a);
  const summary = useSummary(a);
  const { dims, robust } = partDims(a, summary);
  const s = a.stats;
  const parents = a.parents.map(id => byId.get(id)).filter(Boolean) as Asset[];
  return (
    <details className="disclosure" open>
      <summary>About this model <span className="sub">{ROLE_LABEL[roleOf(a)]} · {OPERATION_LABEL[a.operation] ?? a.operation} {new Date(a.created).toLocaleString()}</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
      <div className="disclosure-body">
        {summary?.description && <p className="hint-text">{summary.description}</p>}
        <div>
          <Metric label={robust ? 'Part size (L × W × H)' : 'Smallest box (L × W × H)'} value={fmtDims(dims)} unit={units} />
          <Metric label="Scanner box (X × Y × Z)" value={fmtDims(s.dimensions)} unit={units} />
          <Metric label={a.kind === 'mesh' ? 'Triangles' : 'Points'} value={fmtCount(a.kind === 'mesh' ? s.triangles : s.points)} />
          <Metric label="Point spacing" value={fmtLen(s.spacing, 3)} unit={units} />
          {s.surface_area != null && <Metric label="Surface area" value={fmtLen(s.surface_area)} unit={`${units}²`} />}
          {s.volume != null && <Metric label="Volume" value={fmtLen(s.volume)} unit={`${units}³`} />}
          {a.kind === 'mesh' && <Metric label="Watertight" value={s.watertight ? 'yes' : 'no'} />}
          <Metric label="Colours" value={s.has_colors ? 'yes' : 'no'} />
        </div>
        {parents.length > 0 && (
          <div className="stack tight">
            <span className="caption">Made from</span>
            <div className="chip-row">
              {parents.map(p => <button key={p.id} type="button" className="chip" onClick={() => useStore.getState().activate(p.id)}>{p.name}</button>)}
            </div>
          </div>
        )}
        {a.scalars?.length ? <div className="chip-row">{a.scalars.map(sc => <Badge key={sc.name}>{sc.name}: {fmtLen(sc.min)} … {fmtLen(sc.max)} {sc.unit}</Badge>)}</div> : null}
        {report && <ReportView asset={a} report={report} />}
      </div>
    </details>
  );
}
