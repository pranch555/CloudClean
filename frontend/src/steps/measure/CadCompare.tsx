import { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ExternalLink, Gauge, Layers, TriangleAlert, Upload, CheckCircle2, XCircle, Eye } from 'lucide-react';
import { create } from 'zustand';
import { api } from '../../lib/api';
import type { Asset } from '../../lib/types';
import { submitJob } from '../../lib/jobs';
import { fmtCount, fmtLen, fmtPct, fmtSigned } from '../../lib/format';
import { pickFiles } from '../../lib/importing';
import { roleOf } from '../../lib/journey';
import { useProjectAssets, useStore } from '../../store';
import { getViewer } from '../../viewer/instance';
import { Badge, Button, Empty, Field, NumberInput, Segmented, Select, Stat } from '../../ui/primitives';
import { Block } from '../StepFrame';
import { CadGlyph } from './glyphs';
import { Histogram } from './Histogram';

export const isCad = (a: Asset) => roleOf(a) === 'cad' || /\.(step|stp|iges|igs|brep|stl|obj)$/i.test(String(a.params?.source ?? ''));

interface CadSetup {
  scanId: string;
  refId: string;
  align: 'auto' | 'icp' | 'none';
  tolerance: number;
  maxDistance: number;
  busy: boolean;
}

const useCad = create<CadSetup>(() => ({ scanId: '', refId: '', align: 'auto', tolerance: 0.1, maxDistance: 0, busy: false }));

const ALIGN_HELP: Record<CadSetup['align'], string> = {
  auto: 'Finds how the scan sits on the CAD model by itself.',
  icp: 'Only fine-tunes: use when the scan is already roughly in place.',
  none: 'Compares as they are — the scan is already in the CAD coordinates.',
};

/** Scans and CAD models of this project (of the whole workspace when the project has none). */
function useChoices() {
  const project = useProjectAssets();
  const all = useStore(s => s.assets);
  return useMemo(() => {
    const refs = (list: Asset[]) => list.filter(a => a.kind === 'mesh' && a.operation !== 'compare').sort((a, b) => Number(isCad(b)) - Number(isCad(a)) || b.created.localeCompare(a.created));
    const scanList = (list: Asset[]) => list.filter(a => a.kind !== 'image' && a.operation !== 'compare' && !isCad(a)).sort((a, b) => b.created.localeCompare(a.created));
    const references = refs(project).length ? refs(project) : refs(all);
    const scans = scanList(project).length ? scanList(project) : scanList(all);
    return { references, scans };
  }, [project, all]);
}

/** Scan versus design: how far every point is from the CAD surface, as a colour map and as numbers. */
export function CadCompare() {
  const activeId = useStore(s => s.activeId);
  const byId = useStore(s => s.byId);
  const units = useStore(s => s.display.units);
  const setup = useCad();
  const { references, scans } = useChoices();
  const active = activeId ? byId.get(activeId) : undefined;
  const result = active?.operation === 'compare' && active.kind === 'pointcloud' ? active : undefined;

  // sensible picks: the current model as the scan, the newest CAD import as the reference
  const key = `${scans.map(s => s.id).join()}|${references.map(r => r.id).join()}`;
  useEffect(() => {
    const cur = useCad.getState();
    const patch: Partial<CadSetup> = {};
    const activeIsScan = !!active && scans.some(s => s.id === active.id);
    if (activeIsScan && cur.scanId !== active!.id) patch.scanId = active!.id;
    else if (!scans.some(s => s.id === cur.scanId) && scans[0]) patch.scanId = scans[0].id;
    if (!references.some(r => r.id === cur.refId) && references[0]) patch.refId = references[0].id;
    if (Object.keys(patch).length) useCad.setState(patch);
  }, [active?.id, key]);

  if (!scans.length && !result) {
    return <Empty icon={<CadGlyph size={24} />} title="Nothing to compare yet">Scan the part, or bring in a scan together with its CAD model (STEP, IGES, STL).</Empty>;
  }
  const ref = references.find(r => r.id === setup.refId);
  const refIsScan = !!ref && !isCad(ref) && ref.operation !== 'import';

  const form = (
    <div className="fields">
      <Field label="Scan" htmlFor="cad-scan" inline={false}>
        <Select id="cad-scan" value={setup.scanId} onChange={scanId => useCad.setState({ scanId })} options={[{ value: '', label: 'Choose a scan…' }, ...scans.map(a => ({ value: a.id, label: `${a.name}${a.kind === 'mesh' ? ' (mesh)' : ''}` }))]} />
      </Field>
      <Field
        label="CAD model"
        htmlFor="cad-ref"
        inline={false}
        help={
          !references.length
            ? 'Bring in the design as STEP, IGES, STL or OBJ.'
            : refIsScan
              ? 'This mesh was made from a scan, not from the design: the check then compares two scans. Add the CAD file for a true design check.'
              : 'STEP and IGES files are turned into a fine mesh when they are imported.'
        }
      >
        {references.length ? (
          <div className="row">
            <div className="grow">
              <Select id="cad-ref" value={setup.refId} onChange={refId => useCad.setState({ refId })} options={references.map(a => ({ value: a.id, label: `${isCad(a) ? 'CAD · ' : ''}${a.name}` }))} />
            </div>
            <Button size="md" variant="ghost" icon={<Upload size={15} />} onClick={() => pickFiles('.step,.stp,.iges,.igs,.stl,.obj,.ply')}>Add</Button>
          </div>
        ) : (
          <Button icon={<Upload size={15} />} onClick={() => pickFiles('.step,.stp,.iges,.igs,.stl,.obj,.ply')}>Add the CAD model</Button>
        )}
      </Field>
      <Field label="Line them up" inline={false} help={ALIGN_HELP[setup.align]}>
        <Segmented value={setup.align} onChange={align => useCad.setState({ align })} options={[{ value: 'auto', label: 'Automatically' }, { value: 'icp', label: 'Fine-tune only' }, { value: 'none', label: 'As they are' }]} />
      </Field>
      <Field label="Tolerance ±" help="Points closer than this to the CAD surface count as good.">
        <NumberInput value={setup.tolerance} step={0.01} min={0.001} unit={units} onChange={tolerance => useCad.setState({ tolerance })} />
      </Field>
      <Field label="Ignore farther than" help="Leftover table or fixture points farther away are left out, and the report says how many. 0 = automatic (3 % of the part size).">
        <NumberInput value={setup.maxDistance} step={0.1} min={0} unit={units} onChange={maxDistance => useCad.setState({ maxDistance })} />
      </Field>
    </div>
  );

  return (
    <>
      {result && <CadResults asset={result} />}
      {result ? (
        <details className="disclosure">
          <summary>Compare again <span className="sub">another scan or setting</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
          <div className="disclosure-body">{form}</div>
        </details>
      ) : (
        <Block title="Compare to CAD">
          <p className="hint-text">Checks the scan against its design: how far every point lies from the CAD surface, shown as a colour map and as numbers.</p>
          {form}
        </Block>
      )}
    </>
  );
}

export function CadFooter() {
  const { scanId, refId, align, tolerance, maxDistance, busy } = useCad();
  const ready = !!scanId && !!refId && scanId !== refId;
  const run = async () => {
    useCad.setState({ busy: true });
    const job = await submitJob('/api/compare', { scan_id: scanId, reference_id: refId, params: { align, tolerance, max_distance: maxDistance } }, () => useCad.setState({ busy: false }));
    if (!job) useCad.setState({ busy: false });
  };
  return (
    <Button variant="primary" size="lg" block icon={<CadGlyph size={18} />} loading={busy} disabled={!ready} onClick={run}>
      {!scanId ? 'Choose a scan' : !refId ? 'Add the CAD model' : 'Compare to CAD'}
    </Button>
  );
}

/* eslint-disable @typescript-eslint/no-explicit-any */
export function CadResults({ asset }: { asset: Asset }) {
  const units = useStore(s => s.display.units);
  const display = useStore(s => s.display);
  const byId = useStore(s => s.byId);
  const assets = useStore(s => s.assets);
  const [report, setReport] = useState<Record<string, any> | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let alive = true;
    setReport(null);
    setFailed(false);
    api.asset(asset.id).then(r => alive && setReport(r.report)).catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [asset.id]);
  if (failed) return <p className="err-text"><TriangleAlert size={14} aria-hidden /> The report of this comparison could not be loaded.</p>;
  if (!report) return <p className="hint-text">Loading the result…</p>;

  const st = report.stats ?? {};
  const tol = Number(report.tolerance ?? 0.1);
  const within = Number(st.within_tolerance_pct ?? 0);
  const pass = within >= 95;
  const warn = !pass && within >= 80;
  const [scanId, refId] = asset.parents;
  const coverageAsset = assets.find(a => a.operation === 'compare' && a.kind === 'mesh' && a.parents.join() === asset.parents.join() && a.created >= asset.created);
  const dims = report.dimensions ?? {};
  const warnings: string[] = [...(report.warnings ?? []), ...(report.alignment?.warning ? [report.alignment.warning] : [])];
  const heat = display.colorMode === 'scalar' && display.scalar?.name === 'deviation';
  const covered = report.coverage?.covered_area_pct;

  const showHeatmap = () => {
    useStore.setState({ visible: [asset.id], activeId: asset.id });
    useStore.getState().setDisplay({ colorMode: 'scalar', scalar: { name: 'deviation', style: display.scalar?.name === 'deviation' ? display.scalar.style : { kind: 'diverging', min: -tol * 4, max: tol * 4, tolerance: tol, steps: 0 } } });
    window.setTimeout(() => getViewer()?.fit(), 60);
  };
  const showCoverage = () => {
    if (!coverageAsset) return;
    const thr = Number(report.coverage?.threshold ?? tol * 3);
    useStore.setState({ visible: [coverageAsset.id], activeId: coverageAsset.id });
    useStore.getState().setDisplay({ colorMode: 'scalar', scalar: { name: 'reference_distance', style: { kind: 'sequential', min: 0, max: thr * 3, tolerance: 0, steps: 6 } } });
    window.setTimeout(() => getViewer()?.fit(), 60);
  };
  const Icon = pass ? CheckCircle2 : warn ? TriangleAlert : XCircle;

  return (
    <div className="meas-cad">
      <div className={`verdict verdict-${pass ? 'pass' : warn ? 'warn' : 'fail'}`} role="status">
        <Icon size={22} aria-hidden />
        <div>
          <div className="verdict-title">{pass ? 'Within tolerance' : warn ? 'Mostly within tolerance' : 'Out of tolerance'}</div>
          <div className="verdict-sub">
            <b className="mono">{within.toFixed(1)} %</b> of <span className="mono">{Number(st.points ?? 0).toLocaleString()}</span> points lie within ±<span className="mono">{fmtLen(tol, 3)}</span> {units} of the CAD surface
          </div>
        </div>
      </div>

      <div className="row wrap">
        <Button size="sm" variant={heat ? 'primary' : 'secondary'} icon={<Gauge size={14} />} aria-pressed={heat} onClick={showHeatmap}>Deviation map</Button>
        <Button size="sm" icon={<Layers size={14} />} disabled={!coverageAsset} onClick={showCoverage}>Coverage on CAD</Button>
        <Button size="sm" variant="ghost" icon={<Eye size={14} />} onClick={() => useStore.setState(s => ({ visible: [...new Set([...s.visible, asset.id, refId].filter(id => byId.has(id)))] }))}>Show the CAD too</Button>
        <a className="btn btn-ghost btn-sm" href={`/api/assets/${asset.id}/inspection-report`} target="_blank" rel="noreferrer"><ExternalLink size={13} aria-hidden /><span className="btn-label">Report</span></a>
      </div>

      <div className="stat-grid">
        <Stat label="Average" value={fmtSigned(st.mean, 3)} unit={units} sub="+ extra material" />
        <Stat label="Spread (σ)" value={fmtLen(st.std, 3)} unit={units} />
        <Stat label="RMS" value={fmtLen(st.rms, 3)} unit={units} />
        <Stat label="5 % below" value={fmtSigned(st.p05, 3)} unit={units} />
        <Stat label="95 % below" value={fmtSigned(st.p95, 3)} unit={units} />
        <Stat label="Average gap" value={fmtLen(st.abs_mean, 3)} unit={units} />
        <Stat label="Lowest" value={fmtSigned(st.min, 3)} unit={units} />
        <Stat label="Highest" value={fmtSigned(st.max, 3)} unit={units} />
        <Stat label="CAD covered" value={covered != null ? `${Number(covered).toFixed(1)} %` : '–'} tone={covered != null && covered < 60 ? 'warning' : undefined} sub={covered != null && covered < 60 ? 'parts were not scanned' : undefined} />
      </div>

      {report.histogram && (
        <Block title="How the deviation spreads">
          <Histogram edges={report.histogram.edges} counts={report.histogram.counts} tolerance={tol} unit={units} />
          <p className="meas-legend caption">
            <span className="swatch neg" /> material missing <span className="swatch mid" /> within ±{fmtLen(tol, 3)} <span className="swatch pos" /> extra material
          </p>
        </Block>
      )}

      {dims.reference && (
        <Block title="Size in the CAD's axes">
          <div className="meas-table-wrap">
            <table className="data-table meas-table">
              <thead><tr><th /><th>CAD</th><th>Scan</th><th>Difference</th></tr></thead>
              <tbody>
                {['X', 'Y', 'Z'].map((ax, i) => {
                  const d = dims.difference?.[i] ?? (dims.scan?.[i] ?? 0) - (dims.reference?.[i] ?? 0);
                  return (
                    <tr key={ax}>
                      <td className={`axis-${ax.toLowerCase()} strong`}>{ax}</td>
                      <td className="mono">{fmtLen(dims.reference[i], 3)}</td>
                      <td className="mono">{fmtLen(dims.scan?.[i], 3)}</td>
                      <td className={`mono ${Math.abs(d) > tol ? 'delta-warn' : ''}`}>{fmtSigned(d, 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="caption">Outer extents of the scanned points. A face that was not scanned makes the scan look smaller along that axis.</p>
        </Block>
      )}

      <Block title="Line-up and scale">
        <table className="kv">
          <tbody>
            <tr><td>Method</td><td>{report.alignment?.method ?? '–'}</td></tr>
            <tr><td>Points on the surface after line-up</td><td className="mono">{report.alignment?.fitness != null ? fmtPct(report.alignment.fitness) : '–'}</td></tr>
            <tr><td>Line-up error (RMSE)</td><td className="mono">{fmtLen(report.alignment?.rmse, 4)} {units}</td></tr>
            <tr><td>Left out as not part</td><td className="mono">{fmtCount(Number(st.excluded ?? 0))} pts</td></tr>
            {report.scale_estimate && (
              <tr>
                <td>Best-fit scale (not applied)</td>
                <td className="mono meas-nowrap">
                  {Number(report.scale_estimate.factor).toFixed(5)} · {fmtSigned(report.scale_estimate.percent, 3)} %
                  {Math.abs(report.scale_estimate.percent) > 0.1 && <> <Badge tone="warning" title="The scan is larger or smaller than the CAD throughout. Check the scanner calibration (Accuracy tab).">check calibration</Badge></>}
                </td>
              </tr>
            )}
          </tbody>
        </table>
        {warnings.map((w, i) => <p key={i} className="warn-text"><TriangleAlert size={13} aria-hidden /> <span>{w}</span></p>)}
        <p className="caption">Scan: {byId.get(scanId)?.name ?? '–'} · CAD: {byId.get(refId)?.name ?? '–'}</p>
      </Block>
    </div>
  );
}
