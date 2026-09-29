import { useEffect, useMemo, useState } from 'react';
import { CheckCircle2, ChevronDown, Crosshair, ExternalLink, ImagePlus, TriangleAlert, Upload, XCircle } from 'lucide-react';
import { create } from 'zustand';
import { api } from '../../lib/api';
import type { Asset } from '../../lib/types';
import { submitJob } from '../../lib/jobs';
import { fmtLen, fmtSigned } from '../../lib/format';
import { pickFiles } from '../../lib/importing';
import { useProjectAssets, useStore } from '../../store';
import { getViewer } from '../../viewer/instance';
import { Badge, Button, Empty, Field, NumberInput, Segmented, Select } from '../../ui/primitives';
import { Block } from '../StepFrame';
import { CadResults, isCad } from './CadCompare';
import { CadGlyph } from './glyphs';

/* Measure -> Golden model (docs/golden-model.md): the scan against the part as it should be. */

type Align = 'auto' | 'icp' | 'none';
interface Setup {
  scanId: string;
  goldenId: string;
  tolerance: number;
  align: Align;
  busy: boolean;
}
const useGolden = create<Setup>(() => ({ scanId: '', goldenId: '', tolerance: 0.1, align: 'auto', busy: false }));

type Vec3 = [number, number, number];
interface Region {
  id: number;
  kind: 'missing' | 'thin' | 'rough' | 'off';
  sign: number;
  name: string;
  why: string;
  advice: string;
  rescan: boolean;
  area_mm2: number;
  share_pct: number;
  deviation: number | null;
  spread: number | null;
  view: { target: Vec3; from: Vec3 };
}
interface Measurement {
  id: number;
  kind: string;
  name: string;
  golden: number | null;
  scan: number | null;
  difference: number | null;
  uncertainty: number | null;
  status: 'ok' | 'off' | 'close' | 'not_measured';
  reason?: string;
  region?: number | null;
  toward?: string | null;
}
interface Report {
  verdict: 'match' | 'differs' | 'incomplete';
  headline: string;
  summary: string[];
  tolerance: number;
  regions: Region[];
  measurements: Measurement[];
  surface: { scanned_pct: number; shares_pct: Record<string, number> };
  legend: { code: number; key: string; label: string; color: string }[];
  compare_asset?: { id: string; name: string };
  warnings?: string[];
}

const KIND_LABEL: Record<Region['kind'], string> = { missing: 'Not scanned', thin: 'Too few points', rough: 'Rough', off: 'Off' };
const STATUS: Record<Measurement['status'], { label: string; tone: 'ok' | 'critical' | 'warning' | 'neutral' }> = {
  ok: { label: 'Matches', tone: 'ok' },
  off: { label: 'Off', tone: 'critical' },
  close: { label: 'Too close to call', tone: 'warning' },
  not_measured: { label: 'Not measured', tone: 'neutral' },
};
const ALIGN_HELP: Record<Align, string> = {
  auto: 'Finds how the scan sits on the golden model by itself.',
  icp: 'Only fine-tunes: use when the scan is already roughly in place.',
  none: 'Checks them as they are: the scan is already in the golden model’s coordinates.',
};

const isResult = (a: Asset) => a.operation === 'compare' || a.operation === 'golden_check';

/** Golden models (meshes, CAD first) and scans of this project (of the whole workspace when the project has none). */
function useChoices() {
  const project = useProjectAssets();
  const all = useStore(s => s.assets);
  return useMemo(() => {
    const goldens = (list: Asset[]) => list.filter(a => a.kind === 'mesh' && !isResult(a)).sort((a, b) => Number(isCad(b)) - Number(isCad(a)) || b.created.localeCompare(a.created));
    const scanList = (list: Asset[]) => list.filter(a => a.kind !== 'image' && !isResult(a) && !isCad(a)).sort((a, b) => b.created.localeCompare(a.created));
    return { goldens: goldens(project).length ? goldens(project) : goldens(all), scans: scanList(project).length ? scanList(project) : scanList(all) };
  }, [project, all]);
}

function useProjectGolden(): string | null {
  return useStore(s => s.projects.find(p => p.id === s.projectId)?.golden_asset_id ?? null);
}

/** The check that belongs to the selected asset: the check's own mesh, or the compared scan it made. */
function useCheckFor(active: Asset | undefined): Asset | undefined {
  const assets = useStore(s => s.assets);
  if (!active) return undefined;
  if (active.operation === 'golden_check') return active;
  if (active.operation !== 'compare') return undefined;
  return assets.filter(a => a.operation === 'golden_check' && a.parents.join() === active.parents.join() && a.created >= active.created).sort((a, b) => a.created.localeCompare(b.created))[0];
}

export function GoldenCheck() {
  const activeId = useStore(s => s.activeId);
  const byId = useStore(s => s.byId);
  const units = useStore(s => s.display.units);
  const setup = useGolden();
  const projectGolden = useProjectGolden();
  const { goldens, scans } = useChoices();
  const active = activeId ? byId.get(activeId) : undefined;
  const check = useCheckFor(active);
  const oldCompare = !check && active?.operation === 'compare' && active.kind === 'pointcloud' ? active : undefined;

  // sensible picks: the selected scan, and the project's golden model (else the newest CAD import)
  const key = `${scans.map(s => s.id).join()}|${goldens.map(g => g.id).join()}|${projectGolden}`;
  useEffect(() => {
    const cur = useGolden.getState();
    const patch: Partial<Setup> = {};
    const activeIsScan = !!active && scans.some(s => s.id === active.id);
    if (activeIsScan && cur.scanId !== active!.id) patch.scanId = active!.id;
    else if (!scans.some(s => s.id === cur.scanId) && scans[0]) patch.scanId = scans[0].id;
    if (projectGolden && goldens.some(g => g.id === projectGolden) && cur.goldenId !== projectGolden && !cur.goldenId) patch.goldenId = projectGolden;
    else if (!goldens.some(g => g.id === cur.goldenId) && goldens[0]) patch.goldenId = goldens.find(g => g.id === projectGolden)?.id ?? goldens[0].id;
    if (Object.keys(patch).length) useGolden.setState(patch);
  }, [active?.id, key]);

  if (!scans.length && !check && !oldCompare) {
    return (
      <Empty icon={<CadGlyph size={24} />} title="Nothing to check yet">
        Scan the part, and bring in its golden model: the CAD file (STEP, IGES) or a trusted mesh (STL, OBJ, PLY).
      </Empty>
    );
  }
  const golden = goldens.find(g => g.id === setup.goldenId);
  const fromScan = !!golden && !isCad(golden) && golden.operation !== 'import';

  const form = (
    <div className="fields">
      <Field
        label="Golden model"
        htmlFor="gold-ref"
        inline={false}
        help={
          !goldens.length
            ? 'The part as it should be: bring in its CAD file (STEP, IGES) or a trusted mesh (STL, OBJ, PLY).'
            : fromScan
              ? 'This mesh was made from a scan: the check then compares two scans. Add the CAD file for a check against the design.'
              : setup.goldenId === projectGolden
                ? 'This project’s golden model. Every scan of the project is checked against it.'
                : 'It becomes this project’s golden model when you run the check.'
        }
      >
        {goldens.length ? (
          <div className="row">
            <div className="grow">
              <Select id="gold-ref" value={setup.goldenId} onChange={goldenId => useGolden.setState({ goldenId })} options={goldens.map(a => ({ value: a.id, label: `${a.id === projectGolden ? '★ ' : ''}${isCad(a) ? 'CAD · ' : ''}${a.name}` }))} />
            </div>
            <Button size="md" variant="ghost" icon={<Upload size={15} />} onClick={() => pickFiles('.step,.stp,.iges,.igs,.stl,.obj,.ply')}>Add</Button>
          </div>
        ) : (
          <Button icon={<Upload size={15} />} onClick={() => pickFiles('.step,.stp,.iges,.igs,.stl,.obj,.ply')}>Add the golden model</Button>
        )}
      </Field>
      <Field label="Scan" htmlFor="gold-scan" inline={false}>
        <Select id="gold-scan" value={setup.scanId} onChange={scanId => useGolden.setState({ scanId })} options={[{ value: '', label: 'Choose a scan…' }, ...scans.map(a => ({ value: a.id, label: `${a.name}${a.kind === 'mesh' ? ' (mesh)' : ''}` }))]} />
      </Field>
      <Field label="Tolerance ±" help="How far the scan may be from the golden model and still match.">
        <NumberInput value={setup.tolerance} step={0.01} min={0.001} unit={units} onChange={tolerance => useGolden.setState({ tolerance })} />
      </Field>
      <details className="disclosure">
        <summary>Line-up <span className="sub">{setup.align === 'auto' ? 'automatic' : setup.align === 'icp' ? 'fine-tune only' : 'as they are'}</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
        <div className="disclosure-body">
          <Field label="Line them up" inline={false} help={ALIGN_HELP[setup.align]}>
            <Segmented value={setup.align} onChange={align => useGolden.setState({ align })} options={[{ value: 'auto', label: 'Automatically' }, { value: 'icp', label: 'Fine-tune only' }, { value: 'none', label: 'As they are' }]} />
          </Field>
        </div>
      </details>
    </div>
  );

  return (
    <>
      {check && <GoldenResults check={check} />}
      {oldCompare && <CadResults asset={oldCompare} />}
      {check || oldCompare ? (
        <details className="disclosure">
          <summary>Check again <span className="sub">another scan or setting</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
          <div className="disclosure-body">{form}</div>
        </details>
      ) : (
        <Block title="Check against the golden model">
          <p className="hint-text">The golden model is the part as it should be. CloudClean lines your scan up with it and shows what was not scanned or is off, what to scan again, and every size of the golden model measured on your scan.</p>
          {form}
        </Block>
      )}
    </>
  );
}

export function GoldenFooter() {
  const { scanId, goldenId, tolerance, align, busy } = useGolden();
  const ready = !!scanId && !!goldenId && scanId !== goldenId;
  const run = async () => {
    useGolden.setState({ busy: true });
    const job = await submitJob('/api/golden-check', { scan_id: scanId, golden_id: goldenId, tolerance, align, remember: true }, () => useGolden.setState({ busy: false }));
    if (!job) useGolden.setState({ busy: false });
  };
  return (
    <Button variant="primary" size="lg" block icon={<CadGlyph size={18} />} loading={busy} disabled={!ready} onClick={run}>
      {!scanId ? 'Choose a scan' : !goldenId ? 'Add the golden model' : 'Check against the golden model'}
    </Button>
  );
}

type View = 'problems' | 'golden' | 'scan';

/** Can't scan it again? Fill the missing areas from photos of the part (cloudclean/photo_fill.py). */
function FillFromPhotos({ scanId }: { scanId: string }) {
  const photos = useProjectAssets().filter(a => a.kind === 'image').length;
  const scan = useStore(s => s.byId.get(scanId));
  const [busy, setBusy] = useState(false);
  if (!scan || scan.operation === 'photo_fill') return null;
  const fill = async () => {
    setBusy(true);
    const job = await submitJob('/api/fill-from-photos', { asset_id: scanId }, () => setBusy(false));
    if (!job) setBusy(false);
  };
  return (
    <div className="gold-fill">
      <p className="hint-text">
        <b>Can’t scan it again?</b>{' '}
        {photos >= 3
          ? `CloudClean can fill the missing areas from your ${photos} photos of the part. Photos are only good to about 1–2 mm, so filled areas make the model complete but are left out of this check.`
          : 'Add 12 or more photos of the part, all the way round (Scan → Photos), and CloudClean can fill the missing areas from them. Photos are only good to about 1–2 mm, so filled areas are left out of this check.'}
      </p>
      {photos >= 3 && <Button size="sm" icon={<ImagePlus size={14} />} loading={busy} onClick={fill}>Fill from photos</Button>}
    </div>
  );
}

function GoldenResults({ check }: { check: Asset }) {
  const units = useStore(s => s.display.units);
  const display = useStore(s => s.display);
  const byId = useStore(s => s.byId);
  const [report, setReport] = useState<Report | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let alive = true;
    setReport(null);
    setFailed(false);
    api.asset(check.id).then(r => alive && setReport(r.report as Report)).catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [check.id]);
  if (failed) return <p className="err-text"><TriangleAlert size={14} aria-hidden /> The result of this check could not be loaded.</p>;
  if (!report) return <p className="hint-text">Loading the result…</p>;

  const tol = Number(report.tolerance ?? 0.1);
  const compareId = report.compare_asset?.id && byId.has(report.compare_asset.id) ? report.compare_asset.id : undefined;
  const scanId = check.parents[0];
  const view: View | null =
    display.colorMode === 'original' && useStore.getState().activeId === check.id ? 'problems'
      : display.colorMode === 'scalar' && display.scalar?.name === 'golden_deviation' ? 'golden'
        : display.colorMode === 'scalar' && display.scalar?.name === 'deviation' ? 'scan' : null;
  const diverging = (name: string) => ({ colorMode: 'scalar' as const, scalar: { name, style: { kind: 'diverging' as const, min: -tol * 4, max: tol * 4, tolerance: tol, steps: 0 } } });
  const show = (v: View) => {
    if (v === 'scan' && compareId) {
      useStore.setState({ visible: [compareId], activeId: compareId });
      useStore.getState().setDisplay(diverging('deviation'));
    } else {
      useStore.setState({ visible: [check.id], activeId: check.id });
      useStore.getState().setDisplay(v === 'golden' ? diverging('golden_deviation') : { colorMode: 'original', scalar: null });
    }
    window.setTimeout(() => getViewer()?.fit(), 60);
  };
  const focus = (r: Region) => {
    useStore.setState({ visible: [check.id], activeId: check.id });
    useStore.getState().setDisplay({ colorMode: 'original', scalar: null });
    window.setTimeout(() => getViewer()?.viewFrom(r.view.target, r.view.from), 60);
  };
  const colour = (r: Region) => report.legend.find(l => l.key === (r.kind === 'off' ? (r.sign > 0 ? 'off_out' : 'off_in') : r.kind))?.color ?? 'var(--ink-3)';
  const rescan = report.regions.filter(r => r.rescan);
  const differs = report.regions.filter(r => r.kind === 'off');
  const Icon = report.verdict === 'match' ? CheckCircle2 : report.verdict === 'incomplete' ? TriangleAlert : XCircle;
  const tone = report.verdict === 'match' ? 'pass' : report.verdict === 'incomplete' ? 'warn' : 'fail';
  const compareAsset = compareId ? byId.get(compareId) : undefined;

  const area = (r: Region, n: number) => (
    <li key={r.id} className="gold-area">
      <span className="gold-num" style={{ borderColor: colour(r) }} aria-hidden>{n}</span>
      <div className="gold-area-head">
        <span className="gold-area-name">{r.name}</span>
        <Badge tone={r.kind === 'off' ? 'critical' : r.kind === 'missing' ? 'info' : 'warning'}>{KIND_LABEL[r.kind]}{r.kind === 'off' && r.deviation != null ? ` ${fmtSigned(r.deviation, 3)} ${units}` : ''}</Badge>
        <span className="gold-area-size">{r.area_mm2 >= 10 ? r.area_mm2.toFixed(0) : r.area_mm2.toFixed(1)} mm² · {r.share_pct.toFixed(1)} %</span>
      </div>
      <p className="gold-why">{r.why}</p>
      <p className="gold-advice">{r.advice}</p>
      <Button size="sm" variant="ghost" icon={<Crosshair size={14} />} onClick={() => focus(r)}>Show me</Button>
    </li>
  );

  return (
    <div className="meas-cad gold">
      <div className={`verdict verdict-${tone}`} role="status">
        <Icon size={22} aria-hidden />
        <div>
          <div className="verdict-title">{report.headline}</div>
          <ul className="gold-summary">{report.summary.map((s, i) => <li key={i}>{s}</li>)}</ul>
        </div>
      </div>

      <div className="gold-views">
        <Segmented
          size="sm"
          ariaLabel="What the colours show"
          value={view ?? 'problems'}
          onChange={show}
          options={[
            { value: 'problems', label: 'Problems', title: 'The golden model coloured by what the check found' },
            { value: 'golden', label: 'Deviation', title: 'The golden model coloured by how far the scan sits from it (blue: less material, red: more)' },
            ...(compareId ? [{ value: 'scan' as View, label: 'Scan points', title: 'Every scan point coloured by its distance to the golden model' }] : []),
          ]}
        />
        <a className="btn btn-ghost btn-sm" href={`/api/assets/${check.id}/golden-report`} target="_blank" rel="noreferrer"><ExternalLink size={13} aria-hidden /><span className="btn-label">Report</span></a>
      </div>
      <p className="gold-legend">
        {report.legend.map(l => (
          <span key={l.key}><i style={{ background: l.color }} aria-hidden />{l.label} <span className="mono">{(report.surface.shares_pct[l.key] ?? 0).toFixed(1)} %</span></span>
        ))}
      </p>

      {rescan.length > 0 && (
        <Block title={`Scan again (${rescan.length})`}>
          <ol className="gold-list">{rescan.map((r, i) => area(r, i + 1))}</ol>
          {rescan.some(r => r.kind === 'missing' || r.kind === 'thin') && <FillFromPhotos scanId={scanId} />}
        </Block>
      )}
      {differs.length > 0 && (
        <Block title={`Different from the golden model (${differs.length})`}>
          <ol className="gold-list">{differs.map((r, i) => area(r, rescan.length + i + 1))}</ol>
        </Block>
      )}

      <Block title="Measurements">
        {report.measurements.length ? (
          <ul className="gold-meas">
            {report.measurements.map(m => {
              const region = m.region != null ? report.regions[m.region] : undefined;
              return (
                <li key={m.id} className={`gold-m gold-m-${m.status}`}>
                  <span className="gold-m-name">{m.name}</span>
                  <Badge tone={STATUS[m.status].tone}>{STATUS[m.status].label}</Badge>
                  {m.status === 'not_measured' ? (
                    <span className="gold-m-note">
                      Golden <b className="mono">{m.golden != null ? fmtLen(m.golden, 3) : '–'}</b>. {m.reason}
                      {region && <> <button type="button" className="link" onClick={() => focus(region)}>Show the area</button></>}
                    </span>
                  ) : (
                    <span className="gold-m-vals">
                      <span>Golden <b className="mono">{fmtLen(m.golden, 3)}</b></span>
                      <span>Scan <b className="mono">{fmtLen(m.scan, 3)}</b></span>
                      <span title={m.uncertainty != null ? `known to ±${fmtLen(m.uncertainty, 3)} ${units}` : undefined}>
                        Difference <b className="mono">{fmtSigned(Math.abs(m.difference ?? 0) < 0.0005 ? 0 : m.difference, 3)}</b> {units}
                      </span>
                      {m.kind === 'position' && m.toward && (m.scan ?? 0) > 0.0005 && <span>moved toward {m.toward}</span>}
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="hint-text">The golden model has no flat or round faces to measure: use the colours and the areas above.</p>
        )}
        <p className="caption">A size matches when it is within ±{fmtLen(tol, 3)} {units}, allowing for how precisely the scan pins it down. Positions are measured after lining the scan up with the golden model.</p>
      </Block>

      {(report.warnings ?? []).map((w, i) => <p key={i} className="warn-text"><TriangleAlert size={13} aria-hidden /> <span>{w}</span></p>)}

      {compareAsset && (
        <details className="disclosure">
          <summary>Deviation details <span className="sub">every scan point</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
          <div className="disclosure-body"><CadResults asset={compareAsset} /></div>
        </details>
      )}
      <p className="caption">Scan: {byId.get(scanId)?.name ?? '–'} · Golden model: {byId.get(check.parents[1])?.name ?? '–'}</p>
    </div>
  );
}
