import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown, CheckCircle2, ChevronDown, Crosshair, Eye, ExternalLink, ImagePlus, RefreshCw, ScanLine, TriangleAlert, Upload, X } from 'lucide-react';
import { create } from 'zustand';
import { api } from '../../lib/api';
import type { Asset } from '../../lib/types';
import { submitJob } from '../../lib/jobs';
import { fmtLen, fmtSigned } from '../../lib/format';
import { pickFiles } from '../../lib/importing';
import { goldenGrade, regionKey, useGoldenPins, type GoldenMeasurement, type GoldenRegion, type GoldenReport } from '../../lib/golden';
import { useProjectAssets, useStore } from '../../store';
import { getViewer } from '../../viewer/instance';
import { Button, Empty, Field, NumberInput, Segmented, Select } from '../../ui/primitives';
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

/** Start a check (the setup form, and "Check again" on an older result). */
async function runCheck(body: { scan_id: string; golden_id: string; tolerance: number; align: Align }, done?: () => void) {
  const up_axis = useStore.getState().display.upAxis;
  const job = await submitJob('/api/golden-check', { ...body, remember: true, up_axis }, () => done?.());
  if (!job) done?.();
  return job;
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
  // a result is open: running again is the second thing to do, not the first
  const showing = useStore(s => {
    const a = s.activeId ? s.byId.get(s.activeId) : undefined;
    return a?.operation === 'golden_check' || a?.operation === 'compare';
  });
  const ready = !!scanId && !!goldenId && scanId !== goldenId;
  const run = async () => {
    useGolden.setState({ busy: true });
    await runCheck({ scan_id: scanId, golden_id: goldenId, tolerance, align }, () => useGolden.setState({ busy: false }));
  };
  return (
    <Button variant={showing ? 'secondary' : 'primary'} size="lg" block icon={<CadGlyph size={18} />} loading={busy} disabled={!ready} onClick={run}>
      {!scanId ? 'Choose a scan' : !goldenId ? 'Add the golden model' : showing ? 'Run the check again' : 'Check against the golden model'}
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

/* ---------------------------------------------------------------- the result */

/** Plain words for the colours of the check (the legend keys of cloudclean/golden.py). */
const SHARE_LABEL: Record<string, string> = {
  good: 'Matches',
  missing: 'Not scanned',
  thin: 'Too few points',
  rough: 'Rough: points scatter',
  off_out: 'More material',
  off_in: 'Less material',
};

const AREA_STATUS: Record<string, string> = {
  missing: 'Not scanned',
  thin: 'Too few points',
  rough: 'Rough',
  off_out: 'More material',
  off_in: 'Less material',
};

const MEAS_STATUS: Record<GoldenMeasurement['status'], { label: string; tone: string }> = {
  off: { label: 'Off', tone: 'fail' },
  close: { label: 'Too close to call', tone: 'warn' },
  ok: { label: 'Matches', tone: 'pass' },
  not_measured: { label: 'Not measured', tone: 'none' },
};

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const pct = (v: number) => (v > 0 && v < 0.1 ? '<0.1' : v > 99.9 && v < 100 ? '99.9' : v.toFixed(1));

/** The share of the golden surface that matches, as a ring. */
function MatchRing({ value, tone }: { value: number | null; tone: string }) {
  const r = 34;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(100, value ?? 0));
  return (
    <div className={`gold-ring gold-ring-${tone}`} aria-hidden>
      <svg viewBox="0 0 84 84" width="84" height="84">
        <circle cx="42" cy="42" r={r} className="gold-ring-track" />
        <circle cx="42" cy="42" r={r} className="gold-ring-value" strokeDasharray={`${(c * v) / 100} ${c}`} transform="rotate(-90 42 42)" />
      </svg>
      <div className="gold-ring-text">
        <b>{value == null ? '–' : value >= 99.95 ? '100' : value.toFixed(value >= 10 ? 0 : 1)}<small>%</small></b>
        <span>match</span>
      </div>
    </div>
  );
}

/** Where a deviation sits against the tolerance: the green band is ±tolerance. */
function DeviationMeter({ value, tol, units }: { value: number; tol: number; units: string }) {
  const span = Math.max(4 * tol, Math.abs(value) * 1.25);
  const at = (x: number) => `${((x + span) / (2 * span)) * 100}%`;
  return (
    <div className="gold-meter" role="img" aria-label={`${fmtSigned(value, 2)} ${units}, tolerance ±${fmtLen(tol)} ${units}`}>
      <div className="gold-meter-track">
        <span className="gold-meter-band" style={{ left: at(-tol), width: `${(tol / span) * 100}%` }} />
        <span className="gold-meter-zero" style={{ left: at(0) }} />
        <span className={`gold-meter-mark ${value < 0 ? 'is-in' : 'is-out'}`} style={{ left: at(value) }} />
      </div>
      <div className="gold-meter-labels">
        <span>less material</span>
        <span className="mono">±{+tol.toFixed(4)} ok</span>
        <span>more material</span>
      </div>
    </div>
  );
}

function GoldenResults({ check }: { check: Asset }) {
  const units = useStore(s => s.display.units);
  const display = useStore(s => s.display);
  const byId = useStore(s => s.byId);
  const [report, setReport] = useState<GoldenReport | null>(null);
  const [failed, setFailed] = useState(false);
  const active = useGoldenPins(s => s.active);
  const picked = useGoldenPins(s => s.picked);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let alive = true;
    setReport(null);
    setFailed(false);
    api.asset(check.id).then(r => alive && setReport(r.report as GoldenReport)).catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [check.id]);

  const tol = Number(report?.tolerance ?? 0.1);
  const colour = (r: GoldenRegion) => report?.legend.find(l => l.key === regionKey(r))?.color ?? 'var(--ink-3)';
  // one numbering for the list and the pins: areas that differ first (they decide the verdict), then what to scan again
  const ordered = useMemo(() => {
    if (!report) return [] as GoldenRegion[];
    const off = report.regions.filter(r => r.kind === 'off').sort((a, b) => Math.abs(b.deviation ?? 0) * b.area_mm2 - Math.abs(a.deviation ?? 0) * a.area_mm2);
    return [...off, ...report.regions.filter(r => r.rescan)];
  }, [report]);
  const numberOf = (r: GoldenRegion) => ordered.indexOf(r) + 1;

  // the pins on the 3D view, while this result is on screen
  useEffect(() => {
    if (!report) return;
    useGoldenPins.setState({
      checkId: check.id,
      active: null,
      pins: ordered.map((r, i) => ({ n: i + 1, region: r.id, pos: r.pin ?? r.view.target, normal: r.normal, color: colour(r), label: r.name })),
    });
    return () => {
      useGoldenPins.setState({ checkId: null, pins: [], active: null });
      getViewer()?.spotlight(null);
    };
  }, [report, check.id]);

  const focus = (r: GoldenRegion) => {
    const st = useStore.getState();
    const keepColours = st.display.colorMode === 'scalar' && st.display.scalar?.name === 'golden_deviation';
    useStore.setState({ visible: [check.id], activeId: check.id });
    if (!keepColours) st.setDisplay({ colorMode: 'original', scalar: null });
    useGoldenPins.setState({ active: r.id });
    window.setTimeout(() => {
      const v = getViewer();
      v?.viewFrom(r.view.target, r.view.from, 650, r.view.radius);
      // older checks coloured the raw CAD mesh, where an area can have hardly any vertices of its own: no spotlight
      if ((report?.version ?? 1) >= 2) v?.spotlight(check.id, 'check_region', r.id).then(n => {
          if (n === 0) v.spotlight(null);
        });
    }, 60);
  };
  const unfocus = () => {
    useGoldenPins.setState({ active: null });
    getViewer()?.spotlight(null);
    window.setTimeout(() => getViewer()?.fit([check.id]), 30);
  };

  // a pin was clicked on the 3D view: show that area and bring its card into view
  useEffect(() => {
    if (!picked || !report) return;
    const r = report.regions.find(x => x.id === picked.region);
    if (!r) return;
    focus(r);
    listRef.current?.querySelector(`[data-area="${r.id}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [picked?.at]);

  if (failed) return <p className="err-text"><TriangleAlert size={14} aria-hidden /> The result of this check could not be loaded.</p>;
  if (!report) return <p className="hint-text">Loading the result…</p>;

  const grade = goldenGrade(report);
  const compareId = report.compare_asset?.id && byId.has(report.compare_asset.id) ? report.compare_asset.id : undefined;
  const scanId = check.parents[0];
  const view: View | null =
    display.colorMode === 'original' && useStore.getState().activeId === check.id ? 'problems'
      : display.colorMode === 'scalar' && display.scalar?.name === 'golden_deviation' ? 'golden'
        : display.colorMode === 'scalar' && display.scalar?.name === 'deviation' ? 'scan' : null;
  const diverging = (name: string) => ({ colorMode: 'scalar' as const, scalar: { name, style: { kind: 'diverging' as const, min: -tol * 4, max: tol * 4, tolerance: tol, steps: 0 } } });
  const show = (v: View) => {
    useGoldenPins.setState({ active: null });
    getViewer()?.spotlight(null);
    if (v === 'scan' && compareId) {
      useStore.setState({ visible: [compareId], activeId: compareId });
      useStore.getState().setDisplay(diverging('deviation'));
    } else {
      useStore.setState({ visible: [check.id], activeId: check.id });
      useStore.getState().setDisplay(v === 'golden' ? diverging('golden_deviation') : { colorMode: 'original', scalar: null });
    }
    window.setTimeout(() => getViewer()?.fit(), 60);
  };

  const differs = ordered.filter(r => r.kind === 'off');
  const rescan = ordered.filter(r => r.rescan);
  const ms = report.measurements;
  const count = (s: GoldenMeasurement['status']) => ms.filter(m => m.status === s).length;
  const measured = ms.length - count('not_measured');
  const shares = report.surface.shares_pct;
  const matchPct = grade.matchPct ?? 0;
  const tolText = `±${+tol.toFixed(4)} ${units}`;

  const problems: string[] = [];
  if (count('off')) problems.push(`${plural(count('off'), 'size')} off`);
  if (differs.length) problems.push(`${plural(differs.length, 'area')} off`);
  const sub =
    grade.key === 'match' ? `Every size and the whole scanned surface are within ${tolText} of the golden model.`
      : grade.key === 'incomplete'
        ? rescan.length
          ? `Everything that was scanned is within ${tolText}, but ${plural(rescan.length, 'area needs', 'areas need')} scanning again.`
          : `Everything that was scanned is within ${tolText}, but ${plural(count('not_measured'), 'size')} could not be measured.`
        : `${pct(matchPct)} % of the surface is within ${tolText}. ${problems.length ? `${problems.join(' and ')}: see below.` : ''}`;
  const notes = report.summary.filter(s => !/ of the golden surface was scanned| match, .* off, .* not measured/.test(s));
  const legacy = (report.version ?? 1) < 2;

  const area = (r: GoldenRegion) => {
    const key = regionKey(r);
    const on = active === r.id;
    return (
      <li key={r.id} data-area={r.id} className={`gold-card ${on ? 'is-on' : ''}`}>
        <div className="gold-card-head">
          <span className="gold-pin" style={{ '--pin': colour(r) } as React.CSSProperties} aria-label={`Area ${numberOf(r)}`}>{numberOf(r)}</span>
          <div className="gold-card-title">
            <h4>{r.name}</h4>
            <div className="gold-card-tags">
              <span className={`gold-tag gold-tag-${key}`}>
                <i style={{ background: colour(r) }} aria-hidden />
                {AREA_STATUS[key]}
                {r.kind === 'off' && r.deviation != null && <b className="mono">{fmtSigned(r.deviation, 2)} {units}</b>}
                {r.kind === 'rough' && r.spread != null && <b className="mono">±{fmtLen(r.spread, 2)} {units}</b>}
              </span>
              <span className="gold-card-size mono">{r.area_mm2 >= 10 ? r.area_mm2.toFixed(0) : r.area_mm2.toFixed(1)} mm² · {pct(r.share_pct)} % of the part</span>
            </div>
          </div>
        </div>
        {r.kind === 'off' && r.deviation != null && <DeviationMeter value={r.deviation} tol={tol} units={units} />}
        <p className="gold-card-why">{r.why}</p>
        <p className="gold-card-todo"><span>What to do</span>{r.advice}</p>
        <div className="gold-card-actions">
          {on ? (
            <>
              <span className="gold-showing"><Eye size={15} aria-hidden /> Showing it in the 3D view</span>
              <Button size="sm" variant="ghost" icon={<X size={14} />} onClick={unfocus}>Whole part</Button>
            </>
          ) : (
            <Button size="sm" icon={<Crosshair size={15} />} onClick={() => focus(r)} data-guide={numberOf(r) === 1 ? 'golden.show-me' : undefined}>Show me</Button>
          )}
        </div>
      </li>
    );
  };

  const measRow = (m: GoldenMeasurement) => {
    const region = m.region != null ? report.regions[m.region] : undefined;
    const st = MEAS_STATUS[m.status];
    const diff = Math.abs(m.difference ?? 0) < 0.0005 ? 0 : m.difference;
    return (
      <li key={m.id} className={`gold-row gold-row-${m.status}`}>
        <div className="gold-row-main">
          <span className="gold-row-name">{m.name}</span>
          {m.status === 'not_measured' ? (
            <span className="gold-row-note">
              {m.reason}
              {region && <> <button type="button" className="link" onClick={() => focus(region)}>Show the area</button></>}
            </span>
          ) : m.kind === 'position' ? (
            <span className="gold-row-vals">{(m.scan ?? 0) < 0.0005 ? 'Exactly where the golden model has it' : <>Moved <b className="mono">{fmtLen(m.scan, 3)} {units}</b>{m.toward ? ` ${m.toward.startsWith('to ') || /^(up|down|sideways)/.test(m.toward) ? m.toward : `toward ${m.toward}`}` : ''}</>}</span>
          ) : (
            <span className="gold-row-vals">
              Golden <b className="mono">{fmtLen(m.golden, 3)}</b> <span aria-hidden>→</span> scan <b className="mono">{fmtLen(m.scan, 3)}</b> {units}
            </span>
          )}
        </div>
        {m.status !== 'not_measured' ? (
          <span className={`gold-diff gold-diff-${st.tone}`} title={`${st.label}${m.uncertainty != null ? ` · known to ±${fmtLen(m.uncertainty, 3)} ${units}` : ''}`}>
            {m.kind === 'position' ? fmtLen(m.scan, 2) : fmtSigned(diff, 2)}
            <small>{st.label}</small>
          </span>
        ) : (
          <span className="gold-diff gold-diff-none"><small>{st.label}</small></span>
        )}
      </li>
    );
  };
  const order: GoldenMeasurement['status'][] = ['off', 'close', 'ok'];
  const shown = order.flatMap(s => ms.filter(m => m.status === s));
  const notMeasured = ms.filter(m => m.status === 'not_measured');
  const scrollTo = (id: string) => listRef.current?.querySelector(`#${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  return (
    <div className="gold" ref={listRef}>
      <section className={`gold-hero gold-hero-${grade.tone}`} role="status" aria-label={`${grade.label}. ${sub}`} data-guide="golden.verdict">
        <MatchRing value={grade.matchPct} tone={grade.tone} />
        <div className="gold-hero-text">
          <div className="gold-hero-kicker">{grade.tone === 'pass' ? <CheckCircle2 size={15} aria-hidden /> : <TriangleAlert size={15} aria-hidden />} Golden model check</div>
          <h3 className="gold-hero-title">{grade.label}</h3>
          <p className="gold-hero-sub">{sub}</p>
        </div>
      </section>

      <div className="gold-glance">
        <button type="button" className="gold-stat" onClick={() => scrollTo('gold-surface')}>
          <span className="gold-stat-value mono">{pct(report.surface.scanned_pct)}<small>%</small></span>
          <span className="gold-stat-label">of the part scanned</span>
        </button>
        <button type="button" className={`gold-stat ${count('off') ? 'is-fail' : ''}`} onClick={() => scrollTo('gold-sizes')} disabled={!ms.length}>
          <span className="gold-stat-value mono">{count('ok')}<small>/{measured}</small></span>
          <span className="gold-stat-label">{measured ? 'sizes match' : 'no sizes to measure'}</span>
        </button>
        <button type="button" className={`gold-stat ${differs.length ? 'is-fail' : ''}`} onClick={() => scrollTo('gold-areas')} disabled={!ordered.length}>
          <span className="gold-stat-value mono">{ordered.length}</span>
          <span className="gold-stat-label">{ordered.length === 1 ? 'area to look at' : 'areas to look at'}</span>
        </button>
      </div>

      {legacy && (
        <div className="gold-legacy">
          <p>This check was made by an older version: its areas are named by coordinates and “Show me” can miss them. Run it again for plain names, numbered pins and a sharper colour map.</p>
          <LegacyRerun check={check} report={report} />
        </div>
      )}

      <section className="gold-section" id="gold-surface">
        <div className="gold-section-head">
          <h3>The surface</h3>
          <a className="gold-report-link" href={`/api/assets/${check.id}/golden-report`} target="_blank" rel="noreferrer"><ExternalLink size={14} aria-hidden /> Printable report</a>
        </div>
        <div className="gold-bar" role="img" aria-label={Object.entries(shares).filter(([, v]) => v > 0).map(([k, v]) => `${SHARE_LABEL[k] ?? k} ${pct(v)} %`).join(', ')}>
          {report.legend.map(l => ((shares[l.key] ?? 0) > 0 ? <span key={l.key} style={{ flexGrow: shares[l.key], background: l.color }} title={`${SHARE_LABEL[l.key] ?? l.label}: ${pct(shares[l.key])} %`} /> : null))}
        </div>
        <ul className="gold-shares">
          {report.legend.filter(l => (shares[l.key] ?? 0) > 0 || l.key === 'good').map(l => (
            <li key={l.key}>
              <i style={{ background: l.color }} aria-hidden />
              <span>{SHARE_LABEL[l.key] ?? l.label}</span>
              <b className="mono">{pct(shares[l.key] ?? 0)} %</b>
            </li>
          ))}
        </ul>
        <div className="gold-views" data-guide="golden.colour-by">
          <span className="gold-views-label">Colour the model by</span>
          <Segmented
            size="sm"
            ariaLabel="Colour the model by"
            value={view ?? 'problems'}
            onChange={show}
            options={[
              { value: 'problems', label: 'What was found', title: 'The golden model coloured by what the check found (the colours above)' },
              { value: 'golden', label: 'Distance', title: 'The golden model coloured by how far the scan sits from it: blue less material, red more' },
              ...(compareId ? [{ value: 'scan' as View, label: 'Scan points', title: 'Every scan point coloured by its distance to the golden model' }] : []),
            ]}
          />
        </div>
      </section>

      {ordered.length > 0 && (
        <section className="gold-section" id="gold-areas" data-guide="golden.areas">
          <div className="gold-section-head">
            <h3>Areas to look at</h3>
            <span className="gold-section-note">The numbers match the pins on the model</span>
          </div>
          {differs.length > 0 && (
            <>
              <div className="gold-group"><span className="gold-group-dot is-off" aria-hidden />Different from the golden model <span className="mono">{differs.length}</span></div>
              <ol className="gold-cards">{differs.map(area)}</ol>
            </>
          )}
          {rescan.length > 0 && (
            <>
              <div className="gold-group"><ScanLine size={15} aria-hidden />Scan these again <span className="mono">{rescan.length}</span></div>
              <ol className="gold-cards">{rescan.map(area)}</ol>
              {rescan.some(r => r.kind === 'missing' || r.kind === 'thin') && <FillFromPhotos scanId={scanId} />}
            </>
          )}
        </section>
      )}

      <section className="gold-section" id="gold-sizes">
        <div className="gold-section-head">
          <h3>Sizes</h3>
          <span className="gold-section-note">golden → scan · difference</span>
        </div>
        {ms.length ? (
          <>
            {shown.length > 0 && <ul className="gold-rows">{shown.map(measRow)}</ul>}
            {notMeasured.length > 0 && (
              <details className="gold-more">
                <summary>
                  <ArrowDown size={14} aria-hidden /> {plural(notMeasured.length, 'size')} could not be measured
                </summary>
                <ul className="gold-rows">{notMeasured.map(measRow)}</ul>
              </details>
            )}
          </>
        ) : (
          <p className="hint-text">The golden model has no flat or round faces to measure: use the colours and the areas above.</p>
        )}
        <p className="gold-fine">A size matches when it is within {tolText}, allowing for how precisely the scan pins it down. Positions are measured after lining the scan up with the golden model.</p>
      </section>

      {(notes.length > 0 || (report.warnings ?? []).length > 0) && (
        <details className="gold-more gold-notes">
          <summary><ArrowDown size={14} aria-hidden /> Notes on this check ({notes.length + (report.warnings ?? []).length})</summary>
          <ul>
            {notes.map((s, i) => <li key={`n${i}`}>{s}</li>)}
            {(report.warnings ?? []).map((w, i) => <li key={`w${i}`} className="is-warn">{w}</li>)}
          </ul>
        </details>
      )}

      {compareId && byId.get(compareId) && (
        <details className="disclosure">
          <summary>Deviation details <span className="sub">every scan point</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
          <div className="disclosure-body"><CadResults asset={byId.get(compareId)!} /></div>
        </details>
      )}
      <p className="gold-fine">Scan: {byId.get(scanId)?.name ?? '–'} · Golden model: {byId.get(check.parents[1])?.name ?? '–'}</p>
    </div>
  );
}

/** Run an older check again with the same scan, golden model and settings. */
function LegacyRerun({ check, report }: { check: Asset; report: GoldenReport }) {
  const [busy, setBusy] = useState(false);
  const [scanId, goldenId] = check.parents;
  const ok = !!useStore(s => s.byId.get(scanId)) && !!useStore(s => s.byId.get(goldenId));
  const align = ((check.params?.align as Align | undefined) ?? 'auto') as Align;
  return (
    <Button size="sm" icon={<RefreshCw size={14} />} loading={busy} disabled={!ok} onClick={async () => {
      setBusy(true);
      await runCheck({ scan_id: scanId, golden_id: goldenId, tolerance: Number(report.tolerance ?? 0.1), align }, () => setBusy(false));
    }}>
      {ok ? 'Run the check again' : 'The scan or golden model was deleted'}
    </Button>
  );
}
