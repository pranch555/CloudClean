import { useEffect, useState, type ReactNode } from 'react';
import { create } from 'zustand';
import { CheckCircle2, ChevronDown, CircleHelp, Info, Loader2, RefreshCw, TriangleAlert, XCircle } from 'lucide-react';
import { clearSelection } from '../../lib/actions';
import { fmtCount, fmtLen, fmtPct, fmtSigned, OPERATION_LABEL } from '../../lib/format';
import { roleOf } from '../../lib/journey';
import {
  addDims, checkReference, compareScans, defaultTolerance, dims3, isOriginal, loadDrift, referenceLine, selectionRegion, takeSelection,
  Unavailable, useCap, useCompareScans, useDrifts, useSelectionOn, type CompareScansResult, type Direction, type DriftResult, type ReferenceResult,
  type ReferenceSpec, type RegionPick,
} from '../../lib/measureTools';
import type { Asset } from '../../lib/types';
import { useProjectAssets, useStore } from '../../store';
import { Badge, Button, Field, Metric, NumberInput, Progress, Segmented, Select } from '../../ui/primitives';
import { Block, ResultCard } from '../StepFrame';
import { AccuracyGlyph } from './glyphs';
import { Histogram } from './Histogram';
import { DIRS, fmtPpm } from './format';
import { NotHere, PickButtons, SelectionStatus, SelectStep, StepItem, StepList, SwitchRow } from './parts';

const SIZE_NAMES = ['Length', 'Width', 'Height'];

const MADE_FROM: Record<string, string> = {
  clean: 'Cleaned from',
  merge: 'Merged from',
  mesh: 'Meshed from',
  edit: 'Edited from',
  texture: 'Coloured from',
  autopilot: 'Made by Autopilot from',
};

/** Is it the scanner or the software? Three checks, written for anyone. */
export function AccuracyTab({ target }: { target: Asset | undefined }) {
  const drift = useDrifts(s => (target ? s.byId[target.id] : undefined));
  const run = useCompareScans(s => s.run);
  const ref = useRefCheck(s => s.result);
  const driftStatus = !target ? null : isOriginal(target) ? <Badge>original scan</Badge> : drift?.status === 'done' ? verdictBadge(drift.drift.verdict) : drift?.status === 'loading' ? <Loader2 size={15} className="spin muted" aria-label="checking" /> : null;
  const repeatStatus = run?.status === 'done' && run.result ? compareBadge(run.result.verdict) : run?.status === 'running' ? <Loader2 size={15} className="spin muted" aria-label="running" /> : null;
  const refStatus = ref && target && ref.assetId === target.id ? referenceBadge(ref.r.verdict) : null;

  return (
    <>
      <p className="hint-text">When a size looks a little off, these checks tell you whether the software or the scanner is responsible.</p>
      <CheckCard n={1} title="Did processing change the size?" tests="tests the software" status={driftStatus} defaultOpen>
        <DriftCheck target={target} />
      </CheckCard>
      <CheckCard n={2} guide="measure.repeatability" title="Is the scanner repeatable?" tests="tests the scanner" status={repeatStatus}>
        <RepeatCheck target={target} />
      </CheckCard>
      <CheckCard n={3} guide="measure.known-size" title="Check against a known size" tests="tests the true size" status={refStatus}>
        <ReferenceCheck target={target} />
      </CheckCard>
      <Explainer />
    </>
  );
}

function CheckCard({ n, title, tests, status, defaultOpen, children, guide }: { n: number; title: string; tests: string; status?: ReactNode; defaultOpen?: boolean; children: ReactNode; guide?: string }) {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <details className="disclosure acc-check" data-guide={guide} open={open} onToggle={e => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
      <summary>
        <span className="index-bubble">{n}</span>
        <span className="acc-check-title">
          <span>{title}</span>
          <span className="sub">{tests}</span>
        </span>
        <span className="acc-check-status">{status}</span>
        <ChevronDown size={16} className="chev" aria-hidden />
      </summary>
      {open && <div className="disclosure-body acc-check-body">{children}</div>}
    </details>
  );
}

function verdictBadge(v: string) {
  return v === 'unchanged' ? <Badge tone="ok"><CheckCircle2 size={12} aria-hidden /> unchanged</Badge> : v === 'moved' ? <Badge tone="ok"><CheckCircle2 size={12} aria-hidden /> moved only</Badge> : <Badge tone="warning"><TriangleAlert size={12} aria-hidden /> changed</Badge>;
}
function compareBadge(v: string) {
  return v === 'consistent' ? <Badge tone="ok"><CheckCircle2 size={12} aria-hidden /> agree</Badge> : v === 'differ' ? <Badge tone="warning"><TriangleAlert size={12} aria-hidden /> differ</Badge> : <Badge tone="info"><CircleHelp size={12} aria-hidden /> unsure</Badge>;
}
function referenceBadge(v: string) {
  return v === 'ok' ? <Badge tone="ok"><CheckCircle2 size={12} aria-hidden /> within</Badge> : v === 'marginal' ? <Badge tone="warning"><TriangleAlert size={12} aria-hidden /> close</Badge> : <Badge tone="danger"><XCircle size={12} aria-hidden /> outside</Badge>;
}

// ------------------------------------------------------------------------------------------------ 1. processing drift
function DriftCheck({ target }: { target: Asset | undefined }) {
  const cap = useCap('accuracy');
  const state = useDrifts(s => (target ? s.byId[target.id] : undefined));
  const byId = useStore(s => s.byId);
  const assets = useStore(s => s.assets);
  const original = !!target && isOriginal(target);
  useEffect(() => {
    if (target && !original && cap !== 'no') void loadDrift(target.id);
  }, [target?.id, cap, original]);

  if (!target) return <p className="hint-text">Click a model in the list on the left.</p>;
  if (original) {
    const children = assets.filter(a => a.parents.includes(target.id) && a.kind !== 'image');
    return (
      <div className="acc-note">
        <Info size={17} aria-hidden />
        <div className="stack tight">
          <span><b>{target.name}</b> is an original scan: nothing has processed it, so its size is exactly what the scanner delivered.</span>
          {children.length > 0 ? (
            <span className="acc-made">
              Made from it:{' '}
              {children.slice(0, 4).map(c => (
                <button key={c.id} type="button" className="chip" onClick={() => useStore.getState().activate(c.id)}>{c.name}</button>
              ))}
            </span>
          ) : (
            <span className="muted">Clean or mesh it, then check the result here.</span>
          )}
        </div>
      </div>
    );
  }
  if (cap === 'no' || state?.status === 'unavailable') return <NotHere what="Checking what processing did to the size" />;
  if (!state || state.status === 'loading') {
    return (
      <p className="acc-loading"><Loader2 size={15} className="spin" aria-hidden /> Comparing {target.name} with {byId.get(target.parents[0])?.name ?? 'the model it was made from'}…</p>
    );
  }
  if (state.status === 'error') {
    return (
      <p className="err-text" role="alert">
        <TriangleAlert size={14} aria-hidden />
        <span>{state.message} <button type="button" className="link" onClick={() => loadDrift(target.id, true)}>Try again</button></span>
      </p>
    );
  }
  return (
    <>
      <DriftView target={target} d={state.drift} />
      <Chain target={target} />
    </>
  );
}

function DriftView({ target, d }: { target: Asset; d: DriftResult }) {
  const units = useStore(s => s.display.units);
  const byId = useStore(s => s.byId);
  const parents = (target.operation === 'merge' ? target.parents : [d.parent_id ?? target.parents[0]]).map(id => byId.get(id ?? '')?.name).filter(Boolean);
  const made = MADE_FROM[target.operation] ?? `Made (${(OPERATION_LABEL[target.operation] ?? d.operation ?? 'processed').toLowerCase()}) from`;
  const disp = d.displacement;
  const before = dims3(d.dimensions_before), after = dims3(d.dimensions_after);
  const change = dims3(d.dimension_change) ?? (before && after ? [after[0] - before[0], after[1] - before[1], after[2] - before[2]] : null);
  const tol = d.tolerance ?? 0.01;
  const tc = d.transform_check;
  const tone = d.verdict === 'changed' ? 'warn' : 'ok';
  const title = d.verdict === 'unchanged'
    ? 'No — the size is unchanged'
    : d.verdict === 'moved'
      ? 'No — it was only moved; shape and size are unchanged'
      : `Yes — the surface changed${disp ? ` (95 % within ${fmtLen(disp.p95, 3)} ${units})` : ''}`;
  return (
    <ResultCard tone={tone} title={title}>
      <span className="ink-2">
        {made} <b>{parents.length ? parents.join(' and ') : 'the model it came from'}</b>.
      </span>
      {disp && (
        <div>
          <Metric label="95 % of the surface moved less than" value={fmtLen(disp.p95, 4)} unit={units} tone={disp.p95 > tol ? 'warn' : 'ok'} />
          <Metric label="Average movement" value={fmtLen(disp.mean, 4)} unit={units} />
          <Metric label="Largest movement" value={fmtLen(disp.max, 4)} unit={units} />
          <Metric label="Average offset (+ outward)" value={fmtSigned(disp.signed_mean, 4)} unit={units} />
          {d.identical_fraction != null && <Metric label="Points left exactly as scanned" value={fmtPct(d.identical_fraction, 2)} />}
          {d.unsupported_fraction != null && d.unsupported_fraction > 0.005 && <Metric label="New surface (not in the parent)" value={fmtPct(d.unsupported_fraction, 1)} />}
          {tc && !tc.identity && (
            <Metric
              label="Placed by"
              value={tc.rigid ? `a turn of ${(tc.rotation_deg ?? 0).toFixed(2)}°, no scaling` : `a transform that resizes (${fmtPpm(tc.scale_ppm)} ppm)`}
              tone={tc.rigid ? undefined : 'danger'}
            />
          )}
        </div>
      )}
      {before && after && change && (
        <div className="meas-table-wrap">
          <table className="data-table meas-table">
            <thead><tr><th>Size</th><th>Before</th><th>After</th><th>Change</th></tr></thead>
            <tbody>
              {SIZE_NAMES.map((n, i) => (
                <tr key={n}>
                  <td>{n}</td>
                  <td className="mono">{fmtLen(before[i], 3)}</td>
                  <td className="mono">{fmtLen(after[i], 3)}</td>
                  <td className={`mono ${Math.abs(change[i]) > tol ? 'delta-warn' : ''}`}>{fmtSigned(change[i], 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <span className="caption">{d.sentence}</span>
    </ResultCard>
  );
}

/** Every processing step back to the original scan, each with its own verdict (checked on request). */
function Chain({ target }: { target: Asset }) {
  const byId = useStore(s => s.byId);
  const drifts = useDrifts(s => s.byId);
  const units = useStore(s => s.display.units);
  const [running, setRunning] = useState(false);
  const chain: Asset[] = [];
  let cur = byId.get(target.parents[0]);
  while (cur && chain.length < 12) {
    chain.push(cur);
    if (isOriginal(cur)) break;
    cur = byId.get(cur.parents[0]);
  }
  const earlier = chain.filter(a => !isOriginal(a));
  if (!earlier.length) return null;
  const pending = earlier.filter(a => drifts[a.id]?.status !== 'done');
  const checkAll = async () => {
    setRunning(true);
    for (const a of pending) await loadDrift(a.id);
    setRunning(false);
  };
  const steps = [...chain].reverse();
  return (
    <div className="acc-chain">
      <div className="acc-chain-head">
        <span className="strong">Every step from the scan</span>
        {pending.length > 0 && (
          <Button size="sm" variant="ghost" icon={<RefreshCw size={14} />} loading={running} onClick={checkAll}>Check {pending.length === 1 ? 'it' : `all ${pending.length}`}</Button>
        )}
      </div>
      <ol className="acc-steps">
        {[...steps, target].map(a => {
          const d = drifts[a.id];
          return (
            <li key={a.id} className={a.id === target.id ? 'is-current' : ''}>
              <span className="acc-step-dot" aria-hidden />
              <span className="acc-step-name truncate" title={a.name}>{a.name}</span>
              <span className="acc-step-op">{isOriginal(a) ? 'scan' : (OPERATION_LABEL[a.operation] ?? a.operation).toLowerCase()}</span>
              <span className="acc-step-verdict">
                {isOriginal(a) ? (
                  <span className="muted">start</span>
                ) : d?.status === 'done' ? (
                  <>
                    {d.drift.displacement && <span className="mono muted">{fmtLen(d.drift.displacement.p95, 3)} {units}</span>}
                    {verdictBadge(d.drift.verdict)}
                  </>
                ) : d?.status === 'loading' ? (
                  <Loader2 size={14} className="spin muted" aria-label="checking" />
                ) : d?.status === 'error' ? (
                  <span className="danger-text" title={d.message}>failed</span>
                ) : (
                  <span className="muted">not checked</span>
                )}
              </span>
            </li>
          );
        })}
      </ol>
      <p className="caption">The number is how far 95 % of the surface moved in that step.</p>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ 2. repeatability
/** How different two models are in size (0 = the same box). */
function sizeDistance(a: Asset, b: Asset): number {
  const dims = (x: Asset) => [...(x.stats.oriented_dimensions ?? x.stats.dimensions ?? [1, 1, 1])].sort((p, q) => q - p);
  const da = dims(a), db = dims(b);
  return da.reduce((sum, v, i) => sum + Math.abs(Math.log(Math.max(v, 1e-6) / Math.max(db[i] ?? 1e-6, 1e-6))), 0);
}

function RepeatCheck({ target }: { target: Asset | undefined }) {
  const project = useProjectAssets();
  const byId = useStore(s => s.byId);
  const cap = useCap('accuracy');
  const run = useCompareScans(s => s.run);
  const job = useStore(s => (run?.jobId ? s.jobs.find(j => j.id === run.jobId) : undefined));
  const scans = project
    .filter(a => a.kind !== 'image' && a.operation !== 'compare' && roleOf(a) !== 'cad')
    .sort((a, b) => Number(isOriginal(b)) - Number(isOriginal(a)) || b.created.localeCompare(a.created));
  const originals = scans.filter(isOriginal);
  const [aId, setA] = useState(() => run?.aId ?? (target && isOriginal(target) ? target.id : originals[0]?.id ?? ''));
  const [bId, setB] = useState(() => {
    if (run?.bId) return run.bId;
    // the other scan most like A in size is most likely the same part scanned again
    const a = originals.find(x => x.id === aId);
    const others = originals.filter(x => x.id !== aId);
    return (a ? [...others].sort((x, y) => sizeDistance(a, x) - sizeDistance(a, y)) : others)[0]?.id ?? '';
  });
  const options = [{ value: '', label: 'Choose a scan…' }, ...scans.map(a => ({ value: a.id, label: `${a.name}${isOriginal(a) ? '' : ` (${(OPERATION_LABEL[a.operation] ?? a.operation).toLowerCase()})`}` }))];
  const running = run?.status === 'running';
  const same = !!run && run.aId === aId && run.bId === bId;

  return (
    <>
      <p className="hint-text">Scan the same part twice without changing anything, then compare the two raw scans. They are lined up without resizing either, and CloudClean measures whether one comes out larger than the other.</p>
      <div className="fields compact">
        <Field label="Scan A" htmlFor="acc-a"><Select id="acc-a" value={aId} onChange={setA} options={options} /></Field>
        <Field label="Scan B" htmlFor="acc-b"><Select id="acc-b" value={bId} onChange={setB} options={options} /></Field>
      </div>
      {scans.length < 2 && <p className="caption">This project has one scan. Scan the part once more to compare.</p>}
      {cap === 'no' || (same && run?.status === 'unavailable') ? (
        <NotHere what="Comparing two scans" />
      ) : (
        <Button icon={<AccuracyGlyph size={16} />} loading={running} disabled={!aId || !bId || aId === bId || running} onClick={() => compareScans(aId, bId)}>
          {aId && aId === bId ? 'Choose two different scans' : 'Compare the two scans'}
        </Button>
      )}
      {running && (
        <div className="acc-progress">
          <Progress value={job?.progress?.fraction ?? null} />
          <span className="caption">{job?.progress?.label ? `${job.progress.label[0].toUpperCase()}${job.progress.label.slice(1)}…` : 'Lining the scans up…'} This can take a minute.</span>
        </div>
      )}
      {same && run?.status === 'failed' && <p className="err-text" role="alert"><TriangleAlert size={14} aria-hidden /> <span>{run.error}</span></p>}
      {same && run?.status === 'done' && run.result && <CompareView r={run.result} a={byId.get(aId)} b={byId.get(bId)} />}
    </>
  );
}

function CompareView({ r, a, b }: { r: CompareScansResult; a?: Asset; b?: Asset }) {
  const units = useStore(s => s.display.units);
  const ea = dims3(r.extents?.a), eb = dims3(r.extents?.b), ed = dims3(r.extents?.difference);
  const lengthDiff = r.length_difference_from_scale_mm ?? (ea ? r.scale_ppm * 1e-6 * ea[0] : null);
  const unsure = r.verdict !== 'consistent' && r.verdict !== 'differ';
  const tone = r.verdict === 'consistent' ? 'ok' : r.verdict === 'differ' ? 'warn' : 'info';
  const title = r.verdict === 'consistent' ? 'Yes — the two scans agree' : r.verdict === 'differ' ? 'No — the two scans differ in size' : 'Can’t tell yet — check how the scans were lined up';
  const sep = r.separation;
  const axis = r.axis_scale_ppm ?? {};
  const axes = (['length', 'width', 'height'] as const).filter(k => axis[k] != null);
  return (
    <>
      <ResultCard tone={tone} title={title}>
        {unsure && r.alignment?.ambiguous && (
          <span className="ink-2">The part looks almost the same turned round, so the two scans may be lined up the wrong way. Treat the numbers below with care.</span>
        )}
        <div className={`acc-hero ${unsure ? 'is-muted' : ''}`}>
          <span className="acc-hero-value">{fmtPpm(r.scale_ppm)}</span>
          <span className="acc-hero-unit">ppm</span>
          {r.scale_se_ppm != null && <span className="acc-hero-pm mono">± {Math.round(r.scale_se_ppm)}</span>}
        </div>
        <span>
          {b?.name ?? 'Scan B'} comes out {r.scale_ppm >= 0 ? 'larger' : 'smaller'} than {a?.name ?? 'scan A'}
          {lengthDiff != null && ea && <> by <b className="mono">{fmtLen(Math.abs(lengthDiff), 3)} {units}</b> over its {fmtLen(ea[0], 1)} {units} length</>}.
        </span>
        <div>
          {sep && <Metric label="Gap between the surfaces (median)" value={fmtLen(sep.median_abs ?? sep.mean, 4)} unit={units} />}
          {sep && <Metric label="95 % of the gaps below" value={fmtLen(sep.p95, 4)} unit={units} />}
          {r.offset_mm != null && <Metric label="Surface offset after scaling" value={fmtSigned(r.offset_mm, 4)} unit={units} />}
          {r.overlap_fraction != null && <Metric label="Overlap of the two scans" value={fmtPct(r.overlap_fraction, 0)} />}
          {r.noise && <Metric label="Scanner noise, scan A" value={fmtLen(r.noise.a, 4)} unit={units} />}
          {r.noise && <Metric label="Scanner noise, scan B" value={fmtLen(r.noise.b, 4)} unit={units} />}
        </div>
      </ResultCard>
      {ea && eb && ed && (
        <div className="meas-table-wrap">
          <table className="data-table meas-table">
            <thead><tr><th>Size</th><th>A</th><th>B</th><th>Difference</th></tr></thead>
            <tbody>
              {SIZE_NAMES.map((n, i) => (
                <tr key={n}>
                  <td>{n}</td>
                  <td className="mono">{fmtLen(ea[i], 3)}</td>
                  <td className="mono">{fmtLen(eb[i], 3)}</td>
                  <td className="mono">{fmtSigned(ed[i], 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {axes.length > 0 && (
        <p className="caption">
          Scale along each axis: {axes.map((k, i) => (
            <span key={k}>{i ? ' · ' : ''}{k} <span className="mono">{fmtPpm(axis[k])}</span></span>
          ))} ppm
        </p>
      )}
      {r.separation_histogram && (
        <Block title="Gap between the two surfaces">
          <Histogram edges={r.separation_histogram.edges} counts={r.separation_histogram.counts} tolerance={r.tolerance ?? 0.02} unit={units} label="gap between the scans" />
        </Block>
      )}
      <p className="caption">100 ppm is 0.01 mm over 100 mm.</p>
      {r.sentence && (
        <details className="meas-values">
          <summary>Full report <ChevronDown size={14} className="chev" aria-hidden /></summary>
          <p className="caption">{r.sentence}</p>
        </details>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------------ 3. known size
type RefType = 'known_length' | 'diameter' | 'sphere_pair';

interface RefCheckState {
  type: RefType;
  nominal: Record<RefType, number>;
  direction: Direction;
  within: boolean;
  tolerance: number;
  ballA: RegionPick | null;
  busy: boolean;
  error: string | null;
  result: { r: ReferenceResult; assetId: string; spec: ReferenceSpec } | null;
}

const useRefCheck = create<RefCheckState>(() => ({
  type: 'known_length',
  nominal: { known_length: 25, diameter: 10, sphere_pair: 100 },
  direction: 'length',
  within: true,
  tolerance: 0,
  ballA: null,
  busy: false,
  error: null,
  result: null,
}));

const REF_TEXT: Record<RefType, { label: string; help: string; nominal: string }> = {
  known_length: { label: 'Length', help: 'A gauge block, or anything with two parallel ends whose length you know. Both end faces are found by themselves.', nominal: 'Certified length' },
  diameter: { label: 'Diameter', help: 'A gauge pin, a pin or a ring gauge whose diameter you know.', nominal: 'Certified diameter' },
  sphere_pair: { label: 'Ball bar', help: 'Two balls with a certified distance between their centres.', nominal: 'Certified centre distance' },
};

function ReferenceCheck({ target }: { target: Asset | undefined }) {
  const s = useRefCheck();
  const units = useStore(x => x.display.units);
  const cap = useCap('accuracy');
  const sel = useSelectionOn(target);
  const has = !!sel && sel.count > 0;
  const nominal = s.nominal[s.type];
  const set = (p: Partial<RefCheckState>) => useRefCheck.setState(p);
  const keptA = !!s.ballA && s.ballA.assetId === target?.id;
  const autoTol = defaultTolerance(nominal);
  const ready = !!target && nominal > 0 && (s.type === 'known_length' || (s.type === 'diameter' ? has : keptA && has));

  const run = async () => {
    if (!target) return;
    let spec: ReferenceSpec;
    const tolerance = s.tolerance > 0 ? s.tolerance : undefined;
    if (s.type === 'known_length') {
      const region = s.within && has ? selectionRegion(false) ?? undefined : undefined;
      spec = { type: 'known_length', direction: s.direction, nominal, region, tolerance };
    } else if (s.type === 'diameter') {
      const region = selectionRegion(false);
      if (!region) return;
      spec = { type: 'diameter', region, nominal, tolerance };
    } else {
      const region = selectionRegion(false);
      if (!region || !s.ballA) return;
      spec = { type: 'sphere_pair', region_a: s.ballA.region, region_b: region, nominal, tolerance };
    }
    set({ busy: true, error: null });
    try {
      const r = await checkReference(target.id, spec);
      set({ result: { r, assetId: target.id, spec }, busy: false });
      const line = referenceLine(r);
      if (line) addDims([{ kind: 'reference', value: r.measured, a: line[0], b: line[1] }], { tool: 'reference', title: `Known ${REF_TEXT[s.type].label.toLowerCase()} · certified ${fmtLen(r.nominal, 3)} ${units}`, detail: `error ${fmtSigned(r.error, 4)} ${units} (${fmtPpm(r.error_ppm)} ppm)`, assetId: target.id });
      if (s.type === 'sphere_pair') set({ ballA: null });
    } catch (err) {
      set({ busy: false, error: err instanceof Unavailable ? 'unavailable' : (err as Error).message });
    }
  };

  if (cap === 'no' || s.error === 'unavailable') return <NotHere what="Checking against a known size" />;
  const result = s.result && target && s.result.assetId === target.id ? s.result.r : null;

  return (
    <>
      <Segmented value={s.type} onChange={type => set({ type, error: null })} options={(Object.keys(REF_TEXT) as RefType[]).map(k => ({ value: k, label: REF_TEXT[k].label }))} />
      <p className="hint-text">{REF_TEXT[s.type].help}</p>

      {s.type === 'known_length' && (
        <>
          <Field label="Along" inline={false}>
            <div className="meas-fit"><Segmented size="sm" ariaLabel="Along" value={s.direction} onChange={direction => set({ direction })} options={DIRS} /></div>
          </Field>
          <SwitchRow label="Only the selected area" checked={s.within && has} disabled={!has} onChange={within => set({ within })} help={has ? 'Measures just the selected block, not the whole scan.' : 'Select the block if the scan holds anything else (a stand, a table).'} />
          <div className="meas-pick">
            <PickButtons disabled={!target} />
            <SelectionStatus target={target} idle="Nothing selected: the whole scan is measured" />
          </div>
        </>
      )}
      {s.type === 'diameter' && (
        <StepList>
          <SelectStep n={1} target={target}>Select the round surface of the pin or ring.</SelectStep>
        </StepList>
      )}
      {s.type === 'sphere_pair' && (
        <StepList>
          {keptA ? (
            <StepItem n={1} done>
              <span className="meas-picked">
                <CheckCircle2 size={14} aria-hidden /> Ball A kept · <span className="mono">{fmtCount(s.ballA!.count)}</span> points
                <button type="button" className="link" onClick={() => set({ ballA: null })}>Choose again</button>
              </span>
            </StepItem>
          ) : (
            <SelectStep
              n={1}
              target={target}
              done={false}
              extra={has ? <Button size="sm" onClick={() => { const p = target && takeSelection(target); if (p) { set({ ballA: p }); clearSelection(); } }}>Keep as ball A</Button> : undefined}
            >
              Select the first ball, then keep it.
            </SelectStep>
          )}
          <SelectStep n={2} target={target} locked={!keptA} done={keptA ? undefined : false}>Select the second ball.</SelectStep>
        </StepList>
      )}

      <Field label={REF_TEXT[s.type].nominal} htmlFor="acc-nominal">
        <NumberInput id="acc-nominal" value={nominal} step={0.001} min={0.001} unit={units} onChange={v => set({ nominal: { ...s.nominal, [s.type]: v } })} />
      </Field>
      <Field label="Allowed error ±" htmlFor="acc-tol" help={s.tolerance > 0 ? 'Your own limit.' : <>0 = automatic: 0.020 {units} + 100 ppm of the size, here <span className="mono">±{fmtLen(autoTol, 4)}</span> {units}.</>}>
        <NumberInput id="acc-tol" value={s.tolerance} step={0.001} min={0} unit={units} onChange={tolerance => set({ tolerance })} />
      </Field>

      <Button icon={<AccuracyGlyph size={16} />} loading={s.busy} disabled={!ready || s.busy} onClick={run}>
        {!target ? 'Pick the scanned artefact' : `Check against ${fmtLen(nominal, 3)} ${units}`}
      </Button>
      {s.error && <p className="err-text" role="alert"><TriangleAlert size={14} aria-hidden /> <span>{s.error}</span></p>}
      {result && <ReferenceView r={result} />}
    </>
  );
}

function ReferenceView({ r }: { r: ReferenceResult }) {
  const units = useStore(s => s.display.units);
  const tol = r.tolerance ?? defaultTolerance(r.nominal);
  const tone = r.verdict === 'ok' ? 'ok' : r.verdict === 'marginal' ? 'warn' : 'danger';
  const head = r.verdict === 'ok' ? 'Within the allowed error' : r.verdict === 'marginal' ? 'Just outside the allowed error' : 'Outside the allowed error';
  return (
    <ResultCard tone={tone} title={<>{head}: <span className="mono">{fmtSigned(r.error, 4)} {units}</span> <span className="muted">({fmtPpm(r.error_ppm)} ppm)</span></>}>
      <ErrorGauge error={r.error} tolerance={tol} uncertainty={r.uncertainty ?? null} units={units} />
      <div>
        <Metric label="Measured" value={fmtLen(r.measured, 4)} unit={units} />
        <Metric label="Certified" value={fmtLen(r.nominal, 4)} unit={units} />
        <Metric label="Error" value={<>{fmtSigned(r.error, 4)}<span className="metric-unit">{units}</span> · {fmtSigned(r.error_pct, 3)}<span className="metric-unit">%</span></>} tone={tone === 'danger' ? 'danger' : tone === 'warn' ? 'warn' : 'ok'} />
        {r.uncertainty != null && <Metric label="Measuring uncertainty (1σ)" value={`± ${fmtLen(r.uncertainty, 4)}`} unit={units} />}
      </div>
      {r.significant != null && (
        <span className="ink-2">
          {r.significant
            ? 'The error is larger than the measuring uncertainty, so it is real — this is the scanner’s scale error.'
            : 'The error is within the measuring uncertainty — it may be just noise.'}
        </span>
      )}
      <span className="caption">{r.sentence}</span>
    </ResultCard>
  );
}

/**
 * Where the error falls: the allowed band (±tolerance) and the "close" band (±2×) around zero, the measured error as
 * an ink needle and its uncertainty as whiskers. A legend names the bands, so the colour is never the only cue.
 */
function ErrorGauge({ error, tolerance, uncertainty, units }: { error: number; tolerance: number; uncertainty: number | null; units: string }) {
  const W = 320, H = 86, padX = 18, bandT = 38, bandH = 20;
  const unc = uncertainty ?? 0;
  const span = Math.max(3 * tolerance, Math.abs(error) * 1.25, (Math.abs(error) + unc) * 1.1, 1e-6);
  const x = (v: number) => padX + ((Math.max(-span, Math.min(span, v)) + span) / (2 * span)) * (W - 2 * padX);
  const ticks = [-2 * tolerance, -tolerance, 0, tolerance, 2 * tolerance].filter(t => Math.abs(t) <= span);
  const digits = tolerance < 0.01 ? 4 : 3;
  const ex = x(error);
  return (
    <div className="acc-gauge-wrap">
      <svg className="acc-gauge" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Error ${error.toFixed(4)} ${units}; allowed ±${tolerance.toFixed(4)} ${units}`}>
        <rect className="gauge-out" x={padX} y={bandT} width={W - 2 * padX} height={bandH} rx={5} />
        <rect className="gauge-close" x={x(-2 * tolerance)} y={bandT} width={x(2 * tolerance) - x(-2 * tolerance)} height={bandH} />
        <rect className="gauge-ok" x={x(-tolerance)} y={bandT} width={x(tolerance) - x(-tolerance)} height={bandH} />
        <line className="gauge-zero" x1={x(0)} x2={x(0)} y1={bandT} y2={bandT + bandH} />
        {ticks.map(t => (
          <g key={t}>
            <line className="gauge-tick" x1={x(t)} x2={x(t)} y1={bandT + bandH} y2={bandT + bandH + 5} />
            <text className="gauge-label" x={x(t)} y={bandT + bandH + 17} textAnchor="middle">{t === 0 ? '0' : fmtSigned(t, digits)}</text>
          </g>
        ))}
        {unc > 0 && (
          <g className="gauge-unc">
            <line x1={x(error - unc)} x2={x(error + unc)} y1={bandT - 8} y2={bandT - 8} />
            <line x1={x(error - unc)} x2={x(error - unc)} y1={bandT - 12} y2={bandT - 4} />
            <line x1={x(error + unc)} x2={x(error + unc)} y1={bandT - 12} y2={bandT - 4} />
          </g>
        )}
        <line className="gauge-needle" x1={ex} x2={ex} y1={bandT - 3} y2={bandT + bandH + 3} />
        <circle className="gauge-dot" cx={ex} cy={bandT + bandH / 2} r={4.5} />
        <text className="gauge-value" x={Math.min(W - 44, Math.max(44, ex))} y={bandT - 18} textAnchor="middle">{fmtSigned(error, 4)} {units}</text>
      </svg>
      <div className="gauge-legend caption">
        <span><i className="gauge-key is-ok" /> within ±{fmtLen(tolerance, digits)}</span>
        <span><i className="gauge-key is-close" /> close</span>
        <span><i className="gauge-key is-out" /> outside</span>
        {unc > 0 && <span><i className="gauge-key is-unc" /> uncertainty</span>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ explanation
function Explainer() {
  return (
    <Block title="Scanner or software — how to tell" tone="sunken">
      <ul className="acc-explain">
        <li>
          <b>Check 1 tests the software.</b> Cleaning only removes points, merging only moves scans without resizing them, and a mesh follows the points. When it says “unchanged”, processing did not change the size of your part.
        </li>
        <li>
          <b>Check 2 tests the scanner.</b> Two scans of the same part should agree. If one comes out larger, the scanner’s scale is drifting — calibration, warm-up, temperature or markers — not the software.
        </li>
        <li>
          <b>Check 3 tests the true size.</b> Only something whose size you know can show whether the scanner reads true.
        </li>
      </ul>
      <div className="acc-tip">
        <AccuracyGlyph size={18} />
        <p>
          <b>The most reliable test:</b> scan a gauge block (certified to about a micrometre), the 20 mm grid of the calibration board, or a ball bar, then use check 3. The error in ppm is the scanner’s scale error — 100 ppm is 0.01 mm over 100 mm.
        </p>
      </div>
    </Block>
  );
}
