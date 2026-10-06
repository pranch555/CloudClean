import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown, ChevronDown, ExternalLink, ImagePlus, RefreshCw, ScanLine, Sparkles, TriangleAlert, Upload } from 'lucide-react';
import { create } from 'zustand';
import { api } from '../../lib/api';
import type { Asset } from '../../lib/types';
import { submitJob } from '../../lib/jobs';
import { pickFiles } from '../../lib/importing';
import { goldenGrade, regionKey, useGoldenPins, type GoldenMeasurement, type GoldenRegion, type GoldenReport } from '../../lib/golden';
import { enter, pulse } from '../../lib/motion';
import { useProjectAssets, useStore } from '../../store';
import { getViewer } from '../../viewer/instance';
import { Button, Empty, Field, NumberInput, Segmented, Select } from '../../ui/primitives';
import { Block } from '../StepFrame';
import { CadResults, isCad } from './CadCompare';
import { CadGlyph } from './glyphs';
import { AreaCard } from './golden/AreaCard';
import { GoldenHero } from './golden/Hero';
import { pinPointing, pointAt, pointingCheck, settlePointing, type Pointing } from './golden/pointing';
import { Sizes, dimensionOf } from './golden/Sizes';
import { SurfaceBar } from './golden/SurfaceBar';
import { useReveal } from './golden/useSeen';
import { STATUS_CODE, plural, pct } from './golden/words';

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

/** The cards of the areas, entering as the list scrolls into view. */
function AreaList({ checkKey, children }: { checkKey: string; children: React.ReactNode }) {
  const [ref, seen] = useReveal<HTMLDivElement>(checkKey, root => enter(root.querySelectorAll('.ga'), { y: 14, step: 70, delay: 60 }));
  return <div ref={ref} className={`ga-list ${seen ? 'is-seen' : ''}`}>{children}</div>;
}

function GoldenResults({ check }: { check: Asset }) {
  const units = useStore(s => s.display.units);
  const display = useStore(s => s.display);
  const byId = useStore(s => s.byId);
  const onStage = useStore(s => s.visible.includes(check.id));
  const [report, setReport] = useState<GoldenReport | null>(null);
  const [failed, setFailed] = useState(false);
  const [hot, setHot] = useState<string | null>(null);
  const [pinnedKey, setPinnedKey] = useState<string | null>(null);
  const active = useGoldenPins(s => s.active);
  const picked = useGoldenPins(s => s.picked);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let alive = true;
    setReport(null);
    setFailed(false);
    setHot(null);
    setPinnedKey(null);
    api.asset(check.id).then(r => alive && setReport(r.report as GoldenReport)).catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [check.id]);

  const version = report?.version ?? 1;
  const tol = Number(report?.tolerance ?? 0.1);
  const colour = (r: GoldenRegion) => report?.legend.find(l => l.key === regionKey(r))?.color ?? 'var(--ink-3)';
  // one numbering for the list and the pins: areas that differ first (they decide the verdict), then what to scan again
  const ordered = useMemo(() => {
    if (!report) return [] as GoldenRegion[];
    const listed = report.regions.filter(r => r.kind === 'off' || r.rescan);
    if (listed.length && listed.every(r => r.number != null)) return [...listed].sort((a, b) => a.number! - b.number!);
    const off = report.regions.filter(r => r.kind === 'off').sort((a, b) => Math.abs(b.deviation ?? 0) * b.area_mm2 - Math.abs(a.deviation ?? 0) * a.area_mm2);
    return [...off, ...report.regions.filter(r => r.rescan)];
  }, [report]);
  const numberOf = (r: GoldenRegion) => r.number ?? ordered.indexOf(r) + 1;
  const areaNumber = (id: number) => {
    const r = ordered.find(x => x.id === id);
    return r ? numberOf(r) : null;
  };

  // the pins on the 3D view and the lights of the panel, while this result is on screen
  useEffect(() => {
    if (!report) return;
    pointingCheck(check.id);
    useGoldenPins.setState({
      checkId: check.id,
      active: null,
      pins: ordered.map(r => ({ n: numberOf(r), region: r.id, pos: r.pin ?? r.view.target, normal: r.normal, color: colour(r), label: r.name })),
    });
    return () => {
      pointingCheck(null);
      useGoldenPins.setState({ checkId: null, pins: [], active: null });
      getViewer()?.spotlight(null);
    };
  }, [report, check.id]);

  const focus = (r: GoldenRegion) => {
    const st = useStore.getState();
    const keepColours = st.display.colorMode === 'scalar' && st.display.scalar?.name === 'golden_deviation';
    useStore.setState({ visible: [check.id], activeId: check.id });
    if (!keepColours) st.setDisplay({ colorMode: 'original', scalar: null });
    setPinnedKey(null);
    pinPointing(null);
    useGoldenPins.setState({ active: r.id });
    window.setTimeout(() => {
      getViewer()?.viewFrom(r.view.target, r.view.from, 650, r.view.radius);
      // older checks coloured the raw CAD mesh, where an area can have hardly any vertices of its own: no spotlight
      if (version >= 2) settlePointing();
    }, 60);
  };
  const unfocus = () => {
    useGoldenPins.setState({ active: null });
    settlePointing();
    window.setTimeout(() => getViewer()?.fit([check.id]), 30);
  };

  const showArea = (id: number) => {
    const r = report?.regions.find(x => x.id === id);
    if (!r) return;
    focus(r);
    const card = listRef.current?.querySelector<HTMLElement>(`[data-area="${r.id}"]`);
    card?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    window.setTimeout(() => pulse(card ?? null), 350);
  };

  // a pin was clicked on the 3D view: show that area and bring its card into view
  useEffect(() => {
    if (picked && report) showArea(picked.region);
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
    setPinnedKey(null);
    pinPointing(null);
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
  const scrollTo = (id: string) => listRef.current?.querySelector(`#${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // ---- pointing at the model: the surface colours, the areas, the sizes
  const statusPointing = (key: string): Pointing => ({ kind: 'status', code: STATUS_CODE[key] ?? 0 });
  const pointStatus = (key: string | null) => {
    setHot(key);
    pointAt(key ? statusPointing(key) : null);
  };
  const pinStatus = (key: string) => {
    const next = pinnedKey === key ? null : key;
    setPinnedKey(next);
    pinPointing(next ? statusPointing(next) : null);
    // a kept colour can be a few small places: bring them into view (the scalar may still be loading: wait a moment)
    const v = getViewer() as unknown as { frameHighlight?: (duration?: number) => Promise<boolean> } | null;
    if (next && next !== 'good' && onStage) window.setTimeout(() => void v?.frameHighlight?.(650), 260);
  };
  const sizePointing = (m: GoldenMeasurement): Pointing => ({ kind: 'size', faces: m.faces ?? [], dimension: dimensionOf(m, units) });
  const pointSize = (m: GoldenMeasurement | null) => pointAt(m ? sizePointing(m) : null);
  const pinSize = (m: GoldenMeasurement | null) => {
    setPinnedKey(null);
    pinPointing(m ? sizePointing(m) : null);
  };
  const showSize = (m: GoldenMeasurement) => {
    if (!onStage) show('problems');
    pinSize(m);
    if (m.ends) {
      const [a, b] = m.ends;
      window.setTimeout(() => getViewer()?.lookAtPoint([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2], 550), 80);
    }
  };

  const slices = report.legend.map(l => ({ key: l.key, value: shares[l.key] ?? 0, color: l.color }));
  const card = (r: GoldenRegion) => (
    <AreaCard
      key={r.id}
      checkId={check.id}
      r={r}
      n={numberOf(r)}
      color={colour(r)}
      on={active === r.id}
      tol={tol}
      units={units}
      pictures={version >= 2}
      onShow={() => focus(r)}
      onWhole={unfocus}
      onPoint={id => {
        if (version >= 2) pointAt(id != null ? { kind: 'region', id } : null);
      }}
    />
  );

  return (
    <div className="gold" ref={listRef}>
      <GoldenHero
        checkKey={check.id}
        grade={grade}
        title={grade.label}
        sub={sub}
        slices={slices}
        tolText={tolText}
        onPoint={pointStatus}
        stats={[
          { value: report.surface.scanned_pct, decimals: 1, suffix: '%', label: 'of the part scanned', onClick: () => scrollTo('gold-surface') },
          { value: count('ok'), decimals: 0, suffix: `/${measured}`, label: measured ? 'sizes match' : 'no sizes to measure', fail: count('off') > 0, disabled: !ms.length, onClick: () => scrollTo('gold-sizes') },
          { value: ordered.length, decimals: 0, label: ordered.length === 1 ? 'area to look at' : 'areas to look at', fail: differs.length > 0, disabled: !ordered.length, onClick: () => scrollTo('gold-areas') },
        ]}
      />

      {version < 3 && (
        <div className="gold-legacy">
          <Sparkles size={18} aria-hidden />
          <div>
            <p>
              {version < 2
                ? 'This check was made by an older CloudClean: its areas are named by coordinates and its sizes by face positions. '
                : 'This check was made before the latest update. '}
              Run it again (about a minute) for plain names, a drawing of the part, and sizes you can point at on the model.
            </p>
            <LegacyRerun check={check} report={report} />
          </div>
        </div>
      )}

      <section className="gold-section" id="gold-surface">
        <div className="gold-section-head">
          <h3>The surface</h3>
          <a className="gold-report-link" href={`/api/assets/${check.id}/golden-report`} target="_blank" rel="noreferrer"><ExternalLink size={14} aria-hidden /> Printable report</a>
        </div>
        <SurfaceBar checkKey={check.id} shares={slices} tolText={tolText} hot={hot ?? pinnedKey} pinned={pinnedKey} canLight={onStage} onPoint={pointStatus} onPin={pinStatus} />
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
          <AreaList checkKey={check.id}>
            {differs.length > 0 && (
              <>
                <div className="gold-group"><span className="gold-group-dot is-off" aria-hidden />Different from the golden model <span className="mono">{differs.length}</span></div>
                <ol className="ga-cards">{differs.map(card)}</ol>
              </>
            )}
            {rescan.length > 0 && (
              <>
                <div className="gold-group"><ScanLine size={15} aria-hidden />Scan these again <span className="mono">{rescan.length}</span></div>
                <ol className="ga-cards">{rescan.map(card)}</ol>
                {rescan.some(r => r.kind === 'missing' || r.kind === 'thin') && <FillFromPhotos scanId={scanId} />}
              </>
            )}
          </AreaList>
        </section>
      )}

      <section className="gold-section" id="gold-sizes" data-guide="golden.sizes">
        <div className="gold-section-head">
          <h3>Sizes</h3>
          <span className="gold-section-note">designed → scanned</span>
        </div>
        {ms.length ? (
          <Sizes
            checkKey={check.id}
            report={report}
            units={units}
            tolText={tolText}
            areaNumber={areaNumber}
            onPoint={pointSize}
            onPin={pinSize}
            onShowArea={showArea}
            onShowSize={showSize}
          />
        ) : (
          <p className="hint-text">The golden model has no flat or round faces to measure: use the colours and the areas above.</p>
        )}
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
    <Button size="sm" variant="primary" icon={<RefreshCw size={14} />} loading={busy} disabled={!ok} onClick={async () => {
      setBusy(true);
      await runCheck({ scan_id: scanId, golden_id: goldenId, tolerance: Number(report.tolerance ?? 0.1), align }, () => setBusy(false));
    }}>
      {ok ? 'Update this check' : 'The scan or golden model was deleted'}
    </Button>
  );
}
