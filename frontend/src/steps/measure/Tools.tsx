import { Fragment, useEffect, useRef } from 'react';
import { CheckCircle2, Focus, TriangleAlert, X } from 'lucide-react';
import { clearSelection, setTool } from '../../lib/actions';
import { fmtCount, fmtLen } from '../../lib/format';
import { remeasureAll } from '../../lib/measure';
import {
  acrossLine, addDims, measureAngle, measureCaliper, measureDiameter, measureExtent, measureFaces, measurePlane, measureSection, measureSphere,
  selectionRegion, showOnModel, takeSelection, Unavailable, useCap, useDimInfo, useSelectionOn, type PartAxis,
} from '../../lib/measureTools';
import { partDims, useSummary, type PartSummary } from '../../lib/summary';
import type { Asset } from '../../lib/types';
import { useStore, type DimLine, type MeasureAxis, type Vec3 } from '../../store';
import { Button, Field, IconButton, Segmented } from '../../ui/primitives';
import { ToolTile } from '../StepFrame';
import { AngleGlyph, CaliperGlyph, DiameterGlyph, ExtentGlyph, FlatnessGlyph, PointsGlyph, SectionGlyph, SphereGlyph, StepsGlyph, type Glyph } from './glyphs';
import { CopyButton, NotHere, SelectionStatus, SelectStep, StepItem, StepList, SwitchRow, PickButtons } from './parts';
import { DIR_NAME, DIRS, prefixOf, unitText, valueText } from './format';
import { ProfilePicker, SectionDrawing } from './SectionView';
import { setError, setTab, setUi, useMeasureUi, type ToolId } from './state';

interface ToolDef {
  id: ToolId;
  label: string;
  sub: string;
  glyph: Glyph;
  lead: string;
  /** needs the part-measurement routes of the newer server (Contract 3) */
  server: boolean;
}

export const TOOLS: ToolDef[] = [
  { id: 'p2p', label: 'Point to point', sub: 'click two points', glyph: PointsGlyph, server: false, lead: 'The distance between two points you click. Each point snaps to the nearest scanned surface.' },
  { id: 'across', label: 'Caliper', sub: 'between two faces', glyph: CaliperGlyph, server: true, lead: 'The distance between two opposite faces, measured like calipers: the jaws sit on the faces, not on stray points.' },
  { id: 'extent', label: 'Overall size', sub: 'end to end', glyph: ExtentGlyph, server: true, lead: 'The overall size in one direction, from one end of the part to the other. A few stray points are ignored.' },
  { id: 'steps', label: 'Heights & steps', sub: 'head height, shoulders', glyph: StepsGlyph, server: true, lead: 'Finds every flat face across the part — its ends, the shoulder under a head, the steps of a shaft — and the distance between each pair, like a height gauge. Use it for head heights and the length under a head.' },
  { id: 'diameter', label: 'Diameter', sub: 'shafts, pins, holes', glyph: DiameterGlyph, server: true, lead: 'Fits a cylinder to a round surface — a shank, a pin or a hole — and gives its diameter.' },
  { id: 'angle', label: 'Angle', sub: 'between two faces', glyph: AngleGlyph, server: true, lead: 'The angle between two flat faces, or between a face and a straight edge.' },
  { id: 'flatness', label: 'Flatness', sub: 'of one face', glyph: FlatnessGlyph, server: true, lead: 'How flat a face is: the gap between the two parallel planes that just hold all of its points.' },
  { id: 'sphere', label: 'Ball / sphere', sub: 'balls and domes', glyph: SphereGlyph, server: true, lead: 'Fits a ball to a round, dome-shaped surface and gives its diameter and centre.' },
  { id: 'section', label: 'Cross-section', sub: 'cut across the part', glyph: SectionGlyph, server: true, lead: 'Cuts across the part and draws the outline of the cut, with its width and height.' },
];

const toolDef = (id: ToolId) => TOOLS.find(t => t.id === id)!;

/** app guide ids (cloudclean/assistant/guide.py) of the tools */
const TOOL_GUIDE: Record<ToolId, string> = {
  p2p: 'measure.p2p', across: 'measure.caliper', extent: 'measure.overall', steps: 'measure.heights', diameter: 'measure.diameter',
  angle: 'measure.angle', flatness: 'measure.flatness', sphere: 'measure.sphere', section: 'measure.section',
};
const PART: PartAxis[] = ['length', 'width', 'height'];

/** Size of the part along one of its axes (robust when the summary is known). */
function partRange(target: Asset, summary: PartSummary | null, axis: PartAxis): number {
  return partDims(target, summary).dims[PART.indexOf(axis)] ?? 0;
}

// ------------------------------------------------------------------------------------------------ running a tool
export async function runTool(tool: ToolId, target: Asset, summary: PartSummary | null) {
  const ui = useMeasureUi.getState();
  const units = useStore.getState().display.units;
  const hasSel = (useStore.getState().selectionCounts[target.id] ?? 0) > 0;
  const L = (v: number | null | undefined, d = 3) => fmtLen(v, d);
  const typical = (rms: number) => `typical gap to the fit ${L(rms)} ${units}`;
  setUi({ busy: tool });
  setError(tool, null);
  try {
    let lines: DimLine[] = [];
    const assetId = target.id;
    if (tool === 'across' || tool === 'extent') {
      const dir = ui[tool];
      const region = ui.within && hasSel ? selectionRegion(false) : null;
      const where = region ? ' · within the selection' : '';
      if (tool === 'across') {
        const r = await measureCaliper(assetId, dir, region);
        const flat = [r.face_a, r.face_b].filter(f => f?.kind === 'plane').length;
        const faces = flat === 2 ? `flat faces at both ends, ${r.parallelism_deg.toFixed(2)}° from parallel` : flat === 1 ? 'one flat face and one rounded end' : 'rounded ends: the jaws touch the highest points';
        lines = addDims([{ kind: 'caliper', value: r.distance, a: r.a, b: r.b }], { tool, title: `Caliper across ${DIR_NAME[dir]}`, detail: faces + where, warnings: r.warnings, assetId });
      } else {
        const r = await measureExtent(assetId, dir, region);
        // draw the robust extent (the value shown), not the full one reaching the farthest stray point
        const d = r.direction ?? [0, 0, 0];
        const lo = (r.robust_min ?? r.min) - r.min, hi = (r.robust_max ?? r.max) - r.max;
        const a = r.a.map((x, i) => x + lo * d[i]) as Vec3;
        const b = r.b.map((x, i) => x + hi * d[i]) as Vec3;
        const every = Math.abs(r.length - r.robust_length) >= 0.0005 ? ` · ${L(r.length)} ${units} counting every point` : '';
        lines = addDims([{ kind: 'extent', value: r.robust_length ?? r.length, a, b }], { tool, title: `Overall size along ${DIR_NAME[dir]}`, detail: `stray points ignored${every}${where}`, assetId });
      }
    } else if (tool === 'steps') {
      const dir = ui.steps;
      const region = ui.within && hasSel ? selectionRegion(false) : null;
      const r = await measureFaces(assetId, dir, region);
      const f = r.faces;
      if (f.length < 2) throw new Error(`Only one flat face across ${DIR_NAME[dir]} was found.`);
      // dimension lines run parallel to the direction just outside the faces, fanned round the axis so they never
      // hide behind each other whatever the view
      const d = r.direction;
      const u = sideways(d);
      const v = [d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]];
      const fan = (i: number) => {
        const t = (i * Math.PI) / 2;
        return u.map((x, j) => Math.cos(t) * x + Math.sin(t) * v[j]);
      };
      const off = (k: number, by: number, i: number) => f[k].point.map((x, j) => x + by * fan(i)[j]) as Vec3;
      const notFlat = (k: number) => (f[k].rms > 0.1 ? ` (not flat, ${L(f[k].rms)})` : '');
      const found = `Face to face from plane fits · ${f.map(x => `${x.label ?? 'face'} ${L(x.position, 2)}`).join(' · ')} ${units}`;
      // named distances (head height, length under the head, recess depth, overall), else every neighbouring pair
      const key = r.key_distances?.length ? r.key_distances : r.steps.map(s => ({ ...s, what: `Face ${s.from + 1} → ${s.to + 1}` }));
      const items = key.map((s, i) => {
        const by = Math.max(...f.slice(s.from, s.to + 1).map(x => x.radius_range[1])) * (1.1 + 0.1 * Math.floor(i / 4));
        return { kind: 'faces', value: s.distance, a: off(s.from, by, i), b: off(s.to, by, i) };
      });
      const title = (what: string) => what.charAt(0).toUpperCase() + what.slice(1);
      lines = addDims(items, {
        tool,
        title: key.map(s => title(s.what)),
        detail: key.map(s => `${found}${notFlat(s.from)}${notFlat(s.to)}`),
        warnings: r.warnings,
        assetId,
      });
    } else if (tool === 'diameter') {
      const region = selectionRegion(false);
      if (!region || !hasSel) throw new Error('Select the round surface first.');
      const r = await measureDiameter(assetId, region, ui.axisHint === 'auto' ? null : ui.axisHint);
      const [a, b] = r.a && r.b ? [r.a, r.b] : acrossLine(r.center, r.axis, r.radius);
      const shape = r.kind === 'circle' ? 'circle (a thin ring)' : 'cylinder';
      lines = addDims([{ kind: 'diameter', value: r.diameter, a, b }], {
        tool,
        title: `Diameter · ${shape}`,
        detail: `${Math.round(r.coverage_deg)}° of the round scanned · ${L(r.length, 2)} ${units} long · ${typical(r.rms)}`,
        warnings: r.warnings,
        assetId,
      });
    } else if (tool === 'flatness') {
      const region = selectionRegion(ui.facing);
      if (!region || !hasSel) throw new Error('Select a flat face first.');
      const r = await measurePlane(assetId, region);
      const size = partRange(target, summary, 'length') || 10;
      const a = r.a ?? (r.point as Vec3);
      const b = r.b ?? (r.point.map((x, i) => x + 0.08 * size * r.normal[i]) as Vec3);
      lines = addDims([{ kind: 'flatness', value: r.flatness, a, b }], { tool, title: 'Flatness', detail: `${fmtCount(r.points_used)} points · ${typical(r.rms)}${ui.facing ? ' · side facing you' : ''}`, warnings: r.warnings, assetId });
    } else if (tool === 'sphere') {
      const region = selectionRegion(false);
      if (!region || !hasSel) throw new Error('Select the ball first.');
      const r = await measureSphere(assetId, region);
      const [a, b] = r.a && r.b ? [r.a, r.b] : acrossLine(r.center, [0, 0, 1], r.radius);
      const cover = r.coverage != null ? `${Math.round(r.coverage * 100)} % of the ball scanned · ` : '';
      lines = addDims([{ kind: 'sphere', value: r.diameter, a, b }], { tool, title: 'Sphere', detail: `${cover}centre ${r.center.map(x => x.toFixed(2)).join(', ')} · ${typical(r.rms)}`, warnings: r.warnings, assetId });
    } else if (tool === 'angle') {
      const faceA = ui.faceA;
      const regionB = selectionRegion(ui.facing);
      if (!faceA) throw new Error('Keep face A first.');
      if (!regionB || !hasSel) throw new Error('Select face B first.');
      const r = await measureAngle(assetId, faceA.region, regionB);
      const word = (t: string) => (t === 'line' ? 'edge' : 'face');
      const a = (r.a ?? r.fit_a.point) as Vec3, b = (r.b ?? r.fit_b.point) as Vec3;
      const other = r.supplement_deg ?? 180 - r.angle_deg;
      const detail = r.normals_angle_deg != null ? `the other way round ${other.toFixed(2)}° · between the outward sides ${r.normals_angle_deg.toFixed(2)}°` : `the other way round ${other.toFixed(2)}°`;
      lines = addDims([{ kind: 'angle', value: r.angle_deg, a, b }], { tool, title: `Angle · ${word(r.fit_a.type)} to ${word(r.fit_b.type)}`, detail, assetId });
      setUi({ faceA: null });
      clearSelection();
    } else if (tool === 'section') {
      const dir = ui.sectionDir;
      const range = partRange(target, summary, dir);
      const at = Math.min(Math.max(ui.sectionAt ?? range / 2, 0), range || Infinity);
      const r = await measureSection(assetId, { direction: dir, at });
      if (r.basis) {
        const { origin: o, u, v } = r.basis;
        const [lo, hi] = [r.bbox2d.min, r.bbox2d.max];
        const P = (s: number, t: number) => [0, 1, 2].map(i => o[i] + s * u[i] + t * v[i]) as Vec3;
        const mu = (lo[0] + hi[0]) / 2, mv = (lo[1] + hi[1]) / 2;
        const where = `${L(at, 2)} ${units}`;
        lines = addDims(
          [
            { kind: 'section', value: r.width, a: P(lo[0], mv), b: P(hi[0], mv) },
            { kind: 'section', value: r.height, a: P(mu, lo[1]), b: P(mu, hi[1]) },
          ],
          { tool, title: [`Section width at ${where}`, `Section height at ${where}`], detail: `cut across ${DIR_NAME[dir]} · ${fmtCount(r.points_used)} points on the cut`, assetId },
        );
      }
      setUi({ section: { result: r, direction: dir, at, assetId, dimIds: lines.map(l => l.id) } });
    }
    setUi({ last: { ...useMeasureUi.getState().last, [tool]: lines.map(l => l.id) } });
  } catch (err) {
    setError(tool, err instanceof Unavailable ? 'unavailable' : (err as Error).message);
  } finally {
    setUi({ busy: null });
  }
}

/** A unit vector square to a direction (for placing a dimension line beside the part). */
function sideways(d: number[]): number[] {
  const helper = Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const c = [d[1] * helper[2] - d[2] * helper[1], d[2] * helper[0] - d[0] * helper[2], d[0] * helper[1] - d[1] * helper[0]];
  const n = Math.hypot(c[0], c[1], c[2]) || 1;
  return c.map(x => x / n);
}

function keepFaceA(target: Asset) {
  const pick = takeSelection(target, useMeasureUi.getState().facing);
  if (!pick) return;
  setUi({ faceA: pick });
  setError('angle', null);
  clearSelection();
}

function chooseTool(id: ToolId | null) {
  setUi({ tool: id });
  const vt = useStore.getState().tool;
  if (id === 'p2p') setTool('measure');
  else if (vt === 'measure') setTool('navigate');
}

// ------------------------------------------------------------------------------------------------ grid + inspector
/** The measuring tools as tiles; the chosen tool opens right under its row, pointing at its tile. */
export function ToolGrid({ target }: { target: Asset | undefined }) {
  const active = useMeasureUi(s => s.tool);
  const viewTool = useStore(s => s.tool);
  const cap = useCap('measure');

  // M (or the rail's ruler) starts two-point measuring: show its guidance
  useEffect(() => {
    if (viewTool === 'measure' && useMeasureUi.getState().tool !== 'p2p') setUi({ tool: 'p2p' });
  }, [viewTool]);

  const rows: ToolDef[][] = [];
  for (let i = 0; i < TOOLS.length; i += 2) rows.push(TOOLS.slice(i, i + 2));
  return (
    <div className={`tiles meas-tools ${cap === 'no' ? 'is-limited' : ''}`}>
      {rows.map((row, r) => (
        <Fragment key={r}>
          {row.map(t => (
            <ToolTile
              key={t.id}
              icon={<t.glyph size={19} />}
              label={t.label}
              sub={t.server && cap === 'no' ? 'needs an update' : t.sub}
              active={active === t.id}
              onClick={() => chooseTool(active === t.id ? null : t.id)}
              guide={TOOL_GUIDE[t.id]}
            />
          ))}
          {active && row.some(t => t.id === active) && <Inspector tool={active} target={target} side={row[0].id === active ? 'left' : 'right'} />}
        </Fragment>
      ))}
    </div>
  );
}

const smooth = (): ScrollBehavior => (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth');

function Inspector({ tool, target, side }: { tool: ToolId; target: Asset | undefined; side: 'left' | 'right' }) {
  const def = toolDef(tool);
  const cap = useCap('measure');
  const error = useMeasureUi(s => s.errors[tool]);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollIntoView({ block: 'nearest', behavior: smooth() });
  }, [tool]);
  const missing = def.server && (cap === 'no' || error === 'unavailable');
  return (
    <div ref={ref} className={`meas-insp notch-${side}`} role="region" aria-label={`${def.label} tool`}>
      <div className="meas-insp-head">
        <span className="meas-insp-glyph"><def.glyph size={18} /></span>
        <h4 className="meas-insp-title">{def.label}</h4>
        <IconButton size="sm" label="Close this tool" onClick={() => chooseTool(null)}>
          <X size={15} />
        </IconButton>
      </div>
      <p className="meas-insp-lead">{def.lead}</p>
      {missing ? (
        <NotHere what={`Measuring ${def.label === 'Caliper' ? 'with the caliper' : def.label.toLowerCase()}`}>
          <p className="caption">Point to point and the thread analysis work now.</p>
        </NotHere>
      ) : (
        <>
          {tool === 'p2p' && <PointsBody />}
          {(tool === 'across' || tool === 'extent' || tool === 'steps') && <DirectionBody tool={tool} target={target} />}
          {tool === 'diameter' && <DiameterBody target={target} />}
          {tool === 'angle' && <AngleBody target={target} />}
          {tool === 'flatness' && <FlatnessBody target={target} />}
          {tool === 'sphere' && <SphereBody target={target} />}
          {tool === 'section' && <SectionBody target={target} />}
          {error && error !== 'unavailable' && (
            <p className="err-text" role="alert"><TriangleAlert size={14} aria-hidden /> <span>{error}</span></p>
          )}
          <LatestResult tool={tool} />
        </>
      )}
    </div>
  );
}

function PointsBody() {
  const viewTool = useStore(s => s.tool);
  const axis = useStore(s => s.measureAxis);
  const thread = useStore(s => s.thread);
  const open = useStore(s => s.measurements.some(m => !m.b));
  const measuring = viewTool === 'measure';
  const setAxis = (a: MeasureAxis) => {
    useStore.setState({ measureAxis: a });
    remeasureAll();
  };
  const help = axis === 'none'
    ? 'Direct: the straight line between the two points.'
    : axis === 'thread'
      ? 'Along the thread axis, crest to crest gives the pitch even when the two points are not exactly in line.'
      : `Only the part of the distance along ${axis.toUpperCase()}; any sideways offset is left out.`;
  return (
    <>
      <StepList>
        <StepItem n={1} done={measuring}>Press <b>Start measuring</b> below, or <kbd className="kbd">M</kbd>.</StepItem>
        <StepItem n={2} done={open} locked={!measuring}>Click the first point on the model.</StepItem>
        <StepItem n={3} locked={!measuring}>Click the second point. The distance is drawn on the model and listed below.</StepItem>
      </StepList>
      <Field label="Measure along" inline={false} help={help}>
        <div className="meas-fit">
          <Segmented
            size="sm"
            ariaLabel="Measure along"
            value={axis}
            onChange={setAxis}
            options={[
              { value: 'none', label: 'Direct' },
              { value: 'x', label: 'X' },
              { value: 'y', label: 'Y' },
              { value: 'z', label: 'Z' },
              { value: 'thread', label: 'Thread axis', title: thread ? 'The axis found by the thread analysis' : 'Analyse the thread first' },
            ]}
          />
        </div>
      </Field>
      {axis === 'thread' && !thread && (
        <p className="warn-text">
          <TriangleAlert size={13} aria-hidden />
          <span>
            No thread axis yet — <button type="button" className="link" onClick={() => setTab('thread')}>analyse the thread</button> first.
          </span>
        </p>
      )}
    </>
  );
}

function DirectionBody({ tool, target }: { tool: 'across' | 'extent' | 'steps'; target: Asset | undefined }) {
  const dir = useMeasureUi(s => s[tool]);
  const within = useMeasureUi(s => s.within);
  const sel = useSelectionOn(target);
  const has = !!sel && sel.count > 0;
  return (
    <>
      <Field label="Direction" inline={false} help="Length, width and height follow the part itself, longest to shortest. X, Y and Z follow the scanner.">
        <div className="meas-fit">
          <Segmented size="sm" ariaLabel="Direction" value={dir} onChange={d => setUi({ [tool]: d })} options={DIRS} />
        </div>
      </Field>
      <SwitchRow
        label="Only within the selection"
        checked={within && has}
        disabled={!has}
        onChange={v => setUi({ within: v })}
        help={has ? <>Measures just the <span className="mono">{fmtCount(sel!.count)}</span> selected points — for example only the head.</> : 'Select part of the model first to measure only there, for example the head.'}
      />
      <div className="meas-pick">
        <PickButtons disabled={!target} />
        <SelectionStatus target={target} idle="Nothing selected: the whole part is measured" />
      </div>
    </>
  );
}

function DiameterBody({ target }: { target: Asset | undefined }) {
  const hint = useMeasureUi(s => s.axisHint);
  return (
    <>
      <StepList>
        <SelectStep n={1} target={target}>Select the round surface: box the shank, pin or hole. Leave out ends and chamfers.</SelectStep>
        <StepItem n={2}>Press <b>Measure diameter</b>.</StepItem>
      </StepList>
      <Field label="Runs along" inline={false} help="Automatic finds the axis by itself. Pick a part axis only if the result looks wrong.">
        <Segmented size="sm" ariaLabel="Axis of the round surface" value={hint} onChange={v => setUi({ axisHint: v })} options={[{ value: 'auto', label: 'Automatic' }, { value: 'length', label: 'Length' }, { value: 'width', label: 'Width' }, { value: 'height', label: 'Height' }]} />
      </Field>
    </>
  );
}

function AngleBody({ target }: { target: Asset | undefined }) {
  const faceA = useMeasureUi(s => s.faceA);
  const facing = useMeasureUi(s => s.facing);
  const kept = !!faceA && faceA.assetId === target?.id;
  return (
    <>
      <StepList>
        {kept ? (
          <StepItem n={1} done>
            <span className="meas-picked">
              <CheckCircle2 size={14} aria-hidden /> Face A kept · <span className="mono">{fmtCount(faceA!.count)}</span> points
              <button type="button" className="link" onClick={() => setUi({ faceA: null })}>Choose again</button>
            </span>
          </StepItem>
        ) : (
          <SelectStep n={1} target={target} done={false}>Select the first face, then press <b>Keep as face A</b>.</SelectStep>
        )}
        <SelectStep n={2} target={target} locked={!kept} done={kept ? undefined : false}>Select the second face (B).</SelectStep>
        <StepItem n={3} locked={!kept}>Press <b>Measure the angle</b>.</StepItem>
      </StepList>
      <SwitchRow label="Only the side facing me" checked={facing} onChange={v => setUi({ facing: v })} help="Leaves out points on the far side of the part that a box also catches." />
    </>
  );
}

function FlatnessBody({ target }: { target: Asset | undefined }) {
  const facing = useMeasureUi(s => s.facing);
  return (
    <>
      <StepList>
        <SelectStep n={1} target={target}>Select a flat face. Stay a little inside its edges.</SelectStep>
        <StepItem n={2}>Press <b>Measure flatness</b>.</StepItem>
      </StepList>
      <SwitchRow label="Only the side facing me" checked={facing} onChange={v => setUi({ facing: v })} help="Leaves out points on the far side of the part that a box also catches." />
      <p className="caption">Scanner noise adds to flatness, so a very flat face reads slightly high.</p>
    </>
  );
}

function SphereBody({ target }: { target: Asset | undefined }) {
  return (
    <StepList>
      <SelectStep n={1} target={target}>Select the ball or dome. The more of it is scanned, the better the fit.</SelectStep>
      <StepItem n={2}>Press <b>Fit the sphere</b>.</StepItem>
    </StepList>
  );
}

function SectionBody({ target }: { target: Asset | undefined }) {
  const dir = useMeasureUi(s => s.sectionDir);
  const at = useMeasureUi(s => s.sectionAt);
  const units = useStore(s => s.display.units);
  const summary = useSummary(target);
  if (!target) return <p className="hint-text">Pick a model first.</p>;
  const range = partRange(target, summary, dir);
  const pos = at == null ? range / 2 : Math.min(Math.max(at, 0), range);
  return (
    <>
      <Field label="Cut across" inline={false} help="The cut is square to this direction of the part.">
        <Segmented size="sm" ariaLabel="Cut across" value={dir} onChange={d => setUi({ sectionDir: d, sectionAt: null })} options={[{ value: 'length', label: 'Length' }, { value: 'width', label: 'Width' }, { value: 'height', label: 'Height' }]} />
      </Field>
      <Field label={<>Where <span className="mono muted">{fmtLen(pos, 2)} {units}</span></>} inline={false}>
        <ProfilePicker summary={summary} direction={dir} range={range} at={pos} units={units} onChange={v => setUi({ sectionAt: v })} />
      </Field>
    </>
  );
}

/** The newest result of a tool, as a large readout with its facts and warnings. */
function LatestResult({ tool }: { tool: ToolId }) {
  const ids = useMeasureUi(s => s.last[tool]);
  const section = useMeasureUi(s => s.section);
  const dims = useStore(s => s.dims);
  const units = useStore(s => s.display.units);
  const info = useDimInfo(s => s.byId);
  const ref = useRef<HTMLDivElement>(null);
  const lines = (ids ?? []).map(id => dims.find(d => d.id === id)).filter((d): d is DimLine => !!d);
  const key = lines.map(l => l.id).join();
  useEffect(() => {
    if (key) ref.current?.scrollIntoView({ block: 'nearest', behavior: smooth() });
  }, [key]);
  if (!lines.length) return null;
  const first = info[lines[0].id];
  return (
    <div ref={ref} className="meas-latest" aria-live="polite">
      {lines.map(l => (
        <div key={l.id} className="meas-readout">
          <div className="meas-readout-title">{info[l.id]?.title ?? l.kind}</div>
          <div className="meas-readout-row">
            <span className="meas-tag">{l.label}</span>
            <span className="meas-readout-value">
              {prefixOf(l.kind)}{valueText(l)}<span className="meas-unit">{unitText(l)}</span>
            </span>
            <span className="meas-readout-actions">
              <IconButton size="sm" label="Show it on the model" tip="top" onClick={() => showOnModel(l.a, l.b, l.assetId)}>
                <Focus size={15} />
              </IconButton>
              <CopyButton text={`${valueText(l)} ${unitText(l)}`} />
            </span>
          </div>
        </div>
      ))}
      {tool === 'section' && section && ids?.some(id => section.dimIds.includes(id)) && <SectionDrawing shot={section} units={units} />}
      {first?.detail && <p className="meas-latest-detail">{first.detail}</p>}
      {first?.warnings.map((w, i) => (
        <p key={i} className="warn-text"><TriangleAlert size={13} aria-hidden /> <span>{w}</span></p>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ footer action
/** The one obvious next action for the chosen tool (sticky at the bottom of the panel). */
export function DimensionsFooter({ target }: { target: Asset | undefined }) {
  const tool = useMeasureUi(s => s.tool);
  const busy = useMeasureUi(s => s.busy);
  const faceA = useMeasureUi(s => s.faceA);
  const across = useMeasureUi(s => s.across);
  const extent = useMeasureUi(s => s.extent);
  const steps = useMeasureUi(s => s.steps);
  const error = useMeasureUi(s => (tool ? s.errors[tool] : null));
  const viewTool = useStore(s => s.tool);
  const sel = useSelectionOn(target);
  const summary = useSummary(target);
  const cap = useCap('measure');
  if (!tool) return null;
  const def = toolDef(tool);
  if (def.server && (cap === 'no' || error === 'unavailable')) return null;
  const Glyph = def.glyph;
  if (tool === 'p2p') {
    return viewTool === 'measure' ? (
      <Button size="lg" block variant="secondary" onClick={() => setTool('navigate')}>Done measuring</Button>
    ) : (
      <Button size="lg" block variant="primary" icon={<Glyph size={18} />} disabled={!target} onClick={() => setTool('measure')}>
        {target ? 'Start measuring' : 'Pick a model to measure'}
      </Button>
    );
  }
  const has = !!sel && sel.count > 0;
  const keptA = !!faceA && faceA.assetId === target?.id;
  let label = '';
  let ready = !!target;
  let action: () => void = () => {
    if (target) void runTool(tool, target, summary);
  };
  switch (tool) {
    case 'across':
      label = `Caliper across the ${DIR_NAME[across]}`;
      break;
    case 'extent':
      label = `Overall size along the ${DIR_NAME[extent]}`;
      break;
    case 'steps':
      label = `Find the faces along ${DIR_NAME[steps]}`;
      break;
    case 'diameter':
      label = has ? 'Measure diameter' : 'Select the round surface first';
      ready &&= has;
      break;
    case 'flatness':
      label = has ? 'Measure flatness' : 'Select a flat face first';
      ready &&= has;
      break;
    case 'sphere':
      label = has ? 'Fit the sphere' : 'Select the ball first';
      ready &&= has;
      break;
    case 'angle':
      if (!keptA) {
        label = has ? 'Keep as face A' : 'Select the first face';
        action = () => {
          if (target) keepFaceA(target);
        };
      } else label = has ? 'Measure the angle' : 'Now select face B';
      ready &&= has;
      break;
    case 'section':
      label = 'Cut here';
      break;
  }
  if (!target) label = 'Pick a model to measure';
  return (
    <Button variant="primary" size="lg" block icon={<Glyph size={18} />} loading={busy === tool} disabled={!ready || (!!busy && busy !== tool)} onClick={action}>
      {label}
    </Button>
  );
}
