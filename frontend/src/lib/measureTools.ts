import { useEffect } from 'react';
import { create } from 'zustand';
import { api, ApiError } from './api';
import { jobStarted, submitJob } from './jobs';
import { dimNotes, nextMeasureLabel } from './measure';
import { uid } from './uid';
import type { Asset, Job } from './types';
import { useStore, type DimLine, type Vec3 } from '../store';
import { getViewer } from '../viewer/instance';

/*
 * Clients for the measuring tools (docs/v3-plan.md Contract 3) and the accuracy checks (Contract 5), plus the small
 * amount of shared state the Measure step needs: which of those server features exist, the facts behind every
 * dimension drawn on the model, and cached accuracy results.
 *
 * Nothing here changes a model: every request only measures. The one exception is `compareScans`, which starts a
 * server job (it still writes no geometry).
 */

// ------------------------------------------------------------------------------------------------ types (Contract 3)
/** A direction D (Contract 2): the part's own axes (longest → shortest) or the scanner's axes. */
export type Direction = 'length' | 'width' | 'height' | 'x' | 'y' | 'z';
export type PartAxis = 'length' | 'width' | 'height';
/** A region of Contract 2: screen box / lasso, box, obb, cylinder, slab, end, all … */
export type Region = Record<string, unknown>;

export interface ExtentResult {
  length: number;
  robust_length: number;
  min: number;
  max: number;
  robust_min?: number;
  robust_max?: number;
  direction: number[];
  direction_name?: string;
  frame?: 'part' | 'world';
  a: Vec3;
  b: Vec3;
  points_used: number;
}

export interface CaliperFace {
  kind?: 'plane' | 'contact';
  point: number[] | null;
  normal: number[];
  rms: number | null;
  points: number;
  flatness?: number;
  angle_to_direction_deg?: number;
}

export interface CaliperResult {
  distance: number;
  parallelism_deg: number;
  face_a: CaliperFace;
  face_b: CaliperFace;
  a: Vec3;
  b: Vec3;
  extent?: number;
  direction?: number[];
  direction_name?: string;
  points_used?: number;
  warnings?: string[];
}

export interface FaceFound {
  /** mm from the part's start along the direction */
  position: number;
  point: Vec3;
  normal: number[];
  tilt_deg: number;
  rms: number;
  flatness: number;
  points: number;
  /** +1: faces towards the end of the direction, -1: towards its start, null: unknown */
  facing: 1 | -1 | null;
  radius_range: [number, number];
  size_across: [number, number];
  /** what the face is: end face, shoulder, floor of a recess */
  label?: string;
}

export interface FacesResult {
  direction: number[];
  direction_name: string;
  faces: FaceFound[];
  steps: { from: number; to: number; distance: number }[];
  /** the distances people ask for: head height, length under the head, recess depth, overall */
  key_distances?: { what: string; from: number; to: number; distance: number }[];
  overall: number;
  a: Vec3;
  b: Vec3;
  centre_line: { point: Vec3; direction: number[] };
  points_used: number;
  warnings?: string[];
}

export interface DiameterResult {
  kind?: 'cylinder' | 'circle';
  diameter: number;
  radius: number;
  axis: number[];
  center: number[];
  length: number;
  rms: number;
  coverage_deg: number;
  points_used: number;
  inliers?: number;
  points_in_region?: number;
  a: Vec3;
  b: Vec3;
  warnings?: string[];
}

export interface SphereResult {
  center: number[];
  radius: number;
  diameter: number;
  rms: number;
  coverage?: number;
  points_used: number;
  a?: Vec3;
  b?: Vec3;
  warnings?: string[];
}

export interface PlaneResult {
  point: number[];
  normal: number[];
  rms: number;
  flatness: number;
  points_used: number;
  a?: Vec3;
  b?: Vec3;
  warnings?: string[];
}

export interface AngleFit {
  type: 'plane' | 'line';
  normal?: number[];
  direction?: number[];
  point: number[];
  rms: number;
  flatness?: number;
  length?: number;
  points_used: number;
}

export interface AngleResult {
  angle_deg: number;
  supplement_deg?: number;
  normals_angle_deg?: number;
  fit_a: AngleFit;
  fit_b: AngleFit;
  a?: Vec3;
  b?: Vec3;
}

export interface SectionResult {
  polylines: number[][][];
  closed?: boolean[];
  width: number;
  height: number;
  bbox2d: { min: number[]; max: number[] };
  basis?: { origin: number[]; u: number[]; v: number[]; normal: number[] };
  length?: number;
  points_used: number;
  plane_label?: string;
}

// ------------------------------------------------------------------------------------------------ server features
/**
 * Which server features exist. The production server may be older than this UI: a missing route answers 404 "Not
 * Found" (or 405 when only the page's GET fallback matches the path). Detected once, quietly, and shown as "not
 * available on this server yet" instead of errors.
 */
export type Cap = 'unknown' | 'yes' | 'no';
export type CapGroup = 'measure' | 'accuracy';

export const useCaps = create<Record<CapGroup, Cap>>(() => ({ measure: 'unknown', accuracy: 'unknown' }));

// POST {} is answered by validation (422) when the route exists, so probing never runs anything.
const PROBE: Record<CapGroup, string> = { measure: '/api/measure/extent', accuracy: '/api/accuracy/reference' };
const probing: Partial<Record<CapGroup, Promise<Cap>>> = {};

export function isMissingRoute(err: unknown): boolean {
  if (!(err instanceof ApiError)) return false;
  return err.status === 405 || (err.status === 404 && /^not found$/i.test(err.message.trim()));
}

export class Unavailable extends Error {
  constructor() {
    super('Not available on this server yet');
  }
}

function setCap(group: CapGroup, cap: Cap) {
  if (useCaps.getState()[group] !== cap) useCaps.setState({ [group]: cap } as Partial<Record<CapGroup, Cap>>);
}

export function probeCap(group: CapGroup): Promise<Cap> {
  const cur = useCaps.getState()[group];
  if (cur !== 'unknown') return Promise.resolve(cur);
  probing[group] ??= api.post(PROBE[group], {}).then(
    () => {
      setCap(group, 'yes');
      return 'yes' as Cap;
    },
    err => {
      if (isMissingRoute(err)) setCap(group, 'no');
      else if (err instanceof ApiError && err.status > 0) setCap(group, 'yes');
      const cap = useCaps.getState()[group];
      if (cap === 'unknown') delete probing[group]; // server unreachable: ask again later
      return cap;
    },
  );
  return probing[group]!;
}

/** Whether a group of server features exists (probes once, in the background). */
export function useCap(group: CapGroup): Cap {
  const cap = useCaps(s => s[group]);
  useEffect(() => {
    if (cap === 'unknown') void probeCap(group);
  }, [cap, group]);
  return cap;
}

async function call<T>(group: CapGroup, url: string, body: unknown): Promise<T> {
  try {
    const res = await api.post<T>(url, body);
    setCap(group, 'yes');
    return res;
  } catch (err) {
    if (isMissingRoute(err)) {
      setCap(group, 'no');
      throw new Unavailable();
    }
    if (err instanceof ApiError && err.status > 0) setCap(group, 'yes');
    throw err;
  }
}

// ------------------------------------------------------------------------------------------------ measuring (Contract 3)
const opt = <T,>(v: T | null | undefined) => (v == null ? undefined : v);

export const measureExtent = (assetId: string, direction: Direction, region?: Region | null) =>
  call<ExtentResult>('measure', '/api/measure/extent', { asset_id: assetId, direction, region: opt(region) });

export const measureCaliper = (assetId: string, direction: Direction, region?: Region | null) =>
  call<CaliperResult>('measure', '/api/measure/caliper', { asset_id: assetId, direction, region: opt(region) });

export const measureFaces = (assetId: string, direction: Direction, region?: Region | null) =>
  call<FacesResult>('measure', '/api/measure/faces', { asset_id: assetId, direction, region: opt(region) });

export const measureDiameter = (assetId: string, region?: Region | null, axisHint?: PartAxis | null) =>
  call<DiameterResult>('measure', '/api/measure/diameter', { asset_id: assetId, region: opt(region), axis_hint: opt(axisHint) });

export const measureSphere = (assetId: string, region: Region) => call<SphereResult>('measure', '/api/measure/sphere', { asset_id: assetId, region });

export const measurePlane = (assetId: string, region: Region) => call<PlaneResult>('measure', '/api/measure/plane', { asset_id: assetId, region });

export const measureAngle = (assetId: string, regionA: Region, regionB: Region) =>
  call<AngleResult>('measure', '/api/measure/angle', { asset_id: assetId, region_a: regionA, region_b: regionB });

export const measureSection = (assetId: string, plane: { direction: Direction; at: number } | { point: number[]; normal: number[] }) =>
  call<SectionResult>('measure', '/api/measure/section', { asset_id: assetId, plane });

// ------------------------------------------------------------------------------------------------ regions from the selection
/** The current box / lasso selection (or a region the assistant highlighted) as a Contract 2 region. */
export function selectionRegion(facingOnly = false): Region | null {
  const sel = useStore.getState().selection;
  if (!sel) return null;
  if (sel.region) return sel.region;
  if (sel.view_projection.length !== 16 || sel.polygon.length < 3) return null;
  return { view_projection: sel.view_projection, polygon: sel.polygon, visible_only: facingOnly };
}

/** How much of the current selection lies on a model, and on which other models. */
export function useSelectionOn(target: Asset | undefined): { count: number; others: string[]; label: string | null } | null {
  const selection = useStore(s => s.selection);
  const counts = useStore(s => s.selectionCounts);
  if (!selection) return null;
  const count = target ? counts[target.id] ?? 0 : 0;
  return { count, others: Object.keys(counts).filter(id => id !== target?.id && counts[id] > 0), label: selection.label ?? null };
}

/** A region kept for a later step (face A of an angle, ball A of a ball bar). */
export interface RegionPick {
  region: Region;
  count: number;
  assetId: string;
}

export function takeSelection(target: Asset, facingOnly = false): RegionPick | null {
  const region = selectionRegion(facingOnly);
  const count = useStore.getState().selectionCounts[target.id] ?? 0;
  return region && count ? { region, count, assetId: target.id } : null;
}

// ------------------------------------------------------------------------------------------------ dimensions on the model
export interface DimSpec {
  kind: string;
  value: number | null;
  a: Vec3;
  b: Vec3;
  unit?: string;
}

/** What a dimension is and how it was measured, shown next to it in the results list. */
export interface DimInfo {
  id: string;
  tool: string;
  title: string;
  detail: string;
  warnings: string[];
  created: number;
}

export const useDimInfo = create<{ byId: Record<string, DimInfo> }>(() => ({ byId: {} }));

/**
 * Draw dimensions on the model (store.dims, drawn by viewport/Labels) with the next free labels D1, D2 … and
 * remember what they are. `title` / `detail` may be one per dimension.
 */
export function addDims(specs: DimSpec[], info: { tool: string; title: string | string[]; detail?: string | string[]; warnings?: string[]; assetId?: string }): DimLine[] {
  const st = useStore.getState();
  const taken: string[] = [];
  const lines: DimLine[] = specs.map(s => {
    const label = nextMeasureLabel(taken);
    taken.push(label);
    return { id: uid(), label, kind: s.kind, value: s.value, unit: s.unit ?? (s.kind === 'angle' ? '°' : st.display.units), a: s.a, b: s.b, assetId: info.assetId, source: 'tool' };
  });
  const now = Date.now();
  const pick = (v: string | string[] | undefined, i: number) => (Array.isArray(v) ? v[i] ?? v[0] ?? '' : v ?? '');
  const entries = lines.map((l, i) => {
    const d: DimInfo = { id: l.id, tool: info.tool, title: pick(info.title, i), detail: pick(info.detail, i), warnings: info.warnings ?? [], created: now };
    dimNotes.set(l.id, [d.title, d.detail, ...d.warnings].filter(Boolean).join('; '));
    return [l.id, d] as const;
  });
  useDimInfo.setState(s => ({ byId: { ...s.byId, ...Object.fromEntries(entries) } }));
  useStore.setState(s => ({ dims: [...s.dims, ...lines] }));
  return lines;
}

export function removeDims(ids: string[]) {
  useStore.setState(s => ({ dims: s.dims.filter(d => !ids.includes(d.id)) }));
}

const sub3 = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross3 = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a: number[]) => {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};

/** A diameter line through `center`, across `axis`, turned so it reads across the screen from the current view. */
export function acrossLine(center: number[], axis: number[], radius: number): [Vec3, Vec3] {
  const ax = norm3(axis);
  let e: number[] | null = null;
  const cam = getViewer()?.cameraState();
  if (cam) {
    const c = cross3(ax, norm3(sub3(cam.target, cam.position)));
    if (Math.hypot(c[0], c[1], c[2]) > 1e-6) e = norm3(c);
  }
  if (!e) e = norm3(cross3(ax, Math.abs(ax[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  const p = (s: number) => [0, 1, 2].map(i => center[i] + s * radius * e![i]) as Vec3;
  return [p(-1), p(1)];
}

/** Bring a dimension into view: centre it and zoom so it spans about a third of the view. */
export function showOnModel(a: Vec3, b: Vec3, assetId?: string | null) {
  const st = useStore.getState();
  if (assetId && st.byId.has(assetId) && !st.visible.includes(assetId)) useStore.setState({ visible: [...st.visible, assetId] });
  const v = getViewer();
  if (!v) return;
  v.lookAtPoint([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]);
  window.setTimeout(() => {
    const pa = v.project(a), pb = v.project(b);
    if (!pa || !pb) return;
    const px = Math.hypot(pa.x - pb.x, pa.y - pb.y);
    const want = 0.36 * Math.min(v.canvas.clientWidth, v.canvas.clientHeight);
    if (px > 2 && Math.abs(want / px - 1) > 0.15) v.zoomBy(Math.min(6, Math.max(0.2, want / px)));
  }, 460);
}

/** Copy text; works on plain-HTTP lab addresses too, where the async clipboard API does not exist. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall back below */
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  }
  ta.remove();
  return ok;
}

export function downloadText(text: string, name: string, type = 'text/csv') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

// ------------------------------------------------------------------------------------------------ accuracy (Contract 5)
export interface DistStats {
  signed_mean: number;
  mean: number;
  rms: number;
  p95: number;
  max: number;
  median?: number;
  median_abs?: number;
  std?: number;
  p05?: number;
  p95_signed?: number;
  count?: number;
}

export type Dims3 = { length: number; width: number; height: number } | number[];

export function dims3(d: Dims3 | null | undefined): [number, number, number] | null {
  if (!d) return null;
  if (Array.isArray(d)) return d.length >= 3 && d.slice(0, 3).every(x => typeof x === 'number') ? [d[0], d[1], d[2]] : null;
  return typeof d.length === 'number' ? [d.length, d.width, d.height] : null;
}

export interface DriftResult {
  asset_id?: string;
  parent_id?: string | null;
  operation?: string;
  verdict: 'unchanged' | 'moved' | 'changed' | string;
  sentence: string;
  tolerance?: number;
  displacement?: DistStats | null;
  moved_points?: DistStats | null;
  unsupported_fraction?: number;
  identical_fraction?: number | null;
  kept_fraction?: number | null;
  reverse?: DistStats | null;
  transform_check?: { rigid?: boolean; identity?: boolean; rotation_deg?: number; determinant?: number; scale_ppm?: number; mirrored?: boolean } | null;
  dimensions_before?: Dims3;
  dimensions_after?: Dims3;
  dimension_change?: Dims3;
  points?: { parent: number; child: number };
}

export type DriftState = { status: 'loading' } | { status: 'done'; drift: DriftResult } | { status: 'error'; message: string } | { status: 'unavailable' };

export const useDrifts = create<{ byId: Record<string, DriftState> }>(() => ({ byId: {} }));

const putDrift = (id: string, s: DriftState) => useDrifts.setState(x => ({ byId: { ...x.byId, [id]: s } }));

/** How far the operation that made a model moved or resized its parent's surface. Cached per model. */
export async function loadDrift(assetId: string, force = false): Promise<DriftState> {
  const cur = useDrifts.getState().byId[assetId];
  if (cur && !force && cur.status !== 'error') return cur;
  if (useCaps.getState().accuracy === 'no') {
    putDrift(assetId, { status: 'unavailable' });
    return { status: 'unavailable' };
  }
  putDrift(assetId, { status: 'loading' });
  let next: DriftState;
  try {
    next = { status: 'done', drift: await call<DriftResult>('accuracy', '/api/accuracy/drift', { asset_id: assetId }) };
  } catch (err) {
    next = err instanceof Unavailable ? { status: 'unavailable' } : { status: 'error', message: (err as Error).message };
  }
  putDrift(assetId, next);
  return next;
}

/** True for models nothing has processed yet (their size is what the scanner delivered). */
export const isOriginal = (a: Asset) => !a.parents.length || a.operation === 'import' || a.operation === 'capture';

export interface CompareScansResult {
  verdict: 'consistent' | 'differ' | 'uncertain' | string;
  tolerance?: number;
  scale_ppm: number;
  scale_se_ppm?: number;
  scale_with_offset_ppm?: number;
  offset_mm?: number;
  offset_se_mm?: number | null;
  axis_scale_ppm?: Partial<Record<PartAxis, number>>;
  axis_scale_se_ppm?: Partial<Record<PartAxis, number>>;
  length_difference_from_scale_mm?: number;
  separation?: DistStats | null;
  separation_histogram?: { edges: number[]; counts: number[]; underflow?: number; overflow?: number };
  overlap_fraction?: number;
  noise?: { a: number; b: number; combined: number };
  spacing?: number;
  extents?: { a: Dims3; b: Dims3; difference: Dims3 };
  extents_common?: { a: Dims3; b: Dims3; difference: Dims3 } | null;
  alignment?: { ambiguous?: boolean; best_candidate?: string };
  sentence?: string;
}

/** The object holding `key` in a job's output (the output may hold it directly or one level down). */
export function findResult<T>(output: unknown, key: string, depth = 2): T | null {
  if (!output || typeof output !== 'object') return null;
  if (key in (output as object)) return output as T;
  if (depth <= 0) return null;
  for (const v of Object.values(output as Record<string, unknown>)) {
    const r = findResult<T>(v, key, depth - 1);
    if (r) return r;
  }
  return null;
}

export interface CompareRun {
  aId: string;
  bId: string;
  status: 'running' | 'done' | 'failed' | 'unavailable';
  jobId?: string;
  result?: CompareScansResult;
  error?: string;
}

export const useCompareScans = create<{ run: CompareRun | null }>(() => ({ run: null }));

/** Two scans of the same part, lined up without changing their size: how consistent is the scanner? (a job) */
export async function compareScans(aId: string, bId: string) {
  const set = (run: CompareRun) => useCompareScans.setState({ run });
  if ((await probeCap('accuracy')) === 'no') return set({ aId, bId, status: 'unavailable' });
  set({ aId, bId, status: 'running' });
  const job = await submitJob('/api/accuracy/compare-scans', { a_id: aId, b_id: bId }, done => {
    const result = findResult<CompareScansResult>(done.output, 'scale_ppm');
    if (done.status === 'done' && result) set({ aId, bId, jobId: done.id, status: 'done', result });
    else set({ aId, bId, jobId: done.id, status: 'failed', error: done.status === 'done' ? 'The comparison finished without a result.' : done.error || `The comparison was ${done.status}.` });
  });
  if (!job) return set({ aId, bId, status: 'failed', error: 'The comparison could not be started.' });
  useCompareScans.setState(s => (s.run && s.run.status === 'running' && !s.run.jobId ? { run: { ...s.run, jobId: job.id } } : s));
}

export interface ReferenceResult {
  type: string;
  measured: number;
  nominal: number;
  error: number;
  error_ppm: number;
  error_pct: number;
  uncertainty?: number | null;
  tolerance?: number;
  verdict: 'ok' | 'marginal' | 'off' | string;
  significant?: boolean;
  sentence: string;
  details?: Record<string, unknown>;
}

export type ReferenceSpec =
  | { type: 'known_length'; direction: Direction; region?: Region; nominal: number; tolerance?: number }
  | { type: 'diameter'; region: Region; nominal: number; tolerance?: number }
  | { type: 'sphere_pair'; region_a: Region; region_b: Region; nominal: number; tolerance?: number };

/** The acceptance the server assumes when none is given: 0.02 mm + 100 ppm of the nominal. */
export const defaultTolerance = (nominal: number) => 0.02 + 1e-4 * Math.abs(nominal);

const isJob = (x: unknown): x is Job => !!x && typeof x === 'object' && 'status' in x && 'kind' in x && 'logs' in x;

/** Measure a known artefact and compare with its certified size (answers directly, or through a job). */
export async function checkReference(assetId: string, reference: ReferenceSpec): Promise<ReferenceResult> {
  const res = await call<ReferenceResult | Job>('accuracy', '/api/accuracy/reference', { asset_id: assetId, reference });
  if (!isJob(res)) return res;
  return new Promise((resolve, reject) => {
    jobStarted(res, done => {
      const r = findResult<ReferenceResult>(done.output, 'measured');
      if (done.status === 'done' && r) resolve(r);
      else reject(new Error(done.error || 'The check did not finish.'));
    });
  });
}

/** Where to draw a reference measurement on the model, from the details the server returns. */
export function referenceLine(r: ReferenceResult): [Vec3, Vec3] | null {
  const d = r.details ?? {};
  const v3 = (x: unknown): Vec3 | null => (Array.isArray(x) && x.length === 3 && x.every(n => typeof n === 'number') ? (x as Vec3) : null);
  const a = v3(d.a), b = v3(d.b);
  if (a && b) return [a, b];
  const point = v3(d.point) ?? v3(d.center), axis = v3(d.axis);
  if (point && axis && typeof d.radius === 'number') return acrossLine(point, axis, d.radius);
  return null;
}
