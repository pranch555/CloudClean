import { api, ApiError } from './api';
import { uid } from './uid';
import { useStore, type DistanceResult, type Measurement, type Vec3 } from '../store';

const PALETTE = ['#ee4b1f', '#2f6fd6', '#178f64', '#8a57d4', '#cc4577', '#a88100'];
export const measureColor = (i: number) => PALETTE[i % PALETTE.length];

/** Axis the distances are projected on: a world axis or the fitted thread axis. */
export function currentAxis(): Vec3 | 'x' | 'y' | 'z' | null {
  const { measureAxis, thread } = useStore.getState();
  if (measureAxis === 'none') return null;
  if (measureAxis === 'thread') return thread ? thread.axis : null;
  return measureAxis;
}

function localResult(a: Vec3, b: Vec3): DistanceResult {
  const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
  const out: DistanceResult = { distance: Math.hypot(dx, dy, dz), dx, dy, dz };
  const axis = currentAxis();
  if (axis) {
    const v: Vec3 = axis === 'x' ? [1, 0, 0] : axis === 'y' ? [0, 1, 0] : axis === 'z' ? [0, 0, 1] : axis;
    const n = Math.hypot(...v) || 1;
    const along = (dx * v[0] + dy * v[1] + dz * v[2]) / n;
    out.along_axis = Math.abs(along);
    out.perpendicular = Math.sqrt(Math.max(out.distance ** 2 - along ** 2, 0));
  }
  return out;
}

function patch(id: string, p: Partial<Measurement>) {
  useStore.setState(s => ({ measurements: s.measurements.map(m => (m.id === id ? { ...m, ...p } : m)) }));
}

let surfaceSnap = true;

/**
 * Snap both points to full-resolution data on the server and compute distances (along the chosen axis too).
 * Newer servers move each click onto a surface fitted to its nearest scan points, which scatters ~2.5x less than
 * the nearest raw point and comes with an uncertainty; older ones snap to the nearest point.
 */
export async function resolveMeasurement(m: Measurement) {
  if (!m.b) return;
  patch(m.id, { result: localResult(m.a, m.b) });
  if (!m.assetId) return;
  if (surfaceSnap) {
    try {
      const s = await api.post<{ points: Vec3[]; uncertainty_mm: (number | null)[] }>('/api/measure/snap', { asset_id: m.assetId, points: [m.a, m.b] });
      const [a, b] = s.points;
      const result = localResult(a, b);
      const u = s.uncertainty_mm ?? [];
      if (u[0] != null && u[1] != null) result.uncertainty = Math.hypot(u[0], u[1]);
      patch(m.id, { a, b, snapped: true, result });
      return;
    } catch (err) {
      if (err instanceof ApiError && (err.status === 405 || (err.status === 404 && /^not found$/i.test(err.message.trim())))) surfaceSnap = false;
    }
  }
  try {
    const axis = currentAxis();
    const res = await api.post<DistanceResult & { points: Vec3[] }>('/api/measure/distance', { asset_id: m.assetId, points: [m.a, m.b], axis: axis ?? undefined });
    patch(m.id, { a: res.points[0], b: res.points[1], snapped: true, result: { distance: res.distance, dx: res.dx, dy: res.dy, dz: res.dz, along_axis: res.along_axis, perpendicular: res.perpendicular } });
  } catch {
    /* keep the on-screen picks */
  }
}

/**
 * Next free dimension label (D1, D2 …). Two-point measurements and the measuring tools share one sequence so no two
 * dimensions on the model carry the same name; numbers are not reused until everything is cleared.
 */
export function nextMeasureLabel(taken: string[] = []): string {
  const st = useStore.getState();
  let n = 0;
  for (const label of [...st.measurements.map(m => m.label), ...st.dims.map(d => d.label), ...taken]) {
    const m = /^D(\d+)$/.exec(label);
    if (m) n = Math.max(n, Number(m[1]));
  }
  return `D${n + 1}`;
}

/** A click on the surface: finish the open measurement, or start a new one. */
export function addMeasurePoint(point: Vec3, assetId: string | null) {
  const st = useStore.getState();
  const open = st.measurements.find(m => !m.b);
  if (open) {
    const done = { ...open, b: point, assetId: open.assetId ?? assetId };
    useStore.setState({ measurements: st.measurements.map(m => (m.id === open.id ? done : m)) });
    resolveMeasurement(done);
    return;
  }
  useStore.setState({ measurements: [...st.measurements, { id: uid(), label: nextMeasureLabel(), assetId, a: point, b: null, snapped: false, result: null }] });
}

export function remeasureAll() {
  for (const m of useStore.getState().measurements) if (m.b) resolveMeasurement({ ...m, snapped: false });
}

/** Extra facts about a tool dimension (fit quality, coverage …) for the CSV; filled by the measuring tools. */
export const dimNotes = new Map<string, string>();

/** Dimension lines that are not measurements (the part-size lines drawn from the size readout). */
export const isSizeLine = (id: string) => id.startsWith('size-');

const csvCell = (v: unknown) => {
  const s = typeof v === 'number' ? (Number.isFinite(v) ? v.toFixed(5) : '') : v == null ? '' : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** Every measurement on the model: two-point distances and the results of the measuring tools / assistant. */
export function csvOfMeasurements(units: string): string {
  const st = useStore.getState();
  const name = (id?: string | null) => (id ? st.byId.get(id)?.name ?? '' : '');
  const rows: unknown[][] = [['label', 'kind', 'value', 'unit', 'along_axis', 'perpendicular', 'dx', 'dy', 'dz', 'ax', 'ay', 'az', 'bx', 'by', 'bz', 'pitches', 'pitch', 'model', 'note']];
  for (const m of st.measurements) {
    if (!m.b || !m.result) continue;
    const r = m.result;
    const pitch = m.crests && r.along_axis != null ? r.along_axis / m.crests : '';
    const sigma = r.uncertainty != null ? `; ±${r.uncertainty.toFixed(3)} (1 sigma)` : '';
    const note = (r.along_axis != null ? `along ${st.measureAxis === 'thread' ? 'thread axis' : st.measureAxis}${m.snapped ? '' : '; not snapped'}` : m.snapped ? '' : 'not snapped') + sigma;
    rows.push([m.label, 'distance', r.distance, units, r.along_axis ?? '', r.perpendicular ?? '', r.dx, r.dy, r.dz, ...m.a, ...m.b, m.crests ?? '', pitch, name(m.assetId), note]);
  }
  for (const d of st.dims) {
    if (isSizeLine(d.id)) continue;
    const unit = d.kind === 'angle' ? 'deg' : d.unit;
    rows.push([d.label, d.kind, d.value ?? '', unit, '', '', d.b[0] - d.a[0], d.b[1] - d.a[1], d.b[2] - d.a[2], ...d.a, ...d.b, '', '', name(d.assetId), dimNotes.get(d.id) ?? (d.source === 'assistant' ? 'from the assistant' : '')]);
  }
  return rows.map(r => r.map(csvCell).join(',')).join('\n');
}
