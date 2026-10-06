import { create } from 'zustand';

/* Golden model check: how a result reads at a glance (Measure -> Golden model, the model list, the report). */

export type GoldenGradeKey = 'match' | 'mostly' | 'differs' | 'incomplete';

export interface GoldenGrade {
  key: GoldenGradeKey;
  /** short label for chips and headings */
  label: string;
  tone: 'pass' | 'warn' | 'fail';
  /** share of the golden surface that matches within the tolerance, 0-100 (null when unknown) */
  matchPct: number | null;
}

/** At or above this share of matching surface a check that found problems is "mostly matches", not a failure. */
export const MOSTLY_MATCHES_PCT = 95;

interface GradeInput {
  verdict?: string;
  surface?: { shares_pct?: Record<string, number> };
}

export function goldenGrade(r: GradeInput | null | undefined): GoldenGrade {
  const good = r?.surface?.shares_pct?.good;
  const matchPct = typeof good === 'number' && Number.isFinite(good) ? good : null;
  if (r?.verdict === 'match') return { key: 'match', label: 'Matches', tone: 'pass', matchPct };
  if (r?.verdict === 'incomplete') return { key: 'incomplete', label: 'Scan more', tone: 'warn', matchPct };
  if (matchPct != null && matchPct >= MOSTLY_MATCHES_PCT) return { key: 'mostly', label: 'Mostly matches', tone: 'warn', matchPct };
  return { key: 'differs', label: 'Does not match', tone: 'fail', matchPct };
}

/* ---- the check's report (cloudclean/golden.py, docs/golden-model.md) */
export type Vec3 = [number, number, number];

export interface GoldenRegion {
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
  center: Vec3;
  normal: Vec3;
  /** v2: a point on the area for its numbered pin */
  pin?: Vec3;
  face_kind?: string;
  view: { target: Vec3; from: Vec3; radius?: number };
}

export interface GoldenMeasurement {
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

export interface GoldenReport {
  version?: number;
  verdict: 'match' | 'differs' | 'incomplete';
  headline: string;
  summary: string[];
  tolerance: number;
  match_pct?: number;
  regions: GoldenRegion[];
  measurements: GoldenMeasurement[];
  counts?: Record<string, number>;
  surface: { scanned_pct: number; shares_pct: Record<string, number> };
  legend: { code: number; key: string; label: string; color: string }[];
  compare_asset?: { id: string; name: string };
  warnings?: string[];
}

/** The legend key (and colour) of an area: off areas split into more / less material. */
export const regionKey = (r: Pick<GoldenRegion, 'kind' | 'sign'>) => (r.kind === 'off' ? (r.sign > 0 ? 'off_out' : 'off_in') : r.kind);

/* ---- numbered pins on the 3D view (viewport/GoldenPins.tsx), kept in step with the list in the panel */
export interface GoldenPin {
  n: number;
  region: number;
  pos: Vec3;
  normal: Vec3;
  color: string;
  label: string;
}

interface PinState {
  /** the golden_check asset the pins sit on (null: no pins) */
  checkId: string | null;
  pins: GoldenPin[];
  /** the area "Show me" is on (region id) */
  active: number | null;
  /** a pin was clicked: GoldenCheck scrolls to that area's card and shows it */
  picked: { region: number; at: number } | null;
}

export const useGoldenPins = create<PinState>(() => ({ checkId: null, pins: [], active: null, picked: null }));
