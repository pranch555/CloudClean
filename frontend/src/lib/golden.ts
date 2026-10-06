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
  /** v3: one sentence that lets anyone find the area ("Inside the head: the very bottom of the hex socket.") */
  where?: string;
  /** v3: the number the list and the pins show (1-based) */
  number?: number;
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
  /** golden faces the size is taken between (ids of the check mesh's `check_face` scalar) */
  faces?: number[];
  /* ---- v3: plain words and a place on the model */
  /** one sentence: what this size is, from where to where ("From the top of the head down to the bottom of the hex socket.") */
  what?: string;
  /** comparative words when the scan is bigger / smaller than the golden value ("deeper" / "shallower") */
  more?: string;
  less?: string;
  /** heading this size is listed under ("Overall size", "Head", "Hex socket", "Shaft", "Holes", "Other faces") */
  group?: string;
  /** the dimension line, golden frame: two points on the measured surfaces */
  ends?: [Vec3, Vec3] | null;
  /** why this number deserves care (few points, a listed area on one of its faces), else null */
  caveat?: string | null;
  /** listed areas (region ids) on the faces of this size */
  regions?: number[];
  /** several measurements of the same size of one feature (the 3 widths of a hex socket): one key, shown as one row */
  series?: string;
  series_index?: number;
  series_size?: number;
}

/** v3: a cut through the golden model along its main axis, for the drawing in Sizes (golden-frame mm). */
export interface GoldenSection {
  /** drawing x = (p - origin)·u, drawing y = (p - origin)·v */
  origin: Vec3;
  u: Vec3;
  v: Vec3;
  /** closed outlines of the cut, flattened [x0, y0, x1, y1, ...]; holes are separate loops (even-odd fill) */
  loops: number[][];
  /** [xmin, ymin, xmax, ymax] */
  bounds: [number, number, number, number];
  /** per golden face id used by a measurement: its trace in the cut, as segments [x1, y1, x2, y2] */
  faces: Record<string, number[][]>;
  /** the outline of the whole part seen from the side (silhouette), flattened loops; optional */
  silhouette?: number[][];
}

export interface GoldenReport {
  version?: number;
  /** v3: one or two sentences on the sizes that are off when they share a cause, else null */
  sizes_story?: string | null;
  /** v3: the drawing */
  section?: GoldenSection | null;
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
  /** what the part is, in words (v3 adds `summary`: "A round part with a head and a shaft; a hex socket in the head.") */
  part?: { axis: number | null; ends: string[] | null; up_axis: string; summary?: string; kind?: string };
}

/* ---- what the panel points at on the 3D view (Measure -> Golden model), read by the viewport overlays */
export interface GoldenDimension {
  /** golden frame, the two ends of the line */
  ends: [Vec3, Vec3];
  /** the text on the line ("12.70 → 14.52 mm") */
  label: string;
  tone: 'pass' | 'warn' | 'fail' | 'none';
}

interface FocusState {
  /** a dimension line on the 3D view (a size is hovered or picked), else null */
  dimension: GoldenDimension | null;
}

export const useGoldenFocus = create<FocusState>(() => ({ dimension: null }));

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
