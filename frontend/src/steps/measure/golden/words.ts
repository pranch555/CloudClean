import { fmtLen, fmtSigned } from '../../../lib/format';
import type { GoldenMeasurement } from '../../../lib/golden';

/* Plain words for Measure -> Golden model: what each colour means, and what a size's difference says. */

/** The colours of the check (cloudclean/golden.py LEGEND keys): a name and one line anyone understands. */
export const SHARE: Record<string, { label: string; line: (tol: string) => string }> = {
  good: { label: 'Matches', line: tol => `The scan sits within ${tol} of the golden model here.` },
  missing: { label: 'Not scanned', line: () => 'No scan points landed here: the scanner never saw these surfaces.' },
  thin: { label: 'Too few points', line: () => 'Only a few points landed here: the scanner saw it at a steep angle.' },
  rough: { label: 'Rough', line: () => 'The points scatter here, usually from a shiny or dark surface.' },
  off_out: { label: 'More material', line: tol => `The scan sits outside the golden surface by more than ${tol}: there is extra material.` },
  off_in: { label: 'Less material', line: tol => `The scan sits inside the golden surface by more than ${tol}: material is missing.` },
};

/** Legend key -> the check_status code painted on the check mesh. */
export const STATUS_CODE: Record<string, number> = { good: 0, missing: 1, thin: 2, rough: 3, off_out: 4, off_in: 5 };

export const AREA_STATUS: Record<string, string> = {
  missing: 'Not scanned',
  thin: 'Too few points',
  rough: 'Rough',
  off_out: 'More material',
  off_in: 'Less material',
};

export const MEAS_STATUS: Record<GoldenMeasurement['status'], { label: string; tone: 'pass' | 'fail' | 'warn' | 'none' }> = {
  off: { label: 'Off', tone: 'fail' },
  close: { label: 'Too close to call', tone: 'warn' },
  ok: { label: 'Matches', tone: 'pass' },
  not_measured: { label: 'Not measured', tone: 'none' },
};

/** Comparative words for a size (v3 reports carry their own; older ones fall back on the kind). */
export function compareWords(m: GoldenMeasurement): { more: string; less: string } {
  if (m.more && m.less) return { more: m.more, less: m.less };
  switch (m.kind) {
    case 'thickness':
      return { more: 'thicker', less: 'thinner' };
    case 'gap':
      return { more: 'wider', less: 'narrower' };
    case 'step':
      return { more: 'longer', less: 'shorter' };
    default:
      return { more: 'bigger', less: 'smaller' };
  }
}

/** The heading a size goes under. */
export const groupOf = (m: GoldenMeasurement): string =>
  m.group ?? (m.kind === 'size' ? 'Overall size' : m.kind === 'diameter' || m.kind === 'position' ? 'Holes and round faces' : 'Between faces');

/** "1.98 mm thinner", "0.16 mm thinner", "exact", for one size. */
export function differenceWords(m: GoldenMeasurement, units: string): string {
  if (m.status === 'not_measured' || m.difference == null) return 'Not measured';
  if (m.kind === 'position') {
    const off = m.scan ?? 0;
    return off < 0.0005 ? 'Right where it should be' : `${fmtLen(off, 2)} ${units} off${m.toward ? `, ${m.toward.startsWith('to ') || /^(up|down|sideways)/.test(m.toward) ? m.toward : `toward ${m.toward}`}` : ''}`;
  }
  const d = m.difference;
  if (Math.abs(d) < 0.005) return 'Exactly as designed';
  const { more, less } = compareWords(m);
  return `${fmtLen(Math.abs(d), 2)} ${units} ${d > 0 ? more : less}`;
}

/** One sentence for the open row: the two values, in words. */
export function storyOf(m: GoldenMeasurement, units: string, tolText: string): string {
  if (m.status === 'not_measured') return m.reason ?? 'This size could not be measured on the scan.';
  if (m.kind === 'position') {
    return (m.scan ?? 0) < 0.0005
      ? 'The scan has it exactly where the golden model does.'
      : `Its centre sits ${fmtLen(m.scan, 2)} ${units} from where the golden model has it${m.toward ? ` (${m.toward})` : ''}. You allowed ${tolText}.`;
  }
  const d = m.difference ?? 0;
  const head = Math.abs(d) < 0.005 ? 'Your scan measures it exactly as designed' : `Your scan measures it ${differenceWords(m, units)} than the golden model`;
  return `${head}: ${fmtLen(m.golden, 3)} ${units} designed, ${fmtLen(m.scan, 3)} ${units} scanned. You allowed ${tolText}.`;
}

export const signedDiff = (m: GoldenMeasurement) => {
  if (m.kind === 'position') return fmtLen(m.scan, 2);
  const d = Math.abs(m.difference ?? 0) < 0.0005 ? 0 : m.difference;
  return fmtSigned(d, 2);
};

export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
export const pct = (v: number) => (v > 0 && v < 0.1 ? '<0.1' : v > 99.9 && v < 100 ? '99.9' : v.toFixed(1));
