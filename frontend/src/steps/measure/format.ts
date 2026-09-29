import { fmtLen } from '../../lib/format';
import type { Direction } from '../../lib/measureTools';

/** How a dimension value reads: lengths to the micrometre (3 decimals in mm), angles to 0.01°. */
export const valueText = (d: { kind: string; value: number | null }) => (d.value == null || !Number.isFinite(d.value) ? '–' : d.kind === 'angle' ? d.value.toFixed(2) : fmtLen(d.value, 3));

export const unitText = (d: { kind: string; unit: string }) => (d.kind === 'angle' ? '°' : d.unit);

/** Ø in front of diameters, as on a drawing. */
export const prefixOf = (kind: string) => (kind === 'diameter' || kind === 'sphere' ? 'Ø ' : '');

export const dimText = (d: { kind: string; value: number | null; unit: string }) => `${prefixOf(d.kind)}${valueText(d)}${d.kind === 'angle' ? '°' : ` ${d.unit}`}`;

export const DIRS: { value: Direction; label: string; title: string }[] = [
  { value: 'length', label: 'Length', title: 'The longest direction of the part itself' },
  { value: 'width', label: 'Width', title: 'Across the part: its second-longest direction' },
  { value: 'height', label: 'Height', title: 'The shortest direction of the part' },
  { value: 'x', label: 'X', title: 'The scanner’s X axis' },
  { value: 'y', label: 'Y', title: 'The scanner’s Y axis' },
  { value: 'z', label: 'Z', title: 'The scanner’s Z axis' },
];

export const DIR_NAME: Record<Direction, string> = { length: 'the length', width: 'the width', height: 'the height', x: 'X', y: 'Y', z: 'Z' };

export const fmtDeg = (v: number | null | undefined, digits = 2) => (v == null || !Number.isFinite(v) ? '–' : `${v.toFixed(digits)}°`);

export const fmtPpm = (v: number | null | undefined, signed = true) => (v == null || !Number.isFinite(v) ? '–' : `${signed && v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(Math.round(v)).toLocaleString()}`);
