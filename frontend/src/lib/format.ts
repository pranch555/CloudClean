import type { Asset } from './types';

export const fmtCount = (n: number | undefined | null): string => {
  if (n == null) return '–';
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k`;
  return String(n);
};

/** Lengths keep enough significant digits for metrology without noise. */
export const fmtLen = (v: number | undefined | null, digits?: number): string => {
  if (v == null || Number.isNaN(v)) return '–';
  if (digits != null) return v.toFixed(digits);
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
};

export const fmtSigned = (v: number | undefined | null, digits = 3): string =>
  v == null || Number.isNaN(v) ? '–' : `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(digits)}`;

export const fmtPct = (v: number | undefined | null, digits = 1): string =>
  v == null || Number.isNaN(v) ? '–' : `${(v * 100).toFixed(digits)}%`;

export const fmtDims = (d: number[] | undefined): string => (d ? d.map(x => fmtLen(x)).join(' × ') : '–');

export const fmtDuration = (s: number): string => {
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
};

export const fmtAgo = (iso: string): string => {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
};

export const humanize = (k: string): string => k.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());

export const countLabel = (a: Asset): string =>
  a.kind === 'mesh' ? `${fmtCount(a.stats.triangles)} tris` : a.kind === 'image' ? `${a.stats.width}×${a.stats.height}` : `${fmtCount(a.stats.points)} pts`;

export const OPERATION_LABEL: Record<string, string> = {
  import: 'Imported',
  clean: 'Cleaned',
  merge: 'Merged',
  mesh: 'Meshed',
  texture: 'Coloured',
  edit: 'Edited',
  compare: 'Inspection',
  capture: 'Captured',
  autopilot: 'Autopilot',
  photos: 'Made from photos',
};

export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
