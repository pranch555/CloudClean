import type { Asset, Step } from './types';

const CAD_SOURCE = /\.(step|stp|iges|igs|brep)$/i;

/** What a model is, in the user's words. */
export type Role = 'scan' | 'cleaned' | 'merged' | 'edited' | 'mesh' | 'cad' | 'inspection' | 'photo' | 'fromPhotos' | 'filled';

export function roleOf(a: Asset): Role {
  if (a.kind === 'image') return 'photo';
  const src = String(a.params?.source ?? '');
  if (a.operation === 'import' && CAD_SOURCE.test(src)) return 'cad';
  if (a.operation === 'compare' || a.operation === 'golden_check') return 'inspection';
  if (a.operation === 'photos') return 'fromPhotos';
  if (a.operation === 'import' || a.operation === 'capture') return a.kind === 'mesh' ? 'mesh' : 'scan';
  if (a.operation === 'merge') return 'merged';
  if (a.operation === 'photo_fill') return 'filled';
  if (a.operation === 'clean') return 'cleaned';
  if (a.kind === 'mesh') return 'mesh';
  return 'edited';
}

export const ROLE_LABEL: Record<Role, string> = {
  scan: 'Scan',
  cleaned: 'Cleaned',
  merged: 'Merged',
  edited: 'Edited',
  mesh: 'Mesh',
  cad: 'CAD model',
  inspection: 'Inspection',
  photo: 'Photo',
  fromPhotos: 'From photos',
  filled: 'Filled from photos',
};

export interface StepInfo {
  id: Step;
  n: number;
  label: string;
  verb: string;
  purpose: string;
  optional?: boolean;
}

export const STEPS: StepInfo[] = [
  { id: 'capture', n: 1, label: 'Scan', verb: 'Scan a part', purpose: 'Capture the part with the scanner and turntable, or bring in scan files.' },
  { id: 'clean', n: 2, label: 'Clean', verb: 'Clean the scan', purpose: 'Remove stray points, noise and the table. Your part keeps its size.' },
  { id: 'align', n: 3, label: 'Align', verb: 'Align scans', purpose: 'Combine several scans of the same part — only when it adds surface.', optional: true },
  { id: 'mesh', n: 4, label: 'Mesh', verb: 'Build a mesh', purpose: 'Turn points into a solid surface you can print, measure and export.' },
  { id: 'measure', n: 5, label: 'Measure', verb: 'Measure & inspect', purpose: 'Sizes, diameters, threads, angles — and a check against the golden model.' },
  { id: 'export', n: 6, label: 'Export', verb: 'Export', purpose: 'Save STL, OBJ, PLY and a measurement report.' },
];

export const stepInfo = (s: Step) => STEPS.find(x => x.id === s)!;

export type StepStatus = 'done' | 'todo' | 'skipped';

/** Progress of a project through the journey, derived from its models. */
export function journeyStatus(assets: Asset[], exported: boolean, measured: boolean): Record<Step, StepStatus> {
  const geo = assets.filter(a => a.kind !== 'image');
  const roles = geo.map(roleOf);
  const scans = roles.filter(r => r === 'scan').length;
  return {
    capture: scans > 0 || roles.includes('mesh') ? 'done' : 'todo',
    clean: roles.includes('cleaned') || roles.includes('edited') ? 'done' : 'todo',
    align: roles.includes('merged') ? 'done' : scans <= 1 && scans + roles.filter(r => r === 'cleaned').length > 0 ? 'skipped' : 'todo',
    mesh: geo.some(a => a.kind === 'mesh' && roleOf(a) === 'mesh' && a.operation !== 'import') ? 'done' : 'todo',
    measure: measured || roles.includes('inspection') ? 'done' : 'todo',
    export: exported ? 'done' : 'todo',
  };
}

/** The step we suggest next: the first one not done (skipping optional ones that do not apply). */
export function suggestedStep(status: Record<Step, StepStatus>): Step {
  for (const s of STEPS) if (status[s.id] === 'todo') return s.id;
  return 'export';
}
