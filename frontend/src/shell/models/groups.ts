/*
 * The model list, sorted by what each thing IS: scans from the scanner, what was made from them, checks against
 * the golden model, the CAD / golden models and photos. Display names here never rename files: a generated result
 * with its automatic name reads as "Golden check" / "Deviation map", a name the user typed is shown as typed.
 */
import type { Asset } from '../../lib/types';
import { fmtCount } from '../../lib/format';
import { roleOf } from '../../lib/journey';

export type GroupId = 'scans' | 'made' | 'checks' | 'cad' | 'photos';

export interface GroupInfo {
  id: GroupId;
  title: string;
  /** one line: what is in this group, in plain words */
  blurb: string;
  /** data-guide id of the group (cloudclean/assistant/guide.py) */
  guide: string;
}

export const GROUPS: Record<GroupId, GroupInfo> = {
  scans: { id: 'scans', title: 'Scans', blurb: 'From the scanner, or opened from files.', guide: 'models.scans' },
  made: { id: 'made', title: 'Made from scans', blurb: 'Cleaned, merged and meshed versions. Your original scans stay as they are.', guide: 'models.made' },
  checks: { id: 'checks', title: 'Checks', blurb: 'Your scans compared with the golden model. Click a verdict to see the details.', guide: 'models.checks' },
  cad: { id: 'cad', title: 'CAD & golden models', blurb: 'The part as it should be. Checks compare your scans with the ★ golden model.', guide: 'models.cad' },
  photos: { id: 'photos', title: 'Photos', blurb: 'Pictures of the real part. Click one to see it big; × deletes it.', guide: 'models.photos' },
};

export const GROUP_ORDER: GroupId[] = ['scans', 'made', 'checks', 'cad', 'photos'];

const CHECK_OPS = new Set(['golden_check', 'compare', 'compare_scans']);

export const isCheck = (a: Asset) => CHECK_OPS.has(a.operation);

const isCoverage = (a: Asset) => a.operation === 'compare' && (!!a.scalars?.some(s => s.name === 'reference_distance') || a.name.endsWith(' · coverage'));

export function groupOf(a: Asset, golden: string | null): GroupId {
  if (a.kind === 'image') return 'photos';
  if (isCheck(a)) return 'checks';
  const role = roleOf(a);
  if (a.id === golden || role === 'cad') return 'cad';
  if (a.operation === 'import' || a.operation === 'capture' || role === 'fromPhotos') return 'scans';
  return 'made';
}

/** Colour family of a kind tag (the group colours, tokens --k-* in styles/models.css). */
export type Tone = 'scan' | 'made' | 'check' | 'cad' | 'photo' | 'golden';

/** What a model is, as a short tag. Null when the group title already says it (a plain scan among Scans). */
export function kindTag(a: Asset): { label: string; tone: Tone } | null {
  const role = roleOf(a);
  switch (a.operation) {
    case 'clean': return { label: 'Cleaned', tone: 'made' };
    case 'merge': return { label: 'Merged', tone: 'made' };
    case 'mesh': return { label: 'Mesh', tone: 'made' };
    case 'texture': return { label: 'Coloured', tone: 'made' };
    case 'photo_fill': return { label: 'Filled from photos', tone: 'made' };
    case 'photos': return { label: 'From photos', tone: 'photo' };
    case 'edit': return { label: 'Edited', tone: 'made' };
  }
  if (role === 'cad') return { label: 'CAD', tone: 'cad' };
  if (a.operation === 'import' || a.operation === 'capture') return a.kind === 'mesh' ? { label: 'Mesh file', tone: 'scan' } : null;
  return role === 'mesh' ? { label: 'Mesh', tone: 'made' } : { label: 'Edited', tone: 'made' };
}

/** "390k points", "3.73M triangles" */
export const countText = (a: Asset) => (a.kind === 'mesh' ? `${fmtCount(a.stats.triangles)} triangles` : `${fmtCount(a.stats.points)} points`);

export type CheckKind = 'golden' | 'deviation' | 'coverage' | 'repeatability';

export function checkKind(a: Asset): CheckKind | null {
  if (a.operation === 'golden_check') return 'golden';
  if (a.operation === 'compare_scans') return 'repeatability';
  if (a.operation === 'compare') return isCoverage(a) ? 'coverage' : 'deviation';
  return null;
}

export const CHECK_TITLE: Record<CheckKind, string> = {
  golden: 'Golden check',
  deviation: 'Deviation map',
  coverage: 'Coverage on CAD',
  repeatability: 'Repeatability check',
};

/** What the colours of a check result show, one short line. */
export const CHECK_BLURB: Record<CheckKind, string> = {
  golden: 'The golden model coloured by what the check found',
  deviation: 'Scan points coloured by distance',
  coverage: 'CAD surface coloured by how well it was scanned',
  repeatability: 'Two scans of the same part laid over each other',
};

/** Was the name made by CloudClean (then a friendly title is shown) or typed by the user (shown as typed)? */
function autoNamed(a: Asset, kind: CheckKind): boolean {
  switch (kind) {
    case 'golden': return a.name.includes(' · check of ');
    case 'coverage': return a.name.endsWith(' · coverage');
    case 'deviation': return a.name.includes(' vs ');
    case 'repeatability': return a.name.endsWith(' · repeatability');
  }
}

/** The name shown in the list: friendly for generated results, the real name otherwise. */
export function displayName(a: Asset): string {
  const kind = checkKind(a);
  return kind && autoNamed(a, kind) ? CHECK_TITLE[kind] : a.name;
}

/** Long file names wrap at _ and - instead of in the middle of a word. */
export const breakable = (name: string) => name.replace(/([_\-.])/g, '$1​');

/** What a model was made from: other models, and how many photos. */
export function madeFrom(a: Asset, byId: Map<string, Asset>): { models: Asset[]; photos: number } {
  if (a.operation === 'photos') return { models: [], photos: a.parents.length };
  const models: Asset[] = [];
  let photos = 0;
  for (const id of a.parents) {
    const p = byId.get(id);
    if (!p) continue;
    if (p.kind === 'image') photos += 1;
    else models.push(p);
  }
  return { models, photos };
}

const ms = (iso: string) => Date.parse(iso) || 0;

/** A generated result is saved within seconds of the check that made it. */
const SAME_RUN_MS = 120_000;

export interface Entry {
  asset: Asset;
  /** results that belong to it (one level): the deviation map of a golden check, the coverage of a compare */
  children: Asset[];
}

/**
 * Checks with the results that belong to them nested one level: a golden check holds the deviation map it made
 * (known from its report, `links[checkId] = compareId`, else: same scan and golden model, saved just before it);
 * a deviation map holds the coverage made by the same compare. Newest first.
 */
export function checkEntries(checks: Asset[], links: Record<string, string | null | undefined>): Entry[] {
  const golden = checks.filter(a => a.operation === 'golden_check');
  const linkedCompare = new Set(Object.values(links).filter(Boolean) as string[]);
  const ownerOf = new Map<string, string>();
  for (const c of checks) {
    const kind = checkKind(c);
    if (kind !== 'deviation') continue;
    let owner = golden.find(g => links[g.id] === c.id);
    if (!owner && !linkedCompare.has(c.id)) {
      owner = golden
        .filter(g => links[g.id] == null && g.parents.join() === c.parents.join() && ms(g.created) - ms(c.created) >= 0 && ms(g.created) - ms(c.created) <= SAME_RUN_MS)
        .sort((x, y) => x.created.localeCompare(y.created))[0];
    }
    if (owner) ownerOf.set(c.id, owner.id);
  }
  for (const v of checks) {
    if (checkKind(v) !== 'coverage') continue;
    const compare = checks
      .filter(c => checkKind(c) === 'deviation' && c.parents.join() === v.parents.join() && ms(v.created) - ms(c.created) >= 0 && ms(v.created) - ms(c.created) <= SAME_RUN_MS)
      .sort((x, y) => y.created.localeCompare(x.created))[0];
    if (compare) ownerOf.set(v.id, ownerOf.get(compare.id) ?? compare.id);
  }
  const top = checks.filter(a => !ownerOf.has(a.id)).sort((x, y) => y.created.localeCompare(x.created));
  return top.map(asset => ({
    asset,
    children: checks.filter(c => ownerOf.get(c.id) === asset.id).sort((x, y) => x.created.localeCompare(y.created)),
  }));
}

/** "5 Oct, 10:06" */
export const whenText = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
};
