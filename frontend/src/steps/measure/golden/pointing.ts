import { useGoldenFocus, useGoldenPins, type GoldenDimension } from '../../../lib/golden';
import { useStore } from '../../../store';
import { getViewer } from '../../../viewer/instance';

/*
 * What the golden check panel points at on the 3D view while the pointer (or keyboard focus) is on something: a
 * colour of the surface bar, an area card, a size. Leaving puts back the resting state: a colour the user pinned,
 * else the area "Show me" is on, else nothing.
 *
 * The viewer lights a set of values of one per-vertex scalar of the check mesh (cloudclean/web/jobs_golden.py):
 * check_status (the colours), check_region (the areas), check_face (the golden faces a size is taken between).
 */

export type Pointing =
  | { kind: 'status'; code: number }
  | { kind: 'region'; id: number }
  | { kind: 'size'; faces: number[]; dimension: GoldenDimension | null };

interface Lights {
  highlight?: (assetId: string | null, name?: string, values?: number[], opts?: { glow?: boolean }) => Promise<number>;
  spotlight: (assetId: string | null, name?: string, value?: number) => Promise<number>;
}

let checkId: string | null = null;
let pinned: Pointing | null = null;
let shown = '';
let leaving = 0;

const keyOf = (p: Pointing | null) => (p ? (p.kind === 'size' ? `size:${p.faces.join(',')}:${p.dimension?.ends.flat().join(',') ?? ''}` : `${p.kind}:${p.kind === 'status' ? p.code : p.id}`) : 'rest');

/** The check whose mesh the pointing lights (null: the panel is gone; everything is put out). */
export function pointingCheck(id: string | null) {
  checkId = id;
  pinned = null;
  shown = '';
  if (!id) {
    useGoldenFocus.setState({ dimension: null });
  }
}

/** A colour the user clicked stays lit until clicked again (null: none). */
export function pinPointing(p: Pointing | null) {
  pinned = p;
  apply(p ?? rest());
}

export const pinnedPointing = () => pinned;

/** The pointer is on something (p) or left it (null). */
export function pointAt(p: Pointing | null) {
  window.clearTimeout(leaving);
  // leaving waits a moment: moving from one colour to the next must not flash the resting state in between
  if (!p) leaving = window.setTimeout(() => apply(rest()), 110);
  else apply(p);
}

/** Put the resting state back now (after "Show me" or "Whole part" changed it). */
export function settlePointing() {
  shown = '';
  apply(rest());
}

function rest(): Pointing | null {
  if (pinned) return pinned;
  const active = useGoldenPins.getState().active;
  return active != null ? { kind: 'region', id: active } : null;
}

function apply(p: Pointing | null) {
  const key = keyOf(p);
  if (key === shown || !checkId) return;
  shown = key;
  useGoldenFocus.setState({ dimension: p?.kind === 'size' ? p.dimension : null });
  const v = getViewer() as unknown as Lights | null;
  if (!v) return;
  const id = checkId;
  const onStage = useStore.getState().visible.includes(id);
  const light = (name: string, values: number[], glow: boolean) =>
    v.highlight ? v.highlight(id, name, values, { glow }) : v.spotlight(id, name, values[0]);
  const clear = () => (v.highlight ? v.highlight(null) : v.spotlight(null));
  if (!p || !onStage) {
    clear();
    return;
  }
  const done = (n: number) => {
    // nothing of it on this mesh (an older check, a face too small to colour): show the whole part instead
    if (n === 0 && shown === key) clear();
  };
  if (p.kind === 'status') light('check_status', [p.code], true).then(done);
  else if (p.kind === 'region') light('check_region', [p.id], p !== pinned && p.id !== useGoldenPins.getState().active).then(done);
  else if (p.faces.length) light('check_face', p.faces, true).then(done);
  else clear();
}
