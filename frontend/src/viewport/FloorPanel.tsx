import { useEffect } from 'react';
import * as THREE from 'three';
import { create } from 'zustand';
import { ArrowDown, ArrowUp, MousePointerClick, RefreshCw, RotateCcw, RotateCw, Undo2 } from 'lucide-react';
import { submitJob } from '../lib/jobs';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';
import { FLAT_ENOUGH, flatAreaAt, levelTurn, placementMatrix, rowMajor } from '../viewer/levelling';
import { Button, IconButton, Segmented } from '../ui/primitives';

/*
 * The Floor: a solid floor under the models that a model can be lined up with - click a flat part of the model and
 * it turns so that part is level, standing on the floor; tip it by steps; save it as a new model standing on the
 * floor. The turn is only shown (Viewer.setPlacement) until it is saved, so the data never changes by itself.
 */

type Quat = [number, number, number, number];
type Step = '90' | '15' | '5' | '1';

interface FloorState {
  on: boolean;
  /** the model being lined up, and its turn as shown (about its centre); null = as stored */
  assetId: string | null;
  turn: Quat | null;
  history: (Quat | null)[];
  step: Step;
  note: string | null;
  /** the floor was switched off with a turn that is not saved yet: ask what to do */
  asking: boolean;
  saving: boolean;
}

export const useFloor = create<FloorState>(() => ({ on: false, assetId: null, turn: null, history: [], step: '90', note: null, asking: false, saving: false }));

const quat = (q: Quat | null) => (q ? new THREE.Quaternion(...q) : new THREE.Quaternion());
const toArr = (q: THREE.Quaternion): Quat => [q.x, q.y, q.z, q.w];
const reduceMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function worldUp() {
  return useStore.getState().display.upAxis === 'z' ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
}

function centreOf(id: string): THREE.Vector3 | null {
  const geom = getViewer()?.items.get(id)?.geometry;
  if (!geom) return null;
  if (!geom.boundingBox) geom.computeBoundingBox();
  return geom.boundingBox!.getCenter(new THREE.Vector3());
}

function show(id: string, q: THREE.Quaternion | null) {
  const v = getViewer();
  const c = centreOf(id);
  if (!v || !c) return;
  v.setPlacement(id, q ? placementMatrix(q, c) : null);
}

let anim = 0;
/** Turn the shown model to q (animated), remembering the previous turn for Undo. */
function turnTo(id: string, q: THREE.Quaternion, note: string | null = null, reframe = false) {
  const s = useFloor.getState();
  const same = s.assetId === id;
  const from = same ? quat(s.turn) : new THREE.Quaternion();
  useFloor.setState({ assetId: id, turn: toArr(q), history: [...(same ? s.history : []), same ? s.turn : null].slice(-30), note });
  cancelAnimationFrame(anim);
  const start = performance.now();
  const dur = reduceMotion() ? 0 : 420;
  const tick = (now: number) => {
    const k = dur ? Math.min((now - start) / dur, 1) : 1;
    const e = 1 - Math.pow(1 - k, 3);
    show(id, new THREE.Quaternion().slerpQuaternions(from, q, e));
    if (k < 1) anim = requestAnimationFrame(tick);
    else if (reframe) getViewer()?.fit(undefined, undefined, true);
  };
  anim = requestAnimationFrame(tick);
}

function target(): string | null {
  const s = useFloor.getState();
  const st = useStore.getState();
  if (s.assetId && st.visible.includes(s.assetId)) return s.assetId;
  const a = st.activeId && st.visible.includes(st.activeId) ? st.activeId : st.visible.find(id => st.byId.get(id)?.kind !== 'image');
  return a ?? null;
}

/** Tip the model by the chosen step: about the view's left-right line, its depth line, or the upright. */
function tip(axis: 'pitch' | 'roll' | 'spin', sign: 1 | -1) {
  const v = getViewer();
  const id = target();
  if (!v || !id) return;
  const s = useFloor.getState();
  if (s.assetId && s.assetId !== id && s.turn) return;
  const up = worldUp();
  let a: THREE.Vector3;
  if (axis === 'spin') a = up.clone();
  else {
    const dir = new THREE.Vector3(axis === 'pitch' ? 1 : 0, 0, axis === 'pitch' ? 0 : -1).applyQuaternion(v.camera.quaternion);
    a = dir.addScaledVector(up, -dir.dot(up));
    if (a.lengthSq() < 1e-8) a = new THREE.Vector3(1, 0, 0).applyQuaternion(v.camera.quaternion);
    a.normalize();
  }
  const deg = Number(s.step) * sign;
  const q = new THREE.Quaternion().setFromAxisAngle(a, THREE.MathUtils.degToRad(deg)).multiply(quat(s.assetId === id ? s.turn : null));
  turnTo(id, q, null, Math.abs(deg) >= 90);
}

/** A click on the model: the flat area there becomes level, the model standing above it. */
function levelAt(clientX: number, clientY: number) {
  const v = getViewer();
  if (!v) return;
  const hit = v.pickPoint(clientX, clientY, 10);
  if (!hit) return;
  const id = v.assetAt(hit);
  const s = useFloor.getState();
  if (!id) return;
  if (s.assetId && s.assetId !== id && s.turn) {
    useFloor.setState({ note: 'Save or start over on the other model first - one model at a time.' });
    return;
  }
  const it = v.items.get(id);
  if (!it) return;
  const local = new THREE.Vector3(...v.worldToLocal(id, hit));
  if (!it.geometry.boundingBox) it.geometry.computeBoundingBox();
  const diag = it.geometry.boundingBox!.getSize(new THREE.Vector3()).length();
  const area = flatAreaAt(it.geometry.getAttribute('position').array, local, diag);
  if (!area) {
    useFloor.setState({ note: 'Too few points there - click on the surface of the model.' });
    return;
  }
  const q = levelTurn(quat(s.assetId === id ? s.turn : null), area.up, worldUp());
  turnTo(id, q, area.flatness > FLAT_ENOUGH ? 'That spot is curved, so it was levelled to its average direction. A flat face gives an exact result.' : null, true);
}

function undo() {
  const s = useFloor.getState();
  if (!s.assetId || !s.history.length) return;
  const prev = s.history[s.history.length - 1];
  cancelAnimationFrame(anim);
  useFloor.setState({ turn: prev, history: s.history.slice(0, -1), note: null });
  show(s.assetId, prev ? quat(prev) : null);
}

function startOver() {
  const s = useFloor.getState();
  cancelAnimationFrame(anim);
  if (s.assetId) getViewer()?.setPlacement(s.assetId, null);
  useFloor.setState({ assetId: null, turn: null, history: [], note: null, asking: false });
}

function save(thenHide = false) {
  const s = useFloor.getState();
  const st = useStore.getState();
  const asset = s.assetId ? st.byId.get(s.assetId) : undefined;
  const c = s.assetId ? centreOf(s.assetId) : null;
  if (!asset || !s.turn || !c) return;
  const ops = [
    { op: 'transform', matrix: rowMajor(placementMatrix(quat(s.turn), c)) },
    { op: 'center', mode: 'bbox_bottom', up: st.display.upAxis },   // standing on the floor, at the origin
  ];
  useFloor.setState({ saving: true, asking: false });
  submitJob('/api/edit', { asset_id: asset.id, ops, name: `${asset.name} · on the floor` }, job => {
    useFloor.setState({ saving: false });
    if (job.status !== 'done') return;
    getViewer()?.setPlacement(asset.id, null);
    useFloor.setState({ assetId: null, turn: null, history: [], note: null, on: thenHide ? false : useFloor.getState().on });
  }).then(job => {
    if (!job) useFloor.setState({ saving: false });
  });
}

/** The Floor button: on shows the floor; off hides it (asking first when a turn is not saved yet). */
export function toggleFloor() {
  const s = useFloor.getState();
  if (!s.on) useFloor.setState({ on: true, note: null });
  else if (s.turn && !s.saving) useFloor.setState({ asking: true });
  else {
    startOver();
    useFloor.setState({ on: false });
  }
}

export function FloorPanel() {
  const f = useFloor();
  const tool = useStore(s => s.tool);
  const visible = useStore(s => s.visible);
  const byId = useStore(s => s.byId);
  useStore(s => s.activeId);
  const id = f.on ? target() : null;
  const name = id ? byId.get(id)?.name : undefined;

  // the floor on the stage
  useEffect(() => {
    getViewer()?.setFloor(f.on);
  }, [f.on]);

  // the model being lined up was hidden or removed: show it as stored again
  useEffect(() => {
    if (f.assetId && !visible.includes(f.assetId)) startOver();
  }, [visible, f.assetId]);

  // a click (not a drag) on a model while the floor is out levels the flat area under the cursor
  useEffect(() => {
    const v = getViewer();
    if (!v || !f.on || f.asking || tool !== 'navigate') return;
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => {
      down = e.button === 0 && !e.shiftKey ? { x: e.clientX, y: e.clientY } : null;
    };
    const onUp = (e: PointerEvent) => {
      if (!down || e.button !== 0 || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      down = null;
      levelAt(e.clientX, e.clientY);
    };
    v.canvas.addEventListener('pointerdown', onDown);
    v.canvas.addEventListener('pointerup', onUp);
    return () => {
      v.canvas.removeEventListener('pointerdown', onDown);
      v.canvas.removeEventListener('pointerup', onUp);
    };
  }, [f.on, f.asking, tool]);

  // a model that was reloaded (or reset by side-by-side panes) shows its unsaved turn again
  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    return v.subscribe(() => {
      const s = useFloor.getState();
      if (s.on && s.assetId && s.turn && v.items.has(s.assetId) && !v.isPlaced(s.assetId)) show(s.assetId, quat(s.turn));
    });
  }, []);

  if (!f.on) return null;
  const changed = !!f.turn && f.assetId === id;

  return (
    <section className="hud floor-panel" aria-label="Floor" data-guide="view.floor-panel">
      <header className="floor-head">
        <strong>Stand it on the floor</strong>
        {name && <span className="muted truncate" title={name}>{name}</span>}
      </header>
      {!id ? (
        <p className="hint-text">Show a model to line it up with the floor.</p>
      ) : f.asking ? (
        <div className="stack tight">
          <p className="hint-text">Keep the model the way it stands now? It is saved as a new model standing on the floor; the model you started from stays as it was.</p>
          <div className="floor-actions">
            <Button variant="primary" loading={f.saving} onClick={() => save(true)}>Keep it, hide the floor</Button>
            <Button onClick={() => { startOver(); useFloor.setState({ on: false }); }}>Put it back</Button>
          </div>
        </div>
      ) : (
        <>
          <p className="floor-tip"><MousePointerClick size={16} aria-hidden /> <span>Click a flat part of the model - its bottom, the top of its base, a flat side - to make it level.</span></p>
          {tool !== 'navigate' && <p className="hint-text">Switch to Move the view (V) to click on the model.</p>}
          <div className="floor-row" role="group" aria-label="Tip the model">
            <span className="floor-label">Tip it</span>
            <div className="chip-row">
              <button type="button" className="chip" onClick={() => tip('roll', -1)}><RotateCcw size={14} aria-hidden /> Left</button>
              <button type="button" className="chip" onClick={() => tip('roll', 1)}>Right <RotateCw size={14} aria-hidden /></button>
              <button type="button" className="chip" onClick={() => tip('pitch', 1)}><ArrowDown size={14} aria-hidden /> Towards you</button>
              <button type="button" className="chip" onClick={() => tip('pitch', -1)}><ArrowUp size={14} aria-hidden /> Away</button>
              <button type="button" className="chip" onClick={() => tip('spin', 1)}><RefreshCw size={14} aria-hidden /> Turn round</button>
            </div>
          </div>
          <div className="floor-row">
            <span className="floor-label">By</span>
            <Segmented size="sm" ariaLabel="Tip by" value={f.step} onChange={step => useFloor.setState({ step })} options={[{ value: '90', label: '90°' }, { value: '15', label: '15°' }, { value: '5', label: '5°' }, { value: '1', label: '1°' }]} />
          </div>
          {f.note && <p className="hint-text floor-note" role="status">{f.note}</p>}
          <div className="floor-actions">
            <Button variant="primary" disabled={!changed} loading={f.saving} onClick={() => save()} title="Saves a new model standing on the floor. The one you started from stays as it was.">Save</Button>
            <IconButton size="sm" label="Undo the last turn" disabled={!changed || !f.history.length} onClick={undo}><Undo2 size={15} /></IconButton>
            <Button size="sm" variant="ghost" disabled={!changed} onClick={startOver}>Start over</Button>
          </div>
          <p className="hint-text">Saves a new model. Sizes never change - it is only turned.</p>
        </>
      )}
    </section>
  );
}
