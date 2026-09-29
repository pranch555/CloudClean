import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';

const COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];

/** Manual point pairs for merging symmetric parts: click a spot on the moving scan, then the same spot on the reference. */
export function PairOverlay() {
  const pairing = useStore(s => s.pairing);
  const labels = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const v = getViewer();
    if (!v || !pairing) return;
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => {
      if (e.button === 0) down = { x: e.clientX, y: e.clientY };
    };
    const onUp = (e: PointerEvent) => {
      if (!down || e.button !== 0 || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      down = null;
      const hit = v.pickPoint(e.clientX, e.clientY, 10);
      if (!hit) return;
      const id = v.assetAt(hit);
      const P = useStore.getState().pairing;
      if (!P || !id) return;
      const local = v.worldToLocal(id, hit);
      const pending = { ...P.pending };
      if (id === P.movId) pending.source = local;
      else if (id === P.refId) pending.target = local;
      else return;
      if (pending.source && pending.target) {
        useStore.setState({ pairing: { ...P, list: [...P.list, { source: pending.source, target: pending.target }], pending: {} } });
      } else useStore.setState({ pairing: { ...P, pending } });
    };
    v.canvas.addEventListener('pointerdown', onDown);
    v.canvas.addEventListener('pointerup', onUp);
    return () => {
      v.canvas.removeEventListener('pointerdown', onDown);
      v.canvas.removeEventListener('pointerup', onUp);
    };
  }, [!!pairing]);

  const specs = (() => {
    const v = getViewer();
    if (!pairing || !v) return [];
    const toWorld = (id: string, p: [number, number, number]) => {
      const obj = v.items.get(id)?.object;
      const w = new THREE.Vector3(...p);
      if (obj) obj.localToWorld(w);
      return w.toArray() as [number, number, number];
    };
    const out: { position: [number, number, number]; color: string; label: string }[] = [];
    pairing.list.forEach((pr, i) => {
      const color = COLORS[i % COLORS.length];
      out.push({ position: toWorld(pairing.movId, pr.source), color, label: String(i + 1) }, { position: toWorld(pairing.refId, pr.target), color, label: String(i + 1) });
    });
    if (pairing.pending.source) out.push({ position: toWorld(pairing.movId, pairing.pending.source), color: '#ffffff', label: '?' });
    if (pairing.pending.target) out.push({ position: toWorld(pairing.refId, pairing.pending.target), color: '#ffffff', label: '?' });
    return out;
  })();

  useEffect(() => {
    const v = getViewer();
    if (!v || !pairing) return;
    v.setMarkers(specs.map((s, i) => ({ id: String(i), position: s.position, color: s.color })), [], 'pairs');
    return () => v.setMarkers([], [], 'pairs');
  });

  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    return v.subscribe(() => {
      const el = labels.current;
      if (!el) return;
      [...el.children].forEach(node => {
        const p = (node as HTMLElement).dataset.pos!.split(',').map(Number) as [number, number, number];
        const s = v.project(p);
        (node as HTMLElement).style.display = s ? '' : 'none';
        if (s) (node as HTMLElement).style.transform = `translate(${s.x + 8}px, ${s.y - 22}px)`;
      });
    });
  }, []);

  if (!pairing) return null;
  return (
    <div ref={labels} className="labels-layer" aria-hidden>
      {specs.map((s, i) => (
        <div key={i} className="pair-label" data-pos={s.position.join(',')} style={{ background: s.color }}>
          {s.label}
        </div>
      ))}
    </div>
  );
}
