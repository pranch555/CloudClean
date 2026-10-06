import { useEffect, useLayoutEffect, useRef } from 'react';
import * as THREE from 'three';
import { useGoldenPins } from '../lib/golden';
import { animate, pulse, reducedMotion, springy, stagger, utils } from '../lib/motion';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';

/**
 * Measure -> Golden model: a numbered pin on every area the check lists, the same numbers as the cards in the panel.
 * Pins on the far side of the part fade out; clicking a pin shows that area (GoldenCheck handles `picked`).
 */
export function GoldenPins() {
  const checkId = useGoldenPins(s => s.checkId);
  const pins = useGoldenPins(s => s.pins);
  const active = useGoldenPins(s => s.active);
  const onStage = useStore(s => !!checkId && s.visible.includes(checkId));
  const layer = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const v = getViewer();
    if (!v || !onStage) return;
    const eye = new THREE.Vector3();
    const p = new THREE.Vector3();
    const n = new THREE.Vector3();
    // hidden behind another part of the model? Tested with the viewer's pick pass once the camera has stopped (the
    // pick redraws the view, so it must not run every frame, nor twice for the same camera)
    const hidden = new Map<string, boolean>();
    let tested = '';
    let timer = 0;
    const camKey = () => {
      const c = v.camera;
      return [...c.position.toArray(), ...c.quaternion.toArray(), c.zoom].map(x => x.toFixed(4)).join();
    };
    const facing = (pos: number[], nor: number[] | undefined) => {
      const cam = v.camera;
      p.set(pos[0], pos[1], pos[2]);
      n.set(nor?.[0] ?? 0, nor?.[1] ?? 0, nor?.[2] ?? 0);
      if (n.lengthSq() === 0) return true;
      const ortho = (cam as THREE.OrthographicCamera).isOrthographicCamera;
      const toward = ortho ? -new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion).dot(n) : eye.copy(cam.position).sub(p).normalize().dot(n);
      return toward >= -0.05;
    };
    const occlusion = () => {
      const el = layer.current;
      if (!el) return;
      tested = camKey();
      const rect = v.canvas.getBoundingClientRect();
      for (const node of el.children as HTMLCollectionOf<HTMLElement>) {
        const pos = node.dataset.pos?.split(',').map(Number);
        if (!pos || pos.length !== 3) continue;
        const s = v.project(pos as [number, number, number]);
        if (!s) continue;
        const hit = v.pickPoint(rect.left + s.x, rect.top + s.y, 2);
        p.set(pos[0], pos[1], pos[2]);
        const cam = v.camera.position;
        const blocked = !!hit && hit.distanceTo(p) > 1.0 && hit.distanceTo(cam) < p.distanceTo(cam);
        hidden.set(node.dataset.pos!, blocked);
        node.classList.toggle('is-behind', blocked || !facing(pos, node.dataset.normal?.split(',').map(Number)));
      }
    };
    const place = () => {
      const el = layer.current;
      if (!el) return;
      for (const node of el.children as HTMLCollectionOf<HTMLElement>) {
        const pos = node.dataset.pos?.split(',').map(Number);
        if (!pos || pos.length !== 3) continue;
        const s = v.project(pos as [number, number, number]);
        node.style.display = s ? '' : 'none';
        if (!s) continue;
        node.style.transform = `translate(${s.x}px, ${s.y}px)`;
        node.classList.toggle('is-behind', !!hidden.get(node.dataset.pos!) || !facing(pos, node.dataset.normal?.split(',').map(Number)));
      }
      if (camKey() !== tested) {
        window.clearTimeout(timer);
        timer = window.setTimeout(occlusion, 160);
      }
    };
    place();
    const off = v.subscribe(place);
    return () => {
      off();
      window.clearTimeout(timer);
    };
  }, [onStage, pins]);

  // a new set of pins pops up from their anchors, one after another (the heads scale through --pin-pop, so the
  // hover / active scale on their transform keeps working)
  const arrival = onStage ? pins.map(p => p.region).join(',') : '';
  useLayoutEffect(() => {
    const el = layer.current;
    if (!el || !arrival || reducedMotion()) return;
    const heads = el.querySelectorAll<HTMLElement>('.gold-pin3d span');
    utils.set(heads, { '--pin-pop': 0, opacity: 0 }); // hidden through the stagger's wait, not shown then popped
    const a = animate(heads, {
      '--pin-pop': [0, 1],
      opacity: { from: 0, to: 1, duration: 180, ease: 'out(2)' },
      ease: springy(),
      delay: stagger(55, { start: 120 }),
    });
    return () => void a.complete();
  }, [arrival]);

  // the area "Show me" is on: its pin says "here" once the view has flown to it, and once more
  useEffect(() => {
    if (active == null || !onStage) return;
    const ring = () => pulse(layer.current?.querySelector<HTMLElement>(`[data-region="${active}"] span`) ?? null);
    const timers = [window.setTimeout(ring, 650), window.setTimeout(ring, 1400)];
    return () => timers.forEach(t => window.clearTimeout(t));
  }, [active, onStage]);

  if (!onStage || !pins.length) return null;
  return (
    <div ref={layer} className="gold-pins" role="group" aria-label="Areas the golden check found">
      {pins.map(pin => (
        <button
          key={pin.region}
          type="button"
          className={`gold-pin3d ${active === pin.region ? 'is-active' : ''} ${active != null && active !== pin.region ? 'is-dim' : ''}`}
          data-pos={pin.pos.join(',')}
          data-normal={pin.normal.join(',')}
          data-region={pin.region}
          style={{ '--pin': pin.color } as React.CSSProperties}
          title={`${pin.n}. ${pin.label}`}
          aria-label={`Area ${pin.n}: ${pin.label}`}
          onClick={() => useGoldenPins.setState({ picked: { region: pin.region, at: Date.now() } })}
        >
          <span>{pin.n}</span>
        </button>
      ))}
    </div>
  );
}
