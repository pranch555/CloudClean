import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { X } from 'lucide-react';
import { fmtLen } from '../lib/format';
import { addMeasurePoint, measureColor } from '../lib/measure';
import { cssVar } from '../lib/theme';
import { useStore, type Vec3 } from '../store';
import { getViewer } from '../viewer/instance';

/**
 * Everything drawn *on* the model: two-point measurements (the M tool), dimension lines from measuring tools and the
 * assistant, notes, and thread crests. 3D lines and end marks come from the viewer; value tags are HTML, glued to
 * the line midpoints every frame.
 */
export function Labels() {
  const tool = useStore(s => s.tool);
  const measurements = useStore(s => s.measurements);
  const dims = useStore(s => s.dims);
  const annotations = useStore(s => s.annotations);
  const thread = useStore(s => s.thread);
  const axisMode = useStore(s => s.measureAxis);
  const units = useStore(s => s.display.units);
  const layer = useRef<HTMLDivElement>(null);

  // M tool: click two points
  useEffect(() => {
    const v = getViewer();
    if (!v || tool !== 'measure') return;
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => {
      if (e.button === 0) down = { x: e.clientX, y: e.clientY };
    };
    const onUp = (e: PointerEvent) => {
      if (!down || e.button !== 0 || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      down = null;
      const hit = v.pickPoint(e.clientX, e.clientY, 10);
      if (hit) addMeasurePoint(hit.toArray() as Vec3, v.assetAt(hit));
    };
    v.canvas.addEventListener('pointerdown', onDown);
    v.canvas.addEventListener('pointerup', onUp);
    return () => {
      v.canvas.removeEventListener('pointerdown', onDown);
      v.canvas.removeEventListener('pointerup', onUp);
    };
  }, [tool]);

  // pivot tool: click a point to turn around it, then back to navigating
  useEffect(() => {
    const v = getViewer();
    if (!v || tool !== 'pivot') return;
    const onClick = (e: MouseEvent) => {
      const hit = v.pickPoint(e.clientX, e.clientY, 10);
      if (!hit) return;
      v.lookAtPoint(hit.toArray() as [number, number, number]);
      useStore.getState().setDisplay({ rotatePivot: 'center' });
      useStore.setState({ tool: 'navigate' });
    };
    v.canvas.addEventListener('click', onClick);
    return () => v.canvas.removeEventListener('click', onClick);
  }, [tool]);

  // 3D lines and end marks
  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    const markers = measurements.flatMap((m, i) => [m.a, ...(m.b ? [m.b] : [])].map((p, j) => ({ id: `${m.id}-${j}`, position: p, color: measureColor(i) })));
    const segments = measurements.filter(m => m.b).map(m => ({ from: m.a, to: m.b!, color: measureColor(measurements.indexOf(m)) }));
    v.setMarkers(markers, segments, 'measure', 0.8);
  }, [measurements]);

  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    const ink = cssVar('--ink-1', '#17161a');
    const signal = cssVar('--signal', '#ee4b1f');
    v.setMarkers(
      dims.flatMap(d => [{ id: `${d.id}a`, position: d.a, color: signal }, { id: `${d.id}b`, position: d.b, color: signal }]),
      dims.map(d => ({ from: d.a, to: d.b, color: ink })),
      'dims',
      0.55,
    );
  }, [dims]);

  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    v.setMarkers(annotations.map(a => ({ id: a.id, position: a.position, color: cssVar('--signal', '#ee4b1f') })), [], 'notes', 0.7);
  }, [annotations]);

  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    if (!thread) {
      v.setMarkers([], [], 'thread');
      return;
    }
    const c = new THREE.Vector3(...thread.center), ax = new THREE.Vector3(...thread.axis).normalize();
    const half = thread.length / 2 + thread.major_diameter * 0.4;
    v.setMarkers(
      thread.crest_points.map((p, i) => ({ id: `c${i}`, position: p, color: '#178f64' })),
      [{ from: c.clone().addScaledVector(ax, -half).toArray() as Vec3, to: c.clone().addScaledVector(ax, half).toArray() as Vec3, color: '#178f64' }],
      'thread',
      0.55,
    );
  }, [thread]);

  // keep the HTML tags glued to their 3D anchors
  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    return v.subscribe(() => {
      const el = layer.current;
      if (!el) return;
      for (const node of el.children as HTMLCollectionOf<HTMLElement>) {
        const p = node.dataset.pos?.split(',').map(Number) as Vec3 | undefined;
        if (!p) continue;
        const s = v.project(p);
        node.style.display = s ? '' : 'none';
        if (s) node.style.transform = node.dataset.kind === 'note' ? `translate(${s.x + 10}px, ${s.y - 16}px)` : `translate(${s.x}px, ${s.y}px) translate(-50%, -135%)`;
      }
    });
  }, []);

  const mid = (a: Vec3, b: Vec3) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2].join(',');
  const done = measurements.filter(m => m.b);

  return (
    <div ref={layer} className="labels-layer" role="group" aria-label="Dimensions on the model">
      {done.map(m => (
        <div key={m.id} className="dim-tag" data-pos={mid(m.a, m.b!)}>
          <b style={{ background: measureColor(measurements.indexOf(m)) }}>{m.label}</b>
          {m.result ? fmtLen(axisMode !== 'none' && m.result.along_axis != null ? m.result.along_axis : m.result.distance, 3) : '…'}
          <span className="unit">{units}</span>
          {axisMode !== 'none' && m.result?.along_axis != null ? <span className="axis-tag">∥{axisMode === 'thread' ? 'axis' : axisMode}</span> : null}
        </div>
      ))}
      {dims.map(d => (
        <div key={d.id} className="dim-tag" data-pos={mid(d.a, d.b)}>
          <b>{d.label}</b>
          {d.value != null ? fmtLen(d.value, 3) : '—'}
          <span className="unit">{d.kind === 'angle' ? '°' : d.unit}</span>
          <button type="button" className="icon-link" style={{ color: 'inherit', opacity: 0.6, pointerEvents: 'auto' }} aria-label={`Remove ${d.label}`} onClick={() => useStore.setState(s => ({ dims: s.dims.filter(x => x.id !== d.id) }))}>
            <X size={12} />
          </button>
        </div>
      ))}
      {annotations.map(a => (
        <div key={a.id} className="note-tag" data-kind="note" data-pos={a.position.join(',')}>
          {a.text}
        </div>
      ))}
    </div>
  );
}
