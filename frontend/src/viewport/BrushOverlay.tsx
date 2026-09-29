import { useEffect, useRef } from 'react';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';

export const brushRadius = () => {
  const st = useStore.getState();
  if (st.brush.radius > 0) return st.brush.radius;
  const a = st.byId.get(st.brush.assetId ?? st.activeId ?? '');
  return a ? Math.max(a.stats.diagonal / 40, a.stats.spacing * 6) : 1;
};

/**
 * Surface-smoothing brush: left-drag paints spheres on the surface (GPU pick at ~30 Hz), the painted area is
 * highlighted, and a circle shows the brush size at the cursor. Right-drag still pans, middle-drag rotates.
 */
export function BrushOverlay() {
  const tool = useStore(s => s.tool);
  const cursor = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const v = getViewer();
    if (!v || tool !== 'brush') return;
    let painting = false;
    let last = 0;
    let lastPoint: { x: number; y: number } | null = null;

    const paintAt = (e: PointerEvent) => {
      const hit = v.pickPoint(e.clientX, e.clientY, 6);
      if (!hit) return;
      const st = useStore.getState();
      const owner = v.assetAt(hit);
      if (st.brush.assetId && owner && owner !== st.brush.assetId) return; // stay on one asset per stroke set
      const r = brushRadius();
      const sphere: [number, number, number, number] = [hit.x, hit.y, hit.z, r];
      const prev = st.brush.spheres[st.brush.spheres.length - 1];
      if (prev && Math.hypot(prev[0] - hit.x, prev[1] - hit.y, prev[2] - hit.z) < r * 0.35) return; // spacing along the stroke
      useStore.setState({ brush: { ...st.brush, assetId: st.brush.assetId ?? owner, spheres: [...st.brush.spheres, sphere] } });
      v.paintSpheres([sphere], false, st.brush.assetId ?? owner);
    };

    const moveCursor = (e: PointerEvent) => {
      const el = cursor.current;
      if (!el) return;
      const rect = v.canvas.getBoundingClientRect();
      const now = performance.now();
      if (now - last > 60 || painting) {
        const hit = v.pickPoint(e.clientX, e.clientY, 4);
        if (hit) {
          const px = Math.max(6, v.pixelsFor(brushRadius(), hit) * 2);
          el.style.width = el.style.height = `${px}px`;
          el.style.display = '';
        } else el.style.display = 'none';
        last = now;
      }
      el.style.transform = `translate(${e.clientX - rect.left + v.canvas.offsetLeft}px, ${e.clientY - rect.top + v.canvas.offsetTop}px) translate(-50%, -50%)`;
    };

    const down = (e: PointerEvent) => {
      if (e.button !== 0) return;
      painting = true;
      lastPoint = { x: e.clientX, y: e.clientY };
      paintAt(e);
    };
    const move = (e: PointerEvent) => {
      moveCursor(e);
      if (!painting || !lastPoint) return;
      if (Math.hypot(e.clientX - lastPoint.x, e.clientY - lastPoint.y) < 4) return;
      lastPoint = { x: e.clientX, y: e.clientY };
      paintAt(e);
    };
    const up = () => {
      painting = false;
    };
    const leave = () => {
      if (cursor.current) cursor.current.style.display = 'none';
    };
    v.canvas.addEventListener('pointerdown', down);
    v.canvas.addEventListener('pointermove', move);
    v.canvas.addEventListener('pointerleave', leave);
    window.addEventListener('pointerup', up);
    return () => {
      v.canvas.removeEventListener('pointerdown', down);
      v.canvas.removeEventListener('pointermove', move);
      v.canvas.removeEventListener('pointerleave', leave);
      window.removeEventListener('pointerup', up);
    };
  }, [tool]);

  if (tool !== 'brush') return null;
  return <div ref={cursor} className="brush-cursor" style={{ display: 'none' }} aria-hidden />;
}
