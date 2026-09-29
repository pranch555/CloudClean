import { useEffect, useState } from 'react';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';

/** Box / lasso drawing on top of the canvas. Middle/right mouse still navigate while a tool is active. */
export function SelectionOverlay() {
  const tool = useStore(s => s.tool);
  const [path, setPath] = useState<[number, number][] | null>(null);

  useEffect(() => {
    const v = getViewer();
    if (!v || (tool !== 'box' && tool !== 'lasso')) return;
    const canvas = v.canvas;
    let pts: [number, number][] | null = null;
    let additive = false;
    const local = (e: PointerEvent): [number, number] => {
      const r = canvas.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    const down = (e: PointerEvent) => {
      if (e.button !== 0) return;
      additive = e.shiftKey;
      pts = [local(e)];
      setPath(pts);
    };
    const move = (e: PointerEvent) => {
      if (!pts) return;
      const p = local(e);
      if (tool === 'box') pts = [pts[0], p];
      else {
        const last = pts[pts.length - 1];
        if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 3) return;
        pts = [...pts, p];
      }
      setPath(pts);
    };
    const up = () => {
      if (!pts) return;
      const W = canvas.clientWidth, H = canvas.clientHeight;
      let poly: [number, number][] = pts;
      if (tool === 'box') {
        const [[x0, y0], [x1, y1] = pts[0]] = pts;
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
      }
      pts = null;
      setPath(null);
      const area = Math.abs(poly.reduce((acc, [x, y], i) => {
        const [nx, ny] = poly[(i + 1) % poly.length];
        return acc + x * ny - nx * y;
      }, 0)) / 2;
      if (poly.length < 3 || area < 16) return;
      const ndc = poly.map(([x, y]) => [(x / W) * 2 - 1, 1 - (y / H) * 2] as [number, number]);
      const visibleOnly = false;
      const counts = v.selectPolygon(ndc, additive, visibleOnly);
      const st = useStore.getState();
      const prev = additive && st.selection ? st.selection : null;
      useStore.setState({
        // additive selections are applied on the server as successive polygons would need a union; keep the latest polygon
        // plus the flag so the user sees what will be applied.
        selection: { view_projection: v.viewProjection(), polygon: prev && prev.view_projection.join() === v.viewProjection().join() ? mergePolys(prev.polygon, ndc) : ndc, count: Object.values(counts).reduce((a, b) => a + b, 0) },
        selectionCounts: counts,
      });
    };
    canvas.addEventListener('pointerdown', down);
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    return () => {
      canvas.removeEventListener('pointerdown', down);
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
  }, [tool]);

  if (!path) return null;
  let d: string;
  if (tool === 'box' && path.length === 2) {
    const [[x0, y0], [x1, y1]] = path;
    d = `M${x0},${y0}H${x1}V${y1}H${x0}Z`;
  } else d = `M${path.map(p => p.join(',')).join('L')}Z`;
  return (
    <svg className="selection-svg" aria-hidden>
      <path d={d} />
    </svg>
  );
}

/**
 * Shift-adding a second region: the server op takes one polygon, so disjoint regions are joined with a
 * zero-width bridge (even-odd fill keeps both areas selected and the bridge covers no area).
 */
function mergePolys(a: [number, number][], b: [number, number][]): [number, number][] {
  return [...a, a[0], ...b, b[0]];
}
