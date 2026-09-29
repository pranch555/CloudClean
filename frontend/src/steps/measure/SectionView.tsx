import { useId, useRef, type PointerEvent as ReactPointerEvent } from 'react';
import { fmtLen } from '../../lib/format';
import type { PartSummary } from '../../lib/summary';
import type { PartAxis } from '../../lib/measureTools';
import { Slider } from '../../ui/primitives';
import type { SectionShot } from './state';

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * Where to cut: the part's silhouette along its length (from the summary profile) with the cut drawn in vermilion.
 * Drag on the silhouette or use the slider (keyboard) to move the cut.
 */
export function ProfilePicker({ summary, direction, range, at, units, onChange }: { summary: PartSummary | null; direction: PartAxis; range: number; at: number; units: string; onChange: (v: number) => void }) {
  const svg = useRef<SVGSVGElement>(null);
  const W = 320, H = 86, padX = 12, padY = 12;
  const L = range > 0 ? range : 1;
  const slices = direction === 'length' ? (summary?.profile?.slices ?? []).filter(s => Number.isFinite(s.from) && Number.isFinite(s.to)) : [];
  const x = (t: number) => padX + (clamp(t, 0, L) / L) * (W - 2 * padX);
  const maxW = Math.max(...slices.map(s => s.width ?? 0), 1e-9);
  const half = (w: number | null | undefined) => ((w ?? 0) / maxW) * (H / 2 - padY);

  let body = '';
  if (slices.length > 1) {
    const top = slices.map(s => [x((s.from + s.to) / 2), H / 2 - half(s.width)]);
    const bot = slices.map(s => [x((s.from + s.to) / 2), H / 2 + half(s.width)]).reverse();
    const f = slices[0], l = slices[slices.length - 1];
    body = [
      `M${x(f.from).toFixed(1)},${(H / 2 - half(f.width)).toFixed(1)}`,
      ...top.map(p => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`),
      `L${x(l.to).toFixed(1)},${(H / 2 - half(l.width)).toFixed(1)}`,
      `L${x(l.to).toFixed(1)},${(H / 2 + half(l.width)).toFixed(1)}`,
      ...bot.map(p => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`),
      `L${x(f.from).toFixed(1)},${(H / 2 + half(f.width)).toFixed(1)}Z`,
    ].join(' ');
  }

  const fromPointer = (e: ReactPointerEvent<SVGSVGElement>) => {
    const r = svg.current?.getBoundingClientRect();
    if (!r || r.width <= 0) return;
    const px = ((e.clientX - r.left) / r.width) * W;
    onChange(+clamp(((px - padX) / (W - 2 * padX)) * L, 0, L).toFixed(2));
  };
  const step = L > 50 ? 0.1 : L > 5 ? 0.01 : 0.001;

  return (
    <div className="meas-profile">
      <svg
        ref={svg}
        viewBox={`0 0 ${W} ${H}`}
        className="meas-profile-svg"
        role="img"
        aria-label={`Cut at ${at.toFixed(2)} ${units} of ${L.toFixed(2)} ${units} along the ${direction}`}
        onPointerDown={e => {
          e.currentTarget.setPointerCapture(e.pointerId);
          fromPointer(e);
        }}
        onPointerMove={e => e.buttons & 1 && fromPointer(e)}
      >
        <line className="prof-centre" x1={4} x2={W - 4} y1={H / 2} y2={H / 2} />
        {body ? <path className="prof-body" d={body} /> : <rect className="prof-body" x={padX} y={H / 2 - 14} width={W - 2 * padX} height={28} rx={5} />}
        <line className="prof-end" x1={x(0)} x2={x(0)} y1={6} y2={H - 6} />
        <line className="prof-end" x1={x(L)} x2={x(L)} y1={6} y2={H - 6} />
        <line className="prof-cut-hit" x1={x(at)} x2={x(at)} y1={0} y2={H} />
        <line className="prof-cut" x1={x(at)} x2={x(at)} y1={3} y2={H - 3} />
        <path className="prof-cut-mark" d={`M${x(at) - 5},0 L${x(at) + 5},0 L${x(at)},6 Z M${x(at) - 5},${H} L${x(at) + 5},${H} L${x(at)},${H - 6} Z`} />
      </svg>
      <Slider value={clamp(at, 0, L)} min={0} max={L} step={step} onChange={v => onChange(+v.toFixed(3))} format={v => `${v.toFixed(2)} ${units}`} label={`Position along the ${direction}`} />
      <div className="meas-profile-scale">
        <span>start</span>
        <span className="mono">{fmtLen(L, 2)} {units}</span>
      </div>
    </div>
  );
}

/**
 * The cut outline as on a drawing: section hatching inside closed outlines, width and height dimension lines with
 * arrowheads and values. The plane's own axes are used, so the drawing is true to scale.
 */
export function SectionDrawing({ shot, units }: { shot: SectionShot; units: string }) {
  const uid = useId().replace(/:/g, '');
  const r = shot.result;
  if (!r.basis || !r.polylines?.length) return null;
  const { origin: o, u, v } = r.basis;
  const proj = (p: number[]) => {
    const d0 = p[0] - o[0], d1 = p[1] - o[1], d2 = p[2] - o[2];
    return [d0 * u[0] + d1 * u[1] + d2 * u[2], d0 * v[0] + d1 * v[1] + d2 * v[2]];
  };
  const [x0, y0] = r.bbox2d.min;
  const [x1, y1] = r.bbox2d.max;
  const w = Math.max(x1 - x0, 1e-9), h = Math.max(y1 - y0, 1e-9);
  const VW = 320, VH = 232, m = { l: 16, r: 78, t: 14, b: 48 };
  const s = Math.min((VW - m.l - m.r) / w, (VH - m.t - m.b) / h);
  const ox = m.l + (VW - m.l - m.r - w * s) / 2, oy = m.t + (VH - m.t - m.b - h * s) / 2;
  const X = (x: number) => ox + (x - x0) * s;
  const Y = (y: number) => oy + (y1 - y) * s;
  const path = (pl: number[][], close: boolean) => pl.map((p, i) => {
    const [a, b] = proj(p);
    return `${i ? 'L' : 'M'}${X(a).toFixed(1)},${Y(b).toFixed(1)}`;
  }).join('') + (close ? 'Z' : '');

  const closedIdx = r.polylines.map((_, i) => i).filter(i => r.closed?.[i]);
  const filled = closedIdx.map(i => path(r.polylines[i], true)).join(' ');
  const open = r.polylines.filter((_, i) => !r.closed?.[i]);

  const L = X(x0), R = X(x1), T = Y(y1), B = Y(y0);
  const dy = B + 20, dx = R + 20;
  const wText = `${fmtLen(r.width, 3)}`, hText = `${fmtLen(r.height, 3)}`;

  return (
    <figure className="meas-section">
      <svg viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label={`Cross-section ${wText} by ${hText} ${units}`}>
        <defs>
          <pattern id={`hatch-${uid}`} patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">
            <line className="sec-hatch" x1="0" y1="0" x2="0" y2="5" />
          </pattern>
          <marker id={`arr-${uid}`} viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path className="sec-arrow" d="M0,1.5 L10,5 L0,8.5 Z" />
          </marker>
        </defs>
        {/* centre lines, as on a drawing */}
        <line className="sec-centre" x1={(L + R) / 2} x2={(L + R) / 2} y1={T - 8} y2={B + 8} />
        <line className="sec-centre" x1={L - 8} x2={R + 8} y1={(T + B) / 2} y2={(T + B) / 2} />
        {filled && <path className="sec-fill" d={filled} fill={`url(#hatch-${uid})`} fillRule="evenodd" />}
        {filled && <path className="sec-outline" d={filled} />}
        {open.map((pl, i) => <path key={i} className="sec-outline is-open" d={path(pl, false)} />)}
        {/* width */}
        <line className="sec-ext" x1={L} x2={L} y1={B + 4} y2={dy + 5} />
        <line className="sec-ext" x1={R} x2={R} y1={B + 4} y2={dy + 5} />
        <line className="sec-dim" x1={L} x2={R} y1={dy} y2={dy} markerStart={`url(#arr-${uid})`} markerEnd={`url(#arr-${uid})`} />
        <text className="sec-value" x={(L + R) / 2} y={dy + 18} textAnchor="middle">{wText}<tspan className="sec-unit"> {units}</tspan></text>
        {/* height */}
        <line className="sec-ext" x1={R + 4} x2={dx + 5} y1={T} y2={T} />
        <line className="sec-ext" x1={R + 4} x2={dx + 5} y1={B} y2={B} />
        <line className="sec-dim" x1={dx} x2={dx} y1={T} y2={B} markerStart={`url(#arr-${uid})`} markerEnd={`url(#arr-${uid})`} />
        <text className="sec-value" x={dx + 8} y={(T + B) / 2 + 4} textAnchor="start">{hText}</text>
      </svg>
      <figcaption className="caption">
        Cut across the {shot.direction} at <span className="mono">{fmtLen(shot.at, 2)} {units}</span>
        {r.length ? <> · outline <span className="mono">{fmtLen(r.length, 2)} {units}</span></> : null}
        {closedIdx.length === 0 ? ' · the outline is open where the scan has gaps' : ''}
      </figcaption>
    </figure>
  );
}
