import { useState } from 'react';
import { fmtSigned } from '../../lib/format';
import { scalarColor, type ScalarStyle } from '../../viewer/colormaps';

/*
 * Distribution of signed distances (deviation from CAD, separation between two scans). Dataviz marks: bars capped
 * at 24 px with a 2 px surface gap, rounded data-ends and a square baseline, a recessive hairline axis, the
 * tolerance band behind the bars, and a tooltip per bar (hover or keyboard focus). Bars take their colour from the
 * same diverging map as the 3D deviation view — neutral inside the tolerance, blue below, red above — so the chart
 * and the model read as one.
 */
export function Histogram({ edges, counts, tolerance, unit, label = 'deviation' }: { edges: number[]; counts: number[]; tolerance: number; unit: string; label?: string }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!counts.length || edges.length !== counts.length + 1) return null;
  const W = 320, H = 136, padL = 4, padR = 4, padT = 10, padB = 24;
  const finite = edges.filter(Number.isFinite);
  const lo = Math.min(...finite), hi = Math.max(...finite);
  const n = counts.length;
  const total = counts.reduce((a, b) => a + b, 0) || 1;
  const max = Math.max(...counts, 1);
  const slot = (W - padL - padR) / n;
  const barW = Math.min(24, Math.max(slot - 2, 1));
  const x = (v: number) => padL + ((v - lo) / (hi - lo || 1)) * (W - padL - padR);
  const range = Math.max(Math.abs(lo), Math.abs(hi), tolerance * 1.0001);
  const style: ScalarStyle = { kind: 'diverging', min: -range, max: range, tolerance, steps: 0 };
  const bandL = Math.max(padL, x(-tolerance)), bandR = Math.min(W - padR, x(tolerance));
  const ticks = [lo, -tolerance, 0, tolerance, hi].filter((t, i, arr) => t >= lo && t <= hi && arr.indexOf(t) === i);
  const plotH = H - padT - padB;
  const within = counts.reduce((acc, c, i) => {
    const mid = (Number.isFinite(edges[i]) && Number.isFinite(edges[i + 1])) ? (edges[i] + edges[i + 1]) / 2 : NaN;
    return Math.abs(mid) <= tolerance ? acc + c : acc;
  }, 0);

  return (
    <div className="histogram">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Histogram of ${label}: ${Math.round((within / total) * 100)} % within ±${tolerance} ${unit}`} onMouseLeave={() => setHover(null)}>
        <rect x={bandL} y={padT} width={Math.max(bandR - bandL, 0)} height={plotH} className="hist-band" />
        {counts.map((c, i) => {
          const a = edges[i], b = edges[i + 1];
          const mid = Number.isFinite(a) && Number.isFinite(b) ? (a + b) / 2 : Number.isFinite(a) ? a : b;
          const h = c > 0 ? Math.max((c / max) * plotH, 1.5) : 0;
          const bx = padL + i * slot + (slot - barW) / 2;
          const top = H - padB - h;
          const r = Math.min(4, barW / 2, h);
          const d = h > 0 ? `M${bx},${H - padB} V${top + r} Q${bx},${top} ${bx + r},${top} H${bx + barW - r} Q${bx + barW},${top} ${bx + barW},${top + r} V${H - padB} Z` : '';
          return (
            <g key={i} tabIndex={0} role="img" aria-label={`${fmtSigned(a, 3)} to ${fmtSigned(b, 3)} ${unit}: ${c.toLocaleString()} points`} onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} onBlur={() => setHover(null)}>
              <rect className="hist-hit" x={padL + i * slot} y={padT} width={slot} height={plotH} />
              {d && <path d={d} fill={`#${scalarColor(mid, style).getHexString()}`} opacity={hover == null || hover === i ? 1 : 0.5} />}
            </g>
          );
        })}
        <line x1={padL} x2={W - padR} y1={H - padB + 0.5} y2={H - padB + 0.5} className="hist-axis" />
        {ticks.map((t, i) => (
          <text key={i} x={x(t)} y={H - 7} className="hist-tick" textAnchor={i === 0 ? 'start' : i === ticks.length - 1 ? 'end' : 'middle'}>
            {t === 0 ? '0' : fmtSigned(t, tolerance < 0.01 ? 3 : 2)}
          </text>
        ))}
      </svg>
      {hover != null && (
        <div className="hist-tooltip mono" role="status">
          {Number.isFinite(edges[hover]) ? fmtSigned(edges[hover], 3) : '−∞'} … {Number.isFinite(edges[hover + 1]) ? fmtSigned(edges[hover + 1], 3) : '+∞'} {unit}
          <b>{counts[hover].toLocaleString()} pts · {((counts[hover] / total) * 100).toFixed(1)} %</b>
        </div>
      )}
    </div>
  );
}
