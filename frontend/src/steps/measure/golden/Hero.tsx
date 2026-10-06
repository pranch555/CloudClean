import { CheckCircle2, TriangleAlert } from 'lucide-react';
import type { GoldenGrade } from '../../../lib/golden';
import { animate, countUp, draw, enter, reducedMotion, stagger, useEntrance, utils, EASE } from '../../../lib/motion';
import { SHARE, pct } from './words';

/* The verdict of a golden check in one look: an instrument dial (a tick ring, one coloured arc per share of the
   surface, the match in the middle) beside the verdict in words, and three numbers that jump to their sections. */

export interface Slice {
  key: string;
  value: number;
  color: string;
}

const C = 80; // centre of the 160 x 160 dial
const R_ARC = 57;
const TICKS = 60;

const polar = (deg: number, r: number) => {
  const a = (deg * Math.PI) / 180;
  return [C + r * Math.sin(a), C - r * Math.cos(a)];
};

function arcPath(from: number, to: number, r: number) {
  const [x1, y1] = polar(from, r);
  const [x2, y2] = polar(to, r);
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 ${to - from > 180 ? 1 : 0} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

/** Start and end angle of every slice: the shares round the dial, small ones kept visible, with hairline gaps. */
function layout(slices: Slice[]) {
  const shown = slices.filter(s => s.value > 0);
  const gap = shown.length > 1 ? 2.2 : 0;
  const minSweep = 3.2;
  const free = 360 - gap * shown.length;
  const raw = shown.map(s => Math.max((s.value / 100) * free, minSweep));
  const scale = free / raw.reduce((a, b) => a + b, 0);
  let at = gap / 2;
  return shown.map((s, i) => {
    const sweep = raw[i] * scale;
    const out = { ...s, from: at, to: at + sweep };
    at += sweep + gap;
    return out;
  });
}

export function GoldenHero({
  checkKey,
  grade,
  title,
  sub,
  slices,
  tolText,
  stats,
  onPoint,
}: {
  checkKey: string;
  grade: GoldenGrade;
  title: string;
  sub: string;
  slices: Slice[];
  tolText: string;
  stats: { value: number; decimals: number; suffix?: string; label: string; fail?: boolean; disabled?: boolean; onClick: () => void }[];
  onPoint: (key: string | null) => void;
}) {
  const arcs = layout(slices);
  const match = grade.matchPct ?? 0;
  const ref = useEntrance<HTMLDivElement>(checkKey, root => {
    const out = [];
    const ticks = root.querySelectorAll('.gh-tick');
    if (!reducedMotion()) {
      // the tick ring sweeps on clockwise, then the ticks inside the matching share light up
      utils.set(ticks, { opacity: 0 });
      out.push(animate(ticks, { opacity: [0, 1], duration: 260, delay: stagger(9), ease: 'out(2)' }));
      out.push(animate(root.querySelectorAll('.gh-tick.is-lit'), { strokeWidth: [1.2, 2], duration: 420, delay: stagger(7, { start: 420 }), ease: EASE.out }));
    }
    let start = 260;
    for (const el of root.querySelectorAll<SVGPathElement>('.gh-arc')) {
      const sweep = Number(el.dataset.sweep) || 10;
      const dur = Math.max(140, (sweep / 360) * 1100);
      out.push(draw(el, { duration: dur, delay: start, ease: sweep > 180 ? 'inOut(3)' : 'out(2)' }));
      start += dur * 0.85;
    }
    out.push(countUp(root.querySelector('.gh-num'), match, { decimals: match >= 10 ? 0 : 1, duration: 1300, delay: 260, format: v => (v >= 99.95 ? '100' : v.toFixed(match >= 10 ? 0 : 1)) }));
    out.push(enter(root.querySelectorAll('.gh-word'), { y: 16, step: 55, delay: 180, duration: 600 }));
    out.push(enter(root.querySelectorAll('.gh-kicker, .gh-sub'), { y: 6, step: 120, delay: 120 }));
    out.push(enter(root.querySelectorAll('.gh-stat'), { y: 12, step: 70, delay: 520 }));
    root.querySelectorAll<HTMLElement>('.gh-stat-value b').forEach((el, i) => {
      const s = stats[i];
      if (s) out.push(countUp(el, s.value, { decimals: s.decimals, duration: 900, delay: 600 + i * 70 }));
    });
    return out;
  });

  return (
    <div className="gh" ref={ref}>
      <section className={`gh-hero gh-${grade.tone}`} role="status" aria-label={`${grade.label}. ${sub}`} data-guide="golden.verdict">
        <svg className="gh-dial" viewBox="0 0 160 160" width="148" height="148" aria-hidden onMouseLeave={() => onPoint(null)}>
          <g className="gh-ticks">
            {Array.from({ length: TICKS }, (_, i) => {
              const deg = (i * 360) / TICKS;
              const long = i % 5 === 0;
              const [x1, y1] = polar(deg, long ? 68 : 71.5);
              const [x2, y2] = polar(deg, 76);
              return <line key={i} className={`gh-tick ${long ? 'is-long' : ''} ${deg <= (match / 100) * 360 ? 'is-lit' : ''}`} x1={x1} y1={y1} x2={x2} y2={y2} />;
            })}
          </g>
          <circle className="gh-track" cx={C} cy={C} r={R_ARC} />
          {arcs.map(a => (
            <path
              key={a.key}
              className="gh-arc"
              data-key={a.key}
              data-sweep={a.to - a.from}
              d={arcPath(a.from, a.to, R_ARC)}
              stroke={a.color}
              onMouseEnter={() => onPoint(a.key)}
            >
              <title>{`${SHARE[a.key]?.label ?? a.key}: ${pct(a.value)} %`}</title>
            </path>
          ))}
          <circle className="gh-inner" cx={C} cy={C} r={44} />
        </svg>
        <div className="gh-readout" aria-hidden>
          <b><span className="gh-num">{match >= 99.95 ? '100' : match.toFixed(match >= 10 ? 0 : 1)}</span><small>%</small></b>
          <span>match</span>
        </div>
        <div className="gh-text">
          <div className="gh-kicker">{grade.tone === 'pass' ? <CheckCircle2 size={15} aria-hidden /> : <TriangleAlert size={15} aria-hidden />} Golden model check</div>
          <h3 className="gh-title">
            {title.split(' ').map((w, i) => (
              <span key={i} className="gh-word">{w}{' '}</span>
            ))}
          </h3>
          <p className="gh-sub">{sub}</p>
          <p className="gh-tol mono">tolerance {tolText}</p>
        </div>
      </section>
      <div className="gh-stats">
        {stats.map(s => (
          <button key={s.label} type="button" className={`gh-stat ${s.fail ? 'is-fail' : ''}`} onClick={s.onClick} disabled={s.disabled}>
            <span className="gh-stat-value mono">
              <b>{s.value.toFixed(s.decimals)}</b>
              {s.suffix && <small>{s.suffix}</small>}
            </span>
            <span className="gh-stat-label">{s.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
