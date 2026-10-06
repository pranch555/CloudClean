import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { JSAnimation } from 'animejs';
import { useGoldenFocus, useGoldenPins, type GoldenDimension as Dimension } from '../lib/golden';
import { animate, draw, enter, EASE, reducedMotion, springy } from '../lib/motion';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';

/** Below this many pixels on screen the line would be a smudge: a dot and the label instead. */
const SHORT_PX = 14;
/** arrowhead length and half width, end tick half length, gap between the line and its label, screen margin (px) */
const ARROW_L = 9;
const ARROW_W = 3.4;
const TICK = 7;
const GAP = 9;
const MARGIN = 8;

const keyOf = (d: Dimension | null) => (d ? `${d.ends.flat().join(',')}|${d.label}|${d.tone}` : '');

/**
 * Measure -> Golden model: the size the panel points at (useGoldenFocus.dimension), drawn on the 3D view like a
 * dimension on an engineering drawing: a line between the two measured surfaces with an arrowhead and a short tick at
 * each end, and the value in a pill beside its middle. Only while the check's mesh is on the stage; it follows the
 * camera and hides when an end leaves the view. Never takes a click.
 */
export function GoldenDimension() {
  const dimension = useGoldenFocus(s => s.dimension);
  const checkId = useGoldenPins(s => s.checkId);
  const onStage = useStore(s => !!checkId && s.visible.includes(checkId));
  const want = onStage ? dimension : null;
  const wantKey = keyOf(want);
  const [shown, setShown] = useState<Dimension | null>(want);
  const root = useRef<HTMLDivElement>(null);
  const leaving = useRef<JSAnimation | null>(null);

  // a new size shows at once (its drawing animates in); a cleared one fades out first
  useEffect(() => {
    leaving.current?.cancel();
    leaving.current = null;
    const el = root.current;
    if (want) {
      if (el) el.style.opacity = '';
      setShown(want);
      return;
    }
    if (!el || reducedMotion()) {
      setShown(null);
      return;
    }
    leaving.current = animate(el, {
      opacity: 0,
      duration: 150,
      ease: 'out(2)',
      onComplete: () => {
        leaving.current = null;
        setShown(null);
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wantKey]);

  useEffect(() => () => void leaving.current?.cancel(), []);

  if (!shown) return null;
  return (
    <div ref={root} className="gold-dim" data-tone={shown.tone} aria-hidden="true">
      <DimensionDrawing key={keyOf(shown)} dim={shown} />
    </div>
  );
}

function DimensionDrawing({ dim }: { dim: Dimension }) {
  const art = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const label = useRef<HTMLDivElement>(null);
  /** arrowheads, ticks and the dot grow in with this (0 -> 1, a little overshoot) */
  const pop = useRef({ s: reducedMotion() ? 1 : 0 });
  const place = useRef<() => void>(() => {});

  // follow the camera: project both ends every rendered frame and lay the drawing out in screen space
  useLayoutEffect(() => {
    const v = getViewer();
    const box = art.current, s = svg.current, lab = label.current;
    if (!v || !box || !s || !lab) return;
    const halves = [...s.querySelectorAll<SVGLineElement>('.gold-dim-half')];
    const ends = [...s.querySelectorAll<SVGGElement>('.gold-dim-end')];
    const line = s.querySelector<SVGGElement>('.gold-dim-line')!;
    const dot = s.querySelector<SVGGElement>('.gold-dim-dot')!;
    let size: { w: number; h: number } | null = null;
    let last = '';
    place.current = () => {
      const W = v.container.clientWidth, H = v.container.clientHeight;
      const a = v.project(dim.ends[0]), b = v.project(dim.ends[1]);
      const out = (p: { x: number; y: number } | null) => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.x < 0 || p.y < 0 || p.x > W || p.y > H;
      if (out(a) || out(b)) {
        if (last !== 'hidden') box.style.display = 'none';
        last = 'hidden';
        return;
      }
      const k = pop.current.s;
      const key = `${a!.x.toFixed(2)},${a!.y.toFixed(2)},${b!.x.toFixed(2)},${b!.y.toFixed(2)},${W},${H},${k.toFixed(3)}`;
      if (key === last) return;
      last = key;
      box.style.display = '';
      if (!size || !size.w) size = { w: lab.offsetWidth, h: lab.offsetHeight };
      const dx = b!.x - a!.x, dy = b!.y - a!.y;
      const len = Math.hypot(dx, dy);
      const mx = (a!.x + b!.x) / 2, my = (a!.y + b!.y) / 2;
      const short = len < SHORT_PX;
      line.style.display = short ? 'none' : '';
      dot.style.display = short ? '' : 'none';
      // the label's side: right of a steep line, above a flat one; the other side when that one leaves the view
      let nx = 1, ny = 0;
      if (short) {
        dot.setAttribute('transform', `translate(${mx} ${my}) scale(${k})`);
      } else {
        const ux = dx / len, uy = dy / len;
        nx = -uy;
        ny = ux;
        if (Math.abs(uy) > Math.abs(ux) ? nx < 0 : ny > 0) {
          nx = -nx;
          ny = -ny;
        }
        for (const [i, h] of halves.entries()) {
          const p = i % 2 ? b! : a!;
          h.setAttribute('x1', `${mx}`);
          h.setAttribute('y1', `${my}`);
          h.setAttribute('x2', `${p.x}`);
          h.setAttribute('y2', `${p.y}`);
        }
        // short lines get smaller arrowheads so the two never overlap
        const scale = k * Math.min(1, len / (ARROW_L * 3));
        for (const [i, g] of ends.entries()) {
          const p = i ? b! : a!;
          const angle = (Math.atan2(i ? uy : -uy, i ? ux : -ux) * 180) / Math.PI;
          g.setAttribute('transform', `translate(${p.x} ${p.y}) rotate(${angle}) scale(${scale})`);
        }
      }
      const { w, h } = size;
      const sideOf = (sx: number, sy: number) => {
        const reach = (short ? 6 : 0) + Math.abs(sx) * (w / 2) + Math.abs(sy) * (h / 2) + GAP;
        const cx = mx + sx * reach, cy = my + sy * reach;
        const over = Math.max(0, MARGIN - (cx - w / 2)) + Math.max(0, cx + w / 2 - (W - MARGIN)) + Math.max(0, MARGIN - (cy - h / 2)) + Math.max(0, cy + h / 2 - (H - MARGIN));
        return { cx, cy, over };
      };
      const one = sideOf(nx, ny);
      const other = one.over > 0 ? sideOf(-nx, -ny) : null;
      const pick = other && other.over < one.over ? other : one;
      const left = Math.min(Math.max(pick.cx - w / 2, MARGIN), Math.max(MARGIN, W - MARGIN - w));
      const top = Math.min(Math.max(pick.cy - h / 2, MARGIN), Math.max(MARGIN, H - MARGIN - h));
      lab.style.transform = `translate(${left.toFixed(1)}px, ${top.toFixed(1)}px)`;
    };
    place.current();
    const off = v.subscribe(() => place.current());
    return () => {
      off();
      place.current = () => {};
    };
  }, [dim]);

  // arrive: the line draws out from its middle to both ends, the ends pop, the value rises in
  useLayoutEffect(() => {
    const s = svg.current, pill = label.current?.firstElementChild;
    if (!s || !pill || reducedMotion()) return;
    const anims = [
      draw(s.querySelectorAll('.gold-dim-half'), { duration: 380, step: 0, ease: EASE.out }),
      animate(pop.current, { s: [0, 1], delay: 230, ease: springy(), onUpdate: () => place.current() }),
      enter(pill, { y: 8, delay: 120, duration: 420 }),
    ];
    return () => {
      for (const a of anims) a?.cancel();
    };
  }, []);

  const parts = dim.label.split(/(→)/);
  return (
    <div ref={art} className="gold-dim-art">
      <svg ref={svg} className="gold-dim-svg">
        <g className="gold-dim-line">
          <line className="gold-dim-half gold-dim-casing" />
          <line className="gold-dim-half gold-dim-casing" />
          <line className="gold-dim-half gold-dim-ink" />
          <line className="gold-dim-half gold-dim-ink" />
          {[0, 1].map(i => (
            <g key={i} className="gold-dim-end">
              <line className="gold-dim-casing" x1={0} y1={-TICK} x2={0} y2={TICK} />
              <line className="gold-dim-ink gold-dim-tick" x1={0} y1={-TICK} x2={0} y2={TICK} />
              <path className="gold-dim-head" d={`M0 0 L${-ARROW_L} ${-ARROW_W} L${-ARROW_L} ${ARROW_W} Z`} />
            </g>
          ))}
        </g>
        <g className="gold-dim-dot" style={{ display: 'none' }}>
          <circle r={4} />
        </g>
      </svg>
      <div ref={label} className="gold-dim-label">
        <div className="gold-dim-pill">
          <i className="gold-dim-pip" />
          {parts.map((p, i) => (p === '→' ? <span key={i} className="gold-dim-to">→</span> : p ? <span key={i}>{p.trim()}</span> : null))}
        </div>
      </div>
    </div>
  );
}
