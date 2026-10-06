import { useEffect, useLayoutEffect, useRef } from 'react';
import { animate, reducedMotion, shake, stagger, svg, utils } from '../../lib/motion';

export type TickState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

const N = 24;
const TICKS = Array.from({ length: N }, (_, i) => {
  const a = (i / N) * Math.PI * 2;
  const s = Math.sin(a);
  const c = -Math.cos(a);
  return { x1: 12 + s * 7.6, y1: 12 + c * 7.6, x2: 12 + s * 10.6, y2: 12 + c * 10.6 };
});

/**
 * A turntable of 24 ticks for work in progress. Indeterminate: a lit head steps round tick by tick, like the
 * turntable's motor. With a fraction: that share of the ticks is lit. When the work finishes the ticks fold away
 * and a check draws itself; when it fails, a cross draws and the dial gives a short shake.
 */
export function TickLoader({ state, fraction, size = 22, label }: { state: TickState; fraction?: number | null; size?: number; label?: string }) {
  const ref = useRef<SVGSVGElement>(null);
  const spinning = state === 'running' && fraction == null;
  const lit = state === 'running' && fraction != null ? Math.round(Math.max(0, Math.min(1, fraction)) * N) : 0;

  useEffect(() => {
    const el = ref.current;
    if (!el || !spinning) return;
    const ticks = [...el.querySelectorAll<SVGLineElement>('.tl-tick')];
    const paint = (head: number) => ticks.forEach((t, i) => (t.style.opacity = String(Math.max(0.16, 1 - ((head - i + N) % N) / 9))));
    if (reducedMotion()) {
      paint(N - 1);
      return () => ticks.forEach(t => (t.style.opacity = ''));
    }
    const o = { p: 0 };
    const a = animate(o, { p: N, duration: 1150, ease: 'linear', loop: true, onUpdate: () => paint(Math.floor(o.p) % N) });
    return () => {
      a.pause();
      ticks.forEach(t => (t.style.opacity = ''));
    };
  }, [spinning]);

  // finished while on screen: before the first paint of the new state, so the mark never flashes complete
  const was = useRef(state);
  useLayoutEffect(() => {
    const el = ref.current;
    const before = was.current;
    const now = state;
    was.current = state;
    if (!el || reducedMotion() || (before !== 'running' && before !== 'queued')) return;
    if (now !== 'done' && now !== 'failed') return;
    const ticks = el.querySelectorAll('.tl-tick');
    animate(ticks, { opacity: [0.85, 0], duration: 260, delay: stagger(9), ease: 'out(2)' });
    const mark = el.querySelectorAll(now === 'done' ? '.tl-ring, .tl-check' : '.tl-ring, .tl-cross');
    const d = svg.createDrawable(mark);
    utils.set(d, { draw: '0 0' });
    animate(d, { draw: ['0 0', '0 1'], duration: 480, delay: stagger(140, { start: 180 }), ease: 'out(3)' });
    if (now === 'failed') setTimeout(() => shake(el, 3), 420);
  }, [state]);

  return (
    <svg ref={ref} className={`tick-loader is-${state} ${fraction != null ? 'is-determinate' : ''}`} viewBox="0 0 24 24" width={size} height={size} role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      {TICKS.map((t, i) => (
        <line key={i} className={`tl-tick ${i < lit ? 'is-lit' : ''}`} {...t} />
      ))}
      <circle className="tl-ring" cx="12" cy="12" r="10.4" transform="rotate(-90 12 12)" />
      <path className="tl-mark tl-check" d="M7.6 12.4l3 3 5.9-6.4" />
      <path className="tl-mark tl-cross" d="M8.9 8.9l6.2 6.2M15.1 8.9l-6.2 6.2" />
    </svg>
  );
}
