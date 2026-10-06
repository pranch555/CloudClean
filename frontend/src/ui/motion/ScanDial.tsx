import { useId, useMemo, useRef, type ReactNode } from 'react';
import { ArrowRight } from 'lucide-react';
import type { Step } from '../../lib/types';
import type { StepStatus } from '../../lib/journey';
import { STEPS } from '../../lib/journey';
import { animate, EASE, invertInOut, once, reducedMotion, scramble, settle, springy, stagger, svg, timeline, useEntrance, useOnChange, utils } from '../../lib/motion';

/*
 * The home screen's signature: the scanner's turntable seen from above, drawn like an instrument dial.
 *
 *   outer ring   144 ticks (every 2.5 deg, a major one every 30 deg): the turntable's graduation
 *   step ring    the six steps of the project you worked on last: done in ink, next in signal, the rest faint
 *   platter      that project's part, sitting on the turntable
 *   scan head    the vermilion hand: it makes one turn (one scan pass, lighting the ticks it passes) and stops
 *                on the step to do next
 *   readouts     four instrument readouts in the corners (projects, models, points scanned, last scan)
 *
 * First visit in a session: the full pass (about 1.4 s). Later visits: the readouts shuffle, and the hand moves
 * only when the next step changed. Reduced motion: everything is simply there.
 */

const VB = 480;
const C = VB / 2;
const N_TICKS = 144;
const R_TICK_OUT = 228;
const R_TICK_IN = 218;
const R_MAJOR_IN = 207;
const R_HAIR = 200;
const R_LABEL = 186;
const R_ARC = 167;
const R_DASH = 152;
const R_PLATTER = 139;
const R_PART = 112;
const SWEEP_MS = 1150;
const SWEEP_AT = 120;

export interface DialReadout {
  label: string;
  value: string;
}

const polar = (r: number, deg: number): [number, number] => {
  const a = (deg * Math.PI) / 180;
  return [C + r * Math.sin(a), C - r * Math.cos(a)];
};

const f2 = (v: number) => v.toFixed(2);

const arcPath = (r: number, from: number, to: number) => {
  const [x0, y0] = polar(r, from);
  const [x1, y1] = polar(r, to);
  return `M${f2(x0)} ${f2(y0)}A${r} ${r} 0 ${to - from > 180 ? 1 : 0} 1 ${f2(x1)} ${f2(y1)}`;
};

/** An annular sector between radii r0..r1 and angles a0..a1 (degrees, 0 = top, clockwise). */
const sector = (r0: number, r1: number, a0: number, a1: number) => {
  const [x0, y0] = polar(r1, a0);
  const [x1, y1] = polar(r1, a1);
  const [x2, y2] = polar(r0, a1);
  const [x3, y3] = polar(r0, a0);
  return `M${f2(x0)} ${f2(y0)}A${r1} ${r1} 0 0 1 ${f2(x1)} ${f2(y1)}L${f2(x2)} ${f2(y2)}A${r0} ${r0} 0 0 0 ${f2(x3)} ${f2(y3)}Z`;
};

const hexagon = (r: number) => `M${Array.from({ length: 6 }, (_, i) => polar(r, i * 60 + 30).map(f2).join(' ')).join('L')}Z`;

/** Angle of the middle of step i's arc (0 = top, clockwise). */
const stepAngle = (i: number) => i * 60 + 30;

export function ScanDial({ status, next, cover, readouts, caption, onOpen, label }: {
  status: Record<Step, StepStatus> | null;
  next: Step | null;
  cover: string | null;
  readouts: DialReadout[];
  caption: ReactNode;
  onOpen: () => void;
  /** accessible name of the caption button */
  label: string;
}) {
  const uid = useId().replace(/:/g, '');
  const nextIndex = next ? STEPS.findIndex(s => s.id === next) : 0;
  const target = stepAngle(Math.max(0, nextIndex));
  const angle = useRef(target);
  const nums = useRef<(HTMLSpanElement | null)[]>([]);
  const values = readouts.map(r => r.value);
  // decided once per mount (in render, memoised: StrictMode's double effects must not use it up)
  const firstPass = useMemo(() => !reducedMotion() && once('home-dial'), []);

  const ticks = useMemo(
    () =>
      Array.from({ length: N_TICKS }, (_, i) => {
        const major = i % 12 === 0;
        return (
          <g key={i} transform={`rotate(${(i * 360) / N_TICKS} ${C} ${C})`}>
            <line className={`dial-tick ${major ? 'is-major' : ''}`} x1={C} x2={C} y1={C - R_TICK_OUT} y2={C - (major ? R_MAJOR_IN : R_TICK_IN)} />
          </g>
        );
      }),
    [],
  );

  const stateOf = (s: Step): string => (s === next ? 'next' : status ? status[s] : 'todo');
  const charsFor = (v: string) => (/^[\d.,\sMk]+$/.test(v) ? '0-9' : 'A-Z0-9');

  const root = useEntrance<HTMLDivElement>('dial', el => {
    const head = el.querySelector<SVGGElement>('.dial-head');
    const trail = el.querySelector<SVGGElement>('.dial-trail');
    const tickEls = [...el.querySelectorAll<SVGLineElement>('.dial-tick')];
    const arcs = [...el.querySelectorAll<SVGPathElement>('.dial-arc')];
    const labels = [...el.querySelectorAll<SVGTextElement>('.dial-label')];
    const rings = [...el.querySelectorAll<SVGCircleElement>('.dial-ring')];
    const platter = el.querySelector<SVGGElement>('.dial-platter');
    const glow = el.querySelector<SVGPathElement>('.dial-glow');
    const readoutEls = [...el.querySelectorAll<HTMLElement>('.dial-readout')];
    utils.set(head!, { rotate: target });
    angle.current = target;

    if (reducedMotion()) {
      nums.current.forEach((n, i) => n && (n.textContent = values[i] ?? ''));
      return null;
    }
    const shuffle = (delay: number) => nums.current.map((n, i) => scramble(n, values[i] ?? '', { delay: delay + i * 90, chars: charsFor(values[i] ?? ''), rate: 30 }));

    // a return visit: the readouts re-read, nothing else moves
    if (!firstPass) return shuffle(0);

    // ---- the first pass: one turn of the turntable, the hand ending on the step to do next
    const start = target - 360;
    /** when the hand (inOut(2) from `start` to `target`) passes `deg` */
    const passAt = (deg: number) => SWEEP_AT + invertInOut((((deg - start) % 360) + 360) % 360 / 360, 2) * SWEEP_MS;
    const ringDraw = svg.createDrawable(rings);
    utils.set(tickEls, { opacity: 0.12 });
    utils.set([...arcs, ...labels, ...readoutEls], { opacity: 0 });
    utils.set(ringDraw, { draw: '0 0' });
    nums.current.forEach(n => n && (n.textContent = ''));

    const tl = timeline({ defaults: { ease: EASE.out } });
    tl.add(ringDraw, { draw: ['0 0', '0 1'], duration: 1000, ease: EASE.inOut, delay: stagger(140) }, 0)
      .add(platter!, { opacity: [0, 1], scale: [0.9, 1], duration: 800, ease: settle() }, 40)
      .add(head!, { rotate: [start, target], duration: SWEEP_MS, ease: 'inOut(2)' }, SWEEP_AT)
      .add(trail!, { opacity: [0, 1], duration: 260, ease: 'out(2)' }, SWEEP_AT)
      .add(trail!, { opacity: 0, duration: 380, ease: 'out(2)' }, SWEEP_AT + SWEEP_MS - 300)
      .add(tickEls, {
        opacity: [0.12, 1],
        translateY: [{ to: -6, duration: 80, ease: 'out(2)' }, { to: 0, duration: 300, ease: 'out(3)' }],
        delay: (_?: unknown, i = 0) => passAt((i * 360) / N_TICKS),
        duration: 380,
      }, 0)
      .add(readoutEls, { opacity: [0, 1], translateY: [8, 0], duration: 480, delay: stagger(90) }, 280);
    arcs.forEach((arc, i) => {
      const at = passAt(i * 60 + 1.5);
      const d = svg.createDrawable(arc);
      utils.set(d, { draw: '0 0' });
      tl.add(arc, { opacity: [0, 1], duration: 1 }, at)
        .add(d, { draw: ['0 0', '0 1'], duration: 240, ease: 'out(2)' }, at)
        .add(labels[i], { opacity: [0, 1], translateY: [4, 0], duration: 320 }, passAt(stepAngle(i)) - 40);
    });
    if (glow) tl.add(glow, { opacity: [0, 0.5, 0], duration: 700, ease: 'inOut(2)' }, SWEEP_AT + SWEEP_MS - 120);
    return [tl, ...shuffle(420)];
  });

  // the hand follows when the next step changes while home is open
  useOnChange(target, to => {
    const head = root.current?.querySelector('.dial-head');
    if (!head) return;
    let delta = to - (((angle.current % 360) + 360) % 360);
    if (delta > 180) delta -= 360;
    if (delta < -180) delta += 360;
    angle.current += delta;
    if (reducedMotion()) utils.set(head, { rotate: angle.current });
    else animate(head, { rotate: angle.current, ease: springy() });
  });
  // a readout that changes while home is open (a scan finished, a project was made) re-reads
  useOnChange(values.join('|'), () => nums.current.forEach((n, i) => scramble(n, values[i] ?? '', { chars: charsFor(values[i] ?? '') })));

  const nextI = next ? STEPS.findIndex(s => s.id === next) : -1;

  return (
    <div className="dial" ref={root}>
      <div className="dial-stage">
        <div className="dial-face" onClick={onOpen} aria-hidden>
          <svg viewBox={`0 0 ${VB} ${VB}`} className="dial-svg">
            <defs>
              <radialGradient id={`${uid}-platter`} cx="50%" cy="36%" r="72%">
                <stop offset="0" style={{ stopColor: 'var(--vp-top)' }} />
                <stop offset="1" style={{ stopColor: 'var(--vp-bottom)' }} />
              </radialGradient>
              <clipPath id={`${uid}-part`}>
                <circle cx={C} cy={C} r={R_PART} />
              </clipPath>
              {/* the step names run along the ring; on the lower half the path runs the other way so they stay upright */}
              {STEPS.map((s, i) => {
                const a = stepAngle(i);
                const lower = a > 100 && a < 260;
                const [x0, y0] = polar(R_LABEL, lower ? a + 28 : a - 28);
                const [x1, y1] = polar(R_LABEL, lower ? a - 28 : a + 28);
                return <path key={s.id} id={`${uid}-lbl-${i}`} d={`M${f2(x0)} ${f2(y0)}A${R_LABEL} ${R_LABEL} 0 0 ${lower ? 0 : 1} ${f2(x1)} ${f2(y1)}`} />;
              })}
            </defs>

            <g className="dial-ticks">{ticks}</g>
            <circle className="dial-ring" cx={C} cy={C} r={R_HAIR} transform={`rotate(-90 ${C} ${C})`} />
            <circle className="dial-ring is-dashed" cx={C} cy={C} r={R_DASH} transform={`rotate(-90 ${C} ${C})`} />

            <g className="dial-steps">
              {nextI >= 0 && <path className="dial-glow" d={arcPath(R_ARC, nextI * 60 + 1.5, nextI * 60 + 58.5)} />}
              {STEPS.map((s, i) => (
                <path key={s.id} className={`dial-arc is-${stateOf(s.id)}`} d={arcPath(R_ARC, i * 60 + 1.5, i * 60 + 58.5)} />
              ))}
              {STEPS.map((s, i) => (
                <text key={s.id} className={`dial-label is-${stateOf(s.id)}`} dominantBaseline="central">
                  <textPath href={`#${uid}-lbl-${i}`} startOffset="50%" textAnchor="middle">
                    {s.n} · {s.label.toUpperCase()}
                  </textPath>
                </text>
              ))}
            </g>

            <g className="dial-platter">
              <circle className="dial-platter-disc" cx={C} cy={C} r={R_PLATTER} fill={`url(#${uid}-platter)`} />
              <circle className="dial-groove" cx={C} cy={C} r={R_PART + 13} />
              {cover ? (
                <image href={cover} x={C - R_PART} y={C - R_PART} width={R_PART * 2} height={R_PART * 2} preserveAspectRatio="xMidYMid meet" clipPath={`url(#${uid}-part)`} />
              ) : (
                <g className="dial-part-drawing">
                  <circle cx={C} cy={C} r={64} />
                  <circle cx={C} cy={C} r={46} strokeDasharray="3 5" />
                  <path d={hexagon(30)} />
                </g>
              )}
            </g>

            <g className="dial-head">
              <g className="dial-trail">
                {Array.from({ length: 8 }, (_, i) => (
                  <path key={i} style={{ opacity: 0.16 - i * 0.019 }} d={sector(R_PLATTER + 3, R_HAIR - 3, -4.5 * (i + 1), -4.5 * i)} />
                ))}
              </g>
              {/* the hand leaves a gap where the step names run, so the name it points at stays readable */}
              <path className="dial-hand" d={`M${C} ${C - R_PLATTER + 2}V${C - R_LABEL + 9}M${C} ${C - R_LABEL - 9}V${C - R_TICK_OUT - 3}`} />
              <path className="dial-hand-tip" d={`M${C} ${C - R_TICK_OUT - 3} l-6.5 -11 h13 z`} />
              <circle className="dial-hand-dot" cx={C} cy={C - R_PLATTER + 2} r={3.5} />
            </g>
            {!cover && <path className="dial-cross" d={`M${C - 7} ${C}h14M${C} ${C - 7}v14`} />}
          </svg>
        </div>

        {readouts.slice(0, 4).map((r, i) => (
          <div key={r.label} className={`dial-readout dial-readout-${i}`}>
            <span className="dial-readout-label">{r.label}</span>
            <span className="dial-readout-value" ref={n => { nums.current[i] = n; }} />
          </div>
        ))}
      </div>

      <button type="button" className="dial-caption" onClick={onOpen} aria-label={label}>
        {caption}
        <ArrowRight size={16} aria-hidden />
      </button>
    </div>
  );
}
