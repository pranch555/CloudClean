import { Fragment, useLayoutEffect, useRef } from 'react';
import { Check } from 'lucide-react';
import { journeyStatus, STEPS, suggestedStep, type StepStatus } from '../lib/journey';
import type { Step } from '../lib/types';
import { animate, draw, reducedMotion, springy, stagger, svg, utils } from '../lib/motion';
import { useProjectAssets, useStore } from '../store';

/*
 * The six steps of a part as a machine axis: numbered stations with a graduated ruler between them.
 *
 *   carriage   the current step is a dark carriage that travels along the axis with a spring (the leading edge
 *              first, the trailing edge a beat later, so it stretches a little like something with mass). It is
 *              drawn as a copy of the rail in "current" colours, clipped to the carriage, so the text changes
 *              colour exactly at its edge.
 *   ruler      the graduation between two stations fills in once the first of them is done: the line draws,
 *              then its ticks come up one after another.
 *   check      a step that gets done draws its check mark.
 */
export function Journey() {
  const step = useStore(s => s.step);
  const goStep = useStore(s => s.goStep);
  const assets = useProjectAssets();
  const projectId = useStore(s => s.projectId);
  const exported = useStore(s => !!(projectId && s.exported[projectId]));
  const measured = useStore(s => s.measurements.some(m => m.b) || s.dims.length > 0 || !!s.thread);
  const status = journeyStatus(assets, exported, measured);
  const next = suggestedStep(status);
  const nav = useRef<HTMLElement>(null);
  const index = STEPS.findIndex(s => s.id === step);
  const movingUntil = useRef(0);
  const shown = useRef(index);

  // ---- carriage: placed before paint on mount, follows resizes, springs to a new step
  useLayoutEffect(() => {
    const el = nav.current;
    if (!el) return;
    const target = () => {
      const btn = el.querySelectorAll<HTMLElement>(':scope > .journey-step')[shown.current];
      if (!btn) return null;
      return { l: btn.offsetLeft, r: el.clientWidth - btn.offsetLeft - btn.offsetWidth };
    };
    const put = () => {
      const t = target();
      if (t && performance.now() > movingUntil.current) utils.set(el, { '--cl': `${t.l}px`, '--cr': `${t.r}px` });
    };
    put();
    const ro = new ResizeObserver(put);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useLayoutEffect(() => {
    const el = nav.current;
    const from = shown.current;
    if (!el || from === index) return;
    shown.current = index;
    const btn = el.querySelectorAll<HTMLElement>(':scope > .journey-step')[index];
    if (!btn) return;
    const l = `${btn.offsetLeft}px`;
    const r = `${el.clientWidth - btn.offsetLeft - btn.offsetWidth}px`;
    if (reducedMotion()) {
      utils.set(el, { '--cl': l, '--cr': r });
      return;
    }
    // a new move takes over from wherever the carriage is (anime replaces the running tweens)
    const right = index > from;
    movingUntil.current = performance.now() + 900;
    // the leading edge springs (its overshoot only widens the carriage); the trailing edge must not overshoot,
    // or the carriage would cut into the label, so it eases in without a bounce
    animate(el, { [right ? '--cr' : '--cl']: right ? r : l, ease: springy() });
    animate(el, { [right ? '--cl' : '--cr']: right ? l : r, ease: 'out(4)', duration: 560, delay: 50 });
  }, [index]);

  // ---- ruler segments and check marks: animate only what changed
  const filled = STEPS.map(s => (status[s.id] !== 'todo' ? '1' : '0')).join('');
  const done = STEPS.map(s => (status[s.id] === 'done' ? '1' : '0')).join('');
  const was = useRef({ filled, done });
  useLayoutEffect(() => {
    const el = nav.current;
    const before = was.current;
    was.current = { filled, done };
    if (!el || reducedMotion() || (before.filled === filled && before.done === done)) return;
    const segs = el.querySelectorAll<HTMLElement>(':scope > .journey-seg');
    const btns = el.querySelectorAll<HTMLElement>(':scope > .journey-step');
    [...filled].forEach((c, i) => {
      const seg = segs[i];
      if (c !== '1' || before.filled[i] === '1' || !seg) return;
      draw(seg.querySelectorAll('.seg-base'), { duration: 320, ease: 'out(2)' });
      animate(seg.querySelectorAll('.seg-tick'), { scaleY: [0, 1], duration: 380, delay: stagger(80, { start: 160 }), ease: 'outBack(2)' });
    });
    [...done].forEach((c, i) => {
      const path = btns[i]?.querySelector('.journey-num svg path');
      if (c === '1' && before.done[i] !== '1' && path) animate(svg.createDrawable(path), { draw: ['0 0', '0 1'], duration: 420, delay: 120, ease: 'out(3)' });
    });
  }, [filled, done]);

  return (
    <nav className="journey" aria-label="Steps for this part" data-guide="steps" ref={nav}>
      <span className="journey-carriage" aria-hidden />
      {STEPS.map((s, i) => {
        const st = status[s.id];
        const current = s.id === step;
        return (
          <Fragment key={s.id}>
            {i > 0 && <Segment filled={status[STEPS[i - 1].id] !== 'todo'} />}
            <button
              type="button"
              className={stepClass(s.id, st, current, next)}
              aria-current={current ? 'step' : undefined}
              title={`${s.n}. ${s.verb} — ${s.purpose}${st === 'done' ? ' (done)' : st === 'skipped' ? ' (not needed for a single scan)' : s.id === next ? ' (suggested next)' : ''}`}
              onClick={() => goStep(s.id)}
            >
              <span className="journey-num">{st === 'done' && !current ? <Check size={15} strokeWidth={2.5} aria-label="done" /> : s.n}</span>
              <span className="journey-label">{s.label}</span>
            </button>
          </Fragment>
        );
      })}
      {/* the carriage's face: the same rail in "current" colours, clipped to the carriage */}
      <div className="journey-ink" aria-hidden inert>
        {STEPS.map((s, i) => (
          <Fragment key={s.id}>
            {i > 0 && <Segment filled={status[STEPS[i - 1].id] !== 'todo'} />}
            <button type="button" tabIndex={-1} className={stepClass(s.id, status[s.id], s.id === step, null)}>
              <span className="journey-num">{s.n}</span>
              <span className="journey-label">{s.label}</span>
            </button>
          </Fragment>
        ))}
      </div>
    </nav>
  );
}

function stepClass(id: Step, st: StepStatus, current: boolean, next: Step | null) {
  return `journey-step ${current ? 'is-current' : ''} ${st === 'done' ? 'is-done' : ''} ${st === 'skipped' ? 'is-skipped' : ''} ${id === next ? 'is-next' : ''}`;
}

/** The graduation between two stations: a line with a tall tick between two short ones. */
function Segment({ filled }: { filled: boolean }) {
  return (
    <span className={`journey-seg ${filled ? 'is-filled' : ''}`} aria-hidden>
      <svg viewBox="0 0 20 16" width="20" height="16">
        <path className="seg-base" d="M1 8H19" />
        <path className="seg-tick" d="M5 5v6" />
        <path className="seg-tick" d="M10 2.5v11" />
        <path className="seg-tick" d="M15 5v6" />
      </svg>
    </span>
  );
}
