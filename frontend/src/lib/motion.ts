import { useEffect, useLayoutEffect, useRef } from 'react';
import { animate, createTimeline, spring, stagger, svg, utils, type JSAnimation, type Timeline } from 'animejs';

/*
 * Motion for CloudClean (anime.js v4): one vocabulary for the whole app, so movement reads like one instrument.
 *
 *   enter(targets)       things arrive: rise a few pixels and fade in, in a quick stagger
 *   countUp(el, to)      a number rolls up to its value (mono digits, so nothing jumps)
 *   draw(paths)          a line or an arc draws itself (SVG stroke)
 *   springTo(el, props)  an indicator slides to its new place with a short spring
 *   pulse(el)            a brief "here" ring, e.g. when a pin or card is picked
 *
 * Rules: quick (150-600 ms), never in the way of a click, no movement on plain re-renders (only when something
 * arrives or changes), and everything snaps to its end state when the system asks for reduced motion.
 */

export const reducedMotion = (): boolean => {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
};

/** The house eases: `out` for arrivals, `inOut` for moves, `snap` for things that should feel decisive. */
export const EASE = { out: 'out(3)', inOut: 'inOut(3)', snap: 'outExpo', back: 'outBack(1.6)' } as const;

/** A short, lively spring for indicators and markers (slight overshoot, settles in about half a second). */
export const springy = () => spring({ bounce: 0.32, duration: 520 });
/** A calmer spring for bigger things (panels, cards). */
export const settle = () => spring({ bounce: 0.12, duration: 600 });

type Targets = Parameters<typeof animate>[0];
const empty = (t: Targets) => t == null || (Array.isArray(t) && t.length === 0) || (t instanceof NodeList && t.length === 0);

/** Things arrive: rise `y` px and fade in, staggered. Returns the animation (or null when nothing to do). */
export function enter(targets: Targets, opts: { y?: number; x?: number; delay?: number; step?: number; duration?: number; from?: 'first' | 'center' | 'last'; scale?: number } = {}): JSAnimation | null {
  if (empty(targets)) return null;
  const { y = 10, x = 0, delay = 0, step = 45, duration = 520, from = 'first', scale } = opts;
  if (reducedMotion()) {
    utils.set(targets, { opacity: 1, translateY: 0, translateX: 0, ...(scale != null ? { scale: 1 } : {}) });
    return null;
  }
  // the start state at once: later items of a stagger must not show at full strength until their turn
  utils.set(targets, { opacity: 0, translateY: y, ...(x ? { translateX: x } : {}), ...(scale != null ? { scale } : {}) });
  return animate(targets, {
    opacity: { from: 0, to: 1, duration: duration * 0.7, ease: 'out(2)' },
    translateY: { from: y, to: 0 },
    ...(x ? { translateX: { from: x, to: 0 } } : {}),
    ...(scale != null ? { scale: { from: scale, to: 1 } } : {}),
    duration,
    delay: stagger(step, { start: delay, from }),
    ease: EASE.out,
  });
}

/** A number rolls up to `to` (written into el.textContent with `decimals`). */
export function countUp(el: Element | null, to: number, opts: { from?: number; decimals?: number; duration?: number; delay?: number; format?: (v: number) => string } = {}): JSAnimation | null {
  if (!el || !Number.isFinite(to)) return null;
  const { from = 0, decimals = 0, duration = 900, delay = 0, format } = opts;
  const show = (v: number) => (format ? format(v) : v.toFixed(decimals));
  if (reducedMotion()) {
    el.textContent = show(to);
    return null;
  }
  const o = { v: from };
  el.textContent = show(from);
  return animate(o, {
    v: to,
    duration,
    delay,
    ease: EASE.snap,
    onUpdate: () => {
      el.textContent = show(o.v);
    },
  });
}

/** SVG lines or arcs draw themselves from nothing (`to`: how much of each is drawn at the end, 0-1). */
export function draw(targets: Targets, opts: { duration?: number; delay?: number; step?: number; to?: number; ease?: string } = {}): JSAnimation | null {
  if (empty(targets)) return null;
  const { duration = 900, delay = 0, step = 60, to = 1, ease = EASE.inOut } = opts;
  const drawables = svg.createDrawable(targets as Parameters<typeof svg.createDrawable>[0]);
  if (reducedMotion()) {
    utils.set(drawables, { draw: `0 ${to}` });
    return null;
  }
  utils.set(drawables, { draw: '0 0' });
  return animate(drawables, { draw: [`0 0`, `0 ${to}`], duration, delay: stagger(step, { start: delay }), ease });
}

/** An indicator slides to new values with a spring (e.g. { left: '40%' } or { translateX: 120 }). */
export function springTo(targets: Targets, props: Record<string, number | string>, opts: { calm?: boolean; delay?: number } = {}): JSAnimation | null {
  if (empty(targets)) return null;
  if (reducedMotion()) {
    utils.set(targets, props);
    return null;
  }
  return animate(targets, { ...props, ease: opts.calm ? settle() : springy(), delay: opts.delay ?? 0 });
}

/** A brief ring that says "here" (the element needs position: relative or absolute). */
export function pulse(el: HTMLElement | null, color = 'var(--signal)'): void {
  if (!el || reducedMotion()) return;
  const ring = document.createElement('span');
  ring.setAttribute('aria-hidden', 'true');
  Object.assign(ring.style, {
    position: 'absolute', inset: '-2px', borderRadius: 'inherit', pointerEvents: 'none',
    boxShadow: `0 0 0 2px ${color}`, opacity: '0.9',
  } satisfies Partial<CSSStyleDeclaration>);
  el.appendChild(ring);
  animate(ring, { scale: [1, 1.08], opacity: [0.9, 0], duration: 650, ease: EASE.out, onComplete: () => ring.remove() });
}

/** A timeline that does not play under reduced motion (callers jump it to the end with .seek(duration)). */
export function timeline(opts: Parameters<typeof createTimeline>[0] = {}): Timeline {
  return createTimeline({ ...opts, ...(reducedMotion() ? { autoplay: false } : {}) });
}

/**
 * Run an entrance once, when `key` changes (a new result, a new step), on the element `ref` points to. `run` gets
 * the root element and returns the animations it started; they jump to their end if the key changes again or the
 * component goes away, so nothing is left half faded.
 */
export function useEntrance<T extends HTMLElement>(key: unknown, run: (root: T) => (JSAnimation | Timeline | null | undefined)[] | JSAnimation | Timeline | null | void) {
  const ref = useRef<T>(null);
  useLayoutEffect(() => {
    const root = ref.current;
    if (!root) return;
    const out = run(root);
    const list = (Array.isArray(out) ? out : [out]).filter(Boolean) as (JSAnimation | Timeline)[];
    return () => {
      for (const a of list) {
        try {
          a.complete();
        } catch {
          /* already done */
        }
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return ref;
}

/** Re-run `run` whenever `value` changes after the first render (e.g. slide a marker to a new value). */
export function useOnChange<V>(value: V, run: (value: V, previous: V | undefined) => void) {
  const prev = useRef<V | undefined>(undefined);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      prev.current = value;
      return;
    }
    run(value, prev.current);
    prev.current = value;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
}

export { animate, stagger, utils, svg, createTimeline };

/* ------------------------------------------------------------------------------------------------------------------
 * More of the same vocabulary (home dial, journey axis, jobs, toasts):
 *
 *   scramble(el, text)     a readout shuffles through digits and settles on its value (TypeShuffle)
 *   shake(el)              a short spring shake that says "no" (a failed job)
 *   flipFrom(el, rect)     an element that moved in the layout glides from where it was (FLIP, with a spring)
 *   once(key)              true the first time per session: a screen's one longer choreographed moment
 *   invertInOut(p, power)  when an inOut(power) motion reaches progress p (to time things to a sweeping hand)
 * ---------------------------------------------------------------------------------------------------------------- */
import { text as animeText } from 'animejs';

/** A readout shuffles through `chars` and settles on `to` (mono digits keep the width). */
export function scramble(el: Element | null, to: string, opts: { chars?: string; delay?: number; from?: 'left' | 'right' | 'center' | 'random'; settle?: number; rate?: number } = {}): JSAnimation | null {
  if (!el) return null;
  if (reducedMotion()) {
    el.textContent = to;
    return null;
  }
  const { chars = '0-9', delay = 0, from = 'left', settle = 260, rate = 26 } = opts;
  return animate(el, {
    textContent: animeText.scrambleText({ text: to, chars, from, settleDuration: settle, revealRate: rate }),
    delay,
  });
}

/** A short spring shake (failure). */
export function shake(el: Element | null, amount = 5): JSAnimation | null {
  if (!el || reducedMotion()) return null;
  return animate(el, { translateX: [0, -amount, amount * 0.8, -amount * 0.5, amount * 0.25, 0], duration: 420, ease: 'out(2)' });
}

/** FLIP: `el` has just moved in the layout; it glides from `before` (its old rect) to where it is now. */
export function flipFrom(el: HTMLElement | null, before: DOMRect | undefined, opts: { calm?: boolean } = {}): JSAnimation | null {
  if (!el || !before || reducedMotion()) return null;
  const now = el.getBoundingClientRect();
  const dx = before.left - now.left;
  const dy = before.top - now.top;
  if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) return null;
  return animate(el, { translateX: [dx, 0], translateY: [dy, 0], ease: opts.calm ? settle() : springy() });
}

const played = new Set<string>();
/** True the first time it is asked for `key` in this session (one longer moment per screen, then quick ones). */
export function once(key: string): boolean {
  if (played.has(key)) return false;
  played.add(key);
  return true;
}

/** Inverse of anime's inOut(power) ease: the time fraction at which the eased progress reaches `p`. */
export function invertInOut(p: number, power = 2): number {
  // inOut(k)(t) = t < .5 ? (2t)^k / 2 : 1 - (2 - 2t)^k / 2
  const q = Math.min(1, Math.max(0, p));
  return q < 0.5 ? Math.pow(2 * q, 1 / power) / 2 : 1 - Math.pow(2 * (1 - q), 1 / power) / 2;
}
