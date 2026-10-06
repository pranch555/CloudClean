import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import type { JSAnimation, Timeline } from 'animejs';
import { reducedMotion } from '../../../lib/motion';

/** True once the element has scrolled into view (at once under reduced motion): sections further down the panel
 *  play their entrance when the user gets to them, not while they are off screen. */
export function useSeen(ref: RefObject<Element | null>, key: unknown): boolean {
  const [seen, setSeen] = useState(() => reducedMotion());
  useEffect(() => {
    setSeen(reducedMotion());
    const el = ref.current;
    if (!el || reducedMotion() || typeof IntersectionObserver === 'undefined') {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver(
      entries => {
        if (entries.some(e => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin: '0px 0px -12% 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return seen;
}

type Played = JSAnimation | Timeline | null | undefined;

/**
 * An entrance that plays once per `key` (a check), when the element first scrolls into view. Until then the
 * element carries no `is-seen` class, which the CSS uses to keep what will animate hidden (never under reduced
 * motion). Returns the ref for the element and whether it has been seen.
 */
export function useReveal<T extends HTMLElement>(key: unknown, run: (root: T) => Played[] | Played | void): [RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const seen = useSeen(ref, key);
  useLayoutEffect(() => {
    const root = ref.current;
    if (!root || !seen) return;
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
  }, [key, seen]);
  return [ref, seen];
}
