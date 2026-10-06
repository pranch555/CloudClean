import { useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import { useStore } from '../store';

export type ResolvedTheme = 'paper' | 'carbon';

const media = () => window.matchMedia('(prefers-color-scheme: dark)');

export function resolveTheme(theme: 'paper' | 'carbon' | 'system'): ResolvedTheme {
  if (theme === 'system') return media().matches ? 'carbon' : 'paper';
  return theme;
}

/** Applies the theme and text size to <html> and returns the resolved theme (re-renders on OS changes). */
export function useThemeEffect(): ResolvedTheme {
  const theme = useStore(s => s.theme);
  const textSize = useStore(s => s.textSize);
  const [resolved, setResolved] = useState<ResolvedTheme>(() => resolveTheme(theme));
  useEffect(() => {
    const apply = () => {
      const r = resolveTheme(theme);
      document.documentElement.dataset.theme = r;
      setResolved(r);
    };
    apply();
    if (theme !== 'system') return;
    const m = media();
    m.addEventListener('change', apply);
    return () => m.removeEventListener('change', apply);
  }, [theme]);
  useEffect(() => {
    document.documentElement.dataset.text = textSize;
  }, [textSize]);
  return resolved;
}

type TransitionDoc = Document & { startViewTransition?: (update: () => Promise<void>) => { ready: Promise<void>; finished: Promise<void> } };

/**
 * Paper <-> Carbon with an iris: the new theme opens as a circle from `origin` (the toggle) until it covers the
 * window. Uses the View Transitions API (the old page is a still picture, the new one is live underneath the
 * circle, so clicks are never blocked for long); without it, or with reduced motion, the theme simply switches.
 */
export function switchTheme(next: ResolvedTheme, origin?: Element | null) {
  const apply = () => {
    document.documentElement.dataset.theme = next;
    flushSync(() => useStore.getState().set({ theme: next }));
  };
  const doc = document as TransitionDoc;
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  if (!doc.startViewTransition || reduce || !origin) {
    apply();
    return;
  }
  const r = origin.getBoundingClientRect();
  const x = r.left + r.width / 2;
  const y = r.top + r.height / 2;
  const radius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
  const root = document.documentElement;
  root.classList.add('theme-iris');
  // (rendering is paused while this runs, so it must not wait for frames: the new view is live and the 3D view
  // repaints in the new colours as soon as the circle starts to open)
  const vt = doc.startViewTransition(async () => apply());
  vt.ready
    .then(() => {
      root.animate(
        { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
        { duration: 720, easing: 'cubic-bezier(0.65, 0, 0.35, 1)', pseudoElement: '::view-transition-new(root)' },
      );
    })
    .catch(() => undefined);
  vt.finished.finally(() => root.classList.remove('theme-iris'));
}

/** Current value of a CSS custom property on <html> (theme-aware colours for the WebGL viewer). */
export function cssVar(name: string, fallback = ''): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
