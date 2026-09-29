import { useEffect, useState } from 'react';
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

/** Current value of a CSS custom property on <html> (theme-aware colours for the WebGL viewer). */
export function cssVar(name: string, fallback = ''): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
