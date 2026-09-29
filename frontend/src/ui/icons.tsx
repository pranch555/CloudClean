import type { ReactElement } from 'react';
import type { Step } from '../lib/types';

type P = { size?: number; className?: string };

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.75,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
});

/** CloudClean mark: a "C" traced by scan points around a signal dot — raw points becoming a measured part. */
export function Logo({ size = 28, className }: P) {
  const dots = Array.from({ length: 9 }, (_, i) => {
    const a = (-40 + i * 35) * (Math.PI / 180); // 280° arc, open to the right
    return [16 - Math.cos(a) * 10.5, 16 - Math.sin(a) * 10.5];
  });
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" className={className} aria-hidden>
      <rect x="0.5" y="0.5" width="31" height="31" rx="9" fill="var(--primary)" />
      {dots.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={i === 0 || i === 8 ? 1.25 : 1.6} fill="var(--ink-on-primary)" opacity={0.55 + (i % 4) * 0.12} />
      ))}
      <circle cx="16" cy="16" r="3.2" fill="var(--signal)" />
    </svg>
  );
}

export function ScanGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <rect x="8" y="2.75" width="8" height="4.5" rx="1.5" />
      <path d="M10 7.25 5 17.25M14 7.25l5 10" strokeDasharray="1.6 2.2" />
      <ellipse cx="12" cy="18.5" rx="8" ry="2.75" />
      <path d="M9.5 15.25c.8-.6 1.6-.9 2.5-.9s1.7.3 2.5.9" />
    </svg>
  );
}

export function CleanGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M12 4a8 8 0 0 1 0 16" />
      <circle cx="12" cy="4" r="0.6" fill="currentColor" />
      <circle cx="6.4" cy="6.3" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="4.1" cy="11.2" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="5.2" cy="16.4" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="9.1" cy="19.5" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="2.2" cy="7.2" r="0.6" fill="currentColor" stroke="none" opacity="0.5" />
      <circle cx="1.8" cy="15" r="0.6" fill="currentColor" stroke="none" opacity="0.5" />
    </svg>
  );
}

export function AlignGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <rect x="3" y="3" width="11" height="11" rx="2.5" />
      <rect x="10" y="10" width="11" height="11" rx="2.5" strokeDasharray="2 2" />
      <path d="M12 12.5v-2M12 12.5h-2" />
    </svg>
  );
}

export function MeshGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M12 3 21 19H3z" />
      <path d="M7.5 11h9L12 19z" />
    </svg>
  );
}

export function MeasureGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M3 6v12M21 6v12" />
      <path d="M3 12h18" />
      <path d="m6.5 9.5-3.5 2.5 3.5 2.5M17.5 9.5l3.5 2.5-3.5 2.5" />
    </svg>
  );
}

export function ExportGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M4 13v5.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V13" />
      <path d="M12 15V3.5M7.5 8 12 3.5 16.5 8" />
    </svg>
  );
}

export const STEP_GLYPH: Record<Step, (p: P) => ReactElement> = {
  capture: ScanGlyph,
  clean: CleanGlyph,
  align: AlignGlyph,
  mesh: MeshGlyph,
  measure: MeasureGlyph,
  export: ExportGlyph,
};

/** Tiny dimension-line mark used in size readouts: |<->| */
export function DimMark({ size = 14, className }: P) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" className={className} aria-hidden>
      <path d="M2 4v8M14 4v8M2 8h12M4.5 6 2 8l2.5 2M11.5 6 14 8l-2.5 2" />
    </svg>
  );
}
