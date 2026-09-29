import type { ReactElement } from 'react';

/*
 * Measuring glyphs, drawn in the same language as the step glyphs (ui/icons.tsx): 24 px grid, 1.75 stroke, round
 * caps and joins, currentColor. Each one shows what the tool touches on the part, like a symbol on a drawing.
 */
type P = { size?: number; className?: string };
export type Glyph = (p: P) => ReactElement;

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

/** Two picked points and the straight line between them. */
export function PointsGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M7.4 15.9 16.6 8.1" strokeDasharray="2.2 2.4" />
      <circle cx="5.5" cy="17.5" r="2.25" fill="currentColor" stroke="none" />
      <circle cx="18.5" cy="6.5" r="2.25" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Calipers: a beam with two jaws closing on the part. */
export function CaliperGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M3 5h18" />
      <path d="M5 5v10.5L7.5 18" />
      <path d="M17 5v10.5L14.5 18" />
      <rect x="7.75" y="9" width="6.5" height="6" rx="1" strokeWidth="1.4" />
    </svg>
  );
}

/** A bolt with its overall length drawn above it. */
export function ExtentGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M3 5.5h18M3 3.5v4M21 3.5v4" />
      <rect x="3" y="10.5" width="5" height="9" rx="1" />
      <path d="M8 13h12a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1H8" />
    </svg>
  );
}

/** A stepped shaft seen from the side, with a height between two of its faces. */
export function StepsGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M3 8h6v8H3z" />
      <path d="M9 10h12v4H9" />
      <path d="M3 4.5h6M3 3v3M9 3v3" />
      <path d="M9 20h12M9 18.5v3M21 18.5v3" />
    </svg>
  );
}

/** Ø: a circle measured straight through its centre. */
export function DiameterGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <circle cx="12" cy="12" r="8.25" />
      <path d="M7.2 16.8 16.8 7.2" />
      <path d="M13.6 7.2h3.2v3.2M10.4 16.8H7.2v-3.2" />
    </svg>
  );
}

/** Two faces meeting at an angle, with the arc between them. */
export function AngleGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M4 19h16.5" />
      <path d="M4 19 14.5 5" />
      <path d="M11.5 19a7.5 7.5 0 0 0-2.9-5.9" />
    </svg>
  );
}

/** A surface between the two parallel planes that hold it. */
export function FlatnessGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M3 7h18M3 17h18" strokeWidth="1.25" opacity="0.6" />
      <path d="M3 12.5c1.5-2.2 3-2.2 4.5 0s3 2.2 4.5 0 3-2.2 4.5 0 3 2.2 4.5 0" />
    </svg>
  );
}

/** A ball with its equator. */
export function SphereGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <circle cx="12" cy="12" r="8.25" />
      <path d="M3.75 12a8.25 3.2 0 0 0 16.5 0" />
      <path d="M3.75 12a8.25 3.2 0 0 1 16.5 0" strokeDasharray="1.6 2.2" strokeWidth="1.25" />
    </svg>
  );
}

/** A cut face, hatched as on a drawing. */
export function SectionGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <circle cx="12" cy="12" r="7.5" />
      <path d="M12.9 4.6 4.6 12.9M15.9 5.9 5.9 15.9M18.1 8.1 8.1 18.1M19.4 11.1 11.1 19.4" strokeWidth="1.1" />
    </svg>
  );
}

/** A threaded rod: flanks crossing the shank. */
export function ThreadGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M7 3.5v17M17 3.5v17" />
      <path d="m7 8.5 10-3M7 13l10-3M7 17.5l10-3" />
    </svg>
  );
}

/** A CAD solid with scan points on one face. */
export function CadGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M12 3.2 19.8 7.6v8.8L12 20.8l-7.8-4.4V7.6z" />
      <path d="M4.2 7.6 12 12l7.8-4.4M12 12v8.8" />
      <circle cx="15.4" cy="13.4" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="17.2" cy="11" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="15.2" cy="16.6" r="0.9" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** A target: how close to the true size. */
export function AccuracyGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <circle cx="12" cy="12" r="8.25" />
      <circle cx="12" cy="12" r="4.25" />
      <circle cx="12" cy="12" r="1.1" fill="currentColor" stroke="none" />
      <path d="M12 1.8v3M12 19.2v3M1.8 12h3M19.2 12h3" strokeWidth="1.4" />
    </svg>
  );
}

/** A part drawn with its length and height dimensions. */
export function DimensionsGlyph({ size = 20, className }: P) {
  return (
    <svg {...base(size)} className={className}>
      <rect x="8" y="9" width="12.5" height="11" rx="1.5" />
      <path d="M8 5h12.5M8 3.5v3M20.5 3.5v3" />
      <path d="M4 9v11M2.5 9h3M2.5 20h3" />
    </svg>
  );
}

/** Glyph for a dimension kind in the results list. */
export function kindGlyph(kind: string): Glyph {
  switch (kind) {
    case 'caliper':
      return CaliperGlyph;
    case 'extent':
      return ExtentGlyph;
    case 'faces':
      return StepsGlyph;
    case 'diameter':
      return DiameterGlyph;
    case 'angle':
      return AngleGlyph;
    case 'flatness':
    case 'plane':
      return FlatnessGlyph;
    case 'sphere':
      return SphereGlyph;
    case 'section':
      return SectionGlyph;
    case 'reference':
      return AccuracyGlyph;
    case 'thread':
      return ThreadGlyph;
    default:
      return PointsGlyph;
  }
}
