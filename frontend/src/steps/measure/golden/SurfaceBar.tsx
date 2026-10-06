import { MousePointer2, Pin } from 'lucide-react';
import { animate, reducedMotion, EASE, enter } from '../../../lib/motion';
import { useReveal } from './useSeen';
import { SHARE, pct } from './words';

/* The surface of the golden model as one bar, one colour per verdict. Pointing at a colour (or its row) lights
   those places on the 3D model; clicking keeps them lit while you turn the model. */

export interface Share {
  key: string;
  value: number;
  color: string;
}

export function SurfaceBar({
  checkKey,
  shares,
  tolText,
  hot,
  pinned,
  canLight,
  onPoint,
  onPin,
}: {
  checkKey: string;
  shares: Share[];
  tolText: string;
  /** the colour being pointed at (or pinned) */
  hot: string | null;
  pinned: string | null;
  /** the golden check's model is on the 3D view (else the lights have nothing to show) */
  canLight: boolean;
  onPoint: (key: string | null) => void;
  onPin: (key: string) => void;
}) {
  const shown = shares.filter(s => s.value > 0 || s.key === 'good');
  const [ref, seen] = useReveal<HTMLDivElement>(checkKey, root => {
    const veil = root.querySelector<HTMLElement>('.gs-veil');
    if (!veil || reducedMotion()) {
      if (veil) veil.style.display = 'none';
      return null;
    }
    // a scan line sweeps across and leaves the colours behind it
    const a = animate(veil, {
      left: ['0%', '100%'],
      duration: 1100,
      delay: 350,
      ease: EASE.inOut,
      onComplete: () => {
        veil.style.opacity = '0';
      },
    });
    const b = enter(root.querySelectorAll('.gs-row'), { y: 8, step: 50, delay: 500 });
    return [a, b];
  });
  const hotShare = shown.find(s => s.key === hot);

  return (
    <div className={`gs ${hot ? 'has-hot' : ''} ${seen ? 'is-seen' : ''}`} ref={ref} onMouseLeave={() => onPoint(null)} data-guide="golden.surface">
      <div className="gs-track">
        <div className="gs-bar" role="group" aria-label="The golden surface by what the check found">
          {shown.map(s =>
            s.value > 0 ? (
              <button
                key={s.key}
                type="button"
                className={`gs-seg ${hot === s.key ? 'is-hot' : ''}`}
                style={{ flexGrow: s.value, background: s.color }}
                aria-label={`${SHARE[s.key]?.label ?? s.key}: ${pct(s.value)} %`}
                aria-pressed={pinned === s.key}
                onMouseEnter={() => onPoint(s.key)}
                onFocus={() => onPoint(s.key)}
                onBlur={() => onPoint(null)}
                onClick={() => onPin(s.key)}
              />
            ) : null,
          )}
        </div>
        <span className="gs-veil" aria-hidden><i /></span>
      </div>
      <p className="gs-readout" aria-live="polite">
        {hotShare ? (
          <>
            <b>{SHARE[hotShare.key]?.label}</b> <span className="mono">{pct(hotShare.value)} % of the surface</span>
            <span className="gs-line">
              {SHARE[hotShare.key]?.line(tolText)}
              {!canLight && <em> Set “Colour the model by” to What was found to see it on the model.</em>}
            </span>
          </>
        ) : (
          <span className="gs-line">
            <MousePointer2 size={13} aria-hidden className="gs-hint-icon" /> Point at a colour to light it up on the model. Click to keep it lit.
          </span>
        )}
      </p>
      <ul className="gs-rows">
        {shown.map(s => (
          <li key={s.key}>
            <button
              type="button"
              className={`gs-row ${hot === s.key ? 'is-hot' : ''} ${pinned === s.key ? 'is-pinned' : ''}`}
              aria-pressed={pinned === s.key}
              disabled={s.value <= 0}
              onMouseEnter={() => onPoint(s.key)}
              onFocus={() => onPoint(s.key)}
              onBlur={() => onPoint(null)}
              onClick={() => onPin(s.key)}
            >
              <i style={{ background: s.color }} aria-hidden />
              <span>{SHARE[s.key]?.label ?? s.key}</span>
              {pinned === s.key && <Pin size={12} className="gs-pin" aria-label="kept lit" />}
              <b className="mono">{pct(s.value)} %</b>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
