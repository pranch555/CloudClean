import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { MapPin, X } from 'lucide-react';
import { clearGuide, useGuide } from '../lib/guide';

const SHOW_MS = 10000;
const PAD = 6;

/**
 * The ring the assistant puts around a control it is showing you ("Heights & steps is here"), with its name and
 * path. It follows the control while things move, and goes away on its own, on a click or with Esc. When the
 * control is not on screen right now (e.g. it only appears while scanning), a note with the path shows instead.
 */
export function GuideSpotlight() {
  const spot = useGuide(s => s.spot);
  const [rect, setRect] = useState<DOMRect | null>(null);

  useEffect(() => {
    if (!spot) return;
    let raf = 0;
    const track = () => {
      const el = spot.missing ? null : [...document.querySelectorAll(`[data-guide~="${CSS.escape(spot.id)}"]`)].find(e => e.getBoundingClientRect().width > 0);
      const r = el?.getBoundingClientRect() ?? null;
      setRect(cur => (cur && r && cur.x === r.x && cur.y === r.y && cur.width === r.width && cur.height === r.height ? cur : r));
      raf = requestAnimationFrame(track);
    };
    track();
    const timer = window.setTimeout(clearGuide, SHOW_MS);
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && clearGuide();
    const onDown = () => clearGuide();
    window.addEventListener('keydown', onKey);
    // the first click anywhere dismisses it (and still does what it was for)
    const arm = window.setTimeout(() => window.addEventListener('pointerdown', onDown, true), 400);
    return () => {
      cancelAnimationFrame(raf);
      window.clearTimeout(timer);
      window.clearTimeout(arm);
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onDown, true);
    };
  }, [spot]);

  if (!spot) return null;
  const vw = document.documentElement.clientWidth;
  const vh = window.innerHeight;
  if (spot.missing || !rect) {
    return createPortal(
      <div className="guide-note" role="status">
        <MapPin size={16} aria-hidden />
        <span>
          <b>{spot.label}</b> is at <span className="guide-where">{spot.where}</span>
        </span>
        <button type="button" className="icon-link" aria-label="Close" onClick={clearGuide}>
          <X size={14} />
        </button>
      </div>,
      document.body,
    );
  }
  const ring = { left: rect.left - PAD, top: rect.top - PAD, width: rect.width + 2 * PAD, height: rect.height + 2 * PAD };
  // the name tag sits below the ring when there is room, else above; always inside the window
  const tagW = Math.min(320, vw - 16);
  const below = ring.top + ring.height + 10 + 64 < vh;
  const tagLeft = Math.min(Math.max(ring.left, 8), vw - 8 - tagW);
  const tagTop = below ? ring.top + ring.height + 10 : Math.max(8, ring.top - 10 - 64);
  return createPortal(
    <>
      <div className="guide-ring" style={ring} aria-hidden />
      <div className="guide-tag" role="status" style={{ left: tagLeft, top: tagTop, maxWidth: tagW }}>
        <MapPin size={15} aria-hidden />
        <span>
          <b>{spot.label}</b>
          <span className="guide-where">{spot.where}</span>
        </span>
      </div>
    </>,
    document.body,
  );
}
