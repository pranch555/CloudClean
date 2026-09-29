import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

type Side = 'top' | 'bottom' | 'left' | 'right';

interface Tip {
  el: HTMLElement;
  text: string;
  side: Side;
}

const EDGE = 8;
const GAP = 10;
const HOVER_DELAY = 250;

/** The side a tooltip prefers: `data-tip-side`, else away from the rail it sits in, else above. */
function preferredSide(el: HTMLElement): Side {
  const s = el.dataset.tipSide as Side | undefined;
  if (s) return s;
  if (el.closest('.tool-rail')) return 'right';
  if (el.closest('.view-rail')) return 'left';
  return 'top';
}

/**
 * Tooltips for every element with `data-tip`. One layer at the top of the page draws them, so a panel can't
 * clip them, and places each where it fits: on its preferred side when there is room, else the opposite one,
 * shifted to stay inside the window. Hover shows it after a short delay; keyboard focus shows it at once.
 */
export function TooltipLayer() {
  const [tip, setTip] = useState<Tip | null>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let timer = 0;
    const show = (el: HTMLElement, delay: number) => {
      window.clearTimeout(timer);
      const text = el.dataset.tip;
      if (!text) return;
      timer = window.setTimeout(() => setTip({ el, text, side: preferredSide(el) }), delay);
    };
    const hide = () => {
      window.clearTimeout(timer);
      setTip(null);
    };
    const over = (e: PointerEvent) => {
      const el = (e.target as Element | null)?.closest?.<HTMLElement>('[data-tip]');
      if (el) show(el, HOVER_DELAY);
    };
    const out = (e: PointerEvent) => {
      const el = (e.target as Element | null)?.closest?.('[data-tip]');
      const to = (e.relatedTarget as Element | null)?.closest?.('[data-tip]');
      if (el && el !== to) hide();
    };
    const focus = (e: FocusEvent) => {
      const el = (e.target as Element | null)?.closest?.<HTMLElement>('[data-tip]');
      if (el && el.matches(':focus-visible')) show(el, 0);
    };
    document.addEventListener('pointerover', over);
    document.addEventListener('pointerout', out);
    document.addEventListener('pointerdown', hide, true);
    document.addEventListener('focusin', focus);
    document.addEventListener('focusout', hide);
    window.addEventListener('scroll', hide, true);
    window.addEventListener('resize', hide);
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && hide();
    document.addEventListener('keydown', onKey);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener('pointerover', over);
      document.removeEventListener('pointerout', out);
      document.removeEventListener('pointerdown', hide, true);
      document.removeEventListener('focusin', focus);
      document.removeEventListener('focusout', hide);
      window.removeEventListener('scroll', hide, true);
      window.removeEventListener('resize', hide);
      document.removeEventListener('keydown', onKey);
    };
  }, []);

  // the text of a tip can change while it shows (e.g. "Copy" → "Copied")
  useEffect(() => {
    if (!tip) return;
    const mo = new MutationObserver(() => {
      if (!tip.el.isConnected) return setTip(null);
      const text = tip.el.dataset.tip;
      if (!text) setTip(null);
      else if (text !== tip.text) setTip({ ...tip, text });
    });
    mo.observe(tip.el, { attributes: true, attributeFilter: ['data-tip'] });
    return () => mo.disconnect();
  }, [tip]);

  useLayoutEffect(() => {
    if (!tip || !box.current) {
      setPos(null);
      return;
    }
    const a = tip.el.getBoundingClientRect();
    const w = box.current.offsetWidth;
    const h = box.current.offsetHeight;
    const vw = document.documentElement.clientWidth;
    const vh = window.innerHeight;
    const fits: Record<Side, boolean> = {
      top: a.top - GAP - h >= EDGE,
      bottom: a.bottom + GAP + h <= vh - EDGE,
      left: a.left - GAP - w >= EDGE,
      right: a.right + GAP + w <= vw - EDGE,
    };
    const opposite: Record<Side, Side> = { top: 'bottom', bottom: 'top', left: 'right', right: 'left' };
    const side = fits[tip.side] ? tip.side : fits[opposite[tip.side]] ? opposite[tip.side] : tip.side;
    let top = side === 'top' ? a.top - GAP - h : side === 'bottom' ? a.bottom + GAP : a.top + a.height / 2 - h / 2;
    let left = side === 'left' ? a.left - GAP - w : side === 'right' ? a.right + GAP : a.left + a.width / 2 - w / 2;
    left = Math.min(Math.max(left, EDGE), vw - EDGE - w);
    top = Math.min(Math.max(top, EDGE), vh - EDGE - h);
    setPos({ top, left });
  }, [tip]);

  if (!tip) return null;
  return createPortal(
    <div ref={box} className="tooltip" role="tooltip" style={pos ? { top: pos.top, left: pos.left } : { top: 0, left: 0, visibility: 'hidden' }}>
      {tip.text}
    </div>,
    document.body,
  );
}
