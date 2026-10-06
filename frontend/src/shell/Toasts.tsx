import { useLayoutEffect, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from 'lucide-react';
import { useStore, type Toast } from '../store';
import { animate, draw, reducedMotion, shake, springy, utils } from '../lib/motion';

const ICON = { info: Info, ok: CheckCircle2, warn: AlertTriangle, error: XCircle };

/*
 * How long each toast lives, so its hairline can show the time left. The store's toast() takes `ms` (0 = stays
 * until closed) but does not keep it on the toast, so it is noted here as toasts are made; the defaults are the
 * store's (4.2 s, warnings 7 s, errors 9 s).
 */
const LIFE = new Map<number, number>();
{
  const make = useStore.getState().toast;
  useStore.setState({
    toast: t => {
      const id = make(t);
      LIFE.set(id, t.ms ?? (t.kind === 'error' ? 9000 : t.kind === 'warn' ? 7000 : 4200));
      return id;
    },
  });
}

/**
 * Toasts spring in from the side, leave with a short fade, and the rest of the stack glides to its new place
 * (FLIP). A hairline along the bottom runs down while the toast is on screen.
 */
export function Toasts() {
  const toasts = useStore(s => s.toasts);
  const dismiss = useStore(s => s.dismiss);
  const [leaving, setLeaving] = useState<Toast[]>([]);
  const [known, setKnown] = useState(toasts);
  const els = useRef(new Map<number, HTMLDivElement>());
  const tops = useRef(new Map<number, number>());

  // a toast that left the store stays a moment to animate out (decided while rendering, so it never unmounts)
  if (known !== toasts) {
    const gone = known.filter(t => !toasts.some(x => x.id === t.id));
    setKnown(toasts);
    if (gone.length && !reducedMotion()) setLeaving(l => [...l, ...gone]);
  }

  const shown = [...toasts, ...leaving.filter(l => !toasts.some(t => t.id === l.id))].sort((a, b) => a.id - b.id);

  // arrivals spring in, departures fade out, everyone else glides from where they were
  useLayoutEffect(() => {
    const now = new Map<number, number>();
    for (const [id, el] of els.current) now.set(id, el.getBoundingClientRect().top - Number(utils.get(el, 'translateY', false) || 0));
    if (!reducedMotion()) {
      for (const [id, el] of els.current) {
        const before = tops.current.get(id);
        const isLeaving = leaving.some(l => l.id === id) && !toasts.some(t => t.id === id);
        if (before == null) {
          animate(el, { opacity: [0, 1], translateX: [28, 0], scale: [0.96, 1], ease: springy() });
          const kind = el.dataset.kind;
          if (kind === 'ok' || kind === 'error') draw(el.querySelectorAll('.toast-icon circle, .toast-icon path'), { duration: 520, step: 140, delay: 80, ease: 'out(3)' });
          if (kind === 'error') setTimeout(() => shake(el, 4), 260);
        } else if (isLeaving && el.dataset.leaving !== '1') {
          el.dataset.leaving = '1';
          animate(el, {
            opacity: 0,
            translateX: 22,
            scale: 0.97,
            duration: 220,
            ease: 'in(2)',
            onComplete: () => setLeaving(l => l.filter(x => x.id !== id)),
          });
        } else {
          const dy = before - (now.get(id) ?? before);
          if (Math.abs(dy) > 0.5) animate(el, { translateY: [Number(utils.get(el, 'translateY', false) || 0) + dy, 0], ease: springy() });
        }
      }
    }
    tops.current = now;
  }, [`${shown.map(t => t.id)}|${leaving.map(t => t.id)}`]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="toasts" role="status" aria-live="polite">
      {shown.map(t => {
        const Icon = ICON[t.kind];
        const life = LIFE.get(t.id) ?? 0;
        const out = !toasts.some(x => x.id === t.id);
        return (
          <div
            key={t.id}
            ref={el => {
              if (el) els.current.set(t.id, el);
              else els.current.delete(t.id);
            }}
            className={`toast toast-${t.kind}`}
            data-kind={t.kind}
            aria-hidden={out || undefined}
          >
            <Icon size={16} className="toast-icon" aria-hidden />
            <div className="toast-text">
              <div className="toast-title">{t.title}</div>
              {t.body && <div className="toast-body">{t.body}</div>}
              {t.action && (
                <button type="button" className="link" onClick={() => { t.action!.run(); dismiss(t.id); }}>
                  {t.action.label}
                </button>
              )}
            </div>
            <button type="button" className="toast-close" aria-label="Dismiss" onClick={() => dismiss(t.id)}>
              <X size={13} />
            </button>
            {life > 0 && <span className="toast-life" style={{ animationDuration: `${life}ms` }} aria-hidden />}
          </div>
        );
      })}
    </div>
  );
}
