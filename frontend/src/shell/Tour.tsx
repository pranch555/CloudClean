import { useEffect, useLayoutEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { ArrowLeft, ArrowRight, X } from 'lucide-react';
import { local, useStore } from '../store';
import { Button, IconButton } from '../ui/primitives';

interface TourStep {
  target: string;
  title: string;
  body: string;
  side: 'below' | 'right' | 'left' | 'above';
}

const STEPS: TourStep[] = [
  { target: '.journey', side: 'below', title: 'Your part’s journey', body: 'Six steps from scan to export. Ticks show what is done; the red dot marks what we suggest next. Open any step at any time.' },
  { target: '.models', side: 'right', title: 'Every model of this part', body: 'Scans and every result made from them. Click one to work on it; the eye shows or hides it. Nothing is overwritten — each operation adds a new model.' },
  { target: '.side', side: 'left', title: 'What to do now', body: 'Each step says what it does in plain words, with good settings already chosen. The big button at the bottom does the work.' },
  { target: '.size-tag', side: 'below', title: 'The real size of the part', body: 'Length × width × height along the part itself, ignoring stray points. The ruler draws the dimensions onto the model.' },
  { target: '.ask-bar', side: 'above', title: 'Just ask', body: '“Rotate the turntable 45°”, “measure the shank diameter”, “delete the table”. The assistant can do anything you can, and shows its work.' },
];

export const TOUR_KEY = 'tourDone';

export function startTour() {
  local.set(TOUR_KEY, false);
  window.dispatchEvent(new Event('cloudclean:tour'));
}

/** First-run coach marks: a spotlight on each part of the workspace with one short explanation. */
export function Tour() {
  const screen = useStore(s => s.screen);
  const [index, setIndex] = useState<number | null>(null);
  const [rect, setRect] = useState<DOMRect | null>(null);

  useEffect(() => {
    const start = () => setIndex(0);
    window.addEventListener('cloudclean:tour', start);
    let t: number | undefined;
    if (screen === 'workspace' && !local.get(TOUR_KEY, false)) t = window.setTimeout(start, 1400);
    return () => {
      window.removeEventListener('cloudclean:tour', start);
      window.clearTimeout(t);
    };
  }, [screen]);

  // skip steps whose target is not on screen (e.g. no model shown yet → no size tag)
  const steps = STEPS.filter(s => document.querySelector(s.target));
  const step = index != null ? steps[index] : undefined;

  useLayoutEffect(() => {
    if (!step) return;
    const place = () => setRect(document.querySelector(step.target)?.getBoundingClientRect() ?? null);
    place();
    window.addEventListener('resize', place);
    return () => window.removeEventListener('resize', place);
  }, [step?.target, index]);

  useEffect(() => {
    if (index == null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') finish();
      else if (e.key === 'ArrowRight') next();
      else if (e.key === 'ArrowLeft') setIndex(i => Math.max(0, (i ?? 0) - 1));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  if (index == null || !step || !rect || screen !== 'workspace') return null;

  const finish = () => {
    local.set(TOUR_KEY, true);
    setIndex(null);
  };
  function next() {
    if (index == null) return;
    if (index >= steps.length - 1) finish();
    else setIndex(index + 1);
  }

  const pad = 8;
  const spot = { left: rect.left - pad, top: rect.top - pad, width: rect.width + pad * 2, height: rect.height + pad * 2 };
  const cardW = 340;
  const card: React.CSSProperties = { width: cardW };
  if (step.side === 'below') Object.assign(card, { top: spot.top + spot.height + 14, left: Math.min(Math.max(16, spot.left + spot.width / 2 - cardW / 2), window.innerWidth - cardW - 16) });
  if (step.side === 'above') Object.assign(card, { top: Math.max(16, spot.top - 200), left: Math.min(Math.max(16, spot.left + spot.width / 2 - cardW / 2), window.innerWidth - cardW - 16) });
  if (step.side === 'right') Object.assign(card, { top: Math.max(16, spot.top + 40), left: spot.left + spot.width + 16 });
  if (step.side === 'left') Object.assign(card, { top: Math.max(16, spot.top + 40), left: Math.max(16, spot.left - cardW - 16) });

  return createPortal(
    <div className="tour" role="dialog" aria-modal="true" aria-label={`Tour, ${index + 1} of ${steps.length}: ${step.title}`}>
      {/* four panels around the spotlight: dims everything else, no masks or giant shadows to go wrong */}
      <div className="tour-scrim" style={{ left: 0, top: 0, right: 0, height: Math.max(0, spot.top) }} />
      <div className="tour-scrim" style={{ left: 0, top: spot.top + spot.height, right: 0, bottom: 0 }} />
      <div className="tour-scrim" style={{ left: 0, top: spot.top, width: Math.max(0, spot.left), height: spot.height }} />
      <div className="tour-scrim" style={{ left: spot.left + spot.width, top: spot.top, right: 0, height: spot.height }} />
      <div className="tour-spot" style={spot} />
      <div className="tour-card" style={card}>
        <div className="row spread">
          <span className="caption">{index + 1} of {steps.length}</span>
          <IconButton size="sm" label="End the tour (Esc)" onClick={finish}><X size={15} /></IconButton>
        </div>
        <h3 className="tour-title">{step.title}</h3>
        <p className="tour-body">{step.body}</p>
        <div className="row spread">
          <span className="tour-dots" aria-hidden>{steps.map((_, i) => <span key={i} className={i === index ? 'is-on' : ''} />)}</span>
          <span className="row">
            {index > 0 && <Button size="sm" variant="ghost" icon={<ArrowLeft size={14} />} onClick={() => setIndex(index - 1)}>Back</Button>}
            <Button size="sm" variant="primary" onClick={next} autoFocus key={index}>{index >= steps.length - 1 ? 'Start working' : 'Next'} {index < steps.length - 1 && <ArrowRight size={14} aria-hidden />}</Button>
          </span>
        </div>
      </div>
    </div>,
    document.body,
  );
}
