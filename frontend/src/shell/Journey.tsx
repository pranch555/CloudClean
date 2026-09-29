import { Check } from 'lucide-react';
import { journeyStatus, STEPS, suggestedStep } from '../lib/journey';
import { useProjectAssets, useStore } from '../store';

/** The six steps of a part, as a rail of pills: done ✓, current (ink), suggested next (signal dot). */
export function Journey() {
  const step = useStore(s => s.step);
  const goStep = useStore(s => s.goStep);
  const assets = useProjectAssets();
  const projectId = useStore(s => s.projectId);
  const exported = useStore(s => !!(projectId && s.exported[projectId]));
  const measured = useStore(s => s.measurements.some(m => m.b) || s.dims.length > 0 || !!s.thread);
  const status = journeyStatus(assets, exported, measured);
  const next = suggestedStep(status);

  return (
    <nav className="journey" aria-label="Steps for this part" data-guide="steps">
      {STEPS.map(s => {
        const st = status[s.id];
        const current = s.id === step;
        return (
          <button
            key={s.id}
            type="button"
            className={`journey-step ${current ? 'is-current' : ''} ${st === 'done' ? 'is-done' : ''} ${st === 'skipped' ? 'is-skipped' : ''} ${s.id === next ? 'is-next' : ''}`}
            aria-current={current ? 'step' : undefined}
            title={`${s.n}. ${s.verb} — ${s.purpose}${st === 'done' ? ' (done)' : st === 'skipped' ? ' (not needed for a single scan)' : s.id === next ? ' (suggested next)' : ''}`}
            onClick={() => goStep(s.id)}
          >
            <span className="journey-num">{st === 'done' && !current ? <Check size={15} strokeWidth={2.5} aria-label="done" /> : s.n}</span>
            <span className="journey-label">{s.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
