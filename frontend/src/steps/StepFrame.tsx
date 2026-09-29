import type { ReactNode } from 'react';
import { ArrowRight, CheckCircle2, Info, TriangleAlert, XCircle } from 'lucide-react';
import type { Asset, Step } from '../lib/types';
import { fmtCount } from '../lib/format';
import { journeyStatus, ROLE_LABEL, roleOf, stepInfo, STEPS, suggestedStep } from '../lib/journey';
import { sizeText, Thumb } from '../shell/ModelsPanel';
import { useProjectAssets, useStore } from '../store';
import { STEP_GLYPH } from '../ui/icons';
import { Button } from '../ui/primitives';

/** Shared anatomy of a step panel: header, body (scrolls), footer (sticky main action + next step). */
export function StepFrame({ step, children, footer, title, purpose }: { step: Step; children: ReactNode; footer?: ReactNode; title?: string; purpose?: ReactNode }) {
  const info = stepInfo(step);
  const Glyph = STEP_GLYPH[step];
  return (
    <>
      <div className="side-scroll">
        <div className="step">
          <header className="step-head">
            <div className="step-kicker">
              <span className="step-glyph"><Glyph size={18} /></span>
              Step {info.n} of {STEPS.length}
              {info.optional && <span className="badge">optional</span>}
            </div>
            <h2 className="step-title">{title ?? info.verb}</h2>
            <p className="step-purpose">{purpose ?? info.purpose}</p>
          </header>
          {children}
        </div>
      </div>
      {footer && <footer className="side-foot">{footer}</footer>}
    </>
  );
}

/** The model a step works on. Prompts the user to pick one when none is selected. */
export function TargetCard({ asset, empty, label = 'Working on' }: { asset: Asset | undefined; empty?: ReactNode; label?: string }) {
  if (!asset) {
    return (
      <div className="target-card is-empty">
        <Info size={18} aria-hidden />
        <div className="small">{empty ?? 'Click a model in the list on the left to work on it.'}</div>
      </div>
    );
  }
  const role = roleOf(asset);
  return (
    <div className="target-card">
      <Thumb asset={asset} size={44} />
      <div className="grow">
        <div className="target-label">{label}</div>
        <div className="target-name truncate" title={asset.name}>{asset.name}</div>
        <div className="target-meta">
          <span className={`role role-${role}`}>{ROLE_LABEL[role]}</span>
          <span className="mono">{sizeText(asset)} mm</span>
          <span className="mono">{asset.kind === 'mesh' ? `${fmtCount(asset.stats.triangles)} tri` : `${fmtCount(asset.stats.points)} pts`}</span>
        </div>
      </div>
    </div>
  );
}

/** "Next: Align →" — goes to the suggested next step after this one. */
export function NextStepButton({ from, label }: { from: Step; label?: string }) {
  const assets = useProjectAssets();
  const projectId = useStore(s => s.projectId);
  const exported = useStore(s => !!(projectId && s.exported[projectId]));
  const goStep = useStore(s => s.goStep);
  const status = journeyStatus(assets, exported, false);
  const order = STEPS.map(s => s.id);
  let next = suggestedStep(status);
  if (order.indexOf(next) <= order.indexOf(from)) {
    next = order.slice(order.indexOf(from) + 1).find(s => status[s] !== 'skipped') ?? from;
  }
  if (next === from) return null;
  const info = stepInfo(next);
  return (
    <Button block variant="ghost" onClick={() => goStep(next)}>
      {label ?? `Next: ${info.label}`} <ArrowRight size={16} aria-hidden />
    </Button>
  );
}

/** Outcome of the last operation, in plain words, with its key numbers. */
export function ResultCard({ tone = 'ok', title, children, actions }: { tone?: 'ok' | 'warn' | 'danger' | 'info'; title: ReactNode; children?: ReactNode; actions?: ReactNode }) {
  const Icon = tone === 'ok' ? CheckCircle2 : tone === 'warn' ? TriangleAlert : tone === 'danger' ? XCircle : Info;
  return (
    <div className={`result-card tone-${tone}`} role="status">
      <div className="result-head">
        <Icon size={20} aria-hidden />
        <div className="result-title">{title}</div>
      </div>
      {children && <div className="result-body">{children}</div>}
      {actions && <div className="result-actions">{actions}</div>}
    </div>
  );
}

/** Big friendly choice between a few options, each with a one-line explanation. */
export function ChoiceCards<T extends string>({ value, onChange, options, columns = 1 }: { value: T; onChange: (v: T) => void; options: { value: T; title: string; body: string; icon?: ReactNode; badge?: string }[]; columns?: 1 | 2 | 3 }) {
  return (
    <div className={`choices cols-${columns}`} role="radiogroup">
      {options.map(o => (
        <button key={o.value} type="button" role="radio" aria-checked={o.value === value} className={`choice ${o.value === value ? 'is-on' : ''}`} onClick={() => onChange(o.value)}>
          <span className="choice-radio" aria-hidden />
          {o.icon && <span className="choice-icon">{o.icon}</span>}
          <span className="choice-text">
            <span className="choice-title">
              {o.title}
              {o.badge && <span className="badge badge-signal">{o.badge}</span>}
            </span>
            <span className="choice-body">{o.body}</span>
          </span>
        </button>
      ))}
    </div>
  );
}

/** A labelled block inside a step. */
export function Block({ title, children, aside, tone, guide }: { title?: ReactNode; children: ReactNode; aside?: ReactNode; tone?: 'signal' | 'sunken'; guide?: string }) {
  return (
    <section className={`block ${tone ? `block-${tone}` : ''}`} data-guide={guide}>
      {(title || aside) && (
        <div className="block-head">
          {title && <h3 className="block-title">{title}</h3>}
          {aside && <div className="block-aside">{aside}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

/** Tool tile (select, measure...) with icon, label and shortcut. */
export function ToolTile({ icon, label, sub, keys, active, disabled, onClick, guide }: { icon: ReactNode; label: string; sub?: string; keys?: string; active?: boolean; disabled?: boolean; onClick: () => void; guide?: string }) {
  return (
    <button type="button" className={`tile ${active ? 'is-active' : ''}`} disabled={disabled} onClick={onClick} aria-pressed={active} data-guide={guide}>
      <span className="tile-icon">{icon}</span>
      <span className="tile-text">
        <span className="tile-label">{label}</span>
        {sub && <span className="tile-sub">{sub}</span>}
      </span>
      {keys && <kbd className="kbd">{keys}</kbd>}
    </button>
  );
}
