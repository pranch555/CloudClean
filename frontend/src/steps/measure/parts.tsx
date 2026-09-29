import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Check, CheckCircle2, Copy, Info, Lasso, SquareDashedMousePointer, TriangleAlert } from 'lucide-react';
import { clearSelection, setTool } from '../../lib/actions';
import { fmtCount } from '../../lib/format';
import { copyText, useSelectionOn } from '../../lib/measureTools';
import type { Asset } from '../../lib/types';
import { useStore } from '../../store';
import { IconButton, Switch } from '../../ui/primitives';

/** Numbered guidance: what to do, in order, with each finished step ticked. */
export function StepList({ children }: { children: ReactNode }) {
  return <ol className="meas-steps">{children}</ol>;
}

export function StepItem({ n, done, locked, children }: { n: number; done?: boolean; locked?: boolean; children: ReactNode }) {
  return (
    <li className={`meas-step ${done ? 'is-done' : ''} ${locked ? 'is-locked' : ''}`}>
      <span className="meas-step-n" aria-label={done ? `Step ${n}, done` : `Step ${n}`}>
        {done ? <Check size={13} strokeWidth={2.75} aria-hidden /> : n}
      </span>
      <div className="meas-step-body">{children}</div>
    </li>
  );
}

/** Box and lasso, the two ways to select part of the model. */
export function PickButtons({ disabled }: { disabled?: boolean }) {
  const tool = useStore(s => s.tool);
  return (
    <span className="meas-pick-tools">
      <button type="button" className={`chip ${tool === 'box' ? 'is-on' : ''}`} disabled={disabled} aria-pressed={tool === 'box'} onClick={() => setTool(tool === 'box' ? 'navigate' : 'box')}>
        <SquareDashedMousePointer size={14} aria-hidden /> Box <kbd className="kbd">B</kbd>
      </button>
      <button type="button" className={`chip ${tool === 'lasso' ? 'is-on' : ''}`} disabled={disabled} aria-pressed={tool === 'lasso'} onClick={() => setTool(tool === 'lasso' ? 'navigate' : 'lasso')}>
        <Lasso size={14} aria-hidden /> Lasso <kbd className="kbd">L</kbd>
      </button>
    </span>
  );
}

/** What is selected on the current model right now (and a way to clear it). */
export function SelectionStatus({ target, idle }: { target: Asset | undefined; idle?: ReactNode }) {
  const sel = useSelectionOn(target);
  const byId = useStore(s => s.byId);
  if (sel && sel.count > 0) {
    return (
      <span className="meas-picked">
        <CheckCircle2 size={14} aria-hidden />
        <span className="mono">{fmtCount(sel.count)}</span> points{sel.label ? ` · ${sel.label}` : ''}
        <button type="button" className="link" onClick={clearSelection}>Clear</button>
      </span>
    );
  }
  if (sel && sel.others.length) {
    return (
      <span className="meas-picked is-warn">
        <TriangleAlert size={14} aria-hidden />
        <span>The selection is on {byId.get(sel.others[0])?.name ?? 'another model'} — select on {target?.name ?? 'this model'}.</span>
      </span>
    );
  }
  return <span className="caption">{idle ?? 'Shift adds to a selection · V moves the view again'}</span>;
}

/** A "select this area" step: the instruction, box / lasso, and what is selected. */
export function SelectStep({ n, target, done, locked, children, extra }: { n: number; target: Asset | undefined; done?: boolean; locked?: boolean; children: ReactNode; extra?: ReactNode }) {
  const sel = useSelectionOn(target);
  const has = !!sel && sel.count > 0;
  return (
    <StepItem n={n} done={done ?? has} locked={locked}>
      <div>{children}</div>
      {!locked && (
        <div className="meas-pick">
          <PickButtons disabled={!target} />
          {extra ?? <SelectionStatus target={target} />}
        </div>
      )}
    </StepItem>
  );
}

/** A labelled switch with a one-line explanation under it. */
export function SwitchRow({ label, help, checked, disabled, onChange }: { label: string; help?: ReactNode; checked: boolean; disabled?: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className={`meas-switch ${disabled ? 'is-disabled' : ''}`}>
      <Switch checked={checked} disabled={disabled} onChange={onChange} label={label} />
      <div className="meas-switch-text">
        <span className="meas-switch-label">{label}</span>
        {help && <span className="caption">{help}</span>}
      </div>
    </div>
  );
}

/** Calm state for features the server behind this page does not have yet. */
export function NotHere({ what, children }: { what: string; children?: ReactNode }) {
  return (
    <div className="meas-unavail" role="note">
      <Info size={17} aria-hidden />
      <div>
        <div className="meas-unavail-title">Not available on this server yet</div>
        <p>{what} needs a newer CloudClean server. It appears here as soon as the server is updated{children ? '' : ' — nothing else to do.'}</p>
        {children}
      </div>
    </div>
  );
}

/** Copy button that confirms in place (a tick for a moment) instead of a toast. */
export function CopyButton({ text, label = 'Copy the value', size = 'sm', icon }: { text: string; label?: string; size?: 'sm' | 'md'; icon?: ReactNode }) {
  const [done, setDone] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return (
    <IconButton
      size={size}
      label={done ? 'Copied' : label}
      tip="top"
      className={done ? 'is-copied' : ''}
      onClick={async () => {
        if (!(await copyText(text))) return;
        setDone(true);
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setDone(false), 1400);
      }}
    >
      {done ? <Check size={15} aria-hidden /> : icon ?? <Copy size={15} aria-hidden />}
    </IconButton>
  );
}
