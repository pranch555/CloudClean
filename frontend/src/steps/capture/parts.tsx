import type { ReactNode } from 'react';
import { create } from 'zustand';
import { Info, TriangleAlert, XCircle } from 'lucide-react';
import { Button, Dialog } from '../../ui/primitives';

/*
 * Small pieces shared by the Scan step's blocks: one action at a time with its error shown where it was started,
 * a callout, and a confirmation dialog in the app's own style (never the browser's confirm()).
 */

interface ActionState {
  busy: string | null;
  error: { label: string; message: string } | null;
  /** the automation job started by the last save, if any */
  savedJob: string | null;
}

export const useCaptureAction = create<ActionState>(() => ({ busy: null, error: null, savedJob: null }));

/** Run a user action; its failure (one sentence from the server) is kept for the block that shows `label`. */
export async function runAction(label: string, fn: () => Promise<unknown>): Promise<boolean> {
  useCaptureAction.setState({ busy: label, error: null });
  try {
    await fn();
    return true;
  } catch (err) {
    useCaptureAction.setState({ error: { label, message: (err as Error).message || 'That did not work.' } });
    return false;
  } finally {
    useCaptureAction.setState({ busy: null });
  }
}

export function ActionError({ labels }: { labels: string[] }) {
  const error = useCaptureAction(s => s.error);
  if (!error || !labels.includes(error.label)) return null;
  return (
    <p className="err-text cap-error" role="alert">
      <XCircle size={14} aria-hidden />
      <span>{error.message}</span>
    </p>
  );
}

export function Callout({ tone = 'info', children, action }: { tone?: 'info' | 'warn' | 'danger'; children: ReactNode; action?: ReactNode }) {
  const Icon = tone === 'info' ? Info : tone === 'warn' ? TriangleAlert : XCircle;
  return (
    <div className={`cap-callout tone-${tone}`} role={tone === 'info' ? undefined : 'note'}>
      <Icon size={16} aria-hidden />
      <div className="grow">{children}</div>
      {action && <div className="cap-callout-action">{action}</div>}
    </div>
  );
}

export interface ConfirmChoice {
  label: string;
  variant?: 'primary' | 'secondary' | 'danger' | 'signal' | 'ghost';
  onPick: () => void;
}

/** A decision with a short explanation and two or three plain choices (Cancel is always there). */
export function ConfirmDialog({ title, icon, children, choices, onCancel, cancelLabel = 'Cancel' }: { title: string; icon?: ReactNode; children: ReactNode; choices: ConfirmChoice[]; onCancel: () => void; cancelLabel?: string }) {
  return (
    <Dialog title={title} icon={icon} onClose={onCancel} width={480}>
      <div className="cap-dialog">
        <div className="cap-dialog-text">{children}</div>
        <div className="cap-dialog-actions">
          <Button variant="ghost" onClick={onCancel}>{cancelLabel}</Button>
          {choices.map(c => (
            <Button key={c.label} variant={c.variant ?? 'secondary'} onClick={c.onPick}>
              {c.label}
            </Button>
          ))}
        </div>
      </div>
    </Dialog>
  );
}

/** The recording dot used on Start / Resume buttons. */
export function RecDot() {
  return <span className="cap-rec" aria-hidden />;
}
