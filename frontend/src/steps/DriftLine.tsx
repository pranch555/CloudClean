import { useState } from 'react';
import { CheckCircle2, Info, Loader2, Move3d, TriangleAlert } from 'lucide-react';
import { fmtLen } from '../lib/format';
import type { Asset } from '../lib/types';
import { useStore } from '../store';
import { useDrift } from './useReport';

const AXES = ['length', 'width', 'height'] as const;

/**
 * Accuracy promise, checked: did the operation that made this model move or resize the surface? One plain line with
 * the numbers that matter; the accuracy module's full explanation is one click away.
 */
export function DriftLine({ asset }: { asset: Asset }) {
  const d = useDrift(asset);
  const units = useStore(s => s.display.units);
  const [open, setOpen] = useState(false);
  if (d === 'loading') {
    return (
      <div className="drift-line">
        <Loader2 size={15} className="spin" aria-hidden /> Checking whether the size changed…
      </div>
    );
  }
  if (!d) return null;
  const change = d.dimension_change ?? {};
  const worst = AXES.map(a => [a, change[a]] as const).filter(([, v]) => typeof v === 'number').sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]))[0];
  const p95 = d.displacement?.p95;
  let tone: 'ok' | 'info' | 'warn' = 'ok';
  let text: string;
  if (d.verdict === 'unchanged') {
    text = `Size unchanged${p95 != null && p95 > 0 ? ` — the surface stayed within ${fmtLen(p95, 3)} ${units}` : ' — no point was moved'}.`;
  } else if (d.verdict === 'moved') {
    tone = 'info';
    text = 'Moved as a solid part — its shape and size are unchanged.';
  } else {
    const w = worst ? Math.abs(worst[1]) : 0;
    tone = w > 0.05 ? 'warn' : 'info';
    text = worst
      ? `Size changed by up to ${fmtLen(w, 3)} ${units} (${worst[0]} ${worst[1] > 0 ? 'grew' : 'shrank'})${p95 != null ? `; 95 % of the surface within ${fmtLen(p95, 3)} ${units} of the original` : ''}.`
      : 'The surface changed — see details.';
  }
  const Icon = tone === 'ok' ? CheckCircle2 : tone === 'info' ? (d.verdict === 'moved' ? Move3d : Info) : TriangleAlert;
  return (
    <div className={`drift-line is-${tone}`}>
      <Icon size={15} aria-hidden />
      <span>
        {text}{' '}
        <button type="button" className="link small" onClick={() => setOpen(o => !o)} aria-expanded={open}>{open ? 'Hide details' : 'Details'}</button>
        {open && <span className="drift-details">{d.sentence}</span>}
      </span>
    </div>
  );
}
