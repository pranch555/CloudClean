import { useState } from 'react';
import { TriangleAlert } from 'lucide-react';
import type { Asset } from '../../lib/types';
import { useStore } from '../../store';
import { useReport } from '../useReport';

/** The nearest merge this model was made from (itself included), walking back through its parents. */
function useMergeAncestor(asset: Asset | undefined): Asset | undefined {
  const byId = useStore(s => s.byId);
  const seen = new Set<string>();
  const queue = asset ? [asset] : [];
  while (queue.length) {
    const a = queue.shift()!;
    if (seen.has(a.id)) continue;
    seen.add(a.id);
    if (a.operation === 'merge') return a;
    for (const p of a.parents) {
      const parent = byId.get(p);
      if (parent) queue.push(parent);
    }
  }
  return undefined;
}

/**
 * Measurements on a merge are only as good as the agreement of its scans. When the pre-merge check flagged an
 * ambiguous pose or a doubled skin, say so next to every measurement taken on the merge or anything made from it.
 */
export function MergeCaveat({ asset }: { asset: Asset | undefined }) {
  const merge = useMergeAncestor(asset);
  const report = useReport(merge);
  const [open, setOpen] = useState(false);
  const caveat = report?.caveat as { headline?: string; sentence?: string } | undefined;
  if (!merge || !caveat?.sentence) return null;
  return (
    <div className="drift-line is-warn" role="note">
      <TriangleAlert size={15} aria-hidden />
      <span>
        {caveat.headline ?? 'The scans in this merge did not fully agree: check key sizes with calipers.'}{' '}
        <button type="button" className="link small" onClick={() => setOpen(o => !o)} aria-expanded={open}>{open ? 'Hide' : 'Why'}</button>
        {open && (
          <span className="drift-details">
            {caveat.sentence}
            {merge.id !== asset?.id ? ` (from ${merge.name})` : ''}
          </span>
        )}
      </span>
    </div>
  );
}
