import { useEffect } from 'react';
import { create } from 'zustand';
import { api } from '../../lib/api';
import { goldenGrade, type GoldenGrade } from '../../lib/golden';

/** What the model list shows of a golden check, read once from its report (GET /api/assets/{id}) and kept. */
export interface CheckInfo {
  grade: GoldenGrade;
  /** the deviation map (compared scan) this check made */
  compareId: string | null;
  tolerance: number | null;
  headline: string | null;
}

export const useCheckInfo = create<{ info: Record<string, CheckInfo | 'none'> }>(() => ({ info: {} }));

const inflight = new Set<string>();

function load(id: string) {
  if (inflight.has(id) || useCheckInfo.getState().info[id]) return;
  inflight.add(id);
  api
    .asset(id)
    .then(a => {
      const r = a.report;
      const info: CheckInfo | 'none' = r
        ? {
            grade: goldenGrade(r),
            compareId: typeof r.compare_asset?.id === 'string' ? r.compare_asset.id : null,
            tolerance: typeof r.tolerance === 'number' ? r.tolerance : typeof a.params?.tolerance === 'number' ? (a.params.tolerance as number) : null,
            headline: typeof r.headline === 'string' ? r.headline : null,
          }
        : 'none';
      useCheckInfo.setState(s => ({ info: { ...s.info, [id]: info } }));
    })
    .catch(() => undefined) // shown without a verdict; tried again next time the list mounts
    .finally(() => inflight.delete(id));
}

/** Load (lazily, once) the reports of these golden checks; returns what is known so far. */
export function useCheckInfos(ids: string[]): Record<string, CheckInfo | 'none'> {
  const key = ids.join();
  useEffect(() => {
    ids.forEach(load);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return useCheckInfo(s => s.info);
}
