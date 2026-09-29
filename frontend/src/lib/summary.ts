import { create } from 'zustand';
import { api, ApiError } from './api';
import type { Asset } from './types';

/** Part understanding from GET /api/assets/{id}/summary (docs/v3-plan.md Contract 3). */
export interface PartSummary {
  asset_id: string;
  kind: string;
  units: string;
  count: number;
  spacing: number;
  aabb: { min: number[]; max: number[]; size: number[] };
  part_frame: { origin: number[]; length: number[]; width: number[]; height: number[] };
  dimensions: { length: number; width: number; height: number };
  dimensions_raw?: { length: number; width: number; height: number };
  profile?: { axis: string; slices: { from: number; to: number; width: number; height: number; count: number }[] };
  features?: {
    planes?: { point: number[]; normal: number[]; area_mm2: number; flatness: number; label?: string }[];
    cylinders?: { point: number[]; axis: number[]; radius: number; length: number; rms: number; coverage_deg?: number }[];
  };
  description?: string;
  computed_at?: string;
}

interface SummaryState {
  byId: Record<string, PartSummary | 'loading' | 'missing'>;
}

export const useSummaries = create<SummaryState>(() => ({ byId: {} }));

let endpoint = true;

export async function loadSummary(asset: Asset): Promise<PartSummary | null> {
  const key = `${asset.id}:${asset.created}`;
  const cur = useSummaries.getState().byId[key];
  if (cur && cur !== 'loading') return cur === 'missing' ? null : cur;
  if (cur === 'loading' || !endpoint || asset.kind === 'image') return null;
  useSummaries.setState(s => ({ byId: { ...s.byId, [key]: 'loading' } }));
  try {
    const summary = await api.get<PartSummary>(`/api/assets/${asset.id}/summary`);
    useSummaries.setState(s => ({ byId: { ...s.byId, [key]: summary } }));
    return summary;
  } catch (err) {
    if (err instanceof ApiError && (err.status === 404 || err.status === 405) && /not found/i.test(err.message) && !/asset/i.test(err.message)) endpoint = false;
    useSummaries.setState(s => ({ byId: { ...s.byId, [key]: 'missing' } }));
    return null;
  }
}

export function useSummary(asset: Asset | undefined): PartSummary | null {
  const key = asset ? `${asset.id}:${asset.created}` : '';
  const v = useSummaries(s => (key ? s.byId[key] : undefined));
  if (asset && v === undefined) queueMicrotask(() => loadSummary(asset));
  return v && v !== 'loading' && v !== 'missing' ? v : null;
}

/** Length × width × height of a model: robust part dimensions when known, else the smallest box around its points. */
export function partDims(asset: Asset, summary: PartSummary | null): { dims: [number, number, number]; robust: boolean } {
  if (summary?.dimensions) {
    const d = summary.dimensions;
    return { dims: [d.length, d.width, d.height], robust: true };
  }
  const p = asset.part?.dimensions;
  if (p) return { dims: [p.length, p.width, p.height], robust: true };
  const o = asset.stats.oriented_dimensions ?? asset.stats.dimensions;
  const s = [...o].sort((a, b) => b - a) as [number, number, number];
  return { dims: s, robust: false };
}
