import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import { isMissingRoute } from '../lib/measureTools';
import type { Asset } from '../lib/types';

/* eslint-disable @typescript-eslint/no-explicit-any */
const reports = new Map<string, Record<string, any> | null>();

/** The report of the operation that made an asset (cached per asset). */
export function useReport(asset: Asset | undefined): Record<string, any> | null {
  const [report, setReport] = useState<Record<string, any> | null>(asset ? reports.get(asset.id) ?? null : null);
  useEffect(() => {
    if (!asset) return setReport(null);
    if (reports.has(asset.id)) return setReport(reports.get(asset.id)!);
    let alive = true;
    api.asset(asset.id).then(r => {
      reports.set(asset.id, r.report ?? null);
      if (alive) setReport(r.report ?? null);
    }).catch(() => alive && setReport(null));
    return () => {
      alive = false;
    };
  }, [asset?.id]);
  return report;
}

export interface Drift {
  asset_id: string;
  parent_id: string | null;
  operation: string;
  displacement?: { signed_mean: number; mean: number; rms: number; p95: number; max: number } | null;
  dimensions_before?: Record<string, number>;
  dimensions_after?: Record<string, number>;
  dimension_change?: Record<string, number>;
  verdict: 'unchanged' | 'changed' | 'moved' | string;
  sentence: string;
}

const drifts = new Map<string, Drift | null>();
let driftEndpoint = true;

/** How far the operation that made this asset moved or resized the surface (docs/v3-plan.md Contract 5). */
export function useDrift(asset: Asset | undefined): Drift | null | 'loading' {
  const [d, setD] = useState<Drift | null | 'loading'>(asset && drifts.has(asset.id) ? drifts.get(asset.id)! : null);
  useEffect(() => {
    if (!asset || !asset.parents.length || asset.operation === 'import' || asset.operation === 'capture' || !driftEndpoint) return setD(null);
    if (drifts.has(asset.id)) return setD(drifts.get(asset.id)!);
    let alive = true;
    setD('loading');
    api.post<Drift>('/api/accuracy/drift', { asset_id: asset.id }).then(r => {
      drifts.set(asset.id, r);
      if (alive) setD(r);
    }).catch(err => {
      if (isMissingRoute(err)) driftEndpoint = false;
      drifts.set(asset.id, null);
      if (alive) setD(null);
    });
    return () => {
      alive = false;
    };
  }, [asset?.id]);
  return d;
}
