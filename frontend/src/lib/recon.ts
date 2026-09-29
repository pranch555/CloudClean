import { useEffect, useState } from 'react';
import { api } from './api';

export interface ReconStatus {
  available: boolean;
  reason: string;
}

let cached: { at: number; value: ReconStatus } | null = null;

/** Whether the server can reconstruct from photos (the GPU container on the DGX Spark), with the reason when not. */
export function useReconStatus(): ReconStatus | null {
  const [status, setStatus] = useState<ReconStatus | null>(cached && Date.now() - cached.at < 60_000 ? cached.value : null);
  useEffect(() => {
    if (cached && Date.now() - cached.at < 60_000) return;
    let live = true;
    api
      .get<ReconStatus>('/api/photos-to-3d/status')
      .catch(() => ({ available: false, reason: 'The server did not say whether it can work with photos.' }))
      .then(value => {
        cached = { at: Date.now(), value };
        if (live) setStatus(value);
      });
    return () => {
      live = false;
    };
  }, []);
  return status;
}
