import { useEffect } from 'react';
import { setTab as setMeasureTab } from '../steps/measure/state';
import { api } from './api';
import type { Job } from './types';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';

const FINAL = new Set(['done', 'failed', 'cancelled']);
const known = new Map<string, string>();
const finishedHandlers = new Map<string, (job: Job) => void>();
let timer: number | undefined;

export const isFinal = (j: Job) => FINAL.has(j.status);

export function jobStarted(job: Job, onFinished?: (job: Job) => void) {
  known.set(job.id, job.status);
  if (onFinished) finishedHandlers.set(job.id, onFinished);
  useStore.setState(s => ({ jobs: [job, ...s.jobs.filter(j => j.id !== job.id)] }));
  schedule(250);
}

/** POST an operation that returns a job; shows errors as toasts. */
export async function submitJob(url: string, body: unknown, onFinished?: (job: Job) => void): Promise<Job | null> {
  try {
    const job = await api.post<Job>(url, body);
    jobStarted(job, onFinished);
    useStore.getState().toast({ kind: 'info', title: `Started: ${job.title}`, ms: 2600 });
    return job;
  } catch (err) {
    useStore.getState().toast({ kind: 'error', title: 'Could not start', body: (err as Error).message });
    return null;
  }
}

function schedule(ms: number) {
  window.clearTimeout(timer);
  timer = window.setTimeout(poll, ms);
}

async function poll() {
  let active = false;
  try {
    const jobs = await api.jobs();
    const finished: Job[] = [];
    for (const j of jobs) {
      const prev = known.get(j.id);
      if (prev && prev !== j.status && FINAL.has(j.status)) finished.push(j);
      known.set(j.id, j.status);
    }
    useStore.setState({ jobs });
    active = jobs.some(j => !FINAL.has(j.status));
    for (const j of finished.reverse()) await onFinished(j);
  } catch {
    /* server busy or restarting */
  }
  schedule(active ? 700 : 2500);
}

async function onFinished(job: Job) {
  const st = useStore.getState();
  const custom = finishedHandlers.get(job.id);
  finishedHandlers.delete(job.id);
  if (job.status === 'done') {
    await st.refreshAssets();
    st.refreshProjects().catch(() => undefined);
    const s = useStore.getState();
    const results = (job.result || []).map(id => s.byId.get(id)).filter(Boolean);
    if (results.length && !s.pairing) {
      const main = job.kind === 'merge' || job.kind === 'compare' || job.kind === 'golden_check' ? results[0]! : results[results.length - 1]!;
      if (job.kind === 'import') {
        useStore.setState({ visible: [...new Set([...s.visible, main.id])], activeId: main.id });
      } else if (job.kind !== 'autopilot') {
        useStore.setState({ visible: [main.id], selected: [], activeId: main.id });
      }
      if (job.kind === 'golden_check') {
        // the golden model coloured by what the check found; Measure -> Golden model shows the rest
        st.setDisplay({ colorMode: 'original', scalar: null });
        useStore.getState().set({ step: 'measure' });
        setMeasureTab('cad');
      }
      if (job.kind === 'compare') {
        const dev = main.scalars?.find(x => x.name === 'deviation');
        if (dev) {
          const tol = Number((job.payload.params as Record<string, unknown> | undefined)?.tolerance ?? 0.1);
          const range = Math.max(Math.abs(dev.min ?? tol * 4), Math.abs(dev.max ?? tol * 4), tol * 2);
          st.setDisplay({ colorMode: 'scalar', scalar: { name: 'deviation', style: { kind: 'diverging', min: -Math.min(range, tol * 5), max: Math.min(range, tol * 5), tolerance: tol, steps: 0 } } });
          useStore.getState().set({ step: 'measure' });
        }
      }
      window.setTimeout(() => getViewer()?.fit(undefined, undefined, true), 60);
    }
    const warned = job.logs.some(l => l.includes('WARNING'));
    st.toast({ kind: warned ? 'warn' : 'ok', title: `${job.title} finished`, body: warned ? 'Finished with warnings — see the log.' : undefined, action: warned ? { label: 'Show log', run: () => useStore.setState({ jobsOpen: true, expandedJob: job.id }) } : undefined });
  } else if (job.status === 'failed') {
    st.toast({ kind: 'error', title: `${job.title} failed`, body: job.error || undefined, action: { label: 'Show log', run: () => useStore.setState({ jobsOpen: true, expandedJob: job.id }) } });
  }
  custom?.(job);
}

export async function cancelJob(id: string) {
  try {
    await api.post(`/api/jobs/${id}/cancel`);
    schedule(150);
  } catch (err) {
    useStore.getState().toast({ kind: 'error', title: 'Cancel failed', body: (err as Error).message });
  }
}

export function useJobPolling() {
  useEffect(() => {
    api.jobs().then(jobs => {
      jobs.forEach(j => known.set(j.id, j.status));
      useStore.setState({ jobs });
      schedule(jobs.some(j => !FINAL.has(j.status)) ? 500 : 2500);
    }).catch(() => schedule(2500));
    return () => window.clearTimeout(timer);
  }, []);
}

export const lastLog = (j: Job) => (j.logs.length ? j.logs[j.logs.length - 1].replace(/^\[\s*[\d.]+s\]\s*/, '') : 'Waiting…');
