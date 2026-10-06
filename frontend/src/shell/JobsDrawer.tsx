import { useEffect, useLayoutEffect, useRef } from 'react';
import { ChevronDown, ChevronRight, X } from 'lucide-react';
import { cancelJob, lastLog } from '../lib/jobs';
import { fmtDuration } from '../lib/format';
import { useStore } from '../store';
import { Badge, Button, IconButton, Progress } from '../ui/primitives';
import { TickLoader } from '../ui/motion/TickLoader';
import { enter } from '../lib/motion';

const TONE = { queued: 'neutral', running: 'signal', done: 'good', failed: 'critical', cancelled: 'warning' } as const;
const LABEL = { queued: 'waiting', running: 'running', done: 'done', failed: 'failed', cancelled: 'cancelled' } as const;

/** Every operation runs as a job in its own process: progress, full log and cancel. */
export function JobsDrawer() {
  const open = useStore(s => s.jobsOpen);
  const jobs = useStore(s => s.jobs);
  const expanded = useStore(s => s.expandedJob);
  const pre = useRef<HTMLPreElement>(null);
  const openJob = jobs.find(j => j.id === expanded);

  useEffect(() => {
    if (pre.current) pre.current.scrollTop = pre.current.scrollHeight;
  }, [openJob?.logs.length]);

  // rows arrive in a short stagger when the drawer opens; later only a new job slides in
  const body = useRef<HTMLDivElement>(null);
  const seen = useRef<Set<string> | null>(null);
  const ids = jobs.map(j => j.id).join(',');
  useLayoutEffect(() => {
    const el = body.current;
    if (!open || !el) {
      seen.current = null;
      return;
    }
    const rows = [...el.querySelectorAll<HTMLElement>('.job')];
    if (!seen.current) enter(rows.slice(0, 10), { y: 8, step: 40, delay: 60, duration: 420 });
    else enter(rows.filter(r => !seen.current!.has(r.dataset.job ?? '')), { y: -10, duration: 460 });
    seen.current = new Set(jobs.map(j => j.id));
  }, [open, ids]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!open) return null;
  return (
    <section className="jobs-drawer" aria-label="Jobs">
      <header className="jobs-head">
        <span className="panel-title">Jobs</span>
        <span className="caption">Each operation runs on its own and can be cancelled.</span>
        <span className="spacer" />
        <IconButton size="sm" label="Close" onClick={() => useStore.setState({ jobsOpen: false })}>
          <X size={16} />
        </IconButton>
      </header>
      <div className="jobs-body" ref={body}>
        {jobs.length === 0 && <div className="caption" style={{ padding: 16 }}>Operations you start appear here with their full logs.</div>}
        {jobs.map(j => {
          const isOpen = j.id === expanded;
          const duration = j.started ? (j.finished || Date.now() / 1000) - j.started : 0;
          return (
            <div key={j.id} className="job" data-job={j.id}>
              <button type="button" className="job-head" onClick={() => useStore.setState({ expandedJob: isOpen ? null : j.id })} aria-expanded={isOpen}>
                {isOpen ? <ChevronDown size={15} aria-hidden /> : <ChevronRight size={15} aria-hidden />}
                <TickLoader state={j.status} fraction={j.status === 'running' ? j.progress?.fraction ?? null : null} />
                <Badge tone={TONE[j.status]}>{LABEL[j.status]}</Badge>
                <span className="job-title">
                  {j.title}
                  <span className="job-sub mono">{j.error || (j.status === 'running' ? j.progress?.label || lastLog(j) : '')}</span>
                </span>
                <span className="mono job-time">{duration ? fmtDuration(duration) : ''}</span>
                {j.status === 'running' || j.status === 'queued' ? (
                  <Button size="sm" variant="ghost" onClick={e => { e.stopPropagation(); cancelJob(j.id); }}>Cancel</Button>
                ) : <span />}
              </button>
              {j.status === 'running' && <Progress value={j.progress?.fraction} indeterminate={!j.progress} />}
              {isOpen && <pre ref={pre} className="job-log mono">{j.logs.join('\n') || 'Waiting…'}</pre>}
            </div>
          );
        })}
      </div>
    </section>
  );
}
