import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, Clock, FolderOpen, Loader2, PlayCircle, Save, UploadCloud, XCircle } from 'lucide-react';
import { api, upload } from '../../lib/api';
import type { Job } from '../../lib/types';
import { jobStarted, submitJob } from '../../lib/jobs';
import { fmtAgo } from '../../lib/format';
import { useStore } from '../../store';
import { Badge, Button, Field, NumberInput, Progress, Section, Segmented, Select, Switch, TextInput } from '../../ui/primitives';
import { AssessmentView, type Assessment } from '../../steps/Assessment';

interface Settings {
  watch_enabled: boolean;
  watch_folder: string;
  output_folder: string;
  recursive: boolean;
  settle_seconds: number;
  group_seconds: number;
  formats: string[];
  preset: string;
  remove_plane: boolean;
  merge_method: string;
  merge_mode: 'auto' | 'ask' | 'always' | 'never';
  mesh: { method: string; watertight: boolean; depth: number; smooth_iterations: number; target_triangles: number };
  reference_id: string;
  compare: Record<string, unknown>;
  auto_on_upload: boolean;
  auto_on_capture: boolean;
}

interface HistoryEntry {
  item: string;
  files: string[];
  job_id: string | null;
  status: string;
  outputs: string[];
  warnings: string[];
  error: string | null;
  started: string | number;
  finished: string | number | null;
  assessment?: Assessment;
}

interface Status {
  watching: boolean;
  enabled?: boolean;
  folder?: string;
  error?: string | null;
  pending_items: { item: string; files: string[]; settled: boolean }[];
  history: HistoryEntry[];
}

const STAGES = ['Import', 'Clean', 'Merge', 'Mesh', 'Inspect', 'Export'];
const FORMATS = ['stl', 'ply', 'obj', 'glb', 'off'];

export function AutopilotPanel() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [dirty, setDirty] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [dropping, setDropping] = useState(false);
  const [uploading, setUploading] = useState<number | null>(null);
  const [name, setName] = useState('');
  const jobs = useStore(s => s.jobs);
  const assets = useStore(s => s.assets);
  const selected = useStore(s => s.selected);
  const units = useStore(s => s.display.units);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.get<Settings>('/api/autopilot/settings').then(setSettings).catch(err => useStore.getState().toast({ kind: 'error', title: 'Autopilot unavailable', body: err.message }));
    const load = () => api.get<Status>('/api/autopilot/status').then(setStatus).catch(() => undefined);
    load();
    const t = window.setInterval(load, 2500);
    return () => window.clearInterval(t);
  }, []);

  const patch = (p: Partial<Settings>) => {
    setSettings(s => (s ? { ...s, ...p } : s));
    setDirty(true);
  };
  const patchMesh = (p: Partial<Settings['mesh']>) => settings && patch({ mesh: { ...settings.mesh, ...p } });

  const save = async () => {
    if (!settings) return;
    try {
      setSettings(await api.put<Settings>('/api/autopilot/settings', settings));
      setDirty(false);
      useStore.getState().toast({ kind: 'ok', title: settings.watch_enabled ? 'Autopilot is watching the folder' : 'Autopilot settings saved' });
    } catch (err) {
      useStore.getState().toast({ kind: 'error', title: 'Could not save', body: (err as Error).message });
    }
  };

  const send = async (files: File[]) => {
    if (!files.length) return;
    if (dirty) await save();
    setUploading(0);
    try {
      const res = await upload('/api/autopilot/upload', files, f => setUploading(f), name.trim() ? { name: name.trim() } : undefined);
      const job: Job | undefined = res.job ?? res.jobs?.[0] ?? (res.id ? res : undefined);
      if (job) jobStarted(job);
      useStore.getState().toast({ kind: 'info', title: `Autopilot started on ${files.length} file${files.length > 1 ? 's' : ''}` });
      setName('');
    } catch (err) {
      useStore.getState().toast({ kind: 'error', title: 'Autopilot upload failed', body: (err as Error).message });
    } finally {
      setUploading(null);
    }
  };

  const decide = async (jobId: string, decision: 'merge' | 'best' | 'separate' | 'cancel') => {
    try {
      const res = await api.post<{ job?: Job } & Partial<Job>>('/api/autopilot/decide', { job_id: jobId, decision });
      const job = (res.job ?? (res.id ? res : undefined)) as Job | undefined;
      if (job?.id) jobStarted(job);
      useStore.getState().toast({ kind: 'ok', title: decision === 'cancel' ? 'Cancelled' : 'Autopilot continues with your decision' });
      api.get<Status>('/api/autopilot/status').then(setStatus).catch(() => undefined);
    } catch (err) {
      useStore.getState().toast({ kind: 'error', title: 'Could not apply the decision', body: (err as Error).message });
    }
  };

  const running = jobs.find(j => j.kind === 'autopilot' && (j.status === 'running' || j.status === 'queued'));
  const stageIndex = running?.progress?.label ? STAGES.findIndex(s => running.progress!.label!.toLowerCase().includes(s.toLowerCase().replace('inspect', 'compar'))) : -1;
  const meshRefs = assets.filter(a => a.kind === 'mesh' && a.operation !== 'compare');
  const ticked = selected.filter(id => assets.find(a => a.id === id)?.kind !== 'image');

  return (
    <div className="page">
      <div className="settings-head">
        <h3>Automations</h3>
        <p>Drop scans, get a finished part. Every item is cleaned, checked before merging, meshed, optionally inspected against CAD, and exported — with no clicks. Warnings are never hidden.</p>
      </div>

      <div
        data-own-drop
        className={`dropzone ${dropping ? 'is-over' : ''}`}
        onDragOver={e => { e.preventDefault(); setDropping(true); }}
        onDragLeave={() => setDropping(false)}
        onDrop={e => { e.preventDefault(); setDropping(false); send([...e.dataTransfer.files]); }}
        onClick={() => fileInput.current?.click()}
        role="button"
        tabIndex={0}
      >
        <input ref={fileInput} type="file" multiple hidden onChange={e => { send([...(e.target.files ?? [])]); e.target.value = ''; }} />
        {uploading != null ? <Loader2 size={26} className="spin" aria-hidden /> : <UploadCloud size={26} aria-hidden />}
        <div className="dropzone-title">{uploading != null ? `Uploading ${Math.round(uploading * 100)}%` : 'Drop all scans of one part here'}</div>
        <div className="dropzone-sub">→ {settings?.formats.map(f => f.toUpperCase()).join(' + ') ?? 'STL'} in {settings?.output_folder || 'workspace/exports'}</div>
      </div>
      <Field label="Part name" help="Optional. Defaults to the file names.">
        <TextInput value={name} onChange={setName} placeholder="bracket_rev_b" />
      </Field>
      {ticked.length > 0 && (
        <Button icon={<PlayCircle size={15} />} onClick={() => submitJob('/api/autopilot/run', { asset_ids: ticked, name: name.trim() || undefined })}>
          Run autopilot on {ticked.length} ticked asset{ticked.length > 1 ? 's' : ''}
        </Button>
      )}

      {running && (
        <Section title="Running now" tone="accent">
          <div className="stepper" aria-label="Pipeline progress">
            {STAGES.map((s, i) => (
              <div key={s} className={`step-pill ${i < stageIndex ? 'is-done' : i === stageIndex ? 'is-current' : ''}`}>
                <span className="bar" />
                <span>{s}</span>
              </div>
            ))}
          </div>
          <Progress value={running.progress?.fraction} indeterminate={!running.progress} />
          <p className="hint-text mono">{running.title} · {running.progress?.label ?? running.status}</p>
        </Section>
      )}

      <Section title="Watched folder">
        {settings && (
          <>
            <Field label="Watch folder">
              <Switch checked={settings.watch_enabled} onChange={v => patch({ watch_enabled: v })} />
            </Field>
            <Field label="Folder on the server" inline={false} help="Point Revo Metro's export folder or a network share here. Each sub-folder, or files that arrive together, is one part.">
              <TextInput mono value={settings.watch_folder} placeholder="/mnt/scans/incoming" onChange={v => patch({ watch_folder: v })} />
            </Field>
            <Field label="Output folder" inline={false} help="Empty = workspace/exports. Results go into <folder>/<part>/.">
              <TextInput mono value={settings.output_folder} placeholder="/mnt/scans/finished" onChange={v => patch({ output_folder: v })} />
            </Field>
            <div className="watch-status">
              {status?.watching ? <Badge tone="good">watching</Badge> : settings.watch_enabled ? <Badge tone="warning">not running</Badge> : <Badge>off</Badge>}
              {status?.error && <span className="warn-text">{status.error}</span>}
              {!!status?.pending_items.length && <span className="hint-inline">{status.pending_items.length} part(s) waiting for files to settle</span>}
            </div>
          </>
        )}
      </Section>

      {settings && (
        <Section title="Recipe" collapsible defaultOpen>
          <Field label="Export formats" inline={false}>
            <div className="chip-toggle-row">
              {FORMATS.map(f => (
                <button key={f} type="button" className={`chip-toggle ${settings.formats.includes(f) ? 'is-on' : ''}`} aria-pressed={settings.formats.includes(f)} onClick={() => patch({ formats: settings.formats.includes(f) ? settings.formats.filter(x => x !== f) : [...settings.formats, f] })}>
                  {f.toUpperCase()}
                </button>
              ))}
            </div>
          </Field>
          <Field label="Cleaning">
            <Segmented size="sm" value={settings.preset} onChange={preset => patch({ preset })} options={['light', 'standard', 'aggressive'].map(p => ({ value: p, label: p[0].toUpperCase() + p.slice(1) }))} />
          </Field>
          <Field label="Remove table / turntable">
            <Switch checked={settings.remove_plane} onChange={remove_plane => patch({ remove_plane })} />
          </Field>
          <Field label="Several scans of one part" inline={false} help="Check first merges only when the scans add surface and line up exactly; if they would stack into a doubled skin it uses the best scan or waits for you.">
            <Select
              value={settings.merge_mode ?? 'auto'}
              onChange={merge_mode => patch({ merge_mode })}
              options={[
                { value: 'auto', label: 'Check first, merge only when it helps (recommended)' },
                { value: 'ask', label: 'Always ask me before merging' },
                { value: 'never', label: 'Never merge: mesh each scan separately' },
                { value: 'always', label: 'Always merge' },
              ]}
            />
          </Field>
          <Field label="Merge alignment">
            <Select value={settings.merge_method} onChange={merge_method => patch({ merge_method })} options={['auto', 'markers', 'icp', 'none'].map(v => ({ value: v, label: v }))} />
          </Field>
          <Field label="Mesh method">
            <Select value={settings.mesh.method} onChange={method => patchMesh({ method })} options={[{ value: 'poisson', label: 'poisson' }, { value: 'bpa', label: 'bpa' }]} />
          </Field>
          <Field label="Watertight" help="Closes unscanned areas. Off keeps only surface backed by scan data.">
            <Switch checked={settings.mesh.watertight} onChange={watertight => patchMesh({ watertight })} />
          </Field>
          <Field label="Detail depth (0 = auto)"><NumberInput value={settings.mesh.depth} step={1} min={0} onChange={depth => patchMesh({ depth })} /></Field>
          <Field label="Smoothing passes"><NumberInput value={settings.mesh.smooth_iterations} step={1} min={0} onChange={smooth_iterations => patchMesh({ smooth_iterations })} /></Field>
          <Field label="Max triangles (0 = all)"><NumberInput value={settings.mesh.target_triangles} step={10000} min={0} onChange={target_triangles => patchMesh({ target_triangles })} /></Field>
          <Field label="Inspect against CAD" inline={false} help="Adds a deviation report to every part.">
            <Select value={settings.reference_id} onChange={reference_id => patch({ reference_id })} options={[{ value: '', label: 'Off' }, ...meshRefs.map(a => ({ value: a.id, label: a.name }))]} />
          </Field>
          {settings.reference_id && (
            <Field label="Tolerance ±">
              <NumberInput value={Number(settings.compare.tolerance ?? 0.1)} step={0.01} min={0.001} unit={units} onChange={tolerance => patch({ compare: { ...settings.compare, tolerance } })} />
            </Field>
          )}
          <Field label="Run on every upload" help="Files dropped anywhere in the app go straight through autopilot.">
            <Switch checked={settings.auto_on_upload} onChange={auto_on_upload => patch({ auto_on_upload })} />
          </Field>
          <Field label="Run when a capture is saved">
            <Switch checked={settings.auto_on_capture} onChange={auto_on_capture => patch({ auto_on_capture })} />
          </Field>
          <Section title="Timing" collapsible defaultOpen={false}>
            <Field label="File settle time"><NumberInput value={settings.settle_seconds} step={1} min={0} unit="s" onChange={settle_seconds => patch({ settle_seconds })} /></Field>
            <Field label="Group files within"><NumberInput value={settings.group_seconds} step={5} min={0} unit="s" onChange={group_seconds => patch({ group_seconds })} /></Field>
            <Field label="Include sub-folders"><Switch checked={settings.recursive} onChange={recursive => patch({ recursive })} /></Field>
          </Section>
        </Section>
      )}

      <div className={`sticky-save ${dirty ? 'is-visible' : ''}`}>
        <Button variant="primary" block icon={<Save size={15} />} disabled={!dirty} onClick={save}>
          Save autopilot settings
        </Button>
      </div>

      <Section title="History">
        {!status?.history.length && <p className="hint-text">Finished parts appear here with their export paths and warnings.</p>}
        <ul className="history">
          {status?.history.map((h, i) => (
            <li key={`${h.job_id}-${i}`} className="history-item">
              <span className={`history-icon status-${h.status}`}>
                {h.status === 'done' ? (h.warnings.length ? <AlertTriangle size={15} /> : <CheckCircle2 size={15} />) : h.status === 'failed' ? <XCircle size={15} /> : <Clock size={15} />}
              </span>
              <div className="history-text">
                <div className="history-title">{h.item}</div>
                <div className="hint-inline">{h.files.length} file{h.files.length === 1 ? '' : 's'} · {h.status}{h.started ? ` · ${fmtAgo(typeof h.started === 'number' ? new Date(h.started * 1000).toISOString() : h.started)}` : ''}</div>
                {h.status === 'needs_decision' && h.assessment && h.job_id && (
                  <AssessmentView
                    a={h.assessment}
                    actions={
                      <>
                        <Button size="sm" variant={h.assessment.recommendation === 'merge' ? 'primary' : 'secondary'} onClick={() => decide(h.job_id!, 'merge')}>Merge</Button>
                        {h.assessment.best_index != null && <Button size="sm" variant={h.assessment.recommendation === 'use_best' ? 'primary' : 'secondary'} onClick={() => decide(h.job_id!, 'best')}>Best scan only</Button>}
                        <Button size="sm" onClick={() => decide(h.job_id!, 'separate')}>Keep separate</Button>
                        <Button size="sm" variant="ghost" onClick={() => decide(h.job_id!, 'cancel')}>Cancel</Button>
                      </>
                    }
                  />
                )}
                {h.outputs.map(o => <div key={o} className="mono path"><FolderOpen size={12} aria-hidden /> {o}</div>)}
                {h.warnings.map((w, j) => <div key={j} className="warn-text">{w}</div>)}
                {h.error && <div className="err-text">{h.error}</div>}
              </div>
            </li>
          ))}
        </ul>
      </Section>
    </div>
  );
}
