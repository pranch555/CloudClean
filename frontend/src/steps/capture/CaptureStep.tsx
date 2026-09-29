import { useEffect, useState } from 'react';
import { ArrowRight, Cable, CheckCircle2, ChevronDown, Copy, Eye, Info, LocateFixed, Pause, RefreshCw, RotateCw, Save, Square, Trash2, TriangleAlert, Upload, Wifi, XCircle } from 'lucide-react';
import * as THREE from 'three';
import { fmtCount } from '../../lib/format';
import { api } from '../../lib/api';
import { useAuth } from '../../lib/auth';
import { pickFiles } from '../../lib/importing';
import { programRunning, turntable, useTurntable } from '../../lib/turntable';
import { useStore } from '../../store';
import { getViewer } from '../../viewer/instance';
import { scalarColor } from '../../viewer/colormaps';
import { CleanGlyph } from '../../ui/icons';
import { Button, Field, Metric, Segmented, Switch, TextInput } from '../../ui/primitives';
import { AssessmentView } from '../Assessment';
import { Block, NextStepButton, ResultCard, StepFrame } from '../StepFrame';
import {
  applyLiveStyle,
  captureAction,
  connectCapture,
  decidePending,
  densityStyle,
  discardCapture,
  loadAutoOnCapture,
  loadDrivers,
  loadPending,
  openStream,
  saveCapture,
  useCapture,
  type Hole,
} from '../../features/capture/captureStore';
import { driverMeta, fmtArea, fmtClock, HOLE_WORD, STATE_WORD } from '../../features/capture/vocabulary';
import { ScannerChooser, ScannerSummary, useScannerSetup, type ScannerSetup } from './ScannerBlock';
import { TurntableBlock } from './TurntableBlock';
import { PhotosBlock } from './PhotosBlock';
import { ActionError, ConfirmDialog, RecDot, runAction, useCaptureAction } from './parts';

const LIVE = new Set(['running', 'paused']);

/** Step 1 — Scan: connect a scanner (and turntable), scan with live guidance, save; or bring in files. */
export function CaptureStep() {
  const setup = useScannerSetup();
  const status = useCapture(s => s.status);
  const hasPending = useCapture(s => s.pending.length > 0);
  const [discarding, setDiscarding] = useState(false);
  const state = status?.state ?? 'idle';
  const session = !!status?.active && state !== 'closed';
  const points = status?.points ?? 0;
  // once scanning stops, keeping the scan is the next thing to do
  const saveFirst = points > 0 && state !== 'running' && (!!status?.unsaved || !!status?.saved_asset_id);

  // fresh scanner availability and waiting scans each time the step opens (the viewport HUD keeps the live stream,
  // the Z-up view and the turntable status going while the Scan step is shown)
  useEffect(() => {
    openStream();
    loadPending();
    loadDrivers();
  }, []);

  const disconnect = () => {
    if (status?.unsaved) setDiscarding(true);
    else runAction('discard', discardCapture);
  };

  return (
    <StepFrame step="capture" footer={<CaptureFooter setup={setup} />}>
      {!session ? (
        <>
          <ScannerChooser setup={setup} />
          {hasPending && <PendingBlock />}
          <div className="divider" />
          <TurntableBlock />
          <div className="divider" />
          <FilesBlock />
          <div className="divider" />
          <PhotosBlock />
        </>
      ) : (
        <>
          <SessionCard />
          {saveFirst && <SaveBlock onDiscard={() => setDiscarding(true)} />}
          {status?.device?.phase !== undefined && <MarkerMapBlock />}
          {hasPending && <PendingBlock />}
          <GuidanceBlock />
          {points > 0 && !saveFirst && <SaveBlock onDiscard={() => setDiscarding(true)} />}
          <LiveViewBlock />
          <div className="divider" />
          <TurntableBlock />
          <div className="divider" />
          <ScannerSummary setup={setup} onDisconnect={disconnect} />
          <FilesBlock compact />
        </>
      )}
      {discarding && (
        <ConfirmDialog
          title="Discard this scan?"
          icon={<Trash2 size={20} className="danger-text" />}
          onCancel={() => setDiscarding(false)}
          cancelLabel="Keep it"
          choices={[{ label: `Discard ${fmtCount(points)} points`, variant: 'danger', onPick: () => { setDiscarding(false); runAction('discard', discardCapture); } }]}
        >
          <p>The points captured in this session have not been saved and will be gone. Scans you already saved stay in the model list.</p>
        </ConfirmDialog>
      )}
    </StepFrame>
  );
}

/* ------------------------------------------------------------------ footer: the one live control */

function CaptureFooter({ setup }: { setup: ScannerSetup }) {
  const status = useCapture(s => s.status);
  const busy = useCaptureAction(s => s.busy);
  const ttSync = useTurntable(s => programRunning(s.status) && !!s.status?.program?.sync_scan);
  const ttBusy = useTurntable(s => s.busy);
  const driversError = useCapture(s => s.driversError);
  const state = status?.state ?? 'idle';
  const session = !!status?.active && state !== 'closed';
  const points = status?.points ?? 0;
  const labels = ['connect', 'start', 'pause', 'resume', 'stop', 'reconnect-error'];
  const kind = setup.drivers.find(d => d.id === status?.driver)?.kind ?? setup.driver?.kind;
  const startWord = kind === 'bridge' ? 'Start receiving scans' : kind === 'folder' ? 'Start watching' : 'Start scanning';
  const act = (label: string, action: string) => runAction(label, () => captureAction(action));

  let main;
  if (session && ttSync && LIVE.has(state)) {
    // the turntable scan starts and pauses the scanner at every stop: one control, not two that fight
    main = (
      <div className="stack tight">
        <p className="caption cap-foot-note">
          <RotateCw size={13} aria-hidden /> The turntable scan is starting and pausing the scanner.
        </p>
        <Button variant="danger" size="lg" block icon={<Square size={15} />} loading={ttBusy === 'program-stop'} onClick={() => turntable.stopProgram()}>
          Stop the turntable scan
        </Button>
      </div>
    );
  } else if (!session) {
    const meta = driverMeta(setup.driver);
    const ready = !!setup.driver?.available;
    main = (
      <Button variant="primary" size="lg" block icon={<Cable size={18} />} loading={busy === 'connect'} disabled={!ready} onClick={() => runAction('connect', () => connectCapture(setup.driverId, setup.values))}>
        {ready ? meta.connect : setup.driver ? `${meta.title} is not ready` : driversError ? 'No scanners to connect' : 'Looking for scanners…'}
      </Button>
    );
  } else if (state === 'running') {
    main = (
      <div className="cap-transport">
        <Button variant="primary" size="lg" icon={<Pause size={18} />} loading={busy === 'pause'} onClick={() => act('pause', 'pause')}>Pause</Button>
        <Button size="lg" icon={<Square size={15} />} loading={busy === 'stop'} onClick={() => act('stop', 'stop')}>Stop</Button>
      </div>
    );
  } else if (state === 'paused') {
    main = (
      <div className="cap-transport">
        <Button variant="signal" size="lg" icon={<RecDot />} loading={busy === 'resume'} onClick={() => act('resume', 'resume')}>Resume</Button>
        <Button size="lg" icon={<Square size={15} />} loading={busy === 'stop'} onClick={() => act('stop', 'stop')}>Stop</Button>
      </div>
    );
  } else if (state === 'error') {
    main = (
      <Button variant="primary" size="lg" block icon={<RefreshCw size={17} />} loading={busy === 'reconnect-error'} disabled={!!status?.unsaved} onClick={() => runAction('reconnect-error', () => connectCapture(status?.driver ?? setup.driverId, status?.settings ?? setup.values))}>
        {status?.unsaved ? 'Save or discard, then reconnect' : 'Reconnect the scanner'}
      </Button>
    );
  } else {
    main = (
      <Button variant={points > 0 ? 'secondary' : 'signal'} size="lg" block icon={<RecDot />} loading={busy === 'start'} onClick={() => act('start', 'start')}>
        {points > 0 ? 'Scan more' : startWord}
      </Button>
    );
  }
  return (
    <>
      <ActionError labels={labels} />
      {main}
      <NextStepButton from="capture" />
    </>
  );
}

/* ------------------------------------------------------------------ live session */

const DROPPED_WORD: Record<string, string> = { too_fast: 'moving too fast', tracking_lost: 'tracking lost', empty: 'no points' };

function SessionCard() {
  const status = useCapture(s => s.status)!;
  const streamOk = useCapture(s => s.connected);
  const drivers = useCapture(s => s.drivers);
  const d = drivers.find(x => x.id === status.driver);
  const meta = driverMeta(d ?? { id: status.driver ?? '', name: status.driver_name ?? 'Scanner', description: '' });
  const state = status.state;
  const dropped = Object.entries(status.dropped ?? {}).filter(([, n]) => n > 0);
  const droppedTotal = dropped.reduce((a, [, n]) => a + n, 0);
  const Icon = meta.icon;

  return (
    <section className={`cap-session is-${state}`} aria-label="Scanner session">
      <div className="cap-session-head">
        <span className={`cap-tally is-${state}`} aria-hidden />
        <div className="grow">
          <div className="cap-state" aria-live="polite">{STATE_WORD[state] ?? state}</div>
          <div className="cap-session-sub">
            <Icon size={13} aria-hidden /> <span className="truncate">{meta.title}</span>
          </div>
        </div>
        {!streamOk && (
          <span className="cap-conn is-warn" title="The live view is reconnecting to the server">
            <span className="dot warn" aria-hidden /> Reconnecting
          </span>
        )}
      </div>
      <div className="cap-readouts" role="group" aria-label="Live numbers">
        <Readout value={fmtCount(status.points ?? 0)} label="points" />
        <Readout value={fmtCount(status.frames ?? 0)} label="frames" />
        <Readout value={(status.fps ?? 0).toFixed(1)} label="frames / s" />
        <Readout value={fmtClock(status.elapsed_s)} label="time" />
      </div>
      {droppedTotal > 0 && (
        <p className="caption">
          Left out {fmtCount(droppedTotal)} frame{droppedTotal > 1 ? 's' : ''}: {dropped.map(([k, n]) => `${fmtCount(n)} ${DROPPED_WORD[k] ?? k.replace(/_/g, ' ')}`).join(', ')}.
        </p>
      )}
      {status.error && (
        <p className="err-text" role="alert">
          <XCircle size={14} aria-hidden /> <span>{status.error}</span>
        </p>
      )}
    </section>
  );
}

function Readout({ value, label }: { value: string; label: string }) {
  return (
    <div className="cap-readout">
      <span className="cap-readout-value mono">{value}</span>
      <span className="cap-readout-label">{label}</span>
    </div>
  );
}

function MarkerMapBlock() {
  const status = useCapture(s => s.status)!;
  const busy = useCaptureAction(s => s.busy);
  const dev = status.device ?? {};
  const mapping = dev.phase === 'mapping';
  const frozen = !!dev.map_frozen;
  const count = dev.map_markers ?? 0;
  const running = status.state === 'running';
  const hasPoints = (status.points ?? 0) > 0;
  const stage = mapping ? 1 : frozen ? 3 : count ? 3 : 1;

  const command = (label: string, cmd: string) =>
    runAction(label, async () => {
      await captureAction(`driver/${cmd}`);
      if (!running && cmd === 'map_markers') await captureAction('start');
    });

  const text = mapping
    ? 'Sweep slowly over every marker on and around the part until each one shows red. Nothing is recorded yet.'
    : frozen
      ? `The map is fixed with ${count} markers. The surface scan now tracks against it, so every side lines up. To turn or flip the part, the markers must be on the part itself.`
      : count
        ? `Building the map while scanning (${count} markers). For the best accuracy, map the markers first.`
        : 'Map the markers first, as in Revo Metro: sweep over every marker, finish the map, then scan the surface.';

  return (
    <Block guide="capture.markers" title="Marker map" tone={mapping ? 'signal' : undefined} aside={<span className="mono small">{count} marker{count === 1 ? '' : 's'}</span>}>
      <ol className="cap-stages" aria-label="Marker map steps">
        {['Map markers', 'Finish map', 'Scan surface'].map((label, i) => {
          const n = i + 1;
          const done = frozen ? n < 3 : false;
          const current = !done && n === stage;
          return (
            <li key={label} className={`${done ? 'is-done' : ''} ${current ? 'is-current' : ''}`}>
              <span className="cap-stage-n">{done ? <CheckCircle2 size={14} aria-hidden /> : n}</span>
              <span>{label}</span>
            </li>
          );
        })}
      </ol>
      <p className="hint-text">{text}</p>
      <div className="row wrap">
        {mapping ? (
          <Button variant="primary" loading={busy === 'finish-map'} disabled={count < 4} onClick={() => command('finish-map', 'finish_map')}>
            Finish the map · {count}
          </Button>
        ) : (
          <Button variant={frozen ? 'secondary' : 'primary'} loading={busy === 'map-markers'} disabled={hasPoints} onClick={() => command('map-markers', 'map_markers')}>
            {frozen ? 'Map again' : 'Map the markers first'}
          </Button>
        )}
        {(count > 0 || mapping) && (
          <Button variant="ghost" loading={busy === 'clear-map'} disabled={hasPoints} onClick={() => command('clear-map', 'clear_map')}>
            Clear the map
          </Button>
        )}
      </div>
      {mapping && count < 4 && <p className="caption">Needs at least 4 markers before the map can be finished.</p>}
      {hasPoints && !mapping && <p className="caption">Save or discard this scan before starting a new map — a new map is a new coordinate frame.</p>}
      <ActionError labels={['finish-map', 'map-markers', 'clear-map']} />
    </Block>
  );
}

function PendingBlock() {
  const pending = useCapture(s => s.pending);
  const busy = useCaptureAction(s => s.busy);
  return (
    <Block title={<>New scan waiting <span className="count-chip">{pending.length}</span></>} tone="signal">
      <p className="hint-text">Each export is lined up with the scans before it and checked. Adding a scan that covers the same area, or does not line up exactly, doubles the surface — so you decide.</p>
      {pending.map(p => (
        <div key={p.scan_id} className="cap-pending">
          <div className="row spread">
            <span className="strong truncate">{p.name}</span>
            <span className="caption mono">{fmtCount(p.points)} pts</span>
          </div>
          <AssessmentView
            a={p.assessment}
            actions={
              <>
                <Button size="sm" variant={p.assessment.recommendation === 'merge' ? 'primary' : 'secondary'} loading={busy === `fuse-${p.scan_id}`} onClick={() => runAction(`fuse-${p.scan_id}`, () => decidePending(p.scan_id, 'fuse'))}>
                  Add to this scan
                </Button>
                <Button size="sm" variant={p.assessment.recommendation !== 'merge' ? 'primary' : 'secondary'} loading={busy === `keep-${p.scan_id}`} onClick={() => runAction(`keep-${p.scan_id}`, async () => { await decidePending(p.scan_id, 'keep_separate'); await useStore.getState().refreshAssets(); })}>
                  Keep it separately
                </Button>
                <Button size="sm" variant="ghost" loading={busy === `drop-${p.scan_id}`} onClick={() => runAction(`drop-${p.scan_id}`, () => decidePending(p.scan_id, 'discard'))}>
                  Discard
                </Button>
              </>
            }
          />
          <ActionError labels={[`fuse-${p.scan_id}`, `keep-${p.scan_id}`, `drop-${p.scan_id}`]} />
        </div>
      ))}
    </Block>
  );
}

function defaultName() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return `Capture ${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function SaveBlock({ onDiscard }: { onDiscard: () => void }) {
  const status = useCapture(s => s.status)!;
  const busy = useCaptureAction(s => s.busy);
  const [name, setName] = useState('');
  const [auto, setAuto] = useState<boolean | null>(null);
  const [serverAuto, setServerAuto] = useState(true);
  const job = useCaptureAction(s => s.savedJob);
  // the automation started by an earlier save (also after a reload): an unfinished autopilot job on the saved scan
  const automating = useStore(s => !!status.saved_asset_id && s.jobs.some(j => j.kind === 'autopilot' && !['done', 'failed', 'cancelled'].includes(j.status) && ((j.payload?.asset_ids as string[] | undefined) ?? []).includes(status.saved_asset_id!)));
  const live = LIVE.has(status.state);
  const running = status.state === 'running';

  useEffect(() => {
    loadAutoOnCapture().then(setServerAuto);
  }, []);

  if (!status.unsaved && status.saved_asset_id) return <SavedCard assetId={status.saved_asset_id} automated={!!job || automating} onDiscard={() => runAction('discard', discardCapture)} />;

  const save = () =>
    runAction('save', async () => {
      const res = await saveCapture(name.trim() || undefined, auto ?? serverAuto);
      useCaptureAction.setState({ savedJob: res.job?.id ?? null });
      setName('');
      useStore.getState().toast({ kind: 'ok', title: 'Scan saved', body: res.job ? 'Your automation is processing it — see Jobs.' : 'It is in the model list on the left.' });
    });

  if (running) {
    // while scanning: one quiet line; the name and options come when scanning stops
    return (
      <div className="cap-save-line">
        <Save size={15} aria-hidden />
        <span className="grow">Keep what you have so far?</span>
        <Button size="sm" variant="secondary" loading={busy === 'save'} onClick={save}>
          Save now
        </Button>
        <ActionError labels={['save']} />
      </div>
    );
  }

  return (
    <Block guide="capture.save" title="Save this scan" tone={!live ? 'signal' : undefined} aside={<span className="mono small">{fmtCount(status.points ?? 0)} points</span>}>
      <Field label="Name" inline={false}>
        <TextInput value={name} placeholder={defaultName()} onChange={setName} onEnter={save} />
      </Field>
      <Field label="Then clean it automatically" help="Runs your automation on the saved scan (Settings → Automations). The scan itself is kept exactly as captured.">
        <Switch checked={auto ?? serverAuto} onChange={setAuto} label="Then clean it automatically" />
      </Field>
      <div className="row wrap">
        <Button variant="primary" icon={<Save size={16} />} loading={busy === 'save'} onClick={save}>
          Save scan
        </Button>
        <Button variant="ghost" icon={<Trash2 size={15} />} disabled={live} onClick={onDiscard} title={live ? 'Stop scanning first' : undefined}>
          Discard
        </Button>
      </div>
      <ActionError labels={['save', 'discard']} />
    </Block>
  );
}

function SavedCard({ assetId, automated, onDiscard }: { assetId: string; automated: boolean; onDiscard: () => void }) {
  const asset = useStore(s => s.byId.get(assetId));
  const cleanIt = () => {
    const st = useStore.getState();
    st.activate(assetId, false);
    useStore.setState({ visible: [assetId] });
    st.goStep('clean');
  };
  return (
    <ResultCard
      tone="ok"
      title={asset ? `Saved as “${asset.name}”` : 'Scan saved'}
      actions={
        <>
          <Button variant="primary" icon={<CleanGlyph size={16} />} onClick={cleanIt}>
            {automated ? 'Go to Clean' : 'Clean this scan'} <ArrowRight size={15} aria-hidden />
          </Button>
          <Button variant="ghost" onClick={onDiscard} title="Ends the scanner session. Your saved scan stays in the model list.">Close session</Button>
        </>
      }
    >
      {asset && <Metric label="Points" value={fmtCount(asset.stats.points)} />}
      <span className="caption">{automated ? 'Your automation is processing it — follow it in Jobs at the top.' : 'Scanning more adds to this session; save again to keep the additions.'}</span>
    </ResultCard>
  );
}

const SEVERITY_ICON = { ok: CheckCircle2, info: Info, warning: TriangleAlert, error: XCircle } as const;

function GuidanceBlock() {
  const g = useCapture(s => s.guidance);
  const timeline = useCapture(s => s.timeline);
  if (!g) return null;
  const c = g.coverage?.completeness;
  const messages = g.messages?.length ? g.messages : g.status ? [g.status] : [];
  const holes = g.holes ?? [];
  return (
    <Block guide="capture.guidance" title="Guidance" aside={c != null ? <span className="cap-complete mono">{Math.round(c * 100)} % complete</span> : undefined}>
      {messages.length > 0 && (
        <ul className="cap-guide">
          {messages.map((m, i) => {
            const Icon = SEVERITY_ICON[m.severity as keyof typeof SEVERITY_ICON] ?? Info;
            return (
              <li key={`${m.code}-${i}`} className={`sev-${m.severity}`}>
                <Icon size={15} aria-hidden />
                <span>{m.message}</span>
              </li>
            );
          })}
        </ul>
      )}
      {holes.length > 0 && (
        <>
          <div className="cap-subhead">Scan these areas next</div>
          <ol className="cap-holes">
            {holes.slice(0, 6).map((h, i) => <HoleRow key={h.id} hole={h} index={i} />)}
          </ol>
        </>
      )}
      {timeline.length > 3 && <CoverageSparkline points={timeline.map(p => p.c)} />}
    </Block>
  );
}

function HoleRow({ hole: h, index }: { hole: Hole; index: number }) {
  const look = () => {
    const st = useCapture.getState();
    // looking at a gap is taking the view: stop following the scanner, or the next pose would pull it back
    if (st.follow && LIVE.has(st.status?.state ?? '')) useCapture.setState({ follow: false });
    getViewer()?.lookFrom(new THREE.Vector3(...h.direction));
  };
  return (
    <li className={`cap-hole kind-${h.kind}`}>
      <span className="cap-hole-n mono">{index + 1}</span>
      <div className="grow">
        <div className="cap-hole-hint">{h.hint || h.message || HOLE_WORD[h.kind]}</div>
        <div className="caption">
          {HOLE_WORD[h.kind]} · {fmtArea(h.area_mm2)}
          {h.region ? ` · ${h.region}` : ''}
        </div>
      </div>
      <Button size="sm" variant="ghost" icon={<Eye size={14} />} onClick={look}>
        Show
      </Button>
    </li>
  );
}

function CoverageSparkline({ points }: { points: number[] }) {
  const W = 320;
  const H = 64;
  const y = (v: number) => H - 6 - Math.max(0, Math.min(1, v)) * (H - 14);
  const n = points.length;
  const x = (i: number) => (i / Math.max(n - 1, 1)) * W;
  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p).toFixed(1)}`).join('');
  const area = `${line}L${W},${H - 6}L0,${H - 6}Z`;
  const last = points[n - 1] ?? 0;
  return (
    <div className="sparkline cap-spark">
      <div className="row spread">
        <span className="cap-subhead">Coverage over time</span>
        <span className="spark-label mono">{Math.round(last * 100)} %</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Coverage ${Math.round(last * 100)} percent; the target is 90 percent`}>
        <line x1={0} x2={W} y1={y(0.9)} y2={y(0.9)} className="spark-target" />
        <text x={2} y={y(0.9) - 4} className="cap-spark-note">target 90 %</text>
        <line x1={0} x2={W} y1={H - 6} y2={H - 6} className="cap-spark-base" />
        <path d={area} className="cap-spark-area" />
        <path d={line} className="spark-line" />
        <circle cx={W} cy={y(last)} r={3.5} className="spark-dot" />
      </svg>
    </div>
  );
}

/** The density key, drawn with the same colours as the live points: thin surface on the left, target density on the right. */
function densityRamp() {
  const style = densityStyle();
  const c = new THREE.Color();
  const stops = Array.from({ length: 25 }, (_, i) => `#${scalarColor((i / 24) * 1.2, style, c).getHexString()} ${((i / 24) * 100).toFixed(1)}%`);
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}

function LiveViewBlock() {
  useStore(s => s.theme); // the key follows the stage colours
  const follow = useCapture(s => s.follow);
  const colorBy = useCapture(s => s.colorBy);
  const status = useCapture(s => s.status)!;
  const drivers = useCapture(s => s.drivers);
  const caps = drivers.find(d => d.id === status.driver)?.capabilities;
  const streaming = caps?.streaming !== false;
  return (
    <Block title="Live view">
      {streaming && (
        <Field label={<span className="row"><LocateFixed size={15} aria-hidden /> Follow the scanner</span>} help="Dragging the view switches to free view; the chip in the view brings it back.">
          <Switch checked={follow} onChange={on => useCapture.setState({ follow: on })} label="Follow the scanner" />
        </Field>
      )}
      <Field label="Colour the points by" inline={false}>
        <Segmented
          size="sm"
          ariaLabel="Colour the points by"
          value={colorBy}
          onChange={v => {
            useCapture.setState({ colorBy: v });
            applyLiveStyle();
          }}
          options={[{ value: 'density', label: 'Density' }, { value: 'color', label: 'Texture' }, { value: 'solid', label: 'Plain' }]}
        />
      </Field>
      {colorBy === 'density' && (
        <div className="cap-density-key">
          <span className="cap-density-ramp" style={{ background: densityRamp() }} aria-hidden />
          <div className="row spread caption">
            <span>thin — scan more here</span>
            <span>target density</span>
          </div>
        </div>
      )}
    </Block>
  );
}

/* ------------------------------------------------------------------ bring in files */

function FilesBlock({ compact }: { compact?: boolean }) {
  const [copied, setCopied] = useState(false);
  const { enabled, user } = useAuth();
  const [key, setKey] = useState<string | null>(null);
  useEffect(() => {
    if (enabled && user?.admin) api.get<{ key: string }>('/api/auth/machine-key').then(r => setKey(r.key)).catch(() => undefined);
  }, [enabled, user?.admin]);
  // with accounts on, the bridge signs in with the machine key (Settings -> Users)
  const keyArg = !enabled ? '' : ` --key ${key ?? '<key from the admin: Settings → Users>'}`;
  const cmd = `cloudclean bridge --server ${location.origin} --watch "<Revo Metro export folder>"${keyArg}`;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard blocked: the text is selectable */
    }
  };
  const bridge = (
    <details className="disclosure">
      <summary>
        <Wifi size={16} aria-hidden /> How the Revo Metro bridge works
        <ChevronDown size={16} className="chev" aria-hidden />
      </summary>
      <div className="disclosure-body">
        <ol className="cap-howto">
          <li>
            <span className="index-bubble">1</span>
            <div>
              On the PC that runs Revo Metro, start the bridge once. It watches Revo Metro’s export folder:
              <div className="cap-code">
                <code className="mono">{cmd}</code>
                <button type="button" className="icon-btn icon-btn-sm" aria-label="Copy the command" data-tip={copied ? 'Copied' : 'Copy'} onClick={copy}>
                  {copied ? <CheckCircle2 size={15} /> : <Copy size={15} />}
                </button>
              </div>
            </div>
          </li>
          <li>
            <span className="index-bubble">2</span>
            <div>Scan in Revo Metro and export each scan as usual.</div>
          </li>
          <li>
            <span className="index-bubble">3</span>
            <div>Each export appears here within seconds, lined up with the scans before it and checked. You decide whether to add it.</div>
          </li>
        </ol>
      </div>
    </details>
  );
  if (compact) return bridge;
  return (
    <Block guide="capture.files" title="Or bring in files">
      <p className="hint-text">Already scanned? Open PLY, STL, OBJ and other scan files — or drop them anywhere on the window.</p>
      <div className="row">
        <Button icon={<Upload size={16} />} onClick={() => pickFiles()}>
          Open scan files
        </Button>
      </div>
      {bridge}
    </Block>
  );
}
