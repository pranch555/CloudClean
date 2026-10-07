import { useEffect, useMemo, useState } from 'react';
import { CheckCircle2, ChevronDown, CircleSlash, Link2, Loader2, Play, Plus, RefreshCw, RotateCcw, RotateCw, Square, TriangleAlert, Unplug, X, XCircle } from 'lucide-react';
import { local } from '../../store';
import {
  estimateProgramSeconds,
  fmtDeg,
  fmtMinutes,
  KIND_LABEL,
  PHASE_LABEL,
  programRunning,
  searchTurntables,
  turntable,
  useTurntable,
  type TurntableCapabilities,
  type TurntableDevice,
  type TurntableProgram,
  type TurntableStatus,
} from '../../lib/turntable';
import { Button, Field, IconButton, NumberInput, Progress, Segmented, Slider, Switch } from '../../ui/primitives';
import { captureAction, useCapture } from '../../features/capture/captureStore';
import { TiltGlyph, TurntableDial } from '../../features/capture/TurntableDial';
import { Block } from '../StepFrame';
import { Callout, ConfirmDialog } from './parts';

const FALLBACK_CAPS: TurntableCapabilities = { tilt: false, tilt_range: null, speed_range: [25, 90], interval_range: [5, 30], max_rotations: 5 };
const DEFAULT_PROGRAM: TurntableProgram = { interval_deg: 30, frames_per_stop: 3, direction: 'cw', rotations: [{ tilt_deg: 0 }], sync_scan: true };
const NUDGES = [-90, -15, -5, 5, 15, 90];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const signed = (d: number) => `${d > 0 ? '+' : d < 0 ? '−' : ''}${Math.abs(d)}°`;

/** Step 1's turntable section: connect, see the platter, turn it, and run a turntable scan. */
export function TurntableBlock() {
  const support = useTurntable(s => s.support);
  const status = useTurntable(s => s.status);
  const offline = useTurntable(s => s.offline);
  if (support === 'no') {
    return (
      <Block guide="capture.turntable" title="Turntable">
        <Callout tone="info">
          <b>Turntable control is not installed on the server yet.</b> Update CloudClean on the DGX Spark to turn and tilt the Revopoint turntable from here. Until then, turn the part by hand between passes.
        </Callout>
      </Block>
    );
  }
  if (!status && offline) {
    return (
      <Block guide="capture.turntable" title="Turntable">
        <Callout tone="warn">Cannot reach the server to ask about the turntable. It will show up here as soon as the connection is back.</Callout>
      </Block>
    );
  }
  if (!status) {
    return (
      <Block guide="capture.turntable" title="Turntable" aside={<span className="cap-conn"><Loader2 size={13} className="spin" aria-hidden /> Checking…</span>}>
        <div className="cap-skeleton" aria-busy="true">
          <span />
        </div>
      </Block>
    );
  }
  return status.connected ? <TurntableControls status={status} /> : <TurntableConnect status={status} />;
}

function ConnChip({ status }: { status: TurntableStatus }) {
  return status.connected ? (
    <span className="cap-conn is-on">
      <span className={`dot ${status.moving ? 'live' : 'ok'}`} aria-hidden /> Connected
    </span>
  ) : (
    <span className="cap-conn">
      <span className="dot" aria-hidden /> Not connected
    </span>
  );
}

function rssiWord(rssi: number | null): string | null {
  if (rssi == null) return null;
  return rssi >= -60 ? 'strong signal' : rssi >= -75 ? 'good signal' : 'weak signal';
}

function SignalBars({ rssi }: { rssi: number | null }) {
  if (rssi == null) return null;
  const n = rssi >= -60 ? 3 : rssi >= -75 ? 2 : 1;
  return (
    <span className="tt-bars" aria-hidden>
      {[1, 2, 3].map(i => <i key={i} className={i <= n ? 'on' : ''} />)}
    </span>
  );
}

function TurntableConnect({ status }: { status: TurntableStatus }) {
  const devices = useTurntable(s => s.devices);
  const searching = useTurntable(s => s.searching);
  const busy = useTurntable(s => s.busy);
  const error = useTurntable(s => s.error);
  const [pick, setPick] = useState('');

  useEffect(() => {
    if (devices == null && !searching) searchTurntables(4);
  }, []);

  const list: TurntableDevice[] = devices ?? [{ id: 'simulated', name: 'Simulated dual-axis turntable', kind: 'simulated', rssi: null }];
  useEffect(() => {
    if (pick && list.some(d => d.id === pick)) return;
    // only a real table is picked for you: the practice one must be chosen on purpose (it moves nothing)
    const real = list.filter(d => d.kind !== 'simulated');
    setPick((real.find(d => d.remembered) ?? real[0])?.id ?? '');
  }, [devices]);
  const chosen = list.find(d => d.id === pick);
  const noBluetooth = status.bluetooth && status.bluetooth.installed === false;

  return (
    <Block guide="capture.turntable" title="Turntable" aside={<ConnChip status={status} />}>
      <p className="hint-text">A Revopoint turntable turns — and tilts — the part for you, and holds still at every step while the scanner records.</p>
      <div className="choices tt-devices" role="radiogroup" aria-label="Turntable">
        {list.map(d => {
          const sim = d.kind === 'simulated';
          return (
            <button key={d.id} type="button" role="radio" aria-checked={d.id === pick} className={`choice tt-device ${d.id === pick ? 'is-on' : ''}`} onClick={() => setPick(d.id)} onDoubleClick={() => turntable.connect(d.id, String(d.kind ?? 'auto'))}>
              <span className="choice-radio" aria-hidden />
              <span className="choice-icon">
                <RotateCw size={18} />
              </span>
              <span className="choice-text">
                <span className="choice-title">
                  <span className="truncate">{sim ? 'Practice turntable (simulated)' : d.name ?? d.id}</span>
                  {d.remembered && <span className="badge">last used</span>}
                </span>
                <span className="choice-body tt-device-meta">
                  {sim ? 'Nothing physical moves: for trying a turntable scan without the table' : KIND_LABEL[String(d.kind)] ?? 'Turntable'}
                  {rssiWord(d.rssi) && (
                    <>
                      {' · '}
                      <SignalBars rssi={d.rssi} /> {rssiWord(d.rssi)}
                    </>
                  )}
                </span>
              </span>
            </button>
          );
        })}
        {searching && (
          <div className="tt-searching" role="status">
            <Loader2 size={15} className="spin" aria-hidden /> Looking for turntables nearby…
          </div>
        )}
      </div>
      {noBluetooth && <p className="caption">Bluetooth support is not installed on the server, so only the simulated turntable is listed.</p>}
      {!searching && devices && !noBluetooth && !devices.some(d => d.kind !== 'simulated') && (
        <div className="tt-missing" role="status">
          <b>Your turntable was not found.</b>
          <ul>
            <li>Is it switched on? Its light should be on.</li>
            <li>Is Revo Metro open on any computer? Close it (or disconnect the turntable in it): the turntable accepts one connection at a time and hides while it is connected.</li>
            <li>Is it within about 10 m of the computer running CloudClean?</li>
          </ul>
          <span>Then press Search again.</span>
        </div>
      )}
      {(error || status.error) && (
        <p className="err-text" role="alert">
          <XCircle size={14} aria-hidden /> <span>{error ?? status.error}</span>
        </p>
      )}
      <div className="row wrap">
        <Button variant={chosen?.kind === 'simulated' ? 'secondary' : 'primary'} icon={<Link2 size={15} />} loading={busy === 'connect'} disabled={!chosen} onClick={() => chosen && turntable.connect(chosen.id, String(chosen.kind ?? 'auto'))}>
          {!chosen ? 'Choose a turntable' : chosen.kind === 'simulated' ? 'Use the practice turntable' : 'Connect the turntable'}
        </Button>
        <Button variant="ghost" icon={<RefreshCw size={15} />} disabled={searching} onClick={() => searchTurntables(4)}>
          Search again
        </Button>
      </div>
    </Block>
  );
}

function TurntableControls({ status }: { status: TurntableStatus }) {
  const busy = useTurntable(s => s.busy);
  const error = useTurntable(s => s.error);
  const captureState = useCapture(s => s.status?.state);
  const caps = status.capabilities ?? FALLBACK_CAPS;
  const running = programRunning(status);
  const moving = status.moving;
  const locked = moving || running || !!busy;
  const [speedLo, speedHi] = caps.speed_range ?? [25, 90];
  const [speed, setSpeed] = useState(() => clamp(Math.round(status.speed_s_per_rev ?? 53), speedLo, speedHi));
  const [confirmTilt, setConfirmTilt] = useState<number | null>(null);

  useEffect(() => {
    if (status.speed_s_per_rev) setSpeed(clamp(Math.round(status.speed_s_per_rev), speedLo, speedHi));
  }, [status.speed_s_per_rev]);

  const speedArg = () => (status.speed_s_per_rev != null && Math.round(status.speed_s_per_rev) === speed ? null : speed);
  const rotate = (deg: number) => turntable.rotate(deg, speedArg());
  const scanning = captureState === 'running';
  const tilt = (deg: number) => (scanning ? setConfirmTilt(deg) : turntable.tilt(deg));

  return (
    <Block
      guide="capture.turntable"
      title="Turntable"
      aside={
        <>
          <ConnChip status={status} />
          <IconButton size="sm" label="Disconnect the turntable" tip="top" disabled={busy === 'disconnect'} onClick={() => turntable.disconnect()}>
            <Unplug size={15} />
          </IconButton>
        </>
      }
    >
      {status.kind === 'simulated' && (
        <p className="tt-sim-note" role="status">
          <b>This is the practice turntable: nothing physical moves.</b> To use your table,{' '}
          <button type="button" className="link" onClick={() => turntable.disconnect()}>disconnect this one</button> and connect your turntable.
        </p>
      )}
      <div className="tt-head">
        <TurntableDial status={status} size={138} />
        <div className="tt-side">
          <div className="tt-name truncate" title={status.name ?? undefined}>
            {status.name ?? 'Turntable'}
          </div>
          <dl className="tt-facts">
            {caps.tilt && (
              <div>
                <dt>Tilt</dt>
                <dd>
                  <TiltGlyph tilt={status.tilt_deg} size={34} />
                  <span className="mono">{fmtDeg(status.tilt_deg, 0, true)}</span>
                </dd>
              </div>
            )}
            <div>
              <dt>Speed</dt>
              <dd>
                <span className="mono">{status.speed_s_per_rev != null ? Math.round(status.speed_s_per_rev) : '–'}</span> <span className="unit">s/turn</span>
              </dd>
            </div>
            <div>
              <dt>Now</dt>
              <dd className={moving ? 'is-live' : ''}>{moving ? (status.device_state?.tilting ? 'Tilting' : 'Turning') : running ? PHASE_LABEL[status.program?.phase ?? ''] ?? 'Running' : 'Still'}</dd>
            </div>
          </dl>
          <Button variant="danger" block className={`tt-stop ${moving || running ? 'is-live' : ''}`} icon={<Square size={14} />} onClick={() => turntable.stop()}>
            Stop
          </Button>
        </div>
      </div>

      {!status.validated && (
        <Callout tone="warn" action={<Button size="sm" variant="secondary" disabled={locked} onClick={() => rotate(5)}>Try +5°</Button>}>
          Control protocol not yet confirmed on your turntable — try a small rotation first.
        </Callout>
      )}
      {(error || status.error) && (
        <p className="err-text" role="alert">
          <XCircle size={14} aria-hidden /> <span>{error ?? status.error}</span>
        </p>
      )}
      {status.last_move?.note && !moving && <p className="caption">{status.last_move.note}</p>}

      {caps.continuous && <SpinSection status={status} speed={speed} />}

      <ProgramSection status={status} caps={caps} speed={speed} locked={locked || !!status.spin} />

      <details className="disclosure tt-manual" open={local.get('ttManualOpen', false)} onToggle={e => local.set('ttManualOpen', (e.target as HTMLDetailsElement).open)}>
        <summary>
          Turn by hand <span className="sub">nudge, tilt, speed</span>
          <ChevronDown size={16} className="chev" aria-hidden />
        </summary>
        <div className="disclosure-body">
          <ManualControls status={status} caps={caps} locked={locked} rotate={rotate} tilt={tilt} speed={speed} setSpeed={setSpeed} />
        </div>
      </details>

      {confirmTilt != null && (
        <ConfirmDialog
          title="Tilt while scanning?"
          icon={<TriangleAlert size={20} className="warn-glyph" />}
          onCancel={() => setConfirmTilt(null)}
          choices={[
            { label: 'Tilt anyway', variant: 'secondary', onPick: () => { const d = confirmTilt; setConfirmTilt(null); turntable.tilt(d); } },
            {
              label: 'Pause, then tilt',
              variant: 'primary',
              onPick: async () => {
                const d = confirmTilt;
                setConfirmTilt(null);
                try {
                  await captureAction('pause');
                } catch {
                  /* the tilt still waits for the user's choice */
                }
                turntable.tilt(d);
              },
            },
          ]}
        >
          <p>The scanner is recording. Tilting to <b className="mono">{fmtDeg(confirmTilt, 0, true)}</b> moves the part while frames are captured, so they can blur or lose tracking.</p>
          <p className="caption">Pause first and press Resume when the platter is still.</p>
        </ConfirmDialog>
      )}
    </Block>
  );
}

function ManualControls({ status, caps, locked, rotate, tilt, speed, setSpeed }: { status: TurntableStatus; caps: TurntableCapabilities; locked: boolean; rotate: (d: number) => void; tilt: (d: number) => void; speed: number; setSpeed: (v: number) => void }) {
  const [by, setBy] = useState(() => local.get('ttRotateBy', 30));
  const [tLo, tHi] = caps.tilt_range ?? [-30, 30];
  const current = Math.round(status.tilt_deg ?? 0);
  const [target, setTarget] = useState(current);
  const [speedLo, speedHi] = caps.speed_range ?? [25, 90];

  useEffect(() => {
    if (!status.moving) setTarget(current);
  }, [current]);

  return (
    <>
      <div className="stack tight">
        <div className="field-label">Turn by</div>
        <div className="tt-nudges" role="group" aria-label="Turn the platter by a fixed step">
          {NUDGES.map(d => (
            <button key={d} type="button" className="tt-nudge mono" disabled={locked} onClick={() => rotate(d)} aria-label={`Turn ${d > 0 ? 'clockwise' : 'counter-clockwise'} by ${Math.abs(d)} degrees`}>
              {signed(d)}
            </button>
          ))}
        </div>
        <div className="row">
          <div style={{ width: 120 }}>
            <NumberInput value={by} step={1} min={-720} max={720} unit="°" label="Degrees to turn" onChange={v => { const r = Math.round(v); setBy(r); local.set('ttRotateBy', r); }} />
          </div>
          <Button icon={by < 0 ? <RotateCcw size={15} /> : <RotateCw size={15} />} disabled={locked || !by} onClick={() => rotate(by)}>
            Turn {signed(by)}
          </Button>
        </div>
        <p className="caption">+ turns clockwise, seen from above. The turntable takes whole degrees.</p>
      </div>

      {caps.tilt && (
        <Field label={<>Tilt <span className="caption">now {fmtDeg(current, 0, true)}</span></>} inline={false}>
          <Slider value={target} min={tLo} max={tHi} step={1} label="Tilt target" format={v => fmtDeg(v, 0, true)} onChange={v => setTarget(Math.round(v))} />
          <div className="row">
            <Button size="sm" variant="secondary" disabled={locked || target === current} onClick={() => tilt(target)}>
              Tilt to {fmtDeg(target, 0, true)}
            </Button>
            <Button size="sm" variant="ghost" disabled={locked || current === 0} onClick={() => { setTarget(0); tilt(0); }}>
              Level it
            </Button>
          </div>
        </Field>
      )}

      <Field label="Turning speed" inline={false} help="Used for your turns and for the turntable scan. Slower gives the scanner more time at every angle.">
        <Slider value={speed} min={speedLo} max={speedHi} step={1} label="Turning speed" format={v => `${v} s/turn`} onChange={v => setSpeed(Math.round(v))} />
      </Field>
    </>
  );
}

/** Turn while scanning: the table keeps turning until stopped, holding while the scan is paused (the user's way to
 *  scan with the MetroY's laser lines, which only build a surface while the part moves through them). */
function SpinSection({ status, speed }: { status: TurntableStatus; speed: number }) {
  const busy = useTurntable(s => s.busy);
  const spin = status.spin;
  const program = programRunning(status);
  const scanState = useCapture(s => s.status?.state);
  const now = !spin ? null : spin.turning ? 'Turning' : spin.held_by_scan ? `Holding while the scan is ${scanState === 'paused' ? 'paused' : 'stopped'}` : 'Holding';
  return (
    <section className={`tt-program tt-spin ${spin ? 'is-running' : ''}`} aria-label="Turn while scanning" data-guide="capture.turntable-spin">
      <div className="tt-program-head">
        <h4>{spin && <span className={`dot ${spin.turning ? 'live' : ''}`} aria-hidden />} Turn while scanning</h4>
        <span className="caption">The table keeps turning until you stop it. It holds when the scan is paused or stopped and turns again when the scan runs.</span>
      </div>
      {spin ? (
        <>
          <p className="tt-spin-now" aria-live="polite">{now} · {Math.round(status.speed_s_per_rev ?? speed)} s per turn</p>
          <Button variant="danger" block icon={<Square size={14} />} loading={busy === 'spin-stop'} onClick={() => turntable.spin(false)}>
            Stop turning
          </Button>
        </>
      ) : (
        <Button variant="primary" block icon={<Play size={16} />} disabled={program || status.moving || !!busy} loading={busy === 'spin'} onClick={() => turntable.spin(true, speed)}>
          Start turning
        </Button>
      )}
    </section>
  );
}

function ProgramSection({ status, caps, speed, locked }: { status: TurntableStatus; caps: TurntableCapabilities; speed: number; locked: boolean }) {
  const captureState = useCapture(s => s.status?.state);
  const captureActive = useCapture(s => !!s.status?.active && s.status.state !== 'closed');
  const fps = useCapture(s => s.status?.fps ?? 0);
  // a laser-line scanner (the MetroY) sees a few lines per frame: it scans while the table turns, never at stops
  const sweeps = useCapture(s => captureActive && !!s.drivers.find(d => d.id === s.status?.driver)?.capabilities?.sweeps);
  const [form, setFormState] = useState<TurntableProgram>(() => ({ ...DEFAULT_PROGRAM, ...local.get<Partial<TurntableProgram>>('ttProgram', {}) }));
  const [confirm, setConfirm] = useState(false);
  const busy = useTurntable(s => s.busy);
  const running = programRunning(status);
  const p = status.program;
  const [iLo, iHi] = caps.interval_range ?? [5, 30];
  const [tLo, tHi] = caps.tilt_range ?? [0, 0];
  const maxTurns = caps.max_rotations ?? 5;

  const setForm = (patch: Partial<TurntableProgram>) =>
    setFormState(f => {
      const next = { ...f, ...patch };
      local.set('ttProgram', next);
      return next;
    });

  // keep the form within what this table can do
  const plan: TurntableProgram = useMemo(
    () => ({
      ...form,
      interval_deg: clamp(Math.round(form.interval_deg), iLo, iHi),
      frames_per_stop: clamp(Math.round(form.frames_per_stop), 1, 100),
      rotations: form.rotations.slice(0, maxTurns).map(r => ({ tilt_deg: caps.tilt ? clamp(Math.round(r.tilt_deg), tLo, tHi) : 0 })),
      speed_s_per_rev: speed,
      mode: sweeps && form.sync_scan ? 'continuous' : 'step',
    }),
    [form, caps, speed, sweeps],
  );
  const sweep = plan.mode === 'continuous';
  const stops = sweep ? 1 : Math.ceil(360 / plan.interval_deg - 1e-9);
  const seconds = estimateProgramSeconds(plan, speed, status.tilt_deg ?? 0, fps > 1 ? fps : 10);
  const tilts = plan.rotations.some(r => r.tilt_deg !== Math.round(status.tilt_deg ?? 0));

  const start = () => {
    if (captureState === 'running' && !plan.sync_scan && tilts) setConfirm(true);
    else turntable.startProgram(plan);
  };

  if (running && p) return <ProgramProgress status={status} />;

  return (
    <section className="tt-program" aria-label="Turntable scan">
      <div className="tt-program-head">
        <h4>Scan all the way round</h4>
        <span className="caption">{sweep ? 'The table turns slowly all the way round while the scanner records: its laser lines build the surface as the part moves through them.' : `The table stops every ${plan.interval_deg}° and the scanner records at each stop.`}</span>
      </div>
      {p && !running && <ProgramResult status={status} />}
      {!sweep && (
        <>
          <Field label="Stop every" inline={false}>
            <Slider value={plan.interval_deg} min={iLo} max={iHi} step={1} label="Stop every" format={v => `${v}° · ${Math.ceil(360 / v - 1e-9)} stops`} onChange={v => setForm({ interval_deg: Math.round(v) })} />
          </Field>
          <Field label="Frames at each stop">
            <NumberInput value={plan.frames_per_stop} step={1} min={1} max={100} onChange={v => setForm({ frames_per_stop: Math.round(v) })} />
          </Field>
        </>
      )}
      <Field label="Direction" inline={false}>
        <Segmented size="sm" ariaLabel="Direction" value={plan.direction} onChange={direction => setForm({ direction })} options={[{ value: 'cw', label: 'Clockwise' }, { value: 'ccw', label: 'Counter-clockwise' }]} />
      </Field>

      <div className="tt-turns">
        <div className="tt-turns-head">
          <span className="field-label">Turns</span>
          <span className="caption">{caps.tilt ? 'each at its own tilt — tilt shows the top and underside' : 'full turns, one after the other'}</span>
        </div>
        <ol className="tt-turn-list">
          {plan.rotations.map((r, i) => (
            <li key={i} className="tt-turn">
              <span className="index-bubble">{i + 1}</span>
              <span className="grow">{caps.tilt ? (r.tilt_deg === 0 ? 'Level' : r.tilt_deg > 0 ? 'Tilted' : 'Tilted back') : 'Full turn'}</span>
              {caps.tilt && (
                <div style={{ width: 104 }}>
                  <NumberInput value={r.tilt_deg} step={1} min={tLo} max={tHi} unit="°" label={`Tilt of turn ${i + 1}`} onChange={v => setForm({ rotations: form.rotations.map((x, j) => (j === i ? { tilt_deg: Math.round(v) } : x)) })} />
                </div>
              )}
              <IconButton size="sm" label={`Remove turn ${i + 1}`} disabled={plan.rotations.length === 1} onClick={() => setForm({ rotations: form.rotations.filter((_, j) => j !== i) })}>
                <X size={15} />
              </IconButton>
            </li>
          ))}
        </ol>
        {plan.rotations.length < maxTurns && (
          <Button size="sm" variant="ghost" icon={<Plus size={15} />} onClick={() => setForm({ rotations: [...plan.rotations, { tilt_deg: caps.tilt ? clamp(plan.rotations.length === 1 ? 20 : -plan.rotations[plan.rotations.length - 1].tilt_deg || 20, tLo, tHi) : 0 }] })}>
            Add a turn{caps.tilt ? ' at another tilt' : ''}
          </Button>
        )}
      </div>

      <Field label="Start and pause scanning with the turntable" help={sweeps ? 'Scanning starts with the table and pauses only while it tilts between turns.' : 'Scanning pauses while the platter moves and records at every stop — Revo Metro’s turntable sync.'}>
        <Switch checked={plan.sync_scan} onChange={sync_scan => setForm({ sync_scan })} label="Start and pause scanning with the turntable" />
      </Field>
      {plan.sync_scan && !captureActive && <p className="caption tt-hint"><TriangleAlert size={13} aria-hidden /> Connect the scanner above first, so it can record at each stop.</p>}

      <Button variant="primary" block icon={<Play size={16} />} disabled={locked} loading={busy === 'program'} onClick={start}>
        Start: {plan.rotations.length} turn{plan.rotations.length > 1 ? 's' : ''}{sweep ? '' : ` · ${stops * plan.rotations.length} stops`} · about {fmtMinutes(seconds)}
      </Button>

      {confirm && (
        <ConfirmDialog
          title="Tilt while scanning?"
          icon={<TriangleAlert size={20} className="warn-glyph" />}
          onCancel={() => setConfirm(false)}
          choices={[
            { label: 'Start anyway', variant: 'secondary', onPick: () => { setConfirm(false); turntable.startProgram(plan); } },
            { label: 'Use turntable sync', variant: 'primary', onPick: () => { setConfirm(false); setForm({ sync_scan: true }); turntable.startProgram({ ...plan, sync_scan: true }); } },
          ]}
        >
          <p>This scan tilts the platter while the scanner keeps recording, so frames taken during the tilt can blur or lose tracking.</p>
          <p className="caption">With turntable sync the scanner pauses whenever the platter moves.</p>
        </ConfirmDialog>
      )}
    </section>
  );
}

function ProgramProgress({ status }: { status: TurntableStatus }) {
  const p = status.program!;
  const captureFrames = useCapture(s => s.status?.frames ?? 0);
  const phase = p.phase ? PHASE_LABEL[p.phase] ?? p.phase : 'Starting';
  return (
    <section className="tt-program is-running" aria-live="polite">
      <div className="tt-program-head row spread">
        <h4>
          <span className="dot live" aria-hidden /> Turntable scan
        </h4>
        <span className="mono tt-pct">{Math.round((p.progress ?? 0) * 100)} %</span>
      </div>
      <Progress value={p.progress ?? 0} />
      <div className="tt-program-now">
        <span>
          Turn <b className="mono">{p.rotation || 1}</b> of <span className="mono">{p.rotations}</span>
          {p.mode !== 'continuous' && (
            <>
              {' '}· stop <b className="mono">{Math.max(p.stop, 1)}</b> of <span className="mono">{p.stops_per_rotation}</span>
            </>
          )}
        </span>
        <span className="tt-phase">{phase}{p.tilt_deg ? ` · tilt ${fmtDeg(p.tilt_deg, 0, true)}` : ''}</span>
      </div>
      <div className="caption">
        {p.mode === 'continuous' && p.sync_scan ? (p.capture?.linked ? `Scanning while the table turns · ${captureFrames} frames` : 'Waiting for the scanner') : p.sync_scan ? (p.capture?.linked ? `${p.capture.frames} frames recorded at the stops so far` : 'Waiting for the scanner at each stop') : `Holding still at every stop; the scanner is not linked`}
      </div>
      {p.warnings?.map(w => (
        <p key={w} className="warn-text">
          <TriangleAlert size={13} aria-hidden /> <span>{w}</span>
        </p>
      ))}
      <Button variant="danger" block icon={<Square size={14} />} onClick={() => turntable.stopProgram()}>
        Stop the turntable scan
      </Button>
    </section>
  );
}

function ProgramResult({ status }: { status: TurntableStatus }) {
  const p = status.program!;
  const dismissedKey = `${p.started}|${p.finished}`;
  const [dismissed, setDismissed] = useState(() => sessionStorageGet('ttResultSeen') === dismissedKey);
  if (dismissed) return null;
  const Icon = p.state === 'done' ? CheckCircle2 : p.state === 'stopped' ? CircleSlash : XCircle;
  const title =
    p.state === 'done'
      ? `Done: ${p.rotations} turn${p.rotations > 1 ? 's' : ''}, ${p.completed_stops} stops`
      : p.state === 'stopped'
        ? `Stopped at turn ${p.rotation}, stop ${p.stop} of ${p.stops_per_rotation}`
        : `Stopped by a problem: ${p.error ?? 'unknown error'}`;
  return (
    <div className={`tt-result tone-${p.state === 'done' ? 'ok' : p.state === 'stopped' ? 'info' : 'danger'}`}>
      <Icon size={16} aria-hidden />
      <div className="grow">
        <div className="strong">{title}</div>
        <div className="caption">
          {p.elapsed_s ? `${fmtMinutes(p.elapsed_s)}` : ''}
          {p.capture?.linked ? ` · ${p.capture.frames} frames recorded` : ''}
          {p.state === 'done' && p.capture?.linked ? ' · the scan is paused — save it or scan more' : ''}
        </div>
        {p.warnings
          ?.filter(w => !/left paused/i.test(w))
          .slice(0, 2)
          .map(w => (
            <div key={w} className="caption">{w}</div>
          ))}
      </div>
      <IconButton size="sm" label="Dismiss" onClick={() => { sessionStorageSet('ttResultSeen', dismissedKey); setDismissed(true); }}>
        <X size={14} />
      </IconButton>
    </div>
  );
}

function sessionStorageGet(key: string): string | null {
  try {
    return sessionStorage.getItem(`cloudclean.${key}`);
  } catch {
    return null;
  }
}

function sessionStorageSet(key: string, value: string) {
  try {
    sessionStorage.setItem(`cloudclean.${key}`, value);
  } catch {
    /* private window */
  }
}
