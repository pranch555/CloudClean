import { useEffect, useState } from 'react';
import { ArrowRight, CheckCircle2, ChevronDown, CircleSlash, Link2, Loader2, RefreshCw, RotateCw, Square, TriangleAlert, Unplug, XCircle } from 'lucide-react';
import { fmtDeg, KIND_LABEL, programRunning, searchTurntables, turntable, useTurntable, useTurntablePolling, type TurntableDevice } from '../../lib/turntable';
import { useStore } from '../../store';
import { Button } from '../../ui/primitives';
import { loadDrivers, useCapture, type DriverInfo } from '../../features/capture/captureStore';
import { TiltGlyph, TurntableDial } from '../../features/capture/TurntableDial';
import { availability, driverMeta, sortDrivers, STATE_WORD } from '../../features/capture/vocabulary';

/** Settings → Scanner & turntable: what CloudClean can see and control, with a quick turntable test. */
export function DevicesSection() {
  return (
    <div className="stack loose cap-devices">
      <div className="settings-head">
        <h3>Scanner & turntable</h3>
        <p>What CloudClean can see and control on the machine it runs on. You connect the scanner and scan in step 1, Scan.</p>
      </div>
      <ScannerStatus />
      <TurntableStatusSection />
    </div>
  );
}

function goScan() {
  const st = useStore.getState();
  st.set({ settingsOpen: null });
  if (st.projectId) st.goStep('capture');
  else st.goHome();
}

function ScannerStatus() {
  const drivers = useCapture(s => s.drivers);
  const error = useCapture(s => s.driversError);
  const status = useCapture(s => s.status);
  const [loading, setLoading] = useState(!drivers.length);

  useEffect(() => {
    loadDrivers().finally(() => setLoading(false));
  }, []);

  const session = !!status?.active && status.state !== 'closed';
  const live = session ? driverMeta(drivers.find(d => d.id === status!.driver) ?? { id: status!.driver ?? '', name: status!.driver_name ?? 'Scanner', description: '' }) : null;
  const sorted = sortDrivers(drivers);

  return (
    <section className="stack">
      <div className="row spread">
        <h4 className="cap-settings-title">Scanner</h4>
        <Button size="sm" variant="ghost" icon={<RefreshCw size={14} />} loading={loading} onClick={() => { setLoading(true); loadDrivers().finally(() => setLoading(false)); }}>
          Check again
        </Button>
      </div>
      <div className={`status-line ${session ? 'ok' : 'off'}`}>
        <span className={`dot ${session ? (status!.state === 'running' ? 'live' : 'ok') : ''}`} aria-hidden />
        {session ? (
          <span>
            <b>{live!.title}</b> · {STATE_WORD[status!.state]?.toLowerCase() ?? status!.state}
          </span>
        ) : (
          <span>No scanner connected</span>
        )}
        <span className="spacer" />
        <Button size="sm" variant="secondary" onClick={goScan}>
          {session ? 'Open the Scan step' : 'Connect in the Scan step'} <ArrowRight size={14} aria-hidden />
        </Button>
      </div>
      {error && !drivers.length && (
        <p className="err-text">
          <XCircle size={14} aria-hidden /> <span>Cannot list the scanners: {error}</span>
        </p>
      )}
      {sorted.length > 0 && (
        <ul className="cap-driver-table" aria-label="Scanners">
          {sorted.map(d => (
            <DriverRow key={d.id} d={d} />
          ))}
        </ul>
      )}
    </section>
  );
}

function DriverRow({ d }: { d: DriverInfo }) {
  const meta = driverMeta(d);
  const a = availability(d);
  const Icon = meta.icon;
  const AIcon = a.tone === 'ok' ? CheckCircle2 : a.tone === 'warn' ? TriangleAlert : CircleSlash;
  const [open, setOpen] = useState(false);
  const found = d.devices?.[0];
  const details = d.reason || found;
  return (
    <li className="cap-driver-row">
      <span className="choice-icon">
        <Icon size={17} />
      </span>
      <div className="grow">
        <div className="row">
          <span className="strong">{meta.title}</span>
          <span className={`cap-avail tone-${a.tone}`}>
            <AIcon size={13} aria-hidden /> {a.word}
          </span>
        </div>
        <div className="caption">{meta.body}</div>
        {open && (
          <div className="cap-driver-detail">
            {found && (
              <div className="mono small">
                {found.product} · {found.hardware_id}
                {found.video_nodes?.length ? ` · ${found.video_nodes.join(', ')}` : ''}
              </div>
            )}
            {d.reason && <p className="small">{d.reason}</p>}
          </div>
        )}
      </div>
      {details && (
        <button type="button" className="link small" aria-expanded={open} onClick={() => setOpen(o => !o)}>
          {open ? 'Less' : a.tone === 'ok' ? 'Details' : 'Why?'}
        </button>
      )}
    </li>
  );
}

function TurntableStatusSection() {
  useTurntablePolling(true);
  const support = useTurntable(s => s.support);
  const status = useTurntable(s => s.status);
  const devices = useTurntable(s => s.devices);
  const searching = useTurntable(s => s.searching);
  const busy = useTurntable(s => s.busy);
  const error = useTurntable(s => s.error);

  useEffect(() => {
    if (support !== 'no' && devices == null && status && !status.connected) searchTurntables(4);
  }, [support, status?.connected]);

  if (support === 'no') {
    return (
      <section className="stack">
        <h4 className="cap-settings-title">Turntable</h4>
        <div className="status-line off">
          <span className="dot" aria-hidden /> Turntable control is not installed on the server yet.
        </div>
        <p className="hint-text">Update CloudClean on the DGX Spark to connect the Revopoint turntable over Bluetooth, turn and tilt it, and run turntable scans.</p>
      </section>
    );
  }
  if (!status) {
    return (
      <section className="stack">
        <h4 className="cap-settings-title">Turntable</h4>
        <div className="status-line off">
          <Loader2 size={15} className="spin" aria-hidden /> Checking…
        </div>
      </section>
    );
  }

  const running = programRunning(status);
  const locked = status.moving || running || !!busy;
  const noBluetooth = status.bluetooth && status.bluetooth.installed === false;

  return (
    <section className="stack">
      <h4 className="cap-settings-title">Turntable</h4>
      <div className={`status-line ${status.connected ? 'ok' : 'off'}`}>
        <span className={`dot ${status.connected ? (status.moving ? 'live' : 'ok') : ''}`} aria-hidden />
        {status.connected ? (
          <span>
            <b>{status.name ?? 'Turntable'}</b> · {KIND_LABEL[status.kind ?? ''] ?? status.kind}
            {status.firmware ? <span className="muted"> · firmware {status.firmware}</span> : null}
          </span>
        ) : (
          <span>Not connected</span>
        )}
      </div>

      {status.connected ? (
        <div className="cap-devices-tt">
          <TurntableDial status={status} size={128} />
          <div className="stack tight grow">
            <dl className="tt-facts">
              {status.capabilities?.tilt && (
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
            </dl>
            {!status.validated && (
              <p className="warn-text">
                <TriangleAlert size={14} aria-hidden /> <span>Control protocol not yet confirmed on your turntable — try a small rotation first.</span>
              </p>
            )}
            <div className="row wrap">
              <Button variant="primary" icon={<RotateCw size={15} />} disabled={locked} loading={busy === 'rotate'} onClick={() => turntable.rotate(10)}>
                Test: rotate 10°
              </Button>
              <Button variant="danger" icon={<Square size={13} />} onClick={() => turntable.stop()}>
                Stop
              </Button>
              <Button variant="ghost" icon={<Unplug size={15} />} loading={busy === 'disconnect'} onClick={() => turntable.disconnect()}>
                Disconnect
              </Button>
            </div>
          </div>
        </div>
      ) : (
        <div className="stack tight">
          <ul className="cap-driver-table" aria-label="Turntables nearby">
            {(devices ?? [{ id: 'simulated', name: 'Simulated dual-axis turntable', kind: 'simulated', rssi: null } as TurntableDevice]).map(d => (
              <li key={d.id} className="cap-driver-row">
                <span className="choice-icon">
                  <RotateCw size={17} />
                </span>
                <div className="grow">
                  <div className="row">
                    <span className="strong">{d.name ?? d.id}</span>
                    {d.remembered && <span className="badge">last used</span>}
                  </div>
                  <div className="caption">
                    {KIND_LABEL[String(d.kind)] ?? 'Turntable'}
                    {d.rssi != null ? ` · signal ${d.rssi} dBm` : ''}
                    {d.kind === 'simulated' ? ' · works without hardware' : ''}
                  </div>
                </div>
                <Button size="sm" variant={d.kind === 'simulated' ? 'secondary' : 'primary'} icon={<Link2 size={14} />} loading={busy === 'connect'} onClick={() => turntable.connect(d.id, String(d.kind ?? 'auto'))}>
                  Connect
                </Button>
              </li>
            ))}
          </ul>
          <div className="row">
            <Button size="sm" variant="ghost" icon={searching ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />} disabled={searching} onClick={() => searchTurntables(4)}>
              {searching ? 'Looking for turntables nearby…' : 'Search again'}
            </Button>
          </div>
          {noBluetooth && <p className="caption">Bluetooth support is not installed on the server, so only the simulated turntable is listed.</p>}
        </div>
      )}

      {(error || status.error) && (
        <p className="err-text" role="alert">
          <XCircle size={14} aria-hidden /> <span>{error ?? status.error}</span>
        </p>
      )}

      {status.log && status.log.length > 0 && (
        <details className="disclosure">
          <summary>
            Activity <span className="sub">the last {status.log.length} turntable messages</span>
            <ChevronDown size={16} className="chev" aria-hidden />
          </summary>
          <div className="disclosure-body">
            <pre className="cap-log mono">{status.log.join('\n')}</pre>
          </div>
        </details>
      )}
    </section>
  );
}
