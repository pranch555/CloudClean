import { useState } from 'react';
import { ChevronDown, TriangleAlert, XCircle, CheckCircle2, Ruler, Eraser } from 'lucide-react';
import { api } from '../../lib/api';
import { setTool } from '../../lib/actions';
import { fmtCount, fmtLen, fmtSigned } from '../../lib/format';
import { remeasureAll } from '../../lib/measure';
import { selectionRegion } from '../../lib/measureTools';
import type { Asset } from '../../lib/types';
import { useStore, type ThreadResult } from '../../store';
import { getViewer } from '../../viewer/instance';
import { Badge, Button, Stat } from '../../ui/primitives';
import { Block } from '../StepFrame';
import { ThreadGlyph } from './glyphs';
import { SelectStep, StepItem, StepList } from './parts';
import { setError, setTab, setUi, useMeasureUi } from './state';

export async function analyseThread(target: Asset) {
  const st = useStore.getState();
  const onTarget = (st.selectionCounts[target.id] ?? 0) > 0;
  setUi({ busy: 'thread' });
  setError('thread', null);
  try {
    const region = onTarget ? selectionRegion(false) ?? undefined : undefined;
    const res = await api.post<ThreadResult>('/api/measure/thread', { asset_id: target.id, region });
    useStore.setState({ thread: { ...res, asset_id: target.id } });
  } catch (err) {
    setError('thread', (err as Error).message);
  } finally {
    setUi({ busy: null });
  }
}

/** Pitch, diameters and the standard size of a screw or nut thread. */
export function ThreadTab({ target }: { target: Asset | undefined }) {
  const thread = useStore(s => s.thread);
  const error = useMeasureUi(s => s.errors.thread);
  const mine = thread && target && thread.asset_id === target.id ? thread : null;
  return (
    <>
      <Block title="Find the thread">
        <p className="hint-text">Finds the pitch, the diameters and the standard size (M10 × 1.5, 3/8-16 UNC …) of a screw or a nut.</p>
        <StepList>
          <SelectStep n={1} target={target}>Select only the threaded part — leave out the head and the chamfer. Skip this to use the whole scan.</SelectStep>
          <StepItem n={2}>Press <b>Analyse the thread</b>. It takes a few seconds.</StepItem>
        </StepList>
      </Block>
      {error && <p className="err-text" role="alert"><TriangleAlert size={14} aria-hidden /> <span>{error}</span></p>}
      {mine && <ThreadResultView t={mine} />}
      {thread && !mine && (
        <p className="hint-text">The last thread analysis was on {useStore.getState().byId.get(thread.asset_id)?.name ?? 'another model'}.</p>
      )}
    </>
  );
}

export function ThreadFooter({ target }: { target: Asset | undefined }) {
  const busy = useMeasureUi(s => s.busy);
  const onTarget = useStore(s => !!target && (s.selectionCounts[target.id] ?? 0) > 0);
  return (
    <Button variant="primary" size="lg" block icon={<ThreadGlyph size={18} />} loading={busy === 'thread'} disabled={!target || (!!busy && busy !== 'thread')} onClick={() => target && analyseThread(target)}>
      {!target ? 'Pick the scanned screw or nut' : onTarget ? 'Analyse the selected thread' : 'Analyse the thread'}
    </Button>
  );
}

function ThreadResultView({ t }: { t: ThreadResult }) {
  const units = useStore(s => s.display.units);
  const best = t.standards?.[0];
  // ISO 965: 6g is the usual class for external threads (bolts), 6H for internal ones (nuts)
  const tolClass = t.kind === 'internal' ? '6H' : t.kind === 'external' ? '6g' : null;
  const tol = tolClass ? best?.tolerance?.[tolClass] : undefined;
  const tone = best?.match === 'close' ? 'pass' : best?.match === 'possible' ? 'warn' : 'fail';
  const Icon = tone === 'pass' ? CheckCircle2 : tone === 'warn' ? TriangleAlert : XCircle;
  const kind = t.kind === 'internal' ? 'Internal thread (nut)' : t.kind === 'external' ? 'External thread (bolt)' : 'Thread';

  const alongAxis = () => {
    useStore.setState({ measureAxis: 'thread' });
    remeasureAll();
    setUi({ tool: 'p2p' });
    setTab('dimensions');
    setTool('measure');
  };
  const clear = () => {
    useStore.setState({ thread: null });
    getViewer()?.setMarkers([], [], 'thread');
  };

  return (
    <div className="meas-thread">
      <div className={`verdict verdict-${tone}`} role="status">
        <Icon size={22} aria-hidden />
        <div className="grow">
          <div className="verdict-title">{best ? best.designation : 'No standard size matches'}</div>
          <div className="verdict-sub">
            {kind} · {t.handedness}-hand · {t.crest_count} crests
            {best && best.match !== 'close' ? ` · ${best.match === 'possible' ? 'a possible match' : 'a poor match'}` : ''}
          </div>
        </div>
        <Badge tone={t.confidence === 'high' ? 'ok' : t.confidence === 'medium' ? 'warning' : 'danger'}>{t.confidence} confidence</Badge>
      </div>

      <div className="stat-grid">
        <Stat label="Pitch" value={fmtLen(t.pitch, 4)} unit={units} sub={`± ${fmtLen(t.pitch_se, 4)}${t.tpi ? ` · ${t.tpi.toFixed(1)} TPI` : ''}`} />
        <Stat label="Major Ø" value={fmtLen(t.major_diameter, 3)} unit={units} sub="over the crests" />
        <Stat label="Minor Ø" value={fmtLen(t.minor_diameter, 3)} unit={units} sub="at the roots" />
        <Stat label="Pitch Ø" value={fmtLen(t.pitch_diameter, 3)} unit={units} sub={t.pitch_diameter == null ? 'needs both flanks scanned' : 'mid-flank'} />
        <Stat label="Flank angle" value={t.flank_angle_deg != null ? `${t.flank_angle_deg.toFixed(1)}°` : '–'} sub={t.flank_angle_deg == null ? 'needs both flanks scanned' : '60° is standard'} />
        <Stat label="Axis" value={`±${t.axis_uncertainty_deg.toFixed(3)}°`} sub="uncertainty" />
      </div>

      {best && (
        <Block title="Against the standard">
          <div className="meas-table-wrap">
            <table className="data-table meas-table">
              <thead>
                <tr><th /><th>Scan</th><th>{best.designation}</th><th>Difference</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td>Pitch</td>
                  <td className="mono">{fmtLen(t.pitch, 4)}</td>
                  <td className="mono">{fmtLen(best.pitch, 4)}</td>
                  <td className="mono">{fmtSigned(best.pitch_deviation, 4)}</td>
                </tr>
                <tr>
                  <td>Major Ø</td>
                  <td className="mono">{fmtLen(t.major_diameter, 3)}</td>
                  <td className="mono">{fmtLen(best.major_diameter, 3)}</td>
                  <td className={`mono ${tol && tol.within === false ? 'delta-warn' : ''}`}>{fmtSigned(best.major_diameter_deviation, 3)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          {tol && (
            <p className={tol.within ? 'meas-ok-line' : 'warn-text'}>
              {tol.within ? <CheckCircle2 size={14} aria-hidden /> : <TriangleAlert size={14} aria-hidden />}
              <span>
                Major diameter {tol.within ? 'within' : 'outside'} ISO {tolClass} for {t.kind === 'internal' ? 'nuts' : 'bolts'}:{' '}
                <span className="mono">{fmtLen(tol.major_min, 3)} – {tol.major_max != null ? fmtLen(tol.major_max, 3) : 'no upper limit'}</span> {units}
              </span>
            </p>
          )}
          {t.standards.length > 1 && (
            <p className="caption">Also close: {t.standards.slice(1).map(s => `${s.designation} (${s.match})`).join(', ')}</p>
          )}
        </Block>
      )}

      {t.per_crest_pitches?.length > 1 && (
        <Block title="Pitch of each turn">
          <PitchChart pitches={t.per_crest_pitches} nominal={best?.pitch ?? t.pitch} units={units} />
        </Block>
      )}

      {t.warnings.map((w, i) => (
        <p key={i} className="warn-text"><TriangleAlert size={13} aria-hidden /> <span>{w}</span></p>
      ))}

      <div className="row wrap">
        <Button size="sm" icon={<Ruler size={14} />} onClick={alongAxis}>Measure along the thread axis</Button>
        <Button size="sm" variant="ghost" icon={<Eraser size={14} />} onClick={clear}>Clear</Button>
      </div>
      <p className="caption">
        <span className="mono">{fmtCount(t.points_used)}</span> points · scan noise <span className="mono">{fmtLen(t.noise_rms, 3)}</span> {units} ·{' '}
        <span className="mono">{Math.round(t.angular_coverage_deg)}°</span> of the circumference scanned · green dots on the model mark the crests
      </p>
    </div>
  );
}

/**
 * Pitch of every turn around the nominal pitch: one series (a 2 px line with dots ringed in the surface colour),
 * the nominal as a hairline, hover or focus on a turn for its value, and the numbers in a table underneath.
 */
function PitchChart({ pitches, nominal, units }: { pitches: number[]; nominal: number; units: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 320, H = 132, padL = 46, padR = 10, padT = 12, padB = 22;
  const dev = pitches.map(p => p - nominal);
  const raw = Math.max(...dev.map(Math.abs), nominal * 0.004, 0.002);
  const span = niceCeil(raw);
  const n = pitches.length;
  const x = (i: number) => padL + (n === 1 ? 0.5 : i / (n - 1)) * (W - padL - padR);
  const y = (d: number) => padT + (1 - (d + span) / (2 * span)) * (H - padT - padB);
  const slot = (W - padL - padR) / Math.max(n - 1, 1);
  const digits = span < 0.01 ? 4 : 3;
  const h = hover != null ? { i: hover, p: pitches[hover], d: dev[hover] } : null;

  return (
    <>
      <div className="pitch-chart">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Pitch of ${n} turns, from ${fmtLen(Math.min(...pitches), 4)} to ${fmtLen(Math.max(...pitches), 4)} ${units}`} onMouseLeave={() => setHover(null)}>
          {[span, 0, -span].map(t => (
            <g key={t}>
              <line className={t === 0 ? 'pc-nominal' : 'pc-grid'} x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} />
              <text className="pc-tick" x={padL - 6} y={y(t) + 3.5} textAnchor="end">{t === 0 ? fmtLen(nominal, 3) : fmtSigned(t, digits)}</text>
            </g>
          ))}
          <path className="pc-line" d={dev.map((d, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(d).toFixed(1)}`).join('')} />
          {dev.map((d, i) => (
            <g key={i} tabIndex={0} role="img" aria-label={`Turn ${i + 1}: ${fmtLen(pitches[i], 4)} ${units}`} onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} onBlur={() => setHover(null)}>
              <rect className="pc-hit" x={x(i) - slot / 2} y={padT} width={slot} height={H - padT - padB} />
              <circle className={`pc-dot ${hover === i ? 'is-on' : ''}`} cx={x(i)} cy={y(d)} r={hover === i ? 5.5 : 4} />
            </g>
          ))}
          <text className="pc-tick" x={padL} y={H - 6} textAnchor="start">turn 1</text>
          <text className="pc-tick" x={W - padR} y={H - 6} textAnchor="end">turn {n}</text>
        </svg>
        {h && (
          <div className="hist-tooltip mono" role="status">
            turn {h.i + 1} → {h.i + 2}
            <b>{fmtLen(h.p, 4)} {units} · {fmtSigned(h.d, 4)}</b>
          </div>
        )}
      </div>
      <details className="meas-values">
        <summary>Show the values <ChevronDown size={14} className="chev" aria-hidden /></summary>
        <div className="meas-table-wrap">
          <table className="data-table meas-table">
            <thead><tr><th>Turn</th><th>Pitch</th><th>Difference</th></tr></thead>
            <tbody>
              {pitches.map((p, i) => (
                <tr key={i}><td className="mono">{i + 1} → {i + 2}</td><td className="mono">{fmtLen(p, 4)}</td><td className="mono">{fmtSigned(dev[i], 4)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

/** 1, 2 or 5 times a power of ten, at least v. */
function niceCeil(v: number): number {
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}
