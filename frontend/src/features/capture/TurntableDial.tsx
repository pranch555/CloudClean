import { useLayoutEffect, useRef } from 'react';
import { fmtAngle, fmtDeg, programRunning, type TurntableStatus } from '../../lib/turntable';

/*
 * The turntable seen from above, drawn like an instrument bezel: a hairline rim with degree ticks, the platter with
 * its index mark, and a vermilion pointer at the current angle (0° at the top, + = clockwise, as the table counts).
 * While it turns, the pointer glides between polls at the table's speed, a trail shows the way it came and a ghost
 * pointer shows where it will stop. During a program the stops are dots on the rim: done, current, to come.
 */

const C = 60;
const RIM = 56; // hairline rim; ticks run inward from it
const PLATTER = 37; // the platter disc
const BAND = 43.5; // free band between platter and ticks: pointer, trail, ghost
const LABEL = 66; // graduation numbers, outside the rim
const STOPS = 51.5; // program stops sit on the tick band

const rad = (d: number) => (d * Math.PI) / 180;
const polar = (deg: number, r: number): [number, number] => [C + r * Math.sin(rad(deg)), C - r * Math.cos(rad(deg))];
const f = (n: number) => n.toFixed(2);
// pointer head: apex just outside the platter, base against the ticks
const HEAD = `M${C} ${C - PLATTER - 1.5}l-4.4 8.5h8.8z`;

/** SVG arc along a circle of radius r from `from` to `to` (degrees, either direction, up to a full turn). */
function arcPath(from: number, to: number, r: number): string {
  let span = to - from;
  if (Math.abs(span) < 0.05) return '';
  if (Math.abs(span) >= 359.9) span = Math.sign(span) * 359.9;
  const [x0, y0] = polar(from, r);
  const [x1, y1] = polar(from + span, r);
  return `M${f(x0)} ${f(y0)}A${r} ${r} 0 ${Math.abs(span) > 180 ? 1 : 0} ${span > 0 ? 1 : 0} ${f(x1)} ${f(y1)}`;
}

interface Motion {
  angle: number; // last polled angle (cumulative)
  at: number; // performance.now() of that poll
  rate: number; // deg/s, signed; 0 when still
  target: number | null;
  start: number | null;
}

const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/**
 * Follows the polled angle and, while the platter turns, extrapolates it frame by frame at the table's speed (clamped
 * at the target, and to 1.5 s past the last poll so a stalled poll never spins the dial on). `draw` gets the angle.
 */
function useGlidingAngle(status: TurntableStatus | null, draw: (angle: number, m: Motion) => void) {
  const motion = useRef<Motion>({ angle: 0, at: 0, rate: 0, target: null, start: null });
  const idleAngle = useRef<number | null>(null);
  const moveKey = useRef<string | null>(null);
  const prevAngle = useRef<number | null>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;

  const angle = status?.connected ? status.angle_deg : null;
  const rotating = !!status?.moving && (status.device_state?.rotating !== undefined ? !!status.device_state.rotating : status.last_move?.type !== 'tilt');
  const move = status?.last_move;
  const speed = status?.speed_s_per_rev ?? status?.program?.speed_s_per_rev ?? null;
  const progDir = status?.program?.direction;

  // (layout effects, so the motion is updated before the draw loop below runs)
  useLayoutEffect(() => {
    if (angle == null) return;
    const m = motion.current;
    if (!rotating) {
      idleAngle.current = angle;
      m.rate = 0;
      m.target = null;
      m.start = null;
    } else {
      // direction: what the angle did since the last poll, else what the command asked for
      let sign = 0;
      if (prevAngle.current != null && Math.abs(angle - prevAngle.current) > 0.05) sign = Math.sign(angle - prevAngle.current);
      else if (move?.type === 'rotate' && move.state === 'running') sign = Math.sign(move.commanded_deg);
      else if (progDir) sign = progDir === 'ccw' ? -1 : 1;
      m.rate = speed ? (sign * 360) / speed : 0;
      const key = move?.type === 'rotate' && move.state === 'running' ? `${move.started}|${move.commanded_deg}` : null;
      if (key && key !== moveKey.current) {
        moveKey.current = key;
        m.start = idleAngle.current ?? angle;
      }
      if (m.start == null) m.start = idleAngle.current ?? angle;
      m.target = key ? m.start + move!.commanded_deg : null;
    }
    m.angle = angle;
    m.at = performance.now();
    prevAngle.current = angle;
  }, [angle, rotating, move?.started, move?.state, move?.commanded_deg, speed, progDir]);

  // drawn before paint, then every frame while the platter turns
  useLayoutEffect(() => {
    let raf = 0;
    const still = reducedMotion();
    const frame = () => {
      const m = motion.current;
      let a = m.angle;
      if (m.rate && !still) {
        a = m.angle + m.rate * Math.min((performance.now() - m.at) / 1000, 1.5);
        if (m.target != null) a = m.rate > 0 ? Math.min(a, m.target) : Math.max(a, m.target);
      }
      drawRef.current(a, m);
      if (m.rate && !still) raf = requestAnimationFrame(frame);
    };
    frame();
    return () => cancelAnimationFrame(raf);
  }, [angle, rotating]);
}

type StopDot = { angle: number; state: 'done' | 'current' | 'todo' };

/** Where the program's stops are on the rim (they repeat every turn), given the angle the program started at. */
function programStops(status: TurntableStatus, start: number): StopDot[] {
  const p = status.program!;
  const sign = p.direction === 'ccw' ? -1 : 1;
  const doneInTurn = p.completed_stops - Math.max(0, p.rotation - 1) * p.stops_per_rotation;
  let acc = 0;
  return p.moves.map((mv, i) => {
    acc += mv;
    const k = i + 1;
    const state: StopDot['state'] = k <= doneInTurn ? 'done' : k === p.stop ? 'current' : 'todo';
    return { angle: start + sign * acc, state };
  });
}

/**
 * `compact` drops the degree labels and the value in the middle (for the small viewport widget, which shows the
 * value beside the dial).
 */
export function TurntableDial({ status, size = 156, compact = false, className = '' }: { status: TurntableStatus | null; size?: number; compact?: boolean; className?: string }) {
  const pointer = useRef<SVGGElement>(null);
  const trail = useRef<SVGPathElement>(null);
  const ghostArc = useRef<SVGPathElement>(null);
  const ghost = useRef<SVGPathElement>(null);
  const value = useRef<SVGTextElement>(null);
  const progStart = useRef<{ key: string; angle: number } | null>(null);
  const connected = !!status?.connected && status.angle_deg != null;

  useGlidingAngle(status, (a, m) => {
    pointer.current?.setAttribute('transform', `rotate(${f(a)} ${C} ${C})`);
    if (value.current) value.current.textContent = connected ? fmtAngle(a) : '–';
    const moving = m.rate !== 0;
    trail.current?.setAttribute('d', moving && m.start != null ? arcPath(m.start, a, BAND) : '');
    ghostArc.current?.setAttribute('d', moving && m.target != null ? arcPath(a, m.target, BAND) : '');
    if (ghost.current) {
      ghost.current.style.display = moving && m.target != null ? '' : 'none';
      if (m.target != null) ghost.current.setAttribute('transform', `rotate(${f(m.target)} ${C} ${C})`);
    }
  });

  // the angle a running program started from: `turned_deg` counts finished moves only, so it is re-derived while the
  // platter stands still and kept while it turns
  let stops: StopDot[] = [];
  const p = status?.program;
  if (connected && p && programRunning(status) && p.mode !== 'continuous' && p.moves?.length) {
    const key = `${p.started ?? ''}|${p.interval_deg}|${p.rotations}`;
    if (!status!.moving || progStart.current?.key !== key) {
      progStart.current = { key, angle: status!.angle_deg! - (p.direction === 'ccw' ? -1 : 1) * p.turned_deg };
    }
    stops = programStops(status!, progStart.current.angle);
  }

  const ticks = [];
  for (let d = 0; d < 360; d += compact ? 30 : 10) {
    const major = d % 90 === 0 ? 2 : d % 30 === 0 ? 1 : 0;
    const [x0, y0] = polar(d, RIM);
    const [x1, y1] = polar(d, RIM - (major === 2 ? 8 : major === 1 ? 5.5 : 3.5));
    ticks.push(<line key={d} x1={f(x0)} y1={f(y0)} x2={f(x1)} y2={f(y1)} className={`tt-tick t${major}`} />);
  }
  const moving = !!status?.moving;
  const tilt = status?.capabilities?.tilt ? status.tilt_deg : null;
  const angleText = connected ? fmtAngle(status!.angle_deg) : '–';
  const label = connected
    ? `Turntable at ${angleText.replace('°', ' degrees')}${tilt != null ? `, tilted ${fmtDeg(tilt, 0, true).replace('°', ' degrees')}` : ''}${moving ? ', moving' : ''}`
    : 'Turntable not connected';

  return (
    <svg className={`tt-dial ${compact ? 'is-compact' : ''} ${moving ? 'is-moving' : ''} ${connected ? '' : 'is-off'} ${className}`} width={compact ? size : Math.round((size * 160) / 144)} height={size} viewBox={compact ? '2 2 116 116' : '-20 -12 160 144'} role="img" aria-label={label}>
      <circle cx={C} cy={C} r={RIM} className="tt-rim" />
      {ticks}
      {!compact &&
        [0, 90, 180, 270].map(d => {
          const [x, y] = polar(d, LABEL);
          return (
            <text key={d} x={f(x)} y={f(y)} className="tt-label" textAnchor="middle" dominantBaseline="central">
              {d}°
            </text>
          );
        })}
      <circle cx={C} cy={C} r={PLATTER} className="tt-platter" />
      {!compact && <circle cx={C} cy={C} r={PLATTER - 7} className="tt-platter-ring" />}
      <path ref={trail} className="tt-trail" d="" />
      <path ref={ghostArc} className="tt-ghost-arc" d="" />
      {stops.map((s, i) => {
        const [x, y] = polar(s.angle, compact ? RIM - 4 : STOPS);
        return <circle key={i} cx={f(x)} cy={f(y)} r={s.state === 'current' ? 3.4 : 2.5} className={`tt-stop-dot is-${s.state}`} />;
      })}
      <path ref={ghost} d={HEAD} className="tt-ghost" style={{ display: 'none' }} />
      {connected && (
        <g ref={pointer} className="tt-pointer" transform={`rotate(${f(status!.angle_deg!)} ${C} ${C})`}>
          {compact ? (
            <line x1={C} y1={C} x2={C} y2={C - RIM + 6} className="tt-needle" />
          ) : (
            <line x1={C} y1={C - PLATTER + 2} x2={C} y2={C - PLATTER + 11} className="tt-index" />
          )}
          <path d={HEAD} className="tt-pointer-head" />
        </g>
      )}
      {compact && <circle cx={C} cy={C} r={6} className="tt-hub" />}
      {!compact && (
        <>
          {/* written by the draw loop, not by React */}
          <text ref={value} x={C} y={C - 4} className="tt-value" textAnchor="middle" dominantBaseline="central" />
          <text x={C} y={C + 13} className="tt-caption" textAnchor="middle" dominantBaseline="central">
            {connected ? (moving ? 'turning' : 'angle') : 'not connected'}
          </text>
        </>
      )}
    </svg>
  );
}

/** Side view of the platter tilted by `tilt` degrees (+ = right side up) with a small dimension arc. */
export function TiltGlyph({ tilt, size = 44 }: { tilt: number | null; size?: number }) {
  const t = Math.max(-35, Math.min(35, tilt ?? 0));
  const cx = 22;
  const cy = 15;
  const phi = rad(t);
  const L = 15;
  const R = 10;
  const [x0, y0] = [cx - L * Math.cos(phi), cy + L * Math.sin(phi)];
  const [x1, y1] = [cx + L * Math.cos(phi), cy - L * Math.sin(phi)];
  const [ax, ay] = [cx + R * Math.cos(phi), cy - R * Math.sin(phi)];
  return (
    <svg className="tt-tilt-glyph" width={size} height={(size * 28) / 44} viewBox="0 0 44 28" aria-hidden>
      <path d="M9 26.5h26" className="tt-tilt-base" />
      <path d={`M${cx} 26.5V${cy}`} className="tt-tilt-post" />
      <path d={`M${cx - 17} ${cy}H${cx + 17}`} className="tt-tilt-ref" />
      {Math.abs(t) > 0.4 && <path d={`M${cx + R} ${cy}A${R} ${R} 0 0 ${t > 0 ? 0 : 1} ${f(ax)} ${f(ay)}`} className="tt-tilt-arc" />}
      <path d={`M${f(x0)} ${f(y0)}L${f(x1)} ${f(y1)}`} className="tt-tilt-platter" />
      <circle cx={cx} cy={cy} r={1.7} className="tt-tilt-pivot" />
    </svg>
  );
}
