import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import { CheckCircle2, ChevronDown, Info, PanelRightOpen, Square, TriangleAlert, XCircle } from 'lucide-react';
import * as THREE from 'three';
import { cssVar } from '../../lib/theme';
import { fmtAngle, fmtDeg, PHASE_LABEL, programRunning, turntable, useTurntable, useTurntablePolling } from '../../lib/turntable';
import { local, useStore, panelShown } from '../../store';
import { ScanGlyph } from '../../ui/icons';
import { Button, Empty, IconButton } from '../../ui/primitives';
import { getViewer } from '../../viewer/instance';
import { applyLiveStyle, loadDrivers, loadPending, openStream, useCapture, type Guidance, type Hole } from './captureStore';
import { TurntableDial } from './TurntableDial';
import { fmtClock, HOLE_TOKEN, HOLE_WORD, STATE_WORD, TRACKING_HELP, TRACKING_WORD, type Tone } from './vocabulary';

/*
 * The live capture HUD over the 3D view (Scan step only):
 *   top centre   — what to do now, as a pill ("Move closer", "Tracking lost — go back to scanned surface");
 *   bottom left  — the instrument cluster: completeness ring, tracking, distance, speed, density;
 *   on the model — numbered markers and labels on the gaps still to scan;
 *   bottom right — the turntable dial while a turntable is connected (with Stop).
 * Only the widgets themselves take the pointer; everything around them lets camera drags through.
 * Status colours always come with an icon and a word.
 */

const ICON = { ok: CheckCircle2, info: Info, warning: TriangleAlert, error: XCircle };
const START_UP_MS = 12000;
const TONE_ICON: Record<Tone, typeof Info> = { ok: CheckCircle2, warn: TriangleAlert, danger: XCircle, off: Info };

export function GuidanceHud() {
  const step = useStore(s => s.step);
  const screen = useStore(s => s.screen);
  const capture = screen === 'workspace' && step === 'capture';
  const status = useCapture(s => s.status);
  const active = capture && !!status?.active && status.state !== 'closed';
  const hasGuidance = useCapture(s => !!s.guidance);

  useCaptureMode(capture);

  if (!capture) return null;
  return (
    <>
      <HoleLayer active={active} />
      {active ? (
        <>
          <StatusPill />
          {hasGuidance && <Instruments />}
        </>
      ) : (
        <EmptyStage />
      )}
      <TurntableWidget />
    </>
  );
}

/** Entering the Scan step: live stream, pending scans, Z up, turntable status. Leaving it: show the saved scan. */
function useCaptureMode(capture: boolean) {
  const was = useRef(capture);
  const theme = useStore(s => s.theme);
  const session = useCapture(s => (s.status?.active && s.status.state !== 'closed' ? s.status.session_id : null));
  const hiddenFor = useRef<string | null>(null);
  useTurntablePolling(capture);

  // back in a capture session after a reload: the app's start-up puts the newest model on stage, over the live cloud
  // that already shows it — clear the stage once, as connecting does. Only during start-up, so a model the user
  // shows later is never hidden again (the eye brings models back).
  const shown = useStore(s => s.visible.length);
  useEffect(() => {
    if (!capture || !session || hiddenFor.current === session) return;
    if (performance.now() > START_UP_MS) {
      hiddenFor.current = session;
      return;
    }
    if (shown) {
      hiddenFor.current = session;
      useStore.setState({ visible: [] });
    }
  }, [capture, session, shown]);

  // the density colours follow the stage (light or dark); the theme attribute is set after this render
  useEffect(() => {
    if (!capture) return;
    const raf = requestAnimationFrame(() => applyLiveStyle());
    return () => cancelAnimationFrame(raf);
  }, [theme, capture]);

  useEffect(() => {
    if (capture) {
      openStream();
      loadPending();
      if (!useCapture.getState().drivers.length) loadDrivers();
      if (useStore.getState().display.upAxis !== 'z') useStore.getState().setDisplay({ upAxis: 'z' });
      applyLiveStyle();
    } else if (was.current) {
      // leaving the Scan step with a saved capture and nothing on stage: put the saved scan there
      const saved = useCapture.getState().status?.saved_asset_id;
      const st = useStore.getState();
      if (saved && st.byId.has(saved) && st.visible.length === 0) useStore.setState({ visible: [saved], activeId: st.activeId ?? saved });
    }
    was.current = capture;
  }, [capture]);
}

interface Placement {
  bottom: number;
  inset: number;
  /** measured at least once (hidden until then) */
  placed: boolean;
  /** later moves glide; the first placement is instant */
  glide: boolean;
}

/**
 * Where a bottom-corner widget can sit: above whatever is anchored at the bottom of the view that it would meet (the
 * Ask bar, and the banners that move down there on narrower screens), and out past the side rail when rising brings
 * it alongside. Re-measured whenever any of them changes size or the banners come and go.
 */
function useHudPlacement(ref: RefObject<HTMLElement | null>, side: 'left' | 'right', shown = true): Placement {
  const [place, setPlace] = useState<Omit<Placement, 'glide'>>({ bottom: 16, inset: 14, placed: false });
  const [glide, setGlide] = useState(false);
  useEffect(() => {
    if (!place.placed || glide) return;
    const raf = requestAnimationFrame(() => setGlide(true));
    return () => cancelAnimationFrame(raf);
  }, [place.placed, glide]);
  useLayoutEffect(() => {
    const el = ref.current;
    const host = el?.closest('.viewport') as HTMLElement | null;
    if (!el || !host) return;
    const gap = 10;
    const measure = () => {
      const h = host.getBoundingClientRect();
      const w = el.offsetWidth;
      const ht = el.offsetHeight;
      const lows = [host.querySelector('.ask-bar'), ...host.querySelectorAll('.banner-stack > *')]
        .filter((e): e is Element => !!e)
        .map(e => e.getBoundingClientRect())
        .filter(r => r.width > 0 && r.top > h.top + h.height / 2);
      const span = (inset: number) => (side === 'left' ? [h.left + inset, h.left + inset + w] : [h.right - inset - w, h.right - inset]);
      const lift = (inset: number) => {
        const [x0, x1] = span(inset);
        let b = 16;
        for (const r of lows) if (x0 < r.right + gap && x1 > r.left - gap) b = Math.max(b, Math.round(h.bottom - r.top) + gap);
        return b;
      };
      let inset = 14;
      let bottom = lift(inset);
      const rail = host.querySelector(side === 'left' ? '.tool-rail' : '.view-rail')?.getBoundingClientRect();
      if (rail && rail.width > 0) {
        const [x0, x1] = span(inset);
        if (h.bottom - bottom - ht < rail.bottom + gap && x0 < rail.right + gap && x1 > rail.left - gap) {
          inset = side === 'left' ? Math.round(rail.right - h.left) + gap : Math.round(h.right - rail.left) + gap;
          bottom = lift(inset);
        }
      }
      setPlace(p => (p.placed && p.bottom === bottom && p.inset === inset ? p : { bottom, inset, placed: true }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(host);
    ro.observe(el);
    const ask = host.querySelector('.ask-bar');
    if (ask) ro.observe(ask);
    const stack = host.querySelector('.banner-stack');
    const mo = new MutationObserver(measure);
    if (stack) {
      ro.observe(stack);
      mo.observe(stack, { childList: true, subtree: true });
    }
    window.addEventListener('resize', measure);
    return () => {
      ro.disconnect();
      mo.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [ref, side, shown]);
  return { ...place, glide };
}

/** True while the 3D view holding `ref` is narrower than `width` px. */
function useNarrowView(ref: RefObject<HTMLElement | null>, width: number): boolean {
  const [narrow, setNarrow] = useState(false);
  useLayoutEffect(() => {
    const host = ref.current?.closest('.viewport') as HTMLElement | null;
    if (!host) return;
    const ro = new ResizeObserver(() => setNarrow(host.clientWidth < width));
    ro.observe(host);
    setNarrow(host.clientWidth < width);
    return () => ro.disconnect();
  }, [ref, width]);
  return narrow;
}

/* ------------------------------------------------------------------ top centre */

function StatusPill() {
  const g = useCapture(s => s.guidance);
  const state = useCapture(s => s.status?.state ?? 'idle');
  const fallback = state === 'connected' ? 'Ready — press Start scanning' : state === 'paused' ? 'Paused — press Resume to go on' : STATE_WORD[state] ?? state;
  const sev = g?.status?.severity ?? (state === 'error' ? 'error' : 'info');
  const Icon = ICON[sev] ?? Info;
  const message = g?.status?.message ?? fallback;
  return (
    <div className={`hud banner cap-status sev-${sev}`} role="status" aria-live="polite">
      <Icon size={17} aria-hidden className="cap-status-icon" />
      <span className="cap-status-text">{message}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ bottom left: instruments */

function Instruments() {
  const g = useCapture(s => s.guidance);
  const status = useCapture(s => s.status)!;
  const streaming = useCapture(s => s.drivers.find(d => d.id === s.status?.driver)?.capabilities?.streaming !== false);
  const ref = useRef<HTMLDivElement>(null);
  const place = useHudPlacement(ref, 'left');
  const tracking = g?.tracking?.state ?? status.tracking ?? 'n/a';
  const trackTone: Tone = tracking === 'lost' ? 'danger' : tracking === 'weak' ? 'warn' : tracking === 'n/a' || tracking === 'none' ? 'off' : 'ok';
  const speed = g?.speed;
  const density = g?.density?.dense_ratio ?? null;
  const complete = g?.coverage?.completeness ?? null;
  const [savedFolded, setFolded] = useState(() => local.get('captureGaugesFolded', false));
  const narrow = useNarrowView(ref, 600);
  const [peek, setPeek] = useState(false);
  // a narrow 3D view has no room for the full cluster: it stays folded unless opened for a look
  const folded = narrow ? !peek : savedFolded;
  const fold = () => {
    if (narrow) return setPeek(p => !p);
    setFolded(f => {
      local.set('captureGaugesFolded', !f);
      return !f;
    });
  };

  return (
    <div ref={ref} className={`hud cap-gauges ${folded ? 'is-folded' : ''} ${place.glide ? 'is-placed' : ''}`} style={{ bottom: place.bottom, left: place.inset, visibility: place.placed ? undefined : 'hidden' }} role="group" aria-label="Scan quality">
      <button type="button" className={`cap-gauges-head is-${status.state}`} aria-expanded={!folded} title={folded ? 'Show the gauges' : 'Fold the gauges away'} onClick={fold}>
        <span className="cap-gauges-tally" aria-hidden />
        <span className="cap-gauges-state">{STATE_WORD[status.state] ?? status.state}</span>
        {folded && complete != null && <span className="cap-gauges-pct mono">{Math.round(complete * 100)} %</span>}
        <span className="cap-gauges-clock mono">{fmtClock(status.elapsed_s)}</span>
        <ChevronDown size={15} aria-hidden className="cap-gauges-chev" />
      </button>
      {!folded && <CompletenessRing value={complete} />}
      {!folded && <div className="cap-gauge-rows">
        <GaugeRow label="Tracking" tone={trackTone} value={TRACKING_WORD[tracking] ?? tracking} word title={TRACKING_HELP[tracking]} />
        {streaming && <DistanceRow d={g?.distance} />}
        {streaming && (
          <GaugeRow
            label="Speed"
            tone={speed?.too_fast ? 'danger' : speed?.value_mm_s != null ? 'ok' : 'off'}
            value={speed?.value_mm_s != null ? `${speed.value_mm_s.toFixed(0)}` : '–'}
            unit={speed?.value_mm_s != null ? 'mm/s' : undefined}
            bar={speed?.value_mm_s != null && speed.limit_mm_s ? speed.value_mm_s / speed.limit_mm_s : undefined}
            note={speed?.too_fast ? 'too fast' : undefined}
            title={speed?.limit_mm_s ? `Frames faster than ${speed.limit_mm_s} mm/s are left out` : undefined}
          />
        )}
        <GaugeRow
          label="Density"
          tone={density == null ? 'off' : density < 0.6 ? 'warn' : 'ok'}
          value={density != null ? `${Math.round(density * 100)}` : '–'}
          unit={density != null ? '%' : undefined}
          bar={density ?? undefined}
          title="Share of the scanned surface that reached the target point density"
        />
      </div>}
    </div>
  );
}

function CompletenessRing({ value }: { value: number | null }) {
  const r = 22;
  const len = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value ?? 0));
  const done = value != null && value >= 0.9;
  const tick = 0.9 * 360 - 90;
  const tx = 28 + r * Math.cos((tick * Math.PI) / 180);
  const ty = 28 + r * Math.sin((tick * Math.PI) / 180);
  return (
    <div className={`cap-ring ${done ? 'is-done' : ''}`} title="Share of the part's surface scanned so far (target 90 %)">
      <svg viewBox="0 0 56 56" aria-hidden>
        <circle cx="28" cy="28" r={r} className="cap-ring-track" />
        <circle cx="28" cy="28" r={r} className="cap-ring-fill" style={{ strokeDasharray: `${(v * len).toFixed(1)} ${len.toFixed(1)}` }} transform="rotate(-90 28 28)" />
        <circle cx={tx.toFixed(2)} cy={ty.toFixed(2)} r="2.2" className="cap-ring-target" />
      </svg>
      <div className="cap-ring-value mono">{value != null ? Math.round(value * 100) : '–'}<small>%</small></div>
      <div className="cap-ring-label">{done ? <><CheckCircle2 size={11} aria-hidden /> complete</> : 'complete'}</div>
    </div>
  );
}

function GaugeRow({ label, tone, value, unit, bar, note, title, word }: { label: string; tone: Tone; value: string; unit?: string; bar?: number; note?: string; title?: string; word?: boolean }) {
  const Icon = TONE_ICON[tone];
  return (
    <div className={`cap-gauge tone-${tone}`} title={title}>
      <Icon size={13} aria-hidden className="cap-gauge-icon" />
      <span className="cap-gauge-label">{label}</span>
      <span className="cap-gauge-value">
        <span className={word ? 'cap-gauge-word' : 'mono'}>{value}</span>
        {unit && !note && <span className="cap-gauge-unit">{unit}</span>}
        {note && <span className="cap-gauge-note">{note}</span>}
      </span>
      {bar != null ? (
        <span className="cap-bar" aria-hidden>
          <span style={{ transform: `scaleX(${Math.max(0, Math.min(1, bar))})` }} />
        </span>
      ) : (
        <span aria-hidden />
      )}
    </div>
  );
}

function DistanceRow({ d }: { d?: Guidance['distance'] }) {
  const has = d?.value_mm != null && d.min_mm != null && d.max_mm != null;
  const out = d?.state === 'too_close' || d?.state === 'too_far';
  const tone: Tone = !has ? 'off' : out ? 'danger' : 'ok';
  const Icon = TONE_ICON[tone];
  // the bar shows the working range with some margin either side, the sweet spot and where the scanner is
  const lo = has ? d!.min_mm! - (d!.max_mm! - d!.min_mm!) * 0.2 : 0;
  const hi = has ? d!.max_mm! + (d!.max_mm! - d!.min_mm!) * 0.2 : 1;
  const at = (v: number) => `${(Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1))) * 100).toFixed(1)}%`;
  return (
    <div className={`cap-gauge tone-${tone}`} title={has ? `Working range ${d!.min_mm}–${d!.max_mm} mm${d!.optimal_mm ? `, best at ${d!.optimal_mm} mm` : ''}` : undefined}>
      <Icon size={13} aria-hidden className="cap-gauge-icon" />
      <span className="cap-gauge-label">Distance</span>
      <span className="cap-gauge-value">
        <span className="mono">{d?.value_mm != null ? d.value_mm.toFixed(0) : '–'}</span>
        {d?.value_mm != null && !out && <span className="cap-gauge-unit">mm</span>}
        {out && <span className="cap-gauge-note">{d!.state === 'too_close' ? 'too close' : 'too far'}</span>}
      </span>
      <span className="cap-range" aria-hidden>
        {has && <span className="cap-range-ok" style={{ left: at(d!.min_mm!), right: `calc(100% - ${at(d!.max_mm!)})` }} />}
        {has && d!.optimal_mm != null && <span className="cap-range-best" style={{ left: at(d!.optimal_mm) }} />}
        {has && <span className="cap-range-now" style={{ left: at(d!.value_mm!) }} />}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ gaps on the model */

const NO_HOLES: Hole[] = [];

function HoleLayer({ active }: { active: boolean }) {
  const all = useCapture(s => s.guidance?.holes) ?? NO_HOLES;
  const holes = active ? all.slice(0, 6) : NO_HOLES;
  const theme = useStore(s => s.theme);
  const layer = useRef<HTMLDivElement>(null);
  const key = holes.map(h => `${h.id}:${h.kind}:${h.center.join()}`).join('|');

  // 3D: a dot on each gap and a line out along the direction to scan it from (colours from the theme tokens)
  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    const raf = requestAnimationFrame(() => {
      const r = v.radius || 50;
      const color = (k: keyof typeof HOLE_TOKEN) => cssVar(HOLE_TOKEN[k], '#ee4b1f');
      v.setMarkers(
        holes.map((h, i) => ({ id: String(i), position: h.center, color: color(h.kind) })),
        holes.map(h => ({ from: h.center, to: new THREE.Vector3(...h.center).addScaledVector(new THREE.Vector3(...h.direction), r * 0.45).toArray() as [number, number, number], color: color(h.kind) })),
        'capture',
      );
    });
    return () => {
      cancelAnimationFrame(raf);
      v.setMarkers([], [], 'capture');
    };
  }, [key, theme]);

  // HTML labels glued to the gaps every frame; a label that would cover an earlier one steps down below it
  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    return v.subscribe(() => {
      const el = layer.current;
      if (!el) return;
      const hs = useCapture.getState().guidance?.holes ?? [];
      const placed: { x: number; y: number; w: number; h: number }[] = [];
      [...el.children].forEach((node, i) => {
        const n = node as HTMLElement;
        const h = hs[i];
        const s = h ? v.project(h.center) : null;
        n.style.display = s ? '' : 'none';
        if (!s) return;
        const w = n.offsetWidth;
        const ht = n.offsetHeight;
        const x = s.x + 10;
        let y = s.y - 12;
        for (let k = 0; k < placed.length; k++) {
          const hit = placed.find(p => x < p.x + p.w + 4 && x + w + 4 > p.x && y < p.y + p.h + 3 && y + ht + 3 > p.y);
          if (!hit) break;
          y = hit.y + hit.h + 4;
        }
        placed.push({ x, y, w, h: ht });
        n.style.transform = `translate(${x}px, ${y}px)`;
      });
    });
  }, []);

  return (
    <div ref={layer} className="labels-layer cap-hole-labels" aria-hidden>
      {holes.map((h, i) => (
        <div key={h.id} className={`hole-label kind-${h.kind}`}>
          <b>{i + 1}</b> {HOLE_WORD[h.kind]}
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ nothing connected */

function EmptyStage() {
  const visible = useStore(s => s.visible.length);
  const rightOpen = useStore(s => panelShown(s, 'right'));
  if (visible) return null;
  return (
    <div className="stage-empty">
      <Empty
        icon={<ScanGlyph size={26} />}
        title="Ready when you are"
        action={
          !rightOpen ? (
            <Button variant="primary" icon={<PanelRightOpen size={16} />} onClick={() => useStore.getState().setLayout({ rightOpen: true })}>
              Show the Scan panel
            </Button>
          ) : undefined
        }
      >
        Pick your scanner in the Scan panel and connect it. The part builds up here as you scan.
      </Empty>
    </div>
  );
}

/* ------------------------------------------------------------------ bottom right: turntable */

function TurntableWidget() {
  const status = useTurntable(s => s.status);
  const ref = useRef<HTMLDivElement>(null);
  const place = useHudPlacement(ref, 'right', !!status?.connected);
  if (!status?.connected) return null;
  const p = status.program;
  const running = programRunning(status);
  const tilt = status.capabilities?.tilt ? status.tilt_deg : null;
  return (
    <div ref={ref} className={`hud cap-tt ${status.moving || running ? 'is-live' : ''} ${place.glide ? 'is-placed' : ''}`} style={{ bottom: place.bottom, right: place.inset, visibility: place.placed ? undefined : 'hidden' }} role="group" aria-label="Turntable">
      <TurntableDial status={status} size={60} compact />
      <div className="cap-tt-text">
        <div className="cap-tt-top">
          <span className="cap-tt-angle mono">{fmtAngle(status.angle_deg)}</span>
          {tilt != null && <span className="cap-tt-tilt mono">tilt {fmtDeg(tilt, 0, true)}</span>}
        </div>
        <div className="cap-tt-sub">
          {running && p ? (
            <span>
              <b>{PHASE_LABEL[p.phase ?? ''] ?? 'Running'}</b> · {p.rotations > 1 ? `turn ${p.rotation}/${p.rotations} · ` : ''}stop {Math.max(1, p.stop)}/{p.stops_per_rotation}
            </span>
          ) : (
            <span className={status.moving ? 'is-live' : ''}>{status.moving ? (status.device_state?.tilting ? 'Tilting' : 'Turning') : 'Still'}</span>
          )}
        </div>
        {running && p && (
          <span className="cap-tt-progress" aria-hidden>
            <span style={{ transform: `scaleX(${Math.max(0, Math.min(1, p.progress ?? 0))})` }} />
          </span>
        )}
      </div>
      <IconButton size="sm" label="Stop the turntable" tip="top" className="cap-tt-stop" onClick={() => turntable.stop()}>
        <Square size={13} />
      </IconButton>
    </div>
  );
}
