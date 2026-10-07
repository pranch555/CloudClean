import { useEffect, useState } from 'react';
import { create } from 'zustand';
import { api } from '../../lib/api';
import { local } from '../../store';
import { useCapture } from './captureStore';

/*
 * The scanner's camera view (Scan → Camera view, and the floating view over the 3D view): what its two IR cameras
 * see, live, and the exposure settings — surface preset, Auto / Manual, laser brightness, exposure, gain, marker light.
 * Server side: routes_capture "Camera view". Pictures are polled as JPEGs a few times a second, only while a view is
 * on screen; before scanning, a view on screen also keeps the camera preview running (laser on, nothing recorded).
 */

export type Surface = 'general' | 'dark' | 'reflective';
export type Cams = 'both' | 'left' | 'right';

export interface CameraSettings {
  surface: Surface;
  mode: 'auto' | 'manual';
  laser_level: number;
  level_max: number;
  laser_pct: number;
  laser_pulse: number;
  exposure_us: number;
  gain: number;
  marker_light: number;
}

export interface CameraReadout {
  frames?: number;
  lines_seen?: boolean;
  stripe_brightness?: number | null;
  saturated_pct?: number | null;
  points_per_frame?: number | null;
  depth_mm?: number | null;
  target?: number;
  band?: [number, number];
  saturation_max_pct?: number;
}

export interface CameraInfo {
  available: boolean;
  reason?: string;
  state?: string;
  driver?: string;
  settings?: CameraSettings;
  limits?: {
    laser_level: [number, number];
    exposure_us: [number, number];
    gain: [number, number];
    marker_light: [number, number];
    /** Revo Metro's starting values per surface */
    presets?: Record<Surface, { exposure_us: number; gain: number; laser_level: number; level_max: number; marker_light: number }>;
  };
  auto?: { state: 'off' | 'waiting' | 'adjusting' | 'steady' | 'limit'; message: string; verdict: Verdict };
  readout?: CameraReadout;
  streaming?: boolean;
  preview?: boolean;
  starting?: boolean;
  error?: string | null;
  gain_map?: string;
  /** driver-setting values that bring these settings back next time (POST only) */
  remembered?: Record<string, unknown>;
}

export type Verdict = 'none' | 'dim' | 'good' | 'bright';

/** The live numbers in every status message (status.device.camera). */
export interface CameraBrief {
  mode: string;
  surface: string;
  laser_pct: number;
  gain: number;
  verdict: Verdict;
  auto_state: string;
  auto_message: string;
  stripe_brightness: number | null;
  saturated_pct: number | null;
  points_per_frame: number | null;
  streaming: boolean;
  preview: boolean;
}

export { CAMERA_KEYS } from './captureStore';

interface CameraState {
  info: CameraInfo | null;
  cams: Cams;
  overlay: boolean;
  /** the small camera view over the 3D view */
  float: boolean;
  /** the user switched the camera off until scanning starts (the laser goes off too) */
  paused: boolean;
  error: string | null;
}

export const useCamera = create<CameraState>(() => ({
  info: null,
  cams: local.get<Cams>('cameraCams', 'both'),
  overlay: local.get('cameraOverlay', true),
  float: local.get('cameraFloat', false),
  paused: false,
  error: null,
}));

export function setCameraView(patch: Partial<Pick<CameraState, 'cams' | 'overlay' | 'float'>>) {
  useCamera.setState(patch);
  if (patch.cams) local.set('cameraCams', patch.cams);
  if (patch.overlay !== undefined) local.set('cameraOverlay', patch.overlay);
  if (patch.float !== undefined) local.set('cameraFloat', patch.float);
}

export async function loadCamera() {
  try {
    const info = await api.get<CameraInfo>('/api/capture/camera');
    useCamera.setState({ info });
  } catch {
    /* the server is restarting: the next poll tries again */
  }
}

function remember(info: CameraInfo) {
  const keep = info.remembered;
  if (!keep || !info.driver) return;
  // the scanner chooser starts from these next time (and a reconnect sends them back)
  useCapture.setState(s => {
    const all = (s.lastSettings ?? {}) as Record<string, Record<string, unknown>>;
    const status = s.status?.active && s.status.driver === info.driver ? { ...s.status, settings: { ...s.status.settings, ...keep } } : s.status;
    return { lastSettings: { ...all, [info.driver!]: { ...(all[info.driver!] ?? {}), ...keep } }, status };
  });
}

let pending: Partial<CameraSettings> = {};
let timer: number | undefined;
let inflight: Promise<void> | null = null;

/** Change camera settings. The view updates at once; the server gets the latest values ~0.1 s later (a slider
 * being dragged sends one request, not fifty). */
export function setCamera(changes: Partial<CameraSettings>) {
  const info = useCamera.getState().info;
  if (info?.settings) {
    const next = { ...info.settings, ...changes };
    if (changes.laser_level !== undefined) next.laser_pct = Math.round((1000 * changes.laser_level) / next.level_max) / 10;
    useCamera.setState({ info: { ...info, settings: next }, error: null });
  }
  pending = { ...pending, ...changes };
  window.clearTimeout(timer);
  timer = window.setTimeout(flush, 110);
}

async function flush() {
  if (inflight) {
    await inflight;
  }
  const body = pending;
  pending = {};
  if (!Object.keys(body).length) return;
  inflight = (async () => {
    try {
      const info = await api.post<CameraInfo>('/api/capture/camera', body);
      remember(info);
      if (!Object.keys(pending).length) useCamera.setState({ info, error: null });
    } catch (err) {
      useCamera.setState({ error: (err as Error).message || 'The scanner did not take that setting.' });
      loadCamera();
    } finally {
      inflight = null;
    }
  })();
  await inflight;
}

/** Keep the camera info fresh (settings auto exposure changed, readout) while a view is on screen. */
export function useCameraInfo(on: boolean, everyMs = 600) {
  useEffect(() => {
    if (!on) return;
    let t: number | undefined;
    let alive = true;
    const tick = async () => {
      if (!alive) return;
      if (!document.hidden && !Object.keys(pending).length) await loadCamera();
      if (alive) t = window.setTimeout(tick, everyMs);
    };
    tick();
    return () => {
      alive = false;
      window.clearTimeout(t);
    };
  }, [on, everyMs]);
}

// ---------------------------------------------------------------------------------------------- preview

let holders = 0;
let keepAlive: number | undefined;
let release: number | undefined;

async function preview(on: boolean) {
  try {
    const info = await api.post<CameraInfo>('/api/capture/camera/preview', { on });
    useCamera.setState({ info });
  } catch {
    /* no session (any more) */
  }
}

/**
 * While `want` (a camera view on screen, connected but not scanning), keep the camera preview running: the scanner
 * streams with the laser on and auto exposure working, nothing is recorded. Several views share one preview; it
 * ends when the last one goes (and by itself on the server a few seconds after the pictures stop being asked for).
 */
export function usePreviewHold(want: boolean) {
  useEffect(() => {
    if (!want) return;
    holders += 1;
    if (holders === 1) {
      // a view that comes straight back (the panel and the floating view swapping, a re-render) keeps the preview:
      // switching it off and on again would cycle the laser and the scanner's start-up for nothing
      window.clearTimeout(release);
      preview(true);
      window.clearInterval(keepAlive);
      keepAlive = window.setInterval(() => preview(true), 3000);
    }
    return () => {
      holders -= 1;
      if (holders === 0) {
        window.clearTimeout(release);
        release = window.setTimeout(() => {
          if (holders > 0) return;
          window.clearInterval(keepAlive);
          preview(false);
        }, 1200);
      }
    };
  }, [want]);
}

// ---------------------------------------------------------------------------------------------- pictures

const FRAME_MS = 160;

/** The latest camera picture as an object URL, polled while `on` (about 6 a second; paused in a hidden tab). */
export function useCameraFeed(on: boolean, cams: Cams, overlay: boolean, width: number): { url: string | null; waiting: boolean } {
  const [url, setUrl] = useState<string | null>(null);
  const [waiting, setWaiting] = useState(true);
  const bucket = Math.max(160, Math.round(width / 160) * 160);
  useEffect(() => {
    if (!on) {
      setWaiting(true);
      return;
    }
    let alive = true;
    let shown: string | null = null;
    let t: number | undefined;
    let misses = 0;
    const tick = async () => {
      if (!alive) return;
      if (document.hidden) {
        t = window.setTimeout(tick, 500);
        return;
      }
      const started = performance.now();
      try {
        const res = await fetch(`/api/capture/camera/view?cams=${cams}&overlay=${overlay ? 1 : 0}&width=${bucket}`, { cache: 'no-store' });
        if (res.status === 200) {
          const blob = await res.blob();
          if (!alive) return;
          const next = URL.createObjectURL(blob);
          const old = shown;
          shown = next;
          setUrl(next);
          setWaiting(false);
          misses = 0;
          if (old) window.setTimeout(() => URL.revokeObjectURL(old), 400);
        } else if (++misses > 6) {
          setWaiting(true);
        }
      } catch {
        if (++misses > 6) setWaiting(true);
      }
      const spent = performance.now() - started;
      if (alive) t = window.setTimeout(tick, Math.max(40, FRAME_MS - spent));
    };
    tick();
    return () => {
      alive = false;
      window.clearTimeout(t);
      const last = shown;
      if (last) window.setTimeout(() => URL.revokeObjectURL(last), 1000);
    };
  }, [on, cams, overlay, bucket]);
  return { url: on ? url : null, waiting };
}

// ---------------------------------------------------------------------------------------------- words

export const SURFACE_WORD: Record<Surface, string> = { general: 'Normal', dark: 'Dark', reflective: 'Shiny' };

/** The laser lines in plain words, from the verdict and the readout. */
export function linesWord(verdict: Verdict | undefined, r: { saturated_pct?: number | null; stripe_brightness?: number | null } | undefined): { word: string; tone: 'ok' | 'warn' | 'danger' | 'off'; detail?: string } {
  const sat = r?.saturated_pct ?? null;
  switch (verdict) {
    case 'good':
      return { word: 'bright enough', tone: 'ok' };
    case 'dim':
      return { word: 'too dim', tone: 'warn', detail: 'Parts of the lines may be missed' };
    case 'bright':
      return {
        word: sat != null && sat >= 1 ? `${Math.round(sat)} % too bright` : 'too bright',
        tone: 'danger',
        detail: 'Washed-out lines measure less exactly',
      };
    default:
      return { word: 'none seen yet', tone: 'off', detail: 'Point the scanner at the part' };
  }
}

export const fmtExposure = (us: number) => (us >= 1000 ? `${(us / 1000).toFixed(2)} ms` : `${Math.round(us)} µs`);
