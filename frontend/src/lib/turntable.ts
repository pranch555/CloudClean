import { useEffect } from 'react';
import { create } from 'zustand';
import { api, ApiError } from './api';

/*
 * Revopoint turntable (docs/v3-plan.md Contract 4, cloudclean/web/routes_turntable.py).
 *
 * Every command answers with the full status, so the store is updated from command responses and from a shared poll
 * (~2 Hz while someone is watching: the Scan step, its viewport widget, Settings → Scanner & turntable).
 * Servers without the turntable routes answer 404: the UI then says so once and the poll slows right down — no
 * toasts, no errors in a loop.
 */

export type TurntableKind = 'dual_axis' | 'large' | 'simulated';

export interface TurntableCapabilities {
  tilt: boolean;
  tilt_range: [number, number] | null;
  speed_range: [number, number];
  interval_range: [number, number];
  max_rotations?: number;
  continuous?: boolean;
  whole_degrees?: boolean;
}

export interface TurntableMove {
  type: 'rotate' | 'tilt';
  requested_deg: number;
  commanded_deg: number;
  speed_s_per_rev?: number | null;
  note?: string;
  state: 'running' | 'done' | 'stopped' | 'error';
  started?: string;
  finished?: string | null;
  error?: string | null;
}

export interface TurntableProgramState {
  state: 'starting' | 'running' | 'done' | 'stopped' | 'error';
  mode: 'step' | 'continuous';
  /** setting speed | tilting | rotating | settling | capturing | dwelling | leveling | null */
  phase: string | null;
  rotation: number;
  rotations: number;
  stop: number;
  stops_per_rotation: number;
  total_stops: number;
  completed_stops: number;
  progress: number;
  interval_deg: number;
  moves: number[];
  frames_per_stop: number;
  direction: 'cw' | 'ccw';
  speed_s_per_rev: number | null;
  sync_scan: boolean;
  tilt_deg: number | null;
  tilts: number[];
  turned_deg: number;
  started?: string;
  finished?: string | null;
  elapsed_s: number;
  error: string | null;
  warnings: string[];
  notes: string[];
  capture?: { linked: boolean; frames: number };
}

export interface TurntableStatus {
  connected: boolean;
  device: string | null;
  name: string | null;
  kind: TurntableKind | null;
  /** cumulative position, + = clockwise seen from above (not wrapped) */
  angle_deg: number | null;
  angle_wrapped_deg?: number | null;
  tilt_deg: number | null;
  moving: boolean;
  speed_s_per_rev: number | null;
  direction: 'cw' | 'ccw';
  program: TurntableProgramState | null;
  /** turning continuously until stopped; with follow_scan it holds while the live scan is paused or stopped */
  spin?: { follow_scan: boolean; turning: boolean; held_by_scan: boolean; since: string } | null;
  error: string | null;
  capabilities: TurntableCapabilities | null;
  validated: boolean;
  firmware?: string | null;
  last_move?: TurntableMove | null;
  device_state?: Record<string, unknown>;
  bluetooth?: { available: boolean | null; reason: string | null; installed?: boolean };
  remembered_device?: { id: string; name?: string | null; kind?: string | null } | null;
  log?: string[];
}

export interface TurntableDevice {
  id: string;
  name: string | null;
  kind: TurntableKind | string | null;
  rssi: number | null;
  connected?: boolean;
  remembered?: boolean;
}

export interface TurntableProgram {
  mode?: 'step' | 'continuous';
  interval_deg: number;
  frames_per_stop: number;
  direction: 'cw' | 'ccw';
  speed_s_per_rev?: number | null;
  rotations: { tilt_deg: number }[];
  sync_scan: boolean;
}

/** 'unknown' until the first answer; 'no' when the server has no turntable routes (older CloudClean). */
export type TurntableSupport = 'unknown' | 'yes' | 'no';

interface TurntableState {
  status: TurntableStatus | null;
  support: TurntableSupport;
  /** the last poll could not reach the server */
  offline: boolean;
  devices: TurntableDevice[] | null;
  searching: boolean;
  /** label of the command in flight (e.g. 'rotate', 'connect') */
  busy: string | null;
  /** the last command's failure, one sentence; cleared by the next command */
  error: string | null;
}

export const useTurntable = create<TurntableState>(() => ({
  status: null,
  support: 'unknown',
  offline: false,
  devices: null,
  searching: false,
  busy: null,
  error: null,
}));

const BASE = '/api/turntable';

function accept(status: TurntableStatus | null | undefined) {
  if (status && typeof status === 'object' && 'connected' in status) {
    useTurntable.setState({ status, support: 'yes', offline: false });
  }
}

function missing(err: unknown) {
  return err instanceof ApiError && (err.status === 404 || err.status === 405);
}

export async function refreshTurntable(): Promise<TurntableStatus | null> {
  try {
    const status = await api.get<TurntableStatus>(`${BASE}/status`);
    accept(status);
    return status;
  } catch (err) {
    if (missing(err)) useTurntable.setState({ support: 'no', status: null, offline: false });
    else useTurntable.setState({ offline: true });
    return null;
  }
}

/** Run one command: one at a time, the answer (a status) replaces the store's status, failures become `error`. */
async function command(label: string, url: string, body?: unknown): Promise<boolean> {
  useTurntable.setState({ busy: label, error: null });
  try {
    accept(await api.post<TurntableStatus>(`${BASE}${url}`, body ?? {}));
    return true;
  } catch (err) {
    if (missing(err)) useTurntable.setState({ support: 'no' });
    else useTurntable.setState({ error: (err as Error).message || 'The turntable did not answer.' });
    return false;
  } finally {
    useTurntable.setState({ busy: null });
    schedule(250); // pick up the move quickly
  }
}

export const turntable = {
  connect: (device: string, kind = 'auto') => command('connect', '/connect', { device, kind: device === 'simulated' ? 'simulated' : kind }),
  disconnect: () => command('disconnect', '/disconnect'),
  rotate: (degrees: number, speed_s_per_rev?: number | null) => command('rotate', '/rotate', { degrees, ...(speed_s_per_rev ? { speed_s_per_rev } : {}) }),
  tilt: (degrees: number) => command('tilt', '/tilt', { degrees }),
  stop: () => command('stop', '/stop'),
  speed: (s_per_rev: number) => command('speed', '/speed', { s_per_rev }),
  startProgram: (program: TurntableProgram) => command('program', '/program', program),
  stopProgram: () => command('program-stop', '/program/stop'),
  spin: (on: boolean, speed_s_per_rev?: number | null) => command(on ? 'spin' : 'spin-stop', '/spin', { on, follow_scan: true, ...(on && speed_s_per_rev ? { speed_s_per_rev } : {}) }),
  clearError: () => useTurntable.setState({ error: null }),
};

/** Nearby turntables (a Bluetooth scan of `seconds`; 0 = only the simulated and remembered ones). */
export async function searchTurntables(seconds = 4): Promise<TurntableDevice[]> {
  if (useTurntable.getState().searching) return useTurntable.getState().devices ?? [];
  useTurntable.setState({ searching: true });
  try {
    const res = await api.get<TurntableDevice[] | { devices: TurntableDevice[] }>(`${BASE}/devices?scan_seconds=${seconds}`);
    const devices = Array.isArray(res) ? res : res.devices ?? [];
    useTurntable.setState({ devices, support: 'yes' });
    return devices;
  } catch (err) {
    if (missing(err)) useTurntable.setState({ support: 'no' });
    else useTurntable.setState({ error: (err as Error).message });
    return [];
  } finally {
    useTurntable.setState({ searching: false });
  }
}

// ------------------------------------------------------------------ shared poll
let watchers = 0;
let timer: number | undefined;
let polling = false;

function nextDelay(): number {
  const s = useTurntable.getState();
  if (s.support === 'no') return 20000;
  if (document.hidden) return 4000;
  if (s.offline) return 3000;
  const busy = s.status?.moving || s.status?.program?.state === 'running' || s.status?.program?.state === 'starting';
  return busy ? 350 : 500;
}

function schedule(ms = nextDelay()) {
  window.clearTimeout(timer);
  if (!watchers) return;
  timer = window.setTimeout(tick, ms);
}

async function tick() {
  if (polling) return schedule();
  polling = true;
  try {
    await refreshTurntable();
  } finally {
    polling = false;
    schedule();
  }
}

/** Keep the turntable status fresh while the calling component is mounted (and `enabled`). */
export function useTurntablePolling(enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    watchers += 1;
    if (watchers === 1) schedule(0);
    return () => {
      watchers -= 1;
      if (!watchers) window.clearTimeout(timer);
    };
  }, [enabled]);
}

// ------------------------------------------------------------------ helpers
export const wrap360 = (deg: number) => ((deg % 360) + 360) % 360;

export const fmtDeg = (v: number | null | undefined, digits = 1, signed = false): string => {
  if (v == null || !Number.isFinite(v)) return '–';
  const r = Number(v.toFixed(digits));
  const abs = Math.abs(r).toFixed(digits);
  if (r === 0) return `${abs}°`;
  return `${r < 0 ? '−' : signed ? '+' : ''}${abs}°`;
};

/** A platter position on the 0…360° dial, e.g. "127.4°" (never "360.0°"). */
export const fmtAngle = (deg: number | null | undefined): string => {
  if (deg == null || !Number.isFinite(deg)) return '–';
  const w = Math.round(wrap360(deg) * 10) / 10;
  return fmtDeg(w >= 360 ? 0 : w, 1);
};

export const KIND_LABEL: Record<string, string> = { dual_axis: 'Dual-axis', large: 'Large', simulated: 'Simulated' };

export const PHASE_LABEL: Record<string, string> = {
  'setting speed': 'Setting the speed',
  tilting: 'Tilting',
  rotating: 'Turning',
  settling: 'Settling',
  capturing: 'Scanning',
  dwelling: 'Holding still',
  leveling: 'Levelling',
};

export function programRunning(s: TurntableStatus | null | undefined): boolean {
  return !!s?.program && (s.program.state === 'running' || s.program.state === 'starting');
}

/** Rough duration of a program in seconds: one revolution per turn at the chosen speed, a pause at every stop, tilts at 6°/s. */
export function estimateProgramSeconds(p: TurntableProgram, speed: number, fromTilt = 0, fps = 10): number {
  const turns = p.rotations.length;
  const stops = p.mode === 'continuous' ? 1 : Math.ceil(360 / Math.max(1, p.interval_deg) - 1e-9);
  const perStop = 0.3 + 0.1 + (p.sync_scan ? Math.max(0.6, p.frames_per_stop / Math.max(fps, 1)) + 0.4 : 1);
  let tilt = 0;
  let at = fromTilt;
  for (const r of p.rotations) {
    tilt += Math.abs(r.tilt_deg - at) / 6;
    at = r.tilt_deg;
  }
  if (at !== 0 && p.rotations.some(r => r.tilt_deg)) tilt += Math.abs(at) / 6;
  return turns * speed + turns * stops * perStop + tilt;
}

export const fmtMinutes = (s: number): string => (s < 90 ? `${Math.max(1, Math.round(s / 5) * 5)} s` : `${Math.round(s / 60)} min`);
