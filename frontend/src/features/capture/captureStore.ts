import { create } from 'zustand';
import { api } from '../../lib/api';
import { jobStarted } from '../../lib/jobs';
import { projectParam } from '../../lib/projects';
import type { Asset, Job } from '../../lib/types';
import { useStore } from '../../store';
import { lutTexture } from '../../viewer/colormaps';
import { getViewer } from '../../viewer/instance';

export interface SettingSchema {
  key: string;
  label: string;
  type: 'number' | 'boolean' | 'select' | 'text';
  default: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string | number; label: string }[];
  unit?: string;
  help?: string;
}

export interface DriverInfo {
  id: string;
  name: string;
  kind: string;
  description: string;
  available: boolean;
  reason?: string | null;
  settings: SettingSchema[];
  capabilities?: { range_mm?: [number, number] | null; optimal_mm?: number | null; streaming?: boolean; provides_pose?: boolean; detected?: boolean; sweeps?: boolean };
  /** metroy_usb: what was found on the USB bus */
  devices?: { hardware_id: string; product: string; serial?: string; bus?: string; video_nodes: string[]; kernel_drivers?: string[] }[];
}

export interface CaptureStatus {
  type: 'status';
  active: boolean;
  state: string;
  session_id: string | null;
  driver?: string;
  driver_name?: string;
  error?: string | null;
  frames?: number;
  fused_frames?: number;
  scans?: number;
  dropped?: Record<string, number>;
  points?: number;
  display_points?: number;
  fps?: number;
  elapsed_s?: number;
  point_distance_mm?: number | null;
  /** the live view keeps one point per cell of this size (coarser than the point distance) */
  display_voxel_mm?: number | null;
  tracking?: string;
  has_colors?: boolean;
  saved_asset_id?: string | null;
  pending_scans?: number;
  unsaved?: boolean;
  settings?: Record<string, unknown>;
  /** Driver state, e.g. the MetroY's marker map: phase 'mapping' | 'scanning', map_markers, map_frozen. */
  device?: { phase?: string; map_markers?: number; map_frozen?: boolean; [k: string]: unknown };
}

export interface Hole {
  id: string | number;
  kind: 'missing' | 'sparse' | 'edge';
  center: [number, number, number];
  direction: [number, number, number];
  area_mm2: number;
  region?: string;
  hint?: string;
  message?: string;
}

export interface Guidance {
  type: 'guidance';
  state: string;
  status: { severity: 'ok' | 'info' | 'warning' | 'error'; code: string; message: string };
  messages: { code: string; severity: string; message: string }[];
  tracking?: { state: string; fitness?: number | null; rmse_mm?: number | null };
  speed?: { value_mm_s: number | null; limit_mm_s: number | null; too_fast: boolean };
  distance?: { value_mm: number | null; min_mm: number | null; max_mm: number | null; optimal_mm: number | null; state: string };
  frame?: { points: number; fused: boolean };
  density?: { cell_mm: number; target_per_mm2: number; median_ratio: number | null; cells: number; dense_cells: number; dense_ratio: number | null };
  coverage?: { completeness: number | null; covered_area_mm2?: number; missing_area_mm2?: number; estimated_area_mm2?: number; method?: string };
  holes: Hole[];
  sensor?: { position: [number, number, number]; direction: [number, number, number] } | null;
}

interface CaptureState {
  drivers: DriverInfo[];
  /** why the driver list could not be loaded (the server is unreachable or has no capture routes) */
  driversError: string | null;
  sessionSettings: SettingSchema[];
  lastDriver: string | null;
  /** settings used at the last connect, per driver id ({driver: {key: value}}); very old servers sent one flat dict */
  lastSettings: Record<string, unknown>;
  status: CaptureStatus | null;
  guidance: Guidance | null;
  timeline: { t: number; c: number }[];
  connected: boolean;
  colorBy: 'density' | 'color' | 'solid' | 'height';
  pending: PendingScan[];
  /**
   * Look from the scanner while capturing (as Revo Metro does). Any drag, wheel or double-click in the viewport turns
   * it off at once (see Viewport → viewer.onInteract), so the camera is never pulled away from the user; the
   * "Follow scanner" chip turns it back on.
   */
  follow: boolean;
}

export interface PendingScan {
  scan_id: string;
  name: string;
  points: number;
  file?: string | null;
  assessment: import('../../steps/Assessment').Assessment;
}

export async function loadPending() {
  try {
    const res = await api.get<{ pending: PendingScan[] }>('/api/capture/pending');
    useCapture.setState({ pending: res.pending ?? [] });
  } catch {
    /* no session */
  }
}

export async function decidePending(scanId: string, decision: 'fuse' | 'discard' | 'keep_separate') {
  await api.post(`/api/capture/pending/${scanId}`, { decision });
  useCapture.setState(s => ({ pending: s.pending.filter(p => p.scan_id !== scanId) }));
}

export const useCapture = create<CaptureState>(() => ({
  drivers: [],
  driversError: null,
  sessionSettings: [],
  lastDriver: null,
  lastSettings: {},
  status: null,
  guidance: null,
  timeline: [],
  connected: false,
  colorBy: 'density',
  pending: [],
  follow: true,
}));

// Sparse surface (0) recedes into the background and surface at target density (≥ 1) stands out: on the dark Carbon
// stage sparse is dark and dense light (dataviz dark-mode anchor flip), on the light Paper stage the other way round.
// On Paper the ramp stops at a mid blue: its near-black end made a well scanned part one dark silhouette, and the
// shape shading cannot show on near black.
export const DENSITY_STYLE = { kind: 'sequential' as const, min: 1.2, max: 0, tolerance: 0, steps: 0 };
export const DENSITY_STYLE_LIGHT = { ...DENSITY_STYLE, min: 0, max: 1.2, part: [0, 0.68] as [number, number] };

const darkStage = () => typeof document !== 'undefined' && document.documentElement.dataset.theme === 'carbon';
export const densityStyle = () => (darkStage() ? DENSITY_STYLE : DENSITY_STYLE_LIGHT);

// one texture per theme for the whole app: re-creating it per packet leaked GPU memory
const densityLuts: { dark?: ReturnType<typeof lutTexture>; light?: ReturnType<typeof lutTexture> } = {};

export function applyLiveStyle() {
  const v = getViewer();
  if (!v) return;
  const live = v.liveCloud();
  const mode = useCapture.getState().colorBy;
  const s = useCapture.getState().status;
  // the live points are one per display cell: sized from the point distance alone (finer) they left gaps between
  // them, and neither the colours nor the shape shading could show a surface
  const spacing = Math.max(s?.point_distance_mm ?? 0, s?.display_voxel_mm ?? 0);
  if (spacing) live.spacing = spacing;
  const lut = darkStage() ? (densityLuts.dark ??= lutTexture(DENSITY_STYLE)) : (densityLuts.light ??= lutTexture(DENSITY_STYLE_LIGHT));
  live.style(mode === 'color' && !s?.has_colors ? 'solid' : mode, mode === 'density' ? lut : undefined);
  v.invalidate();
}

/** The user grabbed the view: stop following the scanner so the camera stays where they put it. */
export function userTookTheView() {
  const c = useCapture.getState();
  const st = c.status?.state;
  if (c.follow && (st === 'running' || st === 'paused')) useCapture.setState({ follow: false });
}

const MAGIC = 0x43435054;
let socket: WebSocket | null = null;
let retry: number | undefined;
let wantOpen = false;
let fittedEpoch: string | null = null;

function handleBinary(buf: ArrayBuffer) {
  const v = getViewer();
  if (!v || buf.byteLength < 16) return;
  const head = new DataView(buf, 0, 16);
  if (head.getUint32(0, true) !== MAGIC) return;
  const count = head.getUint32(8, true);
  const flags = head.getUint32(12, true);
  let off = 16;
  const xyz = new Float32Array(buf.slice(off, off + count * 12));
  if (flags & 8) {
    // the scanner's current frame: replaces the previous one, fused or not
    v.setLiveFrame(xyz);
    const st = useCapture.getState().status;
    const key = `frame-${st?.session_id}`;
    if (!useCapture.getState().follow && v.liveCloud().count < 2000 && fittedEpoch !== key &&
        fittedEpoch !== `${st?.session_id}` && count > 200) {
      fittedEpoch = key;
      v.fit(undefined, 'iso');
    }
    v.invalidate();
    return;
  }
  off += count * 12;
  let rgb: Uint8Array | undefined;
  if (flags & 1) {
    rgb = new Uint8Array(buf.slice(off, off + count * 3));
    off += count * 3;
  }
  let density: Float32Array | undefined;
  if (flags & 2) {
    density = new Float32Array(buf.slice(off, off + count * 4));
    off += count * 4;
  }
  const live = v.liveCloud();
  if (flags & 4) {
    live.update(new Uint32Array(buf.slice(off, off + count * 4)), xyz, density);
  } else {
    const first = live.count === 0;
    live.append(xyz, rgb, density);
    if (first) applyLiveStyle();
    const st = useCapture.getState().status;
    const key = `${st?.session_id}`;
    if (live.count > 2000 && fittedEpoch !== key && !useCapture.getState().follow) {
      fittedEpoch = key;
      v.fit(undefined, 'iso');
    }
  }
  v.invalidate();
}

export function openStream() {
  wantOpen = true;
  if (socket && socket.readyState <= 1) return;
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/capture/stream`;
  socket = new WebSocket(url);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => useCapture.setState({ connected: true });
  socket.onmessage = ev => {
    if (typeof ev.data !== 'string') return handleBinary(ev.data as ArrayBuffer);
    let msg: any;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (msg.type === 'status') {
      const prev = useCapture.getState().status;
      useCapture.setState({ status: msg });
      // the marker squares belong to a scan in progress: gone once it is stopped, saved or closed
      if (msg.state !== 'running' && msg.state !== 'paused') {
        getViewer()?.setLiveMarkers([]);
        getViewer()?.setMapMarkers([]);
      }
      if ((msg.point_distance_mm && msg.point_distance_mm !== prev?.point_distance_mm) ||
          (msg.display_voxel_mm && msg.display_voxel_mm !== prev?.display_voxel_mm)) applyLiveStyle();
    } else if (msg.type === 'guidance') {
      const c = msg.coverage?.completeness;
      useCapture.setState(s => ({
        guidance: msg,
        timeline: c == null ? s.timeline : [...s.timeline.slice(-299), { t: Date.now(), c }],
      }));
    } else if (msg.type === 'live_view') {
      const v = getViewer();
      if (!v) return;
      v.setLiveMarkers(msg.markers ?? []);
      if (Array.isArray(msg.map)) v.setMapMarkers(msg.map);
      const st = useCapture.getState();
      // never move the camera while a drag is in progress (belt and braces: the drag itself turns follow off)
      if (st.follow && st.status?.state === 'running' && Array.isArray(msg.pose) && !v.controls.busy) v.viewFromScanner(msg.pose);
    } else if (msg.type === 'pending_scan') {
      useCapture.setState(s => ({ pending: [...s.pending.filter(p => p.scan_id !== msg.scan_id), msg] }));
    } else if (msg.type === 'pending_resolved') {
      useCapture.setState(s => ({ pending: s.pending.filter(p => p.scan_id !== msg.scan_id) }));
    } else if (msg.type === 'reset') {
      getViewer()?.liveCloud().reset();
      useCapture.setState({ timeline: [] });
      fittedEpoch = null;
    }
  };
  socket.onclose = () => {
    useCapture.setState({ connected: false });
    socket = null;
    if (wantOpen) retry = window.setTimeout(openStream, 1500);
  };
}

export function closeStream() {
  wantOpen = false;
  window.clearTimeout(retry);
  socket?.close();
  socket = null;
}

export async function loadDrivers() {
  try {
    const d = await api.get<{ drivers: DriverInfo[]; session_settings: SettingSchema[]; last_driver: string | null; last_settings: Record<string, unknown> }>('/api/capture/drivers');
    useCapture.setState({ drivers: d.drivers ?? [], sessionSettings: d.session_settings ?? [], lastDriver: d.last_driver, lastSettings: d.last_settings ?? {}, driversError: null });
  } catch (err) {
    useCapture.setState({ driversError: (err as Error).message || 'The scanner service did not answer.' });
  }
}

/** The settings last used with `driverId` (the server remembers them per driver). */
export function rememberedSettings(driverId: string): Record<string, unknown> {
  const { lastSettings, lastDriver } = useCapture.getState();
  const nested = lastSettings?.[driverId];
  if (nested && typeof nested === 'object' && !Array.isArray(nested)) return nested as Record<string, unknown>;
  const flat = !Object.values(lastSettings ?? {}).some(v => v && typeof v === 'object');
  return flat && driverId === lastDriver ? lastSettings : {};
}

export async function captureAction(action: string, body: unknown = {}) {
  const st = await api.post<CaptureStatus | { status: CaptureStatus }>(`/api/capture/${action}`, body);
  const status = 'status' in st && typeof st.status === 'object' ? (st as { status: CaptureStatus }).status : (st as CaptureStatus);
  if (status?.type === 'status') useCapture.setState({ status });
  return st;
}

/** Open a capture session: the live view starts empty and the stored models step aside. */
export async function connectCapture(driver: string, settings: Record<string, unknown>) {
  await captureAction('connect', { driver, settings });
  getViewer()?.liveCloud().reset();
  getViewer()?.setLiveFrame(null);
  useCapture.setState({ timeline: [], guidance: null });
  useStore.setState({ visible: [] });
  loadDrivers();
}

/** Save the fused cloud as a scan of the open project; with `autoProcess` the server's automation runs on it. */
export async function saveCapture(name: string | undefined, autoProcess: boolean | null): Promise<{ asset: Asset | null; job: Job | null }> {
  const res = (await captureAction('save', { name: name || undefined, auto_process: autoProcess, project_id: projectParam(useStore.getState().projectId) })) as { asset?: Asset; job?: Job | null };
  const st = useStore.getState();
  await st.refreshAssets().catch(() => undefined);
  st.refreshProjects().catch(() => undefined);
  if (res.asset?.id) useStore.setState({ activeId: res.asset.id });
  if (res.job?.id) jobStarted(res.job);
  return { asset: res.asset ?? null, job: res.job ?? null };
}

/** Close the session and forget what it captured (saved scans are not touched). */
export async function discardCapture() {
  await captureAction('discard');
  getViewer()?.liveCloud().reset();
  getViewer()?.setLiveFrame(null);
  getViewer()?.setLiveMarkers([]);
  getViewer()?.setMapMarkers([]);
  useCapture.setState({ guidance: null, timeline: [], pending: [] });
}

/** Whether the server runs its automation on saved captures by default (Settings → Automations). */
export async function loadAutoOnCapture(): Promise<boolean> {
  try {
    const s = await api.get<{ auto_on_capture?: boolean }>('/api/autopilot/settings');
    return s.auto_on_capture ?? true;
  } catch {
    return true;
  }
}
