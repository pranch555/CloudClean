import { Camera, Cpu, FolderSync, Radar, Usb, Wifi, type LucideIcon } from 'lucide-react';
import type { DriverInfo, Hole } from './captureStore';

/*
 * The words the Scan step uses for scanners, states and guidance, in one place so the step, the viewport HUD and
 * Settings say the same thing. Backend names stay the fallback for drivers this list does not know yet.
 */

export interface DriverMeta {
  title: string;
  short: string;
  body: string;
  icon: LucideIcon;
  connect: string;
}

export const DRIVER_META: Record<string, DriverMeta> = {
  metroy_usb: { title: 'MetroY by USB', short: 'MetroY', body: 'Plugged into the computer running CloudClean. No Revo Metro needed.', icon: Usb, connect: 'Connect the MetroY' },
  revo_bridge: { title: 'Revo Metro bridge', short: 'Revo Metro bridge', body: 'Scan in Revo Metro on your PC; every export is sent here and checked.', icon: Wifi, connect: 'Start the bridge' },
  simulated: { title: 'Simulated scanner', short: 'simulator', body: 'Practise scanning and the live guidance without any hardware.', icon: Radar, connect: 'Start the simulator' },
  folder: { title: 'Watch a folder', short: 'folder watch', body: 'Adds every scan exported to a folder or network share.', icon: FolderSync, connect: 'Start watching the folder' },
  revopoint_sdk: { title: 'Revopoint SDK', short: 'Revopoint SDK', body: 'Streams through Revopoint’s own SDK, once you have it.', icon: Cpu, connect: 'Connect through the SDK' },
  realsense: { title: 'Intel RealSense', short: 'RealSense', body: 'Live depth frames from a D400-series camera.', icon: Camera, connect: 'Connect the RealSense' },
  orbbec: { title: 'Orbbec camera', short: 'Orbbec camera', body: 'Live depth frames from a Femto or Gemini camera.', icon: Camera, connect: 'Connect the Orbbec camera' },
};

/** MetroY first; the four that matter for a MetroY workshop, then everything else. */
export const DRIVER_ORDER = ['metroy_usb', 'revo_bridge', 'simulated', 'folder', 'revopoint_sdk', 'realsense', 'orbbec'];
export const PRIMARY_DRIVERS = new Set(['metroy_usb', 'revo_bridge', 'simulated', 'folder']);

export function driverMeta(d: Pick<DriverInfo, 'id' | 'name' | 'description'> | undefined): DriverMeta {
  if (!d) return { title: 'Scanner', short: 'scanner', body: '', icon: Radar, connect: 'Connect' };
  return DRIVER_META[d.id] ?? { title: d.name, short: d.name, body: d.description, icon: Radar, connect: `Connect ${d.name}` };
}

export function sortDrivers(list: DriverInfo[]): DriverInfo[] {
  const rank = (id: string) => (DRIVER_ORDER.indexOf(id) < 0 ? 99 : DRIVER_ORDER.indexOf(id));
  return [...list].sort((a, b) => rank(a.id) - rank(b.id));
}

export type Tone = 'ok' | 'warn' | 'danger' | 'off';

/** One or two words for whether a scanner can be used right now; the full reason goes in a callout. */
export function availability(d: DriverInfo): { word: string; tone: Tone } {
  if (d.available) return { word: 'Ready', tone: 'ok' };
  const r = (d.reason ?? '').toLowerCase();
  if (/not installed|pip install|failed to load/.test(r)) return { word: 'Not installed', tone: 'off' };
  if (/no .*found|not found|not plugged/.test(r)) return { word: 'Not found', tone: 'off' };
  if (/runs on linux|linux only/.test(r)) return { word: 'Linux only', tone: 'off' };
  return { word: 'Needs setup', tone: 'warn' };
}

/** A plain one- or two-sentence version of the server's reason; null when there is none (show the reason itself). */
export function reasonShort(d: DriverInfo): string | null {
  const r = (d.reason ?? '').toLowerCase();
  if (!r) return null;
  if (d.id === 'metroy_usb') {
    if (/no metroy scanner found/.test(r)) return 'CloudClean cannot see a MetroY on the USB ports of the computer it runs on. Is it plugged into your PC instead? Then scan in Revo Metro and use the Revo Metro bridge.';
    if (/runs on linux/.test(r)) return 'The MetroY is plugged in, but native capture runs on Linux (the DGX Spark). On this computer, scan in Revo Metro and use the Revo Metro bridge.';
    if (/opencv/.test(r)) return 'The MetroY is plugged in, but a software part (OpenCV) is missing on the server.';
    if (/permission/.test(r)) return 'The MetroY is plugged in, but the server is not yet allowed to talk to it — it needs a one-line device rule.';
    if (/video node/.test(r)) return 'The MetroY is plugged in, but its camera is not available to the server.';
  }
  if (d.id === 'revopoint_sdk') return 'Revopoint does not publish an SDK for the MetroY yet; this works once you have one.';
  if (/pyrealsense2|pyorbbecsdk/.test(r)) return 'The camera’s software is not installed on the server.';
  return null;
}

export const STATE_WORD: Record<string, string> = {
  idle: 'Not connected',
  created: 'Connecting',
  connected: 'Ready to scan',
  running: 'Scanning',
  paused: 'Paused',
  stopped: 'Stopped',
  finished: 'Finished',
  error: 'Something went wrong',
  closed: 'Closed',
};

export const TRACKING_WORD: Record<string, string> = {
  ok: 'Good',
  weak: 'Weak',
  lost: 'Lost',
  device: 'Good',
  world: 'Fixed',
  'n/a': '–',
  none: '–',
};

export const TRACKING_HELP: Record<string, string> = {
  ok: 'CloudClean knows where every frame belongs.',
  weak: 'Frames only just line up — slow down or go back over scanned surface.',
  lost: 'Frames no longer line up — go back over surface you already scanned.',
  device: 'The scanner reports where it is (markers), so every frame is placed exactly.',
  world: 'Scans arrive already in place.',
};

export const HOLE_WORD: Record<Hole['kind'], string> = { missing: 'Not scanned', sparse: 'Thin', edge: 'Open edge' };

/** Data colours for the three kinds of gap (token names: the 3D markers read them with cssVar). */
export const HOLE_TOKEN: Record<Hole['kind'], string> = { missing: '--signal', sparse: '--warn-glyph', edge: '--series-4' };

export const fmtArea = (mm2: number | null | undefined): string => (mm2 == null ? '–' : mm2 >= 100 ? `${(mm2 / 100).toFixed(1)} cm²` : `${Math.round(mm2)} mm²`);

/** 0:07, 2:14, 1:02:05 — a recording clock. */
export function fmtClock(seconds: number | null | undefined): string {
  const s = Math.max(0, Math.floor(seconds ?? 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}
