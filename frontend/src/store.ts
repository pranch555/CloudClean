import { create } from 'zustand';
import { api } from './lib/api';
import { panelFit, type PanelMode } from './lib/panelFit';
import { loadProjects } from './lib/projects';
import type { Asset, ColorMode, Job, ParamsResponse, Project, Screen, ScreenSelection, Step, ViewportTool } from './lib/types';
import type { ScalarStyle } from './viewer/colormaps';
import type { ClipSettings, Pane } from './viewer/Viewer';

export const local = {
  get<T>(key: string, fallback: T): T {
    try {
      const v = localStorage.getItem(`cloudclean.${key}`);
      return v == null ? fallback : (JSON.parse(v) as T);
    } catch {
      return fallback;
    }
  },
  set(key: string, value: unknown) {
    try {
      localStorage.setItem(`cloudclean.${key}`, JSON.stringify(value));
    } catch {
      /* storage unavailable (private window) */
    }
  },
};

export interface Toast {
  id: number;
  kind: 'info' | 'ok' | 'warn' | 'error';
  title: string;
  body?: string;
  action?: { label: string; run: () => void };
}

export type Vec3 = [number, number, number];

export interface DistanceResult {
  distance: number;
  dx: number;
  dy: number;
  dz: number;
  along_axis?: number;
  perpendicular?: number;
  /** 1-sigma uncertainty of the distance in mm, when both points were snapped onto a local surface fit */
  uncertainty?: number;
}

/** One two-point measurement. `b` is null while the second point is being placed. */
export interface Measurement {
  id: string;
  label: string;
  assetId: string | null;
  a: Vec3;
  b: Vec3 | null;
  snapped: boolean;
  result: DistanceResult | null;
  crests?: number;
}

/** A dimension drawn in the 3D view: from a measuring tool, the assistant or a report. */
export interface DimLine {
  id: string;
  label: string;
  kind: string;          // distance | extent | caliper | diameter | angle | sphere | ...
  value: number | null;
  unit: string;
  a: Vec3;
  b: Vec3;
  assetId?: string;
  source: 'tool' | 'assistant';
}

export interface Annotation {
  id: string;
  position: Vec3;
  text: string;
}

export type SmoothMethod = 'denoise' | 'smooth' | 'remove_spikes';

export interface BrushState {
  spheres: [number, number, number, number][];
  radius: number;
  strength: number;
  method: SmoothMethod;
  iterations: number;
  scope: 'brush' | 'selection' | 'all';
  assetId: string | null;
}

export type MeasureAxis = 'none' | 'x' | 'y' | 'z' | 'thread';

export interface ThreadStandard {
  designation: string;
  series: string;
  major_diameter: number;
  pitch: number;
  tpi?: number;
  pitch_deviation: number;
  major_diameter_deviation: number;
  match: 'close' | 'possible' | 'poor';
  tolerance?: Record<string, { major_min?: number; major_max?: number; within?: boolean }>;
}

export interface ThreadResult {
  asset_id: string;
  asset_name?: string;
  kind: string;
  handedness: string;
  pitch: number;
  pitch_se: number;
  tpi?: number;
  per_crest_pitches: number[];
  pitch_max_deviation: number;
  crest_count: number;
  major_diameter: number;
  minor_diameter: number;
  /** null when only one flank was scanned (a single scan of a bolt sees one flank of every turn) */
  pitch_diameter: number | null;
  thread_depth: number;
  flank_angle_deg: number | null;
  axis: Vec3;
  center: Vec3;
  axis_uncertainty_deg: number;
  length: number;
  angular_coverage_deg: number;
  points_used: number;
  noise_rms: number;
  crest_points: Vec3[];
  crests: number[];
  confidence: 'high' | 'medium' | 'low';
  /** 'crest by crest', or 'helix fit' when the crests could not be followed (one-flank scans) */
  method?: string;
  both_flanks?: boolean;
  warnings: string[];
  notes: string[];
  standards: ThreadStandard[];
  profile?: { t: number[]; r_mean: number[]; r_max: number[]; r_min: number[] };
}

export interface PairState {
  movId: string;
  refId: string;
  list: { source: [number, number, number]; target: [number, number, number] }[];
  pending: { source?: [number, number, number]; target?: [number, number, number] };
  prevVisible: string[];
}

export interface Layout {
  left: number;          // column widths in px
  right: number;
  leftOpen: boolean;
  rightOpen: boolean;
}

export const DEFAULT_LAYOUT: Layout = { left: 288, right: 392, leftOpen: true, rightOpen: true };

export type Theme = 'paper' | 'carbon' | 'system';

export interface Display {
  colorMode: ColorMode;
  pointScale: number;
  wireframe: boolean;
  showBox: boolean;
  showGrid: boolean;
  upAxis: 'y' | 'z';
  rotateStyle: 'free' | 'turntable';
  rotatePivot: 'center' | 'cursor';
  projection: 'perspective' | 'orthographic';
  scalar: { name: string; style: ScalarStyle } | null;
  units: string;
  /** Size readout: the part's own axes (robust, default) or the scanner's world axes. */
  sizeFrame: 'part' | 'world';
}

interface State {
  params: ParamsResponse | null;
  booted: boolean;         // first asset/project load finished
  screen: Screen;
  projects: Project[];
  projectId: string | null;
  assets: Asset[];          // every asset of the workspace
  byId: Map<string, Asset>;
  selected: string[];       // extra models picked for multi-model steps (align)
  visible: string[];
  activeId: string | null;  // the current model: what every step works on
  step: Step;
  rightTab: 'step' | 'assistant';
  tool: ViewportTool;
  selection: ScreenSelection | null;
  selectionCounts: Record<string, number>;
  measurements: Measurement[];
  dims: DimLine[];
  annotations: Annotation[];
  brush: BrushState;
  panes: Pane[];
  layout: Layout;
  /** how each side panel is shown at the current window width (docked, or a drawer over the 3D view) */
  panelModes: { left: PanelMode; right: PanelMode };
  /** open drawers (only meaningful for a side whose mode is 'drawer'; not saved) */
  drawers: { left: boolean; right: boolean };
  measureAxis: MeasureAxis;
  thread: ThreadResult | null;
  display: Display;
  clip: ClipSettings;
  jobs: Job[];
  jobsOpen: boolean;
  expandedJob: string | null;
  settingsOpen: string | null; // section id or null
  theme: Theme;
  textSize: 'normal' | 'large';
  toasts: Toast[];
  loadingNames: string[] | null;
  pairing: PairState | null;
  savedPairs: Record<string, PairState['list']>;
  exported: Record<string, boolean>; // project id -> exported this session (journey status)

  set: (patch: Partial<State>) => void;
  setDisplay: (patch: Partial<Display>) => void;
  /** open/close requests for a side that is currently a drawer open/close its drawer instead */
  setLayout: (patch: Partial<Layout>) => void;
  setPanelModes: (modes: { left: PanelMode; right: PanelMode }) => void;
  refreshAssets: () => Promise<void>;
  refreshProjects: () => Promise<void>;
  openProject: (id: string, step?: Step) => void;
  goHome: () => void;
  goStep: (step: Step) => void;
  activate: (id: string, show?: boolean) => void;
  toggleSelect: (id: string) => void;
  toggleVisible: (id: string) => void;
  showOnly: (ids: string[]) => void;
  toast: (t: Omit<Toast, 'id'> & { ms?: number }) => number;
  dismiss: (id: number) => void;
}

let toastSeq = 0;

const initialLayout: Layout = { ...DEFAULT_LAYOUT, ...local.get('layout3', {}) };
const initialFit = panelFit(typeof window === 'undefined' ? 1920 : window.innerWidth, initialLayout);
const initialModes = { left: initialFit.left, right: initialFit.right };

/** Whether a side panel is on screen: docked and open, or its drawer is open. */
export function panelShown(s: Pick<State, 'layout' | 'panelModes' | 'drawers'>, side: 'left' | 'right'): boolean {
  return s.panelModes[side] === 'drawer' ? s.drawers[side] : side === 'left' ? s.layout.leftOpen : s.layout.rightOpen;
}

export const useStore = create<State>((set, get) => ({
  params: null,
  booted: false,
  screen: local.get<Screen>('screen', 'home'),
  projects: [],
  projectId: local.get<string | null>('project', null),
  assets: [],
  byId: new Map(),
  selected: [],
  visible: [],
  activeId: null,
  step: local.get<Step>('step', 'capture'),
  rightTab: 'step',
  tool: 'navigate',
  selection: null,
  selectionCounts: {},
  measurements: [],
  dims: [],
  annotations: [],
  panes: [],
  layout: initialLayout,
  panelModes: initialModes,
  drawers: { left: false, right: initialModes.right === 'drawer' && initialLayout.rightOpen },
  brush: { spheres: [], radius: 0, strength: 0.7, method: 'denoise', iterations: 5, scope: 'brush', assetId: null },
  measureAxis: 'none',
  thread: null,
  display: {
    colorMode: 'original',
    pointScale: 1,
    wireframe: false,
    showBox: false,
    showGrid: true,
    upAxis: local.get('up', 'y'),
    rotateStyle: local.get('rotateStyle', 'free'),
    rotatePivot: local.get('rotatePivot', 'center'),
    projection: 'perspective',
    scalar: null,
    units: local.get('units', 'mm'),
    sizeFrame: local.get('sizeFrame', 'part'),
  },
  clip: { enabled: false, axis: 'x', position: 0, flip: false },
  jobs: [],
  jobsOpen: false,
  expandedJob: null,
  settingsOpen: null,
  theme: local.get<Theme>('theme', 'paper'),
  textSize: local.get<'normal' | 'large'>('textSize', 'normal'),
  toasts: [],
  loadingNames: null,
  pairing: null,
  savedPairs: {},
  exported: {},

  set: patch => {
    set(patch);
    if (patch.screen) local.set('screen', patch.screen);
    if (patch.step) local.set('step', patch.step);
    if (patch.projectId !== undefined) local.set('project', patch.projectId);
    if (patch.theme) local.set('theme', patch.theme);
    if (patch.textSize) local.set('textSize', patch.textSize);
  },

  setLayout: patch => {
    const st = get();
    const rest = { ...patch };
    const drawers = { ...st.drawers };
    let drawerChange = false;
    for (const side of ['left', 'right'] as const) {
      const key = side === 'left' ? 'leftOpen' : 'rightOpen';
      if (rest[key] === undefined || st.panelModes[side] !== 'drawer') continue;
      drawers[side] = !!rest[key];
      delete rest[key];
      drawerChange = true;
    }
    // two drawers never stack: the one just opened wins
    if (drawerChange && drawers.left && drawers.right) {
      if (patch.rightOpen && !st.drawers.right) drawers.left = false;
      else drawers.right = false;
    }
    const layout = { ...st.layout, ...rest };
    set(drawerChange ? { layout, drawers } : { layout });
    local.set('layout3', layout);
  },

  setPanelModes: modes => {
    const st = get();
    if (st.panelModes.left === modes.left && st.panelModes.right === modes.right) return;
    // a side turning into a drawer starts closed, except the step panel, which keeps showing if it was
    const drawers = {
      left: modes.left === 'drawer' ? (st.panelModes.left === 'drawer' ? st.drawers.left : false) : false,
      right: modes.right === 'drawer' ? (st.panelModes.right === 'drawer' ? st.drawers.right : st.layout.rightOpen) : false,
    };
    if (drawers.left && drawers.right) drawers.left = false;
    set({ panelModes: modes, drawers });
  },

  setDisplay: patch => {
    set(s => ({ display: { ...s.display, ...patch } }));
    if (patch.upAxis) local.set('up', patch.upAxis);
    if (patch.rotateStyle) local.set('rotateStyle', patch.rotateStyle);
    if (patch.rotatePivot) local.set('rotatePivot', patch.rotatePivot);
    if (patch.units) local.set('units', patch.units);
    if (patch.sizeFrame) local.set('sizeFrame', patch.sizeFrame);
  },

  refreshAssets: async () => {
    const list = await api.assets();
    const byId = new Map(list.map(a => [a.id, a]));
    const s = get();
    set({
      assets: list,
      byId,
      selected: s.selected.filter(id => byId.has(id)),
      visible: s.visible.filter(id => byId.has(id)),
      activeId: s.activeId && byId.has(s.activeId) ? s.activeId : null,
    });
  },

  refreshProjects: async () => {
    const projects = await loadProjects(get().assets);
    const current = get().projectId;
    set({ projects, projectId: current && projects.some(p => p.id === current) ? current : projects[0]?.id ?? null });
  },

  openProject: (id, step) => {
    const s = get();
    const inProject = projectAssets({ assets: s.assets, projects: s.projects, projectId: id }).filter(a => a.kind !== 'image');
    const newest = [...inProject].sort((a, b) => a.created.localeCompare(b.created)).pop();
    const keepActive = s.activeId && inProject.some(a => a.id === s.activeId);
    s.set({
      screen: 'workspace',
      projectId: id,
      step: step ?? s.step,
      activeId: keepActive ? s.activeId : newest?.id ?? null,
      visible: keepActive ? s.visible.filter(v => inProject.some(a => a.id === v)) : newest ? [newest.id] : [],
      selected: [],
      panes: [],
    });
  },

  goHome: () => get().set({ screen: 'home', panes: [], tool: 'navigate' }),

  goStep: step => get().set({ step, rightTab: 'step', screen: 'workspace' }),

  activate: (id, show = true) => {
    const a = get().byId.get(id);
    if (!a) return;
    const visible = get().visible;
    set({
      activeId: id,
      visible: show && a.kind !== 'image' && !visible.includes(id) && !get().pairing ? [...visible, id] : visible,
    });
  },

  toggleSelect: id => {
    const sel = get().selected;
    set({ selected: sel.includes(id) ? sel.filter(x => x !== id) : [...sel, id] });
  },

  toggleVisible: id => {
    const v = get().visible;
    set({ visible: v.includes(id) ? v.filter(x => x !== id) : [...v, id] });
  },

  showOnly: ids => set({ visible: ids.filter(id => get().byId.get(id)?.kind !== 'image') }),

  toast: t => {
    const id = ++toastSeq;
    const { ms, ...rest } = t;
    set(s => ({ toasts: [...s.toasts.slice(-3), { id, ...rest }] }));
    const life = ms ?? (t.kind === 'error' ? 9000 : t.kind === 'warn' ? 7000 : 4200);
    if (life > 0) setTimeout(() => get().dismiss(id), life);
    return id;
  },

  dismiss: id => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })),
}));

/** Assets that belong to the open project (all of them when the server has no projects). */
export function projectAssets(s: Pick<State, 'assets' | 'projects' | 'projectId'>): Asset[] {
  const p = s.projects.find(x => x.id === s.projectId);
  if (!p || p.virtual) return s.assets;
  return s.assets.filter(a => a.project === p.id);
}

export function useProjectAssets(): Asset[] {
  const assets = useStore(s => s.assets);
  const projects = useStore(s => s.projects);
  const projectId = useStore(s => s.projectId);
  return projectAssets({ assets, projects, projectId });
}

export function useActive(): Asset | undefined {
  return useStore(s => (s.activeId ? s.byId.get(s.activeId) : undefined));
}

/** The current model when it is of one of the given kinds. */
export function useTarget(kinds: Asset['kind'][] = ['pointcloud', 'mesh']): Asset | undefined {
  const a = useActive();
  return a && kinds.includes(a.kind) ? a : undefined;
}

export const toast = (t: Omit<Toast, 'id'> & { ms?: number }) => useStore.getState().toast(t);
