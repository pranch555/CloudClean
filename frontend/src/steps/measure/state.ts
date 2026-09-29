import { create } from 'zustand';
import { local } from '../../store';
import type { Direction, PartAxis, RegionPick, SectionResult } from '../../lib/measureTools';

export type MeasureTab = 'dimensions' | 'thread' | 'cad' | 'accuracy';
export type ToolId = 'p2p' | 'across' | 'extent' | 'steps' | 'diameter' | 'angle' | 'flatness' | 'sphere' | 'section';

/** The latest cross-section, kept for the drawing in the panel. */
export interface SectionShot {
  result: SectionResult;
  direction: PartAxis;
  at: number;
  assetId: string;
  dimIds: string[];
}

interface MeasureUi {
  tab: MeasureTab;
  tool: ToolId | null;
  across: Direction;
  extent: Direction;
  /** steps: the direction the faces are found across */
  steps: Direction;
  /** across / extent: only within the current selection */
  within: boolean;
  /** flatness / angle: only the surface facing the camera */
  facing: boolean;
  axisHint: 'auto' | PartAxis;
  sectionDir: PartAxis;
  /** mm from the part's start along the section direction; null = the middle */
  sectionAt: number | null;
  faceA: RegionPick | null;
  busy: string | null;
  errors: Record<string, string | null>;
  /** dimension ids of the latest result of each tool */
  last: Partial<Record<ToolId, string[]>>;
  section: SectionShot | null;
}

const TABS: MeasureTab[] = ['dimensions', 'thread', 'cad', 'accuracy'];
const saved = local.get<string>('measureTab', 'dimensions');

export const useMeasureUi = create<MeasureUi>(() => ({
  tab: (TABS as string[]).includes(saved) ? (saved as MeasureTab) : 'dimensions',
  tool: null,
  across: 'length',
  extent: 'length',
  steps: 'length',
  within: false,
  facing: true,
  axisHint: 'auto',
  sectionDir: 'length',
  sectionAt: null,
  faceA: null,
  busy: null,
  errors: {},
  last: {},
  section: null,
}));

export function setTab(tab: MeasureTab) {
  useMeasureUi.setState({ tab });
  local.set('measureTab', tab);
}

export const setUi = (patch: Partial<MeasureUi>) => useMeasureUi.setState(patch);

export function setError(key: string, message: string | null) {
  useMeasureUi.setState(s => ({ errors: { ...s.errors, [key]: message } }));
}
