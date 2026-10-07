export type AssetKind = 'pointcloud' | 'mesh' | 'image';

export interface GeometryStats {
  kind?: string;
  points?: number;
  vertices?: number;
  triangles?: number;
  has_colors: boolean;
  has_normals: boolean;
  bbox_min: number[];
  bbox_max: number[];
  dimensions: number[];
  oriented_dimensions?: number[];
  diagonal: number;
  spacing: number;
  surface_area?: number;
  watertight?: boolean;
  volume?: number;
  width?: number;
  height?: number;
  focal_35mm?: number;
}

export interface ScalarInfo {
  name: string;
  unit: string;
  description: string;
  min: number | null;
  max: number | null;
}

export interface Asset {
  id: string;
  name: string;
  kind: AssetKind;
  created: string;
  operation: string;
  parents: string[];
  params: Record<string, unknown>;
  stats: GeometryStats;
  textured?: boolean;
  scalars?: ScalarInfo[];
  file?: string;
  project?: string;
  has_thumbnail?: boolean;
  /** robust part frame + dimensions computed by the server when the asset was created (v3) */
  part?: { dimensions: { length: number; width: number; height: number }; frame?: { origin: number[]; length: number[]; width: number[]; height: number[] } };
}

export interface Project {
  id: string;
  name: string;
  created: string;
  updated: string;
  description?: string;
  cover_asset_id?: string | null;
  /** the part as it should be (CAD or a trusted mesh): Measure -> Golden model checks scans against it */
  golden_asset_id?: string | null;
  counts?: { scans?: number; meshes?: number; results?: number; photos?: number; total?: number };
  /** true when the server has no project support and this is the client-side "everything" project */
  virtual?: boolean;
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export interface Job {
  id: string;
  kind: string;
  title: string;
  status: JobStatus;
  created: number;
  started: number | null;
  finished: number | null;
  logs: string[];
  result: string[];
  error: string | null;
  payload: Record<string, unknown>;
  progress: { fraction: number; label: string | null } | null;
  output: Record<string, unknown>;
}

export interface ParamField {
  default: unknown;
  type: string;
}

export type ParamSchema = Record<string, ParamField>;

export interface ParamsResponse {
  schema: { clean: ParamSchema; merge: ParamSchema; mesh: ParamSchema; texture: ParamSchema };
  presets: Record<string, Record<string, unknown>>;
  formats: { pointcloud: string[]; mesh: string[] };
  workspace: string;
}

export type ColorMode = 'original' | 'solid' | 'asset' | 'normal' | 'scalar' | 'height';

export type ViewName = 'front' | 'back' | 'left' | 'right' | 'top' | 'bottom' | 'iso' | 'fit';

/** The journey through one part. Any step can be opened at any time. */
export type Step = 'capture' | 'clean' | 'align' | 'mesh' | 'measure' | 'export';

export type Screen = 'home' | 'workspace';

export type ViewportTool = 'navigate' | 'box' | 'lasso' | 'measure' | 'pivot' | 'brush';

/** A resolved region (docs/v3-plan.md Contract 2): world shapes the browser can test points against. */
export interface ResolvedRegion {
  shapes: (
    | { type: 'obb'; center: number[]; axes: number[][]; half: number[] }
    | { type: 'sphere'; center: number[]; radius: number }
    | { type: 'cylinder'; point: number[]; axis: number[]; radius: number; half_length: number }
    | { type: 'screen'; view_projection: number[]; polygon: [number, number][] }
  )[];
  invert?: boolean;
}

export interface ScreenSelection {
  view_projection: number[];
  polygon: [number, number][];
  asset_id?: string;
  count: number;
  /** set when the selection came from a region (assistant or a measure tool) rather than a lasso/box */
  region?: Record<string, unknown>;
  label?: string;
}
