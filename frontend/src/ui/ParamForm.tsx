import { useMemo } from 'react';
import type { ParamSchema } from '../lib/types';
import { humanize } from '../lib/format';
import { Field, NumberInput, Section, Select, Switch, TextInput } from './primitives';

export interface ParamInfo {
  basic: string[];
  labels: Record<string, string>;
  help?: Record<string, string>;
  enums?: Record<string, (string | [number | string, string])[]>;
  units?: Record<string, string>;
}

export const PARAM_INFO: Record<string, ParamInfo> = {
  clean: {
    basic: ['remove_plane', 'sor', 'sor_std_ratio', 'radius_filter', 'cluster', 'cluster_keep_ratio', 'voxel_size', 'scale'],
    labels: {
      remove_plane: 'Remove table / turntable', sor: 'Remove noise outliers', sor_std_ratio: 'Outlier strictness (σ)',
      radius_filter: 'Remove sparse stray points', cluster: 'Remove floating debris', cluster_keep_ratio: 'Keep pieces ≥ share of main',
      voxel_size: 'Downsample voxel', scale: 'Scale factor', crop_min: 'Crop box min (x, y, z)', crop_max: 'Crop box max (x, y, z)',
      dedupe: 'Drop duplicate points', plane_distance_multiplier: 'Plane thickness (× spacing)', plane_min_fraction: 'Min plane share of points',
      sor_neighbors: 'Outlier neighbours', radius_multiplier: 'Sparse radius (× spacing)', radius_min_neighbors: 'Min neighbours in radius',
      cluster_eps_multiplier: 'Debris gap (× spacing)', max_filter_removal: 'Safety: max removal per filter', normals: 'Compute normals',
      normal_neighbors: 'Normal neighbours', recompute_normals: 'Recompute normals from scratch',
    },
    help: {
      remove_plane: 'Only removed when it is wider than the item and the item sits on one side of it.',
      sor_std_ratio: 'Lower removes more: 1.5 aggressive, 2 balanced, 3 gentle.',
      cluster_keep_ratio: '0.05 keeps separate parts ≥ 5% of the main piece. Raise it to keep only the main object.',
      voxel_size: 'Keep 0 for full scanner resolution (maximum accuracy).',
      scale: 'Multiplies coordinates, e.g. 1000 converts metres to millimetres.',
    },
    units: { voxel_size: 'mm' },
  },
  merge: {
    basic: ['method', 'ransac_trials', 'refine_passes', 'dedupe_voxel', 'final_sor'],
    labels: {
      method: 'Alignment', ransac_trials: 'Global search attempts', refine_passes: 'Refinement passes (3+ scans)',
      dedupe_voxel: 'Thin overlap to voxel', final_sor: 'Clean merged cloud', feature_voxel: 'Feature voxel (0 = auto)',
      icp_threshold_multiplier: 'Final ICP distance (× spacing)', min_fitness: 'Warn below overlap',
      layer_ratio: 'Doubled skin above (× noise)', layer_min_mm: 'Ignore gaps below',
    },
    help: { method: 'auto finds the pose even if the item was flipped, and lines up on marker stickers when both scans share 3 or more · markers uses only the stickers · icp for scans already roughly aligned · none just combines.' },
    enums: { method: ['auto', 'markers', 'icp', 'none'] },
    units: { dedupe_voxel: 'mm', layer_min_mm: 'mm' },
  },
  mesh: {
    basic: ['method', 'watertight', 'depth', 'smooth_iterations', 'target_triangles'],
    labels: {
      method: 'Method', watertight: 'Watertight (fill holes)', depth: 'Detail depth (0 = auto)', smooth_iterations: 'Smoothing passes',
      target_triangles: 'Max triangles (0 = no limit)', max_auto_depth: 'Auto depth cap', linear_fit: 'Linear fit',
      trim: 'Trim unsupported surface', trim_distance_multiplier: 'Trim distance (× spacing)', density_quantile: 'Density trim quantile',
      min_component_ratio: 'Drop islands < share of largest', transfer_colors: 'Copy scan colours',
    },
    help: {
      method: 'poisson is smooth and closes small gaps · bpa connects the measured points directly.',
      watertight: 'Closes unscanned areas such as the bottom. Off keeps only surface backed by scan data.',
      depth: '9 coarse · 10–11 detailed · 12 very fine (slow, needs lots of RAM).',
      smooth_iterations: 'Taubin smoothing keeps size and volume.',
    },
    enums: { method: ['poisson', 'bpa'], trim: ['distance', 'density', 'none'] },
  },
  texture: {
    basic: ['blend_power', 'texture_size', 'unseen'],
    labels: {
      blend_power: 'Photo blend sharpness', texture_size: 'Bake UV texture', unseen: 'Areas no photo sees',
      min_cos: 'Min viewing angle (cos)', edge_margin_px: 'Silhouette margin (px)', visibility_tolerance: 'Occlusion tolerance',
      texture_max_triangles: 'Triangles for UV texture',
    },
    help: {
      blend_power: '1 = soft average of photos · 4+ = prefer the most head-on photo.',
      texture_size: 'Vertex colours are always made. A UV texture also exports an image-textured GLB / OBJ.',
      unseen: 'diffuse spreads nearby colours · scan keeps scanner colour · gray is neutral.',
    },
    enums: { unseen: ['diffuse', 'scan', 'gray'], texture_size: [[0, 'Off'], [2048, '2048 px'], [4096, '4096 px'], [8192, '8192 px']] },
  },
};

export type ParamValues = Record<string, unknown>;

export const defaultsOf = (schema: ParamSchema, overrides: ParamValues = {}): ParamValues =>
  Object.fromEntries(Object.entries(schema).map(([k, f]) => [k, k in overrides ? overrides[k] : f.default]));

/** Renders a parameter dataclass schema as a form. Basic settings first, the rest under "Advanced". */
export function ParamForm({ group, schema, values, onChange }: { group: string; schema: ParamSchema; values: ParamValues; onChange: (v: ParamValues) => void }) {
  const info = PARAM_INFO[group] ?? { basic: [], labels: {} };
  const keys = useMemo(() => [...info.basic.filter(k => k in schema), ...Object.keys(schema).filter(k => !info.basic.includes(k))], [schema, info]);
  const set = (k: string, v: unknown) => onChange({ ...values, [k]: v });

  const row = (key: string) => {
    const { type } = schema[key];
    const id = `p-${group}-${key}`;
    const opts = info.enums?.[key];
    const v = values[key];
    let control;
    if (opts) {
      const options = opts.map(o => (Array.isArray(o) ? { value: String(o[0]), label: o[1] } : { value: o, label: o }));
      control = <Select id={id} value={String(v)} onChange={x => set(key, type === 'int' ? parseInt(x, 10) : x)} options={options} />;
    } else if (type === 'bool') {
      control = <Switch checked={!!v} onChange={x => set(key, x)} label={info.labels[key] || humanize(key)} />;
    } else if (type === 'int' || type === 'float') {
      control = <NumberInput id={id} value={Number(v ?? 0)} step={type === 'int' ? 1 : 'any'} unit={info.units?.[key]} onChange={x => set(key, type === 'int' ? Math.round(x) : x)} />;
    } else {
      const text = v == null ? '' : Array.isArray(v) ? v.join(', ') : String(v);
      control = <TextInput id={id} mono value={text} placeholder="x, y, z" onChange={x => set(key, x.trim() ? x.split(/[,\s]+/).filter(Boolean).map(Number) : null)} />;
    }
    return (
      <Field key={key} htmlFor={id} label={info.labels[key] || humanize(key)} help={info.help?.[key]}>
        {control}
      </Field>
    );
  };

  const advanced = keys.filter(k => !info.basic.includes(k));
  return (
    <div className="param-form">
      <div className="fields">{keys.filter(k => info.basic.includes(k)).map(row)}</div>
      {advanced.length > 0 && (
        <Section title="Advanced settings" collapsible defaultOpen={false}>
          <div className="fields">{advanced.map(row)}</div>
        </Section>
      )}
    </div>
  );
}
