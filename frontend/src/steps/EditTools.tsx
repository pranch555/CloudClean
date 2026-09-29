import { useEffect, useMemo, useState } from 'react';
import { ChevronDown, Minus, Play, Plus } from 'lucide-react';
import { api } from '../lib/api';
import { submitJob } from '../lib/jobs';
import { humanize } from '../lib/format';
import { uid } from '../lib/uid';
import type { Asset } from '../lib/types';
import { useStore } from '../store';
import { Badge, Button, Field, NumberInput, Select, Switch, TextInput } from '../ui/primitives';

interface ArgSpec {
  type: string;
  default: unknown;
  description: string;
  required?: boolean;
  choices?: string[];
}
interface OpSpec {
  description: string;
  applies_to: string[];
  args: Record<string, ArgSpec>;
}
export type Catalogue = Record<string, OpSpec>;

interface QueuedOp {
  key: string;
  op: string;
  args: Record<string, unknown>;
}

/** Friendly names for edit operations (the server catalogue has the full descriptions). */
export const OP_LABEL: Record<string, string> = {
  crop_box: 'Crop to a box',
  cut_plane: 'Cut with a plane',
  delete_sphere: 'Delete a ball-shaped region',
  delete_region: 'Delete a region',
  keep_region: 'Keep only a region',
  remove_outliers: 'Remove stray points',
  remove_small_components: 'Remove loose bits',
  downsample: 'Thin out points',
  align_floor: 'Sit flat on the floor',
  align_principal: 'Line up with the X/Y/Z axes',
  center: 'Move to the origin',
  rotate: 'Rotate',
  translate: 'Move',
  mirror: 'Mirror',
  transform: 'Apply a transform matrix',
  scale: 'Scale (changes size!)',
  fill_holes: 'Fill holes',
  repair: 'Repair the surface',
  smooth: 'Smooth',
  simplify: 'Use fewer triangles',
  subdivide: 'Subdivide triangles',
  flip_normals: 'Flip inside / outside',
  recompute_normals: 'Recompute surface directions',
  to_pointcloud: 'Convert to points',
  paint: 'Paint one colour',
};

const GROUPS: { title: string; ops: string[] }[] = [
  { title: 'Remove', ops: ['crop_box', 'cut_plane', 'delete_sphere', 'remove_outliers', 'remove_small_components', 'downsample'] },
  { title: 'Position', ops: ['align_floor', 'align_principal', 'center', 'rotate', 'translate', 'mirror', 'transform', 'scale'] },
  { title: 'Surface', ops: ['fill_holes', 'repair', 'smooth', 'simplify', 'subdivide', 'flip_normals', 'recompute_normals', 'to_pointcloud'] },
  { title: 'Other', ops: ['paint'] },
];

let catalogueCache: Promise<Catalogue> | null = null;

export function useCatalogue(): Catalogue | null {
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  useEffect(() => {
    catalogueCache ??= api.get<Catalogue>('/api/edit/ops');
    catalogueCache.then(setCatalogue).catch(err => {
      catalogueCache = null;
      useStore.getState().toast({ kind: 'error', title: 'Could not load the edit tools', body: err.message });
    });
  }, []);
  return catalogue;
}

/**
 * A recipe of edits applied in order to the full-resolution data of one model; the result is a new model.
 * `quick` lists one-click operations shown as chips above the recipe.
 */
export function EditRecipe({ target, quick }: { target: Asset | undefined; quick: { op: string; label: string; args?: Record<string, unknown> }[] }) {
  const catalogue = useCatalogue();
  const [queue, setQueue] = useState<QueuedOp[]>([]);
  const [adding, setAdding] = useState('align_floor');

  const available = useMemo(() => {
    if (!catalogue) return [];
    return Object.keys(catalogue).filter(op => !target || catalogue[op].applies_to.includes(target.kind));
  }, [catalogue, target]);

  const add = (op: string, args?: Record<string, unknown>) => {
    if (!catalogue?.[op]) return;
    const up = useStore.getState().display.upAxis;
    const defaults: Record<string, unknown> = Object.fromEntries(Object.entries(catalogue[op].args).map(([k, a]) => [k, k === 'up' ? up : a.default]));
    if (op === 'rotate') Object.assign(defaults, { axis: defaults.axis ?? up, degrees: defaults.degrees ?? 90 });
    if (op === 'translate') defaults.offset ??= [0, 0, 0];
    if (op === 'scale') defaults.factor ??= 1;
    if (op === 'paint') defaults.rgb ??= [0.7, 0.72, 0.75];
    if (target && op === 'delete_sphere') {
      defaults.center = target.stats.bbox_min.map((v, i) => +((v + target.stats.bbox_max[i]) / 2).toFixed(3));
      defaults.radius ??= +(target.stats.diagonal / 10).toFixed(3);
    }
    if (target && op === 'crop_box') {
      defaults.min = target.stats.bbox_min.map(v => +v.toFixed(3));
      defaults.max = target.stats.bbox_max.map(v => +v.toFixed(3));
    }
    if (target && op === 'cut_plane') {
      defaults.point = target.stats.bbox_min.map((v, i) => +((v + target.stats.bbox_max[i]) / 2).toFixed(3));
      defaults.normal = defaults.normal ?? (up === 'z' ? [0, 0, 1] : [0, 1, 0]);
    }
    setQueue(q => [...q, { key: uid(), op, args: { ...defaults, ...args } }]);
  };

  const run = () => {
    if (!target || !queue.length) return;
    const names = queue.map(q => (OP_LABEL[q.op] ?? humanize(q.op)).toLowerCase()).join(', ');
    submitJob('/api/edit', { asset_id: target.id, ops: queue.map(q => ({ op: q.op, ...q.args })), name: `${target.name} · ${names.length > 42 ? 'edited' : names}` }, job => {
      if (job.status === 'done') setQueue([]);
    });
  };

  const options = GROUPS.flatMap(g => g.ops.filter(op => available.includes(op)).map(op => ({ value: op, label: `${g.title} · ${OP_LABEL[op] ?? humanize(op)}` })))
    .concat(available.filter(op => !GROUPS.some(g => g.ops.includes(op)) && !['select_screen', 'delete_region', 'keep_region'].includes(op)).map(op => ({ value: op, label: OP_LABEL[op] ?? humanize(op) })));

  return (
    <div className="stack">
      <div className="chip-row">
        {quick.filter(q => available.includes(q.op)).map(q => (
          <button key={q.label} type="button" className="chip" onClick={() => add(q.op, q.args)} disabled={!catalogue || !target}>
            <Plus size={13} aria-hidden /> {q.label}
          </button>
        ))}
      </div>
      {queue.length > 0 && (
        <ol className="recipe">
          {queue.map((q, i) => (
            <li key={q.key} className="recipe-step">
              <div className="recipe-head">
                <span className="index-bubble">{i + 1}</span>
                <span className="recipe-name">{OP_LABEL[q.op] ?? humanize(q.op)}</span>
                {q.op === 'scale' && <Badge tone="warning">changes the size</Badge>}
                <button type="button" className="icon-btn icon-btn-sm" aria-label="Remove this step" onClick={() => setQueue(qs => qs.filter(x => x.key !== q.key))}>
                  <Minus size={15} />
                </button>
              </div>
              {catalogue?.[q.op] && <OpArgs spec={catalogue[q.op]} values={q.args} onChange={args => setQueue(qs => qs.map(x => (x.key === q.key ? { ...x, args } : x)))} />}
            </li>
          ))}
        </ol>
      )}
      <details className="disclosure">
        <summary>More edits <span className="sub">every operation</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
        <div className="disclosure-body">
          <div className="row">
            <div className="grow"><Select value={adding} onChange={setAdding} options={options} /></div>
            <Button size="sm" icon={<Plus size={14} />} onClick={() => add(adding)} disabled={!catalogue || !available.includes(adding)}>Add</Button>
          </div>
          {catalogue?.[adding] && <p className="hint-text">{catalogue[adding].description}</p>}
        </div>
      </details>
      {queue.length > 0 && (
        <Button variant="primary" block icon={<Play size={15} />} disabled={!target} onClick={run}>
          Apply {queue.length} edit{queue.length > 1 ? 's' : ''} → new model
        </Button>
      )}
    </div>
  );
}

function OpArgs({ spec, values, onChange }: { spec: OpSpec; values: Record<string, unknown>; onChange: (v: Record<string, unknown>) => void }) {
  const entries = Object.entries(spec.args).filter(([, a]) => !['polygon', 'matrix16', 'region'].includes(a.type));
  if (!entries.length) return null;
  const set = (k: string, v: unknown) => onChange({ ...values, [k]: v });
  return (
    <div className="fields compact">
      {entries.map(([k, a]) => {
        const v = values[k];
        let control;
        switch (a.type) {
          case 'vec3':
            control = <VectorInput value={Array.isArray(v) ? (v as number[]) : [0, 0, 0]} onChange={x => set(k, x)} />;
            break;
          case 'choice':
            control = <Select value={String(v ?? a.choices?.[0] ?? '')} onChange={x => set(k, x)} options={(a.choices ?? []).map(c => ({ value: c, label: c }))} />;
            break;
          case 'axis':
            control = Array.isArray(v) ? <VectorInput value={v as number[]} onChange={x => set(k, x)} /> : <Select value={String(v ?? 'z')} onChange={x => set(k, x)} options={['x', 'y', 'z'].map(c => ({ value: c, label: c.toUpperCase() }))} />;
            break;
          case 'center':
            control = Array.isArray(v) ? <VectorInput value={v as number[]} onChange={x => set(k, x)} /> : <Select value={String(v ?? 'centroid')} onChange={x => set(k, x)} options={['centroid', 'bbox', 'origin'].map(c => ({ value: c, label: c }))} />;
            break;
          case 'boolean':
            control = <Switch checked={!!v} onChange={x => set(k, x)} />;
            break;
          case 'number':
          case 'integer':
            control = <NumberInput value={Number(v ?? 0)} step={a.type === 'integer' ? 1 : 'any'} onChange={x => set(k, a.type === 'integer' ? Math.round(x) : x)} />;
            break;
          default:
            control = <TextInput mono value={v == null ? '' : JSON.stringify(v)} onChange={x => { try { set(k, JSON.parse(x)); } catch { /* keep typing */ } }} />;
        }
        return (
          <Field key={k} label={humanize(k)} help={a.description} inline={a.type !== 'vec3'}>
            {control}
          </Field>
        );
      })}
    </div>
  );
}

export function VectorInput({ value, onChange }: { value: number[]; onChange: (v: number[]) => void }) {
  return (
    <div className="vector-input">
      {['x', 'y', 'z'].map((ax, i) => (
        <label key={ax} className={`vec axis-${ax}`}>
          <span>{ax.toUpperCase()}</span>
          <NumberInput value={value[i] ?? 0} onChange={x => onChange(value.map((y, j) => (j === i ? x : y)))} />
        </label>
      ))}
    </div>
  );
}
