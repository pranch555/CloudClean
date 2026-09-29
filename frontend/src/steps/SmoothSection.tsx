import { Brush, Eraser, Play } from 'lucide-react';
import { setTool } from '../lib/actions';
import { submitJob } from '../lib/jobs';
import { fmtLen } from '../lib/format';
import type { Asset } from '../lib/types';
import { useStore, type SmoothMethod } from '../store';
import { getViewer } from '../viewer/instance';
import { brushRadius } from '../viewport/BrushOverlay';
import { Button, Field, NumberInput, Segmented, Slider } from '../ui/primitives';
import { Block, ToolTile } from './StepFrame';

const METHODS: { value: SmoothMethod; label: string; title: string; help: string }[] = [
  { value: 'denoise', label: 'Denoise', title: 'Feature-preserving', help: 'Removes scanner roughness and ripples while keeping sharp edges such as the rim of a screw head and thread crests.' },
  { value: 'smooth', label: 'Smooth', title: 'Blend the surface', help: 'Blends the surface evenly (volume-preserving Taubin for meshes, moving least squares for points). Rounds sharp edges more than Denoise.' },
  { value: 'remove_spikes', label: 'Flatten bumps', title: 'Edit out protrusions', help: 'Finds small bumps and dents that stick out of the surrounding surface and pulls them back onto it. Larger real features are kept.' },
];

export function clearBrush() {
  const st = useStore.getState();
  useStore.setState({ brush: { ...st.brush, spheres: [], assetId: null } });
  getViewer()?.clearSelection();
}

/** Brush / selection / whole-object smoothing that runs on the full-resolution data and reports how far it moved. */
export function SmoothSection({ target }: { target: Asset | undefined }) {
  const brush = useStore(s => s.brush);
  const tool = useStore(s => s.tool);
  const selection = useStore(s => s.selection);
  const counts = useStore(s => s.selectionCounts);
  const units = useStore(s => s.display.units);
  const method = METHODS.find(m => m.value === brush.method)!;
  const set = (p: Partial<typeof brush>) => useStore.setState(s => ({ brush: { ...s.brush, ...p } }));
  const radius = brushRadius();
  const hasSelection = !!(selection && target && counts[target.id]);
  const scopeReady = brush.scope === 'all' || (brush.scope === 'brush' ? brush.spheres.length > 0 && (!brush.assetId || brush.assetId === target?.id) : hasSelection);

  const apply = () => {
    if (!target) return;
    const op: Record<string, unknown> = { op: brush.method, strength: brush.strength };
    if (brush.method !== 'remove_spikes') op.iterations = brush.iterations;
    if (brush.method === 'smooth' && target.kind === 'pointcloud') op.op = 'smooth_points';
    if (brush.method === 'smooth' && target.kind === 'mesh') op.preserve_edges = true;
    if (brush.scope === 'brush') op.region = { spheres: brush.spheres };
    if (brush.scope === 'selection' && selection) op.region = { view_projection: selection.view_projection, polygon: selection.polygon, visible_only: true };
    const label = { denoise: 'denoised', smooth: 'smoothed', remove_spikes: 'bumps flattened' }[brush.method];
    submitJob('/api/edit', { asset_id: target.id, ops: [op], name: `${target.name} · ${label}` }, job => {
      if (job.status !== 'done') return;
      clearBrush();
      setTool('navigate');
    });
  };

  return (
    <Block title="Smooth rough spots" guide="clean.smooth" tone={tool === 'brush' ? 'signal' : undefined}>
      <Segmented value={brush.method} onChange={v => set({ method: v })} options={METHODS.map(m => ({ value: m.value, label: m.label, title: m.title }))} />
      <p className="hint-text">{method.help}</p>

      <Field label="Apply to">
        <Segmented size="sm" value={brush.scope} onChange={scope => set({ scope })} options={[{ value: 'brush', label: 'Brush' }, { value: 'selection', label: 'Selection' }, { value: 'all', label: 'Whole part' }]} />
      </Field>

      {brush.scope === 'brush' && (
        <>
          <div className="tiles">
            <ToolTile icon={<Brush size={17} />} label={tool === 'brush' ? 'Painting…' : 'Paint rough areas'} sub="drag over the surface" keys="S" active={tool === 'brush'} onClick={() => setTool(tool === 'brush' ? 'navigate' : 'brush')} />
            <ToolTile icon={<Eraser size={17} />} label="Clear paint" sub={`${brush.spheres.length} dabs`} disabled={!brush.spheres.length} onClick={clearBrush} />
          </div>
          <Field label="Brush radius" help={brush.radius ? undefined : `Auto: ${fmtLen(radius, 2)} ${units} from the part size.`}>
            <NumberInput value={+radius.toFixed(3)} step={0.1} min={0.01} unit={units} onChange={r => set({ radius: r })} />
          </Field>
          {brush.assetId && target && brush.assetId !== target.id && <p className="warn-text">The paint is on a different asset than the one selected.</p>}
        </>
      )}
      {brush.scope === 'selection' && !hasSelection && <p className="hint-text">Use Box (B) or Lasso (L) to select the area first. Only the surface facing you is changed.</p>}

      <Field label="Strength" inline={false}>
        <Slider value={brush.strength} min={0.1} max={1} step={0.05} format={v => `${Math.round(v * 100)}%`} onChange={strength => set({ strength })} />
      </Field>
      {brush.method !== 'remove_spikes' && (
        <Field label="Passes">
          <NumberInput value={brush.iterations} step={1} min={1} max={50} onChange={n => set({ iterations: Math.round(n) })} />
        </Field>
      )}
      <p className="caption">The painted area blends into the untouched surface. The result is a new model whose report says how far the surface moved, so you can confirm the dimensions did not change.</p>
      <Button variant="primary" block icon={<Play size={15} />} disabled={!target || !scopeReady} onClick={apply}>
        {method.label} {brush.scope === 'all' ? 'whole part' : brush.scope === 'brush' ? 'painted area' : 'selection'}
      </Button>
    </Block>
  );
}
