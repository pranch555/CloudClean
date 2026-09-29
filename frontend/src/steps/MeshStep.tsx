import { useState } from 'react';
import { ArrowRight, ChevronDown, Droplets, Palette, Shield, Waves } from 'lucide-react';
import { fmtCount, fmtLen } from '../lib/format';
import { submitJob } from '../lib/jobs';
import { useStore, useTarget } from '../store';
import { MeshGlyph } from '../ui/icons';
import { defaultsOf, ParamForm } from '../ui/ParamForm';
import { Button, Field, Metric, Segmented } from '../ui/primitives';
import { AutoColour } from '../features/colour/AutoColour';
import { ColourSection } from '../features/colour/ColourSection';
import { EditRecipe } from './EditTools';
import { HolesSection } from './HolesSection';
import { Block, ChoiceCards, NextStepButton, ResultCard, StepFrame, TargetCard } from './StepFrame';
import { DriftLine } from './DriftLine';
import { useReport } from './useReport';

const DETAIL: Record<string, number> = { auto: 0, fine: 11, finest: 12 };
const SMOOTH: Record<string, number> = { none: 0, light: 3, more: 8 };

export function MeshStep() {
  const params = useStore(s => s.params)!;
  const target = useTarget();
  const [values, setValues] = useState(() => defaultsOf(params.schema.mesh));
  const [busy, setBusy] = useState(false);
  const detail = Object.entries(DETAIL).find(([, v]) => v === Number(values.depth))?.[0] ?? 'custom';
  const smooth = Object.entries(SMOOTH).find(([, v]) => v === Number(values.smooth_iterations))?.[0] ?? 'custom';
  const isMesh = target?.kind === 'mesh';

  const run = async () => {
    if (!target) return;
    setBusy(true);
    await submitJob('/api/mesh', { asset_id: target.id, params: values }, () => setBusy(false));
    setBusy(false);
  };

  return (
    <StepFrame
      step="mesh"
      footer={
        <>
          {isMesh ? (
            <Button variant="primary" size="lg" block onClick={() => useStore.getState().goStep('measure')}>
              Measure this mesh <ArrowRight size={17} aria-hidden />
            </Button>
          ) : (
            <Button variant="primary" size="lg" block icon={<MeshGlyph size={18} />} loading={busy} disabled={!target} onClick={run}>
              {target ? 'Build the mesh' : 'Pick a scan to mesh'}
            </Button>
          )}
          {!isMesh && <NextStepButton from="mesh" />}
        </>
      }
    >
      <TargetCard asset={target} empty="Click a cleaned or merged scan in the list on the left." />
      {target?.operation === 'mesh' && <MeshResult />}

      {!isMesh && (
        <>
          <Block title="What kind of surface?" guide="mesh.kind">
            <ChoiceCards
              value={values.watertight ? 'closed' : 'open'}
              onChange={v => setValues(x => ({ ...x, watertight: v === 'closed' }))}
              options={[
                { value: 'open', title: 'True to the scan', body: 'Only surface backed by scan points. Best for measuring — nothing is invented.', badge: 'for measuring', icon: <Shield size={18} /> },
                { value: 'closed', title: 'Closed, watertight', body: 'Also closes unscanned areas such as the bottom. Best for 3D printing.', icon: <Droplets size={18} /> },
              ]}
            />
          </Block>
          <Field inline={false} label="Detail" help="Finer detail keeps sharp features and small threads, but takes longer and needs more memory.">
            <Segmented value={detail} onChange={d => d !== 'custom' && setValues(x => ({ ...x, depth: DETAIL[d] }))} options={[{ value: 'auto', label: 'Automatic' }, { value: 'fine', label: 'Fine' }, { value: 'finest', label: 'Finest' }]} />
          </Field>
          <Field inline={false} label="Smoothing" help="Volume-preserving; keeps the size. More smoothing rounds sharp edges.">
            <Segmented value={smooth} onChange={v => v !== 'custom' && setValues(x => ({ ...x, smooth_iterations: SMOOTH[v] }))} options={[{ value: 'none', label: 'None' }, { value: 'light', label: 'Light' }, { value: 'more', label: 'More' }]} />
          </Field>
          <details className="disclosure">
            <summary>Fine-tune <span className="sub">method, trimming, triangle count</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
            <div className="disclosure-body">
              <ParamForm group="mesh" schema={params.schema.mesh} values={values} onChange={setValues} />
            </div>
          </details>
        </>
      )}

      {isMesh && target && <HolesSection mesh={target} />}

      {isMesh && (
        <Block title="Repair & finish" guide="mesh.repair">
          <p className="hint-text">Each set of edits makes a new mesh; the original stays.</p>
          <EditRecipe
            target={target}
            quick={[
              { op: 'fill_holes', label: 'Fill holes' },
              { op: 'repair', label: 'Repair the surface' },
              { op: 'remove_small_components', label: 'Remove loose bits' },
              { op: 'smooth', label: 'Smooth' },
              { op: 'simplify', label: 'Use fewer triangles' },
              { op: 'align_floor', label: 'Sit flat on the floor' },
            ]}
          />
        </Block>
      )}

      <div className="divider" />
      <Block title="Colour from photos" guide="mesh.colour">
        <AutoColour target={target} />
        <details className="disclosure" data-guide="mesh.colour-by-hand">
          <summary><Palette size={16} aria-hidden /> Line up one photo by hand <span className="sub">when automatic does not work</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
          <div className="disclosure-body">
            <ColourSection />
          </div>
        </details>
      </Block>
    </StepFrame>
  );
}

function MeshResult() {
  const target = useTarget()!;
  const units = useStore(s => s.display.units);
  const r = useReport(target);
  if (!r) return null;
  const d = r.deviation ?? {};
  const spacing = r.spacing ?? target.stats.spacing;
  const good = d.p95 != null && spacing ? d.p95 <= spacing * 1.5 : true;
  return (
    <ResultCard tone={good ? 'ok' : 'warn'} title={good ? 'Mesh built — it follows the scan closely' : 'Mesh built — it strays from the scan in places'}>
      <Metric label="Triangles" value={fmtCount(target.stats.triangles)} />
      <Metric label="95 % of scan points within" value={fmtLen(d.p95, 3)} unit={units} tone={good ? 'ok' : 'warn'} />
      <Metric label="Average distance to the scan" value={fmtLen(d.mean, 3)} unit={units} />
      <Metric label="Watertight" value={target.stats.watertight ? 'yes' : 'no'} />
      <DriftLine asset={target} />
      <span className="caption"><Waves size={12} aria-hidden style={{ display: 'inline', verticalAlign: -2 }} /> Point spacing of the scan is {fmtLen(spacing, 3)} {units}; a mesh within about that distance keeps the scan’s accuracy.</span>
    </ResultCard>
  );
}
