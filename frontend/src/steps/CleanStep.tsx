import { useEffect, useState } from 'react';
import { ChevronDown, Lasso, SquareDashedMousePointer } from 'lucide-react';
import { setTool } from '../lib/actions';
import { fmtCount, fmtPct } from '../lib/format';
import { submitJob } from '../lib/jobs';
import { useStore, useTarget } from '../store';
import { CleanGlyph } from '../ui/icons';
import { defaultsOf, ParamForm } from '../ui/ParamForm';
import { Button, Field, Metric, Switch } from '../ui/primitives';
import { EditRecipe } from './EditTools';
import { SmoothSection } from './SmoothSection';
import { photoSize, TrueSizeBlock } from './TrueSize';
import { Block, ChoiceCards, NextStepButton, ResultCard, StepFrame, TargetCard, ToolTile } from './StepFrame';
import { DriftLine } from './DriftLine';
import { useReport } from './useReport';

const PRESETS: { value: string; title: string; body: string; badge?: string }[] = [
  { value: 'light', title: 'Gentle', body: 'Only obvious stray points. Every edge stays exactly as scanned.' },
  { value: 'standard', title: 'Standard', body: 'Scanner noise, stray points and loose bits floating near the part.', badge: 'recommended' },
  { value: 'aggressive', title: 'Thorough', body: 'Also thin fringes and sparse areas. Can nibble very thin edges.' },
];

export function CleanStep() {
  const params = useStore(s => s.params)!;
  const tool = useStore(s => s.tool);
  const target = useTarget();
  const byId = useStore(s => s.byId);
  const size = photoSize(target, byId);
  const [preset, setPreset] = useState('standard');
  const [values, setValues] = useState(() => defaultsOf(params.schema.clean, params.presets.standard));
  const [busy, setBusy] = useState(false);
  const cloud = target?.kind === 'pointcloud';

  useEffect(() => {
    setValues(v => ({ ...defaultsOf(params.schema.clean, params.presets[preset]), remove_plane: v.remove_plane }));
  }, [preset]);

  const run = async () => {
    if (!target) return;
    setBusy(true);
    await submitJob('/api/clean', { asset_ids: [target.id], preset, params: values }, () => setBusy(false));
    setBusy(false);
  };

  return (
    <StepFrame
      step="clean"
      footer={
        <>
          <Button variant="primary" size="lg" block icon={<CleanGlyph size={18} />} loading={busy} disabled={!cloud} onClick={run}>
            {cloud ? 'Clean this scan' : 'Pick a point cloud to clean'}
          </Button>
          <NextStepButton from="clean" />
        </>
      }
    >
      <TargetCard asset={target} empty="Click the scan you want to clean in the list on the left." />
      {target?.operation === 'clean' && <CleanResult />}
      {target && size.photos && <TrueSizeBlock key={target.id} target={target} scaled={size.scaled} />}

      <Block title="How thorough?" guide="clean.presets clean.table">
        <ChoiceCards value={preset} onChange={setPreset} options={PRESETS} />
      </Block>

      <Field label="Remove the table or turntable" help="Only removed when it is wider than the part and the part sits on one side of it.">
        <Switch checked={!!values.remove_plane} onChange={remove_plane => setValues(v => ({ ...v, remove_plane }))} label="Remove the table or turntable" />
      </Field>

      <details className="disclosure">
        <summary>Fine-tune <span className="sub">every cleaning setting</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
        <div className="disclosure-body">
          <ParamForm group="clean" schema={params.schema.clean} values={values} onChange={setValues} />
        </div>
      </details>

      <div className="divider" />

      <Block title="Remove things by hand" guide="clean.by-hand">
        <p className="hint-text">Select what you don't want, then Delete — or keep only the selection. Shift adds to it.</p>
        <div className="tiles">
          <ToolTile icon={<SquareDashedMousePointer size={17} />} label="Box" sub="drag a rectangle" keys="B" active={tool === 'box'} onClick={() => setTool(tool === 'box' ? 'navigate' : 'box')} />
          <ToolTile icon={<Lasso size={17} />} label="Lasso" sub="draw around" keys="L" active={tool === 'lasso'} onClick={() => setTool(tool === 'lasso' ? 'navigate' : 'lasso')} />
        </div>
      </Block>

      <Block title="Position & crop" guide="clean.position clean.scale">
        <EditRecipe
          target={target}
          quick={[
            { op: 'align_floor', label: 'Sit flat on the floor' },
            { op: 'center', label: 'Move to the origin', args: { mode: 'bbox_bottom' } },
            { op: 'align_principal', label: 'Line up with X/Y/Z' },
            { op: 'crop_box', label: 'Crop to a box' },
            { op: 'cut_plane', label: 'Cut with a plane' },
            { op: 'remove_small_components', label: 'Remove loose bits' },
          ]}
        />
      </Block>

      <div className="divider" />
      <SmoothSection target={target} />
    </StepFrame>
  );
}

function CleanResult() {
  const target = useTarget()!;
  const r = useReport(target);
  if (!r) return null;
  const removed = r.removed_fraction ?? (r.input_points ? 1 - r.output_points / r.input_points : null);
  return (
    <ResultCard tone={removed != null && removed > 0.25 ? 'warn' : 'ok'} title={`Cleaned — removed ${fmtPct(removed, 1)} of the points`}>
      <Metric label="Points kept" value={`${fmtCount(r.output_points)} of ${fmtCount(r.input_points)}`} />
      {(r.steps ?? []).filter((s: { removed: number }) => s.removed > 0).slice(0, 4).map((s: { step: string; removed: number }) => (
        <Metric key={s.step} label={s.step.replace(/_/g, ' ')} value={`−${fmtCount(s.removed)}`} />
      ))}
      <DriftLine asset={target} />
    </ResultCard>
  );
}
