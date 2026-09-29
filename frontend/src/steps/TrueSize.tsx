import { useState } from 'react';
import { Ruler } from 'lucide-react';
import { submitJob } from '../lib/jobs';
import type { Asset } from '../lib/types';
import { useStore } from '../store';
import { Button, Field, NumberInput } from '../ui/primitives';
import { Block } from './StepFrame';

/**
 * Whether a model was made from photos (itself or anything it came from), so its size is only estimated, and the
 * scale factor already applied on the way (an edit with op scale), if any. `sheet`: the photos were taken on the
 * printed scale sheet, so the model came out at true size (with this uncertainty in %).
 */
export function photoSize(asset: Asset | undefined, byId: Map<string, Asset>): { photos: boolean; scaled: number | null; sheet: { uncertainty: number | null; barChecked: boolean } | null } {
  const seen = new Set<string>();
  let scaled: number | null = null;
  let sheet: { uncertainty: number | null; barChecked: boolean } | null = null;
  const walk = (a: Asset | undefined): boolean => {
    if (!a || seen.has(a.id)) return false;
    seen.add(a.id);
    if (a.operation === 'photos') {
      if (a.params?.size === 'scale sheet') {
        sheet = { uncertainty: typeof a.params.size_uncertainty_pct === 'number' ? a.params.size_uncertainty_pct : null, barChecked: typeof a.params.ruler_mm === 'number' };
      }
      return true;
    }
    if (!a.parents.some(p => walk(byId.get(p)))) return false;
    const ops = (a.params?.ops as { op?: string; factor?: number }[] | undefined) ?? [];
    for (const op of ops) if (op.op === 'scale' && op.factor) scaled = (scaled ?? 1) * op.factor;
    return true;
  };
  return { photos: walk(asset), scaled, sheet };
}

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2));

/**
 * "Set the true size": photos alone do not show how big a part is, so a model made from photos has a guessed size; one length the user
 * knows scales it to true size (edit op scale, factor = true / on the model).
 */
export function TrueSizeBlock({ target, scaled }: { target: Asset; scaled: number | null }) {
  const measurements = useStore(s => s.measurements);
  const byId = useStore(s => s.byId);
  const sheet = photoSize(target, byId).sheet;
  const dims = target.part?.dimensions;
  const known = [
    ...(dims ? ([['Overall length', dims.length], ['Overall width', dims.width], ['Overall height', dims.height]] as const) : []),
    ...measurements.filter(m => m.assetId === target.id && m.result?.distance).map(m => [m.label || 'Measured distance', m.result!.distance] as const),
  ].filter(([, v]) => v > 0);
  const [onModel, setOnModel] = useState<number>(known[0]?.[1] ?? 0);
  const [truth, setTruth] = useState<number>(0);
  const factor = onModel > 0 && truth > 0 ? truth / onModel : null;
  const sane = factor != null && factor > 0.2 && factor < 5;

  const [again, setAgain] = useState(false);
  if (sheet && scaled == null && !again) {
    return (
      <Block guide="clean.scale" title="True size">
        <p className="hint-text">
          Made from photos on the printed scale sheet, so it is already at its true size{sheet.uncertainty != null ? ` (±${sheet.uncertainty < 0.1 ? sheet.uncertainty.toFixed(2) : sheet.uncertainty.toFixed(1)} %)` : ''}, standing on the sheet (height 0).
          {!sheet.barChecked && ' The print was not checked with its 100 mm bar: a printer that scales the page scales the model too.'}{' '}
          <button type="button" className="link" onClick={() => setAgain(true)}>Set it by hand instead</button>
        </p>
      </Block>
    );
  }
  if (scaled != null && !again) {
    return (
      <Block guide="clean.scale" title="True size">
        <p className="hint-text">
          Made from photos and scaled ×{scaled.toFixed(4)} to a length you gave, so it is true to size.{' '}
          <button type="button" className="link" onClick={() => setAgain(true)}>Set it again</button>
        </p>
      </Block>
    );
  }

  const run = () => {
    if (!factor) return;
    submitJob('/api/edit', {
      asset_id: target.id,
      name: `${target.name} · true size`,
      ops: [{ op: 'scale', factor, center: 'centroid' }],
    });
  };

  return (
    <Block guide="clean.scale" title="Set the true size" tone="signal">
      <p className="hint-text">
        This model was made from photos, and photos alone do not show how big a part is: its size is a guess and can be off by many times. Pick a length you know, type its real length, and CloudClean scales the whole model to match.
      </p>
      {known.length > 0 && (
        <div className="chip-row" role="group" aria-label="Which length do you know?">
          {known.map(([label, v]) => (
            <button key={`${label}-${v}`} type="button" className={`chip ${Math.abs(onModel - v) < 1e-9 ? 'is-on' : ''}`} aria-pressed={Math.abs(onModel - v) < 1e-9} onClick={() => setOnModel(v)}>
              {label} · {fmt(v)} mm
            </button>
          ))}
        </div>
      )}
      <Field label="Length on the model" help="Pick one above, or measure it (Measure → Dimensions → Point to point) and type it here.">
        <NumberInput value={Number(onModel.toFixed(3))} onChange={setOnModel} min={0} unit="mm" width={130} />
      </Field>
      <Field label="Its real length" help="From a drawing, a caliper or the ruler in your photos.">
        <NumberInput value={truth} onChange={setTruth} min={0} unit="mm" width={130} />
      </Field>
      {factor != null && (
        <p className={sane ? 'hint-text' : 'err-text'} role={sane ? undefined : 'alert'}>
          {sane
            ? `Scale by ×${factor.toFixed(4)}: the model gets ${Math.abs((factor - 1) * 100).toFixed(1)}% ${factor >= 1 ? 'bigger' : 'smaller'}.`
            : `×${factor.toFixed(2)} is a very big change — check that both lengths are of the same feature and in mm.`}
        </p>
      )}
      <Button variant="primary" icon={<Ruler size={16} />} disabled={!sane} onClick={run}>
        Scale to true size → new model
      </Button>
    </Block>
  );
}
