import { Ruler } from 'lucide-react';
import { fmtLen } from '../../lib/format';
import { acrossLine, addDims, useCap } from '../../lib/measureTools';
import { partDims, useSummary, type PartSummary } from '../../lib/summary';
import type { Asset } from '../../lib/types';
import { useStore, type Vec3 } from '../../store';
import { DimMark } from '../../ui/icons';
import { Button } from '../../ui/primitives';
import { sizeLines } from '../../viewport/SizeTag';
import { DiameterGlyph, FlatnessGlyph } from './glyphs';
import { CopyButton } from './parts';

const SIZE_IDS = ['size-L', 'size-W', 'size-H'];
const AXES = [
  { key: 'L', name: 'Length' },
  { key: 'W', name: 'Width' },
  { key: 'H', name: 'Height' },
];

/** The part's size like a caliper display: robust length × width × height along the part itself. */
export function PartSizeCard({ asset }: { asset: Asset }) {
  const summary = useSummary(asset);
  const units = useStore(s => s.display.units);
  const drawn = useStore(s => s.dims.some(d => SIZE_IDS.includes(d.id)));
  const cap = useCap('measure');
  const { dims, robust } = partDims(asset, summary);
  const raw = summary?.dimensions_raw;
  const canDraw = !!summary?.part_frame;
  const text = `${dims.map(v => fmtLen(v, 3)).join(' × ')} ${units}`;

  const toggle = () => {
    useStore.setState(s => ({ dims: [...s.dims.filter(d => !SIZE_IDS.includes(d.id)), ...(drawn ? [] : sizeLines(asset, summary, 'part', units))] }));
  };

  return (
    <section className="meas-size" aria-label="Part size" data-guide="measure.part-size">
      <div className="meas-size-head">
        <span className="meas-size-title"><DimMark size={15} /> Part size</span>
        <span className="spacer" />
        <CopyButton text={text} label="Copy the size" />
        {canDraw && (
          <Button size="sm" variant={drawn ? 'primary' : 'secondary'} icon={<Ruler size={14} />} aria-pressed={drawn} onClick={toggle}>
            {drawn ? 'On the model' : 'Draw on model'}
          </Button>
        )}
      </div>
      <div className="meas-size-row">
        {dims.map((v, i) => (
          <div key={AXES[i].key} className="meas-size-dim">
            <span className="meas-size-axis">{AXES[i].name}</span>
            <span className="meas-size-value">{fmtLen(v, v >= 100 ? 2 : 3)}</span>
          </div>
        ))}
        <span className="meas-size-unit">{units}</span>
      </div>
      <div className="meas-size-scale" aria-hidden />
      <p className="meas-size-note">
        {robust ? 'End to end along the part itself, stray points ignored — scanner noise adds a little at each end; for face-to-face sizes use Heights & steps or Caliper' : 'Smallest box around every point, stray ones included'}
        {raw && robust && Math.max(Math.abs(raw.length - dims[0]), Math.abs(raw.width - dims[1]), Math.abs(raw.height - dims[2])) >= 0.001 && (
          <> · counting every point <span className="mono">{fmtLen(raw.length, 2)} × {fmtLen(raw.width, 2)} × {fmtLen(raw.height, 2)}</span></>
        )}
      </p>
      {summary?.description ? (
        <p className="meas-size-desc">{summary.description}</p>
      ) : cap === 'no' ? (
        <p className="meas-size-desc">The robust size (stray points ignored) and a description of the shape come with a newer server.</p>
      ) : null}
      <Features asset={asset} summary={summary} />
    </section>
  );
}

/** Round and flat features the part analysis found: one click puts them on the model and in the list. */
function Features({ asset, summary }: { asset: Asset; summary: PartSummary | null }) {
  const units = useStore(s => s.display.units);
  const cylinders = [...(summary?.features?.cylinders ?? [])].filter(c => c.radius > 0).sort((a, b) => b.length - a.length).slice(0, 4);
  const planes = [...(summary?.features?.planes ?? [])].sort((a, b) => b.area_mm2 - a.area_mm2).slice(0, 2);
  if (!cylinders.length && !planes.length) return null;
  const size = summary?.dimensions.length ?? 10;

  const addCylinder = (c: (typeof cylinders)[number]) => {
    const [a, b] = acrossLine(c.point, c.axis, c.radius);
    addDims([{ kind: 'diameter', value: 2 * c.radius, a, b }], {
      tool: 'feature',
      title: 'Diameter · found on the part',
      detail: `${fmtLen(c.length, 2)} ${units} long${c.coverage_deg != null ? ` · ${Math.round(c.coverage_deg)}° of the round scanned` : ''} · typical gap to the fit ${fmtLen(c.rms, 3)} ${units}`,
      assetId: asset.id,
    });
  };
  const addPlane = (p: (typeof planes)[number]) => {
    const a = p.point as Vec3;
    const b = p.point.map((x, i) => x + 0.08 * size * p.normal[i]) as Vec3;
    addDims([{ kind: 'flatness', value: p.flatness, a, b }], {
      tool: 'feature',
      title: `Flatness · ${p.label ?? 'flat face'}`,
      detail: `found on the part · about ${Math.round(p.area_mm2).toLocaleString()} ${units}²`,
      assetId: asset.id,
    });
  };

  return (
    <div className="meas-found">
      <span className="meas-found-title">Found on the part <span className="muted">· click to add</span></span>
      <div className="chip-row">
        {cylinders.map((c, i) => (
          <button key={`c${i}`} type="button" className="chip meas-feature" title={`Round section ${fmtLen(c.length, 1)} ${units} long — add its diameter to the measurements`} onClick={() => addCylinder(c)}>
            <DiameterGlyph size={15} />
            <span className="mono">Ø {fmtLen(2 * c.radius, 3)}</span>
          </button>
        ))}
        {planes.map((p, i) => (
          <button key={`p${i}`} type="button" className="chip meas-feature" title={`${p.label ?? 'Flat face'} — add its flatness to the measurements`} onClick={() => addPlane(p)}>
            <FlatnessGlyph size={15} />
            <span className="truncate">Flatness · {p.label ?? 'flat face'}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
