import { Ruler } from 'lucide-react';
import { fmtLen } from '../lib/format';
import { partDims, useSummary, type PartSummary } from '../lib/summary';
import type { Asset } from '../lib/types';
import { useStore, type DimLine, type Vec3 } from '../store';
import { DimMark } from '../ui/icons';
import { IconButton, Segmented } from '../ui/primitives';

const SIZE_IDS = ['size-L', 'size-W', 'size-H'];

/** Dimension lines along the edges of the part's own box (or the world box), offset so they clear the model. */
export function sizeLines(asset: Asset, summary: PartSummary | null, frame: 'part' | 'world', units: string): DimLine[] {
  const mk = (id: string, label: string, value: number, a: Vec3, b: Vec3): DimLine => ({ id, label, kind: 'extent', value, unit: units, a, b, assetId: asset.id, source: 'tool' });
  if (frame === 'part' && summary?.part_frame) {
    const { origin: o, length: L, width: W, height: H } = summary.part_frame;
    const { length: l, width: w, height: h } = summary.dimensions;
    const off = Math.max(l, w, h) * 0.06;
    const at = (a: number, b: number, c: number): Vec3 => [0, 1, 2].map(i => o[i] + a * L[i] + b * W[i] + c * H[i]) as Vec3;
    return [
      mk('size-L', 'L', l, at(0, -off, -off), at(l, -off, -off)),
      mk('size-W', 'W', w, at(-off, 0, -off), at(-off, w, -off)),
      mk('size-H', 'H', h, at(-off, -off, 0), at(-off, -off, h)),
    ];
  }
  const { bbox_min: lo, bbox_max: hi } = asset.stats;
  const d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const off = Math.max(...d) * 0.06;
  return [
    mk('size-L', 'X', d[0], [lo[0], lo[1] - off, lo[2] - off], [hi[0], lo[1] - off, lo[2] - off]),
    mk('size-W', 'Y', d[1], [lo[0] - off, lo[1], lo[2] - off], [lo[0] - off, hi[1], lo[2] - off]),
    mk('size-H', 'Z', d[2], [lo[0] - off, lo[1] - off, lo[2]], [lo[0] - off, lo[1] - off, hi[2]]),
  ];
}

export function SizeTag() {
  const a = useStore(s => (s.activeId ? s.byId.get(s.activeId) : undefined));
  const visible = useStore(s => s.visible);
  const units = useStore(s => s.display.units);
  const frame = useStore(s => s.display.sizeFrame);
  const drawn = useStore(s => s.dims.some(d => SIZE_IDS.includes(d.id)));
  const summary = useSummary(a && a.kind !== 'image' ? a : undefined);
  if (!a || a.kind === 'image' || !visible.includes(a.id)) return null;

  const world = frame === 'world';
  const { dims, robust } = world ? { dims: a.stats.dimensions as [number, number, number], robust: false } : partDims(a, summary);
  const axes = world ? ['X', 'Y', 'Z'] : ['L', 'W', 'H'];
  const note = world ? 'along the scanner’s axes' : robust ? 'robust · stray points ignored' : 'smallest box around every point';

  const toggleLines = (show: boolean, f = frame) => {
    useStore.setState(s => ({ dims: [...s.dims.filter(d => !SIZE_IDS.includes(d.id)), ...(show ? sizeLines(a, summary, f, units) : [])] }));
  };

  return (
    <div className="hud size-tag" aria-label="Size of the current model" data-guide="size-tag">
      <div className="size-head">
        <DimMark size={15} />
        <span>{world ? 'World box' : 'Part size'}</span>
        <span className="muted">·</span>
        <span className="truncate" title={a.name}>{a.name}</span>
      </div>
      <div className="size-row">
        {dims.map((v, i) => (
          <span key={i} className="size-dim">
            <span className={`size-axis ${world ? `axis-${axes[i].toLowerCase()}` : ''}`}>{axes[i]}</span>
            <span className="size-value">{fmtLen(v, v >= 100 ? 2 : 3)}</span>
          </span>
        ))}
        <span className="size-unit">{units}</span>
      </div>
      <div className="size-foot">
        <Segmented size="sm" ariaLabel="Size along" value={frame} onChange={f => { useStore.getState().setDisplay({ sizeFrame: f }); if (drawn) toggleLines(true, f); }} options={[{ value: 'part', label: 'Part', title: 'Length × width × height along the part itself — what calipers would measure' }, { value: 'world', label: 'XYZ', title: 'Extents along the scanner’s coordinate axes (changes when the part is rotated)' }]} />
        <span className="truncate" title={note}>{note}</span>
        <span className="spacer" />
        <IconButton size="sm" label={drawn ? 'Hide the dimension lines' : 'Draw the dimensions on the model'} active={drawn} onClick={() => toggleLines(!drawn)}>
          <Ruler size={15} />
        </IconButton>
      </div>
    </div>
  );
}
