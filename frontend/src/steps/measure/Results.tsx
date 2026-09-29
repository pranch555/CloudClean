import { ClipboardList, Download, Focus, Trash2 } from 'lucide-react';
import { AssistantGlyph } from '../../features/assistant/AssistantMark';
import { fmtLen } from '../../lib/format';
import { csvOfMeasurements, isSizeLine, measureColor } from '../../lib/measure';
import { downloadText, showOnModel, useDimInfo } from '../../lib/measureTools';
import { useStore, type Measurement, type Vec3 } from '../../store';
import { Button, IconButton, NumberInput } from '../../ui/primitives';
import { Block } from '../StepFrame';
import { kindGlyph } from './glyphs';
import { dimText, prefixOf, unitText, valueText } from './format';
import { CopyButton } from './parts';

interface Row {
  id: string;
  label: string;
  kind: string;
  title: string;
  value: number | null;
  unit: string;
  detail: string;
  warn: boolean;
  source: 'points' | 'tool' | 'assistant';
  a: Vec3;
  b: Vec3;
  assetId?: string | null;
  color?: string;
  m?: Measurement;
}

const labelNumber = (l: string) => {
  const m = /^D(\d+)$/.exec(l);
  return m ? Number(m[1]) : Number.POSITIVE_INFINITY;
};

/** Every measurement on the model in one list: two-point distances, tool results and the assistant's dimensions. */
export function ResultsList() {
  const measurements = useStore(s => s.measurements);
  const dims = useStore(s => s.dims);
  const axis = useStore(s => s.measureAxis);
  const units = useStore(s => s.display.units);
  const info = useDimInfo(s => s.byId);

  const axisName = axis === 'thread' ? 'the thread axis' : axis.toUpperCase();
  const rows: Row[] = [];
  measurements.forEach((m, i) => {
    if (!m.b) return;
    const r = m.result;
    const along = axis !== 'none' && r?.along_axis != null;
    rows.push({
      id: m.id,
      label: m.label,
      kind: 'distance',
      title: along ? `Point to point · along ${axisName}` : 'Point to point',
      value: r ? (along ? r.along_axis! : r.distance) : null,
      unit: units,
      detail: !r ? 'measuring…' : (r.uncertainty != null ? `±${fmtLen(r.uncertainty, 3)} (1σ) · ` : '') + (along ? `direct ${fmtLen(r.distance, 3)} · sideways ${fmtLen(r.perpendicular, 3)}` : `ΔX ${fmtLen(r.dx, 3)} · ΔY ${fmtLen(r.dy, 3)} · ΔZ ${fmtLen(r.dz, 3)}`),
      warn: false,
      source: 'points',
      a: m.a,
      b: m.b,
      assetId: m.assetId,
      color: measureColor(i),
      m,
    });
  });
  for (const d of dims) {
    if (isSizeLine(d.id)) continue;
    const meta = info[d.id];
    rows.push({
      id: d.id,
      label: d.label,
      kind: d.kind,
      title: meta?.title ?? (d.source === 'assistant' ? d.label : d.kind),
      value: d.value,
      unit: d.unit,
      detail: meta?.detail ?? '',
      warn: !!meta?.warnings.length,
      source: d.source === 'assistant' ? 'assistant' : 'tool',
      a: d.a,
      b: d.b,
      assetId: d.assetId,
    });
  }
  rows.sort((x, y) => labelNumber(x.label) - labelNumber(y.label));

  const remove = (row: Row) => {
    if (row.source === 'points') useStore.setState(s => ({ measurements: s.measurements.filter(x => x.id !== row.id) }));
    else useStore.setState(s => ({ dims: s.dims.filter(x => x.id !== row.id) }));
  };

  const clearAll = () => {
    const st = useStore.getState();
    const keptMeasurements = st.measurements;
    const keptDims = st.dims.filter(d => !isSizeLine(d.id));
    useStore.setState(s => ({ measurements: [], dims: s.dims.filter(d => isSizeLine(d.id)) }));
    st.toast({
      kind: 'info',
      title: `Cleared ${rows.length} measurement${rows.length === 1 ? '' : 's'}`,
      action: { label: 'Undo', run: () => useStore.setState(s => ({ measurements: [...keptMeasurements, ...s.measurements], dims: [...s.dims, ...keptDims] })) },
    });
  };

  const text = () =>
    rows.map(r => [r.label, r.title, `${prefixOf(r.kind)}${valueText(r)}`, unitText(r)].join('\t')).join('\n');

  const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');

  return (
    <Block
      guide="measure.results"
      title={<>Measurements {rows.length > 0 && <span className="meas-count mono">{rows.length}</span>}</>}
      aside={
        rows.length > 0 && (
          <span className="meas-aside">
            <CopyButton text={text()} label="Copy all as text" icon={<ClipboardList size={15} aria-hidden />} />
            <Button size="sm" variant="ghost" icon={<Download size={14} />} onClick={() => downloadText(csvOfMeasurements(units), `measurements-${stamp}.csv`)}>
              CSV
            </Button>
            <Button size="sm" variant="ghost" onClick={clearAll}>Clear</Button>
          </span>
        )
      }
    >
      {rows.length === 0 ? (
        <div className="meas-empty">
          <svg viewBox="0 0 120 40" aria-hidden className="meas-empty-art">
            <path d="M8 8v24M112 8v24M8 20h104" />
            <path d="m16 15-8 5 8 5M104 15l8 5-8 5" />
          </svg>
          <p>Nothing measured yet. Pick a tool above: every result is drawn on the model and collected here, ready to copy or save as CSV.</p>
        </div>
      ) : (
        <ul className="meas-list">
          {rows.map(r => (
            <ResultRow key={r.id} row={r} onRemove={() => remove(r)} />
          ))}
        </ul>
      )}
    </Block>
  );
}

function ResultRow({ row: r, onRemove }: { row: Row; onRemove: () => void }) {
  const axis = useStore(s => s.measureAxis);
  const Glyph = kindGlyph(r.kind);
  const along = r.m?.result?.along_axis;
  const pitchHelper = r.source === 'points' && axis !== 'none' && along != null;
  const crests = r.m?.crests ?? 1;
  return (
    <li className={`meas-row ${r.warn ? 'has-warn' : ''}`}>
      <span className="meas-tag" style={r.color ? { background: r.color } : undefined}>{r.label}</span>
      <div className="meas-row-main">
        <div className="meas-row-top">
          <span className="meas-kind" aria-hidden><Glyph size={15} /></span>
          <span className="meas-title truncate" title={r.title}>{r.title}</span>
          {r.source === 'assistant' && <span className="meas-by" title="Measured by the assistant"><AssistantGlyph size={14} /> assistant</span>}
          <span className="meas-value" aria-label={dimText({ kind: r.kind, value: r.value, unit: r.unit })}>
            {prefixOf(r.kind)}{valueText(r)}<span className="meas-unit">{unitText(r)}</span>
          </span>
        </div>
        <div className="meas-row-sub">
          <span className="meas-detail truncate" title={r.detail}>{r.detail}{r.m && !r.m.snapped && r.m.result ? ' · on screen points' : ''}</span>
          <span className="meas-actions">
            <IconButton size="sm" label="Show it on the model" tip="top" onClick={() => showOnModel(r.a, r.b, r.assetId)}>
              <Focus size={15} />
            </IconButton>
            <CopyButton text={`${valueText(r)} ${unitText(r)}`} />
            <IconButton size="sm" label={`Delete ${r.label}`} tip="top" className="meas-delete" onClick={onRemove}>
              <Trash2 size={15} />
            </IconButton>
          </span>
        </div>
        {pitchHelper && (
          <div className="meas-pitch">
            <span>Spans</span>
            <NumberInput value={crests} step={1} min={1} width={70} onChange={n => useStore.setState(s => ({ measurements: s.measurements.map(x => (x.id === r.id ? { ...x, crests: Math.max(1, Math.round(n)) } : x)) }))} />
            <span>{crests === 1 ? 'pitch' : 'pitches'} → pitch</span>
            <b className="mono">{fmtLen(along! / crests, 4)}</b>
            <span className="muted">{r.unit}</span>
          </div>
        )}
      </div>
    </li>
  );
}

