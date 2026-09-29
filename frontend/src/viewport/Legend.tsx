import { useStore } from '../store';
import { gradientCss } from '../viewer/colormaps';
import { fmtSigned, fmtLen } from '../lib/format';
import { NumberInput, Popover, Segmented } from '../ui/primitives';

/** Colour legend for data fields (deviation from CAD, scan density, …) with editable range and tolerance. */
export function Legend() {
  const display = useStore(s => s.display);
  const setDisplay = useStore(s => s.setDisplay);
  const sc = display.scalar;
  if (display.colorMode !== 'scalar' || !sc) return null;
  const st = sc.style;
  const diverging = st.kind === 'diverging';
  const lo = diverging ? -Math.max(Math.abs(st.min), Math.abs(st.max)) : st.min;
  const hi = diverging ? -lo : st.max;
  const ticks = diverging ? [lo, -st.tolerance, 0, st.tolerance, hi] : [lo, (lo + hi) / 2, hi];
  const update = (patch: Partial<typeof st>) => setDisplay({ scalar: { ...sc, style: { ...st, ...patch } } });
  const title = sc.name === 'deviation' ? 'Deviation from CAD' : sc.name === 'reference_distance' ? 'Distance to nearest scan point' : sc.name === 'density' ? 'Scan density' : sc.name;

  return (
    <div className="hud hud-legend" aria-label={`${title} colour legend`}>
      <div className="legend-head">
        <span className="legend-title">{title}</span>
        <Popover
          side="top"
          align="end"
          trigger={({ toggle }) => (
            <button type="button" className="link" onClick={toggle}>
              Adjust
            </button>
          )}
        >
          <div className="popover-body legend-form">
            {diverging ? (
              <>
                <label className="mini-field"><span>Tolerance ±</span><NumberInput value={st.tolerance} step={0.01} min={0} unit={display.units} onChange={tolerance => update({ tolerance })} /></label>
                <label className="mini-field"><span>Colour range ±</span><NumberInput value={hi} step={0.05} min={0.0001} unit={display.units} onChange={v => update({ min: -v, max: v })} /></label>
              </>
            ) : (
              <>
                <label className="mini-field"><span>Min</span><NumberInput value={st.min} step="any" onChange={min => update({ min })} /></label>
                <label className="mini-field"><span>Max</span><NumberInput value={st.max} step="any" onChange={max => update({ max })} /></label>
              </>
            )}
            <label className="mini-field"><span>Bands</span>
              <Segmented size="sm" value={String(st.steps)} onChange={v => update({ steps: Number(v) })} options={[{ value: '0', label: 'Smooth' }, { value: '8', label: '8' }, { value: '12', label: '12' }]} />
            </label>
          </div>
        </Popover>
      </div>
      <div className="legend-wrap">
        <div className="legend-bar" style={{ background: gradientCss(st) }} />
        {diverging && <div className="legend-band" style={{ left: `${((-st.tolerance - lo) / (hi - lo)) * 100}%`, width: `${((2 * st.tolerance) / (hi - lo)) * 100}%` }} aria-hidden />}
      </div>
      <div className="legend-ticks mono">
        {ticks.map((t, i) => (
          <span key={i} style={{ left: `${((t - lo) / (hi - lo || 1)) * 100}%` }}>
            {diverging ? (t === 0 ? '0' : fmtSigned(t, st.tolerance < 0.01 ? 3 : 2)) : fmtLen(t)}
          </span>
        ))}
      </div>
      {diverging && (
        <div className="legend-keys">
          <span><i className="swatch neg" /> below surface</span>
          <span><i className="swatch mid" /> in tolerance</span>
          <span><i className="swatch pos" /> above surface</span>
        </div>
      )}
    </div>
  );
}
