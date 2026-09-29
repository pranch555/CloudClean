import type { Asset } from '../lib/types';
import { fmtDims, fmtLen, fmtPct, humanize } from '../lib/format';
import { useStore } from '../store';
import { Badge, Section } from '../ui/primitives';

/* eslint-disable @typescript-eslint/no-explicit-any */
export function ReportView({ asset: a, report: r }: { asset: Asset; report: Record<string, any> }) {
  const units = useStore(s => s.display.units);
  switch (a.operation) {
    case 'clean':
      return (
        <Section title="Cleaning report">
          <p className="hint-text">
            Kept <b className="mono">{r.output_points?.toLocaleString()}</b> of <span className="mono">{r.input_points?.toLocaleString()}</span> points ({fmtPct(r.removed_fraction)} removed)
          </p>
          <table className="kv">
            <tbody>
              {(r.steps ?? []).map((st: any, i: number) => (
                <tr key={i}><td>{humanize(st.step)}</td><td className="mono">−{st.removed.toLocaleString()}</td></tr>
              ))}
            </tbody>
          </table>
        </Section>
      );
    case 'merge':
      return (
        <Section title="Alignment report">
          <table className="data-table">
            <thead><tr><th>Scan</th><th>Overlap</th><th>RMSE</th><th>Colour</th></tr></thead>
            <tbody>
              {(r.scans ?? []).map((sc: any) => (
                <tr key={sc.index}>
                  <td>{sc.reference ? `${sc.index + 1} (ref)` : sc.index + 1}{sc.ambiguous && <Badge tone="warning" title="A different orientation fits almost as well">symmetric?</Badge>}</td>
                  <td className="mono">{sc.fitness != null ? fmtPct(sc.fitness, 0) : '–'}</td>
                  <td className="mono">{sc.rmse != null ? fmtLen(sc.rmse) : '–'}</td>
                  <td className="mono">{sc.color_agreement != null ? fmtPct(sc.color_agreement, 0) : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {(r.scans ?? []).filter((sc: any) => sc.warning).map((sc: any) => (
            <p key={sc.index} className="warn-text">Scan {sc.index + 1}: {sc.warning}</p>
          ))}
        </Section>
      );
    case 'mesh': {
      const d = r.deviation ?? {};
      return (
        <Section title="Accuracy vs scan points">
          <table className="kv">
            <tbody>
              <tr><td>Mean deviation</td><td className="mono">{fmtLen(d.mean)} {units}</td></tr>
              <tr><td>95% of points within</td><td className="mono">{fmtLen(d.p95)} {units}</td></tr>
              <tr><td>Max deviation</td><td className="mono">{fmtLen(d.max)} {units}</td></tr>
              {r.depth && <tr><td>Poisson depth</td><td className="mono">{r.depth}</td></tr>}
              <tr><td>Trim</td><td>{r.trim ?? '–'}</td></tr>
              {r.cloud_dimensions && <tr><td>Scan size</td><td className="mono">{fmtDims(r.cloud_dimensions)}</td></tr>}
            </tbody>
          </table>
        </Section>
      );
    }
    case 'texture':
      return (
        <Section title="Colour report">
          <table className="kv">
            <tbody>
              <tr><td>Photo coverage</td><td className="mono">{fmtPct(r.coverage)}</td></tr>
              {(r.views ?? []).map((v: any, i: number) => <tr key={i}><td>{v.image}</td><td className="mono">{v.visible_vertices.toLocaleString()} vertices</td></tr>)}
            </tbody>
          </table>
          {r.texture_error && <p className="err-text">UV texture failed: {r.texture_error}</p>}
        </Section>
      );
    case 'edit':
      return (
        <Section title="Edit report">
          <table className="kv">
            <tbody>
              {(r.ops ?? []).map((o: any, i: number) => (
                <tr key={i}>
                  <td>{i + 1}. {humanize(o.op)}</td>
                  <td className="mono">
                    {o.moved
                      ? `moved mean ${fmtLen(o.moved.mean, 3)} · 95% ${fmtLen(o.moved.p95, 3)} · max ${fmtLen(o.moved.max, 3)} ${units}`
                      : o.before != null ? `${o.before.toLocaleString()} → ${o.after.toLocaleString()}` : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      );
    case 'compare': {
      const st = r.stats;
      if (!st) return null;
      return (
        <Section title="Inspection summary">
          <table className="kv">
            <tbody>
              <tr><td>Within ±{fmtLen(r.tolerance)} {units}</td><td className="mono">{st.within_tolerance_pct != null ? `${Number(st.within_tolerance_pct).toFixed(1)}%` : '–'}</td></tr>
              <tr><td>Mean / RMS</td><td className="mono">{fmtLen(st.mean)} / {fmtLen(st.rms)}</td></tr>
              <tr><td>P5 … P95</td><td className="mono">{fmtLen(st.p05)} … {fmtLen(st.p95)}</td></tr>
            </tbody>
          </table>
          <button type="button" className="link" onClick={() => useStore.getState().goStep('measure')}>Open in Measure</button>
        </Section>
      );
    }
    default:
      return null;
  }
}
