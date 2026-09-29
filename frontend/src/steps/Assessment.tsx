import type { ReactNode } from 'react';
import { AlertTriangle, CheckCircle2, CopyCheck, Info } from 'lucide-react';
import { fmtLen } from '../lib/format';

export interface ScanAssessment {
  index: number;
  name?: string;
  points?: number;
  noise_mm?: number;
  fitness?: number;
  rmse_over_spacing?: number;
  overlap?: number;
  coverage_gain_pct?: number;
  coverage_gain_mm2?: number;
  layering_mm?: number;
  separation_ratio?: number;
  doubled_surface?: boolean;
  ambiguous?: boolean;
  alternative_angle?: number;
  already_aligned?: boolean;
  recommendation?: string;
}

export interface Assessment {
  recommendation: 'single' | 'merge' | 'use_best' | 'ask';
  best_index: number | null;
  reasons: string[];
  scans: ScanAssessment[];
  transforms?: number[][][];
}

const HEADLINE: Record<string, { title: string; tone: 'pass' | 'warn' | 'fail'; icon: typeof Info }> = {
  merge: { title: 'Merging will help', tone: 'pass', icon: CheckCircle2 },
  use_best: { title: 'Merging is not needed', tone: 'warn', icon: CopyCheck },
  ask: { title: 'Check before merging', tone: 'fail', icon: AlertTriangle },
  single: { title: 'Only one scan', tone: 'pass', icon: Info },
};

// Values may be fractions (0..1) or percentages; show percentages either way.
const pct = (v?: number) => (v == null ? '–' : `${(v <= 1.0001 ? v * 100 : v).toFixed(v <= 0.1 ? 1 : 0)}%`);

/** Result of the pre-merge check: headline, plain-language reasons and a per-scan table, plus the decision buttons. */
export function AssessmentView({ a, names, actions }: { a: Assessment; names?: string[]; actions?: ReactNode }) {
  const h = HEADLINE[a.recommendation] ?? HEADLINE.ask;
  const scans = a.scans.filter(s => s.index > 0);
  const nameOf = (s: ScanAssessment) => s.name ?? names?.[s.index] ?? `Scan ${s.index + 1}`;
  return (
    <div className="assess-card">
      <div className={`verdict verdict-${h.tone}`}>
        <h.icon size={20} aria-hidden />
        <div>
          <div className="verdict-title">{h.title}</div>
          {a.best_index != null && a.recommendation !== 'merge' && (
            <div className="verdict-sub">Best single scan: <b>{names?.[a.best_index] ?? a.scans.find(s => s.index === a.best_index)?.name ?? `scan ${a.best_index + 1}`}</b></div>
          )}
        </div>
      </div>
      <ul className="reason-list">
        {a.reasons.map((r, i) => (
          <li key={i}>
            <Info size={13} aria-hidden /> {r}
          </li>
        ))}
      </ul>
      {scans.length > 0 && (
        <table className="data-table assess-table">
          <thead>
            <tr><th>vs reference</th><th title="Surface this scan adds that the others do not have">New surface</th><th title="Share of points that land on the other scan after alignment">Overlap</th><th title="Gap between the two surfaces where they overlap, relative to scanner noise (above 1.5 = doubled skin)">Layering</th></tr>
          </thead>
          <tbody>
            {scans.map(s => (
              <tr key={s.index}>
                <td>
                  {nameOf(s)}
                  {s.ambiguous && <span className="badge badge-warning" title={`A pose rotated ${s.alternative_angle ?? '?'}° fits almost as well`}>symmetric?</span>}
                  {s.doubled_surface && <span className="badge badge-critical" title={`Surfaces are ${fmtLen(s.layering_mm, 3)} mm apart where they overlap`}>doubled</span>}
                </td>
                <td>{pct(s.coverage_gain_pct)}</td>
                <td>{pct(s.overlap)}</td>
                <td className={s.doubled_surface ? 'delta-warn' : ''}>{s.separation_ratio != null ? `${s.separation_ratio.toFixed(2)}×` : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {actions && <div className="decision-row">{actions}</div>}
    </div>
  );
}
