import { useEffect, useRef, useState } from 'react';
import { Crosshair, Eye, X } from 'lucide-react';
import { fmtLen, fmtSigned } from '../../../lib/format';
import { regionKey, type GoldenRegion, type Vec3 } from '../../../lib/golden';
import { springTo, useOnChange } from '../../../lib/motion';
import { getViewer } from '../../../viewer/instance';
import { Button } from '../../../ui/primitives';
import { AREA_STATUS, pct } from './words';

/* One area the check lists (Measure -> Golden model -> Areas to look at): a picture of the part with the area lit,
   its plain name and where to find it, what was found, and what to do about it. */

interface Thumbs {
  renderAreaThumbnail?: (assetId: string, o: { scalar: string; values: number[]; target: Vec3; from: Vec3; radius?: number; size?: number }) => Promise<string | null>;
}

const pictures = new Map<string, string>();
let queue: Promise<unknown> = Promise.resolve();

/** The picture of one area, rendered off screen once per check (one at a time, so the 3D view stays smooth). */
function useAreaPicture(checkId: string, r: GoldenRegion, enabled: boolean) {
  const key = `${checkId}/${r.id}`;
  const [src, setSrc] = useState<string | null>(() => pictures.get(key) ?? null);
  useEffect(() => {
    if (!enabled || pictures.has(key)) return;
    let alive = true;
    const make = async (attempt: number): Promise<void> => {
      const v = getViewer() as unknown as Thumbs | null;
      if (!v?.renderAreaThumbnail) return;
      const url = await v.renderAreaThumbnail(checkId, { scalar: 'check_region', values: [r.id], target: r.view.target, from: r.view.from, radius: r.view.radius, size: 176 });
      if (url) {
        pictures.set(key, url);
        if (alive) setSrc(url);
      } else if (attempt < 3 && alive) {
        // the model may still be loading into the 3D view
        await new Promise(res => window.setTimeout(res, 900));
        return make(attempt + 1);
      }
    };
    queue = queue.then(() => (alive ? make(0) : undefined)).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [key, enabled]);
  return src;
}

/** Where a deviation sits against the tolerance: the green band is ±tolerance; the marker springs to its place. */
function DeviationMeter({ value, tol, units }: { value: number; tol: number; units: string }) {
  const span = Math.max(4 * tol, Math.abs(value) * 1.25);
  const at = (x: number) => ((x + span) / (2 * span)) * 100;
  const mark = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    // arrive from the middle (zero), so the eye follows how far off it is
    if (mark.current) {
      mark.current.style.left = `${at(0)}%`;
      springTo(mark.current, { left: `${at(value)}%` }, { delay: 450 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useOnChange(value, v => {
    if (mark.current) springTo(mark.current, { left: `${at(v)}%` });
  });
  return (
    <div className="ga-meter" role="img" aria-label={`${fmtSigned(value, 2)} ${units}, tolerance ±${fmtLen(tol)} ${units}`}>
      <div className="ga-meter-track">
        <span className="ga-meter-band" style={{ left: `${at(-tol)}%`, width: `${(tol / span) * 100}%` }} />
        <span className="ga-meter-zero" style={{ left: `${at(0)}%` }} />
        <span ref={mark} className={`ga-meter-mark ${value < 0 ? 'is-in' : 'is-out'}`} style={{ left: `${at(value)}%` }} />
      </div>
      <div className="ga-meter-labels">
        <span>less material</span>
        <span className="mono">±{+tol.toFixed(4)} ok</span>
        <span>more material</span>
      </div>
    </div>
  );
}

export function AreaCard({
  checkId,
  r,
  n,
  color,
  on,
  tol,
  units,
  pictures: withPicture,
  onShow,
  onWhole,
  onPoint,
}: {
  checkId: string;
  r: GoldenRegion;
  n: number;
  color: string;
  on: boolean;
  tol: number;
  units: string;
  pictures: boolean;
  onShow: () => void;
  onWhole: () => void;
  onPoint: (id: number | null) => void;
}) {
  const key = regionKey(r);
  const src = useAreaPicture(checkId, r, withPicture);
  return (
    <li
      data-area={r.id}
      className={`ga ${on ? 'is-on' : ''}`}
      style={{ '--pin': color } as React.CSSProperties}
      onMouseEnter={() => onPoint(r.id)}
      onMouseLeave={() => onPoint(null)}
    >
      <div className={`ga-head ${withPicture ? '' : 'no-pic'}`}>
        {withPicture ? (
          <button type="button" className={`ga-pic ${src ? 'has-pic' : ''}`} onClick={onShow} aria-label={`Show area ${n} on the model`}>
            {src ? <img src={src} alt="" draggable={false} /> : <span className="ga-pic-scan" aria-hidden />}
            <span className="ga-num mono" aria-hidden>{n}</span>
          </button>
        ) : (
          <span className="ga-num ga-num-solo mono" aria-label={`Area ${n}`}>{n}</span>
        )}
        <div className="ga-title">
          <h4>{r.name}</h4>
          {r.where && <p className="ga-where">{r.where}</p>}
          <div className="ga-tags">
            <span className={`ga-tag ga-tag-${key}`}>
              <i style={{ background: color }} aria-hidden />
              {AREA_STATUS[key]}
              {r.kind === 'off' && r.deviation != null && <b className="mono">{fmtSigned(r.deviation, 2)} {units}</b>}
              {r.kind === 'rough' && r.spread != null && <b className="mono">±{fmtLen(r.spread, 2)} {units}</b>}
            </span>
            <span className="ga-size mono">{r.area_mm2 >= 10 ? r.area_mm2.toFixed(0) : r.area_mm2.toFixed(1)} mm² · {pct(r.share_pct)} %</span>
          </div>
        </div>
      </div>
      {r.kind === 'off' && r.deviation != null && <DeviationMeter value={r.deviation} tol={tol} units={units} />}
      <p className="ga-why">{r.why}</p>
      <p className="ga-todo"><span>What to do</span>{r.advice}</p>
      <div className="ga-actions">
        {on ? (
          <>
            <span className="ga-showing"><Eye size={15} aria-hidden /> Showing it on the model</span>
            <Button size="sm" variant="ghost" icon={<X size={14} />} onClick={onWhole}>Whole part</Button>
          </>
        ) : (
          <Button size="sm" icon={<Crosshair size={15} />} onClick={onShow} data-guide={n === 1 ? 'golden.show-me' : undefined}>Show me</Button>
        )}
      </div>
    </li>
  );
}
