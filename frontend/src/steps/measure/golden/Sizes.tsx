import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown, ChevronDown, Crosshair, Lightbulb, MousePointer2, TriangleAlert } from 'lucide-react';
import { fmtLen } from '../../../lib/format';
import type { GoldenDimension, GoldenMeasurement, GoldenReport, GoldenSection, Vec3 } from '../../../lib/golden';
import { animate, draw, enter, reducedMotion, springy, EASE } from '../../../lib/motion';
import { Button } from '../../../ui/primitives';
import { useReveal } from './useSeen';
import { MEAS_STATUS, differenceWords, groupOf, plural, signedDiff, storyOf } from './words';

/* Measure -> Golden model -> Sizes: every size the golden model is built to, measured again on the scan.
   A drawing of the part (a cut along its axis) shows where the size is taken; pointing at a size slides the
   dimension line there, and lights the same faces with a dimension line on the 3D model. */

type Tone = GoldenDimension['tone'];

/** Rows: a size, or several equal sizes of one feature shown as one (the three widths of a hex socket). */
interface Row {
  key: string;
  main: GoldenMeasurement;
  all: GoldenMeasurement[];
}

const toneOf = (m: GoldenMeasurement): Tone => MEAS_STATUS[m.status].tone;
const worst = (ms: GoldenMeasurement[]) => ['off', 'close', 'ok', 'not_measured'].map(s => ms.find(m => m.status === s)).find(Boolean) ?? ms[0];

export function dimensionOf(m: GoldenMeasurement, units: string): GoldenDimension | null {
  if (!m.ends) return null;
  const label = m.status === 'not_measured' ? `${fmtLen(m.golden, 2)} ${units} · not measured` : `${fmtLen(m.golden, 2)} → ${fmtLen(m.scan, 2)} ${units}`;
  return { ends: m.ends, label, tone: toneOf(m) };
}

function buildRows(ms: GoldenMeasurement[]): { group: string; rows: Row[] }[] {
  const groups = new Map<string, Row[]>();
  const series = new Map<string, Row>();
  for (const m of ms) {
    const g = groupOf(m);
    if (!groups.has(g)) groups.set(g, []);
    if (m.series) {
      const s = series.get(m.series);
      if (s) {
        s.all.push(m);
        s.main = worst(s.all);
        continue;
      }
      const row = { key: `s:${m.series}`, main: m, all: [m] };
      series.set(m.series, row);
      groups.get(g)!.push(row);
      continue;
    }
    groups.get(g)!.push({ key: `m:${m.id}`, main: m, all: [m] });
  }
  const order = (g: string) => (g === 'Overall size' ? 0 : 1);
  return [...groups.entries()].sort((a, b) => order(a[0]) - order(b[0])).map(([group, rows]) => ({ group, rows }));
}

/* ------------------------------------------------------------------ the drawing */

interface Geom {
  e1: [number, number, number, number];
  e2: [number, number, number, number];
  d: [number, number, number, number];
  lx: number;
  ly: number;
}

function useDrawingFrame(section: GoldenSection, wideAtPlus: boolean) {
  return useMemo(() => {
    const flip = wideAtPlus ? -1 : 1;
    const [x0, y0, x1, y1] = section.bounds;
    const X = (x: number) => flip * x;
    const Y = (y: number) => -y;
    const xs = [X(x0), X(x1)].sort((a, b) => a - b);
    const ys = [Y(y0), Y(y1)].sort((a, b) => a - b);
    const w = xs[1] - xs[0];
    const h = ys[1] - ys[0];
    const size = Math.max(w, h);
    const pad = size * 0.16;
    const box = { minX: xs[0], maxX: xs[1], minY: ys[0], maxY: ys[1], w, h, pad, size };
    const view = `${xs[0] - pad} ${ys[0] - pad * 1.25} ${w + 2 * pad} ${h + pad * 2.5}`;
    const project = (p: Vec3): [number, number] => {
      const o = section.origin;
      const d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
      const x = d[0] * section.u[0] + d[1] * section.u[1] + d[2] * section.u[2];
      const y = d[0] * section.v[0] + d[1] * section.v[1] + d[2] * section.v[2];
      return [X(x), Y(y)];
    };
    const path = (loop: number[]) => {
      let s = '';
      for (let i = 0; i < loop.length; i += 2) s += `${i ? 'L' : 'M'}${X(loop[i]).toFixed(2)} ${Y(loop[i + 1]).toFixed(2)}`;
      return `${s}Z`;
    };
    const seg = (s: number[]) => `M${X(s[0]).toFixed(2)} ${Y(s[1]).toFixed(2)}L${X(s[2]).toFixed(2)} ${Y(s[3]).toFixed(2)}`;
    return { box, view, project, path, seg };
  }, [section, wideAtPlus]);
}

/** Where the dimension line goes: outside the part, on the side nearest the two points, with extension lines. */
function geometry(a: [number, number], b: [number, number], box: ReturnType<typeof useDrawingFrame>['box']): Geom {
  const gap = box.pad * 0.55;
  const horizontal = Math.abs(b[0] - a[0]) >= Math.abs(b[1] - a[1]);
  if (horizontal) {
    const midY = (a[1] + b[1]) / 2;
    const above = midY <= (box.minY + box.maxY) / 2;
    const y = above ? box.minY - gap : box.maxY + gap;
    return { e1: [a[0], a[1], a[0], y], e2: [b[0], b[1], b[0], y], d: [a[0], y, b[0], y], lx: (a[0] + b[0]) / 2, ly: above ? y - gap * 0.45 : y + gap * 0.75 };
  }
  const midX = (a[0] + b[0]) / 2;
  const left = midX <= (box.minX + box.maxX) / 2;
  const x = left ? box.minX - gap : box.maxX + gap;
  return { e1: [a[0], a[1], x, a[1]], e2: [b[0], b[1], x, b[1]], d: [x, a[1], x, b[1]], lx: x, ly: Math.min(a[1], b[1]) - gap * 0.45 };
}

function Drawing({ checkKey, section, wideAtPlus, focus, units }: { checkKey: string; section: GoldenSection; wideAtPlus: boolean; focus: GoldenMeasurement | null; units: string }) {
  const f = useDrawingFrame(section, wideAtPlus);
  const ext1 = useRef<SVGLineElement>(null);
  const ext2 = useRef<SVGLineElement>(null);
  const dim = useRef<SVGLineElement>(null);
  const label = useRef<SVGGElement>(null);
  const last = useRef<Geom | null>(null);
  const [ref, seen] = useReveal<HTMLDivElement>(checkKey, root => [
    draw(root.querySelectorAll('.gb-outline'), { duration: 1300, step: 80, ease: EASE.inOut }),
    reducedMotion() ? null : animate(root.querySelectorAll('.gb-cut'), { opacity: [0, 1], duration: 700, delay: 900, ease: 'out(2)' }),
    draw(root.querySelectorAll('.gb-axis'), { duration: 900, delay: 300 }),
  ]);

  const geom = useMemo(() => {
    if (!focus?.ends) return null;
    return geometry(f.project(focus.ends[0]), f.project(focus.ends[1]), f.box);
  }, [focus, f]);

  // the dimension line slides like a caliper from the last size to this one
  useEffect(() => {
    const els = [ext1.current, ext2.current, dim.current];
    if (!geom || els.some(e => !e) || !label.current) return;
    const set = (g: Geom) => {
      const [a, b, c] = els as SVGLineElement[];
      for (const [el, v] of [[a, g.e1], [b, g.e2], [c, g.d]] as const) {
        el.setAttribute('x1', String(v[0]));
        el.setAttribute('y1', String(v[1]));
        el.setAttribute('x2', String(v[2]));
        el.setAttribute('y2', String(v[3]));
      }
      label.current!.setAttribute('transform', `translate(${g.lx} ${g.ly})`);
    };
    const from = last.current;
    last.current = geom;
    if (!from || reducedMotion()) {
      set(geom);
      return;
    }
    const flat = (g: Geom) => [...g.e1, ...g.e2, ...g.d, g.lx, g.ly];
    const keys = flat(geom).map((_, i) => `k${i}`);
    const o: Record<string, number> = Object.fromEntries(flat(from).map((v, i) => [keys[i], v]));
    const to = Object.fromEntries(flat(geom).map((v, i) => [keys[i], v]));
    const anim = animate(o, {
      ...to,
      ease: springy(),
      onUpdate: () => {
        const v = keys.map(k => o[k]);
        set({ e1: [v[0], v[1], v[2], v[3]], e2: [v[4], v[5], v[6], v[7]], d: [v[8], v[9], v[10], v[11]], lx: v[12], ly: v[13] });
      },
    });
    return () => {
      anim.pause();
    };
  }, [geom]);

  const tone = focus ? toneOf(focus) : 'none';
  const fs = f.box.size * 0.058;
  const traces = (focus?.faces ?? []).flatMap(id => section.faces[String(id)] ?? []);
  const text = focus ? (focus.status === 'not_measured' ? `${fmtLen(focus.golden, 2)} · not measured` : `${fmtLen(focus.golden, 2)} → ${fmtLen(focus.scan, 2)}`) : '';
  const hatch = f.box.size * 0.028;

  return (
    <div className={`gb ${seen ? 'is-seen' : ''}`} ref={ref}>
      <svg viewBox={f.view} className={`gb-svg gb-${tone}`} role="img" aria-label={focus ? `Drawing of the part: ${focus.name}, ${text} ${units}` : 'Drawing of the part, cut along its axis'}>
        <defs>
          <pattern id={`gb-hatch-${checkKey}`} width={hatch} height={hatch} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2={hatch} className="gb-hatch-line" />
          </pattern>
          <marker id={`gb-arrow-${checkKey}`} viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth={f.box.size * 0.045} markerHeight={f.box.size * 0.045} orient="auto-start-reverse">
            <path d="M0,1 L10,5 L0,9 z" className="gb-arrowhead" />
          </marker>
        </defs>
        <line className="gb-axis" x1={f.box.minX - f.box.pad * 0.6} y1={0} x2={f.box.maxX + f.box.pad * 0.6} y2={0} />
        <path className="gb-cut" d={section.loops.map(f.path).join('')} fill={`url(#gb-hatch-${checkKey})`} fillRule="evenodd" />
        {section.loops.map((l, i) => (
          <path key={i} className="gb-outline" d={f.path(l)} />
        ))}
        {traces.map((s, i) => (
          <path key={`${focus?.id}-${i}`} className="gb-face" d={f.seg(s)} />
        ))}
        <g className={`gb-dim ${geom ? '' : 'is-off'}`}>
          <line ref={ext1} className="gb-ext" />
          <line ref={ext2} className="gb-ext" />
          <line ref={dim} className="gb-dimline" markerStart={`url(#gb-arrow-${checkKey})`} markerEnd={`url(#gb-arrow-${checkKey})`} />
          <g ref={label}>
            <text className="gb-label" fontSize={fs} textAnchor="middle" dominantBaseline="middle">{text}</text>
          </g>
        </g>
      </svg>
      <p className="gb-caption">
        {focus ? (
          <>
            <b>{focus.name}</b>
            {focus.what && <span className="gb-what">{focus.what}</span>}
          </>
        ) : (
          <>
            <MousePointer2 size={12} aria-hidden /> Point at a size to see where it is.
          </>
        )}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ the list */

export function Sizes({
  checkKey,
  report,
  units,
  tolText,
  areaNumber,
  onPoint,
  onPin,
  onShowArea,
  onShowSize,
}: {
  checkKey: string;
  report: GoldenReport;
  units: string;
  tolText: string;
  areaNumber: (regionId: number) => number | null;
  onPoint: (m: GoldenMeasurement | null) => void;
  onPin: (m: GoldenMeasurement | null) => void;
  onShowArea: (regionId: number) => void;
  onShowSize: (m: GoldenMeasurement) => void;
}) {
  const ms = report.measurements;
  const measuredList = useMemo(() => ms.filter(m => m.status !== 'not_measured'), [ms]);
  const notMeasured = useMemo(() => ms.filter(m => m.status === 'not_measured'), [ms]);
  const groups = useMemo(() => buildRows(measuredList), [measuredList]);
  const extra = useMemo(() => buildRows(notMeasured), [notMeasured]);
  const [hover, setHover] = useState<GoldenMeasurement | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const firstOff = measuredList.find(m => m.status === 'off' && m.ends) ?? measuredList.find(m => m.ends) ?? null;
  const openRow = [...groups, ...extra].flatMap(g => g.rows).find(r => r.key === open);
  const focus = hover ?? openRow?.main ?? firstOff;
  const count = (s: GoldenMeasurement['status']) => ms.filter(m => m.status === s).length;
  const wideAtPlus = /wide|head/.test(report.part?.ends?.[1] ?? '');

  const [ref, seen] = useReveal<HTMLDivElement>(checkKey, root => [
    enter(root.querySelectorAll('.gz-tally i'), { y: 0, scale: 0.2, step: 35, delay: 100, duration: 420 }),
    enter(root.querySelectorAll('.gz-group > .gz-rows > .gz-row'), { y: 10, step: 40, delay: 200 }),
  ]);

  const point = (m: GoldenMeasurement | null) => {
    setHover(m);
    onPoint(m ?? openRow?.main ?? null);
  };
  const toggle = (r: Row) => {
    const next = open === r.key ? null : r.key;
    setOpen(next);
    onPin(next ? r.main : null);
  };

  const row = (r: Row) => {
    const m = r.main;
    const st = MEAS_STATUS[m.status];
    const isOpen = open === r.key;
    const caveatAreas = (m.regions ?? []).map(id => ({ id, n: areaNumber(id) })).filter(a => a.n != null);
    return (
      <li key={r.key} className={`gz-row gz-${st.tone} ${isOpen ? 'is-open' : ''} ${focus === m ? 'is-focus' : ''}`}>
        <button type="button" className="gz-main" aria-expanded={isOpen} onMouseEnter={() => point(m)} onFocus={() => point(m)} onClick={() => toggle(r)}>
          <span className="gz-text">
            <span className="gz-name">
              {m.name}
              {r.all.length > 1 && <span className="gz-count mono">×{r.all.length}</span>}
              {m.caveat && <TriangleAlert size={13} className="gz-caveat-icon" aria-label="read the note" />}
            </span>
            {m.what && <span className="gz-what">{m.what}</span>}
            {r.all.length > 1 && m.status !== 'not_measured' && (
              <span className="gz-series mono">{r.all.map(x => (x.scan != null ? fmtLen(x.scan, 2) : '–')).join(' · ')} <span>scanned</span></span>
            )}
          </span>
          {m.status !== 'not_measured' ? (
            <span className={`gz-diff gz-diff-${st.tone}`}>
              <b className="mono">{signedDiff(m)}</b>
              <small>{st.label}</small>
            </span>
          ) : (
            <span className="gz-diff gz-diff-none"><small>{st.label}</small></span>
          )}
          <ChevronDown size={15} className="gz-chev" aria-hidden />
        </button>
        {isOpen && (
          <div className="gz-detail">
            <p className="gz-story">{storyOf(m, units, tolText)}</p>
            {m.status !== 'not_measured' && m.kind !== 'position' && (
              <p className="gz-words">
                <b>{differenceWords(m, units)}</b>
                {m.uncertainty != null && <span> · measured to within ±{fmtLen(m.uncertainty, 3)} {units}</span>}
              </p>
            )}
            {m.caveat && <p className="gz-caveat"><TriangleAlert size={14} aria-hidden /> {m.caveat}</p>}
            <div className="gz-actions">
              {m.ends && <Button size="sm" icon={<Crosshair size={14} />} onClick={() => onShowSize(m)}>Show on the model</Button>}
              {caveatAreas.map(a => (
                <Button key={a.id} size="sm" variant="ghost" onClick={() => onShowArea(a.id)}>Area {a.n}</Button>
              ))}
              {m.status === 'not_measured' && m.region != null && areaNumber(m.region) != null && (
                <Button size="sm" variant="ghost" onClick={() => onShowArea(m.region!)}>Show the area</Button>
              )}
            </div>
          </div>
        )}
      </li>
    );
  };

  return (
    <div className={`gz ${seen ? 'is-seen' : ''}`} ref={ref} onMouseLeave={() => point(null)}>
      <p className="gz-intro">
        Every size the golden model is drawn with, measured again on your scan. <b>{count('ok')} of {measuredList.length}</b> match
        {count('off') ? <>, <b className="gz-fail">{count('off')} {count('off') === 1 ? 'is' : 'are'} off</b></> : ''}
        {count('close') ? <>, {count('close')} too close to call</> : ''}.
      </p>
      <div className="gz-tally" aria-hidden>
        {[...measuredList].sort((a, b) => ['off', 'close', 'ok'].indexOf(a.status) - ['off', 'close', 'ok'].indexOf(b.status)).map(m => (
          <i key={m.id} className={`gz-dot gz-dot-${toneOf(m)}`} title={m.name} />
        ))}
      </div>
      {report.sizes_story && (
        <div className="gz-story-box">
          <Lightbulb size={16} aria-hidden />
          <p><b>What the sizes say. </b>{report.sizes_story}</p>
        </div>
      )}
      {report.section && report.section.loops.length > 0 && <Drawing checkKey={checkKey} section={report.section} wideAtPlus={wideAtPlus} focus={focus} units={units} />}
      {groups.map(g => (
        <div key={g.group} className="gz-group">
          <h4>{g.group}</h4>
          <ul className="gz-rows">{g.rows.map(row)}</ul>
        </div>
      ))}
      {extra.length > 0 && (
        <details className="gold-more">
          <summary>
            <ArrowDown size={14} aria-hidden /> {plural(notMeasured.length, 'size')} could not be measured
          </summary>
          {extra.map(g => (
            <div key={g.group} className="gz-group">
              <h4>{g.group}</h4>
              <ul className="gz-rows">{g.rows.map(row)}</ul>
            </div>
          ))}
        </details>
      )}
      <p className="gold-fine">A size matches when it is within {tolText}, allowing for how precisely the scan pins it down. Positions are measured after lining the scan up with the golden model.</p>
    </div>
  );
}
