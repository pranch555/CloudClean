import { useEffect, useState } from 'react';
import { CircleDashed, Eye, Loader2 } from 'lucide-react';
import { api, ApiError } from '../lib/api';
import { fmtLen } from '../lib/format';
import { submitJob } from '../lib/jobs';
import { uid } from '../lib/uid';
import type { Asset } from '../lib/types';
import { useStore, type Vec3 } from '../store';
import { getViewer } from '../viewer/instance';
import { Button, NumberInput } from '../ui/primitives';
import { Block } from './StepFrame';

interface Hole {
  id: number;
  vertices: number;
  perimeter: number;
  diameter: number;
  area: number;
  center: Vec3;
  normal: Vec3;
  outer: boolean;
}

interface HolesResult {
  holes: Hole[];
  count: number;
  holes_to_fill: number;
  boundary_edges: number;
  watertight: boolean;
}

/** Holes of a mesh (Revo Metro's hole detection): what is missing, where, and filling only what you choose. */
export function HolesSection({ mesh }: { mesh: Asset }) {
  const units = useStore(s => s.display.units);
  const [res, setRes] = useState<HolesResult | null | 'loading' | 'unsupported'>('loading');
  const [ticked, setTicked] = useState<number[]>([]);
  const [maxD, setMaxD] = useState(2);

  useEffect(() => {
    let alive = true;
    setRes('loading');
    setTicked([]);
    api.get<HolesResult>(`/api/assets/${mesh.id}/holes`)
      .then(r => alive && setRes(r))
      .catch(err => alive && setRes(err instanceof ApiError && (err.status === 404 || err.status === 405) && /not found/i.test(err.message) && !/asset/i.test(err.message) ? 'unsupported' : null));
    return () => {
      alive = false;
    };
  }, [mesh.id]);

  if (res === 'unsupported' || res === null) return null;

  const show = (h: Hole) => {
    const v = getViewer();
    if (!v) return;
    v.lookAtPoint(h.center);
    setTimeout(() => v.zoomBy(2.5), 420);
    useStore.setState(s => ({ annotations: [...s.annotations.filter(a => !a.id.startsWith('hole-')), { id: `hole-${uid()}`, position: h.center, text: `${h.outer ? 'Open edge' : 'Hole'} · about ${fmtLen(h.diameter, 2)} ${units} across` }] }));
  };

  const fill = (body: Record<string, unknown>) =>
    submitJob('/api/holes/fill', { asset_id: mesh.id, ...body }, job => {
      if (job.status === 'done') useStore.setState(s => ({ annotations: s.annotations.filter(a => !a.id.startsWith('hole-')) }));
    });

  if (res === 'loading') {
    return (
      <Block guide="mesh.holes" title="Holes">
        <p className="hint-text row"><Loader2 size={15} className="spin" aria-hidden /> Looking for holes…</p>
      </Block>
    );
  }

  const holes = res.holes.filter(h => !h.outer);
  const outer = res.holes.find(h => h.outer);
  const largest = holes[0];
  const underMax = holes.filter(h => h.diameter <= maxD).length;

  return (
    <Block guide="mesh.holes" title="Holes" aside={res.watertight ? <span className="badge badge-good">watertight</span> : <span className="badge">{res.count} found</span>}>
      {res.watertight ? (
        <p className="hint-text">No holes: the surface is closed all the way round.</p>
      ) : (
        <>
          <p className="hint-text">
            {holes.length > 0 && <>{holes.length} gap{holes.length === 1 ? '' : 's'} in the surface{largest ? `, the largest about ${fmtLen(largest.diameter, 2)} ${units} across` : ''}. </>}
            {outer && <>The open edge where the scan ends (about {fmtLen(outer.diameter, 1)} {units} across) is left as it is.</>}
            {' '}Filled surface is not scanned — the report of the new mesh says how much was added.
          </p>
          {holes.length > 0 && (
            <ul className="plain-list hole-rows">
              {holes.slice(0, 6).map(h => (
                <li key={h.id} className="item-row">
                  <input type="checkbox" className="check" checked={ticked.includes(h.id)} aria-label={`Choose hole ${h.id + 1}`} onChange={e => setTicked(t => (e.target.checked ? [...t, h.id] : t.filter(x => x !== h.id)))} />
                  <CircleDashed size={16} aria-hidden className="muted" />
                  <div className="grow">
                    <span className="strong mono">⌀ {fmtLen(h.diameter, 2)} {units}</span>
                    <span className="caption">{h.vertices} edge points · {fmtLen(h.area, 2)} {units}²</span>
                  </div>
                  <Button size="sm" variant="ghost" icon={<Eye size={14} />} onClick={() => show(h)}>Show</Button>
                </li>
              ))}
              {holes.length > 6 && <li className="caption" style={{ paddingLeft: 12 }}>…and {holes.length - 6} smaller</li>}
            </ul>
          )}
          {holes.length > 0 && (
            <div className="stack tight">
              <div className="row wrap">
                <span className="small">Fill every hole up to</span>
                <NumberInput value={maxD} step={0.5} min={0.05} unit={units} width={112} label="Largest hole to fill" onChange={setMaxD} />
                <Button size="sm" variant="primary" disabled={!underMax} onClick={() => fill({ max_diameter: maxD })}>Fill {underMax}</Button>
              </div>
              {ticked.length > 0 && <Button size="sm" onClick={() => fill({ hole_ids: ticked })}>Fill the {ticked.length} ticked hole{ticked.length === 1 ? '' : 's'}</Button>}
            </div>
          )}
        </>
      )}
    </Block>
  );
}
