import { useEffect, useState } from 'react';
import { Anchor, ChevronDown, CircleDot, Columns3, Combine, Crosshair, Images, Layers, ShieldCheck, Undo2 } from 'lucide-react';
import * as THREE from 'three';
import { rigidFromPairs } from '../lib/kabsch';
import type { Asset } from '../lib/types';
import { submitJob } from '../lib/jobs';
import { api } from '../lib/api';
import { fmtLen } from '../lib/format';
import { roleOf } from '../lib/journey';
import { useProjectAssets, useStore } from '../store';
import { getViewer } from '../viewer/instance';
import { defaultsOf, ParamForm } from '../ui/ParamForm';
import { Badge, Button, IconButton } from '../ui/primitives';
import { Thumb } from '../shell/ModelsPanel';
import { AssessmentView, type Assessment } from './Assessment';
import { Block, NextStepButton, StepFrame } from './StepFrame';
import { AlignGlyph } from '../ui/icons';
import { sendToAssistant, useAssistant } from '../features/assistant/assistantStore';

const pairKey = (movId: string, refId: string) => `${movId}>${refId}`;

async function startPairing(mov: Asset, ref: Asset) {
  const st = useStore.getState();
  const v = getViewer();
  if (!v) return;
  const saved = st.savedPairs[pairKey(mov.id, ref.id)] ?? [];
  useStore.setState({ pairing: { movId: mov.id, refId: ref.id, list: saved, pending: {}, prevVisible: st.visible }, visible: [ref.id, mov.id], tool: 'navigate' });
  await v.sync([ref, mov], st.activeId);
  const center = (s: Asset['stats']) => s.bbox_min.map((x, i) => (x + s.bbox_max[i]) / 2);
  const cr = center(ref.stats), cm = center(mov.stats);
  const gap = 0.2 * Math.max(ref.stats.diagonal, mov.stats.diagonal);
  v.setOffset(mov.id, [ref.stats.bbox_max[0] - mov.stats.bbox_min[0] + gap, cr[1] - cm[1], cr[2] - cm[2]]);
  v.fit([ref.id, mov.id]);
}

export function stopPairing() {
  const P = useStore.getState().pairing;
  if (!P) return;
  getViewer()?.clearOffsets();
  getViewer()?.setMarkers([], [], 'pairs');
  useStore.setState(s => ({ pairing: null, visible: P.prevVisible, savedPairs: { ...s.savedPairs, [pairKey(P.movId, P.refId)]: P.list } }));
}

export function AlignStep() {
  const params = useStore(s => s.params)!;
  const pairing = useStore(s => s.pairing);
  const savedPairs = useStore(s => s.savedPairs);
  const selected = useStore(s => s.selected);
  const byId = useStore(s => s.byId);
  const panes = useStore(s => s.panes);
  const project = useProjectAssets();
  const list = selected.map(id => byId.get(id)).filter((a): a is Asset => !!a && a.kind !== 'image');
  const [values, setValues] = useState(() => defaultsOf(params.schema.merge));
  const ref = list[0];
  const [assessment, setAssessment] = useState<{ key: string; a: Assessment } | null>(null);
  const [checking, setChecking] = useState(false);
  const key = list.map(a => a.id).join(',');
  const current = assessment?.key === key ? assessment.a : null;
  const candidates = project.filter(a => a.kind === 'pointcloud' && ['scan', 'cleaned', 'edited', 'fromPhotos'].includes(roleOf(a)));

  const pairsPayload = () => {
    const pairs: Record<number, { source: number[][]; target: number[][] }> = {};
    list.forEach((a, i) => {
      const p = savedPairs[pairKey(a.id, ref.id)] ?? [];
      if (i > 0 && p.length >= 3) pairs[i] = { source: p.map(x => x.source), target: p.map(x => x.target) };
    });
    return Object.keys(pairs).length ? pairs : null;
  };

  const check = async () => {
    setChecking(true);
    const job = await submitJob('/api/merge/assess', { asset_ids: list.map(a => a.id), params: values }, done => {
      setChecking(false);
      const a = done.output?.assessment as Assessment | undefined;
      if (done.status === 'done' && a) setAssessment({ key, a });
    });
    if (!job) setChecking(false);
  };

  // merging reuses exactly the alignment that was checked, unless point pairs were picked
  const run = () => {
    const pairs = pairsPayload();
    const checked = !pairs && current?.transforms;
    // the check's verdict travels with the merge so measurements on it can say when the scans disagreed
    submitJob('/api/merge', { asset_ids: list.map(a => a.id), params: values, pairs, transforms: checked ? current.transforms : null, assessment: checked ? { ...current, transforms: undefined } : null });
  };

  const compare = (kind: 'raw' | 'auto' | 'pairs') => {
    const ids = list.map(a => a.id);
    let transforms: Record<string, number[]> | undefined;
    if (kind === 'auto' && current?.transforms) {
      transforms = {};
      list.forEach((a, i) => {
        const m = current.transforms![i];
        if (m) transforms![a.id] = new THREE.Matrix4().set(...(m.flat() as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number])).toArray();
      });
    } else if (kind === 'pairs') {
      transforms = {};
      list.forEach((a, i) => {
        if (i === 0) return;
        const p = savedPairs[pairKey(a.id, ref.id)] ?? [];
        const m = rigidFromPairs(p.map(x => x.source), p.map(x => x.target));
        if (m) transforms![a.id] = m.toArray();
      });
    }
    const label = kind === 'auto' ? 'Preview: checked alignment' : kind === 'pairs' ? 'Preview: your point pairs' : 'All scans as captured';
    useStore.setState({ visible: ids, panes: [...list.map(a => ({ id: a.id, label: a.name, assetIds: [a.id] })), { id: 'result', label, assetIds: ids, transforms, tone: 'result' as const }] });
    setTimeout(() => getViewer()?.fit(ids), 80);
  };

  const pairsReady = list.slice(1).some(a => (savedPairs[pairKey(a.id, ref.id)]?.length ?? 0) >= 3);
  const useBest = () => {
    if (current?.best_index == null) return;
    const best = list[current.best_index];
    useStore.setState({ visible: [best.id], activeId: best.id, selected: [] });
    useStore.getState().goStep('mesh');
    useStore.getState().toast({ kind: 'ok', title: `Continuing with ${best.name}`, body: 'The mesh will be built from this scan alone.' });
  };

  return (
    <StepFrame
      step="align"
      purpose="Combine several scans of the same part. CloudClean checks first: merging only helps when each scan adds surface the others miss and they line up exactly."
      footer={
        <>
          {current ? (
            <Button size="lg" block data-guide="align.check" icon={<ShieldCheck size={17} />} loading={checking} disabled={list.length < 2 || !!pairing} onClick={check}>Check again</Button>
          ) : (
            <Button variant="primary" size="lg" block data-guide="align.check" icon={<ShieldCheck size={17} />} loading={checking} disabled={list.length < 2 || !!pairing} onClick={check}>
              {list.length >= 2 ? `Check ${list.length} scans` : 'Pick two or more scans'}
            </Button>
          )}
          <NextStepButton from="align" />
        </>
      }
    >
      <Block title="Scans to combine" guide="align.scans" aside={candidates.length > 1 && list.length < 2 ? <button type="button" className="link" onClick={() => useStore.setState({ selected: candidates.map(a => a.id) })}>Pick all {candidates.length}</button> : undefined}>
        {list.length === 0 ? (
          <div className="target-card is-empty">
            <AlignGlyph size={18} />
            <div className="small">Tick the scans in the list on the left. The first one you tick is the reference the others move onto.</div>
          </div>
        ) : (
          <ul className="plain-list">
            {list.map((a, i) => (
              <li key={a.id} className="item-row">
                <span className={`index-bubble ${i === 0 ? 'signal' : ''}`}>{i + 1}</span>
                <Thumb asset={a} size={36} />
                <div className="grow">
                  <span className="strong truncate">{a.name}</span>
                  {i === 0 ? <span className="caption">reference — stays where it is</span> : <span className="caption">moves onto the reference</span>}
                </div>
                {i > 0 && (
                  <span className="item-actions">
                    <IconButton size="sm" label="Make this the reference" onClick={() => useStore.setState(s => ({ selected: [a.id, ...s.selected.filter(id => id !== a.id)] }))}>
                      <Anchor size={15} />
                    </IconButton>
                    <IconButton size="sm" label={(savedPairs[pairKey(a.id, ref.id)]?.length ?? 0) ? `${savedPairs[pairKey(a.id, ref.id)].length} matching point pairs — edit` : 'Pick matching points by hand (for symmetric parts)'} disabled={!!pairing} onClick={() => startPairing(a, ref)}>
                      <Crosshair size={15} />
                    </IconButton>
                    {(savedPairs[pairKey(a.id, ref.id)]?.length ?? 0) > 0 && <span className="badge">{savedPairs[pairKey(a.id, ref.id)].length}</span>}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </Block>

      {pairing && (
        <Block title="Picking matching points" tone="signal">
          <p className="hint-text">Click a spot on <b>{byId.get(pairing.movId)?.name}</b> (right), then the same spot on the reference (left). Use distinct features spread around the part.</p>
          <div className="row spread">
            <span className="row">
              {pairing.list.length} pair{pairing.list.length === 1 ? '' : 's'}
              {pairing.list.length < 3 ? <Badge tone="warning">need 3 or more</Badge> : <Badge tone="good">ready</Badge>}
            </span>
            <span className="row">
              <Button size="sm" variant="ghost" icon={<Undo2 size={14} />} onClick={() => useStore.setState({ pairing: pairing.pending.source || pairing.pending.target ? { ...pairing, pending: {} } : { ...pairing, list: pairing.list.slice(0, -1) } })}>Undo</Button>
              <Button size="sm" variant="primary" onClick={stopPairing}>Done</Button>
            </span>
          </div>
        </Block>
      )}

      {current && (
        <Block title="Result of the check">
          <AssessmentView
            a={current}
            names={list.map(a => a.name)}
            actions={
              <>
                {current.recommendation === 'merge' && <Button variant="primary" icon={<Combine size={15} />} onClick={run}>Merge</Button>}
                {current.best_index != null && current.recommendation !== 'merge' && <Button variant="primary" onClick={useBest}>Use the best scan</Button>}
                {current.recommendation !== 'merge' && <Button variant="danger" icon={<Combine size={15} />} onClick={() => confirm('Merge anyway? The check found a reason not to (see above). Measurements on a doubled surface read too large.') && run()}>Merge anyway</Button>}
              </>
            }
          />
        </Block>
      )}

      {list.length >= 2 && <Stickers list={list} values={values} />}

      {list.length === 2 && <PhotoOptions list={list} photos={project.filter(a => a.kind === 'image').length} />}

      <Block title="Preview before merging" guide="align.preview">
        <p className="hint-text">Each scan in its own window plus one with them together. They share the camera: turn one and they all turn.</p>
        <div className="chip-row">
          <Button size="sm" icon={<Columns3 size={14} />} disabled={list.length < 2} onClick={() => compare('raw')}>Side by side</Button>
          <Button size="sm" icon={<Layers size={14} />} disabled={!current?.transforms} title={current ? undefined : 'Run the check first'} onClick={() => compare('auto')}>Preview the merge</Button>
          <Button size="sm" icon={<Crosshair size={14} />} disabled={!pairsReady} title={pairsReady ? undefined : 'Pick 3+ point pairs first'} onClick={() => compare('pairs')}>Preview point pairs</Button>
          {panes.length > 0 && <Button size="sm" variant="ghost" onClick={() => useStore.setState({ panes: [] })}>One view</Button>}
        </div>
      </Block>

      <details className="disclosure" data-guide="align.settings">
        <summary>Fine-tune <span className="sub">alignment settings</span><ChevronDown size={16} className="chev" aria-hidden /></summary>
        <div className="disclosure-body">
          <ParamForm group="merge" schema={params.schema.merge} values={values} onChange={setValues} />
        </div>
      </details>
    </StepFrame>
  );
}

interface StickerScan {
  index: number;
  asset_id: string;
  name: string;
  ok: boolean;
  transform?: number[][];
  common?: number;
  marker_rms_mm?: number;
  markers_moving?: number;
  markers_reference?: number;
  ambiguous?: boolean;
  warning?: string;
  error?: string;
}

const toMatrix = (m: number[][]) => new THREE.Matrix4().set(...(m.flat() as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number])).toArray();

/**
 * Marker stickers on the part leave small round holes in every scan. Three or more that two scans share fix the
 * one way they fit, even when the part looks alike from several sides (cloudclean/markers.py).
 */
function Stickers({ list, values }: { list: Asset[]; values: Record<string, unknown> }) {
  const key = list.map(a => a.id).join();
  const [state, setState] = useState<{ key: string; busy?: boolean; scans?: StickerScan[]; error?: string } | null>(null);
  const cur = state?.key === key ? state : null;
  const ok = !!cur?.scans?.length && cur.scans.every(s => s.ok);

  // the stickers found are marked on the models: orange on the reference, blue on the scans that move
  useEffect(() => () => getViewer()?.setMarkers([], [], 'stickers'), []);
  useEffect(() => {
    if (!cur?.scans) getViewer()?.setMarkers([], [], 'stickers');
  }, [key, cur?.scans]);

  const find = async () => {
    setState({ key, busy: true });
    try {
      const res = await api.post<{ scans: StickerScan[] }>('/api/merge/markers-preview', { asset_ids: list.map(a => a.id) });
      setState({ key, scans: res.scans });
      const found = await Promise.all(list.map(a => api.get<{ markers: { center: [number, number, number] }[] }>(`/api/assets/${a.id}/markers`).catch(() => ({ markers: [] }))));
      getViewer()?.setMarkers(
        found.flatMap((f, i) => f.markers.map((m, k) => ({ id: `${list[i].id}-${k}`, position: m.center, color: i === 0 ? '#ee4b1f' : '#2f6fd6' }))),
        [],
        'stickers',
      );
    } catch (err) {
      setState({ key, error: (err as Error).message });
    }
  };
  const transforms = () => [null, ...(cur?.scans ?? []).map(s => s.transform ?? null)];
  const preview = () => {
    const ids = list.map(a => a.id);
    const t: Record<string, number[]> = {};
    (cur?.scans ?? []).forEach(s => {
      if (s.transform) t[s.asset_id] = toMatrix(s.transform);
    });
    useStore.setState({ visible: ids, panes: [...list.map(a => ({ id: a.id, label: a.name, assetIds: [a.id] })), { id: 'result', label: 'Preview: lined up on stickers', assetIds: ids, transforms: t, tone: 'result' as const }] });
    setTimeout(() => getViewer()?.fit(ids), 80);
  };
  const merge = () => submitJob('/api/merge', { asset_ids: list.map(a => a.id), params: { ...values, method: 'markers' }, transforms: transforms(), name: `${list[0].name} + ${list.length - 1} on stickers` });

  return (
    <Block title="Line up on stickers" guide="align.stickers">
      <p className="hint-text">Marker stickers on the part leave small round holes in each scan. When two scans share three or more, they fit exactly one way — even if the part looks the same from several sides.</p>
      <div className="row">
        <Button icon={<CircleDot size={15} />} loading={cur?.busy} onClick={find}>{cur?.scans ? 'Find them again' : 'Find the stickers'}</Button>
      </div>
      {cur?.error && <p className="err-text" role="alert">{cur.error}</p>}
      {cur?.scans && (
        <ul className="plain-list">
          {cur.scans.map(s => (
            <li key={s.asset_id} className="item-row">
              <span className="index-bubble">{s.index + 1}</span>
              <div className="grow">
                <span className="strong truncate">{s.name}</span>
                {s.ok ? (
                  <span className="caption" style={{ whiteSpace: 'normal' }}>
                    {s.common} stickers in common with {list[0].name} · fit {fmtLen(s.marker_rms_mm, 3)} mm
                    {s.ambiguous ? ' · another fit explains almost as many: check the preview' : ''}
                    {s.warning ? ` · ${s.warning}` : ''}
                  </span>
                ) : (
                  <span className="caption warn-text" style={{ whiteSpace: 'normal' }}>{s.error}</span>
                )}
              </div>
              {s.ok ? <Badge tone="good">lined up</Badge> : <Badge tone="warning">not enough</Badge>}
            </li>
          ))}
        </ul>
      )}
      {ok && (
        <div className="row">
          <Button icon={<Layers size={15} />} onClick={preview}>Preview</Button>
          <Button variant="primary" icon={<Combine size={15} />} onClick={merge}>Merge on stickers</Button>
        </div>
      )}
    </Block>
  );
}

/**
 * A symmetric part can line up in more than one way. The assistant draws every way the two scans fit and compares
 * the pictures with the reference photos of the real part (assistant tool compare_merge_options).
 */
function PhotoOptions({ list, photos }: { list: Asset[]; photos: number }) {
  const streaming = useAssistant(s => s.streaming);
  const ask = () => {
    const st = useStore.getState();
    st.set({ rightTab: 'assistant' });
    st.setLayout({ rightOpen: true });
    sendToAssistant(`Compare the ways ${list[1].name} can line up on ${list[0].name} with my reference photos of the part, and tell me which option matches the real part.`);
  };
  return (
    <Block title="Which way do the scans fit?" guide="align.photos">
      <p className="hint-text">
        A part that looks alike from several sides can line up in more than one way. The assistant draws every way these two scans fit and compares them with your photos of the real part.
      </p>
      <div className="row">
        <Button icon={<Images size={15} />} disabled={streaming} onClick={ask}>Compare with my photos</Button>
        <span className="caption">{photos ? `${photos} photo${photos === 1 ? '' : 's'} in this project` : 'No photos yet: add some with Photos in the assistant, or it judges by shape alone'}</span>
      </div>
    </Block>
  );
}
