import { useEffect, useRef, useState } from 'react';
import { create } from 'zustand';
import { Camera, Eye, FileJson, Paintbrush, Trash2 } from 'lucide-react';
import { submitJob } from '../../lib/jobs';
import { useStore, useTarget } from '../../store';
import { getViewer } from '../../viewer/instance';
import { defaultsOf, ParamForm } from '../../ui/ParamForm';
import { Button, Empty, Field, Section, Slider } from '../../ui/primitives';

interface SavedView {
  image_id: string;
  mesh_id: string;
  view_matrix: number[];
  fov: number;
  aspect: number;
}

const loadViews = (): SavedView[] => {
  try {
    return JSON.parse(localStorage.getItem('cloudclean.views') || '[]');
  } catch {
    return [];
  }
};

interface AlignState {
  align: { meshId: string; photoId: string; prevVisible: string[] } | null;
  opacity: number;
  views: SavedView[];
}

export const useAlign = create<AlignState>(() => ({ align: null, opacity: 0.5, views: loadViews() }));

const saveViews = (views: SavedView[]) => {
  useAlign.setState({ views });
  try {
    localStorage.setItem('cloudclean.views', JSON.stringify(views));
  } catch {
    /* ignore */
  }
};

function startAlign(meshId: string, photoId: string) {
  const st = useStore.getState();
  const photo = st.byId.get(photoId);
  const v = getViewer();
  if (!photo || !v) return;
  const prev = useAlign.getState().align;
  useAlign.setState({ align: { meshId, photoId, prevVisible: prev ? prev.prevVisible : st.visible } });
  useStore.setState({ visible: [meshId], tool: 'navigate' });
  const aspect = (photo.stats.width ?? 4) / (photo.stats.height ?? 3);
  v.lockAspect(aspect);
  if (photo.stats.focal_35mm) {
    const half = aspect >= 1 ? 12 : 18;
    v.setFov((2 * Math.atan(half / photo.stats.focal_35mm) * 180) / Math.PI);
  }
  if (!prev) setTimeout(() => v.fit([meshId]), 50);
}

export function stopAlign() {
  const a = useAlign.getState().align;
  if (!a) return;
  const v = getViewer();
  v?.lockAspect(null);
  v?.setFov(35);
  useAlign.setState({ align: null });
  useStore.setState({ visible: a.prevVisible });
}

export function ColourSection() {
  const params = useStore(s => s.params)!;
  const assets = useStore(s => s.assets);
  const activeId = useStore(s => s.activeId);
  const mesh = useTarget(['mesh']) ?? null;
  const photos = assets.filter(a => a.kind === 'image');
  const { align, opacity, views } = useAlign();
  const [photoId, setPhotoId] = useState<string | null>(null);
  const [values, setValues] = useState(() => defaultsOf(params.schema.texture));
  const [fov, setFovState] = useState(35);
  const [roll, setRoll] = useState(0);
  const byId = useStore(s => s.byId);
  const chosen = photoId && byId.has(photoId) ? photoId : activeId && byId.get(activeId)?.kind === 'image' ? activeId : photos[photos.length - 1]?.id ?? null;
  const validViews = views.filter(v => byId.has(v.image_id));
  const photo = align ? byId.get(align.photoId) : null;
  const halfSensor = photo && (photo.stats.width ?? 0) < (photo.stats.height ?? 0) ? 18 : 12;

  useEffect(() => () => stopAlign(), []);

  return (
    <div className="stack">
      <p className="hint-text">Line each photo up with the mesh, save the view, then apply. Photos from several sides give full coverage; use even light with fixed exposure and white balance.</p>
      {!mesh && <p className="warn-text">Click a mesh first — colour goes onto a mesh.</p>}

      <Section title={`Photos · ${photos.length}`}>
        {photos.length ? (
          <div className="photo-grid">
            {photos.map(p => (
              <button key={p.id} type="button" className={`photo-pick ${p.id === chosen ? 'is-active' : ''}`} title={p.name} onClick={() => { setPhotoId(p.id); if (align && mesh) startAlign(mesh.id, p.id); }}>
                <img src={`/api/assets/${p.id}/image`} alt={p.name} loading="lazy" />
              </button>
            ))}
          </div>
        ) : (
          <Empty title="No photos yet">Drop JPG / PNG photos of the part onto the window.</Empty>
        )}
      </Section>

      {align ? (
        <Section title="Aligning photo" tone="accent">
          <p className="hint-text">Drag to rotate, right-drag to pan, scroll to zoom. Match the lens first, then line up the outline.</p>
          <Field label="Photo opacity" inline={false}>
            <Slider value={opacity} min={0} max={1} step={0.01} format={v => `${Math.round(v * 100)}%`} onChange={v => useAlign.setState({ opacity: v })} />
          </Field>
          <Field label="Lens (35 mm equiv.)" inline={false}>
            <Slider value={fov} min={8} max={90} step={0.1} format={v => `${(halfSensor / Math.tan((v * Math.PI) / 360)).toFixed(0)} mm`} onChange={v => { setFovState(v); getViewer()?.setFov(v); }} />
          </Field>
          <Field label="Roll" inline={false}>
            <Slider value={roll} min={-45} max={45} step={0.1} format={v => `${v.toFixed(1)}°`} onChange={v => { getViewer()?.roll(((v - roll) * Math.PI) / 180); setRoll(v); }} />
          </Field>
          <div className="row gap-1">
            <Button variant="primary" icon={<Camera size={14} />} onClick={() => {
              const v = getViewer();
              if (!v) return;
              saveViews([...views, { ...v.captureView(), image_id: align.photoId, mesh_id: align.meshId }]);
              useStore.getState().toast({ kind: 'ok', title: 'View saved', body: 'Align another photo or finish and apply colours.' });
            }}>
              Save view
            </Button>
            <Button onClick={stopAlign}>Finish aligning</Button>
          </div>
        </Section>
      ) : (
        <Button block icon={<Eye size={15} />} disabled={!mesh || !chosen} onClick={() => mesh && chosen && startAlign(mesh.id, chosen)}>
          Align selected photo to mesh
        </Button>
      )}

      <Section title={`Saved views · ${validViews.length}`}>
        {validViews.length === 0 && <p className="hint-text">No views yet. Align a photo and press Save view.</p>}
        <ul className="view-list">
          {validViews.map((sv, i) => (
            <li key={i} className="view-item">
              <img src={`/api/assets/${sv.image_id}/image`} alt="" />
              <span className="view-name">{byId.get(sv.image_id)?.name}</span>
              <button type="button" className="link" onClick={() => { if (mesh) { startAlign(mesh.id, sv.image_id); setTimeout(() => getViewer()?.applyView(sv), 80); } }}>show</button>
              <button type="button" className="icon-link" title="Camera JSON for `cloudclean texture --view`" onClick={() => {
                const data = { image: byId.get(sv.image_id)?.name, view_matrix: sv.view_matrix, fov: sv.fov, aspect: sv.aspect };
                const a = document.createElement('a');
                a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
                a.download = `${byId.get(sv.image_id)?.name ?? 'photo'}_camera.json`;
                a.click();
              }}><FileJson size={14} /></button>
              <button type="button" className="icon-link danger" aria-label="Remove view" onClick={() => saveViews(views.filter(x => x !== sv))}><Trash2 size={14} /></button>
            </li>
          ))}
        </ul>
      </Section>

      <ParamForm group="texture" schema={params.schema.texture} values={values} onChange={setValues} />
      <Button variant="primary" block icon={<Paintbrush size={15} />} disabled={!mesh || !validViews.length || !!align} onClick={() => mesh && submitJob('/api/texture', { asset_id: mesh.id, params: values, views: validViews.map(({ image_id, view_matrix, fov, aspect }) => ({ image_id, view_matrix, fov, aspect })) })}>
        Apply colours from {validViews.length} view{validViews.length === 1 ? '' : 's'}
      </Button>
    </div>
  );
}

/** Photo drawn exactly over the letterboxed canvas while aligning. */
export function PhotoOverlay() {
  const { align, opacity } = useAlign();
  const img = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const v = getViewer();
    if (!v || !align) return;
    const place = () => {
      const r = v.canvasRect();
      if (img.current) Object.assign(img.current.style, { left: `${r.left}px`, top: `${r.top}px`, width: `${r.width}px`, height: `${r.height}px` });
    };
    place();
    return v.subscribe(place);
  }, [align]);
  if (!align) return null;
  return <img ref={img} className="photo-overlay" src={`/api/assets/${align.photoId}/image`} style={{ opacity }} alt="" />;
}
