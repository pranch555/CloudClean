/*
 * Photos of the real part: tiles that open a full-size photo viewer, and a way to delete every photo
 * (the × on a tile, Delete in the viewer, "Delete all photos" in the Photos group). Used by the model list and
 * by Scan -> Make a 3D model from photos.
 */
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { create } from 'zustand';
import { ChevronLeft, ChevronRight, Download, Paintbrush, Trash2, X } from 'lucide-react';
import { api } from '../../lib/api';
import { guideTo } from '../../lib/guide';
import type { Asset } from '../../lib/types';
import { useStore } from '../../store';
import { getViewer } from '../../viewer/instance';
import { pickPhotoToLineUp, stopAlign, useAlign } from '../../features/colour/ColourSection';
import { Button, Popover } from '../../ui/primitives';
import { whenText } from './groups';

/* ------------------------------------------------------------------ deleting */

/** Delete photos from the workspace; tells the user how it went. Returns how many were deleted. */
export async function deletePhotos(photos: Asset[]): Promise<number> {
  const st = useStore.getState();
  const results = await Promise.allSettled(photos.map(p => api.del(`/api/assets/${p.id}`)));
  const gone = photos.filter((_, i) => results[i].status === 'fulfilled');
  const failed = results.find((r): r is PromiseRejectedResult => r.status === 'rejected');
  gone.forEach(p => getViewer()?.forget(p.id));
  // a photo being lined up by hand (Mesh -> Colour from photos) goes away with it
  const al = useAlign.getState();
  if (al.align && gone.some(p => p.id === al.align!.photoId)) stopAlign();
  if (al.pick && gone.some(p => p.id === al.pick)) pickPhotoToLineUp(null);
  await st.refreshAssets().catch(() => undefined);
  st.refreshProjects().catch(() => undefined);
  if (gone.length) st.toast({ kind: 'ok', title: gone.length === 1 && photos.length === 1 ? `Deleted the photo “${gone[0].name}”` : `Deleted ${gone.length} photo${gone.length > 1 ? 's' : ''}` });
  if (failed) st.toast({ kind: 'error', title: gone.length ? `${photos.length - gone.length} photos could not be deleted` : 'Could not delete the photo', body: (failed.reason as Error)?.message });
  return gone.length;
}

/** An inline "are you sure?" with a red button and a way out. */
export function ConfirmDelete({ title, body, action, onConfirm, onCancel }: { title: ReactNode; body?: ReactNode; action: string; onConfirm: () => void | Promise<unknown>; onCancel: () => void }) {
  const [busy, setBusy] = useState(false);
  return (
    <div className="mp-confirm" role="alertdialog" aria-label={typeof title === 'string' ? title : 'Confirm delete'}>
      <div className="mp-confirm-title">
        <Trash2 size={16} aria-hidden /> {title}
      </div>
      {body && <p className="mp-confirm-body">{body}</p>}
      <div className="mp-confirm-actions">
        <Button
          size="sm"
          variant="danger"
          loading={busy}
          autoFocus
          onClick={async () => {
            setBusy(true);
            try {
              await onConfirm();
            } finally {
              setBusy(false);
            }
          }}
        >
          {action}
        </Button>
        <Button size="sm" variant="ghost" onClick={onCancel} disabled={busy}>
          Keep
        </Button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ tiles */

const imageUrl = (id: string) => `/api/assets/${id}/image`;

/** One photo: click to see it big; the × deletes it (after a "sure?"). */
export function PhotoTile({ photo, onOpen }: { photo: Asset; onOpen: () => void }) {
  return (
    <div className="ph-tile">
      <button type="button" className="ph-open" onClick={onOpen} aria-label={`Open the photo ${photo.name}`} title={`${photo.name}\nClick to see it full size`} data-guide="photo.viewer">
        <img src={imageUrl(photo.id)} alt="" loading="lazy" draggable={false} />
      </button>
      <Popover
        align="end"
        trigger={({ toggle, open }) => (
          <button type="button" className={`ph-del ${open ? 'is-open' : ''}`} onClick={toggle} aria-label={`Delete the photo ${photo.name}`} aria-expanded={open} data-tip="Delete this photo" data-guide="photo.delete">
            <X size={14} strokeWidth={2.5} aria-hidden />
          </button>
        )}
      >
        {({ close }) => (
          <ConfirmDelete
            title="Delete this photo?"
            body={<><b>{photo.name}</b> is removed from the project. Models made or coloured from it stay.</>}
            action="Delete photo"
            onConfirm={async () => {
              await deletePhotos([photo]);
              close();
            }}
            onCancel={close}
          />
        )}
      </Popover>
    </div>
  );
}

/** Photos as a grid of tiles; with `max`, the rest is one "+N more" tile that opens the viewer. */
export function PhotoGrid({ photos, max, label }: { photos: Asset[]; max?: number; label?: string }) {
  const ids = photos.map(p => p.id);
  const cut = max && photos.length > max ? max - 1 : photos.length;
  const rest = photos.length - cut;
  return (
    <ul className="ph-grid" aria-label={label ?? `${photos.length} photo${photos.length === 1 ? '' : 's'}`}>
      {photos.slice(0, cut).map(p => (
        <li key={p.id}>
          <PhotoTile photo={p} onOpen={() => openPhotoViewer(ids, p.id)} />
        </li>
      ))}
      {rest > 0 && (
        <li>
          <button type="button" className="ph-more" onClick={() => openPhotoViewer(ids, photos[cut].id)} aria-label={`See the other ${rest} photos`}>
            <span className="ph-more-n">+{rest}</span>
            <span>more</span>
          </button>
        </li>
      )}
    </ul>
  );
}

/* ------------------------------------------------------------------ the photo viewer */

interface ViewerState {
  ids: string[] | null;
  index: number;
}

const usePhotoViewer = create<ViewerState>(() => ({ ids: null, index: 0 }));

export function openPhotoViewer(ids: string[], id: string) {
  usePhotoViewer.setState({ ids, index: Math.max(0, ids.indexOf(id)) });
}

const closePhotoViewer = () => usePhotoViewer.setState({ ids: null, index: 0 });

/*
 * The viewer is drawn by one mounted host (the model list, or the Scan step's photo block), whichever came first,
 * so it works whichever panel the photo was opened from.
 */
const useHosts = create<{ hosts: string[] }>(() => ({ hosts: [] }));

export function PhotoViewerHost() {
  const id = useId();
  const mine = useHosts(s => s.hosts[0] === id);
  const open = usePhotoViewer(s => !!s.ids);
  useEffect(() => {
    useHosts.setState(s => ({ hosts: [...s.hosts, id] }));
    return () => {
      useHosts.setState(s => ({ hosts: s.hosts.filter(h => h !== id) }));
      if (!useHosts.getState().hosts.length) closePhotoViewer();
    };
  }, [id]);
  return mine && open ? <PhotoViewer /> : null;
}

/** Open Mesh -> Colour from photos -> "Line up one photo by hand" (a closed disclosure) with this photo chosen. */
function lineUpByHand(photoId: string) {
  pickPhotoToLineUp(photoId);
  guideTo('mesh.colour-by-hand', 'Line up one photo by hand', 'Mesh → Colour from photos → Line up one photo by hand', { step: 'mesh', right: 'step' });
  const started = performance.now();
  const open = () => {
    const d = document.querySelector<HTMLDetailsElement>('details[data-guide~="mesh.colour-by-hand"]');
    if (d) d.open = true;
    else if (performance.now() - started < 2500) setTimeout(open, 120);
  };
  setTimeout(open, 60);
}

function PhotoViewer() {
  const ids = usePhotoViewer(s => s.ids) ?? [];
  const index = usePhotoViewer(s => s.index);
  const byId = useStore(s => s.byId);
  const list = ids.map(id => byId.get(id)).filter((a): a is Asset => !!a && a.kind === 'image');
  const i = Math.min(index, list.length - 1);
  const photo = list[i];
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const titleId = useId();

  const go = (d: number) => {
    if (list.length < 2) return;
    usePhotoViewer.setState({ index: (i + d + list.length) % list.length });
  };

  // every photo gone (deleted here or elsewhere): close
  useEffect(() => {
    if (!list.length) closePhotoViewer();
  }, [list.length]);

  useEffect(() => setConfirming(false), [photo?.id]);

  // focus moves into the viewer and back to where it was when it closes
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    const grid = opener?.closest('.ph-grid');
    box.current?.focus();
    return () => {
      // back to the photo it was opened from; if that one was deleted, to the photos it was among
      if (opener?.isConnected) opener.focus();
      else if (grid?.isConnected) grid.querySelector<HTMLElement>('.ph-open, .ph-more')?.focus();
    };
  }, []);

  // keys go to the viewer only: the app's shortcuts (Delete, F, 1-7...) must not act behind it
  const keys = useRef({ go, confirming, setConfirming });
  keys.current = { go, confirming, setConfirming };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = keys.current;
      e.stopPropagation();
      if (e.key === 'Escape') {
        e.preventDefault();
        if (k.confirming) k.setConfirming(false);
        else closePhotoViewer();
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        if ((e.target as HTMLElement).closest?.('input, textarea')) return;
        e.preventDefault();
        k.go(e.key === 'ArrowLeft' ? -1 : 1);
      } else if (e.key === 'Delete') {
        e.preventDefault();
        k.setConfirming(true);
      } else if (e.key === 'Tab' && box.current) {
        const items = [...box.current.querySelectorAll<HTMLElement>('button, [href], [tabindex]:not([tabindex="-1"])')].filter(el => !el.hasAttribute('disabled') && el.offsetParent !== null);
        if (!items.length) return;
        const first = items[0], last = items[items.length - 1];
        if (e.shiftKey && (document.activeElement === first || document.activeElement === box.current)) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, []);

  // the neighbours load in the background, so stepping through is instant
  useEffect(() => {
    if (list.length < 2) return;
    for (const d of [1, -1]) new Image().src = imageUrl(list[(i + d + list.length) % list.length].id);
  }, [photo?.id, list.length]);

  if (!photo) return null;
  const ext = (photo.file?.split('.').pop() || 'jpg').toLowerCase();
  const px = photo.stats.width && photo.stats.height ? `${photo.stats.width} × ${photo.stats.height} px` : null;
  const del = async () => {
    setBusy(true);
    await deletePhotos([photo]);
    setBusy(false);
    setConfirming(false);
  };

  return createPortal(
    <div className="pv" onPointerDown={e => e.target === e.currentTarget && closePhotoViewer()} data-guide="photo.viewer">
      <div ref={box} className="pv-box" role="dialog" aria-modal="true" aria-labelledby={titleId} tabIndex={-1}>
        <button type="button" className="pv-close" onClick={closePhotoViewer} aria-label="Close the photo viewer (Esc)" data-tip="Close (Esc)" data-tip-side="bottom">
          <X size={20} />
        </button>
        <div className="pv-stage" onPointerDown={e => e.target === e.currentTarget && closePhotoViewer()}>
          {list.length > 1 && (
            <button type="button" className="pv-nav pv-prev" onClick={() => go(-1)} aria-label="Previous photo (←)" data-tip="Previous (←)">
              <ChevronLeft size={26} />
            </button>
          )}
          <img key={photo.id} className="pv-img" src={imageUrl(photo.id)} alt={photo.name} draggable={false} />
          {list.length > 1 && (
            <button type="button" className="pv-nav pv-next" onClick={() => go(1)} aria-label="Next photo (→)" data-tip="Next (→)">
              <ChevronRight size={26} />
            </button>
          )}
        </div>
        <div className="pv-card">
          <div className="pv-info">
            <h2 className="pv-name" id={titleId} title={photo.name}>{photo.name}</h2>
            <div className="pv-meta">
              {list.length > 1 && <span className="pv-count">Photo {i + 1} of {list.length}</span>}
              {px && <span className="mono">{px}</span>}
              <span>added {whenText(photo.created)}</span>
            </div>
          </div>
          {confirming ? (
            <div className="pv-confirm" role="alertdialog" aria-label="Delete this photo?">
              <span className="pv-confirm-text"><b>Delete this photo?</b> It is removed from the project. Models made or coloured from it stay.</span>
              <div className="pv-actions">
                <Button variant="danger" size="sm" icon={<Trash2 size={15} />} loading={busy} onClick={del} autoFocus>
                  Delete photo
                </Button>
                <Button variant="ghost" size="sm" onClick={() => setConfirming(false)} disabled={busy}>
                  Keep
                </Button>
              </div>
            </div>
          ) : (
            <div className="pv-actions">
              <a className="btn btn-secondary btn-sm" href={imageUrl(photo.id)} download={`${photo.name}.${ext}`}>
                <Download size={15} aria-hidden />
                <span className="btn-label">Download</span>
              </a>
              <Button
                size="sm"
                variant="ghost"
                icon={<Paintbrush size={15} />}
                title="Mesh → Colour from photos: paints the real colours of the part onto your scan or mesh"
                onClick={() => {
                  closePhotoViewer();
                  guideTo('mesh.colour', 'Colour from photos', 'Mesh → Colour from photos', { step: 'mesh', right: 'step' });
                }}
              >
                Colour a model with photos
              </Button>
              <button
                type="button"
                className="link pv-link"
                onClick={() => {
                  closePhotoViewer();
                  lineUpByHand(photo.id);
                }}
              >
                or line this one up by hand
              </button>
              <span className="spacer" />
              <Button variant="danger" size="sm" icon={<Trash2 size={15} />} onClick={() => setConfirming(true)} data-guide="photo.delete">
                Delete
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
