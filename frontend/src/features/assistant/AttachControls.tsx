import { useRef, type ClipboardEvent, type DragEvent } from 'react';
import { Camera, Check, ImagePlus, Laptop, X } from 'lucide-react';
import { useProjectAssets, useStore } from '../../store';
import type { Asset } from '../../lib/types';
import { uid } from '../../lib/uid';
import { Popover, Switch } from '../../ui/primitives';
import { getViewer } from '../../viewer/instance';
import { attachmentFromFile, isPhotoFile, MAX_ATTACHMENTS, shrinkImage } from './attachments';
import { setKeepPhotos, useAssistant } from './assistantStore';

const room = () => MAX_ATTACHMENTS - useAssistant.getState().attachments.length;

function full() {
  useStore.getState().toast({ kind: 'warn', title: `You can attach ${MAX_ATTACHMENTS} images per message` });
}

/** Attach image files (photos from this computer, pasted or dropped pictures). Non-images are ignored. */
export async function attachFiles(files: File[]) {
  const images = files.filter(isPhotoFile);
  if (!images.length) return;
  if (room() <= 0) return full();
  try {
    const added = await Promise.all(images.slice(0, room()).map(attachmentFromFile));
    useAssistant.setState(s => ({ attachments: [...s.attachments, ...added] }));
    if (images.length > added.length) full();
  } catch (err) {
    useStore.getState().toast({ kind: 'error', title: 'Could not attach the image', body: (err as Error).message });
  }
}

/** Attach a picture of the 3D view as it is on screen. */
export function attachViewPicture() {
  const v = getViewer();
  if (!v) return;
  if (room() <= 0) return full();
  useAssistant.setState(s => ({ attachments: [...s.attachments, { id: uid(), url: v.screenshot(1400), caption: 'screenshot of the 3D view', name: 'view.png' }] }));
}

/** Attach a photo that is already saved in the project. */
export async function attachSavedPhoto(asset: Asset) {
  const st = useAssistant.getState();
  if (st.attachments.some(a => a.assetId === asset.id)) return;
  if (room() <= 0) return full();
  try {
    const blob = await (await fetch(`/api/assets/${asset.id}/image`)).blob();
    const url = await shrinkImage(blob);
    useAssistant.setState(s => ({ attachments: [...s.attachments, { id: uid(), url, caption: `reference photo '${asset.name}' (${asset.id})`, name: asset.name, assetId: asset.id }] }));
  } catch (err) {
    useStore.getState().toast({ kind: 'error', title: 'Could not attach the photo', body: (err as Error).message });
  }
}

/** Paste handler for any chat input: pictures on the clipboard become attachments. */
export function pasteImages(e: ClipboardEvent) {
  const files = [...e.clipboardData.files].filter(isPhotoFile);
  if (!files.length) return;
  e.preventDefault();
  attachFiles(files);
}

/** Drop handlers for any chat surface: dropped pictures become attachments (and do not import as scans). */
export const dropImages = {
  onDragOver: (e: DragEvent) => {
    if (e.dataTransfer.types.includes('Files')) e.preventDefault();
  },
  onDrop: (e: DragEvent) => {
    const files = [...e.dataTransfer.files].filter(isPhotoFile);
    if (!files.length) return;
    e.preventDefault();
    e.stopPropagation();
    attachFiles(files);
  },
};

/**
 * "Photos" button of every chat input: photos from this computer, photos already saved in the project, or a
 * picture of the 3D view; plus whether new photos are kept in the project as reference photos of the part.
 */
export function AttachButton({ compact = false, view = true }: { compact?: boolean; view?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const count = useAssistant(s => s.attachments.length);
  const keep = useAssistant(s => s.keepPhotos);
  const attachedIds = useAssistant(s => s.attachments.map(a => a.assetId).filter(Boolean).join());
  const photos = useProjectAssets().filter(a => a.kind === 'image').slice(-12).reverse();
  const disabled = count >= MAX_ATTACHMENTS;
  return (
    <>
      <input ref={input} type="file" accept="image/*,.heic,.heif" multiple hidden onChange={e => { attachFiles([...(e.target.files ?? [])]); e.target.value = ''; }} />
      <Popover
        align="start"
        side="top"
        trigger={({ toggle, open }) => (
          <button
            type="button"
            className={`attach-btn ${compact ? 'is-compact' : ''}`}
            data-guide="photos"
            onClick={toggle}
            aria-expanded={open}
            aria-label={`Add photos or pictures to the message${count ? ` (${count} attached)` : ''}`}
            data-tip={compact ? 'Add photos' : undefined}
            disabled={disabled}
          >
            <ImagePlus size={16} aria-hidden />
            {!compact && <span>Photos</span>}
            {count > 0 && <span className="attach-count">{count}</span>}
          </button>
        )}
      >
        {({ close }) => (
          <div className="menu attach-menu">
            <button type="button" className="menu-item" onClick={() => { close(); input.current?.click(); }}>
              <Laptop size={16} />
              <div className="grow">
                <div className="menu-item-title">From this computer…</div>
                <div className="menu-item-sub">Reference photos of the part you are scanning</div>
              </div>
            </button>
            {view && (
              <button type="button" className="menu-item" onClick={() => { close(); attachViewPicture(); }}>
                <Camera size={16} />
                <div className="grow">
                  <div className="menu-item-title">Picture of the 3D view</div>
                  <div className="menu-item-sub">Exactly what is on screen now</div>
                </div>
              </button>
            )}
            {photos.length > 0 && (
              <>
                <div className="menu-sep" />
                <div className="menu-label">Saved in this project</div>
                <div className="attach-grid">
                  {photos.map(p => {
                    const on = attachedIds.includes(p.id);
                    return (
                      <button key={p.id} type="button" className={`attach-thumb ${on ? 'is-on' : ''}`} onClick={() => attachSavedPhoto(p)} aria-pressed={on} aria-label={`Attach ${p.name}`} title={p.name}>
                        <img src={`/api/assets/${p.id}/image`} alt="" loading="lazy" />
                        {on && <span className="attach-check"><Check size={12} /></span>}
                      </button>
                    );
                  })}
                </div>
              </>
            )}
            <div className="menu-sep" />
            <label className="attach-keep">
              <Switch checked={keep} onChange={setKeepPhotos} label="Keep new photos in the project" />
              <span>
                Keep new photos in the project
                <span className="menu-item-sub">The assistant can look at them again later</span>
              </span>
            </label>
          </div>
        )}
      </Popover>
    </>
  );
}

/** The pictures attached to the next message, each removable. */
export function AttachmentStrip({ small = false }: { small?: boolean }) {
  const attachments = useAssistant(s => s.attachments);
  if (!attachments.length) return null;
  return (
    <div className={`attachments ${small ? 'is-small' : ''}`} aria-label="Attached to your next message">
      {attachments.map(a => (
        <div key={a.id} className="attachment" title={a.name}>
          <img src={a.url} alt={a.caption} />
          <button type="button" aria-label={`Remove ${a.name}`} onClick={() => useAssistant.setState(s => ({ attachments: s.attachments.filter(x => x.id !== a.id) }))}>
            <X size={11} />
          </button>
        </div>
      ))}
    </div>
  );
}
