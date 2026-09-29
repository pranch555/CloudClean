import { create } from 'zustand';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';

/** Thumbnails rendered in this browser session (data URLs), shown until the server copy is listed. */
export const useThumbs = create<{ urls: Record<string, string> }>(() => ({ urls: {} }));

const pending = new Set<string>();
let uploadSupported = true;

/**
 * Render an iso thumbnail of a model that has just loaded in the viewer, keep it for this session and store it on
 * the server (PUT /api/assets/{id}/thumbnail) so the Models list and the Home screen show it next time.
 */
export async function ensureThumbnail(id: string) {
  const a = useStore.getState().byId.get(id);
  if (!a || a.kind === 'image' || a.has_thumbnail || useThumbs.getState().urls[id] || pending.has(id)) return;
  const v = getViewer();
  if (!v || !v.items.has(id)) return;
  pending.add(id);
  try {
    const url = v.renderThumbnail(id, 176);
    if (!url) return;
    useThumbs.setState(s => ({ urls: { ...s.urls, [id]: url } }));
    if (!uploadSupported) return;
    const blob = await (await fetch(url)).blob();
    const res = await fetch(`/api/assets/${id}/thumbnail`, { method: 'PUT', headers: { 'Content-Type': 'image/png' }, body: blob });
    if (res.status === 404 || res.status === 405) uploadSupported = false;
  } catch {
    /* a thumbnail is a nicety; never bother the user about it */
  } finally {
    pending.delete(id);
  }
}
