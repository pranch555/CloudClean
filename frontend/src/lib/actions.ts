import type { ViewName, ViewportTool } from './types';
import { submitJob } from './jobs';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';

export function setTool(tool: ViewportTool) {
  const st = useStore.getState();
  // leaving the measure tool drops an unfinished measurement but keeps completed ones
  if (tool !== 'measure' && st.tool === 'measure') useStore.setState({ measurements: st.measurements.filter(m => m.b) });
  useStore.setState({ tool });
}

export function viewCamera(view: ViewName) {
  getViewer()?.fit(undefined, view);
}

export function clearSelection() {
  getViewer()?.clearSelection();
  useStore.setState({ selection: null, selectionCounts: {} });
}

/** Apply the current screen selection to the full-resolution data on the server (creates new assets). */
export function applySelection(mode: 'delete' | 'keep', visibleOnly: boolean) {
  const st = useStore.getState();
  const sel = st.selection;
  if (!sel) return;
  const ids = Object.keys(st.selectionCounts);
  if (!ids.length) {
    st.toast({ kind: 'warn', title: 'Nothing selected on a visible asset' });
    return;
  }
  for (const id of ids) {
    const a = st.byId.get(id);
    const ops = sel.region
      ? [{ op: mode === 'delete' ? 'delete_region' : 'keep_region', region: sel.region }]
      : [{ op: 'select_screen', view_projection: sel.view_projection, polygon: sel.polygon, mode, visible_only: visibleOnly }];
    submitJob('/api/edit', {
      asset_id: id,
      name: `${a?.name ?? 'model'} · ${mode === 'delete' ? 'selection removed' : 'selection kept'}`,
      ops,
    });
  }
  clearSelection();
}

export function toggleProjection() {
  const st = useStore.getState();
  st.setDisplay({ projection: st.display.projection === 'perspective' ? 'orthographic' : 'perspective' });
}

export async function copyScreenshot() {
  const v = getViewer();
  if (!v) return;
  const url = v.screenshot(2400);
  const a = document.createElement('a');
  a.href = url;
  a.download = `cloudclean-view-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`;
  a.click();
}

/*
 * Undo / redo over the model history. Every operation makes a new model and keeps its input, so "undo" steps back to
 * the model the current one was made from (the newer one stays in the list) and "redo" steps forward again.
 */
const redoStack: string[] = [];

export function undoModel() {
  const st = useStore.getState();
  const a = st.activeId ? st.byId.get(st.activeId) : undefined;
  const parentId = a?.parents.find(id => st.byId.get(id) && st.byId.get(id)!.kind !== 'image');
  if (!a || !parentId) {
    st.toast({ kind: 'info', title: 'Nothing to undo', body: a ? `${a.name} is an original scan.` : 'Click a model first.', ms: 2500 });
    return;
  }
  redoStack.push(a.id);
  useStore.setState({ activeId: parentId, visible: [parentId] });
  st.toast({ kind: 'info', title: `Back to ${st.byId.get(parentId)!.name}`, body: `${a.name} is still in your list. Ctrl+Y goes forward again.`, ms: 3500 });
}

export function redoModel() {
  const st = useStore.getState();
  let id = redoStack.pop();
  while (id && !st.byId.has(id)) id = redoStack.pop();
  if (!id) {
    // no undo to redo: go to the newest model made from the current one
    const child = st.activeId ? [...st.assets].reverse().find(x => x.parents.includes(st.activeId!)) : undefined;
    if (!child) return;
    id = child.id;
  }
  useStore.setState({ activeId: id, visible: [id] });
}
