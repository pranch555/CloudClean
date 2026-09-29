import { useState } from 'react';
import { Crop, Loader2, LocateFixed, Move3d, Trash2, X } from 'lucide-react';
import { applySelection, clearSelection } from '../lib/actions';
import { fmtCount } from '../lib/format';
import { submitJob } from '../lib/jobs';
import { useStore } from '../store';
import { useCapture } from '../features/capture/captureStore';
import { draftForAssistant } from '../features/assistant/assistantStore';
import { AssistantGlyph } from '../features/assistant/AssistantMark';
import { Button, IconButton, Switch } from '../ui/primitives';

export function Banners() {
  return (
    <div className="banner-stack">
      <FollowChip />
      <LoadingPill />
      <SelectionBar />
    </div>
  );
}

/**
 * Live capture: the view follows the scanner (as Revo Metro does) until you grab it. Grabbing switches to a free
 * view immediately — the camera is never pulled back while you are looking around — and this chip brings the
 * scanner view back.
 */
function FollowChip() {
  const step = useStore(s => s.step);
  const screen = useStore(s => s.screen);
  const follow = useCapture(s => s.follow);
  const state = useCapture(s => s.status?.state);
  const live = screen === 'workspace' && step === 'capture' && (state === 'running' || state === 'paused');
  if (!live) return null;
  return follow ? (
    <div className="hud banner">
      <LocateFixed size={15} aria-hidden />
      <span>Following the scanner</span>
      <span className="muted small">— drag to look around</span>
      <Button size="sm" variant="ghost" onClick={() => useCapture.setState({ follow: false })}>Free view</Button>
    </div>
  ) : (
    <div className="hud banner">
      <Move3d size={15} aria-hidden />
      <span>Free view</span>
      <Button size="sm" variant="secondary" icon={<LocateFixed size={14} />} onClick={() => useCapture.setState({ follow: true })}>Follow scanner</Button>
    </div>
  );
}

function LoadingPill() {
  const names = useStore(s => s.loadingNames);
  if (!names) return null;
  return (
    <div className="hud banner loading-pill">
      <Loader2 size={15} className="spin" aria-hidden />
      <span className="truncate" style={{ maxWidth: 360 }}>Loading {names.join(', ')}</span>
    </div>
  );
}

function SelectionBar() {
  const selection = useStore(s => s.selection);
  const counts = useStore(s => s.selectionCounts);
  const byId = useStore(s => s.byId);
  const measuring = useStore(s => s.step === 'measure' && s.screen === 'workspace');
  const [visibleOnly, setVisibleOnly] = useState(false);
  if (!selection) return null;
  const ids = Object.keys(counts);
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const names = ids.map(id => byId.get(id)?.name).filter(Boolean);
  const region = selection.region;

  const apply = (mode: 'delete' | 'keep') => {
    if (!region) {
      applySelection(mode, visibleOnly);
      return;
    }
    for (const id of ids.length ? ids : selection.asset_id ? [selection.asset_id] : []) {
      const a = byId.get(id);
      submitJob('/api/edit', { asset_id: id, name: `${a?.name ?? 'model'} · ${mode === 'delete' ? `without ${selection.label ?? 'region'}` : `only ${selection.label ?? 'region'}`}`, ops: [{ op: mode === 'delete' ? 'delete_region' : 'keep_region', region }] });
    }
    clearSelection();
  };

  return (
    <div className="hud selection-bar" role="region" aria-label="Selection actions">
      <div className="sel-count">
        <b className="mono">{fmtCount(total)} {total === 1 ? 'point' : 'points'}</b>
        <span className="sel-on">{region ? selection.label ?? 'highlighted region' : 'selected'}{names.length ? ` on ${names.slice(0, 2).join(', ')}${names.length > 2 ? ` +${names.length - 2}` : ''}` : ''}</span>
      </div>
      {!region && !measuring && (
        <label className="sel-toggle" title="Only select the surface facing you (ignore points hidden behind it)">
          <Switch checked={visibleOnly} onChange={setVisibleOnly} label="Front surface only" />
          <span>Front only</span>
        </label>
      )}
      {!measuring && (
        <>
          <Button size="sm" variant="danger" icon={<Trash2 size={14} />} disabled={!total} onClick={() => apply('delete')}>Delete</Button>
          <Button size="sm" icon={<Crop size={14} />} disabled={!total} onClick={() => apply('keep')}>Keep only</Button>
        </>
      )}
      <Button size="sm" variant="ghost" icon={<AssistantGlyph size={14} className="is-brand" />} onClick={() => draftForAssistant(region ? `With the highlighted region (${selection.label ?? 'region'}): ` : 'With the points I selected: ')}>Ask</Button>
      <IconButton size="sm" label="Clear the selection (Esc)" onClick={clearSelection}>
        <X size={15} />
      </IconButton>
    </div>
  );
}
