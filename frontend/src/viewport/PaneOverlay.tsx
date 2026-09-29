import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';

/** Frames and labels for the side-by-side comparison panes (the 3D itself is drawn by the viewer). */
export function PaneOverlay() {
  const panes = useStore(s => s.panes);
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const v = getViewer();
    if (!v || !panes.length) return;
    const place = () => {
      const el = host.current;
      if (!el) return;
      const rects = v.paneRects();
      [...el.children].forEach((node, i) => {
        const r = rects[i];
        const style = (node as HTMLElement).style;
        if (!r) {
          style.display = 'none';
          return;
        }
        style.display = '';
        style.transform = `translate(${r.x}px, ${r.y}px)`;
        style.width = `${r.w}px`;
        style.height = `${r.h}px`;
      });
    };
    place();
    return v.subscribe(place);
  }, [panes]);

  if (!panes.length) return null;
  return (
    <>
      <div className="pane-overlay" ref={host} aria-hidden>
        {panes.map(p => (
          <div key={p.id} className={`pane-frame ${p.tone === 'result' ? 'is-result' : ''}`}>
            <span className="pane-label">{p.label}</span>
          </div>
        ))}
      </div>
      <button type="button" className="hud pane-exit" onClick={() => useStore.setState({ panes: [] })}>
        <X size={13} aria-hidden /> Exit comparison
      </button>
    </>
  );
}
