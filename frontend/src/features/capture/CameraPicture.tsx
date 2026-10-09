import { useLayoutEffect, useRef, useState } from 'react';
import { Camera, CameraOff } from 'lucide-react';
import { type Cams, useCamera, useCameraFeed } from './cameraStore';

/*
 * The viewfinder: what the scanner's two IR cameras see, as Revo Metro shows it. A dark monitor in both themes (the
 * picture is infrared grey), crop marks at the corners, which camera is which, whether it is live, and — with the
 * overlay — a key for the colours CloudClean paints on it. Used in Scan → Camera view and over the 3D view.
 */

const ASPECT: Record<Cams, number> = { both: 964 / 360, left: 4 / 3, right: 4 / 3 };

function useWidth(ref: React.RefObject<HTMLElement | null>) {
  const [w, setW] = useState(480);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

export function CameraPicture({ on, scanning, compact, onTurnOn }: { on: boolean; scanning: boolean; compact?: boolean; onTurnOn?: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useWidth(ref);
  const cams = useCamera(s => s.cams);
  const overlay = useCamera(s => s.overlay);
  const info = useCamera(s => s.info);
  const dpr = typeof window !== 'undefined' ? Math.min(2, window.devicePixelRatio || 1) : 1;
  const { url, waiting } = useCameraFeed(on, cams, overlay, width * dpr);
  const live = on && !!url && !waiting;

  let wait: { icon?: 'off'; title: string; body?: string } | null = null;
  if (!on) wait = { icon: 'off', title: 'Camera off', body: scanning ? undefined : 'The laser is off until you turn it on or start scanning.' };
  else if (info?.error) wait = { title: 'The camera stopped', body: info.error };
  else if (!live) wait = info?.starting || (!info?.streaming && !scanning) ? { title: 'Turning the laser on…', body: 'The scanner needs a few seconds.' } : { title: 'Waiting for the first picture…' };

  return (
    <>
    <div ref={ref} className={`cam-finder ${compact ? 'is-compact' : ''} ${live ? 'is-live' : ''}`} style={{ aspectRatio: String(ASPECT[cams]) }}>
      {url && on && <img className="cam-img" src={url} alt={`What the scanner's ${cams === 'both' ? 'two cameras see' : `${cams} camera sees`}`} draggable={false} />}
      <span className="cam-crop tl" aria-hidden />
      <span className="cam-crop tr" aria-hidden />
      <span className="cam-crop bl" aria-hidden />
      <span className="cam-crop br" aria-hidden />
      {live && cams === 'both' && (
        <>
          <span className="cam-tag" style={{ left: 8 }}>Left</span>
          <span className="cam-tag" style={{ left: 'calc(50% + 8px)' }}>Right</span>
          <span className="cam-split" aria-hidden />
        </>
      )}
      {live && cams !== 'both' && <span className="cam-tag" style={{ left: 8 }}>{cams === 'left' ? 'Left camera' : 'Right camera'}</span>}
      {live && (
        <span className={`cam-live ${scanning ? 'is-rec' : ''}`} title={scanning ? 'Scanning: what you see is being recorded' : 'Live, not recording: press Start scanning to record'}>
          <i aria-hidden /> {scanning ? 'Live' : compact ? 'Live' : 'Live · not recording'}
        </span>
      )}
      {wait && (
        <div className={`cam-wait ${wait.icon === 'off' ? 'is-off' : 'is-busy'}`} role="status">
          {wait.icon === 'off' ? <CameraOff size={compact ? 18 : 22} aria-hidden /> : <Camera size={compact ? 18 : 22} aria-hidden />}
          <b>{wait.title}</b>
          {wait.body && !compact && <span>{wait.body}</span>}
          {wait.icon === 'off' && onTurnOn && (
            <button type="button" className="cam-wait-btn" onClick={onTurnOn}>
              Turn the camera on
            </button>
          )}
          {wait.icon !== 'off' && <span className="cam-sweep" aria-hidden />}
        </div>
      )}
    </div>
    {overlay && !compact && (
      <div className={`cam-legend ${live ? '' : 'is-idle'}`} aria-label="What the colours on the picture mean">
        <span className="k-line">laser lines CloudClean found</span>
        <span className="k-sat">too bright</span>
        <span className="k-mk">markers in both cameras</span>
        <span className="k-mk-one">not matched</span>
      </div>
    )}
    </>
  );
}
