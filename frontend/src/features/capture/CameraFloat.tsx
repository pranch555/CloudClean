import { Camera, X } from 'lucide-react';
import { fmtCount } from '../../lib/format';
import { IconButton, Segmented } from '../../ui/primitives';
import { CameraPicture } from './CameraPicture';
import { type CameraBrief, type Cams, linesWord, setCameraView, useCamera, useCameraInfo, usePreviewHold } from './cameraStore';
import { useCapture } from './captureStore';

/*
 * The camera view over the 3D view, top right (Revo Metro keeps its camera pictures beside the model while scanning):
 * the picture, the laser lines in one word and what auto exposure is doing. Scan → Camera view → "Show over the 3D
 * view" opens it; the cross puts it back in the panel.
 */
export function CameraFloat() {
  const status = useCapture(s => s.status);
  const brief = status?.device?.camera as CameraBrief | undefined;
  const cams = useCamera(s => s.cams);
  const paused = useCamera(s => s.paused);
  const available = useCamera(s => s.info?.available !== false);
  const state = status?.state ?? 'idle';
  const scanning = state === 'running' || state === 'paused';
  const usable = !!status?.active && state !== 'error' && state !== 'closed';
  const on = usable && (scanning || !paused);

  usePreviewHold(usable && !scanning && !paused);
  useCameraInfo(usable, 1500);

  if (!usable || !available) return null;
  const live = on && !!brief?.streaming;
  const w = live ? linesWord(brief?.verdict, brief ?? undefined) : null;
  return (
    <div className="hud cam-float" role="group" aria-label="Camera view">
      <div className="cam-float-head">
        <Camera size={15} aria-hidden className="cam-float-icon" />
        <span className="cam-float-title">Camera</span>
        {w && <span className={`cam-pill tone-${w.tone}`}>Lines {w.word}</span>}
        <span className="grow" />
        <Segmented<Cams>
          size="sm"
          ariaLabel="Which camera"
          value={cams}
          onChange={v => setCameraView({ cams: v })}
          options={[
            { value: 'both', label: 'L + R', title: 'Both cameras' },
            { value: 'left', label: 'L', title: 'Left camera' },
            { value: 'right', label: 'R', title: 'Right camera' },
          ]}
        />
        <IconButton size="sm" label="Put the camera view back in the Scan panel" tip="bottom" onClick={() => setCameraView({ float: false })}>
          <X size={14} />
        </IconButton>
      </div>
      <CameraPicture on={on} scanning={scanning} compact onTurnOn={() => useCamera.setState({ paused: false })} />
      {live && (
        <div className="cam-float-foot">
          <span className="truncate">{brief?.mode === 'manual' ? 'Manual settings' : brief?.auto_message || 'Auto'}</span>
          {brief?.points_per_frame != null && <span className="mono">{fmtCount(brief.points_per_frame)} pts</span>}
        </div>
      )}
    </div>
  );
}
