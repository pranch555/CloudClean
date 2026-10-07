import { useEffect, useState } from 'react';
import { Axis3d, Brush, Camera, Crosshair, Eye, Fullscreen, Grid3x3, Lasso, Maximize2, MousePointer2, Rotate3d, Ruler, Scissors, Shrink, SlidersHorizontal, SquareDashedMousePointer } from 'lucide-react';
import type { ColorMode, ViewName, ViewportTool } from '../lib/types';
import { copyScreenshot, setTool, viewCamera } from '../lib/actions';
import { useStore } from '../store';
import { getViewer } from '../viewer/instance';
import { heightGradientCss } from '../viewer/colormaps';
import { Field, IconButton, Popover, Segmented, Select, Slider, Switch } from '../ui/primitives';

const TOOLS: { id: ViewportTool; label: string; key: string; icon: typeof MousePointer2 }[] = [
  { id: 'navigate', label: 'Move the view', key: 'V', icon: MousePointer2 },
  { id: 'box', label: 'Select with a box', key: 'B', icon: SquareDashedMousePointer },
  { id: 'lasso', label: 'Select by drawing around', key: 'L', icon: Lasso },
  { id: 'measure', label: 'Measure between two points', key: 'M', icon: Ruler },
  { id: 'brush', label: 'Smoothing brush', key: 'S', icon: Brush },
  { id: 'pivot', label: 'Pick the point to turn around', key: 'P', icon: Crosshair },
];

const VIEWS: { id: ViewName; label: string; key: string }[] = [
  { id: 'front', label: 'Front', key: '1' },
  { id: 'right', label: 'Right', key: '3' },
  { id: 'top', label: 'Top', key: '7' },
  { id: 'back', label: 'Back', key: 'Ctrl 1' },
  { id: 'left', label: 'Left', key: 'Ctrl 3' },
  { id: 'bottom', label: 'Bottom', key: 'Ctrl 7' },
];

export function ToolRail() {
  const tool = useStore(s => s.tool);
  return (
    <div className="hud tool-rail" role="toolbar" aria-label="Tools" aria-orientation="vertical" data-guide="view.tools">
      {TOOLS.map((t, i) => (
        <span key={t.id} style={{ display: 'contents' }}>
          {i === 3 && <span className="rail-sep" aria-hidden />}
          <IconButton label={`${t.label}  (${t.key})`} tip="top" data-guide={`tool.${t.id}`} active={tool === t.id} onClick={() => setTool(tool === t.id && t.id !== 'navigate' ? 'navigate' : t.id)}>
            <t.icon size={19} />
          </IconButton>
        </span>
      ))}
    </div>
  );
}

export function ViewRail() {
  const display = useStore(s => s.display);
  const setDisplay = useStore(s => s.setDisplay);
  const clip = useStore(s => s.clip);
  const visible = useStore(s => s.visible);
  const byId = useStore(s => s.byId);
  useStore(s => s.theme); // the height key follows the stage colours
  const [recording, setRecording] = useState(false);
  const [full, setFull] = useState(false);

  useEffect(() => {
    const onChange = () => setFull(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  const toggleFull = () => {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => undefined);
    else (document.querySelector('.stage') as HTMLElement | null)?.requestFullscreen().catch(() => undefined);
  };
  const scalarNames = [...new Set(visible.flatMap(id => byId.get(id)?.scalars?.map(x => x.name) ?? []))];

  const colorOptions: { value: ColorMode; label: string }[] = [
    { value: 'original', label: 'Scan colours' },
    { value: 'solid', label: 'Plain' },
    { value: 'asset', label: 'One colour per model' },
    { value: 'normal', label: 'Surface direction' },
    { value: 'height', label: 'Height' },
    ...(scalarNames.length ? [{ value: 'scalar' as ColorMode, label: 'Data (deviation, density…)' }] : []),
  ];

  const bounds = () => {
    const b = getViewer()?.sceneBounds();
    return b && !b.isEmpty() ? b : null;
  };

  const record = async () => {
    const v = getViewer();
    if (!v) return;
    if (!v.recording) {
      v.startRecording();
      setRecording(true);
      useStore.getState().toast({ kind: 'info', title: 'Recording the 3D view', body: 'Press the record button again to stop and save the video.' });
      return;
    }
    const blob = await v.stopRecording();
    setRecording(false);
    if (!blob) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `cloudclean-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.webm`;
    a.click();
  };

  return (
    <div className="hud view-rail" role="toolbar" aria-label="View" aria-orientation="vertical">
      <IconButton label="Fit the view  (F)" tip="top" data-guide="view.fit" onClick={() => viewCamera('fit')}>
        <Maximize2 size={18} />
      </IconButton>
      <Popover
        side="left"
        trigger={({ toggle, open }) => (
          <IconButton label="Standard views" data-guide="view.standard" tip={open ? undefined : 'top'} active={open} onClick={toggle}>
            <Axis3d size={18} />
          </IconButton>
        )}
      >
        {({ close }) => (
          <div style={{ width: 280 }}>
            <div className="view-grid">
              {VIEWS.map(v => (
                <button key={v.id} type="button" className="view-btn" onClick={() => { viewCamera(v.id); close(); }}>
                  <span>{v.label}</span>
                  <span className="kbd-soft">{v.key}</span>
                </button>
              ))}
            </div>
            <div className="popover-foot stack tight">
              <button type="button" className="view-btn" style={{ gridAutoFlow: 'column', justifyContent: 'center', gap: 8 }} onClick={() => { viewCamera('iso'); close(); }}>
                <span>Isometric</span><span className="kbd-soft">0</span>
              </button>
              <Segmented size="sm" ariaLabel="Projection" value={display.projection} onChange={p => setDisplay({ projection: p })} options={[{ value: 'perspective', label: 'Perspective' }, { value: 'orthographic', label: 'Orthographic' }]} />
            </div>
          </div>
        )}
      </Popover>
      <IconButton label={display.rotateStyle === 'free' ? 'Rotation: free — click for turntable' : 'Rotation: turntable — click for free'} tip="top" active={display.rotateStyle === 'turntable'} onClick={() => setDisplay({ rotateStyle: display.rotateStyle === 'free' ? 'turntable' : 'free' })}>
        <Rotate3d size={18} />
      </IconButton>
      <span className="rail-sep" aria-hidden style={{ height: 1, margin: '3px 6px', background: 'var(--line)' }} />
      <Popover
        side="left"
        trigger={({ toggle, open }) => (
          <IconButton label="Section view" data-guide="view.section" tip={open ? undefined : 'top'} active={open || clip.enabled} onClick={toggle}>
            <Scissors size={18} />
          </IconButton>
        )}
      >
        <div className="popover-body">
          <div className="popover-title">Section view</div>
          <p className="hint-text">Cuts the view (not the data) so you can look inside and measure wall thickness.</p>
          <Field label="Cut the view">
            <Switch
              checked={clip.enabled}
              onChange={enabled => {
                const b = bounds();
                const i = { x: 0, y: 1, z: 2 }[clip.axis];
                const mid = b ? (b.min.getComponent(i) + b.max.getComponent(i)) / 2 : 0;
                useStore.setState({ clip: { ...clip, enabled, position: enabled && clip.position === 0 ? mid : clip.position } });
              }}
            />
          </Field>
          <Field label="Across">
            <Segmented size="sm" value={clip.axis} onChange={axis => {
              const b = bounds();
              const i = { x: 0, y: 1, z: 2 }[axis];
              useStore.setState({ clip: { ...clip, axis, position: b ? (b.min.getComponent(i) + b.max.getComponent(i)) / 2 : 0 } });
            }} options={[{ value: 'x', label: 'X' }, { value: 'y', label: 'Y' }, { value: 'z', label: 'Z' }]} />
          </Field>
          {(() => {
            const b = bounds();
            const i = { x: 0, y: 1, z: 2 }[clip.axis];
            const lo = b ? b.min.getComponent(i) : -1, hi = b ? b.max.getComponent(i) : 1;
            return (
              <Field label="Position" inline={false}>
                <Slider value={Math.min(Math.max(clip.position, lo), hi)} min={lo} max={hi} step={(hi - lo) / 500 || 0.01} format={v => `${v.toFixed(2)} ${display.units}`} onChange={position => useStore.setState({ clip: { ...clip, position, enabled: true } })} />
              </Field>
            );
          })()}
          <Field label="Show the other side">
            <Switch checked={clip.flip} onChange={flip => useStore.setState({ clip: { ...clip, flip } })} />
          </Field>
        </div>
      </Popover>
      <Popover
        side="left"
        trigger={({ toggle, open }) => (
          <IconButton label="Display" data-guide="view.display" tip={open ? undefined : 'top'} active={open} onClick={toggle}>
            <SlidersHorizontal size={18} />
          </IconButton>
        )}
      >
        <div className="popover-body" style={{ width: 320 }}>
          <div className="popover-title"><Eye size={14} aria-hidden /> Display</div>
          <Field label="Colour">
            <Select value={display.colorMode} onChange={colorMode => {
              if (colorMode === 'scalar' && !display.scalar && scalarNames[0]) {
                setDisplay({ colorMode, scalar: { name: scalarNames[0], style: { kind: scalarNames[0] === 'deviation' ? 'diverging' : 'sequential', min: -0.4, max: 0.4, tolerance: 0.1, steps: 0 } } });
              } else setDisplay({ colorMode });
            }} options={colorOptions} />
          </Field>
          {display.colorMode === 'scalar' && scalarNames.length > 1 && (
            <Field label="Data">
              <Select value={display.scalar?.name ?? scalarNames[0]} onChange={name => setDisplay({ scalar: { name, style: display.scalar?.style ?? { kind: 'sequential', min: 0, max: 1, tolerance: 0, steps: 0 } } })} options={scalarNames.map(n => ({ value: n, label: n }))} />
            </Field>
          )}
          {display.colorMode === 'height' && (
            <div className="colour-key">
              <span className="colour-key-ramp" style={{ background: heightGradientCss(document.documentElement.dataset.theme === 'carbon') }} aria-hidden />
              <div className="row spread caption">
                <span>bottom</span>
                <span>top (up is {display.upAxis.toUpperCase()})</span>
              </div>
            </div>
          )}
          <Field label="Shade the points (show the shape)" help="Darkens edges and slopes so the shape stands out. Only the picture changes, not the scan.">
            <Switch checked={display.shade} onChange={shade => setDisplay({ shade })} label="Shade the points (show the shape)" />
          </Field>
          <Field label="Point size" inline={false}>
            <Slider value={display.pointScale} min={0.3} max={4} step={0.05} format={v => `${v.toFixed(2)}×`} onChange={pointScale => setDisplay({ pointScale })} />
          </Field>
          <Field label="Wireframe"><Switch checked={display.wireframe} onChange={wireframe => setDisplay({ wireframe })} /></Field>
          <Field label="Box around the model"><Switch checked={display.showBox} onChange={showBox => setDisplay({ showBox })} /></Field>
          <Field label="Floor grid"><Switch checked={display.showGrid} onChange={showGrid => setDisplay({ showGrid })} /></Field>
          <Field label="Up is">
            <Segmented size="sm" value={display.upAxis} onChange={upAxis => setDisplay({ upAxis })} options={[{ value: 'y', label: 'Y' }, { value: 'z', label: 'Z' }]} />
          </Field>
          <Field label="Turn around">
            <Segmented size="sm" value={display.rotatePivot} onChange={rotatePivot => setDisplay({ rotatePivot })} options={[{ value: 'center', label: 'Model centre', title: 'Turn around the middle of the model wherever you drag (as Revo Metro)' }, { value: 'cursor', label: 'Cursor', title: 'Turn around the point you press on' }]} />
          </Field>
          <Field label="Units shown" help="Labels only — the data is never converted.">
            <Select value={display.units} onChange={units => setDisplay({ units })} options={['mm', 'cm', 'm', 'in'].map(u => ({ value: u, label: u }))} />
          </Field>
        </div>
      </Popover>
      <IconButton label={display.showGrid ? 'Hide the floor grid' : 'Show the floor grid'} tip="top" active={display.showGrid} onClick={() => setDisplay({ showGrid: !display.showGrid })}>
        <Grid3x3 size={18} />
      </IconButton>
      <span className="rail-sep" aria-hidden style={{ height: 1, margin: '3px 6px', background: 'var(--line)' }} />
      <IconButton data-guide="view.fullscreen" label={full ? 'Leave full screen (Esc)' : 'Full screen 3D view'} tip="top" active={full} onClick={toggleFull}>
        {full ? <Shrink size={18} /> : <Fullscreen size={18} />}
      </IconButton>
      <IconButton label="Save a picture of the view" data-guide="view.picture" tip="top" onClick={copyScreenshot}>
        <Camera size={18} />
      </IconButton>
      <IconButton data-guide="view.record" label={recording ? 'Stop recording and save the video' : 'Record a video of the view'} tip="top" className={recording ? 'is-recording' : ''} onClick={record}>
        <span className="rec-dot" aria-hidden />
      </IconButton>
    </div>
  );
}
