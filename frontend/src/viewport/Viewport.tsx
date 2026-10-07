import { useEffect, useRef, useState } from 'react';
import { ScanLine, Upload } from 'lucide-react';
import { Viewer } from '../viewer/Viewer';
import { getViewer, setViewer } from '../viewer/instance';
import { useProjectAssets, useStore } from '../store';
import { assetColorVar } from '../shell/ModelsPanel';
import { pickFiles } from '../lib/importing';
import { cssVar, type ResolvedTheme } from '../lib/theme';
import { ensureThumbnail } from '../lib/thumbnails';
import { Button, Empty } from '../ui/primitives';
import { userTookTheView } from '../features/capture/captureStore';
import { GuidanceHud } from '../features/capture/GuidanceHud';
import { PhotoOverlay } from '../features/colour/ColourSection';
import { ToolRail, ViewRail } from './Rails';
import { SizeTag } from './SizeTag';
import { Legend } from './Legend';
import { Labels } from './Labels';
import { GoldenPins } from './GoldenPins';
import { GoldenDimension } from './GoldenDimension';
import { SelectionOverlay } from './SelectionOverlay';
import { BrushOverlay } from './BrushOverlay';
import { PairOverlay } from './PairOverlay';
import { PaneOverlay } from './PaneOverlay';
import { Banners } from './Banners';
import { AskBar } from './AskBar';

const hexToRgb = (hex: string): [number, number, number] => {
  const h = hex.replace('#', '');
  return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255) as [number, number, number];
};

export function Viewport({ theme }: { theme: ResolvedTheme }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const visible = useStore(s => s.visible);
  const byId = useStore(s => s.byId);
  const assets = useStore(s => s.assets);
  const activeId = useStore(s => s.activeId);
  const display = useStore(s => s.display);
  const tool = useStore(s => s.tool);
  const clip = useStore(s => s.clip);
  const step = useStore(s => s.step);
  const panes = useStore(s => s.panes);
  const projectAssets = useProjectAssets();

  useEffect(() => {
    const v = new Viewer(canvasRef.current!, hostRef.current!);
    v.onLoading = names => useStore.setState({ loadingNames: names });
    v.onItemLoaded = id => ensureThumbnail(id);
    v.onInteract = userTookTheView;
    setViewer(v);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (import.meta.env.DEV) (window as any).__cc = { viewer: v, store: useStore };
    setReady(true);
    return () => {
      setViewer(null);
      v.dispose();
    };
  }, []);

  // theme → viewer (colours come from the CSS tokens so there is one source of truth)
  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    const grid = cssVar('--vp-grid', '0.2, 0.18, 0.15').split(',').map(Number) as [number, number, number];
    v.setTheme({
      dark: theme === 'carbon',
      points: cssVar('--vp-points', '#8c8579'),
      mesh: cssVar('--vp-mesh', '#b3ac9f'),
      highlight: cssVar('--vp-highlight', '#ee4b1f'),
      grid: grid.length === 3 && grid.every(Number.isFinite) ? grid : hexToRgb('#333333'),
      bgTop: cssVar('--vp-top', '#f8f6f2'),
      bgBottom: cssVar('--vp-bottom', '#e2ded4'),
    });
  }, [ready, theme]);

  // store → viewer
  useEffect(() => {
    const v = getViewer();
    if (!v) return;
    v.colorFor = id => cssVar(assetColorVar(useStore.getState().assets, id), '#2f6fd6');
    const metas = visible.map(id => byId.get(id)).filter(a => a && a.kind !== 'image') as NonNullable<ReturnType<typeof byId.get>>[];
    const first = v.visibleCount() === 0;
    v.sync(metas, activeId)
      .then(current => {
        if (current && first && metas.length) v.fit(undefined, undefined, false);
      })
      .catch(err => useStore.getState().toast({ kind: 'error', title: 'Could not show the model', body: String(err?.message ?? err) }));
  }, [ready, visible, byId, activeId, assets]);

  useEffect(() => {
    getViewer()?.setSettings({ colorMode: display.colorMode, pointScale: display.pointScale, wireframe: display.wireframe, showBox: display.showBox, showGrid: display.showGrid, scalar: display.scalar });
  }, [ready, display.colorMode, display.pointScale, display.wireframe, display.showBox, display.showGrid, display.scalar]);

  useEffect(() => {
    const v = getViewer();
    v?.setSettings({ shade: display.shade });
    v?.invalidate(); // also with only the live scan on screen
  }, [ready, display.shade]);

  useEffect(() => {
    getViewer()?.setUpAxis(display.upAxis);
  }, [ready, display.upAxis]);

  useEffect(() => {
    getViewer()?.setRotateStyle(display.rotateStyle);
  }, [ready, display.rotateStyle]);

  useEffect(() => {
    getViewer()?.setRotatePivot(display.rotatePivot);
  }, [ready, display.rotatePivot]);

  useEffect(() => {
    getViewer()?.setProjection(display.projection);
  }, [ready, display.projection]);

  useEffect(() => {
    const v = getViewer();
    if (v) v.toolActive = tool !== 'navigate';
  }, [ready, tool]);

  useEffect(() => {
    getViewer()?.setClip(clip);
  }, [ready, clip]);

  useEffect(() => {
    getViewer()?.setLiveVisible(step === 'capture');
  }, [ready, step]);

  useEffect(() => {
    getViewer()?.setPanes(panes);
  }, [ready, panes]);

  const geometryCount = projectAssets.filter(a => a.kind !== 'image').length;
  const showEmpty = ready && visible.length === 0 && step !== 'capture';

  return (
    <div className={`viewport tool-${tool}`} ref={hostRef}>
      <canvas ref={canvasRef} className="viewport-canvas" />
      {ready && (
        <>
          <PhotoOverlay />
          <Labels />
          <GoldenPins />
          <GoldenDimension />
          <SelectionOverlay />
          <BrushOverlay />
          <PairOverlay />
          <PaneOverlay />
          <SizeTag />
          <ToolRail />
          <ViewRail />
          <Legend />
          <GuidanceHud />
          <Banners />
          <AskBar />
        </>
      )}
      {showEmpty && (
        <div className="stage-empty">
          {geometryCount ? (
            <Empty title="Nothing on the stage">Click a model on the left, or its eye, to show it here.</Empty>
          ) : (
            <Empty
              icon={<ScanLine size={24} />}
              title="Bring in your first part"
              action={
                <div className="row">
                  <Button variant="primary" icon={<ScanLine size={16} />} onClick={() => useStore.getState().goStep('capture')}>Scan a part</Button>
                  <Button icon={<Upload size={16} />} onClick={() => pickFiles()}>Open scan files</Button>
                </div>
              }
            >
              Scan with the MetroY, or drop PLY, STL, OBJ or STEP files anywhere on the window.
            </Empty>
          )}
        </div>
      )}
    </div>
  );
}
