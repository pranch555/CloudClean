import { useEffect, useLayoutEffect, useState } from 'react';
import { PanelLeftOpen, PanelRightOpen } from 'lucide-react';
import { api } from './lib/api';
import { useJobPolling } from './lib/jobs';
import { useThemeEffect, type ResolvedTheme } from './lib/theme';
import { panelFit } from './lib/panelFit';
import { useStore } from './store';
import { TopBar } from './shell/TopBar';
import { ModelsPanel } from './shell/ModelsPanel';
import { SidePanel } from './shell/SidePanel';
import { Gutter } from './shell/Gutter';
import { JobsDrawer } from './shell/JobsDrawer';
import { Toasts } from './shell/Toasts';
import { DropOverlay } from './shell/DropOverlay';
import { SettingsDialog } from './shell/SettingsDialog';
import { useShortcuts } from './shell/shortcuts';
import { Tour } from './shell/Tour';
import { HomeScreen } from './home/HomeScreen';
import { Viewport } from './viewport/Viewport';
import { IconButton } from './ui/primitives';
import { TooltipLayer } from './ui/TooltipLayer';
import { GuideSpotlight } from './ui/GuideSpotlight';
import { FloatingChat } from './features/assistant/FloatingChat';

export default function App() {
  const theme = useThemeEffect();
  const screen = useStore(s => s.screen);
  useJobPolling();
  useShortcuts();

  useEffect(() => {
    (async () => {
      try {
        const params = await api.params();
        useStore.setState({ params });
        const st = useStore.getState();
        await st.refreshAssets();
        await st.refreshProjects();
        useStore.setState({ booted: true });
        const s = useStore.getState();
        if (s.screen === 'workspace' && s.projectId) s.openProject(s.projectId);
        else if (s.screen === 'workspace') s.goHome();
      } catch (err) {
        useStore.getState().toast({ kind: 'error', title: 'Cannot reach the CloudClean server', body: (err as Error).message, ms: 0 });
      }
    })();
  }, []);

  return (
    <div className="app">
      <TopBar home={screen === 'home'} />
      {screen === 'home' && <HomeScreen />}
      <Workspace theme={theme} hidden={screen === 'home'} />
      <FloatingChat />
      <JobsDrawer />
      <Toasts />
      <DropOverlay />
      <SettingsDialog />
      <Tour />
      <TooltipLayer />
      <GuideSpotlight />
    </div>
  );
}

function useWindowWidth(): number {
  const [width, setWidth] = useState(() => window.innerWidth);
  useEffect(() => {
    const on = () => setWidth(window.innerWidth);
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, []);
  return width;
}

/**
 * The workspace stays mounted behind Home so loaded models (and the WebGL context) survive a trip home.
 * Side panels fit the window: docked panels shrink to leave the 3D view room, and when the window is too narrow
 * the model list, then the step panel, become drawers that slide over the view (see lib/panelFit).
 */
function Workspace({ theme, hidden }: { theme: ResolvedTheme; hidden: boolean }) {
  const layout = useStore(s => s.layout);
  const drawers = useStore(s => s.drawers);
  const setLayout = useStore(s => s.setLayout);
  const width = useWindowWidth();
  const fit = panelFit(width, layout);
  useLayoutEffect(() => {
    useStore.getState().setPanelModes({ left: fit.left, right: fit.right });
  }, [fit.left, fit.right]);

  const leftDocked = fit.left === 'dock' && layout.leftOpen;
  const rightDocked = fit.right === 'dock' && layout.rightOpen;
  const leftDrawer = fit.left === 'drawer' && drawers.left;
  const rightDrawer = fit.right === 'drawer' && drawers.right;
  const columns = [
    leftDocked ? `${fit.leftWidth}px` : '0px',
    leftDocked ? '10px' : '0px',
    'minmax(0, 1fr)',
    rightDocked ? '10px' : '0px',
    rightDocked ? `${fit.rightWidth}px` : '0px',
  ].join(' ');
  // a click on the 3D view puts an open drawer away (the click still reaches the view)
  const closeDrawers = () => {
    if (leftDrawer || rightDrawer) setLayout({ ...(leftDrawer ? { leftOpen: false } : {}), ...(rightDrawer ? { rightOpen: false } : {}) });
  };

  return (
    <div className="ws-body" style={{ gridTemplateColumns: columns, display: hidden ? 'none' : undefined }}>
      {leftDocked ? <ModelsPanel /> : <div className="panel-stub" aria-hidden />}
      {leftDocked ? <Gutter panel="left" label="the model list" /> : <div className="panel-stub" aria-hidden />}
      <main className="stage" aria-label="3D view" onPointerDownCapture={closeDrawers}>
        <Viewport theme={theme} />
        {!leftDocked && !leftDrawer && (
          <IconButton label="Show the model list (Ctrl B)" className="hud edge-btn edge-left" onClick={() => setLayout({ leftOpen: true })}>
            <PanelLeftOpen size={18} />
          </IconButton>
        )}
        {!rightDocked && !rightDrawer && (
          <IconButton label="Show the side panel (Ctrl I)" className="hud edge-btn edge-right" onClick={() => setLayout({ rightOpen: true })}>
            <PanelRightOpen size={18} />
          </IconButton>
        )}
      </main>
      {rightDocked ? <Gutter panel="right" grow={-1} label="the side panel" /> : <div className="panel-stub" aria-hidden />}
      {rightDocked ? <SidePanel /> : <div className="panel-stub" aria-hidden />}
      {leftDrawer && (
        <div className="drawer drawer-left" style={{ width: fit.leftDrawer }}>
          <ModelsPanel />
        </div>
      )}
      {rightDrawer && (
        <div className="drawer drawer-right" style={{ width: fit.rightDrawer }}>
          <SidePanel />
        </div>
      )}
    </div>
  );
}
