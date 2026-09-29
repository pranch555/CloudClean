import type { ResolvedRegion, Step, ViewName, ViewportTool } from './types';
import { setTool } from './actions';
import { uid } from './uid';
import { useStore, type DimLine, type Vec3 } from '../store';
import { getViewer } from '../viewer/instance';
import { useCapture } from '../features/capture/captureStore';
import { setTab as setMeasureTab, type MeasureTab } from '../steps/measure/state';
import { guideTo, type GuideNav } from './guide';

/**
 * Apply a UI event from the assistant (docs/v3-plan.md Contract 7). Everything the user can do with the mouse the
 * assistant can do here; unknown actions are ignored so an older browser never breaks on a newer server.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function applyUi(d: any) {
  const st = useStore.getState();
  const v = getViewer();
  const refresh = () => st.refreshAssets().catch(() => undefined);
  switch (d?.action) {
    case 'show':
      refresh().then(() => {
        const ids = (d.asset_ids ?? []).filter((id: string) => useStore.getState().byId.has(id));
        useStore.setState(s => ({ visible: d.exclusive === false ? [...new Set([...s.visible, ...ids])] : ids, activeId: ids[ids.length - 1] ?? s.activeId, screen: 'workspace' }));
        setTimeout(() => getViewer()?.fit(ids.length ? ids : undefined), 150);
      });
      break;
    case 'focus':
      refresh().then(() => {
        if (d.asset_id) useStore.getState().activate(d.asset_id);
        setTimeout(() => getViewer()?.fit(d.asset_id ? [d.asset_id] : undefined), 150);
      });
      break;
    case 'select':
      useStore.setState({ selected: d.asset_ids ?? [] });
      break;
    case 'display': {
      const patch: Record<string, unknown> = {};
      if (d.color_mode) patch.colorMode = d.color_mode;
      if (d.color_mode === 'scalar' && d.scalar) {
        patch.scalar = { name: d.scalar, style: d.scalar === 'deviation' ? { kind: 'diverging', min: -0.4, max: 0.4, tolerance: 0.1, steps: 0 } : { kind: 'sequential', min: 0, max: 1, tolerance: 0, steps: 0 } };
      }
      if (d.point_scale != null) patch.pointScale = Number(d.point_scale);
      if (d.wireframe != null) patch.wireframe = !!d.wireframe;
      if (d.show_grid != null) patch.showGrid = !!d.show_grid;
      if (d.show_box != null) patch.showBox = !!d.show_box;
      if (d.projection === 'perspective' || d.projection === 'orthographic') patch.projection = d.projection;
      st.setDisplay(patch);
      if (d.theme === 'paper' || d.theme === 'carbon' || d.theme === 'system') st.set({ theme: d.theme });
      break;
    }
    case 'camera':
      if (!v) break;
      if (d.view) v.fit(d.asset_ids?.length ? d.asset_ids : undefined, d.view as ViewName);
      else if (d.asset_ids?.length) v.fit(d.asset_ids);
      if (d.orbit) setTimeout(() => v.orbit(Number(d.orbit.yaw_deg ?? 0), Number(d.orbit.pitch_deg ?? 0)), d.view ? 460 : 0);
      if (d.zoom) setTimeout(() => v.zoomBy(Number(d.zoom)), d.view || d.orbit ? 520 : 0);
      if (Array.isArray(d.look_at) && d.look_at.length === 3) v.lookAtPoint(d.look_at as [number, number, number]);
      break;
    case 'navigate':
      if (d.screen === 'home') st.goHome();
      else if (d.step) st.goStep(d.step as Step);
      else if (d.screen === 'workspace') st.set({ screen: 'workspace' });
      if (d.panel === 'settings') st.set({ settingsOpen: 'appearance' });
      if (['dimensions', 'thread', 'cad', 'accuracy'].includes(d.panel)) {
        if (!d.step) st.goStep('measure');
        setMeasureTab(d.panel as MeasureTab);
      }
      break;
    case 'open': {
      // legacy panel names from older servers
      const map: Record<string, Step> = { process: 'clean', inspect: 'measure', capture: 'capture', autopilot: 'export' };
      if (map[d.panel]) st.goStep(map[d.panel]);
      break;
    }
    case 'guide':
      // the assistant's app guide: open the place of a feature and ring its control
      if (d.feature) guideTo(String(d.feature), String(d.label ?? d.feature), String(d.where ?? ''), (d.nav ?? {}) as GuideNav);
      break;
    case 'tool':
      if (['navigate', 'box', 'lasso', 'measure', 'brush', 'pivot'].includes(d.tool)) setTool(d.tool as ViewportTool);
      break;
    case 'section':
      useStore.setState(s => ({ clip: { ...s.clip, ...(d.enabled != null ? { enabled: !!d.enabled } : {}), ...(d.axis ? { axis: d.axis } : {}), ...(d.position != null ? { position: Number(d.position) } : {}), ...(d.flip != null ? { flip: !!d.flip } : {}) } }));
      break;
    case 'highlight': {
      const region = d.resolved as ResolvedRegion | undefined;
      if (!v || !region?.shapes?.length) break;
      const counts = v.highlightRegion(d.asset_id ?? null, region);
      const total = Object.values(counts).reduce((a, b) => a + b, 0);
      useStore.setState({
        selection: { view_projection: [], polygon: [], asset_id: d.asset_id, count: total, region: d.region ?? undefined, label: d.label ?? 'Highlighted region' },
        selectionCounts: counts,
        screen: 'workspace',
      });
      break;
    }
    case 'clear_highlight':
      v?.clearSelection();
      useStore.setState({ selection: null, selectionCounts: {} });
      break;
    case 'measure_overlay': {
      const items: DimLine[] = (d.items ?? []).filter((it: { a?: unknown; b?: unknown }) => Array.isArray(it.a) && Array.isArray(it.b)).map((it: { label?: string; kind?: string; value?: number; unit?: string; a: Vec3; b: Vec3; asset_id?: string }) => ({
        id: uid(),
        label: it.label ?? 'Dimension',
        kind: it.kind ?? 'distance',
        value: it.value ?? null,
        unit: it.unit ?? st.display.units,
        a: it.a,
        b: it.b,
        assetId: it.asset_id,
        source: 'assistant' as const,
      }));
      useStore.setState(s => ({ dims: d.replace ? [...s.dims.filter(x => x.source !== 'assistant'), ...items] : [...s.dims, ...items] }));
      break;
    }
    case 'annotate':
      useStore.setState(s => ({ annotations: [...s.annotations, ...(d.items ?? []).filter((it: { position?: unknown }) => Array.isArray(it.position)).map((it: { position: Vec3; text: string }) => ({ id: uid(), position: it.position, text: String(it.text ?? '') }))] }));
      break;
    case 'clear_annotations':
      useStore.setState({ annotations: [], dims: st.dims.filter(x => x.source !== 'assistant') });
      break;
    case 'follow_scanner':
      useCapture.setState({ follow: !!d.on });
      break;
    case 'layout':
      if (d.assets_open != null) st.setLayout({ leftOpen: !!d.assets_open });
      if (d.panel_open != null) st.setLayout({ rightOpen: !!d.panel_open });
      if (d.assistant_open != null) st.set({ rightTab: d.assistant_open ? 'assistant' : 'step' });
      break;
    case 'project':
      if (d.project_id) st.refreshProjects().then(() => useStore.getState().openProject(d.project_id)).catch(() => undefined);
      break;
    default:
      break;
  }
}

/** What the assistant is told about the screen with every message (Contract 7 context). */
export function uiContext() {
  const st = useStore.getState();
  const v = getViewer();
  const done = st.measurements.filter(m => m.b && m.result);
  return {
    active_id: st.activeId,
    selected_ids: st.selected,
    visible_ids: st.visible,
    units: st.display.units,
    project_id: st.projectId,
    screen: st.screen,
    step: st.step,
    tool: st.tool,
    theme: st.theme,
    camera: v?.cameraState(),
    section: st.clip.enabled ? st.clip : { enabled: false },
    measurements: [
      ...done.map(m => ({ label: m.label, value: +(m.result!.distance).toFixed(4), a: m.a, b: m.b })),
      ...st.dims.map(dm => ({ label: dm.label, value: dm.value, a: dm.a, b: dm.b })),
    ].slice(-12),
    highlight: st.selection?.region ? { asset_id: st.selection.asset_id, label: st.selection.label, count: st.selection.count } : undefined,
    selection: st.selection && st.selection.polygon.length ? { view_projection: st.selection.view_projection, polygon: st.selection.polygon, asset_ids: Object.keys(st.selectionCounts), count: st.selection.count } : undefined,
  };
}
