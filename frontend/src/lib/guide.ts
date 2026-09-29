import { create } from 'zustand';
import { useStore } from '../store';
import { setTab, setUi, type MeasureTab, type ToolId } from '../steps/measure/state';
import type { Step } from './types';

/**
 * "Take me there": the assistant's app guide (cloudclean/assistant/guide.py) sends a feature id, its name, where it
 * is and how to get there. This opens that place and rings the control carrying `data-guide="<id>"`.
 */
export interface GuideNav {
  screen?: 'home' | 'workspace';
  step?: Step;
  right?: 'step' | 'assistant';
  left?: boolean;
  measure_tab?: MeasureTab;
  measure_tool?: ToolId;
  settings?: string;
  jobs?: boolean;
}

export interface GuideSpot {
  id: string;
  label: string;
  where: string;
  /** set when the control could not be found on screen: the spotlight shows the path instead */
  missing?: boolean;
  at: number;
}

export const useGuide = create<{ spot: GuideSpot | null }>(() => ({ spot: null }));

export const clearGuide = () => useGuide.setState({ spot: null });

function navigate(nav: GuideNav) {
  const st = useStore.getState();
  if (nav.settings) {
    st.set({ settingsOpen: nav.settings as never });
    return;
  }
  if (nav.jobs) st.set({ jobsOpen: true });
  if (nav.screen === 'home') {
    st.goHome();
    return;
  }
  if (st.screen !== 'workspace' && (nav.screen === 'workspace' || nav.step || nav.left || nav.right)) {
    if (st.projectId) st.openProject(st.projectId, nav.step);
    else st.set({ screen: 'workspace' });
  }
  if (nav.step) useStore.getState().goStep(nav.step);
  if (nav.right) {
    useStore.getState().set({ rightTab: nav.right });
    useStore.getState().setLayout({ rightOpen: true });
  }
  if (nav.left) useStore.getState().setLayout({ leftOpen: true });
  if (nav.measure_tab) setTab(nav.measure_tab);
  if (nav.measure_tool) setUi({ tool: nav.measure_tool });
}

const visible = (el: Element) => {
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
};

/** Open the place of a feature and ring its control; if the control never shows up, show where it is instead. */
export function guideTo(id: string, label: string, where: string, nav: GuideNav = {}) {
  navigate(nav);
  const started = performance.now();
  const find = () => [...document.querySelectorAll(`[data-guide~="${CSS.escape(id)}"]`)].find(visible);
  const poll = () => {
    const el = find();
    if (el) {
      el.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
      useGuide.setState({ spot: { id, label, where, at: Date.now() } });
      return;
    }
    // panels mount after navigating: give them a moment
    if (performance.now() - started < 2500) setTimeout(poll, 120);
    else useGuide.setState({ spot: { id, label, where, missing: true, at: Date.now() } });
  };
  setTimeout(poll, 60);
}
