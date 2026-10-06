import { create } from 'zustand';
import { local, useStore } from '../../store';
import type { GroupId } from './groups';

interface PanelUi {
  /** groups folded shut (kept between visits) */
  collapsed: GroupId[];
  /** a model to bring into view and focus in the list ("Made from" links); `n` re-triggers the same id */
  reveal: { id: string; n: number } | null;
}

export const usePanelUi = create<PanelUi>(() => ({
  collapsed: local.get<GroupId[]>('models.collapsed', []),
  reveal: null,
}));

export function setCollapsed(id: GroupId, collapsed: boolean) {
  const cur = usePanelUi.getState().collapsed;
  const next = collapsed ? [...new Set([...cur, id])] : cur.filter(g => g !== id);
  usePanelUi.setState({ collapsed: next });
  local.set('models.collapsed', next);
}

/** Work on a model and bring its row into view (its group opens). */
export function revealModel(id: string) {
  useStore.getState().activate(id);
  usePanelUi.setState(s => ({ reveal: { id, n: (s.reveal?.n ?? 0) + 1 } }));
}
