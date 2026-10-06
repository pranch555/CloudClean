import { useEffect } from 'react';
import { applySelection, clearSelection, redoModel, setTool, toggleProjection, undoModel, viewCamera } from '../lib/actions';
import { STEPS } from '../lib/journey';
import { useStore, panelShown } from '../store';
import { focusAsk } from '../features/assistant/assistantStore';
import { chatWindowOpen, expandChat, minimizeChat, useChatWindow } from '../features/assistant/chatWindow';

/*
 * Keyboard map (numpad-style views like CAD packages):
 *   F fit · 1 front · 3 right · 7 top · Ctrl+1/3/7 back/left/bottom · 0 iso · 5 perspective/ortho
 *   V move view · B box · L lasso · M measure · S smoothing brush · P pick pivot · Del delete selection · Esc cancel
 *   Ctrl+Z / Ctrl+Y step back / forward through the models made from each other
 *   Alt+1..6 journey steps · Ctrl+K ask · Ctrl+J assistant · Ctrl+B model list · Ctrl+I side panel · ` jobs
 *   (while the chat is popped out, Ctrl+J opens or minimizes the floating chat, and Esc in it minimizes it)
 */
export function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const st = useStore.getState();
      const typing = (e.target as HTMLElement).closest('input, textarea, select, [contenteditable="true"]');
      const mod = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();
      if (mod && key === 'k') {
        e.preventDefault();
        focusAsk();
        return;
      }
      if (mod && key === 'b') {
        e.preventDefault();
        st.setLayout({ leftOpen: !panelShown(st, 'left') });
        return;
      }
      if (mod && key === 'i') {
        e.preventDefault();
        st.setLayout({ rightOpen: !panelShown(st, 'right') });
        return;
      }
      if (mod && e.key === '.') {
        e.preventDefault();
        const hide = panelShown(st, 'left') || panelShown(st, 'right');
        st.setLayout({ leftOpen: !hide, rightOpen: !hide });
        return;
      }
      if (mod && key === 'j') {
        e.preventDefault();
        if (useChatWindow.getState().place === 'floating') {
          if (st.screen !== 'workspace') st.set({ screen: 'workspace' });
          if (chatWindowOpen() && st.screen === 'workspace') minimizeChat(true);
          else expandChat();
          return;
        }
        const showing = st.rightTab === 'assistant' && panelShown(st, 'right');
        st.set({ rightTab: showing ? 'step' : 'assistant', screen: 'workspace' });
        st.setLayout({ rightOpen: true });
        return;
      }
      if (e.altKey && /^[1-6]$/.test(e.key)) {
        e.preventDefault();
        st.goStep(STEPS[Number(e.key) - 1].id);
        return;
      }
      if (typing || st.settingsOpen || st.screen !== 'workspace') return;
      if (mod && key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undoModel();
        return;
      }
      if (mod && (key === 'y' || (key === 'z' && e.shiftKey))) {
        e.preventDefault();
        redoModel();
        return;
      }
      const k = e.key;
      if (k === 'Escape') {
        if (st.selection) clearSelection();
        else if (st.tool !== 'navigate') setTool('navigate');
        else if (st.jobsOpen) st.set({ jobsOpen: false });
        return;
      }
      if (mod) {
        if (k === '1') viewCamera('back');
        else if (k === '3') viewCamera('left');
        else if (k === '7') viewCamera('bottom');
        else return;
        e.preventDefault();
        return;
      }
      switch (k) {
        case 'f': case 'F': viewCamera('fit'); break;
        case '1': viewCamera('front'); break;
        case '3': viewCamera('right'); break;
        case '7': viewCamera('top'); break;
        case '0': viewCamera('iso'); break;
        case '5': toggleProjection(); break;
        case 'v': case 'V': setTool('navigate'); break;
        case 'b': case 'B': setTool('box'); break;
        case 'l': case 'L': setTool('lasso'); break;
        case 'm': case 'M': setTool('measure'); break;
        case 'p': case 'P': setTool('pivot'); break;
        case 's': case 'S': setTool('brush'); break;
        case '`': st.set({ jobsOpen: !st.jobsOpen }); break;
        case 'Delete': case 'Backspace':
          // in Measure a selection is what you measure, never something to delete by accident
          if (st.selection && st.step !== 'measure') applySelection('delete', false);
          break;
        default: return;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}
