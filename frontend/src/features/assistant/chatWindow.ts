import { create } from 'zustand';
import { local, useStore } from '../../store';

/*
 * Where the conversation with the assistant lives, and the floating chat window's place on screen.
 *
 *   docked    the full chat is the side panel's Assistant tab; under the 3D view the Ask bar shows the latest
 *             reply in a small box (when the Assistant tab is not on screen)
 *   floating  the chat is a small window over the app (drag it by its header, resize it from its corner) that stays
 *             open while you work in any step; minimized it is a bubble you can drag anywhere. The Ask bar steps
 *             aside, and the Assistant tab offers to dock the chat back.
 *
 * Position, size and minimized are remembered in this browser (localStorage).
 */

export interface XY {
  x: number;
  y: number;
}
export interface WH {
  w: number;
  h: number;
}

interface ChatWindowState {
  place: 'docked' | 'floating';
  minimized: boolean;
  /** top-left corner of the floating window in the browser window (px); null = not placed yet */
  pos: XY | null;
  size: WH;
  /** top-left corner of the minimized bubble; null = its default spot (where the Ask bar is) */
  bubble: XY | null;
  /** a reply finished while the chat was minimized */
  unread: boolean;
  /** index of the assistant message whose reply box was hidden: it shows again with the next reply */
  replyHidden: number;
  /** the reply box under the 3D view shows more of the answer */
  replyTall: boolean;
  /** one-shot focus requests (bumped counters, so asking twice focuses twice) */
  focusInput: number;
  focusBubble: number;
}

export const MIN_W = 300;
export const MIN_H = 320;
export const COMPACT_H = 460;
const DEFAULT_W = 392;
const EDGE = 8;

type Saved = Pick<ChatWindowState, 'place' | 'minimized' | 'pos' | 'size' | 'bubble' | 'replyTall'>;
const saved = local.get<Partial<Saved>>('chatWindow', {});
const num = (v: unknown, fallback: number) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);
const xy = (v: unknown): XY | null => (v && typeof v === 'object' && 'x' in v && 'y' in v ? { x: num((v as XY).x, 0), y: num((v as XY).y, 0) } : null);

export const useChatWindow = create<ChatWindowState>(() => ({
  place: saved.place === 'floating' ? 'floating' : 'docked',
  minimized: !!saved.minimized,
  pos: xy(saved.pos),
  size: { w: Math.max(MIN_W, num(saved.size?.w, DEFAULT_W)), h: Math.max(MIN_H, num(saved.size?.h, COMPACT_H)) },
  bubble: xy(saved.bubble),
  unread: false,
  replyHidden: -1,
  replyTall: !!saved.replyTall,
  focusInput: 0,
  focusBubble: 0,
}));

useChatWindow.subscribe(s => {
  const keep: Saved = { place: s.place, minimized: s.minimized, pos: s.pos, size: s.size, bubble: s.bubble, replyTall: s.replyTall };
  local.set('chatWindow', keep);
});

/** The browser window (layout viewport) size. */
export const viewportSize = (): WH => ({ w: document.documentElement.clientWidth || window.innerWidth, h: window.innerHeight });

/** Below this the floating chat becomes a sheet along the bottom of the window. */
export const isSheet = (v: WH = viewportSize()) => v.w < 720 || v.h < 520;

/** Keep a box of size `size` at `pos` inside the browser window (it shrinks first when the window is smaller). */
export function clampBox(pos: XY, size: WH, v: WH = viewportSize()): XY & WH {
  const w = Math.max(Math.min(size.w, v.w - 2 * EDGE), Math.min(MIN_W, v.w - 2 * EDGE));
  const h = Math.max(Math.min(size.h, v.h - 2 * EDGE), Math.min(MIN_H, v.h - 2 * EDGE));
  return { w, h, x: Math.round(Math.min(Math.max(pos.x, EDGE), v.w - w - EDGE)), y: Math.round(Math.min(Math.max(pos.y, EDGE), v.h - h - EDGE)) };
}

const stageRect = () => document.querySelector('.stage')?.getBoundingClientRect() ?? null;

/** First placement: over the right of the 3D view, beside the view buttons and down at the bottom. */
function defaultPos(size: WH): XY {
  const v = viewportSize();
  const r = stageRect();
  if (!r || r.width < size.w + 160) return { x: v.w - size.w - 24, y: v.h - size.h - 24 };
  return { x: r.right - size.w - 76, y: r.bottom - size.h - 16 };
}

/** Where the bubble sits until it is dragged: where the Ask bar is, at the bottom centre of the 3D view. */
export function defaultBubble(w: number, h: number): XY {
  const v = viewportSize();
  const r = stageRect();
  if (!r || r.width === 0) return { x: v.w / 2 - w / 2, y: v.h - h - 24 };
  return { x: r.left + r.width / 2 - w / 2, y: r.bottom - h - 18 };
}

/** Pop the chat out into the floating window (or bring it up when it is already out). */
export function popOutChat() {
  const s = useChatWindow.getState();
  const st = useStore.getState();
  if (st.screen !== 'workspace') st.set({ screen: 'workspace' });
  // the side panel goes back to the step's tools: that is the point of popping the chat out
  if (st.rightTab === 'assistant') st.set({ rightTab: 'step' });
  if (s.place === 'floating') {
    expandChat();
    return;
  }
  const size = s.size.h > MIN_H ? s.size : { w: DEFAULT_W, h: COMPACT_H };
  const pos = s.pos ?? defaultPos(size);
  const at = clampBox(pos, size);
  useChatWindow.setState({ place: 'floating', minimized: false, unread: false, pos: { x: at.x, y: at.y }, size, focusInput: s.focusInput + 1 });
}

/** Put the chat back into the side panel's Assistant tab. */
export function dockChat() {
  useChatWindow.setState({ place: 'docked', minimized: false, unread: false });
  const st = useStore.getState();
  allowAssistantTab(() => st.set({ rightTab: 'assistant', screen: 'workspace' }));
  st.setLayout({ rightOpen: true });
  setTimeout(() => document.querySelector<HTMLTextAreaElement>('.side .composer textarea')?.focus(), 80);
}

export function minimizeChat(focusBubble = false) {
  const s = useChatWindow.getState();
  useChatWindow.setState({ minimized: true, unread: false, focusBubble: focusBubble ? s.focusBubble + 1 : s.focusBubble });
}

export function expandChat(focus = true) {
  const s = useChatWindow.getState();
  useChatWindow.setState({ minimized: false, unread: false, focusInput: focus ? s.focusInput + 1 : s.focusInput });
}

export const setChatPos = (pos: XY) => useChatWindow.setState({ pos: { x: pos.x, y: pos.y } });
export const setChatSize = (size: WH) => useChatWindow.setState({ size: { w: size.w, h: size.h } });
export const setBubblePos = (bubble: XY) => useChatWindow.setState({ bubble: { x: bubble.x, y: bubble.y } });
export const hideReply = (index: number) => useChatWindow.setState({ replyHidden: index });
export const showReply = () => useChatWindow.setState({ replyHidden: -1 });
export const setReplyTall = (replyTall: boolean) => useChatWindow.setState({ replyTall });

/** The chat is floating and on screen as the window (not the bubble). */
export const chatWindowOpen = () => {
  const s = useChatWindow.getState();
  return s.place === 'floating' && !s.minimized;
};

/** A reply finished: when the floating chat is minimized its bubble says so. */
export function noteReplyDone() {
  const s = useChatWindow.getState();
  if (s.place === 'floating' && s.minimized) useChatWindow.setState({ unread: true });
}

/*
 * While the chat floats, "open the assistant" requests from anywhere (the Ask CloudClean button, Home, a step's
 * "ask the assistant" button, the app guide) bring the floating chat up instead of swapping the step's tools out
 * of the side panel. Only a click on the side panel's own Assistant tab shows that tab (with "Dock it here").
 */
let tabAllowed = false;

export function allowAssistantTab(fn: () => void) {
  tabAllowed = true;
  try {
    fn();
  } finally {
    tabAllowed = false;
  }
}

useStore.subscribe((s, prev) => {
  if (tabAllowed || s.rightTab !== 'assistant' || prev.rightTab === 'assistant') return;
  if (useChatWindow.getState().place !== 'floating') return;
  useStore.setState({ rightTab: 'step' });
  expandChat();
});
