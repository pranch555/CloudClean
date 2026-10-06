import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent } from 'react';
import { createPortal } from 'react-dom';
import { ChevronUp, GripVertical, Minus, Plus } from 'lucide-react';
import { useStore } from '../../store';
import { IconButton } from '../../ui/primitives';
import { lastExchange, newConversation, runningStep, useAssistant } from './assistantStore';
import { AssistantGlyph, AssistantMark } from './AssistantMark';
import { AssistantPanel, HistoryMenu, ModelPill, PanelDockIcon } from './AssistantPanel';
import {
  clampBox,
  COMPACT_H,
  defaultBubble,
  dockChat,
  expandChat,
  isSheet,
  MIN_H,
  MIN_W,
  minimizeChat,
  setBubblePos,
  setChatPos,
  setChatSize,
  useChatWindow,
  viewportSize,
  type WH,
  type XY,
} from './chatWindow';

const EDGE = 8;
const STEP = 16;
const BIG_STEP = 64;
/** a press that moves less than this is a click, not a drag */
const SLOP = 4;

// focus requests already carried out (module level: they survive the window <-> bubble swap)
let focusedInput = 0;
let focusedBubble = 0;

function useViewport(): WH {
  const [v, setV] = useState(viewportSize);
  useEffect(() => {
    const on = () => setV(viewportSize());
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, []);
  return v;
}

const clampPoint = (p: XY, w: number, h: number, v: WH): XY => ({
  x: Math.round(Math.min(Math.max(p.x, EDGE), Math.max(EDGE, v.w - w - EDGE))),
  y: Math.round(Math.min(Math.max(p.y, EDGE), Math.max(EDGE, v.h - h - EDGE))),
});

const arrow = (e: ReactKeyboardEvent): XY | null => {
  const d = e.shiftKey ? BIG_STEP : STEP;
  switch (e.key) {
    case 'ArrowLeft': return { x: -d, y: 0 };
    case 'ArrowRight': return { x: d, y: 0 };
    case 'ArrowUp': return { x: 0, y: -d };
    case 'ArrowDown': return { x: 0, y: d };
    default: return null;
  }
};

/** Follow the pointer from a press until release; `onMove` gets the offset, `onEnd` whether it was a drag. */
function track(e: ReactPointerEvent, onMove: (dx: number, dy: number) => void, onEnd: (dragged: boolean) => void) {
  const x0 = e.clientX;
  const y0 = e.clientY;
  let dragged = false;
  const move = (ev: PointerEvent) => {
    const dx = ev.clientX - x0;
    const dy = ev.clientY - y0;
    if (!dragged && Math.hypot(dx, dy) < SLOP) return;
    dragged = true;
    onMove(dx, dy);
  };
  const end = () => {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', end);
    window.removeEventListener('pointercancel', end);
    document.body.classList.remove('is-moving-chat');
    onEnd(dragged);
  };
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', end);
  window.addEventListener('pointercancel', end);
  document.body.classList.add('is-moving-chat');
}

/**
 * The chat popped out of the side panel: a small window over the app you can drag by its header, resize from its
 * corner and minimize to a bubble (Esc), so you can keep talking to the assistant while you clean, mesh or measure
 * with the step's tools in the side panel. It stays open across steps; on Home it waits until you are back.
 */
export function FloatingChat() {
  const place = useChatWindow(s => s.place);
  const minimized = useChatWindow(s => s.minimized);
  const screen = useStore(s => s.screen);
  const vp = useViewport();
  if (place !== 'floating' || screen !== 'workspace') return null;
  return createPortal(minimized ? <ChatBubble vp={vp} /> : <ChatWindow vp={vp} />, document.body);
}

function ChatWindow({ vp }: { vp: WH }) {
  const pos = useChatWindow(s => s.pos);
  const size = useChatWindow(s => s.size);
  const focusInput = useChatWindow(s => s.focusInput);
  const streaming = useAssistant(s => s.streaming);
  const ref = useRef<HTMLElement>(null);
  const titleId = useId();
  const [sheetTall, setSheetTall] = useState(false);
  const sheet = isSheet(vp);
  const box = clampBox(pos ?? { x: vp.w - size.w - 24, y: vp.h - size.h - 24 }, size, vp);
  const tall = sheet ? sheetTall : box.h > COMPACT_H + 40;

  useEffect(() => {
    if (focusInput <= focusedInput) return;
    const raf = requestAnimationFrame(() => {
      focusedInput = focusInput;
      ref.current?.querySelector<HTMLTextAreaElement>('.composer textarea')?.focus();
    });
    return () => cancelAnimationFrame(raf);
  }, [focusInput]);

  const startMove = (e: ReactPointerEvent) => {
    if (sheet || e.button !== 0) return;
    const t = e.target as Element;
    if (t.closest('button, a, input, textarea, select') && !t.closest('.chat-float-grip')) return;
    const el = ref.current;
    if (!el) return;
    e.preventDefault();
    let last: XY = { x: box.x, y: box.y };
    el.classList.add('is-dragging');
    track(
      e,
      (dx, dy) => {
        last = clampBox({ x: box.x + dx, y: box.y + dy }, { w: box.w, h: box.h }, viewportSize());
        el.style.left = `${last.x}px`;
        el.style.top = `${last.y}px`;
      },
      dragged => {
        el.classList.remove('is-dragging');
        if (dragged) setChatPos(last);
      },
    );
  };

  const startResize = (e: ReactPointerEvent) => {
    if (sheet || e.button !== 0) return;
    const el = ref.current;
    if (!el) return;
    e.preventDefault();
    const v = viewportSize();
    let last: WH = { w: box.w, h: box.h };
    el.classList.add('is-dragging');
    track(
      e,
      (dx, dy) => {
        last = { w: Math.round(Math.min(Math.max(box.w + dx, MIN_W), v.w - box.x - EDGE)), h: Math.round(Math.min(Math.max(box.h + dy, MIN_H), v.h - box.y - EDGE)) };
        el.style.width = `${last.w}px`;
        el.style.height = `${last.h}px`;
      },
      dragged => {
        el.classList.remove('is-dragging');
        if (dragged) setChatSize(last);
      },
    );
  };

  const moveKeys = (e: ReactKeyboardEvent) => {
    const d = arrow(e);
    if (!d || sheet) return;
    e.preventDefault();
    setChatPos(clampBox({ x: box.x + d.x, y: box.y + d.y }, { w: box.w, h: box.h }, vp));
  };

  const resizeKeys = (e: ReactKeyboardEvent) => {
    const d = arrow(e);
    if (!d || sheet) return;
    e.preventDefault();
    setChatSize({ w: Math.max(MIN_W, Math.min(box.w + d.x, vp.w - box.x - EDGE)), h: Math.max(MIN_H, Math.min(box.h + d.y, vp.h - box.y - EDGE)) });
  };

  // compact <-> tall, keeping the bottom edge where it is (the window grows upwards, away from the Ask bar's spot)
  const toggleSize = () => {
    if (sheet) {
      setSheetTall(!sheetTall);
      return;
    }
    // tall = the height of the 3D view, so the top bar and the steps stay in sight
    const stage = document.querySelector('.stage')?.getBoundingClientRect();
    const top = stage && stage.height > 0 ? stage.top + 8 : 72;
    const bottom = stage && stage.height > 0 ? stage.bottom - 8 : vp.h - EDGE;
    const h = tall ? COMPACT_H : Math.max(COMPACT_H + 80, Math.round(bottom - top));
    const y = tall ? box.y + box.h - h : Math.max(top, Math.min(box.y + box.h, bottom) - h);
    const next = clampBox({ x: box.x, y }, { w: box.w, h }, vp);
    setChatSize({ w: box.w, h: next.h });
    setChatPos({ x: next.x, y: next.y });
  };

  const onKeyDown = (e: ReactKeyboardEvent) => {
    if (e.key !== 'Escape' || (e.target as Element).closest('.popover')) return;
    e.preventDefault();
    e.stopPropagation();
    minimizeChat(true);
  };

  const style = sheet
    ? { left: EDGE, right: EDGE, bottom: EDGE, height: tall ? vp.h - 72 : Math.min(Math.round(vp.h * 0.56), 480) }
    : { left: box.x, top: box.y, width: box.w, height: box.h };

  return (
    <section ref={ref} className={`chat-float ${sheet ? 'is-sheet' : ''}`} style={style} role="dialog" aria-modal="false" aria-labelledby={titleId} onKeyDown={onKeyDown}>
      <header className="chat-float-head" onPointerDown={startMove} onDoubleClick={e => !(e.target as Element).closest('button') && toggleSize()}>
        {sheet ? (
          <span className="chat-float-handle" aria-hidden />
        ) : (
          <button type="button" className="chat-float-grip" aria-label="Move the chat (drag, or use the arrow keys)" data-tip="Drag to move" data-tip-side="bottom" onKeyDown={moveKeys}>
            <GripVertical size={15} aria-hidden />
          </button>
        )}
        <span className={`chat-float-title assistant-tab-glyph ${streaming ? 'is-working' : ''}`}>
          <AssistantGlyph size={16} />
        </span>
        <span className="chat-float-name" id={titleId}>Assistant</span>
        <ModelPill dotOnly />
        <span className="spacer" />
        <HistoryMenu />
        <IconButton size="sm" tip="bottom" label="New conversation" onClick={newConversation} disabled={streaming}>
          <Plus size={16} />
        </IconButton>
        <IconButton size="sm" tip="bottom" label={tall ? 'Make the chat compact' : 'Make the chat tall'} onClick={toggleSize} data-guide="chat.size">
          <HeightIcon tall={!tall} />
        </IconButton>
        <IconButton size="sm" tip="bottom" label="Dock the chat" onClick={dockChat} data-guide="chat.dock">
          <PanelDockIcon />
        </IconButton>
        <IconButton size="sm" tip="bottom" label="Minimize the chat (Esc)" onClick={() => minimizeChat(true)} data-guide="chat.minimize">
          <Minus size={16} />
        </IconButton>
      </header>
      <AssistantPanel variant="float" />
      {!sheet && (
        <button type="button" className="chat-float-resize" aria-label="Resize the chat (drag, or use the arrow keys)" data-tip="Drag to resize" onPointerDown={startResize} onKeyDown={resizeKeys}>
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
            <path d="M11 4 4 11M11 8 8 11" />
          </svg>
        </button>
      )}
    </section>
  );
}

/** The minimized chat: a small bubble you can drag anywhere. It shows when the assistant works or has a new reply. */
function ChatBubble({ vp }: { vp: WH }) {
  const bubble = useChatWindow(s => s.bubble);
  const unread = useChatWindow(s => s.unread);
  const focusBubble = useChatWindow(s => s.focusBubble);
  const streaming = useAssistant(s => s.streaming);
  const messages = useAssistant(s => s.messages);
  const ref = useRef<HTMLButtonElement>(null);
  const [dims, setDims] = useState({ w: 150, h: 48 });
  const suppressClick = useRef(false);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    if (w !== dims.w || h !== dims.h) setDims({ w, h });
  });

  useEffect(() => {
    if (focusBubble <= focusedBubble) return;
    focusedBubble = focusBubble;
    ref.current?.focus();
  }, [focusBubble]);

  const at = clampPoint(bubble ?? defaultBubble(dims.w, dims.h), dims.w, dims.h, vp);
  const ex = lastExchange(messages);
  const step = streaming ? runningStep(ex?.answer) : null;
  const state = streaming ? (step ? `Working: ${step.label}` : 'Thinking…') : unread ? 'New reply' : null;

  const onPointerDown = (e: ReactPointerEvent) => {
    if (e.button !== 0) return;
    const el = ref.current;
    if (!el) return;
    let last = at;
    track(
      e,
      (dx, dy) => {
        last = clampPoint({ x: at.x + dx, y: at.y + dy }, el.offsetWidth, el.offsetHeight, viewportSize());
        el.style.left = `${last.x}px`;
        el.style.top = `${last.y}px`;
        el.classList.add('is-dragging');
      },
      dragged => {
        el.classList.remove('is-dragging');
        if (!dragged) return;
        suppressClick.current = true;
        setBubblePos(last);
      },
    );
  };

  const spoken = streaming ? 'The assistant is working' : unread ? 'The assistant has a new reply' : '';
  return (
    <>
      <span className="visually-hidden" role="status">{spoken}</span>
      <button
        ref={ref}
        type="button"
        className={`chat-bubble ${streaming ? 'is-working' : ''} ${unread ? 'is-unread' : ''}`}
        style={{ left: at.x, top: at.y }}
        onPointerDown={onPointerDown}
        onClick={() => {
          if (suppressClick.current) {
            suppressClick.current = false;
            return;
          }
          expandChat();
        }}
        onKeyDown={e => {
          const d = arrow(e);
          if (!d) return;
          e.preventDefault();
          setBubblePos(clampPoint({ x: at.x + d.x, y: at.y + d.y }, dims.w, dims.h, vp));
        }}
        aria-label={`Open the chat${state ? ` (${state})` : ''}. Drag or use the arrow keys to move it.`}
      >
        <span className="chat-bubble-mark">
          <AssistantMark size={30} />
          {unread && !streaming && <span className="chat-bubble-dot" aria-hidden />}
        </span>
        <span className="chat-bubble-text">
          <span className="chat-bubble-name">Assistant</span>
          {state && <span className="chat-bubble-state truncate">{state}</span>}
        </span>
        <ChevronUp size={15} className="chat-bubble-open" aria-hidden />
      </button>
    </>
  );
}

/** A height dimension, as on a drawing: arrows out to the extension lines (make tall) or in to the middle (compact). */
function HeightIcon({ tall }: { tall: boolean }) {
  return (
    <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {tall ? (
        <>
          <path d="M5 3h14M5 21h14M12 6.5v11" />
          <path d="m8.5 9.5 3.5-3.5 3.5 3.5M8.5 14.5l3.5 3.5 3.5-3.5" />
        </>
      ) : (
        <>
          <path d="M5 12h14M12 3v6M12 15v6" />
          <path d="m8.5 5.5 3.5 3.5 3.5-3.5M8.5 18.5l3.5-3.5 3.5 3.5" />
        </>
      )}
    </svg>
  );
}
