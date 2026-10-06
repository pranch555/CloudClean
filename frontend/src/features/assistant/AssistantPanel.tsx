import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, ArrowUp, Brain, Check, ChevronDown, ChevronRight, History, Image as ImageIcon, Loader2, PictureInPicture2, Plus, Settings2, Square, Trash2, X } from 'lucide-react';
import { api } from '../../lib/api';
import { humanize } from '../../lib/format';
import type { Step } from '../../lib/types';
import { useStore } from '../../store';
import { getViewer } from '../../viewer/instance';
import { Badge, Button, IconButton, Popover, Progress } from '../../ui/primitives';
import { AttachButton, AttachmentStrip, attachFiles, pasteImages } from './AttachControls';
import { AssistantGlyph, AssistantMark } from './AssistantMark';
import { loadConversation, newConversation, refreshAssistantStatus, sendToAssistant, stopAssistant, useAssistant, type ConfirmRequest, type Msg, type ToolRecord } from './assistantStore';
import { dockChat, expandChat, popOutChat, useChatWindow } from './chatWindow';
import { renderMarkdown } from './markdown';

export const SUGGESTIONS: Record<Step, string[]> = {
  capture: [
    'Connect the scanner and start a turntable scan: 15° steps, one turn flat and one tilted 20°',
    'Rotate the turntable 45°',
    'How is the scan going? What is still missing?',
  ],
  clean: [
    'Clean this scan and tell me how much it removed',
    'Remove the table under the part',
    'Delete the points I selected',
  ],
  align: [
    'Should I merge these scans? Check first',
    'Show the scans side by side',
  ],
  mesh: [
    'Build a watertight mesh and show it',
    'Why is the mesh bumpy? Suggest better settings',
  ],
  measure: [
    'What are the length, width and height of this part?',
    'Measure the diameter of the shank',
    'Find the thread pitch and the nearest standard',
    'Compare this scan with the CAD model at ±0.05 mm',
  ],
  export: [
    'Export an STL of the latest mesh',
    'Summarise every measurement of this part',
  ],
};

/**
 * The whole conversation with its message box. `panel` is the side panel's Assistant tab (with its own top bar);
 * `float` is the body of the floating chat window, whose header carries the same tools, in compact type.
 */
export function AssistantPanel({ variant = 'panel' }: { variant?: 'panel' | 'float' }) {
  const { messages, streaming, status, attachments, draft } = useAssistant();
  const step = useStore(s => s.step);
  const activeId = useStore(s => s.activeId);
  const selection = useStore(s => s.selection);
  const byId = useStore(s => s.byId);
  const [input, setInput] = useState('');
  const [focused, setFocused] = useState(false);
  const [dropping, setDropping] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const compact = variant === 'float';

  useEffect(() => {
    refreshAssistantStatus();
  }, []);

  useEffect(() => {
    if (!draft) return;
    setInput(draft);
    useAssistant.setState({ draft: '' });
    setTimeout(() => textarea.current?.focus(), 0);
  }, [draft]);

  // opening the chat shows the latest messages; after that it follows new ones while you are near the end
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  // a message you just sent always comes into view; after that the chat follows the answer while you are near the end
  const count = useRef(messages.length);
  useEffect(() => {
    const el = scroller.current;
    const sent = messages.length > count.current;
    count.current = messages.length;
    if (el && (sent || el.scrollHeight - el.scrollTop - el.clientHeight < 200)) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const t = textarea.current;
    if (!t) return;
    t.style.height = 'auto';
    t.style.height = `${Math.min(t.scrollHeight, compact ? 140 : 200)}px`;
  }, [input, compact]);

  const send = (text = input) => {
    if ((!text.trim() && !attachments.length) || streaming) return;
    setInput('');
    sendToAssistant(text);
  };

  const offline = status && (!status.reachable || !status.model);
  const contextChips = useMemo(() => {
    const chips: string[] = [];
    const a = activeId ? byId.get(activeId) : undefined;
    if (a) chips.push(a.name);
    if (selection) chips.push(`${selection.count.toLocaleString()} points ${selection.region ? 'highlighted' : 'selected'}`);
    return chips;
  }, [activeId, selection, byId]);
  const chips = compact && focused && !input && !streaming && messages.length > 0;

  return (
    <div
      className={`assistant ${compact ? 'is-compact' : ''} ${dropping ? 'is-dropping' : ''}`}
      data-own-drop
      onDragOver={e => {
        if (e.dataTransfer.types.includes('Files')) {
          e.preventDefault();
          setDropping(true);
        }
      }}
      onDragLeave={() => setDropping(false)}
      onDrop={e => {
        if (!e.dataTransfer.files.length) return;
        e.preventDefault();
        e.stopPropagation();
        setDropping(false);
        attachFiles([...e.dataTransfer.files]);
      }}
    >
      {!compact && (
        <div className="assistant-bar">
          <ModelPill />
          <span className="spacer" />
          <HistoryMenu />
          <IconButton size="sm" label="New conversation" onClick={newConversation} disabled={streaming}>
            <Plus size={16} />
          </IconButton>
          <IconButton size="sm" tip="bottom" label="Pop out the chat" data-guide="chat.popout" onClick={popOutChat}>
            <PictureInPicture2 size={16} />
          </IconButton>
          <IconButton size="sm" label="Assistant settings" onClick={() => useStore.getState().set({ settingsOpen: 'assistant' })}>
            <Settings2 size={16} />
          </IconButton>
        </div>
      )}

      <div className="chat" ref={scroller}>
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-hero">
              <AssistantMark size={compact ? 36 : 48} />
              <h3 className="display">Tell CloudClean what you need</h3>
              <p>It can run the scanner and turntable, clean, align, mesh, edit any part of a model, measure in any direction and export. Every change makes a new model, so nothing is lost.</p>
            </div>
            {offline && <OfflineCard onRetry={refreshAssistantStatus} />}
            <div className="suggestions">
              {SUGGESTIONS[step].map(s => (
                <button key={s} type="button" className="suggestion" onClick={() => send(s)} disabled={!!offline}>
                  <AssistantGlyph size={16} />
                  <span>{s}</span>
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (m.role === 'user' ? <UserBubble key={i} msg={m} /> : <AssistantMessage key={i} msg={m} live={streaming && i === messages.length - 1} />))}
      </div>

      <div className="composer">
        {chips && (
          <div className="ask-chips composer-chips" aria-label={`Suggestions for ${step}`}>
            {SUGGESTIONS[step].map(s => (
              <button key={s} type="button" className="ask-chip" onMouseDown={e => e.preventDefault()} onClick={() => send(s)}>
                {s}
              </button>
            ))}
          </div>
        )}
        {contextChips.length > 0 && (
          <div className="context-chips" aria-label="Sent with your message">
            {contextChips.map(c => <span key={c} className="context-chip truncate">{c}</span>)}
          </div>
        )}
        <AttachmentStrip small={compact} />
        <div className="composer-box">
          <textarea
            ref={textarea}
            rows={1}
            value={input}
            aria-label="Message to the assistant"
            placeholder={offline ? (compact ? 'Connect an LLM server in Settings' : 'Connect an LLM server in Settings to start') : compact ? 'Ask while you work…' : 'Ask to scan, clean, measure, rotate the turntable…'}
            onChange={e => setInput(e.target.value)}
            onPaste={pasteImages}
            onFocus={() => setFocused(true)}
            onBlur={() => setTimeout(() => setFocused(false), 150)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <div className="composer-actions">
            <AttachButton compact={compact} />
            <span className="spacer" />
            {streaming ? (
              <Button size="sm" icon={<Square size={12} />} onClick={stopAssistant}>Stop</Button>
            ) : (
              <button type="button" className="ask-send" disabled={!input.trim() && !attachments.length} onClick={() => send()} aria-label="Send">
                <ArrowUp size={17} />
              </button>
            )}
          </div>
        </div>
        {!compact && <div className="composer-foot">Enter to send · Shift+Enter for a new line · add photos of your part with Photos, or drop / paste them</div>}
      </div>
    </div>
  );
}

/** The LLM's name and whether it answers (the dot), at the top of the chat. */
export function ModelPill({ dotOnly = false }: { dotOnly?: boolean }) {
  const status = useAssistant(s => s.status);
  const tone = !status ? '' : status.reachable ? (status.error ? 'warn' : 'ok') : 'danger';
  const text = status ? (status.reachable ? status.model || 'no model' : 'offline') : 'checking…';
  if (dotOnly) {
    const say = !status ? 'Checking the language model' : status.reachable ? `Language model: ${status.model || 'none loaded'}` : 'The language model is offline';
    return <span className={`dot model-dot ${tone}`} role="img" aria-label={say} data-tip={say} data-tip-side="bottom" />;
  }
  return (
    <span className={`model-pill ${status?.reachable ? (status.error ? 'is-warn' : 'is-ok') : 'is-off'}`} title={status?.error ?? status?.base_url ?? ''}>
      <span className={`dot ${tone}`} />
      <span className="truncate">{text}</span>
    </span>
  );
}

const UserBubble = memo(function UserBubble({ msg }: { msg: Msg }) {
  return (
    <div className="msg msg-user">
      {msg.text && <div className="bubble">{msg.text}</div>}
      {!!msg.images?.length && (
        <div className="bubble-images">
          {msg.images.map((im, i) => (im.url ? <img key={i} src={im.url} alt={im.caption} title={im.caption} /> : <span key={i} className="image-chip"><ImageIcon size={12} aria-hidden />{im.caption}</span>))}
        </div>
      )}
    </div>
  );
});

const AssistantMessage = memo(function AssistantMessage({ msg, live }: { msg: Msg; live: boolean }) {
  const [showReasoning, setShowReasoning] = useState(false);
  const empty = !msg.text && !msg.tools?.length && !msg.error;
  const html = useMemo(() => (msg.text ? renderMarkdown(msg.text) : ''), [msg.text]);
  return (
    <div className="msg msg-assistant">
      {msg.reasoning && (
        <button type="button" className="reasoning-toggle" onClick={() => setShowReasoning(!showReasoning)} aria-expanded={showReasoning}>
          {showReasoning ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          <Brain size={13} aria-hidden /> {live && !msg.text && !msg.tools?.length ? 'Thinking…' : 'How it reasoned'}
        </button>
      )}
      {showReasoning && msg.reasoning && <div className="reasoning mono">{msg.reasoning}</div>}
      {msg.tools?.map(t => <ToolCard key={t.call_id} tool={t} />)}
      {msg.confirms?.map(c => <ConfirmCard key={c.call_id} req={c} />)}
      {msg.gallery && msg.gallery.images.length > 0 && <Gallery gallery={msg.gallery} />}
      {html && <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />}
      {live && empty && !msg.reasoning && <div className="typing" aria-label="The assistant is thinking"><span /><span /><span /></div>}
      {msg.error && <div className="msg-error"><AlertTriangle size={14} aria-hidden /> {msg.error}</div>}
      {msg.stopped && <div className="msg-note">Stopped</div>}
    </div>
  );
});

/** Pictures a tool showed while answering (e.g. the merge options it compared). */
export function Gallery({ gallery, small = false }: { gallery: NonNullable<Msg['gallery']>; small?: boolean }) {
  return (
    <figure className={`msg-gallery ${small ? 'is-small' : ''}`}>
      {gallery.title && <figcaption>{gallery.title}</figcaption>}
      <div className="msg-gallery-grid">
        {gallery.images.map(im => (
          <a key={im.url} href={im.url} target="_blank" rel="noreferrer" title={`${im.caption} — open full size`}>
            <img src={im.url} alt={im.caption} loading="lazy" />
            {!small && <span>{im.caption}</span>}
          </a>
        ))}
      </div>
    </figure>
  );
}

function ToolCard({ tool }: { tool: ToolRecord }) {
  const [open, setOpen] = useState(false);
  const byId = useStore(s => s.byId);
  const p = tool.progress;
  const args = Object.entries(tool.arguments ?? {}).filter(([, v]) => v != null && v !== '').slice(0, 3);
  return (
    <div className={`tool-call ${tool.running ? 'is-running' : tool.ok ? 'is-ok' : 'is-failed'}`}>
      <button type="button" className="tool-call-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="tool-call-icon">{tool.running ? <Loader2 size={14} className="spin" /> : tool.ok ? <Check size={14} /> : <X size={14} />}</span>
        <span className="tool-call-name">{humanize(tool.name)}</span>
        <span className="tool-call-args mono">{args.map(([k, v]) => `${k}=${typeof v === 'object' ? '…' : String(v)}`).join(' ')}</span>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
      </button>
      {tool.running && p && (
        <div className="tool-call-progress">
          <Progress value={p.progress ?? null} indeterminate={p.progress == null} />
          <span className="mono">{p.label || p.last_log || p.status}</span>
        </div>
      )}
      {tool.summary && <div className="tool-call-summary">{tool.summary}</div>}
      {!!tool.asset_ids?.length && (
        <div className="chip-row">
          {tool.asset_ids.filter(id => byId.has(id)).map(id => (
            <button key={id} type="button" className="chip" onClick={() => { useStore.setState({ visible: [id], activeId: id }); setTimeout(() => getViewer()?.fit([id]), 80); }}>
              {byId.get(id)!.name}
            </button>
          ))}
        </div>
      )}
      {open && <pre className="tool-call-json mono">{JSON.stringify(tool.arguments, null, 2)}</pre>}
    </div>
  );
}

/** "Delete these?" from the assistant. The answer is kept per request, so the chat, the floating chat and the reply box agree. */
export function ConfirmCard({ req }: { req: ConfirmRequest }) {
  const decision = useAssistant(s => s.decisions[req.call_id] as 'deleted' | 'kept' | undefined);
  const [busy, setBusy] = useState(false);
  const byId = useStore(s => s.byId);
  const decide = (d: 'deleted' | 'kept') => useAssistant.setState(s => ({ decisions: { ...s.decisions, [req.call_id]: d } }));
  const confirm = async () => {
    setBusy(true);
    for (const id of req.asset_ids) {
      await api.del(`/api/assets/${id}`).catch(() => undefined);
      getViewer()?.forget(id);
    }
    await useStore.getState().refreshAssets();
    decide('deleted');
    setBusy(false);
  };
  return (
    <div className="confirm-card">
      <div className="confirm-title"><AlertTriangle size={15} aria-hidden /> {req.message}</div>
      <div className="chip-row">{req.asset_ids.map(id => <Badge key={id}>{byId.get(id)?.name ?? id}</Badge>)}</div>
      {!decision ? (
        <div className="row">
          <Button size="sm" variant="danger" icon={<Trash2 size={13} />} onClick={confirm} loading={busy}>Delete</Button>
          <Button size="sm" variant="ghost" onClick={() => decide('kept')} disabled={busy}>Keep</Button>
        </div>
      ) : (
        <div className="msg-note">{decision === 'deleted' ? 'Deleted.' : 'Kept.'}</div>
      )}
    </div>
  );
}

function OfflineCard({ onRetry }: { onRetry: () => void }) {
  const status = useAssistant(s => s.status)!;
  return (
    <div className="offline-card">
      <div className="offline-title"><AlertTriangle size={15} aria-hidden /> {status.reachable ? 'No model loaded' : 'The LLM server is not answering'}</div>
      <p className="hint-text">{status.error ?? `Nothing answered at ${status.base_url}`}</p>
      <p className="hint-text">Start an OpenAI-compatible server with tool calling on the DGX Spark (vLLM, SGLang, Ollama) and set its address in Settings → Assistant.</p>
      <div className="row">
        <Button size="sm" onClick={onRetry}>Check again</Button>
        <Button size="sm" variant="ghost" onClick={() => useStore.getState().set({ settingsOpen: 'assistant' })}>Open settings</Button>
      </div>
    </div>
  );
}

export function HistoryMenu() {
  const current = useAssistant(s => s.conversationId);
  const [list, setList] = useState<{ id: string; title: string; updated: string; turns: number }[]>([]);
  return (
    <Popover
      align="end"
      trigger={({ toggle, open }) => (
        <IconButton size="sm" label="Past conversations" active={open} onClick={() => { if (!open) api.get<typeof list>('/api/assistant/conversations').then(setList).catch(() => setList([])); toggle(); }}>
          <History size={16} />
        </IconButton>
      )}
    >
      {({ close }) => (
        <div className="menu history-menu">
          {list.length === 0 && <div className="menu-label">No saved conversations</div>}
          {list.map(c => (
            <div key={c.id} className={`menu-item ${c.id === current ? 'is-active' : ''}`}>
              <button type="button" className="grow" style={{ textAlign: 'left' }} onClick={() => { loadConversation(c.id); close(); }}>
                <div className="menu-item-title truncate">{c.title}</div>
                <div className="menu-item-sub">{c.turns} message{c.turns === 1 ? '' : 's'} · {new Date(c.updated).toLocaleString()}</div>
              </button>
              <IconButton size="sm" label="Delete conversation" onClick={async () => { await api.del(`/api/assistant/conversations/${c.id}`).catch(() => undefined); setList(l => l.filter(x => x.id !== c.id)); }}>
                <Trash2 size={14} />
              </IconButton>
            </div>
          ))}
        </div>
      )}
    </Popover>
  );
}

/**
 * The Assistant tab while the chat is popped out: where the chat went, and a way to dock it back here. Drawn as an
 * empty slot on a drawing sheet, with a leader to where the chat is now.
 */
export function PoppedOutNotice() {
  const minimized = useChatWindow(s => s.minimized);
  const streaming = useAssistant(s => s.streaming);
  return (
    <div className="chat-away">
      <svg className="chat-away-art" viewBox="0 0 220 120" aria-hidden>
        <rect className="slot" x="8" y="8" width="96" height="104" rx="10" />
        <path className="slot-lines" d="M22 30h56M22 44h68M22 58h44" />
        <rect className="window" x="128" y="22" width="84" height="64" rx="9" />
        <path className="window-bar" d="M128 38h84" />
        <path className="leader" d="M104 60 C 116 60, 116 54, 128 54" />
        <circle className="node" cx="104" cy="60" r="3" />
        <path className="arrow" d="M122 50l6 4-6 4" />
      </svg>
      <h3 className="display">The chat is popped out</h3>
      <p className="hint-text">
        It floats over the app{minimized ? ', minimized to a bubble,' : ''} so you can keep talking while you use the step's tools here.
        {streaming ? ' It is working on your request right now.' : ''}
      </p>
      <div className="row">
        <Button variant="primary" icon={<PanelDockIcon />} onClick={dockChat} data-guide="chat.dock">Dock it here</Button>
        <Button variant="ghost" icon={<AssistantGlyph size={15} />} onClick={() => expandChat()}>{minimized ? 'Open the chat' : 'Go to the chat'}</Button>
      </div>
    </div>
  );
}

/** "Dock" drawn like the app's panel icons: a window sliding into the right-hand panel. */
export function PanelDockIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M15 3v18" />
      <path d="M7 12h5" />
      <path d="m10 9 3 3-3 3" />
    </svg>
  );
}
