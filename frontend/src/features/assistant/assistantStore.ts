import { create } from 'zustand';
import { api, postSse, upload } from '../../lib/api';
import { applyUi, uiContext } from '../../lib/applyUi';
import { local, panelShown, useStore } from '../../store';
import type { Attachment } from './attachments';

export interface ToolRecord {
  call_id: string;
  name: string;
  arguments: Record<string, unknown>;
  ok?: boolean;
  summary?: string;
  asset_ids?: string[];
  progress?: { status?: string; progress?: number | null; label?: string | null; last_log?: string };
  running?: boolean;
}

export interface ConfirmRequest {
  call_id: string;
  action: string;
  asset_ids: string[];
  message: string;
}

export interface Msg {
  role: 'user' | 'assistant';
  text: string;
  reasoning?: string;
  tools?: ToolRecord[];
  confirms?: ConfirmRequest[];
  error?: string;
  // `url` exists only for images sent in this session; the server keeps captions, never image data
  images?: { url?: string; caption: string }[];
  /** pictures a tool showed while answering (e.g. the merge options it compared), shown under the answer */
  gallery?: { title?: string; images: { url: string; caption: string }[] };
  stopped?: boolean;
}

export interface AssistantStatus {
  configured: boolean;
  reachable: boolean;
  model: string;
  models: string[];
  error: string | null;
  base_url?: string;
}

interface AssistantState {
  messages: Msg[];
  conversationId: string | null;
  streaming: boolean;
  status: AssistantStatus | null;
  attachments: Attachment[];
  /** photos from this computer are saved in the project as reference photos of the part when sent */
  keepPhotos: boolean;
  draft: string;
}

export const useAssistant = create<AssistantState>(() => ({
  messages: [],
  conversationId: null,
  streaming: false,
  status: null,
  attachments: [],
  keepPhotos: local.get('keepPhotos', true),
  draft: '',
}));

/**
 * Ctrl K: put the cursor where you ask CloudClean something — the Home prompt, the open conversation, or the Ask
 * bar under the 3D view. (Finding features is the assistant's job; there is no separate search.)
 */
export function focusAsk() {
  const st = useStore.getState();
  const focus = (sel: string) => {
    const el = document.querySelector<HTMLElement>(sel);
    el?.focus();
    return !!el;
  };
  if (st.screen === 'home') {
    focus('.home-ask input');
    return;
  }
  const chatShown = st.rightTab === 'assistant' && panelShown(st, 'right');
  if (chatShown ? focus('.composer textarea') : focus('.ask-bar .ask-input')) return;
  st.set({ rightTab: 'assistant' });
  st.setLayout({ rightOpen: true });
  setTimeout(() => focus('.composer textarea'), 60);
}

export function setKeepPhotos(keep: boolean) {
  local.set('keepPhotos', keep);
  useAssistant.setState({ keepPhotos: keep });
}

/**
 * Save the new photos of a message in the project (full resolution, as photo assets) so the assistant can look at
 * them again later and they can colour the mesh; their captions then name the saved photo.
 */
async function keepPhotos(atts: Attachment[]): Promise<Attachment[]> {
  const fresh = atts.filter(a => a.file && !a.assetId);
  if (!fresh.length || !useAssistant.getState().keepPhotos) return atts;
  const pid = useStore.getState().projectId;
  try {
    const res = await upload('/api/upload', fresh.map(a => a.file!), undefined, pid && pid !== '__all__' ? { project_id: pid } : undefined);
    const saved: { id: string; name: string }[] = res?.images ?? [];
    useStore.getState().refreshAssets().catch(() => undefined);
    return atts.map(a => {
      const m = saved[fresh.indexOf(a)];
      return m ? { ...a, assetId: m.id, caption: `reference photo '${m.name}' (${m.id}), kept in the project` } : a;
    });
  } catch (err) {
    useStore.getState().toast({ kind: 'warn', title: 'The photos were sent but not kept in the project', body: (err as Error).message });
    return atts;
  }
}

let abort: AbortController | null = null;

const updateLast = (fn: (m: Msg) => Msg) =>
  useAssistant.setState(s => ({ messages: s.messages.map((m, i) => (i === s.messages.length - 1 ? fn({ ...m }) : m)) }));

export async function refreshAssistantStatus() {
  try {
    useAssistant.setState({ status: await api.get<AssistantStatus>('/api/assistant/status') });
  } catch (err) {
    useAssistant.setState({ status: { configured: false, reachable: false, model: '', models: [], error: (err as Error).message } });
  }
}

/** Send a message (with the current attachments and the UI context) and stream the answer. */
export async function sendToAssistant(text: string) {
  const a = useAssistant.getState();
  if ((!text.trim() && !a.attachments.length) || a.streaming) return;
  useAssistant.setState(s => ({
    messages: [...s.messages, { role: 'user', text, images: s.attachments.map(x => ({ url: x.url, caption: x.caption })) }, { role: 'assistant', text: '', tools: [], confirms: [] }],
    attachments: [],
    draft: '',
    streaming: true,
  }));
  const ctrl = new AbortController();
  abort = ctrl;
  try {
    const sent = await keepPhotos(a.attachments);
    const images = sent.map(x => x.url);
    const imageNotes = sent.map(x => x.caption);
    for await (const ev of postSse('/api/assistant/chat', { conversation_id: a.conversationId, message: text, context: uiContext(), images, image_notes: imageNotes }, ctrl.signal)) {
      const d = ev.data;
      switch (ev.event) {
        case 'conversation':
          useAssistant.setState({ conversationId: d.id });
          break;
        case 'delta':
          updateLast(m => ({ ...m, text: m.text + d.text }));
          break;
        case 'reasoning':
          updateLast(m => ({ ...m, reasoning: (m.reasoning ?? '') + d.text }));
          break;
        case 'retract': // the model said the same as in its previous step: take the repeat back out
          updateLast(m => ({ ...m, text: m.text.slice(0, Math.max(0, m.text.length - Number(d.chars || 0))) }));
          break;
        case 'tool_start':
          updateLast(m => ({ ...m, tools: [...(m.tools ?? []), { call_id: d.call_id, name: d.name, arguments: d.arguments ?? {}, running: true }] }));
          break;
        case 'tool_progress':
          updateLast(m => ({ ...m, tools: (m.tools ?? []).map(t => (t.call_id === d.call_id ? { ...t, progress: d } : t)) }));
          break;
        case 'tool_end':
          updateLast(m => ({ ...m, tools: (m.tools ?? []).map(t => (t.call_id === d.call_id ? { ...t, running: false, ok: d.ok, summary: d.summary, asset_ids: d.asset_ids } : t)) }));
          if (d.asset_ids?.length) useStore.getState().refreshAssets().catch(() => undefined);
          break;
        case 'ui':
          if (d.action === 'images') updateLast(m => ({ ...m, gallery: { title: d.title, images: d.images ?? [] } }));
          else applyUi(d);
          break;
        case 'confirm':
          updateLast(m => ({ ...m, confirms: [...(m.confirms ?? []), d] }));
          break;
        case 'error':
          updateLast(m => ({ ...m, error: d.message }));
          break;
        case 'done':
          if (d?.stopped) updateLast(m => ({ ...m, stopped: true }));
          break;
      }
    }
  } catch (err) {
    if ((err as Error).name !== 'AbortError') updateLast(m => ({ ...m, error: (err as Error).message }));
  } finally {
    useAssistant.setState({ streaming: false });
    abort = null;
    useStore.getState().refreshAssets().catch(() => undefined);
    useStore.getState().refreshProjects().catch(() => undefined);
  }
}

export async function stopAssistant() {
  const id = useAssistant.getState().conversationId;
  if (id) await api.post('/api/assistant/stop', { conversation_id: id }).catch(() => undefined);
  setTimeout(() => abort?.abort(), 1500);
}

export function newConversation() {
  if (useAssistant.getState().streaming) return;
  useAssistant.setState({ messages: [], conversationId: null });
}

export async function loadConversation(id: string) {
  const conv = await api.get<{ id: string; transcript: Msg[] }>(`/api/assistant/conversations/${id}`);
  useAssistant.setState({
    conversationId: conv.id,
    messages: conv.transcript.map(t => ({ role: t.role, text: t.text ?? '', reasoning: t.reasoning, tools: t.tools, error: t.error, images: t.images, stopped: t.stopped })),
  });
}

/** Open the Assistant tab and put text in the composer (e.g. "With the points I selected: "). */
export function draftForAssistant(text: string) {
  useAssistant.setState({ draft: text });
  useStore.getState().set({ rightTab: 'assistant', screen: 'workspace' });
  useStore.getState().setLayout({ rightOpen: true });
}

/** The one-line state of the last reply, for the Ask bar. */
export function lastActivity(): { running: string | null; reply: string } {
  const msgs = useAssistant.getState().messages;
  const last = [...msgs].reverse().find(m => m.role === 'assistant');
  if (!last) return { running: null, reply: '' };
  const tool = last.tools?.find(t => t.running);
  const running = tool ? `${tool.name.replace(/_/g, ' ')}${tool.progress?.label ? ` · ${tool.progress.label}` : ''}${tool.progress?.progress != null ? ` · ${Math.round(tool.progress.progress * 100)}%` : ''}` : null;
  return { running, reply: last.error ? `⚠ ${last.error}` : last.text };
}
