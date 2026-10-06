import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { AlertTriangle, Check, ChevronDown, ChevronUp, Image as ImageIcon, Loader2, PanelRight, PictureInPicture2, X } from 'lucide-react';
import { humanize } from '../../lib/format';
import { useStore } from '../../store';
import { IconButton } from '../../ui/primitives';
import { runningStep, type Msg } from './assistantStore';
import { hideReply, popOutChat, setReplyTall, useChatWindow } from './chatWindow';
import { ConfirmCard, Gallery } from './AssistantPanel';
import { renderMarkdown } from './markdown';

const TRAIL = 4;

/** Open the full conversation in the side panel's Assistant tab. */
export function openChatInPanel() {
  const st = useStore.getState();
  st.set({ rightTab: 'assistant', screen: 'workspace' });
  st.setLayout({ rightOpen: true });
}

/**
 * The latest exchange in short, on top of the Ask bar: your question in one line and the answer as it arrives
 * (markdown, clipped with a fade; scroll it or show more). While the assistant works, the rule under the question
 * is its progress line and the step it is on is named. Open the whole chat in the side panel, pop it out, or hide
 * the box until the next reply.
 */
export function ReplyBox({ question, answer, index, live }: { question: Msg | null; answer: Msg; index: number; live: boolean }) {
  const tall = useChatWindow(s => s.replyTall);
  const body = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ above: false, below: false });
  const html = useMemo(() => (answer.text ? renderMarkdown(answer.text) : ''), [answer.text]);
  const step = runningStep(answer);
  const tools = answer.tools ?? [];
  const shown = tools.slice(-TRAIL);
  const pendingConfirms = (answer.confirms ?? []).length;

  const measure = useCallback(() => {
    const el = body.current;
    if (!el) return;
    const above = el.scrollTop > 2;
    const below = el.scrollHeight - el.scrollTop - el.clientHeight > 2;
    setEdges(e => (e.above === above && e.below === below ? e : { above, below }));
  }, []);
  useLayoutEffect(measure, [measure, html, tall, tools.length, pendingConfirms, answer.error, answer.gallery]);
  // the box's height follows the 3D view's (a third of it at most): measure again when that changes
  useEffect(() => {
    const el = body.current;
    if (!el) return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [measure]);

  const status = step ? `Working: ${step.label}` : live ? (answer.text ? 'Answering…' : 'Thinking…') : null;
  // what a screen reader hears: the step and when it is done, not every percent
  const running = tools.find(t => t.running);
  const spoken = running ? `Working: ${humanize(running.name)}` : live ? 'Thinking' : answer.error ? 'The assistant ran into a problem' : 'Reply ready';
  const question1 = question?.text?.replace(/\s+/g, ' ').trim() || (question?.images?.length ? 'Photos of the part' : '');
  const canGrow = edges.below || edges.above || tall;

  return (
    <div className={`ask-reply ${tall ? 'is-tall' : ''} ${live ? 'is-live' : ''}`} role="region" aria-label="Latest reply" data-guide="chat.reply">
      <span className="visually-hidden" role="status">{spoken}</span>
      <div className="ask-reply-head">
        <span className="ask-reply-tag">Latest reply</span>
        {question1 && (
          <span className="ask-reply-q" title={question?.text}>
            <span className="ask-reply-you">You:</span> {question1}
          </span>
        )}
        {!!question?.images?.length && (
          <span className="ask-reply-thumbs" role="img" aria-label={`${question.images.length} photo${question.images.length === 1 ? '' : 's'} sent`}>
            {question.images.slice(0, 3).map((im, i) => (im.url ? <img key={i} src={im.url} alt="" title={im.caption} /> : <span key={i} className="ask-reply-thumb-chip" title={im.caption}><ImageIcon size={12} aria-hidden /></span>))}
            {question.images.length > 3 && <span className="ask-reply-more mono">+{question.images.length - 3}</span>}
          </span>
        )}
        <span className="spacer" />
        <span className="ask-reply-actions">
          <IconButton size="sm" tip="top" label="Open the full chat in the side panel" onClick={openChatInPanel}>
            <PanelRight size={15} />
          </IconButton>
          <IconButton size="sm" tip="top" label="Pop out the chat" data-guide="chat.popout" onClick={popOutChat}>
            <PictureInPicture2 size={15} />
          </IconButton>
          <IconButton size="sm" tip="top" label="Hide this reply (the next one shows again)" onClick={() => hideReply(index)}>
            <X size={15} />
          </IconButton>
        </span>
      </div>

      {/* the rule under the question doubles as the progress line while the assistant works */}
      <div className={`ask-reply-rule ${live ? (step?.progress != null ? 'is-progress' : 'is-busy') : ''}`} style={step?.progress != null ? ({ '--p': step.progress } as CSSProperties) : undefined} aria-hidden />

      {(status || tools.length > 0) && (
        <div className="ask-reply-steps">
          {status && (
            <span className="ask-reply-status">
              <Loader2 size={13} className="spin" aria-hidden />
              <span className="truncate">{status}</span>
            </span>
          )}
          {!step && tools.length > 0 && (
            <span className="ask-reply-trail" aria-label="What it did">
              {tools.length > TRAIL && <span className="ask-reply-more mono">+{tools.length - TRAIL}</span>}
              {shown.map(t => (
                <span key={t.call_id} className={`ask-reply-tool ${t.running ? 'is-running' : t.ok ? 'is-ok' : 'is-failed'}`} title={t.summary ?? humanize(t.name)}>
                  {t.running ? <Loader2 size={12} className="spin" aria-hidden /> : t.ok ? <Check size={12} aria-hidden /> : <X size={12} aria-hidden />}
                  {humanize(t.name)}
                </span>
              ))}
            </span>
          )}
        </div>
      )}

      <div className="ask-reply-frame" hidden={!html && !answer.confirms?.length && !answer.gallery?.images.length && !answer.error && !answer.stopped}>
        <div
          ref={body}
          className={`ask-reply-body ${edges.above ? 'has-above' : ''} ${edges.below ? 'has-below' : ''}`}
          onScroll={measure}
          tabIndex={edges.above || edges.below ? 0 : -1}
          aria-label="Reply text"
        >
          {html && <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />}
          {answer.confirms?.map(c => <ConfirmCard key={c.call_id} req={c} />)}
          {answer.gallery && answer.gallery.images.length > 0 && <Gallery gallery={answer.gallery} small />}
          {answer.error && <div className="msg-error"><AlertTriangle size={14} aria-hidden /> {answer.error}</div>}
          {answer.stopped && <div className="msg-note">Stopped</div>}
        </div>
        {canGrow && (
          <button type="button" className="ask-reply-grow" onClick={() => setReplyTall(!tall)} aria-expanded={tall}>
            {tall ? <>Show less <ChevronDown size={13} aria-hidden /></> : <>Show more <ChevronUp size={13} aria-hidden /></>}
          </button>
        )}
      </div>
    </div>
  );
}
