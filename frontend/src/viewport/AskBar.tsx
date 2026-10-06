import { useState } from 'react';
import { ArrowUp, Loader2, Square } from 'lucide-react';
import { useStore, panelShown } from '../store';
import { lastExchange, runningStep, sendToAssistant, stopAssistant, useAssistant } from '../features/assistant/assistantStore';
import { SUGGESTIONS } from '../features/assistant/AssistantPanel';
import { AttachButton, AttachmentStrip, dropImages, pasteImages } from '../features/assistant/AttachControls';
import { AssistantGlyph } from '../features/assistant/AssistantMark';
import { showReply, useChatWindow } from '../features/assistant/chatWindow';
import { ReplyBox } from '../features/assistant/ReplyBox';

/**
 * The assistant, one line away: type, press Enter, watch it work. Above the line, the latest reply in short (when
 * the full chat is not on screen). The full thread is the Assistant tab, or the floating chat when popped out;
 * while the chat floats, this bar steps aside so there is one place to type.
 */
export function AskBar() {
  const [text, setText] = useState('');
  const [focused, setFocused] = useState(false);
  const streaming = useAssistant(s => s.streaming);
  const attached = useAssistant(s => s.attachments.length);
  const messages = useAssistant(s => s.messages);
  const place = useChatWindow(s => s.place);
  const replyHidden = useChatWindow(s => s.replyHidden);
  const step = useStore(s => s.step);
  const rightTab = useStore(s => s.rightTab);
  const rightOpen = useStore(s => panelShown(s, 'right'));
  const threadVisible = rightTab === 'assistant' && rightOpen;

  // the element stays (other overlays measure it to keep clear of it); it is just not shown while the chat floats
  if (place === 'floating') return <div className="hud ask-bar is-away" aria-hidden />;

  const ex = lastExchange(messages);
  const live = streaming && !!ex && ex.index === messages.length - 1;
  const replyShown = !!ex && !threadVisible && replyHidden !== ex.index;
  const hiddenReply = !!ex && !threadVisible && replyHidden === ex.index;
  const running = runningStep(ex?.answer);

  const send = (t = text) => {
    if ((!t.trim() && !attached) || streaming) return;
    setText('');
    sendToAssistant(t);
  };

  return (
    <div className={`hud ask-bar ${replyShown ? 'has-reply' : ''}`} data-own-drop {...dropImages}>
      {replyShown && <ReplyBox question={ex.question} answer={ex.answer} index={ex.index} live={live} />}
      {streaming && threadVisible && (
        <div className="ask-status" aria-live="polite">
          <Loader2 size={14} className="spin" aria-hidden />
          <span className="truncate">{running ? `Working: ${running.label}` : ex?.answer.text ? 'Answering…' : 'Thinking…'}</span>
        </div>
      )}
      {!threadVisible && <AttachmentStrip small />}
      <div className="ask-row" role="search" aria-label="Ask CloudClean">
        {hiddenReply ? (
          <button type="button" className={`ask-glyph-btn ${live ? 'is-live' : ''}`} onClick={showReply} aria-label="Show the latest reply" data-tip="Show the latest reply">
            <AssistantGlyph size={18} />
            <span className="ask-glyph-dot" aria-hidden />
          </button>
        ) : (
          <AssistantGlyph size={18} />
        )}
        <input
          className="ask-input"
          value={text}
          placeholder="Ask CloudClean to measure, clean, rotate the turntable…"
          onChange={e => setText(e.target.value)}
          onPaste={pasteImages}
          onFocus={() => setFocused(true)}
          onBlur={() => setTimeout(() => setFocused(false), 150)}
          onKeyDown={e => {
            if (e.key === 'Enter') {
              e.preventDefault();
              send();
            }
            e.stopPropagation(); // typing here must not trigger viewport shortcuts
          }}
          aria-label="Ask CloudClean"
        />
        <AttachButton compact />
        {streaming ? (
          <button type="button" className="ask-send" onClick={stopAssistant} aria-label="Stop">
            <Square size={13} />
          </button>
        ) : (
          <button type="button" className="ask-send" disabled={!text.trim() && !attached} onClick={() => send()} aria-label="Send">
            <ArrowUp size={17} />
          </button>
        )}
      </div>
      {focused && !text && !streaming && (
        <div className="ask-chips">
          {SUGGESTIONS[step].map(s => (
            <button key={s} type="button" className="ask-chip" onMouseDown={e => e.preventDefault()} onClick={() => send(s)}>
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
