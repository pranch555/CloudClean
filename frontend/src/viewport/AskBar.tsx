import { useState } from 'react';
import { ArrowUp, Loader2, Square } from 'lucide-react';
import { useStore, panelShown } from '../store';
import { lastActivity, sendToAssistant, stopAssistant, useAssistant } from '../features/assistant/assistantStore';
import { SUGGESTIONS } from '../features/assistant/AssistantPanel';
import { AttachButton, AttachmentStrip, dropImages, pasteImages } from '../features/assistant/AttachControls';
import { AssistantGlyph } from '../features/assistant/AssistantMark';

/** The assistant, one line away: type, press Enter, watch it work. The full thread lives in the Assistant tab. */
export function AskBar() {
  const [text, setText] = useState('');
  const [focused, setFocused] = useState(false);
  const streaming = useAssistant(s => s.streaming);
  const attached = useAssistant(s => s.attachments.length);
  useAssistant(s => s.messages); // re-render as the reply streams
  const step = useStore(s => s.step);
  const rightTab = useStore(s => s.rightTab);
  const rightOpen = useStore(s => panelShown(s, 'right'));
  const { running, reply } = lastActivity();
  const threadVisible = rightTab === 'assistant' && rightOpen;

  const openThread = () => {
    useStore.getState().set({ rightTab: 'assistant' });
    useStore.getState().setLayout({ rightOpen: true });
  };

  const send = (t = text) => {
    if ((!t.trim() && !attached) || streaming) return;
    setText('');
    sendToAssistant(t);
  };

  const snippet = reply.replace(/[#*_`>|]/g, '').replace(/\s+/g, ' ').trim();

  return (
    <div className="hud ask-bar" role="search" aria-label="Ask CloudClean" data-own-drop {...dropImages}>
      {streaming && (
        <div className="ask-status" aria-live="polite">
          <Loader2 size={14} className="spin" aria-hidden />
          <span className="truncate">{running ? `Working: ${running}` : snippet ? 'Answering…' : 'Thinking…'}</span>
          <span className="spacer" />
          {!threadVisible && <button type="button" className="link" onClick={openThread}>Show</button>}
        </div>
      )}
      {!streaming && snippet && !threadVisible && (
        <div className="ask-reply" onClick={openThread} title="Open the conversation">
          {snippet.slice(0, 280)}
        </div>
      )}
      {!threadVisible && <AttachmentStrip small />}
      <div className="ask-row">
        <AssistantGlyph size={18} />
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
