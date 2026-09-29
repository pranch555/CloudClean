import { lazy, Suspense } from 'react';
import { PanelRightClose } from 'lucide-react';
import { stepInfo } from '../lib/journey';
import { useStore } from '../store';
import { useAssistant } from '../features/assistant/assistantStore';
import { AssistantPanel } from '../features/assistant/AssistantPanel';
import { AssistantGlyph } from '../features/assistant/AssistantMark';
import { STEP_GLYPH } from '../ui/icons';
import { IconButton } from '../ui/primitives';
import { CleanStep } from '../steps/CleanStep';
import { AlignStep } from '../steps/AlignStep';
import { MeshStep } from '../steps/MeshStep';
import { ExportStep } from '../steps/ExportStep';

const CaptureStep = lazy(() => import('../steps/capture/CaptureStep').then(m => ({ default: m.CaptureStep })));
const MeasureStep = lazy(() => import('../steps/measure/MeasureStep').then(m => ({ default: m.MeasureStep })));

export function SidePanel() {
  const step = useStore(s => s.step);
  const tab = useStore(s => s.rightTab);
  const params = useStore(s => s.params);
  const streaming = useAssistant(s => s.streaming);
  const set = useStore(s => s.set);
  const info = stepInfo(step);
  const Glyph = STEP_GLYPH[step];

  return (
    <aside className="island side" aria-label={tab === 'assistant' ? 'Assistant' : `Step: ${info.label}`}>
      <div className="side-head">
        <div className="segmented side-tabs" role="tablist">
          <button type="button" role="tab" aria-selected={tab === 'step'} className={tab === 'step' ? 'is-on' : ''} onClick={() => set({ rightTab: 'step' })}>
            <Glyph size={16} /> {info.label}
          </button>
          <button type="button" role="tab" aria-selected={tab === 'assistant'} className={tab === 'assistant' ? 'is-on' : ''} onClick={() => set({ rightTab: 'assistant' })}>
            <span className={`assistant-tab-glyph ${streaming ? 'is-working' : ''}`}>
              <AssistantGlyph size={16} />
            </span>
            Assistant
            {streaming && <span className="visually-hidden">, working</span>}
          </button>
        </div>
        <IconButton size="sm" label="Hide this panel (Ctrl I)" onClick={() => useStore.getState().setLayout({ rightOpen: false })}>
          <PanelRightClose size={17} />
        </IconButton>
      </div>
      {tab === 'assistant' ? (
        <AssistantPanel />
      ) : !params ? (
        <div className="side-scroll" />
      ) : (
        <Suspense fallback={<div className="side-scroll" />}>
          {step === 'capture' ? <CaptureStep /> : step === 'clean' ? <CleanStep /> : step === 'align' ? <AlignStep /> : step === 'mesh' ? <MeshStep /> : step === 'measure' ? <MeasureStep /> : <ExportStep />}
        </Suspense>
      )}
    </aside>
  );
}
