import type { Asset } from '../../lib/types';
import { useTarget } from '../../store';
import { Segmented } from '../../ui/primitives';
import { Block, NextStepButton, StepFrame, TargetCard } from '../StepFrame';
import { AccuracyTab } from './AccuracyTab';
import { MergeCaveat } from './MergeCaveat';
import { AccuracyGlyph, CadGlyph, DimensionsGlyph, ThreadGlyph, type Glyph } from './glyphs';
import { GoldenCheck, GoldenFooter } from './GoldenCheck';
import { PartSizeCard } from './PartSize';
import { ResultsList } from './Results';
import { setTab, useMeasureUi, type MeasureTab } from './state';
import { ThreadFooter, ThreadTab } from './ThreadTab';
import { DimensionsFooter, ToolGrid } from './Tools';

const TABS: { value: MeasureTab; label: string; title: string; glyph: Glyph }[] = [
  { value: 'dimensions', label: 'Dimensions', title: 'The part’s size and tools to measure anything on it', glyph: DimensionsGlyph },
  { value: 'thread', label: 'Thread', title: 'Pitch, diameters and standard size of a screw thread', glyph: ThreadGlyph },
  { value: 'cad', label: 'Golden model', title: 'Check the scan against its golden model (CAD or a trusted mesh): what is off, what to scan again, every size', glyph: CadGlyph },
  { value: 'accuracy', label: 'Accuracy', title: 'When a size looks off: is it the scanner or the software?', glyph: AccuracyGlyph },
];

/** Step 5: get any dimension of the part in any direction, check threads, check against the golden model, and trust the numbers. */
export function MeasureStep() {
  const target = useTarget();
  const tab = useMeasureUi(s => s.tab);
  return (
    <StepFrame
      step="measure"
      footer={
        <>
          {tab === 'dimensions' && <DimensionsFooter target={target} />}
          {tab === 'thread' && <ThreadFooter target={target} />}
          {tab === 'cad' && <GoldenFooter />}
          <NextStepButton from="measure" />
        </>
      }
    >
      <TargetCard asset={target} label="Measuring" empty="Click the part you want to measure in the list on the left." />
      <MergeCaveat asset={target} />
      <nav className="meas-nav" aria-label="What to measure">
        <Segmented
          value={tab}
          onChange={setTab}
          ariaLabel="What to measure"
          options={TABS.map(t => ({
            value: t.value,
            title: t.title,
            guide: `measure.${t.value === 'dimensions' ? 'dimensions' : t.value}`,
            label: (
              <span className="meas-tab">
                <t.glyph size={19} />
                <span>{t.label}</span>
              </span>
            ),
          }))}
        />
      </nav>
      {tab === 'dimensions' && <DimensionsTab target={target} />}
      {tab === 'thread' && <ThreadTab target={target} />}
      {tab === 'cad' && <GoldenCheck />}
      {tab === 'accuracy' && <AccuracyTab target={target} />}
    </StepFrame>
  );
}

function DimensionsTab({ target }: { target: Asset | undefined }) {
  return (
    <>
      {target && <PartSizeCard asset={target} />}
      <Block title="Measure">
        <ToolGrid target={target} />
      </Block>
      <ResultsList />
    </>
  );
}
