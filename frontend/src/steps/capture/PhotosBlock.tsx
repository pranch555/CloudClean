import { useState } from 'react';
import { Camera, ChevronDown, ImagePlus, Printer, Sparkles } from 'lucide-react';
import { pickFiles } from '../../lib/importing';
import { isFinal, submitJob } from '../../lib/jobs';
import { projectParam } from '../../lib/projects';
import { useReconStatus } from '../../lib/recon';
import { local, useProjectAssets, useStore } from '../../store';
import { Button, Progress, Segmented } from '../../ui/primitives';
import { PhotoGrid, PhotoViewerHost } from '../../shell/models/photos';
import { Block } from '../StepFrame';
import { Callout } from './parts';

/** tiles shown before "+N more" (which opens the photo viewer on the rest) */
const SHOWN = 10;

type Paper = 'a4' | 'letter';

/** US Letter where it is the usual paper (US, Canada, Mexico, the Philippines...), A4 everywhere else. */
function defaultPaper(): Paper {
  const lang = typeof navigator !== 'undefined' ? navigator.language : '';
  return /-(US|CA|MX|PH|CL|CO|VE|GT|CR|PA|DO|SV|NI|HN|PR)$/i.test(lang) ? 'letter' : 'a4';
}

/** What the printed 100 mm bar measured: empty = not measured; 90-110 mm is plausible, anything else is a typo. */
function barValue(text: string): { mm: number | null; error: string | null } {
  if (!text.trim()) return { mm: null, error: null };
  const mm = Number(text.replace(',', '.'));
  if (!Number.isFinite(mm)) return { mm: null, error: 'Type the length in mm, e.g. 100.2' };
  if (mm < 90 || mm > 110) return { mm: null, error: 'The bar should measure about 100 mm. Print the sheet again at 100 % (actual size), then measure it again.' };
  return { mm, error: null };
}

/** "Make a 3D model from photos": the project's photos turned into a coloured point cloud (job photos_to_3d). */
export function PhotosBlock() {
  const photos = useProjectAssets().filter(a => a.kind === 'image');
  const running = useStore(s => s.jobs.find(j => j.kind === 'photos_to_3d' && !isFinal(j)));
  const status = useReconStatus();
  const [starting, setStarting] = useState(false);
  const [paper, setPaperState] = useState<Paper>(() => local.get<Paper>('scaleSheet.paper', defaultPaper()));
  const [barText, setBarText] = useState<string>(() => local.get<string>('scaleSheet.barMm', ''));
  const bar = barValue(barText);

  const setPaper = (p: Paper) => {
    setPaperState(p);
    local.set('scaleSheet.paper', p);
  };
  const setBar = (text: string) => {
    setBarText(text);
    local.set('scaleSheet.barMm', text);   // the same printed sheet is used for part after part
  };

  const enough = photos.length >= 2;
  const start = async () => {
    setStarting(true);
    const pid = projectParam(useStore.getState().projectId);
    await submitJob('/api/photos-to-3d', { ...(pid ? { project_id: pid } : {}), ...(bar.mm ? { ruler_mm: bar.mm } : {}) });
    setStarting(false);
  };

  return (
    <Block guide="photos-to-3d" title="Make a 3D model from photos">
      <p className="hint-text">
        No scanner at hand? Take 12 or more photos of the part, about every 30° all the way round, and CloudClean builds a coloured 3D model from them. Put the part on the printed scale sheet and the model comes out at its true size; without the sheet its size is unknown until you give one known length. Its surface is still about 0.5–1.5 mm off (holes and small details more), so measure diameters and tolerances with the scanner.
      </p>

      <section className="p3d-sheet" aria-labelledby="p3d-sheet-title" data-guide="photos-to-3d.sheet">
        <a className="p3d-sheet-thumb" style={{ aspectRatio: paper === 'a4' ? '210 / 297' : '215.9 / 279.4' }} href={`/api/photos/scale-sheet?paper=${paper}`} target="_blank" rel="noreferrer" tabIndex={-1} aria-hidden>
          <img src={`/api/photos/scale-sheet?paper=${paper}&format=png`} alt="" loading="lazy" />
        </a>
        <div className="p3d-sheet-body">
          <h4 className="p3d-sheet-title" id="p3d-sheet-title">True size from a printed sheet</h4>
          <p className="hint-text">Put the part in the middle of the sheet and photograph it all round, keeping some of the black squares in every photo.</p>
          <div className="p3d-sheet-actions">
            <a className="btn btn-secondary btn-sm" href={`/api/photos/scale-sheet?paper=${paper}`} target="_blank" rel="noreferrer">
              <Printer size={14} aria-hidden />
              <span className="btn-label">Print the scale sheet</span>
            </a>
            <Segmented<Paper> size="sm" ariaLabel="Paper size" value={paper} onChange={setPaper} options={[{ value: 'a4', label: 'A4' }, { value: 'letter', label: 'US Letter' }]} />
          </div>
          <p className="p3d-sheet-note">Print at 100 % (actual size), not “fit to page”.</p>
          <label className="p3d-bar">
            <span className="p3d-bar-label">The 100 mm bar measures</span>
            <span className="input-wrap p3d-bar-input">
              <input className="input input-number" type="text" inputMode="decimal" placeholder="100.0" value={barText} onChange={e => setBar(e.target.value)} aria-invalid={!!bar.error} aria-describedby="p3d-bar-help" />
              <span className="input-unit">mm</span>
            </span>
          </label>
          <p id="p3d-bar-help" className={bar.error ? 'err-text' : 'p3d-sheet-note'} role={bar.error ? 'alert' : undefined}>
            {bar.error ?? 'Optional: measure the printed bar with a caliper. It corrects a printer that prints a little big or small.'}
          </p>
        </div>
      </section>

      {photos.length > 0 && <PhotoGrid photos={photos} max={SHOWN} label={`${photos.length} photos in this project`} />}
      <PhotoViewerHost />
      <p className="p3d-count">
        <Camera size={15} aria-hidden />
        {photos.length === 0 ? 'No photos in this project yet' : `${photos.length} photo${photos.length > 1 ? 's' : ''} in this project`}
        {photos.length > 0 && photos.length < 12 && <span className="muted"> · 12 or more work best</span>}
      </p>
      {running ? (
        <div className="p3d-running" role="status">
          <Progress value={running.progress?.fraction ?? null} indeterminate={!running.progress} />
          <span className="hint-text">{running.progress?.label || 'Starting'}… usually 1–3 minutes (the first run also downloads the model).</span>
        </div>
      ) : (
        <div className="row wrap">
          <Button icon={<ImagePlus size={16} />} onClick={() => pickFiles('image/*,.heic,.heif')}>
            Add photos
          </Button>
          <Button variant="primary" icon={<Sparkles size={16} />} loading={starting} disabled={!enough || status?.available === false || !!bar.error} onClick={start}>
            Make the 3D model
          </Button>
        </div>
      )}
      {!enough && !running && <p className="hint-text">Add at least 2 photos to start.</p>}
      <p className="hint-text">Already scanned this part? <b>Mesh → Colour from photos</b> paints these photos' real colours onto your scan instead.</p>
      {status && !status.available && <Callout tone="warn">{status.reason}</Callout>}
      <details className="disclosure">
        <summary>
          <Camera size={16} aria-hidden /> How to take good photos
          <ChevronDown size={16} className="chev" aria-hidden />
        </summary>
        <div className="disclosure-body">
          <ul className="p3d-tips">
            <li>Walk all the way round the part: a photo about every 30° (12 per round), keeping the same height. A second round from higher up adds the top (24–36 photos rebuild the most). Neighbouring photos must overlap: jumps of 90° or more, or changing height and angle at once, cannot be linked.</li>
            <li>Keep the whole part in frame and sharp. Even light, no flash; matte or sprayed parts work best.</li>
            <li>Put the part on the printed scale sheet, flat on the table (tape its corners down if it curls). Its dots help the photos line up, and its black squares give the true size. Keep some squares in every photo.</li>
            <li>No sheet? Put the part on a patterned surface (newspaper) with a ruler next to it, then open Clean → Set the true size and type one length you know, or just tell the assistant the real length.</li>
          </ul>
        </div>
      </details>
    </Block>
  );
}
