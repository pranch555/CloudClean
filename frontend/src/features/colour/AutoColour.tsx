import { useState } from 'react';
import { ImagePlus, Paintbrush } from 'lucide-react';
import { pickFiles } from '../../lib/importing';
import { isFinal, submitJob } from '../../lib/jobs';
import { projectParam } from '../../lib/projects';
import { useReconStatus } from '../../lib/recon';
import type { Asset } from '../../lib/types';
import { useProjectAssets, useStore } from '../../store';
import { Button, Progress, Switch } from '../../ui/primitives';
import { Callout } from '../../steps/capture/parts';
import { useReport } from '../../steps/useReport';

const SHOWN = 8;
const MIN_PHOTOS = 3;

/**
 * "Colour from photos", automatic: the photos of the project are placed around the part (where each was taken),
 * lined up with the scanned model and projected onto it (job colour_from_photos). Works on meshes (colours and a
 * texture image) and point clouds (a colour per point).
 */
export function AutoColour({ target }: { target: Asset | undefined }) {
  const photos = useProjectAssets().filter(a => a.kind === 'image');
  const status = useReconStatus();
  const running = useStore(s => s.jobs.find(j => j.kind === 'colour_from_photos' && !isFinal(j)));
  const [texture, setTexture] = useState(true);
  const [starting, setStarting] = useState(false);
  const mesh = target?.kind === 'mesh';
  const enough = photos.length >= MIN_PHOTOS;
  const report = useReport(target?.operation === 'texture' ? target : undefined);
  const align = report?.alignment as { trusted?: boolean; within_mm?: number; fitness?: number; coverage?: number } | undefined;

  const start = async () => {
    if (!target) return;
    setStarting(true);
    const pid = projectParam(useStore.getState().projectId);
    await submitJob('/api/colour-from-photos', { asset_id: target.id, texture_size: mesh && texture ? 4096 : 0, ...(pid ? { project_id: pid } : {}) });
    setStarting(false);
  };

  return (
    <div className="stack">
      <p className="hint-text">
        Give it photos of the part you scanned — 12 or more, about every 30° all the way round at the same height (6, every 60°, is the fewest that worked in tests) — and CloudClean works out where each was taken, lines them up with your
        model and paints the real colours onto it. Your model's shape and size are not changed.
      </p>
      {align && report && (
        <Callout tone={align.trusted && !(report.warnings ?? []).length ? 'info' : 'warn'}>
          <b>{target?.name}</b> was coloured from {report.used} of {report.photos} photos, lined up with the model to{' '}
          {Math.round((align.fitness ?? 0) * 100)}% within {(align.within_mm ?? 0).toFixed(2)} mm;{' '}
          {Math.round((report.coverage ?? 0) * 100)}% of the surface was seen in a photo, the rest filled in from around it.
          {(report.warnings as string[] | undefined)?.map(w => (
            <span key={w} className="p3d-warning">
              {' '}
              {w}
            </span>
          ))}
        </Callout>
      )}
      {target ? (
        <p className="p3d-count">
          Colours <b>{target.name}</b> <span className="muted">· {mesh ? 'mesh: colours and a texture image' : 'point cloud: a colour per point'}</span>
        </p>
      ) : (
        <p className="warn-text">Click the scan or mesh to colour in the Models list first.</p>
      )}
      {photos.length > 0 && (
        <ul className="p3d-strip" aria-label={`${photos.length} photos in this project`}>
          {photos.slice(-SHOWN).map(p => (
            <li key={p.id} className="p3d-thumb">
              <img src={`/api/assets/${p.id}/image`} alt={p.name} loading="lazy" />
            </li>
          ))}
          {photos.length > SHOWN && <li className="p3d-more">+{photos.length - SHOWN}</li>}
        </ul>
      )}
      <p className="p3d-count">
        {photos.length === 0 ? 'No photos in this project yet' : `${photos.length} photo${photos.length > 1 ? 's' : ''} in this project`}
        {photos.length > 0 && photos.length < 12 && <span className="muted"> · 12 or more work best</span>}
      </p>
      {mesh && (
        <Switch checked={texture} onChange={setTexture} label="Also make a texture image (sharper colours; for GLB / OBJ export)" />
      )}
      {running ? (
        <div className="p3d-running" role="status">
          <Progress value={running.progress?.fraction ?? null} indeterminate={!running.progress} />
          <span className="hint-text">{running.progress?.label || 'Starting'}… usually 2–4 minutes.</span>
        </div>
      ) : (
        <div className="row wrap">
          <Button icon={<ImagePlus size={16} />} onClick={() => pickFiles('image/*,.heic,.heif')}>
            Add photos
          </Button>
          <Button variant="primary" icon={<Paintbrush size={16} />} loading={starting} disabled={!target || !enough || status?.available === false} onClick={start}>
            Colour it from {enough ? photos.length : ''} photos
          </Button>
        </div>
      )}
      {!enough && !running && <p className="hint-text">Add at least {MIN_PHOTOS} photos of the part to start.</p>}
      {status && !status.available && <Callout tone="warn">{status.reason}</Callout>}
    </div>
  );
}
