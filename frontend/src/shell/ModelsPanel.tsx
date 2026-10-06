import { useState } from 'react';
import { Box, Camera, Crosshair, FileInput, FolderInput, PanelLeftClose, Plus, ScanLine, Search, Upload, X } from 'lucide-react';
import type { Asset } from '../lib/types';
import { guideTo } from '../lib/guide';
import { importPaths, pickFiles } from '../lib/importing';
import { useProjectAssets, useStore } from '../store';
import { Button, Empty, IconButton, Popover } from '../ui/primitives';
import { checkEntries, displayName, GROUPS, groupOf, type Entry, type GroupId } from './models/groups';
import { useCheckInfos } from './models/checkInfo';
import { GroupCard } from './models/GroupCard';
import { ModelRow } from './models/ModelRow';
import { PhotoGrid, PhotoViewerHost } from './models/photos';

// kept here: other parts of the app import them from this file
export { SERIES_VARS, assetColorVar, sizeText, Thumb } from './models/thumb';

const MODEL_GROUPS: Exclude<GroupId, 'photos'>[] = ['scans', 'made', 'checks', 'cad'];

const newestFirst = (x: Asset, y: Asset) => y.created.localeCompare(x.created);

/**
 * The left panel: every model of the project, sorted into what each thing is (scans, what was made from them,
 * checks, CAD / golden models, photos), each group a card of its own colour. Click = work on it, double-click =
 * show only this, Ctrl-click = pick several; the eye shows or hides a model in the 3D view.
 */
export function ModelsPanel() {
  const all = useProjectAssets();
  const step = useStore(s => s.step);
  const booted = useStore(s => s.booted);
  const golden = useStore(s => s.projects.find(p => p.id === s.projectId)?.golden_asset_id ?? null);
  const picked = useStore(s => s.selected.length);
  const [query, setQuery] = useState('');
  const picking = step === 'align';

  const geometry = all.filter(a => a.kind !== 'image');
  const photosAll = all.filter(a => a.kind === 'image');
  const goldenChecks = geometry.filter(a => a.operation === 'golden_check').map(a => a.id);
  const infos = useCheckInfos(goldenChecks);

  const q = query.trim().toLowerCase();
  const matches = (a: Asset) => !q || a.name.toLowerCase().includes(q) || displayName(a).toLowerCase().includes(q);

  const by: Record<GroupId, Asset[]> = { scans: [], made: [], checks: [], cad: [], photos: [] };
  for (const a of all) by[groupOf(a, golden)].push(a);
  const links = Object.fromEntries(goldenChecks.map(id => {
    const info = infos[id];
    return [id, info === undefined ? undefined : info === 'none' ? null : info.compareId];
  }));
  const flat = (list: Asset[]): Entry[] => list.map(asset => ({ asset, children: [] }));
  const entries: Record<Exclude<GroupId, 'photos'>, Entry[]> = {
    scans: flat([...by.scans].sort(newestFirst)),
    made: flat([...by.made].sort(newestFirst)),
    checks: checkEntries(by.checks, links),
    cad: flat([...by.cad].sort((x, y) => Number(y.id === golden) - Number(x.id === golden) || newestFirst(x, y))),
  };
  const shownEntries = (g: Exclude<GroupId, 'photos'>) => (q ? entries[g].filter(e => matches(e.asset) || e.children.some(matches)) : entries[g]);
  const photos = photosAll.filter(matches);
  const anyMatch = MODEL_GROUPS.some(g => shownEntries(g).length > 0) || photos.length > 0;

  return (
    <aside className="island models" aria-label="Models in this project" data-guide="models">
      <header className="panel-head">
        <div className="panel-title">
          Models <span className="count-chip">{geometry.length}</span>
        </div>
        <span className="spacer" />
        <AddMenu />
        <IconButton size="sm" label="Hide the model list (Ctrl B)" onClick={() => useStore.getState().setLayout({ leftOpen: false })}>
          <PanelLeftClose size={17} />
        </IconButton>
      </header>

      {geometry.length > 6 && (
        <label className="models-search">
          <Search size={15} aria-hidden />
          <input className="input" placeholder="Find a model" value={query} onChange={e => setQuery(e.target.value)} aria-label="Find a model" />
          {query && (
            <button type="button" className="mp-search-clear" aria-label="Clear" onClick={() => setQuery('')}>
              <X size={14} />
            </button>
          )}
        </label>
      )}

      <div className="models-scroll mp-scroll">
        {!booted ? (
          <div className="skeleton-list" aria-label="Loading models">{[0, 1, 2, 3].map(i => <div key={i} className="skeleton-row"><span /><span /></div>)}</div>
        ) : (
          <>
            {geometry.length === 0 && (
              <Empty
                icon={<Box size={24} />}
                title="No models yet"
                action={
                  <div className="stack tight" style={{ width: 220 }}>
                    <Button variant="primary" icon={<ScanLine size={16} />} onClick={() => useStore.getState().goStep('capture')}>Scan a part</Button>
                    <Button icon={<Upload size={16} />} onClick={() => pickFiles()}>Open scan files</Button>
                  </div>
                }
              >
                Scan with the MetroY, or drop PLY, STL, OBJ and STEP files anywhere.
              </Empty>
            )}

            {picking && geometry.length > 0 && (
              <div className="mp-picking" role="note">
                <Crosshair size={16} aria-hidden />
                <span>
                  <b>Pick the scans to align.</b> The first one you pick is the reference.
                  {picked > 0 && <span className="mp-picking-n"> {picked} picked</span>}
                </span>
              </div>
            )}

            {MODEL_GROUPS.map(g => {
              const list = shownEntries(g);
              if (!list.length) return null;
              const ids = entries[g].flatMap(e => [e.asset.id, ...e.children.map(c => c.id)]);
              return (
                <GroupCard key={g} id={g} ids={ids} count={ids.length}>
                  <ul className="mp-list" role="tree" aria-label={GROUPS[g].title}>
                    {list.map(e => (
                      <EntryRows key={e.asset.id} entry={e} group={g} picking={picking} golden={golden} check={infos[e.asset.id]} />
                    ))}
                  </ul>
                </GroupCard>
              );
            })}

            {photos.length > 0 && (
              <GroupCard id="photos" ids={[]} count={photosAll.length} photos={photosAll}>
                <PhotoGrid photos={photos} />
              </GroupCard>
            )}

            {q && !anyMatch && <p className="mp-nomatch">Nothing matches “{query}”.</p>}
          </>
        )}
      </div>
      <PhotoViewerHost />
    </aside>
  );
}

function EntryRows({ entry, group, picking, golden, check }: { entry: Entry; group: GroupId; picking: boolean; golden: string | null; check: ReturnType<typeof useCheckInfos>[string] | undefined }) {
  return (
    <>
      <ModelRow asset={entry.asset} group={group} picking={picking} golden={entry.asset.id === golden} check={check} />
      {entry.children.map(c => (
        <ModelRow key={c.id} asset={c} group={group} picking={picking} child />
      ))}
    </>
  );
}

function AddMenu() {
  const [pathText, setPathText] = useState('');
  return (
    <Popover
      align="end"
      trigger={({ toggle, open }) => (
        <Button size="sm" variant="primary" icon={<Plus size={15} />} onClick={toggle} aria-expanded={open} aria-label="Add" title="Add a scan, files or photos" data-guide="add">
          Add
        </Button>
      )}
    >
      {({ close }) => (
        <div className="menu" style={{ width: 320 }}>
          <button type="button" className="menu-item" onClick={() => { close(); useStore.getState().goStep('capture'); }}>
            <ScanLine size={16} />
            <div>
              <div className="menu-item-title">Scan a part</div>
              <div className="menu-item-sub">MetroY, turntable and live guidance</div>
            </div>
          </button>
          <button type="button" className="menu-item" onClick={() => { close(); pickFiles(); }}>
            <Upload size={16} />
            <div>
              <div className="menu-item-title">Open files</div>
              <div className="menu-item-sub">Scans, meshes, CAD (STEP, STL) and photos</div>
            </div>
          </button>
          <button type="button" className="menu-item" onClick={() => { close(); guideTo('photos-to-3d', 'Make a 3D model from photos', 'Scan → Make a 3D model from photos', { step: 'capture', right: 'step' }); }}>
            <Camera size={16} />
            <div>
              <div className="menu-item-title">3D model from photos</div>
              <div className="menu-item-sub">20–60 photos all the way round, no scanner needed</div>
            </div>
          </button>
          <div className="menu-sep" />
          <div className="stack tight" style={{ padding: '6px 10px 10px' }}>
            <div className="menu-item-title row"><FolderInput size={15} aria-hidden /> Import from the server’s disk</div>
            <div className="menu-item-sub">Fastest for big scans — nothing is uploaded.</div>
            <textarea className="input mono" rows={3} spellCheck={false} value={pathText} placeholder={'/data/scans/part_scan1.ply'} onChange={e => setPathText(e.target.value)} style={{ fontSize: 12.5 }} />
            <div className="row">
              <Button size="sm" icon={<FileInput size={14} />} disabled={!pathText.trim()} onClick={async () => {
                if (await importPaths(pathText.split(/\r?\n/).map(s => s.trim()).filter(Boolean))) { setPathText(''); close(); }
              }}>Import</Button>
              {pathText && <IconButton size="sm" label="Clear" onClick={() => setPathText('')}><X size={14} /></IconButton>}
            </div>
          </div>
        </div>
      )}
    </Popover>
  );
}
