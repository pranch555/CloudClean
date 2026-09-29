import { useMemo, useState } from 'react';
import { Box, Camera, Download, Eye, EyeOff, FileInput, FolderInput, FolderOutput, Image as ImageIcon, MoreHorizontal, PanelLeftClose, Pencil, Plus, ScanLine, Search, Shapes, Sparkle, Trash2, Upload, X } from 'lucide-react';
import { api } from '../lib/api';
import type { Asset } from '../lib/types';
import { fmtCount } from '../lib/format';
import { guideTo } from '../lib/guide';
import { importPaths, pickFiles } from '../lib/importing';
import { moveAsset, projectsSupported } from '../lib/projects';
import { ROLE_LABEL, roleOf } from '../lib/journey';
import { useThumbs } from '../lib/thumbnails';
import { useProjectAssets, useStore } from '../store';
import { getViewer } from '../viewer/instance';
import { Button, Empty, IconButton, Popover } from '../ui/primitives';

export const SERIES_VARS = ['--series-1', '--series-2', '--series-3', '--series-4', '--series-5', '--series-6', '--series-7', '--series-8'];

/** Stable colour per model (order of creation among geometry assets). */
export const assetColorVar = (assets: Asset[], id: string) => {
  const geo = assets.filter(a => a.kind !== 'image');
  const i = geo.findIndex(a => a.id === id);
  return SERIES_VARS[(i < 0 ? 0 : i) % SERIES_VARS.length];
};

interface Node {
  asset: Asset;
  children: Node[];
}

function buildTree(assets: Asset[]): Node[] {
  const byId = new Map(assets.map(a => [a.id, { asset: a, children: [] as Node[] }]));
  const roots: Node[] = [];
  for (const a of assets) {
    const node = byId.get(a.id)!;
    const parent = a.parents.map(p => byId.get(p)).find(p => p && p.asset.kind !== 'image');
    if (parent) parent.children.push(node);
    else roots.push(node);
  }
  const sort = (nodes: Node[], newestFirst: boolean) => {
    nodes.sort((x, y) => (newestFirst ? y.asset.created.localeCompare(x.asset.created) : x.asset.created.localeCompare(y.asset.created)));
    nodes.forEach(n => sort(n.children, false));
  };
  sort(roots, true);
  return roots;
}

/** Part size in the model's own axes when known (smallest box), else the scanner-frame box. */
export function sizeText(a: Asset): string {
  const p = a.part?.dimensions;
  const d = p ? [p.length, p.width, p.height] : a.stats.oriented_dimensions ?? a.stats.dimensions;
  if (!d) return '';
  return [...d].sort((x, y) => y - x).map(v => (v >= 100 ? v.toFixed(0) : v.toFixed(1))).join(' × ');
}

export function Thumb({ asset, size = 52 }: { asset: Asset; size?: number }) {
  const local = useThumbs(s => s.urls[asset.id]);
  const assets = useStore(s => s.assets);
  const [failed, setFailed] = useState(false);
  const src = local ?? (asset.has_thumbnail && !failed ? `/api/assets/${asset.id}/thumbnail?v=${encodeURIComponent(asset.created)}` : null);
  const Icon = asset.kind === 'mesh' ? Shapes : asset.kind === 'image' ? ImageIcon : Sparkle;
  return (
    <span className="thumb" style={{ width: size, height: size }}>
      {asset.kind === 'image' ? <img src={`/api/assets/${asset.id}/image`} alt="" loading="lazy" /> : src ? <img src={src} alt="" onError={() => setFailed(true)} /> : <Icon size={20} aria-hidden />}
      {asset.kind !== 'image' && <span className="swatch" style={{ background: `var(${assetColorVar(assets, asset.id)})` }} aria-hidden />}
    </span>
  );
}

export function ModelsPanel() {
  const all = useProjectAssets();
  const step = useStore(s => s.step);
  const [query, setQuery] = useState('');
  const geometry = all.filter(a => a.kind !== 'image');
  const cad = geometry.filter(a => roleOf(a) === 'cad');
  const parts = geometry.filter(a => roleOf(a) !== 'cad');
  const photos = all.filter(a => a.kind === 'image');
  const q = query.trim().toLowerCase();
  const tree = useMemo(() => buildTree(parts), [parts]);
  const flat = q ? geometry.filter(a => a.name.toLowerCase().includes(q)).reverse() : null;
  const picking = step === 'align';
  const booted = useStore(s => s.booted);

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
        </label>
      )}

      <div className="models-scroll">
        {!booted ? (
          <div className="skeleton-list" aria-label="Loading models">{[0, 1, 2, 3].map(i => <div key={i} className="skeleton-row"><span /><span /></div>)}</div>
        ) : geometry.length === 0 ? (
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
        ) : flat ? (
          <ul className="model-list" role="tree">
            {flat.map(a => <ModelRow key={a.id} asset={a} depth={0} picking={picking} />)}
            {!flat.length && <li className="caption" style={{ padding: 12 }}>Nothing matches “{query}”.</li>}
          </ul>
        ) : (
          <>
            {picking && <div className="group-label">Pick the scans to align · the first pick is the reference</div>}
            <ul className="model-list" role="tree">
              {tree.map(n => <TreeNode key={n.asset.id} node={n} depth={0} picking={picking} />)}
            </ul>
            {cad.length > 0 && (
              <>
                <div className="group-label">CAD models</div>
                <ul className="model-list" role="tree">
                  {cad.map(a => <ModelRow key={a.id} asset={a} depth={0} picking={picking} />)}
                </ul>
              </>
            )}
          </>
        )}

        {photos.length > 0 && (
          <>
            <div className="group-label">Photos · {photos.length}</div>
            <div className="photo-strip">
              {photos.map(p => (
                <button key={p.id} type="button" className="photo-thumb" title={p.name} onClick={() => useStore.getState().set({ activeId: p.id, step: 'mesh', screen: 'workspace' })}>
                  <img src={`/api/assets/${p.id}/image`} alt={p.name} loading="lazy" />
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </aside>
  );
}

function TreeNode({ node, depth, picking }: { node: Node; depth: number; picking: boolean }) {
  return (
    <>
      <ModelRow asset={node.asset} depth={depth} picking={picking} />
      {node.children.map(c => <TreeNode key={c.asset.id} node={c} depth={Math.min(depth + 1, 5)} picking={picking} />)}
    </>
  );
}

function ModelRow({ asset: a, depth, picking }: { asset: Asset; depth: number; picking: boolean }) {
  const [renaming, setRenaming] = useState(false);
  const pickIndex = useStore(s => s.selected.indexOf(a.id));
  const visible = useStore(s => s.visible.includes(a.id));
  const active = useStore(s => s.activeId === a.id);
  const { activate, toggleSelect, toggleVisible } = useStore.getState();
  const role = roleOf(a);
  const size = sizeText(a);

  return (
    <li
      role="treeitem"
      aria-selected={active}
      className={`model ${active ? 'is-active' : ''} ${pickIndex >= 0 ? 'is-picked' : ''}`}
      style={{ '--depth': depth } as React.CSSProperties}
      tabIndex={0}
      onKeyDown={e => {
        if (e.target !== e.currentTarget) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          if (picking) toggleSelect(a.id);
          activate(a.id);
        } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
          const rows = [...document.querySelectorAll<HTMLElement>('.model[role="treeitem"]')];
          rows[rows.indexOf(e.currentTarget) + (e.key === 'ArrowDown' ? 1 : -1)]?.focus();
        } else if (e.key === 'F2') setRenaming(true);
      }}
      onClick={e => {
        if (picking || e.ctrlKey || e.metaKey) toggleSelect(a.id);
        activate(a.id);
      }}
      onDoubleClick={() => {
        useStore.setState({ visible: [a.id], activeId: a.id });
        setTimeout(() => getViewer()?.fit([a.id]), 80);
      }}
      title={`${a.name}\n${ROLE_LABEL[role]} · ${new Date(a.created).toLocaleString()}\nClick to work on it · double-click to show only this`}
    >
      {depth > 0 && <span className="model-branch" aria-hidden />}
      <Thumb asset={a} size={44} />
      <div className="model-text">
        {renaming ? (
          <input
            className="input model-rename"
            autoFocus
            defaultValue={a.name}
            aria-label="New name"
            onClick={e => e.stopPropagation()}
            onKeyDown={e => {
              if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
              if (e.key === 'Escape') setRenaming(false);
              e.stopPropagation();
            }}
            onBlur={async e => {
              const name = e.target.value.trim();
              setRenaming(false);
              if (!name || name === a.name) return;
              try {
                await api.patch(`/api/assets/${a.id}`, { name });
                await useStore.getState().refreshAssets();
              } catch (err) {
                useStore.getState().toast({ kind: 'error', title: 'Rename failed', body: (err as Error).message });
              }
            }}
          />
        ) : (
          <div className="model-name">{a.name}</div>
        )}
        <div className="model-meta">
          <span className={`role role-${role}`}>{ROLE_LABEL[role]}</span>
          <span>·</span>
          <span>{a.kind === 'mesh' ? `${fmtCount(a.stats.triangles)} tri` : `${fmtCount(a.stats.points)} pts`}</span>
        </div>
        {size && <div className="model-meta"><span className="mono">{size} mm</span></div>}
      </div>
      <div className="model-actions" onClick={e => e.stopPropagation()}>
        {picking ? (
          <button type="button" className={`pick ${pickIndex >= 0 ? 'is-on' : ''}`} aria-label={pickIndex >= 0 ? `Picked #${pickIndex + 1}` : 'Pick for alignment'} onClick={() => toggleSelect(a.id)}>
            {pickIndex >= 0 ? pickIndex + 1 : ''}
          </button>
        ) : (
          <IconButton size="sm" label={visible ? 'Hide in the 3D view' : 'Show in the 3D view'} className={`eye ${visible ? 'is-on' : ''}`} onClick={() => toggleVisible(a.id)}>
            {visible ? <Eye size={16} /> : <EyeOff size={16} />}
          </IconButton>
        )}
        <RowMenu asset={a} onRename={() => setRenaming(true)} />
      </div>
    </li>
  );
}

function RowMenu({ asset: a, onRename }: { asset: Asset; onRename: () => void }) {
  const formats = useStore(s => s.params?.formats);
  const projects = useStore(s => s.projects);
  const others = projects.filter(p => !p.virtual && p.id !== a.project);
  const move = async (projectId: string, name: string) => {
    try {
      await moveAsset(a.id, projectId);
      await useStore.getState().refreshAssets();
      await useStore.getState().refreshProjects();
      useStore.getState().toast({ kind: 'ok', title: `Moved to ${name}` });
    } catch (err) {
      useStore.getState().toast({ kind: 'error', title: 'Could not move the model', body: (err as Error).message });
    }
  };
  const del = async () => {
    if (!confirm(`Delete “${a.name}”? Its files are removed from the workspace. Models made from it stay.`)) return;
    try {
      await api.del(`/api/assets/${a.id}`);
      getViewer()?.forget(a.id);
      await useStore.getState().refreshAssets();
      useStore.getState().refreshProjects().catch(() => undefined);
    } catch (err) {
      useStore.getState().toast({ kind: 'error', title: 'Delete failed', body: (err as Error).message });
    }
  };
  const fmt = a.kind === 'mesh' ? 'stl' : (formats?.pointcloud?.[0] ?? 'ply');
  return (
    <Popover
      align="end"
      trigger={({ toggle, open }) => (
        <IconButton size="sm" label="More" className="more" active={open} onClick={toggle} data-guide="model.menu">
          <MoreHorizontal size={16} />
        </IconButton>
      )}
    >
      {({ close }) => (
        <div className="menu">
          <button type="button" className="menu-item" onClick={() => { close(); onRename(); }}><Pencil size={15} /><div className="menu-item-title">Rename</div></button>
          <a className="menu-item" href={`/api/assets/${a.id}/download?format=${fmt}`} download onClick={() => close()}><Download size={15} /><div className="menu-item-title">Download {fmt.toUpperCase()}</div></a>
          <button type="button" className="menu-item" onClick={() => { close(); useStore.getState().set({ activeId: a.id, step: 'export' }); }}><FileInput size={15} /><div className="menu-item-title">Details & more formats</div></button>
          {projectsSupported() && others.length > 0 && (
            <>
              <div className="menu-sep" />
              <div className="menu-label"><FolderOutput size={13} style={{ display: 'inline', verticalAlign: -2 }} aria-hidden /> Move to project</div>
              {others.slice(0, 8).map(p => (
                <button key={p.id} type="button" className="menu-item" onClick={() => { close(); move(p.id, p.name); }}><div className="menu-item-title truncate">{p.name}</div></button>
              ))}
            </>
          )}
          <div className="menu-sep" />
          <button type="button" className="menu-item danger-text" onClick={() => { close(); del(); }}><Trash2 size={15} /><div className="menu-item-title">Delete</div></button>
        </div>
      )}
    </Popover>
  );
}

function AddMenu() {
  const [pathText, setPathText] = useState('');
  return (
    <Popover
      align="end"
      trigger={({ toggle, open }) => (
        <Button size="sm" variant="primary" icon={<Plus size={15} />} onClick={toggle} aria-expanded={open} data-guide="add">
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
