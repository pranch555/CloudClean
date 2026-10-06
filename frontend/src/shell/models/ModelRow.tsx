import { useEffect, useRef, useState } from 'react';
import { BadgeCheck, Download, EyeClosed, FileInput, FolderOutput, MoreHorizontal, Pencil, Star, Trash2 } from 'lucide-react';
import { api } from '../../lib/api';
import { moveAsset, projectParam, projectsSupported } from '../../lib/projects';
import type { Asset } from '../../lib/types';
import { useStore } from '../../store';
import { getViewer } from '../../viewer/instance';
import { setTab } from '../../steps/measure/state';
import { IconButton, Popover } from '../../ui/primitives';
import { Thumb, sizeText } from './thumb';
import { breakable, CHECK_BLURB, CHECK_TITLE, checkKind, countText, displayName, kindTag, madeFrom, whenText, type GroupId } from './groups';
import type { CheckInfo } from './checkInfo';
import { ConfirmDelete } from './photos';
import { revealModel, usePanelUi } from './panelState';

/** Eye with a filled pupil: "shown in the 3D view". */
function EyeOn({ size = 17 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M2.06 12.35a1 1 0 0 1 0-.7 10.75 10.75 0 0 1 19.88 0 1 1 0 0 1 0 .7 10.75 10.75 0 0 1-19.88 0" />
      <circle cx="12" cy="12" r="3.2" fill="currentColor" />
    </svg>
  );
}

/** Show / hide one model in the 3D view. Shown: a filled signal-red eye; hidden: a calm closed eye. */
export function EyeToggle({ on, onClick, name, guide }: { on: boolean; onClick: () => void; name: string; guide?: string }) {
  const tip = on ? 'Shown in the 3D view · click to hide' : 'Hidden · click to show in the 3D view';
  return (
    <button type="button" className={`mp-eye ${on ? 'is-on' : ''}`} onClick={onClick} aria-pressed={on} aria-label={`${on ? 'Hide' : 'Show'} ${name} in 3D`} data-tip={tip} data-tip-side="left" data-guide={guide}>
      {on ? <EyeOn /> : <EyeClosed size={17} aria-hidden />}
    </button>
  );
}

/** Opens Measure -> Golden model on this check. */
export function openCheck(id: string) {
  const st = useStore.getState();
  st.activate(id);
  setTab('cad');
  st.goStep('measure');
}

interface RowProps {
  asset: Asset;
  group: GroupId;
  picking: boolean;
  /** nested under the check that made it (one level) */
  child?: boolean;
  golden?: boolean;
  check?: CheckInfo | 'none';
}

export function ModelRow({ asset: a, group, picking, child, golden, check }: RowProps) {
  const [renaming, setRenaming] = useState(false);
  const pickIndex = useStore(s => s.selected.indexOf(a.id));
  const visible = useStore(s => s.visible.includes(a.id));
  const active = useStore(s => s.activeId === a.id);
  const byId = useStore(s => s.byId);
  const reveal = usePanelUi(s => (s.reveal?.id === a.id ? s.reveal.n : 0));
  const { activate, toggleSelect, toggleVisible } = useStore.getState();
  const ref = useRef<HTMLLIElement>(null);
  const name = displayName(a);
  const custom = name === a.name;
  const ck = checkKind(a);
  const tag = group === 'checks' ? null : kindTag(a);
  const size = sizeText(a);
  const from = madeFrom(a, byId);

  // a model picked elsewhere (the 3D view, a "Made from" link, a new result) scrolls into view
  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: 'nearest' });
  }, [active]);
  useEffect(() => {
    if (!reveal) return;
    ref.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    ref.current?.focus({ preventScroll: true });
  }, [reveal]);

  const onKey = (e: React.KeyboardEvent<HTMLLIElement>) => {
    if (e.target !== e.currentTarget) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (picking) toggleSelect(a.id);
      activate(a.id);
    } else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) {
      e.preventDefault();
      const rows = [...(e.currentTarget.closest('.models')?.querySelectorAll<HTMLElement>('.mp-row[role="treeitem"]') ?? [])];
      const i = rows.indexOf(e.currentTarget);
      const next = e.key === 'Home' ? rows[0] : e.key === 'End' ? rows[rows.length - 1] : rows[i + (e.key === 'ArrowDown' ? 1 : -1)];
      next?.focus();
      next?.scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'F2') setRenaming(true);
  };

  const tooltip = [
    a.name,
    `${ck ? CHECK_TITLE[ck] : tag?.label ?? 'Scan'} · ${countText(a)} · ${new Date(a.created).toLocaleString()}`,
    picking ? 'Click to pick it for aligning' : 'Click to work on it · double-click to show only this',
  ].join('\n');

  return (
    <li
      ref={ref}
      data-asset={a.id}
      role="treeitem"
      aria-selected={active}
      aria-level={child ? 2 : 1}
      className={`mp-row ${child ? 'is-child' : ''} ${active ? 'is-active' : ''} ${pickIndex >= 0 ? 'is-picked' : ''} ${visible ? 'is-shown' : ''}`}
      tabIndex={0}
      onKeyDown={onKey}
      onClick={e => {
        if (picking || e.ctrlKey || e.metaKey) toggleSelect(a.id);
        activate(a.id);
      }}
      onDoubleClick={() => {
        useStore.setState({ visible: [a.id], activeId: a.id });
        setTimeout(() => getViewer()?.fit([a.id]), 80);
      }}
      title={tooltip}
    >
      <Thumb asset={a} size={child ? 34 : 44} />
      <div className="mp-text">
        {active && !child && <span className="mp-flag">Working on</span>}
        {renaming ? (
          <RenameInput asset={a} onDone={() => setRenaming(false)} />
        ) : (
          <div className="mp-name">{breakable(name)}</div>
        )}

        {ck ? (
          <CheckLines asset={a} kind={ck} custom={custom} check={check} child={child} />
        ) : (
          <>
            <div className="mp-tags">
              {golden && (
                <span className="mp-tag t-golden" data-guide="model.golden" title="The golden model of this project: golden checks compare scans with it">
                  <Star size={12} fill="currentColor" aria-hidden /> Golden model
                </span>
              )}
              {tag && <span className={`mp-tag t-${tag.tone}`}>{tag.label}</span>}
              <span className="mp-countline">{countText(a)}</span>
            </div>
            {size && <SizeText size={size} />}
            {(from.models.length > 0 || from.photos > 0) && <MadeFrom models={from.models} photos={from.photos} />}
          </>
        )}
      </div>
      <div className="mp-actions" onClick={e => e.stopPropagation()} onDoubleClick={e => e.stopPropagation()}>
        {picking ? (
          <button type="button" className={`mp-pick ${pickIndex >= 0 ? 'is-on' : ''}`} aria-pressed={pickIndex >= 0} aria-label={pickIndex >= 0 ? `Picked #${pickIndex + 1} for aligning` : `Pick ${name} for aligning`} data-tip={pickIndex >= 0 ? (pickIndex === 0 ? 'Picked first: the reference' : `Picked #${pickIndex + 1}`) : 'Pick for aligning'} data-tip-side="left" onClick={() => toggleSelect(a.id)}>
            {pickIndex >= 0 ? pickIndex + 1 : ''}
          </button>
        ) : (
          <EyeToggle on={visible} onClick={() => toggleVisible(a.id)} name={name} guide="model.show" />
        )}
        <RowMenu asset={a} golden={!!golden} onRename={() => setRenaming(true)} />
      </div>
    </li>
  );
}

const pct = (v: number) => (v >= 99.95 ? '100' : v.toFixed(v >= 10 ? 0 : 1));

/** "107 × 35.0 × 35.0 mm": numbers in mono, the × and the unit in the (narrower) text face. */
function SizeText({ size }: { size: string }) {
  const parts = size.split(' × ');
  return (
    <div className="mp-size" title="Size of the part along its own length, width and height">
      {parts.map((p, i) => (
        <span key={i}>
          {i > 0 && <span className="mp-x">×</span>}
          <span className="mono">{p}</span>
        </span>
      ))}
      <span className="mp-unit">mm</span>
    </div>
  );
}

/** "Made from 0916_05_pc + 0916_06_pc" — each name opens that model. */
function MadeFrom({ models, photos }: { models: Asset[]; photos: number }) {
  const shown = models.slice(0, 3);
  return (
    <div className="mp-from" data-guide="model.made-from">
      <span className="mp-from-label">From</span>
      {shown.map((m, i) => (
        <span key={m.id} className="mp-from-item">
          {i > 0 && <span className="mp-from-sep">+</span>}
          <ParentLink asset={m} />
        </span>
      ))}
      {models.length > shown.length && <span className="mp-from-item">+ {models.length - shown.length} more</span>}
      {photos > 0 && (
        <span className="mp-from-item">
          {models.length > 0 && <span className="mp-from-sep">+</span>}
          {photos} photo{photos === 1 ? '' : 's'}
        </span>
      )}
    </div>
  );
}

function ParentLink({ asset }: { asset: Asset }) {
  return (
    <button
      type="button"
      className="mp-link"
      title={`${asset.name}\nClick to go to it`}
      onClick={e => {
        e.stopPropagation();
        revealModel(asset.id);
      }}
      onDoubleClick={e => e.stopPropagation()}
    >
      {displayName(asset)}
    </button>
  );
}

/** The lines under a check: verdict, how much matched, what was compared with what, when. */
function CheckLines({ asset: a, kind, custom, check, child }: { asset: Asset; kind: NonNullable<ReturnType<typeof checkKind>>; custom: boolean; check?: CheckInfo | 'none'; child?: boolean }) {
  const byId = useStore(s => s.byId);
  const [p0, p1] = a.parents.map(id => byId.get(id));
  const info = check && check !== 'none' ? check : null;
  const tol = info?.tolerance ?? (typeof a.params?.tolerance === 'number' ? (a.params.tolerance as number) : null);
  if (child) {
    return <div className="mp-meta">{CHECK_BLURB[kind]}</div>;
  }
  return (
    <>
      {kind === 'golden' && (
        <div className="mp-tags">
          {info ? (
            <button
              type="button"
              className={`mp-verdict v-${info.grade.tone}`}
              data-guide="model.check-result"
              title={`${info.headline ?? info.grade.label}\nClick to open the result in Measure → Golden model`}
              onClick={e => {
                e.stopPropagation();
                openCheck(a.id);
              }}
              onDoubleClick={e => e.stopPropagation()}
            >
              <BadgeCheck size={14} aria-hidden /> {info.grade.label}
            </button>
          ) : check === 'none' ? null : (
            <span className="mp-verdict is-loading" aria-hidden>
              Reading the result…
            </span>
          )}
          {custom && <span className="mp-tag t-check">Golden check</span>}
        </div>
      )}
      {kind !== 'golden' && custom && (
        <div className="mp-tags">
          <span className="mp-tag t-check">{CHECK_TITLE[kind]}</span>
        </div>
      )}
      {kind === 'golden' && info?.grade.matchPct != null && (
        <div className="mp-meta">
          <span title={`${pct(info.grade.matchPct)}% of the golden model's surface matches the scan within ±${tol ?? 1} mm`}>
            <b>{pct(info.grade.matchPct)}%</b> within ±{tol ?? 1} mm
          </span>
        </div>
      )}
      {kind !== 'golden' && <div className="mp-meta">{CHECK_BLURB[kind]}{tol != null && kind === 'deviation' ? ` · ±${tol} mm` : ''}</div>}
      {(p0 || p1) && (
        <div className="mp-from">
          {p0 ? <ParentLink asset={p0} /> : <span>a deleted scan</span>}
          <span className="mp-from-sep">{kind === 'repeatability' ? 'and' : 'vs'}</span>
          {p1 ? <ParentLink asset={p1} /> : <span>a deleted model</span>}
        </div>
      )}
      <div className="mp-meta mp-when">{whenText(a.created)}</div>
    </>
  );
}

function RenameInput({ asset: a, onDone }: { asset: Asset; onDone: () => void }) {
  return (
    <input
      className="input mp-rename"
      autoFocus
      defaultValue={a.name}
      aria-label="New name"
      onClick={e => e.stopPropagation()}
      onFocus={e => e.target.select()}
      onKeyDown={e => {
        if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
        if (e.key === 'Escape') onDone();
        e.stopPropagation();
      }}
      onBlur={async e => {
        const name = e.target.value.trim();
        onDone();
        if (!name || name === a.name) return;
        try {
          await api.patch(`/api/assets/${a.id}`, { name });
          await useStore.getState().refreshAssets();
        } catch (err) {
          useStore.getState().toast({ kind: 'error', title: 'Rename failed', body: (err as Error).message });
        }
      }}
    />
  );
}

async function deleteModel(a: Asset) {
  try {
    await api.del(`/api/assets/${a.id}`);
    getViewer()?.forget(a.id);
    await useStore.getState().refreshAssets();
    useStore.getState().refreshProjects().catch(() => undefined);
    useStore.getState().toast({ kind: 'ok', title: `Deleted “${displayName(a)}”` });
  } catch (err) {
    useStore.getState().toast({ kind: 'error', title: 'Delete failed', body: (err as Error).message });
  }
}

async function makeGolden(a: Asset) {
  const st = useStore.getState();
  const pid = projectParam(st.projectId);
  if (!pid) return;
  try {
    await api.patch(`/api/projects/${pid}`, { golden_asset_id: a.id });
    await st.refreshProjects();
    st.toast({ kind: 'ok', title: `“${a.name}” is now the golden model`, body: 'Measure → Golden model checks scans against it.' });
  } catch (err) {
    st.toast({ kind: 'error', title: 'Could not make it the golden model', body: (err as Error).message });
  }
}

function RowMenu({ asset: a, golden, onRename }: { asset: Asset; golden: boolean; onRename: () => void }) {
  const formats = useStore(s => s.params?.formats);
  const projects = useStore(s => s.projects);
  const projectId = useStore(s => s.projectId);
  const [confirm, setConfirm] = useState(false);
  const others = projects.filter(p => !p.virtual && p.id !== a.project);
  const ck = checkKind(a);
  const canBeGolden = a.kind === 'mesh' && !ck && !golden && !!projectParam(projectId) && projectsSupported();
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
  const fmt = a.kind === 'mesh' ? 'stl' : (formats?.pointcloud?.[0] ?? 'ply');
  return (
    <Popover
      align="end"
      trigger={({ toggle, open }) => (
        <IconButton
          size="sm"
          label={`More for ${displayName(a)}: rename, download, delete…`}
          className="mp-more"
          active={open}
          onClick={() => {
            setConfirm(false);
            toggle();
          }}
          data-guide="model.menu"
        >
          <MoreHorizontal size={16} />
        </IconButton>
      )}
    >
      {({ close }) =>
        confirm ? (
          <ConfirmDelete
            title={`Delete “${displayName(a)}”?`}
            body="Its files are removed from the workspace. Models made from it stay."
            action="Delete"
            onConfirm={async () => {
              await deleteModel(a);
              close();
            }}
            onCancel={() => setConfirm(false)}
          />
        ) : (
          <div className="menu">
            {ck === 'golden' && (
              <button type="button" className="menu-item" onClick={() => { close(); openCheck(a.id); }}>
                <BadgeCheck size={15} />
                <div>
                  <div className="menu-item-title">Open the result</div>
                  <div className="menu-item-sub">Measure → Golden model</div>
                </div>
              </button>
            )}
            {canBeGolden && (
              <button type="button" className="menu-item" data-guide="model.make-golden" onClick={() => { close(); makeGolden(a); }}>
                <Star size={15} />
                <div>
                  <div className="menu-item-title">Make it the golden model</div>
                  <div className="menu-item-sub">The part as it should be: checks compare scans with it</div>
                </div>
              </button>
            )}
            <button type="button" className="menu-item" onClick={() => { close(); onRename(); }}><Pencil size={15} /><div className="menu-item-title">Rename <span className="mp-key">F2</span></div></button>
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
            <button type="button" className="menu-item danger-text" onClick={() => setConfirm(true)}><Trash2 size={15} /><div className="menu-item-title">Delete…</div></button>
          </div>
        )
      }
    </Popover>
  );
}
