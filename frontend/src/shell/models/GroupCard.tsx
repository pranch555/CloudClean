import { useEffect, useState, type ReactNode } from 'react';
import { ChevronDown, ClipboardCheck, Eye, EyeClosed, Images, Layers, MoreHorizontal, ImagePlus, Trash2 } from 'lucide-react';
import { pickFiles } from '../../lib/importing';
import type { Asset } from '../../lib/types';
import { useStore } from '../../store';
import { ScanGlyph } from '../../ui/icons';
import { CadGlyph } from '../../steps/measure/glyphs';
import { IconButton, Popover } from '../../ui/primitives';
import { GROUPS, type GroupId } from './groups';
import { setCollapsed, usePanelUi } from './panelState';
import { ConfirmDelete, deletePhotos } from './photos';

const ICON: Record<GroupId, ReactNode> = {
  scans: <ScanGlyph size={18} />,
  made: <Layers size={17} strokeWidth={1.9} aria-hidden />,
  checks: <ClipboardCheck size={17} strokeWidth={1.9} aria-hidden />,
  cad: <CadGlyph size={18} />,
  photos: <Images size={17} strokeWidth={1.9} aria-hidden />,
};

/**
 * One kind of thing in the model list, as a card of its own colour: icon, plain title, count, what it holds in
 * one line; it folds shut, and "show all / hide all" puts the whole group in or out of the 3D view.
 */
export function GroupCard({ id, ids, count, photos, children }: { id: GroupId; ids: string[]; count: number; photos?: Asset[]; children: ReactNode }) {
  const info = GROUPS[id];
  const collapsed = usePanelUi(s => s.collapsed.includes(id));
  const reveal = usePanelUi(s => s.reveal);
  const activeId = useStore(s => s.activeId);
  const shown = useStore(s => s.visible.filter(v => ids.includes(v)).length);

  // the model being worked on, or one asked for ("Made from" link), opens its group
  useEffect(() => {
    if (collapsed && activeId && ids.includes(activeId)) setCollapsed(id, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);
  useEffect(() => {
    if (collapsed && reveal && ids.includes(reveal.id)) setCollapsed(id, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reveal?.n]);

  const allShown = ids.length > 0 && shown === ids.length;
  const showAll = () => {
    const vis = useStore.getState().visible;
    useStore.setState({ visible: allShown ? vis.filter(v => !ids.includes(v)) : [...vis, ...ids.filter(x => !vis.includes(x))] });
  };
  const bodyId = `mp-body-${id}`;

  return (
    <section className={`mp-group g-${id} ${collapsed ? 'is-collapsed' : ''}`} aria-label={info.title} data-guide={info.guide}>
      <header className="mp-head">
        <button type="button" className="mp-toggle" aria-expanded={!collapsed} aria-controls={bodyId} onClick={() => setCollapsed(id, !collapsed)} title={collapsed ? `Open ${info.title}` : `Fold ${info.title} away`}>
          <ChevronDown size={15} className="mp-chev" aria-hidden />
          <span className="mp-icon" aria-hidden>{ICON[id]}</span>
          <span className="mp-title">{info.title}</span>
          <span className="mp-count" aria-label={`${count} in this group`}>{count}</span>
        </button>
        {ids.length > 0 && (
          <GroupEye
            state={allShown ? 'all' : shown ? 'some' : 'none'}
            onClick={showAll}
            label={allShown ? `Hide all ${info.title} in 3D` : `Show all ${info.title} in 3D`}
            tip={allShown ? 'All shown in the 3D view · click to hide them all' : shown ? `${shown} of ${ids.length} shown in the 3D view · click to show them all` : 'Show all of these in the 3D view'}
          />
        )}
        {photos && photos.length > 0 && <PhotosMenu photos={photos} />}
      </header>
      {!collapsed && (
        <div className="mp-body" id={bodyId}>
          <p className="mp-blurb">{info.blurb}</p>
          {children}
        </div>
      )}
    </section>
  );
}

/** The eye of a whole group, in the same column as the eyes of its rows: show or hide them all. */
function GroupEye({ state, onClick, label, tip }: { state: 'all' | 'some' | 'none'; onClick: () => void; label: string; tip: string }) {
  return (
    <button type="button" className={`mp-eye mp-groupeye is-${state}`} onClick={onClick} aria-label={label} aria-pressed={state === 'all' ? true : state === 'some' ? 'mixed' : false} data-tip={tip} data-tip-side="left" data-guide="models.show-all">
      {state === 'none' ? <EyeClosed size={17} aria-hidden /> : <Eye size={17} strokeWidth={2} aria-hidden />}
    </button>
  );
}

function PhotosMenu({ photos }: { photos: Asset[] }) {
  return (
    <Popover
      align="end"
      trigger={({ toggle, open }) => (
        <IconButton size="sm" label="Photo options: add, delete all" className="mp-groupmenu" active={open} onClick={toggle}>
          <MoreHorizontal size={16} />
        </IconButton>
      )}
    >
      {({ close }) => <PhotosMenuBody photos={photos} close={close} />}
    </Popover>
  );
}

/** Add photos, or delete them all (after a "sure?"; the step is local to this open menu). */
function PhotosMenuBody({ photos, close }: { photos: Asset[]; close: () => void }) {
  const [confirm, setConfirm] = useState(false);
  const n = photos.length;
  if (confirm) {
    return (
      <ConfirmDelete
        title={`Delete all ${n} photo${n === 1 ? '' : 's'}?`}
        body="They are removed from this project. Models made or coloured from them stay."
        action={`Delete ${n} photo${n === 1 ? '' : 's'}`}
        onConfirm={async () => {
          await deletePhotos(photos);
          close();
        }}
        onCancel={() => setConfirm(false)}
      />
    );
  }
  return (
    <div className="menu">
      <button type="button" className="menu-item" onClick={() => { close(); pickFiles('image/*,.heic,.heif'); }}>
        <ImagePlus size={15} />
        <div>
          <div className="menu-item-title">Add photos…</div>
          <div className="menu-item-sub">JPG, PNG or phone photos (HEIC) of the real part</div>
        </div>
      </button>
      <div className="menu-sep" />
      <button type="button" className="menu-item danger-text" data-guide="photos.delete-all" onClick={() => setConfirm(true)}>
        <Trash2 size={15} />
        <div className="menu-item-title">Delete all {n} photos…</div>
      </button>
    </div>
  );
}
