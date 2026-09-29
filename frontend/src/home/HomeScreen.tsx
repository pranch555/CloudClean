import { useEffect, useMemo, useState } from 'react';
import { setTab as setMeasureTab } from '../steps/measure/state';
import { ArrowRight, ArrowUp, FolderOpen, FolderPlus, Gauge, HardDrive, RotateCw, ScanLine, Upload } from 'lucide-react';
import { api } from '../lib/api';
import { fmtAgo } from '../lib/format';
import { journeyStatus, STEPS } from '../lib/journey';
import { createProject, projectsSupported } from '../lib/projects';
import { uploadFiles } from '../lib/importing';
import type { Asset, Project } from '../lib/types';
import { useThumbs } from '../lib/thumbnails';
import { projectAssets, useStore } from '../store';
import { refreshAssistantStatus, sendToAssistant, useAssistant } from '../features/assistant/assistantStore';
import { AttachButton, AttachmentStrip, dropImages, pasteImages } from '../features/assistant/AttachControls';
import { AssistantGlyph } from '../features/assistant/AssistantMark';
import { Logo } from '../ui/icons';
import { Button } from '../ui/primitives';

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? 'Working late.' : h < 12 ? 'Good morning.' : h < 18 ? 'Good afternoon.' : 'Good evening.';
}

export function HomeScreen() {
  const projects = useStore(s => s.projects);
  const assets = useStore(s => s.assets);
  const [ask, setAsk] = useState('');
  const attached = useAssistant(s => s.attachments.length);
  const [naming, setNaming] = useState<null | 'scan' | 'files'>(null);
  const [name, setName] = useState('');

  const start = async (kind: 'scan' | 'files', partName: string) => {
    const st = useStore.getState();
    let projectId = st.projectId;
    if (projectsSupported() && partName.trim()) {
      try {
        const p = await createProject(partName.trim());
        await st.refreshProjects();
        if (p) projectId = p.id;
      } catch (err) {
        st.toast({ kind: 'error', title: 'Could not create the project', body: (err as Error).message });
        return;
      }
    }
    if (projectId) useStore.getState().openProject(projectId, kind === 'scan' ? 'capture' : 'clean');
    setNaming(null);
    setName('');
    if (kind === 'files') {
      const input = document.createElement('input');
      input.type = 'file';
      input.multiple = true;
      input.onchange = () => input.files?.length && uploadFiles([...input.files]);
      input.click();
    }
  };

  const submitAsk = () => {
    const text = ask.trim();
    if (!text && !useAssistant.getState().attachments.length) return;
    const st = useStore.getState();
    if (st.projectId) st.openProject(st.projectId);
    else st.set({ screen: 'workspace' });
    st.set({ rightTab: 'assistant' });
    st.setLayout({ rightOpen: true });
    setAsk('');
    sendToAssistant(text);
  };

  return (
    <main className="home" aria-label="Home">
      <div className="home-inner">
        <section className="home-hero">
          <div className="home-copy">
            <p className="home-kicker">
              <span className="dot live" aria-hidden /> CloudClean · scan, clean, measure — on your own hardware
            </p>
            <h1 className="home-title display">
              {greeting()}
              <br />
              <em>What are we measuring today?</em>
            </h1>
            <form className="home-ask" onSubmit={e => { e.preventDefault(); submitAsk(); }} data-own-drop {...dropImages}>
              <AttachmentStrip small />
              <div className="home-ask-row">
                <AssistantGlyph size={20} />
                <input value={ask} onChange={e => setAsk(e.target.value)} onPaste={pasteImages} placeholder="Tell CloudClean what to do — e.g. “measure the thread pitch”" aria-label="Ask CloudClean" />
                <AttachButton compact view={false} />
                <button type="submit" className="ask-send" disabled={!ask.trim() && !attached} aria-label="Send"><ArrowUp size={18} /></button>
              </div>
            </form>
          </div>
          <BoltDrawing />
        </section>

        <section className="home-actions" aria-label="Start">
          <ActionCard
            icon={<ScanLine size={24} />}
            title="Scan a part"
            body="MetroY and turntable, with live coverage guidance."
            open={naming === 'scan'}
            onOpen={() => { setNaming('scan'); setName(''); }}
            name={name}
            setName={setName}
            onGo={() => start('scan', name)}
            cta="Start scanning"
          />
          <ActionCard
            icon={<Upload size={24} />}
            title="Open scan files"
            body="PLY, STL, OBJ, STEP — Revo Metro exports too."
            open={naming === 'files'}
            onOpen={() => { setNaming('files'); setName(''); }}
            name={name}
            setName={setName}
            onGo={() => start('files', name)}
            cta="Choose files"
          />
          <button type="button" className="action-card" onClick={() => { const st = useStore.getState(); setMeasureTab('cad'); if (st.projectId) st.openProject(st.projectId, 'measure'); }}>
            <span className="action-icon"><Gauge size={24} /></span>
            <span className="action-title">Check against the golden model</span>
            <span className="action-body">What was not scanned, what to scan again, and every size: matches or off.</span>
            <span className="action-go">Open the check <ArrowRight size={16} /></span>
          </button>
        </section>

        <section className="home-projects" aria-label="Projects">
          <div className="home-section-head">
            <h2>Projects</h2>
            <span className="count-chip">{projects.length}</span>
            <span className="spacer" />
            {projectsSupported() && (
              <Button size="sm" icon={<FolderPlus size={15} />} onClick={() => { setNaming('scan'); setName(''); window.scrollTo({ top: 0, behavior: 'smooth' }); }}>New project</Button>
            )}
          </div>
          <div className="project-grid">
            {projects.map(p => <ProjectCard key={p.id} project={p} assets={projectAssets({ assets, projects, projectId: p.id })} />)}
            {!projects.length && <div className="caption">No projects yet — scan a part or open a file to start one.</div>}
          </div>
        </section>

        <StatusStrip />
      </div>
    </main>
  );
}

function ActionCard({ icon, title, body, open, onOpen, name, setName, onGo, cta }: { icon: React.ReactNode; title: string; body: string; open: boolean; onOpen: () => void; name: string; setName: (v: string) => void; onGo: () => void; cta: string }) {
  return (
    <div className={`action-card ${open ? 'is-open' : ''}`} role={open ? undefined : 'button'} tabIndex={open ? -1 : 0} onClick={() => !open && onOpen()} onKeyDown={e => !open && (e.key === 'Enter' || e.key === ' ') && onOpen()}>
      <span className="action-icon">{icon}</span>
      <span className="action-title">{title}</span>
      <span className="action-body">{body}</span>
      {open ? (
        <form className="action-form" onSubmit={e => { e.preventDefault(); onGo(); }} onClick={e => e.stopPropagation()}>
          <input className="input" autoFocus value={name} placeholder={projectsSupported() ? 'Name the part, e.g. “M20 bolt”' : 'Part name (optional)'} onChange={e => setName(e.target.value)} />
          <Button variant="primary" type="submit">{cta} <ArrowRight size={16} /></Button>
        </form>
      ) : (
        <span className="action-go">{cta} <ArrowRight size={16} /></span>
      )}
    </div>
  );
}

function ProjectCard({ project: p, assets }: { project: Project; assets: Asset[] }) {
  const openProject = useStore(s => s.openProject);
  const exported = useStore(s => !!s.exported[p.id]);
  const local = useThumbs(s => s.urls);
  const geometry = assets.filter(a => a.kind !== 'image');
  const cover = useMemo(() => {
    const byId = new Map(geometry.map(a => [a.id, a]));
    const preferred = p.cover_asset_id ? byId.get(p.cover_asset_id) : undefined;
    const withThumb = [...geometry].reverse().find(a => a.has_thumbnail || local[a.id]);
    return preferred && (preferred.has_thumbnail || local[preferred.id]) ? preferred : withThumb;
  }, [geometry, p.cover_asset_id, local]);
  const status = journeyStatus(assets, exported, false);
  const done = STEPS.filter(s => status[s.id] !== 'todo').length;

  return (
    <button type="button" className="project-card" onClick={() => openProject(p.id)}>
      <span className="project-cover">
        {cover ? <img src={local[cover.id] ?? `/api/assets/${cover.id}/thumbnail`} alt="" /> : <Logo size={40} />}
      </span>
      <span className="project-body">
        <span className="project-name">{p.name}</span>
        <span className="project-meta">{geometry.length} model{geometry.length === 1 ? '' : 's'} · {fmtAgo(p.updated)}</span>
        <span className="project-progress" aria-label={`${done} of ${STEPS.length} steps done`}>
          {STEPS.map(s => <span key={s.id} className={`pip ${status[s.id] === 'done' ? 'is-done' : status[s.id] === 'skipped' ? 'is-skipped' : ''}`} title={`${s.label}: ${status[s.id]}`} />)}
          <span className="caption">{done}/{STEPS.length}</span>
        </span>
      </span>
    </button>
  );
}

interface DriverInfo { id: string; name: string; available: boolean }

function StatusStrip() {
  const params = useStore(s => s.params);
  const assistant = useAssistant(s => s.status);
  const [drivers, setDrivers] = useState<DriverInfo[] | null>(null);
  const [capture, setCapture] = useState<{ active: boolean; state: string; driver_name?: string } | null>(null);
  const [turntable, setTurntable] = useState<{ connected: boolean; name?: string; kind?: string | null; validated?: boolean } | null>(null);

  useEffect(() => {
    refreshAssistantStatus();
    api.get<{ drivers: DriverInfo[] }>('/api/capture/drivers').then(d => setDrivers(d.drivers)).catch(() => setDrivers([]));
    api.get<{ active: boolean; state: string; driver_name?: string }>('/api/capture/status').then(setCapture).catch(() => setCapture(null));
    api.get<{ connected: boolean; name?: string; kind?: string | null; validated?: boolean }>('/api/turntable/status').then(setTurntable).catch(() => setTurntable(null));
  }, []);

  const metroy = drivers?.find(d => d.id === 'metroy_usb');
  const scannerText = capture?.active ? `${capture.driver_name} · ${capture.state}` : metroy ? (metroy.available ? 'MetroY found on USB — ready to connect' : 'MetroY not plugged into this computer') : 'checking…';
  return (
    <section className="status-strip" aria-label="System status">
      <StatusItem icon={<ScanLine size={16} />} label="Scanner" text={scannerText} tone={capture?.active || metroy?.available ? 'ok' : metroy ? 'warn' : undefined} onClick={() => useStore.getState().set({ settingsOpen: 'devices' })} />
      <StatusItem icon={<RotateCw size={16} />} label="Turntable" text={turntable ? (turntable.connected ? `${turntable.name ?? turntable.kind}${turntable.validated === false ? ' · protocol not yet confirmed' : ''}` : 'not connected') : 'not available yet'} tone={turntable?.connected ? 'ok' : undefined} onClick={() => useStore.getState().set({ settingsOpen: 'devices' })} />
      <StatusItem icon={<AssistantGlyph size={16} />} label="Assistant" text={assistant ? (assistant.reachable ? assistant.model || 'no model loaded' : 'LLM server offline') : 'checking…'} tone={assistant?.reachable ? (assistant.error ? 'warn' : 'ok') : assistant ? 'danger' : undefined} onClick={() => useStore.getState().set({ settingsOpen: 'assistant' })} />
      <StatusItem icon={<HardDrive size={16} />} label="Workspace" text={params?.workspace ?? '…'} mono onClick={() => useStore.getState().set({ settingsOpen: 'about' })} />
      <StatusItem icon={<FolderOpen size={16} />} label="Automations" text="Watch a folder, process scans hands-free" onClick={() => useStore.getState().set({ settingsOpen: 'automations' })} />
    </section>
  );
}

function StatusItem({ icon, label, text, tone, mono, onClick }: { icon: React.ReactNode; label: string; text: string; tone?: 'ok' | 'warn' | 'danger'; mono?: boolean; onClick: () => void }) {
  return (
    <button type="button" className="status-item" onClick={onClick}>
      <span className="status-icon">{icon}</span>
      <span className="grow">
        <span className="status-label">{label} {tone && <span className={`dot ${tone}`} aria-hidden />}</span>
        <span className={`status-text truncate ${mono ? 'mono' : ''}`} title={text}>{text}</span>
      </span>
    </button>
  );
}

/** Signature illustration: a dimensioned side view of a hex bolt, drawn like an engineering sheet. */
function BoltDrawing() {
  return (
    <svg className="bolt-drawing" viewBox="0 0 460 300" role="img" aria-label="Illustration: a nominal M20 bolt drawn with dimension lines">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 1 9 5 0 9z" fill="currentColor" />
        </marker>
        <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(-62)">
          <line x1="0" y1="0" x2="0" y2="7" stroke="currentColor" strokeWidth="1.1" opacity="0.55" />
        </pattern>
      </defs>
      <g className="bd-ink" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round">
        {/* head */}
        <path d="M60 92 h56 v116 h-56 z" />
        <path d="M60 130 h56 M60 170 h56" opacity="0.5" />
        <path d="M60 92 q-8 58 0 116" opacity="0.4" />
        {/* shank + thread */}
        <path d="M116 118 h92 v64 h-92" />
        <rect x="208" y="112" width="178" height="76" fill="url(#hatch)" />
        <path d="M208 112 h178 l10 10 v56 l-10 10 h-178" />
        {/* centre line */}
        <path d="M40 150 h380" strokeDasharray="14 4 3 4" strokeWidth="1" opacity="0.55" />
      </g>
      <g className="bd-dim" fill="none" stroke="currentColor" strokeWidth="1.1">
        {/* overall length */}
        <path d="M60 236 v34 M396 236 v34" opacity="0.6" />
        <path d="M60 262 H396" markerStart="url(#arr)" markerEnd="url(#arr)" />
        {/* thread diameter */}
        <path d="M396 112 h40 M396 188 h40" opacity="0.6" />
        <path d="M428 112 V188" markerStart="url(#arr)" markerEnd="url(#arr)" />
        {/* head across flats */}
        <path d="M60 92 v-40 M116 92 v-40" opacity="0.6" />
        <path d="M60 60 H116" markerStart="url(#arr)" markerEnd="url(#arr)" />
        {/* pitch */}
        <path d="M300 112 v-34 M318 112 v-34" opacity="0.6" />
        <path d="M300 84 H318" markerStart="url(#arr)" markerEnd="url(#arr)" />
      </g>
      <g className="bd-text" fontFamily="var(--font-mono)" fontSize="13">
        <rect x="190" y="251" width="78" height="22" rx="11" className="bd-chip" />
        <text x="229" y="266" textAnchor="middle">100.00</text>
        <rect x="401" y="139" width="56" height="22" rx="11" className="bd-chip" />
        <text x="429" y="154" textAnchor="middle">M20</text>
        <rect x="58" y="30" width="60" height="22" rx="11" className="bd-chip" />
        <text x="88" y="45" textAnchor="middle">30 AF</text>
        <rect x="282" y="52" width="56" height="22" rx="11" className="bd-chip" />
        <text x="310" y="67" textAnchor="middle">P 2.5</text>
      </g>
    </svg>
  );
}
