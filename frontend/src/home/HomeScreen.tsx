import { useEffect, useMemo, useState } from 'react';
import { setTab as setMeasureTab } from '../steps/measure/state';
import { ArrowRight, ArrowUp, FolderOpen, FolderPlus, Gauge, HardDrive, RotateCw, ScanLine, Upload } from 'lucide-react';
import { api } from '../lib/api';
import { fmtAgo, fmtCount } from '../lib/format';
import { journeyStatus, roleOf, stepInfo, STEPS, suggestedStep } from '../lib/journey';
import { createProject, projectsSupported } from '../lib/projects';
import { uploadFiles } from '../lib/importing';
import type { Asset, Project } from '../lib/types';
import { useThumbs } from '../lib/thumbnails';
import { enter, once, reducedMotion, stagger, timeline, useEntrance, utils, animate, EASE } from '../lib/motion';
import { projectAssets, useStore } from '../store';
import { refreshAssistantStatus, sendToAssistant, useAssistant } from '../features/assistant/assistantStore';
import { AttachButton, AttachmentStrip, dropImages, pasteImages } from '../features/assistant/AttachControls';
import { AssistantGlyph } from '../features/assistant/AssistantMark';
import { ScanDial, type DialReadout } from '../ui/motion/ScanDial';
import { Logo } from '../ui/icons';
import { Button } from '../ui/primitives';

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? 'Working late.' : h < 12 ? 'Good morning.' : h < 18 ? 'Good afternoon.' : 'Good evening.';
}

/** Words of a headline, each in a clipping box so it can rise into place (rendered by React, animated by anime). */
function Words({ text }: { text: string }) {
  return (
    <>
      {text.split(' ').map((w, i) => (
        <span key={i}>
          {i > 0 && ' '}
          <span className="w">
            <span>{w}</span>
          </span>
        </span>
      ))}
    </>
  );
}

const thumbUrl = (a: Asset, local: Record<string, string>) => local[a.id] ?? `/api/assets/${a.id}/thumbnail`;

/** The model that stands for a project: its cover, else the newest model with a picture. */
function coverOf(p: Project, geometry: Asset[], local: Record<string, string>): Asset | undefined {
  const byId = new Map(geometry.map(a => [a.id, a]));
  const preferred = p.cover_asset_id ? byId.get(p.cover_asset_id) : undefined;
  const withThumb = [...geometry].reverse().find(a => a.has_thumbnail || local[a.id]);
  return preferred && (preferred.has_thumbnail || local[preferred.id]) ? preferred : withThumb;
}

/** "12 MIN AGO", "5 H AGO", "2 DAYS AGO", "OCT 05": short enough for an instrument readout. */
function readoutAgo(iso: string | undefined): string {
  if (!iso) return '—';
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return 'NOW';
  if (s < 3600) return `${Math.floor(s / 60)} MIN AGO`;
  if (s < 86400) return `${Math.floor(s / 3600)} H AGO`;
  if (s < 86400 * 7) {
    const d = Math.floor(s / 86400);
    return `${d} DAY${d === 1 ? '' : 'S'} AGO`;
  }
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: '2-digit' }).toUpperCase();
}

export function HomeScreen() {
  const projects = useStore(s => s.projects);
  const assets = useStore(s => s.assets);
  const exported = useStore(s => s.exported);
  const local = useThumbs(s => s.urls);
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

  // ---- the dial: the project worked on last, and four readouts about all of them
  const dial = useMemo(() => {
    const latest = [...projects].sort((x, y) => y.updated.localeCompare(x.updated))[0];
    const geometry = assets.filter(a => a.kind !== 'image');
    const points = geometry.reduce((n, a) => n + (a.kind === 'pointcloud' ? (a.stats?.points ?? 0) : 0), 0);
    const checks = geometry.filter(a => roleOf(a) === 'inspection').length;
    const lastWork = geometry.reduce<string | undefined>((m, a) => (!m || a.created > m ? a.created : m), undefined);
    const readouts: DialReadout[] = [
      { label: 'Models', value: String(geometry.length).padStart(2, '0') },
      { label: 'Checks', value: String(checks).padStart(2, '0') },
      { label: 'Points', value: points ? fmtCount(points) : '0' },
      { label: 'Last change', value: readoutAgo(lastWork) },
    ];
    if (!latest) return { latest: null, status: null, next: null, cover: null, readouts };
    const own = projectAssets({ assets, projects, projectId: latest.id });
    const status = journeyStatus(own, !!exported[latest.id], false);
    const next = suggestedStep(status);
    const c = coverOf(latest, own.filter(a => a.kind !== 'image'), local);
    return { latest, status, next, cover: c ? thumbUrl(c, local) : null, readouts };
  }, [projects, assets, exported, local]);

  const openLatest = () => {
    if (dial.latest) useStore.getState().openProject(dial.latest.id, dial.next ?? undefined);
    else {
      setNaming('scan');
      setName('');
    }
  };

  // ---- arrival: the headline rises word by word, then the cards, the projects, the status strip
  const first = useMemo(() => once('home'), []);
  const root = useEntrance<HTMLElement>('home', el => {
    const words = el.querySelectorAll('.home-title .w > span');
    const grid = el.querySelector<HTMLElement>('.project-grid');
    const cards = el.querySelectorAll('.project-card');
    const cols = grid ? Math.max(1, getComputedStyle(grid).gridTemplateColumns.split(' ').length) : 1;
    if (reducedMotion()) {
      utils.set(words, { translateY: '0%' });
      return null;
    }
    const k = first ? 1 : 0.6; // later visits: the same moves, quicker
    const out = [
      enter(el.querySelectorAll('.home-kicker'), { y: 6, duration: 420 }),
      animate(words, { translateY: ['108%', '0%'], duration: 760 * k, delay: stagger(42 * k, { start: 60 * k }), ease: EASE.out }),
      enter(el.querySelectorAll('.home-ask'), { y: 12, delay: 300 * k, duration: 560 }),
      enter(el.querySelectorAll('.action-card'), { y: 18, step: 70 * k, delay: 380 * k, duration: 620, scale: 0.985 }),
      enter(el.querySelectorAll('.home-section-head'), { y: 8, delay: 520 * k }),
      cards.length
        ? animate(cards, {
            opacity: { from: 0, to: 1, duration: 420, ease: 'out(2)' },
            translateY: { from: 16, to: 0 },
            scale: { from: 0.97, to: 1 },
            duration: 640,
            ease: EASE.out,
            delay: stagger(70 * k, { grid: [cols, Math.ceil(cards.length / cols)], from: 'center', start: 560 * k }),
          })
        : null,
      enter(el.querySelectorAll('.status-item'), { y: 8, step: 45, delay: 680 * k }),
    ];
    return out;
  });

  const nextInfo = dial.next ? stepInfo(dial.next) : null;
  const caption = dial.latest ? (
    <>
      <span className="dial-caption-kicker">Continue</span>
      <span className="dial-caption-name truncate">{dial.latest.name}</span>
      {nextInfo && <span className="dial-caption-next">· next: {nextInfo.label}</span>}
    </>
  ) : (
    <>
      <span className="dial-caption-kicker">Start</span>
      <span className="dial-caption-name">your first scan</span>
    </>
  );

  return (
    <main className="home" aria-label="Home" ref={root}>
      <div className="home-inner">
        <section className="home-hero">
          <div className="home-copy">
            <p className="home-kicker">
              <span className="dot live" aria-hidden /> CloudClean · scan, clean, measure — on your own hardware
            </p>
            <h1 className="home-title display">
              <span className="home-title-line"><Words text={greeting()} /></span>
              <em className="home-title-line"><Words text="What are we measuring today?" /></em>
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
          <ScanDial
            status={dial.status}
            next={dial.next}
            cover={dial.cover}
            readouts={dial.readouts}
            caption={caption}
            onOpen={openLatest}
            label={dial.latest ? `Continue ${dial.latest.name}${nextInfo ? `: next step ${nextInfo.label}` : ''}` : 'Start your first scan'}
          />
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
            {projects.map((p, i) => <ProjectCard key={p.id} index={i} project={p} assets={projectAssets({ assets, projects, projectId: p.id })} />)}
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

/** A project card. Its picture is revealed by a scanner line sweeping across it, once it has loaded. */
function ProjectCard({ project: p, assets, index }: { project: Project; assets: Asset[]; index: number }) {
  const openProject = useStore(s => s.openProject);
  const exported = useStore(s => !!s.exported[p.id]);
  const local = useThumbs(s => s.urls);
  const geometry = assets.filter(a => a.kind !== 'image');
  const cover = useMemo(() => coverOf(p, geometry, local), [geometry, p, local]);
  const status = journeyStatus(assets, exported, false);
  const done = STEPS.filter(s => status[s.id] !== 'todo').length;

  const ref = useEntrance<HTMLButtonElement>(cover?.id ?? 'none', el => {
    const img = el.querySelector<HTMLImageElement>('.project-cover img');
    const line = el.querySelector<HTMLElement>('.project-scan');
    if (!img || !line || reducedMotion()) return null;
    const hidden = 'inset(0% 100% 0% 0%)';
    utils.set(img, { clipPath: hidden });
    const tl = timeline({ autoplay: false, delay: 620 + index * 90 });
    tl.add(line, { opacity: [0, 1], duration: 140, ease: 'out(2)' }, 0)
      .add(line, { left: ['0%', '100%'], duration: 820, ease: 'inOut(2)' }, 0)
      .add(img, { clipPath: [hidden, 'inset(0% 0% 0% 0%)'], duration: 820, ease: 'inOut(2)' }, 0)
      .add(line, { opacity: 0, duration: 220, ease: 'out(2)' }, 700);
    const go = () => tl.play();
    if (img.complete && img.naturalWidth) go();
    else {
      img.addEventListener('load', go, { once: true });
      img.addEventListener('error', () => tl.seek(tl.duration), { once: true });
    }
    return tl;
  });

  return (
    <button type="button" className="project-card" onClick={() => openProject(p.id)} ref={ref}>
      <span className="project-cover">
        {cover ? <img src={thumbUrl(cover, local)} alt="" /> : <Logo size={40} />}
        <span className="project-scan" aria-hidden />
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
