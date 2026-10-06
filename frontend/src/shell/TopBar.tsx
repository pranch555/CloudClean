import { useState } from 'react';
import { ChevronDown, FolderPlus, Home, Moon, Settings, Sun, Terminal } from 'lucide-react';
import { lastLog } from '../lib/jobs';
import { createProject, projectsSupported } from '../lib/projects';
import { fmtAgo } from '../lib/format';
import { resolveTheme, switchTheme } from '../lib/theme';
import { useStore, panelShown } from '../store';
import { Logo } from '../ui/icons';
import { AssistantGlyph } from '../features/assistant/AssistantMark';
import { Button, IconButton, Popover } from '../ui/primitives';
import { Journey } from './Journey';
import { AccountChip } from './AccountSettings';

export function TopBar({ home = false }: { home?: boolean }) {
  const set = useStore(s => s.set);
  const goHome = useStore(s => s.goHome);
  const jobs = useStore(s => s.jobs);
  const jobsOpen = useStore(s => s.jobsOpen);
  const rightTab = useStore(s => s.rightTab);
  const rightShown = useStore(s => panelShown(s, 'right'));
  const theme = useStore(s => s.theme);
  const running = jobs.find(j => j.status === 'running');
  const queued = jobs.filter(j => j.status === 'queued').length;
  const failed = !running && jobs[0]?.status === 'failed';
  const dark = resolveTheme(theme) === 'carbon';
  const assistantOn = !home && rightTab === 'assistant' && rightShown;

  const openAssistant = () => {
    if (home) {
      set({ screen: 'workspace', rightTab: 'assistant' });
      useStore.getState().setLayout({ rightOpen: true });
      return;
    }
    if (assistantOn) set({ rightTab: 'step' });
    else {
      set({ rightTab: 'assistant' });
      useStore.getState().setLayout({ rightOpen: true });
    }
  };

  return (
    <header className="topbar">
      <div className="topbar-left">
        <button type="button" className="brand-btn" onClick={goHome} title="Home — all projects" data-guide="home">
          <Logo size={30} />
          <span className="brand-name">CloudClean</span>
        </button>
        {!home && (
          <>
            <span className="crumb-sep" aria-hidden>/</span>
            <ProjectSwitcher />
          </>
        )}
      </div>

      <div>{!home && <Journey />}</div>

      <div className="topbar-right">
        {(running || queued > 0 || failed || jobs.length > 0) && (
          <button type="button" data-guide="jobs" className={`pill-btn jobs-btn ${running || queued ? 'is-busy' : ''} ${failed ? 'is-failed' : ''} ${jobsOpen ? 'is-on' : ''}`} onClick={() => set({ jobsOpen: !jobsOpen })} title="Jobs and logs">
            {running ? <span className="ring" style={{ '--p': running.progress?.fraction ?? 0.25 } as React.CSSProperties} data-indeterminate={running.progress == null} aria-hidden /> : <Terminal size={15} aria-hidden />}
            <span className="job-text">{running ? `${running.title} · ${running.progress?.label || lastLog(running)}` : queued ? `${queued} waiting` : failed ? `${jobs[0].title} failed` : 'Jobs'}</span>
          </button>
        )}
        <IconButton className="theme-toggle" data-guide="theme" label={dark ? 'Switch to the light Paper theme' : 'Switch to the dark Carbon theme'} tip="bottom" onClick={e => switchTheme(dark ? 'paper' : 'carbon', e.currentTarget)}>
          <span key={dark ? 'sun' : 'moon'} className="theme-icon">{dark ? <Sun size={18} /> : <Moon size={18} />}</span>
        </IconButton>
        <IconButton label="Settings" tip="bottom" data-guide="settings" onClick={() => set({ settingsOpen: 'appearance' })}>
          <Settings size={18} />
        </IconButton>
        <AccountChip />
        <button type="button" data-guide="assistant" className={`pill-btn ask-btn ${assistantOn ? 'is-on' : ''}`} onClick={openAssistant} title="Assistant (Ctrl J)" aria-label="Ask CloudClean (Ctrl J)">
          <AssistantGlyph size={16} />
          <span className="ask-label">Ask CloudClean</span>
        </button>
      </div>
    </header>
  );
}

function ProjectSwitcher() {
  const projects = useStore(s => s.projects);
  const projectId = useStore(s => s.projectId);
  const openProject = useStore(s => s.openProject);
  const goHome = useStore(s => s.goHome);
  const current = projects.find(p => p.id === projectId);
  const [name, setName] = useState('');

  const create = async (close: () => void) => {
    const n = name.trim();
    if (!n) return;
    try {
      const p = await createProject(n);
      await useStore.getState().refreshProjects();
      if (p) openProject(p.id, 'capture');
      setName('');
      close();
    } catch (err) {
      useStore.getState().toast({ kind: 'error', title: 'Could not create the project', body: (err as Error).message });
    }
  };

  return (
    <Popover
      align="start"
      trigger={({ toggle, open }) => (
        <button type="button" className="project-switch" onClick={toggle} aria-expanded={open} title="Switch project" data-guide="projects">
          <span className="truncate">{current?.name ?? 'Project'}</span>
          <ChevronDown size={15} aria-hidden className="muted" />
        </button>
      )}
    >
      {({ close }: { close: () => void }) => (
        <div className="menu" style={{ minWidth: 'min(300px, calc(100vw - 40px))' }}>
          <div className="menu-label">Projects</div>
          {projects.map(p => (
            <button key={p.id} type="button" className={`menu-item ${p.id === projectId ? 'is-active' : ''}`} onClick={() => { openProject(p.id); close(); }}>
              <div className="grow">
                <div className="menu-item-title truncate">{p.name}</div>
                <div className="menu-item-sub">{p.counts?.total ?? 0} models · updated {fmtAgo(p.updated)}</div>
              </div>
            </button>
          ))}
          {projectsSupported() && (
            <>
              <div className="menu-sep" />
              <div className="stack tight" style={{ padding: '6px 8px 8px' }}>
                <div className="row">
                  <input className="input" style={{ height: 34 }} value={name} placeholder="New project name" onChange={e => setName(e.target.value)} onKeyDown={e => e.key === 'Enter' && create(close)} />
                  <Button size="sm" variant="primary" icon={<FolderPlus size={14} />} disabled={!name.trim()} onClick={() => create(close)}>Create</Button>
                </div>
              </div>
            </>
          )}
          <div className="menu-sep" />
          <button type="button" className="menu-item" onClick={() => { goHome(); close(); }}>
            <Home size={16} />
            <div className="menu-item-title">All projects</div>
          </button>
        </div>
      )}
    </Popover>
  );
}
