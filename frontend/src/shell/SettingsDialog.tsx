import { lazy, Suspense, useEffect, useState } from 'react';
import { Bot, Cpu, Info, Keyboard, MonitorCog, Moon, Palette, ScanLine, Sun, UserRound, Users } from 'lucide-react';
import { useAuth } from '../lib/auth';
import { AccountSection, UsersSection } from './AccountSettings';
import { api } from '../lib/api';
import { useStore } from '../store';
import { refreshAssistantStatus, useAssistant } from '../features/assistant/assistantStore';
import { Button, Dialog, Field, NumberInput, Segmented, Select, Switch, TextInput } from '../ui/primitives';
import { ChoiceCards } from '../steps/StepFrame';
import { startTour } from './Tour';

const AutopilotPanel = lazy(() => import('../features/autopilot/AutopilotPanel').then(m => ({ default: m.AutopilotPanel })));
const DevicesSection = lazy(() => import('../steps/capture/DevicesSection').then(m => ({ default: m.DevicesSection })));

const SECTIONS = [
  { id: 'appearance', label: 'Appearance', icon: Palette },
  { id: 'devices', label: 'Scanner & turntable', icon: ScanLine },
  { id: 'assistant', label: 'Assistant', icon: Bot },
  { id: 'automations', label: 'Automations', icon: Cpu },
  { id: 'about', label: 'Shortcuts & about', icon: Info },
];

export function SettingsDialog() {
  const open = useStore(s => s.settingsOpen);
  const { enabled, user } = useAuth();
  if (!open) return null;
  const sections = [
    ...SECTIONS,
    ...(enabled && user ? [{ id: 'account', label: 'Account', icon: UserRound }] : []),
    ...(enabled && user?.admin ? [{ id: 'users', label: 'Users', icon: Users }] : []),
  ];
  const section = sections.some(s => s.id === open) ? open : 'appearance';
  const close = () => useStore.getState().set({ settingsOpen: null });
  return (
    <Dialog title="Settings" onClose={close} icon={<MonitorCog size={20} />}>
      <div className="settings">
        <nav className="settings-nav" aria-label="Settings sections">
          {sections.map(s => (
            <button key={s.id} type="button" className={`settings-tab ${s.id === section ? 'is-on' : ''}`} onClick={() => useStore.getState().set({ settingsOpen: s.id })} data-guide={s.id === 'about' ? 'settings.about' : `settings.${s.id}`}>
              <s.icon size={17} aria-hidden /> {s.label}
            </button>
          ))}
        </nav>
        <div className="settings-body">
          <Suspense fallback={<p className="caption">Loading…</p>}>
            {section === 'appearance' && <Appearance />}
            {section === 'devices' && <DevicesSection />}
            {section === 'assistant' && <AssistantSettings />}
            {section === 'automations' && <AutopilotPanel />}
            {section === 'about' && <About />}
            {section === 'account' && <AccountSection />}
            {section === 'users' && <UsersSection />}
          </Suspense>
        </div>
      </div>
    </Dialog>
  );
}

function Appearance() {
  const theme = useStore(s => s.theme);
  const textSize = useStore(s => s.textSize);
  const display = useStore(s => s.display);
  const set = useStore(s => s.set);
  return (
    <div className="stack loose">
      <div className="settings-head">
        <h3>Appearance</h3>
        <p>How CloudClean looks. Changes apply immediately and are remembered on this computer.</p>
      </div>
      <Field label="Theme" inline={false}>
        <ChoiceCards
          columns={3}
          value={theme}
          onChange={t => set({ theme: t })}
          options={[
            { value: 'paper', title: 'Paper', body: 'Light, warm and calm. Best in bright rooms.', icon: <Sun size={18} /> },
            { value: 'carbon', title: 'Carbon', body: 'Dark, for dim labs and long sessions.', icon: <Moon size={18} /> },
            { value: 'system', title: 'Match system', body: 'Follow the computer’s light / dark setting.', icon: <MonitorCog size={18} /> },
          ]}
        />
      </Field>
      <Field label="Text size" help="Larger text makes every label and number easier to read.">
        <Segmented value={textSize} onChange={v => set({ textSize: v })} options={[{ value: 'normal', label: 'Normal' }, { value: 'large', label: 'Larger' }]} />
      </Field>
      <Field label="Units shown" help="Labels only — scan data is never converted.">
        <Select value={display.units} onChange={units => useStore.getState().setDisplay({ units })} options={['mm', 'cm', 'm', 'in'].map(u => ({ value: u, label: u }))} />
      </Field>
      <Field label="Part size readout" help="Part axes = length × width × height along the part itself (what calipers measure). World = along the scanner’s X, Y, Z.">
        <Segmented value={display.sizeFrame} onChange={sizeFrame => useStore.getState().setDisplay({ sizeFrame })} options={[{ value: 'part', label: 'Part axes' }, { value: 'world', label: 'World XYZ' }]} />
      </Field>
      <Field label="Turn the view around" help="Model centre behaves like Revo Metro; Cursor turns around the point you press on.">
        <Segmented value={display.rotatePivot} onChange={rotatePivot => useStore.getState().setDisplay({ rotatePivot })} options={[{ value: 'center', label: 'Model centre' }, { value: 'cursor', label: 'Cursor' }]} />
      </Field>
    </div>
  );
}

interface LlmSettings {
  base_url: string;
  model: string;
  api_key_set: boolean;
  vision: boolean;
  thinking: boolean;
  temperature: number;
  max_steps: number;
  request_timeout: number;
}

function AssistantSettings() {
  const status = useAssistant(s => s.status);
  const [s, setS] = useState<LlmSettings | null>(null);
  const [key, setKey] = useState('');
  useEffect(() => {
    api.get<LlmSettings>('/api/assistant/settings').then(setS).catch(() => undefined);
    refreshAssistantStatus();
  }, []);
  const save = async () => {
    if (!s) return;
    try {
      const body: Record<string, unknown> = { base_url: s.base_url, model: s.model, vision: s.vision, thinking: !!s.thinking, temperature: s.temperature, max_steps: s.max_steps, request_timeout: s.request_timeout };
      if (key) body.api_key = key;
      setS(await api.put<LlmSettings>('/api/assistant/settings', body));
      setKey('');
      await refreshAssistantStatus();
      useStore.getState().toast({ kind: 'ok', title: 'Assistant settings saved' });
    } catch (err) {
      useStore.getState().toast({ kind: 'error', title: 'Could not save', body: (err as Error).message });
    }
  };
  return (
    <div className="stack loose">
      <div className="settings-head">
        <h3>Assistant</h3>
        <p>The assistant runs on your own LLM server (for example vLLM on the DGX Spark) and uses the same tools as the app. Nothing leaves your network.</p>
      </div>
      <div className={`status-line ${status?.reachable ? 'ok' : 'off'}`}>
        <span className={`dot ${status?.reachable ? (status.error ? 'warn' : 'ok') : 'danger'}`} />
        {status ? (status.reachable ? `Connected · ${status.model || 'no model loaded'}` : `Not reachable${status.error ? ` · ${status.error}` : ''}`) : 'Checking…'}
      </div>
      {s ? (
        <div className="fields">
          <Field label="Server address" inline={false} help="Any OpenAI-compatible /v1 endpoint: vLLM, SGLang, Ollama, llama.cpp, NIM.">
            <TextInput mono value={s.base_url} onChange={base_url => setS({ ...s, base_url })} placeholder="http://localhost:8000/v1" />
          </Field>
          <Field label="Model" inline={false}>
            {status?.models.length ? (
              <Select value={s.model} onChange={model => setS({ ...s, model })} options={[{ value: '', label: 'First model the server offers' }, ...status.models.map(m => ({ value: m, label: m }))]} />
            ) : (
              <TextInput mono value={s.model} onChange={model => setS({ ...s, model })} placeholder="(first model the server offers)" />
            )}
          </Field>
          <Field label="API key" inline={false} help={s.api_key_set ? 'A key is stored. Type to replace it.' : 'Only if your server requires one.'}>
            <input className="input mono" type="password" value={key} onChange={e => setKey(e.target.value)} placeholder={s.api_key_set ? '••••••••' : ''} autoComplete="off" />
          </Field>
          <Field label="Let it see pictures" help="Screenshots and photos you attach are sent to the model (it must support images)."><Switch checked={s.vision} onChange={vision => setS({ ...s, vision })} /></Field>
          <Field label="Think before acting" help="Better on tricky requests, but much slower on a DGX Spark (minutes instead of seconds)."><Switch checked={!!s.thinking} onChange={thinking => setS({ ...s, thinking })} /></Field>
          <Field label="Creativity (temperature)"><NumberInput value={s.temperature} step={0.1} min={0} max={2} onChange={temperature => setS({ ...s, temperature })} /></Field>
          <Field label="Most tool steps per answer"><NumberInput value={s.max_steps} step={1} min={1} max={50} onChange={max_steps => setS({ ...s, max_steps })} /></Field>
          <div className="row"><Button variant="primary" onClick={save}>Save and reconnect</Button></div>
        </div>
      ) : (
        <p className="caption">Loading…</p>
      )}
    </div>
  );
}

const KEYS: [string, string][] = [
  ['Ctrl K', 'Ask CloudClean — type what you want, or where to find something'],
  ['Ctrl J', 'Assistant'],
  ['Alt 1 … 6', 'Journey steps: Scan · Clean · Align · Mesh · Measure · Export'],
  ['F', 'Fit the view'],
  ['1 · 3 · 7 · 0', 'Front · right · top · isometric (Ctrl for the opposite side)'],
  ['5', 'Perspective / orthographic'],
  ['V · B · L', 'Move the view · box select · lasso select'],
  ['M', 'Measure between two points'],
  ['S · P', 'Smoothing brush · pick the point to turn around'],
  ['Del', 'Delete the selected points (makes a new model)'],
  ['Ctrl Z · Ctrl Y', 'Undo / redo: back to the model this one was made from, and forward again'],
  ['Esc', 'Cancel the tool or selection'],
  ['Ctrl B · Ctrl I · Ctrl .', 'Model list · side panel · hide both'],
  ['`', 'Jobs and logs'],
];

interface SystemInfo {
  host: string;
  os: string;
  cpu_count: number;
  memory: { total_gb: number | null; available_gb: number | null };
  disk: { path: string; total_gb: number; free_gb: number; low: boolean };
  gpus: { name: string; utilization_pct: number | null; temperature_c: number | null }[];
}

function ServerCard() {
  const [info, setInfo] = useState<SystemInfo | null | 'none'>(null);
  useEffect(() => {
    api.get<SystemInfo>('/api/system').then(setInfo).catch(() => setInfo('none'));
  }, []);
  if (info === 'none') return null;
  if (!info) return <p className="caption">Checking the server…</p>;
  return (
    <div className="card sunken stack tight">
      <div className="row spread"><span className="strong">This server · {info.host}</span><span className="caption">{info.os}</span></div>
      <div className="mono small">{info.cpu_count} CPU cores · {info.memory.total_gb ?? '?'} GB memory ({info.memory.available_gb ?? '?'} GB free)</div>
      {info.gpus.map((g, i) => <div key={i} className="mono small">{g.name}{g.utilization_pct != null ? ` · ${g.utilization_pct}% busy` : ''}{g.temperature_c != null ? ` · ${g.temperature_c} °C` : ''}</div>)}
      <div className={`small ${info.disk.low ? 'warn-text' : 'mono'}`}>{info.disk.free_gb} GB free of {info.disk.total_gb} GB for scans{info.disk.low ? ' — running low: delete old models or move the workspace' : ''}</div>
    </div>
  );
}

function About() {
  const params = useStore(s => s.params);
  return (
    <div className="stack loose">
      <div className="settings-head">
        <h3>Shortcuts & about</h3>
        <p>CloudClean runs on your own machine. Every operation makes a new model and reports what it changed, so the original scan is never lost.</p>
      </div>
      <div className="stack tight">
        <div className="row"><Keyboard size={16} aria-hidden /><span className="strong">Keyboard</span></div>
        <table className="data-table">
          <tbody>
            {KEYS.map(([k, v]) => (
              <tr key={k}><td style={{ width: 170 }}><span className="kbd" style={{ padding: '0 7px' }}>{k}</span></td><td>{v}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <ServerCard />
      <div className="row">
        <Button data-guide="tour" onClick={() => { useStore.getState().set({ settingsOpen: null, screen: 'workspace' }); setTimeout(startTour, 300); }}>Show the guided tour again</Button>
      </div>
      <table className="kv">
        <tbody>
          <tr><td>Workspace on the server</td><td className="mono">{params?.workspace ?? '–'}</td></tr>
          <tr><td>API for other programs</td><td><a className="link" href="/docs" target="_blank" rel="noreferrer">/docs</a></td></tr>
        </tbody>
      </table>
    </div>
  );
}
