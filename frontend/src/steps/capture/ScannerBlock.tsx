import { useEffect, useMemo, useState } from 'react';
import { Box, CheckCircle2, ChevronDown, CircleSlash, Gem, Moon, RefreshCw, TriangleAlert, Unplug } from 'lucide-react';
import { fmtLen } from '../../lib/format';
import { Button, Field, NumberInput, Select, Switch, TextInput } from '../../ui/primitives';
import { connectCapture, loadDrivers, rememberedSettings, useCapture, type DriverInfo, type SettingSchema } from '../../features/capture/captureStore';
import { availability, driverMeta, PRIMARY_DRIVERS, reasonShort, sortDrivers } from '../../features/capture/vocabulary';
import { Block, ChoiceCards } from '../StepFrame';
import { ActionError, Callout, runAction, useCaptureAction } from './parts';

/** The scanner the user is setting up (shared by the Scanner block and the step's Connect button). */
export interface ScannerSetup {
  drivers: DriverInfo[];
  driver: DriverInfo | undefined;
  driverId: string;
  setDriverId: (id: string) => void;
  schema: SettingSchema[];
  values: Record<string, unknown>;
  setValues: (fn: (v: Record<string, unknown>) => Record<string, unknown>) => void;
  resetValues: () => void;
}

export function useScannerSetup(): ScannerSetup {
  const drivers = useCapture(s => s.drivers);
  const sessionSettings = useCapture(s => s.sessionSettings);
  const lastDriver = useCapture(s => s.lastDriver);
  const liveDriver = useCapture(s => (s.status?.active ? s.status.driver : undefined));
  const [driverId, setDriverId] = useState('');
  const [values, setValuesState] = useState<Record<string, unknown>>({});
  const sorted = useMemo(() => sortDrivers(drivers), [drivers]);

  // the live session's scanner, else the one used last (even when it is unplugged: say why), else the first ready one
  useEffect(() => {
    if (!sorted.length || (driverId && sorted.some(d => d.id === driverId))) return;
    const pick = sorted.find(d => d.id === liveDriver) ?? sorted.find(d => d.id === lastDriver) ?? sorted.find(d => d.available) ?? sorted[0];
    setDriverId(pick.id);
  }, [sorted, lastDriver, liveDriver]);

  const driver = sorted.find(d => d.id === driverId);
  const schema = useMemo(() => [...(driver?.settings ?? []), ...sessionSettings], [driver, sessionSettings]);

  const initial = () => {
    if (!driver) return {};
    const remembered = rememberedSettings(driver.id);
    const st = useCapture.getState().status;
    const live = st?.active && st.driver === driver.id ? st.settings ?? {} : {};
    return Object.fromEntries(schema.map(s => [s.key, s.key in live ? live[s.key] : s.key in remembered ? remembered[s.key] : s.default]));
  };

  // re-initialise only when the scanner or its list of settings changes, not when the driver list is merely reloaded
  const schemaKey = schema.map(s => s.key).join('|');
  useEffect(() => {
    setValuesState(initial());
  }, [driver?.id, schemaKey]);

  return {
    drivers: sorted,
    driver,
    driverId,
    setDriverId,
    schema,
    values,
    setValues: fn => setValuesState(fn),
    resetValues: () => setValuesState(Object.fromEntries(schema.map(s => [s.key, s.default]))),
  };
}

const SURFACES = [
  { match: ['general', 'normal'], title: 'Normal', body: 'Matte or light parts: plastic, cast metal, paint.', icon: Box },
  { match: ['dark'], title: 'Dark', body: 'Black or very dark parts. A longer exposure so the laser shows.', icon: Moon },
  { match: ['reflective', 'shiny'], title: 'Shiny', body: 'Polished or machined metal. A shorter exposure against glare.', icon: Gem },
];

export const surfaceWord = (value: unknown) => SURFACES.find(s => s.match.includes(String(value)))?.title;

function surfaceSetting(schema: SettingSchema[]) {
  return schema.find(s => s.key === 'surface' && s.type === 'select' && s.options?.length);
}

/** Picking the scanner and how to scan with it (no session yet, or "Change" on a connected one). */
export function ScannerChooser({ setup }: { setup: ScannerSetup }) {
  const { drivers, driver, driverId, setDriverId, schema, values, setValues } = setup;
  const driversError = useCapture(s => s.driversError);
  const [more, setMore] = useState(false);
  const primary = drivers.filter(d => PRIMARY_DRIVERS.has(d.id) || d.id === driverId);
  const others = drivers.filter(d => !primary.includes(d));
  const surface = surfaceSetting(schema);
  const ready = !!driver?.available;

  return (
    <>
      <Block title="Scanner" guide="capture.scanner">
        {driversError && !drivers.length ? (
          <Callout tone="danger" action={<Button size="sm" variant="secondary" icon={<RefreshCw size={14} />} onClick={() => loadDrivers()}>Try again</Button>}>
            <b>Cannot list the scanners.</b> {driversError}
          </Callout>
        ) : !drivers.length ? (
          <div className="cap-skeleton" aria-busy="true">
            <span />
            <span />
            <span />
          </div>
        ) : (
          <div className="choices cap-drivers" role="radiogroup" aria-label="Scanner">
            {primary.map(d => <DriverCard key={d.id} d={d} selected={d.id === driverId} onSelect={() => setDriverId(d.id)} />)}
            {more && others.map(d => <DriverCard key={d.id} d={d} selected={d.id === driverId} onSelect={() => setDriverId(d.id)} />)}
          </div>
        )}
        {others.length > 0 && (
          <button type="button" className="link small cap-more" onClick={() => setMore(m => !m)} aria-expanded={more}>
            {more ? 'Fewer scanners' : `More scanners and depth cameras (${others.length})`}
          </button>
        )}
        {driver && !driver.available && driver.reason && <NotReady d={driver} />}
      </Block>

      {ready && surface && (
        <Block title="What are you scanning?">
          <ChoiceCards
            value={String(values.surface ?? surface.default)}
            onChange={v => setValues(x => ({ ...x, surface: v }))}
            options={(surface.options ?? []).map(o => {
              const m = SURFACES.find(s => s.match.includes(String(o.value)));
              return { value: String(o.value), title: m?.title ?? o.label, body: m?.body ?? '', icon: m ? <m.icon size={18} /> : undefined };
            })}
          />
        </Block>
      )}

      {ready && schema.some(s => s !== surface) && <FineTune setup={setup} exclude={surface?.key} />}
    </>
  );
}

function NotReady({ d }: { d: DriverInfo }) {
  const [more, setMore] = useState(false);
  const short = reasonShort(d);
  return (
    <Callout tone="warn">
      <b>{driverMeta(d).title} is not ready.</b> {short ?? d.reason}
      {short && (
        <>
          {' '}
          <button type="button" className="link" aria-expanded={more} onClick={() => setMore(m => !m)}>
            {more ? 'Less' : 'Details'}
          </button>
          {more && <span className="cap-reason-full">{d.reason}</span>}
        </>
      )}
    </Callout>
  );
}

function DriverCard({ d, selected, onSelect }: { d: DriverInfo; selected: boolean; onSelect: () => void }) {
  const meta = driverMeta(d);
  const a = availability(d);
  const Icon = meta.icon;
  const AIcon = a.tone === 'ok' ? CheckCircle2 : a.tone === 'warn' ? TriangleAlert : CircleSlash;
  return (
    <button type="button" role="radio" aria-checked={selected} className={`choice cap-driver ${selected ? 'is-on' : ''} ${d.available ? '' : 'is-unavailable'}`} onClick={onSelect}>
      <span className="choice-radio" aria-hidden />
      <span className="choice-icon">
        <Icon size={18} />
      </span>
      <span className="choice-text">
        <span className="choice-title">
          <span className="truncate">{meta.title}</span>
          <span className={`cap-avail tone-${a.tone}`}>
            <AIcon size={13} aria-hidden /> {a.word}
          </span>
        </span>
        <span className="choice-body">{meta.body}</span>
      </span>
    </button>
  );
}

function FineTune({ setup, exclude }: { setup: ScannerSetup; exclude?: string }) {
  const { driver, schema, values, setValues, resetValues } = setup;
  const sessionKeys = new Set(useCapture(s => s.sessionSettings).map(s => s.key));
  const own = schema.filter(s => s.key !== exclude && !sessionKeys.has(s.key));
  const session = schema.filter(s => sessionKeys.has(s.key));
  const sub = own.slice(0, 3).map(s => s.label.toLowerCase().replace(/ \/ .*/, '')).join(', ');
  return (
    <details className="disclosure cap-fine">
      <summary>
        Fine-tune <span className="sub truncate">{sub || 'capture settings'}</span>
        <ChevronDown size={16} className="chev" aria-hidden />
      </summary>
      <div className="disclosure-body">
        {own.length > 0 && (
          <div className="fields">
            {own.map(s => <SettingField key={s.key} s={s} value={values[s.key]} onChange={v => setValues(x => ({ ...x, [s.key]: v }))} />)}
          </div>
        )}
        {session.length > 0 && (
          <>
            <div className="cap-subhead">While scanning</div>
            <div className="fields">
              {session.map(s => <SettingField key={s.key} s={s} value={values[s.key]} onChange={v => setValues(x => ({ ...x, [s.key]: v }))} />)}
            </div>
          </>
        )}
        <div className="row spread">
          <span className="caption">{driver ? `Remembered for ${driverMeta(driver).title}.` : ''}</span>
          <button type="button" className="link small" onClick={resetValues}>
            Reset to defaults
          </button>
        </div>
      </div>
    </details>
  );
}

function SettingField({ s, value, onChange }: { s: SettingSchema; value: unknown; onChange: (v: unknown) => void }) {
  if (s.type === 'boolean') {
    return (
      <Field label={s.label} help={s.help}>
        <Switch checked={!!value} onChange={onChange} label={s.label} />
      </Field>
    );
  }
  if (s.type === 'select') {
    const options = (s.options ?? []).map(o => ({ value: String(o.value), label: o.label }));
    const long = options.some(o => o.label.length > 22);
    return (
      <Field label={s.label} help={s.help} inline={!long}>
        <Select value={String(value ?? s.default)} onChange={v => onChange(typeof s.default === 'number' ? Number(v) : v)} options={options} />
      </Field>
    );
  }
  if (s.type === 'number') {
    return (
      <Field label={s.label} help={s.help}>
        <NumberInput value={Number(value ?? s.default ?? 0)} step={s.step ?? 'any'} min={s.min} max={s.max} unit={s.unit} onChange={onChange} />
      </Field>
    );
  }
  return (
    <Field label={s.label} help={s.help} inline={false}>
      <TextInput mono value={String(value ?? '')} onChange={onChange} placeholder={s.key === 'folder' ? '/mnt/share/scans' : undefined} />
    </Field>
  );
}

/** A connected scanner in one line, with Change (reconnect with other settings) and Disconnect. */
export function ScannerSummary({ setup, onDisconnect }: { setup: ScannerSetup; onDisconnect: () => void }) {
  const status = useCapture(s => s.status)!;
  const busy = useCaptureAction(s => s.busy);
  const [editing, setEditing] = useState(false);
  const d = setup.drivers.find(x => x.id === status.driver);
  const meta = driverMeta(d ?? { id: status.driver ?? '', name: status.driver_name ?? 'Scanner', description: '' });
  const live = status.state === 'running' || status.state === 'paused';
  const s = status.settings ?? {};
  const facts = [
    surfaceWord(s.surface) ? `${surfaceWord(s.surface)} surface` : null,
    typeof s.point_distance === 'number' && s.point_distance > 0 ? `${fmtLen(s.point_distance, 2)} mm spacing` : null,
    typeof s.tracking === 'string' ? `tracking by ${s.tracking}` : null,
  ].filter(Boolean);
  const Icon = meta.icon;

  useEffect(() => {
    if (!editing) return;
    if (status.driver && setup.driverId !== status.driver) setup.setDriverId(status.driver);
  }, [editing]);

  if (editing) {
    return (
      <div className="cap-change">
        <ScannerChooser setup={setup} />
        <ActionError labels={['reconnect']} />
        <div className="row">
          <Button
            variant="primary"
            icon={<RefreshCw size={15} />}
            loading={busy === 'reconnect'}
            disabled={!setup.driver?.available || live}
            onClick={() => runAction('reconnect', async () => {
              await connectCapture(setup.driverId, setup.values);
              setEditing(false);
            })}
          >
            Reconnect with these settings
          </Button>
          <Button variant="ghost" onClick={() => setEditing(false)}>Cancel</Button>
        </div>
        {live && <p className="caption">Stop scanning first — reconnecting starts a new session.</p>}
      </div>
    );
  }

  return (
    <Block title="Scanner" guide="capture.scanner">
      <div className="cap-summary">
        <span className="choice-icon">
          <Icon size={18} />
        </span>
        <div className="grow">
          <div className="strong truncate">{meta.title}</div>
          <div className="caption truncate">{facts.length ? facts.join(' · ') : status.driver_name}</div>
        </div>
        <Button size="sm" variant="secondary" disabled={live} onClick={() => setEditing(true)}>Change</Button>
      </div>
      <div className="row">
        <button type="button" className="link small cap-disconnect" disabled={live} onClick={onDisconnect}>
          <Unplug size={13} aria-hidden /> Disconnect the scanner
        </button>
      </div>
    </Block>
  );
}
