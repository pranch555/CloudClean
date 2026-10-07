import { Box, CameraOff, Gem, Moon, PictureInPicture2, RotateCcw, TriangleAlert, CheckCircle2, Info, XCircle, WandSparkles } from 'lucide-react';
import { fmtCount } from '../../lib/format';
import { Field, Segmented, Slider, Switch } from '../../ui/primitives';
import { CameraPicture } from '../../features/capture/CameraPicture';
import {
  type CameraInfo,
  type Cams,
  type Surface,
  fmtExposure,
  linesWord,
  setCamera,
  setCameraView,
  SURFACE_WORD,
  useCamera,
  useCameraInfo,
  usePreviewHold,
} from '../../features/capture/cameraStore';
import { useCapture } from '../../features/capture/captureStore';
import { Block } from '../StepFrame';

/*
 * Scan → Camera view: what the scanner's cameras see, live — before scanning (Revo Metro shows it too: the laser is on,
 * nothing is recorded) and while scanning. Below the picture, in plain words: are the laser lines bright enough,
 * what the part is like (Normal / Dark / Shiny), Auto or Manual, and in Manual the four settings themselves.
 */

const SURFACES: { value: Surface; icon: typeof Box; hint: string }[] = [
  { value: 'general', icon: Box, hint: 'Matte or light parts' },
  { value: 'dark', icon: Moon, hint: 'Black or very dark parts: much more laser light' },
  { value: 'reflective', icon: Gem, hint: 'Polished or machined metal: less glare' },
];

const TONE_ICON = { ok: CheckCircle2, warn: TriangleAlert, danger: XCircle, off: Info } as const;

export function CameraBlock() {
  const status = useCapture(s => s.status)!;
  const info = useCamera(s => s.info);
  const cams = useCamera(s => s.cams);
  const overlay = useCamera(s => s.overlay);
  const float = useCamera(s => s.float);
  const paused = useCamera(s => s.paused);
  const error = useCamera(s => s.error);
  const scanning = status.state === 'running' || status.state === 'paused';
  const usable = status.state !== 'error' && status.state !== 'closed';
  const cameraOn = usable && (scanning || !paused);

  usePreviewHold(usable && !scanning && !paused);
  useCameraInfo(usable);

  if (info && !info.available) return null;
  const s = info?.settings;

  return (
    <Block
      guide="capture.camera"
      title="Camera view"
      aside={
        <button type="button" className="link small cam-popout" onClick={() => setCameraView({ float: !float })} aria-pressed={float}>
          <PictureInPicture2 size={14} aria-hidden /> {float ? 'Back in this panel' : 'Show over the 3D view'}
        </button>
      }
    >
      {float ? (
        <div className="cam-elsewhere">
          <PictureInPicture2 size={16} aria-hidden />
          <span className="grow">The picture is over the 3D view, top right.</span>
        </div>
      ) : (
        <CameraPicture on={cameraOn} scanning={scanning} onTurnOn={() => useCamera.setState({ paused: false })} />
      )}

      <div className="cam-tools">
        <Segmented<Cams>
          size="sm"
          ariaLabel="Which camera"
          value={cams}
          onChange={v => setCameraView({ cams: v })}
          options={[
            { value: 'both', label: 'Both' },
            { value: 'left', label: 'Left' },
            { value: 'right', label: 'Right' },
          ]}
        />
        <label className="cam-switch">
          <Switch checked={overlay} onChange={on => setCameraView({ overlay: on })} label="Mark what CloudClean finds" />
          <span>Mark what it finds</span>
        </label>
        {!scanning && !paused && usable && (
          <button type="button" className="link small cam-off" onClick={() => useCamera.setState({ paused: true })} title="Turns the laser off until you start scanning">
            <CameraOff size={13} aria-hidden /> Turn off
          </button>
        )}
      </div>

      <LightMeter info={info} on={cameraOn} />

      {s && (
        <>
          <Field label="What is the part like?" inline={false}>
            <div className="cam-surfaces" role="radiogroup" aria-label="What is the part like?">
              {SURFACES.map(o => (
                <button
                  key={o.value}
                  type="button"
                  role="radio"
                  aria-checked={s.surface === o.value}
                  className={`cam-surface ${s.surface === o.value ? 'is-on' : ''}`}
                  onClick={() => s.surface !== o.value && setCamera({ surface: o.value })}
                  title={o.hint}
                >
                  <o.icon size={16} aria-hidden />
                  <span>{SURFACE_WORD[o.value]}</span>
                </button>
              ))}
            </div>
          </Field>

          <Field label="Exposure" inline={false} help={s.mode === 'auto' ? 'CloudClean keeps the laser lines bright but not washed out, as Revo Metro does. It adjusts only while the part is 22–38 cm away.' : 'Your settings stay exactly as you set them.'}>
            <Segmented
              ariaLabel="Exposure"
              value={s.mode}
              onChange={v => setCamera({ mode: v })}
              options={[
                { value: 'auto', label: 'Auto' },
                { value: 'manual', label: 'Manual' },
              ]}
            />
          </Field>

          {s.mode === 'auto' ? <AutoPanel info={info!} on={cameraOn} /> : <ManualPanel info={info!} />}
        </>
      )}
      {error && (
        <p className="err-text cap-error" role="alert">
          <XCircle size={14} aria-hidden /> <span>{error}</span>
        </p>
      )}
    </Block>
  );
}

// the meter's scale is a gauge, not a ruler: the band CloudClean aims for gets room of its own
const ZONES = [52, 80];

/** "Laser lines: bright enough" — on a scale from too dim through just right to too bright, with where they are. */
export function LightMeter({ info, on }: { info: CameraInfo | null; on: boolean }) {
  const r = info?.readout;
  const live = on && !!info?.streaming;
  const verdict = live ? info?.auto?.verdict : undefined;
  const w = live ? linesWord(verdict, r) : { word: '–', tone: 'off' as const, detail: undefined };
  const Icon = TONE_ICON[w.tone];
  const [lo, hi] = r?.band ?? [175, 230];
  const pos = (v: number) => {
    const x = v <= lo ? (v / lo) * ZONES[0] : v <= hi ? ZONES[0] + ((v - lo) / (hi - lo)) * (ZONES[1] - ZONES[0]) : ZONES[1] + ((v - hi) / (255 - hi)) * (100 - ZONES[1]);
    return `${Math.max(0, Math.min(100, x)).toFixed(1)}%`;
  };
  const value = live ? r?.stripe_brightness : null;
  return (
    <div className={`cam-meter tone-${w.tone}`} title={value != null ? `Brightest laser lines: ${Math.round(value)} of 255 (aim: ${lo}–${hi})` : undefined}>
      <div className="cam-meter-head">
        <Icon size={15} aria-hidden className="cam-meter-icon" />
        <span className="cam-meter-label">Laser lines</span>
        <span className="cam-meter-word" aria-live="polite">{w.word}</span>
      </div>
      <div className="cam-scale" aria-hidden>
        <span className="cam-scale-band" style={{ left: pos(lo), width: `calc(${pos(hi)} - ${pos(lo)})` }} />
        <span className="cam-scale-clip" style={{ left: pos(250) }} />
        {value != null && <span className="cam-scale-needle" style={{ left: pos(value) }} />}
      </div>
      <div className="cam-scale-words caption" aria-hidden style={{ gridTemplateColumns: `${ZONES[0]}fr ${ZONES[1] - ZONES[0]}fr ${100 - ZONES[1]}fr` }}>
        <span>too dim</span>
        <span>just right</span>
        <span>too bright</span>
      </div>
      {live && (
        <div className="cam-meter-foot caption">
          <span>{w.detail ? `${w.detail}.` : r?.points_per_frame != null ? `${fmtCount(r.points_per_frame)} points in each picture` : ''}</span>
          {r?.depth_mm != null && <span className="mono">{Math.round(r.depth_mm / 10)} cm away</span>}
        </div>
      )}
    </div>
  );
}

function AutoPanel({ info, on }: { info: CameraInfo; on: boolean }) {
  const s = info.settings!;
  const a = info.auto;
  const busy = a?.state === 'adjusting';
  return (
    <div className="stack tight">
      {on && a?.message && (
        <p className={`cam-auto is-${a.state}`} aria-live="polite">
          <WandSparkles size={14} aria-hidden className={busy ? 'cam-auto-busy' : undefined} /> <span>{a.message}</span>
        </p>
      )}
      <div className="cap-readouts cam-values" role="group" aria-label="What auto exposure is using">
        <Value value={`${Math.round(s.laser_pct)} %`} label="laser" title={`Laser level ${s.laser_level} of ${s.level_max}`} />
        <Value value={fmtExposure(s.exposure_us)} label="exposure" />
        <Value value={`${s.gain}×`} label="gain" />
        <Value value={String(s.marker_light)} label="markers" title="Marker light" />
      </div>
    </div>
  );
}

function Value({ value, label, title }: { value: string; label: string; title?: string }) {
  return (
    <div className="cap-readout" title={title}>
      <span className="cap-readout-value mono">{value}</span>
      <span className="cap-readout-label">{label}</span>
    </div>
  );
}

function ManualPanel({ info }: { info: CameraInfo }) {
  const s = info.settings!;
  const lim = info.limits;
  const preset = lim?.presets?.[s.surface];
  const atPreset = !!preset && preset.laser_level === s.laser_level && preset.exposure_us === s.exposure_us && preset.gain === s.gain && preset.marker_light === s.marker_light;
  return (
    <div className="fields cam-manual">
      <Field label="Laser brightness" inline={false} help="How long the laser shines for each picture. More for dark parts, less for shiny ones.">
        <Slider value={s.laser_level} min={lim?.laser_level[0] ?? 1} max={lim?.laser_level[1] ?? s.level_max} step={1} onChange={v => setCamera({ laser_level: v })} format={v => `${Math.round((100 * v) / s.level_max)} %`} label="Laser brightness" />
      </Field>
      <Field label="Exposure" inline={false} help="How long the cameras look. Longer also brightens everything around the lines.">
        <Slider value={s.exposure_us} min={lim?.exposure_us[0] ?? 100} max={lim?.exposure_us[1] ?? 2000} step={10} onChange={v => setCamera({ exposure_us: v })} format={fmtExposure} label="Exposure" />
      </Field>
      <Field label="Gain" inline={false} help="Brightens the whole picture. Higher is brighter but grainier.">
        <Slider value={s.gain} min={lim?.gain[0] ?? 1} max={lim?.gain[1] ?? 5} step={1} onChange={v => setCamera({ gain: v })} format={v => `${v}×`} label="Gain" />
      </Field>
      <Field label="Marker light" inline={false} help="Lights up the round marker stickers. Lower it if markers look like white blobs.">
        <Slider value={s.marker_light} min={lim?.marker_light[0] ?? 0} max={lim?.marker_light[1] ?? 255} step={1} onChange={v => setCamera({ marker_light: v })} format={v => String(v)} label="Marker light" />
      </Field>
      {preset && !atPreset && (
        <button type="button" className="link small cam-reset" onClick={() => setCamera({ laser_level: preset.laser_level, exposure_us: preset.exposure_us, gain: preset.gain, marker_light: preset.marker_light, mode: 'manual' })}>
          <RotateCcw size={13} aria-hidden /> Back to the {SURFACE_WORD[s.surface]} starting values
        </button>
      )}
    </div>
  );
}
