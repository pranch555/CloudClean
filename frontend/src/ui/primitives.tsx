import { createContext, useContext, useEffect, useId, useLayoutEffect, useRef, useState, type ButtonHTMLAttributes, type CSSProperties, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, Info, X } from 'lucide-react';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'subtle' | 'signal';

/** The id a Field's label points at: controls inside a Field use it when they have no id of their own. */
const FieldId = createContext<string | undefined>(undefined);

export function Button({ variant = 'secondary', size = 'md', icon, loading, block, children, className = '', ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md' | 'lg'; icon?: ReactNode; loading?: boolean; block?: boolean }) {
  return (
    <button type="button" className={`btn btn-${variant} btn-${size} ${block ? 'btn-block' : ''} ${className}`} disabled={rest.disabled || loading} {...rest}>
      {loading ? <span className="spinner" aria-hidden /> : icon}
      {children && <span className="btn-label">{children}</span>}
    </button>
  );
}

/** Icon-only button. `tip` shows a styled tooltip (above by default) instead of the browser's title bubble. */
export function IconButton({ label, active, size = 'md', tip, children, className = '', ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; active?: boolean; size?: 'sm' | 'md'; tip?: 'top' | 'bottom' }) {
  return (
    <button type="button" aria-label={label} title={tip ? undefined : label} data-tip={tip ? label : undefined} data-tip-side={tip === 'bottom' ? 'bottom' : undefined} aria-pressed={active} className={`icon-btn icon-btn-${size} ${active ? 'is-active' : ''} ${className}`} {...rest}>
      {children}
    </button>
  );
}

export function Switch({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label?: string; disabled?: boolean }) {
  const fieldId = useContext(FieldId);
  return (
    <button type="button" id={fieldId} role="switch" aria-checked={checked} aria-label={fieldId ? undefined : label} disabled={disabled} className={`switch ${checked ? 'is-on' : ''}`} onClick={() => onChange(!checked)}>
      <span className="switch-thumb" />
    </button>
  );
}

export function Segmented<T extends string>({ options, value, onChange, size = 'md', ariaLabel }: { options: { value: T; label: ReactNode; title?: string; guide?: string }[]; value: T; onChange: (v: T) => void; size?: 'sm' | 'md'; ariaLabel?: string }) {
  const fieldId = useContext(FieldId);
  // arrow keys move the choice, as in a native radio group
  const onKey = (e: React.KeyboardEvent) => {
    const i = options.findIndex(o => o.value === value);
    const d = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
    if (!d || i < 0) return;
    e.preventDefault();
    onChange(options[(i + d + options.length) % options.length].value);
    const group = e.currentTarget as HTMLElement;
    requestAnimationFrame(() => (group.querySelector('[aria-checked="true"]') as HTMLElement | null)?.focus());
  };
  return (
    <div className={`segmented segmented-${size}`} role="radiogroup" aria-label={ariaLabel} aria-labelledby={!ariaLabel && fieldId ? `${fieldId}-label` : undefined} onKeyDown={onKey}>
      {options.map(o => (
        <button key={o.value} type="button" role="radio" aria-checked={o.value === value} tabIndex={o.value === value ? 0 : -1} title={o.title} className={o.value === value ? 'is-on' : ''} onClick={() => onChange(o.value)} data-guide={o.guide}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, help, children, inline = true, htmlFor }: { label: ReactNode; help?: ReactNode; children: ReactNode; inline?: boolean; htmlFor?: string }) {
  const auto = useId();
  const id = htmlFor ?? `f${auto.replace(/:/g, '')}`;
  return (
    <div className={`field ${inline ? 'field-inline' : 'field-stacked'}`}>
      <label className="field-label" htmlFor={id} id={`${id}-label`}>
        {label}
      </label>
      <div className="field-control">
        <FieldId.Provider value={htmlFor ? undefined : id}>{children}</FieldId.Provider>
      </div>
      {help && <div className="field-help" id={`${id}-help`}>{help}</div>}
    </div>
  );
}

export function NumberInput({ value, onChange, step = 'any', min, max, unit, id, width, label }: { value: number; onChange: (v: number) => void; step?: number | 'any'; min?: number; max?: number; unit?: string; id?: string; width?: number; label?: string }) {
  const fieldId = useContext(FieldId);
  id ??= fieldId;
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const v = parseFloat(text);
    if (Number.isFinite(v)) onChange(min != null ? Math.max(min, max != null ? Math.min(max, v) : v) : v);
    else setText(String(value));
  };
  return (
    <div className="input-wrap" style={width ? { width } : undefined}>
      <input id={id} aria-label={id ? undefined : label} className="input input-number" type="number" inputMode="decimal" step={step} min={min} max={max} value={text} onChange={e => setText(e.target.value)} onBlur={commit} onKeyDown={e => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} />
      {unit && <span className="input-unit">{unit}</span>}
    </div>
  );
}

export function TextInput({ value, onChange, placeholder, id, mono, onEnter }: { value: string; onChange: (v: string) => void; placeholder?: string; id?: string; mono?: boolean; onEnter?: () => void }) {
  const fieldId = useContext(FieldId);
  id ??= fieldId;
  return <input id={id} className={`input ${mono ? 'mono' : ''}`} type="text" value={value} placeholder={placeholder} spellCheck={false} onChange={e => onChange(e.target.value)} onKeyDown={e => e.key === 'Enter' && onEnter?.()} />;
}

export function Select<T extends string | number>({ value, onChange, options, id, label }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; id?: string; label?: string }) {
  const fieldId = useContext(FieldId);
  id ??= fieldId;
  return (
    <div className="select-wrap">
      <select id={id} aria-label={id ? undefined : label} className="input select" value={String(value)} onChange={e => onChange((typeof value === 'number' ? Number(e.target.value) : e.target.value) as T)}>
        {options.map(o => (
          <option key={String(o.value)} value={String(o.value)}>
            {o.label}
          </option>
        ))}
      </select>
      <ChevronDown size={14} className="select-chevron" aria-hidden />
    </div>
  );
}

export function Slider({ value, min, max, step, onChange, format, label }: { value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string; label?: string }) {
  const fieldId = useContext(FieldId);
  const pct = ((value - min) / (max - min || 1)) * 100;
  return (
    <div className="slider">
      <input type="range" id={fieldId} aria-label={fieldId ? undefined : label} aria-valuetext={format ? format(value) : undefined} min={min} max={max} step={step} value={value} style={{ '--pct': `${pct}%` } as React.CSSProperties} onChange={e => onChange(parseFloat(e.target.value))} />
      {format && <span className="slider-value mono">{format(value)}</span>}
    </div>
  );
}

export function Section({ title, children, actions, collapsible, defaultOpen = true, tone }: { title: ReactNode; children: ReactNode; actions?: ReactNode; collapsible?: boolean; defaultOpen?: boolean; tone?: 'accent' | 'warning' }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={`section ${tone ? `section-${tone}` : ''} ${open ? 'is-open' : ''}`}>
      <header className="section-head">
        {collapsible ? (
          <button type="button" className="section-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
            <ChevronDown size={14} className="section-chevron" aria-hidden />
            <span className="section-title">{title}</span>
          </button>
        ) : (
          <span className="section-title">{title}</span>
        )}
        {actions && <div className="section-actions">{actions}</div>}
      </header>
      {(!collapsible || open) && <div className="section-body">{children}</div>}
    </section>
  );
}

export function Badge({ children, tone = 'neutral', title }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'signal' | 'good' | 'ok' | 'warning' | 'serious' | 'critical' | 'danger' | 'info' | 'ink' | 'pointcloud' | 'mesh' | 'image'; title?: string }) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {children}
    </span>
  );
}

export function Progress({ value, indeterminate, tone = 'accent' }: { value?: number | null; indeterminate?: boolean; tone?: 'accent' | 'good' | 'warning' | 'critical' }) {
  return (
    <div className={`progress progress-${tone} ${indeterminate || value == null ? 'is-indeterminate' : ''}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={value != null ? Math.round(value * 100) : undefined}>
      <div className="progress-bar" style={value != null && !indeterminate ? { transform: `scaleX(${Math.max(0, Math.min(1, value))})` } : undefined} />
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="kbd">{children}</kbd>;
}

export function Empty({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      {icon && <div className="empty-icon">{icon}</div>}
      <div className="empty-title">{title}</div>
      {children && <div className="empty-body">{children}</div>}
      {action && <div className="empty-action">{action}</div>}
    </div>
  );
}

export function Hint({ children }: { children: ReactNode }) {
  return (
    <p className="hint">
      <Info size={13} aria-hidden />
      <span>{children}</span>
    </p>
  );
}

export function Stat({ label, value, unit, tone, sub }: { label: string; value: ReactNode; unit?: string; tone?: 'good' | 'warning' | 'critical'; sub?: ReactNode }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ''}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value mono">
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

/** Lightweight popover anchored to its trigger; closes on outside click / Escape. */
/** Keep this far from the window edges. */
const EDGE = 8;
const GAP = 8;

interface Placement {
  top: number;
  left: number;
  maxHeight: number;
  side: 'bottom' | 'top' | 'left' | 'right';
}

/**
 * Where a popover fits: on the preferred side of its trigger when there is room, else the roomier side, always
 * inside the window (shifted along the edge, and scrolling when taller than the space it has).
 */
function placePopover(anchor: DOMRect, width: number, height: number, side: 'bottom' | 'top' | 'left', align: 'start' | 'end' | 'center'): Placement {
  const vw = document.documentElement.clientWidth;
  const vh = window.innerHeight;
  const clampX = (x: number) => Math.min(Math.max(x, EDGE), Math.max(EDGE, vw - EDGE - width));
  if (side === 'left') {
    const leftRoom = anchor.left - GAP - EDGE;
    const rightRoom = vw - anchor.right - GAP - EDGE;
    const onLeft = width <= leftRoom || leftRoom >= rightRoom;
    const maxHeight = vh - 2 * EDGE;
    const h = Math.min(height, maxHeight);
    return {
      left: clampX(onLeft ? anchor.left - GAP - width : anchor.right + GAP),
      top: Math.min(Math.max(anchor.top, EDGE), vh - EDGE - h),
      maxHeight,
      side: onLeft ? 'left' : 'right',
    };
  }
  const below = vh - anchor.bottom - GAP - EDGE;
  const above = anchor.top - GAP - EDGE;
  let s: 'bottom' | 'top' = side;
  if (s === 'bottom' && height > below && above > below) s = 'top';
  else if (s === 'top' && height > above && below > above) s = 'bottom';
  const room = Math.max(s === 'bottom' ? below : above, 120);
  const h = Math.min(height, room);
  const x = align === 'start' ? anchor.left : align === 'end' ? anchor.right - width : anchor.left + anchor.width / 2 - width / 2;
  return { left: clampX(x), top: s === 'bottom' ? anchor.bottom + GAP : anchor.top - GAP - h, maxHeight: room, side: s };
}

/**
 * A floating panel under (or beside) its trigger. It renders at the top of the page, so no panel can clip it,
 * and it is placed to fit the window: it flips to the side with room, shifts along the edge and scrolls when
 * taller than the space it has; it follows its trigger on scroll and resize.
 */
export function Popover({ trigger, children, align = 'end', side = 'bottom', className = '' }: { trigger: (props: { open: boolean; toggle: () => void }) => ReactNode; children: ReactNode | ((api: { close: () => void }) => ReactNode); align?: 'start' | 'end' | 'center'; side?: 'bottom' | 'top' | 'left'; className?: string }) {
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState<Placement | null>(null);
  const anchor = useRef<HTMLDivElement>(null);
  const pop = useRef<HTMLDivElement>(null);
  const id = useId();

  const focusTrigger = () => anchor.current?.querySelector<HTMLElement>('button, [href], [tabindex]:not([tabindex="-1"])')?.focus();

  useEffect(() => {
    if (!open) return;
    const inside = (t: EventTarget | null) => t instanceof Node && (!!anchor.current?.contains(t) || !!pop.current?.contains(t));
    const onDown = (e: PointerEvent) => !inside(e.target) && setOpen(false);
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      setOpen(false);
      focusTrigger();
    };
    // keyboard focus leaving both the trigger and the popover closes it
    const onFocus = (e: FocusEvent) => !inside(e.target) && setOpen(false);
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    document.addEventListener('focusin', onFocus);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('focusin', onFocus);
    };
  }, [open]);

  useLayoutEffect(() => {
    if (!open) {
      setPlace(null);
      return;
    }
    const update = () => {
      const a = anchor.current?.getBoundingClientRect();
      const p = pop.current;
      if (!a || !p) return;
      // natural size: the full content height even when it currently scrolls
      const next = placePopover(a, p.offsetWidth, p.scrollHeight + 2, side, align);
      setPlace(cur => (cur && cur.top === next.top && cur.left === next.left && cur.maxHeight === next.maxHeight && cur.side === next.side ? cur : next));
    };
    update();
    pop.current?.focus({ preventScroll: true });
    const ro = new ResizeObserver(update);
    if (pop.current) ro.observe(pop.current);
    if (anchor.current) ro.observe(anchor.current);
    window.addEventListener('resize', update);
    window.addEventListener('scroll', update, true);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', update);
      window.removeEventListener('scroll', update, true);
    };
  }, [open, side, align]);

  const style: CSSProperties = place
    ? { top: place.top, left: place.left, maxHeight: place.maxHeight }
    : { top: 0, left: 0, visibility: 'hidden' };
  return (
    <div className="popover-anchor" ref={anchor}>
      {trigger({ open, toggle: () => setOpen(o => !o) })}
      {open &&
        createPortal(
          <div ref={pop} id={id} tabIndex={-1} className={`popover popover-${place?.side ?? side} ${className}`} style={style} role="dialog">
            {typeof children === 'function' ? children({ close: () => setOpen(false) }) : children}
          </div>,
          document.body,
        )}
    </div>
  );
}

/** Modal dialog in a portal: closes on Escape and on a click outside the card. */
export function Dialog({ title, onClose, children, width, icon }: { title: ReactNode; onClose: () => void; children: ReactNode; width?: number; icon?: ReactNode }) {
  const box = useRef<HTMLDivElement>(null);
  const titleId = useId();
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    box.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key !== 'Tab' || !box.current) return;
      // keep keyboard focus inside the dialog
      const items = [...box.current.querySelectorAll<HTMLElement>('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')].filter(el => !el.hasAttribute('disabled') && el.offsetParent !== null);
      if (!items.length) return;
      const first = items[0], last = items[items.length - 1];
      if (e.shiftKey && (document.activeElement === first || document.activeElement === box.current)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      opener?.focus?.();
    };
  }, [onClose]);
  return createPortal(
    <div className="dialog-backdrop" onPointerDown={onClose}>
      <div ref={box} tabIndex={-1} className="dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} style={width ? { width: `min(${width}px, 100%)` } : undefined} onPointerDown={e => e.stopPropagation()}>
        <header className="dialog-head">
          {icon}
          <h2 className="dialog-title" id={titleId}>{title}</h2>
          <span className="spacer" />
          <IconButton label="Close (Esc)" onClick={onClose}>
            <X size={18} />
          </IconButton>
        </header>
        {children}
      </div>
    </div>,
    document.body,
  );
}

/** Label + value pair used in result cards. */
export function Metric({ label, value, unit, tone }: { label: ReactNode; value: ReactNode; unit?: string; tone?: 'ok' | 'warn' | 'danger' }) {
  return (
    <div className={`metric ${tone ? `metric-${tone}` : ''}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value mono">
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </span>
    </div>
  );
}
