import { useEffect, useRef } from 'react';
import { DEFAULT_LAYOUT, useStore } from '../store';

const LIMITS = { left: [220, 440], right: [340, 680] } as const;

/** Drag handle in the gap between two panels. Double-click resets; arrow keys nudge. */
export function Gutter({ panel, grow = 1, label }: { panel: 'left' | 'right'; grow?: 1 | -1; label: string }) {
  const el = useRef<HTMLDivElement>(null);
  const width = useStore(s => s.layout[panel]);

  useEffect(() => {
    const node = el.current;
    if (!node) return;
    let startX = 0;
    let startW = 0;
    const [min, max] = LIMITS[panel];
    const move = (e: PointerEvent) => {
      const w = Math.round(Math.min(max, Math.max(min, startW + (e.clientX - startX) * grow)));
      useStore.getState().setLayout({ [panel]: w });
    };
    const up = () => {
      document.body.classList.remove('is-resizing');
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    const down = (e: PointerEvent) => {
      if (e.button !== 0) return;
      e.preventDefault();
      startX = e.clientX;
      startW = useStore.getState().layout[panel];
      document.body.classList.add('is-resizing');
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    };
    node.addEventListener('pointerdown', down);
    return () => {
      node.removeEventListener('pointerdown', down);
      up();
    };
  }, [panel, grow]);

  const nudge = (delta: number) => {
    const [min, max] = LIMITS[panel];
    useStore.getState().setLayout({ [panel]: Math.min(max, Math.max(min, useStore.getState().layout[panel] + delta)) });
  };

  return (
    <div
      ref={el}
      className="gutter"
      role="separator"
      aria-orientation="vertical"
      aria-label={`Resize ${label}`}
      aria-valuenow={width}
      aria-valuemin={LIMITS[panel][0]}
      aria-valuemax={LIMITS[panel][1]}
      aria-valuetext={`${width} pixels wide`}
      tabIndex={0}
      onDoubleClick={() => useStore.getState().setLayout({ [panel]: DEFAULT_LAYOUT[panel] })}
      onKeyDown={e => {
        if (e.key === 'ArrowLeft') nudge(-24 * grow);
        else if (e.key === 'ArrowRight') nudge(24 * grow);
        else return;
        e.preventDefault();
      }}
      title={`Drag to resize ${label} · double-click to reset`}
    />
  );
}
