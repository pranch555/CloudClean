import paperTile from '../../assets/assistant-logo.png';
import carbonTile from '../../assets/assistant-logo-carbon.png';

type MarkProps = {
  /** CSS pixels; the artwork is 160 px, so it stays sharp up to 80 px on a 2x screen */
  size?: number;
  className?: string;
  /** give the mark a name only when nothing next to it says "assistant" already */
  label?: string;
};

/**
 * The assistant's logo tile. Both artworks are in the page and the theme picks one in CSS
 * (`:root[data-theme]`, set by lib/theme.ts, so "system" follows the OS as well) - no flash when switching.
 */
export function AssistantMark({ size = 48, className, label }: MarkProps) {
  const a11y = label ? { role: 'img', 'aria-label': label } : { 'aria-hidden': true };
  return (
    <span className={`assistant-mark ${className ?? ''}`} style={{ width: size, height: size }} {...a11y}>
      <img className="assistant-mark-paper" src={paperTile} alt="" width={size} height={size} decoding="async" draggable={false} />
      <img className="assistant-mark-carbon" src={carbonTile} alt="" width={size} height={size} decoding="async" draggable={false} />
    </span>
  );
}

type GlyphProps = { size?: number; className?: string };

/**
 * The assistant at text size: the logo's chat bubble, robot arm and cursor as one line glyph in currentColor.
 * Drawn on a 16 px grid; the stroke is set in screen pixels (1.5 px up to 16 px, growing to 2 px at 24 px) so it
 * stays legible from 14 to 20 px.
 */
export function AssistantGlyph({ size = 16, className }: GlyphProps) {
  const px = Math.min(2, Math.max(1.5, 1.5 + (size - 16) * 0.0625));
  return (
    <svg
      className={`assistant-glyph ${className ?? ''}`}
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={(px * 16) / size}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      {/* chat bubble, open where the cursor sits */}
      <path d="M11.89 8.09A5.25 5.25 0 1 0 2.2 9.62L1.1 14.4l3.51-2.6A5.25 5.25 0 0 0 7.48 12.2" />
      {/* robot arm: base, elbow joint, gripper hanging down */}
      <path d="M4.9 10.25 6 4.4l3.4 1.7v1.4" />
      <circle cx="6" cy="4.4" r="1.05" fill="currentColor" stroke="none" />
      {/* cursor */}
      <path d="M9.6 9.6 15 11.8l-2.7.75-.75 2.7z" fill="currentColor" strokeWidth="1" />
    </svg>
  );
}
