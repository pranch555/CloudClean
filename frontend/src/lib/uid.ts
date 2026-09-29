/**
 * A unique id for client-side list keys (attachments, measurements, edit-queue steps).
 *
 * `crypto.randomUUID` exists only in a secure context, so it is undefined whenever the app is opened over plain
 * HTTP - which is how CloudClean is normally reached on a lab network or over Tailscale (http://<host>:8765).
 * Calling it there throws and takes the whole action with it, so prefer it when present and fall back otherwise.
 * These ids never leave the browser, so the fallback only has to avoid collisions within one session.
 */
let counter = 0;

export function uid(): string {
  const c = globalThis.crypto;
  if (typeof c?.randomUUID === 'function') return c.randomUUID();
  counter += 1;
  const rand = typeof c?.getRandomValues === 'function'
    ? [...c.getRandomValues(new Uint8Array(8))].map(b => b.toString(16).padStart(2, '0')).join('')
    : Math.random().toString(16).slice(2, 18);
  return `${Date.now().toString(36)}-${counter.toString(36)}-${rand}`;
}
