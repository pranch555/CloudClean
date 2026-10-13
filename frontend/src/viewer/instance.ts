import type { Viewer } from './Viewer';

/*
 * The one `Viewer` of the app: viewport/Viewport.tsx sets it when the 3D view mounts, and everything else reaches the
 * 3D view through `getViewer()`.
 */

let current: Viewer | null = null;

export const setViewer = (v: Viewer | null) => {
  current = v;
};

export const getViewer = (): Viewer | null => current;
