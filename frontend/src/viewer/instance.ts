import type { Viewer } from './Viewer';

let current: Viewer | null = null;

export const setViewer = (v: Viewer | null) => {
  current = v;
};

export const getViewer = (): Viewer | null => current;
