import type { Layout } from '../store';

/** How a side panel is shown: docked beside the 3D view, or as a drawer sliding over it. */
export type PanelMode = 'dock' | 'drawer';

const PAD = 20;            // .ws-body side padding, both sides together
const GUTTER = 10;         // resize handle between a panel and the view
const LEFT_MIN = 220;
const RIGHT_MIN = 320;     // the step panel still lays out at this width
const RIGHT_COMFORT = 340; // shrink the step panel this far before touching the model list
const STAGE_MIN = 400;     // narrowest useful 3D view
const STAGE_MIN_WITH_LEFT = 480;

export interface PanelFit {
  left: PanelMode;
  right: PanelMode;
  /** docked widths after shrinking to leave the 3D view room (the preferred widths stay in `layout`) */
  leftWidth: number;
  rightWidth: number;
  /** drawer widths: the preferred width, but never wider than the window allows */
  leftDrawer: number;
  rightDrawer: number;
}

/**
 * Fit the side panels to a window `width` px wide. The step panel stays docked the longest (it is the main
 * control); the model list becomes a drawer first. Docked panels shrink towards their minimum before either
 * one turns into a drawer, so the 3D view never gets narrower than it can be used at.
 */
export function panelFit(width: number, layout: Layout): PanelFit {
  const rightDock = width >= PAD + RIGHT_MIN + GUTTER + STAGE_MIN;
  const leftDock = width >= PAD + LEFT_MIN + GUTTER + (rightDock ? RIGHT_MIN + GUTTER : 0) + STAGE_MIN_WITH_LEFT;
  const leftOn = leftDock && layout.leftOpen;
  const rightOn = rightDock && layout.rightOpen;
  let left = layout.left;
  let right = layout.right;
  let over = PAD + (leftOn ? left + GUTTER : 0) + (rightOn ? right + GUTTER : 0) + (leftOn ? STAGE_MIN_WITH_LEFT : STAGE_MIN) - width;
  const shrink = (w: number, min: number) => {
    const d = Math.max(0, Math.min(over, w - min));
    over -= d;
    return w - d;
  };
  if (over > 0 && rightOn) right = shrink(right, Math.min(right, RIGHT_COMFORT));
  if (over > 0 && leftOn) left = shrink(left, LEFT_MIN);
  if (over > 0 && rightOn) right = shrink(right, RIGHT_MIN);
  const drawerMax = Math.max(260, width - PAD - 56);
  return {
    left: leftDock ? 'dock' : 'drawer',
    right: rightDock ? 'dock' : 'drawer',
    leftWidth: left,
    rightWidth: right,
    leftDrawer: Math.min(layout.left, drawerMax),
    rightDrawer: Math.min(Math.max(layout.right, RIGHT_MIN), drawerMax),
  };
}
