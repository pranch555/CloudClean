import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import type { Asset, ColorMode, ResolvedRegion, ViewName } from '../lib/types';
import type { Vec3 } from '../lib/golden';
import type { JSAnimation } from 'animejs';
import { animate, reducedMotion } from '../lib/motion';
import { shapeTester } from '../lib/regions';
import { FreeControls, type RotatePivot, type RotateStyle } from './controls';
import { lutTexture, scalarColor, type ScalarStyle } from './colormaps';
import { createMeshPickMaterial, createPointMaterial, REVEAL_FRAGMENT_PARS, REVEAL_VERTEX, REVEAL_VERTEX_PARS, revealUniforms } from './pointMaterial';
import { ViewCube } from './ViewCube';

export interface DisplaySettings {
  colorMode: ColorMode;
  pointScale: number;
  wireframe: boolean;
  showBox: boolean;
  showGrid: boolean;
  scalar: { name: string; style: ScalarStyle } | null;
}

export interface Marker {
  id: string;
  position: [number, number, number];
  color: string;
  label?: string;
}

export interface Segment {
  from: [number, number, number];
  to: [number, number, number];
  color: string;
}

export interface Pane {
  id: string;
  label: string;
  assetIds: string[];
  transforms?: Record<string, number[]>;   // optional preview pose per asset (column-major 4x4)
  tone?: 'normal' | 'result';
}

export interface ClipSettings {
  enabled: boolean;
  axis: 'x' | 'y' | 'z';
  position: number; // world coordinate along the axis
  flip: boolean;
}

interface Item {
  meta: Asset;
  object: THREE.Points | THREE.Mesh;
  geometry: THREE.BufferGeometry;
  hasColor: boolean;
  hasNormal: boolean;
  originalColors?: THREE.BufferAttribute;
  scalarName?: string;
  scalars?: Float32Array;
  /** highlight(): this mesh shows some areas in colour and greys out the rest (the target; uFocus fades to it) */
  focused?: boolean;
  /** the highlight's shader uniforms, kept on the item so a rebuilt material picks up where the old one was */
  fx: FocusFx;
  /** setPlacement(): the exact box of the model as it is shown turned (a turned bounding box overshoots the floor) */
  placedBox?: THREE.Box3;
}

/**
 * Shader state of one model, shared by every material it gets (meshes rebuild theirs on a colour mode change).
 * highlight(): uFocus 0..1 greys out what is outside the `focus` mask, uGlow lifts what is in it.
 * Scan-beam reveal: uReveal 0..1 sweeps a laser plane along uRevealAxis (model space, from uRevealMin to uRevealMax);
 * at 1 (the resting value) the shader skips the reveal entirely.
 */
interface FocusFx {
  uFocus: { value: number };
  uGlow: { value: number };
  fade: JSAnimation | null;
  glowFade: JSAnimation | null;
  uReveal: { value: number };
  uRevealAxis: { value: THREE.Vector3 };
  uRevealMin: { value: number };
  uRevealMax: { value: number };
  uBeamWidth: { value: number };
  uRevealTime: { value: number };
  reveal: JSAnimation | null;
  /** performance.now() by which the reveal must be over (a stalled animation is finished by the frame loop) */
  revealDeadline: number;
}

const newFx = (focus = 0): FocusFx => ({
  uFocus: { value: focus },
  uGlow: { value: 0 },
  fade: null,
  glowFade: null,
  ...revealUniforms(),
  reveal: null,
  revealDeadline: 0,
});

/** The reveal uniforms of a model's shader state, by name (the same objects the materials read). */
const REVEAL_UNIFORMS = ['uReveal', 'uRevealAxis', 'uRevealMin', 'uRevealMax', 'uBeamWidth', 'uRevealTime'] as const;

/** The scan-beam reveal: how long the laser plane takes to cross a model the first time it appears. */
const REVEAL_MS = 1100;

/** highlight(): how long the rest of the model takes to grey out / come back, and the glow's breathing. */
const FOCUS_MS = 200;
const GLOW_PERIOD_MS = 1600;
const GLOW_MAX = 0.18;
const GLOW_STEADY = 0.1;

type Listener = () => void;

const PALETTE = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const HIGHLIGHT = new THREE.Color('#ee4b1f');
const FOCUS_REST = new THREE.Color('#b3ac9f');

export interface ViewerTheme {
  dark: boolean;
  points: string;
  mesh: string;
  highlight: string;
  grid: [number, number, number];
  bgTop: string;
  bgBottom: string;
}

export class Viewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly perspective = new THREE.PerspectiveCamera(35, 1, 0.01, 1e6);
  readonly orthographic = new THREE.OrthographicCamera(-1, 1, 1, -1, -1e6, 1e6);
  camera: THREE.PerspectiveCamera | THREE.OrthographicCamera = this.perspective;
  readonly controls: FreeControls;
  readonly items = new Map<string, Item>();
  settings: DisplaySettings = { colorMode: 'original', pointScale: 1, wireframe: false, showBox: true, showGrid: true, scalar: null };
  upAxis: 'y' | 'z' = 'y';
  activeId: string | null = null;
  toolActive = false;
  floorOn = false;
  radius = 50;

  onLoading?: (names: string[] | null) => void;
  onFrame?: (viewer: Viewer) => void;
  /** A model's preview finished loading (thumbnails are made from it). */
  onItemLoaded?: (id: string) => void;
  /** The user grabbed the view with the mouse, wheel or touch. */
  onInteract?: () => void;
  theme: ViewerTheme = { dark: false, points: '#45423c', mesh: '#b3ac9f', highlight: '#ee4b1f', grid: [0.2, 0.18, 0.15], bgTop: '#f8f6f2', bgBottom: '#e2ded4' };
  colorFor: (id: string) => string = () => PALETTE[0];

  private dirty = true;
  private loading = new Map<string, Promise<THREE.BufferGeometry>>();
  private syncToken = 0;
  private helpers = new THREE.Group();
  private boxHelper: THREE.Box3Helper | null = null;
  private grid: THREE.Mesh;
  private markerGroup = new THREE.Group();
  private markerGroups = new Map<string, THREE.Group>();
  private pivotMarker: THREE.Mesh;
  private viewCube = new ViewCube();
  private meshPickMaterial = createMeshPickMaterial();
  private pickTarget = new THREE.WebGLRenderTarget(33, 33, { type: THREE.FloatType, format: THREE.RGBAFormat });
  private clipPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);
  private clip: ClipSettings = { enabled: false, axis: 'x', position: 0, flip: false };
  private lut = lutTexture({ kind: 'diverging', min: -1, max: 1, tolerance: 0.1, steps: 0 });
  private listeners = new Set<Listener>();
  private live: LiveCloud | null = null;
  private liveFrame: THREE.Points | null = null;
  private liveMarkers: THREE.Points | null = null;
  private mapMarkers: THREE.Points | null = null;
  private panes: Pane[] = [];
  private resizeObserver: ResizeObserver;
  /** downloaded per-vertex scalars by asset + name (highlight() runs on hover: it must not fetch again) */
  private scalarCache = new Map<string, Promise<Float32Array | null>>();
  /** the latest highlight() call (an older one still downloading must not win) */
  private highlightToken = 0;
  /** the mesh whose highlighted vertices breathe (highlight(..., { glow: true })), and when it started */
  private glowItem: Item | null = null;
  private glowStart = 0;
  /** what the last applied highlight() lights (frameHighlight() flies to it); null when nothing is highlighted */
  private lit: { it: Item; name: string; values: number[] } | null = null;
  /** models the scan beam is sweeping right now, and every asset that has had its reveal this session */
  private revealing = new Set<Item>();
  private revealed = new Set<string>();

  constructor(readonly canvas: HTMLCanvasElement, readonly container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.localClippingEnabled = true;
    this.renderer.setClearColor(0x000000, 0);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.5;
    pmrem.dispose();

    // studio rig that follows the camera: key from upper left, soft fill from lower right
    for (const cam of [this.perspective, this.orthographic]) {
      const key = new THREE.DirectionalLight(0xffffff, 1.25);
      key.position.set(-0.55, 0.8, 0.6);
      key.target.position.set(0, 0, -1);
      const fill = new THREE.DirectionalLight(0xdfe8ff, 0.35);
      fill.position.set(0.7, -0.35, 0.5);
      fill.target.position.set(0, 0, -1);
      cam.add(key, key.target, fill, fill.target);
    }
    this.scene.add(this.perspective, this.orthographic);

    this.grid = makeGrid();
    this.helpers.add(this.grid, this.markerGroup);
    this.pivotMarker = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: 0xee4b1f, depthTest: false, transparent: true, opacity: 0.9 }));
    this.pivotMarker.renderOrder = 1000;
    this.pivotMarker.visible = false;
    this.helpers.add(this.pivotMarker);
    this.scene.add(this.helpers);

    this.perspective.position.set(120, 90, 160);
    this.perspective.lookAt(0, 0, 0);

    const viewer = this;
    this.controls = new FreeControls({
      get camera() {
        return viewer.camera;
      },
      canvas,
      pickPoint: (x, y) => this.pickPoint(x, y),
      modelCenter: () => {
        const box = this.visibleBox();
        return box.isEmpty() ? null : box.getCenter(new THREE.Vector3());
      },
      sceneRadius: () => this.radius,
      toolActive: () => this.toolActive,
      onChange: () => this.invalidate(),
      onInteract: () => this.onInteract?.(),
      onPivot: p => {
        this.pivotMarker.visible = !!p;
        if (p) {
          this.pivotMarker.position.copy(p);
          this.pivotMarker.scale.setScalar(this.radius / 90);
        }
        this.invalidate();
      },
    });

    canvas.addEventListener('pointermove', this.onCubeHover);
    canvas.addEventListener('pointerleave', () => this.viewCube.hover(null, null) && this.invalidate());
    canvas.addEventListener('pointerdown', this.onCubeClick, { capture: true });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.renderer.setAnimationLoop(this.frame);
  }

  dispose() {
    for (const it of this.items.values()) this.finishReveal(it);
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.renderer.dispose();
  }

  subscribe(fn: Listener) {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  invalidate() {
    this.dirty = true;
  }

  // ------------------------------------------------------------------ frame
  private aspectLock: number | null = null;

  private resize() {
    const W = Math.max(1, this.container.clientWidth);
    const H = Math.max(1, this.container.clientHeight);
    let w = W, h = H;
    if (this.aspectLock) {
      if (W / H > this.aspectLock) w = Math.floor(H * this.aspectLock);
      else h = Math.floor(W / this.aspectLock);
    }
    // letterboxed while aligning a photo: the canvas matches the photo's aspect ratio exactly
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.canvas.style.left = `${Math.floor((W - w) / 2)}px`;
    this.canvas.style.top = `${Math.floor((H - h) / 2)}px`;
    this.renderer.setSize(w, h, false);
    this.perspective.aspect = w / h;
    this.perspective.updateProjectionMatrix();
    this.updateOrthoFrustum();
    this.invalidate();
  }

  private updateOrthoFrustum() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    const half = this.radius * 1.2;
    this.orthographic.left = (-half * w) / h;
    this.orthographic.right = (half * w) / h;
    this.orthographic.top = half;
    this.orthographic.bottom = -half;
    this.orthographic.updateProjectionMatrix();
  }

  private frame = (now: number) => {
    if (this.glowItem) {
      // a slow sine from 0 up to GLOW_MAX and back: drawn every frame only while a glowing highlight is on screen
      const t = (now - this.glowStart) / GLOW_PERIOD_MS;
      this.glowItem.fx.uGlow.value = GLOW_MAX * 0.5 * (1 - Math.cos(2 * Math.PI * t));
      if (this.glowItem.object.visible) this.dirty = true;
    }
    // a reveal whose animation stalled (its engine paused, a tab put away mid-sweep) must not leave a model half drawn
    for (const it of this.revealing) if (now > it.fx.revealDeadline) this.finishReveal(it);
    const moving = this.controls.update(now);
    if (!this.dirty && !moving) return;
    this.dirty = false;
    this.updateClipping();
    const cam = this.camera;
    const dist = cam.position.distanceTo(this.controls.target);
    if (cam === this.perspective) {
      cam.near = Math.max(dist / 5000, this.radius * 1e-5);
      cam.far = dist + this.radius * 20;
      cam.updateProjectionMatrix();
    }
    const buf = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    const scale = buf.y / (2 * Math.tan(THREE.MathUtils.degToRad(this.perspective.fov) / 2));
    const orthoPx = (buf.y * this.orthographic.zoom) / (this.orthographic.top - this.orthographic.bottom);
    const pointUniforms = (m: THREE.ShaderMaterial) => {
      m.uniforms.uScale.value = scale;
      m.uniforms.uOrtho.value = cam === this.orthographic ? 1 : 0;
      m.uniforms.uOrthoPx.value = orthoPx;
    };
    for (const it of this.items.values()) if (it.object instanceof THREE.Points) pointUniforms(it.object.material as THREE.ShaderMaterial);
    if (this.live) pointUniforms(this.live.material);
    cam.updateMatrixWorld();
    if (this.panes.length) this.renderPanes(cam);
    else this.renderer.render(this.scene, cam);
    this.viewCube.render(this.renderer, cam, this.container.clientWidth, this.container.clientHeight);
    this.onFrame?.(this);
    this.listeners.forEach(l => l());
  };

  /**
   * Side-by-side comparison: each pane shows a subset of the assets (optionally moved by a preview transform, e.g.
   * the alignment a merge would use) through the same camera, so the views stay in step while you orbit.
   */
  setPanes(panes: Pane[]) {
    this.panes = panes;
    for (const it of this.items.values()) {
      it.object.matrixAutoUpdate = true;
      it.object.matrix.identity();
      it.placedBox = undefined;
    }
    this.invalidate();
  }

  paneRects(width = this.container.clientWidth, height = this.container.clientHeight): { pane: Pane; x: number; y: number; w: number; h: number }[] {
    const n = this.panes.length;
    if (!n) return [];
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    const gap = 6;
    return this.panes.map((pane, i) => {
      const r = Math.floor(i / cols), c = i % cols;
      const inRow = Math.min(cols, n - r * cols);
      const w = (width - gap * (inRow + 1)) / inRow;
      const h = (height - gap * (rows + 1)) / rows;
      return { pane, x: gap + c * (w + gap), y: gap + r * (h + gap), w, h };
    });
  }

  private renderPanes(cam: THREE.PerspectiveCamera | THREE.OrthographicCamera) {
    const H = this.container.clientHeight;
    const wanted = new Set(this.panes.flatMap(p => p.assetIds));
    const prevVisible = new Map<string, boolean>();
    for (const [id, it] of this.items) prevVisible.set(id, it.object.visible);
    const aspectCam = cam as THREE.PerspectiveCamera;
    const prevAspect = aspectCam.aspect;
    this.renderer.setScissorTest(true);
    for (const { pane, x, y, w, h } of this.paneRects()) {
      for (const [id, it] of this.items) it.object.visible = pane.assetIds.includes(id) && wanted.has(id);
      for (const id of pane.assetIds) {
        const it = this.items.get(id);
        const m = pane.transforms?.[id];
        if (!it) continue;
        it.object.matrixAutoUpdate = !m;
        if (m) it.object.matrix.fromArray(m);          // column-major, as three.js stores matrices
        else it.object.matrix.identity();
        it.object.updateMatrixWorld(true);
      }
      if (cam === this.perspective) {
        aspectCam.aspect = w / h;
        aspectCam.updateProjectionMatrix();
      }
      this.renderer.setViewport(x, H - y - h, w, h);
      this.renderer.setScissor(x, H - y - h, w, h);
      this.renderer.render(this.scene, cam);
    }
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, this.container.clientWidth, H);
    if (cam === this.perspective) {
      aspectCam.aspect = prevAspect;
      aspectCam.updateProjectionMatrix();
    }
    for (const [id, it] of this.items) it.object.visible = prevVisible.get(id) ?? false;
  }

  // ------------------------------------------------------------------ camera
  setRotateStyle(style: RotateStyle) {
    this.controls.style = style;
  }

  setRotatePivot(mode: RotatePivot) {
    this.controls.pivotMode = mode;
  }

  setUpAxis(axis: 'y' | 'z') {
    this.upAxis = axis;
    this.controls.up.set(0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0);
    this.viewCube.setUpAxis(axis);
    this.grid.rotation.set(axis === 'y' ? -Math.PI / 2 : 0, 0, 0);
    (this.grid.material as THREE.ShaderMaterial).uniforms.uUpZ.value = axis === 'z' ? 1 : 0;
    this.placeGrid();
    this.invalidate();
  }

  setProjection(kind: 'perspective' | 'orthographic') {
    const from = this.camera;
    const to = kind === 'perspective' ? this.perspective : this.orthographic;
    if (from === to) return;
    to.position.copy(from.position);
    to.quaternion.copy(from.quaternion);
    if (to === this.orthographic) {
      const dist = from.position.distanceTo(this.controls.target);
      const visibleH = 2 * dist * Math.tan(THREE.MathUtils.degToRad(this.perspective.fov) / 2);
      this.updateOrthoFrustum();
      this.orthographic.zoom = (this.orthographic.top - this.orthographic.bottom) / visibleH;
      this.orthographic.updateProjectionMatrix();
    } else {
      const visibleH = (this.orthographic.top - this.orthographic.bottom) / this.orthographic.zoom;
      const dist = visibleH / (2 * Math.tan(THREE.MathUtils.degToRad(this.perspective.fov) / 2));
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(from.quaternion);
      to.position.copy(this.controls.target).addScaledVector(fwd, -dist);
    }
    this.camera = to;
    this.invalidate();
  }

  private visibleBox(ids?: string[]) {
    const box = new THREE.Box3();
    for (const [id, it] of this.items) {
      if (ids ? !ids.includes(id) : !it.object.visible) continue;
      if (it.placedBox) box.union(it.placedBox);
      else box.expandByObject(it.object);
    }
    if (this.liveFrame && this.liveFrame.visible && this.liveFrame.geometry.boundingBox) {
      box.union(this.liveFrame.geometry.boundingBox);
    }
    if (this.live && this.live.count > 0 && this.live.points.visible) {
      this.live.points.geometry.computeBoundingBox();
      box.union(this.live.points.geometry.boundingBox!);
    }
    return box;
  }

  private viewDirection(name: ViewName): { dir: THREE.Vector3; up: THREE.Vector3 } {
    const zUp = this.upAxis === 'z';
    const up = zUp ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    const front = zUp ? new THREE.Vector3(0, -1, 0) : new THREE.Vector3(0, 0, 1);
    const right = new THREE.Vector3(1, 0, 0);
    switch (name) {
      case 'front': return { dir: front, up };
      case 'back': return { dir: front.clone().negate(), up };
      case 'right': return { dir: right, up };
      case 'left': return { dir: right.clone().negate(), up };
      case 'top': return { dir: up, up: front.clone().negate() };
      case 'bottom': return { dir: up.clone().negate(), up: front };
      default: return { dir: front.clone().add(right).add(up.clone().multiplyScalar(0.8)).normalize(), up };
    }
  }

  private pendingFit: { ids?: string[]; view?: ViewName; animate: boolean } | null = null;

  /**
   * Frame the given (or all visible) assets, keeping the current orientation unless a view is named. Models that are
   * still loading are framed as soon as they arrive (a fit requested right after showing a big model used to find
   * nothing to frame and silently do nothing).
   */
  fit(ids?: string[], view?: ViewName, animate = true) {
    const waiting = ids ? ids.some(id => !this.items.has(id)) : this.loading.size > [...this.loading.keys()].filter(id => this.items.has(id)).length;
    if (waiting) {
      this.pendingFit = { ids, view, animate };
      if (ids && ids.every(id => !this.items.has(id))) return;
    } else this.pendingFit = null;
    const box = this.visibleBox(ids);
    if (box.isEmpty()) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    this.radius = Math.max(sphere.radius, 1e-6);
    this.updateOrthoFrustum();
    this.placeGrid();
    const cam = this.camera;
    let dir: THREE.Vector3;
    let quat: THREE.Quaternion;
    if (view && view !== 'fit') {
      const v = this.viewDirection(view);
      dir = v.dir;
      quat = lookQuaternion(dir, v.up);
    } else {
      dir = new THREE.Vector3(0, 0, 1).applyQuaternion(cam.quaternion);
      quat = cam.quaternion.clone();
    }
    const dist = (this.radius / Math.sin(THREE.MathUtils.degToRad(this.perspective.fov) / 2)) * 1.08;
    const pos = sphere.center.clone().addScaledVector(dir, dist);
    const zoom = cam === this.orthographic ? (this.orthographic.top - this.orthographic.bottom) / (this.radius * 2.3) : undefined;
    this.controls.animateTo(pos, quat, sphere.center, zoom, animate ? 420 : 0);
    this.invalidate();
  }

  /** Arbitrary direction (used by the view cube and capture guidance arrows). */
  lookFrom(direction: THREE.Vector3) {
    const box = this.visibleBox();
    const center = box.isEmpty() ? this.controls.target.clone() : box.getCenter(new THREE.Vector3());
    const dist = this.camera.position.distanceTo(this.controls.target);
    const dir = direction.clone().normalize();
    const up = Math.abs(dir.dot(this.controls.up)) > 0.98 ? new THREE.Vector3(0, 0, 1).applyQuaternion(this.camera.quaternion).multiplyScalar(-Math.sign(dir.dot(this.controls.up))) : this.controls.up.clone();
    this.controls.animateTo(center.clone().addScaledVector(dir, dist), lookQuaternion(dir, up), center);
  }

  private onCubeHover = (e: PointerEvent) => {
    const r = this.canvas.getBoundingClientRect();
    const cube = this.viewCube.rect(this.container.clientWidth);
    const lx = e.clientX - r.left - cube.x;
    const ly = e.clientY - r.top - cube.y;
    const inside = lx >= 0 && ly >= 0 && lx <= cube.w && ly <= cube.h;
    if (this.viewCube.hover(inside ? lx : null, inside ? ly : null)) this.invalidate();
    this.canvas.style.cursor = inside ? 'pointer' : '';
  };

  private onCubeClick = (e: PointerEvent) => {
    const r = this.canvas.getBoundingClientRect();
    const cube = this.viewCube.rect(this.container.clientWidth);
    const lx = e.clientX - r.left - cube.x;
    const ly = e.clientY - r.top - cube.y;
    if (lx < 0 || ly < 0 || lx > cube.w || ly > cube.h) return;
    e.stopImmediatePropagation();
    const dir = this.viewCube.click(lx, ly);
    if (dir) this.lookFrom(dir);
  };

  // ------------------------------------------------------------------ assets
  private fetchGeometry(meta: Asset): Promise<THREE.BufferGeometry> {
    if (!this.loading.has(meta.id)) {
      const p = fetch(`/api/assets/${meta.id}/preview`)
        .then(r => {
          if (!r.ok) throw new Error(`preview ${r.status}`);
          return r.arrayBuffer();
        })
        .then(buf => toFloat32(new PLYLoader().parse(buf)))
        .catch(err => {
          this.loading.delete(meta.id);
          throw err;
        });
      this.loading.set(meta.id, p);
    }
    return this.loading.get(meta.id)!;
  }

  async sync(metas: Asset[], activeId: string | null): Promise<boolean> {
    const token = ++this.syncToken;
    this.activeId = activeId;
    const wanted = new Set(metas.map(m => m.id));
    const missing = metas.filter(m => !this.items.has(m.id));
    if (missing.length) this.onLoading?.(missing.map(m => m.name));
    try {
      await Promise.all(
        missing.map(async meta => {
          const geom = await this.fetchGeometry(meta);
          if (!this.items.has(meta.id)) {
            const it = this.build(meta, geom);
            it.object.visible = false;
            this.items.set(meta.id, it);
            this.scene.add(it.object);
            window.setTimeout(() => this.onItemLoaded?.(meta.id), 400);
          }
        }),
      );
    } finally {
      if (token === this.syncToken) this.onLoading?.(null);
    }
    if (token !== this.syncToken) return false;
    for (const [id, it] of this.items) {
      const show = wanted.has(id);
      // a model's very first appearance this session gets the scan-beam reveal (not later toggles or reloads); the
      // sweep starts hidden in this same task, so no frame ever shows it whole first
      if (show && !it.object.visible && !this.revealed.has(id)) this.startReveal(it);
      it.object.visible = show;
      const meta = metas.find(m => m.id === id);
      if (meta) it.meta = meta;
    }
    await this.applyScalars();
    this.refreshBox();
    this.placeGrid();
    this.invalidate();
    const pf = this.pendingFit;
    if (pf && (!pf.ids || pf.ids.every(id => this.items.has(id)))) {
      this.pendingFit = null;
      this.fit(pf.ids, pf.view, pf.animate);
    }
    return true;
  }

  forget(id: string) {
    const it = this.items.get(id);
    if (it) {
      it.fx.fade?.cancel();
      it.fx.glowFade?.cancel();
      this.finishReveal(it);
      if (this.glowItem === it) this.glowItem = null;
      if (this.lit?.it === it) this.lit = null;
      this.scene.remove(it.object);
      it.geometry.dispose();
      (it.object.material as THREE.Material).dispose();
      this.items.delete(id);
    }
    this.loading.delete(id);
    for (const key of [...this.scalarCache.keys()]) if (key.startsWith(`${id}/`)) this.scalarCache.delete(key);
    this.invalidate();
  }

  /** Drop a cached preview (after the asset changed on the server). */
  reload(id: string) {
    this.forget(id);
  }

  visibleCount() {
    let n = 0;
    for (const it of this.items.values()) if (it.object.visible) n++;
    return n;
  }

  // ------------------------------------------------------------------ scan-beam reveal
  /**
   * A model's first appearance: a thin laser plane in the signal colour sweeps across it along its longest axis as
   * seen on screen (top to bottom when that axis stands up in the view, else left to right), drawing the model as it
   * passes (shader part in pointMaterial.ts). Not under reduced motion. The frame loop redraws only while it runs.
   */
  private startReveal(it: Item) {
    this.revealed.add(it.meta.id);
    if (reducedMotion()) return;
    const box = it.geometry.boundingBox;
    if (!box || box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    it.object.updateMatrixWorld();
    const camUp = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    const camRight = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    // the model axis that looks longest on screen (an axis pointing at the camera would sweep unseen)
    let best = { k: 0, score: -1, up: 0, right: 0 };
    for (let k = 0; k < 3; k++) {
      const w = new THREE.Vector3().setComponent(k, 1).transformDirection(it.object.matrixWorld);
      const up = w.dot(camUp), right = w.dot(camRight);
      const score = size.getComponent(k) * Math.hypot(up, right);
      if (score > best.score) best = { k, score, up, right };
    }
    if (!(best.score > 0)) return;
    // t grows downward on screen (or rightward): the plane starts at the top (or left) end
    const sign = Math.abs(best.up) > Math.abs(best.right) ? (best.up > 0 ? -1 : 1) : best.right >= 0 ? 1 : -1;
    const fx = it.fx;
    fx.reveal?.cancel();
    fx.uRevealAxis.value.set(0, 0, 0).setComponent(best.k, sign);
    const a = box.min.getComponent(best.k) * sign, b = box.max.getComponent(best.k) * sign;
    fx.uRevealMin.value = Math.min(a, b);
    fx.uRevealMax.value = Math.max(a, b);
    fx.uReveal.value = 0;
    fx.uRevealTime.value = 0;
    const t0 = performance.now();
    fx.revealDeadline = t0 + REVEAL_MS + 1500;
    this.revealing.add(it);
    const sweep = { p: 0 };
    fx.reveal = animate(sweep, {
      p: 1,
      duration: REVEAL_MS,
      ease: 'inOut(1.6)',
      onUpdate: () => {
        fx.uReveal.value = Math.min(sweep.p, 0.9999); // 1 switches the reveal off: only onComplete gets there
        fx.uRevealTime.value = (performance.now() - t0) / 1000;
        this.invalidate();
      },
      onComplete: () => this.finishReveal(it),
    });
    this.invalidate();
  }

  /** End a reveal now, fully drawn (also on forget / dispose / a stalled animation). */
  private finishReveal(it: Item) {
    const fx = it.fx;
    fx.reveal?.cancel();
    fx.reveal = null;
    fx.uReveal.value = 1;
    this.revealing.delete(it);
    this.invalidate();
  }

  private build(meta: Asset, geom: THREE.BufferGeometry): Item {
    const hasColor = !!geom.attributes.color;
    geom.computeBoundingBox();
    geom.computeBoundingSphere();
    let object: THREE.Points | THREE.Mesh;
    const fx = newFx();
    if (meta.kind === 'mesh' && geom.index) {
      if (!geom.attributes.normal) geom.computeVertexNormals();
      // PLYLoader already turns the file's sRGB colours into the linear ones lit materials expect (kept as normalised
      // bytes for uchar colours): converting them again scrambled every coloured mesh
      object = new THREE.Mesh(geom, this.meshMaterial(fx));
    } else {
      const mat = createPointMaterial();
      for (const name of REVEAL_UNIFORMS) mat.uniforms[name] = fx[name];
      object = new THREE.Points(geom, mat);
    }
    object.userData.assetId = meta.id;
    // every model carries a (zeroed) selection mask: a shader attribute with no buffer reads a stale generic value
    // from whatever used that attribute slot last, which painted whole meshes in the highlight colour
    geom.setAttribute('selected', new THREE.BufferAttribute(new Float32Array(geom.attributes.position.count), 1));
    geom.setAttribute('focus', new THREE.BufferAttribute(new Float32Array(geom.attributes.position.count), 1));
    const it: Item = { meta, object, geometry: geom, hasColor, hasNormal: !!geom.attributes.normal, fx };
    this.applyMaterial(it);
    return it;
  }

  private meshMaterial(fx: FocusFx): THREE.MeshStandardMaterial {
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.62, metalness: 0.04, side: THREE.DoubleSide });
    // highlight(): uFocus 1 turns every vertex outside the focus mask the plain model colour; uGlow lifts the ones
    // inside it toward white and makes them faintly emissive (both fade with uFocus). The scan-beam reveal
    // (pointMaterial.ts) cuts away what its plane has not reached and lights the band at its front.
    mat.onBeforeCompile = shader => {
      shader.uniforms.uHighlight = { value: HIGHLIGHT };
      shader.uniforms.uFocus = fx.uFocus;
      shader.uniforms.uGlow = fx.uGlow;
      shader.uniforms.uFocusRest = { value: FOCUS_REST };
      for (const name of REVEAL_UNIFORMS) shader.uniforms[name] = fx[name];
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>\nattribute float selected;\nattribute float focus;\nvarying float vSelected;\nvarying float vFocus;\n${REVEAL_VERTEX_PARS}`)
        .replace('#include <begin_vertex>', `#include <begin_vertex>\nvSelected = selected;\nvFocus = focus;\n${REVEAL_VERTEX}`);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\nuniform vec3 uHighlight;\nuniform float uFocus;\nuniform float uGlow;\nuniform vec3 uFocusRest;\nvarying float vSelected;\nvarying float vFocus;\n${REVEAL_FRAGMENT_PARS}`)
        .replace(
          '#include <color_fragment>',
          '#include <color_fragment>\ndiffuseColor.rgb = mix(diffuseColor.rgb, uHighlight, step(0.5, vSelected) * 0.65);\n' +
            'float focusIn = smoothstep(0.25, 0.75, vFocus);\n' +
            'diffuseColor.rgb = mix(diffuseColor.rgb, uFocusRest, uFocus * (1.0 - focusIn) * 0.85);\n' +
            'float focusGlow = uGlow * uFocus * focusIn;\n' +
            'diffuseColor.rgb = mix(diffuseColor.rgb, vec3(1.0), focusGlow * 0.4);\n' +
            'totalEmissiveRadiance += diffuseColor.rgb * focusGlow * 0.6;\n' +
            'if (uReveal < 1.0) {\n' +
            '  vec3 beam = revealBeam();\n' +
            '  diffuseColor.rgb = mix(diffuseColor.rgb, uHighlight, clamp(beam.y + beam.z, 0.0, 1.0));\n' +
            '  totalEmissiveRadiance += uHighlight * (beam.x * 1.15 + beam.y * 0.35);\n' +
            '}',
        );
    };
    return mat;
  }

  private applyMaterial(it: Item) {
    const { colorMode, pointScale, wireframe } = this.settings;
    const solid = this.colorFor(it.meta.id);
    const scalarOn = colorMode === 'scalar' && !!it.scalars;
    if (it.object instanceof THREE.Points) {
      const m = it.object.material as THREE.ShaderMaterial;
      const u = m.uniforms;
      u.uMode.value = scalarOn ? 3 : colorMode === 'original' ? 0 : colorMode === 'normal' ? 2 : 1;
      u.uColor.value.set(colorMode === 'asset' ? solid : this.theme.points);
      u.uHighlight.value.set(this.theme.highlight);
      u.uHasColor.value = it.hasColor;
      u.uHasNormal.value = it.hasNormal;
      u.uHasScalar.value = scalarOn;
      u.uSize.value = Math.max(it.meta.stats?.spacing || 0, this.radius * 2e-4) * 1.9 * pointScale;
      u.uLut.value = this.lut.texture;
      u.uRange.value.set(this.lut.lo, this.lut.hi);
      m.clippingPlanes = this.clip.enabled ? [this.clipPlane] : [];
      this.invalidate();
      return;
    }
    const mesh = it.object as THREE.Mesh;
    if (colorMode === 'normal') {
      const old = mesh.material as THREE.Material;
      if (!(old instanceof THREE.MeshNormalMaterial)) {
        mesh.material = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
        old.dispose();
      }
    } else {
      if (!(mesh.material instanceof THREE.MeshStandardMaterial)) {
        (mesh.material as THREE.Material).dispose();
        mesh.material = this.meshMaterial(it.fx);
      }
      const mat = mesh.material as THREE.MeshStandardMaterial;
      const useVertex = scalarOn || (colorMode === 'original' && it.hasColor);
      if (scalarOn) this.paintMeshScalars(it);
      else if (it.originalColors) {
        it.geometry.setAttribute('color', it.originalColors);
        it.originalColors = undefined;
      }
      mat.vertexColors = useVertex;
      mat.color.set(colorMode === 'asset' ? solid : useVertex ? '#ffffff' : this.theme.mesh);
      mat.needsUpdate = true;
    }
    const mat = mesh.material as THREE.Material & { wireframe?: boolean };
    mat.wireframe = wireframe;
    mat.clippingPlanes = this.clip.enabled ? [this.clipPlane] : [];
    this.invalidate();
  }

  private paintMeshScalars(it: Item) {
    if (!it.scalars || !this.settings.scalar) return;
    if (!it.originalColors && it.geometry.attributes.color) it.originalColors = it.geometry.attributes.color as THREE.BufferAttribute;
    const n = it.geometry.attributes.position.count;
    const out = new Float32Array(n * 3);
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      scalarColor(it.scalars[i], this.settings.scalar.style, c);
      c.convertSRGBToLinear();
      out[i * 3] = c.r;
      out[i * 3 + 1] = c.g;
      out[i * 3 + 2] = c.b;
    }
    it.geometry.setAttribute('color', new THREE.BufferAttribute(out, 3));
  }

  setSettings(patch: Partial<DisplaySettings>) {
    Object.assign(this.settings, patch);
    if (patch.scalar !== undefined && this.settings.scalar) this.lut = lutTexture(this.settings.scalar.style);
    if (patch.showGrid !== undefined) this.placeGrid();
    const needsScalars = patch.scalar !== undefined || patch.colorMode !== undefined;
    const apply = () => {
      for (const it of this.items.values()) this.applyMaterial(it);
      this.refreshBox();
    };
    if (needsScalars) this.applyScalars().then(apply);
    else apply();
  }

  /** Download the active scalar field for visible assets that have it. */
  private async applyScalars() {
    const want = this.settings.colorMode === 'scalar' ? this.settings.scalar?.name : undefined;
    await Promise.all(
      [...this.items.values()].map(async it => {
        if (!want || !it.meta.scalars?.some(s => s.name === want)) {
          if (it.scalars) {
            it.scalars = undefined;
            it.scalarName = undefined;
            it.geometry.deleteAttribute('scalar');
          }
          return;
        }
        if (it.scalarName === want) return;
        const res = await fetch(`/api/assets/${it.meta.id}/scalars/${want}`);
        if (!res.ok) return;
        const values = new Float32Array(await res.arrayBuffer());
        if (values.length !== it.geometry.attributes.position.count) return;
        it.scalars = values;
        it.scalarName = want;
        if (it.object instanceof THREE.Points) it.geometry.setAttribute('scalar', new THREE.BufferAttribute(values, 1));
      }),
    );
    for (const it of this.items.values()) this.applyMaterial(it);
  }

  /** Scalar value of the nearest preview point to a world position (legend probe / tooltips). */
  scalarAt(point: THREE.Vector3): { value: number; assetId: string } | null {
    let best: { value: number; assetId: string; d: number } | null = null;
    for (const it of this.items.values()) {
      if (!it.object.visible || !it.scalars) continue;
      const pos = it.geometry.attributes.position.array as Float32Array;
      for (let i = 0; i < it.scalars.length; i++) {
        const dx = pos[i * 3] - point.x, dy = pos[i * 3 + 1] - point.y, dz = pos[i * 3 + 2] - point.z;
        const d = dx * dx + dy * dy + dz * dz;
        if (!best || d < best.d) best = { value: it.scalars[i], assetId: it.meta.id, d };
      }
    }
    return best ? { value: best.value, assetId: best.assetId } : null;
  }

  refreshBox() {
    if (this.boxHelper) {
      this.helpers.remove(this.boxHelper);
      this.boxHelper.geometry.dispose();
      this.boxHelper = null;
    }
    const it = this.activeId ? this.items.get(this.activeId) : undefined;
    if (!this.settings.showBox || !it || !it.object.visible || !it.geometry.boundingBox) return;
    this.boxHelper = new THREE.Box3Helper(it.placedBox?.clone() ?? it.geometry.boundingBox.clone().applyMatrix4(it.object.matrixWorld), new THREE.Color(this.theme.dark ? '#f3f1ec' : '#17161a'));
    (this.boxHelper.material as THREE.LineBasicMaterial).transparent = true;
    (this.boxHelper.material as THREE.LineBasicMaterial).opacity = 0.35;
    this.helpers.add(this.boxHelper);
    this.invalidate();
  }

  setOffset(id: string, offset: [number, number, number]) {
    this.items.get(id)?.object.position.set(...offset);
    this.items.get(id)?.object.updateMatrixWorld();
    this.refreshBox();
  }

  clearOffsets() {
    for (const it of this.items.values()) {
      it.object.position.set(0, 0, 0);
      it.object.updateMatrixWorld();
    }
    this.refreshBox();
  }

  private placeGrid() {
    const box = this.visibleBox();
    const g = this.grid;
    const mat = g.material as THREE.ShaderMaterial;
    if (box.isEmpty()) {
      g.visible = false;
      return;
    }
    g.visible = this.settings.showGrid || this.floorOn;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const extent = Math.max(size.x, size.y, size.z);
    const step = Math.pow(10, Math.floor(Math.log10(Math.max(extent / 4, 1e-9))));
    const floor = this.upAxis === 'y' ? box.min.y : box.min.z;
    g.position.set(center.x, this.upAxis === 'y' ? floor : center.y, this.upAxis === 'z' ? floor : center.z);
    g.scale.setScalar(extent * 6);
    mat.uniforms.uStep.value = step;
    mat.uniforms.uFade.value = extent * 2.2;
    mat.uniforms.uFloorR.value = extent * 0.85;
    mat.uniforms.uCenter.value.copy(g.position);
  }

  // ------------------------------------------------------------------ floor
  /** The Floor: a solid floor under the models (with the grid), to stand a model on. Independent of the grid switch. */
  setFloor(on: boolean) {
    this.floorOn = on;
    (this.grid.material as THREE.ShaderMaterial).uniforms.uFill.value = on ? (this.theme.dark ? 0.26 : 0.16) : 0;
    this.placeGrid();
    this.invalidate();
  }

  /**
   * Show a model turned / moved (matrix: model -> world) without changing its data, e.g. the Floor tool's preview;
   * null shows it as stored. The floor and the box follow the model as shown.
   */
  setPlacement(id: string, m: THREE.Matrix4 | null) {
    const it = this.items.get(id);
    if (!it) return;
    if (m) {
      it.object.matrixAutoUpdate = false;
      it.object.matrix.copy(m);
      const pos = it.geometry.getAttribute('position');
      const p = new THREE.Vector3();
      const box = new THREE.Box3();
      for (let i = 0; i < pos.count; i++) box.expandByPoint(p.fromBufferAttribute(pos, i).applyMatrix4(m));
      it.placedBox = box;
    } else {
      it.object.matrixAutoUpdate = true;
      it.object.updateMatrix();
      it.placedBox = undefined;
    }
    it.object.updateMatrixWorld(true);
    this.placeGrid();
    this.refreshBox();
    this.invalidate();
  }

  /** True while setPlacement() shows this model moved. */
  isPlaced(id: string) {
    return !!this.items.get(id)?.placedBox;
  }

  // ------------------------------------------------------------------ section plane
  setClip(clip: Partial<ClipSettings>) {
    Object.assign(this.clip, clip);
    for (const it of this.items.values()) this.applyMaterial(it);
    this.invalidate();
  }

  private updateClipping() {
    // three.js keeps fragments with plane distance >= 0: normal −axis keeps coord <= position, +axis keeps coord >= position
    const axis = new THREE.Vector3(this.clip.axis === 'x' ? 1 : 0, this.clip.axis === 'y' ? 1 : 0, this.clip.axis === 'z' ? 1 : 0);
    if (this.clip.flip) this.clipPlane.set(axis, -this.clip.position);
    else this.clipPlane.set(axis.negate(), this.clip.position);
  }

  sceneBounds(): THREE.Box3 {
    return this.visibleBox();
  }

  // ------------------------------------------------------------------ picking
  /** Exact world point under a canvas pixel via a float position render pass (works for points and meshes). */
  pickPoint(clientX: number, clientY: number, radiusPx = 16): THREE.Vector3 | null {
    const visible = [...this.items.values()].filter(it => it.object.visible);
    if (!visible.length && !(this.live && this.live.count)) return null;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = this.renderer.getPixelRatio();
    const W = Math.floor(rect.width * dpr), H = Math.floor(rect.height * dpr);
    const px = Math.floor((clientX - rect.left) * dpr), py = Math.floor((clientY - rect.top) * dpr);
    const size = radiusPx * 2 + 1;
    if (this.pickTarget.width !== size) this.pickTarget.setSize(size, size);
    const cam = this.camera.clone() as THREE.PerspectiveCamera | THREE.OrthographicCamera;
    cam.setViewOffset(W, H, px - radiusPx, py - radiusPx, size, size);
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();

    const swaps: [THREE.Mesh, THREE.Material | THREE.Material[]][] = [];
    const pointMats: THREE.ShaderMaterial[] = [];
    for (const it of visible) {
      if (it.object instanceof THREE.Points) {
        const m = it.object.material as THREE.ShaderMaterial;
        m.uniforms.uPick.value = 1;
        pointMats.push(m);
      } else {
        swaps.push([it.object, it.object.material]);
        this.meshPickMaterial.clippingPlanes = this.clip.enabled ? [this.clipPlane] : [];
        it.object.material = this.meshPickMaterial;
      }
    }
    if (this.live) {
      this.live.material.uniforms.uPick.value = 1;
      pointMats.push(this.live.material);
    }
    const helpersVisible = this.helpers.visible;
    const boxVisible = this.boxHelper?.visible;
    this.helpers.visible = false;
    const prevTarget = this.renderer.getRenderTarget();
    const prevClear = this.renderer.getClearAlpha();
    const prevTone = this.renderer.toneMapping;
    const bg = this.scene.background;
    this.scene.background = null;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.setRenderTarget(this.pickTarget);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear();
    this.renderer.render(this.scene, cam);
    const data = new Float32Array(size * size * 4);
    this.renderer.readRenderTargetPixels(this.pickTarget, 0, 0, size, size, data);
    this.renderer.setRenderTarget(prevTarget);
    this.renderer.setClearColor(0x000000, prevClear);
    this.renderer.toneMapping = prevTone;
    this.scene.background = bg;
    this.helpers.visible = helpersVisible;
    if (this.boxHelper && boxVisible !== undefined) this.boxHelper.visible = boxVisible;
    for (const [mesh, mat] of swaps) mesh.material = mat;
    for (const m of pointMats) m.uniforms.uPick.value = 0;
    this.invalidate();

    let best: THREE.Vector3 | null = null;
    let bestD = Infinity;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = (y * size + x) * 4;
        if (data[i + 3] < 0.5) continue;
        // render target rows start at the bottom
        const d = (x - radiusPx) ** 2 + (size - 1 - y - radiusPx) ** 2;
        if (d < bestD) {
          bestD = d;
          best = new THREE.Vector3(data[i], data[i + 1], data[i + 2]);
        }
      }
    }
    return best;
  }

  /** Which asset owns the point nearest to a world position (for point pairs). */
  assetAt(point: THREE.Vector3): string | null {
    let best: string | null = null;
    let bestD = Infinity;
    for (const [id, it] of this.items) {
      if (!it.object.visible) continue;
      const box = it.geometry.boundingBox!.clone().applyMatrix4(it.object.matrixWorld);
      const d = box.distanceToPoint(point);
      if (d < bestD) {
        bestD = d;
        best = id;
      }
    }
    return best;
  }

  worldToLocal(id: string, point: THREE.Vector3): [number, number, number] {
    const it = this.items.get(id);
    const p = point.clone();
    if (it) it.object.worldToLocal(p);
    return [p.x, p.y, p.z];
  }

  /** CSS-pixel position of a world point, or null when behind the camera. */
  project(point: THREE.Vector3 | [number, number, number]): { x: number; y: number } | null {
    const v = Array.isArray(point) ? new THREE.Vector3(...point) : point.clone();
    v.project(this.camera);
    if (v.z > 1 || v.z < -1) return null;
    return { x: ((v.x + 1) / 2) * this.container.clientWidth, y: ((1 - v.y) / 2) * this.container.clientHeight };
  }

  // ------------------------------------------------------------------ selection
  viewProjection(): number[] {
    this.camera.updateMatrixWorld();
    return new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse).toArray();
  }

  /**
   * Highlight preview points / vertices inside a screen polygon (NDC). visibleOnly uses a full-view position
   * pass so hidden back sides are not selected. Returns selected counts per asset.
   */
  selectPolygon(polygon: [number, number][], additive: boolean, visibleOnly: boolean): Record<string, number> {
    const W = this.container.clientWidth, H = this.container.clientHeight;
    const mask = document.createElement('canvas');
    const maskScale = 0.5;
    mask.width = Math.max(1, Math.floor(W * maskScale));
    mask.height = Math.max(1, Math.floor(H * maskScale));
    const g = mask.getContext('2d', { willReadFrequently: true })!;
    g.beginPath();
    polygon.forEach(([x, y], i) => {
      const sx = ((x + 1) / 2) * mask.width, sy = ((1 - y) / 2) * mask.height;
      if (i) g.lineTo(sx, sy);
      else g.moveTo(sx, sy);
    });
    g.closePath();
    g.fillStyle = '#fff';
    g.fill();
    const pixels = g.getImageData(0, 0, mask.width, mask.height).data;
    const depth = visibleOnly ? this.positionPass(mask.width, mask.height) : null;
    const camPos = this.camera.getWorldPosition(new THREE.Vector3());
    const vp = new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    const e = vp.elements;
    const counts: Record<string, number> = {};
    for (const [id, it] of this.items) {
      if (!it.object.visible) continue;
      const pos = it.geometry.attributes.position.array as Float32Array;
      const n = it.geometry.attributes.position.count;
      const prev = it.geometry.attributes.selected as THREE.BufferAttribute | undefined;
      const sel = additive && prev ? (prev.array as Float32Array) : new Float32Array(n);
      const m = it.object.matrixWorld.elements;
      const tol = Math.max(it.meta.stats?.spacing || 0, this.radius * 1e-3) * 3;
      let count = 0;
      for (let i = 0; i < n; i++) {
        const lx = pos[i * 3], ly = pos[i * 3 + 1], lz = pos[i * 3 + 2];
        const x = m[0] * lx + m[4] * ly + m[8] * lz + m[12];
        const y = m[1] * lx + m[5] * ly + m[9] * lz + m[13];
        const z = m[2] * lx + m[6] * ly + m[10] * lz + m[14];
        const w = e[3] * x + e[7] * y + e[11] * z + e[15];
        if (w <= 0) continue;
        const nx = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w;
        const ny = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w;
        if (nx < -1 || nx > 1 || ny < -1 || ny > 1) continue;
        const px = Math.min(mask.width - 1, Math.floor(((nx + 1) / 2) * mask.width));
        const py = Math.min(mask.height - 1, Math.floor(((1 - ny) / 2) * mask.height));
        if (pixels[(py * mask.width + px) * 4 + 3] < 128) continue;
        if (depth) {
          const di = ((mask.height - 1 - py) * mask.width + px) * 4;
          if (depth[di + 3] > 0.5) {
            const sx = depth[di] - camPos.x, sy = depth[di + 1] - camPos.y, sz = depth[di + 2] - camPos.z;
            const surf = Math.sqrt(sx * sx + sy * sy + sz * sz);
            const px3 = x - camPos.x, py3 = y - camPos.y, pz3 = z - camPos.z;
            if (Math.sqrt(px3 * px3 + py3 * py3 + pz3 * pz3) > surf + tol) continue;
          }
        }
        if (!sel[i]) {
          sel[i] = 1;
        }
      }
      for (let i = 0; i < n; i++) if (sel[i]) count++;
      it.geometry.setAttribute('selected', new THREE.BufferAttribute(sel, 1));
      if (count) counts[id] = count;
    }
    this.invalidate();
    return counts;
  }

  /**
   * Highlight preview points / vertices inside brush spheres [x, y, z, r] (world units). With `reset` the mask is
   * rebuilt, otherwise only the new spheres are added. Returns highlighted counts per asset.
   */
  paintSpheres(spheres: number[][], reset: boolean, onlyId?: string | null): Record<string, number> {
    const counts: Record<string, number> = {};
    for (const [id, it] of this.items) {
      if (!it.object.visible || (onlyId && id !== onlyId)) continue;
      const n = it.geometry.attributes.position.count;
      const prev = it.geometry.attributes.selected as THREE.BufferAttribute | undefined;
      const sel = !reset && prev && prev.array.length === n ? (prev.array as Float32Array) : new Float32Array(n);
      const pos = it.geometry.attributes.position.array as Float32Array;
      const inv = new THREE.Matrix4().copy(it.object.matrixWorld).invert();
      for (const s of spheres) {
        const c = new THREE.Vector3(s[0], s[1], s[2]).applyMatrix4(inv);
        const r2 = s[3] * s[3];
        for (let i = 0; i < n; i++) {
          if (sel[i]) continue;
          const dx = pos[i * 3] - c.x, dy = pos[i * 3 + 1] - c.y, dz = pos[i * 3 + 2] - c.z;
          if (dx * dx + dy * dy + dz * dz <= r2) sel[i] = 1;
        }
      }
      let count = 0;
      for (let i = 0; i < n; i++) if (sel[i]) count++;
      if (prev && prev.array === sel) prev.needsUpdate = true;
      else it.geometry.setAttribute('selected', new THREE.BufferAttribute(sel, 1));
      if (count) counts[id] = count;
    }
    this.invalidate();
    return counts;
  }

  /** CSS pixels covered by a world-space length at a world position (for the brush cursor). */
  pixelsFor(length: number, at: THREE.Vector3): number {
    const cam = this.camera;
    const h = this.canvas.clientHeight || 1;
    if (cam === this.orthographic) return (length * h * this.orthographic.zoom) / (this.orthographic.top - this.orthographic.bottom);
    const d = cam.position.distanceTo(at);
    return (length * h) / (2 * d * Math.tan(THREE.MathUtils.degToRad(this.perspective.fov) / 2));
  }

  clearSelection() {
    for (const it of this.items.values()) {
      it.geometry.setAttribute('selected', new THREE.BufferAttribute(new Float32Array(it.geometry.attributes.position.count), 1));
    }
    this.invalidate();
  }

  private positionPass(w: number, h: number): Float32Array {
    const target = new THREE.WebGLRenderTarget(w, h, { type: THREE.FloatType, format: THREE.RGBAFormat });
    const swaps: [THREE.Mesh, THREE.Material | THREE.Material[]][] = [];
    const pointMats: THREE.ShaderMaterial[] = [];
    for (const it of this.items.values()) {
      if (!it.object.visible) continue;
      if (it.object instanceof THREE.Points) {
        const m = it.object.material as THREE.ShaderMaterial;
        m.uniforms.uPick.value = 1;
        m.uniforms.uScale.value *= w / this.renderer.getDrawingBufferSize(new THREE.Vector2()).x;
        pointMats.push(m);
      } else {
        swaps.push([it.object, it.object.material]);
        it.object.material = this.meshPickMaterial;
      }
    }
    this.helpers.visible = false;
    const tone = this.renderer.toneMapping;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.setRenderTarget(target);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);
    const data = new Float32Array(w * h * 4);
    this.renderer.readRenderTargetPixels(target, 0, 0, w, h, data);
    this.renderer.setRenderTarget(null);
    this.renderer.toneMapping = tone;
    this.helpers.visible = true;
    for (const [mesh, mat] of swaps) mesh.material = mat;
    for (const m of pointMats) m.uniforms.uPick.value = 0;
    target.dispose();
    this.invalidate();
    return data;
  }

  // ------------------------------------------------------------------ markers
  /**
   * Markers live in named groups so independent overlays (measurements, thread crests, merge point pairs, capture
   * holes) can draw at the same time. `size` scales the marker spheres relative to the scene.
   */
  setMarkers(markers: Marker[], segments: Segment[] = [], group = 'default', size = 1) {
    let g = this.markerGroups.get(group);
    if (!g) {
      g = new THREE.Group();
      this.markerGroups.set(group, g);
      this.markerGroup.add(g);
    }
    for (const child of [...g.children]) {
      g.remove(child);
      const obj = child as THREE.Mesh | THREE.Line;
      obj.geometry.dispose();
      (obj.material as THREE.Material).dispose();
    }
    const r = (this.radius / 110) * size;
    for (const m of markers) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), new THREE.MeshBasicMaterial({ color: m.color, depthTest: false, transparent: true, opacity: 0.95 }));
      s.position.set(...m.position);
      s.renderOrder = 999;
      g.add(s);
    }
    for (const seg of segments) {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...seg.from), new THREE.Vector3(...seg.to)]);
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: seg.color, depthTest: false, transparent: true, opacity: 0.9 }));
      line.renderOrder = 998;
      g.add(line);
    }
    this.invalidate();
  }

  // ------------------------------------------------------------------ capture
  liveCloud(): LiveCloud {
    if (!this.live) {
      this.live = new LiveCloud();
      this.live.material.uniforms.uColor.value.set(this.theme.points);
      this.live.material.uniforms.uHighlight.value.set(this.theme.highlight);
      this.live.onFlush = () => this.invalidate();
      this.scene.add(this.live.points);
    }
    return this.live;
  }

  setLiveVisible(visible: boolean) {
    if (this.live) this.live.points.visible = visible;
    if (this.liveFrame) this.liveFrame.visible = visible;
    this.invalidate();
  }

  /** Markers seen in the current frame, as red dots (Revo Metro draws them the same way). */
  setLiveMarkers(points: number[][]) {
    if (!this.liveMarkers) {
      const mat = new THREE.PointsMaterial({ color: 0xff3b3b, size: 11, sizeAttenuation: false, depthTest: false,
                                             transparent: true, opacity: 0.95 });
      this.liveMarkers = new THREE.Points(new THREE.BufferGeometry(), mat);
      this.liveMarkers.frustumCulled = false;
      this.liveMarkers.renderOrder = 998;
      this.scene.add(this.liveMarkers);
    }
    this.liveMarkers.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(points.flat()), 3));
    this.liveMarkers.visible = points.length > 0;
    this.invalidate();
  }

  /** Every marker of the marker map (dimmer than the ones seen right now). */
  setMapMarkers(points: number[][]) {
    if (!this.mapMarkers) {
      const mat = new THREE.PointsMaterial({ color: 0xc0504d, size: 7, sizeAttenuation: false, depthTest: false,
                                             transparent: true, opacity: 0.75 });
      this.mapMarkers = new THREE.Points(new THREE.BufferGeometry(), mat);
      this.mapMarkers.frustumCulled = false;
      this.mapMarkers.renderOrder = 996;
      this.scene.add(this.mapMarkers);
    }
    this.mapMarkers.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(points.flat()), 3));
    this.mapMarkers.visible = points.length > 0;
    this.invalidate();
  }

  /** Put the camera where the scanner is, looking where it looks (pose: 4x4 sensor->world, row-major; the sensor
   *  frame is x right, y down, z forward). */
  viewFromScanner(pose: number[], lookDistance = 300) {
    const pos = new THREE.Vector3(pose[3], pose[7], pose[11]);
    const forward = new THREE.Vector3(pose[2], pose[6], pose[10]).normalize();
    const up = new THREE.Vector3(-pose[1], -pose[5], -pose[9]).normalize();
    const target = pos.clone().addScaledVector(forward, lookDistance);
    this.controls.animateTo(pos, lookQuaternion(forward.clone().negate(), up), target, undefined, 0);
    this.invalidate();
  }

  /** The scanner's current frame, drawn over the fused model and replaced by the next one. */
  setLiveFrame(xyz: Float32Array | null) {
    if (!xyz || !xyz.length) {
      if (this.liveFrame) this.liveFrame.visible = false;
      if (this.liveMarkers) this.liveMarkers.visible = false;
      if (this.mapMarkers) this.mapMarkers.visible = false;
      this.invalidate();
      return;
    }
    if (!this.liveFrame) {
      // the current frame reads as a light touch of the laser, not a curtain over the fused part
      const mat = new THREE.PointsMaterial({ color: new THREE.Color(this.theme.highlight), size: 1.75, sizeAttenuation: false,
                                             depthTest: false, transparent: true, opacity: 0.45 });
      this.liveFrame = new THREE.Points(new THREE.BufferGeometry(), mat);
      this.liveFrame.frustumCulled = false;
      this.liveFrame.renderOrder = 997;
      this.scene.add(this.liveFrame);
    }
    this.liveFrame.geometry.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
    this.liveFrame.geometry.computeBoundingBox();
    this.liveFrame.visible = true;
    this.invalidate();
  }

  removeLiveCloud() {
    this.setLiveFrame(null);
    if (!this.live) return;
    this.scene.remove(this.live.points);
    this.live.dispose();
    this.live = null;
    this.invalidate();
  }

  // ------------------------------------------------------------------ photo alignment
  lockAspect(aspect: number | null) {
    this.aspectLock = aspect;
    if (aspect && this.camera !== this.perspective) this.setProjection('perspective');
    this.resize();
  }

  canvasRect() {
    return { left: this.canvas.offsetLeft, top: this.canvas.offsetTop, width: this.canvas.clientWidth, height: this.canvas.clientHeight };
  }

  setFov(fov: number) {
    this.perspective.fov = fov;
    this.perspective.updateProjectionMatrix();
    this.invalidate();
  }

  /** Rotate the camera about its viewing axis. */
  roll(radians: number) {
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    this.camera.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(fwd, radians));
    this.invalidate();
  }

  captureView() {
    this.perspective.updateMatrixWorld();
    return { view_matrix: [...this.perspective.matrixWorldInverse.elements], fov: this.perspective.fov, aspect: this.perspective.aspect };
  }

  applyView(view: { view_matrix: number[]; fov: number }) {
    const world = new THREE.Matrix4().fromArray(view.view_matrix).invert();
    const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scale = new THREE.Vector3();
    world.decompose(pos, quat, scale);
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(quat);
    const dist = Math.max(this.camera.position.distanceTo(this.controls.target), this.radius);
    this.setFov(view.fov);
    this.controls.animateTo(pos, quat, pos.clone().addScaledVector(fwd, dist), undefined, 0);
  }

  // ------------------------------------------------------------------ theme
  setTheme(patch: Partial<ViewerTheme>) {
    Object.assign(this.theme, patch);
    HIGHLIGHT.set(this.theme.highlight);
    FOCUS_REST.set(this.theme.mesh);
    const gridU = (this.grid.material as THREE.ShaderMaterial).uniforms;
    gridU.uColor.value.setRGB(...this.theme.grid);
    gridU.uStrength.value = this.theme.dark ? 0.8 : 1.15;
    if (this.floorOn) gridU.uFill.value = this.theme.dark ? 0.26 : 0.16;
    (this.pivotMarker.material as THREE.MeshBasicMaterial).color.set(this.theme.highlight);
    this.scene.environmentIntensity = this.theme.dark ? 0.32 : 0.5;
    this.viewCube.setTheme(this.theme.dark);
    if (this.live) {
      this.live.material.uniforms.uColor.value.set(this.theme.points);
      this.live.material.uniforms.uHighlight.value.set(this.theme.highlight);
    }
    if (this.liveFrame) (this.liveFrame.material as THREE.PointsMaterial).color.set(this.theme.highlight);
    for (const it of this.items.values()) this.applyMaterial(it);
    this.refreshBox();
    this.invalidate();
  }

  // ------------------------------------------------------------------ regions
  /**
   * Mark the preview points / vertices inside a resolved region (docs/v3-plan.md Contract 2) on one asset (or all
   * visible ones) and return the marked counts per asset. Uses the same highlight as a lasso selection.
   */
  highlightRegion(assetId: string | null, region: ResolvedRegion): Record<string, number> {
    const tests = region.shapes.map(shapeTester);
    const counts: Record<string, number> = {};
    for (const [id, it] of this.items) {
      if ((assetId && id !== assetId) || !it.object.visible) continue;
      const pos = it.geometry.attributes.position.array as Float32Array;
      const n = it.geometry.attributes.position.count;
      const m = it.object.matrixWorld.elements;
      const sel = new Float32Array(n);
      let count = 0;
      for (let i = 0; i < n; i++) {
        const lx = pos[i * 3], ly = pos[i * 3 + 1], lz = pos[i * 3 + 2];
        const x = m[0] * lx + m[4] * ly + m[8] * lz + m[12];
        const y = m[1] * lx + m[5] * ly + m[9] * lz + m[13];
        const z = m[2] * lx + m[6] * ly + m[10] * lz + m[14];
        let inside = false;
        for (const t of tests) if (t(x, y, z)) { inside = true; break; }
        if (region.invert) inside = !inside;
        if (inside) { sel[i] = 1; count++; }
      }
      it.geometry.setAttribute('selected', new THREE.BufferAttribute(sel, 1));
      if (count) counts[id] = count;
    }
    this.invalidate();
    return counts;
  }

  // ------------------------------------------------------------------ camera helpers (assistant / buttons)
  private orbitCenter(): THREE.Vector3 {
    const box = this.visibleBox();
    return box.isEmpty() ? this.controls.target.clone() : box.getCenter(new THREE.Vector3());
  }

  /** Turn the view around the model: yaw about the up axis, pitch about the screen's horizontal axis (degrees). */
  orbit(yawDeg: number, pitchDeg: number, duration = 450) {
    const cam = this.camera;
    const center = this.orbitCenter();
    const qYaw = new THREE.Quaternion().setFromAxisAngle(this.controls.up, THREE.MathUtils.degToRad(yawDeg));
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion);
    const qPitch = new THREE.Quaternion().setFromAxisAngle(right, THREE.MathUtils.degToRad(-pitchDeg));
    const q = qYaw.multiply(qPitch);
    const pos = cam.position.clone().sub(center).applyQuaternion(q).add(center);
    const target = this.controls.target.clone().sub(center).applyQuaternion(q).add(center);
    this.controls.animateTo(pos, cam.quaternion.clone().premultiply(q), target, undefined, duration);
  }

  /** factor > 1 moves closer. */
  zoomBy(factor: number, duration = 350) {
    const cam = this.camera;
    const t = this.controls.target.clone();
    if (cam === this.orthographic) {
      this.controls.animateTo(cam.position.clone(), cam.quaternion.clone(), t, this.orthographic.zoom * factor, duration);
      return;
    }
    const pos = cam.position.clone().sub(t).multiplyScalar(1 / Math.max(factor, 1e-3)).add(t);
    this.controls.animateTo(pos, cam.quaternion.clone(), t, undefined, duration);
  }

  /**
   * Look at a world point from a given position (Measure -> Golden model: "show me this area"). radius: how much
   * around the point should fill the view (the orthographic camera's zoom; for the perspective camera the distance
   * is already in `from`).
   */
  viewFrom(target: [number, number, number], from: [number, number, number], duration = 600, radius?: number) {
    const t = new THREE.Vector3(...target);
    const eye = new THREE.Vector3(...from);
    // keep the camera's current up unless the new view looks (nearly) along it
    const dir = t.clone().sub(eye).normalize();
    let up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    if (Math.abs(up.dot(dir)) > 0.95) up = this.controls.up.clone();
    if (Math.abs(up.dot(dir)) > 0.95) up = new THREE.Vector3(1, 0, 0);
    const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(eye, t, up));
    const zoom = this.camera === this.orthographic && radius ? (this.orthographic.top - this.orthographic.bottom) / (radius * 2.3) : undefined;
    this.controls.animateTo(eye, q, t, zoom, duration);
  }

  /**
   * Spotlight one area of a mesh: vertices whose scalar `name` equals `value` keep their colours, the rest turn the
   * plain model colour (Measure -> Golden model: "Show me"). spotlight(null) ends it on every model. Returns how
   * many vertices are in the spotlight.
   */
  spotlight(assetId: string | null, name = 'check_region', value = -1): Promise<number> {
    return this.highlight(assetId, name, assetId == null ? [] : [value]);
  }

  /**
   * Highlight part of a mesh: vertices whose per-vertex scalar `name` is one of `values` keep their colours, the rest
   * fade to the plain model colour (Measure -> Golden model: hovering a bar, an area or a size; "Show me"). With
   * `glow` the highlighted vertices also breathe softly so the eye finds them. highlight(null) (or no values) fades
   * every model back. Calls may come fast (the mouse moving along a bar): the newest wins, the fade is re-targeted
   * rather than restarted, and going from one highlighted set to another keeps the rest grey and swaps the mask.
   * Resolves to the number of highlighted vertices (0 when the scalar is missing or nothing matches: then nothing is
   * greyed out).
   */
  async highlight(assetId: string | null, name = 'check_region', values: number[] = [], opts: { glow?: boolean } = {}): Promise<number> {
    const token = ++this.highlightToken;
    const it = assetId ? this.items.get(assetId) : undefined;
    if (!it || !values.length || !(it.object instanceof THREE.Mesh)) {
      this.endHighlights();
      return 0;
    }
    const n = it.geometry.attributes.position.count;
    const scalars = await this.scalarArray(it.meta.id, name, n);
    if (this.items.get(it.meta.id) !== it) return 0;
    const attr = it.geometry.attributes.focus as THREE.BufferAttribute | undefined;
    const current = token === this.highlightToken;
    // write straight into the mask the GPU already has (no new buffer per hover); a superseded call only counts. It
    // is uploaded only when something matched: otherwise the old mask stays on the GPU while the old highlight fades.
    const mask = current && attr && attr.count === n ? (attr.array as Float32Array) : null;
    const count = scalars ? maskOf(scalars, values, mask) : 0;
    if (!current) return count;
    if (!count) {
      this.endHighlights();
      return 0;
    }
    if (mask) attr!.needsUpdate = true;
    else {
      const fresh = new Float32Array(n);
      maskOf(scalars!, values, fresh);
      it.geometry.setAttribute('focus', new THREE.BufferAttribute(fresh, 1));
    }
    this.endHighlights(it);
    it.focused = true;
    this.lit = { it, name, values: [...values] };
    this.fadeFocus(it, 1);
    this.setGlow(it, !!opts.glow);
    this.invalidate();
    return count;
  }

  /**
   * Fly the camera to what the current highlight lights (the last highlight(): model + scalar + values), so a small
   * area is easy to find (Measure -> Golden model: a colour of the surface bar is clicked). It looks at their bounding
   * box's centre from the current direction, or from their mean normal when most of them face away from the camera,
   * and makes them span about 60 % of the view, but never shows less than about a quarter of the part's size (a
   * tiny area still needs its surroundings). Resolves to false when nothing is highlighted.
   */
  async frameHighlight(duration = 650): Promise<boolean> {
    const lit = this.lit;
    if (!lit || this.items.get(lit.it.meta.id) !== lit.it) return false;
    const it = lit.it;
    const n = it.geometry.attributes.position.count;
    const scalars = await this.scalarArray(it.meta.id, lit.name, n);
    if (!scalars || this.lit !== lit) return false;
    const count = maskOf(scalars, lit.values, null);
    if (!count) return false;
    const set = lit.values.length === 1 ? null : new Set(lit.values);
    const one = lit.values[0];
    const pos = it.geometry.attributes.position.array as Float32Array;
    const nor = it.geometry.attributes.normal?.array as Float32Array | undefined;
    it.object.updateMatrixWorld();
    const m = it.object.matrixWorld;
    const nm = new THREE.Matrix3().getNormalMatrix(m);
    const cam = this.camera;
    const ortho = cam === this.orthographic;
    const camPos = cam.position;
    const look = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
    // the lit vertices in world space, their mean normal and how many face the camera
    const pts = new Float32Array(count * 3);
    const p = new THREE.Vector3(), nv = new THREE.Vector3(), mean = new THREE.Vector3();
    let j = 0, facing = 0;
    for (let i = 0; i < n; i++) {
      const s = scalars[i];
      if (set ? !set.has(s) : s !== one) continue;
      p.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).applyMatrix4(m);
      pts[j * 3] = p.x;
      pts[j * 3 + 1] = p.y;
      pts[j * 3 + 2] = p.z;
      j++;
      if (nor) {
        nv.set(nor[i * 3], nor[i * 3 + 1], nor[i * 3 + 2]).applyMatrix3(nm).normalize();
        mean.add(nv);
        const toward = ortho ? -nv.dot(look) : nv.x * (camPos.x - p.x) + nv.y * (camPos.y - p.y) + nv.z * (camPos.z - p.z);
        if (toward > 0) facing++;
      } else facing++;
    }
    // the direction to look from (target -> camera): the current one, unless most of the area faces away
    const dir = new THREE.Vector3(0, 0, 1).applyQuaternion(cam.quaternion);
    if (facing < count / 2 && mean.length() > count * 0.15) dir.copy(mean).normalize();
    let up = new THREE.Vector3(0, 1, 0).applyQuaternion(cam.quaternion);
    if (Math.abs(up.dot(dir)) > 0.95) up = this.controls.up.clone();
    if (Math.abs(up.dot(dir)) > 0.95) up = new THREE.Vector3(1, 0, 0);
    const q = lookQuaternion(dir, up);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
    const upv = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
    // the area's extent on the new screen (2nd-98th percentile once there are enough vertices: a few strays must
    // not pull the view out to the whole part)
    const lo = [0, 0, 0], hi = [0, 0, 0];
    const axes = [right, upv, dir];
    const along = new Float32Array(count);
    const trim = count >= 50 ? Math.floor(count * 0.02) : 0;
    for (let a = 0; a < 3; a++) {
      const ax = axes[a];
      for (let k = 0; k < count; k++) along[k] = ax.x * pts[k * 3] + ax.y * pts[k * 3 + 1] + ax.z * pts[k * 3 + 2];
      along.sort();
      lo[a] = along[trim];
      hi[a] = along[count - 1 - trim];
    }
    const target = new THREE.Vector3()
      .addScaledVector(right, (lo[0] + hi[0]) / 2)
      .addScaledVector(upv, (lo[1] + hi[1]) / 2)
      .addScaledVector(dir, (lo[2] + hi[2]) / 2);
    const aspect = Math.max(this.container.clientWidth, 1) / Math.max(this.container.clientHeight, 1);
    const partBox = it.geometry.boundingBox!.clone().applyMatrix4(m);
    const partSize = Math.max(...partBox.getSize(new THREE.Vector3()).toArray());
    // visible height: the area spans ~60 %, but the view's short side never shows less than ~25 % of the part
    const viewH = Math.max((hi[1] - lo[1]) / 0.6, (hi[0] - lo[0]) / aspect / 0.6, (partSize * 0.25) / Math.min(1, aspect), 1e-6);
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(this.perspective.fov) / 2);
    const dist = Math.max(viewH / (2 * tanHalf), (hi[2] - lo[2]) * 1.5);
    const zoom = ortho ? (this.orthographic.top - this.orthographic.bottom) / viewH : undefined;
    this.controls.animateTo(target.clone().addScaledVector(dir, dist), q, target, zoom, duration);
    this.invalidate();
    return true;
  }

  /** Per-vertex scalar `name` of an asset (null when it has none or the length does not match), downloaded once. */
  private scalarArray(id: string, name: string, n: number): Promise<Float32Array | null> {
    const key = `${id}/${name}`;
    let p = this.scalarCache.get(key);
    if (!p) {
      p = fetch(`/api/assets/${id}/scalars/${encodeURIComponent(name)}`)
        .then(async r => (r.ok ? new Float32Array(await r.arrayBuffer()) : null))
        .catch(() => {
          this.scalarCache.delete(key); // a network hiccup: try again next time
          return null;
        });
      this.scalarCache.set(key, p);
    }
    return p.then(a => (a && a.length === n ? a : null));
  }

  /** Fade every highlighted model (but `keep`) back to its own colours. */
  private endHighlights(keep?: Item) {
    if (!keep || this.lit?.it !== keep) this.lit = null;
    for (const it of this.items.values()) {
      if (it === keep || !it.focused) continue;
      it.focused = false;
      this.fadeFocus(it, 0);
      if (this.glowItem === it) this.setGlow(it, false);
    }
    this.invalidate();
  }

  /** Re-target the grey-out of one model (anime.js picks up from wherever the running fade is). */
  private fadeFocus(it: Item, to: number) {
    const fx = it.fx;
    fx.fade?.cancel();
    fx.fade = null;
    if (fx.uFocus.value === to) return;
    if (reducedMotion()) {
      fx.uFocus.value = to;
      this.invalidate();
      return;
    }
    fx.fade = animate(fx.uFocus, {
      value: to,
      duration: FOCUS_MS,
      ease: to > fx.uFocus.value ? 'out(2)' : 'inOut(2)',
      onUpdate: () => this.invalidate(),
      onComplete: () => {
        fx.fade = null;
        this.invalidate();
      },
    });
  }

  /** Start or stop the breathing of a highlight (one model at a time; a steady lift under reduced motion). */
  private setGlow(it: Item, on: boolean) {
    const fx = it.fx;
    fx.glowFade?.cancel();
    fx.glowFade = null;
    if (on) {
      if (reducedMotion()) {
        if (this.glowItem === it) this.glowItem = null;
        fx.uGlow.value = GLOW_STEADY;
      } else if (this.glowItem !== it) {
        if (this.glowItem) this.setGlow(this.glowItem, false);
        // start from the glow's current lift so a re-highlight never jumps
        const v = Math.min(Math.max(fx.uGlow.value / GLOW_MAX, 0), 1);
        this.glowItem = it;
        this.glowStart = performance.now() - (Math.acos(1 - 2 * v) / (2 * Math.PI)) * GLOW_PERIOD_MS;
      }
      this.invalidate();
      return;
    }
    if (this.glowItem === it) this.glowItem = null;
    if (fx.uGlow.value === 0) return;
    if (reducedMotion()) {
      fx.uGlow.value = 0;
      this.invalidate();
      return;
    }
    fx.glowFade = animate(fx.uGlow, {
      value: 0,
      duration: FOCUS_MS,
      ease: 'out(2)',
      onUpdate: () => this.invalidate(),
      onComplete: () => {
        fx.glowFade = null;
      },
    });
  }

  /** Re-centre the view on a world point, keeping direction and distance. */
  lookAtPoint(p: [number, number, number], duration = 400) {
    const target = new THREE.Vector3(...p);
    const offset = target.clone().sub(this.controls.target);
    this.controls.animateTo(this.camera.position.clone().add(offset), this.camera.quaternion.clone(), target, undefined, duration);
  }

  cameraState() {
    const r = (v: THREE.Vector3) => [v.x, v.y, v.z].map(x => +x.toFixed(3));
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    return { position: r(this.camera.position), target: r(this.controls.target), up: r(up), fov: this.perspective.fov, projection: this.camera === this.orthographic ? 'orthographic' : 'perspective' };
  }

  // ------------------------------------------------------------------ thumbnails
  /** Iso view of one model rendered off screen (transparent background), as a PNG data URL. */
  renderThumbnail(id: string, size = 160): string | null {
    const it = this.items.get(id);
    if (!it) return null;
    it.geometry.computeBoundingSphere();
    const bs = it.geometry.boundingSphere;
    if (!bs || !Number.isFinite(bs.radius) || bs.radius <= 0) return null;
    const S = size * 2; // supersampled, then scaled down
    const target = new THREE.WebGLRenderTarget(S, S);
    target.texture.colorSpace = THREE.SRGBColorSpace;
    const cam = new THREE.PerspectiveCamera(30, 1, 0.01, 1e7);
    const sphere = bs.clone().applyMatrix4(it.object.matrixWorld);
    const v = this.viewDirection('iso');
    const dist = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(15))) * 1.02;
    cam.position.copy(sphere.center).addScaledVector(v.dir, dist);
    cam.up.copy(v.up);
    cam.lookAt(sphere.center);
    cam.near = dist / 100;
    cam.far = dist * 10;
    cam.updateProjectionMatrix();
    const key = new THREE.DirectionalLight(0xffffff, 1.25);
    key.position.set(-0.55, 0.8, 0.6);
    key.target.position.set(0, 0, -1);
    const fill = new THREE.DirectionalLight(0xdfe8ff, 0.35);
    fill.position.set(0.7, -0.35, 0.5);
    fill.target.position.set(0, 0, -1);
    cam.add(key, key.target, fill, fill.target);
    this.scene.add(cam);
    cam.updateMatrixWorld(true);

    const hidden: THREE.Object3D[] = [];
    for (const other of this.items.values()) if (other !== it && other.object.visible) { other.object.visible = false; hidden.push(other.object); }
    for (const o of [this.helpers, this.live?.points, this.liveFrame, this.liveMarkers, this.mapMarkers]) if (o && o.visible) { o.visible = false; hidden.push(o); }
    const wasVisible = it.object.visible;
    it.object.visible = true;
    // a thumbnail shows the whole model, even while its scan-beam reveal is still sweeping on the main view
    const prevReveal = it.fx.uReveal.value;
    it.fx.uReveal.value = 1;
    let prevScale = 0, prevOrtho = 0;
    const pointU = it.object instanceof THREE.Points ? (it.object.material as THREE.ShaderMaterial).uniforms : null;
    if (pointU) {
      prevScale = pointU.uScale.value;
      prevOrtho = pointU.uOrtho.value;
      pointU.uScale.value = S / (2 * Math.tan(THREE.MathUtils.degToRad(15)));
      pointU.uOrtho.value = 0;
    }
    const prevTarget = this.renderer.getRenderTarget();
    const prevAlpha = this.renderer.getClearAlpha();
    this.renderer.setRenderTarget(target);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear();
    this.renderer.render(this.scene, cam);
    const buf = new Uint8Array(S * S * 4);
    this.renderer.readRenderTargetPixels(target, 0, 0, S, S, buf);
    this.renderer.setRenderTarget(prevTarget);
    this.renderer.setClearColor(0x000000, prevAlpha);
    if (pointU) {
      pointU.uScale.value = prevScale;
      pointU.uOrtho.value = prevOrtho;
    }
    it.object.visible = wasVisible;
    it.fx.uReveal.value = prevReveal;
    for (const o of hidden) o.visible = true;
    this.scene.remove(cam);
    target.dispose();
    this.invalidate();

    const c = document.createElement('canvas');
    c.width = c.height = S;
    const g = c.getContext('2d')!;
    const img = g.createImageData(S, S);
    for (let y = 0; y < S; y++) img.data.set(buf.subarray((S - 1 - y) * S * 4, (S - y) * S * 4), y * S * 4);
    g.putImageData(img, 0, 0);
    const out = document.createElement('canvas');
    out.width = out.height = size;
    const og = out.getContext('2d')!;
    og.imageSmoothingQuality = 'high';
    og.drawImage(c, 0, 0, size, size);
    return out.toDataURL('image/png');
  }

  /**
   * One area of a mesh, seen from `from` looking at `target` (the model's own frame, e.g. a golden check's
   * region.view), rendered off screen as a PNG data URL with a transparent background (Measure -> Golden model: the
   * picture on an area's card). Vertices whose scalar `scalar` is one of `values` keep their colours, the rest are
   * greyed out like highlight(); without that scalar (or with no match) the whole model is in colour. With `radius`
   * the camera moves along the same line until the area fills about 70 % of the picture; otherwise it stays at
   * |from - target|. Lit and tone mapped like the main view, in the model's own colours (not the scalar colouring
   * the main view may be showing), never clipped by the section plane. It draws into its own scene, so the main view
   * (its mask, uniforms, visibility, helpers) is untouched, and it works while the main view shows other models or
   * has not loaded this one yet. Resolves to null when the model cannot be loaded or is not a mesh.
   */
  async renderAreaThumbnail(assetId: string, o: { scalar: string; values: number[]; target: Vec3; from: Vec3; radius?: number; size?: number }): Promise<string | null> {
    const size = Math.max(16, Math.round(o.size ?? 176));
    // the model: the loaded one, or (when the main view never showed it) its preview downloaded just for this
    let it = this.items.get(assetId);
    let ownGeometry: THREE.BufferGeometry | null = null;
    if (!it) {
      const geom = await fetch(`/api/assets/${assetId}/preview`)
        .then(r => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`preview ${r.status}`))))
        .then(buf => toFloat32(new PLYLoader().parse(buf)))
        .catch(() => null);
      it = this.items.get(assetId); // it may have arrived while we were downloading
      if (!it) {
        if (!geom || !geom.index) {
          geom?.dispose();
          return null;
        }
        if (!geom.attributes.normal) geom.computeVertexNormals();
        ownGeometry = geom;
      } else geom?.dispose();
    }
    if (it && !(it.object instanceof THREE.Mesh)) return null;
    const src = ownGeometry ?? it!.geometry;
    const n = src.attributes.position.count;
    const scalars = o.values.length ? await this.scalarArray(assetId, o.scalar, n) : null;
    if (it && this.items.get(assetId) !== it) return null; // forgotten meanwhile
    const zeros = new Float32Array(n);
    let mask: Float32Array | null = null;
    if (scalars) {
      mask = new Float32Array(n);
      if (!maskOf(scalars, o.values, mask)) mask = null;
    }
    // a stand-in sharing the model's buffers (no copy on the GPU) with its own mask, in the model's own colours
    const hasColor = ownGeometry ? !!ownGeometry.attributes.color : it!.hasColor;
    const colors = hasColor ? (it?.originalColors ?? src.attributes.color) : undefined;
    const geom = new THREE.BufferGeometry();
    geom.setIndex(src.index);
    geom.setAttribute('position', src.attributes.position);
    if (src.attributes.normal) geom.setAttribute('normal', src.attributes.normal);
    if (colors) geom.setAttribute('color', colors);
    geom.setAttribute('selected', new THREE.BufferAttribute(zeros, 1));
    geom.setAttribute('focus', new THREE.BufferAttribute(mask ?? zeros, 1));
    geom.computeBoundingSphere();
    const fx = newFx(mask ? 1 : 0);
    const mat = this.meshMaterial(fx);
    mat.vertexColors = !!colors;
    mat.color.set(colors ? '#ffffff' : this.theme.mesh);
    const mesh = new THREE.Mesh(geom, mat);
    const world = it ? it.object.matrixWorld.clone() : new THREE.Matrix4();
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(world);
    mesh.frustumCulled = false;
    const scene = new THREE.Scene();
    scene.environment = this.scene.environment;
    scene.environmentIntensity = this.scene.environmentIntensity;
    scene.add(mesh);

    // the camera: on the line from `target` to `from`
    const sphere = geom.boundingSphere!.clone().applyMatrix4(world);
    const fov = 30;
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(fov / 2));
    const t = new THREE.Vector3(...o.target).applyMatrix4(world);
    const f = new THREE.Vector3(...o.from).applyMatrix4(world);
    const dir = f.clone().sub(t);
    let dist = dir.length();
    if (!(dist > 1e-9) || !Number.isFinite(dist)) {
      dir.copy(this.viewDirection('iso').dir);
      dist = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(fov / 2))) * 1.02;
    }
    dir.normalize();
    if (o.radius && o.radius > 0) {
      // the area (radius) spans ~70 % of the picture's height, the rest shows what is around it
      const r = Math.max(o.radius, sphere.radius * 0.02);
      dist = r / (0.7 * tanHalf);
    }
    const cam = new THREE.PerspectiveCamera(fov, 1, Math.max(dist * 0.01, sphere.radius * 1e-4), dist + sphere.radius * 4 + t.distanceTo(sphere.center));
    cam.position.copy(t).addScaledVector(dir, dist);
    let up = this.controls.up.clone();
    if (Math.abs(up.dot(dir)) > 0.95) up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    if (Math.abs(up.dot(dir)) > 0.95) up = new THREE.Vector3(1, 0, 0);
    cam.up.copy(up);
    cam.lookAt(t);
    cam.updateProjectionMatrix();
    const key = new THREE.DirectionalLight(0xffffff, 1.25);
    key.position.set(-0.55, 0.8, 0.6);
    key.target.position.set(0, 0, -1);
    const fill = new THREE.DirectionalLight(0xdfe8ff, 0.35);
    fill.position.set(0.7, -0.35, 0.5);
    fill.target.position.set(0, 0, -1);
    cam.add(key, key.target, fill, fill.target);
    scene.add(cam);
    scene.updateMatrixWorld(true);

    // render supersampled into a float target; tone map + sRGB on the way out like the main view does on screen
    const S = size * 2;
    const target = new THREE.WebGLRenderTarget(S, S, { type: THREE.FloatType, samples: 4 });
    const prevTarget = this.renderer.getRenderTarget();
    const prevColor = this.renderer.getClearColor(new THREE.Color());
    const prevAlpha = this.renderer.getClearAlpha();
    const buf = new Float32Array(S * S * 4);
    try {
      this.renderer.setRenderTarget(target);
      this.renderer.setClearColor(0x000000, 0);
      this.renderer.clear();
      this.renderer.render(scene, cam);
      this.renderer.readRenderTargetPixels(target, 0, 0, S, S, buf);
    } finally {
      this.renderer.setRenderTarget(prevTarget);
      this.renderer.setClearColor(prevColor, prevAlpha);
      target.dispose();
      mat.dispose();
      // free only the stand-in's own buffers: disposing a geometry frees every attribute it holds on the GPU,
      // and the rest belong to the model on the main view
      for (const name of ['position', 'normal', 'color']) geom.deleteAttribute(name);
      geom.setIndex(null);
      geom.dispose();
      if (ownGeometry) ownGeometry.dispose();
      this.invalidate();
    }

    const c = document.createElement('canvas');
    c.width = c.height = S;
    const g = c.getContext('2d')!;
    const img = g.createImageData(S, S);
    const exposure = this.renderer.toneMappingExposure;
    for (let y = 0; y < S; y++) {
      const row = (S - 1 - y) * S * 4; // render targets start at the bottom row
      for (let x = 0; x < S; x++) {
        const i = row + x * 4, oi = (y * S + x) * 4;
        const a = Math.min(buf[i + 3], 1);
        if (a <= 1e-4) continue;
        // edge pixels were averaged with the transparent clear: un-premultiply before tone mapping
        toneMapSRGB(buf[i] / a, buf[i + 1] / a, buf[i + 2] / a, exposure, img.data, oi);
        img.data[oi + 3] = Math.round(a * 255);
      }
    }
    g.putImageData(img, 0, 0);
    const out = document.createElement('canvas');
    out.width = out.height = size;
    const og = out.getContext('2d')!;
    og.imageSmoothingQuality = 'high';
    og.drawImage(c, 0, 0, size, size);
    return out.toDataURL('image/png');
  }

  // ------------------------------------------------------------------ screen recording
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];

  get recording() {
    return !!this.recorder;
  }

  startRecording(fps = 30) {
    if (this.recorder) return;
    const stream = this.canvas.captureStream(fps);
    const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find(t => MediaRecorder.isTypeSupported(t));
    this.renderer.setClearColor(new THREE.Color(this.theme.bgBottom), 1); // an opaque stage: videos have no alpha
    this.recorder = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 8_000_000 } : undefined);
    this.chunks = [];
    this.recorder.ondataavailable = e => {
      if (e.data.size) this.chunks.push(e.data);
    };
    this.recorder.start(250);
    this.invalidate();
  }

  stopRecording(): Promise<Blob | null> {
    return new Promise(resolve => {
      const rec = this.recorder;
      if (!rec) return resolve(null);
      rec.onstop = () => resolve(new Blob(this.chunks, { type: 'video/webm' }));
      rec.stop();
      this.recorder = null;
      this.renderer.setClearColor(0x000000, 0);
      this.invalidate();
    });
  }

  // ------------------------------------------------------------------ misc
  screenshot(maxSize = 1280): string {
    this.dirty = true;
    this.frame(performance.now());
    const src = this.canvas;
    const scale = Math.min(1, maxSize / Math.max(src.width, src.height));
    const out = document.createElement('canvas');
    out.width = Math.round(src.width * scale);
    out.height = Math.round(src.height * scale);
    const g = out.getContext('2d')!;
    const grad = g.createRadialGradient(out.width * 0.5, out.height * 0.35, 0, out.width * 0.5, out.height * 0.5, Math.max(out.width, out.height) * 0.75);
    grad.addColorStop(0, this.theme.bgTop);
    grad.addColorStop(1, this.theme.bgBottom);
    g.fillStyle = grad;
    g.fillRect(0, 0, out.width, out.height);
    g.drawImage(src, 0, 0, out.width, out.height);
    return out.toDataURL('image/png');
  }
}

// ==================================================================== helpers
/**
 * Open3D writes PLY coordinates as doubles and PLYLoader keeps them as Float64 attributes, which WebGL cannot
 * upload. The browser only displays previews, so float32 is plenty; all measurements snap to full data server-side.
 */
function toFloat32(geom: THREE.BufferGeometry): THREE.BufferGeometry {
  for (const [name, attr] of Object.entries(geom.attributes)) {
    const a = attr as THREE.BufferAttribute;
    if (a.array instanceof Float64Array) geom.setAttribute(name, new THREE.BufferAttribute(new Float32Array(a.array), a.itemSize, a.normalized));
  }
  return geom;
}

/** Count the vertices whose scalar is one of `values`, writing 1 / 0 into `out` when given. */
function maskOf(scalars: Float32Array, values: number[], out: Float32Array | null): number {
  const n = scalars.length;
  let count = 0;
  if (values.length === 1) {
    const v = values[0];
    for (let i = 0; i < n; i++) {
      const hit = scalars[i] === v;
      if (out) out[i] = hit ? 1 : 0;
      if (hit) count++;
    }
    return count;
  }
  const set = new Set(values);
  for (let i = 0; i < n; i++) {
    const hit = set.has(scalars[i]);
    if (out) out[i] = hit ? 1 : 0;
    if (hit) count++;
  }
  return count;
}

/** NeutralToneMapping + sRGB encoding (what the renderer does on screen), for pixels read back from a float target. */
function toneMapSRGB(r: number, g: number, b: number, exposure: number, out: Uint8ClampedArray, o: number) {
  r *= exposure;
  g *= exposure;
  b *= exposure;
  const x = Math.min(r, g, b);
  const offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
  r -= offset;
  g -= offset;
  b -= offset;
  const peak = Math.max(r, g, b);
  const start = 0.8 - 0.04;
  if (peak >= start) {
    const d = 1 - start;
    const newPeak = 1 - (d * d) / (peak + d - start);
    const s = newPeak / peak;
    r *= s;
    g *= s;
    b *= s;
    const k = 1 - 1 / (0.15 * (peak - newPeak) + 1);
    r += (newPeak - r) * k;
    g += (newPeak - g) * k;
    b += (newPeak - b) * k;
  }
  const enc = (c: number) => {
    c = Math.min(Math.max(c, 0), 1);
    return 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
  };
  out[o] = enc(r);
  out[o + 1] = enc(g);
  out[o + 2] = enc(b);
}

function lookQuaternion(dirFromTarget: THREE.Vector3, up: THREE.Vector3): THREE.Quaternion {
  const m = new THREE.Matrix4().lookAt(dirFromTarget, new THREE.Vector3(0, 0, 0), up);
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

function makeGrid(): THREE.Mesh {
  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    uniforms: { uStep: { value: 10 }, uFade: { value: 200 }, uCenter: { value: new THREE.Vector3() }, uUpZ: { value: 0 }, uColor: { value: new THREE.Color(0.2, 0.18, 0.15) }, uStrength: { value: 1 }, uFill: { value: 0 }, uFloorR: { value: 100 } },
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      void main() {
        vec4 w = modelMatrix * vec4(position, 1.0);
        vWorld = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }`,
    fragmentShader: /* glsl */ `
      uniform float uStep;
      uniform float uFade;
      uniform vec3 uCenter;
      uniform float uUpZ;
      uniform vec3 uColor;
      uniform float uStrength;
      uniform float uFill;        // the Floor: a solid disc under the model (0 = grid only)
      uniform float uFloorR;      // its radius
      varying vec3 vWorld;
      float lines(vec2 p, float step) {
        vec2 g = abs(fract(p / step - 0.5) - 0.5) / fwidth(p / step);
        return 1.0 - min(min(g.x, g.y), 1.0);
      }
      void main() {
        vec3 d = vWorld - uCenter;
        vec2 p = uUpZ > 0.5 ? vWorld.xy : vWorld.xz;
        float minor = lines(p, uStep);
        float major = lines(p, uStep * 10.0);
        float fade = 1.0 - smoothstep(uFade * 0.25, uFade, length(d));
        float a = max(minor * 0.10, major * 0.26) * fade * uStrength;
        float r = length(d);
        float rim = 1.0 - smoothstep(0.0, 1.5 * fwidth(r), abs(r - uFloorR));
        a = max(a, uFill * max(1.0 - smoothstep(uFloorR * 0.985, uFloorR, r), rim * 3.0));
        if (a < 0.003) discard;
        gl_FragColor = vec4(uColor, a);
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.renderOrder = -1;
  return mesh;
}

/** Growable point buffer for live capture streams. */
export class LiveCloud {
  readonly material = createPointMaterial();
  readonly points: THREE.Points;
  count = 0;
  private capacity = 0;
  private geometry = new THREE.BufferGeometry();
  private pos = new Float32Array(0);
  private col = new Float32Array(0);
  private den = new Float32Array(0);
  spacing = 0.1;

  constructor() {
    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.ensure(200_000);
  }

  private ensure(n: number) {
    if (n <= this.capacity) return;
    let cap = Math.max(this.capacity, 200_000);
    while (cap < n) cap *= 2;
    const grow = (old: Float32Array, k: number) => {
      const a = new Float32Array(cap * k);
      a.set(old.subarray(0, this.count * k));
      return a;
    };
    this.pos = grow(this.pos, 3);
    this.col = grow(this.col, 3);
    this.den = grow(this.den, 1);
    this.capacity = cap;
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    this.geometry.setAttribute('scalar', new THREE.BufferAttribute(this.den, 1));
    this.geometry.setAttribute('selected', new THREE.BufferAttribute(new Float32Array(cap), 1)); // never highlighted
  }

  append(xyz: Float32Array, rgb?: Uint8Array, density?: Float32Array) {
    const n = xyz.length / 3;
    this.ensure(this.count + n);
    this.pos.set(xyz, this.count * 3);
    if (rgb) for (let i = 0; i < rgb.length; i++) this.col[this.count * 3 + i] = rgb[i] / 255;
    if (density) this.den.set(density, this.count);
    for (const name of ['position', 'color', 'scalar']) {
      const attr = this.geometry.attributes[name] as THREE.BufferAttribute;
      attr.addUpdateRange(this.count * attr.itemSize, n * attr.itemSize);
      attr.needsUpdate = true;
    }
    this.count += n;
    this.geometry.setDrawRange(0, this.count);
    this.material.uniforms.uHasColor.value = !!rgb;
  }

  /** Update density (and position) of already streamed points by index. */
  update(indices: Uint32Array, xyz: Float32Array, density?: Float32Array) {
    let lo = Infinity, hi = -1;
    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      if (i >= this.count) continue;
      this.pos[i * 3] = xyz[k * 3];
      this.pos[i * 3 + 1] = xyz[k * 3 + 1];
      this.pos[i * 3 + 2] = xyz[k * 3 + 2];
      if (density) this.den[i] = density[k];
      if (i < lo) lo = i;
      if (i > hi) hi = i;
    }
    if (hi < 0) return;
    this.dirtyLo = Math.min(this.dirtyLo, lo);
    this.dirtyHi = Math.max(this.dirtyHi, hi);
    if (this.flushTimer === undefined) this.flushTimer = window.setTimeout(() => this.flush(), 500);
  }

  private dirtyLo = Infinity;
  private dirtyHi = -1;
  private flushTimer: number | undefined;
  onFlush?: () => void;

  /** Upload the points changed since the last flush in one go: density updates arrive several times a second and
   *  each could span most of the cloud; re-uploading that on every message stalled rotation. */
  private flush() {
    this.flushTimer = undefined;
    if (this.dirtyHi < 0) return;
    for (const name of ['position', 'scalar']) {
      const attr = this.geometry.attributes[name] as THREE.BufferAttribute;
      attr.addUpdateRange(this.dirtyLo * attr.itemSize, (this.dirtyHi - this.dirtyLo + 1) * attr.itemSize);
      attr.needsUpdate = true;
    }
    this.dirtyLo = Infinity;
    this.dirtyHi = -1;
    this.onFlush?.();
  }

  reset() {
    this.count = 0;
    this.geometry.setDrawRange(0, 0);
  }

  style(mode: 'color' | 'density' | 'solid', lut?: { texture: THREE.DataTexture; lo: number; hi: number }) {
    const u = this.material.uniforms;
    u.uMode.value = mode === 'density' ? 3 : mode === 'color' ? 0 : 1;
    u.uHasScalar.value = mode === 'density';
    if (lut) {
      u.uLut.value = lut.texture;
      u.uRange.value.set(lut.lo, lut.hi);
    }
    u.uSize.value = this.spacing * 2;
  }

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
  }
}
