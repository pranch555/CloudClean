import * as THREE from 'three';

/*
 * Camera controls with no rotation limits.
 *
 * OrbitControls clamps the polar angle, which is why a model used to "stop" when turned over the top.
 * These controls rotate the camera with quaternions about its own axes (free / trackball) or about the
 * world up axis (turntable), so the model can be tumbled in any direction indefinitely. Rotation pivots
 * around the model's centre wherever the drag starts, as Revo Metro does (or, optionally, around the point under
 * the cursor); zoom moves towards the cursor, and a release keeps a little inertia.
 */
export type RotateStyle = 'free' | 'turntable';
export type RotatePivot = 'center' | 'cursor';

export interface ControlsHost {
  camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
  canvas: HTMLCanvasElement;
  /** World point under a canvas pixel, or null (GPU pick). */
  pickPoint(clientX: number, clientY: number): THREE.Vector3 | null;
  sceneRadius(): number;
  /** Left mouse is used by a tool (selection / measure), so rotate moves to the middle button. */
  toolActive(): boolean;
  onChange(): void;
  onPivot?(point: THREE.Vector3 | null): void;
  /** The user grabbed the view (drag, wheel, pinch, double-click): e.g. stop following the scanner. */
  onInteract?(): void;
  /** Centre of what is shown (the rotation pivot in 'center' mode), or null when the scene is empty. */
  modelCenter?(): THREE.Vector3 | null;
}

const tmpV = new THREE.Vector3();
const tmpQ = new THREE.Quaternion();
const reduceMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export class FreeControls {
  target = new THREE.Vector3();
  up = new THREE.Vector3(0, 1, 0);
  style: RotateStyle = 'free';
  rotateSpeed = 0.0065;
  zoomSpeed = 1;
  pivotMode: RotatePivot = 'center';
  enabled = true;

  private pointers = new Map<number, { x: number; y: number }>();
  private action: 'rotate' | 'pan' | null = null;
  private pivot = new THREE.Vector3();
  private last = { x: 0, y: 0, t: 0 };
  private velocity = { x: 0, y: 0 };
  private pinch: { dist: number; mid: { x: number; y: number } } | null = null;
  private anim: { from: { p: THREE.Vector3; q: THREE.Quaternion; t: THREE.Vector3; zoom: number }; to: { p: THREE.Vector3; q: THREE.Quaternion; t: THREE.Vector3; zoom: number }; start: number; dur: number } | null = null;
  private dragging = false;

  /** True while a pointer drag / pinch is in progress. */
  get busy() {
    return this.dragging || this.pointers.size > 0;
  }

  constructor(private host: ControlsHost) {
    const c = host.canvas;
    c.addEventListener('pointerdown', this.onDown);
    c.addEventListener('pointermove', this.onMove);
    c.addEventListener('pointerup', this.onUp);
    c.addEventListener('pointercancel', this.onUp);
    c.addEventListener('wheel', this.onWheel, { passive: false });
    c.addEventListener('contextmenu', e => e.preventDefault());
    c.addEventListener('dblclick', this.onDoubleClick);
    c.style.touchAction = 'none';
  }

  get camera() {
    return this.host.camera;
  }

  /** Called every frame. Returns true while the view is still moving. */
  update(now: number): boolean {
    if (this.anim) {
      const k = Math.min((now - this.anim.start) / this.anim.dur, 1);
      const e = 1 - Math.pow(1 - k, 3);
      const { from, to } = this.anim;
      this.camera.position.lerpVectors(from.p, to.p, e);
      this.camera.quaternion.slerpQuaternions(from.q, to.q, e);
      this.target.lerpVectors(from.t, to.t, e);
      if ('isOrthographicCamera' in this.camera) {
        this.camera.zoom = from.zoom + (to.zoom - from.zoom) * e;
        this.camera.updateProjectionMatrix();
      }
      if (k >= 1) this.anim = null;
      this.host.onChange();
      return true;
    }
    if (!this.dragging && (Math.abs(this.velocity.x) > 0.02 || Math.abs(this.velocity.y) > 0.02)) {
      this.rotate(this.velocity.x, this.velocity.y, this.pivot);
      this.velocity.x *= 0.9;
      this.velocity.y *= 0.9;
      return true;
    }
    return false;
  }

  // ------------------------------------------------------------------ core moves
  rotate(dx: number, dy: number, pivot: THREE.Vector3) {
    const cam = this.camera;
    const angleX = -dx * this.rotateSpeed;
    const angleY = -dy * this.rotateSpeed;
    const right = tmpV.set(1, 0, 0).applyQuaternion(cam.quaternion);
    const q = new THREE.Quaternion();
    if (this.style === 'turntable') {
      q.setFromAxisAngle(this.up, angleX);
      q.multiply(tmpQ.setFromAxisAngle(right, angleY));
    } else {
      const upAxis = new THREE.Vector3(0, 1, 0).applyQuaternion(cam.quaternion);
      q.setFromAxisAngle(upAxis, angleX);
      q.multiply(tmpQ.setFromAxisAngle(right, angleY));
    }
    cam.position.sub(pivot).applyQuaternion(q).add(pivot);
    this.target.sub(pivot).applyQuaternion(q).add(pivot);
    cam.quaternion.premultiply(q).normalize();
    this.host.onChange();
  }

  pan(dx: number, dy: number) {
    const cam = this.camera;
    const el = this.host.canvas;
    let worldPerPixel: number;
    if ('isOrthographicCamera' in cam) {
      worldPerPixel = (cam.top - cam.bottom) / cam.zoom / el.clientHeight;
    } else {
      const dist = cam.position.distanceTo(this.target);
      worldPerPixel = (2 * dist * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2)) / el.clientHeight;
    }
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion).multiplyScalar(-dx * worldPerPixel);
    const upv = new THREE.Vector3(0, 1, 0).applyQuaternion(cam.quaternion).multiplyScalar(dy * worldPerPixel);
    cam.position.add(right).add(upv);
    this.target.add(right).add(upv);
    this.host.onChange();
  }

  /** factor < 1 zooms in. Keeps the world point under the cursor fixed on screen. */
  zoomAt(factor: number, clientX: number, clientY: number) {
    const cam = this.camera;
    const hit = this.host.pickPoint(clientX, clientY) ?? this.rayPointAtTargetDepth(clientX, clientY);
    if ('isOrthographicCamera' in cam) {
      const before = this.unproject(clientX, clientY, cam);
      cam.zoom = THREE.MathUtils.clamp(cam.zoom / factor, 1e-4, 1e6);
      cam.updateProjectionMatrix();
      const after = this.unproject(clientX, clientY, cam);
      const shift = before.sub(after);
      cam.position.add(shift);
      this.target.add(shift);
    } else {
      const dist = cam.position.distanceTo(hit);
      const minDist = this.host.sceneRadius() * 1e-4;
      if (factor < 1 && dist * factor < minDist) factor = minDist / Math.max(dist, 1e-12);
      cam.position.sub(hit).multiplyScalar(factor).add(hit);
      this.target.sub(hit).multiplyScalar(factor).add(hit);
    }
    this.host.onChange();
  }

  animateTo(position: THREE.Vector3, quaternion: THREE.Quaternion, target: THREE.Vector3, zoom?: number, duration = 380) {
    const cam = this.camera;
    const ortho = 'isOrthographicCamera' in cam;
    const fromZoom = ortho ? (cam as THREE.OrthographicCamera).zoom : 1;
    this.velocity = { x: 0, y: 0 };
    const to = { p: position.clone(), q: quaternion.clone(), t: target.clone(), zoom: zoom ?? fromZoom };
    if (reduceMotion() || duration <= 0) {
      cam.position.copy(to.p);
      cam.quaternion.copy(to.q);
      this.target.copy(to.t);
      if (ortho) {
        (cam as THREE.OrthographicCamera).zoom = to.zoom;
        cam.updateProjectionMatrix();
      }
      this.host.onChange();
      return;
    }
    this.anim = { from: { p: cam.position.clone(), q: cam.quaternion.clone(), t: this.target.clone(), zoom: fromZoom }, to, start: performance.now(), dur: duration };
  }

  stop() {
    this.anim = null;
    this.velocity = { x: 0, y: 0 };
  }

  // ------------------------------------------------------------------ helpers
  private ndc(clientX: number, clientY: number) {
    const r = this.host.canvas.getBoundingClientRect();
    return new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
  }

  private unproject(clientX: number, clientY: number, cam: THREE.Camera) {
    const n = this.ndc(clientX, clientY);
    return new THREE.Vector3(n.x, n.y, 0).unproject(cam);
  }

  private rayPointAtTargetDepth(clientX: number, clientY: number) {
    const cam = this.camera;
    const ray = new THREE.Raycaster();
    ray.setFromCamera(this.ndc(clientX, clientY), cam);
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(forward, this.target);
    return ray.ray.intersectPlane(plane, new THREE.Vector3()) ?? this.target.clone();
  }

  // ------------------------------------------------------------------ events
  private onDown = (e: PointerEvent) => {
    if (!this.enabled) return;
    this.anim = null;
    this.host.onInteract?.();
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    this.host.canvas.setPointerCapture(e.pointerId);
    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      this.pinch = { dist: Math.hypot(a.x - b.x, a.y - b.y), mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } };
      this.action = null;
      return;
    }
    const tool = this.host.toolActive();
    const leftRotates = !tool && e.button === 0 && !e.shiftKey;
    const middleRotates = tool && e.button === 1;
    if (leftRotates || middleRotates) this.action = 'rotate';
    else if (e.button === 2 || (e.button === 1 && !tool) || (e.button === 0 && e.shiftKey && !tool)) this.action = 'pan';
    else return;
    this.dragging = true;
    this.velocity = { x: 0, y: 0 };
    this.last = { x: e.clientX, y: e.clientY, t: performance.now() };
    if (this.action === 'rotate') {
      const hit = this.pivotMode === 'cursor' ? this.host.pickPoint(e.clientX, e.clientY)
        : this.host.modelCenter?.() ?? null;
      this.pivot.copy(hit ?? this.target);
      this.host.onPivot?.(this.pivot.clone());
    }
  };

  private onMove = (e: PointerEvent) => {
    if (!this.pointers.has(e.pointerId)) return;
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.pinch && this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      this.pan(mid.x - this.pinch.mid.x, mid.y - this.pinch.mid.y);
      if (dist > 0 && this.pinch.dist > 0) this.zoomAt(this.pinch.dist / dist, mid.x, mid.y);
      this.pinch = { dist, mid };
      return;
    }
    if (!this.action) return;
    const dx = e.clientX - this.last.x;
    const dy = e.clientY - this.last.y;
    const now = performance.now();
    if (this.action === 'rotate') {
      this.rotate(dx, dy, this.pivot);
      const dt = Math.max(now - this.last.t, 1);
      this.velocity = { x: (dx / dt) * 16, y: (dy / dt) * 16 };
    } else this.pan(dx, dy);
    this.last = { x: e.clientX, y: e.clientY, t: now };
  };

  private onUp = (e: PointerEvent) => {
    this.pointers.delete(e.pointerId);
    if (this.pointers.size < 2) this.pinch = null;
    if (this.pointers.size === 0) {
      this.dragging = false;
      // inertia only after a flick, never after a slow deliberate drag
      if (this.action !== 'rotate' || performance.now() - this.last.t > 60 || reduceMotion()) this.velocity = { x: 0, y: 0 };
      this.action = null;
      this.host.onPivot?.(null);
    }
  };

  private onWheel = (e: WheelEvent) => {
    if (!this.enabled) return;
    e.preventDefault();
    this.anim = null;
    this.host.onInteract?.();
    const delta = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    const factor = Math.pow(0.9985, -delta * this.zoomSpeed);
    this.zoomAt(factor, e.clientX, e.clientY);
  };

  /** Double click re-centres the orbit target on the clicked surface point. */
  private onDoubleClick = (e: MouseEvent) => {
    this.host.onInteract?.();
    const hit = this.host.pickPoint(e.clientX, e.clientY);
    if (!hit) return;
    const cam = this.camera;
    const offset = hit.clone().sub(this.target);
    this.animateTo(cam.position.clone().add(offset), cam.quaternion.clone(), hit, undefined, 260);
  };
}
