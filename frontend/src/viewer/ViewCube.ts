import * as THREE from 'three';

/*
 * Orientation cube drawn into a corner of the main canvas (scissor viewport). Faces, edges and corners are
 * clickable: the hit point's dominant components give the view direction (e.g. top-front-right corner).
 */
const FACE_LABELS_Y_UP = ['Right', 'Left', 'Top', 'Bottom', 'Front', 'Back']; // +X −X +Y −Y +Z −Z
const FACE_LABELS_Z_UP = ['Right', 'Left', 'Back', 'Front', 'Top', 'Bottom'];

let DARK = false;

function faceTexture(label: string, hovered = false): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d')!;
  g.fillStyle = hovered ? '#ee4b1f' : DARK ? '#232326' : '#fbfaf7';
  g.fillRect(0, 0, 128, 128);
  g.strokeStyle = DARK ? 'rgba(255,255,255,0.16)' : 'rgba(23,22,26,0.16)';
  g.lineWidth = 4;
  g.strokeRect(2, 2, 124, 124);
  g.fillStyle = hovered ? '#17161a' : DARK ? '#d8d4cc' : '#3b3934';
  g.font = '650 24px Instrument Sans Variable, Segoe UI, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(label.toUpperCase(), 64, 66);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class ViewCube {
  readonly size = 104;
  readonly margin = 14;
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  private cube: THREE.Mesh;
  private materials: THREE.MeshBasicMaterial[] = [];
  private labels: string[] = FACE_LABELS_Y_UP;
  private hoverFace = -1;
  private axes: THREE.Group;
  private edges: THREE.LineSegments;

  constructor() {
    const geo = new THREE.BoxGeometry(1, 1, 1);
    this.materials = this.labels.map(l => new THREE.MeshBasicMaterial({ map: faceTexture(l) }));
    this.cube = new THREE.Mesh(geo, this.materials);
    this.scene.add(this.cube);
    this.edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x17161a, transparent: true, opacity: 0.35 }));
    this.scene.add(this.edges);
    this.axes = new THREE.Group();
    const axis = (dir: THREE.Vector3, color: string) => {
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(-0.5, -0.5, -0.5).addScaledVector(dir, 1.25)]);
      return new THREE.Line(g, new THREE.LineBasicMaterial({ color }));
    };
    this.axes.add(axis(new THREE.Vector3(1, 0, 0), '#d9463b'), axis(new THREE.Vector3(0, 1, 0), '#2f9448'), axis(new THREE.Vector3(0, 0, 1), '#2f6fd6'));
    this.scene.add(this.axes);
  }

  setTheme(dark: boolean) {
    DARK = dark;
    (this.edges.material as THREE.LineBasicMaterial).color.set(dark ? 0xf3f1ec : 0x17161a);
    this.materials.forEach((m, i) => {
      m.map?.dispose();
      m.map = faceTexture(this.labels[i], i === this.hoverFace);
      m.needsUpdate = true;
    });
  }

  setUpAxis(up: 'y' | 'z') {
    this.labels = up === 'z' ? FACE_LABELS_Z_UP : FACE_LABELS_Y_UP;
    this.materials.forEach((m, i) => {
      m.map?.dispose();
      m.map = faceTexture(this.labels[i], i === this.hoverFace);
      m.needsUpdate = true;
    });
  }

  /** Rect of the cube inside the canvas (CSS px, origin top-left). */
  rect(canvasWidth: number) {
    return { x: canvasWidth - this.size - this.margin, y: this.margin, w: this.size, h: this.size };
  }

  render(renderer: THREE.WebGLRenderer, mainCamera: THREE.Camera, canvasWidth: number, canvasHeight: number) {
    const r = this.rect(canvasWidth);
    this.camera.position.set(0, 0, 3).applyQuaternion(mainCamera.quaternion);
    this.camera.quaternion.copy(mainCamera.quaternion);
    this.camera.zoom = 1.08;
    this.camera.updateProjectionMatrix();
    renderer.setScissorTest(true);
    renderer.setViewport(r.x, canvasHeight - r.y - r.h, r.w, r.h);
    renderer.setScissor(r.x, canvasHeight - r.y - r.h, r.w, r.h);
    renderer.autoClear = false;
    renderer.clearDepth();
    renderer.render(this.scene, this.camera);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, canvasWidth, canvasHeight);
    renderer.autoClear = true;
  }

  private hit(localX: number, localY: number) {
    const ndc = new THREE.Vector2((localX / this.size) * 2 - 1, -(localY / this.size) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    return ray.intersectObject(this.cube, false)[0] ?? null;
  }

  /** Returns true when the hover state changed (needs a redraw). */
  hover(localX: number | null, localY: number | null): boolean {
    const h = localX == null || localY == null ? null : this.hit(localX, localY);
    const face = h?.face ? h.face.materialIndex : -1;
    if (face === this.hoverFace) return false;
    const prev = this.hoverFace;
    this.hoverFace = face;
    for (const i of [prev, face]) {
      if (i < 0) continue;
      this.materials[i].map?.dispose();
      this.materials[i].map = faceTexture(this.labels[i], i === face);
      this.materials[i].needsUpdate = true;
    }
    return true;
  }

  /** Direction from target to camera for a click, or null when the cube was missed. */
  click(localX: number, localY: number): THREE.Vector3 | null {
    const h = this.hit(localX, localY);
    if (!h) return null;
    const p = h.point;
    const dir = new THREE.Vector3(Math.abs(p.x) > 0.3 ? Math.sign(p.x) : 0, Math.abs(p.y) > 0.3 ? Math.sign(p.y) : 0, Math.abs(p.z) > 0.3 ? Math.sign(p.z) : 0);
    if (dir.lengthSq() === 0 && h.face) dir.copy(h.face.normal);
    return dir.normalize();
  }
}
