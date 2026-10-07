import * as THREE from 'three';

/*
 * Shape shading for point clouds (Display -> "Shade the points (show the shape)"): eye-dome lighting, the
 * screen-space technique CloudCompare and Potree use. Points without normals (live scans, fresh captures) all get
 * the same flat colour, so a scan reads as a silhouette. Eye-dome lighting darkens each pixel by how much nearer its
 * neighbours on screen are (log depth differences over 8 neighbours): edges get an outline and slopes a shade, so the
 * shape shows at any zoom. Display only: the data is never touched.
 *
 * render() replaces renderer.render(scene, camera) for the current viewport (the whole canvas, or one side-by-side
 * pane: it follows the renderer's viewport and scissor):
 *  1. the models (`shapes`: point clouds and meshes) are drawn to the screen exactly as the plain render draws them
 *     (same pass, tone mapping and sRGB output), only without the overlays;
 *  2. a depth pass draws the same models into a float target: view depth and, for points, the splat size in pixels
 *     (points: their own material with uPick = 2; meshes: a depth material that marks them as not shaded);
 *  3. a full-screen pass multiplies the screen by the shade (blending dst * src, alpha kept) where the nearest
 *     thing is a point; empty background (also the opaque one of a recording) and meshes keep their exact colours;
 *  4. the overlays (grid, box, markers, the scanner's current frame...) are drawn on top, depth tested against 1, so
 *     they keep their colours and are never darkened.
 * Without a visible point cloud it is the plain render. Picking, thumbnails and the other off-screen renders call
 * renderer.render themselves and never come here.
 *
 * The response is normalised by what one pixel of a 45° surface spans in depth (from the projection and the
 * viewport height), so the look is the same on any screen, pixel ratio and zoom: a slope of angle a gets about
 * exp(-strength * tan a), close to a headlight on a surface facing the camera, and depth jumps get a thin dark
 * outline (from the 8 pixel neighbours, `radius` away). Slopes are measured about one splat away, so big zoomed-in
 * splats shade as a surface instead of a pattern of scales. A far point seen through a gap between near ones (sparse
 * previews, zoomed in) is shaded like the surface around it rather than as an edge, so gaps do not turn into dark
 * speckle.
 */

const FRAG = /* glsl */ `
  uniform sampler2D tDepth;
  uniform vec4 uRect;       // the viewport in target pixels: x0, y0, x1, y1 (inclusive)
  uniform float uRadius;    // the smallest neighbour distance, pixels
  uniform float uStrength;
  uniform float uFloor;     // the darkest shade
  uniform float uUnit;      // depth change across one pixel of a 45° surface (log2 units; world units in ortho)
  uniform float uOrtho;
  float depthOf(float z) { return uOrtho > 0.5 ? z : log2(max(z, 1e-12)); }
  vec2 neighbour(float r, int i) {
    float a = float(i) * 0.7854;
    vec2 q = clamp(gl_FragCoord.xy + r * vec2(cos(a), sin(a)), uRect.xy, uRect.zw);
    return texelFetch(tDepth, ivec2(q), 0).rg;
  }
  void main() {
    vec2 c = texelFetch(tDepth, ivec2(gl_FragCoord.xy), 0).rg;
    if (c.g < 0.5) discard;   // empty (0) or a mesh (-1): untouched
    float d = depthOf(c.r);

    // outlines: the 8 pixel neighbours. A jump (steeper than ~84°) up to a nearer one is an edge; a far point with
    // nearer neighbours on at least as many sides as its own surface is seen through a gap between near ones
    // (sparse or zoomed-in points): it is shaded like the surface around it (as if at their depth), not as an edge.
    float edge = 10.0 * uRadius * uUnit;
    float jumps = 0.0, level = 0.0, nearSum = 0.0;
    for (int i = 0; i < 8; i++) {
      vec2 n = neighbour(uRadius, i);
      if (n.g == 0.0) continue;
      float diff = d - depthOf(n.r);
      if (diff > edge) { jumps += 1.0; nearSum += d - diff; }
      else if (diff >= -edge) level += 1.0;
    }
    bool gap = jumps >= 2.0 && jumps >= level;
    if (gap) d = nearSum / jumps;

    // slopes: 8 neighbours about one splat away (big zoomed-in splats shade as a surface, not as scales); a plane at
    // angle a sums to about 2.5 * r * uUnit * tan(a). Jumps there belong to the outlines above (kept thin).
    float r = clamp(c.g * 0.6, uRadius, 24.0);
    float unit = r * uUnit;
    float sum = 0.0;
    for (int i = 0; i < 8; i++) {
      vec2 n = neighbour(r, i);
      float diff = d - depthOf(n.r);
      if (n.g != 0.0 && diff <= 10.0 * unit) sum += max(0.0, diff);
    }
    float shade = exp(-uStrength * sum / (2.5 * unit));
    if (!gap) shade *= 1.0 - min(jumps / 3.0, 1.0);
    gl_FragColor = vec4(vec3(mix(uFloor, 1.0, shade)), 1.0);
  }`;

export class EyeDome {
  /** how dark slopes get: a slope of angle a is shaded exp(-strength * tan a) */
  strength = 0.36;
  /** the smallest neighbour distance in CSS pixels (the outline width) */
  radius = 1.4;
  /** the darkest a pixel gets (depth jumps), so colours still read in the outlines */
  floor = 0.18;

  private target = new THREE.WebGLRenderTarget(1, 1, {
    type: THREE.FloatType,
    format: THREE.RGFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    generateMipmaps: false,
    depthBuffer: true,
  });
  private meshDepth = new THREE.ShaderMaterial({
    vertexShader: /* glsl */ `
      #include <clipping_planes_pars_vertex>
      varying float vDepth;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vDepth = -mvPosition.z;
        gl_Position = projectionMatrix * mvPosition;
        #include <clipping_planes_vertex>
      }`,
    fragmentShader: /* glsl */ `
      #include <clipping_planes_pars_fragment>
      varying float vDepth;
      void main() {
        #include <clipping_planes_fragment>
        gl_FragColor = vec4(vDepth, -1.0, 0.0, 1.0);
      }`,
    side: THREE.DoubleSide,
    clipping: true,
  });
  private shade = new THREE.ShaderMaterial({
    uniforms: {
      tDepth: { value: this.target.texture },
      uRect: { value: new THREE.Vector4() },
      uRadius: { value: 1 },
      uStrength: { value: 1 },
      uFloor: { value: 0 },
      uUnit: { value: 1 },
      uOrtho: { value: 0 },
    },
    vertexShader: 'void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: FRAG,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.ZeroFactor,
    blendDst: THREE.SrcColorFactor,
    blendSrcAlpha: THREE.ZeroFactor,
    blendDstAlpha: THREE.OneFactor,
  });
  private quadScene = new THREE.Scene();
  private quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private vp = new THREE.Vector4();
  private scissor = new THREE.Vector4();
  private px = new THREE.Vector4();
  private size = new THREE.Vector2();
  private clear = new THREE.Color();

  constructor() {
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.shade);
    quad.frustumCulled = false;
    this.quadScene.add(quad);
  }

  /**
   * Draw `scene` through `camera` into the renderer's current viewport, shading the point clouds among `shapes` (the
   * models: direct children of the scene). Every other child of the scene except cameras and lights is an overlay.
   */
  render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, shapes: THREE.Object3D[]) {
    const points = shapes.some(o => o.visible && o instanceof THREE.Points && o.geometry.drawRange.count > 0);
    if (!points) {
      renderer.render(scene, camera);
      return;
    }
    const models = new Set(shapes);
    const overlays: THREE.Object3D[] = [];
    for (const o of scene.children) {
      if (!o.visible || models.has(o) || o instanceof THREE.Camera || o instanceof THREE.Light) continue;
      o.visible = false;
      overlays.push(o);
    }
    try {
      renderer.render(scene, camera);
      this.depthPass(renderer, scene, camera, shapes);
      this.shadePass(renderer, camera);
    } finally {
      for (const o of overlays) o.visible = true;
    }
    if (!overlays.length) return;
    const hidden = shapes.filter(o => o.visible);
    for (const o of hidden) o.visible = false;
    const autoClear = renderer.autoClear;
    renderer.autoClear = false;
    try {
      renderer.render(scene, camera);
    } finally {
      renderer.autoClear = autoClear;
      for (const o of hidden) o.visible = true;
    }
  }

  private depthPass(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, shapes: THREE.Object3D[]) {
    // the same pixels as the screen viewport (three rounds viewport and scissor the same way)
    renderer.getViewport(this.vp);
    renderer.getScissor(this.scissor);
    const scissorTest = renderer.getScissorTest();
    const dpr = renderer.getPixelRatio();
    renderer.getDrawingBufferSize(this.size);
    const t = this.target;
    if (t.width !== this.size.x || t.height !== this.size.y) t.setSize(this.size.x, this.size.y);
    this.px.copy(this.vp).multiplyScalar(dpr).round();
    t.viewport.copy(this.px);
    t.scissor.copy(this.px);
    t.scissorTest = true;

    const swapped: [THREE.Mesh, THREE.Material | THREE.Material[]][] = [];
    const pointMats: THREE.ShaderMaterial[] = [];
    const hidden: THREE.Object3D[] = [];
    for (const o of shapes) {
      if (!o.visible) continue;
      if (o instanceof THREE.Points && (o.material as THREE.ShaderMaterial).uniforms?.uPick) {
        const m = o.material as THREE.ShaderMaterial;
        m.uniforms.uPick.value = 2;
        pointMats.push(m);
      } else if (o instanceof THREE.Mesh) {
        const m = (Array.isArray(o.material) ? o.material[0] : o.material) as THREE.Material & { wireframe?: boolean };
        this.meshDepth.clippingPlanes = m.clippingPlanes;
        this.meshDepth.wireframe = !!m.wireframe;
        swapped.push([o, o.material]);
        o.material = this.meshDepth;
      } else {
        o.visible = false;
        hidden.push(o);
      }
    }
    const prevTarget = renderer.getRenderTarget();
    renderer.getClearColor(this.clear);
    const clearAlpha = renderer.getClearAlpha();
    const autoClear = renderer.autoClear;
    try {
      renderer.setRenderTarget(t);
      renderer.setClearColor(0x000000, 0);
      renderer.clear(true, true, false);
      renderer.autoClear = false;
      renderer.render(scene, camera);
    } finally {
      renderer.autoClear = autoClear;
      renderer.setRenderTarget(prevTarget);
      renderer.setClearColor(this.clear, clearAlpha);
      renderer.setViewport(this.vp);
      renderer.setScissor(this.scissor);
      renderer.setScissorTest(scissorTest);
      for (const [mesh, mat] of swapped) mesh.material = mat;
      for (const m of pointMats) m.uniforms.uPick.value = 0;
      for (const o of hidden) o.visible = true;
    }
  }

  private shadePass(renderer: THREE.WebGLRenderer, camera: THREE.Camera) {
    const u = this.shade.uniforms;
    const p = this.px;
    const x1 = Math.min(p.x + p.z, this.size.x) - 1, y1 = Math.min(p.y + p.w, this.size.y) - 1;
    u.uRect.value.set(Math.max(p.x, 0), Math.max(p.y, 0), Math.max(x1, 0), Math.max(y1, 0));
    u.uRadius.value = this.radius * renderer.getPixelRatio();
    u.uStrength.value = this.strength;
    u.uFloor.value = this.floor;
    // one pixel of the viewport: an angle (perspective) or a length (orthographic), from the projection
    const pixel = 2 / (camera.projectionMatrix.elements[5] * Math.max(p.w, 1));
    const ortho = camera instanceof THREE.OrthographicCamera;
    u.uOrtho.value = ortho ? 1 : 0;
    u.uUnit.value = ortho ? pixel : pixel / Math.LN2;
    const autoClear = renderer.autoClear;
    renderer.autoClear = false;
    try {
      renderer.render(this.quadScene, this.quadCamera);
    } finally {
      renderer.autoClear = autoClear;
    }
  }

  dispose() {
    this.target.dispose();
    this.meshDepth.dispose();
    this.shade.dispose();
    (this.quadScene.children[0] as THREE.Mesh).geometry.dispose();
  }
}
