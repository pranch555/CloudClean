import * as THREE from 'three';

/*
 * The scan-beam reveal (Viewer: a model's first appearance), shared by the mesh and point shaders. A laser plane
 * sweeps along an axis in model space: what it has not reached yet is not drawn, a thin bright band in the signal
 * colour (uHighlight) flickers right at the front and a faint afterglow fades behind it. uReveal 0..1 is the plane's
 * progress; at 1, the resting value, the `uReveal < 1.0` test skips all of it (uniform branch: free once finished).
 */
export const REVEAL_VERTEX_PARS = /* glsl */ `
  uniform vec3 uRevealAxis;
  uniform float uRevealMin;
  uniform float uRevealMax;
  varying float vRevealT;`;

/** model-space `position` -> 0 where the sweep starts, 1 where it ends */
export const REVEAL_VERTEX = /* glsl */ `
  vRevealT = (dot(position, uRevealAxis) - uRevealMin) / max(uRevealMax - uRevealMin, 1e-6);`;

export const REVEAL_FRAGMENT_PARS = /* glsl */ `
  uniform float uReveal;
  uniform float uBeamWidth;
  uniform float uRevealTime;
  varying float vRevealT;
  float revealHash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  // discards what the plane has not reached; x: the hot core of the beam, y: its coloured halo, z: the afterglow
  vec3 revealBeam() {
    float w = uBeamWidth;
    float front = mix(-2.0 * w, 1.0 + 8.0 * w, uReveal);
    float d = front - vRevealT;
    if (d < 0.0) discard;
    float flicker = 0.72 + 0.28 * revealHash(floor(gl_FragCoord.xy / 2.0) + floor(uRevealTime * 40.0));
    float core = (1.0 - smoothstep(0.0, w * 0.35, d)) * flicker;
    float halo = (1.0 - smoothstep(0.0, w * 1.4, d)) * (0.55 + 0.45 * flicker);
    float after = exp(-d / (w * 3.5)) * 0.2 * (1.0 - smoothstep(0.75, 1.0, uReveal));
    return vec3(core, halo, after);
  }`;

/** fresh reveal uniforms at rest (fully drawn) */
export const revealUniforms = () => ({
  uReveal: { value: 1 },
  uRevealAxis: { value: new THREE.Vector3(1, 0, 0) },
  uRevealMin: { value: 0 },
  uRevealMax: { value: 1 },
  uBeamWidth: { value: 0.016 },
  uRevealTime: { value: 0 },
});

/*
 * Colour by height (Display -> Colour -> Height, and the live scan's "Height"): a colour ramp along the up axis
 * from the bottom to the top of what is on screen. The uniforms are shared by every material of a viewer (one
 * object each, see heightUniforms), so a new range or up axis costs one assignment, not a pass over the points.
 */
export const heightUniforms = () => {
  const blank = new THREE.DataTexture(new Uint8Array([200, 200, 200, 255]), 1, 1);
  blank.needsUpdate = true;
  return {
    uHeightLut: { value: blank as THREE.Texture },
    uHeightRange: { value: new THREE.Vector2(0, 1) },
    uUp: { value: new THREE.Vector3(0, 1, 0) },
  };
};
export type HeightUniforms = ReturnType<typeof heightUniforms>;

export const HEIGHT_PARS = /* glsl */ `
  uniform sampler2D uHeightLut;
  uniform vec2 uHeightRange;
  uniform vec3 uUp;
  vec3 heightColor(vec3 world) {
    float t = (dot(world, uUp) - uHeightRange.x) / max(uHeightRange.y - uHeightRange.x, 1e-9);
    return texture2D(uHeightLut, vec2(clamp(t, 0.0, 1.0), 0.5)).rgb;
  }`;

/** The same for meshes (Viewer.meshMaterial): uHeightOn 1 replaces the surface colour, the lighting stays. */
export const HEIGHT_MESH_VERTEX_PARS = /* glsl */ `
  varying vec3 vHeightWorld;`;
export const HEIGHT_MESH_VERTEX = /* glsl */ `
  vHeightWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;`;
export const HEIGHT_MESH_FRAGMENT_PARS = /* glsl */ `
  uniform float uHeightOn;
  varying vec3 vHeightWorld;
  ${HEIGHT_PARS}`;
export const HEIGHT_MESH_FRAGMENT = /* glsl */ `
  if (uHeightOn > 0.5) diffuseColor.rgb = heightColor(vHeightWorld);`;

/*
 * Points are round splats sized in world units (density looks right at any zoom) and lit by a headlight
 * using their normals, which makes the surface shape readable. The same material renders a pick pass that
 * writes world position into a float target (exact 3D picking for pivots, measuring and point pairs), and a
 * depth pass for the shape shading (edl.ts: view depth and splat size in pixels; points without normals get
 * their shape from it).
 */
const VERT = /* glsl */ `
  #include <clipping_planes_pars_vertex>
  ${REVEAL_VERTEX_PARS}
  attribute vec3 color;
  attribute float scalar;
  attribute float selected;
  uniform float uSize;
  uniform float uScale;
  uniform float uOrtho;       // 1 when the camera is orthographic
  uniform float uOrthoPx;     // pixels per world unit for ortho cameras
  uniform int uMode;          // 0 scan colour, 1 solid, 2 normals, 3 scalar, 4 height
  uniform vec3 uColor;
  uniform bool uHasColor;
  uniform bool uHasNormal;
  uniform bool uHasScalar;
  uniform vec2 uRange;
  uniform sampler2D uLut;
  uniform vec3 uNanColor;
  uniform int uPick;          // 1 world position (picking), 2 depth + splat size (shape shading)
  ${HEIGHT_PARS}
  varying vec3 vColor;
  varying vec3 vWorld;
  varying float vSelected;
  varying vec2 vDepth;
  void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    #include <clipping_planes_vertex>
    float px = uOrtho > 0.5 ? uSize * uOrthoPx : uSize * uScale / max(-mvPosition.z, 1e-6);
    gl_PointSize = clamp(px, 1.0, 64.0);
    vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
    vSelected = selected;
    vDepth = vec2(-mvPosition.z, gl_PointSize);
    ${REVEAL_VERTEX}
    if (uPick != 0) { vColor = vec3(0.0); return; }
    vec3 base = uColor;
    bool unlitColor = false;
    if (uMode == 0 && uHasColor) { base = color; unlitColor = true; }
    if (uMode == 3 && uHasScalar) {
      if (scalar != scalar) { base = uNanColor; }      // NaN: no value
      else {
        float span = uRange.y - uRange.x;
        if (abs(span) < 1e-12) span = 1e-12;
        float t = clamp((scalar - uRange.x) / span, 0.0, 1.0);
        base = texture2D(uLut, vec2(t, 0.5)).rgb;
      }
      unlitColor = true;
    }
    if (uMode == 4) { base = heightColor(vWorld); unlitColor = true; }
    if (uMode == 2 && uHasNormal) {
      vColor = normalize(normal) * 0.5 + 0.5;
    } else if (uHasNormal) {
      float light = abs(normalize(normalMatrix * normal).z);
      vColor = unlitColor ? base * (0.7 + 0.3 * light) : base * (0.22 + 0.78 * light);
    } else {
      vColor = base;
    }
  }`;

const FRAG = /* glsl */ `
  #include <clipping_planes_pars_fragment>
  ${REVEAL_FRAGMENT_PARS}
  uniform int uPick;
  uniform vec3 uHighlight;
  varying vec3 vColor;
  varying vec3 vWorld;
  varying float vSelected;
  varying vec2 vDepth;
  void main() {
    #include <clipping_planes_fragment>
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    // picking sees every point, revealed or not
    if (uPick == 1) { gl_FragColor = vec4(vWorld, 1.0); return; }
    // the shape shading's depth pass sees only what is drawn
    if (uPick == 2) {
      if (uReveal < 1.0) revealBeam();
      gl_FragColor = vec4(vDepth, 0.0, 1.0);
      return;
    }
    vec3 col = vSelected > 0.5 ? mix(vColor, uHighlight, 0.65) : vColor;
    if (uReveal < 1.0) {
      vec3 beam = revealBeam();
      col = mix(col, uHighlight, clamp(beam.y + beam.z, 0.0, 1.0)) + (uHighlight * 0.9 + 0.12) * beam.x;
    }
    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
  }`;

export function createPointMaterial(): THREE.ShaderMaterial {
  const blank = new THREE.DataTexture(new Uint8Array([200, 200, 200, 255]), 1, 1);
  blank.needsUpdate = true;
  return new THREE.ShaderMaterial({
    uniforms: {
      uSize: { value: 1 },
      uScale: { value: 1 },
      uOrtho: { value: 0 },
      uOrthoPx: { value: 1 },
      uMode: { value: 0 },
      uColor: { value: new THREE.Color('#c9d1dc') },
      uHasColor: { value: false },
      uHasNormal: { value: false },
      uHasScalar: { value: false },
      uRange: { value: new THREE.Vector2(0, 1) },
      uLut: { value: blank },
      uNanColor: { value: new THREE.Color('#4a505b') },
      uPick: { value: 0 },
      uHighlight: { value: new THREE.Color('#ffb547') },
      ...revealUniforms(),
      ...heightUniforms(),
    },
    vertexShader: VERT,
    fragmentShader: FRAG,
    clipping: true,
  });
}

/** Writes world position for meshes during the pick pass. */
export function createMeshPickMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: /* glsl */ `
      #include <clipping_planes_pars_vertex>
      varying vec3 vWorld;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * mvPosition;
        #include <clipping_planes_vertex>
      }`,
    fragmentShader: /* glsl */ `
      #include <clipping_planes_pars_fragment>
      varying vec3 vWorld;
      void main() {
        #include <clipping_planes_fragment>
        gl_FragColor = vec4(vWorld, 1.0);
      }`,
    side: THREE.DoubleSide,
    clipping: true,
  });
}
