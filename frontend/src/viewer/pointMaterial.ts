import * as THREE from 'three';

/*
 * Points are round splats sized in world units (density looks right at any zoom) and lit by a headlight
 * using their normals, which makes the surface shape readable. The same material renders a pick pass that
 * writes world position into a float target (exact 3D picking for pivots, measuring and point pairs).
 */
const VERT = /* glsl */ `
  #include <clipping_planes_pars_vertex>
  attribute vec3 color;
  attribute float scalar;
  attribute float selected;
  uniform float uSize;
  uniform float uScale;
  uniform float uOrtho;       // 1 when the camera is orthographic
  uniform float uOrthoPx;     // pixels per world unit for ortho cameras
  uniform int uMode;          // 0 scan colour, 1 solid, 2 normals, 3 scalar
  uniform vec3 uColor;
  uniform bool uHasColor;
  uniform bool uHasNormal;
  uniform bool uHasScalar;
  uniform vec2 uRange;
  uniform sampler2D uLut;
  uniform vec3 uNanColor;
  uniform int uPick;
  varying vec3 vColor;
  varying vec3 vWorld;
  varying float vSelected;
  void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    #include <clipping_planes_vertex>
    float px = uOrtho > 0.5 ? uSize * uOrthoPx : uSize * uScale / max(-mvPosition.z, 1e-6);
    gl_PointSize = clamp(px, 1.0, 64.0);
    vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
    vSelected = selected;
    if (uPick == 1) { vColor = vec3(0.0); return; }
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
  uniform int uPick;
  uniform vec3 uHighlight;
  varying vec3 vColor;
  varying vec3 vWorld;
  varying float vSelected;
  void main() {
    #include <clipping_planes_fragment>
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    if (uPick == 1) { gl_FragColor = vec4(vWorld, 1.0); return; }
    vec3 col = vSelected > 0.5 ? mix(vColor, uHighlight, 0.65) : vColor;
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
