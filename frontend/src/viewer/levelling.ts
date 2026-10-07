import * as THREE from 'three';

/*
 * Standing a model on the floor (the Floor tool). Pure geometry on the positions the viewer already holds (they
 * are the model's own coordinates), so the turn previewed in the 3D view is exactly the one that is saved.
 */

export interface FlatArea {
  /** unit normal of the flat area, pointing to the side where most of the model is (so the model stands above it) */
  up: THREE.Vector3;
  /** how flat the area is: RMS distance of its points from the fitted plane, relative to the area's radius */
  flatness: number;
  radius: number;
  count: number;
}

export const FLAT_ENOUGH = 0.06;   // RMS / radius: a flat face of a scan is ~0.01-0.03, a curved one far more
const MIN_POINTS = 12;

/** Plane through the points within `radius` of p, or null when too few points. */
function planeNear(pos: ArrayLike<number>, p: THREE.Vector3, radius: number): { normal: THREE.Vector3; rms: number; count: number } | null {
  const r2 = radius * radius;
  let n = 0, sx = 0, sy = 0, sz = 0;
  const idx: number[] = [];
  for (let i = 0; i < pos.length; i += 3) {
    const dx = pos[i] - p.x, dy = pos[i + 1] - p.y, dz = pos[i + 2] - p.z;
    if (dx * dx + dy * dy + dz * dz > r2) continue;
    idx.push(i);
    sx += pos[i]; sy += pos[i + 1]; sz += pos[i + 2];
    n++;
  }
  if (n < MIN_POINTS) return null;
  const cx = sx / n, cy = sy / n, cz = sz / n;
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (const i of idx) {
    const x = pos[i] - cx, y = pos[i + 1] - cy, z = pos[i + 2] - cz;
    xx += x * x; xy += x * y; xz += x * z; yy += y * y; yz += y * z; zz += z * z;
  }
  const { vector, value } = smallestEigen([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]);
  return { normal: vector, rms: Math.sqrt(Math.max(value, 0) / n), count: n };
}

/** Eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix (cyclic Jacobi). */
function smallestEigen(m: number[][]): { vector: THREE.Vector3; value: number } {
  const a = m.map(r => [...r]);
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 32; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off <= 1e-14 * (Math.abs(a[0][0]) + Math.abs(a[1][1]) + Math.abs(a[2][2])) || off === 0) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      if (a[p][q] === 0) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  let k = 0;
  if (a[1][1] < a[k][k]) k = 1;
  if (a[2][2] < a[k][k]) k = 2;
  return { vector: new THREE.Vector3(v[0][k], v[1][k], v[2][k]).normalize(), value: a[k][k] };
}

/**
 * The flat area of the model around point p (model coordinates): the widest patch around it that is still flat,
 * so a click anywhere on a face uses as much of the face as it can. Its `up` points to the side most of the model
 * is on: clicking the underside of a part lays it on that face; clicking the top of a base (the rest of the part
 * above it) keeps the base level and the part upright. null when there are too few points near p.
 */
export function flatAreaAt(pos: ArrayLike<number>, p: THREE.Vector3, diagonal: number): FlatArea | null {
  let best: FlatArea | null = null;
  let narrowest: FlatArea | null = null;
  for (const f of [0.006, 0.012, 0.025, 0.05, 0.1]) {
    const radius = diagonal * f;
    const fit = planeNear(pos, p, radius);
    if (!fit) continue;
    const area = { up: fit.normal, flatness: fit.rms / radius, radius, count: fit.count };
    narrowest ??= area;
    if (area.flatness <= FLAT_ENOUGH) best = area;
    else if (best) break;                       // the face ends here: a wider patch takes in its edges
  }
  const area = best ?? narrowest;
  if (!area) return null;
  // most of the model above the plane: count the sides on a sample of at most ~200k points
  const stride = Math.max(1, Math.floor(pos.length / 3 / 200_000)) * 3;
  const tol = area.radius * 0.05;
  let above = 0, below = 0;
  for (let i = 0; i < pos.length; i += stride) {
    const d = (pos[i] - p.x) * area.up.x + (pos[i + 1] - p.y) * area.up.y + (pos[i + 2] - p.z) * area.up.z;
    if (d > tol) above++;
    else if (d < -tol) below++;
  }
  if (below > above) area.up.negate();
  return area;
}

/** The turn q followed by the smallest turn that makes `upInModel` (model coordinates) point along worldUp. */
export function levelTurn(q: THREE.Quaternion, upInModel: THREE.Vector3, worldUp: THREE.Vector3): THREE.Quaternion {
  const shown = upInModel.clone().applyQuaternion(q).normalize();
  return new THREE.Quaternion().setFromUnitVectors(shown, worldUp).multiply(q).normalize();
}

/** The model shown turned by q about its centre c: T(c) R(q) T(-c). */
export function placementMatrix(q: THREE.Quaternion, c: THREE.Vector3): THREE.Matrix4 {
  return new THREE.Matrix4().makeTranslation(c.x, c.y, c.z)
    .multiply(new THREE.Matrix4().makeRotationFromQuaternion(q))
    .multiply(new THREE.Matrix4().makeTranslation(-c.x, -c.y, -c.z));
}

/** Row-major 4x4 for the backend's `transform` edit. */
export function rowMajor(m: THREE.Matrix4): number[][] {
  const e = m.elements;   // column-major
  return [0, 1, 2, 3].map(r => [0, 1, 2, 3].map(c => +e[c * 4 + r].toFixed(12)));
}
