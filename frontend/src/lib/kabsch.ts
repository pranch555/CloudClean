import * as THREE from 'three';

/**
 * Best-fit rigid transform from matching point pairs (Horn's quaternion method), used only to *preview* how a
 * manual point-pair merge will look. The server recomputes the exact transform when the merge runs.
 */
export function rigidFromPairs(source: number[][], target: number[][]): THREE.Matrix4 | null {
  const n = Math.min(source.length, target.length);
  if (n < 3) return null;
  const ca = [0, 0, 0], cb = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      ca[k] += source[i][k] / n;
      cb[k] += target[i][k] / n;
    }
  }
  // 3x3 covariance
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    const a = [source[i][0] - ca[0], source[i][1] - ca[1], source[i][2] - ca[2]];
    const b = [target[i][0] - cb[0], target[i][1] - cb[1], target[i][2] - cb[2]];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r][c] += a[r] * b[c];
  }
  // symmetric 4x4 whose largest eigenvector is the optimal rotation quaternion (w, x, y, z)
  const [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = S;
  const N = [
    [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
    [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
    [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
    [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
  ];
  const { vectors, values } = jacobiEigen(N);
  let best = 0;
  for (let i = 1; i < 4; i++) if (values[i] > values[best]) best = i;
  const q = new THREE.Quaternion(vectors[1][best], vectors[2][best], vectors[3][best], vectors[0][best]).normalize();
  const m = new THREE.Matrix4().makeRotationFromQuaternion(q);
  const centre = new THREE.Vector3(ca[0], ca[1], ca[2]).applyMatrix4(m);
  m.setPosition(cb[0] - centre.x, cb[1] - centre.y, cb[2] - centre.z);
  return m;
}

/** Eigen decomposition of a small symmetric matrix (cyclic Jacobi). Returns column eigenvectors. */
function jacobiEigen(input: number[][], sweeps = 24): { vectors: number[][]; values: number[] } {
  const n = input.length;
  const a = input.map(row => row.slice());
  const v: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-18) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { vectors: v, values: a.map((row, i) => row[i]) };
}
