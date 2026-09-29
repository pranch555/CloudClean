import type { ResolvedRegion } from './types';

type Shape = ResolvedRegion['shapes'][number];
type Tester = (x: number, y: number, z: number) => boolean;

const norm = (v: number[]) => {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
};

function inPolygon(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-12) + xi) inside = !inside;
  }
  return inside;
}

/** Point-in-shape test for the world shapes of a resolved region (docs/v3-plan.md Contract 2). */
export function shapeTester(shape: Shape): Tester {
  switch (shape.type) {
    case 'obb': {
      const [cx, cy, cz] = shape.center;
      const axes = shape.axes.map(norm);
      const [h0, h1, h2] = shape.half;
      return (x, y, z) => {
        const dx = x - cx, dy = y - cy, dz = z - cz;
        return Math.abs(dx * axes[0][0] + dy * axes[0][1] + dz * axes[0][2]) <= h0
          && Math.abs(dx * axes[1][0] + dy * axes[1][1] + dz * axes[1][2]) <= h1
          && Math.abs(dx * axes[2][0] + dy * axes[2][1] + dz * axes[2][2]) <= h2;
      };
    }
    case 'sphere': {
      const [cx, cy, cz] = shape.center;
      const r2 = shape.radius * shape.radius;
      return (x, y, z) => (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2;
    }
    case 'cylinder': {
      const [px, py, pz] = shape.point;
      const [ax, ay, az] = norm(shape.axis);
      const r2 = shape.radius * shape.radius;
      const h = shape.half_length;
      return (x, y, z) => {
        const dx = x - px, dy = y - py, dz = z - pz;
        const t = dx * ax + dy * ay + dz * az;
        if (Math.abs(t) > h) return false;
        const ex = dx - t * ax, ey = dy - t * ay, ez = dz - t * az;
        return ex * ex + ey * ey + ez * ez <= r2;
      };
    }
    case 'screen': {
      const e = shape.view_projection;
      const poly = shape.polygon;
      return (x, y, z) => {
        const w = e[3] * x + e[7] * y + e[11] * z + e[15];
        if (w <= 0) return false;
        const nx = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w;
        const ny = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w;
        return nx >= -1 && nx <= 1 && ny >= -1 && ny <= 1 && inPolygon(nx, ny, poly);
      };
    }
    default:
      return () => false;
  }
}
