"""Do the two cross-line families measure the same surface at the same place? Static scene, consecutive frames.

For each frame, points on the dominant plane (the paper); the plane of family-0 frames vs family-1 frames, and the
nearest-neighbour distance between the two families' points on it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial import cKDTree
import stripes as S
from flatness import fit_plane

*frames, calib = sys.argv[1:]
tri = S.Triangulator.from_yaml(calib, extra=S.default_extra(calib))
by_fam = {0: [], 1: []}
for f in frames:
    img = np.load(f) if f.endswith(".npy") else S.load_raw(f)
    pts = tri.points(img)
    by_fam[tri.last["family"]].append(pts)
ref = np.concatenate(by_fam[0] + by_fam[1])
c, vt, r, inl = fit_plane(ref, gate=1.0)
n = vt[2]
for fam in (0, 1):
    P = np.concatenate(by_fam[fam])
    d = (P - c) @ n
    on = np.abs(d) < 1.0
    print(f"family {fam}: {len(by_fam[fam])} frames, {on.sum()} points on the paper, offset from the common plane "
          f"median {np.median(d[on]) * 1000:+.0f} um, spread (robust) {1.4826 * np.median(np.abs(d[on] - np.median(d[on]))) * 1000:.0f} um")
A = np.concatenate(by_fam[0]); B = np.concatenate(by_fam[1])
A = A[np.abs((A - c) @ n) < 1.0]; B = B[np.abs((B - c) @ n) < 1.0]
# the families' stripes cross; compare along the normal where they come within 0.3 mm laterally
tA = cKDTree(A[:, :2])
dd, ii = tA.query(B[:, :2], distance_upper_bound=0.3)
ok = np.isfinite(dd)
dz = ((B[ok] - A[ii[ok]]) @ n)
print(f"crossings: {ok.sum()} B points within 0.3 mm (xy) of an A point; normal offset B-A median {np.median(dz) * 1000:+.0f} um, "
      f"robust spread {1.4826 * np.median(np.abs(dz - np.median(dz))) * 1000:.0f} um")
