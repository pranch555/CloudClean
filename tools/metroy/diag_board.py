"""Where do a frame's points lie relative to the dominant plane? Histogram of signed distances (mm)."""
import sys
import numpy as np
from flatness import fit_plane
from stripes import Triangulator, load_raw
from laser_scan import scan
tri = Triangulator.from_yaml(sys.argv[2])
for name, pts in (("steger", tri.points(load_raw(sys.argv[1]))), ("v1", scan(sys.argv[1], sys.argv[2])[0])):
    c, vt, r, inl = fit_plane(pts)
    print(f"{name}: {len(pts)} pts, within 0.2 mm {np.mean(np.abs(r)<0.2):.1%}, 1 mm {np.mean(np.abs(r)<1):.1%}, z median {np.median(pts[:,2]):.0f}")
    h, e = np.histogram(r, bins=[-1e9,-50,-20,-10,-5,-2,-1,-0.2,0.2,1,2,5,10,20,50,1e9])
    print("   ", " ".join(f"[{e[i]:g},{e[i+1]:g}):{h[i]}" for i in range(len(h))))
