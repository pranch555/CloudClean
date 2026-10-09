# How marker-tracked laser-line scanners build a point cloud

Research of 2026-10-09 for the native MetroY pipeline (`cloudclean/capture/metroy/`), after the user's turntable
scans failed (docs/metroy-handoff.md, "Turntable + markers on the user's plate"). **[S]** = a source says so,
**[I]** = our inference. Sources are linked at each claim.

## The pipeline every vendor shares

1. **Each frame:** find the laser stripe centres in both cameras, pair them, triangulate a few thousand points. This
   gives *where the surface is relative to the scanner*. Ours does this well (flat board 31-40 um RMS).
2. **Find the retro-reflective markers** in the same frame, pair them across the cameras, triangulate them. This gives
   *where the scanner is*: the markers do not move, so the rigid motion that puts this frame's markers onto the map is
   the scanner's pose. 3 markers fix a pose mathematically; vendors want 4-5 in every frame.
3. **Register the frame** to the marker map (built while scanning, or first in a marker-only "global marker" scan,
   optimised by bundle adjustment and frozen).
4. **Fuse:** move the frame's points by that pose into the part's coordinates and **average** them into a volume
   (signed distance / vector field), so every spot of the surface is the mean of many frames, not a stack of them.

A frame whose pose is unknown is thrown away - never placed by guess. If a pose is wrong by 0.2 mm, the surface comes
out doubled by 0.2 mm; if frames keep the same pose while the part turns, they pile on one spot.

## 1. Marker centres

- [S] Creaform finds targets by "centroid or ellipse fitting" ([US7912673](https://patents.google.com/patent/US7912673));
  later by fitting an ellipse to the undistorted contour, ring light set to *slightly* saturate, and the target's normal
  chosen from the scan surface around it ([US9816809](https://patents.google.com/patent/US9816809)).
- [S] Centroids are "very sensitive to pattern artefacts (e.g. occlusions or dirt)"; contour ellipse fits can reject
  disturbed edge segments; 1/50-1/100 px is reachable; ellipse eccentricity shifts X/Y but barely Z in rectified
  stereo ([Luhmann 2014](https://isprs-archives.copernicus.org/articles/XL-5/363/2014/isprsarchives-XL-5-363-2014.pdf)).
  Gradient-based dual-conic estimator: [Ouellet & Hebert](https://doi.org/10.1007/s00138-008-0141-3).
- [S] MetroY takes **6 mm** markers (10 mm outer) only; mixing sizes "may affect the scan's accuracy"
  ([Revopedia FAQ](https://revopedia.revopoint3d.com/en/FAQ/3UsingAccessories)). Shining 3D: hold the scanner close to
  90 deg to the markers so they show as circles
  ([support](https://support.shining3d.com/en/support/solutions/articles/60001495728-preparation-for-laser-scanning)).
- [I] Ours: intensity-weighted centroid after threshold + opening. Measured 0.16 mm RMS per marker - about 0.34 px per
  image, ~7x worse than the geometry allows (~0.025 mm). The laser lines cross the markers (red inside the camera
  view's rings) and drag the centroid. **Fix: mask stripe pixels inside the blob and fit an ellipse to the sub-pixel
  edge, dropping edge pieces the stripe touches.** First measure: 100 static frames, per-marker 2D centre spread and
  left/right row difference.

## 2. Pairing markers across the cameras

- [S] Creaform pairs targets by the epipolar constraint and can also solve the pose from the 2D image positions against
  the map directly (Levenberg-Marquardt on reprojection), so a marker seen by one camera still counts
  ([US7912673](https://patents.google.com/patent/US7912673)). No vendor publishes its epipolar tolerance.
- [S] On the MetroX, Revo Metro's scanning-distance setting also limited marker detection; widening it to 200-400 mm
  "very much improved" marker detection ([forum](https://forum.revopoint3d.com/t/metrox-global-marker-scanning-and-marker-detection-issues/36930)).
- [I] Ours keeps only pairs that are unique on their image row within a 150-500 mm depth window (1,087 px of
  disparity). On a plate seen from the side, markers at one distance share a row: ~2.5 markers a frame are dropped as
  ambiguous. **Fix: map-guided pairing** - predict each map marker in both images and accept the candidate that lands
  on it; global assignment (row residual + size + shape) instead of dropping every ambiguous pair; later, pose from
  2D reprojection so one-camera markers help.

## 3. The marker map

- [S] Minimum markers in common: Creaform 3 (otherwise "both positioning features and surface points are discarded"),
  Shining 3D 4 ([OptimScan](https://docs.shining3d.com/optimscan-q12/1.0.8/en-us/scanning/)), Revopoint 5 per frame
  ([Revo Metro manual](https://revopedia.revopoint3d.com/en/Revometro/Usermanual)).
- [S] Layout: Revopoint 6-7 cm apart, irregular, flat spots, extra at edges; closer than 3 cm "will jump and confuse
  the sensors". Shining 3D: "evenly and randomly", "avoid symmetry". HandySCAN guides: never in a straight line.
- [S] Map matching = the largest set of marker pairs whose distances agree within the sensor's accuracy; new markers
  join, re-seen ones are averaged ([US7912673](https://patents.google.com/patent/US7912673)). A recovered pose is only
  valid if no other start leads to a different good pose ([US9325974](https://patents.google.com/patent/US9325974)).
  Frames are not added until the pose is reliable (pose covariance, a 10-of-20 frame filter)
  ([US10401142](https://patents.google.com/patent/US10401142)).
- [S] Everyone recommends **markers first**: scan only the markers, optimise the map (bundle adjustment), freeze it,
  then scan the surface (Revo Global Marker; FreeScan "Generate GMF"; Creaform "Optimize Model").
- [I] Ours already has map-first + freeze (`MarkerTracker.freeze`, averaging refine). Fixed 2026-10-09: a map started
  on few markers could never grow. Open: a robust reprojection bundle adjustment before freezing; a reliability test
  (inliers spread, not in a line) before frames are fused.

## 4. Turntable

- [S] Revopoint's answer for small parts: the dual-axis turntable, marker blocks, or markers around the part, with
  "at least five markers ... facing the scanner directly at each scanning angle"
  ([FAQ](https://revopedia.revopoint3d.com/en/FAQ/3UsingAccessories)). Laser modes require markers.
- [S] EinScan calibrates the turntable axis and reuses it only if nothing moved
  ([docs](https://docs.shining3d.com/einscan/mac/3.2.0/sesp2/en-us/scanparameters/)). Turntable views differ by one angle
  about one axis ([Kosaka, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/IMW/papers/Kosaka_Turntable-Constrained_Camera_Pose_Estimation_CVPRW_2026_paper.pdf)).
- [S] No vendor documents using the turntable's own angle as a prior in marker mode.
- [I] The user's plate has an irregular field of ~25 markers (good). But a scanner looking across the plate sees the
  far markers at a grazing angle: dim and flat. **Look down onto the plate at a steeper angle** (as on 2026-10-06, when
  the scan worked); marker blocks standing up on the plate would give markers that face the scanner. Later: fit the
  table axis from the marker tracks and estimate only the angle per frame (1 degree of freedom, works with 1-2
  markers, the Bluetooth angle as the prior); skip frames while the table stands still.

## 5. Fusion - why the surface should average, not stack

- [S] Creaform/Laval fuse laser curves into a voxel vector field (tangent covariance, distance vector, weight per
  voxel), register new curves to the field, and extract the surface with Marching Cubes
  ([US7487063](https://patents.google.com/patent/US7487063), [CVPR 2003](https://mlanthology.org/cvpr/2003/tubic2003cvpr-d/)).
  Curless & Levoy's weighted signed distance (built on a laser-stripe scanner) weights by view angle and fades near
  edges ([SIGGRAPH 96](https://graphics.stanford.edu/papers/volrange/paper_1_level/paper.html)).
- [S] Revo Metro: a target point distance while scanning, Fast / High-quality fusion, and fusing finer than the scan
  "would introduce artifacts" ([forum](https://forum.revopoint3d.com/t/revo-metro-5-8-7-scanning-point-distance-vs-fusion-point-distance/43542));
  tools to remove markers' holes, isolated points and overlap layers.
- [I] Ours keeps up to 3 points per 0.2 mm cell: every frame's layer survives, so any pose error becomes a second
  surface. **Fix: average** - a sparse weighted TSDF / vector field at 0.1-0.2 mm, weights from incidence angle and
  stripe quality, normals from the two crossing line families; drop points on markers, isolated clusters and spots
  seen from only one angle.

## 6. MetroY Ultra specs ([product page](https://www.revopoint3d.com/products/metroy-ultra-metrology-solution))

Single-frame precision 0.01 mm, accuracy 0.015 mm, volumetric 0.015 + 0.04 mm x L(m); point distance down to
0.05 mm; 200-400 mm working distance; 90 fps; 34 cross / 15 parallel / 1 line; 500 x 6 mm markers. 10 min warm-up
before accuracy work. Exported coordinates are the left camera after rectification (same as ours).

## Ranked changes for CloudClean

| # | change | step | status |
|---|---|---|---|
| 1 | tracker: new markers are not evidence against a pose (map started small never grew) | map | **done 2026-10-09** |
| 2 | detect the far, dim, flat markers (25 above background, axis 0.25) | detection | **done 2026-10-09** |
| 3 | marker centres: mask stripes, sub-pixel ellipse edge fit (0.16 -> ~0.03-0.05 mm) | detection | open - measure static noise first |
| 4 | map-guided pairing + global assignment; later 2D reprojection pose | pairing / tracking | open |
| 5 | averaging fusion (weighted TSDF / vector field) instead of keeping 3 points a cell | fusion | open - the biggest change to "fills the right spots" |
| 6 | turntable: axis from marker tracks, 1-DOF angle with the Bluetooth prior, skip frames while still | turntable | open |
| 7 | reliability before fusing (>= 4-5 inliers, spread, hysteresis); bundle adjustment before freezing a map | map | open |
| 8 | clean-up: points on markers, isolated clusters, one-angle-only spots | fusion | open |
