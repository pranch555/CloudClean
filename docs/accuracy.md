# Accuracy checks (Contract 5)

`cloudclean/accuracy.py` answers "is a dimension off because of the scanner or the software?". Nothing in it changes
geometry: every function measures and reports. The web layer is `cloudclean/web/routes_accuracy.py` plus the job
kind `compare_scans` in `cloudclean/web/jobs_accuracy.py`. Tests are in `tests/test_accuracy.py`. The
investigation that motivated it is in `docs/accuracy-investigation-2026-09-24.md`.

Units are scan units (mm). Signed distances are **+ outside** the reference surface (material added). Fits are
robust (Tukey IRLS, MAD scale), so stray points do not pull them.

## API

### `POST /api/accuracy/drift {asset_id, parent_id?, tolerance = 0.01}`

How far the operation that made an asset moved or resized its parent's surface. The parent's pose comes from the
operation's report:

- `merge`: `report.scans[i].transform`
- `edit` / `compare`: `report.transform`
- `compare_scans`: pose of B

The pose is checked for rigidity and applied first. Response:

```
{asset_id, asset_name, parent_id, parent_name, operation,
 verdict: "unchanged" | "moved" | "changed", sentence,
 displacement: {signed_mean, mean, rms, p95, max, median, median_abs, p05, p95_signed, std, count},
 moved_points: {...} | null,          points that are not the parent's own (merges: the other scan)
 identical_fraction, kept_fraction,   clouds: share of points that are exactly parent points / still present
 removed_surface_points, removed_stray_points,   subsets (clean, crop): what the removed points were
 unsupported_fraction,                surface with no parent data nearby (new coverage, watertight fill)
 reverse: {...} | null,               meshes: the parent's points against the mesh (+ = mesh outside)
 transform_check: {determinant, orthonormality_error, singular_values, scale_ppm, rotation_deg, translation,
                   rigid, mirrored, identity} | null,
 dimensions_before, dimensions_after, dimension_change: {length, width, height},
 noise, band_width, tolerance, points: {parent, child},
 parents: [one entry like the above per parent scan]}   the top level = the first parent
```

**Verdict:**

- **unchanged:** no systematic movement or size change beyond `tolerance`. Random motion within 3× the parent's
  noise is not a change.
- **moved:** a rigid pose change only.
- **changed:** anything else. A non-rigid transform is always `changed`.

**How sizes are compared:**

- A subset (points removed, none moved) is compared with its parent minus the stray points, using trimmed extents
  (the same points before and after).
- Anything else uses `band_extents`: the median position of the outermost band of surface at each end. This does
  not grow with noise, so a smooth mesh and the noisy cloud it came from measure the same.
- Results are cached per (asset, parent, file time).
- Typical time: 6 s for a 750k-point clean, 20 s for a 3.7M-triangle mesh.
- Errors: 400 when the asset was not made from a scan, 404 for an unknown asset.

### `POST /api/accuracy/compare-scans {a_id, b_id, transform?, tolerance = 0.02, save_aligned = true, name?}`

Submits the job **`compare_scans`**, which compares two scans of the same part. That gives the scanner's
repeatability, independent of any processing.

**Alignment.** Scan B is aligned to A by `transform` (a rigid 4×4 B→A, e.g. an approved merge pose; a non-rigid
one gets a 400) or by `align_scans`, then refined against A's surface.

**Job output `comparison`:**

```
{verdict: "consistent" | "differ" | "uncertain", sentence, tolerance,
 scale_ppm, scale_se_ppm,             one uniform scale, B relative to A (+ = B larger)
 scale_with_offset_ppm, offset_mm,    scale fitted together with a uniform thickness offset (+ = B outside A)
 axis_scale_ppm: {length, width, height}, axis_fit_offset_mm,   one scale per part axis of A, plus the offset
 length_difference_from_scale_mm,
 separation: {...stats}, overlap_fraction, histogram,   B's points vs A's surface where both face the same way
 noise: {a, b, combined}, spacing,
 extents: {a, b, difference}, extents_common: {a, b, difference} | null,   robust extents in A's part frame
 alignment: {best_candidate, ambiguous, fine_score, median_separation, best_alternative},
 a: {id, name}, b: {id, name}, aligned_id, transform}
```

**Verdict:**

- **uncertain:** the pose is ambiguous (symmetric part) or the overlap is below 20 %.
- **consistent:** the size difference over A's length, the offset, and the median separation (or 2× the combined
  noise, whichever is larger) are all within `tolerance`.
- **differ:** anything else.

On round parts, read `axis_scale_ppm` together with `axis_fit_offset_mm`. A thickness difference otherwise looks
like a radial scale, so the uniform `scale_ppm` can be misleading there.

**`save_aligned`** saves B moved rigidly into A's coordinates: operation `compare_scans`, parents `[a, b]`,
report = comparison summary. It carries a scalar `separation` (mm, NaN where there is no common surface), which the
viewer can colour.

**Duration:** 1–3 minutes for two 400k-point scans. The multi-start alignment tries 53 poses on round parts.

### `POST /api/accuracy/reference {asset_id, reference}`

Measures a known artefact. `reference` is one of:

| type | fields | measured as |
|---|---|---|
| `known_length` | `direction` (`x`/`y`/`z`/`length`/`width`/`height`/`[x,y,z]`), `region?`, `nominal` | Calipers: robust planes fitted to the extreme band at each end, distance along their mean normal. `details.method` = `faces`; `extent` when the ends are not faces across the direction. |
| `sphere_pair` | `region_a`, `region_b` (one ball each), `nominal` | Centre distance of two robust sphere fits (ball bar). |
| `diameter` | `region`, `axis_hint?`, `nominal` | Robust cylinder fit (pin, shaft). |
| `thread_pitch` | `region?`, `nominal?` | Helix-fit pitch. Without `nominal`, the nearest ISO / Unified / BSP pitch. A coarse absolute scale check that needs no calipers. |

Optional `tolerance`. The default is 0.02 mm + 100 ppm of the nominal; for `thread_pitch` it is 500 ppm, because
real threads are not made to a better lead.

Response: `{type, measured, nominal, error, error_ppm, error_pct, uncertainty (1 σ, from the fit), tolerance,
verdict: "ok" | "marginal" | "off", significant (|error| > 2 σ), sentence, details}`. `details` carries the fitted
features (face planes, sphere or cylinder parameters, thread diameters and lead) and the end points `a` / `b` for
drawing.

Regions use Contract 2 (`cloudclean.regions.region_mask`). The module falls back to box / spheres / screen
selection when that module is absent. Errors: 400 with a sentence.

### `POST /api/accuracy/dimensions {asset_id, trim = 0.0005}`

Robust part dimensions next to what the viewer's boxes show:

```
{dimensions: {length, width, height}, raw, trim, axes, center,
 axis_aligned_box, fitted_box, points, note}
```

- **dimensions:** trimmed extents along the part's axes (principal axes turned to the smallest box).
- **axis_aligned_box:** the scanner-coordinate box the viewer shows as X / Y / Z. It is not a part size.
- **fitted_box:** Open3D's box from the asset stats. It can be tilted several degrees on round parts.

### `POST /api/measure/snap {asset_id, points: [[x, y, z], ...]}`

Refines picked points for measuring. Each point moves onto a local surface fit instead of onto the nearest noisy
scan point:

- **Clouds:** a quadric fitted to the 24 nearest scan points (`SurfaceModel.project`).
- **Meshes:** the closest point on the triangles.

At most 200 points per request. Response, with one entry per point in the same order:

```
{asset_id,
 points:         [[x, y, z], ...]    the refined positions
 uncertainty_mm: [u | null, ...]     1-sigma error of the position along the surface normal: the fit residual
                                      x sqrt(leverage). null on meshes (a mesh carries no noise estimate).
 normals:        [[x, y, z], ...]    outward surface normal at the refined point
 on_surface_fit: [bool, ...]         false = no patch fitted (scan border, too few neighbours): the nearest scan
                                      point is returned and uncertainty_mm is the local noise
 moved_mm:       [d, ...]            distance from the clicked point
 method: "local quadric fit of the 24 nearest scan points" | "closest point on the mesh"}
```

For a two-point distance from snapped points `a`, `b`, the 1-sigma uncertainty is
`sqrt(u_a^2 + u_b^2)`. On a cylinder scanned with 0.03 mm noise, 400 opposite-point distances scatter as follows:

| snapping | σ | worst | predicted σ |
|---|---|---|---|
| nearest raw point | 0.048 mm | 0.138 mm | – |
| this endpoint | 0.018 mm | 0.052 mm | 0.018 mm |

Errors: 400 when `points` is not a list of `[x, y, z]`; 404 for an unknown asset.

## Library

| Function | Purpose |
|---|---|
| `robust_dimensions(geom_or_points, trim=0.0005, frame=None, refine=True)` | Trimmed extents along the part axes, plus raw extents, the frame and the AABB. Noise pushes trimmed extents out by about 2.5 σ per end; use `measure_length` for lengths. |
| `refine_axes(points, center, axes, trim)` | Principal axes turned to the smallest trimmed box. PCA alone tilts a 40 × 30 box ~0.8°, which adds +0.4 mm. |
| `measure_length(geom, direction, region=None)` | Distance between end faces, like calipers. |
| `fit_plane`, `fit_sphere`, `fit_circle`, `fit_cylinder` | Robust fits with standard errors. |
| `fit_thread(points, normals=None)` | Helix fit: pitch (±10 ppm on synthetic threads), handedness, major / minor from the crest / root points, pitch diameter only when both flanks were scanned (`both_flanks`, `profile_coverage`), lead per axial segment, `model` for `thread_offsets`. |
| `thread_offsets(model, points, edges)` | Axial / radial offset of any points against a fitted thread model, per segment. Only where the model saw data. |
| `SurfaceModel(geom).query(q, normals=None, max_angle_deg=30)` | Signed distance to a cloud (local quadric fits) or a mesh (exact closest point), foot normal, gap, ok. |
| `SurfaceModel(geom, k).project(q)` | Points moved onto the local quadric (clouds) or mesh, with the 1-σ uncertainty; backs `/api/measure/snap`. |
| `operation_drift(parent, child, operation, transform=None, tolerance=0.01)` | See the drift endpoint. |
| `transform_check(T)` | Determinant, orthonormality, scale, mirror, rotation. |
| `fit_pose(surface, points, T0, mode, frame_axes=None, normals=None)` | Robust point-to-surface fit. `rigid`, `similarity`, `similarity_offset`, `offset`, `axes`, `axes_offset`. |
| `align_scans(a, b)` | Multi-start alignment (principal-axis flips and rotations about the long axis for round parts), ranked by the share of points within 3 σ of the other scan. Reports near-equal alternatives (`ambiguous`). |
| `compare_scans(a, b, transform=None, tolerance=0.02)` | See the compare-scans endpoint. |
| `reference_check(geom, reference)` | See the reference endpoint. |
| `band_extents`, `trimmed_extents`, `surface_points`, `transformed_geometry`, `jsonable` | Helpers. |

## Related behaviour in other modules (fixes of 2026-09-24)

- **`register.assess_pair` (merge check):** layering counts only overlap points where both scans' local surfaces face
  within 40° of each other. The new key `layering_facing_fraction` gives that share of the overlap. Scans of a
  thread from its two ends see opposite flanks, and that no longer reads as a doubled skin. A real offset still
  does.
- **`analysis.thread_analysis` (the thread tool)** falls back to a helix fit (`accuracy.fit_thread`) when the
  crest-by-crest path finds no thread, or finds one flank only.
  - Result keys are unchanged, plus `method` (`"crest by crest"` | `"helix fit"`), `both_flanks` and
    `missing_flank` (`"+axis"` | `"-axis"` | null).
  - `pitch_diameter` and `flank_angle_deg` are null unless both flanks were scanned. A warning names the end of
    the thread whose flanks are hidden.
- **Edit ops:** cloud `denoise` (also `smooth` with `method: bilateral` on clouds) and mesh `smooth` with
  `method: laplacian` carry a warning in their catalogue description, log `WARNING` when they run, and add
  `warning` to their report entry. Measured effect: thread crests −0.13 mm and −0.04 mm respectively.

## Tools

`tools/accuracy/`:

- `real_bolt.py`: the bolt investigation (per-asset features, drift, scan comparison).
- `symmetry_scan.py`: screw-motion ambiguity of a bolt alignment.
- `synthetic_bench.py`: every processing step on parts of known size.
- `results/`: their outputs from 2026-09-24.
- Synthetic reference parts (`cylinder_mesh`, `box_mesh`, `sphere_pair_mesh`, `thread_rod_mesh`, `scan_surface`)
  live in `tests/synthetic.py`.
