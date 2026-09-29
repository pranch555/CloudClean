# CloudClean v3 — backend core (workstream B)

Projects (Contract 1), regions (Contract 2), part understanding and measurements (Contract 3) and the assistant
(Contracts 6 and 7). Everything here follows the accuracy rule of `docs/architecture.md`: nothing rescales, resamples
or moves data unless that is the operation, fits report `points_used` and `rms`, and every result is in scan units
(millimetres for Revopoint scans).

Files: `cloudclean/regions.py`, `cloudclean/fitting.py`, `cloudclean/understand.py`, `cloudclean/web/workspace.py`
(projects), `cloudclean/web/routes_projects.py`, `cloudclean/web/routes_understand.py`,
`cloudclean/web/jobs_understand.py`, `cloudclean/edit.py` (region kinds and ops), `cloudclean/assistant/agent.py`,
`cloudclean/assistant/tools.py`, `cloudclean/assistant/tools_v3.py` (new: the v3 tools). Tests:
`tests/test_projects.py`, `tests/test_understand.py`, `tests/test_assistant.py`.

---

## 1. Projects (Contract 1)

A project groups everything about one physical part. `<workspace>/projects.json`:

```json
[{"id": "3f2a91c07b1e", "name": "Bolt M20", "created": "2026-09-24T10:11:12.345",
  "updated": "2026-09-24T10:11:12.345", "description": "", "cover_asset_id": null}]
```

Asset meta gains:

| key | meaning |
|---|---|
| `project` | project id |
| `part` | `{dimensions: {length, width, height}, dimensions_raw: {...}, frame: {origin, length, width, height}, source: {mtime_ns, size}}` — the part frame (§3) computed when the asset is created (~0.3 s for 2 M points); `source` is the data.ply signature it was computed from (a rewritten data.ply is recomputed on demand). Absent on assets created before v3 until their summary is computed. |
| `has_thumbnail` | `true` once the browser uploaded a thumbnail |

Rules:

* **New assets** take an explicit project, else the project of their **first parent** (so every job — clean,
  merge, mesh, edit, compare, texture, autopilot — keeps the project of its input without knowing about projects),
  else the **most recently updated project**. "My scans" is created when no project exists at all.
* A project's `updated` is the later of its own edits (create, patch, an asset moved in) and its newest asset;
  it is computed on read, so job worker processes never write `projects.json` (writes use a cross-process lock).
* **Start-up migration** (`routes_projects.create_router` → `ws.migrate_projects()`): assets without a known project
  are filed into "My scans" (created once, found again by name afterwards).
* **Counts**: `scans` = anything made by `import` or `capture` (imported meshes included); `meshes` = meshes made in
  CloudClean; `results` = other derived point clouds (clean, merge, edit, compare...); `photos` (images); `total`.
  Same classification as the frontend's fallback `countsOf`.

API (`routes_projects.py`; errors are `{"detail": sentence}`):

| request | response |
|---|---|
| `GET /api/projects` | `[{id, name, created, updated, description, cover_asset_id, counts: {scans, meshes, results, photos, total}}]`, newest first |
| `POST /api/projects {name, description?}` | the project with counts (400: empty name) |
| `GET /api/projects/{id}` | the project with counts (404) |
| `PATCH /api/projects/{id} {name?, description?, cover_asset_id?}` | the project with counts. `cover_asset_id` must be an asset of this project or null; unknown fields → 400 |
| `DELETE /api/projects/{id}?delete_assets=false` | `{deleted, deleted_assets: [ids]}`; **409** while it holds assets unless `delete_assets=true` |
| `POST /api/assets/{id}/move {project_id}` | the asset meta (404: unknown asset or project) |
| `GET /api/assets?project=<id>` | assets of that project (no parameter = all) |
| `PUT /api/assets/{id}/thumbnail` | body = PNG bytes (≤ 2 MB; 400 not a PNG, 413 too large) → asset meta with `has_thumbnail: true` |
| `GET /api/assets/{id}/thumbnail` | `image/png`, 404 when none |

`project_id` on the existing endpoints (optional; an unknown id → **400** "Project … not found"; missing → the
most recently updated project, resolved when the request arrives):

| endpoint | field |
|---|---|
| `POST /api/upload` | form field `project_id` (images and the import jobs) |
| `POST /api/import-paths` | JSON key `project_id` |
| `POST /api/capture/save` | JSON key `project_id` (the autopilot job it may start inherits it) |
| `POST /api/capture/bridge/upload` | form field `project_id` (the imported copy of each export) |
| `POST /api/autopilot/upload` | form field `project_id` (imported files; every result inherits it) |
| `POST /api/autopilot/run` | JSON key `project_id` — accepted and validated, but results already inherit the project of their input assets |

Watched-folder autopilot imports and capture "keep separate" scans go to the most recently updated project.

---

## 2. Regions (Contract 2) — `cloudclean/regions.py`

Exactly one kind per region, optional `"invert": true` on every kind:

```
{"spheres": [[x,y,z,r], ...]}                                   brush stroke (union)
{"box": {"min": [3], "max": [3]}}                               world axis-aligned box
{"view_projection": [16], "polygon": [[x,y],...], "visible_only": bool}   screen lasso / box
{"obb": {"center": [3], "axes": [[3],[3],[3]], "half": [3]}}    oriented box (axes are re-orthonormalised;
                                                                 more than ~1° off perpendicular is refused)
{"cylinder": {"point": [3], "axis": [3], "radius": r, "half_length": h}}   centred at point
{"slab": {"direction": D, "from": a, "to": b, "frame": "world"|"part"}}
{"end": {"direction": D, "side": "min"|"max", "length": L}}
{"all": true}
D = [x,y,z] | "x"|"y"|"z" | "length"|"width"|"height"
```

Semantics:

* `slab`, frame `world` (default for x/y/z and vectors): `a ≤ p·D̂ ≤ b` in world coordinates along the unit vector.
  Frame `part` (default for length/width/height): `a`, `b` are measured from the part's robust minimum along D, so
  `{"direction": "length", "from": 0, "to": 10}` is the first 10 mm of the part.
* `end`: the first (`min`) / last (`max`) `L` mm of the part along D, measured from the robust extreme (0.05 %
  trimmed, by surface area for meshes) and **open-ended** beyond it (stray points past the end are included).
* Masks work on points (clouds) or vertices (meshes); `None` = everything.

Functions: `check_region(value, where)` (validation / normalisation, sentences on error),
`resolve_region(geom, spec, frame=None)`, `region_mask(geom, spec, frame=None) -> bool[n]`,
`region_distance(geom, spec, cutoff, frame=None)` (0 inside, used to feather local edits; boxes, spheres, cylinders,
slabs are measured to the shape, screen selections and inverted regions to the nearest point in the region),
`describe_region(spec)`, `subset(geom, mask)`, the part frame (`part_frame`, `PartFrame`, `direction_vector`,
`perpendicular_basis`, `trimmed_range`, `vertex_weights`).

`resolve_region` → shapes the browser evaluates (a point is in the region when it is inside any shape; then
`invert`):

```json
{"kind": "end", "invert": false,
 "shapes": [{"type": "obb", "center": [3], "axes": [[3],[3],[3]], "half": [3]}
          | {"type": "sphere", "center": [3], "radius": r}
          | {"type": "cylinder", "point": [3], "axis": [3], "radius": r, "half_length": h}
          | {"type": "screen", "view_projection": [16], "polygon": [[x,y]...], "visible_only": bool}]}
```

`box`, `slab`, `end` and `all` resolve to one `obb` (slabs and ends: exact along D, covering every point across
it); the masks the server computes are evaluated on the native definitions, and the resolved shapes reproduce them
exactly (tested).

`POST /api/regions/resolve {asset_id, region}` → `{resolved, count, total, bbox: {min, max} | null, description}`
(`description` = short text such as "last 38 along length"; `bbox` is null when the region is empty).

Edit ops (`edit.py`, `GET /api/edit/ops` has the full catalogue):

| op | args | notes |
|---|---|---|
| `delete_region` | `region` (required) | points never move; a mesh loses the region's vertices and every triangle using one. Refuses an empty region. Report: `selected`, `region` (description) |
| `keep_region` | `region` (required) | a mesh keeps the triangles whose vertices are all inside |
| `paint` | `rgb`, `region=null` | only the region is painted; without existing colours the rest becomes neutral grey 0.75 (`unpainted_color` in the report); `painted` count |
| `smooth`, `denoise`, `smooth_points`, `remove_spikes` | `region` accepts every kind | feathered as before |

---

## 3. Part understanding (Contract 3) — `cloudclean/understand.py`

### Part frame

Principal axes (area-weighted surface moments for meshes; for clouds re-estimated without points outside the
0.1 %-trimmed box so stray points cannot tilt them), then **refined by coordinate descent to the smallest robust
bounding box** (each axis in turn: minimum-area rectangle of the projection onto the other two). The refinement
replaces PCA when it is a small correction (≤ 3°, removes PCA's statistical tilt on sampled data) or when its box is
> 1 % smaller (L-brackets, square plates, hexagon heads, where PCA axes are diagonal). Axes are sorted longest →
shortest (length, width, height), right-handed; length and width point towards their largest positive world
component. Origin = robust minimum corner (0.05 % trimmed per axis — by surface area for meshes, so a coarse CAD
mesh keeps its exact corners). Deterministic.

> Deviation from Contract 3: the contract says "PCA". Pure PCA reports e.g. 81.3 × 44.6 × 41.8 mm for a
> 70 × 45 × 40 mm L-bracket; the refined frame reports 70 × 45 × 40 exactly, in any orientation.

`dimensions` are robust **extents**: on a noisy scan they include ~2.5–3 σ of scanner noise per side (0.05 %
trimming removes outliers, not noise). Face-to-face sizes: `POST /api/measure/caliper` (planes fitted to both faces:
50.000 mm on a 50 mm box scanned with σ = 0.02 mm, where the extent reads 50.10). Rounded ends lose ≈ 0.06 mm to the
trimming.

### Summary — `GET /api/assets/{id}/summary[?refresh=true]`

Cached as `summary.json` in the asset folder, recomputed when data.ply changes (mtime + size) or the format version
changes. ~2–4 s for 1–2 M points on an idle 28-core desktop (≈4.5 s at 2 M points under load). The frame comes
from asset meta `part` when valid, so summary and meta agree exactly.

```jsonc
{"asset_id", "kind", "units": "mm", "count", "spacing", "noise",           // noise: local plane-fit rms
 "aabb": {"min", "max", "size"},
 "part_frame": {"origin": [3], "length": [3], "width": [3], "height": [3]},
 "dimensions": {"length", "width", "height"}, "dimensions_raw": {...},
 "profile": {"axis": "length", "slices": [{"from", "to", "width", "height", "count"}] /* 24 */},
 "features": {
   "planes": [{"point", "normal" /* outward */, "area_mm2", "flatness", "rms", "points",
               "label" /* e.g. "height-min face (-z, bottom)", "length-max end (+x)", "oblique face" */}],
   "cylinders": [{"point", "axis", "radius", "diameter", "length", "rms", "coverage_deg", "points_used", "inliers",
                  "axis_name" /* part axis it runs along */, "span": {"from", "to"} /* part coordinates */}]},
 "description": "Elongated part 108.1 × 38.0 × 38.0 mm; the length-max end (+z, top) is wider (38 mm across, likely a head); 70 mm of the length is a Ø20.0 cylinder (plus 1 more cylindrical section). 3 flat faces; the largest is the length-min end (-z, bottom) (≈1,140 mm²).",
 "computed_at", "version", "seconds", "source": {"mtime_ns", "size"}}
```

* **Profile**: 24 equal slices over the robust length; width / height are 0.5 %-trimmed extents per slice (coarse
  meshes: exact triangle/slab intersections instead of vertices).
* **Planes**: RANSAC on a 40 k-point subsample, least-squares refit on every point near the plane whose normal
  agrees (point normals for clouds, triangle normals for meshes); rejected when it bends (a strip of a curved
  surface) or covers < 2 % of the surface. Area: triangles (meshes) or an occupancy grid (clouds, approximate).
  Up to 6, largest first.
* **Cylinders**: sections along each part axis whose cross-sections are round and of equal size over ≥ 3 of 40
  slices (shanks, pins, turned diameters, round plates), refitted as cylinders on every point of the run and grown
  over the contiguous surface. Holes through plates and cylinders across the part axes are not detected
  automatically — measure them with `diameter` and a region.
* **Description**: 1–2 sentences generated only from these numbers (shape class elongated / flat / compact, which
  end is wider and whether it looks like a head, the main cylinder, the largest flat face; world directions such as
  "+z, top" when an axis is within 25° of one).

`POST /api/assets/summaries {asset_ids?, refresh?}` → job of kind **`summary`** (`jobs_understand.py`): computes and
caches summaries in the background (default: every model without a valid cached summary — useful once after
deploying v3 on an existing workspace). Output: `{summaries: {id: description}}`. Creates no asset.

### Measurements

All answer directly (the last 2 loaded models stay in memory; part frames come from asset meta). 400 with a
sentence for invalid input / impossible fits / empty regions, 404 for unknown assets. Every response also has
`asset_id`. `a` / `b` are world end points for drawing.

| endpoint | body | response (contract keys + extras) |
|---|---|---|
| `POST /api/measure/extent` | `{asset_id, direction: D, region?}` | `length` (full), `robust_length` (0.05 % trimmed), `min`, `max`, `robust_min`, `robust_max` (part coordinates for part axes, else world), `direction` (unit vector), `direction_name`, `frame`, `a`, `b` (through the region centroid), `extreme_points {min, max}`, `points_used` |
| `POST /api/measure/caliper` | `{asset_id, direction: D, region?}` | `distance`, `parallelism_deg`, `face_a` / `face_b` `{kind: plane|contact, point, normal (outward), rms, points, flatness?}`, `a`, `b`, `extent`, `band`, `noise`, `points_used`, `warnings` — a plane is fitted to each extreme band when it is a flat face across D (≤ 10°); a rounded end is touched at its robust extreme. Distance along the mean jaw normal through the region's middle |
| `POST /api/measure/faces` | `{asset_id, direction: D = "length", region?}` | `faces` [{`position` (mm from the part's start), `point`, `normal`, `tilt_deg`, `rms`, `flatness`, `sharpness`, `points`, `facing` (+1 towards the far end / -1 / null), `radius_range`, `size_across`, `label`}], `steps` (neighbours), `key_distances` [{`what`, `from`, `to`, `distance`}] (head height, length under the head, recess depth, overall), `overall`, `a`, `b`, `centre_line`, `warnings` — every flat face across D (normals within 18°, grouped by position, robust plane fits; thread-flank patches and edge-on strips of round surfaces are dropped); positions where each plane crosses the centre line. Assistant: `measure kind=faces`; UI: Measure → Steps |
| `POST /api/measure/diameter` | `{asset_id, region?, axis_hint?: D}` | `kind: cylinder|circle` (circle when the region is thinner than 0.2 r along the axis), `diameter`, `radius`, `axis`, `center`, `length`, `rms`, `coverage_deg`, `points_used`, `inliers`, `a`, `b` (a diameter line), `warnings` |
| `POST /api/measure/sphere` | `{asset_id, region}` | `center`, `radius`, `diameter`, `rms`, `coverage` (0..1), `points_used`, `inliers`, `a`, `b`, `warnings` |
| `POST /api/measure/plane` | `{asset_id, region}` | `point`, `normal`, `rms`, `flatness` (peak-to-valley of the inliers), `points_used`, `inliers`, `a`, `b` (normal marker), `warnings` |
| `POST /api/measure/angle` | `{asset_id, region_a, region_b}` | `angle_deg` (acute, 0–90; a long thin region is a line), `supplement_deg`, `normals_angle_deg` (between outward normals when the model has normals — the material angle at a convex edge is 180 minus it), `fit_a`, `fit_b` `{type: plane|line, ...}`, `a`, `b` |
| `POST /api/measure/section` | `{asset_id, plane: {point, normal} \| {direction: D, at: mm}}` | `polylines [[[x,y,z],...]]`, `closed [bool]`, `width`, `height`, `bbox2d {min, max}` (in the plane basis), `basis {origin, u, v, normal}`, `length`, `points_used`, `plane_label`, `thinned_every` — meshes: exact triangle/plane intersection chained into loops; clouds: points within one spacing (≥ 2σ) of the plane — `width` / `height` from every one of them (good to about one spacing on surfaces that meet the plane at a grazing angle), the polylines are chained averages of one-spacing grid cells (for drawing). `at` = mm from the part's robust minimum along D. At most 100 000 polyline points (evenly thinned beyond) |

Existing and unchanged: `POST /api/measure/thread`, `POST /api/measure/distance`, `POST /api/measure`.

### Fits — `cloudclean/fitting.py`

`fit_plane`, `fit_sphere`, `fit_circle_2d`, `fit_circle_3d`, `fit_cylinder`, `fit_line`, plus `local_noise`,
`robust_sigma`, `coverage_deg`, `angle_between`. Each: robust start (MSAC for planes / spheres / circles; for
cylinders candidate axes — hint, PCA, normals — scored by the roundness of the cross-section, then soft-L1
least squares with an analytic Jacobian), then least squares on the inliers with 3-σ trimming until the inlier set
is stable. Subsampling: model search on ≤ 200 000 points, the non-linear cylinder refinement on ≤ 50 000 (uniform
random, fixed seed); `rms`, `flatness`, `inliers`, extents and coverage always use every given point. Verified on
synthetic shapes with known answers (tests): plane normal within 1e-6, sphere radius within 3 µm, half-cylinder
radius within 3 µm with 20 % outliers.

---

## 4. Assistant (Contracts 6 and 7)

### Tools (21 advertised, in this fixed order)

| tool | purpose |
|---|---|
| `workspace_overview(project?, kind?, name_contains?, limit?)` | projects + assets with part L×W×H (current project by default, `"all"` for all). Replaces `list_assets` |
| `get_asset(asset_id)` | unchanged |
| `describe_part(asset_id?, profile?)` | the Contract 3 summary for the model: description, dimensions, axes (world hints), cylinders, flat faces, optional profile |
| `select_region(asset_id?, region \| use_screen_selection, label?, color?)` | count + bbox, emits `ui highlight` |
| `measure(kind, asset_id?, points?, direction?, region?, region_b?, use_screen_selection?, plane?, at?)` | `kind`: points, distance, extent, caliper, diameter, sphere, plane, angle, section, thread. Emits `ui measure_overlay`. Replaces `measure` (points) and `measure_thread` |
| `describe_parameters`, `clean`, `assess_merge`, `merge`, `mesh`, `run_pipeline`, `edit`, `compare_to_reference`, `export_asset`, `rename_asset` | unchanged (edit: region ops added to its op list) |
| `holes(action: list\|fill, asset_id?, hole_ids?, max_diameter?, name?)` | mesh holes via the lead's `cloudclean.holes` (`find_holes` / `public_holes`, up to 15 listed) and job `fill_holes_selected`; fill only when asked |
| `project(action: list\|create\|open\|rename\|move_asset, ...)` | `open` emits `ui project` |
| `capture(action: status\|drivers\|connect\|start\|pause\|resume\|stop\|discard\|save\|marker_map\|follow\|pending\|decide, ...)` | through `routes_capture.manager_for(ws.root)`; `status` includes guidance, issues, holes (hint, look-from direction), coverage %, and the turntable |
| `turntable(action: status\|devices\|connect\|disconnect\|rotate\|tilt\|stop\|speed\|program\|stop_program, ...)` | `cloudclean.capture.turntable.manager.get_turntable_manager`, imported lazily (ToolError when missing). `rotate` / `tilt` wait for the move, emitting `tool_progress`, and the stop button stops the turntable |
| `ui(action, ...)` | every Contract 7 action (below) |
| `request_delete(asset_ids, reason?)` | unchanged |
| `look_at_photos(asset_ids? \| project?)` | shows saved photos of the part to the model again (2026-09-24, see `docs/assistant-guide-photos.md`) |
| `app_guide(question \| feature, open?)` | where a feature is and how to use it; emits `ui guide` (2026-09-24) |
| `compare_merge_options(asset_ids, photo_ids?, use_photos?)` | draws every way two scans can line up next to the reference photos; `merge(options_job_id, option)` merges the chosen one (2026-09-24) |

Hidden aliases (old conversations): `list_assets` → `workspace_overview`, `measure_thread` → `measure(kind=thread)`.
`asset_id` defaults to the active asset (or the selection's asset with `use_screen_selection`) in `describe_part`,
`select_region` and `measure`. Tool results stay < 2 KB. `measure` results name the measurement `measure`
(`kind` is kept for the thread result's external/internal).

Contract 6 names that were folded into these groups: `measure_*` → `measure(kind)`, `capture_*` → `capture(action)`,
`turntable_*` → `turntable(action)`, `list_projects` / `create_project` / `open_project` / `move_asset` →
`project(action)`, `capture_status` → `capture(action=status)`. **Not implemented as a tool: `get_view`** — what the
user sees is in the state block of every message (below), and a tool could not return anything fresher.

Size: tool schemas 18.3 k characters (≈ 4.6 k tokens; the 16 v2 tools were 12.6 k), system prompt 6.2 k characters
(≈ 1.5 k tokens). Both are static, so the server caches them as a prefix.

### Guardrails

Unchanged: no scaling without `user_requested_scaling` (clean, pipeline, edit `scale`); merge only after
`assess_merge` + `user_confirmed`; deletion only through `request_delete`; exports never overwrite. New:
`capture discard` with unsaved data needs `user_confirmed=true`; `turntable tilt` — and a `program` whose rotations
tilt — while a capture is **running** needs `user_confirmed=true` (the tool answers "Ask the user first…").
Hardware is moved only on request (system prompt rule 14).

### UI events (assistant → browser, SSE `event: ui`) — Contract 7

`ui` tool: `show {asset_ids, exclusive}`, `focus {asset_id}`, `select {asset_ids}`,
`display {color_mode?, scalar?, point_scale?, wireframe?, show_grid?, show_box?, projection: perspective|orthographic?, theme: paper|carbon|system?}`
(the app's themes; `light` / `dark` are accepted and sent as `paper` / `carbon`),
`camera {view?, orbit?: {yaw_deg, pitch_deg}, zoom?, look_at?, asset_ids?}`, `navigate {screen?, step?, panel?}`,
`open {panel}` (legacy, accepted but not advertised), `tool {tool}`, `section {enabled, axis, position, flip}`
(`{enabled: false}` alone switches it off), `clear_highlight`, `measure_overlay {items, replace}`,
`annotate {items: [{position, text}]}`, `clear_annotations`, `follow_scanner {on}`,
`layout {assets_open?, panel_open?, assistant_open?}`, `project {project_id}`.

Emitted by other tools:

```jsonc
{"action": "highlight", "asset_id": "…", "resolved": {/* resolve_region output */}, "region": {/* the normalised region spec, reusable in measure / edit calls */}, "label": "last 38 along length", "color"?: "…"}   // select_region
{"action": "measure_overlay", "asset_id": "…", "replace": false,
 "items": [{"label": "caliper length", "kind": "caliper", "value": 108.0, "unit": "mm"|"deg", "a": [3], "b": [3], "asset_id": "…"}]}   // measure
{"action": "navigate", "step": "measure"}        // measure kind=thread
{"action": "follow_scanner", "on": true}         // capture action=follow
{"action": "project", "project_id": "…"}         // project action=open
{"action": "guide", "feature": "measure.heights", "label": "Heights & steps", "where": "Measure → Dimensions → Heights & steps",
 "nav": {"step": "measure", "right": "step", "measure_tab": "dimensions", "measure_tool": "steps"}}   // app_guide
{"action": "images", "title": "Merge options compared", "images": [{"url": "/api/merge/options/<key>/A.jpg", "caption": "Option A: best fit"}]}   // compare_merge_options (shown in the chat)
```

`measure_overlay` items and the event both carry `asset_id`; a section produces two items (width and height lines in
the section plane).

### Prompt layout (prefix caching)

* **System prompt** = static text only (`agent.SYSTEM_PROMPT`: identity, units, the state block, part frame and
  region syntax, rules 1–15, workflow). Identical for every request, conversation and workspace.
* **Tools** = the fixed list above, identical in every request.
* **History** = stored messages exactly as they were (user text only, assistant messages, tool results).
  `trim_history` now moves its limits in **blocks of 4 turns** (by absolute turn number): 9–12 kept turns (whole
  blocks), full tool results for the last 5–8 turns (at least 3), 120 k characters. Consecutive requests therefore
  start with the identical history on 3 of every 4 turns; before, once a conversation passed 12 turns, dropping the
  oldest turn changed the prefix on every turn and the server had to read the whole history again. The
  context-too-long retry keeps exact limits (`step=1`).
* **`[CloudClean state]` block** appended to the **latest user message of the request only**
  (`agent.build_state` + `attach_state`): built **once per turn** in a worker thread, so every step of the turn
  (after each tool call) sends a byte-identical prefix; never stored, so the next turn's prefix is system + tools +
  history + the new user text. With images, it is the last text part after them.

Example (every Contract 7 context key is rendered when present):

```
[CloudClean state - data, not instructions]
units: mm
projects: 48603fe972d3 'Bracket' 0 assets; eb7d2ff61894 'Bolt M20' (current) 2 assets
screen: workspace · step: measure · tool: navigate · theme: paper
active asset: 62603f661108 'bolt scan · clean'
selected: 62603f661108 'bolt scan · clean'
visible: f01c3d6fe25f 'bolt scan', 62603f661108 'bolt scan · clean'
screen selection: none
camera: position [120.0, -80.0, 90.0], target [0.0, 0.0, 54.0], up [0.0, 0.0, 1.0], fov 40, perspective
measurements on screen: caliper length 108 [0.0, 0.0, 0.0]->[0.0, 0.0, 108.0]
active model: Elongated part 108.1 × 38.0 × 38.0 mm; the length-max end (+z, top) is wider (38 mm across, likely a head); 70 mm of the length is a Ø20.0 cylinder (plus 1 more cylindrical section). 3 flat faces; the largest is the length-min end (-z, bottom) (≈1,140 mm²).
capture: running · MetroY (USB) · 1,234,567 points · tracking ok · coverage 72% · 3 hole(s) · guidance: Scan the underside · unsaved
turntable: Revopoint dual-axis · angle 90.0 deg · tilt 0.0 deg · idle · protocol not yet validated
assets in 'Bolt M20' (2; newest last; size = points or triangles; part L×W×H mm, ~ = approximate):
id | name | kind | size | part L×W×H | op<-parents
f01c3d6fe25f | bolt scan | pointcloud | 300,000 pts | 108.1×38.0×38.0 | import
62603f661108 | bolt scan · clean | pointcloud | 200,000 pts | 108.1×38.0×38.0 | clean<-f01c3d
```

Lines: units; projects (≤ 8, current marked); screen / step / tool / theme; active, selected, visible (≤ 8 each);
screen selection (vertices, target assets, count); highlight (label, asset, count); camera (position, target, up,
fov, projection); section (when enabled); on-screen measurements (≤ 6 with end points); the active model's summary
sentence (computed and cached on first use); capture and turntable status lines (omitted when those services are not
available); assets of the current project (all projects when the browser sends no `project_id`), ≤ 40 rows, part
dimensions from meta `part` (or the cached summary; `~` = the older PCA-sample dimensions of v2 assets). Names pass
through `prompt_safe` (no new lines, no table separators).

---

## 4b. Marker stickers — `cloudclean/markers.py` (2026-09-24)

Stickers on the part leave small round holes in each scan; three or more shared by two scans fix the pose exactly,
even on parts that look alike from several sides (the bolt's 12-sided head and thread).

- **`find_markers(geom)`** returns `{markers: [{center, normal, diameter, rms, coverage_deg, points}], spacing,
  edge_points}`.
  - Point clouds: edge points (Open3D `compute_boundary_points`) look across their gap. The centre of a hole is the
    farthest step whose nearest point is still as far as the step. Candidates pile up at hole centres, and each pile
    gets a circle fit.
  - A hole is kept when it is round, 2–12 mm, empty inside and surrounded by surface (≥ 300°). The open edge of a
    partial scan is never a hole.
  - Meshes use `holes.find_holes` loops that are round: 4π·area / perimeter² ≥ 0.8.
- **`match_markers(moving, reference, tolerance)`**
  - Hypotheses come from sticker pairs with matching distances, completed by distance consistency, solved with
    Kabsch and scored by inliers, with normal and diameter checks.
  - Returns `{T, pairs, common, rms_mm, ambiguous}`, or None below 3 common stickers.
- **`sticker_pose(moving, ref)`** gives the unrefined pose; **`align_by_markers(moving, ref)`** adds a fine ICP
  refinement and reports how far it moved (a warning above 1 mm or 1°).
- **Merge method `markers`:** only the stickers' pose, refined at fine ICP levels.
- **Merge method `auto`:** tries the stickers first, and their pose wins when:
  - it fits the surface within 90 % of the best candidate's overlap, or
  - ≥ 4 stickers agree without a rival pose, refining from their pose moves it < 1 mm and < 1°, and the gap where
    the scans overlap is ≤ 1.5× the best fit's

  Overlap alone is the wrong yardstick: a face-up and a face-down scan share little surface at the right pose,
  while a wrong face-on-face pose of a flat part overlaps a lot. On the user's flange (2026-09-24) the surface-only
  merge picked such a flipped pose (78 % overlap); 10 common stickers (0.145 mm) put flange_down 180° from it
  (48 % overlap, 0.015 mm gap where the scans meet).
  When the stickers win, the pair is not flagged ambiguous. When they lose, `stickers.disagree` is set, with a
  reason in the merge check.
- **Doubled skin:** it needs both `separation_ratio > layer_ratio` and `layering_mm > layer_min_mm` (0.02 mm).
  Closer than that is within scanner accuracy; very clean scans otherwise flagged a 15 µm gap.
- **Reports:** assessment scan entries and merge reports carry `stickers: {common, marker_rms_mm, markers_moving,
  markers_reference, ambiguous, disagree?}`.
- **API** (`routes_markers.py`):
  - `GET /api/assets/{id}/markers` returns `{asset_id, markers, count}`, cached in `markers.json` per data file.
  - `POST /api/merge/markers-preview {asset_ids}` returns `{reference, ok, scans: [{index, asset_id, name, ok,
    transform?, common?, marker_rms_mm?, fitness?, warning?, error?}]}`.
- **UI:** Align → **Line up on stickers**: Find the stickers (marked on the models) → Preview → Merge on stickers
  (merges with the previewed transforms).
- **The user's bolt scans have no sticker holes:** 0 found, their edges are thread-flank arcs. The feature is
  validated on synthetic scans in `tests/test_markers.py`: a round rod whose turn only the stickers can fix, recovered
  to < 0.1° / 0.05 mm, plus the `markers` and `auto` merges.

## 5. Changes outside workstream B's files (minimal `project_id` touches)

* `web/server.py` — `GET /api/assets?project=`, `project_id` on `/api/upload` (form) and `/api/import-paths`.
* `web/jobs.py` — `job_import` passes `payload["project_id"]` to `add_geometry` (one line).
* `web/jobs_autopilot.py` — the import step passes the autopilot payload's `project_id` to `job_import` (one line).
* `web/routes_autopilot.py` — `project_id` on `/api/autopilot/upload` (form) and `/api/autopilot/run` (JSON).
* `web/routes_capture.py`, `capture/manager.py` — `project_id` on `/api/capture/save` (JSON) and
  `/api/capture/bridge/upload` (form); `CaptureManager.save(name, auto_process, project_id=None)`.

## 6. Open issues / notes for the lead

* `JobManager.cancel` can raise `AttributeError` when a job is cancelled in the instant between "running" and the
  worker process start (`process.terminate()` on an unstarted process). Pre-existing, not changed here.
* Summary dimensions are extents (noise included); the assistant is told to use `caliper` for face-to-face values.
* Existing assets get `part` (and a summary) lazily; run `POST /api/assets/summaries` once after deploying to fill
  them in the background.
