# CloudClean v3 — plan and contracts

Work of 2026-09-24. The user asked for five things:

1. Live capture: dragging the model stops working after a second (the view keeps snapping back).
2. "Measurements are a little off" — find out whether the scanner or the software is responsible.
3. A UI that is easy for anyone, distinctive, and designed to the highest standard (not the generic dark look).
4. Every Revo Metro feature and more — far superior in software and design.
5. The assistant (local LLM) must be able to control everything: capture, turntable rotation and tilt, UI, edits on
   parts of a model; it must know every model and its measurements in any direction.

Accuracy stays the top priority: nothing may silently rescale, resample or move data, and every operation reports
what it did. Existing contracts in `docs/architecture.md` still hold.

## Ownership (parallel work — do not edit files owned by another workstream)

| Workstream | Owns (create/edit) | Must not touch |
|---|---|---|
| **T** turntable | `cloudclean/capture/turntable/**`, `cloudclean/web/routes_turntable.py`, `tests/test_turntable.py`, `docs/turntable.md`, `tools/turntable/**`, optional extra `[turntable]` in `pyproject.toml` | frontend, other routes |
| **A** accuracy | `cloudclean/accuracy.py`, `cloudclean/web/routes_accuracy.py`, `cloudclean/web/jobs_accuracy.py`, `tests/test_accuracy.py`, `tools/accuracy/**`, `docs/accuracy.md`, `docs/accuracy-investigation-2026-09-24.md` | frontend, workspace data on the Spark (read only) |
| **B** backend core | `cloudclean/web/workspace.py` (projects), `cloudclean/web/routes_projects.py`, `cloudclean/fitting.py`, `cloudclean/understand.py`, `cloudclean/regions.py`, `cloudclean/web/routes_understand.py`, `cloudclean/web/jobs_understand.py`, `cloudclean/assistant/**`, `cloudclean/edit.py` (region kinds + region ops only), `tests/test_projects.py`, `tests/test_understand.py`, `tests/test_assistant*.py`, `docs/v3-backend.md` | frontend, capture drivers, turntable package |
| **F** frontend | `frontend/**`, `cloudclean/web/static/**` (build output) | backend |

Router/job modules are already registered in `server.ROUTER_MODULES` / `jobs.JOB_MODULES` as stubs: fill them in, do
not edit the registration lists. Each workstream documents itself in its own doc file; the lead merges them into
`docs/architecture.md` at the end. Nobody deploys to the Spark or restarts its service except the lead.

Tests: `.venv/Scripts/python -m pytest -q` on Windows must stay green (run your own test file plus the full suite
before you finish). The Spark copy is updated by the lead.

## Contract 1 — projects (B)

A project groups everything about one physical part. `<workspace>/projects.json` =
`[{id (12 hex), name, created, updated, description, cover_asset_id|null}]`.

- Asset meta gains `project` (project id). On start-up, assets without one are assigned to a project named
  "My scans" (created once). New assets inherit the project of their first parent; imports, uploads and capture
  saves take an explicit `project_id` (falling back to the most recently updated project).
- `GET /api/projects` → `[{id, name, created, updated, description, cover_asset_id, counts: {scans, meshes, results,
  photos, total}}]` newest first. `POST /api/projects {name, description?}` → project.
  `PATCH /api/projects/{id} {name?, description?, cover_asset_id?}`. `DELETE /api/projects/{id}?delete_assets=false`
  (refuses with 409 when the project still has assets unless `delete_assets=true`).
- `POST /api/assets/{id}/move {project_id}`. `GET /api/assets?project=<id>` filters (no param = all).
- Existing upload/import/capture/autopilot endpoints accept an optional `project_id` (form field or JSON key).
- Thumbnails are rendered by the browser: `PUT /api/assets/{id}/thumbnail` (body = PNG bytes, ≤ 2 MB) and
  `GET /api/assets/{id}/thumbnail` (404 when none). Asset meta gets `has_thumbnail: true`.

## Contract 2 — regions (B)

Backwards compatible extension of the region argument of `cloudclean/edit.py` (exactly one kind per region):

```
{"spheres": [[x,y,z,r], ...]}                                  existing
{"box": {"min": [3], "max": [3]}}                                existing, world axis-aligned
{"view_projection": [16], "polygon": [[x,y],...], "visible_only": bool}   existing, screen lasso/box
{"obb": {"center": [3], "axes": [[3],[3],[3]], "half": [3]}}     new, oriented box in world coordinates
{"cylinder": {"point": [3], "axis": [3], "radius": r, "half_length": h}}   new, centred at point
{"slab": {"direction": D, "from": a, "to": b, "frame": "world"|"part"}}     new
{"end": {"direction": D, "side": "min"|"max", "length": L}}                new: first/last L mm of the part
{"all": true}                                                              new
optional on every kind: "invert": true
D = [x,y,z] | "x" | "y" | "z" | "length" | "width" | "height"   (length/width/height = the part frame axes)
```

`cloudclean/regions.py`: `resolve_region(geom, spec) -> dict` turns any region into **world shapes the browser can
evaluate**: `{"shapes": [{"type": "obb", center, axes, half} | {"type": "sphere", center, radius} |
{"type": "cylinder", point, axis, radius, half_length} | {"type": "screen", view_projection, polygon}],
"invert": bool}`; `region_mask(geom, spec) -> bool[n]` (points or mesh vertices). New edit ops `delete_region(region)`
and `keep_region(region)`; `paint`, `smooth`, `denoise`, `remove_spikes` accept any region kind.

`POST /api/regions/resolve {asset_id, region}` → `{resolved, count, total, bbox: {min, max}}`.

## Contract 3 — part understanding and measurements (B)

**Part frame**: principal axes of the asset (PCA, sorted longest → shortest, named length / width / height),
right-handed, origin at the robust minimum corner (0.05 % trimmed) so coordinates along each axis run from 0.

`GET /api/assets/{id}/summary` (cached in the asset folder, invalidated when data.ply changes) →

```
{asset_id, kind, units: "mm", count, spacing,
 aabb: {min, max, size},
 part_frame: {origin: [3], length: [3], width: [3], height: [3]},
 dimensions: {length, width, height},            robust (trimmed) extents along the part axes
 dimensions_raw: {length, width, height},        full extents
 profile: {axis: "length", slices: [{from, to, width, height, count}] x 24},   cross-section size along the length
 features: {planes: [{point, normal, area_mm2, flatness, label}], cylinders: [{point, axis, radius, length, rms,
            coverage_deg}]},
 description: "one or two plain sentences describing the shape, e.g. which end is thicker",
 computed_at}
```

Measurement endpoints (all distances in scan units, return end points `a`/`b` in world coordinates so the browser
can draw the dimension line):

| endpoint | body | returns |
|---|---|---|
| `POST /api/measure/extent` | `{asset_id, direction: D, region?}` | `{length, robust_length, min, max, direction, a, b, points_used}` |
| `POST /api/measure/caliper` | `{asset_id, direction: D, region?}` | distance between the two opposing faces measured like calipers (planes fitted to the extreme bands): `{distance, parallelism_deg, face_a: {point, normal, rms, points}, face_b, a, b}` |
| `POST /api/measure/diameter` | `{asset_id, region?, axis_hint?: D}` | cylinder (or circle, for a thin slice) fit: `{diameter, radius, axis, center, length, rms, coverage_deg, points_used, a, b}` |
| `POST /api/measure/sphere` | `{asset_id, region}` | `{center, radius, diameter, rms, points_used}` |
| `POST /api/measure/plane` | `{asset_id, region}` | `{point, normal, rms, flatness, points_used}` |
| `POST /api/measure/angle` | `{asset_id, region_a, region_b}` | plane or line fit in each region: `{angle_deg, fit_a, fit_b}` |
| `POST /api/measure/section` | `{asset_id, plane: {point, normal} \| {direction: D, at: mm (part frame)}}` | `{polylines: [[[x,y,z],...]], width, height, bbox2d, points_used}` |

Existing: `POST /api/measure/thread`, `POST /api/measure/distance` (unchanged).

## Contract 4 — turntable (T)

Revopoint turntables are Bluetooth LE devices (Revo Metro links `simpleble.dll`). Dual-axis turntable: rotation
interval 5–30°, speed 25–90 s per revolution, direction CW/CCW, tilt −30…30°, frames per stop, up to 5 rotations each
with its own tilt, "Turntable Sync" (scan start/pause linked to rotation). Large turntable: rotation only.

`cloudclean/capture/turntable/manager.py`:

```python
def get_turntable_manager(workspace) -> TurntableManager      # one per workspace root
class TurntableManager:
    def devices(self, scan_seconds: float = 4.0) -> list[dict]  # [{id, name, kind, rssi}] (BLE scan + "simulated")
    def connect(self, device: str | None = None, kind: str = "auto") -> dict      # -> status
    def disconnect(self) -> dict
    def status(self) -> dict   # {connected, device, name, kind: "dual_axis"|"large"|"simulated"|None,
                               #  angle_deg, tilt_deg, moving, speed_s_per_rev, direction: "cw"|"ccw",
                               #  program: {state, rotation, stop, stops_per_rotation, ...}|None, error,
                               #  capabilities: {tilt, tilt_range, speed_range, interval_range}, validated: bool}
    def rotate(self, degrees: float, speed_s_per_rev: float | None = None, wait: bool = False) -> dict
    def tilt(self, degrees: float, wait: bool = False) -> dict      # absolute tilt, dual-axis only
    def stop(self) -> dict
    def set_speed(self, s_per_rev: float) -> dict
    def start_program(self, program: dict) -> dict
        # {interval_deg, frames_per_stop, direction, speed_s_per_rev, rotations: [{tilt_deg}], sync_scan: bool}
    def stop_program(self) -> dict
```

Routes: `GET /api/turntable/status`, `GET /api/turntable/devices`, `POST /api/turntable/connect {device?, kind?}`,
`POST /api/turntable/disconnect`, `POST /api/turntable/rotate {degrees, speed_s_per_rev?}`,
`POST /api/turntable/tilt {degrees}`, `POST /api/turntable/stop`, `POST /api/turntable/speed {s_per_rev}`,
`POST /api/turntable/program {...}`, `POST /api/turntable/program/stop`. Errors: HTTP 400 with a sentence.
A `simulated` turntable always exists so the UI and assistant work without hardware. `validated: false` while the
BLE protocol has not been confirmed on the real device — the UI shows that.

## Contract 5 — accuracy (A)

`cloudclean/accuracy.py` + routes:

- `POST /api/accuracy/drift {asset_id}` → how far the operation that created the asset moved/resized the surface
  relative to its parent: `{asset_id, parent_id, operation, displacement: {signed_mean, mean, rms, p95, max},
  dimensions_before, dimensions_after, dimension_change, verdict: "unchanged"|"changed"|"moved", sentence}`.
- `POST /api/accuracy/compare-scans {a_id, b_id}` → job kind `compare_scans`: aligns two scans of the same part and
  reports relative scale (ppm), per-axis extent differences and surface separation — tells scanner repeatability apart
  from processing.
- `POST /api/accuracy/reference {asset_id, reference}` — check against a known artefact; `reference` =
  `{type: "known_length", direction: D, region?, nominal}` | `{type: "sphere_pair", region_a, region_b, nominal}` |
  `{type: "diameter", region, nominal}`: `{measured, nominal, error, error_ppm, error_pct, verdict, sentence}`.
- The investigation report `docs/accuracy-investigation-2026-09-24.md` answers the user's question with numbers.

## Contract 6 — assistant (B)

The assistant must be able to do anything the user can do in the UI. Tools (in addition to the current ones):

- **context**: `workspace_overview` (projects, every asset with part dimensions), `describe_part(asset_id)`
  (Contract 3 summary), `get_view` (what the user sees), `capture_status` (session, guidance, holes, turntable).
- **measure**: `measure_extent`, `measure_caliper`, `measure_diameter`, `measure_angle`, `measure_plane`,
  `measure_section`, `measure_sphere` (Contract 3) — each also emits a `ui measure_overlay` so the result is drawn.
- **regions**: `select_region(asset_id, region, label?)` resolves a region (Contract 2), reports how many points it
  holds and highlights it in the viewer; edits accept the same region objects (`delete_region`, `keep_region`,
  `smooth`, `denoise`, `paint`...).
- **capture**: `capture_connect(driver, settings)`, `capture_control(action: start|pause|resume|stop|discard)`,
  `capture_save(name?, auto_process?)`, `capture_marker_map(action)` (driver commands), `capture_follow(on)`.
- **turntable**: `turntable_connect`, `turntable_rotate(degrees)`, `turntable_tilt(degrees)`, `turntable_stop`,
  `turntable_program(...)` (Contract 4).
- **projects**: `list_projects`, `create_project`, `open_project`, `move_asset`.
- **ui**: see Contract 7.

Guardrails stay: never scale without an explicit request, merge only after `assess_merge` + user approval, deletion
only through `request_delete`, never overwrite files; tilting the turntable while a scan is running asks first.
The system prompt lists projects and assets with **part dimensions** (length × width × height), the active asset's
summary sentence, capture/turntable state and the UI state the browser sends.

## Contract 7 — UI events (assistant → browser, SSE `event: ui`) and context (browser → assistant)

```
{action: "show", asset_ids, exclusive}          {action: "focus", asset_id}        {action: "select", asset_ids}
{action: "display", color_mode?, scalar?, point_scale?, wireframe?, show_grid?, show_box?, projection?, theme?}
{action: "camera", view?: front|back|left|right|top|bottom|iso|fit, orbit?: {yaw_deg, pitch_deg},
                   zoom?: factor (>1 closer), look_at?: [x,y,z], asset_ids?}
{action: "navigate", screen?: "home"|"workspace", step?: "capture"|"clean"|"align"|"mesh"|"measure"|"export",
                     panel?: string}
{action: "open", panel}                          (legacy; process→clean, inspect→measure, capture, autopilot)
{action: "tool", tool: "navigate"|"box"|"lasso"|"measure"|"brush"|"pivot"}
{action: "section", enabled, axis: "x"|"y"|"z", position, flip}
{action: "highlight", asset_id, resolved (Contract 2 resolved region), label?, color?}   {action: "clear_highlight"}
{action: "measure_overlay", items: [{label, kind, value, unit, a: [3], b: [3]}], replace?: bool}
{action: "annotate", items: [{position: [3], text}]}          {action: "clear_annotations"}
{action: "follow_scanner", on: bool}
{action: "layout", assets_open?, panel_open?, assistant_open?}
{action: "project", project_id}
```

Context sent with every chat message (`context` in `POST /api/assistant/chat`):
`{active_id, selected_ids, visible_ids, units, project_id, screen, step, tool, theme, camera: {position, target, up,
fov, projection}, section: {enabled, axis, position}, measurements: [{label, value, a, b}], highlight?: {asset_id,
label, count}, selection?: {view_projection, polygon, asset_ids, count}}`.
