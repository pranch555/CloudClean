# CloudClean architecture (v0.2)

CloudClean is a Python package (`cloudclean/`) with an Open3D processing engine, a FastAPI server and a
browser frontend (React + TypeScript + three.js, source in `frontend/`, built into `cloudclean/web/static/`).
The production target is an **NVIDIA DGX Spark** (ARM64, Ubuntu, 128 GB unified memory) serving the UI to
other machines on the LAN. Development also happens on Windows x64. Everything must work on both.

Accuracy is the top priority: no operation may silently rescale, resample or move data unless that is what
the operation is for, and every operation reports what it did.

## Core engine (unchanged contracts)

| Module | Purpose |
|---|---|
| `io.py` | `load(path, repair=True)` / `save(geom, path)` / `describe(geom)` / `preview(geom)` / `to_cloud(geom)` / `estimate_spacing(points)`. `CAD_EXTS` (.step .stp .iges .igs .brep) are routed to `cad.load_cad`. `has_blank_colors` / `drop_blank_colors` discard an all-black colour array on both load and save (see below); `load(..., repair=False)` skips that, for maintenance tools only. |
| `clean.py` | `CleanParams`, `clean_point_cloud(pcd, params, log) -> (pcd, report)` |
| `register.py` | `MergeParams`, `merge_geometries(geoms, params, pairs, log)`, `align_pair`, `refine_icp`, `global_candidates`, `rigid_from_pairs` |
| `mesh.py` | `MeshParams`, `reconstruct_mesh(geom, params, log) -> (mesh, report)`, `surface_deviation`, `cleanup_mesh` |
| `texture.py` | photo colouring |
| `params.py` | `ParamsMixin` (dataclass params with `from_dict`, `update`, `to_dict`) |
| `pipeline.py` | `run_pipeline(...)` used by `cloudclean run` |

### Blank vertex colours

Poisson reconstruction always emits one colour per vertex, zero-filled when the source cloud had no texture (most
Revopoint scans). Stored, that array is indistinguishable from a genuinely black scan: a viewer multiplies the lit
surface by 0 and the part renders as an unlit silhouette. `io.save` and `io.load` therefore drop a colour array whose
maximum is 0, and `mesh.reconstruct_mesh` drops it before `describe`, so `has_colors` is reported truthfully.
Workspaces written before this can be repaired in place with `python tools/repair_colors.py <workspace>`
(`--dry-run` lists what it would touch).

`log` is any `Callable[[str], None]`. Inside a job it is a `JobLog` which additionally has
`log.progress(fraction, label=None)` and `log.output(**values)`. Engine functions must only *call* `log(msg)`;
job bodies may use `getattr(log, "progress", None)`.

## Web layer

- `web/workspace.py` – `Workspace`: every scan / result / photo is an asset folder `assets/<12 hex id>/` with
  `meta.json`, `data.ply` (full resolution), `preview.ply` (≤1.5 M points / ≤600 k triangles), optional
  `report.json`. `ws.add_geometry(geom, name, operation, parents, params, report) -> meta`,
  `ws.load_geometry(id)`, `ws.get(id)`, `ws.update(id, **fields)`, `ws.list()`, `ws.delete(id)`,
  `ws.add_image(...)`, **`ws.add_scalars(id, name, values, unit, description)`** (one float per point/vertex of
  data.ply; also writes a preview-aligned float32 file; listed in `meta["scalars"]`), `ws.scalars_path(id, name)`.
- `web/jobs.py` – `JobManager.submit(kind, title, payload) -> job`, `.get(id)`, `.list()`, `.cancel(id)`,
  **`.wait(id, timeout)`**. Jobs run one at a time in a spawned worker process. Job body signature:
  `def job_x(ws: Workspace, payload: dict, log: JobLog) -> list[str]` (created asset ids).
  A job dict has `id kind title status(queued|running|done|failed|cancelled) created started finished logs
  result error payload progress{fraction,label} output{}`.
  **Feature modules register job kinds** by defining `JOBS = {"kind": fn}` in a module listed in
  `jobs.JOB_MODULES` (`web/jobs_edit.py`, `web/jobs_compare.py`, `web/jobs_autopilot.py`).
- `web/server.py` – core routes (`/api/params`, `/api/assets…`, `/api/upload`, `/api/import-paths`, `/api/clean`,
  `/api/merge`, `/api/mesh`, `/api/texture`, `/api/pipeline`, `/api/jobs…`,
  `GET /api/assets/{id}/scalars/{name}` → float32 LE aligned with preview.ply).
  **Feature modules add routes** via `create_router(workspace, jobs) -> fastapi.APIRouter` in a module listed in
  `server.ROUTER_MODULES` (`web/routes_edit.py`, `routes_compare.py`, `routes_assistant.py`,
  `routes_capture.py`, `routes_autopilot.py`). Router factories may keep state in closures.
- `web/settings.py` – `Settings(ws.root).get(section, defaults)` / `.update(section, values)` →
  `<workspace>/settings.json`.

Rules for feature modules: validate input and raise `HTTPException(400, msg)` with a human sentence; long work
goes into jobs, never into request handlers; never block the event loop (use `def` handlers, which FastAPI runs
in a thread pool, or `asyncio.to_thread`).

---

## Feature 1 – Geometry editing (`cloudclean/edit.py`, job kind `edit`)

`apply_edits(geom, ops: list[dict], log) -> (geom, report)` applies ops in order. Every op validates its
arguments and raises `ValueError` with a clear message. Report: `{"ops": [{"op", "before", "after", ...}],
"transform": 4x4 accumulated rigid/similarity transform or null}`.

| op | args | clouds | meshes |
|---|---|---|---|
| `crop_box` | `min[3] max[3] invert=false` | ✓ | ✓ (drops vertices + their triangles) |
| `cut_plane` | `point[3] normal[3] keep="positive"\|"negative"` | ✓ | ✓ |
| `select_screen` | `view_projection[16]` (three.js column-major `projectionMatrix * matrixWorldInverse`), `polygon[[x,y]…]` in NDC (-1..1), `mode="delete"\|"keep"`, `visible_only=false` | ✓ | ✓ (a triangle is selected when all its vertices are) |
| `delete_sphere` | `center[3] radius` | ✓ | ✓ |
| `transform` | `matrix` 4x4 row-major | ✓ | ✓ |
| `translate` | `offset[3]` | ✓ | ✓ |
| `rotate` | `axis[3]` or `"x"\|"y"\|"z"`, `degrees`, `center="centroid"\|"origin"\|[3]` | ✓ | ✓ |
| `scale` | `factor`, `center="origin"\|"centroid"\|[3]` (explicit user request only) | ✓ | ✓ |
| `mirror` | `axis "x"\|"y"\|"z"`, `center` | ✓ | ✓ (fixes winding) |
| `center` | `mode="bbox"\|"centroid"\|"bbox_bottom"`, `up="z"\|"y"` | ✓ | ✓ |
| `align_principal` | – (PCA axes → X longest, Y, Z shortest) | ✓ | ✓ |
| `align_floor` | `up="z"\|"y"` (fit support plane / flattest large face, rotate so it is the floor at 0) | ✓ | ✓ |
| `downsample` | `voxel` | ✓ | – |
| `remove_outliers` | `neighbors=24 std_ratio=2.0` | ✓ | – |
| `remove_small_components` | `min_ratio=0.02` | ✓ (DBSCAN) | ✓ |
| `simplify` | `target_triangles` or `ratio` | – | ✓ (quadric) |
| `smooth` | `iterations=5 method="taubin"\|"laplacian"` | – | ✓ |
| `fill_holes` | `max_hole_size` (scan units, 0 = all) | – | ✓ (o3d.t fill_holes) |
| `subdivide` | `iterations=1` | – | ✓ (loop) |
| `repair` | – (degenerate/duplicate/non-manifold cleanup) | – | ✓ |
| `flip_normals` / `recompute_normals` | – | ✓ | ✓ |
| `to_pointcloud` | `samples=0` (0 = vertices) | – | ✓ |
| `paint` | `rgb[3] 0..1` | ✓ | ✓ |

API (`routes_edit.py`):
- `POST /api/edit {asset_id, ops, name?}` → job (validates op names/args before submitting).
- `GET /api/edit/ops` → op catalogue `{op: {args: {name: {type, default, description}}, applies_to: [...], description}}` (the assistant and UI both use this).
- `POST /api/export {asset_id, format, folder, filename?}` → writes a file on the **server** disk, returns `{path}` (formats as `EXPORT_FORMATS`). Refuses to overwrite unless `overwrite: true`.
- `POST /api/measure {asset_id, points: [[x,y,z]…]}` → snaps each point to the nearest full-resolution point / surface, returns snapped points, segment lengths, total, and angle when 3 points.

## Feature 2 – Scan vs CAD inspection (`cloudclean/cad.py`, `cloudclean/compare.py`, job kind `compare`)

- `cad.load_cad(path, tolerance=None) -> TriangleMesh` tessellates STEP/IGES/BREP with `cascadio`
  (works on Windows x64 and Linux aarch64). Keep file units (mm). Fall back to OCP (`cadquery-ocp-novtk`) when
  available. Clear error if neither is installed.
- `CompareParams(ParamsMixin)`: `align="auto"|"icp"|"none"`, `tolerance=0.1`, `max_distance=0.0`
  (0 = auto: 3 % of reference diagonal; farther points are excluded as not part of the part),
  `sample_points=0` (0 = all), `estimate_scale=True` (report only, never applied), `coverage_threshold=0.0`
  (0 = auto from scan spacing).
- `compare_to_reference(scan, reference_mesh, params, log) -> dict` with `aligned` (scan cloud in CAD frame),
  `deviation` (signed float per aligned point, + = outside material, NaN = excluded), `reference_distance`
  (per CAD vertex distance to nearest scan point, for coverage), `report`.
- Report: `transform`, `alignment {method, fitness, rmse, candidates}`, `stats {points, excluded, mean, std,
  rms, abs_mean, min, max, p05, p95, within_tolerance_pct}`, `histogram {edges, counts}`,
  `dimensions {reference[3], scan[3], difference[3]}` (axis-aligned in CAD frame), `coverage
  {covered_area_pct, threshold}`, `scale_estimate {factor, percent}`, `tolerance`, `params`.
- Job `compare` payload `{scan_id, reference_id, params}` creates:
  1. `"<scan> vs <ref>"` point cloud, operation `compare`, parents `[scan, ref]`, report, scalar `deviation`
     (unit = scan units).
  2. `"<ref> · coverage"` mesh copy of the reference, operation `compare`, scalar `reference_distance`.
- API (`routes_compare.py`): `POST /api/compare`, `GET /api/assets/{id}/inspection-report` (self-contained HTML
  report: stats, histogram SVG, dimension table, pass/fail against tolerance).

## Feature 3 – Assistant (`cloudclean/assistant/`)

Local LLM through any **OpenAI-compatible** endpoint (vLLM, SGLang, Ollama, llama.cpp `--jinja`, NIM,
LM Studio). Default `base_url = http://localhost:8000/v1`, model = first entry of `/v1/models` when empty.
Recommended on DGX Spark: vLLM with `nvidia/Qwen3.6-35B-A3B-NVFP4` (`--enable-auto-tool-choice
--tool-call-parser qwen3_xml --reasoning-parser qwen3`) or `gpt-oss-120b`; Ollama for the quickest start.

- Settings section `assistant`: `base_url, model, api_key, vision(false), vision_detail(auto|low|high),
  temperature(0.2), max_steps(12), request_timeout(300)`.
- Tools (JSON-schema function tools) call the same workspace / jobs APIs as the UI: `list_assets`,
  `get_asset` (stats + report summary), `describe_parameters(operation)`, `clean`, `merge`, `mesh`,
  `run_pipeline`, `edit` (ops from Feature 1), `compare_to_reference`, `export_asset`, `rename_asset`,
  `measure`, `ui` (viewer actions, see below), `request_delete` (never deletes; asks the user). Job tools submit a
  job, stream its progress, wait for it and return a compact summary (created asset ids, names, key report
  numbers). Tool results given to the model stay small (< 2 KB).
- The system prompt contains: what CloudClean is, units, the accuracy-first rules, the current workspace
  summary, and the UI context sent by the client (active asset, selected ids, visible ids, current screen
  selection polygon if any, camera).
- `POST /api/assistant/chat` body `{conversation_id?, message, context{active_id, selected_ids, visible_ids,
  units, selection?{view_projection, polygon, asset_id}}, image?: "data:image/png;base64,…",
  images?: ["data:image/(png|jpeg|webp);base64,…"], image_notes?: ["photo of the part", …]}` → **Server-Sent
  Events** (`text/event-stream`), each `event: <type>` + `data: <json>`:
  `conversation {id}` · `delta {text}` · `reasoning {text}` · `tool_start {call_id, name, arguments}` ·
  `tool_progress {call_id, job_id, status, progress, label, last_log}` · `tool_end {call_id, ok, summary,
  asset_ids}` · `ui {action, ...}` · `confirm {call_id, action:"delete", asset_ids, message}` ·
  `error {message}` · `done {}`.
- `ui` actions: `show {asset_ids, exclusive}`, `focus {asset_id}`, `select {asset_ids}`,
  `display {color_mode: original|solid|asset|normal|scalar, scalar?, point_scale?, wireframe?}`,
  `camera {view: front|back|left|right|top|bottom|iso|fit}`, `open {panel: process|inspect|capture|autopilot}`.
- Other routes: `GET/PUT /api/assistant/settings` (never returns the key, only `api_key_set`),
  `GET /api/assistant/status` → `{configured, reachable, model, models[], error}`,
  `GET /api/assistant/conversations`, `GET/DELETE /api/assistant/conversations/{id}`,
  `POST /api/assistant/stop {conversation_id}`.
- Images: the user may attach photos of the real part, viewer screenshots or annotated pictures. `image` (one) and
  `images` (many) are merged into one list, `image_notes[i]` captions image *i*; at most 6, each ≤ 9 MB,
  ≤ 24 MB together, otherwise HTTP 400. Each image is sent as an OpenAI `image_url` part preceded by a
  `[image N: caption]` text part (`image_url.detail = vision_detail` unless `auto`). With `vision: false` no image
  is sent and the reply starts with a note saying how many were dropped. The stored conversation never contains
  image data: `messages` keeps the `[image N: caption]` placeholders (so follow-up questions still have the
  captions) and the transcript entry carries `images: [{caption}]`; later turns re-send nothing.
- Conversations persist in `<workspace>/assistant/<id>.json`.

## Feature 4 – Live capture (`cloudclean/capture/`)

Revopoint publishes no SDK for MetroY Ultra and Revo Metro runs only on Windows/macOS, so capture is a
**driver architecture**:

- `ScannerDriver` interface: `info() -> {id, name, kind, description, available, reason}`,
  `settings_schema()`, `connect(settings)`, `start()`, `read(timeout) -> Frame | None`, `stop()`,
  `disconnect()`. `Frame`: `points (N,3) float32` (world or sensor frame), `colors?`, `normals?`,
  `pose? 4x4` (sensor→world when the device tracks), `timestamp`, `meta {distance_mm, exposure, tracking,…}`.
- Drivers: `simulated` (synthetic part on a turntable / hand-held path, with noise, for demos and tests),
  `revo_bridge` (the Windows/macOS PC running Revo Metro pushes each exported scan to CloudClean – see bridge),
  `folder` (watch a local/network folder for new scans), `revopoint_sdk` (loads Revopoint's SDK when the user
  obtains it; reports `available: false` with instructions otherwise), `realsense` / `orbbec` (optional
  generic depth cameras that have ARM64 SDKs).
- `CaptureSession`: frame-to-model tracking (point-to-plane ICP against a voxel map) for frames without pose,
  fusion into a voxel map, and **live guidance**: tracking quality, motion speed, working distance, frame
  density, total coverage, holes/sparse regions with a suggested viewing direction, completeness score.
- API (`routes_capture.py`): `GET /api/capture/drivers`, `POST /api/capture/connect {driver, settings}`,
  `POST /api/capture/start|pause|resume|stop|discard`, `POST /api/capture/save {name, auto_process}` (saves the
  fused cloud as an asset, optionally submits autopilot), `GET /api/capture/status`,
  `WS /api/capture/stream` (binary messages = new display points: header + float32 xyz + uint8 rgb; text
  messages = JSON status/guidance ~5 Hz), `POST /api/capture/bridge/upload` (multipart, from the bridge).
- `cloudclean bridge --server http://spark:8765 --watch <Revo export folder>` runs on the scanning PC and uploads
  every new export automatically.

## Feature 5 – Autopilot (`cloudclean/autopilot.py`, job kind `autopilot`)

Hands-free: scan files in → cleaned, merged, meshed, (optionally inspected) and exported out.

- Settings section `autopilot`: `watch_enabled, watch_folder, output_folder, recursive, settle_seconds(5),
  group_seconds(30)` (files arriving together form one item; a sub-folder is one item), `formats ["stl","ply"]`,
  `preset, remove_plane, merge_method, mesh {method, watertight, depth, smooth_iterations, target_triangles}`,
  `reference_id` (CAD to compare against, optional), `auto_on_upload(false)`, `auto_on_capture(true)`.
- Job `autopilot` payload `{asset_ids? , paths?, name, settings}`: import → clean → merge (≥2) → mesh →
  compare (if reference) → export every format to `output_folder/<name>/` + `report.json`; `log.progress` for
  each stage; `log.output(exports=[paths], mesh_id, compare_id)`.
- `Watcher` thread (polling, no extra dependency) started by `routes_autopilot.create_router` when enabled.
- API: `GET/PUT /api/autopilot/settings`, `GET /api/autopilot/status` (`watching, pending_items, history[]`),
  `POST /api/autopilot/run {asset_ids, name?}`.
- CLI: `cloudclean watch <folder> -o <out> [pipeline options]` (no web server needed).

## Frontend notes

- **Layout grid** (`App.tsx`, `styles/layout.css`). The app is one CSS grid whose seven columns are
  `assets · resizer · stage · resizer · inspector · resizer · assistant`; a closed panel keeps its track at `0px`.
  Panels that are closed still render a `.panel-stub`, because a `display: none` child leaves the grid entirely and
  every later panel would auto-place one column to the left — which put the viewport in a 4 px resizer track.
  Anything added to `.app` must either be a grid item in the right order or be taken out of flow (`position: fixed`).
- **`lib/uid.ts`.** `crypto.randomUUID` exists only in a secure context, so it is undefined over plain HTTP — how
  CloudClean is normally reached on a lab network or over Tailscale. Use `uid()` for client-side keys
  (attachments, measurements, edit-queue steps); calling `crypto.randomUUID` directly breaks those features off
  localhost.
- **`styles/polish.css`** is loaded last and only refines the base system (depth, gradients, active states, motion).
  It changes no ink or surface token, so the validated contrast of `tokens.css` still holds.
