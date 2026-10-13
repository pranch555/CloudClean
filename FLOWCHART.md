# How CloudClean works

Flowcharts of what happens when CloudClean runs: from the server starting, to a button press, a live
scan, a chat with the assistant and an update reaching the DGX Spark. Every box names the file (and
usually the function) that does the step, so you can open it and read on.

- **Which file uses which**, file by file, is in [docs/repo-map.md](docs/repo-map.md). It is generated
  from the code (`python tools/repo_map.py`).
- **Why** it is built this way is in [DECISIONS.md](DECISIONS.md).
- Paths are relative to the repo. Backend files are under `cloudclean/`, frontend files under
  `frontend/src/`.

## Contents

1. [The whole system](#1-the-whole-system)
2. [Starting up](#2-starting-up)
3. [A button press: from the UI to a new model](#3-a-button-press-from-the-ui-to-a-new-model)
4. [A live scan with the MetroY](#4-a-live-scan-with-the-metroy)
5. [The turntable](#5-the-turntable)
6. [A chat with the assistant](#6-a-chat-with-the-assistant)
7. [Measuring and the golden model check](#7-measuring-and-the-golden-model-check)
8. [Photos](#8-photos)
9. [The screen: frontend structure](#9-the-screen-frontend-structure)
10. [Install, deploy and update](#10-install-deploy-and-update)
11. [Command line and autopilot](#11-command-line-and-autopilot)
12. [What is stored where](#12-what-is-stored-where)

Arrow labels: solid arrows are calls or data handed on, dotted arrows are network traffic, and boxes
marked *thread* or *process* run alongside the rest.

---

## 1. The whole system

One Python server (FastAPI) does everything. The browser only shows things and sends requests.
Heavy work runs in a separate worker process, so the server stays responsive. Hardware and the LLM are
reached from the server, never from the browser.

```mermaid
flowchart LR
  user(["You, in a browser<br/>PC, laptop or phone"])
  subgraph host["DGX Spark, or any PC running CloudClean"]
    server["Web server<br/>web/server.py + 20 route modules"]
    jobs["Job queue (thread)<br/>web/jobs.py"]
    worker["Worker process<br/>one job at a time"]
    engine["Engine<br/>clean, merge, mesh, edit,<br/>measure, golden check, photos"]
    capture["Capture session (thread)<br/>capture/session.py"]
    turn["Turntable manager<br/>capture/turntable/manager.py"]
    agent["Assistant loop<br/>assistant/agent.py"]
    disk[("Workspace folder<br/>models, projects, settings,<br/>accounts, chats")]
    updater["Update timer<br/>deploy/update.sh"]
  end
  scanner["MetroY scanner<br/>USB: commands + video"]
  table["Revopoint turntable<br/>Bluetooth LE"]
  llm["LLM server<br/>OpenAI-compatible"]
  recon["Docker image cloudclean-recon<br/>COLMAP + MapAnything"]
  github["GitHub, branch main"]
  revo["PC with Revo Metro<br/>cloudclean bridge"]

  user -. "HTTP /api, WebSocket, SSE" .-> server
  server --> jobs --> worker --> engine
  worker --> disk
  server --> disk
  server --> capture
  scanner -- "laser frames" --> capture
  capture -- "saved scan" --> disk
  server --> turn -. "Bluetooth" .-> table
  server --> agent -. "chat completions" .-> llm
  agent -- "same jobs as the UI" --> jobs
  worker -- "docker run" --> recon
  revo -. "uploads exports" .-> server
  github -. "git fetch every 2 min" .-> updater
  updater -- "restart when idle" --> server
```

Three ways the browser and server talk:

- **Plain HTTP (`/api/...`)** for almost everything. A long operation answers at once with a *job*,
  and the browser polls `/api/jobs` until it is done.
- **One WebSocket (`/api/capture/stream`)** for the live scan: points, status and guidance, many times
  a second.
- **Server-Sent Events (`POST /api/assistant/chat`)** for the assistant's reply, streamed word by word
  with its tool calls.

---

## 2. Starting up

`cloudclean serve` builds the app, plugs in every feature module by name, and starts the background
threads. The browser then loads the prebuilt UI and asks who is signed in.

```mermaid
flowchart TD
  start(["cloudclean serve<br/>cloudclean.sh, CloudClean.bat or the systemd service"])
  start --> cli["cli.py: main, then cmd_serve"]
  cli --> serve["web/server.py: serve, then uvicorn"]
  serve --> app["web/server.py: create_app"]
  app --> ws["web/workspace.py: Workspace<br/>assets/ and uploads/ folders"]
  app --> jm["web/jobs.py: JobManager<br/>starts the job-queue thread"]
  app --> auth["web/auth.py: AuthStore + sign-in check on every /api request<br/>machine key created on first start"]
  app --> core["Core routes in server.py<br/>/api/assets, upload, clean, merge, mesh, texture, jobs"]
  app --> routers["For each module in ROUTER_MODULES:<br/>create_router, include_router"]
  routers --> side["Side effects while building:<br/>routes_projects: file loose models into My scans<br/>routes_capture: CaptureManager<br/>routes_turntable: TurntableManager<br/>routes_autopilot: folder watcher if enabled<br/>routes_assistant: Assistant + chat store"]
  app --> static["Mount web/static at /<br/>the built React UI"]
  serve --> life["Lifespan start-up:<br/>turntable auto-reconnect thread (1, 5, 15, 40 s)<br/>autopilot watcher"]

  browser(["Browser opens the page"]) --> html["index.html, then main.tsx"]
  html --> gate["shell/AuthScreen.tsx: AuthGate<br/>GET /api/auth/status"]
  gate -- "not signed in" --> signin["Sign-in page"]
  gate -- "signed in" --> appx["App.tsx boot:<br/>GET /api/params, /api/assets, /api/projects"]
  appx --> screen["Home screen or the last project"]
```

- **Job modules are not imported at start.** `web/jobs.py: job_function` imports the right
  `jobs_*.py` inside the worker process, the first time that kind of job runs.
- **Sign-in check (`require_sign_in`).** A request passes if it has a session cookie (`cc_session`) or
  the machine key header (`X-CloudClean-Key`, used by the bridge and the updater). Otherwise it gets
  `401 Sign in first`. The capture WebSocket checks the same thing itself.

---

## 3. A button press: from the UI to a new model

Every operation that changes or makes a model (Clean, Merge, Mesh, Edit, golden check, photos…)
follows this path. Clean is the example. Quick answers (measuring, status, export to a folder) skip the
job and answer straight away.

```mermaid
sequenceDiagram
  autonumber
  participant Step as steps/CleanStep.tsx
  participant Jobs as lib/jobs.ts
  participant Route as web/server.py /api/clean
  participant Queue as web/jobs.py JobManager (thread)
  participant Worker as Worker process
  participant Engine as clean.py
  participant Disk as web/workspace.py
  participant View as viewer/Viewer.ts

  Step->>Jobs: submitJob /api/clean with asset, preset, settings
  Jobs->>Route: POST /api/clean
  Route->>Route: check the settings (400 with a sentence if wrong)
  Route->>Queue: submit job "clean"
  Route-->>Jobs: the job, status queued
  Queue->>Worker: spawn a process, run job_clean
  Worker->>Disk: load data.ply
  Worker->>Engine: clean_point_cloud
  Engine-->>Worker: cleaned cloud + report
  Worker->>Disk: add_geometry: a NEW asset folder
  Worker-->>Queue: done, new asset id (also log lines and progress)
  loop every 0.7 s while a job runs, 2.5 s when idle
    Jobs->>Route: GET /api/jobs
  end
  Jobs->>Jobs: job finished: refresh models, show the result, toast
  Jobs->>View: sync the visible models
  View->>Route: GET /api/assets/ID/preview (and scalars for colours)
  View-->>Step: model drawn, result card reads GET /api/assets/ID
```

- **The original is never changed.** Every result is a new asset whose `parents` point back to what it
  was made from. That is also why the viewer never has to invalidate anything: new id, new model.
- **Where each operation goes:**

  | UI action | Route | Job kind | Job body | Engine |
  |---|---|---|---|---|
  | Clean | `/api/clean` | `clean` | `web/jobs.py` | `clean.py` |
  | Merge, check merge | `/api/merge`, `/api/merge/assess` | `merge`, `merge_assess` | `web/jobs.py`, `jobs_merge.py` | `register.py`, `markers.py` |
  | Mesh | `/api/mesh` | `mesh` | `web/jobs.py` | `mesh.py` |
  | Edit, Floor, true size, smooth | `/api/edit` | `edit` | `jobs_edit.py` | `edit.py` |
  | Export to the server | `/api/export` | none: answered at once | `jobs_edit.py: export_asset` | `io.py` |
  | Fill holes | `/api/holes/fill` | `fill_holes_selected` | `jobs_holes.py` | `holes.py` |

  The full list of routes and job kinds is in [docs/repo-map.md](docs/repo-map.md#job-kinds).
- **Cancel.** `POST /api/jobs/ID/cancel` removes a queued job, or stops the worker process of a
  running one.

---

## 4. A live scan with the MetroY

The scanner streams laser frames over USB. CloudClean turns each frame into 3D points in a pool of
processes, places it with the markers, averages it into the surface, and streams what it has to the
browser.

```mermaid
flowchart TD
  subgraph usb["MetroY on USB"]
    hidc["Command channel (HID)<br/>metroy/hid.py"]
    video["Video stream<br/>metroy/v4l2.py"]
  end
  subgraph scanner["metroy/scanner.py: MetroyScanner"]
    grab["grab thread<br/>keeps only the newest frame"]
    feed["feed thread"]
    pool["Pool of worker processes:<br/>rectify, markers.detect,<br/>stripes: Steger centres, pair lines<br/>with the scanner's laser maps,<br/>exposure.measure_light"]
    order["Deliver in capture order<br/>(late frames dropped)"]
    ae["exposure.py: AutoExposure<br/>every ~0.8 s"]
    cam["camera.py: CameraControl (thread)<br/>paced register writes"]
  end
  subgraph driver["drivers/metroy_usb.py: read"]
    world["First marker frame sets the world<br/>(flat plate = floor, Z up)"]
    track["markers.py: MarkerTracker<br/>pose, or lost"]
    crop["Keep only the part on the table"]
  end
  subgraph session["capture/session.py (capture thread)"]
    pose["Pose: from markers, ICP,<br/>or dropped (never guessed)"]
    fuse["fusion.py: SurfaceField<br/>grid.py: density and display maps"]
    guide["guidance.py, about 1 Hz:<br/>coverage, holes, hints,<br/>+ turntable scan_warning"]
  end
  video --> grab --> feed --> pool --> order --> ae --> cam --> hidc
  order -- "points + markers" --> world --> track --> crop
  crop -- "Frame: points, pose, distance" --> pose --> fuse --> guide
  fuse -. "WebSocket every 0.1 s:<br/>binary points + status JSON" .-> browser["features/capture/captureStore.ts"]
  guide -.-> browser
  browser --> hud["GuidanceHud.tsx, CaptureStep.tsx,<br/>Viewer live cloud"]
```

**Steps around the stream:**

```mermaid
flowchart TD
  open["Scan step opens<br/>captureStore: WebSocket /api/capture/stream,<br/>GET /api/capture/drivers"]
  open --> connect["Connect: POST /api/capture/connect<br/>capture/manager.py: new CaptureSession,<br/>driver reads calibration off the scanner"]
  connect --> preview{"Camera view open?"}
  preview -- "yes" --> lease["Preview: laser on, nothing recorded<br/>POST /api/capture/camera/preview renews a 6 s lease<br/>GET /api/capture/camera/view: JPEG pictures"]
  preview -- "no" --> startscan
  lease --> startscan["Start: POST /api/capture/start<br/>the preview becomes the scan, no restart"]
  startscan --> running["Scanning: frames, guidance, banner"]
  running --> save["Save: POST /api/capture/save<br/>fusion.project moves points onto the averaged surface"]
  save --> asset["workspace.add_geometry: new scan<br/>+ density colour"]
  asset --> auto{"Autopilot on capture?<br/>(on by default)"}
  auto -- "yes" --> ap["Job autopilot: clean, mesh, export"]
  auto -- "no" --> done(["Scan in the model list"])
  running --> discard["Discard: POST /api/capture/discard"]
```

**Other scanners** use the same session:
- `simulated` makes frames of a known part.
- `realsense` / `orbbec` give frames without a pose; the session places them with ICP.
- `folder` and `revo_bridge` deliver whole scans. The Revo Metro PC runs `cloudclean bridge`
  (`capture/bridge.py`), which uploads each export to `/api/capture/bridge/upload`.
- Whole scans are checked against the first one (`register.assess_pair`). A doubtful scan waits for
  your decision (`/api/capture/pending/ID`).

---

## 5. The turntable

The browser polls the table's status and sends commands. The server keeps the Bluetooth link on its
own thread and can turn the table for as long as a scan runs.

```mermaid
flowchart TD
  ui["lib/turntable.ts<br/>status every 350-500 ms, commands"] -- "/api/turntable/*" --> routes["web/routes_turntable.py"]
  routes --> mgr["capture/turntable/manager.py: TurntableManager"]
  mgr --> pick{"Which table?"}
  pick -- "real" --> ble["ble.py: Bluetooth on its own asyncio thread<br/>protocol.py: commands and replies"]
  pick -- "practice" --> sim["simulated.py<br/>(labelled, never pre-selected)"]
  ble -. "Bluetooth LE" .-> table(["REVO_DUAL_AXIS_TABLE"])
  mgr --> spin["Turn while scanning: spin thread<br/>turns while the scan runs, holds when it stops"]
  spin -- "reads the scan state" --> capmgr["web/routes_capture.py: manager_for"]
  mgr --> prog["program.py: ProgramRunner thread<br/>a scanner that sweeps (MetroY) gets continuous turns"]
  prog -- "pause / resume / start" --> capmgr
  mgr --> warn["scan_warning: not connected, reconnecting, standing still"]
  warn --> banner["Capture guidance code turntable<br/>shown on the scan banner"]
  boot(["Service start"]) --> recon["Auto-reconnect thread<br/>tries after 1, 5, 15, 40 s<br/>then restores Turn while scanning"]
  recon --> mgr
```

The table that was connected and the spin setting are remembered in `settings.json`, so a restart
(every update restarts the service) brings both back.

---

## 6. A chat with the assistant

The assistant is a loop: the model answers, may call tools, sees the results and answers again. Tools
use the same jobs and workspace as the buttons, and `ui` events move the app on screen.

```mermaid
sequenceDiagram
  autonumber
  participant You as Ask bar, Assistant tab or floating chat
  participant Store as features/assistant/assistantStore.ts
  participant Route as web/routes_assistant.py
  participant Agent as assistant/agent.py
  participant LLM as assistant/llm.py to the LLM server
  participant Tools as assistant/tools.py + tools_v3.py
  participant Jobs as web/jobs.py
  participant UI as lib/applyUi.ts

  You->>Store: message (+ photos, kept as project photos)
  Store->>Route: POST /api/assistant/chat with message and what is on screen
  Route->>Agent: run, answer streamed as Server-Sent Events
  Agent->>Agent: load the chat, build the state block (project, models, scan, table)
  Agent->>Agent: routing.py picks the tools for this turn (at most 12 on a small model)
  loop at most 12 steps
    Agent->>LLM: chat completions, streamed
    LLM-->>Store: words as they come (delta events)
    alt the model calls a tool
      Agent->>Tools: call_tool
      opt a job tool (clean, mesh, golden check ...)
        Tools->>Jobs: submit, then follow it (tool_progress events)
      end
      opt a screen action (show, measure overlay, guide ...)
        Tools-->>UI: ui event, the app navigates, highlights or draws
      end
      Tools-->>Agent: short result (under 2 KB) for the model
    else no tool call
      Agent-->>Store: done, chat saved in workspace/assistant/
    end
  end
  Store->>Store: refresh models and projects
```

- **Guardrails** (`tools.py`): no scaling unless you ask; merge only after a check and your OK; delete
  only through a confirm card; exports never overwrite; the table only moves on request.
- **"Where is…?"** `app_guide` looks the feature up in `assistant/guide.py`, then `lib/guide.ts`
  navigates there and `ui/GuideSpotlight.tsx` rings the control.

---

## 7. Measuring and the golden model check

Quick measurements answer straight away. Comparisons that take longer run as jobs.

```mermaid
flowchart LR
  subgraph direct["Answered at once (no job)"]
    size["Part size<br/>PartSize.tsx: GET /api/assets/ID/summary<br/>understand.get_summary"]
    tools["Tool grid: extent, caliper, heights & steps,<br/>diameter, sphere, plane, angle, section<br/>Tools.tsx: POST /api/measure/*<br/>understand.py + regions.py + fitting.py"]
    pick["Two-point measure<br/>Labels.tsx: /api/measure/snap, /distance<br/>accuracy.SurfaceModel, analysis.point_distance"]
    thread["Thread tab: /api/measure/thread<br/>analysis.thread_analysis"]
    drift["Size kept? /api/accuracy/drift<br/>accuracy.operation_drift"]
  end
  subgraph job["Run as a job"]
    gold["Golden model: POST /api/golden-check<br/>job golden_check"]
    cad["CAD compare: /api/compare<br/>job compare"]
    two["Compare two scans: /api/accuracy/compare-scans<br/>job compare_scans"]
  end
  gold --> g1["jobs_golden.py: drop points filled from photos"]
  g1 --> g2["golden.py: check_against_golden"]
  g2 --> g3["compare.py: line up, signed distance<br/>(moves and turns, never rescales)"]
  g3 --> g4["Spot verdicts: matches, off, rough, not scanned"]
  g4 --> g5["Flat and round faces measured,<br/>ok / off / too close to call (ISO 14253-1)"]
  g5 --> g6["golden_words.py: names from the part<br/>head, shaft, hex socket ..."]
  g6 --> g7["Two new models: aligned scan + check mesh<br/>report v3 and colour fields"]
  g7 --> g8["GoldenCheck.tsx: dial, surface bar, areas, sizes<br/>GoldenPins.tsx + Viewer.highlight"]
```

---

## 8. Photos

Photos are kept as project photos the moment they are added, and can then be used three ways.

```mermaid
flowchart TD
  add["Add photos: drop, upload, or attach in the chat<br/>POST /api/upload: Workspace.add_image<br/>(HEIC turned into JPEG)"] --> photos[("Project photos")]
  photos --> p3d["Photos to 3D: POST /api/photos-to-3d<br/>job photos_to_3d"]
  photos --> col["Colour from photos: POST /api/colour-from-photos<br/>job colour_from_photos"]
  photos --> fill["Fill from photos: POST /api/fill-from-photos<br/>job fill_from_photos"]
  p3d --> docker["docker run cloudclean-recon<br/>tools/recon/photos_to_3d.py:<br/>COLMAP places the photos, scale sheet sets true size,<br/>MapAnything makes the points"]
  docker --> m1["New model: coloured points<br/>for looking, not for measuring"]
  col --> docker2["Same image, COLMAP cameras only"]
  docker2 --> align["photo_align.py: line up with your scan<br/>photo_check.py: do the photos agree? refuse if not"]
  align --> tex["texture.py: colour the points or the mesh<br/>(+ texture image)"]
  tex --> m2["New coloured model, same shape and size"]
  fill --> docker2b["Same image, COLMAP cameras"]
  docker2b --> pf["photo_fill.py: add photo points only where the scan has none"]
  pf --> m3["New model, every point marked scanned or filled<br/>(the golden check ignores filled points)"]
  manual["Manual colour: line one photo up by hand<br/>ColourSection.tsx: POST /api/texture"] --> tex
```

---

## 9. The screen: frontend structure

What is on screen, and where its data comes from.

```mermaid
flowchart TD
  main["main.tsx: fonts and styles"] --> gate["shell/AuthScreen.tsx: AuthGate"]
  gate --> app["App.tsx"]
  app --> top["shell/TopBar.tsx<br/>project, Journey (six steps), jobs, theme, settings, Ask"]
  app --> home["home/HomeScreen.tsx<br/>projects, ask prompt, scanner status"]
  app --> ws["Workspace (kept mounted behind Home)"]
  app --> over["Floating: FloatingChat, JobsDrawer, Toasts,<br/>DropOverlay, SettingsDialog, Tour,<br/>TooltipLayer, GuideSpotlight"]
  ws --> models["shell/ModelsPanel.tsx<br/>models in five groups (shell/models/)"]
  ws --> vp["viewport/Viewport.tsx"]
  ws --> side["shell/SidePanel.tsx"]
  side --> asst["Assistant tab<br/>features/assistant/AssistantPanel.tsx"]
  side --> steps["One step:<br/>Scan (steps/capture/), Clean, Align, Mesh,<br/>Measure (steps/measure/), Export"]
  vp --> viewer["viewer/Viewer.ts (three.js)<br/>controls.ts, edl.ts shading, pointMaterial.ts,<br/>colormaps.ts, ViewCube.ts, levelling.ts"]
  vp --> overlays["Overlays: Labels, GoldenPins, GoldenDimension,<br/>Selection, Brush, Pairs, Panes, SizeTag, Rails,<br/>FloorPanel, Legend, GuidanceHud, Banners, AskBar"]

  subgraph state["State (zustand stores)"]
    store["store.ts: useStore<br/>models, visible, active model, step,<br/>display, layout, jobs, toasts ..."]
    small["Small stores: captureStore, cameraStore,<br/>assistantStore, chatWindow, lib/turntable,<br/>lib/golden, steps/measure/state"]
  end
  top --> store
  models --> store
  steps --> store
  steps --> small
  store -- "effects push changes" --> viewer
  small --> api["lib/api.ts<br/>fetch, upload, Server-Sent Events"]
  store --> api
  api -. "/api" .-> backend(["Backend"])
```

- **Polling:** jobs (`lib/jobs.ts`), the turntable status (`lib/turntable.ts`), the camera picture
  (`cameraStore.ts`).
- **Streaming:** the live scan (`captureStore.ts`, WebSocket) and chat replies (`assistantStore.ts`,
  Server-Sent Events).
- **Remembered per browser (localStorage `cloudclean.*`):** screen, step, project, theme, text size,
  layout, up axis, rotate style, units, shading.

---

## 10. Install, deploy and update

Code is edited on a PC, pushed to GitHub, and the Spark updates itself, but only when nothing is
running.

```mermaid
flowchart TD
  edit(["Change code on the PC"]) --> fe{"frontend/ changed?"}
  fe -- "yes" --> build["npm run build<br/>vite.config.ts writes cloudclean/web/static"]
  fe -- "no" --> test
  build --> test["pytest tests<br/>(also checks docs/repo-map.md is current)"]
  test --> push["git commit + git push origin main"]
  push --> gh[("GitHub")]
  timer(["Spark: update timer, every 2 min<br/>install-autoupdate.sh set it up"]) --> upd["deploy/update.sh"]
  gh -. "git fetch" .-> upd
  upd --> bad{"Newest commit marked bad,<br/>or hand edits on the Spark?"}
  bad -- "yes" --> stop(["Do nothing"])
  bad -- "no" --> busy{"Busy? asks the server with the machine key:<br/>capture/status, jobs, assistant/busy, turntable/status"}
  busy -- "yes" --> wait(["Try again in 2 min"])
  busy -- "no" --> backup["Back up the old code to ~/cc-backups (20 kept)"]
  backup --> merge["git merge --ff-only"]
  merge --> deps{"pyproject.toml changed?"}
  deps -- "yes" --> setup["cloudclean.sh setup: reinstall"]
  deps -- "no" --> restart
  setup --> restart["systemctl --user restart cloudclean"]
  restart --> up{"Up within 90 s?"}
  up -- "yes" --> live(["New version live"])
  up -- "no" --> roll["Mark the commit bad, go back to the old code, restart"]
```

**First install on a machine:**
- `cloudclean.sh` (Linux, macOS, Spark) or `CloudClean.bat` (Windows) makes `.venv`, installs, and
  runs `cloudclean serve`.
- `deploy/install-service.sh` installs a systemd user service that starts at boot.
- `deploy/install-autoupdate.sh` installs the update timer.
- `sudo deploy/install-scanner-access.sh` lets the service open the MetroY with nobody logged in.
- Docker (`Dockerfile`, `compose.yaml`) is the alternative without USB, Bluetooth or photos → 3D.

---

## 11. Command line and autopilot

```mermaid
flowchart LR
  cli["cli.py"] --> serve["serve: the web app (section 2)"]
  cli --> one["info, clean, merge, mesh, texture:<br/>one engine step on files"]
  cli --> run["run: pipeline.py run_pipeline<br/>clean each scan, check merge, mesh, export"]
  cli --> watch["watch: autopilot.py<br/>a folder, in the terminal"]
  cli --> bridge["bridge: capture/bridge.py<br/>on the Revo Metro PC"]
  subgraph web["Autopilot in the web app"]
    src["Folder watcher thread, POST /api/autopilot/run,<br/>/api/autopilot/upload, or a saved scan"] --> job["Job autopilot<br/>jobs_autopilot.py"]
    job --> steps["import, clean, check merge,<br/>merge or best scan, mesh,<br/>compare with a reference, export files"]
    steps --> ask{"Needs a decision?"}
    ask -- "yes" --> wait["Stops: Settings, Automations asks you<br/>POST /api/autopilot/decide continues"]
    ask -- "no" --> out(["Files in the output folder + report.json"])
  end
```

---

## 12. What is stored where

```text
<workspace>/                      ~/cloudclean-workspace on the Spark
├── assets/<12-hex id>/           one folder per model or photo, never changed after creation
│   ├── meta.json                 name, kind, operation, parents, params, stats, project, part frame
│   ├── data.ply                  full resolution
│   ├── preview.ply               what the browser loads (≤ 1.5 M points / ≤ 600 k triangles)
│   ├── report.json               what the operation did and measured
│   ├── scalars_<name>.npy        a value per point (deviation, density …) + .preview.bin for the browser
│   └── thumbnail.png, summary.json, markers.json, textured.glb, image.<ext> …
├── projects.json                 one project per physical part, golden model per project
├── settings.json                 one section per feature (capture, turntable, assistant, autopilot …)
├── auth/                         users, sessions (fingerprints only), activity log, machine_key
├── assistant/<id>.json           chats (no image data, only captions)
├── recon/<job>/                  photo reconstruction working folders
└── uploads/                      files being imported, bridge uploads
~/.cache/cloudclean/metroy/       scanner calibration per serial number, recordings for replay
```
