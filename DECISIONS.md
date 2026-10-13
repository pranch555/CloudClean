# Decisions

Why CloudClean is built the way it is, task by task: what was asked, what was done, why (with the
measurements behind it), what was tried and rejected, and what is still open. Newest first.

How the code fits together is in [FLOWCHART.md](FLOWCHART.md) (how it works) and
[docs/repo-map.md](docs/repo-map.md) (every file and its connections, generated from the code).

- Entries from 2026-09-29 on are backed by commits (hashes after the 2026-10-12 history rewrite).
- Earlier work was squashed into the first public commit (f0d2759). Those entries are rebuilt from the
  docs in `docs/` and the working notes, and say where each part is documented.
- **Keeping it current:** every task that changes the repo adds an entry at the top of the log (newest
  first) and a row in the index, in the same commit. Use the template at the end of this file.

## Rules behind every decision

These come up again and again below. They are the reasons behind most choices.

1. **Accuracy first.** No operation may rescale, resample or move data unless that is its purpose, and
   every operation reports what it did. Results stay in the scanner's units (mm). Filter distances are
   multiples of the scan's own point spacing, so the same settings work at any resolution. The only
   rescale is the explicit "Set the true size" for models made from photos.
2. **Measure before claiming.** A change ships with the number it was checked against: a synthetic part
   with known geometry, a replay of a real recording, or the real scanner. When something is not
   validated, the code and docs say so (`validated: false`, `LONG_PULSES_VALIDATED = False`).
3. **Never guess calibration or poses.** Calibration is read off the scanner. A frame whose pose is
   unknown is dropped, not placed by guess.
4. **Plain words, one obvious next step.** The user found v2 generic and confusing. Labels use the
   user's words, buttons are verbs, numbers carry units, and status is never shown by colour alone.
5. **The assistant is the way to find things.** There is no search bar. Every feature has a plain name
   on screen, an entry in `cloudclean/assistant/guide.py` and a `data-guide` id, so the assistant can
   take the user there.
6. **Long work runs in jobs, never in a request.** Routes check input and submit a job; the job runs in
   a worker process; every result is a new asset; the original is never changed.
7. **The DGX Spark is the real install.** Changes reach it through GitHub (push → auto-update when
   idle). Never edit files on the Spark. Never touch the Spark's other projects or their model servers.
8. **Public repo hygiene.** No workspace data, keys, Tailscale address or other projects' names in Git.
   No Claude attribution in commits or PRs.

## Index

| Date | Task | Area |
|---|---|---|
| 2026-10-13 | [Decision log, flowcharts and a generated repo map](#2026-10-13--decision-log-flowcharts-and-a-generated-repo-map) | docs, tooling |
| 2026-10-12 | [Remove Claude from the GitHub history](#2026-10-12--remove-claude-from-the-github-history) | repo |
| 2026-10-09 | [Thinner, fuller native scans (four parallel improvements)](#2026-10-09--thinner-fuller-native-scans-four-parallel-improvements) | capture |
| 2026-10-09 | [The turntable survives restarts](#2026-10-09--the-turntable-survives-restarts) | turntable |
| 2026-10-09 | [Turntable + markers: "camera shows markers but says no markers"](#2026-10-09--turntable--markers-camera-shows-markers-but-says-no-markers) | capture |
| 2026-10-07 | [Camera view, Dark / Shiny laser light, auto exposure](#2026-10-07--camera-view-dark--shiny-laser-light-auto-exposure) | capture |
| 2026-10-07 | [Turntable scans sweep; Turn while scanning](#2026-10-07--turntable-scans-sweep-turn-while-scanning) | turntable |
| 2026-10-07 | [Point clouds "too black": shape shading and height colour](#2026-10-07--point-clouds-too-black-shape-shading-and-height-colour) | 3D view |
| 2026-10-07 | [A Floor to stand any model on](#2026-10-07--a-floor-to-stand-any-model-on) | 3D view |
| 2026-10-07 | [Native scans stand on the turntable plate](#2026-10-07--native-scans-stand-on-the-turntable-plate) | capture |
| 2026-10-07 | [Revo Metro bridge is optional (wording)](#2026-10-07--revo-metro-bridge-is-optional-wording) | UI |
| 2026-10-06 | [First real use of the scanner and table on the Spark](#2026-10-06--first-real-use-of-the-scanner-and-table-on-the-spark) | capture, turntable |
| 2026-10-06 | [Golden check v3 and motion](#2026-10-06--golden-check-v3-and-motion) | golden check, UI |
| 2026-10-06 | [The Spark updates itself from GitHub](#2026-10-06--the-spark-updates-itself-from-github) | deploy |
| 2026-10-05 | [UI rework: model list, golden check v2, floating chat](#2026-10-05--ui-rework-model-list-golden-check-v2-floating-chat) | UI |
| 2026-09-30 | [Outline check: measure from one photo's silhouette](#2026-09-30--outline-check-measure-from-one-photos-silhouette) | outline check |
| 2026-09-29 | [Publish on GitHub with one-step setup](#2026-09-29--publish-on-github-with-one-step-setup) | deploy |
| 2026-09-25 | [Assistant on a 16k-token model](#2026-09-25--assistant-on-a-16k-token-model) | assistant |
| 2026-09-25 | [Golden model check and fill from photos](#2026-09-25--golden-model-check-and-fill-from-photos) | golden check |
| 2026-09-25 | [Accounts](#2026-09-25--accounts) | accounts |
| 2026-09-25 | [Which LLM runs the assistant](#2026-09-25--which-llm-runs-the-assistant) | assistant |
| 2026-09-25 | [Photos: counts, scale sheet, merging from photos (not shipped)](#2026-09-25--photos-counts-scale-sheet-merging-from-photos-not-shipped) | photos |
| 2026-09-24 | [Photos → 3D and colour from photos](#2026-09-24--photos--3d-and-colour-from-photos) | photos |
| 2026-09-24 | [App guide instead of a search bar; photos in chat; stickers](#2026-09-24--app-guide-instead-of-a-search-bar-photos-in-chat-stickers) | assistant, merge |
| 2026-09-24 | ["The sizes are a little off": accuracy investigation](#2026-09-24--the-sizes-are-a-little-off-accuracy-investigation) | accuracy |
| 2026-09-24 | [v3: redesign, part understanding, turntable, assistant controls everything](#2026-09-24--v3-redesign-part-understanding-turntable-assistant-controls-everything) | all |
| 2026-09-18 | [Native MetroY capture without Revo Metro](#2026-09-18--native-metroy-capture-without-revo-metro) | capture |
| before v3 | [Foundations](#before-v3--foundations) | engine, web |

---

## 2026-10-13 · Decision log, flowcharts and a generated repo map

**Asked:** a decision log of why each change was made, and a flowchart of how the files connect and how
they work, for the repo, kept up to date from now on.

**Done:**
- This file (DECISIONS.md), newest first, with an index and an entry template.
- [FLOWCHART.md](FLOWCHART.md): hand-drawn flowcharts of how CloudClean works. They cover the system,
  startup, a button press through a job, live scanning, the turntable, the assistant, measuring,
  photos, the frontend, deploy and the CLI.
- [docs/repo-map.md](docs/repo-map.md), written by `tools/repo_map.py` from the code. It covers every
  file, what it does, what it uses and what uses it, which API routes the browser calls, and which jobs
  those routes start.
- `tests/test_repo_map.py`. It fails when the map is out of date, when a file does not say what it is
  for, or when the UI calls an `/api` path no route serves.
- Header comments (comments only) on about 90 frontend files and one test, so the map can describe each
  of them.

**Why:**
- **The file map is generated, not drawn.** It covers 383 files and about 1,400 links, which change
  every week. A hand-drawn file graph would be wrong within days. Generated from the code, plus a test,
  it cannot silently go stale.
- **The "how it works" flowcharts are drawn by hand.** Runtime behaviour is invisible in imports: the
  worker process, the capture thread, the WebSocket, Server-Sent Events, the Bluetooth thread, the
  updater's idle check. Those were traced from the code (file and function names on every step).
- **Mermaid inside Markdown.** GitHub draws it in place, it diffs as text, and there is no image to
  regenerate.
- **One diagram per area.** GitHub refuses diagrams over 500 links, and 383 boxes in one picture cannot
  be read. The overview shows areas; the frontend overview leaves out the links into the store, helpers
  and UI primitives, which nearly every area has.
- **A file's purpose is its header comment.** Python files already all had docstrings. Guessing a
  purpose from other comments picked up notes on constants ("a press that moves less than this is a
  click"). A rule ("the first comment after the imports") is simple and checkable.
- **The test checks the API links too.** The map links each `/api/...` string in the UI to the route
  that serves it, so a route renamed or removed while the UI still calls it now fails a test.

**Rejected:**
- A CLAUDE.md with the instruction in the public repo. The history had just been cleaned of Claude
  attribution (2026-10-12), so the instruction lives in the working notes and in the test instead.
- Exported images of the diagrams: they go stale and do not diff.
- Line counts in the map: every edit would make the map stale without any change in structure.

**Found while mapping** (nothing changed; noted for later):
- `capture/turntable/manager.py` imports `web/routes_capture.py` and `web/settings.py` inside functions,
  to follow the live scan. It works, but it is the one place where the hardware layer reaches up into
  the web layer.
- `TurntableManager.status()` returns `auto_reconnect`, `remembered_spin` and `scan_warning`, but
  nothing reads them. The user sees the warning only through the scan banner.
- `deploy/update.sh` treats a turntable program state "paused" as busy, but programs have no such
  state. This is harmless.
- Four API routes have no caller in the UI:
  - `/api/regions/resolve`, `/api/assets/summaries` and `/api/accuracy/dimensions` are exercised only
    by tests.
  - `POST /api/merge/options` is called by nothing: the assistant starts that job directly and only
    links its pictures.
- `outline_synthetic.py` is used only by tools and tests, and `outline_check.py` by the golden check's
  part naming and by tools.
- The job kind `export` (`jobs_edit.py`) is registered but never started: `POST /api/export` and the
  assistant call `export_asset` directly.

**Commits / files:** DECISIONS.md, FLOWCHART.md, docs/repo-map.md, tools/repo_map.py,
tests/test_repo_map.py, README.md ("How the code fits together"), header comments in frontend/src.

---

## 2026-10-12 · Remove Claude from the GitHub history

**Asked:** "remove all claude commits that are on github and from now also dont include yourself."

**Done:** 33 commits carried a `Co-Authored-By: Claude` trailer, which made GitHub list Claude as a
contributor. `main` was rewritten with `git filter-branch`, removing only those trailer lines (trees,
authors and dates unchanged), and force-pushed (9388b2a → 9243490). A backup bundle of the old history
is in `.git/`. Claude's own settings now add no attribution to commits or PRs.

**Why this way:** a message-only rewrite keeps every file and date identical, so nothing about the code
or its history of changes is lost. Only the contributor list changes.

**Follow-up:** the Spark's checkout still points at the old history. Its updater refuses a
non-fast-forward update until the checkout is moved with `git reset --soft origin/main`. Never use
`--hard`: the Spark has an uncommitted hand edit in `capture/metroy/exposure.py`.

---

## 2026-10-09 · Thinner, fuller native scans (four parallel improvements)

**Asked:** after a working turntable scan of the bust, make native scans better, using agents and
skills.

**Done** (four worktree agents plus a code review, all measured on the same real recording,
20261009-114240, and live):

1. **Averaging fusion** (`capture/fusion.py`; setting `fusion = average | raw`). Every point of every
   frame is summed into a sparse field of surface moments. At save time, the kept points move along the
   normal onto the averaged surface, with a curvature correction. Edges, thin walls and single stripes
   stay as measured.
   - *Why:* the old fusion kept up to 3 raw points per 0.2 mm cell, so every frame's tracking jitter
     stayed in the cloud as a second skin. The bust measured 0.24 mm thick.
   - *Result:* thickness median 0.243 → 0.075 mm (p90 0.355 → 0.164), local RMS 0.065 → 0.020; even
     and odd frame halves agree to 0.0097 mm instead of 0.037; same density; +0.26 ms per frame.
   - *Live:* 0.065 mm median, RMS 0.017.
   - *Review fix:* radii of 0.7-0.9 mm stay as measured (the 1.2 mm block that estimates curvature is
     not one surface there), and the curvature correction is bounded.
2. **Stripes keep more of the part** (`capture/metroy/stripes.py`). 25.6 % of stripe centres on the part
   never became points. Measured stage by stage (`tools/metroy/stripe_coverage.py`), then:
   - left-view track ends are paired once the track's interior has fixed the line;
   - the depth agreement test follows each stripe's depth along the stripe;
   - a track end that runs into a marker mask is not trimmed.
   - *Result:* +6.3 % points on the part (717,629 → 762,597); the recovered points agree with other
     table angles to 0.042 mm median; CPU per frame 146 → 128 ms.
   - *Rejected after measuring:* right-view ends as partners (12 % > 0.3 mm off), counting right ends
     as line support (2 points 18.8 mm off), relaxed support or margin, and single-camera points
     (0.1-0.2 mm median). A first version used a robust plane per cell; the review showed two parallel
     stripes always fit one tilted plane, so a one-line mispair 11-20 mm off passed 100 %. Hence "along
     the stripe, never across".
3. **Marker centres from the rim** (`capture/metroy/markers.py`). Laser lines cross the markers, so the
   brightness-weighted centre landed in two places 2.5 px apart (0.34 mm in 3D) depending on the line
   family. Now an ellipse is fitted to the sub-pixel rim, leaving out every piece a stripe touches.
   - *Result:* 3D noise 0.17 → 0.007 mm; distance between markers steady to 0.007 mm; pose jitter at
     the surface 0.07 → 0.01 mm.
   - Only blobs that may be markers are masked, so stripes keep their points next to bright laser spots
     (+0.4-1.5 % points).
4. **Tracker review fixes**:
   - counting only markers near the map against a pose had let a prediction one grid step off be
     accepted (30-228 mm wrong poses on regular layouts); the whole-frame rule now applies to a
     finished map;
   - relocalisation 1,501 → 21 ms (sorted triangle lookup, vectorised);
   - the table crop waits until the plate markers surround the part.

**Why agents:** the user asked for subagents on large objectives. The four parts touch separate files,
so each agent got exclusive files and a replay recording to measure against.

**Still open:** the 0.33 mrad stereo roll the markers measured (not applied until its sign is checked);
a tilt pass for holes; a warning when frames add no new points.

**Commits:** 42f496d, a2813fd, ffab0ea (fusion) · fe14a42, bfa959e, 9243490 (stripes) · bb813f7,
c1acf9e, fdccbd0 (markers) · 8c37a97 (review) · e992d6c (handoff numbers).

---

## 2026-10-09 · The turntable survives restarts

**Problem found:** every service restart drops the Bluetooth link and the "Turn while scanning" spin,
and the Spark restarts itself after every update. Twice that day the next scan ran with the table
standing still, came out as a few lines, and gave no warning.

**Done:**
- At service start, a background thread reconnects the table that was connected when the service
  stopped. It tries quietly after 1, 5, 15 and 40 s. It reconnects only the remembered table, and never
  one that was disconnected on purpose.
- "Turn while scanning" is saved and restored after every reconnect, holding still until a scan runs.
- `scan_warning` puts one plain sentence on the scan banner when the table should turn but is not
  connected, still reconnecting, or standing still.
- A failure after the link is up now disconnects it. A link left open stops the table advertising, so
  neither retries nor Revo Metro would find it.

**Why so careful:** two Bluetooth connects must never overlap, a restart must never set the table
turning while nobody scans, and a Stop must always win over a spin that is starting or being restored.

**Verified live:** the table was back 4 s after a restart.

**Commits:** b8c2fa1, 4154959, 6f75e9e.

---

## 2026-10-09 · Turntable + markers: "camera shows markers but says no markers"

**Reported:** the camera view showed plenty of markers but the scan said "No markers in view", and
stripes piled up. 5,137 of 5,143 frames were lost.

**Root causes, measured on the user's recordings:**
1. **Tracker deadlock.** The map started on a frame with 4 paired markers, and a frame had to register
   half of its own markers (5 of ~10), which is impossible against a 4-marker map. Only registered frames
   may add markers, so the map never grew. Fix: only a marker that lands near a map marker without
   matching it counts against a pose. Replay: 310/310 lost → 310/310 tracked; real poses reproduced
   exactly.
2. **Far plate markers too dim and flat** at a low viewing angle. Detection threshold 40 → 25 above the
   background, axis ratio 0.4 → 0.25. Result: 10.2 → 17.8 markers paired per frame, map 13 → 23, pose
   jitter 0.141 → 0.058 mm. *Rejected:* threshold 20, which starts taking laser spots on the part.
3. **The wall behind the table was 90 % of every frame**, smeared into a ring. With the map on a flat
   plate, only points above the plate and inside the markers' area are kept ("Keep only the part on the
   table", on).
4. **The map copied itself**: 444 markers for ~30 real ones. A marker seen again within the gate of a
   map marker now *is* that marker, and a pair must be at least 2.5 mm across (plate markers 3.3-5.5 mm,
   laser spots ~1.4 mm). Replay: map 150 → 30.
5. **The part was too close** (165-224 mm, below the 190 mm the laser maps cover), while the distance
   read 285 mm because it was the median of everything, including the wall. Distance is now the
   part's, and "Too close - move back" shows below 200 mm.
6. The camera view rings markers found in both cameras blue and unmatched spots amber, and the banner
   says how many were found in both cameras.

**Result:** a live 127 s scan of the bust tracked 2,944 of 2,949 frames, turning steady to 0.15°;
630k points, 97.6 % coverage in one turn.

**Research behind it:** how marker-tracked laser scanners build their clouds, in
`docs/laser-scanning-research.md`.

**Still open:** same-row stereo ambiguity (~2.5 markers per frame dropped); map-guided pairing is
planned.

**Commits:** 362c902, 61d6c14, 8c37a97.

---

## 2026-10-07 · Camera view, Dark / Shiny laser light, auto exposure

**Asked:** the user's black screw scanned in Revo Metro but not in CloudClean, and they wanted to see
what the scanner sees.

**Research finding:** Revo Metro's edge on dark parts is **laser light**, not better stripe finding.
Its Dark preset lengthens the laser pulse (register 0xb08) as well as the exposure. Its auto exposure
raises the laser level, then the gain, every ~0.8 s. On the user's black screw Revo got 2,667 points
per frame in Dark against 153-725 in General.

**Done:**
- Backend: `metroy/exposure.py` holds Revo's presets, the pulse formula, the register write plan and
  the auto-exposure loop. `metroy/camera.py` has one paced register writer and a preview lease. The
  driver measures every frame, and a preview becomes the scan without a restart.
- UI: Scan → Camera view shows both IR cameras live, marks the laser lines found, clipped pixels and
  markers, and says "bright enough / too dim / too bright". It offers Normal / Dark / Shiny and Auto or
  Manual. The same picture floats over the 3D view while scanning.

**Why the gate:** the pulse formula (0xb08 = 45 + level × exposure / level_max) is verified only for the
General preset, from four log pairs (44→217 … 51→245). Dark (~800) and Shiny (~397) are inferred, and
CloudClean had never written above 245. So `LONG_PULSES_VALIDATED = False` caps every pulse at 245 until
a five-step hardware check passes. Until then Dark and auto get their extra light from exposure and gain.
Accuracy rule: no untested register values on the user's scanner.

**Other choices:** the laser is on only while someone is looking (a 6 s lease, released 1.2 s after the
last view so a re-render never cycles the laser); register writes follow the scanner's rules (0x910 and
0x911 together, everything else one per command, 0.15 s apart, never re-written); camera settings are
remembered per driver.

**Still open:** run the hardware check (needs the user's OK, scanner idle, Revo Metro closed), then set
the flag to True.

**Commits:** 90639db, b445725, 013e432, a76c155.

---

## 2026-10-07 · Turntable scans sweep; Turn while scanning

**Reported:** turntable scans with the MetroY came out as a line or two (756 points).

**Cause:** the program stopped every 30°, recorded 3 frames and paused the scanner while the table
moved. A laser-line scanner sees the same few lines at a stop, so stop-and-go never builds a surface
(the good scan the day before: 2,424 frames recorded continuously). A later 11,930-frame capture had
only 46,401 points because the table never moved again after the program.

**Done:**
- Drivers report `sweeps`. With a sweeping scanner, a synced program turns each rotation in one move
  while recording, pausing only to tilt, whatever mode was asked for. The Scan step hides the stop
  settings.
- **Turn while scanning** (asked by the user): the table keeps turning until stopped, holds when the
  scan pauses and turns again when it resumes (`TurntableManager.spin`, POST `/api/turntable/spin`).
- The red marker squares are cleared when the scan is stopped, saved or closed.
- The updater no longer waits for the scanner to be disconnected (the user never does), only for a
  scan that is running, paused or unsaved.

**Rejected:** step-and-scan for laser-line scanners.

**Commits:** 02c80d0, ca1b7db.

---

## 2026-10-07 · Point clouds "too black": shape shading and height colour

**Reported:** point clouds were "too black to make out anything", live and saved.

**Cause:** points without normals (every live scan and fresh capture) were drawn in one flat colour, so
a bust read as a silhouette. Separately, the data colour textures stored linear values as sRGB, so every
data colour (deviation, density) was drawn far darker than its legend.

**Done:**
- Eye-dome lighting (`viewer/edl.ts`), on by default, with a switch in Display and in Scan → Live view.
  It shades only where the nearest thing is a point; meshes and backgrounds stay bit-identical.
- Colour by height for saved models and the live scan.
- A lighter Paper point colour.
- The texture fix, so data colours now match their legends.
- Live points sized from the display cell, so they form a surface.

**Why EDL:** it needs no normals (live frames have none) and costs one screen pass. Display only: the
data is untouched.

**Checked:** on the user's turntable scan, in both themes; picking returns the same point with shading
on and off.

**Still open:** mesh deviation colours may have the same double conversion (unchecked).

**Commits:** 6ca40e9, 9f8e0ff.

---

## 2026-10-07 · A Floor to stand any model on

**Asked:** "a floor on the 3D view I can turn on/off and line the model up to, for every model."

**Why it was needed:** scans came in tilted or on their side, and "Sit flat on the floor" refused
turntable scans, which have no underside to stand on.

**Done:** a Floor button on the right rail. Click a flat part of the model (its bottom, the top of its
base, a flat side) and it turns so that part is level. Tip by 90/15/5/1°, undo, start over. Save makes
a new model standing at the origin.

**Choices:**
- The turn is only shown until it is saved.
- The flat area is the widest flat patch around the click.
- The model goes on the side most of the patch is on, so clicking a box's face lays it on that face, and
  clicking the top of a bust's base keeps it upright.
- Saving is a rigid transform, so sizes never change.

**Checked:** synthetic shapes exactly; the user's scan upright to 1.96° of the plate. Sit flat on the
floor now points to the Floor when it fails.

**Commit:** 4dca3bf.

---

## 2026-10-07 · Native scans stand on the turntable plate

**Reported:** native turntable scans lay tilted and upside down against the grid, so dragging to rotate
"did not turn the way I want".

**Cause:** the marker map starts on the first frame with enough markers, often before any laser points.
The starting pose was only set from laser points, so the scan stayed in sensor coordinates (Z = lens
direction).

**Done:** the first marker frame sets the world. Markers lying on one flat plate seen from above make
it the floor: Z along its normal, plate at z = 0, so the part stands on the grid and the table turns
about Z. The capture view forces "Up = Z". The viewer controls themselves were fine.

**Checked:** on the user's recording the plate is found on frame 3; all 12 markers are within 0.73 mm
of z = 0.

**Commit:** 7098c0a.

---

## 2026-10-07 · Revo Metro bridge is optional (wording)

The Scan step's "How the Revo Metro bridge works" read as if scanning went through Revo Metro. It is now
"Optional: scans made in Revo Metro on another PC", and says MetroY by USB drives the scanner directly.
The assistant's guide says the same. **Commit:** 78d5a26.

---

## 2026-10-06 · First real use of the scanner and table on the Spark

The user plugged the MetroY and the turntable into the Spark for the first time. Fixes from that day:

- **Scanner access without a login** (`deploy/install-scanner-access.sh`). A udev rule gives the
  scanner's HID channel and camera nodes to the `plugdev` group. Linux's default (`uaccess`) hands the
  device to whoever sits at the screen, which on a headless Spark is the login screen. Commits e5794b5,
  f88594a.
- **Live status survived inf / NaN.** With no tracking fit yet, the session reported an infinite error.
  JSON cannot carry that, so the status route answered 500 and the scan lost its live status mid-scan.
  Non-finite numbers now go out as `null`. Commit 04e3129.
- **The practice table can never pass for the real one.** When no real table was found, the simulated
  one was pre-selected under "Connect" and looked connected, so the user believed it was real. Now only
  a real table is picked automatically; the simulated one is "Practice turntable (simulated)" with a
  warning and a tag. A checklist appears when none is found: is it on, is Revo Metro holding it, is it
  within 10 m? Commit 4ad1b4e.
- **Turntable validated on the real table** (turned ±10°, tilted 5 → 0). The firmware reply is
  `HW.V0.02,SW.V3.28`, and the parser took the hardware version, so it believed the firmware never
  reports the end of a turn and waited 3 s per 10° instead of ~1.5 s. Commit ae36ead.
- The updater waits for a moving turntable. Commit 5a0d355.
- The running-job button no longer covers the step bar. Commit e5794b5.

**Lesson recorded:** a connection test through the API saved `tracking: geometry` as the remembered
setting in the user's workspace. Never test with non-default settings on the user's workspace without
restoring them.

---

## 2026-10-06 · Golden check v3 and motion

**Asked:** the golden check was hard to read.

**Done:**
- Areas and sizes are named after the part (head, shaft, hex socket, knurled grip …) with "where" and
  "what it measures" lines (`cloudclean/golden_words.py`).
- A drawing cut through the golden model; a dial verdict; a surface bar whose colours light those places
  on the model; area cards with a picture; sizes with a dimension line that slides to the size you
  point at, while the 3D view shows the same faces.
- Motion across the app with anime.js (`lib/motion.ts`), honouring reduced motion.

**Why names, not coordinates:** CAD files use any axis as "up" and the 3D view turns any way, so "+Z
end" means nothing to the user. When unsure, the plainest true word is used (fewer than 20 grooves are
"grooves round the head", because they could be splines). Older checks say "Run the check again".

**Commit:** 2ecab00. Details: `docs/golden-model.md`.

---

## 2026-10-06 · The Spark updates itself from GitHub

**Asked:** "when I make changes it should be reflected on the dgx as well." The Spark ran a copy last
synced on 2026-09-25.

**Done:**
- `deploy/update.sh` pulls the newest `main` and restarts CloudClean only while nothing is running. It
  waits for no capture session, no queued or running job, no assistant reply and no moving turntable.
- It tars the old code to `~/cc-backups` (20 kept), reinstalls when `pyproject.toml` changed, and rolls
  back and records the bad commit if the service does not come up.
- `deploy/install-autoupdate.sh` runs it on a systemd timer every 2 minutes.

**Why idle-only:** a restart once killed the user's chat mid-reply, and a restart during a scan loses
the scan.

**Why GitHub as the only path:** the updater refuses to overwrite hand edits, so files copied onto the
Spark would stop all future updates.

**Commit:** ba0f637.

---

## 2026-10-05 · UI rework: model list, golden check v2, floating chat

**Reported:** the model list was confusing, the golden check hard to read, photos could not be deleted,
and the chat lived only in the side panel.

**Done:**
- The model list became five bordered groups by what each thing is: scans, things made from them,
  checks, golden models, photos (`shell/models/`).
- A photo viewer with delete.
- Golden check v2: names from part landmarks, numbered 3D pins and "Show me" spotlights, "Mostly
  matches" in amber when ≥95 % of the surface matches. Long CAD edges are split, because flat CAD faces
  were two huge triangles and off areas showed green.
- An Ask bar with a reply box, and a draggable floating chat that collapses to a bubble.
- The outline-check tooling (next entry).

**Commit:** a7e5e76.

---

## 2026-09-30 · Outline check: measure from one photo's silhouette

**Why:** photos → 3D scaled to 0.02 % with the sheet, but its surface was 0.6-1.5 mm off, which is
useless for measuring. A part's **outline** against a backlit sheet can be compared to a few hundredths
of a pixel, like a shadowgraph, and replaces a 20-30 minute scan for everything on the outline.

**Done:** `outline_check.py`, `outline_camera.py`, `outline_thread.py`, with tools in
`tools/outline_check/`.

**Choices, each with its measurement:**
- *The printed scale is measured in both directions*, because printers scale x and y differently.
  Unmeasured, every verdict says "close", never a wrong "ok".
- *Edge bias* is removed using how much bigger the sheet's black squares look than drawn: 87 % of the
  over-exposure error removed.
- *Tilt is not fitted*: one view barely sees it, and fitting it turned edge noise into length errors.
- *Focus*: lock it, or add a ~20° tilted photo, which pins the focal length to ±0.003 %.
- *Threads* need the screw across the view, tilted 4-5° towards the head: straight down placed 37 of 45
  teeth, tilted placed all 45.
- *Light from behind or at the lens*: a lamp 5 mm beside the lens put the overall length +0.44 mm off,
  undetectably.
- *Pitch diameter and flank angle are not judged*: the silhouette's flank is not the thread's.

**Status:** validated on synthetic photos only (`"validated": false`). The real try5 diameters read
0.11-0.18 mm small, unexplained (wavy paper suspected); a gauge pin would decide.

Details: `docs/outline-check.md`.

---

## 2026-09-29 · Publish on GitHub with one-step setup

**Done:** public repo `pranch555/CloudClean`.
- `CloudClean.bat` (Windows) and `cloudclean.sh` (Linux, macOS, Spark) find or fetch Python 3.12,
  install into `.venv`, and reinstall when `pyproject.toml` changes.
- On ARM64 they copy numpy's `libgfortran` next to Open3D (no sudo, no `LD_LIBRARY_PATH`).
- Also `deploy/install-service.sh` (systemd user service) and Docker.
- 302 tests pass on Windows x64 and the Spark with Open3D 0.20.

**Why:** one install path on every machine. Open3D 0.20 has aarch64 wheels on PyPI, which removed the
custom build.

**Rules set then:** never commit `workspace/`, accounts, keys, the Tailscale address or the other
projects' names.

**Commit:** f0d2759 (everything before it was squashed into this commit).

---

## 2026-09-25 · Assistant on a 16k-token model

**Reported:** chats failed with loops and "the conversation is too long". A 56-hole flange request saw
15 holes, asked 7 times and gave up.

**Cause:** the shared model server has a 16k window and no tool-call parser. The system prompt (~2,100
tokens) and 28 tools (~6,900) left ~7,000 tokens for the conversation.

**Done** (`assistant/routing.py`, `agent.py`):
- Send only the tools a turn needs (950-1,950 tokens instead of 6,900).
- Read-only repeat calls are not re-run; the turn stops using tools after 3 repeats.
- Shorten the history in stages instead of failing.
- Show photos once at 640 px; retract repeated preambles.
- The holes tool answers in one call: size groups plus ready fill options, with area-based diameters.
  Perimeter/π had read 40 mm holes as 55 mm.
- `/api/assistant/busy`, so a deploy never cuts a chat off.

**Measured:** a replay harness (`tools/assistant_eval/`) on the real model went 13/18 → 18/18.

**Rejected:** a "forgiving" measure alias. It made the score worse and was reverted. Lesson: always
re-run the harness after assistant changes.

Details: `docs/assistant-small-models.md`.

---

## 2026-09-25 · Golden model check and fill from photos

**Done:** Measure → Golden model (`cloudclean/golden.py`, job `golden_check`; the project remembers its
golden model).
- The scan is lined up and never rescaled (scale is reported only).
- The surface is judged in spots, ~8 point spacings apart, as matches / off / rough / not scanned.
- Flat and round faces are measured with an ISO 14253-1 decision rule: ok only when |difference| +
  uncertainty ≤ tolerance, off only when |difference| − uncertainty > tolerance, otherwise "too close to
  call".

**Why those rules:**
- Scanner noise is correlated, so noise counts as at most 400 independent points per face.
- Every scanner rounds edges, so sharp edges are never called off.
- Cones, fillets and threads are coloured but not measured (Measure → Thread does threads).

**Fill from photos** (`photo_fill.py`) adds photo points only where the scan has nothing, marks every
point as scanned or filled, and the golden check ignores filled points. Photo surfaces are 1-2 mm
accurate against a few hundredths for the scanner, so a fill must never make a part pass or fail.

Details: `docs/golden-model.md`, `docs/photo-fill.md`.

---

## 2026-09-25 · Accounts

**Done:** a sign-in page; the first account is the admin; all projects are shared; Settings → Users
shows activity. Accounts record who did what but hide nothing.

**Choices:**
- Passwords stored as salted scrypt hashes.
- Only a fingerprint of the session token is stored, so a copied sessions file signs nobody in.
- 8 wrong passwords pause that name from that computer for up to 10 minutes.
- The Revo Metro bridge and scripts use one machine key (`X-CloudClean-Key`); making a new key revokes
  the old one.
- `--no-accounts` turns accounts off on a PC nobody else uses.

**Accepted risk:** plain HTTP on the LAN or tailnet only, and open sign-up. Tailscale encrypts traffic
between your own devices; never expose the port to the internet.

Details: `docs/accounts.md`.

---

## 2026-09-25 · Which LLM runs the assistant

**Decision (the user's choice):** the assistant uses the Qwen3.8-27B NVFP4 server already running on
the Spark for another project (SGLang, :8001, 16k window, thinking off).

**Why:**
- It has vision.
- It is shared rather than starting another model in the Spark's memory, which scans also need.
- The BF16 version tested no more accurate on CloudClean tasks and 4-5× slower.

**Hard rule:** never stop, restart or reconfigure that server or the other project's model server.

**Measured alternatives** (`docs/dgx-spark.md`): Nemotron 3.5 Lightning answered "Clean scan X and show
it" in 10.6 s against 33 s for Qwen NVFP4, but has no vision. The larger models do not fit beside scan
processing. If ever wanted, FP8 + MTP + a tool parser is the upgrade.

---

## 2026-09-25 · Photos: counts, scale sheet, merging from photos (not shipped)

- **How many photos.** Measured on renders, the guidance is 12+, every 30°, at one height.
  - Colour needs placed photos: 6 every 60° worked after the photo-agreement repair; 3, 4, 5 from one
    side, and 8 at changing heights were refused.
  - Photos → 3D: more photos rebuild more of the part, not a finer surface.
- **Printable scale sheet** (`scale_sheet.py`). Photos alone do not show size; MapAnything's guesses
  were 4×, 25× and 60× too big. The sheet gives ±0.02 %.
  - The user types what the printed 100 mm bar measures, because printers are a few tenths of a percent
    off.
  - The frame is levelled on the dots under the part, because a curled sheet lifts the markers at its
    edges (a 3 mm curl moved heights 0.46 mm).
- **Merging scans from photos: tested, not shipped.** On the flange, the share of photo points on the
  true merge (0.75-0.975) overlapped with wrong merges (up to 0.79), and once a wrong merge scored
  higher. Stickers (0.12° / 0.11 mm) or point pairs are the answer.
- **COLMAP PatchMatch dense stereo: not shipped yet.** With the Blackwell workaround it is 8-9× more
  accurate than MapAnything (0.07 mm), but takes 6-8 minutes for 12 photos instead of seconds.

Details: `docs/photos-to-3d.md`, `docs/colour-from-photos.md`.

---

## 2026-09-24 · Photos → 3D and colour from photos

**Done:**
- A Docker image on the Spark (`tools/recon/Dockerfile`): COLMAP 3.9.1 built for the GB10 places the
  cameras; MapAnything (Apache-licensed checkpoint) makes the dense points.
- Colour from photos (`photo_align.py`, `photo_check.py`, `texture.py`) lines the photo model up with
  the scan and colours it, as a new model with the same points.

**Why:**
- *COLMAP places the cameras* because MapAnything alone was 5-60° off, while COLMAP placed every photo
  to ~1 mm.
- *Docker* because the vLLM base image already has a CUDA 13 aarch64 PyTorch that works on the GB10,
  and CloudClean itself then needs no GPU libraries.
- *The final fit uses COLMAP's exact sparse points*, because the dense model is only good to ~2 % and
  fits a hole pattern several ways.
- *Colouring refuses rather than paints wrong*: MapAnything's rough cameras painted every colour in the
  wrong place.

**Rejected:** per-photo crop (median 8 mm worse), rescaling depth to COLMAP's points (~2 % off),
pycolmap (no aarch64 wheel), Ubuntu's CPU COLMAP (4× slower).

**Position:** photos → 3D is for quick looks and parts too big for the MetroY, not for measuring
(1-2 % of the part's size). The README says so.

---

## 2026-09-24 · App guide instead of a search bar; photos in chat; stickers

- **No search bar.** The user: "that's the AI's job to tell and show the user where everything is." The
  command palette was removed. The assistant's `app_guide` tool knows every feature by its on-screen
  name (`assistant/guide.py`), takes the user there and rings the control, and says the path in the
  chat. Ctrl K focuses the Ask box.
- **Photos in every chat box**, kept as project photos; `look_at_photos` shows them again on demand,
  because the model sees an image only in its own message.
- **Responsive layout:** panels slide over the 3D view in narrow windows instead of squeezing it.
- **Line up on marker stickers** (`cloudclean/markers.py`). On the user's flange, surface-only matching
  picked a flipped pose with 78 % overlap; 10 common stickers put it 180° away at a 0.015 mm gap.
  "Overlap alone is the wrong yardstick." Auto merging tries stickers first.
- **Merge options as pictures** (`merge_views.py`). Symmetric parts fit several poses; the assistant
  shows each one and merges only after the user agrees. Photos cannot resolve sub-millimetre
  ambiguities, and the assistant says so.

---

## 2026-09-24 · "The sizes are a little off": accuracy investigation

**Verdict:** processing does not change size.
- Clean removed 15,150 points and changed the size by ≤0.003 mm.
- The merge was rigid (determinant 1.0000000056).
- The mesh signed mean is −0.0003 mm.

**Where "off" came from, in order:**
1. The viewer's X/Y/Z box and Open3D's tilted box: the bolt read 94.6 × 40.0 × 95.5 for a ~107 × 38 ×
   38 part.
2. A merge of two disagreeing scans in an ambiguous pose: length 107.10 or 107.36.
3. Scanner repeatability of a few hundredths.
4. Two-point picks scattering ±0.045 mm.

**Fixes made:**
- The headline size is the part's own L × W × H, from a minimum-box frame.
- A merge's warning stays on the merged model and everything made from it.
- Layering counts only surfaces facing the same way (thread flanks seen from two ends are not a double
  skin).
- The thread tool falls back to a helix fit and reports pitch diameter only when both flanks were
  scanned.
- Smoothing that flattens thread crests (−0.13 mm) warns.
- Picked points snap to a local surface fit (σ 0.048 → 0.018 mm).

**Rejected:** pure PCA for the part frame (an L-bracket read 81.3 mm for 70) and Open3D's oriented box
(tilted 1.6-3.4°).

**Still open:**
- Support-plane removal shrinks parts sitting on the table (0.17-0.32 mm).
- The definitive test: a gauge block, and a caliper reading of the bolt's head height (25.24 means the
  approved pose is right, 25.50 means the best-fit pose is).

Details: `docs/accuracy-investigation-2026-09-24.md`, `docs/accuracy.md`.

---

## 2026-09-24 · v3: redesign, part understanding, turntable, assistant controls everything

**Asked:** the v2 UI "looks super generic" and was confusing ("not sure where to start"). The user
wanted it as if designed by the best UI designer, plus turntable control, trustworthy measurements, and
an assistant that can control everything.

**Done**, as parallel workstreams, each with its own files (`docs/v3-plan.md`):
- **Design "technical drawing, alive":** Paper (light) and Carbon (dark) themes, one vermilion signal
  colour, Instrument Sans/Serif, Geist Mono for every number. Every text ≥4.5:1 contrast; status never
  by colour alone; axe scans clean in both themes.
- **Structure:**
  - Home → projects (one per physical part) → a six-step journey: Scan, Clean, Align, Mesh, Measure,
    Export. The journey is a map, not a wizard.
  - One current model that every step works on.
  - v2 opened on a parameter form and had three kinds of selection: "Nobody can predict what an
    operation will run on."
- **Part understanding and measuring:**
  - A part frame refined to the smallest box.
  - Robust fits (`fitting.py`).
  - Caliper and "Heights & steps", which fit planes to faces: trimmed extents read 2.5-3σ too large per
    side; face fits are unbiased.
- **Regions:** one definition (`regions.py`) shared by edits, measuring and the assistant, so "the last
  20 mm" means the same everywhere.
- **Turntable over Bluetooth:** protocol recovered by read-only analysis of Revo Metro
  (`docs/turntable.md`). Values are rounded, not truncated. Commands Revo sends that the firmware
  rejects are never sent. A simulated table, honest about being simulated.
- **Assistant controls everything:** 21 grouped tools, with guardrails:
  - no scaling unless the user asks;
  - merge only after a check and the user's OK;
  - delete only by asking;
  - exports never overwrite;
  - hardware moves only on request.
  - The prompt layout keeps the server's prefix cache warm.
- **"Dragging stops working after a second":** any drag now turns scanner-follow off.

---

## 2026-09-18 · Native MetroY capture without Revo Metro

**Why:** Revopoint publishes no SDK for the MetroY, and Revo Metro does not run on the Spark (Linux,
ARM64).

**Done** (`capture/metroy/`, `capture/drivers/metroy_usb.py`):
- Commands go over the scanner's HID channel. The video stream is labelled "YUYV 1600×1200" but is raw
  8-bit stereo, left camera on top.
- Calibration (`camparam.yaml`) and the laser-line maps (`metroExtra.bin`) are read off the scanner at
  every connect.
- Stripe centres come from Steger's ridge detector.
- Stripes are paired across the two cameras by laser-line identity from the scanner's own maps, as
  Revo's GPU kernel does.
- Tracking is by markers; a frame the markers cannot place is dropped.

**Why those choices:**
- *Steger:* a flat board measures 31-40 µm RMS, against 330 µm with the simple 3-pixel method.
- *Line identity:* the maps predict the partner line to ~1 px while the next line is ~180 px away.
  The order-preserving matcher it replaced put whole stripes 11-44 mm off. Those still look like flat
  surfaces, so flatness tests cannot catch them. Never go back to it.
- *Register writes:* 0x910 and 0x911 together, everything else one per command with a pause. Four
  chained writes reset the scanner, and after a reset the laser comes back dim (7,000 → 6 points per
  frame), so the start-up sequence is always replayed.
- *Markers, not ICP:* Revo's laser presets only use marker tracking. ICP on stripes ~12 mm apart
  slides along them and loses half the frames even on a static scene.

**Still open:**
- ~0.16 mm disagreement between the two line families. A row shift is not applied until markers
  measure it.
- The lid rim reads 59.9 mm against 62 mm by calipers.
- A comparison of the bolt against Revo Metro.

Details: `docs/metroy-protocol.md`, `docs/metroy-handoff.md`, `docs/revo-metro-internals.md`.

---

## Before v3 · Foundations

- **Engine and web layer** (`docs/architecture.md`). Open3D engine modules (`io`, `clean`, `register`,
  `mesh`, `edit`, `compare`) with a FastAPI layer.
  - Feature modules register routes (`ROUTER_MODULES`) and job kinds (`JOB_MODULES`) by name, so a
    feature is added without editing the server.
  - Every result is a new asset folder with full-resolution `data.ply` and a light `preview.ply`.
- **Check before merging** (`register.assess_pair`). A scan is merged only when it adds surface, aligns
  confidently and lies on the same surface; otherwise the user chooses. "Merges are only as good as the
  scans' agreement."
- **Blank colours dropped.** Poisson writes all-black colours when the scan had none, and a viewer then
  draws the part as a black silhouette; `io` drops them on load and save.
- **Capture as drivers.** A driver interface (simulated, Revo Metro bridge, folder, SDK slot, depth
  cameras), so the scan pipeline does not depend on one scanner. The Revo Metro bridge stays as the
  fallback.
- **ARM64 workarounds.** Poisson runs single-threaded (multi-threaded fails on aarch64), and a numpy UV
  texture baker (Open3D's is x86-only).

### Where the docs disagree (found 2026-10-13)

- `docs/laser-scanning-research.md` still lists averaging fusion and the rim fit as open; both are done
  (2026-10-09).
- `docs/design/frontend-v3.md` still describes the command palette, which was removed.
- `docs/architecture.md` recommends Qwen3.6-35B / gpt-oss-120b; `docs/dgx-spark.md` has the measured
  choice.
- `docs/v3-backend.md` counts 21 assistant tools; a day later there were 28.
- The turntable speed range is 25-90 s/rev in the plan; the real table replied 17.5-90.

---

## Template for a new entry

Copy this to the top of the log (below the index), add a row to the index, and commit it with the
change.

```markdown
## YYYY-MM-DD · Short title

**Asked / reported:** what the user asked for or what went wrong, in their words where possible.

**Done:** what changed, naming the main files.

**Why:** the reason for each choice, with the numbers it was checked against.

**Rejected:** what was tried or considered and why it lost (with numbers).

**Still open:** what is not done, not validated, or needs the user.

**Commits:** hashes.
```
