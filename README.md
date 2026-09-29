# CloudClean

Clean, merge, mesh, colour and **inspect** point clouds from 3D scanners such as the **Revopoint MetroY Ultra**.
Use it from the browser workstation (3D viewer, live capture, assistant) or entirely from the terminal. Both run
the same engine. Built to run on an **NVIDIA DGX Spark** — see [docs/dgx-spark.md](docs/dgx-spark.md).

- **Scan**: live capture from the MetroY (native USB driver on the Spark, or the Revo Metro bridge) with coverage guidance, and the **dual-axis turntable** over Bluetooth: turn, tilt, speed, stop and stop-and-go programs synced to the scanner.
- **Colour from photos**: photos of the part you scanned (12 or more, about every 30° all the way round) are placed around it automatically (COLMAP on the Spark's GPU), lined up with your scan or mesh and projected onto it: real colours per point, or colours plus a texture image on a mesh. The shape and size of your model are not touched. Mesh → Colour from photos; see [docs/colour-from-photos.md](docs/colour-from-photos.md).
- **Photos → 3D**: no scanner at hand? 12 or more photos of the part, about every 30° all the way round, become a coloured 3D model in about a minute (COLMAP places the photos, MapAnything builds the surface). Photos do not show the true size, so **Set the true size** (Clean) scales it from one length you know. For the shape and a quick look (surface within about 1–2 % of the part's size), not for measuring: see [docs/photos-to-3d.md](docs/photos-to-3d.md).
- **Clean**: removes scanner noise, stray and sparse points, floating debris and (optionally) the table or turntable surface.
- **Align & merge**: aligns 2 or more scans of the same item (point clouds *or* meshes), even if the item was flipped between scans. It **checks first**: a scan is merged only when it adds new surface, aligns confidently and lies on the same surface as the others. Otherwise you choose: merge anyway, use the best scan, or keep them separate. When you merge anyway, the warning stays with the merged model and everything made from it, so measurements say so. **Line up on stickers**: marker stickers on the part leave small round holes in each scan, and three or more shared by two scans fix the one right way they fit, even when the part looks alike from several sides. **Which way do the scans fit?**: the assistant draws every way two scans can line up and compares them with your photos of the part.
- **Edit**: box / lasso select and delete, crop, cut with a plane, sit on the floor, align axes, mirror, repair, undo / redo. **Smooth brush**: paint rough areas and smooth them; edits blend into the untouched surface and the report shows how far the surface moved. Every edit runs on the full-resolution data and creates a new model.
- **Mesh**: builds a surface that keeps the true size of the item and reports how far the mesh deviates from the scan points. **Holes**: lists every hole with its size, fills the ones you pick.
- **Measure**: the part's size along its own axes, and nine tools: Point to point, Caliper (between two faces), Overall size (end to end), **Heights & steps** (every flat face along the part: head height, length under the head, recess depth, like a height gauge), Diameter, Angle, Flatness, Ball / sphere and Cross-section. **Thread analysis** finds pitch, major / minor / pitch diameter, flank angle and handedness, and matches ISO metric / UNC / UNF / BSP with ISO 6g/6H checks. **Compare to CAD** (STEP, IGES, STL, OBJ) maps the deviation of every point with tolerance pass/fail. **Accuracy checks** tell the scanner apart from the software: whether a processing step changed the size, how two scans of the same part agree, and a known-size check against a gauge block or ball bar.
- **Export**: PLY, STL, OBJ, GLB, **3MF**, plus a printable measurement report and CSV.
- **Projects**: one project per physical part, with every scan, merge, mesh and result in it.
- **Assistant**: chat with the local LLM on the Spark. It knows every model and its dimensions, can measure any dimension and draw it on the model, highlight regions ("the head", "the last 20 mm"), edit, merge, mesh, export, drive the turntable and the scanner, and control the viewer and the app. Add photos of the part from your computer (Photos button, paste or drop) in any chat box; they are kept in the project as reference photos, and the assistant can look at them again whenever it needs to compare the scan with the real part. When scans could line up in more than one way, it draws every option and picks the one that matches your photos (Align → Which way do the scans fit?).

Everything stays in the scanner's units (Revo Scan exports millimetres). Filter distances are set as
multiples of the scan's measured point spacing, so the same settings work at any resolution.

## Quick start

One step on every system. The first start sets everything up (a few minutes; about 800 MB of disk); later starts
take seconds. Then CloudClean opens in your browser at **http://localhost:8765**.

**Windows 10 / 11**: download the repo (*Code → Download ZIP* and unzip, or `git clone`), then double-click
**`CloudClean.bat`**.

**Linux, macOS (Apple silicon) and DGX Spark**:

```bash
git clone https://github.com/pranch555/CloudClean.git
cd CloudClean
./cloudclean.sh                     # add --host 0.0.0.0 to open it from other machines: http://<this-machine>:8765
```

On a server, `deploy/install-service.sh` runs it as a service that starts at boot (DGX Spark:
[docs/dgx-spark.md](docs/dgx-spark.md)).

**Docker** (Linux x86-64 or ARM64): `docker compose up -d`, then http://localhost:8765. Projects stay in `./workspace`.
Live USB scanning, Bluetooth turntables and Photos → 3D need the native install above.

What the launchers do for you: use Python 3.10–3.12 if you have it, or fetch Python 3.12 with
[uv](https://docs.astral.sh/uv/) (no admin rights); make a private `.venv` in this folder; install everything; and on
Linux ARM64 give Open3D the `libgfortran` it needs. They set up again by themselves when an update changes the
requirements. The browser UI is prebuilt, so Node is not needed.

### First steps

1. **Create your account.** The first account is the admin; add others in Settings → Users ([docs/accounts.md](docs/accounts.md)).
2. **New project**, then drop your scans (PLY, OBJ, STL, …) onto the window, or **Scan → Start the simulator** to try
   live scanning without any hardware.
3. Follow the steps along the top: **Scan → Clean → Align → Mesh → Measure → Export**.
4. **Assistant (optional).** The chat needs an OpenAI-compatible LLM server. Open **Settings → Assistant** and enter its
   address, for example `http://localhost:11434/v1` (Ollama), `http://localhost:1234/v1` (LM Studio) or
   `http://localhost:8000/v1` (vLLM / SGLang), plus the model name, and a key if the server needs one; or set
   `CLOUDCLEAN_LLM_BASE_URL`, `CLOUDCLEAN_LLM_MODEL` and `CLOUDCLEAN_LLM_API_KEY`. Pick a model with tool calling;
   the models tested are in [docs/dgx-spark.md](docs/dgx-spark.md#2-llm-server). Everything else works without it.

### Hardware

- **Revopoint MetroY / MetroY Ultra**: plug it into a Linux machine for native capture ([docs/metroy-protocol.md](docs/metroy-protocol.md)),
  or keep Revo Metro on a PC and let `cloudclean bridge` upload each export ([docs/dgx-spark.md](docs/dgx-spark.md#4-scanner--spark-with-no-manual-upload)).
- **Revopoint dual-axis turntable** over Bluetooth ([docs/turntable.md](docs/turntable.md)).
- **Photos → 3D** and COLMAP photo placement need an NVIDIA GPU and a one-time Docker build ([docs/photos-to-3d.md](docs/photos-to-3d.md)).

### Updating

```bash
git pull          # then start CloudClean again; it installs anything new by itself
```

### Using the terminal

`cloudclean` lives in the `.venv`: activate it (`source .venv/bin/activate`, or `.\.venv\Scripts\Activate.ps1` on
Windows) or call `.venv/bin/cloudclean` / `.\.venv\Scripts\cloudclean.exe` directly. See [Terminal](#terminal).

## Exporting from Revo Scan

Export each scan as **PLY point cloud** (keeps colour and normals). If you merged or meshed in Revo Scan,
PLY/OBJ/STL meshes also work. Scan the item once upright and again flipped, with good overlap around the sides.

Supported inputs: `.ply .pcd .xyz .xyzn .xyzrgb .asc .pts .txt .csv` (point clouds), `.ply .obj .stl .off .glb .gltf` (meshes), `.jpg .png .tif .bmp .webp .heic` (photos; iPhone HEIC photos are stored as upright JPEGs).

## Web app

```powershell
cloudclean serve            # opens http://localhost:8765   (on the Spark: --host 0.0.0.0)
```

**Home** lists your projects. Open one and the workspace walks you through six steps (top bar):
**Scan → Clean → Align → Mesh → Measure → Export**. Any step can be opened at any time; each one shows what it
works on, what it does, and one obvious main button. The left panel lists the project's models with their lineage
(scan → clean → merge → mesh), the 3D view is in the middle, the step (or the assistant) is on the right.
A short tour runs on first use. To find anything, ask CloudClean ("where do I measure the head height?"): it answers
with the path and takes you there, ringing the control. **Ctrl K** puts the cursor in the Ask box; Settings has a light (Paper) and dark
(Carbon) theme and larger text. The layout follows the window: in a narrow window the model list (then the step panel)
slides over the 3D view instead of squeezing it, and menus and tooltips always open where they fit.

Viewport:

| Action | Mouse / key |
|---|---|
| Rotate freely (no angle limits; pivots on the point under the cursor) | left-drag |
| Pan | right-drag, Shift+left-drag, middle-drag |
| Zoom towards the cursor | scroll / pinch |
| Re-centre on a surface point, or fit | double-click |
| Standard views | view cube, `1` front, `3` right, `7` top, `Ctrl` + number for the opposite, `0` iso, `5` perspective/ortho, `F` fit |
| Box / lasso select, delete selection | `B` / `L`, `Shift` adds, `Del` deletes (not in Measure), `Esc` clears |
| Measure distance between two points (snaps to full-resolution data) | `M` |
| Smooth brush | `S` |
| Undo / redo the last model change | `Ctrl Z` / `Ctrl Shift Z` |
| Ask CloudClean (type a request or "where is …") | `Ctrl K` |

While scanning, the view follows the scanner until you move it yourself; the "Follow scanner" chip turns it back on.

All results live in `workspace/assets/<id>/` (`data.ply` is full resolution).

## Terminal

```powershell
# everything at once: clean each scan, merge, mesh, export PLY/STL/OBJ/GLB + report.json
cloudclean run scan1.ply scan2.ply scan3.ply -o output --remove-plane

# with colour from a photo (camera JSON saved from the web app's Colour tab: "json" link on a saved view)
cloudclean run scan1.ply scan2.ply -o output --view front.jpg front_camera.json --texture-size 4096

# individual steps
cloudclean info scan1.ply
cloudclean clean scan1.ply -o scan1_clean.ply --preset aggressive --remove-plane
cloudclean merge scan1_clean.ply scan2_clean.ply -o merged.ply            # writes merged.merge.json too
cloudclean merge a.ply b.ply -o merged.ply --pairs pairs.json             # manual point pairs
cloudclean mesh merged.ply -o part.ply --also stl glb --watertight --smooth 5
cloudclean texture part.ply --view front.jpg front.json --view back.jpg back.json -o part_colored.glb

# hands-free: watch a folder, every new part -> <out>/<part>/<part>.stl + .ply + report.json
cloudclean watch D:\Scans\incoming -o D:\Scans\finished --formats stl,ply

# scans are checked before merging: --merge auto (default) | always | never
cloudclean run scan1.ply scan2.ply -o output --merge auto

# stress benchmark with timings, memory and accuracy
python tools/benchmark.py --points 3000000 --depth 11

# every tunable parameter
cloudclean params
cloudclean clean scan.ply --clean-set sor_std_ratio=1.5 --clean-set cluster_keep_ratio=0.3
cloudclean run a.ply b.ply -o out --config settings.json    # {"clean": {...}, "merge": {...}, "mesh": {...}, "texture": {...}}
```

`pairs.json` format (scan index → points on that scan and the matching points on scan 0):

```json
{"1": {"source": [[x, y, z], [x, y, z], [x, y, z]], "target": [[x, y, z], [x, y, z], [x, y, z]]}}
```

## Accuracy notes

- **Size is never rescaled** (except when you ask: *Set the true size* for models made from photos). Cleaning only removes points; a merge moves scans rigidly (checked: no scaling);
  Poisson meshing follows the cloud to a few micrometres on average and trims surface that isn't backed by scan
  data. Every model made by an operation shows how much the operation moved the surface and changed the size.
- **Which size is which.** The big numbers are the part's size along its own axes, end to end with stray points
  ignored; scanner noise adds a little at each end. For face-to-face sizes (head height, length under a head,
  across flats) use **Heights & steps** or **Caliper**, which fit planes to the faces. The "XYZ" box is along the scanner's
  axes and is not a part size.
- **Merges are only as good as the scans' agreement.** Two scans of a symmetric part (a 12-sided head, a thread)
  can fit in more than one pose; the check warns, and the warning stays on the merge. Each scan of a bolt sees
  only one thread flank and one end, so pitch diameter, overall length and head height exist only in the merge.
- **Scanner or software?** See [docs/accuracy-investigation-2026-09-24.md](docs/accuracy-investigation-2026-09-24.md).
  To settle the scanner's absolute scale, scan a gauge block or ball bar twice and use Measure → Accuracy.
- Leave *voxel size* at 0 to keep full scanner resolution. Merging keeps every point unless you set `dedupe_voxel`.
- Cloud *denoise* and mesh *Laplacian* smoothing flatten thread crests (by up to 0.13 / 0.04 mm on a 1" thread);
  prefer the default smoothing for measured parts.
- **Models made from photos are not measurements.** Their size is estimated (set the true size from one known
  length) and their surface is far softer than a scan's; see the Accuracy section of [docs/photos-to-3d.md](docs/photos-to-3d.md).
- Photo colouring is only as good as the photo alignment. Use evenly lit photos with fixed exposure and white balance.

## Tests

```bash
.venv/bin/python -m pip install -e ".[dev]"      # Windows: .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests                 # Windows: .\.venv\Scripts\python.exe -m pytest tests
```

The tests generate scans of an object with exactly known geometry. They add noise, outliers, debris and a table, then place the second scan at a random pose. They check that cleaning leaves < 0.1% stray points, that merging recovers the pose to < 0.2°, that mesh dimensions are within 0.15 mm, and that photo projection reproduces the true surface colours. They also run the whole web API flow.
