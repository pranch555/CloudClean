# Make a 3D model from photos

CloudClean can turn ordinary photos of a part into a coloured 3D point cloud. Two tools do it, in their own Docker
image on the DGX Spark (so CloudClean itself needs no GPU libraries):

- **COLMAP** (classical structure from motion, built with CUDA) works out where each photo was taken. It is exact
  but needs texture and overlap.
- **MapAnything** (Meta, checkpoint `facebook/map-anything-apache`, Apache-2.0) turns the placed photos into dense
  points.

The same cameras also colour a scanned model from photos: see [colour-from-photos.md](colour-from-photos.md).

Code:

- `cloudclean/web/jobs_photos3d.py`: the job `photos_to_3d`
- `cloudclean/web/routes_photos3d.py`: the routes, including the printable scale sheet
- `cloudclean/scale_sheet.py`: the scale sheet (layout, PDF, PNG); `python -m cloudclean.scale_sheet FOLDER` writes
  the PDFs and previews for A4 and US Letter
- `tools/recon/photos_to_3d.py`: the script that runs inside the container; `tools/recon/Dockerfile`: its image
- `tools/recon/dtu_check.py`: the camera-accuracy check on DTU (real photos with known cameras)
- `tools/recon/sheet_check.py`: the accuracy check of true size from the scale sheet (renders a CAD flange on the
  sheet, then measures the result, see [Accuracy with the scale sheet](#accuracy-with-the-scale-sheet))
- the assistant tool `photos_to_3d` in `cloudclean/assistant/tools_v3.py`, and the guide entry `photos-to-3d`
- tests: `tests/test_photos3d.py` (a fake `docker run`, so no Docker or GPU needed) and `tests/test_scale_sheet.py`
  (the sheet, its route, and the marker triangulation and fit on virtual cameras)

## What it does, and what it doesn't

**It does:** turn overlapping photos of a part, taken all the way round (12 or more, see [How many photos?](#how-many-photos)), into a coloured point cloud. About a
minute for 36 photos (the very first run also downloads ~5 GB of model weights). The model lands in the project,
with the photos as its parents.

**True size needs the scale sheet.** Photos alone do not show how big something is. Photograph the part lying on
the printed [scale sheet](#true-size-from-the-scale-sheet) and the model comes out in true millimetres, standing on
the sheet (Z = 0), with the sheet and the table cropped away. Without the sheet, MapAnything guesses the size (4 to
60 times off on our tests): set the true size from one length you know before you use any dimension (see
[Setting the true size](#setting-the-true-size)).

**It is not a replacement for the scanner.** Use it for:

- a quick look at a part, or a reference model
- a rough model when the scanner is not at hand
- parts too big for the MetroY

For accuracy work, use the MetroY: its errors are a few hundredths of a millimetre. With the scale sheet the *size*
is right to about 0.02 % in our renders, but the *surface* you measure on is about 0.6 mm off (median; 1.2 mm for
90 % of it) on a 97 mm flange, outside diameters come out 0.7-1.5 mm too big, and bolt holes are not measurable
(see [Accuracy with the scale sheet](#accuracy-with-the-scale-sheet)). Real photos are worse than these renders.

These photos give poor models:

- thin features (ribs, pins, thread crests): they come out soft, or they are missing
- shiny, transparent or plain featureless surfaces: they come out warped, holed or noisy
- a plain background (white table, white wall): the photos have nothing to line up on

## Taking good photos

- **Enough photos, with overlap.** A photo about every 30° all the way round (12 per round), keeping the same
  height from one photo to the next. A second and third round higher up (24-36 photos) rebuild the most.
  Neighbouring photos must overlap: jumps of 90° or more, or changing height and angle at once, cannot be linked.
- **Move the camera, not the part.** The part and everything around it must stay still. If you have to turn the
  part (on a turntable), put the part on a patterned sheet that turns with it, and use a plain, distant background.
- **Keep the settings fixed.** Lock focus and exposure if you can (on most phones, long-press on the part). Do not
  zoom between shots, and keep about the same distance. The part should fill most of the frame.
- **Sharp photos.** Use good, even light and no flash, and hold the camera steady. Delete blurry photos rather than
  keeping them.
- **Matte parts.** Shiny parts need the same matting spray as for the scanner.
- **The scale sheet under the part.** Put the part in the middle of the printed
  [scale sheet](#true-size-from-the-scale-sheet), flat on the table. Its dots help the photos line up and its black
  squares give the true size: keep some of them in every photo. No sheet? Put the part on something with a pattern
  (newspaper) with a ruler or gauge block next to it, and set the true size afterwards.
- **Any phone resolution works.** COLMAP reads features at up to 1600 px, MapAnything works at ~518 px; bigger photos
  only take longer to upload.
- **Formats:** JPG, PNG, TIFF, WEBP, BMP and HEIC/HEIF (iPhone). HEIC photos are turned into upright JPEGs when
  they are added (EXIF rotation applied, focal length kept), because only Safari can show HEIC.

## How many photos?

Measured on 2026-09-25 with the 36 flange photos and fewer of them (`tools/recon/photos_to_3d.py` on the Spark).
Surface error: distance of the rebuilt part to the MetroY scan after the true size is set. Rebuilt: share of the
scan within 2 mm of the photo model (about 60 % is the most photos from above can rebuild of this scan; the rest is
underneath).

| Photos | Cameras | Surface error (median / 90 %) | Rebuilt | Time |
|---|---|---|---|---|
| 36 (three heights) | exact (COLMAP) | 1.8 / 4.9 mm | 59 % | 54 s |
| 24 (two heights) | exact | 1.8 / 5.8 mm | 57 % | 34 s |
| 12 (one height, every 30°) | exact | 1.8 / 4.6 mm | 54 % | 24 s |
| 6 (one height, every 60°) | exact | 1.1 / 3.7 mm | 47 % | 20 s |
| 8 (every 45°, height changing each time) | rough (COLMAP placed 2) | 1.3 / 4.7 mm | 55 % | 20 s |
| 4 (every 90°) | rough | 2.5 / 6.5 mm | 47 % | 18 s |
| 3 (every 120°) | rough | 2.3 / 5.9 mm | 44 % | 18 s |
| 5 (one side only) | exact | only that side rebuilt | 30 % | 20 s |

- **Take 12 or more**, about every 30° all the way round at the same height; 24-36 at two or three heights rebuild
  the most. The surface is equally rough (about 1-2 % of the part's size) with any number: more photos rebuild more
  of the part, they do not make it finer.
- **3-4 photos still give a model**, from MapAnything's rough camera guess (marked rough): smaller and rougher, and
  on some photo sets that guess fails outright (DTU scan24, see [Accuracy](#accuracy)).
- For colouring a scan the rules are stricter: see [colour-from-photos.md](colour-from-photos.md#how-many-photos).

## Using it

**In the app:**

1. Open **Scan → Make a 3D model from photos**.
2. For true size: **Print the scale sheet** (A4 or US Letter), and optionally type what its 100 mm bar measures
   (see [True size from the scale sheet](#true-size-from-the-scale-sheet)).
3. Click **Add photos**. The photos are saved in the current project, and the block shows how many there are.
4. Click **Make the 3D model**. It uses every photo of the project.

The button stays disabled with fewer than 2 photos, when the bar value cannot be right (outside 90-110 mm), or when
the server cannot run the reconstruction. In that case the block shows the reason. The same block also has *How to
take good photos*.

**With the assistant:**

1. Attach the photos with the **Photos** button in the chat box. With *Keep new photos in the project* on (the
   default), they are saved in the project.
2. Ask, for example: "make a 3D model from these photos".

The tool uses every photo of the current project. If the project also holds other photos, tell the assistant which
ones to use. Keeping one part per project avoids this. When the model is done, the assistant shows it.

**While it runs:** progress shows in the block and in **Jobs** (the jobs button at the top right), with the log.

**The result:**

- A new point cloud named `<first photo> · from N photos`, or the name you gave.
- It is in the same project as the photos, and its parents are the photos.
- Only the more confident points are kept: the top 80% by MapAnything's confidence, with the edges of objects
  masked out.
- **On the scale sheet:** true millimetres, standing on the sheet: Z = 0 is the paper under the part, Z points up,
  X and Y follow the sheet (X towards its right edge, Y towards the top edge where the instructions are printed,
  origin in the middle of the page). The sheet, the table and far background are cropped away. The job log and the
  report say `True size from the scale sheet (±x %)`, with how many markers and photos were used.
- **Without the sheet:** the table and the background are usually in the model too. Remove them in **Clean**
  (*Remove the table or turntable*, *Remove things by hand*, *Position & crop*). The model comes out in the camera's
  frame, usually tilted: **Clean → Position & crop → Sit flat on the floor** fixes that. The report says
  `Size unknown: no scale sheet found (...)`.
- For a surface, mesh it in **Mesh** as usual.

### True size from the scale sheet

A printed sheet with 16 black squares (ArUco markers, 20 mm, dictionary `DICT_4X4_50`, ids 0-15) round a dotted
middle, a 100 mm check bar with mm ticks, and one line of instructions. Get it from **Scan → Make a 3D model from
photos → Print the scale sheet** (or `GET /api/photos/scale-sheet?paper=a4|letter`). The same drawing sits in the
middle of an A4 or a US Letter page, so either paper works for any part up to about 130 mm across.

1. **Print it at 100 % ("actual size"), not "fit to page".** The PDF is drawn at exact size and asks viewers not to
   scale it.
2. **Check the print.** Measure the black 100 mm bar with a caliper, jaws on its two ends, and type the length in
   **The 100 mm bar measures** (it is remembered for the next part). Printers are often a few tenths of a percent
   off, and the photos cannot see that: everything on the page, markers included, is scaled the same way. Leave it
   empty and the model inherits the printer's error (the report then says the print was not checked). A value
   outside 90-110 mm is refused: print again at 100 %.
3. **Lay it flat.** On a flat table; tape the corners if the paper curls (curled edges lift the markers, see below).
4. **Put the part in the middle,** on the dots, clear of the black squares.
5. **Photograph it all round** as usual, keeping some black squares in every photo. The sheet must not move between
   photos.

What the reconstruction does with it (`--scale-sheet`, see [How it runs](#how-it-runs)):

- finds the markers in every photo COLMAP placed, triangulates each corner from all photos that see it, and fits
  the sheet's known layout to them (a similarity, markers that do not fit are left out). That gives the scale, the
  sheet's plane and the frame. It needs at least 4 markers seen in at least 3 photos; otherwise the size stays
  unknown and the report says why.
- levels the frame on COLMAP's points on the dots round the part, which is where the part really stands (a curled
  sheet lifts the markers at its edges, not the middle; on a flat sheet this moves it by 0.01 mm or less).
- moves every output (points, COLMAP's points, cameras) into that frame and crops away what is less than 0.5 mm
  above the sheet (the sheet and the table) and anything more than 150 mm beyond the page.
- reports the markers and photos used, the corner reprojection error (px), how well the layout fits (mm), the
  scale uncertainty (%, about 95 %: from the fit, bootstrapped over the markers) and, when some markers sit more
  than 0.5 mm off the layout, a warning that the sheet may not lie flat.

The uncertainty covers what the photos can tell. It does not cover the printer's scale (step 2) or a sheet that
is not flat.

### Setting the true size

A model made on the scale sheet already has its true size: **Clean** then says so (with the uncertainty, and whether
the print was checked with its bar), with *Set it by hand instead* in case. For any other model made from photos (or
anything made from one), **Set the true size** is at the top of **Clean**:

1. Pick the length you know: *Overall length / width / height* (measured along the part's own axes), or a
   distance you measured on this model with **Measure → Dimensions → Point to point**. You can also type it.
2. Type its real length (from a drawing, a caliper or the ruler in the photos).
3. Click **Scale to true size → new model**. The factor (*true ÷ on the model*) is shown before you click, and a
   change of more than ×5 or less than ×0.2 is refused as a likely mistake.

Afterwards the block says the model is true to size (and by which factor), with *Set it again* in case.
Any other model can be scaled with **Clean → Position & crop → More edits → Position · Scale**.

Or tell the assistant the true size, for example "the ruler marks are 100 mm apart" or "the overall length is
82.0 mm". The assistant measures, then scales. The assistant only rescales a model when you give it the true size.

## How it runs

The job copies the photos to a work folder, `<workspace>/recon/<id>/images`. It then runs:

```bash
docker run --rm --name cloudclean-recon-<id> --gpus all --ipc host -e HF_TOKEN -e TORCH_HOME=/hf/torch \
  -v <work>/images:/in:ro -v <work>/out:/out -v <RECON_CACHE>:/hf -v <repo>/tools/recon:/code:ro \
  cloudclean-recon:latest python3 /code/photos_to_3d.py --images /in --out /out --scale-sheet [--ruler-mm 100.2]
```

(`--scale-sheet` only for this job; colouring a scan from photos runs without it and is unchanged.)

- `HF_TOKEN` is passed through when the CloudClean service has it set (faster weight downloads).
- `TORCH_HOME=/hf/torch` keeps torch.hub's DINOv2 code with the weights, so later runs need no internet.
- The container is named, so **cancelling the job stops it**: cancel terminates CloudClean's worker, the job turns
  that into a clean exit and runs `docker rm -f` on its container. A container left over from a server restart
  mid-job is removed when the next reconstruction starts (only one job runs at a time).

Inside, `photos_to_3d.py` does:

1. **Where was each photo taken?** COLMAP: SIFT features (GPU; on the CPU when the GPU is full, with a line in the
   log), matching every pair (up to 120 photos; beyond that each photo with its 20 neighbours in order), incremental
   mapping. The largest connected set of photos is kept. If it holds fewer than half the photos, MapAnything's own
   camera estimate is used instead, and the model is marked *rough* (warning in the job log and report).
2. **Lens distortion removed:** COLMAP undistorts the placed photos into pinhole images (up to 2400 px).
3. **The scale sheet** (`--scale-sheet`): markers found, triangulated and fitted, the frame levelled on the dots
   (see [True size from the scale sheet](#true-size-from-the-scale-sheet)). Needs COLMAP's cameras: with the rough
   fallback the size stays unknown.
4. **Dense points:** MapAnything, given each undistorted photo, its intrinsics and its COLMAP camera. Its output is
   moved into COLMAP's frame by a similarity fitted on the camera positions (the log says how well they agree).
5. **Clean-up:** a point is kept only where at least 2 of its 4 nearest photos (by viewing direction) see the
   surface at the same depth, within 1 %; then the 80 % most confident points. On the sheet, everything is moved into
   its frame (mm) and the sheet, table and far background are cropped away.
6. **Outputs** in `/out`: `points.ply`, `cameras.json`, the undistorted `images/`, and COLMAP's own triangulated
   points `sparse.ply` (exact, in the cameras' frame). The files are handed to the owner of `/out` (the container
   runs as root), so CloudClean can delete its work folder.

The script is mounted from the repo, so changes to `tools/recon/photos_to_3d.py` need no image rebuild. The work
folder is deleted afterwards, whether the job worked or failed. CloudClean runs as the `dgx` user service
(`docs/dgx-spark.md`), so that user must be able to use Docker (the `docker` group).

### The image `cloudclean-recon:latest`

Built on the Spark from [`tools/recon/Dockerfile`](../tools/recon/Dockerfile): the vLLM image, MapAnything, and
COLMAP 3.9.1 compiled with CUDA for the GB10 (`sm_121`). The build takes about 10 minutes:

```bash
cd ~/cloudclean-deploy/recon
git clone https://github.com/facebookresearch/map-anything
cp <repo>/tools/recon/Dockerfile . && docker build -t cloudclean-recon:latest .
```

On the Spark today, `latest` = `mvs` (MapAnything + CUDA COLMAP). `cloudclean-recon:mapanything-only` is the first
image (no COLMAP), kept for comparison.

Check that the service user can see it: `docker image inspect cloudclean-recon:latest`. This is the same check the
app makes (`GET /api/photos-to-3d/status`).

Choices behind this setup:

- **Why the vLLM base image:** it already has a CUDA 13, aarch64 PyTorch that works on the GB10. Building one
  ourselves is the hard part on the Spark.
- **Why COLMAP for the cameras:** MapAnything alone placed the cameras several degrees off on real photos (DTU scan37:
  4.7° median, 53 mm) and failed outright on others (DTU scan24: 60°), which warps or doubles the model. COLMAP
  placed every photo of both to about 1 mm. pycolmap has no aarch64 wheel and Ubuntu's colmap is CPU only
  (~110 s instead of ~25 s for 49 photos), so COLMAP is compiled in the image.
- **Why not COLMAP's dense stereo (yet):** its PatchMatch stereo gave empty depth maps on the GB10 (0 fused
  points): NVCC miscompiles those kernels for Blackwell GPUs. Upstream's workaround (colmap PR #4213: build the
  PatchMatch library as sm_90 PTX) fixes it, and the Dockerfile now has it. Tested on 2026-09-25 in the separate
  tag `cloudclean-recon:pm-test` (`latest` was not rebuilt): PatchMatch works and its surface is far more accurate
  than MapAnything's (see [Accuracy with the scale sheet](#accuracy-with-the-scale-sheet)), but it takes minutes
  instead of seconds and the pipeline does not use it yet.

### Model weights

The weights are about 5 GB: `facebook/map-anything-apache`, Apache-2.0. Keep this checkpoint. The other one,
`facebook/map-anything`, is non-commercial.

They download on first use into `~/cloudclean-deploy/recon/hf`, which is mounted at `/hf`, so the first run is
slow (about 25 minutes on the Spark's connection without a token). Set `HF_TOKEN` in the service for a faster
download, or fetch the weights once by hand:

```bash
docker run --rm -e HF_TOKEN=hf_xxx -v ~/cloudclean-deploy/recon/hf:/hf cloudclean-recon:latest \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/map-anything-apache')"
```

### Settings

These environment variables are read when CloudClean starts. Set them in the systemd service and restart it:

- `CLOUDCLEAN_RECON_IMAGE`: another image tag (default `cloudclean-recon:latest`)
- `CLOUDCLEAN_RECON_CACHE`: another weights folder (default `~/cloudclean-deploy/recon/hf`)

### GPU memory

The GB10's GPU memory is the Spark's system memory. It is shared with any LLM servers on the Spark (on the test
Spark, 2026-09-25: two model servers, ~34 GB and ~67 GB), which reserve their share up front, so what is left is what the
reconstruction gets: 11-16 GB was free that day. `free -g` and `nvidia-smi` show it.

- MapAnything's weights are 4.9 GB (fp32). The usual `from_pretrained(...).to("cuda")` held two to three copies at
  once and failed with *CUDA error: out of memory* with 11-16 GB free. The script now makes the GPU context first,
  builds the encoder without fetching DINOv2's own weights (the checkpoint overwrites them anyway), copies the
  checkpoint in one tensor at a time, and hands freed memory back; it is the same model (see `load_model`).
- Measured then: 12 photos peak at about 10 GB in all (COLMAP, the model, inference). 24 photos need about 10.5-11 GB,
  so with 16 GB free the Spark keeps only ~5 GB for everything else while it runs.
- COLMAP's SIFT falls back to the CPU when the GPU is full (the log says so). On one 12-photo test that moved the
  cameras' size by 0.1 % (the focal length by 0.05 %); the part's heights stayed within 0.013 mm.

## For developers

**Job `photos_to_3d`**

- Payload: `{photo_ids?: [...], project_id?, name?, ruler_mm?}`
  - `photo_ids` given: exactly those photos, in that order. The job fails if one of them is not a photo.
  - no `photo_ids`: every photo of `project_id`. With no project either, every photo in the workspace, unless
    they belong to several projects: then it refuses ("open the project of the part first") so two parts are
    never mixed.
  - it needs at least 2 photos (`MIN_PHOTOS`).
  - `ruler_mm`: what the scale sheet's 100 mm bar measured on the print (`check_ruler`: 90-110 mm, else refused).
  - the photos are always searched for the scale sheet (`--scale-sheet`).
- It creates one asset:
  - `kind: pointcloud`, `operation: "photos"`
  - `parents` = the photo ids
  - `params` = `{photos, model, poses}`, plus `size: "scale sheet"`, `size_uncertainty_pct` (and `ruler_mm` when
    given) when the sheet was found. **Clean → True size** reads these.
- Report fields:
  - `photos`
  - `model`
  - `points`
  - `seconds`
  - `extent_mm`: [x, y, z], from the 1st to the 99th percentile
  - `poses`: `colmap` (exact cameras) or `mapanything` (rough; the report then also has a `warning`)
  - `registered`: how many photos could be placed (the others are left out)
  - `scale`: `True size from the scale sheet (±x %)` (with `; the print was not checked with its 100 mm bar` when
    no `ruler_mm` was given), or `Size unknown: no scale sheet found (<reason>). ...`
  - `scale_sheet`: what the script found (below)
  - `cameras`: from `cameras.json`

**Script protocol** (`tools/recon/photos_to_3d.py`, stdout; the container's stderr is merged into it):

| Line | Meaning |
|---|---|
| `PROGRESS <fraction 0..1> <label>` | The job maps it to 0.02–0.92 of its own progress, with the label. |
| `RESULT {"photos", "registered", "points", "extent_mm", "seconds", "poses", "model", "scale_sheet"?}` | One line, at the end. Without it, the job fails. `scale_sheet` only with `--scale-sheet`. |
| `LOG <text>` | A line for the job log (how many photos COLMAP placed, the scale sheet, how well the dense cameras agree, how many points were kept). |
| `ERROR <message>` | Logged. The last one becomes the job's error: `The reconstruction did not finish (exit N): <message>`. |
| anything else | The last lines are kept; they are shown in the error when there is no `ERROR` line. |

**Outputs in `/out`:**

- `points.ply`: binary PLY, float x y z and uchar RGB. Units: true mm in the scale sheet's frame when it was found
  (the sheet and table cropped away), else "mm" by MapAnything's size guess (unreliable).
- `sparse.ply`: COLMAP's triangulated points (reprojection error < 2 px), same frame and units (and crop). Exact
  relative to the cameras: this is what lining the photos up with a scan should trust.
- `cameras.json`: `{"model", "poses", "photos", "registered", "units", "scale_sheet", "cameras": [...]}`. `units` is
  `mm, scale sheet` or `mm, estimated`; `scale_sheet` is null without `--scale-sheet`, else:
  - found: `{found: true, markers, views, markers_rejected, reprojection_px, residual_mm, max_residual_mm,
    off_plane_mm, uncertainty_pct, ruler_mm, mm_per_unit, camera_height_mm, marker_residual_mm: {id: mm}, level:
    {points, tilt_deg, lift_mm, rms_mm} | null, cropped_points, warning?, summary}`
  - not found: `{found: false, reason, photos_with_markers, markers_seen, ...}`

  One camera per *placed* photo:
  - `image`: the photo's file name in `/in` (the job names them `000.jpg`, `001.jpg`… in the order of the photos)
  - `file`: the undistorted photo in `/out` (`images/000.jpg`) that `K` describes
  - `K`, `width`, `height`: pinhole intrinsics of that undistorted photo
  - `world_to_camera`, `cam_to_world`: 4×4, OpenCV convention (x right, y down, z forward), same frame and units
    as `points.ply`
- Script options: `--poses auto|colmap|mapanything` (default auto: COLMAP, falling back to MapAnything),
  `--no-dense` (cameras and sparse points only), `--keep` (default 0.8: the share of dense points kept, most
  confident first), `--scale-sheet` (look for the sheet), `--ruler-mm` (the measured bar), `--sheet-crop` (default
  0.5 mm: what is less high above the sheet is cropped), `--no-level` (keep the markers' plane as the ground).
- The sheet's layout is copied in the script (`SHEET_MARKERS`: the container cannot import `cloudclean`);
  `tests/test_scale_sheet.py` checks that it matches `cloudclean/scale_sheet.py`.

**Routes:**

- `GET /api/photos-to-3d/status` → `{available, reason}`
- `POST /api/photos-to-3d {photo_ids?, project_id?, name?, ruler_mm?}` → the submitted job
  - 400: fewer than 2 photos, an id that is not a photo, or a bar length outside 90-110 mm
  - 404: unknown asset
  - 503: Docker or the image is missing; the reason is the detail
- `GET /api/photos/scale-sheet?paper=a4|letter&format=pdf|png` → the scale sheet (default A4 PDF, inline); 400 for
  another paper or format

**Assistant tool:** `photos_to_3d {photo_ids?, name?}`

- By default it uses every photo of the current project.
- It runs the job and shows the new model.
- It hints the model to ask for one known dimension and scale with the edit op `scale`
  (`user_requested_scaling=true`), or to send the user to **Clean → Set the true size**. That hint does not yet
  know about the scale sheet: when the report's `scale` says *True size from the scale sheet*, no scaling is needed.

## Troubleshooting

| What you see | What to do |
|---|---|
| *"…needs the reconstruction container, which runs on the DGX Spark"* | Docker is not installed on this machine (e.g. a Windows dev PC). Use CloudClean on the Spark. |
| *"The reconstruction image cloudclean-recon:latest is not built yet"* (HTTP 503; also in the block and from the assistant) | Build the image (above). |
| *"Docker could not be used: …"* | Docker answered with an error (shown). Usually the service user cannot reach the Docker socket: add it to the `docker` group. |
| The first run takes many minutes | The weights (~5 GB) are downloading. Later runs start in seconds. Pre-download them with the command above (with `HF_TOKEN` for speed). |
| It fails when the Spark has no internet | The first run needs the internet for the weights and torch.hub's DINOv2 code; both are cached in `~/cloudclean-deploy/recon/hf` (`TORCH_HOME=/hf/torch`), so later runs work offline. |
| *"CUDA out of memory"* | The GPU shares the Spark's memory with the other projects' model servers (see [GPU memory](#gpu-memory)): 12 photos need about 10 GB free, 24 about 11 GB. Check with `free -g` and `nvidia-smi`, use fewer photos, or try again when more is free. |
| *"Size unknown: no scale sheet found (…)"* | The reason says what was missing. *No scale sheet markers in the photos*: the black squares were not in the photos (or too small, blurred or glared): keep some in every photo. *Only N markers…* / *in N photos*: fewer than 4 markers in at least 3 photos, or they could not be placed exactly. |
| The log warns that *some markers sit up to X mm off the printed layout* | The sheet does not lie flat, or was printed stretched (e.g. unevenly by the printer). Tape it flat on a board and photograph again; check the bar. The size is still used, with a larger ± (a 3 mm curl gave ±0.08 % and no error in the test). |
| *"The 100 mm bar on the scale sheet cannot measure …"* | The value is outside 90-110 mm: a typo, or the sheet was printed with "fit to page". Print at 100 % and measure again. |
| *"The photos gave no 3D points"* | The photos did not overlap enough to line up. Take more, closer together, all the way round. |
| The model is sparse, warped, doubled or has a smeared background | Add more photos with more overlap, and put a patterned sheet under the part. Delete blurry photos. Do not move the part between shots. Matte shiny surfaces. |
| The size is off | Without the sheet this is expected: photos do not show the true size. Use the scale sheet, or set it from one known length (Clean → Set the true size). On the sheet: was the bar measured? A printer that scales the page by 1 % makes the model 1 % off (the test: −0.99 %, and +0.017 % with the bar measured). |
| The log says *"COLMAP placed 12 of 40 photos"* or the model is marked rough | COLMAP could not match the photos: too little texture or overlap, blur, or the part moved. Add a patterned sheet, more photos closer together, and matte a shiny part. |
| A cancelled job keeps the GPU busy | Should not happen: cancel removes the container `cloudclean-recon-<id>`. If one is left (e.g. CloudClean was killed hard), `docker rm -f $(docker ps -aq --filter name=^cloudclean-recon-)`, or just start the next reconstruction, which clears it. |

## Accuracy

Measured on 2026-09-24 (scripts: `tools/recon/dtu_check.py`, and a comparison against a MetroY scan):

**Cameras** (where each photo was taken):

| Test | COLMAP (used) | MapAnything alone (fallback) |
|---|---|---|
| DTU scan37, 49 real photos, 1600×1200, ~700 mm away | all placed; 1.05 mm median (0.32 % of the camera spread) | 53 mm, 4.7° median |
| DTU scan24, 49 real photos | all placed; 0.78 mm median (0.24 %) | failed: 274 mm, 64° |
| 36 renders of a scanned flange on a patterned sheet, 230 mm away | all placed; 0.13 mm, 0.06° median | 44 mm, 10° |

**Size**: MapAnything's guess was 4×, 25× and 60× too big on these tests. The size must be set from a known length.

**Surface**, after the true size is set (36 photos of the 97 × 94 × 22 mm flange, compared with its MetroY scan):
median 1.8 mm, 90 % of points within 4.8 mm. That is 1–2 % of the part's size: fine to see the shape, colours and
rough proportions, not for measuring. The error comes from MapAnything's depth (it works at ~518 px), not from the
cameras.

**Time** on the Spark, warm: 36 photos in 52 s (COLMAP ~25 s, MapAnything ~25 s).

Tried and not used: cropping each photo to the part so MapAnything sees it larger (`--crop`) made the surface worse
(median 8 mm), and rescaling each photo's depth to COLMAP's points moved it ~2 % off (5 mm). What would make the
surface finer: a proper multi-view stereo on the COLMAP cameras (COLMAP 4.x PatchMatch on a newer base image, or a
GPU plane-sweep of our own).

### Accuracy with the scale sheet

Measured on 2026-09-25 with `tools/recon/sheet_check.py` on the Spark. **Synthetic photos**: a CAD flange of known
size (OD 97, plate 8 thick, overall height 22, hub Ø50, bore Ø28, 4 × Ø11 holes on a 72 pitch circle; sprayed-looking
speckle) lying off-centre on the A4 sheet on a wooden table, ray traced with shadows and anti-aliasing at 2400 × 1800
px (a phone photo as the pipeline sees it; 55° field of view), about 250 mm away at two heights (30° and 55°), with
sensor noise and JPEG. Every number compares the result with the truth **with no alignment of any kind**: the
model's own sheet frame against the CAD placed on the sheet. These are best cases: real photos add focus and motion
blur, rolling shutter, glare, a real print and real paper, and a part that is rarely this well textured.

**(a) Size and frame** (the recovered cameras against the true ones; ± is what the job reports)

| Photos (2 heights) | Markers · corner error · layout fit | Size error | Frame | Cameras |
|---|---|---|---|---|
| 12 (4 runs) | 16 · 0.33-0.43 px · 0.03-0.04 mm | −0.006 to −0.025 % (±0.007-0.015 %) | ≤ 0.016° | 0.09-0.10 mm |
| 24 | 16 · 0.27 px · 0.03 mm | +0.002 % (±0.005 %) | 0.008° | 0.07 mm |
| 36 | 16 · 0.27 px · 0.03 mm | −0.000 % (±0.004 %) | 0.003° | 0.06 mm |
| 24, blurred (σ 1.5 px) | 16 · 0.27 px · 0.06 mm | +0.017 % | 0.006° | 0.07 mm |
| 24, lens distortion (k1 −0.04) | 16 · 0.29 px · 0.03 mm | −0.018 % | 0.003° | 0.07 mm |
| 24, sheet edges curled up 3 mm | 16 · 0.25 px · 0.59 mm (warning) | +0.005 % (±0.08 %) | 0.006° | 0.05 mm |
| 24, printed 1 % big, bar not measured | 16 · 0.33 px · 0.03 mm | **−0.99 %** (±0.005 %: the photos cannot see it) | 0.004° | 2.4 mm |
| the same, bar measured (101.0 mm) | 16 · 0.30 px · 0.03 mm | +0.017 % | 0.003° | 0.11 mm |
| 12, COLMAP's SIFT on the CPU (GPU full) | 16 · 0.64 px · 0.06 mm | −0.10 % | 0.017° | 0.25 mm |

**(b) Dimensions**, measured on the result by fitting circles and planes (errors in mm; heights are above the
sheet). *Dense* is the model you get (MapAnything); *sparse* is COLMAP's own points (`sparse.ply`, few but exact).
A circle counts only with 20 points or more round at least half of it.

| | OD 97 | Plate 8 | Height 22 | Hub Ø50 | Bore Ø28 | Bolt holes Ø11 |
|---|---|---|---|---|---|---|
| Dense, 12 photos (3 runs) | +1.48 to +1.49 | +0.14 to +0.18 | −0.97 to −1.01 | +0.66 to +0.68 | not enough points | Ø off by up to 3.8-4.2, centres up to 0.8-2.3 |
| COLMAP PatchMatch (test image, not in the app), 12 photos, 2000 / 2400 px | −0.07 / −0.06 | +0.09 / +0.07 | +0.08 / +0.07 | −0.05 / −0.05 | −0.03 / −0.03 | Ø within 0.05 / 0.02, centres within 0.06 / 0.04 (4 / 3 holes) |
| Sparse, 12 photos | too few | ±0.00 | ±0.00 | too few | too few | too few |
| Sparse, 24 photos | too few | ±0.00 | ±0.00 | −0.05 | −0.03 | too few |
| Sparse, 36 photos | +0.02 | ±0.00 | ±0.00 | −0.02 | −0.05 | too few |
| Sparse, 24, curled sheet, levelled on the dots | too few | ±0.00 | ±0.00 | −0.04 | −0.01 | too few |
| the same, not levelled (markers' plane) | too few | **−0.46** | **−0.46** | −0.04 | −0.01 | too few |
| Sparse, 24, printed 1 % big, bar not measured | too few | −0.07 | −0.22 | −0.54 | −0.32 | too few |

The flange's axis landed 0.04-0.10 mm (dense) and 0.01-0.04 mm (sparse) from where it lay on the sheet.

**(c) Surface**: distance of the recovered points to the true CAD surface; covered = share of the true surface
(without the bottom face) with a point within 1 mm.

| Photos | Sparse median / 90 % | Dense median / 90 % | Covered: sparse · dense |
|---|---|---|---|
| 12 | 0.018 / 0.058 mm (2,100 points) | 0.60-0.62 / 1.22-1.23 mm (127,000 points) | 27 % · 61 % |
| 12, COLMAP PatchMatch (2000 / 2400 px) | | 0.081 / 0.098 mm and 0.069 / 0.085 mm (104,000 / 156,000 points; 6 / 8 minutes) | · 84 / 85 % |
| 24 | 0.018 / 0.075 mm (4,900 points) | not run (memory, below) | 44 % |
| 36 | 0.019 / 0.098 mm (7,200 points) | not run (memory, below) | 51 % |

Dense runs with 24 and 36 photos were stopped: they needed more than the Spark could spare that day while keeping
6 GB for the other projects. The earlier tests found the dense surface equally rough with any number of photos
([How many photos?](#how-many-photos)).

**What it means**, against the MetroY's few hundredths of a mm:

- **The size is not the problem any more.** The sheet gets it to about ±0.02 % (0.02 mm per 100 mm) in these
  renders, and the frame's ground to ±0.01 mm. In practice the print limits it: measure the bar with a caliper
  (a 1 % printer error stays in the model otherwise), and lay the sheet flat (a 3 mm curl moved all heights by
  0.46 mm until the dots under the part were used as the ground).
- **The dense surface is the limit.** The model you measure on is 0.6 mm off at the median and 1.2 mm for 90 % of
  it: edges are rounded and fat (outside diameters 0.7-1.5 mm too big, the small hub top 1 mm low), the large flat
  plate within 0.2 mm, and bolt holes are not measurable. That is 20-50 times the scanner's error: fine for overall
  size and large flat faces to a few tenths of a millimetre, not for diameters, holes or tolerances.
- **COLMAP's own points are as good as the scanner in these renders** (0.02 mm median, heights to 0.01 mm), but there
  are only a few thousand of them, mostly on textured flat faces: almost none on walls and none in holes.
- **COLMAP's PatchMatch closes most of the gap** (tested in `cloudclean-recon:pm-test`, not in the app yet): on the
  same 12 renders its dense surface is 0.07-0.08 mm off at the median (8-9 times better than MapAnything), diameters
  within 0.07 mm, bolt holes within 0.05 mm, with a consistent +0.07-0.09 mm on the top faces. It takes 6-8 minutes
  for 12 photos, and uses about 1 GB of memory instead of about 10 GB.
