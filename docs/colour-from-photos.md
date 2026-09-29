# Colour from photos

Give your scan or mesh the real colours of the part, taken from ordinary photos of it. You don't have to line
anything up by hand: CloudClean works out where each photo was taken and fits the photos onto your model by itself.

Where: **Mesh → Colour from photos → Colour it from all photos**. You can also ask the assistant, for example
"colour my scan from the photos". To line up a single photo yourself, use **line up one photo by hand** in the same
place.

Code:

- `cloudclean/photo_align.py`: finds the part in the photo model and lines it up with your model
- `cloudclean/texture.py`: `colour_points` (point clouds) and `colorize_mesh` (meshes)
- `cloudclean/web/jobs_colour_photos.py`: the job `colour_from_photos`
- `cloudclean/web/routes_colour_photos.py`: `POST /api/colour-from-photos`
- the assistant tool `colour_from_photos`, and the guide entry `mesh.colour`
- tests: `tests/test_colour_photos.py` (synthetic photos, no Docker or GPU needed)

## What it does

- **The result is a new model.** It has the same points or triangles as the one you picked, now in colour. Your
  original model is kept as it was.
- **A mesh** gets colours on its vertices. It also gets a texture image (4096 × 4096 by default), saved as a textured
  GLB and OBJ.
- **A point cloud** gets a colour on every point.
- **The shape and size never change.** The photos are only used for colour. Every dimension you measure on the
  coloured model is the same as on the original.

It runs on the DGX Spark, because the photos go through the same reconstruction container as
[Make a 3D model from photos](photos-to-3d.md). It takes a few minutes.

## Taking the photos

- **How many:** 12 or more, about every 30° all the way round at the same height (see [How many photos?](#how-many-photos)). 6, every 60°, is the fewest that worked.
- **Go all the way round:** take the photos in one or two rings, one low and one higher up. Neighbouring photos should
  overlap by about two thirds.
- **Keep the whole part in view,** filling most of the frame, and keep it sharp. Don't zoom in and out between
  shots.
- **Put a patterned sheet under the part:** newspaper, or printed random dots. It helps the photos be placed exactly.
  Avoid a plain white table.
- **Light it evenly:** use soft, even light, with no hard shadows and no strong reflections. The photos' colours are
  what ends up on the model, shadows and reflections included.
- **Don't move the part** while you take the photos.

Add the photos to the part's project: use the **Photos** button in the chat, or **Models → + Add**. By
default every photo in the project is used.

## How many photos?

Measured on 2026-09-25: the 36 flange photos (three rings of 12, at 25°, 45° and 65° above the table), and fewer
of them, coloured through the app. "Colour agreement" is how well the colours match the part's true pattern
(0.90 with all 36; it drops to ~0.84 when the model is shifted 0.25 mm).

| Photos | Result | Cameras vs the truth | Colour agreement |
|---|---|---|---|
| 36 (three heights) | coloured, trusted | 0.25 mm / 0.06° | 0.90 |
| 24 (two heights) | coloured, trusted | 0.27 mm / 0.05° | 0.90 |
| 12 (one height, every 30°) | coloured, trusted | 0.33 mm / 0.12° | 0.88 |
| 6 (one height, every 60°) | coloured, trusted - the photo check repaired the placement | 0.39 mm / 0.07° (16.5 mm / 5.9° before) | 0.88 |
| 8 (every 45°, height changing each time) | refused: only 2 could be placed | - | - |
| 4 (every 90°) or 3 (every 120°) | refused: only 2 could be placed | - | - |
| 5 (one side only, every 30°) | refused: the photos do not agree placed on the model | - | - |

- **Take 12 or more**, about every 30° all the way round at the same height. More photos colour more of the
  surface (the top from a higher round, the sides from a lower one) but do not place the colours more exactly.
- **6, every 60° at one height, is the fewest that worked.** Fewer, or bigger jumps, cannot be placed: neighbouring
  photos must show enough of the same surface. Changing height and angle at the same time counts as a big jump.
- **Go all the way round.** Photos of one side only could not be placed on the flange reliably (it looks alike
  from several sides), so they were refused.
- Photos that cannot be placed are left out (the log says how many); fewer than 3 placed stops the job with advice.

## How it works

1. **Placing the photos.** The reconstruction container works out where each photo was taken and builds a rough
   dense 3D model from them. COLMAP places the cameras and also gives its own triangulated points
   (`sparse.ply`), which agree exactly with the cameras. The photo model has its own position and an unknown
   scale: it can easily be 25 times too big or too small.
2. **Lining up with your model.** First CloudClean finds the part in the photo model. It looks for the point the
   photos aim at and removes the table the cameras look down on, using COLMAP's points, which lie exactly on the
   table. (On real photos the dense model's table sat 2 % of the camera distance too low and was 0.7 % thick, so a
   table found in the dense model alone left a ring of it around the part and the fit went wrong.) Then it finds
   the scale, rotation and position that put the photo model onto your model:
   - it guesses the scale from the part's size
   - it tries every orientation
   - it refines scale, rotation and position together with ICP
   - the strongest candidate poses are all refined again on COLMAP's points, and those decide: a hole pattern or a
     round shape can fit several ways on the rough dense model, but only one way on the exact points

   The dense model is only good to about 2 % of the part's size, and its scale can be 2–3 % off against the
   cameras. So the final fit is made on COLMAP's own points on the part, because a scale error there would shift
   every colour.
3. **Checking against the photos.** Placed right, each point of the model looks the same in every photo that
   sees it; placed wrong, points near holes and edges land on the table, differently in each photo. CloudClean scores
   this photo agreement, tries the part turned about each of its axes, and nudges the placement (turn, shift, size,
   within 20° and 15 %) while the agreement improves. This matters with few photos: COLMAP's points then lie mostly
   on flat faces, which still fit when turned - with 6 photos the geometry alone left the cameras 6° and 16 mm off;
   the photo check brought them to 0.07° and 0.4 mm. A placement is kept only if COLMAP's points still lie on the
   model, and if the photos do not clearly agree the job stops instead of painting wrong colours.
4. **Painting.** Each camera is moved into your model's position and scale, and the photos are projected onto the
   model:
   - Each point takes its colour from the photos that see it, and a photo that faces the surface head-on counts
     most.
   - Surface hidden behind other surface in a photo is skipped, and so are the edges of the part's outline in each
     photo (where the background could bleed in).
   - Areas no photo sees, such as the underside, take the colour of the surface around them.

## How good is it

The job log and the result's report show how well the photos lined up:

- **scale ×…**: how much the photo model was scaled to match your model.
- **% within … mm**: the share of the photo points on the part that lie on your model, within that distance.
  About 90 % or more is a good fit.
- **the photos cover …% of the model**: how much of your model the photos show. The underside is never seen, so
  60–80 % is normal.
- **photo agreement** (0-1) and the score a clearly wrong placement gets for this part: the further apart, the
  surer the placement. The job stops when they are too close.
- **trusted** means all of these hold:
  - the fit was made on COLMAP's points, most of which lie on the model
  - the photos cover enough of it, and the size matches
  - the photos agree clearly better placed as found than placed wrong, and no turned placement agrees as well

Measured on 2026-09-24:

- **Synthetic tests** with a known answer: within 0.015 % in scale, 0.03° in rotation and 0.012 mm in position,
  also with the dense photo model 2.5 % too big and its table thick and offset.
- **36 photos of a flange scan** (97 mm; photos rendered from the MetroY scan with a known speckled texture, on a
  patterned sheet, run on the Spark through the app): all 36 photos placed; scale within 0.10 %, rotation within
  0.02°, scan points within 0.13 mm of where they belong (the check itself is good to about that). The colours
  correlate 0.90 with the true texture (point cloud; 0.88 on the mesh) and drop as soon as the model is shifted
  0.25 mm, so they sit well within a quarter of a millimetre. 97 % of the surface was seen directly.
- **Time** on the Spark: ~1 minute for a 434,000-point scan (placing the photos ~50 s, lining up ~10 s, colouring
  ~3 s); ~3.5 minutes for its mesh with a 4096 × 4096 texture (the texture bake takes ~75 s).

Real phone photos have not been measured yet: blur, lighting changes and reflections will lower the colour
agreement, but not the placement, which comes from COLMAP.

Limits:

- **Parts that look the same from several sides** (a plain round pin, a symmetric bracket) can be lined up turned.
  When the photos cannot tell the turns apart either, CloudClean warns "the photos agree almost as well" turned by
  some angle. Check the colours; if they are turned, line up one photo by hand.
- **Only exactly placed photos are used.** When COLMAP cannot place a photo it is left out; with fewer than 3
  placed the job stops. (MapAnything's rough camera guess, which Make a 3D model from photos falls back to, painted
  every colour in the wrong place in tests, so colouring never uses it.)
- **Shiny, transparent or plain surfaces** give few photo points, so they line up less well and their colours are
  less reliable.
- **Use the scan of the part only.** If the scan still includes a large turntable or table, remove it first:
  Clean → Remove the table or turntable.

## Troubleshooting

| Message | What to do |
|---|---|
| "The photos could not be lined up with …: only …% of the part in the photos lies on the model" | Check that the photos and the model show the same part. Then take more photos all the way round, with the whole part in view. |
| "… the photos cover only …% of the model" | The photos show too little of the part, or it was not found in them. Take photos all the way round, with the part filling most of the frame. |
| "Only N of M photos could be placed" | The photos don't overlap enough, or the background is plain. Take more photos with more overlap, on a patterned sheet. |
| "A pose turned …° fits almost as well" | The part looks alike from several sides. Check the colours, and if they are wrong, line up one photo by hand. |
| "Add at least 3 photos of the part" | Add photos to the part's project, or pick them yourself. |
| "Fewer than 3 of the photos could be placed exactly: neighbouring photos must overlap…" | The photos are too far apart (90° or more, or angle and height changing together). Take them about every 30° at the same height. |
| "The photos could not be lined up with … reliably: placed on it, they do not agree on its colours" | Usually photos of one side only, or too few. Take 12 or more all the way round, with the whole part in view. |
| "Making 3D models from photos needs the reconstruction container…" / "not built yet" | This only works on the DGX Spark with the reconstruction image built. See [photos-to-3d.md](photos-to-3d.md). |
| Colours look smeared at the edges | Use more photos from more directions, with even light. |

## Merging scans with photos (tested 2026-09-25, not in the app)

Could 12+ photos of the item decide how two scans fit together? Tested on the flange's two faces (scanned
separately; the 10 shared stickers give the true pose to 0.145 mm) with 36 photos of the merged part standing on
edge, the stickers ignored:

- The geometry alone found four ways; all four were face-on-face, 180° off (flat faces overlap more that way).
- Photos lined up with each candidate merge: the share of the photos' own points lying on the true merge was
  0.75-0.975 depending on the run, on wrong merges 0.42-0.79 - in one run a wrong merge scored higher than the true
  one. Photo agreement (colours) separated them no better (0.64 vs up to 0.56).
- Proposing the right pose from the photos (photos placed on each scan alone, then combined) was 4-170° off: a scan
  of one face can sit on either face of the photographed item, and the dense photo model is only good to ~2 %.

So photos are not used to decide merges automatically. What works: the stickers (Align → Line up on stickers:
0.12° / 0.11 mm here), point pairs, and - for telling apart options that look clearly different - the assistant
comparing drawn options with your photos (Align → Which way do the scans fit?).
`photo_align.align_photo_model` keeps the options this needed (`scale=` a known scale, `partial=` for one scan of a
larger item, and the `candidates` it returns) for a future attempt.
