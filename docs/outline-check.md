# Outline check: a part against its golden model from one photo

The outline check measures a part from its **silhouette** in a photo and compares it with the golden model, like an
optical comparator (shadowgraph). The part lies in the blank middle of the printed **photo check sheet**, lit from
behind on a light pad, and you take one photo, roughly straight down. That replaces a 20-30 minute scan for
everything that shows on the outline:

* **Screws, bolts, pins, shafts** (any part that is round about an axis, threads and knurls included): overall
  length, head height, length under head, head diameter (a knurl's crests), thread major diameter, thread pitch,
  and the thread's minor diameter when the photo is fine enough (48 MP).
* **Threads, groove by groove** ([below](#threads-groove-by-groove)): each tooth's spacing to the next (pitch error),
  the cumulative lead error, each crest's radius, and each groove's depth, on both sides of the screw, against the
  golden model. This needs the screw lying **across** the view.
* **Other parts:** lengths, widths, steps and diameters between the faces seen edge-on.

It does **not** rebuild a 3D surface from photos. That was tried ([photos-to-3d.md](photos-to-3d.md#accuracy-with-the-scale-sheet)):
the sheet placed the cameras to 0.02 % in scale, but the dense surface was 0.6-1.5 mm off. Instead, the golden model
itself is placed where the part lies, and its outline is compared with the edge in the photo, to a few hundredths of
a pixel.

**Status:** validated on synthetic photos only (below), including photos rendered from the real golden model of a
1"-8 socket head screw (McMaster 91251A917). Every value comes with an uncertainty. The terms marked *assumed* in
each budget have not been measured on real photos yet, and the report says so (`"validated": false`).

Code:

* `cloudclean/outline_check.py`: the check (golden model and its profile, pose, outline, fit, measurements,
  uncertainty, overlay)
* `cloudclean/outline_camera.py`: photos, the camera model, calibration, markers and the sheet's pose
* `cloudclean/outline_thread.py`: a thread groove by groove (per tooth and per groove, the resolution, view and
  shadow gates, the thread close-up and chart)
* `cloudclean/outline_synthetic.py`: synthetic test photos (backlit, or front-lit with shadows) and test parts (bolt,
  L-plate, knurled threaded screw), errors put into a real round part's mesh by moving its vertices (its ends and
  radius, or single grooves and teeth of its thread), and a part's exact silhouette
* `tools/outline_check/outline_check.py`: the command line (`calibrate`, `check`, `render`)
* `tools/outline_check/validate.py`: the accuracy tables below; `tools/outline_check/thread_validate.py`: the
  groove-by-groove table
* `tests/test_outline_check.py`: the tests (about 47 s)

## What you need

* The **photo check sheet**, printed at 100 % (actual size, not "fit to page"): `cloudclean-photo-check-sheet-a4.pdf`
  or `-letter.pdf` (`python -m cloudclean.scale_sheet FOLDER` writes them). It is the scale sheet with a blank middle
  of 140 x 180 mm.
* A **caliper**. Measure the sheet's 100 mm bar from end to end (this is `--bar-mm`). If you can, also measure from
  the top edge of marker 15 to the bottom edge of marker 13 (the left column, 2nd and 4th square from the top;
  120.0 mm as drawn; this is `--height-mm`). Printers scale x and y differently.
  * Without `--bar-mm`, the check assumes the print is right to ±1 %, which is far too uncertain for ±0.1 mm.
  * A sheet printed at another scale still works when it is measured. For example, a print at 76.2 % gives a bar
    of 76.2 mm and a height of 91.44 mm. The check then warns that the blank middle is only 107 x 137 mm; the part
    must still fit inside it.
* A **light pad** (LED tracing pad) under the sheet. The paper glows, the black squares stay dark, and the part
  becomes a dark silhouette.
* A **phone**: iPhone main camera at **1x**, **HEIF Max (48 MP)**. The photos are used in the sensor's own pixel
  layout, so it does not matter how you hold the phone.

## Using it

Commands run from the repository folder (`C:\Users\user\Desktop\CloudClean`).

1. **Calibrate the phone, once.** Take 8-15 photos of the sheet alone (no part, light pad on).
   * Tilt the phone 15-40° from straight down, from different sides around the sheet.
   * Shoot from the distance you will use for the checks (25-30 cm).
   * Keep all markers, or nearly all, in view.
   * Use the same camera, zoom (1x) and format as for the checks.
   * Lock focus and exposure at the working distance (long-press on the sheet: AE/AF LOCK).

   ```
   .venv\Scripts\python.exe tools\outline_check\outline_check.py calibrate PHOTOS_OR_FOLDER --bar-mm 100.04 --height-mm 120.03 --out camera.json
   ```

   Aim for a reprojection error under 0.3 px and f known to better than ±0.1 %. The camera file is keyed by the
   photos' EXIF make, model, lens and image size. The check warns when a photo does not match.

2. **Photograph the part.**
   * Lay it in the blank middle, resting as it naturally rests: flat, or a screw on its side, not propped up, clean
     underneath. The paper must lie flat on the pad.
   * Turn the room lights down, so the pad is the main light.
   * Lower the exposure until the paper is light grey, not white (drag the exposure slider down). The check warns
     when the paper around the part is clipped.
   * Hold the phone roughly straight down at the calibration distance, with all markers in view.
   * Take 2-3 photos, moving the phone a little between them. One of them tilted about 20° is worthwhile: a tilted
     photo measures its own focal length from the markers, which removes the focus uncertainty (see below).

3. **Run the check.**

   ```
   .venv\Scripts\python.exe tools\outline_check\outline_check.py check GOLDEN.step PHOTOS_OR_FOLDER --camera camera.json --paper a4 --bar-mm 100.04 --height-mm 120.03 --tolerance 0.1 --out result
   ```

   `GOLDEN` is a STEP / IGES file (tessellated at 0.002 mm) or a mesh (STL / PLY / OBJ) in mm. The command prints one
   line per value and writes `result/report.json` and one overlay picture per photo (`*-outline.png`). The overlay
   shows the golden model's outline where the part lies, coloured by how far the photo's edge is outside (red, more
   material) or inside (blue) it, saturating at the tolerance. Grey marks outline that measures nothing: corners,
   chamfers, fillets. For a threaded golden model it also prints the thread groove by groove and writes a thread
   close-up (`*-thread.png`) and a chart (`*-thread-chart.png`): see [Threads groove by groove](#threads-groove-by-groove).
   For that, lay the screw across the view and hold the camera over the thread, tilted 4-5° towards the head.

   With the MetroY's calibrated camera instead of a phone, use `--metroy-camparam tools/metroy/camparam.yaml` in
   place of `--camera` (`--metroy-camera R` for the right camera). These are the raw unrectified 1600 x 1200 images
   (`tools/outline_check/metroy_photo.py` takes them: laser off, 32 frames averaged). The file has no uncertainties,
   so ±0.1 % on f is assumed. Lay the screw across the scanner's view (horizontal in its picture), and use only the
   scanner's own fill light: a lamp's shadow moves the outline by millimetres.

### Reading the result

For each value, the report gives the golden value, the photo's value, the difference, **U** (the expanded
uncertainty, k = 2) and the verdict (ISO 14253-1, as in `golden.py`):

| Verdict | Meaning |
|---|---|
| ok | \|difference\| + U ≤ tolerance |
| off | \|difference\| − U > tolerance |
| close | in between: too close to call from the photo; measure it by hand |
| not measured | not on the outline in this photo (the reason says which, e.g. a thread's roots at 12 MP) |

Values of round parts have plain names: *Overall length*, *Head height*, *Length under head*, *Head diameter (knurl
crests)*, *Thread major diameter*, *Thread minor diameter*, *Thread pitch*, *Shank diameter*. Other parts use the
golden model check's names, e.g. *Thickness: left face to right face*. Stretches of outline that no golden face
explains get names like *Width: outline edge facing left (-X) to outline edge facing right (+X)*.

With several photos, `combined` holds one value per measurement: the weighted mean of the photos. Each photo's own
uncertainty averages down, but the parts common to all photos do not (camera, caliper, focus, toner, edge). If the
photos disagree more than their uncertainties allow, U grows (Birge ratio, `consistency`).

Each photo's `budget` gives the standard uncertainties that make up U:

| Term | What it covers |
|---|---|
| `statistical` | noise of the edge samples (at most `golden.N_EFF` = 400 per feature counted as independent) |
| `camera_and_scale` | Monte Carlo: the calibration's intrinsics (full covariance), a focus change since the calibration (±0.3 %, *assumed*; replaced by the photo's own focal length when it is tilted ≥ 10°), and the caliper readings (±0.05 mm each, rectangular) |
| `placement` | Monte Carlo: marker corner noise, the part tilting off its resting face (±0.05°, *assumed*: dirt, burrs), the paper's height under the part (±0.05 mm, *assumed*), and features that are not on the outline (± the tolerance, *assumed*: see below) |
| `toner_spread_assumed` | the printer's toner spread, ±0.02 mm per edge (*assumed*). The bar reading carries it too, so it cancels for lengths near 100 mm. |
| `edge_assumed` | the part's edge against the markers' edge in the same photo: 0.15 px (*assumed*). This covers defocus of a part edge above the paper, reflections, and processing. |
| `knurl_roll` | a knurled head only: the knurl's outline depends on how the part is rolled, and the roll comes from the thread, whose start on the real part need not be where the golden model has it. Each side is allowed ± the knurl's ripple (computed from the golden model: 0.009 mm on the 91251A917). |

## How it works

1. **Camera.** OpenCV's pinhole model with square pixels, radial k1-k3 and tangential p1-p2.
   * The calibration (`outline_camera.calibrate`) finds the markers with ArUco.
   * It then finds each marker's four outer edges to sub-pixel precision, at the 50 % crossing on linear light (the
     sRGB curve undone), 8-240 points per edge.
   * Straight lines are fitted through the edge points (in undistorted coordinates once the camera is known) and
     intersected to give the corners.
   * `cv2.calibrateCamera` runs three rounds, each re-fitting every view's marker dilation (below).
   * The full covariance of the intrinsics comes from the Jacobian of all views.
2. **Sheet.** The printed layout, scaled by the caliper readings (x from the bar, y from the 120 mm column), is
   fitted to the corners. The fit gives rotation, position and the markers' **dilation**: how much bigger the black
   squares look than drawn.
   * In the same photo, the part's dark silhouette on the glowing paper has the same edge bias as the black squares.
     Blur on a non-linear tone curve, over-exposure and sharpening all shift such an edge the same way.
   * So the markers' dilation is taken off the part's edges (`edge_correction` "markers").
3. **Silhouette and resting pose.** The blank middle (scaled like the print) is mapped top-down, and the part's dark
   silhouette is found there.
   * The golden mesh's stable resting poses are its convex-hull faces with the centre of mass above them.
   * A round part rolled about its axis counts as one pose.
   * For each pose, the position and turn whose shadow (projected from the camera centre) best overlaps the
     silhouette are searched.
4. **What the outline can measure.**
   * **Round parts** (`turned_profile`). The golden mesh is round when every meridian reaches the outer envelope
     within a short stretch and ±10°, which allows for a helix or a knurl. A hexagon is not round. The axis comes
     from the principal axes, refined on the flat faces square to it. Rays cast inward along 72 meridians give the
     profile:
     * *zones* of one outer radius (a head, a shank, a thread's crests);
     * their texture: a *thread* is periodic along the axis and shifts by a quarter pitch at 90° round (a helix; its
       pitch comes from the flanks' mid-depth crossings). A *knurl* is texture without that period.
     * *shoulders*: flat faces square to the axis that stand out of the outline (the ends, a head's underside), each
       with the fillet or chamfer between it and the next zone.

     A socket's floor inside the head is found and left out.
   * **Other parts:** golden.find_faces' flat and round faces, as in the golden model check. On top, the outline's own
     straight runs and circular arcs where no face is (a drafted or chamfered wall, a face find_faces does not
     recognise) become features of their own for that photo.
5. **Outline.** The golden mesh is projected with the lens distortion. Its occluding contour consists of the edges
   between triangles facing the camera and triangles facing away. It is sampled every 3 px.
   * **Which samples are kept.** Ray casts must show the model solid inside the sample and empty outside it, both
     over the whole search window and at ±0.4 px. That keeps only the silhouette's boundary itself: a knurl or a
     thread also has contour edges a pixel inside it.
   * **Which feature each sample belongs to.** A shoulder takes its rim or fillet where the outline runs across the
     axis. A zone takes its crests where the outline runs along it; a thread also has roots and flanks.
   * Samples on plain round faces are moved onto the exact cylinder, so the tessellation's chords do not bias the
     diameters.
   * **The edge in the photo** is searched along the outline's normal: the 50 % crossing between the local paper
     and part levels, on linear light. It is compared with where the same edge finder puts the *model's own* edge.
     That position is ray cast around the sample and blurred like the photo (the blur is measured from the edges
     themselves). It is 0 on a long straight edge, and inward on a narrow thread crest, a small hole or a tight
     curve, where blur pulls the 50 % line in.
6. **One fit for pose and feature shifts.** Every feature on the outline gets an outward shift d: a face or shoulder
   along its normal, a zone radially. A thread also gets a *stretch* of its flanks along the axis, which is its pitch.
   * A sample on the rim between two features moves with both. So do the part's lowest points and its resting face,
     which set its height and tilt.
   * One robust least-squares fit (Tukey, Levenberg-Marquardt) finds the position, the turn, the **roll** of a
     threaded or knurled part about its own axis, and all the shifts together. Outliers are judged within each
     feature, so a feature whose shift is still settling is never thrown out as a whole.
   * The depth is never free: the part touches the sheet (z = 0).
   * A round part lying on its side **settles** onto the line of its hull under its centre of mass, at whatever roll.
     A thread's last crest lies further along at another roll, so the screw's tilt changes with its roll (4.43-4.57°
     for the 91251A917).
   * **Why roll needs care.** A thread's outline matches at any slide along the axis if the part rolls with it (the
     screw motion). So first the slide is pinned on the shoulders and plain crests (not the thread's crests: where
     they show along the axis depends on the roll), then the roll is found by scanning all round, then everything is
     fitted. The common slide of all shoulders is held at zero; the pose takes it. With only one shoulder on the
     outline (the other end out of the picture or hidden), that shoulder is held. Left free, the part could slide a
     whole pitch with its thread while the shoulder's shift took it up (this happened on the MetroY try5 photos).
     The search window stays within a third of the thread's pitch in pixels.
   * **Search window.** The first window covers at least 1.6 mm at the part: the silhouette's starting pose can be
     1-1.5 mm off along a long part, and its ends must be inside the window to pull it in. The window narrows only
     once the pose has settled in it.
7. **Measurements** follow from the shifts:
   * overall length = golden + d(one end) + d(other end);
   * head height: the head top to the underside;
   * length under head = golden + d(end) − d(underside);
   * diameters = golden + 2 d;
   * pitch = golden × (1 + stretch);
   * for other parts, golden.py's overall size, thickness, gap, step and diameters, and the widths, steps and
     diameters of the outline's own runs and arcs.

### Why the tilt is not fitted

One view barely sees whether the part is tilted, only through perspective. A long part lifted at one end looks
longer: 0.05° on 107 mm moves the far end's outline by about 0.02 mm. Letting the fit tilt the part turned edge
errors of a few hundredths of a pixel into length errors. So the part rests on its face as it physically does
(settled at its roll, for round parts), and a ±0.05° tilt goes into the uncertainty. `fit_tilt=true` turns the fit
back on.

### Focal length, distance and height (the depth problem)

A straight-down photo of a flat sheet cannot tell a longer focal length from a greater distance. The sheet's pose
absorbs any focal-length error exactly, *on the paper*. The part's outline, however, lies at a height h above the
paper, and its size in the photo depends on h / Z (Z is the camera's distance, about 280 mm). A focal-length error ε
therefore scales the outline by about ε·h/Z.

For the bolt below, the outline's end rims are 24-36 mm up, so h/Z ≈ 0.1. A 1 % focal-length error moved:

* the 107.1 mm overall length by 0.117 mm;
* the 25.25 head height by 0.067 mm;
* the Ø36 by 0.026 mm (the `f+1%` row).

The calibration pins f to about ±0.001 % on synthetic photos (real phones: expect ±0.02-0.1 %), so that is not the
risk. **Focus is.** The phone focuses by moving its lens: for a 6.9 mm lens, the image distance changes by 0.8 %
between 25 and 35 cm. Two ways to handle it:

* Lock focus, and photograph at the calibration distance. The check allows ±0.3 % (assumed) for this.
* Or tilt the photo about 20°. A tilted photo of the flat sheet shows its own focal length through perspective
  (±0.003 % on synthetic photos), and that value is used instead (`camera.focal_check` in the report). With the
  calibration's f deliberately 1 % off, the tilted photo still measured everything within 0.0012 mm
  (`f+1%_tilted`).

The same h/Z effect is why features that are **not** on the outline still matter. Take a plate lying flat: its
thickness cannot be seen from above, but its top edges form the outline. A plate 0.05 mm thicker than drawn raises
the outline 0.05 mm and makes a 70 mm length look 0.0125 mm longer (the `plate` row: +0.0146). Such features are
allowed ± the tolerance in the `placement` term.

## Threads groove by groove

`cloudclean/outline_thread.py` reads a thread on the outline tooth by tooth, the way a thread is checked on a
shadowgraph, and compares every tooth and groove with the golden model. It runs on its own whenever the golden model
has a thread (`--set thread=off` skips it).

### What it reports

The two sides of the screw's outline are read separately (side A and side B; they show the helix half a pitch
apart). Teeth and grooves are numbered from the thread's start, the end away from the head; `position` is mm along
the axis from where the full thread starts.

* **Per tooth:** its position along the axis; the **pitch error** (its spacing to the next tooth minus the golden
  pitch); the **lead error** (how far it has moved along the axis since the first full tooth: a stretched or
  shrunk thread shows as a steady drift); the **crest radius** and its error.
* **Per groove:** the **root radius** and its error, and the **groove depth**: from the crests' envelope (a straight
  line through that side's crests, so one nicked crest does not make its neighbours' grooves deeper) down to the
  root, against the golden depth (2.067 mm on the 91251A917).
* Each value has U (k = 2) and a verdict, like everything else.
* **Excluded:** teeth and grooves that are not complete on the golden model itself (the chamfered start, the runout
  under the head). They are listed as "excluded", with the reason.
* **Not measured,** with the reason: a tooth or groove the photo does not show well enough. The reasons are counted
  in the summary, e.g. "groove depths not measured (27): its root is not on the outline here: the line of sight is
  74-84° from the axis here...".

In the measurement table, five rows sum the thread up (the same rows for every photo):

| Row | Value |
|---|---|
| Thread pitch error, worst tooth | the tooth with the worst verdict, then the largest error (side, tooth, position given) |
| Thread lead error (cumulative), worst tooth | the same for the lead error |
| Thread crest radius error, worst tooth | the same for the crests |
| Groove depth, mean | the mean depth of the measured grooves, against the golden depth |
| Groove depth error, worst groove | the groove with the worst verdict |

With several photos, a "worst" row keeps the worst photo's value; the mean depth is averaged as usual.

The full per-tooth and per-groove table is printed by `check` and written to `report.json`
(`photos[i].thread.sides[].teeth[]` / `grooves[]`, with a `summary`). Two pictures per photo:

* `NAME-thread.png`: the thread close up, every outline point coloured by how far the photo's edge is out (red) or
  in (blue) from the golden thread;
* `NAME-thread-chart.png`: pitch error per tooth with the lead error as a line, and depth error per groove, along
  the thread, both sides, with U and the tolerance.

### How

1. The golden model stands at the fitted pose, rolled about its axis as the fit found it (the roll lines its helix
   up with the photo's).
2. The thread's outline is sampled every 1/30 of a pitch. The edge search window is a sixth of the pitch either way
   (at most 8 px), so it reaches into the groove near its root.
3. Each sample's place on the helix, `s = (t − t_root) / pitch − hand · angle / 360°`, says which tooth or groove it
   belongs to (roots at whole s, crests half way).
4. Only samples where the golden model's own exact silhouette is at the sample (within 0.1 px) are kept. Near the
   perspective limit, a root is seen past the next tooth's crest, and the outline there is that crest, not the root.
5. The photo's edge is compared with the golden model's own blurred edge, as in the main check (the model's
   silhouette ray cast and blurred like the photo, through the same edge finder). A narrow crest or root, which blur
   pulls in, then reads the same in both.
6. **A tooth's position** comes from both its flanks at once (least squares in two unknowns: a shift along the axis
   moves one flank out and the other in; a fatter tooth moves both out).
7. **A crest or root** is the median radial deviation of its samples.
8. **The groove depth** is the crests' envelope at the groove minus the root.

U (k = 2) for each value combines:

* the statistical spread, counting neighbouring samples within twice the blur as one (a crest or root with fewer
  than three samples of its own takes the scatter of the whole thread's outline);
* a local edge error of 0.03 px per tooth (*assumed*: pixel grid, demosaicing, compression);
* the blur model: the photo's blur estimate is taken as uncertain by ±10 % (*assumed*), and the value's change with
  it is computed on the golden model;
* a shape correction: where blur is not small against a root or crest, a real change of its height reads smaller
  than it is (a groove filled 0.1 mm shallower is also wider, so blur pulls its root out less than the golden
  root's). The golden tooth's profile is filled or cut by trial amounts, blurred and read the same way; the reading
  is turned back into the real change, and every uncertainty of the reading is scaled by the same gain. Half the
  correction goes into U, because the real defect's shape is not known (flat-filled roots and flat-cut crests are
  modelled). At 7 px/mm a groove 0.10 mm shallower reads 0.067 before the correction and 0.106 after;
* lengths along the part (the lead error over its distance) carry the photo's own scale uncertainty;
* absolute crest and root radii carry the 0.15 px edge term; pitch, lead and depth do not (it cancels).

### What the photo must show

**1. The screw must lie across the view.** The teeth are on the outline only where the line of sight is nearly
square to the axis. Rays that graze along the axis cross several teeth, and the outline closes up to the crests'
envelope. On the 91251A917, from the golden model:

| Angle between the axis and the line of sight | Tooth depth on the outline (of 2.064 mm) |
|---|---|
| 85-90° | 2.06 (all of it) |
| 82° | 1.83 |
| 80° | 1.57 |
| 75° | 0.82 |
| 70° | 0.47 |
| 57° (the MetroY try5 photos) | 0.16-0.22 |

This is why the MetroY photos so far measured no pitch or roll ("the thread's flanks were not found"). The screw
pointed at the scanner, and the scanner looks 33-36° down, so the line of sight ran 55-57° from the axis. It is
geometry, not the method: no edge finder can see teeth that are not on the outline. The check now says so, with the
angle. With a camera tilted about 35°, lay the screw **across** the tilt (horizontal in the scanner's picture). Then
the line of sight is square to it, apart from the screw's own tilt.

**2. Perspective limits one photo to part of the thread.** The line of sight is square to the axis only near the
point under the camera. The angle grows by about 1° for every 5 mm along the screw at 280 mm. A lying screw is also
tilted by its bigger head: 4.5° on the 91251A917, which uses up most of the margin. Within about 5° of square, the
teeth are placed. Roots need closer to square, because the next tooth's crest hides them first. Teeth further out
are reported "not measured" with the local angle.

**Square the camera up with the screw:** hold it over the thread's middle, tilted 4-5° from straight down towards
the head (the camera a little towards the tip). On the 91251A917's exact silhouette at 280 mm (48 MP):

| Camera | Teeth placed (of 45) | Groove depths (of 46) |
|---|---|---|
| straight down | 37 (5-72 mm along the thread) | 19 (36-74 mm) |
| tilted 4.5° towards the head | **45** (all) | **30** (13-72 mm) |
| MetroY, 35° across the screw | 38 | 16 |

For every groove, take a second photo moved along the screw. Each photo is read on its own.

**3. Resolution.** A root is measured only if the edge finder can see it: its samples must be on the clean outline,
and the blur model must move the depth by less than a quarter of the tolerance. Otherwise the report says "groove
depth not measurable at X px/mm; needs about Y px/mm". Y is found by repeating the same tests on the golden model
with a camera that has more pixels per mm and the same blur in pixels. On the 1"-8 thread, the depth passed the test
at the MetroY's 7-9 px/mm when the view was square (see the table below).

**4. Light: from behind, or at the lens.** A root is read against the paper seen through the groove. If the part's
own shadow falls there, the groove looks shallow. A light even 5 mm beside the lens made the 1"-8 roots read
0.12-0.18 mm shallow in the synthetic tests. The check measures this: the paper's brightness next to each sample,
against open paper 2.5-5.5 mm further out, where the model shows open paper.

* A groove whose paper is more than 10 % darker is not measured: "the paper seen through the groove is 31 % darker
  than open paper (the part's own shadow)...".
* A tooth whose flanks' paper is more than 6 % darker is not placed.
* Crests are still measured (they stand against open paper).

The light pad has no such shadow. This matches what the real MetroY photos showed:

* With a desk lamp, the diameters came out +1.5 to +4.4 mm (the shadow joins the silhouette).
* With the scanner's own fill light and the lamp off (try5), the left and right cameras agreed to 0.002 mm on the
  thread's major Ø (25.284 / 25.282) and 0.004 mm on the head Ø (37.929 / 37.933).

A shadow would make the silhouette bigger. Both try5 diameters came out smaller than the golden values, so these
photos show no sign of one. But **the ends are not protected**. Where the shadow falls beyond an end of the part, it
joins the silhouette, the edge finder takes it for part, and nothing in the photo tells the two apart. In the
synthetic MetroY photos of the perfect screw:

| Front light | Overall length | Head height | Length under head |
|---|---|---|---|
| at the lens | +0.006 | +0.009 | −0.003 |
| 5 mm beside the lens | **+0.44** | +0.11 | **+0.33** |
| 30 mm beside the lens | **+0.76** | **+1.47** | **−0.70** |

So lengths from front-lit photos need the light at the lens or behind the part, or a gauge of known length in the
same photo to show the shadow's effect.

**5. A flat sheet.** The try5 markers fitted their layout only to 0.6-1.25 px (usually under 0.3 px), with markers up
to 0.53 mm out of place: the paper was wavy. That moves the scale under the part. Tape the sheet flat, or put it on
glass.

### The real MetroY photos (try5)

`try5` is the real 91251A917 on the A4 sheet, taken with the MetroY's left and right cameras (fill light 200, lamp
off, 32 frames averaged). The screw points at the scanner, and the scanner looks 33° down. The check, with
`--metroy-camparam` and `--bar-mm 100.0`:

| | Left | Right |
|---|---|---|
| Sheet: markers, fit to the layout | 14, 1.25 px | 15, 1.13 px |
| Line of sight from the screw's axis | 57° | 55° |
| Tooth depth on the outline (of 2.07 mm) | 0.22 mm | 0.19 mm |
| Thread groove by groove | not measured (the view) | not measured (the view) |
| Thread pitch, minor Ø, lengths | not measured (the view; the head touches the edge of the blank middle) | the same |
| Thread major Ø (25.400) | 25.2863 (−0.114 ± 0.139) close | 25.2749 (−0.125 ± 0.137) close |
| Head Ø, knurl crests (38.100) | 37.9287 (−0.171 ± 0.150) close | 37.9169 (−0.183 ± 0.147) close |

* Left and right agree to 0.011 mm on the major Ø and 0.012 mm on the head, well within U.
* The previous version gave 25.284 / 25.282 and 37.929 / 37.933 (0.002 / 0.004 apart). The difference is one
  change: the screw's tip, the only shoulder in view, is now held where the photo shows it. Before, it could shift,
  and the right camera's fit had the screw 0.1 mm further along its axis. In this 33° oblique view, that changes its
  distance from the camera and so its scale.
* Both diameters come out 0.11-0.18 mm under the golden values, inside U (so "close", not "off"). This is not
  explained yet. Candidates are the wavy paper (the markers fit only to 1.1-1.25 px), the edge of front-lit dark
  metal, or the screw itself. A gauge pin of known diameter lying next to it would tell.

### Accuracy (synthetic)

Measured on 2026-10-01 with `tools/outline_check/thread_validate.py VALDIR --screw GOLDEN.ply`, on the real golden
model (91251A917). Its errors are made by moving its vertices along its own helix (`outline_synthetic.thread_errors`):

* one groove filled 0.10 mm shallower, all the way round its turn;
* one tooth moved 0.05 mm along the axis, all the way round (its pitch to the previous tooth +0.05, to the next one
  −0.05);
* the thread stretched 0.1 % about the head's underside: a lead error growing to about +0.07 mm over the full thread;
* one crest pushed in 0.10 mm over ±20° on one side only.

The helix turn that carries an error starts and ends facing the camera, so both sides of the outline show it. Each
screw lies across the view, with its errors in the stretch the camera sees square-on. The perfect screw is rendered
the same way. The check uses the true camera and a bar of exactly 100 mm.

"Exact" is what the part's exact silhouette gives at the fitted pose: the method without blur, noise or pixels.

| Camera | px/mm | Teeth placed / groove depths (of 45 / 46) | Tooth moved +0.05: pitch errors (side A; B) | Crest −0.10 | Groove −0.10 (exact) | Other teeth and grooves: worst error (within U) | Perfect screw: worst error (within U) |
|---|---|---|---|---|---|---|---|
| iPhone-like 48 MP, backlit, 280 mm | 22.2 | 33 / 14-15 | +0.051 −0.045 ± 0.007; +0.052 −0.046 ± 0.007 | −0.104 ± 0.019 | −0.097 ± 0.005 (−0.095) | 0.006 (100 %) | 0.006 (99 %) |
| MetroY 1600 x 1200, tilted 35°, 270 mm, light at the lens | 7.1 | 37 / 15-16 | +0.049 −0.045 ± 0.017; +0.048 −0.043 ± 0.016 | −0.111 ± 0.073 | −0.106 ± 0.050 (−0.093) | 0.033 (100 %) | 0.026 (100 %) |
| MetroY straight down, 215 mm, light at the lens | 9.1 | 30 / 11-12 | +0.051 −0.044 ± 0.015; +0.048 −0.041 ± 0.013 | −0.116 ± 0.048 | −0.105 ± 0.034 (−0.096) | 0.023 (100 %) | 0.021 (100 %) |
| MetroY tilted 35°, light **5 mm beside** the lens | 7.1 | 0-2 / 0 (shadow) | not measured | −0.110 ± 0.108 | not measured (roots 22-41 % darker) | 0.049 (100 %) | 0.048 (100 %) |
| MetroY straight down, light 5 mm beside the lens | 9.1 | 0 / 0 (shadow) | not measured | −0.105 ± 0.073 | not measured | 0.044 (100 %) | 0.044 (100 %) |
| MetroY tilted 35°, light **30 mm beside** the lens | 7.1 | 0 / 3-4 | not measured | | | 0.11 (53 %) | 0.11 (64 %) |

The progressive lead error of the stretched thread (+0.035 to +0.067 at the last tooth, depending on the side's first
placed tooth) came back within 0.002 at 48 MP and within 0.004-0.015 on the MetroY, each within its U.

What the table shows:

* **48 MP backlit reads every tooth it places to about 0.006 mm:** pitch, lead, crest and groove depth. Its U is
  0.005-0.008 for a pitch, 0.005-0.013 for a groove depth, 0.02 for a crest radius (mostly the assumed 0.15 px edge
  term, which cancels in pitch and depth) and 0.006-0.06 for the lead (growing with the distance from the first
  tooth: the photo's scale).
* **The MetroY's 7-9 px/mm reads pitch and lead to about 0.015 mm (U 0.011-0.019 per pitch) and crests and grooves
  to about 0.03 mm (U 0.024-0.042 for a groove depth, 0.04-0.07 for a crest).** That needs the screw lying across the
  view and a light at the lens. Groove depths then pass the resolution test on the 1"-8 thread, with the shape
  correction doing real work (0.067 → 0.106 for the shallow groove at 7 px/mm).
* **A light beside the lens shades the grooves and the flanks, and the check refuses them** ("the paper seen through
  the groove is 31 % darker than open paper..."). It still reads crests. But the same shadow put the main check's
  overall length +0.44 to +0.54 off, and that it cannot see (see Light above).
* **With a lamp-like light 30 mm off,** the main fit itself is wrong (head height +1.45). What the thread check still
  reports is mostly *close* rather than *ok*, but many of its errors exceed U: no front-lit photo with a visible
  shadow should be trusted.
* **The 12 MP and 48 MP renders of the accuracy table below** (the screw lying at 23° and 80° to the picture, not
  chosen for the thread; 11 px/mm at 12 MP) were read groove by groove too. 22-41 teeth were placed per photo.
  Pitch errors were within 0.011 (U 0.009-0.012). Crests were within 0.011 of the injected +0.03 (U 0.03). Groove
  depths were within 0.019 (U 0.013-0.029). 93-100 % of the values were within their U at 12 MP; at 48 MP, 90 % of
  the depths were (one side read 0.01 shallow throughout, near the perspective limit).
* Each photo took 26-37 s at 48 MP and 20-24 s on the MetroY's 1600 x 1200, including the thread.

The tests run the same method on a procedural screw. On its exact silhouette, a groove filled 0.10 and a tooth
moved 0.05 come back to 0.001 on both sides; a perfect screw gives 0.000. A screw pointing at the camera is refused
with the reason. A 7 px/mm backlit photo reads every tooth's pitch within 0.03.

## Accuracy on synthetic photos

Measured on 2026-09-30 with `tools/outline_check/validate.py VALDIR --screw GOLDEN.ply`.

**Test setup:**

* Renders of an iPhone-like camera: 4032 x 3024 px, 69° horizontal field of view, mild distortion (k1 0.02,
  principal point 12 px off centre), about 280 mm above the sheet (11 px/mm on the part). The 48 MP cases are
  8064 x 6048 at 21.5-22 px/mm.
* The A4 check sheet on a light pad, with a 6 % brightness fall-off and 12 % vignetting. The part is a dark
  silhouette.
* Every pixel is 16 jittered samples (9 at 48 MP), each traced exactly: the paper through the lens, the part by ray
  casting. Then Gaussian blur (0.8 px; 1.2 px at 48 MP), shot and read noise, the sRGB curve, 8 bits and JPEG q95
  with EXIF.
* The camera is calibrated from 10 tilted sheet-only renders: f to 0.0000 % (±0.0008 %), principal point to 0.05 px,
  reprojection 0.018 px.

The error is the photo's value minus the truth; the injected error is the truth minus the golden value.

### The real golden model: McMaster 91251A917, 1"-8 socket head screw

The golden model is the user's own mesh: 251,166 triangles, a modelled helical thread, a knurled Ø38.1 head, chamfered
ends and a hex socket. The check finds, from the mesh alone:

* the thread: Ø25.400 from −52.07 to 28.48, right hand, pitch 3.1750, minor Ø21.266;
* the knurled head: Ø38.100, knurl ripple 0.009 mm;
* three shoulders: the head top, the head underside and the thread end.

The screw lies on its side, resting on its head's rim and its thread's last crest (4.43-4.57°, depending on roll). It
is photographed perfect, and with errors made by moving its vertices:

* the head's end region +0.05 along the axis;
* the thread end 0.10 shorter;
* everything below the head radially +0.03 (thread major and minor Ø +0.06).

| Case | Overall length 107.950 | Head height 25.400 | Length under head 82.550 | Thread major Ø 25.400 | Thread pitch 3.175 | Head Ø (knurl) 38.100 | Thread minor Ø 21.266 |
|---|---|---|---|---|---|---|---|
| injected (errors) | −0.050 | +0.050 | −0.100 | +0.060 | 0 | 0 | +0.060 |
| **perfect**, straight: error / U | −0.0005 / 0.106 **close** | +0.0014 / 0.077 ok | −0.0019 / 0.069 ok | +0.0004 / 0.068 ok | +0.0000 / 0.003 ok | +0.0009 / 0.069 ok | not measured |
| errors, straight | +0.0014 / 0.106 close | +0.0048 / 0.077 close | −0.0034 / 0.068 close | +0.0013 / 0.070 close | +0.0000 / 0.003 ok | +0.0009 / 0.068 ok | not measured |
| errors, 3 photos | +0.0014 / 0.097 close | +0.0032 / 0.072 close | −0.0018 / 0.064 close | +0.0003 / 0.069 close | −0.0001 / 0.002 ok | +0.0017 / 0.067 ok | not measured |
| errors, tilted 20° | +0.0002 / 0.109 close | +0.0045 / 0.067 close | −0.0044 / 0.081 close | +0.0007 / 0.070 close | −0.0001 / 0.003 ok | +0.0072 / 0.074 ok | not measured |
| errors, **printed 76.2 %** (bar 76.2, height 91.44) | +0.0008 / 0.091 close | +0.0065 / 0.072 close | −0.0056 / 0.062 close | +0.0004 / 0.068 close | +0.0000 / 0.002 ok | +0.0036 / 0.073 ok | not measured |
| errors, **48 MP** | +0.0004 / 0.093 close | +0.0027 / 0.063 close | −0.0023 / 0.064 close | +0.0009 / 0.049 close | +0.0000 / 0.003 ok | −0.0007 / 0.052 ok | **+0.0048 / 0.049 close** |

What the table shows:

* **Every error is within 0.0072 mm; most are within 0.003.**
* The verdicts are what ISO 14253-1 gives for these differences: +0.05 or −0.10 at ±0.1 mm are too close to call,
  and the unchanged head diameter and pitch are ok.
* The perfect screw's overall length comes out *close* rather than ok: U = 0.106 mm at 108 mm (see below).
* The thread's roots are not measurable at 12 MP. The gaps between its teeth are 9-14 px wide there, too narrow to
  see into with a clean edge window. At 48 MP the minor diameter is measured.
* The printed-at-76.2 % sheet works because its bar and height were measured. The blank middle is then 107 x 137 mm,
  so the screw lay along the page.

### The procedural parts: a bolt and an L-plate

The bolt is head Ø36 x 25.25 with a Ø24 shank, 107.1 overall. It lies on its head's rim and its shank's end (4.19°),
and is measured as a round part. The plate is 70 x 40, narrowed to 28 beyond x = 45, 12 thick, with a Ø14 through
hole. Their golden meshes are coarser (180 / 256 segments) than their truth (720 / 1024).

| Case | Value (golden) | Injected | Recovered | Error | U (k=2) | Verdict |
|---|---|---|---|---|---|---|
| straight (0-3°) | Overall length 107.10 | −0.050 | −0.0503 | −0.0003 | 0.097 | close |
| | Head height 25.25 | +0.050 | +0.0491 | −0.0009 | 0.072 | close |
| | Length under head 81.85 | −0.100 | −0.0994 | +0.0006 | 0.068 | close |
| | Head Ø36 | 0 | +0.0007 | +0.0007 | 0.065 | ok |
| | Shank Ø24 | +0.030 | +0.0303 | +0.0003 | 0.066 | ok |
| tilted 20° | all five | | | ≤ 0.0012 | 0.066-0.091 | as straight |
| 3 photos combined | all five | | | ≤ 0.0006 | 0.065-0.092 | as straight |
| printed 1 % (x), 0.5 % (y); bar and height measured | all five | | | ≤ 0.0019 | 0.058-0.081 | as straight |
| the same, bar **not** measured | Overall length | −0.050 | −0.7937 | −0.74 | 1.26 | close |
| calibration f **+1 %**, straight | Overall / head / under / Ø36 / Ø24 | | | +0.117 / +0.067 / +0.050 / +0.026 / +0.013 | 0.065-0.097 | close / ok |
| calibration f **+1 %**, tilted 20° | all five | | | ≤ 0.0012 | 0.066-0.091 | as tilted 20° |
| perfect bolt | all five | 0 | | ≤ 0.0015 | 0.067-0.095 | **ok** (all) |
| over-exposed (paper clipped), marker correction | all five | | | ≤ 0.0095 | 0.065-0.100 | ok / close |
| the same, **no** correction | Ø36 / Ø24 | 0 / +0.03 | −0.0682 / −0.0421 | −0.068 / −0.072 | 0.096-0.097 | close |
| plate: length +0.03, step_x −0.06, hole +0.04, thickness +0.05 | 70 / 40 / 28 / 45 | | | +0.0146 / +0.0077 / +0.0054 / +0.0094 | 0.069-0.084 | ok / close |
| | steps 25 / 12 | +0.09 / 0 | +0.0951 / +0.0022 | +0.005 / +0.002 | 0.028 / 0.013 | close / ok |
| | Ø14 hole | +0.040 | +0.0381 | −0.0019 | 0.076 | close |
| | thickness 12 | +0.050 | not measured (not on the outline) | | | |
| perfect plate | 7 values | 0 | | ≤ 0.0019 | 0.013-0.084 | **ok** (all) |
| plate by its outline alone (no golden faces; test) | the same 7 values | | | ≤ 0.015 | | as with faces |
| 48 MP, straight | all five | | | ≤ 0.0007 | 0.049-0.098 | as straight |

The tests (`tests/test_outline_check.py`) also run a procedural knurled, threaded screw (Ø30 head with 40 knurl
grooves, Ø20 thread of 3 mm pitch) at only 7 px/mm. The lengths come out within 0.016, the major Ø within 0.01, the
pitch within 0.0006 and the knurled head within 0.021 (its knurl ripple is 0.022). Each error is within its U.

Budgets for the real screw's straight photo (standard uncertainties, mm):

| Value | statistical | camera & scale | placement | toner (assumed) | edge (assumed) | knurl roll | U (k=2) |
|---|---|---|---|---|---|---|---|
| Overall length 107.95 | 0.001 | 0.038 | 0.026 | 0.002 | 0.027 | | 0.106 |
| Head height 25.4 | 0.001 | 0.014 | 0.016 | 0.017 | 0.027 | | 0.077 |
| Length under head 82.55 | 0.001 | 0.024 | 0.014 | 0.019 | 0 | | 0.068 |
| Thread major Ø 25.4 | 0.001 | 0.007 | 0.005 | 0.017 | 0.029 | | 0.070 |
| Thread pitch 3.175 | 0.000 | 0.001 | 0.001 | 0.001 | 0 | | 0.003 |
| Head Ø (knurl) 38.1 | 0.001 | 0.009 | 0.009 | 0.014 | 0.027 | 0.007 | 0.068 |

**What it means:**

* **The method itself is exact to a few thousandths of a millimetre on these renders.** That covers the fit, the
  resting pose and roll, the outline and the edge finder with the model's own edge. Injected errors come back to
  ≤ 0.007 mm on the real screw and ≤ 0.002 mm on the bolt. The larger errors in the tables are all things the photo
  cannot see, and U includes them:
  * the plate's unseen thickness;
  * clipped paper;
  * a focal length deliberately 1 % off.
* **U, not the error, sets what the photo can decide.** The caliper reading of the bar (±0.05 mm) alone is
  ±0.031 mm (1σ) on 108 mm. The assumed focus change, tilt, paper flatness and edge terms add the rest. At 12 MP, U
  is about 0.1 mm for 100 mm lengths and 0.07 mm for 25-45 mm widths and diameters; at 48 MP diameters come down to
  0.05 mm. So at ±0.1 mm:
  * a 108 mm length of a perfect screw is *close*, not ok;
  * an error of 0.05 mm is correctly too close to call;
  * a perfect bolt of 107.1 mm is just ok (U 0.095).

  A better caliper reading of the bar (`--bar-u 0.02` for a good caliper and several readings), a locked focus, or
  48 MP photos bring U for long lengths under 0.1 mm.
* The biggest terms are the assumed ones: the edge (0.15 px), the toner spread (±0.02 mm), the focus (±0.3 %), the
  tilt and flatness, and the caliper (±0.05 mm). The next step is to measure them on real photos, with a gauge pin or
  gauge block of known size lying on the sheet. That would replace the assumptions with numbers, and probably shrink
  U.
* **Measure the bar.** An unmeasured print puts every length off by the print's scale error. The check then assumes
  ±1 %, and U (1.26 mm on 107 mm) makes every verdict "close". It never says "ok" wrongly.

### Speed

With the 251,166-triangle golden screw on a 28-thread PC:

| Step | Time |
|---|---|
| Golden model preparation (profile, faces, resting poses), once per check | 4 s |
| Per 12 MP photo, including the Monte Carlo (and the thread groove by groove) | 17-29 s |
| Per 48 MP photo | 26-37 s |
| Per MetroY photo (1600 x 1200) | 20-24 s |

## Limits

* **Only the outline is measured.** Nothing inside the silhouette can be seen in a backlit photo: blind holes,
  recess and pocket depths, flatness, faces seen face-on, anything hidden behind the part. The report lists these as
  "Not measured from this photo", with the reason.
  * A through hole seen along its axis is measured; one seen from the side is not.
  * A thread's pitch diameter (its flanks) is fitted but not judged. Each tooth's flank shift is in the report
    (`flank_radial`), but the silhouette's flank angle is not the thread's flank angle (helix), and thread gauging
    needs more care than this gives it. Flank angles are not reported for the same reason.
* **A thread's minor diameter (one value for the whole thread) needs about 20 px/mm (48 MP at 28 cm)** in the main
  fit. Groove by groove, roots are read with a smaller window and need less (7-9 px/mm sufficed on the 1"-8 thread),
  but only where the view is square to the axis and nothing shades the groove. The thread's flanks need its pitch to
  span at least ~14 px: the search window is held to a third of the pitch, and the check warns when that falls under
  4 px.
* **Threads groove by groove** need the screw lying across the view, a view within about 5° of square to it at the
  tooth (perspective: one photo reads part of a long thread), and light from behind or at the lens (see
  [Threads groove by groove](#threads-groove-by-groove)). Every tooth and groove it cannot read is listed with the
  reason.
* **Ends seen through fillets.** When an end face meets its side in a fillet (the 91251A917's head top), the outline
  there is the fillet's edge, just short of the face. The end is measured assuming its whole end region moves
  together. A fillet of another shape than the golden model's shows up as an end error.
* **Knurls:** the head diameter compares crests with crests at the fitted roll. The roll comes from the thread, so a
  real part whose thread starts elsewhere than the golden model's gets the knurl's ripple as extra uncertainty
  (`knurl_roll`).
* Round parts other than lying on their side (a screw standing on its head) show only circles: diameters, no
  lengths.
* Other parts: flat and round faces (golden.find_faces) and the outline's own straight runs and arcs (at least 150°
  for a diameter). Cones, fillets and spheres shape the outline but give no value.
* **The part must rest stably on the sheet**, in one of the golden model's resting poses, and fit fully inside the
  blank middle (140 x 180 mm at 100 %). A tilt off the resting face is not fitted: ±0.05° is allowed, and U covers
  it.
* Tall parts are worse. The outline's height above the paper multiplies focal-length and focus errors (h/Z). Parts
  much taller than 40 mm, or a camera much closer than 25 cm, also defocus the outline relative to the markers.
* **Front-lit metal is much worse than backlit.** The method needs a dark silhouette on bright paper.
  * With front light, the part's own shading and reflections, and shadows on the paper, move the edges. A desk lamp
    put real MetroY diameters 1.5-4.4 mm off. In synthetic MetroY photos, a light only 5 mm beside the lens put the
    screw's overall length +0.44 mm off (the shadow beyond its end joins the silhouette; this is not detected), and
    shaded the thread's grooves (detected: those grooves are not measured). The MetroY's own fill light with the
    lamp off gave consistent left / right diameters.
  * Shiny round parts (a black-oxide screw less so) mirror the light pad at grazing angles next to their outline.
    That brightens the part right at its edge and can shrink the silhouette. Dull such parts (matting spray) until
    this has been measured on real parts.
  * Transparent parts do not work.
* **Real iPhone photos (not yet tested):**
  * HEIC processing (local tone mapping, sharpening, noise reduction) is not a simple sRGB curve. The marker
    correction removes whatever shifts the markers' edges and the part's edges alike, but not the rest; that is the
    *assumed* 0.15 px edge term.
  * Optical image stabilisation moves the principal point between shots. The per-photo sheet pose absorbs this as a
    small rotation, leaving only a second-order distortion-centre error.
  * Focus breathing is the focal-length problem above.
  * Over-exposure shrinks the silhouette (clipped paper). The markers shrink the same way and the correction removed
    87 % of it here (0.07 → 0.0095 mm), but set the exposure so the paper is light grey. The check warns when it is
    clipped.
* The phone's calibration is only valid for that phone, lens (1x), resolution and focus distance. A photo taken at
  another resolution is scaled with a warning; a crop is refused.
