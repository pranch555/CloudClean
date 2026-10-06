# Golden model check

The golden model is the part as it should be: its CAD file (STEP, IGES) or a trusted mesh (STL, OBJ, PLY). The check
lines your scan up with it and tells you three things:

1. **What was not scanned properly.** These are named areas of the part, such as "Corner under the head" or "Ø6.00
   hole through the middle", each with a sentence saying where it is, its size and why it was flagged.
2. **What to scan again, and how.** For example, "Turn the part over, scan it again and merge the two scans" or
   "Point the scanner straight into the hole".
3. **Whether the measurements match.** Every flat-face distance, diameter and hole position of the golden model is
   measured again on the scan and set against the golden value.

Code: `cloudclean/golden.py` (the check), `cloudclean/web/jobs_golden.py` (the job), `cloudclean/web/routes_golden.py`
(the API and the printable report). The screen is `frontend/src/steps/measure/GoldenCheck.tsx`. Tests are in
`tests/test_golden.py`, with test parts in `tests/golden_synthetic.py`.

## Using it

1. Bring in the golden model with **Add** in Measure → Golden model, or by dropping the file anywhere. STEP and IGES
   are turned into a fine mesh on import.
2. In **Measure → Golden model**, pick the golden model and the scan, and set the tolerance (default ±0.1 mm).
3. Press **Check against the golden model**. The golden model you used becomes the project's golden model (★ in
   the list), so later scans of the project are checked against it by default.

You can also ask the assistant: *"check this scan against the golden model"*. It uses the tool
`check_against_golden`.

## Reading the result

* **The verdict** at the top, with a ring showing how much of the golden surface matches:
  * *Matches*: everything was measured and is within tolerance;
  * *Mostly matches*: something is off, but 95 % or more of the surface is within tolerance (amber, not red:
    the part is close, a few things need a look);
  * *Does not match*: something is off and less than 95 % of the surface matches;
  * *Scan more*: everything scanned so far matches, but some areas or sizes could not be checked.

  Under it: how much of the part was scanned, how many sizes match, and how many areas to look at (each jumps to
  its section).
* **The surface**: one bar and a list of how much of the golden surface matches, was not scanned, had too few points,
  is rough, or has less / more material than designed. **Colour the model by** switches the 3D view between:

  | View | Shows |
  |---|---|
  | What was found | the golden model in the check's colours (below) |
  | Distance | the golden model coloured by how far the scan sits from it (blue less material, red more) |
  | Scan points | every scan point coloured by its distance to the golden model |

  | Colour | Meaning |
  |---|---|
  | sage green | matches |
  | violet | not scanned |
  | amber | too few points: the area was only seen at a glancing angle |
  | orange | rough: the points scatter more than the tolerance allows, often a shiny or dark surface |
  | red | more material: the scan sits outside the golden surface |
  | blue | less material: the scan sits inside it (for a hole, the hole is bigger) |

  Sharp edges are never called rough or off. Every scanner rounds edges a little, and the notes say by how much.
* **Areas to look at**, numbered like the pins on the 3D view:
  * *Different from the golden model* (first: these decide the verdict): how far off, with a bar against the
    tolerance. Either the part really differs there (measure it), or the area came from a separate scan that was
    merged slightly off (redo the line-up in the Align step). A recess floor that reads deeper is often real: drilled
    floors are cone-shaped where the CAD draws them flat.
  * *Scan these again*: not scanned, too few points or rough, each with how to scan it.

  **Show me** turns the view to the area from a side nothing blocks, keeps only that area in colour and greys out
  the rest; **Whole part** goes back. Clicking a pin does the same. Pins of areas on the far side of the part fade.
* **Sizes**: golden value → scanned value and the difference, under plain headings (*Overall size*, *Head*, *Hex
  socket*, *Shaft*, *Holes*, *Other sizes*...), each with one sentence saying what it is, from where to where ("From
  the top of the head down to the bottom of the hex socket."), and the dimension line on the 3D view and on the
  drawing (a cut through the golden model along its axis). Several widths of one feature (the three across a hex
  socket, the grooves of a knurled grip) are one row. A size gets a caveat when one of its faces carries a listed
  area or was only partly scanned ("The scan finds the bottom of the hex socket 2.41 mm deeper than the golden model
  (area 1)."), and when the sizes that are off share a cause, one short story says so above the list ("All 3 sizes
  that are off are measured from the bottom of the hex socket. ..."). Results:

  | Result | Meaning |
  |---|---|
  | Matches | within ±tolerance, even allowing for how precisely the scan pins the value down |
  | Off | outside ±tolerance by more than that margin |
  | Too close to call | within that margin of the limit: measure it by hand |
  | Not measured | not enough of the face was scanned; **Show the area** points to what to scan |

* **Notes**: the scale of the whole scan against the golden model, line-up doubts and other remarks. The
  **Printable report** is a page of all of the above.

Checks made before October 2026 name their areas by coordinates; they show a **Run the check again** button.
The printable report lists the sizes under their headings, with what each one is, its caveat and the sizes story.

## Names

Areas and sizes are named after the part, the way a person would point at them, not after coordinates
(`cloudclean/golden_words.py`): CAD files use any axis as up and the 3D view turns any way. The check first
recognises what the part is built from, conservatively; when it is not sure it uses the plainest true words.

* **The part's kind** (`part.kind`, with one sentence in `part.summary`, e.g. "A round part with a knurled head and a
  threaded shaft; a hex socket in the head."):
  * *Head and shaft* (`head_shaft`: screws, bolts, pins, rivets): the part has a main axis, and a section at one end,
    at most 45 % of the length, steps down abruptly to a section at least 15 % narrower (its outer radius along the
    axis comes from rays cast from outside); no bore runs along the axis (that would be a tube). Its faces are *the
    top of the head*, *the underside of the head* (the flat step where head and shaft meet) and *the tip of the
    shaft*; *side of the head*, *side of the shaft*, *corner under the head*.
  * *Axial* (`axial`): other turned and long parts. The main axis is the axis of their round faces, else their
    longest side when it is clearly the longest. Its ends are *the wide end* and *the narrow end* when their
    cross-sections differ by 12 % or more, else they are named from the 3D view's default front view (*top end*,
    *left end*; the view's up axis is sent with the check). Faces: *flat face at the wide end*, *step*, *round side at
    the narrow end*, *round outside*; with one step the two sections are *the wide part* and *the narrow part*.
  * *Block* (`block`): no main axis; named from the default front view: *top face*, *bottom of the pocket in the
    top*, *front wall of the pocket in the top*, *step facing up*.
* **Features**:
  * a recess in the middle of an end whose walls are six flats 60° apart at one distance from a common centre is a
    *hex socket* (four at 90°: a *square socket*); a round blind hole is named by its size (*Ø6.00 hole in the
    head*); otherwise *recess in the wide end*. Its floor is *the bottom of the hex socket*;
  * flat sides all round a section at one distance from the axis are one feature: six are a *hex head* (or *hex*),
    four a *square*, eight or more the *flat sides* of a polygon or, when they cover only part of the way round, the
    grooves (or ridges) of a *knurled grip* (20 or more; fewer are just *grooves round the head*: they could be
    splines);
  * threads are found from the outline (`outline_check.turned_profile`: one helix of one pitch). A part is only
    called threaded when one was found;
  * holes: *Ø10.00 hole through the middle*, *Ø6.00 hole in the narrow end*, *Ø3.00 cross hole through the shaft*,
    *Ø4.00 hole in the top* (blocks).
* **Areas** (`name`, at most about seven words, and `where`, one sentence that lets anyone find it): *Bottom of the
  hex socket* / "Inside the head: the flat bottom of the hex socket (the six-sided hole a hex key fits in).",
  *Middle of the bottom of the hex socket*, *Rim around the tip of the shaft* (an edge near an end), *Side of the
  head, near the top*, *Thread on the shaft, near the tip*, *Corner under the head* (on the shaft just under the
  head, on the underside next to the shaft, or an area whose normal runs into the underside), *Inside corner by the
  step*, *Part of the round side at the wide end*. Areas are numbered (`number`) like the list and the pins.
* **Sizes** (`name`, `what`, `group`, `more` / `less`): *Overall length*, *Head diameter (left to right)*, *Head
  height*, *Length under the head*, *Hex socket depth*, *Hex socket, across flats* (a series of three), *Socket
  bottom to underside of head*, *Socket bottom to tip*, *Shaft diameter*, *Knurled grip, across the grooves*,
  *Ø10.00 hole through the middle: diameter*, *Ø10.00 hole through the middle: where it sits*, *Length of the
  narrow part*, *Depth of the pocket in the top*, *Front wall thickness*. Anything else is named from one face to
  the other: *Top of head to step on the shaft*. Two sizes with the same name that are not a series get *(1 of 2)*.

## How it works

1. **Line-up.** This is the scan-vs-CAD inspection (`compare.compare_to_reference`): a global search, then a fit on
   the exact golden surface. The scan is only moved and turned, never rescaled; a scale difference is reported, not
   corrected. If another pose fits almost as well, the check tests whether that pose maps the golden model onto
   itself. A round part turned about its axis does, so the choice does not matter and no warning is shown.
2. **Spots.** The golden surface is covered with spots about 8 point spacings apart. Each scan point lands on its
   closest golden point and counts for the nearest spot facing the same way. Each spot is then judged from its
   points and its neighbours' points:
   * point count against the usual density: under 12 % means not scanned, under 35 % means too few points;
   * median distance: beyond the tolerance means off;
   * scatter: more than max(tolerance, 2.5 × the scan's usual scatter) means rough.

   Neighbouring spots with the same problem become an area. Areas smaller than about 0.05 % of the surface are
   coloured but not listed.
3. **Faces.** The golden mesh is split into its faces:
   * **Flat faces** are triangles joined across edges bent less than 2°, flat within a few microns. They must be
     bounded by sharp edges or wide in both directions, so the long thin facets of a tessellated round face do not
     count.
   * **Round faces** are smoothly joined triangles whose normals all lie across one axis, on a circle around it.
     Holes point inward and shafts outward. Pieces of the same cylinder are joined. Arcs under 150° (fillets) are
     left out.
4. **Measurements.** Each golden face is measured on the scan points that landed on it, away from its edges:
   * **Overall size** along X, Y and Z: how far the scan's surface sits beyond the golden model's extreme surfaces.
     Both ends must be scanned.
   * **Thickness / gap:** the nearest opposite flat face that overlaps it.
   * **Step:** between flat faces facing the same way at different levels.
   * **Diameter and position:** a robust circle fit across the golden axis. At least 120° of the circle must be
     scanned.

   Each value gets an uncertainty (twice the standard error; noise is counted as at most 400 independent points
   per face, because scanner noise is correlated). The verdict follows ISO 14253-1: ok when |difference| + U ≤
   tolerance, off when |difference| − U > tolerance, otherwise too close to call.

The check makes two results: the golden model coloured by the check (operation `golden_check`, with scalars
`golden_deviation`, `check_status`, `check_region` and `check_face`: the golden face, flat or round, each vertex
lies on, -1 for none; the ids are those of the sizes' `faces`), and the scan lined up with the golden model
(operation `compare`, scalar `deviation`). Before colouring, long edges of the golden mesh are split (`refine_long_edges`, up
to 520 000 triangles): CAD files draw a flat face as a few huge triangles, and colours painted on their corners,
which sit on never-judged edges, used to hide a whole off face.

Each area carries `view` (`target`, `from`, `radius`: a camera looking along the area's normal, or the nearest
direction no other part of the model blocks, framed with some of the part around it) and `pin` (a point on the
area for its numbered pin).

Report version 3 (`version` 3; every field of version 2 is kept, with plainer text) adds:

* per area: `where` (one sentence) and `number` (the number the list and the pins show: areas that differ, the
  biggest |difference| x area first, then the areas to scan again);
* per size: `what`, `more` / `less` (comparative words for a scan above / below the golden value, not for
  positions), `group` (its heading), `series`, `series_index`, `series_size` (several widths of one feature: one
  row), `ends` (the dimension line, golden frame, two points on the measured surfaces: for flat faces a point on one
  face over the other and its projection, or their closest points when they do not overlap; for overall sizes the
  extreme surfaces where the part reaches both; for diameters two opposite points at the middle; in the drawing's
  plane when the surfaces reach it; null for positions), `faces` (for overall sizes: the extreme flat faces, else empty), `regions` (the listed areas on its
  faces) and `caveat` (a sentence or null);
* `sizes_story`: null, or one to three sentences when all the sizes that are off (or too close to call) are
  measured from one face; the cone note only when that face is a recess floor that reads deeper;
* `part.summary` and `part.kind` (`head_shaft`, `axial` or `block`);
* `section`, the drawing: a cut through the middle of the golden model along its main axis (`u` = the axis, `v` =
  the 3D view's up, or its right when the axis is up), or square to the view's depth for parts with no main axis
  (`u` = right, `v` = up). `loops` are the closed outlines of the cut, `[x0, y0, x1, y1, ...]` with x = (p - origin)
  · u and y = (p - origin) · v (holes are separate loops), simplified to 0.0015 x the diagonal and rounded to
  0.01 mm; `bounds`; `faces` maps each measured face id to its traces in the cut (`[x1, y1, x2, y2]` segments, at
  most 12). Under 60 KB; null when the golden mesh is not closed.

## API

```
POST  /api/golden-check {scan_id, golden_id?, tolerance?, align?, remember?, up_axis?}   -> job "golden_check"
      golden_id defaults to the project's golden model; remember makes golden_id the project's golden model
PATCH /api/projects/{id} {golden_asset_id}                                    (a mesh of that project, or null)
GET   /api/assets/{check_id}/golden-report                                    printable page
```

The job's output is `{check_id, compare_id, verdict, headline, scanned_pct, rescan, measurements_off,
measurements_ok, not_measured}`. The full report is on the check asset (`GET /api/assets/{check_id}`):
`verdict, headline, summary, rescan, regions[], measurements[], surface, counts, alignment, deviation, warnings`.

## Limits

* The golden model must be a mesh. For a point cloud, make a mesh of it first (Mesh step).
* Only flat and round faces are measured. Cones (chamfers, countersinks), spheres, fillets and threads are shown in
  the colours but not measured; Measure → Thread checks threads.
* Positions depend on the best-fit line-up. Sizes between opposite faces and diameters do not.
* A symmetric part lined up with a missing side can be matched either way round. The check says so when it cannot
  tell; look at the colours before trusting the numbers.
