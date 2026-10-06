# Golden model check

The golden model is the part as it should be: its CAD file (STEP, IGES) or a trusted mesh (STL, OBJ, PLY). The check
lines your scan up with it and tells you three things:

1. **What was not scanned properly.** These are named areas of the part, such as "Edge of the narrow end" or "Ø6.00 hole",
   each with its size and why it was flagged.
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
* **Sizes**: golden value → scanned value and the difference, sizes that are off first:

  | Result | Meaning |
  |---|---|
  | Matches | within ±tolerance, even allowing for how precisely the scan pins the value down |
  | Off | outside ±tolerance by more than that margin |
  | Too close to call | within that margin of the limit: measure it by hand |
  | Not measured | not enough of the face was scanned; **Show the area** points to what to scan |

* **Notes**: the scale of the whole scan against the golden model, line-up doubts and other remarks. The
  **Printable report** is a page of all of the above.

Checks made before October 2026 name their areas by coordinates; they show a **Run the check again** button.

## Names

Areas and sizes are named after the part, not after coordinates (`cloudclean/golden_words.py`), because CAD files
use any axis as up and the 3D view turns any way:

* Turned and long parts have a main axis: the axis of their round faces, else their longest side when it is clearly
  the longest. Its ends are *the wide end* and *the narrow end* when their cross-sections differ by 12 % or more,
  else they are named from the 3D view's default front view (*top end*, *left end*; the view's up axis is sent
  with the check).
* Faces: *wide end face*, *step 25.40 mm from the wide end*, *floor of the recess in the wide end* (a floor with
  walls all round, found by casting rays sideways), *wall of the recess…*, *flat side…*, *sloped face…*,
  *Ø6.00 centre hole*, *Ø6.00 cross hole…*, *Ø20.00 round face*. Two faces with the same name get *(1 of 3)*.
* Areas on no single face: *edge of the narrow end* (a chamfer or rounded edge), *inner corner at the step…* (a ray
  along the surface normal hits that step), *side surface 40 mm from the narrow end*, *curved area…*.
* Sizes: *Overall length (end to end)*, *Overall width (left to right)*, *Depth of the recess in the wide end*,
  *Width across the recess… (1 of 3)*, *Thickness below the recess…*.
* Parts with no main axis use the default front view: *top face*, *floor of the pocket in the top*, *step facing
  up…*.

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
`golden_deviation`, `check_status` and `check_region`), and the scan lined up with the golden model (operation
`compare`, scalar `deviation`). Before colouring, long edges of the golden mesh are split (`refine_long_edges`, up
to 520 000 triangles): CAD files draw a flat face as a few huge triangles, and colours painted on their corners,
which sit on never-judged edges, used to hide a whole off face.

Each area carries `view` (`target`, `from`, `radius`: a camera looking along the area's normal, or the nearest
direction no other part of the model blocks, framed with some of the part around it) and `pin` (a point on the
area for its numbered pin). The report has `version` 2, `match_pct` and `part` (`axis`, `ends`, `up_axis`).

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
