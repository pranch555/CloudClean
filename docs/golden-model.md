# Golden model check

The golden model is the part as it should be: its CAD file (STEP, IGES) or a trusted mesh (STL, OBJ, PLY). The check
lines your scan up with it and tells you three things:

1. **What was not scanned properly.** These are named areas of the part, such as "Bottom face" or "Ø6.00 hole",
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

* **The verdict** at the top:
  * *matches*: everything was measured and is within tolerance;
  * *does not match*: something is off;
  * *scan more*: everything scanned so far matches, but some areas or sizes could not be checked.
* **Problems view.** The golden model is coloured by what the check found:

  | Colour | Meaning |
  |---|---|
  | sage green | matches |
  | violet | not scanned |
  | amber | too few points: the area was only seen at a glancing angle |
  | orange | rough: the points scatter more than the tolerance allows, often a shiny or dark surface |
  | red | off, outside: the scan has more material here |
  | blue | off, inside: the scan has less material here (for a hole, the hole is bigger) |

  Sharp edges are never called rough or off. Every scanner rounds edges a little, and the summary says by how much.
* **Scan again.** Each area comes with why it was flagged and what to do. **Show me** turns the view to it.
* **Different from the golden model.** These areas are off by more than the tolerance. Either the part really
  differs there (check it with a caliper or gauge), or that area came from a separate scan that was merged slightly
  off (redo the line-up in the Align step).
* **Measurements:** golden value, scanned value and difference. The result column reads:

  | Result | Meaning |
  |---|---|
  | Matches | within ±tolerance, even allowing for how precisely the scan pins the value down |
  | Off | outside ±tolerance by more than that margin |
  | Too close to call | within that margin of the limit: measure it by hand |
  | Not measured | not enough of the face was scanned; **Show the area** points to what to scan |

* **Deviation on the golden** and **Deviation of the scan** are the colour maps of the distances, as in the former
  CAD compare. The **Report** is a printable page of all of the above.

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
`compare`, scalar `deviation`).

## API

```
POST  /api/golden-check {scan_id, golden_id?, tolerance?, align?, remember?}   -> job "golden_check"
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
