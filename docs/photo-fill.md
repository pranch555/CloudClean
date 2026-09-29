# Fill from photos

When a part cannot be scanned again, CloudClean can fill the areas the scan missed using photos of the part. The
result is a new model, "<scan> + photo fill", where every point is marked as either scanned or filled from photos.

Code: `cloudclean/photo_fill.py` (the fill), `cloudclean/web/jobs_photo_fill.py` (the job) and
`cloudclean/web/routes_photo_fill.py` (`POST /api/fill-from-photos {asset_id, photo_ids?, gap?}`). Tests are in
`tests/test_photo_fill.py`.

## Using it

* **Measure → Golden model**: after a check that found areas not scanned, the **Scan again** list offers **Fill from
  photos**.
* **The assistant**: ask it to *"fill the missing parts of this scan from the photos"*. It uses the tool
  `fill_from_photos`.
* **Photos**: it uses every photo in the project, unless you pick some. You need at least 3 photos; 12 or more all
  the way round, with overlap, work best, the same as for Colour from photos.

## What it does

1. **Places the photos.** Photos → 3D (COLMAP) places the photos and makes a dense photo model.
2. **Lines the model up with the scan.** This is the same line-up as Colour from photos, in its mode for photos
   that show more of the part than the scan.
3. **Fits it again.** The photo points are fitted onto the scan once more, correcting size, turn and shift, where
   both show the part. The dense photo points can be a few percent off their own cameras, and this fit corrects
   that.
4. **Adds only the photo points in gaps.** A photo point is added only where the scan has nothing within the gap
   distance. By default that is the larger of twice the photo surface's error, four scan spacings and 0.5 mm.
   Photos cannot fill gaps narrower than their own error.
5. **Leaves out what is not the part:**
   * **the table or scale sheet under it:** a flat surface much bigger than the part, with the whole part on one
     side of it, together with everything below it;
   * points outside the scan's box, which is grown by 5 %;
   * small stray groups;
   * groups that do not reach the scan.

The report gives:

* the points added and the filled area;
* **how far the photo surface sat from the scan where both have the part** (median and 90 %). This is how accurate
  the filled areas are;
* what was left out, and warnings.

## Accuracy: read this

Photo surfaces are far less accurate than the scanner: about **1–2 mm**, against a few hundredths. Filled areas
make the model complete, for example so it can be meshed watertight or be visually right. **Never measure on
them.** The golden model check leaves points filled from photos out, so a fill can never make a part pass (or
fail); those areas still show as not scanned.

If the photos cannot be placed exactly, the job stops and says so. If the line-up is not fully certain, for
example when the part looks almost the same turned another way, the result says to check that the filled areas sit
where they belong.
