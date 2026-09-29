# "The measurements are a little off": scanner or software?

Investigation of 2026-09-24 on the bolt scans in the Spark workspace (read only), plus a ground-truth bench on
synthetic parts of exactly known size. Everything is reproducible with `tools/accuracy/` (results in
`tools/accuracy/results/`). All sizes in mm; 1 µm = 0.001 mm; 100 ppm = 0.01 mm per 100 mm.

## Verdict in plain words

**The processing steps (clean, mesh) do not change the bolt's size.** Cleaning moved no point at all, and the
merge placed scan 06 by a clean rigid move with no scaling. The mesh follows the cleaned cloud to within a few
micrometres on average. This is the high-confidence part of the answer.

What makes numbers look off, from most to least likely:

1. **The big X / Y / Z numbers in the viewer are not the part's dimensions** (software, high confidence). They
   are the axis-aligned box in the scanner's coordinate system. The bolt lay tilted, so the merged bolt shows
   **94.6 × 40.0 × 95.5**, though it is about **107.1–107.4 long, 37.8 across the head flats and 25.06 across the
   thread crests**. The small "fitted" box is off too: its axis is tilted 1.6–3.4° from the bolt. That makes it
   read 0.5–1.0 mm too long (107.9–108.3) and 44.4 wide on the single scans, 6.6 mm too wide.
2. **The merge combined two scans that disagree, in a pose that is ambiguous** (scan strategy plus software,
   high confidence on the effect, medium on its size). The two scans' surfaces are 0.08 mm apart (median) where
   they overlap. On the head's dimples the gap reaches 0.5 mm. The 12-sided head and the thread let scan 06 fit
   almost equally well one head flat (30°) round. Turning one flat also slides the thread one-twelfth of a pitch
   (0.26 mm) along the axis. The merge the user approved sits one flat away from the best fit I found. As a
   result the **merged overall length is 107.10 or 107.36, and the head height 25.24 or 25.50**, depending on the
   pose. The merge check warned about exactly this ("check before merging": symmetry plus doubled skin).
3. **The scanner (Revo Metro export) is consistent to a few hundredths, not to micrometres** (scanner, medium
   confidence).
   - Axial scale: the two scans of the same bolt differ by +310 to +400 ppm along its axis (0.03–0.04 mm over
     107 mm). Locally along the thread the difference reaches ~1000 ppm (0.03 mm over 35 mm).
   - Diameters: thread crests agree within 0.03 mm, but roots differ by 0.09 mm and the head across-flats by
     0.14 mm.
   - Absolute scale: both scans read the thread pitch long, by +473 and +785 ppm against the nominal 1"-8 UNC
     pitch, so about 0.05–0.08 mm per 100 mm. This only holds if the bolt's own lead is exact, which is not known.
4. **Measuring tools add their own scatter** (software, high confidence). A single two-point measurement on a raw
   cloud scatters ±0.045 mm (1 σ, worst 0.2 mm) because it snaps to one noisy point; on the mesh it is ±0.016 mm.
   CloudClean's thread tool refuses scan 05 ("No thread found"). A single scan cannot give the pitch diameter
   or the overall length at all: each scan saw only one thread flank and only one end of the bolt.

**Most likely:** if you compared the big viewer numbers or the fitted box with calipers, the difference is the
display (1). If you measured with the tools on the merged scan and are 0.05–0.3 mm off, it is (2) plus (3): the
merge of two slightly inconsistent scans, not the cleaning or meshing. A caliper check of the head height
(25.24 vs 25.50) settles which merge pose is right. A gauge block or ball bar settles the scanner's absolute scale
(see "The definitive test" below).

| Step / readout | What it does to the dimensions (real bolt; synthetic ground truth) | Side | Confidence |
|---|---|---|---|
| Viewer X / Y / Z (big numbers) | Axis-aligned box in scanner coordinates: 94.6 × 40.0 × 95.5 (merge) against a bolt ~107.2 × 37.8 × 37.8. Not a part size. | software (display) | high |
| Viewer "fitted" box (small print) | Open3D box tilted 1.6–3.4° off the bolt axis: length 107.85–108.28 (+0.5–1.0 mm); width 44.35 / 44.42 on the single scans (+6.6 mm), 38.3–38.8 on merges | software (display) | high |
| Clean (standard) | No point moved. 15,150 points removed (4,079 stray, 11,071 on the surface). Size change ≤ 0.003 mm. Synthetic: fitted sizes ≤ 1.4 µm, sphere centres ≤ 0.3 µm, thread pitch ≤ 20 ppm | software | high |
| Clean with *remove plane* (part on a table) | Also removes the part's own bottom within 3× spacing of the table: the visible height drops 0.17 mm (box) to 0.32 mm (cylinder). Fitted diameters unchanged. Not used on the bolt. | software | high (synthetic) |
| Merge transform | Rigid: determinant 1.0000000056, orthonormal to 1e-8, rotation 179.99°, no scale | software | high |
| Merge result | Two skins 0.08 mm apart (median; noise explains 0.013), up to 0.5 mm on the head dimples. Pose ambiguity of one head flat changes overall length / head height by 0.26 mm. Synthetic: exact pose ≤ 2 µm; a 0.05 mm inconsistent view adds +0.05 mm to every fitted size | scan strategy + software | high (effect), medium (which pose) |
| Poisson mesh (watertight, depth 10) | Follows the cloud: signed mean −0.0003 mm, p95 0.043 mm. Thread major +0.003, pitch diameter +0.014, length under the head −0.003. Width / height −0.03 / −0.07 mm (it averages the two skins, where the cloud's extremes are the outer skin). 5.1 % of its surface is filled in with no scan data. Synthetic: ≤ 8 µm, thread roots −16 µm | software | high |
| Two-point measure | Cloud: ±0.045–0.048 mm (1 σ), worst 0.13–0.21 mm. Mesh: ±0.015–0.017, worst 0.06. No bias | software | high |
| Thread tool | Fails on scan 05 and on any one-flank view. A single scan has no pitch diameter (one flank hidden) | software + scan strategy | high |
| Scanner: scan 05 vs scan 06 | Axial +310 to +400 ppm. Local lead differs up to ~1000 ppm. Crest Ø +0.03, root Ø −0.09, head across-flats −0.14 mm. Head dimples ±0.1–0.5 mm | scanner | medium |
| Scanner: absolute (thread pitch vs nominal 1"-8 UNC) | +473 ppm (05), +785 ppm (06), i.e. 0.05–0.08 mm long per 100 mm, **if** the bolt's lead is exact | scanner (or the part) | low–medium |

Not covered: the "box capture (before update)" asset (`8564ed23e9b4`). It was made by CloudClean's native MetroY
capture before the stripe/line-identity fixes, so it is obsolete. The open native-capture questions (two laser
families disagree by ~0.16 mm; lid rim 59.9 vs 62) are tracked in `docs/metroy-handoff.md`. They do not apply to
these Revo Metro exports.

## What the bolt is and what each scan saw

- **The part:** a 1"-8 UNC bolt, right-handed in every asset, so nothing was mirrored. The pitch measures
  3.1765–3.1775 against a nominal 3.175. The thread crests measure 25.05–25.08. The head is a 12-sided prism,
  37.7–37.85 across the flats and ~25 tall, with two shallow spherical dimples on every flat.
- **Scan 05 (`03264af7e9f5`):** looks from the tip side. It sees the tip face, the shoulder under the head and
  the thread flanks facing the tip (132,488 flank points facing the tip, 11,896 facing the head).
- **Scan 06 (`ecce23d3a40c`):** the part flipped. It sees the head top and the flanks facing the head
  (19,479 / 134,627).
- **Consequence:** each thread flank and each end of the bolt was measured by one scan only. The pitch diameter,
  the overall length and the head height therefore exist only in the merge, and depend on how the merge placed
  scan 06.
- **Noise:** local plane-fit noise is 0.0074 mm for scan 05 and 0.0176 mm for scan 06. The merge reads 0.030
  because its two skins look like noise.

## Scanner-side findings (scan 05 vs scan 06, no processing involved)

**Alignment.** I tried 53 starting poses (principal-axis flips and 15° steps about the axis), refined each with
ICP, and ranked them by how many of scan 06's points lie within 3 σ (0.057 mm) of scan 05's surface. Candidates
one head flat apart score within 10 % of each other.

- Best fit: 20.1 % of points within 3 σ, median separation 0.076 mm.
- The merge's approved pose, 30.8° away: 18.8 %.

A screw-motion scan over a full turn (`tools/accuracy/symmetry_scan.py`) holds the thread matched while rotating
and advancing by pitch × angle / 360. The head agrees best at 0° (21.2 %), then +30° (19.2 %), −30° (17.6 %),
±60° (14 %), with dips to 8 % between flats. The head therefore does pick a pose, but only weakly. A pure axial
shift is pinned by the thread (±0.1 mm costs 15 % of the agreement) and hardly by the head.

**Surface agreement at the best pose.** Median separation 0.078 mm, p95 0.27 mm; noise alone explains 0.013 mm.
Median signed separation by region:

- head sides −0.061 mm
- thread flanks −0.027 mm
- crests and roots 0.000 mm

The head cross-sections (radius against angle) show the flats and their dimples. In some sectors scan 05
measures a dimple 0.3–0.5 mm deeper than scan 06; in others the two coincide. Concave (and possibly shiny)
dimples are a known weak spot of laser-line scanning: interreflection and view-dependent occlusion. Most of the
"doubled skin" the merge check reported comes from here.

**Scale.**

- Each scan's own helix fit: pitch 3.17650 (05) vs 3.17749 (06). Scan 06 is +312 ppm longer along the axis.
- The whole-part fit with a separate scale per axis plus a thickness offset gives +400 ppm along the length
  (0.043 mm over 107 mm), with a thickness offset of +0.006 mm.
- The two figures agree: **the scans differ by 0.03–0.04 mm over the bolt's length.**
- The lead is not uniform within a scan. Pitch of the head half / tip half: scan 05 +710 / +222 ppm, scan 06
  −92 / +1201 ppm. The same stretch of the same bolt therefore reads up to ~1000 ppm (0.03 mm over 35 mm)
  differently in the two scans: local, non-uniform scale errors of the scan.
- After removing the trend, the peak-to-peak deviation from an ideal helix is 0.044 mm (05) and 0.025 mm (06).
- Across the axis, one uniform scale is confounded with the thickness difference. On the thread section alone the
  radial terms come out about −1300 ppm (0.016 mm on the radius).

**Absolute scale without calipers.** 1"-8 UNC has a 3.175 mm pitch. Scan 05 reads +473 ppm and scan 06 +785 ppm.
The helix fit is unbiased to ±10 ppm on synthetic threads, with both random and scanner-like voxel sampling
(below), so these deviations are in the data. They are either the scanner reading 0.05–0.08 % long along the
axis, or the bolt's lead not being exact: commercial threads are not held to better than a few hundred ppm of
lead. **Suggestive, not proof.**

**Diameters compared without any alignment** (each scan measured in its own frame):

| | scan 05 | scan 06 | merge | clean | mesh |
|---|---|---|---|---|---|
| thread major, from crest flats | 25.051 | 25.082 | 25.062 | 25.061 | 25.064 |
| thread minor, from roots | 21.325 | 21.235 | 21.293 | 21.294 | 21.301 |
| pitch diameter (needs both flanks) | – | – | 23.484 | 23.484 | 23.498 |
| head across flats | 37.828 | 37.689 | 37.852 | 37.852 | 37.754 |
| length under the head (shoulder → tip, scan 05 faces) | 81.862 | – | 81.863 | 81.862 | 81.860 |
| head height (head top from 06, shoulder from 05) | – | – | 25.239 | 25.240 | 25.243 |
| overall length | – | – | 107.102 | 107.102 | 107.102 |

- A 1"-8 UNC class 2A thread should have a major diameter of 24.78–25.35 and a pitch diameter of 23.11–23.29
  (ASME B1.1). The crests fit.
- The merged pitch diameter is 0.2 mm above that range. That is what a 0.11 mm axial misplacement of one scan's
  flanks against the other's would produce (synthetic: 0.05 mm gives −0.087 mm). It could also mean the bolt is
  not a standard 2A thread.
- With the best-fit pose instead of the approved one, head height and overall length become 25.499 and 107.361.
- The merged cloud's head across-flats (37.852) is the outer of the two skins. The mesh (37.754) sits between
  them (scan 06 alone: 37.689).

## Processing steps on the real data (operation drift)

`POST /api/accuracy/drift` (`accuracy.operation_drift`) on the chain:

- **clean vs merge:** "No measurable change". 15,150 points removed (4,079 stray points off the surface, 11,071
  from the surface itself), none moved. Length −0.000, width +0.003, height +0.001 mm.
- **merge vs scan 05:** 45.7 % of the merged points are scan 05's own, unmoved. Scan 06's points lie a median
  0.083 mm (p95 0.32) from scan 05's surface where both see it facing the same way. 23.5 % of the merged surface
  is new coverage from scan 06.
- **merge vs scan 06:** placed by a rigid transform (determinant 1.000000, no scale); the rest mirrors the above.
- **mesh vs clean:**
  - The mesh surface lies −0.0003 mm from the cloud on average (median |d| 0.006, p95 0.043).
  - Seen from the cloud's points the difference is −0.0013 mm (p95 0.104).
  - 5.1 % of the mesh was filled in with no scan data nearby (watertight Poisson).
  - Size, measured with extents that do not depend on noise: length −0.006, width −0.031, height −0.067 mm.
    This is the Poisson surface averaging the two skins.

## Software-side ground truth (synthetic parts)

`tools/accuracy/synthetic_bench.py` samples the reference parts at 0.15 mm spacing with 0.03 mm Gaussian noise.
That is noisier than the bolt scans, so the effects below are upper bounds. Each scan also gets 1 % stray points
and a debris cluster. The results are measured with robust fits against the truth. Errors are in µm, thread pitch
in ppm.

| Step | cylinder Ø20×60: Ø / length | box 40×30×20: x / y / z | 2 spheres Ø20, 60 apart: Ø / centre distance | thread 1"-8: pitch / major / minor / pitch Ø |
|---|---|---|---|---|
| clean light / standard / aggressive | −0.1 / ±0.1 | −1.4…−0.6 | −0.8 / −0.3 | −20…−1 ppm / ≤ 0.5 / ≤ 1.1 / ≤ 0.3 |
| merge two views, approved exact pose | 0.0 / −0.2 | −1.9 / −0.5 / −1.3 | −0.6 / 0.0 | +10 / +0.5 / +0.7 / −2 |
| merge two views, automatic | +0.1 / −0.2 | **−441** / −0.5 / **−238** (flipped pose on a symmetric box; flagged ambiguous) | −0.8 / 0.0 | −6 / 0 / +0.5 / – |
| merge, one view 0.05 mm "thicker" (doubled skin) | **+50 / +50** | **+48 / +49 / +49** | **+50** / −0.4 | −14 / **+50 / +51 / +96** |
| merge, one view's flanks 0.05 mm off axially | | | | pitch Ø **−87** |
| Poisson (trimmed) | −0.7 / +6.0 | −2.9 / +3.8 / −0.1 | −0.2 / −0.3 | +30 / +8 / −16 / −4 |
| Poisson watertight | −0.5 / +6.0 | −3.0 / +3.8 / 0 | −0.1 / −0.3 | +72 / +9 / −15 / −16 |
| Poisson + Taubin ×20 | −0.4 / +5.9 | −3.1 / +4.3 / 0 | −0.1 / −0.3 | +25 / +10 / −22 / −5 |
| ball pivoting | **+16 / +17** | **+15 / +15 / +16** | **+13** / 0 | +4 / +8 / +17 / +30 |
| cloud: denoise | −9 / +0.2 | −0.6 | **−19** / −0.4 | −6 / **−131** / +38 / −6 |
| cloud: smooth (MLS quadratic) | −0.2 / +0.3 | −0.5 | −0.9 / −0.5 | −3 / +7 / −8 / 0 |
| cloud: smooth_points (MLS plane) | −2.6 / +0.4 | −0.3 | −5.7 / −0.4 | −3 / −8 / −2 / −3 |
| cloud: remove_spikes | ≈0 | ≈0 | ≈0 | ≈0 |
| mesh: smooth Taubin ×5 / ×20 | −0.5 / +5.8 | −2.8 / +4.0 | −0.1 | +22 / +9…+15 / −18…−26 / −5 |
| mesh: smooth Laplacian ×5 | −2.5 / +5.5 | −2.7 / +4.6 | −3.0 | +29 / **−41** / −10 / −5 |
| mesh: denoise ×5 | −0.8 / +5.4 | −2.0 / +5.7 | −0.8 | +21 / −14 / −34 / −6 |
| mesh: remove_spikes | as Poisson | as Poisson | as Poisson | as Poisson |

Other bench results:

- **Part on a table, standard clean with support-plane removal:** the visible height drops by 169 µm (box) and
  319 µm (cylinder). Without plane removal, the height above a plane fitted to the table is +78 / +41 µm (noise
  at the top face). The cylinder fit is +0.7 µm either way.
- **Trimmed extents** (0.05 % trimmed, the kind of "part dimensions" a summary shows) read +106…+149 µm large on
  the cleaned cylinder. Noise pushes them out by about 2.5 σ per end. Face fits (calipers) are unbiased (−0.1 µm).
- **Principal axes** of a 40 × 30 box are tilted ~0.8° by random sampling alone, which adds 0.4 mm to its
  extents. `accuracy.robust_dimensions` now turns the axes to the smallest box (40.054 instead of 40.404).
- **Two-point measure** (400 random opposite-point pairs, clicks 0.2 mm off target):
  - cloud: mean +0.8 / +6.7 µm, σ 48 / 44 µm, worst 207 / 134 µm
  - mesh: σ 15 / 17 µm, worst 60 µm
- **Thread tools:**
  - The app's `thread_analysis`: −25 ppm / +8 / −14 µm with random sampling, +26 ppm / +10 / −14 µm with
    Revo-like voxel sampling. It fails ("No thread found") on a one-flank view, which is what every real single
    scan of this bolt is.
  - The helix fit `accuracy.fit_thread`: −8 ppm / +4.5 / +1.1 / +1.5 µm on voxel sampling. On a one-flank view:
    −6 ppm / −0.7 / −0.2 µm, and it reports no pitch diameter.

In short: cleaning and Poisson meshing preserve size to a few micrometres. Ball pivoting inflates it by ~15 µm.
The feature-preserving *denoise* and Laplacian smoothing flatten thread crests by 0.04–0.13 mm. A merge is only
as good as the consistency of its scans and the pose it was given.

## The definitive test

These settle the remaining doubt with no software judgement involved:

1. **Scanner scale and repeatability.** Scan a certified artefact.
   - Artefacts: a gauge block (50–100 mm), a ball bar or two precision balls on a bar, or the MetroY calibration
     board with its certified spacing.
   - If it is shiny, give it a thin matte coat, the same as the bolt had.
   - Scan it twice with the same settings, once lengthwise and once crosswise.
   - For each scan call `POST /api/accuracy/reference`, e.g. `{type: "known_length", direction: "length",
     nominal: 100.000}` or `{type: "sphere_pair", region_a, region_b, nominal}`. Then call
     `POST /api/accuracy/compare-scans` on the pair.
   - Reading the result: an error within the scanner's published accuracy means any bolt deviation comes from
     scanning the bolt, not from the scanner's scale. A consistent +0.05–0.08 % would confirm the thread-pitch hint.
2. **This bolt, with calipers.**
   - Head height: 25.24 means the approved merge pose is right; 25.50 means the best-fit pose is.
   - Overall length (107.10 / 107.36), length under the head (81.86), across flats (37.7–37.85), crests (~25.06).
   - Thread wires or a thread micrometer for the pitch diameter (merge: 23.48).
3. **Repeat scan in the same orientation.** Scan the bolt twice without flipping it and run compare-scans. If the
   pair agrees within ~0.02 mm while 05 and 06 differ by 0.08, the difference is view dependent (the dimples,
   one flank per scan), not the scanner drifting.

## Recommended software fixes

1. **Viewer:**
   - Show robust part dimensions (length / width / height along the part's axes; Contract 3 summary or
     `POST /api/accuracy/dimensions`) as the big numbers.
   - Label the axis-aligned box "scanner box".
   - Stop calling Open3D's box "fitted": it is tilted by degrees on this part. Use the min-box axes
     (`accuracy.refine_axes`).
2. **Merges:**
   - When the assessment was ambiguous or found a doubled skin, keep that verdict on the merged asset and show it
     next to every measurement taken on it and its children.
   - Offer to measure on the single scan that saw the feature.
   - For round parts, report the length uncertainty that the pose ambiguity implies (here ±0.26 mm) and show the
     screw-motion candidates side by side.
3. **Merge check (`register.assess_pair` layering):** measure only where both scans see the surface facing the same
   way (normal agreement, as `SurfaceModel.query(normals=…)` does). Scans of a thread see complementary flanks,
   and a plain nearest-surface distance counts that as doubled skin. On this bolt the head dimples are a real
   doubled skin as well.
4. **Thread tool (`analysis.thread_analysis`):**
   - It rejects real scans: its per-bin coverage test fails on one-flank views.
   - Report the helix-fit pitch (`accuracy.fit_thread`, unbiased to ±10 ppm here).
   - Report the pitch diameter only when both flanks were scanned, and say which flank is missing.
5. **Smoothing:**
   - Cloud *denoise* (−0.13 mm on thread crests) and mesh *Laplacian* (−0.04 mm) change feature sizes. Warn when
     they run over sharp features, or default to MLS quadratic smoothing (≤ 8 µm).
   - Ball pivoting inflates sizes by ~15 µm; prefer Poisson for measuring.
6. **Plane removal** takes 0.17–0.32 mm off a part that sits on the table. Report how many removed points were
   touching the part, or use a smaller distance near the object.
7. **Two-point measure:** snap to a local surface fit (the quadric `SurfaceModel` uses) instead of the nearest raw
   point. That cuts the scatter from ±0.045 to about ±0.015 mm. Show the expected uncertainty next to the value.
8. **Part dimensions from trimmed extents** read ~2.5 noise σ per end too large. Prefer plane fits to end faces
   (`accuracy.measure_length`) for lengths, and say "approximate" on trimmed extents.

## Fixes applied (same day)

The lead implemented fixes 1 (the fitted box now uses the smallest box, e.g. scan 05 reads 107.67 × 38.30 × 37.81)
and 2 (the merge caveat travels with the merged asset). Workstream A implemented 3, 4, 5 and 7. Before / after on the
real bolt:

| Fix | Before | After |
|---|---|---|
| 3. Merge check (05 + 06) counts layering only where both scans face the same way | layering 0.097 mm, p90 0.286, 7.5× noise, doubled skin | 0.085 mm, p90 0.255, 6.6× noise, still flagged doubled. That is correct: the head dimples really differ, and the robust whole-part separation is 0.078 mm. |
| 4. Thread tool, scan 05 | "No thread found: only 1 complete crest(s)" | helix fit: pitch 3.17650, major 25.078, minor 21.305, 21 crests; pitch diameter and flank angle null, with a warning that names the hidden end |
| 4. Thread tool, scan 06 | pitch 3.17676, major 25.218, pitch diameter 23.519 and flank angle 60.5° from one flank, 4 crests, "high" confidence | helix fit: pitch 3.17749, major 25.096, minor 21.207, 21 crests; pitch diameter and flank angle null, with the flank warning |
| 4. Thread tool, merge (both flanks) | pitch 3.17741, major 25.146, pitch diameter 23.520, flank angle 56.4° | unchanged (crest-by-crest path) |
| 5. Cloud denoise / mesh Laplacian | no warning | warning in the op description and the method help, logged `WARNING`, `warning` in the op report |
| 7. Two-point snapping | nearest raw point: σ 0.048 mm, worst 0.138 (synthetic, 0.03 mm noise) | `POST /api/measure/snap` (local quadric of 24 neighbours): σ 0.018 mm, worst 0.052, with a predicted uncertainty of 0.018 |

## Method and reproducibility

- **Data:** the five assets were copied from the Spark workspace. Nothing on the Spark was modified, and no
  mutating API of the running service was called.
- **Tools:**
  - `python tools/accuracy/real_bolt.py <assets>` covers per-asset features, operation drift, scan comparison,
    per-region separation, the thread-only fit and the lengths under both poses.
  - `python tools/accuracy/symmetry_scan.py <assets> --pose pose.npy` runs the screw-motion scan.
  - `python tools/accuracy/synthetic_bench.py` runs the bench.
  - Results are in `tools/accuracy/results/*.json`.
- **Bolt frame:**
  - The axis comes from a least-squares helix fit of the thread (Fourier profile of the thread phase, axis tilt,
    axis position, pitch).
  - Axial faces come from robust plane fits to points whose normal is within 18° of the axis.
  - Head across-flats is the minimum caliper width over directions around the axis.
  - Crest and root diameters use points whose normal is radial.
- **Surface distances:** a quadric is fitted to the 16 nearest points of the reference cloud, so thread crests are
  not biased the way a plane fit is. A distance counts only where both surfaces face the same way (≤ 30–45°) and
  the query lies over the fitted patch.
- **Robust fits:** Tukey IRLS with a MAD scale.
- **Scale fits:** point-to-surface with rigid, similarity, similarity + offset and per-axis scale + offset models.
  Their standard errors come from the fit covariance and are optimistic, because residuals are spatially
  correlated.
- **Validation:** the helix fit recovers the pitch within ±10 ppm and the diameters within ~3 µm on synthetic
  threads (random, grid and voxel sampling, one or two flanks). `tests/test_accuracy.py` checks all of the above
  on synthetic data.
