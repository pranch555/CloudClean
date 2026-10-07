# MetroY on the DGX - handoff

Where the work stands as of 2026-09-21 (end of day), and what to do next. `docs/metroy-protocol.md` is the reference
(wire protocol, registers, calibration files, validation numbers); `docs/revo-metro-internals.md` is what Revo
Metro itself does, from static analysis of its binaries. This file is the state of play and the plan.

## The goal

Capture with the Revopoint MetroY Ultra **directly on the DGX Spark**, inside CloudClean, behaving the way Revo
Metro does - without Revo Metro, which only runs on Windows/macOS. The Windows + `cloudclean.capture.bridge` path
still works and stays the fallback.

**Accuracy is the user's top priority.** Never ship geometry derived from guessed calibration, and say plainly when
a number is not trustworthy.

## What works now (verified on hardware)

The `metroy_usb` driver is a real streaming driver, live in the CloudClean service on the DGX (Capture page:
"Revopoint MetroY (USB, native)"). Code: `cloudclean/capture/metroy/` + `cloudclean/capture/drivers/metroy_usb.py`.

| | |
|---|---|
| Calibration | stereo (`camparam.yaml`) **and laser lines (`metroExtra.bin`) read off the scanner** at every connect; cached per serial in `~/.cache/cloudclean/metroy/` |
| Acquisition state | Revo Metro's start-up replayed register by register, then Revo's surface preset (general / dark / reflective: exposure, gain, fill light) |
| Stream | ~89 fps after the start-up (28 in the boot state), both line families alternating, kernel timestamps, dropped-frame counting |
| Stripe centres | Steger ridge detector + along-stripe fit: flat board **31-40 um RMS** (was 330 um) |
| Correspondence | **laser line identity from the factory maps**, as Revo Metro does: no mis-paired stripes; the box that measured 11-44 mm wrong now measures consistently |
| Throughput | ~150-200 ms CPU per frame over 12 worker processes: ~45 frames/s triangulated (a busy pool skips older frames and always takes the newest), ~6-14k points each |
| Markers | detection, stereo, 3D and a marker-map tracker (Kabsch + triangle-signature relocalization), tested on synthetic data to < 0.15 mm pose error. **Not yet seen a real marker** - see below |
| Session | markers mode (default, Revo's laser-mode behaviour): the driver supplies the pose; frames the markers cannot place are reported lost with the reason, never guessed. Geometry mode: CloudClean's ICP |
| World frame | markers on one flat table seen from above (the turntable plate): the scan stands on it - Z along the plate's normal, plate at z = 0, so a turntable turns about Z (`metroy_usb._table_frame`; the user's plate fits a plane to 0.19 mm, 12 markers). Otherwise Z = sensor up at the first marker frame. Before 2026-10-07 a map started on a frame with markers but no laser points yet stayed in sensor coordinates: such scans lie on their side (Clean -> Sit flat on the floor, or re-scan) |
| Tests | `tests/test_metroy.py` (synthetic step scene through the real rectification, markers, tracker, laser file). Suites: 105 passed on Windows and on the Spark |

## What is missing, in order

1. **Real markers.** Stick retro-reflective markers (Revo Metro's 3 mm / 6 mm dots) on and around a part, connect
   with tracking = Markers, and sweep. Check: markers per frame (driver meta `markers`, `marker_inliers`,
   `map_markers`), pose smoothness, and that a static scanner gives a static pose. Tune `MarkerParams` (threshold,
   min area) on real blobs if needed - they were only ever tested on synthetic ellipses and against false positives
   on a marker-free scene (none at the Revo exposure).
2. **The two line families disagree by ~0.16 mm** on the same spot of a static scene (`tools/metroy/family_consistency.py`).
   The likely cause is a ~0.2-0.25 px vertical misalignment of the rectified pair (a stripe of slope s gets a disparity
   error s*dy, and the families have opposite slopes): shifting the right view by -0.25 px made them agree
   (`row_offset_scan.py`), but the response was not cleanly linear and a SIFT check on the low-texture scene was
   inconclusive. **Markers measure dy directly** (row difference of each stereo pair, ~0.05 px each): once markers
   are in view, log it, and if it is stable apply it as a right-view row shift. Do not apply a correction before then.
3. **Validate end to end** against the user's bolt: a DGX-native scan versus a Revo Metro scan of the same part.
   Revo Metro's own per-frame points and poses can be read from a project's `cache/frames.dataset`
   (docs/revo-metro-internals.md, section 8) for a like-for-like comparison.
4. **Lid rim open question.** The test box's rim reads 59.9 mm above the paper (calipers: 62) and 121.9 mm long
   (calipers: 120.65), from 8 fused frames. Errors in opposite directions are not a scale error, and the paper under
   the box is curled (443 um RMS about a plane), so the reference is soft - but this is not explained. The bolt
   comparison (3) should settle whether there is a real bias.
5. **Geometry tracking on laser frames is weak**: CloudClean's ICP estimates normals on a model that is a set of
   stripes ~12 mm apart, slides along them and reports "lost" about half the time even on a static scene. Revo Metro
   does not offer geometry tracking in laser mode at all. If a no-marker mode matters (turntable), fuse a frame pair of
   both families into one cross-hatched frame before ICP, or constrain ICP to the stripe normals.
6. **Other laser patterns.** Parallel (`0x3001 11`, section 2 of metroExtra.bin) and single line (`0x3001 9`,
   section 3, a 3D surface - formula in docs/revo-metro-internals.md 4.4) are decoded but not wired.
7. **Where the per-frame family ("direction") byte lives on USB** - probably the UVC metadata node `/dev/video1`.
   Not needed today (the family is inferred per frame, reliably), but it would remove the inference.
8. Auto exposure is implemented (section "Camera view" below) but not yet run on the scanner. Still missing: Revo's online constant adjustment of the line maps, flying-point
   removal and bilateral denoise (parameters in docs/revo-metro-internals.md 2.2).

**Open question for the user, still unanswered:** hand-held sweeping or a turntable? With markers, both work the same
way (markers on the table for a turntable).

## Camera view, laser light and auto exposure (2026-10-07, NOT yet run on the scanner)

Scan → Camera view shows the two IR cameras live (before scanning as a preview, and while scanning; "Show over the 3D
view" floats it top right), with the laser lines CloudClean found in green, saturated pixels red and markers ringed
blue, a plain-words readout ("Laser lines: bright enough / too dim / 12 % too bright"), Normal / Dark / Shiny, and Auto
or Manual (laser brightness, exposure, gain, marker light). Code: `metroy/exposure.py` (presets, pulse formula, write
plan, measurement, controller - pure, unit tested), `metroy/camera.py` (the one paced register writer, preview lease,
the picture), `metroy/scanner.py` (wiring), `drivers/metroy_usb.py` (preview lifecycle), API in `web/routes_capture.py`
("Camera view"), tests in `tests/test_camera.py` (fakes and the simulated scanner only).

| | status |
|---|---|
| Presets: general 200 us / gain 1 / level 44 of 51 / marker 30; dark 1000 / 2 / 193 of 255 / 3; reflective 800 / 1 / 90 of 204 / 6 | **verified** (Revo's crosswire_param.json) |
| Marker light per gain on a gain step (`/data/brightnessGainMap.json`, read at connect; fallback = our scanner's copy) | **verified** values (Revo log `set IR tmpLuminance`); reading the file at connect is new |
| `0xb08 = 45 + int(level * exposure_us / level_max)`: general 44 -> 217, 38 -> 194, 48 -> 233, 51 -> 245 | **verified** (four Revo log pairs, general only) |
| the same formula for dark (193 -> **801**, Revo's AE end point 213 -> 880) and reflective (90 -> **397**) | **inferred, not measured**: no SDK log of a dark session survives. Capped at 1045 (dark at 255), never more |
| `0x48d` laser power stays 255 (start-up); AE never touches exposure, only level, gain, marker light | verified (Revo writes 0x48d 255 always) |
| AE target: stripe-centre brightness (95th pct) 200 of 255, band 175-230, <= 3 % saturated, every 0.8 s, only at 220-380 mm | window/interval **verified** from Revo; the target is **CloudClean's choice** (Revo's not recovered) |
| Write rules: 0x910+0x911 in one command; every other register alone; >= 0.15 s between commands from any thread (`MetroyHid.shell` paces itself); a register is never re-written with the value it holds; writes coalesce | as measured before; enforced in code + tests |

What changed on the wire: after `startup()` CloudClean now writes only the registers that differ from what startup()
left (general + markers writes nothing extra; dark writes 0x910/0x911, 0xb08, 0x903, 0xb07 - four commands, one more
than before: **the 0xb08 write for dark/reflective is new**). While streaming, auto exposure writes 0xb08 (and on a gain
step 0x903 + 0xb07) at most once per 0.8 s, one command each - the same registers Revo writes while scanning.
The preview streams with the laser on only while a camera view is open (the browser renews a 6 s lease); it ends by
itself when nobody looks, and Start scanning turns it into the scan without a restart. In preview only ~12 frames a
second are triangulated (the scan uses the full pool).

### Hardware check before trusting Dark (one register at a time; Revo Metro closed)

1. Connect native, tracking = Markers, surface **General**, Camera view open, **Manual**. A flat matte black target
   (the turntable plate) at ~300 mm. Note the readout: stripe brightness, points per frame. Expect ~1-4 grey levels
   above black on the plate, as in the 10-06 recording.
2. Choose **Dark** (writes 0x910 8000 + 0x911 1000 together, then 0xb08 801, 0x903 0x20, 0xb07 3 - one per command,
   0.15 s apart). Watch that the stream keeps running (frames/s in the session card), no re-enumeration (`dmesg`).
3. In Manual, step Laser brightness down to ~22 % (level 55 -> 0xb08 260) and back up in a few steps to 100 % (level
   255 -> 1045). Per step, read the stripe brightness on the same spot. **The formula holds** if brightness above
   black grows roughly in proportion to (0xb08 - 45) while 0x911 stays 1000. **It is wrong** if (a) brightness stops
   growing above some 0xb08 well below 1045 (the pulse is clipped by something else - then the cap or the formula
   for dark is off), (b) it does not change at all (0xb08 is not the pulse in this mode, or not microseconds), or
   (c) it rises as fast at 0xb08 217 with 0x911 200 as at 1045 with 1000 (the pulse is independent of exposure).
   Also compare with Revo Metro on the same spot: Revo's log line `AutoExposure ... Adjusted laser brightness: N`
   with the SDK log's `echo s 0xb08 M` at the same time gives one more (level, pulse) pair for dark - one pair is
   enough to confirm or refute 45 + N * 1000 / 255.
4. Switch to **Auto**: it should settle within a few seconds (Revo ended at level 213, gain 3 on the screw).
   Points per frame on the black plate/screw should rise well above what General gives.
5. Back to General: 0xb08 must return to 217.

Risk: 0xb08 values above 245 have never been written by CloudClean; Revo's own dark preset implies it writes ~800-880.
If the scanner resets after a write (it re-enumerates in ~15 s), note which register and value, disconnect, and do not
retry that value; everything else is the unchanged start-up.

## Where everything is

**Windows laptop** - the development copy of this repo. Revo Metro is installed here. USB captures in
`Desktop\metroy-capture\`.

**DGX Spark** - `ssh spark` (over Tailscale). The scanner is plugged in here.

* CloudClean: `~/code/CloudClean`, served by
  `systemctl --user {status,restart} cloudclean` on port 8765, workspace `~/cloudclean-workspace`. Its venv now
  has `opencv-python-headless` 5.0 (the `[metroy]` extra), same numpy as before.
* Open3D on ARM64 needs `libgfortran.so.5`: a venv made by `./cloudclean.sh setup` has it in Open3D's folder; an
  older venv needs `LD_LIBRARY_PATH=$PWD/.venv/lib/extra` when running by hand.
* `~/metroy-calib`: the older scratch area with its own venv (OpenCV, no SciPy). The scripts in
  `~/code/CloudClean/tools/metroy` import the package and run with either venv
  (`~/metroy-calib/.venv/bin/python` or `.venv/bin/python` when they need SciPy).

**Tools** (`tools/metroy/`):

| File | Purpose |
|---|---|
| `stripes.py` | CLI over the package triangulation: `python stripes.py frame.raw camparam.yaml` |
| `stream_test.py`, `session_test.py` | the capture engine / a real CaptureSession, live, headless |
| `flatness.py` | flatness with random vs systematic split; `--method lines|order|v2|v1` |
| `height_view.py`, `depth_view.py`, `rim_measure.py`, `box_measure.py`, `height_profile.py` | validation on the box |
| `family_consistency.py`, `row_offset_scan.py`, `row_error.py` | the open 0.16 mm family question |
| `coverage_diag.py` | why stripe tracks go unpaired |
| `exposure_test.py`, `gain_test.py`, `filllight_test.py`, `fill_stripes_test.py` | register experiments (all verified) |
| `marker_view.py`, `crop_view.py` | marker detector overlay, full-resolution crops |
| `metroy_hid.py`, `laser_test.py`, `laser_scan*.py`, ... | the earlier prototypes, kept for reference |
| `camparam.yaml`, `metroExtra.bin` | the calibration off our scanner (also in `tests/data/metroy/`) |

## Gotchas that cost time already

* **Rectify with `cv2.stereoRectify(..., flags=0)`** - it reproduces Revopoint's own `pL`/`pR`.
* **The frame is 8-bit 1600x2400, left camera on top**, even though UVC calls it YUYV 1600x1200.
* **Use Revo Metro's video mode** (UVC format 1, frame 1). Y16 modes and UVC control writes reset the scanner.
* **Register writes:** `0x910` only together with `0x911`, in one command; everything else one register per command
  with a pause - four chained in one command reset the scanner. After any reset, run the start-up again (the laser
  comes back dim).
* **The scanner re-enumerates itself** when unhappy (~15 s); wait and retry.
* **Order-preserving stereo matching is wrong** across steps and occlusions (a one-stripe slip is 11-44 mm and still
  looks like a surface). Flatness tests cannot catch it - a mis-paired plane is still flat. Look at a height map.
* **Retro-reflective markers vs stripes:** at the boot exposure with the fill light on, saturated stripes on white
  paper look like markers. Revo's 200 us exposure keeps the background dark; keep it.
* **`VIDIOC_QBUF` clears `timestamp`/`sequence`** in the struct you pass: read them first.
* **Process pools must use `spawn`** (the web server is multi-threaded), and any script driving the engine needs an
  `if __name__ == "__main__":` guard.
* **The test box's lid** has a rim around a recessed panel; "the top plane" is the panel (104 x 74 mm, 56 mm up).

## Quick check that the environment is alive

```bash
ssh spark 'lsusb | grep 2207; ls /dev/video0 /dev/hidraw5'
ssh spark 'cd ~/code/CloudClean && ~/metroy-calib/.venv/bin/python tools/metroy/stream_test.py 5'
ssh spark 'cd ~/code/CloudClean/tools/metroy && ~/metroy-calib/.venv/bin/python flatness.py /tmp/board_3.raw camparam.yaml --gate 0.3'
```

The second should report ~45 frames/s delivered (of ~89 grabbed) and no errors; the third ~32 um RMS with ~99 % of points on
the board (the `/tmp` frames are scratch - re-capture with `seq_grab.py` if they are gone).
