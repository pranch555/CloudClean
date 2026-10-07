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
8. Auto exposure (Revo's `RPALaserAutoExposure`), Revo's online constant adjustment of the line maps, flying-point
   removal and bilateral denoise (parameters in docs/revo-metro-internals.md 2.2).

**Open question for the user, still unanswered:** hand-held sweeping or a turntable? With markers, both work the same
way (markers on the table for a turntable).

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
