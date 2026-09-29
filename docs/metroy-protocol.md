# MetroY Ultra USB protocol (reverse-engineered)

Findings from USB captures of Revo Metro talking to a MetroY Ultra (serial `D26519649H6U10L88`, USB `2207:110c`),
recorded with USBPcap on Windows, 2026-09-18. This is the groundwork for capturing directly on the DGX Spark, where
Revo Metro does not run. Nothing here is from Revopoint documentation; treat every field as observed, not specified.

## Device

A composite USB 3 device (the scanner itself is a small Rockchip Linux computer):

| Interface | Class | Linux node | Role |
|---|---|---|---|
| MI_00 | UVC | `/dev/video0` (+ metadata node) | IR stereo cameras |
| MI_02 | UVC | `/dev/video2` (+ metadata node) | RGB camera, MJPEG 1600x1200 |
| MI_04 | HID | `/dev/hidrawN` | **command channel** (interrupt OUT 0x01, IN 0x85, 1024-byte reports) |

The UVC extension units (unit 4, GUID `6d1a21c2-7e78-4a3a-ac06-7e434b676f9a`) are **never used** by Revo Metro.
Everything vendor-specific goes over the HID interface.

## HID command framing

Every report is 1024 bytes. Host to device:

```
0..3   5a 5a 5a 5a        magic
4      command            (00, 01, 04, 05, 06 seen)
5      group              (05 = file/system, 01 = acquisition)
6..    arguments          (the rest of the buffer is uninitialised host memory - ignore it)
```

Device to host, reply header:

```
0..3   5a 5a 5a 5a
4,5    echo of command, group
6      status             01 = ok, 07 = not found / not ready
8..39  ff padding
40..43 payload length, u32 little-endian
44..   first 980 bytes of the payload; further 1024-byte IN reports carry the rest
```

### `00 05` read file

Arguments: byte 41 = `01`, bytes 44.. = NUL-terminated absolute path on the scanner.

| Path | Result |
|---|---|
| `/data/camparam/camparam.yaml` | **factory stereo calibration** (OpenCV FileStorage, 2062 B) |
| `/data/camparam/metroExtra.bin` | 16074 B, dated 2026-09-02: **the laser line calibration** (decoded, see below) |
| `/data/brightnessGainMap.json` | exposure presets per mode (`lineDark`, `lineReflect`, `surface`, ...) |
| `/data/camparam/calibTemperature.json` | not present (07) |
| `/data/rtnFile`, `/tmp/fpga.ing` | not present (07) |

### `01 01` run a shell command on the scanner

Byte 7 = `0x10`, bytes 40..43 = length of the command including its NUL (u32 LE), bytes 44.. = the command text.
No reply. Revo Metro configures acquisition entirely through this, writing registers of the Rockchip pre-ISP:
`echo s <reg> [values] > /dev/rk_preisp`. Also seen: `sync`, `mkdir /data/deviceactive`, `sh /tmp/.longCmd.sh`.

Other commands: `04 01` + `imu&get_IMUStatus` / `imu&get_IMUVersion` (IMU queries); `00 01` + `/tmp/.longCmd.sh`
(uploads a helper script that records the USB link speed); `05 05` with u32 fields `256, code, 1, value` at byte 40
(codes 102, 103, 111, 113 seen; 113 was 4 in one session and 8 in another, likely a mode or resolution setting).

### Registers (`/dev/rk_preisp`), as used by Revo Metro

Meanings from Revo Metro's own help table (docs/revo-metro-internals.md, section 6) and, where marked, verified on
our scanner from the DGX.

| Register | Values | Meaning | Verified on DGX |
|---|---|---|---|
| `0xb04` | 1 / 0 | laser projector on / off | yes, the stripes disappear and return |
| `0xb07` | 0-255 | IR fill light brightness (lights the retro-reflective markers) | yes: background 26 / 86 / 118 at 0 / 30 / 60 |
| `0x910` | 8000 | frame time, us | yes, **only together with 0x911** (see below) |
| `0x911` | 200 / 800 / 1000 | exposure, us | yes |
| `0x903` | `0x10` / `0x20` | analogue gain, 0x10 per 1x | yes: 0x20 doubles the signal above the black level (16) |
| `0x3001` | 7 / 9 / 11 | laser pattern: cross / single line / parallel (MetroY Ultra) | 7 in use (cross) |
| `0x48d` | 255 | laser brightness | part of the start-up below |
| `0xb08` | 217 | laser exposure times | part of the start-up below |
| `0x707` | `200 600`, `230 630` | depth range, mm | no |
| `0xb10` | `5 15 0 1 2 4` | frame-type sequence | no |
| `0xa00`, `0x701`, `0xb01` | 1 | init / resume stream | part of the start-up below |
| `0x483` | (read) | status query | no |

Start-up order: `0xa00 1`, `0xb07 1`, `05 05 (103,1)`, `05 05 (102,0)`, `0x103 0`, `0x483`, `0x3001 7`, `0x483`,
`0x707 200 600`, `0x903 0x10`, `0x910 7000`, `0x911 5000`, `0x701 1`, `0xb01 1`, `0x910 8000`, `0x911 200`,
`0x903 0x10`, `0xb07 30`, `0x48d 255`, `0xb08 217`, `0x707 230 630`, then `0xb04 1` to scan and `0xb04 0` to stop.

How the scanner tolerates writes (measured 2026-09-21):

* **`0x910` alone** stalled the stream until the watchdog reset the device. `0x910` and `0x911` together in one
  shell command (`echo s 0x910 8000 > /dev/rk_preisp; echo s 0x911 200 > /dev/rk_preisp`) is fine.
* **Four writes chained in one shell command** made the scanner drop off the bus and re-enumerate. One register per
  command with ~150 ms between them is fine (`MetroyHid.startup()` / `acquisition()`).
* **After a reset the scanner comes back with boot defaults**, including a dim laser: stripes that gave ~7,000
  points per frame gave 6. Replaying the start-up above restores it, so capture always runs it first.

Revo Metro's cross-line presets for this model (exposure / gain / fill light): general 200 us / 1x / 30, dark
1000 us / 2x / 3, reflective 800 us / 1x / 6. At 200 us a diffuse surface stays dark under the fill light
(background 16-19) while the laser lines and the markers shine. At the boot exposure the fill light made paper so
bright that saturated stripes on it were detected as false markers.

## Laser line calibration (`metroExtra.bin`)

Decoded 2026-09-21 from data, and confirmed in Revo Metro's own parser (`RPAModelIO.dll`) and GPU matching kernel:

```
0..17    zero
18..31   ASCII date "20260902100715"
32, 33   format version 7, 4 sections
34..     per section: u16 header length (10), u16 payload length, laser type, line count, values per record (20),
         section index, 2, element type (4 = float64); then 2 x lines records
```

| section | lines | pattern |
|---|---|---|
| 0 | 17 | cross, first family |
| 1 | 17 | cross, second family |
| 2 | 15 | parallel |
| 3 | 1 | single line (a 3D surface in mm, a different form) |

For every laser line, record k maps a point on that line in the **rectified left** image to the x of the same line in
the **rectified right** image on the same row, and record k + lines maps right to left:
`x_other = c0 x^2 + c1 x y + c2 y^2 + c3 x + c4 y + c5`. Values 0..5 are the factory map, 6..11 the adjusted one
(only the constant differs; Revo Metro uses it), 12..19 four ROI corners per camera. On board frames the maps predict
the partner stripe to ~1 px after a per-frame offset of 1-2 px, while the next line lands ~180 px away, so a
stripe's line identity - and with it its stereo partner - is unambiguous. The 3D point still comes from the disparity
of the two measured centres through `Q`; the maps only choose the partner.

## Factory calibration (`camparam.yaml`)

Per camera 1600x1200, calibrated 2026-05-15:

- left  f = 1816.74 px, c = (793.87, 603.82), dist = [-0.0611, 0.1093, -6e-5, -3.3e-4, -0.0415]
- right f = 1810.47 px, c = (796.09, 607.46), dist = [-0.0607, 0.1051, 3.4e-4, 2.5e-4, -0.0290]
- stereo R (about 21.5 deg yaw), T = (-126.200, -0.025, 24.191) mm, baseline |T| = 128.50 mm
- rectified: f = 1813.61 px, `pR[0,3] = -233044.9 = -f * 128.5`, and `Q` maps disparity to millimetres
  (`Q[3,2] = 1/128.50`)

## Video

Revo Metro commits UVC format 1 / frame 1 = **YUYV 1600x1200** (3,840,000 B per frame), not the Y16 modes. UVC
advertises 5 fps, but the scanner delivers **~28 frames per second in its boot state and ~89 after Revo Metro's
start-up** (8000 us frame time; kernel timestamps and sequence numbers, no gaps, 2026-09-21). Despite the name the buffer is **8-bit grey, 1600 wide x 2400 tall: the left camera on
top, the right below**. This mode streams stably on Linux with no handshake at all.

In cross mode **successive frames alternate between the two line families** (stripe slope +0.66 / -0.49 in the
rectified left view). Revo Metro reads each frame's family from a "direction" byte whose place in the USB stream is
not yet known; CloudClean infers it per frame from which family's maps the right view confirms.

Verified on the DGX (2026-09-18):

- Rectifying with the factory calibration aligns matching features to the same row. It only matches Revopoint's
  own `pL`/`pR` when `cv2.stereoRectify` is called with `flags=0` (no `CALIB_ZERO_DISPARITY`), which reproduces the
  factory principal points cx = 383.6 / 1216.0 exactly. `alpha=0` zooms 1.9x and pushes most of the scene out of frame.
- Top = left: row error -0.17 px median, versus +6.6 px with the cameras swapped.
- The cameras converge by about 21.5 deg, so the right view is strongly foreshortened. Dense passive SGBM on smooth
  surfaces covers under 10% of the image (depth consistent at 187 mm on a keyboard). The scanner measures with the
  laser lines, not passive stereo, so this is expected and not a calibration problem.

## Laser triangulation on the DGX (`cloudclean/capture/metroy/stripes.py`)

1. Rectify both views with the factory calibration (`stereoRectify(..., flags=0)`).
2. Stripe centres by **Steger's ridge detector** (Gaussian sigma 2 px, Hessian normal, zero crossing of the derivative
   along the normal, evaluated on each rectified row), linked into tracks and smoothed along the stripe. Four rows are
   trimmed at each track end, where a stripe that ends (shadow, occlusion) pulls its centre.
3. **Partner by laser line identity** (the `metroExtra.bin` maps): the line the right view confirms along a whole
   track wins, a two-way check with the reverse maps, one right centre per left centre, depth within the calibrated
   190-450 mm.
4. Triangulate the measured centre pair through `Q`, then a neighbourhood agreement filter.

Validated 2026-09-21:

| Check | Result |
|---|---|
| flat board, 5 frames | RMS **31-40 um** (random 19-28, systematic 23-31); 330-360 um with the old 3-pixel parabola |
| points of a board frame on the board or on the one real surface behind it | 99.4-100 % |
| recessed panel of the lidded box, 6 frames, both families | 103.6-104.0 x 74.1-74.2 mm, 55.7-55.9 mm up |
| same box with the previous order-preserving matcher | whole stripes 11-44 mm off (mis-paired) |
| synthetic box scene through the real rectification (tests/test_metroy.py) | 99.5 % of points within 0.1 mm, no mis-paired stripe |
| lid rim, 8 frames fused | 59.9 mm up (calipers 62), outer length 121.9 mm (calipers 120.65) - **open**, see the handoff |
| the two families on the same spot (static scene) | disagree by ~0.16 mm - **open**, see the handoff |

The video uses SuperSpeed high-bandwidth isochronous transfers (43008-byte bursts, ~5.5 MB per URB), so a USBPcap
snaplen of 65535 truncates the frames. Capture the frames with V4L2 instead of from a pcap.

## Observed on Linux without the HID handshake

Streaming Y16 via plain V4L2 works once after plug-in and then times out, and the device USB-resets and
re-enumerates. The YUYV mode above needs no handshake, but the laser and exposure state do (see the start-up).

## V4L2 gotcha

`VIDIOC_QBUF` writes into the buffer struct it is given: read `timestamp` and `sequence` from a dequeued buffer
**before** re-queueing it, or both read back as zero.
