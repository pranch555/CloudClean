# Revo Metro internals (static analysis)

How Revo Metro V5.8.7.415 (`C:\Program Files\Revo Metro`) turns MetroY Ultra laser-stripe frames into points, and how
it tracks the scanner. Read-only static analysis on 2026-09-21: exports, strings, x86-64 disassembly (capstone) and
the CUDA PTX embedded in `RPAGPUAlgorithm.dll`. Revo Metro was not run. Read this alongside
`docs/metroy-protocol.md` (wire protocol) and `docs/metroy-handoff.md` (state of play).

Labels used below:

* **[verified]** read directly from code or data, or checked numerically against our own `metroExtra.bin`.
* **[inferred]** a strong reading of names, strings or config, but the code path was not traced end to end.
* **[unknown]** open question.

Scratch artefacts (disassembly listings, decrypted configs, extracted PTX) are in the session scratchpad and are
not needed to use this document. Nothing was written to the install directory.

---

## 1. Summary

1. **`metroExtra.bin` is decoded.** Each laser line has two records: one for the left camera and one for the right.
   The 6 coefficients are a **per-line polynomial between the two rectified images**:
   `x_other = c0*x^2 + c1*x*y + c2*y^2 + c3*x + c4*y + c5`, where `(x, y)` is a stripe point in rectified pixels.
   The left record maps left to right and the right record maps right to left **[verified]**. Across our file the
   left and right maps invert each other to 0.2 px RMS (0.004 px for the parallel lines). The single-line group
   uses a different form, a 3D surface in millimetres (section 4).
2. **The two 6-coefficient blocks (A and A') are the "original" and the "adjusted" laser planes.** The reconstruction
   calls the parser with `originLaserPlane = false`, so it uses **A'**, the second block **[verified]**. The two
   blocks differ only in the constant term, by 0.06 to 3.1 px.
3. **The 8 trailing doubles (E) are a 4-corner ROI polygon for each line, per camera**, in rectified pixels. Its
   corners are where the line lands at the near and far ends of the working range. The parser rasterises them into
   per-pixel "lowest and highest possible line index" lookup images **[verified]**.
4. **Line correspondence (GPU kernel `Point2PointMatchKernelH`, read from PTX) [verified]:**
   * For each left stripe point and each candidate line k, predict x in the right image with the L-to-R polynomial.
   * Look for a right stripe point on the same row with |dx| < 1.5 px and |dy| < 1.5 px.
   * Map that prediction back with every R-to-L polynomial and require exactly one left point within 1.5 px, and
     the same line index k.
   * The match must be unique. The 3D point then comes from **disparity through Q**, not from the laser plane.
   So the plane polynomials only decide *which* line a stripe is; precision comes from the stripe centres
   (Steger) and plain stereo.
5. **Laser modes:** register `0x3001` selects the line pattern. For MetroY Ultra (SDK camera type 0x24) the values
   are **7 = multi-line (cross), 9 = single line, 11 = parallel** **[verified]**. Revo Metro's own on-device help
   text says "7-multi-line, 8-single-line", but 8 is the value for other models (section 6).
6. **Each frame carries a "direction" byte:** 0 = cross family L, 1 = cross family R, 2 = single line, 4 = parallel,
   15 = fill-light/white frame. The recon maps it to a `metroExtra` group: 0 and 1 go to groups 0 and 1, 2 to the
   group with 1 line (group 3), 4 to group 2 **[verified]**. Register `0xb10` programs the frame sequence, for
   example `0xb10 5 15 0 1 2 4` for MetroY **[verified string]**. **Where this byte comes from in the USB video
   stream is unknown.** This is the most important open item for capturing on the DGX (section 6.3).
7. **In laser modes, tracking is marker-only.** The laser presets for model `0L8` (our scanner) contain only
   `marker` and `globalMarker` tracking modes **[verified config]**. Geometry/ICP alignment exists in `cspcpp.dll`
   (the "feature" mode), but `param.json` lists it only for the non-laser mode.
8. **The config files are only obfuscated.** Every byte is XORed with `"RevoPoint"[i % 9]` and with
   `"RevoScan"[i % 8]` (`RevoMetro.exe` at 0x140627060) **[verified]**. Decrypted values are quoted below.
9. **Bonus: Revo Metro's own per-frame results can be read.** A project's `cache/frames.dataset` is an SQLite
   database with a renamed header. It holds each frame's points (sensor frame, mm), each frame's 4x4 pose, and its
   markers (section 8). That gives ground truth for validating CloudClean frame by frame, without running Revo
   Metro.

---

## 2. Configuration files

### 2.1 Obfuscation [verified]

`RevoScanConfig::readLaserParamConfig` (and the other readers) call `readAll()`, then the function at
`0x140627060`, then `QJsonDocument::fromJson`. That function does this, with keys built from the literals
`"RevoPoint"` and `"RevoScan"`:

```python
def revo_decrypt(b: bytes) -> bytes:
    k1, k2 = b"RevoPoint", b"RevoScan"
    return bytes(c ^ k1[i % 9] ^ k2[i % 8] for i, c in enumerate(b))
```

`paramCarriedbyPacket.json`, the AppData `revoscan.cfg`/`partSetting.json` and `cameraparam/brightnessGainMap.json`
are stored in plain text. Every other file under `config/` is XORed as above.

### 2.2 Files relevant to laser scanning

Model code: our serial `D26519649H6U10L88` contains `0L8`. `0L8` appears in `supportedDeviceSerial.json`, the
`calibration/0L8` folder and every laser parameter table, and it is the only laser entry these files carry for
this device. Revo Metro records the device as "MetroY Ultra" (`revoscan.cfg`) **[inferred mapping 0L8 = MetroY Ultra]**.

**`sdkConfig_win.ini`** (decrypted, verbatim) is read by `RPALaserRecon` through `IniFile`. The defaults compiled
into the DLL, used when a key is missing, are in brackets:

```ini
[marker]
ellipseAb=0.5        ; [0.5]
fitAvgError=0.3      ; [0.5]
fitMaxError=0.5
fitAvgMaxError=0.5   ; [1.0]
zoomInScale=3        ; [3.0]
                     ; minBrightness [30] (not in file)
[markerMatch]
epipolarError=1.0    ; [0.5]
markerDiameter=3     ; [3.0]
[algSdk]
procThrdCnt=8
isGetPc=1
isGetMarker=1
dropGap=-1
isPrintTime=0
isOrderCallback=0
isLRPairDrop=0
isAutoAdjustCalList=0
algProcAudit=1
isSaveImg=0
auditGap=10
algImgPath=./algTest/
useSecondMatch=1
useAdjustLaserPlane=1   ; [1]
                        ; stegerSize [11] (not in file)
```

**`crosswire_param.json` / `parallel_line_param.json` / `single_line_param.json`**, entry `"model": "0L8"`.
Each file has `accuracymode` and `fastmode`, and each of those has `marker` and `globalMarker` lists of presets
named `general`, `metro_dark` and `metro_reflect`. The values are the same across these modes. Cross, `general`:

```json
{"name": "general", "histogram1": 230, "histogram2": 280, "histogram3": 330, "histogram4": 380,
 "zmin": 200, "zmax": 400, "exposure": 200, "gain": 1, "gainSteps": 5, "zminRange": 200, "zmaxRange": 400,
 "threshold": 10, "multiFrameFusion": 0, "startDepthAutoExpose": false, "ROIHeightScale": 0.7,
 "ROIWidthScale": 0.7, "useDefaultROIArea": false, "laserLuminance": 44, "laserLuminanceMax": 51,
 "fillLightLuminance": 30, "fillLightLuminanceMax": 58, "fillLightLuminanceMin": 47,
 "previewLaserWeightMax": 8, "previewExtractWeight": 2, "targetQuality": 8}
```

| preset (0L8) | exposure | gain | laserLuminance / Max | fillLightLuminance | targetQuality |
|---|---|---|---|---|---|
| cross `general` | 200 | 1 | 44 / 51 | 30 | 8 |
| cross `metro_dark` | 1000 | 2 | 193 / 255 | 3 | 18 |
| cross `metro_reflect` | 800 | 1 | 90 / 204 | 6 | 18 |
| parallel `general` | 200 | 2 | 27 / 51 | 30 | 8 |
| parallel `metro_dark` | 1000 | 3 | 179 / 255 | 2 | 18 |
| parallel `metro_reflect` | 800 | 1 | 204 / 204 | 6 | 18 |
| single `general` | 200 | 1 | 4 / 51 | 30 | 8 |
| single `metro_dark` | 1000 | 1 | 80 / 255 | 5 | 18 (previewLaserWeightMax 12) |
| single `metro_reflect` | 800 | 1 | 28 / 204 | 6 | 18 (previewLaserWeightMax 12) |

All presets: `zmin 200`, `zmax 400` (mm). Parallel uses histogram 205/260/330/380. The observed session wrote
`0x911 200`, which is the cross `general` exposure.

**`laser_param_config.json`** (marker extraction and matching for laser mode; `param_group_1`; Chinese
descriptions translated):

| | high_accuracy | standard_accuracy | meaning |
|---|---|---|---|
| `min_brightness` | 30 | 20 | minimum grey value, 0-255 |
| `saturation` | 0.5 | 0.3 | a candidate blob must fill > saturation x (w x h) of its box |
| `ellipse_ab` | 0.5 | 0.2 | minor/major axis ratio; below this the marker is dropped |
| `fit_error` max / avg | 0.5 / 0.3 | 1.0 / 0.5 | ellipse edge-fit error per edge point |
| `epipolar_error` | 1.0 | 2.0 | px; after rectification matched markers must be on the same row |
| `radius_error_pixel` | 1.5 | 4.0 | px; left/right radius difference must be < 2x this, and it bounds the 3D radius error |

**`algoParameters.json`** (decrypted). Fusion: `extractCredibleRatio 1.0`, `extractPointFrameCount 3`,
`extractIsolation 0.005`, `minPointPitch 0.1`, `metroYFastMinPointPitch 0.05`. Markers: `minAb 0.4`,
`maxMeanFitError 0.1`, `maxMaxFitError 0.2`, `minWidthHeight 5`, `epipolarError 1.5`, `diameter 5`,
`abThreshold 0.4`. Flying-point removal: `isolationRate 0.002`, `neighborCount 5`, `angleDegree 40`,
`maxRemoveRatio 0.05`. Bilateral denoise: `bilateralNeighborCount 16`, `denoiseStrength 0.72`,
`maxDenoiseMoveRatio 0.35`. Laplacian detail re-injection: `detailSharpenRadiusRatio 2.5`,
`detailSharpenStrength 0.35`.

**`brightnessGainMap_MetroY.json`** (decrypted, install copy). The copy on the scanner (`/data/brightnessGainMap.json`)
differs and takes precedence per device **[inferred]**:
`lineGeneral` gain 1..10 -> brightness 37,18,13,10,8,6,6,5,4,3; `lineDark` gain 1..5 -> 28,14,9,7,6;
`lineReflect` gain 1..5 -> 28,14,9,7,5.

**`param.json` 0L8** (non-laser full-field mode **[inferred]**): `feature` (general, metro_dark), `marker`,
`globalMarker`. `general`: exposure 3000, gain 1, gainSteps 10, zmin 200, zmax 400, threshold 5.

**`calibration/stepInfo/.../metroy_ultra*.json`**: factory re-calibration steps (heights 210..333 mm and exposures)
plus laser brightness per step (`k_crossLaserLight` 90..145, `l_parallelLaserLight` 170..270,
`m_singleLaserLight` 15..40). **`calibration/0L8/calibration.json`**: coded target board, 13x13 cells of 20 mm,
12-bit CCT codes, `CenterCricleR 1.5`.

**AppData `revoscan.cfg`**: firmware `v2.9.10.20260520` for our serial; fused point spacing `0.15`.

---

## 3. DLL map

All DLLs are MSVC x64 with RTTI and exported C++ symbols. The PDBs are not shipped. Build paths are
`D:\jenkins_workspace\RPAlgo\<Module>`.

| DLL | Role | Key API / facts |
|---|---|---|
| `3DCamera.dll` (66 MB) | Revopoint camera SDK (`cs::Camera`): USB/HID, pre-ISP register writes, frame pairing, calls the algorithm | `laserScanLineModeChange/Start`, `laserParamCollection`, `setSupportRadiusMarkers`, `getPairedFrame`. Reads `/data/camparam/camparam.yaml` and `/data/camparam/metroExtra.bin` (also `/data/camparam_blue/...` for the blue-light mode) |
| `3dcameraAlg.dll` | glue from SDK to algorithm | `getMarkersPointsGPUPtr(LASER_LINE_TYPE, SINGLE_LASER_DIREC)`, `createMarkersPoints(GPU)`; classes `IRMarkersPoints{Mutile,Paralle,Single}` and `...GPU{Mutile,Parallel}`. Log: "yamlDataSize", "load camera params in multiple line failed" |
| `RPALaserRecon.dll` | laser point cloud pipeline | `Multi/Parallel/SingleLaserReconSyncPipeline`, `MultiLaserReconPipelineGPU`; `init(MetroCameraParams, AlgoParams, const char* iniPath, int)`, `addImage(Mat L, Mat R, int direction, PointData&, vector<MarkerData>&, ReconType)`, `setMarkerSupportRadius(vector<double>)`, `processLaserPlaneAdjustment`, `getCloudUsingDisparity` |
| `RPALineDetect.dll` | stripe centre extraction | `LineDetect::init(LineDetectParam&)`, `getCenterList(Mat, [int], MemoryPool*, LDCenterResultPool&)`, `RPALineDetectUtils::calStegerLDInfo(float, LDInfo&)`, `getCudaStegerRes`, `FftFilter::createDirectionalFilter`. Steps: `PreProcessingStep`, `GetCenterStep`, `PostProcessingStep`; debug dumps `1_adaptiveThresholdImage`, `2_maskImage`, `6_overSizePointsCenters`, `7_overSizeMerge*`, `8_FinalResultPoints` |
| `RPALaserMatch.dll` | CPU line correspondence | `LaserMatch::init(LaserMatchParam&)`, `getLaserMatchPoints(6 x Mat, vector<Point2f> L, vector<Point2f> R, MultiLaserParam, bool, MultiLaserMatchResult&)`, `arrangeAndCombineSecondMatch`, `getTrianglePoints(vector<float> plane, pts, fx, fy, cx, cy, BaseLineType, out)`. Steps `LaserMatchContinuityStep`, `LaserMatchIndependentStep` |
| `RPAGPUAlgorithm.dll` | CUDA versions (plain PTX for sm_75/86/89/90 embedded, LZ4) | `StegerKernel`, `GatherStegerKernel`, `AdaptiveThreshIntegralKernel`, `RemapKernel`, `Point2PointMatchKernelH/V`, `PointToPointSecondMatchKernelH/V`, `LaserTriangleMatchKernelV`, `FailedResultMatchKernelH/V`, `ArrangeAndCombineSecondMatchKernel`. Match methods named `"laser_triangle"` and `"point_to_point_second_match"` |
| `RPAModelIO.dll` | file formats | `CameraParser::loadMetroExtraParam`, `parseMetroExtraGroup{,V1,V2,V3,V8,V9}`, `parseMetroRoiData`, `writeMetroExtraParam`, `loadCameraParams` |
| `RPACalibLaser.dll` | in-app laser (re)calibration | `calibrateLaserPlane`, `calibrateSingleLaserPlane`, `fastAdjustLaserPlane[WithBoardImage]`, `adjustLaserPlaneRealTime`, `laserPlaneAccuracyCheck`, `writeMertroBinTest`. Logs: "multi laser type: UV2UV / QUAD / UV2UV_SEGMENT", "paramType: X_YZ / Y_XZ", "functionL2R Left {}", "functionR2L", "Segment {}: m_f difference ... replacing equation" |
| `RPALaserAutoExposure.dll` | auto exposure from stripe brightness | "Avg Gray", "WeightMap: ON (sigma=" |
| `RPAMarkerDetect.dll`, `markerDetectionForMetroX.dll` | retro-reflective marker detection | `MarkerDetect(int w, int h)`, `NormalEquationEllipseFitter`, `fourQuadrantDetection`, `AccuratePosition`, `calcFitError`. Steps `locateRoi`, `filteredImage`, `pixelGradient`, `blockByBrightness`, `filterValidBlocks` / `filterByMaxWidthHeight`, `recedingWater`, `blockCenterFilter` |
| `RPAMarkerRecon.dll`, `markerMatchByEpipolar.dll` | stereo marker matching and 3D reconstruction | `MarkerRecon(w, h)`, `getClosestRadius`, `findConnectedMarkerComponent`, `Plane3D::fitPlane` |
| `RPAFusionLandmark.dll`, `RPARegisterLandmark.dll`, `markerAlignByTriangle.dll`, `markerFusionByTriangle.dll` | marker-based tracking and global marker map | `FusionLandmark::addFrame(landmarks, pose, ...)`, `alignFrame`, `GlobalMarkers::fuse/deIntegrate`; `RegisterLandmark::align/alignNeighbors/alignExcludeOutlier/optimizeByProjectError...` |
| `cspcpp.dll` (`AS::SLAM`) | older SLAM back end: fusion, ICP/overlap alignment, marker fusion, texture | keys `needRobustICP`, `useRPAlgoICP`, `doICPAfterAlign`, `alignOverlapRatio*`, `markPointThreshold`, ... |
| `RPAFusionTracker.dll`, `TrackerScanner.dll` | for the optical-tracker product (Trackit), **not used by MetroY** | `FusionTracker::PushFrame`, GPIO/`0x487` soft trigger |
| `RPAPipeline.dll` | frame/project containers | `FrameIndex(int, int, bool, FrameFlag)`, `FrameFiles::Info(...)` |
| `RPACamera.dll` | camera parameter containers (`camparam.bin`) | `CameraParamsContainer`, `MonoCameraParam`, `st_map_param` |

---

## 4. `metroExtra.bin` format and meaning

### 4.1 Layout [verified: `CameraParser::loadMetroExtraParam(vector<char>)`, RPAModelIO RVA 0xaf60]

```
0x00  18 bytes   reserved (zeros)                       -> MetroCameraParams+0x788
0x12  14 bytes   ASCII timestamp "YYYYMMDDhhmmss"       -> +0x79a   ("20260902100715")
0x20  u8         extraParamVersion   (ours: 7)          -> +0x7a8
0x21  u8         group count         (ours: 4)          -> +0x7a9
then, for each group (a 0x28-byte struct in memory):
  +0  u16  header length (10; if != 10 the extra bytes are skipped)
  +2  u16  payload length in bytes
  +4  u8   ?  (ours 1,1,2,0; not read by the V3 parser; matches laser type 1=cross, 2=parallel, 0=single [inferred])
  +5  u8   lines per group (17,17,15,1)
  +6  u8   values per record (20)
  +7  u8   group index (0..3)
  +8  u8   ?  (2; not read by V3; probably "2 cameras" [inferred])
  +9  u8   element type: 0=int8, 1=uint16, 2=int32, 3=float32, 4=float64 (ours 4)
  payload: 2 x lines records x values per record elements
```

Our earlier notes were one byte off: the prefix is 18 zero bytes plus 14 date bytes (32 bytes), and `07 04` is
*version, group count*.

`parseMetroExtraGroup` dispatches on the version: 1 -> V1, 2..4 -> V2, **5..7 -> V3**, 8 -> V8, 9 -> V9.
`RPACalibLaser`'s re-calibration requires version 8 or 9 (it logs "error extraParamVersion" otherwise). Our
factory file is version 7, so in-app re-calibration would rewrite it in a newer format **[inferred]**.

### 4.2 V3 record semantics [verified: `parseMetroExtraGroupV3`, RPAModelIO 0x18000e8d0]

With `R = values per record = 20`: `roi = 8 if R >= 8 else 0`, and `ncoef = R - roi - 6 = 6`.

* Records `0 .. N-1` go to **leftPlaneParam[k]**; records `N .. 2N-1` go to **rightPlaneParam[k]**. Record `k` and
  record `k+N` are the same laser line seen from the left and from the right camera. They are not two depths.
* The coefficients are taken from offset 0 (**A**) when the last argument `originLaserPlane` is true, and from
  offset 0x30 (**A'**) when it is false. **Every call in `RPALaserRecon.dll` passes false** (4 call sites, e.g.
  0x1800269c2), so live reconstruction uses **A'**. `RPACalibLaser` logs "parseMetroExtraGroup originLaserPlane:{}"
  and uses both.
* The last 8 values (**E**) are 4 (x, y) points, truncated to int, filled as a convex polygon (`cv::fillConvexPoly`,
  value 255) into a per-line mask for that camera. Left and right polygons of one line share y values, so they are
  in **rectified** coordinates. The two long edges are the line's image at the far and near ends of the calibrated
  depth range. For our file, triangulating the left/right corner pairs gives Z of about 218..430 mm.
* The parser then builds four `CV_8U` images at the camera resolution, initialised to 25 (0x19, meaning "none"):
  **min line index (left), max line index (left), min (right), max (right)** of all line polygons covering each
  pixel. These are the "line index from position" priors.

### 4.3 What A (A') is: UV2UV polynomial for groups 0-2 [verified numerically and from the PTX]

The GPU matching kernel reads the per-line parameters as 6 floats `p` (`k_segment_param_left/right`, indexed
`[group][line][row segment]`) and evaluates, at a stripe point (x, y) in rectified pixels:

```
x_pred = p0*x*x + p1*x*y + p2*y*y + p3*x + p4*y + p5
```

A left record maps a left-image point on line k to the x of the same line in the right image, on the same row.
A right record maps right to left. The large negative "slope" (about -1.5 in x) is correct. Along a row, a fixed
laser plane moves in opposite directions in the two images as depth changes. The polynomial is only meaningful on
the line itself, so its terms are not individually interpretable.

Check on our file: we sampled points inside each line's ROI, mapped L->R with the left record, then R->L with the
right record. The round-trip error is 0.22 px RMS for the cross groups and 0.004 px for the parallel group (A and
A' alike). This is what the calibration log "multi laser type: UV2UV" refers to.

`RPACalibLaser` also knows `UV2UV_SEGMENT`. There the image is split into row bands, with one polynomial per band
(`Segment {}: x range [..]`; the kernel computes `segment = row / (height / nSegments)`). Version 7 has one
function per line **[inferred: nSegments = 1]**.

### 4.4 Single-line group (group 3, 1 line, 2 records): 3D surface [verified numerically + CPU code]

For this group the records are **not** inverse UV maps (the round trip fails). They are an explicit quadric
surface in the **rectified camera frame, in mm**, with horizontal baseline ("paramType X_YZ"):

```
X = p0*Y^2 + p1*Y*Z + p2*Z^2 + p3*Y + p4*Z + p5
```

The left record is in the left rectified frame (p5 = 47.93). The right record is the same surface in the right
rectified frame (p5 = -80.57). The difference, 128.50, **equals the baseline**, and every other coefficient is
identical. The ROI x range 654..843 px corresponds to Z of about 420..219 mm with this surface. That is consistent
with the other groups.

The CPU triangulation `LaserMatch::getTrianglePoints` (RPALaserMatch 0x18001f490, verified) intersects a pixel ray
with this surface:

```
xn = (u - cx)/fx ; yn = (v - cy)/fy ; if baseLineType == 1: swap(xn, yn)
a = p0*xn^2 + p1*xn + p2 ;  b = p3*xn + p4 - yn ;  c = p5
solve a*Z^2 + b*Z + c = 0 (|disc| < 1e-5 -> Z = -b/2a; else take the root in [0, 1000] mm, the smaller if both)
P = (xn*Z, yn*Z, Z) ; if baseLineType == 1: swap(P.x, P.y)
```

The same routine in its unswapped form is `Y = f(X, Z)` ("Y_XZ"), used for scanners with a vertical baseline.
Another CPU helper (0x1800047d0) uses it to project a point from one camera into the other by subtracting
`P[0,3]/fx` (the baseline) and reprojecting.

### 4.5 A versus A'

A' = A except `c5`. A' - A is +0.12..0.75 px (group 0), +2.5..3.1 (group 1), +1.8..2.1 (group 2) and +0.056 mm
(group 3). `RPACalibLaser` has `fastAdjustLaserPlane*` and "Segment {}: m_f difference = ... replacing
equation" (m_f is the constant term). In addition, `RPALaserRecon` runs `adjustLaserPlaneThread` /
`processLaserPlaneAdjustment` when `useAdjustLaserPlane=1`. So Revo Metro nudges the constant term **online**
from matched pairs as well **[inferred from names; not traced]**. For our own pipeline, **use A'**, as Revo Metro
does.

### 4.6 Reference decoder

```python
import struct, numpy as np

def load_metro_extra(path):
    d = open(path, "rb").read()
    date, ver, ngroups = d[18:32].decode(), d[32], d[33]
    off, groups = 34, []
    for _ in range(ngroups):
        hlen, plen = struct.unpack_from("<HH", d, off)
        ltype, nlines, nval, gidx, _two, dtype = d[off + 4:off + 10]
        assert dtype == 4 and nval == 20            # float64, 20 values (true for our file)
        rec = np.frombuffer(d, "<f8", count=plen // 8, offset=off + hlen).reshape(-1, nval)
        n = nlines
        groups.append(dict(index=gidx, type=ltype, lines=n,
                           left=rec[:n, 6:12], right=rec[n:, 6:12],           # A' (what the recon uses)
                           left_orig=rec[:n, 0:6], right_orig=rec[n:, 0:6],   # A
                           roi_left=rec[:n, 12:].reshape(n, 4, 2),
                           roi_right=rec[n:, 12:].reshape(n, 4, 2)))
        off += hlen + plen
    return date, ver, groups

def uv2uv(p, x, y):
    return p[0]*x*x + p[1]*x*y + p[2]*y*y + p[3]*x + p[4]*y + p[5]
```

---

## 5. Stripe detection and line correspondence

### 5.1 Stripe centres [verified names, inferred details]

The pipeline works per row on **rectified** images. `LaserReconPipeline` calls `cv::stereoRectify`,
`initUndistortRectifyMap` and `remap`; the GPU uses `RemapKernel`. The sequence is:

1. Optional directional FFT filter (`FftFilter::createDirectionalFilter(w, h, angle[, width])`).
2. Adaptive threshold with an integral image (`AdaptiveThreshIntegralKernel`), giving a mask.
3. **Steger** centre extraction. The Gaussian derivative kernel is built by `calStegerLDInfo(stegerSize)`
   (default `stegerSize` 11). The PTX constants 0.5 and 4.0 are the usual Steger subpixel bound (|t| <= 0.5 px)
   and the Hessian eigenvalue discriminant.
4. Gather/interpolate (`GatherStegerKernel`, `InterpolateKernel` with `LaserRoi`), and over-size point
   splitting/merging.

This is the precision step our `laser_scan2.py` is missing: its unweighted 3-pixel parabola per row gives 270 µm
RMS.

### 5.2 First match: `Point2PointMatchKernelH` [verified from PTX, sm_89 module]

"H" means a horizontal baseline (same row); "V" is the transposed variant. Inputs are the left and right centre
lists (x, y, valid) with per-row index lists, the image width and height, the number of lines N in the group,
the group index g, the number of row segments S, and `k_camera_q` (the 4x4 Q).

```
for each left centre i (x=xL, y=yL, row r, seg = r / (H/S)):
    hits = 0
    for k in 0..N-1:
        xr = UV2UV(left[g][k][seg], xL, yL);  skip unless 0 < xr < W and 0 < yL < H
        for each right centre j on row r (valid):
            if |xr - xR_j| < 1.5 and |yL - yR_j| < 1.5:
                # back-check
                count = 0
                for m in 0..N-1:
                    xl = UV2UV(right[g][m][seg], xr, yL)
                    if 0 < xl < W: for each left centre l on row r:
                        if |xl - xL_l| < 1.5 and |yL - yL_l| < 1.5: count += 1; last_m = m
                if count == 1: hits += 1; line = k; line_back = last_m; store xR_j, yR_j
    if hits == 1 and line == line_back:
        out.line[i] = line
        X = Q @ [xL, yL, xL - xR, 1];  point = X[:3] / X[3]         # plain disparity triangulation
```

Tolerance is **1.5 px** in x and y (a PTX constant, not configurable). Uniqueness is required in both directions.
The laser model only assigns the line index; **3D comes from disparity through Q**, the same `Q` as in
`camparam.yaml`.

### 5.3 Second match and fallbacks [inferred from names/signatures]

With `useSecondMatch=1`, `PointToPointSecondMatchKernelH` matches centres that the first pass left unmatched. It
uses the ROI polygons (`LaserRoi`) and the min/max line-index masks from section 4.2, then
`ArrangeAndCombineSecondMatch`. `FailedResultMatchKernel` plus `StatGroupsKernel`/`FindMajorIdKernel` re-label
failed groups by majority vote over connected centres. On CPU the equivalents are `LaserMatchContinuityStep`
(continuity along a stripe) and `LaserMatchIndependentStep`. The single-line group uses `"laser_triangle"`
(`LaserTriangleMatchKernelV` / `getTrianglePoints`, section 4.4).

---

## 6. Laser modes, frame types and registers

### 6.1 Line pattern: register `0x3001` [verified: 3DCamera 0x181242120]

The SDK stream formats are `PMLD` (0x104, multi-line), `PSLD` (0x105, single line) and `PPLD` (0x106, parallel).
`laserScanLineModeChange` maps them to mode 0, 1 and 2 and calls `laserScanLineModeStart(mode)`, which writes
the register with 3 retries.

| mode | default models | camera types 0x1d/0x1e/0x24 (MetroY, MetroY Pro, **MetroY Ultra**) |
|---|---|---|
| 0 multi-line / cross | `echo s 0x03001 7` | **7** |
| 1 single line | `echo s 0x03001 8` (9 for types 0x17, 0x20-0x22) | **9** |
| 2 parallel | `echo s 0x03001 9` (11 for type 0x1a/0x22; unsupported on 0x20/0x21) | **11** |

`0x3001 12` is the speckle mode (types 0x20/0x21 only). Camera type 0x24 = "MetroY_Ultra" comes from the
`cs::getCameraTypeName` table (types >= 10 index the name table at type-5). The on-device help string says
"设置激光线扫模式, 7-多线、8-单线" ("set laser line-scan mode, 7 multi-line, 8 single-line"), which is the default
mapping. Our observed start-up sequence writes `0x3001 7`, which is the cross mode.

### 6.2 Other registers [verified: help table embedded in 3DCamera.dll + setter log strings]

| reg | meaning (help text / setter) | our observed values |
|---|---|---|
| `0x48d` | laser brightness 0-255 ("设置激光器亮度值"); variants `0x48d v 1/2/3` per laser | 255 |
| `0xb04` | laser LED enable (1) / disable (0) | 1 / 0 |
| `0x910` | depth **frame time** (µs) ("设置深度帧时间"); SDK uses 6600 (<fw 2.9.45) or 6200 | 7000, 8000 |
| `0x911` | depth **exposure time** (µs) ("设置深度曝光时间") | 5000, 200 |
| `0x903` | depth **gain**, written as `0x%02X`, 0x10 = gain 1 [gain x16 inferred] | 0x10 |
| `0x912` | auto-exposure mode 0 off, 1 frame rate, 2 quality, 3 near | |
| `0x913` | depth ROI x y w h in % (0..100) | |
| `0x914` / `0x916` | HDR mode (0 off, 1 reflective, 2 dark, 3 mixed) / HDR level params | |
| `0x707` | depth range min max (mm) ("set depth range min = %d, max = %d") | 200 600, 230 630 |
| `0x701` | resume IR stream | 1 |
| `0x702` | resolution w h | |
| `0x705` / `0x706` | background threshold [0-40] / gradient threshold [0-1000] | |
| `0xa04` / `0xa05` | multi-frame fusion (double exposure) / fringe pattern | |
| `0xb07` | IR fill-light brightness 0-255 (`0xb07 v 0/1` selects blue/red fill type) | 1, 30, 45 |
| `0xb08` | **laser exposure times** ("set laser times failed" / "set laser exp times failed") | 217 |
| `0xb06` | IR fill-light exposure times | |
| `0xb1b` | used instead of `0x48d` for stream format 0x10b | |
| `0xb10` | **frame-type sequence**: `0xb10 5 15 0 1 2 4` (MetroY family "L R O P W"), `0xb10 2 15 4` ("P W") | |
| `0x3010 0x17 1/0` | enter/leave laser line group | |
| `0x481`, `0x484`, `0x487` | soft trigger | |
| `0x488` | return-trip scan on/off | |
| `0x48b` | digital zoom | |
| `0x48c` | enter (1) / exit (0) laser mode for the PRCZ stream format | |
| `0x4a0 ...` | raw byte commands to the laser driver (e.g. "set near/far camera laser frequency", cross-line exposure time) | |
| `SetCmd s 0x101..0x104` | depth stream format change (0x102), texture-mode pair (0x103), RGB transfer switch (0x104) | `0x103 0` |

`laserParamCollection` applies a preset in this order: `0x910` frame time, `0x911` exposure, gain (property),
`0xb07` fill light, `0x48d` laser brightness, `0xb08` laser exposure times, `0xb06` fill-light exposure times.

### 6.3 Frame "direction" and how the families alternate [verified mapping; source unknown]

`markersPointsCPU::markersAndPointsProcTask` (0x18134c260) reads a signed byte at offset **0x12 of the frame info
struct** (`frame->vfunc_0x50()`). It accepts values in bitmask 0x8017, i.e. **{0, 1, 2, 4, 15}**; anything else
logs "CPU direction(%d) of frame incorrect". The byte is passed as `addImage(..., int direction, ...)`.
`LaserReconSyncPipelineCPUImpl::addImage` (0x180034280) maps it to a `metroExtra` group:

| direction | letter in `0xb10` log | pattern | metroExtra group used |
|---|---|---|---|
| 0 | L | cross, family 1 | group 0 |
| 1 | R | cross, family 2 | group 1 |
| 2 | O | single line | last group with `lines == 1` (group 3) |
| 4 | P | parallel | group 2 if there are more than 3 groups, else the last |
| 15 | W | fill light / white (marker frame) [inferred] | none |

So in cross mode, frames alternate between the two diagonal families. The SDK learns each frame's family from
metadata; it does **not** infer it from the image. **[unknown]** Where the byte lives in the USB stream: UVC
payload-header extension, the UVC metadata node, or bytes embedded in the frame. The SDK's network/file frame
format (`playFramFileProc`) has "frame type", "frame flag", timestamps, format, w, h and extension fields (IMU
quaternion, key events), but the USB path was not traced. Suggested check on the DGX: capture the UVC
metadata node together with frames in cross mode and see which field toggles 0/1 each frame. Our frames
alternate families, so the value can also be verified against the image by fitting stripe slopes.

---

## 7. Tracking in laser mode

* **Markers only [verified config].** The 0L8 cross/parallel/single presets exist only under `marker` and
  `globalMarker`. `param.json`, the non-laser full-field mode, also has `feature` (geometry) tracking.
* Detection runs **in the same laser frames**. `RPALaserRecon` owns a marker thread (`markerThread`,
  `initMarkerDetectionAndMatch`) and returns `vector<MarkerData>` with each point cloud (`isGetMarker=1`). The
  algorithm, from the names: ROI -> gradient -> blocks by brightness -> ellipse fit
  (`NormalEquationEllipseFitter`, four-quadrant check, fit error) -> `AccuratePosition`. Thresholds are in
  `sdkConfig_win.ini` / `laser_param_config.json` (section 2.2).
* **Stereo matching of markers:** same rectified row within `epipolarError` (1.0 px), similar radius
  (`radius_error_pixel`). Then 3D via `MarkerRecon` with a local plane fit for the normal. Marker size prior:
  `markerDiameter=3`. `setSupportRadiusMarkers(vector<double>)` accepts a list of allowed radii
  (`getClosestRadius`).
* **Registration:** `FusionLandmark::alignFrame/addFrame` keeps a global marker map (`GlobalMarkers::fuse`, with
  per-frame de-integration). `RegisterLandmark` does triangle-based correspondence (`markerAlignByTriangle`,
  `alignNeighbors`, clustering coefficients), outlier rejection and refinement by projection error
  (quaternion/axis-angle).
* ICP exists (`cspcpp.dll`: `needRobustICP`, `useRPAlgoICP`, `doICPAfterAlign`) for feature mode and for
  aligning finished scans (`AlignPointCloud`, "ICPRegistration"). No evidence was found that it is used for
  frame-to-frame tracking in laser mode **[inferred]**.
* Per-frame data confirms this (section 8): our cross-mode project stores 4..8 markers per frame, each with
  position, normal, a quality value and `3.0` (size).

Implication for CloudClean: on a hand-held MetroY in laser mode, **retro-reflective markers are the tracking
signal**. Frame-to-model ICP on 17 sparse stripes is a fallback at best. A turntable plus markers on the table
would work with the same machinery.

---

## 8. Revo Metro's own per-frame output (`frames.dataset`) [verified]

`Projects/<name>/data/<guid>/cache/frames.dataset` is an **SQLite 3 database whose 16-byte magic was replaced with
`"RPAlgo_file_sys\0"`**. Copy it and patch the first 16 bytes back to `"SQLite format 3\0"`; never touch the
original. Tables:

```sql
CREATE TABLE FrameInfo (id INTEGER PRIMARY KEY, version INT, seqId INT, droped INT, gapRGB INT,
                        pose BLOB, optimized INT, markerCount INT, markerData BLOB, bin BLOB)
CREATE TABLE FrameRGBD (id INTEGER PRIMARY KEY, version INT, type INT, width INT, height INT,
                        data BLOB, weights BLOB)
CREATE TABLE FrameColor (...)   -- empty in our laser project
CREATE TABLE FrameVersion (version INTEGER PRIMARY KEY)
```

In project `Project09092026145746` (guid 063bd84d..., 13 094 frames, cross mode, `scan_param.scan_mode 1`):

* `FrameRGBD.data` = `u64 n` + `n x float32 (x, y, z)`: that frame's laser points **in the sensor frame, mm**
  (for example about 3 400 points, Z 205..367 mm).
* `FrameInfo.bin` (all other columns are NULL):

  | offset | content |
  |---|---|
  | 0 | u32 (frame counter / 1) |
  | 4..7 | bytes `01 04 00 02` |
  | 8 | u32 0 |
  | 12 | u32 timestamp-like |
  | **16** | **4x4 float64 pose, row-major** (sensor to world, mm) |
  | 144 | marker block: 16-byte header, u32 byte length, u32 1, u32 0 |
  | **172** | **records of 8 float32: x, y, z, nx, ny, nz, quality, 3.0** |

This gives, for every frame Revo Metro captured, the exact point set it produced and the pose it solved. That is a
direct way to validate CloudClean's triangulation and tracking on the same data. `raw_preview.ply`, `fuse.ply`,
`mark.ply` and `marker_framework.mkf` are the fused results.

---

## 9. What this means for our pipeline

1. **Use `metroExtra` A' for line identity**, exactly as in section 5.2 (1.5 px, bidirectional, unique). Then
   triangulate by disparity with the factory `Q`. That replaces the order-preserving matcher in `laser_scan2.py`
   and removes marker blobs naturally, because blobs will not satisfy both polynomials.
2. **Per-frame family:** in cross mode, apply group 0 or group 1 depending on the frame. Until the metadata byte
   is found, try both groups per frame and keep the one with more unique matches. Cross families have opposite
   slopes, so a stripe-direction test separates them cheaply.
3. **Precision:** Steger centres with a Gaussian of about `stegerSize` 11 (sigma about 2 px) instead of the
   3-pixel parabola.
4. **Tracking:** marker-based. Use ellipse fit + epipolar match (1.0 px) + triangle registration, with
   `markerDiameter` 3.
5. **Validation:** compare against `frames.dataset` points and poses from an existing Revo Metro scan (section 8).

## 10. Open questions

* **Where the per-frame direction byte comes from on USB** (section 6.3). Without it, the cross families must be
  inferred from the image.
* Whether `useAdjustLaserPlane` changes the constant term enough to matter for precision
  (`adjustLaserPlaneRealTime`, not traced).
* The exact Steger kernel and thresholds (`LineDetectParam` / `PostProcessingConfig` contents). The PTX for
  `StegerKernel` is extracted and readable if needed.
* Header bytes +4 and +8 of a `metroExtra` group (laser type and camera count are the likely meanings, but V3
  does not read them).
* HID `05 05` codes 102/103/111/113 (`setPropertyExtension`?) were not mapped. The SDK builds them from enums,
  without strings.
* Units of marker record value 3.0 (diameter versus radius, mm) and of `markerDiameter`.
