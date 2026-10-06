# Revopoint turntable control

CloudClean drives Revopoint turntables like Revo Metro does. You can rotate by an angle, tilt the dual-axis
table, set the speed, stop, and run step-and-scan programs that pause and resume the live capture at each stop.
It works with a **simulated turntable** (always available) and with the real one over Bluetooth LE. The
simulated scanner can follow the simulated turntable, so you can try a whole turntable scan without hardware.

Code: `cloudclean/capture/turntable/` (protocol, drivers, program runner, manager), routes in
`cloudclean/web/routes_turntable.py`, tests in `tests/test_turntable.py`, hardware CLI in
`tools/turntable/turntable_cli.py`. The contract is Contract 4 in `docs/v3-plan.md`.

**Status (2026-10-06): validated on the user's dual-axis table.** CloudClean on the DGX Spark connected over
Bluetooth (Revo Metro closed on the laptop, which otherwise holds the table), turned +10 / -10 deg
(`+OK;` then `+OK,TURNANGLE=-287.79` after ~1.5 s) and tilted 5 -> 0 deg (`+OK;`, then `+OK,TOZERO;`; the
tilt reads 0.4 at level and -0.6 before: about half a degree of play). The version reply is
`+DATA=HW.V0.02,SW.V3.28;` (hardware, then firmware: the firmware is the second field, as Revo Metro reads it).
Speed range reply `+DATA=17.5,90.0,2.0;`, tilt range `+DATA=-30.0,30.0,0.3;`. Still open: which way a positive
angle turns seen from above (ask the user), and the large TA500 (never seen). The protocol was recovered from
Revo Metro V5.8.7.415 (below).

Labels, as in `docs/revo-metro-internals.md`:

* **[verified]**: read directly from Revo Metro's code or config, or seen in real traffic in Revo Metro's logs.
* **[inferred]**: a strong reading of the code, but not seen in real traffic.
* **[unknown]**: an open question.

The analysis was read-only. `RevoMetro.exe` was disassembled with pefile and capstone, and the logs and configs
were read. Nothing under `C:\Program Files\Revo Metro` was modified. No command was sent to the turntable, and
Bluetooth was used for passive scans only.

---

## 1. The user's hardware

| Item | Value | Source |
|---|---|---|
| Turntable | **Dual-axis turntable**, advertised name `REVO_DUAL_AXIS_TABLE`, address **E4:8F:80:46:12:43** | `%LOCALAPPDATA%\RevoMetro\config\partSetting.json` (`{"name": "REVO_DUAL_AXIS_TABLE", "type": "0", "uid": "e4:8f:80:46:12:43"}`), log line `--BLE_DEV -- : "REVO_DUAL_AXIS_TABLE" id: "e4:8f:80:46:12:43" type: 0` [verified] |
| Firmware | **3.28** | `DualAxisTurntable_LOG: DualAxisTable Version = 3.28` (logged on 09-09, 09-10, 09-14 and 09-24) [verified] |
| Revo Metro settings | step 10 deg, turnSpeed 50 s/rev, clockwise, tilt per rotation 1: -1, 2: 10, 3: -10, 4: 20, 5: -20 deg | `revoscan.cfg`, `parts.twoAxisTurntable` [verified] |
| Windows BLE scan | Intel Wireless Bluetooth adapter. Two passive scans (12 s: 44 devices, 20 s: 57 devices) found **no** `REVO_*` device | See below |
| DGX Spark | **Has Bluetooth**: `hci0` (MediaTek BT 5.4, USB 13d3:3630 IMC Networks), UP RUNNING, not rfkill-blocked, `bluetoothd` 5.72 active and enabled. User `dgx` reaches BlueZ over D-Bus (`bluetoothctl list` works). `bleak` is **not installed** in `~/code/CloudClean/.venv` | Read-only checks, 2026-09-24 |

The turntable did not advertise because **Revo Metro was holding the connection**. It connected at 11:57:10
today, has logged no disconnect since, and was still running during the scans. A BLE peripheral stops
advertising while it is connected, and this table accepts one central at a time. Listing the GATT services was
therefore skipped, as the brief requires. **Close Revo Metro (or disconnect the turntable in it) before
CloudClean can connect.**

The Spark can drive the table directly. It has a working adapter, and BLE reaches about 10 m, so it works
whenever the Spark is that close to the table (for example when the scanner is plugged into the Spark for
native capture). Only `bleak` is missing: `~/code/CloudClean/.venv/bin/pip install "cloudclean[turntable]"`, or
`pip install bleak`. If the Spark is out of range, there are two fallbacks: (a) a USB Bluetooth 5 dongle
on a USB extension cable near the table (BlueZ supports common Intel, Realtek and MediaTek dongles), or (b) run
the driver on the Windows laptop, which already runs `cloudclean bridge`. The simplest relay is a second CloudClean
instance on the laptop that serves only `/api/turntable/*`; the Spark UI would call it. It is not built because
the Spark has its own adapter.

---

## 2. Transport [verified]

* **Library.** Revo Metro links `simpleble.dll` (SimpleBLE C++ API). The imports are the IAT at `0x141abcf90`
  with thunks at `0x140b9e1d9`..`0x140b9e263`. Of these it uses only `write_command` (write *without* response),
  `notify`, `services`, `characteristics`, `uuid`, `mtu`, `connect`, `disconnect`, `is_connected`, the scan
  calls and the connect/disconnect callbacks. There is no `read`, `write_request` or pairing.
* **Discovery by name.** Revo Metro compares the advertised name exactly (`0x14080b2e0`):
  `REVO_DUAL_AXIS_TABLE` is the dual-axis table (type 0); `REVO_STABILIZER`, or any name containing
  `RevoStabilizer`, is the handheld stabilizer (type 1); `REVO_TA500` is the large turntable (type 2).
  Service UUIDs are not used for discovery. None of the 131 GUID-like strings in the exe is a BLE UUID; they
  are XMP metadata of embedded images.
* **Characteristic.** After connecting, Revo Metro walks every service and characteristic. It splits the
  characteristic UUID at `-`, parses the first group as hex and takes the first one equal to **0xFFE1**
  (`cmp r13d, 0xffe1` at `0x14080de53`; large table at `0x1408182a3`; the stabilizer uses 0xFFB1 at
  `0x14081452c`). It keeps that characteristic's service and characteristic UUIDs and calls
  `notify(service, char, callback)` on the **same** characteristic. This is the classic "transparent UART"
  module: service `0000ffe0-0000-1000-8000-00805f9b34fb`, characteristic `0000ffe1-...`. The service UUID
  is **[inferred]**; Revo Metro never checks it.
* **Writes** (`0x1408116f0`). The ASCII command is split into chunks of `mtu` bytes (the SimpleBLE value,
  stored at object+0x198). Each chunk is sent with `write_command`, with a 10 ms sleep between chunks. No framing
  is added: the `;` is part of the command.
* **Notifications** (lambda `0x14080d250`). Each notification's payload is taken as one complete reply and
  stored as "the last reply" (object+0xf0). The lambda sleeps 100 ms, then broadcasts a condition variable. Each
  command then compares the last reply. Replies are not matched to requests.
* **Connect sequence** (logs + `0x140569180`). Connect, then `+QR,VERSION;`, then the speed-range query. It
  then sets `+CT,TURNSPEED=<mid of range>;` (logged: 53) and `+CR,TILTSPEED=10;`; the latter is a hard-coded
  constant `10.0` that firmware 3.28 refuses. The UI speed is sent as-is from then on.

---

## 3. Dual-axis command set (`DualAxisTurntable`, vtable `0x1416f67f8`)

"CT" commands the turn axis ("X") and "CR" the tilt axis ("Y"); "QT" and "QR" are the matching queries. All
values are **integers**: Revo Metro converts with `cvttss2si` (truncation). CloudClean rounds to the nearest
whole number and reports requested versus commanded values.

| Command | Meaning | Reply | Evidence |
|---|---|---|---|
| `+CT,TURNANGLE=<n>;` | turn **relative** n degrees; n > 0 is what Revo Metro sends for "Clockwise" | fw >= 3.28: **`+OK,TURNANGLE=<position>`** when the move is finished (position in deg, 2 decimals, cumulative, e.g. -145.72). fw < 3.28: `+OK;` only | [verified] `0x14080e9e0`; logs: `Send Command = +CT,TURNANGLE=-10;` then 1.49 s later `doMotionTurn Result = +OK,TURNANGLE=-145.72`, then -155.71, -165.71 ... (0.01 deg repeatability) |
| `+CT,TURNCONTINUE=1;` / `=-1;` | turn until stopped (Revo Metro's "Start" button and "Turntable Sync" use `-1`) | `+OK;` | [verified] `0x1408122c0` (angle == +-2^28 selects these); logs at 11:58:32 today |
| `+CT,STOP;` / `+CR,STOP;` | stop the turn / tilt axis | while turning: **`+OK,TURNANGLE=<position>`**; otherwise `+OK;` [inferred] | [verified] `0x140812b50` / `0x140812e20`; log `Stop Motion Result = +OK,TURNANGLE=-105.01` |
| `+CR,TILTVALUE=<n>;` | tilt to **absolute** n degrees (UI range -30..30; 0 = level) | `+OK;` within 1 s [inferred: Revo Metro waits 1 s for exactly `+OK;` and never logged anything else]. End of move: not announced as far as we know [unknown] | [verified] `0x1408122c0`; logs `+CR,TILTVALUE=-1;` (rotation 1 tilt), `+CR,TILTVALUE=0;` |
| `+CT,TURNSPEED=<s>;` | turn speed in **seconds per revolution** | `+OK;` (reply within 0.6 s; no failure logged) | [verified] logs `+CT,TURNSPEED=53;`, `=54;`. Timing: 10 deg at 53-54 s/rev took 1.49-1.53 s, nominal 1.47-1.50 s |
| `+CR,TILTSPEED=<v>;` | tilt speed | **`+FAIL,ERR=007;`** for v = 10 on fw 3.28 | [verified] logged on every connect. Unit and valid range [unknown]. CloudClean never sends it |
| `+CT,TOZERO;` / `+CR,TOZERO;` | go to the zero position | `+OK;` then, it seems, `+OK,TOZERO;` when there [inferred from the log `Stop Motion Result = +OK,TOZERO;`] | [verified] `0x1408113e0` |
| `+QR,VERSION;` | firmware version | `+DATA=<field>,V<x.yz>...`. Revo Metro splits on `=` and `,` and takes the 4 characters after `V` [inferred format; verified parser `0x14080fbb0`] | logs `DualAxisTable Version = 3.28` |
| `+QT,CHANGEANGLE;` / `+QR,TILTVALUE;` | current turn / tilt position | `+DATA=<deg>` (parsed with `stof`) [inferred] | [verified] `0x14080f630` |
| `+QT,TURNANGLE;` / `+QR,TILTANGLE;` | ranges | `+DATA=a,b,c`, three floats, meaning [unknown] | [verified] `0x140810390` |
| `+QT,TURNSPEED;` / `+QR,TILTSPEED;` | speed ranges | `+DATA=a,b,c`. Revo Metro sets speed = a + (b - a) x 0.5, which logged 53 (so a = 16, b = 90?) [inferred] | [verified] `0x140810ba0`, `0x140569180` |

Replies seen in real traffic: `+OK,TURNANGLE=-145.72` (no `;`), `+OK,TOZERO;` and `+FAIL,ERR=007;`.
CloudClean's parser (`protocol.parse_reply`) accepts `+OK;`, `+OK,<FIELD>[=<value>]`, `+FAIL,ERR=<code>` and
`+DATA=...`, with or without `;`. Its assembler also joins a message split across two notifications, in case
BlueZ negotiates a small MTU.

**Firmware switch [verified].** `doMotionTurn` compares the version string with `"3.28"` (`operator<`,
`0x1405644d0`). Below 3.28, Revo Metro only waits 1 s for `+OK;`. From 3.28 on it waits up to 10 s for a reply
whose part before `=` is `+OK,TURNANGLE`, logging `TIMEOUT !` otherwise. The user's table runs 3.28.

**Directions [verified in code / inferred physically].** The Auto Turntable dialog has two radio buttons.
"Counterclockwise" is button id 0 and checked by default; "Clockwise" is id 1 (`0x140697221` /
`0x14069723b`). `slotTurnTypeChange` stores `checkedId != 0` (`0x14069d3b0`), and the start routine
(`0x14069bf00`) passes it to `PartManagement::setDirection`. The step worker negates the step when the flag is
not 1 (`0x140568c86`). So **"Clockwise" sends positive angles**. The help text defines clockwise as "turns to
the right seen from above". The logs are consistent with this: -10 steps early on 09-09, +10 steps from 15:57
that day on, and the config now says `isClockwise: true`. Whether positive really turns clockwise seen from above is checked in
test step 5 (section 6).

**Error codes** (`RV_BLUETOOTH_ERROR_*`, in enum order): 0x3060004 connect failed, 0x3060005 not supported /
bad axis, 0x3060006 query failed, 0x3060007 command failed.

### 3.1 How Revo Metro runs its programs [verified from code and logs]

* **Turntable Sync** (scan start/pause linked to the table). `PartManagement::slotTurntableStart`
  (`0x14056f020`) sends `+CT,TURNCONTINUE=-1;` (large table: `CT+START(0,0,0,1,0,0);`) when scanning starts.
  `slotTurntableStop` stops it when scanning pauses, then the table is levelled with `+CR,TILTVALUE=0;`.
* **Auto Turntable** (step and scan: `PartManagement.cpp` worker `0x140568b10`, driven by a 200 ms QTimer).
  1. Wait until the scanner has captured "Total Frames" frames at the current stop (`slotSingleShotFinished` sets
     a flag).
  2. Emit `sigTurntableStart`: the scan side pauses.
  3. Send `+CT,TURNANGLE=+-interval;`, wait for `+OK,TURNANGLE=...`, then sleep 100 ms.
  4. `sum += interval`. At the first step of each new rotation (index `sum / 360`), if that rotation's tilt differs
     from the current tilt, send `+CR,TILTVALUE=<tilt>;` and sleep `|delta| x 10 / 60 s + 100 ms`, i.e. Revo
     assumes 6 deg/s. The log `|m_yValue - yPos|= 1 , Y motion cost time milliseconds = 166.667` shows this.
  5. Emit `sigTurnAngleEnd`: the scan side captures the next stop.
  6. Stop when `sum > rotations x 360 + interval`, then send `+CR,TOZERO;` (level the tilt) and emit
     `sigTurntableFinished`.

  The user's run on 09-14 (10 deg, 1 rotation) logged sums 0..370, which is 38 moves ending at 380 deg: Revo
  overlaps by one interval. Tilt values are whole degrees (an int array per rotation).

---

## 4. Large turntable (`REVO_TA500`, class `HumanTuentable`, vtable `0x1416f70a8`) [verified from code, never seen in traffic]

| Command | Meaning |
|---|---|
| `CT+TRUNSINGLE(<d>,<n>);` | turn n = |angle| degrees; d = 1 when angle <= 0, else 0 (`0x140818d50`) |
| `CT+START(0,0,0,1,0,0);` / `CT+START(1,0,0,1,0,0);` | continuous, + / - |
| `CT+SETSPEED(<level>);` | level = 20 if s > 92, 1 if s < 35, else int((s - 32) / 3), for the UI range 35-90 s/rev (`0x140818c20`) |
| `CT+SETDIR(<not flag>);`, `CT+SETSTOP();`, `CT+TOZERO();`, `CT+SPKMUTE(1);` | direction, stop, zero, mute (sent after connect) |
| reply | `CR+OK;` |

The large table has no tilt. The handheld stabilizer (0xFFB1; `AT+AGOFSET=`, `AT+TRACE_EN=`, `+BAT`) is not
supported.

---

## 5. CloudClean implementation

### 5.1 Python API (`cloudclean.capture.turntable`)

```python
from cloudclean.capture.turntable import get_turntable_manager
m = get_turntable_manager(workspace)            # one per workspace root (Workspace or path)
m.devices(scan_seconds=4.0)                     # [{id, name, kind, rssi, connected, remembered}], "simulated" first
m.connect(device=None, kind="auto", options=None)   # device: address, advertised name, "simulated" or None
                                                # (None = last used table, else the first found)
m.rotate(degrees, speed_s_per_rev=None, wait=False) # relative, + = clockwise seen from above, |deg| <= 720
m.tilt(degrees, wait=False)                     # absolute -30..30, dual-axis only
m.set_speed(s_per_rev)                          # 25..90 (large: 35..90)
m.stop()                                        # program + all motion, immediately
m.start_program({...}); m.stop_program()
m.status(); m.disconnect()
```

Every call returns the status. Failures raise `TurntableError` with a one-sentence message; the routes return
HTTP 400. With `wait=False` a move runs in a background thread: `status()["moving"]` is true and
`status()["last_move"]` shows `requested_deg`, `commanded_deg`, `state` (running, done, stopped or error) and the
result. `options={"time_scale": N}` speeds up the simulated clock (tests, demos).

`status()`:

```
{connected, device, name, kind: dual_axis|large|simulated|None, angle_deg (the table's cumulative position),
 angle_wrapped_deg, tilt_deg, moving, speed_s_per_rev, direction: cw|ccw, program: {...}|None, error,
 capabilities: {tilt, tilt_range, speed_range, interval_range, max_rotations, continuous, whole_degrees},
 validated, firmware, last_move, device_state (driver details: last Bluetooth messages, warnings, ...),
 bluetooth: {available, reason, installed}, remembered_device, log: [last 20 lines]}
```

### 5.2 Routes

`GET /api/turntable/status`; `GET /api/turntable/devices?scan_seconds=4` (0 skips the scan);
`POST /api/turntable/connect {device?, kind?, options?}`; `POST /api/turntable/disconnect`;
`POST /api/turntable/rotate {degrees, speed_s_per_rev?, wait?}`; `POST /api/turntable/tilt {degrees, wait?}`;
`POST /api/turntable/stop`; `POST /api/turntable/speed {s_per_rev}`; `POST /api/turntable/program {...}`;
`POST /api/turntable/program/stop`.

All handlers are plain `def`s, so FastAPI runs them in its thread pool. Bluetooth runs on its own asyncio loop
thread inside the driver, so the web event loop is never blocked.

### 5.3 Programs

```
{mode: "step"|"continuous" (default step), interval_deg (5..30, default 30), frames_per_stop (1..100, default 3),
 direction: "cw"|"ccw" (default cw), speed_s_per_rev (optional), rotations: [{tilt_deg}] (1..5, default [{0}]),
 sync_scan (default true), dwell_s (1.0), settle_s (0.3), level_at_end (true), capture_timeout_s (optional)}
```

* **step.** For each rotation, tilt to its `tilt_deg` (with capture paused). Then, per stop: rotate
  `interval_deg`, settle, and capture `frames_per_stop` frames. When 360 is not a multiple of the interval, the
  last move of a rotation is shorter, so **every rotation ends exactly where it started**; there is no 20 deg
  overlap as in Revo Metro. `level_at_end` tilts back to 0.
* **continuous.** One 360 deg move per rotation with the capture running ("Turntable Sync" style).
* **sync_scan.** Uses the workspace's capture session (`routes_capture.manager_for`) the way Revo Metro does.
  **Capture is paused while the platter moves** and resumed at each stop until `frames_per_stop` new frames have
  arrived. The first stop starts a connected session; at the end the capture is **left paused** so it can be
  saved. Without a session, or with `sync_scan=false`, the program dwells `dwell_s` per stop and adds a warning.
* **Progress.** `status()["program"]` holds `{state: starting|running|done|stopped|error, phase: setting
  speed|tilting|rotating|settling|capturing|dwelling|leveling, rotation, rotations, stop, stops_per_rotation,
  total_stops, completed_stops, progress (0..1), turned_deg, moves, tilts, tilt_deg, capture: {linked, frames},
  warnings, notes (values rounded to whole degrees), error, started, finished, elapsed_s}`.

### 5.4 Drivers

* `SimulatedTurntable` turns at 360 / s_per_rev deg/s plus 0.04 s at each start and stop, which matches the real
  1.49-1.53 s for 10 deg. It tilts at 6 deg/s. It is interruptible by `stop()` and thread-safe. When the simulated
  scanner's `Turntable` and `Follow the simulated turntable` settings are on (both default on), the scanner renders
  the part turned and tilted exactly as the simulated table is. The tilt axis is the horizontal axis across the
  scanner's line of sight, so the viewing elevation becomes 35 deg - tilt.
* `RevopointBleTurntable` (`ble.py`) uses `bleak`, which is imported lazily. Without it, calls raise
  "install cloudclean[turntable]". It runs on one asyncio loop thread shared by scans and the connection;
  `stop()` is written immediately, not queued behind a pending command.
  * **Connect.** Find by address, then connect, pick the 0xFFE1 characteristic and subscribe. It then sends
    only **read-only queries**: version, turn and tilt position, turn-speed range, tilt range. It sends no
    speed, no mute and no motion.
  * **rotate.** fw >= 3.28: waits for `+OK,TURNANGLE=` (timeout 1.5 x nominal + 5 s, using 90 s/rev when the
    speed is unknown). Older firmware: waits for the ack, then the nominal time, then queries the position.
  * **tilt.** Waits for `+OK;`, then for any `+OK,TILT...` announcement or Revo Metro's time estimate, then
    confirms with `+QR,TILTVALUE;` (plus or minus 0.6 deg) until the timeout. `+CR,TILTSPEED` is never sent.
  * **Traffic.** The last 200 messages in both directions are in `device_state.last_messages` (latest 12)
    and `driver.traffic`.

---

## 6. The safe first test on the real table

1. **Free the table.** Close Revo Metro, or disconnect the turntable in its Accessories panel. Leave the platter
   **empty** and keep cables clear of the tilt cradle. Switch the table on.
2. **Install bleak** on the machine that will drive it. On the Spark:
   `~/code/CloudClean/.venv/bin/pip install "cloudclean[turntable]"` (or `pip install bleak`). On Windows, the
   same in `.venv`.
3. **Scan** (read-only): `python tools/turntable/turntable_cli.py scan`. Expect `REVO_DUAL_AXIS_TABLE` with
   address `E4:8F:80:46:12:43`.
4. **Info** (read-only queries): `python tools/turntable/turntable_cli.py info --device E4:8F:80:46:12:43`.
   Expect firmware `3.28`, a position and the ranges. **Keep this output**: it confirms the `+DATA` formats
   marked [inferred].
5. **First motion, 10 deg:** `python tools/turntable/turntable_cli.py rotate 10 --yes --speed 60 --device
   E4:8F:80:46:12:43`.
   * Expected traffic: `> +CT,TURNSPEED=60;`, `< +OK;`, `> +CT,TURNANGLE=10;`, and after about 1.7 s
     `< +OK,TURNANGLE=<previous + 10>`.
   * **Watch the direction.** Seen from above it should turn **clockwise**.
   * Then `rotate -10 --yes` should bring it back.
6. **Tilt:** `tilt 5 --yes`, then `tilt 0 --yes`. Note whether the output says `"confirmed": true`.
7. **Emergency stop:** start `rotate 360 --yes --speed 90`. From a second terminal, run `stop`, or press Ctrl+C
   and then run `stop`.
8. **Web or assistant check:** start the server and `POST /api/turntable/connect {"device": "E4:8F:80:46:12:43"}`,
   then `POST /api/turntable/rotate {"degrees": 10, "wait": true}`.

If steps 5-7 behave as described, set `RevopointBleTurntable.validated = True` in `ble.py`, and fix the sign in
`protocol.dual_turn` if the direction was reversed. Send the printed traffic from steps 4-6 so that the [inferred]
rows in section 3 can be promoted.

---

## 7. Open questions

* **Direction.** Does a positive `TURNANGLE` turn clockwise seen from above? The code proves only that Revo
  Metro's "Clockwise" sends positive values.
* **Tilt.** Is the end of a tilt announced (`+OK,TILTVALUE...`?), and what is the real tilt speed?
  `+CR,TILTSPEED=10;` is refused with ERR=007.
* **Query formats.** The exact `+DATA=` replies to `+QR,VERSION;`, `+QT,CHANGEANGLE;`, `+QR,TILTVALUE;` and the
  range queries, and the meaning of the three range values. The reported speed range is probably 16..90.
* **Queries during motion.** Revo Metro never queries while turning. CloudClean only queries during a tilt
  (to confirm it).
* **Zero and position.** `TURNANGLE` positions are cumulative (e.g. -331.85); whether `+CT,TOZERO;` resets them
  or turns back is unknown.
* **Error codes.** The ERR codes other than 007.
* **MTU and splitting on BlueZ/Linux.** The assembler handles split replies; this is untested on hardware.
