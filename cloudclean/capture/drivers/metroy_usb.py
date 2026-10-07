"""Revopoint MetroY / MetroY Ultra plugged into the machine running CloudClean: detection, and native capture.

Detection works everywhere: it enumerates USB, finds the scanner by hardware id and (on Linux) the V4L2 nodes that
`uvcvideo` binds, and reports precisely what was found, so "is CloudClean even seeing my scanner?" always has an
answer.

Capture works on Linux (the DGX Spark), without Revo Metro, through `cloudclean.capture.metroy`:

* the **factory calibration is read off the scanner itself** over its HID command channel (camparam.yaml, the
  stereo calibration, and metroExtra.bin, the laser lines) - nothing is guessed or self-calibrated;
* the stereo IR stream (Revo Metro's own video mode, ~28 frames/s) is captured with V4L2 while the laser runs;
* each frame's laser stripes are located to sub-pixel precision and triangulated through the factory stereo
  calibration into millimetres, in parallel worker processes;
* the scanner is tracked by retro-reflective markers, as Revo Metro does in laser mode: they are lit by the IR fill
  light, found and triangulated in the same frames, and registered to a growing marker map (markers.py). Frames
  the markers cannot place are reported lost, never guessed. Without markers, "geometry" tracking hands the frames
  to CloudClean's frame-to-model alignment instead;
* frames go to the capture session as ``coordinates="sensor"`` with their pose, and CloudClean fuses them.

See docs/metroy-protocol.md for the protocol and docs/metroy-handoff.md for what is validated and what is not.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import threading
from pathlib import Path

import numpy as np

from .base import Frame, ScannerDriver, option, setting

# Fuzhou Rockchip Electronics, "REVO_PRODUCT" - the MetroY family's composite UVC device.
USB_IDS = {("2207", "110c")}
SYSFS_USB = Path("/sys/bus/usb/devices")


def _linux_devices() -> list[dict]:
    """Walk sysfs for a known scanner and collect the V4L2 nodes its interfaces expose. Needs no root."""
    found = []
    if not SYSFS_USB.is_dir():
        return found
    for dev in sorted(SYSFS_USB.iterdir()):
        try:
            vid = (dev / "idVendor").read_text().strip().lower()
            pid = (dev / "idProduct").read_text().strip().lower()
        except OSError:
            continue
        if (vid, pid) not in USB_IDS:
            continue
        product = _read(dev / "product") or "REVO_PRODUCT"
        nodes, drivers = [], set()
        for iface in sorted(dev.glob(f"{dev.name}:*")):
            link = iface / "driver"
            if link.is_symlink():
                drivers.add(link.resolve().name)
            for video in sorted(iface.glob("video4linux/video*")):
                nodes.append(f"/dev/{video.name}")
        found.append({"hardware_id": f"USB\\VID_{vid.upper()}&PID_{pid.upper()}", "product": product,
                      "serial": _read(dev / "serial"), "bus": dev.name,
                      "video_nodes": nodes, "kernel_drivers": sorted(drivers)})
    return found


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _present_instance_ids() -> list[str]:
    """Device instance ids of everything currently attached, via cfgmgr32. Needs no elevation.

    `Enum\\USB` in the registry lists every device ever plugged in, so its keys prove nothing about what is
    connected now, and the volatile `Control` subkey that is often used to tell them apart does not exist for this
    composite device even while it is plugged in. CM_GETIDLIST_FILTER_PRESENT is the supported way to ask.
    """
    import ctypes
    from ctypes import wintypes

    CM_GETIDLIST_FILTER_PRESENT = 0x00000100
    cfgmgr = ctypes.WinDLL("cfgmgr32")
    size = wintypes.ULONG(0)
    if cfgmgr.CM_Get_Device_ID_List_SizeW(ctypes.byref(size), None, CM_GETIDLIST_FILTER_PRESENT) != 0:
        return []
    buf = ctypes.create_unicode_buffer(size.value)
    if cfgmgr.CM_Get_Device_ID_ListW(None, buf, size.value, CM_GETIDLIST_FILTER_PRESENT) != 0:
        return []
    return [s for s in buf[:size.value].split("\0") if s]


def _windows_devices() -> list[dict]:
    """Present MetroY devices on Windows, with the friendly name read from the device's own enum key."""
    try:
        import winreg
    except ImportError:
        return []
    found = []
    for instance_id in _present_instance_ids():
        upper = instance_id.upper()
        match = next((f"USB\\VID_{v.upper()}&PID_{p.upper()}" for v, p in USB_IDS
                      if f"VID_{v.upper()}&PID_{p.upper()}" in upper), None)
        # The composite parent carries the serial; its MI_xx children are the same physical scanner.
        if not match or "&MI_" in upper:
            continue
        product = "REVO_PRODUCT"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, rf"SYSTEM\CurrentControlSet\Enum\{instance_id}") as sub:
                product = _reg(sub, "DeviceDesc").split(";")[-1] or product
        except OSError:
            pass
        found.append({"hardware_id": match, "product": product,
                      "serial": instance_id.rsplit("\\", 1)[-1], "bus": "",
                      "video_nodes": [], "kernel_drivers": []})
    return found


def _reg(key, name: str) -> str:
    try:
        import winreg

        value = winreg.QueryValueEx(key, name)[0]
        return str(value)
    except (OSError, ImportError, IndexError):
        return ""


def detect_scanners() -> list[dict]:
    """Every MetroY-family scanner attached to *this* machine. Never raises."""
    try:
        system = platform.system()
        if system == "Linux":
            return _linux_devices()
        if system == "Windows":
            return _windows_devices()
    except Exception:  # detection is best-effort: a probe failure must never hide the other drivers
        return []
    return []


NOT_FOUND = ("No MetroY scanner found on the USB bus of the machine running CloudClean. Plug it into this machine "
             "(CloudClean only sees scanners on its own USB ports); no other software is needed.")

NOT_LINUX = ("Native capture runs on Linux for now (the DGX Spark): there the scanner's raw camera stream can be read "
             "untouched. Plug the scanner into the DGX Spark, or, on this machine, use the 'Revopoint (Revo Metro "
             "bridge)' driver until native capture works here too.")

NO_OPENCV = ("Found the scanner, but OpenCV is not installed in CloudClean's environment. Install the MetroY extra "
             "(`pip install -e .[metroy]`, which adds opencv-python-headless) and restart CloudClean.")

WORKING_RANGE_MM = (200.0, 430.0)     # the calibrated range of the laser lines (Revo Metro's presets: 200..400)
OPTIMAL_MM = 300.0


TABLE_MIN_MARKERS = 4        # the marker map starts with 4 (MarkerTracker.min_inliers_reloc)
TABLE_MAX_RMS_MM = 0.5       # markers on one flat table: the user's turntable plate fits a plane to 0.19 mm
TABLE_MAX_OFF_MM = 1.5       # any marker further off the plane is on the part, not the table
TABLE_MIN_SPREAD_MM = 20.0   # a row of markers fixes no plane
TABLE_MAX_TILT_DEG = 75.0    # a table is seen from above; a board facing the scanner is a wall, not a floor


def _table_frame(markers: np.ndarray) -> np.ndarray | None:
    """World frame standing on the flat table the markers lie on (the turntable plate, a marker mat), or None.

    Z up along the table's normal (towards the scanner), Y the scanner's forward direction laid flat, X the
    scanner's right; origin on the table under the markers' middle. A turntable turns about its plate's normal, so
    the part then stands on the grid and turns about Z. None unless every marker lies on one plane seen from above."""
    m = np.asarray(markers, np.float64).reshape(-1, 3)
    if len(m) < TABLE_MIN_MARKERS:
        return None
    c = m.mean(axis=0)
    _, _, vt = np.linalg.svd(m - c)
    n = vt[2]
    off = (m - c) @ n
    if np.sqrt(np.mean(off ** 2)) > TABLE_MAX_RMS_MM or np.abs(off).max() > TABLE_MAX_OFF_MM:
        return None
    if np.ptp((m - c) @ vt[1]) < TABLE_MIN_SPREAD_MM:
        return None
    if n @ -c < 0:                                   # the scanner (sensor origin) is above the table
        n = -n
    if n @ np.array([0.0, -1.0, 0.0]) < np.cos(np.radians(TABLE_MAX_TILT_DEG)):
        return None
    forward = np.array([0.0, 0.0, 1.0])
    y = forward - (forward @ n) * n
    y /= np.linalg.norm(y)
    R = np.vstack([np.cross(y, n), y, n])            # rows: world X, Y, Z in sensor coordinates
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ c
    return T


def _initial_pose(pts: np.ndarray, markers: np.ndarray | None = None) -> np.ndarray:
    """The world frame CloudClean uses for a scanner that tracks itself. Markers on a flat table: standing on that
    table (see _table_frame). Otherwise Z up = sensor up, origin at the first frame's centroid - of its points, or
    of its markers when the laser has not come up yet (the same convention the capture session applies to
    untracked devices)."""
    if markers is not None and (T := _table_frame(markers)) is not None:
        return T
    R = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # sensor x->X, y(down)->-Z, z(forward)->Y
    centre = pts if len(pts) else markers
    T = np.eye(4)
    T[:3, :3] = R
    if centre is not None and len(centre):
        T[:3, 3] = -R @ np.asarray(centre, np.float64).reshape(-1, 3).mean(axis=0)
    return T


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "cloudclean" / "metroy"


class MetroyUsbDriver(ScannerDriver):
    id = "metroy_usb"
    name = "Revopoint MetroY (USB, native)"
    kind = "scanner"
    description = ("Captures directly from a MetroY / MetroY Ultra plugged into this machine, without Revo Metro: "
                   "the factory calibration is read off the scanner, the laser stripes are triangulated in "
                   "millimetres, and CloudClean tracks and fuses the frames. Linux only.")

    def __init__(self):
        super().__init__()
        self.devices = detect_scanners()
        self._scanner = None
        self._tracker = None
        self._recording = None
        self.phase = "scanning"       # or "mapping": markers only, nothing fused (Revo Metro's marker scan)
        self._track_lock = threading.Lock()    # commands arrive on request threads, frames on the capture thread

    # -- description
    def availability(self) -> tuple[bool, str]:
        if not self.devices:
            return False, NOT_FOUND
        d = self.devices[0]
        where = f" as {', '.join(d['video_nodes'])}" if d["video_nodes"] else ""
        found = f"Found {d['product']} ({d['hardware_id']}){where}. "
        if platform.system() != "Linux":
            return False, found + NOT_LINUX
        if importlib.util.find_spec("cv2") is None:
            return False, NO_OPENCV
        from ..metroy import hid
        try:
            node = hid.find_hidraw()
        except FileNotFoundError:
            return False, found + "Its HID command channel is missing, so the calibration cannot be read."
        if not d["video_nodes"]:
            return False, found + "No video node is bound (is the uvcvideo module loaded?)."
        # Linux gives a plugged-in scanner only to whoever is logged in at this machine's own screen unless a rule
        # says otherwise: a server with nobody at its screen (CloudClean as a service) then cannot open it
        locked = [n for n in [node, *d["video_nodes"]] if not os.access(n, os.R_OK | os.W_OK)]
        if locked:
            return False, (found + f"CloudClean may not open it yet ({', '.join(locked)}). Run this once on this "
                           "machine, in the CloudClean folder: sudo deploy/install-scanner-access.sh - it lets "
                           "CloudClean use the scanner even when nobody is logged in at the screen (no replugging).")
        return True, found.strip()

    def info(self) -> dict:
        return {**super().info(), "devices": self.devices}

    def settings_schema(self) -> list[dict]:
        return [
            setting("point_distance", "Point distance", "number", 0.2, min=0.05, max=2.0, step=0.05, unit="mm",
                    help="Target spacing of the fused cloud; coverage guidance measures density against it."),
            setting("matching", "Stripe matching", "select", "lines",
                    options=[option("lines", "Laser line identity (factory laser calibration)"),
                             option("order", "Stereo order only (diagnostic: slips across steps)")],
                    help="How a stripe in the left camera is paired with its stripe in the right camera. Line "
                         "identity uses the scanner's own laser calibration and is correct across steps and "
                         "occlusions; stereo order alone can pair the wrong stripes there."),
            setting("tracking", "Tracking", "select", "markers",
                    options=[option("markers", "Markers (as Revo Metro in laser mode)"),
                             option("geometry", "Geometry (no markers; less robust on laser stripes)")],
                    help="Markers: the scanner is located by retro-reflective markers on or around the part - keep "
                         "at least 4 in view. Geometry: frame-to-model alignment, for parts without markers."),
            setting("surface", "Object surface", "select", "general",
                    options=[option("general", "General"), option("dark", "Dark"),
                             option("reflective", "Reflective / shiny")],
                    help="Revo Metro's exposure presets: general 200 us / gain 1, dark 1000 us / gain 2, "
                         "reflective 800 us / gain 1, each with its own marker light."),
            setting("fill_light", "Marker light", "number", -1, min=-1, max=255, step=1,
                    help="IR fill-light brightness that makes the markers shine; -1 = the surface preset's value."),
            setting("record", "Record for diagnosis", "boolean", False,
                    help="Keep every frame's points, markers and pose, and every 10th raw camera image, in "
                         "~/.cache/cloudclean/metroy/recordings/ so a capture can be replayed offline."),
            setting("workers", "Processing workers", "number", 0, min=0, max=32, step=1,
                    help="Parallel triangulation processes; 0 = automatic. The scanner sends ~28 frames a second."),
            setting("turntable", "Turntable", "boolean", False, help="The part sits on a rotating turntable."),
        ]

    def capabilities(self) -> dict:
        return {"range_mm": list(WORKING_RANGE_MM), "optimal_mm": OPTIMAL_MM, "streaming": True,
                "provides_pose": self.settings.get("tracking", "markers") == "markers",
                "detected": bool(self.devices)}

    # -- lifecycle
    def _connect(self) -> None:
        from ..metroy.scanner import MetroyScanner
        from ..metroy.markers import MarkerTracker
        workers = int(self.settings.get("workers") or 0) or None
        by_markers = self.settings.get("tracking", "markers") == "markers"
        fill = int(self.settings.get("fill_light", -1))
        self._recording = None
        record_dir = None
        if self.settings.get("record"):
            import time as _time
            record_dir = _cache_dir() / "recordings" / _time.strftime("%Y%m%d-%H%M%S")
            self._recording = {"dir": record_dir, "frames": []}
        self._scanner = MetroyScanner(cache_dir=_cache_dir(), workers=workers, record_dir=record_dir,
                                      matching=self.settings.get("matching", "lines"), markers=by_markers,
                                      surface=self.settings.get("surface", "general"),
                                      fill_light=fill if fill >= 0 else None)
        self._scanner.open()
        self._tracker = MarkerTracker() if by_markers else None

    def _start(self) -> None:
        self._scanner.start()

    def read(self, timeout: float = 0.1) -> Frame | None:
        if self._scanner is None or not self.running:
            return None
        f = self._scanner.next(timeout)
        if f is None:
            return None
        pts = f.points
        distance = float(np.median(pts[:, 2])) if len(pts) else 0.0
        meta = {"distance_mm": distance, "sequence": f.sequence, "process_ms": 1000.0 * f.process_s,
                "markers": int(len(f.markers)), "marker_points": f.markers,
                **{k: v for k, v in f.info.items() if k in ("family", "assigned", "tracks", "paired")}}
        pose = None
        if self._tracker is not None:
            # the map starts on the first frame with enough markers, often before the laser lines come up: without
            # a pose from that frame's markers the whole scan stayed in sensor coordinates (lying on its side)
            if len(self._tracker.world) == 0:
                self._tracker.initial = _initial_pose(pts, f.markers)
            with self._track_lock:
                res = self._tracker.track(f.markers)
                world = self._tracker.world.copy()
            meta.update(marker_state=res.state, marker_inliers=res.inliers, marker_rmse_mm=res.rmse_mm,
                        map_markers=int(len(world)), map_points=world)
            if self.phase == "mapping":
                meta["preview_only"] = True
            if res.pose is None:
                meta["device_tracking"] = res.state          # "no_markers" / "lost": the session will not fuse it
            else:
                meta["tracking_label"] = "markers"
            pose = res.pose
        if self._recording is not None:
            self._recording["frames"].append({"t": f.timestamp, "sequence": f.sequence, "points": pts,
                                              "markers": f.markers, "pose": pose, "meta": dict(meta)})
        return Frame(points=pts, pose=pose, timestamp=f.timestamp, coordinates="sensor", meta=meta)

    def stop(self) -> None:
        if self._scanner is not None:
            self._scanner.stop()
        super().stop()

    def disconnect(self) -> None:
        super().disconnect()
        rec = getattr(self, "_recording", None)
        if rec and rec["frames"]:
            import pickle
            rec["dir"].mkdir(parents=True, exist_ok=True)
            with open(rec["dir"] / "frames.pkl", "wb") as fh:
                pickle.dump({"settings": dict(self.settings), "frames": rec["frames"],
                             "calibration": self._scanner.files.calibration if self._scanner else None,
                             "laser": self._scanner.files.laser if self._scanner else None}, fh)
            self._recording = None
        if self._scanner is not None:
            self._scanner.close()
            self._scanner = None
        self._tracker = None

    def status(self) -> dict:
        """Live acquisition counters (frames grabbed, processed, dropped) and the marker map, for the UI."""
        out = dict(self._scanner.stats) if self._scanner is not None else {}
        if self._tracker is not None:
            out.update(phase=self.phase, map_markers=int(len(self._tracker.world)), map_frozen=self._tracker.frozen)
        return out

    def command(self, name: str, args: dict | None = None) -> dict:
        """Marker map workflow, as Revo Metro's marker scan:

        map_markers  start a fresh map; frames are tracked and shown but nothing is fused - sweep over every marker
        finish_map   adjust the whole map at once and freeze it; the surface scan then tracks against it
        clear_map    forget the map and go back to mapping while scanning
        """
        if self._tracker is None:
            raise ValueError("marker commands need tracking = Markers")
        with self._track_lock:
            return self._command(name)

    def _command(self, name: str) -> dict:
        if name == "map_markers":
            self._tracker.reset()
            self.phase = "mapping"
            return {"phase": self.phase}
        if name == "finish_map":
            if len(self._tracker.world) < 4:
                raise ValueError(f"only {len(self._tracker.world)} markers mapped - sweep over more markers first")
            info = self._tracker.freeze()
            self.phase = "scanning"
            return {"phase": self.phase, **info}
        if name == "clear_map":
            self._tracker.reset()
            self.phase = "scanning"
            return {"phase": self.phase}
        raise ValueError(f"unknown command '{name}'")
