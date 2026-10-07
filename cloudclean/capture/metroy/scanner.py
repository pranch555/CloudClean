"""A MetroY as a live source of triangulated frames: calibration off the device, laser control, streaming, workers.

    with MetroyScanner(cache_dir) as sc:
        sc.start()
        while ...:
            result = sc.next(timeout=0.1)        # TriangulatedFrame or None

Threads and processes:

* one grabber thread drains the V4L2 stream continuously, so the kernel's buffer ring never fills and the scanner
  never stalls; it keeps only the newest frame waiting;
* a pool of worker processes triangulates frames in parallel (a frame costs a few hundred ms of CPU and the scanner
  sends ~28 a second). Each worker builds its own Triangulator once from the calibration bytes;
* results are handed out strictly in capture order. A result that would arrive after a newer one has already been
  delivered is dropped: tracking needs time to move forward, and a late frame is worth less than a gap.

The laser is switched on in start() and off in stop()/close(), including on errors, so it is never left running.

Camera settings (exposure.py, camera.py): the surface preset, auto or manual, and the laser level / exposure / gain /
marker light are applied after the start-up, including the laser pulse 0xb08 that goes with the level. While
streaming, every change - the user's or the automatic exposure's - goes through one writer thread (CameraControl):
one register per command, paced, nothing re-written. Every processed frame is measured at the stripe centres it
triangulated (a few microseconds) and feeds the automatic exposure and the live readout.

Preview: start(deliver=False) streams, measures, runs the automatic exposure and serves the camera view, but hands no
frame out (nothing reaches tracking or fusion) and triangulates only ~12 frames a second. begin_delivery() turns a
preview into a scan without restarting the stream. The camera view: while someone asks for it (camera_view()), a
frame every ~1/8 s also returns a small copy of its rectified images and what was found in them.
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import queue
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import hid as hidmod
from .camera import CameraControl, camera_payload, compose_view, encode_jpeg
from .exposure import (STARTUP_REGISTERS, AutoExposure, CameraSettings, Light, Readout, measure_light,
                       parse_gain_map)

VIEW_SCALE = 0.3            # camera view: 480 x 360 per camera
VIEW_FPS = 8.0
PREVIEW_FPS = 12.0          # frames triangulated per second while only previewing

# worker-process state: built once per process by _init_worker
_WORKER: dict = {}


def _init_worker(calib_yaml: bytes, laser_file: bytes | None, params: dict) -> None:
    import cv2
    cv2.setNumThreads(1)                      # the pool supplies the parallelism
    from .stripes import Params, Triangulator
    _WORKER["tri"] = Triangulator.from_yaml(calib_yaml, Params(**params), laser_file)


def _triangulate(image: np.ndarray, matching: str, find_markers: bool, view_scale: float = 0.0):
    t0 = time.perf_counter()
    tri = _WORKER["tri"]
    from . import markers
    rl, rr = tri.rectify(image)
    # markers are found in every frame: even without marker tracking their bright discs must be kept out of the
    # stripe detection
    bl, br = markers.detect(rl), markers.detect(rr)
    pts, xs, ys = tri.points_rectified(rl, rr, matching=matching, with_pixels=True,
                                       masks=(markers.mask(rl.shape, bl), markers.mask(rr.shape, br)))
    mk = markers.stereo(bl, br, tri.Q) if find_markers else np.zeros((0, 3))
    light = measure_light(rl, xs, ys, pts)
    view = None
    if view_scale:
        from .camera import make_view
        with np.errstate(divide="ignore", invalid="ignore"):
            xr = xs - tri.off - tri.f * tri.B / pts[:, 2].astype(np.float64)    # the same centres in the right view
        view = make_view(rl, rr, view_scale, np.c_[xs, ys], np.c_[xr, ys], bl, br)
    return pts, mk, dict(tri.last), time.perf_counter() - t0, light, view


@dataclass
class TriangulatedFrame:
    points: np.ndarray            # (N, 3) float32, mm, rectified left camera frame (x right, y down, z forward)
    timestamp: float              # seconds, kernel capture time
    sequence: int
    process_s: float
    info: dict = field(default_factory=dict)
    markers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # (M, 3) mm, same frame


@dataclass
class DeviceFiles:
    calibration: bytes            # camparam.yaml
    laser: bytes | None           # metroExtra.bin
    serial: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.calibration + (self.laser or b"")).hexdigest()[:12]


def device_serial() -> str:
    """The scanner's USB serial number from sysfs ('' if unknown)."""
    root = Path("/sys/bus/usb/devices")
    for dev in root.iterdir() if root.is_dir() else []:
        try:
            if (dev / "idVendor").read_text().strip().lower() == "2207" and \
                    (dev / "idProduct").read_text().strip().lower() == "110c":
                return (dev / "serial").read_text().strip()
        except OSError:
            continue
    return ""


def read_device_files(dev: hidmod.MetroyHid, cache_dir: Path | None = None) -> DeviceFiles:
    """The factory calibration, read off the scanner itself. Cached per serial so a later offline replay can use it,
    but always re-read from the device when it is connected: the device is the authority."""
    dev.drain()
    calibration = dev.read_file(hidmod.CALIBRATION)
    try:
        laser = dev.read_file(hidmod.LASER_CALIBRATION)
    except (FileNotFoundError, IOError):
        laser = None
    files = DeviceFiles(calibration, laser, device_serial())
    if cache_dir is not None:
        where = Path(cache_dir) / (files.serial or "unknown")
        where.mkdir(parents=True, exist_ok=True)
        (where / "camparam.yaml").write_bytes(calibration)
        if laser is not None:
            (where / "metroExtra.bin").write_bytes(laser)
    return files


class MetroyScanner:
    def __init__(self, cache_dir: Path | None = None, workers: int | None = None, matching: str = "lines",
                 params: dict | None = None, markers: bool = False, surface: str = "general",
                 fill_light: int | None = None, record_dir: Path | None = None, record_every: int = 10,
                 camera: CameraSettings | None = None):
        self.cache_dir = cache_dir
        self.workers = workers or max(2, min(12, (os.cpu_count() or 4) - 2))
        self.matching = matching
        self.params = params or {}
        self.find_markers = markers
        self.record_dir = Path(record_dir) if record_dir else None      # raw frames for offline diagnosis
        self.record_every = max(1, int(record_every))
        self._submitted = 0
        if camera is None:                        # the older arguments: a surface preset, optionally its fill light
            camera = CameraSettings.from_preset(surface, "auto", markers)   # no markers: no marker light
            if fill_light is not None:
                camera = replace(camera, marker_light=int(fill_light))
        self.camera = camera
        self.gain_map: dict | None = None
        self.gain_map_source = "defaults"
        self.ae = AutoExposure(camera, None, markers)
        self.readout = Readout()
        self.control: CameraControl | None = None
        self._cam_lock = threading.RLock()
        self.delivering = True                    # False: preview (frames measured and shown, never handed out)
        self._deliver_after = -math.inf
        self._view_until = 0.0
        self._last_view_req = 0.0
        self._last_submit = 0.0
        self._view: tuple[float, dict] | None = None
        self.failed: BaseException | None = None
        self.hid: hidmod.MetroyHid | None = None
        self.files: DeviceFiles | None = None
        self.stream = None
        self.pool: ProcessPoolExecutor | None = None
        self._grab: threading.Thread | None = None
        self._stop = threading.Event()
        self._latest: tuple | None = None
        self._latest_lock = threading.Condition()
        self._inflight: list[tuple[float, int, Future]] = []
        self._out: queue.Queue = queue.Queue(maxsize=8)
        self._last_delivered = -1.0
        self.stats = {"grabbed": 0, "processed": 0, "dropped_late": 0, "dropped_busy": 0, "sequence_gaps": 0,
                      "errors": 0, "last_error": "", "preview_frames": 0}
        self.running = False

    @property
    def preset(self) -> dict:
        """The acquisition values in the older three-key form (exposure_us, gain, fill_light)."""
        cs = self.camera
        return {"exposure_us": cs.exposure_us, "gain": cs.gain, "fill_light": cs.marker_light}

    # -- lifecycle
    def open(self) -> "MetroyScanner":
        from .v4l2 import Stream
        self.hid = hidmod.MetroyHid()
        try:
            self.files = read_device_files(self.hid, self.cache_dir)
            if self.matching == "lines" and self.files.laser is None:
                raise RuntimeError("the scanner has no laser calibration (/data/camparam/metroExtra.bin), which line "
                                   "matching needs")
            self._read_gain_map()
            self.stream = Stream()
        except Exception:
            self.close()
            raise
        # spawn, not fork: the web server is multi-threaded, and forking a threaded process can deadlock
        self.pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_init_worker,
                                        initargs=(self.files.calibration, self.files.laser, self.params))
        return self

    def _read_gain_map(self) -> None:
        """Marker light per gain, off the scanner (/data/brightnessGainMap.json); Revo Metro's values otherwise."""
        try:
            self.gain_map = parse_gain_map(self.hid.read_file(hidmod.EXPOSURE_PRESETS, retries=3))
            self.gain_map_source = "scanner"
        except (OSError, ValueError):
            self.gain_map, self.gain_map_source = None, "defaults"
        self.ae.gain_map = self.gain_map

    def start(self, deliver: bool = True) -> None:
        """Start streaming. deliver=False is a preview: measured, auto exposed and shown, nothing handed out."""
        if self.hid is None:
            raise RuntimeError("open() the scanner first")
        self._stop.clear()
        self.failed = None
        self.delivering = deliver
        self._deliver_after = -math.inf
        self.hid.startup()                        # a known state, whatever the scanner was left in
        self.control = CameraControl(self.hid, on_applied=self._applied)
        self.control.assume(STARTUP_REGISTERS)    # what startup() just wrote: only the differences follow
        with self._cam_lock:
            registers = self.camera.registers()
            self.ae.reset(self.camera, time.monotonic())
        self.control.apply_now(registers)
        self.hid.laser(True)
        self.control.start()
        self.running = True
        with self._cam_lock:
            latest = self.camera.registers()
        if latest != registers:                   # changed while the start-up ran: write it now
            self.control.request(latest)
        self._grab = threading.Thread(target=self._grab_loop, name="metroy-grab", daemon=True)
        self._grab.start()
        self._feeder = threading.Thread(target=self._feed_loop, name="metroy-feed", daemon=True)
        self._feeder.start()

    def begin_delivery(self) -> None:
        """A preview becomes a scan: frames captured from now on are handed out (the stream keeps running)."""
        if self.failed is not None:
            raise IOError(str(self.failed))
        self._deliver_after = time.monotonic()    # V4L2 stamps frames with CLOCK_MONOTONIC, as time.monotonic()
        while True:
            try:
                self._out.get_nowait()
            except queue.Empty:
                break
        self.delivering = True

    def stop(self) -> None:
        self._stop.set()
        with self._latest_lock:
            self._latest_lock.notify_all()
        for t in (getattr(self, "_grab", None), getattr(self, "_feeder", None)):
            if t is not None and t.is_alive():
                t.join(timeout=3.0)
        self._grab = self._feeder = None
        if self.control is not None:
            self.control.stop()
        self.running = False
        self.delivering = True
        self._view = None
        self.readout = Readout()
        if self.hid is not None:
            try:
                self.hid.laser(False)
                self.hid.preisp(hidmod.REG_FILL_LIGHT, 0)
            except OSError:
                pass

    # -- camera settings, automatic exposure and the camera view
    def _applied(self, t: float) -> None:
        with self._cam_lock:
            self.ae.applied(t)

    def set_camera(self, changes: dict) -> CameraSettings:
        """Change surface / mode / laser level / exposure / gain / marker light; streaming or not. Raises ValueError."""
        with self._cam_lock:
            new = self.camera.with_changes(changes, self.find_markers)
            self.camera = new
            self.ae.reset(new, time.monotonic() if self.running else None)
        if self.running and self.control is not None:
            self.control.request(new.registers())
        return new

    def _observe(self, light: Light) -> None:
        self.readout.add(light)
        with self._cam_lock:
            self.ae.observe(light)
            new = self.ae.step(time.monotonic())
            if new is not None:
                self.camera = new
        if new is not None and self.control is not None:
            self.control.request(new.registers())

    def camera_status(self) -> dict:
        with self._cam_lock:
            cs, auto = self.camera, self.ae.status
        control = self.control
        error = str(self.failed) if self.failed else (control.error if control else None)
        return camera_payload(cs, auto, self.readout.summary(), streaming=self.running,
                              preview=self.running and not self.delivering, error=error,
                              gain_map=self.gain_map_source, applied=control.applied if control else None)

    def camera_view(self, cams: str = "both", overlay: bool = True, width: int = 960) -> bytes | None:
        """The latest camera picture as a JPEG (None until one is there). Asking keeps the pictures coming."""
        self._view_until = time.monotonic() + 1.5
        view = self._view
        if view is None or not self.running:
            return None
        return encode_jpeg(compose_view(view[1], cams, overlay, width))

    def close(self) -> None:
        self.stop()
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        if self.hid is not None:
            try:
                self.hid.laser(False)
            except OSError:
                pass
            self.hid.close()
            self.hid = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # -- consumer side
    def next(self, timeout: float = 0.1) -> TriangulatedFrame | None:
        try:
            item = self._out.get(timeout=timeout)
        except queue.Empty:
            return None
        if isinstance(item, BaseException):
            raise item
        return item

    # -- internals
    def _grab_loop(self) -> None:
        last_seq = None
        failures = 0
        while not self._stop.is_set():
            try:
                frame = self.stream.read(timeout=0.5)
            except OSError as exc:
                # the scanner re-enumerates itself when unhappy; report and stop rather than spin
                self.failed = IOError(f"the scanner stream failed ({exc}); it may have reset itself - "
                                      "wait a few seconds and connect again")
                self._out.put(self.failed)
                return
            if frame is None:
                failures += 1
                if failures == 10:
                    self._out.put(TimeoutError("no frames from the scanner for 5 s"))
                continue
            failures = 0
            self.stats["grabbed"] += 1
            if last_seq is not None and frame.sequence != last_seq + 1:
                self.stats["sequence_gaps"] += 1
            last_seq = frame.sequence
            with self._latest_lock:
                if self._latest is not None:
                    self.stats["dropped_busy"] += 1
                self._latest = (frame.image, frame.timestamp, frame.sequence)
                self._latest_lock.notify()

    def _feed_loop(self) -> None:
        """Keep every worker busy with the newest frame; hand results out in capture order."""
        while not self._stop.is_set():
            self._collect()
            if len(self._inflight) >= self.workers:
                time.sleep(0.002)
                continue
            with self._latest_lock:
                if self._latest is None:
                    self._latest_lock.wait(timeout=0.05)
                item, self._latest = self._latest, None
            if item is None:
                continue
            image, ts, seq = item
            now = time.monotonic()
            if not self.delivering and now - self._last_submit < 1.0 / PREVIEW_FPS:
                continue                          # a preview needs a few frames a second, not the whole pool
            self._last_submit = now
            if self.delivering and self.record_dir is not None and self._submitted % self.record_every == 0:
                self.record_dir.mkdir(parents=True, exist_ok=True)
                np.save(self.record_dir / f"raw_{seq:08d}_{ts:.6f}.npy", image)
            self._submitted += 1
            view = 0.0
            if now < self._view_until and now - self._last_view_req >= 1.0 / VIEW_FPS:
                self._last_view_req = now
                view = VIEW_SCALE
            fut = self.pool.submit(_triangulate, image, self.matching, self.find_markers, view)
            self._inflight.append((ts, seq, fut))
        for _, _, fut in self._inflight:
            fut.cancel()
        self._inflight = []

    def _collect(self) -> None:
        # deliver completed results in order; an older frame still running holds back newer finished ones
        while self._inflight and self._inflight[0][2].done():
            ts, seq, fut = self._inflight.pop(0)
            try:
                pts, mk, info, dt, light, view = fut.result()
            except Exception as exc:  # a bad frame must not end the scan
                self.stats["errors"] += 1
                self.stats["last_error"] = str(exc)
                continue
            light.t = ts
            try:
                self._observe(light)
            except Exception as exc:  # the exposure loop must never end the scan either
                self.stats["last_error"] = f"auto exposure: {exc}"
            if view is not None:
                self._view = (ts, view)
            info = {**info, "stripe_peak": round(light.peak, 1), "saturated": round(light.saturated, 4)}
            if not self.delivering or ts < self._deliver_after:
                self.stats["preview_frames"] += 1
                continue
            if ts <= self._last_delivered:
                self.stats["dropped_late"] += 1
                continue
            self._last_delivered = ts
            self.stats["processed"] += 1
            try:
                self._out.put_nowait(TriangulatedFrame(pts, ts, seq, dt, info, mk))
            except queue.Full:
                self.stats["dropped_late"] += 1   # the consumer is behind: newest-first is not possible, drop
