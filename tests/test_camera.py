"""Camera view and exposure for native MetroY capture: Revo Metro's presets and laser pulse, the register write rules,
the automatic exposure loop, the preview lifecycle and the API - all with fakes and the simulated scanner. Nothing
here talks to a real scanner."""
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cloudclean.capture.drivers import create_driver
from cloudclean.capture.metroy import hid as hidmod
from cloudclean.capture.metroy.camera import CameraControl, PreviewLease, compose_view, encode_jpeg
from cloudclean.capture.metroy.exposure import (MARKER_LIGHT_BY_GAIN, MAX_PULSE, STARTUP_REGISTERS, AutoExposure,
                                                CameraSettings, Light, camera_from_driver_settings, laser_pulse,
                                                measure_light, parse_gain_map, remembered, write_plan)
from cloudclean.web.routes_capture import manager_for
from cloudclean.web.server import create_app


# ----------------------------------------------------------------------------------------------- presets and pulse
def test_laser_level_to_pulse_matches_revo_metro():
    # the four (level, 0xb08) pairs read from Revo Metro's own logs, general preset (200 us, max 51)
    for level, pulse in ((44, 217), (38, 194), (48, 233), (51, 245)):
        assert laser_pulse(level, 200, 51) == pulse
    general = CameraSettings.from_preset("general")
    assert general.pulse == 217 == STARTUP_REGISTERS["pulse"]          # what MetroyHid.startup() writes
    dark = CameraSettings.from_preset("dark")
    assert (dark.exposure_us, dark.gain, dark.laser_level, dark.level_max) == (1000, 2, 193, 255)
    assert abs(dark.pulse - 802) <= 2                                   # [inferred] ~5x the 217 CloudClean wrote
    shiny = CameraSettings.from_preset("reflective")
    assert (shiny.exposure_us, shiny.laser_level, shiny.level_max, shiny.pulse) == (800, 90, 204, 397)
    assert laser_pulse(255, 2000, 255) == MAX_PULSE                     # never longer than Revo's own maximum
    assert hidmod.SURFACE_PRESETS["dark"] == {"exposure_us": 1000, "gain": 2, "fill_light": 3}
    assert CameraSettings.from_preset("general", markers=False).marker_light == 0


def test_settings_changes_are_validated_and_remembered():
    cs = CameraSettings.from_preset("general")
    dark = cs.with_changes({"surface": "dark"})
    assert dark.mode == "auto" and dark.laser_level == 193 and dark.marker_light == 3
    manual = dark.with_changes({"laser_level": 120})
    assert manual.mode == "manual" and manual.pulse == 45 + int(120 * 1000 / 255)
    assert dark.with_changes({"laser_level": 120, "mode": "auto"}).mode == "auto"
    for bad in ({"gain": 9}, {"laser_level": 300}, {"exposure_us": "x"}, {"surface": "glass"}, {"focus": 1},
                {"mode": "sport"}):
        with pytest.raises(ValueError):
            cs.with_changes(bad)
    keep = remembered(manual)
    assert keep == {"surface": "dark", "camera_mode": "manual", "laser_level": 120, "exposure_us": 1000, "gain": 2,
                    "fill_light": 3}
    assert camera_from_driver_settings(keep) == manual
    auto = remembered(dark)
    assert auto["camera_mode"] == "auto" and auto["laser_level"] == -1
    assert camera_from_driver_settings(auto) == dark
    # a remembered level that does not fit the surface falls back to the preset rather than failing to connect
    assert camera_from_driver_settings({**keep, "surface": "general"}).laser_level == 44


def test_gain_map_off_the_scanner_or_revo_metro_copy():
    doc = {"lineDark": [{"brightness": b, "gain": g} for g, b in enumerate((9, 4, 2, 1, 1), 1)],
           "lineGeneral": [{"brightness": b, "gain": g} for g, b in enumerate((40, 20, 13, 9, 7), 1)],
           "surface": [{"brightness": 15, "gain": 1}]}
    plain = json.dumps(doc).encode()
    got = parse_gain_map(plain)
    assert got["general"][:3] == (40, 20, 13) and got["dark"] == (9, 4, 2, 1, 1)
    assert got["reflective"] == MARKER_LIGHT_BY_GAIN["reflective"]        # missing: CloudClean's defaults
    obfuscated = bytes(b ^ b"RevoPoint"[i % 9] ^ b"RevoScan"[i % 8] for i, b in enumerate(plain))
    assert parse_gain_map(obfuscated) == got                              # Revo Metro's PC copy
    with pytest.raises(ValueError):
        parse_gain_map(b"\x00\x01garbage")


# ----------------------------------------------------------------------------------------------- write rules
class FakeHid:
    """Records every shell command with its time; never touches a device."""

    def __init__(self):
        self.log: list[tuple[float, str]] = []
        self.startups = 0

    def shell(self, command: str) -> None:
        self.log.append((time.perf_counter(), command))

    def preisp(self, register, *values):
        args = " ".join(str(v) for v in values)
        self.shell(f"echo s 0x{register:x}{' ' + args if args else ''} > /dev/rk_preisp")

    def laser(self, on):
        self.preisp(hidmod.REG_LASER, 1 if on else 0)

    def startup(self, pattern=hidmod.PATTERN_CROSS):
        self.startups += 1
        self.shell("startup")

    def read_file(self, remote, retries=8):
        raise FileNotFoundError(remote)

    def close(self):
        self.log.append((time.perf_counter(), "close"))

    @property
    def commands(self):
        return [c for _, c in self.log]


def _check_rules(commands):
    for c in commands:
        writes = c.count("echo s ")
        if "0x910" in c or "0x911" in c:
            # frame time and exposure only together, and nothing else in that command
            assert "0x910" in c and "0x911" in c and writes == 2, c
        else:
            assert writes == 1, c


def test_writes_follow_the_scanners_rules():
    fake = FakeHid()
    control = CameraControl(fake, gap_s=0.03)
    control.assume(STARTUP_REGISTERS)
    assert control.apply_now(CameraSettings.from_preset("general").registers()) == 0   # startup already holds it
    dark = CameraSettings.from_preset("dark").registers()
    assert control.apply_now(dark) == 4
    cmds = fake.commands
    _check_rules(cmds)
    assert cmds[0] == "echo s 0x910 8000 > /dev/rk_preisp; echo s 0x911 1000 > /dev/rk_preisp"
    assert "echo s 0xb08 801 > /dev/rk_preisp" in cmds and "echo s 0x903 0x20 > /dev/rk_preisp" in cmds
    assert "echo s 0xb07 3 > /dev/rk_preisp" in cmds
    times = [t for t, _ in fake.log]
    assert min(np.diff(times)) >= 0.03 - 1e-3                            # paced, one register at a time
    # nothing that already holds its value is written again
    assert control.apply_now(dark) == 0
    control.apply_now({**dark, "gain": 3})
    assert fake.commands[-1] == "echo s 0x903 0x30 > /dev/rk_preisp" and len(fake.commands) == 5
    # back to a shorter exposure: the shorter laser pulse goes first, the exposure last
    plan = [c for _, c in write_plan(control.applied, CameraSettings.from_preset("general").registers())]
    assert "0xb08" in plan[0] and "0x911" in plan[-1]


def test_live_changes_are_coalesced_on_the_writer_thread():
    fake = FakeHid()
    applied = []
    control = CameraControl(fake, gap_s=0.02, on_applied=applied.append)
    control.assume(STARTUP_REGISTERS)
    control.start()
    try:
        base = CameraSettings.from_preset("general")
        for level in range(20, 52):                                       # a slider being dragged
            control.request(base.with_changes({"laser_level": level}).registers())
        deadline = time.monotonic() + 3
        while (control.pending() or control.applied["pulse"] != base.with_changes({"laser_level": 51}).pulse) \
                and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        control.stop()
    assert control.applied == {**STARTUP_REGISTERS, "pulse": 245}
    assert len(fake.commands) < 32 and all("0xb08" in c for c in fake.commands)   # coalesced; only the pulse
    _check_rules(fake.commands)
    assert applied, "the auto exposure is told when the last change took effect"


def test_shell_commands_are_paced_even_from_two_threads(monkeypatch):
    dev = object.__new__(hidmod.MetroyHid)                 # no device: send() is replaced
    dev._lock, dev._last_shell, dev.min_gap_s = threading.RLock(), 0.0, 0.05
    sent = []
    monkeypatch.setattr(dev, "send", lambda *a, **k: sent.append(time.perf_counter()), raising=False)
    threads = [threading.Thread(target=lambda: [dev.preisp(0xB07, 5) for _ in range(3)]) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sent) == 6 and min(np.diff(sorted(sent))) >= 0.05 - 2e-3


# ----------------------------------------------------------------------------------------------- automatic exposure
def frames(ae, t0, n, peak, sat=0.0, depth=300.0, dt=0.03):
    for i in range(n):
        ae.observe(Light(t0 + i * dt, 400, peak, sat, depth, 2000))
    return t0 + n * dt


def decide(ae, t, peak, sat=0.0, depth=300.0):
    """One 0.8 s interval of frames, then the decision; the scanner confirms a change at once."""
    t = frames(ae, t, 30, peak, sat, depth)
    new = ae.step(t)
    if new is not None:
        ae.applied(t)
    return new, t + 0.01


def test_auto_raises_the_laser_then_the_gain_with_the_marker_light():
    ae = AutoExposure(CameraSettings.from_preset("general"), None, markers=True)
    ae.step(0.0)
    new, t = decide(ae, 0.0, peak=165)                    # dim: the laser goes up first, to its maximum
    assert new.laser_level == 51 and new.gain == 1 and ae.status.state == "adjusting"
    new, t = decide(ae, t, peak=170)                      # laser at full and still dim: one gain step
    assert new.gain == 2 and new.laser_level < 51
    assert new.marker_light == MARKER_LIGHT_BY_GAIN["general"][1] == 21   # markers keep their brightness
    assert "gain" in ae.status.message
    new, t = decide(ae, t, peak=200)                      # on target: steady
    assert new is None and ae.status.state == "steady" and ae.status.verdict == "good"


def test_auto_lowers_when_too_bright_and_drops_gain_first():
    ae = AutoExposure(CameraSettings.from_preset("dark").with_changes({"mode": "auto"}), None)
    ae.step(0.0)
    new, t = decide(ae, 0.0, peak=255, sat=0.12)          # washed out: down
    assert new.laser_level < 193 and new.gain == 2 and "washed out" in ae.status.message
    ae2 = AutoExposure(CameraSettings.from_preset("dark").with_changes({"gain": 3, "mode": "auto"}), None)
    ae2.step(0.0)
    new, _ = decide(ae2, 0.0, peak=245)                   # bright at gain 3: the laser can take it at gain 2
    assert new.gain == 2 and new.marker_light == MARKER_LIGHT_BY_GAIN["dark"][1]


def test_auto_ignores_frames_out_of_range_and_before_a_change_took_effect():
    ae = AutoExposure(CameraSettings.from_preset("general"), None)
    ae.step(0.0)
    new, t = decide(ae, 0.0, peak=60, depth=400.0)        # 400 mm away: Revo acts only between 220 and 380 mm
    assert new is None and ae.status.state == "waiting" and "220-380" in ae.status.message
    new, t = decide(ae, t, peak=60, depth=200.0)
    assert new is None
    ae.observe(Light(t, 3, 60, 0, 300, 10))               # too few stripe centres to judge
    assert ae.step(t + 0.9) is None
    # a change is in flight: frames captured before the scanner confirmed it do not count
    t = t + 0.9
    new = None
    t = frames(ae, t, 30, 165)
    new = ae.step(t)
    assert new is not None
    frames(ae, t, 25, 255, sat=0.5)                       # old frames still arriving (not yet applied)
    assert ae.step(t + 0.85) is None and ae.status.message == "Applying the last change"


def test_manual_mode_never_changes_anything():
    cs = CameraSettings.from_preset("general").with_changes({"laser_level": 20})
    ae = AutoExposure(cs, None)
    ae.step(0.0)
    t = 0.0
    for peak in (40, 255, 120):
        new, t = decide(ae, t, peak=peak, sat=0.5 if peak == 255 else 0.0)
        assert new is None and ae.settings == cs and ae.status.state == "off"
    assert ae.status.verdict == "dim"                      # it still says what it sees


def test_auto_says_when_the_surface_preset_is_wrong():
    ae = AutoExposure(CameraSettings.from_preset("general").with_changes({"laser_level": 51, "gain": 5,
                                                                           "mode": "auto"}), None)
    ae.step(0.0)
    new, _ = decide(ae, 0.0, peak=40)
    assert new is None and ae.status.state == "limit" and "Dark" in ae.status.message


def test_measure_light_reads_the_stripe_centres():
    img = np.full((1200, 1600), 16, np.uint8)
    xs = np.linspace(300, 1300, 500)
    ys = np.linspace(200, 1000, 500).round()
    img[ys.astype(int), np.floor(xs).astype(int)] = 180
    img[ys.astype(int)[:50], np.floor(xs).astype(int)[:50] + 1] = 255   # a few washed out
    pts = np.c_[np.zeros((500, 2)), np.full(500, 300.0)]
    light = measure_light(img, xs, ys, pts, t=1.0)
    assert light.centres > 300 and light.peak == 255 and 0.05 < light.saturated < 0.2
    assert light.depth_mm == 300.0 and light.points == 500
    assert measure_light(img, np.zeros(0), np.zeros(0), np.zeros((0, 3))).centres == 0


# ----------------------------------------------------------------------------------------------- the picture
def test_compose_view_both_cameras_with_overlay():
    grey = np.full((36, 48), 40, np.uint8)
    sat = np.zeros((36, 48), bool)
    sat[5, 5] = True
    view = {"shape": grey.shape, "left": grey, "right": grey, "sat_left": np.packbits(sat),
            "sat_right": np.packbits(np.zeros_like(sat)), "centres_left": np.array([[10, 10]], np.int16),
            "centres_right": np.zeros((0, 2), np.int16), "markers_left": np.array([[20, 20, 3]], np.float32),
            "markers_right": np.zeros((0, 3), np.float32)}
    both = compose_view(view, "both", True, width=100)
    assert both.shape[1] == 100 and both.shape[2] == 3
    full = compose_view(view, "left", True, width=48)
    assert tuple(full[5, 5]) == (255, 64, 64) and tuple(full[10, 10]) == (76, 214, 120)
    plain = compose_view(view, "left", False, width=48)
    assert tuple(plain[5, 5]) == (40, 40, 40)
    assert encode_jpeg(both)[:2] == b"\xff\xd8"


def test_preview_lease_ends_when_nobody_looks():
    ended = []
    lease = PreviewLease(lambda: ended.append(time.monotonic()), lease_s=0.3)
    lease.renew()
    time.sleep(0.2)
    lease.renew()                                          # still looking
    time.sleep(0.2)
    assert not ended and lease.active
    time.sleep(0.6)
    assert len(ended) == 1 and not lease.active


# ----------------------------------------------------------------------------------------------- MetroY driver
class FakeScanner:
    """Stands in for MetroyScanner: records the lifecycle the driver asks for."""

    def __init__(self):
        from cloudclean.capture.metroy.exposure import AutoExposure as AE
        self.calls = []
        self.running = False
        self.delivering = True
        self.failed = None
        self.camera = CameraSettings.from_preset("general")
        self.ae = AE(self.camera)

    def start(self, deliver=True):
        time.sleep(0.05)
        self.calls.append(("start", deliver))
        self.running, self.delivering = True, deliver

    def begin_delivery(self):
        self.calls.append(("deliver",))
        self.delivering = True

    def stop(self):
        self.calls.append(("stop",))
        self.running = False

    def set_camera(self, changes):
        self.camera = self.camera.with_changes(changes)
        return self.camera

    def camera_status(self):
        from cloudclean.capture.metroy.camera import camera_payload
        return camera_payload(self.camera, self.ae.status, {}, streaming=self.running,
                              preview=self.running and not self.delivering)

    def camera_view(self, cams="both", overlay=True, width=960):
        return b"\xff\xd8jpeg" if self.running else None


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_metroy_preview_becomes_the_scan_and_ends_with_its_lease():
    drv = create_driver("metroy_usb")
    drv._scanner = sc = FakeScanner()
    drv.connected = True
    drv._lease.lease_s = 0.4
    drv.camera_preview(True)
    assert _wait(lambda: sc.running)
    assert sc.calls == [("start", False)] and drv.camera()["preview"]
    assert drv.camera_view() == b"\xff\xd8jpeg"
    drv.start()                                            # Start scanning: the same stream, no restart
    assert sc.calls[-1] == ("deliver",) and drv.running
    time.sleep(0.8)
    assert sc.running, "the lease never stops a scan"
    drv.stop()
    assert sc.calls[-1] == ("stop",)
    # a preview nobody looks at ends by itself: the laser does not stay on
    drv.camera_preview(True)
    assert _wait(lambda: sc.running)
    assert _wait(lambda: not sc.running, timeout=3.0) and sc.calls[-1] == ("stop",)
    # settings are remembered with the driver settings
    out = drv.set_camera({"surface": "dark"})
    assert out["settings"]["laser_pulse"] == 801 and drv.settings["surface"] == "dark"


# ----------------------------------------------------------------------------------------------- scanner + workers
def _plane_frame(z_plane=300.0, peak=235.0):
    """A stereo frame (left on top) of the 17 laser sheets on a flat wall z_plane mm away, through the real
    rectification maps' camera model."""
    from pathlib import Path

    from cloudclean.capture.metroy.stripes import Triangulator
    from tests.test_metroy import _laser_planes
    data = Path(__file__).parent / "data" / "metroy"
    tri = Triangulator.from_yaml(str(data / "camparam.yaml"))
    normals, offsets = _laser_planes()
    f, cy = tri.P1[0, 0], tri.P1[1, 2]
    v, u = np.mgrid[0:1200, 0:1600].astype(np.float64)
    imgs = []
    for origin, cx in ((0.0, tri.P1[0, 2]), (tri.B, tri.P2[0, 2])):
        X = np.c_[((u - cx) / f).ravel() * z_plane + origin, ((v - cy) / f).ravel() * z_plane,
                  np.full(u.size, z_plane)]
        stripe = peak * np.exp(-0.5 * ((X @ normals.T - offsets) / 0.5) ** 2).max(1)
        img = 20 + stripe + np.random.default_rng(int(origin)).normal(0, 2, u.size)
        imgs.append(np.clip(img, 0, 255).astype(np.uint8).reshape(1200, 1600))
    # drawn in the rectified camera model and then rectified once more by the workers: the depth comes out
    # somewhat off 300 mm (order matching, no exact geometry here), but inside the auto exposure window
    return (data / "camparam.yaml").read_bytes(), np.vstack(imgs)


class RawFrame:                                   # as v4l2.RawFrame (which needs Linux to import)
    def __init__(self, image, timestamp, sequence):
        self.image, self.timestamp, self.sequence = image, timestamp, sequence


class FakeStream:
    def __init__(self, image, fps=40.0):
        self.image, self.fps, self.seq = image, fps, 0

    def read(self, timeout=0.5):
        time.sleep(1.0 / self.fps)
        self.seq += 1
        return RawFrame(self.image, time.monotonic(), self.seq)

    def close(self):
        pass


@pytest.fixture(scope="module")
def plane_frame():
    pytest.importorskip("cv2")
    return _plane_frame()


def test_scanner_preview_measures_and_auto_exposes_then_delivers(plane_frame, monkeypatch):
    """The whole frame path with fakes: workers triangulate and measure, auto exposure writes the laser pulse through
    the paced writer while streaming, the camera view comes back as a JPEG, nothing is handed out in preview, and
    begin_delivery() makes the same stream a scan."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from cloudclean.capture.metroy import scanner as scmod
    calib, image = plane_frame
    monkeypatch.setattr(scmod, "PREVIEW_FPS", 30.0)
    sc = scmod.MetroyScanner(workers=2, matching="order", markers=True)
    fake = sc.hid = FakeHid()
    sc.stream = FakeStream(image)
    sc.files = scmod.DeviceFiles(calib, None, "test")
    sc.pool = ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn"),
                                  initializer=scmod._init_worker, initargs=(calib, None, {}))
    try:
        sc.start(deliver=False)
        assert sc.hid.commands[0] == "startup" and sc.hid.commands[-1] == "echo s 0xb04 1 > /dev/rk_preisp"
        view = None
        end = time.monotonic() + 60
        while time.monotonic() < end and (view is None or sc.camera.laser_level == 44):
            view = sc.camera_view("both", True, 640) or view
            time.sleep(0.1)
        status = sc.camera_status()
        assert view is not None and view[:2] == b"\xff\xd8"
        r = status["readout"]
        assert r["stripe_brightness"] > 200 and r["points_per_frame"] > 1000 and 220 < r["depth_mm"] < 380, r
        assert sc.camera.laser_level < 44, "too bright stripes: auto exposure lowers the laser"
        assert sc.next(timeout=0.05) is None and sc.stats["preview_frames"] > 0          # preview: nothing out
        live = [c for c in sc.hid.commands if "0xb08" in c]
        assert live and all(c.count("echo s") == 1 for c in live)
        _check_rules(sc.hid.commands[1:])
        sc.begin_delivery()
        frame = None
        end = time.monotonic() + 30
        while frame is None and time.monotonic() < end:
            frame = sc.next(timeout=0.2)
        assert frame is not None and len(frame.points) > 1000 and "stripe_peak" in frame.info
    finally:
        sc.close()
    # the laser is switched off when the scanner closes
    assert fake.commands[-1] == "close" and "echo s 0xb04 0 > /dev/rk_preisp" in fake.commands[-4:]


# ----------------------------------------------------------------------------------------------- simulated + API
def test_simulated_camera_auto_exposure_finds_the_dark_part():
    drv = create_driver("simulated")
    drv.connect({"realtime": False, "surface": "dark", "camera_surface": "general", "simulate_mistakes": False,
                 "turntable": True})
    cam = drv._cam
    t = 0.0
    for i in range(160):
        cam.observe(drv.render(i * 0.05), now=t)
        t += 0.05
    assert cam.ae.status.state == "limit" and "Dark" in cam.ae.status.message     # General cannot light a black part
    drv.set_camera({"surface": "dark", "mode": "auto"})
    for i in range(400):
        cam.observe(drv.render(8 + i * 0.05), now=t)
        t += 0.05
    summary = cam.readout.summary()
    assert cam.camera.surface == "dark" and cam.camera.gain >= 3                   # as Revo Metro on the screw
    assert 175 <= summary["stripe_brightness"] <= 230 and cam.ae.status.verdict == "good", summary


def test_camera_api_round_trip_with_the_simulated_scanner(tmp_path):
    with TestClient(create_app(tmp_path / "ws")) as client:
        assert client.get("/api/capture/camera").json()["available"] is False      # nothing connected
        r = client.post("/api/capture/connect", json={"driver": "simulated", "settings": {
            "realtime": False, "max_frames": 40, "surface": "normal"}})
        assert r.status_code == 200, r.text
        cam = client.get("/api/capture/camera").json()
        assert cam["available"] and cam["settings"]["surface"] == "general" and cam["settings"]["mode"] == "auto"
        assert cam["settings"]["laser_pulse"] == 217 and cam["limits"]["gain"] == [1, 5]
        dark = client.post("/api/capture/camera", json={"surface": "dark"}).json()["settings"]
        assert (dark["laser_level"], dark["laser_pulse"], dark["exposure_us"], dark["gain"]) == (193, 801, 1000, 2)
        manual = client.post("/api/capture/camera", json={"laser_level": 120, "gain": 3}).json()["settings"]
        assert manual["mode"] == "manual" and manual["laser_level"] == 120 and manual["gain"] == 3
        bad = client.post("/api/capture/camera", json={"gain": 9})
        assert bad.status_code == 400 and "Gain" in bad.json()["detail"]
        # remembered with the driver settings, for the next connect
        last = client.get("/api/capture/drivers").json()["last_settings"]["simulated"]
        assert last["camera_surface"] == "dark" and last["camera_mode"] == "manual" and last["laser_level"] == 120
        assert client.get("/api/capture/status").json()["settings"]["gain"] == 3

        # the camera before scanning: a preview, pictures while someone asks, nothing fused
        assert client.post("/api/capture/camera/preview", json={"on": True}).json()["preview"]

        def picture():
            res = client.get("/api/capture/camera/view?cams=both&overlay=1&width=640")
            return res if res.status_code == 200 else None
        res = None
        end = time.monotonic() + 10
        while res is None and time.monotonic() < end:
            res = picture()
            time.sleep(0.1)
        assert res is not None and res.headers["content-type"] == "image/jpeg" and res.content[:2] == b"\xff\xd8"
        assert client.get("/api/capture/camera/view?cams=sideways").status_code == 400
        readout = client.get("/api/capture/camera").json()["readout"]
        assert readout["points_per_frame"] and readout["stripe_brightness"] is not None
        status = client.get("/api/capture/status").json()
        assert status["points"] == 0 and status["device"]["camera"]["preview"]
        assert client.post("/api/capture/camera/preview", json={"on": False}).json()["preview"] is False

        # scanning: the camera is live too, and settings can still change
        assert client.post("/api/capture/start").json()["state"] == "running"
        assert client.post("/api/capture/camera", json={"mode": "auto"}).json()["settings"]["mode"] == "auto"
        client.post("/api/capture/stop")
        client.post("/api/capture/discard")

        # a scanner without a camera says so
        manager = manager_for(tmp_path / "ws")
        manager.connect("revo_bridge", {})
        info = client.get("/api/capture/camera").json()
        assert info["available"] is False and "no camera view" in info["reason"]
        assert client.post("/api/capture/camera", json={"gain": 2}).status_code == 400
        assert client.post("/api/capture/camera/preview", json={"on": True}).status_code == 400
        client.post("/api/capture/discard")
