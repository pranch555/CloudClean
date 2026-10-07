"""Turntable (Contract 4): protocol encoder/decoder, simulated driver, Bluetooth driver against a fake peripheral,
manager, step-and-scan programs linked to capture, routes, and the simulated scanner following the platter."""
import asyncio
import math
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cloudclean.capture.turntable import ble, program as prog, protocol as P
from cloudclean.capture.turntable import simulated as simtt
from cloudclean.capture.turntable.base import TurntableDriver, TurntableError
from cloudclean.capture.turntable.manager import TurntableManager, get_turntable_manager
from cloudclean.capture.turntable.simulated import SimulatedTurntable

FAST = 100.0   # simulated clock speed-up


@pytest.fixture(autouse=True)
def _no_leaked_simulated_turntable():
    yield
    with simtt._ACTIVE_LOCK:
        tt = simtt._ACTIVE() if simtt._ACTIVE is not None else None
    if tt is not None:
        tt.disconnect()
    simtt._ACTIVE = None


def wait_for(pred, timeout=10.0, step=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


# --------------------------------------------------------------------------- protocol
def test_name_classification_and_characteristic():
    assert P.classify_name("REVO_DUAL_AXIS_TABLE") == "dual_axis"
    assert P.classify_name("REVO_TA500") == "large"
    assert P.classify_name("REVO_STABILIZER") == "stabilizer"
    assert P.classify_name("x RevoStabilizer y") == "stabilizer"
    assert P.classify_name("REVO_DUAL_AXIS_TABLE ") is None      # Revo Metro compares exactly
    assert P.classify_name(None) is None and P.classify_name("Galaxy Watch4") is None
    assert P.is_data_characteristic("0000ffe1-0000-1000-8000-00805f9b34fb")
    assert P.is_data_characteristic("0000FFE1-0000-1000-8000-00805F9B34FB")
    assert P.is_data_characteristic("ffe1")
    assert not P.is_data_characteristic("0000ffe2-0000-1000-8000-00805f9b34fb")
    assert not P.is_data_characteristic("0000ffb1-0000-1000-8000-00805f9b34fb")
    assert P.is_data_characteristic("0000ffb1-0000-1000-8000-00805f9b34fb", P.STABILIZER_CHAR_SHORT)
    assert not P.is_data_characteristic("not-a-uuid")


def test_dual_axis_encoders_match_revo_metro():
    # exact strings seen in Revo Metro's logs of the user's table
    assert P.dual_turn(-10) == "+CT,TURNANGLE=-10;"
    assert P.dual_turn(10) == "+CT,TURNANGLE=10;"
    assert P.dual_turn_continuous(False) == "+CT,TURNCONTINUE=-1;"
    assert P.dual_turn_continuous(True) == "+CT,TURNCONTINUE=1;"
    assert P.dual_tilt(-1) == "+CR,TILTVALUE=-1;"
    assert P.dual_tilt(0) == "+CR,TILTVALUE=0;"
    assert P.dual_speed("turn", 53) == "+CT,TURNSPEED=53;"
    assert P.dual_speed("tilt", 10) == "+CR,TILTSPEED=10;"
    assert P.dual_stop("turn") == "+CT,STOP;" and P.dual_stop("tilt") == "+CR,STOP;"
    assert P.dual_to_zero("turn") == "+CT,TOZERO;" and P.dual_to_zero("tilt") == "+CR,TOZERO;"
    assert P.QUERY_VERSION == "+QR,VERSION;" and P.QUERY_TURN_POSITION == "+QT,CHANGEANGLE;"
    assert P.QUERY_TILT_POSITION == "+QR,TILTVALUE;" and P.QUERY_TURN_SPEED_RANGE == "+QT,TURNSPEED;"


def test_large_turntable_encoders():
    assert P.large_turn(30) == "CT+TRUNSINGLE(0,30);"
    assert P.large_turn(-30) == "CT+TRUNSINGLE(1,30);"
    assert P.large_turn_continuous(True) == "CT+START(0,0,0,1,0,0);"
    assert P.large_turn_continuous(False) == "CT+START(1,0,0,1,0,0);"
    # Revo Metro's s/rev -> level mapping (RevoMetro.exe 0x140818c20)
    assert [P.large_speed_level(s) for s in (20, 34, 35, 38, 41, 60, 90, 92, 93, 200)] == [1, 1, 1, 2, 3, 9, 19, 20, 20, 20]
    assert P.large_speed(50) == "CT+SETSPEED(6);"
    assert P.LARGE_STOP == "CT+SETSTOP();" and P.LARGE_TO_ZERO == "CT+TOZERO();"


def test_whole_degrees():
    assert P.whole_degrees(10.4) == 10 and P.whole_degrees(-10.6) == -11 and P.whole_degrees(7) == 7
    with pytest.raises(ValueError):
        P.whole_degrees(float("nan"))


def test_reply_parser():
    r = P.parse_reply("+OK;")
    assert r.kind == "ok"
    r = P.parse_reply("+OK,TURNANGLE=-145.72")           # real reply, no ';'
    assert r.kind == "ok_field" and r.field == "TURNANGLE" and r.number == pytest.approx(-145.72)
    r = P.parse_reply("+OK,TOZERO;")
    assert r.kind == "ok_field" and r.field == "TOZERO" and r.value is None and r.number is None
    r = P.parse_reply("+FAIL,ERR=007;")                  # real reply to +CR,TILTSPEED=10;
    assert r.kind == "fail" and r.code == "007"
    r = P.parse_reply("+DATA=16,90,53;")
    assert r.kind == "data" and r.numbers == [16.0, 90.0, 53.0]
    r = P.parse_reply(" +DATA=-1\r\n")
    assert r.kind == "data" and r.number == -1.0
    assert P.parse_reply("CR+OK;").kind == "ok"
    assert P.parse_reply("hello").kind == "unknown"


def test_version_parsing_and_completion_switch():
    assert P.parse_version(P.parse_reply("+DATA=DUALAXIS,V3.28;")) == "3.28"
    # the user's table: hardware version first, the firmware second (2026-10-06)
    assert P.parse_version(P.parse_reply("+DATA=HW.V0.02,SW.V3.28;")) == "3.28"
    assert P.reports_completion(P.parse_version("HW.V0.02,SW.V3.28")) is True
    assert P.parse_version("V3.27") == "3.27"
    assert P.parse_version("3.30") == "3.30"
    assert P.parse_version(None) is None and P.parse_version("+DATA=") is None
    assert P.reports_completion("3.28") is True
    assert P.reports_completion("3.40") is True
    assert P.reports_completion("3.27") is False
    assert P.reports_completion(None) is None


def test_message_assembler():
    a = P.MessageAssembler()
    assert a.feed(b"+OK;") == ["+OK;"]
    assert a.feed(b"+OK;+OK,TOZERO;") == ["+OK;", "+OK,TOZERO;"]
    assert a.feed(b"+OK,TURNANGLE=-145.72") == [] and a.flush() == ["+OK,TURNANGLE=-145.72"]
    assert a.feed(b"+OK,TURNANGLE=-145.7") == []          # split across two notifications
    assert a.feed(b"2;") == ["+OK,TURNANGLE=-145.72;"]
    assert a.feed(b"+OK,TURNANGLE=1") == []
    assert a.feed(b"+OK;") == ["+OK,TURNANGLE=1", "+OK;"]  # a new message releases the pending one
    assert a.feed(b"CR+OK;\r\n") == ["CR+OK;"]
    assert a.feed(b"") == [] and a.flush() == []


# --------------------------------------------------------------------------- simulated driver
def test_simulated_timing_and_stop():
    tt = SimulatedTurntable(time_scale=20.0)
    with pytest.raises(TurntableError):
        tt.rotate(10)                                       # not connected
    tt.connect()
    tt.set_speed(30)
    t0 = time.monotonic()
    res = tt.rotate(30)
    dt = time.monotonic() - t0
    nominal = (30 / 360 * 30 + 2 * 0.04) / 20.0            # 2.58 s of turntable time, 20x faster
    assert res["angle_deg"] == pytest.approx(30.0) and not res["stopped"]
    assert nominal * 0.8 < dt < nominal + 0.25
    with pytest.raises(TurntableError):
        tt.set_speed(10)                                    # below 25 s/rev
    with pytest.raises(TurntableError):
        tt.tilt(31)

    # stop from another thread in the middle of a long move
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", tt.rotate(-360)))
    th.start()
    assert wait_for(lambda: tt.state()["rotating"])
    time.sleep(0.05)
    stopped = tt.stop()
    th.join(2)
    assert out["r"]["stopped"] and stopped["stopped"]
    angle = tt.state()["angle_deg"]
    assert -330.0 < angle < 30.0                            # frozen part-way between 30 and -330
    assert not tt.state()["moving"]
    time.sleep(0.05)
    assert tt.state()["angle_deg"] == pytest.approx(angle)  # stays put

    res = tt.tilt(-12)
    assert res["tilt_deg"] == pytest.approx(-12.0) and tt.state()["tilt_deg"] == pytest.approx(-12.0)
    tt.rotate_continuous(True)
    time.sleep(0.05)
    assert tt.state()["continuous"]
    tt.stop()
    assert not tt.state()["moving"]
    assert "+CT,TURNANGLE=30;" in tt.commands and "+CR,TILTVALUE=-12;" in tt.commands
    assert simtt.follow_pose() is not None
    tt.disconnect()
    assert simtt.follow_pose() is None


def test_simulated_state_is_thread_safe():
    tt = SimulatedTurntable(time_scale=FAST)
    tt.connect()
    errors = []

    def reader():
        for _ in range(300):
            try:
                st = tt.state()
                assert -1e-6 <= st["angle_deg"] <= 720 + 1e-6
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(4):
        tt.rotate(180)
    for t in threads:
        t.join()
    assert not errors
    assert tt.state()["angle_deg"] == pytest.approx(720.0)
    tt.disconnect()


# --------------------------------------------------------------------------- fake Bluetooth peripheral
class FakeDualAxisPeripheral:
    """Replays the dual-axis protocol the way firmware 3.28 behaved in Revo Metro's logs."""

    def __init__(self, notify, version="V3.28", time_scale=0.01, split_turn_reply=False, ack_turns=False):
        self.notify = notify
        self.version = version
        self.time_scale = time_scale
        self.split = split_turn_reply
        self.ack_turns = ack_turns
        self.is_connected = True
        self.mtu_size = 247
        self.received: list[str] = []
        self.angle = -135.72
        self.tilt = 0.0
        self.speed = 53
        self._turn_handle = None

    async def write_gatt_char(self, char, data, response=False):
        assert response is False, "Revo Metro writes without response"
        assert P.is_data_characteristic(char)
        self.handle(bytes(data).decode("ascii"))

    async def start_notify(self, char, cb):
        pass

    async def stop_notify(self, char):
        pass

    async def disconnect(self):
        self.is_connected = False

    def _send(self, text, delay=0.005):
        loop = asyncio.get_running_loop()
        loop.call_later(delay, self.notify, None, bytearray(text.encode("ascii")))

    def handle(self, cmd):
        self.received.append(cmd)
        body = cmd.rstrip(";")
        if body == "+QR,VERSION":
            self._send(f"+DATA=DUALAXIS,{self.version};")
        elif body == "+QT,CHANGEANGLE":
            self._send(f"+DATA={self.angle:.2f}")
        elif body == "+QR,TILTVALUE":
            self._send(f"+DATA={self.tilt:.0f}")
        elif body in ("+QT,TURNSPEED", "+QR,TILTANGLE"):
            self._send("+DATA=16,90,53;" if "SPEED" in body else "+DATA=-30,30,0;")
        elif body.startswith("+CT,TURNSPEED="):
            self.speed = int(body.split("=")[1])
            self._send("+OK;")
        elif body.startswith("+CR,TILTSPEED="):
            self._send("+FAIL,ERR=007;")
        elif body.startswith("+CT,TURNANGLE="):
            deg = int(body.split("=")[1])
            if deg > 400:
                self._send("+FAIL,ERR=003;")
                return
            if self.ack_turns:
                self._send("+OK;")
            duration = abs(deg) / 360 * self.speed * self.time_scale
            loop = asyncio.get_running_loop()
            self._turn_handle = loop.call_later(duration, self._finish_turn, deg)
        elif body.startswith("+CR,TILTVALUE="):
            self.tilt = float(body.split("=")[1])
            self._send("+OK;")
        elif body in ("+CT,STOP", "+CR,STOP"):
            if body == "+CT,STOP" and self._turn_handle is not None:
                self._turn_handle.cancel()
                self._turn_handle = None
                self.angle += 1.5                     # stopped part-way
                self._send(f"+OK,TURNANGLE={self.angle:.2f}")
            else:
                self._send("+OK;")
        else:
            self._send("+FAIL,ERR=001;")

    def _finish_turn(self, deg):
        self._turn_handle = None
        self.angle += deg
        text = f"+OK,TURNANGLE={self.angle:.2f}"
        if self.split:
            self.notify(None, bytearray(text[:12].encode()))
            self._send(text[12:], 0.01)
        else:
            self.notify(None, bytearray(text.encode()))


@pytest.fixture
def fake_ble(monkeypatch):
    """RevopointBleTurntable with its bleak layer replaced by FakeDualAxisPeripheral."""
    created = []
    options = {}

    async def fake_aconnect(self):
        client = FakeDualAxisPeripheral(self._on_notify, **options)
        self._client, self._char = client, P.DATA_CHAR_UUID
        self.gatt = [{"uuid": P.DATA_SERVICE_UUID, "characteristics": [{"uuid": P.DATA_CHAR_UUID,
                                                                         "properties": ["write-without-response",
                                                                                        "notify"]}]}]
        created.append(client)

    monkeypatch.setattr(ble, "_bleak_state", (True, ""))
    monkeypatch.setattr(ble.RevopointBleTurntable, "_aconnect", fake_aconnect)
    return created, options


def test_ble_driver_against_fake_peripheral(fake_ble):
    created, _ = fake_ble
    drv = ble.RevopointBleTurntable("E4:8F:80:46:12:43", "REVO_DUAL_AXIS_TABLE", "dual_axis")
    assert drv.validated is True      # driven on the user's table, 2026-10-06
    assert ble.RevopointBleTurntable("AA:BB:CC:DD:EE:FF", "REVO_TA500", "large").validated is False
    drv.connect()
    dev = created[0]
    assert drv.firmware == "3.28" and P.reports_completion(drv.firmware)
    st = drv.state()
    assert st["angle_deg"] == pytest.approx(-135.72) and st["tilt_deg"] == pytest.approx(0.0)
    assert drv.reported["turn_speed_range"]["numbers"] == [16.0, 90.0, 53.0]
    assert dev.received[:1] == ["+QR,VERSION;"]
    assert not any(c.startswith(("+CT,TURN", "+CR,TILT")) and "=" in c for c in dev.received), \
        "connecting must not move the table or change its settings"

    assert drv.set_speed(40)["speed_s_per_rev"] == 40 and dev.received[-1] == "+CT,TURNSPEED=40;"
    res = drv.rotate(-10)
    assert dev.received[-1] == "+CT,TURNANGLE=-10;"
    assert res["angle_deg"] == pytest.approx(-145.72) and not res["stopped"]

    res = drv.tilt(-5)
    assert "+CR,TILTVALUE=-5;" in dev.received and res["tilt_deg"] == pytest.approx(-5.0) and res["confirmed"]

    with pytest.raises(TurntableError, match="error 007"):
        drv._command(P.dual_speed("tilt", 10))           # the refusal Revo Metro's log shows

    with pytest.raises(TurntableError, match="refused"):
        drv.rotate(500)

    # emergency stop from another thread while a 180 deg turn is running
    dev.time_scale = 1.0
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", drv.rotate(180)))
    th.start()
    assert wait_for(lambda: drv.state()["rotating"])
    time.sleep(0.1)
    stop = drv.stop()
    th.join(5)
    assert out["r"]["stopped"] and stop["stopped"]
    assert "+CT,STOP;" in dev.received and "+CR,STOP;" in dev.received
    assert drv.state()["angle_deg"] == pytest.approx(-145.72 + 1.5)
    drv.disconnect()
    assert not drv.connected and not dev.is_connected
    with pytest.raises(TurntableError):
        drv.rotate(10)


def test_ble_driver_split_reply_and_old_firmware(fake_ble):
    created, options = fake_ble
    options.update(split_turn_reply=True)
    drv = ble.RevopointBleTurntable("AA:BB:CC:DD:EE:FF", None, "dual_axis")
    drv.connect()
    assert drv.rotate(20)["angle_deg"] == pytest.approx(-115.72)   # reassembled from two notifications
    drv.disconnect()

    options.clear()
    options.update(version="V3.20", ack_turns=True)
    old = ble.RevopointBleTurntable("AA:BB:CC:DD:EE:00", None, "dual_axis")
    old.connect()
    assert P.reports_completion(old.firmware) is False
    res = old.rotate(10)                                # ack + nominal time + position query
    assert res["angle_deg"] == pytest.approx(-125.72)
    old.disconnect()


def test_ble_scan_requires_bleak(monkeypatch):
    monkeypatch.setattr(ble, "_bleak_state", (False, ble.INSTALL_HINT))
    with pytest.raises(TurntableError, match="cloudclean\\[turntable\\]"):
        ble.scan(1)
    drv = ble.RevopointBleTurntable("AA:BB:CC:DD:EE:FF")
    with pytest.raises(TurntableError, match="bleak"):
        drv.connect()


# --------------------------------------------------------------------------- manager
def test_manager_devices_connect_and_moves(tmp_path, monkeypatch):
    m = TurntableManager(tmp_path)
    monkeypatch.setattr(m, "scan_fn", lambda s: [{"id": "E4:8F:80:46:12:43", "address": "E4:8F:80:46:12:43",
                                                  "name": "REVO_DUAL_AXIS_TABLE", "kind": "dual_axis", "rssi": -60}])
    devs = m.devices(1.0)
    assert devs[0]["id"] == "simulated" and devs[0]["kind"] == "simulated"
    assert devs[1]["name"] == "REVO_DUAL_AXIS_TABLE" and devs[1]["kind"] == "dual_axis" and devs[1]["rssi"] == -60
    assert m.status()["bluetooth"]["available"] is True
    assert [d["id"] for d in m.devices(0)] == ["simulated"]     # no scan

    st = m.status()
    assert st["connected"] is False and st["kind"] is None and st["validated"] is False
    with pytest.raises(TurntableError, match="No turntable is connected"):
        m.rotate(10)
    with pytest.raises(TurntableError):
        m.connect("simulated", options={"time_scale": 0})

    st = m.connect("simulated", options={"time_scale": FAST})
    assert st["connected"] and st["kind"] == "simulated" and st["validated"] is True
    assert st["capabilities"]["tilt_range"] == [-30, 30] and st["capabilities"]["speed_range"] == [25, 90]

    st = m.rotate(10.4, speed_s_per_rev=30, wait=True)
    assert st["move"]["commanded_deg"] == 10 and "whole degrees" in st["move"]["note"]
    assert st["angle_deg"] == pytest.approx(10.0) and st["speed_s_per_rev"] == 30 and st["direction"] == "cw"
    st = m.rotate(-20, wait=True)
    assert st["angle_deg"] == pytest.approx(-10.0) and st["direction"] == "ccw"
    assert st["angle_wrapped_deg"] == pytest.approx(350.0)
    for bad, msg in ((0, "at least 1 degree"), (0.3, "at least 1 degree"), (721, "at most 720"),
                     ("x", "number")):
        with pytest.raises(TurntableError, match=msg):
            m.rotate(bad)
    with pytest.raises(TurntableError, match="between 25 and 90"):
        m.rotate(10, speed_s_per_rev=10)

    st = m.tilt(20, wait=True)
    assert st["tilt_deg"] == pytest.approx(20.0)
    with pytest.raises(TurntableError, match="between -30 and 30"):
        m.tilt(-31)
    st = m.set_speed(44.6)
    assert st["speed"]["commanded_s_per_rev"] == 45 and st["speed_s_per_rev"] == 45

    # a move in the background, then an emergency stop
    m.set_speed(90)
    st = m.rotate(360)
    assert st["moving"] and st["move"]["state"] == "running"
    with pytest.raises(TurntableError, match="still moving"):
        m.rotate(10)
    time.sleep(0.05)
    st = m.stop()
    assert not st["moving"] and st["last_move"]["state"] == "stopped"
    assert -10.0 <= st["angle_deg"] < 350.0

    st = m.disconnect()
    assert not st["connected"] and st["kind"] is None
    m.shutdown()


def test_manager_connect_resolves_ble_devices(tmp_path, monkeypatch):
    m = TurntableManager(tmp_path)
    made = []

    class FakeDriver(SimulatedTurntable):
        def __init__(self, address, name, kind):
            super().__init__(time_scale=FAST)
            self.id, self.name, self.kind = address, name or "REVO_DUAL_AXIS_TABLE", kind
            self.validated = False
            made.append(self)

        def capabilities(self):
            return TurntableDriver.capabilities(self)       # kind-dependent (large: no tilt)

    monkeypatch.setattr(m, "ble_factory", FakeDriver)
    monkeypatch.setattr(m, "scan_fn", lambda s: [{"id": "E4:8F:80:46:12:43", "address": "E4:8F:80:46:12:43",
                                                  "name": "REVO_DUAL_AXIS_TABLE", "kind": "dual_axis", "rssi": -50}])
    st = m.connect("REVO_DUAL_AXIS_TABLE")                  # by advertised name
    assert st["device"] == "E4:8F:80:46:12:43" and st["kind"] == "dual_axis" and st["validated"] is False
    assert m.status()["remembered_device"]["id"] == "E4:8F:80:46:12:43"
    m.disconnect()
    monkeypatch.setattr(m, "scan_fn", lambda s: [])
    st = m.connect()                                         # remembered device, no scan needed
    assert st["device"] == "E4:8F:80:46:12:43"
    with pytest.raises(TurntableError, match="not a known Revopoint turntable"):
        m.connect("11:22:33:44:55:66")
    st = m.connect("11:22:33:44:55:66", kind="large")        # explicit kind is accepted
    assert st["kind"] == "large" and st["capabilities"]["tilt"] is False
    with pytest.raises(TurntableError, match="cannot tilt"):
        m.tilt(5)
    m.shutdown()


# --------------------------------------------------------------------------- programs
def test_program_validation():
    caps = SimulatedTurntable().capabilities()
    plan = prog.validate_program({"interval_deg": 30, "rotations": [{"tilt_deg": 0}, {"tilt_deg": 20}]}, caps)
    assert plan["stops_per_rotation"] == 12 and plan["moves"] == [30] * 12 and plan["direction"] == "cw"
    plan = prog.validate_program({"interval_deg": 25}, caps)
    assert plan["stops_per_rotation"] == 15 and sum(plan["moves"]) == 360 and plan["moves"][-1] == 10
    assert any("last move" in n for n in plan["notes"])
    plan = prog.validate_program({"interval_deg": 12.4, "rotations": [7.6], "direction": "counterclockwise"}, caps)
    assert plan["interval_deg"] == 12 and plan["rotations"] == [{"tilt_deg": 8}] and plan["direction"] == "ccw"
    assert len(plan["notes"]) >= 2
    plan = prog.validate_program({"mode": "continuous", "rotations": 2}, caps)
    assert plan["moves"] == [360] and len(plan["rotations"]) == 2
    bad = [({"interval_deg": 3}, "between 5 and 30"), ({"interval_deg": 45}, "between 5 and 30"),
           ({"rotations": [{"tilt_deg": 0}] * 6}, "At most 5"), ({"rotations": [{"tilt_deg": 40}]}, "between -30"),
           ({"frames_per_stop": 0}, "frames_per_stop"), ({"frames_per_stop": 2.5}, "frames_per_stop"),
           ({"direction": "up"}, "direction"), ({"speed_s_per_rev": 10}, "between 25 and 90"),
           ({"sync_scan": "maybe"}, "true or false"), ({"bogus": 1}, "Unknown program setting"),
           ({"rotations": []}, "rotations"), ({"rotations": [{"tilt": 3}]}, "only 'tilt_deg'"),
           ({"mode": "spiral"}, "mode")]
    for spec, msg in bad:
        with pytest.raises(TurntableError, match=msg):
            prog.validate_program(spec, caps)
    large = {"tilt": False, "tilt_range": None, "speed_range": [35, 90], "interval_range": [5, 30]}
    with pytest.raises(TurntableError, match="cannot tilt"):
        prog.validate_program({"rotations": [{"tilt_deg": 10}]}, large)


class FakeCapture:
    """Just enough of CaptureManager: start/pause/resume/status with a frame counter running while 'running'."""

    def __init__(self, turntable, fps=400.0):
        self.session = object()
        self.state = "connected"
        self.frames = 0
        self.events = []
        self.frames_while_moving = 0
        self.turntable = turntable
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, args=(1.0 / fps,), daemon=True)
        self._thread.start()

    def _produce(self, dt):
        while not self._stop.wait(dt):
            if self.state == "running":
                if self.turntable.state()["rotating"]:
                    self.frames_while_moving += 1
                self.frames += 1

    def status(self):
        return {"state": self.state, "frames": self.frames}

    def start(self):
        self.events.append("start")
        self.state = "running"

    def pause(self):
        self.events.append("pause")
        self.state = "paused"

    def resume(self):
        self.events.append("resume")
        self.state = "running"

    def close(self):
        self._stop.set()


def test_program_step_and_scan_with_capture_sync(tmp_path):
    m = TurntableManager(tmp_path)
    m.connect("simulated", options={"time_scale": FAST})
    cap = FakeCapture(m.driver)
    m.capture_provider = lambda: cap
    try:
        st = m.start_program({"interval_deg": 30, "frames_per_stop": 3, "direction": "ccw", "speed_s_per_rev": 25,
                              "rotations": [{"tilt_deg": 0}, {"tilt_deg": 15}], "sync_scan": True, "settle_s": 0})
        assert st["program"]["state"] in ("starting", "running")
        with pytest.raises(TurntableError, match="program is running"):
            m.rotate(10)
        assert wait_for(lambda: m.status()["program"]["state"] not in ("starting", "running"), 30)
        st = m.status()
        p = st["program"]
        assert p["state"] == "done", p
        assert p["completed_stops"] == p["total_stops"] == 24 and p["progress"] == 1.0
        assert p["turned_deg"] == 720 and p["tilts"] == [0, 15]
        assert st["angle_deg"] == pytest.approx(-720.0)            # ccw: negative
        assert st["tilt_deg"] == pytest.approx(0.0)                # levelled at the end
        assert p["capture"]["linked"] and p["capture"]["frames"] >= 24 * 3
        assert cap.frames_while_moving == 0, "capture must be paused while the platter moves"
        assert cap.events[0] == "start" and cap.state == "paused"
        assert cap.events.count("resume") == 23 and cap.events.count("pause") >= 24
        cmds = m.driver.commands
        assert cmds[0] == "+CT,TURNSPEED=25;" and cmds.count("+CT,TURNANGLE=-30;") == 24
        assert "+CR,TILTVALUE=15;" in cmds and cmds[-1] == "+CR,TILTVALUE=0;"
        assert any("left paused" in w for w in p["warnings"])
    finally:
        cap.close()
        m.shutdown()


def test_laser_line_scanner_scans_while_the_table_turns(tmp_path):
    """The user's first MetroY turntable scans stopped every 30 deg and recorded 3 frames there: a laser-line scanner
    then sees the same few lines at each stop and the scan came out as a few lines (756 points). With a scanner that
    sweeps, the program turns each rotation in one move while the scanner records, whatever mode was asked for."""
    from types import SimpleNamespace

    m = TurntableManager(tmp_path)
    m.connect("simulated", options={"time_scale": FAST})
    cap = FakeCapture(m.driver)
    cap.session = SimpleNamespace(driver=SimpleNamespace(capabilities=lambda: {"sweeps": True}))
    m.capture_provider = lambda: cap
    try:
        st = m.start_program({"interval_deg": 30, "frames_per_stop": 3, "speed_s_per_rev": 25,
                              "rotations": [{"tilt_deg": 0}, {"tilt_deg": 15}], "sync_scan": True, "settle_s": 0})
        assert st["program"]["mode"] == "continuous"
        assert wait_for(lambda: m.status()["program"]["state"] not in ("starting", "running"), 30)
        p = m.status()["program"]
        assert p["state"] == "done", p
        assert p["turned_deg"] == 720 and p["completed_stops"] == 2
        assert cap.frames_while_moving > 0, "a laser-line scanner must record while the platter turns"
        assert p["capture"]["frames"] >= cap.frames_while_moving > 0
        cmds = m.driver.commands
        assert cmds.count("+CT,TURNANGLE=360;") == 2 and "+CT,TURNANGLE=30;" not in cmds
        assert any("laser lines" in n for n in p.get("notes", []) + m.program.plan["notes"])
        # without sync (the scanner is not driven by the table) the asked-for stops are kept
        st = m.start_program({"interval_deg": 30, "sync_scan": False, "dwell_s": 0, "settle_s": 0})
        assert st["program"]["mode"] == "step"
        assert wait_for(lambda: m.status()["program"]["state"] not in ("starting", "running"), 30)
    finally:
        cap.close()
        m.shutdown()


def test_program_without_capture_and_stop(tmp_path):
    m = TurntableManager(tmp_path)
    m.capture_provider = lambda: None
    m.connect("simulated", options={"time_scale": 5.0})
    st = m.start_program({"interval_deg": 10, "frames_per_stop": 1, "sync_scan": True, "dwell_s": 0.01,
                          "settle_s": 0})
    assert wait_for(lambda: m.status()["program"]["completed_stops"] >= 2, 10)
    st = m.stop_program()
    assert wait_for(lambda: m.status()["program"]["state"] == "stopped", 5)
    p = m.status()["program"]
    assert 2 <= p["completed_stops"] < p["total_stops"]
    assert any("No capture session" in w for w in p["warnings"])
    assert not m.status()["moving"]
    st = m.rotate(5, wait=True)                                   # usable again
    assert st["move"]["state"] == "done"
    m.shutdown()


def test_program_error_is_reported(tmp_path):
    m = TurntableManager(tmp_path)
    m.capture_provider = lambda: None
    m.connect("simulated", options={"time_scale": FAST})

    def broken(degrees, timeout=None):
        raise TurntableError("The turntable refused +CT,TURNANGLE=30; (error 003)")

    m.driver.rotate = broken
    m.start_program({"interval_deg": 30, "sync_scan": False, "dwell_s": 0})
    assert wait_for(lambda: m.status()["program"]["state"] == "error", 5)
    assert "error 003" in m.status()["program"]["error"]
    m.shutdown()


# --------------------------------------------------------------------------- routes
def test_routes(tmp_path):
    from cloudclean.web.server import create_app

    with TestClient(create_app(tmp_path / "ws")) as client:
        r = client.get("/api/turntable/status")
        assert r.status_code == 200 and r.json()["connected"] is False
        r = client.get("/api/turntable/devices", params={"scan_seconds": 0})
        assert r.status_code == 200 and r.json()[0]["id"] == "simulated"
        r = client.post("/api/turntable/rotate", json={"degrees": 10})
        assert r.status_code == 400 and "No turntable is connected" in r.json()["detail"]
        r = client.post("/api/turntable/connect", json={"device": "simulated", "options": {"time_scale": FAST}})
        assert r.status_code == 200 and r.json()["kind"] == "simulated"
        r = client.post("/api/turntable/rotate", json={"degrees": 15, "speed_s_per_rev": 30, "wait": True})
        assert r.status_code == 200 and r.json()["angle_deg"] == pytest.approx(15.0)
        assert r.json()["move"]["state"] == "done"
        r = client.post("/api/turntable/tilt", json={"degrees": -10, "wait": True})
        assert r.status_code == 200 and r.json()["tilt_deg"] == pytest.approx(-10.0)
        r = client.post("/api/turntable/tilt", json={"degrees": 50})
        assert r.status_code == 400 and "between -30 and 30" in r.json()["detail"]
        r = client.post("/api/turntable/speed", json={"s_per_rev": 60})
        assert r.status_code == 200 and r.json()["speed_s_per_rev"] == 60
        r = client.post("/api/turntable/speed", json={"s_per_rev": 5})
        assert r.status_code == 400
        r = client.post("/api/turntable/program", json={"interval_deg": 30, "frames_per_stop": 1,
                                                          "sync_scan": False, "dwell_s": 0, "settle_s": 0})
        assert r.status_code == 200 and r.json()["program"]["stops_per_rotation"] == 12
        r = client.post("/api/turntable/program", json={"interval_deg": 2})
        assert r.status_code == 400
        assert wait_for(lambda: client.get("/api/turntable/status").json()["program"]["state"] == "done", 20)
        r = client.post("/api/turntable/program/stop")
        assert r.status_code == 200 and r.json()["program"]["state"] == "done"
        r = client.post("/api/turntable/stop")
        assert r.status_code == 200 and r.json()["moving"] is False
        r = client.post("/api/turntable/connect", json={"kind": "bogus"})
        assert r.status_code == 400
        r = client.post("/api/turntable/disconnect")
        assert r.status_code == 200 and r.json()["connected"] is False
        manager = get_turntable_manager(tmp_path / "ws")
        client.post("/api/turntable/connect", json={"device": "simulated", "options": {"time_scale": FAST}})
        assert manager.status()["connected"]
    # the app's lifespan shut the manager down (disconnected the simulated turntable)
    assert manager.closed and simtt.follow_pose() is None


# --------------------------------------------------------------------------- simulated scanner follows the platter
def _eye_angles(pose, center):
    eye = np.asarray(pose)[:3, 3] - center
    az = math.degrees(math.atan2(eye[1], eye[0]))
    el = math.degrees(math.atan2(eye[2], math.hypot(eye[0], eye[1])))
    return az, el


def test_simulated_scanner_follows_turntable():
    from cloudclean.capture.drivers import create_driver

    drv = create_driver("simulated")
    drv.connect({"realtime": False, "turntable": True, "simulate_mistakes": False})
    drv.start()
    center = drv._center
    free_pose, episode = drv.sensor_pose(1.0)
    tt = SimulatedTurntable(time_scale=FAST)
    tt.connect()
    try:
        p0 = drv.render(0.0).meta["true_pose"]
        az0, el0 = _eye_angles(p0, center)
        tt.rotate(90)
        p1 = drv.render(0.1).meta["true_pose"]
        az1, el1 = _eye_angles(p1, center)
        # the part turned 90 deg clockwise (seen from above): the scanner moved +90 deg around it in the part frame
        assert ((az1 - az0 + 180) % 360) - 180 == pytest.approx(90.0, abs=0.5)
        assert el1 == pytest.approx(el0, abs=0.5)
        tt.rotate(-90)
        tt.tilt(20)
        p2 = drv.render(0.2).meta["true_pose"]
        az2, el2 = _eye_angles(p2, center)
        assert el2 == pytest.approx(el0 - 20.0, abs=0.5)         # the tilt changes the view elevation by 20 deg
        frame = drv.render(0.3)
        assert len(frame.points) > 100
        tt.disconnect()
        p3 = drv.sensor_pose(1.0)[0]
        assert np.allclose(p3, free_pose)                        # back to its own motion when disconnected
    finally:
        tt.disconnect()
        drv.disconnect()


def test_end_to_end_capture_with_turntable_program(tmp_path):
    """Real CaptureManager + simulated scanner + simulated turntable: a synced program captures at every stop."""
    from cloudclean.web.server import create_app

    with TestClient(create_app(tmp_path / "ws")) as client:
        r = client.post("/api/capture/connect", json={"driver": "simulated",
                                                      "settings": {"turntable": True, "simulate_mistakes": False,
                                                                   "realtime": True, "colors": False}})
        assert r.status_code == 200, r.text
        r = client.post("/api/turntable/connect", json={"device": "simulated", "options": {"time_scale": 50}})
        assert r.status_code == 200
        r = client.post("/api/turntable/program", json={"interval_deg": 30, "frames_per_stop": 2, "sync_scan": True,
                                                          "settle_s": 0, "rotations": [{"tilt_deg": 0},
                                                                                       {"tilt_deg": 20}]})
        assert r.status_code == 200, r.text
        assert wait_for(lambda: client.get("/api/turntable/status").json()["program"]["state"] != "running", 90)
        prog_state = client.get("/api/turntable/status").json()["program"]
        assert prog_state["state"] == "done", prog_state
        cap = client.get("/api/capture/status").json()
        assert cap["state"] == "paused" and cap["frames"] >= 24 * 2 and cap["points"] > 1000
        assert prog_state["capture"]["frames"] >= 24 * 2
        r = client.post("/api/capture/stop")
        assert r.status_code == 200
