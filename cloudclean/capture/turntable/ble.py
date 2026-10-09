"""Revopoint turntables over Bluetooth LE, with `bleak` (optional dependency: ``pip install "cloudclean[turntable]"``).

bleak is asyncio based; everything Bluetooth runs on one dedicated event-loop thread (`_BleLoop`) and the public
methods of this module are ordinary blocking calls that are safe from any thread (FastAPI's thread pool, the
program runner). Works on Windows (WinRT) and Linux (BlueZ, e.g. the DGX Spark's hci0) alike.

Protocol: see `protocol.py` and `docs/turntable.md`. ``validated`` stays False until CloudClean itself has driven
the user's table successfully (Revo Metro's logs prove the commands, not this implementation).
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque

from . import protocol as P
from .base import ASSUMED_TILT_RATE_DEG_S, TurntableDriver, TurntableError

ACK_TIMEOUT_S = 2.5          # Revo Metro waits 1 s; Linux/BlueZ round trips can be slower
QUERY_TIMEOUT_S = 2.5
FLUSH_DELAY_S = 0.08         # release an unterminated reply after this much silence
CONNECT_TIMEOUT_S = 25.0
FIND_TIMEOUT_S = 10.0
INSTALL_HINT = ('Bluetooth support is not installed on this computer. Install it with '
                'pip install "cloudclean[turntable]" (it adds the bleak package).')
BUSY_HINT = ("Make sure the turntable is switched on, within about 10 m, and not connected to Revo Metro or a "
             "phone: it accepts only one Bluetooth connection at a time.")

_bleak_state: tuple[bool, str] | None = None


def bleak_available() -> tuple[bool, str]:
    """(available, reason). Imports bleak lazily and caches the answer."""
    global _bleak_state
    if _bleak_state is None:
        try:
            import bleak  # noqa: F401
            _bleak_state = (True, "")
        except Exception as exc:  # ImportError, or a broken backend
            _bleak_state = (False, INSTALL_HINT if isinstance(exc, ImportError) else f"bleak failed to load: {exc}")
    return _bleak_state


def _require_bleak() -> None:
    ok, reason = bleak_available()
    if not ok:
        raise TurntableError(reason)


class _BleLoop:
    """One asyncio event loop in a daemon thread, shared by scanning and connections."""
    _instance: "_BleLoop | None" = None
    _guard = threading.Lock()

    @classmethod
    def get(cls) -> "_BleLoop":
        with cls._guard:
            if cls._instance is None or not cls._instance.thread.is_alive():
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="turntable-ble", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro, timeout: float):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return fut.result(timeout)
        except TimeoutError:
            fut.cancel()
            raise TurntableError(f"Bluetooth operation timed out after {timeout:.0f} s") from None


def _describe_error(exc: Exception) -> str:
    text = str(exc) or type(exc).__name__
    return text[:300]


# --------------------------------------------------------------------------- scanning
async def _ascan(seconds: float) -> list[dict]:
    from bleak import BleakScanner

    seen: dict[str, dict] = {}

    def on_adv(device, adv) -> None:
        name = getattr(adv, "local_name", None) or device.name
        entry = seen.setdefault(device.address, {"id": device.address, "address": device.address, "name": None,
                                                 "kind": None, "rssi": None})
        if name and not entry["name"]:
            entry["name"] = name
            entry["kind"] = P.classify_name(name)
        rssi = getattr(adv, "rssi", None)
        if rssi is not None and (entry["rssi"] is None or rssi > entry["rssi"]):
            entry["rssi"] = rssi

    async with BleakScanner(on_adv):
        await asyncio.sleep(seconds)
    return sorted((e for e in seen.values() if e["kind"] in ("dual_axis", "large")),
                  key=lambda e: -(e["rssi"] if e["rssi"] is not None else -999))


def scan(seconds: float = 4.0) -> list[dict]:
    """Revopoint turntables advertising nearby: [{id, address, name, kind, rssi}], strongest first.
    A connected turntable does not advertise, so it will not appear while Revo Metro holds it."""
    _require_bleak()
    try:
        return _BleLoop.get().call(_ascan(float(seconds)), timeout=float(seconds) + 15.0)
    except TurntableError:
        raise
    except Exception as exc:  # BleakError: no adapter, adapter off, permission denied, ...
        raise TurntableError(f"Bluetooth scan failed: {_describe_error(exc)}") from None


# --------------------------------------------------------------------------- the driver
class RevopointBleTurntable(TurntableDriver):
    """Dual-axis (REVO_DUAL_AXIS_TABLE) or large (REVO_TA500) Revopoint turntable over Bluetooth LE."""
    validated = False

    def __init__(self, address: str, name: str | None = None, kind: str = "dual_axis"):
        super().__init__()
        if kind not in ("dual_axis", "large"):
            raise TurntableError(f"Unsupported turntable kind '{kind}'")
        self.address = address
        self.id = address
        self.kind = kind
        # the dual-axis table was driven by CloudClean on the user's table on 2026-10-06 (fw SW 3.28: turn +-10 deg
        # with '+OK,TURNANGLE=' completion, tilt 5 -> 0); the large TA500 never has
        self.validated = kind == "dual_axis"
        self.name = name or (P.NAME_DUAL_AXIS if kind == "dual_axis" else P.NAME_LARGE)
        self._loop: _BleLoop | None = None
        self._client = None
        self._char = None
        self._assembler = P.MessageAssembler()
        self._flush_handle = None
        self._cond = threading.Condition()
        self._cmd_lock = threading.Lock()
        self._replies: deque[tuple[int, float, P.Reply]] = deque(maxlen=200)
        self._seq = 0
        self._stop_epoch = 0
        self.traffic: deque[dict] = deque(maxlen=200)   # last commands / replies, for diagnostics
        self.gatt: list[dict] = []
        self.reported: dict = {}                         # raw answers of the range queries
        self._angle: float | None = None
        self._tilt: float | None = None
        self._speed: int | None = None
        self._rotating = False
        self._tilting = False
        self._continuous = False
        self.warnings: list[str] = []

    # ----------------------------------------------------------------- notifications (loop thread)
    def _on_notify(self, _sender, data: bytearray) -> None:
        for msg in self._assembler.feed(bytes(data)):
            self._on_message(msg)
        if self._assembler.pending:
            if self._flush_handle is not None:
                self._flush_handle.cancel()
            self._flush_handle = asyncio.get_running_loop().call_later(FLUSH_DELAY_S, self._flush)

    def _flush(self) -> None:
        self._flush_handle = None
        for msg in self._assembler.flush():
            self._on_message(msg)

    def _on_message(self, text: str) -> None:
        reply = P.parse_reply(text)
        with self._cond:
            self._seq += 1
            self._replies.append((self._seq, time.monotonic(), reply))
            self.traffic.append({"t": round(time.time(), 3), "dir": "<", "text": text})
            if reply.kind == "ok_field" and reply.field == "TURNANGLE" and reply.number is not None:
                self._angle = reply.number
            self._cond.notify_all()

    def _on_disconnect(self, _client) -> None:
        with self._cond:
            if self.connected:
                self.error = "The Bluetooth connection to the turntable was lost"
            self.connected = False
            self._rotating = self._tilting = self._continuous = False
            self._cond.notify_all()

    # ----------------------------------------------------------------- request / reply
    def _wait_reply(self, pred, since: int, timeout: float, epoch: int | None = None) -> P.Reply | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for seq, _t, reply in self._replies:
                    if seq > since and pred(reply):
                        return reply
                if epoch is not None and epoch != self._stop_epoch:
                    return None
                if not self.connected:
                    raise TurntableError(self.error or "The turntable is not connected")
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cond.wait(min(left, 0.1))

    async def _awrite(self, data: bytes) -> None:
        client, char = self._client, self._char
        if client is None or not client.is_connected:
            raise TurntableError("The turntable is not connected")
        size = max(20, int(getattr(client, "mtu_size", 23) or 23) - 3)
        for i in range(0, len(data), size):
            await client.write_gatt_char(char, data[i:i + size], response=False)
            if i + size < len(data):
                await asyncio.sleep(0.01)      # Revo Metro sleeps 10 ms between chunks

    def _write(self, cmd: str) -> int:
        """Send one command; returns the reply sequence number just before it was sent."""
        if not self.connected or self._loop is None:
            raise TurntableError(self.error or "The turntable is not connected")
        with self._cond:
            seq = self._seq
            self.traffic.append({"t": round(time.time(), 3), "dir": ">", "text": cmd})
        try:
            self._loop.call(self._awrite(cmd.encode("ascii")), timeout=5.0)
        except TurntableError:
            raise
        except Exception as exc:
            raise TurntableError(f"Could not send {cmd.strip()} to the turntable: {_describe_error(exc)}") from None
        return seq

    def _command(self, cmd: str, timeout: float = ACK_TIMEOUT_S) -> P.Reply:
        """A command acknowledged with '+OK;' / '+OK,...' (or 'CR+OK;' on the large table)."""
        with self._cmd_lock:
            seq = self._write(cmd)
            reply = self._wait_reply(lambda r: r.kind in ("ok", "ok_field", "fail"), seq, timeout)
        if reply is None:
            raise TurntableError(f"The turntable did not answer {cmd.strip()}")
        if reply.kind == "fail":
            raise TurntableError(f"The turntable refused {cmd.strip()} (error {reply.code or '?'})")
        return reply

    def _query(self, cmd: str, timeout: float = QUERY_TIMEOUT_S) -> P.Reply | None:
        with self._cmd_lock:
            seq = self._write(cmd)
            return self._wait_reply(lambda r: r.kind in ("data", "fail"), seq, timeout)

    # ----------------------------------------------------------------- lifecycle
    async def _aconnect(self) -> None:
        from bleak import BleakClient, BleakScanner

        device = await BleakScanner.find_device_by_address(self.address, timeout=FIND_TIMEOUT_S)
        if device is None:
            raise TurntableError(f"The turntable {self.name} ({self.address}) was not found. {BUSY_HINT}")
        client = BleakClient(device, disconnected_callback=self._on_disconnect)
        try:
            await client.connect(timeout=CONNECT_TIMEOUT_S)
        except Exception as exc:
            raise TurntableError(f"Could not connect to {self.name} ({self.address}): {_describe_error(exc)}. "
                                 f"{BUSY_HINT}") from None
        try:
            char = None
            gatt = []
            for service in client.services:
                chars = []
                for c in service.characteristics:
                    chars.append({"uuid": c.uuid, "properties": list(c.properties)})
                    if char is None and P.is_data_characteristic(c.uuid):
                        char = c
                gatt.append({"uuid": service.uuid, "characteristics": chars})
            self.gatt = gatt
            if char is None:
                raise TurntableError(f"{self.name} has no 0xFFE1 characteristic; is this a Revopoint turntable?")
            await client.start_notify(char, self._on_notify)
        except (Exception, asyncio.CancelledError):  # CancelledError: the connect timed out in _BleLoop.call
            # a link left open stops the table advertising: the next try (or Revo Metro) would not find it
            try:
                await client.disconnect()
            except Exception:
                pass
            raise
        self._client, self._char = client, char

    def connect(self) -> None:
        _require_bleak()
        self._loop = _BleLoop.get()
        try:
            self._loop.call(self._aconnect(), timeout=FIND_TIMEOUT_S + CONNECT_TIMEOUT_S + 10.0)
        except TurntableError:
            raise
        except Exception as exc:
            raise TurntableError(f"Could not connect to {self.name}: {_describe_error(exc)}. {BUSY_HINT}") from None
        with self._cond:
            self.connected = True
            self.error = None
        self._identify()

    def _identify(self) -> None:
        """Read-only queries Revo Metro also sends after connecting (no motion, no settings)."""
        try:
            reply = self._query(P.QUERY_VERSION)
            self.firmware = P.parse_version(reply) if reply is not None and reply.kind == "data" else None
            if self.kind == "large":
                return
            for key, cmd in (("turn_position", P.QUERY_TURN_POSITION), ("tilt_position", P.QUERY_TILT_POSITION),
                             ("turn_speed_range", P.QUERY_TURN_SPEED_RANGE), ("tilt_range", P.QUERY_TILT_RANGE)):
                r = self._query(cmd)
                self.reported[key] = r.to_dict() if r is not None else None
                if r is not None and r.kind == "data" and r.number is not None:
                    if key == "turn_position":
                        self._angle = r.number
                    elif key == "tilt_position":
                        self._tilt = r.number
        except TurntableError as exc:
            self.warnings.append(f"Identification queries failed: {exc}")

    async def _adisconnect(self) -> None:
        client, char = self._client, self._char
        self._client = None
        if client is None:
            return
        try:
            if client.is_connected and char is not None:
                await client.stop_notify(char)
        except Exception:
            pass
        await client.disconnect()

    def disconnect(self) -> None:
        with self._cond:
            was = self.connected
            self.connected = False
            self._stop_epoch += 1
            self._cond.notify_all()
        if was and self._loop is not None:
            try:
                self._loop.call(self._adisconnect(), timeout=10.0)
            except Exception:
                pass

    # ----------------------------------------------------------------- state
    def state(self) -> dict:
        with self._cond:
            return {"angle_deg": self._angle, "tilt_deg": self._tilt,
                    "moving": self._rotating or self._tilting or self._continuous, "rotating": self._rotating,
                    "tilting": self._tilting, "continuous": self._continuous, "speed_s_per_rev": self._speed,
                    "error": self.error, "firmware": self.firmware, "address": self.address,
                    "reports_completion": P.reports_completion(self.firmware), "warnings": list(self.warnings),
                    "last_messages": list(self.traffic)[-12:]}

    # ----------------------------------------------------------------- motion
    def set_speed(self, s_per_rev: int) -> dict:
        s_per_rev = int(s_per_rev)
        cmd = P.dual_speed("turn", s_per_rev) if self.kind == "dual_axis" else P.large_speed(s_per_rev)
        self._command(cmd)
        self._speed = s_per_rev
        return {"speed_s_per_rev": s_per_rev, "command": cmd.strip()}

    def _expected_turn_s(self, degrees: int) -> float:
        return abs(degrees) / 360.0 * float(self._speed or 90) + 0.5

    def rotate(self, degrees: int, timeout: float | None = None) -> dict:
        degrees = int(degrees)
        if degrees == 0:
            return {"angle_deg": self._angle, "stopped": False, "duration_s": 0.0}
        expected = self._expected_turn_s(degrees)
        timeout = timeout or expected * 1.5 + 5.0
        cmd = P.dual_turn(degrees) if self.kind == "dual_axis" else P.large_turn(degrees)
        epoch = self._stop_epoch
        t0 = time.monotonic()
        with self._cond:
            if self._rotating or self._continuous:
                raise TurntableError("The turntable is already rotating")
            self._rotating = True
        try:
            with self._cmd_lock:
                seq = self._write(cmd)
            completion = P.reports_completion(self.firmware) if self.kind == "dual_axis" else False
            if completion is False:
                # Older firmware / large table: acknowledged only; wait for the nominal time [inferred]
                ack = self._wait_reply(lambda r: r.kind in ("ok", "ok_field", "fail"), seq, ACK_TIMEOUT_S, epoch)
                if ack is not None and ack.kind == "fail":
                    raise TurntableError(f"The turntable refused {cmd.strip()} (error {ack.code or '?'})")
                with self._cond:
                    self._cond.wait_for(lambda: epoch != self._stop_epoch or not self.connected,
                                        timeout=max(0.0, expected - (time.monotonic() - t0)))
                if self.kind == "dual_axis" and epoch == self._stop_epoch:
                    r = self._query(P.QUERY_TURN_POSITION)
                    if r is not None and r.kind == "data" and r.number is not None:
                        self._angle = r.number
            else:
                # Firmware >= 3.28 (or unknown): wait for '+OK,TURNANGLE=<position>' [verified in real logs]
                start_angle = self._angle
                done = self._wait_reply(lambda r: r.kind == "fail" or (r.kind == "ok_field" and r.field == "TURNANGLE"),
                                        seq, timeout, epoch)
                if done is not None and done.kind == "fail":
                    raise TurntableError(f"The turntable refused {cmd.strip()} (error {done.code or '?'})")
                if done is None and epoch == self._stop_epoch:
                    acked = self._wait_reply(lambda r: r.kind == "ok", seq, 0.0) is not None
                    r = self._query(P.QUERY_TURN_POSITION) if completion is None and acked else None
                    if (r is not None and r.kind == "data" and r.number is not None and start_angle is not None
                            and abs(r.number - start_angle - degrees) < 1.0):
                        self._angle = r.number       # unknown firmware without completion message: position agrees
                        self.warnings.append("This firmware does not announce the end of a turn; the position was "
                                             "checked with a query instead.")
                    else:
                        self.stop()
                        raise TurntableError(f"The turntable did not report the end of the {degrees} deg turn within "
                                             f"{timeout:.0f} s; it was stopped")
        finally:
            with self._cond:
                self._rotating = False
        stopped = epoch != self._stop_epoch
        return {"angle_deg": self._angle, "stopped": stopped, "duration_s": round(time.monotonic() - t0, 3),
                "command": cmd.strip()}

    def rotate_continuous(self, clockwise: bool) -> dict:
        cmd = (P.dual_turn_continuous(clockwise) if self.kind == "dual_axis"
               else P.large_turn_continuous(clockwise))
        self._command(cmd)
        with self._cond:
            self._continuous = True
        return {"continuous": True, "command": cmd.strip()}

    def tilt(self, degrees: int, timeout: float | None = None) -> dict:
        if self.kind != "dual_axis":
            raise TurntableError("The large turntable cannot tilt")
        degrees = int(degrees)
        start = self._tilt
        expected = (abs(degrees - start) if start is not None else 60.0) / ASSUMED_TILT_RATE_DEG_S + 0.3
        timeout = timeout or expected * 2.0 + 5.0
        cmd = P.dual_tilt(degrees)
        epoch = self._stop_epoch
        t0 = time.monotonic()
        with self._cond:
            if self._tilting:
                raise TurntableError("The turntable is already tilting")
            self._tilting = True
        confirmed = False
        try:
            with self._cmd_lock:
                seq = self._write(cmd)
                ack = self._wait_reply(lambda r: r.kind in ("ok", "ok_field", "fail"), seq, ACK_TIMEOUT_S, epoch)
            if ack is not None and ack.kind == "fail":
                raise TurntableError(f"The turntable refused {cmd.strip()} (error {ack.code or '?'})")
            # Completion is not announced as far as we know [unknown]: wait Revo Metro's estimate, then confirm
            # the position with +QR,TILTVALUE; until it matches or the timeout passes.
            done = self._wait_reply(lambda r: r.kind == "ok_field" and (r.field or "").startswith("TILT"), seq,
                                    expected, epoch)
            if done is not None:
                confirmed = True
            while epoch == self._stop_epoch and not confirmed and time.monotonic() - t0 < timeout:
                r = self._query(P.QUERY_TILT_POSITION)
                if r is None or r.kind != "data" or r.number is None:
                    break
                self._tilt = r.number
                if abs(r.number - degrees) <= 0.6:
                    confirmed = True
                    break
                with self._cond:
                    self._cond.wait_for(lambda: epoch != self._stop_epoch, timeout=0.3)
            if epoch == self._stop_epoch:
                self._tilt = float(degrees) if not confirmed or self._tilt is None else self._tilt
                if not confirmed:
                    note = f"Tilt to {degrees} deg sent; the end of the move could not be confirmed by a query."
                    if note not in self.warnings:
                        self.warnings.append(note)
        finally:
            with self._cond:
                self._tilting = False
        return {"tilt_deg": self._tilt, "stopped": epoch != self._stop_epoch, "confirmed": confirmed,
                "duration_s": round(time.monotonic() - t0, 3), "command": cmd.strip()}

    def stop(self) -> dict:
        """Emergency stop: written immediately (not queued behind a command waiting for its reply)."""
        with self._cond:
            was = self._rotating or self._tilting or self._continuous
            self._stop_epoch += 1
            self._continuous = False
            self._cond.notify_all()
            seq = self._seq
        if not self.connected:
            return {"stopped": was}
        cmds = [P.dual_stop("turn"), P.dual_stop("tilt")] if self.kind == "dual_axis" else [P.LARGE_STOP]
        for cmd in cmds:
            try:
                self._write(cmd)
            except TurntableError as exc:
                self.warnings.append(f"Stop failed: {exc}")
        # the reply to +CT,STOP; carries the position ('+OK,TURNANGLE=-105.01') [verified in real logs]
        if was and self.kind == "dual_axis":
            try:
                r = self._wait_reply(lambda r: r.kind == "ok_field" and r.field == "TURNANGLE", seq, 1.5)
            except TurntableError:
                r = None
            if r is not None and r.number is not None:
                self._angle = r.number
        return {"stopped": was, "angle_deg": self._angle}
