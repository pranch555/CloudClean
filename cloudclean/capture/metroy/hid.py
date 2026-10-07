"""The MetroY HID command channel (Linux hidraw). See docs/metroy-protocol.md.

Every report is 1024 bytes and starts with the magic 5a 5a 5a 5a, then a command and a group byte. The scanner is a
small Rockchip Linux computer: `00 05` reads a file off it and `01 01` runs a shell command on it, which is how Revo
Metro writes the acquisition registers of the pre-ISP (``echo s <reg> <values> > /dev/rk_preisp``).
"""
from __future__ import annotations

import glob
import os
import select
import struct
import threading
import time

from .exposure import PRESETS

MAGIC = b"\x5a\x5a\x5a\x5a"
REPORT = 1024
FIRST_CHUNK = REPORT - 44          # payload bytes carried by the reply header
OK, NOT_READY = 0x01, 0x07

CALIBRATION = "/data/camparam/camparam.yaml"
LASER_CALIBRATION = "/data/camparam/metroExtra.bin"
EXPOSURE_PRESETS = "/data/brightnessGainMap.json"

REG_LASER = 0xB04                  # 1 = laser projector on, 0 = off (verified)
REG_FILL_LIGHT = 0xB07             # IR fill light brightness 0-255; lights retro-reflective markers (verified)
REG_FRAME_TIME, REG_EXPOSURE, REG_GAIN = 0x910, 0x911, 0x903   # us, us, 0x10 per 1x gain (all verified)
REG_LASER_POWER, REG_LASER_TIMES = 0x48D, 0xB08                 # laser brightness 0-255, laser exposure times
REG_PATTERN = 0x3001               # laser line pattern; MetroY Ultra: 7 cross, 9 single, 11 parallel
PATTERN_CROSS, PATTERN_SINGLE, PATTERN_PARALLEL = 7, 9, 11

# Revo Metro's cross-line presets for the MetroY Ultra (model 0L8, docs/revo-metro-internals.md 2.2): exposure in
# microseconds, analogue gain, fill-light brightness. At these short exposures a diffuse surface stays dark under the
# fill light while the laser lines and the retro-reflective markers shine. The full presets (with the laser level that
# sets the laser pulse 0xb08) are exposure.PRESETS; this is the older three-value view of them.
SURFACE_PRESETS = {k: {"exposure_us": p.exposure_us, "gain": p.gain, "fill_light": p.marker_light}
                   for k, p in PRESETS.items()}
FRAME_TIME_US = 8000
MIN_GAP_S = 0.15                   # between two shell commands: chained writes made the scanner drop off the bus


def find_hidraw(vid: str = "2207", pid: str = "110C") -> str:
    for node in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
        if f"{vid}:{pid}".upper() in os.path.realpath(os.path.join(node, "device")).upper():
            return "/dev/" + os.path.basename(node)
    raise FileNotFoundError("no MetroY HID interface found - is the scanner plugged in?")


class MetroyHid:
    def __init__(self, path: str | None = None):
        self._lock = threading.RLock()      # register writes come from the camera writer and the lifecycle
        self._last_shell = 0.0
        self.min_gap_s = MIN_GAP_S
        self.path = path or find_hidraw()
        try:
            self.fd = os.open(self.path, os.O_RDWR)
        except PermissionError:
            raise PermissionError(
                f"no permission to open {self.path}. Run this once in the CloudClean folder: "
                "sudo deploy/install-scanner-access.sh (it lets CloudClean use the scanner even when nobody is "
                "logged in at the screen; no replugging needed).") from None

    def close(self) -> None:
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- framing
    def send(self, command: int, group: int, args: dict[int, bytes] | None = None) -> None:
        buf = bytearray(REPORT)
        buf[0:4] = MAGIC
        buf[4], buf[5] = command, group
        for offset, data in (args or {}).items():
            buf[offset:offset + len(data)] = data
        # hidraw: the first byte written is the report id, 0 for a device without numbered reports
        with self._lock:
            os.write(self.fd, b"\x00" + bytes(buf))

    def recv(self, timeout: float = 2.0) -> bytes:
        if not select.select([self.fd], [], [], timeout)[0]:
            raise TimeoutError("scanner did not answer")
        return os.read(self.fd, REPORT)

    def drain(self) -> None:
        while select.select([self.fd], [], [], 0.05)[0]:
            os.read(self.fd, REPORT)

    # -- commands
    def read_file(self, remote: str, retries: int = 8) -> bytes:
        """`00 05` - read a file off the scanner. Status 07 means not present / not ready."""
        path = remote.encode() + b"\x00"
        for _ in range(retries):
            self.send(0x00, 0x05, {41: b"\x01", 44: path})
            head = self.recv()
            if head[:4] != MAGIC or head[4:6] != b"\x00\x05":
                raise IOError(f"unexpected reply header {head[:8].hex(' ')}")
            status = head[6]
            if status == NOT_READY:
                time.sleep(0.1)
                continue
            if status != OK:
                raise IOError(f"status 0x{status:02x} for {remote}")
            length = struct.unpack_from("<I", head, 40)[0]
            data = bytearray(head[44:44 + min(length, FIRST_CHUNK)])
            while len(data) < length:
                data += self.recv()[:length - len(data)]
            return bytes(data)
        raise FileNotFoundError(f"{remote}: the scanner answered 'not found / not ready' {retries} times")

    def shell(self, command: str) -> None:
        """`01 01` - run a shell command on the scanner. No reply is sent. Never two within `min_gap_s`, from any
        thread: the scanner drops off the bus when register writes come too close together."""
        data = command.encode()
        with self._lock:
            wait = self._last_shell + self.min_gap_s - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            try:
                self.send(0x01, 0x01, {7: bytes([0x10]), 40: struct.pack("<I", len(data) + 1), 44: data + bytes(1)})
            finally:
                self._last_shell = time.perf_counter()

    def preisp(self, register: int, *values) -> None:
        """Write a register of the scanner's Rockchip pre-ISP, exactly as Revo Metro does.

        Never write exposure 0x910 without 0x911: alone it stalled the stream until the watchdog reset the device."""
        args = " ".join(str(v) for v in values)
        self.shell(f"echo s 0x{register:x}{' ' + args if args else ''} > /dev/rk_preisp")

    def laser(self, on: bool) -> None:
        self.preisp(REG_LASER, 1 if on else 0)

    def param(self, code: int, value: int) -> None:
        """`05 05` - set a numbered parameter (fields 256, code, 1, value), as Revo Metro does at start-up."""
        self.send(0x05, 0x05, {7: bytes([0x10]), 40: struct.pack("<IIII", 256, code, 1, value)})

    def startup(self, pattern: int = PATTERN_CROSS) -> None:
        """Revo Metro's acquisition start-up, replayed register by register (docs/metroy-protocol.md). A scanner that
        has reset itself comes back with boot defaults - dim laser among them - so capture never relies on whatever
        state the scanner happens to be in. Every write is its own command, with a pause."""
        steps = [
            ("reg", 0xA00, (1,)), ("reg", 0xB07, (1,)), ("param", 103, 1), ("param", 102, 0), ("reg", 0x103, (0,)),
            ("reg", 0x3001, (pattern,)), ("reg", 0x707, (200, 600)), ("reg", REG_GAIN, ("0x10",)),
            ("pair", 7000, 5000), ("reg", 0x701, (1,)), ("reg", 0xB01, (1,)), ("pair", 8000, 200),
            ("reg", REG_GAIN, ("0x10",)), ("reg", REG_FILL_LIGHT, (30,)), ("reg", REG_LASER_POWER, (255,)),
            ("reg", REG_LASER_TIMES, (217,)), ("reg", 0x707, (230, 630)),
        ]
        for kind, a, b in steps:
            if kind == "reg":
                self.preisp(a, *b)
            elif kind == "param":
                self.param(a, b)
            else:
                self.shell(f"echo s 0x{REG_FRAME_TIME:x} {a} > /dev/rk_preisp; "
                           f"echo s 0x{REG_EXPOSURE:x} {b} > /dev/rk_preisp")
            time.sleep(0.15)

    def acquisition(self, exposure_us: int, gain: int, fill_light: int, frame_time_us: int = FRAME_TIME_US) -> None:
        """Exposure, gain and fill light, in the order and grouping that the scanner tolerates (all measured on ours):

        * frame time 0x910 and exposure 0x911 go together in ONE command - 0x910 alone stalled the stream until the
          watchdog reset the scanner;
        * every other register goes in a command of its own, with a pause - four writes chained in one command made
          the scanner drop off the bus and re-enumerate.
        """
        self.shell(f"echo s 0x{REG_FRAME_TIME:x} {int(frame_time_us)} > /dev/rk_preisp; "
                   f"echo s 0x{REG_EXPOSURE:x} {int(exposure_us)} > /dev/rk_preisp")
        time.sleep(0.15)
        self.preisp(REG_GAIN, f"0x{int(round(16 * gain)):02x}")
        time.sleep(0.15)
        self.preisp(REG_FILL_LIGHT, int(fill_light))
        time.sleep(0.15)
