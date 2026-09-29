"""Talk to a MetroY / MetroY Ultra over its HID command channel (Linux hidraw). See docs/metroy-protocol.md.

    python metroy_hid.py read /data/camparam/camparam.yaml [out_file]
"""
from __future__ import annotations

import glob
import os
import select
import struct
import sys
import time
from pathlib import Path

MAGIC = b"\x5a\x5a\x5a\x5a"
REPORT = 1024
FIRST_CHUNK = REPORT - 44          # payload bytes carried by the reply header
OK, NOT_READY = 0x01, 0x07


def find_hidraw(vid: str = "2207", pid: str = "110C") -> str:
    for node in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
        if f"{vid}:{pid}".upper() in os.path.realpath(os.path.join(node, "device")).upper():
            return "/dev/" + os.path.basename(node)
    raise FileNotFoundError("no MetroY HID interface found - is the scanner plugged in?")


class MetroyHid:
    def __init__(self, path: str | None = None):
        self.path = path or find_hidraw()
        self.fd = os.open(self.path, os.O_RDWR)

    def close(self) -> None:
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----------------------------------------------------------------- framing
    def send(self, command: int, group: int, args: dict[int, bytes] | None = None) -> None:
        buf = bytearray(REPORT)
        buf[0:4] = MAGIC
        buf[4], buf[5] = command, group
        for offset, data in (args or {}).items():
            buf[offset:offset + len(data)] = data
        # hidraw: the first byte is the report id, 0 for a device without numbered reports.
        os.write(self.fd, b"\x00" + bytes(buf))

    def recv(self, timeout: float = 2.0) -> bytes:
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            raise TimeoutError("scanner did not answer")
        return os.read(self.fd, REPORT)

    def drain(self) -> None:
        while select.select([self.fd], [], [], 0.05)[0]:
            os.read(self.fd, REPORT)

    # ----------------------------------------------------------------- commands
    def hello(self) -> int | None:
        """`06 05` - sent by Revo Metro first and between phases. On Windows it is acknowledged; over Linux
        hidraw the scanner often stays silent, and file reads work either way, so a missing reply is not an error.
        Returns the status byte, or None when there was no reply."""
        self.send(0x06, 0x05)
        try:
            return self.recv(timeout=0.5)[6]
        except TimeoutError:
            return None

    def read_file(self, remote: str, retries: int = 8) -> bytes:
        """`00 05` - read a file off the scanner. Status 07 means not present / not ready."""
        path = remote.encode() + b"\x00"
        for attempt in range(retries):
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
        raise FileNotFoundError(f"{remote}: scanner answered 'not found / not ready' {retries} times")


    def shell(self, command: str) -> None:
        """`01 01` - run a shell command on the scanner (it runs Linux). No reply is sent.

        Revo Metro configures acquisition this way, e.g. ``echo s 0xb04 1 > /dev/rk_preisp`` (see
        docs/metroy-protocol.md for the register list)."""
        data = command.encode()
        self.send(0x01, 0x01, {7: bytes([0x10]), 40: struct.pack("<I", len(data) + 1), 44: data + bytes(1)})

    def preisp(self, register: int, *values) -> None:
        """Write a register on the scanner's Rockchip pre-ISP, exactly as Revo Metro does."""
        args = " ".join(str(v) for v in values)
        self.shell(f"echo s 0x{register:x}{' ' + args if args else ''} > /dev/rk_preisp")

    def param(self, code: int, value: int) -> None:
        """`05 05` - set a numbered parameter (fields 256, code, 1, value). No reply is sent."""
        self.send(0x05, 0x05, {7: bytes([0x10]), 40: struct.pack("<IIII", 256, code, 1, value)})


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "reg":
        with MetroyHid() as dev:
            dev.preisp(int(sys.argv[2], 0), *sys.argv[3:])
    elif len(sys.argv) >= 3 and sys.argv[1] == "read":
        with MetroyHid() as dev:
            dev.drain()
            status = dev.hello()
            print(f"device {dev.path}, hello " + (f"status 0x{status:02x}" if status is not None else "(no reply)"),
                  file=sys.stderr)
            content = dev.read_file(sys.argv[2])
        if len(sys.argv) > 3:
            Path(sys.argv[3]).write_bytes(content)
            print(f"wrote {len(content)} bytes to {sys.argv[3]}", file=sys.stderr)
        else:
            sys.stdout.buffer.write(content)
    else:
        print(__doc__)
