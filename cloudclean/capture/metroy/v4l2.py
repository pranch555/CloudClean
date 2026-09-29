"""Continuous V4L2 capture of the MetroY stereo IR node, in the mode Revo Metro uses. Pure ctypes; Linux only.

UVC advertises it as YUYV 1600x1200 (format 1, frame 1), but the buffer is 8-bit grey 1600 wide by 2400 tall: the
left camera on top, the right below. 800x2400 has the same byte count, so a wrong guess still yields a plausible
picture - check the stereo disparity, never just the size. The Y16 modes and UVC control writes make the scanner
time out and reset itself; this mode streams stably with no handshake at all, at about 28 frames per second.
"""
from __future__ import annotations

import ctypes
import fcntl
import glob
import mmap
import os
import select
from dataclasses import dataclass

import numpy as np

VIDIOC_S_FMT = 0xC0D05605
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
BUF_TYPE_VIDEO_CAPTURE, MEMORY_MMAP = 1, 1

WIDTH, HEIGHT = 1600, 1200         # per camera; the buffer holds two of them


def fourcc(s: str) -> int:
    return sum(ord(c) << (8 * i) for i, c in enumerate(s))


class _Pix(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32), ("pixelformat", ctypes.c_uint32),
                ("field", ctypes.c_uint32), ("bytesperline", ctypes.c_uint32), ("sizeimage", ctypes.c_uint32),
                ("colorspace", ctypes.c_uint32), ("priv", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("ycbcr_enc", ctypes.c_uint32), ("quantization", ctypes.c_uint32), ("xfer_func", ctypes.c_uint32)]


class _FmtUnion(ctypes.Union):
    # v4l2_format is 208 bytes on 64-bit: the union holds a pointer, so it aligns to 8 and pads `type` out
    _fields_ = [("pix", _Pix), ("raw", ctypes.c_char * 200), ("_align", ctypes.c_uint64)]


class _Format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", _FmtUnion)]


class _ReqBufs(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32), ("memory", ctypes.c_uint32),
                ("capabilities", ctypes.c_uint32), ("flags", ctypes.c_uint8), ("reserved", ctypes.c_uint8 * 3)]


class _TimeVal(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _TimeCode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("frames", ctypes.c_uint8),
                ("seconds", ctypes.c_uint8), ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]


class _BufUnion(ctypes.Union):
    _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong), ("fd", ctypes.c_int32)]


class _Buffer(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32), ("bytesused", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("field", ctypes.c_uint32), ("timestamp", _TimeVal),
                ("timecode", _TimeCode), ("sequence", ctypes.c_uint32), ("memory", ctypes.c_uint32),
                ("m", _BufUnion), ("length", ctypes.c_uint32), ("reserved2", ctypes.c_uint32),
                ("request_fd", ctypes.c_int32)]


@dataclass
class RawFrame:
    image: np.ndarray       # (2400, 1600) uint8, left camera on top
    timestamp: float        # seconds, the kernel's capture time (CLOCK_MONOTONIC for uvcvideo)
    sequence: int           # driver frame counter: gaps mean dropped frames


def find_ir_node(vid: str = "2207", pid: str = "110c") -> str:
    """The stereo IR capture node: the scanner's video interface 0 (the RGB camera is interface 2)."""
    for node in sorted(glob.glob("/sys/class/video4linux/video*")):
        dev = os.path.realpath(os.path.join(node, "device"))
        parent = os.path.dirname(dev)
        try:
            ids = (open(os.path.join(parent, "idVendor")).read().strip().lower(),
                   open(os.path.join(parent, "idProduct")).read().strip().lower())
            index = int(open(os.path.join(node, "index")).read().strip())
        except OSError:
            continue
        if ids == (vid, pid) and dev.endswith(":1.0") and index == 0:
            return "/dev/" + os.path.basename(node)
    raise FileNotFoundError("no MetroY IR video node found - is the scanner plugged in?")


class Stream:
    """mmap streaming with a small ring of kernel buffers. Not thread-safe: use from one thread."""

    def __init__(self, dev: str | None = None, buffers: int = 4):
        self.dev = dev or find_ir_node()
        self.fd = os.open(self.dev, os.O_RDWR)
        self.maps: list[mmap.mmap] = []
        try:
            f = _Format(type=BUF_TYPE_VIDEO_CAPTURE)
            f.fmt.pix.width, f.fmt.pix.height = WIDTH, HEIGHT
            f.fmt.pix.pixelformat, f.fmt.pix.field = fourcc("YUYV"), 1
            fcntl.ioctl(self.fd, VIDIOC_S_FMT, f)
            if (f.fmt.pix.width, f.fmt.pix.height) != (WIDTH, HEIGHT):
                raise IOError(f"{self.dev} negotiated {f.fmt.pix.width}x{f.fmt.pix.height}, expected 1600x1200")
            req = _ReqBufs(count=buffers, type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP)
            fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
            for i in range(req.count):
                b = _Buffer(index=i, type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP)
                fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, b)
                self.maps.append(mmap.mmap(self.fd, b.length, offset=b.m.offset))
                fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
            fcntl.ioctl(self.fd, VIDIOC_STREAMON, ctypes.c_uint32(BUF_TYPE_VIDEO_CAPTURE))
        except Exception:
            self._release()
            raise

    def read(self, timeout: float = 1.0) -> RawFrame | None:
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        b = _Buffer(index=0, type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_DQBUF, b)
        # read everything before re-queueing: QBUF writes into the same struct and clears timestamp and sequence
        stamp, sequence, used = b.timestamp.tv_sec + b.timestamp.tv_usec * 1e-6, int(b.sequence), b.bytesused
        try:
            if used != 2 * WIDTH * HEIGHT:
                return None            # a short buffer after a USB hiccup: skip it rather than mis-shape it
            img = np.frombuffer(self.maps[b.index], np.uint8, count=used).copy().reshape(2 * HEIGHT, WIDTH)
        finally:
            fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
        return RawFrame(img, stamp, sequence)

    def close(self) -> None:
        try:
            fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, ctypes.c_uint32(BUF_TYPE_VIDEO_CAPTURE))
        except OSError:
            pass                        # the device may have re-enumerated underneath us
        self._release()

    def _release(self) -> None:
        for m in self.maps:
            m.close()
        self.maps = []
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
