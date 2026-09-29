"""Grab N frames from a V4L2 node with mmap streaming and dump raw bytes. Pure ctypes, no v4l-utils / OpenCV."""
import ctypes
import fcntl
import mmap
import os
import select
import sys

VIDIOC_S_FMT = 0xc0d05605
VIDIOC_REQBUFS = 0xc0145608
VIDIOC_QUERYBUF = 0xc0585609
VIDIOC_QBUF = 0xc058560f
VIDIOC_DQBUF = 0xc0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613


def fourcc(s: str) -> int:
    return sum(ord(c) << (8 * i) for i, c in enumerate(s))


class Pix(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32), ("pixelformat", ctypes.c_uint32),
                ("field", ctypes.c_uint32), ("bytesperline", ctypes.c_uint32), ("sizeimage", ctypes.c_uint32),
                ("colorspace", ctypes.c_uint32), ("priv", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("ycbcr_enc", ctypes.c_uint32), ("quantization", ctypes.c_uint32), ("xfer_func", ctypes.c_uint32)]


class FmtUnion(ctypes.Union):
    # v4l2_format is 208 bytes on 64-bit, not 204: the union contains v4l2_window, which holds a pointer, so it
    # aligns to 8 and pads the leading `type` field out. The c_uint64 forces ctypes to the same alignment.
    _fields_ = [("pix", Pix), ("raw", ctypes.c_char * 200), ("_align", ctypes.c_uint64)]


class Format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", FmtUnion)]


class ReqBufs(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32), ("memory", ctypes.c_uint32),
                ("capabilities", ctypes.c_uint32), ("flags", ctypes.c_uint8), ("reserved", ctypes.c_uint8 * 3)]


class TimeVal(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class TimeCode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("frames", ctypes.c_uint8),
                ("seconds", ctypes.c_uint8), ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]


class BufUnion(ctypes.Union):
    _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong), ("fd", ctypes.c_int32)]


class Buffer(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32), ("bytesused", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("field", ctypes.c_uint32), ("timestamp", TimeVal),
                ("timecode", TimeCode), ("sequence", ctypes.c_uint32), ("memory", ctypes.c_uint32),
                ("m", BufUnion), ("length", ctypes.c_uint32), ("reserved2", ctypes.c_uint32),
                ("request_fd", ctypes.c_int32)]


def grab(dev: str, width: int, height: int, fmt: str, count: int, out_prefix: str) -> None:
    fd = os.open(dev, os.O_RDWR)
    f = Format(type=1)
    f.fmt.pix.width, f.fmt.pix.height = width, height
    f.fmt.pix.pixelformat, f.fmt.pix.field = fourcc(fmt), 1
    assert ctypes.sizeof(Format) == 208, f"v4l2_format must be 208 bytes, got {ctypes.sizeof(Format)}"
    fcntl.ioctl(fd, VIDIOC_S_FMT, f)
    print(f"requested {width}x{height} -> negotiated {f.fmt.pix.width}x{f.fmt.pix.height} "
          f"bpl={f.fmt.pix.bytesperline} sizeimage={f.fmt.pix.sizeimage}")

    req = ReqBufs(count=4, type=1, memory=1)
    fcntl.ioctl(fd, VIDIOC_REQBUFS, req)
    maps = []
    for i in range(req.count):
        b = Buffer(index=i, type=1, memory=1)
        fcntl.ioctl(fd, VIDIOC_QUERYBUF, b)
        maps.append(mmap.mmap(fd, b.length, offset=b.m.offset))
        fcntl.ioctl(fd, VIDIOC_QBUF, b)

    t = ctypes.c_uint32(1)
    fcntl.ioctl(fd, VIDIOC_STREAMON, t)
    got = 0
    while got < count:
        r, _, _ = select.select([fd], [], [], 5.0)
        if not r:
            print("timeout waiting for a frame")
            break
        b = Buffer(index=0, type=1, memory=1)
        fcntl.ioctl(fd, VIDIOC_DQBUF, b)
        data = maps[b.index][:b.bytesused]
        path = f"{out_prefix}_{got}.raw"
        with open(path, "wb") as fh:
            fh.write(data)
        print(f"frame {got}: index={b.index} bytesused={b.bytesused} seq={b.sequence} -> {path}")
        fcntl.ioctl(fd, VIDIOC_QBUF, b)
        got += 1
    fcntl.ioctl(fd, VIDIOC_STREAMOFF, t)
    for m in maps:
        m.close()
    os.close(fd)


if __name__ == "__main__":
    dev, w, h, fmt, n, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
    grab(dev, w, h, fmt, n, prefix)
