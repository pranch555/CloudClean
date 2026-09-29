"""Enumerate V4L2 capture formats without v4l-utils (pure ctypes ioctls)."""
import ctypes
import fcntl
import os
import sys

VIDIOC_QUERYCAP = 0x80685600
VIDIOC_ENUM_FMT = 0xc0405602
VIDIOC_ENUM_FRAMESIZES = 0xc02c564a


class Cap(ctypes.Structure):
    _fields_ = [("driver", ctypes.c_char * 16), ("card", ctypes.c_char * 32), ("bus_info", ctypes.c_char * 32),
                ("version", ctypes.c_uint32), ("capabilities", ctypes.c_uint32), ("device_caps", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 3)]


class FmtDesc(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("description", ctypes.c_char * 32), ("pixelformat", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 4)]


class Discrete(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32)]


class FrameSize(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("pixel_format", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("discrete", Discrete), ("reserved", ctypes.c_uint32 * 2)]


def fourcc(v: int) -> str:
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


for dev in sys.argv[1:]:
    try:
        fd = os.open(dev, os.O_RDWR)
    except OSError as exc:
        print(f"{dev}: cannot open ({exc})")
        continue
    cap = Cap()
    try:
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap)
    except OSError as exc:
        print(f"{dev}: QUERYCAP failed ({exc})")
        os.close(fd)
        continue
    print(f"\n{dev}: card={cap.card.decode()!r} driver={cap.driver.decode()!r} caps=0x{cap.capabilities:08x}")
    for i in range(12):
        f = FmtDesc(index=i, type=1)
        try:
            fcntl.ioctl(fd, VIDIOC_ENUM_FMT, f)
        except OSError:
            break
        sizes = []
        for j in range(32):
            fs = FrameSize(index=j, pixel_format=f.pixelformat)
            try:
                fcntl.ioctl(fd, VIDIOC_ENUM_FRAMESIZES, fs)
            except OSError:
                break
            if fs.type == 1:
                sizes.append(f"{fs.discrete.width}x{fs.discrete.height}")
        joined = ", ".join(sizes) if sizes else "?"
        print(f"   fmt[{i}] {fourcc(f.pixelformat)} {f.description.decode()!r} sizes: {joined}")
    os.close(fd)
