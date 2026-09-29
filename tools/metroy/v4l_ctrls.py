"""List and optionally set V4L2 controls, including vendor/private ones. Pure ctypes (no v4l-utils)."""
import ctypes
import fcntl
import os
import sys

VIDIOC_QUERYCTRL = 0xc0445624
VIDIOC_G_CTRL = 0xc008561b
VIDIOC_S_CTRL = 0xc008561c
NEXT_CTRL = 0x80000000

TYPES = {1: "int", 2: "bool", 3: "menu", 4: "button", 5: "int64", 6: "ctrl-class",
         7: "string", 8: "bitmask", 9: "int-menu"}


class QueryCtrl(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_uint32), ("name", ctypes.c_char * 32),
                ("minimum", ctypes.c_int32), ("maximum", ctypes.c_int32), ("step", ctypes.c_int32),
                ("default_value", ctypes.c_int32), ("flags", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 2)]


class Control(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("value", ctypes.c_int32)]


def listing(dev: str) -> None:
    fd = os.open(dev, os.O_RDWR)
    print(f"{dev}:")
    qid, seen = 0, 0
    while True:
        q = QueryCtrl(id=qid | NEXT_CTRL)
        try:
            fcntl.ioctl(fd, VIDIOC_QUERYCTRL, q)
        except OSError:
            break
        qid = q.id
        seen += 1
        cur = Control(id=q.id)
        try:
            fcntl.ioctl(fd, VIDIOC_G_CTRL, cur)
            value = cur.value
        except OSError:
            value = "?"
        kind = TYPES.get(q.type, q.type)
        vendor = " [VENDOR]" if q.id >= 0x08000000 else ""
        print(f"  0x{q.id:08x} {q.name.decode(errors='replace'):34s} {kind:6s} "
              f"min={q.minimum:<8} max={q.maximum:<10} step={q.step:<6} default={q.default_value:<8} "
              f"now={value}{vendor}")
    if not seen:
        print("  (no controls)")
    os.close(fd)


def put(dev: str, cid: int, value: int) -> None:
    fd = os.open(dev, os.O_RDWR)
    c = Control(id=cid, value=value)
    fcntl.ioctl(fd, VIDIOC_S_CTRL, c)
    back = Control(id=cid)
    fcntl.ioctl(fd, VIDIOC_G_CTRL, back)
    print(f"set 0x{cid:08x} = {value} -> reads back {back.value}")
    os.close(fd)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        put(sys.argv[1], int(sys.argv[2], 0), int(sys.argv[3]))
    else:
        for d in sys.argv[1:]:
            listing(d)
