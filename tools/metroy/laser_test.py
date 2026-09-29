"""Does register 0xb04 switch the laser? Stream continuously, toggle it, and watch the laser-lit pixel fraction."""
import ctypes
import fcntl
import mmap
import os
import select
import sys
import time

import numpy as np

import v4l_grab as v
from metroy_hid import MetroyHid


class Stream:
    """Continuous V4L2 mmap capture of the IR node in Revo Metro's mode (YUYV 1600x1200 = 8-bit 1600x2400)."""

    def __init__(self, dev="/dev/video0"):
        self.fd = os.open(dev, os.O_RDWR)
        f = v.Format(type=1)
        f.fmt.pix.width, f.fmt.pix.height = 1600, 1200
        f.fmt.pix.pixelformat, f.fmt.pix.field = v.fourcc("YUYV"), 1
        fcntl.ioctl(self.fd, v.VIDIOC_S_FMT, f)
        req = v.ReqBufs(count=4, type=1, memory=1)
        fcntl.ioctl(self.fd, v.VIDIOC_REQBUFS, req)
        self.maps = []
        for i in range(req.count):
            b = v.Buffer(index=i, type=1, memory=1)
            fcntl.ioctl(self.fd, v.VIDIOC_QUERYBUF, b)
            self.maps.append(mmap.mmap(self.fd, b.length, offset=b.m.offset))
            fcntl.ioctl(self.fd, v.VIDIOC_QBUF, b)
        fcntl.ioctl(self.fd, v.VIDIOC_STREAMON, ctypes.c_uint32(1))

    def frame(self, timeout=5.0) -> np.ndarray:
        if not select.select([self.fd], [], [], timeout)[0]:
            raise TimeoutError("no frame")
        b = v.Buffer(index=0, type=1, memory=1)
        fcntl.ioctl(self.fd, v.VIDIOC_DQBUF, b)
        img = np.frombuffer(self.maps[b.index][:b.bytesused], np.uint8).copy().reshape(2400, 1600)
        fcntl.ioctl(self.fd, v.VIDIOC_QBUF, b)
        return img

    def latest(self, n=3) -> np.ndarray:
        for _ in range(n - 1):   # drop frames queued before the change took effect
            self.frame()
        return self.frame()

    def close(self):
        fcntl.ioctl(self.fd, v.VIDIOC_STREAMOFF, ctypes.c_uint32(1))
        for m in self.maps:
            m.close()
        os.close(self.fd)


def lit(img: np.ndarray) -> str:
    return f"laser-lit {100 * (img > 200).mean():5.2f}%  mean {img.mean():5.1f}"


if __name__ == "__main__":
    s = Stream()
    try:
        print("baseline      ", lit(s.latest()))
        with MetroyHid() as hid:
            for value in (0, 1, 0, 1):
                hid.preisp(0xB04, value)
                time.sleep(0.6)
                print(f"0xb04 = {value}   ", lit(s.latest()))
                np.save(f"/tmp/b04_{value}.npy", s.latest(1))
    finally:
        s.close()
