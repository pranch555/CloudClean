"""Reassemble UVC video frames from the isochronous packets of a USBPcap capture."""
import struct
import sys

exec(open("parse_usbpcap.py").read().split("def main")[0])

FRAME = 1600 * 1200 * 2


def frames(path, device, endpoint=0x82):
    buf, fid, count = bytearray(), None, 0
    for ts, pkt, orig in packets(path):
        hlen, irp, status, func, info, bus, dev, ep, xfer, dlen = struct.unpack_from("<HQIHBHHBBI", pkt, 0)
        if dev != device or xfer != 0 or ep != endpoint or not (info & 1):
            continue
        start, npk, errs = struct.unpack_from("<III", pkt, 27)
        descs = [struct.unpack_from("<III", pkt, 39 + 12 * i) for i in range(npk)]
        data = pkt[hlen:]
        for off, length, st in descs:
            if st or length < 2 or off + length > len(data):
                continue
            chunk = data[off:off + length]
            h = chunk[0]
            if h < 2 or h > length:
                continue
            flags = chunk[1]
            this_fid = flags & 1
            if fid is not None and this_fid != fid and buf:
                yield ts, bytes(buf)
                buf = bytearray()
            fid = this_fid
            buf += chunk[h:]
            if flags & 2:  # end of frame
                yield ts, bytes(buf)
                buf = bytearray()
                fid = None


if __name__ == "__main__":
    path, device = sys.argv[1], int(sys.argv[2])
    n = 0
    for ts, fr in frames(path, device):
        print(f"frame {n}: {len(fr)} bytes{'  <- complete' if len(fr) == FRAME else ''}")
        if len(fr) == FRAME:
            open(f"frame_{n}.raw", "wb").write(fr)
        n += 1
        if n >= 12:
            break
