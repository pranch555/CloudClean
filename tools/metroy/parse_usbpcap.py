"""Parse a USBPcap (DLT 249) capture and summarise the MetroY's control / HID traffic."""
import collections
import struct
import sys

UVC_REQ = {0x01: "SET_CUR", 0x81: "GET_CUR", 0x82: "GET_MIN", 0x83: "GET_MAX", 0x84: "GET_RES",
           0x85: "GET_LEN", 0x86: "GET_INFO", 0x87: "GET_DEF"}
STD_REQ = {0x00: "GET_STATUS", 0x01: "CLEAR_FEATURE", 0x03: "SET_FEATURE", 0x05: "SET_ADDRESS",
           0x06: "GET_DESCRIPTOR", 0x08: "GET_CONFIGURATION", 0x09: "SET_CONFIGURATION",
           0x0A: "GET_INTERFACE", 0x0B: "SET_INTERFACE"}
HID_REQ = {0x01: "GET_REPORT", 0x09: "SET_REPORT", 0x0A: "SET_IDLE", 0x0B: "SET_PROTOCOL"}
XFER = {0: "iso", 1: "intr", 2: "ctrl", 3: "bulk"}


def packets(path):
    with open(path, "rb") as f:
        gh = f.read(24)
        magic, _, _, _, _, _, linktype = struct.unpack("<IHHiIII", gh)
        assert magic == 0xA1B2C3D4 and linktype == 249, (hex(magic), linktype)
        while True:
            rh = f.read(16)
            if len(rh) < 16:
                return
            ts_s, ts_us, incl, orig = struct.unpack("<IIII", rh)
            yield ts_s + ts_us / 1e6, f.read(incl), orig


def main(path):
    t0 = None
    ctrl_rows, hid_rows = [], []
    counts = collections.Counter()
    devices = collections.Counter()
    pending = {}   # irp -> setup info (to attach the data from the completion)
    for ts, pkt, orig in packets(path):
        t0 = t0 if t0 is not None else ts
        hlen, irp, status, func, info, bus, dev, ep, xfer, dlen = struct.unpack_from("<HQIHBHHBBI", pkt, 0)
        to_host = bool(info & 1)
        counts[(dev, XFER.get(xfer, xfer), ep)] += 1
        devices[dev] += 1
        payload = pkt[hlen:]
        t = ts - t0
        if xfer == 2:
            stage = pkt[27]
            if stage == 0 and len(payload) >= 8:
                bm, breq, wval, widx, wlen = struct.unpack_from("<BBHHH", payload, 0)
                pending[irp] = (t, dev, bm, breq, wval, widx, wlen, payload[8:])
            elif stage in (1, 3) and irp in pending:
                t_s, d, bm, breq, wval, widx, wlen, out_data = pending.pop(irp)
                data = payload if to_host else (out_data or payload)
                ctrl_rows.append((t_s, d, bm, breq, wval, widx, wlen, bytes(data), status))
            elif stage == 2 and irp in pending:
                t_s, d, bm, breq, wval, widx, wlen, out_data = pending.pop(irp)
                ctrl_rows.append((t_s, d, bm, breq, wval, widx, wlen, bytes(out_data), status))
        elif xfer == 1 and payload:
            hid_rows.append((t, dev, ep, to_host, bytes(payload)))

    print(f"devices seen (address: packets): {dict(devices)}")
    print("packets by (device, type, endpoint):")
    for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])):
        print(f"   {k}: {v}")

    print(f"\n=== control transfers: {len(ctrl_rows)} ===")
    kinds = collections.Counter()
    for t, d, bm, breq, wval, widx, wlen, data, st in ctrl_rows:
        typ = (bm >> 5) & 3
        if typ == 0:
            name = STD_REQ.get(breq, hex(breq))
            key = f"std {name}"
        elif typ == 1:
            iface = widx & 0xFF
            if iface == 4 or breq in (0x09, 0x0A, 0x0B) and not (widx >> 8):
                key = f"class HID {HID_REQ.get(breq, hex(breq))} if={iface}"
            else:
                key = f"class UVC {UVC_REQ.get(breq, hex(breq))} unit={widx >> 8} if={iface} sel={wval >> 8}"
        else:
            key = f"vendor bReq=0x{breq:02x} wVal=0x{wval:04x} wIdx=0x{widx:04x}"
        kinds[key] += 1
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"   {v:5d}  {k}")

    print("\n=== every non-standard control transfer, in order ===")
    for t, d, bm, breq, wval, widx, wlen, data, st in ctrl_rows:
        typ = (bm >> 5) & 3
        if typ == 0:
            continue
        direction = "IN " if bm & 0x80 else "OUT"
        iface, unit, sel = widx & 0xFF, widx >> 8, wval >> 8
        rq = (HID_REQ if iface == 4 else UVC_REQ).get(breq, hex(breq)) if typ == 1 else f"vendor 0x{breq:02x}"
        print(f"{t:9.3f}s dev{d} {direction} {rq:9s} unit={unit:<3d} if={iface} sel={sel:<3d} "
              f"wValue=0x{wval:04x} len={wlen:<4d} st=0x{st:08x} data={data[:48].hex(' ')}"
              f"{' …' if len(data) > 48 else ''}")

    print(f"\n=== interrupt transfers: {len(hid_rows)} (first 40) ===")
    for t, d, ep, to_host, data in hid_rows[:40]:
        print(f"{t:9.3f}s dev{d} ep=0x{ep:02x} {'IN ' if to_host else 'OUT'} {data[:48].hex(' ')}")


if __name__ == "__main__":
    main(sys.argv[1])
