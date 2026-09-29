"""Talk to a Revopoint turntable without the web server - for the first hardware test (docs/turntable.md).

    python tools/turntable/turntable_cli.py scan [--seconds 8]
    python tools/turntable/turntable_cli.py info   [--device E4:8F:80:46:12:43]     # read-only queries
    python tools/turntable/turntable_cli.py rotate 10 --yes [--speed 60] [--device ...]
    python tools/turntable/turntable_cli.py tilt 5 --yes [--device ...]
    python tools/turntable/turntable_cli.py stop [--device ...]

Needs bleak (pip install "cloudclean[turntable]"), a Bluetooth adapter, and the turntable switched on and NOT
connected to Revo Metro (close Revo Metro first: the table accepts one connection at a time). Every command prints
the raw traffic so the result can be checked against docs/turntable.md. Add --simulated to try it without hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloudclean.capture.turntable import ble, protocol as P  # noqa: E402
from cloudclean.capture.turntable.base import TurntableError  # noqa: E402
from cloudclean.capture.turntable.simulated import SimulatedTurntable  # noqa: E402


def _print_traffic(drv) -> None:
    for item in getattr(drv, "traffic", []):
        print(f"  {item['dir']} {item['text']}")
    for cmd in getattr(drv, "commands", []):
        print(f"  > {cmd}")


def _driver(args):
    if args.simulated:
        drv = SimulatedTurntable(time_scale=1.0)
        drv.connect()
        return drv
    address, name, kind = args.device, None, "dual_axis"
    if address is None:
        print(f"Scanning {args.seconds:g} s for Revopoint turntables ...")
        found = ble.scan(args.seconds)
        if not found:
            raise TurntableError("No Revopoint turntable found. Is it on, near, and disconnected from Revo Metro?")
        address, name, kind = found[0]["address"], found[0]["name"], found[0]["kind"]
    print(f"Connecting to {name or 'turntable'} {address} ({kind}) ...")
    drv = ble.RevopointBleTurntable(address, name, kind)
    drv.connect()
    return drv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["scan", "info", "rotate", "tilt", "stop"])
    ap.add_argument("value", nargs="?", type=float, help="degrees for rotate (relative, + = clockwise) / tilt "
                                                         "(absolute)")
    ap.add_argument("--device", help="Bluetooth address (default: the first turntable found)")
    ap.add_argument("--seconds", type=float, default=8.0, help="scan time")
    ap.add_argument("--speed", type=float, help="seconds per revolution for rotate (25..90)")
    ap.add_argument("--yes", action="store_true", help="really move the turntable")
    ap.add_argument("--simulated", action="store_true", help="use the simulated turntable")
    args = ap.parse_args(argv)
    try:
        if args.action == "scan":
            found = ble.scan(args.seconds)
            print(json.dumps(found, indent=2) if found else "No Revopoint turntable is advertising (a table that is "
                                                            "connected to Revo Metro does not advertise).")
            return 0
        if args.action in ("rotate", "tilt"):
            if args.value is None:
                ap.error(f"{args.action} needs a value in degrees")
            if not args.yes:
                ap.error(f"{args.action} moves the turntable: add --yes (and keep hands and cables clear)")
        drv = _driver(args)
        try:
            print(json.dumps({"firmware": drv.firmware, **{k: v for k, v in drv.state().items()
                                                            if k != "last_messages"}}, indent=2, default=str))
            if args.action == "info":
                print("GATT:", json.dumps(getattr(drv, "gatt", []), indent=2))
                print("Reported ranges:", json.dumps(getattr(drv, "reported", {}), indent=2))
            elif args.action == "rotate":
                if args.speed is not None:
                    print(drv.set_speed(P.whole_degrees(args.speed, "speed")))
                print(drv.rotate(P.whole_degrees(args.value)))
            elif args.action == "tilt":
                print(drv.tilt(P.whole_degrees(args.value, "tilt")))
            elif args.action == "stop":
                print(drv.stop())
        finally:
            print("Traffic:")
            _print_traffic(drv)
            drv.disconnect()
        return 0
    except TurntableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
