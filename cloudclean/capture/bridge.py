"""`cloudclean bridge`: runs on the Windows/macOS PC next to Revo Metro and pushes every new scan export to a
CloudClean server (e.g. a DGX Spark), where it appears in the live capture view with coverage guidance.

Only needs Python + httpx on the scanning PC (no Open3D / numpy):

    python -m cloudclean.capture.bridge --server http://spark:8765 --watch "C:/Users/me/Documents/RevoMetro/Export"

It polls the folder, waits until a file has stopped growing, uploads it to /api/capture/bridge/upload and
remembers uploaded files (path + size + mtime) in a small state file so a restart does not upload them again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Keep in sync with cloudclean.io.SUPPORTED_EXTS minus CAD formats (not imported: it would pull in Open3D).
SCAN_EXTS = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts", ".asc", ".txt", ".csv",
             ".obj", ".stl", ".off", ".gltf", ".glb"}
DEFAULT_STATE = Path.home() / ".cloudclean" / "bridge_state.json"


def _print(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class BridgeState:
    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}
        self.data.setdefault("uploaded", {})

    @staticmethod
    def key(server: str, path: Path) -> str:
        return f"{server.rstrip('/')}|{os.path.normcase(str(Path(path).resolve()))}"

    def done(self, server: str, path: Path, st) -> bool:
        entry = self.data["uploaded"].get(self.key(server, path))
        return bool(entry) and entry.get("size") == st.st_size and entry.get("mtime") == st.st_mtime

    def mark(self, server: str, path: Path, st) -> None:
        self.data["uploaded"][self.key(server, path)] = {"size": st.st_size, "mtime": st.st_mtime,
                                                         "uploaded": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)


def list_candidates(folder: Path, recursive: bool, extensions: set[str]):
    pattern = "**/*" if recursive else "*"
    for path in sorted(folder.glob(pattern)):
        if path.name.startswith(".") or path.suffix.lower() not in extensions:
            continue
        try:
            if path.is_file():
                yield path, path.stat()
        except OSError:
            continue


def upload(client, server: str, path: Path, import_asset: bool = True) -> dict:
    with open(path, "rb") as fh:
        res = client.post(f"{server.rstrip('/')}/api/capture/bridge/upload",
                          files={"files": (path.name, fh, "application/octet-stream")},
                          data={"source": str(path), "import_asset": "true" if import_asset else "false"})
    if res.status_code != 200:
        try:
            detail = res.json().get("detail", res.text)
        except ValueError:
            detail = res.text
        raise RuntimeError(f"server answered {res.status_code}: {detail}")
    return res.json()


def run(server: str, folder: Path, recursive: bool = False, settle: float = 3.0, once: bool = False,
        interval: float = 2.0, state_path: Path = DEFAULT_STATE, transport=None, include_existing: bool = True,
        import_asset: bool = True, key: str | None = None) -> int:
    import httpx

    folder = Path(folder)
    if not folder.is_dir():
        _print(f"Folder not found: {folder}")
        return 2
    state = BridgeState(state_path)
    extensions = set(SCAN_EXTS)
    headers = {"X-CloudClean-Key": key} if key else {}
    with httpx.Client(transport=transport, timeout=httpx.Timeout(600.0, connect=10.0), headers=headers) as client:
        try:
            res = client.get(f"{server.rstrip('/')}/api/capture/bridge/ping")
            if res.status_code == 401:
                _print("The server wants the bridge key: add --key <key> (CloudClean -> Settings -> Users)")
                return 3
            info = res.json()
            extensions = {e.lower() for e in info.get("extensions", [])} or extensions
            _print(f"Connected to {server}")
        except Exception as exc:
            _print(f"Cannot reach {server} yet ({exc}) - will keep trying")
        _print(f"Watching {folder}{' (with sub-folders)' if recursive else ''} for {', '.join(sorted(extensions))}")
        if not include_existing:
            for path, st in list_candidates(folder, recursive, extensions):
                if not state.done(server, path, st):
                    state.mark(server, path, st)
        seen: dict[Path, tuple[int, float, float]] = {}
        uploaded = failed = 0
        final_pass = False
        while True:
            now = time.monotonic()
            waiting = 0
            for path, st in list_candidates(folder, recursive, extensions):
                if state.done(server, path, st):
                    continue
                prev = seen.get(path)
                if prev is None or prev[0] != st.st_size or prev[1] != st.st_mtime:
                    seen[path] = (st.st_size, st.st_mtime, now)
                    if settle > 0:
                        waiting += 1
                        continue
                    prev = seen[path]
                if now - prev[2] < settle or st.st_size == 0:
                    waiting += 1
                    continue
                size_mb = st.st_size / 1e6
                _print(f"Uploading {path.name} ({size_mb:.1f} MB) ...")
                t0 = time.monotonic()
                try:
                    result = upload(client, server, path, import_asset)
                except Exception as exc:
                    failed += 1
                    _print(f"  FAILED: {exc} - will retry")
                    seen.pop(path, None)
                    continue
                state.mark(server, path, st)
                uploaded += 1
                messages = "; ".join(r.get("message", "") for r in result.get("results", []))
                _print(f"  done in {time.monotonic() - t0:.1f}s - {messages}")
            if once:
                if waiting and settle > 0 and not final_pass:
                    time.sleep(settle + 0.05)   # one more pass for files that were still settling, then exit
                    final_pass = True
                    continue
                break
            time.sleep(interval)
        _print(f"Uploaded {uploaded} file(s){f', {failed} failed' if failed else ''}")
        return 1 if failed else 0


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        prog="cloudclean bridge",
        description="Upload every new Revo Metro export from this PC to a CloudClean server for live capture.")
    parser.add_argument("--server", required=True, help="CloudClean URL, e.g. http://spark:8765")
    parser.add_argument("--watch", required=True, help="Folder Revo Metro exports scans to")
    parser.add_argument("--recursive", action="store_true", help="Also watch sub-folders")
    parser.add_argument("--settle", type=float, default=3.0,
                        help="Seconds a file must stay unchanged before it is uploaded (default 3)")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds (default 2)")
    parser.add_argument("--once", action="store_true", help="Upload what is there now and exit")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Only upload files that appear after the bridge starts")
    parser.add_argument("--no-import", action="store_true",
                        help="Only add exports to the live capture, do not also import each one as its own asset")
    parser.add_argument("--state", default=str(DEFAULT_STATE), help=f"State file (default {DEFAULT_STATE})")
    parser.add_argument("--key", default=os.environ.get("CLOUDCLEAN_KEY"),
                        help="the bridge key from CloudClean -> Settings -> Users (or set CLOUDCLEAN_KEY)")
    return parser


def main(argv=None, transport=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args.server, Path(args.watch), args.recursive, args.settle, args.once, args.interval,
                   Path(args.state), transport=transport, include_existing=not args.skip_existing,
                   import_asset=not args.no_import, key=args.key)
    except KeyboardInterrupt:
        _print("Stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
