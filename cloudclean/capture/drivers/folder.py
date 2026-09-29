"""Watch a local or network folder: every new scan file that stops growing becomes one frame chunk.

The chunk is delivered in its file coordinates (`coordinates="scan"`); the capture session registers it to
the model built so far (global registration + ICP, like `register.align_pair`) unless alignment is off, and
assesses it: whether it is fused or kept aside until the user decides follows the session's `fuse_mode`."""
from __future__ import annotations

import os
import time
from pathlib import Path

from .base import Frame, ScannerDriver, setting

SETTLE_DEFAULT = 3.0


def scan_extensions() -> set[str]:
    from ...io import SUPPORTED_EXTS

    return set(SUPPORTED_EXTS)


def load_scan_frame(path: Path, name: str | None = None, source: str | None = None) -> Frame:
    """Load a scan file (cloud or mesh) as a `scan` frame chunk."""
    import numpy as np

    from ...io import is_cloud, load, to_cloud

    try:
        geom = load(path)
    except Exception as exc:  # a broken export must not stop the capture
        return Frame(points=np.empty((0, 3)), timestamp=time.time(), coordinates="scan",
                     meta={"name": name or Path(path).stem, "file": source or str(path), "error": str(exc)})
    cloud = geom if is_cloud(geom) else to_cloud(geom)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    normals = np.asarray(cloud.normals) if cloud.has_normals() else None
    return Frame(points=np.asarray(cloud.points), colors=colors, normals=normals, timestamp=time.time(),
                 meta={"name": name or Path(path).stem, "file": source or str(path),
                       "kind": "pointcloud" if is_cloud(geom) else "mesh"}, coordinates="scan")


class FolderWatcher:
    """Polling watcher (no extra dependency): reports files whose size and mtime were stable for `settle` s."""

    def __init__(self, folder: Path, extensions: set[str], recursive: bool = False, settle: float = SETTLE_DEFAULT,
                 include_existing: bool = False):
        self.folder = Path(folder)
        self.extensions = {e.lower() for e in extensions}
        self.recursive = recursive
        self.settle = settle
        self.pending: dict[Path, tuple[int, float, float]] = {}   # path -> (size, mtime, first seen unchanged)
        self.done: set[tuple[str, int, float]] = set()
        if not include_existing:
            for path, st in self._listing():
                self.done.add((str(path), st.st_size, st.st_mtime))

    def _listing(self):
        pattern = "**/*" if self.recursive else "*"
        for path in self.folder.glob(pattern):
            if path.suffix.lower() not in self.extensions or path.name.startswith("."):
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if path.is_file():
                yield path, st

    def poll(self) -> list[Path]:
        now = time.monotonic()
        ready = []
        for path, st in self._listing():
            ident = (str(path), st.st_size, st.st_mtime)
            if ident in self.done:
                continue
            prev = self.pending.get(path)
            if prev is None or prev[0] != st.st_size or prev[1] != st.st_mtime:
                self.pending[path] = (st.st_size, st.st_mtime, now)
                if self.settle > 0:
                    continue
                prev = self.pending[path]
            if now - prev[2] >= self.settle and st.st_size > 0:
                ready.append(path)
                self.done.add(ident)
                self.pending.pop(path, None)
        return sorted(ready, key=lambda p: (p.stat().st_mtime if p.exists() else 0, str(p)))


class FolderDriver(ScannerDriver):
    id = "folder"
    name = "Watch folder"
    kind = "folder"
    description = ("Watches a folder (local or network share) where scans are exported. Each new file is added "
                   "to the capture, aligned to what was scanned before and checked before it is fused.")

    def __init__(self):
        super().__init__()
        self._watcher: FolderWatcher | None = None
        self._queue: list[Path] = []
        self._last_poll = 0.0

    def settings_schema(self) -> list[dict]:
        return [
            setting("folder", "Folder", "text", "", help="Folder on the CloudClean computer (or a mounted share)."),
            setting("recursive", "Include sub-folders", "boolean", False),
            setting("settle_seconds", "Wait until a file is unchanged for", "number", SETTLE_DEFAULT, min=0.0,
                    max=120.0, step=0.5, unit="s"),
            setting("include_existing", "Also add files already in the folder", "boolean", False),
            setting("point_distance", "Point distance", "number", 0.0, min=0.0, max=10.0, step=0.01, unit="mm",
                    help="0 = measure from the first scan."),
        ]

    def capabilities(self) -> dict:
        return {"range_mm": None, "optimal_mm": None, "streaming": False, "provides_pose": False}

    def _connect(self) -> None:
        folder = Path(os.path.expanduser(self.settings["folder"].strip().strip('"')))
        if not self.settings["folder"].strip():
            raise ValueError("Choose the folder to watch")
        if not folder.is_dir():
            raise ValueError(f"Folder not found: {folder}")
        self.folder = folder

    def _start(self) -> None:
        if self._watcher is None:
            self._watcher = FolderWatcher(self.folder, scan_extensions(), self.settings["recursive"],
                                          float(self.settings["settle_seconds"]), self.settings["include_existing"])

    def read(self, timeout: float = 0.1) -> Frame | None:
        if not self.running or self._watcher is None:
            time.sleep(min(timeout, 0.05))
            return None
        if not self._queue:
            now = time.monotonic()
            if now - self._last_poll < 1.0:
                time.sleep(min(timeout, 1.0 - (now - self._last_poll)))
                return None
            self._last_poll = now
            self._queue.extend(self._watcher.poll())
            if not self._queue:
                return None
        path = self._queue.pop(0)
        return load_scan_frame(path)
