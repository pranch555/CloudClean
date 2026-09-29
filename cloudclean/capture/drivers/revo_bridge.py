"""Revo Metro bridge driver: scans exported on the Windows/macOS scanning PC are pushed to CloudClean by
`cloudclean bridge` (POST /api/capture/bridge/upload) and arrive here through a queue."""
from __future__ import annotations

import queue
import time
from pathlib import Path

from .base import Frame, ScannerDriver, setting
from .folder import load_scan_frame


class RevoBridgeDriver(ScannerDriver):
    id = "revo_bridge"
    name = "Revopoint (Revo Metro bridge)"
    kind = "bridge"
    description = ("For the MetroY / MetroY Ultra: scan in Revo Metro on the Windows or macOS PC and run "
                   "`cloudclean bridge --server <this server> --watch <Revo Metro export folder>` there. Every "
                   "export is uploaded, aligned to the model and checked: further scans are only added when you "
                   "decide (or automatically when they add coverage and align reliably - setting fuse_mode).")

    def __init__(self):
        super().__init__()
        self.queue: queue.Queue = queue.Queue()
        self.received = 0

    def settings_schema(self) -> list[dict]:
        return [
            setting("point_distance", "Point distance", "number", 0.0, min=0.0, max=10.0, step=0.01, unit="mm",
                    help="The point distance chosen in Revo Metro. 0 = measure it from the first export."),
            setting("turntable", "Turntable", "boolean", False),
            setting("delete_after_load", "Delete uploaded copy after loading", "boolean", True),
        ]

    def capabilities(self) -> dict:
        return {"range_mm": None, "optimal_mm": None, "streaming": False, "provides_pose": False}

    def push(self, path: Path, name: str | None = None, source: str | None = None) -> None:
        """Queue an uploaded file (called from the upload route)."""
        self.queue.put((Path(path), name, source))
        self.received += 1

    @property
    def pending(self) -> int:
        return self.queue.qsize()

    def read(self, timeout: float = 0.1) -> Frame | None:
        if not self.running:
            time.sleep(min(timeout, 0.05))
            return None
        try:
            path, name, source = self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
        try:
            return load_scan_frame(path, name, source)
        finally:
            if self.settings.get("delete_after_load", True):
                Path(path).unlink(missing_ok=True)

    def disconnect(self) -> None:
        super().disconnect()
        if self.settings.get("delete_after_load", True):
            while True:
                try:
                    path, _, _ = self.queue.get_nowait()
                except queue.Empty:
                    break
                Path(path).unlink(missing_ok=True)
