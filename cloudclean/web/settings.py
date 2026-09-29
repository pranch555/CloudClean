"""Persistent app settings stored in <workspace>/settings.json, one section per feature."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_lock = threading.Lock()


class Settings:
    def __init__(self, workspace_root):
        self.path = Path(workspace_root) / "settings.json"

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def get(self, section: str, defaults: dict | None = None) -> dict:
        with _lock:
            return {**(defaults or {}), **self._read().get(section, {})}

    def update(self, section: str, values: dict) -> dict:
        with _lock:
            data = self._read()
            data[section] = {**data.get(section, {}), **values}
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str))
            os.replace(tmp, self.path)
            return data[section]
