"""Shared helpers for parameter dataclasses (dict / CLI "key=value" overrides)."""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from typing import Any


def _coerce(value: Any, current: Any) -> Any:
    """Convert CLI strings / JSON values to the type of the field's current value."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if isinstance(current, bool):
        if text.lower() in {"1", "true", "yes", "on"}:
            return True
        if text.lower() in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Expected a boolean, got '{value}'")
    if isinstance(current, int):
        return int(float(text))
    if isinstance(current, float):
        return float(text)
    if current is None or isinstance(current, (list, tuple)):
        if text.lower() in {"none", "null", ""}:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if "," in text:
                return [float(x) for x in text.split(",")]
            return text
    return text


class ParamsMixin:
    @classmethod
    def from_dict(cls, data: dict | None = None):
        obj = cls()
        obj.update(data or {})
        return obj

    def update(self, data: dict) -> None:
        valid = {f.name for f in fields(self)}
        for key, value in data.items():
            if key not in valid:
                raise ValueError(
                    f"Unknown parameter '{key}' for {type(self).__name__}. "
                    f"Valid parameters: {', '.join(sorted(valid))}"
                )
            setattr(self, key, _coerce(value, getattr(self, key)))

    def update_from_strings(self, pairs: list[str] | None) -> None:
        data = {}
        for pair in pairs or []:
            if "=" not in pair:
                raise ValueError(f"Expected key=value, got '{pair}'")
            key, value = pair.split("=", 1)
            data[key.strip()] = value
        self.update(data)

    def to_dict(self) -> dict:
        return asdict(self)
