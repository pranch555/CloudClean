"""Scanner driver interface, shared settings helpers and the driver registry.

Conventions used by every driver:

* Units are millimetres.
* A *sensor frame* is x right, y down, z forward (OpenCV camera convention).
* ``Frame.pose`` is the 4x4 sensor->world transform when the device tracks itself.
* ``Frame.coordinates`` says how ``Frame.points`` must be interpreted by the capture session:
  ``"sensor"`` (a live depth frame; needs ``pose`` or frame-to-model tracking),
  ``"world"`` (already in the model frame, fused as is) or
  ``"scan"`` (a complete exported scan in its own coordinate system; registered to the model).
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    points: np.ndarray                      # (N, 3) float32
    colors: np.ndarray | None = None        # (N, 3) uint8
    normals: np.ndarray | None = None       # (N, 3) float32
    pose: np.ndarray | None = None          # 4x4 sensor -> world, None when the device does not track
    timestamp: float = 0.0                  # seconds (monotonic within a session)
    meta: dict = field(default_factory=dict)  # distance_mm, exposure_ms, tracking, source file, ...
    coordinates: str = "sensor"             # sensor | world | scan (see module docstring)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 3)
        if self.colors is not None:
            colors = np.asarray(self.colors)
            if colors.dtype != np.uint8:
                colors = np.clip(np.round(colors * 255.0 if colors.max(initial=0) <= 1.0 else colors), 0, 255)
            self.colors = colors.astype(np.uint8).reshape(-1, 3)
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float32).reshape(-1, 3)
        if self.pose is not None:
            self.pose = np.asarray(self.pose, dtype=float).reshape(4, 4)
        if self.coordinates not in ("sensor", "world", "scan"):
            raise ValueError(f"Unknown frame coordinates '{self.coordinates}'")


# --------------------------------------------------------------------------- settings schema helpers
def setting(key: str, label: str, type: str, default, help: str = "", **extra) -> dict:
    """One entry of a settings schema: {key, label, type: number|boolean|select|text, default, ...}."""
    if type not in ("number", "boolean", "select", "text"):
        raise ValueError(f"bad setting type {type}")
    item = {"key": key, "label": label, "type": type, "default": default}
    item.update({k: v for k, v in extra.items() if v is not None})
    if help:
        item["help"] = help
    return item


def option(value: str, label: str) -> dict:
    return {"value": value, "label": label}


def revo_settings(point_distance: float = 0.1) -> list[dict]:
    """Settings that mirror what Revo Metro offers for the MetroY family."""
    return [
        setting("scan_mode", "Scan mode", "select", "full_field_ir",
                options=[option("cross_laser", "Cross laser (fine detail)"),
                         option("parallel_lines", "Parallel laser lines (large areas)"),
                         option("full_field_ir", "Full-field IR structured light (fast fill)")],
                help="Laser modes are more accurate on detail; full-field IR fills area fastest."),
        setting("point_distance", "Accuracy / point distance", "number", point_distance, min=0.05, max=2.0,
                step=0.01, unit="mm", help="Target spacing of the fused cloud. Guidance measures density against it."),
        setting("exposure_mode", "Exposure", "select", "auto",
                options=[option("auto", "Auto"), option("manual", "Manual")]),
        setting("exposure_ms", "Exposure time", "number", 8.0, min=1.0, max=30.0, step=0.5, unit="ms",
                help="Only used with manual exposure. Dark parts need longer, shiny parts shorter exposure."),
        setting("laser_brightness", "Laser / projector brightness", "number", 6, min=1, max=10, step=1),
        setting("tracking", "Tracking", "select", "feature",
                options=[option("marker", "Markers"), option("feature", "Geometry features"),
                         option("texture", "Texture"), option("hybrid", "Hybrid (markers + features)")]),
        setting("surface", "Object surface", "select", "normal",
                options=[option("normal", "Normal"), option("dark", "Dark"),
                         option("reflective", "Reflective / shiny")]),
        setting("turntable", "Turntable", "boolean", True, help="The part sits on a rotating turntable."),
    ]


def resolve_settings(schema: list[dict], values: dict | None, extra_keys=()) -> dict:
    """Merge user values over schema defaults, coercing and validating each one. Raises ValueError."""
    values = dict(values or {})
    by_key = {s["key"]: s for s in schema}
    unknown = [k for k in values if k not in by_key and k not in extra_keys]
    if unknown:
        raise ValueError(f"Unknown capture setting(s): {', '.join(sorted(unknown))}. "
                         f"Valid: {', '.join(sorted(list(by_key) + list(extra_keys)))}")
    out = {}
    for s in schema:
        key, value = s["key"], values.get(s["key"], s["default"])
        kind = s["type"]
        try:
            if kind == "number":
                if isinstance(value, bool):
                    raise ValueError
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError
                if "min" in s and value < s["min"] or "max" in s and value > s["max"]:
                    raise ValueError(f"{s['label']} must be between {s.get('min')} and {s.get('max')}")
                if isinstance(s["default"], int) and not isinstance(s["default"], bool) and value.is_integer():
                    value = int(value)
            elif kind == "boolean":
                if isinstance(value, str):
                    if value.strip().lower() not in ("1", "0", "true", "false", "yes", "no", "on", "off"):
                        raise ValueError
                    value = value.strip().lower() in ("1", "true", "yes", "on")
                value = bool(value)
            elif kind == "select":
                allowed = [o["value"] for o in s.get("options", [])]
                if value not in allowed:
                    raise ValueError(f"{s['label']} must be one of: {', '.join(allowed)}")
            else:
                value = "" if value is None else str(value)
        except ValueError as exc:
            raise ValueError(str(exc) or f"Invalid value {value!r} for {s['label']}") from None
        out[key] = value
    for k in extra_keys:
        if k in values:
            out[k] = values[k]
    return out


# --------------------------------------------------------------------------- driver interface
class ScannerDriver(ABC):
    id = "base"
    name = "Scanner"
    kind = "scanner"          # scanner | simulated | bridge | folder | depth_camera
    description = ""

    def __init__(self):
        self.settings: dict = {}
        self.connected = False
        self.running = False

    # -- description
    def availability(self) -> tuple[bool, str]:
        """(available, reason). Override when the driver depends on hardware or optional libraries."""
        return True, ""

    def info(self) -> dict:
        available, reason = self.availability()
        return {"id": self.id, "name": self.name, "kind": self.kind, "description": self.description,
                "available": bool(available), "reason": reason}

    def settings_schema(self) -> list[dict]:
        return []

    def capabilities(self) -> dict:
        """Static facts the session uses for guidance. sweeps: each frame sees only a few laser lines, so a surface
        builds up only while the part moves through them (a turntable then turns while scanning, never stop-and-go)."""
        return {"range_mm": None, "optimal_mm": None, "streaming": True, "provides_pose": False, "sweeps": False}

    # -- lifecycle
    def connect(self, settings: dict | None = None) -> dict:
        available, reason = self.availability()
        if not available:
            raise RuntimeError(reason or f"{self.name} is not available")
        self.settings = resolve_settings(self.settings_schema(), settings)
        self._connect()
        self.connected = True
        return self.settings

    def _connect(self) -> None:
        pass

    def start(self) -> None:
        if not self.connected:
            raise RuntimeError("Connect the scanner first")
        self._start()
        self.running = True

    def _start(self) -> None:
        pass

    @abstractmethod
    def read(self, timeout: float = 0.1) -> Frame | None:
        """Next frame, or None when nothing arrived within `timeout` seconds."""

    def command(self, name: str, args: dict | None = None) -> dict:
        """Driver-specific actions (e.g. the MetroY's marker mapping). Raises ValueError when unsupported."""
        raise ValueError(f"{self.name} has no command '{name}'")

    def status(self) -> dict:
        """Live driver state for the UI (counters, phase); empty when there is nothing to report."""
        return {}

    def stop(self) -> None:
        self.running = False

    def disconnect(self) -> None:
        self.stop()
        self.connected = False

    @property
    def exhausted(self) -> bool:
        """True when the driver will never deliver another frame (e.g. a finite simulation)."""
        return False


# --------------------------------------------------------------------------- registry
DRIVERS = {
    "simulated": "cloudclean.capture.drivers.simulated:SimulatedDriver",
    "metroy_usb": "cloudclean.capture.drivers.metroy_usb:MetroyUsbDriver",
    "revo_bridge": "cloudclean.capture.drivers.revo_bridge:RevoBridgeDriver",
    "folder": "cloudclean.capture.drivers.folder:FolderDriver",
    "revopoint_sdk": "cloudclean.capture.drivers.revopoint_sdk:RevopointSdkDriver",
    "realsense": "cloudclean.capture.drivers.depth_camera:RealSenseDriver",
    "orbbec": "cloudclean.capture.drivers.depth_camera:OrbbecDriver",
}


def driver_class(driver_id: str):
    if driver_id not in DRIVERS:
        raise KeyError(f"Unknown scanner driver '{driver_id}'. Available: {', '.join(DRIVERS)}")
    module, cls = DRIVERS[driver_id].split(":")
    return getattr(importlib.import_module(module), cls)


def create_driver(driver_id: str) -> ScannerDriver:
    return driver_class(driver_id)()


def available_drivers() -> list[dict]:
    """Info + settings schema + capabilities of every registered driver (unavailable ones included)."""
    out = []
    for driver_id in DRIVERS:
        try:
            drv = create_driver(driver_id)
            out.append({**drv.info(), "settings": drv.settings_schema(), "capabilities": drv.capabilities()})
        except Exception as exc:  # a broken optional driver must not hide the others
            out.append({"id": driver_id, "name": driver_id, "kind": "scanner", "description": "",
                        "available": False, "reason": f"driver failed to load: {exc}", "settings": [],
                        "capabilities": {}})
    return out
