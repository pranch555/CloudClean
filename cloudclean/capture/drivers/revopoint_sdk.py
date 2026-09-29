"""Slot for Revopoint's own SDK (direct streaming from a MetroY / MetroY Ultra without Revo Metro).

Revopoint publishes no SDK for the MetroY family: it has to be requested from customer@revopoint3d.com (the
consumer-scanner SDK trial only exposes hardware settings, no point data). Until a build with point data is
available this driver reports `available: false` with instructions.

When the SDK arrives, wiring it in is limited to `_SdkBinding` below (every place is marked
``TODO(revopoint-sdk)``): load the library, open the device, map the settings, grab frames. Everything else
(tracking, fusion, guidance, UI) already works through the generic `ScannerDriver` interface. Note that the
CloudClean server runs on Linux aarch64 (DGX Spark); a Windows-only SDK must instead run on the scanning PC
and push frames to CloudClean (see `cloudclean.capture.bridge`)."""
from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
import platform
import sys
from pathlib import Path

from .base import Frame, ScannerDriver, revo_settings, setting

ENV_VAR = "REVOPOINT_SDK_PATH"
PYTHON_MODULES = ("revopoint", "revopoint_sdk", "revosdk", "RevoScanSDK")
LIBRARY_PATTERNS = ("*revo*sdk*", "*Revo*SDK*", "*revopoint*", "*RevoScan*", "*MetroY*")
LIBRARY_SUFFIXES = {".dll", ".so", ".dylib"}
HOW_TO_GET = ("Revopoint does not publish an SDK for the MetroY scanners. Ask customer@revopoint3d.com for the "
              "MetroY SDK (with point-cloud streaming, for Linux aarch64 if CloudClean runs on a DGX Spark), then "
              f"set the SDK folder in the settings or the {ENV_VAR} environment variable. Until then use the "
              "'Revopoint (Revo Metro bridge)' driver.")


def find_sdk(sdk_path: str = "") -> dict:
    """Look for the SDK: an importable Python module or a shared library in the configured folder."""
    folder = (sdk_path or os.environ.get(ENV_VAR, "")).strip().strip('"')
    found = {"folder": folder or None, "python_module": None, "library": None}
    if folder:
        base = Path(folder).expanduser()
        if not base.exists():
            found["error"] = f"SDK folder not found: {base}"
            return found
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        for pattern in LIBRARY_PATTERNS:
            libs = [p for p in base.rglob(pattern) if p.suffix.lower() in LIBRARY_SUFFIXES]
            if libs:
                found["library"] = str(sorted(libs)[0])
                break
    for name in PYTHON_MODULES:
        try:
            if importlib.util.find_spec(name) is not None:
                found["python_module"] = name
                break
        except (ImportError, ValueError):
            continue
    return found


class _SdkBinding:
    """Thin adapter around Revopoint's SDK. Only this class needs real SDK calls."""

    WIRED = False   # TODO(revopoint-sdk): set to True once the calls below are implemented against the real SDK

    def __init__(self, found: dict):
        self.found = found
        self.module = None
        self.lib = None

    def load(self) -> None:
        if self.found.get("python_module"):
            self.module = importlib.import_module(self.found["python_module"])
        elif self.found.get("library"):
            self.lib = ctypes.CDLL(self.found["library"])
        # TODO(revopoint-sdk): initialise the SDK (licence / context creation).

    def open(self, settings: dict) -> dict:
        # TODO(revopoint-sdk): enumerate devices, open the first MetroY (USB or Wi-Fi) and return
        # {"serial", "model", "range_mm": [near, far], "optimal_mm"} from the device info.
        raise NotImplementedError("Revopoint SDK device opening is not wired yet")

    def configure(self, settings: dict) -> None:
        # TODO(revopoint-sdk): map settings -> SDK parameters:
        #   scan_mode (cross_laser | parallel_lines | full_field_ir), point_distance (mm),
        #   exposure_mode / exposure_ms, laser_brightness, tracking (marker|feature|texture|hybrid),
        #   surface (normal|dark|reflective), turntable (bool).
        raise NotImplementedError("Revopoint SDK settings are not wired yet")

    def start(self) -> None:
        # TODO(revopoint-sdk): start the depth / point stream.
        raise NotImplementedError

    def grab(self, timeout: float) -> Frame | None:
        # TODO(revopoint-sdk): wait up to `timeout` for the next frame and convert it:
        #   Frame(points=(N,3) float32 mm in sensor coordinates, colors=(N,3) uint8 or None,
        #         pose=4x4 sensor->world if the SDK tracks (markers / features) else None,
        #         timestamp=seconds, meta={"distance_mm": ..., "exposure_ms": ..., "tracking": ...},
        #         coordinates="sensor")  # or "world" if the SDK already returns fused, tracked points
        raise NotImplementedError

    def stop(self) -> None:
        # TODO(revopoint-sdk): stop streaming.
        pass

    def close(self) -> None:
        # TODO(revopoint-sdk): close the device and release the SDK.
        pass


class RevopointSdkDriver(ScannerDriver):
    id = "revopoint_sdk"
    name = "Revopoint SDK (direct)"
    kind = "scanner"
    description = "Streams directly from a MetroY scanner through Revopoint's SDK, once you have obtained it."

    def __init__(self):
        super().__init__()
        self._binding: _SdkBinding | None = None
        self._device: dict = {}

    def settings_schema(self) -> list[dict]:
        return [setting("sdk_path", "SDK folder", "text", "", help=f"Or set the {ENV_VAR} environment variable.")] + \
            revo_settings(0.1)

    def availability(self, sdk_path: str = "") -> tuple[bool, str]:
        found = find_sdk(sdk_path or self.settings.get("sdk_path", ""))
        if found.get("error"):
            return False, f"{found['error']}. {HOW_TO_GET}"
        if not (found["python_module"] or found["library"]):
            return False, HOW_TO_GET
        where = found["python_module"] or found["library"]
        if not _SdkBinding.WIRED:
            return False, (f"Revopoint SDK found ({where}), but CloudClean's binding is not wired to it yet - "
                           "implement the TODO(revopoint-sdk) calls in cloudclean/capture/drivers/revopoint_sdk.py.")
        return True, f"SDK: {where} ({platform.system()} {platform.machine()})"

    def capabilities(self) -> dict:
        return {"range_mm": self._device.get("range_mm"), "optimal_mm": self._device.get("optimal_mm"),
                "streaming": True, "provides_pose": True}

    def connect(self, settings: dict | None = None) -> dict:
        from .base import resolve_settings

        resolved = resolve_settings(self.settings_schema(), settings)
        available, reason = self.availability(resolved["sdk_path"])
        if not available:
            raise RuntimeError(reason)
        self.settings = resolved
        self._binding = _SdkBinding(find_sdk(resolved["sdk_path"]))
        self._binding.load()
        self._device = self._binding.open(resolved)
        self._binding.configure(resolved)
        self.connected = True
        return self.settings

    def _start(self) -> None:
        self._binding.start()

    def read(self, timeout: float = 0.1) -> Frame | None:
        if not self.running or self._binding is None:
            return None
        return self._binding.grab(timeout)

    def stop(self) -> None:
        if self.running and self._binding is not None:
            self._binding.stop()
        super().stop()

    def disconnect(self) -> None:
        super().disconnect()
        if self._binding is not None:
            self._binding.close()
            self._binding = None
