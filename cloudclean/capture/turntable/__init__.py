"""Revopoint turntable control (Contract 4 in docs/v3-plan.md, reference in docs/turntable.md).

* `protocol`  - the Bluetooth LE command set recovered from Revo Metro (pure Python, unit tested)
* `base`      - driver interface, `TurntableError`, capability tables
* `simulated` - `SimulatedTurntable` (realistic timing, always available)
* `ble`       - `RevopointBleTurntable` over `bleak` (optional: ``pip install "cloudclean[turntable]"``)
* `program`   - step-and-scan programs linked to the live capture
* `manager`   - `TurntableManager` / `get_turntable_manager(workspace)` used by the routes and the assistant

Importing this package never imports bleak; `ble` does that on the first scan / connection.
"""
from .base import TurntableDriver, TurntableError
from .ble import RevopointBleTurntable
from .manager import TurntableManager, get_turntable_manager
from .simulated import SimulatedTurntable

__all__ = ["RevopointBleTurntable", "SimulatedTurntable", "TurntableDriver", "TurntableError", "TurntableManager",
           "get_turntable_manager"]
