"""Turntable driver interface shared by the simulated and the Bluetooth drivers.

Conventions (all drivers):

* Angles in degrees. ``rotate(+d)`` turns the platter **clockwise seen from above** - the direction Revo Metro
  labels "Clockwise" (it sends a positive ``TURNANGLE``). ``angle_deg`` is the table's own cumulative position
  (the Revopoint firmware reports e.g. -145.72 after several moves), not wrapped to 0..360.
* Tilt is **absolute** (0 = level), dual-axis tables only.
* The Revopoint protocol carries whole degrees and whole seconds per revolution; drivers take ints, the manager
  rounds and reports requested vs. commanded values.
* ``rotate`` / ``tilt`` block until the move has finished (or was stopped) and must be safe to interrupt with
  ``stop()`` from another thread.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod

# Ranges Revo Metro offers in its UI [verified: lang/RevoScan_en_US.qm + PartSetAutoDialog]
DUAL_AXIS_CAPABILITIES = {"tilt": True, "tilt_range": [-30, 30], "speed_range": [25, 90], "interval_range": [5, 30],
                          "max_rotations": 5, "continuous": True, "whole_degrees": True}
LARGE_CAPABILITIES = {"tilt": False, "tilt_range": None, "speed_range": [35, 90], "interval_range": [5, 30],
                      "max_rotations": 5, "continuous": True, "whole_degrees": True}

# Tilt rate Revo Metro assumes when waiting for a tilt to finish: |delta| * 10 / 60 s  (= 6 deg/s) [verified
# formula at RevoMetro.exe 0x140568e4c, logged as "Y motion cost time milliseconds = 166.667" for 1 deg].
ASSUMED_TILT_RATE_DEG_S = 6.0


class TurntableError(Exception):
    """A user-facing failure; the message is one plain sentence."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class TurntableDriver(ABC):
    id = "base"
    name = "Turntable"
    kind = "dual_axis"            # dual_axis | large | simulated
    validated = False             # True once the protocol was confirmed on the real device

    def __init__(self):
        self.lock = threading.RLock()
        self.connected = False
        self.firmware: str | None = None
        self.error: str | None = None

    # -- description
    def capabilities(self) -> dict:
        return dict(LARGE_CAPABILITIES if self.kind == "large" else DUAL_AXIS_CAPABILITIES)

    def info(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "validated": self.validated,
                "firmware": self.firmware}

    # -- lifecycle
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    # -- state
    @abstractmethod
    def state(self) -> dict:
        """{angle_deg, tilt_deg, moving, speed_s_per_rev, error, ...}; must not block on a running move."""

    # -- motion (blocking, interruptible by stop())
    @abstractmethod
    def set_speed(self, s_per_rev: int) -> dict: ...

    @abstractmethod
    def rotate(self, degrees: int, timeout: float | None = None) -> dict:
        """Relative turn; returns {"angle_deg", "stopped": bool, "duration_s", ...}."""

    def tilt(self, degrees: int, timeout: float | None = None) -> dict:
        raise TurntableError(f"The {self.name} cannot tilt")

    def rotate_continuous(self, clockwise: bool) -> dict:
        raise TurntableError(f"The {self.name} cannot rotate continuously")

    @abstractmethod
    def stop(self) -> dict:
        """Stop every axis now (emergency stop); wakes any rotate()/tilt() waiting in another thread."""
