"""A simulated Revopoint dual-axis turntable with realistic timing.

Rotation runs at 360 / s_per_rev degrees per second plus a small start/stop overhead (the user's real table took
1.49-1.53 s for 10 deg at 53-54 s/rev, i.e. ~0.03-0.06 s more than the nominal time), tilt at the 6 deg/s Revo Metro
assumes. Moves are interruptible (`stop()` freezes the platter where it is) and every method is thread-safe.

``time_scale`` > 1 runs the clock faster (tests, demos). While connected, the simulated scanner driver
(`capture/drivers/simulated.py`) can follow the platter: see `follow_pose()`.
"""
from __future__ import annotations

import threading
import time
import weakref

from .base import ASSUMED_TILT_RATE_DEG_S, DUAL_AXIS_CAPABILITIES, TurntableDriver, TurntableError

START_STOP_OVERHEAD_S = 0.04
DEFAULT_SPEED_S_PER_REV = 53     # what Revo Metro set on the user's table after connecting (logs)

_ACTIVE: "weakref.ReferenceType[SimulatedTurntable] | None" = None
_ACTIVE_LOCK = threading.Lock()


def follow_pose() -> tuple[float, float] | None:
    """(angle_deg, tilt_deg) of the connected simulated turntable, or None. Used by the simulated scanner."""
    with _ACTIVE_LOCK:
        tt = _ACTIVE() if _ACTIVE is not None else None
    if tt is None or not tt.connected:
        return None
    st = tt.state()
    return st["angle_deg"], st["tilt_deg"]


class _Motion:
    __slots__ = ("axis", "t0", "start", "target", "rate", "overhead", "direction")

    def __init__(self, axis: str, t0: float, start: float, target: float | None, rate: float, overhead: float,
                 direction: float = 1.0):
        self.axis, self.t0, self.start, self.target = axis, t0, start, target
        self.rate, self.overhead, self.direction = rate, overhead, direction

    def value(self, now: float) -> tuple[float, bool]:
        """(position, finished) at simulation time `now`."""
        run = max(0.0, now - self.t0 - self.overhead)
        if self.target is None:                                      # continuous
            return self.start + self.direction * self.rate * run, False
        span = self.target - self.start
        done = self.rate * run
        if done >= abs(span):
            return self.target, now - self.t0 >= abs(span) / self.rate + 2 * self.overhead
        return self.start + (done if span > 0 else -done), False


class SimulatedTurntable(TurntableDriver):
    id = "simulated"
    name = "Simulated dual-axis turntable"
    kind = "simulated"
    validated = True   # nothing to validate: this is the reference behaviour

    def __init__(self, time_scale: float = 1.0, speed_s_per_rev: float = DEFAULT_SPEED_S_PER_REV,
                 angle_deg: float = 0.0, tilt_deg: float = 0.0):
        super().__init__()
        if not time_scale > 0:
            raise ValueError("time_scale must be positive")
        self.time_scale = float(time_scale)
        self.firmware = "sim-3.28"
        self._cond = threading.Condition(self.lock)
        self._angle = float(angle_deg)
        self._tilt = float(tilt_deg)
        self._speed = float(speed_s_per_rev)
        self._turn: _Motion | None = None
        self._tilting: _Motion | None = None
        self._epoch = 0                       # incremented by stop(): waiting moves see it and give up
        self.commands: list[str] = []         # protocol-equivalent log (for tests and the UI)

    def capabilities(self) -> dict:
        return dict(DUAL_AXIS_CAPABILITIES)

    # ----------------------------------------------------------------- clock
    def _now(self) -> float:
        return time.monotonic() * self.time_scale

    def _settle(self, now: float | None = None) -> None:
        """Fold finished motions into the resting position (call with the lock held)."""
        now = self._now() if now is None else now
        for attr, pos_attr in (("_turn", "_angle"), ("_tilting", "_tilt")):
            m = getattr(self, attr)
            if m is None:
                continue
            value, finished = m.value(now)
            if finished:
                setattr(self, pos_attr, value)
                setattr(self, attr, None)
                self._cond.notify_all()

    # ----------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        global _ACTIVE
        with self.lock:
            self.connected = True
            self.error = None
        with _ACTIVE_LOCK:
            _ACTIVE = weakref.ref(self)

    def disconnect(self) -> None:
        global _ACTIVE
        self.stop()
        with self.lock:
            self.connected = False
            self._cond.notify_all()
        with _ACTIVE_LOCK:
            if _ACTIVE is not None and _ACTIVE() is self:
                _ACTIVE = None

    def _require(self) -> None:
        if not self.connected:
            raise TurntableError("The simulated turntable is not connected")

    # ----------------------------------------------------------------- state
    def state(self) -> dict:
        with self.lock:
            now = self._now()
            self._settle(now)
            angle = self._turn.value(now)[0] if self._turn else self._angle
            tilt = self._tilting.value(now)[0] if self._tilting else self._tilt
            return {"angle_deg": round(angle, 3), "tilt_deg": round(tilt, 3),
                    "moving": self._turn is not None or self._tilting is not None,
                    "rotating": self._turn is not None, "tilting": self._tilting is not None,
                    "continuous": bool(self._turn is not None and self._turn.target is None),
                    "speed_s_per_rev": self._speed, "error": self.error}

    # ----------------------------------------------------------------- motion
    def set_speed(self, s_per_rev: int) -> dict:
        with self.lock:
            self._require()
            lo, hi = DUAL_AXIS_CAPABILITIES["speed_range"]
            if not lo <= s_per_rev <= hi:
                raise TurntableError(f"Speed must be between {lo} and {hi} s per revolution")
            self._speed = float(s_per_rev)
            self.commands.append(f"+CT,TURNSPEED={int(s_per_rev)};")
            return {"speed_s_per_rev": self._speed}

    def _wait(self, attr: str, epoch: int, timeout: float | None) -> bool:
        """Wait (lock held) until motion `attr` finished; False when stopped or timed out."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._settle()
            if self._epoch != epoch:
                return False
            if getattr(self, attr) is None:
                return True
            m = getattr(self, attr)
            remaining = (abs(m.target - m.start) / m.rate + 2 * m.overhead) if m.target is not None else 0.05
            elapsed = self._now() - m.t0
            step = max(0.002, min(0.05, (remaining - elapsed) / self.time_scale + 0.001))
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                step = min(step, left)
            self._cond.wait(step)

    def rotate(self, degrees: int, timeout: float | None = None) -> dict:
        t_start = time.monotonic()
        with self.lock:
            self._require()
            self._settle()
            if self._turn is not None:
                raise TurntableError("The turntable is already rotating")
            rate = 360.0 / self._speed
            self._turn = _Motion("turn", self._now(), self._angle, self._angle + degrees, rate,
                                 START_STOP_OVERHEAD_S)
            self.commands.append(f"+CT,TURNANGLE={int(degrees)};")
            epoch = self._epoch
            finished = self._wait("_turn", epoch, timeout)
            if not finished and self._epoch == epoch:     # timed out: stop rather than leave it spinning
                self._stop_locked()
                raise TurntableError(f"The rotation did not finish within {timeout:.0f} s and was stopped")
            return {"angle_deg": round(self._angle, 3), "stopped": not finished,
                    "duration_s": round(time.monotonic() - t_start, 3)}

    def rotate_continuous(self, clockwise: bool) -> dict:
        with self.lock:
            self._require()
            self._settle()
            if self._turn is not None:
                raise TurntableError("The turntable is already rotating")
            self._turn = _Motion("turn", self._now(), self._angle, None, 360.0 / self._speed,
                                 START_STOP_OVERHEAD_S, 1.0 if clockwise else -1.0)
            self.commands.append(f"+CT,TURNCONTINUE={1 if clockwise else -1};")
            return {"continuous": True}

    def tilt(self, degrees: int, timeout: float | None = None) -> dict:
        t_start = time.monotonic()
        with self.lock:
            self._require()
            lo, hi = DUAL_AXIS_CAPABILITIES["tilt_range"]
            if not lo <= degrees <= hi:
                raise TurntableError(f"Tilt must be between {lo} and {hi} degrees")
            self._settle()
            if self._tilting is not None:
                raise TurntableError("The turntable is already tilting")
            self._tilting = _Motion("tilt", self._now(), self._tilt, float(degrees), ASSUMED_TILT_RATE_DEG_S,
                                    START_STOP_OVERHEAD_S)
            self.commands.append(f"+CR,TILTVALUE={int(degrees)};")
            epoch = self._epoch
            finished = self._wait("_tilting", epoch, timeout)
            if not finished and self._epoch == epoch:
                self._stop_locked()
                raise TurntableError(f"The tilt did not finish within {timeout:.0f} s and was stopped")
            return {"tilt_deg": round(self._tilt, 3), "stopped": not finished,
                    "duration_s": round(time.monotonic() - t_start, 3)}

    def _stop_locked(self) -> None:
        now = self._now()
        if self._turn is not None:
            self._angle = self._turn.value(now)[0]
            self._turn = None
        if self._tilting is not None:
            self._tilt = self._tilting.value(now)[0]
            self._tilting = None
        self._epoch += 1
        self._cond.notify_all()

    def stop(self) -> dict:
        with self.lock:
            was_moving = self._turn is not None or self._tilting is not None
            self._stop_locked()
            if self.connected:
                self.commands.extend(["+CT,STOP;", "+CR,STOP;"])
            return {"stopped": was_moving, "angle_deg": round(self._angle, 3), "tilt_deg": round(self._tilt, 3)}
