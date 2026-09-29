"""The turntable behind the web routes and the assistant (Contract 4 in docs/v3-plan.md).

One `TurntableManager` per workspace root (`get_turntable_manager`). It owns at most one connected driver
(simulated or Bluetooth), runs manual moves in a background thread (``wait=False``) and step-and-scan programs
(`program.ProgramRunner`) that can drive the live capture session. Every call returns the status dict; failures
raise `TurntableError` whose message is one sentence (the routes turn it into HTTP 400).
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from . import ble
from .base import TurntableDriver, TurntableError
from .program import ProgramRunner, validate_program
from .protocol import classify_name, whole_degrees
from .simulated import SimulatedTurntable

SIMULATED_DEVICE = {"id": "simulated", "name": "Simulated dual-axis turntable", "kind": "simulated", "rssi": None}
MAX_MANUAL_TURN_DEG = 720
WAIT_LIMIT_S = 300.0

_MANAGERS: dict[str, "TurntableManager"] = {}
_MANAGERS_LOCK = threading.Lock()


def _root_of(workspace) -> Path:
    if isinstance(workspace, (str, os.PathLike)):      # note: a Path has its own `.root` ("\\" or "/")
        return Path(workspace).resolve()
    return Path(workspace.root).resolve()


def get_turntable_manager(workspace) -> "TurntableManager":
    """The manager of a workspace (a `Workspace` or its root path); created on first use."""
    root = _root_of(workspace)
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(str(root))
        if manager is None or manager.closed:
            manager = TurntableManager(root)
            _MANAGERS[str(root)] = manager
        return manager


class TurntableManager:
    def __init__(self, root):
        self.root = Path(root)
        self.closed = False
        self.driver: TurntableDriver | None = None
        self.direction = "cw"
        self.error: str | None = None
        self.program: ProgramRunner | None = None
        self._op_lock = threading.RLock()        # serializes connect/disconnect/start/stop
        self._scan_lock = threading.Lock()
        self._move: dict | None = None
        self._move_thread: threading.Thread | None = None
        self._known: dict[str, dict] = {}        # devices seen by the last scans, by id
        self._bluetooth = {"available": None, "reason": None}
        self._log_lines: deque[str] = deque(maxlen=200)
        # hooks (tests swap these)
        self.scan_fn = ble.scan
        self.ble_factory = ble.RevopointBleTurntable
        self.capture_provider = None

    # ----------------------------------------------------------------- helpers
    def log(self, msg: str) -> None:
        self._log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _settings(self):
        from ...web.settings import Settings
        return Settings(self.root)

    def _remembered(self) -> dict | None:
        try:
            return self._settings().get("turntable", {}).get("last_device")
        except Exception:
            return None

    def _remember(self, info: dict) -> None:
        try:
            self._settings().update("turntable", {"last_device": {k: info.get(k) for k in ("id", "name", "kind")}})
        except Exception:
            pass

    def _capture_manager(self):
        if self.capture_provider is not None:
            return self.capture_provider()
        from ...web.routes_capture import manager_for
        return manager_for(self.root)

    def _require_driver(self) -> TurntableDriver:
        drv = self.driver
        if drv is None or not drv.connected:
            raise TurntableError("No turntable is connected - connect one first (the simulated turntable is "
                                 "always available)")
        return drv

    def _program_running(self) -> bool:
        return self.program is not None and self.program.running

    def _move_running(self) -> bool:
        return self._move_thread is not None and self._move_thread.is_alive()

    def _require_idle(self) -> TurntableDriver:
        drv = self._require_driver()
        if self._program_running():
            raise TurntableError("A turntable program is running - stop it first")
        if self._move_running() or drv.state().get("moving"):
            raise TurntableError("The turntable is still moving - wait for it or stop it first")
        return drv

    # ----------------------------------------------------------------- devices
    def devices(self, scan_seconds: float = 4.0) -> list[dict]:
        """[{id, name, kind, rssi, connected, remembered}] - the simulated turntable plus Revopoint turntables
        advertising nearby (a Bluetooth scan of `scan_seconds`, 0 = no scan, max 15)."""
        seconds = max(0.0, min(float(scan_seconds or 0.0), 15.0))
        found: list[dict] = []
        if seconds > 0:
            with self._scan_lock:
                try:
                    found = self.scan_fn(seconds)
                    self._bluetooth = {"available": True, "reason": None}
                except TurntableError as exc:
                    self._bluetooth = {"available": False, "reason": str(exc)}
                    self.log(f"Bluetooth scan: {exc}")
            for dev in found:
                self._known[dev["id"]] = dev
        remembered = self._remembered()
        drv = self.driver
        out = [{**SIMULATED_DEVICE, "connected": bool(drv and drv.kind == "simulated" and drv.connected),
                "remembered": False}]
        ids = set()
        for dev in found:
            ids.add(dev["id"])
            out.append({"id": dev["id"], "name": dev.get("name"), "kind": dev.get("kind"), "rssi": dev.get("rssi"),
                        "connected": bool(drv and drv.connected and drv.id == dev["id"]),
                        "remembered": bool(remembered and remembered.get("id") == dev["id"])})
        # a connected table does not advertise: list it anyway
        if drv is not None and drv.connected and drv.kind != "simulated" and drv.id not in ids:
            out.append({"id": drv.id, "name": drv.name, "kind": drv.kind, "rssi": None, "connected": True,
                        "remembered": bool(remembered and remembered.get("id") == drv.id)})
        return out

    def _resolve(self, device: str | None, kind: str) -> dict:
        """Turn a device id / name / None into {id, name, kind}."""
        if device is None:
            remembered = self._remembered()
            if remembered and remembered.get("id"):
                return {**remembered, "kind": kind if kind != "auto" else remembered.get("kind")}
            found = [d for d in self.devices(4.0) if d["kind"] in ("dual_axis", "large")]
            if not found:
                raise TurntableError("No Revopoint turntable was found nearby. Switch it on and disconnect it from "
                                     "Revo Metro (it accepts one connection at a time), or connect to 'simulated'.")
            device = found[0]["id"]
        key = str(device).strip()
        info = self._known.get(key) or next((d for d in self._known.values()
                                             if d["id"].lower() == key.lower() or d.get("name") == key), None)
        if info is None:
            remembered = self._remembered()
            if remembered and remembered.get("id", "").lower() == key.lower():
                info = remembered
        if info is None and kind == "auto":
            with self._scan_lock:
                try:
                    for dev in self.scan_fn(4.0):
                        self._known[dev["id"]] = dev
                except TurntableError as exc:
                    raise TurntableError(f"Cannot look up turntable '{key}': {exc}") from None
            info = self._known.get(key) or next((d for d in self._known.values()
                                                 if d["id"].lower() == key.lower() or d.get("name") == key), None)
        name = (info or {}).get("name")
        resolved_kind = kind if kind != "auto" else ((info or {}).get("kind") or classify_name(name))
        if resolved_kind not in ("dual_axis", "large"):
            raise TurntableError(f"'{key}' is not a known Revopoint turntable; pass kind 'dual_axis' or 'large' if "
                                 "you are sure")
        return {"id": (info or {}).get("id", key), "name": name, "kind": resolved_kind}

    # ----------------------------------------------------------------- lifecycle
    def connect(self, device: str | None = None, kind: str = "auto", options: dict | None = None) -> dict:
        kind = (kind or "auto").strip().lower()
        if kind not in ("auto", "dual_axis", "large", "simulated"):
            raise TurntableError("kind must be one of: auto, dual_axis, large, simulated")
        options = dict(options or {})
        with self._op_lock:
            if self._program_running():
                raise TurntableError("A turntable program is running - stop it first")
            if device == "simulated" or kind == "simulated":
                unknown = set(options) - {"time_scale"}
                if unknown:
                    raise TurntableError(f"Unknown simulated turntable option(s): {', '.join(sorted(unknown))}")
                scale = options.get("time_scale", 1.0)
                try:
                    scale = float(scale)
                except (TypeError, ValueError):
                    raise TurntableError("time_scale must be a number") from None
                if not 0.1 <= scale <= 1000:
                    raise TurntableError("time_scale must be between 0.1 and 1000")
                new = SimulatedTurntable(time_scale=scale)
                info = SIMULATED_DEVICE
            else:
                if options:
                    raise TurntableError("Options are only supported for the simulated turntable")
                info = self._resolve(device, kind)
                new = self.ble_factory(info["id"], info.get("name"), info["kind"])
            self._disconnect_current()
            try:
                new.connect()
            except TurntableError as exc:
                self.error = str(exc)
                self.log(f"Connect failed: {exc}")
                raise
            except Exception as exc:  # never leak a raw exception to the route
                self.error = f"Could not connect: {exc}"
                raise TurntableError(self.error) from None
            self.driver, self.error = new, None
            self.log(f"Connected to {new.name}" + (f" (firmware {new.firmware})" if new.firmware else ""))
            if new.kind != "simulated":
                self._remember({"id": new.id, "name": new.name, "kind": new.kind})
            return self.status()

    def _disconnect_current(self) -> None:
        drv = self.driver
        self.driver = None
        if drv is None:
            return
        try:
            if drv.connected and drv.state().get("moving"):
                drv.stop()
        except Exception:
            pass
        try:
            drv.disconnect()
        except Exception as exc:
            self.log(f"Disconnect: {exc}")
        if self._move_thread is not None:
            self._move_thread.join(timeout=5)
        self.log(f"Disconnected from {drv.name}")

    def disconnect(self) -> dict:
        with self._op_lock:
            if self.program is not None and self.program.running:
                self.program.stop()
            self._disconnect_current()
            return self.status()

    def shutdown(self) -> None:
        with self._op_lock:
            try:
                if self.program is not None and self.program.running:
                    self.program.stop(wait=5)
                self._disconnect_current()
            finally:
                self.closed = True
                with _MANAGERS_LOCK:
                    if _MANAGERS.get(str(self.root.resolve())) is self:
                        del _MANAGERS[str(self.root.resolve())]

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        drv = self.driver
        connected = bool(drv is not None and drv.connected)
        st = {}
        if drv is not None:
            try:
                st = drv.state()
            except Exception as exc:
                st = {"error": str(exc)}
        angle = st.get("angle_deg")
        move = dict(self._move) if self._move else None
        program = self.program.snapshot() if self.program is not None else None
        return {
            "connected": connected,
            "device": drv.id if drv is not None else None,
            "name": drv.name if drv is not None else None,
            "kind": drv.kind if drv is not None else None,
            "angle_deg": angle,
            "angle_wrapped_deg": None if angle is None else round(float(angle) % 360.0, 3),
            "tilt_deg": st.get("tilt_deg"),
            "moving": bool(st.get("moving")) or self._move_running(),
            "speed_s_per_rev": st.get("speed_s_per_rev"),
            "direction": self.direction,
            "program": program,
            "error": self.error or st.get("error") or (move or {}).get("error"),
            "capabilities": drv.capabilities() if drv is not None else None,
            "validated": bool(drv.validated) if drv is not None else False,
            "firmware": getattr(drv, "firmware", None),
            "last_move": move,
            "device_state": {k: v for k, v in st.items() if k not in ("angle_deg", "tilt_deg", "speed_s_per_rev",
                                                                       "moving", "error")},
            "bluetooth": dict(self._bluetooth, installed=ble.bleak_available()[0]),
            "remembered_device": self._remembered(),
            "log": list(self._log_lines)[-20:],
        }

    # ----------------------------------------------------------------- manual motion
    def _launch(self, move: dict, fn) -> threading.Thread:
        move.update({"state": "running", "started": datetime.now().isoformat(timespec="seconds"), "finished": None,
                     "error": None, "result": None})
        self._move = move

        def run():
            try:
                result = fn()
                move["result"] = result
                move["state"] = "stopped" if result.get("stopped") else "done"
            except TurntableError as exc:
                move["state"], move["error"] = "error", str(exc)
                self.log(f"{move['type'].capitalize()} failed: {exc}")
            except Exception as exc:
                move["state"], move["error"] = "error", f"{type(exc).__name__}: {exc}"
            finally:
                move["finished"] = datetime.now().isoformat(timespec="seconds")

        self._move_thread = threading.Thread(target=run, name="turntable-move", daemon=True)
        self._move_thread.start()
        return self._move_thread

    def _finish(self, move: dict, thread: threading.Thread, wait: bool) -> dict:
        """Called without the operation lock: optionally wait for the move, then report."""
        if wait:
            thread.join(timeout=WAIT_LIMIT_S)
            if move["state"] == "error":
                raise TurntableError(move["error"])
        return {**self.status(), "move": dict(move)}

    def _check_speed(self, s_per_rev: float, caps: dict) -> tuple[int, float]:
        try:
            requested = float(s_per_rev)
            commanded = whole_degrees(requested, "speed")
        except (TypeError, ValueError):
            raise TurntableError("speed_s_per_rev must be a number") from None
        lo, hi = caps.get("speed_range") or [25, 90]
        if not lo <= commanded <= hi:
            raise TurntableError(f"Speed must be between {lo} and {hi} seconds per revolution")
        return commanded, requested

    def rotate(self, degrees: float, speed_s_per_rev: float | None = None, wait: bool = False) -> dict:
        """Turn by `degrees` (relative; positive = clockwise seen from above). Whole degrees are commanded."""
        with self._op_lock:
            drv = self._require_idle()
            caps = drv.capabilities()
            try:
                requested = float(degrees)
                commanded = whole_degrees(requested, "rotation")
            except (TypeError, ValueError):
                raise TurntableError("degrees must be a number") from None
            if commanded == 0:
                raise TurntableError("Rotate by at least 1 degree (the turntable takes whole degrees)")
            if abs(commanded) > MAX_MANUAL_TURN_DEG:
                raise TurntableError(f"Rotate at most {MAX_MANUAL_TURN_DEG} degrees per command")
            speed = None
            if speed_s_per_rev is not None:
                speed, _ = self._check_speed(speed_s_per_rev, caps)
            self.direction = "cw" if commanded > 0 else "ccw"
            move = {"type": "rotate", "requested_deg": requested, "commanded_deg": commanded,
                    "speed_s_per_rev": speed}
            if commanded != requested:
                move["note"] = f"Requested {requested:g} deg, turning {commanded} deg (whole degrees only)."
            self.log(f"Rotate {commanded:+d} deg" + (f" at {speed} s/rev" if speed else ""))

            def run():
                if speed is not None:
                    drv.set_speed(speed)
                return drv.rotate(commanded)
            thread = self._launch(move, run)
        return self._finish(move, thread, wait)

    def tilt(self, degrees: float, wait: bool = False) -> dict:
        """Absolute tilt (0 = level); dual-axis turntables only."""
        with self._op_lock:
            drv = self._require_idle()
            caps = drv.capabilities()
            if not caps.get("tilt"):
                raise TurntableError(f"The {drv.name} cannot tilt")
            try:
                requested = float(degrees)
                commanded = whole_degrees(requested, "tilt")
            except (TypeError, ValueError):
                raise TurntableError("degrees must be a number") from None
            lo, hi = caps["tilt_range"]
            if not lo <= commanded <= hi:
                raise TurntableError(f"Tilt must be between {lo} and {hi} degrees")
            move = {"type": "tilt", "requested_deg": requested, "commanded_deg": commanded}
            if commanded != requested:
                move["note"] = f"Requested {requested:g} deg, tilting to {commanded} deg (whole degrees only)."
            self.log(f"Tilt to {commanded} deg")
            thread = self._launch(move, lambda: drv.tilt(commanded))
        return self._finish(move, thread, wait)

    def set_speed(self, s_per_rev: float) -> dict:
        with self._op_lock:
            drv = self._require_idle()
            commanded, requested = self._check_speed(s_per_rev, drv.capabilities())
            result = drv.set_speed(commanded)
            self.log(f"Speed {commanded} s/rev")
            out = {**self.status(), "speed": {"requested_s_per_rev": requested, "commanded_s_per_rev": commanded,
                                              **result}}
            if commanded != requested:
                out["speed"]["note"] = f"Requested {requested:g} s/rev, set {commanded} s/rev (whole seconds only)."
            return out

    def stop(self) -> dict:
        """Stop everything now: a running program and any motion."""
        prog = self.program
        if prog is not None and prog.running:
            prog.stop()
            self.log("Program stopped")
        drv = self.driver
        if drv is not None and drv.connected:
            try:
                drv.stop()
            except TurntableError as exc:
                self.error = str(exc)
                raise
            self.log("Stop")
        if self._move_thread is not None:
            self._move_thread.join(timeout=10)
        return self.status()

    # ----------------------------------------------------------------- programs
    def start_program(self, program: dict) -> dict:
        """{interval_deg, frames_per_stop, direction, speed_s_per_rev, rotations: [{tilt_deg}], sync_scan, ...}
        (see program.py)."""
        with self._op_lock:
            drv = self._require_idle()
            plan = validate_program(program, drv.capabilities())
            self.direction = plan["direction"]
            runner = ProgramRunner(drv, plan, capture=self._capture_manager, log=lambda m: self.log(f"Program: {m}"))
            self.program = runner
            self.log(f"Program: {len(plan['rotations'])} rotation(s), {plan['stops_per_rotation']} stop(s) of "
                     f"{plan['interval_deg']} deg, {plan['frames_per_stop']} frame(s) per stop, "
                     f"{plan['direction']}, sync_scan={plan['sync_scan']}")
            runner.start()
            return self.status()

    def stop_program(self) -> dict:
        with self._op_lock:
            if self.program is not None and self.program.running:
                self.program.stop()
                self.log("Program stopped")
            return self.status()
