"""The turntable behind the web routes and the assistant (Contract 4 in docs/v3-plan.md).

One `TurntableManager` per workspace root (`get_turntable_manager`). It owns at most one connected driver
(simulated or Bluetooth), runs manual moves in a background thread (``wait=False``) and step-and-scan programs
(`program.ProgramRunner`) that can drive the live capture session. Every call returns the status dict; failures
raise `TurntableError` whose message is one sentence (the routes turn it into HTTP 400).

Restarts. Every service restart drops the Bluetooth link (the Spark restarts itself after each update), and a
turntable scan then ran with the table standing still: every frame saw the same side and the scan barely grew.
So the manager keeps, in the workspace settings (section "turntable"), the table to connect again (``reconnect``,
cleared by an explicit disconnect) and the Turn while scanning setting (``spin``, cleared by an explicit spin off or
Stop). At service start `start_auto_reconnect` connects that table in the background, a few quiet tries, and the
turning is picked up again after every (re)connect. ``status()["scan_warning"]`` says so in plain words while a scan
runs and the table that should turn it does not.
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
# pause before each reconnect try at service start: the first lets the service finish starting, the later ones give
# a table that is being switched on or freed time; a Bluetooth try that finds nothing takes ~10 s more (FIND_TIMEOUT_S)
RECONNECT_DELAYS_S = (1.0, 5.0, 15.0, 40.0)
# longest Bluetooth try (find 10 s + connect 25 s + 10 s, then up to 5 queries): waited out, two connects never
# overlap; below systemd's 90 s stop timeout so a stopping service still lets go of the table
RECONNECT_JOIN_S = 85.0
NOT_TURNING_GRACE_S = 2.0        # the scan follower turns the table within 0.2 s of the scan running: no flicker
FOLLOW_RETRY_S = 5.0             # a turn the table refused is tried again this much later, not every 0.2 s

NOT_CONNECTED_WARNING = ("The turntable is not connected - the scan sees the same side over and over. Reconnect it "
                         "(Scan → Turntable)")
RECONNECTING_WARNING = ("The turntable is still reconnecting - the scan sees the same side over and over. Pause the "
                        "scan until it turns (Scan → Turntable)")
NOT_TURNING_WARNING = ("The turntable is not turning - the scan sees the same side over and over. Press Start "
                       "turning (Scan → Turntable)")
STALLED_WARNING = ("The turntable stopped turning - the scan sees the same side over and over. Press Stop turning, "
                   "then Start turning (Scan → Turntable)")

_MANAGERS: dict[str, "TurntableManager"] = {}
_MANAGERS_LOCK = threading.Lock()
_UNSET = object()
_ASK = object()
_IN_SCAN_STATE = threading.local()


def _retryable(error: str | None) -> bool:
    """Bluetooth missing on this computer is not cured by waiting; a table that is off, out of range or held by Revo
    Metro may be free a moment later."""
    return not (error == ble.INSTALL_HINT or str(error).startswith("bleak failed to load"))


def _valid_target(target) -> dict | None:
    """A saved table to reconnect ({id, name, kind[, options]}), or None when the saved value is unusable."""
    if not isinstance(target, dict) or not target.get("id"):
        return None
    kind = "simulated" if target["id"] == "simulated" else target.get("kind")
    return {**target, "kind": kind} if kind in ("dual_axis", "large", "simulated") else None


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
        self._spin: dict | None = None            # turning continuously until stopped (spin())
        self._op_lock = threading.RLock()        # serializes connect/disconnect/start/stop
        self._scan_lock = threading.Lock()
        self._move: dict | None = None
        self._move_thread: threading.Thread | None = None
        self._known: dict[str, dict] = {}        # devices seen by the last scans, by id
        self._bluetooth = {"available": None, "reason": None}
        self._log_lines: deque[str] = deque(maxlen=200)
        # what survives a restart (settings section "turntable"; only this manager writes these keys): read once
        self._target = _UNSET                    # the table to connect again at service start, None = disconnected
        self._spin_saved: dict | None = None     # the Turn while scanning setting {on, follow_scan, speed, direction}
        # _spin, _spin_saved and _spin_gen change together under this short lock: stop() takes no operation lock
        # (an emergency stop never waits), and a spin being started or restored must see a Stop that came in
        self._spin_lock = threading.Lock()
        self._spin_gen = 0                       # +1 on every Stop / end of turning
        self._auto: dict | None = None           # the reconnect after a restart (status()["auto_reconnect"])
        self._auto_thread: threading.Thread | None = None
        self._auto_cancel = threading.Event()
        self._not_turning_since: float | None = None
        # hooks (tests swap these)
        self.scan_fn = ble.scan
        self.ble_factory = ble.RevopointBleTurntable
        self.capture_provider = None
        self.reconnect_delays_s = RECONNECT_DELAYS_S

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

    def _save(self, **values) -> None:
        try:
            self._settings().update("turntable", values)
        except Exception as exc:  # a full disk must not break turning the table
            self.log(f"Could not save the turntable settings: {exc}")

    def _section(self) -> dict:
        try:
            return self._settings().get("turntable", {})
        except Exception:
            return {}

    def _reconnect_target(self) -> dict | None:
        """The table to connect again at service start: the one connected when the service stopped, None after an
        explicit disconnect. Settings written before this was kept name the last real table only (`last_device`)."""
        if self._target is _UNSET:
            section = self._section()
            self._target = _valid_target(section["reconnect"] if "reconnect" in section else section.get("last_device"))
        return self._target

    def _set_reconnect(self, target: dict | None) -> None:
        self._target = target
        self._save(reconnect=target)

    def _saved_spin(self) -> dict:
        """The Turn while scanning setting kept across restarts: {on, follow_scan, speed_s_per_rev, direction}."""
        if self._spin_saved is None:
            saved = self._section().get("spin")
            self._spin_saved = dict(saved) if isinstance(saved, dict) else {}
        return self._spin_saved

    def _persist_spin(self) -> None:
        # under the spin lock and always the latest value: two writers never leave an older one on disk
        with self._spin_lock:
            self._save(spin=dict(self._saved_spin()))

    def _spin_off_saved(self) -> None:
        """An explicit spin off: the table is not turned again after a reconnect or restart."""
        with self._spin_lock:
            saved = self._saved_spin()
            if not saved.get("on"):
                return
            self._spin_saved = {**saved, "on": False}
            self._save(spin=dict(self._spin_saved))

    def _capture_manager(self):
        if self.capture_provider is not None:
            return self.capture_provider()
        from ...web.routes_capture import manager_for
        return manager_for(self.root)

    def _capture_sweeps(self) -> bool:
        """The connected scanner sees only a few laser lines per frame (the MetroY): it needs the part turning."""
        try:
            session = getattr(self._capture_manager(), "session", None)
            return bool(session is not None and session.driver.capabilities().get("sweeps"))
        except Exception:
            return False

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
        if self._spin is not None:
            raise TurntableError("The turntable is set to turn while scanning - stop turning first")
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
        self._cancel_auto_reconnect()        # the user's own connect wins over the reconnect after a restart
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
                target = {"id": SIMULATED_DEVICE["id"], "name": SIMULATED_DEVICE["name"], "kind": "simulated",
                          "options": {"time_scale": scale}}
            else:
                if options:
                    raise TurntableError("Options are only supported for the simulated turntable")
                info = self._resolve(device, kind)
                target = {"id": info["id"], "name": info.get("name"), "kind": info["kind"]}
            new = self._driver_for(target)
            self._end_spin()                 # a spin belongs to the table it turned; the new one picks up the setting
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
            self._install(new, target, "Connected to")
            return self.status()

    def _driver_for(self, target: dict) -> TurntableDriver:
        if target["kind"] == "simulated":
            scale = (target.get("options") or {}).get("time_scale", 1.0)
            ok = isinstance(scale, (int, float)) and 0.1 <= scale <= 1000
            return SimulatedTurntable(time_scale=float(scale) if ok else 1.0)
        return self.ble_factory(target["id"], target.get("name"), target["kind"])

    def _install(self, new: TurntableDriver, target: dict, verb: str) -> None:
        """Make a freshly connected driver the current one (operation lock held): remember it for the next restart
        and turn it while scanning again if that was on."""
        self.driver, self.error = new, None
        self.log(f"{verb} {new.name}" + (f" (firmware {new.firmware})" if new.firmware else ""))
        if new.kind != "simulated":
            self._remember({"id": new.id, "name": new.name, "kind": new.kind})
        self._set_reconnect(target)
        self._restore_spin()

    # ----------------------------------------------------------------- reconnect after a restart
    def start_auto_reconnect(self) -> dict | None:
        """At service start: connect again, in the background, the table that was connected when the service stopped
        (every restart drops the Bluetooth link, and a turntable scan then ran with the table standing still).

        Never blocks: the tries run in a thread, `reconnect_delays_s` of them with growing pauses (1, 5, 15, 40 s;
        a Bluetooth try that finds nothing takes ~10 s more). Quiet: a table that is off, out of range or held by
        Revo Metro on another computer is only logged and shown in ``status()["auto_reconnect"]`` (state waiting,
        connecting, connected or gave_up), never raised and never put in ``status()["error"]``. Only the remembered
        table: the practice (simulated) one only when that was what was connected, nothing after the user
        disconnected on purpose. A connect or disconnect by the user stops the tries. Turn while scanning is picked
        up again once connected (see `_restore_spin`). Returns ``status()["auto_reconnect"]``."""
        target = self._reconnect_target()
        with self._op_lock:
            busy = self._auto_thread is not None and self._auto_thread.is_alive()
            if self.closed or target is None or busy or (self.driver is not None and self.driver.connected):
                auto = self._auto
                return dict(auto) if auto else None
            delays = tuple(float(d) for d in self.reconnect_delays_s)
            name = target.get("name") or target["id"]
            self._auto_cancel = cancel = threading.Event()
            self._auto = auto = {"state": "waiting", "device": target["id"], "name": name, "kind": target["kind"],
                                 "attempt": 0, "attempts": len(delays), "error": None,
                                 "message": f"Reconnecting {name} after the restart"}
            snapshot = dict(auto)            # before the thread can move on
            self.log(f"Reconnecting {name} after the restart (up to {len(delays)} tries)")
            self._auto_thread = threading.Thread(target=self._auto_reconnect, args=(target, delays, cancel, auto),
                                                 name="turntable-reconnect", daemon=True)
            self._auto_thread.start()
            return snapshot

    def _auto_reconnect(self, target: dict, delays: tuple, cancel: threading.Event, auto: dict) -> None:
        name, last = auto["name"], None
        try:
            for i, delay in enumerate(delays, 1):
                auto.update(state="waiting", attempt=i)
                if cancel.wait(delay) or self.closed:
                    return self._auto_stopped(auto)
                auto["state"] = "connecting"
                result = self._auto_try(target, cancel, auto)
                if result is None:           # connected, or stopped by the user
                    return
                last = auto["error"] = result
                self.log(f"Reconnect try {i} of {len(delays)} failed: {last}")
                if not _retryable(last):
                    break
        except Exception as exc:  # never die silently with the status stuck on "connecting"
            last = auto["error"] = f"{type(exc).__name__}: {exc}"
            self.log(f"Reconnecting failed: {last}")
        auto.update(state="gave_up", message=f"Could not reconnect {name} after the restart - connect it in Scan → "
                                              f"Turntable when it is free")
        self.log(f"Gave up reconnecting {name} after the restart")

    def _auto_try(self, target: dict, cancel: threading.Event, auto: dict) -> str | None:
        """One reconnect try: None when connected (or stopped by the user), else the error."""
        new = self._driver_for(target)
        try:
            if new.kind == "simulated":
                new.connect()
            else:
                with self._scan_lock:        # finding the table is a Bluetooth scan: never alongside devices()
                    if cancel.is_set() or self.closed:   # the scan lock may have been held for seconds
                        self._auto_stopped(auto)
                        return None
                    new.connect()
        except Exception as exc:  # TurntableError, or a broken Bluetooth stack
            self._quiet_disconnect(new)
            return str(exc) if isinstance(exc, TurntableError) else f"{type(exc).__name__}: {exc}"
        # the Bluetooth connect ran outside the operation lock (it takes seconds); the user may have connected or
        # disconnected meanwhile, and then their choice stands
        with self._op_lock:
            if cancel.is_set() or self.closed or (self.driver is not None and self.driver.connected):
                self._quiet_disconnect(new)
                self._auto_stopped(auto)
                return None
            self._install(new, target, "Reconnected")
            auto.update(state="connected", error=None, message=f"Reconnected {target.get('name') or target['id']} "
                                                               "after the restart")
        return None

    def _auto_stopped(self, auto: dict) -> None:
        auto["state"] = "stopped"
        if not self.closed:
            self.log(f"Stopped reconnecting {auto['name']}: the turntable was connected or disconnected by hand")

    @staticmethod
    def _quiet_disconnect(drv: TurntableDriver) -> None:
        try:
            drv.disconnect()
        except Exception:
            pass

    def _cancel_auto_reconnect(self) -> None:
        """Stop the reconnect after a restart and wait out a try in flight (outside the operation lock, which the try
        needs to finish): two Bluetooth connects to one table at once fail both, and a link left open by a stopping
        service keeps the table from advertising to the next start."""
        self._auto_cancel.set()
        th = self._auto_thread
        if th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout=RECONNECT_JOIN_S)
        self._auto = None

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
        """Disconnect on purpose: the table is not connected again at the next service start (and a scan no longer
        warns that it is missing). The Turn while scanning setting is kept for the next connect."""
        self._cancel_auto_reconnect()
        with self._op_lock:
            if self.program is not None and self.program.running:
                self.program.stop()
            self._end_spin()
            self._disconnect_current()
            self._set_reconnect(None)
            return self.status()

    def shutdown(self) -> None:
        """The service stops: let go of the table but keep what to reconnect and the turning for the next start."""
        self._cancel_auto_reconnect()
        with self._op_lock:
            try:
                if self.program is not None and self.program.running:
                    self.program.stop(wait=5)
                self._end_spin()
                self._disconnect_current()
            finally:
                self.closed = True
                with _MANAGERS_LOCK:
                    if _MANAGERS.get(str(self.root.resolve())) is self:
                        del _MANAGERS[str(self.root.resolve())]

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        """Everything the UI and the assistant show (keys in docs/turntable.md 5.1). After a restart:

        * ``auto_reconnect`` - the reconnect at service start: {state: waiting|connecting|connected|gave_up,
          device, name, kind, attempt, attempts, error, message}, or None (nothing to reconnect, or the user
          connected / disconnected by hand since).
        * ``remembered_spin`` - the Turn while scanning setting kept across restarts {on, follow_scan,
          speed_s_per_rev, direction}, or None; it turns the table again after every (re)connect while ``on``.
        * ``scan_warning`` - see `scan_warning`: one plain sentence while a scan runs and the table that should turn
          it is not connected or not turning, else None (meant for the capture guidance banner).
        """
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
            "spin": ({k: v for k, v in self._spin.items() if k != "stop"} if self._spin is not None else None),
            "error": self.error or st.get("error") or (move or {}).get("error"),
            "capabilities": drv.capabilities() if drv is not None else None,
            "validated": bool(drv.validated) if drv is not None else False,
            "firmware": getattr(drv, "firmware", None),
            "last_move": move,
            "device_state": {k: v for k, v in st.items() if k not in ("angle_deg", "tilt_deg", "speed_s_per_rev",
                                                                       "moving", "error")},
            "bluetooth": dict(self._bluetooth, installed=ble.bleak_available()[0]),
            "remembered_device": self._remembered(),
            "auto_reconnect": dict(auto) if (auto := self._auto) else None,
            "remembered_spin": dict(self._saved_spin()) or None,
            "scan_warning": self.scan_warning(),
            "log": list(self._log_lines)[-20:],
        }

    def scan_warning(self, scan_state=_ASK) -> str | None:
        """``status()["scan_warning"]``: one plain sentence while a scan runs but the turntable that should turn it
        does not - the part then shows the scanner the same side frame after frame, the frames pile up on one spot
        and the scan barely grows (the "lines only" scans after a service restart dropped the Bluetooth link).

        The table should turn when Turn while scanning is on (live, or kept from before a restart) and the table is
        in use (connected, or to be reconnected: not disconnected on purpose). Warns when it is not connected
        (`NOT_CONNECTED_WARNING`, or `RECONNECTING_WARNING` while the reconnect after a restart still tries) or
        connected but still for more than `NOT_TURNING_GRACE_S` (`NOT_TURNING_WARNING`, or `STALLED_WARNING` when
        the turning runs but the table stands). None otherwise: no scan running, turning off, a program drives the
        table (it holds the capture while it moves), or it turns.

        `scan_state` is the capture's state ("running", ...) when the caller knows it - the capture session must pass
        its own (asking would call back into it); left out, it is read from the workspace's capture manager."""
        saved = self._saved_spin()
        drv = self.driver
        connected = drv is not None and drv.connected
        if (self._spin is None and not saved.get("on")) or self._program_running() \
                or (not connected and self._reconnect_target() is None):
            self._not_turning_since = None
            return None
        state = self._scan_state() if scan_state is _ASK else scan_state
        if state != "running":
            self._not_turning_since = None
            return None
        if not connected:
            auto = self._auto
            trying = auto is not None and auto["state"] in ("waiting", "connecting")
            return RECONNECTING_WARNING if trying else NOT_CONNECTED_WARNING
        try:
            st = drv.state()
        except Exception:
            st = {}
        if st.get("continuous") or st.get("rotating"):
            self._not_turning_since = None
            return None
        now = time.monotonic()
        if self._not_turning_since is None:
            self._not_turning_since = now
        if now - self._not_turning_since < NOT_TURNING_GRACE_S:
            return None
        # Turn while scanning on but not running here (a restore that failed): Start turning shows; it is running
        # but the table stands (refused, or stopped by the firmware): the page shows Stop turning instead
        return NOT_TURNING_WARNING if self._spin is None else STALLED_WARNING

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
        """Stop everything now: a running program, turning while scanning and any motion. Turning while scanning
        stays off until started again (also after a reconnect or restart)."""
        with self._spin_lock:                # first: a spin being started or restored right now sees it and backs out
            self._spin_gen += 1
            spin, self._spin = self._spin, None
            was_on = bool(self._saved_spin().get("on"))
            if was_on:
                self._spin_saved = {**self._spin_saved, "on": False}
        if spin is not None:
            spin["stop"].set()
            self.log("Stopped turning")
        try:
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
        finally:
            if was_on:
                self._persist_spin()         # after the motion: an emergency stop never waits on the disk
        if self._move_thread is not None:
            self._move_thread.join(timeout=10)
        return self.status()

    # ----------------------------------------------------------------- turning while scanning
    def spin(self, on: bool = True, follow_scan: bool = True, speed_s_per_rev: float | None = None,
             direction: str | None = None) -> dict:
        """Turn continuously until stopped (on=False, stop() or disconnect) - how a laser-line scanner scans a part
        all the way round. With follow_scan the table holds while the live scan is paused or stopped and turns again
        as soon as it runs; it starts turning at once whatever the scan is doing.

        The setting is kept (settings "turntable.spin") and picked up again after a reconnect or a service restart
        (see `_restore_spin`); on=False (and stop()) turns it off for good, also when no table is connected."""
        with self._op_lock:
            if not on:
                self._end_spin("Stopped turning")
                self._spin_off_saved()
                return self.status()
            if self._spin is not None:
                return self.status()
            self._start_spin(follow_scan, speed_s_per_rev, direction)
            return self.status()

    def _start_spin(self, follow_scan: bool, speed_s_per_rev: float | None, direction: str | None,
                    restored: bool = False, gen: int | None = None) -> None:
        """Start turning (operation lock held) and keep the setting. A `restored` spin waits for the scan to run
        before it turns, as if the scan had paused it: a restart or reconnect never sets the table turning on its own
        while nobody scans. A Stop that comes in meanwhile (`gen` changed; stop() takes no operation lock) wins."""
        gen = self._spin_gen if gen is None else gen
        drv = self._require_idle()
        caps = drv.capabilities()
        if not caps.get("continuous"):
            raise TurntableError(f"The {drv.name} cannot turn continuously")
        if direction is not None and direction not in ("cw", "ccw"):
            raise TurntableError("direction must be 'cw' or 'ccw'")
        speed = self._check_speed(speed_s_per_rev, caps)[0] if speed_s_per_rev is not None else None
        if direction is not None:
            self.direction = direction
        if speed is not None:
            drv.set_speed(speed)
        state = self._scan_state()
        hold = restored and state != "running"
        if not hold:
            drv.rotate_continuous(self.direction == "cw")
        if speed is None:                    # keep the speed it turns at now, so a restart turns it the same
            known = drv.state().get("speed_s_per_rev")
            speed = int(round(known)) if isinstance(known, (int, float)) else None
        stop = threading.Event()
        spin = {"follow_scan": bool(follow_scan), "turning": not hold, "held_by_scan": hold, "restored": restored,
                "since": datetime.now().isoformat(timespec="seconds"), "stop": stop}
        with self._spin_lock:
            started = self._spin_gen == gen
            if started:
                self._spin = spin
                self._spin_saved = {"on": True, "follow_scan": bool(follow_scan), "speed_s_per_rev": speed,
                                    "direction": self.direction}
                self._save(spin=dict(self._spin_saved))
        if not started:
            if not hold:
                try:
                    drv.stop()
                except TurntableError as exc:
                    self.error = str(exc)
            self.log("Stopped before it began turning")
            return
        if restored:
            self.log("Turn while scanning is on again: " + ("holding until the scan runs" if hold
                                                            else f"turning {self.direction}"))
        else:
            self.log(f"Turning {self.direction}" + (" while scanning" if follow_scan else "") + " until stopped")
        if follow_scan or hold:
            threading.Thread(target=self._follow_scan, args=(stop, state), name="turntable-spin", daemon=True).start()

    def _restore_spin(self) -> None:
        """After a (re)connect (operation lock held): turn while scanning again if it was on when the link or the
        service went down - holding until the scan runs (without follow_scan it then turns until stopped). A saved
        speed this table cannot do is left out (it keeps its own)."""
        gen = self._spin_gen                 # a Stop from here on wins over the restore
        saved = self._saved_spin()
        drv = self.driver
        if not saved.get("on") or self._spin is not None or drv is None or not drv.connected:
            return
        speed = saved.get("speed_s_per_rev")
        lo, hi = drv.capabilities().get("speed_range") or [25, 90]
        if not isinstance(speed, (int, float)) or not lo <= speed <= hi:
            speed = None
        try:
            self._start_spin(bool(saved.get("follow_scan", True)), speed, saved.get("direction"), restored=True,
                             gen=gen)
        except Exception as exc:  # a failed restore must not fail the (re)connect; scan_warning tells the user
            self.log(f"Could not turn while scanning again: {exc}")

    def _end_spin(self, msg: str | None = None) -> None:
        with self._spin_lock:
            self._spin_gen += 1
            spin, self._spin = self._spin, None
        if spin is None:
            return
        spin["stop"].set()
        drv = self.driver
        if spin.get("turning") and drv is not None and drv.connected:
            try:
                drv.stop()
            except TurntableError as exc:
                self.error = str(exc)
        if msg:
            self.log(msg)

    def _scan_state(self) -> str | None:
        # the capture side may ask our status while building its own (the guidance banner shows scan_warning):
        # a nested call answers None instead of asking the capture manager again, round and round
        if getattr(_IN_SCAN_STATE, "busy", False):
            return None
        _IN_SCAN_STATE.busy = True
        try:
            cm = self._capture_manager()
            if cm is None or getattr(cm, "session", None) is None:
                return None
            return cm.status().get("state")
        except Exception:
            return None
        finally:
            _IN_SCAN_STATE.busy = False

    def _follow_scan(self, stop: threading.Event, last: str | None) -> None:
        """While the scan runs a held table turns - checked every 0.2 s, not only when the scan changes, so a scan
        that started just before a restored spin is caught too; a failed turn is tried again after `FOLLOW_RETRY_S`.
        With follow_scan the table holds when the scan stops running (on that change only, so the table also turns
        before a scan starts). Only our own hold is undone: a table the firmware stopped is not restarted here."""
        retry_at = 0.0
        while not stop.wait(0.2):
            state = self._scan_state()
            spin = self._spin
            if spin is None:
                return
            turn = state == "running" and not spin["turning"] and time.monotonic() >= retry_at
            hold = state != "running" and last == "running" and spin["turning"] and spin["follow_scan"]
            last = state
            if not (turn or hold):
                continue
            with self._op_lock:
                spin, drv = self._spin, self.driver
                if stop.is_set() or spin is None or drv is None or not drv.connected:
                    return
                try:
                    if turn and not spin["turning"]:
                        drv.rotate_continuous(self.direction == "cw")
                        if stop.is_set() or self._spin is not spin:     # Stop came in meanwhile: it wins
                            drv.stop()
                            return
                        spin.update(turning=True, held_by_scan=False)
                        self.log("Scan running: turning")
                    elif hold and spin["turning"]:
                        drv.stop()
                        spin.update(turning=False, held_by_scan=True)
                        self.log(f"Scan {state or 'closed'}: holding the table")
                except TurntableError as exc:
                    self.error = str(exc)
                    self.log(f"Turning while scanning: {exc}")
                    retry_at = time.monotonic() + FOLLOW_RETRY_S

    # ----------------------------------------------------------------- programs
    def start_program(self, program: dict) -> dict:
        """{interval_deg, frames_per_stop, direction, speed_s_per_rev, rotations: [{tilt_deg}], sync_scan, ...}
        (see program.py)."""
        with self._op_lock:
            drv = self._require_idle()
            program = dict(program or {})
            # stopping at each stop gives a laser-line scanner a few lines per stop, never a surface (the user's
            # first turntable scans: 756 points); it records while the table turns all the way round instead
            swept = (program.get("sync_scan", True) is not False and program.get("mode") != "continuous"
                     and self._capture_sweeps())
            if swept:
                program["mode"] = "continuous"
            plan = validate_program(program, drv.capabilities())
            if swept:
                plan["notes"].append("The scanner records laser lines, so the table turns all the way round while "
                                     "it scans (stopping would record only a few lines at each stop).")
            self.direction = plan["direction"]
            runner = ProgramRunner(drv, plan, capture=self._capture_manager, log=lambda m: self.log(f"Program: {m}"))
            self.program = runner
            each = ("one continuous sweep" if plan["mode"] == "continuous" else
                    f"{plan['stops_per_rotation']} stop(s) of {plan['interval_deg']} deg, "
                    f"{plan['frames_per_stop']} frame(s) per stop")
            self.log(f"Program: {len(plan['rotations'])} rotation(s), {each}, {plan['direction']}, "
                     f"sync_scan={plan['sync_scan']}")
            runner.start()
            return self.status()

    def stop_program(self) -> dict:
        with self._op_lock:
            if self.program is not None and self.program.running:
                self.program.stop()
                self.log("Program stopped")
            return self.status()
