"""Turntable control (Contract 4 in docs/v3-plan.md; protocol and usage in docs/turntable.md).

GET  /api/turntable/status                  -> status (see TurntableManager.status)
GET  /api/turntable/devices?scan_seconds=4  -> [{id, name, kind, rssi, connected, remembered}]
                                               ("simulated" always first; scan_seconds=0 skips the Bluetooth scan)
POST /api/turntable/connect     {device?, kind?: auto|dual_axis|large|simulated, options?: {time_scale}}
POST /api/turntable/disconnect
POST /api/turntable/rotate      {degrees, speed_s_per_rev?, wait?: false}   relative, + = clockwise seen from above
POST /api/turntable/tilt        {degrees, wait?: false}                     absolute, dual-axis only
POST /api/turntable/stop                                                    stops a program and all motion
POST /api/turntable/speed       {s_per_rev}
POST /api/turntable/program     {mode?, interval_deg, frames_per_stop, direction, speed_s_per_rev?,
                                 rotations: [{tilt_deg}], sync_scan, dwell_s?, settle_s?, level_at_end?,
                                 capture_timeout_s?}
POST /api/turntable/program/stop
POST /api/turntable/spin        {on: true|false, follow_scan?: true, speed_s_per_rev?, direction?: cw|ccw}
                                turn continuously until stopped; with follow_scan the table holds while the live
                                scan is paused or stopped and turns again when it runs (status `spin`)

Every response is the status dict (rotate/tilt add `move`, speed adds `speed`). Errors: HTTP 400 with one sentence.
At startup the table that was connected when the service stopped is connected again in the background, and Turn
while scanning picked up again (status `auto_reconnect`, `remembered_spin`; `scan_warning` while a scan runs without
the table turning).
Handlers are plain `def`s: FastAPI runs them in its thread pool, so Bluetooth scans and waits never block the
event loop.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..capture.turntable import TurntableError, get_turntable_manager


class ConnectReq(BaseModel):
    device: str | None = None
    kind: str = "auto"
    options: dict | None = None


class RotateReq(BaseModel):
    degrees: float
    speed_s_per_rev: float | None = None
    wait: bool = False


class TiltReq(BaseModel):
    degrees: float
    wait: bool = False


class SpeedReq(BaseModel):
    s_per_rev: float


class SpinReq(BaseModel):
    on: bool = True
    follow_scan: bool = True
    speed_s_per_rev: float | None = None
    direction: str | None = None


def create_router(workspace, jobs) -> APIRouter:
    manager = get_turntable_manager(workspace)

    @asynccontextmanager
    async def lifespan(app):
        # a restart dropped the Bluetooth link (the Spark restarts itself after every update): connect the table it
        # had again and pick up Turn while scanning - in a background thread, startup never waits on Bluetooth
        manager.start_auto_reconnect()
        yield
        await asyncio.to_thread(manager.shutdown)

    router = APIRouter(prefix="/api/turntable", lifespan=lifespan)

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TurntableError as exc:
            raise HTTPException(400, str(exc)) from None

    @router.get("/status")
    def status():
        return manager.status()

    @router.get("/devices")
    def devices(scan_seconds: float = 4.0):
        return call(manager.devices, scan_seconds)

    @router.post("/connect")
    def connect(req: ConnectReq | None = None):
        req = req or ConnectReq()
        return call(manager.connect, req.device, req.kind, req.options)

    @router.post("/disconnect")
    def disconnect():
        return call(manager.disconnect)

    @router.post("/rotate")
    def rotate(req: RotateReq):
        return call(manager.rotate, req.degrees, req.speed_s_per_rev, req.wait)

    @router.post("/tilt")
    def tilt(req: TiltReq):
        return call(manager.tilt, req.degrees, req.wait)

    @router.post("/stop")
    def stop():
        return call(manager.stop)

    @router.post("/speed")
    def speed(req: SpeedReq):
        return call(manager.set_speed, req.s_per_rev)

    @router.post("/program")
    def program(req: dict):
        return call(manager.start_program, req)

    @router.post("/program/stop")
    def program_stop():
        return call(manager.stop_program)

    @router.post("/spin")
    def spin(req: SpinReq):
        return call(manager.spin, req.on, req.follow_scan, req.speed_s_per_rev, req.direction)

    return router
