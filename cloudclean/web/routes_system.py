"""Server health for the UI (Revo Metro's "PC performance check" and disk-space warnings, for the machine that runs
CloudClean — usually the DGX Spark).

GET /api/system -> {host, os, python, cpu_count, memory: {total_gb, available_gb}, disk: {path, total_gb, free_gb,
                    low}, gpus: [{name, memory_total_mb, memory_used_mb, utilization_pct, temperature_c}], version}
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

from fastapi import APIRouter


def _memory() -> dict:
    try:
        if sys.platform.startswith("linux"):
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = int(v.split()[0]) * 1024
            return {"total_gb": round(info["MemTotal"] / 1e9, 1), "available_gb": round(info.get("MemAvailable", 0) / 1e9, 1)}
        if sys.platform == "win32":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = MEMSTAT()
            m.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return {"total_gb": round(m.ullTotalPhys / 1e9, 1), "available_gb": round(m.ullAvailPhys / 1e9, 1)}
    except Exception:  # pragma: no cover - best effort
        pass
    return {"total_gb": None, "available_gb": None}


def _gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None  # unified-memory GPUs report [N/A] for memory

        gpus.append({"name": parts[0], "memory_total_mb": num(parts[1]), "memory_used_mb": num(parts[2]),
                     "utilization_pct": num(parts[3]), "temperature_c": num(parts[4])})
    return gpus


def create_router(workspace, jobs) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/system")
    def system():
        usage = shutil.disk_usage(workspace.root)
        free_gb = usage.free / 1e9
        return {
            "host": platform.node(),
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory": _memory(),
            "disk": {"path": str(workspace.root), "total_gb": round(usage.total / 1e9, 1), "free_gb": round(free_gb, 1),
                     "low": free_gb < 10},
            "gpus": _gpus(),
        }

    return router
