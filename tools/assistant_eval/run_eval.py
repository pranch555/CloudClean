#!/usr/bin/env python
"""Evaluation harness for the CloudClean chat assistant: scripted conversations through the real chat API of a local
CloudClean server talking to the real LLM, then machine checks of the workspace.

  python tools/assistant_eval/run_eval.py --code <folder that contains the cloudclean package> --runs 2
  python tools/assistant_eval/run_eval.py --code . --runs 2 --compare <BEFORE summary.json>
  python tools/assistant_eval/run_eval.py --code . --selftest        (no LLM: oracle actions through the REST API)
  python tools/assistant_eval/run_eval.py --from-logs <out1>,<out2> --out <dir>   (one summary of several runs)

--scenarios A,B,... picks scenarios (A-H as requested, I = the user's failure reproduced faithfully); 9 scenarios x
--runs 2 = 18 LLM conversations. Assets are generated once, deterministically (byte-identical on every rebuild),
under %TEMP%/cloudclean_assistant_eval/assets-v1 unless --assets names a folder.

Per scenario (scenarios.py) and run:
  1. start `python -m cloudclean.cli serve --port 8790 --workspace <fresh> --no-browser --no-accounts` with
     PYTHONPATH=<code> (cwd outside the repo; the imported cloudclean is verified first) and CLOUDCLEAN_LLM_* set,
  2. create the project, import the synthetic assets (POST /api/import-paths), set the assistant settings,
  3. send each user turn to POST /api/assistant/chat, recording every server-sent event,
  4. check the outcome (checks.py: holes of the new meshes by geometry, answer text, created assets),
  5. stop the server (kill by PID, whole process tree).
The SSH tunnel to the model (ssh -N -L 18001:localhost:8001 spark) is started once if the port does not answer yet
and killed by PID at the end. One conversation at a time, one request at a time.

Writes to --out: <scenario>_run<k>.json (events, tool calls, stored conversation, check), server logs,
summary.md / summary.json and conversations.log (one line per LLM conversation).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _venv_python() -> Path | None:
    for p in (REPO / ".venv" / "Scripts" / "python.exe", REPO / ".venv" / "bin" / "python"):
        if p.exists():
            return p
    return None


def _ensure_deps() -> None:
    """Re-run with the repo's venv when this interpreter lacks the packages (open3d, httpx ...)."""
    try:
        import httpx  # noqa: F401
        import numpy  # noqa: F401
        import open3d  # noqa: F401
        import PIL  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        py = _venv_python()
        if py is not None and Path(sys.executable).resolve() != py.resolve():
            sys.exit(subprocess.call([str(py), *sys.argv]))
        raise


_ensure_deps()
import httpx  # noqa: E402

sys.path.insert(0, str(HERE))
import scenarios as S  # noqa: E402
import synth_assets  # noqa: E402
from checks import load_mesh  # noqa: E402

DEFAULT_LLM = os.environ.get("CLOUDCLEAN_LLM_BASE_URL") or "http://localhost:18001/v1"
DEFAULT_MODEL = os.environ.get("CLOUDCLEAN_LLM_MODEL") or "RadixArk/Qwen3.8-27B-NVFP4"
DEFAULT_KEY = os.environ.get("CLOUDCLEAN_LLM_API_KEY") or ""
OVERFLOW_RE = re.compile(r"context (window|length)|too long for the model|maximum context", re.I)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- processes
def kill_tree(pid: int) -> None:
    """Stop a process and its children, by PID."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(1.0)
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def llm_models(base: str, key: str, timeout: float = 10.0) -> dict | None:
    try:
        r = httpx.get(base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


class Tunnel:
    """ssh -N -L <local>:localhost:<remote> <host>, started only if the local port does not answer yet."""

    def __init__(self, host: str, local_port: int, remote_port: int, base: str, key: str, log_path: Path):
        self.host, self.local, self.remote, self.base, self.key = host, local_port, remote_port, base, key
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    def ensure(self) -> dict:
        info = llm_models(self.base, self.key)
        if info is not None:
            log(f"LLM already reachable at {self.base} (tunnel not started by this run)")
            return info
        if port_open(self.local):
            raise RuntimeError(f"port {self.local} is in use but {self.base}/models does not answer")
        cmd = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes",
               "-L", f"{self.local}:localhost:{self.remote}", self.host]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with open(self.log_path, "ab") as err_log:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err_log,
                                         creationflags=flags)
        log(f"started tunnel pid {self.proc.pid}: {' '.join(cmd)}")
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                err = self.log_path.read_bytes()[-500:].decode("utf-8", "replace")
                raise RuntimeError(f"ssh tunnel exited (code {self.proc.returncode}): {err}")
            info = llm_models(self.base, self.key, timeout=5)
            if info is not None:
                return info
            time.sleep(1.0)
        raise RuntimeError(f"the LLM does not answer at {self.base} through the tunnel")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            kill_tree(self.proc.pid)
            log(f"stopped tunnel pid {self.proc.pid}")
        self.proc = None


class Server:
    def __init__(self, python: str, code: Path, workspace: Path, port: int, env: dict, log_path: Path):
        self.python, self.code, self.ws, self.port, self.env, self.log_path = python, code, workspace, port, env, log_path
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}"

    def start(self, timeout: float = 180.0) -> None:
        if port_open(self.port):
            raise RuntimeError(f"port {self.port} is already in use - stop what runs there or pass --port")
        self.ws.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        cmd = [self.python, "-m", "cloudclean.cli", "serve", "--port", str(self.port), "--workspace", str(self.ws),
               "--no-browser", "--no-accounts"]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, cwd=str(self.ws.parent), env=self.env, stdin=subprocess.DEVNULL,
                                     stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags,
                                     start_new_session=os.name != "nt")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited (code {self.proc.returncode}): {self.log_tail()}")
            try:
                if httpx.get(self.base + "/api/assistant/settings", timeout=3).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"the server did not start within {timeout:.0f} s: {self.log_tail()}")

    def log_tail(self, n: int = 1500) -> str:
        try:
            return self.log_path.read_bytes()[-n:].decode("utf-8", "replace")
        except OSError:
            return ""

    def stop(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                kill_tree(self.proc.pid)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
            self.proc = None
        if getattr(self, "_log", None):
            self._log.close()
            self._log = None
        deadline = time.monotonic() + 15
        while port_open(self.port) and time.monotonic() < deadline:
            time.sleep(0.3)


# --------------------------------------------------------------------------- REST helper
class Rest:
    def __init__(self, base: str):
        self.base = base
        self.client = httpx.Client(base_url=base, timeout=120)

    def get(self, path: str, **kw):
        r = self.client.get(path, **kw)
        r.raise_for_status()
        return r.json()

    def send(self, method: str, path: str, body: dict):
        r = self.client.request(method, path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def wait_jobs(self, job_ids: list[str], timeout: float = 600.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        while True:
            jobs = {j["id"]: j for j in self.get("/api/jobs")}
            mine = [jobs.get(i) for i in job_ids]
            if all(j and j["status"] in ("done", "failed", "cancelled") for j in mine):
                bad = [j for j in mine if j["status"] != "done"]
                if bad:
                    raise RuntimeError(f"job {bad[0]['title']} {bad[0]['status']}: {bad[0].get('error')}")
                return mine
            if time.monotonic() > deadline:
                raise RuntimeError("jobs did not finish in time")
            time.sleep(0.5)

    def job(self, path: str, body: dict) -> list[str]:
        """POST a job-starting endpoint, wait, return the created asset ids."""
        job = self.send("POST", path, body)
        return self.wait_jobs([job["id"]])[0].get("result") or []

    def close(self):
        self.client.close()


# --------------------------------------------------------------------------- setup
def base_context(project_id: str, active: str | None, visible: list[str]) -> dict:
    """What the browser sends as `context` (frontend/src/lib/applyUi.ts uiContext)."""
    return {"active_id": active, "selected_ids": [], "visible_ids": visible, "units": "mm", "project_id": project_id,
            "screen": "workspace", "step": "mesh", "tool": "navigate", "theme": "carbon",
            "camera": {"position": [215.0, -310.0, 250.0], "target": [0.0, 0.0, 6.0], "up": [0.0, 0.0, 1.0],
                       "fov": 45, "projection": "perspective"},
            "section": {"enabled": False}, "measurements": []}


def import_assets(rest: Rest, paths: list[str], project_id: str) -> list[str]:
    res = rest.send("POST", "/api/import-paths", {"paths": paths, "project_id": project_id})
    jobs = rest.wait_jobs([j["id"] for j in res["jobs"]])
    return [j["result"][0] for j in jobs]


def setup_scenario(rest: Rest, scn: S.Scenario, assets: dict) -> tuple[str, dict]:
    for name, keys in scn.extra_projects:
        pid = rest.send("POST", "/api/projects", {"name": name})["id"]
        import_assets(rest, [assets[k] for k in keys], pid)
    pid = rest.send("POST", "/api/projects", {"name": scn.project})["id"]
    ids = dict(zip(scn.imports, import_assets(rest, [assets[k] for k in scn.imports], pid)))
    if scn.golden:
        rest.send("PATCH", f"/api/projects/{pid}", {"golden_asset_id": ids[scn.golden]})
    return pid, ids


def apply_ui(context: dict, events: list[dict], known: set[str]) -> None:
    """Mirror what the browser does with the assistant's ui events (show/focus/select/project/navigate)."""
    for e in events:
        action = e.get("action")
        if action == "show":
            ids = [i for i in e.get("asset_ids") or [] if i in known]
            context["visible_ids"] = ids if e.get("exclusive", True) is not False else \
                list(dict.fromkeys([*context.get("visible_ids", []), *ids]))
            if ids:
                context["active_id"] = ids[-1]
            context["screen"] = "workspace"
        elif action == "focus" and e.get("asset_id") in known:
            context["active_id"] = e["asset_id"]
        elif action == "select":
            context["selected_ids"] = [i for i in e.get("asset_ids") or [] if i in known]
        elif action == "project" and e.get("project_id"):
            context["project_id"] = e["project_id"]
        elif action == "navigate":
            if e.get("step"):
                context["step"] = e["step"]
            if e.get("screen"):
                context["screen"] = e["screen"]


# --------------------------------------------------------------------------- one chat turn (SSE)
def chat_turn(base: str, conv_id: str, message: str, context: dict, images: list[str], notes: list[str],
              timeout: float, stop_grace: float = 120.0) -> dict:
    body: dict = {"conversation_id": conv_id, "message": message, "context": context}
    if images:
        body["images"], body["image_notes"] = images, notes
    rec: dict = {"message": message, "images": len(images), "text": "", "reasoning_chars": 0, "tool_calls": [],
                 "ui_events": [], "errors": [], "other_events": [], "timeline": [], "done": None,
                 "http_status": None, "stopped_by_harness": False, "pings": 0}
    calls: dict[str, dict] = {}
    t0 = time.monotonic()
    deadline, hard_stop = t0 + timeout, None

    def handle(event: str, raw: str) -> bool:
        try:
            d = json.loads(raw) if raw else {}
        except ValueError:
            d = {"raw": raw[:500]}
        t = round(time.monotonic() - t0, 2)
        if event == "delta":
            rec["text"] += d.get("text", "")
        elif event == "reasoning":
            rec["reasoning_chars"] += len(d.get("text", ""))
        elif event == "tool_start":
            c = {"call_id": d.get("call_id"), "name": d.get("name"), "arguments": d.get("arguments"), "t_start": t}
            calls[d.get("call_id")] = c
            rec["tool_calls"].append(c)
            rec["timeline"].append([t, "tool_start", d.get("name")])
        elif event == "tool_progress":
            c = calls.get(d.get("call_id"))
            if c is not None:
                c["progress_events"] = c.get("progress_events", 0) + 1
                c["last_progress"] = {k: d.get(k) for k in ("status", "progress", "label")}
        elif event == "tool_end":
            c = calls.get(d.get("call_id"))
            if c is None:
                c = {"call_id": d.get("call_id"), "name": None, "arguments": None}
                rec["tool_calls"].append(c)
            c.update(ok=d.get("ok"), summary=d.get("summary"), asset_ids=d.get("asset_ids"), t_end=t)
            rec["timeline"].append([t, "tool_end", f"{c.get('name')}: {'ok' if d.get('ok') else 'FAILED'} "
                                                   f"{str(d.get('summary'))[:160]}"])
        elif event == "ui":
            rec["ui_events"].append({"t": t, **{k: v for k, v in d.items() if k != "images"}})
        elif event == "error":
            rec["errors"].append(str(d.get("message")))
            rec["timeline"].append([t, "error", str(d.get("message"))[:300]])
        elif event == "done":
            rec["done"] = d
            rec["timeline"].append([t, "done", json.dumps(d)])
            return True
        elif event == "conversation":
            rec["conversation"] = d
        else:
            rec["other_events"].append({"t": t, "event": event, "data": d})
            rec["timeline"].append([t, event, json.dumps(d)[:200]])
        return False

    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0)) as client:
            with client.stream("POST", f"{base}/api/assistant/chat", json=body) as resp:
                rec["http_status"] = resp.status_code
                if resp.status_code != 200:
                    rec["errors"].append(f"HTTP {resp.status_code}: {resp.read().decode('utf-8', 'replace')[:500]}")
                else:
                    event, data = None, []
                    for line in resp.iter_lines():
                        now = time.monotonic()
                        if hard_stop is None and now > deadline:
                            rec["stopped_by_harness"] = True
                            hard_stop = now + stop_grace
                            try:
                                httpx.post(f"{base}/api/assistant/stop", json={"conversation_id": conv_id}, timeout=10)
                            except httpx.HTTPError:
                                pass
                        if hard_stop is not None and now > hard_stop:
                            rec["errors"].append("harness: gave up waiting after stop")
                            break
                        if line.startswith(":"):
                            rec["pings"] += 1
                        elif line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            data.append(line[5:].lstrip())
                        elif line == "" and event is not None:
                            finished = handle(event, "\n".join(data))
                            event, data = None, []
                            if finished:
                                break
    except httpx.HTTPError as exc:
        rec["errors"].append(f"harness: {type(exc).__name__}: {exc}")
        try:
            httpx.post(f"{base}/api/assistant/stop", json={"conversation_id": conv_id}, timeout=10)
        except httpx.HTTPError:
            pass
    rec["seconds"] = round(time.monotonic() - t0, 1)
    return rec


# --------------------------------------------------------------------------- outcome + metrics
class Outcome:
    """What a check sees (scenarios.py)."""

    def __init__(self, ws: Path, rest: Rest, truth: dict, ids: dict, before: set, after: list, turns: list):
        self.ws, self.rest, self.truth, self.assets, self.turns = ws, rest, truth, ids, turns
        self.by_id = {m["id"]: m for m in after}
        self.new_assets = [m for m in after if m["id"] not in before]
        self.final_text = turns[-1]["text"] if turns else ""
        self.tool_calls = [c for t in turns for c in t.get("tool_calls", [])]

    def descends(self, asset_id: str, root: str) -> bool:
        seen, todo = set(), [asset_id]
        while todo:
            a = todo.pop()
            if a == root:
                return True
            if a in seen or a not in self.by_id:
                continue
            seen.add(a)
            todo += list(self.by_id[a].get("parents") or [])
        return False

    def load_mesh(self, asset_id: str):
        path = self.ws / "assets" / asset_id / "data.ply"
        if not path.exists():  # storage changed: fetch it through the API instead
            r = self.rest.client.get(f"/api/assets/{asset_id}/download", params={"format": "ply"})
            r.raise_for_status()
            path = Path(tempfile.mkdtemp()) / "mesh.ply"
            path.write_bytes(r.content)
        return load_mesh(path)


def _canon(args) -> str:
    return json.dumps(args, sort_keys=True, default=str) if not isinstance(args, str) else args


def _call_label(c: dict) -> str:
    a = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
    sub = a.get("action") or a.get("kind") or a.get("operation") or ""
    return f"{c.get('name')}{'.' + str(sub) if sub else ''}" + ("" if c.get("ok", True) else "!")


def compact_calls(calls: list[dict]) -> str:
    out: list[list] = []
    for c in calls:
        label = _call_label(c)
        if out and out[-1][0] == label:
            out[-1][1] += 1
        else:
            out.append([label, 1])
    return " > ".join(f"{label}x{n}" if n > 1 else label for label, n in out) or "-"


def conversation_steps(conv: dict | None) -> tuple[list[int], list[int]]:
    """LLM responses and tool-result characters per user turn, from the stored conversation."""
    steps, chars = [], []
    for m in (conv or {}).get("messages", []):
        if m.get("role") == "user":
            steps.append(0)
            chars.append(0)
        elif steps and m.get("role") == "assistant":
            steps[-1] += 1
        elif chars and m.get("role") == "tool":
            chars[-1] += len(m.get("content") or "")
    return steps, chars


def last_message(conv: dict | None) -> str:
    """Content of the last assistant message of the conversation (what the model ended with)."""
    for m in reversed((conv or {}).get("messages", [])):
        if m.get("role") == "user":
            return ""
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return m["content"]
    return ""


def metrics(turns: list[dict], conv: dict | None) -> dict:
    calls = [c for t in turns for c in t["tool_calls"]]
    repeats = 0
    for t in turns:
        counts = Counter((c.get("name"), _canon(c.get("arguments"))) for c in t["tool_calls"])
        repeats += sum(n - 1 for n in counts.values())
    errors = [e for t in turns for e in t["errors"]]
    steps, chars = conversation_steps(conv)
    return {"tool_calls": len(calls), "failed_tool_calls": sum(1 for c in calls if c.get("ok") is False),
            "repeated_calls": repeats, "overflow": any(OVERFLOW_RE.search(e) for e in errors),
            "errors": errors, "llm_steps": sum(steps), "steps_per_turn": steps, "tool_result_chars": chars,
            "seconds": round(sum(t["seconds"] for t in turns), 1), "calls": compact_calls(calls),
            "stopped_by_harness": any(t["stopped_by_harness"] for t in turns),
            "final": turns[-1]["text"] if turns else "", "final_last_message": last_message(conv)}


# --------------------------------------------------------------------------- one scenario run
def run_scenario(scn: S.Scenario, run: int, args, assets: dict, env: dict, out: Path, selftest: bool,
                 ledger: Path) -> dict:
    tag = f"{scn.id}_run{run}"
    ws = out / "ws" / tag
    server = Server(args.python, args.code, ws, args.port, env, out / f"{tag}_server.log")
    result: dict = {"scenario": scn.id, "title": scn.title, "run": run, "selftest": selftest,
                    "started": datetime.now().isoformat(timespec="seconds"), "workspace": str(ws)}
    rest = None
    try:
        server.start()
        rest = Rest(server.base)
        if not selftest:
            rest.send("PUT", "/api/assistant/settings", args.settings)
        result["assistant_settings"] = rest.get("/api/assistant/settings")
        pid, ids = setup_scenario(rest, scn, assets)
        result["setup"] = {"project_id": pid, "assets": ids}
        before = {m["id"] for m in rest.get("/api/assets")}
        active = ids.get(scn.active) if scn.active else None
        visible = [ids[k] for k in scn.visible] if scn.visible else ([active] if active else [])
        context = base_context(pid, active, visible)
        turns: list[dict] = []
        result["turns"] = turns          # filled in place, so a crash keeps the turns so far
        conv = None
        truth = assets[scn.truth]
        if selftest:
            outcome0 = Outcome(ws, rest, truth, ids, before, rest.get("/api/assets"),
                               [{"text": "", "tool_calls": [], "errors": [], "seconds": 0, "stopped_by_harness": False}])
            result["negative_control"] = scn.check(outcome0)
            text = scn.oracle(rest, ids)
            turns.append({"message": "(oracle)", "text": text, "tool_calls": [], "errors": [], "seconds": 0.0,
                          "stopped_by_harness": False, "ui_events": []})
        else:
            status = rest.get("/api/assistant/status")
            result["assistant_status"] = {k: status.get(k) for k in ("reachable", "model", "error", "base_url")}
            if not status.get("reachable"):
                raise RuntimeError(f"the assistant cannot reach the LLM: {status.get('error')}")
            conv_id = f"eval-{scn.id}-{run}-{datetime.now().strftime('%H%M%S')}"
            result["conversation_id"] = conv_id
            with open(ledger, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{tag}\t{conv_id}\t{len(scn.turns)} turn(s)\n")
            for i, turn in enumerate(scn.turns, 1):
                images, notes = [], []
                if turn.image:
                    raw = Path(assets[turn.image]).read_bytes()
                    images = ["data:image/png;base64," + base64.b64encode(raw).decode("ascii")]
                    notes = [turn.image_note]
                log(f"  {tag} turn {i}/{len(scn.turns)}: {turn.message[:70]!r}" + (" [+image]" if images else ""))
                rec = chat_turn(server.base, conv_id, turn.message, json.loads(json.dumps(context)), images, notes,
                                args.turn_timeout)
                rec["context_sent"] = {k: context.get(k) for k in ("project_id", "active_id", "visible_ids")}
                turns.append(rec)
                log(f"    {rec['seconds']}s, {len(rec['tool_calls'])} tool call(s): "
                    f"{compact_calls(rec['tool_calls'])}" + (f", errors: {rec['errors']}" if rec["errors"] else ""))
                known = {m["id"] for m in rest.get("/api/assets")}
                apply_ui(context, rec["ui_events"], known)
            try:
                conv = rest.get(f"/api/assistant/conversations/{conv_id}")
            except httpx.HTTPError as exc:
                conv = {"error": str(exc)}
        after = rest.get("/api/assets")
        outcome = Outcome(ws, rest, truth, ids, before, after, turns)
        try:
            result["check"] = scn.check(outcome)
        except Exception as exc:
            result["check"] = {"success": False, "reason": f"check failed: {type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()}
        for m in outcome.new_assets:
            if m["kind"] == "mesh":
                try:
                    h = rest.get(f"/api/assets/{m['id']}/holes")
                    m["holes_api"] = {k: h.get(k) for k in ("count", "holes_to_fill", "watertight")}
                except Exception:
                    pass
        result["new_assets"] = [{k: m.get(k) for k in ("id", "name", "kind", "operation", "parents", "params",
                                                       "holes_api")} | {"stats": {k: (m.get("stats") or {}).get(k)
                                                                         for k in ("points", "triangles", "watertight")}}
                                for m in outcome.new_assets]
        result["conversation"] = conv
        result["metrics"] = metrics(turns, conv)
    except Exception as exc:
        result["harness_error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result.setdefault("check", {"success": False, "reason": f"harness error: {type(exc).__name__}: {exc}"})
        result.setdefault("metrics", metrics(result.get("turns") or [], None))
        log(f"  {tag}: HARNESS ERROR {exc}")
    finally:
        if rest is not None:
            rest.close()
        server.stop()
    result["success"] = bool(result["check"].get("success"))
    (out / f"{tag}.json").write_text(json.dumps(result, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- summary
def _short(text: str, n: int = 150) -> str:
    t = " ".join((text or "").split()).replace("|", "/")
    return t if len(t) <= n else t[: n - 1] + "…"


def write_summary(results: list[dict], meta: dict, out: Path, extra_md: Path | None, compare: Path | None) -> str:
    lines = [f"# CloudClean assistant eval - {meta['label']}", "",
             f"- code: `{meta['code']}` (cloudclean imported from `{meta['cloudclean_file']}`)",
             f"- model: `{meta.get('model')}` at {meta.get('llm_base')}, context window "
             f"{meta.get('context_window') or '?'} tokens; assistant settings sent: `{json.dumps(meta.get('settings'))}`",
             f"- date: {meta['date']}; runs per scenario: {meta['runs']}; assets: {meta.get('assets', '?')}; "
             f"output: `{out}`",
             "- success = the scenario's machine check (scenarios.py / checks.py); repeats = identical tool calls "
             "(same name and arguments) within one user turn; steps = LLM responses", "",
             "| Scenario | Run | Success | Tool calls | Repeats | Overflow | Steps | Seconds | Final answer (shortened) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["metrics"]
        lines.append(f"| {r['scenario']} | {r['run']} | {'**yes**' if r['success'] else 'no'} | {m['tool_calls']} | "
                     f"{m['repeated_calls']} | {'**yes**' if m['overflow'] else 'no'} | {m['llm_steps']} | "
                     f"{m['seconds']:.0f} | {_short(m['final'])} |")
    lines += ["", "## Per scenario", "", "| Scenario | What | Success | Avg tool calls | Avg repeats | Overflows | "
              "Avg seconds |", "|---|---|---|---|---|---|---|"]
    agg = {}
    for r in results:
        a = agg.setdefault(r["scenario"], {"title": r["title"], "n": 0, "ok": 0, "calls": 0, "rep": 0, "ovf": 0, "s": 0})
        a["n"] += 1
        a["ok"] += r["success"]
        a["calls"] += r["metrics"]["tool_calls"]
        a["rep"] += r["metrics"]["repeated_calls"]
        a["ovf"] += r["metrics"]["overflow"]
        a["s"] += r["metrics"]["seconds"]
    for sid, a in agg.items():
        lines.append(f"| {sid} | {a['title']} | {a['ok']}/{a['n']} | {a['calls'] / a['n']:.1f} | {a['rep'] / a['n']:.1f} "
                     f"| {a['ovf']} | {a['s'] / a['n']:.0f} |")
    total_ok = sum(r["success"] for r in results)
    lines += ["", f"**Total: {total_ok}/{len(results)} succeeded.**", ""]
    if compare and compare.exists():
        prev = json.loads(compare.read_text(encoding="utf-8"))
        pagg: dict = {}
        for r in prev.get("results", []):
            p = pagg.setdefault(r["scenario"], {"n": 0, "ok": 0, "calls": 0, "ovf": 0})
            p["n"] += 1
            p["ok"] += r["success"]
            p["calls"] += r["tool_calls"]
            p["ovf"] += r["overflow"]
        lines += [f"## Compared with `{compare}`", "", "| Scenario | Before success | Now success | Before avg calls | "
                  "Now avg calls | Before overflows | Now overflows |", "|---|---|---|---|---|---|---|"]
        def rate(x):
            return "%d/%d" % (x["ok"], x["n"]) if x else "-"

        def avg_calls(x):
            return "%.1f" % (x["calls"] / x["n"]) if x else "-"

        for sid in sorted(set(pagg) | set(agg)):
            p, a = pagg.get(sid), agg.get(sid)
            lines.append(f"| {sid} | {rate(p)} | {rate(a)} | {avg_calls(p)} | {avg_calls(a)} | "
                         f"{p['ovf'] if p else '-'} | {a['ovf'] if a else '-'} |")
        lines.append("")
    if meta.get("combined_from"):
        lines += ["Combined from: " + ", ".join(f"`{d}`" for d in meta["combined_from"]), ""]
    lines += ["## Details", ""]
    for r in results:
        m = r["metrics"]
        lines.append(f"### {r['scenario']} run {r['run']} - {'SUCCESS' if r['success'] else 'FAIL'}: "
                     f"{_short(r['check'].get('reason', ''), 300)}")
        lines.append(f"- calls: `{m['calls']}`")
        if m.get("steps_per_turn"):
            lines.append(f"- LLM steps per turn: {m['steps_per_turn']}; tool-result chars per turn: "
                         f"{m.get('tool_result_chars')}")
        if m["errors"]:
            lines.append("- errors: " + "; ".join(_short(e, 300) for e in m["errors"]))
        if r.get("harness_error"):
            lines.append(f"- harness error: {r['harness_error']}")
        lines.append(f"- answer text of the last turn: {_short(m['final'], 600) or '(none)'}")
        if m.get("final_last_message") and _short(m["final_last_message"], 600) != _short(m["final"], 600):
            lines.append(f"- last assistant message: {_short(m['final_last_message'], 400)}")
        lines.append(f"- log: `{r['scenario']}_run{r['run']}.json`")
        lines.append("")
    md = "\n".join(lines)
    (out / "summary.md").write_text(md, encoding="utf-8")
    if extra_md:
        extra_md.parent.mkdir(parents=True, exist_ok=True)
        extra_md.write_text(md, encoding="utf-8")
    compact = [{"scenario": r["scenario"], "run": r["run"], "success": r["success"],
                "reason": r["check"].get("reason"), **{k: r["metrics"][k] for k in
                                                       ("tool_calls", "failed_tool_calls", "repeated_calls", "overflow",
                                                        "llm_steps", "seconds", "calls", "errors")},
                "final": _short(r["metrics"]["final"], 400),
                "final_last_message": _short(r["metrics"].get("final_last_message", ""), 400)} for r in results]
    (out / "summary.json").write_text(json.dumps({"meta": meta, "results": compact}, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    return md


# --------------------------------------------------------------------------- main
class _Enough(Exception):
    """--max-conversations reached."""


def summarize_logs(args) -> int:
    """--from-logs: one summary over the <scenario>_run<k>.json logs of several output folders."""
    folders = [Path(d.strip()).resolve() for d in args.from_logs.split(",") if d.strip()]
    results, metas = [], []
    for d in folders:
        results += [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*_run*.json"))]
        if (d / "summary.json").exists():
            metas.append(json.loads((d / "summary.json").read_text(encoding="utf-8"))["meta"])
    if not results:
        log(f"no logs in {', '.join(map(str, folders))}")
        return 1
    results.sort(key=lambda r: (r["run"], r["scenario"]))
    meta = dict(metas[0]) if metas else {"code": "?", "cloudclean_file": "?", "runs": "?", "date": "?"}
    codes = {m.get("code") for m in metas}
    if len(codes) > 1:
        log(f"WARNING: the logs come from different code folders: {codes}")
    meta["assets"] = " / ".join(dict.fromkeys(str(m.get("assets", "v1")) for m in metas)) or "?"
    meta["label"] = args.label or meta.get("label", "combined")
    meta["runs"] = max(r["run"] for r in results)
    meta["combined_from"] = [str(d) for d in folders]
    out = Path(args.out).resolve() if args.out else folders[0]
    out.mkdir(parents=True, exist_ok=True)
    meta["out"] = str(out)
    md = write_summary(results, meta, out, Path(args.summary_md) if args.summary_md else None,
                       Path(args.compare) if args.compare else None)
    print(md.split("## Details")[0])
    log(f"summary: {out / 'summary.md'} (from {len(results)} logs)")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code", help="folder that contains the cloudclean package to test (the repo root, or a "
                                  "snapshot's parent folder); required unless --from-logs")
    p.add_argument("--runs", type=int, default=2, help="runs per scenario (default 2)")
    p.add_argument("--scenarios", default="all", help="comma separated ids, e.g. A,B,G (default all)")
    p.add_argument("--out", help="output folder (default: %%TEMP%%/cloudclean_assistant_eval/<time>_<label>)")
    p.add_argument("--label", help="name in the summary (default: the --code folder name)")
    p.add_argument("--summary-md", help="also write the summary to this file")
    p.add_argument("--compare", help="summary.json of an earlier run (e.g. BEFORE) to compare with")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--python", help="interpreter for the server (default: the repo's .venv)")
    p.add_argument("--llm-base", default=DEFAULT_LLM)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=DEFAULT_KEY)
    p.add_argument("--settings", default='{"vision": true}',
                   help="assistant settings to PUT before each run, JSON (default: {\"vision\": true})")
    p.add_argument("--turn-timeout", type=float, default=900.0, help="seconds per user turn before the harness "
                                                                     "presses stop (default 900)")
    p.add_argument("--no-tunnel", action="store_true", help="do not start the SSH tunnel (the LLM URL must answer)")
    p.add_argument("--tunnel-host", default="spark")
    p.add_argument("--tunnel-remote-port", type=int, default=8001)
    p.add_argument("--assets", help="folder for the synthetic assets (default: %%TEMP%%/cloudclean_assistant_eval/"
                                    f"assets-{synth_assets.ASSET_VERSION}[-posed])")
    p.add_argument("--posed-flange", action="store_true",
                   help="store the flange in a scanner-frame pose (untidy coordinates) instead of axis-aligned at "
                        "the origin; compare only with runs made the same way")
    p.add_argument("--max-conversations", type=int, default=0, help="stop after this many LLM conversations")
    p.add_argument("--selftest", action="store_true",
                   help="no LLM: for each scenario check that a no-op fails and an oracle action (REST API) passes")
    p.add_argument("--from-logs", help="run nothing: write one summary from the JSON logs of these output folders "
                                       "(comma separated), into --out")
    args = p.parse_args(argv)

    if args.from_logs:
        return summarize_logs(args)
    if not args.code:
        p.error("--code is required")
    args.code = Path(args.code).resolve()
    if not (args.code / "cloudclean" / "__init__.py").exists():
        p.error(f"{args.code} has no cloudclean package")
    args.python = args.python or str(_venv_python() or sys.executable)
    try:
        args.settings = json.loads(args.settings)
    except ValueError as exc:
        p.error(f"--settings is not JSON: {exc}")
    label = args.label or ("selftest" if args.selftest else args.code.name)
    root = Path(tempfile.gettempdir()) / "cloudclean_assistant_eval"
    out = Path(args.out).resolve() if args.out else root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{label}"
    out.mkdir(parents=True, exist_ok=True)
    wanted = list(S.BY_ID) if args.scenarios == "all" else [s.strip().upper() for s in args.scenarios.split(",")]
    unknown = [s for s in wanted if s not in S.BY_ID]
    if unknown:
        p.error(f"unknown scenario(s) {unknown}; known: {', '.join(S.BY_ID)}")

    env = {**os.environ, "PYTHONPATH": str(args.code), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "CLOUDCLEAN_LLM_BASE_URL": args.llm_base, "CLOUDCLEAN_LLM_MODEL": args.model,
           "CLOUDCLEAN_LLM_API_KEY": args.api_key}
    (out / "ws").mkdir(exist_ok=True)
    probe = subprocess.run([args.python, "-c", "import cloudclean; print(cloudclean.__file__)"], env=env,
                           cwd=str(out / "ws"), capture_output=True, text=True)
    imported = probe.stdout.strip()
    if probe.returncode or not imported or not Path(imported).resolve().is_relative_to(args.code / "cloudclean"):
        log(f"ERROR: the server would import cloudclean from {imported or probe.stderr[-400:]}, not from {args.code}")
        return 2
    log(f"code under test: {imported}")
    variant = synth_assets.asset_version(args.posed_flange)
    asset_dir = Path(args.assets) if args.assets else root / f"assets-{variant}"
    assets = synth_assets.build_assets(asset_dir, log, posed=args.posed_flange)
    assets.update(synth_assets.build_real_assets(asset_dir, assets["flange_scan"], log))
    variant += f"+{synth_assets.REAL_VERSION}"
    meta = {"label": label, "code": str(args.code), "cloudclean_file": imported, "llm_base": args.llm_base,
            "model": args.model, "settings": args.settings, "runs": args.runs, "scenarios": wanted,
            "assets": variant, "date": datetime.now().isoformat(timespec="seconds"), "selftest": args.selftest,
            "out": str(out)}

    tunnel = None
    results: list[dict] = []
    ledger = out / "conversations.log"
    try:
        if not args.selftest:
            if args.no_tunnel:
                info = llm_models(args.llm_base, args.api_key)
                if info is None:
                    log(f"ERROR: no LLM answers at {args.llm_base}")
                    return 2
            else:
                port = int(re.search(r":(\d+)", args.llm_base.split("//", 1)[-1]).group(1))
                tunnel = Tunnel(args.tunnel_host, port, args.tunnel_remote_port, args.llm_base, args.api_key,
                                out / "tunnel.log")
                info = tunnel.ensure()
            model_rows = info.get("data", []) if isinstance(info, dict) else []
            row = next((m for m in model_rows if m.get("id") == args.model), None)
            if row is None:
                log(f"ERROR: model {args.model} is not served (served: {[m.get('id') for m in model_rows]})")
                return 2
            meta["context_window"] = row.get("max_model_len") or row.get("context_length")
            log(f"model {args.model}, context window {meta['context_window']}")
        count = 0
        for run in range(1, args.runs + 1):
            for sid in wanted:
                if args.max_conversations and count >= args.max_conversations and not args.selftest:
                    log(f"stopping: --max-conversations {args.max_conversations} reached")
                    raise _Enough
                scn = S.BY_ID[sid]
                log(f"scenario {sid} run {run}: {scn.title}")
                r = run_scenario(scn, run, args, assets, env, out, args.selftest, ledger)
                count += 0 if args.selftest else 1
                results.append(r)
                if args.selftest:
                    neg = r.get("negative_control", {})
                    ok = r["success"] and not neg.get("success")
                    log(f"  selftest {sid}: {'OK' if ok else 'BROKEN'} (no-op -> {neg.get('success')}, "
                        f"oracle -> {r['success']}: {r['check'].get('reason')})")
                else:
                    m = r["metrics"]
                    log(f"  -> {'SUCCESS' if r['success'] else 'FAIL'} ({r['check'].get('reason')}); "
                        f"{m['tool_calls']} calls, {m['repeated_calls']} repeats, overflow {m['overflow']}, "
                        f"{m['seconds']}s")
    except (_Enough, KeyboardInterrupt):
        pass
    finally:
        if tunnel is not None:
            tunnel.close()
    if not results:
        return 1
    md = write_summary(results, meta, out, Path(args.summary_md) if args.summary_md else None,
                       Path(args.compare) if args.compare else None)
    print(md.split("## Details")[0])
    log(f"summary: {out / 'summary.md'}")
    if args.selftest:
        broken = [r["scenario"] for r in results if not (r["success"] and not r.get("negative_control", {}).get("success"))]
        return 1 if broken else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
