"""Autopilot: scan files in -> cleaned, merged, meshed, (optionally inspected) and exported out, hands-free.

- `AutopilotSettings` – the `autopilot` settings section.
- `group_files` / `suggest_name` – which files belong to one item and what it is called.
- `AutopilotState` – processed files + run history, persisted as JSON so restarts don't reprocess.
- `Watcher` – polling thread (no extra dependency) that waits for files to settle, groups them into items and
  hands each item to a `submit(name, files)` callable (the web layer submits an `autopilot` job).
- `run_autopilot_cli` – the same watching in the terminal, processing in-process with `run_pipeline`.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .clean import CleanParams
from .io import CAD_EXTS, SUPPORTED_EXTS
from .mesh import MeshParams
from .params import ParamsMixin
from .register import MergeParams

SCAN_EXTS = SUPPORTED_EXTS - CAD_EXTS
MESH_FORMATS = ["ply", "stl", "obj", "glb", "off"]
MERGE_METHODS = ["auto", "markers", "icp", "none"]
# auto: check first, merge only when it helps and is safe; ask: always stop for the user's decision;
# always: merge without checking (old behaviour); never: mesh and export every scan separately
MERGE_MODES = ["auto", "ask", "always", "never"]
DECISIONS = ["merge", "best", "separate", "cancel"]
TEMP_SUFFIXES = (".tmp", ".part", ".crdownload", ".partial", ".download")
HISTORY_LIMIT = 100
MESH_DEVIATION_WARN = 3.0  # warn when mesh deviation p95 exceeds this many point spacings


@dataclass
class AutopilotSettings(ParamsMixin):
    watch_enabled: bool = False
    watch_folder: str = ""
    output_folder: str = ""               # empty -> <workspace>/exports (web) ; CLI always passes -o
    recursive: bool = True                # look into sub-folders (each top-level sub-folder is one item)
    settle_seconds: float = 5.0           # a file is ready once its size/mtime stayed unchanged this long
    group_seconds: float = 30.0           # loose files arriving within this gap form one item
    formats: list = field(default_factory=lambda: ["stl", "ply"])
    preset: str = "standard"
    # Parts on a turntable/table are usually scanned together with the support surface, and nobody is there to
    # crop it. The plane remover is guarded (only removes a plane wider than the object with the object entirely
    # on one side of it), so enabling it is safe for scans without a table too.
    remove_plane: bool = True
    merge_method: str = "auto"            # alignment method used when scans are merged
    merge_mode: str = "auto"              # auto | ask | always | never (see MERGE_MODES)
    # watertight False: unscanned areas are not invented; set it True for printable closed meshes.
    mesh: dict = field(default_factory=lambda: {"method": "poisson", "watertight": False, "depth": 0,
                                                "smooth_iterations": 0, "target_triangles": 0})
    reference_id: str = ""                # CAD asset to compare against (optional)
    compare: dict = field(default_factory=dict)   # CompareParams overrides for the optional comparison
    auto_on_upload: bool = False
    auto_on_capture: bool = True

    @classmethod
    def load(cls, data: dict | None) -> "AutopilotSettings":
        """Build from stored settings, ignoring keys this version does not know."""
        valid = {f for f in cls.__dataclass_fields__}
        return cls.from_dict({k: v for k, v in (data or {}).items() if k in valid})

    def validate(self) -> "AutopilotSettings":
        """Raise ValueError with a human sentence when a value is unusable."""
        if isinstance(self.formats, str):
            self.formats = [f for f in re.split(r"[,\s]+", self.formats) if f]
        self.formats = [str(f).strip(".").lower() for f in self.formats or []]
        if not self.formats:
            raise ValueError("Choose at least one export format")
        bad = [f for f in self.formats if f not in MESH_FORMATS]
        if bad:
            raise ValueError(f"Unsupported export format '{bad[0]}'. Choose from: {', '.join(MESH_FORMATS)}")
        if self.preset not in CleanParams.PRESETS:
            raise ValueError(f"Unknown cleaning preset '{self.preset}'. Choose from: {', '.join(CleanParams.PRESETS)}")
        if self.merge_method not in MERGE_METHODS:
            raise ValueError(f"Unknown merge method '{self.merge_method}'. Choose from: {', '.join(MERGE_METHODS)}")
        if self.merge_mode not in MERGE_MODES:
            raise ValueError(f"Unknown merge mode '{self.merge_mode}'. Choose from: {', '.join(MERGE_MODES)}")
        if not isinstance(self.mesh, dict):
            raise ValueError("mesh settings must be an object")
        mesh = self.mesh_params()
        if mesh.method not in ("poisson", "bpa"):
            raise ValueError(f"Unknown mesh method '{mesh.method}' (use poisson or bpa)")
        if not isinstance(self.compare, dict):
            raise ValueError("compare settings must be an object")
        if self.settle_seconds < 0 or self.group_seconds < 0:
            raise ValueError("settle_seconds and group_seconds must not be negative")
        self.reference_id = (self.reference_id or "").strip()
        self.watch_folder = (self.watch_folder or "").strip().strip('"')
        self.output_folder = (self.output_folder or "").strip().strip('"')
        return self

    def clean_params(self) -> CleanParams:
        p = CleanParams.preset(self.preset)
        p.remove_plane = bool(self.remove_plane)
        return p

    def merge_params(self) -> MergeParams:
        return MergeParams.from_dict({"method": self.merge_method})

    def mesh_params(self) -> MeshParams:
        return MeshParams.from_dict(self.mesh or {})


# --------------------------------------------------------------------------- grouping / naming
_SUFFIX = re.compile(
    r"(?:[\s_\-.]*\(\d+\)"                                              # "name (3)"
    r"|[\s_\-.]+(?:scan|take|view|shot|pass|copy)?[\s_\-.]*\d+"          # "name_1", "name-scan2", "name view 3"
    r"|[\s_\-.]*(?:scan|take|view|shot|pass|copy)[\s_\-.]*\d*)$", re.IGNORECASE)


def strip_scan_suffix(stem: str) -> str:
    """'gear_1' -> 'gear', 'gear-scan2' -> 'gear', 'gear (3)' -> 'gear'. Never returns an empty string."""
    name = stem.strip()
    while True:
        shorter = _SUFFIX.sub("", name).rstrip(" _-.")
        if not shorter or shorter == name:
            return name
        name = shorter


def suggest_name(paths) -> str:
    """Item name from filenames: their common prefix after removing scan numbering."""
    stems = [strip_scan_suffix(Path(p).stem) for p in paths]
    if not stems:
        return "item"
    if len(set(stems)) == 1:
        return stems[0]
    prefix = os.path.commonprefix(stems).rstrip(" _-.(")
    return prefix if len(prefix) >= 2 else stems[0]


def is_candidate(path, root=None, exclude=()) -> bool:
    """A scan file autopilot may pick up (not a temp/partial download, not CAD, not in an excluded folder)."""
    p = Path(path)
    name = p.name
    if name.startswith(("~$", ".")) or name.lower().endswith(TEMP_SUFFIXES):
        return False
    if p.suffix.lower() not in SCAN_EXTS:
        return False
    if root is not None and any(part.startswith(".") for part in _relative(p, root).parts[:-1]):
        return False  # hidden sub-folders (.git, .Trash, sync-tool caches)
    return not any(_inside(p, ex) for ex in exclude if ex)


def _relative(path: Path, root) -> Path:
    try:
        return Path(os.path.abspath(path)).relative_to(os.path.abspath(root))
    except ValueError:
        return Path(Path(path).name)


def _inside(path, folder) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(os.path.abspath(folder))
        return True
    except ValueError:
        return False


def group_files(paths_with_mtimes, group_seconds: float, root=None, exclude=()) -> list[dict]:
    """Group scan files into items.

    paths_with_mtimes: iterable of (path, arrival time in seconds). Files in the same top-level sub-folder of
    `root` form one item named after the folder. Loose files form one item as long as each arrives within
    `group_seconds` of the previous one; the item is named from their common filename prefix.
    Returns [{"name", "files": [str], "folder": str | None, "first": t, "last": t}] ordered by arrival."""
    folders: dict[str, list] = {}
    loose: list = []
    for path, t in paths_with_mtimes:
        if not is_candidate(path, root, exclude):
            continue
        rel = _relative(Path(path), root) if root is not None else None
        if rel is not None and len(rel.parts) > 1:
            folders.setdefault(rel.parts[0], []).append((str(path), float(t)))
        else:
            loose.append((str(path), float(t)))

    groups = []
    for folder, items in folders.items():
        items.sort(key=lambda x: (x[1], x[0]))
        groups.append({"name": folder, "folder": folder, "files": [p for p, _ in items],
                       "first": items[0][1], "last": items[-1][1]})
    loose.sort(key=lambda x: (x[1], x[0]))
    current: list = []
    for item in loose:
        if current and item[1] - current[-1][1] > group_seconds:
            groups.append(_loose_group(current))
            current = []
        current.append(item)
    if current:
        groups.append(_loose_group(current))
    return sorted(groups, key=lambda g: (g["first"], g["name"]))


def _loose_group(items) -> dict:
    files = [p for p, _ in items]
    return {"name": suggest_name(files), "folder": None, "files": files, "first": items[0][1], "last": items[-1][1]}


# --------------------------------------------------------------------------- warnings / reports
def collect_warnings(merge_report: dict | None = None, mesh_report: dict | None = None,
                     compare_report: dict | None = None, names: list[str] | None = None) -> list[str]:
    """Human-readable warnings from engine reports (low merge overlap, symmetric ambiguity, mesh deviation)."""
    warnings = []
    for info in (merge_report or {}).get("scans", []):
        i = info.get("index", 0)
        label = names[i] if names and i < len(names) else f"scan {i + 1}"
        if info.get("warning"):
            warnings.append(f"Merge - {label}: {info['warning']}")
        if info.get("ambiguous"):
            alt = (info.get("alternatives") or [{}])[0]
            angle = alt.get("angle_from_best")
            rotated = f" rotated {angle:.0f} deg" if isinstance(angle, (int, float)) else ""
            warnings.append(f"Merge - {label}: a pose{rotated} fits almost as well; the part may be symmetric. "
                            "Check the alignment and use manual point pairs if it is wrong.")
    if mesh_report:
        dev, spacing = mesh_report.get("deviation") or {}, mesh_report.get("spacing") or 0
        if spacing and dev.get("p95") is not None and dev["p95"] > MESH_DEVIATION_WARN * spacing:
            warnings.append(f"Mesh deviates from the scan points: p95 {dev['p95']:.4f} > "
                            f"{MESH_DEVIATION_WARN:g}x point spacing ({spacing:.4f}). Check the mesh before using it.")
    if compare_report:
        for key in ("warning", "warnings"):
            value = compare_report.get(key)
            for w in ([value] if isinstance(value, str) else value or []):
                warnings.append(f"Compare: {w}")
    return warnings


def unique_folder(parent, name: str) -> Path:
    """<parent>/<name>, or '<name> (2)', '<name> (3)' … when a non-empty folder of that name exists."""
    parent = Path(parent)
    candidate, n = parent / name, 2
    while candidate.exists() and any(candidate.iterdir()):
        candidate, n = parent / f"{name} ({n})", n + 1
    return candidate


def safe_item_name(name: str) -> str:
    return re.sub(r"[^\w\-. ()]+", "_", name or "").strip(" .") or "item"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- persistent state
class AutopilotState:
    """Processed files ({path: {size, mtime}}) and the last runs, stored in one JSON file."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.processed: dict[str, dict] = {}
        self.entries: list[dict] = []
        try:
            data = json.loads(self.path.read_text())
            self.processed = dict(data.get("processed", {}))
            self.entries = list(data.get("history", []))[-HISTORY_LIMIT:]
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"processed": self.processed, "history": self.entries}, indent=2, default=str))
        os.replace(tmp, self.path)

    @staticmethod
    def key(path) -> str:
        return os.path.normcase(os.path.abspath(path))

    def is_processed(self, path, size: int, mtime: float) -> bool:
        with self.lock:
            rec = self.processed.get(self.key(path))
            return bool(rec) and rec.get("size") == size and abs(rec.get("mtime", 0) - mtime) < 1e-3

    def mark_processed(self, files: list[tuple[str, int, float]]) -> None:
        with self.lock:
            for path, size, mtime in files:
                self.processed[self.key(path)] = {"path": str(path), "size": size, "mtime": mtime}
            self._save()

    def add(self, item: str, files: list[str], job: dict | None, **extra) -> dict:
        job = job or {}
        output = job.get("output") or {}
        entry = {"item": item, "files": [str(f) for f in files], "job_id": job.get("id"),
                 "status": job.get("status", "queued"), "outputs": output.get("exports", []),
                 "warnings": output.get("warnings", []), "error": job.get("error"),
                 "started": _now(), "finished": _now() if job.get("status") in ("done", "failed") else None, **extra}
        if output.get("notes"):
            entry["notes"] = output["notes"]
        if job.get("status") == "done" and output.get("decision_needed"):
            entry.update(_decision_fields(job))
        with self.lock:
            self.entries = (self.entries + [entry])[-HISTORY_LIMIT:]
            self._save()
        return entry

    def refresh(self, get_job: Callable[[str], dict] | None) -> None:
        """Update unfinished entries from the job manager."""
        if get_job is None:
            return
        changed = False
        with self.lock:
            for entry in self.entries:
                if entry["status"] not in ("queued", "running") or not entry.get("job_id"):
                    continue
                try:
                    job = get_job(entry["job_id"])
                except KeyError:
                    entry.update(status="interrupted", finished=entry.get("finished") or _now())
                    changed = True
                    continue
                output = job.get("output") or {}
                new = {"status": job["status"], "outputs": output.get("exports", entry["outputs"]),
                       "warnings": output.get("warnings", entry["warnings"]), "error": job.get("error")}
                if output.get("notes"):
                    new["notes"] = output["notes"]
                if job["status"] == "done" and output.get("decision_needed"):
                    new.update(_decision_fields(job))
                if job.get("finished"):
                    new["finished"] = datetime.fromtimestamp(job["finished"]).isoformat(timespec="seconds")
                if any(entry.get(k) != v for k, v in new.items()):
                    entry.update(new)
                    changed = True
            if changed:
                self._save()

    def history(self) -> list[dict]:
        with self.lock:
            return [dict(e) for e in reversed(self.entries)]

    def find(self, job_id: str) -> dict | None:
        """The (live) history entry of a job."""
        with self.lock:
            return next((e for e in reversed(self.entries) if job_id and e.get("job_id") == job_id), None)

    def update(self, job_id: str, **fields) -> dict | None:
        with self.lock:
            entry = self.find(job_id)
            if entry is not None:
                entry.update(fields)
                self._save()
            return entry


def _decision_fields(job: dict) -> dict:
    """History fields of a run that stopped because the user must decide whether to merge."""
    from .register import assessment_summary

    output = job.get("output") or {}
    assessment = output.get("assessment") or {}
    return {"status": "needs_decision",
            "assessment": assessment_summary(assessment, assessment.get("names")),
            "decision": {"cleaned_ids": list(output.get("cleaned_ids") or []),
                         "transforms": assessment.get("transforms"), "best_index": assessment.get("best_index"),
                         "settings": (job.get("payload") or {}).get("settings")}}


# --------------------------------------------------------------------------- watcher
class Watcher:
    """Polls a folder, waits for files to settle, groups them into items and submits each item once.

    submit(name, files) -> job dict (at least {"id", "status"}); get_job(job_id) -> job dict (KeyError if gone).
    """

    def __init__(self, settings: AutopilotSettings, state: AutopilotState, submit: Callable[[str, list[str]], dict],
                 get_job: Callable[[str], dict] | None = None, exclude=(), poll_seconds: float = 1.0):
        self.settings = settings
        self.folder = Path(settings.watch_folder)
        self.state = state
        self.submit = submit
        self.get_job = get_job
        self.exclude = [Path(e) for e in exclude if e]
        if settings.output_folder:
            self.exclude.append(Path(settings.output_folder))
        self.poll_seconds = poll_seconds
        self.error: str | None = None
        self._seen: dict[str, dict] = {}   # path -> {size, mtime, changed (monotonic), arrival}
        self._initial = True
        self._pending: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- scanning
    def _scan(self):
        pattern = self.folder.rglob("*") if self.settings.recursive else self.folder.glob("*")
        for p in pattern:
            if not is_candidate(p, self.folder, self.exclude):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if p.is_file() and st.st_size > 0:
                yield p, st.st_size, st.st_mtime

    @staticmethod
    def _readable(path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                f.read(1)
            return True
        except OSError:
            return False

    def poll_once(self, force: bool = False) -> list[dict]:
        """One polling pass. Returns the history entries of items submitted in this pass.
        force=True treats every file as settled and every group as quiet (for `--once`)."""
        with self._lock:
            if not self.folder.is_dir():
                self.error = f"Watch folder not found: {self.folder}"
                self._pending = []
                return []
            self.error = None
            now_mono, now_wall = time.monotonic(), time.time()
            current = {}
            for p, size, mtime in self._scan():
                if self.state.is_processed(p, size, mtime):
                    continue
                key = str(p)
                prev = self._seen.get(key)
                if prev is None:
                    # files already there when watching starts are grouped by modification time; files that
                    # arrive later by when they showed up (copies often keep old mtimes)
                    arrival = mtime if (self._initial or force) else now_wall
                    rec = {"size": size, "mtime": mtime, "changed": now_mono, "arrival": arrival}
                elif (prev["size"], prev["mtime"]) != (size, mtime):
                    rec = {**prev, "size": size, "mtime": mtime, "changed": now_mono}
                else:
                    rec = prev
                current[key] = rec
            self._seen = current
            self._initial = False

            groups = group_files([(k, r["arrival"]) for k, r in current.items()], self.settings.group_seconds,
                                 self.folder, self.exclude)
            submitted, pending = [], []
            for g in groups:
                recs = [current[f] for f in g["files"]]
                settled = force or all(now_mono - r["changed"] >= self.settings.settle_seconds for r in recs)
                quiet = force or now_mono - max(r["changed"] for r in recs) >= self.settings.group_seconds
                if settled and quiet and not all(self._readable(Path(f)) for f in g["files"]):
                    for r in recs:  # still locked by the writer (Windows): wait a bit longer
                        r["changed"] = now_mono
                    settled = False
                if not (settled and quiet):
                    last = max(r["changed"] for r in recs)
                    pending.append({"item": g["name"], "files": g["files"], "settled": settled,
                                    "ready_in": round(max(0.0, max(self.settings.settle_seconds,
                                                                   self.settings.group_seconds)
                                                          - (now_mono - last)), 1)})
                    continue
                self.state.mark_processed([(f, current[f]["size"], current[f]["mtime"]) for f in g["files"]])
                for f in g["files"]:
                    self._seen.pop(f, None)
                try:
                    job = self.submit(g["name"], list(g["files"]))
                except Exception as exc:  # keep watching; the failure is visible in the history
                    job = {"id": None, "status": "failed", "error": str(exc)}
                submitted.append(self.state.add(g["name"], g["files"], job))
            self._pending = pending
            return submitted

    # ----------------------------------------------------------------- thread
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:  # never let the thread die silently
                self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.poll_seconds)

    def start(self) -> "Watcher":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="cloudclean-autopilot-watcher", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def pending(self) -> list[dict]:
        return list(self._pending)

    @property
    def watching(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        self.state.refresh(self.get_job)
        return {"watching": self.watching, "folder": str(self.folder), "error": self.error,
                "pending_items": self.pending(), "history": self.state.history()}


# --------------------------------------------------------------------------- terminal runner
def process_item(files: list[str], out_dir, settings: AutopilotSettings, name: str, log=print,
                 decide: Callable[[dict], str] | None = None) -> dict:
    """Run the full pipeline in-process for one item and write everything into out_dir.
    decide(assessment) -> "merge" | "best" | "separate" is asked in merge_mode "ask" (interactive terminal)."""
    from .pipeline import run_pipeline

    out = Path(out_dir)
    names = [Path(f).stem for f in files]
    if settings.reference_id:
        log("Note: CAD comparison (reference_id) needs the web workspace - skipped in the terminal runner")
    report = run_pipeline(list(files), out, settings.clean_params(), settings.merge_params(),
                          settings.mesh_params(), skip_clean=False, formats=settings.formats,
                          merge_mode=settings.merge_mode, decide=decide, log=log)
    # name the deliverables after the item instead of "mesh.<fmt>" ("mesh_<scan>.<fmt>" -> "<item>_<scan>.<fmt>")
    stem = safe_item_name(name)
    exports = []
    for p in report["outputs"].get("mesh", []):
        src = Path(p)
        suffix = src.stem[len("mesh"):] if src.stem.startswith("mesh") else f"_{src.stem}"
        dst = src.with_name(f"{stem}{suffix}{src.suffix}")
        if dst != src:
            os.replace(src, dst)
        exports.append(str(dst))
    report["outputs"]["mesh"] = exports
    report["item"] = name
    report["warnings"] = collect_warnings(report.get("merge"), report.get("mesh"), names=names)
    for m in report.get("meshes", []):
        report["warnings"] += collect_warnings(mesh_report=m.get("report"))
    for w in report["warnings"]:
        log(f"WARNING: {w}")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def terminal_decision(assessment: dict) -> str:
    """Ask in the terminal whether to merge (merge_mode "ask" in an interactive terminal)."""
    print("\nThe scans were checked before merging:", flush=True)
    for reason in assessment.get("reasons", []):
        print(f"  - {reason}", flush=True)
    default = {"merge": "m", "use_best": "b"}.get(assessment.get("recommendation"), "s")
    choices = {"m": "merge", "b": "best", "s": "separate"}
    while True:
        answer = input(f"Merge (m), use only the best scan {assessment.get('best_index', 0) + 1} (b) or mesh each "
                       f"scan separately (s)? [{default}] ").strip().lower() or default
        if answer[:1] in choices:
            return choices[answer[:1]]


def run_autopilot_cli(folder, out, settings: AutopilotSettings | None = None, once: bool = False,
                      poll_seconds: float = 1.0) -> list[dict]:
    """`cloudclean watch`: watch `folder` and write every item to `out/<item>/`. Returns the history entries.
    once=True processes the files currently in the folder (not yet processed) and returns."""
    from .pipeline import make_logger

    settings = AutopilotSettings.load((settings or AutopilotSettings()).to_dict()).validate()
    folder, out = Path(folder).resolve(), Path(out).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Watch folder not found: {folder}")
    out.mkdir(parents=True, exist_ok=True)
    settings.watch_folder, settings.output_folder = str(folder), str(out)
    state = AutopilotState(out / "autopilot_state.json")
    interactive = sys.stdin is not None and sys.stdin.isatty()
    decide = terminal_decision if settings.merge_mode == "ask" and interactive else None

    def submit(name: str, files: list[str]) -> dict:
        target = unique_folder(out, safe_item_name(name))
        print(f"\n=== {name}: {len(files)} file(s) -> {target}", flush=True)
        for f in files:
            print(f"    {f}", flush=True)
        log = make_logger(f"[{name}] ")
        try:
            report = process_item(files, target, settings, name, log, decide=decide)
        except Exception as exc:
            print(f"error: {name} failed: {exc}", flush=True)
            return {"id": None, "status": "failed", "error": str(exc)}
        for note in report.get("notes", []):
            print(f"--- NOTE ({name}): {note}", flush=True)
        for w in report["warnings"]:
            print(f"!!! WARNING ({name}): {w}", flush=True)
        print(f"=== {name}: done -> {', '.join(report['outputs']['mesh'])}", flush=True)
        return {"id": None, "status": "done", "output": {"exports": report["outputs"]["mesh"],
                                                         "warnings": report["warnings"]}}

    watcher = Watcher(settings, state, submit, poll_seconds=poll_seconds)
    if once:
        entries = watcher.poll_once(force=True)
        print(f"Processed {len(entries)} item(s)" if entries else f"Nothing new to process in {folder}", flush=True)
        return entries

    print(f"Watching {folder} (settle {settings.settle_seconds:g}s, group {settings.group_seconds:g}s) -> {out}")
    print("Press Ctrl+C to stop", flush=True)
    entries: list[dict] = []
    announced: set[str] = set()
    while True:
        entries += watcher.poll_once()
        if watcher.error and watcher.error not in announced:
            print(f"warning: {watcher.error}", flush=True)
            announced.add(watcher.error)
        for p in watcher.pending():
            key = f"{p['item']}|{len(p['files'])}"
            if key not in announced:
                print(f"Waiting for '{p['item']}' ({len(p['files'])} file(s)) to finish arriving...", flush=True)
                announced.add(key)
        time.sleep(poll_seconds)
