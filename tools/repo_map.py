"""Repository map: what every file is for and how the files connect. Writes docs/repo-map.md.

Everything is read from the code, so the map cannot drift from it without the check noticing:
- Python imports (also the ones inside functions) and the modules loaded by name: the server's ROUTER_MODULES,
  the job runner's JOB_MODULES, the scanner driver registry (any "cloudclean.x.y" string);
- TypeScript / CSS imports in frontend/src, lazy import() included, and index.html's entry script;
- the API routes each route module serves, the /api/... paths each frontend file (and deploy/update.sh) calls,
  and which route module answers each call;
- the job kinds each module submits and the job module that runs each kind (JOBS = {...});
- each file's purpose: the module docstring, or the first comment of a TypeScript / CSS / shell file.

    python tools/repo_map.py            # rewrite docs/repo-map.md
    python tools/repo_map.py --check    # exit 1 when docs/repo-map.md is out of date (tests/test_repo_map.py)

How the system works (hand-drawn flows) is in FLOWCHART.md; why it is built this way is in DECISIONS.md.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "repo-map.md"

ROOT_FILES = ["cloudclean.sh", "CloudClean.bat", "Dockerfile", "compose.yaml"]
SCANNED = ("cloudclean/", "frontend/src/", "frontend/index.html", "frontend/vite.config.ts", "tests/", "tools/",
           "deploy/")
CODE_EXT = {".py": "py", ".ts": "ts", ".tsx": "ts", ".css": "css", ".sh": "sh", ".html": "html", ".bat": "sh"}
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "websocket"}

# Areas, in the order the map lists them: (key, title, path prefixes or exact paths). First match wins.
AREAS: list[tuple[str, str, tuple[str, ...]]] = [
    ("fe_app", "Frontend: app root and store", ("frontend/index.html", "frontend/vite.config.ts",
                                                "frontend/src/main.tsx", "frontend/src/App.tsx",
                                                "frontend/src/store.ts")),
    ("fe_home", "Frontend: home screen", ("frontend/src/home/",)),
    ("fe_models", "Frontend: model list", ("frontend/src/shell/models/",)),
    ("fe_shell", "Frontend: shell (top bar, journey, panels, dialogs)", ("frontend/src/shell/",)),
    ("fe_capture_step", "Frontend: Scan step", ("frontend/src/steps/capture/",)),
    ("fe_golden", "Frontend: golden check panel", ("frontend/src/steps/measure/golden/",)),
    ("fe_measure", "Frontend: Measure step", ("frontend/src/steps/measure/",)),
    ("fe_steps", "Frontend: Clean, Align, Mesh, Export steps", ("frontend/src/steps/",)),
    ("fe_assistant", "Frontend: assistant chat", ("frontend/src/features/assistant/",)),
    ("fe_capture", "Frontend: live capture and turntable", ("frontend/src/features/capture/",)),
    ("fe_features", "Frontend: colour and autopilot", ("frontend/src/features/",)),
    ("fe_viewport", "Frontend: 3D view overlays", ("frontend/src/viewport/",)),
    ("fe_viewer", "Frontend: 3D viewer (three.js)", ("frontend/src/viewer/",)),
    ("fe_lib", "Frontend: helpers and small stores", ("frontend/src/lib/",)),
    ("fe_ui", "Frontend: UI primitives", ("frontend/src/ui/",)),
    ("fe_styles", "Frontend: styles", ("frontend/src/styles/",)),
    ("web_core", "Backend: web server core", ("cloudclean/web/server.py", "cloudclean/web/workspace.py",
                                             "cloudclean/web/jobs.py", "cloudclean/web/settings.py",
                                             "cloudclean/web/auth.py", "cloudclean/web/__init__.py")),
    ("web_routes", "Backend: API routes", ("cloudclean/web/routes_",)),
    ("web_jobs", "Backend: background jobs", ("cloudclean/web/jobs_",)),
    ("assistant", "Backend: assistant", ("cloudclean/assistant/",)),
    ("drivers", "Backend: scanner drivers", ("cloudclean/capture/drivers/",)),
    ("metroy", "Backend: MetroY native capture", ("cloudclean/capture/metroy/",)),
    ("turntable", "Backend: turntable", ("cloudclean/capture/turntable/",)),
    ("capture", "Backend: capture session", ("cloudclean/capture/",)),
    ("cli", "Backend: command line and pipelines", ("cloudclean/__init__.py", "cloudclean/cli.py",
                                                    "cloudclean/pipeline.py", "cloudclean/autopilot.py")),
    ("measure", "Backend: measuring and checks", tuple(f"cloudclean/{m}.py" for m in (
        "understand", "fitting", "regions", "analysis", "accuracy", "golden", "golden_words", "scale_sheet"))),
    ("outline", "Backend: outline check (photo vs golden model)", ("cloudclean/outline_",)),
    ("photos", "Backend: photos and colour", tuple(f"cloudclean/{m}.py" for m in (
        "photo_align", "photo_check", "photo_fill", "texture"))),
    ("engine", "Backend: geometry engine", ("cloudclean/",)),
    ("deploy", "Setup and deploy", ("deploy/", *ROOT_FILES)),
    ("tests", "Tests", ("tests/",)),
    ("tools_metroy", "Tools: MetroY experiments", ("tools/metroy/",)),
    ("tools", "Tools: other", ("tools/",)),
]
AREA_TITLE = {key: title for key, title, _ in AREAS}
# Areas without a diagram of their own (the index lists them; a diagram would only say "uses the package").
NO_DIAGRAM = {"fe_styles", "tests", "tools_metroy", "tools"}


@dataclass
class File:
    path: str
    lang: str
    area: str
    purpose: str = ""
    uses: set[str] = field(default_factory=set)        # imports
    loads: set[str] = field(default_factory=set)       # modules loaded by name (registries)
    mentions: set[str] = field(default_factory=set)    # scripts a script runs or installs
    routes: list[tuple[str, str]] = field(default_factory=list)  # (METHOD, path) served
    calls: set[str] = field(default_factory=set)       # /api/... paths called (normalised)
    submits: set[str] = field(default_factory=set)     # job kinds submitted
    runs: set[str] = field(default_factory=set)        # job kinds implemented


# --------------------------------------------------------------------------- collecting files

def list_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.splitlines()
        paths = sorted({p for p in out if (ROOT / p).is_file()})
    except (OSError, subprocess.CalledProcessError):  # no git (e.g. a copied tree): walk it
        paths = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
                       and not any(part in {".git", "node_modules", "__pycache__", ".venv"} for part in p.parts))
    keep = []
    for p in paths:
        if p.startswith("cloudclean/web/static/") or ".egg-info/" in p:
            continue
        if p in ROOT_FILES or p == "tools/recon/Dockerfile":
            keep.append(p)
        elif p.startswith(SCANNED) and Path(p).suffix in CODE_EXT:
            keep.append(p)
    return keep


def area_of(path: str) -> str:
    for key, _, prefixes in AREAS:
        if any(path == p or p.endswith(("/", "_")) and path.startswith(p) for p in prefixes):
            return key
    return "tools" if path.startswith("tools/") else "deploy"


def lang_of(path: str) -> str:
    if Path(path).name == "Dockerfile" or path.endswith(".yaml"):
        return "sh"
    return CODE_EXT.get(Path(path).suffix, "sh")


# --------------------------------------------------------------------------- purposes

def first_sentences(text: str, limit: int = 190) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("; "))
    if stop > limit // 2:
        return cut[:stop + 1]
    return cut[:cut.rfind(" ")].rstrip(",;:-") + " …"


def py_purpose(tree: ast.Module) -> str:
    doc = ast.get_docstring(tree)
    if doc:
        return first_sentences(doc.strip().split("\n\n")[0])
    return "Package marker (empty)." if not tree.body else ""


HEADER_COMMENT_RE = re.compile(r"/\*+(.*?)\*/|((?://[^\n]*(?:\n|$))+)", re.S)
IMPORT_STMT_RE = re.compile(r"""(?:import|@import)\b[^'";]*?['"][^'"]+['"]\)?;?""", re.S)
DIRECTIVE_RE = re.compile(r"^\s*(eslint|@ts-|prettier|global\b)")


def header_comment(src: str) -> str:
    """A TypeScript / CSS file's purpose: its header comment, the first comment before any code (imports are
    skipped). A divider such as `/* ==== buttons */` is a section title, not a purpose."""
    rest = src
    while True:
        rest = rest.lstrip()
        m = HEADER_COMMENT_RE.match(rest)
        if m:
            body = m.group(1) if m.group(1) is not None else re.sub(r"^[ \t]*//[ \t]?", "", m.group(2), flags=re.M)
            body = re.sub(r"^[ \t]*\*[ \t]?", "", body, flags=re.M)
            body = re.sub(r"[=\-]{4,}", " ", body).strip()
            if len(body) >= 30 and not DIRECTIVE_RE.match(body):
                return first_sentences(body.split("\n\n")[0])
            rest = rest[m.end():]
            continue
        m = IMPORT_STMT_RE.match(rest)
        if not m:
            return ""
        rest = rest[m.end():]


def sh_purpose(src: str) -> str:
    lines: list[str] = []
    for line in src.splitlines()[:40]:
        s = line.strip()
        if s.startswith("#!") or s.lower().startswith("@echo"):
            continue
        m = re.match(r"^(?:#|rem\s|::)\s?(.*)$", s, re.I)
        if m:
            text = m.group(1).strip()
            if not text and lines:
                break
            if text and not text.startswith(("syntax=", "-*-")):
                lines.append(text)
        elif lines:
            break
    return first_sentences(" ".join(lines)) if lines else ""


# --------------------------------------------------------------------------- Python

def module_name(path: str) -> str:
    parts = Path(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def scan_python(f: File, src: str, modules: dict[str, str], by_path: set[str]) -> None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        f.purpose = "(could not be parsed)"
        return
    f.purpose = py_purpose(tree)
    here = Path(f.path)
    package = module_name(f.path).split(".")
    if here.name != "__init__.py":
        package = package[:-1]

    def resolve(name: str) -> str | None:
        sibling = (here.parent / f"{name.split('.')[0]}.py").as_posix()
        if sibling in by_path and not name.startswith("cloudclean"):
            return sibling  # tools import their neighbours by bare name
        parts = name.split(".")
        while parts:
            hit = modules.get(".".join(parts))
            if hit:
                return hit
            parts.pop()
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = resolve(alias.name)
                if target:
                    f.uses.add(target)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package[:len(package) - (node.level - 1)]
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            for alias in node.names:
                target = modules.get(f"{base}.{alias.name}") or resolve(base)
                if target:
                    f.uses.add(target)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            m = re.fullmatch(r"(cloudclean(?:\.\w+)+)(?::\w+)?", node.value)
            if m and m.group(1) in modules:
                f.loads.add(modules[m.group(1)])
        elif isinstance(node, ast.Call):
            # jobs.submit("kind", ...) everywhere; run_job(ctx, "kind", ...) in the assistant's tools
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            pos = {"submit": 0, "run_job": 1}.get(name)
            if pos is not None and len(node.args) > pos and isinstance(node.args[pos], ast.Constant) \
                    and isinstance(node.args[pos].value, str):
                f.submits.add(node.args[pos].value)

    prefix = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "APIRouter":
            for kw in node.keywords:
                if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                    prefix = kw.value.value
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) \
                        and dec.func.attr in HTTP_METHODS and dec.args and isinstance(dec.args[0], ast.Constant):
                    f.routes.append((dec.func.attr.upper(), prefix + dec.args[0].value))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "JOBS" for t in node.targets) \
                and isinstance(node.value, ast.Dict):
            f.runs |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    f.uses.discard(f.path)
    f.loads.discard(f.path)


# --------------------------------------------------------------------------- TypeScript, CSS, HTML, shell

TS_IMPORT_RE = re.compile(r"""(?:^\s*import\s+(?:[^'";]*?\s+from\s+)?|^\s*export\s+[^'";]*?\s+from\s+|\bimport\s*\(\s*)
                              ['"]([^'"]+)['"]""", re.M | re.X)
API_RE = re.compile(r"""(?<!startsWith\()(['"`])(/api/[^'"`\n]*)\1""")  # a startsWith() test is not a call


def resolve_ts(base: Path, spec: str, by_path: set[str]) -> str | None:
    if not spec.startswith("."):
        return None  # a package (react, three, zustand ...)
    parts: list[str] = []
    for part in (base.parent.as_posix() + "/" + spec).split("/"):
        if part == "..":
            parts.pop()
        elif part not in (".", ""):
            parts.append(part)
    norm = "/".join(parts)
    for cand in (norm, norm + ".ts", norm + ".tsx", norm + ".css", norm + "/index.ts", norm + "/index.tsx"):
        if cand in by_path:
            return cand
    return None


def normalise_api(raw: str) -> str:
    path = re.sub(r"\$\{[^}]*\}", "{}", raw).split("?")[0].rstrip()
    return path + "{}" if path.endswith("/") else path


def scan_text(f: File, src: str, by_path: set[str]) -> None:
    if f.lang == "ts":
        f.purpose = header_comment(src)
        for spec in TS_IMPORT_RE.findall(src):
            target = resolve_ts(Path(f.path), spec, by_path)
            if target and target != f.path:
                f.uses.add(target)
        f.calls |= {normalise_api(m.group(2)) for m in API_RE.finditer(src)}
    elif f.lang == "css":
        f.purpose = header_comment(src)
    elif f.lang == "html":
        f.purpose = "The page the browser loads: the root element, the start-up intro and the entry script."
        for spec in re.findall(r"<script[^>]+src=['\"]/?([^'\"]+)['\"]", src):
            cand = f"{Path(f.path).parent.as_posix()}/{spec}"
            if cand in by_path:
                f.uses.add(cand)
    else:
        f.purpose = sh_purpose(src)
        for other in by_path:
            if other != f.path and area_of(other) == "deploy" and re.search(
                    rf"(?<![\w-]){re.escape(Path(other).name)}(?![\w-])", src):
                f.mentions.add(other)
        if re.search(r"\bcloudclean(?:\.exe)?\"?\s+serve\b|\"cloudclean\",\s*\"serve\"|-m cloudclean", src):
            f.mentions.add("cloudclean/cli.py")
        f.calls |= {normalise_api("/api/" + p) for p in re.findall(r"\$\(api\s+([\w/-]+)\)", src)}


# --------------------------------------------------------------------------- API matching

def _segments(path: str) -> list[str]:
    return path.strip("/").split("/")


def match_score(call: str, route: str) -> int | None:
    """How well a called path fits a route pattern: None = not at all, higher = more literal segments agree."""
    a, b = _segments(call), _segments(route)
    if b and re.fullmatch(r"\{\w+:path\}", b[-1]) and len(a) >= len(b):
        a = a[:len(b) - 1] + ["/".join(a[len(b) - 1:])]
    if len(a) != len(b):
        return None
    score = 0
    for x, y in zip(a, b):
        if x == y:
            score += 2
        elif re.fullmatch(r"\{\w+(?::\w+)?\}", y):
            continue
        elif "{}" in x and re.fullmatch(re.escape(x).replace(r"\{\}", ".*"), y):
            score += 1
        else:
            return None
    return score


@dataclass
class Repo:
    files: dict[str, File]
    route_owner: dict[str, list[tuple[str, str]]]      # route path -> [(method, file)]
    call_targets: dict[tuple[str, str], list[str]]     # (caller, called path) -> route paths
    unmatched: dict[str, set[str]]                     # caller -> paths no route answers
    job_owner: dict[str, str]                          # job kind -> file that runs it

    def edges(self) -> set[tuple[str, str, str]]:
        """(source, target, kind), kind = imports / loads / mentions / http / job."""
        out: set[tuple[str, str, str]] = set()
        for f in self.files.values():
            out |= {(f.path, t, "imports") for t in f.uses}
            out |= {(f.path, t, "loads") for t in f.loads}
            out |= {(f.path, t, "mentions") for t in f.mentions if t in self.files}
            out |= {(f.path, self.job_owner[k], "job") for k in f.submits
                    if k in self.job_owner and self.job_owner[k] != f.path}
        for (caller, _), routes in self.call_targets.items():
            for r in routes:
                out |= {(caller, owner, "http") for _, owner in self.route_owner[r] if owner != caller}
        return out


def scan() -> Repo:
    paths = list_files()
    by_path = set(paths)
    modules = {module_name(p): p for p in paths if p.endswith(".py")}
    files: dict[str, File] = {}
    for p in paths:
        f = File(p, lang_of(p), area_of(p))
        src = (ROOT / p).read_text(encoding="utf-8", errors="replace")
        if f.lang == "py":
            scan_python(f, src, modules, by_path)
        else:
            scan_text(f, src, by_path)
        files[p] = f

    route_owner: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for f in files.values():
        if f.path.startswith("cloudclean/"):
            for method, route in f.routes:
                route_owner[route].append((method, f.path))
    call_targets: dict[tuple[str, str], list[str]] = {}
    unmatched: dict[str, set[str]] = defaultdict(set)
    for f in files.values():
        if f.path.startswith("cloudclean/") or f.area in ("tests", "tools", "tools_metroy"):
            continue  # backend /api strings are messages; tests and tools call the API through TestClient
        for call in sorted(f.calls):
            scored = [(s, r) for r in route_owner if (s := match_score(call, r)) is not None]
            best = max((s for s, _ in scored), default=None)
            hits = sorted(r for s, r in scored if s == best)
            if not hits:  # a base URL the file adds paths to at run time ('/api/turntable' + '/status')
                hits = sorted(r for r in route_owner if r.startswith(call.rstrip("{}").rstrip("/") + "/"))
            owners = {owner for r in hits for _, owner in route_owner[r]}
            # several routes fit a path built at run time (`/api/capture/${action}`): fine when one module serves them
            if hits and (len(hits) <= 3 or len(owners) == 1):
                call_targets[(f.path, call)] = hits
            else:
                unmatched[f.path].add(call)
    job_owner = {kind: f.path for f in files.values() for kind in sorted(f.runs)}
    return Repo(files, dict(route_owner), call_targets, dict(unmatched), job_owner)


# --------------------------------------------------------------------------- rendering

def label(path: str) -> str:
    for base in ("cloudclean/", "frontend/src/"):
        if path.startswith(base):
            return path[len(base):]
    return path


def node_id(path: str) -> str:
    return "n_" + re.sub(r"\W", "_", path)


def area_id(area: str) -> str:
    return "a_" + area


def md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ") or "—"


ARROWS = {"imports": "-->", "loads": "-.->|loads|", "mentions": "-.->|runs|", "http": "-.->|HTTP|",
          "job": "==>|job|"}
STRENGTH = ("job", "http", "loads", "mentions", "imports")  # which arrow an area-to-area link is drawn with


def mermaid(lines: list[str]) -> list[str]:
    return ["```mermaid", *lines, "```", ""]


FE_SHARED = ("fe_app", "fe_lib", "fe_ui")  # used by nearly every frontend area: drawn once, not a dozen times


def area_box(repo: Repo, key: str) -> str:
    count = sum(1 for f in repo.files.values() if f.area == key)
    return f'{area_id(key)}["{AREA_TITLE[key].split(": ", 1)[-1]}<br/>{count} files"]'


def area_overview(repo: Repo, edges: set[tuple[str, str, str]], frontend: bool) -> list[str]:
    """Areas of one side and how they connect. The frontend's links into the store, helpers and UI primitives are
    left out (every area has them); its calls to the backend end on one box."""
    present = {f.area for f in repo.files.values()}
    shown = [k for k, _, _ in AREAS if k in present and k not in NO_DIAGRAM and k.startswith("fe_") == frontend]
    agg: dict[tuple[str, str], set[str]] = defaultdict(set)
    for s, t, kind in edges:
        a, b = repo.files[s].area, repo.files[t].area
        if frontend and a in shown and not b.startswith("fe_") and kind == "http":
            agg[(a, "backend")].add(kind)
        elif a != b and a in shown and b in shown and not (frontend and b in FE_SHARED):
            agg[(a, b)].add(kind)
    lines = ["flowchart LR"]
    if frontend:
        lines.append('  subgraph shared["Used by nearly every area"]')
        lines += [f"    {area_box(repo, k)}" for k in FE_SHARED if k in shown]
        lines.append("  end")
        lines += [f"  {area_box(repo, k)}" for k in shown if k not in FE_SHARED]
        lines.append(f'  {area_id("backend")}(["Backend API routes"]):::other')
    else:
        lines += [f"  {area_box(repo, k)}" for k in shown]
    for (a, b), kinds in sorted(agg.items()):
        lines.append(f"  {area_id(a)} {ARROWS[next(k for k in STRENGTH if k in kinds)]} {area_id(b)}")
    if frontend:
        lines.append("  classDef other stroke-dasharray: 4 3")
    return mermaid(lines)


def area_diagram(repo: Repo, area: str, edges: set[tuple[str, str, str]]) -> list[str]:
    inside = sorted(p for p, f in repo.files.items() if f.area == area)
    lines = ["flowchart LR", f'  subgraph {area_id(area)}["{AREA_TITLE[area]}"]']
    lines += [f'    {node_id(p)}["{label(p)}"]' for p in inside]
    lines.append("  end")
    outer: set[str] = set()
    links: set[str] = set()
    incoming: dict[str, set[str]] = defaultdict(set)
    for s, t, kind in edges:
        sa, ta = repo.files[s].area, repo.files[t].area
        if sa in NO_DIAGRAM and sa != area:
            continue  # tests, tools and styles are listed in the index, not drawn into every area
        if sa == area and ta == area:
            links.add(f"  {node_id(s)} {ARROWS[kind]} {node_id(t)}")
        elif sa == area:
            outer.add(ta)
            links.add(f"  {node_id(s)} {ARROWS[kind]} {area_id(ta)}")
        elif ta == area:
            incoming[sa].add(kind)  # who uses this area: one arrow per area (the index names each file)
    outer |= set(incoming)
    lines += [f'  {area_id(a)}(["{AREA_TITLE[a]}"]):::other' for a in sorted(outer)]
    lines += sorted(links)
    lines += [f"  {area_id(a)} {ARROWS[next(k for k in STRENGTH if k in kinds)]} {area_id(area)}"
              for a, kinds in sorted(incoming.items())]
    lines.append("  classDef other stroke-dasharray: 4 3")
    return mermaid(lines)


def http_diagram(repo: Repo, edges: set[tuple[str, str, str]]) -> list[str]:
    """Which part of the UI calls which route module: one arrow per area and module (the index names files)."""
    pairs = sorted({(repo.files[s].area, t) for s, t, k in edges if k == "http"})
    lines = ["flowchart LR", '  subgraph callers["Callers"]']
    lines += [f'    {area_id(k)}(["{title}"])' for k, title, _ in AREAS if any(a == k for a, _ in pairs)]
    lines += ["  end", '  subgraph routes["Backend: API routes and the server core"]']
    lines += [f'    {node_id(p)}["{label(p)}"]' for p in sorted({t for _, t in pairs})]
    lines.append("  end")
    lines += [f"  {area_id(a)} -.-> {node_id(t)}" for a, t in pairs]
    return mermaid(lines)


COMMON = {"cloudclean/io.py", "cloudclean/params.py"}  # used by nearly every job: named once, not drawn


def request_diagram(repo: Repo, edges: set[tuple[str, str, str]]) -> list[str]:
    """Route modules -> the jobs they start -> the engine modules each job uses."""
    starts = sorted({(s, t) for s, t, k in edges if k == "job" and repo.files[s].area in ("web_routes", "web_core")})
    route_files = sorted({s for s, _ in starts})
    job_files = sorted({t for _, t in starts} - set(route_files))
    engine_areas = {"engine", "measure", "outline", "photos", "cli"}
    uses = sorted({(s, t) for s, t, k in edges if k == "imports" and s in {t for _, t in starts}
                   and repo.files[t].area in engine_areas and t not in COMMON})
    lines = ["flowchart LR", '  subgraph R["API routes that start jobs"]']
    lines += [f'    {node_id(p)}["{label(p)}"]' for p in route_files]
    lines += ["  end", '  subgraph J["Job bodies, run in a worker process"]']
    lines += [f'    {node_id(p)}["{label(p)}"]' for p in job_files]
    lines += ["  end", '  subgraph E["Engine, measuring and photo modules (all also use io.py)"]']
    lines += [f'    {node_id(p)}["{label(p)}"]' for p in sorted({t for _, t in uses})]
    lines.append("  end")
    lines += [f"  {node_id(s)} ==> {node_id(t)}" for s, t in starts]
    lines += [f"  {node_id(s)} --> {node_id(t)}" for s, t in uses]
    return mermaid(lines)


def render(repo: Repo) -> str:
    edges = repo.edges()
    files = repo.files
    used_by: dict[str, set[str]] = defaultdict(set)
    tested_by: dict[str, set[str]] = defaultdict(set)
    for s, t, kind in edges:
        if kind != "http":
            (tested_by if files[s].area == "tests" and files[t].area != "tests" else used_by)[t].add(s)
    callers_of: dict[str, set[str]] = defaultdict(set)
    for (caller, _), routes in repo.call_targets.items():
        for r in routes:
            callers_of[r].add(caller)
    counts: dict[str, int] = defaultdict(int)
    for _, _, k in edges:
        counts[k] += 1

    out = [
        "# Repository map",
        "",
        "<!-- Generated by tools/repo_map.py from the code itself. Do not edit by hand: run"
        " `python tools/repo_map.py` after changing files; tests/test_repo_map.py fails while this is out of date."
        " -->",
        "",
        "What every file is for and how the files connect, read straight from the code: imports, modules loaded by"
        " name, the API routes the browser calls and the background jobs those routes start. For **how the system"
        " works** step by step see [FLOWCHART.md](../FLOWCHART.md); for **why** it is built this way see"
        " [DECISIONS.md](../DECISIONS.md).",
        "",
        f"{len(files)} files · {counts['imports']} imports · {counts['http']} browser → API links ·"
        f" {counts['job']} job links · {counts['loads']} modules loaded by name ·"
        f" {sum(len(v) for v in repo.route_owner.values())} API routes · {len(repo.job_owner)} job kinds",
        "",
        "**Reading the diagrams:** `A --> B` A imports B · `A -.->|HTTP| B` A calls an API route that B serves ·"
        " `A ==>|job| B` A starts a background job that B runs · `A -.->|loads| B` A loads B by name (router, job"
        " and driver registries) · `A -.->|runs| B` a script runs or installs B · rounded dashed boxes are other"
        " areas.",
        "",
        "## Contents",
        "",
        "1. [The whole app at a glance](#the-whole-app-at-a-glance)",
        "2. [Browser → API routes](#browser--api-routes)",
        "3. [API routes → jobs → engine](#api-routes--jobs--engine)",
        "4. [Inside each area](#inside-each-area)",
        "5. [API routes: who serves, who calls](#api-routes-who-serves-who-calls)",
        "6. [Job kinds](#job-kinds)",
        "7. [File index](#file-index)",
        "",
        "## The whole app at a glance",
        "",
        "The areas of the code and how they connect; tests, tools and styles are in the file index. The browser"
        " talks to the backend only over HTTP (`/api/...`, one WebSocket for the live scan, Server-Sent Events for"
        " the assistant).",
        "",
        "### Backend areas",
        "",
        *area_overview(repo, edges, frontend=False),
        "### Frontend areas",
        "",
        *area_overview(repo, edges, frontend=True),
        "## Browser → API routes",
        "",
        "Which part of the UI (and which deploy script) calls which backend module. The exact files and routes"
        " are in the [route table](#api-routes-who-serves-who-calls).",
        "",
        *http_diagram(repo, edges),
        "## API routes → jobs → engine",
        "",
        "Long work never runs inside a request: a route checks its input and submits a job (thick arrow), the job"
        " runs in a worker process (`web/jobs.py`) and calls the engine. Routes that answer at once (measuring,"
        " status, settings) are in the area diagrams below.",
        "",
        *request_diagram(repo, edges),
        "## Inside each area",
        "",
    ]
    for key, title, _ in AREAS:
        if key not in NO_DIAGRAM and any(f.area == key for f in files.values()):
            out += [f"### {title}", "", *area_diagram(repo, key, edges)]

    out += ["## API routes: who serves, who calls", "",
            "Callers are frontend files and deploy scripts. A route no file here calls is used by the assistant,"
            " the CLI, the Revo Metro bridge or the tests.", "",
            "| Route | Method | Served by | Called from |", "|---|---|---|---|"]
    for route in sorted(repo.route_owner):
        for method, owner in sorted(repo.route_owner[route]):
            calls = ", ".join(f"`{label(c)}`" for c in sorted(callers_of.get(route, ())))
            out.append(f"| `{md_cell(route)}` | {method} | `{label(owner)}` | {md_cell(calls)} |")
    out.append("")
    if repo.unmatched:
        out += ["Paths called that no route answers by the rules above (built at run time, or a route that is gone):",
                ""]
        out += [f"- `{label(c)}`: " + ", ".join(f"`{p}`" for p in sorted(repo.unmatched[c]))
                for c in sorted(repo.unmatched)]
        out.append("")

    out += ["## Job kinds", "", "| Job kind | Started by | Run by |", "|---|---|---|"]
    for kind in sorted(repo.job_owner):
        starters = ", ".join(f"`{label(f.path)}`" for f in sorted(files.values(), key=lambda f: f.path)
                             if kind in f.submits and f.path.startswith("cloudclean/"))
        out.append(f"| `{kind}` | {md_cell(starters)} | `{label(repo.job_owner[kind])}` |")
    out += ["", "## File index", ""]

    for key, title, _ in AREAS:
        members = sorted(p for p, f in files.items() if f.area == key)
        if not members:
            continue
        out += [f"### {title}", "", "| File | What it does | Uses | Used by | API, jobs, tests |",
                "|---|---|---|---|---|"]
        for p in members:
            f = files[p]
            extra = []
            if f.routes:
                extra.append("serves " + ", ".join(f"{m} `{r}`" for m, r in sorted(f.routes, key=lambda x: x[::-1])))
            calls = sorted({r for (c, _), rs in repo.call_targets.items() if c == p for r in rs})
            if calls:
                extra.append("calls " + ", ".join(f"`{r}`" for r in calls))
            if f.submits:
                extra.append("starts jobs " + ", ".join(f"`{k}`" for k in sorted(f.submits)))
            if f.runs:
                extra.append("runs jobs " + ", ".join(f"`{k}`" for k in sorted(f.runs)))
            if tested_by.get(p):
                extra.append("tested by " + ", ".join(f"`{Path(t).name}`" for t in sorted(tested_by[p])))
            uses = ", ".join(f"`{label(u)}`" for u in sorted(f.uses | f.loads | f.mentions))
            users = ", ".join(f"`{label(u)}`" for u in sorted(used_by.get(p, ())))
            out.append(f"| [{label(p)}](../{p}) | {md_cell(f.purpose)} | {md_cell(uses)} | {md_cell(users)} |"
                       f" {md_cell('; '.join(extra))} |")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 when docs/repo-map.md is out of date")
    args = ap.parse_args()
    text = render(scan())
    current = OUT.read_text(encoding="utf-8").replace("\r\n", "\n") if OUT.exists() else ""
    if args.check:
        print("docs/repo-map.md is up to date." if current == text
              else "docs/repo-map.md is out of date: run  python tools/repo_map.py")
        return 0 if current == text else 1
    if current == text:
        print("docs/repo-map.md is already up to date.")
    else:
        OUT.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
