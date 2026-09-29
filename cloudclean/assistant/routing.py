"""Which tools the model sees in a turn.

A small context window (24k tokens or less, e.g. the shared 16k Qwen server) cannot hold all 28 tool descriptions
(about 7k tokens) next to the system prompt, the conversation and a screenshot. Such a turn gets:
  - a core set (the viewer, the app guide, asset details, the workspace overview),
  - the tools the user's words ask for (keywords below),
  - the tools of the step the user is on,
  - the tools used in the conversation's last turns (follow-ups such as "now fill them"),
at most MAX_SMALL tools, and the app guide without its list of feature ids. The model can still call any tool by
name; it only sees the descriptions of these. Big windows get every tool, unchanged."""
from __future__ import annotations

import re

CORE = ("ui", "app_guide", "get_asset", "workspace_overview")
MAX_SMALL = 12

STEP_TOOLS = {
    "capture": ("capture", "turntable", "look_at_photos", "photos_to_3d"),
    "clean": ("clean", "edit", "select_region"),
    "align": ("assess_merge", "merge", "compare_merge_options"),
    "mesh": ("mesh", "holes", "colour_from_photos"),
    "measure": ("measure", "describe_part", "check_against_golden"),
    "export": ("export_asset", "rename_asset"),
}

# words (lower case, matched at word starts) -> tools they call for
KEYWORDS = {
    "holes": ("hole", "gap", "fill", "sticker", "marker", "watertight", "patch", "plug", "close"),
    "mesh": ("mesh", "surface", "poisson", "watertight", "stl", "triangl", "solid"),
    "measure": ("measure", "diameter", "length", "width", "height", "thick", "distance", "how big", "how long",
                "how wide", "size", "angle", "radius", "dimension", "bore", "mm", "caliper", "flat"),
    "describe_part": ("describe", "what is", "what kind", "shape", "part", "feature", "dimension"),
    "clean": ("clean", "noise", "noisy", "outlier", "stray", "floor", "tidy", "speck", "junk"),
    "select_region": ("select", "region", "highlight", "area", "these points", "this part"),
    "edit": ("rotate", "move", "crop", "cut", "trim", "remove", "delete points", "scale", "mirror", "flip",
             "level", "transform", "resize", "smooth", "true size", "straighten"),
    "assess_merge": ("merge", "combine", "align", "join", "stitch", "both scans", "two scans"),
    "merge": ("merge", "combine", "join", "stitch"),
    "compare_merge_options": ("which way", "orientation", "upside down", "flipped", "fits"),
    "run_pipeline": ("pipeline", "everything", "all steps", "process it", "automatic", "start to finish"),
    "compare_to_reference": ("compare", "reference", "deviation", "heatmap", "heat map"),
    "check_against_golden": ("golden", "master", "cad", "compare", "deviation", "inspect", "tolerance", "rescan",
                             "scan again", "missing", "match", "pass", "fail", "check"),
    "look_at_photos": ("photo", "picture", "image", "looks like", "real part", "screenshot"),
    "photos_to_3d": ("from photos", "from pictures", "photogrammetry", "3d from", "model from photo",
                     "scale sheet", "ruler"),
    "colour_from_photos": ("colour", "color", "texture", "paint"),
    "fill_from_photos": ("fill from photo", "photo fill", "fill the missing", "fill in from photo"),
    "export_asset": ("export", "save", "download", "file", "stl", "obj", "ply", "step file"),
    "rename_asset": ("rename", "call it", "name it"),
    "project": ("project", "folder", "move to"),
    "capture": ("scan", "scanner", "capture", "connect", "coverage", "metroy"),
    "turntable": ("turntable", "table", "spin", "degrees"),
    "request_delete": ("delete", "get rid of", "remove the model", "remove model"),
    "describe_parameters": ("parameter", "setting", "option", "default"),
}

SLIM_APP_GUIDE = ("The app guide: where a feature is in CloudClean, what it does and how to use it. question = the "
                  "user's words (e.g. 'where do I measure the head height'); it finds the feature and, with open, "
                  "takes the user there. Always say where it is in words too.")


def _hits(text: str, words: tuple[str, ...]) -> int:
    return sum(1 for w in words if re.search(r"(?<![a-z])" + re.escape(w), text))


def select_tools(available: list[str], message: str, step: str | None = None,
                 recent: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Names of the tools to describe to a small-window model this turn, in `available` order."""
    text = (message or "").lower()
    scores = {name: _hits(text, words) for name, words in KEYWORDS.items()}
    chosen: list[str] = [n for n in CORE if n in available]
    for name in sorted((n for n, s in scores.items() if s > 0), key=lambda n: -scores[n]):
        if name in available and name not in chosen:
            chosen.append(name)
    for name in list(recent) + list(STEP_TOOLS.get(step or "", ())):
        if name in available and name not in chosen:
            chosen.append(name)
    chosen = chosen[:MAX_SMALL]
    return [n for n in available if n in chosen]


def recent_tools(messages: list[dict], turns: int = 2) -> list[str]:
    """Tools called in the conversation's last `turns` user turns (newest first)."""
    seen: list[str] = []
    users = 0
    for m in reversed(messages):
        if m.get("role") == "user":
            users += 1
            if users > turns:
                break
        for c in m.get("tool_calls") or []:
            name = (c.get("function") or {}).get("name")
            if name and name not in seen:
                seen.append(name)
    return seen


def schemas_for(tools: dict, names: list[str], small: bool) -> list[dict]:
    """The tools' schemas; the app guide without its feature-id list when the window is small."""
    out = []
    for name in names:
        schema = tools[name].schema()
        if small and name == "app_guide":
            schema = {**schema, "function": {**schema["function"], "description": SLIM_APP_GUIDE}}
        out.append(schema)
    return out
