"""The evaluation scenarios: what is imported, what the viewer shows, what the user types, how success is checked,
and (for run_eval.py --selftest) what a correct assistant would do, done through the REST API instead of the chat.

A check receives an `Outcome` (see run_eval.py): .assets {key: id}, .new_assets [meta], .descends(id, root_id),
.load_mesh(id) -> (V, F), .truth, .turns [turn records], .final_text, .tool_calls. It returns
{"success": bool, "reason": str, ...details}.

A-H are the requested scenarios. I is the user's failure reproduced faithfully (see its comment): A-H did not make
the BEFORE model loop or overflow, because the 15 listed holes included sticker holes it could act on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from checks import flange_check, mentions, numbers_in

USER_A = ("fill all the small holes (from the marker stickers used for scanning). Keep the 4 holes uniformly around "
          "the flange and the hole in the center")


@dataclass
class Turn:
    message: str
    image: str | None = None                    # asset key of a PNG attached like the chat's screenshot button
    image_note: str = "screenshot of the 3D view"


@dataclass
class Scenario:
    id: str
    title: str
    project: str
    imports: list[str]                          # asset keys (synth_assets.build_assets) imported into the project
    turns: list[Turn]
    check: Callable
    active: str | None = None                   # asset key active in the viewer (None: nothing open)
    visible: list[str] | None = None
    golden: str | None = None                   # asset key made the project's golden model
    extra_projects: list[tuple[str, list[str]]] = field(default_factory=list)
    oracle: Callable | None = None              # selftest: (rest, ids) -> final text a correct answer would give
    truth: str = "truth"                        # asset key of the ground truth the check uses


# --------------------------------------------------------------------------- checks
def check_flange_fill(root_key: str) -> Callable:
    def check(o) -> dict:
        root = o.assets[root_key]
        new = [m for m in o.new_assets if m["kind"] == "mesh" and o.descends(m["id"], root)]
        if not new:
            return {"success": False, "reason": "no new mesh was made", "candidates": []}
        results = []
        for m in new:
            V, F = o.load_mesh(m["id"])
            r = flange_check(V, F, o.truth)
            results.append({"id": m["id"], "name": m["name"], "operation": m.get("operation"), **r})
        good = [r for r in results if r["success"]]

        def badness(r):
            return (r["flaws_total"] - r["flaws_filled"]) + 10 * (r["design_total"] - r["design_open"]) + \
                r["loops_by_kind"]["other"]

        best = good[0] if good else min(results, key=badness)
        reason = f"ok: '{best['name']}'" if good else f"best of {len(results)} new mesh(es) '{best['name']}': " + \
            "; ".join(best["problems"])
        return {"success": bool(good), "reason": reason, "best": best["id"], "candidates": results}
    return check


def check_new_mesh(root_key: str) -> Callable:
    def check(o) -> dict:
        root = o.assets[root_key]
        new = [m for m in o.new_assets if m["kind"] == "mesh" and o.descends(m["id"], root)]
        shown = {i for c in o.tool_calls if c.get("name") == "ui" and (c.get("arguments") or {}).get("action") == "show"
                 for i in ((c.get("arguments") or {}).get("asset_ids") or [])}
        info = [{"id": m["id"], "name": m["name"], "triangles": (m.get("stats") or {}).get("triangles"),
                 "watertight": (m.get("stats") or {}).get("watertight"), "shown": m["id"] in shown,
                 "params": m.get("params")} for m in new]
        if not new:
            return {"success": False, "reason": "no mesh was made", "meshes": info}
        any_shown = any(i["shown"] for i in info)
        return {"success": True, "reason": f"{len(new)} mesh(es) made, " + ("one shown" if any_shown else "none shown"),
                "meshes": info}
    return check


def check_answer_number(target: float, tol: float) -> Callable:
    def check(o) -> dict:
        nums = numbers_in(o.final_text)
        hits = [v for v in nums if abs(v - target) <= tol]
        used = sorted({c.get("name") for c in o.tool_calls})
        return {"success": bool(hits), "reason": (f"answer gives {hits[0]:g}" if hits else
                                                   f"no number within {target:g} ± {tol:g} in the answer"),
                "numbers": nums[:20], "tools_used": used}
    return check


def check_answer_mentions(phrases: list[str], tool: str | None = None) -> Callable:
    def check(o) -> dict:
        found = mentions(o.final_text, phrases)
        called = any(c.get("name") == tool for c in o.tool_calls) if tool else None
        missing = [p for p, ok in found.items() if not ok]
        return {"success": not missing, "reason": ("answer names " + " and ".join(repr(p) for p in phrases) if not
                                                   missing else "answer does not mention " + ", ".join(map(repr, missing))),
                "mentions": found, f"{tool}_called": called}
    return check


def check_created_operation(operation: str) -> Callable:
    def check(o) -> dict:
        made = [m for m in o.new_assets if m.get("operation") == operation]
        return {"success": bool(made), "reason": (f"{operation} asset '{made[0]['name']}' made" if made else
                                                  f"no {operation} asset was made"),
                "created": [{"id": m["id"], "name": m["name"], "operation": m.get("operation")} for m in o.new_assets]}
    return check


# --------------------------------------------------------------------------- oracles (--selftest only)
def oracle_fill(root_key: str, mesh_first: bool = False, max_diameter: float = 15) -> Callable:
    def act(rest, ids) -> str:
        target = ids[root_key]
        if mesh_first:
            target = rest.job("/api/mesh", {"asset_id": target, "params": {}})[0]
        rest.job("/api/holes/fill", {"asset_id": target, "max_diameter": max_diameter})
        return "Filled the small holes; the bore and the 4 lightening holes stay open."
    return act


def oracle_mesh(root_key: str) -> Callable:
    def act(rest, ids) -> str:
        rest.job("/api/mesh", {"asset_id": ids[root_key], "params": {"watertight": True}})
        return "Made a watertight mesh."
    return act


def oracle_text(text: str) -> Callable:
    return lambda rest, ids: text


def oracle_golden(scan_key: str) -> Callable:
    def act(rest, ids) -> str:
        rest.job("/api/golden-check", {"scan_id": ids[scan_key]})
        return "Checked against the golden model."
    return act


# --------------------------------------------------------------------------- the scenarios
SCENARIOS = [
    Scenario("A", "user's case: fill sticker holes, keep bore + 4 holes (mesh)", "Flange", ["flange"],
             [Turn(USER_A)], check_flange_fill("flange"), active="flange", oracle=oracle_fill("flange")),
    Scenario("B", "same on the scan: mesh first, then fill all but the 5 big holes", "Flange", ["flange_scan"],
             [Turn("can you fill the holes in this model except for the 5 big holes")],
             check_flange_fill("flange_scan"), active="flange_scan", oracle=oracle_fill("flange_scan", True)),
    Scenario("C", "watertight mesh of a small scan and show it", "Bracket", ["bracket_scan"],
             [Turn("make a watertight mesh of this scan and show it")], check_new_mesh("bracket_scan"),
             active="bracket_scan", oracle=oracle_mesh("bracket_scan")),
    Scenario("D", "diameter of the centre hole (40 mm)", "Flange", ["flange"],
             [Turn("what is the diameter of the centre hole?")], check_answer_number(40.0, 0.5), active="flange",
             oracle=oracle_text("The centre bore is Ø 40.0 mm.")),
    Scenario("E", "where is scan-vs-CAD (Measure > Golden model)", "Flange", ["flange"],
             [Turn("where do I find the tool to check a scan against the CAD?")],
             check_answer_mentions(["measure", "golden model"], tool="app_guide"), active="flange",
             oracle=oracle_text("It is in Measure → Golden model.")),
    Scenario("F", "user's case with a viewer screenshot attached", "Flange", ["flange"],
             [Turn(USER_A, image="flange_png")], check_flange_fill("flange"), active="flange",
             oracle=oracle_fill("flange")),
    Scenario("G", "3 short turns (projects, show, count holes) then the user's case", "Flange", ["flange"],
             [Turn("list the projects"), Turn("show the flange"), Turn("how many holes does it have"), Turn(USER_A)],
             check_flange_fill("flange"), active=None, extra_projects=[("Bracket", ["bracket_scan"])],
             oracle=oracle_fill("flange")),
    Scenario("H", "check a scan against the golden model", "Stepped tube", ["tube_cad", "tube_scan"],
             [Turn("check this scan against the golden model")], check_created_operation("golden_check"),
             active="tube_scan", visible=["tube_scan", "tube_cad"], golden="tube_cad",
             oracle=oracle_golden("tube_scan")),
    # The user's failure as it happened: on a real scan mesh the 15 largest holes (all the holes tool lists) are the
    # design holes and unscanned patches, so no sticker hole is visible; a busier project; screenshot attached.
    # Success as A: only the 5 design holes stay open (patches, stickers and gaps are all filled).
    Scenario("I", "user's case, faithful: stickers hidden beyond the 15 listed holes, busy project, screenshot",
             "Flange", [*(f"busy_scan_{k}" for k in range(1, 5)), *(f"busy_clean_{k}" for k in range(1, 5)),
                        "busy_merge", "flange_real"],
             [Turn(USER_A, image="flange_real_png")], check_flange_fill("flange_real"), active="flange_real",
             truth="truth_real", oracle=oracle_fill("flange_real", max_diameter=21)),
]
BY_ID = {s.id: s for s in SCENARIOS}
