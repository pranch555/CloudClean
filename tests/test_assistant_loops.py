"""The assistant with a small model (a 16k window, like the shared Qwen server): which tools it is shown, what stops
it looping on the same lookup, and how a turn that outgrows the window is shortened instead of failing.

Background: a user asked it to fill a flange's sticker holes but keep the 5 big ones. The model listed the holes 7
times (the list showed 15 of 56), repeated itself, and the turn ended with "the conversation is too long"."""
from __future__ import annotations

import json

import numpy as np
import open3d as o3d
import pytest

from cloudclean.assistant import routing
from cloudclean.assistant.agent import is_read_only, trim_history
from cloudclean.assistant.llm import LLMClient
from cloudclean.assistant.tools import TOOLS
from cloudclean.holes import fill_splits, find_holes, hole_groups
from tests.test_assistant import chat, env, llm, of, text_chunks, tool_chunks  # noqa: F401 (fixtures)


def plate_with_holes() -> o3d.geometry.TriangleMesh:
    """A 150 mm plate: a Ø40 bore and 4 Ø25 holes around it, 30 sticker spots Ø6 and 10 tiny gaps Ø1.6."""
    n = 250
    xs = np.linspace(-75, 75, n)
    X, Y = np.meshgrid(xs, xs)
    V = np.c_[X.ravel(), Y.ravel(), np.zeros(n * n)]
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F = np.vstack([np.c_[a, b, c], np.c_[a, c, d]])
    cen = V[F].mean(1)[:, :2]
    rng = np.random.default_rng(0)
    circles = [((0.0, 0.0), 20.0)] + [((45 * np.cos(t), 45 * np.sin(t)), 12.5)
                                      for t in np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4]
    for count, r, gap in ((35, 3.0, 6), (45, 0.8, 4)):
        while len(circles) < count:
            p = rng.uniform(-68, 68, 2)
            if all(np.hypot(*(p - np.array(q))) > rr + gap for q, rr in circles):
                circles.append((tuple(p), r))
    cut = np.zeros(len(F), dtype=bool)
    for q, r in circles:
        cut |= np.hypot(cen[:, 0] - q[0], cen[:, 1] - q[1]) < r
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F[~cut]))
    mesh.remove_unreferenced_vertices()
    return mesh


def test_holes_are_grouped_by_size_with_a_ready_fill():
    groups = hole_groups(find_holes(plate_with_holes())["holes"])
    assert [g["count"] for g in groups] == [5, 30, 10]
    assert groups[0]["diameter_min"] == pytest.approx(25, abs=1) and groups[0]["diameter_max"] == pytest.approx(40, abs=1)
    assert groups[1]["round"] == 30 and "marker stickers" in groups[1]["note"]
    split = fill_splits(groups)[0]
    assert (split["fills"], split["keeps"]) == (40, 5) and 6.5 < split["max_diameter"] < 24


def test_the_small_window_tool_choice():
    names = routing.select_tools(list(TOOLS), "fill all the small holes but keep the 4 big ones and the centre hole",
                                 "mesh", [])
    assert {"ui", "app_guide", "get_asset", "holes", "mesh"} <= set(names)
    assert "turntable" not in names and "merge" not in names and len(names) <= routing.MAX_SMALL
    assert set(routing.select_tools(list(TOOLS), "hi", "home", [])) == set(routing.CORE)
    assert "holes" in routing.select_tools(list(TOOLS), "and now the other ones", "home", ["holes"])
    slim = routing.schemas_for(TOOLS, ["app_guide"], small=True)[0]["function"]["description"]
    full = routing.schemas_for(TOOLS, ["app_guide"], small=False)[0]["function"]["description"]
    assert slim == routing.SLIM_APP_GUIDE and len(full) > 4 * len(slim)
    msgs = [{"role": "user", "content": "a"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "mesh"}}]},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "holes"}}]},
            {"role": "user", "content": "c"}]
    assert routing.recent_tools(msgs) == ["holes", "mesh"]


def test_what_counts_as_a_lookup():
    assert is_read_only("holes", {"action": "list"}) and is_read_only("holes", {})
    assert not is_read_only("holes", {"action": "fill"}) and is_read_only("measure", {"kind": "distance"})
    assert is_read_only("project", {"action": "list"}) and not is_read_only("project", {"action": "create"})


def test_a_turn_that_outgrows_the_window_shortens_its_own_tool_results():
    big = "x" * 5000
    msgs = [{"role": "user", "content": "fill"}] + [
        m for i in range(4) for m in ({"role": "assistant", "content": "", "tool_calls": [{"id": str(i)}]},
                                      {"role": "tool", "tool_call_id": str(i), "content": big})]
    kept = trim_history(msgs, keep_turns=3, full_tool_turns=1, max_chars=24_000, step=1, compact_current=True)
    sizes = [len(m["content"]) for m in kept if m["role"] == "tool"]
    assert sizes[:2] == [300 + len(" ...(shortened: the turn got too long)")] * 2 and sizes[2:] == [5000, 5000]
    assert all(len(m["content"]) == 5000 for m in trim_history(msgs) if m["role"] == "tool")


@pytest.fixture
def small_model(monkeypatch):
    async def window(self, model, timeout=None):
        return 16384
    monkeypatch.setattr(LLMClient, "context_length", window)


def test_a_small_model_sees_only_the_tools_it_needs(env, small_model):
    client, _, _, llm = env
    llm.scripts.append(lambda body: text_chunks("Sure."))
    chat(client, "fill all the small holes of this mesh", context={"step": "mesh"})
    names = [t["function"]["name"] for t in llm.requests[0]["tools"]]
    assert "holes" in names and "ui" in names and "turntable" not in names and len(names) <= routing.MAX_SMALL
    guide = next(t for t in llm.requests[0]["tools"] if t["function"]["name"] == "app_guide")
    assert guide["function"]["description"] == routing.SLIM_APP_GUIDE


def test_the_listing_loop_is_cut_short(env, small_model):
    """The model asks for the same hole list again and again (with arguments the list ignores, as it did): it is not
    shown the same answer twice, and after 3 repeats the turn goes on without tools."""
    client, ws, _, llm = env
    plate = ws.add_geometry(plate_with_holes(), "flange mesh", "import")
    for k, extra in enumerate([{}, {}, {"name": "x"}, {}, {"name": "y"}]):
        llm.scripts.append(lambda body, k=k, extra=extra: tool_chunks(
            "holes", {"action": "list", "asset_id": plate["id"], **extra}, f"call_{k}"))
    llm.scripts.append(lambda body: text_chunks("I will fill the 40 small holes and keep the 5 big ones."))
    events = chat(client, "fill all the small sticker holes, keep the 5 big ones", context={"step": "mesh"})

    ends = of(events, "tool_end")
    assert "5 holes" in ends[0]["summary"] and "30 sticker spots" in ends[0]["summary"]   # size groups, not 15 of 56
    first = json.loads([m for m in llm.requests[1]["messages"] if m["role"] == "tool"][0]["content"])
    assert first["fill_options"][0]["call"].startswith("holes action=fill max_diameter=")
    assert "not run again" in ends[1]["summary"] and "same result" in ends[2]["summary"]
    later = [m for m in llm.requests[-1]["messages"] if m["role"] == "tool"]
    assert all("will not change" in m["content"] for m in later[1:])
    assert "tools" not in llm.requests[-1] or not llm.requests[-1]["tools"]          # the loop ended tool use
    assert "You repeated the same lookup" in llm.requests[-1]["messages"][-1]["content"]
    assert not of(events, "error")


def text_then_tool(text: str, name: str, arguments: dict, call_id: str) -> list[dict]:
    from tests.test_assistant import chunk
    return [chunk({"role": "assistant", "content": ""}), chunk({"content": text}),
            chunk({"tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                   "function": {"name": name, "arguments": json.dumps(arguments)}}]}),
            chunk({}, "tool_calls")]


def test_a_repeated_preamble_is_taken_back(env, small_model):
    """Small models restate the same sentence before every tool call: the repeat is taken back out of the answer
    the user sees and out of the conversation the model reads."""
    client, ws, _, llm = env
    plate = ws.add_geometry(plate_with_holes(), "flange mesh", "import")
    said = "I can see the flange in your photos. Let me look at the holes."
    llm.scripts.append(lambda body: text_then_tool(said, "get_asset", {"asset_id": plate["id"]}, "c1"))
    llm.scripts.append(lambda body: text_then_tool(said.replace("look at", "check"), "holes",
                                                   {"action": "list", "asset_id": plate["id"]}, "c2"))
    llm.scripts.append(lambda body: text_chunks("There are 5 big holes and 40 small ones."))
    events = chat(client, "fill the small holes", context={"step": "mesh"})
    shown = "".join(d["text"] for d in of(events, "delta"))
    taken = sum(d["chars"] for d in of(events, "retract"))
    assert taken == len(said.replace("look at", "check"))
    assert shown.count("I can see the flange") == 2                         # streamed twice, one taken back
    stored = [m for m in llm.requests[-1]["messages"] if m["role"] == "assistant"]
    assert stored[0]["content"].startswith("I can see the flange") and stored[1]["content"] == ""


def test_photos_are_shown_once_in_a_small_window(env, small_model, tmp_path):
    from PIL import Image
    client, ws, _, llm = env
    assert client.put("/api/assistant/settings", json={"vision": True}).status_code == 200
    for k in range(3):
        src = tmp_path / f"p{k}.png"
        Image.new("RGB", (1200, 900), (40 * k, 90, 160)).save(src)
        ws.add_image(src, f"flange photo {k}")
    llm.scripts.append(lambda body: tool_chunks("look_at_photos", {}, "c1"))
    llm.scripts.append(lambda body: tool_chunks("workspace_overview", {}, "c2"))
    llm.scripts.append(lambda body: text_chunks("Done."))
    chat(client, "look at my photos of the flange and tell me what you see")

    def pictures(body):
        return [p for m in body["messages"] if isinstance(m.get("content"), list)
                for p in m["content"] if p.get("type") == "image_url"]
    assert len(pictures(llm.requests[1])) == 2                              # the next step: at most 2
    assert pictures(llm.requests[2]) == []                                  # and only that step
