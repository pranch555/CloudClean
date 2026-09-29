# The assistant with a small model

CloudClean's assistant was tuned on a shared Qwen3.8-27B NVFP4 server (SGLang, owned by another project on the same
Spark, so it could not be reconfigured for CloudClean). That server has a **16k-token context window** and no tool-call
parser: the model writes its tool calls as text, and the client reads them from there (`assistant/llm.py`).

## The problem

The window has to hold everything below at once, before the model writes a word:

* the system prompt (about 2,100 tokens);
* the descriptions of all 28 tools (about 6,900 tokens);
* the state block;
* the conversation;
* any screenshot (about 1,000 tokens at 896 px);
* every tool result of the current turn.

That left about 7,000 tokens. On 2026-09-25 a user asked to "fill the small holes but keep the 5 big ones" on a
flange with 56 holes:

1. `holes list` showed 15 of the 56, and the model had no way to see the rest.
2. It asked again 7 times, repeating the same sentence each time.
3. The turn ended with "the conversation is too long".

## What a small window (24k tokens or less) gets now

* **Only the tools the turn needs** (`assistant/routing.py`):
  * a core set: `ui`, `app_guide`, `get_asset`, `workspace_overview`;
  * tools whose keywords appear in the message;
  * the tools of the current step;
  * tools used in the last two turns;
  * at most 12 tools in all. `app_guide` is sent without its list of feature ids.

  This typically takes 950 to 1,950 tokens instead of 6,900. The model can still call any tool by name.
* **A shorter state block:** 15 asset rows instead of 40.
* **No lookup loops** (`agent.py`):
  * Read-only calls are recognised. That covers the read-only tools (`measure`, `get_asset`, ...) and read-only
    actions (`holes list`, `project list`, `capture status`, ...).
  * The same call again is not run again, and a call whose result is identical to an earlier one in the turn is
    not shown again. Either way the model is told to act on what it has.
  * After 3 such repeats the turn continues without tools, and the model is asked to answer.
* **Photos a tool showed** (`look_at_photos`): at most 2, at 640 px, and only for the next step. Before, all
  of them went along with every later step of the turn.
* **A repeated preamble is taken back.** Small models restate the same sentence before every tool call. When a
  step's words match the previous step's (80 % the same), the chat panel removes them (event `retract`) and the
  conversation keeps an empty message instead.
* **A turn that outgrows the window is shortened, not failed:**
  * first overflow: fewer older turns;
  * second: this turn's earlier tool results are shortened, keeping the last two, and photos are dropped;
  * only a third overflow ends the turn, with a plain message that keeps the conversation.
* **The holes tool answers in one call:**
  * size groups, each with its count, diameter range and how many are round. Many round holes of one size are
    flagged as typical of marker-sticker spots;
  * ready `fill_options`, for example "`holes action=fill max_diameter=12.3` fills the 40 holes up to that size and
    keeps the 5 from 25 mm up".

  `fill` also takes `min_diameter` and `except_ids`. Diameters are now area-based: perimeter / π read 40 mm holes
  as 55 mm, because hole edges zig-zag along triangles.

## Deploying without cutting a chat off

`GET /api/assistant/busy` returns `{active_turns}`. The deploy's idle check waits until no capture, job or chat is
running. The server also closes open streams within 5 s on shutdown (`timeout_graceful_shutdown`); before, a
restart with an open chat waited for systemd's 90 s stop timeout and was then killed.

## Tests

`tests/test_assistant_loops.py` covers:

* the tool choice;
* the loop cut-off (replaying that holes loop);
* the shortening of a long turn;
* photos shown once;
* taking back a repeated preamble;
* the hole groups.

`tools/assistant_eval/` replays real conversations against the real model; it is run by hand, never by pytest.
