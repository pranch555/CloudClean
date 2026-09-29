"""Async client for OpenAI-compatible chat servers (vLLM, SGLang, Ollama, llama.cpp, LM Studio, NIM).

Only `httpx` is used. `LLMClient.chat` streams parsed events:

- `{"type": "content", "text"}` – answer text as it arrives
- `{"type": "reasoning", "text"}` – thinking (`reasoning_content` / `reasoning` fields or `<think>` tags)
- `{"type": "end", "content", "reasoning", "tool_calls": [{"id", "name", "arguments"}], "finish_reason",
  "usage"}` – once, after the stream finished (tool call arguments are the raw JSON string)

Failures raise `LLMError` with a sentence that can be shown to the user as is.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import AsyncIterator, Callable
from urllib.parse import urlsplit

import httpx


class LLMError(RuntimeError):
    """A problem talking to the model server, phrased for the user."""


class ToolsUnsupported(LLMError):
    """The server / model refused the `tools` parameter."""


class ContextTooLong(LLMError):
    """The prompt exceeds the model's context window."""


_TOOL_REJECTION = ("does not support tools", "support tool", "tool choice", "tool_choice", "tool-call-parser",
                   "enable-auto-tool-choice", "tools are not supported", "tool calling", "function calling",
                   "tools is not supported", "unsupported parameter: 'tools'", "unrecognized request argument")
_CONTEXT_ERRORS = ("maximum context length", "context length", "context window", "too many tokens",
                   "prompt is too long", "exceeds the model", "input is too long")


def normalize_base_url(url: str) -> str:
    """Accept `host:port`, `http://host:port`, `.../v1` or a pasted `.../v1/chat/completions`."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.I):
        url = "http://" + url
    for suffix in ("/chat/completions", "/completions", "/models"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    if urlsplit(url).path in ("", "/"):
        url += "/v1"
    return url


def _error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body.strip()[:500] or "no details"
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            msg = err.get("message") or err.get("detail") or err.get("msg")
            if msg:
                return str(msg)[:500]
        elif isinstance(err, str):
            return err[:500]
        if data.get("detail"):
            return str(data["detail"])[:500]
        if data.get("message"):
            return str(data["message"])[:500]
    return body.strip()[:500]


# --------------------------------------------------------------------------- text tag splitting
class TagSplitter:
    """Routes `<think>…</think>` in streamed content to reasoning and captures `<tool_call>…</tool_call>`
    blocks (models whose server has no tool parser write tool calls as text). Tags may be split across chunks."""

    OPEN = {"<think>": "reasoning", "<tool_call>": "tool"}
    CLOSE = {"reasoning": "</think>", "tool": "</tool_call>"}

    def __init__(self):
        self.buf = ""
        self.mode = "content"
        self.tool_texts: list[str] = []
        self._tool = ""

    @staticmethod
    def _holdback(text: str, tags) -> int:
        """Length of the longest suffix of text that is a proper prefix of one of the tags."""
        best = 0
        for tag in tags:
            for n in range(min(len(tag) - 1, len(text)), 0, -1):
                if text.endswith(tag[:n]):
                    best = max(best, n)
                    break
        return best

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.buf += text
        out: list[tuple[str, str]] = []
        while self.buf:
            if self.mode == "content":
                tags = [*self.OPEN, "</think>"]
                hits = [(self.buf.find(t), t) for t in tags if t in self.buf]
                if not hits:
                    keep = self._holdback(self.buf, tags)
                    emit, self.buf = self.buf[: len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                    if emit:
                        out.append(("content", emit))
                    break
                pos, tag = min(hits)
                if pos:
                    out.append(("content", self.buf[:pos]))
                self.buf = self.buf[pos + len(tag):]
                if tag != "</think>":  # a stray closing tag (template opened <think> itself) is dropped
                    self.mode = self.OPEN[tag]
            else:
                close = self.CLOSE[self.mode]
                pos = self.buf.find(close)
                if pos < 0:
                    keep = self._holdback(self.buf, [close])
                    emit, self.buf = self.buf[: len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                    self._route(emit, out)
                    break
                self._route(self.buf[:pos], out)
                self.buf = self.buf[pos + len(close):]
                if self.mode == "tool":
                    self.tool_texts.append(self._tool)
                    self._tool = ""
                self.mode = "content"
        return out

    def _route(self, text: str, out: list) -> None:
        if not text:
            return
        if self.mode == "reasoning":
            out.append(("reasoning", text))
        else:
            self._tool += text

    def flush(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        rest, self.buf = self.buf, ""
        if self.mode == "content" and rest:
            out.append(("content", rest))
        elif self.mode == "reasoning" and rest:
            out.append(("reasoning", rest))
        elif self.mode == "tool":
            self._tool += rest
            if self._tool.strip():
                self.tool_texts.append(self._tool)
            self._tool = ""
        return out


def parse_text_tool_call(text: str) -> dict | None:
    """Parse a tool call written as text: Hermes JSON `{"name", "arguments"}` or Qwen XML
    `<function=name><parameter=x>value</parameter></function>`."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("name"):
            args = data.get("arguments", data.get("parameters", {}))
            return {"name": str(data["name"]),
                    "arguments": args if isinstance(args, str) else json.dumps(args)}
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"<function=([\w.\-]+)>(.*?)(?:</function>|$)", text, re.S)
    if not m:
        return None
    args = {}
    for p in re.finditer(r"<parameter=([\w.\-]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", m.group(2), re.S):
        raw = p.group(2).strip()
        try:
            args[p.group(1)] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            args[p.group(1)] = raw
    return {"name": m.group(1), "arguments": json.dumps(args)}


# --------------------------------------------------------------------------- stream assembly
class StreamAssembler:
    """Accumulates chat.completion.chunk deltas (or a whole non-streamed response)."""

    def __init__(self):
        self.splitter = TagSplitter()
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: list[dict] = []
        self._by_index: dict = {}
        self.finish_reason = None
        self.usage = None

    def _text(self, kind: str, text: str) -> list[dict]:
        (self.content if kind == "content" else self.reasoning).append(text)
        return [{"type": kind, "text": text}]

    def _tool_delta(self, tc: dict, position: int) -> None:
        fn = tc.get("function") or {}
        index = tc.get("index", position)
        slot = self._by_index.get(index)
        new_id = tc.get("id")
        if slot is not None and new_id and slot["id"] and new_id != slot["id"] and fn.get("name"):
            slot = None  # some servers reuse index 0 for every call
        if slot is None:
            slot = {"id": new_id or "", "name": "", "arguments": ""}
            self.calls.append(slot)
            self._by_index[index] = slot
        if new_id and not slot["id"]:
            slot["id"] = new_id
        if fn.get("name"):
            slot["name"] = fn["name"] if not slot["name"] or slot["name"] == fn["name"] else slot["name"] + fn["name"]
        args = fn.get("arguments")
        if isinstance(args, (dict, list)):  # non-standard servers send an object instead of a JSON string
            slot["arguments"] = json.dumps(args)
        elif args:
            slot["arguments"] += args

    def feed_chunk(self, chunk: dict) -> list[dict]:
        out: list[dict] = []
        if not isinstance(chunk, dict):
            return out
        if chunk.get("error"):
            err = chunk["error"]
            raise LLMError(f"The LLM server reported an error: {err.get('message', err) if isinstance(err, dict) else err}")
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or choice.get("message") or {}
            for key in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(key), str) and delta[key]:
                    out += self._text("reasoning", delta[key])
                    break
            if isinstance(delta.get("content"), str) and delta["content"]:
                for kind, text in self.splitter.feed(delta["content"]):
                    out += self._text(kind, text)
            for pos, tc in enumerate(delta.get("tool_calls") or []):
                self._tool_delta(tc, pos)
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return out

    def finish(self) -> dict:
        out = [e for kind, text in self.splitter.flush() for e in self._text(kind, text)]
        calls = [c for c in self.calls if c["name"]]
        if not calls:
            for text in self.splitter.tool_texts:
                parsed = parse_text_tool_call(text)
                if parsed:
                    calls.append({"id": "", **parsed})
        for c in calls:
            c["id"] = c["id"] or f"call_{uuid.uuid4().hex[:12]}"
            c["arguments"] = c["arguments"] or "{}"
        self._trailing = out
        return {"type": "end", "content": "".join(self.content), "reasoning": "".join(self.reasoning),
                "tool_calls": calls, "finish_reason": self.finish_reason, "usage": self.usage}


# --------------------------------------------------------------------------- client
class LLMClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 300.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key or ""
        self.timeout = float(timeout or 300.0)
        self.transport = transport

    def _client(self, timeout: float | None = None) -> httpx.AsyncClient:
        t = float(timeout or self.timeout)
        headers = {"Accept": "application/json, text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return httpx.AsyncClient(base_url=self.base_url + "/", headers=headers, transport=self.transport,
                                 timeout=httpx.Timeout(t, connect=min(10.0, t)))

    def _friendly(self, exc: Exception, timeout: float | None = None) -> LLMError:
        where = self.base_url
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return LLMError(f"Cannot reach the LLM server at {where} — is vLLM/Ollama running?")
        if isinstance(exc, httpx.TimeoutException):
            return LLMError(f"The LLM server at {where} did not answer within {timeout or self.timeout:.0f} s. "
                            "It may still be loading the model or be overloaded (raise request_timeout in settings).")
        if isinstance(exc, (httpx.RemoteProtocolError, httpx.ReadError)):
            return LLMError(f"The connection to the LLM server at {where} was dropped ({exc}). "
                            "The server may have crashed or run out of memory.")
        if isinstance(exc, httpx.UnsupportedProtocol) or isinstance(exc, httpx.InvalidURL):
            return LLMError(f"Invalid LLM server URL '{where}' - use e.g. http://localhost:8000/v1")
        return LLMError(f"LLM request to {where} failed: {exc}")

    def _http_error(self, status: int, msg: str, model: str = "") -> LLMError:
        low = msg.lower()
        if status in (401, 403):
            return LLMError(f"The LLM server rejected the request (HTTP {status}: {msg}). Check the API key.")
        if any(k in low for k in _CONTEXT_ERRORS):
            return ContextTooLong(f"The conversation is too long for the model's context window ({msg}).")
        if status == 404:
            return LLMError(f"The LLM server returned HTTP 404 ({msg}). Check the model name "
                            f"('{model or 'auto'}') and that the URL ends with /v1.")
        return LLMError(f"The LLM server returned HTTP {status}: {msg}")

    async def context_length(self, model: str, timeout: float | None = None) -> int | None:
        """The context window of `model` in tokens when the server says (max_model_len / context_length), else None."""
        try:
            async with self._client(timeout) as client:
                resp = await client.get("models")
            data = resp.json() if resp.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return None
        items = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
        for m in items or []:
            if isinstance(m, dict) and model in (m.get("id"), m.get("name"), m.get("root")):
                for k in ("max_model_len", "context_length", "max_context_length"):
                    if isinstance(m.get(k), int) and m[k] > 0:
                        return int(m[k])
        return None

    async def list_models(self, timeout: float | None = None) -> list[str]:
        try:
            async with self._client(timeout) as client:
                resp = await client.get("models")
        except httpx.HTTPError as exc:
            raise self._friendly(exc, timeout) from exc
        if resp.status_code != 200:
            raise self._http_error(resp.status_code, _error_message(resp.text))
        try:
            data = resp.json()
        except ValueError:
            raise LLMError(f"{self.base_url}/models did not return JSON - is this an OpenAI-compatible server?")
        items = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
        return [str(m.get("id") or m.get("name")) for m in items or [] if isinstance(m, dict)
                and (m.get("id") or m.get("name"))]

    async def chat(self, messages: list[dict], model: str, tools: list[dict] | None = None,
                   temperature: float | None = None, max_tokens: int | None = None,
                   should_stop: Callable[[], bool] | None = None, thinking: bool | None = None) -> AsyncIterator[dict]:
        body: dict = {"model": model, "messages": messages, "stream": True,
                      "stream_options": {"include_usage": True}}
        if thinking is not None:
            # Qwen3 / hybrid reasoning models: vLLM and SGLang pass this to the chat template; other servers ignore it.
            body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        for _attempt in range(3):
            asm = StreamAssembler()
            try:
                async with self._client() as client:
                    async with client.stream("POST", "chat/completions", json=body) as resp:
                        if resp.status_code != 200:
                            msg = _error_message((await resp.aread()).decode("utf-8", "replace"))
                            if "stream_options" in body and "stream_options" in msg:
                                body.pop("stream_options")  # older servers reject it
                                continue
                            if tools and body.get("tool_choice") == "auto" and "auto" in msg.lower() and \
                                    "tool" in msg.lower():
                                # vLLM without --enable-auto-tool-choice: with tool_choice "none" it still renders
                                # the tools into the chat template, and Qwen-style models then write <tool_call>
                                # blocks as text, which TagSplitter / parse_text_tool_call turn into tool calls.
                                body["tool_choice"] = "none"
                                continue
                            if tools and resp.status_code in (400, 404, 422, 500, 501) and \
                                    any(k in msg.lower() for k in _TOOL_REJECTION):
                                raise ToolsUnsupported(msg)
                            raise self._http_error(resp.status_code, msg, model)
                        if "event-stream" not in resp.headers.get("content-type", "") and \
                                "json" in resp.headers.get("content-type", ""):
                            # server ignored stream=true and returned the whole completion
                            try:
                                chunk = json.loads(await resp.aread())
                            except ValueError:
                                raise LLMError("The LLM server returned invalid JSON")
                            for event in asm.feed_chunk(chunk):
                                yield event
                        else:
                            async for event in self._sse(resp, asm, should_stop):
                                yield event
            except LLMError:
                raise
            except httpx.HTTPError as exc:
                raise self._friendly(exc) from exc
            end = asm.finish()
            for event in asm._trailing:
                yield event
            yield end
            return

    @staticmethod
    async def _sse(resp: httpx.Response, asm: StreamAssembler, should_stop) -> AsyncIterator[dict]:
        data_lines: list[str] = []

        def parse(payload: str):
            if payload.strip() == "[DONE]":
                return None
            try:
                return json.loads(payload)
            except ValueError:
                return {}

        async for line in resp.aiter_lines():
            if should_stop is not None and should_stop():
                return
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
                # most servers send one JSON object per data line; parse eagerly when it is complete
                chunk = parse("\n".join(data_lines))
                if chunk is None:
                    return
                if chunk:
                    data_lines = []
                    for event in asm.feed_chunk(chunk):
                        yield event
                continue
            if not line.strip():
                if data_lines:
                    chunk = parse("\n".join(data_lines))
                    data_lines = []
                    if chunk is None:
                        return
                    for event in asm.feed_chunk(chunk or {}):
                        yield event
            elif line.lstrip().startswith("{"):  # NDJSON servers without the data: prefix
                chunk = parse(line)
                for event in asm.feed_chunk(chunk or {}):
                    yield event
