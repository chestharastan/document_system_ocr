"""LLM providers for the structure engine (contract §2.4): one JSON answer per page from an OpenAI-compatible endpoint
(the owner's llama-server, vLLM, ...) or from Claude.

Configuration is server-side only, from the environment (never from the UI):
    LLM_PROVIDER   openai | anthropic | none          (default openai: the local llama-server)
    LLM_BASE_URL   OpenAI-compatible base URL          (default http://127.0.0.1:8082/v1)
    LLM_MODEL      model id                            (default qwen3-14b; claude-opus-5-5 for anthropic)
    LLM_API_KEY    bearer token / Anthropic key        (optional; the Anthropic SDK also reads ANTHROPIC_API_KEY)
    LLM_TIMEOUT_S  per-request timeout in seconds      (default 120; 600 for a 14B model partly on the CPU)
    LLM_EXTRA_BODY JSON merged into every OpenAI-compatible request body (e.g. '{"top_k": 20}'), optional
    LLM_GRAMMAR    auto | gbnf | schema: how the answer shape is enforced on an OpenAI-compatible server (auto = a
                   whitespace-free GBNF grammar when /models says llama.cpp, else response_format json_schema)
    LLM_WAIT_IF_BUSY 1 = queue behind a request that is already running on a single-slot llama-server (default 0:
                   the app falls back to the phone tier at once; the teacher labeller waits for an idle slot itself)
    LLM_ALLOW_LOAD 1 = allow requests to a model a llama-server router lists as "unloaded" (default 0: such a model
                   counts as unreachable, so the engine never makes a shared server swap models)
Engine limits (llm_structure): LLM_MAX_LINES (lines per call, 150), LLM_MAX_PAGE_LINES (900), LLM_PROMPT_CHARS
(line-text characters per call, 7000), LLM_PAGE_BUDGET_S (wall time per page, default 2.5 x LLM_TIMEOUT_S).

``get_provider()`` builds (and caches) the configured provider, ``llm_status()`` is what /pipeline/info and
/llm/status report. Every provider has ``complete_json(system, user, schema, max_tokens) -> dict`` (the parsed JSON,
schema-constrained on the server where the API allows it, retried once with a repair prompt when the text is not
JSON), ``reachable()`` (cheap probe, cached a few seconds) and ``info()``. Token usage of the last call is in
``provider.last_usage`` ({"prompt_tokens", "completion_tokens"})."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager

import httpx

log = logging.getLogger("app.llm")

DEFAULT_BASE_URL = "http://127.0.0.1:8082/v1"
DEFAULT_OPENAI_MODEL = "qwen3-14b"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_TIMEOUT_S = 120.0
PROBE_TTL_S = 10.0          # reachability probes are cached this long (/pipeline/info is polled by the UI)
OFF = {"", "none", "off", "no", "0", "false", "disabled"}


class LLMError(RuntimeError):
    """The provider could not deliver a JSON answer (unreachable, HTTP error, refusal, malformed twice)."""


class LLMTruncated(LLMError):
    """The answer hit max_tokens: the caller may retry with a larger budget or salvage ``partial`` (the cut text)."""

    def __init__(self, msg: str, partial: str = ""):
        super().__init__(msg)
        self.partial = partial


def parse_partial_json(text: str) -> dict | None:
    """A JSON object cut off mid-way (max_tokens) -> the longest prefix that ends after a complete value, closed with
    the brackets still open (so complete elements survive and the cut one is dropped). None when nothing parses."""
    if not isinstance(text, str):
        return None
    t = _THINK_RE.sub("", text)
    start = t.find("{")
    if start < 0:
        return None
    t = t[start:]
    cuts, stack, in_str, esc = [], [], False, False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}":
            if stack:
                stack.pop()
            cuts.append((i + 1, "".join(reversed(stack))))
    for end, closers in reversed(cuts[-400:]):
        try:
            out = json.loads(t[:end].rstrip().rstrip(",") + closers)
        except ValueError:
            continue
        if isinstance(out, dict):
            return out
    return None


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_json_text(text: str) -> dict:
    """Model text -> dict. Tolerates code fences, a leaked <think> block and prose around the object."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty answer")
    t = _THINK_RE.sub("", text)
    t = _FENCE_RE.sub("", t).strip()
    try:
        out = json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("no JSON object in the answer")
        out = json.loads(t[a:b + 1])  # raises JSONDecodeError (a ValueError)
    if not isinstance(out, dict):
        raise ValueError("answer is not a JSON object")
    return out


def _repair_prompt(err: Exception) -> str:
    return (f"Your previous answer was not valid JSON ({err}). Return the same content again as ONE valid JSON object "
            f"matching the schema, with no code fences, comments or text around it.")


# ------------------------------------------------------------------------------------------------ compact grammar
_GBNF_BASE = r'''
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
string ::= "\"" char* "\""
integer ::= "-"? ([0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9] | [1-9] [0-9] [0-9] [0-9] | [1-9] [0-9] [0-9] [0-9] [0-9])
boolean ::= "true" | "false"
'''


def schema_to_gbnf(schema: dict) -> str:
    """JSON schema (the subset the structure engine uses: object with fixed required properties, array, string with
    enum, integer, boolean, anyOf) -> a llama.cpp GBNF grammar that admits only MINIFIED JSON: no whitespace at all.
    llama.cpp's own json_schema grammar allows free whitespace and models then pretty-print (a newline plus
    indentation costs a token per value), which made answers ~2-3x longer than needed on a slow server."""
    rules: dict[str, str] = {}

    def lit(t: str) -> str:
        return '"' + t.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def visit(node: dict, name: str) -> str:
        if "anyOf" in node:
            alts = [visit(alt, f"{name}-{k}") for k, alt in enumerate(node["anyOf"])]
            rules[name] = " | ".join(alts)
            return name
        t = node.get("type")
        if "enum" in node:
            rules[name] = " | ".join(lit(json.dumps(v, ensure_ascii=False)) for v in node["enum"])
            return name
        if t == "object":
            props = node.get("properties", {})
            parts = []
            for k, (key, sub) in enumerate(props.items()):
                ref = visit(sub, f"{name}-{key}")
                parts.append(('"," ' if k else "") + lit(json.dumps(key)) + ' ":" ' + ref)
            rules[name] = '"{" ' + " ".join(parts) + ' "}"' if parts else '"{}"'
            return name
        if t == "array":
            item = visit(node.get("items", {"type": "string"}), f"{name}-item")
            rules[name] = f'"[" ({item} ("," {item})*)? "]"'
            return name
        if t in ("string", "integer", "boolean"):
            return t
        raise ValueError(f"schema node not supported by the compact grammar: {node}")
    root = visit(schema, "root")
    if root != "root":
        rules["root"] = root
    return "\n".join(f"{k} ::= {v}" for k, v in rules.items()) + _GBNF_BASE


class LLMProvider:
    """Interface. ``name`` is the provider key reported in the status."""

    name = "base"
    model = ""

    def __init__(self):
        self._local = threading.local()   # per-thread call deadline (see ``deadline``)
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0}
        self.stats: dict = {"calls": 0, "repairs": 0, "truncated": 0, "failed": 0}  # over the provider's lifetime
        self._probe: tuple[float, bool] | None = None

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        raise NotImplementedError

    def reachable(self) -> bool:
        """Cached probe (PROBE_TTL_S); subclasses implement ``_probe_once``."""
        now = time.monotonic()
        if self._probe is not None and now - self._probe[0] < PROBE_TTL_S:
            return self._probe[1]
        try:
            ok = bool(self._probe_once())
        except Exception as exc:  # noqa: BLE001 - a probe never raises
            log.debug("llm probe failed: %r", exc)
            ok = False
        self._probe = (now, ok)
        return ok

    def _probe_once(self) -> bool:
        return False

    def busy(self) -> bool | None:
        """True when the server is already generating for someone else (all slots taken), None when unknown."""
        return None

    @contextmanager
    def deadline(self, seconds: float | None):
        """Calls made by this thread inside the block give up after ``seconds`` in total (a synchronous web request
        must not hang for the full LLM_TIMEOUT_S). None = no extra limit."""
        prev = getattr(self._local, "deadline", None)
        self._local.deadline = (time.monotonic() + float(seconds)) if seconds else None
        try:
            yield
        finally:
            self._local.deadline = prev

    @contextmanager
    def stop_when(self, check):
        """While inside: a streamed answer is checked every ~16 chunks with ``check(text_so_far)``; True stops the
        generation (the server cancels it) and raises LLMTruncated with the partial text (degenerate loops)."""
        prev = getattr(self._local, "stop_check", None)
        self._local.stop_check = check
        try:
            yield
        finally:
            self._local.stop_check = prev

    def _time_left(self, default: float) -> float:
        dl = getattr(getattr(self, "_local", None), "deadline", None)
        if dl is None:
            return default
        left = dl - time.monotonic()
        if left <= 0:
            raise LLMError("deadline reached before the LLM call")
        return min(default, left)

    def info(self) -> dict:
        return {"provider": self.name, "model": self.model}


class OpenAICompatProvider(LLMProvider):
    """Chat completions on an OpenAI-compatible server. The llama.cpp server turns ``response_format.json_schema`` into
    a grammar (so the answer is valid JSON of that shape) and ``chat_template_kwargs.enable_thinking=false`` switches
    Qwen3 thinking off; other servers ignore what they do not know."""

    name = "openai"

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_OPENAI_MODEL, api_key: str | None = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S, transport=None, extra_body: dict | None = None, grammar: str = "auto"):
        super().__init__()
        self.grammar_mode = grammar   # auto (GBNF when the server is llama.cpp) | gbnf | schema
        self._is_llamacpp: bool | None = None
        self._gbnf_cache: dict = {}
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.extra_body = extra_body or {}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, transport=transport,
                                    timeout=httpx.Timeout(self.timeout_s, connect=5.0))

    def _probe_once(self) -> bool:
        r = self._client.get("/models", timeout=httpx.Timeout(3.0, connect=2.0))
        if r.status_code != 200:
            return False
        try:
            data = r.json().get("data", [])
            ids = [m.get("id") for m in data]
        except (ValueError, AttributeError):
            return True  # not JSON (a proxy): the endpoint answers, assume the model is there
        if ids and self.model not in ids:
            log.warning("llm model %r is not served at %s (available: %s)", self.model, self.base_url, ids)
        self._is_llamacpp = any(isinstance(m, dict) and m.get("owned_by") == "llamacpp" for m in data)
        for m in data:
            status = m.get("status") if isinstance(m, dict) else None
            if m.get("id") == self.model and isinstance(status, dict) and status.get("value") not in (None, "loaded") \
                    and os.environ.get("LLM_ALLOW_LOAD", "0") not in ("1", "true", "yes"):
                # llama-server router mode: a request would load this model and evict the one in use
                log.warning("llm model %r is %s at %s: not used (LLM_ALLOW_LOAD=1 allows loading it)",
                            self.model, status.get("value"), self.base_url)
                return False
        return True

    def busy(self) -> bool | None:
        """llama.cpp ``GET /slots`` (router mode needs ?model=): every slot processing -> True. A single-slot server
        then queues our request behind the running one (minutes on a slow laptop), so interactive callers fall back to
        the phone tier instead (LLM_WAIT_IF_BUSY=1 waits). None when the endpoint does not exist or says nothing."""
        try:
            r = self._client.get("/slots", params={"model": self.model}, timeout=httpx.Timeout(3.0, connect=2.0))
            if r.status_code != 200:
                r = self._client.get(self.base_url.rsplit("/v1", 1)[0] + "/slots", params={"model": self.model},
                                     timeout=httpx.Timeout(3.0, connect=2.0))
            if r.status_code != 200:
                return None
            slots = r.json()
        except (httpx.HTTPError, ValueError):
            return None
        if not isinstance(slots, list) or not slots:
            return None
        return all(isinstance(sl, dict) and sl.get("is_processing") for sl in slots)

    def _use_gbnf(self) -> bool:
        if self.grammar_mode == "gbnf":
            return True
        if self.grammar_mode != "auto":
            return False
        if self._is_llamacpp is None:
            self._probe = None
            self.reachable()
        return bool(self._is_llamacpp)

    def _request(self, messages: list, schema: dict, max_tokens: int) -> tuple[str, str]:
        body = {"model": self.model, "messages": messages, "max_tokens": int(max_tokens), "temperature": 0,
                "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        if self._use_gbnf():
            key = json.dumps(schema, sort_keys=True)
            if key not in self._gbnf_cache:
                self._gbnf_cache[key] = schema_to_gbnf(schema)
            body["grammar"] = self._gbnf_cache[key]   # llama.cpp: minified JSON only (see schema_to_gbnf)
        else:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}}
        body.update(self.extra_body)
        body["stream"], body["stream_options"] = True, {"include_usage": True}
        self.stats["calls"] += 1
        limit = self._time_left(self.timeout_s)
        t_end = time.monotonic() + limit
        try:
            # streamed: we can stop at the deadline ourselves, and closing the stream makes llama-server cancel the
            # generation (a non-streamed request that timed out on our side may keep the only slot busy)
            with self._client.stream("POST", "/chat/completions", json=body,
                                     timeout=httpx.Timeout(limit, connect=5.0)) as r:
                if r.status_code != 200:
                    r.read()
                    self.stats["failed"] += 1
                    raise LLMError(f"HTTP {r.status_code} from {self.base_url}: {r.text[:300]}")
                if "text/event-stream" not in (r.headers.get("content-type") or ""):
                    r.read()
                    return self._parse_full(r)
                parts, finish, usage, tm = [], "", {}, {}
                check = getattr(self._local, "stop_check", None)
                for n_ev, line in enumerate(r.iter_lines()):
                    if check is not None and n_ev % 16 == 15 and check("".join(parts)):
                        self.stats["truncated"] += 1
                        self._set_usage({"prompt_tokens": 0, "completion_tokens": n_ev}, {})
                        raise LLMTruncated("answer stopped: it repeats line ids (degenerate loop)", "".join(parts))
                    if time.monotonic() > t_end:
                        self.stats["failed"] += 1
                        raise LLMError(f"timeout after {limit:.0f}s talking to {self.base_url} (generation cancelled)")
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        ev = json.loads(payload)
                    except ValueError:
                        continue
                    for ch in ev.get("choices") or []:
                        parts.append((ch.get("delta") or {}).get("content") or "")
                        finish = ch.get("finish_reason") or finish
                    usage = ev.get("usage") or usage
                    tm = ev.get("timings") or tm
        except httpx.TimeoutException as exc:
            self.stats["failed"] += 1
            raise LLMError(f"timeout after {limit:.0f}s talking to {self.base_url}") from exc
        except httpx.HTTPError as exc:
            self.stats["failed"] += 1
            raise LLMError(f"cannot reach {self.base_url}: {exc}") from exc
        self._set_usage(usage, tm)
        return "".join(parts), finish

    def _set_usage(self, usage: dict, tm: dict) -> None:
        self.last_usage = {"prompt_tokens": int(usage.get("prompt_tokens") or 0),
                           "completion_tokens": int(usage.get("completion_tokens") or 0)}
        if tm:  # llama.cpp extra: prompt vs generation split
            self.last_usage["server_timings"] = {k: round(float(tm[k]), 1) for k in ("prompt_ms", "predicted_ms", "prompt_per_second", "predicted_per_second") if k in tm}

    def _parse_full(self, r) -> tuple[str, str]:
        """A server that ignores ``stream`` and answers with one JSON body."""
        try:
            data = r.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            finish = choice.get("finish_reason") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {r.text[:300]}") from exc
        self._set_usage(data.get("usage") or {}, data.get("timings") or {})
        return text, finish

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text, finish = self._request(messages, schema, max_tokens)
        try:
            out = parse_json_text(text)
        except ValueError as exc:
            if finish == "length":
                self.stats["truncated"] += 1
                raise LLMTruncated(f"answer truncated at {max_tokens} tokens", text) from exc
            log.warning("llm answer was not JSON (%s); asking for a repair", exc)
            self.stats["repairs"] += 1
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": _repair_prompt(exc)}]
            text, finish = self._request(messages, schema, max_tokens)
            try:
                out = parse_json_text(text)
            except ValueError as exc2:
                self.stats["failed"] += 1
                raise LLMError(f"malformed JSON after repair: {exc2}") from exc2
        if finish == "length":
            self.stats["truncated"] += 1
            raise LLMTruncated(f"answer truncated at {max_tokens} tokens", text)
        return out

    def info(self) -> dict:
        return {"provider": self.name, "model": self.model, "base_url": self.base_url}


class AnthropicProvider(LLMProvider):
    """Claude through the official SDK; structured output with ``output_config.format`` (json_schema) so the text block
    is valid JSON of the schema. Streaming keeps long pages clear of HTTP timeouts."""

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, api_key: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S,
                 client=None):
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self._client = client  # tests inject a stub

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed when this provider is configured
            kw = {"timeout": self.timeout_s, "max_retries": 1}
            if self.api_key:
                kw["api_key"] = self.api_key
            self._client = anthropic.Anthropic(**kw)
        return self._client

    def _probe_once(self) -> bool:
        self._get_client().models.retrieve(self.model)
        return True

    def _request(self, system: str, messages: list, schema: dict, max_tokens: int) -> tuple[str, str]:
        import anthropic
        client = self._get_client()
        self.stats["calls"] += 1
        try:
            with client.messages.stream(model=self.model, max_tokens=int(max_tokens), system=system, messages=messages,
                                        output_config={"format": {"type": "json_schema", "schema": schema}},
                                        timeout=self._time_left(self.timeout_s)) as stream:
                msg = stream.get_final_message()
        except anthropic.RateLimitError as exc:
            self.stats["failed"] += 1
            raise LLMError(f"anthropic rate limit: {exc}") from exc
        except anthropic.APIStatusError as exc:
            self.stats["failed"] += 1
            raise LLMError(f"anthropic API error {exc.status_code}: {getattr(exc, 'message', exc)}") from exc
        except anthropic.APIConnectionError as exc:
            self.stats["failed"] += 1
            raise LLMError(f"anthropic connection error: {exc}") from exc
        usage = getattr(msg, "usage", None)
        self.last_usage = {"prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                           "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0)}
        stop = getattr(msg, "stop_reason", None) or ""
        if stop == "refusal":
            details = getattr(msg, "stop_details", None)
            raise LLMError(f"anthropic refusal ({getattr(details, 'category', None)})")
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        return text, stop

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        messages = [{"role": "user", "content": user}]
        text, stop = self._request(system, messages, schema, max_tokens)
        try:
            out = parse_json_text(text)
        except ValueError as exc:
            if stop == "max_tokens":
                self.stats["truncated"] += 1
                raise LLMTruncated(f"answer truncated at {max_tokens} tokens", text) from exc
            self.stats["repairs"] += 1
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": _repair_prompt(exc)}]
            text, stop = self._request(system, messages, schema, max_tokens)
            try:
                out = parse_json_text(text)
            except ValueError as exc2:
                self.stats["failed"] += 1
                raise LLMError(f"malformed JSON after repair: {exc2}") from exc2
        if stop == "max_tokens":
            self.stats["truncated"] += 1
            raise LLMTruncated(f"answer truncated at {max_tokens} tokens", text)
        return out


# ------------------------------------------------------------------------------------------------ configuration
_cache: dict = {"key": None, "provider": None}


def llm_config() -> dict:
    provider = (os.environ.get("LLM_PROVIDER") or "openai").strip().lower()
    if provider in OFF:
        provider = "none"
    model = (os.environ.get("LLM_MODEL") or "").strip()
    if not model:
        model = DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else DEFAULT_OPENAI_MODEL
    try:
        timeout_s = float(os.environ.get("LLM_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
    except ValueError:
        timeout_s = DEFAULT_TIMEOUT_S
    extra = os.environ.get("LLM_EXTRA_BODY") or ""
    try:
        extra_body = json.loads(extra) if extra.strip() else {}
        if not isinstance(extra_body, dict):
            raise ValueError("not an object")
    except ValueError as exc:
        log.warning("LLM_EXTRA_BODY ignored (%s)", exc)
        extra_body = {}
    return {"provider": provider, "model": model, "base_url": (os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).strip(),
            "api_key": os.environ.get("LLM_API_KEY") or None, "timeout_s": timeout_s, "extra_body": extra_body,
            "grammar": (os.environ.get("LLM_GRAMMAR") or "auto").strip().lower()}


def get_provider() -> LLMProvider | None:
    """The configured provider (cached until the configuration changes), None when LLM_PROVIDER is none/unknown."""
    cfg = llm_config()
    key = (cfg["provider"], cfg["model"], cfg["base_url"], cfg["api_key"], cfg["timeout_s"], json.dumps(cfg["extra_body"], sort_keys=True), cfg["grammar"])
    if _cache["key"] == key:
        return _cache["provider"]
    if cfg["provider"] == "openai":
        prov: LLMProvider | None = OpenAICompatProvider(cfg["base_url"], cfg["model"], cfg["api_key"], cfg["timeout_s"],
                                                         extra_body=cfg["extra_body"], grammar=cfg["grammar"])
    elif cfg["provider"] == "anthropic":
        prov = AnthropicProvider(cfg["model"], cfg["api_key"], cfg["timeout_s"])
    else:
        if cfg["provider"] != "none":
            log.warning("unknown LLM_PROVIDER %r: LLM engine disabled", cfg["provider"])
        prov = None
    _cache["key"], _cache["provider"] = key, prov
    return prov


def reset_provider() -> None:
    _cache["key"], _cache["provider"] = None, None


def llm_status() -> dict:
    """Contract §2.4: {"configured", "provider", "model", "reachable"} (+ base_url for OpenAI-compatible endpoints)."""
    prov = get_provider()
    if prov is None:
        return {"configured": False, "provider": None, "model": None, "reachable": False}
    return {"configured": True, **prov.info(), "reachable": prov.reachable()}
