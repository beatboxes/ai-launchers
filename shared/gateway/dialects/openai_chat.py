"""OpenAI Chat Completions dialect (DESIGN §3.1) for xAI, OpenAI, Moonshot, DeepSeek, Ollama,
OpenRouter, OpenCode, NVIDIA and generic OpenAI-compatible servers (rules in ``chat_profiles``).

``OpenAIChatDialect().execute(ctx)`` -> internal events. Pure helpers (unit-tested directly):

``build_request(req, model_spec, provider, profile=None, caps=None, session_id=None) -> BuiltRequest``
    NormalizedRequest -> ``body`` (dict), ``headers``, ``names`` (ToolNameMap), ``notes``
    (``[(key, text)]`` of dropped/adapted features), ``chat_only``.
``ChatStreamParser(names, provider_id, model_id, ...)``
    ``feed(chunk_dict) -> [events]`` for every SSE ``data`` object (or a whole non-stream
    ``chat.completion``), ``finish(done) -> [events]`` at end of stream.
``chat_url(provider, model_spec, profile=None) -> str``

Stream mapping: ``delta.content`` -> TextDelta (a leading ``<think>…</think>`` section becomes
thinking); ``delta.reasoning_content``/``delta.reasoning`` -> ThinkingDelta (no signatures: the
emitter adds the synthetic ``fgw1.chat.*`` one); ``delta.tool_calls[i]`` accumulated per index
(id-only first chunks, whole calls in one delta, interleaved parallel calls, missing indexes) and
emitted as ToolCall when the next index starts and the previous arguments parse as a JSON object,
else at the end in index order; usage -> Usage(prompt - cached, completion, cached); finish_reason
-> Finish; ``{"error": …}`` -> GatewayError before the first event, StreamError afterwards.
"""

import itertools
import json
import time

from .. import errors
from ..chat_profiles import get_profile
from ..compat import json_dumps_compact, removesuffix
from ..events import Finish, TextDelta, ThinkingDelta, ToolCall, Usage
from ..schema import scrub
from ..signatures import keep_thinking_for
from ..toolnames import ToolNameMap, decode_tool_id, encode_tool_id, new_tool_id
from ..transport import TransportError, iter_sse
from .base import Dialect, join_url, send_with_auth_retry

__all__ = ["OpenAIChatDialect", "BuiltRequest", "ChatStreamParser", "build_request", "chat_url", "FINISH_REASONS",
           "SIGNATURE_TARGET", "CONTENT_FILTER_TEXT", "IMAGE_BELOW", "DOCUMENT_BELOW", "PDF_OMITTED", "IMAGE_OMITTED",
           "STRUCTURED_OUTPUT_NOTE", "MISSING_RESULT_TEXT", "OLLAMA_PROBE_RETRY"]

SIGNATURE_TARGET = "chat"
CHAT_PATH = "/chat/completions"
CONTENT_FILTER_TEXT = "[blocked by provider content filter]"
IMAGE_BELOW = "[image attached below]"
DOCUMENT_BELOW = "[document attached below]"
PDF_OMITTED = "[PDF omitted: provider lacks document input]"
IMAGE_OMITTED = "[image omitted: model lacks vision input]"
STRUCTURED_OUTPUT_NOTE = "Respond with ONLY a JSON object matching this JSON schema: "
MISSING_RESULT_TEXT = "ERROR: no result was provided for this tool call"
EMPTY_RESULT_TEXT = "(no output)"
OLLAMA_PROBE_RETRY = 300.0  # seconds before a failed /api/show probe is retried
OLLAMA_PROBE_TIMEOUT = 10.0
MAX_STOP_SEQUENCES = 4

FINISH_REASONS = {
    "stop": "end_turn", "end_turn": "end_turn", "eos": "end_turn", "content_filter": "end_turn",
    "length": "max_tokens", "max_tokens": "max_tokens", "model_length": "max_tokens",
    "tool_calls": "tool_use", "function_call": "tool_use", "tool_use": "tool_use",
}

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


# =========================================================================================
# request building
# =========================================================================================

class BuiltRequest(object):
    __slots__ = ("body", "headers", "names", "notes", "chat_only", "profile")

    def __init__(self, body, headers, names, notes, chat_only, profile):
        self.body = body
        self.headers = headers
        self.names = names
        self.notes = notes
        self.chat_only = chat_only
        self.profile = profile


def chat_url(provider, model_spec, profile=None):
    """``{base}/chat/completions`` (``options.chat_path`` / chat-dialect ``path_override`` win)."""
    profile = profile or get_profile(provider.profile)
    base = provider.base_url or profile.default_base_url
    if not base:
        raise errors.GatewayError(500, "api_error", "[%s/%s] provider has no base_url configured"
                                  % (provider.id, model_spec.id), False)
    path = (provider.options or {}).get("chat_path") or CHAT_PATH
    if model_spec.path_override and model_spec.dialect_override in (None, "openai_chat"):
        path = model_spec.path_override
    return join_url(base, path)


def _image_part(block):
    if block.url:
        url = block.url
    elif block.data:
        url = "data:%s;base64,%s" % (block.media_type or "image/png", block.data)
    else:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _document_text(block):
    head = "[document: %s]\n" % block.title if block.title else ""
    return head + (block.text or "")


def _pdf_part(block):
    return {"type": "file", "file": {"filename": block.title or "document.pdf",
                                     "file_data": "data:%s;base64,%s" % (block.media_type or "application/pdf",
                                                                         block.data)}}


class _Converter(object):
    """Anthropic messages -> chat ``messages`` for one request."""

    def __init__(self, req, profile, names, chat_only, vision, notes):
        self.req = req
        self.profile = profile
        self.names = names
        self.chat_only = chat_only
        self.vision = vision
        self.notes = notes
        self.out = []
        self.pending = []  # tool_call ids of the last assistant message still awaiting a tool message
        self.names_by_id = req.tool_use_names_by_id()

    def note(self, key, text):
        if all(k != key for k, _ in self.notes):
            self.notes.append((key, text))

    # ---- media -----------------------------------------------------------------------------
    def user_part(self, block):
        """Content part for a user-turn block, or None (dropped)."""
        if block.type == "text":
            return {"type": "text", "text": block.text} if block.text else None
        if block.type == "image":
            if not self.vision:
                self.note("vision", "images replaced by a placeholder (model lacks vision input)")
                return {"type": "text", "text": IMAGE_OMITTED}
            return _image_part(block)
        if block.type == "document":
            return self.document_part(block)
        return None

    def document_part(self, block):
        if block.text is not None:
            return {"type": "text", "text": _document_text(block)}
        if block.data and self.profile.pdf_input and (block.media_type or "application/pdf") == "application/pdf":
            return _pdf_part(block)
        if block.url:
            return {"type": "text", "text": "[document: %s]" % block.url}
        self.note("pdf", "PDF documents replaced by a placeholder (provider lacks document input)")
        return {"type": "text", "text": PDF_OMITTED}

    def tool_result_content(self, result, deferred):
        """Text for a tool message; media go to ``deferred`` (parts of the following user message)."""
        lines = []
        text = result.result_text()
        if text:
            lines.append(text)
        for media in result.result_media():
            if media.type == "image":
                part = _image_part(media) if self.vision else None
                if part is None:
                    self.note("vision", "images replaced by a placeholder (model lacks vision input)")
                    lines.append(IMAGE_OMITTED)
                else:
                    deferred.append(part)
                    lines.append(IMAGE_BELOW)
            else:
                part = self.document_part(media)
                if part.get("type") == "file":
                    deferred.append(part)
                    lines.append(DOCUMENT_BELOW)
                else:
                    lines.append(part["text"])
        body = "\n".join(lines) or EMPTY_RESULT_TEXT
        return ("ERROR: " + body) if result.is_error else body

    # ---- messages ----------------------------------------------------------------------------
    def close_pending(self):
        for tid in self.pending:
            self.note("missing_result", "synthesized tool messages for tool calls without a result")
            self.out.append({"role": "tool", "tool_call_id": tid, "content": MISSING_RESULT_TEXT})
        self.pending = []

    def assistant(self, msg):
        self.close_pending()
        texts, thinking, calls = [], [], []
        for b in msg.blocks:
            if b.type == "text" and b.text:
                texts.append(b.text)
            elif b.type == "thinking":
                if b.thinking and keep_thinking_for(b.signature, SIGNATURE_TARGET):
                    thinking.append(b.thinking)
            elif b.type == "tool_use":
                args = json_dumps_compact(b.input if isinstance(b.input, dict) else {})
                if self.chat_only:
                    texts.append("[called tool %s with input %s]" % (b.name, args))
                    continue
                calls.append({"id": decode_tool_id(b.id), "type": "function",
                              "function": {"name": self.names.upstream(b.name), "arguments": args}})
        if not texts and not calls:
            return
        m = {"role": "assistant", "content": "\n\n".join(texts) if texts else None}
        if calls:
            m["tool_calls"] = calls
        if self.profile.echo_reasoning_content:
            if thinking:
                m["reasoning_content"] = "\n\n".join(thinking)
            elif calls and self.profile.reasoning_placeholder:
                m["reasoning_content"] = " "
        self.out.append(m)
        self.pending = [c["id"] for c in calls]

    def user(self, msg):
        deferred, parts = [], []
        for b in msg.blocks:
            if b.type != "tool_result":
                continue
            tid = decode_tool_id(b.tool_use_id)
            content = self.tool_result_content(b, deferred)
            if not self.chat_only and tid in self.pending:
                self.out.append({"role": "tool", "tool_call_id": tid, "content": content})
                self.pending.remove(tid)
            else:
                if not self.chat_only:
                    self.note("orphan_result", "tool results without a matching tool call sent as user text")
                label = self.names_by_id.get(b.tool_use_id) or b.tool_use_id
                parts.append({"type": "text", "text": "[tool result for %s]\n%s" % (label, content)})
        for b in msg.blocks:
            if b.type != "tool_result":
                part = self.user_part(b)
                if part is not None:
                    parts.append(part)
        parts = deferred + parts
        if not parts:
            return
        self.close_pending()
        if all(p.get("type") == "text" for p in parts):
            self.out.append({"role": "user", "content": "\n\n".join(p["text"] for p in parts)})
        else:
            self.out.append({"role": "user", "content": parts})

    def run(self, system_text):
        if system_text:
            self.out.append({"role": "system", "content": system_text})
        for msg in self.req.messages:
            if msg.role == "assistant":
                self.assistant(msg)
            else:
                self.user(msg)
        self.close_pending()
        return self.out


def _parameters(schema, mode):
    params = scrub(schema if isinstance(schema, dict) else {}, mode)
    if not isinstance(params, dict):
        params = {}
    if "type" not in params:
        params["type"] = "object"
    if params.get("type") == "object" and not isinstance(params.get("properties"), dict):
        params["properties"] = {}
    return params


def build_request(req, model_spec, provider, profile=None, caps=None, session_id=None):
    """NormalizedRequest -> BuiltRequest (pure; no I/O). ``caps``: Ollama ``/api/show`` capabilities."""
    profile = profile or get_profile(provider.profile).with_options(provider.options)
    notes = []
    tools_cap = None if caps is None else caps.get("tools")
    chat_only = bool(provider.chat_only) or model_spec.tools is False or tools_cap is False
    vision = bool(profile.vision) and not (caps is not None and caps.get("vision") is False)
    names = ToolNameMap(req.all_tool_names(), regex=provider.tool_name_regex or profile.tool_name_regex,
                        maxlen=provider.max_tool_name or profile.max_tool_name)
    if any(names.is_renamed(n) for n in req.all_tool_names()):
        notes.append(("tool_names", "tool names shortened/sanitized for the provider (reverse-mapped)"))

    system_parts = [s for s in req.system if s]
    response_format = None
    fmt = req.output_format if isinstance(req.output_format, dict) else None
    if fmt and fmt.get("type") == "json_schema" and isinstance(fmt.get("schema"), dict):
        schema = scrub(fmt["schema"], "basic")
        if profile.response_format:
            response_format = {"type": "json_schema",
                               "json_schema": {"name": "output", "schema": schema, "strict": False}}
        else:
            system_parts.append(STRUCTURED_OUTPUT_NOTE + json_dumps_compact(schema))
            notes.append(("structured_output", "structured output requested via the system prompt "
                                               "(provider lacks response_format json_schema)"))

    conv = _Converter(req, profile, names, chat_only, vision, notes)
    body = {"model": model_spec.id, "messages": conv.run("\n\n".join(system_parts))}

    if req.tools and chat_only:
        notes.append(("chat_only", "model has no tool support: tools omitted (chat-only)"))
    elif req.tools:
        mode = provider.schema_mode or profile.schema_mode
        tools = []
        for t in req.tools:
            fn = {"name": names.upstream(t.name)}
            if t.description:
                fn["description"] = t.description
            fn["parameters"] = _parameters(t.input_schema, mode)
            tools.append({"type": "function", "function": fn})
        body["tools"] = tools
        choice = _tool_choice(req.tool_choice, names, profile, notes)
        if choice is not None:
            body["tool_choice"] = choice
        if req.disable_parallel_tool_use and profile.parallel_param:
            body["parallel_tool_calls"] = False

    limit = model_spec.max_output or profile.default_max_tokens
    max_tokens = int(req.max_tokens or 0) or limit or 4096
    if limit:
        max_tokens = min(max_tokens, int(limit))
    body[profile.max_tokens_field or "max_tokens"] = max_tokens

    dropped = profile.dropped_params(model_spec)
    for name, value in (("temperature", req.temperature), ("top_p", req.top_p)):
        if value is None:
            continue
        if name in dropped:
            notes.append(("drop_" + name, "%s dropped (not supported by this model)" % name))
        else:
            body[name] = value
    stops = [s for s in (req.stop_sequences or []) if isinstance(s, str) and s]
    if stops:
        if "stop" in dropped:
            notes.append(("drop_stop", "stop sequences dropped (not supported by this model)"))
        else:
            body["stop"] = stops[:MAX_STOP_SEQUENCES]
    if req.top_k is not None:
        notes.append(("drop_top_k", "top_k dropped (not part of chat completions)"))

    effort = profile.effort_value(req.effort, model_spec)
    if effort is not None and profile.probe_ollama and not (caps or {}).get("thinking"):
        notes.append(("drop_effort", "reasoning_effort dropped (model has no thinking capability)"))
        effort = None
    if effort is not None:
        if profile.effort_style == "openrouter":
            body["reasoning"] = {"effort": effort}
        else:
            body["reasoning_effort"] = effort

    if response_format is not None:
        body["response_format"] = response_format
    if profile.prompt_cache_key and session_id:
        body["prompt_cache_key"] = session_id
    body["stream"] = True
    if profile.stream_usage:
        body["stream_options"] = {"include_usage": True}
    for k, v in profile.extra_body.items():
        body.setdefault(k, json.loads(json.dumps(v)))
    extra = (provider.options or {}).get("extra_body")
    if isinstance(extra, dict):
        body.update(json.loads(json.dumps(extra)))

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    headers.update(profile.headers)
    headers.update(provider.headers or {})
    if profile.session_header and session_id:
        headers[profile.session_header] = session_id
    return BuiltRequest(body, headers, names, notes, chat_only, profile)


def _tool_choice(choice, names, profile, notes):
    if not isinstance(choice, dict):
        return None
    t = choice.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        if profile.tool_choice_required:
            return "required"
        notes.append(("tool_choice", "tool_choice 'any' sent as 'auto' (provider lacks 'required')"))
        return "auto"
    if t == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": names.upstream(choice["name"])}}
    if t == "none":
        return "none"
    return None


# =========================================================================================
# stream parsing
# =========================================================================================

def _int(v):
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return 0


def usage_event(u):
    """OpenAI-style usage dict -> ``Usage(prompt - cached, completion, cached)``."""
    prompt = _int(u.get("prompt_tokens", u.get("input_tokens")))
    completion = _int(u.get("completion_tokens", u.get("output_tokens")))
    details = u.get("prompt_tokens_details")
    cached = _int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    if not cached:
        cached = _int(u.get("prompt_cache_hit_tokens") or u.get("cached_tokens"))
    cached = min(cached, prompt)
    total = _int(u.get("total_tokens"))
    if total > prompt + completion:  # xAI reports reasoning tokens outside completion_tokens
        completion = total - prompt
    return Usage(prompt - cached, completion, cached)


class _Call(object):
    __slots__ = ("index", "id", "name", "args", "emitted")

    def __init__(self, index):
        self.index = index
        self.id = None
        self.name = ""
        self.args = []
        self.emitted = False

    def raw_args(self):
        return "".join(self.args).strip()


def _complete_args(raw):
    """True when ``raw`` already parses as a JSON object (safe to emit early)."""
    if not raw:
        return False
    try:
        return isinstance(json.loads(raw), dict)
    except ValueError:
        return False


def final_args(raw):
    """-> (JSON object string, note or None). Invalid/non-object arguments become ``"{}"``."""
    if not raw:
        return "{}", None
    note = None
    try:
        obj = json.loads(raw)
    except ValueError:
        try:
            obj, _end = json.JSONDecoder().raw_decode(raw)
            note = "trailing data after tool arguments ignored"
        except ValueError:
            return "{}", "invalid JSON tool arguments replaced by {}"
    if isinstance(obj, str):  # double-encoded
        try:
            inner = json.loads(obj)
        except ValueError:
            inner = None
        if isinstance(inner, dict):
            obj = inner
    if not isinstance(obj, dict):
        return "{}", "tool arguments were not a JSON object; replaced by {}"
    return json_dumps_compact(obj), note


class ChatStreamParser(object):
    """Stateful chat-completions chunk parser producing internal events (see module docstring)."""

    def __init__(self, names=None, provider_id="?", model_id="?", context_window=None, est_tokens=None,
                 used_ids=None, log=None):
        self.names = names or ToolNameMap([])
        self.provider_id = provider_id
        self.model_id = model_id
        self.context_window = context_window
        self.est_tokens = est_tokens
        self.used_ids = set(used_ids or ())
        self.log = log
        self.notes = []
        self.finish_reason = None
        self.usage = None
        self.message_mode = False
        self._calls = {}
        self._last_index = None
        self._emitted_tools = 0
        self._block = -1
        self._kind = None
        self._tag_state = "detect"   # leading <think> detection: detect | inside | outside
        self._tag_buf = ""

    # ---- public ------------------------------------------------------------------------------
    def feed(self, obj, event_name=None):
        """Events for one chunk. Raises ``GatewayError`` for in-stream errors."""
        if not isinstance(obj, dict):
            return []
        if obj.get("error") or event_name == "error":
            raise self.error_for(obj)
        out = []
        if isinstance(obj.get("usage"), dict):
            self.usage = obj["usage"]
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) not in (0, None):
                continue
            if isinstance(choice.get("usage"), dict):  # Moonshot puts usage on the choice
                self.usage = choice["usage"]
            delta = choice.get("delta")
            if not isinstance(delta, dict) and isinstance(choice.get("message"), dict):
                delta = choice["message"]
                self.message_mode = True
            if isinstance(delta, dict):
                out.extend(self._delta(delta))
            fr = choice.get("finish_reason")
            if isinstance(fr, str) and fr:
                self.finish_reason = fr
        return out

    def finish(self, done=True):
        """Flush buffered content and tool calls; Usage + Finish. Raises on a truncated stream."""
        out = self._flush_tags()
        truncated = self.finish_reason in ("length", "max_tokens", "model_length")
        out.extend(self._emit_calls(final=True, drop_invalid=truncated))
        if self.finish_reason is None and not done:
            raise errors.GatewayError(502, "api_error", "[%s/%s] upstream stream ended before completion"
                                      % (self.provider_id, self.model_id), True)
        if self.usage is not None:
            out.append(usage_event(self.usage))
        stop = FINISH_REASONS.get(self.finish_reason or "stop", "end_turn")
        if self.finish_reason == "content_filter":
            self._kind = None
            out.extend(self._text(CONTENT_FILTER_TEXT))
        if self._emitted_tools and stop == "end_turn":
            stop = "tool_use"
        elif not self._emitted_tools and stop == "tool_use":
            stop = "end_turn"
        out.append(Finish(stop))
        return out

    def error_for(self, obj):
        """GatewayError for an in-stream error object."""
        err = obj.get("error")
        code, message = obj.get("code"), obj.get("message")
        if isinstance(err, dict):
            code = err.get("code") if err.get("code") is not None else err.get("type")
            message = err.get("message") or message
        elif isinstance(err, str):
            message = err
        if not isinstance(message, str) or not message:
            message = json_dumps_compact(obj)[:500]
        status = None
        if isinstance(code, int) and not isinstance(code, bool):
            status = code
        elif isinstance(code, str) and code.isdigit():
            status = int(code)
        if status is not None and 400 <= status <= 599:
            return errors.map_upstream_error(status, json.dumps(obj), {}, self.provider_id, self.model_id,
                                             self.context_window, self.est_tokens)
        return errors.map_error_code(str(code or ""), message, self.provider_id, self.model_id,
                                     self.context_window, self.est_tokens)

    # ---- deltas ------------------------------------------------------------------------------
    def _delta(self, delta):
        out = []
        reasoning = delta.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning:
            reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            out.extend(self._thinking(reasoning))
        content = delta.get("content")
        if isinstance(content, list):  # some servers send content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(content, str) and content:
            out.extend(self._content(content))
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for pos, tc in enumerate(calls):
                out.extend(self._tool_delta(tc, pos))
        fc = delta.get("function_call")  # legacy functions API
        if isinstance(fc, dict):
            out.extend(self._tool_delta({"index": 0, "function": fc}, 0))
        return out

    def _thinking(self, text):
        if self._kind != "thinking":
            self._block += 1
            self._kind = "thinking"
        return [ThinkingDelta(self._block, text)]

    def _text(self, text):
        if self._kind != "text":
            self._block += 1
            self._kind = "text"
        return [TextDelta(self._block, text)]

    def _content(self, text):
        """Route content, splitting a leading ``<think>…</think>`` section into thinking."""
        if self._tag_state == "outside":
            return self._text(text)
        self._tag_buf += text
        if self._tag_state == "detect":
            lead = self._tag_buf.lstrip()
            if not lead or (len(lead) < len(_THINK_OPEN) and _THINK_OPEN.startswith(lead)):
                return []
            if not lead.startswith(_THINK_OPEN):
                self._tag_state = "outside"
                buf, self._tag_buf = self._tag_buf, ""
                return self._text(buf)
            self._tag_state = "inside"
            self._tag_buf = lead[len(_THINK_OPEN):]
        idx = self._tag_buf.find(_THINK_CLOSE)
        if idx >= 0:
            thought, rest = self._tag_buf[:idx], self._tag_buf[idx + len(_THINK_CLOSE):].lstrip()
            self._tag_state, self._tag_buf = "outside", ""
            out = self._thinking(thought) if thought else []
            return out + (self._text(rest) if rest else [])
        keep = 0
        for n in range(min(len(_THINK_CLOSE) - 1, len(self._tag_buf)), 0, -1):
            if _THINK_CLOSE.startswith(self._tag_buf[-n:]):
                keep = n
                break
        emit = self._tag_buf[:len(self._tag_buf) - keep]
        self._tag_buf = self._tag_buf[len(self._tag_buf) - keep:]
        return self._thinking(emit) if emit else []

    def _flush_tags(self):
        buf, self._tag_buf = self._tag_buf, ""
        if self._tag_state == "inside":
            return self._thinking(buf) if buf else []
        if self._tag_state == "detect" and buf.strip():
            self._tag_state = "outside"
            return self._text(buf)
        return []

    def _tool_delta(self, tc, pos):
        if not isinstance(tc, dict):
            return []
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        tid = tc.get("id") if isinstance(tc.get("id"), (str, int)) and tc.get("id") != "" else None
        idx = tc.get("index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            if self.message_mode:
                idx = pos
            elif self._last_index is None:
                idx = 0
            else:
                last = self._calls[self._last_index]
                starts_new = (tid is not None and last.id is not None and str(tid) != last.id) or \
                    (fn.get("name") and last.name and _complete_args(last.raw_args()))
                idx = max(self._calls) + 1 if starts_new else self._last_index
        out = []
        call = self._calls.get(idx)
        if call is None:
            out.extend(self._emit_calls(final=False))
            call = self._calls[idx] = _Call(idx)
        self._last_index = idx
        if tid is not None and call.id is None:
            call.id = str(tid)
        name = fn.get("name")
        if isinstance(name, str) and name and name != call.name:
            if not call.name:
                call.name = name
            elif (call.name + name) in self.names.upstream_names():  # name streamed in fragments
                call.name += name
        args = fn.get("arguments")
        if isinstance(args, (dict, list)):
            call.args.append(json.dumps(args, ensure_ascii=False))
        elif isinstance(args, str):
            call.args.append(args)
        return out

    def _emit_calls(self, final, drop_invalid=False):
        out = []
        for idx in sorted(self._calls):
            call = self._calls[idx]
            if call.emitted:
                continue
            raw = call.raw_args()
            if not final and (not call.name or not _complete_args(raw)):
                break  # keep index order: later calls wait for this one
            call.emitted = True
            if not call.name:
                self._note("tool call without a name dropped")
                continue
            if drop_invalid and raw and not _complete_args(raw):
                self._note("truncated tool call dropped (max tokens reached)")
                continue
            input_json, note = final_args(raw)
            if note:
                self._note(note)
            tool_id = encode_tool_id(call.id)
            if tool_id in self.used_ids:
                tool_id = new_tool_id()
            self.used_ids.add(tool_id)
            self._emitted_tools += 1
            self._kind = None  # text after a tool call opens a new block
            out.append(ToolCall(tool_id, self.names.original(call.name), input_json))
        return out

    def _note(self, text):
        self.notes.append(text)
        if self.log is not None:
            self.log.warning("[%s/%s] %s", self.provider_id, self.model_id, text)


# =========================================================================================
# dialect
# =========================================================================================

def _ollama_root(base_url):
    """Ollama API root (``/api/...``) for an OpenAI-compatible base ending in ``/v1``."""
    return removesuffix((base_url or "").rstrip("/"), "/v1")


class OpenAIChatDialect(Dialect):
    name = "openai_chat"

    def execute(self, ctx):
        provider = ctx.provider
        profile = get_profile(provider.profile).with_options(provider.options)
        url = chat_url(provider, ctx.model, profile)
        caps = self.ollama_caps(ctx) if profile.probe_ollama else None
        built = build_request(ctx.req, ctx.model, provider, profile, caps, ctx.session_id)
        for key, text in built.notes:
            ctx.runtime.log_once(ctx.log, "openai_chat:%s:%s" % (ctx.model.id, key), "[%s/%s] %s",
                                 provider.id, ctx.model.id, text)
        ctx.trace("openai_chat_request", provider=provider.id, model=ctx.model.id, profile=profile.name,
                  tools=len(built.body.get("tools") or []), messages=len(built.body["messages"]),
                  notes=[t for _, t in built.notes])
        used_ids = set(b.id for m in ctx.req.messages for b in m.blocks if b.type == "tool_use" and b.id)
        parser = ChatStreamParser(built.names, provider.id, ctx.model.id, ctx.model.context, ctx.est_tokens,
                                  used_ids, ctx.log)
        payload = json_dumps_compact(built.body).encode("utf-8")
        resp = send_with_auth_retry(ctx, "POST", url, built.headers, payload, stream=True)
        committed = False
        try:
            for ev in self._events(resp, parser):
                committed = True
                yield ev
        except errors.GatewayError as err:
            if not committed:
                raise
            yield err.to_stream_error()
        except TransportError as exc:
            err = errors.map_upstream_error(None, exc.message, {}, provider.id, ctx.model.id, ctx.model.context,
                                            ctx.est_tokens)
            if not committed:
                raise err from exc
            yield err.to_stream_error()
        finally:
            resp.close()
            if parser.notes:
                ctx.trace("openai_chat_notes", notes=parser.notes)

    # ---- response handling ---------------------------------------------------------------------
    def _events(self, resp, parser):
        lines = resp.iter_lines()
        head = []
        first = None
        for line in lines:
            if isinstance(line, (bytes, bytearray)):
                line = bytes(line).decode("utf-8", "replace")
            head.append(line)
            if line.strip():
                first = line.strip()
                break
        if first is None:
            raise errors.GatewayError(502, "api_error", "[%s/%s] upstream returned an empty response"
                                      % (parser.provider_id, parser.model_id), True)
        if first[:1] in ("{", "["):  # server ignored stream:true (or NDJSON chunks)
            rest = (ln.decode("utf-8", "replace") if isinstance(ln, (bytes, bytearray)) else ln for ln in lines)
            for ev in self._json_events("\n".join(itertools.chain(head, rest)), parser):
                yield ev
            return
        done = False
        for sse in iter_sse(itertools.chain(head, lines)):
            data = sse.data.strip()
            if not data:
                continue
            if data == "[DONE]":
                done = True
                break
            try:
                obj = json.loads(data)
            except ValueError:
                if sse.event == "error":
                    raise parser.error_for({"error": {"message": data}}) from None
                parser.notes.append("unparseable stream chunk skipped")
                continue
            for ev in parser.feed(obj, sse.event):
                yield ev
        for ev in parser.finish(done):
            yield ev

    def _json_events(self, text, parser):
        try:
            objs = [json.loads(text)]
        except ValueError:
            try:
                objs = [json.loads(line) for line in text.splitlines() if line.strip()]
            except ValueError:
                raise errors.GatewayError(502, "api_error", "[%s/%s] unparseable upstream response: %s"
                                          % (parser.provider_id, parser.model_id, " ".join(text.split())[:200]),
                                          True) from None
        for obj in objs:
            for ev in parser.feed(obj):
                yield ev
        for ev in parser.finish(True):
            yield ev

    # ---- Ollama capabilities ---------------------------------------------------------------------
    def ollama_caps(self, ctx):
        """``{"tools", "thinking", "vision", "probed", "ts"}`` for ``ctx.model.id`` (cached per runtime)."""
        table = ctx.runtime.setdefault("ollama_caps", dict)
        model_id = ctx.model.id
        now = time.time()
        with ctx.runtime.lock:
            hit = table.get(model_id)
        if hit is not None and (hit.get("probed") or now - hit.get("ts", 0) < OLLAMA_PROBE_RETRY):
            return hit
        caps = self._probe_ollama(ctx, model_id, now)
        with ctx.runtime.lock:
            table[model_id] = caps
        return caps

    def _probe_ollama(self, ctx, model_id, now):
        url = join_url(_ollama_root(ctx.provider.base_url or "http://localhost:11434"), "/api/show")
        body = json_dumps_compact({"model": model_id}).encode("utf-8")
        try:
            resp = send_with_auth_retry(ctx, "POST", url, {"Content-Type": "application/json"}, body, stream=False,
                                        timeout=OLLAMA_PROBE_TIMEOUT)
            try:
                data = resp.json()
            finally:
                resp.close()
        except (errors.GatewayError, ValueError) as exc:
            ctx.runtime.log_once(ctx.log, "ollama_probe:" + model_id, "[%s/%s] /api/show probe failed (%s); "
                                 "assuming tools without thinking", ctx.provider.id, model_id,
                                 getattr(exc, "message", None) or type(exc).__name__)
            return {"tools": True, "thinking": False, "vision": None, "probed": False, "ts": now}
        capabilities = data.get("capabilities") if isinstance(data, dict) else None
        if not isinstance(capabilities, list):  # Ollama without capability reporting
            return {"tools": True, "thinking": False, "vision": None, "probed": True, "ts": now}
        caps = {"tools": "tools" in capabilities, "thinking": "thinking" in capabilities,
                "vision": "vision" in capabilities, "probed": True, "ts": now}
        if not caps["tools"]:
            ctx.runtime.log_once(ctx.log, "ollama_chat_only:" + model_id, "[%s/%s] model has no tool support; "
                                 "serving it chat-only", ctx.provider.id, model_id)
        return caps
