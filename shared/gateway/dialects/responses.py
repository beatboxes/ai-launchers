"""OpenAI Responses dialect (DESIGN §3.2): openai_api, chatgpt_codex, grok_cli_proxy, xai_api.

One translator; per-target differences live in ``responses_targets``.

Request: ``instructions`` (joined system, never empty) + flat ``input`` items (user/assistant messages,
``reasoning`` items rebuilt from matching ``fgw1.<tag>`` thinking signatures, ``function_call`` /
``function_call_output`` with tool-result images deferred to one following user message), flat
function ``tools`` (names shortened via ``ToolNameMap``, schemas scrubbed), ``tool_choice``,
``parallel_tool_calls``, ``reasoning`` + ``include:["reasoning.encrypted_content"]`` for reasoning
models, ``store:false``, ``stream:true``, ``prompt_cache_key`` = session id (OpenAI targets; the grok
targets carry the session id in ``x-grok-conv-id`` instead).

Stream: ``StreamParser`` turns Responses SSE events into internal events (thinking keyed by
output_index, signature on ``output_item.done``, tool calls on ``output_item.done``, usage/finish on
``response.completed``/``response.incomplete``, ``response.failed``/``error`` -> GatewayError).

Grok sticky fallback: a qualifying pre-commit failure of ``grok_cli_proxy`` (connection error, 404, 410,
426, 5xx, 400/403 mentioning version/upgrade/client) sets ``ProviderRuntime["grok_fallback"]`` and
delegates to ``provider.fallback`` for the rest of the process.
"""

import hashlib
import json
import re

from .. import errors
from .. import signatures
from ..events import Finish, TextDelta, ThinkingDelta, ThinkingSignature, ToolCall, Usage
from ..responses_targets import get_target
from ..schema import scrub
from ..toolnames import ToolNameMap, decode_tool_id, encode_tool_id, new_tool_id
from ..transport import TransportError, iter_sse
from .base import Dialect, send_with_auth_retry

__all__ = ["ResponsesDialect", "StreamParser", "PreparedRequest", "build_request", "target_for",
           "fallback_qualifies", "DEFAULT_INSTRUCTIONS", "FALLBACK_FLAG"]

DEFAULT_INSTRUCTIONS = "You are a helpful coding assistant."
FALLBACK_FLAG = "grok_fallback"
PDF_OMITTED = "[PDF omitted: provider lacks document input]"
IMAGE_NOTE = "[image attached below]"
DOCUMENT_NOTE = "[document attached below]"
INCLUDE_ENCRYPTED = "reasoning.encrypted_content"
_FALLBACK_TEXT_RE = re.compile(r"version|upgrade|client", re.I)
_EFFORT_MAP = {"none": "low", "minimal": "low", "low": "low", "medium": "medium", "high": "high",
               "xhigh": "high", "max": "high"}


def target_for(ctx):
    """The ``ResponsesTarget`` for this request (model ``target_override`` wins)."""
    return get_target(ctx.provider.effective_target(ctx.model))


def _limit_id(value, limit):
    """Deterministically shorten ids over ``limit`` chars (same input -> same output)."""
    if not limit or len(value) <= limit:
        return value
    return value[:limit - 9] + "_" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def _effort(ctx):
    if ctx.background:
        return "low"
    return _EFFORT_MAP.get(ctx.req.effort or "medium", "medium")


# ---------------------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------------------

class PreparedRequest(object):
    """Everything ``execute`` sends: ``url``, ``headers`` (without auth), ``body`` (dict), ``names``."""

    __slots__ = ("url", "headers", "body", "names", "target", "lite")

    def __init__(self, url, headers, body, names, target, lite):
        self.url = url
        self.headers = headers
        self.body = body
        self.names = names
        self.target = target
        self.lite = lite

    def data(self):
        return json.dumps(self.body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class _InputBuilder(object):
    """Normalized messages -> Responses ``input`` items."""

    def __init__(self, target, names, keep_reasoning):
        self.target = target
        self.names = names
        self.keep_reasoning = keep_reasoning
        self.items = []

    def call_id(self, tool_use_id):
        return _limit_id(decode_tool_id(tool_use_id or "") or "call_missing", self.target.max_call_id)

    # ---- content parts -------------------------------------------------------------------
    def media_part(self, blk):
        """input_image / input_file part for a media block, or None when unsupported."""
        if blk.type == "image":
            if blk.url:
                return {"type": "input_image", "image_url": blk.url}
            if blk.data:
                return {"type": "input_image",
                        "image_url": "data:%s;base64,%s" % (blk.media_type or "image/png", blk.data)}
            return None
        if blk.type == "document" and self.target.input_file:
            if blk.url:
                return {"type": "input_file", "file_url": blk.url}
            if blk.data:
                return {"type": "input_file", "filename": blk.title or "document.pdf",
                        "file_data": "data:%s;base64,%s" % (blk.media_type or "application/pdf", blk.data)}
        return None

    def user_part(self, blk):
        if blk.type == "text":
            return {"type": "input_text", "text": blk.text} if blk.text else None
        if blk.type == "document" and blk.text is not None:
            text = ("%s\n\n%s" % (blk.title, blk.text)) if blk.title else blk.text
            return {"type": "input_text", "text": text} if text else None
        if blk.is_media():
            part = self.media_part(blk)
            if part is not None:
                return part
            if blk.type == "document":
                return {"type": "input_text", "text": "[document: %s]" % blk.url if blk.url else PDF_OMITTED}
        return None

    # ---- messages ------------------------------------------------------------------------
    def user(self, msg, known_calls, answered):
        parts = []
        for blk in msg.blocks:
            if blk.type != "tool_result":
                part = self.user_part(blk)
                if part is not None:
                    parts.append(part)
                continue
            output, attached = self.tool_output(blk)
            cid = self.call_id(blk.tool_use_id)
            if cid in known_calls:
                answered.add(cid)
                self.items.append({"type": "function_call_output", "call_id": cid, "output": output})
            else:  # orphan result (history truncated): keep the content as user text
                parts.append({"type": "input_text", "text": "[tool result %s]\n%s" % (blk.tool_use_id, output)})
            parts.extend(attached)
        if parts:
            self.items.append({"type": "message", "role": "user", "content": parts})

    def tool_output(self, blk):
        """(output text, deferred media parts) of a tool_result block."""
        text = blk.result_text()
        notes, attached = [], []
        for m in blk.result_media():
            part = self.media_part(m)
            if part is not None:
                attached.append(part)
                note = IMAGE_NOTE if m.type == "image" else DOCUMENT_NOTE
            else:
                note = PDF_OMITTED if m.type == "document" else "[image omitted]"
            if note not in notes:
                notes.append(note)
        output = "\n".join([t for t in [text] + notes if t])
        if blk.is_error:
            output = "ERROR: " + output
        return output, attached

    def assistant(self, msg, known_calls):
        texts = []

        def flush():
            if texts:
                self.items.append({"type": "message", "role": "assistant",
                                   "content": [{"type": "output_text", "text": t} for t in texts]})
                del texts[:]

        for blk in msg.blocks:
            if blk.type == "text":
                if blk.text:
                    texts.append(blk.text)
            elif blk.type == "thinking":
                item = self.reasoning_item(blk)
                if item is not None:
                    flush()
                    self.items.append(item)
            elif blk.type == "tool_use":
                flush()
                cid = self.call_id(blk.id)
                known_calls.add(cid)
                self.items.append({"type": "function_call", "call_id": cid, "name": self.names.upstream(blk.name),
                                   "arguments": json.dumps(blk.input or {}, ensure_ascii=False,
                                                           separators=(",", ":"))})
        flush()

    def reasoning_item(self, blk):
        if not self.keep_reasoning or not signatures.keep_thinking_for(blk.signature, self.target.sig_tag):
            return None
        decoded = signatures.decode_signature(blk.signature)
        payload = decoded[1] if decoded else {}
        enc = payload.get("enc")
        if not isinstance(enc, str) or not enc:
            return None  # without encrypted content the item cannot be replayed (store:false)
        summary = [s for s in payload.get("summary") or [] if isinstance(s, str)]
        return {"type": "reasoning", "summary": [{"type": "summary_text", "text": s} for s in summary],
                "encrypted_content": enc}

    def build(self, messages):
        known_calls, answered = set(), set()
        for msg in messages:
            if msg.role == "assistant":
                self.assistant(msg, known_calls)
            else:
                self.user(msg, known_calls, answered)
        missing = known_calls - answered
        if missing:  # every function_call needs an output or the request is rejected
            out = []
            for item in self.items:
                out.append(item)
                if item.get("type") == "function_call" and item["call_id"] in missing:
                    out.append({"type": "function_call_output", "call_id": item["call_id"],
                                "output": "[no result: the tool call was interrupted]"})
            self.items = out
        return self.items


def _instructions(req, target):
    text = "\n\n".join(s for s in req.system if s and s.strip())
    if req.output_format and not target.text_format and isinstance(req.output_format.get("schema"), dict):
        text = (text + "\n\n" if text else "") + "Respond with ONLY a JSON object matching this JSON schema: " + \
            json.dumps(req.output_format["schema"], ensure_ascii=False, separators=(",", ":"))
    return text if text.strip() else DEFAULT_INSTRUCTIONS


def _tool_choice(choice, names):
    kind = (choice or {}).get("type")
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "name": names.upstream(choice["name"])}
    return "auto"


def build_request(ctx, target=None, environ=None):
    """Translate ``ctx.req`` for ``target`` (default: ``target_for(ctx)``) -> ``PreparedRequest``."""
    target = target or target_for(ctx)
    req, spec, provider = ctx.req, ctx.model, ctx.provider
    reasoning = target.is_reasoning(spec)
    lite = target.is_lite(spec)
    names = ToolNameMap(req.all_tool_names(), regex=provider.tool_name_regex, maxlen=provider.max_tool_name or 64)
    schema_mode = provider.schema_mode or target.schema_mode

    body = {"model": spec.id, "instructions": _instructions(req, target),
            "input": _InputBuilder(target, names, reasoning).build(req.messages)}
    if req.tools and spec.tools is not False:
        tools = []
        for t in req.tools:
            params = scrub(t.input_schema or {}, schema_mode)
            if not isinstance(params, dict) or not params:
                params = {"type": "object", "properties": {}}
            tools.append({"type": "function", "name": names.upstream(t.name), "description": t.description or "",
                          "parameters": params, "strict": False})
        body["tools"] = tools
        body["tool_choice"] = _tool_choice(req.tool_choice, names)
        body["parallel_tool_calls"] = not req.disable_parallel_tool_use
    if reasoning:
        if target.reasoning_style == "openai":
            body["reasoning"] = {"effort": _effort(ctx), "summary": "auto"}
        elif spec.effort_param:
            body["reasoning"] = {"effort": _effort(ctx)}
        body["include"] = [INCLUDE_ENCRYPTED]
    if req.output_format and target.text_format and isinstance(req.output_format.get("schema"), dict):
        body["text"] = {"format": {"type": "json_schema", "name": "output",
                                   "schema": scrub(req.output_format["schema"], "basic"), "strict": False}}
    body["store"] = False
    body["stream"] = True
    if target.cache_key and ctx.session_id:
        body["prompt_cache_key"] = ctx.session_id
    target.apply_body_rules(body, req, spec, reasoning, lite)

    headers = target.headers(ctx.session_id, ctx.runtime, lite, environ,
                             client_identifier=(provider.options or {}).get("client_identifier"))
    headers.update(provider.headers or {})
    url = target.url(provider.base_url, spec.path_override)
    return PreparedRequest(url, headers, body, names, target, lite)


# ---------------------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------------------

def _int(value):
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _usage(response):
    u = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(u, dict):
        return None
    details = u.get("input_tokens_details") if isinstance(u.get("input_tokens_details"), dict) else {}
    inp, cached = _int(u.get("input_tokens")), _int(details.get("cached_tokens"))
    cached = min(cached, inp)
    return Usage(inp - cached, _int(u.get("output_tokens")), cached)


def _tool_args(raw, complete=True):
    """Arguments string -> JSON object string; None if incomplete and unparsable."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return "{}"
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return raw.strip()
    if not complete:
        return None
    # a completed but malformed call: surface it so Claude Code reports the bad input to the model
    return json.dumps({"_raw_arguments": raw if isinstance(raw, str) else json.dumps(raw)}, ensure_ascii=False)


class StreamParser(object):
    """Responses SSE events -> internal events.

    ``feed(event_type, data) -> List[Event]`` (``data`` = decoded JSON object) raises
    ``errors.GatewayError`` for ``response.failed`` / ``error``; ``done`` turns True on a terminal
    event; ``end() -> List[Event]`` raises a retryable connection-style GatewayError if the stream
    stopped before a terminal event.
    """

    def __init__(self, sig_tag, names=None, provider_id="?", model_id="?", context_window=None, est_tokens=None):
        self.sig_tag = sig_tag
        self.names = names
        self.provider_id = provider_id
        self.model_id = model_id
        self.context_window = context_window
        self.est_tokens = est_tokens
        self.done = False
        self.tool_calls = 0
        self._reasoning = {}   # key -> {"open", "emitted", "part", "parts": {part: text}}
        self._text = set()     # keys that streamed text
        self._calls = {}       # key -> {"call_id", "name", "args"}
        self._done_keys = set()
        self._done_ids = set()

    # ---- helpers -------------------------------------------------------------------------
    @staticmethod
    def _key(data, item=None):
        idx = data.get("output_index")
        if isinstance(idx, int) and not isinstance(idx, bool):
            return idx
        return data.get("item_id") or (item or {}).get("id") or 0

    def _rstate(self, key):
        st = self._reasoning.get(key)
        if st is None:
            st = self._reasoning[key] = {"open": False, "emitted": False, "part": None, "parts": {}}
        return st

    def _open_thinking(self, key, out):
        st = self._rstate(key)
        if not st["open"]:
            st["open"] = True
            out.append(ThinkingDelta(key, ""))
        return st

    def _thinking_delta(self, key, part, delta, out):
        if not isinstance(delta, str) or not delta:
            return
        st = self._open_thinking(key, out)
        prefix = "\n\n" if st["emitted"] and st["part"] is not None and st["part"] != part else ""
        st["part"] = part
        st["emitted"] = True
        st["parts"][part] = st["parts"].get(part, "") + delta
        out.append(ThinkingDelta(key, prefix + delta))

    def _error(self, err):
        err = err if isinstance(err, dict) else {}
        codes = [c for c in (err.get("code"), err.get("type")) if isinstance(c, str) and c and c != "error"]
        message = err.get("message") or (codes[0] if codes else "upstream stream error")
        extra = {k: err[k] for k in ("resets_at", "resets_in_seconds", "plan_type") if k in err}
        mapped = None
        for code in codes or [None]:  # e.g. code "invalid_value" is unknown but type "invalid_request_error" is not
            mapped = errors.map_error_code(code, message, self.provider_id, self.model_id, self.context_window,
                                           self.est_tokens, extra or None)
            if (mapped.status, mapped.err_type) != (502, "api_error"):
                break
        return mapped

    # ---- items ---------------------------------------------------------------------------
    def _item_done(self, key, item, out):
        if key in self._done_keys or (item.get("id") and item.get("id") in self._done_ids):
            return
        self._done_keys.add(key)
        if item.get("id"):
            self._done_ids.add(item["id"])
        kind = item.get("type")
        if kind == "reasoning":
            self._reasoning_done(key, item, out)
        elif kind == "function_call":
            call = self._calls.pop(key, {})
            args = _tool_args(item.get("arguments") if item.get("arguments") is not None else call.get("args"))
            self._emit_call(item.get("call_id") or call.get("call_id") or item.get("id"),
                            item.get("name") or call.get("name"), args, out)
        elif kind == "message" and key not in self._text:
            text = "".join(p.get("text") or p.get("refusal") or "" for p in item.get("content") or []
                           if isinstance(p, dict) and p.get("type") in ("output_text", "refusal"))
            if text:
                self._text.add(key)
                out.append(TextDelta(key, text))

    def _reasoning_done(self, key, item, out):
        summaries = [p.get("text") for p in item.get("summary") or []
                     if isinstance(p, dict) and isinstance(p.get("text"), str)]
        st = self._rstate(key)
        if not summaries:
            summaries = [v for k, v in sorted(st["parts"].items(), key=lambda kv: kv[0][1]) if k[0] == "s"]
        if not st["emitted"]:
            text = "\n\n".join(s for s in summaries if s)
            if not text:
                text = "\n\n".join(p.get("text") for p in item.get("content") or []
                                   if isinstance(p, dict) and isinstance(p.get("text"), str))
            if text:
                st["open"] = True
                st["emitted"] = True
                out.append(ThinkingDelta(key, text))
        self._open_thinking(key, out)
        enc = item.get("encrypted_content")
        payload = {"enc": enc if isinstance(enc, str) and enc else None, "summary": summaries}
        out.append(ThinkingSignature(key, signatures.encode_signature(self.sig_tag, payload)))

    def _emit_call(self, call_id, name, args, out):
        if not name or args is None:
            return
        original = self.names.original(name) if self.names is not None else name
        self.tool_calls += 1
        out.append(ToolCall(encode_tool_id(call_id) if call_id else new_tool_id(), original, args))

    def _flush_leftovers(self, response, out):
        """Items a server reported only in the final response, or calls cut short (complete JSON only)."""
        output = response.get("output") if isinstance(response, dict) else None
        for idx, item in enumerate(output if isinstance(output, list) else []):
            if isinstance(item, dict):
                if item.get("type") == "function_call" and item.get("status") == "incomplete":
                    item = dict(item, arguments=_tool_args(item.get("arguments"), complete=False))
                    if item["arguments"] is None:
                        continue
                self._item_done(idx, item, out)
        for key in sorted(self._calls, key=str):
            call = self._calls[key]
            self._emit_call(call.get("call_id"), call.get("name"), _tool_args(call.get("args"), False), out)
        self._calls.clear()

    # ---- public --------------------------------------------------------------------------
    def feed(self, etype, data):
        out = []
        if not isinstance(data, dict) or self.done:
            return out
        etype = data.get("type") or etype or ""
        if etype == "response.output_item.added":
            item = data.get("item") if isinstance(data.get("item"), dict) else {}
            key = self._key(data, item)
            if item.get("type") == "reasoning":
                self._open_thinking(key, out)
            elif item.get("type") == "function_call":
                self._calls[key] = {"call_id": item.get("call_id"), "name": item.get("name"),
                                    "args": item.get("arguments") or ""}
        elif etype in ("response.reasoning_summary_text.delta", "response.reasoning_summary.delta"):
            self._thinking_delta(self._key(data), ("s", _int(data.get("summary_index"))), data.get("delta"), out)
        elif etype in ("response.reasoning_text.delta", "response.reasoning.delta"):
            self._thinking_delta(self._key(data), ("c", _int(data.get("content_index"))), data.get("delta"), out)
        elif etype in ("response.output_text.delta", "response.refusal.delta"):
            delta = data.get("delta")
            if isinstance(delta, str) and delta:
                key = self._key(data)
                self._text.add(key)
                out.append(TextDelta(key, delta))
        elif etype == "response.function_call_arguments.delta":
            call = self._calls.get(self._key(data))
            if call is not None and isinstance(data.get("delta"), str):
                call["args"] += data["delta"]
        elif etype == "response.function_call_arguments.done":
            call = self._calls.get(self._key(data))
            if call is not None and isinstance(data.get("arguments"), str):
                call["args"] = data["arguments"]
        elif etype == "response.output_item.done":
            item = data.get("item") if isinstance(data.get("item"), dict) else {}
            self._item_done(self._key(data, item), item, out)
        elif etype in ("response.completed", "response.done", "response.incomplete", "response.failed"):
            response = data.get("response") if isinstance(data.get("response"), dict) else {}
            status = response.get("status") or etype.split(".", 1)[1]
            if etype == "response.failed" or status == "failed":
                raise self._error(response.get("error"))
            self._flush_leftovers(response, out)
            usage = _usage(response)
            if usage is not None:
                out.append(usage)
            if etype == "response.incomplete" or status == "incomplete":
                details = response.get("incomplete_details") if isinstance(response.get("incomplete_details"),
                                                                           dict) else {}
                reason = details.get("reason") or "unknown"
                if reason in ("max_output_tokens", "max_tokens"):
                    out.append(Finish("max_tokens"))
                else:
                    out.append(TextDelta("incomplete", "[response incomplete: %s]" % reason))
                    out.append(Finish("tool_use" if self.tool_calls else "end_turn"))
            else:
                out.append(Finish("tool_use" if self.tool_calls else "end_turn"))
            self.done = True
        elif etype == "error":
            raise self._error(data.get("error") if isinstance(data.get("error"), dict) else data)
        return out

    def end(self):
        """Call when the upstream stream is exhausted."""
        if self.done:
            return []
        raise errors.map_upstream_error(None, "stream ended before response.completed", {}, self.provider_id,
                                        self.model_id, self.context_window, self.est_tokens)


# ---------------------------------------------------------------------------------------
# dialect
# ---------------------------------------------------------------------------------------

def fallback_qualifies(exc):
    """Grok proxy failure that switches the provider to its fallback (pre-commit only)."""
    if exc.connection_error:
        return True
    status = exc.upstream_status
    if not isinstance(status, int):
        return False
    if status in (404, 410, 426) or status >= 500:
        return True
    if status in (400, 403):
        return bool(_FALLBACK_TEXT_RE.search(exc.upstream_body or exc.message or ""))
    return False


def _delegate(ctx, fallback):
    from . import get_dialect

    return get_dialect(fallback.dialect).execute(ctx.with_provider(fallback))


class ResponsesDialect(Dialect):
    name = "responses"

    def execute(self, ctx):
        target = target_for(ctx)
        fallback = ctx.provider.fallback if target.fallback_capable else None
        if fallback is not None and ctx.runtime.get(FALLBACK_FLAG):
            yield from _delegate(ctx, fallback)
            return
        gen = self._execute(ctx, target)
        try:
            try:
                first = next(gen)
            except StopIteration:
                return
            except errors.GatewayError as exc:
                if fallback is None or not fallback_qualifies(exc):
                    raise
                ctx.runtime.set(FALLBACK_FLAG, True)
                reason = "connection error" if exc.connection_error else "HTTP %s" % exc.upstream_status
                ctx.runtime.log_once(ctx.log, FALLBACK_FLAG,
                                     "[%s] grok CLI proxy unavailable (%s); using %s at %s for this process",
                                     ctx.provider.id, reason, fallback.dialect, fallback.base_url)
                ctx.trace("grok_fallback", provider=ctx.provider.id, reason=reason)
                yield from _delegate(ctx, fallback)
                return
            yield first
            yield from gen
        finally:
            gen.close()

    def _execute(self, ctx, target):
        prepared = build_request(ctx, target)
        ctx.trace("upstream_request", provider=ctx.provider.id, target=target.name, model=ctx.model.id,
                  url=prepared.url, lite=prepared.lite, tools=len(prepared.body.get("tools") or []))
        resp = send_with_auth_retry(ctx, "POST", prepared.url, prepared.headers, prepared.data(), stream=True,
                                    unauthorized=target.unauthorized)
        parser = StreamParser(target.sig_tag, prepared.names, ctx.provider.id, ctx.model.id, ctx.model.context,
                              ctx.est_tokens)
        started = False
        try:
            try:
                for sse in iter_sse(resp):
                    if sse.data.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(sse.data)
                    except ValueError:
                        ctx.runtime.log_once(ctx.log, "responses_bad_json", "[%s] ignoring non-JSON stream data",
                                             ctx.provider.id)
                        continue
                    for ev in parser.feed(sse.event, data):
                        started = True
                        yield ev
                    if parser.done:
                        break
                parser.end()
            except TransportError as exc:
                raise errors.map_upstream_error(None, exc.message, {}, ctx.provider.id, ctx.model.id,
                                                ctx.model.context, ctx.est_tokens)
        except errors.GatewayError as exc:
            if not started:
                raise
            yield exc.to_stream_error()
        finally:
            resp.close()
