"""Anthropic-compatible upstreams (DESIGN §3.4 as amended by §0.A): DeepSeek, Kimi, OpenRouter,
Ollama (>= 0.14) and OpenCode Zen/Go ``/messages`` models.

Request: Claude Code's original body (``req.raw``) with ``model`` -> upstream id (``[1m]``
stripped), ``max_tokens`` clamped, gateway-signed (``fgw1.*``) / unsigned thinking blocks removed
(real upstream signatures kept), mid-conversation ``role:"system"`` messages folded with the shared
``model`` helpers, and the ``x-anthropic-billing-header`` system block kept. Headers:
``anthropic-version: 2023-06-01``, auth from the provider (Bearer or ``x-api-key`` by auth style),
``anthropic-beta`` forwarded minus the folded mid-conversation betas unless ``options.drop_betas``.

Response: the upstream Anthropic SSE (or a plain JSON message) is PARSED into internal events, so
the server's SSEEmitter/Aggregator own the client-facing protocol (requested model echoed,
``stream:false`` supported). ``redacted_thinking`` and server-tool blocks are dropped.

Provider options: ``messages_path`` (default ``/v1/messages``; model ``path_override`` wins),
``drop_betas`` (True = send none, or a list of beta-name prefixes), ``max_output`` (clamp when the
model has none), ``upstream_stream`` (default True), ``drop_fields`` (top-level body keys to
remove), ``fallback_on_404`` (non-Ollama providers: switch to ``fallback`` on a bare 404).

Ollama < 0.14 answers ``/v1/messages`` with a plain-text 404: the provider then switches stickily
(runtime flag ``ollama_chat_fallback``) to its ``fallback`` spec (openai_chat/ollama).
"""

import json
import re
import uuid

from .. import errors
from ..compat import json_dumps_compact
from ..events import STOP_REASONS, Finish, TextDelta, ThinkingDelta, ThinkingSignature, ToolCall, Usage
from ..model import apply_tool_changes, fold_system_messages_raw
from ..signatures import PREFIX as SIGNATURE_PREFIX
from ..transport import TransportError, iter_sse
from .base import Dialect, join_url, send_with_auth_retry

__all__ = ["AnthropicPassthroughDialect", "ANTHROPIC_VERSION", "DEFAULT_MESSAGES_PATH", "FALLBACK_FLAG",
           "build_request_body", "build_headers", "request_url", "upstream_model", "forwarded_betas"]

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MESSAGES_PATH = "/v1/messages"
FALLBACK_FLAG = "ollama_chat_fallback"
FOLDED_BETA_PREFIXES = ("mid-conversation-system-", "mid-conversation-tool-changes-")
DEFAULT_MAX_TOKENS = 32000
MIN_THINKING_BUDGET = 1024

_ONE_M_RE = re.compile(r"\[1m\]$", re.I)

#: Anthropic error type -> HTTP status (for ``event: error`` inside a stream)
_ERROR_STATUS = {
    "invalid_request_error": 400, "authentication_error": 401, "billing_error": 402, "permission_error": 403,
    "not_found_error": 404, "request_too_large": 413, "rate_limit_error": 429, "api_error": 500,
    "timeout_error": 504, "overloaded_error": 529,
}

_STOP_ALIASES = {"model_context_window_exceeded": "max_tokens"}


# ---------------------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------------------

def upstream_model(ctx):
    """The upstream model id (``ctx.model.id``) without a trailing ``[1m]``."""
    return _ONE_M_RE.sub("", ctx.model.id or getattr(ctx.resolution, "model", "") or "")


def _options(ctx):
    return ctx.provider.options or {}


def request_url(ctx):
    path = ctx.model.path_override or _options(ctx).get("messages_path") or DEFAULT_MESSAGES_PATH
    return join_url(ctx.provider.base_url, path)


def _is_foreign_thinking(block):
    """Thinking blocks the upstream cannot verify: gateway-signed (``fgw1.*``) or unsigned."""
    if not isinstance(block, dict) or block.get("type") != "thinking":
        return False
    sig = block.get("signature")
    return not isinstance(sig, str) or not sig or sig.startswith(SIGNATURE_PREFIX + ".")


def _strip_foreign_thinking(messages):
    """-> (messages, tool_turn_lost_thinking). Never mutates the input; assistant messages left
    empty are dropped (the Messages API merges the then-adjacent user turns).
    ``tool_turn_lost_thinking``: the LAST assistant message calls tools and no longer starts with a
    thinking block, which thinking-enabled Anthropic-style APIs reject."""
    out = []
    last_lost = False
    for msg in messages:
        content = msg.get("content")
        if msg.get("role") != "assistant":
            out.append(msg)
            continue
        if not isinstance(content, list):
            out.append(msg)
            last_lost = False
            continue
        kept = [b for b in content if not _is_foreign_thinking(b)]
        lost = len(kept) != len(content)
        kinds = [b.get("type") for b in kept if isinstance(b, dict)]
        last_lost = lost and "tool_use" in kinds and "thinking" not in kinds and "redacted_thinking" not in kinds
        if not lost:
            out.append(msg)
        elif kept:
            nm = dict(msg)
            nm["content"] = kept
            out.append(nm)
    return out, last_lost


def _int_or_none(v):
    if isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def build_request_body(ctx, stream=True):
    """Upstream JSON body (a new dict; ``ctx.req.raw`` is never mutated)."""
    raw = ctx.req.raw if isinstance(ctx.req.raw, dict) else {}
    opts = _options(ctx)
    body = dict(raw)
    body["model"] = upstream_model(ctx)

    fold = fold_system_messages_raw(raw.get("messages") or [])
    messages, last_lost_thinking = _strip_foreign_thinking(fold.messages)
    body["messages"] = messages
    if "tools" in raw or fold.tool_additions or fold.tool_removals:
        tools = apply_tool_changes(raw.get("tools"), fold)
        if tools or "tools" in raw:
            body["tools"] = tools

    limit = ctx.model.max_output or _int_or_none(opts.get("max_output"))
    requested = _int_or_none(raw.get("max_tokens"))
    if requested is None or requested <= 0:
        requested = DEFAULT_MAX_TOKENS
    max_tokens = min(requested, limit) if limit else requested
    body["max_tokens"] = max_tokens

    thinking = raw.get("thinking")
    if isinstance(thinking, dict):
        if last_lost_thinking and thinking.get("type") in ("enabled", "adaptive"):
            # The trailing assistant turn lost its (foreign) thinking block; Anthropic-style APIs reject
            # a thinking-enabled continuation whose last assistant turn does not start with one.
            body.pop("thinking", None)
            _drop_thinking_edits(body)
        elif thinking.get("type") == "enabled":
            budget = _int_or_none(thinking.get("budget_tokens"))
            if budget is not None and budget >= max_tokens:
                if max_tokens - 1 >= MIN_THINKING_BUDGET:
                    body["thinking"] = dict(thinking, budget_tokens=max_tokens - 1)
                else:
                    body.pop("thinking", None)
                    _drop_thinking_edits(body)

    body["stream"] = bool(stream)
    for key in opts.get("drop_fields") or ():
        body.pop(key, None)
    return body


def _drop_thinking_edits(body):
    cm = body.get("context_management")
    if not isinstance(cm, dict) or not isinstance(cm.get("edits"), list):
        return
    edits = [e for e in cm["edits"]
             if not (isinstance(e, dict) and str(e.get("type", "")).startswith("clear_thinking"))]
    if edits:
        body["context_management"] = dict(cm, edits=edits)
    else:
        body.pop("context_management", None)


def forwarded_betas(ctx):
    """``anthropic-beta`` values to forward (folded mid-conversation betas never are)."""
    drop = _options(ctx).get("drop_betas")
    if drop is True:
        return []
    if isinstance(drop, str):
        drop = [drop]
    prefixes = FOLDED_BETA_PREFIXES + tuple(d for d in (drop or ()) if isinstance(d, str) and d)
    raw = (ctx.req.headers or {}).get("anthropic-beta") or ""
    out = []
    for beta in raw.split(","):
        beta = beta.strip()
        if beta and beta not in out and not beta.startswith(prefixes):
            out.append(beta)
    return out


def build_headers(ctx, stream=True):
    """Request headers except auth (``send_with_auth_retry`` adds those)."""
    h = {"Content-Type": "application/json", "anthropic-version": ANTHROPIC_VERSION,
         "Accept": "text/event-stream" if stream else "application/json"}
    ua = (ctx.req.headers or {}).get("user-agent")
    if ua:
        h["User-Agent"] = ua
    betas = forwarded_betas(ctx)
    if betas:
        h["anthropic-beta"] = ",".join(betas)
    for k, v in (ctx.provider.headers or {}).items():
        h[k] = v
    return h


# ---------------------------------------------------------------------------------------
# response
# ---------------------------------------------------------------------------------------

def _stream_error(ctx, data):
    """``event: error`` payload -> GatewayError (Anthropic message preserved, same table as HTTP)."""
    err = data.get("error") if isinstance(data.get("error"), dict) else {}
    etype = str(err.get("type") or "api_error")
    message = str(err.get("message") or etype)
    status = _ERROR_STATUS.get(etype, 500)
    body = json.dumps({"type": "error", "error": {"type": etype, "message": message}})
    return errors.map_upstream_error(status, body, {}, ctx.provider.id, ctx.model.id, ctx.model.context,
                                     ctx.est_tokens)


def _malformed(ctx, what):
    return errors.GatewayError(502, "api_error", "[%s/%s] upstream sent %s" % (ctx.provider.id, ctx.model.id, what),
                               True)


class _Translator(object):
    """Anthropic stream records (decoded SSE ``data`` objects) -> internal events."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.blocks = {}       # index -> dict(type, sig, id, name, parts, input)
        self.usage = {}
        self.stop_reason = None
        self.stop_sequence = None
        self.done = False

    def _usage(self, u):
        if not isinstance(u, dict):
            return
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            v = u.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.usage[key] = int(v)

    def _drop(self, btype):
        self.ctx.runtime.log_once(self.ctx.log, "passthrough_drop_%s" % btype,
                                  "[%s] dropping upstream %s content blocks (not supported through the gateway)",
                                  self.ctx.provider.id, btype)

    def feed(self, data):
        """One record -> list of events. Raises GatewayError for upstream ``error`` records."""
        etype = data.get("type")
        if etype == "error":
            raise _stream_error(self.ctx, data)
        if etype == "message_start":
            msg = data.get("message")
            self._usage(msg.get("usage") if isinstance(msg, dict) else None)
            return []
        if etype == "content_block_start":
            return self._start(data.get("index"), data.get("content_block"))
        if etype == "content_block_delta":
            return self._delta(data.get("index"), data.get("delta"))
        if etype == "content_block_stop":
            return self._stop(data.get("index"))
        if etype == "message_delta":
            delta = data.get("delta") if isinstance(data.get("delta"), dict) else {}
            if delta.get("stop_reason"):
                self.stop_reason = delta["stop_reason"]
                self.stop_sequence = delta.get("stop_sequence")
            self._usage(data.get("usage"))
            return []
        if etype == "message_stop":
            self.done = True
        return []  # ping / unknown

    def _start(self, idx, cb):
        cb = cb if isinstance(cb, dict) else {}
        btype = cb.get("type")
        if btype == "text":
            self.blocks[idx] = {"type": "text"}
            return [TextDelta(idx, cb["text"])] if cb.get("text") else []
        if btype == "thinking":
            self.blocks[idx] = {"type": "thinking", "sig": cb.get("signature") or ""}
            return [ThinkingDelta(idx, cb.get("thinking") or "")]
        if btype == "tool_use":
            self.blocks[idx] = {"type": "tool_use", "id": cb.get("id"), "name": cb.get("name"), "parts": [],
                                "input": cb.get("input")}
            return []
        self.blocks[idx] = {"type": "skip"}
        self._drop(btype or "unknown")
        return []

    def _delta(self, idx, delta):
        delta = delta if isinstance(delta, dict) else {}
        dtype = delta.get("type")
        blk = self.blocks.get(idx)
        if blk is None:  # tolerate a missing content_block_start
            blk = self.blocks[idx] = {"type": {"text_delta": "text", "thinking_delta": "thinking",
                                               "signature_delta": "thinking"}.get(dtype, "skip"), "sig": ""}
        if blk["type"] == "skip":
            return []
        if dtype == "text_delta":
            text = delta.get("text") or ""
            return [TextDelta(idx, text)] if text else []
        if dtype == "thinking_delta":
            text = delta.get("thinking") or ""
            return [ThinkingDelta(idx, text)] if text else []
        if dtype == "signature_delta":
            blk["sig"] = delta.get("signature") or blk.get("sig") or ""
            return []
        if dtype == "input_json_delta" and blk["type"] == "tool_use":
            blk["parts"].append(delta.get("partial_json") or "")
        return []

    def _stop(self, idx):
        blk = self.blocks.pop(idx, None)
        if blk is None:
            return []
        if blk["type"] == "thinking" and blk.get("sig"):
            return [ThinkingSignature(idx, blk["sig"])]
        if blk["type"] == "tool_use":
            return [self._tool_call(blk)]
        return []

    def _tool_call(self, blk):
        raw = "".join(blk["parts"]).strip()
        if raw:
            try:
                args = json.loads(raw)
            except ValueError:
                raise _malformed(self.ctx, "malformed tool input JSON for %r" % (blk.get("name"),))
        else:
            args = blk.get("input") if isinstance(blk.get("input"), dict) else {}
        if not isinstance(args, dict) or not blk.get("name"):
            raise _malformed(self.ctx, "an invalid tool_use block")
        tool_id = blk.get("id") or ("toolu_" + uuid.uuid4().hex[:24])
        return ToolCall(str(tool_id), str(blk["name"]), json_dumps_compact(args))

    def finish(self):
        """Events closing the message. Raises GatewayError if the upstream stream was truncated."""
        if self.blocks and any(b["type"] == "tool_use" for b in self.blocks.values()):
            raise _malformed(self.ctx, "a truncated stream (tool_use block never closed)")
        if not self.done and self.stop_reason is None:
            raise _malformed(self.ctx, "a truncated stream (no message_stop)")
        out = []
        if self.usage:
            out.append(Usage(self.usage.get("input_tokens", 0), self.usage.get("output_tokens", 0),
                             self.usage.get("cache_read_input_tokens", 0),
                             self.usage.get("cache_creation_input_tokens", 0)))
        reason = _STOP_ALIASES.get(self.stop_reason, self.stop_reason)
        if reason not in STOP_REASONS:
            reason = "end_turn"
        out.append(Finish(reason, self.stop_sequence if reason == "stop_sequence" else None))
        return out


def _message_records(msg):
    """A non-streaming Anthropic Message JSON -> equivalent stream records."""
    yield {"type": "message_start", "message": {"usage": msg.get("usage") or {}}}
    for i, b in enumerate(msg.get("content") or []):
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            yield {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}
            yield {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b.get("text")}}
        elif t == "thinking":
            yield {"type": "content_block_start", "index": i,
                   "content_block": {"type": "thinking", "thinking": b.get("thinking") or "",
                                     "signature": b.get("signature") or ""}}
        elif t == "tool_use":
            yield {"type": "content_block_start", "index": i,
                   "content_block": {"type": "tool_use", "id": b.get("id"), "name": b.get("name"),
                                     "input": b.get("input") if isinstance(b.get("input"), dict) else {}}}
        else:
            yield {"type": "content_block_start", "index": i, "content_block": {"type": t}}
        yield {"type": "content_block_stop", "index": i}
    yield {"type": "message_delta", "delta": {"stop_reason": msg.get("stop_reason") or "end_turn",
                                              "stop_sequence": msg.get("stop_sequence")}, "usage": {}}
    yield {"type": "message_stop"}


def _records(resp, ctx):
    """Decoded records from an SSE or JSON upstream response."""
    ctype = (resp.headers.get("content-type") or "").lower()
    if "json" in ctype and "event-stream" not in ctype:
        try:
            msg = json.loads(resp.read().decode("utf-8"))
        except ValueError:
            raise _malformed(ctx, "an unparsable JSON response")
        if not isinstance(msg, dict):
            raise _malformed(ctx, "an unexpected JSON response")
        if msg.get("type") == "error":
            raise _stream_error(ctx, msg)
        for rec in _message_records(msg):
            yield rec
        return
    for sse in iter_sse(resp):
        if not sse.data or sse.data == "[DONE]":
            continue
        try:
            data = json.loads(sse.data)
        except ValueError:
            ctx.runtime.log_once(ctx.log, "passthrough_bad_sse", "[%s] ignoring a non-JSON SSE data line",
                                 ctx.provider.id)
            continue
        if not isinstance(data, dict):
            continue
        if sse.event == "error" and data.get("type") != "error":
            data = {"type": "error", "error": data.get("error") if isinstance(data.get("error"), dict) else data}
        if not data.get("type") and sse.event not in ("message", ""):
            data = dict(data, type=sse.event)
        yield data


# ---------------------------------------------------------------------------------------
# dialect
# ---------------------------------------------------------------------------------------

def _is_missing_endpoint(ctx, exc):
    """Ollama < 0.14 (no /v1/messages route): a bare, non-JSON 404 from the router."""
    p = ctx.provider
    if p.fallback is None or exc.upstream_status != 404:
        return False
    if p.profile != "ollama" and not (p.options or {}).get("fallback_on_404"):
        return False
    text = (exc.upstream_body or "").strip()
    if "page not found" in text.lower():
        return True
    try:
        return not isinstance(json.loads(text), dict)
    except ValueError:
        return True


class AnthropicPassthroughDialect(Dialect):
    name = "anthropic_passthrough"

    def execute(self, ctx):
        if ctx.provider.fallback is not None and ctx.runtime.get(FALLBACK_FLAG):
            for ev in self._delegate(ctx):
                yield ev
            return
        stream = bool(_options(ctx).get("upstream_stream", True))
        body = build_request_body(ctx, stream)
        url = request_url(ctx)
        ctx.trace("upstream_request", provider=ctx.provider.id, dialect=self.name, url=url, model=body["model"],
                  stream=stream)
        try:
            resp = send_with_auth_retry(ctx, "POST", url, build_headers(ctx, stream),
                                        json_dumps_compact(body).encode("utf-8"), stream=True)
        except errors.GatewayError as exc:
            if not _is_missing_endpoint(ctx, exc):
                raise
            ctx.runtime.set(FALLBACK_FLAG, True)
            ctx.runtime.log_once(ctx.log, FALLBACK_FLAG,
                                 "[%s] %s has no Anthropic Messages endpoint (HTTP 404); using %s at %s for this "
                                 "process — upgrade Ollama to >= 0.14", ctx.provider.id, ctx.provider.base_url,
                                 ctx.provider.fallback.dialect, ctx.provider.fallback.base_url)
            ctx.trace("sticky_fallback", provider=ctx.provider.id, flag=FALLBACK_FLAG)
            for ev in self._delegate(ctx):
                yield ev
            return

        emitted = False
        tr = _Translator(ctx)
        try:
            for rec in _records(resp, ctx):
                for ev in tr.feed(rec):
                    emitted = True
                    yield ev
            for ev in tr.finish():
                emitted = True
                yield ev
        except errors.GatewayError as exc:
            if not emitted:
                raise
            yield exc.to_stream_error()
        except TransportError as exc:  # connection dropped mid-stream
            err = errors.map_upstream_error(None, exc.message, {}, ctx.provider.id, ctx.model.id,
                                            ctx.model.context, ctx.est_tokens)
            if not emitted:
                raise err
            yield err.to_stream_error()
        finally:
            resp.close()

    @staticmethod
    def _delegate(ctx):
        from . import get_dialect

        fb = ctx.provider.fallback
        return get_dialect(fb.dialect).execute(ctx.with_provider(fb))
