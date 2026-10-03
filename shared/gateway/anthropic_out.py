"""Internal events -> Anthropic Messages responses (DESIGN §2.2–§2.4).

``SSEEmitter(write, requested_model, message_id=None, input_estimate=0)``
    Streams ``event: <type>\\ndata: <json>\\n\\n`` frames through ``write(bytes)`` (one call per
    event). ``start()`` -> ``message_start`` (echoing the requested model verbatim) + ``ping``;
    ``feed(event)``; ``ping()``; ``finish()`` (idempotent: close block -> ``message_delta`` ->
    ``message_stop``); ``error(err_type, message)``; ``committed``.
``Aggregator(requested_model, message_id=None, input_estimate=0)``
    Same state machine, builds the non-streaming Message JSON. ``result()`` raises the equivalent
    ``GatewayError`` when a ``StreamError`` was fed.
``error_response(err) -> (status, headers, body_bytes)``; ``estimate_tokens(req) -> int``;
``new_message_id() -> "msg_<24hex>"``. ``GatewayError``, ``map_upstream_error`` and
``prompt_too_long`` are re-exported from ``errors`` (never redefined here).

Block grammar (shared by both): exactly one block open at a time; a key or kind change closes
the open block; empty text never opens a block; every thinking block gets exactly one
``signature_delta`` right before its ``content_block_stop`` (``signatures.synthetic_signature``
when the dialect sent none); a ``ToolCall`` is a complete ``tool_use`` block with one
``input_json_delta`` carrying the full arguments; ``stop_reason`` is forced to ``tool_use`` when
any tool call was emitted and defaults to ``end_turn``.
"""

import json
import math
import os

from .compat import json_dumps_compact
from .errors import GatewayError, map_upstream_error, prompt_too_long
from .events import STOP_REASONS, Finish, StreamError, TextDelta, ThinkingDelta, ThinkingSignature, ToolCall, Usage
from .signatures import synthetic_signature

__all__ = [
    "GatewayError", "map_upstream_error", "prompt_too_long", "new_message_id", "new_tool_id", "SSEEmitter",
    "Aggregator", "error_response", "estimate_tokens", "sse_frame", "STREAM_ERROR_STATUS",
    "BYTES_PER_TOKEN", "IMAGE_TOKENS", "PDF_BYTES_PER_TOKEN",
]

BYTES_PER_TOKEN = 3.6
IMAGE_TOKENS = 1600
PDF_BYTES_PER_TOKEN = 750.0

#: HTTP status used when a mid-stream error type has to become a real HTTP error (stream:false)
STREAM_ERROR_STATUS = {
    "invalid_request_error": 400, "authentication_error": 401, "permission_error": 403,
    "not_found_error": 404, "request_too_large": 413, "rate_limit_error": 429, "api_error": 502,
    "overloaded_error": 529,
}


def new_message_id():
    return "msg_" + os.urandom(12).hex()


def new_tool_id():
    return "toolu_" + os.urandom(12).hex()


def sse_frame(event_type, data):
    """One SSE frame ``event: <type>\\ndata: <compact json>\\n\\n`` as UTF-8 bytes."""
    return ("event: %s\ndata: %s\n\n" % (event_type, json_dumps_compact(data))).encode("utf-8", "replace")


def _text(value):
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _tool_args(input_json):
    """-> (json object string, dict). Invalid/non-object arguments become ``{}``."""
    if isinstance(input_json, dict):
        return json_dumps_compact(input_json), input_json
    s = _text(input_json).strip()
    if s:
        try:
            obj = json.loads(s)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            return s, obj
    return "{}", {}


def _ceil_tokens(nbytes):
    return int(math.ceil(nbytes / BYTES_PER_TOKEN)) if nbytes > 0 else 0


class _MessageMachine(object):
    """Event -> Anthropic content-block state machine. Subclasses render the transitions."""

    def __init__(self, requested_model, message_id=None, input_estimate=0):
        self.requested_model = requested_model
        self.message_id = message_id or new_message_id()
        self.input_estimate = max(0, int(input_estimate or 0))
        self.started = False
        self.done = False
        self._index = -1
        self._open = None          # (kind, key) of the open block
        self._thinking = []        # text pieces of the open thinking block
        self._signature = None     # signature of the open thinking block
        self._tool_calls = 0
        self._usage = None
        self._finish = None
        self._out_bytes = 0
        self.stream_error = None   # StreamError fed/raised, if any

    # ---- rendering hooks (subclasses) -------------------------------------------------
    def _r_start(self, message):
        raise NotImplementedError

    def _r_block_start(self, index, block):
        raise NotImplementedError

    def _r_delta(self, index, delta):
        raise NotImplementedError

    def _r_block_stop(self, index):
        raise NotImplementedError

    def _r_message_delta(self, delta, usage):
        raise NotImplementedError

    def _r_message_stop(self):
        raise NotImplementedError

    def _r_error(self, err_type, message):
        raise NotImplementedError

    # ---- state machine -----------------------------------------------------------------
    def _start_message(self):
        message = {
            "id": self.message_id, "type": "message", "role": "assistant", "model": self.requested_model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": self.input_estimate, "output_tokens": 0,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }
        self.started = True
        self._r_start(message)

    def _ensure_started(self):
        if not self.started:
            self._start_message()

    def _close_block(self):
        if self._open is None:
            return
        kind = self._open[0]
        index = self._index
        if kind == "thinking":
            sig = self._signature or synthetic_signature("".join(self._thinking))
            self._r_delta(index, {"type": "signature_delta", "signature": sig})
        self._open = None
        self._thinking = []
        self._signature = None
        self._r_block_stop(index)

    def _open_block(self, kind, key, block):
        self._close_block()
        self._index += 1
        self._open = (kind, key)
        self._r_block_start(self._index, block)

    def _is_open(self, kind, key):
        return self._open is not None and self._open[0] == kind and self._open[1] == key

    def feed(self, event):
        """Consume one internal event (ignored after ``finish()``/``error()``)."""
        if self.done:
            return
        if isinstance(event, StreamError):
            self.error(event.err_type, event.message, event.retryable)
            return
        self._ensure_started()
        if isinstance(event, TextDelta):
            text = _text(event.text)
            if not text:
                return
            if not self._is_open("text", event.key):
                self._open_block("text", event.key, {"type": "text", "text": ""})
            self._out_bytes += len(text.encode("utf-8", "replace"))
            self._r_delta(self._index, {"type": "text_delta", "text": text})
        elif isinstance(event, ThinkingDelta):
            if not self._is_open("thinking", event.key):
                self._open_block("thinking", event.key, {"type": "thinking", "thinking": "", "signature": ""})
            text = _text(event.text)
            if text:
                self._thinking.append(text)
                self._out_bytes += len(text.encode("utf-8", "replace"))
                self._r_delta(self._index, {"type": "thinking_delta", "thinking": text})
        elif isinstance(event, ThinkingSignature):
            if not self._is_open("thinking", event.key):
                self._open_block("thinking", event.key, {"type": "thinking", "thinking": "", "signature": ""})
            sig = _text(event.signature)
            if sig:
                self._signature = sig
        elif isinstance(event, ToolCall):
            args, _ = _tool_args(event.input_json)
            tool_id = _text(event.id) or new_tool_id()
            self._open_block("tool_use", None, {"type": "tool_use", "id": tool_id, "name": _text(event.name),
                                                "input": {}})
            self._out_bytes += len(args.encode("utf-8", "replace"))
            self._r_delta(self._index, {"type": "input_json_delta", "partial_json": args})
            self._close_block()
            self._tool_calls += 1
        elif isinstance(event, Usage):
            self._usage = event
        elif isinstance(event, Finish):
            self._finish = event

    def _stop(self):
        fin = self._finish
        reason = fin.stop_reason if fin is not None else None
        if self._tool_calls:
            reason = "tool_use"
        elif reason not in STOP_REASONS or reason == "tool_use":
            reason = "end_turn"
        seq = fin.stop_sequence if (reason == "stop_sequence" and isinstance(fin.stop_sequence, str)) else None
        return reason, seq

    def _final_usage(self):
        u = self._usage
        inp = u.input_tokens if u is not None else 0
        cache_read = u.cache_read if u is not None else 0
        cache_write = u.cache_write if u is not None else 0
        out = u.output_tokens if u is not None else 0
        if inp + cache_read + cache_write <= 0:
            inp = self.input_estimate
        if out <= 0:
            out = _ceil_tokens(self._out_bytes)
        return {"input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write}

    def finish(self):
        """Close the open block, emit ``message_delta`` + ``message_stop``. Idempotent."""
        if self.done:
            return
        self._ensure_started()
        self._close_block()
        reason, seq = self._stop()
        self._r_message_delta({"stop_reason": reason, "stop_sequence": seq}, self._final_usage())
        self._r_message_stop()
        self.done = True

    def error(self, err_type, message, retryable=None):
        """Terminate with an error (SSE ``error`` event / Aggregator raises). Idempotent."""
        if self.done:
            return
        err_type = err_type or "api_error"
        if retryable is None:
            retryable = err_type == "overloaded_error"
        self.stream_error = StreamError(err_type, _text(message) or err_type, retryable)
        self.done = True
        self._r_error(err_type, self.stream_error.message)


class SSEEmitter(_MessageMachine):
    """Anthropic SSE writer. ``write(bytes)`` must send one chunk and flush."""

    def __init__(self, write, requested_model, message_id=None, input_estimate=0):
        _MessageMachine.__init__(self, requested_model, message_id, input_estimate)
        self._write = write
        self.committed = False
        self.events_written = 0

    def _send(self, event_type, data):
        self.committed = True
        self.events_written += 1
        self._write(sse_frame(event_type, data))

    def start(self):
        """Emit ``message_start`` + one ``ping`` (no-op once started)."""
        self._ensure_started()

    def ping(self):
        if self.done:
            return
        if not self.started:
            self._start_message()
            return
        self._send("ping", {"type": "ping"})

    def _r_start(self, message):
        self._send("message_start", {"type": "message_start", "message": message})
        self._send("ping", {"type": "ping"})

    def _r_block_start(self, index, block):
        self._send("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})

    def _r_delta(self, index, delta):
        self._send("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})

    def _r_block_stop(self, index):
        self._send("content_block_stop", {"type": "content_block_stop", "index": index})

    def _r_message_delta(self, delta, usage):
        self._send("message_delta", {"type": "message_delta", "delta": delta, "usage": usage})

    def _r_message_stop(self):
        self._send("message_stop", {"type": "message_stop"})

    def _r_error(self, err_type, message):
        self._send("error", {"type": "error", "error": {"type": err_type, "message": message}})


class Aggregator(_MessageMachine):
    """Builds the non-streaming Anthropic Message from the same event stream."""

    def __init__(self, requested_model, message_id=None, input_estimate=0):
        _MessageMachine.__init__(self, requested_model, message_id, input_estimate)
        self._message = None
        self._blocks = []

    def _r_start(self, message):
        self._message = message

    def _r_block_start(self, index, block):
        self._blocks.append(dict(block))

    def _r_delta(self, index, delta):
        blk = self._blocks[index]
        t = delta["type"]
        if t == "text_delta":
            blk["text"] += delta["text"]
        elif t == "thinking_delta":
            blk["thinking"] += delta["thinking"]
        elif t == "signature_delta":
            blk["signature"] = delta["signature"]
        elif t == "input_json_delta":
            blk["_json"] = blk.get("_json", "") + delta["partial_json"]

    def _r_block_stop(self, index):
        blk = self._blocks[index]
        if blk.get("type") == "tool_use":
            blk["input"] = _tool_args(blk.pop("_json", "{}"))[1]

    def _r_message_delta(self, delta, usage):
        self._message.update(delta)
        self._message["usage"] = dict(usage)

    def _r_message_stop(self):
        self._message["content"] = self._blocks

    def _r_error(self, err_type, message):
        pass

    def result(self):
        """The Anthropic Message dict; raises ``GatewayError`` if the stream failed."""
        if self.stream_error is not None:
            err = self.stream_error
            raise GatewayError(STREAM_ERROR_STATUS.get(err.err_type, 502), err.err_type, err.message, err.retryable)
        self.finish()
        return self._message


# ---------------------------------------------------------------------------------------
# errors / estimates
# ---------------------------------------------------------------------------------------

def error_response(err):
    """``GatewayError`` (anything else -> 500 api_error) -> (status, headers, body bytes)."""
    if not isinstance(err, GatewayError):
        err = GatewayError(500, "api_error", "internal gateway error (%s)" % type(err).__name__, False)
    body = json_dumps_compact(err.body()).encode("utf-8", "replace")
    headers = {"Content-Type": "application/json"}
    headers.update(err.headers())
    return err.status, headers, body


def _b64_len(data):
    """Decoded size of a base64 payload without decoding it."""
    if not isinstance(data, str):
        return 0
    n = len(data.rstrip("="))
    return n * 3 // 4


def _block_cost(block):
    """-> (text bytes, fixed tokens) for one model.Block."""
    t = block.type
    if t == "text":
        return len(_text(block.text).encode("utf-8", "replace")), 0
    if t == "thinking":
        return len(_text(block.thinking).encode("utf-8", "replace")), 0
    if t == "image":
        return 0, IMAGE_TOKENS
    if t == "document":
        if block.text is not None:
            return len(block.text.encode("utf-8", "replace")), 0
        if block.data is not None:
            return 0, int(math.ceil(_b64_len(block.data) / PDF_BYTES_PER_TOKEN))
        return 0, IMAGE_TOKENS
    if t == "tool_use":
        return len(_text(block.name).encode("utf-8")) + len(json_dumps_compact(block.input or {}).encode("utf-8")), 0
    if t == "tool_result":
        nbytes, fixed = 0, 0
        for b in block.content or []:
            nb, fx = _block_cost(b)
            nbytes += nb
            fixed += fx
        return nbytes, fixed
    return 0, 0


def estimate_tokens(req):
    """Local input-token estimate (DESIGN §2.4): ceil(UTF-8 bytes of system + messages + tools text
    / 3.6) + 1600 per image + PDF bytes / 750."""
    nbytes, fixed = 0, 0
    for s in req.system or []:
        nbytes += len(_text(s).encode("utf-8", "replace"))
    for t in req.tools or []:
        nbytes += len(_text(t.name).encode("utf-8", "replace"))
        nbytes += len(_text(t.description).encode("utf-8", "replace"))
        nbytes += len(json_dumps_compact(t.input_schema or {}).encode("utf-8", "replace"))
    for m in req.messages or []:
        for b in m.blocks:
            nb, fx = _block_cost(b)
            nbytes += nb
            fixed += fx
    return _ceil_tokens(nbytes) + fixed

