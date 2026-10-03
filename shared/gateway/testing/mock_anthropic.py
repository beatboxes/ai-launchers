"""Mock Anthropic-compatible upstream (kind ``anthropic``): DeepSeek / Kimi / OpenRouter / Ollama >= 0.14 /
OpenCode ``/messages`` as seen by ``dialects/anthropic_passthrough``.

Enforced quirks (realistic Anthropic error bodies):
* auth: ``options["auth"]`` = ``"bearer"`` (default) | ``"x-api-key"`` | ``"none"``; ``options["key"]``
  (if set) must match exactly -> else 401 ``authentication_error``;
* ``anthropic-version`` header required (``options["require_version"]``, default True) -> 400;
* ``model`` must be the raw upstream id: no ``claude-via-`` prefix, no ``[1m]`` suffix -> 400; with
  ``options["models"]`` other ids -> 404 ``not_found_error``;
* no ``role:"system"`` messages (they must be folded) -> 400 "Unexpected role";
* every history thinking block must carry a signature this mock issued (``options["signature"]``,
  default ``"mocksig-<n>"`` prefix) -> 400 "Invalid `signature` in `thinking` block";
* ``max_tokens`` <= ``options["max_tokens_limit"]`` when set -> 400.
Paths: ``POST …/v1/messages`` (or ``options["messages_path"]``, matched as a suffix),
``POST …/messages/count_tokens``, ``GET …/v1/models``. ``options["old_ollama"]`` answers the
Messages path with Ollama < 0.14's plain-text ``404 page not found``.

Behaviour switches: ``options["thinking"]`` (default True: a thinking block + ``signature_delta``
precedes tool calls), ``options["fail"] = {"status", "body", "headers", "times"}`` (HTTP error for the
first ``times`` requests, default all), ``options["stream_error"] = {"type", "message"}``
(``event: error`` right after ``message_start``), ``options["force_json"]``
(JSON reply even for ``stream:true``), ``options["truncate"]`` (drop the connection before
``message_stop``). The Brain decides the reply (tool call -> text ``DONE <sha8>``).
"""

import json
import uuid

from .mock_upstreams import anthropic_brain_inputs, register_kind

__all__ = ["DEFAULT_SIGNATURE_PREFIX", "anthropic_factory"]

DEFAULT_SIGNATURE_PREFIX = "mocksig-"


def _error(resp, status, etype, message, headers=None):
    resp.send_json(status, {"type": "error", "error": {"type": etype, "message": message}}, headers)


def _check_auth(server, req):
    style = server.options.get("auth", "bearer")
    expected = server.options.get("key")
    if style == "none":
        return None
    if style == "x-api-key":
        got = req.header("x-api-key")
    else:
        got = req.bearer()
    if not got or (expected is not None and got != expected):
        return "invalid %s" % ("x-api-key" if style == "x-api-key" else "bearer token")
    return None


def _check_body(server, body):
    """-> (status, error_type, message) or None."""
    model = body.get("model")
    if not isinstance(model, str) or not model:
        return 400, "invalid_request_error", "model: Field required"
    if model.lower().startswith("claude-via-") or model.lower().endswith("[1m]"):
        return 400, "invalid_request_error", "model: %r is not a valid model id" % model
    models = server.options.get("models")
    if models and model not in models:
        return 404, "not_found_error", "model: %s" % model
    if not isinstance(body.get("max_tokens"), int) or body["max_tokens"] < 1:
        return 400, "invalid_request_error", "max_tokens: Field required"
    limit = server.options.get("max_tokens_limit")
    if limit and body["max_tokens"] > limit:
        return 400, "invalid_request_error", "max_tokens: %d > %d, which is the maximum allowed number of output " \
                                             "tokens for %s" % (body["max_tokens"], limit, model)
    prefix = server.options.get("signature", DEFAULT_SIGNATURE_PREFIX)
    for i, m in enumerate(body.get("messages") or []):
        role = m.get("role") if isinstance(m, dict) else None
        if role not in ("user", "assistant"):
            return 400, "invalid_request_error", ("messages.%d: Unexpected role %r. The Messages API accepts a "
                                                  "top-level `system` parameter, not \"system\" as an input message "
                                                  "role." % (i, role))
        content = m.get("content")
        for j, b in enumerate(content if isinstance(content, list) else []):
            if isinstance(b, dict) and b.get("type") == "thinking":
                sig = b.get("signature")
                if not isinstance(sig, str) or not sig.startswith(prefix):
                    return 400, "invalid_request_error", "messages.%d.content.%d: Invalid `signature` in " \
                                                         "`thinking` block" % (i, j)
    return None


def _usage_in(body):
    return max(1, len(json.dumps(body.get("messages") or [])) // 4)


def anthropic_factory(server):
    state = server.state
    state.setdefault("count", 0)
    state.setdefault("fail_count", 0)

    def next_n():
        with server.lock:
            state["count"] += 1
            return state["count"]

    def plan(body):
        """-> (content blocks list, stop_reason) for the Brain's decision."""
        tools, results, background = anthropic_brain_inputs(body)
        reply = server.brain.decide(tools, results, background)
        n = next_n()
        prefix = server.options.get("signature", DEFAULT_SIGNATURE_PREFIX)
        blocks = []
        if reply.thinking and server.options.get("thinking", True):
            blocks.append({"type": "thinking", "thinking": reply.thinking, "signature": "%s%d" % (prefix, n)})
        if reply.kind == "tool_call":
            blocks.append({"type": "tool_use", "id": "toolu_mock%018d" % n, "name": reply.tool_name,
                           "input": reply.arguments})
            return blocks, "tool_use"
        blocks.append({"type": "text", "text": reply.text})
        return blocks, "end_turn"

    def stream(resp, body, blocks, stop_reason):
        opts = server.options
        w = resp.start_sse()

        def send(etype, **fields):
            data = {"type": etype}
            data.update(fields)
            w.event(etype, data)

        def delta(i, **d):
            send("content_block_delta", index=i, delta=d)

        send("message_start", message={
            "id": "msg_mock_%s" % uuid.uuid4().hex[:20], "type": "message", "role": "assistant",
            "model": body["model"], "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": _usage_in(body), "output_tokens": 1, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": int(opts.get("cache_read", 0))}})
        send("ping")
        se = opts.get("stream_error")
        if se:
            send("error", error={"type": se.get("type", "overloaded_error"),
                                 "message": se.get("message", "Overloaded")})
            w.close()
            return
        out_tokens = 0
        for i, b in enumerate(blocks):
            if b["type"] == "thinking":
                send("content_block_start", index=i,
                     content_block={"type": "thinking", "thinking": "", "signature": ""})
                for part in (b["thinking"][:8], b["thinking"][8:]):
                    if part:
                        delta(i, type="thinking_delta", thinking=part)
                delta(i, type="signature_delta", signature=b["signature"])
            elif b["type"] == "tool_use":
                send("content_block_start", index=i,
                     content_block={"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}})
                args = json.dumps(b["input"])
                third = max(1, len(args) // 3)
                for part in (args[:third], args[third:2 * third], args[2 * third:]):
                    delta(i, type="input_json_delta", partial_json=part)
                out_tokens += len(args) // 4
            else:
                send("content_block_start", index=i, content_block={"type": "text", "text": ""})
                for part in (b["text"][:4], b["text"][4:]):
                    if part:
                        delta(i, type="text_delta", text=part)
                out_tokens += max(1, len(b["text"]) // 4)
            send("content_block_stop", index=i)
        if opts.get("truncate"):
            resp.close_connection()
            return
        send("message_delta", delta={"stop_reason": stop_reason, "stop_sequence": None},
             usage={"output_tokens": max(1, out_tokens)})
        send("message_stop")
        w.close()

    def message_json(body, blocks, stop_reason):
        return {"id": "msg_mock_%s" % uuid.uuid4().hex[:20], "type": "message", "role": "assistant",
                "model": body["model"], "content": blocks, "stop_reason": stop_reason, "stop_sequence": None,
                "usage": {"input_tokens": _usage_in(body), "output_tokens": 7, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": int(server.options.get("cache_read", 0))}}

    def handle(req, resp):
        opts = server.options
        path = req.path.rstrip("/")
        messages_path = opts.get("messages_path", "/v1/messages")
        if req.method == "GET" and path.endswith("/models"):
            ids = opts.get("models") or ["mock-model"]
            resp.send_json(200, {"data": [{"type": "model", "id": i, "display_name": i,
                                           "created_at": "2026-01-01T00:00:00Z"} for i in ids],
                                 "has_more": False, "first_id": ids[0], "last_id": ids[-1]})
            return
        if req.method != "POST":
            return
        is_count = path.endswith("/messages/count_tokens")
        if not is_count and not path.endswith(messages_path):
            return
        if opts.get("old_ollama"):
            resp.send_text(404, "404 page not found")
            return
        fail = opts.get("fail")
        if fail:
            with server.lock:
                state["fail_count"] += 1
                hit = state["fail_count"] <= int(fail.get("times", 1 << 30))
            if hit:
                fbody = fail.get("body") or {"type": "error", "error": {"type": "api_error", "message": "mock failure"}}
                if isinstance(fbody, str):
                    resp.send_text(int(fail.get("status", 500)), fbody, headers=fail.get("headers"))
                else:
                    resp.send_json(int(fail.get("status", 500)), fbody, fail.get("headers"))
                return
        auth_problem = _check_auth(server, req)
        if auth_problem:
            _error(resp, 401, "authentication_error", auth_problem)
            return
        if opts.get("require_version", True) and req.header("anthropic-version") != "2023-06-01":
            _error(resp, 400, "invalid_request_error", "anthropic-version: header is required")
            return
        body = req.json
        if not isinstance(body, dict):
            _error(resp, 400, "invalid_request_error", "request body must be a JSON object")
            return
        if is_count:
            resp.send_json(200, {"input_tokens": _usage_in(body)})
            return
        problem = _check_body(server, body)
        if problem:
            _error(resp, *problem)
            return
        blocks, stop_reason = plan(body)
        if body.get("stream") and not opts.get("force_json"):
            stream(resp, body, blocks, stop_reason)
        else:
            resp.send_json(200, message_json(body, blocks, stop_reason))

    return handle


register_kind("anthropic", anthropic_factory)
