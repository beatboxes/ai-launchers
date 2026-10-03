"""Quirk-enforcing mock upstreams for the Responses dialect (DESIGN §7).

Kinds (``MockServer(kind, brain, options)``), all answering ``POST …/responses`` with a realistic
Responses SSE stream driven by the shared ``Brain`` (turn 1: reasoning + ``function_call`` Bash;
turn 2: reasoning + ``DONE <sha8>`` text; usage incl. ``cached_tokens``), and ``GET …/models``:

``openai_responses``  Bearer required (``options["api_key"]`` = exact key); rejects ``max_tokens``,
                      ``max_output_tokens`` < 16, temperature/top_p on reasoning ids, and
                      ``include:["reasoning.encrypted_content"]`` unless ``store`` is false.
                      Errors ``{"error": {message, type, param, code}}``.
``chatgpt_codex``     Bearer must be a JWT with a future ``exp`` (else 401 ``token_expired``, or
                      ``options["expired_status"]`` e.g. 403); headers ``ChatGPT-Account-ID`` (=
                      ``options["account_id"]`` if set), ``originator: codex_cli_rs``, ``version``,
                      ``session-id``, ``User-Agent: codex_cli_rs/…``, ``Accept: text/event-stream``;
                      ``stream:true``, ``store:false``, non-empty ``instructions`` (Responses-Lite ids:
                      lite header, no instructions, leading developer message); rejects
                      max_output_tokens/temperature/top_p/truncation/user/metadata/previous_response_id/
                      max_completion_tokens with 400 ``{"detail": "Unsupported parameter: …"}``; tool names
                      and call_ids <= 64; turn 2 must echo turn 1's ``encrypted_content`` before its
                      ``function_call`` (``options["require_reasoning_echo"]``, default on);
                      ``options["usage_limit"]`` -> 429 ``usage_limit_reached`` with ``resets_at``.
``grok_proxy``        Bearer + ``X-XAI-Token-Auth: xai-grok-cli``, ``x-authenticateresponse``,
                      ``x-grok-client-version/-identifier``, ``x-grok-client-mode: headless``,
                      ``x-grok-conv-id/-session-id/-req-id``; rejects ``max_output_tokens``/``temperature``
                      and root ``oneOf``/``anyOf``/``allOf`` tool schemas; ``options["version_gate"]``
                      (True = every version, or a minimum "x.y.z") -> ``options["version_gate_status"]``
                      (426 default, or 400) "client version … please upgrade". Errors ``{"code","error"}``.
``xai_responses``     Bearer; rejects ``max_tokens`` and root combinators. Errors ``{"code","error"}``.

Every kind verifies that replayed ``reasoning`` items carry ``encrypted_content`` this server issued
(no item ids — ``store`` is false) and that each ``function_call_output`` follows its ``function_call``.

Other options: ``reasoning`` (default True: emit a 2-part reasoning summary when reasoning was
requested), ``http_error`` ((status, body) for every POST), ``fail`` (``{"code","message"}`` ->
``response.failed`` right after ``response.created``), ``error_event`` (``error`` event after the
reply), ``incomplete`` (reason for ``response.incomplete``), ``truncate`` (end the stream after the
first output item without a terminal event).
"""

import json
import re
import time
import uuid

from .. import catalog
from ..compat import jwt_claims
from .mock_upstreams import register_kind

__all__ = ["KINDS", "INCLUDE_ENCRYPTED", "CODEX_REJECTED_PARAMS", "GROK_REQUIRED_HEADERS"]

KINDS = ("openai_responses", "chatgpt_codex", "grok_proxy", "xai_responses")
INCLUDE_ENCRYPTED = "reasoning.encrypted_content"
CODEX_REJECTED_PARAMS = ("max_output_tokens", "temperature", "top_p", "truncation", "user", "metadata",
                         "previous_response_id", "max_completion_tokens")
CODEX_REQUIRED_HEADERS = ("ChatGPT-Account-ID", "originator", "version", "session-id")
GROK_REQUIRED_HEADERS = {
    "X-XAI-Token-Auth": "xai-grok-cli", "x-authenticateresponse": "authenticate-response",
    "x-grok-client-version": None, "x-grok-client-identifier": None, "x-grok-client-mode": "headless",
    "x-grok-conv-id": None, "x-grok-session-id": None, "x-grok-req-id": None,
}
_FAMILY = {"openai_responses": "openai", "chatgpt_codex": "codex", "grok_proxy": "grok", "xai_responses": "xai"}
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class _Reject(Exception):
    def __init__(self, status, body):
        Exception.__init__(self, "HTTP %d" % status)
        self.status = status
        self.body = body


def _err(kind, status, message, code=None, param=None):
    if kind == "chatgpt_codex":
        return _Reject(status, {"detail": message})
    if kind in ("grok_proxy", "xai_responses"):
        return _Reject(status, {"code": code or "invalid-argument", "error": message})
    return _Reject(status, {"error": {"message": message, "type": "invalid_request_error", "param": param,
                                      "code": code}})


def _vtuple(version):
    return tuple(int(x) for x in re.findall(r"\d+", version or "")[:3]) or (0,)


def _hex(n=24):
    return uuid.uuid4().hex[:n]


def _chunks(text, n=3):
    size = max(1, -(-len(text) // n))
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


# ---------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------

def _check_auth(kind, server, req):
    token = req.bearer()
    opts = server.options
    if kind == "chatgpt_codex":
        claims = jwt_claims(token or "")
        if "exp" not in claims:
            raise _Reject(401, {"detail": "Could not parse your authentication token. Please try signing in again."})
        if float(claims["exp"]) <= time.time():
            raise _Reject(int(opts.get("expired_status", 401)), {
                "error": {"message": "Provided authentication token is expired. Please try signing in again.",
                          "type": None, "code": "token_expired", "param": None}, "status": 401})
        return
    if not token or (opts.get("api_key") and token != opts["api_key"]):
        if kind == "openai_responses":
            raise _Reject(401, {"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error",
                                          "param": None, "code": "invalid_api_key"}})
        raise _Reject(401, {"code": "unauthenticated", "error": "Incorrect API key provided."})


def _check_headers(kind, server, req):
    opts = server.options
    if kind == "chatgpt_codex":
        for name in CODEX_REQUIRED_HEADERS:
            if not req.header(name):
                raise _err(kind, 400, "Missing required header: %s" % name)
        if opts.get("account_id") and req.header("ChatGPT-Account-ID") != opts["account_id"]:
            raise _Reject(403, {"detail": "Account mismatch"})
        if req.header("originator") != "codex_cli_rs":
            raise _err(kind, 400, "Unknown originator")
        if not (req.header("User-Agent") or "").startswith("codex_cli_rs/"):
            raise _err(kind, 400, "Unsupported client")
        if "text/event-stream" not in (req.header("Accept") or ""):
            raise _err(kind, 400, "Accept must include text/event-stream")
    elif kind == "grok_proxy":
        for name, expected in GROK_REQUIRED_HEADERS.items():
            value = req.header(name)
            if not value or (expected is not None and value != expected):
                raise _err(kind, 400, "missing or invalid header: %s" % name, "invalid-argument")
        gate = opts.get("version_gate")
        version = req.header("x-grok-client-version")
        if gate is True or (isinstance(gate, str) and _vtuple(version) < _vtuple(gate)):
            raise _err(kind, int(opts.get("version_gate_status", 426)),
                       "grok client version %s is no longer supported; please upgrade" % version,
                       "client-upgrade-required")


def _root_combinator(schema):
    return isinstance(schema, dict) and any(k in schema for k in ("oneOf", "anyOf", "allOf"))


def _check_body(kind, server, req, body):
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise _err(kind, 400, "Missing required parameter: 'model'.", "missing_required_parameter", "model")
    if not isinstance(body.get("input"), list):
        raise _err(kind, 400, "Invalid type for 'input': expected an array.", "invalid_type", "input")
    if body.get("stream") is not True:
        raise _err(kind, 400, "Stream must be set to true" if kind == "chatgpt_codex" else
                   "mock supports stream:true only")
    tools = body.get("tools") or []
    for t in tools:
        name = t.get("name") if isinstance(t, dict) else None
        if not isinstance(name, str) or t.get("type") != "function" or not _NAME_RE.match(name):
            raise _err(kind, 400, "Invalid 'tools[].name': %r" % (name,), "invalid_value", "tools")
        if kind in ("grok_proxy", "xai_responses") and _root_combinator(t.get("parameters")):
            raise _err(kind, 400, "Invalid tool schema for '%s': root-level oneOf/anyOf/allOf is not supported"
                       % name)
    include = body.get("include") or []
    if kind == "chatgpt_codex":
        for key in CODEX_REJECTED_PARAMS:
            if key in body:
                raise _err(kind, 400, "Unsupported parameter: %s" % key)
        if body.get("store") is not False:
            raise _err(kind, 400, "Store must be set to false")
        lite = catalog.is_responses_lite(body["model"])
        if lite:
            if req.header("x-openai-internal-codex-responses-lite") != "true":
                raise _err(kind, 400, "Responses-Lite model requires the responses-lite header")
            if "instructions" in body:
                raise _err(kind, 400, "Unsupported parameter: instructions")
            first = body["input"][0] if body["input"] else {}
            if first.get("type") != "message" or first.get("role") != "developer":
                raise _err(kind, 400, "Responses-Lite requests must start with a developer message")
        elif not isinstance(body.get("instructions"), str) or not body["instructions"].strip():
            raise _err(kind, 400, "Instructions are required")
        if body.get("reasoning") is not None and INCLUDE_ENCRYPTED not in include:
            raise _err(kind, 400, "include must contain reasoning.encrypted_content when store is false")
        for i, item in enumerate(body["input"]):
            cid = item.get("call_id") if isinstance(item, dict) else None
            if isinstance(cid, str) and len(cid) > 64:
                raise _err(kind, 400, "Invalid 'input[%d].call_id': string too long. Expected a string with "
                                      "maximum length 64" % i)
    elif kind == "openai_responses":
        if "max_tokens" in body:
            raise _err(kind, 400, "Unsupported parameter: 'max_tokens'. In the Responses API, this parameter has "
                                  "moved to 'max_output_tokens'.", "unsupported_parameter", "max_tokens")
        mot = body.get("max_output_tokens")
        if mot is not None and (not isinstance(mot, int) or mot < 16):
            raise _err(kind, 400, "Invalid 'max_output_tokens': integer below minimum value. Expected a value >= "
                                  "16, but got %r instead." % (mot,), "integer_below_min_value", "max_output_tokens")
        if catalog.is_openai_reasoning(body["model"]):
            for key in ("temperature", "top_p"):
                if key in body:
                    raise _err(kind, 400, "Unsupported parameter: '%s' is not supported with this model." % key,
                               "unsupported_parameter", key)
        if INCLUDE_ENCRYPTED in include and body.get("store") is not False:
            raise _err(kind, 400, "Encrypted content is only supported when 'store' is false.", "invalid_value",
                       "include")
    elif kind == "grok_proxy":
        for key in ("max_output_tokens", "temperature"):
            if key in body:
                raise _err(kind, 400, "Argument not supported: %s" % key)
    elif kind == "xai_responses" and "max_tokens" in body:
        raise _err(kind, 400, "Argument not supported: max_tokens")
    _check_items(kind, server, body)


def _check_items(kind, server, body):
    """Replayed reasoning must be ours (store:false -> encrypted_content, no ids); outputs follow calls."""
    require_echo = server.options.get("require_reasoning_echo", kind == "chatgpt_codex")
    with server.lock:
        issued = set(server.state.get("issued_enc", ()))
        calls = dict(server.state.get("calls", {}))
    seen_enc, seen_calls = set(), set()
    for i, item in enumerate(body["input"]):
        if not isinstance(item, dict):
            raise _err(kind, 400, "Invalid 'input[%d]': expected an object." % i)
        t = item.get("type")
        if t == "reasoning":
            if "id" in item:
                raise _err(kind, 400, "Item with id '%s' not found. Items are not persisted when `store` is set "
                                      "to false." % item["id"])
            enc = item.get("encrypted_content")
            if enc not in issued:
                raise _err(kind, 400, "The encrypted content for item input[%d] could not be verified." % i)
            seen_enc.add(enc)
        elif t == "function_call":
            seen_calls.add(item.get("call_id"))
            enc = calls.get(item.get("call_id"))
            if require_echo and enc and enc not in seen_enc:
                raise _err(kind, 400, "Item 'input[%d]' of type 'function_call' was provided without its required "
                                      "'reasoning' item." % i)
        elif t == "function_call_output" and item.get("call_id") not in seen_calls:
            raise _err(kind, 400, "No tool call found for function call output with call_id %s."
                       % item.get("call_id"))


# ---------------------------------------------------------------------------------------
# streaming reply
# ---------------------------------------------------------------------------------------

def _stream(kind, server, req, body, reply, resp):
    opts = server.options
    want_enc = INCLUDE_ENCRYPTED in (body.get("include") or [])
    reasoning = bool(opts.get("reasoning", True)) and (want_enc or body.get("reasoning") is not None)
    results = [it for it in body["input"] if isinstance(it, dict) and it.get("type") == "function_call_output"]
    base = {"id": "resp_" + _hex(32), "object": "response", "created_at": int(time.time()), "model": body["model"],
            "status": "in_progress", "output": [], "usage": None}
    w = resp.start_sse()
    seq = [0]

    def ev(etype, **fields):
        seq[0] += 1
        data = {"type": etype, "sequence_number": seq[0]}
        data.update(fields)
        w.event(etype, data)

    ev("response.created", response=dict(base))
    ev("response.in_progress", response=dict(base))
    if opts.get("fail"):
        ev("response.failed", response=dict(base, status="failed", error=opts["fail"]))
        w.close()
        return
    output, enc = [], None
    if reasoning:
        idx, rs_id = len(output), "rs_" + _hex(32)
        summaries = [reply.thinking or "Considering the request.", "Checking the constraints."]
        ev("response.output_item.added", output_index=idx, item={"id": rs_id, "type": "reasoning", "summary": []})
        for si, text in enumerate(summaries):
            ev("response.reasoning_summary_part.added", item_id=rs_id, output_index=idx, summary_index=si,
               part={"type": "summary_text", "text": ""})
            for chunk in _chunks(text):
                ev("response.reasoning_summary_text.delta", item_id=rs_id, output_index=idx, summary_index=si,
                   delta=chunk)
            ev("response.reasoning_summary_text.done", item_id=rs_id, output_index=idx, summary_index=si, text=text)
            ev("response.reasoning_summary_part.done", item_id=rs_id, output_index=idx, summary_index=si,
               part={"type": "summary_text", "text": text})
        item = {"id": rs_id, "type": "reasoning", "summary": [{"type": "summary_text", "text": s} for s in summaries]}
        if want_enc:
            enc = "enc_%s_%s" % (kind, _hex(32))
            item["encrypted_content"] = enc
            with server.lock:
                server.state.setdefault("issued_enc", set()).add(enc)
        ev("response.output_item.done", output_index=idx, item=item)
        output.append(item)
    if opts.get("truncate"):
        w.close()
        return
    idx = len(output)
    if reply.kind == "tool_call":
        fc_id, call_id = "fc_" + _hex(32), "call_" + _hex(24)
        args = json.dumps(reply.arguments)
        item = {"id": fc_id, "type": "function_call", "status": "in_progress", "arguments": "", "call_id": call_id,
                "name": reply.tool_name}
        ev("response.output_item.added", output_index=idx, item=dict(item))
        for chunk in _chunks(args):
            ev("response.function_call_arguments.delta", item_id=fc_id, output_index=idx, delta=chunk)
        ev("response.function_call_arguments.done", item_id=fc_id, output_index=idx, arguments=args)
        item.update(status="completed", arguments=args)
        ev("response.output_item.done", output_index=idx, item=item)
        with server.lock:
            server.state.setdefault("calls", {})[call_id] = enc
        text = args
    else:
        msg_id, text = "msg_" + _hex(32), reply.text or ""
        ev("response.output_item.added", output_index=idx,
           item={"id": msg_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []})
        ev("response.content_part.added", item_id=msg_id, output_index=idx, content_index=0,
           part={"type": "output_text", "text": "", "annotations": []})
        for chunk in _chunks(text):
            ev("response.output_text.delta", item_id=msg_id, output_index=idx, content_index=0, delta=chunk)
        part = {"type": "output_text", "text": text, "annotations": []}
        ev("response.output_text.done", item_id=msg_id, output_index=idx, content_index=0, text=text)
        ev("response.content_part.done", item_id=msg_id, output_index=idx, content_index=0, part=part)
        item = {"id": msg_id, "type": "message", "status": "completed", "role": "assistant", "content": [part]}
        ev("response.output_item.done", output_index=idx, item=item)
    output.append(item)
    if opts.get("error_event"):
        ev("error", **opts["error_event"])
        w.close()
        return
    inp = max(1, len(req.body or b"") // 4)
    cached = inp // 2 if results else 0
    out_tokens = len(text) // 4 + 8
    usage = {"input_tokens": inp, "input_tokens_details": {"cached_tokens": cached}, "output_tokens": out_tokens,
             "output_tokens_details": {"reasoning_tokens": 4 if reasoning else 0}, "total_tokens": inp + out_tokens}
    if opts.get("incomplete"):
        ev("response.incomplete", response=dict(base, status="incomplete", output=output, usage=usage,
                                                incomplete_details={"reason": opts["incomplete"]}))
    else:
        ev("response.completed", response=dict(base, status="completed", output=output, usage=usage))
    w.close()


# ---------------------------------------------------------------------------------------
# kinds
# ---------------------------------------------------------------------------------------

def _handle(kind, server, req, resp):
    opts = server.options
    path = req.path.rstrip("/")
    if req.method == "GET" and path.endswith("/models"):
        _check_auth(kind, server, req)
        resp.send_json(200, {"object": "list", "data": [
            {"id": m.id, "object": "model", "owned_by": kind} for m in catalog.get_catalog(_FAMILY[kind])]})
        return
    if req.method != "POST" or not path.endswith("/responses"):
        raise _err(kind, 404, "Not found: %s %s" % (req.method, req.path), "not_found")
    if opts.get("http_error"):
        status, body = opts["http_error"]
        raise _Reject(int(status), body)
    _check_auth(kind, server, req)
    _check_headers(kind, server, req)
    if kind == "chatgpt_codex" and opts.get("usage_limit"):
        now = int(time.time())
        raise _Reject(429, {"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached",
                                      "plan_type": "plus", "resets_at": now + 3600, "resets_in_seconds": 3600}})
    body = req.json
    if not isinstance(body, dict):
        raise _err(kind, 400, "We could not parse the JSON body of your request.", "invalid_json")
    _check_body(kind, server, req, body)
    tools = [t["name"] for t in body.get("tools") or []]
    results = [it.get("output") or "" for it in body["input"]
               if isinstance(it, dict) and it.get("type") == "function_call_output"]
    reply = server.brain.decide(tools, results, background=not tools)
    _stream(kind, server, req, body, reply, resp)


def _factory(kind):
    def factory(server):
        def handle(req, resp):
            try:
                _handle(kind, server, req, resp)
            except _Reject as rej:
                resp.send_json(rej.status, rej.body)
        return handle
    return factory


for _kind in KINDS:
    register_kind(_kind, _factory(_kind))
