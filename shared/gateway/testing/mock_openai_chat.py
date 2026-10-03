"""Quirk-enforcing OpenAI Chat Completions mock upstreams (DESIGN §7), driven by the testing Brain.

Kinds (``MockServer(kind, brain, options)``): ``xai_chat``, ``openai_chat``, ``moonshot_chat``,
``deepseek_chat``, ``nvidia_chat``, ``ollama_chat``, ``openrouter_chat``, ``generic_chat``.

Routes: ``POST …/chat/completions``, ``GET …/models``; ``ollama_chat`` also ``GET /api/tags``,
``POST /api/show`` (capabilities) and ``GET /api/version``.

Every kind checks ``Authorization: Bearer <options["api_key"]>`` (default ``DEFAULT_KEYS[kind]``;
``ollama_chat`` checks only when a key is configured), honours ``stream``, validates OpenAI message
structure (a ``tool`` message must answer a ``tool_calls`` id of the preceding assistant turn and
every call must be answered) and tool names, and rejects provider quirks with realistic 4xx bodies:

* ``xai_chat``      reasoning ids (no ``non-reasoning``): 400 on stop/presence_penalty/frequency_penalty;
                    400 on ``reasoning_effort`` unless the id is in ``options["effort_models"]``;
                    400 on root ``anyOf``/``oneOf``/``allOf`` tool schemas; xAI ``{"code","error"}`` bodies.
* ``openai_chat``   ``^(o\\d|gpt-5|gpt-6)``: 400 "Unsupported parameter: 'max_tokens' … use
                    'max_completion_tokens'", 400 on temperature/top_p != 1; reasoning_effort only for
                    those ids and only none/minimal/low/medium/high; tool_choice/parallel_tool_calls need tools.
* ``moonshot_chat`` 400 on any temperature/top_p and on ``tool_choice: "required"``; once it emitted
                    reasoning with a tool call, that assistant turn must come back with ``reasoning_content``.
* ``deepseek_chat`` 400 on image parts; same reasoning_content rule; usage ``prompt_cache_hit_tokens``.
* ``nvidia_chat``   strict tool-name regex (vLLM-style errors); 400 on ``stream_options``; usage in the
                    final chunk.
* ``ollama_chat``   ``options["models"]`` name -> capabilities; 404 unknown model; 400
                    ``"<model>" does not support thinking`` for reasoning_effort on non-thinking models;
                    400 ``… does not support tools``; streams ``reasoning``; whole tool calls; finish "stop".
* ``openrouter_chat`` processing comments, ``reasoning`` object validation, ``reasoning`` deltas, usage
                    with cost when ``usage.include``.
* ``generic_chat``  rejects any body key outside a conservative allow-list (vLLM ``extra_forbidden``).

Options (all kinds): ``api_key``, ``ignore_stream`` (answer JSON to stream:true), ``reasoning``
(force on/off), ``tool_style`` (``split`` | ``id_only_first`` | ``whole`` | ``no_ids``), ``finish_tool``,
``stream_error`` (error object sent mid-stream after a partial text chunk, or right after the role
chunk with ``stream_error_at="start"``), ``fail_status`` +
``fail_body`` (answer every chat request with that error).
State: ``server.state["reasoning_tool_ids"]`` — tool-call ids emitted together with reasoning.
"""

import json
import re
import time
import uuid

from . import SENTINEL
from .mock_upstreams import register_kind

__all__ = ["KINDS", "DEFAULT_KEYS", "DEFAULT_OLLAMA_MODELS", "TOOL_NAME_RE", "chat_brain_inputs"]

KINDS = ("xai_chat", "openai_chat", "moonshot_chat", "deepseek_chat", "nvidia_chat", "ollama_chat",
         "openrouter_chat", "generic_chat")
DEFAULT_KEYS = dict((k, "sk-%s-%s" % (k.split("_")[0], SENTINEL)) for k in KINDS)
DEFAULT_KEYS["ollama_chat"] = None
DEFAULT_OLLAMA_MODELS = {
    "qwen3:8b": ["completion", "tools", "thinking"],
    "llama3.2:3b": ["completion", "tools"],
    "gemma3:4b": ["completion", "vision"],
}
TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_OPENAI_REASONING = re.compile(r"^(o\d|gpt-5|gpt-6)")
_GENERIC_KEYS = frozenset(("model", "messages", "tools", "tool_choice", "max_tokens", "temperature", "top_p", "stop",
                           "stream", "stream_options", "response_format", "n", "seed", "user", "presence_penalty",
                           "frequency_penalty"))


class _Reject(Exception):
    def __init__(self, status, body):
        Exception.__init__(self, status)
        self.status = status
        self.body = body


# ---------------------------------------------------------------------------------------
# error bodies per provider
# ---------------------------------------------------------------------------------------

def _err(kind, message, status=400, code=None, param=None):
    if kind == "xai_chat":
        body = {"code": "Client specified an invalid argument" if status == 400 else "Unauthenticated",
                "error": message}
    elif kind in ("nvidia_chat", "generic_chat"):
        body = {"object": "error", "message": message, "type": "BadRequestError" if status == 400 else "Unauthorized",
                "param": param, "code": status}
    elif kind == "openrouter_chat":
        body = {"error": {"message": message, "code": status}}
    elif kind == "moonshot_chat":
        body = {"error": {"message": message, "type": "invalid_request_error" if status == 400
                          else "invalid_authentication_error"}}
    else:  # openai, deepseek, ollama (OpenAI-compatible bodies)
        body = {"error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}}
    return _Reject(status, body)


def _mask(key):
    return (key[:3] + "***" + key[-2:]) if key and len(key) > 6 else "***"


# ---------------------------------------------------------------------------------------
# brain inputs / validation
# ---------------------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def chat_brain_inputs(body):
    """(offered tool names, tool-result texts in order, background flag) from a chat body."""
    tools = []
    for t in body.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            tools.append(fn["name"])
    results = [_text_of(m.get("content")) for m in body.get("messages") or []
               if isinstance(m, dict) and m.get("role") == "tool"]
    return tools, results, not tools


def _check_messages(kind, msgs):
    if not isinstance(msgs, list) or not msgs:
        raise _err(kind, "'messages' must be a non-empty array", param="messages")
    pending, group = set(), set()
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            raise _err(kind, "Invalid value for 'messages[%d].role'" % i, param="messages")
        role = m["role"]
        if role == "tool":
            tid = m.get("tool_call_id")
            if tid not in group:
                if kind == "xai_chat":
                    raise _err(kind, "tool_call_id %r does not match any tool call of the previous assistant "
                                     "message" % (tid,))
                raise _err(kind, "Invalid parameter: messages with role 'tool' must be a response to a preceeding "
                                 "message with 'tool_calls'.", param="messages.[%d].role" % i)
            if not isinstance(m.get("content"), (str, list)):
                raise _err(kind, "Invalid type for 'messages[%d].content'" % i)
            pending.discard(tid)
            continue
        if pending:
            raise _err(kind, "An assistant message with 'tool_calls' must be followed by tool messages responding to "
                             "each 'tool_call_id'. The following tool_call_ids did not have response messages: %s"
                       % ", ".join(sorted(pending)), param="messages.[%d].role" % i)
        group = set()
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                if not tc.get("id") or not fn.get("name") or not isinstance(fn.get("arguments"), str):
                    raise _err(kind, "Invalid 'messages[%d].tool_calls': id, function.name and function.arguments "
                                     "(string) are required" % i)
                json.loads(fn["arguments"])  # a mock handler error if the gateway sent invalid JSON
                group.add(tc["id"])
            pending = set(group)
        elif role == "assistant" and m.get("content") is None:
            raise _err(kind, "Invalid 'messages[%d]': assistant content is required without tool_calls" % i)
    if pending:
        raise _err(kind, "An assistant message with 'tool_calls' must be followed by tool messages responding to "
                         "each 'tool_call_id'. The following tool_call_ids did not have response messages: %s"
                   % ", ".join(sorted(pending)))


def _check_tools(kind, body):
    tools = body.get("tools")
    if tools is None:
        if body.get("tool_choice") is not None and kind == "openai_chat":
            raise _err(kind, "Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are "
                             "specified.", param="tool_choice")
        if body.get("parallel_tool_calls") is not None and kind == "openai_chat":
            raise _err(kind, "Invalid value for 'parallel_tool_calls': 'parallel_tool_calls' is only allowed when "
                             "'tools' are specified.", param="parallel_tool_calls")
        return
    if not isinstance(tools, list):
        raise _err(kind, "'tools' must be an array", param="tools")
    if not tools and kind != "ollama_chat":
        raise _err(kind, "Invalid 'tools': empty array. Expected an array with minimum length 1, but got an empty "
                         "array instead.", code="empty_array", param="tools")
    for i, t in enumerate(tools):
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or t.get("type") != "function":
            raise _err(kind, "Invalid 'tools[%d]': expected {type: function, function: {...}}" % i)
        name = fn.get("name") or ""
        if kind != "ollama_chat" and not TOOL_NAME_RE.match(name):
            if kind == "xai_chat":
                raise _err(kind, "Invalid function name %r: must match ^[a-zA-Z0-9_-]{1,64}$" % name)
            if kind == "nvidia_chat":
                raise _err(kind, "1 validation error: tools.%d.function.name: String should match pattern "
                                 "'^[a-zA-Z0-9_-]{1,64}$'" % i)
            raise _err(kind, "Invalid 'tools[%d].function.name': string does not match pattern. Expected a string "
                             "that matches the pattern '^[a-zA-Z0-9_-]+$' with at most 64 characters." % i,
                       code="invalid_value", param="tools[%d].function.name" % i)
        params = fn.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise _err(kind, "Invalid 'tools[%d].function.parameters': expected an object" % i)
        if kind == "xai_chat" and isinstance(params, dict):
            for comb in ("anyOf", "oneOf", "allOf"):
                if comb in params:
                    raise _err(kind, "Invalid parameters schema for function %r: root-level %s is not supported"
                               % (name, comb))


def _reasoning_turns(kind, server, msgs):
    """Moonshot/DeepSeek thinking: assistant tool-call turns we answered with reasoning must echo it."""
    with server.lock:
        ids = set(server.state.get("reasoning_tool_ids") or ())
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        if any(tc.get("id") in ids for tc in m["tool_calls"]) and not m.get("reasoning_content"):
            if kind == "moonshot_chat":
                raise _err(kind, "thinking is enabled but reasoning_content is missing in assistant tool call "
                                 "message at index %d" % i)
            raise _err(kind, "Missing `reasoning_content` field in the assistant message at message index %d" % i)


# ---------------------------------------------------------------------------------------
# per-kind behaviour
# ---------------------------------------------------------------------------------------

def _ollama_models(server):
    return server.options.get("models") or DEFAULT_OLLAMA_MODELS


def _validate(kind, server, body):
    if not isinstance(body, dict):
        raise _err(kind, "We could not parse the JSON body of your request.")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise _err(kind, "you must provide a model parameter", param="model")
    msgs = body.get("messages")
    _check_messages(kind, msgs)
    _check_tools(kind, body)
    if body.get("stream_options") is not None and not body.get("stream") and kind == "openai_chat":
        raise _err(kind, "The 'stream_options' parameter is only allowed when 'stream' is enabled.",
                   param="stream_options")
    if kind == "xai_chat":
        if "non-reasoning" not in model:
            for p in ("stop", "presence_penalty", "frequency_penalty"):
                if p in body:
                    raise _err(kind, "Argument not supported on this model: %s (model %s)" % (p, model))
        if "reasoning_effort" in body and model not in (server.options.get("effort_models") or ()):
            raise _err(kind, "Argument not supported on this model: reasoning_effort (model %s)" % model)
    elif kind == "openai_chat":
        reasoning = bool(_OPENAI_REASONING.match(model))
        if reasoning and "max_tokens" in body:
            raise _err(kind, "Unsupported parameter: 'max_tokens' is not supported with this model. Use "
                             "'max_completion_tokens' instead.", code="unsupported_parameter", param="max_tokens")
        if reasoning:
            for p in ("temperature", "top_p"):
                if p in body and body[p] != 1:
                    raise _err(kind, "Unsupported value: '%s' does not support %s with this model. Only the default "
                                     "(1) value is supported." % (p, body[p]), code="unsupported_value", param=p)
        effort = body.get("reasoning_effort")
        if effort is not None and (not reasoning or effort not in ("none", "minimal", "low", "medium", "high")):
            raise _err(kind, "Unsupported value: 'reasoning_effort' does not support %r with this model." % (effort,),
                       code="unsupported_value", param="reasoning_effort")
    elif kind == "moonshot_chat":
        if "temperature" in body:
            raise _err(kind, "invalid temperature: only 1 is allowed for this model")
        if "top_p" in body:
            raise _err(kind, "invalid top_p: only 0.95 is allowed for this model")
        if body.get("tool_choice") == "required":
            raise _err(kind, "tool_choice 'required' is not supported")
        _reasoning_turns(kind, server, msgs)
    elif kind == "deepseek_chat":
        for i, m in enumerate(msgs):
            for p in m.get("content") if isinstance(m.get("content"), list) else ():
                if isinstance(p, dict) and p.get("type") != "text":
                    raise _err(kind, "Failed to deserialize the JSON body into the target type: messages[%d]: "
                                     "unknown variant `%s`, expected `text`" % (i, p.get("type")))
        _reasoning_turns(kind, server, msgs)
    elif kind == "nvidia_chat":
        if "stream_options" in body and server.options.get("reject_stream_options", True):
            raise _err(kind, "1 validation error: stream_options: Extra inputs are not permitted",
                       param="stream_options")
    elif kind == "ollama_chat":
        models = _ollama_models(server)
        if model not in models:
            raise _Reject(404, {"error": {"message": "model %r not found, try pulling it first" % model,
                                          "type": "api_error", "param": None, "code": None}})
        caps = models[model]
        effort = body.get("reasoning_effort")
        if effort is not None:
            if "thinking" not in caps:
                raise _err(kind, "%s does not support thinking" % json.dumps(model))
            if effort not in ("high", "medium", "low", "none"):
                raise _err(kind, "invalid reasoning value: %r (must be \"high\", \"medium\", \"low\", or \"none\")"
                           % (effort,))
        if body.get("tools") and "tools" not in caps:
            raise _err(kind, "registry.ollama.ai/library/%s does not support tools" % model)
    elif kind == "openrouter_chat":
        r = body.get("reasoning")
        if r is not None and (not isinstance(r, dict) or
                              r.get("effort") not in (None, "minimal", "low", "medium", "high", "xhigh")):
            raise _err(kind, "Invalid reasoning parameter: %s" % json.dumps(r))
    elif kind == "generic_chat":
        extra = sorted(k for k in body if k not in _GENERIC_KEYS)
        if extra:
            raise _err(kind, "[{'type': 'extra_forbidden', 'loc': ('body', %r), 'msg': 'Extra inputs are not "
                             "permitted'}]" % extra[0])


def _emits_reasoning(kind, server, body):
    forced = server.options.get("reasoning")
    if forced is not None:
        return bool(forced)
    model = body.get("model") or ""
    if kind == "xai_chat":
        return "non-reasoning" not in model
    if kind in ("moonshot_chat", "deepseek_chat"):
        return True
    if kind == "ollama_chat":
        return body.get("reasoning_effort") not in (None, "none")
    if kind == "openrouter_chat":
        return isinstance(body.get("reasoning"), dict)
    return False


_REASONING_FIELD = {"ollama_chat": "reasoning", "openrouter_chat": "reasoning"}
_TOOL_STYLE = {"xai_chat": "whole", "ollama_chat": "whole", "deepseek_chat": "id_only_first",
               "generic_chat": "id_only_first"}


def _tool_id(kind, name, n):
    h = uuid.uuid4().hex
    if kind == "moonshot_chat":
        return "functions.%s:%d" % (name, n)          # Kimi-style ids ('.' and ':' -> encoded by the gateway)
    if kind == "deepseek_chat":
        return "call_00_" + h[:24]
    if kind == "nvidia_chat":
        return "chatcmpl-tool-" + h
    if kind == "openrouter_chat":
        return "toolu_vrtx_01" + h[:22]
    if kind == "xai_chat":
        return "call_%08d" % (int(h[:8], 16) % 100000000)
    return "call_" + h[:24]


def _usage(kind, body, completion, reasoning_tokens):
    prompt = max(1, len(json.dumps(body.get("messages") or [])) // 4)
    if kind == "deepseek_chat":
        hit = prompt // 2
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
                "prompt_cache_hit_tokens": hit, "prompt_cache_miss_tokens": prompt - hit}
    if kind == "moonshot_chat":
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
                "cached_tokens": prompt // 3}
    u = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    if kind in ("openai_chat", "xai_chat", "openrouter_chat"):
        u["prompt_tokens_details"] = {"cached_tokens": prompt // 4}
    if kind == "xai_chat":  # xAI: reasoning tokens are outside completion_tokens but inside total_tokens
        u["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        u["total_tokens"] += reasoning_tokens
    if kind == "openrouter_chat":
        u["cost"] = 0.000123
    return u


def _pieces(text, n=3):
    if not text:
        return []
    step = max(1, (len(text) + n - 1) // n)
    return [text[i:i + step] for i in range(0, len(text), step)]


class _Chat(object):
    """One brain-driven reply rendered in a kind's wire format."""

    def __init__(self, kind, server, body):
        self.kind = kind
        self.server = server
        self.body = body
        self.model = body.get("model")
        self.id = "chatcmpl-" + uuid.uuid4().hex[:20]
        self.created = int(time.time())
        tools, results, background = chat_brain_inputs(body)
        self.reply = server.brain.decide(tools, results, background)
        self.reasoning = None
        if _emits_reasoning(kind, server, body):  # thinking-first providers reason before every answer
            always = kind in ("moonshot_chat", "deepseek_chat", "xai_chat")
            self.reasoning = self.reply.thinking or ("Composing the answer." if always else None)
        self.rfield = _REASONING_FIELD.get(kind, "reasoning_content")
        self.call = None
        if self.reply.kind == "tool_call":
            n = sum(len(m.get("tool_calls") or []) for m in body.get("messages") or [] if isinstance(m, dict))
            self.call = {"id": _tool_id(kind, self.reply.tool_name, n), "name": self.reply.tool_name,
                         "arguments": json.dumps(self.reply.arguments, separators=(",", ":"))}
            if self.reasoning:
                with server.lock:
                    server.state.setdefault("reasoning_tool_ids", set()).add(self.call["id"])
        self.finish = (server.options.get("finish_tool") or ("stop" if kind == "ollama_chat" else "tool_calls")) \
            if self.call else "stop"
        produced = self.call["arguments"] if self.call else (self.reply.text or "")
        self.usage = _usage(kind, body, max(1, len(produced) // 4), len(self.reasoning or "") // 4)

    def chunk(self, delta, finish=None, **extra):
        c = {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if self.kind == "ollama_chat":
            c["system_fingerprint"] = "fp_ollama"
        c.update(extra)
        return c

    def tool_chunks(self):
        style = self.server.options.get("tool_style") or _TOOL_STYLE.get(self.kind, "split")
        call = self.call
        head = {"index": 0, "type": "function"}
        if style != "no_ids":
            head["id"] = call["id"]
        if style == "whole":
            head["function"] = {"name": call["name"], "arguments": call["arguments"]}
            return [{"tool_calls": [head]}]
        out = []
        if style == "id_only_first":
            out.append({"tool_calls": [head]})
            out.append({"tool_calls": [{"index": 0, "function": {"name": call["name"]}}]})
        else:
            head["function"] = {"name": call["name"], "arguments": ""}
            out.append({"tool_calls": [head]})
        for piece in _pieces(call["arguments"]):
            out.append({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]})
        return out

    def stream(self, resp):
        w = resp.start_sse()
        try:
            if self.kind == "openrouter_chat":
                w.comment("OPENROUTER PROCESSING")
            first = {"role": "assistant"} if self.kind == "xai_chat" else {"role": "assistant", "content": ""}
            w.data(self.chunk(first))
            err = self.server.options.get("stream_error")
            if err:
                if self.server.options.get("stream_error_at", "middle") != "start":
                    w.data(self.chunk({"content": "partial answer "}))
                w.data(err)
                return
            for piece in _pieces(self.reasoning or "", 2):
                w.data(self.chunk({self.rfield: piece}))
            for piece in _pieces(self.reply.text or ""):
                w.data(self.chunk({"content": piece}))
            if self.call:
                for delta in self.tool_chunks():
                    w.data(self.chunk(delta))
            body_usage = (self.body.get("stream_options") or {}).get("include_usage") or \
                (self.kind == "openrouter_chat" and (self.body.get("usage") or {}).get("include"))
            if self.kind == "moonshot_chat":
                fin = self.chunk({}, self.finish)
                fin["choices"][0]["usage"] = self.usage
                w.data(fin)
            elif self.kind == "nvidia_chat":
                w.data(self.chunk({}, self.finish, usage=self.usage))
            else:
                w.data(self.chunk({}, self.finish))
                if body_usage:
                    w.data({"id": self.id, "object": "chat.completion.chunk", "created": self.created,
                            "model": self.model, "choices": [], "usage": self.usage})
            w.done()
        finally:
            w.close()

    def message(self):
        msg = {"role": "assistant", "content": self.reply.text}
        if self.reasoning:
            msg[self.rfield] = self.reasoning
        if self.call:
            msg["tool_calls"] = [{"id": self.call["id"], "type": "function",
                                  "function": {"name": self.call["name"], "arguments": self.call["arguments"]}}]
        return {"id": self.id, "object": "chat.completion", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "message": msg, "finish_reason": self.finish}], "usage": self.usage}


# ---------------------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------------------

def _authorized(kind, server, req):
    expected = server.options.get("api_key", DEFAULT_KEYS.get(kind))
    if not expected:
        return
    got = req.bearer()
    if got == expected:
        return
    if not got:
        msg = "You didn't provide an API key. You need to provide your API key in an Authorization header using " \
              "Bearer auth (i.e. Authorization: Bearer YOUR_KEY)."
    else:
        msg = "Incorrect API key provided: %s." % _mask(got)
    if kind == "openai_chat":
        raise _Reject(401, {"error": {"message": msg, "type": "invalid_request_error", "param": None,
                                      "code": "invalid_api_key"}})
    if kind == "moonshot_chat":
        raise _Reject(401, {"error": {"message": "Invalid Authentication", "type": "invalid_authentication_error"}})
    if kind == "nvidia_chat":
        raise _Reject(401, {"status": 401, "title": "Unauthorized", "detail": "Authentication failed"})
    if kind == "openrouter_chat":
        raise _Reject(401, {"error": {"message": "No auth credentials found" if not got else "User not found.",
                                      "code": 401}})
    raise _err(kind, msg, status=401, code="invalid_api_key")


def _factory(kind):
    def factory(server):
        def handle(req, resp):
            try:
                _route(kind, server, req, resp)
            except _Reject as rej:
                resp.send_json(rej.status, rej.body)
        return handle
    return factory


def _route(kind, server, req, resp):
    path = req.path.rstrip("/")
    if kind == "ollama_chat" and path.startswith("/api/"):
        if server.options.get("api_key"):
            _authorized(kind, server, req)
        _ollama_api(server, req, resp, path)
        return
    _authorized(kind, server, req)
    if req.method == "GET" and path.endswith("/models"):
        ids = list(_ollama_models(server)) if kind == "ollama_chat" else \
            list(server.options.get("model_ids") or ["mock-model"])
        resp.send_json(200, {"object": "list", "data": [{"id": i, "object": "model", "created": 0, "owned_by": kind}
                                                        for i in ids]})
        return
    if req.method != "POST" or not path.endswith("/chat/completions"):
        return  # framework answers 404
    if server.options.get("fail_status"):
        resp.send_json(int(server.options["fail_status"]), server.options.get("fail_body") or {})
        return
    body = req.json
    _validate(kind, server, body)
    chat = _Chat(kind, server, body)
    if body.get("stream") and not server.options.get("ignore_stream"):
        chat.stream(resp)
    else:
        resp.send_json(200, chat.message())


def _ollama_api(server, req, resp, path):
    models = _ollama_models(server)
    if path == "/api/version":
        resp.send_json(200, {"version": server.options.get("version", "0.12.6")})
    elif path == "/api/tags" and req.method == "GET":
        resp.send_json(200, {"models": [{"name": n, "model": n, "modified_at": "2026-09-01T00:00:00Z",
                                         "size": 5200000000, "digest": "sha256:" + "0" * 64,
                                         "details": {"format": "gguf", "family": n.split(":")[0]}}
                                        for n in models]})
    elif path == "/api/show" and req.method == "POST":
        body = req.json or {}
        name = body.get("model") or body.get("name")
        with server.lock:
            server.state["show_calls"] = server.state.get("show_calls", 0) + 1
        if name not in models:
            resp.send_json(404, {"error": "model '%s' not found" % name})
            return
        resp.send_json(200, {"capabilities": list(models[name]), "details": {"format": "gguf"},
                             "model_info": {"general.architecture": name.split(":")[0]}})


for _kind in KINDS:
    register_kind(_kind, _factory(_kind))
