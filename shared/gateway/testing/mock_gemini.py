"""Mock Gemini upstreams (DESIGN §7): kinds ``gemini_api`` and ``vertex``.

``gemini_api``  ``POST /v1beta/models/{m}:streamGenerateContent?alt=sse`` — requires
                ``x-goog-api-key`` (missing -> 403 PERMISSION_DENIED; wrong, when ``options["api_key"]``
                is set -> 400 ``API_KEY_INVALID``, like AI Studio).
``vertex``      ``POST /v1/projects/{p}/locations/{l}/publishers/google/models/{m}:streamGenerateContent``
                — requires ``Authorization: Bearer`` (``options["token"]`` to pin the value; wrong ->
                401 UNAUTHENTICATED; API keys -> 401) and ``x-goog-user-project`` (missing -> 403);
                ``options["project"]``/``["location"]`` pin the path. Error bodies are list-wrapped
                (``[{"error": …}]``) as Vertex streaming endpoints do.

Both validate the request like the real API (Google error envelopes): roles only ``user``/``model``,
alternating, first turn ``user``, non-empty parts with exactly one data field, ``functionResponse``
only in user turns with an object ``response``, and the turn after a model turn with N
``functionCall`` parts must be ONE user content with N matching ``functionResponse`` parts; tool
names must match the Gemini regex; ``parametersJsonSchema`` must not contain ``$schema``/``$ref``/
``$defs``/``definitions``; thinkingConfig rules per model generation; unknown fields are rejected.
For ``gemini-3*`` models (or ``options["require_signatures"]``) the first ``functionCall`` part of
every model turn must carry a ``thoughtSignature`` this mock issued FOR THE SAME MODEL (signatures
are model-bound; ``skip_thought_signature_validator`` only with ``options["allow_skip_signature"]``);
with ``options["signature_errors"] == "stream"`` such failures are reported as a 200 stream ending in
``finishReason: MISSING_THOUGHT_SIGNATURE`` instead of a 400.

Options: ``legacy`` (reject ``parametersJsonSchema``/``responseJsonSchema`` with "Unknown name …
Cannot find field" and validate ``parameters`` as the OpenAPI subset), ``mode`` (``"429_transient"``
-> per-minute quota with RetryInfo 2s, ``"429_daily"`` -> PerDay quota, ``"midstream_error"`` -> one
chunk then an in-stream INTERNAL error), ``call_ids`` (functionCall parts carry an ``id``),
``cached`` (cachedContentTokenCount in usage), ``known_models`` (others -> 404).

Replies come from the shared ``Brain`` (turn 1: thought + ``functionCall`` with ``thoughtSignature``;
turn 2: thought + text whose last part carries a ``thoughtSignature``), streamed as realistic
``data: <GenerateContentResponse>\\r\\n\\r\\n`` chunks with ``usageMetadata`` and ``finishReason``.
``server.state["issued"]`` maps every issued signature to its model; ``server.state["sig_checks"]``
lists the validated ones.
"""

import base64
import binascii
import json
import os
import re

from .mock_upstreams import register_kind

__all__ = ["GEMINI_NAME_RE", "DUMMY_SIGNATURE", "gemini_brain_inputs", "validate_request"]

GEMINI_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\-.:]{0,63}$")
DUMMY_SIGNATURE = "skip_thought_signature_validator"

_API_PATH = re.compile(r"^/v1beta/models/([^/:]+):(streamGenerateContent|generateContent)$")
_VERTEX_PATH = re.compile(r"^/v1/projects/([^/]+)/locations/([^/]+)/publishers/google/models/([^/:]+):"
                          r"(streamGenerateContent|generateContent)$")
_TOP_KEYS = {"contents", "systemInstruction", "tools", "toolConfig", "generationConfig", "safetySettings",
             "cachedContent", "labels"}
_GEN_KEYS = {"maxOutputTokens", "temperature", "topP", "topK", "stopSequences", "thinkingConfig", "candidateCount",
             "responseMimeType", "responseSchema", "responseJsonSchema", "presencePenalty", "frequencyPenalty",
             "seed", "responseModalities", "mediaResolution", "responseLogprobs", "logprobs"}
_PART_DATA = ("text", "inlineData", "fileData", "functionCall", "functionResponse", "executableCode",
              "codeExecutionResult")
_PART_KEYS = set(_PART_DATA) | {"thought", "thoughtSignature", "videoMetadata", "partMetadata"}
_OPENAPI_KEYS = {"type", "format", "description", "nullable", "enum", "properties", "required", "items", "minItems",
                 "maxItems", "minimum", "maximum", "minLength", "maxLength", "pattern", "anyOf", "title", "default",
                 "example", "propertyOrdering", "minProperties", "maxProperties"}
_FORBIDDEN_JSON_SCHEMA = ("$schema", "$ref", "$defs", "definitions")
_SUBSCHEMA_ONE = ("items", "additionalProperties", "not", "contains", "if", "then", "else", "propertyNames")
_SUBSCHEMA_LIST = ("anyOf", "oneOf", "allOf", "prefixItems")
_MISSING_SIG = ("Function call is missing a thought_signature in functionCall parts. This is required for tools to "
                "work correctly, and missing thought_signature may lead to degraded model performance. Additional "
                "data, function call `default_api:%s` , position %d. Please refer to "
                "https://ai.google.dev/gemini-api/docs/thought-signatures for more details.")
_QUOTA_MSG = ("You exceeded your current quota, please check your plan and billing details. For more information on "
              "this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.\n* Quota exceeded for metric: "
              "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: %d, model: %s\n"
              "Please retry in %s.")


class _Reject(Exception):
    def __init__(self, code, status, message, details=None):
        Exception.__init__(self, message)
        self.code, self.status, self.message, self.details = code, status, message, details


def _bad(message):
    return _Reject(400, "INVALID_ARGUMENT", message)


def _unknown(name, where):
    return _bad("Invalid JSON payload received. Unknown name \"%s\" at '%s': Cannot find field." % (name, where))


def _is_gemini3(model):
    m = re.match(r"^gemini-(\d+)", model or "")
    return bool(m) and int(m.group(1)) >= 3


def _b64_ok(value):
    if not isinstance(value, str) or not value:
        return False
    padded = value + "=" * (-len(value) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            decoder(padded.encode("ascii"))
            return True
        except (binascii.Error, ValueError):
            continue
    return False


# ---------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------

def _walk_json_schema(node, where):
    if not isinstance(node, dict):
        return
    for k in _FORBIDDEN_JSON_SCHEMA:
        if k in node:
            raise _bad("* GenerateContentRequest.%s: unsupported JSON Schema keyword '%s' in parametersJsonSchema"
                       % (where, k))
    for name, sub in (node.get("properties") or {}).items() if isinstance(node.get("properties"), dict) else []:
        _walk_json_schema(sub, "%s.properties.%s" % (where, name))
    for k in _SUBSCHEMA_ONE:
        _walk_json_schema(node.get(k), "%s.%s" % (where, k))
    for k in _SUBSCHEMA_LIST:
        for i, sub in enumerate(node.get(k) or [] if isinstance(node.get(k), list) else []):
            _walk_json_schema(sub, "%s.%s[%d]" % (where, k, i))


def _walk_openapi(node, where, root=False):
    if not isinstance(node, dict):
        raise _bad("* GenerateContentRequest.%s: schema must be an object" % where)
    for k in node:
        if k not in _OPENAPI_KEYS:
            raise _unknown(k, where)
    typ = str(node.get("type", "")).upper()
    props = node.get("properties")
    if typ == "OBJECT" and not props and root:
        raise _bad("* GenerateContentRequest.%s.properties: should be non-empty for OBJECT type" % where)
    for name, sub in (props or {}).items():
        _walk_openapi(sub, "%s.properties[%s].value" % (where, name))
    if "items" in node:
        _walk_openapi(node["items"], where + ".items")
    for i, sub in enumerate(node.get("anyOf") or []):
        _walk_openapi(sub, "%s.any_of[%d]" % (where, i))


def _validate_tools(body, options):
    names = []
    tools = body.get("tools")
    if tools is None:
        return names
    if not isinstance(tools, list):
        raise _bad("Invalid JSON payload received. 'tools' must be a list.")
    for ti, tool in enumerate(tools):
        decls = tool.get("functionDeclarations") if isinstance(tool, dict) else None
        for di, d in enumerate(decls or []):
            where = "tools[%d].function_declarations[%d]" % (ti, di)
            name = d.get("name") if isinstance(d, dict) else None
            if not isinstance(name, str) or not GEMINI_NAME_RE.match(name):
                raise _bad("* GenerateContentRequest.%s.name: Invalid function name. Must start with a letter or an "
                           "underscore. Must be alphameric (a-z, A-Z, 0-9), underscores (_), dots (.), colons (:), or "
                           "dashes (-), with a maximum length of 64." % where)
            if name in names:
                raise _bad("* GenerateContentRequest.%s.name: Duplicate function declaration name: %s" % (where, name))
            names.append(name)
            for k in d:
                if k not in ("name", "description", "parameters", "parametersJsonSchema", "response",
                             "responseJsonSchema", "behavior"):
                    raise _unknown(k, where)
            if "parameters" in d and "parametersJsonSchema" in d:
                raise _bad("* GenerateContentRequest.%s: parameters and parameters_json_schema are mutually exclusive"
                           % where)
            if "parametersJsonSchema" in d:
                if options.get("legacy"):
                    raise _unknown("parametersJsonSchema", where)
                _walk_json_schema(d["parametersJsonSchema"], where + ".parameters_json_schema")
            if "parameters" in d:
                _walk_openapi(d["parameters"], where + ".parameters", root=True)
    cfg = body.get("toolConfig")
    if cfg is not None:
        fcc = cfg.get("functionCallingConfig") if isinstance(cfg, dict) else None
        if not isinstance(fcc, dict):
            raise _bad("* GenerateContentRequest.tool_config: function_calling_config is required")
        mode = fcc.get("mode", "AUTO")
        if mode not in ("AUTO", "ANY", "NONE", "VALIDATED", "MODE_UNSPECIFIED"):
            raise _bad("Invalid value at 'tool_config.function_calling_config.mode' (type.googleapis.com/"
                       "google.cloud.aiplatform.v1.FunctionCallingConfig.Mode), \"%s\"" % mode)
        allowed = fcc.get("allowedFunctionNames")
        if allowed is not None:
            if mode not in ("ANY", "VALIDATED"):
                raise _bad("* GenerateContentRequest.tool_config.function_calling_config.allowed_function_names: "
                           "only allowed when mode is ANY or VALIDATED")
            for n in allowed:
                if n not in names:
                    raise _bad("* GenerateContentRequest.tool_config.function_calling_config.allowed_function_names: "
                               "function %r is not declared" % n)
    return names


def _validate_generation(body, model, options):
    gen = body.get("generationConfig")
    if gen is None:
        return
    if not isinstance(gen, dict):
        raise _bad("Invalid JSON payload received. 'generationConfig' must be an object.")
    for k in gen:
        if k not in _GEN_KEYS or (options.get("legacy") and k == "responseJsonSchema"):
            raise _unknown(k, "generation_config")
    mot = gen.get("maxOutputTokens")
    if mot is not None and (not isinstance(mot, int) or not 1 <= mot <= 65536):
        raise _bad("Unable to submit request because it has a maxOutputTokens value of %s but the supported range is "
                   "from 1 (inclusive) to 65537 (exclusive). Update the value and try again." % (mot,))
    if len(gen.get("stopSequences") or []) > 5:
        raise _bad("* GenerateContentRequest.generation_config.stop_sequences: at most 5 stop sequences are allowed")
    if "responseJsonSchema" in gen:
        _walk_json_schema(gen["responseJsonSchema"], "generation_config.response_json_schema")
    tc = gen.get("thinkingConfig")
    if tc is None:
        return
    for k in tc:
        if k not in ("includeThoughts", "thinkingBudget", "thinkingLevel"):
            raise _unknown(k, "generation_config.thinking_config")
    if "thinkingLevel" in tc and "thinkingBudget" in tc:
        raise _bad("thinking_budget and thinking_level are not supported together.")
    if "thinkingLevel" in tc:
        if not _is_gemini3(model):
            raise _bad("Thinking level is not supported for this model.")
        if str(tc["thinkingLevel"]).lower() not in ("minimal", "low", "medium", "high"):
            raise _bad("Invalid value at 'generation_config.thinking_config.thinking_level', \"%s\""
                       % tc["thinkingLevel"])
    if tc.get("thinkingBudget") == 0 and "-pro" in model:
        raise _bad("Budget 0 is invalid. This model only works in thinking mode.")


def _validate_part(p, role, ci, pi):
    where = "contents[%d].parts[%d]" % (ci, pi)
    if not isinstance(p, dict):
        raise _bad("Invalid JSON payload received. %s must be an object." % where)
    for k in p:
        if k not in _PART_KEYS:
            raise _unknown(k, where)
    data = [k for k in _PART_DATA if k in p]
    if not data:
        raise _bad("* GenerateContentRequest.%s.data: required oneof field 'data' must have one initialized field"
                    % where)
    if len(data) > 1:
        raise _bad("Invalid JSON payload received. Oneof field 'data' is already set. Cannot set '%s'" % data[1])
    if "text" in p and not isinstance(p["text"], str):
        raise _bad("Invalid value at '%s.text' (TYPE_STRING)" % where)
    if "thoughtSignature" in p and not _b64_ok(p["thoughtSignature"]):
        raise _bad("Invalid value at '%s.thought_signature' (TYPE_BYTES), Base64 decoding failed" % where)
    if "inlineData" in p:
        blob = p["inlineData"]
        if not isinstance(blob, dict) or not blob.get("mimeType") or not _b64_ok(blob.get("data")):
            raise _bad("Invalid value at '%s.inline_data' (TYPE_BYTES), Base64 decoding failed" % where)
    if "functionCall" in p:
        fc = p["functionCall"]
        if role != "model":
            raise _bad("* GenerateContentRequest.%s.function_call: function calls are only allowed in model turns"
                       % where)
        if not isinstance(fc, dict) or not fc.get("name") or not isinstance(fc.get("args", {}), dict):
            raise _bad("* GenerateContentRequest.%s.function_call: name and object args are required" % where)
    if "functionResponse" in p:
        fr = p["functionResponse"]
        if role != "user":
            raise _bad("* GenerateContentRequest.%s.function_response: function responses must be in user turns"
                       % where)
        if not isinstance(fr, dict) or not fr.get("name"):
            raise _bad("* GenerateContentRequest.%s.function_response.name: name is required" % where)
        if not isinstance(fr.get("response"), dict):
            raise _bad("Invalid value at '%s.function_response.response' (type.googleapis.com/google.protobuf.Struct)"
                       ", %s" % (where, json.dumps(fr.get("response"))))


def _validate_signatures(contents, model, options, state):
    required = options.get("require_signatures")
    if required is None:
        required = _is_gemini3(model)
    if not required:
        return
    for ci, c in enumerate(contents):
        if c["role"] != "model":
            continue
        for pi, p in enumerate(c["parts"]):
            if "functionCall" not in p:
                continue
            sig = p.get("thoughtSignature")
            if not sig:
                raise _bad(_MISSING_SIG % (p["functionCall"].get("name"), pi + 1))
            if sig == DUMMY_SIGNATURE:
                if not options.get("allow_skip_signature"):
                    raise _bad("Corrupted thought signature.")
            elif state["issued"].get(sig) != model:
                raise _bad("Corrupted thought signature.")
            state["sig_checks"].append(sig)
            break  # only the first functionCall part of a model turn is validated


def validate_request(body, model, options, state):
    """Raise ``_Reject`` for anything the real API would 400 on (see module docstring)."""
    if not isinstance(body, dict):
        raise _bad("Invalid JSON payload received. Expected an object.")
    for k in body:
        if k not in _TOP_KEYS:
            raise _unknown(k, "")
    contents = body.get("contents")
    if not isinstance(contents, list) or not contents:
        raise _bad("* GenerateContentRequest.contents: contents is not specified")
    prev = None
    for ci, c in enumerate(contents):
        role = c.get("role") if isinstance(c, dict) else None
        if role not in ("user", "model"):
            raise _bad("Please use a valid role: user, model.")
        parts = c.get("parts")
        if not isinstance(parts, list) or not parts:
            raise _bad("* GenerateContentRequest.contents[%d].parts: contents.parts must not be empty." % ci)
        if ci == 0 and role != "user":
            raise _bad("Please ensure that multiturn requests start with a user turn.")
        if role == prev:
            raise _bad("Please ensure that multiturn requests alternate between user and model.")
        prev = role
        for pi, p in enumerate(parts):
            _validate_part(p, role, ci, pi)
    for ci, c in enumerate(contents):
        calls = [p["functionCall"] for p in c["parts"] if "functionCall" in p]
        responses = [p["functionResponse"] for p in c["parts"] if "functionResponse" in p]
        if responses:
            prev_calls = [p["functionCall"] for p in contents[ci - 1]["parts"] if "functionCall" in p] if ci else []
            if not prev_calls:
                raise _bad("Please ensure that function response turn comes immediately after a function call turn.")
        if c["role"] != "model" or not calls or ci + 1 >= len(contents):
            continue
        nxt = [p["functionResponse"] for p in contents[ci + 1]["parts"] if "functionResponse" in p]
        if len(nxt) != len(calls):
            raise _bad("Please ensure that the number of function response parts is equal to the number of function "
                       "call parts of the function call turn.")
        if sorted(fc["name"] for fc in calls) != sorted(fr["name"] for fr in nxt):
            raise _bad("Please ensure that the function response names match the function call names of the "
                       "previous turn.")
        call_ids = set(fc.get("id") for fc in calls if fc.get("id"))
        for fr in nxt:
            if fr.get("id") and fr["id"] not in call_ids:
                raise _bad("Function response id %r does not match any function call id of the previous turn."
                           % fr["id"])
    si = body.get("systemInstruction")
    if si is not None and (not isinstance(si, dict) or not isinstance(si.get("parts"), list) or
                           not all(isinstance(p, dict) and isinstance(p.get("text"), str) for p in si["parts"])):
        raise _bad("* GenerateContentRequest.system_instruction: parts with text are required")
    _validate_tools(body, options)
    _validate_generation(body, model, options)
    _validate_signatures(contents, model, options, state)


# ---------------------------------------------------------------------------------------
# replies
# ---------------------------------------------------------------------------------------

def gemini_brain_inputs(body):
    """(offered tool names, function response texts in order) from a Gemini request body."""
    offered, results = [], []
    for tool in body.get("tools") or []:
        for d in (tool.get("functionDeclarations") or []) if isinstance(tool, dict) else []:
            if isinstance(d, dict) and d.get("name"):
                offered.append(d["name"])
    for c in body.get("contents") or []:
        for p in c.get("parts") or []:
            fr = p.get("functionResponse") if isinstance(p, dict) else None
            if isinstance(fr, dict):
                r = fr.get("response") if isinstance(fr.get("response"), dict) else {}
                v = r.get("output", r.get("error", r))
                results.append(v if isinstance(v, str) else json.dumps(v))
    return offered, results


def _issue_signature(server, model):
    sig = base64.b64encode(os.urandom(48)).decode("ascii")
    with server.lock:
        server.state["issued"][sig] = model
    return sig


def _reply_chunks(server, model, body_len, reply, vertex):
    with server.lock:
        server.state["n"] += 1
        n = server.state["n"]
    meta = {"modelVersion": model, "responseId": "mock-gemini-%d" % n}
    if vertex:
        meta["createTime"] = "2026-10-03T00:00:00.000000Z"
    prompt = max(1, body_len // 4)
    usage = {"promptTokenCount": prompt, "totalTokenCount": prompt}

    def chunk(parts, finish=None, final_usage=None):
        cand = {"content": {"role": "model", "parts": parts}, "index": 0}
        if finish:
            cand["finishReason"] = finish
        out = {"candidates": [cand], "usageMetadata": final_usage or usage}
        out.update(meta)
        return out

    chunks = []
    if reply.thinking:
        chunks.append(chunk([{"text": reply.thinking, "thought": True}]))
    final = {"promptTokenCount": prompt, "candidatesTokenCount": 7, "thoughtsTokenCount": 11,
             "totalTokenCount": prompt + 18}
    if server.options.get("cached"):
        final["cachedContentTokenCount"] = min(prompt, 64)
    if reply.kind == "tool_call":
        call = {"name": reply.tool_name, "args": reply.arguments}
        if server.options.get("call_ids"):
            call["id"] = "fc-%04d" % n
        chunks.append(chunk([{"functionCall": call, "thoughtSignature": _issue_signature(server, model)}], "STOP",
                            final))
    else:
        text = reply.text or ""
        head, tail = text[:len(text) // 2], text[len(text) // 2:]
        if head:
            chunks.append(chunk([{"text": head}]))
        chunks.append(chunk([{"text": tail, "thoughtSignature": _issue_signature(server, model)}], "STOP", final))
    return chunks


def _send_error(resp, vertex, code, status, message, details=None):
    err = {"code": code, "message": message, "status": status}
    if details:
        err["details"] = details
    resp.send_json(code, [{"error": err}] if vertex else {"error": err})


def _quota_details(model, per_day):
    quota_id = "GenerateRequestsPer%sPerProjectPerModel-FreeTier" % ("Day" if per_day else "Minute")
    return [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
            "quotaId": quota_id, "quotaDimensions": {"location": "global", "model": model},
            "quotaValue": "250" if per_day else "10"}]},
        {"@type": "type.googleapis.com/google.rpc.Help", "links": [{
            "description": "Learn more about Gemini API quotas",
            "url": "https://ai.google.dev/gemini-api/docs/rate-limits"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "39s" if per_day else "2s"},
    ]


def _check_auth(server, req, vertex, project):
    opts = server.options
    if vertex:
        if req.header("x-goog-api-key"):
            raise _Reject(401, "UNAUTHENTICATED", "API keys are not supported by this API. Expected OAuth2 access "
                          "token or other authentication credentials that assert a principal. See "
                          "https://cloud.google.com/docs/authentication", [
                              {"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                               "reason": "CREDENTIALS_MISSING", "domain": "googleapis.com"}])
        token = req.bearer()
        if not token or (opts.get("token") is not None and token != opts["token"]):
            raise _Reject(401, "UNAUTHENTICATED", "Request had invalid authentication credentials. Expected OAuth 2 "
                          "access token, login cookie or other valid authentication credential. See "
                          "https://developers.google.com/identity/sign-in/web/devconsole-project.", [
                              {"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                               "reason": "ACCESS_TOKEN_EXPIRED" if token else "CREDENTIALS_MISSING",
                               "domain": "googleapis.com"}])
        if not req.header("x-goog-user-project"):
            raise _Reject(403, "PERMISSION_DENIED", "Your application is authenticating by using local Application "
                          "Default Credentials. The aiplatform.googleapis.com API requires a quota project, which is "
                          "not set by default. To learn how to set your quota project, see "
                          "https://cloud.google.com/docs/authentication/adc-troubleshooting/user-creds .")
        if opts.get("project") is not None and project != opts["project"]:
            raise _Reject(403, "PERMISSION_DENIED", "Permission 'aiplatform.endpoints.predict' denied on resource "
                          "'//aiplatform.googleapis.com/projects/%s' (or it may not exist)." % project)
        return
    key = req.header("x-goog-api-key") or (req.query.get("key") or [None])[0]
    if not key:
        raise _Reject(403, "PERMISSION_DENIED", "Method doesn't allow unregistered callers (callers without "
                      "established identity). Please use API Key or other form of API consumer identity to call "
                      "this API.")
    if opts.get("api_key") is not None and key != opts["api_key"]:
        raise _Reject(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", [
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID",
             "domain": "googleapis.com", "metadata": {"service": "generativelanguage.googleapis.com"}}])


def _make_factory(vertex):
    def factory(server):
        with server.lock:
            server.state.setdefault("issued", {})
            server.state.setdefault("sig_checks", [])
            server.state.setdefault("n", 0)

        def handle(req, resp):
            m = (_VERTEX_PATH if vertex else _API_PATH).match(req.path)
            if req.method != "POST" or not m:
                _send_error(resp, vertex, 404, "NOT_FOUND", "The requested URL %s was not found on this server."
                            % req.path)
                return
            if vertex:
                project, location, model, method = m.groups()
            else:
                project, location, (model, method) = None, None, m.groups()
            opts = server.options
            try:
                _check_auth(server, req, vertex, project)
                if vertex and opts.get("location") is not None and location != opts["location"]:
                    raise _Reject(400, "INVALID_ARGUMENT", "Location %s is not supported for this request." % location)
                known = opts.get("known_models")
                if known is not None and model not in known:
                    msg = ("Publisher Model `projects/%s/locations/%s/publishers/google/models/%s` was not found or "
                           "your project does not have access to it." % (project, location, model)) if vertex else \
                        ("models/%s is not found for API version v1beta, or is not supported for generateContent. "
                         "Call ListModels to see the list of available models and their supported methods." % model)
                    raise _Reject(404, "NOT_FOUND", msg)
                mode = opts.get("mode")
                if mode in ("429_transient", "429_daily"):
                    daily = mode == "429_daily"
                    raise _Reject(429, "RESOURCE_EXHAUSTED", _QUOTA_MSG % (250 if daily else 10, model,
                                                                           "39.2s" if daily else "1.6s"),
                                  _quota_details(model, daily))
                with server.lock:
                    validate_request(req.json, model, opts, server.state)
            except _Reject as rej:
                if opts.get("signature_errors") == "stream" and "thought" in rej.message:
                    w = resp.start_sse(content_type="text/event-stream")
                    w.raw(("data: %s\r\n\r\n" % json.dumps({
                        "candidates": [{"content": {"role": "model", "parts": []}, "index": 0,
                                        "finishReason": "MISSING_THOUGHT_SIGNATURE"}],
                        "modelVersion": model})).encode("utf-8"))
                    w.close()
                    return
                _send_error(resp, vertex, rej.code, rej.status, rej.message, rej.details)
                return
            offered, results = gemini_brain_inputs(req.json)
            reply = server.brain.decide(offered, results, background=not offered)
            chunks = _reply_chunks(server, model, len(req.body or b""), reply, vertex)
            sse = method == "streamGenerateContent" and (req.query.get("alt") or [""])[0] == "sse"
            if not sse:
                resp.send_json(200, chunks if method == "streamGenerateContent" else chunks[-1])
                return
            w = resp.start_sse(content_type="text/event-stream")
            try:
                if mode == "midstream_error":
                    w.raw(("data: %s\r\n\r\n" % json.dumps(chunks[0])).encode("utf-8"))
                    w.raw(("data: %s\r\n\r\n" % json.dumps({"error": {
                        "code": 500, "status": "INTERNAL",
                        "message": "An internal error has occurred. Please retry or report in "
                                   "https://developers.generativeai.google/guide/troubleshooting"}})).encode("utf-8"))
                    return
                for c in chunks:
                    w.raw(("data: %s\r\n\r\n" % json.dumps(c)).encode("utf-8"))
            finally:
                w.close()
        return handle
    return factory


register_kind("gemini_api", _make_factory(vertex=False))
register_kind("vertex", _make_factory(vertex=True))
