"""Gemini dialect (DESIGN §3.3): targets ``gemini_api`` (Google AI Studio key) and ``vertex``
(Vertex AI, gcloud ADC). Both speak ``streamGenerateContent?alt=sse`` with identical bodies and
identical SSE chunks (``data: <GenerateContentResponse JSON>``).

Request translation (``GeminiDialect.build_body``):
* ``contents`` roles are only ``user``/``model``; consecutive same-role messages are merged; a
  leading model turn gets a ``user: "(continue)"`` turn prepended; system -> ``systemInstruction``.
* assistant blocks: text -> ``{text}``, tool_use -> ``{functionCall:{name,args,id?}}`` (``id`` only
  for ids that came from Gemini), thinking blocks signed for this target (``fgw1.<target>.{"s":…}``)
  put their ``thoughtSignature`` on the NEXT text/functionCall part (thought text is never sent
  back); a functionCall without one uses the per-provider LRU (tool id -> signature); for
  ``gemini-3*`` the first functionCall of a model turn that still has none gets
  ``skip_thought_signature_validator``.
* user blocks: ALL ``functionResponse`` parts first (one content per turn; ``response`` is
  ``{"output": text}`` or ``{"error": text}``), then tool-result media as ``inlineData``, then the
  user's own text/images/documents.
* tools -> ``functionDeclarations`` with ``parametersJsonSchema`` (schema mode ``gemini_json``),
  names via ``ToolNameMap`` (Gemini regex, leading letter). After a 400 rejecting
  ``parametersJsonSchema`` the provider switches stickily to ``parameters`` + ``gemini_openapi``.
* a 400 / ``MISSING_THOUGHT_SIGNATURE`` about thought signatures (signatures are model-bound, so a
  ``/model`` switch invalidates them; ``-latest`` aliases are not recognised as gemini-3) is retried
  once pre-commit with dummy signatures only, sticky for that session id.

Stream translation (``GeminiStreamParser``): thought text -> ThinkingDelta, text -> TextDelta,
functionCall -> ToolCall; a functionCall part carrying ``thoughtSignature`` (or any signed part
before the first answer text) is preceded by an empty thinking block holding
``ThinkingSignature(fgw1.<target>.{"s": sig})``; signatures on later text parts are dropped so the
answer text stays one block; finishReason / usageMetadata -> Finish / Usage.
"""

import json
import logging
import re
from urllib.parse import quote

from .. import errors
from ..compat import json_dumps_compact
from ..events import Finish, TextDelta, ThinkingDelta, ThinkingSignature, ToolCall, Usage
from ..signatures import decode_signature, encode_signature, keep_thinking_for
from ..transport import TransportError, iter_sse
from .base import LRU, Dialect, join_url, send_with_auth_retry

__all__ = ["GeminiDialect", "GeminiStreamParser", "DUMMY_SIGNATURE", "DEFAULT_BASE_URL", "vertex_host",
           "is_gemini3", "thinking_style"]

LOG = logging.getLogger("ai_gateway")

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DUMMY_SIGNATURE = "skip_thought_signature_validator"
DEFAULT_MAX_OUTPUT = 65536
MAX_STOP_SEQUENCES = 5
SIG_LRU_KEY = "gemini_sig_lru"
FALLBACK_KEY = "gemini_schema_fallback"
SIG_RESET_KEY = "gemini_sig_reset_sessions"  # LRU: session id -> True (send dummy signatures only)
CONTINUE_TEXT = "(continue)"
MALFORMED_NOTE = "[model produced a malformed tool call; please retry]"

_BLOCKED = ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY")
_MALFORMED = ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS")
_VERSION_RE = re.compile(r"/v\d+(?:alpha|beta)?\d*$")
_FAMILY_RE = re.compile(r"^gemini-(\d+)(?:\.(\d+))?")
_SCHEMA_FIELD_RE = re.compile(r"parametersJsonSchema|parameters_json_schema|responseJsonSchema|"
                              r"response_json_schema", re.I)
_UNKNOWN_FIELD_RE = re.compile(r"Unknown name|Cannot find field", re.I)
_DECLARATION_RE = re.compile(r"function_?declarations", re.I)
_API_KEY_RE = re.compile(r"API_KEY_INVALID|API key not valid|API key expired", re.I)
_SIGNATURE_RE = re.compile(r"thought[_ ]?signature", re.I)
_DURATION_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)?\s*$")
# google.rpc.Code name -> HTTP status (in-stream error objects may carry only the status name)
_RPC_HTTP = {"INVALID_ARGUMENT": 400, "FAILED_PRECONDITION": 400, "OUT_OF_RANGE": 400, "UNAUTHENTICATED": 401,
             "PERMISSION_DENIED": 403, "NOT_FOUND": 404, "RESOURCE_EXHAUSTED": 429, "CANCELLED": 499,
             "INTERNAL": 500, "UNKNOWN": 500, "DATA_LOSS": 500, "UNAVAILABLE": 503, "DEADLINE_EXCEEDED": 504}


# ---------------------------------------------------------------------------------------
# model / endpoint helpers
# ---------------------------------------------------------------------------------------

def _bare_model(model_id):
    """``models/x`` / ``publishers/google/models/x`` -> ``x``."""
    m = model_id or ""
    for prefix in ("publishers/google/models/", "models/"):
        if m.startswith(prefix):
            return m[len(prefix):]
    return m


def _family(model_id):
    m = _FAMILY_RE.match(_bare_model(model_id))
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def is_gemini3(model_id):
    """``gemini-3*`` and later: thinkingLevel + strict thought-signature validation."""
    fam = _family(model_id)
    return fam is not None and fam[0] >= 3


def thinking_style(model_id, spec=None):
    """``"level"`` (gemini-3+), ``"budget"`` (2.5, or other models flagged ``reasoning``) or None."""
    fam = _family(model_id)
    if fam is not None and fam[0] >= 3:
        return "level"
    if fam == (2, 5) or (spec is not None and spec.reasoning):
        return "budget"
    return None


def _can_disable_thinking(model_id):
    """2.5 Flash / Flash-Lite accept ``thinkingBudget: 0``; 2.5 Pro only works in thinking mode."""
    return _family(model_id) == (2, 5) and "-pro" not in _bare_model(model_id)


def vertex_host(location):
    loc = (location or "global").strip()
    return "aiplatform.googleapis.com" if loc == "global" else "%s-aiplatform.googleapis.com" % loc


def _with_version(base, version):
    """Append ``/<version>`` unless ``base`` already ends in an API version segment."""
    base = base.rstrip("/")
    return base if _VERSION_RE.search(base) else base + "/" + version


def _sig_lru(runtime):
    return runtime.setdefault(SIG_LRU_KEY, lambda: LRU(4096))


def _upstream_call_id(tool_id, decode):
    """The Gemini ``functionCall.id`` behind an Anthropic tool_use id, or None when the id was not
    issued by Gemini (generated ``toolu_<hex>`` ids, Anthropic-native ``toolu_…`` ids)."""
    if not tool_id:
        return None
    if tool_id.startswith("toolu_"):
        original = decode(tool_id)
        return original if original != tool_id else None
    return tool_id


def _int(v):
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def _duration(v):
    """Google duration ("2s", "1.5s", "250ms", {"seconds": …}) -> seconds or None."""
    if isinstance(v, dict):
        try:
            return float(v.get("seconds") or 0) + float(v.get("nanos") or 0) / 1e9
        except (TypeError, ValueError):
            return None
    m = _DURATION_RE.match(str(v)) if v is not None else None
    if not m:
        return None
    return float(m.group(1)) / (1000.0 if m.group(2) == "ms" else 1.0)


# ---------------------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------------------

def _google_error_obj(text):
    try:
        obj = json.loads(text or "")
    except (TypeError, ValueError):
        return None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    err = obj.get("error") if isinstance(obj, dict) else None
    return err if isinstance(err, dict) else None


def _reclassify_google_429(err, prefix):
    """Google 429 bodies: ``RetryInfo``/``QuotaFailure`` decide transient vs terminal.

    ``errors.map_upstream_error`` also matches "You exceeded your current quota" (the text of every
    AI Studio 429, including per-minute ones) as a terminal billing error; a per-minute quota with
    a short ``retryDelay`` is transient, a ``PerDay`` quota or a delay above
    ``errors.TERMINAL_RETRY_DELAY`` is terminal.
    """
    if err.status != 429 or err.upstream_status != 429:
        return err
    g = _google_error_obj(err.upstream_body)
    if g is None or str(g.get("status", "")).upper() != "RESOURCE_EXHAUSTED":
        return err
    delay, per_day, structured = None, False, False
    for d in g.get("details") or []:
        if not isinstance(d, dict):
            continue
        typ = str(d.get("@type", ""))
        if typ.endswith("RetryInfo"):
            structured = True
            delay = _duration(d.get("retryDelay"))
        elif typ.endswith("QuotaFailure"):
            structured = True
            for v in d.get("violations") or []:
                if isinstance(v, dict) and re.search(r"per ?day", " ".join(
                        str(v.get(k, "")) for k in ("quotaId", "quotaMetric", "description")), re.I):
                    per_day = True
    if not structured:
        return err
    terminal = per_day or (delay is not None and delay > errors.TERMINAL_RETRY_DELAY)
    if terminal != err.should_retry:  # map_upstream_error already agrees
        return err
    msg = " ".join(str(g.get("message") or "quota exceeded").split())
    if terminal:
        return errors.GatewayError(429, "rate_limit_error", prefix + "quota exhausted: " + msg, False,
                                   upstream_status=429, upstream_body=err.upstream_body,
                                   upstream_headers=err.upstream_headers)
    return errors.GatewayError(429, "rate_limit_error", prefix + "rate limited: " + msg, True,
                               delay if delay is not None else err.retry_after, upstream_status=429,
                               upstream_body=err.upstream_body, upstream_headers=err.upstream_headers)


def _schema_rejected(err):
    """400 caused by ``parametersJsonSchema``/``responseJsonSchema`` (older API surface)."""
    if err.upstream_status != 400:
        return False
    text = (err.upstream_body or "") + " " + (err.message or "")
    return bool(_SCHEMA_FIELD_RE.search(text) or (_UNKNOWN_FIELD_RE.search(text) and _DECLARATION_RE.search(text)))


def _api_key_rejected(status, text):
    """AI Studio answers a bad key with 400 ``API_KEY_INVALID`` (not 401): treat it as unauthorized."""
    return status in (400, 403) and bool(_API_KEY_RE.search(text or ""))


def _signature_rejected(err):
    """400 / ``MISSING_THOUGHT_SIGNATURE`` about thought signatures (missing, or minted by another model)."""
    return err.status == 400 and bool(_SIGNATURE_RE.search((err.upstream_body or "") + " " + (err.message or "")))


# ---------------------------------------------------------------------------------------
# request translation
# ---------------------------------------------------------------------------------------

def _text_ok(text):
    return isinstance(text, str) and bool(text.strip())


def _has_parts(msg):
    """True when ``msg`` yields at least one Gemini part (see ``_Converter``)."""
    for b in msg.blocks:
        if b.type == "text" and _text_ok(b.text):
            return True
        if msg.role == "assistant" and b.type == "tool_use":
            return True
        if msg.role != "assistant" and b.type in ("tool_result", "image", "document"):
            return True
    return False


class _Converter(object):
    """NormalizedRequest messages -> Gemini ``contents``.

    ``dummy_only``: ignore every stored signature and give the first functionCall of each model
    turn the dummy (after Gemini rejected the real ones, e.g. following a model switch).
    """

    def __init__(self, names, target, lru, decode_id, names_by_id, dummy_only=False):
        self.names = names
        self.target = target
        self.lru = lru
        self.decode_id = decode_id
        self.names_by_id = names_by_id
        self.dummy_only = dummy_only

    def contents(self, messages, gemini3):
        groups = []  # [role, blocks]
        for m in messages:
            if not _has_parts(m):
                continue
            role = "model" if m.role == "assistant" else "user"
            if groups and groups[-1][0] == role:
                groups[-1][1].extend(m.blocks)
            else:
                groups.append([role, list(m.blocks)])
        out = []
        for role, blocks in groups:
            parts = self._model_parts(blocks) if role == "model" else self._user_parts(blocks)
            if not parts:
                continue
            if out and out[-1]["role"] == role:
                out[-1]["parts"].extend(parts)
            else:
                out.append({"role": role, "parts": parts})
        if not out or out[0]["role"] != "user":
            out.insert(0, {"role": "user", "parts": [{"text": CONTINUE_TEXT}]})
        if gemini3 or self.dummy_only:
            for c in out:
                if c["role"] != "model":
                    continue
                first = next((p for p in c["parts"] if "functionCall" in p), None)
                if first is not None and not first.get("thoughtSignature"):
                    first["thoughtSignature"] = DUMMY_SIGNATURE
        return out

    # ---- model turns ---------------------------------------------------------------------
    def _signature(self, block):
        sig = block.signature
        if self.dummy_only or not keep_thinking_for(sig, self.target):
            return None
        decoded = decode_signature(sig)
        s = decoded[1].get("s") if decoded else None
        return s if isinstance(s, str) and s else None

    def _model_parts(self, blocks):
        parts, pending = [], None
        for b in blocks:
            if b.type == "thinking":
                pending = self._signature(b) or pending
            elif b.type == "text" and _text_ok(b.text):
                part = {"text": b.text}
                if pending:
                    part["thoughtSignature"] = pending
                    pending = None
                parts.append(part)
            elif b.type == "tool_use":
                call = {"name": self.names.upstream(b.name), "args": dict(b.input or {})}
                gid = _upstream_call_id(b.id, self.decode_id)
                if gid:
                    call["id"] = gid
                part = {"functionCall": call}
                sig = pending or (None if self.dummy_only else self.lru.get(b.id))
                pending = None
                if sig:
                    part["thoughtSignature"] = sig
                parts.append(part)
        return parts

    # ---- user turns ----------------------------------------------------------------------
    def _user_parts(self, blocks):
        responses, media, rest = [], [], []
        for b in blocks:
            if b.type == "tool_result":
                output = self._result_output(b, media)
                name = self.names_by_id.get(b.tool_use_id)
                if not name:  # orphaned result: keep the information as plain text
                    rest.append({"text": "[tool result %s]\n%s" % (b.tool_use_id, output)})
                    continue
                fr = {"name": self.names.upstream(name), "response": {"error" if b.is_error else "output": output}}
                gid = _upstream_call_id(b.tool_use_id, self.decode_id)
                if gid:
                    fr["id"] = gid
                responses.append({"functionResponse": fr})
            elif b.type == "text" and _text_ok(b.text):
                rest.append({"text": b.text})
            elif b.type in ("image", "document"):
                rest.append(_media_part(b))
        return responses + media + rest

    @staticmethod
    def _result_output(block, media):
        notes = []
        for m in block.result_media():
            part = _media_part(m)
            if "inlineData" in part:
                media.append(part)
                notes.append("[image attached below]" if m.type == "image" else "[document attached below]")
            else:
                notes.append(part["text"])
        text = block.result_text()
        return "\n".join(([text] if text else []) + notes)


def _media_part(b):
    if b.type == "image":
        if b.data:
            return {"inlineData": {"mimeType": b.media_type or "image/png", "data": b.data}}
        return {"text": "[image: %s]" % (b.url or "unavailable")}
    if b.text is not None:
        return {"text": ("[document: %s]\n" % b.title if b.title else "") + b.text}
    if b.data:
        return {"inlineData": {"mimeType": b.media_type or "application/pdf", "data": b.data}}
    return {"text": "[document: %s]" % (b.url or b.title or "unavailable")}


def _tool_config(choice, names):
    if not isinstance(choice, dict):
        return None
    t = choice.get("type")
    if t == "auto":
        cfg = {"mode": "AUTO"}
    elif t == "any":
        cfg = {"mode": "ANY"}
    elif t == "tool" and choice.get("name"):
        cfg = {"mode": "ANY", "allowedFunctionNames": [names.upstream(choice["name"])]}
    elif t == "none":
        cfg = {"mode": "NONE"}
    else:
        return None
    return {"functionCallingConfig": cfg}


def _thinking_config(req, spec, model_id):
    style = thinking_style(model_id, spec)
    if style is None or req.is_probe():
        return None
    if style == "level":
        return {"includeThoughts": True,
                "thinkingLevel": "low" if req.effort in ("none", "minimal", "low") else "high"}
    budget = 0 if req.effort == "none" and _can_disable_thinking(model_id) else -1
    return {"includeThoughts": True, "thinkingBudget": budget}


# ---------------------------------------------------------------------------------------
# stream translation
# ---------------------------------------------------------------------------------------

class GeminiStreamParser(object):
    """``GenerateContentResponse`` chunks -> internal events (one instance per upstream response).

    ``feed(chunk) -> List[Event]`` per decoded chunk; ``close() -> List[Event]`` once at the end
    (note text, Usage, Finish). Both raise ``errors.GatewayError`` for in-stream errors,
    ``MISSING_THOUGHT_SIGNATURE`` and empty streams; the dialect decides pre/post commit.
    """

    def __init__(self, names, target, lru, provider_id="", model_id="", context_window=None, est_tokens=None):
        from .. import toolnames

        self.names = names
        self.target = target
        self.lru = lru
        self.provider_id = provider_id
        self.model_id = model_id
        self.context_window = context_window
        self.est_tokens = est_tokens
        self._encode_id = toolnames.encode_tool_id
        self._key = 0
        self._cur = None
        self.chunks = 0
        self.tool_calls = 0
        self.text_emitted = False
        self.content_seen = False
        self.malformed = False
        self.finish_reason = None
        self.block_reason = None
        self.usage = None

    @property
    def _prefix(self):
        return "[%s/%s] " % (self.provider_id or "?", self.model_id or "?")

    def _block(self, kind):
        """Key for ``kind`` content: continue the open block of that kind, else open a new one."""
        if self._cur != kind:
            self._key += 1
            self._cur = kind
        return self._key

    def _new_block(self, kind):
        self._cur = None
        return self._block(kind)

    def feed(self, chunk):
        if not isinstance(chunk, dict):
            return []
        if "error" in chunk:
            raise self._stream_error(chunk)
        self.chunks += 1
        if isinstance(chunk.get("usageMetadata"), dict):
            self.usage = chunk["usageMetadata"]
        feedback = chunk.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            self.block_reason = str(feedback["blockReason"])
        cands = chunk.get("candidates")
        cand = cands[0] if isinstance(cands, list) and cands and isinstance(cands[0], dict) else None
        if cand is None:
            return []
        out = []
        content = cand.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        for part in parts if isinstance(parts, list) else []:
            out.extend(self._part(part))
        if cand.get("finishReason"):
            self.finish_reason = str(cand["finishReason"])
            if self.finish_reason == "MISSING_THOUGHT_SIGNATURE":
                raise errors.GatewayError(
                    400, "invalid_request_error",
                    self._prefix + "Gemini rejected the history: a function call is missing its thought signature "
                    "(MISSING_THOUGHT_SIGNATURE)", False)
        return out

    def _part(self, part):
        """Events for one part. A ``thoughtSignature`` becomes an empty signed thinking block emitted
        BEFORE the part — always for a functionCall (also remembered in the LRU), for other parts
        only while no answer text has been emitted in this message: a thinking block must never split
        the answer text (Gemini 3 signs the LAST text part; text-only signatures are not validated)."""
        if not isinstance(part, dict):
            return []
        out = []
        sig = part.get("thoughtSignature")
        sig = sig if isinstance(sig, str) and sig else None
        call = part.get("functionCall")
        if sig and (isinstance(call, dict) or not self.text_emitted):
            key = self._new_block("signature")
            out.append(ThinkingDelta(key, ""))
            out.append(ThinkingSignature(key, encode_signature(self.target, {"s": sig})))
        text = part.get("text")
        if isinstance(text, str) and text:
            self.content_seen = True
            if part.get("thought"):
                out.append(ThinkingDelta(self._block("thinking"), text))
            else:
                self.text_emitted = True
                out.append(TextDelta(self._block("text"), text))
        if isinstance(call, dict):
            self.content_seen = True
            name = call.get("name")
            if not isinstance(name, str) or not name:
                self.malformed = True
                return out
            tool_id = self._encode_id(call.get("id") or None)
            if sig:
                self.lru.put(tool_id, sig)
            self._cur = None
            self.tool_calls += 1
            out.append(ToolCall(tool_id, self.names.original(name), json_dumps_compact(_args(call.get("args")))))
        return out

    def close(self):
        if self.chunks == 0:
            raise errors.GatewayError(502, "api_error", self._prefix + "upstream returned an empty stream", True)
        out = []
        reason = self.finish_reason
        note = None
        if reason in _BLOCKED:
            note = "[response blocked by provider (%s)]" % reason
        elif reason in _MALFORMED or self.malformed:
            note = MALFORMED_NOTE
        elif self.block_reason and not self.content_seen:
            note = "[prompt blocked by provider (%s)]" % self.block_reason
        if note and not self.tool_calls:
            out.append(TextDelta(self._new_block("note"), note))
        if self.usage is not None:
            u = self.usage
            cached = _int(u.get("cachedContentTokenCount"))
            out.append(Usage(max(0, _int(u.get("promptTokenCount")) - cached),
                             _int(u.get("candidatesTokenCount")) + _int(u.get("thoughtsTokenCount")), cached))
        if self.tool_calls:
            stop = "tool_use"
        elif reason == "MAX_TOKENS":
            stop = "max_tokens"
        else:
            stop = "end_turn"
        out.append(Finish(stop))
        return out

    def _stream_error(self, chunk):
        err = chunk.get("error") if isinstance(chunk.get("error"), dict) else {}
        code = err.get("code")
        status = code if isinstance(code, int) and 400 <= code < 600 else \
            _RPC_HTTP.get(str(err.get("status", "")).upper(), 500)
        mapped = errors.map_upstream_error(status, json.dumps(chunk), {}, self.provider_id, self.model_id,
                                           self.context_window, self.est_tokens)
        return _reclassify_google_429(mapped, self._prefix)


def _args(args):
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _iter_chunks(resp):
    """Decoded chunks of an SSE (``alt=sse``) or JSON-array response."""
    ctype = (resp.headers.get("content-type") or "").lower()
    if "json" in ctype and "event-stream" not in ctype:
        try:
            obj = json.loads(resp.text() or "null")
        except ValueError:
            raise errors.GatewayError(502, "api_error", "Gemini returned an unparseable JSON response", True)
        for item in obj if isinstance(obj, list) else [obj]:
            yield item
        return
    for sse in iter_sse(resp):
        data = (sse.data or "").strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except ValueError:
            LOG.debug("gemini: skipping undecodable SSE chunk (%d bytes)", len(data))
            continue
        for item in obj if isinstance(obj, list) else [obj]:
            yield item


# ---------------------------------------------------------------------------------------
# dialect
# ---------------------------------------------------------------------------------------

class GeminiDialect(Dialect):
    name = "gemini"

    # ---- request ---------------------------------------------------------------------------
    @staticmethod
    def target(ctx):
        return ctx.provider.effective_target(ctx.model) or "gemini_api"

    def tool_names(self, ctx):
        from .. import toolnames
        from ..presets import GEMINI_TOOL_NAME_REGEX

        names = ctx.req.all_tool_names()
        choice = ctx.req.tool_choice or {}
        if choice.get("name") and choice["name"] not in names:
            names.append(choice["name"])
        return toolnames.ToolNameMap(names, regex=ctx.provider.tool_name_regex or GEMINI_TOOL_NAME_REGEX,
                                     maxlen=ctx.provider.max_tool_name or 64, leading_letter=True)

    def build_body(self, ctx, schema_fallback=False):
        """-> (Gemini request body, ToolNameMap). No I/O; reads the provider's signature state."""
        from .. import schema, toolnames

        req, spec = ctx.req, ctx.model
        model_id = _bare_model(spec.id)
        names = self.tool_names(ctx)
        conv = _Converter(names, self.target(ctx), _sig_lru(ctx.runtime), toolnames.decode_tool_id,
                          req.tool_use_names_by_id(), dummy_only=self._signatures_reset(ctx))
        body = {"contents": conv.contents(req.messages, is_gemini3(model_id))}
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(req.system)}]}
        mode = ctx.provider.schema_mode or "gemini_json"
        if req.tools and spec.tools and not ctx.provider.chat_only:
            decls = []
            for t in req.tools:
                d = {"name": names.upstream(t.name)}
                if t.description:
                    d["description"] = t.description
                raw = t.input_schema if isinstance(t.input_schema, dict) and t.input_schema else \
                    {"type": "object", "properties": {}}
                if schema_fallback:
                    params = schema.scrub(raw, "gemini_openapi")
                    if params.get("properties") or str(params.get("type", "object")).lower() != "object":
                        d["parameters"] = params  # OpenAPI mode rejects OBJECT without properties
                else:
                    d["parametersJsonSchema"] = schema.scrub(raw, mode)
                decls.append(d)
            body["tools"] = [{"functionDeclarations": decls}]
            tc = _tool_config(req.tool_choice, names)
            if tc:
                body["toolConfig"] = tc
        cap = spec.max_output or DEFAULT_MAX_OUTPUT
        gen = {"maxOutputTokens": max(1, min(int(req.max_tokens or cap), cap))}
        if req.temperature is not None:
            gen["temperature"] = req.temperature
        if req.top_p is not None:
            gen["topP"] = req.top_p
        if req.top_k is not None:
            gen["topK"] = req.top_k
        if req.stop_sequences:
            gen["stopSequences"] = list(req.stop_sequences)[:MAX_STOP_SEQUENCES]
        thinking = _thinking_config(req, spec, model_id)
        if thinking:
            gen["thinkingConfig"] = thinking
        fmt = req.output_format
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema" and isinstance(fmt.get("schema"), dict):
            gen["responseMimeType"] = "application/json"
            if schema_fallback:
                gen["responseSchema"] = schema.scrub(fmt["schema"], "gemini_openapi")
            else:
                gen["responseJsonSchema"] = schema.scrub(fmt["schema"], mode)
        body["generationConfig"] = gen
        return body, names

    def endpoint(self, ctx):
        """-> (url, headers without auth) for the provider's target."""
        model = _bare_model(ctx.model.id)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        headers.update(ctx.provider.headers or {})
        base = (ctx.provider.base_url or "").strip()
        if self.target(ctx) == "vertex":
            project, location = self._vertex_scope(ctx)
            base = _with_version(base or "https://" + vertex_host(location), "v1")
            path = "projects/%s/locations/%s/publishers/google/models/%s:streamGenerateContent" % (
                quote(project, safe=""), quote(location, safe=""), quote(model, safe=""))
            headers.setdefault("x-goog-user-project", project)
        else:
            base = _with_version(base or DEFAULT_BASE_URL, "v1beta")
            resource = model if model.startswith("tunedModels/") else "models/" + quote(model, safe="")
            path = resource + ":streamGenerateContent"
        return join_url(base, path) + "?alt=sse", headers

    def _vertex_scope(self, ctx):
        opts = ctx.provider.options or {}
        project = opts.get("project") or self._auth_value(ctx, "project")
        location = opts.get("location") or self._auth_value(ctx, "location") or "global"
        if not project:
            raise errors.GatewayError(
                401, "authentication_error",
                "[%s/%s] Vertex AI needs a Google Cloud project — set GOOGLE_CLOUD_PROJECT (or the provider's "
                "options.project)" % (ctx.provider.id, ctx.model.id), False)
        return str(project), str(location)

    @staticmethod
    def _auth_value(ctx, name):
        from ..auth.base import AuthError

        fn = getattr(ctx.auth, name, None)
        if not callable(fn):
            return None
        try:
            return fn()
        except AuthError as exc:
            raise errors.GatewayError(401, "authentication_error",
                                      "[%s/%s] %s" % (ctx.provider.id, ctx.model.id, exc), False)

    # ---- execution -------------------------------------------------------------------------
    def execute(self, ctx):
        while True:
            resp, parser, body = self._open(ctx)
            started = False
            try:
                for ev in self._events(ctx, resp, parser):
                    started = True
                    yield ev
                return
            except errors.GatewayError as exc:
                if started:
                    yield exc.to_stream_error()
                    return
                if not self._reset_signatures(ctx, exc, body):  # in-stream MISSING_THOUGHT_SIGNATURE
                    raise
            finally:
                resp.close()

    @staticmethod
    def _signatures_reset(ctx):
        lru = ctx.runtime.get(SIG_RESET_KEY)
        return lru is not None and ctx.session_id in lru

    def _reset_signatures(self, ctx, err, body):
        """After Gemini rejected the history's thought signatures (missing, or minted by another model
        after ``/model``), send only dummy signatures for this session from now on. True => retry once
        (only when that changes ``body``)."""
        if not _signature_rejected(err) or self._signatures_reset(ctx) or not _dummies_would_change(body):
            return False
        ctx.runtime.setdefault(SIG_RESET_KEY, lambda: LRU(1024)).put(ctx.session_id, True)
        ctx.runtime.log_once(ctx.log, SIG_RESET_KEY,
                             "[%s] Gemini rejected the thought signatures of a conversation (e.g. after /model); "
                             "resending it with skip_thought_signature_validator", ctx.provider.id)
        ctx.log.debug("[%s] signature reset for session %s", ctx.provider.id, ctx.session_id)
        return True

    def _open(self, ctx):
        """Send the request (sticky schema fallback, signature reset, 401 refresh); -> (response, parser)."""
        prefix = "[%s/%s] " % (ctx.provider.id, ctx.model.id)
        fallback = bool(ctx.runtime.get(FALLBACK_KEY))
        while True:
            body, names = self.build_body(ctx, fallback)
            url, headers = self.endpoint(ctx)
            ctx.trace("upstream_request", dialect=self.name, target=self.target(ctx), url=url,
                      schema_fallback=fallback)
            try:
                resp = send_with_auth_retry(ctx, "POST", url, headers,
                                            json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                            stream=True, unauthorized=_api_key_rejected)
            except errors.GatewayError as exc:
                if not fallback and _uses_json_schema(body) and _schema_rejected(exc):
                    ctx.runtime.set(FALLBACK_KEY, True)
                    ctx.runtime.log_once(ctx.log, FALLBACK_KEY,
                                         "[%s] Gemini rejected parametersJsonSchema; using `parameters` (OpenAPI "
                                         "subset) for this provider from now on", ctx.provider.id)
                    fallback = True
                    continue
                if self._reset_signatures(ctx, exc, body):
                    continue
                raise _reclassify_google_429(exc, prefix)
            parser = GeminiStreamParser(names, self.target(ctx), _sig_lru(ctx.runtime), ctx.provider.id,
                                        ctx.model.id, ctx.model.context, ctx.est_tokens)
            return resp, parser, body

    @staticmethod
    def _events(ctx, resp, parser):
        try:
            for chunk in _iter_chunks(resp):
                for ev in parser.feed(chunk):
                    yield ev
        except (TransportError, OSError) as exc:
            raise errors.map_upstream_error(None, str(exc), {}, ctx.provider.id, ctx.model.id, ctx.model.context,
                                            ctx.est_tokens)
        for ev in parser.close():
            yield ev


def _dummies_would_change(body):
    """False when every model turn already carries only the dummy on its first functionCall."""
    for c in body.get("contents", []):
        if c.get("role") != "model":
            continue
        first_call = True
        for p in c.get("parts", []):
            sig = p.get("thoughtSignature")
            if "functionCall" in p and first_call:
                first_call = False
                if sig != DUMMY_SIGNATURE:
                    return True
            elif sig:
                return True
    return False


def _uses_json_schema(body):
    if "responseJsonSchema" in body.get("generationConfig", {}):
        return True
    return any("parametersJsonSchema" in d for t in body.get("tools") or [] for d in t.get("functionDeclarations", []))
