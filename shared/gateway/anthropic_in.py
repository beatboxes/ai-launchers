"""Parse an Anthropic Messages request into a ``model.NormalizedRequest`` (DESIGN §3.0).

``parse_messages_request(body, headers) -> NormalizedRequest``

* ``role:"system"`` messages inside ``messages`` (Claude Code 2.1.288, may be the LAST message) are
  folded first with ``model.fold_system_messages_raw``; their ``tool_addition``/``tool_removal``
  blocks are applied to ``tools`` with ``model.apply_tool_changes``.
* ``system``: string or text blocks -> list of non-empty strings; ``cache_control`` dropped; the
  ``x-anthropic-billing-header:`` block removed (``anthropic_passthrough`` works from ``raw``).
* blocks: text, image (base64|url), document (base64 PDF|text|content|url), thinking(+signature),
  redacted_thinking, tool_use, tool_result (string or text/image/document/search_result list),
  ``search_result`` -> text. Unknown blocks and server tools are dropped and noted in ``dropped``
  (``"block:<type>"`` / ``"tool:<type>"``, each once).
* effort: ``output_config.effort``, else derived from ``thinking``; ``output_config.format`` ->
  ``output_format``; session id from ``X-Claude-Code-Session-Id``, else ``metadata.user_id``'s
  ``session_id``. Streaming is decided ONLY from ``body["stream"]`` (Claude Code sends
  ``Accept: application/json`` on streaming requests).
* Malformed input raises ``errors.GatewayError(400, "invalid_request_error", ...)``.
"""

import json

from .errors import GatewayError
from .model import (EFFORTS, Block, Message, NormalizedRequest, ToolDef, apply_tool_changes,
                    fold_system_messages_raw, split_effort_from_thinking)

__all__ = ["parse_messages_request", "BILLING_HEADER_PREFIX", "SESSION_HEADER", "SENSITIVE_HEADERS"]

BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"
SESSION_HEADER = "x-claude-code-session-id"
#: request headers never copied into ``NormalizedRequest.headers``
SENSITIVE_HEADERS = ("authorization", "x-api-key", "proxy-authorization", "cookie")

_TOOL_CHOICES = ("auto", "any", "tool", "none")
_DEFAULT_SCHEMA = {"type": "object", "properties": {}}


def _bad(path, message):
    return GatewayError(400, "invalid_request_error", "%s: %s" % (path, message) if path else message)


class _Notes(object):
    """Ordered, de-duplicated list of dropped-item notes."""

    def __init__(self):
        self.items = []

    def add(self, note):
        if note not in self.items:
            self.items.append(note)


# ---------------------------------------------------------------------------------------
# small validators
# ---------------------------------------------------------------------------------------

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _number(body, key, integer=False):
    v = body.get(key)
    if v is None:
        return None
    if integer:
        if _is_int(v):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        raise _bad(key, "expected an integer")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    raise _bad(key, "expected a number")


def _max_tokens(body):
    if body.get("max_tokens") is None:
        return 32000
    v = _number(body, "max_tokens", integer=True)
    if v < 1:
        raise _bad("max_tokens", "must be greater than or equal to 1")
    return v


def _stream(body):
    v = body.get("stream")
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    raise _bad("stream", "expected a boolean")


def _lower_headers(headers):
    out = {}
    if not headers:
        return out
    items = headers.items() if hasattr(headers, "items") else headers
    for k, v in items:
        lk = str(k).lower()
        if lk in SENSITIVE_HEADERS:
            continue
        v = str(v)
        out[lk] = (out[lk] + ", " + v) if lk in out else v
    return out


def _session_id(lheaders, metadata):
    sid = (lheaders.get(SESSION_HEADER) or "").strip()
    if sid:
        return sid
    if not isinstance(metadata, dict):
        return None
    uid = metadata.get("user_id")
    if isinstance(uid, str) and uid.strip().startswith("{"):
        try:
            uid = json.loads(uid)
        except ValueError:
            return None
    if isinstance(uid, dict):
        sid = uid.get("session_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
    return None


# ---------------------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------------------

def _parse_system(system, notes):
    if system is None:
        return []
    if isinstance(system, str):
        return [system] if system.strip() and not _is_billing(system) else []
    if not isinstance(system, list):
        raise _bad("system", "expected a string or a list of text blocks")
    out = []
    for i, blk in enumerate(system):
        if isinstance(blk, str):
            blk = {"type": "text", "text": blk}
        if not isinstance(blk, dict):
            raise _bad("system.%d" % i, "expected an object")
        if blk.get("type") != "text":
            notes.add("system:%s" % (blk.get("type"),))
            continue
        text = blk.get("text")
        if not isinstance(text, str):
            raise _bad("system.%d.text" % i, "expected a string")
        if text.strip() and not _is_billing(text):
            out.append(text)
    return out


def _is_billing(text):
    return text.lstrip().lower().startswith(BILLING_HEADER_PREFIX)


# ---------------------------------------------------------------------------------------
# content blocks
# ---------------------------------------------------------------------------------------

def _str_field(blk, key, path, required=True):
    v = blk.get(key)
    if v is None and not required:
        return None
    if not isinstance(v, str):
        raise _bad("%s.%s" % (path, key), "expected a string")
    return v


def _image(blk, path, notes):
    src = blk.get("source")
    if not isinstance(src, dict):
        raise _bad(path + ".source", "expected an object")
    st = src.get("type")
    if st == "base64":
        return Block.of_image_base64(_str_field(src, "media_type", path + ".source"),
                                     _str_field(src, "data", path + ".source"))
    if st == "url":
        return Block.of_image_url(_str_field(src, "url", path + ".source"))
    notes.add("image_source:%s" % (st,))
    return None


def _content_text(content):
    """Text of a document ``content`` source / search_result ``content`` (string or blocks)."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "\n".join(parts)


def _document(blk, path, notes):
    src = blk.get("source")
    if not isinstance(src, dict):
        raise _bad(path + ".source", "expected an object")
    title = blk.get("title") if isinstance(blk.get("title"), str) else None
    st = src.get("type")
    if st == "base64":
        media = src.get("media_type") if isinstance(src.get("media_type"), str) else "application/pdf"
        return Block.of_document_base64(media, _str_field(src, "data", path + ".source"), title)
    if st == "text":
        return Block.of_document_text(_str_field(src, "data", path + ".source"), title)
    if st == "content":
        return Block.of_document_text(_content_text(src.get("content")), title)
    if st == "url":
        return Block.of_document_url(_str_field(src, "url", path + ".source"), title)
    notes.add("document_source:%s" % (st,))
    return None


def _search_result(blk):
    lines = []
    if isinstance(blk.get("title"), str) and blk["title"]:
        lines.append("Search result: " + blk["title"])
    if isinstance(blk.get("source"), str) and blk["source"]:
        lines.append("Source: " + blk["source"])
    body = _content_text(blk.get("content"))
    if body:
        lines.append(body)
    return Block.of_text("\n".join(lines))


def _tool_input(v, path):
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v) if v.strip() else {}
        except ValueError:
            raise _bad(path, "tool input must be a JSON object")
        if isinstance(parsed, dict):
            return parsed
    raise _bad(path, "tool input must be an object")


def _result_content(content, path, notes):
    if content is None:
        return []
    if isinstance(content, str):
        return [Block.of_text(content)] if content else []
    if not isinstance(content, list):
        raise _bad(path, "expected a string or a list of content blocks")
    out = []
    for i, blk in enumerate(content):
        p = "%s.%d" % (path, i)
        if not isinstance(blk, dict):
            raise _bad(p, "expected an object")
        t = blk.get("type")
        if t == "text":
            text = _str_field(blk, "text", p)
            if text:
                out.append(Block.of_text(text))
        elif t == "image":
            b = _image(blk, p, notes)
            if b is not None:
                out.append(b)
        elif t == "document":
            b = _document(blk, p, notes)
            if b is not None:
                out.append(b)
        elif t == "search_result":
            out.append(_search_result(blk))
        else:
            notes.add("block:%s" % (t,))
    return out


def _parse_block(blk, path, notes):
    """One message content block -> Block, or None when dropped."""
    if not isinstance(blk, dict):
        raise _bad(path, "expected an object")
    t = blk.get("type")
    if not isinstance(t, str) or not t:
        raise _bad(path + ".type", "field required")
    if t == "text":
        text = _str_field(blk, "text", path)
        return Block.of_text(text) if text else None
    if t == "image":
        return _image(blk, path, notes)
    if t == "document":
        return _document(blk, path, notes)
    if t == "thinking":
        thinking = blk.get("thinking")
        sig = blk.get("signature")
        return Block.of_thinking(thinking if isinstance(thinking, str) else "",
                                 sig if isinstance(sig, str) and sig else None)
    if t == "redacted_thinking":
        return Block.of_redacted_thinking(_str_field(blk, "data", path))
    if t == "tool_use":
        tid = _str_field(blk, "id", path)
        name = _str_field(blk, "name", path)
        if not tid or not name:
            raise _bad(path, "tool_use needs a non-empty id and name")
        return Block.of_tool_use(tid, name, _tool_input(blk.get("input"), path + ".input"))
    if t == "tool_result":
        tid = _str_field(blk, "tool_use_id", path)
        return Block.of_tool_result(tid, _result_content(blk.get("content"), path + ".content", notes),
                                    bool(blk.get("is_error")))
    if t == "search_result":
        return _search_result(blk)
    notes.add("block:%s" % (t,))
    return None


def _parse_messages(raw_messages, notes):
    out = []
    for i, msg in enumerate(raw_messages):
        path = "messages.%d" % i
        role = msg.get("role")
        if role not in ("user", "assistant"):
            raise _bad(path + ".role", "unexpected role %r (expected 'user' or 'assistant')" % (role,))
        content = msg.get("content")
        if isinstance(content, str):
            blocks = [Block.of_text(content)] if content else []
        elif isinstance(content, list):
            blocks = []
            for j, blk in enumerate(content):
                b = _parse_block(blk, "%s.content.%d" % (path, j), notes)
                if b is not None:
                    blocks.append(b)
        elif content is None:
            blocks = []
        else:
            raise _bad(path + ".content", "expected a string or a list of content blocks")
        if blocks:
            out.append(Message(role=role, blocks=blocks))
    return out


# ---------------------------------------------------------------------------------------
# tools / tool_choice
# ---------------------------------------------------------------------------------------

def _parse_tools(raw_tools, notes):
    out, seen = [], set()
    for i, t in enumerate(raw_tools):
        typ = t.get("type")
        if typ not in (None, "custom"):
            notes.add("tool:%s" % (typ,))
            continue
        name = t.get("name")
        if not isinstance(name, str) or not name:
            raise _bad("tools.%d.name" % i, "field required")
        if name in seen:
            continue
        seen.add(name)
        desc = t.get("description")
        schema = t.get("input_schema")
        out.append(ToolDef(name=name, description=desc if isinstance(desc, str) else "",
                           input_schema=schema if isinstance(schema, dict) else dict(_DEFAULT_SCHEMA)))
    return out


def _parse_tool_choice(tc):
    """-> (tool_choice dict or None, disable_parallel_tool_use)."""
    if tc is None:
        return None, False
    if not isinstance(tc, dict):
        raise _bad("tool_choice", "expected an object")
    t = tc.get("type")
    if t not in _TOOL_CHOICES:
        raise _bad("tool_choice.type", "expected one of %s" % ", ".join(_TOOL_CHOICES))
    choice = {"type": t}
    if t == "tool":
        name = tc.get("name")
        if not isinstance(name, str) or not name:
            raise _bad("tool_choice.name", "field required for type 'tool'")
        choice["name"] = name
    return choice, bool(tc.get("disable_parallel_tool_use", False))


def _stop_sequences(body):
    v = body.get("stop_sequences")
    if v is None:
        return []
    if not isinstance(v, list):
        raise _bad("stop_sequences", "expected a list of strings")
    return [s for s in v if isinstance(s, str) and s]


# ---------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------

def parse_messages_request(body, headers=None):
    """Validate and normalize an Anthropic Messages (or count_tokens) request body."""
    if not isinstance(body, dict):
        raise _bad("", "request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _bad("model", "field required")
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise _bad("messages", "at least one message is required")
    for i, m in enumerate(raw_messages):
        if not isinstance(m, dict):
            raise _bad("messages.%d" % i, "expected an object")
    raw_tools = body.get("tools")
    if raw_tools is not None and not isinstance(raw_tools, list):
        raise _bad("tools", "expected a list")
    for i, t in enumerate(raw_tools or []):
        if not isinstance(t, dict):
            raise _bad("tools.%d" % i, "expected an object")

    notes = _Notes()
    fold = fold_system_messages_raw(raw_messages)
    tools = _parse_tools(apply_tool_changes(raw_tools or [], fold), notes)
    messages = _parse_messages(fold.messages, notes)
    if not messages:
        raise _bad("messages", "no message has content")
    system = _parse_system(body.get("system"), notes)
    tool_choice, no_parallel = _parse_tool_choice(body.get("tool_choice"))

    effort, thinking_requested, budget = split_effort_from_thinking(body.get("thinking"))
    output_format = None
    oc = body.get("output_config")
    if isinstance(oc, dict):
        if oc.get("effort") in EFFORTS:
            effort = oc["effort"]
        if isinstance(oc.get("format"), dict):
            output_format = oc["format"]
    if output_format is None and isinstance(body.get("output_format"), dict):  # pre-GA beta spelling
        output_format = body["output_format"]

    lheaders = _lower_headers(headers)
    return NormalizedRequest(
        model=model,
        system=system,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        disable_parallel_tool_use=no_parallel,
        max_tokens=_max_tokens(body),
        temperature=_number(body, "temperature"),
        top_p=_number(body, "top_p"),
        top_k=_number(body, "top_k", integer=True),
        stop_sequences=_stop_sequences(body),
        stream=_stream(body),
        effort=effort,
        thinking_requested=thinking_requested,
        thinking_budget=budget,
        output_format=output_format,
        session_id=_session_id(lheaders, body.get("metadata")),
        raw=body,
        headers=lheaders,
        dropped=notes.items,
    )
