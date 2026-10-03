"""Normalized (parsed) Anthropic Messages request — the input every dialect translates from.

``anthropic_in.parse_messages_request(body, headers) -> NormalizedRequest`` (owned by the core
agent) produces these objects; this module only defines the target structure, its invariants and
small helpers. Nothing here performs I/O.

Invariants of a ``NormalizedRequest`` produced by ``anthropic_in``
------------------------------------------------------------------
* ``system``: list of non-empty strings in order; ``cache_control`` dropped; the
  ``x-anthropic-billing-header:`` block removed (``anthropic_passthrough`` uses ``raw`` instead).
* ``messages``: roles are only ``"user"`` / ``"assistant"``. Claude Code 2.1.288 also sends
  ``role:"system"`` messages *inside* ``messages`` (beta ``mid-conversation-system-2026-04-07``,
  sometimes as the LAST message); they are folded by ``fold_system_messages_raw`` BEFORE parsing:
  their text becomes ``<system-reminder>`` text blocks of the adjacent user message, their
  ``tool_addition``/``tool_removal`` blocks update ``tools``. Consecutive same-role messages may
  occur (dialects merge if their wire format requires it). Message order is preserved.
* every ``Block.type`` is one of ``BLOCK_TYPES``; ``search_result`` became text; unknown block
  types were dropped (and noted in ``dropped``).
* ``tool_use``: ``id`` and ``name`` non-empty strings, ``input`` always a dict.
* ``tool_result``: ``tool_use_id`` string, ``content`` always a list of text/image/document blocks
  (string content -> one text block; may be empty), ``is_error`` bool.
* ``thinking``: ``thinking`` text (may be ""), ``signature`` string or None (unvalidated);
  ``redacted_thinking``: ``data``.
* ``image``: base64 (``data`` + ``media_type``) or ``url``. ``document``: base64 PDF (``data`` +
  ``media_type``), plain text (``text`` + ``media_type="text/plain"``) or ``url``; optional ``title``.
* ``tools``: custom tools only (entries whose ``type`` is absent or ``"custom"``); server tools
  (``web_search_*`` …) dropped + noted. Names unique (first wins). ``defer_loading`` ignored.
* ``tool_choice``: ``None`` or a dict ``{"type": "auto"|"any"|"tool"|"none"[, "name": str]}``.
* ``effort``: ``None`` or one of ``EFFORTS``; from ``output_config.effort``, else derived from
  ``thinking`` (adaptive -> "medium"; enabled.budget_tokens <=4096 "low", <=16384 "medium", else
  "high"; disabled -> "none").
* ``output_format``: ``None`` or the ``output_config.format`` dict, e.g.
  ``{"type": "json_schema", "schema": {...}}`` (Claude Code's background title request uses it).
* ``max_tokens`` int (> 0; Claude Code sends 32000 for unknown models; probes send 1).
* ``session_id``: ``X-Claude-Code-Session-Id`` header, else ``metadata.user_id`` JSON's
  ``session_id``, else None.
* ``raw``: the original request dict (never mutated); ``headers``: request headers as a plain dict
  with lower-cased keys (no Authorization / x-api-key).
"""

import copy
import dataclasses
from typing import Any, Dict, List, Optional

__all__ = [
    "BLOCK_TYPES", "EFFORTS", "Block", "Message", "ToolDef", "NormalizedRequest",
    "FoldResult", "fold_system_messages_raw", "SYSTEM_REMINDER_OPEN", "SYSTEM_REMINDER_CLOSE",
]

BLOCK_TYPES = ("text", "image", "document", "thinking", "redacted_thinking", "tool_use", "tool_result")
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

SYSTEM_REMINDER_OPEN = "<system-reminder>\n"
SYSTEM_REMINDER_CLOSE = "\n</system-reminder>"


@dataclasses.dataclass
class Block:
    """One content block. Only the fields relevant to ``type`` are set (see module invariants)."""

    type: str
    text: Optional[str] = None            # text / text-document body
    data: Optional[str] = None            # base64 payload (image/document) or redacted_thinking data
    media_type: Optional[str] = None      # image/png, application/pdf, text/plain ...
    url: Optional[str] = None             # url image/document source
    id: Optional[str] = None              # tool_use id
    name: Optional[str] = None            # tool_use name
    input: Optional[Dict[str, Any]] = None  # tool_use input (dict)
    tool_use_id: Optional[str] = None     # tool_result -> tool_use id
    content: Optional[List["Block"]] = None  # tool_result content (text|image|document blocks)
    is_error: bool = False                # tool_result
    signature: Optional[str] = None       # thinking signature (opaque, may be "fgw1.<target>.<b64>")
    thinking: Optional[str] = None        # thinking text
    title: Optional[str] = None           # document title

    # ---- constructors -------------------------------------------------------------------
    @classmethod
    def of_text(cls, text):
        return cls(type="text", text=text)

    @classmethod
    def of_image_base64(cls, media_type, data):
        return cls(type="image", media_type=media_type, data=data)

    @classmethod
    def of_image_url(cls, url):
        return cls(type="image", url=url)

    @classmethod
    def of_document_base64(cls, media_type, data, title=None):
        return cls(type="document", media_type=media_type, data=data, title=title)

    @classmethod
    def of_document_text(cls, text, title=None):
        return cls(type="document", media_type="text/plain", text=text, title=title)

    @classmethod
    def of_document_url(cls, url, title=None):
        return cls(type="document", url=url, title=title)

    @classmethod
    def of_thinking(cls, thinking, signature=None):
        return cls(type="thinking", thinking=thinking, signature=signature)

    @classmethod
    def of_redacted_thinking(cls, data):
        return cls(type="redacted_thinking", data=data)

    @classmethod
    def of_tool_use(cls, id, name, input=None):  # noqa: A002
        return cls(type="tool_use", id=id, name=name, input=dict(input or {}))

    @classmethod
    def of_tool_result(cls, tool_use_id, content=None, is_error=False):
        if isinstance(content, str):
            content = [cls.of_text(content)]
        return cls(type="tool_result", tool_use_id=tool_use_id, content=list(content or []),
                   is_error=bool(is_error))

    # ---- helpers ------------------------------------------------------------------------
    def is_media(self):
        return self.type in ("image", "document")

    def result_text(self, joiner="\n"):
        """tool_result: concatenated text of its text (and text-document) blocks."""
        parts = []
        for b in self.content or []:
            if b.type == "text" and b.text:
                parts.append(b.text)
            elif b.type == "document" and b.text:
                parts.append(b.text)
        return joiner.join(parts)

    def result_media(self):
        """tool_result: its image/document blocks that carry binary/url payloads."""
        return [b for b in (self.content or []) if b.type == "image" or (b.type == "document" and not b.text)]


@dataclasses.dataclass
class Message:
    role: str                      # "user" | "assistant"
    blocks: List[Block] = dataclasses.field(default_factory=list)

    def text(self, joiner=""):
        return joiner.join(b.text for b in self.blocks if b.type == "text" and b.text)

    def tool_uses(self):
        return [b for b in self.blocks if b.type == "tool_use"]

    def tool_results(self):
        return [b for b in self.blocks if b.type == "tool_result"]


@dataclasses.dataclass
class ToolDef:
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = dataclasses.field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclasses.dataclass
class NormalizedRequest:
    model: str
    system: List[str] = dataclasses.field(default_factory=list)
    messages: List[Message] = dataclasses.field(default_factory=list)
    tools: List[ToolDef] = dataclasses.field(default_factory=list)
    tool_choice: Optional[Dict[str, Any]] = None
    disable_parallel_tool_use: bool = False
    max_tokens: int = 32000
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: List[str] = dataclasses.field(default_factory=list)
    stream: bool = True
    effort: Optional[str] = None
    thinking_requested: bool = False
    thinking_budget: Optional[int] = None
    output_format: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    raw: Dict[str, Any] = dataclasses.field(default_factory=dict)
    headers: Dict[str, str] = dataclasses.field(default_factory=dict)
    dropped: List[str] = dataclasses.field(default_factory=list)

    def tool_names(self):
        """Offered tool names in order."""
        return [t.name for t in self.tools]

    def all_tool_names(self):
        """Offered tool names + names used by tool_use blocks in history (for ToolNameMap)."""
        seen = []
        for t in self.tools:
            if t.name not in seen:
                seen.append(t.name)
        for m in self.messages:
            for b in m.blocks:
                if b.type == "tool_use" and b.name and b.name not in seen:
                    seen.append(b.name)
        return seen

    def tool_use_names_by_id(self):
        """tool_use id -> tool name, from assistant history (Gemini functionResponse needs it)."""
        out = {}
        for m in self.messages:
            for b in m.blocks:
                if b.type == "tool_use" and b.id:
                    out[b.id] = b.name
        return out

    def is_probe(self):
        """``/model`` switch probes send ``max_tokens: 1``."""
        return self.max_tokens is not None and self.max_tokens <= 1


# ---------------------------------------------------------------------------------------
# role:"system" messages inside `messages` (Claude Code 2.1.288)
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass
class FoldResult:
    messages: List[Dict[str, Any]]                  # raw messages, only user/assistant roles
    tool_additions: List[Dict[str, Any]]            # tool definitions ({name, description, input_schema})
    tool_reference_additions: List[str]             # tool_addition by reference (name only)
    tool_removals: List[str]                        # names removed via tool_removal
    folded: int = 0                                 # number of system messages folded


def _system_texts(content):
    """Text pieces + tool changes of a raw role:system message content."""
    texts, adds, ref_adds, removals = [], [], [], []
    if isinstance(content, str):
        if content.strip():
            texts.append(content)
        return texts, adds, ref_adds, removals
    for blk in content or []:
        if not isinstance(blk, dict):
            continue
        t = blk.get("type")
        if t == "text":
            if (blk.get("text") or "").strip():
                texts.append(blk["text"])
        elif t == "tool_addition":
            tool = blk.get("tool") or {}
            if tool.get("type") == "tool_definition" and isinstance(tool.get("definition"), dict):
                adds.append(tool["definition"])
            elif tool.get("type") == "tool_reference" and tool.get("name"):
                ref_adds.append(tool["name"])
        elif t == "tool_removal":
            tool = blk.get("tool") or {}
            if tool.get("name"):
                removals.append(tool["name"])
    return texts, adds, ref_adds, removals


def _as_block_list(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return list(content or [])


def fold_system_messages_raw(messages, wrap=True):
    """Fold Claude Code's mid-conversation ``role:"system"`` messages into user messages.

    Works on RAW Anthropic message dicts (never mutates the input) so both ``anthropic_in`` and
    ``anthropic_passthrough`` share one policy:

    * each system message's text (string or text blocks) becomes text block(s)
      ``"<system-reminder>\\n" + text + "\\n</system-reminder>"`` (``wrap=False``: bare text);
    * placement: appended to the immediately preceding message if it is a user message; otherwise
      prepended to the next user message; if an assistant message comes first (or the list ends),
      a standalone user message is inserted at that position;
    * ``tool_addition`` (definition or reference) / ``tool_removal`` blocks are returned in the
      ``FoldResult`` for the caller to apply to ``tools``; ``clear_at`` and other keys are ignored.
    """
    out = []           # List[Dict[str, Any]]
    pending = []       # List[Dict[str, Any]]
    adds, ref_adds, removals = [], [], []
    folded = 0

    def make_blocks(texts):
        blocks = []
        for t in texts:
            body = (SYSTEM_REMINDER_OPEN + t.strip("\n") + SYSTEM_REMINDER_CLOSE) if wrap else t
            blocks.append({"type": "text", "text": body})
        return blocks

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            folded += 1
            texts, a, r, rm = _system_texts(msg.get("content"))
            adds.extend(a)
            ref_adds.extend(r)
            removals.extend(rm)
            blocks = make_blocks(texts)
            if not blocks:
                continue
            if out and out[-1].get("role") == "user" and not pending:
                prev = dict(out[-1])
                prev["content"] = _as_block_list(prev.get("content")) + blocks
                out[-1] = prev
            else:
                pending.extend(blocks)
            continue
        if pending:
            if role == "user":
                msg = dict(msg)
                msg["content"] = pending + _as_block_list(msg.get("content"))
            else:
                out.append({"role": "user", "content": pending})
            pending = []
        out.append(msg)
    if pending:
        out.append({"role": "user", "content": pending})
    return FoldResult(messages=out, tool_additions=adds, tool_reference_additions=ref_adds,
                      tool_removals=removals, folded=folded)


def apply_tool_changes(tools, fold):
    """Return a new raw ``tools`` list with ``fold``'s additions/removals applied.

    Additions by definition are appended if the name is new; removals drop the tool from the
    offered list (history mapping should still use ``NormalizedRequest.all_tool_names``).
    Additions by reference re-enable a previously removed tool only if its definition is known.
    """
    tools = [copy.copy(t) for t in (tools or []) if isinstance(t, dict)]
    by_name = {}
    for t in tools:
        by_name.setdefault(t.get("name"), t)
    removed = set()
    for name in fold.tool_removals:
        removed.add(name)
    for d in fold.tool_additions:
        name = d.get("name")
        if not name:
            continue
        removed.discard(name)
        if name not in by_name:
            tools.append(dict(d))
            by_name[name] = tools[-1]
    for name in fold.tool_reference_additions:
        removed.discard(name)
    return [t for t in tools if t.get("name") not in removed]


def split_effort_from_thinking(thinking):
    """Map an Anthropic ``thinking`` param to (effort, requested, budget). Pure helper for
    ``anthropic_in``: adaptive -> medium; enabled budget <=4096 low, <=16384 medium, else high;
    disabled -> none."""
    if not isinstance(thinking, dict):
        return None, False, None
    t = thinking.get("type")
    if t == "adaptive":
        return "medium", True, None
    if t == "enabled":
        budget = thinking.get("budget_tokens")
        try:
            budget = int(budget)
        except (TypeError, ValueError):
            budget = None
        if budget is None:
            return "medium", True, None
        if budget <= 4096:
            return "low", True, budget
        if budget <= 16384:
            return "medium", True, budget
        return "high", True, budget
    if t == "disabled":
        return "none", False, None
    return None, False, None


__all__ += ["apply_tool_changes", "split_effort_from_thinking"]
