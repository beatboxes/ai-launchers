"""Tool-name and tool-id mapping between Claude Code and upstream providers (DESIGN §3.0).

``ToolNameMap(names, regex=None, maxlen=64, leading_letter=False)``
    Built per request from the offered tools + the names used in history
    (``NormalizedRequest.all_tool_names()``). Names that already satisfy ``regex`` (default
    ``^[a-zA-Z0-9_-]{1,64}$``) and ``maxlen`` are kept verbatim. Others get every character the
    regex does not allow replaced by ``_``; if that changed anything, or the name is longer than
    ``maxlen``, the result is ``name[:maxlen-9] + "_" + sha1(name)[:8]``. ``leading_letter``
    (Gemini) prefixes ``_`` when the name would start with something other than a letter or ``_``.
    Collisions (with kept names or other shortened names) extend the hash.
    ``upstream(name)`` -> provider-safe name; ``original(name)`` -> Claude Code name (unknown names
    are returned unchanged).

Tool ids:
    ``encode_tool_id(id)``  upstream id -> Anthropic tool_use id: ids matching
                            ``^[A-Za-z0-9_-]{1,128}$`` pass through; others become
                            ``toolu_x`` + b64url(id); missing/empty ids get ``new_tool_id()``.
    ``decode_tool_id(id)``  inverse of ``encode_tool_id`` (anything else is returned unchanged).
    ``new_tool_id()``       ``toolu_<24 hex>``.
"""

import hashlib
import re
import uuid

from .compat import b64url_decode, b64url_encode

__all__ = ["ToolNameMap", "DEFAULT_TOOL_NAME_REGEX", "encode_tool_id", "decode_tool_id", "new_tool_id",
           "TOOL_ID_RE", "ENCODED_ID_PREFIX"]

DEFAULT_TOOL_NAME_REGEX = r"^[a-zA-Z0-9_-]{1,64}$"
TOOL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
ENCODED_ID_PREFIX = "toolu_x"
_LEADING_OK = re.compile(r"^[A-Za-z_]")
_HASH_LEN = 8


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class ToolNameMap(object):
    """Bidirectional, deterministic mapping of tool names for one request."""

    def __init__(self, names, regex=None, maxlen=64, leading_letter=False):
        self.regex = re.compile(regex or DEFAULT_TOOL_NAME_REGEX)
        self.maxlen = max(16, int(maxlen or 64))
        self.leading_letter = bool(leading_letter)
        self._char_ok = {}
        self._up = {}    # original -> upstream
        self._down = {}  # upstream -> original
        self._order = []
        for n in names or []:
            if isinstance(n, str) and n and n not in self._order:
                self._order.append(n)
        # Valid names are reserved first so a shortened name can never shadow a real one.
        pending = []
        for n in self._order:
            if self._valid(n):
                self._up[n] = n
                self._down[n] = n
            else:
                pending.append(n)
        for n in pending:
            short = self._shorten(n)
            self._up[n] = short
            self._down[short] = n

    # ---- public API ------------------------------------------------------------------------
    def upstream(self, name):
        """Provider-safe name for ``name`` (names not seen at construction are mapped on the fly)."""
        if name in self._up:
            return self._up[name]
        if not isinstance(name, str) or not name:
            return name
        short = name if self._valid(name) else self._shorten(name)
        if short in self._down and self._down[short] != name:
            short = self._shorten(name)
        self._up[name] = short
        self._down[short] = name
        self._order.append(name)
        return short

    def original(self, name):
        """Claude Code name for an upstream name; unknown names are returned unchanged."""
        return self._down.get(name, name)

    def is_renamed(self, name):
        return self._up.get(name, name) != name

    def items(self):
        """(original, upstream) pairs in first-seen order."""
        return [(n, self._up[n]) for n in self._order]

    def upstream_names(self):
        return list(self._down)

    def __len__(self):
        return len(self._up)

    def __repr__(self):
        renamed = sum(1 for k, v in self._up.items() if k != v)
        return "ToolNameMap(%d names, %d renamed)" % (len(self._up), renamed)

    # ---- internals -------------------------------------------------------------------------
    def _valid(self, name):
        if len(name) > self.maxlen or not self.regex.match(name):
            return False
        return not self.leading_letter or bool(_LEADING_OK.match(name))

    def _allowed(self, ch):
        ok = self._char_ok.get(ch)
        if ok is None:
            ok = bool(self.regex.match("a" + ch)) and ch not in "\r\n"
            self._char_ok[ch] = ok
        return ok

    def _shorten(self, name):
        cleaned = "".join(ch if self._allowed(ch) else "_" for ch in name)
        if self.leading_letter and not _LEADING_OK.match(cleaned):
            cleaned = "_" + cleaned
        digest = _sha1(name)
        hash_len = _HASH_LEN
        while True:
            if hash_len <= len(digest):
                suffix = digest[:hash_len]
            else:  # practically unreachable: 40 hex chars exhausted -> counter
                suffix = digest + str(hash_len - len(digest))
            keep = max(1, self.maxlen - len(suffix) - 1)
            candidate = cleaned[:keep] + "_" + suffix
            if candidate not in self._down and candidate not in self._up:
                return candidate
            hash_len += 1


# ---------------------------------------------------------------------------------------
# tool ids
# ---------------------------------------------------------------------------------------

def new_tool_id():
    """A fresh Anthropic-style tool_use id: ``toolu_<24 hex>``."""
    return "toolu_" + uuid.uuid4().hex[:24]


def encode_tool_id(tool_id):
    """Upstream tool-call id -> Anthropic tool_use id (see module docstring)."""
    if tool_id is None or tool_id == "":
        return new_tool_id()
    if not isinstance(tool_id, str):
        tool_id = str(tool_id)
    if TOOL_ID_RE.match(tool_id):
        return tool_id
    return ENCODED_ID_PREFIX + b64url_encode(tool_id)


def decode_tool_id(tool_id):
    """Anthropic tool_use id -> the upstream id it was encoded from (unchanged if not encoded).

    Only ids whose decoded form could NOT have passed through verbatim are treated as encoded,
    so a genuine upstream id that happens to start with ``toolu_x`` round-trips unchanged.
    """
    if not isinstance(tool_id, str) or not tool_id.startswith(ENCODED_ID_PREFIX):
        return tool_id
    try:
        original = b64url_decode(tool_id[len(ENCODED_ID_PREFIX):]).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return tool_id
    if not original or TOOL_ID_RE.match(original) or b64url_encode(original) != tool_id[len(ENCODED_ID_PREFIX):]:
        return tool_id
    return original
