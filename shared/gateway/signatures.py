"""Thinking-block signatures (DESIGN §3.0). Owned by Phase 0; used by anthropic_out (synthetic),
responses (reasoning items), gemini (thoughtSignature) and every non-passthrough dialect (filtering).

Format: ``fgw1.<target>.<b64url(compact json)>`` with target in ``TARGETS``. The synthetic signature
for chat-style reasoning (no upstream blob) is ``fgw1.chat.<sha1-12>`` (12 hex chars, not JSON).

``encode_signature(target, payload) -> str``
``decode_signature(sig) -> Optional[Tuple[str, dict]]``  (synthetic chat sigs decode to ("chat", {}))
``signature_target(sig) -> Optional[str]``                (None for non-fgw1 signatures)
``synthetic_signature(text) -> str``
``keep_thinking_for(sig, target) -> bool``                True only for fgw1 sigs of exactly ``target``
"""

import hashlib
import json
import re

from .compat import b64url_decode, b64url_encode

__all__ = ["PREFIX", "TARGETS", "encode_signature", "decode_signature", "signature_target", "synthetic_signature",
           "keep_thinking_for"]

PREFIX = "fgw1"
TARGETS = ("codex_chatgpt", "openai_api", "grok_cli", "xai_api", "gemini_api", "vertex", "chat")
_SYNTH_RE = re.compile(r"^fgw1\.chat\.[0-9a-f]{12}$")


def encode_signature(target, payload):
    if target not in TARGETS:
        raise ValueError("unknown signature target %r" % (target,))
    blob = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return "%s.%s.%s" % (PREFIX, target, b64url_encode(blob))


def signature_target(sig):
    if not isinstance(sig, str) or not sig.startswith(PREFIX + "."):
        return None
    parts = sig.split(".", 2)
    if len(parts) != 3 or parts[1] not in TARGETS:
        return None
    return parts[1]


def decode_signature(sig):
    target = signature_target(sig)
    if target is None:
        return None
    if _SYNTH_RE.match(sig):
        return "chat", {}
    try:
        payload = json.loads(b64url_decode(sig.split(".", 2)[2]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return target, payload


def synthetic_signature(text):
    if isinstance(text, str):
        text = text.encode("utf-8")
    return "%s.chat.%s" % (PREFIX, hashlib.sha1(text or b"").hexdigest()[:12])


def keep_thinking_for(sig, target):
    """Non-passthrough dialects keep a history thinking block only if its signature was produced for
    the same target (``fgw1.<target>.*``); everything else (Anthropic/foreign/mismatched) is dropped."""
    return signature_target(sig) == target
