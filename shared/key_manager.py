"""API-key storage and secret-source helpers shared by every launcher.

``credentials.json`` (``<ail_home>/credentials.json``) keeps the v0.1 format for backward
compatibility — ``{"<launcher name>": {"key": "<secret>"}}`` — and is always written atomically with
mode 0600 (owner-only ACL on Windows) via ``gateway.atomicio``.

A key transport's secret is resolved from an ordered source list (``gateway.secrets`` syntax):
``env:<VAR>`` for every manifest env var → the configured ``op://`` reference →
``json:<credentials.json>#<launcher>.key`` → transport-specific extras (e.g. the ``OPENAI_API_KEY`` a
``codex login --with-api-key`` stored in ``~/.codex/auth.json``).

``peek`` resolves the local sources (env, json, literal) without threads, subprocesses or network
(dry-run, ``keys list``); ``op://`` references are only resolved by ``gateway.secrets``.
Secret values are never printed: ``redact`` gives a short display hint.
"""

import os
import re

from .utils import credentials_path, home_dir, read_json, write_json

__all__ = [
    "creds_path", "load_creds", "stored_key", "set_key", "remove_key", "check_key", "secret_sources",
    "peek", "redact", "display_source", "describe_sources", "codex_auth_path",
]

_MAX_KEY_LEN = 8192


def creds_path():
    return credentials_path()


def load_creds():
    return read_json(creds_path())


def stored_key(launcher):
    entry = load_creds().get(launcher)
    if isinstance(entry, dict) and isinstance(entry.get("key"), str) and entry["key"].strip():
        return entry["key"].strip()
    return None


def check_key(value):
    """Problem with a candidate key (``None`` when acceptable)."""
    if not value:
        return "empty key"
    if len(value) > _MAX_KEY_LEN:
        return "key is implausibly long (%d characters)" % len(value)
    if re.search(r"[\s\x00-\x1f\x7f]", value):
        return "key contains whitespace or control characters"
    return None


def set_key(launcher, value):
    """Store ``value`` for ``launcher`` (0600, atomic); returns the credentials path."""
    value = (value or "").strip()
    problem = check_key(value)
    if problem:
        raise ValueError(problem)
    creds = load_creds()
    creds[launcher] = {"key": value}
    return write_json(creds_path(), creds, private=True)


def remove_key(launcher):
    creds = load_creds()
    if launcher not in creds:
        return False
    del creds[launcher]
    write_json(creds_path(), creds, private=True)
    return True


def codex_auth_path(environ=None):
    """``$CODEX_HOME/auth.json`` or ``~/.codex/auth.json``."""
    environ = os.environ if environ is None else environ
    base = environ.get("CODEX_HOME") or str(home_dir() / ".codex")
    return os.path.join(base, "auth.json")


def secret_sources(launcher, env_vars, op_ref=None, extra=None):
    """Ordered ``gateway.secrets`` source list for a key transport of ``launcher``."""
    out = ["env:%s" % v for v in env_vars or [] if v]
    if op_ref:
        out.append(op_ref)
    out.append("json:%s#%s.key" % (creds_path(), launcher))
    out.extend(extra or [])
    return out


def _json_lookup(spec):
    path, _, dotted = spec[len("json:"):].rpartition("#")
    if not path or not dotted:
        return None
    node = read_json(path)
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node.strip() if isinstance(node, str) and node.strip() else None


def peek(sources, environ=None):
    """``(value, label)`` from the first local source with a value (env/json/literal), else ``(None, None)``.

    ``op://`` references are skipped (no subprocess/network); callers report them separately. Labels
    match ``gateway.secrets.source_label`` (``env:VAR``, ``json:credentials.json``, …).
    """
    from .gateway.secrets import source_label

    environ = os.environ if environ is None else environ
    for src in sources or []:
        value = None
        if src.startswith("env:"):
            value = (environ.get(src[4:]) or "").strip() or None
        elif src.startswith("json:"):
            value = _json_lookup(src)
        elif src.startswith("literal:"):
            value = src[len("literal:"):].strip() or None
        if value:
            return value, source_label(src)
    return None, None


def redact(value):
    """Display hint for a secret: a short prefix/suffix only for long keys, never the value."""
    if not value:
        return "(none)"
    if len(value) < 20:
        return "****** (%d chars)" % len(value)
    return "%s…%s (%d chars)" % (value[:4], value[-2:], len(value))


def display_source(label):
    """Human label for a source label (``json:credentials.json`` -> ``credentials.json``)."""
    if not label:
        return "unset"
    return label[len("json:"):] if label == "json:%s" % creds_path().name else label


def describe_sources(sources):
    """Comma-separated, secret-free description of a source list (for help/instructions)."""
    from .gateway.secrets import source_label

    return ", ".join(s if s.startswith("op://") else display_source(source_label(s)) for s in sources)
