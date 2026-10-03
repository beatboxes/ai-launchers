"""Logging, redaction and the optional JSONL request trace.

* The ``"ai_gateway"`` logger never writes to stdout/stderr (Claude Code owns the terminal): it
  has a ``NullHandler`` and does not propagate unless ``setup_logging(log_file=...)`` adds a
  rotating file handler.
* ``redact(text, secrets)`` removes every SecretStore value plus generic credential shapes
  (``sk-…``, ``xai-…``, ``nvapi-…``, ``Bearer …``, JWTs ``eyJ…``, Google ``AIza…`` / ``ya29.…``,
  ``"api_key": "…"``-style pairs).
* ``Tracer(path, secrets)`` appends one redacted JSON object per call: ``tracer(name, fields)``.
  Request/response bodies are only traced when ``AI_GATEWAY_TRACE_BODIES=1`` (``Tracer.bodies``),
  and then redacted too. ``AI_GATEWAY_TRACE_FILE`` names the default trace file.
"""

import json
import logging
import logging.handlers
import os
import re
import threading

from .compat import rfc3339_format

__all__ = ["LOGGER_NAME", "setup_logging", "redact", "redact_obj", "RedactingFilter", "Tracer", "trace_bodies_enabled",
           "trace_file_from_env", "TRACE_FILE_ENV", "TRACE_BODIES_ENV"]

LOGGER_NAME = "ai_gateway"
TRACE_FILE_ENV = "AI_GATEWAY_TRACE_FILE"
TRACE_BODIES_ENV = "AI_GATEWAY_TRACE_BODIES"
REPLACEMENT = "***"

_HANDLER_MARK = "_ai_gateway_handler"
_KEYS = (r"api[_-]?key|x-api-key|x-goog-api-key|authorization|proxy-authorization|access[_-]?token|"
         r"refresh[_-]?token|id[_-]?token|auth[_-]?token|client[_-]?secret|password|cookie")
_PATTERNS = (
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=\-]{6,}"), r"\1 " + REPLACEMENT),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}(?:\.[A-Za-z0-9_\-]*)?"), "eyJ" + REPLACEMENT),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"), "sk-" + REPLACEMENT),
    (re.compile(r"\bxai-[A-Za-z0-9_\-]{6,}"), "xai-" + REPLACEMENT),
    (re.compile(r"\bnvapi-[A-Za-z0-9_\-]{6,}"), "nvapi-" + REPLACEMENT),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"), "AIza" + REPLACEMENT),
    (re.compile(r"\bya29\.[0-9A-Za-z_\-.]{10,}"), "ya29." + REPLACEMENT),
    (re.compile(r"(?i)([\"']?(?<![A-Za-z0-9])(?:%s)[\"']?\s*[:=]\s*[\"']?)(?!\*\*\*)([^\"'\s,;}&]{4,})" % _KEYS),
     r"\1" + REPLACEMENT),
    (re.compile(r"(?i)([?&](?:key|token|access_token)=)([^&\s\"']+)"), r"\1" + REPLACEMENT),
)

_root = logging.getLogger(LOGGER_NAME)
_root.addHandler(logging.NullHandler())
_root.propagate = False


def redact(text, secrets=None, replacement=REPLACEMENT):
    """``text`` with secret values (``SecretStore`` or an iterable of strings) and generic
    credential patterns replaced."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if secrets is not None:
        if hasattr(secrets, "values_for_redaction"):
            values = secrets.values_for_redaction()
        else:
            values = [v for v in secrets if isinstance(v, str)]
        for v in sorted(values, key=len, reverse=True):
            if len(v) >= 4 and v in text:
                text = text.replace(v, replacement)
    for rx, repl in _PATTERNS:
        text = rx.sub(repl, text)
    return text


def redact_obj(obj, secrets=None, _depth=0):
    """Recursively redact every string inside dicts/lists (keys kept)."""
    if _depth > 20:
        return obj
    if isinstance(obj, str):
        return redact(obj, secrets)
    if isinstance(obj, dict):
        return {k: redact_obj(v, secrets, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v, secrets, _depth + 1) for v in obj]
    return obj


class RedactingFilter(logging.Filter):
    """Rewrites each record's message (and formatted traceback) through ``redact``."""

    def __init__(self, secrets=None):
        logging.Filter.__init__(self)
        self.secrets = secrets

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        record.msg = redact(msg, self.secrets)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text, self.secrets)
        return True


def _private_file(path):
    """Create ``path`` (and its directory) with owner-only permissions if it does not exist."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(path):
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
        except OSError:
            pass


class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating handler whose (re)created files are owner-only (0600 on POSIX)."""

    def _open(self):
        _private_file(self.baseFilename)
        return logging.handlers.RotatingFileHandler._open(self)


def _level(level):
    if isinstance(level, int):
        return level
    value = logging.getLevelName(str(level or "INFO").upper())
    return value if isinstance(value, int) else logging.INFO


def setup_logging(level="INFO", log_file=None, secrets=None, max_bytes=5 * 1024 * 1024, backup_count=3):
    """Configure the ``ai_gateway`` logger: rotating UTF-8 file (redacted) or a NullHandler.

    Never logs to stdout/stderr and never propagates to the root logger. Re-calling replaces the
    handler installed by the previous call.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for h in list(logger.handlers):
        if getattr(h, _HANDLER_MARK, False):
            logger.removeHandler(h)
            h.close()
    if log_file:
        handler = _PrivateRotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count,
                                              encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s"))
    else:
        handler = logging.NullHandler()
    handler.addFilter(RedactingFilter(secrets))
    setattr(handler, _HANDLER_MARK, True)
    logger.addHandler(handler)
    logger.setLevel(_level(level))
    logger.propagate = False
    return logger


def trace_bodies_enabled(environ=None):
    env = os.environ if environ is None else environ
    return str(env.get(TRACE_BODIES_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def trace_file_from_env(environ=None):
    env = os.environ if environ is None else environ
    return env.get(TRACE_FILE_ENV) or None


class Tracer(object):
    """Append-only JSONL trace. ``tracer(name, fields)`` never raises."""

    def __init__(self, path, secrets=None, bodies=None):
        self.path = path
        self.secrets = secrets
        self.bodies = trace_bodies_enabled() if bodies is None else bool(bodies)
        self._lock = threading.Lock()
        _private_file(path)

    def __call__(self, name, fields=None):
        rec = {"ts": rfc3339_format(timespec="milli"), "event": name}
        rec.update(redact_obj(dict(fields or {}), self.secrets))
        try:
            line = json.dumps(rec, ensure_ascii=False, default=str, sort_keys=False)
        except (TypeError, ValueError):
            line = json.dumps({"ts": rec["ts"], "event": name, "error": "unserializable trace fields"})
        line = redact(line, self.secrets)
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError:
            pass

    def __repr__(self):
        return "Tracer(%r, bodies=%r)" % (self.path, self.bodies)
