"""Gateway errors and upstream-error mapping (DESIGN §2.4). Owned by Phase 0; ``anthropic_out``
re-exports these names and must not redefine them. Dialects import this module directly.

``GatewayError(status, err_type, message, should_retry=False, retry_after=None, ...)``
    ``.body()``    -> ``{"type": "error", "error": {"type": err_type, "message": message}}``
    ``.headers()`` -> ``{"x-should-retry": "true"|"false"[, "retry-after": "<int s>",
                        "retry-after-ms": "<int ms>"]}``
    ``.to_stream_error()`` -> ``events.StreamError`` for failures after the SSE commit point
    (retryable -> ``overloaded_error`` so Claude Code retries; else the mapped type).

``map_upstream_error(status, body_text, headers, provider, model, context_window=None,
est_tokens=None, auth_hint=None) -> GatewayError``; ``status=None`` means a connection error.

``map_error_code(code, message, provider, model, ...)`` maps in-stream error codes (Responses
``response.failed``, Gemini stream errors …) through the same table.

``prompt_too_long(n, m)`` builds the exact context-overflow error Claude Code parses:
``prompt is too long: N tokens > M maximum``.
"""

import json
import math
import re
import time
from email.utils import parsedate_to_datetime
from .compat import rfc3339_format, rfc3339_to_epoch

__all__ = [
    "GatewayError", "map_upstream_error", "map_error_code", "prompt_too_long",
    "TERMINAL_RETRY_DELAY", "ERROR_TYPES", "parse_retry_after",
]

ERROR_TYPES = ("invalid_request_error", "authentication_error", "permission_error", "not_found_error",
               "request_too_large", "rate_limit_error", "api_error", "overloaded_error")

#: Google ``retryDelay`` above this many seconds is treated as a terminal quota.
TERMINAL_RETRY_DELAY = 300.0

_BODY_KEEP = 65536


class GatewayError(Exception):
    """An error with a definite Anthropic-shaped HTTP response."""

    def __init__(self, status, err_type, message, should_retry=False, retry_after=None,
                 upstream_status=None, upstream_body=None, upstream_headers=None, connection_error=False):
        Exception.__init__(self, message)
        self.status = int(status)
        self.err_type = err_type
        self.message = message
        self.should_retry = bool(should_retry)
        self.retry_after = None if retry_after is None else max(0.0, float(retry_after))
        self.upstream_status = upstream_status
        self.upstream_body = upstream_body
        self.upstream_headers = dict(upstream_headers or {})
        self.connection_error = bool(connection_error)

    def body(self):
        return {"type": "error", "error": {"type": self.err_type, "message": self.message}}

    def headers(self):
        h = {"x-should-retry": "true" if self.should_retry else "false"}
        if self.retry_after is not None:
            h["retry-after"] = str(max(1, int(math.ceil(self.retry_after))))
            h["retry-after-ms"] = str(int(round(self.retry_after * 1000)))
        return h

    def to_stream_error(self):
        from .events import StreamError

        err_type = "overloaded_error" if self.should_retry else self.err_type
        return StreamError(err_type, self.message, self.should_retry)

    def __str__(self):
        return self.message

    def __repr__(self):
        return "GatewayError(%d, %r, %r, should_retry=%r, retry_after=%r)" % (
            self.status, self.err_type, self.message, self.should_retry, self.retry_after)


def prompt_too_long(n, m):
    """The exact context-overflow error Claude Code recognises (and compacts on)."""
    n, m = int(n), int(m)
    if n <= m:
        n = m + 1
    return GatewayError(400, "invalid_request_error", "prompt is too long: %d tokens > %d maximum" % (n, m), False)


# ---------------------------------------------------------------------------------------
# body parsing
# ---------------------------------------------------------------------------------------

def _hget(headers, name):
    if not headers:
        return None
    try:
        v = headers.get(name)
        if v is not None:
            return v
    except Exception:
        pass
    lname = name.lower()
    try:
        for k, v in headers.items():
            if str(k).lower() == lname:
                return v
    except Exception:
        pass
    return None


def _parse_duration(s):
    """Google duration ("30s", "1.5s", "250ms") or bare number -> seconds."""
    if s is None:
        return None
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    if isinstance(s, dict):  # {"seconds": "30", "nanos": 0}
        try:
            return float(s.get("seconds") or 0) + float(s.get("nanos") or 0) / 1e9
        except (TypeError, ValueError):
            return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h)?\s*$", str(s))
    if not m:
        return None
    v = float(m.group(1))
    unit = m.group(2) or "s"
    return v / 1000.0 if unit == "ms" else v * {"s": 1, "m": 60, "h": 3600}[unit]


def parse_retry_after(headers, now=None):
    """``retry-after-ms`` / ``retry-after`` (seconds or HTTP date) -> seconds, or None."""
    ms = _hget(headers, "retry-after-ms")
    if ms is not None:
        try:
            return max(0.0, float(ms) / 1000.0)
        except (TypeError, ValueError):
            pass
    ra = _hget(headers, "retry-after")
    if ra is None:
        return None
    try:
        return max(0.0, float(ra))
    except (TypeError, ValueError):
        pass
    try:
        dt = parsedate_to_datetime(str(ra))
        return max(0.0, dt.timestamp() - (time.time() if now is None else now))
    except Exception:
        return None


class _Parsed(object):
    __slots__ = ("message", "codes", "obj", "google_details", "dicts")

    def __init__(self):
        self.message = ""
        self.codes = []          # lower-cased type/code/status strings
        self.obj = None
        self.google_details = []
        self.dicts = []          # every dict visited (for resets_at lookups)


def _walk_dicts(obj, out, depth=0):
    if depth > 6:
        return
    if isinstance(obj, dict):
        out.append(obj)
        for v in obj.values():
            _walk_dicts(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:20]:
            _walk_dicts(v, out, depth + 1)


def _collapse(text, limit=1000):
    text = re.sub(r"<[^>]{1,200}>", " ", text) if "<html" in text.lower() or "<body" in text.lower() else text
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _parse_body(body_text):
    p = _Parsed()
    text = body_text or ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    obj = None
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            obj = json.loads(stripped)
        except ValueError:
            obj = None
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):  # Gemini stream error arrays
        obj = obj[0]
    p.obj = obj
    if isinstance(obj, dict):
        _walk_dicts(obj, p.dicts)
        err = obj.get("error")
        detail = obj.get("detail")
        msg = None
        if isinstance(err, dict):  # Anthropic / OpenAI / Google
            msg = err.get("message") or err.get("msg")
            for k in ("type", "code", "status", "reason"):
                if err.get(k) is not None:
                    p.codes.append(str(err.get(k)).lower())
            if isinstance(err.get("details"), list):
                p.google_details = [d for d in err["details"] if isinstance(d, dict)]
        elif isinstance(err, str):  # xAI {"code": ..., "error": "..."}
            msg = err
        if detail is not None:  # FastAPI / ChatGPT backend
            if isinstance(detail, str):
                msg = msg or detail
            elif isinstance(detail, dict):
                msg = msg or detail.get("message") or json.dumps(detail)[:500]
                for k in ("type", "code", "status"):
                    if detail.get(k) is not None:
                        p.codes.append(str(detail.get(k)).lower())
            elif isinstance(detail, list):  # FastAPI validation errors
                parts = []
                for d in detail[:5]:
                    if isinstance(d, dict):
                        parts.append(str(d.get("msg") or d))
                msg = msg or "; ".join(parts)
        for k in ("type", "code", "status"):
            v = obj.get(k)
            if isinstance(v, (str, int)) and str(v).lower() != "error":
                p.codes.append(str(v).lower())
        if not msg and isinstance(obj.get("message"), str):
            msg = obj["message"]
        for d in p.dicts:  # Gemini details may also hold reasons
            if isinstance(d.get("reason"), str):
                p.codes.append(d["reason"].lower())
        p.message = _collapse(str(msg)) if msg else _collapse(stripped)
    else:
        p.message = _collapse(stripped)
    return p


# ---------------------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------------------

_OVERFLOW_RE = re.compile(
    r"context_length_exceeded|maximum context length|prompt is too long|exceeds the context window|"
    r"input token count.*exceed|too many tokens|maximum prompt length|context window exceeded", re.I | re.S)

_OVERFLOW_NUMS = [
    (re.compile(r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)", re.I), "nm"),
    (re.compile(r"input token count\D{0,20}(\d+)\D{0,80}?maximum number of tokens allowed\D{0,20}(\d+)", re.I), "nm"),
    (re.compile(r"maximum context length is (\d+) tokens.*?(?:requested|resulted in|contains|have)\s+(\d+)", re.I | re.S), "mn"),
    (re.compile(r"maximum prompt length is (\d+).*?contains (\d+)", re.I | re.S), "mn"),
    (re.compile(r"(\d+)\s*tokens?\s*>\s*(\d+)", re.I), "nm"),
]

_CREDITS_RE = re.compile(
    r"used all available credits|run out of credits|out of credits|monthly spending limit|spending limit|"
    r"credits? (?:are |is )?(?:exhausted|depleted)|insufficient (?:balance|credits|funds)|"
    r"exceeded your current quota|billing (?:hard )?limit|purchase more credits", re.I)

_OVERLOADED_RE = re.compile(r"overloaded|server is busy|capacity", re.I)


def _overflow_numbers(text):
    for rx, order in _OVERFLOW_NUMS:
        m = rx.search(text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return (a, b) if order == "nm" else (b, a)
    return None, None


def _overflow_error(text, context_window, est_tokens):
    n, m = _overflow_numbers(text)
    if not n:
        n = est_tokens or ((context_window + 1) if context_window else 200001)
    if not m:
        m = context_window or max(1, int(n) - 1)
    return prompt_too_long(n, m)


def _google_retry_delay(parsed):
    for d in parsed.google_details:
        if str(d.get("@type", "")).endswith("RetryInfo"):
            return _parse_duration(d.get("retryDelay"))
    return None


def _google_per_day(parsed):
    for d in parsed.google_details:
        if str(d.get("@type", "")).endswith("QuotaFailure"):
            for v in d.get("violations") or []:
                if isinstance(v, dict):
                    blob = " ".join(str(v.get(k, "")) for k in ("quotaId", "quotaMetric", "description"))
                    if re.search(r"per ?day|perday", blob, re.I):
                        return True
    return False


def _reset_info(parsed):
    """(reset_epoch or None, reset_in_seconds or None) from Codex-style fields."""
    for d in parsed.dicts:
        if "resets_at" in d or "resets_in_seconds" in d:
            at, rin = d.get("resets_at"), d.get("resets_in_seconds")
            epoch = None
            try:
                if isinstance(at, (int, float)) and not isinstance(at, bool):
                    epoch = float(at)
                elif isinstance(at, str):
                    epoch = float(at) if at.replace(".", "", 1).isdigit() else rfc3339_to_epoch(at)
            except (ValueError, TypeError):
                epoch = None
            try:
                rin = float(rin) if rin is not None else None
            except (TypeError, ValueError):
                rin = None
            if epoch is None and rin is not None:
                epoch = time.time() + rin
            return epoch, rin
    return None, None


def _classify_429(parsed, headers):
    """-> (terminal: bool, retry_after: Optional[float], reset_epoch: Optional[float])."""
    codes = " ".join(parsed.codes)
    msg = parsed.message or ""
    retry_after = parse_retry_after(headers)
    g_delay = _google_retry_delay(parsed)
    if retry_after is None and g_delay is not None:
        retry_after = g_delay
    reset_epoch, reset_in = _reset_info(parsed)
    if "usage_limit_reached" in codes or "usage_limit_reached" in msg:
        return True, None, reset_epoch
    if "insufficient_quota" in codes or "insufficient_quota" in msg:
        return True, None, reset_epoch
    if "quota_exhausted" in codes or "QUOTA_EXHAUSTED" in msg or _google_per_day(parsed) or re.search(r"PerDay", msg):
        return True, None, reset_epoch
    if g_delay is not None and g_delay > TERMINAL_RETRY_DELAY:
        return True, None, time.time() + g_delay
    # Google says "You exceeded your current quota" even for per-minute limits; when the
    # structured RetryInfo/QuotaFailure details are present they decide (not the billing text).
    if g_delay is not None or any(str(d.get("@type", "")).endswith("QuotaFailure") for d in parsed.google_details):
        return False, retry_after, None
    if _CREDITS_RE.search(msg) or _CREDITS_RE.search(codes):
        return True, None, reset_epoch
    if retry_after is not None and retry_after > TERMINAL_RETRY_DELAY * 12:  # > 1 h: give up
        return True, None, time.time() + retry_after
    return False, retry_after, None


def map_upstream_error(status, body_text, headers, provider, model, context_window=None, est_tokens=None,
                       auth_hint=None):
    """Map an upstream HTTP failure to a ``GatewayError`` (DESIGN §2.4).

    ``status`` None = connection error (``body_text`` = description). ``auth_hint`` (e.g.
    "run `codex login`") is appended to authentication errors.
    """
    prefix = "[%s/%s] " % (provider or "?", model or "?")
    text = body_text if isinstance(body_text, str) else (body_text or b"").decode("utf-8", "replace")
    kept = text[:_BODY_KEEP]
    hdrs = dict(headers.items()) if hasattr(headers, "items") else {}

    def err(st, typ, msg, retry, retry_after=None):
        return GatewayError(st, typ, msg, retry, retry_after, upstream_status=status, upstream_body=kept,
                            upstream_headers=hdrs, connection_error=status is None)

    if status is None:
        return err(502, "api_error", prefix + "connection to upstream failed: " + (_collapse(text, 300) or "error"),
                   True)

    parsed = _parse_body(text)
    msg = parsed.message or ("HTTP %d" % status)
    blob = msg + " " + " ".join(parsed.codes)

    if 400 <= status < 500 and status != 429 and _OVERFLOW_RE.search(blob):
        e = _overflow_error(msg, context_window, est_tokens)
        return err(e.status, e.err_type, e.message, False)
    if status == 401:
        hint = (" — " + auth_hint) if auth_hint else " — check the API key or re-login"
        return err(401, "authentication_error", prefix + "upstream rejected the credentials: " + msg + hint, False)
    if status == 402:
        return err(429, "rate_limit_error", prefix + "billing/credits exhausted: " + msg, False)
    if status == 403:
        return err(403, "permission_error", prefix + msg, False)
    if status == 404:
        return err(404, "not_found_error", prefix + "model or endpoint not found: " + msg, False)
    if status == 413:
        return err(413, "request_too_large", prefix + msg, False)
    if status == 429:
        terminal, retry_after, reset_epoch = _classify_429(parsed, hdrs)
        if terminal:
            extra = ""
            if reset_epoch:
                extra = " (resets at %s)" % rfc3339_format(reset_epoch, "seconds")
            return err(429, "rate_limit_error", prefix + "quota exhausted: " + msg + extra, False)
        return err(429, "rate_limit_error", prefix + "rate limited: " + msg, True, retry_after)
    if status in (503, 529) or (status >= 500 and _OVERLOADED_RE.search(blob)):
        return err(529, "overloaded_error", prefix + "upstream overloaded: " + msg, True, parse_retry_after(hdrs))
    if status >= 500 or status in (408, 409):
        return err(502, "api_error", prefix + "upstream error %d: %s" % (status, msg), True, parse_retry_after(hdrs))
    # other 4xx (400, 422, 405, …)
    return err(400, "invalid_request_error", prefix + msg, False)


_CODE_STATUS = {
    "context_length_exceeded": 400,
    "invalid_prompt": 400,
    "invalid_request_error": 400,
    "invalid_argument": 400,
    "rate_limit_exceeded": 429,
    "rate_limit_error": 429,
    "resource_exhausted": 429,
    "usage_limit_reached": 429,
    "insufficient_quota": 429,
    "server_error": 503,
    "overloaded_error": 529,
    "unavailable": 503,
    "internal": 500,
    "internal_error": 500,
    "api_error": 500,
    "authentication_error": 401,
    "unauthenticated": 401,
    "permission_denied": 403,
    "not_found": 404,
    "model_not_found": 404,
}


def map_error_code(code, message, provider, model, context_window=None, est_tokens=None, extra=None):
    """Map an in-stream error code (e.g. Responses ``response.failed.error.code``) via the HTTP table.

    Unknown codes map to 502 ``api_error`` (retryable). ``extra`` (dict) is merged into the
    synthetic error object (e.g. ``resets_at``).
    """
    c = (code or "").lower()
    status = _CODE_STATUS.get(c, 500)
    if status == 500 and _OVERFLOW_RE.search(message or ""):
        status = 400
    err = {"code": c, "message": message or c or "upstream stream error"}
    if extra:
        err.update(extra)
    if c in ("usage_limit_reached", "insufficient_quota"):
        err["type"] = c
    return map_upstream_error(status, json.dumps({"error": err}), {}, provider, model, context_window, est_tokens)
