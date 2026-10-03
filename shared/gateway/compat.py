"""Python 3.8-compatible helpers shared by every module (stdlib only, no side effects).

- ``rfc3339_parse(s) -> datetime`` (aware, UTC) — accepts ``Z``/``z``, ``+HH:MM``/``+HHMM`` offsets,
  any number of fractional digits (truncated to microseconds), ``T``/``t``/space separator; a
  missing offset is treated as UTC. Raises ``ValueError`` on anything else.
- ``rfc3339_to_epoch(s) -> float``
- ``rfc3339_format(value=None, timespec="auto") -> str`` — UTC with a ``Z`` suffix; ``value`` is a
  datetime (naive = UTC), an epoch number, or None (now).
- ``b64url_encode(data) -> str`` (no padding), ``b64url_decode(s) -> bytes`` (padding/whitespace
  tolerant; also accepts the standard alphabet).
- ``jwt_claims(token) -> dict`` — payload of a JWT WITHOUT verification; ``{}`` on any failure.
- ``removeprefix``/``removesuffix`` (str methods only exist from 3.9).
- ``json_dumps_compact(obj) -> str`` — ``separators=(",", ":")``, ``ensure_ascii=False``.
- ``utcnow_epoch() -> float``, ``utcnow() -> datetime`` (aware UTC).
"""

import base64
import binascii
import calendar
import datetime as _dt
import json
import re
import time

__all__ = [
    "rfc3339_parse", "rfc3339_to_epoch", "rfc3339_format", "b64url_encode", "b64url_decode",
    "jwt_claims", "removeprefix", "removesuffix", "json_dumps_compact", "utcnow_epoch", "utcnow",
    "UTC",
]

UTC = _dt.timezone.utc

_RFC3339_RE = re.compile(
    r"^\s*(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?"
    r"\s*([Zz]|[+-]\d{2}(?::?\d{2})?)?\s*$"
)


def rfc3339_parse(s):
    """Parse an RFC 3339 / ISO 8601 timestamp into an aware UTC ``datetime``."""
    if not isinstance(s, str):
        raise ValueError("timestamp must be a string")
    m = _RFC3339_RE.match(s)
    if not m:
        raise ValueError("not an RFC 3339 timestamp: %r" % (s,))
    year, month, day, hour, minute, second = (int(m.group(i)) for i in range(1, 7))
    frac = m.group(7) or ""
    micro = int((frac + "000000")[:6]) if frac else 0
    leap = 0
    if second == 60:  # leap second -> clamp, add it back as a delta
        second, leap = 59, 1
    tzs = m.group(8)
    if not tzs or tzs in ("Z", "z"):
        offset = _dt.timedelta(0)
    else:
        sign = -1 if tzs[0] == "-" else 1
        digits = tzs[1:].replace(":", "")
        hh = int(digits[:2])
        mm = int(digits[2:4]) if len(digits) >= 4 else 0
        if hh > 23 or mm > 59:
            raise ValueError("bad UTC offset in %r" % (s,))
        offset = sign * _dt.timedelta(hours=hh, minutes=mm)
    try:
        dt = _dt.datetime(year, month, day, hour, minute, second, micro, tzinfo=_dt.timezone(offset))
    except ValueError as exc:
        raise ValueError("invalid timestamp %r: %s" % (s, exc))
    if leap:
        dt = dt + _dt.timedelta(seconds=1)
    return dt.astimezone(UTC)


def rfc3339_to_epoch(s):
    """RFC 3339 string -> POSIX seconds (float)."""
    dt = rfc3339_parse(s)
    return calendar.timegm(dt.utctimetuple()) + dt.microsecond / 1e6


def rfc3339_format(value=None, timespec="auto"):
    """Format as ``YYYY-MM-DDTHH:MM:SS[.ffffff]Z`` in UTC.

    ``timespec``: ``"auto"`` (microseconds only when non-zero), ``"seconds"``, ``"micro"``
    (always 6 digits), ``"milli"`` (always 3 digits).
    """
    if value is None:
        dt = _dt.datetime.now(UTC)
    elif isinstance(value, _dt.datetime):
        dt = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        dt = _dt.datetime.fromtimestamp(float(value), UTC)
    else:
        raise TypeError("rfc3339_format expects datetime, epoch number or None")
    base = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if timespec == "seconds" or (timespec == "auto" and dt.microsecond == 0):
        return base + "Z"
    if timespec == "milli":
        return "%s.%03dZ" % (base, dt.microsecond // 1000)
    if timespec in ("auto", "micro"):
        return "%s.%06dZ" % (base, dt.microsecond)
    raise ValueError("unknown timespec %r" % (timespec,))


def b64url_encode(data):
    """URL-safe base64 without padding. ``str`` input is UTF-8 encoded."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return base64.urlsafe_b64encode(bytes(data)).rstrip(b"=").decode("ascii")


def b64url_decode(s):
    """Decode URL-safe (or standard) base64, tolerating missing/extra padding and whitespace.

    Raises ``ValueError`` on invalid input.
    """
    if isinstance(s, bytes):
        s = s.decode("ascii", "strict")
    s = "".join(s.split()).rstrip("=")
    s = s.replace("+", "-").replace("/", "_")
    if len(s) % 4 == 1:
        raise ValueError("invalid base64 length")
    s += "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s.encode("ascii"), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64: %s" % exc)


def jwt_claims(token):
    """Return the (unverified) claims dict of a JWT, or ``{}`` if it cannot be decoded."""
    try:
        if isinstance(token, bytes):
            token = token.decode("ascii")
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        claims = json.loads(b64url_decode(parts[1]).decode("utf-8"))
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def removeprefix(s, prefix):
    return s[len(prefix):] if prefix and s.startswith(prefix) else s


def removesuffix(s, suffix):
    return s[:-len(suffix)] if suffix and s.endswith(suffix) else s


def json_dumps_compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def utcnow_epoch():
    return time.time()


def utcnow():
    return _dt.datetime.now(UTC)
