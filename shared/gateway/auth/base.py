"""Auth provider contract (DESIGN §4).

``AuthProvider``
    kind: str
    available() -> bool                       cheap, no network; may read files
    headers(force_refresh=False) -> Dict      thread-safe, single-flight refresh; raises AuthError
    on_unauthorized(used) -> bool             called after an upstream 401 with the headers that were
                                              used; True => credentials changed (refreshed or replaced
                                              by another thread/process) -> caller retries ONCE
    describe() -> Dict                        source, expires_at, account hint — NEVER secrets

``RefreshingAuth`` implements the common "cached token + refresh" pattern for codex_chatgpt,
grok_cli and gcloud_adc. Subclasses implement ``_load()``, ``_refresh(token)`` and
``_token_headers(token)``; the base class supplies the per-provider lock, expiry margin and the
"someone else already refreshed" short-circuit. Cross-process safety (file locks, re-read, adopt
newer on-disk tokens) is the subclass's job inside ``_refresh``.
"""

import threading
import time

__all__ = ["AuthError", "AuthProvider", "Token", "RefreshingAuth", "redact_headers"]


class AuthError(Exception):
    """Credentials missing/expired/unrefreshable. ``hint`` tells the user how to fix it
    (e.g. "run `codex login`"). ``terminal`` = retrying cannot help."""

    def __init__(self, message, hint=None, terminal=True):
        Exception.__init__(self, message)
        self.message = message
        self.hint = hint
        self.terminal = terminal

    def __str__(self):
        return self.message + ((" — " + self.hint) if self.hint else "")


def redact_headers(headers):
    """Copy of ``headers`` with credential-bearing values replaced (for logs/describe)."""
    out = {}
    for k, v in (headers or {}).items():
        lk = k.lower()
        if lk in ("authorization", "x-api-key", "x-goog-api-key", "proxy-authorization", "cookie") or \
                "token" in lk or "secret" in lk or "key" in lk:
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


class AuthProvider(object):
    kind = "base"

    def __init__(self, provider_id=""):
        self.provider_id = provider_id
        self._lock = threading.RLock()  # per-provider single-flight lock

    def available(self):
        raise NotImplementedError

    def headers(self, force_refresh=False):
        raise NotImplementedError

    def on_unauthorized(self, used):
        return False

    def describe(self):
        return {"kind": self.kind, "provider": self.provider_id, "available": self._safe_available()}

    def relogin_hint(self):
        """Short user-facing hint used in 401 errors (e.g. "run `grok login`")."""
        return None

    def _safe_available(self):
        try:
            return bool(self.available())
        except Exception:
            return False

    def __repr__(self):
        return "%s(provider=%r)" % (type(self).__name__, self.provider_id)


class Token(object):
    """An access token plus metadata. ``expires_at`` is POSIX seconds or None (unknown)."""

    __slots__ = ("access_token", "expires_at", "extra")

    def __init__(self, access_token, expires_at=None, extra=None):
        self.access_token = access_token
        self.expires_at = None if expires_at is None else float(expires_at)
        self.extra = dict(extra or {})

    def expires_within(self, seconds, now=None):
        if self.expires_at is None:
            return False
        return self.expires_at - (time.time() if now is None else now) < seconds

    def __repr__(self):
        return "Token(access_token=<redacted>, expires_at=%r)" % (self.expires_at,)


class RefreshingAuth(AuthProvider):
    """Cached token with single-flight refresh.

    Subclass hooks (called with ``self._lock`` held):
      ``_load() -> Optional[Token]``      read current credentials (file/env/cache), no network
      ``_refresh(token) -> Token``        obtain a new token (may hit the network / take file locks);
                                          raise ``AuthError`` when impossible
      ``_token_headers(token) -> Dict``   request headers for a token
    """

    kind = "refreshing"
    refresh_margin = 300.0  # refresh when the token expires within this many seconds

    def __init__(self, provider_id=""):
        AuthProvider.__init__(self, provider_id)
        self._token = None  # Optional[Token]

    # ---- hooks -------------------------------------------------------------------------
    def _load(self):
        raise NotImplementedError

    def _refresh(self, token):
        raise NotImplementedError

    def _token_headers(self, token):
        return {"Authorization": "Bearer " + token.access_token}

    # ---- contract ----------------------------------------------------------------------
    def available(self):
        with self._lock:
            try:
                return (self._token or self._load()) is not None
            except AuthError:
                return False

    def headers(self, force_refresh=False):
        with self._lock:
            tok = self._token or self._load()
            if tok is None:
                raise AuthError("no credentials for provider %r" % (self.provider_id,), self.relogin_hint())
            if force_refresh or tok.expires_within(self.refresh_margin):
                tok = self._refresh(tok)
            self._token = tok
            return dict(self._token_headers(tok))

    def on_unauthorized(self, used):
        with self._lock:
            if self._token is not None and dict(self._token_headers(self._token)) != dict(used or {}):
                return True  # another thread already replaced the token we used
            try:
                current = self._token or self._load()
                if current is None:
                    return False
                self._token = self._refresh(current)
                return True
            except AuthError:
                return False

    def current_token(self):
        """The cached token (no refresh), for describe()/tests."""
        with self._lock:
            return self._token

    def describe(self):
        d = AuthProvider.describe(self)
        tok = self._token
        if tok is not None and tok.expires_at is not None:
            from ..compat import rfc3339_format

            d["expires_at"] = rfc3339_format(tok.expires_at, "seconds")
        return d
