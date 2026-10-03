"""ChatGPT (Codex CLI login) auth for the ``chatgpt_codex`` Responses target (DESIGN §4).

Credentials: ``options["auth_path"]`` | ``$CODEX_HOME/auth.json`` | ``~/.codex/auth.json`` written by
``codex login`` (file store only — keyring-stored credentials are not readable). Shape::

    {"OPENAI_API_KEY": null, "tokens": {"id_token", "access_token", "refresh_token", "account_id"},
     "last_refresh": "<RFC3339>", ...unknown keys preserved...}

* A non-empty ``OPENAI_API_KEY`` means an API-key login: this kind is then unavailable and
  ``api_key()`` returns the key so the launcher can offer the ``openai`` route instead.
* Expiry = the access token's JWT ``exp``; account id = ``tokens.account_id``, else the id token's
  (then access token's) ``["https://api.openai.com/auth"]["chatgpt_account_id"]`` claim.
* Refresh (``exp - now < 300`` or after a 401) runs under ``ExclusiveFileLock(<auth.json>.lock)``:
  re-read; if the on-disk credentials differ from ours and are fresh, adopt them (another process
  refreshed); else POST JSON ``{grant_type, client_id, refresh_token}`` to
  ``https://auth.openai.com/oauth/token`` (``AI_GATEWAY_OPENAI_AUTH_URL``), re-read, merge
  ``tokens.{access_token,id_token,refresh_token}`` + ``last_refresh`` (RFC3339 Z) keeping unknown
  fields, atomic pretty write 0600. ``invalid_grant`` / ``refresh_token_reused`` / any 400/401 ->
  re-read; a rotated on-disk token is adopted, else terminal ``AuthError`` ("run `codex login`").

``LockedFileAuth`` (also used by ``grok_cli``) holds that lock/re-read/adopt/refresh/merge/write
skeleton; subclasses supply ``auth_path``, ``_parse``, ``_call_token_endpoint`` and ``_merge``.
"""

import json
import os
import time

from ..atomicio import atomic_write_json
from ..compat import jwt_claims, rfc3339_format
from ..filelock import ExclusiveFileLock, LockTimeout
from ..transport import HttpClient, TransportError
from .base import AuthError, RefreshingAuth, Token

__all__ = ["CodexChatGPTAuth", "LockedFileAuth", "oauth_error_code", "CODEX_CLIENT_ID", "OPENAI_TOKEN_URL",
           "TERMINAL_OAUTH_ERRORS"]

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
_OPENAI_AUTH_CLAIM = "https://api.openai.com/auth"
KEYRING_HINT = ("codex login (file-based); keyring-stored creds unsupported — set "
                "cli_auth_credentials_store = \"file\"")

#: OAuth error codes after which the stored refresh token can never work again
TERMINAL_OAUTH_ERRORS = frozenset([
    "invalid_grant", "refresh_token_reused", "refresh_token_expired", "refresh_token_invalidated",
    "invalid_token", "invalid_client", "unauthorized_client", "access_denied",
])


def oauth_error_code(payload):
    """Error code from an OAuth/OpenAI error body (``{"error": "x"}``, ``{"error": {"code": "x"}}``,
    ``{"code": "x"}``, ``{"detail": {"code": "x"}}``) or None."""
    if not isinstance(payload, dict):
        return None
    for node in (payload, payload.get("detail")):
        if not isinstance(node, dict):
            continue
        err = node.get("error")
        if isinstance(err, str) and err:
            return err
        if isinstance(err, dict):
            for k in ("code", "type"):
                if isinstance(err.get(k), str) and err.get(k):
                    return err[k]
        if isinstance(node.get("code"), str) and node.get("code"):
            return node["code"]
    return None


class LockedFileAuth(RefreshingAuth):
    """OAuth credentials kept in a JSON file shared with a CLI (see module docstring)."""

    kind = "locked_file"
    cli_hint = None  # e.g. "run `codex login`"
    label = "login"

    def __init__(self, provider_id="", options=None, environ=None, http=None):
        RefreshingAuth.__init__(self, provider_id)
        self.options = dict(options or {})
        self._environ = environ
        self._http = http
        self.lock_timeout = float(self.options.get("lock_timeout", 10.0))

    # ---- environment -------------------------------------------------------------------
    def _getenv(self, name):
        env = os.environ if self._environ is None else self._environ
        value = env.get(name)
        return value or None

    def _home(self):
        home = self._getenv("USERPROFILE" if os.name == "nt" else "HOME")
        return home or os.path.expanduser("~")

    def _client(self):
        if self._http is None:
            self._http = HttpClient(timeout=30.0, connect_timeout=15.0, environ=self._environ)
        return self._http

    # ---- subclass hooks ----------------------------------------------------------------
    def auth_path(self):
        raise NotImplementedError

    def _parse(self, data):
        """``Token`` (``extra["refresh_token"]`` set) from the file's JSON; ``AuthError`` if unusable."""
        raise NotImplementedError

    def _call_token_endpoint(self, current, refresh_token):
        """POST the refresh request -> ``(status, payload)``; may raise ``AuthError``."""
        raise NotImplementedError

    def _merge(self, data, current, payload, refresh_token):
        """Apply a successful token response to ``data`` (the freshly re-read file) in place."""
        raise NotImplementedError

    def relogin_hint(self):
        return self.cli_hint

    # ---- file access -------------------------------------------------------------------
    def _read_raw(self):
        """Parsed file (dict) or None if missing; ``ValueError`` if malformed."""
        path = self.auth_path()
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise AuthError("cannot read %s: %s" % (path, exc), self.cli_hint)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON value is not an object")
        return data

    def _read(self):
        try:
            data = self._read_raw()
        except ValueError as exc:
            raise AuthError("cannot parse %s: %s" % (self.auth_path(), exc), self.cli_hint)
        if data is None:
            raise AuthError("no %s credentials at %s" % (self.label, self.auth_path()), self.cli_hint)
        return data

    def _load(self):
        return self._parse(self._read())

    def _post(self, url, headers, body):
        """``(status, payload-or-None)``; transport failures -> non-terminal ``AuthError``."""
        try:
            resp = self._client().request("POST", url, headers, body, stream=False)
        except TransportError as exc:
            raise AuthError("%s token refresh failed: %s" % (self.label, exc.message), self.cli_hint,
                            terminal=False)
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        return resp.status, payload

    # ---- refresh skeleton --------------------------------------------------------------
    @staticmethod
    def _same(a, b):
        return a.access_token == b.access_token and a.extra.get("refresh_token") == b.extra.get("refresh_token")

    def _refresh(self, token):
        path = self.auth_path()
        lock = ExclusiveFileLock(path + ".lock", timeout=self.lock_timeout)
        try:
            lock.acquire()
        except LockTimeout:
            raise AuthError("timed out after %.0fs waiting for %s.lock (another process is refreshing the %s)"
                            % (self.lock_timeout, path, self.label), None, terminal=False)
        except OSError as exc:
            raise AuthError("cannot lock %s.lock: %s" % (path, exc), None, terminal=False)
        try:
            return self._refresh_locked(token, path)
        except OSError as exc:  # e.g. the write-back failed (disk full, permissions)
            raise AuthError("cannot update %s: %s" % (path, exc), None, terminal=False)
        finally:
            lock.release()

    def _refresh_locked(self, token, path):
        initial = self._read()
        disk = self._parse(initial)
        if not self._same(disk, token) and not disk.expires_within(self.refresh_margin):
            return disk  # another process (or the CLI) already refreshed: adopt
        refresh_token = disk.extra.get("refresh_token")
        if not refresh_token:
            raise AuthError("%s has no refresh token" % path, self.cli_hint)
        status, payload = self._call_token_endpoint(disk, refresh_token)
        if 200 <= status < 300 and isinstance(payload, dict) and isinstance(payload.get("access_token"), str) \
                and payload["access_token"]:
            try:
                data = self._read_raw()
            except ValueError:
                data = initial  # torn write by a lock-less writer: our merge repairs it
            if data is None:
                raise AuthError("%s was removed while refreshing (logged out?)" % path, self.cli_hint)
            self._merge(data, disk, payload, refresh_token)
            atomic_write_json(path, data, indent=2, mode=0o600)
            return self._parse(data)
        code = oauth_error_code(payload)
        if status in (400, 401) or code in TERMINAL_OAUTH_ERRORS:
            latest = self._parse(self._read())
            if latest.extra.get("refresh_token") != refresh_token:
                return latest  # rotated meanwhile by a writer that does not take our lock
            raise AuthError("%s is no longer valid (%s)" % (self.label, code or "HTTP %d" % status), self.cli_hint)
        raise AuthError("%s token refresh failed (HTTP %d%s)" % (self.label, status, (", " + code) if code else ""),
                        self.cli_hint, terminal=False)

    def _safe_load(self):
        try:
            return self._token or self._load()
        except AuthError:
            return None


class CodexChatGPTAuth(LockedFileAuth):
    """``codex_chatgpt`` auth kind (see module docstring)."""

    kind = "codex_chatgpt"
    cli_hint = "run `codex login`"
    label = "ChatGPT login"

    def auth_path(self):
        explicit = self.options.get("auth_path")
        if explicit:
            return os.path.expanduser(explicit)
        home = self._getenv("CODEX_HOME")
        if home:
            return os.path.join(os.path.expanduser(home), "auth.json")
        return os.path.join(self._home(), ".codex", "auth.json")

    def _read(self):
        try:
            data = self._read_raw()
        except ValueError as exc:
            raise AuthError("cannot parse %s: %s" % (self.auth_path(), exc), self.cli_hint)
        if data is None:
            raise AuthError("no Codex credentials at %s" % self.auth_path(), KEYRING_HINT)
        return data

    @staticmethod
    def _account_id(tokens, id_claims, access_claims):
        acct = tokens.get("account_id")
        if isinstance(acct, str) and acct:
            return acct
        for claims in (id_claims, access_claims):
            auth = claims.get(_OPENAI_AUTH_CLAIM)
            if isinstance(auth, dict) and isinstance(auth.get("chatgpt_account_id"), str) \
                    and auth["chatgpt_account_id"]:
                return auth["chatgpt_account_id"]
        return None

    def _parse(self, data):
        if isinstance(data.get("OPENAI_API_KEY"), str) and data["OPENAI_API_KEY"].strip():
            raise AuthError("Codex is logged in with an API key, not ChatGPT", "use the `openai` route with that key")
        tokens = data.get("tokens")
        access = tokens.get("access_token") if isinstance(tokens, dict) else None
        if not isinstance(access, str) or not access:
            raise AuthError("no ChatGPT tokens in %s" % self.auth_path(), self.cli_hint)
        id_token = tokens.get("id_token") if isinstance(tokens.get("id_token"), str) else None
        id_claims = jwt_claims(id_token) if id_token else {}
        access_claims = jwt_claims(access)
        exp = access_claims.get("exp")
        expires_at = float(exp) if isinstance(exp, (int, float)) and not isinstance(exp, bool) else None
        refresh = tokens.get("refresh_token") if isinstance(tokens.get("refresh_token"), str) else None
        auth = id_claims.get(_OPENAI_AUTH_CLAIM) if isinstance(id_claims.get(_OPENAI_AUTH_CLAIM), dict) else {}
        extra = {"refresh_token": refresh, "account_id": self._account_id(tokens, id_claims, access_claims),
                 "email": id_claims.get("email") if isinstance(id_claims.get("email"), str) else None,
                 "plan": auth.get("chatgpt_plan_type") if isinstance(auth.get("chatgpt_plan_type"), str) else None,
                 "last_refresh": data.get("last_refresh") if isinstance(data.get("last_refresh"), str) else None}
        return Token(access, expires_at, extra)

    def _token_url(self):
        return self._getenv("AI_GATEWAY_OPENAI_AUTH_URL") or OPENAI_TOKEN_URL

    def _call_token_endpoint(self, current, refresh_token):
        body = json.dumps({"grant_type": "refresh_token", "client_id": CODEX_CLIENT_ID,
                           "refresh_token": refresh_token}).encode("utf-8")
        return self._post(self._token_url(), {"Content-Type": "application/json", "Accept": "application/json"},
                          body)

    def _merge(self, data, current, payload, refresh_token):
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            tokens = data["tokens"] = {}
        tokens["access_token"] = payload["access_token"]
        if isinstance(payload.get("id_token"), str) and payload["id_token"]:
            tokens["id_token"] = payload["id_token"]
        new_rt = payload.get("refresh_token")
        tokens["refresh_token"] = new_rt if isinstance(new_rt, str) and new_rt else refresh_token
        data["last_refresh"] = rfc3339_format(time.time(), "micro")

    def _token_headers(self, token):
        h = {"Authorization": "Bearer " + token.access_token}
        if token.extra.get("account_id"):
            h["ChatGPT-Account-ID"] = token.extra["account_id"]
        return h

    # ---- public extras -----------------------------------------------------------------
    def account_id(self):
        """ChatGPT account id (for the ``ChatGPT-Account-ID`` header) or None."""
        with self._lock:
            tok = self._safe_load()
            return tok.extra.get("account_id") if tok is not None else None

    def api_key(self):
        """``OPENAI_API_KEY`` stored by an API-key ``codex login`` (for the ``openai`` route), else None."""
        try:
            data = self._read_raw()
        except (ValueError, AuthError):
            return None
        key = data.get("OPENAI_API_KEY") if data else None
        return key.strip() if isinstance(key, str) and key.strip() else None

    def describe(self):
        with self._lock:
            tok = self._safe_load()
        d = {"kind": self.kind, "provider": self.provider_id, "available": tok is not None,
             "source": "codex auth.json", "path": self.auth_path()}
        if tok is not None:
            if tok.expires_at is not None:
                d["expires_at"] = rfc3339_format(tok.expires_at, "seconds")
            acct = tok.extra.get("account_id")
            d["account"] = tok.extra.get("email") or (acct[:8] + "…" if acct else None)
            if tok.extra.get("plan"):
                d["plan"] = tok.extra["plan"]
            if tok.extra.get("last_refresh"):
                d["last_refresh"] = tok.extra["last_refresh"]
        elif self.api_key():
            d["hint"] = "codex is logged in with an API key; use the openai route"
        elif not os.path.isfile(self.auth_path()):
            d["hint"] = KEYRING_HINT
        else:
            d["hint"] = self.cli_hint
        return d
