"""Grok Build CLI login auth (``grok login``) for the ``grok_cli_proxy`` target (DESIGN §4).

File: ``options["auth_path"]`` | ``$GROK_AUTH_PATH`` | ``$GROK_HOME/auth.json`` | ``~/.grok/auth.json``;
a JSON object of entries keyed ``<issuer>::<client id>``::

    {"https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828": {"auth_mode": "oidc", "key": "<access>",
      "refresh_token": "...", "expires_at": "<RFC3339>", "create_time": "...", "oidc_issuer": "...",
      "oidc_client_id": "...", "principal_type"?: "...", "principal_id"?: "..."}, ...}

* Entry: the key above, else the first entry with ``auth_mode`` in (``oidc``, ``external``) and a
  ``key``. ``auth_mode == "api_key"`` / ``xai::api_key`` entries are exposed via ``api_key_entry()``
  (static xAI key route); ``web_login`` entries are ignored.
* Expiry: ``expires_at`` (RFC3339 or epoch) else ``create_time + 30 days`` else the JWT ``exp``.
* Refresh (300 s early or after a 401) under ``<auth file>.lock``: re-read, adopt a fresher on-disk
  entry, else form-POST ``grant_type=refresh_token&refresh_token&client_id`` (+ ``principal_type`` /
  ``principal_id``) to the issuer's ``token_endpoint`` from ``{issuer}/.well-known/
  openid-configuration`` (cached per process; fallback ``{issuer}/oauth2/token`` =
  ``https://auth.x.ai/oauth2/token``). ``AI_GATEWAY_XAI_TOKEN_URL`` replaces the endpoint,
  ``AI_GATEWAY_XAI_OIDC_ISSUER`` the issuer used for discovery. Result: ``key``, ``expires_at = now +
  expires_in``, ``create_time = now``, ``refresh_token`` (old one kept if none returned); atomic
  pretty JSON 0600 write keeping every other field/entry. ``invalid_grant`` -> terminal ("run `grok login`").
"""

import os
import threading
import time
from urllib.parse import urlencode, urlsplit

from ..compat import jwt_claims, rfc3339_format, rfc3339_to_epoch
from ..transport import TransportError, _is_loopback
from .base import AuthError, Token
from .codex_chatgpt import LockedFileAuth

__all__ = ["GrokCliAuth", "GROK_ENTRY_KEY", "GROK_CLIENT_ID", "XAI_ISSUER", "XAI_TOKEN_URL", "clear_discovery_cache"]

GROK_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_ISSUER = "https://auth.x.ai"
GROK_ENTRY_KEY = XAI_ISSUER + "::" + GROK_CLIENT_ID
XAI_TOKEN_URL = XAI_ISSUER + "/oauth2/token"
LOGIN_TTL = 30 * 24 * 3600.0
_OIDC_MODES = ("oidc", "external")

_DISCOVERY = {}  # issuer -> token_endpoint
_DISCOVERY_LOCK = threading.Lock()


def clear_discovery_cache():
    with _DISCOVERY_LOCK:
        _DISCOVERY.clear()


def _to_epoch(value):
    """RFC3339 string / epoch seconds / epoch milliseconds -> POSIX seconds, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) / 1000.0 if value > 1e11 else float(value)
    if isinstance(value, str) and value.strip():
        try:
            return rfc3339_to_epoch(value)
        except ValueError:
            try:
                return _to_epoch(float(value))
            except ValueError:
                return None
    return None


def _like(existing, epoch):
    """``epoch`` formatted like ``existing`` (number -> same unit; string/absent -> RFC3339 Z)."""
    if isinstance(existing, (int, float)) and not isinstance(existing, bool):
        return int(epoch * 1000) if existing > 1e11 else int(epoch)
    if isinstance(existing, str) and "." in existing:
        return rfc3339_format(epoch, "micro")
    return rfc3339_format(epoch, "seconds")


def _endpoint_ok(url):
    try:
        p = urlsplit(url)
    except ValueError:
        return False
    return bool(p.hostname) and (p.scheme == "https" or (p.scheme == "http" and _is_loopback(p.hostname)))


class GrokCliAuth(LockedFileAuth):
    """``grok_cli`` auth kind (see module docstring)."""

    kind = "grok_cli"
    cli_hint = "run `grok login`"
    label = "Grok login"

    def auth_path(self):
        explicit = self.options.get("auth_path") or self._getenv("GROK_AUTH_PATH")
        if explicit:
            return os.path.expanduser(explicit)
        home = self._getenv("GROK_HOME")
        if home:
            return os.path.join(os.path.expanduser(home), "auth.json")
        return os.path.join(self._home(), ".grok", "auth.json")

    # ---- entry selection ---------------------------------------------------------------
    @staticmethod
    def _entries(data):
        return [(k, v) for k, v in data.items() if isinstance(v, dict)]

    @classmethod
    def _select(cls, data, preferred=None):
        entries = cls._entries(data)
        for name in (preferred, GROK_ENTRY_KEY):
            entry = data.get(name) if name else None
            if isinstance(entry, dict) and entry.get("auth_mode", "oidc") in _OIDC_MODES and entry.get("key"):
                return name, entry
        for name, entry in entries:
            if entry.get("auth_mode") in _OIDC_MODES and isinstance(entry.get("key"), str) and entry["key"]:
                return name, entry
        return None, None

    def _parse(self, data):
        name, entry = self._select(data, self._token.extra.get("entry") if self._token is not None else None)
        if entry is None:
            if self._api_key_in(data):
                raise AuthError("Grok is logged in with an API key, not an OIDC session",
                                "use the `xai` route with that key")
            raise AuthError("no Grok login session in %s" % self.auth_path(), self.cli_hint)
        key = entry["key"]
        if not isinstance(key, str):
            raise AuthError("malformed Grok login entry in %s" % self.auth_path(), self.cli_hint)
        expires_at = _to_epoch(entry.get("expires_at"))
        if expires_at is None:
            created = _to_epoch(entry.get("create_time"))
            if created is not None:
                expires_at = created + LOGIN_TTL
            else:
                exp = jwt_claims(key).get("exp")
                expires_at = float(exp) if isinstance(exp, (int, float)) and not isinstance(exp, bool) else None
        issuer, _, client = name.partition("::")
        extra = {"entry": name, "auth_mode": entry.get("auth_mode", "oidc"),
                 "refresh_token": entry.get("refresh_token") if isinstance(entry.get("refresh_token"), str) else None,
                 "issuer": entry.get("oidc_issuer") or (issuer if issuer.startswith("http") else XAI_ISSUER),
                 "client_id": entry.get("oidc_client_id") or (client if client else GROK_CLIENT_ID)}
        for k in ("principal_type", "principal_id"):
            if isinstance(entry.get(k), str) and entry[k]:
                extra[k] = entry[k]
        return Token(key, expires_at, extra)

    @staticmethod
    def _api_key_in(data):
        for name, entry in GrokCliAuth._entries(data):
            if entry.get("auth_mode") == "api_key" or name == "xai::api_key":
                key = entry.get("key") or entry.get("api_key")
                if isinstance(key, str) and key.strip():
                    return key.strip()
        return None

    # ---- token endpoint ----------------------------------------------------------------
    def _token_endpoint(self, issuer):
        override = self._getenv("AI_GATEWAY_XAI_TOKEN_URL")
        if override:
            return override
        issuer = (self._getenv("AI_GATEWAY_XAI_OIDC_ISSUER") or issuer or XAI_ISSUER).rstrip("/")
        with _DISCOVERY_LOCK:
            cached = _DISCOVERY.get(issuer)
        if cached:
            return cached
        endpoint = None
        try:
            resp = self._client().request("GET", issuer + "/.well-known/openid-configuration",
                                          {"Accept": "application/json"}, stream=False, timeout=15)
            if resp.status == 200:
                doc = resp.json()
                ep = doc.get("token_endpoint") if isinstance(doc, dict) else None
                if isinstance(ep, str) and _endpoint_ok(ep):
                    endpoint = ep
        except (TransportError, ValueError):
            endpoint = None
        if endpoint is None:
            return issuer + "/oauth2/token"  # not cached: discovery is retried next refresh
        with _DISCOVERY_LOCK:
            _DISCOVERY[issuer] = endpoint
        return endpoint

    def _call_token_endpoint(self, current, refresh_token):
        fields = [("grant_type", "refresh_token"), ("refresh_token", refresh_token),
                  ("client_id", current.extra.get("client_id") or GROK_CLIENT_ID)]
        for k in ("principal_type", "principal_id"):
            if current.extra.get(k):
                fields.append((k, current.extra[k]))
        return self._post(self._token_endpoint(current.extra.get("issuer")),
                          {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
                          urlencode(fields).encode("ascii"))

    def _merge(self, data, current, payload, refresh_token):
        name = current.extra["entry"]
        entry = data.get(name)
        if not isinstance(entry, dict):
            entry = data[name] = {"auth_mode": current.extra.get("auth_mode", "oidc")}
        now = time.time()
        access = payload["access_token"]
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, str) and expires_in.isdigit():
            expires_in = int(expires_in)
        if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool) and expires_in > 0:
            expires_at = now + float(expires_in)
        else:
            exp = jwt_claims(access).get("exp")
            expires_at = float(exp) if isinstance(exp, (int, float)) and not isinstance(exp, bool) else now + 3600.0
        entry["key"] = access
        new_rt = payload.get("refresh_token")
        entry["refresh_token"] = new_rt if isinstance(new_rt, str) and new_rt else refresh_token
        entry["expires_at"] = _like(entry.get("expires_at"), expires_at)
        entry["create_time"] = _like(entry.get("create_time"), now)
        if "id_token" in entry and isinstance(payload.get("id_token"), str):
            entry["id_token"] = payload["id_token"]

    # ---- public extras -----------------------------------------------------------------
    def api_key_entry(self):
        """Key of an ``api_key`` login entry (for the static ``xai`` route), else None."""
        try:
            data = self._read_raw()
        except (ValueError, AuthError):
            return None
        return self._api_key_in(data) if data else None

    def describe(self):
        with self._lock:
            tok = self._safe_load()
        d = {"kind": self.kind, "provider": self.provider_id, "available": tok is not None,
             "source": "grok auth.json", "path": self.auth_path(), "api_key_entry": bool(self.api_key_entry())}
        if tok is not None:
            d["entry"] = tok.extra.get("entry")
            d["auth_mode"] = tok.extra.get("auth_mode")
            if tok.expires_at is not None:
                d["expires_at"] = rfc3339_format(tok.expires_at, "seconds")
        else:
            d["hint"] = "use the xai route with the stored API key" if d["api_key_entry"] else self.cli_hint
        return d
