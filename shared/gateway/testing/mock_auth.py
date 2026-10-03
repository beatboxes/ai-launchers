"""Mock OAuth token servers (DESIGN §7) registered as ``MockServer`` kinds.

``openai_oauth``  ``POST /oauth/token`` JSON ``{grant_type, client_id, refresh_token}`` (ChatGPT/Codex).
``xai_oidc``      ``GET /.well-known/openid-configuration`` (``token_endpoint`` -> itself) and
                  ``POST /oauth2/token`` form ``grant_type, refresh_token, client_id[, principal_*]``.
``google_oauth``  ``POST /token`` form ``grant_type, client_id, client_secret, refresh_token`` (no rotation).

Rotation (openai_oauth / xai_oidc): every successful refresh spends the presented refresh token and
issues a new one. Presenting an already-spent token answers 400 (``refresh_token_reused`` /
``invalid_grant``) and REVOKES its whole successor chain — exactly what makes an unsynchronized
multi-process refresh fail loudly. State lives in ``server.state["chain"]`` (``TokenChain``):
``refreshes`` (successful network refreshes), ``reuse_events``, ``valid``/``revoked`` sets,
``latest()``. Helpers: ``refresh_count(server)``, ``token_chain(server)``, ``make_jwt(claims)``.

Options (``MockServer(kind, options={...})``):
  common        ``refresh_tokens`` (initially valid; default ``["rt-0"]``), ``accept_unknown`` (False),
                ``expires_in``, ``delay`` (s, before answering), ``fail_status`` + ``fail_count``
                (answer the next N refreshes with that status), ``on_request(req)`` hook,
                ``omit_refresh_token`` (answer without a new RT; the old one stays valid)
  openai_oauth  ``client_id`` (Codex id), ``account_id``, ``email``, ``jwt`` (True: JWT access tokens
                with ``exp``), ``error_style`` ("flat" ``{"error": code}`` | "openai" nested)
  xai_oidc      ``client_id`` (Grok id), ``discovery_status`` (200), ``jwt`` (False)
  google_oauth  ``client_id``, ``client_secret``, ``refresh_token`` (accepted RT, default "g-rt")
"""

import json
import threading
import time
from urllib.parse import parse_qs

from ..compat import b64url_encode
from .mock_upstreams import register_kind

__all__ = ["TokenChain", "token_chain", "refresh_count", "make_jwt", "CODEX_CLIENT_ID", "GROK_CLIENT_ID",
           "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"]

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
GROK_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
GOOGLE_CLIENT_ID = "mock-client.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "mock-client-secret"


def make_jwt(claims, header=None):
    """Unsigned JWT-shaped token (``header.payload.mocksig``) carrying ``claims``."""
    head = header or {"alg": "RS256", "typ": "JWT", "kid": "mock"}
    return "%s.%s.mocksig" % (b64url_encode(json.dumps(head, separators=(",", ":"))),
                              b64url_encode(json.dumps(claims, separators=(",", ":"))))


class TokenChain(object):
    """Refresh-token rotation with reuse detection (thread-safe)."""

    def __init__(self, initial=("rt-0",), prefix="rt", rotate=True, accept_unknown=False):
        self.valid = set(initial)
        self.successor = {}  # spent RT -> the RT issued for it
        self.revoked = set()
        self.refreshes = 0
        self.reuse_events = 0
        self.rotate = rotate
        self.accept_unknown = accept_unknown
        self._prefix = prefix
        self._n = 0
        self._latest = list(initial)[-1] if initial else None
        self._lock = threading.Lock()

    def latest(self):
        with self._lock:
            return self._latest

    def redeem(self, rt):
        """-> ``(new_rt, None)`` on success, ``(None, "reused"|"revoked"|"unknown")`` otherwise."""
        with self._lock:
            if rt in self.revoked:
                return None, "revoked"
            if rt in self.successor:
                nxt = self.successor[rt]
                while nxt is not None and nxt not in self.revoked:
                    self.revoked.add(nxt)
                    self.valid.discard(nxt)
                    nxt = self.successor.get(nxt)
                self.reuse_events += 1
                return None, "reused"
            if rt not in self.valid and not self.accept_unknown:
                return None, "unknown"
            self.refreshes += 1
            if not self.rotate:
                self.valid.add(rt)
                return rt, None
            self._n += 1
            new = "%s-%d" % (self._prefix, self._n)
            self.valid.discard(rt)
            self.successor[rt] = new
            self.valid.add(new)
            self._latest = new
            return new, None


def token_chain(server):
    return server.state["chain"]


def refresh_count(server):
    """Successful network refreshes served so far."""
    return server.state["chain"].refreshes


def _form(req):
    try:
        q = parse_qs((req.body or b"").decode("utf-8"), keep_blank_values=True)
    except UnicodeDecodeError:
        return {}
    return {k: v[0] for k, v in q.items()}


def _pre(server, req, resp):
    """Shared preamble: hook, delay, forced failures. True if a response was already sent."""
    hook = server.options.get("on_request")
    if hook is not None:
        hook(req)
    delay = float(server.options.get("delay") or 0)
    if delay:
        time.sleep(delay)
    with server.lock:
        left = server.state.get("fail_left", 0)
        if left > 0:
            server.state["fail_left"] = left - 1
            resp.send_json(int(server.options.get("fail_status", 500)),
                           {"error": {"message": "mock token endpoint failure", "type": "server_error"}})
            return True
    return False


def _init_state(server, chain):
    o = server.options
    server.state["chain"] = chain
    server.state["fail_left"] = int(o.get("fail_count") or (1 if o.get("fail_status") else 0))
    server.state["issued"] = 0


def _rotating_chain(server, prefix):
    o = server.options
    return TokenChain(o.get("refresh_tokens") or ("rt-0",), prefix, rotate=not o.get("omit_refresh_token"),
                      accept_unknown=bool(o.get("accept_unknown")))


def _next_serial(server):
    with server.lock:
        server.state["issued"] += 1
        return server.state["issued"]


# ---------------------------------------------------------------------------------------
# openai_oauth
# ---------------------------------------------------------------------------------------

def _openai_factory(server):
    _init_state(server, _rotating_chain(server, "rt-openai"))
    o = server.options
    client_id = o.get("client_id", CODEX_CLIENT_ID)
    account = o.get("account_id", "acct-mock-0001")
    email = o.get("email", "dev@example.com")
    expires_in = int(o.get("expires_in", 3600))

    def error(resp, status, code, message):
        if o.get("error_style") == "openai":
            body = {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}}
        else:
            body = {"error": code, "error_description": message}
        resp.send_json(status, body)

    def handle(req, resp):
        if req.path != "/oauth/token" or req.method != "POST":
            return
        if _pre(server, req, resp):
            return
        body = req.json if isinstance(req.json, dict) else {}
        if body.get("grant_type") != "refresh_token":
            return error(resp, 400, "unsupported_grant_type", "grant_type must be refresh_token")
        if body.get("client_id") != client_id:
            return error(resp, 401, "invalid_client", "unknown client")
        rt = body.get("refresh_token")
        new_rt, why = token_chain(server).redeem(rt) if isinstance(rt, str) else (None, "unknown")
        if why == "reused":
            return error(resp, 400, "refresh_token_reused", "Your refresh token has already been used to generate "
                                                            "a new access token. Please try signing in again.")
        if why is not None:
            return error(resp, 400, "invalid_grant", "refresh token is invalid or revoked")
        n = _next_serial(server)
        now = int(time.time())
        auth_claim = {"chatgpt_account_id": account, "chatgpt_plan_type": "plus"}
        access = make_jwt({"iss": server.url, "exp": now + expires_in, "iat": now, "jti": "at-%d" % n,
                           "https://api.openai.com/auth": auth_claim}) if o.get("jwt", True) else "at-openai-%d" % n
        out = {"access_token": access, "token_type": "Bearer", "expires_in": expires_in,
               "id_token": make_jwt({"email": email, "exp": now + expires_in, "iat": now,
                                     "https://api.openai.com/auth": auth_claim}),
               "scope": "openid profile email offline_access"}
        if not o.get("omit_refresh_token"):
            out["refresh_token"] = new_rt
        resp.send_json(200, out)

    return handle


# ---------------------------------------------------------------------------------------
# xai_oidc
# ---------------------------------------------------------------------------------------

def _xai_factory(server):
    _init_state(server, _rotating_chain(server, "rt-xai"))
    server.state["discovery_hits"] = 0
    o = server.options
    client_id = o.get("client_id", GROK_CLIENT_ID)
    expires_in = int(o.get("expires_in", 21600))

    def handle(req, resp):
        if req.path == "/.well-known/openid-configuration" and req.method == "GET":
            with server.lock:
                server.state["discovery_hits"] += 1
            status = int(o.get("discovery_status", 200))
            if status != 200:
                return resp.send_json(status, {"error": "not_found"})
            return resp.send_json(200, {"issuer": server.url, "token_endpoint": server.url + "/oauth2/token",
                                        "authorization_endpoint": server.url + "/oauth2/auth",
                                        "jwks_uri": server.url + "/.well-known/jwks.json",
                                        "grant_types_supported": ["authorization_code", "refresh_token"]})
        if req.path != "/oauth2/token" or req.method != "POST":
            return
        if _pre(server, req, resp):
            return
        form = _form(req)
        with server.lock:
            server.state["last_form_keys"] = sorted(form)
            server.state["last_principal"] = (form.get("principal_type"), form.get("principal_id"))
        if form.get("grant_type") != "refresh_token":
            return resp.send_json(400, {"error": "unsupported_grant_type"})
        if form.get("client_id") != client_id:
            return resp.send_json(401, {"error": "invalid_client", "error_description": "unknown client"})
        new_rt, why = token_chain(server).redeem(form.get("refresh_token", ""))
        if why is not None:
            desc = "refresh token already used" if why == "reused" else "refresh token is invalid or revoked"
            return resp.send_json(400, {"error": "invalid_grant", "error_description": desc})
        n = _next_serial(server)
        now = int(time.time())
        access = make_jwt({"iss": server.url, "exp": now + expires_in, "iat": now, "jti": "xat-%d" % n}) \
            if o.get("jwt") else "xai-at-%d" % n
        out = {"access_token": access, "token_type": "Bearer", "expires_in": expires_in}
        if not o.get("omit_refresh_token"):
            out["refresh_token"] = new_rt
        resp.send_json(200, out)

    return handle


# ---------------------------------------------------------------------------------------
# google_oauth
# ---------------------------------------------------------------------------------------

def _google_factory(server):
    o = server.options
    _init_state(server, TokenChain((o.get("refresh_token", "g-rt"),), "g-rt", rotate=False))
    expires_in = int(o.get("expires_in", 3599))

    def handle(req, resp):
        if req.path != "/token" or req.method != "POST":
            return
        if _pre(server, req, resp):
            return
        form = _form(req)
        if form.get("grant_type") != "refresh_token":
            return resp.send_json(400, {"error": "unsupported_grant_type"})
        if form.get("client_id") != o.get("client_id", GOOGLE_CLIENT_ID) or \
                form.get("client_secret") != o.get("client_secret", GOOGLE_CLIENT_SECRET):
            return resp.send_json(401, {"error": "invalid_client",
                                        "error_description": "The OAuth client was not found."})
        _, why = token_chain(server).redeem(form.get("refresh_token", ""))
        if why is not None:
            return resp.send_json(400, {"error": "invalid_grant",
                                        "error_description": "Token has been expired or revoked."})
        n = _next_serial(server)
        resp.send_json(200, {"access_token": "ya29.mock-%d" % n, "expires_in": expires_in, "token_type": "Bearer",
                             "scope": "https://www.googleapis.com/auth/cloud-platform"})

    return handle


register_kind("openai_oauth", _openai_factory)
register_kind("xai_oidc", _xai_factory)
register_kind("google_oauth", _google_factory)
