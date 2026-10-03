"""Dialect execution contract (DESIGN "Interfaces (frozen)").

A ``Dialect`` turns a ``RequestContext`` into an iterator of internal events (``events.py``):

* Failures BEFORE the first yielded event raise ``errors.GatewayError`` (the server then sends a
  real HTTP error status; nothing has been committed to Claude Code yet). Sticky fallbacks (grok
  proxy -> api.x.ai, Gemini schema fallback, Ollama passthrough -> chat) happen here, transparently.
* After the first yield, failures are YIELDED as ``events.StreamError`` (use
  ``GatewayError.to_stream_error()``), then the generator returns.
* The generator must release upstream resources in ``finally`` (``response.close()``); the server
  may close it early (client disconnect) — honour ``GeneratorExit``.
* A dialect never writes to the client and never sleeps/retries on 429/5xx (Claude Code retries);
  it only retries once after a 401-triggered credential refresh (``send_with_auth_retry``).

``ProviderRuntime`` holds per-provider sticky state shared by all requests in this process.
Well-known keys (use these names):
  ``"grok_fallback"``            bool   grok_cli_proxy failed -> use ``provider.fallback`` for the process
  ``"gemini_schema_fallback"``   bool   Gemini rejected ``parametersJsonSchema`` -> ``parameters``+openapi
  ``"ollama_caps"``              dict   model id -> {"tools": bool, "thinking": bool, "ts": epoch}
  ``"ollama_chat_fallback"``     bool   Ollama < 0.14 (404 on /v1/messages) -> openai_chat
  ``"gemini_sig_lru"``           LRU    tool_use id -> thoughtSignature (4096 entries)
  ``"versions"``                 dict   cached CLI versions (codex/grok)
"""

import collections
import logging
import threading

__all__ = ["ProviderRuntime", "LRU", "RequestContext", "Dialect", "send_with_auth_retry", "join_url"]

LOG = logging.getLogger("ai_gateway")


class LRU(object):
    """Thread-safe bounded mapping (least-recently-used eviction)."""

    def __init__(self, capacity=4096):
        self.capacity = int(capacity)
        self._d = collections.OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            if key not in self._d:
                return default
            self._d.move_to_end(key)
            return self._d[key]

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

    def __contains__(self, key):
        with self._lock:
            return key in self._d

    def __len__(self):
        with self._lock:
            return len(self._d)


class ProviderRuntime(object):
    """Per-provider, per-process mutable state. All access through ``lock`` or the helpers."""

    def __init__(self, provider_id):
        self.provider_id = provider_id
        self.lock = threading.RLock()
        self.state = {}  # Dict[str, Any]
        self._logged = set()

    def get(self, key, default=None):
        with self.lock:
            return self.state.get(key, default)

    def set(self, key, value):
        with self.lock:
            self.state[key] = value

    def setdefault(self, key, factory):
        """Return ``state[key]``, creating it with ``factory()`` first if missing."""
        with self.lock:
            if key not in self.state:
                self.state[key] = factory()
            return self.state[key]

    def log_once(self, log, key, msg, *args):
        """Log ``msg`` at WARNING once per (provider, key) for the process lifetime."""
        with self.lock:
            if key in self._logged:
                return
            self._logged.add(key)
        (log or LOG).warning(msg, *args)


class RequestContext(object):
    """Everything a dialect needs for one request.

    ``req``             model.NormalizedRequest
    ``resolution``      router.Resolution (provider, model, model_spec, requested, role, background)
    ``provider``        config.ProviderSpec actually used (may be a fallback spec)
    ``model``           config.ModelSpec (catalog entry or default) — ``model.id`` is the UPSTREAM id
    ``runtime``         ProviderRuntime of the (parent) provider
    ``auth``            auth.AuthProvider
    ``http``            transport.HttpClient
    ``session_id``      Claude Code session id (X-Claude-Code-Session-Id) or a per-request uuid
    ``log``             logging.Logger (never log secrets / full bodies above DEBUG)
    ``requested_model`` the model string Claude Code sent (echoed in message_start)
    ``est_tokens``      local input-token estimate (anthropic_out.estimate_tokens) or None
    ``tracer``          optional callable(event_name, dict) for AI_GATEWAY_TRACE_FILE
    ``secrets``         config.SecretStore (only for dialects that must read one, e.g. passthrough)
    """

    __slots__ = ("req", "resolution", "provider", "model", "runtime", "auth", "http", "session_id", "log",
                 "requested_model", "est_tokens", "tracer", "secrets")

    def __init__(self, req, resolution, provider, model, runtime, auth, http, session_id, log=None,
                 requested_model=None, est_tokens=None, tracer=None, secrets=None):
        self.req = req
        self.resolution = resolution
        self.provider = provider
        self.model = model
        self.runtime = runtime
        self.auth = auth
        self.http = http
        self.session_id = session_id
        self.log = log or LOG
        self.requested_model = requested_model if requested_model is not None else getattr(req, "model", None)
        self.est_tokens = est_tokens
        self.tracer = tracer
        self.secrets = secrets

    @property
    def background(self):
        return bool(getattr(self.resolution, "background", False))

    def trace(self, name, **fields):
        if self.tracer is not None:
            try:
                self.tracer(name, fields)
            except Exception:
                pass

    def with_provider(self, provider):
        """Shallow copy targeting another ProviderSpec (sticky fallbacks). Runtime/auth are kept."""
        c = RequestContext(self.req, self.resolution, provider, self.model, self.runtime, self.auth, self.http,
                           self.session_id, self.log, self.requested_model, self.est_tokens, self.tracer,
                           self.secrets)
        return c


class Dialect(object):
    name = ""

    def execute(self, ctx):
        """-> Iterator[events.Event]. See module docstring for the error contract."""
        raise NotImplementedError

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.name)


def join_url(base, path):
    """``base`` + ``path`` with exactly one slash between them."""
    if not path:
        return base
    return base.rstrip("/") + "/" + path.lstrip("/")


def _auth_headers(ctx):
    from .. import errors
    from ..auth.base import AuthError

    try:
        return ctx.auth.headers()
    except AuthError as exc:
        hint = exc.hint or (ctx.auth.relogin_hint() if hasattr(ctx.auth, "relogin_hint") else None)
        raise errors.GatewayError(401, "authentication_error",
                                  "[%s/%s] %s%s" % (ctx.provider.id, ctx.model.id, exc.message,
                                                    (" — " + hint) if hint else ""), False)


def send_with_auth_retry(ctx, method, url, headers, body, stream=True, unauthorized=None, timeout=None):
    """Send a request with provider auth; return a 2xx ``Response`` or raise ``GatewayError``.

    * merges ``ctx.auth.headers()`` over ``headers`` (auth wins); ``AuthError`` -> 401
      ``authentication_error`` GatewayError;
    * ``TransportError`` -> ``errors.map_upstream_error(None, ...)`` (502 api_error, retryable,
      ``connection_error=True``);
    * HTTP 401 — or any non-2xx for which ``unauthorized(status, body_text)`` returns True (e.g.
      ChatGPT 403 ``token_expired``) — calls ``ctx.auth.on_unauthorized(used_headers)``; on True the
      request is re-sent ONCE with fresh headers;
    * any remaining status >= 400 -> the body is read and ``errors.map_upstream_error(status, text,
      headers, provider.id, model.id, context_window=model.context, est_tokens=ctx.est_tokens,
      auth_hint=auth.relogin_hint())`` is raised. The GatewayError carries ``upstream_status`` /
      ``upstream_body`` so callers can implement sticky fallbacks by catching it.
    """
    from .. import errors
    from ..transport import TransportError

    def attempt(auth_hdrs):
        h = dict(headers or {})
        h.update(auth_hdrs)
        try:
            return ctx.http.request(method, url, h, body, stream=stream, timeout=timeout)
        except TransportError as exc:
            raise errors.map_upstream_error(None, exc.message, {}, ctx.provider.id, ctx.model.id,
                                            ctx.model.context, ctx.est_tokens)

    def hint():
        try:
            return ctx.auth.relogin_hint()
        except Exception:
            return None

    used = _auth_headers(ctx)
    resp = attempt(used)
    if resp.status >= 400:
        text = resp.text()
        is_unauth = resp.status == 401 or (unauthorized is not None and unauthorized(resp.status, text))
        if is_unauth:
            refreshed = False
            try:
                refreshed = bool(ctx.auth.on_unauthorized(used))
            except Exception as exc:  # never leak auth internals; treat as not refreshed
                ctx.log.warning("[%s] credential refresh failed: %s", ctx.provider.id, type(exc).__name__)
            if refreshed:
                ctx.trace("auth_retry", provider=ctx.provider.id)
                resp = attempt(_auth_headers(ctx))
                if resp.status < 400:
                    return resp
                text = resp.text()
            status = 401 if resp.status == 401 or (unauthorized is not None and unauthorized(resp.status, text)) \
                else resp.status
            raise errors.map_upstream_error(status, text, resp.headers, ctx.provider.id, ctx.model.id,
                                            ctx.model.context, ctx.est_tokens, auth_hint=hint())
        raise errors.map_upstream_error(resp.status, text, resp.headers, ctx.provider.id, ctx.model.id,
                                        ctx.model.context, ctx.est_tokens, auth_hint=hint())
    return resp
