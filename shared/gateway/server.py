"""The gateway HTTP server (DESIGN §2.2, §5.3, §8.2).

``Gateway(table, secrets, host="127.0.0.1", port=0, token=None, log=None, trace_file=None, http=None,
commit_timeout=20.0, heartbeat_interval=15.0, launcher_name=None)``
    ``start() -> Gateway`` (bound and serving on a daemon thread when it returns), ``stop()``,
    ``url``, ``port``, ``token``, context manager, ``auth_for(provider_id)``,
    ``runtime_for(provider_id)``, ``router``.

Endpoints (routing on the PATH only — Claude Code appends ``?beta=true``):
    ``POST /v1/messages``               stream (SSE) or ``stream:false`` (Aggregator -> JSON)
    ``POST /v1/messages/count_tokens``  ``{"input_tokens": <local estimate>}``
    ``GET  /v1/models`` (+ ``/{id}``)   picker ids (``Router.models_response``)
    ``GET  /health``                    no auth
Every other request needs ``Authorization: Bearer <token>`` or ``x-api-key: <token>``.

Per message request: parse -> route (with the token estimate) -> pre-flight context check
(``estimate > 1.1 × context`` -> prompt-too-long 400) -> local ``ok`` for ``max_tokens<=1`` probes on
chat-only/CLI routes -> ``get_dialect(provider.effective_dialect(model)).execute(ctx)``.
Streaming commits (200 + ``message_start``) on the first event or after ``commit_timeout`` seconds;
``GatewayError`` before that becomes a real HTTP error; afterwards an SSE ``error`` event
(retryable -> ``overloaded_error``). ``ping`` every ``heartbeat_interval`` seconds while upstream is
silent. A client disconnect stops the stream, closes the upstream responses and the dialect
generator. Nothing is ever written to stdout/stderr (logging goes through ``tracing``).
"""

import hmac
import json
import logging
import os
import secrets as _stdlib_secrets
import socket
import socketserver
import threading
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from . import __version__
from .anthropic_in import parse_messages_request
from .anthropic_out import Aggregator, SSEEmitter, error_response, estimate_tokens, new_message_id
from .auth import AuthError, make_auth
from .compat import json_dumps_compact
from .config import SecretStore, apply_upstream_env_overrides, describe_upstream_env_overrides
from .dialects import get_dialect
from .dialects.base import ProviderRuntime, RequestContext
from .errors import GatewayError, prompt_too_long
from .events import Finish, TextDelta, Usage
from .router import Router
from .tracing import LOGGER_NAME, Tracer, trace_file_from_env
from .transport import HEARTBEAT, HttpClient, TransportError, iter_with_heartbeat

__all__ = ["Gateway", "PREFLIGHT_FACTOR", "MAX_BODY_BYTES"]

PREFLIGHT_FACTOR = 1.1
MAX_BODY_BYTES = 256 * 1024 * 1024


class _ClientGone(Exception):
    """The client closed the connection (write/read failed)."""


def _unexpected(exc):
    """Map a non-GatewayError escaping a dialect to a GatewayError."""
    if isinstance(exc, TransportError):
        return GatewayError(502, "api_error", "upstream connection failed: %s" % exc.message, True,
                            connection_error=True)
    return GatewayError(500, "api_error", "internal gateway error (%s: %s)" % (type(exc).__name__, exc), False)


def _close_quietly(events):
    close = getattr(events, "close", None)
    if close is not None:
        try:
            close()
        except Exception:  # a misbehaving dialect generator must not break cleanup
            pass


def _guarded(events, cancel):
    """Iterate ``events`` until ``cancel`` is set; always closes ``events`` in the iterating thread."""
    try:
        for ev in events:
            if cancel.is_set():
                return
            yield ev
    finally:
        _close_quietly(events)


class _RequestHttp(object):
    """Per-request facade over the shared ``HttpClient`` that remembers upstream responses, so an
    aborted request (client gone, shutdown) can close them and unblock the dialect's reader."""

    def __init__(self, http):
        self._http = http
        self._lock = threading.Lock()
        self._responses = []
        self._cancelled = False

    def request(self, method, url, headers=None, body=None, stream=True, timeout=None):
        with self._lock:
            if self._cancelled:
                raise TransportError("request aborted (client disconnected)")
        resp = self._http.request(method, url, headers, body, stream=stream, timeout=timeout)
        with self._lock:
            cancelled = self._cancelled
            if not cancelled:
                self._responses.append(resp)
        if cancelled:
            resp.close()
            raise TransportError("request aborted (client disconnected)")
        return resp

    def cancel(self):
        with self._lock:
            self._cancelled = True
            responses, self._responses = self._responses, []
        for r in responses:
            try:
                r.close()
            except Exception:
                pass

    def close(self):
        """No-op: the shared client outlives every request (``Gateway.stop`` closes it)."""

    def __getattr__(self, name):
        return getattr(self._http, name)


# ---------------------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------------------

class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = os.name != "nt"  # Windows SO_REUSEADDR allows port hijacking
    request_queue_size = 128

    def __init__(self, address, handler, gateway):
        self.gateway = gateway
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        ThreadingHTTPServer.__init__(self, address, handler)

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # TCPServer.server_bind only: HTTPServer's version calls socket.getfqdn(), which can block on DNS.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port

    def handle_error(self, request, client_address):
        self.gateway.log.debug("connection error from %s", client_address, exc_info=True)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ai-gateway/" + __version__
    sys_version = ""
    timeout = 300  # idle keep-alive connections

    # ---- connection bookkeeping / quiet logging ------------------------------------------
    def setup(self):
        BaseHTTPRequestHandler.setup(self)
        self.server.gateway._track(self.connection, True)
        self._headers_sent = False
        self._body_consumed = False

    def finish(self):
        try:
            BaseHTTPRequestHandler.finish(self)
        finally:
            self.server.gateway._track(self.connection, False)

    def log_message(self, fmt, *args):
        try:
            text = fmt % args
        except (TypeError, ValueError):
            text = str(fmt)
        self.server.gateway.log.debug("http %s: %s", self.address_string(), text)

    def send_error(self, code, message=None, explain=None):
        """Malformed HTTP from ``http.server`` -> Anthropic-shaped JSON (never HTML)."""
        err_type = "invalid_request_error" if code < 500 else "api_error"
        body = json_dumps_compact({"type": "error", "error": {"type": err_type,
                                                              "message": message or "HTTP %d" % code}})
        self.close_connection = True
        try:
            self.send_response(code, message)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.send_header("Connection", "close")
            self.end_headers()
            if getattr(self, "command", None) != "HEAD":
                self.wfile.write(body.encode("utf-8"))
        except OSError:
            pass

    def do_GET(self):
        self._dispatch()

    do_POST = do_HEAD = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_GET

    # ---- request helpers -------------------------------------------------------------------
    def _authorized(self):
        expected = self.server.gateway._token_bytes
        candidates = []
        auth = self.headers.get("Authorization") or ""
        if auth[:7].lower() == "bearer ":
            candidates.append(auth[7:].strip())
        key = self.headers.get("x-api-key")
        if key:
            candidates.append(key.strip())
        ok = False
        for c in candidates:
            ok = hmac.compare_digest(c.encode("utf-8"), expected) or ok
        return ok

    def _has_body(self):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            return True
        try:
            return int(self.headers.get("Content-Length") or 0) > 0
        except ValueError:
            return True

    def _read_exact(self, n):
        data = self.rfile.read(n)
        if len(data) < n:
            raise _ClientGone("request body truncated")
        return data

    def _read_chunked(self, limit):
        parts, total = [], 0
        while True:
            line = self.rfile.readline(1024)
            if not line:
                raise _ClientGone("request body truncated")
            try:
                size = int(line.split(b";", 1)[0].strip(), 16)
            except ValueError:
                raise GatewayError(400, "invalid_request_error", "malformed chunked request body")
            if size == 0:
                while self.rfile.readline(65537) not in (b"\r\n", b"\n", b""):
                    pass
                return b"".join(parts)
            total += size
            if total > limit:
                self.close_connection = True
                raise GatewayError(413, "request_too_large", "request body exceeds %d bytes" % limit)
            parts.append(self._read_exact(size))
            self.rfile.readline(3)

    def _read_body(self):
        limit = self.server.gateway.max_body_bytes
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            data = self._read_chunked(limit)
        else:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self.close_connection = True
                raise GatewayError(400, "invalid_request_error", "invalid Content-Length")
            if n > limit:
                self.close_connection = True
                raise GatewayError(413, "request_too_large", "request body exceeds %d bytes" % limit)
            data = self._read_exact(n) if n > 0 else b""
        self._body_consumed = True
        enc = (self.headers.get("Content-Encoding") or "identity").strip().lower()
        if enc in ("", "identity"):
            return data
        wbits = {"gzip": 16 + zlib.MAX_WBITS, "x-gzip": 16 + zlib.MAX_WBITS, "deflate": zlib.MAX_WBITS}.get(enc)
        if wbits is None:
            raise GatewayError(400, "invalid_request_error", "unsupported Content-Encoding %r" % enc)
        try:
            d = zlib.decompressobj(wbits)
            out = d.decompress(data, limit + 1)
        except zlib.error:
            raise GatewayError(400, "invalid_request_error", "invalid %s request body" % enc)
        if len(out) > limit:
            raise GatewayError(413, "request_too_large", "request body exceeds %d bytes" % limit)
        return out

    def _json_body(self):
        raw = self._read_body()
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise GatewayError(400, "invalid_request_error", "request body is not valid JSON")

    def _send_bytes(self, status, headers, body, info):
        if self._has_body() and not self._body_consumed:
            self.close_connection = True  # unread body would desync the keep-alive stream
        info["status"] = status
        try:
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("request-id", info["rid"])
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self._headers_sent = True
            if self.command != "HEAD":
                self.wfile.write(body)
        except OSError as exc:
            raise _ClientGone(exc)

    def _send_json(self, status, obj, info):
        self._send_bytes(status, {"Content-Type": "application/json"},
                         json_dumps_compact(obj).encode("utf-8", "replace"), info)

    def _send_error(self, err, info):
        info["error"] = err.err_type
        if self._headers_sent:
            info["status"] = info.get("status") or err.status
            self.close_connection = True
            return
        status, headers, body = error_response(err)
        self._send_bytes(status, headers, body, info)

    # ---- dispatch -------------------------------------------------------------------------
    def _dispatch(self):
        gw = self.server.gateway
        self._headers_sent = False
        self._body_consumed = False
        method = self.command
        path = urlsplit(self.path).path or "/"
        if len(path) > 1:
            path = path.rstrip("/")
        info = {"rid": "req_" + os.urandom(12).hex(), "method": method, "path": path, "status": None}
        t0 = time.monotonic()
        try:
            if path == "/health" and method in ("GET", "HEAD"):
                self._send_json(200, {"status": "ok", "version": __version__}, info)
                return
            if not self._authorized():
                self.close_connection = True
                raise GatewayError(401, "authentication_error",
                                   "invalid or missing gateway token (send Authorization: Bearer <token> or x-api-key)")
            if path == "/v1/messages":
                self._require(method, "POST")
                self._messages(info)
            elif path == "/v1/messages/count_tokens":
                self._require(method, "POST")
                req = parse_messages_request(self._json_body(), self.headers)
                self._send_json(200, {"input_tokens": estimate_tokens(req)}, info)
            elif path == "/v1/models":
                self._require(method, "GET", "HEAD")
                self._send_json(200, gw.router.models_response(), info)
            elif path.startswith("/v1/models/"):
                self._require(method, "GET", "HEAD")
                model_id = unquote(path[len("/v1/models/"):])
                entry = gw.router.model_entry(model_id)
                if entry is None:
                    raise gw.router.not_found(model_id)
                self._send_json(200, entry, info)
            else:
                raise GatewayError(404, "not_found_error", "unknown endpoint %s %s" % (method, path))
        except GatewayError as err:
            self._safe_send_error(err, info)
        except _ClientGone:
            info["client_gone"] = True
            self.close_connection = True
        except Exception as exc:
            gw.log.exception("unhandled error in %s %s", method, path)
            self._safe_send_error(_unexpected(exc), info)
        finally:
            info["duration_ms"] = int((time.monotonic() - t0) * 1000)
            gw._finish_request(info)

    def _safe_send_error(self, err, info):
        try:
            self._send_error(err, info)
        except _ClientGone:
            info["client_gone"] = True
            self.close_connection = True

    def _require(self, method, *allowed):
        if method not in allowed:
            raise GatewayError(405, "invalid_request_error", "method %s not allowed (use %s)"
                               % (method, "/".join(allowed)))

    # ---- /v1/messages -------------------------------------------------------------------
    def _messages(self, info):
        gw = self.server.gateway
        body = self._json_body()
        if gw.tracer is not None and gw.tracer.bodies:
            info["request_body"] = body
        req = parse_messages_request(body, self.headers)
        info.update(requested=req.model, stream=req.stream, max_tokens=req.max_tokens)
        gw._note_dropped(req.dropped)
        est = estimate_tokens(req)
        info["est_tokens"] = est
        res = gw.router.resolve(req.model, est)
        info.update(provider=res.provider.id, model=res.model, role=res.role)
        window = res.model_spec.context
        if window and est > PREFLIGHT_FACTOR * window:
            raise prompt_too_long(est, window)
        dialect = res.provider.effective_dialect(res.model_spec)
        info["dialect"] = dialect
        if req.is_probe() and (res.provider.chat_only or dialect == "cli"):
            info["probe"] = True
            events = iter([TextDelta(0, "ok"), Usage(est, 1), Finish("end_turn")])
            http = None
        else:
            if res.background:
                req.effort = "low"
            http = _RequestHttp(gw.http)
            ctx = gw._context(req, res, est, info["rid"], http)
            events = gw._execute(dialect, ctx)
        if req.stream:
            self._stream(events, req, est, info, http)
        else:
            self._aggregate(events, req, est, info, http)

    def _aggregate(self, events, req, est, info, http):
        gw = self.server.gateway
        counts = info.setdefault("events", {})
        agg = Aggregator(req.model, new_message_id(), est)
        aborted = False
        try:
            for ev in events:
                if gw._stopping.is_set():
                    aborted = True
                    raise GatewayError(503, "overloaded_error", "gateway is shutting down", True)
                counts[ev.kind] = counts.get(ev.kind, 0) + 1
                agg.feed(ev)
                if agg.done:
                    break
            result = agg.result()
        except GatewayError:
            raise
        except Exception as exc:
            aborted = True
            gw.log.exception("dialect failure (%s)", info.get("dialect"))
            raise _unexpected(exc)
        finally:
            if aborted and http is not None:
                http.cancel()
            _close_quietly(events)
        self._send_json(200, result, info)

    def _start_sse(self, info):
        info["status"] = 200
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("request-id", info["rid"])
            self.end_headers()
        except OSError as exc:
            raise _ClientGone(exc)
        self._headers_sent = True

    def _stream(self, events, req, est, info, http):
        gw = self.server.gateway
        counts = info.setdefault("events", {})
        last = [time.monotonic()]
        gone = [False]

        def write(data):
            try:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()
            except (OSError, ValueError) as exc:
                gone[0] = True
                raise _ClientGone(exc)
            last[0] = time.monotonic()

        emitter = SSEEmitter(write, req.model, new_message_id(), est)
        t0 = time.monotonic()

        def commit(reason):
            self._start_sse(info)
            emitter.start()
            info["commit_ms"] = int((time.monotonic() - t0) * 1000)
            info["commit"] = reason

        cancel = threading.Event()
        guarded = _guarded(events, cancel)
        tick = max(0.005, min(1.0, gw.commit_timeout, gw.heartbeat_interval) / 2.0)
        upstream = iter_with_heartbeat(guarded, tick)
        aborted = False
        try:
            try:
                for item in upstream:
                    if gw._stopping.is_set():
                        aborted = True
                        if not emitter.committed:
                            raise GatewayError(503, "overloaded_error", "gateway is shutting down", True)
                        emitter.error("api_error", "gateway is shutting down")
                        break
                    if item is HEARTBEAT:
                        now = time.monotonic()
                        if not emitter.committed:
                            if now - t0 >= gw.commit_timeout:
                                commit("timeout")
                        elif now - last[0] >= gw.heartbeat_interval:
                            emitter.ping()
                            counts["ping"] = counts.get("ping", 0) + 1
                        continue
                    if not emitter.committed:
                        commit("event")
                    counts[item.kind] = counts.get(item.kind, 0) + 1
                    emitter.feed(item)
                    if emitter.done:
                        break
                if not emitter.done:
                    if not emitter.committed:
                        commit("end")
                    emitter.finish()
            except GatewayError as err:
                if not emitter.committed:
                    raise
                info["error"] = err.err_type
                se = err.to_stream_error()
                emitter.error(se.err_type, se.message, se.retryable)
            except _ClientGone:
                raise
            except Exception as exc:
                aborted = True
                gw.log.exception("dialect failure (%s)", info.get("dialect"))
                err = _unexpected(exc)
                if not emitter.committed:
                    raise err
                info["error"] = err.err_type
                se = err.to_stream_error()
                emitter.error(se.err_type, se.message, se.retryable)
        except _ClientGone:
            aborted = True
            info["client_gone"] = True
            self.close_connection = True
            raise
        finally:
            if emitter.stream_error is not None:
                info["error"] = emitter.stream_error.err_type
            info["sse_events"] = emitter.events_written
            cancel.set()
            if aborted and http is not None:
                http.cancel()
            upstream.close()
            try:
                guarded.close()
            except ValueError:  # still running on the reader thread; it sees `cancel` and closes itself
                pass
            if self._headers_sent and not gone[0]:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    self.close_connection = True
                    info["client_gone"] = True


# ---------------------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------------------

class Gateway(object):
    """In-process Anthropic-compatible gateway (see module docstring)."""

    commit_timeout = 20.0
    heartbeat_interval = 15.0
    max_body_bytes = MAX_BODY_BYTES

    def __init__(self, table, secrets=None, host="127.0.0.1", port=0, token=None, log=None, trace_file=None,
                 http=None, commit_timeout=None, heartbeat_interval=None, launcher_name=None):
        self.log = log or logging.getLogger(LOGGER_NAME)
        for line in describe_upstream_env_overrides(table):
            self.log.info("upstream override: %s", line)
        self.table = apply_upstream_env_overrides(table)
        self.secrets = secrets if secrets is not None else SecretStore()
        self.host = host or "127.0.0.1"
        self._port = int(port or 0)
        self.token = token or _stdlib_secrets.token_urlsafe(32)
        self._token_bytes = self.token.encode("utf-8")
        self._own_http = http is None
        self.http = http if http is not None else HttpClient()
        if commit_timeout is not None:
            self.commit_timeout = float(commit_timeout)
        if heartbeat_interval is not None:
            self.heartbeat_interval = float(heartbeat_interval)
        self.router = Router(self.table, launcher_name=launcher_name)
        trace_path = trace_file or trace_file_from_env()
        self.tracer = Tracer(trace_path, self.secrets) if trace_path else None
        self._lock = threading.Lock()
        self._auth = {}
        self._runtime = {}
        self._logged = set()
        self._conns = set()
        self._stopping = threading.Event()
        self._httpd = None
        self._thread = None

    # ---- lifecycle --------------------------------------------------------------------
    def start(self):
        if self._httpd is not None:
            return self
        self._stopping.clear()
        httpd = _HTTPServer((self.host, self._port), _Handler, self)
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                        name="ai-gateway-http", daemon=True)
        self._thread.start()
        self.log.info("gateway listening on %s (%d providers)", self.url, len(self.table.providers))
        return self

    def stop(self):
        httpd, self._httpd = self._httpd, None
        if httpd is None:
            return
        self._stopping.set()
        httpd.shutdown()
        httpd.server_close()
        with self._lock:
            conns = list(self._conns)
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(5)
            self._thread = None
        if self._own_http:
            self.http.close()
        self.log.info("gateway stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    @property
    def port(self):
        return self._httpd.server_address[1] if self._httpd is not None else self._port

    @property
    def url(self):
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(self.host, self.host)
        if ":" in host:
            host = "[%s]" % host
        return "http://%s:%d" % (host, self.port)

    def __repr__(self):
        return "Gateway(%s, providers=%r)" % (self.url, list(self.table.providers))

    # ---- per-provider singletons ------------------------------------------------------
    def auth_for(self, provider_id, fallback=False):
        """The provider's AuthProvider (one per provider id). ``fallback=True`` returns a separate
        instance only when the fallback's auth dict differs from the parent's."""
        spec = self.table.providers[provider_id]
        key, auth = provider_id, spec.auth
        if fallback and spec.fallback is not None and spec.fallback.auth != spec.auth:
            key, auth = provider_id + "#fallback", spec.fallback.auth
        with self._lock:
            inst = self._auth.get(key)
            if inst is None:
                inst = make_auth(auth, self.secrets, provider_id)
                self._auth[key] = inst
            return inst

    def runtime_for(self, provider_id):
        with self._lock:
            rt = self._runtime.get(provider_id)
            if rt is None:
                rt = ProviderRuntime(provider_id)
                self._runtime[provider_id] = rt
            return rt

    # ---- request internals -------------------------------------------------------------
    def _track(self, conn, add):
        with self._lock:
            if add:
                self._conns.add(conn)
            else:
                self._conns.discard(conn)

    def _note_dropped(self, dropped):
        for item in dropped or ():
            with self._lock:
                if item in self._logged:
                    continue
                self._logged.add(item)
            self.log.info("dropped unsupported request content: %s (logged once)", item)

    def _context(self, req, res, est, rid, http):
        provider = res.provider
        try:
            auth = self.auth_for(provider.id)
        except AuthError as exc:
            raise GatewayError(401, "authentication_error", "[%s] %s" % (provider.id, exc), False)
        except Exception as exc:
            raise GatewayError(500, "api_error", "[%s] cannot initialise credentials (%s: %s)"
                               % (provider.id, type(exc).__name__, exc), False)
        tracer = None
        if self.tracer is not None:
            def tracer(name, fields, _t=self.tracer):
                _t(name, dict(fields or {}, rid=rid))
        return RequestContext(req, res, provider, res.model_spec, self.runtime_for(provider.id), auth, http,
                              req.session_id or str(uuid.uuid4()), log=self.log, requested_model=req.model,
                              est_tokens=est, tracer=tracer, secrets=self.secrets)

    def _execute(self, dialect_name, ctx):
        try:
            dialect = get_dialect(dialect_name)
        except Exception as exc:
            raise GatewayError(500, "api_error", "dialect %r is unavailable (%s: %s)"
                               % (dialect_name, type(exc).__name__, exc), False)
        return dialect.execute(ctx)

    def _finish_request(self, info):
        self.log.info("%s %s -> %s %s%s %dms%s", info.get("method"), info.get("path"), info.get("status"),
                      ("%s,%s " % (info["provider"], info["model"])) if info.get("provider") else "",
                      "stream" if info.get("stream") else "", info.get("duration_ms", 0),
                      (" error=%s" % info["error"]) if info.get("error") else "")
        if self.tracer is not None:
            self.tracer("request", info)
