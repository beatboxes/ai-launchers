"""HTTP transport (stdlib ``http.client``). FROZEN INTERFACE (DESIGN §8.1); internals: a thread-safe
keep-alive pool keyed by (scheme, host, port, proxy), one retry on a stale reused connection, CONNECT
tunnelling, NO_PROXY. Signatures:

``HttpClient(timeout=600.0, connect_timeout=30.0, user_agent=None, ssl_context=None, environ=None)``
    ``.request(method, url, headers=None, body=None, stream=True, timeout=None) -> Response``
        Never raises on HTTP status. Raises ``TransportError`` on DNS/connect/TLS/proxy/read
        failures. ``stream=False`` reads the whole body before returning. Defaults ``Host``,
        ``User-Agent``, ``Accept-Encoding: identity`` and ``Content-Length``. Honours HTTPS_PROXY /
        HTTP_PROXY / ALL_PROXY (+ lowercase; https via CONNECT, http via absolute-URI requests,
        ``Proxy-Authorization: Basic`` from the proxy URL) and NO_PROXY; loopback is never proxied.
        A connection returns to the pool only after its body was fully read (``read()`` /
        ``iter_lines()`` to EOF) and the server did not ask to close it; a request on a REUSED
        connection failing before any response byte is retried once on a fresh connection.
    ``.close()`` closes idle pooled connections.
``Response``: ``status`` int, ``reason`` str, ``headers`` (``CaseInsensitiveDict``), ``url``,
    ``read() -> bytes`` (whole remaining body, cached), ``text()``, ``json()``,
    ``iter_lines() -> Iterator[bytes]`` (line terminators stripped; b"" for blank lines),
    ``close()`` (idempotent, unblocks a reader on another thread; an unfinished body discards the
    connection), context manager.
``iter_sse(source) -> Iterator[SSEEvent]`` — ``source`` = Response or iterable of lines
    (bytes/str). Multi-line data joined with "\\n", comments ignored, CRLF tolerated, BOM stripped,
    event defaults to "message", trailing event without blank line still dispatched.
``iter_ndjson(source) -> Iterator[Any]`` — one JSON value per non-blank line (ValueError if bad).
``iter_with_heartbeat(iterable, interval, on_close=None) -> Iterator[item | HEARTBEAT]`` — reads
    ``iterable`` on a daemon thread; yields ``HEARTBEAT`` whenever nothing arrived for ``interval``
    seconds; re-raises the reader's exception in the consumer; on early close sets a stop flag and
    calls ``on_close`` (e.g. ``response.close``) so the reader thread unblocks.
``make_ssl_context(cafile=None, environ=None) -> ssl.SSLContext`` — system trust store PLUS
    ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` / ``CURL_CA_BUNDLE`` / ``SSL_CERT_DIR`` when set.
``proxy_for_url(url, environ=None) -> Optional[str]``, ``bypass_proxy(host, port, no_proxy) -> bool``.
"""

import base64
import http.client
import ipaddress
import json
import os
import queue
import select
import socket
import ssl
import threading
import time
from urllib.parse import unquote, urlsplit

__all__ = [
    "TransportError", "CaseInsensitiveDict", "Response", "HttpClient", "SSEEvent", "iter_sse",
    "iter_ndjson", "iter_with_heartbeat", "HEARTBEAT", "make_ssl_context", "proxy_for_url",
    "bypass_proxy", "DEFAULT_USER_AGENT",
]

DEFAULT_USER_AGENT = "ai-gateway/1.0.0"


class TransportError(Exception):
    """Connection-level failure (no HTTP status). ``url`` never contains credentials."""

    def __init__(self, message, url=None, cause=None):
        Exception.__init__(self, message)
        self.message = message
        self.url = url
        self.cause = cause


class CaseInsensitiveDict(object):
    """Minimal case-insensitive mapping preserving the original key spelling."""

    __slots__ = ("_d",)

    def __init__(self, data=None):
        self._d = {}
        if data:
            items = data.items() if hasattr(data, "items") else data
            for k, v in items:
                self[k] = v

    def __setitem__(self, key, value):
        self._d[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self._d[key.lower()][1]

    def __delitem__(self, key):
        del self._d[key.lower()]

    def __contains__(self, key):
        return isinstance(key, str) and key.lower() in self._d

    def __iter__(self):
        return (k for k, _ in self._d.values())

    def __len__(self):
        return len(self._d)

    def get(self, key, default=None):
        v = self._d.get(key.lower())
        return default if v is None else v[1]

    def items(self):
        return [(k, v) for k, v in self._d.values()]

    def keys(self):
        return [k for k, _ in self._d.values()]

    def values(self):
        return [v for _, v in self._d.values()]

    def copy(self):
        return CaseInsensitiveDict(self.items())

    def to_dict(self):
        return dict(self.items())

    def __eq__(self, other):
        if isinstance(other, CaseInsensitiveDict):
            other = other.to_dict()
        if not hasattr(other, "items"):
            return NotImplemented
        return {k.lower(): v for k, v in self.items()} == {k.lower(): v for k, v in other.items()}

    def __repr__(self):
        return "CaseInsensitiveDict(%r)" % (self.to_dict(),)


# ---------------------------------------------------------------------------------------
# TLS / proxies
# ---------------------------------------------------------------------------------------

_CA_ENV = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
_ssl_cache = {}
_ssl_cache_lock = threading.Lock()


def make_ssl_context(cafile=None, environ=None):
    """Default verifying context + extra CA bundles from the environment (cached per inputs)."""
    env = os.environ if environ is None else environ
    files = []
    for f in [cafile] + [env.get(k) for k in _CA_ENV]:
        if f and f not in files and os.path.isfile(f):
            files.append(f)
    capath = env.get("SSL_CERT_DIR")
    capath = capath if capath and os.path.isdir(capath) else None
    key = (tuple(files), capath)
    with _ssl_cache_lock:
        ctx = _ssl_cache.get(key)
        if ctx is None:
            ctx = ssl.create_default_context()
            for f in files:
                ctx.load_verify_locations(cafile=f)
            if capath:
                ctx.load_verify_locations(capath=capath)
            _ssl_cache[key] = ctx
        return ctx


def _strip_brackets(host):
    return host[1:-1] if host.startswith("[") and host.endswith("]") else host


def _is_loopback(host):
    host = _strip_brackets(host or "").lower()
    if host in ("localhost", "localhost.localdomain") or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def bypass_proxy(host, port, no_proxy):
    """curl-style NO_PROXY matching: ``*``, exact host, domain suffix (``.x.com``, ``x.com``,
    ``*.x.com``), IP literal, CIDR (``10.0.0.0/8``), optional ``:port``."""
    if not no_proxy:
        return False
    host = _strip_brackets((host or "").lower()).rstrip(".")
    try:
        host_ip = ipaddress.ip_address(host)
    except ValueError:
        host_ip = None
    for entry in no_proxy.replace(";", ",").replace(" ", ",").split(","):
        e = entry.strip().lower()
        if not e:
            continue
        if e == "*":
            return True
        eport = None
        if e.startswith("["):  # [v6]:port
            close = e.find("]")
            if close > 0:
                rest = e[close + 1:]
                e, eport = e[1:close], (rest[1:] if rest.startswith(":") else None)
        elif e.count(":") == 1:
            e, eport = e.split(":", 1)
        if eport and port is not None and str(port) != eport:
            continue
        if "/" in e and host_ip is not None:
            try:
                if host_ip in ipaddress.ip_network(e, strict=False):
                    return True
            except ValueError:
                pass
            continue
        if host_ip is not None:
            try:
                if host_ip == ipaddress.ip_address(e):
                    return True
            except ValueError:
                pass
            continue
        e = e.lstrip("*").lstrip(".").rstrip(".")
        if e and (host == e or host.endswith("." + e)):
            return True
    return False


def proxy_for_url(url, environ=None):
    """Proxy URL to use for ``url`` or None (direct). Only http(s) proxies are supported."""
    env = os.environ if environ is None else environ
    parts = urlsplit(url)
    host = parts.hostname or ""
    if _is_loopback(host):
        return None
    no_proxy = env.get("no_proxy") or env.get("NO_PROXY") or ""
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if bypass_proxy(host, port, no_proxy):
        return None
    names = ("https_proxy", "HTTPS_PROXY") if parts.scheme == "https" else ("http_proxy", "HTTP_PROXY")
    for name in names + ("all_proxy", "ALL_PROXY"):
        v = env.get(name)
        if v:
            if "://" not in v:
                v = "http://" + v
            if urlsplit(v).scheme in ("http", "https"):
                return v
            return None  # socks etc. unsupported -> direct
    return None


def _safe_url(url):
    """URL without userinfo/query (for error messages)."""
    try:
        p = urlsplit(url)
        host = p.hostname or ""
        if p.port:
            host += ":%d" % p.port
        return "%s://%s%s" % (p.scheme, host, p.path)
    except Exception:
        return "<url>"


# ---------------------------------------------------------------------------------------
# Response / client
# ---------------------------------------------------------------------------------------

def _strip_eol(line):
    if line.endswith(b"\r\n"):
        return line[:-2]
    if line.endswith(b"\n") or line.endswith(b"\r"):
        return line[:-1]
    return line


class Response(object):
    """One HTTP response. ``conn`` is released to the client's pool (``release(conn)``) once the
    body has been read completely and the server allows reuse; otherwise it is closed."""

    def __init__(self, status, reason, headers, raw, conn, url, body=None, release=None):
        self.status = int(status)
        self.reason = reason or ""
        self.headers = headers if isinstance(headers, CaseInsensitiveDict) else CaseInsensitiveDict(headers)
        self.url = url
        self._raw = raw
        self._conn = conn
        self._body = body
        self._release = release
        self._closed = raw is None
        self._lock = threading.Lock()

    @property
    def closed(self):
        return self._closed

    def _finish(self, complete):
        """Body done: return the connection to the pool when ``complete`` and reusable, else close it."""
        with self._lock:
            conn, self._conn = self._conn, None
            self._closed = True
        raw = self._raw
        reusable = bool(complete and conn is not None and self._release is not None and raw is not None
                        and not raw.will_close)
        if raw is not None:
            try:
                raw.close()  # closes the response's file object only; the socket stays with ``conn``
            except Exception:
                pass
        if conn is None:
            return
        if reusable:
            self._release(conn)
        else:
            conn.close()

    def read(self):
        """The whole (remaining) body. Subsequent calls return the same bytes."""
        if self._body is not None:
            return self._body
        if self._raw is None:
            self._body = b""
            return self._body
        try:
            body = self._raw.read()
        except (OSError, http.client.HTTPException, ValueError) as exc:
            self._finish(False)
            raise TransportError("error reading response body: %s" % exc, _safe_url(self.url), exc)
        self._body = body
        self._finish(True)
        return body

    def text(self, encoding="utf-8"):
        return self.read().decode(encoding, "replace")

    def json(self):
        return json.loads(self.read().decode("utf-8"))

    def iter_lines(self):
        """Yield body lines without terminators. Releases/closes the connection at EOF or on error."""
        if self._body is not None:
            for line in self._body.splitlines():
                yield line
            return
        raw = self._raw
        if raw is None:
            return
        complete = False
        try:
            while True:
                try:
                    line = raw.readline()
                except (OSError, http.client.HTTPException, ValueError, AttributeError) as exc:
                    if self._closed:
                        return
                    raise TransportError("error reading response stream: %s" % exc, _safe_url(self.url), exc)
                if not line:
                    if self._closed:
                        return
                    if not raw.chunked and raw.length:  # Content-Length body cut short
                        raise TransportError("connection closed before the full response body was received",
                                             _safe_url(self.url))
                    complete = True
                    return
                yield _strip_eol(line)
        finally:
            if complete:
                self._finish(True)
            else:
                self.close()

    def close(self):
        """Idempotent and thread-safe; unblocks a reader on another thread. An unfinished body
        discards the connection (it is never returned to the pool)."""
        with self._lock:
            if self._closed and self._conn is None:
                return
            self._closed = True
            conn, self._conn = self._conn, None
        sock = getattr(conn, "sock", None) if conn is not None else None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            if self._raw is not None:
                self._raw.close()
        except Exception:
            pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return "Response(%d %s %s)" % (self.status, self.reason, _safe_url(self.url))


def _readable(sock):
    """True if ``sock`` has data or EOF/error pending (an idle keep-alive socket must not)."""
    if hasattr(select, "poll"):
        p = select.poll()
        p.register(sock, select.POLLIN | select.POLLPRI | select.POLLERR | select.POLLHUP)
        return bool(p.poll(0))
    r, _, x = select.select([sock], [], [sock], 0)  # pragma: no cover - Windows
    return bool(r or x)


class _Conn(object):
    """A pooled connection: the ``http.client`` connection plus routing data. Owned by exactly one
    request (or the idle pool) at a time."""

    __slots__ = ("http", "sock", "key", "absolute", "proxy_headers", "idle_since")

    def __init__(self, http_conn, key, absolute=False, proxy_headers=None):
        self.http = http_conn
        self.sock = http_conn.sock  # kept: http.client drops it when a response "will close"
        self.key = key
        self.absolute = absolute
        self.proxy_headers = proxy_headers or {}
        self.idle_since = 0.0

    def usable(self, idle_timeout):
        sock = self.http.sock
        if sock is None or sock is not self.sock or time.monotonic() - self.idle_since > idle_timeout:
            return False
        try:
            if sock.fileno() < 0:
                return False
            pending = getattr(sock, "pending", None)  # buffered TLS bytes on an idle conn = garbage
            if pending is not None and pending():
                return False
            return not _readable(sock)
        except (OSError, ValueError):
            return False

    def close(self):
        sock, self.sock = self.sock, None
        try:
            self.http.close()
        except Exception:
            pass
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


#: errors that mean a REUSED keep-alive connection was already dead before any response byte
_STALE_ERRORS = (http.client.RemoteDisconnected, http.client.ImproperConnectionState, ConnectionResetError,
                 ConnectionAbortedError, BrokenPipeError, ssl.SSLZeroReturnError, ssl.SSLEOFError)


def _idna(host):
    try:
        host.encode("ascii")
        return host
    except UnicodeEncodeError:
        return host.encode("idna").decode("ascii")


def _host_header(scheme, host, port):
    h = _idna(host)
    if ":" in h:
        h = "[%s]" % h
    return h if port == (443 if scheme == "https" else 80) else "%s:%d" % (h, port)


class HttpClient(object):
    """Thread-safe HTTP/1.1 client with a keep-alive pool keyed by (scheme, host, port, proxy).

    A connection is checked out exclusively per request and returns to the pool only after its
    response body was read completely and the server did not ask to close it. A request on a
    REUSED connection that fails before any response byte (stale keep-alive) is retried once on a
    fresh connection; nothing else is ever retried. https goes through proxies via CONNECT
    (``Proxy-Authorization: Basic`` from the proxy URL's userinfo); plain http via absolute-URI
    requests. ``connect_timeout`` covers TCP connect + proxy CONNECT + TLS handshake; ``timeout``
    (or the per-request ``timeout``) every subsequent socket read/write.
    """

    #: idle connections kept per route; older idle connections are dropped after ``idle_timeout`` s
    max_idle_per_route = 8
    idle_timeout = 55.0
    #: target schemes tunnelled through a proxy with CONNECT (others use absolute-URI requests)
    _tunnel_schemes = ("https",)

    def __init__(self, timeout=600.0, connect_timeout=30.0, user_agent=None, ssl_context=None, environ=None):
        self.timeout = float(timeout)
        self.connect_timeout = float(connect_timeout)
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self._ssl_context = ssl_context
        self._environ = environ
        self._pool = {}  # key -> [_Conn] (LIFO)
        self._pool_lock = threading.Lock()
        self._closed = False

    def _ssl(self):
        if self._ssl_context is None:
            self._ssl_context = make_ssl_context(environ=self._environ)
        return self._ssl_context

    def proxy_for(self, url):
        return proxy_for_url(url, self._environ)

    # ---- connections -------------------------------------------------------------------
    def _route(self, parts, url):
        scheme = (parts.scheme or "").lower()
        if scheme not in ("http", "https"):
            raise TransportError("unsupported URL scheme %r" % scheme, _safe_url(url))
        host = parts.hostname
        if not host:
            raise TransportError("URL has no host", _safe_url(url))
        try:
            port = parts.port or (443 if scheme == "https" else 80)
        except ValueError:
            raise TransportError("invalid port in URL", _safe_url(url))
        return (scheme, host, port, self.proxy_for(url))

    def _new_http(self, scheme, host, port, timeout):
        if scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=timeout, context=self._ssl())
        return http.client.HTTPConnection(host, port, timeout=timeout)

    def _open(self, key, timeout):
        scheme, host, port, proxy = key
        absolute, proxy_headers = False, {}
        if proxy:
            pp = urlsplit(proxy)
            try:
                phost, pport = pp.hostname, pp.port or 80
            except ValueError:
                phost, pport = None, None
            if pp.scheme != "http" or not phost:
                raise TransportError("unsupported proxy URL (only http://host:port proxies are supported)")
            if pp.username is not None:
                cred = "%s:%s" % (unquote(pp.username), unquote(pp.password or ""))
                proxy_headers["Proxy-Authorization"] = "Basic " + base64.b64encode(cred.encode("utf-8")).decode("ascii")
            if scheme in self._tunnel_schemes:
                conn = self._new_http(scheme, phost, pport, timeout)
                tunnel_headers = {"Host": "%s:%d" % (_host_header("http", host, 80), port)}
                tunnel_headers.update(proxy_headers)
                conn.set_tunnel(_idna(host), port, headers=tunnel_headers)
                proxy_headers = {}
            else:
                conn = http.client.HTTPConnection(phost, pport, timeout=timeout)
                absolute = True
        else:
            conn = self._new_http(scheme, host, port, timeout)
        try:
            conn.connect()
        except BaseException:
            conn.close()
            raise
        return _Conn(conn, key, absolute, proxy_headers)

    def _connect(self, key, timeout, safe):
        try:
            return self._open(key, timeout)
        except TransportError:
            raise
        except socket.timeout as exc:
            raise TransportError("timed out connecting to %s" % safe, safe, exc)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            via = " via proxy" if key[3] else ""
            raise TransportError("connection to %s%s failed: %s" % (safe, via, exc), safe, exc)

    def _checkout(self, key):
        """An idle, still-healthy pooled connection for ``key`` or None."""
        while True:
            with self._pool_lock:
                idle = self._pool.get(key)
                conn = idle.pop() if idle else None
            if conn is None:
                return None
            if conn.usable(self.idle_timeout):
                return conn
            conn.close()

    def _release(self, conn):
        conn.idle_since = time.monotonic()
        with self._pool_lock:
            if not self._closed:
                idle = self._pool.setdefault(conn.key, [])
                if len(idle) < self.max_idle_per_route:
                    idle.append(conn)
                    return
        conn.close()

    def _idle_count(self):
        with self._pool_lock:
            return sum(len(v) for v in self._pool.values())

    @staticmethod
    def _send(conn, method, target, headers, body, read_timeout):
        c = conn.http
        if conn.sock is not None:
            conn.sock.settimeout(read_timeout)
        c.putrequest(method, target, skip_host=True, skip_accept_encoding=True)
        for k, v in headers.items():
            c.putheader(k, v)
        c.endheaders(body)
        return c.getresponse()

    # ---- public ------------------------------------------------------------------------
    def request(self, method, url, headers=None, body=None, stream=True, timeout=None):
        """Send one request. See module docstring for the contract."""
        if isinstance(body, str):
            body = body.encode("utf-8")
        method = method.upper()
        parts = urlsplit(url)
        safe = _safe_url(url)
        key = self._route(parts, url)
        scheme, host, port = key[0], key[1], key[2]
        read_timeout = self.timeout if timeout is None else float(timeout)
        connect_timeout = min(self.connect_timeout, read_timeout)
        hdrs = CaseInsensitiveDict(headers or {})
        if "host" not in hdrs:
            hdrs["Host"] = _host_header(scheme, host, port)
        if "user-agent" not in hdrs:
            hdrs["User-Agent"] = self.user_agent
        if "accept-encoding" not in hdrs:
            hdrs["Accept-Encoding"] = "identity"
        if "content-length" not in hdrs and "transfer-encoding" not in hdrs:
            if body is not None:
                hdrs["Content-Length"] = str(len(body))
            elif method in ("POST", "PUT", "PATCH"):
                hdrs["Content-Length"] = "0"
        keep_alive = "close" not in hdrs.get("connection", "").lower()
        path = (parts.path or "/") + (("?" + parts.query) if parts.query else "")
        for attempt in (0, 1):
            pooled = self._checkout(key) if keep_alive and attempt == 0 else None
            conn = pooled or self._connect(key, connect_timeout, safe)
            target, send_headers = path, hdrs
            if conn.absolute:
                target = "%s://%s%s" % (scheme, _host_header(scheme, host, port), path)
                if conn.proxy_headers:
                    send_headers = hdrs.copy()
                    for k, v in conn.proxy_headers.items():
                        send_headers[k] = v
            try:
                raw = self._send(conn, method, target, send_headers, body, read_timeout)
                break
            except (OSError, http.client.HTTPException, ValueError) as exc:
                conn.close()
                if pooled is not None and isinstance(exc, _STALE_ERRORS):
                    continue  # stale keep-alive connection: one retry on a fresh connection
                if isinstance(exc, socket.timeout):
                    raise TransportError("timed out talking to %s" % safe, safe, exc)
                raise TransportError("connection to %s failed: %s" % (safe, exc), safe, exc)
        rheaders = CaseInsensitiveDict()
        for k, v in raw.getheaders():
            if k in rheaders:
                rheaders[k] = rheaders[k] + ", " + v
            else:
                rheaders[k] = v
        resp = Response(raw.status, raw.reason, rheaders, raw, conn, url,
                        release=self._release if keep_alive else None)
        if not stream or method == "HEAD":
            resp.read()
        return resp

    def close(self):
        """Close every idle pooled connection. The client stays usable (without pooling)."""
        with self._pool_lock:
            self._closed = True
            conns = [c for idle in self._pool.values() for c in idle]
            self._pool.clear()
        for c in conns:
            c.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------
# stream parsers
# ---------------------------------------------------------------------------------------

class SSEEvent(object):
    __slots__ = ("event", "data", "id", "retry")

    def __init__(self, event="message", data="", id=None, retry=None):  # noqa: A002
        self.event = event
        self.data = data
        self.id = id
        self.retry = retry

    def json(self):
        return json.loads(self.data)

    def __eq__(self, other):
        return isinstance(other, SSEEvent) and (self.event, self.data, self.id, self.retry) == (
            other.event, other.data, other.id, other.retry)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "SSEEvent(event=%r, data=%r, id=%r)" % (self.event, self.data[:200], self.id)


def _lines_of(source):
    if hasattr(source, "iter_lines"):
        return source.iter_lines()
    return source


def iter_sse(source):
    event, data, last_id, retry = None, [], None, None
    first = True
    for raw in _lines_of(source):
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
        if line.endswith("\r\n"):
            line = line[:-2]
        elif line.endswith("\n") or line.endswith("\r"):
            line = line[:-1]
        if first:
            first = False
            if line.startswith("\ufeff"):
                line = line[1:]
        if line == "":
            if data or event is not None:
                yield SSEEvent(event or "message", "\n".join(data), last_id, retry)
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data.append(value)
        elif field == "event":
            event = value
        elif field == "id":
            if "\x00" not in value:
                last_id = value
        elif field == "retry":
            if value.isdigit():
                retry = int(value)
    if data:
        yield SSEEvent(event or "message", "\n".join(data), last_id, retry)


def iter_ndjson(source):
    for raw in _lines_of(source):
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except ValueError:
            raise ValueError("invalid NDJSON line: %r" % (line[:200],))


class _Heartbeat(object):
    __slots__ = ()

    def __repr__(self):
        return "HEARTBEAT"


HEARTBEAT = _Heartbeat()
_ITEM, _END, _ERR = 0, 1, 2


def iter_with_heartbeat(iterable, interval, on_close=None, max_queue=1024):
    """See module docstring. ``interval`` <= 0 disables heartbeats (plain pass-through thread)."""
    q = queue.Queue(maxsize=max_queue)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def reader():
        try:
            for item in iterable:
                if not put((_ITEM, item)):
                    return
        except BaseException as exc:  # propagate everything (incl. TransportError) to the consumer
            put((_ERR, exc))
        else:
            put((_END, None))

    t = threading.Thread(target=reader, name="gw-upstream-reader", daemon=True)
    t.start()
    wait = interval if interval and interval > 0 else None
    try:
        while True:
            try:
                kind, value = q.get(timeout=wait)
            except queue.Empty:
                yield HEARTBEAT
                continue
            if kind == _ITEM:
                yield value
            elif kind == _END:
                return
            else:
                raise value
    finally:
        stop.set()
        if on_close is not None:
            try:
                on_close()
            except Exception:
                pass
