"""Mock upstream framework (DESIGN §7). FRAMEWORK ONLY — each dialect agent registers its
quirk-enforcing kinds in its own module ``testing/mock_<dialect>.py`` via ``register_kind``.

``MockServer(kind=None, brain=None, options=None, handler=None)``
    ThreadingHTTPServer on 127.0.0.1:0 (HTTP/1.1, keep-alive). ``start()``/``stop()``, context
    manager, ``url`` (``http://127.0.0.1:<port>``), ``port``. Records every request as a dict
    ``{"method", "path", "query" (dict of lists), "raw_path", "headers" (dict), "body_json",
    "body_raw" (str, only when not JSON)}`` in ``requests`` (thread-safe). ``state`` (dict) +
    ``lock`` for handler state; ``errors`` collects handler exceptions; ``options`` per kind.
``register_kind(name, handler_factory)``: ``handler_factory(server) -> handle(req, resp)``; called
    once per server start. ``handle`` gets a ``MockRequest`` and a ``MockResponder``.
    Unknown kinds are looked up by importing the modules in ``MOCK_MODULES`` first.
``Brain``: scripted model (turn 1 -> Bash tool call, turn 2 -> verify no sentinel -> ``DONE <sha8>``,
    background -> short text).
Helpers: ``sha8``, ``sse_event_bytes``, ``sse_data_bytes``, ``anthropic_brain_inputs``.
"""

import hashlib
import importlib
import json
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from ..transport import CaseInsensitiveDict
from . import SENTINEL

__all__ = [
    "MockServer", "MockRequest", "MockResponder", "ChunkWriter", "SSEWriter", "register_kind", "get_kind_factory",
    "REGISTRY", "MOCK_MODULES", "Brain", "BrainReply", "SENTINEL", "DEFAULT_COMMAND", "sha8", "sse_event_bytes",
    "sse_data_bytes", "anthropic_brain_inputs",
]

DEFAULT_COMMAND = "printf hello > out.txt && env"

#: modules imported (in order) to find an unregistered kind
MOCK_MODULES = ("mock_openai_chat", "mock_responses", "mock_gemini", "mock_anthropic", "mock_auth")

REGISTRY = {}  # Dict[str, Callable]
_REG_LOCK = threading.Lock()


def register_kind(name, handler_factory):
    """Register a mock kind. ``handler_factory(server) -> handle(req: MockRequest, resp: MockResponder)``."""
    with _REG_LOCK:
        REGISTRY[name] = handler_factory


def get_kind_factory(name):
    with _REG_LOCK:
        f = REGISTRY.get(name)
    if f is not None:
        return f
    for mod in MOCK_MODULES:
        try:
            importlib.import_module("." + mod, __package__)
        except ImportError:
            continue
        with _REG_LOCK:
            f = REGISTRY.get(name)
        if f is not None:
            return f
    raise KeyError("unknown mock kind %r (registered: %s)" % (name, ", ".join(sorted(REGISTRY))))


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def sha8(text):
    if isinstance(text, str):
        text = text.encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:8]


def _dump(data):
    return data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))


def sse_event_bytes(event, data):
    """``event: <event>\\ndata: <json>\\n\\n`` (``data`` str is sent verbatim)."""
    return ("event: %s\ndata: %s\n\n" % (event, _dump(data))).encode("utf-8")


def sse_data_bytes(data):
    """``data: <json>\\n\\n`` (OpenAI / Gemini style; str verbatim, e.g. ``"[DONE]"``)."""
    return ("data: %s\n\n" % _dump(data)).encode("utf-8")


# ---------------------------------------------------------------------------------------
# request / response objects
# ---------------------------------------------------------------------------------------

class MockRequest(object):
    __slots__ = ("method", "path", "query", "raw_path", "headers", "body", "json")

    def __init__(self, method, raw_path, headers, body):
        parts = urlsplit(raw_path)
        self.method = method
        self.raw_path = raw_path
        self.path = parts.path
        self.query = parse_qs(parts.query, keep_blank_values=True)
        self.headers = CaseInsensitiveDict(headers)
        self.body = body
        self.json = None
        if body:
            try:
                self.json = json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self.json = None

    def header(self, name, default=None):
        return self.headers.get(name, default)

    def bearer(self):
        v = self.headers.get("authorization") or ""
        return v[7:] if v.lower().startswith("bearer ") else None

    def to_record(self):
        rec = {"method": self.method, "path": self.path, "query": self.query, "raw_path": self.raw_path,
               "headers": self.headers.to_dict(), "body_json": self.json}
        if self.json is None and self.body:
            rec["body_raw"] = self.body.decode("utf-8", "replace")
        return rec


class ChunkWriter(object):
    """HTTP/1.1 chunked body writer. ``close()`` writes the terminator (idempotent)."""

    def __init__(self, handler):
        self._h = handler
        self.closed = False

    def write(self, data):
        if self.closed:
            raise RuntimeError("chunked body already closed")
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not data:
            return
        self._h.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self._h.wfile.flush()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._h.wfile.write(b"0\r\n\r\n")
            self._h.wfile.flush()
        except OSError:
            pass


class SSEWriter(ChunkWriter):
    def event(self, name, data):
        self.write(sse_event_bytes(name, data))

    def data(self, data):
        self.write(sse_data_bytes(data))

    def comment(self, text=""):
        self.write((": %s\n\n" % text).encode("utf-8"))

    def raw(self, data):
        self.write(data)

    def done(self):
        self.write(b"data: [DONE]\n\n")


class MockResponder(object):
    def __init__(self, handler):
        self._h = handler
        self.sent = False
        self.status = None

    def _start(self, status, content_type, headers, length=None, chunked=False):
        if self.sent:
            raise RuntimeError("response already started")
        self.sent = True
        self.status = status
        self._h.send_response(status)
        if content_type:
            self._h.send_header("Content-Type", content_type)
        for k, v in (headers or {}).items():
            self._h.send_header(k, v)
        if chunked:
            self._h.send_header("Transfer-Encoding", "chunked")
            self._h.send_header("Cache-Control", "no-cache")
        else:
            self._h.send_header("Content-Length", str(length or 0))
        self._h.end_headers()

    def send_bytes(self, status, data, content_type="application/octet-stream", headers=None):
        self._start(status, content_type, headers, length=len(data))
        if data and self._h.command != "HEAD":
            self._h.wfile.write(data)
            self._h.wfile.flush()

    def send_json(self, status, obj, headers=None):
        self.send_bytes(status, json.dumps(obj).encode("utf-8"), "application/json", headers)

    def send_text(self, status, text, content_type="text/plain; charset=utf-8", headers=None):
        self.send_bytes(status, text.encode("utf-8"), content_type, headers)

    def start_chunked(self, status=200, content_type="application/octet-stream", headers=None):
        self._start(status, content_type, headers, chunked=True)
        return ChunkWriter(self._h)

    def start_sse(self, status=200, headers=None, content_type="text/event-stream; charset=utf-8"):
        self._start(status, content_type, headers, chunked=True)
        return SSEWriter(self._h)

    def close_connection(self):
        """Abruptly drop the connection (chaos tests). Nothing more can be sent."""
        self.sent = True
        self._h.close_connection = True
        try:
            self._h.wfile.flush()
        except OSError:
            pass
        try:
            import socket

            self._h.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


# ---------------------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MockUpstream/1.0"

    def log_message(self, fmt, *args):
        pass

    def _dispatch(self):
        mock = self.server.mock  # MockServer
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        req = MockRequest(self.command, self.path, list(self.headers.items()), body)
        mock._record(req)
        resp = MockResponder(self)
        try:
            mock._handle(req, resp)
        except Exception as exc:  # recorded so tests can assert no handler errors
            mock._error(exc)
            if not resp.sent:
                try:
                    resp.send_json(500, {"error": {"message": "mock handler error: %s" % exc, "type": "mock_error"}})
                except OSError:
                    pass
            else:
                self.close_connection = True
            return
        if not resp.sent:
            resp.send_json(404, {"error": {"message": "mock: no response for %s %s" % (req.method, req.path),
                                           "type": "not_found"}})

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _dispatch


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


class MockServer(object):
    def __init__(self, kind=None, brain=None, options=None, handler=None):
        if kind is None and handler is None:
            raise ValueError("MockServer needs a kind or a handler")
        self.kind = kind
        self.brain = brain if brain is not None else Brain()
        self.options = dict(options or {})
        self.state = {}
        self.lock = threading.RLock()
        self.requests = []  # List[Dict[str, Any]]
        self.errors = []    # List[str]
        self._handler_arg = handler
        self._handle_fn = None
        self._httpd = None
        self._thread = None

    # ---- lifecycle --------------------------------------------------------------------
    def start(self):
        if self._httpd is not None:
            return self
        if self._handler_arg is not None:
            h = self._handler_arg
            self._handle_fn = lambda req, resp: h(self, req, resp)
        else:
            self._handle_fn = get_kind_factory(self.kind)(self)
        self._httpd = _Server(("127.0.0.1", 0), _Handler)
        self._httpd.mock = self
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                        name="mock-%s" % (self.kind or "custom"), daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    @property
    def port(self):
        return self._httpd.server_address[1] if self._httpd else None

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.port if self._httpd else None

    # ---- recording ---------------------------------------------------------------------
    def _record(self, req):
        with self.lock:
            self.requests.append(req.to_record())

    def _error(self, exc):
        with self.lock:
            self.errors.append("%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc()))

    def _handle(self, req, resp):
        self._handle_fn(req, resp)

    def requests_for(self, path_prefix="", method=None):
        with self.lock:
            return [r for r in self.requests
                    if r["path"].startswith(path_prefix) and (method is None or r["method"] == method)]

    def last_request(self):
        with self.lock:
            return self.requests[-1] if self.requests else None

    def clear(self):
        with self.lock:
            del self.requests[:]
            del self.errors[:]


# ---------------------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------------------

class BrainReply(object):
    __slots__ = ("kind", "tool_name", "arguments", "text", "thinking")

    def __init__(self, kind, tool_name=None, arguments=None, text=None, thinking=None):
        self.kind = kind            # "tool_call" | "text"
        self.tool_name = tool_name
        self.arguments = arguments  # dict for tool_call
        self.text = text
        self.thinking = thinking

    def __repr__(self):
        return "BrainReply(%r, tool=%r, text=%r)" % (self.kind, self.tool_name, self.text)


_SHORT_RE = re.compile(r"_[0-9a-f]{8,}$")


class Brain(object):
    """Scripted model shared by every mock kind (dialect-agnostic inputs).

    ``decide(offered_tools, tool_results, background=False) -> BrainReply``:
      * background, or no tools offered -> text ``background_text``;
      * no tool results yet -> call the Bash tool (name resolved via ``find_tool``) with
        ``{"command": command, "description": ...}``;
      * otherwise -> if any tool result contains ``SENTINEL``, record a leak and reply
        ``"LEAK <sentinel> in tool output"``; else reply ``"DONE <sha8(last tool result)>"``.
    """

    def __init__(self, command=DEFAULT_COMMAND, tool="Bash", background_text="Background reply.",
                 thinking="Planning the tool call."):
        self.command = command
        self.tool = tool
        self.background_text = background_text
        self.thinking = thinking
        self.leaks = []
        self.decisions = []
        self._lock = threading.Lock()

    def find_tool(self, offered):
        """Exact name, then case-insensitive, then a shortened form ``<prefix>_<hex8>``."""
        offered = list(offered or [])
        if self.tool in offered:
            return self.tool
        for n in offered:
            if n.lower() == self.tool.lower():
                return n
        for n in offered:
            if _SHORT_RE.search(n) and self.tool.lower().startswith(_SHORT_RE.sub("", n).lower()):
                return n
        return None

    def decide(self, offered_tools, tool_results, background=False):
        tool_results = [t if isinstance(t, str) else json.dumps(t) for t in (tool_results or [])]
        if background or not offered_tools:
            reply = BrainReply("text", text=self.background_text)
        else:
            name = self.find_tool(offered_tools)
            if name is None:
                reply = BrainReply("text", text="NO-TOOL %s not offered" % self.tool)
            elif not tool_results:
                reply = BrainReply("tool_call", tool_name=name,
                                   arguments={"command": self.command, "description": "Run the shell command"},
                                   thinking=self.thinking)
            elif any(SENTINEL in t for t in tool_results):
                with self._lock:
                    self.leaks.append(tool_results[-1])
                reply = BrainReply("text", text="LEAK %s in tool output" % SENTINEL)
            else:
                reply = BrainReply("text", text="DONE %s" % sha8(tool_results[-1]))
        with self._lock:
            self.decisions.append(reply)
        return reply


def anthropic_brain_inputs(body):
    """(offered tool names, tool result texts in order, background flag) from an Anthropic body."""
    body = body or {}
    tools = [t.get("name") for t in body.get("tools") or [] if isinstance(t, dict) and t.get("name")]
    results = []
    for m in body.get("messages") or []:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                results.append(c or "")
    background = "background" in str(body.get("model", "")).lower() or not tools
    return tools, results, background


# ---------------------------------------------------------------------------------------
# built-in kinds
# ---------------------------------------------------------------------------------------

def _echo_factory(server):
    def handle(req, resp):
        resp.send_json(200, req.to_record())
    return handle


register_kind("echo", _echo_factory)
