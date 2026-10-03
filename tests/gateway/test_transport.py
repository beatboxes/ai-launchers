"""transport internals: keep-alive pool (reuse, Connection: close, chunked/length bodies, mid-stream
close), stale-connection retry (only for reused connections, only before any response byte),
CONNECT proxy tunnelling (plain + TLS), absolute-URI http proxying, NO_PROXY, timeouts, concurrency."""

import base64
import json
import os
import shutil
import socket
import socketserver
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.parse import parse_qs, urlsplit

from ._pkg import mod

transport = mod("transport")


# ---------------------------------------------------------------------------------------
# test servers
# ---------------------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def setup(self):
        BaseHTTPRequestHandler.setup(self)
        with self.server.lock:
            self.server.connections += 1
            self.conn_id = self.server.connections
        self.on_conn = 0

    def _send(self, status, body, headers=None):
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _chunk(self, data):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _dispatch(self):
        self.on_conn += 1
        parts = urlsplit(self.path)
        q = parse_qs(parts.query)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        with self.server.lock:
            self.server.log.append({"conn": self.conn_id, "path": parts.path, "raw_path": self.path,
                                    "method": self.command, "headers": dict(self.headers.items()), "body": body})
        p = parts.path
        if p == "/len":
            self._send(200, b"x" * int(q.get("n", ["5"])[0]), {"Content-Type": "text/plain"})
        elif p == "/echo":
            self._send(200, json.dumps({"headers": dict(self.headers.items()), "body": body.decode("utf-8"),
                                        "path": self.path}).encode("utf-8"), {"Content-Type": "application/json"})
        elif p == "/chunked":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._chunk(b"line1\nli")
            self._chunk(b"ne2\n")
            self.wfile.write(b"0\r\n\r\n")
        elif p == "/close":
            self._send(200, b"bye", {"Connection": "close"})
            self.close_connection = True
        elif p == "/stream":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for i in range(50):
                    self._chunk(b"data: %d\n\n" % i)
                    time.sleep(0.1)
                self.wfile.write(b"0\r\n\r\n")
            except OSError:
                self.close_connection = True
        elif p == "/slow":
            time.sleep(float(q.get("s", ["1"])[0]))
            self._send(200, b"late")
        elif p == "/drop-reused":  # served on a fresh connection, dropped on a reused one
            if self.on_conn > 1:
                self.close_connection = True
                return
            self._send(200, b"fresh")
        elif p == "/drop":
            self.close_connection = True
        elif p == "/partial-reused":  # response bytes then a dead connection
            if self.on_conn > 1:
                self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nonly-10-by")
                self.wfile.flush()
                self.close_connection = True
                return
            self._send(200, b"ok")
        elif p == "/server-closes":  # keep-alive response, then the server closes the idle socket
            self._send(200, b"done")
            self.close_connection = True
        elif p == "/truncated":
            self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Length: 50\r\n\r\nabc\ndef\n")
            self.wfile.flush()
            self.close_connection = True
        else:
            self._send(404, b"nope")

    do_GET = do_POST = do_HEAD = _dispatch


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, handler=_Handler):
        self.lock = threading.Lock()
        self.connections = 0
        self.log = []
        ThreadingHTTPServer.__init__(self, ("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def handle_error(self, request, client_address):
        pass  # clients hanging up early (timeouts, mid-stream close) are part of the tests

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.server_address[1]

    def stop(self):
        self.shutdown()
        self.server_close()


class _ConnectProxy(socketserver.ThreadingTCPServer):
    """Minimal CONNECT proxy: records the CONNECT head and tunnels every target to ``target_port``."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, target_port, require_auth=None):
        self.target_port = target_port
        self.require_auth = require_auth
        self.connects = []
        self.lock = threading.Lock()
        socketserver.ThreadingTCPServer.__init__(self, ("127.0.0.1", 0), _ConnectHandler)
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    @property
    def port(self):
        return self.server_address[1]

    def stop(self):
        self.shutdown()
        self.server_close()


class _ConnectHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        head = b""
        while b"\r\n\r\n" not in head:
            data = sock.recv(4096)
            if not data:
                return
            head += data
        lines = head.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")
        headers = {}
        for ln in lines[1:]:
            k, _, v = ln.partition(":")
            headers[k.strip().lower()] = v.strip()
        with self.server.lock:
            self.server.connects.append({"line": lines[0], "headers": headers})
        if not lines[0].startswith("CONNECT "):
            sock.sendall(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n")
            return
        if self.server.require_auth and headers.get("proxy-authorization") != self.server.require_auth:
            sock.sendall(b"HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\n\r\n")
            return
        upstream = socket.create_connection(("127.0.0.1", self.server.target_port), timeout=10)
        sock.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")

        def pipe(src, dst):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        t = threading.Thread(target=pipe, args=(upstream, sock), daemon=True)
        t.start()
        pipe(sock, upstream)
        t.join(10)
        upstream.close()


def _client(**kw):
    kw.setdefault("timeout", 10)
    kw.setdefault("environ", {})
    return transport.HttpClient(**kw)


# ---------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------

class KeepAliveTests(unittest.TestCase):
    def setUp(self):
        self.srv = _Server()
        self.client = _client()

    def tearDown(self):
        self.client.close()
        self.srv.stop()

    def test_reuse_content_length_chunked_and_iter_lines(self):
        u = self.srv.url
        self.assertEqual(self.client.request("GET", u + "/len?n=3", stream=False).read(), b"xxx")
        self.assertEqual(self.client.request("GET", u + "/len?n=4").read(), b"xxxx")
        r = self.client.request("GET", u + "/chunked")
        self.assertEqual(list(r.iter_lines()), [b"line1", b"line2"])
        r = self.client.request("GET", u + "/len?n=2")
        self.assertEqual(list(r.iter_lines()), [b"xx"])
        r = self.client.request("POST", u + "/echo", {"Content-Type": "application/json"}, '{"a":1}')
        self.assertEqual(json.loads(r.json()["body"]), {"a": 1})
        r = self.client.request("HEAD", u + "/len?n=9")
        self.assertEqual(r.headers.get("content-length"), "9")
        self.assertEqual(r.read(), b"")
        self.assertEqual(self.srv.connections, 1, "all requests on one keep-alive connection")
        self.assertEqual(self.client._idle_count(), 1)

    def test_default_headers(self):
        r = self.client.request("POST", self.srv.url + "/echo", stream=False)
        h = {k.lower(): v for k, v in r.json()["headers"].items()}
        self.assertEqual(h["accept-encoding"], "identity")
        self.assertEqual(h["user-agent"], transport.DEFAULT_USER_AGENT)
        self.assertEqual(h["content-length"], "0")
        self.assertEqual(h["host"], "127.0.0.1:%d" % self.srv.server_address[1])
        self.assertNotIn("connection", h)
        r = self.client.request("POST", self.srv.url + "/echo", {"Accept-Encoding": "gzip", "Host": "x.test"},
                                b"abc", stream=False)
        h = {k.lower(): v for k, v in r.json()["headers"].items()}
        self.assertEqual((h["accept-encoding"], h["host"], h["content-length"]), ("gzip", "x.test", "3"))

    def test_connection_close_not_pooled(self):
        u = self.srv.url
        self.assertEqual(self.client.request("GET", u + "/close").read(), b"bye")
        self.assertEqual(self.client._idle_count(), 0)
        self.client.request("GET", u + "/len").read()
        self.assertEqual(self.srv.connections, 2)
        # caller-requested close: never pooled
        self.client.request("GET", u + "/len", {"Connection": "close"}).read()
        self.client.request("GET", u + "/len").read()
        self.assertEqual(self.srv.connections, 3, "pooled conn reused after the Connection: close request")

    def test_unread_or_closed_midstream_discards_connection(self):
        u = self.srv.url
        r = self.client.request("GET", u + "/stream")
        lines = r.iter_lines()
        self.assertEqual(next(lines), b"data: 0")
        r.close()
        self.assertTrue(r.closed)
        self.assertEqual(list(lines), [])
        self.assertEqual(self.client._idle_count(), 0, "mid-stream close must not pool the connection")
        self.assertEqual(self.client.request("GET", u + "/len").read(), b"xxxxx")
        self.assertEqual(self.srv.connections, 2)
        # generator abandoned early (GeneratorExit) -> discarded too (it had reused the /len connection)
        r = self.client.request("GET", u + "/stream")
        it = r.iter_lines()
        next(it)
        it.close()
        self.assertTrue(r.closed)
        self.assertEqual(self.client._idle_count(), 0)
        self.client.request("GET", u + "/len").read()
        self.assertEqual(self.srv.connections, 3)

    def test_close_from_other_thread_unblocks_reader(self):
        r = self.client.request("GET", self.srv.url + "/stream")
        got = []
        done = threading.Event()

        def reader():
            for line in r.iter_lines():
                got.append(line)
            done.set()

        t = threading.Thread(target=reader)
        t.start()
        time.sleep(0.25)
        r.close()
        self.assertTrue(done.wait(3), "reader must unblock on close()")
        t.join(3)
        self.assertTrue(got)
        self.assertEqual(self.client._idle_count(), 0)

    def test_server_closed_idle_connection_is_detected(self):
        u = self.srv.url
        self.assertEqual(self.client.request("GET", u + "/server-closes").read(), b"done")
        time.sleep(0.2)  # server has closed the pooled socket by now
        self.assertEqual(self.client.request("GET", u + "/len").read(), b"xxxxx")
        self.assertEqual(self.srv.connections, 2)

    def test_idle_timeout(self):
        self.client.idle_timeout = 0.05
        self.client.request("GET", self.srv.url + "/len").read()
        time.sleep(0.15)
        self.client.request("GET", self.srv.url + "/len").read()
        self.assertEqual(self.srv.connections, 2)

    def test_truncated_body_raises(self):
        r = self.client.request("GET", self.srv.url + "/truncated")
        with self.assertRaises(transport.TransportError):
            list(r.iter_lines())
        with self.assertRaises(transport.TransportError):
            self.client.request("GET", self.srv.url + "/truncated", stream=False)
        self.assertEqual(self.client._idle_count(), 0)

    def test_read_timeout(self):
        t0 = time.monotonic()
        with self.assertRaises(transport.TransportError) as cm:
            self.client.request("GET", self.srv.url + "/slow?s=2", timeout=0.3)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertIn("timed out", str(cm.exception))
        self.assertEqual(len([e for e in self.srv.log if e["path"] == "/slow"]), 1, "timeouts are never retried")

    def test_client_close_and_reuse_after(self):
        self.client.request("GET", self.srv.url + "/len").read()
        self.assertEqual(self.client._idle_count(), 1)
        self.client.close()
        self.assertEqual(self.client._idle_count(), 0)
        self.assertEqual(self.client.request("GET", self.srv.url + "/len").read(), b"xxxxx")
        self.assertEqual(self.client._idle_count(), 0, "closed client no longer pools")

    def test_concurrency(self):
        errors = []
        bodies = []

        def worker(i):
            try:
                for j in range(10):
                    n = (i * 10 + j) % 7 + 1
                    if j % 3 == 0:
                        r = self.client.request("GET", self.srv.url + "/chunked")
                        bodies.append(b"\n".join(r.iter_lines()) == b"line1\nline2")
                    else:
                        bodies.append(self.client.request("GET", self.srv.url + "/len?n=%d" % n).read() == b"x" * n)
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(len(bodies), 120)
        self.assertTrue(all(bodies))
        self.assertLessEqual(self.srv.connections, 12)
        self.assertLessEqual(self.client._idle_count(), self.client.max_idle_per_route)


class StaleRetryTests(unittest.TestCase):
    def setUp(self):
        self.srv = _Server()
        self.client = _client()

    def tearDown(self):
        self.client.close()
        self.srv.stop()

    def _hits(self, path):
        return [e for e in self.srv.log if e["path"] == path]

    def test_reused_connection_dropped_before_response_is_retried_once(self):
        u = self.srv.url
        self.assertEqual(self.client.request("GET", u + "/drop-reused").read(), b"fresh")
        r = self.client.request("POST", u + "/drop-reused", {"Content-Type": "text/plain"}, b"payload")
        self.assertEqual(r.read(), b"fresh")
        hits = self._hits("/drop-reused")
        self.assertEqual(len(hits), 3)
        self.assertEqual([h["conn"] for h in hits], [1, 1, 2])
        self.assertEqual(hits[2]["body"], b"payload", "body re-sent on the fresh connection")

    def test_fresh_connection_failure_not_retried(self):
        with self.assertRaises(transport.TransportError):
            self.client.request("GET", self.srv.url + "/drop")
        self.assertEqual(len(self._hits("/drop")), 1)
        self.assertEqual(self.srv.connections, 1)

    def test_reused_failure_after_response_bytes_not_retried(self):
        u = self.srv.url
        self.client.request("GET", u + "/partial-reused").read()
        with self.assertRaises(transport.TransportError):
            self.client.request("GET", u + "/partial-reused", stream=False)
        self.assertEqual(len(self._hits("/partial-reused")), 2)
        self.assertEqual(self.srv.connections, 1)

    def test_only_one_retry(self):
        u = self.srv.url
        self.client.request("GET", u + "/len").read()

        orig = self.client._open
        opened = []

        def open_then_kill(key, timeout):
            conn = orig(key, timeout)
            opened.append(conn)
            conn.http.sock.shutdown(socket.SHUT_RDWR)  # the fresh connection dies too
            return conn

        self.client._open = open_then_kill
        pooled = self.client._pool[next(iter(self.client._pool))][0]
        pooled.http.sock.shutdown(socket.SHUT_WR)  # client side half-dead: send fails or EOF
        with mock.patch.object(transport._Conn, "usable", lambda self, idle_timeout: True):  # force stale path
            with self.assertRaises(transport.TransportError):
                self.client.request("GET", u + "/len")
        self.assertEqual(len(opened), 1, "exactly one fresh connection for the retry")


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.backend = _Server()

    def tearDown(self):
        self.backend.stop()

    def test_connect_tunnel_plain_http(self):
        proxy = _ConnectProxy(self.backend.server_address[1], require_auth="Basic " +
                              base64.b64encode(b"u:p@ss").decode())
        try:
            env = {"HTTPS_PROXY": "http://u:p%%40ss@127.0.0.1:%d" % proxy.port,
                   "HTTP_PROXY": "http://u:p%%40ss@127.0.0.1:%d" % proxy.port}
            c = _client(environ=env)
            c._tunnel_schemes = ("http", "https")  # tunnel plain http to test CONNECT without TLS
            r = c.request("GET", "http://upstream.test:8080/echo?q=1", stream=False)
            self.assertEqual(r.status, 200)
            self.assertEqual(r.json()["path"], "/echo?q=1", "origin-form request inside the tunnel")
            self.assertEqual(r.json()["headers"]["Host"], "upstream.test:8080")
            self.assertNotIn("Proxy-Authorization", r.json()["headers"], "proxy creds never reach the origin")
            c.request("GET", "http://upstream.test:8080/len", stream=False)
            self.assertEqual(len(proxy.connects), 1, "the tunnel is reused (keep-alive)")
            self.assertEqual(proxy.connects[0]["line"], "CONNECT upstream.test:8080 HTTP/1.0")
            self.assertEqual(proxy.connects[0]["headers"]["host"], "upstream.test:8080")
            c.close()
            bad = _client(environ={"HTTP_PROXY": "http://u:wrong@127.0.0.1:%d" % proxy.port})
            bad._tunnel_schemes = ("http",)
            with self.assertRaises(transport.TransportError) as cm:
                bad.request("GET", "http://upstream.test:8080/len")
            self.assertIn("407", str(cm.exception))
            self.assertNotIn("wrong", str(cm.exception))
        finally:
            proxy.stop()

    def test_connect_tunnel_tls(self):
        openssl = shutil.which("openssl")
        if not openssl:
            self.skipTest("openssl not available to mint a test certificate")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = tmp.name
        ext = os.path.join(d, "ext.cnf")
        with open(ext, "w") as f:
            f.write("subjectAltName=DNS:upstream.test\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,"
                    "keyEncipherment\nextendedKeyUsage=serverAuth\nauthorityKeyIdentifier=keyid\n"
                    "subjectKeyIdentifier=hash\n")
        cmds = [
            [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "ca.key", "-out", "ca.pem",
             "-days", "2", "-subj", "/CN=gw-test-ca", "-addext", "basicConstraints=critical,CA:TRUE",
             "-addext", "keyUsage=critical,keyCertSign,cRLSign"],
            [openssl, "req", "-newkey", "rsa:2048", "-nodes", "-keyout", "leaf.key", "-out", "leaf.csr",
             "-subj", "/CN=upstream.test"],
            [openssl, "x509", "-req", "-in", "leaf.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
             "-out", "leaf.pem", "-days", "2", "-extfile", "ext.cnf"],
        ]
        for cmd in cmds:
            try:
                r = subprocess.run(cmd, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            except (OSError, subprocess.SubprocessError) as exc:
                self.skipTest("openssl failed: %s" % exc)
            if r.returncode != 0:
                self.skipTest("openssl failed: %s" % r.stderr.decode("utf-8", "replace")[-200:])
        srv = _Server()
        sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        sctx.load_cert_chain(os.path.join(d, "leaf.pem"), os.path.join(d, "leaf.key"))
        srv.socket = sctx.wrap_socket(srv.socket, server_side=True)
        proxy = _ConnectProxy(srv.server_address[1])
        try:
            env = {"https_proxy": "http://127.0.0.1:%d" % proxy.port, "SSL_CERT_FILE": os.path.join(d, "ca.pem")}
            c = _client(environ=env)
            r = c.request("GET", "https://upstream.test/echo", stream=False)
            self.assertEqual(r.status, 200)
            self.assertEqual(r.json()["headers"]["Host"], "upstream.test")
            self.assertEqual(c.request("GET", "https://upstream.test/len").read(), b"xxxxx")
            self.assertEqual(len(proxy.connects), 1)
            self.assertEqual(proxy.connects[0]["line"], "CONNECT upstream.test:443 HTTP/1.0")
            self.assertNotIn("proxy-authorization", proxy.connects[0]["headers"])
            c.close()
            untrusted = _client(environ={"https_proxy": env["https_proxy"]},
                                ssl_context=ssl.create_default_context())
            with self.assertRaises(transport.TransportError) as cm:
                untrusted.request("GET", "https://upstream.test/len")
            self.assertIn("CERTIFICATE_VERIFY_FAILED", str(cm.exception).upper().replace(" ", "_"))
        finally:
            proxy.stop()
            srv.stop()

    def test_absolute_uri_http_proxy_keepalive(self):
        env = {"http_proxy": "http://user:pw@127.0.0.1:%d" % self.backend.server_address[1]}
        c = _client(environ=env)
        self.assertEqual(c.proxy_for("http://origin.test/x"), env["http_proxy"])
        r = c.request("GET", "http://origin.test:81/echo?a=b", stream=False)
        self.assertEqual(r.json()["path"], "http://origin.test:81/echo?a=b")
        self.assertEqual(r.json()["headers"]["Host"], "origin.test:81")
        self.assertEqual(r.json()["headers"]["Proxy-Authorization"], "Basic dXNlcjpwdw==")
        c.request("GET", "http://origin.test:81/len").read()
        self.assertEqual(self.backend.connections, 1)
        c.close()

    def test_unsupported_proxy_scheme(self):
        c = _client(environ={"HTTPS_PROXY": "https://secret:pw@proxy.test:443"})
        with self.assertRaises(transport.TransportError) as cm:
            c.request("GET", "https://api.example.test/")
        self.assertNotIn("secret", str(cm.exception))

    def test_no_proxy_bypass_goes_direct(self):
        proxy = _ConnectProxy(self.backend.server_address[1])
        try:
            env = {"HTTPS_PROXY": "http://127.0.0.1:%d" % proxy.port, "NO_PROXY": "skip.invalid,.corp.invalid"}
            c = _client(environ=env)
            for url in ("https://skip.invalid/x", "https://a.corp.invalid/x"):
                self.assertIsNone(c.proxy_for(url))
                with self.assertRaises(transport.TransportError):
                    c.request("GET", url)  # direct -> DNS failure, the proxy is never contacted
            self.assertEqual(proxy.connects, [])
            self.assertIsNotNone(c.proxy_for("https://other.invalid/x"))
        finally:
            proxy.stop()

    def test_no_proxy_matching(self):
        b = transport.bypass_proxy
        cases = [
            ("api.x.ai", 443, "x.ai", True), ("api.x.ai", 443, ".x.ai", True), ("x.ai", 443, ".x.ai", True),
            ("api.x.ai", 443, "*.x.ai", True), ("API.X.AI", 443, "x.ai", True), ("notx.ai", 443, "x.ai", False),
            ("x.ai", 443, "*", True), ("x.ai", 443, "", False), ("x.ai", 443, " foo , x.ai ", True),
            ("x.ai", 8443, "x.ai:8443", True), ("x.ai", 443, "x.ai:8443", False),
            ("10.1.2.3", 443, "10.0.0.0/8", True), ("11.1.2.3", 443, "10.0.0.0/8", False),
            ("10.1.2.3", 443, "10.1.2.3", True), ("::1", 443, "::1", True), ("::2", 443, "[::2]:443", True),
            ("::2", 80, "[::2]:443", False), ("x.ai.", 443, "x.ai", True),
        ]
        for host, port, no_proxy, want in cases:
            self.assertEqual(b(host, port, no_proxy), want, (host, port, no_proxy))
        env = {"https_proxy": "http://p:1", "HTTPS_PROXY": "http://P:2", "no_proxy": "a.test", "NO_PROXY": "b.test"}
        self.assertEqual(transport.proxy_for_url("https://c.test/", env), "http://p:1", "lowercase wins")
        self.assertIsNone(transport.proxy_for_url("https://a.test/", env))
        self.assertEqual(transport.proxy_for_url("https://b.test/", env), "http://p:1", "lowercase no_proxy wins")
        self.assertIsNone(transport.proxy_for_url("https://localhost:1/", {"HTTPS_PROXY": "http://p:1"}))
        self.assertEqual(transport.proxy_for_url("http://c.test/", {"HTTP_PROXY": "http://h:3"}), "http://h:3")


if __name__ == "__main__":
    unittest.main()
