"""server.Gateway end to end over real HTTP, with a scripted fake dialect patched in place of
``server.get_dialect``: auth, path-only routing, count_tokens, models, probes, pre-flight
prompt-too-long, commit point, heartbeats, client disconnects, stream:false, concurrency, tracing,
and the ``python -m <pkg> serve`` entry point."""

import contextlib
import http.client
import io
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
from unittest import mock

from ._pkg import PKG, REPO_ROOT, mod
from .test_anthropic_out import check_grammar, parse_sse

server = mod("server")
config = mod("config")
ev = mod("events")
errors = mod("errors")
ao = mod("anthropic_out")
ai = mod("anthropic_in")
testing = mod("testing")

SENTINEL = testing.SENTINEL
MODEL = "claude-via-xai,grok-4.7"


def table_dict():
    return {
        "providers": {
            "xai": {"display_name": "xAI API", "dialect": "openai_chat", "profile": "xai",
                    "base_url": "http://127.0.0.1:9/v1", "auth": {"kind": "api_key", "secret": "xai"},
                    "allow_unlisted": True,
                    "models": [{"id": "grok-4.7", "context": 2000000}, {"id": "grok-fast", "context": 2000000},
                               {"id": "tiny", "context": 1000}]},
            "opencode": {"display_name": "OpenCode CLI", "dialect": "cli", "auth": {"kind": "none"},
                         "chat_only": True, "allow_unlisted": True, "options": {"cli": "opencode"},
                         "models": [{"id": "opencode/grok-code", "tools": False}]},
            "grok": {"display_name": "Grok", "dialect": "responses", "target": "grok_cli_proxy",
                     "base_url": "http://127.0.0.1:9/v1", "auth": {"kind": "api_key", "secret": "xai"},
                     "allow_unlisted": True,
                     "fallback": {"dialect": "openai_chat", "profile": "xai", "base_url": "http://127.0.0.1:9/v1"}},
            "other": {"dialect": "openai_chat", "base_url": "http://127.0.0.1:9/v1",
                      "auth": {"kind": "api_key", "secret": "other"}, "allow_unlisted": True,
                      "fallback": {"base_url": "http://127.0.0.1:9/v2", "auth": {"kind": "none"}}},
        },
        "roles": {"default": "xai,grok-4.7", "background": "xai,grok-fast"},
    }


def make_table():
    return config.RouteTable.from_dict(table_dict())


def body(model=MODEL, stream=True, text="hi", **kw):
    b = {"model": model, "max_tokens": 1000, "stream": stream, "messages": [{"role": "user", "content": text}]}
    b.update(kw)
    return b


class FakeDialect(object):
    """Scripted dialect: ``script(ctx, fake)`` is a generator function (or raises)."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.names = []
        self.closed = threading.Event()
        self.lock = threading.Lock()

    def get(self, name):
        with self.lock:
            self.names.append(name)
        return self

    def execute(self, ctx):
        with self.lock:
            self.calls.append(ctx)
        return self.script(ctx, self)


def simple_script(ctx, fake):
    yield ev.ThinkingDelta(0, "plan")
    yield ev.TextDelta(1, "Running it.")
    yield ev.ToolCall("toolu_1", "Bash", '{"command":"ls"}')
    yield ev.Usage(321, 12, 4)
    yield ev.Finish("tool_use")


class GatewayTestBase(unittest.TestCase):
    commit_timeout = 5.0
    heartbeat_interval = 5.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gw-server-")
        self.trace = os.path.join(self.tmp, "trace.jsonl")
        self.secrets = config.SecretStore({"xai": "xai-key-" + SENTINEL, "other": "other-key-" + SENTINEL})
        self.gw = server.Gateway(make_table(), self.secrets, trace_file=self.trace,
                                 commit_timeout=self.commit_timeout, heartbeat_interval=self.heartbeat_interval,
                                 launcher_name="ai-test")
        self.gw.start()

    def tearDown(self):
        self.gw.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def patch(self, script):
        fake = FakeDialect(script)
        p = mock.patch.object(server, "get_dialect", new=fake.get)
        p.start()
        self.addCleanup(p.stop)
        return fake

    def forbid_dialect(self):
        def boom(name):
            raise AssertionError("dialect must not be called (%s)" % name)
        p = mock.patch.object(server, "get_dialect", new=boom)
        p.start()
        self.addCleanup(p.stop)

    def request(self, method, path, payload=None, headers=None, token=True, conn=None, raw_body=None):
        own = conn is None
        if own:
            conn = http.client.HTTPConnection("127.0.0.1", self.gw.port, timeout=30)
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = "Bearer " + self.gw.token
        h.update(headers or {})
        data = raw_body if raw_body is not None else (json.dumps(payload).encode("utf-8")
                                                      if payload is not None else None)
        conn.request(method, path, body=data, headers=h)
        resp = conn.getresponse()
        raw = resp.read()
        if own:
            conn.close()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, raw

    def post(self, payload, path="/v1/messages?beta=true", **kw):
        return self.request("POST", path, payload, **kw)

    def sse(self, payload, model_id=None, **kw):
        status, headers, raw = self.post(payload, **kw)
        self.assertEqual(status, 200, raw)
        self.assertTrue(headers["content-type"].startswith("text/event-stream"))
        frames = parse_sse(raw)
        return frames, check_grammar(self, frames, model_id or payload["model"])

    def error_of(self, raw):
        data = json.loads(raw.decode("utf-8"))
        self.assertEqual(data["type"], "error")
        return data["error"]

    def trace_records(self):
        if not os.path.exists(self.trace):
            return []
        with open(self.trace, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def request_traces(self, n, timeout=10):
        """The ``request`` trace records once at least ``n`` exist (written after the response)."""
        deadline = time.monotonic() + timeout
        while True:
            recs = [r for r in self.trace_records() if r["event"] == "request"]
            if len(recs) >= n or time.monotonic() > deadline:
                return recs
            time.sleep(0.02)


class BasicEndpointTests(GatewayTestBase):
    def test_health_without_auth(self):
        status, headers, raw = self.request("GET", "/health", token=False)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["status"], "ok")
        self.assertTrue(self.gw.url.startswith("http://127.0.0.1:"))
        self.assertEqual(len(self.gw.token) >= 32, True)

    def test_auth_rejection_and_acceptance(self):
        self.patch(simple_script)
        for hdrs in ({}, {"Authorization": "Bearer wrong"}, {"x-api-key": "wrong"},
                     {"Authorization": self.gw.token}):
            status, headers, raw = self.post(body(), token=False, headers=hdrs)
            self.assertEqual(status, 401, hdrs)
            self.assertEqual(self.error_of(raw)["type"], "authentication_error")
            self.assertEqual(headers["x-should-retry"], "false")
        status, _, raw = self.request("GET", "/v1/models", token=False)
        self.assertEqual(status, 401)
        status, _, _ = self.post(body(), token=False, headers={"x-api-key": self.gw.token})
        self.assertEqual(status, 200)
        status, _, _ = self.post(body(), token=False, headers={"Authorization": "bearer " + self.gw.token,
                                                                "x-api-key": "stale"})
        self.assertEqual(status, 200)

    def test_path_only_routing(self):
        fake = self.patch(simple_script)
        for path in ("/v1/messages", "/v1/messages?beta=true", "/v1/messages/?beta=true&x=1"):
            frames, _ = self.sse(body(), path=path)
            self.assertEqual(frames[0][0], "message_start")
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(fake.names, ["openai_chat"] * 3)

    def test_count_tokens(self):
        self.forbid_dialect()
        for name in ("count_tokens_request.json", "count_tokens_system_request.json"):
            fx = testing.load_fixture(name)
            status, _, raw = self.post(fx["body"], path=fx["path"])
            self.assertEqual(status, 200)
            expected = ao.estimate_tokens(ai.parse_messages_request(fx["body"], fx["headers"]))
            self.assertEqual(json.loads(raw), {"input_tokens": expected})
        status, _, raw = self.post({"model": "m"}, path="/v1/messages/count_tokens")
        self.assertEqual(status, 400)

    def test_models_endpoints(self):
        fx = testing.load_fixture("models_request.json")
        status, _, raw = self.request("GET", fx["path"])
        self.assertEqual(status, 200)
        data = json.loads(raw)
        self.assertEqual(data["data"][0]["id"], "claude-via-xai,grok-4.7[1m]")
        self.assertFalse(data["has_more"])
        quoted = urllib.parse.quote("claude-via-xai,grok-4.7[1m]", safe="")
        status, _, raw = self.request("GET", "/v1/models/" + quoted)
        self.assertEqual((status, json.loads(raw)["id"]), (200, "claude-via-xai,grok-4.7[1m]"))
        status, _, raw = self.request("GET", "/v1/models/claude-via-nope,x?beta=true")
        self.assertEqual(status, 404)
        self.assertEqual(self.error_of(raw)["type"], "not_found_error")

    def test_unknown_endpoint_and_method(self):
        status, _, raw = self.request("GET", "/v1/other")
        self.assertEqual((status, self.error_of(raw)["type"]), (404, "not_found_error"))
        status, _, raw = self.request("GET", "/v1/messages")
        self.assertEqual((status, self.error_of(raw)["type"]), (405, "invalid_request_error"))

    def test_malformed_requests(self):
        self.forbid_dialect()
        status, headers, raw = self.post(None, raw_body=b"{not json")
        self.assertEqual((status, self.error_of(raw)["type"]), (400, "invalid_request_error"))
        self.assertEqual(headers["x-should-retry"], "false")
        status, _, raw = self.post({"model": MODEL, "messages": "x"})
        self.assertEqual(status, 400)
        self.assertIn("messages", self.error_of(raw)["message"])

    def test_gzip_request_body(self):
        import gzip

        fake = self.patch(simple_script)
        data = gzip.compress(json.dumps(body(stream=False)).encode("utf-8"))
        status, _, raw = self.post(None, raw_body=data, headers={"Content-Encoding": "gzip"})
        self.assertEqual(status, 200, raw)
        self.assertEqual(len(fake.calls), 1)
        status, _, raw = self.post(None, raw_body=b"xx", headers={"Content-Encoding": "br"})
        self.assertEqual(status, 400)

    def test_unknown_model_404_stream_and_nonstream(self):
        self.forbid_dialect()
        fx = testing.load_fixture("nonstream_retry_after_404_request.json")
        for stream in (True, False):
            b = dict(fx["body"], stream=stream)
            status, headers, raw = self.post(b)
            self.assertEqual(status, 404)
            err = self.error_of(raw)
            self.assertEqual(err["type"], "not_found_error")
            self.assertIn("model 'claude-via-nope,x' is not routable", err["message"])
            self.assertIn("(run `ai-test models`)", err["message"])
            self.assertEqual(headers["x-should-retry"], "false")

    def test_dialect_unavailable(self):
        def missing(name):
            raise ImportError("No module named 'dialects.%s'" % name)
        with mock.patch.object(server, "get_dialect", new=missing):
            status, _, raw = self.post(body())
        self.assertEqual(status, 500)
        self.assertIn("openai_chat", self.error_of(raw)["message"])

    def test_keep_alive_reuse(self):
        self.patch(simple_script)
        conn = http.client.HTTPConnection("127.0.0.1", self.gw.port, timeout=30)
        try:
            for payload in (body(), body(stream=False), body()):
                status, _, raw = self.request("POST", "/v1/messages?beta=true", payload, conn=conn)
                self.assertEqual(status, 200)
            status, _, _ = self.request("GET", "/v1/models", conn=conn)
            self.assertEqual(status, 200)
            status, _, _ = self.request("GET", "/v1/models", conn=conn, token=False)
            self.assertEqual(status, 401)
        finally:
            conn.close()

    def test_nothing_written_to_stdout_or_stderr(self):
        self.patch(simple_script)
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            self.post(body())
            self.post(body(model="claude-via-nope,x"))
            self.post(None, raw_body=b"garbage")
            s = socket.create_connection(("127.0.0.1", self.gw.port), timeout=10)
            s.sendall(b"GARBAGE\r\n\r\n")
            s.recv(65536)
            s.close()
            s = socket.create_connection(("127.0.0.1", self.gw.port), timeout=10)
            s.close()
            time.sleep(0.2)
        self.assertEqual((err.getvalue(), out.getvalue()), ("", ""))


class MessagesTests(GatewayTestBase):
    def test_stream_success_and_context(self):
        fake = self.patch(simple_script)
        fx = testing.load_fixture("turn1_request.json")
        frames, replay = self.sse(fx["body"], headers={"X-Claude-Code-Session-Id": "sess-1"})
        self.assertEqual(replay["model"], MODEL)
        self.assertEqual([b["type"] for b in replay["content"]], ["thinking", "text", "tool_use"])
        self.assertEqual(replay["stop_reason"], "tool_use")
        self.assertEqual(replay["usage"]["input_tokens"], 321)
        ctx = fake.calls[0]
        self.assertEqual(ctx.requested_model, MODEL)
        self.assertEqual(ctx.session_id, "sess-1")
        self.assertEqual((ctx.provider.id, ctx.model.id), ("xai", "grok-4.7"))
        self.assertEqual(ctx.resolution.model, "grok-4.7")
        self.assertIs(ctx.auth, self.gw.auth_for("xai"))
        self.assertIs(ctx.runtime, self.gw.runtime_for("xai"))
        self.assertIs(ctx.secrets, self.gw.secrets)
        self.assertEqual(ctx.est_tokens, ao.estimate_tokens(ctx.req))
        self.assertEqual(ctx.req.effort, "high")
        self.assertTrue(callable(ctx.http.request))
        self.assertEqual(ctx.auth.headers(), {"Authorization": "Bearer xai-key-" + SENTINEL})
        # second request: same auth/runtime instances, generated session id without the header
        b = body()
        self.sse(b)
        ctx2 = fake.calls[1]
        self.assertIs(ctx2.auth, ctx.auth)
        self.assertIs(ctx2.runtime, ctx.runtime)
        self.assertTrue(ctx2.session_id and ctx2.session_id != "sess-1")

    def test_background_effort_low(self):
        fake = self.patch(simple_script)
        fx = testing.load_fixture("background_request.json")
        self.sse(fx["body"])
        ctx = fake.calls[0]
        self.assertTrue(ctx.background)
        self.assertEqual(ctx.req.effort, "low")
        self.assertEqual(ctx.model.id, "grok-fast")
        self.assertEqual(ctx.req.output_format["type"], "json_schema")

    def test_stream_false_aggregation(self):
        self.patch(simple_script)
        status, headers, raw = self.post(body(stream=False))
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json")
        msg = json.loads(raw)
        self.assertEqual(msg["model"], MODEL)
        self.assertRegex(msg["id"], r"^msg_[0-9a-f]{24}$")
        self.assertEqual(msg["content"][2], {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                                             "input": {"command": "ls"}})
        self.assertEqual(msg["content"][0]["signature"][:10], "fgw1.chat.")
        self.assertEqual(msg["stop_reason"], "tool_use")

    def test_stream_false_stream_error_becomes_http_error(self):
        def script(ctx, fake):
            yield ev.TextDelta(0, "partial")
            yield ev.StreamError("overloaded_error", "upstream overloaded", True)
        self.patch(script)
        status, headers, raw = self.post(body(stream=False))
        self.assertEqual(status, 529)
        self.assertEqual(self.error_of(raw)["type"], "overloaded_error")
        self.assertEqual(headers["x-should-retry"], "true")

    def test_probe_shortcut_for_cli_routes(self):
        self.forbid_dialect()
        model = "claude-via-opencode,opencode/grok-code"
        frames, replay = self.sse(body(model=model, max_tokens=1))
        self.assertEqual(replay["content"], [{"type": "text", "text": "ok"}])
        status, _, raw = self.post(body(model=model, max_tokens=1, stream=False))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["content"], [{"type": "text", "text": "ok"}])

    def test_probe_forwarded_for_api_routes(self):
        fake = self.patch(simple_script)
        self.sse(body(max_tokens=1))
        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(fake.calls[0].req.is_probe())

    def test_preflight_prompt_too_long(self):
        self.forbid_dialect()
        b = body(model="claude-via-xai,tiny", text="x" * 5000)
        est = ao.estimate_tokens(ai.parse_messages_request(b, {}))
        self.assertGreater(est, 1100)
        for stream in (True, False):
            status, headers, raw = self.post(dict(b, stream=stream))
            self.assertEqual(status, 400)
            err = self.error_of(raw)
            self.assertEqual(err["message"], "prompt is too long: %d tokens > 1000 maximum" % est)
            self.assertEqual(headers["x-should-retry"], "false")
        # just under 1.1 x context goes through
        fake = self.patch(simple_script)
        self.sse(body(model="claude-via-xai,tiny", text="x" * 3500))
        self.assertEqual(len(fake.calls), 1)

    def test_error_before_commit_is_real_http_status(self):
        def script(ctx, fake):
            raise errors.GatewayError(429, "rate_limit_error", "[xai/grok-4.7] rate limited", True, retry_after=3)
            yield  # pragma: no cover - makes this a generator
        self.patch(script)
        for stream in (True, False):
            status, headers, raw = self.post(body(stream=stream))
            self.assertEqual(status, 429)
            self.assertEqual(headers["x-should-retry"], "true")
            self.assertEqual(headers["retry-after"], "3")
            self.assertEqual(self.error_of(raw)["type"], "rate_limit_error")

    def test_error_after_commit_is_sse_error_event(self):
        def script(ctx, fake):
            yield ev.TextDelta(0, "partial")
            raise errors.GatewayError(503, "overloaded_error", "busy", True)
        self.patch(script)
        frames, replay = self.sse(body())
        self.assertEqual(replay, {"error": {"type": "overloaded_error", "message": "busy"}})

        def script2(ctx, fake):
            yield ev.TextDelta(0, "partial")
            yield ev.StreamError("api_error", "terminal failure", False)
            yield ev.TextDelta(0, "never")
        self.patch(script2)
        frames, replay = self.sse(body())
        self.assertEqual(replay, {"error": {"type": "api_error", "message": "terminal failure"}})

        def script3(ctx, fake):
            yield ev.TextDelta(0, "partial")
            raise RuntimeError("dialect bug")
        self.patch(script3)
        frames, replay = self.sse(body())
        self.assertEqual(replay["error"]["type"], "api_error")

    def test_unexpected_exception_before_commit(self):
        def script(ctx, fake):
            raise KeyError("oops")
            yield  # pragma: no cover
        self.patch(script)
        status, headers, raw = self.post(body())
        self.assertEqual((status, headers["x-should-retry"]), (500, "false"))

    def test_trace_file(self):
        self.patch(simple_script)
        b = body(text="my key is xai-key-" + SENTINEL)
        self.sse(b, headers={"X-Claude-Code-Session-Id": "s"})
        self.post(body(model="claude-via-nope,x"))
        recs = self.request_traces(2)
        self.assertEqual(len(recs), 2)
        ok, missing = recs
        self.assertEqual((ok["method"], ok["path"], ok["status"], ok["provider"], ok["model"]),
                         ("POST", "/v1/messages", 200, "xai", "grok-4.7"))
        self.assertTrue(ok["stream"])
        self.assertEqual(ok["events"], {"thinking": 1, "text": 1, "tool_call": 1, "usage": 1, "finish": 1})
        self.assertIn("duration_ms", ok)
        self.assertIn("commit_ms", ok)
        self.assertNotIn("request_body", ok)
        self.assertEqual((missing["status"], missing["error"]), (404, "not_found_error"))
        with open(self.trace, encoding="utf-8") as f:
            blob = f.read()
        self.assertNotIn(SENTINEL, blob)
        self.assertNotIn("my key is", blob)
        # bodies only with AI_GATEWAY_TRACE_BODIES=1, and redacted
        self.gw.tracer.bodies = True
        self.sse(b)
        last = self.request_traces(3)[-1]
        self.assertEqual(last["request_body"]["messages"][0]["content"], "my key is ***")
        with open(self.trace, encoding="utf-8") as f:
            self.assertNotIn(SENTINEL, f.read())

    def test_concurrent_requests(self):
        def script(ctx, fake):
            time.sleep(0.5)
            yield ev.TextDelta(0, ctx.req.messages[0].text())
            yield ev.Finish("end_turn")
        self.patch(script)
        results = queue.Queue()

        def worker(i):
            try:
                if i % 2 == 0:
                    replay = self.sse(body(text="req-%d" % i))[1]
                else:
                    replay = json.loads(self.post(body(text="req-%d" % i, stream=False))[2])
                results.put((i, replay["content"][0]["text"]))
            except Exception as exc:  # surfaced by the assertion below
                results.put((i, repr(exc)))

        t0 = time.monotonic()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        elapsed = time.monotonic() - t0
        got = dict(results.get_nowait() for _ in range(10))
        self.assertEqual(got, {i: "req-%d" % i for i in range(10)})
        self.assertLess(elapsed, 3.0)

    def test_auth_and_runtime_singletons(self):
        a = self.gw.auth_for("grok")
        self.assertIs(a, self.gw.auth_for("grok"))
        self.assertIs(self.gw.auth_for("grok", fallback=True), a)  # fallback inherits the same auth dict
        other = self.gw.auth_for("other")
        fb = self.gw.auth_for("other", fallback=True)
        self.assertIsNot(fb, other)
        self.assertEqual(fb.kind, "none")
        self.assertIs(self.gw.runtime_for("grok"), self.gw.runtime_for("grok"))
        self.assertIsNot(self.gw.runtime_for("grok"), self.gw.runtime_for("xai"))
        results = []
        threads = [threading.Thread(target=lambda: results.append(self.gw.auth_for("xai"))) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(len(set(map(id, results))), 1)


class TimingTests(GatewayTestBase):
    commit_timeout = 0.3
    heartbeat_interval = 0.1

    def test_commit_after_timeout_then_error_event(self):
        def script(ctx, fake):
            time.sleep(1.0)
            raise errors.GatewayError(429, "rate_limit_error", "slow", True)
            yield  # pragma: no cover
        self.patch(script)
        conn = http.client.HTTPConnection("127.0.0.1", self.gw.port, timeout=30)
        t0 = time.monotonic()
        conn.request("POST", "/v1/messages?beta=true", json.dumps(body()).encode("utf-8"),
                     {"Authorization": "Bearer " + self.gw.token, "Content-Type": "application/json"})
        resp = conn.getresponse()
        headers_after = time.monotonic() - t0
        raw = resp.read()
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertLess(headers_after, 0.9)
        frames = parse_sse(raw)
        self.assertEqual(frames[0][0], "message_start")
        self.assertGreaterEqual([n for n, _ in frames].count("ping"), 3)
        self.assertEqual(frames[-1], ("error", {"type": "error", "error": {"type": "overloaded_error",
                                                                          "message": "slow"}}))
        rec = self.request_traces(1)[-1]
        self.assertEqual(rec["commit"], "timeout")

    def test_heartbeat_pings_during_silence(self):
        def script(ctx, fake):
            yield ev.TextDelta(0, "a")
            time.sleep(0.75)
            yield ev.TextDelta(0, "b")
            yield ev.Finish("end_turn")
        self.patch(script)
        frames, replay = self.sse(body())
        names = [n for n, _ in frames]
        first, second = names.index("content_block_delta"), len(names) - 1 - names[::-1].index("content_block_delta")
        self.assertGreaterEqual(names[first:second].count("ping"), 4)
        self.assertEqual(replay["content"], [{"type": "text", "text": "ab"}])

    def test_client_disconnect_closes_dialect_generator(self):
        def script(ctx, fake):
            try:
                for i in range(3000):
                    yield ev.TextDelta(0, "chunk%d " % i)
                    time.sleep(0.01)
                yield ev.Finish("end_turn")
            finally:
                fake.closed.set()
        fake = self.patch(script)
        s = socket.create_connection(("127.0.0.1", self.gw.port), timeout=10)
        payload = json.dumps(body()).encode("utf-8")
        s.sendall(b"POST /v1/messages?beta=true HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer " +
                  self.gw.token.encode() + b"\r\nContent-Type: application/json\r\nContent-Length: " +
                  str(len(payload)).encode() + b"\r\n\r\n" + payload)
        got = b""
        while b"content_block_delta" not in got:
            got += s.recv(4096)
        s.close()
        self.assertTrue(fake.closed.wait(10), "dialect generator was not closed after the client left")
        recs = self.request_traces(1)
        self.assertTrue(recs and recs[-1].get("client_gone"))
        # the gateway keeps serving
        self.patch(simple_script)
        self.sse(body())

    def test_upstream_response_closed_on_disconnect(self):
        """A dialect blocked on an upstream read is unblocked through ctx.http (response.close())."""
        blocker = threading.Event()

        class FakeResponse(object):
            closed = False

            def close(self):
                self.closed = True
                blocker.set()

        responses = []

        class FakeHttp(object):
            def request(self, method, url, headers=None, body=None, stream=True, timeout=None):
                r = FakeResponse()
                responses.append(r)
                return r

            def close(self):
                pass

        self.gw.http = FakeHttp()

        def script(ctx, fake):
            try:
                ctx.http.request("POST", "http://upstream/x", {}, b"{}")
                yield ev.TextDelta(0, "first")
                blocker.wait(30)  # "upstream read" until the response is closed
                yield ev.TextDelta(0, "late")
            finally:
                fake.closed.set()
        fake = self.patch(script)
        s = socket.create_connection(("127.0.0.1", self.gw.port), timeout=10)
        payload = json.dumps(body()).encode("utf-8")
        s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nx-api-key: " + self.gw.token.encode() +
                  b"\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        got = b""
        while b"content_block_delta" not in got:
            got += s.recv(4096)
        s.close()
        self.assertTrue(fake.closed.wait(10))
        self.assertTrue(responses and responses[0].closed)


class LifecycleTests(unittest.TestCase):
    def test_context_manager_stop_and_env_override(self):
        with mock.patch.dict(os.environ, {"AI_GATEWAY_UPSTREAM_XAI": "http://127.0.0.1:4321/v1/"}):
            with server.Gateway(make_table(), config.SecretStore(), token="fixed-token") as gw:
                self.assertEqual(gw.table.providers["xai"].base_url, "http://127.0.0.1:4321/v1")
                self.assertEqual(gw.token, "fixed-token")
                port = gw.port
                idle = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                idle.request("GET", "/health")
                resp = idle.getresponse()
                self.assertEqual(resp.status, 200)
                resp.read()  # the keep-alive connection now idles in the server's handler thread
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
        idle.sock.settimeout(5)
        try:
            self.assertEqual(idle.sock.recv(10), b"")  # stop() shut the idle connection down
        except ConnectionResetError:
            pass
        idle.close()
        gw.stop()  # idempotent

    def test_fixed_port_bind(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        gw = server.Gateway(make_table(), port=port).start()
        try:
            self.assertEqual(gw.port, port)
            self.assertEqual(gw.url, "http://127.0.0.1:%d" % port)
            with self.assertRaises(OSError):
                server.Gateway(make_table(), port=port).start()
        finally:
            gw.stop()


class MainEntryPointTests(unittest.TestCase):
    def test_serve_prints_only_export_lines(self):
        env = dict(os.environ)
        env["XAI_API_KEY"] = "xai-test-" + SENTINEL
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("AI_GATEWAY_TOKEN", None)
        env.pop("AI_GATEWAY_TRACE_FILE", None)
        proc = subprocess.Popen([sys.executable, "-m", PKG, "serve", "--preset", "xai", "--port", "0"],
                                cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        lines = queue.Queue()
        threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()
        try:
            got = [lines.get(timeout=60).rstrip("\n") for _ in range(3)]
            self.assertTrue(got[0].startswith("export ANTHROPIC_BASE_URL=http://127.0.0.1:"), got)
            self.assertTrue(got[1].startswith("export ANTHROPIC_AUTH_TOKEN="), got)
            self.assertEqual(got[2], "export ANTHROPIC_MODEL='claude-via-xai,grok-4.7[1m]'")
            url = urllib.parse.urlsplit(got[0].split("=", 1)[1])
            token = got[1].split("=", 1)[1]
            conn = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
            conn.request("GET", "/v1/models?limit=1000", headers={"Authorization": "Bearer " + token})
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            self.assertTrue(json.loads(resp.read())["data"])
            conn.close()
        finally:
            proc.terminate()
            try:
                out, err = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate(timeout=30)
        self.assertEqual(out, "")
        self.assertEqual(err, "")
        if os.name != "nt":
            self.assertEqual(proc.returncode, 0)

    def test_serve_bad_arguments(self):
        r = subprocess.run([sys.executable, "-m", PKG, "serve", "--preset", "nope"], cwd=REPO_ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(r.stdout, "")
        self.assertIn("nope", r.stderr)
        main = mod("__main__")
        self.assertEqual(main.export_lines("http://h:1", "t'k", "claude-via-a,b[1m]", "powershell"),
                         ["$env:ANTHROPIC_BASE_URL = 'http://h:1'", "$env:ANTHROPIC_AUTH_TOKEN = 't''k'",
                          "$env:ANTHROPIC_MODEL = 'claude-via-a,b[1m]'"])
        self.assertEqual(main.export_lines("u", "t", None, "cmd"),
                         ['set "ANTHROPIC_BASE_URL=u"', 'set "ANTHROPIC_AUTH_TOKEN=t"'])
        store, missing = main.load_secrets(mod("presets").route_table_from_presets(["xai", "gemini"]),
                                           environ={"GOOGLE_API_KEY": "g-key"})
        self.assertEqual((store.get("gemini"), missing), ("g-key", ["xai"]))


if __name__ == "__main__":
    unittest.main()
