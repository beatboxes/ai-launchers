"""anthropic_passthrough dialect (DESIGN §3.4 + §0.A) against the ``anthropic`` mock upstream."""

import copy
import json
import unittest
from unittest import mock

from ._pkg import mod

ev = mod("events")
model = mod("model")
config = mod("config")
errors = mod("errors")
presets = mod("presets")
dbase = mod("dialects.base")
dialects = mod("dialects")
pt = mod("dialects.anthropic_passthrough")
transport = mod("transport")
auth_pkg = mod("auth")
testing = mod("testing")
mu = mod("testing.mock_upstreams")
ma = mod("testing.mock_anthropic")

KEY = "FAKEKEY-SENTINEL-deepseek-0123456789"
BETAS = ("claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,"
         "mid-conversation-system-2026-04-07,mid-conversation-tool-changes-2026-07-01,effort-2025-11-24")
HEADERS = {"anthropic-beta": BETAS, "anthropic-version": "2023-06-01",
           "user-agent": "claude-cli/2.1.288 (external, sdk-cli)", "x-claude-code-session-id": "sess-1"}


class Resolution(object):
    def __init__(self, provider, model_id, spec, requested):
        self.provider, self.model, self.model_spec, self.requested = provider, model_id, spec, requested
        self.role, self.background = None, False


def provider_for(url, preset="deepseek", overrides=None):
    ov = {"base_url": url}
    ov.update(overrides or {})
    return presets.provider_from_preset(preset, ov)


def make_ctx(provider, model_id, body, headers=None, secrets=None, runtime=None):
    secrets = secrets if secrets is not None else config.SecretStore({provider.id: KEY})
    spec = provider.model_spec(model_id)
    req = model.NormalizedRequest(model=body.get("model", model_id), max_tokens=body.get("max_tokens", 32000),
                                  stream=bool(body.get("stream", True)), raw=body,
                                  headers=dict(HEADERS if headers is None else headers))
    return dbase.RequestContext(req, Resolution(provider, model_id, spec, req.model), provider, spec,
                                runtime or dbase.ProviderRuntime(provider.id),
                                auth_pkg.make_auth(provider.auth, secrets, provider.id),
                                transport.HttpClient(timeout=20, environ={}), "sess-1",
                                requested_model=req.model, est_tokens=1000, secrets=secrets)


def run(ctx):
    return list(pt.AnthropicPassthroughDialect().execute(ctx))


def user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


TOOLS = [{"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {
    "command": {"type": "string"}}}}, {"name": "Read", "description": "read", "input_schema": {"type": "object"}}]


def turn1_body(**kw):
    body = {"model": "claude-via-deepseek,deepseek-v4-pro[1m]", "max_tokens": 32000, "stream": True,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.288; cc_entrypoint=cli;"},
                       {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}],
            "messages": [user("Run: printf hello > out.txt")], "tools": copy.deepcopy(TOOLS),
            "thinking": {"type": "adaptive", "display": "updates"}, "output_config": {"effort": "high"},
            "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
            "metadata": {"user_id": "{\"session_id\":\"sess-1\"}"}}
    body.update(kw)
    return body


def events_of(events, cls):
    return [e for e in events if isinstance(e, cls)]


# =========================================================================================
# request building (no network)
# =========================================================================================

class RequestBodyTests(unittest.TestCase):
    def setUp(self):
        self.p = provider_for("https://api.deepseek.com/anthropic")

    def test_turn1_fixture(self):
        fx = testing.load_fixture("turn1_request.json")
        raw = fx["body"]
        raw["model"] = "claude-via-deepseek,deepseek-v4-pro[1m]"
        snapshot = json.dumps(raw, sort_keys=True)
        hdrs = {k.lower(): v for k, v in fx["headers"].items()}
        ctx = make_ctx(self.p, "deepseek-v4-pro", raw, headers=hdrs)
        body = pt.build_request_body(ctx)
        self.assertEqual(json.dumps(raw, sort_keys=True), snapshot, "req.raw must never be mutated")
        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertTrue(body["system"][0]["text"].startswith("x-anthropic-billing-header:"))  # kept
        self.assertEqual([m["role"] for m in body["messages"]], ["user"])  # system message folded
        self.assertIn("<system-reminder>", body["messages"][0]["content"][-1]["text"])
        self.assertEqual(len(body["tools"]), len(raw["tools"]))
        self.assertEqual(body["max_tokens"], 32000)
        self.assertIs(body["stream"], True)
        self.assertEqual(body["thinking"], raw["thinking"])
        self.assertEqual(body["output_config"], raw["output_config"])  # forwarded untouched
        h = pt.build_headers(ctx)
        self.assertEqual(h["anthropic-version"], "2023-06-01")
        self.assertEqual(h["Accept"], "text/event-stream")
        self.assertNotIn("mid-conversation", h["anthropic-beta"])
        self.assertIn("claude-code-20250219", h["anthropic-beta"])
        self.assertEqual(h["User-Agent"], fx["headers"]["User-Agent"])
        self.assertNotIn("Authorization", h)

    def test_fold_with_tool_changes_and_trailing_system(self):
        raw = turn1_body(messages=[
            user("hi"),
            {"role": "system", "content": [
                {"type": "text", "text": "env info"},
                {"type": "tool_addition", "tool": {"type": "tool_definition",
                                                   "definition": {"name": "Grep", "input_schema": {"type": "object"}}}},
                {"type": "tool_removal", "tool": {"name": "Read"}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "system", "content": "<total_tokens>5 left</total_tokens>"}])
        body = pt.build_request_body(make_ctx(self.p, "deepseek-v4-pro", raw))
        self.assertEqual([m["role"] for m in body["messages"]], ["user", "assistant", "user"])
        self.assertEqual(body["messages"][0]["content"][-1]["text"], "<system-reminder>\nenv info\n</system-reminder>")
        self.assertIn("<total_tokens>", body["messages"][2]["content"][0]["text"])
        self.assertEqual([t["name"] for t in body["tools"]], ["Bash", "Grep"])

    def test_thinking_signature_filter(self):
        fgw = "fgw1.chat.0123456789ab"
        raw = turn1_body(messages=[
            user("a"),
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": fgw}]},
            user("b"),
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "real", "signature": "upstream-sig"},
                                              {"type": "thinking", "thinking": "unsigned", "signature": ""},
                                              {"type": "text", "text": "answer"}]},
            user("c")])
        body = pt.build_request_body(make_ctx(self.p, "deepseek-v4-pro", raw))
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["user", "user", "assistant", "user"])  # thinking-only assistant turn dropped
        kept = body["messages"][2]["content"]
        self.assertEqual([b.get("signature") for b in kept if b["type"] == "thinking"], ["upstream-sig"])
        self.assertIn("thinking", body)  # last assistant turn did not call tools
        self.assertIn("context_management", body)

    def test_tool_turn_losing_foreign_thinking_disables_thinking(self):
        sig = mod("signatures").encode_signature("gemini_api", {"s": "abc"})
        raw = turn1_body(messages=[
            user("go"),
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": sig},
                                              {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                                               "input": {"command": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]}])
        body = pt.build_request_body(make_ctx(self.p, "deepseek-v4-pro", raw))
        self.assertNotIn("thinking", body)
        self.assertNotIn("context_management", body)
        self.assertEqual([b["type"] for b in body["messages"][1]["content"]], ["tool_use"])

    def test_max_tokens_clamp_and_budget(self):
        p = provider_for("https://x.example/anthropic", overrides={"models": [{"id": "small", "max_output": 8000},
                                                                              {"id": "tiny", "max_output": 900}]})
        body = pt.build_request_body(make_ctx(p, "small", turn1_body(
            thinking={"type": "enabled", "budget_tokens": 10000})))
        self.assertEqual(body["max_tokens"], 8000)
        self.assertEqual(body["thinking"], {"type": "enabled", "budget_tokens": 7999})
        tiny = turn1_body(thinking={"type": "enabled", "budget_tokens": 2000})
        body = pt.build_request_body(make_ctx(p, "tiny", tiny))
        self.assertEqual(body["max_tokens"], 900)
        self.assertNotIn("thinking", body)
        self.assertNotIn("context_management", body)
        body = pt.build_request_body(make_ctx(p, "unlisted", turn1_body(max_tokens=50000)))
        self.assertEqual(body["max_tokens"], 50000)  # unknown limit -> unchanged
        p2 = provider_for("https://x.example/anthropic", overrides={"options": {"max_output": 4096},
                                                                    "models": []})
        self.assertEqual(pt.build_request_body(make_ctx(p2, "m", turn1_body()))["max_tokens"], 4096)
        self.assertEqual(pt.build_request_body(make_ctx(p2, "m", turn1_body(max_tokens=None)))["max_tokens"], 4096)

    def test_betas_options_url_and_fields(self):
        p = provider_for("https://x.example/anthropic", overrides={"options": {"drop_betas": True}})
        self.assertNotIn("anthropic-beta", pt.build_headers(make_ctx(p, "m", turn1_body())))
        p = provider_for("https://x.example/anthropic", overrides={"options": {"drop_betas": ["effort-", "context-"],
                                                                               "drop_fields": ["metadata"]},
                                                                   "headers": {"X-Title": "t"}})
        ctx = make_ctx(p, "m", turn1_body())
        h = pt.build_headers(ctx, stream=False)
        self.assertEqual(h["anthropic-beta"], "claude-code-20250219,interleaved-thinking-2025-05-14")
        self.assertEqual((h["Accept"], h["X-Title"]), ("application/json", "t"))
        self.assertNotIn("metadata", pt.build_request_body(ctx, stream=False))
        self.assertIs(pt.build_request_body(ctx, stream=False)["stream"], False)
        self.assertEqual(pt.request_url(ctx), "https://x.example/anthropic/v1/messages")
        zen = presets.provider_from_preset("opencode-zen")
        ctx = make_ctx(zen, "qwen3.6-plus", turn1_body())
        self.assertEqual(pt.request_url(ctx), "https://opencode.ai/zen/v1/messages")
        self.assertEqual(zen.effective_dialect(ctx.model), "anthropic_passthrough")
        self.assertEqual(pt.upstream_model(make_ctx(p, "kimi-k3[1m]", turn1_body())), "kimi-k3")


# =========================================================================================
# against the mock upstream
# =========================================================================================

class MockUpstreamTests(unittest.TestCase):
    def mock(self, **options):
        opts = {"auth": "bearer", "key": KEY}
        opts.update(options)
        srv = mu.MockServer(kind="anthropic", options=opts).start()
        self.addCleanup(srv.stop)
        return srv

    def assertNoMockErrors(self, srv):
        self.assertEqual(srv.errors, [])

    def test_stream_tool_call_then_done_with_real_signature(self):
        srv = self.mock(models=["deepseek-v4-pro"])
        p = provider_for(srv.url)
        events = run(make_ctx(p, "deepseek-v4-pro", turn1_body()))
        self.assertEqual(events[0], ev.ThinkingDelta(0, ""))
        thinking = "".join(e.text for e in events_of(events, ev.ThinkingDelta))
        self.assertEqual(thinking, "Planning the tool call.")
        sigs = events_of(events, ev.ThinkingSignature)
        self.assertEqual(len(sigs), 1)
        self.assertTrue(sigs[0].signature.startswith(ma.DEFAULT_SIGNATURE_PREFIX))  # real upstream sig, not fgw1
        calls = events_of(events, ev.ToolCall)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "Bash")
        self.assertEqual(json.loads(calls[0].input_json)["command"], mu.DEFAULT_COMMAND)
        self.assertEqual(events[-1], ev.Finish("tool_use"))
        usage = events_of(events, ev.Usage)[-1]
        self.assertGreater(usage.input_tokens, 0)
        self.assertGreater(usage.output_tokens, 0)
        rec = srv.requests_for("/v1/messages", "POST")[-1]
        self.assertEqual(rec["headers"].get("Authorization"), "Bearer " + KEY)
        self.assertEqual(rec["headers"].get("anthropic-version"), "2023-06-01")
        self.assertNotIn("mid-conversation", rec["headers"].get("anthropic-beta"))
        self.assertEqual(rec["body_json"]["model"], "deepseek-v4-pro")
        self.assertIs(rec["body_json"]["stream"], True)

        # turn 2: echo the real-signed thinking + an older gateway-signed one (must be stripped)
        old = {"role": "assistant", "content": [{"type": "thinking", "thinking": "old",
                                                 "signature": "fgw1.chat.0123456789ab"},
                                                {"type": "text", "text": "earlier answer"}]}
        assistant = {"role": "assistant", "content": [
            {"type": "thinking", "thinking": thinking, "signature": sigs[0].signature},
            {"type": "tool_use", "id": calls[0].id, "name": "Bash", "input": json.loads(calls[0].input_json)}]}
        body2 = turn1_body(messages=[user("first"), old, user("Run: printf hello > out.txt"), assistant,
                                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0].id,
                                                                   "content": "hello"}]},
                                     {"role": "system", "content": "<total_tokens>9 left</total_tokens>"}])
        events2 = run(make_ctx(p, "deepseek-v4-pro", body2))
        text = "".join(e.text for e in events_of(events2, ev.TextDelta))
        self.assertEqual(text, "DONE " + mu.sha8("hello"))
        self.assertEqual(events2[-1], ev.Finish("end_turn"))
        self.assertNoMockErrors(srv)
        self.assertEqual(srv.brain.leaks, [])

    def test_non_stream_upstream(self):
        for opts in ({"upstream_stream": False}, {}):
            srv = self.mock(force_json=not opts)
            p = provider_for(srv.url, overrides={"options": opts})
            events = run(make_ctx(p, "deepseek-v4-pro", turn1_body()))
            self.assertEqual([type(e).__name__ for e in events],
                             ["ThinkingDelta", "ThinkingSignature", "ToolCall", "Usage", "Finish"], opts)
            self.assertEqual(events[0].text, "Planning the tool call.")
            self.assertEqual(events[-1].stop_reason, "tool_use")
            self.assertEqual(srv.last_request()["body_json"]["stream"], not opts)

    def test_x_api_key_auth_style(self):
        srv = self.mock(auth="x-api-key")
        p = provider_for(srv.url, overrides={"auth": {"style": "x-api-key"}})
        events = run(make_ctx(p, "deepseek-v4-pro", turn1_body()))
        self.assertEqual(events[-1].stop_reason, "tool_use")
        hdrs = srv.last_request()["headers"]
        self.assertEqual(hdrs.get("x-api-key"), KEY)
        self.assertNotIn("Authorization", hdrs)

    def test_opencode_messages_path(self):
        srv = self.mock(messages_path="/zen/v1/messages")
        zen = presets.provider_from_preset("opencode-zen", {"base_url": srv.url + "/zen/v1"})
        ctx = make_ctx(zen, "qwen3.6-plus", turn1_body(model="claude-via-opencode-zen,qwen3.6-plus"),
                       secrets=config.SecretStore({"opencode-zen": KEY}))
        events = run(ctx)
        self.assertEqual(events[-1].stop_reason, "tool_use")
        rec = srv.last_request()
        self.assertEqual(rec["path"], "/zen/v1/messages")
        self.assertEqual(rec["body_json"]["model"], "qwen3.6-plus")

    def test_http_error_mapping(self):
        cases = [
            ({"status": 400, "body": {"type": "error", "error": {"type": "invalid_request_error",
                                                                 "message": "tools.0: bad schema"}}},
             400, "invalid_request_error", False, "tools.0: bad schema"),
            ({"status": 401, "body": {"type": "error", "error": {"type": "authentication_error",
                                                                 "message": "invalid x-api-key"}}},
             401, "authentication_error", False, "invalid x-api-key"),
            ({"status": 429, "body": {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
              "headers": {"retry-after": "7"}}, 429, "rate_limit_error", True, "slow down"),
            ({"status": 402, "body": {"error": {"message": "Insufficient Balance", "type": "unknown_error"}}},
             429, "rate_limit_error", False, "Insufficient Balance"),
            ({"status": 529, "body": {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}},
             529, "overloaded_error", True, "Overloaded"),
            ({"status": 400, "body": {"type": "error", "error": {"type": "invalid_request_error",
                                                                 "message": "prompt is too long: 300000 tokens > "
                                                                            "200000 maximum"}}},
             400, "invalid_request_error", False, "prompt is too long: 300000 tokens > 200000 maximum"),
        ]
        for fail, status, etype, retry, msg in cases:
            srv = self.mock(fail=fail)
            p = provider_for(srv.url)
            with self.assertRaises(errors.GatewayError) as cm:
                run(make_ctx(p, "deepseek-v4-pro", turn1_body()))
            e = cm.exception
            self.assertEqual((e.status, e.err_type, e.should_retry), (status, etype, retry), fail)
            self.assertIn(msg, e.message)
            if fail.get("headers"):
                self.assertEqual(e.retry_after, 7.0)
        srv = self.mock(models=["deepseek-v4-pro"])
        with self.assertRaises(errors.GatewayError) as cm:
            run(make_ctx(provider_for(srv.url), "deepseek-v4-nope", turn1_body()))
        self.assertEqual((cm.exception.status, cm.exception.err_type), (404, "not_found_error"))

    def test_stream_error_before_and_after_content(self):
        srv = self.mock(stream_error={"type": "overloaded_error", "message": "Overloaded"})
        with self.assertRaises(errors.GatewayError) as cm:
            run(make_ctx(provider_for(srv.url), "deepseek-v4-pro", turn1_body()))
        self.assertEqual((cm.exception.status, cm.exception.err_type, cm.exception.should_retry),
                         (529, "overloaded_error", True))
        srv = self.mock(stream_error={"type": "invalid_request_error", "message": "bad thing"})
        with self.assertRaises(errors.GatewayError) as cm:
            run(make_ctx(provider_for(srv.url), "deepseek-v4-pro", turn1_body()))
        self.assertEqual((cm.exception.status, cm.exception.should_retry), (400, False))
        self.assertIn("bad thing", cm.exception.message)
        srv = self.mock(truncate=True)
        events = run(make_ctx(provider_for(srv.url), "deepseek-v4-pro", turn1_body()))
        self.assertEqual(len(events_of(events, ev.ToolCall)), 1)
        self.assertIsInstance(events[-1], ev.StreamError)
        self.assertTrue(events[-1].retryable)
        self.assertEqual(events[-1].err_type, "overloaded_error")

    def test_generator_close_releases_upstream(self):
        srv = self.mock()
        gen = pt.AnthropicPassthroughDialect().execute(make_ctx(provider_for(srv.url), "deepseek-v4-pro",
                                                                turn1_body()))
        self.assertIsInstance(next(gen), ev.ThinkingDelta)
        gen.close()  # must not raise


# =========================================================================================
# Ollama < 0.14 sticky fallback
# =========================================================================================

class FakeChat(dbase.Dialect):
    name = "openai_chat"

    def __init__(self):
        self.calls = []

    def execute(self, ctx):
        self.calls.append((ctx.provider.dialect, ctx.provider.profile, ctx.provider.base_url))
        yield ev.TextDelta(0, "via chat fallback")
        yield ev.Finish("end_turn")


class OllamaFallbackTests(unittest.TestCase):
    def test_404_switches_stickily(self):
        srv = mu.MockServer(kind="anthropic", options={"auth": "none", "old_ollama": True}).start()
        self.addCleanup(srv.stop)
        p = presets.provider_from_preset("ollama", {"base_url": srv.url,
                                                    "fallback": {"base_url": srv.url + "/v1"}})
        runtime = dbase.ProviderRuntime(p.id)
        fake = FakeChat()
        with mock.patch.dict(dialects._instances, {"openai_chat": fake}):
            with self.assertLogs("ai_gateway", level="WARNING") as logs:
                events = run(make_ctx(p, "qwen3:8b", turn1_body(model="claude-via-ollama,qwen3:8b"),
                                      runtime=runtime))
            self.assertIn("upgrade Ollama", "\n".join(logs.output))
            self.assertEqual(events, [ev.TextDelta(0, "via chat fallback"), ev.Finish("end_turn")])
            self.assertTrue(runtime.get(pt.FALLBACK_FLAG))
            self.assertEqual(fake.calls, [("openai_chat", "ollama", srv.url + "/v1")])
            self.assertEqual(len(srv.requests_for("/v1/messages")), 1)
            run(make_ctx(p, "qwen3:8b", turn1_body(model="qwen3:8b"), runtime=runtime))
            self.assertEqual(len(srv.requests_for("/v1/messages")), 1)  # sticky: no new passthrough attempt
            self.assertEqual(len(fake.calls), 2)

    def test_model_404_and_non_ollama_404_do_not_fall_back(self):
        srv = mu.MockServer(kind="anthropic", options={"auth": "none", "models": ["llama3.3"]}).start()
        self.addCleanup(srv.stop)
        p = presets.provider_from_preset("ollama", {"base_url": srv.url})
        runtime = dbase.ProviderRuntime(p.id)
        with self.assertRaises(errors.GatewayError) as cm:
            run(make_ctx(p, "missing", turn1_body(model="missing"), runtime=runtime))
        self.assertEqual(cm.exception.status, 404)
        self.assertFalse(runtime.get(pt.FALLBACK_FLAG))
        old = mu.MockServer(kind="anthropic", options={"auth": "bearer", "old_ollama": True}).start()
        self.addCleanup(old.stop)
        ds = provider_for(old.url, overrides={"fallback": {"dialect": "openai_chat", "base_url": old.url + "/v1"}})
        runtime = dbase.ProviderRuntime(ds.id)
        with self.assertRaises(errors.GatewayError) as cm:
            run(make_ctx(ds, "deepseek-v4-pro", turn1_body(), runtime=runtime))
        self.assertEqual(cm.exception.status, 404)
        self.assertFalse(runtime.get(pt.FALLBACK_FLAG))


# =========================================================================================
# integration with the other components (skipped while they are unavailable)
# =========================================================================================

def _optional(name):
    try:
        return mod(name)
    except ImportError as exc:
        raise unittest.SkipTest("%s not available yet: %s" % (name, exc))


class IntegrationTests(unittest.TestCase):
    def test_ollama_fallback_with_real_openai_chat(self):
        ain = _optional("anthropic_in")
        _optional("dialects.openai_chat")
        _optional("testing.mock_openai_chat")
        old = mu.MockServer(kind="anthropic", options={"auth": "none", "old_ollama": True}).start()
        self.addCleanup(old.stop)
        chat = mu.MockServer(kind="ollama_chat").start()
        self.addCleanup(chat.stop)
        p = presets.provider_from_preset("ollama", {"base_url": old.url, "fallback": {"base_url": chat.url + "/v1"}})
        runtime = dbase.ProviderRuntime(p.id)
        with self.assertLogs("ai_gateway", level="WARNING"):
            for _ in range(2):
                ctx = make_ctx(p, "qwen3:8b", turn1_body(model="claude-via-ollama,qwen3:8b"), runtime=runtime)
                ctx.req = ain.parse_messages_request(ctx.req.raw, HEADERS)
                events = run(ctx)
                self.assertEqual(events_of(events, ev.ToolCall)[0].name, "Bash")
        self.assertEqual(len(old.requests_for("/v1/messages")), 1)
        self.assertEqual(len(chat.requests_for("/v1/chat/completions", "POST")), 2)
        self.assertEqual(old.errors + chat.errors, [])

    def test_gateway_stream_false_and_beta_filtering(self):
        server = _optional("server")
        srv = mu.MockServer(kind="anthropic", options={"key": KEY}).start()
        self.addCleanup(srv.stop)
        table = presets.route_table_from_presets(["kimi"])
        with mock.patch.dict("os.environ", {config.upstream_env_name("kimi"): srv.url}):
            gw = server.Gateway(table, config.SecretStore({"kimi": KEY})).start()
        self.addCleanup(gw.stop)
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", gw.port, timeout=30)
        self.addCleanup(conn.close)
        body = {"model": "claude-via-kimi,kimi-k3[1m]", "max_tokens": 32000, "stream": False,
                "messages": [user("hi")], "tools": copy.deepcopy(TOOLS)}
        conn.request("POST", "/v1/messages?beta=true", json.dumps(body),
                     {"Authorization": "Bearer " + gw.token, "Content-Type": "application/json",
                      "anthropic-version": "2023-06-01", "anthropic-beta": BETAS})
        resp = conn.getresponse()
        msg = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(resp.status, 200, msg)
        self.assertEqual(msg["model"], "claude-via-kimi,kimi-k3[1m]")
        self.assertEqual([b["type"] for b in msg["content"]], ["thinking", "tool_use"])
        self.assertTrue(msg["content"][0]["signature"].startswith(ma.DEFAULT_SIGNATURE_PREFIX))
        self.assertEqual(msg["stop_reason"], "tool_use")
        rec = srv.last_request()
        self.assertEqual((rec["body_json"]["model"], rec["body_json"]["stream"]), ("kimi-k3", True))
        self.assertNotIn("mid-conversation", rec["headers"].get("anthropic-beta", ""))
        self.assertEqual(srv.errors, [])


# =========================================================================================
# stream translation edge cases (records fed directly)
# =========================================================================================

class TranslatorTests(unittest.TestCase):
    def ctx(self):
        return make_ctx(provider_for("https://x.example/anthropic"), "deepseek-v4-pro", turn1_body())

    def feed(self, records):
        tr = pt._Translator(self.ctx())
        out = []
        for r in records:
            out.extend(tr.feed(r))
        return out + tr.finish()

    def test_dropped_blocks_ids_and_stop_mapping(self):
        with self.assertLogs("ai_gateway", level="WARNING") as logs:
            out = self._feed_mixed()
        self.assertEqual(len(logs.output), 2)  # redacted_thinking + server_tool_use, each logged once
        self.assertEqual(out[:2], [ev.TextDelta(2, "Hi"), ev.TextDelta(2, " there")])
        self.assertEqual(out[2].name, "Read")
        self.assertEqual(json.loads(out[2].input_json), {"file_path": "/a"})
        self.assertRegex(out[2].id, r"^toolu_[0-9a-f]{24}$")
        self.assertEqual(out[3], ev.Usage(12, 5, 4, 0))
        self.assertEqual(out[4], ev.Finish("max_tokens"))

    def _feed_mixed(self):
        return self.feed([
            {"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 4}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "redacted_thinking", "data": "x"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "server_tool_use", "id": "s"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": "Hi"}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": " there"}},
            {"type": "content_block_stop", "index": 2},
            {"type": "content_block_start", "index": 3,
             "content_block": {"type": "tool_use", "name": "Read", "input": {"file_path": "/a"}}},
            {"type": "content_block_stop", "index": 3},
            {"type": "message_delta", "delta": {"stop_reason": "model_context_window_exceeded"},
             "usage": {"output_tokens": 5, "input_tokens": 12}},
            {"type": "message_stop"}])

    def test_malformed_and_truncated(self):
        with self.assertRaises(errors.GatewayError) as cm:
            self.feed([{"type": "content_block_start", "index": 0,
                        "content_block": {"type": "tool_use", "id": "t", "name": "Bash", "input": {}}},
                       {"type": "content_block_delta", "index": 0,
                        "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}},
                       {"type": "content_block_stop", "index": 0}])
        self.assertTrue(cm.exception.should_retry)
        with self.assertRaises(errors.GatewayError):
            self.feed([{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                       {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}}])
        out = self.feed([{"type": "message_delta", "delta": {"stop_reason": "stop_sequence",
                                                             "stop_sequence": "END"}}])
        self.assertEqual(out, [ev.Finish("stop_sequence", "END")])

    def test_message_records_roundtrip(self):
        msg = {"content": [{"type": "thinking", "thinking": "hm", "signature": "S"},
                           {"type": "text", "text": "yo"},
                           {"type": "tool_use", "id": "toolu_9", "name": "Bash", "input": {"command": "ls"}}],
               "stop_reason": "tool_use", "usage": {"input_tokens": 3, "output_tokens": 4}}
        out = self.feed(list(pt._message_records(msg)))
        self.assertEqual(out, [ev.ThinkingDelta(0, "hm"), ev.ThinkingSignature(0, "S"), ev.TextDelta(1, "yo"),
                               ev.ToolCall("toolu_9", "Bash", '{"command":"ls"}'), ev.Usage(3, 4),
                               ev.Finish("tool_use")])


if __name__ == "__main__":
    unittest.main()
