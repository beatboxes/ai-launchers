"""`doctor` static checks and `doctor --live` (get_magic(n=7) -> 42 tool round trip) against mock upstreams."""

import json
import os
import stat
import threading
import unittest

from ._util import SENTINEL, LauncherTestCase, base_launcher, fake_key, mock_kind_or_skip
from shared.gateway.testing.mock_upstreams import Brain, BrainReply, MockServer


class MagicBrain(Brain):
    """Turn 1: call get_magic(n); turn 2: report the tool result."""

    def __init__(self, n=7):
        Brain.__init__(self, tool="get_magic", thinking="I should call get_magic.")
        self.n = n

    def decide(self, offered_tools, tool_results, background=False):
        if not offered_tools:
            reply = BrainReply("text", text=self.background_text)
        elif not tool_results:
            reply = BrainReply("tool_call", tool_name=self.find_tool(offered_tools), arguments={"n": self.n},
                               thinking=self.thinking)
        else:
            reply = BrainReply("text", text="The magic number is %s." % tool_results[-1])
        self.decisions.append(reply)
        return reply


def _anthropic_sse_handler(n):
    """Minimal Anthropic Messages SSE upstream for live_round_trip unit tests (framework-only mock)."""
    def handle(server, req, resp):
        body = req.json or {}
        blocks = [b for m in body.get("messages", []) if isinstance(m.get("content"), list) for b in m["content"]]
        results = [b for b in blocks if b.get("type") == "tool_result"]
        w = resp.start_sse()
        w.event("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": body.get("model"), "content": [],
            "usage": {"input_tokens": 11, "output_tokens": 0}}})
        if not results:
            w.event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {
                "type": "tool_use", "id": "toolu_1", "name": "get_magic", "input": {}}})
            for part in ('{"n"', ': %d}' % n):
                w.event("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                "delta": {"type": "input_json_delta", "partial_json": part}})
            stop = "tool_use"
        else:
            w.event("content_block_start", {"type": "content_block_start", "index": 0,
                                            "content_block": {"type": "text", "text": ""}})
            text = "It is %s." % results[0]["content"]
            w.event("content_block_delta", {"type": "content_block_delta", "index": 0,
                                            "delta": {"type": "text_delta", "text": text}})
            stop = "end_turn"
        w.event("content_block_stop", {"type": "content_block_stop", "index": 0})
        w.event("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop},
                                  "usage": {"output_tokens": 5}})
        w.event("message_stop", {"type": "message_stop"})
        w.close()
    return handle


class LiveRoundTripUnitTests(LauncherTestCase):
    def test_round_trip_ok(self):
        with MockServer(handler=_anthropic_sse_handler(7)) as mock:
            info = base_launcher.live_round_trip(mock.url + "/v1/messages", {"Authorization": "Bearer x"}, "m")
            self.assertEqual(len(mock.requests), 2)
            second = mock.requests[1]["body_json"]
        self.assertIn("42", info["text"])
        self.assertEqual(info["input_tokens"], 22)
        self.assertEqual(second["messages"][1]["content"][0]["input"], {"n": 7})
        self.assertEqual(second["messages"][2]["content"][0]["content"], "42")
        first_tools = second["tools"]
        self.assertEqual(first_tools[0]["name"], "get_magic")

    def test_round_trip_wrong_argument(self):
        with MockServer(handler=_anthropic_sse_handler(8)) as mock:
            with self.assertRaises(base_launcher.LaunchError) as cm:
                base_launcher.live_round_trip(mock.url + "/v1/messages", {}, "m")
        self.assertIn("expected 7", str(cm.exception))

    def test_http_error(self):
        def handle(server, req, resp):
            resp.send_json(401, {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}})
        with MockServer(handler=handle) as mock:
            with self.assertRaises(base_launcher.LaunchError) as cm:
                base_launcher.live_round_trip(mock.url + "/v1/messages", {}, "m")
        self.assertIn("HTTP 401: bad key", str(cm.exception))


class DoctorStaticTests(LauncherTestCase):
    def test_static_ok_with_warnings(self):
        os.environ["XAI_API_KEY"] = fake_key("xai")
        os.environ["HTTPS_PROXY"] = "http://user:secretpw@proxy.example:3128"
        self.install_stub_claude(fetch=False)
        self.write_json(os.path.join(self.home, ".claude", "settings.json"),
                        {"env": {"ANTHROPIC_BASE_URL": "https://x", "CLAUDE_CODE_USE_BEDROCK": "1"},
                         "apiKeyHelper": "/bin/echo"})
        before = set(threading.enumerate())
        rc, out, err = self.run_cli("grok", "doctor")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(set(threading.enumerate()), before)
        self.assertIn("version 2.1.288", out)
        self.assertIn("[ok] xai (api-key): available via env:XAI_API_KEY", out)
        self.assertIn("[warn] grok (login): unavailable", out)
        self.assertIn("ANTHROPIC_BASE_URL, CLAUDE_CODE_USE_BEDROCK", out)
        self.assertIn("apiKeyHelper", out)
        self.assertIn("HTTPS_PROXY=http://***@proxy.example:3128", out)
        self.assertNotIn("secretpw", out)
        self.assertNotIn(SENTINEL, out + err)
        self.assertIn("all checks passed", out)

    def test_failures(self):
        os.environ["AI_LAUNCHERS_CLAUDE_BIN"] = os.path.join(self.tmp, "missing")
        rc, out, _ = self.run_cli("codex", "doctor")
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] ", out)
        self.assertIn("no usable transport", out)
        self.assertIn("2 problem(s)", out)

    def test_credentials_permissions_and_adc_project(self):
        if os.name == "nt":
            self.skipTest("POSIX permission bits")
        self.install_stub_claude(fetch=False)
        creds = self.write_json(os.path.join(self.ail, "credentials.json"), {"gemini-wrap": {"key": fake_key("g")}})
        os.chmod(creds, 0o644)
        self.write_json(os.path.join(self.home, ".config", "gcloud", "application_default_credentials.json"),
                        {"type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r",
                         "quota_project_id": "my-proj"})
        rc, out, _ = self.run_cli("gemini", "doctor")
        self.assertEqual(rc, 0, out)
        self.assertIn("readable by other users", out)
        self.assertIn("gemini-vertex: project my-proj", out)
        self.assertIn("gemini-vertex: location global", out)
        self.assertEqual(stat.S_IMODE(os.stat(creds).st_mode), 0o644, "doctor must not modify files")
        self.write_config({"providers": {"gemini-vertex": {"options": {"project": "cfg-proj",
                                                                        "location": "us-central1"}}}})
        rc, out, _ = self.run_cli("gemini", "doctor", "--auth", "adc")
        self.assertEqual(rc, 0, out)
        self.assertIn("gemini-vertex: project cfg-proj", out)
        self.assertIn("gemini-vertex: location us-central1", out)


class DoctorLiveTests(LauncherTestCase):
    def live(self, launcher, kind, upstream_var, suffix, options, brain, *extra):
        mock_kind_or_skip(self, kind)
        self.install_stub_claude(fetch=False)
        with MockServer(kind, brain=brain, options=options) as mock:
            os.environ[upstream_var] = mock.url + suffix
            rc, out, err = self.run_cli(launcher, "doctor", "--live", *extra)
            return rc, out, err, list(mock.requests), list(mock.errors)

    def test_gateway_route_ok(self):
        key = fake_key("xai")
        os.environ["XAI_API_KEY"] = key
        rc, out, err, reqs, errors = self.live("grok", "xai_chat", "AI_GATEWAY_UPSTREAM_XAI", "/v1",
                                               {"api_key": key}, MagicBrain(7))
        self.assertEqual(errors, [])
        self.assertEqual(rc, 0, out + err)
        self.assertIn("[ok] claude-via-xai,grok-4.7[1m]", out)
        self.assertIn("PAID", out)
        chats = [r for r in reqs if r["path"].endswith("/chat/completions")]
        self.assertEqual(len(chats), 2)
        self.assertTrue(all(r["body_json"]["stream"] for r in chats))
        self.assertEqual([t["function"]["name"] for t in chats[0]["body_json"]["tools"]], ["get_magic"])
        self.assertNotIn(SENTINEL, out + err)

    def test_gateway_route_wrong_tool_argument(self):
        key = fake_key("xai")
        os.environ["XAI_API_KEY"] = key
        rc, out, err, _, _ = self.live("grok", "xai_chat", "AI_GATEWAY_UPSTREAM_XAI", "/v1", {"api_key": key},
                                       MagicBrain(8), "--model", "grok-4.3")
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] claude-via-xai,grok-4.3[1m]", out)
        self.assertIn("expected 7", out)

    def test_direct_route_ok(self):
        key = fake_key("deepseek")
        os.environ["DEEPSEEK_API_KEY"] = key
        rc, out, err, reqs, errors = self.live("deepseek", "anthropic", "AI_GATEWAY_UPSTREAM_DEEPSEEK", "",
                                               {"key": key}, MagicBrain(7))
        self.assertEqual(errors, [])
        self.assertEqual(rc, 0, out + err)
        self.assertIn("[ok] deepseek-v4-pro", out)
        msgs = [r for r in reqs if r["path"].endswith("/v1/messages")]
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["body_json"]["model"], "deepseek-v4-pro")
        self.assertEqual(msgs[0]["headers"].get("Authorization") or msgs[0]["headers"].get("authorization"),
                         "Bearer " + key)

    def test_live_failure_reports_upstream_error(self):
        key = fake_key("openai")
        os.environ["OPENAI_API_KEY"] = key
        rc, out, err, _, _ = self.live("codex", "openai_responses", "AI_GATEWAY_UPSTREAM_OPENAI", "/v1",
                                       {"api_key": "a-different-key"}, MagicBrain(7), "--auth", "api-key")
        self.assertEqual(rc, 1, out + err)
        self.assertIn("[FAIL] claude-via-openai,gpt-5.5", out)
        self.assertIn("401", out)
        self.assertNotIn(SENTINEL, out + err)


if __name__ == "__main__":
    unittest.main()
