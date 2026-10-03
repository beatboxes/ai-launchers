"""responses_targets: per-target URLs, headers, body rules and CLI version detection (DESIGN §3.2)."""

import os
import re
import shutil
import stat
import tempfile
import time
import unittest
import uuid

from ._pkg import mod

rt = mod("responses_targets")
config = mod("config")
model = mod("model")
dbase = mod("dialects.base")

PROBE_ENV = {"PATH": ""}


def _req(**kw):
    kw.setdefault("model", "claude-via-p,m")
    return model.NormalizedRequest(**kw)


class TargetTableTests(unittest.TestCase):
    def test_lookup_and_tags(self):
        tags = {n: rt.get_target(n).sig_tag for n in ("openai_api", "chatgpt_codex", "grok_cli_proxy", "xai_api")}
        self.assertEqual(tags, {"openai_api": "openai_api", "chatgpt_codex": "codex_chatgpt",
                                "grok_cli_proxy": "grok_cli", "xai_api": "xai_api"})
        self.assertIs(rt.get_target(None), rt.get_target("openai_api"))
        with self.assertRaises(ValueError):
            rt.get_target("nope")
        self.assertEqual(set(rt.TARGETS), set(config.DIALECT_TARGETS["responses"]))
        self.assertTrue(rt.get_target("grok_cli_proxy").fallback_capable)
        self.assertFalse(rt.get_target("xai_api").fallback_capable)

    def test_urls(self):
        self.assertEqual(rt.get_target("chatgpt_codex").url(""), "https://chatgpt.com/backend-api/codex/responses")
        self.assertEqual(rt.get_target("openai_api").url(None), "https://api.openai.com/v1/responses")
        self.assertEqual(rt.get_target("grok_cli_proxy").url(""), "https://cli-chat-proxy.grok.com/v1/responses")
        self.assertEqual(rt.get_target("xai_api").url("http://127.0.0.1:9/v1/"), "http://127.0.0.1:9/v1/responses")
        self.assertEqual(rt.get_target("openai_api").url("https://opencode.ai/zen/v1", "/responses"),
                         "https://opencode.ai/zen/v1/responses")
        self.assertEqual(rt.get_target("xai_api").url("https://x", "custom/path"), "https://x/custom/path")

    def test_reasoning_and_lite(self):
        oa, cx, gk = rt.get_target("openai_api"), rt.get_target("chatgpt_codex"), rt.get_target("grok_cli_proxy")
        self.assertTrue(oa.is_reasoning(config.ModelSpec("gpt-5.5")))
        self.assertTrue(oa.is_reasoning(config.ModelSpec("o3-pro")))
        self.assertFalse(oa.is_reasoning(config.ModelSpec("gpt-4.1")))
        self.assertTrue(oa.is_reasoning(config.ModelSpec("custom", reasoning=True)))
        self.assertTrue(gk.is_reasoning(config.ModelSpec("grok-4.7")))
        self.assertFalse(gk.is_reasoning(config.ModelSpec("grok-4.20-0309-non-reasoning", context=2000000)))
        self.assertTrue(cx.is_lite(config.ModelSpec("gpt-6.1")))
        self.assertTrue(cx.is_lite(config.ModelSpec("custom", responses_lite=True)))
        self.assertFalse(cx.is_lite(config.ModelSpec("gpt-5.5")))
        self.assertFalse(oa.is_lite(config.ModelSpec("gpt-6.1")))

    def test_unauthorized_predicate(self):
        cx = rt.get_target("chatgpt_codex")
        self.assertTrue(cx.unauthorized(403, '{"detail":{"code":"token_expired"}}'))
        self.assertTrue(cx.unauthorized(403, '{"error":{"code":"invalid_token"}}'))
        self.assertFalse(cx.unauthorized(403, '{"detail":"forbidden"}'))
        self.assertFalse(cx.unauthorized(400, "token_expired"))
        self.assertFalse(rt.get_target("openai_api").unauthorized(403, "token_expired"))


class HeaderTests(unittest.TestCase):
    def test_codex_headers(self):
        env = {"AI_GATEWAY_CODEX_VERSION": "0.171.2", "TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.5.1"}
        h = rt.get_target("chatgpt_codex").headers("sess-1", lite=False, environ=env)
        self.assertEqual(h["originator"], "codex_cli_rs")
        self.assertEqual(h["version"], "0.171.2")
        self.assertEqual(h["session-id"], "sess-1")
        self.assertEqual(h["Accept"], "text/event-stream")
        self.assertEqual(h["Content-Type"], "application/json")
        self.assertRegex(h["User-Agent"], r"^codex_cli_rs/0\.171\.2 \(\S.* \S+; \S+\) iTerm\.app/3\.5\.1$")
        self.assertNotIn(rt.LITE_HEADER, h)
        h = rt.get_target("chatgpt_codex").headers("sess-1", lite=True, environ=env)
        self.assertEqual(h[rt.LITE_HEADER], "true")
        for v in h.values():
            self.assertTrue(all(" " <= ch <= "~" for ch in v), v)

    def test_grok_headers(self):
        env = {"AI_GATEWAY_GROK_CLIENT_VERSION": "2.0.1"}
        t = rt.get_target("grok_cli_proxy")
        h1 = t.headers("conv-9", environ=env)
        h2 = t.headers("conv-9", environ=env, client_identifier="my-id")
        expected = {"X-XAI-Token-Auth": "xai-grok-cli", "x-authenticateresponse": "authenticate-response",
                    "x-grok-client-version": "2.0.1", "x-grok-client-identifier": rt.GROK_CLIENT_IDENTIFIER,
                    "x-grok-client-mode": "headless", "x-grok-conv-id": "conv-9", "x-grok-session-id": "conv-9"}
        for k, v in expected.items():
            self.assertEqual(h1[k], v, k)
        self.assertEqual(h2["x-grok-client-identifier"], "my-id")
        self.assertNotEqual(h1["x-grok-req-id"], h2["x-grok-req-id"])
        uuid.UUID(h1["x-grok-req-id"])

    def test_api_targets_minimal(self):
        h = rt.get_target("openai_api").headers("s", environ=PROBE_ENV)
        self.assertEqual(h, {"Content-Type": "application/json", "Accept": "text/event-stream"})
        h = rt.get_target("xai_api").headers("s", environ=PROBE_ENV)
        self.assertEqual(h["x-grok-conv-id"], "s")
        self.assertNotIn("X-XAI-Token-Auth", h)

    def test_header_injection_is_neutralized(self):
        env = {"AI_GATEWAY_CODEX_VERSION": "1.0\r\nX-Evil: 1"}
        h = rt.get_target("chatgpt_codex").headers("a\nb", environ=env)
        self.assertNotIn("\n", h["version"] + h["session-id"] + h["User-Agent"])

    def test_terminal_user_agent(self):
        self.assertEqual(rt.terminal_user_agent({"TERM_PROGRAM": "vscode"}), "vscode")
        self.assertEqual(rt.terminal_user_agent({"WEZTERM_VERSION": "2024"}), "WezTerm/2024")
        self.assertEqual(rt.terminal_user_agent({"WT_SESSION": "x", "TERM": "xterm"}), "WindowsTerminal")
        self.assertEqual(rt.terminal_user_agent({"TERM": "xterm-256color"}), "xterm-256color")
        self.assertEqual(rt.terminal_user_agent({}), "unknown")


class BodyRuleTests(unittest.TestCase):
    def _body(self):
        return {"model": "m", "instructions": "SYS", "input": [{"type": "message", "role": "user", "content": []}],
                "truncation": "auto", "user": "u", "metadata": {}, "previous_response_id": "r",
                "max_completion_tokens": 5, "max_tokens": 5}

    def test_openai_api(self):
        t = rt.get_target("openai_api")
        req = _req(max_tokens=32000, temperature=0.3, top_p=0.9)
        b = t.apply_body_rules(self._body(), req, config.ModelSpec("gpt-5.5", max_output=128000), reasoning=True)
        self.assertEqual(b["max_output_tokens"], 32000)
        self.assertNotIn("temperature", b)
        self.assertNotIn("top_p", b)
        b = t.apply_body_rules({}, req, config.ModelSpec("gpt-4.1", max_output=16000), reasoning=False)
        self.assertEqual((b["max_output_tokens"], b["temperature"], b["top_p"]), (16000, 0.3, 0.9))
        b = t.apply_body_rules({}, _req(max_tokens=1), config.ModelSpec("gpt-5.5"), reasoning=True)
        self.assertEqual(b["max_output_tokens"], rt.MIN_OUTPUT_TOKENS)

    def test_chatgpt_codex(self):
        t = rt.get_target("chatgpt_codex")
        body = self._body()
        body.update(max_output_tokens=10, temperature=1, top_p=1)
        b = t.apply_body_rules(body, _req(temperature=0.5, top_p=0.5), config.ModelSpec("gpt-5.5"), True)
        for key in ("max_output_tokens", "temperature", "top_p", "truncation", "user", "metadata",
                    "previous_response_id", "max_completion_tokens"):
            self.assertNotIn(key, b)
        self.assertEqual(b["instructions"], "SYS")
        lite = t.apply_body_rules(self._body(), _req(), config.ModelSpec("gpt-6.1"), True, lite=True)
        self.assertNotIn("instructions", lite)
        self.assertEqual(lite["input"][0], {"type": "message", "role": "developer",
                                            "content": [{"type": "input_text", "text": "SYS"}]})
        self.assertEqual(lite["input"][1]["role"], "user")

    def test_grok_targets(self):
        req = _req(max_tokens=100000, temperature=0.2, top_p=0.7)
        b = rt.get_target("grok_cli_proxy").apply_body_rules({}, req, config.ModelSpec("grok-4.7", max_output=64000),
                                                              True)
        self.assertEqual(b, {"top_p": 0.7})
        b = rt.get_target("xai_api").apply_body_rules({}, req, config.ModelSpec("grok-4.7", max_output=64000), True)
        self.assertEqual(b, {"max_output_tokens": 64000, "temperature": 0.2, "top_p": 0.7})


def _write_stub(directory, name, body_posix, body_windows):
    """A fake CLI on PATH: POSIX sh script (chmod +x) or a .cmd twin on Windows."""
    if os.name == "nt":
        path = os.path.join(directory, name + ".cmd")
        with open(path, "w", encoding="utf-8") as f:
            f.write("@echo off\r\n" + body_windows + "\r\n")
    else:
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\n" + body_posix + "\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class VersionTests(unittest.TestCase):
    def setUp(self):
        rt.clear_version_cache()
        self.tmp = tempfile.mkdtemp(prefix="gw-versions-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(rt.clear_version_cache)
        self.env = {"PATH": self.tmp}

    def _counter_stub(self, name, output):
        counter = os.path.join(self.tmp, name + ".count")
        _write_stub(self.tmp, name, 'echo x >> "%s"\necho "%s"' % (counter, output),
                    'echo x>>"%s"\r\necho %s' % (counter, output))
        return counter

    @staticmethod
    def _count(path):
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            return len([ln for ln in f if ln.strip()])

    def test_env_override_wins(self):
        runtime = dbase.ProviderRuntime("p")
        runtime.setdefault("versions", dict)["codex"] = "0.1.0"
        env = dict(self.env, AI_GATEWAY_CODEX_VERSION=" 9.9.9 ", AI_GATEWAY_GROK_CLIENT_VERSION="3.2.1")
        self.assertEqual(rt.codex_version(runtime, env), "9.9.9")
        self.assertEqual(rt.grok_client_version(None, env), "3.2.1")

    def test_detects_and_caches_codex(self):
        counter = self._counter_stub("codex", "codex-cli 0.170.3")
        runtime = dbase.ProviderRuntime("codex")
        self.assertEqual(rt.codex_version(runtime, self.env), "0.170.3")
        self.assertEqual(runtime.get("versions"), {"codex": "0.170.3"})
        self.assertEqual(rt.codex_version(runtime, self.env), "0.170.3")
        self.assertEqual(rt.codex_version(dbase.ProviderRuntime("other"), self.env), "0.170.3")  # module cache
        self.assertEqual(rt.codex_version(None, self.env), "0.170.3")
        self.assertEqual(self._count(counter), 1)

    def test_detects_grok(self):
        self._counter_stub("grok", "grok 1.4.2-beta.1 (build abc)")
        self.assertEqual(rt.grok_client_version(None, self.env), "1.4.2-beta.1")

    def test_missing_or_broken_binary_uses_pinned(self):
        self.assertEqual(rt.codex_version(None, self.env), rt.CODEX_PINNED_VERSION)
        self.assertEqual(rt.grok_client_version(None, self.env), rt.GROK_PINNED_VERSION)
        rt.clear_version_cache()
        _write_stub(self.tmp, "codex", 'echo "no version here"', "echo no version here")
        self.assertEqual(rt.codex_version(None, self.env), rt.CODEX_PINNED_VERSION)
        rt.clear_version_cache()
        _write_stub(self.tmp, "grok", 'echo "grok 1.2.3"; exit 3', "echo grok 1.2.3\r\nexit /b 3")
        self.assertEqual(rt.grok_client_version(None, self.env), rt.GROK_PINNED_VERSION)

    def test_hanging_binary_times_out(self):
        _write_stub(self.tmp, "codex", "exec sleep 20", "ping -n 21 127.0.0.1 >nul")
        old = rt.VERSION_TIMEOUT
        rt.VERSION_TIMEOUT = 0.5
        self.addCleanup(setattr, rt, "VERSION_TIMEOUT", old)
        t0 = time.time()
        self.assertEqual(rt.codex_version(None, self.env), rt.CODEX_PINNED_VERSION)
        self.assertLess(time.time() - t0, 10)

    def test_detect_cli_version_direct(self):
        self._counter_stub("codex", "codex-cli 0.200.0")
        self.assertEqual(rt.detect_cli_version("codex", self.env, timeout=5), "0.200.0")
        self.assertIsNone(rt.detect_cli_version("definitely-not-a-binary-" + uuid.uuid4().hex[:6], self.env))


class UserAgentTests(unittest.TestCase):
    def test_shape(self):
        ua = rt.codex_user_agent("0.160.0", {"TERM": "xterm-256color"})
        m = re.match(r"^codex_cli_rs/0\.160\.0 \((.+) (\S+); (\S+)\) xterm-256color$", ua)
        self.assertIsNotNone(m, ua)


if __name__ == "__main__":
    unittest.main()
