"""End-to-end with the REAL installed Claude Code (DESIGN §7). Opt-in: ``RUN_E2E=1``.

Claude Code -> in-process Gateway -> anthropic_passthrough -> ``anthropic`` mock upstream (pointed at via
``AI_GATEWAY_UPSTREAM_DEEPSEEK``). The mock's Brain makes Claude Code run ``printf hello > out.txt && env``
with its Bash tool, verifies the tool output carries no key sentinel, and answers ``DONE <sha8>``.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from ._pkg import mod

RUN_E2E = os.environ.get("RUN_E2E") == "1"


@unittest.skipUnless(RUN_E2E, "set RUN_E2E=1 to run the real Claude Code end-to-end test")
class ClaudeCodePassthroughE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e2e = mod("testing.claude_e2e")
        if not cls.e2e.claude_available():
            raise unittest.SkipTest("Claude Code (`claude`) is not installed")
        try:
            cls.server = mod("server")
            cls.server.Gateway
        except (ImportError, AttributeError) as exc:
            raise unittest.SkipTest("gateway server.py (agent A) is not available yet: %s" % exc)
        cls.config = mod("config")
        cls.presets = mod("presets")
        cls.launchkit = mod("launchkit")
        cls.mu = mod("testing.mock_upstreams")
        mod("testing.mock_anthropic")
        cls.secret = mod("testing").SENTINEL + "-deepseek-e2e-key"

    def test_tool_round_trip_through_passthrough(self):
        config, presets = self.config, self.presets
        srv = self.mu.MockServer(kind="anthropic", options={"auth": "bearer", "key": self.secret,
                                                            "models": ["deepseek-v4-pro", "deepseek-v4-flash"]})
        srv.start()
        self.addCleanup(srv.stop)
        table = presets.route_table_from_presets(["deepseek"])
        secrets = config.SecretStore({"deepseek": self.secret})
        with mock.patch.dict(os.environ, {config.upstream_env_name("deepseek"): srv.url}):
            gw = self.server.Gateway(table, secrets).start()
        self.addCleanup(gw.stop)

        spec = table.providers["deepseek"].model_spec("deepseek-v4-pro")
        default_id = table.picker_id("deepseek", spec)
        base = dict(os.environ, DEEPSEEK_API_KEY=self.secret)  # must be scrubbed from the child env
        env = self.launchkit.build_child_env(base, gw.url, gw.token, default_id, "claude-via-background",
                                             context=spec.context, secret_values=secrets)
        tmp = tempfile.mkdtemp(prefix="gw-e2e-")
        self.addCleanup(shutil.rmtree, tmp, True)
        work = os.path.join(tmp, "work")
        res = self.e2e.run_e2e(None, env, work, timeout=300, home=os.path.join(tmp, "home"))
        info = res.summary() + "\n--- mock errors ---\n" + "\n".join(srv.errors)

        self.assertFalse(res.timed_out, info)
        self.assertEqual(res.rc, 0, info)
        self.assertIn("DONE", res.result_text or "", info)
        self.assertFalse(res.result_event.get("is_error"), info)
        with open(os.path.join(work, "out.txt"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "hello")
        self.assertEqual(srv.brain.leaks, [], "provider key leaked into the Bash tool environment")
        self.assertEqual(srv.errors, [], info)

        posts = [r for r in srv.requests_for("", "POST") if r["path"].endswith("/v1/messages")]
        self.assertGreaterEqual(len(posts), 2, info)
        for r in posts:
            self.assertEqual(r["headers"].get("Authorization"), "Bearer " + self.secret)
            self.assertIs(r["body_json"]["stream"], True)
            self.assertIn(r["body_json"]["model"], ("deepseek-v4-pro", "deepseek-v4-flash"))
            self.assertNotIn("system", [m.get("role") for m in r["body_json"]["messages"]])
        # turn 2 echoed the upstream's real thinking signature (the mock rejects anything else)
        echoed = [b for m in posts[-1]["body_json"]["messages"] if isinstance(m.get("content"), list)
                  for b in m["content"] if b.get("type") == "thinking"]
        self.assertTrue(echoed and all(b["signature"].startswith("mocksig-") for b in echoed), echoed)

        cache = os.path.join(res.home, ".claude", "cache", "gateway-models.json")
        self.assertTrue(os.path.isfile(cache), "GET /v1/models was not fetched/cached by Claude Code")
        with open(cache, "r", encoding="utf-8") as f:
            models = json.load(f)
        self.assertEqual(models.get("baseUrl", "").rstrip("/"), gw.url)
        self.assertIn(default_id, [m.get("id") for m in models.get("models", [])])
        self.assertTrue(res.settings_unchanged)
        self.assertTrue(res.claude_json_user_entry_intact)
        self.assertFalse(res.claude_json_has_fry_entries)


if __name__ == "__main__":
    unittest.main()
