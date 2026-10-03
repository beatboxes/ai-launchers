"""End to end through the REAL launcher entry points and the REAL installed Claude Code (opt-in: RUN_E2E=1).

`python <x>/<x>-wrap.py launch claude -- -p … --permission-mode dontAsk …` with each launcher's API-key
transport pointed at a quirk-enforcing mock upstream (AI_GATEWAY_UPSTREAM_<ID>). The mock Brain makes
Claude Code run `printf hello > out.txt && env` with its Bash tool, checks that no key sentinel leaked
into the tool output, and answers `DONE <sha8>`; the test also checks the Bash `env` output itself (it
must show the gateway variables and never the provider key). Login/ADC variants are covered by the
gateway E2E matrix.
"""

import os
import shutil
import sys
import tempfile
import unittest

from ._util import REPO_ROOT, SENTINEL, mock_kind_or_skip

RUN_E2E = os.environ.get("RUN_E2E") == "1"

# launcher -> (mock kind, key env var, upstream override var, path suffix, mock options key)
CASES = {
    "grok": ("xai_chat", "XAI_API_KEY", "AI_GATEWAY_UPSTREAM_XAI", "/v1", "api_key"),
    "codex": ("openai_responses", "OPENAI_API_KEY", "AI_GATEWAY_UPSTREAM_OPENAI", "/v1", "api_key"),
    "gemini": ("gemini_api", "GEMINI_API_KEY", "AI_GATEWAY_UPSTREAM_GEMINI", "", "api_key"),
    "deepseek": ("anthropic", "DEEPSEEK_API_KEY", "AI_GATEWAY_UPSTREAM_DEEPSEEK", "", "key"),
    "kimi": ("anthropic", "MOONSHOT_API_KEY", "AI_GATEWAY_UPSTREAM_KIMI", "", "key"),
}


def _tool_results(events):
    """Texts of every tool_result Claude Code sent back (stream-json ``user`` events)."""
    out = []
    for ev in events:
        content = (ev.get("message") or {}).get("content") if ev.get("type") == "user" else None
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                c = block.get("content")
                if isinstance(c, list):
                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                out.append(c if isinstance(c, str) else "")
    return out


@unittest.skipUnless(RUN_E2E, "set RUN_E2E=1 to run the launchers against the real Claude Code")
class LauncherE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shared.gateway.testing import claude_e2e

        if not claude_e2e.claude_available():
            raise unittest.SkipTest("Claude Code (`claude`) is not installed")
        cls.e2e = claude_e2e

    def run_launcher(self, name):
        kind, key_var, upstream_var, suffix, key_opt = CASES[name]
        mu = mock_kind_or_skip(self, kind)
        key = "%s-%s-e2e-key" % (SENTINEL, name)
        tmp = tempfile.mkdtemp(prefix="ail-e2e-%s-" % name)
        self.addCleanup(shutil.rmtree, tmp, True)
        with mu.MockServer(kind, options={key_opt: key}) as srv:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("AI_GATEWAY_", "AI_LAUNCHERS_"))}
            env.update({key_var: key, upstream_var: srv.url + suffix})
            argv = [sys.executable, os.path.join(REPO_ROOT, name, "%s-wrap.py" % name), "launch", "claude"]
            res = self.e2e.run_e2e(argv, env, os.path.join(tmp, "work"), timeout=300, home=os.path.join(tmp, "home"))
            info = res.summary() + "\n--- mock errors ---\n" + "\n".join(srv.errors)
            self.assertFalse(res.timed_out, info)
            self.assertEqual(res.rc, 0, info)
            self.assertIn("DONE", res.result_text or "", info)
            with open(os.path.join(res.workdir, "out.txt"), "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "hello")
            self.assertEqual(srv.brain.leaks, [], "provider key leaked into the Bash tool environment")
            env_dumps = [t for t in _tool_results(res.events) if "ANTHROPIC_BASE_URL=" in t]
            self.assertTrue(env_dumps, "the Bash tool's `env` output was not captured\n" + info)
            for dump in env_dumps:
                self.assertNotIn(SENTINEL, dump)
                self.assertNotIn(key, dump)
                self.assertIn("ANTHROPIC_BASE_URL=http://127.0.0.1:", dump)
            self.assertEqual(srv.errors, [], info)
            self.assertNotIn(SENTINEL, res.stderr)
            self.assertTrue(res.settings_unchanged, "~/.claude/settings.json was modified")
            self.assertTrue(res.claude_json_user_entry_intact, "~/.claude.json user entry was lost")
            gw_log = os.path.join(res.home, ".ai-launchers", "logs", "%s-wrap.log" % name)
            if os.path.exists(gw_log):
                with open(gw_log, "r", encoding="utf-8") as f:
                    self.assertNotIn(SENTINEL, f.read())
            return res, srv

    def test_grok_wrap_api_key(self):
        self.run_launcher("grok")

    def test_codex_wrap_api_key(self):
        self.run_launcher("codex")

    def test_gemini_wrap_api_key(self):
        self.run_launcher("gemini")

    def test_deepseek_wrap_api_key(self):
        self.run_launcher("deepseek")

    def test_kimi_wrap_api_key(self):
        self.run_launcher("kimi")


if __name__ == "__main__":
    unittest.main()
