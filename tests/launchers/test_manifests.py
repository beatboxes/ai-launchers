"""Manifests v0.2.0 (plan "ai-launchers changes" table), manifest validation, wrappers and .cmd shims."""

import contextlib
import copy
import io
import json
import os
import unittest

from ._util import (DIRECT_LAUNCHERS, GATEWAY_LAUNCHERS, LAUNCHERS, REPO_ROOT, LauncherTestCase, base_launcher,
                    manifest_path)
from shared.gateway import catalog, launchkit, presets


def _raw(name):
    with open(manifest_path(name), "r", encoding="utf-8") as f:
        return json.load(f)


EXPECTED = {
    "grok": ("gateway", [("xai", "xai", "api-key", "grok-4.7", "grok-4.20-0309-non-reasoning"),
                         ("grok", "grok", "login", "grok-4.7", "grok-4.20-0309-non-reasoning")]),
    "codex": ("gateway", [("openai", "openai", "api-key", "gpt-5.5", "gpt-5.4-mini"),
                          ("codex", "codex", "login", "gpt-5.5", "gpt-5.5")]),
    "gemini": ("gateway", [("gemini", "gemini", "api-key", "gemini-3.1-pro-preview", "gemini-3.8-flash"),
                           ("gemini-vertex", "gemini-vertex", "adc", "gemini-3.1-pro-preview", "gemini-3.8-flash")]),
}


class ManifestContentTests(unittest.TestCase):
    def test_all_manifests_valid(self):
        for name in LAUNCHERS:
            m = _raw(name)
            self.assertEqual(base_launcher.manifest_problems(m), [], name)
            self.assertEqual(m["name"], "%s-wrap" % name)
            self.assertEqual(m["version"], "0.2.0")
            base_launcher.load_manifest(manifest_path(name))

    def test_gateway_transports_match_plan(self):
        for name, (mode, transports) in EXPECTED.items():
            m = _raw(name)
            self.assertEqual(m["mode"], mode)
            got = [(t["id"], t["preset"], t["auth"], t["default_model"], t["background_model"])
                   for t in m["transports"]]
            self.assertEqual(got, transports, name)
            self.assertEqual(m["transports"][0]["auth"], "api-key", "API key transport first")

    def test_key_env_vars(self):
        env = {name: _raw(name)["transports"][0]["env"] for name in LAUNCHERS}
        self.assertEqual(env["grok"], ["XAI_API_KEY"])
        self.assertEqual(env["codex"], ["OPENAI_API_KEY"])
        self.assertEqual(env["gemini"], ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
        self.assertEqual(env["deepseek"], ["DEEPSEEK_API_KEY"])
        self.assertEqual(env["kimi"][0], "MOONSHOT_API_KEY")

    def test_deepseek_direct(self):
        m = _raw("deepseek")
        self.assertEqual(m["mode"], "direct")
        t = m["transports"][0]
        self.assertEqual(t["base_url"], "https://api.deepseek.com/anthropic")
        pro, flash = "deepseek-v4-pro[1m]", "deepseek-v4-flash[1m]"
        self.assertEqual(t["model_env"], {
            "ANTHROPIC_MODEL": pro, "ANTHROPIC_DEFAULT_OPUS_MODEL": pro, "ANTHROPIC_DEFAULT_SONNET_MODEL": pro,
            "ANTHROPIC_DEFAULT_FABLE_MODEL": pro, "ANTHROPIC_DEFAULT_HAIKU_MODEL": flash})
        self.assertEqual(t["env_extra"], {"CLAUDE_CODE_EFFORT_LEVEL": "max",
                                          "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})

    def test_kimi_direct(self):
        m = _raw("kimi")
        self.assertEqual(m["mode"], "direct")
        t = m["transports"][0]
        self.assertEqual(t["base_url"], "https://api.moonshot.ai/anthropic")
        k3 = "kimi-k3[1m]"
        self.assertEqual(t["model_env"], {
            "ANTHROPIC_MODEL": k3, "ANTHROPIC_DEFAULT_OPUS_MODEL": k3, "ANTHROPIC_DEFAULT_SONNET_MODEL": k3,
            "ANTHROPIC_DEFAULT_FABLE_MODEL": k3, "CLAUDE_CODE_SUBAGENT_MODEL": k3,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-k2.7-code"})

    def test_models_exist_in_catalog(self):
        for name in GATEWAY_LAUNCHERS:
            for t in _raw(name)["transports"]:
                family = presets.PRESETS[t["preset"]]["catalog"]
                for key in ("default_model", "background_model"):
                    self.assertIsNotNone(catalog.find_model(family, t[key]), (name, t["id"], t[key]))
        for name in DIRECT_LAUNCHERS:
            t = _raw(name)["transports"][0]
            family = presets.PRESETS[t["preset"]]["catalog"]
            for value in t["model_env"].values():
                self.assertIsNotNone(catalog.find_model(family, value.replace("[1m]", "")), (name, value))

    def test_direct_model_vars_match_launchkit(self):
        allowed = set(launchkit.MODEL_ENV_KEYS.values())
        for name in DIRECT_LAUNCHERS:
            self.assertLessEqual(set(_raw(name)["transports"][0]["model_env"]), allowed)


class ManifestValidationTests(unittest.TestCase):
    def _problems(self, name, mutate):
        m = copy.deepcopy(_raw(name))
        mutate(m)
        return "\n".join(base_launcher.manifest_problems(m))

    def test_detects_errors(self):
        cases = [
            ("grok", lambda m: m.update(mode="router"), "mode must be"),
            ("grok", lambda m: m.update(version="0.2"), "version"),
            ("grok", lambda m: m["transports"].reverse(), "must come before"),
            ("grok", lambda m: m["transports"][0].update(preset="nope"), "unknown preset"),
            ("grok", lambda m: m["transports"][1].update(auth="api-key"), "auth must be 'login'"),
            ("grok", lambda m: m["transports"][1].update(id="xai"), "duplicate id"),
            ("grok", lambda m: m["transports"][0].update(bogus=1), "unknown key 'bogus'"),
            ("grok", lambda m: m["transports"][0].update(env=[]), "env must be"),
            ("grok", lambda m: m["transports"][1].update(env=["X"]), "only applies to api-key"),
            ("grok", lambda m: m["transports"][0].update(model_env={"ANTHROPIC_MODEL": "x"}), "direct mode"),
            ("grok", lambda m: m.update(transports=[]), "non-empty"),
            ("deepseek", lambda m: m["transports"][0]["model_env"].pop("ANTHROPIC_MODEL"), "ANTHROPIC_MODEL required"),
            ("deepseek", lambda m: m["transports"][0]["model_env"].update(ANTHROPIC_FOO="x"), "not a model variable"),
            ("deepseek", lambda m: m["transports"][0]["env_extra"].update(ANTHROPIC_AUTH_TOKEN="x"), "may not set"),
            ("deepseek", lambda m: m["transports"].append(dict(m["transports"][0], id="d2")), "exactly one"),
            ("kimi", lambda m: m["transports"][0].update(base_url="ftp://x"), "base_url must be"),
        ]
        for name, mutate, needle in cases:
            self.assertIn(needle, self._problems(name, mutate), needle)

    def test_load_manifest_errors(self):
        with self.assertRaises(base_launcher.ManifestError):
            base_launcher.load_manifest(os.path.join(REPO_ROOT, "does-not-exist.json"))


class WrapperTests(LauncherTestCase):
    def test_wrappers_are_thin(self):
        for name in LAUNCHERS:
            path = os.path.join(REPO_ROOT, name, "%s-wrap.py" % name)
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
            self.assertIn("from shared.base_launcher import run", src)
            self.assertLess(len(src.splitlines()), 20, name)

    def test_cmd_shims_prefer_py_launcher(self):
        for name in LAUNCHERS:
            with open(os.path.join(REPO_ROOT, name, "%s-wrap.cmd" % name), "rb") as f:
                data = f.read()
            text = data.decode("ascii")
            self.assertNotIn("\n", text.replace("\r\n", ""), "CRLF line endings")
            self.assertIn("where py", text)
            self.assertIn('py -3 "%%~dp0%s-wrap.py" %%*' % name, text)
            self.assertIn('python "%%~dp0%s-wrap.py" %%*' % name, text)
            self.assertLess(text.index("py -3"), text.index("python "))

    def test_invalid_manifest_exits_2(self):
        bad = self.write_json(os.path.join(self.tmp, "bad.json"), {"name": "x"})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = base_launcher.run(bad, ["--version"])
        self.assertEqual(rc, 2)
        self.assertIn("transports must be a non-empty list", err.getvalue())

    def test_version_and_help(self):
        for name in LAUNCHERS:
            rc, out, _ = self.run_cli(name, "--version")
            self.assertEqual(rc, 0)
            self.assertIn("%s-wrap 0.2.0" % name, out)
        rc, out, _ = self.run_cli("grok", "--help")
        self.assertEqual(rc, 0)
        self.assertIn(os.path.join(self.ail, "credentials.json"), out)
        self.assertNotIn(".f/credentials", out)
        for word in ("launch claude", "--auth", "models", "keys set", "doctor", "XAI_API_KEY", "grok login"):
            self.assertIn(word, out)

    def test_unknown_launch_args_hint(self):
        rc, _, err = self.run_cli("grok", "launch", "claude", "-p", "hi")
        self.assertEqual(rc, 2)
        self.assertIn("after `--`", err)


if __name__ == "__main__":
    unittest.main()
