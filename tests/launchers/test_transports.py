"""Transport selection (order, --auth filter, login/ADC availability, key sources), route tables per
launcher, --model routing, user overrides, discovery-cache merge and direct-mode environments."""

import json
import os
import time
import unittest
from unittest import mock

from ._util import SENTINEL, LauncherTestCase, base_launcher, fake_key
from shared.gateway import compat
from shared.gateway import config as gwconfig


class FakeAuth(object):
    def __init__(self, ok):
        self.ok = ok

    def available(self):
        return self.ok

    def describe(self):
        return {"kind": "fake", "source": "fake file"}

    def relogin_hint(self):
        return "run fake login"


def fake_logins(*available_ids):
    """Patch make_auth: login/ADC transports in ``available_ids`` are available."""
    return mock.patch("shared.gateway.auth.make_auth",
                      side_effect=lambda auth, store, pid: FakeAuth(pid in available_ids))


def jwt(claims):
    return "%s.%s.sig" % (compat.b64url_encode(json.dumps({"alg": "none"})), compat.b64url_encode(json.dumps(claims)))


class SelectionTests(LauncherTestCase):
    def ids(self, launcher, auth="auto", dry_run=False):
        selected, _ = launcher.resolve(auth, dry_run=dry_run)
        return [t.id for t in selected]

    def test_order_and_auth_filter(self):
        os.environ["XAI_API_KEY"] = fake_key("xai")
        with fake_logins("grok"):
            launcher = self.launcher("grok")
            self.assertEqual(self.ids(launcher), ["xai", "grok"])
            self.assertEqual(self.ids(launcher, "api-key"), ["xai"])
            self.assertEqual(self.ids(launcher, "login"), ["grok"])
            self.assertEqual(self.ids(launcher, "adc"), [])
            self.assertEqual(self.ids(launcher, dry_run=True), ["xai", "grok"])
            excluded = [t for t in launcher.transports if t.id == "xai"][0]
            launcher.resolve("login")
            self.assertIn("excluded by --auth login", excluded.reason)

    def test_no_transport_instructions(self):
        with fake_logins():
            rc, out, err = self.run_cli("grok", "launch", "claude")
        self.assertEqual(rc, 2)
        for needle in ("no usable transport", "XAI_API_KEY", "grok-wrap keys set", "grok login", "doctor"):
            self.assertIn(needle, err)
        with fake_logins():
            rc, _, err = self.run_cli("gemini", "launch", "claude", "--auth", "adc")
        self.assertEqual(rc, 2)
        self.assertIn("gcloud auth application-default login", err)
        self.assertNotIn("GEMINI_API_KEY", err)

    def test_login_only_codex(self):
        with fake_logins("codex"):
            launcher = self.launcher("codex")
            selected, _ = launcher.resolve()
            table = launcher.route_table(selected)
        self.assertEqual(list(table.providers), ["codex"])
        self.assertEqual(table.roles, {"default": "codex,gpt-5.5", "background": "codex,gpt-5.5"})
        self.assertEqual(table.providers["codex"].target, "chatgpt_codex")

    def test_codex_auth_json_api_key(self):
        key = fake_key("codexfile")
        self.write_json(os.path.join(self.home, ".codex", "auth.json"), {"OPENAI_API_KEY": key, "tokens": None})
        launcher = self.launcher("codex")
        selected, store = launcher.resolve()
        self.assertEqual([t.id for t in selected], ["openai"])
        self.assertEqual(store.get("openai"), key)
        self.assertEqual(selected[0].source, "json:auth.json")

    def test_grok_api_key_entry_feeds_xai(self):
        key = fake_key("grokfile")
        self.write_json(os.path.join(self.home, ".grok", "auth.json"),
                        {"xai::api_key": {"auth_mode": "api_key", "key": key}})
        launcher = self.launcher("grok")
        selected, store = launcher.resolve()
        self.assertEqual([t.id for t in selected], ["xai"])
        self.assertEqual(store.get("xai"), key)
        self.assertIn("api_key entry", selected[0].source)

    def test_real_codex_login_file(self):
        claims = {"exp": int(time.time()) + 3600, "https://api.openai.com/auth": {"chatgpt_account_id": "acct"}}
        self.write_json(os.path.join(self.home, ".codex", "auth.json"), {
            "OPENAI_API_KEY": None, "last_refresh": compat.rfc3339_format(),
            "tokens": {"access_token": jwt(claims), "id_token": jwt(claims), "refresh_token": "rt",
                       "account_id": "acct"}})
        launcher = self.launcher("codex")
        selected, _ = launcher.resolve()
        self.assertEqual([t.id for t in selected], ["codex"])
        self.assertIn("codex login", selected[0].source)

    def test_vertex_needs_credentials(self):
        launcher = self.launcher("gemini")
        selected, _ = launcher.resolve("adc")
        self.assertEqual(selected, [])
        self.write_json(os.path.join(self.home, ".config", "gcloud", "application_default_credentials.json"),
                        {"type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r",
                         "quota_project_id": "proj"})
        selected, _ = launcher.resolve("adc")
        self.assertEqual([t.id for t in selected], ["gemini-vertex"])

    def test_op_ref_resolved_only_outside_dry_run(self):
        from shared.gateway.testing import fake_bins

        ref = "op://Personal/DeepSeek/credential"
        key = fake_key("op")
        fake_bins.make_fake_bins(self.bin, op_values={ref: key})
        self.write_config({"providers": {"deepseek": {"op_ref": ref}}})
        launcher = self.launcher("deepseek")
        selected, store = launcher.resolve(dry_run=True)
        self.assertEqual([t.id for t in selected], ["deepseek"])
        self.assertIn("not resolved in dry-run", selected[0].source)
        self.assertFalse(store.has("deepseek"))
        self.assertEqual(fake_bins.read_calls(self.bin, "op"), [])
        selected, store = launcher.resolve()
        self.assertEqual(store.get("deepseek"), key)
        self.assertEqual(selected[0].source, "op:DeepSeek")
        self.assertEqual(len(fake_bins.read_calls(self.bin, "op")), 1)

    def test_env_beats_op_and_credentials(self):
        env_key, file_key = fake_key("env"), fake_key("file")
        self.write_json(os.path.join(self.ail, "credentials.json"), {"grok-wrap": {"key": file_key}})
        self.write_config({"providers": {"xai": {"op_ref": "op://x/y/z"}}})
        os.environ["XAI_API_KEY"] = env_key
        with fake_logins():
            selected, store = self.launcher("grok").resolve()
        self.assertEqual(store.get("xai"), env_key)
        self.assertEqual(selected[0].source, "env:XAI_API_KEY")


class RouteTableTests(LauncherTestCase):
    def table(self, name, model=None, logins=("grok", "codex", "gemini-vertex"), config=None):
        for var in ("XAI_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
            os.environ[var] = fake_key(var.lower())
        with fake_logins(*logins):
            launcher = self.launcher(name, config=config)
            selected, _ = launcher.resolve()
            return launcher, launcher.route_table(selected, model)

    def test_grok_table(self):
        launcher, table = self.table("grok")
        self.assertEqual(list(table.providers), ["xai", "grok"])
        xai, grok = table.providers["xai"], table.providers["grok"]
        self.assertEqual((xai.dialect, xai.profile, xai.base_url), ("openai_chat", "xai", "https://api.x.ai/v1"))
        self.assertEqual(xai.auth, {"kind": "api_key", "secret": "xai", "style": "bearer"})
        self.assertEqual((grok.dialect, grok.target), ("responses", "grok_cli_proxy"))
        self.assertEqual(grok.fallback.base_url, "https://api.x.ai/v1")
        self.assertEqual(table.roles, {"default": "xai,grok-4.7", "background": "xai,grok-4.20-0309-non-reasoning"})
        self.assertEqual(launcher.default_route(table)[2], "claude-via-xai,grok-4.7[1m]")
        self.assertEqual(table.problems(), [])

    def test_codex_table(self):
        _, table = self.table("codex")
        self.assertEqual(list(table.providers), ["openai", "codex"])
        self.assertEqual(table.providers["openai"].target, "openai_api")
        self.assertEqual(table.providers["codex"].target, "chatgpt_codex")
        self.assertEqual(table.roles, {"default": "openai,gpt-5.5", "background": "openai,gpt-5.4-mini"})

    def test_gemini_table(self):
        _, table = self.table("gemini")
        self.assertEqual(list(table.providers), ["gemini", "gemini-vertex"])
        self.assertEqual(table.providers["gemini"].auth["style"], "x-goog-api-key")
        self.assertEqual(table.providers["gemini-vertex"].target, "vertex")
        self.assertEqual(table.roles["default"], "gemini,gemini-3.1-pro-preview")

    def test_model_flag(self):
        _, table = self.table("grok", "grok-4.3")
        self.assertEqual(table.roles["default"], "xai,grok-4.3")
        _, table = self.table("grok", "grok,grok-4.6")
        self.assertEqual(table.roles, {"default": "grok,grok-4.6", "background": "grok,grok-4.20-0309-non-reasoning"})
        _, table = self.table("grok", "claude-via-xai,grok-4.7[1m]")
        self.assertEqual(table.roles["default"], "xai,grok-4.7")
        _, table = self.table("codex", "codex,gpt-5.5")
        self.assertEqual(table.roles, {"default": "codex,gpt-5.5", "background": "codex,gpt-5.5"})
        with self.assertRaises(base_launcher.LaunchError) as cm:
            self.table("grok", "nosuchprovider,model-x")
        self.assertIn("grok-wrap models", str(cm.exception))

    def test_user_overrides(self):
        cfg = {"providers": {"xai": {"base_url": "http://127.0.0.1:9/v1/", "default_model": "grok-4.6",
                                     "models": ["grok-4.6", {"id": "grok-x", "context": 100000}]}}}
        _, table = self.table("grok", config=cfg)
        xai = table.providers["xai"]
        self.assertEqual(xai.base_url, "http://127.0.0.1:9/v1")
        self.assertEqual([m.id for m in xai.models], ["grok-4.6", "grok-x"])
        self.assertEqual(table.roles["default"], "xai,grok-4.6")

    def test_invalid_config_warns(self):
        os.makedirs(self.ail)
        with open(os.path.join(self.ail, "config.json"), "w") as f:
            f.write("{not json")
        os.environ["XAI_API_KEY"] = fake_key("xai")
        os.environ["AI_LAUNCHERS_CLAUDE_BIN"] = os.path.join(self.tmp, "nowhere")
        rc, _, err = self.run_cli("grok", "launch", "claude", "--dry-run")
        self.assertEqual(rc, 0, err)
        self.assertIn("warning: ignoring %s" % os.path.join(self.ail, "config.json"), err)
        cfg = {"providers": {"xai": {"env": {"not": "a list"}}, "grok": "nonsense"}}
        self.write_config(cfg)
        rc, out, _ = self.run_cli("grok", "doctor")
        self.assertIn("[warn] ignoring providers.xai.env: expected a JSON list", out)

    def test_legacy_config(self):
        os.environ["MY_XAI"] = fake_key("legacy")
        cfg = {"providers": {"xai": {"base_url": "https://api.x.ai/v1/chat/completions", "env_var": "MY_XAI"}}}
        with fake_logins():
            launcher = self.launcher("grok", config=cfg)
            selected, store = launcher.resolve()
        self.assertEqual(store.get("xai"), os.environ["MY_XAI"])
        self.assertTrue(any("CCR-style" in w for w in selected[0].warnings))
        table = launcher.route_table(selected)
        self.assertEqual(table.providers["xai"].base_url, "https://api.x.ai/v1")

    def test_upstream_env_override(self):
        os.environ["AI_GATEWAY_UPSTREAM_XAI"] = "http://127.0.0.1:1234/v1"
        _, table = self.table("grok")
        self.assertEqual(table.providers["xai"].base_url, "http://127.0.0.1:1234/v1")

    def test_discovery_cache_merge(self):
        path = os.path.join(self.ail, "cache", "models.json")
        entry = {"url": "https://api.x.ai/v1/models", "fetched_at": time.time(), "models": ["grok-9", "grok-4.7"]}
        self.write_json(path, {"version": 1, "entries": {"xai": entry}})
        _, table = self.table("grok")
        ids = [m.id for m in table.providers["xai"].models]
        self.assertIn("grok-9", ids)
        self.assertEqual(ids.count("grok-4.7"), 1)
        self.write_json(path, {"version": 1, "entries": {"xai": dict(entry, fetched_at=time.time() - 7 * 3600)}})
        _, table = self.table("grok")
        self.assertNotIn("grok-9", [m.id for m in table.providers["xai"].models])
        self.write_json(path, {"version": 1, "entries": {"xai": dict(entry, url="http://elsewhere/models")}})
        _, table = self.table("grok")
        self.assertNotIn("grok-9", [m.id for m in table.providers["xai"].models])


class DirectEnvTests(LauncherTestCase):
    def env(self, name, var, model=None):
        key = fake_key(name)
        os.environ[var] = key
        os.environ["COPY_OF_KEY"] = "prefix-%s" % key
        launcher = self.launcher(name)
        selected, store = launcher.resolve()
        return key, launcher.direct_env(selected[0], store, store.get(selected[0].secret_name), model)

    def test_deepseek(self):
        key, env = self.env("deepseek", "DEEPSEEK_API_KEY")
        pro, flash = "deepseek-v4-pro[1m]", "deepseek-v4-flash[1m]"
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "https://api.deepseek.com/anthropic")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], key)
        self.assertEqual(env["ANTHROPIC_API_KEY"], "")
        for var in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                    "ANTHROPIC_DEFAULT_FABLE_MODEL"):
            self.assertEqual(env[var], pro, var)
        self.assertEqual(env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], flash)
        self.assertEqual(env["CLAUDE_CODE_EFFORT_LEVEL"], "max")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")
        self.assertNotIn("DEEPSEEK_API_KEY", env)
        self.assertNotIn("COPY_OF_KEY", env)
        self.assertNotIn("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", env)
        self.assertEqual([k for k, v in env.items() if SENTINEL in v], ["ANTHROPIC_AUTH_TOKEN"])

    def test_deepseek_model_flag(self):
        _, env = self.env("deepseek", "DEEPSEEK_API_KEY", "deepseek-v4-flash")
        self.assertEqual(env["ANTHROPIC_MODEL"], "deepseek-v4-flash[1m]")
        self.assertEqual(env["ANTHROPIC_DEFAULT_OPUS_MODEL"], "deepseek-v4-flash[1m]")
        self.assertEqual(env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "deepseek-v4-flash[1m]")

    def test_kimi(self):
        key, env = self.env("kimi", "KIMI_API_KEY")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "https://api.moonshot.ai/anthropic")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], key)
        for var in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                    "ANTHROPIC_DEFAULT_FABLE_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
            self.assertEqual(env[var], "kimi-k3[1m]", var)
        self.assertEqual(env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "kimi-k2.7-code")
        self.assertNotIn("KIMI_API_KEY", env)
        _, env = self.env("kimi", "MOONSHOT_API_KEY", "kimi-k2.7-code")
        self.assertEqual(env["ANTHROPIC_MODEL"], "kimi-k2.7-code")
        self.assertEqual(env["CLAUDE_CODE_SUBAGENT_MODEL"], "kimi-k2.7-code")

    def test_overrides(self):
        os.environ["AI_GATEWAY_UPSTREAM_DEEPSEEK"] = "http://127.0.0.1:5/anthropic/"
        self.write_config({"providers": {"deepseek": {"default_model": "deepseek-v4-flash[1m]",
                                                      "background_model": "deepseek-v4-flash"}}})
        _, env = self.env("deepseek", "DEEPSEEK_API_KEY")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:5/anthropic")
        self.assertEqual(env["ANTHROPIC_MODEL"], "deepseek-v4-flash[1m]")
        self.assertEqual(env["ANTHROPIC_DEFAULT_FABLE_MODEL"], "deepseek-v4-flash[1m]")
        self.assertEqual(env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "deepseek-v4-flash")

    def test_direct_rejects_provider_model_pair(self):
        os.environ["DEEPSEEK_API_KEY"] = fake_key("d")
        os.environ["AI_LAUNCHERS_CLAUDE_BIN"] = os.path.join(self.bin, "missing-claude")
        rc, _, err = self.run_cli("deepseek", "launch", "claude", "--dry-run", "--model", "deepseek,x")
        self.assertEqual(rc, 2)
        self.assertIn("bare deepseek model id", err)


class BaseEnvTests(LauncherTestCase):
    def test_gateway_child_env_hygiene(self):
        from shared.gateway import launchkit

        key = fake_key("xai")
        os.environ.update({"XAI_API_KEY": key, "OTHER": "has %s inside" % key, "ANTHROPIC_SMALL_FAST_MODEL": "x",
                           "CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_API_KEY": "sk-ant-real"})
        launcher = self.launcher("grok")
        with fake_logins():
            selected, store = launcher.resolve()
        env = launchkit.build_child_env(launcher._base_env(), "http://127.0.0.1:1", "tok", "claude-via-xai,grok-4.7",
                                        "claude-via-background", 2000000, secret_values=store)
        for var in ("XAI_API_KEY", "OTHER", "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_USE_BEDROCK"):
            self.assertNotIn(var, env)
        self.assertEqual(env["ANTHROPIC_API_KEY"], "")
        self.assertFalse([k for k, v in env.items() if SENTINEL in v])


if __name__ == "__main__":
    unittest.main()
