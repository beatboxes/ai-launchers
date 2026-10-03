"""Preset / catalog CONTENT (plan Appendix A) and consistency."""

import json
import os
import re
import unittest

from ._pkg import mod, package_dir

config = mod("config")
presets = mod("presets")
catalog = mod("catalog")

ALL = ("xai", "grok", "openai", "codex", "gemini", "gemini-vertex", "deepseek", "kimi", "openrouter", "ollama",
       "opencode-zen", "opencode-go", "opencode", "nvidia")

SECRET_ENV = {
    "xai": ["XAI_API_KEY"], "openai": ["OPENAI_API_KEY"], "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"], "kimi": ["MOONSHOT_API_KEY"], "openrouter": ["OPENROUTER_API_KEY"],
    "opencode-zen": ["OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY"],
    "opencode-go": ["OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"], "nvidia": ["NVIDIA_API_KEY"],
}


def ids(name):
    return [m.id for m in presets.provider_from_preset(name).models]


class PresetConsistencyTests(unittest.TestCase):
    def test_every_preset_builds_a_valid_route_table(self):
        self.assertEqual(sorted(presets.preset_names()), sorted(ALL))
        for name in ALL:
            spec = presets.provider_from_preset(name)
            self.assertEqual(spec.problems(), [], name)
            roles = presets.default_roles(name) or {"default": "%s,qwen3:8b" % name}  # ollama: discovered
            table = presets.route_table_from_presets([name], roles=roles)
            self.assertEqual(table.problems(), [], name)
            again = config.RouteTable.from_dict(json.loads(json.dumps(table.to_dict())))
            self.assertEqual(again, table, name)

    def test_roles_exist_in_catalogs_and_picker_ids(self):
        for name in ALL:
            p = presets.PRESETS[name]
            spec = presets.provider_from_preset(name)
            for key in ("default_model", "background_model"):
                mid = p.get(key)
                if name == "ollama":
                    self.assertIsNone(mid)  # no hard-coded local model
                    continue
                self.assertTrue(spec.lists_model(mid), "%s %s=%s not in catalog" % (name, key, mid))
            table = presets.route_table_from_presets([name], roles=presets.default_roles(name) or
                                                     {"default": "ollama,x"})
            for m in spec.models:
                pid = table.picker_id(name, m)
                self.assertRegex(pid, r"(?i)claude|anthropic")
                self.assertNotRegex(pid, r"\s")
                self.assertEqual(pid.count(","), 1, pid)
                self.assertEqual(pid.endswith("[1m]"), (m.context or 0) >= 1000000, pid)

    def test_picker_ids_for_defaults(self):
        expected = {
            "xai": "claude-via-xai,grok-4.7[1m]", "grok": "claude-via-grok,grok-4.7[1m]",
            "openai": "claude-via-openai,gpt-5.5", "codex": "claude-via-codex,gpt-5.5",
            "gemini": "claude-via-gemini,gemini-3.1-pro-preview[1m]",
            "deepseek": "claude-via-deepseek,deepseek-v4-pro[1m]", "kimi": "claude-via-kimi,kimi-k3[1m]",
            "opencode": "claude-via-opencode,nemotron-3-ultra-free",
        }
        for name, pid in expected.items():
            table = presets.route_table_from_presets([name])
            self.assertEqual(table.picker_id(name, presets.PRESETS[name]["default_model"]), pid)

    def test_secret_env(self):
        for name in ALL:
            want = SECRET_ENV.get(name, [])
            self.assertEqual(presets.secret_env_vars(name), want, name)
            spec = presets.provider_from_preset(name)
            if want:
                self.assertEqual(spec.auth["kind"], "api_key", name)
                self.assertEqual(spec.auth["secret"], name)
            else:
                self.assertNotEqual(spec.auth["kind"], "api_key", name)
            for v in want:
                self.assertRegex(v, r"^[A-Z][A-Z0-9_]*_API_KEY$")

    def test_catalog_values_sane(self):
        for fam in catalog.families():
            for m in catalog.get_catalog(fam):
                self.assertNotIn(",", m.id)
                self.assertIsNotNone(m.context, "%s/%s has no context" % (fam, m.id))
                self.assertGreaterEqual(m.context, 32000, "%s/%s" % (fam, m.id))
                if m.max_output is not None:
                    self.assertGreaterEqual(m.max_output, 4096, "%s/%s" % (fam, m.id))
                    self.assertLess(m.max_output, m.context, "%s/%s" % (fam, m.id))
                self.assertEqual(m.extra, {}, "%s/%s has unknown keys" % (fam, m.id))
            for retired in ("deepseek-chat", "deepseek-reasoner", "o1-mini", "codex-mini", "gemini-2.0-flash"):
                self.assertNotIn(retired, [m.id for m in catalog.get_catalog(fam)])

    def test_notes_and_no_todo_left(self):
        for name in ALL:
            self.assertTrue(presets.PRESETS[name].get("notes"), name)
        for fn in ("presets.py", "catalog.py"):
            with open(os.path.join(package_dir(), fn), "r", encoding="utf-8") as f:
                self.assertNotIn("TODO", f.read(), fn)


class AppendixAContentTests(unittest.TestCase):
    def test_xai_and_grok(self):
        self.assertEqual(ids("xai"), ["grok-4.7", "grok-4.6", "grok-4.5", "grok-4.3", "grok-4.20-0309-reasoning",
                                      "grok-4.20-0309-non-reasoning", "grok-build-0.1", "grok-composer-2.5-fast",
                                      "grok-4.20-multi-agent-0309"])
        xai = presets.provider_from_preset("xai")
        self.assertEqual((xai.base_url, xai.dialect, xai.profile), ("https://api.x.ai/v1", "openai_chat", "xai"))
        ma = xai.find_model("grok-4.20-multi-agent-0309")
        self.assertEqual((xai.effective_dialect(ma), xai.effective_target(ma)), ("responses", "xai_api"))
        for m in xai.models:
            self.assertEqual(catalog.is_xai_reasoning(m.id, m), m.id != "grok-4.20-0309-non-reasoning", m.id)
            self.assertFalse(m.effort_param)
        self.assertEqual(presets.default_roles("xai")["background"], "xai,grok-4.20-0309-non-reasoning")
        grok = presets.provider_from_preset("grok")
        self.assertEqual((grok.dialect, grok.target, grok.base_url),
                         ("responses", "grok_cli_proxy", "https://cli-chat-proxy.grok.com/v1"))
        self.assertEqual((grok.fallback.dialect, grok.fallback.profile, grok.fallback.base_url),
                         ("openai_chat", "xai", "https://api.x.ai/v1"))
        self.assertNotIn("grok-4.20-multi-agent-0309", ids("grok"))
        self.assertEqual(presets.PRESETS["grok"]["default_model"], "grok-4.7")

    def test_openai_and_codex(self):
        o = ids("openai")
        for mid in ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"):
            self.assertIn(mid, o)
        self.assertTrue(any(m.endswith("-pro") for m in o) and any(m.endswith("-codex") for m in o))
        self.assertEqual(presets.default_roles("openai"), {"default": "openai,gpt-5.5",
                                                           "background": "openai,gpt-5.4-mini"})
        op = presets.provider_from_preset("openai")
        self.assertEqual((op.target, op.base_url), ("openai_api", "https://api.openai.com/v1"))
        self.assertFalse(op.find_model("gpt-4.1").reasoning)
        c = presets.provider_from_preset("codex")
        self.assertEqual((c.target, c.auth["kind"]), ("chatgpt_codex", "codex_chatgpt"))
        self.assertEqual(presets.default_roles("codex"), {"default": "codex,gpt-5.5", "background": "codex,gpt-5.5"})
        for mid in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-sol", "gpt-6.1-sol"):
            self.assertIn(mid, ids("codex"))
        for m in c.models:
            self.assertEqual(m.responses_lite, catalog.is_responses_lite(m.id), m.id)
        self.assertTrue(c.find_model("gpt-6.1-sol").responses_lite)
        self.assertFalse(c.find_model("gpt-5.6-sol").responses_lite)

    def test_gemini(self):
        want = ["gemini-3.1-pro-preview", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                "gemini-3.5-flash-lite", "gemini-2.5-pro"]
        self.assertEqual(ids("gemini"), want)
        self.assertEqual(ids("gemini-vertex"), want)
        v = presets.provider_from_preset("gemini-vertex")
        self.assertEqual((v.target, v.auth["kind"], v.base_url), ("vertex", "gcloud_adc", ""))
        self.assertEqual(presets.default_roles("gemini")["background"], "gemini,gemini-3.8-flash")

    def test_anthropic_compatible(self):
        cases = {"deepseek": "https://api.deepseek.com/anthropic", "kimi": "https://api.moonshot.ai/anthropic",
                 "openrouter": "https://openrouter.ai/api", "ollama": "http://localhost:11434"}
        for name, url in cases.items():
            p = presets.provider_from_preset(name)
            self.assertEqual((p.dialect, p.base_url), ("anthropic_passthrough", url), name)
            self.assertEqual(p.auth.get("style", "bearer") if p.auth["kind"] == "api_key" else "none",
                             "none" if name == "ollama" else "bearer")
        self.assertEqual(ids("deepseek"), ["deepseek-v4-pro", "deepseek-v4-flash"])
        self.assertEqual(ids("kimi"), ["kimi-k3", "kimi-k2.7-code", "kimi-k2.6", "kimi-k2.5"])
        self.assertEqual(presets.default_roles("kimi"),
                         {"default": "kimi,kimi-k3", "background": "kimi,kimi-k2.7-code"})
        o = presets.provider_from_preset("ollama")
        self.assertTrue(o.allow_unlisted)
        self.assertEqual((o.fallback.dialect, o.fallback.profile, o.fallback.base_url),
                         ("openai_chat", "ollama", "http://localhost:11434/v1"))
        self.assertEqual(o.models, [])

    def test_opencode_zen_and_go(self):
        zen = presets.provider_from_preset("opencode-zen")
        self.assertEqual((zen.base_url, zen.dialect, zen.options.get("messages_path")),
                         ("https://opencode.ai/zen/v1", "openai_chat", "/messages"))
        for mid in ("deepseek-v4-flash-free", "mimo-v2.5-free", "hy3-free", "nemotron-3-ultra-free",
                    "north-mini-code-free", "big-pickle", "glm-5.1", "glm-5", "kimi-k2.5", "kimi-k2.6",
                    "qwen3.6-plus", "qwen3.5-plus"):
            self.assertIn(mid, ids("opencode-zen"))
        go = presets.provider_from_preset("opencode-go")
        self.assertEqual(go.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(ids("opencode-go"), ["glm-5.1", "glm-5", "kimi-k2.5", "kimi-k2.6", "kimi-k3",
                                              "deepseek-v4-pro", "deepseek-v4-flash", "mimo-v2-pro", "mimo-v2.5-pro",
                                              "mimo-v2.5", "qwen3.6-plus", "qwen3.5-plus"])
        for p in (zen, go):
            for m in p.models:
                d = p.effective_dialect(m)
                if re.match(r"^(claude-|qwen3\.[5-8]|minimax-)", m.id):
                    self.assertEqual((d, m.path_override), ("anthropic_passthrough", "/messages"), m.id)
                elif re.match(r"^(gpt-|grok-)", m.id):
                    self.assertEqual((d, p.effective_target(m), m.path_override),
                                     ("responses", "openai_api", "/responses"), m.id)
                else:
                    self.assertEqual((d, m.path_override), ("openai_chat", None), m.id)

    def test_opencode_cli_and_nvidia(self):
        oc = presets.provider_from_preset("opencode")
        self.assertEqual((oc.dialect, oc.chat_only, oc.options.get("cli")), ("cli", True, "opencode"))
        self.assertEqual(ids("opencode"), ["nemotron-3-ultra-free", "mimo-v2.5-free", "north-mini-code-free",
                                           "deepseek-v4-flash-free"])
        self.assertTrue(all(not m.tools for m in oc.models))
        nv = presets.provider_from_preset("nvidia")
        self.assertEqual((nv.dialect, nv.profile, nv.base_url),
                         ("openai_chat", "nvidia", "https://integrate.api.nvidia.com/v1"))
        self.assertGreaterEqual(sum(1 for m in ids("nvidia") if "nemotron" in m), 3)


if __name__ == "__main__":
    unittest.main()
