"""chat_profiles: profile table (DESIGN §3.1) — field names, drops, effort mapping, options overrides."""

import unittest

from ._pkg import mod

cp = mod("chat_profiles")
config = mod("config")
catalog = mod("catalog")

MS = config.ModelSpec


class ProfileTableTests(unittest.TestCase):
    def test_lookup(self):
        self.assertEqual(cp.get_profile(None).name, "generic")
        self.assertEqual(cp.get_profile("no-such-profile").name, "generic")
        for name in ("xai", "openai", "moonshot", "deepseek", "ollama", "openrouter", "opencode", "nvidia", "generic"):
            self.assertEqual(cp.get_profile(name).name, name)
            self.assertIs(cp.get_profile(name), cp.PROFILES[name])

    def test_max_tokens_fields_and_flags(self):
        self.assertEqual(cp.get_profile("openai").max_tokens_field, "max_completion_tokens")
        for name in ("xai", "moonshot", "deepseek", "ollama", "openrouter", "opencode", "nvidia", "generic"):
            self.assertEqual(cp.get_profile(name).max_tokens_field, "max_tokens", name)
        self.assertFalse(cp.get_profile("nvidia").stream_usage)
        self.assertEqual(cp.get_profile("nvidia").tool_name_regex, r"^[a-zA-Z0-9_-]{1,64}$")
        self.assertTrue(cp.get_profile("moonshot").echo_reasoning_content)
        self.assertTrue(cp.get_profile("deepseek").echo_reasoning_content)
        self.assertFalse(cp.get_profile("openai").echo_reasoning_content)
        self.assertEqual(cp.get_profile("xai").schema_mode, "no_root_combinators")
        self.assertEqual(cp.get_profile("generic").schema_mode, "basic")
        self.assertTrue(cp.get_profile("ollama").probe_ollama)
        self.assertEqual(cp.get_profile("ollama").default_base_url, "http://localhost:11434/v1")
        orp = cp.get_profile("openrouter")
        self.assertEqual(set(orp.headers), {"HTTP-Referer", "X-Title"})
        self.assertEqual(orp.extra_body, {"usage": {"include": True}})
        self.assertEqual(cp.get_profile("xai").session_header, "x-grok-conv-id")
        self.assertTrue(cp.get_profile("openai").prompt_cache_key)
        for name, has in (("openai", True), ("xai", True), ("openrouter", True), ("moonshot", False),
                          ("deepseek", False), ("nvidia", False), ("generic", False), ("ollama", False)):
            self.assertEqual(cp.get_profile(name).parallel_param, has, name)

    def test_xai_rules(self):
        p = cp.get_profile("xai")
        drops = {"stop", "presence_penalty", "frequency_penalty"}
        self.assertEqual(set(p.dropped_params(catalog.find_model("xai", "grok-4.7"))), drops)
        self.assertEqual(set(p.dropped_params(catalog.find_model("xai", "grok-4.20-0309-non-reasoning"))), set())
        self.assertEqual(set(p.dropped_params(MS("grok-9-unknown"))), drops)            # id rule
        self.assertEqual(set(p.dropped_params(MS("grok-9-non-reasoning"))), set())
        # reasoning_effort only where the catalog says effort_param
        self.assertIsNone(p.effort_value("high", catalog.find_model("xai", "grok-4.7")))
        eff = MS("grok-3-mini", reasoning=True, effort_param=True)
        self.assertEqual(p.effort_value("high", eff), "high")
        self.assertEqual(p.effort_value("max", eff), "high")
        self.assertEqual(p.effort_value("medium", eff), "high")
        self.assertEqual(p.effort_value("minimal", eff), "low")
        self.assertIsNone(p.effort_value(None, eff))

    def test_openai_rules(self):
        p = cp.get_profile("openai")
        for mid in ("gpt-5.5", "gpt-6.1", "o3", "o4-mini"):
            self.assertEqual(set(p.dropped_params(MS(mid))), {"temperature", "top_p"}, mid)
            self.assertEqual(p.effort_value("xhigh", MS(mid)), "high")
            self.assertEqual(p.effort_value("max", MS(mid)), "high")
            self.assertEqual(p.effort_value("medium", MS(mid)), "medium")
            self.assertEqual(p.effort_value("low", MS(mid)), "low")
        for mid in ("gpt-4.1", "gpt-4o-mini"):
            self.assertEqual(set(p.dropped_params(MS(mid))), set())
            self.assertIsNone(p.effort_value("high", MS(mid)))

    def test_moonshot_deepseek_rules(self):
        m = cp.get_profile("moonshot")
        self.assertEqual(set(m.dropped_params(MS("kimi-k3"))), {"temperature", "top_p"})
        self.assertIsNone(m.effort_value("high", MS("kimi-k3")))
        self.assertFalse(m.tool_choice_required)
        d = cp.get_profile("deepseek")
        self.assertEqual(set(d.dropped_params(MS("deepseek-reasoner"))), {"temperature"})
        self.assertEqual(set(d.dropped_params(MS("deepseek-v4-thinking"))), {"temperature"})
        self.assertEqual(set(d.dropped_params(catalog.find_model("deepseek", "deepseek-v4-pro"))), {"temperature"})
        self.assertEqual(set(d.dropped_params(MS("deepseek-v4-chat"))), set())
        self.assertFalse(d.vision)

    def test_effort_styles(self):
        o = cp.get_profile("openrouter")
        self.assertEqual(o.effort_style, "openrouter")
        self.assertEqual(o.effort_value("max", MS("x-ai/grok-4.7")), "high")
        self.assertIsNone(o.effort_value("none", MS("x-ai/grok-4.7")))
        ol = cp.get_profile("ollama")
        self.assertEqual(ol.effort_value("medium", MS("qwen3:8b")), "medium")
        self.assertIsNone(ol.effort_value("none", MS("qwen3:8b")))
        for name in ("moonshot", "deepseek", "opencode", "nvidia", "generic"):
            self.assertIsNone(cp.get_profile(name).effort_value("high", MS("m")), name)

    def test_with_options(self):
        base = cp.get_profile("generic")
        self.assertIs(base.with_options({}), base)
        self.assertIs(base.with_options({"unrelated": 1}), base)
        p = base.with_options({"stream_usage": False, "max_tokens_field": "max_completion_tokens",
                               "default_max_tokens": 1000, "response_format": True})
        self.assertIsNot(p, base)
        self.assertEqual((p.stream_usage, p.max_tokens_field, p.default_max_tokens, p.response_format),
                         (False, "max_completion_tokens", 1000, True))
        self.assertTrue(base.stream_usage)  # shared instance untouched
        self.assertEqual(repr(p), "ChatProfile('generic')")


if __name__ == "__main__":
    unittest.main()
