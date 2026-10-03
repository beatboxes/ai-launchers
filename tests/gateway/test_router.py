"""router: every DESIGN §5.2 resolution rule, the 404 message and the /v1/models listing (§5.3)."""

import re
import threading
import unittest

from ._pkg import mod

router = mod("router")
config = mod("config")
errors = mod("errors")
presets = mod("presets")

M = 1000000


def make_table(**overrides):
    d = {
        "providers": {
            "xai": {"display_name": "xAI API", "dialect": "openai_chat", "profile": "xai",
                    "base_url": "https://api.x.ai/v1", "auth": {"kind": "api_key", "secret": "xai"},
                    "allow_unlisted": True,
                    "models": [{"id": "grok-4.7", "context": 2 * M, "reasoning": True},
                               {"id": "grok-4.20-0309-non-reasoning", "context": 2 * M},
                               {"id": "grok-4.3", "context": 2 * M}]},
            "codex": {"display_name": "ChatGPT (codex login)", "dialect": "responses", "target": "chatgpt_codex",
                      "base_url": "https://chatgpt.com/backend-api/codex", "auth": {"kind": "codex_chatgpt"},
                      "models": [{"id": "gpt-5.5", "context": 400000}]},
            "ollama": {"display_name": "Ollama", "dialect": "anthropic_passthrough", "profile": "ollama",
                       "base_url": "http://localhost:11434", "auth": {"kind": "none"},
                       "fallback": {"dialect": "openai_chat", "profile": "ollama",
                                    "base_url": "http://localhost:11434/v1"}},
            "opencode": {"display_name": "OpenCode CLI", "dialect": "cli", "auth": {"kind": "none"},
                         "chat_only": True, "allow_unlisted": True, "options": {"cli": "opencode"},
                         "models": [{"id": "opencode/grok-code", "tools": False}]},
            "gem": {"display_name": "Gemini API", "dialect": "gemini", "target": "gemini_api",
                    "base_url": "https://generativelanguage.googleapis.com",
                    "auth": {"kind": "api_key", "secret": "gemini", "style": "x-goog-api-key"},
                    "models": [{"id": "gemini-3.1-pro-preview", "context": 1048576}]},
        },
        "roles": {"default": "xai,grok-4.7", "background": "xai,grok-4.20-0309-non-reasoning",
                  "longContext": "gem,gemini-3.1-pro-preview", "think": "codex,gpt-5.5"},
        "aliases": {"fry-grok-4-3": "xai,grok-4.3", "fast": "xai,grok-4.20-0309-non-reasoning"},
        "provider_aliases": {"grok": ["xai"], "openai": ["codex"], "local-ollama": ["ollama"]},
        "long_context_threshold": 60000,
    }
    d.update(overrides)
    return config.RouteTable.from_dict(d)


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.table = make_table()
        self.r = router.Router(self.table, launcher_name="ai-grok")

    def route(self, name, est=None):
        res = self.r.resolve(name, est)
        return res.provider.id, res.model

    def assertNotFound(self, name, est=None):
        with self.assertRaises(router.RouteNotFound) as cm:
            self.r.resolve(name, est)
        e = cm.exception
        self.assertIsInstance(e, errors.GatewayError)
        self.assertEqual((e.status, e.err_type, e.should_retry), (404, "not_found_error", False))
        return e

    def test_prefix_and_1m_stripped(self):
        res = self.r.resolve("claude-via-xai,grok-4.7[1m]")
        self.assertEqual((res.provider.id, res.model, res.role, res.background), ("xai", "grok-4.7", None, False))
        self.assertEqual(res.requested, "claude-via-xai,grok-4.7[1m]")
        self.assertEqual(res.model_spec.context, 2 * M)
        self.assertEqual(self.route("CLAUDE-VIA-xai,grok-4.7[1M]"), ("xai", "grok-4.7"))
        self.assertEqual(self.route("xai,grok-4.7"), ("xai", "grok-4.7"))
        self.assertEqual(self.route("claude-via-xai, grok-4.7 "), ("xai", "grok-4.7"))

    def test_role_aliases(self):
        res = self.r.resolve("claude-via-background")
        self.assertEqual((res.provider.id, res.model, res.role, res.background),
                         ("xai", "grok-4.20-0309-non-reasoning", "background", True))
        res = self.r.resolve("claude-via-default")
        self.assertEqual((res.model, res.role, res.background), ("grok-4.7", "default", False))
        res = self.r.resolve("claude-via-subagent")  # unset -> default route
        self.assertEqual((res.model, res.role, res.background), ("grok-4.7", "subagent", False))
        res = self.r.resolve("claude-via-longcontext")
        self.assertEqual((res.provider.id, res.role), ("gem", "longContext"))
        for ignored in ("claude-via-think", "claude-via-websearch", "claude-via-webSearch"):
            res = self.r.resolve(ignored)
            self.assertEqual((res.model, res.role), ("grok-4.7", "default"), ignored)

    def test_unset_background_uses_default_but_stays_background(self):
        r = router.Router(make_table(roles={"default": "xai,grok-4.7"}))
        res = r.resolve("claude-via-background")
        self.assertEqual((res.model, res.role, res.background), ("grok-4.7", "background", True))

    def test_legacy_fry_forms(self):
        self.assertEqual(self.route("claude-via-ollama,fry-grok-4-3"), ("xai", "grok-4.3"))  # via alias table
        # no alias entry: built-in conversion, grok -> xai through provider_aliases
        self.assertEqual(self.route("ollama,fry-grok-4-20-0309-reasoning"), ("xai", "grok-4.20-0309-reasoning"))
        self.assertEqual(self.route("ollama,fry-grok-4-20-0309-non-reasoning"),
                         ("xai", "grok-4.20-0309-non-reasoning"))
        self.assertEqual(self.route("ollama,fry-grok-4-7"), ("xai", "grok-4.7"))
        self.assertEqual(self.route("ollama,fry-grok-build"), ("xai", "grok-build"))
        self.assertEqual(self.route("ollama,fry-codex-gpt-5.5"), ("codex", "gpt-5.5"))
        self.assertEqual(self.route("ollama,fry-opencode-opencode/grok-code"), ("opencode", "opencode/grok-code"))
        self.assertNotFound("local-ollama,llama3")  # ollama lists nothing and disallows unlisted
        self.r.set_discovered("ollama", ["llama3", "qwen3:8b"])
        self.assertEqual(self.route("local-ollama,llama3"), ("ollama", "llama3"))
        self.assertEqual(self.route("claude-via-ollama,qwen3:8b"), ("ollama", "qwen3:8b"))

    def test_grok_provider_preferred_when_configured(self):
        table = presets.route_table_from_presets(["xai", "grok"])
        r = router.Router(table)
        res = r.resolve("ollama,fry-grok-4-3")
        self.assertEqual((res.provider.id, res.model), ("grok", "grok-4.3"))
        self.assertIsNotNone(res.provider.fallback)  # primary spec, never the fallback
        self.assertEqual(res.provider.dialect, "responses")

    def test_provider_aliases_and_listing_rules(self):
        self.assertEqual(self.route("openai,gpt-5.5"), ("codex", "gpt-5.5"))
        self.assertEqual(self.route("grok,grok-4.7"), ("xai", "grok-4.7"))
        self.assertEqual(self.route("XAI,GROK-4.7"), ("xai", "grok-4.7"))
        self.assertEqual(self.route("xai,grok-9-unlisted"), ("xai", "grok-9-unlisted"))
        self.assertNotFound("codex,gpt-9")
        self.assertNotFound("nobody,model")
        self.assertNotFound("xai,")
        res = self.r.resolve("xai,grok-9-unlisted")
        self.assertEqual(res.model_spec.id, "grok-9-unlisted")
        self.assertIsNone(res.model_spec.context)

    def test_bare_ids_and_aliases(self):
        self.assertEqual(self.route("grok-4.7"), ("xai", "grok-4.7"))
        self.assertEqual(self.route("gpt-5.5"), ("codex", "gpt-5.5"))
        self.assertEqual(self.route("claude-via-gpt-5.5"), ("codex", "gpt-5.5"))
        res = self.r.resolve("fast")
        self.assertEqual((res.model, res.role, res.background), ("grok-4.20-0309-non-reasoning", None, False))
        self.assertEqual(self.route("claude-via-FAST"), ("xai", "grok-4.20-0309-non-reasoning"))
        self.assertNotFound("llama3")
        self.r.set_discovered("ollama", ["llama3"])
        self.assertEqual(self.route("llama3"), ("ollama", "llama3"))

    def test_claude_family_names(self):
        for name in ("claude-sonnet-4-6", "claude-opus-4-1-20250805", "opus", "sonnet", "fable", "Sonnet[1m]",
                     "anthropic/claude-3", "claude-via-claude-sonnet-5"):
            res = self.r.resolve(name)
            self.assertEqual((res.model, res.role, res.background), ("grok-4.7", "default", False), name)
        for name in ("claude-3-5-haiku-20241022", "haiku", "claude-haiku-4-5"):
            res = self.r.resolve(name)
            self.assertEqual((res.model, res.role, res.background),
                             ("grok-4.20-0309-non-reasoning", "background", True), name)

    def test_long_context(self):
        self.assertEqual(self.route("claude-via-xai,grok-4.7", 70000), ("gem", "gemini-3.1-pro-preview"))
        self.assertEqual(self.r.resolve("claude-via-xai,grok-4.7", 70000).role, "longContext")
        self.assertEqual(self.route("claude-via-xai,grok-4.7", 60000), ("xai", "grok-4.7"))
        self.assertEqual(self.route("claude-via-default", 70000), ("gem", "gemini-3.1-pro-preview"))
        self.assertEqual(self.route("claude-sonnet-4-6", 70000), ("gem", "gemini-3.1-pro-preview"))
        self.assertEqual(self.route("claude-via-codex,gpt-5.5", 70000), ("codex", "gpt-5.5"))
        self.assertEqual(self.route("claude-via-background", 900000), ("xai", "grok-4.20-0309-non-reasoning"))
        self.assertEqual(self.route("claude-via-xai,grok-4.7", None), ("xai", "grok-4.7"))
        r = router.Router(make_table(roles={"default": "xai,grok-4.7"}))
        self.assertEqual(r.resolve("claude-via-xai,grok-4.7", 10 ** 6).model, "grok-4.7")

    def test_not_found_message(self):
        e = self.assertNotFound("claude-via-nope,x")
        ids = [d["id"] for d in self.r.models_response()["data"]]
        self.assertGreater(len(ids), 5)
        self.assertEqual(e.message, "model 'claude-via-nope,x' is not routable via this launcher; available: %s… "
                                    "(run `ai-grok models`)" % ", ".join(ids[:5]))
        for bad in ("", "   ", None, "claude-via-", "nothing-matches"):
            self.assertNotFound(bad)
        e = router.RouteNotFound("m", ["a", "b"])
        self.assertIn("available: a, b (run the launcher's `models` command)", e.message)
        self.assertEqual(e.body()["error"]["type"], "not_found_error")

    def test_resolution_value_semantics(self):
        a = self.r.resolve("claude-via-xai,grok-4.7")
        b = self.r.resolve("claude-via-xai,grok-4.7")
        self.assertEqual(a, b)
        self.assertEqual(a.route(), "xai,grok-4.7")
        self.assertIn("xai,grok-4.7", repr(a))


class ModelsListingTests(unittest.TestCase):
    def setUp(self):
        self.table = make_table()
        self.r = router.Router(self.table)

    def test_shape_and_order(self):
        body = self.r.models_response()
        self.assertEqual(set(body), {"data", "has_more", "first_id", "last_id"})
        self.assertFalse(body["has_more"])
        data = body["data"]
        self.assertEqual(body["first_id"], data[0]["id"])
        self.assertEqual(body["last_id"], data[-1]["id"])
        self.assertEqual(data[0]["id"], "claude-via-xai,grok-4.7[1m]")  # default first; 2M ctx -> [1m]
        ids = [d["id"] for d in data]
        self.assertEqual(len(ids), len(set(ids)))
        providers = [i.split(",", 1)[0][len("claude-via-"):] for i in ids[1:]]
        order = [p for i, p in enumerate(providers) if p not in providers[:i]]
        self.assertEqual(order, ["xai", "codex", "opencode", "gem"])
        for d in data:
            self.assertEqual(set(d), {"type", "id", "display_name", "created_at", "description"})
            self.assertEqual(d["type"], "model")
            self.assertEqual(d["created_at"], "2026-01-01T00:00:00Z")
            self.assertTrue(re.search(r"(claude|anthropic)", d["id"], re.I))
            self.assertTrue(d["id"].startswith("claude-via-"))

    def test_entries_text(self):
        by_id = {d["id"]: d for d in self.r.models_response()["data"]}
        e = by_id["claude-via-xai,grok-4.7[1m]"]
        self.assertEqual(e["display_name"], "grok-4.7 · xAI API")
        self.assertEqual(e["description"], "API key · tools · 2M ctx")
        self.assertEqual(by_id["claude-via-codex,gpt-5.5"]["description"], "ChatGPT login · tools · 400k ctx")
        self.assertEqual(by_id["claude-via-opencode,opencode/grok-code"]["description"],
                         "opencode CLI · chat-only (no tools) · unknown ctx")
        self.assertEqual(by_id["claude-via-gem,gemini-3.1-pro-preview[1m]"]["description"],
                         "API key · tools · 1M ctx")

    def test_listed_ids_resolve_back(self):
        self.r.set_discovered("ollama", ["llama3"])
        for d in self.r.models_response()["data"]:
            res = self.r.resolve(d["id"])
            self.assertEqual(self.table.picker_id(res.provider.id, res.model_spec), d["id"])

    def test_discovered_and_cap(self):
        self.r.set_discovered("ollama", ["m%02d" % i for i in range(80)] + ["m00", "", "bad,id", 5])
        data = self.r.models_response()["data"]
        oll = [d for d in data if d["id"].startswith("claude-via-ollama,")]
        self.assertEqual(len(oll), router.MAX_MODELS_PER_PROVIDER)
        self.assertEqual(oll[0]["description"], "no key · tools · unknown ctx")
        self.assertEqual(self.r.discovered("ollama")[:2], ["m00", "m01"])
        self.assertEqual(len(self.r.discovered("ollama")), 80)

    def test_model_entry(self):
        e = self.r.model_entry("claude-via-xai,grok-4.7[1m]")
        self.assertEqual(e["id"], "claude-via-xai,grok-4.7[1m]")
        self.assertEqual(self.r.model_entry("claude-via-xai,grok-4.7")["id"], "claude-via-xai,grok-4.7[1m]")
        self.assertEqual(self.r.model_entry("CLAUDE-VIA-CODEX,GPT-5.5")["id"], "claude-via-codex,gpt-5.5")
        e = self.r.model_entry("claude-via-background")
        self.assertEqual(e["id"], "claude-via-background")
        self.assertEqual(e["display_name"], "grok-4.20-0309-non-reasoning · xAI API")
        self.assertIsNone(self.r.model_entry("claude-via-nope,x"))
        self.assertIsNone(self.r.model_entry(""))

    def test_no_default_role(self):
        r = router.Router(make_table(roles={}))
        data = r.models_response()["data"]
        self.assertEqual(data[0]["id"], "claude-via-xai,grok-4.7[1m]")
        with self.assertRaises(router.RouteNotFound):
            r.resolve("claude-via-default")

    def test_concurrent_discovery_updates(self):
        def worker(i):
            for j in range(50):
                self.r.set_discovered("ollama", ["w%d-%d" % (i, j)])
                self.r.models_response()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(len(self.r.discovered("ollama")), 1)


if __name__ == "__main__":
    unittest.main()
