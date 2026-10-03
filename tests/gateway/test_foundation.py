"""Foundation tests: compat, events, model helpers, config/SecretStore, static auth, RefreshingAuth,
transport (against local MockServers), dialect base helpers, presets/catalog, mock framework, fixtures."""

import copy
import datetime
import json
import os
import pickle
import socket
import threading
import time
import unittest

from ._pkg import mod

compat = mod("compat")
events = mod("events")
model = mod("model")
config = mod("config")
errors = mod("errors")
transport = mod("transport")
auth_pkg = mod("auth")
auth_base = mod("auth.base")
auth_static = mod("auth.static")
dialects = mod("dialects")
dialects_base = mod("dialects.base")
presets = mod("presets")
catalog = mod("catalog")
testing = mod("testing")
mocks = mod("testing.mock_upstreams")


# =========================================================================================
# compat
# =========================================================================================

class CompatTests(unittest.TestCase):
    def test_rfc3339_parse_variants(self):
        utc = datetime.timezone.utc
        cases = {
            "2026-10-03T02:14:08Z": datetime.datetime(2026, 10, 3, 2, 14, 8, tzinfo=utc),
            "2026-10-03t02:14:08z": datetime.datetime(2026, 10, 3, 2, 14, 8, tzinfo=utc),
            "2026-10-03T02:14:08.999Z": datetime.datetime(2026, 10, 3, 2, 14, 8, 999000, tzinfo=utc),
            "2026-10-03T02:14:08.123456789Z": datetime.datetime(2026, 10, 3, 2, 14, 8, 123456, tzinfo=utc),
            "2026-10-03T07:44:08+05:30": datetime.datetime(2026, 10, 3, 2, 14, 8, tzinfo=utc),
            "2026-10-02T18:14:08-0800": datetime.datetime(2026, 10, 3, 2, 14, 8, tzinfo=utc),
            "2026-10-03 02:14:08": datetime.datetime(2026, 10, 3, 2, 14, 8, tzinfo=utc),
            "2026-10-03T02:14:08+00": datetime.datetime(2026, 10, 3, 2, 14, 8, tzinfo=utc),
            "2016-12-31T23:59:60Z": datetime.datetime(2017, 1, 1, 0, 0, 0, tzinfo=utc),
        }
        for s, want in cases.items():
            got = compat.rfc3339_parse(s)
            self.assertEqual(got, want, s)
            self.assertEqual(got.tzinfo, utc)
        for bad in ("", "2026-10-03", "2026-13-03T00:00:00Z", "yesterday", "2026-10-03T25:00:00Z",
                    "2026-10-03T00:00:00+24:00", None, 12):
            with self.assertRaises(ValueError, msg=repr(bad)):
                compat.rfc3339_parse(bad)

    def test_rfc3339_epoch_and_format(self):
        self.assertEqual(compat.rfc3339_to_epoch("1970-01-01T00:00:01.5Z"), 1.5)
        self.assertEqual(compat.rfc3339_format(0), "1970-01-01T00:00:00Z")
        self.assertEqual(compat.rfc3339_format(1.25), "1970-01-01T00:00:01.250000Z")
        self.assertEqual(compat.rfc3339_format(1.25, "seconds"), "1970-01-01T00:00:01Z")
        self.assertEqual(compat.rfc3339_format(1.25, "milli"), "1970-01-01T00:00:01.250Z")
        self.assertEqual(compat.rfc3339_format(datetime.datetime(2026, 1, 2, 3, 4, 5)), "2026-01-02T03:04:05Z")
        tz = datetime.timezone(datetime.timedelta(hours=2))
        self.assertEqual(compat.rfc3339_format(datetime.datetime(2026, 1, 2, 5, 4, 5, tzinfo=tz)),
                         "2026-01-02T03:04:05Z")
        now = compat.rfc3339_format()
        self.assertTrue(now.endswith("Z"))
        self.assertAlmostEqual(compat.rfc3339_to_epoch(now), time.time(), delta=5)
        for t in (0, 1700000000, 1700000000.123456):
            self.assertAlmostEqual(compat.rfc3339_to_epoch(compat.rfc3339_format(t, "micro")), t, places=5)
        with self.assertRaises(TypeError):
            compat.rfc3339_format("now")

    def test_b64url(self):
        for data in (b"", b"f", b"fo", b"foo", b"\xff\xfe\xfd\x00", os.urandom(33)):
            enc = compat.b64url_encode(data)
            self.assertNotIn("=", enc)
            self.assertNotIn("+", enc)
            self.assertNotIn("/", enc)
            self.assertEqual(compat.b64url_decode(enc), data)
            self.assertEqual(compat.b64url_decode(enc + "=="), data)  # extra padding tolerated
        self.assertEqual(compat.b64url_encode("hé"), compat.b64url_encode("hé".encode("utf-8")))
        self.assertEqual(compat.b64url_decode("+/8="), b"\xfb\xff")  # standard alphabet accepted
        self.assertEqual(compat.b64url_decode(" -_8 \n"), b"\xfb\xff")
        for bad in ("a", "@@@@", "abc!"):
            with self.assertRaises(ValueError):
                compat.b64url_decode(bad)

    def test_jwt_claims(self):
        claims = {"exp": 1790000000, "https://api.openai.com/auth": {"chatgpt_account_id": "acc"}}
        tok = "%s.%s.sig" % (compat.b64url_encode(b'{"alg":"none"}'), compat.b64url_encode(json.dumps(claims)))
        self.assertEqual(compat.jwt_claims(tok), claims)
        self.assertEqual(compat.jwt_claims(tok.encode()), claims)
        for bad in ("", "abc", "a.b.c", "a.%s.c" % compat.b64url_encode("[1,2]"), None, 5):
            self.assertEqual(compat.jwt_claims(bad), {})

    def test_misc(self):
        self.assertEqual(compat.removeprefix("claude-via-x", "claude-via-"), "x")
        self.assertEqual(compat.removeprefix("x", "claude-via-"), "x")
        self.assertEqual(compat.removeprefix("abc", ""), "abc")
        self.assertEqual(compat.removesuffix("grok[1m]", "[1m]"), "grok")
        self.assertEqual(compat.removesuffix("abc", ""), "abc")
        self.assertEqual(compat.json_dumps_compact({"a": [1, 2], "b": "é"}), '{"a":[1,2],"b":"é"}')
        self.assertAlmostEqual(compat.utcnow_epoch(), time.time(), delta=2)
        self.assertEqual(compat.utcnow().tzinfo, datetime.timezone.utc)


# =========================================================================================
# events
# =========================================================================================

class EventTests(unittest.TestCase):
    def test_equality_repr_hash(self):
        a = events.TextDelta(0, "hi")
        self.assertEqual(a, events.TextDelta(0, "hi"))
        self.assertNotEqual(a, events.TextDelta(1, "hi"))
        self.assertNotEqual(a, events.ThinkingDelta(0, "hi"))
        self.assertEqual(hash(a), hash(events.TextDelta(0, "hi")))
        self.assertEqual(repr(a), "TextDelta(key=0, text='hi')")
        self.assertEqual(repr(events.ToolCall("t1", "Bash", "{}")), "ToolCall(id='t1', name='Bash', input_json='{}')")
        self.assertFalse(hasattr(a, "__dict__"))
        self.assertEqual(len({a, events.TextDelta(0, "hi"), events.TextDelta(0, "x")}), 2)

    def test_defaults_and_coercion(self):
        u = events.Usage("10", None)
        self.assertEqual((u.input_tokens, u.output_tokens, u.cache_read, u.cache_write), (10, 0, 0, 0))
        self.assertEqual(events.Finish("end_turn"), events.Finish("end_turn", None))
        self.assertEqual(events.ThinkingDelta(3).text, "")
        self.assertTrue(events.StreamError("overloaded_error", "x", 1).retryable is True)
        self.assertEqual(events.Usage(1, 2, 3, 4).to_dict(),
                         {"event": "usage", "input_tokens": 1, "output_tokens": 2, "cache_read": 3, "cache_write": 4})
        kinds = {c.kind for c in (events.TextDelta, events.ThinkingDelta, events.ThinkingSignature, events.ToolCall,
                                  events.Usage, events.Finish, events.StreamError)}
        self.assertEqual(len(kinds), 7)


# =========================================================================================
# model
# =========================================================================================

class ModelTests(unittest.TestCase):
    def test_block_constructors(self):
        B = model.Block
        self.assertEqual(B.of_text("x"), B(type="text", text="x"))
        tr = B.of_tool_result("toolu_1", "out", is_error=1)
        self.assertEqual(tr.content, [B.of_text("out")])
        self.assertIs(tr.is_error, True)
        tr2 = B.of_tool_result("toolu_1", [B.of_text("a"), B.of_image_base64("image/png", "AAA"),
                                           B.of_document_text("doc"), B.of_document_base64("application/pdf", "JV")])
        self.assertEqual(tr2.result_text(), "a\ndoc")
        self.assertEqual([b.type for b in tr2.result_media()], ["image", "document"])
        tu = B.of_tool_use("toolu_1", "Bash", None)
        self.assertEqual(tu.input, {})
        self.assertEqual(B.of_thinking("t", "sig").signature, "sig")
        self.assertEqual(B.of_redacted_thinking("d").data, "d")
        self.assertEqual(B.of_image_url("http://x").url, "http://x")
        self.assertTrue(B.of_document_url("http://x", "T").is_media())
        for t in model.BLOCK_TYPES:
            self.assertIsInstance(t, str)

    def test_normalized_request_helpers(self):
        B = model.Block
        req = model.NormalizedRequest(
            model="m", tools=[model.ToolDef("Bash"), model.ToolDef("Read")],
            messages=[model.Message("assistant", [B.of_tool_use("t1", "Gone", {}), B.of_tool_use("t2", "Bash", {})]),
                      model.Message("user", [B.of_tool_result("t1", "x")])], max_tokens=1)
        self.assertEqual(req.tool_names(), ["Bash", "Read"])
        self.assertEqual(req.all_tool_names(), ["Bash", "Read", "Gone"])
        self.assertEqual(req.tool_use_names_by_id(), {"t1": "Gone", "t2": "Bash"})
        self.assertTrue(req.is_probe())
        self.assertEqual(req.messages[0].tool_uses()[1].name, "Bash")
        self.assertEqual(len(req.messages[1].tool_results()), 1)
        self.assertEqual(model.ToolDef("x").input_schema, {"type": "object", "properties": {}})

    def test_fold_system_messages_fixture(self):
        fx = testing.load_fixture("turn2_tool_result_request.json")
        raw = fx["body"]["messages"]
        before = copy.deepcopy(raw)
        res = model.fold_system_messages_raw(raw)
        self.assertEqual(raw, before, "input must not be mutated")
        self.assertEqual(res.folded, 2)
        self.assertEqual([m["role"] for m in res.messages], ["user", "assistant", "user"])
        first = res.messages[0]["content"]
        self.assertTrue(first[-1]["text"].startswith("<system-reminder>\n# Environment"))
        self.assertTrue(first[-1]["text"].endswith("\n</system-reminder>"))
        last = res.messages[-1]["content"]
        self.assertEqual(last[0]["type"], "tool_result")
        self.assertIn("<total_tokens>", last[-1]["text"])
        self.assertEqual(last[-1].get("cache_control"), None)

    def test_fold_system_messages_placement(self):
        sysmsg = {"role": "system", "content": [
            {"type": "text", "text": "S"},
            {"type": "tool_addition", "tool": {"type": "tool_definition",
                                               "definition": {"name": "New", "input_schema": {"type": "object"}}}},
            {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "Ref"}},
            {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "Old"}}], "clear_at": "next_user_message"}
        # first message
        r = model.fold_system_messages_raw([sysmsg, {"role": "user", "content": "hi"}])
        self.assertEqual(r.messages, [{"role": "user", "content": [
            {"type": "text", "text": "<system-reminder>\nS\n</system-reminder>"}, {"type": "text", "text": "hi"}]}])
        self.assertEqual([d["name"] for d in r.tool_additions], ["New"])
        self.assertEqual(r.tool_reference_additions, ["Ref"])
        self.assertEqual(r.tool_removals, ["Old"])
        # after assistant, before assistant -> standalone user message
        r = model.fold_system_messages_raw([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                            {"role": "system", "content": "S"}, {"role": "assistant", "content": "c"}],
                                           wrap=False)
        self.assertEqual([m["role"] for m in r.messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual(r.messages[2]["content"], [{"type": "text", "text": "S"}])
        # trailing after assistant
        r = model.fold_system_messages_raw([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                            {"role": "system", "content": "S"}])
        self.assertEqual(r.messages[-1]["role"], "user")
        # empty system message dropped, no-op otherwise
        r = model.fold_system_messages_raw([{"role": "user", "content": "a"}, {"role": "system", "content": " "}])
        self.assertEqual(r.messages, [{"role": "user", "content": "a"}])
        tools = [{"name": "Old"}, {"name": "Keep"}]
        fold = model.fold_system_messages_raw([sysmsg])
        self.assertEqual([t["name"] for t in model.apply_tool_changes(tools, fold)], ["Keep", "New"])

    def test_effort_from_thinking(self):
        f = model.split_effort_from_thinking
        self.assertEqual(f({"type": "adaptive", "display": "updates"}), ("medium", True, None))
        self.assertEqual(f({"type": "enabled", "budget_tokens": 4096}), ("low", True, 4096))
        self.assertEqual(f({"type": "enabled", "budget_tokens": 16384}), ("medium", True, 16384))
        self.assertEqual(f({"type": "enabled", "budget_tokens": 16385}), ("high", True, 16385))
        self.assertEqual(f({"type": "disabled"}), ("none", False, None))
        self.assertEqual(f(None), (None, False, None))


class SignatureTests(unittest.TestCase):
    def test_round_trip_and_filtering(self):
        sigs = mod("signatures")
        s = sigs.encode_signature("codex_chatgpt", {"enc": "gAAA+/=", "summary": ["hé"]})
        self.assertTrue(s.startswith("fgw1.codex_chatgpt."))
        self.assertEqual(sigs.decode_signature(s), ("codex_chatgpt", {"enc": "gAAA+/=", "summary": ["hé"]}))
        self.assertEqual(sigs.signature_target(s), "codex_chatgpt")
        syn = sigs.synthetic_signature("thinking text")
        self.assertRegex(syn, r"^fgw1\.chat\.[0-9a-f]{12}$")
        self.assertEqual(sigs.decode_signature(syn), ("chat", {}))
        self.assertEqual(sigs.synthetic_signature(""), sigs.synthetic_signature(b""))
        for foreign in ("", None, "EqQBCkgIARABGAIiQ...", "fgw1.unknown.abc", "fgw1.vertex.%%%"):
            self.assertIsNone(sigs.decode_signature(foreign))
        self.assertTrue(sigs.keep_thinking_for(s, "codex_chatgpt"))
        self.assertFalse(sigs.keep_thinking_for(s, "openai_api"))
        self.assertFalse(sigs.keep_thinking_for("EqQBCkg", "chat"))
        self.assertTrue(sigs.keep_thinking_for(syn, "chat"))
        with self.assertRaises(ValueError):
            sigs.encode_signature("anthropic", {})
        self.assertEqual(sigs.signature_target("fgw1.chat.SPIKESIG2"), "chat")  # spike fixture value


# =========================================================================================
# config + SecretStore
# =========================================================================================

def _table_dict():
    return {
        "version": 1, "picker_prefix": "claude-via-", "long_context_threshold": 60000,
        "providers": {
            "xai": {"display_name": "xAI API", "dialect": "openai_chat", "profile": "xai",
                    "base_url": "https://api.x.ai/v1", "auth": {"kind": "api_key", "secret": "xai", "style": "bearer"},
                    "headers": {}, "allow_unlisted": True, "chat_only": False, "options": {},
                    "models": [{"id": "grok-4.7", "context": 2000000, "max_output": 64000, "reasoning": True,
                                "tools": True}]},
            "grok": {"dialect": "responses", "target": "grok_cli_proxy",
                     "base_url": "https://cli-chat-proxy.grok.com/v1", "auth": {"kind": "grok_cli"},
                     "models": ["grok-4.3"],
                     "fallback": {"dialect": "openai_chat", "profile": "xai", "base_url": "https://api.x.ai/v1"}}},
        "roles": {"default": "xai,grok-4.7", "background": "xai,grok-4.20-0309-non-reasoning", "longContext": None,
                  "subagent": None},
        "aliases": {"fry-grok-4-3": "grok,grok-4.3"},
        "provider_aliases": {"grok": ["xai"], "xai": ["grok"], "codex": ["openai"], "openai": ["codex"],
                             "local-ollama": ["ollama"]},
    }


class ConfigTests(unittest.TestCase):
    def test_design_example_round_trip(self):
        t = config.RouteTable.from_dict(_table_dict())
        self.assertEqual(t.problems(), [])
        t.validate()
        self.assertEqual(list(t.providers), ["xai", "grok"])
        grok = t.providers["grok"]
        self.assertEqual(grok.fallback.dialect, "openai_chat")
        self.assertEqual(grok.fallback.auth, {"kind": "grok_cli"})  # inherited
        self.assertEqual([m.id for m in grok.fallback.models], ["grok-4.3"])
        d = t.to_dict()
        self.assertEqual(d["providers"]["grok"]["fallback"],
                         {"dialect": "openai_chat", "profile": "xai", "base_url": "https://api.x.ai/v1"})
        self.assertEqual(config.RouteTable.from_dict(json.loads(json.dumps(d))), t)
        self.assertEqual(t.providers["xai"].find_model("GROK-4.7").id, "grok-4.7")
        self.assertEqual(t.providers["xai"].model_spec("other").id, "other")
        self.assertEqual(t.canonical_provider("grok"), "grok")
        self.assertEqual(t.canonical_provider("codex"), None)
        self.assertEqual(config.split_route(" xai , grok-4.7 "), ("xai", "grok-4.7"))
        self.assertEqual(config.split_route("a,b,c"), ("a", "b,c"))
        self.assertEqual(config.split_route("opus"), (None, "opus"))

    def test_picker_id(self):
        t = config.RouteTable.from_dict(_table_dict())
        self.assertEqual(t.picker_id("xai", "grok-4.7"), "claude-via-xai,grok-4.7[1m]")
        self.assertEqual(t.picker_id("grok", "grok-4.3"), "claude-via-grok,grok-4.3")
        self.assertEqual(t.picker_id("x", config.ModelSpec("m", context=200000)), "claude-via-x,m")

    def test_model_spec_round_trip(self):
        m = config.ModelSpec.from_dict({"id": "a", "context": "1000", "weird": 1, "_c": "comment"})
        self.assertEqual(m.context, 1000)
        self.assertEqual(m.extra, {"weird": 1})
        self.assertEqual(config.ModelSpec.from_dict(m.to_dict()), m)
        self.assertEqual(config.ModelSpec("x").to_dict(), {"id": "x"})
        with self.assertRaises(config.ConfigError):
            config.ModelSpec.from_dict({"context": 1})

    def test_validation_problems(self):
        d = _table_dict()
        d["providers"]["bad"] = {"dialect": "smoke_signals", "base_url": "ftp://x", "auth": {"kind": "api_key", "key": "sk-1"},
                                 "frobnicate": 1, "schema_mode": "weird", "models": [{"id": "a,b"}, {"id": "a,b"}]}
        d["providers"]["resp"] = {"dialect": "responses", "base_url": "https://x", "auth": {"kind": "none"}}
        d["providers"]["chat"] = {"dialect": "openai_chat", "target": "vertex", "auth": {"kind": "none"}}
        d["roles"]["default"] = "nope,model"
        d["roles"]["mystery"] = "xai,grok-4.7"
        d["aliases"]["x"] = "grok,not-listed"
        t = config.RouteTable.from_dict(d)
        probs = "\n".join(t.problems())
        for needle in ("unknown dialect 'smoke_signals'", "base_url must be http(s)", "needs 'secret'",
                       "inline secret", "unknown key 'frobnicate'", "unknown schema_mode", "invalid model id",
                       "duplicate model id", "needs target in", "takes no target", "base_url required",
                       "provider 'nope' is not configured", "unknown role 'mystery'", "not listed by provider 'grok'"):
            self.assertIn(needle, probs)
        with self.assertRaises(config.ConfigError) as cm:
            t.validate()
        self.assertGreater(len(cm.exception.problems), 10)
        bad_prefix = config.RouteTable.from_dict(dict(_table_dict(), picker_prefix="via-"))
        self.assertIn("picker_prefix", " ".join(bad_prefix.problems()))
        no_default = config.RouteTable.from_dict(dict(_table_dict(), roles={}))
        self.assertIn("roles.default is required", no_default.problems())
        provider_aliased = config.RouteTable.from_dict(dict(_table_dict(), roles={"default": "codex,gpt-5.5"}))
        self.assertIn("provider 'codex' is not configured", " ".join(provider_aliased.problems()))
        aliased = config.RouteTable.from_dict(dict(_table_dict(), roles={"default": "fry-grok-4-3"}))
        self.assertEqual(aliased.problems(), [])

    def test_env_overrides(self):
        t = config.RouteTable.from_dict(_table_dict())
        env = {"AI_GATEWAY_UPSTREAM_XAI": "http://127.0.0.1:1111/v1/",
               "AI_GATEWAY_UPSTREAM_GROK_FALLBACK": "http://127.0.0.1:2222/v1",
               "AI_GATEWAY_UPSTREAM_GROK": ""}
        t2 = config.apply_upstream_env_overrides(t, env)
        self.assertEqual(t2.providers["xai"].base_url, "http://127.0.0.1:1111/v1")
        self.assertEqual(t2.providers["grok"].base_url, "https://cli-chat-proxy.grok.com/v1")
        self.assertEqual(t2.providers["grok"].fallback.base_url, "http://127.0.0.1:2222/v1")
        self.assertEqual(t.providers["xai"].base_url, "https://api.x.ai/v1", "original untouched")
        self.assertEqual(config.upstream_env_name("gemini-vertex"), "AI_GATEWAY_UPSTREAM_GEMINI_VERTEX")
        self.assertEqual(config.upstream_env_name("local.ollama", True), "AI_GATEWAY_UPSTREAM_LOCAL_OLLAMA_FALLBACK")
        self.assertEqual(len(config.describe_upstream_env_overrides(t, env)), 2)


class SecretStoreTests(unittest.TestCase):
    SECRET = "sk-FAKEKEY-SENTINEL-123456"

    def test_basic_and_redaction(self):
        s = config.SecretStore({"xai": self.SECRET, "empty": ""})
        self.assertTrue(s.has("xai"))
        self.assertFalse(s.has("empty"))
        self.assertIn("xai", s)
        self.assertEqual(len(s), 1)
        self.assertEqual(s.get("xai"), self.SECRET)
        self.assertEqual(s.names(), ["xai"])
        for text in (repr(s), str(s), "%s" % s, "{}".format(s)):
            self.assertNotIn(self.SECRET, text)
            self.assertIn("xai", text)
        self.assertEqual(s.redact("key=%s!" % self.SECRET), "key=***!")
        s.set("xai", None)
        self.assertFalse(s.has("xai"))
        with self.assertRaises(ValueError):
            s.set("", "x")

    def test_never_serializes(self):
        s = config.SecretStore({"xai": self.SECRET})
        with self.assertRaises(TypeError):
            pickle.dumps(s)
        with self.assertRaises(TypeError):
            json.dumps(s)
        with self.assertRaises(TypeError):
            json.dumps({"store": s})
        c = copy.deepcopy({"s": s})["s"]
        self.assertEqual(c.get("xai"), self.SECRET)
        self.assertIsNot(c, s)
        self.assertEqual(copy.copy(s).get("xai"), self.SECRET)


# =========================================================================================
# auth
# =========================================================================================

class StaticAuthTests(unittest.TestCase):
    def test_styles(self):
        store = config.SecretStore({"k": "KEY"})
        cases = {"bearer": {"Authorization": "Bearer KEY"}, "x-api-key": {"x-api-key": "KEY"},
                 "x-goog-api-key": {"x-goog-api-key": "KEY"}, "none": {}}
        for style, want in cases.items():
            a = auth_static.StaticKeyAuth("k", store, style, provider_id="p")
            self.assertTrue(a.available())
            self.assertEqual(a.headers(), want)
            d = a.describe()
            self.assertNotIn("KEY", json.dumps(d))
            self.assertEqual(d["source"], "secret:k")
        with self.assertRaises(ValueError):
            auth_static.StaticKeyAuth("k", store, "basic")

    def test_missing_key_and_rotation(self):
        store = config.SecretStore()
        a = auth_static.StaticKeyAuth("k", store, provider_id="p")
        self.assertFalse(a.available())
        with self.assertRaises(auth_base.AuthError) as cm:
            a.headers()
        self.assertIn("'p'", str(cm.exception))
        self.assertFalse(a.on_unauthorized({}))
        store.set("k", "one")
        used = a.headers()
        self.assertFalse(a.on_unauthorized(used))
        store.set("k", "two")
        self.assertTrue(a.on_unauthorized(used))  # key changed since -> retry once

    def test_make_auth(self):
        store = config.SecretStore({"xai": "K"})
        a = auth_pkg.make_auth({"kind": "api_key", "secret": "xai", "style": "x-api-key"}, store, "xai")
        self.assertIsInstance(a, auth_static.StaticKeyAuth)
        self.assertEqual(a.headers(), {"x-api-key": "K"})
        self.assertIsInstance(auth_pkg.make_auth({"kind": "none"}, store, "o"), auth_static.NoAuth)
        self.assertIsInstance(auth_pkg.make_auth(None, store, "o"), auth_static.NoAuth)
        self.assertEqual(auth_pkg.make_auth({"kind": "none"}, store, "o").headers(), {})
        with self.assertRaises(ValueError):
            auth_pkg.make_auth({"kind": "magic"}, store, "o")
        self.assertEqual(set(auth_pkg.AUTH_CLASSES), {"codex_chatgpt", "grok_cli", "gcloud_adc"})

    def test_redact_headers(self):
        h = auth_base.redact_headers({"Authorization": "Bearer x", "x-goog-api-key": "k", "Accept": "a",
                                      "ChatGPT-Account-ID": "acc"})
        self.assertEqual(h["Authorization"], "<redacted>")
        self.assertEqual(h["x-goog-api-key"], "<redacted>")
        self.assertEqual(h["Accept"], "a")


class _FakeRefreshing(auth_base.RefreshingAuth):
    kind = "fake"

    def __init__(self, expires_in=3600):
        auth_base.RefreshingAuth.__init__(self, "fake")
        self.refreshes = 0
        self.expires_in = expires_in
        self.fail = False

    def _load(self):
        return auth_base.Token("t0", time.time() + self.expires_in)

    def _refresh(self, token):
        if self.fail:
            raise auth_base.AuthError("refresh failed", "run `fake login`")
        time.sleep(0.05)
        self.refreshes += 1
        return auth_base.Token("t%d" % self.refreshes, time.time() + 3600)


class RefreshingAuthTests(unittest.TestCase):
    def test_headers_and_expiry(self):
        a = _FakeRefreshing()
        self.assertTrue(a.available())
        self.assertEqual(a.headers(), {"Authorization": "Bearer t0"})
        self.assertEqual(a.refreshes, 0)
        self.assertEqual(a.headers(force_refresh=True), {"Authorization": "Bearer t1"})
        b = _FakeRefreshing(expires_in=10)  # inside the 300 s margin -> refresh on first use
        self.assertEqual(b.headers(), {"Authorization": "Bearer t1"})
        self.assertIn("expires_at", b.describe())
        self.assertNotIn("t1", json.dumps(b.describe()))
        self.assertNotIn("t1", repr(b.current_token()))

    def test_single_flight_on_unauthorized(self):
        a = _FakeRefreshing()
        used = a.headers()
        results = []
        threads = [threading.Thread(target=lambda: results.append(a.on_unauthorized(used))) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, [True] * 8)
        self.assertEqual(a.refreshes, 1, "only one refresh for concurrent 401s")
        self.assertEqual(a.headers(), {"Authorization": "Bearer t1"})

    def test_refresh_failure(self):
        a = _FakeRefreshing()
        used = a.headers()
        a.fail = True
        self.assertFalse(a.on_unauthorized(used))
        with self.assertRaises(auth_base.AuthError) as cm:
            a.headers(force_refresh=True)
        self.assertIn("fake login", str(cm.exception))


# =========================================================================================
# transport
# =========================================================================================

def _sse_handler(server, req, resp):
    if req.path == "/json":
        resp.send_json(200, {"ok": True, "ua": req.header("user-agent"), "body": req.json},
                       headers={"X-Thing": "1"})
    elif req.path == "/err":
        resp.send_json(int(req.query.get("status", ["500"])[0]), {"error": {"message": "boom"}})
    elif req.path == "/sse":
        w = resp.start_sse()
        w.raw(b": comment line\n\n")
        w.event("message_start", {"type": "message_start"})
        w.raw(b"event: multi\r\ndata: line1\r\ndata: line2\r\nid: 7\r\n\r\n")
        w.raw(b"data: {\"a\":\n")  # split across chunks
        w.raw(b"data: 1}\n\n")
        w.data("[DONE]")
        w.close()
    elif req.path == "/slow":
        w = resp.start_sse()
        w.data({"n": 1})
        time.sleep(float(req.query.get("pause", ["0.6"])[0]))
        w.data({"n": 2})
        w.close()
    elif req.path == "/ndjson":
        w = resp.start_chunked(content_type="application/x-ndjson")
        w.write(b'{"a":1}\n\n{"b":2}\n')
        w.close()
    elif req.path == "/hang":
        w = resp.start_sse()
        w.data({"n": 1})
        time.sleep(3)
        w.close()


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = mocks.MockServer(handler=_sse_handler).start()
        cls.client = transport.HttpClient(timeout=10, user_agent="test-ua/1", environ={})

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_json_request(self):
        r = self.client.request("POST", self.srv.url + "/json", {"Content-Type": "application/json"},
                                b'{"x":1}', stream=False)
        self.assertEqual(r.status, 200)
        self.assertEqual(r.headers.get("x-thing"), "1")
        self.assertEqual(r.headers["X-THING"], "1")
        self.assertEqual(r.json(), {"ok": True, "ua": "test-ua/1", "body": {"x": 1}})
        self.assertEqual(r.read(), r.read())
        self.assertTrue(r.closed)
        rec = self.srv.requests_for("/json")[-1]
        self.assertEqual(rec["headers"].get("Content-Length"), "7")

    def test_http_errors_do_not_raise(self):
        for status in (400, 401, 404, 429, 500, 529):
            r = self.client.request("GET", self.srv.url + "/err?status=%d" % status)
            self.assertEqual(r.status, status)
            self.assertIn("boom", r.text())

    def test_connection_refused(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        with self.assertRaises(transport.TransportError) as cm:
            self.client.request("GET", "http://127.0.0.1:%d/x?secret=1" % port)
        self.assertNotIn("secret", str(cm.exception))
        with self.assertRaises(transport.TransportError):
            self.client.request("GET", "ftp://127.0.0.1/x")

    def test_sse_parsing_over_http(self):
        r = self.client.request("GET", self.srv.url + "/sse")
        evs = list(transport.iter_sse(r))
        self.assertTrue(r.closed)
        self.assertEqual([(e.event, e.data) for e in evs], [
            ("message_start", '{"type":"message_start"}'),
            ("multi", "line1\nline2"),
            ("message", '{"a":\n1}'),
            ("message", "[DONE]"),
        ])
        self.assertEqual(evs[1].id, "7")
        self.assertEqual(evs[2].json(), {"a": 1})

    def test_iter_lines_and_ndjson(self):
        r = self.client.request("GET", self.srv.url + "/ndjson")
        self.assertEqual(list(transport.iter_ndjson(r)), [{"a": 1}, {"b": 2}])
        r = self.client.request("GET", self.srv.url + "/ndjson", stream=False)
        self.assertEqual(list(r.iter_lines()), [b'{"a":1}', b"", b'{"b":2}'])
        with self.assertRaises(ValueError):
            list(transport.iter_ndjson([b"{bad"]))

    def test_heartbeat_with_real_stream(self):
        r = self.client.request("GET", self.srv.url + "/slow?pause=0.6")
        items = list(transport.iter_with_heartbeat(transport.iter_sse(r), 0.1, on_close=r.close))
        beats = [i for i in items if i is transport.HEARTBEAT]
        data = [i.json() for i in items if i is not transport.HEARTBEAT]
        self.assertEqual(data, [{"n": 1}, {"n": 2}])
        self.assertGreaterEqual(len(beats), 3)
        self.assertLess(items.index(beats[0]), len(items) - 1)

    def test_heartbeat_early_close_unblocks_reader(self):
        r = self.client.request("GET", self.srv.url + "/hang")
        it = transport.iter_with_heartbeat(transport.iter_sse(r), 0.05, on_close=r.close)
        first = next(it)
        self.assertEqual(first.json(), {"n": 1})
        t0 = time.time()
        it.close()
        self.assertTrue(r.closed)
        deadline = time.time() + 2
        while time.time() < deadline and any(t.name == "gw-upstream-reader" and t.is_alive()
                                             for t in threading.enumerate()):
            time.sleep(0.02)
        self.assertFalse(any(t.name == "gw-upstream-reader" and t.is_alive() for t in threading.enumerate()))
        self.assertLess(time.time() - t0, 2.5)

    def test_proxy_selection(self):
        env = {"HTTPS_PROXY": "http://user:pw@proxy:3128", "NO_PROXY": "internal.example,10.0.0.0/8,.corp,[::2]:443",
               "http_proxy": "proxy2:8080"}
        self.assertEqual(transport.proxy_for_url("https://api.x.ai/v1", env), "http://user:pw@proxy:3128")
        self.assertEqual(transport.proxy_for_url("http://api.x.ai/v1", env), "http://proxy2:8080")
        for direct in ("https://internal.example/x", "https://a.internal.example/x", "https://10.1.2.3/x",
                       "https://x.corp/", "http://127.0.0.1:9/x", "http://localhost/x", "http://[::1]:5/x",
                       "https://[::2]/x"):
            self.assertIsNone(transport.proxy_for_url(direct, env), direct)
        self.assertIsNone(transport.proxy_for_url("https://x.ai/", {"ALL_PROXY": "socks5://p:1"}))
        self.assertEqual(transport.proxy_for_url("https://x.ai/", {"ALL_PROXY": "http://p:1"}), "http://p:1")
        self.assertTrue(transport.bypass_proxy("anything", 443, "*"))
        self.assertTrue(transport.bypass_proxy("api.x.ai", 443, "*.x.ai"))
        self.assertTrue(transport.bypass_proxy("x.ai", 8443, "x.ai:8443"))
        self.assertFalse(transport.bypass_proxy("x.ai", 443, "x.ai:8443"))
        self.assertFalse(transport.bypass_proxy("notx.ai", 443, "x.ai"))
        self.assertFalse(transport.bypass_proxy("x.ai", 443, ""))

    def test_connect_proxy_tunnel(self):
        """HTTPS_PROXY for an http:// target uses an absolute-URI request through the proxy."""
        seen = []

        def proxy_handler(server, req, resp):
            seen.append((req.raw_path, req.header("proxy-authorization")))
            resp.send_json(200, {"via": "proxy"})

        with mocks.MockServer(handler=proxy_handler) as proxy:
            env = {"http_proxy": "http://u:p@127.0.0.1:%d" % proxy.port}
            c = transport.HttpClient(timeout=5, environ=env)
            r = c.request("GET", "http://upstream.invalid:81/v1/models?x=1", stream=False)
            self.assertEqual(r.json(), {"via": "proxy"})
        self.assertEqual(seen[0][0], "http://upstream.invalid:81/v1/models?x=1")
        self.assertEqual(seen[0][1], "Basic dTpw")

    def test_ssl_context(self):
        ctx = transport.make_ssl_context(environ={})
        self.assertEqual(ctx.verify_mode, __import__("ssl").CERT_REQUIRED)
        self.assertTrue(ctx.check_hostname)
        import ssl as _ssl

        cafile = _ssl.get_default_verify_paths().cafile or os.environ.get("SSL_CERT_FILE")
        if cafile and os.path.isfile(cafile):
            c2 = transport.make_ssl_context(environ={"REQUESTS_CA_BUNDLE": cafile})
            self.assertIs(c2, transport.make_ssl_context(environ={"REQUESTS_CA_BUNDLE": cafile}))
        transport.make_ssl_context(environ={"SSL_CERT_FILE": "/nonexistent/ca.pem"})  # ignored, no crash

    def test_case_insensitive_dict(self):
        d = transport.CaseInsensitiveDict({"Content-Type": "a"})
        d["content-type"] = "b"
        self.assertEqual(len(d), 1)
        self.assertEqual(d.get("CONTENT-TYPE"), "b")
        self.assertIn("Content-type", d)
        self.assertEqual(d, {"content-TYPE": "b"})
        del d["CONTENT-TYPE"]
        self.assertEqual(len(d), 0)


class SSEParserTests(unittest.TestCase):
    def test_grammar(self):
        lines = ["﻿event: a", "data: 1", "", ": comment", "data:2", "data:  3", "", "event: e", "",
                 "id: 9", "retry: 1500", "data: x: y", "", "field-without-colon", "data", "", "data: tail"]
        evs = list(transport.iter_sse(lines))
        self.assertEqual([(e.event, e.data) for e in evs], [
            ("a", "1"), ("message", "2\n 3"), ("e", ""), ("message", "x: y"), ("message", ""), ("message", "tail")])
        self.assertEqual((evs[3].id, evs[3].retry), ("9", 1500))
        crlf = list(transport.iter_sse([b"event: z\r\n", b"data: q\r\n", b"\r\n"]))
        self.assertEqual(crlf, [transport.SSEEvent("z", "q")])

    def test_heartbeat_propagates_errors(self):
        def gen():
            yield 1
            time.sleep(0.25)
            raise RuntimeError("upstream died")

        out = []
        with self.assertRaises(RuntimeError):
            for item in transport.iter_with_heartbeat(gen(), 0.05):
                out.append(item)
        self.assertEqual(out[0], 1)
        self.assertIn(transport.HEARTBEAT, out)
        self.assertEqual(list(transport.iter_with_heartbeat(iter([1, 2, 3]), 0)), [1, 2, 3])


# =========================================================================================
# dialect base helpers
# =========================================================================================

class _Res(object):
    background = False


def _ctx(server_url, auth, est=None):
    prov = config.ProviderSpec(id="p", base_url=server_url, auth={"kind": "none"})
    return dialects_base.RequestContext(
        req=model.NormalizedRequest(model="claude-via-p,m"), resolution=_Res(), provider=prov,
        model=config.ModelSpec("m", context=1000), runtime=dialects_base.ProviderRuntime("p"), auth=auth,
        http=transport.HttpClient(timeout=5, environ={}), session_id="s", est_tokens=est)


class _CountingAuth(auth_base.AuthProvider):
    kind = "counting"

    def __init__(self, refresh_ok=True):
        auth_base.AuthProvider.__init__(self, "p")
        self.n = 0
        self.refresh_ok = refresh_ok

    def available(self):
        return True

    def headers(self, force_refresh=False):
        return {"Authorization": "Bearer tok%d" % self.n}

    def on_unauthorized(self, used):
        if not self.refresh_ok:
            return False
        self.n += 1
        return True

    def relogin_hint(self):
        return "run `x login`"


def _auth_handler(server, req, resp):
    if req.path == "/needs-tok1":
        if req.bearer() == "tok1":
            resp.send_json(200, {"ok": 1})
        else:
            resp.send_json(401, {"error": {"message": "expired token"}})
    elif req.path == "/403-expired":
        if req.bearer() == "tok1":
            resp.send_json(200, {"ok": 1})
        else:
            resp.send_json(403, {"detail": {"code": "token_expired"}})
    elif req.path == "/overflow":
        resp.send_json(400, {"error": {"message": "This model's maximum context length is 1000 tokens. However, "
                                                  "your messages resulted in 1500 tokens.",
                                       "code": "context_length_exceeded"}})
    else:
        resp.send_json(503, {"error": {"message": "unavailable"}})


class DialectBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = mocks.MockServer(handler=_auth_handler).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_401_refresh_retry_once(self):
        auth = _CountingAuth()
        ctx = _ctx(self.srv.url, auth)
        r = dialects_base.send_with_auth_retry(ctx, "POST", self.srv.url + "/needs-tok1", {"X-A": "1"}, b"{}")
        self.assertEqual(r.status, 200)
        r.close()
        recs = self.srv.requests_for("/needs-tok1")
        self.assertEqual([x["headers"]["Authorization"] for x in recs[-2:]], ["Bearer tok0", "Bearer tok1"])
        self.assertEqual(recs[-1]["headers"]["X-A"], "1")

    def test_401_no_refresh(self):
        ctx = _ctx(self.srv.url, _CountingAuth(refresh_ok=False))
        with self.assertRaises(errors.GatewayError) as cm:
            dialects_base.send_with_auth_retry(ctx, "POST", self.srv.url + "/needs-tok1", {}, b"{}")
        e = cm.exception
        self.assertEqual((e.status, e.err_type, e.should_retry), (401, "authentication_error", False))
        self.assertIn("run `x login`", e.message)

    def test_custom_unauthorized_predicate(self):
        ctx = _ctx(self.srv.url, _CountingAuth())
        r = dialects_base.send_with_auth_retry(ctx, "POST", self.srv.url + "/403-expired", {}, b"{}",
                                               unauthorized=lambda st, txt: st == 403 and "token_expired" in txt)
        self.assertEqual(r.status, 200)
        r.close()

    def test_error_mapping_and_connection_error(self):
        ctx = _ctx(self.srv.url, _CountingAuth(), est=1500)
        with self.assertRaises(errors.GatewayError) as cm:
            dialects_base.send_with_auth_retry(ctx, "POST", self.srv.url + "/overflow", {}, b"{}")
        self.assertEqual(cm.exception.message, "prompt is too long: 1500 tokens > 1000 maximum")
        self.assertEqual(cm.exception.upstream_status, 400)
        with self.assertRaises(errors.GatewayError) as cm:
            dialects_base.send_with_auth_retry(ctx, "POST", self.srv.url + "/other", {}, b"{}")
        self.assertEqual((cm.exception.status, cm.exception.err_type), (529, "overloaded_error"))
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        with self.assertRaises(errors.GatewayError) as cm:
            dialects_base.send_with_auth_retry(ctx, "GET", "http://127.0.0.1:%d/" % port, {}, None)
        self.assertTrue(cm.exception.connection_error)
        self.assertEqual((cm.exception.status, cm.exception.should_retry), (502, True))

    def test_auth_error_becomes_401(self):
        ctx = _ctx(self.srv.url, auth_static.StaticKeyAuth("missing", config.SecretStore(), provider_id="p"))
        with self.assertRaises(errors.GatewayError) as cm:
            dialects_base.send_with_auth_retry(ctx, "GET", self.srv.url + "/x", {}, None)
        self.assertEqual(cm.exception.status, 401)
        self.assertEqual(len(self.srv.requests_for("/x")), 0)

    def test_runtime_lru_registry(self):
        rt = dialects_base.ProviderRuntime("p")
        self.assertIs(rt.setdefault("ollama_caps", dict), rt.setdefault("ollama_caps", dict))
        rt.set("grok_fallback", True)
        self.assertTrue(rt.get("grok_fallback"))
        lru = dialects_base.LRU(2)
        lru.put("a", 1)
        lru.put("b", 2)
        lru.get("a")
        lru.put("c", 3)
        self.assertEqual((lru.get("a"), lru.get("b"), lru.get("c"), len(lru)), (1, None, 3, 2))
        self.assertEqual(dialects_base.join_url("http://h/v1/", "/chat/completions"), "http://h/v1/chat/completions")
        with self.assertRaises(ValueError):
            dialects.get_dialect("telepathy")
        self.assertEqual(set(dialects.REGISTRY), set(config.DIALECTS))
        ctx = _ctx("http://x", _CountingAuth())
        c2 = ctx.with_provider(config.ProviderSpec(id="p", dialect="openai_chat", base_url="http://y"))
        self.assertEqual((c2.provider.base_url, c2.runtime), ("http://y", ctx.runtime))
        self.assertFalse(ctx.background)
        ctx.trace("x", a=1)


# =========================================================================================
# presets / catalog
# =========================================================================================

class PresetCatalogTests(unittest.TestCase):
    REQUIRED = ("xai", "grok", "openai", "codex", "gemini", "gemini-vertex", "deepseek", "kimi", "openrouter",
                "ollama", "opencode-zen", "opencode-go", "opencode", "nvidia")

    def test_all_presets_build_and_validate(self):
        for name in self.REQUIRED:
            self.assertIn(name, presets.PRESETS)
            spec = presets.provider_from_preset(name)
            self.assertEqual(spec.problems(), [], name)
            for k in presets.LAUNCHER_KEYS:
                self.assertNotIn(k, spec.to_dict())
        t = presets.route_table_from_presets(list(self.REQUIRED))
        self.assertEqual(t.problems(), [])
        self.assertEqual(t.roles["default"], "xai,grok-4.7")
        self.assertEqual(config.RouteTable.from_dict(json.loads(json.dumps(t.to_dict()))), t)

    def test_specific_presets(self):
        g = presets.provider_from_preset("grok")
        self.assertEqual((g.dialect, g.target, g.fallback.dialect, g.fallback.base_url),
                         ("responses", "grok_cli_proxy", "openai_chat", "https://api.x.ai/v1"))
        self.assertEqual(g.fallback.auth, {"kind": "grok_cli"})
        self.assertEqual(presets.provider_from_preset("codex").target, "chatgpt_codex")
        self.assertEqual(presets.provider_from_preset("openai").target, "openai_api")
        self.assertEqual(presets.provider_from_preset("gemini-vertex").auth["kind"], "gcloud_adc")
        self.assertEqual(presets.provider_from_preset("ollama").fallback.profile, "ollama")
        self.assertEqual(presets.provider_from_preset("opencode").dialect, "cli")
        zen = presets.provider_from_preset("opencode-zen")
        dialects_seen = {zen.effective_dialect(m) for m in zen.models}
        self.assertTrue({"openai_chat", "anthropic_passthrough", "responses"} <= dialects_seen)
        xai = presets.provider_from_preset("xai")
        ma = xai.find_model("grok-4.20-multi-agent-0309")
        self.assertEqual((xai.effective_dialect(ma), xai.effective_target(ma)), ("responses", "xai_api"))
        self.assertEqual(presets.secret_env_vars("gemini"), ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
        renamed = presets.provider_from_preset("xai", {"id": "xai2", "auth": {"style": "x-api-key"},
                                                       "models": [{"id": "only"}]})
        self.assertEqual(renamed.id, "xai2")
        self.assertEqual(renamed.auth, {"kind": "api_key", "secret": "xai2", "style": "x-api-key"})
        self.assertEqual([m.id for m in renamed.models], ["only"])
        nofb = presets.provider_from_preset("grok", {"fallback": None})
        self.assertIsNone(nofb.fallback)
        self.assertEqual(presets.default_roles("xai"),
                         {"default": "xai,grok-4.7", "background": "xai,grok-4.20-0309-non-reasoning"})
        with self.assertRaises(KeyError):
            presets.provider_from_preset("nope")

    def test_catalog(self):
        self.assertEqual(set(catalog.families()), set(catalog.CATALOGS))
        a = catalog.get_catalog("xai")
        a[0].context = 1
        self.assertNotEqual(catalog.get_catalog("xai")[0].context, 1)
        self.assertEqual(catalog.get_catalog("unknown"), [])
        self.assertEqual(catalog.find_model("gemini", "gemini-3.8-flash").context, 1048576)
        for fam in catalog.families():
            ids = [m.id for m in catalog.get_catalog(fam)]
            for retired in ("deepseek-chat", "deepseek-reasoner", "o1-mini", "codex-mini", "gemini-2.0-flash"):
                self.assertNotIn(retired, ids)
        self.assertTrue(catalog.is_responses_lite("gpt-6.1"))
        self.assertTrue(catalog.is_responses_lite("gpt-6"))
        self.assertFalse(catalog.is_responses_lite("gpt-5.5"))
        self.assertTrue(catalog.is_openai_reasoning("o3"))
        self.assertTrue(catalog.is_openai_reasoning("gpt-5.4-mini"))
        self.assertFalse(catalog.is_openai_reasoning("gpt-4.1"))
        self.assertFalse(catalog.is_xai_reasoning("grok-4.20-0309-non-reasoning"))
        self.assertTrue(catalog.is_xai_reasoning("grok-9"))


# =========================================================================================
# mock framework + fixtures
# =========================================================================================

class MockFrameworkTests(unittest.TestCase):
    def test_echo_kind_and_recording(self):
        with mocks.MockServer("echo") as srv:
            c = transport.HttpClient(timeout=5, environ={})
            r = c.request("POST", srv.url + "/v1/x?beta=true&a=1", {"Content-Type": "application/json",
                                                                    "X-Custom": "v"}, b'{"k": [1]}', stream=False)
            rec = r.json()
            self.assertEqual(rec["path"], "/v1/x")
            self.assertEqual(rec["query"], {"beta": ["true"], "a": ["1"]})
            self.assertEqual(rec["body_json"], {"k": [1]})
            c.request("PUT", srv.url + "/raw", {}, b"not json", stream=False)
            self.assertEqual(srv.requests_for("/raw")[0]["body_raw"], "not json")
            self.assertEqual(len(srv.requests), 2)
            self.assertEqual(srv.last_request()["method"], "PUT")
            srv.clear()
            self.assertEqual(srv.requests, [])
        self.assertIsNone(srv.url)

    def test_registry_errors_and_sse_writer(self):
        def factory(server):
            def handle(req, resp):
                if req.path == "/boom":
                    raise ValueError("kaput")
                if req.path == "/nothing":
                    return
                w = resp.start_sse()
                w.event("ping", {"type": "ping"})
                w.data("[DONE]")
                w.comment("bye")
                w.close()
                w.close()
            return handle

        mocks.register_kind("unit-test-kind", factory)
        with mocks.MockServer("unit-test-kind") as srv:
            c = transport.HttpClient(timeout=5, environ={})
            evs = list(transport.iter_sse(c.request("GET", srv.url + "/sse")))
            self.assertEqual([(e.event, e.data) for e in evs], [("ping", '{"type":"ping"}'), ("message", "[DONE]")])
            r = c.request("GET", srv.url + "/boom", stream=False)
            self.assertEqual(r.status, 500)
            self.assertEqual(len(srv.errors), 1)
            self.assertIn("kaput", srv.errors[0])
            self.assertEqual(c.request("GET", srv.url + "/nothing", stream=False).status, 404)
        with self.assertRaises(KeyError):
            mocks.MockServer("no-such-kind").start()
        with self.assertRaises(ValueError):
            mocks.MockServer()

    def test_brain(self):
        b = mocks.Brain()
        r1 = b.decide(["Read", "Bash"], [])
        self.assertEqual((r1.kind, r1.tool_name), ("tool_call", "Bash"))
        self.assertEqual(r1.arguments["command"], "printf hello > out.txt && env")
        r2 = b.decide(["Bash"], ["PATH=/bin\nHOME=/x"])
        self.assertEqual(r2.text, "DONE " + mocks.sha8("PATH=/bin\nHOME=/x"))
        r3 = b.decide(["Bash"], ["XAI_API_KEY=" + mocks.SENTINEL])
        self.assertTrue(r3.text.startswith("LEAK"))
        self.assertEqual(len(b.leaks), 1)
        self.assertEqual(b.decide(["Bash"], [], background=True).text, "Background reply.")
        self.assertEqual(b.decide([], []).kind, "text")
        self.assertEqual(b.find_tool(["bash"]), "bash")
        self.assertEqual(b.find_tool(["Ba_0123abcd"]), "Ba_0123abcd")
        self.assertIsNone(b.find_tool(["Read"]))
        self.assertTrue(b.decide(["Read"], []).text.startswith("NO-TOOL"))
        self.assertEqual(len(mocks.sha8("x")), 8)
        self.assertEqual(mocks.sse_event_bytes("a", {"b": 1}), b'event: a\ndata: {"b":1}\n\n')
        self.assertEqual(mocks.sse_data_bytes("[DONE]"), b"data: [DONE]\n\n")

    def test_anthropic_brain_inputs_on_fixtures(self):
        tools, results, bg = mocks.anthropic_brain_inputs(testing.load_fixture("turn1_request.json")["body"])
        self.assertIn("Bash", tools)
        self.assertEqual((results, bg), ([], False))
        tools, results, bg = mocks.anthropic_brain_inputs(testing.load_fixture("turn2_tool_result_request.json")["body"])
        self.assertEqual(results, ["(Bash completed with no output)"])
        _, _, bg = mocks.anthropic_brain_inputs(testing.load_fixture("background_request.json")["body"])
        self.assertTrue(bg)


class FixtureTests(unittest.TestCase):
    def test_fixtures_are_scrubbed_and_complete(self):
        names = testing.list_fixtures()
        for want in ("turn1_request.json", "turn2_tool_result_request.json", "background_request.json",
                     "count_tokens_request.json", "models_request.json", "tool_result_image_request.json",
                     "turn2_thinking_roundtrip_request.json", "turn1_mcp_long_tool_name_request.json"):
            self.assertIn(want, names)
        for n in names:
            with open(testing.fixture_path(n), "rb") as f:
                raw = f.read()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), n)
            text = raw.decode("utf-8")
            for bad in ("/tmp/", "Authorization", "Bearer ", "x-api-key"):
                self.assertNotIn(bad, text, "%s contains %r" % (n, bad))
            fx = json.loads(text)
            self.assertEqual(set(fx), {"method", "path", "headers", "body"})

    def test_spike_facts(self):
        t1 = testing.load_fixture("turn1_request.json")
        self.assertEqual(t1["path"], "/v1/messages?beta=true")
        b = t1["body"]
        self.assertEqual(b["thinking"]["type"], "adaptive")
        self.assertEqual(b["max_tokens"], 32000)
        self.assertIn("context_management", b)
        self.assertIn("effort", b["output_config"])
        self.assertTrue(b["system"][0]["text"].startswith("x-anthropic-billing-header:"))
        self.assertIn("system", [m["role"] for m in b["messages"]])
        self.assertIn("X-Claude-Code-Session-Id", t1["headers"])
        img = testing.load_fixture("tool_result_image_request.json")["body"]["messages"]
        found = [c for m in img if isinstance(m["content"], list) for blk in m["content"]
                 if blk.get("type") == "tool_result" for c in (blk.get("content") or []) if isinstance(c, dict)
                 and c.get("type") == "image"]
        self.assertEqual(found[0]["source"]["type"], "base64")
        bg = testing.load_fixture("background_request.json")["body"]
        self.assertEqual(bg["model"], "claude-via-background")
        self.assertEqual(bg["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(testing.load_fixture("models_request.json")["path"], "/v1/models?limit=1000")
        m1 = testing.load_fixture("turn1_1m_suffix_request.json")
        self.assertEqual(m1["body"]["model"], "claude-via-xai,grok-4.7")
        self.assertIn("context-1m-2025-08-07", m1["headers"]["anthropic-beta"])
        th = testing.load_fixture("turn2_thinking_roundtrip_request.json")["body"]["messages"]
        sigs = [blk.get("signature") for m in th if m["role"] == "assistant" for blk in m["content"]
                if blk.get("type") == "thinking"]
        self.assertEqual(sigs, ["fgw1.chat.SPIKESIG2"])


if __name__ == "__main__":
    unittest.main()
