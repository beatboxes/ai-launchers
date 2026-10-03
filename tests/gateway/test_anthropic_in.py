"""anthropic_in.parse_messages_request: golden fixtures recorded from Claude Code 2.1.288 + unit cases."""

import copy
import json
import unittest

from ._pkg import mod

ai = mod("anthropic_in")
model = mod("model")
errors = mod("errors")
testing = mod("testing")

SESSION = "00000000-0000-4000-8000-000000000001"
REMINDER = model.SYSTEM_REMINDER_OPEN


def fixture(name):
    return testing.load_fixture(name)


def parse(body, headers=None):
    return ai.parse_messages_request(body, headers or {})


def base_body(**kw):
    body = {"model": "claude-via-xai,grok-4.7", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return body


class GoldenFixtureTests(unittest.TestCase):
    def test_every_fixture_parses(self):
        names = testing.list_fixtures()
        self.assertGreaterEqual(len(names), 11)
        parsed = 0
        for name in names:
            fx = fixture(name)
            body = fx.get("body")
            if body is None:
                self.assertEqual(fx["method"], "GET", name)
                continue
            before = copy.deepcopy(body)
            req = parse(body, fx["headers"])
            parsed += 1
            self.assertEqual(body, before, "%s: raw body mutated" % name)
            self.assertIs(req.raw, body)
            self.assertEqual(req.model, body["model"])
            self.assertEqual(req.session_id, SESSION, name)
            self.assertEqual(req.stream, bool(body.get("stream")), name)
            self.assertNotIn("authorization", req.headers)
            self.assertIn("x-claude-code-session-id", req.headers)
            for s in req.system:
                self.assertTrue(s.strip())
                self.assertFalse(s.startswith("x-anthropic-billing-header:"), name)
            for m in req.messages:
                self.assertIn(m.role, ("user", "assistant"), name)
                self.assertTrue(m.blocks, name)
                for b in m.blocks:
                    self.assertIn(b.type, model.BLOCK_TYPES)
            self.assertEqual(req.dropped, [], name)
            names_ = req.tool_names()
            self.assertEqual(len(names_), len(set(names_)))
        self.assertGreaterEqual(parsed, 10)

    def test_turn1_folds_trailing_system_message(self):
        fx = fixture("turn1_request.json")
        raw = fx["body"]
        self.assertEqual(raw["messages"][-1]["role"], "system")
        req = parse(raw, fx["headers"])
        self.assertEqual(len(req.messages), 1)
        user = req.messages[0]
        self.assertEqual(user.role, "user")
        self.assertTrue(user.blocks[-1].text.startswith(REMINDER))
        self.assertTrue(user.blocks[-1].text.endswith(model.SYSTEM_REMINDER_CLOSE))
        self.assertIn("You are powered by the model claude-via-xai,grok-4.7", user.blocks[-1].text)
        self.assertEqual(user.blocks[1].text, "Run: printf hello > out.txt")
        # system: billing header dropped, the two cached text blocks kept (cache_control gone)
        self.assertEqual(len(raw["system"]), 3)
        self.assertEqual(len(req.system), 2)
        self.assertEqual(req.system[0], "You are a Claude agent, built on Anthropic's Claude Agent SDK.")
        # effort/thinking/max_tokens
        self.assertEqual(req.effort, "high")
        self.assertTrue(req.thinking_requested)
        self.assertIsNone(req.thinking_budget)
        self.assertEqual(req.max_tokens, 32000)
        self.assertTrue(req.stream)
        self.assertIsNone(req.output_format)
        self.assertIsNone(req.temperature)
        self.assertEqual(len(req.tools), 20)
        bash = [t for t in req.tools if t.name == "Bash"][0]
        self.assertEqual(bash.input_schema["type"], "object")
        self.assertIn("command", bash.input_schema["properties"])
        self.assertTrue(bash.description)

    def test_streaming_decided_by_body_only(self):
        fx = fixture("turn1_request.json")
        self.assertEqual(fx["headers"]["Accept"], "application/json")
        self.assertTrue(parse(fx["body"], fx["headers"]).stream)
        fx = fixture("nonstream_retry_after_404_request.json")
        req = parse(fx["body"], fx["headers"])
        self.assertFalse(req.stream)
        self.assertEqual(req.model, "claude-via-nope,x")

    def test_tool_result_image_and_last_system_message(self):
        fx = fixture("tool_result_image_request.json")
        req = parse(fx["body"], fx["headers"])
        roles = [m.role for m in req.messages]
        self.assertEqual(roles, ["user", "assistant", "user"])
        tool_use = req.messages[1].blocks[0]
        self.assertEqual(tool_use.type, "tool_use")
        self.assertEqual(tool_use.name, "Read")
        self.assertIsInstance(tool_use.input, dict)
        last = req.messages[2]
        self.assertEqual(last.blocks[0].type, "tool_result")
        self.assertEqual(last.blocks[0].tool_use_id, tool_use.id)
        img = last.blocks[0].content[0]
        self.assertEqual((img.type, img.media_type), ("image", "image/png"))
        self.assertTrue(img.data.startswith("iVBOR"))
        self.assertEqual(last.blocks[0].result_media(), [img])
        # the trailing role:"system" <total_tokens> message lands AFTER the tool_result
        self.assertEqual(last.blocks[1].type, "text")
        self.assertIn("<total_tokens>", last.blocks[1].text)
        self.assertTrue(last.blocks[1].text.startswith(REMINDER))
        self.assertEqual(req.tool_use_names_by_id(), {tool_use.id: "Read"})

    def test_thinking_roundtrip(self):
        fx = fixture("turn2_thinking_roundtrip_request.json")
        req = parse(fx["body"], fx["headers"])
        asst = req.messages[1]
        self.assertEqual([b.type for b in asst.blocks], ["thinking", "tool_use"])
        self.assertEqual(asst.blocks[0].thinking, "Let me think.")
        self.assertEqual(asst.blocks[0].signature, "fgw1.chat.SPIKESIG2")
        result = req.messages[2].blocks[0]
        self.assertEqual(result.type, "tool_result")
        self.assertEqual(result.result_text(), "(Bash completed with no output)")
        self.assertFalse(result.is_error)

    def test_tool_result_string_content(self):
        fx = fixture("turn2_tool_result_request.json")
        req = parse(fx["body"], fx["headers"])
        res = req.messages[2].blocks[0]
        self.assertEqual([b.type for b in res.content], ["text"])
        self.assertEqual(req.messages[1].blocks[0].input["command"], "printf hello > out.txt")

    def test_background_structured_output(self):
        fx = fixture("background_request.json")
        req = parse(fx["body"], fx["headers"])
        self.assertEqual(req.model, "claude-via-background")
        self.assertEqual(req.tools, [])
        self.assertEqual(req.output_format["type"], "json_schema")
        self.assertEqual(req.output_format["schema"]["required"], ["title"])
        self.assertEqual(req.effort, "high")  # from output_config.effort (no thinking key)
        self.assertFalse(req.thinking_requested)
        self.assertEqual(len(req.system), 2)

    def test_count_tokens_bodies(self):
        fx = fixture("count_tokens_request.json")
        req = parse(fx["body"], fx["headers"])
        self.assertEqual(req.max_tokens, 32000)  # absent -> default
        self.assertFalse(req.stream)
        self.assertEqual(req.messages[0].text(), "foo")
        fx = fixture("count_tokens_system_request.json")
        req = parse(fx["body"], fx["headers"])
        self.assertEqual(len(req.tools), 10)

    def test_long_mcp_tool_name_verbatim(self):
        fx = fixture("turn1_mcp_long_tool_name_request.json")
        req = parse(fx["body"], fx["headers"])
        long_names = [n for n in req.tool_names() if len(n) > 64]
        self.assertEqual(len(long_names), 1)
        self.assertTrue(long_names[0].startswith("mcp__stub__"))

    def test_1m_fixture_model_already_stripped(self):
        fx = fixture("turn1_1m_suffix_request.json")
        req = parse(fx["body"], fx["headers"])
        self.assertEqual(req.model, "claude-via-xai,grok-4.7")
        self.assertIn("context-1m-2025-08-07", req.headers["anthropic-beta"])


class SystemAndToolChangeTests(unittest.TestCase):
    def test_system_string_and_billing_string(self):
        self.assertEqual(parse(base_body(system="Be brief.")).system, ["Be brief."])
        self.assertEqual(parse(base_body(system="x-anthropic-billing-header: cc_version=1;")).system, [])
        self.assertEqual(parse(base_body(system="   ")).system, [])
        req = parse(base_body(system=[{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                                      {"type": "text", "text": ""}, {"type": "image", "source": {}}]))
        self.assertEqual(req.system, ["a"])
        self.assertEqual(req.dropped, ["system:image"])

    def test_system_message_folding_positions(self):
        body = base_body(messages=[
            {"role": "system", "content": "first"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "system", "content": [{"type": "text", "text": "mid"}]},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "q2"},
            {"role": "system", "content": "last"},
        ])
        req = parse(body)
        self.assertEqual([m.role for m in req.messages], ["user", "assistant", "user", "assistant", "user"])
        self.assertTrue(req.messages[0].blocks[0].text.startswith(REMINDER + "first"))
        self.assertEqual(req.messages[0].blocks[1].text, "q")
        self.assertIn("mid", req.messages[2].blocks[0].text)  # standalone user message before "b"
        self.assertEqual(req.messages[4].blocks[0].text, "q2")
        self.assertIn("last", req.messages[4].blocks[1].text)

    def test_tool_addition_and_removal(self):
        tools = [{"name": "Bash", "input_schema": {"type": "object"}},
                 {"name": "Old", "input_schema": {"type": "object"}}]
        body = base_body(tools=tools, messages=[
            {"role": "user", "content": "q"},
            {"role": "system", "content": [
                {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": {
                    "name": "New", "description": "n", "input_schema": {"type": "object", "properties": {}}}}},
                {"type": "tool_removal", "tool": {"name": "Old"}},
            ], "clear_at": 3},
        ])
        req = parse(body)
        self.assertEqual(req.tool_names(), ["Bash", "New"])
        self.assertEqual(len(req.messages), 1)  # a system message without text adds no reminder
        self.assertEqual(req.messages[0].text(), "q")

    def test_server_tools_dropped_and_custom_kept(self):
        body = base_body(tools=[
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
            {"type": "custom", "name": "A", "description": "d", "input_schema": {"type": "object"},
             "defer_loading": True, "cache_control": {"type": "ephemeral"}},
            {"name": "B"},
            {"name": "A", "description": "duplicate"},
        ])
        req = parse(body)
        self.assertEqual(req.tool_names(), ["A", "B"])
        self.assertEqual(req.tools[0].description, "d")
        self.assertEqual(req.tools[1].input_schema, {"type": "object", "properties": {}})
        self.assertEqual(req.dropped, ["tool:web_search_20250305"])


class BlockTests(unittest.TestCase):
    def blocks(self, content, role="user"):
        req = parse(base_body(messages=[{"role": "user", "content": "q"}, {"role": role, "content": content}]))
        return req.messages[-1].blocks if len(req.messages) > 1 else [], req

    def test_all_block_kinds(self):
        content = [
            {"type": "text", "text": "t", "citations": None},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}},
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBE"},
             "title": "doc"},
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "plain"}},
            {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": "c1"},
                                                                           {"type": "text", "text": "c2"}]}},
            {"type": "document", "source": {"type": "url", "url": "https://x/a.pdf"}},
            {"type": "search_result", "source": "https://s", "title": "T",
             "content": [{"type": "text", "text": "body"}]},
            {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True, "content": [
                {"type": "text", "text": "out"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"}},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "CC"}},
                {"type": "search_result", "source": "s", "title": "t", "content": "x"},
                {"type": "tool_reference", "tool_name": "Z"},
            ]},
            {"type": "server_tool_use", "id": "srv", "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": "srv", "content": []},
        ]
        blocks, req = self.blocks(content)
        types = [b.type for b in blocks]
        self.assertEqual(types, ["text", "image", "image", "document", "document", "document", "document", "text",
                                 "tool_result"])
        self.assertEqual(blocks[1].media_type, "image/jpeg")
        self.assertEqual(blocks[2].url, "https://x/y.png")
        self.assertEqual((blocks[3].media_type, blocks[3].data, blocks[3].title), ("application/pdf", "JVBE", "doc"))
        self.assertEqual((blocks[4].text, blocks[4].media_type), ("plain", "text/plain"))
        self.assertEqual(blocks[5].text, "c1\nc2")
        self.assertEqual(blocks[6].url, "https://x/a.pdf")
        self.assertEqual(blocks[7].text, "Search result: T\nSource: https://s\nbody")
        tr = blocks[8]
        self.assertTrue(tr.is_error)
        self.assertEqual([b.type for b in tr.content], ["text", "image", "document", "text"])
        self.assertEqual(req.dropped, ["block:tool_reference", "block:server_tool_use", "block:web_search_tool_result"])

    def test_assistant_blocks(self):
        content = [
            {"type": "thinking", "thinking": "hmm", "signature": ""},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "tool_use", "id": "toolu_x", "name": "Bash", "input": {"command": "ls"}},
            {"type": "tool_use", "id": "toolu_y", "name": "Read", "input": "{\"file_path\": \"a\"}"},
            {"type": "tool_use", "id": "toolu_z", "name": "Noop"},
        ]
        blocks, _ = self.blocks(content, role="assistant")
        self.assertEqual(blocks[0].signature, None)
        self.assertEqual(blocks[0].thinking, "hmm")
        self.assertEqual(blocks[1].data, "opaque")
        self.assertEqual(blocks[2].input, {"command": "ls"})
        self.assertEqual(blocks[3].input, {"file_path": "a"})
        self.assertEqual(blocks[4].input, {})

    def test_empty_text_dropped_and_empty_messages_removed(self):
        req = parse(base_body(messages=[{"role": "user", "content": [{"type": "text", "text": ""}]},
                                        {"role": "user", "content": "real"}]))
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.messages[0].text(), "real")


class ParamTests(unittest.TestCase):
    def test_defaults(self):
        req = parse({"model": "m", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(req.max_tokens, 32000)
        self.assertFalse(req.stream)
        self.assertIsNone(req.effort)
        self.assertIsNone(req.tool_choice)
        self.assertFalse(req.disable_parallel_tool_use)
        self.assertIsNone(req.session_id)
        self.assertEqual(req.stop_sequences, [])

    def test_sampling_and_choice(self):
        req = parse(base_body(temperature=0.5, top_p=0.9, top_k=40, stop_sequences=["END", "", 3], stream=True,
                              tool_choice={"type": "tool", "name": "Bash", "disable_parallel_tool_use": True}))
        self.assertEqual((req.temperature, req.top_p, req.top_k), (0.5, 0.9, 40))
        self.assertEqual(req.stop_sequences, ["END"])
        self.assertEqual(req.tool_choice, {"type": "tool", "name": "Bash"})
        self.assertTrue(req.disable_parallel_tool_use)
        self.assertTrue(req.stream)
        for t in ("auto", "any", "none"):
            self.assertEqual(parse(base_body(tool_choice={"type": t})).tool_choice, {"type": t})

    def test_effort_sources(self):
        cases = [
            ({"thinking": {"type": "adaptive", "display": "updates"}}, "medium", True, None),
            ({"thinking": {"type": "enabled", "budget_tokens": 2000}}, "low", True, 2000),
            ({"thinking": {"type": "enabled", "budget_tokens": 10000}}, "medium", True, 10000),
            ({"thinking": {"type": "enabled", "budget_tokens": 30000}}, "high", True, 30000),
            ({"thinking": {"type": "disabled"}}, "none", False, None),
            ({"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}}, "max", True, None),
            ({"output_config": {"effort": "bogus"}}, None, False, None),
            ({"output_config": {"effort": "xhigh"}}, "xhigh", False, None),
        ]
        for extra, effort, requested, budget in cases:
            req = parse(base_body(**extra))
            self.assertEqual((req.effort, req.thinking_requested, req.thinking_budget), (effort, requested, budget),
                             extra)

    def test_output_format_spellings(self):
        fmt = {"type": "json_schema", "schema": {"type": "object"}}
        self.assertEqual(parse(base_body(output_config={"format": fmt})).output_format, fmt)
        self.assertEqual(parse(base_body(output_format=fmt)).output_format, fmt)

    def test_session_id_sources(self):
        uid = json.dumps({"device_id": "d", "account_uuid": "", "session_id": "sess-meta"})
        body = base_body(metadata={"user_id": uid})
        self.assertEqual(parse(body).session_id, "sess-meta")
        self.assertEqual(parse(body, {"X-Claude-Code-Session-Id": "sess-hdr"}).session_id, "sess-hdr")
        self.assertIsNone(parse(base_body(metadata={"user_id": "plain-user"})).session_id)
        self.assertIsNone(parse(base_body(metadata={"user_id": "{not json"})).session_id)

    def test_headers_lowercased_without_credentials(self):
        req = parse(base_body(), [("Authorization", "Bearer s3cret"), ("X-Api-Key", "k"), ("Anthropic-Beta", "a"),
                                  ("anthropic-beta", "b"), ("Cookie", "c")])
        self.assertEqual(req.headers, {"anthropic-beta": "a, b"})


class MalformedTests(unittest.TestCase):
    def assert400(self, body, needle=None):
        with self.assertRaises(errors.GatewayError) as cm:
            parse(body)
        self.assertEqual(cm.exception.status, 400)
        self.assertEqual(cm.exception.err_type, "invalid_request_error")
        if needle:
            self.assertIn(needle, cm.exception.message)

    def test_malformed_bodies(self):
        self.assert400([], "JSON object")
        self.assert400({"messages": [{"role": "user", "content": "x"}]}, "model")
        self.assert400(base_body(messages=[]), "messages")
        self.assert400(base_body(messages="hi"), "messages")
        self.assert400(base_body(messages=["hi"]), "messages.0")
        self.assert400(base_body(messages=[{"role": "tool", "content": "x"}]), "role")
        self.assert400(base_body(messages=[{"role": "user", "content": 5}]), "content")
        self.assert400(base_body(messages=[{"role": "user", "content": [{"text": "x"}]}]), "type")
        self.assert400(base_body(messages=[{"role": "user", "content": [{"type": "text", "text": 3}]}]), "text")
        self.assert400(base_body(messages=[{"role": "assistant", "content": [
            {"type": "tool_use", "id": "", "name": "x", "input": {}}]}]), "tool_use")
        self.assert400(base_body(messages=[{"role": "assistant", "content": [
            {"type": "tool_use", "id": "a", "name": "x", "input": [1]}]}]), "input")
        self.assert400(base_body(messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png"}}]}]), "data")
        self.assert400(base_body(max_tokens=0), "max_tokens")
        self.assert400(base_body(max_tokens="10"), "max_tokens")
        self.assert400(base_body(stream="yes"), "stream")
        self.assert400(base_body(temperature="hot"), "temperature")
        self.assert400(base_body(tools={"name": "x"}), "tools")
        self.assert400(base_body(tools=[{"description": "nameless"}]), "tools.0.name")
        self.assert400(base_body(tool_choice={"type": "tool"}), "tool_choice")
        self.assert400(base_body(tool_choice={"type": "sometimes"}), "tool_choice")
        self.assert400(base_body(system=5), "system")
        self.assert400(base_body(stop_sequences="END"), "stop_sequences")
        self.assert400(base_body(messages=[{"role": "user", "content": ""}]), "no message has content")

    def test_system_only_conversation_becomes_user_message(self):
        req = parse(base_body(messages=[{"role": "system", "content": "ctx"}]))
        self.assertEqual(req.messages[0].role, "user")
        self.assertIn("ctx", req.messages[0].text())


if __name__ == "__main__":
    unittest.main()
