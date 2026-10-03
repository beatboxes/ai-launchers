"""openai_chat dialect: request goldens per profile, stream parsing, and execute() end-to-end against the
quirk-enforcing mock kinds (xai/openai/moonshot/deepseek/nvidia/ollama/openrouter/generic)."""

import copy
import json
import logging
import unittest

from ._pkg import mod

events = mod("events")
model = mod("model")
config = mod("config")
errors = mod("errors")
presets = mod("presets")
catalog = mod("catalog")
signatures = mod("signatures")
transport = mod("transport")
auth_pkg = mod("auth")
auth_static = mod("auth.static")
dialects = mod("dialects")
dbase = mod("dialects.base")
oc = mod("dialects.openai_chat")
cp = mod("chat_profiles")
tn = mod("toolnames")
mocks = mod("testing.mock_upstreams")
mchat = mod("testing.mock_openai_chat")

B = model.Block
M = model.Message
LONG_MCP = "mcp__stub__get_the_magic_number_for_an_extremely_long_tool_name_that_exceeds_limits_x"
LONG_UP = tn.ToolNameMap([LONG_MCP]).upstream(LONG_MCP)
KIMI_ID = "functions.Bash:0"
KIMI_ENC = tn.encode_tool_id(KIMI_ID)
THOUGHT = "I should run it."
SCHEMA_BASH = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
               "properties": {"command": {"type": "string"}, "description": {"type": "string"}},
               "required": ["command"], "additionalProperties": False}
SCHEMA_MCP = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
TOOLS = [model.ToolDef("Bash", "Run a shell command", SCHEMA_BASH), model.ToolDef(LONG_MCP, "", SCHEMA_MCP)]
LOG = logging.getLogger("test_openai_chat")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


def rich_request(**kw):
    sig = signatures.synthetic_signature(THOUGHT)
    msgs = [
        M("user", [B.of_text("<system-reminder>\nctx\n</system-reminder>"), B.of_text("Run it")]),
        M("assistant", [B.of_thinking(THOUGHT, sig), B.of_thinking("foreign", "fgw1.openai_api.e30"),
                        B.of_redacted_thinking("zzz"), B.of_text("Running."),
                        B.of_tool_use(KIMI_ENC, "Bash", {"command": "ls"}),
                        B.of_tool_use("call_2", LONG_MCP, {"n": 7})]),
        M("user", [B.of_tool_result(KIMI_ENC, [B.of_text("a.txt"), B.of_image_base64("image/png", "iVBO")]),
                   B.of_tool_result("call_2", "boom", is_error=True), B.of_text("continue")]),
    ]
    args = dict(model="claude-via-x,m", system=["You are Claude Code.", "Env: linux"], messages=msgs,
                tools=list(TOOLS), tool_choice={"type": "any"}, disable_parallel_tool_use=True, max_tokens=32000,
                temperature=0.2, top_p=0.9, stop_sequences=["STOP"], effort="high", session_id="sess-1")
    args.update(kw)
    return model.NormalizedRequest(**args)


def base_golden(model_id, max_tokens=32000):
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are Claude Code.\n\nEnv: linux"},
            {"role": "user", "content": "<system-reminder>\nctx\n</system-reminder>\n\nRun it"},
            {"role": "assistant", "content": "Running.", "tool_calls": [
                {"id": KIMI_ID, "type": "function", "function": {"name": "Bash", "arguments": "{\"command\":\"ls\"}"}},
                {"id": "call_2", "type": "function", "function": {"name": LONG_UP, "arguments": "{\"n\":7}"}}]},
            {"role": "tool", "tool_call_id": KIMI_ID, "content": "a.txt\n" + oc.IMAGE_BELOW},
            {"role": "tool", "tool_call_id": "call_2", "content": "ERROR: boom"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO"}},
                                         {"type": "text", "text": "continue"}]},
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "Bash", "description": "Run a shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                                                                "description": {"type": "string"}},
                               "required": ["command"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": LONG_UP, "parameters": SCHEMA_MCP}},
        ],
        "tool_choice": "required",
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["STOP"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def provider(name="generic", **kw):
    d = {"base_url": "http://up.invalid/v1", "profile": None if name == "generic" else name}
    d.update(kw)
    return config.ProviderSpec.from_dict("p", d)


def build(req, prov, spec, caps=None):
    return oc.build_request(req, spec, prov, None, caps, req.session_id)


# =========================================================================================
# request goldens
# =========================================================================================

class RequestGoldenTests(unittest.TestCase):
    maxDiff = None

    def test_generic(self):
        b = build(rich_request(), provider(), config.ModelSpec("m"))
        self.assertEqual(b.body, base_golden("m", 16384))
        self.assertEqual(b.headers, {"Content-Type": "application/json", "Accept": "text/event-stream"})
        self.assertEqual(b.names.original(LONG_UP), LONG_MCP)
        self.assertIn("tool_names", [k for k, _ in b.notes])
        self.assertFalse(b.chat_only)

    def test_openai(self):
        spec = config.ModelSpec("gpt-5.5", max_output=128000, reasoning=True)
        b = build(rich_request(), provider("openai"), spec)
        g = base_golden("gpt-5.5")
        del g["max_tokens"], g["temperature"], g["top_p"]
        g["max_completion_tokens"] = 32000
        g["parallel_tool_calls"] = False
        g["reasoning_effort"] = "high"
        g["prompt_cache_key"] = "sess-1"
        self.assertEqual(b.body, g)
        keys = [k for k, _ in b.notes]
        self.assertIn("drop_temperature", keys)
        self.assertIn("drop_top_p", keys)
        # non-reasoning OpenAI ids keep sampling params and get no effort
        b2 = build(rich_request(effort="max"), provider("openai"), config.ModelSpec("gpt-4.1"))
        self.assertEqual((b2.body["temperature"], b2.body["top_p"]), (0.2, 0.9))
        self.assertNotIn("reasoning_effort", b2.body)
        b3 = build(rich_request(effort="xhigh"), provider("openai"), spec)
        self.assertEqual(b3.body["reasoning_effort"], "high")

    def test_xai(self):
        prov = presets.provider_from_preset("xai", {"base_url": "http://up.invalid/v1"})
        spec = prov.model_spec("grok-4.7")
        b = build(rich_request(), prov, spec)
        g = base_golden("grok-4.7")
        del g["stop"]
        g["parallel_tool_calls"] = False
        self.assertEqual(b.body, g)
        self.assertEqual(b.headers["x-grok-conv-id"], "sess-1")
        self.assertIn("drop_stop", [k for k, _ in b.notes])
        nr = build(rich_request(), prov, prov.model_spec("grok-4.20-0309-non-reasoning"))
        self.assertEqual(nr.body["stop"], ["STOP"])
        eff = config.ModelSpec("grok-3-mini", reasoning=True, effort_param=True)
        self.assertEqual(build(rich_request(effort="medium"), prov, eff).body["reasoning_effort"], "high")

    def test_xai_root_combinator_schema_flattened(self):
        tool = model.ToolDef("Pick", "", {"anyOf": [{"type": "object", "properties": {"a": {"type": "string"}},
                                                     "required": ["a"]},
                                                    {"type": "object", "properties": {"b": {"type": "string"}},
                                                     "required": ["b"]}]})
        prov = presets.provider_from_preset("xai", {"base_url": "http://up.invalid/v1"})
        b = build(model.NormalizedRequest(model="m", messages=[M("user", [B.of_text("hi")])], tools=[tool]),
                  prov, prov.model_spec("grok-4.7"))
        params = b.body["tools"][0]["function"]["parameters"]
        self.assertEqual(params, {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}})
        g = build(model.NormalizedRequest(model="m", messages=[M("user", [B.of_text("hi")])], tools=[tool]),
                  provider(), config.ModelSpec("m"))
        self.assertIn("anyOf", g.body["tools"][0]["function"]["parameters"])

    def test_moonshot(self):
        b = build(rich_request(), provider("moonshot"), config.ModelSpec("kimi-k3"))
        g = base_golden("kimi-k3")
        del g["temperature"], g["top_p"], g["stream_options"]
        g["tool_choice"] = "auto"
        g["messages"][2]["reasoning_content"] = THOUGHT  # only the chat-signed thinking block
        self.assertEqual(b.body, g)
        # tool-call turn without kept thinking gets a placeholder (thinking-enabled Kimi rejects it otherwise)
        req = rich_request()
        req.messages[1].blocks = [blk for blk in req.messages[1].blocks if blk.type != "thinking"]
        self.assertEqual(build(req, provider("moonshot"), config.ModelSpec("kimi-k3")).body["messages"][2]
                         ["reasoning_content"], " ")

    def test_deepseek(self):
        prov = presets.provider_from_preset("deepseek", {"dialect": "openai_chat", "base_url": "http://u/v1"})
        spec = prov.model_spec("deepseek-v4-pro")
        b = build(rich_request(), prov, spec)
        g = base_golden("deepseek-v4-pro")
        del g["temperature"]
        g["messages"][2]["reasoning_content"] = THOUGHT
        g["messages"][3]["content"] = "a.txt\n" + oc.IMAGE_OMITTED   # no vision on DeepSeek's API
        g["messages"][5]["content"] = "continue"
        self.assertEqual(b.body, g)
        self.assertIn("vision", [k for k, _ in b.notes])

    def test_ollama_caps(self):
        prov = presets.provider_from_preset("ollama").fallback
        self.assertEqual(prov.profile, "ollama")
        spec = prov.model_spec("qwen3:8b")
        b = build(rich_request(), prov, spec, {"tools": True, "thinking": True, "vision": True})
        g = base_golden("qwen3:8b")
        g["reasoning_effort"] = "high"
        self.assertEqual(b.body, g)
        b2 = build(rich_request(), prov, spec, {"tools": True, "thinking": False, "vision": None})
        self.assertNotIn("reasoning_effort", b2.body)
        self.assertIn("drop_effort", [k for k, _ in b2.notes])
        # no tools capability -> chat-only: tools omitted, tool history flattened to text, no vision
        b3 = build(rich_request(), prov, spec, {"tools": False, "thinking": False, "vision": False})
        self.assertTrue(b3.chat_only)
        for k in ("tools", "tool_choice", "parallel_tool_calls"):
            self.assertNotIn(k, b3.body)
        roles = [m["role"] for m in b3.body["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(b3.body["messages"][2]["content"],
                         "Running.\n\n[called tool Bash with input {\"command\":\"ls\"}]\n\n"
                         "[called tool %s with input {\"n\":7}]" % LONG_MCP)
        self.assertEqual(b3.body["messages"][3]["content"],
                         "[tool result for Bash]\na.txt\n%s\n\n[tool result for %s]\nERROR: boom\n\ncontinue"
                         % (oc.IMAGE_OMITTED, LONG_MCP))

    def test_openrouter(self):
        prov = provider("openrouter", headers={"X-Extra": "1"})
        b = build(rich_request(effort="max"), prov, config.ModelSpec("x-ai/grok-4.7"))
        g = base_golden("x-ai/grok-4.7")
        g["parallel_tool_calls"] = False
        g["reasoning"] = {"effort": "high"}
        g["usage"] = {"include": True}
        self.assertEqual(b.body, g)
        self.assertEqual(b.headers["HTTP-Referer"], cp.OPENROUTER_REFERER)
        self.assertEqual(b.headers["X-Title"], cp.OPENROUTER_TITLE)
        self.assertEqual(b.headers["X-Extra"], "1")
        self.assertNotIn("reasoning", build(rich_request(effort="none"), prov, config.ModelSpec("x")).body)

    def test_nvidia_and_opencode(self):
        prov = presets.provider_from_preset("nvidia", {"base_url": "http://u/v1"})
        b = build(rich_request(), prov, prov.model_spec("moonshotai/kimi-k2.5"))
        g = base_golden("moonshotai/kimi-k2.5", 16384)
        del g["stream_options"]
        self.assertEqual(b.body, g)
        o = build(rich_request(), provider("opencode"), config.ModelSpec("qwen3-coder"))
        self.assertEqual(o.body, base_golden("qwen3-coder"))

    def test_max_tokens_clamp_and_probe(self):
        spec = config.ModelSpec("m", max_output=8000)
        self.assertEqual(build(rich_request(), provider(), spec).body["max_tokens"], 8000)
        self.assertEqual(build(rich_request(max_tokens=1), provider(), spec).body["max_tokens"], 1)
        self.assertEqual(build(rich_request(), provider("opencode"), config.ModelSpec("m")).body["max_tokens"], 32000)

    def test_tool_choice_mapping(self):
        prov, spec = provider("openai"), config.ModelSpec("gpt-4.1")
        cases = [({"type": "auto"}, "auto"), ({"type": "any"}, "required"), ({"type": "none"}, "none"),
                 ({"type": "tool", "name": LONG_MCP}, {"type": "function", "function": {"name": LONG_UP}}),
                 (None, None)]
        for choice, want in cases:
            body = build(rich_request(tool_choice=choice, disable_parallel_tool_use=False), prov, spec).body
            self.assertEqual(body.get("tool_choice"), want, choice)
            self.assertNotIn("parallel_tool_calls", body)
        no_tools = build(rich_request(tools=[], tool_choice={"type": "auto"}, messages=[M("user", [B.of_text("x")])]),
                         prov, spec).body
        for k in ("tools", "tool_choice", "parallel_tool_calls"):
            self.assertNotIn(k, no_tools)

    def test_structured_output(self):
        fmt = {"type": "json_schema", "schema": {"$schema": "x", "type": "object",
                                                 "properties": {"title": {"type": "string"}}, "required": ["title"]}}
        req = model.NormalizedRequest(model="m", system=["Title it."], messages=[M("user", [B.of_text("s")])],
                                      output_format=fmt, max_tokens=100)
        o = build(req, provider("openai"), config.ModelSpec("gpt-4.1")).body
        self.assertEqual(o["response_format"], {"type": "json_schema", "json_schema": {
            "name": "output", "strict": False,
            "schema": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}}})
        self.assertEqual(o["messages"][0], {"role": "system", "content": "Title it."})
        g = build(req, provider(), config.ModelSpec("m"))
        self.assertNotIn("response_format", g.body)
        self.assertEqual(g.body["messages"][0]["content"],
                         "Title it.\n\n" + oc.STRUCTURED_OUTPUT_NOTE +
                         '{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}')
        self.assertIn("structured_output", [k for k, _ in g.notes])

    def test_documents(self):
        msgs = [M("user", [B.of_document_base64("application/pdf", "JVBERi0", title="spec.pdf"),
                           B.of_document_text("hello doc", title="notes"), B.of_document_url("https://x/y.pdf"),
                           B.of_image_url("https://x/i.png"), B.of_text("summarize")]),
                M("assistant", [B.of_tool_use("t1", "Read", {"file_path": "a.pdf"})]),
                M("user", [B.of_tool_result("t1", [B.of_document_base64("application/pdf", "JVBE")])])]
        req = model.NormalizedRequest(model="m", messages=msgs, tools=[model.ToolDef("Read")])
        o = build(req, provider("openai"), config.ModelSpec("gpt-4.1")).body["messages"]
        self.assertEqual(o[0]["content"], [
            {"type": "file", "file": {"filename": "spec.pdf", "file_data": "data:application/pdf;base64,JVBERi0"}},
            {"type": "text", "text": "[document: notes]\nhello doc"},
            {"type": "text", "text": "[document: https://x/y.pdf]"},
            {"type": "image_url", "image_url": {"url": "https://x/i.png"}},
            {"type": "text", "text": "summarize"}])
        self.assertEqual(o[2], {"role": "tool", "tool_call_id": "t1", "content": oc.DOCUMENT_BELOW})
        self.assertEqual(o[3]["content"][0]["type"], "file")
        g = build(req, provider(), config.ModelSpec("m")).body["messages"]
        self.assertEqual(g[0]["content"][0], {"type": "text", "text": oc.PDF_OMITTED})
        self.assertEqual(g[2]["content"], oc.PDF_OMITTED)
        self.assertEqual(len(g), 3)  # no deferred user message without attachable media

    def test_history_repairs(self):
        msgs = [M("user", [B.of_text("go")]),
                M("assistant", [B.of_tool_use("a", "Bash", {}), B.of_tool_use("b", "Bash", {})]),
                M("user", [B.of_tool_result("a", ""), B.of_tool_result("zzz", "late")]),
                M("assistant", [B.of_thinking("only thinking", signatures.synthetic_signature("x"))]),
                M("assistant", [B.of_text("done")])]
        req = model.NormalizedRequest(model="m", messages=msgs, tools=[model.ToolDef("Bash")])
        out = build(req, provider(), config.ModelSpec("m"))
        self.assertEqual(out.body["messages"], [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "Bash", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": oc.EMPTY_RESULT_TEXT},
            {"role": "tool", "tool_call_id": "b", "content": oc.MISSING_RESULT_TEXT},
            {"role": "user", "content": "[tool result for zzz]\nlate"},
            {"role": "assistant", "content": "done"}])
        keys = [k for k, _ in out.notes]
        self.assertIn("missing_result", keys)
        self.assertIn("orphan_result", keys)

    def test_options_and_urls(self):
        prov = provider(options={"stream_usage": False, "extra_body": {"seed": 1}, "chat_path": "/v2/chat"},
                        base_url="http://h/api/")
        b = build(rich_request(), prov, config.ModelSpec("m"))
        self.assertNotIn("stream_options", b.body)
        self.assertEqual(b.body["seed"], 1)
        self.assertEqual(oc.chat_url(prov, config.ModelSpec("m")), "http://h/api/v2/chat")
        self.assertEqual(oc.chat_url(provider(), config.ModelSpec("m")), "http://up.invalid/v1/chat/completions")
        self.assertEqual(oc.chat_url(provider(), config.ModelSpec("m", path_override="/x/chat")), "http://up.invalid/v1/x/chat")
        self.assertEqual(oc.chat_url(provider(), config.ModelSpec("m", dialect_override="responses",
                                                                  path_override="/responses")),
                         "http://up.invalid/v1/chat/completions")
        self.assertEqual(oc.chat_url(provider("ollama", base_url=""), config.ModelSpec("m")),
                         "http://localhost:11434/v1/chat/completions")
        with self.assertRaises(errors.GatewayError):
            oc.chat_url(provider(base_url=""), config.ModelSpec("m"))

    def test_request_not_mutated(self):
        req = rich_request()
        before = copy.deepcopy(req)
        build(req, provider("moonshot"), config.ModelSpec("kimi-k3"))
        self.assertEqual(req, before)


# =========================================================================================
# stream parsing
# =========================================================================================

def chunk(delta=None, finish=None, **extra):
    c = {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta or {},
                                                                     "finish_reason": finish}]}
    c.update(extra)
    return c


def tc(index=None, id=None, name=None, args=None, typ=True):  # noqa: A002
    d = {}
    if index is not None:
        d["index"] = index
    if id is not None:
        d["id"] = id
    if typ and id is not None:
        d["type"] = "function"
    fn = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    if fn:
        d["function"] = fn
    return chunk({"tool_calls": [d]})


def parse(chunks, names=None, done=True, used=None):
    p = oc.ChatStreamParser(names or tn.ToolNameMap(["Bash", "Read", LONG_MCP]), "p", "m", used_ids=used)
    per_feed = [p.feed(c) for c in chunks]
    tail = p.finish(done)
    return [e for evs in per_feed for e in evs] + tail, per_feed, p


def kinds(evs):
    return [type(e).__name__ for e in evs]


class StreamParseTests(unittest.TestCase):
    def test_text_reasoning_keys(self):
        evs, _, _ = parse([chunk({"role": "assistant", "content": ""}), chunk({"reasoning_content": "think "}),
                           chunk({"reasoning": "more"}), chunk({"content": "Hel"}), chunk({"content": "lo"}),
                           chunk({"reasoning_content": "again"}), chunk({}, "stop")])
        self.assertEqual(evs, [events.ThinkingDelta(0, "think "), events.ThinkingDelta(0, "more"),
                               events.TextDelta(1, "Hel"), events.TextDelta(1, "lo"),
                               events.ThinkingDelta(2, "again"), events.Finish("end_turn")])
        both, _, _ = parse([chunk({"reasoning_content": "r", "reasoning": "r"})])
        self.assertEqual(both[0], events.ThinkingDelta(0, "r"))

    def test_think_tags_split_across_chunks(self):
        pieces = ["\n<thi", "nk>plan a", "nd b</th", "ink>\n\nAnswer", " <think>literal"]
        evs, _, _ = parse([chunk({"content": p}) for p in pieces] + [chunk({}, "stop")])
        thinking = "".join(e.text for e in evs if isinstance(e, events.ThinkingDelta))
        text = "".join(e.text for e in evs if isinstance(e, events.TextDelta))
        self.assertEqual(thinking, "plan and b")
        self.assertEqual(text, "Answer <think>literal")
        plain, _, _ = parse([chunk({"content": "  "}), chunk({"content": "no tags"}), chunk({}, "stop")])
        self.assertEqual("".join(e.text for e in plain if isinstance(e, events.TextDelta)), "  no tags")
        unterminated, _, _ = parse([chunk({"content": "<think>still thinking"})])
        self.assertEqual(unterminated[0], events.ThinkingDelta(0, "still thinking"))

    def test_sequential_parallel_calls_emitted_early(self):
        chunks = [tc(0, "call_a", "Bash", ""), tc(0, args='{"command":'), tc(0, args='"ls"}'),
                  tc(1, "call_b", "Read", '{"file_path":"x"}'), chunk({}, "tool_calls"),
                  chunk(None, None, choices=[], usage={"prompt_tokens": 10, "completion_tokens": 3})]
        evs, per_feed, _ = parse(chunks)
        self.assertEqual(per_feed[3], [events.ToolCall("call_a", "Bash", '{"command":"ls"}')])  # when index 1 starts
        self.assertEqual(evs[-3:], [events.ToolCall("call_b", "Read", '{"file_path":"x"}'), events.Usage(10, 3, 0),
                                    events.Finish("tool_use")])

    def test_interleaved_calls_flushed_in_index_order(self):
        chunks = [tc(0, "a", "Bash", ""), tc(1, "b", "Read", ""), tc(1, args='{"file_path"'), tc(0, args='{"command"'),
                  tc(1, args=':"f"}'), tc(0, args=':"ls"}'), chunk({}, "tool_calls")]
        evs, per_feed, _ = parse(chunks)
        self.assertEqual(per_feed[1], [])  # index 0 args incomplete -> not emitted yet
        self.assertEqual([e for e in evs if isinstance(e, events.ToolCall)],
                         [events.ToolCall("a", "Bash", '{"command":"ls"}'),
                          events.ToolCall("b", "Read", '{"file_path":"f"}')])

    def test_id_only_first_chunk_and_whole_call(self):
        evs, _, _ = parse([tc(0, "call_x"), tc(0, name="Bash"), tc(0, args='{"command":"pwd"}'),
                           chunk({}, "tool_calls")])
        self.assertEqual(evs[0], events.ToolCall("call_x", "Bash", '{"command":"pwd"}'))
        whole, _, _ = parse([tc(0, "call_y", "Bash", '{"command":"a"}'), chunk({}, "stop")])
        self.assertEqual(whole, [events.ToolCall("call_y", "Bash", '{"command":"a"}'), events.Finish("tool_use")])

    def test_missing_ids_and_indexes(self):
        evs, _, _ = parse([tc(None, None, "Bash", '{"command":"a"}'), tc(None, None, "Read", '{"file_path":"b"}'),
                           chunk({}, "tool_calls")])
        calls = [e for e in evs if isinstance(e, events.ToolCall)]
        self.assertEqual([c.name for c in calls], ["Bash", "Read"])
        self.assertTrue(all(c.id.startswith("toolu_") for c in calls))
        self.assertNotEqual(calls[0].id, calls[1].id)
        # no index but distinct ids
        evs, _, _ = parse([tc(None, "i1", "Bash", ""), tc(None, None, args="{}"), tc(None, "i2", "Read", "{}"),
                           chunk({}, "tool_calls")])
        self.assertEqual([(e.id, e.name) for e in evs if isinstance(e, events.ToolCall)],
                         [("i1", "Bash"), ("i2", "Read")])

    def test_ids_encoded_deduplicated_and_names_reverse_mapped(self):
        names = tn.ToolNameMap(["Bash", LONG_MCP])
        evs, _, _ = parse([tc(0, KIMI_ID, LONG_UP, '{"n":7}'), tc(1, "dup", "Bash", "{}"), tc(2, "dup", "Bash", "{}"),
                           chunk({}, "tool_calls")], names=names, used=["used_before"])
        calls = [e for e in evs if isinstance(e, events.ToolCall)]
        self.assertEqual(calls[0], events.ToolCall(KIMI_ENC, LONG_MCP, '{"n":7}'))
        self.assertEqual(calls[1].id, "dup")
        self.assertNotEqual(calls[2].id, "dup")
        again, _, _ = parse([tc(0, "used_before", "Bash", "{}"), chunk({}, "tool_calls")], used=["used_before"])
        self.assertNotEqual(again[0].id, "used_before")

    def test_bad_arguments(self):
        cases = [("", "{}"), ("{bad json", "{}"), ('{"a":1}{"a":1}', '{"a":1}'), ('"{\\"a\\":2}"', '{"a":2}'),
                 ("[1,2]", "{}"), ('  {"x": "é"} ', '{"x":"é"}')]
        for raw, want in cases:
            evs, _, p = parse([tc(0, "c", "Bash", raw), chunk({}, "tool_calls")])
            self.assertEqual(evs[0].input_json, want, raw)
        _, _, p = parse([tc(0, "c", "Bash", "{bad"), chunk({}, "tool_calls")])
        self.assertTrue(any("invalid JSON" in n for n in p.notes))
        dict_args, _, _ = parse([chunk({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "Bash",
                                                                                         "arguments": {"k": 1}}}]})])
        self.assertEqual(dict_args[0].input_json, '{"k":1}')

    def test_finish_reasons(self):
        for fr, want in (("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "end_turn"),
                         ("function_call", "end_turn"), ("eos", "end_turn"), ("weird", "end_turn"), (None, "end_turn")):
            evs, _, _ = parse([chunk({"content": "x"}), chunk({}, fr)])
            self.assertEqual(evs[-1], events.Finish(want), fr)
        evs, _, _ = parse([chunk({"content": "x"}), chunk({}, "content_filter")])
        self.assertEqual(evs[-2:], [events.TextDelta(1, oc.CONTENT_FILTER_TEXT), events.Finish("end_turn")])
        legacy, _, _ = parse([chunk({"function_call": {"name": "Bash", "arguments": '{"command":'}}),
                              chunk({"function_call": {"arguments": '"x"}'}}), chunk({}, "function_call")])
        self.assertEqual(kinds(legacy), ["ToolCall", "Finish"])
        self.assertEqual((legacy[0].name, legacy[0].input_json, legacy[1].stop_reason),
                         ("Bash", '{"command":"x"}', "tool_use"))
        # truncated tool call at max_tokens is dropped
        trunc, _, p = parse([tc(0, "c", "Bash", '{"command":"l'), chunk({}, "length")])
        self.assertEqual(trunc, [events.Finish("max_tokens")])
        self.assertTrue(p.notes)

    def test_truncated_stream(self):
        with self.assertRaises(errors.GatewayError) as cm:
            parse([chunk({"content": "x"})], done=False)
        self.assertTrue(cm.exception.should_retry)
        evs, _, _ = parse([chunk({"content": "x"}, "stop")], done=False)  # finish seen, no [DONE]: fine
        self.assertEqual(evs[-1], events.Finish("end_turn"))

    def test_usage_variants(self):
        def usage_of(*chunks):
            evs, _, _ = parse([chunk({"content": "x"})] + list(chunks))
            return [e for e in evs if isinstance(e, events.Usage)]

        self.assertEqual(usage_of(chunk({}, "stop"), chunk(None, None, choices=[], usage={
            "prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 30}})),
            [events.Usage(70, 20, 30)])
        self.assertEqual(usage_of(chunk({}, "stop", usage={"prompt_tokens": 100, "completion_tokens": 5,
                                                           "prompt_cache_hit_tokens": 64,
                                                           "prompt_cache_miss_tokens": 36})),
                         [events.Usage(36, 5, 64)])
        moon = chunk({}, "stop")
        moon["choices"][0]["usage"] = {"prompt_tokens": 50, "completion_tokens": 9, "cached_tokens": 10}
        self.assertEqual(usage_of(moon), [events.Usage(40, 9, 10)])
        self.assertEqual(usage_of(chunk({}, "stop", usage={"prompt_tokens": 10, "completion_tokens": 4,
                                                           "total_tokens": 30,
                                                           "completion_tokens_details": {"reasoning_tokens": 16}})),
                         [events.Usage(10, 20, 0)])
        self.assertEqual(usage_of(chunk({}, "stop", usage=None)), [])
        self.assertEqual(usage_of(chunk({}, "stop", usage={"prompt_tokens": "7", "completion_tokens": None})),
                         [events.Usage(7, 0, 0)])

    def test_in_stream_errors(self):
        p = oc.ChatStreamParser(None, "p", "m", context_window=1000, est_tokens=1500)
        p.feed(chunk({"content": "x"}))
        cases = [
            ({"error": {"message": "server overloaded", "type": "server_error"}}, (529, "overloaded_error", True)),
            ({"error": {"message": "Rate limit", "code": 429}}, (429, "rate_limit_error", True)),
            ({"error": {"message": "Provider returned error", "code": 502}}, (502, "api_error", True)),
            ({"error": {"message": "This model's maximum context length is 1000 tokens, you requested 1500",
                        "code": "context_length_exceeded"}}, (400, "invalid_request_error", False)),
            ({"error": "plain string error"}, (502, "api_error", True)),
        ]
        for obj, want in cases:
            with self.assertRaises(errors.GatewayError) as cm:
                p.feed(obj)
            e = cm.exception
            self.assertEqual((e.status, e.err_type, e.should_retry), want, obj)
        with self.assertRaises(errors.GatewayError) as cm:
            p.feed({"message": "boom"}, "error")
        self.assertEqual(cm.exception.status, 502)
        with self.assertRaises(errors.GatewayError) as cm:
            p.feed({"error": {"message": "prompt is too long", "code": "context_length_exceeded"}})
        self.assertEqual(cm.exception.message, "prompt is too long: 1500 tokens > 1000 maximum")

    def test_non_stream_message(self):
        obj = {"object": "chat.completion", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "ok", "reasoning_content": "hmm",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}},
                           {"id": "c2", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
        evs, _, _ = parse([obj])
        self.assertEqual(evs, [events.ThinkingDelta(0, "hmm"), events.TextDelta(1, "ok"),
                               events.ToolCall("c1", "Bash", "{}"), events.ToolCall("c2", "Read", "{}"),
                               events.Usage(3, 2, 0), events.Finish("tool_use")])

    def test_text_after_tool_call_gets_new_key(self):
        evs, _, _ = parse([chunk({"content": "a"}), tc(0, "c", "Bash", "{}"), tc(1, "d", "Bash", "{}"),
                           chunk({"content": "b"}), chunk({}, "tool_calls")])
        self.assertEqual(evs[:3], [events.TextDelta(0, "a"), events.ToolCall("c", "Bash", "{}"),
                                   events.TextDelta(1, "b")])

    def test_ignores_other_choices_and_garbage(self):
        evs, _, _ = parse([None, "x", {"choices": [{"index": 1, "delta": {"content": "other"}}]},
                           {"choices": [{"index": 0, "delta": {"content": "mine"}}]}, chunk({}, "stop")])
        self.assertEqual(evs, [events.TextDelta(0, "mine"), events.Finish("end_turn")])


# =========================================================================================
# execute() end-to-end against the mock kinds
# =========================================================================================

class _Res(object):
    def __init__(self, provider, model_id, background=False):
        self.provider = provider
        self.model = model_id
        self.model_spec = provider.model_spec(model_id)
        self.requested = "claude-via-%s,%s" % (provider.id, model_id)
        self.role = None
        self.background = background


def make_ctx(prov, model_id, req, secrets=None, auth=None, runtime=None, est=None):
    secrets = secrets if secrets is not None else config.SecretStore()
    auth = auth or auth_pkg.make_auth(prov.auth, secrets, prov.id)
    return dbase.RequestContext(req, _Res(prov, model_id), prov, prov.model_spec(model_id),
                                runtime or dbase.ProviderRuntime(prov.id), auth,
                                transport.HttpClient(timeout=20, environ={}), req.session_id or "sess-e2e", log=LOG,
                                est_tokens=est)


def turn1_request(tools=None, **kw):
    args = dict(model="claude-via-x", system=["You are Claude Code."],
                messages=[M("user", [B.of_text("Run: printf hello > out.txt")])],
                tools=list(TOOLS if tools is None else tools), max_tokens=32000, effort="high",
                thinking_requested=True, session_id="sess-e2e")
    args.update(kw)
    return model.NormalizedRequest(**args)


def turn2_request(turn1, evs, result="hello"):
    """What Claude Code sends back after executing the tool (thinking carries the synthetic signature)."""
    thinking = "".join(e.text for e in evs if isinstance(e, events.ThinkingDelta))
    text = "".join(e.text for e in evs if isinstance(e, events.TextDelta))
    blocks = []
    if thinking:
        blocks.append(B.of_thinking(thinking, signatures.synthetic_signature(thinking)))
    if text:
        blocks.append(B.of_text(text))
    calls = [e for e in evs if isinstance(e, events.ToolCall)]
    for c in calls:
        blocks.append(B.of_tool_use(c.id, c.name, json.loads(c.input_json)))
    msgs = list(turn1.messages) + [M("assistant", blocks),
                                   M("user", [B.of_tool_result(c.id, result) for c in calls])]
    t2 = copy.copy(turn1)
    t2.messages = msgs
    return t2


def run(ctx):
    return list(dialects.get_dialect("openai_chat").execute(ctx))


KIND_SETUPS = {
    # kind: (preset, overrides, model id, url suffix)
    "xai_chat": ("xai", {}, "grok-4.7", "/v1"),
    "openai_chat": ("openai", {"dialect": "openai_chat", "target": None, "profile": "openai"}, "gpt-5.5", "/v1"),
    "moonshot_chat": ("kimi", {"dialect": "openai_chat", "profile": "moonshot"}, "kimi-k3", "/v1"),
    "deepseek_chat": ("deepseek", {"dialect": "openai_chat", "profile": "deepseek"}, "deepseek-v4-pro", "/v1"),
    "nvidia_chat": ("nvidia", {}, "moonshotai/kimi-k2.5", "/v1"),
    "openrouter_chat": ("openrouter", {"dialect": "openai_chat", "profile": "openrouter"}, "x-ai/grok-4.7",
                        "/api/v1"),
}


def provider_for(kind, url):
    if kind == "ollama_chat":
        return presets.provider_from_preset("ollama", {"fallback": {"base_url": url + "/v1"}}).fallback
    if kind == "generic_chat":
        return config.ProviderSpec(id="local", base_url=url + "/v1", auth={"kind": "api_key", "secret": "local"})
    name, ov, _, suffix = KIND_SETUPS[kind]
    ov = dict(ov, base_url=url + suffix)
    return presets.provider_from_preset(name, ov)


def secrets_for(kind, prov):
    store = config.SecretStore()
    if prov.auth.get("kind") == "api_key":
        store.set(prov.auth["secret"], mchat.DEFAULT_KEYS[kind])
    return store


class ExecuteE2ETests(unittest.TestCase):
    maxDiff = None

    def two_turns(self, kind, model_id=None, options=None, brain=None, tools=None):
        with mocks.MockServer(kind, brain=brain, options=options) as srv:
            prov = provider_for(kind, srv.url)
            model_id = model_id or (KIND_SETUPS[kind][2] if kind in KIND_SETUPS else "mock-model")
            secrets = secrets_for(kind, prov)
            runtime = dbase.ProviderRuntime(prov.id)
            t1 = turn1_request(tools=tools)
            ev1 = run(make_ctx(prov, model_id, t1, secrets, runtime=runtime))
            calls = [e for e in ev1 if isinstance(e, events.ToolCall)]
            self.assertEqual(len(calls), 1, ev1)
            self.assertEqual(ev1[-1], events.Finish("tool_use"))
            self.assertEqual(json.loads(calls[0].input_json)["command"], mocks.DEFAULT_COMMAND)
            self.assertTrue(tn.TOOL_ID_RE.match(calls[0].id), calls[0].id)
            t2 = turn2_request(t1, ev1)
            ev2 = run(make_ctx(prov, model_id, t2, secrets, runtime=runtime))
            text = "".join(e.text for e in ev2 if isinstance(e, events.TextDelta))
            self.assertEqual(text, "DONE " + mocks.sha8("hello"))
            self.assertEqual(ev2[-1], events.Finish("end_turn"))
            self.assertEqual(srv.errors, [])
            chats = srv.requests_for("", "POST")
            chats = [r for r in chats if r["path"].endswith("/chat/completions")]
            self.assertEqual(len(chats), 2)
            for r in chats:
                self.assertTrue(r["body_json"]["stream"])
            return ev1, ev2, srv, chats, runtime

    def test_xai(self):
        ev1, _, _, chats, _ = self.two_turns("xai_chat")
        self.assertTrue(any(isinstance(e, events.ThinkingDelta) for e in ev1))
        body = chats[0]["body_json"]
        self.assertEqual(chats[0]["headers"].get("x-grok-conv-id"), "sess-e2e")
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["stream_options"], {"include_usage": True})
        usage = [e for e in ev1 if isinstance(e, events.Usage)][0]
        self.assertGreater(usage.cache_read, 0)
        self.assertGreater(usage.output_tokens, 0)

    def test_xai_rejects_quirks_when_unscrubbed(self):
        with mocks.MockServer("xai_chat") as srv:
            prov = provider_for("xai_chat", srv.url)
            ctx = make_ctx(prov, "grok-4.7", turn1_request(stop_sequences=["X"]), secrets_for("xai_chat", prov))
            run(ctx)  # dropped by the profile -> accepted
            raw = transport.HttpClient(timeout=10, environ={}).request(
                "POST", srv.url + "/v1/chat/completions",
                {"Authorization": "Bearer " + mchat.DEFAULT_KEYS["xai_chat"], "Content-Type": "application/json"},
                json.dumps({"model": "grok-4.7", "messages": [{"role": "user", "content": "x"}], "stop": ["X"]}),
                stream=False)
            self.assertEqual(raw.status, 400)
            e = errors.map_upstream_error(raw.status, raw.text(), raw.headers, "xai", "grok-4.7")
            self.assertEqual(e.err_type, "invalid_request_error")
            self.assertIn("stop", e.message)

    def test_openai_chat_variant(self):
        _, _, _, chats, _ = self.two_turns("openai_chat")
        body = chats[0]["body_json"]
        self.assertIn("max_completion_tokens", body)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(body["prompt_cache_key"], "sess-e2e")

    def test_moonshot_reasoning_round_trip_and_encoded_ids(self):
        ev1, _, srv, chats, _ = self.two_turns("moonshot_chat")
        call = [e for e in ev1 if isinstance(e, events.ToolCall)][0]
        self.assertTrue(call.id.startswith("toolu_x"))
        self.assertEqual(tn.decode_tool_id(call.id), KIMI_ID)
        asst = [m for m in chats[1]["body_json"]["messages"] if m["role"] == "assistant"][0]
        self.assertEqual(asst["tool_calls"][0]["id"], KIMI_ID)
        self.assertTrue(asst["reasoning_content"])
        self.assertNotIn("temperature", chats[0]["body_json"])

    def test_moonshot_rejects_missing_reasoning(self):
        with mocks.MockServer("moonshot_chat") as srv:
            prov = provider_for("moonshot_chat", srv.url)
            secrets = secrets_for("moonshot_chat", prov)
            t1 = turn1_request()
            ev1 = run(make_ctx(prov, "kimi-k3", t1, secrets))
            stripped = [e for e in ev1 if not isinstance(e, events.ThinkingDelta)]
            t2 = turn2_request(t1, stripped)
            prov.profile = "generic"  # no reasoning_content echo -> Kimi's thinking check fires
            with self.assertRaises(errors.GatewayError) as cm:
                run(make_ctx(prov, "kimi-k3", t2, secrets))
            self.assertEqual(cm.exception.status, 400)
            self.assertIn("reasoning_content is missing", cm.exception.message)

    def test_deepseek(self):
        ev1, ev2, _, chats, _ = self.two_turns("deepseek_chat")
        u = [e for e in ev1 if isinstance(e, events.Usage)][0]
        self.assertGreater(u.cache_read, 0)
        self.assertNotIn("temperature", chats[0]["body_json"])

    def test_nvidia_long_mcp_tool_name(self):
        brain = mocks.Brain(tool=LONG_MCP)
        with mocks.MockServer("nvidia_chat", brain=brain) as srv:
            prov = provider_for("nvidia_chat", srv.url)
            secrets = secrets_for("nvidia_chat", prov)
            evs = run(make_ctx(prov, "moonshotai/kimi-k2.5", turn1_request(), secrets))
            call = [e for e in evs if isinstance(e, events.ToolCall)][0]
            self.assertEqual(call.name, LONG_MCP)
            body = srv.requests_for("/v1/chat")[0]["body_json"]
            self.assertIn(LONG_UP, [t["function"]["name"] for t in body["tools"]])
            self.assertNotIn("stream_options", body)
            self.assertTrue(any(isinstance(e, events.Usage) for e in evs))  # usage in the final chunk
            self.assertEqual(srv.errors, [])

    def test_nvidia(self):
        self.two_turns("nvidia_chat")

    def test_openrouter(self):
        ev1, _, _, chats, _ = self.two_turns("openrouter_chat")
        self.assertEqual(chats[0]["headers"].get("HTTP-Referer"), cp.OPENROUTER_REFERER)
        self.assertEqual(chats[0]["body_json"]["reasoning"], {"effort": "high"})
        self.assertTrue(any(isinstance(e, events.ThinkingDelta) for e in ev1))
        call = [e for e in ev1 if isinstance(e, events.ToolCall)][0]
        self.assertTrue(call.id.startswith("toolu_vrtx_01"))  # valid upstream ids pass through

    def test_generic_with_missing_ids(self):
        ev1, _, _, chats, _ = self.two_turns("generic_chat", options={"tool_style": "no_ids"})
        call = [e for e in ev1 if isinstance(e, events.ToolCall)][0]
        self.assertRegex(call.id, r"^toolu_[0-9a-f]{24}$")
        self.assertEqual(chats[1]["body_json"]["messages"][-1]["tool_call_id"], call.id)

    def test_ollama_thinking_model(self):
        ev1, _, srv, chats, runtime = self.two_turns("ollama_chat", model_id="qwen3:8b")
        self.assertEqual(chats[0]["body_json"]["reasoning_effort"], "high")
        self.assertTrue(any(isinstance(e, events.ThinkingDelta) for e in ev1))
        self.assertEqual(srv.state.get("show_calls"), 1)  # probed once, cached in the runtime
        self.assertTrue(runtime.get("ollama_caps")["qwen3:8b"]["thinking"])
        self.assertEqual(srv.requests_for("/api/show")[0]["body_json"], {"model": "qwen3:8b"})

    def test_ollama_non_thinking_and_chat_only(self):
        chats = self.two_turns("ollama_chat", model_id="llama3.2:3b")[3]
        self.assertNotIn("reasoning_effort", chats[0]["body_json"])  # mock would 400 "does not support thinking"
        with mocks.MockServer("ollama_chat") as srv:
            prov = provider_for("ollama_chat", srv.url)
            runtime = dbase.ProviderRuntime(prov.id)
            evs = run(make_ctx(prov, "gemma3:4b", turn1_request(), runtime=runtime))
            self.assertEqual("".join(e.text for e in evs if isinstance(e, events.TextDelta)), "Background reply.")
            self.assertNotIn("tools", srv.requests_for("/v1/chat")[0]["body_json"])
            self.assertFalse(runtime.get("ollama_caps")["gemma3:4b"]["tools"])
            self.assertEqual(srv.errors, [])

    def test_ollama_probe_failure_cached_briefly(self):
        def handler(server, req, resp):
            if req.path == "/api/show":
                resp.send_text(500, "nope")
            else:
                mchat_factory = mocks.get_kind_factory("ollama_chat")(server)
                mchat_factory(req, resp)

        with mocks.MockServer(handler=handler) as srv:
            prov = provider_for("ollama_chat", srv.url)
            runtime = dbase.ProviderRuntime(prov.id)
            for _ in range(2):
                evs = run(make_ctx(prov, "qwen3:8b", turn1_request(), runtime=runtime))
                self.assertTrue(any(isinstance(e, events.ToolCall) for e in evs))
            self.assertEqual(len(srv.requests_for("/api/show")), 1)
            caps = runtime.get("ollama_caps")["qwen3:8b"]
            self.assertFalse(caps["probed"])
            self.assertNotIn("reasoning_effort", srv.requests_for("/v1/chat")[0]["body_json"])

    def test_ollama_unknown_model_404(self):
        with mocks.MockServer("ollama_chat") as srv:
            prov = provider_for("ollama_chat", srv.url)
            with self.assertRaises(errors.GatewayError) as cm:
                run(make_ctx(prov, "nope:1b", turn1_request()))
            self.assertEqual((cm.exception.status, cm.exception.err_type), (404, "not_found_error"))

    def test_non_stream_upstream(self):
        ev1, _, _, _, _ = self.two_turns("generic_chat", options={"ignore_stream": True})
        self.assertTrue(any(isinstance(e, events.Usage) for e in ev1))

    def test_background_structured_output(self):
        fmt = {"type": "json_schema", "schema": {"type": "object", "properties": {"title": {"type": "string"}}}}
        with mocks.MockServer("openai_chat") as srv:
            prov = provider_for("openai_chat", srv.url)
            req = turn1_request(tools=[], output_format=fmt, effort="low")
            evs = run(make_ctx(prov, "gpt-5.4-mini", req, secrets_for("openai_chat", prov)))
            self.assertEqual("".join(e.text for e in evs if isinstance(e, events.TextDelta)), "Background reply.")
            body = srv.requests_for("/v1/chat")[0]["body_json"]
            self.assertEqual(body["response_format"]["type"], "json_schema")
            self.assertEqual(body["reasoning_effort"], "low")
            for k in ("tools", "tool_choice", "parallel_tool_calls"):
                self.assertNotIn(k, body)

    def test_unauthorized_refresh_retry(self):
        class Rotating(auth_static.StaticKeyAuth):
            def __init__(self, store, good):
                auth_static.StaticKeyAuth.__init__(self, "k", store, provider_id="xai")
                self.good = good
                self.calls = 0

            def on_unauthorized(self, used):
                self.calls += 1
                self.store.set("k", self.good)
                return auth_static.StaticKeyAuth.on_unauthorized(self, used)

        with mocks.MockServer("xai_chat") as srv:
            prov = provider_for("xai_chat", srv.url)
            store = config.SecretStore({"k": "stale-key"})
            auth = Rotating(store, mchat.DEFAULT_KEYS["xai_chat"])
            evs = run(make_ctx(prov, "grok-4.7", turn1_request(), auth=auth))
            self.assertEqual(auth.calls, 1)
            self.assertTrue(any(isinstance(e, events.ToolCall) for e in evs))
            auths = [r["headers"].get("Authorization") for r in srv.requests_for("/v1/chat")]
            self.assertEqual(auths, ["Bearer stale-key", "Bearer " + mchat.DEFAULT_KEYS["xai_chat"]])
            # a key that stays wrong -> 401 authentication_error with a hint, no further retry
            bad = config.SecretStore({"xai": "wrong"})
            with self.assertRaises(errors.GatewayError) as cm:
                run(make_ctx(prov, "grok-4.7", turn1_request(), bad))
            e = cm.exception
            self.assertEqual((e.status, e.err_type, e.should_retry), (401, "authentication_error", False))
            self.assertIn("check the API key", e.message)
            self.assertNotIn("wrong", e.message)

    def test_mid_stream_error_after_commit(self):
        err = {"error": {"message": "upstream overloaded", "type": "server_error"}}
        with mocks.MockServer("openai_chat", options={"stream_error": err}) as srv:
            prov = provider_for("openai_chat", srv.url)
            evs = run(make_ctx(prov, "gpt-5.5", turn1_request(), secrets_for("openai_chat", prov)))
            self.assertEqual(evs[0], events.TextDelta(0, "partial answer "))
            self.assertEqual(evs[-1], events.StreamError("overloaded_error", evs[-1].message, True))
        with mocks.MockServer("openai_chat", options={"stream_error": err, "stream_error_at": "start"}) as srv:
            prov = provider_for("openai_chat", srv.url)
            with self.assertRaises(errors.GatewayError) as cm:
                run(make_ctx(prov, "gpt-5.5", turn1_request(), secrets_for("openai_chat", prov)))
            self.assertEqual(cm.exception.status, 529)

    def test_http_errors_mapped(self):
        overflow = {"error": {"message": "This model's maximum context length is 1000 tokens. However, your "
                                         "messages resulted in 1500 tokens.", "code": "context_length_exceeded"}}
        with mocks.MockServer("deepseek_chat", options={"fail_status": 400, "fail_body": overflow}) as srv:
            prov = provider_for("deepseek_chat", srv.url)
            with self.assertRaises(errors.GatewayError) as cm:
                run(make_ctx(prov, "deepseek-v4-pro", turn1_request(), secrets_for("deepseek_chat", prov)))
            self.assertEqual(cm.exception.message, "prompt is too long: 1500 tokens > 1000 maximum")
        balance = {"error": {"message": "Insufficient Balance", "type": "unknown_error"}}
        with mocks.MockServer("deepseek_chat", options={"fail_status": 402, "fail_body": balance}) as srv:
            prov = provider_for("deepseek_chat", srv.url)
            with self.assertRaises(errors.GatewayError) as cm:
                run(make_ctx(prov, "deepseek-v4-pro", turn1_request(), secrets_for("deepseek_chat", prov)))
            self.assertEqual((cm.exception.status, cm.exception.should_retry), (429, False))

    def test_connection_error(self):
        import socket

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        prov = config.ProviderSpec(id="local", base_url="http://127.0.0.1:%d/v1" % port, auth={"kind": "none"})
        with self.assertRaises(errors.GatewayError) as cm:
            run(make_ctx(prov, "m", turn1_request()))
        self.assertTrue(cm.exception.connection_error)

    def test_mock_quirks_enforced(self):
        """The mocks reject what the profiles strip (so e2e passes prove the profiles work)."""
        def post(srv, kind, body, path="/v1/chat/completions"):
            key = mchat.DEFAULT_KEYS[kind]
            h = {"Content-Type": "application/json"}
            if key:
                h["Authorization"] = "Bearer " + key
            r = transport.HttpClient(timeout=10, environ={}).request("POST", srv.url + path, h, json.dumps(body),
                                                                    stream=False)
            return r.status, r.text()

        msgs = [{"role": "user", "content": "hi"}]
        cases = [
            ("openai_chat", {"model": "gpt-5.5", "messages": msgs, "max_tokens": 5}, "max_completion_tokens"),
            ("openai_chat", {"model": "o3", "messages": msgs, "temperature": 0.5}, "temperature"),
            ("moonshot_chat", {"model": "kimi-k3", "messages": msgs, "temperature": 1}, "temperature"),
            ("nvidia_chat", {"model": "m", "messages": msgs, "tools": [
                {"type": "function", "function": {"name": "a.b", "parameters": {}}}]}, "pattern"),
            ("xai_chat", {"model": "grok-4.7", "messages": msgs, "tools": [
                {"type": "function", "function": {"name": "x" * 65, "parameters": {}}}]}, "Invalid function name"),
            ("xai_chat", {"model": "grok-4.7", "messages": msgs + [{"role": "tool", "tool_call_id": "nope",
                                                                     "content": "x"}]}, "tool_call_id"),
            ("ollama_chat", {"model": "llama3.2:3b", "messages": msgs, "reasoning_effort": "high"},
             '\\"llama3.2:3b\\" does not support thinking'),
            ("generic_chat", {"model": "m", "messages": msgs, "parallel_tool_calls": False}, "extra_forbidden"),
            ("deepseek_chat", {"model": "m", "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:x"}}]}]}, "unknown variant"),
        ]
        for kind, body, needle in cases:
            with mocks.MockServer(kind) as srv:
                status, text = post(srv, kind, body)
                self.assertEqual(status, 400, (kind, text))
                self.assertIn(needle, text, kind)
        with mocks.MockServer("ollama_chat") as srv:
            c = transport.HttpClient(timeout=10, environ={})
            tags = c.request("GET", srv.url + "/api/tags", stream=False).json()
            self.assertEqual(sorted(m["name"] for m in tags["models"]), sorted(mchat.DEFAULT_OLLAMA_MODELS))
            show = c.request("POST", srv.url + "/api/show", {"Content-Type": "application/json"},
                             json.dumps({"model": "qwen3:8b"}), stream=False).json()
            self.assertIn("thinking", show["capabilities"])
        with mocks.MockServer("openai_chat") as srv:
            r = transport.HttpClient(timeout=10, environ={}).request(
                "POST", srv.url + "/v1/chat/completions", {"Content-Type": "application/json"},
                json.dumps({"model": "m", "messages": msgs}), stream=False)
            self.assertEqual(r.status, 401)
            self.assertIn("invalid_api_key", r.text())


if __name__ == "__main__":
    unittest.main()
