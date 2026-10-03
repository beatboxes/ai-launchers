"""Responses dialect (DESIGN §3.2): request goldens per target, input conversion, stream parsing, signature
round trips, Responses-Lite, end-to-end execute() against every mock kind, 401 refresh, terminal 429 and the
sticky Grok fallback."""

import json
import logging
import socket
import unittest

from ._pkg import mod

ev = mod("events")
model = mod("model")
config = mod("config")
errors = mod("errors")
sigs = mod("signatures")
compat = mod("compat")
transport = mod("transport")
dbase = mod("dialects.base")
dialects = mod("dialects")
rd = mod("dialects.responses")
rt = mod("responses_targets")
toolnames = mod("toolnames")
auth_base = mod("auth.base")
auth_static = mod("auth.static")
mocks = mod("testing.mock_upstreams")
mock_responses = mod("testing.mock_responses")

Block, Message, ToolDef = model.Block, model.Message, model.ToolDef
LONG_TOOL = "mcp__very_long_server_name_for_testing__" + "x" * 45  # 85 chars, like real MCP tools
BASH_SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
               "properties": {"command": {"type": "string"}}, "required": ["command"],
               "additionalProperties": False}
BASH_PARAMS = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"],
               "additionalProperties": False}


# =========================================================================================
# helpers
# =========================================================================================

class _Res(object):
    def __init__(self, background=False):
        self.background = background


def provider(target, base_url="", models=None, **kw):
    kw.setdefault("auth", {"kind": "none"})
    return config.ProviderSpec(id=kw.pop("id", target), dialect="responses", target=target, base_url=base_url,
                               models=models or [], allow_unlisted=True, **kw)


def ctx_for(prov, model_id, req, auth=None, runtime=None, session="sess-1", background=False, est=None):
    runtime = runtime or dbase.ProviderRuntime(prov.id)
    runtime.setdefault("versions", dict).update({"codex": "0.160.0", "grok": "1.0.0"})
    return dbase.RequestContext(
        req=req, resolution=_Res(background), provider=prov, model=prov.model_spec(model_id), runtime=runtime,
        auth=auth or auth_static.NoAuth(prov.id), http=transport.HttpClient(timeout=20, environ={}),
        session_id=session, est_tokens=est)


def turn1(tools=None, **kw):
    kw.setdefault("system", ["You are Claude Code.", "Be terse."])
    kw.setdefault("messages", [Message("user", [Block.of_text("Run the shell command and report")])])
    kw.setdefault("effort", "high")
    kw.setdefault("session_id", "sess-1")
    if tools is None:
        tools = [ToolDef("Bash", "Run a shell command", BASH_SCHEMA),
                 ToolDef(LONG_TOOL, "An MCP tool", {"type": "object", "properties": {}})]
    return model.NormalizedRequest(model="claude-via-test", tools=tools, **kw)


def assistant_from_events(events):
    """Rebuild the assistant message Claude Code would send back from internal events."""
    blocks, thinking = [], {}
    for e in events:
        if isinstance(e, ev.ThinkingDelta):
            if e.key not in thinking:
                thinking[e.key] = Block.of_thinking("")
                blocks.append(thinking[e.key])
            thinking[e.key].thinking += e.text
        elif isinstance(e, ev.ThinkingSignature):
            thinking[e.key].signature = e.signature
        elif isinstance(e, ev.TextDelta):
            if blocks and blocks[-1].type == "text" and getattr(blocks[-1], "_key", None) == e.key:
                blocks[-1].text += e.text
            else:
                b = Block.of_text(e.text)
                b._key = e.key
                blocks.append(b)
        elif isinstance(e, ev.ToolCall):
            blocks.append(Block.of_tool_use(e.id, e.name, json.loads(e.input_json)))
    return Message("assistant", blocks)


def run(ctx):
    return list(rd.ResponsesDialect().execute(ctx))


def text_of(events):
    return "".join(e.text for e in events if isinstance(e, ev.TextDelta))


def jwt(exp):
    return "e30.%s.sig" % compat.b64url_encode(json.dumps({"exp": exp}))


class FakeCodexAuth(auth_base.AuthProvider):
    """Authorization + ChatGPT-Account-ID like CodexChatGPTAuth; refresh mints a fresh JWT."""

    kind = "codex_chatgpt"

    def __init__(self, expired_first=False):
        auth_base.AuthProvider.__init__(self, "codex")
        now = compat.utcnow_epoch()
        self.token = jwt(now - 60 if expired_first else now + 3600)
        self.refreshes = 0

    def available(self):
        return True

    def headers(self, force_refresh=False):
        return {"Authorization": "Bearer " + self.token, "ChatGPT-Account-ID": "acct-123"}

    def on_unauthorized(self, used):
        self.refreshes += 1
        self.token = jwt(compat.utcnow_epoch() + 3600 + self.refreshes)
        return True

    def relogin_hint(self):
        return "run `codex login`"


def static_auth(name="k", value="test-key"):
    return auth_static.StaticKeyAuth(name, config.SecretStore({name: value}), provider_id=name)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def feed_all(parser, items):
    out = []
    for etype, data in items:
        data = dict(data, type=etype)
        out.extend(parser.feed(etype, data))
    out.extend(parser.end())
    return out


# =========================================================================================
# request goldens
# =========================================================================================

class RequestGoldenTests(unittest.TestCase):
    def simple(self, **kw):
        return turn1(tools=[ToolDef("Bash", "Run", BASH_SCHEMA)],
                     messages=[Message("user", [Block.of_text("hi")])], system=["SYS A", "SYS B"], **kw)

    def common(self, model_id):
        return {
            "model": model_id, "instructions": "SYS A\n\nSYS B",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "tools": [{"type": "function", "name": "Bash", "description": "Run", "parameters": BASH_PARAMS,
                       "strict": False}],
            "tool_choice": "auto", "parallel_tool_calls": True, "store": False, "stream": True,
        }

    def test_openai_api(self):
        p = provider("openai_api", "https://api.openai.com/v1", [config.ModelSpec("gpt-5.5", max_output=128000,
                                                                                  reasoning=True)])
        prep = rd.build_request(ctx_for(p, "gpt-5.5", self.simple()))
        want = self.common("gpt-5.5")
        want.update(reasoning={"effort": "high", "summary": "auto"}, include=["reasoning.encrypted_content"],
                    prompt_cache_key="sess-1", max_output_tokens=32000)
        self.assertEqual(prep.body, want)
        self.assertEqual(prep.url, "https://api.openai.com/v1/responses")
        self.assertEqual(json.loads(prep.data().decode("utf-8")), want)

    def test_openai_api_non_reasoning(self):
        p = provider("openai_api", models=[config.ModelSpec("gpt-4.1", max_output=16000)])
        prep = rd.build_request(ctx_for(p, "gpt-4.1", self.simple(temperature=0.2, top_p=0.5)))
        want = self.common("gpt-4.1")
        want.update(prompt_cache_key="sess-1", max_output_tokens=16000, temperature=0.2, top_p=0.5)
        self.assertEqual(prep.body, want)

    def test_chatgpt_codex(self):
        p = provider("chatgpt_codex", "https://chatgpt.com/backend-api/codex", max_tool_name=64)
        prep = rd.build_request(ctx_for(p, "gpt-5.5", self.simple(temperature=0.5)))
        want = self.common("gpt-5.5")
        want.update(reasoning={"effort": "high", "summary": "auto"}, include=["reasoning.encrypted_content"],
                    prompt_cache_key="sess-1")
        self.assertEqual(prep.body, want)
        self.assertEqual(prep.url, "https://chatgpt.com/backend-api/codex/responses")
        self.assertEqual(prep.headers["originator"], "codex_cli_rs")
        self.assertEqual(prep.headers["version"], "0.160.0")
        self.assertEqual(prep.headers["session-id"], "sess-1")
        self.assertFalse(prep.lite)

    def test_grok_cli_proxy(self):
        p = provider("grok_cli_proxy", "https://cli-chat-proxy.grok.com/v1", schema_mode="no_root_combinators",
                     headers={"X-Extra": "1"})
        prep = rd.build_request(ctx_for(p, "grok-4.7", self.simple(temperature=0.5)))
        want = self.common("grok-4.7")
        want.update(include=["reasoning.encrypted_content"])
        self.assertEqual(prep.body, want)
        self.assertEqual(prep.headers["x-grok-client-version"], "1.0.0")
        self.assertEqual(prep.headers["x-grok-conv-id"], "sess-1")
        self.assertEqual(prep.headers["X-Extra"], "1")

    def test_xai_api(self):
        spec = config.ModelSpec("grok-4.20-multi-agent-0309", max_output=64000, reasoning=True,
                                dialect_override="responses", target_override="xai_api", path_override="/responses")
        p = config.ProviderSpec(id="xai", dialect="openai_chat", profile="xai", base_url="https://api.x.ai/v1",
                                models=[spec])
        ctx = ctx_for(p, spec.id, self.simple(max_tokens=100000))
        self.assertEqual(rd.target_for(ctx).name, "xai_api")
        prep = rd.build_request(ctx)
        want = self.common(spec.id)
        want.update(include=["reasoning.encrypted_content"], max_output_tokens=64000)
        self.assertEqual(prep.body, want)
        self.assertEqual(prep.url, "https://api.x.ai/v1/responses")
        spec.effort_param = True
        self.assertEqual(rd.build_request(ctx).body["reasoning"], {"effort": "high"})

    def test_effort_tool_choice_and_parallel(self):
        p = provider("openai_api")
        body = rd.build_request(ctx_for(p, "gpt-5.5", self.simple(effort="max", tool_choice={"type": "any"},
                                                                    disable_parallel_tool_use=True))).body
        self.assertEqual((body["reasoning"]["effort"], body["tool_choice"], body["parallel_tool_calls"]),
                         ("high", "required", False))
        body = rd.build_request(ctx_for(p, "gpt-5.5", self.simple(effort=None, tool_choice={"type": "none"}))).body
        self.assertEqual((body["reasoning"]["effort"], body["tool_choice"]), ("medium", "none"))
        body = rd.build_request(ctx_for(p, "gpt-5.5", self.simple(effort="high"), background=True)).body
        self.assertEqual(body["reasoning"]["effort"], "low")
        req = turn1(tool_choice={"type": "tool", "name": LONG_TOOL})
        body = rd.build_request(ctx_for(p, "gpt-5.5", req)).body
        short = body["tool_choice"]["name"]
        self.assertEqual(body["tool_choice"]["type"], "function")
        self.assertLessEqual(len(short), 64)
        self.assertIn(short, [t["name"] for t in body["tools"]])

    def test_no_tools_no_tool_params_and_default_instructions(self):
        p = provider("chatgpt_codex")
        body = rd.build_request(ctx_for(p, "gpt-5.5", turn1(tools=[], system=[]))).body
        self.assertEqual(body["instructions"], rd.DEFAULT_INSTRUCTIONS)
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            self.assertNotIn(key, body)

    def test_schema_modes(self):
        schema = {"$schema": "x", "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]}]}
        req = turn1(tools=[ToolDef("Pick", "", schema)])
        grok = rd.build_request(ctx_for(provider("grok_cli_proxy"), "grok-4.7", req)).body["tools"][0]
        self.assertNotIn("anyOf", grok["parameters"])
        self.assertEqual(grok["parameters"]["type"], "object")
        self.assertEqual(set(grok["parameters"]["properties"]), {"a", "b"})
        oa = rd.build_request(ctx_for(provider("openai_api"), "gpt-5.5", req)).body["tools"][0]
        self.assertIn("anyOf", oa["parameters"])
        self.assertNotIn("$schema", oa["parameters"])

    def test_structured_output(self):
        schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
        req = turn1(tools=[], output_format={"type": "json_schema", "schema": schema})
        body = rd.build_request(ctx_for(provider("openai_api"), "gpt-5.5", req)).body
        self.assertEqual(body["text"], {"format": {"type": "json_schema", "name": "output", "schema": schema,
                                                   "strict": False}})
        body = rd.build_request(ctx_for(provider("grok_cli_proxy"), "grok-4.7", req)).body
        self.assertNotIn("text", body)
        self.assertIn("Respond with ONLY a JSON object matching this JSON schema", body["instructions"])

    def test_responses_lite(self):
        p = provider("chatgpt_codex", models=[config.ModelSpec("gpt-6.1", reasoning=True)])
        prep = rd.build_request(ctx_for(p, "gpt-6.1", turn1()))
        self.assertTrue(prep.lite)
        self.assertNotIn("instructions", prep.body)
        self.assertEqual(prep.body["input"][0]["role"], "developer")
        self.assertEqual(prep.body["input"][0]["content"][0]["text"], "You are Claude Code.\n\nBe terse.")
        self.assertEqual(prep.headers[rt.LITE_HEADER], "true")
        p = provider("chatgpt_codex", models=[config.ModelSpec("my-model", responses_lite=True)])
        self.assertTrue(rd.build_request(ctx_for(p, "my-model", turn1())).lite)


# =========================================================================================
# input conversion
# =========================================================================================

def _history(own_tag, foreign_tag="xai_api"):
    """user(text+image+pdf) / assistant(thinking x3, text, tool_use) / user(tool_result w/ image, text)."""
    own = sigs.encode_signature(own_tag, {"enc": "ENC1", "summary": ["I think", "more"]})
    foreign = sigs.encode_signature(foreign_tag, {"enc": "ENC-FOREIGN", "summary": ["x"]})
    return [
        Message("user", [Block.of_text("look"), Block.of_image_base64("image/png", "AAAA"),
                         Block.of_document_base64("application/pdf", "PDFDATA", title="spec.pdf"),
                         Block.of_document_text("plain body", title="notes.txt")]),
        Message("assistant", [Block.of_thinking("I think\n\nmore", own), Block.of_thinking("x", foreign),
                              Block.of_thinking("anthropic", "EqAnthropicSignature=="),
                              Block.of_redacted_thinking("zzz"),
                              Block.of_text("Calling"), Block.of_tool_use("toolu_1", LONG_TOOL, {"a": 1})]),
        Message("user", [Block.of_tool_result("toolu_1", [Block.of_text("out"), Block.of_image_base64("image/png",
                                                                                                     "IMG")],
                                              is_error=True),
                         Block.of_text("<system-reminder>\nnote\n</system-reminder>")]),
    ]


class InputConversionTests(unittest.TestCase):
    def build(self, target, model_id, messages, **kw):
        p = provider(target, **kw)
        return rd.build_request(ctx_for(p, model_id, turn1(messages=messages)))

    def test_openai_api_history(self):
        prep = self.build("openai_api", "gpt-5.5", _history("openai_api"))
        short = prep.names.upstream(LONG_TOOL)
        self.assertEqual(prep.body["input"], [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                {"type": "input_file", "filename": "spec.pdf", "file_data": "data:application/pdf;base64,PDFDATA"},
                {"type": "input_text", "text": "notes.txt\n\nplain body"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I think"},
                                              {"type": "summary_text", "text": "more"}],
             "encrypted_content": "ENC1"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Calling"}]},
            {"type": "function_call", "call_id": "toolu_1", "name": short, "arguments": '{"a":1}'},
            {"type": "function_call_output", "call_id": "toolu_1", "output": "ERROR: out\n[image attached below]"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,IMG"},
                {"type": "input_text", "text": "<system-reminder>\nnote\n</system-reminder>"}]},
        ])
        self.assertLessEqual(len(short), 64)
        self.assertEqual(prep.names.original(short), LONG_TOOL)
        self.assertIn(short, [t["name"] for t in prep.body["tools"]])

    def test_codex_drops_foreign_thinking_and_pdf(self):
        body = self.build("chatgpt_codex", "gpt-5.5", _history("codex_chatgpt", "openai_api")).body
        reasoning = [i for i in body["input"] if i["type"] == "reasoning"]
        self.assertEqual([r["encrypted_content"] for r in reasoning], ["ENC1"])
        self.assertEqual(body["input"][0]["content"][2], {"type": "input_text", "text": rd.PDF_OMITTED})
        # the openai_api-signed history is dropped entirely on the codex target
        body = self.build("chatgpt_codex", "gpt-5.5", _history("openai_api")).body
        self.assertEqual([i for i in body["input"] if i["type"] == "reasoning"], [])

    def test_reasoning_dropped_for_non_reasoning_model_or_missing_enc(self):
        body = self.build("openai_api", "gpt-4.1", _history("openai_api")).body
        self.assertEqual([i for i in body["input"] if i["type"] == "reasoning"], [])
        no_enc = sigs.encode_signature("openai_api", {"enc": None, "summary": ["s"]})
        msgs = [Message("user", [Block.of_text("q")]), Message("assistant", [Block.of_thinking("s", no_enc),
                                                                             Block.of_text("a")]),
                Message("user", [Block.of_text("q2")])]
        body = self.build("openai_api", "gpt-5.5", msgs).body
        self.assertEqual([i["type"] for i in body["input"]], ["message", "message", "message"])

    def test_tool_ids(self):
        weird = toolnames.encode_tool_id("call|weird/id")
        long_id = "toolu_" + "a" * 100
        msgs = [Message("user", [Block.of_text("q")]),
                Message("assistant", [Block.of_tool_use(weird, "Bash", {}), Block.of_tool_use(long_id, "Bash", {})]),
                Message("user", [Block.of_tool_result(weird, "r1"), Block.of_tool_result(long_id, "r2")])]
        oa = self.build("openai_api", "gpt-5.5", msgs).body["input"]
        self.assertEqual([i["call_id"] for i in oa[1:]], ["call|weird/id", long_id, "call|weird/id", long_id])
        cx = self.build("chatgpt_codex", "gpt-5.5", msgs).body["input"]
        ids = [i["call_id"] for i in cx[1:]]
        self.assertTrue(all(len(i) <= 64 for i in ids))
        self.assertEqual(ids[1], ids[3])
        self.assertNotEqual(ids[1], long_id)

    def test_orphans(self):
        msgs = [Message("user", [Block.of_text("q")]),
                Message("assistant", [Block.of_tool_use("toolu_a", "Bash", {"command": "ls"})]),
                Message("user", [Block.of_tool_result("toolu_unknown", "stray"), Block.of_text("go on")])]
        items = self.build("openai_api", "gpt-5.5", msgs).body["input"]
        self.assertEqual([i["type"] for i in items], ["message", "function_call", "function_call_output", "message"])
        self.assertEqual(items[2]["call_id"], "toolu_a")
        self.assertIn("interrupted", items[2]["output"])
        self.assertEqual(items[3]["content"][0]["text"], "[tool result toolu_unknown]\nstray")

    def test_tool_result_pdf_and_url_media(self):
        msgs = [Message("user", [Block.of_text("q")]),
                Message("assistant", [Block.of_tool_use("toolu_a", "Read", {})]),
                Message("user", [Block.of_tool_result("toolu_a", [
                    Block.of_document_base64("application/pdf", "PDF"), Block.of_image_url("https://x/i.png")])])]
        grok = self.build("grok_cli_proxy", "grok-4.7", msgs).body["input"]
        self.assertEqual(grok[2]["output"], rd.PDF_OMITTED + "\n" + rd.IMAGE_NOTE)
        self.assertEqual(grok[3]["content"], [{"type": "input_image", "image_url": "https://x/i.png"}])
        oa = self.build("openai_api", "gpt-5.5", msgs).body["input"]
        self.assertEqual(oa[2]["output"], rd.DOCUMENT_NOTE + "\n" + rd.IMAGE_NOTE)
        self.assertEqual(oa[3]["content"][0]["type"], "input_file")


# =========================================================================================
# stream parsing
# =========================================================================================

def _added(idx, item):
    return ("response.output_item.added", {"output_index": idx, "item": item})


def _done(idx, item):
    return ("response.output_item.done", {"output_index": idx, "item": item})


def _completed(output=None, inp=100, cached=40, out=7, status="completed"):
    return ("response.completed", {"response": {"status": status, "output": output or [],
                                                "usage": {"input_tokens": inp, "output_tokens": out,
                                                          "input_tokens_details": {"cached_tokens": cached}}}})


class StreamParserTests(unittest.TestCase):
    def parser(self, names=None, tag="openai_api"):
        return rd.StreamParser(tag, names, "p", "m", context_window=1000, est_tokens=1200)

    def test_reasoning_text_usage(self):
        items = [
            ("response.created", {"response": {"status": "in_progress"}}),
            _added(0, {"type": "reasoning", "id": "rs_0", "summary": []}),
            ("response.reasoning_summary_text.delta", {"output_index": 0, "summary_index": 0, "delta": "Plan"}),
            ("response.reasoning_summary_text.delta", {"output_index": 0, "summary_index": 0, "delta": " A"}),
            ("response.reasoning_summary_text.delta", {"output_index": 0, "summary_index": 1, "delta": "Then B"}),
            _done(0, {"type": "reasoning", "id": "rs_0", "encrypted_content": "E0",
                      "summary": [{"type": "summary_text", "text": "Plan A"},
                                  {"type": "summary_text", "text": "Then B"}]}),
            _added(1, {"type": "reasoning", "id": "rs_1", "summary": []}),
            _done(1, {"type": "reasoning", "id": "rs_1", "encrypted_content": "E1",
                      "summary": [{"type": "summary_text", "text": "Only at done"}]}),
            _done(2, {"type": "reasoning", "id": "rs_2", "encrypted_content": "E2", "summary": []}),
            _added(3, {"type": "message", "id": "m", "role": "assistant", "content": []}),
            ("response.output_text.delta", {"output_index": 3, "content_index": 0, "delta": "Hel"}),
            ("response.output_text.delta", {"output_index": 3, "content_index": 0, "delta": "lo"}),
            _done(3, {"type": "message", "id": "m", "content": [{"type": "output_text", "text": "Hello"}]}),
            ("response.unknown_future_event", {"x": 1}),
            _completed(),
        ]
        out = feed_all(self.parser(), items)
        sig = lambda enc, summ: sigs.encode_signature("openai_api", {"enc": enc, "summary": summ})  # noqa: E731
        self.assertEqual(out, [
            ev.ThinkingDelta(0, ""), ev.ThinkingDelta(0, "Plan"), ev.ThinkingDelta(0, " A"),
            ev.ThinkingDelta(0, "\n\nThen B"), ev.ThinkingSignature(0, sig("E0", ["Plan A", "Then B"])),
            ev.ThinkingDelta(1, ""), ev.ThinkingDelta(1, "Only at done"),
            ev.ThinkingSignature(1, sig("E1", ["Only at done"])),
            ev.ThinkingDelta(2, ""), ev.ThinkingSignature(2, sig("E2", [])),
            ev.TextDelta(3, "Hel"), ev.TextDelta(3, "lo"),
            ev.Usage(60, 7, 40), ev.Finish("end_turn"),
        ])
        self.assertEqual(sigs.decode_signature(out[4].signature), ("openai_api", {"enc": "E0",
                                                                                 "summary": ["Plan A", "Then B"]}))

    def test_parallel_function_calls(self):
        names = toolnames.ToolNameMap(["Bash", LONG_TOOL])
        short = names.upstream(LONG_TOOL)
        items = [
            _added(0, {"type": "function_call", "id": "fc0", "call_id": "call_0", "name": "Bash", "arguments": ""}),
            _added(1, {"type": "function_call", "id": "fc1", "call_id": "call|1", "name": short, "arguments": ""}),
            ("response.function_call_arguments.delta", {"output_index": 0, "delta": '{"comm'}),
            ("response.function_call_arguments.delta", {"output_index": 1, "delta": '{"q":'}),
            ("response.function_call_arguments.delta", {"output_index": 0, "delta": 'and":"ls"}'}),
            ("response.function_call_arguments.delta", {"output_index": 1, "delta": '2}'}),
            _done(1, {"type": "function_call", "id": "fc1", "call_id": "call|1", "name": short,
                      "arguments": '{"q":2}'}),
            _done(0, {"type": "function_call", "id": "fc0", "call_id": "call_0", "name": "Bash",
                      "arguments": '{"command":"ls"}'}),
            _completed(inp=10, cached=0, out=3),
        ]
        out = feed_all(self.parser(names), items)
        self.assertEqual(out, [ev.ToolCall(toolnames.encode_tool_id("call|1"), LONG_TOOL, '{"q":2}'),
                               ev.ToolCall("call_0", "Bash", '{"command":"ls"}'),
                               ev.Usage(10, 3, 0), ev.Finish("tool_use")])
        self.assertEqual(toolnames.decode_tool_id(out[0].id), "call|1")

    def test_tool_call_fallbacks(self):
        items = [
            _added(0, {"type": "function_call", "call_id": "c0", "name": "Bash", "arguments": ""}),
            ("response.function_call_arguments.delta", {"output_index": 0, "delta": '{"a":1}'}),
            _done(0, {"type": "function_call", "call_id": "c0", "name": "Bash"}),  # no arguments in done
            _done(1, {"type": "function_call", "call_id": "", "name": "Bash", "arguments": "not json"}),
            _done(2, {"type": "function_call", "call_id": "c2", "name": "Bash", "arguments": ""}),
            _completed(),
        ]
        out = feed_all(self.parser(), items)
        calls = [e for e in out if isinstance(e, ev.ToolCall)]
        self.assertEqual(calls[0], ev.ToolCall("c0", "Bash", '{"a":1}'))
        self.assertTrue(calls[1].id.startswith("toolu_"))
        self.assertEqual(json.loads(calls[1].input_json), {"_raw_arguments": "not json"})
        self.assertEqual(calls[2].input_json, "{}")

    def test_completed_only_output(self):
        output = [
            {"type": "reasoning", "id": "r", "encrypted_content": "E", "summary": [{"type": "summary_text",
                                                                                     "text": "S"}]},
            {"type": "message", "id": "m", "content": [{"type": "output_text", "text": "Hi"},
                                                       {"type": "refusal", "refusal": "!"}]},
            {"type": "function_call", "id": "f", "call_id": "c", "name": "Bash", "arguments": "{}"},
        ]
        out = feed_all(self.parser(tag="xai_api"), [_completed(output, inp=5, cached=9, out=1)])
        self.assertEqual(out[:2], [ev.ThinkingDelta(0, "S"),
                                   ev.ThinkingSignature(0, sigs.encode_signature("xai_api", {"enc": "E",
                                                                                             "summary": ["S"]}))])
        self.assertEqual(out[2:], [ev.TextDelta(1, "Hi!"), ev.ToolCall("c", "Bash", "{}"), ev.Usage(0, 1, 5),
                                   ev.Finish("tool_use")])

    def test_streamed_items_not_duplicated_by_completed_output(self):
        msg = {"type": "message", "id": "m", "content": [{"type": "output_text", "text": "Hi"}]}
        items = [_added(0, msg), ("response.output_text.delta", {"output_index": 0, "delta": "Hi"}), _done(0, msg),
                 _completed([msg])]
        self.assertEqual(text_of(feed_all(self.parser(), items)), "Hi")

    def test_refusal_and_reasoning_text(self):
        items = [("response.reasoning_text.delta", {"output_index": 0, "content_index": 0, "delta": "raw"}),
                 ("response.refusal.delta", {"output_index": 1, "delta": "no"}), _completed()]
        out = feed_all(self.parser(), items)
        self.assertEqual(out[:3], [ev.ThinkingDelta(0, ""), ev.ThinkingDelta(0, "raw"), ev.TextDelta(1, "no")])

    def test_incomplete(self):
        base = {"status": "incomplete", "usage": {"input_tokens": 3, "output_tokens": 2}}
        out = feed_all(self.parser(), [("response.incomplete", {"response": dict(
            base, incomplete_details={"reason": "max_output_tokens"})})])
        self.assertEqual(out, [ev.Usage(3, 2, 0), ev.Finish("max_tokens")])
        out = feed_all(self.parser(), [
            _added(0, {"type": "function_call", "call_id": "c", "name": "Bash"}),
            ("response.function_call_arguments.delta", {"output_index": 0, "delta": '{"comm'}),
            ("response.incomplete", {"response": dict(base, incomplete_details={"reason": "content_filter"})})])
        self.assertEqual(out, [ev.Usage(3, 2, 0), ev.TextDelta("incomplete", "[response incomplete: content_filter]"),
                               ev.Finish("end_turn")])

    def failed(self, err, etype="response.failed"):
        data = {"response": {"status": "failed", "error": err}} if etype == "response.failed" else err
        with self.assertRaises(errors.GatewayError) as cm:
            feed_all(self.parser(), [("response.created", {"response": {}}), (etype, data)])
        return cm.exception

    def test_failed_mapping(self):
        e = self.failed({"code": "context_length_exceeded",
                         "message": "Your input exceeds the context window of this model."})
        self.assertEqual((e.status, e.message), (400, "prompt is too long: 1200 tokens > 1000 maximum"))
        e = self.failed({"code": "rate_limit_exceeded", "message": "Rate limit reached"})
        self.assertEqual((e.status, e.err_type, e.should_retry), (429, "rate_limit_error", True))
        e = self.failed({"code": "server_error", "message": "boom"})
        self.assertEqual((e.status, e.err_type, e.should_retry), (529, "overloaded_error", True))
        e = self.failed({"code": "usage_limit_reached", "message": "The usage limit has been reached",
                         "resets_at": 1900000000})
        self.assertEqual((e.status, e.should_retry), (429, False))
        self.assertIn("resets at", e.message)

    def test_error_events(self):
        e = self.failed({"code": "rate_limit_exceeded", "message": "slow down", "param": None}, etype="error")
        self.assertEqual(e.status, 429)
        e = self.failed({"error": {"type": "invalid_request_error", "code": "invalid_value", "message": "bad"}},
                        etype="error")
        self.assertEqual((e.status, e.err_type), (400, "invalid_request_error"))

    def test_truncated_stream(self):
        p = self.parser()
        p.feed("response.output_text.delta", {"type": "response.output_text.delta", "output_index": 0, "delta": "x"})
        with self.assertRaises(errors.GatewayError) as cm:
            p.end()
        self.assertTrue(cm.exception.connection_error)
        self.assertTrue(cm.exception.should_retry)

    def test_garbage_is_ignored(self):
        p = self.parser()
        self.assertEqual(p.feed("x", "not a dict"), [])
        self.assertEqual(p.feed("response.output_item.done", {"type": "response.output_item.done", "item": 5}), [])
        self.assertEqual(p.feed("response.output_text.delta", {"type": "response.output_text.delta", "delta": None}),
                         [])
        out = feed_all(p, [("response.completed", {"response": {"usage": {"input_tokens": "x"}}})])
        self.assertEqual(out, [ev.Usage(0, 0, 0), ev.Finish("end_turn")])
        self.assertEqual(p.feed("response.output_text.delta", {"output_index": 0, "delta": "late"}), [])


# =========================================================================================
# end to end against the mock kinds
# =========================================================================================

class MockRoundTripTests(unittest.TestCase):
    def round_trip(self, kind, prov, model_id, auth=None, base_path="/v1", options=None):
        with mocks.MockServer(kind, options=options) as srv:
            prov.base_url = srv.url + base_path
            runtime = dbase.ProviderRuntime(prov.id)
            req1 = turn1()
            evs1 = run(ctx_for(prov, model_id, req1, auth, runtime))
            calls = [e for e in evs1 if isinstance(e, ev.ToolCall)]
            self.assertEqual(len(calls), 1, evs1)
            self.assertEqual(calls[0].name, "Bash")
            self.assertEqual(json.loads(calls[0].input_json)["command"], mocks.DEFAULT_COMMAND)
            self.assertEqual(evs1[-1], ev.Finish("tool_use"))
            assistant = assistant_from_events(evs1)
            # foreign-target thinking in history must be dropped (the mock would reject unknown enc)
            foreign_tag = "openai_api" if rt.get_target(prov.target).sig_tag != "openai_api" else "xai_api"
            assistant.blocks.insert(0, Block.of_thinking("old", sigs.encode_signature(
                foreign_tag, {"enc": "enc-from-elsewhere", "summary": ["old"]})))
            msgs = req1.messages + [assistant, Message("user", [Block.of_tool_result(calls[0].id, "hello")])]
            evs2 = run(ctx_for(prov, model_id, turn1(messages=msgs), auth, runtime))
            self.assertEqual(text_of(evs2), "DONE " + mocks.sha8("hello"))
            self.assertEqual(evs2[-1], ev.Finish("end_turn"))
            usage = [e for e in evs2 if isinstance(e, ev.Usage)][-1]
            self.assertGreater(usage.cache_read, 0)
            self.assertEqual(srv.errors, [])
            posts = srv.requests_for("", "POST")
            self.assertEqual(len(posts), 2)
            return srv, evs1, evs2, posts

    def test_openai_responses(self):
        prov = provider("openai_api", auth={"kind": "api_key", "secret": "k"})
        srv, evs1, _, posts = self.round_trip("openai_responses", prov, "gpt-5.5", static_auth(),
                                              options={"api_key": "test-key"})
        thinking = [e for e in evs1 if isinstance(e, ev.ThinkingDelta)]
        self.assertEqual("".join(e.text for e in thinking), "Planning the tool call.\n\nChecking the constraints.")
        enc_items = [i for i in posts[1]["body_json"]["input"] if i["type"] == "reasoning"]
        self.assertEqual(len(enc_items), 1)
        self.assertTrue(enc_items[0]["encrypted_content"].startswith("enc_openai_responses_"))
        self.assertEqual(posts[0]["path"], "/v1/responses")

    def test_chatgpt_codex(self):
        prov = provider("chatgpt_codex", auth={"kind": "codex_chatgpt"}, max_tool_name=64)
        _, _, _, posts = self.round_trip("chatgpt_codex", prov, "gpt-5.5", FakeCodexAuth(),
                                         base_path="/backend-api/codex", options={"account_id": "acct-123"})
        h = posts[0]["headers"]
        self.assertEqual(posts[0]["path"], "/backend-api/codex/responses")
        self.assertEqual((h["originator"], h["version"], h["session-id"]), ("codex_cli_rs", "0.160.0", "sess-1"))
        self.assertTrue(h["User-Agent"].startswith("codex_cli_rs/0.160.0 ("))
        for name in [t["name"] for t in posts[0]["body_json"]["tools"]]:
            self.assertLessEqual(len(name), 64)

    def test_chatgpt_codex_lite(self):
        prov = provider("chatgpt_codex", auth={"kind": "codex_chatgpt"},
                        models=[config.ModelSpec("gpt-6.1", reasoning=True)])
        _, _, _, posts = self.round_trip("chatgpt_codex", prov, "gpt-6.1", FakeCodexAuth(),
                                         base_path="/backend-api/codex")
        self.assertEqual(posts[0]["headers"]["x-openai-internal-codex-responses-lite"], "true")
        self.assertNotIn("instructions", posts[1]["body_json"])

    def test_grok_proxy(self):
        prov = provider("grok_cli_proxy", schema_mode="no_root_combinators")
        _, _, _, posts = self.round_trip("grok_proxy", prov, "grok-4.7", static_auth())
        self.assertEqual(posts[0]["headers"]["X-XAI-Token-Auth"], "xai-grok-cli")
        self.assertNotEqual(posts[0]["headers"]["x-grok-req-id"], posts[1]["headers"]["x-grok-req-id"])

    def test_xai_responses(self):
        prov = provider("xai_api", models=[config.ModelSpec("grok-4.20-multi-agent-0309", reasoning=True,
                                                            max_output=64000)])
        _, _, _, posts = self.round_trip("xai_responses", prov, "grok-4.20-multi-agent-0309", static_auth())
        self.assertEqual(posts[0]["body_json"]["max_output_tokens"], 32000)

    def test_non_reasoning_model_has_no_thinking(self):
        prov = provider("openai_api", models=[config.ModelSpec("gpt-4.1")])
        with mocks.MockServer("openai_responses") as srv:
            prov.base_url = srv.url + "/v1"
            evs = run(ctx_for(prov, "gpt-4.1", turn1(), static_auth()))
        self.assertFalse([e for e in evs if isinstance(e, (ev.ThinkingDelta, ev.ThinkingSignature))])
        self.assertEqual(evs[-1], ev.Finish("tool_use"))

    def test_background_request(self):
        prov = provider("chatgpt_codex")
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        req = turn1(tools=[], output_format={"type": "json_schema", "schema": schema})
        with mocks.MockServer("chatgpt_codex") as srv:
            prov.base_url = srv.url
            evs = run(ctx_for(prov, "gpt-5.4-mini", req, FakeCodexAuth(), background=True))
            body = srv.last_request()["body_json"]
        self.assertEqual(text_of(evs), "Background reply.")
        self.assertEqual(body["reasoning"]["effort"], "low")
        self.assertEqual(body["text"]["format"]["schema"], schema)


class MockQuirkTests(unittest.TestCase):
    """The mocks themselves enforce the documented quirks (so a regression in the dialect is caught)."""

    def post(self, kind, body, headers=None, options=None, path="/v1/responses"):
        with mocks.MockServer(kind, options=options) as srv:
            h = {"Content-Type": "application/json", "Authorization": "Bearer " + jwt(compat.utcnow_epoch() + 600)}
            h.update(headers or {})
            r = transport.HttpClient(timeout=10, environ={}).request("POST", srv.url + path, h,
                                                                    json.dumps(body), stream=False)
            self.assertEqual(srv.errors, [])
            return r.status, r.json() if r.status >= 400 else r.text()

    def codex_headers(self):
        return {"ChatGPT-Account-ID": "a", "originator": "codex_cli_rs", "version": "0.160.0", "session-id": "s",
                "User-Agent": "codex_cli_rs/0.160.0 (Linux 6; x86_64) xterm", "Accept": "text/event-stream"}

    def base_body(self, **kw):
        body = {"model": "gpt-5.5", "instructions": "x", "input": [], "store": False, "stream": True}
        body.update(kw)
        return body

    def test_codex_rejections(self):
        for key in ("max_output_tokens", "temperature", "top_p", "truncation", "user", "previous_response_id"):
            st, body = self.post("chatgpt_codex", self.base_body(**{key: 1}), self.codex_headers())
            self.assertEqual((st, body), (400, {"detail": "Unsupported parameter: %s" % key}))
        st, body = self.post("chatgpt_codex", self.base_body(store=True), self.codex_headers())
        self.assertEqual(body["detail"], "Store must be set to false")
        st, body = self.post("chatgpt_codex", self.base_body(instructions=""), self.codex_headers())
        self.assertEqual(body["detail"], "Instructions are required")
        st, body = self.post("chatgpt_codex", self.base_body(), dict(self.codex_headers(), originator=""))
        self.assertEqual(st, 400)
        st, body = self.post("chatgpt_codex", self.base_body(), dict(self.codex_headers(),
                                                                     Authorization="Bearer " + jwt(1)))
        self.assertEqual((st, body["error"]["code"]), (401, "token_expired"))
        st, body = self.post("chatgpt_codex", self.base_body(), self.codex_headers(), options={"usage_limit": True})
        self.assertEqual((st, body["error"]["type"]), (429, "usage_limit_reached"))
        self.assertIn("resets_at", body["error"])
        bad = self.base_body(input=[{"type": "reasoning", "summary": [], "encrypted_content": "forged"}])
        st, body = self.post("chatgpt_codex", bad, self.codex_headers())
        self.assertIn("could not be verified", body["detail"])
        orphan = self.base_body(input=[{"type": "function_call_output", "call_id": "c", "output": "x"}])
        st, body = self.post("chatgpt_codex", orphan, self.codex_headers())
        self.assertIn("No tool call found", body["detail"])
        st, _ = self.post("chatgpt_codex", self.base_body(), self.codex_headers())
        self.assertEqual(st, 200)

    def test_openai_rejections(self):
        st, body = self.post("openai_responses", self.base_body(max_tokens=10))
        self.assertEqual((st, body["error"]["code"]), (400, "unsupported_parameter"))
        st, body = self.post("openai_responses", self.base_body(max_output_tokens=1))
        self.assertEqual(body["error"]["code"], "integer_below_min_value")
        st, body = self.post("openai_responses", self.base_body(temperature=0.5))
        self.assertEqual(body["error"]["param"], "temperature")
        st, body = self.post("openai_responses", self.base_body(store=True, include=["reasoning.encrypted_content"]))
        self.assertEqual(st, 400)
        st, _ = self.post("openai_responses", self.base_body(), options={"api_key": "other"})
        self.assertEqual(st, 401)

    def test_grok_rejections(self):
        hdrs = rt.get_target("grok_cli_proxy").headers("s", environ={"AI_GATEWAY_GROK_CLIENT_VERSION": "1.0.0"})
        tool = {"type": "function", "name": "t", "parameters": {"anyOf": [{"type": "object"}]}}
        st, body = self.post("grok_proxy", self.base_body(tools=[tool]), hdrs)
        self.assertEqual(st, 400)
        self.assertIn("root-level oneOf/anyOf/allOf", body["error"])
        st, body = self.post("grok_proxy", self.base_body(max_output_tokens=5), hdrs)
        self.assertEqual(body["error"], "Argument not supported: max_output_tokens")
        st, body = self.post("grok_proxy", self.base_body(), dict(hdrs, **{"x-grok-client-mode": "tty"}))
        self.assertEqual(st, 400)
        st, body = self.post("grok_proxy", self.base_body(), hdrs, options={"version_gate": True})
        self.assertEqual(st, 426)
        self.assertIn("upgrade", body["error"])
        st, body = self.post("grok_proxy", self.base_body(), hdrs,
                             options={"version_gate": "2.0.0", "version_gate_status": 400})
        self.assertEqual(st, 400)
        st, _ = self.post("grok_proxy", self.base_body(), hdrs, options={"version_gate": "0.9.0"})
        self.assertEqual(st, 200)
        st, _ = self.post("xai_responses", self.base_body(max_tokens=3))
        self.assertEqual(st, 400)

    def test_models_listing(self):
        with mocks.MockServer("openai_responses") as srv:
            r = transport.HttpClient(timeout=10, environ={}).request(
                "GET", srv.url + "/v1/models", {"Authorization": "Bearer k"}, stream=False)
            self.assertIn("gpt-5.5", [m["id"] for m in r.json()["data"]])


# =========================================================================================
# auth, errors and pre/post-commit behaviour through execute()
# =========================================================================================

class ExecuteErrorTests(unittest.TestCase):
    def codex(self, srv):
        return provider("chatgpt_codex", srv.url + "/backend-api/codex")

    def test_401_refresh_and_retry(self):
        with mocks.MockServer("chatgpt_codex") as srv:
            auth = FakeCodexAuth(expired_first=True)
            evs = run(ctx_for(self.codex(srv), "gpt-5.5", turn1(), auth))
            self.assertEqual(evs[-1], ev.Finish("tool_use"))
            self.assertEqual(auth.refreshes, 1)
            posts = srv.requests_for("/backend-api", "POST")
            self.assertEqual(len(posts), 2)
            self.assertNotEqual(posts[0]["headers"]["Authorization"], posts[1]["headers"]["Authorization"])
            self.assertEqual(posts[1]["headers"]["ChatGPT-Account-ID"], "acct-123")

    def test_403_token_expired_refresh(self):
        with mocks.MockServer("chatgpt_codex", options={"expired_status": 403}) as srv:
            auth = FakeCodexAuth(expired_first=True)
            evs = run(ctx_for(self.codex(srv), "gpt-5.5", turn1(), auth))
            self.assertEqual(evs[-1], ev.Finish("tool_use"))
            self.assertEqual(auth.refreshes, 1)

    def test_usage_limit_is_terminal(self):
        with mocks.MockServer("chatgpt_codex", options={"usage_limit": True}) as srv:
            with self.assertRaises(errors.GatewayError) as cm:
                run(ctx_for(self.codex(srv), "gpt-5.5", turn1(), FakeCodexAuth()))
        e = cm.exception
        self.assertEqual((e.status, e.err_type, e.should_retry), (429, "rate_limit_error", False))
        self.assertIn("resets at", e.message)

    def test_failed_before_first_event_raises(self):
        opts = {"fail": {"code": "context_length_exceeded", "message": "Your input exceeds the context window"}}
        with mocks.MockServer("openai_responses", options=opts) as srv:
            p = provider("openai_api", srv.url + "/v1", [config.ModelSpec("gpt-5.5", context=1000, reasoning=True)])
            with self.assertRaises(errors.GatewayError) as cm:
                run(ctx_for(p, "gpt-5.5", turn1(), static_auth(), est=5000))
        self.assertEqual(cm.exception.message, "prompt is too long: 5000 tokens > 1000 maximum")

    def test_error_after_commit_becomes_stream_error(self):
        opts = {"error_event": {"code": "server_error", "message": "boom"}}
        with mocks.MockServer("openai_responses", options=opts) as srv:
            p = provider("openai_api", srv.url + "/v1")
            evs = run(ctx_for(p, "gpt-5.5", turn1(), static_auth()))
        self.assertIsInstance(evs[0], ev.ThinkingDelta)
        self.assertEqual(evs[-1].kind, "error")
        self.assertEqual((evs[-1].err_type, evs[-1].retryable), ("overloaded_error", True))

    def test_truncated_stream(self):
        with mocks.MockServer("openai_responses", options={"truncate": True}) as srv:
            p = provider("openai_api", srv.url + "/v1")
            evs = run(ctx_for(p, "gpt-5.5", turn1(), static_auth()))
            self.assertEqual((evs[-1].kind, evs[-1].retryable), ("error", True))
            p = provider("openai_api", srv.url + "/v1", [config.ModelSpec("gpt-4.1")])
            with self.assertRaises(errors.GatewayError) as cm:
                run(ctx_for(p, "gpt-4.1", turn1(), static_auth()))  # no reasoning -> nothing yielded yet
            self.assertTrue(cm.exception.connection_error)

    def test_incomplete_via_mock(self):
        with mocks.MockServer("xai_responses", options={"incomplete": "max_output_tokens"}) as srv:
            evs = run(ctx_for(provider("xai_api", srv.url + "/v1"), "grok-4.7", turn1(), static_auth()))
        self.assertEqual(evs[-1], ev.Finish("max_tokens"))

    def test_generator_close_releases_upstream(self):
        with mocks.MockServer("openai_responses") as srv:
            gen = rd.ResponsesDialect().execute(ctx_for(provider("openai_api", srv.url + "/v1"), "gpt-5.5", turn1(),
                                                        static_auth()))
            self.assertIsInstance(next(gen), ev.ThinkingDelta)
            gen.close()
            self.assertEqual(srv.errors, [])

    def test_registry(self):
        self.assertIsInstance(dialects.get_dialect("responses"), rd.ResponsesDialect)


# =========================================================================================
# grok sticky fallback
# =========================================================================================

def _chat_mock(server, req, resp):
    """Minimal OpenAI chat-completions upstream (used only when agent C's xai_chat kind is unavailable)."""
    body = req.json or {}
    tools = [t["function"]["name"] for t in body.get("tools") or []]
    results = [m.get("content") or "" for m in body.get("messages") or [] if m.get("role") == "tool"]
    reply = server.brain.decide(tools, results, background=not tools)
    w = resp.start_sse()
    if reply.kind == "tool_call":
        delta = {"tool_calls": [{"index": 0, "id": "call_chat1", "type": "function",
                                 "function": {"name": reply.tool_name, "arguments": json.dumps(reply.arguments)}}]}
        finish = "tool_calls"
    else:
        delta, finish = {"content": reply.text}, "stop"
    w.data({"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta}]})
    w.data({"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {},
                                                                       "finish_reason": finish}]})
    w.data({"id": "c", "object": "chat.completion.chunk", "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
    w.done()
    w.close()


class GrokFallbackTests(unittest.TestCase):
    def grok(self, proxy_url, fallback):
        p = config.ProviderSpec.from_dict("grok", {
            "dialect": "responses", "target": "grok_cli_proxy", "base_url": proxy_url, "auth": {"kind": "grok_cli"},
            "allow_unlisted": True, "schema_mode": "no_root_combinators", "fallback": fallback})
        return p

    def run_fallback(self, proxy_options, expect_fallback=True):
        with mocks.MockServer("grok_proxy", options=proxy_options) as proxy, \
                mocks.MockServer("xai_responses") as xai:
            p = self.grok(proxy.url + "/v1", {"dialect": "responses", "target": "xai_api",
                                              "base_url": xai.url + "/v1"})
            runtime = dbase.ProviderRuntime("grok")
            auth = static_auth("grok", "grok-login-token")
            if not expect_fallback:
                with self.assertRaises(errors.GatewayError) as cm:
                    run(ctx_for(p, "grok-4.7", turn1(), auth, runtime))
                self.assertFalse(runtime.get(rd.FALLBACK_FLAG))
                return cm.exception, proxy, xai
            with self.assertLogs("ai_gateway", level="WARNING") as logs:
                evs = run(ctx_for(p, "grok-4.7", turn1(), auth, runtime))
                evs2 = run(ctx_for(p, "grok-4.7", turn1(), auth, runtime))
            self.assertEqual(evs[-1], ev.Finish("tool_use"))
            self.assertEqual(evs2[-1], ev.Finish("tool_use"))
            self.assertTrue(runtime.get(rd.FALLBACK_FLAG))
            self.assertEqual(len([m for m in logs.output if "grok CLI proxy unavailable" in m]), 1)
            self.assertEqual(len(proxy.requests_for("/v1/responses", "POST")), 1)  # sticky: proxy tried once
            xai_posts = xai.requests_for("/v1/responses", "POST")
            self.assertEqual(len(xai_posts), 2)
            self.assertEqual(xai_posts[0]["headers"]["Authorization"], "Bearer grok-login-token")
            self.assertNotIn("X-XAI-Token-Auth", xai_posts[0]["headers"])
            return evs, proxy, xai

    def test_version_gate_426(self):
        self.run_fallback({"version_gate": True})

    def test_version_gate_400_message(self):
        self.run_fallback({"version_gate": "99.0.0", "version_gate_status": 400})

    def test_5xx_and_404_and_410(self):
        for status in (502, 404, 410):
            self.run_fallback({"http_error": (status, {"code": "x", "error": "gone"})})

    def test_403_client_message(self):
        self.run_fallback({"http_error": (403, {"code": "permission-denied", "error": "unsupported client"})})

    def test_non_qualifying_errors_do_not_switch(self):
        for status, body in ((429, {"code": "rate", "error": "too many requests"}),
                             (400, {"code": "invalid-argument", "error": "Argument not supported: foo"}),
                             (403, {"code": "permission-denied", "error": "forbidden"})):
            exc, proxy, xai = self.run_fallback({"http_error": (status, body)}, expect_fallback=False)
            self.assertEqual(xai.requests_for("", "POST"), [])
        exc, _, _ = self.run_fallback({"http_error": (401, {"code": "unauth", "error": "bad token"})},
                                      expect_fallback=False)
        self.assertEqual(exc.status, 401)

    def test_connection_error(self):
        with mocks.MockServer("xai_responses") as xai:
            p = self.grok("http://127.0.0.1:%d/v1" % free_port(), {"dialect": "responses", "target": "xai_api",
                                                                    "base_url": xai.url + "/v1"})
            runtime = dbase.ProviderRuntime("grok")
            with self.assertLogs("ai_gateway", level="WARNING"):
                evs = run(ctx_for(p, "grok-4.7", turn1(), static_auth(), runtime))
            self.assertEqual(evs[-1], ev.Finish("tool_use"))
            self.assertTrue(runtime.get(rd.FALLBACK_FLAG))

    def test_proxy_success_keeps_primary(self):
        with mocks.MockServer("grok_proxy") as proxy, mocks.MockServer("xai_responses") as xai:
            p = self.grok(proxy.url + "/v1", {"dialect": "responses", "target": "xai_api", "base_url": xai.url})
            runtime = dbase.ProviderRuntime("grok")
            evs = run(ctx_for(p, "grok-4.7", turn1(), static_auth(), runtime))
            self.assertEqual(evs[-1], ev.Finish("tool_use"))
            self.assertFalse(runtime.get(rd.FALLBACK_FLAG))
            self.assertEqual(xai.requests, [])

    def test_fallback_to_openai_chat(self):
        try:
            mod("dialects.openai_chat").OpenAIChatDialect
        except (ImportError, AttributeError) as exc:
            self.skipTest("openai_chat dialect not available yet: %s" % exc)
        try:
            mocks.get_kind_factory("xai_chat")
            chat = mocks.MockServer("xai_chat", options={"api_key": "test-key"})
        except KeyError:
            chat = mocks.MockServer(handler=_chat_mock)
        with mocks.MockServer("grok_proxy", options={"version_gate": True}) as proxy, chat:
            p = self.grok(proxy.url + "/v1", {"dialect": "openai_chat", "profile": "xai",
                                              "base_url": chat.url + "/v1", "schema_mode": "no_root_combinators"})
            runtime = dbase.ProviderRuntime("grok")
            with self.assertLogs("ai_gateway", level="WARNING"):
                evs1 = run(ctx_for(p, "grok-4.7", turn1(), static_auth(), runtime))
            calls = [e for e in evs1 if isinstance(e, ev.ToolCall)]
            self.assertEqual([c.name for c in calls], ["Bash"])
            msgs = turn1().messages + [assistant_from_events(evs1),
                                       Message("user", [Block.of_tool_result(calls[0].id, "hello")])]
            evs2 = run(ctx_for(p, "grok-4.7", turn1(messages=msgs), static_auth(), runtime))
            self.assertEqual(text_of(evs2), "DONE " + mocks.sha8("hello"))
            self.assertEqual(len(proxy.requests_for("/v1/responses", "POST")), 1)
            self.assertEqual(len(chat.requests_for("/v1/chat/completions", "POST")), 2)


if __name__ == "__main__":
    unittest.main()
