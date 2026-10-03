"""Gemini dialect (DESIGN §3.3): request goldens, stream parsing, execute() against the
``gemini_api`` / ``vertex`` mocks (two turns with thought-signature echo), sticky schema fallback,
429 classification and 401 refresh."""

import copy
import json
import logging
import re
import unittest

from ._pkg import mod

ev = mod("events")
md = mod("model")
config = mod("config")
errors = mod("errors")
sigs = mod("signatures")
transport = mod("transport")
presets = mod("presets")
auth_base = mod("auth.base")
dbase = mod("dialects.base")
dialects = mod("dialects")
gem = mod("dialects.gemini")
mocks = mod("testing.mock_upstreams")
mg = mod("testing.mock_gemini")
toolnames = mod("toolnames")

B = md.Block
M = md.Message
G3 = "gemini-3.1-pro-preview"
G25_PRO = "gemini-2.5-pro"
G25_FLASH = "gemini-2.5-flash"
NAME_RE = re.compile(presets.GEMINI_TOOL_NAME_REGEX)
LONG_MCP = "mcp__claude_in_chrome_extension_server__take_full_page_screenshot_and_annotate_dom_nodes"
BASH = md.ToolDef("Bash", "Run a shell command", {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
    "properties": {"command": {"type": "string", "description": "the command"},
                   "timeout": {"type": "number", "maximum": 600000, "exclusiveMinimum": 0},
                   "description": {"type": "string"}},
    "required": ["command"], "additionalProperties": False})
READ = md.ToolDef("Read", "Read a file", {"type": "object", "properties": {"file_path": {"type": "string"}},
                                          "required": ["file_path"]})
REFS = md.ToolDef(LONG_MCP, "Screenshot", {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
    "properties": {"node": {"$ref": "#/$defs/node"}, "mode": {"type": ["string", "null"], "enum": ["a", "b"]}},
    "$defs": {"node": {"type": "object", "properties": {"selector": {"type": "string"}}}}})
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
PDF = "JVBERi0xLjQKJcfsj6IKMSAwIG9iago8PD4+CmVuZG9iagp0cmFpbGVyCjw8Pj4KJSVFT0YK"


def gsig(s, target="gemini_api"):
    return sigs.encode_signature(target, {"s": s})


class _Res(object):
    def __init__(self, provider, model_id, spec, background=False):
        self.provider, self.model, self.model_spec = provider, model_id, spec
        self.requested, self.role, self.background = "claude-via-%s,%s" % (provider.id, model_id), None, background


class FakeAuth(auth_base.AuthProvider):
    """API-key (``style="key"``) or Bearer+quota-project (``style="bearer"``) auth with scripted refreshes."""

    kind = "fake"

    def __init__(self, values=("k1",), style="key", project="proj-1", location="global"):
        auth_base.AuthProvider.__init__(self, "fake")
        self.values, self.i, self.style = list(values), 0, style
        self._project, self._location = project, location
        self.refreshes = 0

    def available(self):
        return True

    def headers(self, force_refresh=False):
        v = self.values[self.i]
        if self.style == "key":
            return {"x-goog-api-key": v}
        return {"Authorization": "Bearer " + v, "x-goog-user-project": self._project}

    def on_unauthorized(self, used):
        if self.i + 1 < len(self.values):
            self.i += 1
            self.refreshes += 1
            return True
        return False

    def describe(self):
        return {"kind": self.kind, "style": self.style}

    def relogin_hint(self):
        return "run `gcloud auth application-default login`"

    def project(self):
        if self._project is None:
            raise auth_base.AuthError("no Google Cloud project", "set GOOGLE_CLOUD_PROJECT")
        return self._project

    def location(self):
        return self._location


def provider(preset="gemini", base_url=None, **over):
    ov = dict(over)
    if base_url is not None:
        ov["base_url"] = base_url
    return presets.provider_from_preset(preset, ov)


def make_ctx(req, prov=None, model_id=G3, auth=None, runtime=None, session_id="sess-1"):
    prov = prov or provider()
    spec = prov.model_spec(model_id)
    return dbase.RequestContext(req, _Res(prov, model_id, spec), prov, spec,
                                runtime or dbase.ProviderRuntime(prov.id), auth or FakeAuth(),
                                transport.HttpClient(timeout=20, environ={}), session_id, est_tokens=50,
                                requested_model="claude-via-%s,%s" % (prov.id, model_id))


def make_req(messages, tools=None, **kw):
    kw.setdefault("effort", "high")
    kw.setdefault("thinking_requested", True)
    return md.NormalizedRequest(model="claude-via-gemini," + G3, system=kw.pop("system", ["You are Claude Code."]),
                                messages=messages, tools=list(tools or []), **kw)


def user(*blocks):
    return M("user", [B.of_text(b) if isinstance(b, str) else b for b in blocks])


def assistant(*blocks):
    return M("assistant", [B.of_text(b) if isinstance(b, str) else b for b in blocks])


def body_for(req, model_id=G3, prov=None, runtime=None, fallback=False):
    ctx = make_ctx(req, prov, model_id, runtime=runtime)
    return gem.GeminiDialect().build_body(ctx, fallback)[0]


def structurally_valid(test, body, model_id=G3, options=None):
    """Run the mock's validator (signatures not required unless asked)."""
    opts = {"require_signatures": False}
    opts.update(options or {})
    try:
        mg.validate_request(copy.deepcopy(body), model_id, opts, {"issued": {}, "sig_checks": []})
    except Exception as exc:  # noqa: BLE001 - surface the mock's 400 message
        test.fail("mock rejected body: %s\n%s" % (getattr(exc, "message", exc), json.dumps(body, indent=1)[:3000]))


def walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            for x in walk_keys(v):
                yield x
    elif isinstance(obj, list):
        for v in obj:
            for x in walk_keys(v):
                yield x


def assistant_from_events(events):
    """What Claude Code echoes next turn: one block per key (thinking gets the emitted signature or the
    SSEEmitter's synthetic one), tool calls as tool_use blocks."""
    blocks, by_key = [], {}
    for e in events:
        if isinstance(e, (ev.ThinkingDelta, ev.TextDelta, ev.ThinkingSignature)):
            b = by_key.get(e.key)
            if b is None:
                b = B.of_thinking("") if not isinstance(e, ev.TextDelta) else B.of_text("")
                by_key[e.key] = b
                blocks.append(b)
            if isinstance(e, ev.ThinkingDelta):
                b.thinking += e.text
            elif isinstance(e, ev.TextDelta):
                b.text += e.text
            else:
                b.signature = e.signature
        elif isinstance(e, ev.ToolCall):
            blocks.append(B.of_tool_use(e.id, e.name, json.loads(e.input_json)))
    for b in blocks:
        if b.type == "thinking" and not b.signature:
            b.signature = sigs.synthetic_signature(b.thinking)
    return M("assistant", blocks)


# =========================================================================================
# request goldens
# =========================================================================================

class BodyTests(unittest.TestCase):
    def test_thinking_config_25_vs_3(self):
        def tc(model_id, **kw):
            gen = body_for(make_req([user("hi")], **kw), model_id)["generationConfig"]
            return gen.get("thinkingConfig")

        self.assertEqual(tc(G25_PRO), {"includeThoughts": True, "thinkingBudget": -1})
        self.assertEqual(tc(G25_PRO, effort="none"), {"includeThoughts": True, "thinkingBudget": -1})
        self.assertEqual(tc(G25_FLASH, effort="none"), {"includeThoughts": True, "thinkingBudget": 0})
        self.assertEqual(tc(G25_FLASH, effort="low"), {"includeThoughts": True, "thinkingBudget": -1})
        self.assertEqual(tc(G3), {"includeThoughts": True, "thinkingLevel": "high"})
        self.assertEqual(tc(G3, effort=None), {"includeThoughts": True, "thinkingLevel": "high"})
        for effort in ("none", "minimal", "low"):
            self.assertEqual(tc("gemini-3.8-flash", effort=effort), {"includeThoughts": True, "thinkingLevel": "low"})
        self.assertEqual(tc(G3, effort="max")["thinkingLevel"], "high")
        self.assertIsNone(tc(G3, max_tokens=1))           # /model probe
        self.assertIsNone(tc("gemma-3-27b-it"))           # non-thinking model
        for model_id in (G3, G25_PRO, G25_FLASH):
            structurally_valid(self, body_for(make_req([user("hi")], effort="none"), model_id), model_id)

    def test_golden_simple_body(self):
        req = make_req([user("Run it", "<system-reminder>\nx\n</system-reminder>")], tools=[BASH],
                       system=["S1", "S2"], tool_choice={"type": "auto"}, max_tokens=32000, temperature=0.5)
        self.assertEqual(body_for(req), {
            "contents": [{"role": "user", "parts": [{"text": "Run it"},
                                                    {"text": "<system-reminder>\nx\n</system-reminder>"}]}],
            "systemInstruction": {"parts": [{"text": "S1\n\nS2"}]},
            "tools": [{"functionDeclarations": [{
                "name": "Bash", "description": "Run a shell command",
                "parametersJsonSchema": {"type": "object",
                                         "properties": {"command": {"type": "string", "description": "the command"},
                                                        "timeout": {"type": "number", "maximum": 600000,
                                                                    "exclusiveMinimum": 0},
                                                        "description": {"type": "string"}},
                                         "required": ["command"], "additionalProperties": False}}]}],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            "generationConfig": {"maxOutputTokens": 32000, "temperature": 0.5,
                                 "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "high"}},
        })

    def test_roles_merge_and_continue(self):
        req = make_req([
            assistant("Hello"),
            user("a"), user("b"),
            assistant(B.of_thinking("private chain", sigs.synthetic_signature("private chain")), "c"),
            assistant(B.of_redacted_thinking("xx"), "d"),
            user(""), user("e"),
        ], system=[])
        body = body_for(req)
        self.assertNotIn("systemInstruction", body)
        self.assertEqual(body["contents"], [
            {"role": "user", "parts": [{"text": "(continue)"}]},
            {"role": "model", "parts": [{"text": "Hello"}]},
            {"role": "user", "parts": [{"text": "a"}, {"text": "b"}]},
            {"role": "model", "parts": [{"text": "c"}, {"text": "d"}]},
            {"role": "user", "parts": [{"text": "e"}]},
        ])
        self.assertNotIn("private chain", json.dumps(body))
        self.assertNotIn("thought", list(walk_keys(body)))
        structurally_valid(self, body)
        empty = body_for(make_req([user("")], system=[]))
        self.assertEqual(empty["contents"], [{"role": "user", "parts": [{"text": "(continue)"}]}])

    def test_function_response_grouping_parallel_and_images(self):
        req = make_req([
            user("go"),
            assistant("Running both.", B.of_tool_use("toolu_aaaaaaaaaaaaaaaaaaaaaaaa", "Bash", {"command": "ls"}),
                      B.of_tool_use("toolu_bbbbbbbbbbbbbbbbbbbbbbbb", "Read", {"file_path": "/x.png"}),
                      B.of_tool_use("toolu_cccccccccccccccccccccccc", "Read", {"file_path": "/y.pdf"})),
            user(B.of_tool_result("toolu_aaaaaaaaaaaaaaaaaaaaaaaa", "boom", is_error=True),
                 B.of_tool_result("toolu_bbbbbbbbbbbbbbbbbbbbbbbb", [B.of_image_base64("image/png", PNG)])),
            user(B.of_tool_result("toolu_cccccccccccccccccccccccc",
                                  [B.of_text("pdf follows"), B.of_document_base64("application/pdf", PDF),
                                   B.of_image_url("https://example.com/a.png")]),
                 "next please", B.of_image_base64("image/jpeg", PNG), B.of_document_text("notes", title="n.txt"),
                 B.of_image_url("https://example.com/b.png")),
        ], tools=[BASH, READ])
        body = body_for(req, G25_PRO)
        self.assertEqual(len(body["contents"]), 3)
        self.assertEqual(body["contents"][1]["parts"][1:], [
            {"functionCall": {"name": "Bash", "args": {"command": "ls"}}},
            {"functionCall": {"name": "Read", "args": {"file_path": "/x.png"}}},
            {"functionCall": {"name": "Read", "args": {"file_path": "/y.pdf"}}}])
        self.assertEqual(body["contents"][2], {"role": "user", "parts": [
            {"functionResponse": {"name": "Bash", "response": {"error": "boom"}}},
            {"functionResponse": {"name": "Read", "response": {"output": "[image attached below]"}}},
            {"functionResponse": {"name": "Read", "response": {
                "output": "pdf follows\n[document attached below]\n[image: https://example.com/a.png]"}}},
            {"inlineData": {"mimeType": "image/png", "data": PNG}},
            {"inlineData": {"mimeType": "application/pdf", "data": PDF}},
            {"text": "next please"},
            {"inlineData": {"mimeType": "image/jpeg", "data": PNG}},
            {"text": "[document: n.txt]\nnotes"},
            {"text": "[image: https://example.com/b.png]"},
        ]})
        structurally_valid(self, body, G25_PRO)

    def test_orphan_tool_result_becomes_text(self):
        body = body_for(make_req([user(B.of_tool_result("toolu_zz", "late output"), "hi")]))
        self.assertEqual(body["contents"], [{"role": "user", "parts": [
            {"text": "[tool result toolu_zz]\nlate output"}, {"text": "hi"}]}])

    def test_thought_signature_placement(self):
        tu = B.of_tool_use("toolu_aaaaaaaaaaaaaaaaaaaaaaaa", "Bash", {"command": "ls"})
        tu2 = B.of_tool_use("toolu_bbbbbbbbbbbbbbbbbbbbbbbb", "Bash", {"command": "pwd"})
        res = user(B.of_tool_result(tu.id, "a"), B.of_tool_result(tu2.id, "b"))
        hist = [user("go"),
                assistant(B.of_thinking("I should look", sigs.synthetic_signature("I should look")),
                          B.of_thinking("", gsig("S1")), "Let me check.", tu,
                          B.of_thinking("", gsig("S2")), tu2,
                          B.of_thinking("", gsig("TRAILING"))),
                res]
        body = body_for(make_req(hist, tools=[BASH]), G25_PRO)
        self.assertEqual(body["contents"][1]["parts"], [
            {"text": "Let me check.", "thoughtSignature": "S1"},
            {"functionCall": {"name": "Bash", "args": {"command": "ls"}}},
            {"functionCall": {"name": "Bash", "args": {"command": "pwd"}}, "thoughtSignature": "S2"}])
        dumped = json.dumps(body)
        self.assertNotIn("I should look", dumped)
        self.assertNotIn("TRAILING", dumped)
        # signatures for another target / non-fgw1 signatures are never sent
        hist2 = [user("go"), assistant(B.of_thinking("t", gsig("VSIG", "vertex")), tu),
                 user(B.of_tool_result(tu.id, "a"))]
        body2 = body_for(make_req(hist2, tools=[BASH]), G25_PRO)
        self.assertEqual(body2["contents"][1]["parts"], [{"functionCall": {"name": "Bash", "args": {"command": "ls"}}}])
        hist3 = [user("go"), assistant(B.of_thinking("t", "EqQBCkYIBhgCKkAnthropicNative"), tu),
                 user(B.of_tool_result(tu.id, "a"))]
        self.assertNotIn("thoughtSignature", json.dumps(body_for(make_req(hist3, tools=[BASH]), G25_PRO)))
        vprov = provider("gemini-vertex", options={"project": "p"})
        body4 = body_for(make_req(hist2, tools=[BASH]), G25_PRO, prov=vprov)
        self.assertEqual(body4["contents"][1]["parts"][0]["thoughtSignature"], "VSIG")

    def test_dummy_signature_and_lru(self):
        tu = B.of_tool_use("toolu_aaaaaaaaaaaaaaaaaaaaaaaa", "Bash", {"command": "ls"})
        tu2 = B.of_tool_use("toolu_bbbbbbbbbbbbbbbbbbbbbbbb", "Bash", {"command": "pwd"})
        hist = [user("go"), assistant("Checking.", tu, tu2),
                user(B.of_tool_result(tu.id, "a"), B.of_tool_result(tu2.id, "b")),
                assistant(B.of_tool_use("toolu_cccccccccccccccccccccccc", "Bash", {"command": "id"})),
                user(B.of_tool_result("toolu_cccccccccccccccccccccccc", "c"))]
        req = make_req(hist, tools=[BASH])
        body = body_for(req, G3)
        turn1, turn2 = body["contents"][1]["parts"], body["contents"][3]["parts"]
        self.assertEqual(turn1[0], {"text": "Checking."})
        self.assertEqual(turn1[1]["thoughtSignature"], gem.DUMMY_SIGNATURE)
        self.assertNotIn("thoughtSignature", turn1[2])
        self.assertEqual(turn2[0]["thoughtSignature"], gem.DUMMY_SIGNATURE)
        structurally_valid(self, body, G3, {"require_signatures": True, "allow_skip_signature": True})
        self.assertNotIn("thoughtSignature", json.dumps(body_for(req, G25_PRO)))  # 2.5: never a dummy
        rt = dbase.ProviderRuntime("gemini")
        rt.setdefault("gemini_sig_lru", lambda: dbase.LRU(4096)).put(tu.id, "FROM_LRU")
        body = body_for(req, G3, runtime=rt)
        self.assertEqual(body["contents"][1]["parts"][1]["thoughtSignature"], "FROM_LRU")
        self.assertEqual(body["contents"][3]["parts"][0]["thoughtSignature"], gem.DUMMY_SIGNATURE)

    def test_tool_names_schema_and_choice(self):
        odd = md.ToolDef("9lives", "", {"type": "object", "properties": {"x": {"type": "integer"}}})
        tu = B.of_tool_use("toolu_aaaaaaaaaaaaaaaaaaaaaaaa", LONG_MCP, {"node": {"selector": "a"}})
        hist = [user("go"), assistant(tu), user(B.of_tool_result(tu.id, "ok"))]
        req = make_req(hist, tools=[BASH, REFS, odd], tool_choice={"type": "tool", "name": LONG_MCP})
        body = body_for(req)
        decls = body["tools"][0]["functionDeclarations"]
        names = [d["name"] for d in decls]
        self.assertEqual(names[0], "Bash")
        for n in names:
            self.assertRegex(n, NAME_RE)
        self.assertEqual(names[2][0], "_")  # Gemini names must start with a letter or "_"
        self.assertNotIn("description", decls[2])
        long_up = names[1]
        self.assertLessEqual(len(long_up), 64)
        self.assertNotEqual(long_up, LONG_MCP)
        self.assertEqual(body["toolConfig"],
                         {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [long_up]}})
        self.assertEqual(body["contents"][1]["parts"][0]["functionCall"]["name"], long_up)
        self.assertEqual(body["contents"][2]["parts"][0]["functionResponse"]["name"], long_up)
        for k in ("$schema", "$ref", "$defs"):
            self.assertNotIn(k, list(walk_keys(body["tools"])))
        self.assertEqual(decls[1]["parametersJsonSchema"]["properties"]["node"]["properties"],
                         {"selector": {"type": "string"}})
        self.assertEqual(decls[1]["parametersJsonSchema"]["properties"]["mode"]["type"], "string")
        structurally_valid(self, body, G3, {"require_signatures": False})
        for choice, cfg in (({"type": "auto"}, {"mode": "AUTO"}), ({"type": "any"}, {"mode": "ANY"}),
                            ({"type": "none"}, {"mode": "NONE"})):
            b = body_for(make_req([user("x")], tools=[BASH], tool_choice=choice))
            self.assertEqual(b["toolConfig"], {"functionCallingConfig": cfg})
        self.assertNotIn("toolConfig", body_for(make_req([user("x")], tools=[BASH])))
        self.assertNotIn("tools", body_for(make_req([user("x")])))

    def test_schema_fallback_body(self):
        req = make_req([user("x")], tools=[BASH, READ, md.ToolDef("Noop", "nothing")],
                       output_format={"type": "json_schema", "schema": {
                           "type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"],
                           "additionalProperties": False}})
        body = body_for(req, fallback=True)
        decls = body["tools"][0]["functionDeclarations"]
        self.assertNotIn("parametersJsonSchema", json.dumps(body))
        self.assertEqual(decls[0]["parameters"]["required"], ["command"])
        self.assertNotIn("additionalProperties", list(walk_keys(decls)))
        self.assertNotIn("parameters", decls[2])  # OpenAPI OBJECT without properties is rejected upstream
        gen = body["generationConfig"]
        self.assertEqual(gen["responseMimeType"], "application/json")
        self.assertNotIn("responseJsonSchema", gen)
        self.assertEqual(gen["responseSchema"]["properties"], {"title": {"type": "string"}})
        structurally_valid(self, body, G3, {"legacy": True})

    def test_generation_config(self):
        req = make_req([user("x")], max_tokens=100000, temperature=0.2, top_p=0.9, top_k=40,
                       stop_sequences=["a", "b", "c", "d", "e", "f", "g"],
                       output_format={"type": "json_schema", "schema": {
                           "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
                           "properties": {"title": {"type": "string"}}}})
        gen = body_for(req)["generationConfig"]
        self.assertEqual(gen["maxOutputTokens"], 65536)
        self.assertEqual((gen["temperature"], gen["topP"], gen["topK"]), (0.2, 0.9, 40))
        self.assertEqual(gen["stopSequences"], ["a", "b", "c", "d", "e"])
        self.assertEqual(gen["responseMimeType"], "application/json")
        self.assertEqual(gen["responseJsonSchema"], {"type": "object", "properties": {"title": {"type": "string"}}})
        unlisted = body_for(make_req([user("x")], max_tokens=200000), "gemini-9-ultra")["generationConfig"]
        self.assertEqual(unlisted["maxOutputTokens"], gem.DEFAULT_MAX_OUTPUT)
        self.assertEqual(body_for(make_req([user("x")], max_tokens=1))["generationConfig"], {"maxOutputTokens": 1})

    def test_call_ids_only_when_issued_by_gemini(self):
        encoded = toolnames.encode_tool_id("calls/7:x")
        ids = ["toolu_0123456789abcdef01234567", "toolu_01AbCdEfGhIjKlMnOpQrStUv", "fc-0001", encoded]
        uses = [B.of_tool_use(i, "Bash", {"command": str(n)}) for n, i in enumerate(ids)]
        hist = [user("go"), assistant(*uses), user(*[B.of_tool_result(i, "r") for i in ids])]
        body = body_for(make_req(hist, tools=[BASH]), G25_PRO)
        calls = [p["functionCall"].get("id") for p in body["contents"][1]["parts"]]
        resps = [p["functionResponse"].get("id") for p in body["contents"][2]["parts"]]
        self.assertEqual(calls, [None, None, "fc-0001", "calls/7:x"])
        self.assertEqual(resps, calls)
        structurally_valid(self, body, G25_PRO)

    def test_endpoints(self):
        d = gem.GeminiDialect()
        req = make_req([user("x")])
        url, hdrs = d.endpoint(make_ctx(req, provider()))
        self.assertEqual(url, "https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent"
                              "?alt=sse" % G3)
        self.assertEqual(hdrs["Accept"], "text/event-stream")
        url, _ = d.endpoint(make_ctx(req, provider(base_url="https://proxy.example/v1beta/")))
        self.assertEqual(url, "https://proxy.example/v1beta/models/%s:streamGenerateContent?alt=sse" % G3)
        vprov = provider("gemini-vertex")
        url, hdrs = d.endpoint(make_ctx(req, vprov, auth=FakeAuth(style="bearer")))
        self.assertEqual(url, "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/global/publishers/"
                              "google/models/%s:streamGenerateContent?alt=sse" % G3)
        self.assertEqual(hdrs["x-goog-user-project"], "proj-1")
        url, _ = d.endpoint(make_ctx(req, vprov, auth=FakeAuth(style="bearer", location="us-central1")))
        self.assertTrue(url.startswith("https://us-central1-aiplatform.googleapis.com/v1/projects/proj-1/locations/"
                                       "us-central1/"), url)
        opt = provider("gemini-vertex", options={"project": "other", "location": "europe-west4"})
        url, hdrs = d.endpoint(make_ctx(req, opt, auth=FakeAuth(style="bearer")))
        self.assertIn("europe-west4-aiplatform.googleapis.com/v1/projects/other/locations/europe-west4/", url)
        self.assertEqual(hdrs["x-goog-user-project"], "other")
        self.assertEqual(gem.vertex_host("global"), "aiplatform.googleapis.com")
        with self.assertRaises(errors.GatewayError) as cm:
            d.endpoint(make_ctx(req, vprov, auth=FakeAuth(style="bearer", project=None)))
        self.assertEqual(cm.exception.status, 401)
        self.assertIn("GOOGLE_CLOUD_PROJECT", cm.exception.message)

    def test_claude_code_fixtures(self):
        try:
            ain = mod("anthropic_in")
            parse = ain.parse_messages_request
        except Exception as exc:  # noqa: BLE001 - agent A's module may be mid-edit
            self.skipTest("anthropic_in unavailable: %s" % exc)
        testing = mod("testing")
        for name in testing.list_fixtures():
            fx = testing.load_fixture(name)
            if not fx["path"].startswith("/v1/messages") or "count_tokens" in fx["path"]:
                continue
            with self.subTest(fixture=name):
                req = parse(fx["body"], fx["headers"])
                body = body_for(req, G3)
                structurally_valid(self, body, G3, {"require_signatures": True, "allow_skip_signature": True})
                self.assertNotIn("x-anthropic-billing-header", json.dumps(body))


# =========================================================================================
# stream parsing
# =========================================================================================

def _cand(parts, finish=None):
    c = {"content": {"role": "model", "parts": parts}, "index": 0}
    if finish:
        c["finishReason"] = finish
    return {"candidates": [c]}


class StreamParserTests(unittest.TestCase):
    def parse(self, chunks, names=None, target="gemini_api"):
        self.lru = dbase.LRU(16)
        names = names or toolnames.ToolNameMap(["Bash", LONG_MCP], regex=presets.GEMINI_TOOL_NAME_REGEX,
                                               leading_letter=True)
        p = gem.GeminiStreamParser(names, target, self.lru, "gemini", G3)
        out = []
        for c in chunks:
            out.extend(p.feed(c))
        return out + p.close()

    def test_text_thinking_and_keys(self):
        out = self.parse([_cand([{"text": "Plan", "thought": True}]), _cand([{"text": " more", "thought": True}]),
                          _cand([{"text": "Hel"}]), _cand([{"text": "lo"}], "STOP")])
        self.assertEqual(out, [ev.ThinkingDelta(1, "Plan"), ev.ThinkingDelta(1, " more"), ev.TextDelta(2, "Hel"),
                               ev.TextDelta(2, "lo"), ev.Finish("end_turn")])

    def test_signature_precedes_function_call(self):
        names = toolnames.ToolNameMap(["Bash", LONG_MCP], regex=presets.GEMINI_TOOL_NAME_REGEX, leading_letter=True)
        up = names.upstream(LONG_MCP)
        out = self.parse([_cand([{"text": "think", "thought": True}]),
                          _cand([{"text": "Calling."},
                                 {"functionCall": {"name": up, "args": {"node": {"selector": "a"}}},
                                  "thoughtSignature": "SIG1"},
                                 {"functionCall": {"name": "Bash", "args": '{"command": "ls"}', "id": "fc-9"}}],
                                "STOP")], names)
        tool_id = out[4].id
        self.assertRegex(tool_id, r"^toolu_[0-9a-f]{24}$")
        self.assertEqual(out, [
            ev.ThinkingDelta(1, "think"), ev.TextDelta(2, "Calling."),
            ev.ThinkingDelta(3, ""), ev.ThinkingSignature(3, gsig("SIG1")),
            ev.ToolCall(tool_id, LONG_MCP, '{"node":{"selector":"a"}}'),
            ev.ToolCall("fc-9", "Bash", '{"command":"ls"}'),
            ev.Finish("tool_use")])
        self.assertEqual(self.lru.get(tool_id), "SIG1")
        self.assertIsNone(self.lru.get("fc-9"))
        self.assertEqual(sigs.decode_signature(out[3].signature), ("gemini_api", {"s": "SIG1"}))

    def test_signature_on_text_and_trailing_empty_part(self):
        out = self.parse([_cand([{"text": "Hel"}]), _cand([{"text": "lo", "thoughtSignature": "T1"}]),
                          _cand([{"text": "", "thoughtSignature": "T2"}], "STOP")], target="vertex")
        self.assertEqual(out, [ev.TextDelta(1, "Hel"), ev.ThinkingDelta(2, ""),
                               ev.ThinkingSignature(2, gsig("T1", "vertex")), ev.TextDelta(3, "lo"),
                               ev.ThinkingDelta(4, ""), ev.ThinkingSignature(4, gsig("T2", "vertex")),
                               ev.Finish("end_turn")])

    def test_finish_reasons(self):
        def finish(reason, parts=None):
            return self.parse([_cand(parts or [{"text": "x"}], reason)])

        self.assertEqual(finish("STOP")[-1], ev.Finish("end_turn"))
        self.assertEqual(finish("MAX_TOKENS")[-1], ev.Finish("max_tokens"))
        self.assertEqual(finish("FINISH_REASON_UNSPECIFIED")[-1], ev.Finish("end_turn"))
        self.assertEqual(self.parse([_cand([{"text": "x"}])])[-1], ev.Finish("end_turn"))
        for reason in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
            out = finish(reason)
            self.assertEqual(out[-1], ev.Finish("end_turn"))
            self.assertIsInstance(out[-2], ev.TextDelta)
            self.assertIn(reason, out[-2].text)
            self.assertNotEqual(out[-2].key, out[0].key)
        out = self.parse([_cand([], "MALFORMED_FUNCTION_CALL")])
        self.assertEqual(out, [ev.TextDelta(1, gem.MALFORMED_NOTE), ev.Finish("end_turn")])
        self.assertEqual(gem.MALFORMED_NOTE, "[model produced a malformed tool call; please retry]")
        out = finish("STOP", [{"functionCall": {"name": "Bash", "args": {}}}])
        self.assertEqual(out[-1], ev.Finish("tool_use"))
        out = finish("MAX_TOKENS", [{"functionCall": {"name": "Bash", "args": {}}}])
        self.assertEqual(out[-1], ev.Finish("tool_use"))
        out = self.parse([_cand([{"functionCall": {"args": {}}}], "STOP")])
        self.assertEqual(out, [ev.TextDelta(1, gem.MALFORMED_NOTE), ev.Finish("end_turn")])
        with self.assertRaises(errors.GatewayError) as cm:
            self.parse([_cand([], "MISSING_THOUGHT_SIGNATURE")])
        self.assertEqual((cm.exception.status, cm.exception.err_type), (400, "invalid_request_error"))
        self.assertIn("MISSING_THOUGHT_SIGNATURE", cm.exception.message)
        out = self.parse([{"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "usageMetadata": {
            "promptTokenCount": 9}}])
        self.assertEqual(out, [ev.TextDelta(1, "[prompt blocked by provider (PROHIBITED_CONTENT)]"),
                               ev.Usage(9, 0, 0), ev.Finish("end_turn")])

    def test_usage(self):
        out = self.parse([
            dict(_cand([{"text": "a"}]), usageMetadata={"promptTokenCount": 100, "totalTokenCount": 100}),
            dict(_cand([{"text": "b"}], "STOP"), usageMetadata={
                "promptTokenCount": 100, "cachedContentTokenCount": 30, "candidatesTokenCount": 20,
                "thoughtsTokenCount": 5, "totalTokenCount": 125})])
        self.assertEqual(out[-2:], [ev.Usage(70, 25, 30), ev.Finish("end_turn")])

    def test_errors_and_malformed_input(self):
        with self.assertRaises(errors.GatewayError) as cm:
            self.parse([{"error": {"code": 500, "message": "An internal error has occurred.", "status": "INTERNAL"}}])
        self.assertEqual((cm.exception.status, cm.exception.should_retry), (502, True))
        with self.assertRaises(errors.GatewayError) as cm:
            self.parse([{"error": {"message": "slow down", "status": "RESOURCE_EXHAUSTED", "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}]}}])
        self.assertEqual((cm.exception.status, cm.exception.should_retry, cm.exception.retry_after), (429, True, 3.0))
        with self.assertRaises(errors.GatewayError) as cm:
            self.parse([])
        self.assertEqual((cm.exception.status, cm.exception.err_type, cm.exception.should_retry),
                         (502, "api_error", True))
        out = self.parse(["junk", {"candidates": "nope"}, {"candidates": [{"content": {"parts": ["x", None]}}]},
                          _cand([{"functionCall": {"name": "Bash", "args": "not json"}}], "STOP")])
        self.assertEqual([type(e) for e in out], [ev.ToolCall, ev.Finish])
        self.assertEqual(out[0].input_json, "{}")


# =========================================================================================
# execute() against the mocks
# =========================================================================================

class _MockCase(unittest.TestCase):
    def start(self, kind, **options):
        srv = mocks.MockServer(kind, options=options).start()
        self.addCleanup(srv.stop)
        return srv

    def run_turn(self, ctx):
        return list(gem.GeminiDialect().execute(ctx))

    def assertNoMockErrors(self, srv):
        self.assertEqual(srv.errors, [])
        self.assertEqual(srv.brain.leaks, [])


class ExecuteTests(_MockCase):
    def two_turns(self, srv, prov, auth, model_id=G3, strip_thinking=False, runtime=None):
        runtime = runtime or dbase.ProviderRuntime(prov.id)
        hist = [user("Run the shell command and report")]
        events = self.run_turn(make_ctx(make_req(hist, tools=[BASH, READ]), prov, model_id, auth, runtime))
        calls = [e for e in events if isinstance(e, ev.ToolCall)]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "Bash")
        self.assertEqual(json.loads(calls[0].input_json)["command"], mocks.DEFAULT_COMMAND)
        sig_events = [e for e in events if isinstance(e, ev.ThinkingSignature)]
        self.assertEqual(len(sig_events), 1)
        self.assertEqual(sigs.signature_target(sig_events[0].signature), prov.target)
        self.assertLess(events.index(sig_events[0]), events.index(calls[0]))
        self.assertEqual(events[-1], ev.Finish("tool_use"))
        self.assertIsInstance(events[-2], ev.Usage)
        echo = assistant_from_events(events)
        if strip_thinking:
            echo.blocks = [b for b in echo.blocks if b.type != "thinking"]
        hist += [echo, user(B.of_tool_result(calls[0].id, "hello"))]
        events2 = self.run_turn(make_ctx(make_req(hist, tools=[BASH, READ]), prov, model_id, auth, runtime))
        text = "".join(e.text for e in events2 if isinstance(e, ev.TextDelta))
        self.assertEqual(text, "DONE %s" % mocks.sha8("hello"))
        self.assertEqual(events2[-1], ev.Finish("end_turn"))
        req2 = srv.requests[-1]["body_json"]
        self.assertNotIn(srv.brain.thinking, json.dumps(req2))
        self.assertNoMockErrors(srv)
        return calls[0], req2, runtime

    def test_two_turns_gemini_api_signature_echo(self):
        srv = self.start("gemini_api", api_key="k1")
        prov = provider(base_url=srv.url)
        _, req2, _ = self.two_turns(srv, prov, FakeAuth(["k1"]))
        rec = srv.requests[0]
        self.assertEqual(rec["path"], "/v1beta/models/%s:streamGenerateContent" % G3)
        self.assertEqual(rec["query"], {"alt": ["sse"]})
        self.assertEqual(rec["headers"]["x-goog-api-key"], "k1")
        self.assertNotIn("Authorization", rec["headers"])
        model_turn = req2["contents"][1]["parts"]
        self.assertIn(model_turn[0]["thoughtSignature"], srv.state["issued"])
        self.assertEqual(srv.state["sig_checks"], [model_turn[0]["thoughtSignature"]])
        self.assertEqual(req2["contents"][2]["parts"], [
            {"functionResponse": {"name": "Bash", "response": {"output": "hello"}}}])

    def test_two_turns_vertex_with_call_ids(self):
        srv = self.start("vertex", token="tok", project="proj-1", location="us-central1", call_ids=True)
        prov = provider("gemini-vertex", base_url=srv.url)
        auth = FakeAuth(["tok"], style="bearer", location="us-central1")
        call, req2, _ = self.two_turns(srv, prov, auth)
        self.assertEqual(call.id, "fc-0001")
        rec = srv.requests[0]
        self.assertEqual(rec["path"], "/v1/projects/proj-1/locations/us-central1/publishers/google/models/%s:"
                                      "streamGenerateContent" % G3)
        self.assertEqual((rec["headers"]["Authorization"], rec["headers"]["x-goog-user-project"]),
                         ("Bearer tok", "proj-1"))
        self.assertEqual(req2["contents"][1]["parts"][0]["functionCall"]["id"], "fc-0001")
        self.assertEqual(req2["contents"][2]["parts"][0]["functionResponse"]["id"], "fc-0001")

    def test_lru_supplies_signature_when_thinking_is_stripped(self):
        srv = self.start("gemini_api")
        prov = provider(base_url=srv.url)
        _, req2, rt = self.two_turns(srv, prov, FakeAuth(), strip_thinking=True)
        self.assertIn(req2["contents"][1]["parts"][0]["thoughtSignature"], srv.state["issued"])
        self.assertEqual(len(rt.get("gemini_sig_lru")), 1)  # only functionCall signatures are remembered

    def test_dummy_signature_path(self):
        tu = B.of_tool_use("toolu_aaaaaaaaaaaaaaaaaaaaaaaa", "Bash", {"command": "ls"})
        hist = [user("go"), assistant(tu), user(B.of_tool_result(tu.id, "hello"))]
        strict = self.start("gemini_api")
        with self.assertRaises(errors.GatewayError) as cm:
            self.run_turn(make_ctx(make_req(hist, tools=[BASH]), provider(base_url=strict.url)))
        self.assertEqual((cm.exception.status, cm.exception.err_type), (400, "invalid_request_error"))
        self.assertIn("Corrupted thought signature", cm.exception.message)
        self.assertEqual(len(strict.requests), 1)  # already all-dummy: no pointless signature-reset retry
        lenient = self.start("gemini_api", allow_skip_signature=True)
        events = self.run_turn(make_ctx(make_req(hist, tools=[BASH]), provider(base_url=lenient.url)))
        self.assertEqual(events[-1], ev.Finish("end_turn"))
        sent = lenient.requests[-1]["body_json"]["contents"][1]["parts"][0]
        self.assertEqual(sent["thoughtSignature"], gem.DUMMY_SIGNATURE)
        # 2.5 models never validate signatures and never get the dummy
        events = self.run_turn(make_ctx(make_req(hist, tools=[BASH]), provider(base_url=strict.url), G25_PRO))
        self.assertEqual(events[-1], ev.Finish("end_turn"))
        self.assertNotIn("thoughtSignature", strict.requests[-1]["body_json"]["contents"][1]["parts"][0])

    def test_model_switch_resets_signatures_once_per_session(self):
        for mode in (None, "stream"):  # rejection as HTTP 400, or as finishReason MISSING_THOUGHT_SIGNATURE
            with self.subTest(signature_errors=mode):
                srv = self.start("gemini_api", allow_skip_signature=True, signature_errors=mode)
                prov = provider(base_url=srv.url)
                rt = dbase.ProviderRuntime(prov.id)
                hist = [user("go")]
                events = self.run_turn(make_ctx(make_req(hist, tools=[BASH]), prov, G3, runtime=rt))
                call = [e for e in events if isinstance(e, ev.ToolCall)][0]
                hist += [assistant_from_events(events), user(B.of_tool_result(call.id, "hello"))]
                srv.clear()
                with self.assertLogs("ai_gateway", level="WARNING") as logs:
                    events = self.run_turn(make_ctx(make_req(hist, tools=[BASH]), prov, "gemini-3.8-flash", runtime=rt))
                self.assertIn("skip_thought_signature_validator", logs.output[0])
                self.assertEqual(events[-1], ev.Finish("end_turn"))
                sent = [r["body_json"]["contents"][1]["parts"][0]["thoughtSignature"] for r in srv.requests]
                self.assertEqual(len(sent), 2)
                self.assertEqual(srv.state["issued"][sent[0]], G3)
                self.assertEqual(sent[1], gem.DUMMY_SIGNATURE)
                hist += [assistant_from_events(events), user("again")]
                srv.clear()
                events = self.run_turn(make_ctx(make_req(hist, tools=[BASH]), prov, "gemini-3.8-flash", runtime=rt))
                self.assertEqual(events[-1], ev.Finish("end_turn"))
                self.assertEqual(len(srv.requests), 1)  # sticky for the session: dummies straight away
                body = srv.requests[0]["body_json"]
                self.assertEqual(body["contents"][1]["parts"][0]["thoughtSignature"], gem.DUMMY_SIGNATURE)
                self.assertNotIn("thoughtSignature", body["contents"][3]["parts"][0])
                other = gem.GeminiDialect().build_body(make_ctx(make_req(hist, tools=[BASH]), prov, G3, runtime=rt,
                                                                session_id="sess-2"))[0]
                self.assertEqual(other["contents"][1]["parts"][0]["thoughtSignature"], sent[0])
                self.assertNoMockErrors(srv)

    def test_unrecognised_gemini3_alias_gets_dummy_on_retry(self):
        srv = self.start("gemini_api", allow_skip_signature=True, require_signatures=True)
        tu = B.of_tool_use("toolu_aaaaaaaaaaaaaaaaaaaaaaaa", "Bash", {"command": "ls"})
        hist = [user("go"), assistant(tu), user(B.of_tool_result(tu.id, "hello"))]
        with self.assertLogs("ai_gateway", level="WARNING"):
            events = self.run_turn(make_ctx(make_req(hist, tools=[BASH]), provider(base_url=srv.url),
                                            "gemini-flash-latest"))
        self.assertEqual(events[-1], ev.Finish("end_turn"))
        first, second = [r["body_json"]["contents"][1]["parts"][0] for r in srv.requests]
        self.assertNotIn("thoughtSignature", first)
        self.assertEqual(second["thoughtSignature"], gem.DUMMY_SIGNATURE)

    def test_schema_fallback_is_sticky(self):
        srv = self.start("gemini_api", legacy=True)
        prov = provider(base_url=srv.url)
        rt = dbase.ProviderRuntime(prov.id)
        with self.assertLogs("ai_gateway", level="WARNING") as logs:
            events = self.run_turn(make_ctx(make_req([user("go")], tools=[BASH, REFS]), prov, runtime=rt))
            self.assertEqual(events[-1], ev.Finish("tool_use"))
            self.assertTrue(rt.get("gemini_schema_fallback"))
            first, second = [r["body_json"]["tools"][0]["functionDeclarations"][0] for r in srv.requests]
            self.assertIn("parametersJsonSchema", first)
            self.assertIn("parameters", second)
            srv.clear()
            events = self.run_turn(make_ctx(make_req([user("go")], tools=[BASH]), prov, runtime=rt))
            self.assertEqual(events[-1], ev.Finish("tool_use"))
            self.assertEqual(len(srv.requests), 1)
            self.assertIn("parameters", srv.requests[0]["body_json"]["tools"][0]["functionDeclarations"][0])
        self.assertEqual(len([r for r in logs.records if "parametersJsonSchema" in r.getMessage()]), 1)

    def test_non_schema_400_does_not_trigger_fallback(self):
        srv = self.start("gemini_api")
        rt = dbase.ProviderRuntime("gemini")
        t1 = B.of_tool_use("toolu_aaaaaaaaaaaaaaaaaaaaaaaa", "Bash", {"command": "ls"})
        t2 = B.of_tool_use("toolu_bbbbbbbbbbbbbbbbbbbbbbbb", "Bash", {"command": "pwd"})
        req = make_req([user("go"), assistant(t1, t2), user(B.of_tool_result(t1.id, "only one"))], tools=[BASH])
        with self.assertRaises(errors.GatewayError) as cm:
            self.run_turn(make_ctx(req, provider(base_url=srv.url), G25_PRO, runtime=rt))
        self.assertEqual((cm.exception.status, cm.exception.err_type), (400, "invalid_request_error"))
        self.assertIn("number of function response parts", cm.exception.message)
        self.assertIsNone(rt.get("gemini_schema_fallback"))
        self.assertEqual(len(srv.requests), 1)

    def test_429_classification(self):
        for kind, prov_name, auth in (("gemini_api", "gemini", FakeAuth()),
                                      ("vertex", "gemini-vertex", FakeAuth(style="bearer"))):
            with self.subTest(kind=kind):
                srv = self.start(kind, mode="429_transient")
                with self.assertRaises(errors.GatewayError) as cm:
                    self.run_turn(make_ctx(make_req([user("x")]), provider(prov_name, base_url=srv.url), auth=auth))
                e = cm.exception
                self.assertEqual((e.status, e.err_type, e.should_retry, e.retry_after),
                                 (429, "rate_limit_error", True, 2.0))
                self.assertEqual(e.headers()["x-should-retry"], "true")
                srv = self.start(kind, mode="429_daily")
                with self.assertRaises(errors.GatewayError) as cm:
                    self.run_turn(make_ctx(make_req([user("x")]), provider(prov_name, base_url=srv.url), auth=auth))
                e = cm.exception
                self.assertEqual((e.status, e.should_retry), (429, False))
                self.assertIn("quota exhausted", e.message)

    def test_401_refresh_retry_vertex(self):
        srv = self.start("vertex", token="fresh")
        prov = provider("gemini-vertex", base_url=srv.url)
        auth = FakeAuth(["stale", "fresh"], style="bearer")
        events = self.run_turn(make_ctx(make_req([user("hi")]), prov, auth=auth))
        self.assertEqual(events[-1], ev.Finish("end_turn"))
        self.assertEqual([r["headers"]["Authorization"] for r in srv.requests], ["Bearer stale", "Bearer fresh"])
        self.assertEqual(auth.refreshes, 1)
        srv.clear()
        with self.assertRaises(errors.GatewayError) as cm:
            self.run_turn(make_ctx(make_req([user("hi")]), prov, auth=FakeAuth(["stale"], style="bearer")))
        self.assertEqual((cm.exception.status, cm.exception.err_type), (401, "authentication_error"))
        self.assertIn("gcloud auth application-default login", cm.exception.message)
        self.assertEqual(len(srv.requests), 1)

    def test_invalid_api_key_is_unauthorized(self):
        srv = self.start("gemini_api", api_key="good")
        prov = provider(base_url=srv.url)
        with self.assertRaises(errors.GatewayError) as cm:
            self.run_turn(make_ctx(make_req([user("hi")]), prov, auth=FakeAuth(["bad"])))
        self.assertEqual((cm.exception.status, cm.exception.err_type), (401, "authentication_error"))
        auth = FakeAuth(["bad", "good"])
        events = self.run_turn(make_ctx(make_req([user("hi")]), prov, auth=auth))
        self.assertEqual(events[-1], ev.Finish("end_turn"))
        self.assertEqual(auth.refreshes, 1)

    def test_midstream_error_after_commit(self):
        srv = self.start("gemini_api", mode="midstream_error")
        events = self.run_turn(make_ctx(make_req([user("go")], tools=[BASH]), provider(base_url=srv.url)))
        self.assertEqual(events[0], ev.ThinkingDelta(1, srv.brain.thinking))
        self.assertEqual(events[-1], ev.StreamError("overloaded_error", events[-1].message, True))
        self.assertEqual(len(events), 2)

    def test_not_found_and_background_structured_output(self):
        srv = self.start("gemini_api", known_models=["gemini-3.8-flash"])
        with self.assertRaises(errors.GatewayError) as cm:
            self.run_turn(make_ctx(make_req([user("hi")]), provider(base_url=srv.url)))
        self.assertEqual((cm.exception.status, cm.exception.err_type), (404, "not_found_error"))
        req = make_req([user("Summarize this session")], effort="low", thinking_requested=False,
                       output_format={"type": "json_schema", "schema": {
                           "type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"],
                           "additionalProperties": False}})
        events = self.run_turn(make_ctx(req, provider(base_url=srv.url), "gemini-3.8-flash"))
        self.assertEqual("".join(e.text for e in events if isinstance(e, ev.TextDelta)), srv.brain.background_text)
        gen = srv.requests[-1]["body_json"]["generationConfig"]
        self.assertEqual(gen["thinkingConfig"]["thinkingLevel"], "low")
        self.assertIn("responseJsonSchema", gen)

    def test_registry_and_generator_close(self):
        self.assertIsInstance(dialects.get_dialect("gemini"), gem.GeminiDialect)
        srv = self.start("gemini_api")
        gen = gem.GeminiDialect().execute(make_ctx(make_req([user("go")], tools=[BASH]), provider(base_url=srv.url)))
        self.assertIsInstance(next(gen), ev.ThinkingDelta)
        gen.close()  # client disconnect: must not raise, upstream released in finally
        self.assertEqual(srv.brain.leaks, [])


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
