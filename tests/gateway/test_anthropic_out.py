"""anthropic_out: SSE grammar (property test over random event interleavings), Aggregator == SSE
replay, error responses and the token estimate."""

import json
import math
import random
import re
import unittest

from ._pkg import mod

ao = mod("anthropic_out")
ev = mod("events")
errors = mod("errors")
model = mod("model")
signatures = mod("signatures")

MODEL = "claude-via-xai,grok-4.7"


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def parse_sse(raw):
    """bytes -> [(event, data_dict)]; asserts the exact ``event:``/``data:`` framing."""
    text = raw.decode("utf-8")
    assert text.endswith("\n\n"), "stream must end with a blank line"
    out = []
    for frame in text[:-2].split("\n\n"):
        lines = frame.split("\n")
        assert len(lines) == 2, frame
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), frame
        name = lines[0][len("event: "):]
        data = json.loads(lines[1][len("data: "):])
        assert data["type"] == name, frame
        out.append((name, data))
    return out


class Collector(object):
    def __init__(self):
        self.chunks = []

    def __call__(self, data):
        assert isinstance(data, bytes)
        self.chunks.append(data)

    def events(self):
        return parse_sse(b"".join(self.chunks))


def run_emitter(events, message_id="msg_000000000000000000000001", est=7, pings=()):
    col = Collector()
    em = ao.SSEEmitter(col, MODEL, message_id, est)
    for i, e in enumerate(events):
        if i in pings:
            em.ping()
        em.feed(e)
    em.finish()
    return em, col


def run_aggregator(events, message_id="msg_000000000000000000000001", est=7):
    agg = ao.Aggregator(MODEL, message_id, est)
    for e in events:
        agg.feed(e)
    return agg


def check_grammar(tc, frames, model_id=MODEL):
    """Assert the Anthropic SSE grammar; returns the replayed Message (or the error dict)."""
    if frames and frames[0][0] == "error":  # error before anything was committed
        tc.assertEqual(len(frames), 1)
        return {"error": frames[0][1]["error"]}
    tc.assertGreaterEqual(len(frames), 2)
    tc.assertEqual(frames[0][0], "message_start")
    msg = frames[0][1]["message"]
    tc.assertEqual(msg["model"], model_id)
    tc.assertRegex(msg["id"], r"^msg_")
    tc.assertEqual((msg["type"], msg["role"], msg["content"], msg["stop_reason"], msg["stop_sequence"]),
                   ("message", "assistant", [], None, None))
    tc.assertEqual(set(msg["usage"]), {"input_tokens", "output_tokens", "cache_creation_input_tokens",
                                       "cache_read_input_tokens"})
    tc.assertEqual(frames[1][0], "ping")
    replay = dict(msg)
    blocks = []
    open_idx = None
    open_type = None
    deltas = []
    finished = False
    for i, (name, data) in enumerate(frames[1:], 1):
        tc.assertFalse(finished, "event after message_stop")
        if name == "ping":
            continue
        if name == "error":
            tc.assertEqual(i, len(frames) - 1, "error must be the last event")
            tc.assertEqual(set(data["error"]), {"type", "message"})
            return {"error": data["error"]}
        if name == "content_block_start":
            tc.assertIsNone(open_idx, "overlapping blocks")
            tc.assertEqual(data["index"], len(blocks))
            cb = dict(data["content_block"])
            tc.assertIn(cb["type"], ("text", "thinking", "tool_use"))
            if cb["type"] == "text":
                tc.assertEqual(cb, {"type": "text", "text": ""})
            elif cb["type"] == "thinking":
                tc.assertEqual(cb, {"type": "thinking", "thinking": "", "signature": ""})
            else:
                tc.assertEqual(cb["input"], {})
                tc.assertTrue(cb["id"])
            open_idx, open_type, deltas = data["index"], cb["type"], []
            blocks.append(cb)
        elif name == "content_block_delta":
            tc.assertEqual(data["index"], open_idx)
            d = data["delta"]
            deltas.append(d["type"])
            cb = blocks[open_idx]
            if open_type == "text":
                tc.assertEqual(d["type"], "text_delta")
                tc.assertTrue(d["text"])
                cb["text"] += d["text"]
            elif open_type == "thinking":
                tc.assertIn(d["type"], ("thinking_delta", "signature_delta"))
                if d["type"] == "thinking_delta":
                    tc.assertNotIn("signature_delta", deltas[:-1], "thinking after signature")
                    cb["thinking"] += d["thinking"]
                else:
                    tc.assertTrue(d["signature"])
                    cb["signature"] = d["signature"]
            else:
                tc.assertEqual(d["type"], "input_json_delta")
                cb["input"] = json.loads(d["partial_json"])
                tc.assertIsInstance(cb["input"], dict)
        elif name == "content_block_stop":
            tc.assertEqual(data["index"], open_idx)
            if open_type == "thinking":
                tc.assertEqual(deltas.count("signature_delta"), 1)
                tc.assertEqual(deltas[-1], "signature_delta", "signature_delta must precede the stop")
            elif open_type == "text":
                tc.assertTrue(deltas, "empty text block opened")
            else:
                tc.assertEqual(deltas, ["input_json_delta"])
            open_idx = open_type = None
        elif name == "message_delta":
            tc.assertIsNone(open_idx, "message_delta with an open block")
            tc.assertIn(data["delta"]["stop_reason"], ev.STOP_REASONS)
            tc.assertIn("stop_sequence", data["delta"])
            u = data["usage"]
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                tc.assertIsInstance(u[k], int)
            replay.update(data["delta"])
            replay["usage"] = u
        elif name == "message_stop":
            tc.assertEqual(i, len(frames) - 1, "message_stop must be last")
            tc.assertIsNotNone(replay["stop_reason"], "message_stop without message_delta")
            finished = True
        else:
            tc.fail("unknown event %r" % name)
    tc.assertTrue(finished, "stream not terminated")
    replay["content"] = blocks
    if any(b["type"] == "tool_use" for b in blocks):
        tc.assertEqual(replay["stop_reason"], "tool_use")
    return replay


def _norm_ids(message):
    """Generated tool ids are random per state machine; compare them by shape only."""
    for b in message["content"]:
        if b["type"] == "tool_use" and re.match(r"^toolu_[0-9a-f]{24}$", b["id"]):
            b["id"] = "<generated>"
    return message


def random_events(rng, allow_error=True):
    out = []
    keys = [0, 1, "k"]
    for _ in range(rng.randint(0, 25)):
        r = rng.random()
        key = rng.choice(keys)
        if r < 0.3:
            out.append(ev.TextDelta(key, rng.choice(["", "a", "héllo ", "\n", "x" * rng.randint(1, 5), "€😀"])))
        elif r < 0.5:
            out.append(ev.ThinkingDelta(key, rng.choice(["", "t", "think more"])))
        elif r < 0.6:
            out.append(ev.ThinkingSignature(key, rng.choice(["", "sig-%d" % rng.randint(0, 9),
                                                             "fgw1.chat.0123456789ab"])))
        elif r < 0.75:
            out.append(ev.ToolCall(rng.choice(["toolu_%d" % rng.randint(0, 99), ""]), rng.choice(["Bash", "Read"]),
                                   rng.choice(['{"command":"ls"}', "{}", "", "not json", "[1,2]",
                                               '{"a": {"b": [1, "é"]}}'])))
        elif r < 0.85:
            out.append(ev.Usage(rng.randint(0, 50), rng.randint(0, 50), rng.randint(0, 5)))
        elif r < 0.95:
            out.append(ev.Finish(rng.choice(["end_turn", "max_tokens", "tool_use", "stop_sequence", None, "weird"]),
                                 rng.choice([None, "STOP"])))
        elif allow_error:
            out.append(ev.StreamError(rng.choice(["overloaded_error", "api_error", "rate_limit_error"]), "boom",
                                      rng.random() < 0.5))
    return out


# ---------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------

class ReexportTests(unittest.TestCase):
    def test_reexports_are_the_errors_objects(self):
        self.assertIs(ao.GatewayError, errors.GatewayError)
        self.assertIs(ao.map_upstream_error, errors.map_upstream_error)
        self.assertIs(ao.prompt_too_long, errors.prompt_too_long)

    def test_ids(self):
        ids = {ao.new_message_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)
        for i in ids:
            self.assertRegex(i, r"^msg_[0-9a-f]{24}$")
        self.assertRegex(ao.new_tool_id(), r"^toolu_[0-9a-f]{24}$")


class SSEGrammarPropertyTests(unittest.TestCase):
    def test_random_interleavings(self):
        rng = random.Random(20261003)
        for case in range(600):
            events = random_events(rng)
            pings = set(rng.sample(range(len(events) + 1), min(3, len(events) + 1)))
            em, col = run_emitter(events, pings=pings)
            frames = col.events()
            replay = check_grammar(self, frames)
            self.assertTrue(em.committed)
            self.assertTrue(em.done)
            agg = run_aggregator(events)
            if "error" in replay:
                with self.assertRaises(errors.GatewayError) as cm:
                    agg.result()
                self.assertEqual(cm.exception.err_type, replay["error"]["type"])
                continue
            self.assertEqual(_norm_ids(agg.result()), _norm_ids(replay), "case %d: %r" % (case, events))
            # semantic oracle: all non-empty text survives in order, one tool_use per ToolCall
            texts = "".join(e.text for e in events if isinstance(e, ev.TextDelta))
            self.assertEqual("".join(b.get("text", "") for b in replay["content"]), texts)
            calls = [e for e in events if isinstance(e, ev.ToolCall)]
            self.assertEqual(sum(1 for b in replay["content"] if b["type"] == "tool_use"), len(calls))
            self.assertEqual(replay["stop_reason"] == "tool_use", bool(calls))

    def test_every_chunk_is_one_event(self):
        em, col = run_emitter([ev.TextDelta(0, "a"), ev.ToolCall("t", "Bash", "{}")])
        for chunk in col.chunks:
            self.assertEqual(len(parse_sse(chunk)), 1)
        self.assertEqual(em.events_written, len(col.chunks))


class SSEEmitterTests(unittest.TestCase):
    def test_message_start_exact(self):
        col = Collector()
        em = ao.SSEEmitter(col, MODEL, "msg_abc", 42)
        self.assertFalse(em.committed)
        em.start()
        em.start()
        self.assertTrue(em.committed)
        frames = col.events()
        self.assertEqual(frames, [
            ("message_start", {"type": "message_start", "message": {
                "id": "msg_abc", "type": "message", "role": "assistant", "model": MODEL, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 42, "output_tokens": 0, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0}}}),
            ("ping", {"type": "ping"}),
        ])
        self.assertEqual(col.chunks[0][:21], b"event: message_start\n")

    def test_thinking_signatures(self):
        _, col = run_emitter([ev.ThinkingDelta("r", "abc"), ev.ThinkingDelta("r", "def"), ev.TextDelta(0, "hi")])
        frames = col.events()
        sig = [d["delta"]["signature"] for n, d in frames
               if n == "content_block_delta" and d["delta"]["type"] == "signature_delta"]
        self.assertEqual(sig, [signatures.synthetic_signature("abcdef")])
        self.assertRegex(sig[0], r"^fgw1\.chat\.[0-9a-f]{12}$")
        # explicit signature (last wins), emitted once right before the stop
        _, col = run_emitter([ev.ThinkingDelta(1, "x"), ev.ThinkingSignature(1, "s1"), ev.ThinkingSignature(1, "s2"),
                              ev.ToolCall("toolu_1", "Bash", '{"command":"ls"}')])
        replay = check_grammar(self, col.events())
        self.assertEqual(replay["content"][0], {"type": "thinking", "thinking": "x", "signature": "s2"})
        self.assertEqual(replay["content"][1]["input"], {"command": "ls"})
        self.assertEqual(replay["stop_reason"], "tool_use")

    def test_signature_for_unopened_key_opens_empty_thinking(self):
        _, col = run_emitter([ev.TextDelta(0, "a"), ev.ThinkingSignature(5, "fgw1.openai_api.e30")])
        replay = check_grammar(self, col.events())
        self.assertEqual(replay["content"], [{"type": "text", "text": "a"},
                                             {"type": "thinking", "thinking": "", "signature": "fgw1.openai_api.e30"}])

    def test_empty_text_never_opens_and_does_not_close(self):
        _, col = run_emitter([ev.ThinkingDelta(0, "t"), ev.TextDelta(1, ""), ev.ThinkingDelta(0, "u")])
        replay = check_grammar(self, col.events())
        self.assertEqual(len(replay["content"]), 1)
        self.assertEqual(replay["content"][0]["thinking"], "tu")
        _, col = run_emitter([ev.TextDelta(0, "")])
        self.assertEqual(check_grammar(self, col.events())["content"], [])

    def test_key_change_and_kind_change_open_new_blocks(self):
        _, col = run_emitter([ev.TextDelta(0, "a"), ev.TextDelta(0, "b"), ev.TextDelta(1, "c"),
                              ev.ThinkingDelta(1, "d"), ev.TextDelta(1, "e")])
        replay = check_grammar(self, col.events())
        self.assertEqual([(b["type"], b.get("text", b.get("thinking"))) for b in replay["content"]],
                         [("text", "ab"), ("text", "c"), ("thinking", "d"), ("text", "e")])

    def test_tool_call_normalization(self):
        _, col = run_emitter([ev.ToolCall("", "Bash", "not json"), ev.ToolCall("toolu_x", "Read", " [1] "),
                              ev.ToolCall("toolu_y", "Edit", {"a": 1})])
        frames = col.events()
        partial = [d["delta"]["partial_json"] for n, d in frames if n == "content_block_delta"]
        self.assertEqual(partial, ["{}", "{}", '{"a":1}'])
        starts = [d["content_block"] for n, d in frames if n == "content_block_start"]
        self.assertRegex(starts[0]["id"], r"^toolu_[0-9a-f]{24}$")
        self.assertEqual(starts[1]["id"], "toolu_x")
        # full args string passes through verbatim when it is a JSON object
        _, col = run_emitter([ev.ToolCall("t", "Bash", '{"command": "printf hello > out.txt"}')])
        partial = [d["delta"]["partial_json"] for n, d in col.events() if n == "content_block_delta"]
        self.assertEqual(partial, ['{"command": "printf hello > out.txt"}'])

    def test_stop_reasons(self):
        def stop(events):
            frames = run_emitter(events)[1].events()
            md = [d for n, d in frames if n == "message_delta"][0]
            return md["delta"]["stop_reason"], md["delta"]["stop_sequence"]

        self.assertEqual(stop([]), ("end_turn", None))
        self.assertEqual(stop([ev.TextDelta(0, "a")]), ("end_turn", None))
        self.assertEqual(stop([ev.Finish("max_tokens")]), ("max_tokens", None))
        self.assertEqual(stop([ev.Finish("tool_use")]), ("end_turn", None))
        self.assertEqual(stop([ev.Finish("stop_sequence", "END")]), ("stop_sequence", "END"))
        self.assertEqual(stop([ev.Finish("end_turn", "END")]), ("end_turn", None))
        self.assertEqual(stop([ev.Finish("refusal")]), ("refusal", None))
        self.assertEqual(stop([ev.Finish("bogus")]), ("end_turn", None))
        self.assertEqual(stop([ev.ToolCall("t", "Bash", "{}"), ev.Finish("end_turn")]), ("tool_use", None))

    def test_usage(self):
        def usage(events, est=10):
            frames = run_emitter(events, est=est)[1].events()
            return [d for n, d in frames if n == "message_delta"][0]["usage"]

        self.assertEqual(usage([ev.Usage(100, 20, 30, 4), ev.Usage(120, 25, 30, 0)]),
                         {"input_tokens": 120, "output_tokens": 25, "cache_read_input_tokens": 30,
                          "cache_creation_input_tokens": 0})
        u = usage([ev.TextDelta(0, "x" * 36)], est=11)
        self.assertEqual((u["input_tokens"], u["output_tokens"]), (11, 10))
        u = usage([ev.TextDelta(0, "abc"), ev.Usage(0, 0)], est=5)
        self.assertEqual((u["input_tokens"], u["output_tokens"]), (5, 1))

    def test_finish_idempotent_and_feed_after_finish_ignored(self):
        col = Collector()
        em = ao.SSEEmitter(col, MODEL)
        em.feed(ev.TextDelta(0, "a"))
        em.finish()
        n = len(col.chunks)
        em.finish()
        em.feed(ev.TextDelta(0, "b"))
        em.ping()
        em.error("api_error", "late")
        self.assertEqual(len(col.chunks), n)
        self.assertEqual(col.events()[-1][0], "message_stop")

    def test_error_event(self):
        col = Collector()
        em = ao.SSEEmitter(col, MODEL)
        em.feed(ev.TextDelta(0, "partial"))
        em.feed(ev.StreamError("overloaded_error", "upstream overloaded", True))
        em.finish()
        frames = col.events()
        self.assertEqual(frames[-1], ("error", {"type": "error", "error": {"type": "overloaded_error",
                                                                          "message": "upstream overloaded"}}))
        self.assertNotIn("message_stop", [n for n, _ in frames])
        self.assertTrue(em.done)
        self.assertEqual(em.stream_error.err_type, "overloaded_error")
        # error() before start writes only the error event
        col = Collector()
        em = ao.SSEEmitter(col, MODEL)
        em.error("api_error", "boom")
        self.assertEqual(col.events(), [("error", {"type": "error", "error": {"type": "api_error",
                                                                              "message": "boom"}})])

    def test_ping_before_start_commits(self):
        col = Collector()
        em = ao.SSEEmitter(col, MODEL)
        em.ping()
        em.ping()
        self.assertEqual([n for n, _ in col.events()], ["message_start", "ping", "ping"])

    def test_unicode_and_newlines_framing(self):
        _, col = run_emitter([ev.TextDelta(0, "line1\n\nline2   é 😀"), ev.TextDelta(0, "\ud800")])
        replay = check_grammar(self, col.events())
        self.assertTrue(replay["content"][0]["text"].startswith("line1\n\nline2   é 😀"))


class AggregatorTests(unittest.TestCase):
    def test_full_message(self):
        agg = run_aggregator([ev.ThinkingDelta(0, "plan"), ev.ThinkingSignature(0, "sig"),
                              ev.TextDelta(1, "I'll run it."),
                              ev.ToolCall("toolu_1", "Bash", '{"command":"ls"}'),
                              ev.Usage(50, 12, 5), ev.Finish("tool_use")], message_id="msg_x", est=3)
        self.assertEqual(agg.result(), {
            "id": "msg_x", "type": "message", "role": "assistant", "model": MODEL,
            "content": [{"type": "thinking", "thinking": "plan", "signature": "sig"},
                        {"type": "text", "text": "I'll run it."},
                        {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 50, "output_tokens": 12, "cache_read_input_tokens": 5,
                      "cache_creation_input_tokens": 0}})
        self.assertIs(agg.result(), agg.result())

    def test_empty_stream(self):
        res = run_aggregator([], est=9).result()
        self.assertEqual(res["content"], [])
        self.assertEqual(res["stop_reason"], "end_turn")
        self.assertEqual(res["usage"]["input_tokens"], 9)

    def test_stream_error_raises_equivalent_gateway_error(self):
        cases = [("overloaded_error", True, 529), ("rate_limit_error", False, 429), ("api_error", True, 502),
                 ("invalid_request_error", False, 400), ("authentication_error", False, 401),
                 ("not_found_error", False, 404), ("something_else", False, 502)]
        for err_type, retry, status in cases:
            agg = run_aggregator([ev.TextDelta(0, "x"), ev.StreamError(err_type, "msg " + err_type, retry),
                                  ev.TextDelta(0, "ignored")])
            with self.assertRaises(errors.GatewayError) as cm:
                agg.result()
            e = cm.exception
            self.assertEqual((e.status, e.err_type, e.message, e.should_retry),
                             (status, err_type, "msg " + err_type, retry))

    def test_prompt_too_long_stream_error_keeps_message(self):
        se = errors.prompt_too_long(300000, 200000).to_stream_error()
        agg = run_aggregator([se])
        with self.assertRaises(errors.GatewayError) as cm:
            agg.result()
        self.assertEqual(cm.exception.status, 400)
        self.assertRegex(cm.exception.message, r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)")


class ErrorResponseTests(unittest.TestCase):
    def test_gateway_error(self):
        err = errors.GatewayError(429, "rate_limit_error", "slow down", True, retry_after=1.5)
        status, headers, body = ao.error_response(err)
        self.assertEqual(status, 429)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["x-should-retry"], "true")
        self.assertEqual(headers["retry-after"], "2")
        self.assertEqual(headers["retry-after-ms"], "1500")
        self.assertEqual(json.loads(body.decode("utf-8")),
                         {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})

    def test_prompt_too_long_and_terminal(self):
        status, headers, body = ao.error_response(ao.prompt_too_long(250000, 200000))
        self.assertEqual((status, headers["x-should-retry"]), (400, "false"))
        self.assertIn("prompt is too long: 250000 tokens > 200000 maximum", body.decode("utf-8"))
        self.assertNotIn("retry-after", headers)

    def test_unexpected_exception(self):
        status, headers, body = ao.error_response(RuntimeError("secret detail"))
        self.assertEqual(status, 500)
        self.assertEqual(headers["x-should-retry"], "false")
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["error"]["type"], "api_error")
        self.assertNotIn("secret detail", data["error"]["message"])


class EstimateTests(unittest.TestCase):
    def test_formula(self):
        B = model.Block
        req = model.NormalizedRequest(
            model="m", system=["s" * 10],
            tools=[model.ToolDef("T", "desc", {"type": "object"})],
            messages=[model.Message("user", [B.of_text("é" * 5), B.of_image_base64("image/png", "AAAA"),
                                             B.of_document_base64("application/pdf", "A" * 4000),
                                             B.of_document_text("doc")]),
                      model.Message("assistant", [B.of_thinking("th"), B.of_tool_use("id", "Bash", {"a": 1})]),
                      model.Message("user", [B.of_tool_result("id", [B.of_text("out"),
                                                                     B.of_image_url("https://x")])])])
        text_bytes = 10 + (1 + 4 + len('{"type":"object"}')) + 10 + 3 + 2 + (4 + len('{"a":1}')) + 3
        expected = int(math.ceil(text_bytes / 3.6)) + 1600 * 2 + int(math.ceil(3000 / 750.0))
        self.assertEqual(ao.estimate_tokens(req), expected)

    def test_fixture_scale(self):
        ai = mod("anthropic_in")
        testing = mod("testing")
        fx = testing.load_fixture("turn1_request.json")
        req = ai.parse_messages_request(fx["body"], fx["headers"])
        raw_len = len(json.dumps(fx["body"]).encode("utf-8"))
        est = ao.estimate_tokens(req)
        self.assertTrue(raw_len / 8 < est < raw_len / 3, (raw_len, est))
        self.assertEqual(ao.estimate_tokens(model.NormalizedRequest(model="m")), 0)


if __name__ == "__main__":
    unittest.main()
