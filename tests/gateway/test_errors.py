"""errors.py: GatewayError, map_upstream_error (DESIGN §2.4 table), map_error_code, prompt_too_long."""

import json
import re
import time
import unittest

from ._pkg import mod

errors = mod("errors")
events = mod("events")
compat = mod("compat")

CC_REGEX = re.compile(r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)")  # Claude Code's own regex


def m(status, body, headers=None, **kw):
    if not isinstance(body, str):
        body = json.dumps(body)
    return errors.map_upstream_error(status, body, headers or {}, "prov", "mdl", **kw)


class GatewayErrorTests(unittest.TestCase):
    def test_body_headers_stream_error(self):
        e = errors.GatewayError(429, "rate_limit_error", "slow down", True, 1.2)
        self.assertEqual(e.body(), {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
        self.assertEqual(e.headers(), {"x-should-retry": "true", "retry-after": "2", "retry-after-ms": "1200"})
        self.assertEqual(str(e), "slow down")
        self.assertIn("429", repr(e))
        self.assertEqual(e.to_stream_error(), events.StreamError("overloaded_error", "slow down", True))
        t = errors.GatewayError(400, "invalid_request_error", "bad")
        self.assertEqual(t.headers(), {"x-should-retry": "false"})
        self.assertEqual(t.to_stream_error(), events.StreamError("invalid_request_error", "bad", False))
        self.assertTrue(isinstance(t, Exception))

    def test_prompt_too_long(self):
        e = errors.prompt_too_long(1500, 1000)
        self.assertEqual((e.status, e.err_type, e.should_retry), (400, "invalid_request_error", False))
        self.assertEqual(e.message, "prompt is too long: 1500 tokens > 1000 maximum")
        self.assertEqual(CC_REGEX.search(e.message).groups(), ("1500", "1000"))
        self.assertEqual(errors.prompt_too_long(5, 10).message, "prompt is too long: 11 tokens > 10 maximum")


class MapUpstreamErrorTests(unittest.TestCase):
    def test_connection_error(self):
        e = m(None, "connection refused")
        self.assertEqual((e.status, e.err_type, e.should_retry, e.connection_error), (502, "api_error", True, True))
        self.assertIn("[prov/mdl]", e.message)
        self.assertIsNone(e.upstream_status)

    def test_context_overflow_variants(self):
        cases = [
            ({"error": {"message": "This model's maximum context length is 131072 tokens. However, you requested "
                                   "140000 tokens (139000 in the messages, 1000 in the completion).",
                        "code": "context_length_exceeded"}}, (140000, 131072)),
            ({"type": "error", "error": {"type": "invalid_request_error",
                                         "message": "prompt is too long: 210000 tokens > 200000 maximum"}},
             (210000, 200000)),
            ({"error": {"code": 400, "status": "INVALID_ARGUMENT",
                        "message": "The input token count (1200000) exceeds the maximum number of tokens allowed "
                                   "(1048576)."}}, (1200000, 1048576)),
            ({"code": "Client specified an invalid argument",
              "error": "This model's maximum prompt length is 131072 but the request contains 200000 tokens."},
             (200000, 131072)),
            ({"detail": "Your input exceeds the context window of this model. Please adjust your input and try "
                        "again."}, (5000, 4000)),
            ({"error": {"message": "too many tokens", "code": "context_length_exceeded"}}, (5000, 4000)),
        ]
        for body, (n, mx) in cases:
            e = m(400, body, context_window=4000, est_tokens=5000)
            self.assertEqual((e.status, e.err_type, e.should_retry), (400, "invalid_request_error", False), body)
            self.assertEqual(e.message, "prompt is too long: %d tokens > %d maximum" % (n, mx), body)
            self.assertEqual(CC_REGEX.search(e.message).groups(), (str(n), str(mx)))
        # 413 + overflow text also maps to prompt-too-long; nothing known -> still well-formed
        e = m(413, {"error": {"message": "exceeds the context window"}})
        self.assertTrue(CC_REGEX.search(e.message))
        n, mx = (int(x) for x in CC_REGEX.search(e.message).groups())
        self.assertGreater(n, mx)
        e = m(400, {"error": {"message": "prompt is too long"}}, est_tokens=300)
        self.assertEqual(e.message, "prompt is too long: 300 tokens > 299 maximum")

    def test_status_table(self):
        e = m(400, {"error": {"message": "Unsupported parameter: 'max_tokens'", "type": "invalid_request_error"}})
        self.assertEqual((e.status, e.err_type, e.should_retry), (400, "invalid_request_error", False))
        self.assertEqual(e.message, "[prov/mdl] Unsupported parameter: 'max_tokens'")
        self.assertEqual(m(422, {"detail": [{"msg": "field required"}]}).message, "[prov/mdl] field required")
        e = m(401, {"error": {"message": "invalid api key"}}, auth_hint="run `codex login`")
        self.assertEqual((e.status, e.err_type, e.should_retry), (401, "authentication_error", False))
        self.assertIn("run `codex login`", e.message)
        self.assertIn("re-login", m(401, "nope").message)
        self.assertEqual((m(403, "forbidden").status, m(403, "forbidden").err_type), (403, "permission_error"))
        e = m(404, {"error": {"message": "model not found"}})
        self.assertEqual((e.status, e.err_type, e.should_retry), (404, "not_found_error", False))
        self.assertEqual((m(413, "too big").status, m(413, "too big").err_type), (413, "request_too_large"))
        for st in (500, 502, 504, 408, 409):
            e = m(st, "<html><body><h1>Bad Gateway</h1></body></html>")
            self.assertEqual((e.status, e.err_type, e.should_retry), (502, "api_error", True), st)
            self.assertNotIn("<html", e.message)
        for st, body in ((503, "unavailable"), (529, {"type": "error", "error": {"type": "overloaded_error",
                                                                                  "message": "Overloaded"}}),
                         (500, {"error": {"message": "The server is overloaded"}})):
            e = m(st, body, {"retry-after": "3"})
            self.assertEqual((e.status, e.err_type, e.should_retry, e.retry_after), (529, "overloaded_error", True, 3.0))
        e = m(402, {"error": {"message": "Insufficient Balance"}})
        self.assertEqual((e.status, e.err_type, e.should_retry), (429, "rate_limit_error", False))
        e = m(418, "teapot")
        self.assertEqual((e.status, e.err_type), (400, "invalid_request_error"))
        self.assertEqual(e.upstream_status, 418)
        self.assertEqual(e.upstream_body, "teapot")

    def test_429_transient(self):
        e = m(429, {"error": {"message": "Rate limit reached", "type": "rate_limit_exceeded"}},
              {"Retry-After-Ms": "1500"})
        self.assertEqual((e.status, e.err_type, e.should_retry, e.retry_after), (429, "rate_limit_error", True, 1.5))
        self.assertEqual(e.headers()["x-should-retry"], "true")
        self.assertEqual(e.headers()["retry-after"], "2")
        e = m(429, "slow", {"retry-after": "7"})
        self.assertEqual(e.retry_after, 7.0)
        date = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(time.time() + 30))
        self.assertAlmostEqual(m(429, "slow", {"Retry-After": date}).retry_after, 30, delta=3)
        g = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Resource has been exhausted",
                       "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                    "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content",
                                                    "quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]},
                                   {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}]}}
        e = m(429, g)
        self.assertEqual((e.should_retry, e.retry_after), (True, 34.0))
        self.assertEqual(m(429, [g]).retry_after, 34.0)  # Gemini stream error arrays

    def test_429_terminal(self):
        resets = int(time.time()) + 3600
        cases = [
            {"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached",
                       "plan_type": "plus", "resets_at": resets}},
            {"detail": {"code": "usage_limit_reached", "message": "limit", "resets_in_seconds": 3600}},
            {"error": {"message": "You exceeded your current quota, please check your plan and billing details.",
                       "type": "insufficient_quota", "code": "insufficient_quota"}},
            {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded",
                       "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                    "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
                                   {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "20s"}]}},
            {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "slow down",
                       "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "900s"}]}},
            {"error": {"code": 429, "message": "x", "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                                                  "reason": "QUOTA_EXHAUSTED"}]}},
            {"code": "Some resource has been exhausted",
             "error": "Your team abc has either used all available credits or reached its monthly spending limit."},
        ]
        for body in cases:
            e = m(429, body, {"retry-after": "5"})
            self.assertEqual((e.status, e.err_type, e.should_retry), (429, "rate_limit_error", False), body)
            self.assertEqual(e.headers()["x-should-retry"], "false")
            self.assertIsNone(e.retry_after)
            self.assertIn("quota exhausted", e.message)
        e = m(429, cases[0])
        self.assertIn("resets at %s" % compat.rfc3339_format(resets, "seconds"), e.message)
        self.assertIn("resets at", m(429, cases[1]).message)

    def test_map_error_code(self):
        e = errors.map_error_code("context_length_exceeded", "Your input exceeds the context window of this model.",
                                  "codex", "gpt-5.5", context_window=400000, est_tokens=410000)
        self.assertEqual(e.message, "prompt is too long: 410000 tokens > 400000 maximum")
        e = errors.map_error_code("rate_limit_exceeded", "Rate limit reached", "codex", "gpt-5.5")
        self.assertEqual((e.status, e.should_retry), (429, True))
        e = errors.map_error_code("usage_limit_reached", "limit", "codex", "gpt-5.5",
                                  extra={"resets_at": int(time.time()) + 60})
        self.assertEqual((e.status, e.should_retry), (429, False))
        self.assertIn("resets at", e.message)
        e = errors.map_error_code("server_error", "oops", "codex", "gpt-5.5")
        self.assertEqual((e.status, e.err_type, e.should_retry), (529, "overloaded_error", True))
        e = errors.map_error_code("something_new", "weird", "codex", "gpt-5.5")
        self.assertEqual((e.status, e.err_type, e.should_retry), (502, "api_error", True))
        e = errors.map_error_code(None, "maximum context length is 10 tokens, you requested 20", "p", "m")
        self.assertEqual(e.message, "prompt is too long: 20 tokens > 10 maximum")

    def test_parse_retry_after(self):
        self.assertIsNone(errors.parse_retry_after({}))
        self.assertEqual(errors.parse_retry_after({"retry-after": "abc"}), None)
        self.assertEqual(errors.parse_retry_after({"retry-after-ms": "250", "retry-after": "9"}), 0.25)


if __name__ == "__main__":
    unittest.main()
