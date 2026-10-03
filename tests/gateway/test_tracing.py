"""tracing: redaction, file-only logging (never stdout/stderr), JSONL tracer."""

import contextlib
import io
import json
import logging
import os
import shutil
import stat
import tempfile
import threading
import unittest

from ._pkg import mod

tracing = mod("tracing")
config = mod("config")
testing = mod("testing")

SENTINEL = testing.SENTINEL


class RedactTests(unittest.TestCase):
    def test_secret_store_values(self):
        store = config.SecretStore({"xai": "abc-" + SENTINEL, "short": "ab"})
        out = tracing.redact("key=abc-%s and ab" % SENTINEL, store)
        self.assertNotIn(SENTINEL, out)
        self.assertIn("ab", out)  # values shorter than 4 chars are not redacted
        self.assertEqual(tracing.redact("x " + SENTINEL + "-zz", [SENTINEL + "-zz"]), "x ***")

    def test_generic_patterns(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlLXZhbHVl"
        cases = [
            "sk-proj-AbCdEf0123456789", "sk-ant-api03-xyzXYZ0123", "sk-or-v1-0123456789abcdef",
            "xai-0123456789abcdefABCDEF", "nvapi-ZYXWVU9876543210",
            "AIzaSyA0123456789abcdefghijklmnopqrstu", "ya29.a0AfH6SMBx0123456789abcdef",
            jwt,
        ]
        for secret in cases:
            out = tracing.redact("before %s after" % secret)
            self.assertNotIn(secret, out, secret)
            self.assertTrue(out.startswith("before ") and out.endswith(" after"), out)
        self.assertEqual(tracing.redact("Authorization: Bearer abcdef123456"), "Authorization: *** ***")
        self.assertEqual(tracing.redact("bearer tok_0123456789"), "bearer ***")
        self.assertEqual(tracing.redact('{"api_key": "plainvalue1", "refresh_token": "rt_1234"}'),
                         '{"api_key": "***", "refresh_token": "***"}')
        self.assertEqual(tracing.redact("https://x/y?key=abc123&alt=sse"), "https://x/y?key=***&alt=sse")
        self.assertEqual(tracing.redact("FOO_API_KEY=abcd1234"), "FOO_API_KEY=***")

    def test_no_false_positives(self):
        text = ('{"input_tokens": 1234, "output_tokens": 5, "paid_tokens": 7, "max_tokens": 32000, '
                '"model": "claude-via-xai,grok-4.7", "path": "/v1/messages?beta=true", "skills": "sk-a"}')
        self.assertEqual(tracing.redact(text), text)
        self.assertIsNone(tracing.redact(None))
        self.assertEqual(tracing.redact(42), "42")

    def test_redact_obj(self):
        store = config.SecretStore({"k": SENTINEL + "x"})
        obj = {"a": [SENTINEL + "x", {"b": "Bearer zzzzzzzz"}], "n": 3, "t": ("sk-0123456789",)}
        out = tracing.redact_obj(obj, store)
        self.assertEqual(out, {"a": ["***", {"b": "Bearer ***"}], "n": 3, "t": ["sk-***"]})


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gw-tracing-")
        self.logger = logging.getLogger(tracing.LOGGER_NAME)
        self.saved = (list(self.logger.handlers), self.logger.level, self.logger.propagate)

    def tearDown(self):
        for h in list(self.logger.handlers):
            if h not in self.saved[0]:
                self.logger.removeHandler(h)
                h.close()
        self.logger.setLevel(self.saved[1])
        self.logger.propagate = self.saved[2]
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_logger_is_silent(self):
        self.assertFalse(self.logger.propagate)
        self.assertTrue(any(isinstance(h, logging.NullHandler) for h in self.logger.handlers))
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            self.logger.error("this must not reach the terminal")
            self.logger.warning("nor this")
        self.assertEqual((err.getvalue(), out.getvalue()), ("", ""))

    def test_file_logging_redacted_and_rotating(self):
        path = os.path.join(self.tmp, "sub", "gateway.log")
        store = config.SecretStore({"xai": "xyz-" + SENTINEL})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            logger = tracing.setup_logging("DEBUG", path, store, max_bytes=2000, backup_count=2)
            self.assertIs(logger, self.logger)
            logger.info("token %s used", "xyz-" + SENTINEL)
            try:
                raise ValueError("boom with xyz-" + SENTINEL)
            except ValueError:
                logger.exception("failed: Bearer abcdefghijkl")
            for i in range(200):
                logger.debug("filler line %d %s", i, "x" * 40)
        self.assertEqual(err.getvalue(), "")
        files = sorted(os.listdir(os.path.dirname(path)))
        self.assertIn("gateway.log", files)
        self.assertIn("gateway.log.1", files)
        self.assertLessEqual(len(files), 3)
        blob = ""
        for f in files:
            with open(os.path.join(os.path.dirname(path), f), encoding="utf-8") as fh:
                blob += fh.read()
        self.assertNotIn(SENTINEL, blob)
        self.assertNotIn("abcdefghijkl", blob)
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode) & 0o077, 0)

    def test_setup_replaces_own_handler_and_null_without_file(self):
        path = os.path.join(self.tmp, "a.log")
        tracing.setup_logging("INFO", path)
        tracing.setup_logging("WARNING", path)
        ours = [h for h in self.logger.handlers if getattr(h, "_ai_gateway_handler", False)]
        self.assertEqual(len(ours), 1)
        self.assertEqual(self.logger.level, logging.WARNING)
        tracing.setup_logging("bogus-level")
        ours = [h for h in self.logger.handlers if getattr(h, "_ai_gateway_handler", False)]
        self.assertIsInstance(ours[0], logging.NullHandler)
        self.assertEqual(self.logger.level, logging.INFO)


class TracerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gw-trace-")
        self.path = os.path.join(self.tmp, "trace.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def lines(self):
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_jsonl_redacted(self):
        store = config.SecretStore({"k": 'q"uote\\' + SENTINEL})
        tr = tracing.Tracer(self.path, store, bodies=False)
        tr("request", {"status": 200, "secret_echo": 'q"uote\\' + SENTINEL, "obj": object(),
                       "hdr": "Bearer abcdefgh1234"})
        tr("plain")
        recs = self.lines()
        self.assertEqual([r["event"] for r in recs], ["request", "plain"])
        self.assertEqual(recs[0]["status"], 200)
        self.assertEqual(recs[0]["secret_echo"], "***")
        self.assertEqual(recs[0]["hdr"], "Bearer ***")
        self.assertRegex(recs[0]["ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
        with open(self.path, encoding="utf-8") as f:
            self.assertNotIn(SENTINEL, f.read())
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode) & 0o077, 0)

    def test_bodies_flag_from_env(self):
        self.assertTrue(tracing.trace_bodies_enabled({"AI_GATEWAY_TRACE_BODIES": "1"}))
        self.assertFalse(tracing.trace_bodies_enabled({"AI_GATEWAY_TRACE_BODIES": "0"}))
        self.assertFalse(tracing.trace_bodies_enabled({}))
        self.assertEqual(tracing.trace_file_from_env({"AI_GATEWAY_TRACE_FILE": "/x"}), "/x")
        self.assertIsNone(tracing.trace_file_from_env({}))
        self.assertTrue(tracing.Tracer(self.path, bodies=True).bodies)

    def test_thread_safe_appends_and_never_raises(self):
        tr = tracing.Tracer(self.path)

        def worker(i):
            for j in range(50):
                tr("e", {"i": i, "j": j, "pad": "x" * 500})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(len(self.lines()), 400)
        bad = tracing.Tracer(os.path.join(self.tmp, "dir-as-file"))
        os.remove(bad.path)
        os.mkdir(bad.path)
        bad("e", {"x": 1})  # OSError swallowed


if __name__ == "__main__":
    unittest.main()
