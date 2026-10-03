"""secrets.resolve_many / load_into: source order, env/literal/json/op, parallel memoized op reads,
signed-out and missing op handling, labels, and no secret values in warnings or logs."""

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from ._pkg import mod

secrets = mod("secrets")
config = mod("config")
fake_bins = mod("testing.fake_bins")

VALUE = "FAKEKEY-SENTINEL-op-value-123"


class SecretsTests(unittest.TestCase):
    def setUp(self):
        secrets.clear_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = os.path.join(self.tmp.name, "bin")
        self.refs = {"op://Private/xai/credential": VALUE, "op://Private/openai/credential": "sk-op-openai",
                     "op://Private/deepseek/credential": "ds-op", "op://Private/kimi/credential": "kimi-op"}
        fake_bins.make_fake_bins(self.bin, op_values=self.refs)
        self.env = {"PATH": self.bin, "XAI_API_KEY": "  xai-from-env \n", "EMPTY": "  "}
        self.cred = os.path.join(self.tmp.name, "credentials.json")
        with open(self.cred, "w", encoding="utf-8") as f:
            json.dump({"providers": {"xai": {"api_key": "xai-from-json"}, "list": ["a", "b"]}, "num": 5}, f)

    def tearDown(self):
        secrets.clear_cache()
        self.tmp.cleanup()

    def _op_calls(self):
        return fake_bins.read_calls(self.bin, "op")

    def test_cheap_sources_and_order(self):
        specs = {
            "env_first": ["env:MISSING", "env:EMPTY", "env:XAI_API_KEY", "literal:never"],
            "literal": ["literal:  lit-value "],
            "json": ["json:%s#providers.xai.api_key" % self.cred],
            "json_list": ["json:%s#providers.list.1" % self.cred],
            "json_missing_key": ["json:%s#providers.nope" % self.cred, "literal:fallback"],
            "json_non_string": ["json:%s#num" % self.cred],
            "json_missing_file": ["json:%s#a" % os.path.join(self.tmp.name, "nope.json")],
            "none": ["env:MISSING"],
            "empty": [],
        }
        warnings = []
        values = secrets.resolve_many(specs, environ=self.env, warnings=warnings)
        self.assertEqual(values, {"env_first": "xai-from-env", "literal": "lit-value", "json": "xai-from-json",
                                  "json_list": "b", "json_missing_key": "fallback", "json_non_string": None,
                                  "json_missing_file": None, "none": None, "empty": None})
        self.assertEqual(warnings, [])
        self.assertEqual(self._op_calls(), [], "no op call when no op source is needed")

    def test_json_tilde_expansion(self):
        home = self.tmp.name
        with mock.patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
            values = secrets.resolve_many({"k": ["json:~/credentials.json#providers.xai.api_key"]}, environ=self.env)
        self.assertEqual(values, {"k": "xai-from-json"})

    def test_malformed_sources_warn_without_values(self):
        bad = os.path.join(self.tmp.name, "bad.json")
        with open(bad, "w") as f:
            f.write("{not json")
        warnings = []
        values = secrets.resolve_many({"a": ["json:%s#x" % bad], "b": ["json:no-hash"], "c": ["raw-" + VALUE],
                                       "d": [123]}, environ=self.env, warnings=warnings)
        self.assertEqual(values, {"a": None, "b": None, "c": None, "d": None})
        self.assertEqual(len(warnings), 4, warnings)
        self.assertNotIn(VALUE, "\n".join(warnings))

    def test_op_parallel_and_memoized(self):
        fake_bins.set_behaviour(self.bin, op_delay=0.6)
        specs = {"xai": ["op://Private/xai/credential"], "openai": ["op://Private/openai/credential"],
                 "deepseek": ["op://Private/deepseek/credential"], "kimi": ["op://Private/kimi/credential"],
                 "env_wins": ["env:XAI_API_KEY", "op://Private/never/credential"]}
        t0 = time.monotonic()
        with self.assertLogs("ai_gateway", "DEBUG") as logs:
            secrets.LOG.debug("start")
            values = secrets.resolve_many(specs, timeout=10, environ=self.env)
        elapsed = time.monotonic() - t0
        self.assertEqual(values, {"xai": VALUE, "openai": "sk-op-openai", "deepseek": "ds-op", "kimi": "kimi-op",
                                  "env_wins": "xai-from-env"})
        self.assertLess(elapsed, 4 * 0.6, "op reads must run concurrently (took %.2fs)" % elapsed)
        calls = self._op_calls()
        self.assertEqual(sorted(c["argv"][-1] for c in calls), sorted(self.refs))
        for c in calls:
            self.assertEqual(c["argv"][0], "read")
        self.assertNotIn(VALUE, "\n".join(logs.output))
        # memoized per process
        again = secrets.resolve_many({"x": ["op://Private/xai/credential"]}, environ=self.env)
        self.assertEqual(again, {"x": VALUE})
        self.assertEqual(len(self._op_calls()), 4)
        secrets.clear_cache()
        secrets.resolve_many({"x": ["op://Private/xai/credential"]}, environ=self.env)
        self.assertEqual(len(self._op_calls()), 5)

    def test_op_order_before_env(self):
        values = secrets.resolve_many({"k": ["op://Private/xai/credential", "env:XAI_API_KEY"],
                                       "fallback": ["op://Private/missing/credential", "env:XAI_API_KEY"]},
                                      environ=self.env)
        self.assertEqual(values, {"k": VALUE, "fallback": "xai-from-env"})

    def test_op_not_signed_in_single_warning(self):
        fake_bins.set_behaviour(self.bin, op_signed_in=False)
        warnings = []
        specs = {n: ["op://Private/%s/credential" % n] for n in ("xai", "openai", "deepseek")}
        specs["env"] = ["op://Private/kimi/credential", "env:XAI_API_KEY"]
        with self.assertLogs("ai_gateway", "WARNING") as logs:
            values = secrets.resolve_many(specs, environ=self.env, warnings=warnings)
        self.assertEqual(values, {"xai": None, "openai": None, "deepseek": None, "env": "xai-from-env"})
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("not signed in", warnings[0])
        self.assertEqual(len(logs.output), 1)

    def test_op_missing_binary(self):
        warnings = []
        values = secrets.resolve_many({"a": ["op://Private/xai/credential"], "b": ["op://Private/openai/credential"]},
                                      environ={"PATH": os.path.join(self.tmp.name, "empty")}, warnings=warnings)
        self.assertEqual(values, {"a": None, "b": None})
        self.assertEqual(len(warnings), 1)
        self.assertIn("`op` not found", warnings[0])

    def test_op_failure_and_timeout(self):
        warnings = []
        values = secrets.resolve_many({"a": ["op://Private/unknown/credential"]}, environ=self.env,
                                      warnings=warnings)
        self.assertEqual(values, {"a": None})
        self.assertEqual(len(warnings), 1)
        self.assertIn("'unknown'", warnings[0])
        fake_bins.set_behaviour(self.bin, op_delay=5)
        warnings = []
        t0 = time.monotonic()
        values = secrets.resolve_many({"a": ["op://Private/xai/credential"]}, timeout=0.5, environ=self.env,
                                      warnings=warnings)
        self.assertLess(time.monotonic() - t0, 4)
        self.assertEqual(values, {"a": None})
        self.assertIn("timed out", warnings[0])

    def test_load_into_and_labels(self):
        store = config.SecretStore()
        warnings = []
        labels = secrets.load_into(store, {
            "xai": ["env:NOPE", "env:XAI_API_KEY"],
            "openai": ["op://Private/openai/credential"],
            "lit": ["literal:abc"],
            "json": ["json:%s#providers.xai.api_key" % self.cred],
            "missing": ["env:NOPE"],
        }, environ=self.env, warnings=warnings)
        self.assertEqual(labels, {"xai": "env:XAI_API_KEY", "openai": "op:openai", "lit": "literal",
                                  "json": "json:credentials.json"})
        self.assertEqual(store.get("xai"), "xai-from-env")
        self.assertEqual(store.get("openai"), "sk-op-openai")
        self.assertEqual(store.get("json"), "xai-from-json")
        self.assertFalse(store.has("missing"))
        self.assertEqual(warnings, [])
        for label in labels.values():
            self.assertNotIn("abc", label)
        self.assertEqual(secrets.source_label("json:C:\\Users\\me\\creds.json#a.b"), "json:creds.json")
        self.assertEqual(secrets.source_label("literal:" + VALUE), "literal")
        self.assertEqual(secrets.source_label(VALUE), "unknown")


if __name__ == "__main__":
    unittest.main()
