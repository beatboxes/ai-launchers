"""`models [--refresh] [--json]`: catalog ∪ discovered ids with picker ids; discovery uses the free list
endpoints in parallel with a 3 s timeout and a 6 h cache (~/.ai-launchers/cache/models.json, 0600)."""

import json
import os
import stat
import time
import unittest

from ._util import LauncherTestCase, fake_key
from shared.gateway.testing.mock_upstreams import MockServer


def list_handler(payload, status=200, delay=0.0):
    def handle(server, req, resp):
        if delay:
            time.sleep(delay)
        if req.method == "GET" and req.path.endswith("/models"):
            resp.send_json(status, payload)
    return handle


class ModelsTests(LauncherTestCase):
    def models_json(self, launcher, *extra):
        rc, out, err = self.run_cli(launcher, "models", "--json", *extra)
        self.assertEqual(rc, 0, err)
        data = json.loads(out)
        return {t["id"]: t for t in data["transports"]}, data

    def test_discovery_cache_and_refresh(self):
        key = fake_key("xai")
        os.environ["XAI_API_KEY"] = key
        payload = {"data": [{"id": "grok-9"}, {"id": "grok-2-image"}, {"id": "grok-4.7"}, {"id": "bad,id"}]}
        with MockServer(handler=list_handler(payload)) as mock:
            os.environ["AI_GATEWAY_UPSTREAM_XAI"] = mock.url + "/v1"
            transports, data = self.models_json("grok")
            self.assertEqual(len(mock.requests), 1)
            self.assertEqual(mock.requests[0]["path"], "/v1/models")
            self.assertEqual(mock.requests[0]["headers"].get("Authorization"), "Bearer " + key)
            models = {m["id"]: m for m in transports["xai"]["models"]}
            self.assertEqual(models["grok-9"]["source"], "discovered")
            self.assertEqual(models["grok-9"]["picker_id"], "claude-via-xai,grok-9")
            self.assertEqual(models["grok-4.7"]["source"], "catalog")
            self.assertNotIn("grok-2-image", models)
            self.assertNotIn("bad,id", models)
            self.assertEqual(data["discovery_errors"], {})
            self.assertEqual(transports["xai"]["default_model"], "grok-4.7")
            self.assertFalse(transports["grok"]["available"])
            cache = os.path.join(self.ail, "cache", "models.json")
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(os.stat(cache).st_mode), 0o600)
            self.models_json("grok")                       # fresh cache: no request
            self.assertEqual(len(mock.requests), 1)
            self.models_json("grok", "--refresh")          # forced
            self.assertEqual(len(mock.requests), 2)
            launcher = self.launcher("grok")               # launches pick discovered ids up
            selected, _ = launcher.resolve()
            table = launcher.route_table(selected, "grok-9")
            self.assertEqual(table.roles["default"], "xai,grok-9")
            rc, out, _ = self.run_cli("grok", "models")
            self.assertIn("claude-via-xai,grok-9", out)
            self.assertIn("(discovered)", out)

    def test_discovery_failure_keeps_catalog(self):
        os.environ["OPENAI_API_KEY"] = fake_key("openai")
        with MockServer(handler=list_handler({"error": {"message": "bad key"}}, status=401)) as mock:
            os.environ["AI_GATEWAY_UPSTREAM_OPENAI"] = mock.url + "/v1"
            transports, data = self.models_json("codex")
        self.assertIn("HTTP 401", data["discovery_errors"]["openai"])
        self.assertIn("gpt-5.5", [m["id"] for m in transports["openai"]["models"]])
        self.assertIn("claude-via-codex,gpt-5.5", [m["picker_id"] for m in transports["codex"]["models"]])

    def test_discovery_timeout(self):
        os.environ["XAI_API_KEY"] = fake_key("xai")
        with MockServer(handler=list_handler({"data": []}, delay=6.0)) as mock:
            os.environ["AI_GATEWAY_UPSTREAM_XAI"] = mock.url + "/v1"
            t0 = time.time()
            _, data = self.models_json("grok")
            elapsed = time.time() - t0
        self.assertIn("xai", data["discovery_errors"])
        self.assertLess(elapsed, 5.5)

    def test_gemini_list_format(self):
        key = fake_key("gemini")
        os.environ["GEMINI_API_KEY"] = key
        payload = {"models": [
            {"name": "models/gemini-9-pro", "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/text-embedding-5", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-9-flash-image", "supportedGenerationMethods": ["generateContent"]}]}
        with MockServer(handler=list_handler(payload)) as mock:
            os.environ["AI_GATEWAY_UPSTREAM_GEMINI"] = mock.url
            transports, _ = self.models_json("gemini")
            req = mock.requests[0]
        self.assertEqual(req["path"], "/v1beta/models")
        self.assertEqual(req["headers"].get("x-goog-api-key"), key)
        ids = [m["id"] for m in transports["gemini"]["models"] if m["source"] == "discovered"]
        self.assertEqual(ids, ["gemini-9-pro"])

    def test_passthrough_vendor_list_url(self):
        key = fake_key("deepseek")
        os.environ["DEEPSEEK_API_KEY"] = key
        with MockServer(handler=list_handler({"data": [{"id": "deepseek-v5"}]})) as mock:
            self.write_config({"providers": {"deepseek": {"list_url": mock.url + "/models"}}})
            transports, data = self.models_json("deepseek")
            self.assertEqual(mock.requests[0]["headers"].get("Authorization"), "Bearer " + key)
        t = transports["deepseek"]
        self.assertEqual(data["mode"], "gateway")
        self.assertEqual((t["default_model"], t["background_model"]), ("deepseek-v4-pro", "deepseek-v4-flash"))
        found = [m for m in t["models"] if m["id"] == "deepseek-v5"]
        self.assertEqual(found[0]["source"], "discovered")
        self.assertEqual(found[0]["picker_id"], "claude-via-deepseek,deepseek-v5")
        pro = [m for m in t["models"] if m["id"] == "deepseek-v4-pro"]
        self.assertEqual(pro[0]["picker_id"], "claude-via-deepseek,deepseek-v4-pro[1m]")

    def test_no_keys_no_network(self):
        transports, data = self.models_json("grok")
        self.assertEqual(data["discovery_errors"], {})
        self.assertFalse(os.path.exists(os.path.join(self.ail, "cache", "models.json")))
        self.assertIn("XAI_API_KEY", transports["xai"]["reason"])


if __name__ == "__main__":
    unittest.main()
