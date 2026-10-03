"""auth.grok_cli: entry selection, expiry rules, OIDC discovery (cached, fallback, overrides), form
refresh + write-back (other entries/unknown fields kept, RT kept when none returned, 0600), adopt-newer,
terminal invalid_grant, single-flight, lock timeout, and a 4-process race against the xai_oidc mock."""

import json
import multiprocessing
import os
import stat
import tempfile
import threading
import time
import unittest
from urllib.parse import parse_qs

from ._pkg import mod

grok = mod("auth.grok_cli")
auth_base = mod("auth.base")
compat = mod("compat")
filelock = mod("filelock")
mocks = mod("testing.mock_upstreams")
mock_auth = mod("testing.mock_auth")

KEY = grok.GROK_ENTRY_KEY


def entry(issuer, key="xai-at-0", rt="rt-0", expires_in=-60, **extra):
    e = {"auth_mode": "oidc", "key": key, "refresh_token": rt, "oidc_issuer": issuer,
         "oidc_client_id": grok.GROK_CLIENT_ID, "create_time": compat.rfc3339_format(time.time() - 100, "seconds"),
         "unknown_entry_field": {"keep": True}}
    if expires_in is not None:
        e["expires_at"] = compat.rfc3339_format(time.time() + expires_in, "seconds")
    e.update(extra)
    return e


def write_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def standard(issuer, **kw):
    return {"web::session": {"auth_mode": "web_login", "key": "web-cookie-value"},
            KEY: entry(issuer, **kw),
            "xai::api_key": {"auth_mode": "api_key", "key": "xai-static-key"},
            "version": 3}


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _race_worker(env, barrier, out_q):
    try:
        a = grok.GrokCliAuth("grok", {}, environ=env)
        a.available()
        barrier.wait(60)
        out_q.put(("ok", a.headers()["Authorization"]))
    except Exception as exc:
        out_q.put(("err", "%s: %s" % (type(exc).__name__, exc)))


class GrokAuthTests(unittest.TestCase):
    def setUp(self):
        grok.clear_discovery_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "auth.json")
        self.srv = mocks.MockServer("xai_oidc", options={"delay": 0.05}).start()
        self.env = {"GROK_HOME": self.tmp.name}

    def tearDown(self):
        self.srv.stop()
        self.tmp.cleanup()
        grok.clear_discovery_cache()

    def restart(self, **options):
        self.srv.stop()
        self.srv = mocks.MockServer("xai_oidc", options=options).start()

    def auth(self, **options):
        return grok.GrokCliAuth("grok", options, environ=self.env)

    def token_posts(self):
        return [parse_qs(r["body_raw"]) for r in self.srv.requests_for("/oauth2/token", "POST")]

    def test_fresh_headers_selection_and_describe(self):
        data = write_file(self.path, standard(self.srv.url, expires_in=3600))
        a = self.auth()
        self.assertTrue(a.available())
        self.assertEqual(a.headers(), {"Authorization": "Bearer xai-at-0"})
        self.assertEqual(a.api_key_entry(), "xai-static-key")
        self.assertEqual(mock_auth.refresh_count(self.srv), 0)
        d = a.describe()
        self.assertEqual(d["entry"], KEY)
        self.assertTrue(d["api_key_entry"])
        blob = json.dumps(d)
        for secret in ("xai-at-0", "rt-0", "xai-static-key", "web-cookie-value"):
            self.assertNotIn(secret, blob)
        self.assertEqual(a.relogin_hint(), "run `grok login`")
        self.assertEqual(read_file(self.path), data)

    def test_entry_fallback_and_modes(self):
        write_file(self.path, {"web": {"auth_mode": "web_login", "key": "w"},
                               KEY: {"auth_mode": "web_login", "key": "w2"},
                               "https://other.issuer::client-x": {"auth_mode": "external", "key": "ext-key",
                                                                  "refresh_token": "r"},
                               "https://later::y": {"auth_mode": "oidc", "key": "later-key"}})
        a = self.auth()
        self.assertEqual(a.headers(), {"Authorization": "Bearer ext-key"})
        tok = a.current_token()
        self.assertEqual(tok.extra["issuer"], "https://other.issuer")
        self.assertEqual(tok.extra["client_id"], "client-x")
        self.assertIsNone(a.api_key_entry())

    def test_api_key_only(self):
        write_file(self.path, {"xai::api_key": {"auth_mode": "api_key", "key": " xai-only "}})
        a = self.auth()
        self.assertFalse(a.available())
        self.assertEqual(a.api_key_entry(), "xai-only")
        with self.assertRaises(auth_base.AuthError) as cm:
            a.headers()
        self.assertIn("xai", cm.exception.hint)
        self.assertIn("xai route", a.describe()["hint"])

    def test_missing_file(self):
        a = self.auth()
        self.assertFalse(a.available())
        self.assertIsNone(a.api_key_entry())
        self.assertEqual(a.describe()["hint"], "run `grok login`")

    def test_paths(self):
        self.assertEqual(self.auth().auth_path(), self.path)
        a = grok.GrokCliAuth("grok", {}, environ={"GROK_AUTH_PATH": "/a/b.json", "GROK_HOME": "/c"})
        self.assertEqual(a.auth_path(), "/a/b.json")
        a = grok.GrokCliAuth("grok", {}, environ={"HOME": "/h", "USERPROFILE": "/h"})
        self.assertEqual(a.auth_path(), os.path.join("/h", ".grok", "auth.json"))
        self.assertEqual(grok.GrokCliAuth("grok", {"auth_path": "/o.json"}, environ={}).auth_path(), "/o.json")

    def test_expiry_rules(self):
        now = time.time()
        cases = [
            ({"expires_at": compat.rfc3339_format(now + 1000)}, now + 1000),
            ({"expires_at": int(now + 2000)}, int(now + 2000)),
            ({"expires_at": int((now + 3000) * 1000)}, int((now + 3000) * 1000) / 1000.0),
            ({"create_time": compat.rfc3339_format(now - 10)}, now - 10 + 30 * 86400),
            ({"key": mock_auth.make_jwt({"exp": int(now + 4000)})}, int(now + 4000)),
            ({}, None),
        ]
        for fields, want in cases:
            e = {"auth_mode": "oidc", "key": "k"}
            e.update(fields)
            write_file(self.path, {KEY: e})
            tok = self.auth()._load()
            if want is None:
                self.assertIsNone(tok.expires_at, fields)
            else:
                self.assertAlmostEqual(tok.expires_at, want, delta=1.5, msg=fields)

    def test_refresh_happy_path_write_back(self):
        before = write_file(self.path, standard(self.srv.url, principal_type="User", principal_id="u-1"))
        a = self.auth()
        h = a.headers()
        self.assertEqual(self.srv.state["discovery_hits"], 1)
        posts = self.token_posts()
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0], {"grant_type": ["refresh_token"], "refresh_token": ["rt-0"],
                                    "client_id": [grok.GROK_CLIENT_ID], "principal_type": ["User"],
                                    "principal_id": ["u-1"]})
        after = read_file(self.path)
        e = after[KEY]
        self.assertEqual(h, {"Authorization": "Bearer " + e["key"]})
        self.assertNotEqual(e["key"], "xai-at-0")
        self.assertEqual(e["refresh_token"], mock_auth.token_chain(self.srv).latest())
        self.assertAlmostEqual(compat.rfc3339_to_epoch(e["expires_at"]), time.time() + 21600, delta=30)
        self.assertTrue(e["expires_at"].endswith("Z"))
        self.assertAlmostEqual(compat.rfc3339_to_epoch(e["create_time"]), time.time(), delta=30)
        self.assertEqual(e["unknown_entry_field"], {"keep": True})
        self.assertEqual(e["principal_id"], "u-1")
        for k in ("web::session", "xai::api_key", "version"):
            self.assertEqual(after[k], before[k])
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        with open(self.path, "rb") as f:
            self.assertTrue(f.read().startswith(b'{\n  "web::session": {'))
        # second refresh: discovery is cached per process
        a.headers(force_refresh=True)
        self.assertEqual(self.srv.state["discovery_hits"], 1)
        self.assertEqual(mock_auth.refresh_count(self.srv), 2)
        self.assertEqual(self.token_posts()[-1]["refresh_token"], [e["refresh_token"]])

    def test_numeric_times_keep_their_type(self):
        e = entry(self.srv.url, expires_in=None)
        e["expires_at"] = int(time.time() - 60)
        e["create_time"] = int((time.time() - 100) * 1000)
        write_file(self.path, {KEY: e})
        self.auth().headers()
        after = read_file(self.path)[KEY]
        self.assertIsInstance(after["expires_at"], int)
        self.assertAlmostEqual(after["expires_at"], time.time() + 21600, delta=30)
        self.assertGreater(after["create_time"], 1e12)

    def test_keeps_old_refresh_token_when_none_returned(self):
        self.restart(omit_refresh_token=True)
        write_file(self.path, standard(self.srv.url))
        self.auth().headers()
        self.assertEqual(read_file(self.path)[KEY]["refresh_token"], "rt-0")
        self.assertEqual(mock_auth.refresh_count(self.srv), 1)

    def test_discovery_fallback_and_overrides(self):
        self.restart(discovery_status=404)
        write_file(self.path, standard(self.srv.url))
        a = self.auth()
        a.headers()
        self.assertEqual(self.srv.state["discovery_hits"], 1)
        self.assertEqual(mock_auth.refresh_count(self.srv), 1, "fallback {issuer}/oauth2/token used")
        a.headers(force_refresh=True)
        self.assertEqual(self.srv.state["discovery_hits"], 2, "failed discovery is not cached")
        # explicit token URL: no discovery at all
        self.restart()
        grok.clear_discovery_cache()
        write_file(self.path, standard("https://auth.x.ai.invalid"))
        self.env["AI_GATEWAY_XAI_TOKEN_URL"] = self.srv.url + "/oauth2/token"
        self.auth().headers()
        self.assertEqual(self.srv.state["discovery_hits"], 0)
        self.assertEqual(mock_auth.refresh_count(self.srv), 1)
        # issuer override: discovery against the override, not the entry's issuer
        del self.env["AI_GATEWAY_XAI_TOKEN_URL"]
        self.restart()
        write_file(self.path, standard("https://auth.x.ai.invalid"))
        self.env["AI_GATEWAY_XAI_OIDC_ISSUER"] = self.srv.url
        self.auth().headers()
        self.assertEqual(self.srv.state["discovery_hits"], 1)
        self.assertEqual(mock_auth.refresh_count(self.srv), 1)

    def test_endpoint_validation(self):
        self.assertTrue(grok._endpoint_ok("https://auth.x.ai/oauth2/token"))
        self.assertTrue(grok._endpoint_ok("http://127.0.0.1:5/t"))
        self.assertFalse(grok._endpoint_ok("http://evil.example/t"))
        self.assertFalse(grok._endpoint_ok("file:///etc/passwd"))

    def test_invalid_grant_terminal(self):
        self.restart(refresh_tokens=["not-ours"])
        before = write_file(self.path, standard(self.srv.url))
        a = self.auth()
        with self.assertRaises(auth_base.AuthError) as cm:
            a.headers()
        self.assertTrue(cm.exception.terminal)
        self.assertEqual(cm.exception.hint, "run `grok login`")
        self.assertIn("invalid_grant", str(cm.exception))
        self.assertEqual(read_file(self.path), before)

    def test_adopt_newer_on_disk(self):
        write_file(self.path, standard(self.srv.url, expires_in=3600))
        a = self.auth()
        used = a.headers()
        write_file(self.path, standard(self.srv.url, key="xai-at-other", rt="rt-other", expires_in=3600))
        self.assertTrue(a.on_unauthorized(used))
        self.assertEqual(a.headers(), {"Authorization": "Bearer xai-at-other"})
        self.assertEqual(mock_auth.refresh_count(self.srv), 0)
        self.assertEqual(self.srv.state["discovery_hits"], 0)

    def test_single_flight_threads(self):
        self.restart(delay=0.3)
        write_file(self.path, standard(self.srv.url, expires_in=3600))
        a = self.auth()
        used = a.headers()
        results = []
        threads = [threading.Thread(target=lambda: results.append(a.on_unauthorized(used))) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(results, [True] * 8)
        self.assertEqual(mock_auth.refresh_count(self.srv), 1)

    def test_lock_timeout(self):
        write_file(self.path, standard(self.srv.url))
        a = self.auth(lock_timeout=0.2)
        with filelock.ExclusiveFileLock(self.path + ".lock"):
            with self.assertRaises(auth_base.AuthError) as cm:
                a.headers()
        self.assertFalse(cm.exception.terminal)
        self.assertEqual(mock_auth.refresh_count(self.srv), 0)

    def test_four_process_race(self):
        self.restart(delay=0.3)
        write_file(self.path, standard(self.srv.url))
        ctx = multiprocessing.get_context("spawn")
        barrier = ctx.Barrier(4)
        q = ctx.Queue()
        procs = [ctx.Process(target=_race_worker, args=(dict(self.env), barrier, q)) for _ in range(4)]
        for p in procs:
            p.start()
        try:
            results = [q.get(timeout=120) for _ in procs]
        finally:
            for p in procs:
                p.join(60)
                if p.is_alive():
                    p.terminate()
        self.assertEqual([r[0] for r in results], ["ok"] * 4, results)
        final = read_file(self.path)
        self.assertEqual({r[1] for r in results}, {"Bearer " + final[KEY]["key"]})
        chain = mock_auth.token_chain(self.srv)
        self.assertEqual(chain.refreshes, 1)
        self.assertEqual(chain.reuse_events, 0)
        self.assertEqual(final[KEY]["refresh_token"], chain.latest())
        self.assertEqual(final["xai::api_key"], {"auth_mode": "api_key", "key": "xai-static-key"})


if __name__ == "__main__":
    unittest.main()
