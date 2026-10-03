"""auth.codex_chatgpt: load/headers, refresh + write-back (unknown fields kept, 0600, last_refresh Z),
adopt-newer-on-disk, terminal errors with hint, single-flight across threads, lock timeout, and a
4-process (spawn) race against the rotating mock token endpoint (exactly one network refresh)."""

import json
import multiprocessing
import os
import stat
import tempfile
import threading
import time
import unittest

from ._pkg import mod

codex = mod("auth.codex_chatgpt")
auth_pkg = mod("auth")
auth_base = mod("auth.base")
compat = mod("compat")
config = mod("config")
filelock = mod("filelock")
mocks = mod("testing.mock_upstreams")
mock_auth = mod("testing.mock_auth")

AUTH_CLAIM = "https://api.openai.com/auth"


def _jwt(exp_in, **claims):
    claims.setdefault("exp", int(time.time() + exp_in))
    return mock_auth.make_jwt(claims)


def write_auth(path, exp_in=-60, rt="rt-0", account_id="acct-file", **extra):
    tokens = {"id_token": _jwt(3600, email="dev@example.com",
                               **{AUTH_CLAIM: {"chatgpt_account_id": "acct-idtoken", "chatgpt_plan_type": "pro"}}),
              "access_token": _jwt(exp_in, jti="at-%s" % rt), "refresh_token": rt,
              "token_field_unknown": "keep-me"}
    if account_id is not None:
        tokens["account_id"] = account_id
    data = {"OPENAI_API_KEY": None, "tokens": tokens, "last_refresh": "2026-01-01T00:00:00Z",
            "unknown_top": {"nested": [1, 2]}}
    data.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def read_auth(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _race_worker(env, barrier, out_q):
    try:
        a = codex.CodexChatGPTAuth("codex", {}, environ=env)
        a.available()
        barrier.wait(60)
        out_q.put(("ok", a.headers()["Authorization"]))
    except Exception as exc:  # reported to the parent
        out_q.put(("err", "%s: %s" % (type(exc).__name__, exc)))


class CodexAuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        self.path = os.path.join(self.home, "auth.json")
        self.srv = mocks.MockServer("openai_oauth", options={"delay": 0.05}).start()
        self.env = {"CODEX_HOME": self.home, "AI_GATEWAY_OPENAI_AUTH_URL": self.srv.url + "/oauth/token"}

    def tearDown(self):
        self.srv.stop()
        self.tmp.cleanup()

    def auth(self, **options):
        return codex.CodexChatGPTAuth("codex", options, environ=self.env)

    def refreshes(self):
        return mock_auth.refresh_count(self.srv)

    # ---- loading -----------------------------------------------------------------------
    def test_fresh_token_headers_and_describe(self):
        data = write_auth(self.path, exp_in=3600)
        a = self.auth()
        self.assertTrue(a.available())
        h = a.headers()
        self.assertEqual(h, {"Authorization": "Bearer " + data["tokens"]["access_token"],
                             "ChatGPT-Account-ID": "acct-file"})
        self.assertEqual(a.account_id(), "acct-file")
        self.assertEqual(self.refreshes(), 0)
        d = a.describe()
        blob = json.dumps(d)
        for secret in (data["tokens"]["access_token"], "rt-0", data["tokens"]["id_token"]):
            self.assertNotIn(secret, blob)
        self.assertEqual(d["account"], "dev@example.com")
        self.assertEqual(d["plan"], "pro")
        self.assertTrue(d["expires_at"].endswith("Z"))
        self.assertEqual(a.relogin_hint(), "run `codex login`")
        self.assertIsNone(a.api_key())

    def test_account_id_from_id_token_claim(self):
        write_auth(self.path, exp_in=3600, account_id=None)
        a = self.auth()
        self.assertEqual(a.account_id(), "acct-idtoken")
        self.assertEqual(a.headers()["ChatGPT-Account-ID"], "acct-idtoken")

    def test_paths(self):
        self.assertEqual(self.auth().auth_path(), self.path)
        self.assertEqual(self.auth(auth_path="/x/y.json").auth_path(), "/x/y.json")
        a = codex.CodexChatGPTAuth("codex", {}, environ={"HOME": "/h", "USERPROFILE": "/h"})
        self.assertEqual(a.auth_path(), os.path.join("/h", ".codex", "auth.json"))

    def test_api_key_file_not_available(self):
        write_auth(self.path, exp_in=3600, OPENAI_API_KEY="sk-file-key")
        a = self.auth()
        self.assertFalse(a.available())
        self.assertEqual(a.api_key(), "sk-file-key")
        with self.assertRaises(auth_base.AuthError) as cm:
            a.headers()
        self.assertIn("API key", str(cm.exception))
        d = a.describe()
        self.assertFalse(d["available"])
        self.assertIn("openai route", d["hint"])
        self.assertNotIn("sk-file-key", json.dumps(d))

    def test_missing_and_malformed_file(self):
        a = self.auth()
        self.assertFalse(a.available())
        self.assertIn("cli_auth_credentials_store", a.describe()["hint"])
        with self.assertRaises(auth_base.AuthError) as cm:
            a.headers()
        self.assertIn("cli_auth_credentials_store", str(cm.exception))
        with open(self.path, "w") as f:
            f.write("{broken")
        self.assertFalse(a.available())
        self.assertFalse(a.on_unauthorized({"Authorization": "Bearer x"}))

    def test_availability_and_state_follow_expiry_and_refresh_token(self):
        write_auth(self.path, exp_in=3600)
        self.assertTrue(self.auth().available())
        self.assertEqual(self.auth().describe()["state"], "valid")
        write_auth(self.path, exp_in=-60)
        a = self.auth()
        self.assertTrue(a.available(), "an expired access token is renewed with the refresh token")
        self.assertEqual(a.describe()["state"], "expired (refreshed on first request)")
        write_auth(self.path, exp_in=-60, rt=None)
        a = self.auth()
        self.assertFalse(a.available(), "expired and nothing to renew it with: the transport is unusable")
        d = a.describe()
        self.assertEqual((d["available"], d["state"], d["hint"]), (False, "expired, no refresh token",
                                                                    "run `codex login`"))
        with self.assertRaises(auth_base.AuthError):
            a.headers()
        self.assertEqual(self.refreshes(), 0)

    # ---- refresh -----------------------------------------------------------------------
    def test_refresh_happy_path_write_back(self):
        old = write_auth(self.path, exp_in=-60)
        a = self.auth()
        h = a.headers()
        self.assertEqual(self.refreshes(), 1)
        req = self.srv.requests_for("/oauth/token")[-1]
        self.assertEqual(req["body_json"], {"grant_type": "refresh_token", "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
                                            "refresh_token": "rt-0"})
        self.assertEqual(req["headers"].get("Content-Type"), "application/json")
        new = read_auth(self.path)
        self.assertEqual(h["Authorization"], "Bearer " + new["tokens"]["access_token"])
        self.assertNotEqual(new["tokens"]["access_token"], old["tokens"]["access_token"])
        self.assertEqual(new["tokens"]["refresh_token"], mock_auth.token_chain(self.srv).latest())
        self.assertNotEqual(new["tokens"]["id_token"], old["tokens"]["id_token"])
        self.assertEqual(new["tokens"]["account_id"], "acct-file")
        self.assertEqual(new["tokens"]["token_field_unknown"], "keep-me")
        self.assertEqual(new["unknown_top"], {"nested": [1, 2]})
        self.assertIsNone(new["OPENAI_API_KEY"])
        self.assertTrue(new["last_refresh"].endswith("Z"))
        self.assertLess(abs(compat.rfc3339_to_epoch(new["last_refresh"]) - time.time()), 60)
        with open(self.path, "rb") as f:
            self.assertTrue(f.read().startswith(b'{\n  "OPENAI_API_KEY": null,\n  "tokens": {'))
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        self.assertTrue(os.path.exists(self.path + ".lock"))
        self.assertFalse([n for n in os.listdir(self.home) if n.endswith(".tmp")])
        # cached: no further refresh while fresh
        self.assertEqual(a.headers(), h)
        self.assertEqual(self.refreshes(), 1)

    def test_on_unauthorized_refreshes_and_short_circuits(self):
        write_auth(self.path, exp_in=3600)
        a = self.auth()
        used = a.headers()
        self.assertTrue(a.on_unauthorized(used))
        self.assertEqual(self.refreshes(), 1)
        self.assertNotEqual(a.headers(), used)
        self.assertTrue(a.on_unauthorized(used), "stale `used` headers: already replaced, no refresh")
        self.assertEqual(self.refreshes(), 1)

    def test_on_unauthorized_single_flight_threads(self):
        self.srv.options["delay"] = 0.3
        write_auth(self.path, exp_in=3600)
        a = self.auth()
        used = a.headers()
        results = []
        threads = [threading.Thread(target=lambda: results.append(a.on_unauthorized(used))) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(results, [True] * 8)
        self.assertEqual(self.refreshes(), 1)

    def test_adopt_newer_on_disk(self):
        write_auth(self.path, exp_in=3600, rt="rt-0")
        a = self.auth()
        used = a.headers()
        newer = write_auth(self.path, exp_in=3600, rt="rt-other-process")  # another process refreshed
        self.assertTrue(a.on_unauthorized(used))
        self.assertEqual(self.refreshes(), 0, "fresh on-disk token adopted without a network refresh")
        self.assertEqual(a.headers()["Authorization"], "Bearer " + newer["tokens"]["access_token"])
        self.assertEqual(read_auth(self.path), newer, "adopting never rewrites the file")

    def test_stale_newer_on_disk_refreshes_with_disk_token(self):
        write_auth(self.path, exp_in=3600, rt="rt-0")
        self.srv.stop()
        self.srv = mocks.MockServer("openai_oauth", options={"refresh_tokens": ["rt-disk"]}).start()
        self.env["AI_GATEWAY_OPENAI_AUTH_URL"] = self.srv.url + "/oauth/token"
        a = self.auth()
        a.headers()
        write_auth(self.path, exp_in=-5, rt="rt-disk")  # rotated elsewhere but already expiring
        a.headers(force_refresh=True)
        self.assertEqual(self.srv.requests_for("/oauth/token")[-1]["body_json"]["refresh_token"], "rt-disk")
        self.assertEqual(self.refreshes(), 1)

    def test_invalid_grant_is_terminal(self):
        for style in ("flat", "openai"):
            self.srv.stop()
            self.srv = mocks.MockServer("openai_oauth", options={"refresh_tokens": ["someone-else"],
                                                                 "error_style": style}).start()
            self.env["AI_GATEWAY_OPENAI_AUTH_URL"] = self.srv.url + "/oauth/token"
            before = write_auth(self.path, exp_in=-60)
            a = self.auth()
            with self.assertRaises(auth_base.AuthError) as cm:
                a.headers()
            self.assertTrue(cm.exception.terminal)
            self.assertEqual(cm.exception.hint, "run `codex login`")
            self.assertIn("codex login", str(cm.exception))
            self.assertEqual(read_auth(self.path), before)
            self.assertFalse(a.on_unauthorized({"Authorization": "Bearer x"}))

    def test_reused_token_but_rotated_on_disk_adopts(self):
        write_auth(self.path, exp_in=-60, rt="rt-0")
        mock_auth.token_chain(self.srv).redeem("rt-0")  # the codex CLI already spent rt-0 ...
        rotated = {}

        def cli_writes_file(req):  # ... and writes its result while we are talking to the server
            rotated.update(write_auth(self.path, exp_in=3600, rt="rt-cli"))

        self.srv.options["on_request"] = cli_writes_file
        self.srv.options["error_style"] = "openai"
        a = self.auth()
        h = a.headers()
        self.assertEqual(h["Authorization"], "Bearer " + rotated["tokens"]["access_token"])
        self.assertEqual(mock_auth.token_chain(self.srv).reuse_events, 1)

    def test_transient_failure_not_terminal(self):
        self.srv.stop()
        self.srv = mocks.MockServer("openai_oauth", options={"fail_status": 503, "fail_count": 1}).start()
        self.env["AI_GATEWAY_OPENAI_AUTH_URL"] = self.srv.url + "/oauth/token"
        before = write_auth(self.path, exp_in=-60)
        a = self.auth()
        with self.assertRaises(auth_base.AuthError) as cm:
            a.headers()
        self.assertFalse(cm.exception.terminal)
        self.assertEqual(read_auth(self.path), before)
        a.headers()
        self.assertEqual(self.refreshes(), 1)
        self.env["AI_GATEWAY_OPENAI_AUTH_URL"] = "http://127.0.0.1:9/oauth/token"  # connection refused
        with self.assertRaises(auth_base.AuthError) as cm:
            a.headers(force_refresh=True)
        self.assertFalse(cm.exception.terminal)

    def test_lock_timeout(self):
        write_auth(self.path, exp_in=-60)
        a = self.auth(lock_timeout=0.3)
        with filelock.ExclusiveFileLock(self.path + ".lock"):
            t0 = time.monotonic()
            with self.assertRaises(auth_base.AuthError) as cm:
                a.headers()
            self.assertLess(time.monotonic() - t0, 3)
        self.assertFalse(cm.exception.terminal)
        self.assertIn(".lock", str(cm.exception))
        self.assertEqual(self.refreshes(), 0)
        a.headers()
        self.assertEqual(self.refreshes(), 1)

    def test_make_auth(self):
        a = auth_pkg.make_auth({"kind": "codex_chatgpt", "auth_path": self.path}, config.SecretStore(), "codex")
        self.assertIsInstance(a, codex.CodexChatGPTAuth)
        self.assertEqual(a.auth_path(), self.path)
        self.assertEqual(a.kind, "codex_chatgpt")

    def test_four_process_race(self):
        self.srv.options["delay"] = 0.3
        write_auth(self.path, exp_in=-60)
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
        final = read_auth(self.path)
        self.assertEqual({r[1] for r in results}, {"Bearer " + final["tokens"]["access_token"]})
        chain = mock_auth.token_chain(self.srv)
        self.assertEqual(chain.refreshes, 1, "exactly one network refresh across 4 processes")
        self.assertEqual(chain.reuse_events, 0)
        self.assertEqual(final["tokens"]["refresh_token"], chain.latest())
        self.assertIn(final["tokens"]["refresh_token"], chain.valid)
        self.assertEqual(final["unknown_top"], {"nested": [1, 2]})


class MockOpenAIOAuthTests(unittest.TestCase):
    def test_reuse_revokes_successor_chain(self):
        tp = mod("transport")
        with mocks.MockServer("openai_oauth") as srv:
            c = tp.HttpClient(timeout=5, environ={})

            def post(rt):
                return c.request("POST", srv.url + "/oauth/token", {"Content-Type": "application/json"},
                                 json.dumps({"grant_type": "refresh_token", "client_id": mock_auth.CODEX_CLIENT_ID,
                                             "refresh_token": rt}), stream=False)

            r1 = post("rt-0")
            self.assertEqual(r1.status, 200)
            body = r1.json()
            self.assertGreater(compat.jwt_claims(body["access_token"])["exp"], time.time())
            rt1 = body["refresh_token"]
            r2 = post("rt-0")
            self.assertEqual((r2.status, r2.json()), (400, {"error": "refresh_token_reused",
                                                            "error_description": r2.json()["error_description"]}))
            r3 = post(rt1)
            self.assertEqual(r3.status, 400, "successor revoked after reuse")
            self.assertEqual(mock_auth.refresh_count(srv), 1)
            bad = c.request("POST", srv.url + "/oauth/token", {"Content-Type": "application/json"},
                            json.dumps({"grant_type": "refresh_token", "client_id": "x", "refresh_token": rt1}),
                            stream=False)
            self.assertEqual(bad.status, 401)
            c.close()


if __name__ == "__main__":
    unittest.main()
