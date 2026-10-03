"""Keyless transports end to end: the REAL launcher entry points + the REAL installed Claude Code against
quirk-enforcing mock upstreams and mock token servers (opt-in: RUN_E2E=1), plus always-on checks.

E2E (``RUN_E2E=1``): every case makes Claude Code run ``printf hello > out.txt && env`` with its Bash tool
through the gateway and answer ``DONE <sha8>``; the test checks rc 0, ``out.txt``, the Bash ``env`` dump
(the gateway variables are there; no access/refresh/id token, client secret or key is) and that no
credential reaches Claude Code's stdout/stderr or the gateway log/trace (``--debug``).

* grok-wrap ``--auth login``: Grok Build ``auth.json`` -> CLI-proxy mock (headers, session ids);
  version-gated proxy -> sticky fallback to api.x.ai chat with the same token; expired login -> OIDC
  discovery + refresh -> rotated ``auth.json`` write-back.
* codex-wrap ``--auth login``: expired ChatGPT login -> exactly one refresh + write-back -> Codex backend
  mock (headers, body rules, encrypted reasoning echo); a 401 mid-session -> refresh -> retry.
* gemini-wrap ``--auth adc``: authorized_user ADC file -> Google token mock -> Vertex mock (Bearer +
  x-goog-user-project, projects/locations path, thoughtSignature echo); no ADC file -> fake ``gcloud``.
* ``--model`` as a picker id, a bare id and a legacy fry alias on the API-key routes.

Always on: direct HTTP against a running Gateway (``max_tokens:1`` probes, 404 for unknown models,
``/v1/models``, count_tokens) and ``doctor`` on valid / expired / missing login files.
"""

import json
import os
import shutil
import stat
import sys
import tempfile
import time
import unittest

from ._util import REPO_ROOT, SENTINEL, LauncherTestCase, fake_key, mock_kind_or_skip
from .test_e2e import RUN_E2E, _tool_results
from shared.gateway import compat
from shared.gateway.testing import fake_bins
from shared.gateway.testing.mock_auth import (GOOGLE_CLIENT_ID, GROK_CLIENT_ID, make_jwt, refresh_count,
                                              token_chain)
from shared.gateway.testing.mock_responses import GROK_REQUIRED_HEADERS
from shared.gateway.testing.mock_upstreams import MockServer, get_kind_factory

GROK_ENTRY = "https://auth.x.ai::" + GROK_CLIENT_ID
OPENAI_AUTH_CLAIM = "https://api.openai.com/auth"
ACCOUNT = "acct-e2e-0001"
CREDENTIAL_KEYS = ("key", "refresh_token", "access_token", "id_token", "client_secret", "api_key")
_SCRUB = ("AI_GATEWAY_", "AI_LAUNCHERS_", "ANTHROPIC_", "GOOGLE_", "CLOUDSDK_", "GCLOUD_", "CODEX_", "GROK_",
          "XAI_", "OPENAI_", "GEMINI_", "DEEPSEEK_", "MOONSHOT_", "KIMI_")
_SESSION_HEADERS = {"codex": ("session-id",), "grok": ("x-grok-conv-id", "x-grok-session-id")}


def _rfc3339(epoch):
    return compat.rfc3339_format(epoch, "seconds")


def _header(rec, name):
    for k, v in (rec.get("headers") or {}).items():
        if k.lower() == name.lower():
            return v
    return None


def _bearer(rec):
    value = _header(rec, "Authorization") or ""
    return value[7:] if value.lower().startswith("bearer ") else None


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj, mode=0o600):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    if os.name != "nt":
        os.chmod(path, mode)
    return path


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def _session_id(res):
    for ev in res.events:
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            return ev.get("session_id")
    return None


def _credentials(obj):
    """Every credential value (``CREDENTIAL_KEYS``) in a JSON document, for leak checks."""
    if isinstance(obj, dict):
        return [s for k, v in obj.items()
                for s in ([v] if k in CREDENTIAL_KEYS and isinstance(v, str) else _credentials(v))]
    if isinstance(obj, list):
        return [s for v in obj for s in _credentials(v)]
    return []


def _unauthorized_once(at_post):
    """chatgpt_codex mock whose ``at_post``-th POST (1-based) is answered once with 401 token_expired."""
    def handle(server, req, resp):
        with server.lock:
            inner = server.state.get("_inner")
            if inner is None:
                inner = server.state["_inner"] = get_kind_factory("chatgpt_codex")(server)
            if req.method == "POST":
                server.state["_posts"] = server.state.get("_posts", 0) + 1
            reject = req.method == "POST" and server.state["_posts"] == at_post
        if reject:
            resp.send_json(401, {"error": {"message": "Provided authentication token is expired. Please try "
                                                      "signing in again.", "type": None, "code": "token_expired",
                                           "param": None}, "status": 401})
            return
        inner(req, resp)
    return handle


# ---------------------------------------------------------------------------------------
# E2E with the real Claude Code
# ---------------------------------------------------------------------------------------

@unittest.skipUnless(RUN_E2E, "set RUN_E2E=1 to run the launchers against the real Claude Code")
class LoginE2E(unittest.TestCase):
    """Login / ADC transports (and --model forms) through the real launchers and the real Claude Code."""

    @classmethod
    def setUpClass(cls):
        from shared.gateway.testing import claude_e2e

        if not claude_e2e.claude_available():
            raise unittest.SkipTest("Claude Code (`claude`) is not installed")
        cls.e2e = claude_e2e

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ail-e2e-login-")
        self.home = os.path.join(self.tmp, "home")
        self.bins = os.path.join(self.tmp, "bin")
        os.makedirs(self.home)

    def tearDown(self):
        shutil.rmtree(self.tmp, True)

    def mock(self, kind, **options):
        mu = mock_kind_or_skip(self, kind)
        srv = mu.MockServer(kind, options=options).start()
        self.addCleanup(srv.stop)
        return srv

    def custom_mock(self, handler, **options):
        srv = MockServer(handler=handler, options=options).start()
        self.addCleanup(srv.stop)
        return srv

    # ---- running ------------------------------------------------------------------------
    def launch(self, launcher, env_extra, *args, **behaviour):
        """``<launcher>-wrap launch claude <args> -- -p …`` with fake CLIs first on PATH."""
        fake_bins.make_fake_bins(self.bins, **behaviour)
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith(_SCRUB)}
        env["PATH"] = os.pathsep.join([self.bins, env.get("PATH") or os.defpath])
        env.update(env_extra)
        argv = [sys.executable, os.path.join(REPO_ROOT, launcher, "%s-wrap.py" % launcher), "launch", "claude"]
        argv += list(args)
        return self.e2e.run_e2e(argv, env, os.path.join(self.tmp, "work"), timeout=300, home=self.home)

    def info(self, res, *mocks):
        parts = [res.summary()]
        for m in mocks:
            parts.append("--- %s requests: %s" % (m.kind or "custom", [(r["method"], r["path"]) for r in m.requests]))
            parts.extend("--- %s error: %s" % (m.kind or "custom", e) for e in m.errors)
        logs = os.path.join(self.home, ".ai-launchers", "logs")
        for name in sorted(os.listdir(logs)) if os.path.isdir(logs) else []:
            if name.endswith(".log"):
                with open(os.path.join(logs, name), "r", encoding="utf-8", errors="replace") as f:
                    parts.append("--- %s (tail) ---\n%s" % (name, f.read()[-3000:]))
        return "\n".join(parts)

    def assert_round_trip(self, res, launcher, secrets, *mocks):
        """rc 0, DONE, out.txt, Bash env dump present and free of every credential, nothing leaked anywhere."""
        info = self.info(res, *mocks)
        self.assertFalse(res.timed_out, info)
        self.assertEqual(res.rc, 0, info)
        self.assertIn("DONE", res.result_text or "", info)
        with open(os.path.join(res.workdir, "out.txt"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "hello")
        for m in mocks:
            self.assertEqual(m.errors, [], info)
            self.assertEqual(m.brain.leaks, [], "credential leaked into the Bash tool output\n" + info)
        dumps = [t for t in _tool_results(res.events) if "ANTHROPIC_BASE_URL=" in t]
        self.assertTrue(dumps, "the Bash tool's `env` output was not captured\n" + info)
        secrets = sorted(set(s for s in secrets if s and len(s) >= 6))
        self.assertTrue(secrets)
        for dump in dumps:
            self.assertIn("ANTHROPIC_BASE_URL=http://127.0.0.1:", dump)
            self.assertNotIn(SENTINEL, dump)
            for s in secrets:
                self.assertNotIn(s, dump, "a credential reached Claude Code's environment")
        logs = os.path.join(self.home, ".ai-launchers", "logs")
        texts = {"stdout": res.stdout, "stderr": res.stderr}
        for name in os.listdir(logs) if os.path.isdir(logs) else []:
            with open(os.path.join(logs, name), "r", encoding="utf-8", errors="replace") as f:
                texts[name] = f.read()
        self.assertIn("%s-wrap.log" % launcher, texts, "gateway log missing\n" + info)
        for where, text in texts.items():
            self.assertNotIn(SENTINEL, text, where)
            for s in secrets:
                self.assertNotIn(s, text, "credential found in %s" % where)
        self.assertTrue(res.settings_unchanged, "~/.claude/settings.json was modified")
        self.assertTrue(res.claude_json_user_entry_intact, "~/.claude.json user entry was lost")
        return info

    def assert_session_headers(self, res, kind, records):
        sid = _session_id(res)
        self.assertTrue(sid)
        for rec in records:
            for name in _SESSION_HEADERS[kind]:
                self.assertEqual(_header(rec, name), sid, name)

    # ---- grok -------------------------------------------------------------------------------
    def grok_auth(self, oidc, expired=False):
        now = time.time()
        exp = int(now) + (-600 if expired else 3600)
        entry = {"auth_mode": "oidc", "key": make_jwt({"exp": exp, "sub": "u-1"}),
                 "refresh_token": "rt-grok-%s-0" % SENTINEL, "expires_at": _rfc3339(exp),
                 "create_time": _rfc3339(now - 7200), "oidc_issuer": oidc.url, "oidc_client_id": GROK_CLIENT_ID,
                 "user_label": "e2e@example.com"}
        data = {GROK_ENTRY: entry, "https://other.example::cli": {"auth_mode": "web_login", "note": "keep me"}}
        return _write_json(os.path.join(self.home, ".grok", "auth.json"), data), data

    @staticmethod
    def grok_env(proxy, chat=None):
        env = {"AI_GATEWAY_UPSTREAM_GROK": proxy.url + "/v1"}
        if chat is not None:
            env["AI_GATEWAY_UPSTREAM_GROK_FALLBACK"] = chat.url + "/v1"
        return env

    def test_grok_login_proxy(self):
        oidc = self.mock("xai_oidc", refresh_tokens=["rt-grok-%s-0" % SENTINEL])
        path, data = self.grok_auth(oidc)
        token = data[GROK_ENTRY]["key"]
        proxy = self.mock("grok_proxy", api_key=token)
        res = self.launch("grok", self.grok_env(proxy), "--auth", "login", "--debug",
                          grok_version_output="grok 1.2.3")
        self.assert_round_trip(res, "grok", _credentials(data), proxy, oidc)
        posts = proxy.requests_for("/v1/responses", "POST")
        self.assertGreaterEqual(len(posts), 2)
        for rec in posts:
            for name, expected in GROK_REQUIRED_HEADERS.items():
                self.assertTrue(_header(rec, name), name)
                if expected is not None:
                    self.assertEqual(_header(rec, name), expected)
            self.assertEqual(_header(rec, "x-grok-client-version"), "1.2.3")
            self.assertEqual(_bearer(rec), token)
            self.assertNotIn("max_output_tokens", rec["body_json"])
            self.assertNotIn("temperature", rec["body_json"])
        self.assertEqual(len({_header(r, "x-grok-req-id") for r in posts}), len(posts), "x-grok-req-id is per request")
        self.assert_session_headers(res, "grok", posts)
        self.assertEqual(refresh_count(oidc), 0)
        self.assertEqual(oidc.requests, [], "a fresh login must not touch the token endpoint")
        self.assertEqual(_read_json(path), data, "auth.json must not be rewritten without a refresh")

    def test_grok_login_version_gate_falls_back_to_api(self):
        oidc = self.mock("xai_oidc", refresh_tokens=["rt-grok-%s-0" % SENTINEL])
        _, data = self.grok_auth(oidc)
        token = data[GROK_ENTRY]["key"]
        proxy = self.mock("grok_proxy", api_key=token, version_gate=True)
        chat = self.mock("xai_chat", api_key=token)
        res = self.launch("grok", self.grok_env(proxy, chat), "--auth", "login", "--debug")
        info = self.assert_round_trip(res, "grok", _credentials(data), proxy, chat)
        self.assertEqual(len(proxy.requests_for("", "POST")), 1, info)
        chats = chat.requests_for("/v1/chat/completions", "POST")
        self.assertGreaterEqual(len(chats), 2, info)
        for rec in chats:
            self.assertEqual(_bearer(rec), token)
            self.assertTrue(rec["body_json"].get("stream"))
            self.assertEqual(rec["body_json"].get("model"), "grok-4.7")
        with open(os.path.join(self.home, ".ai-launchers", "logs", "grok-wrap.log"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read().count("grok CLI proxy unavailable"), 1)

    def test_grok_login_expired_refreshes_and_writes_back(self):
        oidc = self.mock("xai_oidc", refresh_tokens=["rt-grok-%s-0" % SENTINEL], jwt=True)
        path, before = self.grok_auth(oidc, expired=True)
        if os.name != "nt":
            os.chmod(path, 0o644)  # the write-back restores owner-only permissions
        proxy = self.mock("grok_proxy")
        res = self.launch("grok", self.grok_env(proxy), "--auth", "login", "--debug")
        after = _read_json(path)
        self.assert_round_trip(res, "grok", _credentials(before) + _credentials(after), proxy, oidc)
        self.assertEqual(refresh_count(oidc), 1)
        self.assertEqual(len([r for r in oidc.requests if r["path"].endswith("openid-configuration")]), 1)
        old, new = before[GROK_ENTRY], after[GROK_ENTRY]
        self.assertNotEqual(new["key"], old["key"])
        self.assertEqual(new["refresh_token"], token_chain(oidc).latest())
        self.assertNotEqual(new["refresh_token"], old["refresh_token"])
        self.assertGreater(compat.rfc3339_to_epoch(new["expires_at"]), time.time() + 3000)
        for k in ("auth_mode", "oidc_issuer", "oidc_client_id", "user_label"):
            self.assertEqual(new[k], old[k], k)
        self.assertEqual(after["https://other.example::cli"], before["https://other.example::cli"])
        if os.name != "nt":
            self.assertEqual(_mode(path), 0o600)
        posts = proxy.requests_for("/v1/responses", "POST")
        self.assertTrue(posts)
        self.assertEqual({_bearer(r) for r in posts}, {new["key"]})

    # ---- codex ------------------------------------------------------------------------------
    def codex_auth(self, expired):
        now = int(time.time())
        claim = {OPENAI_AUTH_CLAIM: {"chatgpt_account_id": ACCOUNT, "chatgpt_plan_type": "plus"}}
        tokens = {"id_token": make_jwt(dict(claim, email="dev@example.com", exp=now + 3600)),
                  "access_token": make_jwt(dict(claim, exp=now + (-120 if expired else 3600))),
                  "refresh_token": "rt-codex-%s-0" % SENTINEL, "account_id": ACCOUNT}
        data = {"OPENAI_API_KEY": None, "tokens": tokens, "last_refresh": _rfc3339(now - 10 * 86400),
                "agent_identity": {"keep": "me"}}
        return _write_json(os.path.join(self.home, ".codex", "auth.json"), data), data

    @staticmethod
    def codex_env(oauth, backend):
        return {"AI_GATEWAY_OPENAI_AUTH_URL": oauth.url + "/oauth/token",
                "AI_GATEWAY_UPSTREAM_CODEX": backend.url + "/backend-api/codex"}

    def assert_codex_requests(self, res, backend, posts, info):
        self.assertGreaterEqual(len(posts), 2, info)
        for rec in posts:
            body = rec["body_json"]
            self.assertEqual(_header(rec, "originator"), "codex_cli_rs")
            self.assertEqual(_header(rec, "version"), "0.161.2")
            self.assertTrue((_header(rec, "User-Agent") or "").startswith("codex_cli_rs/0.161.2 "))
            self.assertEqual(_header(rec, "ChatGPT-Account-ID"), ACCOUNT)
            self.assertIs(body.get("store"), False)
            self.assertIs(body.get("stream"), True)
            self.assertTrue((body.get("instructions") or "").strip())
            for key in ("max_output_tokens", "temperature", "top_p"):
                self.assertNotIn(key, body)
            self.assertIn("reasoning.encrypted_content", body.get("include") or [])
        self.assert_session_headers(res, "codex", posts)
        # turn 2 replays turn 1's reasoning (encrypted_content) right before its function_call
        last = posts[-1]["body_json"]["input"]
        kinds = [i.get("type") for i in last]
        self.assertIn("function_call", kinds, info)
        call = kinds.index("function_call")
        self.assertGreater(call, 0)
        self.assertEqual(last[call - 1].get("type"), "reasoning")
        self.assertEqual(last[call - 1].get("encrypted_content"), backend.state["calls"][last[call]["call_id"]],
                         "turn 2 must replay the encrypted_content issued with turn 1's function_call")
        self.assertNotIn("id", last[call - 1])

    def test_codex_login_expired_refreshes_once(self):
        path, before = self.codex_auth(expired=True)
        if os.name != "nt":
            os.chmod(path, 0o644)  # the write-back restores owner-only permissions
        oauth = self.mock("openai_oauth", refresh_tokens=[before["tokens"]["refresh_token"]], account_id=ACCOUNT)
        backend = self.mock("chatgpt_codex", account_id=ACCOUNT)
        res = self.launch("codex", self.codex_env(oauth, backend), "--auth", "login", "--debug",
                          codex_version_output="codex-cli 0.161.2")
        after = _read_json(path)
        info = self.assert_round_trip(res, "codex", _credentials(before) + _credentials(after), backend, oauth)
        self.assertEqual(refresh_count(oauth), 1, info)
        t0, t1 = before["tokens"], after["tokens"]
        self.assertNotEqual(t1["access_token"], t0["access_token"])
        self.assertNotEqual(t1["id_token"], t0["id_token"])
        self.assertEqual(t1["refresh_token"], token_chain(oauth).latest())
        self.assertEqual(t1["account_id"], ACCOUNT)
        self.assertGreater(compat.rfc3339_to_epoch(after["last_refresh"]), time.time() - 600)
        self.assertEqual(after["agent_identity"], {"keep": "me"})
        self.assertIsNone(after["OPENAI_API_KEY"])
        if os.name != "nt":
            self.assertEqual(_mode(path), 0o600)
        posts = backend.requests_for("/backend-api/codex/responses", "POST")
        self.assertEqual({_bearer(r) for r in posts}, {t1["access_token"]})
        self.assert_codex_requests(res, backend, posts, info)

    def test_codex_login_401_mid_session_refreshes_and_retries(self):
        path, before = self.codex_auth(expired=False)
        oauth = self.mock("openai_oauth", refresh_tokens=[before["tokens"]["refresh_token"]], account_id=ACCOUNT)
        backend = self.custom_mock(_unauthorized_once(2), account_id=ACCOUNT)
        res = self.launch("codex", self.codex_env(oauth, backend), "--auth", "login",
                          codex_version_output="codex-cli 0.161.2")
        after = _read_json(path)
        info = self.assert_round_trip(res, "codex", _credentials(before) + _credentials(after), backend, oauth)
        self.assertEqual(refresh_count(oauth), 1, info)
        posts = backend.requests_for("/backend-api/codex/responses", "POST")
        self.assertEqual(len(posts), 3, info)
        old, new = before["tokens"]["access_token"], after["tokens"]["access_token"]
        self.assertNotEqual(old, new)
        self.assertEqual([_bearer(r) for r in posts], [old, old, new])
        self.assertEqual(posts[1]["body_json"], posts[2]["body_json"], "the retry re-sends the same request")
        self.assertEqual(after["tokens"]["refresh_token"], token_chain(oauth).latest())
        self.assertEqual(after["agent_identity"], {"keep": "me"})
        self.assert_codex_requests(res, backend, [posts[0], posts[2]], info)

    # ---- gemini (Vertex via ADC) ----------------------------------------------------------------
    def assert_vertex(self, res, vertex, token, project, info):
        posts = vertex.requests_for("/v1/projects/", "POST")
        self.assertGreaterEqual(len(posts), 2, info)
        for rec in posts:
            self.assertEqual(_bearer(rec), token)
            self.assertEqual(_header(rec, "x-goog-user-project"), project)
            self.assertIsNone(_header(rec, "x-goog-api-key"))
            self.assertTrue(rec["path"].startswith(
                "/v1/projects/%s/locations/global/publishers/google/models/gemini-3.1-pro-preview:"
                "streamGenerateContent" % project), rec["path"])
            self.assertEqual(rec["query"].get("alt"), ["sse"])
        issued = vertex.state["issued"]
        model_turns = [c for c in posts[-1]["body_json"]["contents"] if c["role"] == "model"]
        calls = [p for c in model_turns for p in c["parts"] if "functionCall" in p]
        self.assertTrue(calls, info)
        self.assertIn(calls[0].get("thoughtSignature"), issued, "turn 2 must echo turn 1's thoughtSignature")
        self.assertTrue(vertex.state["sig_checks"])

    def test_gemini_adc_authorized_user(self):
        secret = "gcs-%s" % SENTINEL
        oauth = self.mock("google_oauth", refresh_token="g-rt-%s" % SENTINEL, client_secret=secret)
        adc = {"type": "authorized_user", "client_id": GOOGLE_CLIENT_ID, "client_secret": secret,
               "refresh_token": "g-rt-%s" % SENTINEL, "quota_project_id": "adc-quota-project",
               "universe_domain": "googleapis.com"}
        adc_path = _write_json(os.path.join(self.tmp, "adc", "creds.json"), adc)
        vertex = self.mock("vertex", token="ya29.mock-1", project="e2e-project", location="global")
        env = {"GOOGLE_APPLICATION_CREDENTIALS": adc_path, "GOOGLE_CLOUD_PROJECT": "e2e-project",
               "AI_GATEWAY_GOOGLE_TOKEN_URL": oauth.url + "/token", "AI_GATEWAY_UPSTREAM_GEMINI_VERTEX": vertex.url}
        res = self.launch("gemini", env, "--auth", "adc", "--debug", gcloud_token_rc=1)
        info = self.assert_round_trip(res, "gemini", _credentials(adc) + ["ya29.mock-1"], vertex, oauth)
        self.assertEqual(refresh_count(oauth), 1, info)
        self.assert_vertex(res, vertex, "ya29.mock-1", "e2e-project", info)
        self.assertEqual(fake_bins.read_calls(self.bins, "gcloud"), [], "authorized_user ADC never runs gcloud")
        self.assertEqual(_read_json(adc_path), adc, "the ADC file is never written")

    def test_gemini_adc_via_gcloud(self):
        token = "ya29.fake-gcloud-%s" % SENTINEL
        vertex = self.mock("vertex", token=token, project="fake-gcloud-project", location="global")
        env = {"AI_GATEWAY_UPSTREAM_GEMINI_VERTEX": vertex.url}
        res = self.launch("gemini", env, "--auth", "adc", "--debug", gcloud_token=token)
        info = self.assert_round_trip(res, "gemini", [token], vertex)
        self.assert_vertex(res, vertex, token, "fake-gcloud-project", info)
        calls = [c["argv"] for c in fake_bins.read_calls(self.bins, "gcloud")]
        self.assertEqual(calls.count(["auth", "application-default", "print-access-token"]), 1, calls)
        self.assertEqual(calls.count(["config", "get-value", "project"]), 1, calls)

    # ---- --model forms on the API-key routes ------------------------------------------------------
    def key_launch(self, launcher, kind, key_var, upstream_var, suffix, model):
        key = fake_key("%s-model" % launcher)
        srv = self.mock(kind, api_key=key)
        res = self.launch(launcher, {key_var: key, upstream_var: srv.url + suffix}, "--model", model)
        info = self.assert_round_trip(res, launcher, [key], srv)
        posts = srv.requests_for("", "POST")
        self.assertGreaterEqual(len(posts), 2, info)
        return res, {r["body_json"].get("model") for r in posts}, info

    def test_grok_wrap_picker_id(self):
        res, models, info = self.key_launch("grok", "xai_chat", "XAI_API_KEY", "AI_GATEWAY_UPSTREAM_XAI", "/v1",
                                            "claude-via-xai,grok-4.3")
        self.assertEqual(models, {"grok-4.3"}, info)
        self.assertIn("claude-via-xai,grok-4.3[1m]", res.stderr)

    def test_grok_wrap_legacy_fry_alias(self):
        _, models, info = self.key_launch("grok", "xai_chat", "XAI_API_KEY", "AI_GATEWAY_UPSTREAM_XAI", "/v1",
                                          "ollama,fry-grok-4-3")
        self.assertEqual(models, {"grok-4.3"}, info)

    def test_codex_wrap_bare_model_id(self):
        res, models, info = self.key_launch("codex", "openai_responses", "OPENAI_API_KEY",
                                            "AI_GATEWAY_UPSTREAM_OPENAI", "/v1", "gpt-5.4-mini")
        self.assertEqual(models, {"gpt-5.4-mini"}, info)
        self.assertIn("claude-via-openai,gpt-5.4-mini", res.stderr)


# ---------------------------------------------------------------------------------------
# always on: direct HTTP against a running Gateway
# ---------------------------------------------------------------------------------------

class GatewayHttpTests(LauncherTestCase):
    """What Claude Code's ``/model`` probe, discovery and ``/context`` do, without Claude Code."""

    def gateway(self, launcher, kind, key_var, upstream_var, suffix):
        from shared.gateway.server import Gateway

        mock_kind_or_skip(self, kind)
        key = fake_key(launcher)
        srv = MockServer(kind, options={"api_key": key}).start()
        self.addCleanup(srv.stop)
        os.environ.update({key_var: key, upstream_var: srv.url + suffix})
        launcher_obj = self.launcher(launcher)
        selected, store = launcher_obj.resolve("api-key")
        table = launcher_obj.route_table(selected)
        gw = Gateway(table, store, launcher_name="%s-wrap" % launcher).start()
        self.addCleanup(gw.stop)
        return gw, srv, launcher_obj.default_route(table)[2]

    def call(self, gw, method, path, body=None):
        from shared.gateway.transport import HttpClient

        client = HttpClient(timeout=30, connect_timeout=5)
        self.addCleanup(client.close)
        headers = {"Authorization": "Bearer %s" % gw.token, "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        resp = client.request(method, gw.url + path, headers, json.dumps(body) if body is not None else None,
                              stream=False)
        return resp.status, resp.headers, resp.text()

    @staticmethod
    def sse_types(text):
        return [json.loads(line[6:]).get("type") for line in text.splitlines() if line.startswith("data: ")]

    def check_route(self, launcher, kind, key_var, upstream_var, suffix, upstream_max_key, upstream_max):
        gw, srv, default_id = self.gateway(launcher, kind, key_var, upstream_var, suffix)
        probe = {"model": default_id, "max_tokens": 1, "messages": [{"role": "user", "content": "Hi"}]}
        for stream in (True, False):
            status, headers, text = self.call(gw, "POST", "/v1/messages?beta=true", dict(probe, stream=stream))
            self.assertEqual(status, 200, text)
            if stream:
                self.assertIn("text/event-stream", headers.get("content-type"))
                types = self.sse_types(text)
                self.assertEqual((types[0], types[-1]), ("message_start", "message_stop"), types)
                self.assertIn('"model":"%s"' % default_id, text.replace(" ", ""))
            else:
                msg = json.loads(text)
                self.assertEqual((msg["type"], msg["model"]), ("message", default_id))
                self.assertTrue(msg["stop_reason"])
        posts = srv.requests_for("", "POST")
        self.assertEqual(len(posts), 2)
        self.assertEqual({r["body_json"].get(upstream_max_key) for r in posts}, {upstream_max})
        self.assertTrue(all(r["body_json"].get("stream") is True for r in posts), "upstream is always streamed")

        unknown = dict(probe, model="claude-via-nosuch,model-x")
        for stream in (True, False):
            status, headers, text = self.call(gw, "POST", "/v1/messages?beta=true", dict(unknown, stream=stream))
            self.assertEqual(status, 404, text)
            err = json.loads(text)
            self.assertEqual((err["type"], err["error"]["type"]), ("error", "not_found_error"))
            self.assertIn("%s-wrap models" % launcher, err["error"]["message"])
            self.assertEqual(headers.get("x-should-retry"), "false")
        self.assertEqual(len(srv.requests_for("", "POST")), 2, "unknown models never reach the upstream")

        status, _, text = self.call(gw, "GET", "/v1/models?limit=1000")
        self.assertEqual(status, 200, text)
        models = json.loads(text)
        ids = [m["id"] for m in models["data"]]
        self.assertEqual(ids[0], default_id)
        self.assertTrue(all(i.startswith("claude-via-") for i in ids))
        self.assertIs(models["has_more"], False)
        self.assertEqual((models["first_id"], models["last_id"]), (ids[0], ids[-1]))
        status, _, text = self.call(gw, "GET", "/v1/models/" + default_id)
        self.assertEqual((status, json.loads(text)["id"]), (200, default_id))
        status, _, _ = self.call(gw, "GET", "/v1/models/claude-via-nosuch,model-x")
        self.assertEqual(status, 404)

        status, _, text = self.call(gw, "POST", "/v1/messages/count_tokens?beta=true",
                                    {"model": default_id, "messages": [{"role": "user", "content": "foo " * 400}]})
        self.assertEqual(status, 200, text)
        self.assertGreater(json.loads(text)["input_tokens"], 100)
        self.assertEqual(len(srv.requests), 2, "count_tokens and /v1/models are answered locally")
        status, _, _ = self.call(gw, "GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(srv.errors, [])

    def test_xai_chat_route(self):
        self.check_route("grok", "xai_chat", "XAI_API_KEY", "AI_GATEWAY_UPSTREAM_XAI", "/v1", "max_tokens", 1)

    def test_openai_responses_route(self):
        self.check_route("codex", "openai_responses", "OPENAI_API_KEY", "AI_GATEWAY_UPSTREAM_OPENAI", "/v1",
                         "max_output_tokens", 16)

    def test_unauthorized_client_rejected(self):
        gw, srv, default_id = self.gateway("grok", "xai_chat", "XAI_API_KEY", "AI_GATEWAY_UPSTREAM_XAI", "/v1")
        from shared.gateway.transport import HttpClient

        client = HttpClient(timeout=30, connect_timeout=5)
        self.addCleanup(client.close)
        resp = client.request("POST", gw.url + "/v1/messages", {"Authorization": "Bearer wrong"},
                              json.dumps({"model": default_id, "max_tokens": 1, "messages": []}), stream=False)
        self.assertEqual(resp.status, 401)
        resp.read()
        self.assertEqual(srv.requests, [])


# ---------------------------------------------------------------------------------------
# always on: doctor on login files
# ---------------------------------------------------------------------------------------

class DoctorLoginStateTests(LauncherTestCase):
    """``doctor --auth login|adc`` on valid / expired / missing credential files (no network)."""

    def doctor(self, launcher, auth):
        self.install_stub_claude(fetch=False)
        rc, out, err = self.run_cli(launcher, "doctor", "--auth", auth)
        self.assertNotIn(SENTINEL, out + err)
        return rc, out

    def codex_file(self, exp_delta, refresh=True):
        claim = {OPENAI_AUTH_CLAIM: {"chatgpt_account_id": ACCOUNT}}
        tokens = {"access_token": make_jwt(dict(claim, exp=int(time.time()) + exp_delta)), "account_id": ACCOUNT}
        if refresh:
            tokens["refresh_token"] = "rt-%s" % SENTINEL
        self.write_json(os.path.join(self.home, ".codex", "auth.json"), {"OPENAI_API_KEY": None, "tokens": tokens})

    def grok_file(self, exp_delta, refresh=True):
        entry = {"auth_mode": "oidc", "key": "xai-%s" % SENTINEL, "expires_at": _rfc3339(time.time() + exp_delta)}
        if refresh:
            entry["refresh_token"] = "rt-%s" % SENTINEL
        self.write_json(os.path.join(self.home, ".grok", "auth.json"), {GROK_ENTRY: entry})

    def check_states(self, launcher, write, tid):
        ok = "[ok] %s (login): available via %s login" % (tid, tid)
        rc, out = self.doctor(launcher, "login")
        self.assertEqual(rc, 1, out)
        self.assertIn("[warn] %s (login): unavailable — run `%s login`" % (tid, tid), out)
        self.assertIn("[FAIL] no usable transport", out)

        write(3600)
        rc, out = self.doctor(launcher, "login")
        self.assertEqual(rc, 0, out)
        self.assertIn(ok, out)
        self.assertNotIn("expired", out)

        write(-3600)
        rc, out = self.doctor(launcher, "login")
        self.assertEqual(rc, 0, out)
        self.assertIn(ok, out)
        self.assertIn("state=expired (refreshed on first request)", out)

        write(-3600, refresh=False)
        rc, out = self.doctor(launcher, "login")
        self.assertEqual(rc, 1, out)
        self.assertIn("[warn] %s (login): unavailable — run `%s login`" % (tid, tid), out)
        self.assertIn("state=expired, no refresh token", out)
        rc, _, err = self.run_cli(launcher, "launch", "claude", "--auth", "login", "--dry-run")
        self.assertEqual(rc, 2, err)
        self.assertIn("no usable transport", err)
        self.assertIn("state=expired, no refresh token", err)

    def test_codex_login_states(self):
        self.check_states("codex", self.codex_file, "codex")

    def test_grok_login_states(self):
        self.check_states("grok", self.grok_file, "grok")

    def test_vertex_adc_states(self):
        rc, out = self.doctor("gemini", "adc")
        self.assertEqual(rc, 1, out)
        self.assertIn("[warn] gemini-vertex (adc): unavailable — run `gcloud auth application-default login`", out)
        self.write_json(os.path.join(self.home, ".config", "gcloud", "application_default_credentials.json"),
                        {"type": "authorized_user", "client_id": "c", "client_secret": "s-%s" % SENTINEL,
                         "refresh_token": "r-%s" % SENTINEL, "quota_project_id": "quota-proj"})
        rc, out = self.doctor("gemini", "adc")
        self.assertEqual(rc, 0, out)
        self.assertIn("[ok] gemini-vertex (adc): available via gcloud ADC (", out)
        self.assertIn("source=adc authorized_user", out)
        self.assertIn("[ok] gemini-vertex: project quota-proj", out)


if __name__ == "__main__":
    unittest.main()
