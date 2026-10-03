"""`launch claude` end to end with a stub `claude` (AI_LAUNCHERS_CLAUDE_BIN -> a Python script that records
argv/env and calls GET $ANTHROPIC_BASE_URL/v1/models): env hygiene, gateway reachable while the child
runs and stopped afterwards, exit-code passthrough, Claude config files untouched, one-time ToS notice."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest

from ._util import KEY_ENV, LAUNCHERS, REPO_ROOT, SENTINEL, LauncherTestCase, fake_key
from shared.gateway import compat


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _gateway_threads():
    return [t for t in threading.enumerate() if t.name.startswith("ai-gateway")]


def _wait_no_gateway_threads(timeout=5.0):
    deadline = time.time() + timeout
    while _gateway_threads() and time.time() < deadline:
        time.sleep(0.05)
    return _gateway_threads()


class GatewayLaunchTests(LauncherTestCase):
    def seed_claude_config(self):
        settings = os.path.join(self.home, ".claude", "settings.json")
        claude_json = os.path.join(self.home, ".claude.json")
        self.write_json(settings, {"model": "sonnet"})
        self.write_json(claude_json, {"additionalModelOptionsCache": [{"value": "user-entry"}]})
        return {p: _read(p) for p in (settings, claude_json)}

    def test_env_hygiene_gateway_and_cleanup(self):
        key = fake_key("xai")
        os.environ["XAI_API_KEY"] = key
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-inherited"
        os.environ["ANTHROPIC_SMALL_FAST_MODEL"] = "claude-haiku"
        seeded = self.seed_claude_config()
        record = self.install_stub_claude(rc=7)
        rc, out, err = self.run_cli("grok", "launch", "claude", "--", "--foo", "bar baz")
        self.assertEqual(rc, 7, err)
        rec = self.read_record(record)
        env = rec["env"]
        self.assertEqual(rec["argv"], ["--foo", "bar baz"])
        self.assertEqual(env["ANTHROPIC_API_KEY"], "")
        self.assertNotIn("XAI_API_KEY", env)
        self.assertNotIn("ANTHROPIC_SMALL_FAST_MODEL", env)
        self.assertFalse([k for k, v in env.items() if SENTINEL in v], "provider key leaked into the child env")
        self.assertGreaterEqual(len(env["ANTHROPIC_AUTH_TOKEN"]), 20)
        default = "claude-via-xai,grok-4.7[1m]"
        for var in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                    "ANTHROPIC_DEFAULT_FABLE_MODEL"):
            self.assertEqual(env[var], default, var)
        self.assertEqual(env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "claude-via-background")
        self.assertEqual(env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"], "1")
        self.assertIn("127.0.0.1", env["NO_PROXY"])
        # the gateway answered the child (with the per-launch token) ...
        self.assertEqual(rec.get("models_status"), 200, rec.get("models_error"))
        ids = [m["id"] for m in rec["models"]["data"]]
        self.assertEqual(ids[0], default)
        self.assertTrue(all(i.startswith("claude-via-") for i in ids))
        # ... and is gone after the child exited
        port = int(env["ANTHROPIC_BASE_URL"].rsplit(":", 1)[1])
        self.assertEqual(_wait_no_gateway_threads(), [])
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
        # Claude Code's own config is never touched; logs are written, without secrets
        for path, data in seeded.items():
            self.assertEqual(_read(path), data, path)
        log = os.path.join(self.ail, "logs", "grok-wrap.log")
        self.assertTrue(os.path.isfile(log))
        self.assertNotIn(SENTINEL.encode(), _read(log))
        self.assertIn("grok-wrap: Claude Code -> %s via xai [env:XAI_API_KEY]" % default, err)
        self.assertNotIn(SENTINEL, out + err)
        self.assertFalse(os.path.exists(os.path.join(self.ail, "state.json")))

    def test_debug_never_prints_token_or_key(self):
        os.environ["GEMINI_API_KEY"] = fake_key("gemini")
        record = self.install_stub_claude(fetch=False)
        rc, out, err = self.run_cli("gemini", "launch", "claude", "--debug", "--model", "gemini-3.8-flash")
        self.assertEqual(rc, 0, err)
        env = self.read_record(record)["env"]
        self.assertEqual(env["ANTHROPIC_MODEL"], "claude-via-gemini,gemini-3.8-flash[1m]")
        self.assertIn("child environment changes", err)
        self.assertIn("ANTHROPIC_AUTH_TOKEN=<per-launch random token>", err)
        self.assertNotIn(env["ANTHROPIC_AUTH_TOKEN"], out + err)
        self.assertNotIn(SENTINEL, out + err)
        self.assertTrue(os.path.isfile(os.path.join(self.ail, "logs", "gemini-wrap.log")))

    def test_tos_notice_shown_once_for_login_routes(self):
        claims = {"exp": int(time.time()) + 3600, "https://api.openai.com/auth": {"chatgpt_account_id": "acct"}}
        token = "%s.%s.sig" % (compat.b64url_encode('{"alg":"none"}'), compat.b64url_encode(json.dumps(claims)))
        self.write_json(os.path.join(self.home, ".codex", "auth.json"), {
            "OPENAI_API_KEY": None, "tokens": {"access_token": token, "id_token": token, "refresh_token": "rt",
                                               "account_id": "acct"}})
        self.install_stub_claude(fetch=False)
        rc, _, err = self.run_cli("codex", "launch", "claude")
        self.assertEqual(rc, 0, err)
        self.assertIn("terms of service", err)
        with open(os.path.join(self.ail, "state.json"), encoding="utf-8") as f:
            self.assertIn("tos:codex", json.load(f)["notices"])
        rc, _, err = self.run_cli("codex", "launch", "claude")
        self.assertEqual(rc, 0, err)
        self.assertNotIn("terms of service", err)

    def test_claude_missing(self):
        os.environ["XAI_API_KEY"] = fake_key("xai")
        os.environ["AI_LAUNCHERS_CLAUDE_BIN"] = os.path.join(self.tmp, "no-such-claude")
        rc, _, err = self.run_cli("grok", "launch", "claude")
        self.assertEqual(rc, 2)
        self.assertIn("does not exist", err)
        self.assertEqual(_gateway_threads(), [])
        self.assertFalse(os.path.exists(os.path.join(self.ail, "logs")))

    def test_busy_port(self):
        os.environ["XAI_API_KEY"] = fake_key("xai")
        self.install_stub_claude(fetch=False)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        self.addCleanup(sock.close)
        rc, _, err = self.run_cli("grok", "launch", "claude", "--port", str(sock.getsockname()[1]))
        self.assertEqual(rc, 3)
        self.assertIn("cannot start the gateway", err)
        self.assertEqual(_gateway_threads(), [])

    def test_config_port(self):
        os.environ["XAI_API_KEY"] = fake_key("xai")
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        self.write_config({"gateway": {"port": port}})
        record = self.install_stub_claude(fetch=False)
        rc, _, err = self.run_cli("grok", "launch", "claude")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.read_record(record)["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:%d" % port)


class AllLaunchersHygieneTests(LauncherTestCase):
    """Every launcher (incl. DeepSeek/Kimi passthrough) keeps the provider key out of Claude Code's env."""

    DEFAULTS = {"grok": "claude-via-xai,grok-4.7[1m]", "codex": "claude-via-openai,gpt-5.5",
                "gemini": "claude-via-gemini,gemini-3.1-pro-preview[1m]",
                "deepseek": "claude-via-deepseek,deepseek-v4-pro[1m]", "kimi": "claude-via-kimi,kimi-k3[1m]"}

    def test_provider_key_never_in_child_env(self):
        for name in LAUNCHERS:
            with self.subTest(launcher=name):
                key = fake_key("%s-hygiene" % name)
                for var in KEY_ENV.values():
                    os.environ.pop(var, None)
                os.environ[KEY_ENV[name]] = key
                os.environ["SOME_COPY"] = "copied:%s" % key
                record = self.install_stub_claude(rc=0, fetch=True)
                rc, out, err = self.run_cli(name, "launch", "claude", "--", "-p", "hi")
                self.assertEqual(rc, 0, err)
                rec = self.read_record(record)
                env = rec["env"]
                self.assertEqual([k for k, v in env.items() if key in v or SENTINEL in v], [],
                                 "provider key leaked into the child env")
                self.assertNotIn(key, " ".join(rec["argv"]))
                self.assertNotIn(KEY_ENV[name], env)
                self.assertEqual(env["ANTHROPIC_API_KEY"], "")
                self.assertTrue(env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:"))
                self.assertEqual(env["ANTHROPIC_MODEL"], self.DEFAULTS[name])
                self.assertEqual(env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "claude-via-background")
                self.assertEqual(env.get("CLAUDE_CODE_EFFORT_LEVEL"), "max" if name == "deepseek" else None)
                self.assertEqual(rec.get("models_status"), 200, rec.get("models_error"))
                self.assertEqual(rec["models"]["data"][0]["id"], self.DEFAULTS[name])
                self.assertNotIn(SENTINEL, out + err)
                self.assertEqual(_wait_no_gateway_threads(), [])


class SubprocessLaunchTests(LauncherTestCase):
    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_sigint_while_claude_runs(self):
        """Ctrl+C belongs to Claude Code: the launcher survives it and still cleans up."""
        os.environ["XAI_API_KEY"] = fake_key("xai")
        os.environ["STUB_SIGINT"] = "1"
        record = self.install_stub_claude(rc=0, fetch=True)
        proc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "grok", "grok-wrap.py"), "launch", "claude"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, universal_newlines=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        rec = self.read_record(record)
        self.assertEqual(rec.get("models_status"), 200, rec)

    def test_real_entry_point_and_overhead(self):
        """The installed entry point (`python grok/grok-wrap.py`) reaches claude quickly."""
        os.environ["XAI_API_KEY"] = fake_key("xai")
        record = self.install_stub_claude(fetch=False)
        stub = os.environ["AI_LAUNCHERS_CLAUDE_BIN"]
        src = _read(stub).decode("utf-8")  # record the stub's start time too
        with open(stub, "w", encoding="utf-8") as f:
            f.write("import time as _t; _START = _t.time()\n" + src.replace(
                '"env": dict(os.environ)}', '"env": dict(os.environ), "start": _START}'))
        t0 = time.time()
        proc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "grok", "grok-wrap.py"), "launch", "claude"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, universal_newlines=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        overhead = self.read_record(record)["start"] - t0
        # interpreter start + imports + secret resolution + gateway start + the stub's own interpreter
        # start (~0.1 s measured); the bound is generous for slow CI machines
        self.assertLess(overhead, 1.0, "launch overhead %.3fs" % overhead)


if __name__ == "__main__":
    unittest.main()
