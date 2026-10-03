"""launchkit: child env hygiene, find_claude precedence, run_child signals/rc, claude_version."""

import contextlib
import io
import os
import signal
import stat
import subprocess
import sys
import tempfile
import shutil
import textwrap
import time
import unittest

from ._pkg import PKG, REPO_ROOT, mod

lk = mod("launchkit")
config = mod("config")

SECRET = "FAKEKEY-SENTINEL-xai-0123456789"
POSIX = os.name != "nt"


def _executable(path, body):
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class TempDirMixin(object):
    def mkdtemp(self):
        d = tempfile.mkdtemp(prefix="gw-launchkit-")
        self.addCleanup(shutil.rmtree, d, True)
        return d


# =========================================================================================
# environments
# =========================================================================================

DIRTY = {
    "PATH": "/usr/bin", "HOME": "/home/u", "KEEP_ME": "1",
    "ANTHROPIC_API_KEY": "sk-ant-real", "ANTHROPIC_AUTH_TOKEN": "old", "ANTHROPIC_MODEL": "opus",
    "ANTHROPIC_SMALL_FAST_MODEL": "x", "ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION": "us-east-1",
    "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME": "n", "ANTHROPIC_DEFAULT_OPUS_MODEL_DESCRIPTION": "d",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL_SUPPORTED_CAPABILITIES": "c", "ANTHROPIC_UNIX_SOCKET": "/tmp/s",
    "ANTHROPIC_CUSTOM_HEADERS": "X-A: b", "ANTHROPIC_BETAS": "foo", "CLAUDE_CODE_USE_BEDROCK": "1",
    "CLAUDE_CODE_USE_VERTEX": "1", "CLAUDE_CODE_USE_GATEWAY": "1", "CLAUDE_CODE_USE_CCR_V2": "1",
    "CLAUDE_CODE_OAUTH_TOKEN": "tok", "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "3",
    "_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL": "1", "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1000",
    "CLAUDE_CODE_GZIP_REQUEST_BODIES": "1", "NO_PROXY": "corp.example, localhost", "no_proxy": "corp.example",
    "XAI_API_KEY": SECRET, "SOME_URL": "https://x/?k=" + SECRET,
}
UNSET = ("ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION", "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
         "ANTHROPIC_DEFAULT_OPUS_MODEL_DESCRIPTION", "ANTHROPIC_DEFAULT_HAIKU_MODEL_SUPPORTED_CAPABILITIES",
         "ANTHROPIC_UNIX_SOCKET", "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_BETAS", "CLAUDE_CODE_USE_BEDROCK",
         "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_GATEWAY", "CLAUDE_CODE_USE_CCR_V2", "CLAUDE_CODE_OAUTH_TOKEN",
         "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR", "_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL",
         "CLAUDE_CODE_MAX_CONTEXT_TOKENS", "CLAUDE_CODE_GZIP_REQUEST_BODIES")


@unittest.skipUnless(POSIX, "exact-case env semantics (Windows covered by EnvCaseTests)")
class ChildEnvTests(unittest.TestCase):
    def build(self, **kw):
        args = dict(base_env=dict(DIRTY), base_url="http://127.0.0.1:5555/", token="gw-token",
                    default_id="claude-via-xai,grok-4.7[1m]", background_id="claude-via-background",
                    context=2000000, secret_values=config.SecretStore({"xai": SECRET}))
        args.update(kw)
        return lk.build_child_env(**args)

    def test_set_unset_matrix(self):
        base = dict(DIRTY)
        env = self.build(base_env=base)
        self.assertEqual(base, DIRTY, "base_env must not be mutated")
        expected = {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:5555", "ANTHROPIC_AUTH_TOKEN": "gw-token",
            "ANTHROPIC_API_KEY": "", "ANTHROPIC_MODEL": "claude-via-xai,grok-4.7[1m]",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-via-xai,grok-4.7[1m]",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-via-xai,grok-4.7[1m]",
            "ANTHROPIC_DEFAULT_FABLE_MODEL": "claude-via-xai,grok-4.7[1m]",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-via-background",
            "ANTHROPIC_CUSTOM_MODEL_OPTION": "claude-via-xai,grok-4.7[1m]",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "grok-4.7 (xai)",
            "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "KEEP_ME": "1", "PATH": "/usr/bin", "HOME": "/home/u",
            "NO_PROXY": "corp.example,localhost,127.0.0.1,::1", "no_proxy": "corp.example,localhost,127.0.0.1,::1",
        }
        for k, v in expected.items():
            self.assertEqual(env.get(k), v, k)
        self.assertTrue(env["ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION"])
        for k in UNSET + ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_CODE_USE_GATEWAY"):
            self.assertNotIn(k, env)
        # provider secrets never reach the child
        self.assertNotIn("XAI_API_KEY", env)
        self.assertNotIn("SOME_URL", env)
        self.assertFalse([k for k, v in env.items() if SECRET in v])
        self.assertTrue(all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()))

    def test_compact_window_only_below_200k(self):
        self.assertEqual(self.build(context=128000)["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "128000")
        self.assertEqual(self.build(context=32000)["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "100000")
        self.assertNotIn("CLAUDE_CODE_AUTO_COMPACT_WINDOW", self.build(context=200000))
        self.assertNotIn("CLAUDE_CODE_AUTO_COMPACT_WINDOW", self.build(context=None))
        self.assertNotIn("CLAUDE_CODE_MAX_CONTEXT_TOKENS", self.build(context=128000))

    def test_no_proxy_without_duplicates_and_missing(self):
        base = {"PATH": "/bin"}
        env = self.build(base_env=base, secret_values=None)
        self.assertEqual(env["NO_PROXY"], "127.0.0.1,localhost,::1")
        self.assertEqual(env["no_proxy"], env["NO_PROXY"])
        env = self.build(base_env={"no_proxy": "::1,127.0.0.1,localhost"}, secret_values=None)
        self.assertEqual(env["NO_PROXY"], "::1,127.0.0.1,localhost")

    def test_extra_and_secret_guards(self):
        env = self.build(extra={"CLAUDE_CODE_SUBAGENT_MODEL": "claude-via-xai,grok-4.3", "KEEP_ME": None,
                                "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": 1})
        self.assertEqual(env["CLAUDE_CODE_SUBAGENT_MODEL"], "claude-via-xai,grok-4.3")
        self.assertEqual(env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"], "1")
        self.assertNotIn("KEEP_ME", env)
        for kw in ({"extra": {"LEAK": "x" + SECRET}}, {"token": SECRET}):
            with self.assertRaises(ValueError) as cm:
                self.build(**kw)
            self.assertNotIn(SECRET, str(cm.exception))
        env = self.build(secret_values=[SECRET, "", None, "abc"])  # list form; short values ignored
        self.assertNotIn("XAI_API_KEY", env)
        env = self.build(secret_values=None)
        self.assertEqual(env["XAI_API_KEY"], SECRET)  # nothing to scrub without the secret list

    def test_direct_env(self):
        base = dict(DIRTY, DEEPSEEK_API_KEY=SECRET, CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY="1")
        models = {"MODEL": "deepseek-v4-pro[1m]", "OPUS": "deepseek-v4-pro[1m]", "SONNET": "deepseek-v4-pro[1m]",
                  "FABLE": "deepseek-v4-pro[1m]", "HAIKU": "deepseek-v4-flash[1m]", "SUBAGENT": "deepseek-v4-flash[1m]"}
        env = lk.build_direct_env(base, "https://api.deepseek.com/anthropic", SECRET, models,
                                  extra={"CLAUDE_CODE_EFFORT_LEVEL": "max"}, secret_values=[SECRET, "sk-x-other"])
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "https://api.deepseek.com/anthropic")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], SECRET)
        self.assertEqual(env["ANTHROPIC_API_KEY"], "")
        self.assertEqual(env["ANTHROPIC_MODEL"], "deepseek-v4-pro[1m]")
        self.assertEqual(env["ANTHROPIC_DEFAULT_FABLE_MODEL"], "deepseek-v4-pro[1m]")
        self.assertEqual(env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "deepseek-v4-flash[1m]")
        self.assertEqual(env["CLAUDE_CODE_SUBAGENT_MODEL"], "deepseek-v4-flash[1m]")
        self.assertEqual(env["CLAUDE_CODE_EFFORT_LEVEL"], "max")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")
        self.assertNotIn("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", env)
        self.assertNotIn("ANTHROPIC_CUSTOM_MODEL_OPTION", env)
        for k in UNSET:
            self.assertNotIn(k, env)
        self.assertEqual([k for k, v in env.items() if SECRET in v], ["ANTHROPIC_AUTH_TOKEN"])
        with self.assertRaises(ValueError):
            lk.build_direct_env(base, "https://x", "k", {"BOGUS": "m"})
        with self.assertRaises(ValueError):
            lk.build_direct_env(base, "https://x", SECRET, {"MODEL": "m"}, extra={"X": SECRET},
                                secret_values=[SECRET])
        oll = lk.build_direct_env({"PATH": "/bin"}, "http://localhost:11434", "ollama",
                                  {"MODEL": "qwen3:8b", "HAIKU": "qwen3:8b"}, context=40000)
        self.assertEqual(oll["ANTHROPIC_AUTH_TOKEN"], "ollama")
        self.assertIn("localhost", oll["NO_PROXY"].split(","))
        self.assertEqual(oll["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "100000")
        self.assertNotIn("ANTHROPIC_DEFAULT_OPUS_MODEL", oll)


class EnvCaseTests(unittest.TestCase):
    """Windows env names are case-insensitive: one NO_PROXY, case-variant unsets."""

    def test_windows_semantics(self):
        env = lk._Env({"Path": "C:\\bin", "no_proxy": "corp", "anthropic_betas": "x", "Claude_Code_Use_Vertex": "1",
                       "ANTHROPIC_MODEL": "a"}, True)
        env.unset_matching()
        env.add_no_proxy()
        env.set("anthropic_model", "b")
        self.assertEqual(env.d, {"Path": "C:\\bin", "NO_PROXY": "corp,127.0.0.1,localhost,::1",
                                 "anthropic_model": "b"})


# =========================================================================================
# find_claude / claude_version
# =========================================================================================

class FindClaudeTests(TempDirMixin, unittest.TestCase):
    def setUp(self):
        self.tmp = self.mkdtemp()
        self.home = os.path.join(self.tmp, "home")
        os.makedirs(self.home)
        self.empty = os.path.join(self.tmp, "empty")
        os.makedirs(self.empty)
        self.stub = os.path.join(self.tmp, "claude_stub.py")
        with open(self.stub, "w", encoding="utf-8") as f:
            f.write("import sys\nprint('2.1.288 (Claude Code)')\n")

    def env(self, **kw):
        e = {"PATH": self.empty, "HOME": self.home, "USERPROFILE": self.home}
        e.update(kw)
        return e

    def test_env_override_precedence(self):
        other = os.path.join(self.tmp, "other.py")
        shutil.copy(self.stub, other)
        self.assertEqual(lk.find_claude(self.env(AI_LAUNCHERS_CLAUDE_BIN=self.stub, FRY_CLAUDE_BIN=other)),
                         [sys.executable, self.stub])
        self.assertEqual(lk.find_claude(self.env(FRY_CLAUDE_BIN=other)), [sys.executable, other])
        with self.assertRaises(lk.ClaudeNotFound):
            lk.find_claude(self.env(AI_LAUNCHERS_CLAUDE_BIN=os.path.join(self.tmp, "missing")))
        js = os.path.join(self.tmp, "cli.js")
        open(js, "w").close()
        bindir = os.path.join(self.tmp, "nodebin")
        os.makedirs(bindir)
        node = _executable(os.path.join(bindir, "node.exe" if os.name == "nt" else "node"), "#!/bin/sh\n")
        self.assertEqual(lk.find_claude(self.env(AI_LAUNCHERS_CLAUDE_BIN=js, PATH=bindir)), [node, js])

    @unittest.skipUnless(POSIX, "POSIX PATH lookup")
    def test_native_on_path_and_home_fallback(self):
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        native = _executable(os.path.join(bindir, "claude"), "#!/bin/sh\necho 2.1.288\n")
        self.assertEqual(lk.find_claude(self.env(PATH=bindir)), [native])
        self.assertEqual(lk.find_claude(self.env(PATH=bindir, AI_LAUNCHERS_CLAUDE_BIN=self.stub)),
                         [sys.executable, self.stub])
        with self.assertRaises(lk.ClaudeNotFound) as cm:
            lk.find_claude(self.env())
        self.assertIn("npm install -g @anthropic-ai/claude-code", str(cm.exception))
        local = os.path.join(self.home, ".local", "bin")
        os.makedirs(local)
        home_claude = _executable(os.path.join(local, "claude"), "#!/bin/sh\n")
        self.assertEqual(lk.find_claude(self.env()), [home_claude])

    def test_npm_shim_resolution(self):
        prefix = os.path.join(self.tmp, "npm")
        os.makedirs(prefix)
        shim = os.path.join(prefix, "claude.cmd")
        open(shim, "w").close()
        nodebin = os.path.join(self.tmp, "nodebin")
        os.makedirs(nodebin)
        node = _executable(os.path.join(nodebin, "node.exe" if os.name == "nt" else "node"), "#!/bin/sh\n")
        environ = self.env(PATH=nodebin, ComSpec="C:\\Windows\\cmd.exe")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(lk._npm_shim_argv(shim, environ, True), ["C:\\Windows\\cmd.exe", "/c", shim])
        self.assertIn("cmd.exe", err.getvalue())
        self.assertIsNone(lk._npm_shim_argv(shim, environ, False))
        cli_js = os.path.join(prefix, "node_modules", "@anthropic-ai", "claude-code", "cli.js")
        os.makedirs(os.path.dirname(cli_js))
        open(cli_js, "w").close()
        self.assertEqual(lk._npm_shim_argv(shim, environ, True), [node, cli_js])
        local_node = _executable(os.path.join(prefix, "node.exe"), "")  # node next to the shim wins (npm's rule)
        self.assertEqual(lk._npm_shim_argv(shim, environ, True), [local_node, cli_js])

    def test_claude_version(self):
        self.assertEqual(lk.claude_version([sys.executable, self.stub]), "2.1.288")
        bad = os.path.join(self.tmp, "bad.py")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("import sys\nsys.exit(2)\n")
        self.assertIsNone(lk.claude_version([sys.executable, bad]))
        self.assertIsNone(lk.claude_version([os.path.join(self.tmp, "nope")]))


# =========================================================================================
# run_child
# =========================================================================================

RUNNER = textwrap.dedent("""
    import os, signal, sys
    sys.path.insert(0, %(root)r)
    import importlib
    lk = importlib.import_module(%(pkg)r + ".launchkit")
    hits = []
    def record(signum, frame):
        hits.append(signum)
    signal.signal(signal.SIGINT, record)
    rc = lk.run_child([sys.executable, %(child)r], dict(os.environ))
    print("RC=%%d RESTORED=%%s HITS=%%d" %% (rc, signal.getsignal(signal.SIGINT) is record, len(hits)))
    sys.stdout.flush()
""")

CHILD_SIGINT = textwrap.dedent("""
    import os, signal, sys, time
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:   # SIG_IGN would have been inherited across exec
        sys.exit(43)
    got = []
    signal.signal(signal.SIGINT, lambda s, f: got.append(s))
    os.killpg(os.getpgrp(), signal.SIGINT)   # what a terminal Ctrl+C does: the whole foreground group
    time.sleep(0.5)
    sys.exit(42 if got else 1)
""")

CHILD_SIGTERM = textwrap.dedent("""
    import os, signal, sys, time
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(5))
    open(%(ready)r, "w").close()
    time.sleep(30)
    sys.exit(9)
""")


class RunChildTests(TempDirMixin, unittest.TestCase):
    def test_return_codes(self):
        self.assertEqual(lk.run_child([sys.executable, "-c", "import sys; sys.exit(7)"], dict(os.environ)), 7)
        tmp = self.mkdtemp()
        rc = lk.run_child([sys.executable, "-c", "import os; open('here', 'w').close()"], dict(os.environ), cwd=tmp)
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(os.path.join(tmp, "here")))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(lk.run_child([os.path.join(tmp, "no-such-claude")], dict(os.environ)), 127)
        self.assertIn("cannot run", err.getvalue())
        self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)
        if POSIX:
            plain = os.path.join(tmp, "not-executable")
            with open(plain, "w") as f:
                f.write("#!/bin/sh\n")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(lk.run_child([plain], dict(os.environ)), 126)

    @unittest.skipUnless(POSIX, "POSIX signals")
    def test_signal_death_maps_to_128_plus_n(self):
        rc = lk.run_child([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
                          dict(os.environ))
        self.assertEqual(rc, 128 + signal.SIGKILL)

    def _runner(self, child_src, **fmt):
        tmp = self.mkdtemp()
        child = os.path.join(tmp, "child.py")
        with open(child, "w", encoding="utf-8") as f:
            f.write(child_src % fmt if fmt else child_src)
        runner = os.path.join(tmp, "runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(RUNNER % {"root": REPO_ROOT, "pkg": PKG, "child": child})
        # own session/process group so the child's group-wide SIGINT cannot reach the test runner
        return subprocess.Popen([sys.executable, runner], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, start_new_session=True)

    @unittest.skipUnless(POSIX, "POSIX process groups")
    def test_parent_survives_group_sigint_and_returns_child_rc(self):
        proc = self._runner(CHILD_SIGINT)
        out, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 0, err)
        self.assertIn("RC=42 RESTORED=True HITS=0", out)

    @unittest.skipUnless(POSIX, "POSIX signal forwarding")
    def test_sigterm_forwarded_to_child(self):
        ready = os.path.join(self.mkdtemp(), "ready")
        proc = self._runner(CHILD_SIGTERM, ready=ready)
        try:
            deadline = time.time() + 30
            while not os.path.exists(ready) and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(os.path.exists(ready), "child never started")
            proc.send_signal(signal.SIGTERM)
            out, err = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertEqual(proc.returncode, 0, err)
        self.assertIn("RC=5 RESTORED=True HITS=0", out)


if __name__ == "__main__":
    unittest.main()
