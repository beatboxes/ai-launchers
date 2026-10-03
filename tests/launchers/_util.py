"""Sandbox helpers for launcher tests: temp HOME/AIL_HOME, scrubbed ``os.environ`` (no real keys, logins,
proxies or CLIs), in-process CLI runs with captured output, file-tree snapshots and a stub ``claude``."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared import base_launcher  # noqa: E402
from shared.gateway import secrets as gw_secrets  # noqa: E402

LAUNCHERS = ("grok", "codex", "gemini", "deepseek", "kimi")
SENTINEL = "FAKEKEY-SENTINEL"
KEY_ENV = {"grok": "XAI_API_KEY", "codex": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
           "deepseek": "DEEPSEEK_API_KEY", "kimi": "MOONSHOT_API_KEY"}
_SCRUB_PREFIXES = ("ANTHROPIC_", "CLAUDE_", "AI_GATEWAY_", "AI_LAUNCHERS_", "FRY_", "GOOGLE_", "GCLOUD_",
                   "CLOUDSDK_", "CODEX_", "GROK_", "OPENAI_", "XAI_", "GEMINI_", "DEEPSEEK_", "MOONSHOT_", "KIMI_",
                   "OP_", "STUB_")
_SCRUB_EXACT = {"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "AIL_HOME", "AIL_BIN"}

STUB_CLAUDE = textwrap.dedent('''\
    """Stub `claude`: records argv/env (+ one GET $ANTHROPIC_BASE_URL/v1/models) to $STUB_RECORD."""
    import json, os, sys, urllib.error, urllib.request
    if sys.argv[1:] == ["--version"]:
        print("2.1.288 (Claude Code)")
        sys.exit(0)
    if os.environ.get("STUB_SIGINT") == "1":  # Ctrl+C reaches the launcher while claude runs
        import signal, time
        os.kill(os.getppid(), signal.SIGINT)
        time.sleep(0.5)
    out = {"argv": sys.argv[1:], "env": dict(os.environ)}
    base = os.environ.get("ANTHROPIC_BASE_URL")
    if os.environ.get("STUB_FETCH") == "1" and base:
        req = urllib.request.Request(base + "/v1/models?limit=1000", headers={
            "Authorization": "Bearer " + os.environ.get("ANTHROPIC_AUTH_TOKEN", ""),
            "anthropic-version": "2023-06-01"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=10) as r:
                out["models_status"] = r.status
                out["models"] = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            out["models_status"] = e.code
        except Exception as e:
            out["models_error"] = repr(e)
    with open(os.environ["STUB_RECORD"], "w", encoding="utf-8") as f:
        json.dump(out, f)
    sys.exit(int(os.environ.get("STUB_RC", "0")))
''')


def manifest_path(launcher):
    return os.path.join(REPO_ROOT, launcher, "config.example.json")


def fake_key(tag):
    return "sk-%s-%s-0123456789abcdef" % (tag, SENTINEL)


def _system_path():
    if os.name == "nt":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        return [os.path.join(root, "System32"), root]
    return [d for d in ("/usr/bin", "/bin") if os.path.isdir(d)]


class LauncherTestCase(unittest.TestCase):
    """Every test runs with HOME/USERPROFILE/AIL_HOME/CODEX_HOME/GROK_HOME/CLOUDSDK_CONFIG/CLAUDE_CONFIG_DIR
    inside a fresh temp dir, PATH = an empty temp bin dir + the system dirs (no gcloud/claude/op)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ail-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.home = os.path.join(self.tmp, "home")
        self.ail = os.path.join(self.home, ".ai-launchers")
        self.bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.home)
        os.makedirs(self.bin)
        env = {k: v for k, v in os.environ.items()
               if not k.upper().startswith(_SCRUB_PREFIXES) and k.upper() not in _SCRUB_EXACT}
        env.update({
            "HOME": self.home, "USERPROFILE": self.home, "AIL_HOME": self.ail,
            "CODEX_HOME": os.path.join(self.home, ".codex"), "GROK_HOME": os.path.join(self.home, ".grok"),
            "CLOUDSDK_CONFIG": os.path.join(self.home, ".config", "gcloud"),
            "CLAUDE_CONFIG_DIR": os.path.join(self.home, ".claude"),
            "PATH": os.pathsep.join([self.bin] + _system_path()),
        })
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        gw_secrets.clear_cache()
        self.addCleanup(gw_secrets.clear_cache)

    # ---- helpers ---------------------------------------------------------------------------
    def run_cli(self, launcher, *argv, **kw):
        """Run ``<launcher>-wrap <argv>`` in-process -> (rc, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            if "stdin" in kw:
                stack.enter_context(mock.patch.object(sys, "stdin", kw["stdin"]))
            rc = base_launcher.run(manifest_path(launcher), list(argv))
        return rc, out.getvalue(), err.getvalue()

    def launcher(self, name, config=None):
        return base_launcher.Launcher(base_launcher.load_manifest(manifest_path(name)), config=config)

    def write_json(self, path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        return path

    def write_config(self, cfg):
        return self.write_json(os.path.join(self.ail, "config.json"), cfg)

    def snapshot(self):
        """{relative path: (is_dir, size, mtime_ns, mode)} for everything under the temp root."""
        snap = {}
        for dirpath, dirnames, filenames in os.walk(self.tmp):
            for name in dirnames + filenames:
                full = os.path.join(dirpath, name)
                st = os.lstat(full)
                snap[os.path.relpath(full, self.tmp)] = (name in dirnames, st.st_size, st.st_mtime_ns, st.st_mode)
        return snap

    def install_stub_claude(self, rc=0, fetch=True):
        """Point AI_LAUNCHERS_CLAUDE_BIN at a Python stub; returns the record path it writes."""
        stub = os.path.join(self.tmp, "stub_claude.py")
        with open(stub, "w", encoding="utf-8") as f:
            f.write(STUB_CLAUDE)
        record = os.path.join(self.tmp, "claude_record.json")
        os.environ.update({"AI_LAUNCHERS_CLAUDE_BIN": stub, "STUB_RECORD": record, "STUB_RC": str(rc),
                           "STUB_FETCH": "1" if fetch else "0"})
        return record

    @staticmethod
    def read_record(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def mock_kind_or_skip(testcase, kind):
    """``testing.mock_upstreams.get_kind_factory(kind)`` or skip with a clear message."""
    from shared.gateway.testing import mock_upstreams

    try:
        mock_upstreams.get_kind_factory(kind)
    except KeyError:
        testcase.skipTest("mock upstream kind %r is not available yet (owned by another component)" % kind)
    return mock_upstreams
