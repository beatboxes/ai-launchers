"""`launch claude --dry-run` prints the plan and touches nothing: no files, threads, subprocesses, sockets,
1Password lookups or gcloud calls (v0.1 bug H1: dry-run spawned the bridge and wrote CCR config)."""

import contextlib
import os
import socket
import subprocess
import threading
import unittest
from unittest import mock

from ._util import DIRECT_LAUNCHERS, KEY_ENV, LAUNCHERS, SENTINEL, LauncherTestCase, fake_key


def _forbid(what, violations):
    """Records the attempt (the launcher may swallow exceptions from auth modules) and raises."""
    def fail(*a, **k):
        violations.append(what)
        raise AssertionError("dry-run must not %s" % what)
    return fail


class DryRunTests(LauncherTestCase):
    def dry_run(self, launcher, *extra):
        threads_before = set(threading.enumerate())
        snap = self.snapshot()
        violations = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(subprocess, "Popen", side_effect=_forbid("spawn", violations)))
            stack.enter_context(mock.patch.object(socket.socket, "connect", _forbid("connect", violations)))
            stack.enter_context(mock.patch.object(socket.socket, "bind", _forbid("bind", violations)))
            stack.enter_context(mock.patch("shared.gateway.secrets.load_into",
                                           side_effect=_forbid("resolve secrets", violations)))
            stack.enter_context(mock.patch("threading.Thread.start", _forbid("start threads", violations)))
            rc, out, err = self.run_cli(launcher, "launch", "claude", "--dry-run", *extra)
        self.assertEqual(violations, [])
        self.assertEqual(set(threading.enumerate()), threads_before)
        self.assertEqual(self.snapshot(), snap, "dry-run changed the file system")
        return rc, out, err

    def test_every_launcher(self):
        from shared.gateway.testing import fake_bins

        fake_bins.make_fake_bins(self.bin)  # gcloud/codex/grok/op on PATH must not be invoked either
        os.environ.update({var: fake_key(name) for name, var in KEY_ENV.items()})
        self.write_config({"providers": {"xai": {"op_ref": "op://Personal/xAI/credential"}}})
        stub = os.path.join(self.tmp, "claude-stub.py")
        with open(stub, "w") as f:
            f.write("raise SystemExit(0)\n")
        os.environ["AI_LAUNCHERS_CLAUDE_BIN"] = stub
        for name in LAUNCHERS:
            rc, out, err = self.dry_run(name, "--", "-p", "hi there")
            self.assertEqual(rc, 0, err)
            self.assertNotIn(SENTINEL, out + err, name)
            self.assertIn("[dry-run] %s-wrap" % name, out)
            self.assertIn("argv: ", out)
            self.assertIn("stub.py -p 'hi there'" if os.name != "nt" else "hi there", out)
            self.assertIn("ANTHROPIC_API_KEY=", out)
            self.assertIn("- %s" % KEY_ENV[name], out)
            if name in DIRECT_LAUNCHERS:
                self.assertIn("ANTHROPIC_AUTH_TOKEN=<redacted>", out)
            else:
                self.assertIn("route table", out)
                self.assertIn("ANTHROPIC_AUTH_TOKEN=<per-launch random token>", out)
                self.assertIn("ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-via-background", out)
        self.assertEqual(fake_bins.read_calls(self.bin), [])

    def test_debug_prints_full_table_and_port(self):
        os.environ["XAI_API_KEY"] = fake_key("xai")
        os.environ["AI_LAUNCHERS_CLAUDE_BIN"] = os.path.join(self.tmp, "nowhere", "claude")
        rc, out, err = self.dry_run("grok", "--debug", "--port", "8899", "--model", "grok-4.3")
        self.assertEqual(rc, 0, err)
        self.assertIn('"picker_prefix": "claude-via-"', out)
        self.assertIn('"default": "xai,grok-4.3"', out)
        self.assertIn("ANTHROPIC_BASE_URL=http://127.0.0.1:8899", out)
        self.assertIn("<claude not found>", out)

    def test_no_transport(self):
        rc, out, err = self.dry_run("codex", "--auth", "api-key")
        self.assertEqual(rc, 2)
        self.assertIn("OPENAI_API_KEY", err)

    def test_unresolved_op_ref_is_reported(self):
        self.write_config({"providers": {"kimi": {"op_ref": "op://Personal/Kimi/credential"}}})
        rc, out, err = self.dry_run("kimi")
        self.assertEqual(rc, 0, err)
        self.assertIn("op://Personal/Kimi/credential (not resolved in dry-run)", out)


if __name__ == "__main__":
    unittest.main()
