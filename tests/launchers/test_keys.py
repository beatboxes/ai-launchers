"""`keys set|remove|list`: credentials.json round trip, 0600, manifest env-var detection (the v0.1
`GROK-WRAP_API_KEY` bug), hidden prompt / stdin / inline-argument paths, redaction."""

import io
import json
import os
import stat
import unittest
from unittest import mock

from ._util import SENTINEL, LauncherTestCase, fake_key, key_manager


class KeysTests(LauncherTestCase):
    def creds(self):
        with open(os.path.join(self.ail, "credentials.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def test_set_list_remove_round_trip(self):
        key = fake_key("xai")
        rc, out, err = self.run_cli("grok", "keys", "set", key)
        self.assertEqual(rc, 0, err)
        self.assertIn("shell history", err)                  # inline argument warning
        self.assertNotIn(key, out + err)
        self.assertEqual(self.creds(), {"grok-wrap": {"key": key}})
        if os.name != "nt":
            mode = stat.S_IMODE(os.stat(os.path.join(self.ail, "credentials.json")).st_mode)
            self.assertEqual(mode, 0o600)
        rc, out, _ = self.run_cli("grok", "keys", "list")
        self.assertEqual(rc, 0)
        self.assertIn("active: credentials.json", out)
        self.assertNotIn(key, out)
        self.assertNotIn(SENTINEL, out)
        rc, out, _ = self.run_cli("grok", "keys", "remove")
        self.assertEqual(rc, 0)
        self.assertIn("removed", out)
        self.assertEqual(self.creds(), {})
        rc, out, _ = self.run_cli("grok", "keys", "remove")
        self.assertIn("no stored key", out)

    def test_other_launchers_entries_preserved(self):
        self.write_json(os.path.join(self.ail, "credentials.json"),
                        {"codex-wrap": {"key": "sk-other", "stored": True}})
        self.run_cli("grok", "keys", "set", fake_key("xai"))
        creds = self.creds()
        self.assertEqual(creds["codex-wrap"], {"key": "sk-other", "stored": True})
        self.assertIn("grok-wrap", creds)

    def test_existing_world_readable_file_tightened(self):
        if os.name == "nt":
            self.skipTest("POSIX permission bits")
        path = self.write_json(os.path.join(self.ail, "credentials.json"), {})
        os.chmod(path, 0o644)
        self.run_cli("kimi", "keys", "set", fake_key("kimi"))
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_list_uses_manifest_env_vars(self):
        os.environ["XAI_API_KEY"] = fake_key("xai")
        os.environ["GROK-WRAP_API_KEY"] = "wrong-variable"
        rc, out, _ = self.run_cli("grok", "keys", "list")
        self.assertEqual(rc, 0)
        self.assertIn("active: env:XAI_API_KEY", out)
        self.assertNotIn("GROK-WRAP", out)
        self.assertNotIn(SENTINEL, out)

    def test_list_second_env_var(self):
        os.environ["GOOGLE_API_KEY"] = fake_key("google")
        rc, out, _ = self.run_cli("gemini", "keys", "list")
        self.assertIn("active: env:GOOGLE_API_KEY", out)
        self.assertIn("env GEMINI_API_KEY", out)
        self.assertIn("gemini-vertex", out)

    def test_list_reports_op_ref_without_resolving(self):
        self.write_config({"providers": {"deepseek": {"op_ref": "op://Personal/DeepSeek/credential"}}})
        with mock.patch("subprocess.run", side_effect=AssertionError("keys list must not run op")):
            rc, out, _ = self.run_cli("deepseek", "keys", "list")
        self.assertEqual(rc, 0)
        self.assertIn("op://Personal/DeepSeek/credential (resolved at launch)", out)

    def test_set_prompts_hidden_on_tty(self):
        key = fake_key("tty")
        tty = io.StringIO()
        tty.isatty = lambda: True
        with mock.patch("getpass.getpass", return_value=key) as gp:
            rc, out, err = self.run_cli("codex", "keys", "set", stdin=tty)
        self.assertEqual(rc, 0, err)
        gp.assert_called_once()
        self.assertNotIn("shell history", err)
        self.assertEqual(self.creds()["codex-wrap"]["key"], key)

    def test_set_from_pipe(self):
        key = fake_key("pipe")
        rc, _, err = self.run_cli("deepseek", "keys", "set", stdin=io.StringIO(key + "\n"))
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.creds()["deepseek-wrap"]["key"], key)

    def test_set_rejects_bad_keys(self):
        for bad in ("", "has space"):
            rc, _, err = self.run_cli("grok", "keys", "set", stdin=io.StringIO(bad))
            self.assertEqual(rc, 2, bad)
            self.assertIn("key not stored", err)
        rc, _, err = self.run_cli("grok", "keys", "set", "line\nbreak")
        self.assertEqual(rc, 2)
        self.assertIn("whitespace or control characters", err)
        self.assertFalse(os.path.exists(os.path.join(self.ail, "credentials.json")))

    def test_legacy_entry_format_resolves(self):
        key = fake_key("legacy")
        self.write_json(os.path.join(self.ail, "credentials.json"), {"kimi-wrap": {"key": key, "stored": True}})
        launcher = self.launcher("kimi")
        selected, store = launcher.resolve(dry_run=False)
        self.assertEqual([t.id for t in selected], ["kimi"])
        self.assertEqual(store.get("kimi"), key)
        self.assertEqual(key_manager.display_source(selected[0].source), "credentials.json")

    def test_redact(self):
        key = fake_key("r")
        hint = key_manager.redact(key)
        self.assertNotIn(SENTINEL, hint)
        self.assertTrue(hint.startswith(key[:4]))
        self.assertNotIn("short", key_manager.redact("short-key"))
        self.assertEqual(key_manager.redact(None), "(none)")


if __name__ == "__main__":
    unittest.main()
