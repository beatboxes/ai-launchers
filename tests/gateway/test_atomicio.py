"""atomicio: atomic replace, modes, JSON formatting, symlinks, Windows replace retry, icacls argv."""

import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from ._pkg import mod

atomicio = mod("atomicio")

POSIX = os.name != "nt"


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


class AtomicIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _leftovers(self):
        return [n for n in os.listdir(self.dir) if n.endswith(".tmp")]

    def test_write_bytes_new_and_replace(self):
        p = os.path.join(self.dir, "a.bin")
        atomicio.atomic_write_bytes(p, b"one")
        with open(p, "rb") as f:
            self.assertEqual(f.read(), b"one")
        if POSIX:
            self.assertEqual(_mode(p), 0o600)
        atomicio.atomic_write_bytes(p, bytearray(b"two"), mode=0o640)
        with open(p, "rb") as f:
            self.assertEqual(f.read(), b"two")
        if POSIX:
            self.assertEqual(_mode(p), 0o640)
        self.assertEqual(self._leftovers(), [])

    @unittest.skipUnless(POSIX, "POSIX permission bits")
    def test_mode_none_preserves_existing(self):
        p = os.path.join(self.dir, "settings.json")
        with open(p, "w") as f:
            f.write("{}")
        os.chmod(p, 0o644)
        atomicio.atomic_write_bytes(p, b"{\"a\":1}", mode=None)
        self.assertEqual(_mode(p), 0o644)
        fresh = os.path.join(self.dir, "fresh.json")
        atomicio.atomic_write_bytes(fresh, b"x", mode=None)
        self.assertEqual(_mode(fresh), 0o600)

    def test_write_json_pretty(self):
        p = os.path.join(self.dir, "auth.json")
        obj = {"b": 1, "a": {"ü": [1, 2]}}
        atomicio.atomic_write_json(p, obj)
        with open(p, "rb") as f:
            raw = f.read()
        self.assertEqual(raw.decode("utf-8"), json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
        self.assertTrue(raw.startswith(b'{\n  "b": 1'))
        self.assertEqual(json.loads(raw.decode("utf-8")), obj)
        if POSIX:
            self.assertEqual(_mode(p), 0o600)

    def test_failure_leaves_no_temp_and_keeps_old(self):
        p = os.path.join(self.dir, "keep.json")
        atomicio.atomic_write_bytes(p, b"old")
        with self.assertRaises(TypeError):
            atomicio.atomic_write_bytes(p, "not bytes")
        with self.assertRaises(TypeError):
            atomicio.atomic_write_json(p, {"x": object()})
        with mock.patch.object(atomicio.os, "replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                atomicio.atomic_write_bytes(p, b"new")
        with open(p, "rb") as f:
            self.assertEqual(f.read(), b"old")
        self.assertEqual(self._leftovers(), [])

    @unittest.skipUnless(POSIX and hasattr(os, "symlink"), "symlinks")
    def test_symlink_written_through(self):
        target = os.path.join(self.dir, "real.json")
        link = os.path.join(self.dir, "link.json")
        atomicio.atomic_write_bytes(target, b"1")
        os.symlink(target, link)
        atomicio.atomic_write_bytes(link, b"2")
        self.assertTrue(os.path.islink(link))
        with open(target, "rb") as f:
            self.assertEqual(f.read(), b"2")

    def test_windows_replace_retry(self):
        p = os.path.join(self.dir, "w.json")
        real_replace = os.replace
        calls = []

        def flaky(src, dst):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError(13, "in use")
            return real_replace(src, dst)

        with mock.patch.object(atomicio, "_IS_WINDOWS", True), \
                mock.patch.object(atomicio.os, "replace", side_effect=flaky), \
                mock.patch.object(atomicio.time, "sleep") as sleep:
            atomicio.atomic_write_bytes(p, b"ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleep.call_count, 2)
        with open(p, "rb") as f:
            self.assertEqual(f.read(), b"ok")
        # gives up after 1 + 10 attempts; POSIX never retries
        with mock.patch.object(atomicio, "_IS_WINDOWS", True), \
                mock.patch.object(atomicio.os, "replace", side_effect=PermissionError(13, "in use")) as rep, \
                mock.patch.object(atomicio.time, "sleep"):
            with self.assertRaises(PermissionError):
                atomicio.atomic_write_bytes(p, b"no")
        self.assertEqual(rep.call_count, 11)
        with mock.patch.object(atomicio, "_IS_WINDOWS", False), \
                mock.patch.object(atomicio.os, "replace", side_effect=PermissionError(13, "denied")) as rep:
            with self.assertRaises(PermissionError):
                atomicio.atomic_write_bytes(p, b"no")
        self.assertEqual(rep.call_count, 1)
        self.assertEqual(self._leftovers(), [])

    def test_restrict_permissions(self):
        p = os.path.join(self.dir, "secret")
        with open(p, "w") as f:
            f.write("x")
        if POSIX:
            os.chmod(p, 0o666)
            self.assertTrue(atomicio.restrict_permissions(p))
            self.assertEqual(_mode(p), 0o600)
            self.assertFalse(atomicio.restrict_permissions(os.path.join(self.dir, "missing")))

    def test_icacls_argv(self):
        argv = atomicio._icacls_argv("C:\\Users\\me\\.codex\\auth.json", {"USERNAME": "me", "USERDOMAIN": "CORP"})
        self.assertEqual(argv, ["icacls", "C:\\Users\\me\\.codex\\auth.json", "/inheritance:r", "/grant:r",
                                "CORP\\me:F"])
        self.assertEqual(atomicio._icacls_argv("f", {"USERNAME": "me"})[-1], "me:F")
        self.assertIsNone(atomicio._icacls_argv("f", {}))

    def test_restrict_permissions_windows_branch(self):
        with mock.patch.object(atomicio, "_IS_WINDOWS", True), \
                mock.patch.dict(os.environ, {"USERNAME": "me", "USERDOMAIN": "PC"}), \
                mock.patch.object(atomicio.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0)
            self.assertTrue(atomicio.restrict_permissions("x.json"))
        argv = run.call_args[0][0]
        self.assertEqual(argv, ["icacls", "x.json", "/inheritance:r", "/grant:r", "PC\\me:F"])
        self.assertNotIn("shell", run.call_args[1])
        self.assertEqual(run.call_args[1]["timeout"], 15)
        with mock.patch.object(atomicio, "_IS_WINDOWS", True), \
                mock.patch.object(atomicio.subprocess, "run", side_effect=OSError("no icacls")):
            self.assertFalse(atomicio.restrict_permissions("x.json"))


if __name__ == "__main__":
    unittest.main()
