"""filelock.ExclusiveFileLock: mutual exclusion across fds/threads/processes, timeout, 0600 lock file."""

import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from ._pkg import REPO_ROOT, PKG, mod

filelock = mod("filelock")

_HOLDER = """
import sys, time
sys.path.insert(0, %r)
import importlib
fl = importlib.import_module(%r + ".filelock")
lock = fl.ExclusiveFileLock(sys.argv[1], timeout=5)
lock.acquire()
print("locked", flush=True)
time.sleep(float(sys.argv[2]))
lock.release()
"""


class FileLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "auth.json.lock")

    def tearDown(self):
        self.tmp.cleanup()

    def test_acquire_release_and_mode(self):
        lock = filelock.ExclusiveFileLock(self.path, timeout=1)
        with lock as held:
            self.assertIs(held, lock)
            self.assertTrue(lock.locked)
            self.assertTrue(os.path.isfile(self.path))
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode) & 0o077, 0)
        self.assertFalse(lock.locked)
        self.assertTrue(os.path.isfile(self.path), "lock file is kept")
        lock.release()  # idempotent
        with lock:  # reusable
            pass
        self.assertIn("auth.json.lock", repr(lock))

    def test_timeout_when_held_by_another_descriptor(self):
        holder = filelock.ExclusiveFileLock(self.path).acquire()
        try:
            other = filelock.ExclusiveFileLock(self.path, timeout=0.3, poll=0.02)
            t0 = time.monotonic()
            with self.assertRaises(filelock.LockTimeout) as cm:
                other.acquire()
            elapsed = time.monotonic() - t0
            self.assertGreaterEqual(elapsed, 0.25)
            self.assertLess(elapsed, 2.0)
            self.assertIsInstance(cm.exception, TimeoutError)
            self.assertIn("auth.json.lock", str(cm.exception))
            self.assertFalse(other.locked)
            with self.assertRaises(filelock.LockTimeout):
                filelock.ExclusiveFileLock(self.path, timeout=0).acquire()  # single attempt
        finally:
            holder.release()
        with filelock.ExclusiveFileLock(self.path, timeout=0.5):
            pass

    def test_not_reentrant(self):
        lock = filelock.ExclusiveFileLock(self.path).acquire()
        try:
            with self.assertRaises(RuntimeError):
                lock.acquire()
        finally:
            lock.release()

    def test_threads_mutually_exclusive(self):
        inside = []
        overlaps = []
        counter = {"n": 0}

        def worker():
            for _ in range(5):
                with filelock.ExclusiveFileLock(self.path, timeout=10, poll=0.005):
                    inside.append(1)
                    if len(inside) > 1:
                        overlaps.append(1)
                    n = counter["n"]
                    time.sleep(0.002)
                    counter["n"] = n + 1
                    inside.pop()

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(overlaps, [])
        self.assertEqual(counter["n"], 30)

    def test_cross_process(self):
        code = _HOLDER % (REPO_ROOT, PKG)
        p = subprocess.Popen([sys.executable, "-c", code, self.path, "1.0"], stdout=subprocess.PIPE,
                             universal_newlines=True)
        try:
            self.assertEqual(p.stdout.readline().strip(), "locked")
            with self.assertRaises(filelock.LockTimeout):
                filelock.ExclusiveFileLock(self.path, timeout=0.2).acquire()
            t0 = time.monotonic()
            with filelock.ExclusiveFileLock(self.path, timeout=10):
                self.assertGreater(time.monotonic() - t0, 0.2)
        finally:
            p.wait(timeout=30)
            p.stdout.close()
        self.assertEqual(p.returncode, 0)

    def test_released_when_holder_dies(self):
        code = _HOLDER % (REPO_ROOT, PKG)
        p = subprocess.Popen([sys.executable, "-c", code, self.path, "30"], stdout=subprocess.PIPE,
                             universal_newlines=True)
        try:
            self.assertEqual(p.stdout.readline().strip(), "locked")
            p.kill()
            p.wait(timeout=30)
            with filelock.ExclusiveFileLock(self.path, timeout=5):
                pass
        finally:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=30)
            p.stdout.close()


if __name__ == "__main__":
    unittest.main()
