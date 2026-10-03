"""Cross-process exclusive file lock (DESIGN §4, §8.2 B).

``ExclusiveFileLock(path, timeout=10.0, poll=0.05)`` — context manager with ``acquire()`` /
``release()``; ``acquire`` raises ``LockTimeout`` (a ``TimeoutError``) when the lock is still held
by someone else after ``timeout`` seconds (``timeout <= 0`` = a single attempt).

* POSIX: ``fcntl.flock(fd, LOCK_EX | LOCK_NB)`` polled every ``poll`` seconds. ``flock`` locks
  belong to the open file description, so two ``ExclusiveFileLock`` objects exclude each other
  even inside one process (threads) — and the Codex / Grok CLIs' own ``flock`` users.
* Windows: ``msvcrt.locking(fd, LK_NBLCK, 1)`` on byte 0, polled the same way (overlaps the
  ``LockFileEx`` range used by Rust CLIs).

The lock file is created with mode 0600 and never deleted (deleting lock files races). Locks are
released by the OS when the holding process dies, so a stale lock file can never block.
"""

import errno
import os
import time

_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:  # pragma: no cover - exercised on Windows only
    import msvcrt
else:
    import fcntl

__all__ = ["ExclusiveFileLock", "LockTimeout"]

_BUSY = (errno.EACCES, errno.EAGAIN, getattr(errno, "EWOULDBLOCK", errno.EAGAIN), getattr(errno, "EDEADLK", -1),
         getattr(errno, "EDEADLOCK", -1))


class LockTimeout(TimeoutError):
    """The lock could not be acquired within the timeout."""

    def __init__(self, path, timeout):
        TimeoutError.__init__(self, "timed out after %.1fs waiting for lock %s" % (timeout, path))
        self.path = path
        self.timeout = timeout


def _try_lock(fd):
    """True if the lock was taken, False if someone else holds it; other errors propagate."""
    try:
        if _IS_WINDOWS:  # pragma: no cover
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if isinstance(exc, (BlockingIOError, PermissionError)) or exc.errno in _BUSY:
            return False
        raise


def _unlock(fd):
    if _IS_WINDOWS:  # pragma: no cover
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


class ExclusiveFileLock(object):
    """Exclusive advisory lock on ``path`` (created 0600 if missing). Not re-entrant."""

    def __init__(self, path, timeout=10.0, poll=0.05):
        self.path = os.fspath(path)
        self.timeout = float(timeout)
        self.poll = max(0.001, float(poll))
        self._fd = None

    @property
    def locked(self):
        return self._fd is not None

    def acquire(self, timeout=None):
        """Block until the lock is held (polling); ``LockTimeout`` after ``timeout`` seconds."""
        if self._fd is not None:
            raise RuntimeError("lock %s is already held by this object" % self.path)
        timeout = self.timeout if timeout is None else float(timeout)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
        fd = os.open(self.path, flags, 0o600)
        deadline = time.monotonic() + max(0.0, timeout)
        try:
            while not _try_lock(fd):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LockTimeout(self.path, timeout)
                time.sleep(min(self.poll, remaining))
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def release(self):
        """Release the lock (no-op if not held)."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        except OSError:
            pass  # closing the descriptor releases it anyway
        finally:
            os.close(fd)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def __repr__(self):
        return "ExclusiveFileLock(%r, locked=%r)" % (self.path, self.locked)
