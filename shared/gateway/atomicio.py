"""Atomic file writes and owner-only permissions (DESIGN §4, §8.2 B).

``atomic_write_bytes(path, data, mode=0o600)``
    Writes ``data`` to a temp file in the SAME directory, ``fsync``s it, applies ``mode`` and
    ``os.replace``s it over ``path`` (readers see the old or the new file, never a torn one).
    ``mode=None`` keeps the existing file's permission bits (0600 for a new file). A symlinked
    ``path`` is written through (the link is kept). On Windows ``os.replace`` is retried 10 x 50 ms
    on ``PermissionError`` (another process — antivirus, the Codex CLI — has the target open).
``atomic_write_json(path, obj, indent=2, mode=0o600)``
    ``json.dumps(obj, indent=indent, ensure_ascii=False)`` + newline, UTF-8, via the above.
``restrict_permissions(path) -> bool``
    Best effort owner-only access: POSIX ``chmod 600``; Windows ``icacls <path> /inheritance:r
    /grant:r <DOMAIN\\user>:F`` (list args, no shell/PowerShell). Returns True on success.
"""

import json
import os
import stat
import subprocess
import tempfile
import time

__all__ = ["atomic_write_bytes", "atomic_write_json", "restrict_permissions"]

_IS_WINDOWS = os.name == "nt"
_REPLACE_RETRIES = 10
_REPLACE_DELAY = 0.05
_DEFAULT_MODE = 0o600


def _replace(src, dst):
    for attempt in range(_REPLACE_RETRIES + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if not _IS_WINDOWS or attempt == _REPLACE_RETRIES:
                raise
            time.sleep(_REPLACE_DELAY)


def _fsync_dir(directory):
    if _IS_WINDOWS:
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path, data, mode=_DEFAULT_MODE):
    """Atomically replace ``path`` with ``data`` (bytes). See module docstring."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("atomic_write_bytes expects bytes, got %s" % type(data).__name__)
    path = os.fspath(path)
    if os.path.islink(path):
        path = os.path.realpath(path)
    if mode is None:
        try:
            mode = stat.S_IMODE(os.stat(path).st_mode)
        except OSError:
            mode = _DEFAULT_MODE
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        _replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(directory)


def atomic_write_json(path, obj, indent=2, mode=_DEFAULT_MODE):
    """Atomically write ``obj`` as pretty JSON (UTF-8, trailing newline)."""
    text = json.dumps(obj, indent=indent, ensure_ascii=False) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def _icacls_argv(path, environ=None):
    """``icacls`` command granting full control to the current user only, or None if unknown."""
    env = os.environ if environ is None else environ
    user = env.get("USERNAME")
    if not user:
        return None
    domain = env.get("USERDOMAIN")
    principal = "%s\\%s" % (domain, user) if domain else user
    return ["icacls", os.fspath(path), "/inheritance:r", "/grant:r", "%s:F" % principal]


def restrict_permissions(path):
    """Make ``path`` readable/writable by its owner only (best effort, never raises)."""
    path = os.fspath(path)
    if not _IS_WINDOWS:
        try:
            os.chmod(path, 0o600)
            return True
        except OSError:
            return False
    argv = _icacls_argv(path)
    if argv is None:
        return False
    try:
        r = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
