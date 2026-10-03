"""Shared launcher utilities: per-user paths and JSON config/state I/O.

Every path is re-resolved on each call (never captured at import), so ``AIL_HOME``/``HOME`` changes
apply immediately (tests, portable installs):

  ail_home()        $AIL_HOME, else ~/.ai-launchers
  config_path()     <ail_home>/config.json      user overrides: providers.<transport id>, gateway.port
  credentials_path() <ail_home>/credentials.json API keys stored by ``keys set`` (0600)
  state_path()      <ail_home>/state.json       one-time notices
  cache_dir()       <ail_home>/cache            models.json discovery cache
  logs_dir()        <ail_home>/logs             rotating gateway logs (+ optional JSONL trace)

Readers never create anything (dry-run safe); only ``write_json``/``save_*`` touch the disk, atomically
with mode 0600 (``gateway.atomicio``).
"""

import json
import os
from pathlib import Path

__all__ = [
    "VERSION", "home_dir", "ail_home", "config_path", "credentials_path", "state_path", "cache_dir",
    "logs_dir", "read_json", "json_error", "write_json", "load_config", "save_config", "load_state", "save_state",
]

VERSION = "0.2.0"


def home_dir():
    """User home: ``USERPROFILE`` (then ``HOME``) on Windows, ``HOME`` elsewhere."""
    if os.name == "nt":
        value = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    else:
        value = os.environ.get("HOME")
    return Path(value) if value else Path.home()


def ail_home():
    value = os.environ.get("AIL_HOME")
    return Path(value) if value else home_dir() / ".ai-launchers"


def config_path():
    return ail_home() / "config.json"


def credentials_path():
    return ail_home() / "credentials.json"


def state_path():
    return ail_home() / "state.json"


def cache_dir():
    return ail_home() / "cache"


def logs_dir():
    return ail_home() / "logs"


def read_json(path, default=None):
    """Parsed JSON object at ``path``; ``default`` (``{}``) when missing, unreadable, invalid or not an object."""
    fallback = {} if default is None else default
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return fallback
    return data if isinstance(data, dict) else fallback


def json_error(path):
    """Why ``path`` is not a readable JSON object (``None`` when it is, or when it does not exist)."""
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        return "%s: %s" % (path, exc)
    return None if isinstance(data, dict) else "%s: expected a JSON object" % path


def write_json(path, obj, private=False):
    """Atomically write ``obj`` as pretty JSON with mode 0600 (parents created). ``private`` also applies
    an owner-only ACL on Windows (``icacls``) — used for credentials."""
    from .gateway import atomicio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomicio.atomic_write_json(str(path), obj, indent=2, mode=0o600)
    if private:
        atomicio.restrict_permissions(str(path))
    return path


def load_config():
    """``~/.ai-launchers/config.json`` (``{}`` if absent/invalid; never raises)."""
    return read_json(config_path())


def save_config(cfg):
    return write_json(config_path(), cfg)


def load_state():
    return read_json(state_path())


def save_state(state):
    return write_json(state_path(), state)
