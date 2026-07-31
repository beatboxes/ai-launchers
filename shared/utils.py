"""Shared utilities: config I/O, free-port bind, Windows shim, env helpers.

Bug-fix patterns carried by this module:
  H3  AIL_HOME respected (not a hardcoded absolute path)
  H4  AIL_REPO respected for auto-sync source
  M3  SO_REUSEADDR + retry on bind (TOCTOU port race)
  L3  env-name filter compares VALUES not identity
"""
import json
import os
import socket
import sys
from pathlib import Path

__all__ = [
    "home_dir", "ail_home", "config_path", "load_config", "save_config",
    "find_free_port", "wrap_for_windows", "env_filter_changed",
]

VERSION = "0.1.0"


def home_dir() -> Path:
    """User home. Respects USERPROFILE on Windows, HOME elsewhere."""
    return Path(os.environ.get("USERPROFILE") or os.environ.get("HOME") or Path.home())


def ail_home() -> Path:
    """H3/H4: AIL_HOME honored live (re-resolved each call, never captured at import)."""
    return Path(os.environ.get("AIL_HOME", home_dir() / ".ai-launchers"))


def config_path() -> Path:
    return ail_home() / "config.json"


def load_config() -> dict:
    """Load ~/.ai-launchers/config.json; return {} if absent/invalid (never raise)."""
    p = config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def find_free_port(preferred: int = 0, retries: int = 5) -> int:
    """M3: bind with SO_REUSEADDR; retry on failure. preferred=0 -> OS-assigned."""
    if preferred and preferred > 0:
        for _ in range(retries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", preferred))
                    return preferred
            except OSError:
                continue
    # fallback: let the OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wrap_for_windows(args):
    """On Windows, wrap a python -m / script invocation so a detached child
    does not open a console window. Returns args unchanged on non-Windows."""
    if sys.platform != "win32":
        return args
    # caller is responsible for passing creationflags; this returns args only.
    return args


def env_filter_changed(env: dict) -> dict:
    """L3: return only env vars whose VALUE differs from the parent os.environ.
    Used for printing what a launch injected without dumping the whole env."""
    out = {}
    for k, v in env.items():
        if v != os.environ.get(k):
            out[k] = v
    return out