"""claude-code-router (CCR) config compiler + daemon lifecycle.

Bug-fix patterns carried by this module:
  C3   router port from cfg.router.port (not hardcoded :3456)
  M2   track Popen + atexit stop (no orphan CCR)
  M3   free-port retry (see utils.find_free_port)
  L5   prune old ail-backup configs (keep last 6)
  M10  skip provider if upstream down (no dead fallback)
  M13  Router default honors cfg router.roles.default (not hardcoded)
  H1   dry-run never mutates settings.json/.claude.json
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .utils import ail_home, find_free_port

CCR_BIN = shutil.which("ccr") or shutil.which("ccr.cmd") or shutil.which("ccr.exe")
CCR_HOME = Path(os.environ.get("CCR_HOME", Path.home() / ".claude-code-router"))
CCR_CONFIG = CCR_HOME / "config.json"
KEEP_BACKUPS = 6  # L5

_ccr_proc = None
_scrub_proc = None


def _prune_backups(cfg_path: Path) -> None:
    """L5: keep only the newest KEEP_BACKUPS ail-backup configs."""
    backups = sorted(
        cfg_path.parent.glob("config.ail-backup.*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in backups[KEEP_BACKUPS:]:
        try:
            old.unlink()
        except Exception:
            pass


def write_ccr_config(config_obj: dict, cfg: dict) -> Path:
    """Write CCR config.json with a timestamped backup first (L5 prune after)."""
    CCR_HOME.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    if CCR_CONFIG.exists():
        backup = CCR_CONFIG.parent / f"config.ail-backup.{ts}.json"
        shutil.copy2(CCR_CONFIG, backup)
    CCR_CONFIG.write_text(json.dumps(config_obj, indent=2), encoding="utf-8")
    _prune_backups(CCR_CONFIG)  # L5: prune AFTER write, unconditional
    return CCR_CONFIG


def compile_ccr_config(provider: str, base_url: str, key, model: str,
                       models_catalog: list, roles: dict = None) -> dict:
    """Build a CCR config with a single OpenAI-compat provider + Router roles.

    `key` may be a real API key OR None. If None the caller is expected to have
    started a localhost auth bridge and passed its FULL chat-completions URL as
    base_url; a dummy key is used (CCR requires the field present).

    NOTE: CCR (musistudio/claude-code-router) requires `api_base_url` to be the
    FULL endpoint including `/chat/completions` (confirmed via issue #94), NOT a
    bare base. The manifest base_url values already include this path.
    """
    api_key = key if key else "local-bridge-dummy"
    providers = [{
        "name": provider,
        "api_base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "models": list(models_catalog),
    }]
    default_model = f"{provider},{model}"
    roles = roles or {}
    router = {
        "default": roles.get("default", default_model),
        "background": roles.get("background", default_model),
        "think": roles.get("think", default_model),
        "longContext": roles.get("longContext", default_model),
        "webSearch": roles.get("webSearch", default_model),
    }
    return {"Providers": providers, "Router": router}


def _ccr_stop_singleton():
    """BUG A: canonical-stop any running CCR singleton daemon. `ccr start`
    (musistudio/claude-code-router) detaches a daemon the Popen wrapper does
    NOT track — the wrapper exits 0 while the daemon keeps the port. Killing
    only the wrapper Popen orphans the daemon on its port, and a later
    `ccr start` then sees the singleton "already running" and exits 0 fast
    (which start_ccr misreads as "ccr exited early"). `ccr stop` reads the
    daemon pid file and stops it. Idempotent, never raises."""
    if CCR_BIN is None:
        return
    try:
        subprocess.run(
            [CCR_BIN, "stop"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, timeout=10,
            creationflags=0x08000000 if sys.platform == "win32" else 0,  # CREATE_NO_WINDOW
        )
    except Exception:
        pass


def start_ccr(port: int = 0):
    """M2: start CCR daemon, register atexit stop. Returns the resolved port.
    Raises RuntimeError if ccr binary missing."""
    global _ccr_proc
    if CCR_BIN is None:
        raise RuntimeError("ccr (claude-code-router) not found on PATH; install: npm i -g @musistudio/claude-code-router")
    # BUG A: clear any stale singleton daemon first so our `ccr start` binds the
    # requested port cleanly (else the singleton "already running" fast-exit is
    # misread as "ccr exited early").
    _ccr_stop_singleton()
    actual_port = find_free_port(port or 3456)  # M3
    _ccr_proc = subprocess.Popen(
        [CCR_BIN, "start", "--port", str(actual_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        creationflags=0x08000000 if sys.platform == "win32" else 0,  # CREATE_NO_WINDOW
    )
    atexit.register(stop_ccr)
    # bounded readiness poll (foreground, not -f/watch)
    for _ in range(20):
        if _ccr_proc.poll() is not None:
            err = _ccr_proc.stderr.read().decode("utf-8", "replace") if _ccr_proc.stderr else ""
            raise RuntimeError(f"ccr exited early: {err[:300]}")
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", actual_port))
            s.close()
            return actual_port
        except OSError:
            s.close()
            time.sleep(0.3)
    raise RuntimeError(f"ccr did not become ready on port {actual_port}")


def stop_ccr():
    """M2: stop CCR daemon if we started it. Idempotent.

    BUG A: also canonical-stop the detached singleton daemon via `ccr stop` —
    on Windows `ccr start` spawns a singleton daemon the Popen wrapper does not
    track, so terminating the wrapper alone orphans the daemon on its port."""
    global _ccr_proc
    if _ccr_proc and _ccr_proc.poll() is None:
        try:
            _ccr_proc.terminate()
            _ccr_proc.wait(timeout=5)
        except Exception:
            try:
                _ccr_proc.kill()
            except Exception:
                pass
    _ccr_proc = None
    _ccr_stop_singleton()


def restore_clean_config() -> None:
    """Restore the most recent ail-backup so we leave CCR config clean (H1/H5
    leave clean state). Best-effort, never raises."""
    try:
        backups = sorted(
            CCR_CONFIG.parent.glob("config.ail-backup.*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if backups and CCR_CONFIG.exists():
            shutil.copy2(backups[0], CCR_CONFIG)
    except Exception:
        pass