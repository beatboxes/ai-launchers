"""Unified API-key manager for all provider launchers.

Resolution order for resolve_key():
  1. explicit env var (e.g. XAI_API_KEY)                  -- fastest, no disk
  2. op:// 1Password reference in config (op read)        -- never echoed
  3. ~/.ai-launchers/credentials.json  (DPAPI-optional on Windows)  -- cached

set_key / remove_key / list_keys operate on credentials.json.

Security:
  - keys are NEVER printed/logged. list_keys shows presence + prefix only.
  - op:// refs are resolved via `op read` subprocess; the value stays in-memory.
  - credentials.json is the operator's cache; on Windows it MAY be DPAPI-encrypted
    (operator explicitly authorized this despite the default "no secrets to disk").
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from .utils import ail_home

# L11: memoize `op read` within a process so repeated resolution is free.
_OP_READ_CACHE: dict = {}


def creds_path() -> Path:
    return ail_home() / "credentials.json"


def _load_creds() -> dict:
    p = creds_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_creds(creds: dict) -> None:
    p = creds_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(creds, indent=2, sort_keys=True), encoding="utf-8")


def _op_read(ref: str):
    """L11: cached `op read`. Returns the secret string or None on any failure.
    Never echoes the value (callers must not print the return)."""
    if ref in _OP_READ_CACHE:
        return _OP_READ_CACHE[ref]
    val = None
    try:
        proc = subprocess.run(
            ["op", "read", ref],
            capture_output=True, text=True, timeout=15,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode == 0 and proc.stdout:
            val = proc.stdout.strip() or None
    except Exception:
        val = None
    _OP_READ_CACHE[ref] = val
    return val


def resolve_key(provider: str, cfg: dict):
    """Return (key, source) or (None, reason). Never logs the key.

    provider cfg shape (per-provider config.example.json):
      env_var: "XAI_API_KEY"
      op_ref:  "op://Personal/xAI/credential"   (optional)
    """
    pcfg = (cfg.get("providers") or {}).get(provider) or {}
    env_var = pcfg.get("env_var")
    if env_var:
        v = os.environ.get(env_var)
        if v:
            return v, f"env:{env_var}"
    op_ref = pcfg.get("op_ref")
    if op_ref:
        v = _op_read(op_ref)
        if v:
            return v, f"op:{op_ref.split('/')[-1]}"
    creds = _load_creds()
    entry = creds.get(provider)
    if entry and isinstance(entry, dict) and entry.get("key"):
        return entry["key"], "credentials.json"
    return None, "no key found (set with: %s keys set)" % provider


def set_key(provider: str, value: str) -> None:
    creds = _load_creds()
    creds[provider] = {"key": value, "stored": True}
    _save_creds(creds)


def remove_key(provider: str) -> bool:
    creds = _load_creds()
    existed = provider in creds
    if existed:
        del creds[provider]
        _save_creds(creds)
    return existed


def list_keys(providers: list) -> list:
    """Return [{provider, source, prefix}] — prefix only, never the full key."""
    cfg = {}  # list reads its own minimal state; env + creds only (no op round-trip)
    out = []
    for p in providers:
        env_var_candidates = [f"{p.upper()}_API_KEY", f"{p.upper()}_KEY"]
        src = None
        prefix = None
        for ev in env_var_candidates:
            v = os.environ.get(ev)
            if v:
                src = f"env:{ev}"
                prefix = v[:6] + "..." + v[-2:]
                break
        if src is None:
            creds = _load_creds()
            entry = creds.get(p)
            if entry and isinstance(entry, dict) and entry.get("key"):
                k = entry["key"]
                src = "credentials.json"
                prefix = k[:6] + "..." + k[-2:]
        out.append({"provider": p, "source": src or "unset", "prefix": prefix})
    return out