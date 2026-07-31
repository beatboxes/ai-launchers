#!/usr/bin/env python3
"""Regression tests for the ai-launchers monorepo shared package + launchers.

Verifies the Phase-3 bug-fix patterns survive the extraction into shared/:
  C2   auth_bridge.resolve_model_alias inverts ail-grok-* (no grok-build collapse)
  H1   dry-run writes NOTHING (no CCR config, no settings.json)
  L5   write_ccr_config prunes old ail-backups (keep last 6)
  M3   find_free_port retries / returns a port
  C3   router port read from cfg.router.port
  key  set/remove/list round-trip; keys never printed in full

Usage: py -3 test_monorepo.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS: {name}")
    else:
        FAIL += 1; print(f"  FAIL: {name}  {detail}")


# --- import shared package ---
try:
    from shared import utils, key_manager, ccr_bridge, auth_bridge, base_launcher
    check("shared package imports", True)
except Exception as e:
    check("shared package imports", False, str(e))
    sys.exit(1)


print("== C2: resolve_model_alias inverts ail-grok-* (no grok-build collapse) ==")
check("ail-grok-4-5 -> grok-4.5", auth_bridge.resolve_model_alias("ail-grok-4-5") == "grok-4.5",
      auth_bridge.resolve_model_alias("ail-grok-4-5"))
check("ail-grok-4-3 -> grok-4.3", auth_bridge.resolve_model_alias("ail-grok-4-3") == "grok-4.3",
      auth_bridge.resolve_model_alias("ail-grok-4-3"))
check("ail-grok-4-20-0309-reasoning preserved",
      auth_bridge.resolve_model_alias("ail-grok-4-20-0309-reasoning") == "grok-4.20-0309-reasoning",
      auth_bridge.resolve_model_alias("ail-grok-4-20-0309-reasoning"))
check("ail-codex-gpt-4o-mini -> gpt-4o-mini",
      auth_bridge.resolve_model_alias("ail-codex-gpt-4o-mini") == "gpt-4o-mini")

print("== M3: find_free_port returns an int port ==")
p = utils.find_free_port(0)
check("find_free_port(0) returns int > 0", isinstance(p, int) and p > 0, f"got {p}")
p2 = utils.find_free_port(0)
check("two free ports distinct", p != p2, f"{p} == {p2}")

print("== L5: write_ccr_config prunes old backups (keep last 6) ==")
_ccr_home = Path(tempfile.mkdtemp(prefix="ccr_l5_"))
os.environ["CCR_HOME"] = str(_ccr_home)
# re-import ccr_bridge module constant is read at call? ccr_bridge.CCR_HOME captured
# at import; reload to honor the env.
import importlib
importlib.reload(ccr_bridge)
now = int(time.time())
for i in range(8):
    bk = _ccr_home / f"config.ail-backup.{now - i*100}.json"
    bk.write_text("{}", encoding="utf-8")
ccr_bridge.write_ccr_config({"Providers": [], "Router": {"default": "x,y"}}, {})
backups = sorted(_ccr_home.glob("config.ail-backup.*.json"))
check("L5 prunes old backups (<=6 remain)", len(backups) <= 6, f"{len(backups)} remain")
check("L5 keeps at least the newest", len(backups) >= 1, "none kept")
shutil.rmtree(_ccr_home, ignore_errors=True)

print("== C3: base_launcher reads router port from cfg.router.port ==")
# build a launcher with cfg router.port=4444 and check _router_port()
import importlib
importlib.reload(utils)
# write a AIL_HOME config with router.port
_fh = Path(tempfile.mkdtemp(prefix="ail_c3_"))
os.environ["AIL_HOME"] = str(_fh)
importlib.reload(utils)
importlib.reload(key_manager)
importlib.reload(ccr_bridge)
importlib.reload(base_launcher)
(_fh / "config.json").write_text(json.dumps({"router": {"port": 4444}}), encoding="utf-8")
manifest = json.loads((ROOT / "gemini" / "config.example.json").read_text(encoding="utf-8"))
l = base_launcher.Launcher(manifest)
check("C3 _router_port reads cfg.router.port=4444", l._router_port() == 4444, f"got {l._router_port()}")
shutil.rmtree(_fh, ignore_errors=True)

print("== H1: dry-run writes NOTHING (no CCR config, no settings.json) ==")
_fh2 = Path(tempfile.mkdtemp(prefix="ail_h1_"))
_ccr2 = Path(tempfile.mkdtemp(prefix="ccr_h1_"))
os.environ["AIL_HOME"] = str(_fh2)
os.environ["CCR_HOME"] = str(_ccr2)
os.environ["XAI_API_KEY"] = "xai-fake-h1-test-key"
importlib.reload(utils)
importlib.reload(key_manager)
importlib.reload(ccr_bridge)
importlib.reload(base_launcher)
proc = subprocess.run(
    [sys.executable, str(ROOT / "grok" / "grok-wrap.py"), "launch", "claude", "--dry-run"],
    capture_output=True, text=True, timeout=60,
    env={**os.environ, "AIL_HOME": str(_fh2), "CCR_HOME": str(_ccr2),
         "XAI_API_KEY": "xai-fake-h1-test-key"},
)
out = (proc.stdout or "") + (proc.stderr or "")
ccr_written = list(_ccr2.glob("config.json"))
backups_written = list(_ccr2.glob("config.ail-backup.*.json"))
settings_written = (_fh2 / ".claude" / "settings.json").exists()
check("H1 dry-run wrote NO CCR config.json", len(ccr_written) == 0, f"{ccr_written}")
check("H1 dry-run wrote NO ail-backup", len(backups_written) == 0, f"{backups_written}")
check("H1 dry-run wrote NO settings.json", not settings_written, "settings.json created")
check("H1 dry-run printed 'no files mutated'", "no files mutated" in out, out[-200:])
shutil.rmtree(_fh2, ignore_errors=True)
shutil.rmtree(_ccr2, ignore_errors=True)

print("== key_manager set/remove/list round-trip; never prints full key ==")
_fh3 = Path(tempfile.mkdtemp(prefix="ail_km_"))
os.environ["AIL_HOME"] = str(_fh3)
importlib.reload(utils)
importlib.reload(key_manager)
key_manager.set_key("testprov", "sk-abcdef1234567890")
k, src = key_manager.resolve_key("testprov", {"providers": {}})
check("resolve_key returns stored key", k == "sk-abcdef1234567890", f"got {k}")
check("resolve_key source = credentials.json", src == "credentials.json", f"got {src}")
entries = key_manager.list_keys(["testprov"])
e = entries[0]
check("list_keys shows source", e["source"] == "credentials.json", e)
check("list_keys prefix redacts (no full key)", e["prefix"] and "abcdef1234567890" not in (e["prefix"] or ""),
      e["prefix"])
key_manager.remove_key("testprov")
k2, _ = key_manager.resolve_key("testprov", {"providers": {}})
check("remove_key clears it", k2 is None, f"got {k2}")
shutil.rmtree(_fh3, ignore_errors=True)

print()
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
print("OVERALL:", "PASS" if FAIL == 0 else "FAIL")
sys.exit(1 if FAIL else 0)