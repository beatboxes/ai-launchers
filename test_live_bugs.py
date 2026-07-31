#!/usr/bin/env python3
"""Regression tests for two genuine bugs surfaced by the grok-wrap / codex-wrap
bounded live test (2026-07-23). These bugs live in the shared/ package, so the
blast radius spans every launcher that imports shared.base_launcher.

  BUG B  (shared/auth_bridge.py resolve_model_alias):
    base_launcher.py:193 builds effective_model = ail-<cli_target>-<model_dashed>.
    For grok, cli_target=="grok" and the default model is "grok-4.5", so the alias
    is "ail-grok-grok-4-5" (the "grok" prefix is doubled). The bridge's alias map
    keys expected the single-grok form "ail-grok-4-5"; the doubled form fell through
    to the generic path which produced "grok-grok.4-5" — an INVALID id the grok
    CLI rejects ("unknown model id", rc=1). The fix normalizes the doubled
    "ail-grok-grok-" prefix back to "ail-grok-" so the existing alias map + generic
    logic resolve it to the real id "grok-4.5". The codex path is untouched
    (ail-codex-gpt-4o -> gpt-4o still holds).

  BUG A  (shared/ccr_bridge.py CCR singleton lifecycle):
    `ccr start` (musistudio/claude-code-router) detaches a daemon the Popen
    wrapper does NOT track — the wrapper exits 0 immediately while the daemon
    keeps the port. stop_ccr only terminated the wrapper Popen, orphaning the
    daemon on its port (reproduced: after a clean probe exit, `ccr status`
    showed a NEW orphan daemon PID on 3456). A subsequent `ccr start` then saw
    the singleton "already running", exited 0 fast with empty stderr, and
    start_ccr misread that as "ccr exited early". The fix: stop_ccr canonical-
    stops the detached daemon via `ccr stop`, and start_ccr pre-stops any stale
    singleton before starting so a fresh daemon binds cleanly.

Usage: py -3 test_live_bugs.py
"""
import sys
from pathlib import Path
from unittest import mock

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


from shared import auth_bridge, ccr_bridge

print("== BUG B: resolve_model_alias handles doubled ail-grok-grok-* (base_launcher contract) ==")
# base_launcher builds ail-<cli_target>-<model_dashed>; for grok cli_target=="grok"
# doubles the prefix: ail-grok-grok-4-5. Must map to grok-4.5, NOT grok-grok.4-5.
check("ail-grok-grok-4-5 -> grok-4.5",
      auth_bridge.resolve_model_alias("ail-grok-grok-4-5") == "grok-4.5",
      auth_bridge.resolve_model_alias("ail-grok-grok-4-5"))
check("ail-grok-grok-4-3 -> grok-4.3",
      auth_bridge.resolve_model_alias("ail-grok-grok-4-3") == "grok-4.3",
      auth_bridge.resolve_model_alias("ail-grok-grok-4-3"))
check("ail-grok-grok-4-20-0309-reasoning -> grok-4.20-0309-reasoning",
      auth_bridge.resolve_model_alias("ail-grok-grok-4-20-0309-reasoning") == "grok-4.20-0309-reasoning",
      auth_bridge.resolve_model_alias("ail-grok-grok-4-20-0309-reasoning"))
check("ail-grok-grok-build-0-1 -> grok-build-0.1 (no collapse)",
      auth_bridge.resolve_model_alias("ail-grok-grok-build-0-1") == "grok-build-0.1",
      auth_bridge.resolve_model_alias("ail-grok-grok-build-0-1"))
# backward compat: legacy single-grok aliases still resolve
check("ail-grok-4-5 still resolves (legacy single-grok)",
      auth_bridge.resolve_model_alias("ail-grok-4-5") == "grok-4.5",
      auth_bridge.resolve_model_alias("ail-grok-4-5"))
# codex path unaffected (cli_target "codex" does NOT collide with model prefix)
check("ail-codex-gpt-4o -> gpt-4o (codex unaffected)",
      auth_bridge.resolve_model_alias("ail-codex-gpt-4o") == "gpt-4o",
      auth_bridge.resolve_model_alias("ail-codex-gpt-4o"))

print("== BUG A: stop_ccr + start_ccr canonical-stop the detached CCR singleton daemon ==")
# Force a fake ccr binary so the test does not touch any real daemon.
ccr_bridge.CCR_BIN = "/fake/ccr"

class FakeProc:
    def __init__(self):
        self.stderr = None
    def poll(self):
        return 0   # wrapper already exited (daemon detached) — the orphan case
    def terminate(self):
        pass
    def wait(self, timeout=None):
        return 0
    def kill(self):
        pass

calls = []
def fake_run(args, *a, **k):
    calls.append(("run", list(args)))
    class R:
        returncode = 0
        stdout = b""
        stderr = b""
    return R()
def fake_popen(args, *a, **k):
    calls.append(("popen", list(args)))
    return FakeProc()

with mock.patch.object(ccr_bridge.subprocess, "run", side_effect=fake_run), \
     mock.patch.object(ccr_bridge.subprocess, "Popen", side_effect=fake_popen):
    # stop_ccr with no tracked wrapper proc must STILL canonical-stop the singleton.
    ccr_bridge._ccr_proc = None
    calls.clear()
    ccr_bridge.stop_ccr()
    stop_hits = [c for c in calls if c[0] == "run" and c[1][1] == "stop"]
    check("stop_ccr invokes `ccr stop` for the detached daemon",
          len(stop_hits) >= 1, f"calls={calls}")

    # start_ccr must pre-stop a stale singleton BEFORE `ccr start`.
    ccr_bridge._ccr_proc = None
    calls.clear()
    try:
        ccr_bridge.start_ccr(0)
    except Exception:
        pass  # readiness poll raises on fake; we only inspect call ordering
    # c[1] = [bin, subcommand, ...]; subcommand is index 1.
    stop_idx = next((i for i, c in enumerate(calls)
                     if c[0] == "run" and c[1][1] == "stop"), None)
    start_idx = next((i for i, c in enumerate(calls)
                      if c[0] == "popen" and c[1][1] == "start"), None)
    check("start_ccr pre-stops stale singleton before `ccr start`",
          stop_idx is not None and start_idx is not None and stop_idx < start_idx,
          f"stop_idx={stop_idx} start_idx={start_idx} calls={calls}")

print("== BUG C: codex bridge path omits -m (uses ChatGPT-account default) ==")
# The bridge runs only when key is None (base_launcher enters it iff key is None),
# so codex is always on ChatGPT-account auth. Every manifest model id (gpt-4o,
# gpt-5-codex, codex-mini, ...) is an API-key-only model that ChatGPT-account
# codex rejects with rc=1 ("'X' model is not supported when using Codex with a
# ChatGPT account"). The fix omits -m so codex uses its own default.
auth_bridge.CODEX_EXE = "/fake/codex"
codex_calls = []
def fake_codex_run(args, *a, **k):
    codex_calls.append(list(args))
    class R:
        returncode = 0; stdout = "6"; stderr = ""
    return R()
with mock.patch.object(auth_bridge.subprocess, "run", side_effect=fake_codex_run):
    auth_bridge.run_cli("2x3?", "gpt-4o", "codex")
# Fixed args = [exe, "exec", "--skip-git-repo-check", "-"]; -m MUST be absent.
m_idx = next((i for i, a in enumerate(codex_calls[0]) if a == "-m"), None)
check("codex run_cli omits -m (ChatGPT-account default)",
      m_idx is None, f"args={codex_calls[0]}")
check("codex args == [exe, exec, --skip-git-repo-check, -]",
      codex_calls[0] == ["/fake/codex", "exec", "--skip-git-repo-check", "-"],
      f"args={codex_calls[0]}")

print()
print("== BUG D: bridge strips claude <system-reminder> context from forwarded prompt ==")
# claude bundles a <system-reminder>...</system-reminder> CLAUDE.md context
# injection as a LEADING part of the user message content LIST; the bridge
# joins the list parts and must strip that block before forwarding to the
# native CLI, else the operator's private manual is leaked AND the question is
# drowned (model echoes an operand / returns 0). Reproduce the do_POST
# extraction (join text parts) + the BUG D strip, and assert only the question
# survives.
_RE = auth_bridge._SYSTEM_REMINDER_RE
_claude_ctx = ("<system-reminder>\nAs you answer the user's questions, you can use the "
               "following context:\n# claudeMd\n...operator CLAUDE.md with 1Password paths "
               "and host IPs...\n</system-reminder>")
_question = "What is 906 multiplied by 189? Respond with ONLY the numeric answer, no words, no explanation."
# content-list user message as claude/CCR sends it (context part first, question part second)
_content_list = [
    {"type": "text", "text": _claude_ctx},
    {"type": "text", "text": _question},
]
_parts = [p.get("text", "") for p in _content_list if isinstance(p, dict) and p.get("type") == "text"]
_joined = "\n".join(_parts)  # mirror do_POST list handling (pre-fix)
_pre_fix = _joined  # what the bridge forwarded BEFORE the fix
_post_fix = _RE.sub("", _joined)  # what it forwards AFTER the fix
# RED proof: pre-fix prompt carries the 63KB-ish system-reminder (question drowned)
check("pre-fix prompt CONTAINS <system-reminder> (RED)",
      "<system-reminder>" in _pre_fix and "claudeMd" in _pre_fix,
      f"len={len(_pre_fix)} head={_pre_fix[:60]!r}")
# GREEN proof: post-fix prompt is ONLY the question, no reminder, no claudeMd leak
check("post-fix prompt strips <system-reminder> (GREEN)",
      "<system-reminder>" not in _post_fix and "claudeMd" not in _post_fix,
      f"len={len(_post_fix)} val={_post_fix!r}")
check("post-fix prompt == the actual user question (nothing else)",
      _post_fix.strip() == _question,
      f"val={_post_fix!r}")
# Multiple blocks (defensive) — strip ALL, keep interleaved user text.
_two = ("x" + _claude_ctx + "\n" + _question + "\n" + _claude_ctx + "y")
_two_fixed = _RE.sub("", _two)
# Each block + its trailing whitespace is removed; the \n BEFORE the second block
# is user-context whitespace the regex keeps. Expected: "x<question>\ny".
check("post-fix strips MULTIPLE system-reminder blocks, keeps surrounding text",
      _two_fixed == "x" + _question + "\ny" and "<system-reminder>" not in _two_fixed,
      f"val={_two_fixed!r}")

print()
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
print("OVERALL:", "PASS" if FAIL == 0 else "FAIL")
sys.exit(1 if FAIL else 0)