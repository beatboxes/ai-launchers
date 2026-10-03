"""End-to-end harness driving the REAL installed Claude Code (DESIGN §7). Test-only.

``claude_available() -> bool``
``run_e2e(launcher_argv, env, workdir, timeout=300, prompt=DEFAULT_PROMPT, home=None, extra_args=(),
keep_env=()) -> E2EResult``

* builds a throw-away HOME (``home`` or a new temp dir): ``HOME``/``USERPROFILE``,
  ``CLAUDE_CONFIG_DIR=$HOME/.claude``, ``AIL_HOME``/``FRY_HOME``/``CODEX_HOME``/``GROK_HOME``/
  ``CLOUDSDK_CONFIG`` under it; pre-seeds ``$HOME/.claude/settings.json`` = ``{"model":"sonnet"}`` and a
  user ``additionalModelOptionsCache`` entry in both ``$HOME/.claude.json`` (what launchers might
  touch) and ``$CLAUDE_CONFIG_DIR/.claude.json`` (what Claude Code reads when CLAUDE_CONFIG_DIR is set);
* makes ``env`` hermetic: variables of an enclosing Claude Code session (``CLAUDECODE``,
  ``CLAUDE_*`` except launcher-owned ones and ``keep_env``) are dropped; sets
  ``DISABLE_AUTOUPDATER=1`` and ``CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1``;
* runs ``<launcher_argv> -- -p <prompt> --output-format stream-json --verbose --permission-mode dontAsk
  --allowedTools Bash,Read,Write,Edit --strict-mcp-config`` in ``workdir`` (``launcher_argv=None`` runs
  ``claude`` itself with ``env``, e.g. one built by ``launchkit.build_child_env``), stdin closed,
  killing the whole process tree after ``timeout`` seconds;
* returns rc, output, parsed stream-json events, the final result text and the config-file checks.
"""

import json
import os
import signal
import subprocess
import tempfile

from ..launchkit import ClaudeNotFound, claude_version, find_claude

__all__ = ["E2EResult", "run_e2e", "claude_available", "hermetic_env", "DEFAULT_PROMPT", "CLAUDE_FLAGS",
           "SEEDED_SETTINGS", "USER_MODEL_OPTION", "LAUNCHER_OWNED_ENV"]

DEFAULT_PROMPT = "Run the shell command and report the result."
CLAUDE_FLAGS = ("--output-format", "stream-json", "--verbose", "--permission-mode", "dontAsk",
                "--allowedTools", "Bash,Read,Write,Edit", "--strict-mcp-config")
SEEDED_SETTINGS = {"model": "sonnet"}
USER_MODEL_OPTION = {"value": "user-custom-model", "label": "User custom model",
                     "description": "seeded by the gateway e2e harness"}

#: CLAUDE_* variables a launcher legitimately sets (kept); every other CLAUDE* variable is assumed to
#: come from an enclosing Claude Code session and is dropped.
LAUNCHER_OWNED_ENV = frozenset((
    "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_CODE_SUBAGENT_MODEL", "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS",
    "CLAUDE_CODE_EFFORT_LEVEL", "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK",
    "CLAUDE_ENABLE_BYTE_WATCHDOG", "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS",
))
_HOST_ONLY_ENV = frozenset(("SESSION_INGRESS_URL", "CCR_PRELOAD_CLAUDE"))
_FRY_MARKERS = ("(router)", "fry-", "claude-via-")


class E2EResult(object):
    """Outcome of one ``run_e2e`` call (plain attributes)."""

    __slots__ = ("rc", "stdout", "stderr", "events", "result_text", "result_event", "workdir", "home", "argv",
                 "timed_out", "settings_unchanged", "claude_json_user_entry_intact", "claude_json_has_fry_entries")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def summary(self):
        """Short diagnostic text for assertion messages (no environment values)."""
        return "rc=%r timed_out=%r result=%r\n--- stderr tail ---\n%s\n--- stdout tail ---\n%s" % (
            self.rc, self.timed_out, (self.result_text or "")[:500], (self.stderr or "")[-2000:],
            (self.stdout or "")[-2000:])

    def __repr__(self):
        return "E2EResult(rc=%r, result_text=%r, timed_out=%r)" % (self.rc, (self.result_text or "")[:80],
                                                                   self.timed_out)


def claude_available():
    """True when a working ``claude`` binary is installed."""
    try:
        return claude_version(find_claude()) is not None
    except ClaudeNotFound:
        return False


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.dumps(obj, indent=2).encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    return data


def _claude_json_paths(home):
    return [os.path.join(home, ".claude.json"), os.path.join(home, ".claude", ".claude.json")]


def _seed(home):
    settings = os.path.join(home, ".claude", "settings.json")
    seeded = _write_json(settings, SEEDED_SETTINGS)
    for path in _claude_json_paths(home):
        _write_json(path, {"additionalModelOptionsCache": [dict(USER_MODEL_OPTION)]})
    return settings, seeded


def hermetic_env(env, home, keep_env=()):
    """``env`` without enclosing-session variables, rooted at ``home``."""
    keep = LAUNCHER_OWNED_ENV | frozenset(keep_env)
    out = {}
    for k, v in (os.environ if env is None else env).items():
        if v is None or k in _HOST_ONLY_ENV:
            continue
        if (k == "CLAUDECODE" or k.upper().startswith("CLAUDE")) and k not in keep:
            continue
        out[k] = str(v)
    out.update(HOME=home, USERPROFILE=home, CLAUDE_CONFIG_DIR=os.path.join(home, ".claude"),
               DISABLE_AUTOUPDATER="1", CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK="1")
    homes = (("AIL_HOME", ".ai-launchers"), ("FRY_HOME", ".fry"), ("CODEX_HOME", ".codex"), ("GROK_HOME", ".grok"),
             ("CLOUDSDK_CONFIG", os.path.join(".config", "gcloud")))
    if os.name == "nt":
        homes += (("APPDATA", os.path.join("AppData", "Roaming")), ("LOCALAPPDATA", os.path.join("AppData", "Local")))
    for var, sub in homes:
        if not str(out.get(var, "")).startswith(home):
            out[var] = os.path.join(home, sub)
    return out


def _kill_tree(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _parse_events(stdout):
    events = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _model_options(home):
    """All additionalModelOptionsCache entry lists (None for a missing/unparsable file)."""
    out = []
    for path in _claude_json_paths(home):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            out.append((None, None))
            continue
        entries = data.get("additionalModelOptionsCache") if isinstance(data, dict) else None
        out.append((data, entries if isinstance(entries, list) else []))
    return out


def run_e2e(launcher_argv, env, workdir, timeout=300, prompt=DEFAULT_PROMPT, home=None, extra_args=(), keep_env=()):
    """Run Claude Code headless (see module docstring) and return an ``E2EResult``."""
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    home = os.path.abspath(home) if home else tempfile.mkdtemp(prefix="claude-e2e-home-")
    os.makedirs(home, exist_ok=True)
    settings_path, seeded = _seed(home)
    child_env = hermetic_env(env, home, keep_env)
    tail = ["-p", prompt] + list(CLAUDE_FLAGS) + list(extra_args)
    argv = (find_claude() + tail) if launcher_argv is None else (list(launcher_argv) + ["--"] + tail)

    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "cwd": workdir,
              "env": child_env, "encoding": "utf-8", "errors": "replace"}
    if os.name != "nt":
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(argv, **kwargs)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except (subprocess.TimeoutExpired, ValueError):
            stdout, stderr = "", ""
    except BaseException:
        _kill_tree(proc)
        raise

    events = _parse_events(stdout)
    result_event = None
    for ev in events:
        if ev.get("type") == "result":
            result_event = ev
    result_text = result_event.get("result") if result_event else None

    try:
        with open(settings_path, "rb") as f:
            settings_unchanged = f.read() == seeded
    except OSError:
        settings_unchanged = False
    options = _model_options(home)
    intact = all(entries is not None and any(isinstance(e, dict) and e.get("value") == USER_MODEL_OPTION["value"]
                                             for e in entries) for _, entries in options)
    fry = False
    for data, entries in options:
        if isinstance(data, dict) and "selectedModel" in data:
            fry = True
        for e in entries or []:
            blob = " ".join(str(e.get(k, "")) for k in ("value", "label")) if isinstance(e, dict) else str(e)
            if any(m in blob for m in _FRY_MARKERS):
                fry = True
    return E2EResult(rc=None if timed_out else proc.returncode, stdout=stdout or "", stderr=stderr or "",
                     events=events, result_text=result_text if isinstance(result_text, str) else None,
                     result_event=result_event, workdir=workdir, home=home, argv=argv, timed_out=timed_out,
                     settings_unchanged=settings_unchanged, claude_json_user_entry_intact=intact,
                     claude_json_has_fry_entries=fry)
