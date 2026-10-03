"""Launch helpers shared by every launcher (DESIGN §6 as amended by the §0.A spike).

``find_claude(environ=None) -> List[str]``
    argv prefix that runs Claude Code: ``AI_LAUNCHERS_CLAUDE_BIN`` / ``FRY_CLAUDE_BIN`` (a ``.py``
    stub runs under this interpreter, ``.js`` under node) -> native ``claude``/``claude.exe`` on PATH
    -> npm ``claude.cmd`` resolved to ``node <prefix>/node_modules/@anthropic-ai/claude-code/cli.js``
    -> ``cmd /c claude.cmd`` as a last resort (warned) -> well-known install dirs. Raises
    ``ClaudeNotFound`` with an install hint.
``build_child_env(base_env, base_url, token, default_id, background_id, context=None, extra=None,
secret_values=None) -> Dict[str, str]``
    Child env for a gateway launch (DESIGN §6). ``secret_values`` (iterable or ``SecretStore``): every
    inherited variable containing one of them is removed and a secret placed by the caller raises
    ``ValueError`` — provider secrets never reach the child.
``build_direct_env(base_env, base_url, token, models, extra=None, secret_values=None, context=None)``
    Direct launch against a vendor's Anthropic endpoint (DeepSeek, Kimi, OpenRouter, Ollama) with the
    same hygiene; ``models`` keys ``MODEL/OPUS/SONNET/FABLE/HAIKU/SUBAGENT``. ``token`` (the vendor
    key) only ever appears in ``ANTHROPIC_AUTH_TOKEN``.
``run_child(argv, env, cwd=None) -> int``
    Runs Claude Code in the foreground: a no-op Python SIGINT handler (not SIG_IGN, which exec would
    inherit) while it runs, SIGTERM/SIGHUP forwarded on POSIX, no new process group on Windows (the
    console delivers Ctrl+C to the child). Returns the child's exit status (128+N for a signal, 127
    if the binary is missing, 126 if not executable, 130 if interrupted before the child ran).
``claude_version(argv_prefix=None) -> Optional[str]``  (``claude --version``, 10 s timeout)
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import threading

__all__ = ["ClaudeNotFound", "find_claude", "build_child_env", "build_direct_env", "run_child", "claude_version",
           "UNSET_EXACT", "UNSET_PATTERNS", "LOOPBACK_NO_PROXY", "MODEL_ENV_KEYS", "CLAUDE_BIN_ENV",
           "INSTALL_HINT"]

CLAUDE_BIN_ENV = ("AI_LAUNCHERS_CLAUDE_BIN", "FRY_CLAUDE_BIN")
INSTALL_HINT = ("install Claude Code (https://docs.claude.com/en/docs/claude-code/setup, e.g. "
                "`npm install -g @anthropic-ai/claude-code`) or set AI_LAUNCHERS_CLAUDE_BIN to its path")
NPM_CLI_JS = ("node_modules", "@anthropic-ai", "claude-code", "cli.js")
CMD_METACHARS = set("%^&|<>")

#: variables always removed from the child env (on Windows matched case-insensitively)
UNSET_EXACT = ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "CLAUDE_CODE_GZIP_REQUEST_BODIES", "ANTHROPIC_CUSTOM_HEADERS",
               "ANTHROPIC_BETAS", "ANTHROPIC_UNIX_SOCKET", "_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL")
UNSET_PATTERNS = (r"ANTHROPIC_SMALL_FAST_MODEL.*", r"ANTHROPIC_DEFAULT_[A-Z0-9_]+_(?:NAME|DESCRIPTION|"
                  r"SUPPORTED_CAPABILITIES)", r"CLAUDE_CODE_USE_.*", r"CLAUDE_CODE_OAUTH_TOKEN.*")
LOOPBACK_NO_PROXY = ("127.0.0.1", "localhost", "::1")
MODEL_ENV_KEYS = {
    "MODEL": "ANTHROPIC_MODEL",
    "OPUS": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "SONNET": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "FABLE": "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "HAIKU": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "SUBAGENT": "CLAUDE_CODE_SUBAGENT_MODEL",
}
#: Claude Code clamps CLAUDE_CODE_AUTO_COMPACT_WINDOW to [100000, 1000000]; unknown ids get 200000
MIN_COMPACT_WINDOW = 100000
DEFAULT_CLAUDE_WINDOW = 200000

_UNSET_RE = re.compile(r"^(?:%s)$" % "|".join(UNSET_PATTERNS))
_UNSET_RE_I = re.compile(_UNSET_RE.pattern, re.I)
_SEMVER_RE = re.compile(r"(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?)")
_PREFIX_RE = re.compile(r"^claude-via-", re.I)
_ONE_M_RE = re.compile(r"\[1m\]$", re.I)


class ClaudeNotFound(FileNotFoundError):
    """Claude Code could not be located."""


# ---------------------------------------------------------------------------------------
# locating claude
# ---------------------------------------------------------------------------------------

def _warn(msg):
    try:
        sys.stderr.write("warning: %s\n" % msg)
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def _search_path(environ):
    if os.name == "nt":
        for k, v in environ.items():
            if k.upper() == "PATH":
                return v
        return os.defpath
    return environ.get("PATH", os.defpath)


def _which(name, environ):
    return shutil.which(name, path=_search_path(environ))


def _node_for(prefix, environ):
    for cand in (os.path.join(prefix, "node.exe"), os.path.join(prefix, "node")):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return _which("node", environ)


def _npm_shim_argv(shim, environ, windows):
    """``claude.cmd`` npm shim -> ``[node, <prefix>/node_modules/@anthropic-ai/claude-code/cli.js]``,
    else (Windows) ``[ComSpec, "/c", shim]`` with a warning, else None."""
    prefix = os.path.dirname(os.path.abspath(shim))
    cli_js = os.path.join(prefix, *NPM_CLI_JS)
    if os.path.isfile(cli_js):
        node = _node_for(prefix, environ)
        if node:
            return [node, cli_js]
    if windows:
        _warn("running %s through cmd.exe (could not resolve node + cli.js); arguments containing %s may be "
              "mangled" % (shim, "".join(sorted(CMD_METACHARS))))
        return [environ.get("ComSpec") or environ.get("COMSPEC") or "cmd.exe", "/c", shim]
    return None


def _argv_for_path(path, environ, windows):
    low = path.lower()
    if low.endswith(".py"):
        return [sys.executable, path]
    if low.endswith((".js", ".mjs", ".cjs")):
        node = _which("node", environ)
        if not node:
            raise ClaudeNotFound("%s is a JavaScript file but `node` is not on PATH" % path)
        return [node, path]
    if low.endswith((".cmd", ".bat")):
        argv = _npm_shim_argv(path, environ, windows)
        if argv:
            return argv
    return [path]


def find_claude(environ=None):
    """argv prefix that starts Claude Code (see module docstring)."""
    environ = os.environ if environ is None else environ
    windows = os.name == "nt"
    for var in CLAUDE_BIN_ENV:
        value = (environ.get(var) or "").strip()
        if not value:
            continue
        path = value if os.path.isfile(value) else _which(value, environ)
        if not path:
            raise ClaudeNotFound("%s=%s does not exist" % (var, value))
        return _argv_for_path(os.path.abspath(path), environ, windows)
    if windows:
        native = _which("claude.exe", environ)
        if native:
            return [native]
        shim = _which("claude.cmd", environ)
        if shim:
            return _npm_shim_argv(shim, environ, True)
    else:
        found = _which("claude", environ)
        if found:
            return [found]
    home = environ.get("USERPROFILE" if windows else "HOME") or os.path.expanduser("~")
    exe = "claude.exe" if windows else "claude"
    for cand in (os.path.join(home, ".local", "bin", exe), os.path.join(home, ".claude", "local", exe)):
        if os.path.isfile(cand) and (windows or os.access(cand, os.X_OK)):
            return [cand]
    raise ClaudeNotFound("Claude Code (`claude`) not found on PATH - " + INSTALL_HINT)


def claude_version(argv_prefix=None):
    """``X.Y.Z`` from ``claude --version`` (10 s timeout) or None."""
    try:
        argv = list(argv_prefix) if argv_prefix else find_claude()
        proc = subprocess.run(argv + ["--version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=10, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    m = _SEMVER_RE.search(proc.stdout or "") or _SEMVER_RE.search(proc.stderr or "")
    return m.group(1) if (m and proc.returncode == 0) else None


# ---------------------------------------------------------------------------------------
# child environments
# ---------------------------------------------------------------------------------------

class _Env(object):
    """dict wrapper with Windows' case-insensitive variable semantics."""

    def __init__(self, base, windows):
        self.windows = windows
        self.d = {str(k): str(v) for k, v in (base or {}).items() if v is not None}

    def _keys(self, name):
        if self.windows:
            low = name.lower()
            return [k for k in self.d if k.lower() == low]
        return [name] if name in self.d else []

    def get(self, name):
        keys = self._keys(name)
        return self.d[keys[0]] if keys else None

    def unset(self, name):
        for k in self._keys(name):
            del self.d[k]

    def set(self, name, value):
        self.unset(name)
        self.d[name] = str(value)

    def unset_matching(self):
        rx = _UNSET_RE_I if self.windows else _UNSET_RE
        exact = {n.lower() for n in UNSET_EXACT} if self.windows else set(UNSET_EXACT)
        for k in list(self.d):
            if (k.lower() if self.windows else k) in exact or rx.match(k):
                del self.d[k]

    def add_no_proxy(self):
        entries = []
        for name in ("NO_PROXY", "no_proxy"):
            for e in (self.get(name) or "").split(","):
                e = e.strip()
                if e and e not in entries:
                    entries.append(e)
        for e in LOOPBACK_NO_PROXY:
            if e not in entries:
                entries.append(e)
        value = ",".join(entries)
        self.set("NO_PROXY", value)
        if not self.windows:
            self.d["no_proxy"] = value


def _secret_list(secret_values):
    if secret_values is None:
        return []
    if hasattr(secret_values, "values_for_redaction"):
        values = secret_values.values_for_redaction()
    elif isinstance(secret_values, str):
        values = [secret_values]
    else:
        values = list(secret_values)
    return sorted({str(v) for v in values if v and len(str(v)) >= 4}, key=len, reverse=True)


def _scrub(env, secrets, placed, allowed=()):
    """Remove inherited variables that carry a secret; raise if one of ``placed`` (set by us/the
    caller) carries one outside ``allowed``. Never includes a secret value in messages."""
    if not secrets:
        return
    for k in list(env.d):
        if k in allowed:
            continue
        if any(s in env.d[k] for s in secrets):
            if k in placed:
                raise ValueError("refusing to place a provider secret in the child environment (%s)" % k)
            del env.d[k]


def _apply_extra(env, extra, placed):
    for k, v in (extra or {}).items():
        if v is None:
            env.unset(k)
        else:
            env.set(k, v)
            placed.add(k)


def _option_name(default_id):
    """``claude-via-xai,grok-4.7[1m]`` -> ``grok-4.7 (xai)``; role aliases -> the role."""
    bare = _ONE_M_RE.sub("", _PREFIX_RE.sub("", default_id or ""))
    if "," in bare:
        provider, model = bare.split(",", 1)
        return "%s (%s)" % (model, provider)
    return bare or default_id


def build_child_env(base_env, base_url, token, default_id, background_id, context=None, extra=None,
                    secret_values=None):
    """Environment for Claude Code launched against the in-process gateway (DESIGN §6)."""
    windows = os.name == "nt"
    env = _Env(os.environ if base_env is None else base_env, windows)
    env.unset_matching()
    placed = set()

    def put(name, value):
        env.set(name, value)
        placed.add(name)

    put("ANTHROPIC_BASE_URL", base_url.rstrip("/"))
    put("ANTHROPIC_AUTH_TOKEN", token)
    put("ANTHROPIC_API_KEY", "")
    put("ANTHROPIC_MODEL", default_id)
    for family in ("OPUS", "SONNET", "FABLE"):
        put(MODEL_ENV_KEYS[family], default_id)
    put("ANTHROPIC_DEFAULT_HAIKU_MODEL", background_id or default_id)
    put("ANTHROPIC_CUSTOM_MODEL_OPTION", default_id)
    put("ANTHROPIC_CUSTOM_MODEL_OPTION_NAME", _option_name(default_id))
    put("ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION", "Default route via the local AI gateway")
    put("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1")
    put("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    if context and int(context) < DEFAULT_CLAUDE_WINDOW:
        put("CLAUDE_CODE_AUTO_COMPACT_WINDOW", str(max(MIN_COMPACT_WINDOW, int(context))))
    env.add_no_proxy()
    _apply_extra(env, extra, placed)
    _scrub(env, _secret_list(secret_values), placed)
    return env.d


def build_direct_env(base_env, base_url, token, models, extra=None, secret_values=None, context=None):
    """Environment for Claude Code talking directly to a vendor's Anthropic-compatible endpoint."""
    windows = os.name == "nt"
    unknown = sorted(set(models or {}) - set(MODEL_ENV_KEYS))
    if unknown:
        raise ValueError("unknown model role(s) %s (expected %s)" % (", ".join(unknown), ", ".join(MODEL_ENV_KEYS)))
    env = _Env(os.environ if base_env is None else base_env, windows)
    env.unset_matching()
    env.unset("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY")
    placed = set()

    def put(name, value):
        env.set(name, value)
        placed.add(name)

    put("ANTHROPIC_BASE_URL", base_url.rstrip("/"))
    put("ANTHROPIC_AUTH_TOKEN", token)
    put("ANTHROPIC_API_KEY", "")
    for role, name in MODEL_ENV_KEYS.items():
        value = (models or {}).get(role)
        if value:
            put(name, value)
    put("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    if context and int(context) < DEFAULT_CLAUDE_WINDOW:
        put("CLAUDE_CODE_AUTO_COMPACT_WINDOW", str(max(MIN_COMPACT_WINDOW, int(context))))
    env.add_no_proxy()
    _apply_extra(env, extra, placed)
    _scrub(env, _secret_list(secret_values), placed, allowed=("ANTHROPIC_AUTH_TOKEN",))
    return env.d


# ---------------------------------------------------------------------------------------
# running claude
# ---------------------------------------------------------------------------------------

def _noop_handler(signum, frame):
    """Parent ignores Ctrl+C while the child runs (a Python handler is reset on exec, SIG_IGN is not)."""


def run_child(argv, env, cwd=None):
    """Run ``argv`` in the foreground and return its exit status (see module docstring)."""
    argv = [str(a) for a in argv]
    if os.name == "nt" and len(argv) > 2 and argv[1].lower() == "/c" and \
            any(CMD_METACHARS & set(a) for a in argv[3:]):
        _warn("cmd.exe will interpret %s in the arguments" % "".join(sorted(CMD_METACHARS)))
    holder = {"proc": None, "pending": []}

    def forward(signum, frame):
        proc = holder["proc"]
        if proc is None:
            holder["pending"].append(signum)
        elif proc.poll() is None:
            try:
                proc.send_signal(signum)
            except OSError:
                pass

    saved = {}
    if threading.current_thread() is threading.main_thread():
        handlers = [(signal.SIGINT, _noop_handler)]
        if os.name == "nt" and hasattr(signal, "SIGBREAK"):
            handlers.append((signal.SIGBREAK, _noop_handler))
        else:
            handlers += [(getattr(signal, n), forward) for n in ("SIGTERM", "SIGHUP") if hasattr(signal, n)]
        for sig, handler in handlers:
            try:
                saved[sig] = signal.signal(sig, handler)
            except (OSError, ValueError):
                pass
    try:
        try:
            proc = subprocess.Popen(argv, env=env, cwd=cwd)
        except FileNotFoundError as exc:
            _warn("cannot run %s: %s - %s" % (argv[0] if argv else "?", exc.strerror or exc, INSTALL_HINT))
            return 127
        except OSError as exc:  # not executable / exec format error
            _warn("cannot run %s: %s" % (argv[0] if argv else "?", exc.strerror or exc))
            return 126
        holder["proc"] = proc
        for signum in holder["pending"]:
            forward(signum, None)
        try:
            rc = proc.wait()
        except KeyboardInterrupt:  # no handler installed (not the main thread): the child got it too
            rc = proc.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        for sig, old in saved.items():
            try:
                signal.signal(sig, old if old is not None else signal.SIG_DFL)
            except (OSError, ValueError):
                pass
    return 128 - rc if rc < 0 else rc
