"""Chat-only CLI dialect (DESIGN §3.5) — OpenCode only.

The whole transcript (system + every turn, ``<system-reminder>`` blocks stripped, images ->
``[image]``) is flattened into one prompt and piped to ``opencode run --model <provider/model>`` on
STDIN (UTF-8, 300 s timeout, ``CREATE_NO_WINDOW`` on Windows, at most 2 concurrent runs per CLI).
The output is cleaned of ANSI CSI/OSC sequences and ``> build · <model>`` header lines and
returned as one text block (``end_turn``, estimated usage). Probes (``max_tokens <= 1``) are
answered locally with "ok".

Provider options: ``cli`` (only ``"opencode"``), ``cli_bin`` (explicit binary path), ``model_prefix``
(default ``"opencode/"``; ids already containing ``/`` are passed verbatim), ``timeout`` (seconds),
``max_concurrency`` (default 2), ``cwd`` (working directory; default a private scratch directory so
the CLI agent never touches the user's project).
"""

import atexit
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading

from .. import errors
from ..compat import json_dumps_compact
from ..events import Finish, TextDelta, Usage
from .base import Dialect

__all__ = ["CliDialect", "SUPPORTED_CLIS", "DEFAULT_TIMEOUT", "DEFAULT_CONCURRENCY", "PREAMBLE", "build_prompt",
           "clean_output", "strip_ansi", "cli_model_id", "find_cli"]

DEFAULT_TIMEOUT = 300.0
DEFAULT_CONCURRENCY = 2
TAIL_CHARS = 800

#: cli name -> (binary candidates in lookup order, install hint)
SUPPORTED_CLIS = {
    "opencode": (("opencode", "opencode.cmd", "opencode.exe"),
                 "install it with `npm i -g opencode-ai` (https://opencode.ai) or set the provider's options.cli_bin"),
}

PREAMBLE = ("Below is a conversation transcript. Reply as the Assistant to the last User message, in plain text. "
            "Tools are not available in this conversation.")

_ANSI_RE = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"      # OSC ... BEL | ST
    r"|\x1b[PX^_][^\x1b]*\x1b\\"              # DCS / SOS / PM / APC ... ST
    r"|\x1b\[[0-?]*[ -/]*[@-~]"               # CSI
    r"|\x9b[0-?]*[ -/]*[@-~]"                 # 8-bit CSI
    r"|\x1b[ -/]*[0-~]"                       # other ESC sequences (charset, keypad, ESC c ...)
)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HEADER_RE = re.compile(r"^\s*>\s+\S+\s+[·•]\s+\S")
_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>[ \t]*\n?", re.S)
_ONE_M_RE = re.compile(r"\[1m\]$", re.I)

_sem_lock = threading.Lock()
_semaphores = {}   # cli key -> BoundedSemaphore
_scratch = []      # [path] of the lazily created default working directory


# ---------------------------------------------------------------------------------------
# prompt / output
# ---------------------------------------------------------------------------------------

def _clean(text):
    return _REMINDER_RE.sub("", text or "").strip()


def build_prompt(req):
    """Flatten a NormalizedRequest into the CLI prompt (``System:``/``User:``/``Assistant:`` sections)."""
    parts = [PREAMBLE]
    system = "\n\n".join(t for t in (_clean(s) for s in req.system) if t)
    fmt = req.output_format
    if isinstance(fmt, dict) and fmt.get("schema") is not None:
        system = (system + "\n\n" if system else "") + \
            "Respond with ONLY a JSON object matching this JSON schema: " + json_dumps_compact(fmt["schema"])
    if system:
        parts.append("System:\n" + system)
    names = req.tool_use_names_by_id()
    for msg in req.messages:
        pieces = []
        for b in msg.blocks:
            if b.type == "text":
                t = _clean(b.text)
                if t:
                    pieces.append(t)
            elif b.type == "image":
                pieces.append("[image]")
            elif b.type == "document":
                t = _clean(b.text) if b.text else ""
                pieces.append(t or "[document%s]" % ((": " + b.title) if b.title else ""))
            elif b.type == "tool_use":
                pieces.append("Tool call %s: %s" % (b.name, json_dumps_compact(b.input or {})))
            elif b.type == "tool_result":
                body = _clean(b.result_text())
                media = " ".join("[image]" if m.type == "image" else "[document]" for m in b.result_media())
                body = "\n".join(x for x in (body, media) if x) or "(no output)"
                pieces.append("Tool result (%s)%s:\n%s" % (names.get(b.tool_use_id, "tool"),
                                                           " [error]" if b.is_error else "", body))
        if pieces:
            parts.append("%s:\n%s" % ("User" if msg.role == "user" else "Assistant", "\n\n".join(pieces)))
    return "\n\n".join(parts) + "\n"


def strip_ansi(text):
    """Remove ANSI CSI/OSC/other escape sequences and stray control characters (keeps \\n \\t \\r)."""
    return _CTRL_RE.sub("", _ANSI_RE.sub("", text or ""))


def clean_output(text):
    """CLI stdout -> answer text: ANSI stripped, carriage-return redraws resolved, header lines removed."""
    lines = []
    for line in strip_ansi(text).split("\n"):
        line = line.rstrip("\r")
        if "\r" in line:
            line = line.rsplit("\r", 1)[-1]
        if _HEADER_RE.match(line):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip("\n")


def _tail(text, limit=TAIL_CHARS):
    text = clean_output(text).strip()
    return ("…" + text[-limit:]) if len(text) > limit else text


def cli_model_id(model_id, prefix="opencode/"):
    """Route model -> ``--model`` value (``nemotron-3-ultra-free`` -> ``opencode/nemotron-3-ultra-free``)."""
    m = _ONE_M_RE.sub("", model_id or "")
    return m if ("/" in m or not prefix) else prefix + m


def _estimate(text):
    return int(math.ceil(len((text or "").encode("utf-8")) / 3.6))


# ---------------------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------------------

def find_cli(cli, explicit=None):
    """Absolute path of the CLI binary or None (``explicit`` path/name first, then PATH)."""
    if explicit:
        if os.path.isfile(explicit):
            return os.path.abspath(explicit)
        return shutil.which(explicit)
    for name in SUPPORTED_CLIS[cli][0]:
        found = shutil.which(name)
        if found:
            return found
    return None


def _semaphore(key, size):
    with _sem_lock:
        sem = _semaphores.get(key)
        if sem is None:
            sem = _semaphores[key] = threading.BoundedSemaphore(max(1, int(size)))
        return sem


def _scratch_dir():
    with _sem_lock:
        if not _scratch or not os.path.isdir(_scratch[0]):
            path = tempfile.mkdtemp(prefix="ai-gateway-cli-")
            _scratch[:] = [path]
            atexit.register(shutil.rmtree, path, True)
        return _scratch[0]


def _child_env():
    """The user's own environment (the CLI's credentials live there) with colours off. Gateway-held
    secrets (SecretStore) are never added."""
    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    return env


def _kill_tree(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run(argv, prompt, timeout, cwd, env):
    """-> (returncode, stdout, stderr); raises ``subprocess.TimeoutExpired`` / ``OSError``."""
    kwargs = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
              "encoding": "utf-8", "errors": "replace", "cwd": cwd, "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    else:
        kwargs["start_new_session"] = True  # lets a timeout kill the CLI's whole process tree
    proc = subprocess.Popen(argv, **kwargs)
    try:
        out, err = proc.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        raise
    except BaseException:
        _kill_tree(proc)
        raise
    return proc.returncode, out or "", err or ""


class CliDialect(Dialect):
    name = "cli"

    def execute(self, ctx):
        req = ctx.req
        opts = ctx.provider.options or {}
        label = "[%s/%s] " % (ctx.provider.id, ctx.model.id)
        if req.is_probe():
            yield TextDelta(0, "ok")
            yield Usage(ctx.est_tokens or 1, 1)
            yield Finish("end_turn")
            return
        cli = opts.get("cli") or "opencode"
        if cli not in SUPPORTED_CLIS:
            raise errors.GatewayError(400, "invalid_request_error",
                                      label + "unsupported CLI %r (supported: %s)" % (cli, ", ".join(SUPPORTED_CLIS)))
        binary = find_cli(cli, opts.get("cli_bin"))
        if not binary:
            raise errors.GatewayError(503, "api_error", label + "%s CLI not found on PATH — %s" %
                                      (cli, SUPPORTED_CLIS[cli][1]), False)
        model = cli_model_id(ctx.model.id, opts.get("model_prefix", "opencode/"))
        argv = [binary, "run", "--model", model]
        prompt = build_prompt(req)
        timeout = float(opts.get("timeout") or DEFAULT_TIMEOUT)
        redact = ctx.secrets.redact if ctx.secrets is not None else (lambda t: t)
        ctx.trace("cli_run", provider=ctx.provider.id, cli=cli, model=model, prompt_chars=len(prompt))
        with _semaphore(os.path.normcase(binary), opts.get("max_concurrency") or DEFAULT_CONCURRENCY):
            try:
                rc, out, err = _run(argv, prompt, timeout, opts.get("cwd") or _scratch_dir(), _child_env())
            except subprocess.TimeoutExpired:
                raise errors.GatewayError(502, "api_error", label + "%s timed out after %d s" % (cli, timeout), False)
            except OSError as exc:
                raise errors.GatewayError(503, "api_error", label + "cannot run %s (%s) — %s" %
                                          (cli, exc.strerror or exc, SUPPORTED_CLIS[cli][1]), False)
        if rc != 0:
            detail = _tail(err) or _tail(out) or "no output"
            raise errors.GatewayError(502, "api_error", redact(label + "%s exited with status %s: %s" %
                                                               (cli, rc, detail)), False)
        text = clean_output(out)
        if not text.strip():
            detail = _tail(err)
            raise errors.GatewayError(502, "api_error", redact(label + "%s returned no output%s" %
                                                               (cli, (": " + detail) if detail else "")), False)
        yield TextDelta(0, text)
        yield Usage(ctx.est_tokens or _estimate(prompt), _estimate(text))
        yield Finish("end_turn")
