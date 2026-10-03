"""Per-target rules of the Responses dialect (DESIGN §3.2): URL, headers, body rules, CLI versions.

``get_target(name) -> ResponsesTarget`` for ``openai_api``, ``chatgpt_codex``, ``grok_cli_proxy`` and
``xai_api``. A target is immutable data plus small pure helpers; ``dialects/responses.py`` builds the
target-independent body and then calls ``ResponsesTarget.apply_body_rules`` / ``headers``.

| target         | sig tag        | body rules                                                                 |
|----------------|----------------|----------------------------------------------------------------------------|
| openai_api     | openai_api     | ``max_output_tokens`` clamped (>= 16); temperature/top_p only non-reasoning |
| chatgpt_codex  | codex_chatgpt  | delete max_output_tokens, temperature, top_p, truncation, user, metadata,   |
|                |                | previous_response_id, max_completion_tokens; call_ids <= 64; Responses-Lite |
| grok_cli_proxy | grok_cli       | delete max_output_tokens, temperature; schema ``no_root_combinators``      |
| xai_api        | xai_api        | ``max_output_tokens`` clamped; schema ``no_root_combinators``              |

Versions: ``codex_version(runtime=None, environ=None)`` = ``AI_GATEWAY_CODEX_VERSION`` > cached
``codex --version`` > ``CODEX_PINNED_VERSION``; ``grok_client_version`` likewise with
``AI_GATEWAY_GROK_CLIENT_VERSION`` / ``grok --version`` / ``GROK_PINNED_VERSION``. Detected versions are
cached per process (keyed by binary + PATH) and in ``ProviderRuntime.state["versions"]``.
"""

import os
import platform
import re
import shutil
import subprocess
import threading
import uuid

from . import catalog

__all__ = [
    "ResponsesTarget", "TARGETS", "get_target", "codex_version", "grok_client_version", "detect_cli_version",
    "clear_version_cache", "codex_user_agent", "terminal_user_agent", "CODEX_PINNED_VERSION",
    "GROK_PINNED_VERSION", "CODEX_ORIGINATOR", "GROK_CLIENT_IDENTIFIER", "VERSION_TIMEOUT",
    "LITE_HEADER", "MIN_OUTPUT_TOKENS",
]

CODEX_PINNED_VERSION = "0.160.0"
#: Grok Build CLI version sent when ``grok`` is not installed (override: AI_GATEWAY_GROK_CLIENT_VERSION).
GROK_PINNED_VERSION = "0.1.40"
CODEX_ORIGINATOR = "codex_cli_rs"
GROK_CLIENT_IDENTIFIER = "grok-cli"
LITE_HEADER = "x-openai-internal-codex-responses-lite"
VERSION_TIMEOUT = 5.0
#: OpenAI rejects ``max_output_tokens`` below 16 (``/model`` probes send max_tokens 1).
MIN_OUTPUT_TOKENS = 16

_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?)")
_UNAUTHORIZED_RE = re.compile(r"token_expired|invalid_token", re.I)


# ---------------------------------------------------------------------------------------
# CLI versions
# ---------------------------------------------------------------------------------------

_version_cache = {}  # (binary, PATH) -> Optional[str]
_version_lock = threading.Lock()


def clear_version_cache():
    """Forget detected CLI versions (tests)."""
    with _version_lock:
        _version_cache.clear()


def detect_cli_version(binary, environ=None, timeout=None):
    """Run ``<binary> --version`` (resolved on ``environ["PATH"]``) and return the first x.y.z found,
    or None when the binary is missing, fails, times out or prints no version."""
    env = os.environ if environ is None else environ
    path = shutil.which(binary, path=env.get("PATH") or None)
    if not path:
        return None
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        cp = subprocess.run([path, "--version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=VERSION_TIMEOUT if timeout is None else timeout,
                            **kwargs)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if cp.returncode != 0:
        return None
    out = (cp.stdout or b"").decode("utf-8", "replace") + "\n" + (cp.stderr or b"").decode("utf-8", "replace")
    m = _VERSION_RE.search(out)
    return m.group(1) if m else None


def _cli_version(kind, env_var, binary, pinned, runtime, environ):
    env = os.environ if environ is None else environ
    override = _header_safe((env.get(env_var) or "").strip())
    if override:
        return override
    versions = runtime.setdefault("versions", dict) if runtime is not None else None
    if versions is not None:
        with runtime.lock:
            cached = versions.get(kind)
        if cached:
            return cached
    key = (binary, env.get("PATH") or "")
    with _version_lock:  # single flight: concurrent first requests spawn the CLI once
        if key not in _version_cache:
            _version_cache[key] = detect_cli_version(binary, env)
        found = _version_cache[key]
    version = found or pinned
    if versions is not None:
        with runtime.lock:
            versions[kind] = version
    return version


def codex_version(runtime=None, environ=None):
    """``AI_GATEWAY_CODEX_VERSION`` > cached ``codex --version`` > ``CODEX_PINNED_VERSION``."""
    return _cli_version("codex", "AI_GATEWAY_CODEX_VERSION", "codex", CODEX_PINNED_VERSION, runtime, environ)


def grok_client_version(runtime=None, environ=None):
    """``AI_GATEWAY_GROK_CLIENT_VERSION`` > cached ``grok --version`` > ``GROK_PINNED_VERSION``."""
    return _cli_version("grok", "AI_GATEWAY_GROK_CLIENT_VERSION", "grok", GROK_PINNED_VERSION, runtime, environ)


# ---------------------------------------------------------------------------------------
# User-Agent (codex_cli_rs/<ver> (<os> <osver>; <arch>) <term>)
# ---------------------------------------------------------------------------------------

def _header_safe(value):
    """Printable ASCII only (header values must not carry control characters)."""
    return "".join(ch if " " <= ch <= "~" else "_" for ch in (value or ""))


def _linux_release():
    for path in ("/etc/os-release", "/usr/lib/os-release"):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                fields = {}
                for line in f:
                    k, sep, v = line.strip().partition("=")
                    if sep:
                        fields[k] = v.strip().strip("\"'")
        except OSError:
            continue
        name = fields.get("NAME") or fields.get("ID")
        if name:
            return name, fields.get("VERSION_ID") or platform.release()
    return "Linux", platform.release()


_os_info_cache = []


def _os_info():
    """(os type, os version, arch) like the Rust ``os_info`` crate codex uses; computed once."""
    if _os_info_cache:
        return _os_info_cache[0]
    system = platform.system()
    if system == "Darwin":
        info = ("Mac OS", platform.mac_ver()[0] or platform.release())
    elif system == "Windows":
        info = ("Windows", platform.version())
    elif system == "Linux":
        info = _linux_release()
    else:
        info = (system or "Unknown", platform.release())
    arch = platform.machine() or "unknown"
    arch = {"amd64": "x86_64", "x64": "x86_64", "arm64": "arm64", "aarch64": "aarch64"}.get(arch.lower(), arch)
    _os_info_cache.append((info[0] or "Unknown", info[1] or "unknown", arch))
    return _os_info_cache[0]


def terminal_user_agent(environ=None):
    """Terminal token of codex's User-Agent (TERM_PROGRAM[/version], known terminals, TERM, unknown)."""
    env = os.environ if environ is None else environ
    program = (env.get("TERM_PROGRAM") or "").strip()
    if program:
        version = (env.get("TERM_PROGRAM_VERSION") or "").strip()
        return program + ("/" + version if version else "")
    for var, label in (("WEZTERM_VERSION", "WezTerm"), ("KONSOLE_VERSION", "Konsole"), ("VTE_VERSION", "VTE")):
        if env.get(var):
            return "%s/%s" % (label, env[var].strip())
    if env.get("WT_SESSION"):
        return "WindowsTerminal"
    if env.get("KITTY_WINDOW_ID"):
        return "kitty"
    return (env.get("TERM") or "").strip() or "unknown"


def codex_user_agent(version, environ=None):
    os_type, os_version, arch = _os_info()
    return _header_safe("%s/%s (%s %s; %s) %s" % (CODEX_ORIGINATOR, version, os_type, os_version, arch,
                                                   terminal_user_agent(environ)))


# ---------------------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------------------

class ResponsesTarget(object):
    """Static rules for one Responses endpoint flavour (see module docstring).

    ``name``              config target name
    ``sig_tag``           thinking-signature target tag (``signatures.TARGETS``)
    ``default_base``      base URL when the provider has none; ``path`` appended (``/responses``)
    ``schema_mode``       tool-schema scrub mode unless the provider sets ``schema_mode``
    ``max_output``        send ``max_output_tokens`` (clamped)
    ``sampling``          "non_reasoning" | "always" | "top_p" | "never" (temperature/top_p policy)
    ``drop``              body keys always deleted
    ``max_call_id``       call_id length limit (None = unlimited)
    ``reasoning_style``   "openai" (``{effort, summary:"auto"}``) | "xai" (``{effort}`` only if the model
                          has ``effort_param``)
    ``text_format``       structured output via ``text.format`` (else a system-prompt instruction)
    ``input_file``        PDFs/URL documents as ``input_file`` (else a text placeholder)
    ``cache_key``         send ``prompt_cache_key`` (grok targets use ``x-grok-conv-id`` instead)
    """

    __slots__ = ("name", "sig_tag", "default_base", "path", "schema_mode", "max_output", "sampling", "drop",
                 "max_call_id", "reasoning_style", "text_format", "input_file", "cache_key", "fallback_capable")

    def __init__(self, name, sig_tag, default_base, schema_mode="basic", max_output=True, sampling="always",
                 drop=(), max_call_id=None, reasoning_style="openai", text_format=True, input_file=False,
                 cache_key=True, fallback_capable=False, path="/responses"):
        self.name = name
        self.sig_tag = sig_tag
        self.default_base = default_base
        self.path = path
        self.schema_mode = schema_mode
        self.max_output = max_output
        self.sampling = sampling
        self.drop = tuple(drop)
        self.max_call_id = max_call_id
        self.reasoning_style = reasoning_style
        self.text_format = text_format
        self.input_file = input_file
        self.cache_key = cache_key
        self.fallback_capable = fallback_capable

    def __repr__(self):
        return "ResponsesTarget(%r)" % (self.name,)

    # ---- model facts -------------------------------------------------------------------------
    def is_reasoning(self, spec):
        """Reasoning model on this target (decides reasoning/include/sampling params)."""
        model_id = spec.id if spec is not None else ""
        if self.reasoning_style == "xai":
            return catalog.is_xai_reasoning(model_id, spec)
        return bool(spec is not None and spec.reasoning) or catalog.is_openai_reasoning(model_id)

    def is_lite(self, spec):
        """ChatGPT Codex Responses-Lite (gpt-6.x): instructions travel as a developer message."""
        if self.name != "chatgpt_codex" or spec is None:
            return False
        return bool(spec.responses_lite) or catalog.is_responses_lite(spec.id)

    # ---- request pieces -----------------------------------------------------------------------
    def url(self, base_url, path_override=None):
        base = (base_url or self.default_base).rstrip("/")
        path = path_override or self.path
        return base + "/" + path.lstrip("/")

    def unauthorized(self, status, text):
        """``send_with_auth_retry`` predicate: ChatGPT answers expired tokens with 403 sometimes."""
        return self.name == "chatgpt_codex" and status == 403 and bool(_UNAUTHORIZED_RE.search(text or ""))

    def headers(self, session_id, runtime=None, lite=False, environ=None, client_identifier=None):
        """Target headers (auth headers are merged on top by ``send_with_auth_retry``)."""
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        sid = _header_safe(session_id or "")
        if self.name == "chatgpt_codex":
            version = codex_version(runtime, environ)
            h["originator"] = CODEX_ORIGINATOR
            h["User-Agent"] = codex_user_agent(version, environ)
            h["version"] = version
            if sid:
                h["session-id"] = sid
            if lite:
                h[LITE_HEADER] = "true"
        elif self.name == "grok_cli_proxy":
            h["X-XAI-Token-Auth"] = "xai-grok-cli"
            h["x-authenticateresponse"] = "authenticate-response"
            h["x-grok-client-version"] = grok_client_version(runtime, environ)
            h["x-grok-client-identifier"] = _header_safe(client_identifier or GROK_CLIENT_IDENTIFIER)
            h["x-grok-client-mode"] = "headless"
            if sid:
                h["x-grok-conv-id"] = sid
                h["x-grok-session-id"] = sid
            h["x-grok-req-id"] = str(uuid.uuid4())
        elif self.name == "xai_api" and sid:
            h["x-grok-conv-id"] = sid  # xAI prompt-cache routing
        return h

    def apply_body_rules(self, body, req, spec, reasoning, lite=False):
        """Add the target's token/sampling params and delete what it rejects (mutates ``body``)."""
        if self.max_output and req.max_tokens:
            limit = int(req.max_tokens)
            if spec is not None and spec.max_output:
                limit = min(limit, int(spec.max_output))
            body["max_output_tokens"] = max(MIN_OUTPUT_TOKENS, limit)
        allow_temp = self.sampling == "always" or (self.sampling == "non_reasoning" and not reasoning)
        allow_top_p = allow_temp or self.sampling == "top_p"
        if allow_temp and req.temperature is not None:
            body["temperature"] = req.temperature
        if allow_top_p and req.top_p is not None:
            body["top_p"] = req.top_p
        for key in self.drop:
            body.pop(key, None)
        if lite:
            instructions = body.pop("instructions", None)
            if instructions:
                dev = {"type": "message", "role": "developer",
                       "content": [{"type": "input_text", "text": instructions}]}
                body["input"] = [dev] + list(body.get("input") or [])
        return body


_CODEX_DROP = ("max_output_tokens", "temperature", "top_p", "truncation", "user", "metadata",
               "previous_response_id", "max_completion_tokens")

TARGETS = {
    "openai_api": ResponsesTarget(
        "openai_api", "openai_api", "https://api.openai.com/v1", sampling="non_reasoning", input_file=True),
    "chatgpt_codex": ResponsesTarget(
        "chatgpt_codex", "codex_chatgpt", "https://chatgpt.com/backend-api/codex", max_output=False,
        sampling="never", drop=_CODEX_DROP, max_call_id=64),
    "grok_cli_proxy": ResponsesTarget(
        "grok_cli_proxy", "grok_cli", "https://cli-chat-proxy.grok.com/v1", schema_mode="no_root_combinators",
        max_output=False, sampling="top_p", drop=("max_output_tokens", "temperature"),
        reasoning_style="xai", text_format=False, cache_key=False, fallback_capable=True),
    "xai_api": ResponsesTarget(
        "xai_api", "xai_api", "https://api.x.ai/v1", schema_mode="no_root_combinators", reasoning_style="xai",
        cache_key=False),
}


def get_target(name):
    """``ResponsesTarget`` for ``name`` (``None`` -> ``openai_api``); ``ValueError`` if unknown."""
    try:
        return TARGETS[name or "openai_api"]
    except KeyError:
        raise ValueError("unknown responses target %r (known: %s)" % (name, ", ".join(sorted(TARGETS))))
