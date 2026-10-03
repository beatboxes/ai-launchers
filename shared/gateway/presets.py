"""Provider presets (DESIGN §1). Owned for CONTENT by agent F; structure frozen.

``PRESETS``: preset name -> dict with every ``ProviderSpec`` field the preset sets, plus these
launcher-only keys (stripped by ``provider_from_preset``):
  ``catalog``          catalog family for ``models`` (``catalog.get_catalog``)
  ``secret_env``       env vars that may hold the API key, in priority order (api_key auth only)
  ``default_model``    default role model id
  ``background_model`` background role model id
  ``notes``            free text for ``models``/``doctor`` output

``provider_from_preset(name, overrides=None) -> ProviderSpec`` — deep-merges ``overrides`` (a
dict of ProviderSpec fields; ``id`` renames the provider; ``models`` replaces the catalog;
``auth``/``options``/``headers``/``fallback`` are merged key-wise).
``route_table_from_presets(names, overrides=None, roles=None) -> RouteTable`` — providers in
``names`` order; roles default to the FIRST preset's default/background models.
"""

import copy

from .catalog import get_catalog
from .config import ModelSpec, ProviderSpec, RouteTable

__all__ = ["PRESETS", "LAUNCHER_KEYS", "DEFAULT_PROVIDER_ALIASES", "GEMINI_TOOL_NAME_REGEX",
           "provider_from_preset", "preset_names", "secret_env_vars", "default_roles", "route_table_from_presets"]

LAUNCHER_KEYS = ("catalog", "secret_env", "default_model", "background_model", "notes")
GEMINI_TOOL_NAME_REGEX = r"^[a-zA-Z_][a-zA-Z0-9_\-.:]{0,63}$"

DEFAULT_PROVIDER_ALIASES = {
    "grok": ["xai"], "xai": ["grok"], "codex": ["openai"], "openai": ["codex"], "local-ollama": ["ollama"],
}

_CONSERVATIVE = "context/max_output values the provider does not publish are conservative estimates"

PRESETS = {
    "xai": {
        "display_name": "xAI API", "dialect": "openai_chat", "profile": "xai", "base_url": "https://api.x.ai/v1",
        "auth": {"kind": "api_key", "secret": "xai", "style": "bearer"}, "allow_unlisted": True,
        "schema_mode": "no_root_combinators",
        "catalog": "xai", "secret_env": ["XAI_API_KEY"],
        "default_model": "grok-4.7", "background_model": "grok-4.20-0309-non-reasoning",
        "notes": "chat/completions; grok-4.20-multi-agent-0309 is served via Responses (xai_api); "
                 "max_output 64000 is conservative",
    },
    "grok": {
        "display_name": "Grok (grok login)", "dialect": "responses", "target": "grok_cli_proxy",
        "base_url": "https://cli-chat-proxy.grok.com/v1", "auth": {"kind": "grok_cli"}, "allow_unlisted": True,
        "schema_mode": "no_root_combinators",
        "fallback": {"dialect": "openai_chat", "profile": "xai", "base_url": "https://api.x.ai/v1",
                     "schema_mode": "no_root_combinators"},
        "catalog": "grok", "default_model": "grok-4.7", "background_model": "grok-4.20-0309-non-reasoning",
        "notes": "Grok Build login (~/.grok/auth.json); falls back to api.x.ai chat/completions with the same "
                 "token when the CLI proxy rejects the client",
    },
    "openai": {
        "display_name": "OpenAI API", "dialect": "responses", "target": "openai_api",
        "base_url": "https://api.openai.com/v1", "auth": {"kind": "api_key", "secret": "openai", "style": "bearer"},
        "allow_unlisted": True, "catalog": "openai", "secret_env": ["OPENAI_API_KEY"],
        "default_model": "gpt-5.5", "background_model": "gpt-5.4-mini",
        "notes": "Responses API (also serves the Responses-only *-pro / *-codex ids)",
    },
    "codex": {
        "display_name": "ChatGPT (codex login)", "dialect": "responses", "target": "chatgpt_codex",
        "base_url": "https://chatgpt.com/backend-api/codex", "auth": {"kind": "codex_chatgpt"},
        "allow_unlisted": True, "max_tool_name": 64,
        "catalog": "codex", "default_model": "gpt-5.5", "background_model": "gpt-5.5",
        "notes": "ChatGPT subscription via ~/.codex/auth.json (file-based credentials); background requests "
                 "run gpt-5.5 at low effort; gpt-6.x use Responses-Lite; 272k windows follow the Codex CLI",
    },
    "gemini": {
        "display_name": "Gemini API", "dialect": "gemini", "target": "gemini_api",
        "base_url": "https://generativelanguage.googleapis.com",
        "auth": {"kind": "api_key", "secret": "gemini", "style": "x-goog-api-key"}, "allow_unlisted": True,
        "schema_mode": "gemini_json", "tool_name_regex": GEMINI_TOOL_NAME_REGEX,
        "catalog": "gemini", "secret_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "default_model": "gemini-3.1-pro-preview", "background_model": "gemini-3.8-flash",
        "notes": "native streamGenerateContent; gemini-2.5-pro is retiring",
    },
    "gemini-vertex": {
        "display_name": "Vertex AI (gcloud ADC)", "dialect": "gemini", "target": "vertex",
        "base_url": "", "auth": {"kind": "gcloud_adc"}, "allow_unlisted": True,
        "schema_mode": "gemini_json", "tool_name_regex": GEMINI_TOOL_NAME_REGEX,
        "options": {"location": None, "project": None},
        "catalog": "gemini", "default_model": "gemini-3.1-pro-preview", "background_model": "gemini-3.8-flash",
        "notes": "gcloud Application Default Credentials; empty base_url => "
                 "https://{aiplatform|<loc>-aiplatform}.googleapis.com from GOOGLE_CLOUD_LOCATION (default global)",
    },
    "deepseek": {
        "display_name": "DeepSeek", "dialect": "anthropic_passthrough", "profile": "deepseek",
        "base_url": "https://api.deepseek.com/anthropic",
        "auth": {"kind": "api_key", "secret": "deepseek", "style": "bearer"}, "allow_unlisted": True,
        "catalog": "deepseek", "secret_env": ["DEEPSEEK_API_KEY"],
        "default_model": "deepseek-v4-pro", "background_model": "deepseek-v4-flash",
        "notes": "Anthropic-compatible endpoint (1M window => [1m] picker ids); max_output 64000 is "
                 "conservative; deepseek-chat/reasoner were retired 2026-07-24",
    },
    "kimi": {
        "display_name": "Kimi (Moonshot)", "dialect": "anthropic_passthrough", "profile": "kimi",
        "base_url": "https://api.moonshot.ai/anthropic",
        "auth": {"kind": "api_key", "secret": "kimi", "style": "bearer"}, "allow_unlisted": True,
        "catalog": "kimi", "secret_env": ["MOONSHOT_API_KEY"],
        "default_model": "kimi-k3", "background_model": "kimi-k2.7-code",
        "notes": "Anthropic-compatible endpoint; max_output 32768 is conservative",
    },
    "openrouter": {
        "display_name": "OpenRouter", "dialect": "anthropic_passthrough", "profile": "openrouter",
        "base_url": "https://openrouter.ai/api",
        "auth": {"kind": "api_key", "secret": "openrouter", "style": "bearer"}, "allow_unlisted": True,
        "catalog": "openrouter", "secret_env": ["OPENROUTER_API_KEY"],
        "default_model": "x-ai/grok-4.7", "background_model": "google/gemini-3.8-flash",
        "notes": "Anthropic-compatible endpoint (Bearer only, no x-api-key); any OpenRouter model id works; "
                 "output limits are enforced by OpenRouter per model",
    },
    "ollama": {
        "display_name": "Ollama (local)", "dialect": "anthropic_passthrough", "profile": "ollama",
        "base_url": "http://localhost:11434", "auth": {"kind": "none"}, "allow_unlisted": True,
        "fallback": {"dialect": "openai_chat", "profile": "ollama", "base_url": "http://localhost:11434/v1"},
        "catalog": "ollama", "default_model": None, "background_model": None,
        "notes": "models discovered from /api/tags (no built-in default); Ollama < 0.14 has no /v1/messages "
                 "and falls back to /v1/chat/completions",
    },
    "opencode-zen": {
        "display_name": "OpenCode Zen", "dialect": "openai_chat", "profile": "opencode",
        "base_url": "https://opencode.ai/zen/v1",
        "auth": {"kind": "api_key", "secret": "opencode-zen", "style": "bearer"}, "allow_unlisted": True,
        "options": {"messages_path": "/messages"},
        "catalog": "opencode-zen", "secret_env": ["OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY"],
        "default_model": "deepseek-v4-flash-free", "background_model": "deepseek-v4-flash-free",
        "notes": "per-model wire format: Claude/MiniMax/Qwen on /messages, GPT on /responses, the rest on "
                 "/chat/completions (unlisted ids use chat); '*-free' models work on the free tier; "
                 + _CONSERVATIVE,
    },
    "opencode-go": {
        "display_name": "OpenCode Go", "dialect": "openai_chat", "profile": "opencode",
        "base_url": "https://opencode.ai/zen/go/v1",
        "auth": {"kind": "api_key", "secret": "opencode-go", "style": "bearer"}, "allow_unlisted": True,
        "options": {"messages_path": "/messages"},
        "catalog": "opencode-go", "secret_env": ["OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"],
        "default_model": "glm-5.1", "background_model": "deepseek-v4-flash",
        "notes": "OpenCode Go subscription; Qwen models on /messages, the rest on /chat/completions; "
                 + _CONSERVATIVE,
    },
    "opencode": {
        "display_name": "OpenCode CLI (chat-only)", "dialect": "cli", "base_url": "", "auth": {"kind": "none"},
        "allow_unlisted": True, "chat_only": True, "options": {"cli": "opencode", "model_prefix": "opencode/"},
        "catalog": "opencode", "default_model": "nemotron-3-ultra-free",
        "background_model": "deepseek-v4-flash-free",
        "notes": "`opencode run --model opencode/<id>` with the transcript on stdin; chat-only (no tool calls); "
                 "128k windows are conservative",
    },
    "nvidia": {
        "display_name": "NVIDIA NIM", "dialect": "openai_chat", "profile": "nvidia",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "auth": {"kind": "api_key", "secret": "nvidia", "style": "bearer"}, "allow_unlisted": True,
        "tool_name_regex": r"^[a-zA-Z0-9_-]{1,64}$",
        "catalog": "nvidia", "secret_env": ["NVIDIA_API_KEY"],
        "default_model": "moonshotai/kimi-k2.5", "background_model": "nvidia/nvidia-nemotron-nano-9b-v2",
        "notes": "OpenAI chat/completions; NIM caps output per deployment, so max_output 16384 and the "
                 "128k Nemotron windows are conservative",
    },
}

_MERGED_KEYS = ("auth", "options", "headers")


def preset_names():
    return list(PRESETS)


def _preset(name):
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError("unknown preset %r (known: %s)" % (name, ", ".join(PRESETS)))


def secret_env_vars(name):
    return list(_preset(name).get("secret_env") or [])


def default_roles(name, provider_id=None):
    p = _preset(name)
    pid = provider_id or name
    roles = {}
    if p.get("default_model"):
        roles["default"] = "%s,%s" % (pid, p["default_model"])
    if p.get("background_model"):
        roles["background"] = "%s,%s" % (pid, p["background_model"])
    return roles


def provider_from_preset(name, overrides=None):
    base = copy.deepcopy(_preset(name))
    ov = copy.deepcopy(overrides or {})
    pid = ov.pop("id", name)
    for k in _MERGED_KEYS:
        if k in ov and isinstance(ov[k], dict):
            merged = dict(base.get(k) or {})
            merged.update(ov.pop(k))
            base[k] = merged
    if "fallback" in ov:
        fb = ov.pop("fallback")
        if fb is None:
            base.pop("fallback", None)
        else:
            merged = dict(base.get("fallback") or {})
            merged.update(fb)
            base["fallback"] = merged
    base.update(ov)
    family = base.get("catalog")
    for k in LAUNCHER_KEYS:
        base.pop(k, None)
    if "models" not in base:
        base["models"] = [m.to_dict() for m in get_catalog(family)] if family else []
    else:
        base["models"] = [m.to_dict() if isinstance(m, ModelSpec) else m for m in base["models"]]
    if base.get("auth", {}).get("kind") == "api_key" and pid != name and base["auth"].get("secret") == name:
        base["auth"]["secret"] = pid
    return ProviderSpec.from_dict(pid, base)


def route_table_from_presets(names, overrides=None, roles=None):
    """``names``: list of preset names (or (name, provider_id) tuples). ``overrides``: provider id ->
    override dict. ``roles``: explicit roles (else the first preset's default/background)."""
    overrides = overrides or {}
    providers = {}
    first_roles = None
    for item in names:
        name, pid = (item, item) if isinstance(item, str) else item
        ov = dict(overrides.get(pid) or {})
        ov.setdefault("id", pid)
        spec = provider_from_preset(name, ov)
        providers[spec.id] = spec
        if first_roles is None:
            first_roles = default_roles(name, spec.id)
    aliases = {k: [a for a in v if a != k] for k, v in DEFAULT_PROVIDER_ALIASES.items()}
    return RouteTable(providers=providers, roles=dict(roles if roles is not None else (first_roles or {})),
                      provider_aliases=aliases)
