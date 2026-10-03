"""Built-in model catalogs per family (DESIGN §1). Owned for CONTENT by agent F; structure frozen.

``CATALOGS``: family -> list of ModelSpec dicts (see ``config.ModelSpec`` fields). Families:
  xai, grok, openai, codex, gemini, deepseek, kimi, openrouter, ollama, opencode-zen, opencode-go,
  opencode, nvidia.

``get_catalog(family) -> List[ModelSpec]`` returns fresh copies (callers may mutate).
``families() -> List[str]``; ``find_model(family, model_id) -> Optional[ModelSpec]``.
Rules for ids not in a catalog (used by profiles/targets): ``is_responses_lite(model_id)``,
``is_openai_reasoning(model_id)``, ``is_xai_reasoning(model_id, spec=None)``.

Sources: plan Appendix A (verified 2026-10-03) for ids, endpoints and wire formats. Context /
max-output numbers come from provider documentation where known; where a provider does not
publish them the values are deliberately conservative (smaller window -> earlier compaction,
smaller output cap -> no "max_tokens too large" 400s) and the owning preset's ``notes`` say so.
``max_output`` is the clamp applied to Claude Code's ``max_tokens`` (it sends 32000), so values
>= 32000 never truncate a normal turn. Retired ids (deepseek-chat/reasoner, o1-mini, codex-mini,
gemini-2.0-flash) must not appear.
"""

import copy
import re

from .config import ModelSpec

__all__ = ["CATALOGS", "get_catalog", "families", "find_model", "is_responses_lite", "is_openai_reasoning",
           "is_xai_reasoning"]

_M = 1000000
_GEMINI_CTX = 1048576
_K256 = 262144
_K128 = 131072


def _with(base, **extra):
    d = dict(base)
    d.update(extra)
    return d


# ---- xAI (api.x.ai, OpenAI chat/completions) ---------------------------------------------------
# reasoning_effort is rejected by the grok-4 reasoning family -> effort_param stays False.
_XAI = [
    {"id": "grok-4.7", "context": 2 * _M, "max_output": 64000, "reasoning": True, "vision": True},
    {"id": "grok-4.6", "context": 2 * _M, "max_output": 64000, "reasoning": True, "vision": True},
    {"id": "grok-4.5", "context": 2 * _M, "max_output": 64000, "reasoning": True, "vision": True},
    {"id": "grok-4.3", "context": 2 * _M, "max_output": 64000, "reasoning": True, "vision": True},
    {"id": "grok-4.20-0309-reasoning", "context": 2 * _M, "max_output": 64000, "reasoning": True, "vision": True},
    {"id": "grok-4.20-0309-non-reasoning", "context": 2 * _M, "max_output": 64000, "reasoning": False,
     "vision": True},
    {"id": "grok-build-0.1", "context": 256000, "max_output": 64000, "reasoning": True},
    {"id": "grok-composer-2.5-fast", "context": 256000, "max_output": 64000, "reasoning": True},
    # Responses-only on api.x.ai:
    {"id": "grok-4.20-multi-agent-0309", "context": 2 * _M, "max_output": 64000, "reasoning": True,
     "dialect_override": "responses", "target_override": "xai_api", "path_override": "/responses"},
]

# Grok Build login (cli-chat-proxy Responses, fallback api.x.ai chat/completions with the same token):
# the chat fallback cannot serve Responses-only ids, so those are not offered on this route.
_GROK_LOGIN = [m for m in _XAI if not m.get("dialect_override")]

# ---- OpenAI API (api.openai.com Responses, openai_api) -----------------------------------------
_GPT5 = {"context": 400000, "max_output": 128000, "reasoning": True, "vision": True}
_GPT41 = {"context": 1047576, "max_output": 32768, "reasoning": False, "vision": True}
_OPENAI = [
    _with(_GPT5, id="gpt-5.5"),
    _with(_GPT5, id="gpt-5.5-pro"),
    _with(_GPT5, id="gpt-5.4"),
    _with(_GPT5, id="gpt-5.4-pro"),
    _with(_GPT5, id="gpt-5.4-mini"),
    _with(_GPT5, id="gpt-5.4-nano"),
    _with(_GPT5, id="gpt-5.3-codex"),
    {"id": "o3-pro", "context": 200000, "max_output": 100000, "reasoning": True, "vision": True},
    _with(_GPT41, id="gpt-4.1"),
    _with(_GPT41, id="gpt-4.1-mini"),
    _with(_GPT41, id="gpt-4.1-nano"),
]

# ---- ChatGPT Codex backend (chatgpt_codex) -----------------------------------------------------
# Windows follow the Codex CLI model families (272k input); gpt-6.x use Responses-Lite.
_CODEX_BASE = {"context": 272000, "max_output": 128000, "reasoning": True, "vision": True}
_CODEX = [
    _with(_CODEX_BASE, id="gpt-5.5"),
    _with(_CODEX_BASE, id="gpt-5.6-sol"),
    _with(_CODEX_BASE, id="gpt-5.6-terra"),
    _with(_CODEX_BASE, id="gpt-5.6-luna"),
    _with(_CODEX_BASE, id="gpt-6-sol", responses_lite=True),
    _with(_CODEX_BASE, id="gpt-6-terra", responses_lite=True),
    _with(_CODEX_BASE, id="gpt-6-luna", responses_lite=True),
    _with(_CODEX_BASE, id="gpt-6.1-sol", responses_lite=True),
    _with(_CODEX_BASE, id="gpt-5.4"),
    _with(_CODEX_BASE, id="gpt-5.4-mini"),
    _with(_CODEX_BASE, id="gpt-5.3-codex"),
]

# ---- Gemini (API key and Vertex ADC share the ids) ---------------------------------------------
_GEMINI_BASE = {"context": _GEMINI_CTX, "max_output": 65536, "reasoning": True, "vision": True}
_GEMINI = [
    _with(_GEMINI_BASE, id="gemini-3.1-pro-preview"),
    _with(_GEMINI_BASE, id="gemini-3.8-flash"),
    _with(_GEMINI_BASE, id="gemini-3.7-flash"),
    _with(_GEMINI_BASE, id="gemini-3.6-flash"),
    _with(_GEMINI_BASE, id="gemini-3.5-flash-lite"),
    _with(_GEMINI_BASE, id="gemini-2.5-pro", display_name="gemini-2.5-pro (retiring)"),
]

# ---- DeepSeek (api.deepseek.com/anthropic; [1m] picker ids => 1M window) -----------------------
_DEEPSEEK = [
    {"id": "deepseek-v4-pro", "context": _M, "max_output": 64000, "reasoning": True},
    {"id": "deepseek-v4-flash", "context": _M, "max_output": 64000, "reasoning": True},
]

# ---- Kimi (api.moonshot.ai/anthropic) ------------------------------------------------------------
_KIMI = [
    {"id": "kimi-k3", "context": _M, "max_output": 32768, "reasoning": True, "vision": True},
    {"id": "kimi-k2.7-code", "context": _K256, "max_output": 32768},
    {"id": "kimi-k2.6", "context": _K256, "max_output": 32768},
    {"id": "kimi-k2.5", "context": _K256, "max_output": 32768, "vision": True},
]

# ---- OpenRouter (openrouter.ai/api Anthropic-compatible; discovery adds the rest) --------------
_OPENROUTER = [
    {"id": "x-ai/grok-4.7", "context": 2 * _M, "reasoning": True, "vision": True},
    {"id": "openai/gpt-5.5", "context": 400000, "reasoning": True, "vision": True},
    {"id": "google/gemini-3.1-pro-preview", "context": _GEMINI_CTX, "reasoning": True, "vision": True},
    {"id": "google/gemini-3.8-flash", "context": _GEMINI_CTX, "reasoning": True, "vision": True},
    {"id": "deepseek/deepseek-v4-pro", "context": _M, "reasoning": True},
    {"id": "moonshotai/kimi-k3", "context": _M, "reasoning": True, "vision": True},
]

# ---- OpenCode Zen / Go: per-model wire format on opencode.ai ------------------------------------
# Claude, MiniMax and Qwen 3.5-3.8 are served on /messages (Anthropic), GPT on /responses, the rest
# on /chat/completions. Free/unpublished limits are conservative (128k window, 32k output).
_MESSAGES = {"dialect_override": "anthropic_passthrough", "path_override": "/messages"}
_RESPONSES = {"dialect_override": "responses", "target_override": "openai_api", "path_override": "/responses"}
_FREE = {"context": 128000, "max_output": 32000}
_OPENCODE_ZEN = [
    _with(_FREE, id="deepseek-v4-flash-free", reasoning=True),
    _with(_FREE, id="mimo-v2.5-free"),
    _with(_FREE, id="hy3-free"),
    _with(_FREE, id="nemotron-3-ultra-free", reasoning=True),
    _with(_FREE, id="north-mini-code-free"),
    _with(_FREE, id="big-pickle"),
    {"id": "glm-5.1", "context": 200000, "max_output": 32000, "reasoning": True},
    {"id": "glm-5", "context": 200000, "max_output": 32000, "reasoning": True},
    {"id": "kimi-k2.5", "context": _K256, "max_output": 32768, "vision": True},
    {"id": "kimi-k2.6", "context": _K256, "max_output": 32768},
    _with(_MESSAGES, id="qwen3.6-plus", context=_K256, max_output=32768, reasoning=True),
    _with(_MESSAGES, id="qwen3.5-plus", context=_K256, max_output=32768, reasoning=True),
    _with(_MESSAGES, id="minimax-m2.7", context=200000, max_output=32768, reasoning=True),
    _with(_MESSAGES, id="claude-sonnet-4-5", context=200000, max_output=64000, reasoning=True, vision=True),
    _with(_MESSAGES, id="claude-haiku-4-5", context=200000, max_output=64000, reasoning=True, vision=True),
    _with(_RESPONSES, id="gpt-5.5", context=400000, max_output=128000, reasoning=True, vision=True),
]

_OPENCODE_GO = [
    {"id": "glm-5.1", "context": 200000, "max_output": 32000, "reasoning": True},
    {"id": "glm-5", "context": 200000, "max_output": 32000, "reasoning": True},
    {"id": "kimi-k2.5", "context": _K256, "max_output": 32768, "vision": True},
    {"id": "kimi-k2.6", "context": _K256, "max_output": 32768},
    {"id": "kimi-k3", "context": _M, "max_output": 32768, "reasoning": True, "vision": True},
    {"id": "deepseek-v4-pro", "context": _M, "max_output": 64000, "reasoning": True},
    {"id": "deepseek-v4-flash", "context": _M, "max_output": 64000, "reasoning": True},
    {"id": "mimo-v2-pro", "context": _K128, "max_output": 32000, "reasoning": True},
    {"id": "mimo-v2.5-pro", "context": _K128, "max_output": 32000, "reasoning": True},
    {"id": "mimo-v2.5", "context": _K128, "max_output": 32000, "reasoning": True},
    _with(_MESSAGES, id="qwen3.6-plus", context=_K256, max_output=32768, reasoning=True),
    _with(_MESSAGES, id="qwen3.5-plus", context=_K256, max_output=32768, reasoning=True),
]

# ---- OpenCode CLI (`opencode run --model opencode/<id>`, chat-only) ------------------------------
_CLI = {"context": 128000, "tools": False}
_OPENCODE_CLI = [
    _with(_CLI, id="nemotron-3-ultra-free"),
    _with(_CLI, id="mimo-v2.5-free"),
    _with(_CLI, id="north-mini-code-free"),
    _with(_CLI, id="deepseek-v4-flash-free"),
]

# ---- NVIDIA NIM (integrate.api.nvidia.com/v1, OpenAI chat) ---------------------------------------
# NIM caps completion length per deployment; 16384 is the conservative common ceiling.
_NIM = {"context": _K128, "max_output": 16384}
_NVIDIA = [
    {"id": "moonshotai/kimi-k2.5", "context": _K256, "max_output": 16384, "vision": True},
    {"id": "qwen/qwen3-coder-480b-a35b-instruct", "context": _K256, "max_output": 16384},
    _with(_NIM, id="nvidia/llama-3.3-nemotron-super-49b-v1.5", reasoning=True),
    _with(_NIM, id="nvidia/llama-3.1-nemotron-ultra-253b-v1", reasoning=True),
    _with(_NIM, id="nvidia/nemotron-3-nano-30b-a3b", reasoning=True),
    _with(_NIM, id="nvidia/nvidia-nemotron-nano-9b-v2", reasoning=True),
]

CATALOGS = {
    "xai": _XAI,
    "grok": _GROK_LOGIN,
    "openai": _OPENAI,
    "codex": _CODEX,
    "gemini": _GEMINI,
    "deepseek": _DEEPSEEK,
    "kimi": _KIMI,
    "openrouter": _OPENROUTER,
    "ollama": [],
    "opencode-zen": _OPENCODE_ZEN,
    "opencode-go": _OPENCODE_GO,
    "opencode": _OPENCODE_CLI,
    "nvidia": _NVIDIA,
}


def families():
    return list(CATALOGS)


def get_catalog(family):
    """Fresh ``ModelSpec`` list for ``family`` (unknown family -> ``[]``)."""
    return [ModelSpec.from_dict(copy.deepcopy(d)) for d in CATALOGS.get(family, [])]


def find_model(family, model_id):
    for m in get_catalog(family):
        if m.id == model_id:
            return m
    return None


def is_responses_lite(model_id):
    """ChatGPT Codex Responses-Lite models (gpt-6.x)."""
    return bool(re.match(r"^gpt-6(?:[.-]|$)", model_id or ""))


def is_openai_reasoning(model_id):
    """OpenAI reasoning ids reject temperature/top_p: ``^(o\\d|gpt-5|gpt-6)``."""
    return bool(re.match(r"^(o\d|gpt-5|gpt-6)", model_id or ""))


def is_xai_reasoning(model_id, spec=None):
    """xAI: catalog ``reasoning`` flag wins; otherwise ids without ``non-reasoning`` are reasoning."""
    if spec is not None and spec.id == model_id and (spec.context is not None or spec.reasoning):
        return bool(spec.reasoning)
    return "non-reasoning" not in (model_id or "")
