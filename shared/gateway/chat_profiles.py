"""Per-provider rules for the ``openai_chat`` dialect (DESIGN §3.1 "Profiles").

``get_profile(name) -> ChatProfile`` (``None`` or an unknown name -> ``generic``). Profiles are
shared, read-only instances; ``profile.with_options(provider.options)`` returns a copy with these
keys overridden from a provider's ``options``: ``max_tokens_field``, ``default_max_tokens``,
``stream_usage``, ``parallel_param``, ``echo_reasoning_content``, ``response_format``, ``vision``,
``pdf_input``, ``tool_choice_required``.

profile     max tokens field       drops / rules                                     effort
xai         max_tokens             reasoning ids: stop, presence/frequency_penalty   reasoning_effort low|high,
                                                                                     only ModelSpec.effort_param
openai      max_completion_tokens  reasoning ids ^(o\\d|gpt-5|gpt-6): temperature,   reasoning_effort
                                   top_p                                             (xhigh/max -> high)
moonshot    max_tokens             temperature, top_p (always); echo reasoning       —
deepseek    max_tokens             reasoner/thinking ids: temperature;               —
                                   echo reasoning
ollama      max_tokens             /api/show caps: no thinking -> no effort;         reasoning_effort
                                   no tools -> chat-only
openrouter  max_tokens             headers HTTP-Referer/X-Title; usage.include       reasoning: {effort}
opencode    max_tokens             —                                                 —
nvidia      max_tokens             no stream_options; strict tool names              —
generic     max_tokens             temperature/top_p only when sent                  —
"""

import copy
import re

from . import catalog

__all__ = ["ChatProfile", "get_profile", "PROFILES", "OVERRIDABLE", "STRICT_TOOL_NAME_REGEX"]

STRICT_TOOL_NAME_REGEX = r"^[a-zA-Z0-9_-]{1,64}$"
OVERRIDABLE = ("max_tokens_field", "default_max_tokens", "stream_usage", "parallel_param", "echo_reasoning_content",
               "response_format", "vision", "pdf_input", "tool_choice_required")
SAMPLING_PARAMS = ("temperature", "top_p", "stop", "presence_penalty", "frequency_penalty")

_EFFORT_STANDARD = {"none": "low", "minimal": "low", "low": "low", "medium": "medium", "high": "high",
                    "xhigh": "high", "max": "high"}
_EFFORT_NO_NONE = dict(_EFFORT_STANDARD, none=None)
_EFFORT_LOW_HIGH = {"none": "low", "minimal": "low", "low": "low", "medium": "high", "high": "high",
                    "xhigh": "high", "max": "high"}
_NOTHING = frozenset()


class ChatProfile(object):
    """Rules for one OpenAI-compatible provider family.

    ``effort_style``: ``None`` (never send effort), ``"reasoning_effort"`` (top-level string) or
    ``"openrouter"`` (``reasoning: {"effort": …}``). ``reasoning_placeholder``: when echoing
    ``reasoning_content``, send ``" "`` on assistant tool-call turns that have no kept thinking
    (thinking-mode Moonshot/DeepSeek reject tool-call history without it, e.g. after a model switch).
    """

    def __init__(self, name, max_tokens_field="max_tokens", default_max_tokens=None, stream_usage=True,
                 parallel_param=False, echo_reasoning_content=False, reasoning_placeholder=False,
                 effort_style=None, effort_map=None, schema_mode="basic", tool_name_regex=None, max_tool_name=64,
                 response_format=False, tool_choice_required=True, vision=True, pdf_input=False, headers=None,
                 extra_body=None, probe_ollama=False, prompt_cache_key=False, session_header=None,
                 default_base_url=None):
        self.name = name
        self.max_tokens_field = max_tokens_field
        self.default_max_tokens = default_max_tokens
        self.stream_usage = stream_usage
        self.parallel_param = parallel_param
        self.echo_reasoning_content = echo_reasoning_content
        self.reasoning_placeholder = reasoning_placeholder
        self.effort_style = effort_style
        self.effort_map = dict(effort_map or _EFFORT_STANDARD)
        self.schema_mode = schema_mode
        self.tool_name_regex = tool_name_regex
        self.max_tool_name = max_tool_name
        self.response_format = response_format
        self.tool_choice_required = tool_choice_required
        self.vision = vision
        self.pdf_input = pdf_input
        self.headers = dict(headers or {})
        self.extra_body = dict(extra_body or {})
        self.probe_ollama = probe_ollama
        self.prompt_cache_key = prompt_cache_key
        self.session_header = session_header
        self.default_base_url = default_base_url

    # ---- per-model rules (overridden by subclasses) -----------------------------------------
    def is_reasoning(self, model_spec):
        return bool(getattr(model_spec, "reasoning", False))

    def dropped_params(self, model_spec):
        """Sampling params (``SAMPLING_PARAMS``) that must not be sent for this model."""
        return _NOTHING

    def supports_effort(self, model_spec):
        return self.effort_style is not None

    def effort_value(self, effort, model_spec):
        """Provider effort value for a normalized ``effort`` (``None`` = send nothing)."""
        if effort is None or not self.supports_effort(model_spec):
            return None
        return self.effort_map.get(effort)

    # ---- helpers ---------------------------------------------------------------------------
    def with_options(self, options):
        """Copy with ``OVERRIDABLE`` keys taken from ``options`` (unchanged self if none apply)."""
        opts = options or {}
        keys = [k for k in OVERRIDABLE if k in opts]
        if not keys:
            return self
        clone = copy.copy(self)
        for k in keys:
            setattr(clone, k, opts[k])
        return clone

    def __repr__(self):
        return "ChatProfile(%r)" % (self.name,)


def _model_id(model_spec):
    return getattr(model_spec, "id", None) or ""


class _XaiProfile(ChatProfile):
    def is_reasoning(self, model_spec):
        return catalog.is_xai_reasoning(_model_id(model_spec), model_spec)

    def dropped_params(self, model_spec):
        if self.is_reasoning(model_spec):
            return frozenset(("stop", "presence_penalty", "frequency_penalty"))
        return _NOTHING

    def supports_effort(self, model_spec):
        return bool(getattr(model_spec, "effort_param", False))


class _OpenAIProfile(ChatProfile):
    def is_reasoning(self, model_spec):
        return catalog.is_openai_reasoning(_model_id(model_spec))

    def dropped_params(self, model_spec):
        return frozenset(("temperature", "top_p")) if self.is_reasoning(model_spec) else _NOTHING

    def supports_effort(self, model_spec):
        return self.is_reasoning(model_spec)


class _MoonshotProfile(ChatProfile):
    def dropped_params(self, model_spec):
        return frozenset(("temperature", "top_p"))


_DEEPSEEK_THINKING = re.compile(r"reasoner|think", re.I)


class _DeepSeekProfile(ChatProfile):
    def is_reasoning(self, model_spec):
        return bool(_DEEPSEEK_THINKING.search(_model_id(model_spec))) or bool(getattr(model_spec, "reasoning", False))

    def dropped_params(self, model_spec):
        return frozenset(("temperature",)) if self.is_reasoning(model_spec) else _NOTHING


OPENROUTER_REFERER = "https://github.com/beatboxes/ai-launchers"
OPENROUTER_TITLE = "ai-launchers gateway"

PROFILES = {
    "xai": _XaiProfile(
        "xai", default_max_tokens=64000, parallel_param=True, effort_style="reasoning_effort",
        effort_map=_EFFORT_LOW_HIGH, schema_mode="no_root_combinators", response_format=True,
        session_header="x-grok-conv-id"),
    "openai": _OpenAIProfile(
        "openai", max_tokens_field="max_completion_tokens", default_max_tokens=128000, parallel_param=True,
        effort_style="reasoning_effort", response_format=True, pdf_input=True, prompt_cache_key=True),
    "moonshot": _MoonshotProfile(
        "moonshot", default_max_tokens=32768, stream_usage=False, echo_reasoning_content=True,
        reasoning_placeholder=True, tool_choice_required=False),
    "deepseek": _DeepSeekProfile(
        "deepseek", default_max_tokens=32768, echo_reasoning_content=True, reasoning_placeholder=True, vision=False),
    "ollama": ChatProfile(
        "ollama", effort_style="reasoning_effort", effort_map=_EFFORT_NO_NONE, response_format=True,
        probe_ollama=True, default_base_url="http://localhost:11434/v1"),
    "openrouter": ChatProfile(
        "openrouter", parallel_param=True, effort_style="openrouter", effort_map=_EFFORT_NO_NONE,
        response_format=True, pdf_input=True,
        headers={"HTTP-Referer": OPENROUTER_REFERER, "X-Title": OPENROUTER_TITLE},
        extra_body={"usage": {"include": True}}),
    "opencode": ChatProfile("opencode"),
    "nvidia": ChatProfile(
        "nvidia", default_max_tokens=16384, stream_usage=False, tool_name_regex=STRICT_TOOL_NAME_REGEX),
    "generic": ChatProfile("generic", default_max_tokens=16384),
}


def get_profile(name):
    """The profile called ``name``; ``None`` or an unknown name -> ``generic``."""
    return PROFILES.get(name or "generic") or PROFILES["generic"]
