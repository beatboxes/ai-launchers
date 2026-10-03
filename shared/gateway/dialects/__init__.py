"""Dialect registry. ``get_dialect(name) -> Dialect`` (lazy import, one cached instance per name).

Dialect instances are stateless (state lives in ``ProviderRuntime``), so one instance serves all
requests concurrently.
"""

import importlib
import threading

from .base import Dialect, LRU, ProviderRuntime, RequestContext, join_url, send_with_auth_retry  # noqa: F401

__all__ = ["get_dialect", "REGISTRY", "Dialect", "ProviderRuntime", "RequestContext", "LRU",
           "send_with_auth_retry", "join_url"]

# dialect name -> (module, class name)
REGISTRY = {
    "openai_chat": ("openai_chat", "OpenAIChatDialect"),
    "responses": ("responses", "ResponsesDialect"),
    "gemini": ("gemini", "GeminiDialect"),
    "anthropic_passthrough": ("anthropic_passthrough", "AnthropicPassthroughDialect"),
    "cli": ("cli", "CliDialect"),
}

_instances = {}
_lock = threading.Lock()


def get_dialect(name):
    with _lock:
        inst = _instances.get(name)
        if inst is not None:
            return inst
        try:
            mod_name, cls_name = REGISTRY[name]
        except KeyError:
            raise ValueError("unknown dialect %r (known: %s)" % (name, ", ".join(sorted(REGISTRY))))
        mod = importlib.import_module("." + mod_name, __name__)
        inst = getattr(mod, cls_name)()
        _instances[name] = inst
        return inst
