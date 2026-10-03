"""ai-gateway: stdlib-only Anthropic Messages API gateway for Claude Code.

Speaks the Anthropic Messages API to Claude Code and translates to provider dialects
(openai_chat, responses, gemini, anthropic_passthrough, cli). See DESIGN.md.

The canonical copy lives in ``ai-launchers/shared/gateway/``; it is vendored byte-identical
into ``fry-launch-claude/fry_gateway/`` by ``tools/vendor_gateway.py``. All intra-package
imports are relative, so the package works under either name.

Public names are resolved lazily (PEP 562 ``__getattr__``) so importing the package is cheap
and does not pull in modules owned by other components until they are used.
"""

__version__ = "1.0.0"

# name -> (submodule, attribute)
_LAZY = {
    "Gateway": ("server", "Gateway"),
    "RouteTable": ("config", "RouteTable"),
    "ProviderSpec": ("config", "ProviderSpec"),
    "ModelSpec": ("config", "ModelSpec"),
    "SecretStore": ("config", "SecretStore"),
    "GatewayError": ("errors", "GatewayError"),
    "presets": ("presets", None),
    "catalog": ("catalog", None),
    "launchkit": ("launchkit", None),
}

__all__ = ["__version__"] + sorted(_LAZY)


def __getattr__(name):
    try:
        mod_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    import importlib

    mod = importlib.import_module("." + mod_name, __name__)
    value = mod if attr is None else getattr(mod, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY))
