"""Auth providers. ``make_auth(auth_dict, secrets, provider_id) -> AuthProvider``.

Kinds (DESIGN §4):
  api_key        -> static.StaticKeyAuth(auth["secret"], secrets, auth.get("style", "bearer"))
  none           -> static.NoAuth()
  codex_chatgpt  -> codex_chatgpt.CodexChatGPTAuth(provider_id=..., options=auth)
  grok_cli       -> grok_cli.GrokCliAuth(provider_id=..., options=auth)
  gcloud_adc     -> gcloud_adc.GcloudADCAuth(provider_id=..., options=auth)

The non-static modules are imported lazily (owned by another component). The gateway creates ONE
AuthProvider per provider id and reuses it for that provider's ``fallback`` spec when the fallback's
auth dict equals the parent's (so a single lock serializes refreshes).
"""

from .base import AuthError, AuthProvider, RefreshingAuth, Token, redact_headers  # noqa: F401

__all__ = ["make_auth", "AuthError", "AuthProvider", "RefreshingAuth", "Token", "redact_headers", "AUTH_CLASSES"]

# kind -> (module, class name)
AUTH_CLASSES = {
    "codex_chatgpt": ("codex_chatgpt", "CodexChatGPTAuth"),
    "grok_cli": ("grok_cli", "GrokCliAuth"),
    "gcloud_adc": ("gcloud_adc", "GcloudADCAuth"),
}


def make_auth(auth_dict, secrets, provider_id):
    auth_dict = dict(auth_dict or {"kind": "none"})
    kind = auth_dict.get("kind", "none")
    if kind == "api_key":
        from .static import StaticKeyAuth

        return StaticKeyAuth(auth_dict.get("secret") or provider_id, secrets, auth_dict.get("style", "bearer"),
                             provider_id=provider_id)
    if kind == "none":
        from .static import NoAuth

        return NoAuth(provider_id)
    if kind in AUTH_CLASSES:
        import importlib

        mod_name, cls_name = AUTH_CLASSES[kind]
        mod = importlib.import_module("." + mod_name, __name__)
        return getattr(mod, cls_name)(provider_id=provider_id, options=auth_dict)
    raise ValueError("unknown auth kind %r for provider %r" % (kind, provider_id))
