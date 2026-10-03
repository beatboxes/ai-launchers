"""Static API-key auth and no-auth (DESIGN §4)."""

from .base import AuthError, AuthProvider

__all__ = ["StaticKeyAuth", "NoAuth", "STYLES"]

STYLES = ("bearer", "x-api-key", "x-goog-api-key", "none")


class StaticKeyAuth(AuthProvider):
    """Key read from a ``SecretStore`` entry at request time (so key changes apply immediately).

    style: ``bearer`` -> ``Authorization: Bearer <k>``; ``x-api-key`` -> ``x-api-key: <k>``;
    ``x-goog-api-key`` -> ``x-goog-api-key: <k>``; ``none`` -> no header.
    """

    kind = "api_key"

    def __init__(self, secret_name, store, style="bearer", provider_id=""):
        AuthProvider.__init__(self, provider_id)
        if style not in STYLES:
            raise ValueError("unknown auth style %r" % (style,))
        self.secret_name = secret_name
        self.store = store
        self.style = style

    def available(self):
        return self.style == "none" or bool(self.store is not None and self.store.has(self.secret_name))

    def _key(self):
        key = self.store.get(self.secret_name) if self.store is not None else None
        if not key:
            raise AuthError("no API key for provider %r (secret %r is not set)" % (self.provider_id, self.secret_name),
                            "set the provider's API key (e.g. `<launcher> keys set`)")
        return key

    def headers(self, force_refresh=False):
        if self.style == "none":
            return {}
        key = self._key()
        if self.style == "bearer":
            return {"Authorization": "Bearer " + key}
        if self.style == "x-api-key":
            return {"x-api-key": key}
        return {"x-goog-api-key": key}

    def on_unauthorized(self, used):
        # A static key cannot be refreshed; but if the store changed since `used`, retry once.
        try:
            return self.headers() != dict(used or {})
        except AuthError:
            return False

    def relogin_hint(self):
        return "check the API key for %s" % (self.provider_id or "this provider")

    def describe(self):
        return {"kind": self.kind, "provider": self.provider_id, "source": "secret:%s" % self.secret_name,
                "style": self.style, "available": self.available()}


class NoAuth(AuthProvider):
    kind = "none"

    def available(self):
        return True

    def headers(self, force_refresh=False):
        return {}

    def describe(self):
        return {"kind": self.kind, "provider": self.provider_id, "source": "none", "available": True}
