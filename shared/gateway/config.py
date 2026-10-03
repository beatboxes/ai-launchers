"""Route configuration: ModelSpec, ProviderSpec, RouteTable (JSON-serializable, never holds secrets)
and SecretStore (in-memory only, redacted repr, refuses serialization). DESIGN §5.1.

JSON shape (``RouteTable.to_dict()``)::

    {"version": 1, "picker_prefix": "claude-via-", "long_context_threshold": 60000,
     "providers": {"<id>": {<ProviderSpec fields>, "models": [<ModelSpec dicts>],
                            "fallback": {<partial ProviderSpec; inherits display_name, auth, models,
                                          allow_unlisted, chat_only from the parent>}}},
     "roles": {"default": "<provider>,<model>", "background": ..., "longContext": ..., "subagent": ...},
     "aliases": {"<alias>": "<provider>,<model>"},
     "provider_aliases": {"<provider>": ["<alternative provider>", ...]}}

``AI_GATEWAY_UPSTREAM_<ID>`` (``<ID>`` = provider id upper-cased, non-alphanumerics -> ``_``)
replaces a provider's ``base_url``; ``AI_GATEWAY_UPSTREAM_<ID>_FALLBACK`` replaces its fallback's
``base_url``. Apply with ``apply_upstream_env_overrides(table, environ)``.
"""

import copy
import dataclasses
import re
import threading
from typing import Any, Dict, List, Optional

__all__ = [
    "ConfigError", "ModelSpec", "ProviderSpec", "RouteTable", "SecretStore",
    "DIALECTS", "DIALECT_TARGETS", "AUTH_KINDS", "AUTH_STYLES", "SCHEMA_MODES", "ROLE_NAMES",
    "split_route", "upstream_env_name", "apply_upstream_env_overrides", "describe_upstream_env_overrides",
]

DIALECTS = ("openai_chat", "responses", "gemini", "anthropic_passthrough", "cli")
DIALECT_TARGETS = {
    "responses": ("openai_api", "chatgpt_codex", "grok_cli_proxy", "xai_api"),
    "gemini": ("gemini_api", "vertex"),
}
AUTH_KINDS = ("api_key", "none", "codex_chatgpt", "grok_cli", "gcloud_adc")
AUTH_STYLES = ("bearer", "x-api-key", "x-goog-api-key", "none")
SCHEMA_MODES = ("none", "basic", "no_root_combinators", "gemini_json", "gemini_openapi")
ROLE_NAMES = ("default", "background", "longContext", "subagent", "think", "webSearch")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigError(ValueError):
    """Invalid route configuration. ``problems`` lists every issue found."""

    def __init__(self, problems):
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        ValueError.__init__(self, "; ".join(self.problems))


def split_route(value):
    """``"provider,model"`` -> ``("provider", "model")``; no comma -> ``(None, value)``.

    Splits on the FIRST comma only; both parts stripped.
    """
    if value is None:
        return None, None
    if "," not in value:
        return None, value.strip()
    p, m = value.split(",", 1)
    return p.strip(), m.strip()


# ---------------------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass
class ModelSpec:
    """Static facts about one upstream model id.

    ``effort_param``: the model accepts the provider profile's effort parameter
    (xai ``reasoning_effort``). ``dialect_override``/``target_override``/``path_override``: per-model
    wire format (e.g. OpenCode Zen serves some ids on ``/messages`` or ``/responses``).
    ``responses_lite``: ChatGPT Codex Responses-Lite rules (gpt-6.x).
    """

    id: str
    display_name: Optional[str] = None
    context: Optional[int] = None
    max_output: Optional[int] = None
    reasoning: bool = False
    tools: bool = True
    vision: bool = False
    effort_param: bool = False
    dialect_override: Optional[str] = None
    target_override: Optional[str] = None
    path_override: Optional[str] = None
    responses_lite: bool = False
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


    @classmethod
    def from_dict(cls, d):
        if isinstance(d, str):
            return cls(id=d)
        if not isinstance(d, dict) or not d.get("id"):
            raise ConfigError("model entry needs an 'id': %r" % (d,))
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {}
        extra = dict(d.get("extra") or {})
        for k, v in d.items():
            if k == "extra":
                continue
            if k in known:
                kwargs[k] = v
            elif not k.startswith("_"):
                extra[k] = v
        kwargs["extra"] = extra
        for k in ("context", "max_output"):
            if kwargs.get(k) is not None:
                kwargs[k] = int(kwargs[k])
        return cls(**kwargs)

    def to_dict(self):
        out = {"id": self.id}
        for f in dataclasses.fields(self):
            if f.name == "id":
                continue
            v = getattr(self, f.name)
            default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
            if v != default:
                out[f.name] = copy.deepcopy(v)
        return out

    def label(self):
        return self.display_name or self.id


# ---------------------------------------------------------------------------------------
# ProviderSpec
# ---------------------------------------------------------------------------------------

_PROVIDER_INHERITED = ("display_name", "auth", "models", "allow_unlisted", "chat_only")


@dataclasses.dataclass
class ProviderSpec:
    """One upstream provider.

    ``auth`` is a plain dict: ``{"kind": one of AUTH_KINDS, "secret": "<SecretStore name>" (api_key),
    "style": one of AUTH_STYLES (api_key; default bearer), ...kind-specific options}``.
    ``options``: dialect/profile options (e.g. ``messages_path``, ``key_header``, ``drop_betas``,
    ``project``/``location`` for vertex, ``cli_bin`` for cli).
    """

    id: str
    display_name: str = ""
    dialect: str = "openai_chat"
    target: Optional[str] = None
    profile: Optional[str] = None
    base_url: str = ""
    auth: Dict[str, Any] = dataclasses.field(default_factory=lambda: {"kind": "none"})
    headers: Dict[str, str] = dataclasses.field(default_factory=dict)
    allow_unlisted: bool = False
    chat_only: bool = False
    options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    models: List[ModelSpec] = dataclasses.field(default_factory=list)
    fallback: Optional["ProviderSpec"] = None
    schema_mode: Optional[str] = None
    tool_name_regex: Optional[str] = None
    max_tool_name: Optional[int] = None
    unknown_keys: List[str] = dataclasses.field(default_factory=list, compare=False, repr=False)

    # ---- (de)serialization ---------------------------------------------------------------
    @classmethod
    def from_dict(cls, provider_id, d, parent=None):
        if not isinstance(d, dict):
            raise ConfigError("provider %r must be an object" % (provider_id,))
        known = {f.name for f in dataclasses.fields(cls)} - {"id", "unknown_keys"}
        kwargs = {}
        if parent is not None:
            for k in _PROVIDER_INHERITED:
                kwargs[k] = copy.deepcopy(getattr(parent, k))
        unknown = []
        fb = None
        for k, v in d.items():
            if k == "fallback":
                fb = v
            elif k == "models":
                kwargs["models"] = [ModelSpec.from_dict(m) for m in (v or [])]
            elif k in known:
                kwargs[k] = copy.deepcopy(v)
            elif k == "id":
                continue
            elif not k.startswith("_"):
                unknown.append(k)
        if kwargs.get("max_tool_name") is not None:
            kwargs["max_tool_name"] = int(kwargs["max_tool_name"])
        spec = cls(id=provider_id, unknown_keys=unknown, **kwargs)
        if fb is not None:
            if parent is not None:
                raise ConfigError("provider %r: nested fallback is not supported" % (provider_id,))
            spec.fallback = cls.from_dict(provider_id, fb, parent=spec)
        return spec

    def to_dict(self, parent=None):
        out = {}
        for f in dataclasses.fields(self):
            if f.name in ("id", "unknown_keys", "fallback"):
                continue
            v = getattr(self, f.name)
            if parent is not None and f.name in _PROVIDER_INHERITED:
                if v == getattr(parent, f.name):
                    continue
                if f.name == "models":
                    out["models"] = [m.to_dict() for m in v]
                else:
                    out[f.name] = copy.deepcopy(v)
                continue
            if f.name == "models":
                out["models"] = [m.to_dict() for m in v]
                continue
            default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
            if f.name in ("dialect", "base_url", "auth") or v != default:
                out[f.name] = copy.deepcopy(v)
        if self.fallback is not None:
            out["fallback"] = self.fallback.to_dict(parent=self)
        return out

    # ---- queries ---------------------------------------------------------------------------
    def find_model(self, model_id):
        """Catalog entry for ``model_id`` (exact, then case-insensitive), or None."""
        for m in self.models:
            if m.id == model_id:
                return m
        low = (model_id or "").lower()
        for m in self.models:
            if m.id.lower() == low:
                return m
        return None

    def lists_model(self, model_id):
        return self.find_model(model_id) is not None

    def model_spec(self, model_id):
        """Catalog entry, or a default ``ModelSpec(id=model_id)`` for unlisted ids."""
        return self.find_model(model_id) or ModelSpec(id=model_id)

    def effective_dialect(self, model_spec=None):
        return (model_spec.dialect_override if model_spec and model_spec.dialect_override else self.dialect)

    def effective_target(self, model_spec=None):
        if model_spec and model_spec.target_override:
            return model_spec.target_override
        return self.target

    def label(self):
        return self.display_name or self.id

    # ---- validation ------------------------------------------------------------------------
    def problems(self, where=None):
        where = where or ("provider %r" % self.id)
        out = []
        if not self.id or not _ID_RE.match(self.id):
            out.append("%s: invalid id (letters, digits, '.', '_', '-'; no commas)" % where)
        for k in self.unknown_keys:
            out.append("%s: unknown key %r" % (where, k))
        if self.dialect not in DIALECTS:
            out.append("%s: unknown dialect %r (expected one of %s)" % (where, self.dialect, ", ".join(DIALECTS)))
        targets = DIALECT_TARGETS.get(self.dialect)
        if targets is not None:
            if self.target not in targets:
                out.append("%s: dialect %s needs target in %s (got %r)" % (where, self.dialect, targets, self.target))
        elif self.target is not None:
            out.append("%s: dialect %s takes no target (got %r)" % (where, self.dialect, self.target))
        needs_url = self.dialect not in ("cli",) and not (self.dialect == "gemini" and self.target == "vertex")
        if needs_url and not self.base_url:
            out.append("%s: base_url required" % where)
        if self.base_url and not re.match(r"^https?://[^\s/]+", self.base_url):
            out.append("%s: base_url must be http(s)://… (got %r)" % (where, self.base_url))
        if not isinstance(self.auth, dict):
            out.append("%s: auth must be an object" % where)
        else:
            kind = self.auth.get("kind")
            if kind not in AUTH_KINDS:
                out.append("%s: unknown auth kind %r" % (where, kind))
            if kind == "api_key" and not self.auth.get("secret"):
                out.append("%s: auth kind api_key needs 'secret' (a SecretStore name)" % where)
            style = self.auth.get("style", "bearer")
            if style not in AUTH_STYLES:
                out.append("%s: unknown auth style %r" % (where, style))
            for k, v in self.auth.items():
                if k in ("key", "token", "api_key", "value", "password"):
                    out.append("%s: auth.%s looks like an inline secret; use a SecretStore name" % (where, k))
        if self.schema_mode is not None and self.schema_mode not in SCHEMA_MODES:
            out.append("%s: unknown schema_mode %r" % (where, self.schema_mode))
        if self.tool_name_regex is not None:
            try:
                re.compile(self.tool_name_regex)
            except re.error as exc:
                out.append("%s: bad tool_name_regex: %s" % (where, exc))
        if self.max_tool_name is not None and self.max_tool_name < 16:
            out.append("%s: max_tool_name must be >= 16" % where)
        seen = set()
        for m in self.models:
            if not m.id or "," in m.id:
                out.append("%s: invalid model id %r" % (where, m.id))
            if m.id in seen:
                out.append("%s: duplicate model id %r" % (where, m.id))
            seen.add(m.id)
            if m.dialect_override is not None and m.dialect_override not in DIALECTS:
                out.append("%s: model %r: unknown dialect_override %r" % (where, m.id, m.dialect_override))
            d = m.dialect_override or self.dialect
            t = m.target_override or (self.target if d == self.dialect else None)
            if d in DIALECT_TARGETS and t not in DIALECT_TARGETS[d]:
                out.append("%s: model %r: dialect %s needs a valid target_override" % (where, m.id, d))
        if self.fallback is not None:
            out.extend(self.fallback.problems(where + " fallback"))
        return out


# ---------------------------------------------------------------------------------------
# RouteTable
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass
class RouteTable:
    version: int = 1
    picker_prefix: str = "claude-via-"
    long_context_threshold: int = 60000
    providers: Dict[str, ProviderSpec] = dataclasses.field(default_factory=dict)   # ordered
    roles: Dict[str, Optional[str]] = dataclasses.field(default_factory=dict)
    aliases: Dict[str, str] = dataclasses.field(default_factory=dict)
    provider_aliases: Dict[str, List[str]] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise ConfigError("route table must be an object")
        providers = {}
        for pid, pd in (d.get("providers") or {}).items():
            if pid.startswith("_"):
                continue
            providers[pid] = ProviderSpec.from_dict(pid, pd)
        roles = {k: v for k, v in (d.get("roles") or {}).items() if not k.startswith("_")}
        return cls(
            version=int(d.get("version", 1)),
            picker_prefix=d.get("picker_prefix", "claude-via-"),
            long_context_threshold=int(d.get("long_context_threshold", 60000)),
            providers=providers,
            roles=roles,
            aliases={k: v for k, v in (d.get("aliases") or {}).items() if not k.startswith("_")},
            provider_aliases={k: list(v) for k, v in (d.get("provider_aliases") or {}).items()
                              if not k.startswith("_")},
        )

    def to_dict(self):
        return {
            "version": self.version,
            "picker_prefix": self.picker_prefix,
            "long_context_threshold": self.long_context_threshold,
            "providers": {pid: p.to_dict() for pid, p in self.providers.items()},
            "roles": dict(self.roles),
            "aliases": dict(self.aliases),
            "provider_aliases": {k: list(v) for k, v in self.provider_aliases.items()},
        }

    def copy(self):
        return copy.deepcopy(self)

    # ---- queries -----------------------------------------------------------------------------
    def provider(self, provider_id):
        return self.providers.get(provider_id)

    def canonical_provider(self, provider_id):
        """``provider_id`` if configured, else the first configured entry of its provider_aliases."""
        if provider_id in self.providers:
            return provider_id
        for alt in self.provider_aliases.get(provider_id, []):
            if alt in self.providers:
                return alt
        return None

    def picker_id(self, provider_id, model):
        """``<picker_prefix><provider>,<model>`` + ``[1m]`` when the model's context >= 1,000,000."""
        spec = model if isinstance(model, ModelSpec) else None
        if spec is None:
            p = self.providers.get(provider_id)
            spec = p.model_spec(model) if p else ModelSpec(id=model)
        suffix = "[1m]" if (spec.context or 0) >= 1000000 else ""
        return "%s%s,%s%s" % (self.picker_prefix, provider_id, spec.id, suffix)

    # ---- validation ----------------------------------------------------------------------------
    def problems(self):
        out = []
        if self.version != 1:
            out.append("unsupported route table version %r" % (self.version,))
        if not self.picker_prefix or not re.search(r"claude|anthropic", self.picker_prefix, re.I):
            out.append("picker_prefix must contain 'claude' or 'anthropic' (Claude Code filters discovery ids)")
        if self.long_context_threshold < 0:
            out.append("long_context_threshold must be >= 0")
        for pid, p in self.providers.items():
            if p.id != pid:
                out.append("provider key %r != id %r" % (pid, p.id))
            out.extend(p.problems())
        for role, target in self.roles.items():
            if role not in ROLE_NAMES:
                out.append("unknown role %r (expected one of %s)" % (role, ", ".join(ROLE_NAMES)))
            if target is None:
                continue
            out.extend(self._target_problems("role %r" % role, target))
        for alias, target in self.aliases.items():
            if "," in alias:
                out.append("alias %r must not contain ','" % alias)
            out.extend(self._target_problems("alias %r" % alias, target))
        for pid, alts in self.provider_aliases.items():
            if not isinstance(alts, list) or not all(isinstance(a, str) for a in alts):
                out.append("provider_aliases[%r] must be a list of provider ids" % pid)
        if self.roles.get("default") is None and self.providers:
            out.append("roles.default is required")
        return out

    def _target_problems(self, where, target):
        if not isinstance(target, str) or not target.strip():
            return ["%s: target must be a non-empty string" % where]
        prov, model = split_route(target)
        if prov is None:
            return [] if model in self.aliases else ["%s: %r is not 'provider,model' or a known alias" % (where, target)]
        if not model:
            return ["%s: %r has an empty model" % (where, target)]
        cp = self.canonical_provider(prov)
        if cp is None:
            return ["%s: provider %r is not configured" % (where, prov)]
        p = self.providers[cp]
        if not p.allow_unlisted and not p.lists_model(model):
            return ["%s: model %r is not listed by provider %r (and allow_unlisted is false)" % (where, model, cp)]
        return []

    def validate(self):
        """Raise ``ConfigError`` listing every problem; return self when valid."""
        problems = self.problems()
        if problems:
            raise ConfigError(problems)
        return self


# ---------------------------------------------------------------------------------------
# env overrides (tests point providers at local mocks)
# ---------------------------------------------------------------------------------------

def upstream_env_name(provider_id, fallback=False):
    name = "AI_GATEWAY_UPSTREAM_" + re.sub(r"[^A-Z0-9]", "_", provider_id.upper())
    return name + "_FALLBACK" if fallback else name


def _override_pairs(table, environ):
    for pid, p in table.providers.items():
        v = environ.get(upstream_env_name(pid))
        if v:
            yield pid, False, v
        if p.fallback is not None:
            v = environ.get(upstream_env_name(pid, fallback=True))
            if v:
                yield pid, True, v


def apply_upstream_env_overrides(table, environ=None):
    """Return a COPY of ``table`` with ``AI_GATEWAY_UPSTREAM_<ID>[_FALLBACK]`` base_url overrides applied."""
    import os

    environ = os.environ if environ is None else environ
    out = table.copy()
    for pid, is_fb, url in _override_pairs(out, environ):
        target = out.providers[pid].fallback if is_fb else out.providers[pid]
        target.base_url = url.rstrip("/")
    return out


def describe_upstream_env_overrides(table, environ=None):
    """Human-readable list of the overrides ``apply_upstream_env_overrides`` would apply."""
    import os

    environ = os.environ if environ is None else environ
    return ["%s%s base_url <- %s=%s" % (pid, " fallback" if fb else "", upstream_env_name(pid, fb), url)
            for pid, fb, url in _override_pairs(table, environ)]


# ---------------------------------------------------------------------------------------
# SecretStore
# ---------------------------------------------------------------------------------------

class SecretStore(object):
    """In-memory secret map (name -> value). Never serializes, never prints values.

    ``repr``/``str`` show only the names. Pickling raises ``TypeError``; ``json.dumps`` fails
    naturally. ``copy``/``deepcopy`` produce an in-memory copy.
    """

    __slots__ = ("_data", "_lock")

    def __init__(self, initial=None):
        self._data = {}
        self._lock = threading.Lock()
        for k, v in (initial or {}).items():
            self.set(k, v)

    def get(self, name, default=None):
        with self._lock:
            return self._data.get(name, default)

    def set(self, name, value):
        if not isinstance(name, str) or not name:
            raise ValueError("secret name must be a non-empty string")
        with self._lock:
            if value is None or value == "":
                self._data.pop(name, None)
            else:
                self._data[name] = str(value)

    def delete(self, name):
        with self._lock:
            return self._data.pop(name, None) is not None

    def has(self, name):
        with self._lock:
            return bool(self._data.get(name))

    def names(self):
        with self._lock:
            return sorted(self._data)

    def values_for_redaction(self):
        """All secret values (for log/trace redaction only)."""
        with self._lock:
            return [v for v in self._data.values() if v]

    def redact(self, text, replacement="***"):
        """Replace every occurrence of any stored secret (len >= 4) in ``text``."""
        if not text:
            return text
        for v in sorted(self.values_for_redaction(), key=len, reverse=True):
            if len(v) >= 4 and v in text:
                text = text.replace(v, replacement)
        return text

    def __contains__(self, name):
        return self.has(name)

    def __len__(self):
        with self._lock:
            return len(self._data)

    def __repr__(self):
        return "SecretStore(names=%r, values=<redacted>)" % (self.names(),)

    __str__ = __repr__

    def __reduce_ex__(self, protocol):
        raise TypeError("SecretStore is not serializable")

    def __getstate__(self):
        raise TypeError("SecretStore is not serializable")

    def __copy__(self):
        with self._lock:
            return SecretStore(dict(self._data))

    def __deepcopy__(self, memo):
        return self.__copy__()
