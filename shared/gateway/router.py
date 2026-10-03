"""Model routing (DESIGN §5.2/§5.3): Claude Code model string -> provider + upstream model.

``Router(table, discovered=None, launcher_name=None)``
    ``resolve(requested, est_tokens=None) -> Resolution`` (raises ``RouteNotFound``)
    ``set_discovered(provider_id, model_ids)``  model ids found by live discovery (thread-safe)
    ``models_response() -> dict``               ``GET /v1/models`` body
    ``model_entry(model_id) -> Optional[dict]`` ``GET /v1/models/{id}`` entry

Resolution rules, in order:
1. strip the case-insensitive picker prefix (``claude-via-``) and a trailing ``[1m]``;
2. role aliases ``default|background|longcontext|subagent`` (+ ``think``/``websearch``, accepted and
   treated as ``default``) -> that role's target (unset roles fall back to ``default``);
3. ``provider,model``: legacy fry forms first (``ollama,fry-grok-<x>`` via the alias table or
   ``fry-grok-4-3`` -> ``grok,grok-4.3``; ``ollama,fry-codex-<m>`` -> ``codex,<m>``;
   ``ollama,fry-opencode-<m>`` -> ``opencode,<m>``; ``local-ollama,<m>`` -> ``ollama,<m>``), then the
   provider is canonicalized through ``provider_aliases``; the model must be in the catalog, the
   discovered list, or the provider must ``allow_unlisted``;
4. no comma: exact alias; bare catalog/discovered id (first provider in table order); Claude family
   names (``/haiku/i`` -> background; ``claude*``/``anthropic*``/``opus|sonnet|fable|default`` ->
   default);
5. a request that resolved to the default route switches to ``roles.longContext`` when set and
   ``est_tokens > long_context_threshold``;
6. otherwise ``RouteNotFound`` (404 ``not_found_error``).
"""

import logging
import re
import threading

from .config import ModelSpec, split_route
from .errors import GatewayError

__all__ = ["Resolution", "RouteNotFound", "Router", "MODELS_CREATED_AT", "MAX_MODELS_PER_PROVIDER"]

LOG = logging.getLogger("ai_gateway")

MODELS_CREATED_AT = "2026-01-01T00:00:00Z"
MAX_MODELS_PER_PROVIDER = 50

#: lower-cased role alias -> RouteTable role name
_ROLE_ALIASES = {"default": "default", "background": "background", "longcontext": "longContext",
                 "subagent": "subagent", "think": "think", "websearch": "webSearch"}
_IGNORED_ROLES = ("think", "webSearch")
_CLAUDE_FAMILY = ("opus", "sonnet", "fable", "default")
_SUFFIX_1M = "[1m]"
_LEGACY_GROK_VERSION = re.compile(r"^(\d+)-(\d+)(-.+)?$")
_AUTH_SOURCES = {"api_key": "API key", "codex_chatgpt": "ChatGPT login", "grok_cli": "Grok login",
                 "gcloud_adc": "gcloud ADC", "none": "no key"}


class Resolution(object):
    """Where one request goes. ``provider`` is always the primary spec (never the fallback)."""

    __slots__ = ("provider", "model", "model_spec", "requested", "role", "background")

    def __init__(self, provider, model, model_spec, requested, role=None, background=False):
        self.provider = provider
        self.model = model
        self.model_spec = model_spec
        self.requested = requested
        self.role = role
        self.background = bool(background)

    def route(self):
        """``"<provider>,<model>"``."""
        return "%s,%s" % (self.provider.id, self.model)

    def __eq__(self, other):
        return isinstance(other, Resolution) and all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "Resolution(%s, requested=%r, role=%r, background=%r)" % (
            self.route(), self.requested, self.role, self.background)


class RouteNotFound(GatewayError):
    """404 ``not_found_error`` for a model string no route matches."""

    def __init__(self, requested, available=(), launcher_name=None):
        available = list(available)
        shown = ", ".join(available[:5]) + ("…" if len(available) > 5 else "")
        hint = ("run `%s models`" % launcher_name) if launcher_name else "run the launcher's `models` command"
        GatewayError.__init__(self, 404, "not_found_error",
                              "model '%s' is not routable via this launcher; available: %s (%s)"
                              % (requested, shown or "none", hint), False)
        self.requested = requested


def _strip_1m(name):
    return name[:-len(_SUFFIX_1M)] if name.lower().endswith(_SUFFIX_1M) else name


def _legacy_grok_model(rest):
    """``4-3`` -> ``grok-4.3``; ``4-20-0309-reasoning`` -> ``grok-4.20-0309-reasoning``."""
    m = _LEGACY_GROK_VERSION.match(rest)
    if m:
        return "grok-%s.%s%s" % (m.group(1), m.group(2), m.group(3) or "")
    return "grok-" + rest


def _fmt_ctx(n):
    if not n:
        return "unknown"
    if n >= 1000000:
        return "%gM" % round(n / 1000000.0, 1)
    if n >= 1000:
        return "%dk" % int(round(n / 1000.0))
    return str(n)


class Router(object):
    def __init__(self, table, discovered=None, launcher_name=None):
        self.table = table
        self.launcher_name = launcher_name
        self._lock = threading.Lock()
        self._discovered = {}  # provider id -> List[str]
        self._logged = set()
        for pid, ids in (discovered or {}).items():
            self.set_discovered(pid, ids)

    # ---- discovery --------------------------------------------------------------------
    def set_discovered(self, provider_id, model_ids):
        ids, seen = [], set()
        for m in model_ids or []:
            m = m.strip() if isinstance(m, str) else ""
            if m and "," not in m and m not in seen:
                seen.add(m)
                ids.append(m)
        with self._lock:
            self._discovered[provider_id] = ids

    def discovered(self, provider_id):
        with self._lock:
            return list(self._discovered.get(provider_id, ()))

    def _find_discovered(self, provider_id, model):
        ids = self.discovered(provider_id)
        if model in ids:
            return model
        low = model.lower()
        for m in ids:
            if m.lower() == low:
                return m
        return None

    # ---- resolution -------------------------------------------------------------------
    def resolve(self, requested, est_tokens=None):
        """-> ``Resolution`` for ``requested`` (verbatim Claude Code model string)."""
        res = self._resolve(requested)
        if res is None:
            raise self.not_found(requested)
        return self._long_context(res, est_tokens)

    def not_found(self, requested):
        try:
            available = [e["id"] for e in self.models_response()["data"][:6]]
        except Exception:  # never let listing problems mask the 404
            available = []
        return RouteNotFound(requested, available, self.launcher_name)

    def _log_once(self, key, msg, *args):
        with self._lock:
            if key in self._logged:
                return
            self._logged.add(key)
        LOG.info(msg, *args)

    def _strip(self, name):
        name = name.strip()
        for prefix in (self.table.picker_prefix or "", "claude-via-"):
            if prefix and name.lower().startswith(prefix.lower()):
                name = name[len(prefix):]
                break
        return _strip_1m(name).strip()

    def _resolve(self, requested):
        if not isinstance(requested, str) or not requested.strip():
            return None
        name = self._strip(requested)
        if not name:
            return None
        role = _ROLE_ALIASES.get(name.lower())
        if role is not None:
            return self._resolve_role(role, requested)
        if "," in name:
            return self._resolve_pair(name, requested)
        alias = self._alias(name)
        if alias is not None:
            return self._resolve_target(alias, requested, None, False, 0)
        for pid, provider in self.table.providers.items():
            spec = provider.find_model(name)
            if spec is None:
                disc = self._find_discovered(pid, name)
                spec = ModelSpec(id=disc) if disc is not None else None
            if spec is not None:
                return Resolution(provider, spec.id, spec, requested)
        low = name.lower()
        if "haiku" in low:
            return self._resolve_role("background", requested)
        if low.startswith("claude") or low.startswith("anthropic") or low in _CLAUDE_FAMILY:
            return self._resolve_role("default", requested)
        return None

    def _alias(self, name):
        aliases = self.table.aliases
        if name in aliases:
            return aliases[name]
        low = name.lower()
        for k, v in aliases.items():
            if k.lower() == low:
                return v
        return None

    def _resolve_role(self, role, requested):
        if role in _IGNORED_ROLES:
            self._log_once("role:" + role, "role %r is not supported by the gateway; using the default route", role)
            role = "default"
        target = self.table.roles.get(role) or self.table.roles.get("default")
        if not target:
            return None
        return self._resolve_target(target, requested, role, role == "background", 0)

    def _resolve_target(self, target, requested, role, background, depth):
        """Configured target (``provider,model`` or alias name) -> Resolution (lenient listing)."""
        if depth > 4 or not isinstance(target, str):
            return None
        prov, _ = split_route(target)
        if prov is None:
            alias = self._alias(target.strip())
            return None if alias is None else self._resolve_target(alias, requested, role, background, depth + 1)
        return self._resolve_pair(target, requested, role, background, strict=False, depth=depth)

    def _canonical(self, provider_id):
        pid = self.table.canonical_provider(provider_id)
        if pid is not None:
            return pid
        low = provider_id.lower()
        for cand in self.table.providers:
            if cand.lower() == low:
                return cand
        for alt in self.table.provider_aliases.get(low, []):
            if alt in self.table.providers:
                return alt
        return None

    def _resolve_pair(self, name, requested, role=None, background=False, strict=True, depth=0):
        prov, model = split_route(name)
        if not prov or not model:
            return None
        plow = prov.lower()
        if plow == "ollama" and model.lower().startswith("fry-"):
            mlow = model.lower()
            if mlow.startswith("fry-grok-"):
                alias = self._alias(model) or self._alias("ollama," + model)
                if alias is not None and depth < 4:
                    return self._resolve_target(alias, requested, role, background, depth + 1)
                prov, model = "grok", _legacy_grok_model(model[len("fry-grok-"):])
            elif mlow.startswith("fry-codex-"):
                prov, model = "codex", model[len("fry-codex-"):]
            elif mlow.startswith("fry-opencode-"):
                prov, model = "opencode", model[len("fry-opencode-"):]
        elif plow == "local-ollama":
            prov = "ollama"
        model = _strip_1m(model).strip()
        pid = self._canonical(prov)
        if pid is None or not model:
            return None
        provider = self.table.providers[pid]
        spec = provider.find_model(model)
        if spec is None:
            disc = self._find_discovered(pid, model)
            if disc is None and strict and not provider.allow_unlisted:
                return None
            spec = ModelSpec(id=disc or model)
        return Resolution(provider, spec.id, spec, requested, role, background)

    def _long_context(self, res, est_tokens):
        target = self.table.roles.get("longContext")
        if not target or est_tokens is None or res.background or res.role == "longContext":
            return res
        if est_tokens <= self.table.long_context_threshold:
            return res
        default = self._resolve_role("default", res.requested)
        if default is None or (default.provider.id, default.model) != (res.provider.id, res.model):
            return res
        lc = self._resolve_target(target, res.requested, "longContext", False, 0)
        return lc if lc is not None else res

    # ---- /v1/models -------------------------------------------------------------------
    def _entry(self, provider, spec):
        auth_kind = (provider.auth or {}).get("kind", "none")
        if provider.dialect == "cli":
            source = "%s CLI" % ((provider.options or {}).get("cli") or "local")
        else:
            source = _AUTH_SOURCES.get(auth_kind, auth_kind)
        tools = "chat-only (no tools)" if (provider.chat_only or not spec.tools) else "tools"
        return {
            "type": "model",
            "id": self.table.picker_id(provider.id, spec),
            "display_name": "%s · %s" % (spec.label(), provider.label()),
            "created_at": MODELS_CREATED_AT,
            "description": "%s · %s · %s ctx" % (source, tools, _fmt_ctx(spec.context)),
        }

    def _entries(self):
        out, seen, per = [], set(), {}

        def add(provider, spec):
            if per.get(provider.id, 0) >= MAX_MODELS_PER_PROVIDER:
                return
            e = self._entry(provider, spec)
            if e["id"] in seen:
                return
            seen.add(e["id"])
            per[provider.id] = per.get(provider.id, 0) + 1
            out.append(e)

        default = self._resolve_role("default", "default")
        if default is not None:
            add(default.provider, default.model_spec)
        for pid, provider in self.table.providers.items():
            for spec in provider.models:
                add(provider, spec)
            for mid in self.discovered(pid):
                if provider.find_model(mid) is None:
                    add(provider, ModelSpec(id=mid))
        return out

    def models_response(self):
        data = self._entries()
        return {"data": data, "has_more": False,
                "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None}

    def model_entry(self, model_id):
        """Listing entry for ``model_id`` (exact, case-insensitive, ``[1m]``-insensitive), else an
        entry for any routable id (echoing ``model_id``), else None."""
        if not isinstance(model_id, str) or not model_id.strip():
            return None
        data = self._entries()
        for e in data:
            if e["id"] == model_id:
                return e
        key = _strip_1m(model_id.strip()).lower()
        for e in data:
            if _strip_1m(e["id"]).lower() == key:
                return e
        res = self._resolve(model_id)
        if res is None:
            return None
        e = self._entry(res.provider, res.model_spec)
        e["id"] = model_id
        return e
