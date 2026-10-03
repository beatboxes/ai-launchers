"""Launcher core shared by grok-wrap, codex-wrap, gemini-wrap, deepseek-wrap and kimi-wrap.

Each ``<x>-wrap.py`` calls ``run(<its config.example.json>)``. The manifest (v0.2.0) declares:

* ``mode`` — ``gateway`` (the only mode): Claude Code talks to the in-process stdlib gateway
  (``shared/gateway``), which holds the provider credentials and translates to the provider's dialect
  (Anthropic-compatible vendors such as DeepSeek/Kimi go through ``anthropic_passthrough``).
* ``transports`` — ordered, API-key transports first. Each names a gateway preset; ``auth`` is
  ``api-key`` | ``login`` | ``adc``; optional ``env_extra`` adds non-secret ``CLAUDE_CODE_*`` settings to
  Claude Code's environment and ``list_url`` names a model-list endpoint that cannot be derived from
  ``base_url``. The user's ``~/.ai-launchers/config.json`` may override ``providers.<transport id>.{base_url,
  env, op_ref, models, default_model, background_model, options, list_url}`` and ``gateway.port``.

Commands: ``launch claude`` / ``models`` / ``keys`` / ``doctor`` / ``--version`` / ``--help``.
Invariants: ``~/.claude.json`` and ``~/.claude/settings.json`` are never written; provider secrets
live only in the gateway's in-memory ``SecretStore`` — never in Claude Code's environment or argv (it
gets a random per-launch gateway token); ``--dry-run`` writes no files and starts no threads, servers,
subprocesses or network requests; the gateway logs only to ``~/.ai-launchers/logs/`` while Claude
Code runs.
"""

import argparse
import concurrent.futures
import getpass
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time

from . import key_manager
from .gateway import __version__ as GATEWAY_VERSION
from .gateway import compat, presets
from .gateway import config as gwconfig
from .utils import (VERSION, cache_dir, config_path, home_dir, json_error, load_config, load_state, logs_dir,
                    read_json, save_state, write_json)

__all__ = ["run", "Launcher", "Transport", "ManifestError", "load_manifest", "manifest_problems",
           "LaunchError", "AUTH_CHOICES", "MODELS_CACHE_TTL"]

AUTH_TYPES = ("api-key", "login", "adc")
AUTH_CHOICES = ("auto",) + AUTH_TYPES
AUTH_OF_KIND = {"api_key": "api-key", "codex_chatgpt": "login", "grok_cli": "login", "gcloud_adc": "adc"}
MODES = ("gateway",)
TRANSPORT_KEYS = ("id", "preset", "auth", "env", "op_ref", "default_model", "background_model", "base_url",
                  "models", "options", "setup", "list_url", "env_extra")
USER_KEYS = ("base_url", "env", "env_var", "op_ref", "models", "default_model", "background_model", "options",
             "list_url")
# env_extra: non-secret Claude Code settings only (never auth/provider selection variables)
EXTRA_ENV_RE = re.compile(r"^CLAUDE_CODE_(?!USE_|OAUTH)[A-Z0-9_]+$")
SENSITIVE_ENV_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)", re.I)
TESTED_CLAUDE = re.compile(r"^2\.1\.")
MODELS_CACHE_TTL = 6 * 3600
DISCOVERY_TIMEOUT = 3.0
LIVE_TIMEOUT = 180.0
# (keep regex, drop regex) for ids returned by provider list endpoints (chat-capable models only)
DISCOVERY_FILTERS = {
    "openai": (r"^(gpt-|o\d|codex)", r"(audio|realtime|tts|transcribe|image|search|embedding|moderation|instruct)"),
    "xai": (r"^grok", r"(image|imagine|video)"),
    "gemini": (r"^gemini", r"(embedding|image|tts|live|audio|aqa)"),
}
MAGIC_TOOL = {
    "name": "get_magic",
    "description": "Return the magic number for the integer n.",
    "input_schema": {"type": "object", "properties": {"n": {"type": "integer", "description": "the input number"}},
                     "required": ["n"]},
}
MAGIC_PROMPT = "Call get_magic with n=7, then tell me the number it returned."


class ManifestError(ValueError):
    def __init__(self, path, problems):
        self.problems = list(problems)
        ValueError.__init__(self, "%s: %s" % (path, "; ".join(self.problems)))


class LaunchError(Exception):
    """User-facing failure: ``message`` is printed to stderr, ``code`` becomes the exit status."""

    def __init__(self, message, code=2):
        Exception.__init__(self, message)
        self.code = code


# ---------------------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------------------

def manifest_problems(m):
    """Every problem with a manifest dict (``[]`` when valid)."""
    if not isinstance(m, dict):
        return ["manifest must be a JSON object"]
    out = []
    if not re.match(r"^[a-z0-9][a-z0-9-]*$", str(m.get("name") or "")):
        out.append("name must be a lower-case launcher name (got %r)" % (m.get("name"),))
    if not re.match(r"^\d+\.\d+\.\d+$", str(m.get("version") or "")):
        out.append("version must be MAJOR.MINOR.PATCH (got %r)" % (m.get("version"),))
    mode = m.get("mode")
    if mode not in MODES:
        out.append("mode must be one of %s (got %r)" % (", ".join(MODES), mode))
    transports = m.get("transports")
    if not isinstance(transports, list) or not transports:
        return out + ["transports must be a non-empty list"]
    seen, keyless_seen = set(), False
    for i, t in enumerate(transports):
        where = "transports[%d]" % i
        if not isinstance(t, dict):
            out.append("%s must be an object" % where)
            continue
        tid = t.get("id")
        if not isinstance(tid, str) or not re.match(r"^[a-z0-9][a-z0-9.-]*$", tid):
            out.append("%s: invalid id %r" % (where, tid))
        elif tid in seen:
            out.append("%s: duplicate id %r" % (where, tid))
        seen.add(tid)
        where = "transport %r" % (tid,)
        out.extend("%s: unknown key %r" % (where, k) for k in t if k not in TRANSPORT_KEYS and not k.startswith("_"))
        preset = presets.PRESETS.get(t.get("preset"))
        if preset is None:
            out.append("%s: unknown preset %r" % (where, t.get("preset")))
            continue
        auth = t.get("auth")
        expected = AUTH_OF_KIND.get((preset.get("auth") or {}).get("kind"))
        if auth not in AUTH_TYPES or auth != expected:
            out.append("%s: auth must be %r for preset %r (got %r)" % (where, expected, t["preset"], auth))
        if auth == "api-key":
            if keyless_seen:
                out.append("%s: API-key transports must come before login/ADC transports" % where)
            env = t.get("env", presets.secret_env_vars(t["preset"]))
            if not isinstance(env, list) or not env or not all(isinstance(v, str) and v for v in env):
                out.append("%s: env must be a non-empty list of variable names" % where)
            if t.get("op_ref") and not str(t["op_ref"]).startswith("op://"):
                out.append("%s: op_ref must start with op:// (or be empty)" % where)
        else:
            keyless_seen = True
            for k in ("env", "op_ref"):
                if t.get(k):
                    out.append("%s: %s only applies to api-key transports" % (where, k))
        for k in ("base_url", "list_url"):
            if t.get(k) is not None and not re.match(r"^https?://[^\s/]+", str(t[k])):
                out.append("%s: %s must be http(s)://… (got %r)" % (where, k, t[k]))
        if not (t.get("default_model") or preset.get("default_model")):
            out.append("%s: default_model required" % where)
        extra = t.get("env_extra", {})
        if not isinstance(extra, dict):
            out.append("%s: env_extra must be an object" % where)
            extra = {}
        for k, v in extra.items():
            if not EXTRA_ENV_RE.match(k) or SENSITIVE_ENV_RE.search(k):
                out.append("%s: env_extra may only set non-secret CLAUDE_CODE_* settings (got %r)" % (where, k))
            if not isinstance(v, str):
                out.append("%s: env_extra[%r] must be a string" % (where, k))
    return out


def load_manifest(path):
    path = str(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError) as exc:
        raise ManifestError(path, ["cannot read manifest: %s" % exc])
    problems = manifest_problems(m)
    if problems:
        raise ManifestError(path, problems)
    return m


class Transport(object):
    """One manifest transport merged with the user's ``providers.<id>`` overrides, plus runtime state."""

    def __init__(self, launcher, data, user=None):
        self.launcher = launcher
        self.id = data["id"]
        self.preset = data["preset"]
        self.auth = data["auth"]
        self.warnings = []
        p = presets.PRESETS[self.preset]
        user = {k: v for k, v in (user if isinstance(user, dict) else {}).items() if k in USER_KEYS}
        if "env_var" in user:  # v0.1 key name
            user.setdefault("env", user.pop("env_var"))
        if isinstance(user.get("env"), str):
            user["env"] = [user["env"]]
        for k, kind in (("env", list), ("options", dict)):
            if k in user and not isinstance(user[k], kind):
                self.warnings.append("ignoring providers.%s.%s: expected a JSON %s" % (
                    self.id, k, "list" if kind is list else "object"))
                del user[k]
        if re.search(r"/chat/completions/?$", str(user.get("base_url") or "")):
            self.warnings.append("ignoring providers.%s.base_url %r: v0.1 CCR-style endpoint URL (give the API "
                                 "base, e.g. %s)" % (self.id, user.pop("base_url"), p.get("base_url")))
        merged = dict(data)
        merged.update(user)
        keyed = self.auth == "api-key"
        self.env = [v for v in (merged.get("env") or presets.secret_env_vars(self.preset))
                    if keyed and isinstance(v, str) and v]
        self.op_ref = (merged.get("op_ref") or "") if keyed else ""
        self.default_model = merged.get("default_model") or p.get("default_model")
        self.background_model = merged.get("background_model") or p.get("background_model")
        self.base_url = (merged.get("base_url") or "").rstrip("/")
        models = merged.get("models")
        self.models = models if isinstance(models, list) else None
        self.options = dict(merged.get("options") or {})
        self.setup = data.get("setup") or ""
        self.list_url = merged.get("list_url") or ""
        self.env_extra = dict(data.get("env_extra") or {})
        # runtime (filled by Launcher.resolve)
        self.available = False
        self.source = None
        self.reason = ""
        self.auth_obj = None

    @property
    def secret_name(self):
        return self.id

    def sources(self):
        """Ordered secret sources (``gateway.secrets`` syntax) for an api-key transport."""
        extra = []
        if self.preset == "openai":  # `codex login --with-api-key` stores the key in ~/.codex/auth.json
            extra.append("json:%s#OPENAI_API_KEY" % key_manager.codex_auth_path())
        return key_manager.secret_sources(self.launcher, self.env, self.op_ref, extra)

    def provider_overrides(self):
        ov = {}
        if self.base_url:
            ov["base_url"] = self.base_url
        if self.models is not None:
            ov["models"] = self.models
        if self.options:
            ov["options"] = self.options
        if self.auth == "api-key":
            ov["auth"] = {"secret": self.secret_name}
        elif self.auth == "adc":  # gcloud_adc reads project/location from its auth options too
            scope = {k: self.options[k] for k in ("project", "location") if self.options.get(k)}
            if scope:
                ov["auth"] = scope
        return ov

    def spec(self):
        return presets.provider_from_preset(self.preset, dict(self.provider_overrides(), id=self.id))

    def label(self):
        return "%s (%s)" % (self.id, self.auth)

    def setup_hint(self, launcher):
        if self.auth == "api-key":
            alt = " (or %s)" % ", ".join(self.env[1:]) if len(self.env) > 1 else ""
            return ("set %s%s in the environment, run `%s keys set`, or set providers.%s.op_ref (op://…) in %s"
                    % (self.env[0], alt, launcher, self.id, config_path()))
        return self.setup or "configure %s credentials" % self.id

    def unavailable_text(self, launcher):
        """Why this transport is not used and how to enable it."""
        if self.reason.startswith("excluded by"):
            return self.reason
        return self.setup_hint(launcher) + (" (%s)" % self.reason if self.reason else "")


# ---------------------------------------------------------------------------------------
# launcher
# ---------------------------------------------------------------------------------------

def _say(*parts):
    print(*parts)
    sys.stdout.flush()


def _err(msg):
    print(msg, file=sys.stderr)
    sys.stderr.flush()


def _quote_argv(argv):
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return " ".join(shlex.quote(a) for a in argv)


def _fmt_ctx(n):
    if not n:
        return "?"
    if n >= 1000000:
        return "%gM" % (n / 1000000.0)
    return "%dk" % (n // 1000)


def _describe(d):
    """``auth.describe()`` dict -> short ``k=v`` text (secret-free by contract; noise keys dropped)."""
    return ", ".join("%s=%s" % (k, v) for k, v in sorted((d or {}).items())
                     if k not in ("provider", "available", "kind", "hint") and v not in (None, "", False))


class Launcher(object):
    def __init__(self, manifest, config=None, environ=None):
        self.m = manifest
        self.name = manifest["name"]
        self.mode = manifest["mode"]
        self.version = manifest.get("version") or VERSION
        self.cfg = load_config() if config is None else config
        self.environ = os.environ if environ is None else environ
        users = self.cfg.get("providers") if isinstance(self.cfg.get("providers"), dict) else {}
        self.transports = [Transport(self.name, t, users.get(t["id"])) for t in manifest["transports"]]

    # ---- resolution ---------------------------------------------------------------------
    def resolve(self, auth="auto", dry_run=False):
        """Resolve secrets and availability for transports matching ``auth``.

        Returns ``(available transports in manifest order, SecretStore)``. ``dry_run`` resolves only
        local sources (env, credentials.json, auth files): no threads, subprocesses or network; a
        configured ``op_ref`` then counts as available (unresolved).
        """
        store = gwconfig.SecretStore()
        cands = [t for t in self.transports if auth in ("auto", t.auth)]
        for t in self.transports:
            t.available, t.source, t.reason, t.auth_obj = False, None, "", None
            if t not in cands:
                t.reason = "excluded by --auth %s" % auth
        keyed = [t for t in cands if t.auth == "api-key"]
        if dry_run:
            for t in keyed:
                value, label = key_manager.peek(t.sources(), self.environ)
                if value:
                    store.set(t.secret_name, value)
                    t.source = label
        elif keyed:
            from .gateway import secrets as gwsecrets

            labels = gwsecrets.load_into(store, {t.secret_name: t.sources() for t in keyed})
            for t in keyed:
                t.source = labels.get(t.secret_name) if store.has(t.secret_name) else None
        for t in keyed:
            if not store.has(t.secret_name) and t.preset == "xai":
                self._grok_api_key_entry(t, store)
            t.available = store.has(t.secret_name)
            if not t.available and dry_run and t.op_ref:
                t.available, t.source = True, "%s (not resolved in dry-run)" % t.op_ref
            if not t.available:
                t.reason = "no API key (checked %s)" % key_manager.describe_sources(t.sources())
        for t in cands:
            if t.auth != "api-key":
                self._check_auth(t, store)
        return [t for t in cands if t.available], store

    def _check_auth(self, t, store):
        from .gateway.auth import make_auth

        try:
            spec = t.spec()
            t.auth_obj = make_auth(spec.auth, store, t.id)
            t.available = bool(t.auth_obj.available())
            info = t.auth_obj.describe() or {}
        except Exception as exc:  # broken/missing credential files or auth module: report, don't crash
            t.available, t.reason = False, "%s: %s" % (type(exc).__name__, exc)
            return
        detail = _describe(info)
        if t.available:
            t.source = ("%s login" % t.id if t.auth == "login" else "gcloud ADC") + (" (%s)" % detail if detail else "")
        else:
            t.reason = detail

    def _grok_api_key_entry(self, t, store):
        """``grok login`` may store a plain xAI API key (``auth_mode: api_key``) -> use it for the xai route."""
        try:
            from .gateway.auth.grok_cli import GrokCliAuth

            key = GrokCliAuth(provider_id="grok", options={"kind": "grok_cli"}).api_key_entry()
        except Exception:  # no/unreadable grok auth file or module: simply no key from this source
            return
        if key:
            store.set(t.secret_name, key)
            t.source = "grok auth.json (api_key entry)"

    def warnings(self):
        """Config problems: unreadable config.json and ignored per-transport overrides."""
        out = ["ignoring %s" % problem for problem in [json_error(config_path())] if problem]
        return out + [w for t in self.transports for w in t.warnings]

    def _require(self, auth, dry_run=False):
        selected, store = self.resolve(auth, dry_run=dry_run)
        for w in self.warnings():
            _err("%s: warning: %s" % (self.name, w))
        if not selected:
            lines = ["%s: no usable transport (--auth %s). Set up one of:" % (self.name, auth)]
            for t in self.transports:
                if auth in ("auto", t.auth):
                    lines.append("  - %s: %s" % (t.label(), t.unavailable_text(self.name)))
            lines.append("Then run `%s doctor` to verify." % self.name)
            raise LaunchError("\n".join(lines), 2)
        return selected, store

    def _base_env(self):
        """Parent env minus this launcher's provider key variables (launchkit also scrubs any inherited
        variable containing a resolved secret)."""
        drop = {v.upper() for t in self.transports for v in t.env}
        return {k: v for k, v in self.environ.items() if k.upper() not in drop}

    # ---- route table ----------------------------------------------------------------------
    def route_table(self, selected, model=None):
        first = selected[0]
        roles = {"default": "%s,%s" % (first.id, first.default_model)}
        if first.background_model:
            roles["background"] = "%s,%s" % (first.id, first.background_model)
        table = presets.route_table_from_presets([(t.preset, t.id) for t in selected],
                                                 {t.id: t.provider_overrides() for t in selected}, roles)
        cache = self._models_cache()
        for t in selected:
            ids = self._cached_ids(cache, t, table.providers[t.id])
            prov = table.providers[t.id]
            prov.models.extend(gwconfig.ModelSpec(id=i) for i in ids if not prov.lists_model(i))
        table = gwconfig.apply_upstream_env_overrides(table, self.environ)
        if model:
            self._apply_model(table, selected, model)
        try:
            return table.validate()
        except gwconfig.ConfigError as exc:
            raise LaunchError("%s: invalid route configuration:\n  %s" % (self.name, "\n  ".join(exc.problems)))

    def _apply_model(self, table, selected, model):
        from .gateway.router import RouteNotFound, Router

        try:
            res = Router(table, launcher_name=self.name).resolve(model)
        except RouteNotFound as exc:
            raise LaunchError("%s: --model %s: %s" % (self.name, model, getattr(exc, "message", exc)))
        pid = res.provider.id
        old_pid = gwconfig.split_route(table.roles["default"])[0]
        table.roles["default"] = "%s,%s" % (pid, res.model)
        if pid != old_pid:
            t = next((x for x in selected if x.id == pid), None)
            if t is not None and t.background_model:
                table.roles["background"] = "%s,%s" % (pid, t.background_model)

    @staticmethod
    def default_route(table):
        pid, model = gwconfig.split_route(table.roles["default"])
        spec = table.providers[pid].model_spec(model)
        return pid, spec, table.picker_id(pid, spec)

    # ---- launch ---------------------------------------------------------------------------
    def _find_claude(self):
        from .gateway import launchkit

        try:
            return list(launchkit.find_claude(self.environ)), None
        except OSError as exc:  # launchkit.ClaudeNotFound
            return None, str(exc)

    def _gateway_port(self, port):
        if port is None:
            gw = self.cfg.get("gateway") if isinstance(self.cfg.get("gateway"), dict) else {}
            port = gw.get("port") or 0
        try:
            port = int(port)
        except (TypeError, ValueError):
            raise LaunchError("%s: invalid gateway port %r" % (self.name, port))
        if not 0 <= port <= 65535:
            raise LaunchError("%s: gateway port must be 0..65535 (got %d)" % (self.name, port))
        return port

    def cmd_launch(self, opts, passthrough):
        selected, store = self._require(opts.auth, dry_run=opts.dry_run)
        prefix, why = self._find_claude()
        if prefix is None:
            if not opts.dry_run:
                raise LaunchError("%s: %s" % (self.name, why), 2)
            _err("%s: warning: %s" % (self.name, why))
        argv = (prefix or ["<claude not found>"]) + list(passthrough)
        table = self.route_table(selected, opts.model)
        port = self._gateway_port(opts.port)
        pid, spec, default_id = self.default_route(table)
        background_id = table.picker_prefix + "background"
        via = next(t for t in selected if t.id == pid)
        if opts.dry_run:
            from .gateway import launchkit

            env = launchkit.build_child_env(self._base_env(), "http://127.0.0.1:%s" % (port or "<port>"),
                                            "<per-launch random token>", default_id, background_id, spec.context,
                                            extra=via.env_extra, secret_values=store)
            self._print_plan(selected, store, env, argv, table, opts.debug)
            return 0
        self._tos_notice(selected)
        log, log_file = self._setup_logging(store, opts.debug)
        from .gateway import launchkit
        from .gateway.server import Gateway

        trace = self.environ.get("AI_GATEWAY_TRACE_FILE") or (
            str(logs_dir() / ("%s-trace.jsonl" % self.name)) if opts.debug else None)
        try:
            gw = Gateway(table, store, port=port, log=log, trace_file=trace, launcher_name=self.name).start()
        except OSError as exc:
            raise LaunchError("%s: cannot start the gateway on 127.0.0.1:%d: %s%s" % (
                self.name, port, exc, " (port busy? use --port 0)" if port else ""), 3)
        try:
            env = launchkit.build_child_env(self._base_env(), gw.url, gw.token, default_id, background_id,
                                            spec.context, extra=via.env_extra, secret_values=store)
            _err("%s: Claude Code -> %s via %s [%s] · gateway %s%s" % (
                self.name, default_id, via.id, key_manager.display_source(via.source), gw.url,
                " · log %s" % log_file if log_file else ""))
            if opts.debug:
                self._print_env_diff(env, store, {gw.token: "<per-launch random token>"})
                _err("argv: %s" % store.redact(_quote_argv(argv)))
            return self._run(launchkit, argv, env)
        finally:
            gw.stop()

    def _run(self, launchkit, argv, env):
        try:
            return launchkit.run_child(argv, env)
        except OSError as exc:
            raise LaunchError("%s: failed to run %s: %s" % (self.name, argv[0], exc), 2)

    def _print_plan(self, selected, store, env, argv, table, full):
        _say("[dry-run] %s %s launch (%s mode) — nothing is started or written" % (
            self.name, self.version, self.mode))
        _say("transports:")
        for t in self.transports:
            state = "available" if t in selected else "unavailable"
            detail = key_manager.display_source(t.source) if t in selected else t.unavailable_text(self.name)
            _say("  %-14s %-8s %-11s %s" % (t.id, t.auth, state, store.redact(detail)))
        if table is not None:
            _say("route table (no secrets%s):" % ("" if full else "; --debug prints the full JSON"))
            if full:
                _say(store.redact(json.dumps(table.to_dict(), indent=2, sort_keys=True)))
            else:
                _say("  roles: %s" % "  ".join("%s=%s" % kv for kv in sorted(table.roles.items()) if kv[1]))
                for pid, p in table.providers.items():
                    wire = "/".join(x for x in (p.dialect, p.target or p.profile) if x)
                    fb = " (fallback %s %s)" % (p.fallback.dialect, p.fallback.base_url) if p.fallback else ""
                    _say("  %s: %s %s auth=%s%s" % (pid, wire, p.base_url or "(derived)", p.auth.get("kind"), fb))
                    _say("      models: %s" % ", ".join(m.id for m in p.models))
            for line in gwconfig.describe_upstream_env_overrides(table, self.environ):
                _say("upstream override: %s" % line)
        _say("child environment changes (sensitive values redacted):")
        for line in self._env_diff_lines(env, store, {}):
            _say(line)
        _say("argv: %s" % store.redact(_quote_argv(argv)))

    def _env_diff_lines(self, env, store, placeholders):
        out = []
        for k in sorted(set(self.environ) | set(env)):
            if k not in env:
                out.append("  - %s" % k)
            elif self.environ.get(k) != env[k]:
                v = placeholders.get(env[k], env[k])
                if v and SENSITIVE_ENV_RE.search(k) and not (v.startswith("<") and v.endswith(">")):
                    v = "<redacted>"
                out.append("  %s %s=%s" % ("+" if k not in self.environ else "~", k, store.redact(v)))
        return out

    def _print_env_diff(self, env, store, placeholders):
        _err("child environment changes:")
        for line in self._env_diff_lines(env, store, placeholders):
            _err(line)

    def _tos_notice(self, selected):
        logins = [t for t in selected if t.auth == "login"]
        if not logins:
            return
        state = load_state()
        seen = state.get("notices") if isinstance(state.get("notices"), dict) else {}
        fresh = [t for t in logins if "tos:%s" % t.id not in seen]
        if not fresh:
            return
        for t in fresh:
            _err("%s: note — the %r transport (%s) sends your subscription login to the endpoint built for the "
                 "provider's official CLI. That use may be subject to the provider's terms of service; an API-key "
                 "route is used first whenever one is configured (`%s keys set`). This notice is shown once." % (
                     self.name, t.id, t.spec().label(), self.name))
            seen["tos:%s" % t.id] = compat.rfc3339_format(None, "seconds")
        state["notices"] = seen
        try:
            save_state(state)
        except OSError as exc:
            _err("%s: warning: could not record notice state: %s" % (self.name, exc))

    def _setup_logging(self, store, debug):
        """Gateway logger -> rotating ``logs/<launcher>.log`` only (tracing never writes to the terminal)."""
        from .gateway import tracing

        level = "DEBUG" if debug else "INFO"
        log_file = logs_dir() / ("%s.log" % self.name)
        try:
            return tracing.setup_logging(level, str(log_file), store), log_file
        except OSError as exc:
            _err("%s: warning: gateway logging disabled (%s)" % (self.name, exc))
            return tracing.setup_logging(level, None, store), None

    # ---- models -------------------------------------------------------------------------------
    def _models_cache(self):
        return read_json(cache_dir() / "models.json")

    def _discovery_url(self, t, spec):
        if t.auth != "api-key":
            return None
        if t.list_url:
            return t.list_url
        override = self.environ.get(gwconfig.upstream_env_name(t.id))
        base = (override or spec.base_url).rstrip("/")
        if spec.dialect in ("openai_chat", "responses"):
            return base + "/models"
        if spec.dialect == "gemini" and spec.target == "gemini_api":
            return base + "/v1beta/models?pageSize=1000"
        return None

    def _cached_ids(self, cache, t, spec):
        entry = ((cache.get("entries") or {}) if isinstance(cache, dict) else {}).get(t.id)
        url = self._discovery_url(t, spec)
        if not isinstance(entry, dict) or not url or entry.get("url") != url:
            return []
        if time.time() - float(entry.get("fetched_at") or 0) > MODELS_CACHE_TTL:
            return []
        return [i for i in entry.get("models") or [] if isinstance(i, str) and i and "," not in i]

    def _discover_one(self, t, spec, url, store):
        from .gateway.auth import make_auth
        from .gateway.transport import HttpClient

        headers = make_auth(spec.auth, store, t.id).headers()
        client = HttpClient(timeout=DISCOVERY_TIMEOUT, connect_timeout=DISCOVERY_TIMEOUT, environ=self.environ)
        try:
            resp = client.request("GET", url, headers=headers, stream=False, timeout=DISCOVERY_TIMEOUT)
            if resp.status != 200:
                raise LaunchError("HTTP %d from %s" % (resp.status, url.split("?")[0]))
            data = resp.json()
        finally:
            client.close()
        if isinstance(data, dict) and isinstance(data.get("models"), list):  # Gemini
            ids = [compat.removeprefix(str(m.get("name") or ""), "models/") for m in data["models"]
                   if isinstance(m, dict) and "generateContent" in (m.get("supportedGenerationMethods") or [])]
        else:
            items = data.get("data") if isinstance(data, dict) else None
            ids = [str(m.get("id")) for m in items or [] if isinstance(m, dict) and m.get("id")]
        keep, drop = DISCOVERY_FILTERS.get(presets.PRESETS[t.preset].get("catalog"), (None, None))
        ids = [i for i in ids if i and "," not in i and (not keep or re.search(keep, i)) and
               not (drop and re.search(drop, i))]
        return sorted(set(ids))

    def discover(self, selected, store, force=False):
        """Refresh the discovery cache for available key transports (parallel, 3 s each)."""
        cache = self._models_cache()
        entries = dict(cache.get("entries") or {}) if isinstance(cache, dict) else {}
        jobs, errors = {}, {}
        for t in selected:
            spec = t.spec()
            url = self._discovery_url(t, spec)
            if url and (force or not self._cached_ids({"entries": entries}, t, spec)):
                jobs[t.id] = (t, spec, url)
        if not jobs:
            return errors
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futs = {tid: pool.submit(self._discover_one, t, spec, url, store) for tid, (t, spec, url) in jobs.items()}
            for tid, fut in futs.items():
                try:
                    ids = fut.result(timeout=DISCOVERY_TIMEOUT * 3)
                except Exception as exc:  # network/auth/JSON problems: keep the catalog, report
                    errors[tid] = store.redact(str(exc) or type(exc).__name__)
                    continue
                entries[tid] = {"url": jobs[tid][2], "fetched_at": time.time(), "models": ids}
        try:
            write_json(cache_dir() / "models.json", {"version": 1, "entries": entries})
        except OSError as exc:
            errors["cache"] = str(exc)
        return errors

    def models_report(self, selected):
        cache = self._models_cache()
        report = []
        for t in self.transports:
            spec = t.spec()
            table = gwconfig.RouteTable(providers={t.id: spec})
            cached = self._cached_ids(cache, t, spec)
            entry = {"id": t.id, "preset": t.preset, "auth": t.auth, "display_name": spec.label(),
                     "available": t in selected,
                     "source": key_manager.display_source(t.source) if t in selected else None,
                     "reason": None if t in selected else t.unavailable_text(self.name),
                     "models": []}
            entry["default_model"], entry["background_model"] = t.default_model, t.background_model
            listed = [(m, "catalog" if t.models is None else "config") for m in spec.models]
            listed += [(gwconfig.ModelSpec(id=i), "discovered") for i in cached if not spec.lists_model(i)]
            for m, origin in listed:
                entry["models"].append({
                    "id": m.id, "context": m.context, "source": origin,
                    "picker_id": table.picker_id(t.id, m)})
            report.append(entry)
        return report

    def cmd_models(self, opts):
        selected, store = self.resolve("auto", dry_run=False)
        errors = self.discover(selected, store, force=opts.refresh)
        report = self.models_report(selected)
        if opts.json:
            _say(json.dumps({"launcher": self.name, "version": self.version, "mode": self.mode,
                             "transports": report, "discovery_errors": errors}, indent=2))
            return 0
        _say("%s models — pick in Claude Code with /model <picker id> (or `launch claude --model <id>`)" % self.name)
        for e in report:
            state = "available via %s" % e["source"] if e["available"] else "unavailable — %s" % e["reason"]
            _say("\n%s · %s · %s · %s" % (e["id"], e["display_name"], e["auth"], store.redact(state)))
            for m in e["models"]:
                mark = ("default" if m["id"] == e.get("default_model") else
                        "background" if m["id"] == e.get("background_model") else "")
                origin = "" if m["source"] == "catalog" else " (%s)" % m["source"]
                _say("  %-10s %-36s %-50s %s%s" % (mark, m["id"], m["picker_id"], _fmt_ctx(m["context"]), origin))
            if e["id"] in errors:
                _say("  (discovery failed: %s)" % errors[e["id"]])
        return 0

    # ---- keys -------------------------------------------------------------------------------
    def cmd_keys(self, opts):
        keyed = [t for t in self.transports if t.auth == "api-key"]
        path = key_manager.creds_path()
        if not keyed:
            raise LaunchError("%s: this launcher has no API-key transport" % self.name)
        if opts.action == "set":
            value = opts.value
            if value is None:
                if sys.stdin is not None and sys.stdin.isatty():
                    value = getpass.getpass("%s API key for %s (input hidden): " % (self.name, keyed[0].id))
                else:
                    value = sys.stdin.readline() if sys.stdin is not None else ""
            else:
                _err("%s: warning: a key passed as an argument may be saved in your shell history; run `%s keys "
                     "set` without an argument to type it hidden (or pipe it on stdin)" % (self.name, self.name))
            try:
                key_manager.set_key(self.name, value)
            except ValueError as exc:
                raise LaunchError("%s: key not stored: %s" % (self.name, exc))
            _say("%s: key stored in %s (owner-only permissions); used by the %s transport after %s" % (
                self.name, path, keyed[0].id, ", ".join(keyed[0].env + ([keyed[0].op_ref] if keyed[0].op_ref else []))))
            return 0
        if opts.action == "remove":
            existed = key_manager.remove_key(self.name)
            _say("%s: %s" % (self.name, "stored key removed from %s" % path if existed else "no stored key"))
            return 0
        _say("%s keys (credentials file: %s)" % (self.name, path))
        stored = key_manager.stored_key(self.name)
        for t in self.transports:
            if t.auth != "api-key":
                _say("  %-14s %s transport — no key; check with `%s doctor`" % (t.id, t.auth, self.name))
                continue
            value, label = key_manager.peek(t.sources(), self.environ)
            active = key_manager.display_source(label) if value else (
                "%s (resolved at launch)" % t.op_ref if t.op_ref else "unset")
            _say("  %-14s active: %s%s" % (t.id, active, " " + key_manager.redact(value) if value else ""))
            for var in t.env:
                _say("  %-14s   env %-24s %s" % ("", var, "set" if (self.environ.get(var) or "").strip() else "unset"))
            _say("  %-14s   op_ref %s" % ("", t.op_ref or "(not configured)"))
            _say("  %-14s   credentials.json %s" % ("", "stored" if stored else "unset"))
        return 0

    # ---- doctor -------------------------------------------------------------------------------
    def cmd_doctor(self, opts):
        from .gateway import launchkit

        problems = []

        def line(level, text):
            _say("  [%s] %s" % (level, text))
            if level == "FAIL":
                problems.append(text)

        _say("%s %s doctor (gateway %s · Python %s · %s)" % (
            self.name, self.version, GATEWAY_VERSION, platform.python_version(), platform.platform(terse=True)))
        _say("Claude Code:")
        prefix, why = self._find_claude()
        if prefix is None:
            line("FAIL", "claude not found%s — npm i -g @anthropic-ai/claude-code" % (" (%s)" % why if why else ""))
        else:
            ver = launchkit.claude_version(prefix)
            line("ok" if ver and TESTED_CLAUDE.match(ver) else "warn", "%s — version %s%s" % (
                _quote_argv(prefix), ver or "unknown",
                "" if ver and TESTED_CLAUDE.match(ver) else " (tested with 2.1.x)"))
        for text in self._settings_warnings():
            line("warn", text)
        _say("Files:")
        line("ok", "config %s (%s)" % (config_path(), "present" if config_path().exists() else "absent; optional"))
        creds = key_manager.creds_path()
        if creds.exists() and os.name != "nt" and creds.stat().st_mode & 0o077:
            line("warn", "%s is readable by other users — run: chmod 600 '%s'" % (creds, creds))
        line("ok", "logs %s" % logs_dir())
        _say("Transports (%s mode):" % self.mode)
        selected, store = self.resolve(opts.auth, dry_run=False)
        for w in self.warnings():
            line("warn", w)
        for t in self.transports:
            if opts.auth not in ("auto", t.auth):
                continue
            if t in selected:
                line("ok", "%s: available via %s" % (t.label(), store.redact(key_manager.display_source(t.source))))
            else:
                line("warn", "%s: unavailable — %s" % (t.label(), t.unavailable_text(self.name)))
            if t.auth == "adc" and t in selected:
                self._doctor_adc(t, line)
        if not selected:
            line("FAIL", "no usable transport")
        for text in gwconfig.describe_upstream_env_overrides(
                presets.route_table_from_presets([(t.preset, t.id) for t in self.transports]), self.environ):
            line("warn", "upstream override active: %s" % text)
        _say("Network:")
        for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"):
            value = self.environ.get(var) or self.environ.get(var.lower())
            if value:
                line("ok", "%s=%s" % (var, re.sub(r"//[^/@\s]+@", "//***@", value)))
        if opts.live and selected:
            problems.extend(self._doctor_live(selected, store, opts.model))
        elif selected:
            _say("Tip: `%s doctor --live` runs an end-to-end tool-call check (2 tiny paid requests per route)."
                 % self.name)
        _say("%s" % ("doctor: %d problem(s)" % len(problems) if problems else "doctor: all checks passed"))
        return 1 if problems else 0

    def _settings_warnings(self):
        cfg_dir = self.environ.get("CLAUDE_CONFIG_DIR") or str(home_dir() / ".claude")
        path = os.path.join(cfg_dir, "settings.json")
        data = read_json(path)
        out = []
        env = data.get("env") if isinstance(data.get("env"), dict) else {}
        bad = sorted(k for k in env if k.startswith("ANTHROPIC_") or k.startswith("CLAUDE_CODE_USE_"))
        if bad:
            out.append("%s sets env %s — settings.json env overrides the launcher's environment; remove them"
                       % (path, ", ".join(bad)))
        if data.get("apiKeyHelper"):
            out.append("%s sets apiKeyHelper — it overrides the launcher's ANTHROPIC_AUTH_TOKEN; remove it" % path)
        return out

    @staticmethod
    def _doctor_adc(t, line):
        for attr in ("project", "location"):
            fn = getattr(t.auth_obj, attr, None)
            if fn is None:
                continue
            try:
                line("ok", "%s: %s %s" % (t.id, attr, fn()))
            except Exception as exc:  # gcloud missing / project unset: actionable message
                line("FAIL", "%s: %s unavailable — %s" % (t.id, attr, exc))

    def _doctor_live(self, selected, store, model):
        _say("Live check (tool-call round trip; each route makes 2 tiny PAID requests):")
        from .gateway.server import Gateway

        table = self.route_table(selected, model)
        if model:
            routes = [self.default_route(table)[2]]
        else:
            routes = [table.picker_id(t.id, table.providers[t.id].model_spec(t.default_model)) for t in selected]
        log, _ = self._setup_logging(store, False)
        gw = Gateway(table, store, port=0, log=log, launcher_name=self.name).start()
        try:
            headers = {"Authorization": "Bearer %s" % gw.token}
            return self._live_routes(routes, gw.url + "/v1/messages", headers, store)
        finally:
            gw.stop()

    def _live_routes(self, routes, url, headers, store):
        failures = []
        for label in routes:
            try:
                info = live_round_trip(url, headers, label, self.environ)
            except LaunchError as exc:
                msg = store.redact(str(exc))
                _say("  [FAIL] %s — %s" % (label, msg))
                failures.append("%s: %s" % (label, msg))
                continue
            _say("  [ok] %s — tool_use get_magic(n=7) in %d ms, final answer in %d ms (%s); usage in/out %s/%s"
                 % (label, info["t1_ms"], info["t2_ms"], info["text"][:60].replace("\n", " "),
                    info["input_tokens"], info["output_tokens"]))
        return failures

    # ---- CLI ----------------------------------------------------------------------------------
    def usage(self):
        lines = [
            "%s %s — %s" % (self.name, self.version, self.m.get("description") or "Claude Code launcher"),
            "",
            "Usage:",
            "  %s launch claude [--model M] [--auth auto|api-key|login|adc] [--port N] [--dry-run] [--debug]"
            % self.name,
            "                  [-- <claude args>]",
            "  %s models [--refresh] [--json]      list models and Claude Code picker ids" % self.name,
            "  %s keys set [KEY] | remove | list   manage the stored API key" % self.name,
            "  %s doctor [--live] [--model M] [--auth A]" % self.name,
            "  %s --version | --help" % self.name,
            "",
            "Claude Code talks to an in-process gateway that holds the provider credentials.",
            "Transports (tried in order; --auth filters):",
        ]
        for t in self.transports:
            how = "key from %s, then %s" % (", ".join(t.env) + (", " + t.op_ref if t.op_ref else ""),
                                            key_manager.creds_path()) if t.auth == "api-key" else t.setup
            lines.append("  %-14s %-8s %s" % (t.id, t.auth, how))
        lines += [
            "",
            "Config: %s (providers.<transport>.base_url|env|op_ref|models, gateway.port)" % config_path(),
            "Logs:   %s" % logs_dir(),
            "Everything after `--` is passed to claude unchanged.",
        ]
        return "\n".join(lines)

    def _parser(self):
        p = argparse.ArgumentParser(prog=self.name, add_help=False)
        sub = p.add_subparsers(dest="cmd")
        lp = sub.add_parser("launch", prog="%s launch" % self.name)
        lp.add_argument("agent", nargs="?", default="claude", choices=["claude"])
        lp.add_argument("--model")
        lp.add_argument("--auth", choices=AUTH_CHOICES, default="auto")
        lp.add_argument("--port", type=int)
        lp.add_argument("--dry-run", action="store_true")
        lp.add_argument("--debug", action="store_true")
        mp = sub.add_parser("models", prog="%s models" % self.name)
        mp.add_argument("--refresh", action="store_true")
        mp.add_argument("--json", action="store_true")
        kp = sub.add_parser("keys", prog="%s keys" % self.name)
        kp.add_argument("action", nargs="?", default="list", choices=["set", "remove", "list"])
        kp.add_argument("value", nargs="?")
        dp = sub.add_parser("doctor", prog="%s doctor" % self.name)
        dp.add_argument("--live", action="store_true")
        dp.add_argument("--model")
        dp.add_argument("--auth", choices=AUTH_CHOICES, default="auto")
        return p

    def main(self, argv):
        argv = list(argv)
        if not argv or argv[0] in ("-h", "--help", "help"):
            _say(self.usage())
            return 0
        if argv[0] in ("-V", "--version", "version"):
            _say("%s %s (gateway %s)" % (self.name, self.version, GATEWAY_VERSION))
            return 0
        passthrough = []
        if argv[0] == "launch" and "--" in argv:
            i = argv.index("--")
            argv, passthrough = argv[:i], argv[i + 1:]
        try:
            opts, unknown = self._parser().parse_known_args(argv)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 2
        if unknown:
            _err("%s: unrecognized arguments: %s%s" % (self.name, " ".join(unknown), (
                " (pass claude arguments after `--`)" if opts.cmd == "launch" else "")))
            return 2
        try:
            if opts.cmd == "launch":
                return self.cmd_launch(opts, passthrough)
            if opts.cmd == "models":
                return self.cmd_models(opts)
            if opts.cmd == "keys":
                return self.cmd_keys(opts)
            if opts.cmd == "doctor":
                return self.cmd_doctor(opts)
        except LaunchError as exc:
            _err(str(exc))
            return exc.code
        _err("%s: unknown command %r (see --help)" % (self.name, argv[0]))
        return 2


# ---------------------------------------------------------------------------------------
# live tool-call round trip (doctor --live)
# ---------------------------------------------------------------------------------------

def _read_message(resp):
    """Rebuild an Anthropic Message from an SSE response (raises LaunchError on ``event: error``)."""
    from .gateway.transport import iter_sse

    def obj(value):
        return value if isinstance(value, dict) else {}

    blocks, usage, stop = {}, {}, None
    for ev in iter_sse(resp):
        try:
            d = obj(json.loads(ev.data) if ev.data else {})
        except ValueError:
            continue
        kind = d.get("type") or ev.event
        if kind == "error":
            err = obj(d.get("error"))
            raise LaunchError("stream error %s: %s" % (err.get("type"), err.get("message")))
        if kind == "message_start":
            usage.update(obj(obj(d.get("message")).get("usage")))
        elif kind == "content_block_start" and isinstance(d.get("index"), int):
            block = dict(obj(d.get("content_block")))
            if block.get("type") == "tool_use":
                block["_json"] = ""
            blocks[d["index"]] = block
        elif kind == "content_block_delta":
            block, delta = blocks.get(d.get("index")), obj(d.get("delta"))
            if block is None:
                continue
            dt = delta.get("type")
            if dt == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif dt == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            elif dt == "signature_delta":
                block["signature"] = delta.get("signature", "")
            elif dt == "input_json_delta":
                block["_json"] = block.get("_json", "") + delta.get("partial_json", "")
        elif kind == "message_delta":
            stop = obj(d.get("delta")).get("stop_reason") or stop
            usage.update(obj(d.get("usage")))
    content = []
    for i in sorted(blocks):
        block = blocks[i]
        if "_json" in block:
            raw = block.pop("_json")
            try:
                block["input"] = json.loads(raw) if raw.strip() else block.get("input") or {}
            except ValueError:
                raise LaunchError("tool_use input is not valid JSON: %r" % raw[:200])
        content.append(block)
    return {"content": content, "stop_reason": stop, "usage": usage}


def _post_message(client, url, headers, body):
    hdrs = {"Content-Type": "application/json", "anthropic-version": "2023-06-01", "Accept": "text/event-stream"}
    hdrs.update(headers)
    t0 = time.time()
    resp = client.request("POST", url, headers=hdrs, body=json.dumps(body), stream=True, timeout=LIVE_TIMEOUT)
    try:
        if resp.status != 200:
            text = resp.text()
            try:
                msg = (json.loads(text).get("error") or {}).get("message") or text
            except (ValueError, AttributeError):
                msg = text
            raise LaunchError("HTTP %d: %s" % (resp.status, msg.strip()[:300]))
        if "event-stream" not in (resp.headers.get("content-type") or ""):
            raise LaunchError("expected an SSE stream, got %r" % resp.headers.get("content-type"))
        msg = _read_message(resp)
    finally:
        resp.close()
    return msg, int((time.time() - t0) * 1000)


def live_round_trip(url, headers, model, environ=None):
    """``get_magic(n=7)`` → tool_result ``42`` → final text containing 42, over a streaming /v1/messages."""
    from .gateway.transport import HttpClient, TransportError

    client = HttpClient(timeout=LIVE_TIMEOUT, connect_timeout=15.0, environ=environ)
    user = {"role": "user", "content": MAGIC_PROMPT}
    body = {"model": model, "max_tokens": 2048, "stream": True, "tools": [MAGIC_TOOL],
            "tool_choice": {"type": "any"}, "messages": [user]}
    try:
        m1, t1 = _post_message(client, url, headers, body)
        calls = [b for b in m1["content"] if b.get("type") == "tool_use" and b.get("name") == "get_magic"]
        if not calls:
            raise LaunchError("model did not call get_magic (stop_reason=%s, content types %s)" % (
                m1["stop_reason"], [b.get("type") for b in m1["content"]]))
        n = (calls[0].get("input") or {}).get("n")
        try:
            ok = float(n) == 7
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise LaunchError("get_magic called with n=%r, expected 7" % (n,))
        result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0].get("id"),
                                                "content": "42"}]}
        body2 = dict(body, tool_choice={"type": "auto"},
                     messages=[user, {"role": "assistant", "content": m1["content"]}, result])
        m2, t2 = _post_message(client, url, headers, body2)
    except TransportError as exc:
        raise LaunchError("connection failed: %s" % exc)
    finally:
        client.close()
    text = "".join(b.get("text", "") for b in m2["content"] if b.get("type") == "text")
    if "42" not in text:
        raise LaunchError("final answer does not mention 42: %r" % text[:200])
    u1, u2 = m1["usage"], m2["usage"]
    return {"t1_ms": t1, "t2_ms": t2, "text": text.strip(),
            "input_tokens": int(u1.get("input_tokens") or 0) + int(u2.get("input_tokens") or 0),
            "output_tokens": int(u1.get("output_tokens") or 0) + int(u2.get("output_tokens") or 0)}


# ---------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------

def run(manifest_path, argv=None):
    """Entry point used by every ``<x>-wrap.py``; returns the process exit status."""
    argv = sys.argv[1:] if argv is None else argv
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        _err("invalid launcher manifest %s" % exc)
        return 2
    try:
        return Launcher(manifest).main(argv)
    except KeyboardInterrupt:
        _err("")
        return 130
