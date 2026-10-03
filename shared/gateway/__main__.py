"""Standalone gateway: ``python -m <pkg> serve --preset xai[,grok] [--port N] [--routes file.json]
[--host H] [--log-file F] [--log-level L] [--shell posix|powershell|cmd]``.

Builds a RouteTable from presets (or a routes JSON file), loads API keys (``secrets.load_into`` when
available, else the presets' environment variables), starts the gateway and prints ONLY the
environment lines for Claude Code (``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` /
``ANTHROPIC_MODEL``), then blocks until Ctrl+C. ``AI_GATEWAY_TOKEN`` fixes the gateway token.
"""

import argparse
import importlib
import json
import os
import shlex
import signal
import sys
import time

from . import presets
from .config import ConfigError, RouteTable, SecretStore
from .router import RouteNotFound
from .server import Gateway
from .tracing import setup_logging

__all__ = ["main", "build_table", "load_secrets", "export_lines"]


def build_table(preset_names=None, routes_file=None):
    """RouteTable from a routes JSON file, or from comma-separated preset names."""
    if routes_file:
        with open(routes_file, "r", encoding="utf-8") as f:
            return RouteTable.from_dict(json.load(f))
    names = [n.strip() for n in (preset_names or "").split(",") if n.strip()]
    if not names:
        raise ConfigError("no presets given")
    return presets.route_table_from_presets(names)


def _secret_specs(table):
    """SecretStore name -> ["env:VAR", ...] for every api_key provider (and fallback)."""
    specs = {}
    for pid, provider in table.providers.items():
        for spec in (provider, provider.fallback):
            if spec is None or (spec.auth or {}).get("kind") != "api_key":
                continue
            name = spec.auth.get("secret") or pid
            envs = presets.secret_env_vars(pid) if pid in presets.PRESETS else []
            sources = specs.setdefault(name, [])
            for var in envs:
                if "env:" + var not in sources:
                    sources.append("env:" + var)
    return specs


def load_secrets(table, store=None, environ=None):
    """Fill a SecretStore for ``table``; returns (store, names that stayed empty)."""
    store = store if store is not None else SecretStore()
    specs = _secret_specs(table)
    loader = None
    try:
        loader = getattr(importlib.import_module(".secrets", __package__), "load_into", None)
    except ImportError:
        loader = None
    if loader is not None and environ is None:
        loader(store, specs)
    else:
        env = os.environ if environ is None else environ
        for name, sources in specs.items():
            for src in sources:
                value = env.get(src[4:]) if src.startswith("env:") else None
                if value:
                    store.set(name, value)
                    break
    return store, [n for n in specs if not store.has(n)]


def export_lines(base_url, token, model, shell="posix"):
    pairs = [("ANTHROPIC_BASE_URL", base_url), ("ANTHROPIC_AUTH_TOKEN", token)]
    if model:
        pairs.append(("ANTHROPIC_MODEL", model))
    if shell == "powershell":
        return ["$env:%s = '%s'" % (k, v.replace("'", "''")) for k, v in pairs]
    if shell == "cmd":
        return ['set "%s=%s"' % (k, v) for k, v in pairs]
    return ["export %s=%s" % (k, shlex.quote(v)) for k, v in pairs]


def _default_model(gw):
    try:
        res = gw.router.resolve("default")
    except RouteNotFound:
        return None
    return gw.table.picker_id(res.provider.id, res.model_spec)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m %s" % (__package__ or "gateway"),
                                     description="Anthropic-compatible gateway for Claude Code.")
    sub = parser.add_subparsers(dest="command")
    sp = sub.add_parser("serve", help="run the gateway in the foreground")
    src = sp.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", help="comma-separated provider presets, e.g. xai,grok")
    src.add_argument("--routes", help="route table JSON file")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=0)
    sp.add_argument("--log-file", default=None)
    sp.add_argument("--log-level", default="INFO")
    sp.add_argument("--shell", choices=("posix", "powershell", "cmd"),
                    default="powershell" if os.name == "nt" else "posix")
    args = parser.parse_args(argv)
    if args.command != "serve":
        parser.print_help(sys.stderr)
        return 2
    try:
        table = build_table(args.preset, args.routes)
        table.validate()
    except (ConfigError, KeyError, OSError, ValueError) as exc:
        sys.stderr.write("error: %s\n" % (exc,))
        return 2
    store, missing = load_secrets(table)
    for name in missing:
        sys.stderr.write("warning: no API key for %r (set %s)\n"
                         % (name, " or ".join(presets.secret_env_vars(name)) if name in presets.PRESETS
                            else "its environment variable"))
    setup_logging(args.log_level, args.log_file, store)
    gw = Gateway(table, store, host=args.host, port=args.port, token=os.environ.get("AI_GATEWAY_TOKEN") or None)
    try:
        gw.start()
    except OSError as exc:
        sys.stderr.write("error: cannot listen on %s:%s: %s\n" % (args.host, args.port, exc))
        return 1
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _interrupt)
    try:
        for line in export_lines(gw.url, gw.token, _default_model(gw), args.shell):
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    finally:
        gw.stop()


if __name__ == "__main__":
    sys.exit(main())
