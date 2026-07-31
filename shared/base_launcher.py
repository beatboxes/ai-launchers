"""Base launcher: argparse + dispatch shared by all 6 provider launchers.

Subclasses (grok/codex/gemini/deepseek/kimi) supply a manifest dict (loaded
from their config.example.json, merged with the user's ~/.ai-launchers/config.json):

  manifest = {
    "provider": "grok",
    "version": "0.1.0",
    "base_url": "https://api.x.ai/v1",
    "env_var": "XAI_API_KEY",
    "op_ref":  "op://Personal/xAI/credential",   # optional
    "models":  ["grok-4.5", "grok-4.3", ...],
    "default_model": "grok-4.5",
    "cli_target": "grok",          # optional: 'grok'|'codex' for stored-auth bridge fallback
    "roles": {"default": "grok,grok-4.5"},  # optional CCR role overrides
  }

Dispatch:
  <provider> launch claude [--model M] [--dry-run] [-- <claude-args>]
  <provider> keys set|remove|list
  <provider> models
  <provider> --version
  <provider> --help

H1 honored by design: we never touch ~/.claude/settings.json or ~/.claude.json —
the launched `claude` inherits ANTHROPIC_BASE_URL via env per-process only.
"""
import json
import os
import shutil
import subprocess
import sys
from . import key_manager, ccr_bridge
from .utils import VERSION, load_config, find_free_port, env_filter_changed

CLAUDE_BIN = shutil.which("claude") or shutil.which("claude.cmd") or shutil.which("claude.exe")


class Launcher:
    def __init__(self, manifest: dict):
        self.m = manifest
        self.cfg = load_config()

    def _name(self) -> str:
        """User-facing launcher name (may differ from CCR provider name)."""
        return self.m.get("name", self.m["provider"])

    # ---- merged provider cfg (example defaults <- user config overrides) ----
    def _provider_cfg(self) -> dict:
        merged = {
            "base_url": self.m.get("base_url"),
            "env_var": self.m.get("env_var"),
            "op_ref": self.m.get("op_ref"),
        }
        user = (self.cfg.get("providers") or {}).get(self.m["provider"]) or {}
        merged.update({k: v for k, v in user.items() if k in ("base_url", "env_var", "op_ref")})
        return merged

    def _models(self) -> list:
        return self.m.get("models") or []

    def _default_model(self) -> str:
        return self.m.get("default_model")

    def _router_port(self) -> int:
        return int((self.cfg.get("router") or {}).get("port", 3456))  # C3

    # ---- dispatch ----
    def main(self, argv):
        if not argv:
            self._print_help()
            return 0
        cmd = argv[0]
        rest = argv[1:]
        if cmd in ("-h", "--help", "help"):
            self._print_help(); return 0
        if cmd in ("-v", "--version", "version"):
            print(f"{self._name()} launcher {self.m.get('version', VERSION)}"); return 0
        if cmd == "keys":
            return self._cmd_keys(rest)
        if cmd == "models":
            return self._cmd_models()
        if cmd == "launch":
            return self._cmd_launch(rest)
        print(f"unknown command: {cmd}", file=sys.stderr)
        self._print_help()
        return 2

    # ---- keys ----
    def _cmd_keys(self, rest):
        sub = rest[0] if rest else "list"
        if sub == "set":
            val = None
            if len(rest) >= 2:
                val = rest[1]
            else:
                # read from stdin (so `echo $KEY | grok keys set` / paste works)
                try:
                    val = sys.stdin.readline().strip()
                except Exception:
                    val = None
            if not val:
                print("no key provided (usage: <provider> keys set <KEY>  or pipe via stdin)", file=sys.stderr)
                return 2
            key_manager.set_key(self._name(), val)
            print(f"{self._name()}: key stored in ~/.ai-launchers/credentials.json")
            return 0
        if sub == "remove":
            existed = key_manager.remove_key(self._name())
            print(f"{self._name()}: {'removed' if existed else 'no stored key'}")
            return 0
        if sub == "list":
            entries = key_manager.list_keys([self._name()])
            for e in entries:
                src = e["source"]
                prefix = e["prefix"] or ""
                print(f"{e['provider']}: {src} {prefix}")
            return 0
        print(f"unknown keys subcommand: {sub} (set|remove|list)", file=sys.stderr)
        return 2

    # ---- models ----
    def _cmd_models(self):
        print(f"== {self._name()} models (catalog {self.m.get('version','?')}) ==")
        default = self._default_model()
        for m in self._models():
            mark = " (default)" if m == default else ""
            print(f"  {m}{mark}")
        return 0

    # ---- launch ----
    def _cmd_launch(self, rest):
        # split: ours vs claude passthrough at first bare "--"
        passthrough = []
        if "--" in rest:
            i = rest.index("--")
            own, passthrough = rest[:i], rest[i + 1:]
        else:
            own, passthrough = rest, []

        model = self._default_model()
        dry_run = False
        it = iter(own)
        for tok in it:
            if tok == "--model":
                try:
                    model = next(it)
                except StopIteration:
                    print("--model requires a value", file=sys.stderr); return 2
            elif tok.startswith("--model="):
                model = tok.split("=", 1)[1]
            elif tok == "--dry-run":
                dry_run = True
            elif tok in ("claude",):  # `launch claude` — accept and ignore
                continue
            else:
                print(f"unknown launch flag: {tok}", file=sys.stderr); return 2

        if CLAUDE_BIN is None and not dry_run:
            print("claude CLI not found on PATH (install: npm i -g @anthropic-ai/claude-code)", file=sys.stderr)
            return 2

        # resolve key + decide transport
        pcfg = self._provider_cfg()
        # key_manager resolves by launcher name (_name); credentials are keyed
        # by launcher name, NOT by the CCR provider name.
        resolve_cfg = {"providers": {self._name(): pcfg}}
        key, source = key_manager.resolve_key(self._name(), resolve_cfg)

        base_url = pcfg.get("base_url")
        cli_target = self.m.get("cli_target")
        no_key = bool(self.m.get("no_key"))
        bridge_proc = None
        effective_model = model

        if key is None and no_key:
            # local provider (e.g. ollama) needs no key — use a dummy.
            key, source = "local-no-key", "no-key"
            effective_model = model

        if key is None:
            # try CLI bridge fallback (grok/codex)
            if cli_target and shutil.which(cli_target):
                bport = find_free_port(0)
                bridge_proc = subprocess.Popen(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "auth_bridge.py"),
                     str(bport), cli_target],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    creationflags=0x08000000 if sys.platform == "win32" else 0,
                )
                base_url = f"http://127.0.0.1:{bport}/v1/chat/completions"
                effective_model = f"ail-{cli_target}-{model.replace('.', '-')}"
                source = f"cli-bridge:{cli_target}"
            else:
                print(
                    f"No {self._name()} API key found and no {cli_target or 'cli'} bridge available.\n"
                    f"Set one with:  {self._name()} keys set <KEY>\n"
                    f"  (env: {pcfg.get('env_var') or '(none)'})",
                    file=sys.stderr,
                )
                return 2

        # compile CCR config
        ccr_obj = ccr_bridge.compile_ccr_config(
            self.m["provider"], base_url, key, effective_model,
            self._models(), self.m.get("roles") or {},
        )
        # C3: router roles default must point at our provider/model (compile does this)

        if dry_run:
            print(f"[dry-run] name={self._name()} provider={self.m['provider']} model={effective_model} base={base_url} via={source}")
            print(f"[dry-run] CCR config would be:\n{json.dumps(ccr_obj, indent=2)}")
            print("[dry-run] no files mutated, no daemon started (H1).")
            if bridge_proc and bridge_proc.poll() is None:
                bridge_proc.terminate()
            return 0

        ccr_bridge.write_ccr_config(ccr_obj, self.cfg)
        try:
            port = ccr_bridge.start_ccr(self._router_port())
        except Exception as e:
            print(f"CCR start failed: {e}", file=sys.stderr)
            ccr_bridge.restore_clean_config()
            if bridge_proc and bridge_proc.poll() is None:
                bridge_proc.terminate()
            return 3

        env = dict(os.environ)
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"  # C3
        env["ANTHROPIC_API_KEY"] = "ccr-local"  # CCR ignores but field required
        # surface what changed (L3: value-diff filter)
        changed = env_filter_changed(env)
        print(f"[launch] {self._name()} model={effective_model} via={source} ccr=127.0.0.1:{port}", flush=True)

        rc = 0
        try:
            proc = subprocess.run([CLAUDE_BIN] + passthrough, env=env)
            rc = proc.returncode
        except KeyboardInterrupt:
            rc = 130
        except Exception as e:
            print(f"claude failed: {e}", file=sys.stderr); rc = 4
        finally:
            ccr_bridge.stop_ccr()
            if bridge_proc and bridge_proc.poll() is None:
                bridge_proc.terminate()
            ccr_bridge.restore_clean_config()
        return rc

    # ---- help ----
    def _print_help(self):
        p = self._name()
        print(f"""{p} — launch claude routed to {self.m.get('base_url','<provider API>')}

Usage:
  {p} launch claude [--model <M>] [--dry-run] [-- <claude-args>]
  {p} keys set <KEY>        store API key in ~/.f/credentials.json
  {p} keys remove           remove stored key
  {p} keys list             show key source (never prints the key)
  {p} models                list catalog models
  {p} --version
  {p} --help

Default model: {self._default_model()}
Env var:       {self.m.get('env_var','(none)')}
CLI bridge:    {self.m.get('cli_target','(none) — API key required')}

Everything after `--` is passed straight to `claude`.
""")