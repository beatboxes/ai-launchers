"""Localhost OpenAI-compat bridge for stored-auth CLIs (grok / codex / opencode).

Binds 127.0.0.1 only. Dummy token "local-bridge-dummy". Used when a provider
has a logged-in CLI but no API key (the stored-auth pattern).

Patterns ported from the bug-fixed local_auth_bridge.py (Phase 3):
  H2   which()-based exe discovery (no hardcoded ai-launchers path)
  C2   resolve_model_alias inverts ail-grok-* -> real grok id (no grok-build collapse)
  M7   ThreadingHTTPServer (concurrent probes + completions)
  M8   subprocess encoding="utf-8" (not cp1252 on Windows)
  M9   opencode --prompt-file (argv-length limit on Windows)
  L6   log_message routes to stderr (not silenced)
"""
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

# H2: resolve CLIs from PATH, never hardcoded.
GROK_EXE = shutil.which("grok") or shutil.which("grok.cmd") or shutil.which("grok.exe")
CODEX_EXE = shutil.which("codex") or shutil.which("codex.cmd") or shutil.which("codex.exe")
OPENCODE_EXE = shutil.which("opencode") or shutil.which("opencode.cmd") or shutil.which("opencode.exe")

# C2: reverse the ail-grok-* alias map back to real grok model ids.
_GROK_REAL_ALIASES = {
    "ail-grok-4-5": "grok-4.5",
    "ail-grok-4-3": "grok-4.3",
    "ail-grok-4-20-0309-reasoning": "grok-4.20-0309-reasoning",
    "ail-grok-4-20-0309-non-reasoning": "grok-4.20-0309-non-reasoning",
    "ail-grok-build-0-1": "grok-build-0.1",
}


def resolve_model_alias(model: str) -> str:
    """Map a local alias id back to the real CLI model id. Raises
    ValueError if unresolvable (C2: no silent collapse to invalid grok-build).

    base_launcher builds effective_model = ail-<cli_target>-<model_dashed>.
    For grok, cli_target is "grok" AND the model itself starts with "grok"
    (e.g. grok-4.5), so the alias is "ail-grok-grok-4-5" (the "grok" prefix is
    doubled). Normalize that doubled form back to the single-grok form the
    alias map + generic path expect, so it resolves to "grok-4.5" instead of
    the invalid "grok-grok.4-5" the grok CLI rejects (BUG B). The codex path
    is unaffected: its model ids (gpt-4o) do not start with "codex".
    """
    # BUG B: collapse doubled "ail-grok-grok-" -> "ail-grok-" before lookup.
    if model.startswith("ail-grok-grok-"):
        model = "ail-grok-" + model[len("ail-grok-grok-"):]
    if model in _GROK_REAL_ALIASES:
        return _GROK_REAL_ALIASES[model]
    if model.startswith("ail-grok-"):
        rest = model[len("ail-grok-"):]
        if rest in ("4-5", "4-3", "4-1"):
            return "grok-" + rest.replace("-", ".", 1)
        if rest in ("4-20-0309-reasoning", "4-20-0309-non-reasoning"):
            return "grok-" + rest.replace("-", ".", 1)
        return "grok-" + rest.replace("-", ".", 1)
    if model.startswith("ail-codex-"):
        return model[len("ail-codex-"):]
    if model.startswith("ail-opencode-"):
        return "opencode/" + model[len("ail-opencode-"):]
    return model


def _strip_ansi(txt):
    if not txt:
        return txt
    return re.sub(r'\x1b\[[0-9;]*m', '', txt)


_WRAPPER_PREFIX = (
    "Respond directly to the following request. "
    "Do not enumerate files, skills, plugins, or workspace. "
    "Do not ask for clarification. "
    "Do not write code unless explicitly asked. "
    "Your response should answer the request below concisely:\n\n"
)

# BUG D: claude injects its own <system-reminder>...</system-reminder> context
# (CLAUDE.md, skills, memory) as a leading part of the user message content list.
# Forwarding it to the native CLI drowns the real question and leaks the
# operator's private manual to the provider. Module constant so the regression
# test exercises the SAME pattern the handler applies.
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>\s*", re.DOTALL)


def run_cli(prompt, model, target):
    """target in {'grok','codex','opencode'}."""
    exe = {"grok": GROK_EXE, "codex": CODEX_EXE, "opencode": OPENCODE_EXE}.get(target)
    if not exe:
        return f"[bridge error] {target} CLI not found on PATH (install it and log in)"
    prompt = _WRAPPER_PREFIX + prompt
    tmp_path = None
    try:
        if target == "grok":
            fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="ail_grok_", text=True)
            os.write(fd, prompt.encode("utf-8"))
            os.close(fd)
            args = [exe, "--prompt-file", tmp_path, "-m", model,
                    "--no-alt-screen", "--output-format", "plain"]
            stdin_arg, input_arg = subprocess.DEVNULL, None
        elif target == "opencode":
            # M9: prompt via temp file, not argv (Windows 32K argv limit)
            fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="ail_opencode_", text=True)
            os.write(fd, prompt.encode("utf-8"))
            os.close(fd)
            args = [exe, "run", "--prompt-file", tmp_path, "--model", model]
            stdin_arg, input_arg = subprocess.DEVNULL, None
        else:  # codex
            # BUG C: the bridge runs only when no API key is set (base_launcher
            # enters it iff key is None), so codex is always on a ChatGPT-account
            # auth. Every id in the codex manifest (gpt-4o, gpt-5-codex, codex-mini,
            # ...) is an API-key-only model that ChatGPT-account codex rejects with
            # rc=1 ("'X' model is not supported when using Codex with a ChatGPT
            # account"). Omit -m so codex uses its own default (the ChatGPT-account
            # model, or whatever the operator set in ~/.codex/config.toml). The
            # direct/API-key path is unaffected (it never reaches the bridge).
            args = [exe, "exec", "--skip-git-repo-check", "-"]
            stdin_arg, input_arg = None, prompt
        proc = subprocess.run(
            args, capture_output=True, timeout=120,
            encoding="utf-8", errors="replace", text=True,  # M8: force utf-8
            stdin=stdin_arg, input=input_arg,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
        if proc.returncode == 0:
            out = _strip_ansi(proc.stdout.strip())
            if target == "opencode":
                lines = [l for l in out.splitlines() if l.strip() and not l.startswith(">")]
                return "\n".join(lines) if lines else out
            return out
        out = (proc.stdout or "") + (proc.stderr or "")
        return _strip_ansi(f"[cli error rc={proc.returncode}] {out.strip()}")
    except subprocess.TimeoutExpired:
        return f"[bridge error] {target} timed out after 120s"
    except Exception as e:
        return f"[bridge error] {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


class BridgeHandler(http.server.BaseHTTPRequestHandler):
    TARGET = "grok"  # set per-instance via argv

    def do_GET(self):
        if self.path == "/v1/models":
            data = json.dumps({"object": "list", "data": []}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            req = json.loads(body)
            model = req.get("model", "")
            messages = req.get("messages", [])
            prompt = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    c = m.get("content", "")
                    if isinstance(c, str):
                        prompt = c
                    elif isinstance(c, list):
                        parts = []
                        for part in c:
                            if isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                parts.append(part)
                        prompt = "\n".join(parts)
                    else:
                        prompt = str(c)
                    break
            # BUG D: claude bundles its own <system-reminder>...</system-reminder>
            # context injection (CLAUDE.md, skills, memory) into the user message
            # content list as a leading part. Forwarding that to the native CLI
            # (grok/codex/opencode) both drowns the actual question (the model
            # echoes an operand or returns 0 because it cannot find the prompt
            # in 60+ KB of operator manual) AND leaks the operator's private
            # CLAUDE.md (1Password paths, host IPs, infra) to the provider's
            # cloud. The system-reminder is claude-internal context, never part
            # of the user's question. Strip ALL such blocks (DOTALL, repeated)
            # before forwarding; keep everything else the user wrote verbatim.
            prompt = _SYSTEM_REMINDER_RE.sub("", prompt)
            req_model = model
            if model.startswith("ail-grok-") or model.startswith("ail-codex-") or model.startswith("ail-opencode-"):
                try:
                    model = resolve_model_alias(model)
                except ValueError as ve:
                    self.send_error(400, f"model alias '{req_model}' not resolvable: {ve}")
                    return
            if model != req_model:
                print(f"[bridge alias] requested={req_model} mapped={model}", file=sys.stderr)
            content = run_cli(prompt, model, self.TARGET)
            resp = {
                "id": "chatcmpl-localbridge",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req_model,
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}]
            }
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, fmt, *args):  # L6: stderr, not silenced
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


def run_bridge(port: int, target: str):
    """Start the localhost bridge. target in {'grok','codex','opencode'}."""
    if len(sys.argv) >= 3:
        target = sys.argv[2]
    BridgeHandler.TARGET = target
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), BridgeHandler)  # M7
    server.daemon_threads = True
    print(f"auth_bridge on 127.0.0.1:{port} target={target}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: auth_bridge.py <port> [<grok|codex|opencode>]", file=sys.stderr)
        sys.exit(2)
    run_bridge(int(sys.argv[1]), sys.argv[2] if len(sys.argv) >= 3 else "grok")