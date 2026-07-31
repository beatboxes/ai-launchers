"""Ollama reasoning-field scrub proxy for CCR / local-ollama routing.

For models Ollama reports as NOT supporting `thinking` (POST /api/show), strip
the fields Ollama rejects (reasoning_effort / reasoning.effort) before forwarding.
Thinking-capable models pass through untouched.

Pattern ported from the bug-fixed ollama_scrub_proxy.py (Phase 3):
  L7  per-instance log keyed by bound port (concurrent proxies don't clobber)
"""
import http.server
import json
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request

OLLAMA_BASE = "http://localhost:11434"
CAP_TIMEOUT = 10
UPSTREAM_TIMEOUT = 120
# L7: per-instance log path (set in main() after port parse).
LOG_PATH = os.path.join(tempfile.gettempdir(), "ail_scrub_proxy.log")
_CAP_CACHE: dict = {}


def _log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    print(line, file=sys.stderr, flush=True)


def _set_log_path_for_port(port):  # L7
    global LOG_PATH
    if port:
        LOG_PATH = os.path.join(tempfile.gettempdir(), f"ail_scrub_proxy_{port}.log")


def _model_supports_thinking(model):
    if model in _CAP_CACHE:
        return _CAP_CACHE[model]
    supports = False
    try:
        payload = json.dumps({"model": model}).encode("utf-8")
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/show", data=payload,
                                    headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=CAP_TIMEOUT) as resp:
            info = json.loads(resp.read().decode("utf-8", "replace"))
        caps = info.get("capabilities") or info.get("model_info", {}).get("capabilities") or []
        supports = "thinking" in caps
    except Exception as e:
        _log(f"capability-probe-failed model={model!r} err={e} -> assume non-thinking")
    _CAP_CACHE[model] = supports
    return supports


def _scrub_body(body):
    try:
        obj = json.loads(body.decode("utf-8"))
    except Exception:
        return body, False
    model = obj.get("model", "")
    is_stream = bool(obj.get("stream", False))
    if not model:
        return body, is_stream
    supports = _model_supports_thinking(model)
    stripped = False
    if not supports:
        if "reasoning_effort" in obj:
            del obj["reasoning_effort"]; stripped = True
        reasoning = obj.get("reasoning")
        if isinstance(reasoning, dict):
            if "effort" in reasoning:
                del reasoning["effort"]; stripped = True
            if not reasoning:
                del obj["reasoning"]; stripped = True
    _log(f"scrub model={model!r} supports_thinking={supports} stripped={stripped} stream={is_stream}")
    if stripped:
        return json.dumps(obj).encode("utf-8"), is_stream
    return body, is_stream


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            content_len = int(self.headers.get("Content-Length", 0))
        except Exception:
            self.send_error(400); return
        scrubbed, is_stream = _scrub_body(self.rfile.read(content_len))
        url = f"{OLLAMA_BASE}/v1/chat/completions"
        fwd = {"Content-Type": self.headers.get("Content-Type", "application/json"),
               "Accept": self.headers.get("Accept", "text/event-stream")}
        try:
            req = urllib.request.Request(url, data=scrubbed, headers=fwd, method="POST")
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as upstream:
                self.send_response(upstream.status)
                self.send_header("Connection", "close")
                for h, v in upstream.headers.items():
                    if h.lower() in {"content-length", "transfer-encoding", "connection"}:
                        continue
                    self.send_header(h, v)
                self.end_headers()
                if is_stream:
                    for line in upstream:
                        self.wfile.write(line); self.wfile.flush()
                        if b"[DONE]" in line:
                            break
                else:
                    while True:
                        chunk = upstream.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(chunk); self.wfile.flush()
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(err_body)
        except socket.timeout:
            self.send_error(504, "upstream timeout")
        except Exception as e:
            self.send_error(500, f"proxy error: {e}")

    def log_message(self, fmt, *args):
        pass


def main():
    if len(sys.argv) < 2:
        print("usage: scrub_proxy.py <port>", file=sys.stderr); sys.exit(2)
    port = int(sys.argv[1])
    _set_log_path_for_port(port)  # L7
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"scrub_proxy on 127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()