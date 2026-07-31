"""ai-launchers shared package.

Provider-agnostic building blocks for the `grok`/`codex`/`gemini`/`deepseek`/
`kimi launch claude` launchers. Each module is standalone (no circular imports).

Modules:
  utils.py        config load/save, free-port, Windows shim, version
  key_manager.py  unified set/remove/list; env: + op:// + credentials.json
  ccr_bridge.py   compile/write claude-code-router config, daemon lifecycle
  auth_bridge.py  localhost OpenAI-compat bridge for stored-auth CLIs (grok/codex)
  scrub_proxy.py  ollama reasoning-field scrub proxy
  base_launcher.py argparse + dispatch (launch/keys/models/version/help)
"""

__version__ = "0.1.0"