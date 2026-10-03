# ai-launchers

Run **Claude Code** with other providers' models as its brain — with streaming and full tool use
(Read/Edit/Bash, MCP tools, subagents) — through five small launchers:

```
grok-wrap     launch claude    # xAI Grok: API key, or your `grok login`
codex-wrap    launch claude    # OpenAI GPT: API key, or your ChatGPT/Codex login
gemini-wrap   launch claude    # Google Gemini: API key, or Vertex AI via gcloud ADC
deepseek-wrap launch claude    # DeepSeek: official Anthropic-compatible endpoint
kimi-wrap     launch claude    # Moonshot Kimi: official Anthropic-compatible endpoint
```

Each launcher has a `-wrap` suffix so its shim never shadows the native `grok`/`codex` CLIs.

## How it works

There is **no claude-code-router** any more. Every launcher starts a small stdlib-only Python gateway
(`shared/gateway/`, see its `DESIGN.md`) **in-process** on `127.0.0.1` (random port, random per-launch
bearer token), then runs `claude` with `ANTHROPIC_BASE_URL` pointing at it. The gateway speaks the
Anthropic Messages API to Claude Code and translates every request to the provider's native dialect
(OpenAI chat / Responses, Gemini `streamGenerateContent`, …): SSE streaming with heartbeats, tool calls,
thinking, images/PDFs in tool results, tool-name and JSON-schema fix-ups, proper error statuses (Claude
Code retries transient errors and stops on terminal quota errors). DeepSeek and Kimi already speak the
Anthropic API, so for them the gateway only forwards the request to the vendor's official endpoint with
your key attached. The gateway stops when `claude` exits — there is no daemon to clean up.

Claude Code's model picker (`/model`) lists every model the gateway can route, as
`claude-via-<transport>,<model>` ids (the `claude-via-` prefix is required by Claude Code's gateway model
discovery). `claude-via-background` serves Claude Code's small background requests.

**No provider key or login token ever enters Claude Code's environment or argv** — for every launcher
the gateway holds credentials in memory and Claude Code only gets the random per-launch gateway token,
so nothing Claude Code runs (its Bash tool, hooks, MCP servers) can read your keys. The provider's key
variables (e.g. `DEEPSEEK_API_KEY`) are removed from Claude Code's environment too. The launcher also
never writes `~/.claude.json` or `~/.claude/settings.json`, and never leaves anything running after
`claude` exits.

Inherited `ANTHROPIC_*` / `CLAUDE_CODE_USE_*` variables that would hijack the session are cleared or
overridden, and `ANTHROPIC_API_KEY` is set to empty so Claude Code never asks to approve a key.

## Transports

Each launcher tries its transports **in order — API key first** — and uses every available one
(the first is the default; the others are selectable with `/model` or `--model`). `--auth` restricts
the choice.

| Launcher | Mode | Transports (in order) | Default / background model |
|---|---|---|---|
| `grok-wrap` | gateway | `xai` API key (`XAI_API_KEY`, chat/completions) → `grok` login (Grok CLI proxy, auto-fallback to api.x.ai) | `grok-4.7` / `grok-4.20-0309-non-reasoning` |
| `codex-wrap` | gateway | `openai` API key (`OPENAI_API_KEY`, Responses API) → `codex` ChatGPT login (Codex backend) | `gpt-5.5` / `gpt-5.4-mini` (key), `gpt-5.5` at low effort (login) |
| `gemini-wrap` | gateway | `gemini` API key (`GEMINI_API_KEY` or `GOOGLE_API_KEY`) → `gemini-vertex` Vertex AI via gcloud ADC | `gemini-3.1-pro-preview` / `gemini-3.8-flash` |
| `deepseek-wrap` | gateway | `deepseek` API key (`DEEPSEEK_API_KEY`) → `https://api.deepseek.com/anthropic` (passthrough) | `deepseek-v4-pro[1m]` / `deepseek-v4-flash[1m]`, plus `CLAUDE_CODE_EFFORT_LEVEL=max` |
| `kimi-wrap` | gateway | `kimi` API key (`MOONSHOT_API_KEY` or `KIMI_API_KEY`) → `https://api.moonshot.ai/anthropic` (passthrough) | `kimi-k3[1m]` / `kimi-k2.7-code` |

API keys are resolved (in parallel) from: the env vars above → an `op://` 1Password reference
(`providers.<transport>.op_ref`) → `~/.ai-launchers/credentials.json`. `codex-wrap` also picks up a key
that `codex login --with-api-key` stored in `~/.codex/auth.json`, and `grok-wrap` an API-key entry in
`~/.grok/auth.json`.

### Login and ADC transports

- **Codex (ChatGPT plan)** — `codex login` (Codex CLI). Credentials must be file-based
  (`cli_auth_credentials_store = "file"` in `~/.codex/config.toml`; `$CODEX_HOME` is honoured). Tokens are
  refreshed under a file lock and written back atomically, so the Codex CLI and the launcher can run at
  the same time.
- **Grok (Grok Build login)** — `grok login`. Uses `~/.grok/auth.json` (`$GROK_AUTH_PATH`/`$GROK_HOME`
  honoured), same locking/refresh rules; if the Grok CLI proxy rejects the client it automatically falls
  back to `api.x.ai` with the same token.
- **Vertex AI (Gemini)** — `gcloud auth application-default login`, then set `GOOGLE_CLOUD_PROJECT`
  (optional `GOOGLE_CLOUD_LOCATION`, default `global`). Service-account ADC files work too
  (`GOOGLE_APPLICATION_CREDENTIALS`). The Gemini CLI's consumer login is **not** used.

> **Terms of service.** The `codex` and `grok` login transports send your subscription login to the
> endpoints built for those vendors' own CLIs. That may be subject to the provider's terms; the launcher
> always prefers an API key when one is configured and shows a one-time notice the first time a login
> route is used (recorded in `~/.ai-launchers/state.json`). Use `--auth api-key` to never use them.

## Install

macOS / Linux:

```bash
git clone https://github.com/beatboxes/ai-launchers.git
cd ai-launchers
./install.sh            # shims in ~/.ai-launchers/bin, PATH line tagged "# ai-launchers"
grok-wrap doctor
```

`install.sh` pins the shims to the absolute path of a Python ≥ 3.8 and adds one tagged PATH line to
`~/.bashrc` (if present) and `~/.zshrc` (if present, or created when zsh is your login shell), otherwise
`~/.profile`; fish users get the `fish_add_path` command to run. `./uninstall.sh` removes the shims and
the tagged lines.

Windows (PowerShell):

```powershell
git clone https://github.com/beatboxes/ai-launchers.git
cd ai-launchers
.\install.ps1           # .cmd shims in %USERPROFILE%\.ai-launchers\bin + user PATH
grok-wrap doctor        # in a new terminal
```

`install.ps1` resolves the interpreter once (`py -3`, else a real `python.exe` — the Microsoft Store
alias stub is rejected) and writes it into the shims. The committed `*/<name>-wrap.cmd` files work
without installing (they prefer `py -3` and fall back to `python`). `.\uninstall.ps1` reverses it.

Prerequisites: **Python 3.8+** (stdlib only — nothing to `pip install`) and **Claude Code**
(`npm i -g @anthropic-ai/claude-code`; tested with 2.1.x). Node.js is only needed for Claude Code itself.

## Usage

```bash
grok-wrap keys set                      # prompts (hidden); or: echo "$KEY" | grok-wrap keys set
grok-wrap launch claude                 # default model, interactive
grok-wrap launch claude --model grok-4.3
codex-wrap launch claude --auth login   # force the ChatGPT login transport
gemini-wrap launch claude -- -p "summarise README.md"   # args after -- go to claude unchanged
grok-wrap launch claude --dry-run       # print route table, env changes, argv; start/write nothing
grok-wrap models                        # catalog + discovered models with their picker ids
grok-wrap models --refresh --json
grok-wrap doctor --live                 # end-to-end tool-call check (see below)
```

`launch claude` options: `--model M` (bare id like `grok-4.3`, `transport,model`, or a picker id),
`--auth auto|api-key|login|adc`, `--port N` (fixed gateway port; default random), `--dry-run`, `--debug`
(DEBUG log level, JSONL trace in `~/.ai-launchers/logs/`, env diff printed before start).

`keys set KEY` also works but warns that the key may land in your shell history. `keys list` shows
which source is active (env var / `op://` / credentials.json) with a redacted hint, never the key.
`credentials.json` is written atomically with owner-only permissions (0600; owner-only ACL on Windows).

`models` shows the built-in catalog plus models discovered from the provider's free list endpoint
(cached 6 h in `~/.ai-launchers/cache/models.json`, refreshed in parallel with a 3 s timeout; `--refresh`
forces it). Discovered models also appear in Claude Code's `/model` picker.

### Configuration (optional)

`~/.ai-launchers/config.json` (`$AIL_HOME` relocates the whole directory):

```json
{
  "gateway": {"port": 0},
  "providers": {
    "xai":    {"op_ref": "op://Personal/xAI/credential", "default_model": "grok-4.6"},
    "openai": {"env": ["OPENAI_API_KEY", "MY_OPENAI_KEY"]},
    "gemini-vertex": {"options": {"project": "my-project", "location": "us-central1"}},
    "deepseek": {"default_model": "deepseek-v4-flash"}
  }
}
```

Per transport id you may override `base_url`, `env`, `op_ref`, `models` (a list replaces the catalog),
`default_model`, `background_model` (bare model ids; the `[1m]` picker suffix is added automatically for
1M-context models), `options` and `list_url` (the model-list endpoint used by `models`). v0.1 full
`…/chat/completions` URLs are ignored with a warning.

## Doctor

`<launcher> doctor` checks the `claude` binary and version, Python, config/credential files and
permissions, every transport (key source, login state, ADC project/location), active proxy variables,
upstream overrides, and warns when `~/.claude/settings.json` sets `env.ANTHROPIC_*` or `apiKeyHelper`
(those override the launcher's environment).

`doctor --live [--model M]` starts the gateway and, for each available route (or just `M`), sends one
streaming Anthropic request through `http://127.0.0.1:<port>/v1/messages` — exactly the path Claude Code
uses — offering a `get_magic(n: int)` tool with `tool_choice: any`. It checks that the model calls
`get_magic` with `n=7`, returns the tool result `42`, and checks that the final answer contains 42; it
prints latency and token usage. **Each route costs two tiny paid requests.** Exit status is non-zero on
any failure.

## Troubleshooting

- **401 / "run `codex login`" / "run `grok login`"** — the login expired or its refresh token was
  revoked: log in again with the vendor CLI. For API keys: `keys list`, then fix the env var or
  `keys set`. Logs: `~/.ai-launchers/logs/<launcher>.log`.
- **429 that Claude Code does not retry** — a terminal quota (ChatGPT plan usage limit, Gemini daily
  quota, exhausted credits); the message includes the reset time when the provider sends one. Wait, use
  another transport (`/model claude-via-<other>,…`) or `--auth api-key`. Transient 429s are retried by
  Claude Code automatically.
- **Behind a proxy** — `HTTPS_PROXY`/`NO_PROXY` are honoured for upstream calls. The launcher appends
  `127.0.0.1,localhost,::1` to `NO_PROXY` for Claude Code so it reaches the local gateway directly; if
  your proxy setup intercepts loopback traffic anyway, add those hosts to `NO_PROXY` yourself.
- **The wrong model / an Anthropic login prompt shows up** — check `doctor` for `settings.json` warnings
  (`env.ANTHROPIC_*`, `apiKeyHelper`).
- **"model … is not routable via this launcher"** — run `<launcher> models` and use one of the listed
  ids or picker ids.
- **Port in use with `--port`/`gateway.port`** — omit it to get a random free port.
- **`claude` not found** — `npm i -g @anthropic-ai/claude-code`, or point `AI_LAUNCHERS_CLAUDE_BIN` at the
  binary.

## Development

```bash
python3 -m unittest discover -s tests -v                # unit + integration tests (no network, no keys)
RUN_E2E=1 python3 -m unittest discover -s tests -v      # + end-to-end runs of the real Claude Code
```

The E2E tests drive the installed `claude` against quirk-enforcing local mock upstreams (temporary
HOME, no real keys). `tests/launchers/` covers the launcher layer (manifests, keys, transport
selection, dry-run purity, a stub-`claude` launch, `doctor --live` against mocks); `tests/gateway/`
covers the gateway. The gateway is vendored byte-identical into fry-launch-claude by
`tools/vendor_gateway.py`.

## ECC integration

If [Everything Claude Code](https://github.com/affaan-m/ECC) is installed, its rules in
`~/.claude/rules/` are picked up by every `claude` session these launchers start — no extra wiring.

## License

MIT — see [LICENSE](LICENSE).
