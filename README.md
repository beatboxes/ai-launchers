# ai-launchers

Unified `launch claude` launchers for multiple AI providers, with shared
infrastructure (CCR bridge, key management, Ollama scrub proxy) and [Everything
Claude Code (ECC)](https://github.com/affaan-m/ECC) integration.

Five launchers, one ergonomic interface:

```
grok-wrap    launch claude [--model M] [-- <claude-args>]   # xAI Grok (API key or grok CLI)
codex-wrap   launch claude ...                               # OpenAI (API key or codex CLI)
gemini-wrap  launch claude ...                               # Google Gemini (OpenAI-compat)
deepseek-wrap launch claude ...                             # DeepSeek
kimi-wrap    launch claude ...                               # Moonshot Kimi
```

> Each launcher carries a `-wrap` suffix so its PATH shim doesn't collide with
> the native `grok` and `codex` CLI binaries.

Each routes Claude Code (`claude`) through [claude-code-router
(CCR)](https://github.com/musistudio/claude-code-router) to the chosen provider,
so you get the Claude Code UX backed by any model you have access to.

## Install (Windows)

```powershell
git clone https://github.com/beatboxes/ai-launchers.git
cd ai-launchers
.\install.ps1
# reopen terminal, then:
grok-wrap --help
```

`install.ps1` creates `.cmd` shims in `%USERPROFILE%\.ai-launchers\bin` (added to user
PATH). Reverse with `.\uninstall.ps1`.

## Install (macOS / Linux)

```bash
git clone https://github.com/beatboxes/ai-launchers.git
cd ai-launchers
./install.sh
# new shell, then:
grok-wrap --help
```

Reverse with `./uninstall.sh`.

### Prerequisites

- **Python 3.8+** (`python3` / `py -3`)
- **Node.js** + **claude-code-router**: `npm i -g @musistudio/claude-code-router`
- **Claude Code CLI**: `npm i -g @anthropic-ai/claude-code`
- A **grok** CLI / **codex** CLI (only if you want the stored-auth bridge
  fallback instead of an API key)

## Set a key

Keys are never printed. Stored in `~/.ai-launchers/credentials.json` (operator-authorized
local cache). Resolution order: env var → `op://` 1Password ref → credentials
file.

```powershell
grok-wrap keys set xai-xxxxxxxxxxxx        # or:  $env:XAI_API_KEY="..."; (no file)
grok-wrap keys list                        # shows source + redacted prefix only
grok-wrap keys remove
```

| Launcher | Env var | Default model | Base URL |
|----------|---------|---------------|----------|
| `grok-wrap` | `XAI_API_KEY` | `grok-4.5` | `https://api.x.ai/v1/chat/completions` |
| `codex-wrap` | `OPENAI_API_KEY` | `gpt-4o` | `https://api.openai.com/v1/chat/completions` |
| `gemini-wrap` | `GEMINI_API_KEY` | `gemini-2.5-pro` | `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` |
| `deepseek-wrap` | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` | `https://api.deepseek.com/chat/completions` |
| `kimi-wrap` | `MOONSHOT_API_KEY` | `kimi-k3` | `https://api.moonshot.ai/v1/chat/completions` |

> Model catalogs are current as of 2026-07. `grok models`, `gemini models`,
> etc. print the catalog; edit `<provider>/config.example.json` to add new
> models as providers release them. CCR's `api_base_url` must be the **full**
> `/chat/completions` URL (confirmed via CCR issue #94).

## Launch

```powershell
grok-wrap launch claude                            # default model, interactive
grok-wrap launch claude --model grok-4.3
gemini-wrap launch claude -- --print "hi"         # args after `--` pass to claude
grok-wrap launch claude --dry-run                  # show the CCR plan, mutate nothing
```

What `launch` does (see `shared/`):

1. resolves the API key (env / `op://` / credentials)
2. compiles a CCR `config.json` (one provider + Router roles) — the previous
   config is backed up (`config.ail-backup.<ts>.json`, last 6 kept) and
   restored on exit
3. starts the CCR daemon on a free port (configurable via
   `~/.ai-launchers/config.json` → `router.port`; default 3456)
4. runs `claude` with `ANTHROPIC_BASE_URL` pointed at CCR
5. on exit (incl. Ctrl+C): stops CCR, restores the clean config (no orphan
   daemon, no leftover mutation of `~/.claude/settings.json`)

## `shared/` package

| Module | Responsibility |
|--------|----------------|
| `utils.py` | config I/O, free-port bind w/ retry, env diff filter, AIL_HOME |
| `key_manager.py` | set / remove / list; env → `op://` → credentials (never echoes keys) |
| `ccr_bridge.py` | compile/write CCR config, daemon lifecycle + `atexit` stop, backup prune |
| `auth_bridge.py` | localhost OpenAI-compat bridge for stored-auth CLIs (grok/codex/opencode) |
| `scrub_proxy.py` | strips `reasoning_effort` for non-thinking Ollama models |
| `base_launcher.py` | argparse + dispatch: `launch` / `keys` / `models` / `--version` |

Bug fixes carried in this release:
C2 grok alias collapse · C3 hardcoded port · H1 dry-run mutation · H2 hardcoded
exe · H3 hardcoded debug dir · M2 orphan CCR · M3 port TOCTOU · M7 single-thread
bridge · M8 cp1252 mojibake · M9 opencode argv limit · M10 dead-ollama fallback ·
L5 backup prune · L6 silenced logs · L7 shared scrub log · L11 memoized `op read`.

## ECC integration

If [Everything Claude Code](https://github.com/affaan-m/ECC) is installed
(minimal profile: `install.ps1 --profile minimal --target claude` from the ECC
repo), its rules land in `~/.claude/rules/ecc/{common,python,...}` and are
picked up automatically by every `claude` session this monorepo launches — no
extra wiring needed.

## Troubleshooting

- **`ccr exited early`** — wrong `api_base_url` (must include `/chat/completions`)
  or bad/missing key. Run `<provider> launch claude --dry-run` to inspect the
  generated CCR config.
- **`No <provider> API key found`** — `<provider> keys set <KEY>` (or export
  the env var, or add an `op://` ref to `~/.ai-launchers/config.json`).
- **`claude CLI not found`** — `npm i -g @anthropic-ai/claude-code`.
- **Dry-run still wrote something** — it shouldn't (H1). If it does, file an
  issue; dry-run compiles in-memory only and never writes CCR config.

## License

MIT — see [LICENSE](LICENSE).