# ai-gateway — design spec (source of truth for implementers)

Stdlib-only (Python ≥ 3.8), Windows/macOS/Linux. Speaks the **Anthropic Messages API** to Claude Code and
translates to provider dialects so every model gets streaming + full tool calling. Canonical copy lives in
`ai-launchers/shared/gateway/`; `ai-launchers/tools/vendor_gateway.py` copies it byte-identical into
`fry-launch-claude/fry_gateway/`. **All intra-package imports are relative** (`from . import x`, `from ..events import y`).
Never edit the fry copy by hand.

Verified external facts referenced below live in `/root/.claude/plans/expressive-orbiting-quail.md` Appendix A
(Claude Code 2.1.288 behaviour, provider quirks, ChatGPT Codex backend, Grok Build login, Gemini/Vertex).

---

## 0. Claude Code 2.1.288 behaviours the gateway must honour
1. Do **not** set `CLAUDE_CODE_USE_GATEWAY`. Model discovery uses `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` with
   provider kind firstParty + non-Anthropic `ANTHROPIC_BASE_URL` → `GET {base}/v1/models?limit=1000` (3 s timeout),
   keeps only ids matching `/(claude|anthropic)/i`, reads `id`, `display_name`, `description`. Not gated by
   `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`. Cached per base URL in `$CLAUDE_CONFIG_DIR/cache/gateway-models.json`.
2. Stream watchdogs: first byte within 180 s, idle ≥ 300 s → gateway commits SSE within 20 s and sends `ping` every 15 s
   while upstream is silent.
3. Model-family env: OPUS / SONNET / HAIKU / **FABLE** (`ANTHROPIC_DEFAULT_FABLE_MODEL`). `ANTHROPIC_SMALL_FAST_MODEL`
   wins over HAIKU → unset it in the child env.
4. Context window: unknown models get a default window of **200000**; `[1m]` suffix → 1M (Claude Code strips `[1m]`
   itself before sending and adds beta `context-1m-2025-08-07`). **`CLAUDE_CODE_MAX_CONTEXT_TOKENS` is ignored for
   ids starting with `claude-`** (so for every `claude-via-…` picker id; spike-verified, see §0.A). To shrink the
   compaction window for models below 200k use `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (works for any id; clamped to
   [100000, 1000000]; effective window = min(model window, value)). Smaller real windows are handled by the
   gateway's pre-flight prompt-too-long (§2.4). Router still strips `[1m]` (harmless); picker ids for ≥1M-context
   models carry `[1m]`.
5. A system block starting with `x-anthropic-billing-header:` is sent — drop it for non-Anthropic dialects.
6. `X-Claude-Code-Session-Id` header is sent → use as session id (`prompt_cache_key`, `session-id`, `x-grok-*-id`).
7. Paths carry `?beta=true` → route on path only.
8. Unknown models: `thinking:{type:"adaptive"}`, no temperature/top_k, `max_tokens` default 32000, plus
   `context_management`, `output_config:{effort}`, `metadata`. Signatures are not validated locally (empty/synthetic OK).
   Signed thinking from a different `message.model` is stripped → echo the requested model id in `message_start`.
9. `/model X` in-session sends a `max_tokens:1` "Hi" probe; a 404 shows "There's an issue with the selected model".
10. Non-SSE reply to `stream:true` → full non-streaming re-send (never let that happen; also support `stream:false`).
    A **404 on a `stream:true` request is also re-sent once with `stream:false`** (same body, `X-Stainless-Timeout:
    300`) even with `CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1` → every 404 costs two requests; the `stream:false`
    path must return the same 404. Decide streaming from `body.stream` only (Claude Code sends `Accept:
    application/json` on streaming requests).
11. Retries: honours `x-should-retry`; retries 408/409/429/5xx, connection errors and `overloaded_error` stream errors
    (10× default); honours `retry-after`/`retry-after-ms`. Context overflow detected by
    `prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)`.
12. `POST /v1/messages/count_tokens?beta=true` is called; failure falls back to a local estimate.
13. Images `{type:image,source:{type:base64,media_type,data}}` and PDFs `{type:document,…}` also appear **inside
    `tool_result.content`** (Read tool).
14. Root + `--dangerously-skip-permissions` exits unless `IS_SANDBOX=1` → headless tests use
    `--permission-mode dontAsk --allowedTools Bash,Read,Write,Edit`. `CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1`
    makes streaming bugs fail loudly in tests.
15. **`role:"system"` messages inside `messages`** (beta `mid-conversation-system-2026-04-07`): content is a string or
    text blocks (with `cache_control`), may also carry `tool_addition` (`{tool:{type:"tool_definition",definition}}` or
    `{tool:{type:"tool_reference",name}}`) / `tool_removal` blocks and a `clear_at` key, and **can be the LAST message**
    (e.g. `<total_tokens>…</total_tokens>` after a tool_result). Claude Code stops sending them only if the server
    400s with "Unexpected role"/"input message role"/"role system … not supported". The gateway folds them
    (`model.fold_system_messages_raw`, §3.0) for every dialect incl. passthrough.
16. **Structured output**: the background title request carries `output_config.format = {type:"json_schema",
    schema:{…title…}}`, `tools: []`, no `thinking` (beta `structured-outputs-2025-12-15`) → `NormalizedRequest.output_format`
    (§3.0).

### 0.A Spike results (verified 2026-10-03, real `claude` 2.1.288, mock upstream; fixtures in
`testing/fixtures/claude_code_2.1.288/`)
Confirmed:
- `GET /v1/models?limit=1000` is called at startup (headers `Authorization: Bearer <token>`, `User-Agent:
  claude-code/2.1.288`, `anthropic-version`; no session header). `claude-via-…` ids are accepted and sent verbatim as
  `model`. `$CLAUDE_CONFIG_DIR/cache/gateway-models.json` = `{baseUrl, fetchedAt, models:[{id,display_name,description}]}`
  and stores **every** returned id (the `/(claude|anthropic)/i` filter applies at use).
- `/v1/messages?beta=true` and `/v1/messages/count_tokens?beta=true`; `X-Claude-Code-Session-Id` on both (equals
  `metadata.user_id` JSON's `session_id`). User-Agent `claude-cli/2.1.288 (external, sdk-cli)` (-p) / `(external, cli)`
  (interactive); `x-app: cli`; `anthropic-beta` = `claude-code-20250219,interleaved-thinking-2025-05-14,
  thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,
  mid-conversation-system-2026-04-07,mid-conversation-tool-changes-2026-07-01,effort-2025-11-24,
  thinking-display-updates-2026-08-18` (+`context-1m-2025-08-07` with `[1m]`, +`redact-thinking-2026-02-12` /
  `structured-outputs-2025-12-15` interactive). No gzip request bodies toward a custom base URL.
- Main body keys: `model, messages, system, tools, metadata, max_tokens(32000), thinking({type:"adaptive",
  display:"updates"}), context_management({edits:[{type:"clear_thinking_20251015",keep:"all"}]}),
  output_config({effort:"high"}), stream(true)`; no temperature/top_k. `metadata.user_id` is a JSON **string**
  `{"device_id","account_uuid","session_id"}`. `system[0]` = `x-anthropic-billing-header: cc_version=…;
  cc_entrypoint=…;` (no cache_control), then 2 text blocks with `cache_control:{type:"ephemeral"}`.
- Tools: all custom (`name, description, input_schema` with `$schema` draft 2020-12); schema keywords seen: type,
  description, properties, additionalProperties, required, enum, maxLength, maximum, items, pattern, default, minimum,
  exclusiveMinimum, maxItems, allOf (nested), format, minLength — no root combinators. MCP tools inline as
  `mcp__<server>__<tool>` (85-char name observed, passed verbatim); **no `defer_loading`/ToolSearch** even with 141
  tools.
- Main model = `ANTHROPIC_MODEL`; background (session title, interactive only — none in `-p`) =
  `ANTHROPIC_DEFAULT_HAIKU_MODEL` (`claude-via-background`).
- `count_tokens` only on demand (`/context`: ~15 parallel calls); body `{model, messages, tools}` (+`thinking` when
  history has thinking), messages may be `[{"role":"user","content":"foo"}]`; beta `token-counting-2024-11-01`.
- Tool executes under `--permission-mode dontAsk --allowedTools Bash,Read,Write,Edit` as root (`out.txt` = `hello`).
- `ping` events accepted. With `CLAUDE_ENABLE_BYTE_WATCHDOG=1 CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS=8000` a 25 s silent
  stream fails ("Stream idle timeout - no chunks received") while pings every 2 s keep it alive (event-level idle
  timeout floor is 300 s). Heartbeat design (§2.2) verified.
- Thinking block `{signature:""}` + `signature_delta` (synthetic `fgw1.chat.…`) is sent back **verbatim** next turn.
- Read tool on a PNG → `tool_result.content = [{type:"image",source:{type:"base64",media_type:"image/png",data}}]`.
- 404 → after the stream:false retry, "There's an issue with the selected model (<id>). It may not exist or you may
  not have access to it. Run --model to pick a different model." rc 1. 500 + `x-should-retry:false` → no retry,
  "API Error: 500 <message>. This is a server-side issue, usually temporary …".
- stream:true answered with SSE → no non-streaming fallback (all turns streamed).
- `-p` prints `[claude-code:unrecognized_model] {"model":…}` on stderr for gateway ids (harmless).
Refuted / changed: `CLAUDE_CODE_MAX_CONTEXT_TOKENS` (item 4); 404 non-stream re-send (item 10); new items 15–16.
Not verified (needs interactive TTY): `/model` probe (`max_tokens:1`); research facts kept.

---

## 1. Package layout
```
shared/gateway/
  __init__.py            __version__="1.0.0"; exports Gateway, RouteTable, SecretStore, presets, launchkit
  __main__.py            python -m <pkg> serve --preset xai[,grok] [--port N] [--routes file.json]
  VERSION
  MANIFEST.sha256        generated by tools/vendor_gateway.py; sha256 of every file except itself
  DESIGN.md              this file
  compat.py              py3.8 helpers: rfc3339_parse/format (Z/offsets), b64url enc/dec, jwt_claims(token)->dict
                         (no verification), removeprefix, json_dumps_compact, now()
  config.py              ModelSpec, ProviderSpec, RouteTable, SecretStore (+from_dict/to_dict/validate)
  presets.py             PRESETS: provider id -> base_url/dialect/target/profile/auth kind/catalog family;
                         provider_from_preset(name, overrides)->ProviderSpec
  catalog.py             built-in model catalogs per family (context, max_output, reasoning, tools, vision,
                         per-model dialect/path overrides, effort_param, responses_lite)
  model.py               NormalizedRequest, Message, Block (parsed Anthropic request), fold_system_messages_raw
  anthropic_in.py        parse_messages_request(dict, headers)->NormalizedRequest
  events.py              internal event classes (§2.1)
  errors.py              GatewayError, map_upstream_error, map_error_code, prompt_too_long (Phase 0; §2.4)
  signatures.py          fgw1 thinking signatures: encode/decode/synthetic/keep_thinking_for (Phase 0; §3.0)
  anthropic_out.py       SSEEmitter, Aggregator, error_response, estimate_tokens (+ re-exports errors.*)
  router.py              Router, Resolution, RouteNotFound
  server.py              Gateway (ThreadingHTTPServer on a daemon thread), Handler
  transport.py           HttpClient (keep-alive pool, CONNECT proxy, NO_PROXY), Response, SSEEvent,
                         iter_sse, iter_ndjson, iter_with_heartbeat, make_ssl_context
  toolnames.py           ToolNameMap
  schema.py              scrub(schema, mode)
  secrets.py             resolve_many(specs) -> parallel env / `op read` / literal, memoized
  filelock.py            ExclusiveFileLock(path, timeout) (fcntl.flock / msvcrt.locking)
  atomicio.py            atomic_write_bytes/json(path, data, mode=0o600, indent) with Windows retry; restrict_permissions
  tracing.py             logger setup (rotating file), redaction, AI_GATEWAY_TRACE_FILE JSONL
  launchkit.py           find_claude(), build_child_env(), run_child() (signals), claude_version()
  auth/  __init__.py (make_auth), base.py, static.py, codex_chatgpt.py, grok_cli.py, gcloud_adc.py
  dialects/ __init__.py (REGISTRY), base.py, openai_chat.py, chat_profiles.py, responses.py,
            responses_targets.py, gemini.py, anthropic_passthrough.py, cli.py
  testing/ __init__.py, mock_upstreams.py, mock_auth.py, fake_bins.py, claude_e2e.py, fixtures/
```
Tests live in `ai-launchers/tests/gateway/` (stdlib `unittest`), vendored to `fry-launch-claude/tests/gateway/`.

---

## 2. Internal event model and the Anthropic SSE contract

### 2.1 Events (`events.py`, plain classes with `__slots__`, `__eq__`, `__repr__`)
| Event | Fields | Meaning |
|---|---|---|
| `TextDelta` | `key, text` | text for block `key` |
| `ThinkingDelta` | `key, text` | thinking text (may be `""` to open the block) |
| `ThinkingSignature` | `key, signature` | opaque round-trip blob for that thinking block |
| `ToolCall` | `id, name, input_json` | a **complete** tool call; `input_json` is a JSON object string |
| `Usage` | `input_tokens, output_tokens, cache_read=0, cache_write=0` | last wins |
| `Finish` | `stop_reason, stop_sequence=None` | `end_turn`/`max_tokens`/`tool_use`/`stop_sequence` |
| `StreamError` | `err_type, message, retryable` | error after commit |

Tool args are not live-streamed; dialects buffer and emit `ToolCall` when complete (openai_chat: tool k emitted when
tool k+1 starts **and** k's args parse as JSON, else at finish in index order; Responses: `output_item.done`; Gemini:
functionCall parts are complete). Any key change closes the open block and opens a new one.

### 2.2 `SSEEmitter(write, requested_model, message_id, input_estimate)`
- HTTP `200`, `Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-cache`, `Transfer-Encoding: chunked`
  (handler uses HTTP/1.1); each event one chunk, flushed; terminator `0\r\n\r\n` always written in `finally`.
- Event framing `event: <type>\ndata: <compact json>\n\n`.
- `start()`: `message_start` with `{"id":"msg_<24hex>","type":"message","role":"assistant","model":<requested model
  verbatim>,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":est,"output_tokens":0,
  "cache_creation_input_tokens":0,"cache_read_input_tokens":0}}` then one `ping`.
- Blocks: text `content_block_start {type:text,text:""}` → `text_delta`s; thinking `{type:thinking,thinking:"",
  signature:""}` → `thinking_delta`s → exactly one `signature_delta` immediately before `content_block_stop` (synthetic
  `fgw1.chat.<sha1-12>` if the dialect gave none); tool `{type:tool_use,id,name,input:{}}` → one
  `input_json_delta{partial_json:<full args>}` → stop. Exactly one block open at a time; every opened block closed;
  never open empty text blocks.
- `finish()` idempotent: close open block → `message_delta {"delta":{"stop_reason":R,"stop_sequence":S},"usage":
  {"input_tokens":I,"output_tokens":O,"cache_read_input_tokens":C,"cache_creation_input_tokens":0}}` (R non-null string;
  forced `tool_use` if any ToolCall was emitted; `end_turn` if upstream ended without Finish) → `message_stop`.
- `error(type,msg)`: `event: error` `{"type":"error","error":{"type":T,"message":M}}` then close. Transient mid-stream →
  `overloaded_error` (Claude Code retries); terminal → `api_error`.
- Heartbeat: `iter_with_heartbeat(iterable, interval=15)` runs the upstream reader on its own thread feeding a
  `queue.Queue`; handler `get(timeout=15)` → `ping` on timeout.
- Commit point: handler pulls the first event before writing; commits (200 + `message_start`) on first event **or** after
  20 s. `GatewayError` before commit → proper HTTP error response.

### 2.3 Non-stream (`stream:false`)
`Aggregator` consumes the same event iterator → full Message JSON (content blocks, stop_reason, usage). Upstream is
still streamed; one code path.

### 2.4 Errors — `errors.map_upstream_error(status, body_text, headers, provider, model, context_window=None, est_tokens=None, auth_hint=None) -> GatewayError`
Implemented in `errors.py` (Phase 0, fully tested in `tests/gateway/test_errors.py`); `anthropic_out` imports and
re-exports `GatewayError`, `map_upstream_error`, `prompt_too_long` and must not redefine them; dialects import
`errors` directly. `GatewayError(status, err_type, message, should_retry, retry_after=None)`; body
`{"type":"error","error":{"type":…,"message":…}}`; headers `x-should-retry: true|false`, `retry-after` /
`retry-after-ms` when known. Parses Anthropic, OpenAI `{"error":{…}}`, xAI `{"code","error":"…"}`, Google
`{"error":{"status","details":[RetryInfo|QuotaFailure|ErrorInfo]}}` (also wrapped in a list) and FastAPI
`{"detail":…}` bodies; HTML is stripped. Extra rows: 402 (DeepSeek "Insufficient Balance") → 429 terminal; 408/409/
other 5xx → 502 api_error retryable; other 4xx → 400. `status=None` = connection error → 502 api_error retryable
(`connection_error=True`). `map_error_code(code, message, …)` maps in-stream codes (Responses `response.failed`,
Gemini stream errors) through the same table. `GatewayError.to_stream_error()` → `StreamError` for post-commit
failures (retryable → `overloaded_error`).
| Upstream | Anthropic status / type | x-should-retry |
|---|---|---|
| context overflow (`context_length_exceeded`, `maximum context length`, `prompt is too long`, `exceeds the context window`, `input token count .* exceeds`, `too many tokens`) | 400 `invalid_request_error`, message exactly `prompt is too long: {N} tokens > {M} maximum` (N, M parsed, else estimate/catalog) | false |
| other 400/422 | 400 `invalid_request_error` (`[provider/model] ` + upstream message) | false |
| 401 (after one forced refresh) | 401 `authentication_error` + re-login hint | false |
| 403 | 403 `permission_error` | false |
| 404 / unknown model | 404 `not_found_error` | false |
| 413 | 413 `request_too_large` | false |
| 429 transient | 429 `rate_limit_error` + retry-after | true |
| 429 terminal (Codex `usage_limit_reached`+`resets_at`; Gemini `PerDay`/`QUOTA_EXHAUSTED`/retryDelay>300 s; OpenAI `insufficient_quota`; xAI credits exhausted) | 429 `rate_limit_error` incl. reset time | **false** |
| 500/502/connection error | 502 `api_error` | true |
| 503/529/"overloaded" | 529 `overloaded_error` | true |
Pre-flight: `estimate_tokens > 1.1 × context` (context known) → the prompt-too-long 400 without calling upstream.
`estimate_tokens(req)` = ceil(UTF-8 bytes of system+messages+tools text / 3.6); image 1600 tokens; document bytes/750.
The gateway never retries 429/5xx itself (Claude Code does); only one retry on a stale keep-alive connection and one
after a 401-triggered refresh.

---

## 3. Dialects

### 3.0 Common
- **Normalization** (`anthropic_in.py`): system string|blocks → list of texts (drop `cache_control`; drop the
  `x-anthropic-billing-header:` block except passthrough); message blocks: text, image (base64|url), document (base64
  PDF|text|url), thinking(+signature), redacted_thinking, tool_use, tool_result (content string | [text|image|document]),
  `search_result`→text; unknown blocks and server tools (tools entries with a `type` other than custom/absent, e.g.
  `web_search_*`) dropped + logged once. Keep `tool_choice`, `disable_parallel_tool_use`, `max_tokens`, `temperature`,
  `top_p`, `top_k`, `stop_sequences`. Effort: `output_config.effort`, else from `thinking` (adaptive→medium;
  enabled.budget_tokens ≤4k low, ≤16k medium, else high; disabled→none) — helper `model.split_effort_from_thinking`.
  Ignore `metadata` (except `user_id`'s `session_id`), `context_management`, betas, `thinking.display`.
  Background requests (role `background`) → effort low.
- **Mid-conversation system messages** (§0 item 15): `anthropic_in` first calls
  `model.fold_system_messages_raw(body["messages"])`: each system message's text becomes a text block
  `"<system-reminder>\n…\n</system-reminder>"` appended to the immediately preceding user message, else prepended to
  the next user message, else inserted as a standalone user message at that position; then
  `model.apply_tool_changes(body["tools"], fold)` applies `tool_addition` (definition → appended if new) and
  `tool_removal` (dropped from the offered tools; history still maps via `all_tool_names()`). Tools' `defer_loading`
  key is ignored. `anthropic_passthrough` applies the same two helpers to the raw body and drops the
  `mid-conversation-system-*` / `mid-conversation-tool-changes-*` betas it forwards.
- **Structured output** (§0 item 16): `output_config.format` (`{type:"json_schema", schema}`) →
  `NormalizedRequest.output_format`. openai_chat: `response_format:{type:"json_schema",json_schema:{name:"output",
  schema,strict:false}}` where the profile supports it; responses: `text:{format:{type:"json_schema",name:"output",
  schema,strict:false}}`; gemini: `generationConfig.responseMimeType:"application/json"` + `responseJsonSchema`;
  otherwise (and cli) append to the system prompt "Respond with ONLY a JSON object matching this JSON schema: <schema>".
  Passthrough forwards `output_config` untouched.
- `max_tokens` clamped to `min(request, model.max_output or profile default)`.
- **Tool ids**: upstream ids matching `^[A-Za-z0-9_-]{1,128}$` pass through; otherwise `toolu_x` + b64url(original),
  decoded back on input; missing ids → `toolu_<24hex>`.
- **Tool names** (`toolnames.py`): `ToolNameMap(regex, maxlen)` built per request from tools + history; valid names
  unchanged; else replace invalid chars with `_`, if changed or too long → `name[:maxlen-9] + "_" + sha1(name)[:8]`;
  Gemini forces a leading letter/`_`; collisions extend the hash; reverse-map `ToolCall.name`. Default regex
  `^[a-zA-Z0-9_-]{1,64}$`; Gemini `^[a-zA-Z_][a-zA-Z0-9_\-.:]{0,63}$`.
- **Schema scrub modes** (`schema.py`): `none`; `basic` (drop `$schema`, `$id`); `no_root_combinators` (basic + flatten
  root `oneOf/anyOf/allOf` by merging object branches' properties, required = intersection, root type object — xai,
  grok proxy); `gemini_json` (for `parametersJsonSchema`: basic + inline `$ref`/`$defs`/`definitions` (cycle depth 8 →
  `{}`) + collapse `type:[T,"null"]`→T); `gemini_openapi` (fallback `parameters`: allow-list `type, format,
  description, nullable, enum, properties, required, items, minItems, maxItems, minimum, maximum, minLength, maxLength,
  pattern, anyOf, title`; `const`→`enum`; `exclusiveMin/Max`→`min/max`; drop the rest).
- **Thinking signatures**: `fgw1.<target>.<b64url(json)>`, target tag ∈ {`codex_chatgpt`,`openai_api`,`grok_cli`,
  `xai_api`,`gemini_api`,`vertex`,`chat`}. On input, thinking blocks whose tag ≠ current target are **dropped**;
  non-`fgw1` signatures dropped for all non-passthrough dialects.
- **Images inside tool_result**: tool message/output carries text + `"[image attached below]"`; all such images of one
  user turn go into ONE following user message (chat/Responses) or extra `inlineData` parts in the same user content
  (Gemini). url images: chat/Responses pass URL; Gemini text placeholder. PDFs: Gemini `inlineData application/pdf`;
  openai/openrouter chat and openai_api Responses `file`/`input_file` data URL; others text placeholder
  `[PDF omitted: provider lacks document input]`. Text documents inlined.
- `is_error` tool results: chat/Responses prefix `ERROR: `; Gemini `response:{error:text}`.

### 3.1 `openai_chat` (+ `chat_profiles.py`)
Request `POST {base}/chat/completions`: system → `{role:system}`; assistant → `{role:assistant, content: text|null,
tool_calls:[{id,type:function,function:{name,arguments}}]}` (+`reasoning_content` = joined thinking text when
`profile.echo_reasoning_content`); tool_result → `{role:tool,tool_call_id,content}` then the deferred image user
message; `tools[{type:function,function:{name,description,parameters}}]`; `tool_choice` auto/"required"/
{type:function,function:{name}}/"none"; `parallel_tool_calls:false` only when requested and `profile.parallel_param`;
`stream:true`; `stream_options:{include_usage:true}` if `profile.stream_usage`.
Stream: `delta.content`→TextDelta; `delta.reasoning_content`/`delta.reasoning`→ThinkingDelta; `delta.tool_calls[i]`
accumulate `{id,name,args}` (id-only first chunk and whole-call-in-one-delta both handled); finish_reason stop→end_turn,
length→max_tokens, tool_calls/function_call→tool_use, content_filter→end_turn + text "[blocked by provider content
filter]"; usage chunk → Usage(prompt − cached, completion, cached) from `prompt_tokens_details.cached_tokens` (or
DeepSeek `prompt_cache_hit_tokens`); in-stream `{"error":…}` → StreamError; `[DONE]` ends.
Profiles:
| profile | max tokens field | drops/rules | reasoning | other |
|---|---|---|---|---|
| xai | max_tokens | reasoning models (catalog `reasoning` or id without `non-reasoning`): drop stop, presence_penalty, frequency_penalty | `reasoning_effort` only if model `effort_param` | schema no_root_combinators |
| openai | max_completion_tokens | reasoning ids `^(o\d|gpt-5|gpt-6)`: drop temperature/top_p | reasoning_effort (xhigh/max→high) | — |
| moonshot | max_tokens | drop temperature/top_p always | — | echo_reasoning_content |
| deepseek | max_tokens | drop temperature for reasoner/thinking | — | echo_reasoning_content |
| ollama | max_tokens | probe `/api/show` (cached): no `thinking` → drop reasoning_effort; no `tools` → omit tools, chat-only | — | base http://localhost:11434/v1 |
| openrouter | max_tokens | — | `reasoning:{effort}` | headers HTTP-Referer, X-Title; `usage:{include:true}` |
| opencode | max_tokens | — | — | — |
| nvidia | max_tokens | — | — | stream_usage false; tool regex `^[a-zA-Z0-9_-]{1,64}$` |
| generic | max_tokens | temperature only if sent | — | conservative |

### 3.2 `responses` (one translator; per-target rules in `responses_targets.py`)
Body: `model`, `instructions` (joined system, never empty — fallback "You are a helpful coding assistant."), `input`:
user text/images → `{type:message,role:user,content:[{type:input_text,text}|{type:input_image,image_url:"data:…"}]}`;
assistant text → `{type:message,role:assistant,content:[{type:output_text,text}]}`; thinking with matching-target
signature → `{type:reasoning,summary:[{type:summary_text,text}…],encrypted_content}` (no item ids); tool_use →
`{type:function_call,call_id,name,arguments}`; tool_result → `{type:function_call_output,call_id,output}` then deferred
image message. `tools:[{type:function,name,description,parameters,strict:false}]`; `tool_choice` auto/required/
{type:function,name}/none; `parallel_tool_calls`; reasoning models: `reasoning:{effort,summary:"auto"}` +
`include:["reasoning.encrypted_content"]`; `store:false`; `stream:true`; `prompt_cache_key`=session id.
| target | URL | auth/headers | body rules |
|---|---|---|---|
| openai_api | https://api.openai.com/v1/responses | Bearer key | `max_output_tokens` (clamped); temperature/top_p only for non-reasoning |
| chatgpt_codex | https://chatgpt.com/backend-api/codex/responses | codex_chatgpt auth + `originator: codex_cli_rs`, `User-Agent: codex_cli_rs/<ver> (<os> <osver>; <arch>) <term>`, `version: <ver>`, `session-id`, `Accept: text/event-stream` | delete max_output_tokens, temperature, top_p, truncation, user, metadata, previous_response_id, max_completion_tokens; non-empty instructions; tool names/call_ids ≤64. Catalog `responses_lite` (gpt-6.x): instructions → leading `{type:message,role:developer}` + header `x-openai-internal-codex-responses-lite: true`. Errors `{"detail":…}` |
| grok_cli_proxy | https://cli-chat-proxy.grok.com/v1/responses | grok_cli Bearer + `X-XAI-Token-Auth: xai-grok-cli`, `x-authenticateresponse: authenticate-response`, `x-grok-client-version`, `x-grok-client-identifier`, `x-grok-client-mode: headless`, `x-grok-conv-id`/`x-grok-session-id`=session id, `x-grok-req-id`=uuid4 | delete max_output_tokens, temperature; schema no_root_combinators |
| xai_api | https://api.x.ai/v1/responses | Bearer key or login token | max_output_tokens; for Responses-only ids (grok-4.20-multi-agent-0309) |
Grok fallback: if grok_cli_proxy fails (connection error, 404, 410, 426, 400/403 matching `/version|upgrade|client/i`,
5xx) the provider switches **stickily for the process** to its `fallback` ProviderSpec (openai_chat/xai at api.x.ai,
same token); one log line; pre-commit so transparent.
Versions: Codex `<ver>` = `AI_GATEWAY_CODEX_VERSION`, else cached `codex --version`, else pinned `0.160.0`; Grok same
with `AI_GATEWAY_GROK_CLIENT_VERSION` / `grok --version` / pinned fallback.
Stream: `output_item.added` reasoning → open thinking key=output_index; `reasoning_summary_text.delta` → ThinkingDelta
(second summary part prefixed "\n\n"); `output_text.delta` → TextDelta (key=output_index); `output_item.done`: reasoning
→ ThinkingSignature(`fgw1.<target>.{enc,summary}`) (no summary → empty thinking block + signature); function_call →
ToolCall (arg deltas ignored); `response.completed` → Usage(input − cached, output, cached) + Finish;
`response.incomplete` → max_tokens if `incomplete_details.reason=="max_output_tokens"` else end_turn + note;
`response.failed`/`error` → GatewayError pre-commit (code-mapped: context_length_exceeded→prompt-too-long,
rate_limit_exceeded→429, server_error→529) else StreamError.

### 3.3 `gemini` (targets `gemini_api`, `vertex`)
URLs: gemini_api `{base=https://generativelanguage.googleapis.com}/v1beta/models/{m}:streamGenerateContent?alt=sse`,
header `x-goog-api-key`; vertex `https://{host}/v1/projects/{project}/locations/{loc}/publishers/google/models/{m}:
streamGenerateContent?alt=sse`, host `aiplatform.googleapis.com` for `global` else `{loc}-aiplatform.googleapis.com`,
Bearer + `x-goog-user-project`.
Body: `contents[{role:user|model,parts}]` (merge consecutive same-role; leading model turn gets a `user:"(continue)"`
prepended), `systemInstruction:{parts:[{text}]}`, `tools:[{functionDeclarations:[{name,description,
parametersJsonSchema}]}]`, `toolConfig.functionCallingConfig{mode:AUTO|ANY|NONE,allowedFunctionNames?}`,
`generationConfig{maxOutputTokens,temperature?,topP?,stopSequences?,thinkingConfig:{includeThoughts:true,
thinkingLevel (3.x: low|high) | thinkingBudget (2.5: -1, or 0 when disabled and allowed)}}`.
Parts: assistant text → `{text}`; tool_use → `{functionCall:{name,args,id?}}` (id only if originally from Gemini);
gemini-tagged thinking block payload `{"s":sig}` attached as `thoughtSignature` to the **next** text/functionCall part of
that assistant message (thought text never sent back); missing → side LRU (tool_use id → sig, 4096 entries); still
missing for `gemini-3*` → first functionCall of each model turn gets `"skip_thought_signature_validator"`. Tool results:
**all** `functionResponse{name (via tool_use id → name from history), id?, response:{output|error}}` parts in ONE user
content, then tool-result `inlineData` images, then the user's other text.
Stream: `part.thought && text` → ThinkingDelta; `part.text` → TextDelta; `part.functionCall` → ToolCall; a part carrying
`thoughtSignature` first emits `ThinkingDelta(newkey,"")` + `ThinkingSignature` so the thinking block precedes it.
finishReason STOP→end_turn, MAX_TOKENS→max_tokens, SAFETY/RECITATION/PROHIBITED_CONTENT→end_turn + note,
MALFORMED_FUNCTION_CALL→end_turn + text "[model produced a malformed tool call; please retry]"; tool calls present →
tool_use. usageMetadata: prompt − cached; candidates + thoughts; cached. 429 RESOURCE_EXHAUSTED: parse
RetryInfo.retryDelay / QuotaFailure, classify per §2.4.
Schema fallback: 400 mentioning `parametersJsonSchema`/`Unknown name` → sticky per-provider retry with `parameters` +
`gemini_openapi`.

### 3.4 `anthropic_passthrough`
URL `{base}{messages_path="/v1/messages"}`. Bases: deepseek `https://api.deepseek.com/anthropic`; kimi
`https://api.moonshot.ai/anthropic`; openrouter `https://openrouter.ai/api`; ollama `http://localhost:11434`;
opencode zen/go `https://opencode.ai/zen/v1` (`/zen/go/v1`) with `messages_path="/messages"`. Headers `Authorization:
Bearer` (or `x-api-key` if `profile.key_header`), `anthropic-version: 2023-06-01`; forward `anthropic-beta` unless
`profile.drop_betas`. Body: original JSON with `model` → upstream id (`[1m]` stripped), `max_tokens` clamped,
`fgw1.*`-signed thinking blocks removed, `role:"system"` messages folded (§3.0). **Response (changed in Phase 0 to
keep one code path): the upstream Anthropic SSE is parsed into internal events** like every other dialect —
`text_delta`→TextDelta, `thinking_delta`→ThinkingDelta, `signature_delta`→ThinkingSignature (upstream signature kept
verbatim), `tool_use` block + `input_json_delta`s → one ToolCall at `content_block_stop`, `message_start.usage` +
`message_delta.usage` → Usage, `message_delta.stop_reason` → Finish, `event: error` → StreamError (or GatewayError
before the first event); `redacted_thinking`/server-tool blocks dropped (logged once). The server's SSEEmitter then
echoes the requested model, so no byte forwarding/model rewriting is needed and `stream:false` works via the
Aggregator. Upstream HTTP errors go through `errors.map_upstream_error` (quota → `x-should-retry:false`). Ollama 404 on `/v1/messages` (<0.14) → sticky switch to openai_chat/ollama +
log "upgrade Ollama ≥0.14". count_tokens always local estimate.

### 3.5 `cli` (chat-only; opencode only)
Prompt = system + full transcript flattened ("System:/User:/Assistant:/Tool call/Tool result" sections) with
`<system-reminder>…</system-reminder>` stripped, images → `[image]`. `opencode run --model <provider/model>` with the
prompt on **stdin**, `encoding="utf-8"`, `communicate(timeout=300)`, `CREATE_NO_WINDOW` on Windows,
`Semaphore(2)` per CLI. Output: strip ANSI (all CSI/OSC) + header lines → one TextDelta + end_turn; usage estimated.
rc≠0 → GatewayError(502, api_error, stderr tail, should_retry=False) pre-commit. `max_tokens<=1` probes → local "ok".
`/v1/models` description says "chat-only (no tools)".

---

## 4. Auth (`auth/`)
```python
class AuthProvider:
    kind: str
    def available(self) -> bool
    def headers(self, force_refresh: bool = False) -> Dict[str, str]   # thread-safe, single-flight
    def on_unauthorized(self, used: Dict[str, str]) -> bool            # True => refreshed, retry once
    def describe(self) -> Dict[str, Any]                               # source, expires_at, account hint; never secrets
```
`Dialect.execute` calls `headers()`; on 401 (or chatgpt_codex 403 `token_expired`/`invalid_token`) calls
`on_unauthorized(used)`; True → re-send once (pre-commit); second 401 → `authentication_error` + re-login hint.
Per-provider `threading.Lock`; a thread whose token was already replaced (`used != current`) returns True w/o refreshing.
- `static.py` — `StaticKeyAuth(secret_name, store, style="bearer"|"x-api-key"|"x-goog-api-key"|"none")`.
- `codex_chatgpt.py` — `$CODEX_HOME/auth.json` | `~/.codex/auth.json`. Non-empty `OPENAI_API_KEY` in the file →
  not available for this kind (launcher offers an `openai` route with that key). No file → describe "codex login
  (file-based); keyring-stored creds unsupported — set cli_auth_credentials_store = \"file\"". Token
  `tokens.access_token`, expiry JWT `exp`; account id `tokens.account_id` else id_token claim
  `["https://api.openai.com/auth"]["chatgpt_account_id"]`. Refresh when `exp − now < 300` or on 401:
  `ExclusiveFileLock(<auth.json>.lock, 10s)` → re-read; if on-disk refresh_token differs from ours and its access token
  is fresh, adopt and stop → POST `https://auth.openai.com/oauth/token` JSON `{grant_type:"refresh_token",
  client_id:"app_EMoamEEZ73f0CkXaXp7hrann",refresh_token}` (URL override `AI_GATEWAY_OPENAI_AUTH_URL`) → re-read, merge
  `tokens.{access_token,id_token?,refresh_token}` + `last_refresh` (RFC3339 Z), keep unknown fields,
  `atomic_write_json(indent=2, mode 0600)`. `invalid_grant`/`refresh_token_reused` → re-read; disk newer → adopt; else
  terminal "run `codex login`".
- `grok_cli.py` — `$GROK_AUTH_PATH` > `$GROK_HOME/auth.json` > `~/.grok/auth.json`; lock `auth.json.lock`
  (POSIX `fcntl.flock(LOCK_EX)`; Windows `msvcrt.locking(fd, LK_NBLCK, 1)` at offset 0, 50 ms retry loop). Entry key
  `https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828`, else first entry with `auth_mode in (oidc, external)` and a
  `key`. `auth_mode == api_key` entries / `xai::api_key` → exposed as static key (xai route). `web_login` ignored. Expiry
  `expires_at` (RFC3339) else `create_time + 30d`; refresh 300 s early or on 401: GET
  `{oidc_issuer}/.well-known/openid-configuration` (cached; fallback `https://auth.x.ai/oauth2/token`; override
  `AI_GATEWAY_XAI_TOKEN_URL`) → form POST `grant_type=refresh_token&refresh_token&client_id` (+`principal_type`/
  `principal_id` if present) → set `key`, `expires_at = now + expires_in`, `create_time`; keep old RT if none returned;
  same lock/re-read/adopt/atomic pretty-JSON 0600 write. `invalid_grant` terminal "run `grok login`".
- `gcloud_adc.py` — file `$GOOGLE_APPLICATION_CREDENTIALS` | `$CLOUDSDK_CONFIG/application_default_credentials.json` |
  `%APPDATA%\gcloud\…` (Windows) | `~/.config/gcloud/…`. `type == authorized_user` → form POST
  `https://oauth2.googleapis.com/token` (override `AI_GATEWAY_GOOGLE_TOKEN_URL`) with client_id/client_secret/
  refresh_token; cache in memory until `expires_in − 300`; **no write-back**. Other types or no file but `gcloud` on PATH →
  `gcloud auth application-default print-access-token` (fixed args, 60 s timeout), cached 45 min. Project:
  `GOOGLE_CLOUD_PROJECT`/`CLOUDSDK_CORE_PROJECT`/`GCLOUD_PROJECT` → ADC `quota_project_id` →
  `gcloud config get-value project` → error "set GOOGLE_CLOUD_PROJECT". Location `GOOGLE_CLOUD_LOCATION` or `global`.
  Headers Bearer + `x-goog-user-project`.

---

## 5. Router

### 5.1 RouteTable (JSON-serializable, no secrets)
```json
{"version":1,"picker_prefix":"claude-via-","long_context_threshold":60000,
 "providers":{"xai":{"display_name":"xAI API","dialect":"openai_chat","profile":"xai",
   "base_url":"https://api.x.ai/v1","auth":{"kind":"api_key","secret":"xai","style":"bearer"},
   "headers":{},"allow_unlisted":true,"chat_only":false,"options":{},
   "models":[{"id":"grok-4.7","context":2000000,"max_output":64000,"reasoning":true,"tools":true}]},
  "grok":{"dialect":"responses","target":"grok_cli_proxy","base_url":"https://cli-chat-proxy.grok.com/v1",
   "auth":{"kind":"grok_cli"},"fallback":{"dialect":"openai_chat","profile":"xai","base_url":"https://api.x.ai/v1"}}},
 "roles":{"default":"xai,grok-4.7","background":"xai,grok-4.20-0309-non-reasoning","longContext":null,"subagent":null},
 "aliases":{"fry-grok-4-3":"grok,grok-4.3"},
 "provider_aliases":{"grok":["xai"],"xai":["grok"],"codex":["openai"],"openai":["codex"],"local-ollama":["ollama"]}}
```
Secrets: `auth.secret` names an entry in an in-memory `SecretStore` (redacted repr). `AI_GATEWAY_UPSTREAM_<ID>`
(upper, `-`→`_`) replaces `base_url`; `AI_GATEWAY_UPSTREAM_<ID>_FALLBACK` replaces `fallback.base_url`.

### 5.2 `Router.resolve(requested, est_tokens) -> Resolution(provider, model, model_spec, requested, role, background)`
1. Strip case-insensitive `claude-via-` prefix and trailing `[1m]`.
2. Role aliases `background|default|longcontext|subagent` (from `claude-via-background` …) → that role's target;
   `background=True` for background.
3. `provider,model`: legacy fry forms first — `ollama,fry-grok-<x>` → `grok,<x>` via alias table (`fry-grok-4-3`→
   `grok-4.3`, `fry-grok-4-20-0309-reasoning`→`grok-4.20-0309-reasoning`, `…-non-reasoning`); `ollama,fry-codex-<m>` →
   `codex,<m>`; `ollama,fry-opencode-<m>` → `opencode,<m>`; `local-ollama,<m>` → `ollama,<m>`. Canonicalize provider:
   if not configured try its `provider_aliases` in order. Model in catalog/discovered list or `allow_unlisted` → OK.
4. No comma: exact alias; bare catalog id (first provider in table order listing it); Claude family: `/haiku/i` →
   background; names starting `claude`/`anthropic` or exactly `opus|sonnet|fable|default` → default.
5. longContext: resolved route == roles.default, roles.longContext set, est_tokens > threshold → longContext.
   `think`/`webSearch` roles accepted and ignored (log once).
6. Else `RouteNotFound` → 404 `not_found_error`: "model 'X' is not routable via this launcher; available: <first 5
   picker ids>… (run `<launcher> models`)".

### 5.3 `/v1/models` and probes
`GET /v1/models` (any query) → `{"data":[…],"has_more":false,"first_id":…,"last_id":…}`; entries
`{"type":"model","id":"claude-via-<p>,<m>[1m if context ≥ 1e6]","display_name":"<model> · <provider display>",
"created_at":"2026-01-01T00:00:00Z","description":"<auth source> · tools|chat-only · <ctx> ctx"}`; default route first,
then providers in table order; catalog ∪ discovery cache, ≤50 per provider. `GET /v1/models/{id}` → entry or 404.
Probes: `max_tokens <= 1` on a `cli` route → local "ok" (SSE or JSON per `stream`). API routes forward the probe.

---

## 6. Launch helpers (`launchkit.py`)
- `find_claude()`: `AI_LAUNCHERS_CLAUDE_BIN` / `FRY_CLAUDE_BIN` env → `claude.exe`/`claude` native on PATH → npm
  `claude.cmd` resolved to `node <prefix>/node_modules/@anthropic-ai/claude-code/cli.js` (if it exists) → last resort
  `cmd /c` with `subprocess.list2cmdline` + warning if args contain `%^&|<>`.
- `build_child_env(base_env, base_url, token, default_id, background_id, context, extra)`: sets `ANTHROPIC_BASE_URL`,
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY=""`, `ANTHROPIC_MODEL` + `ANTHROPIC_DEFAULT_{OPUS,SONNET,FABLE}_MODEL` =
  default id, `ANTHROPIC_DEFAULT_HAIKU_MODEL` = background id, `ANTHROPIC_CUSTOM_MODEL_OPTION` (+`_NAME`,
  `_DESCRIPTION`), `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`,
  `CLAUDE_CODE_AUTO_COMPACT_WINDOW=max(100000, context)` **only when the default model's context < 200000** (spike:
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` is ignored for `claude-*` ids, so it is not set), `NO_PROXY`/`no_proxy` +=
  `127.0.0.1,localhost,::1`; unsets `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, `CLAUDE_CODE_GZIP_REQUEST_BODIES`,
  `ANTHROPIC_CUSTOM_HEADERS`, `ANTHROPIC_BETAS`, `ANTHROPIC_SMALL_FAST_MODEL*`,
  `ANTHROPIC_DEFAULT_*_{NAME,DESCRIPTION,SUPPORTED_CAPABILITIES}`, `ANTHROPIC_UNIX_SOCKET`, `CLAUDE_CODE_USE_*`,
  `_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL`, `CLAUDE_CODE_OAUTH_TOKEN*`. Provider secrets never placed in it.
  Also `build_direct_env(...)` for direct Anthropic-endpoint launches (same hygiene, vendor vars).
- `run_child(argv, env)`: installs a **no-op Python SIGINT handler** (not SIG_IGN — inherited across exec) restored
  after; POSIX forwards SIGTERM/SIGHUP to the child; returns rc (130 on interrupt outside child).

---

## 7. Testing
- Unit (`tests/gateway/`): SSE grammar property test over random event interleavings; Aggregator == SSE replay;
  error table incl. exact prompt-too-long format and x-should-retry/retry-after; request parsing from golden fixtures in
  `testing/fixtures/claude_code_2.1.288/` (recorded from the real binary, scrubbed); router rules; toolnames/schema;
  per-dialect request goldens + stream parsing (parallel/interleaved tool calls, missing ids, usage, every finish
  mapping); auth refresh/rotation/write-back (preserve unknown fields, 0600), adopt-newer, **4-process race test** against
  a token mock that revokes reused refresh tokens, lock timeout, ADC paths; transport keep-alive reuse, stale retry,
  CONNECT proxy mock, NO_PROXY, heartbeat; py3.8 AST (`ast.parse(src, feature_version=(3,8))`) + API denylist; no BOM;
  vendor manifest; no secret sentinel in logs/trace.
- Mocks (`testing/mock_upstreams.py`): `MockServer(kind, brain)` on 127.0.0.1:0 recording requests (§8.1); each enforces its
  documented quirks with realistic 400/401 bodies (xai, openai chat, moonshot, nvidia, ollama, openai Responses, chatgpt
  backend, grok proxy incl. version-gate mode, gemini api, vertex, anthropic-compatible) + `mock_auth` (OpenAI token
  endpoint with rotating RT and reuse revocation, xAI OIDC discovery + token, Google token) + `fake_bins` (gcloud, codex
  --version, grok --version, opencode stubs; `.cmd` twins on Windows). Brain: turn 1 calls tool `Bash` with
  `{"command":"printf hello > out.txt && env"}` (name looked up among offered tools, reversing shortening); turn 2
  asserts the tool result contains no `FAKEKEY-SENTINEL` and replies `DONE <sha8 of tool output>`; background/haiku
  requests get a short reply; every dialect streams in its native format incl. reasoning + signatures.
- E2E (`testing/claude_e2e.py`, `RUN_E2E=1`, skipped when claude missing): temp HOME/USERPROFILE,
  `CLAUDE_CONFIG_DIR=$HOME/.claude`, `AIL_HOME`/`FRY_HOME`/`CODEX_HOME`/`GROK_HOME`/ADC under it,
  `AI_GATEWAY_UPSTREAM_<ID>` → mocks, auth URL overrides → mock_auth, `AI_GATEWAY_TRACE_FILE`,
  `DISABLE_AUTOUPDATER=1`, `CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1`, fake keys containing the sentinel; pre-seeded
  `~/.claude/settings.json` `{"model":"sonnet"}` and `~/.claude.json` with a user `additionalModelOptionsCache` entry.
  Command `<launcher> launch claude [--model X] -- -p "Run the shell command and report" --output-format stream-json
  --verbose --permission-mode dontAsk --allowedTools Bash,Read,Write,Edit --strict-mcp-config` in `$tmp/work`.
  Assert rc 0, result success containing `DONE`, `out.txt == "hello"`, quirk-compliant requests, `GET /v1/models?limit=1000`
  seen, all `/v1/messages` streamed (except probes), settings.json byte-identical, `.claude.json` user entry intact and no
  fry entries, sentinel absent from tool output.

---

## 8. Interfaces (frozen)
Phase-0 code is real, working and tested (`tests/gateway/test_foundation.py`, `test_errors.py`, `test_vendor.py`,
`test_py38_compat.py`). Parallel agents code against the signatures below **without editing Phase-0 files**; if a
Phase-0 interface must change, ask the coordinator. Every module is Python 3.8-compatible, stdlib-only, relative
imports only. Tests import the package only through `tests/gateway/_pkg.py` (`from ._pkg import mod`;
`ev = mod("events")`) — `tools/vendor_gateway.py` rewrites its single `PKG = "shared.gateway"` line for fry.

### 8.1 Phase 0 (implemented)
**`compat.py`** — `UTC`; `rfc3339_parse(s) -> datetime` (aware UTC; `Z`/`z`, `±HH:MM`/`±HHMM`/`±HH`, any fraction
digits, `T`/space, no offset = UTC, leap second; `ValueError` otherwise); `rfc3339_to_epoch(s) -> float`;
`rfc3339_format(value=None, timespec="auto"|"seconds"|"milli"|"micro") -> "…Z"` (value: datetime (naive=UTC) | epoch |
None=now); `b64url_encode(bytes|str) -> str` (unpadded); `b64url_decode(str|bytes) -> bytes` (padding/whitespace/std
alphabet tolerant, `ValueError`); `jwt_claims(token) -> dict` (unverified; `{}` on failure); `removeprefix(s, p)`,
`removesuffix(s, p)`; `json_dumps_compact(obj) -> str`; `utcnow_epoch() -> float`; `utcnow() -> datetime`.

**`events.py`** — `Event` base (`__slots__`, value `__eq__`/`__hash__`, `__repr__`, `kind`, `to_dict()`);
`TextDelta(key, text)`, `ThinkingDelta(key, text="")`, `ThinkingSignature(key, signature)`,
`ToolCall(id, name, input_json)`, `Usage(input_tokens, output_tokens, cache_read=0, cache_write=0)` (ints),
`Finish(stop_reason, stop_sequence=None)`, `StreamError(err_type, message, retryable=False)`; `STOP_REASONS`.

**`model.py`** — dataclasses (invariants in the module docstring):
`Block(type, text=None, data=None, media_type=None, url=None, id=None, name=None, input=None, tool_use_id=None,
content=None, is_error=False, signature=None, thinking=None, title=None)` with constructors `of_text`,
`of_image_base64(media_type, data)`, `of_image_url(url)`, `of_document_base64(media_type, data, title=None)`,
`of_document_text(text, title=None)`, `of_document_url(url, title=None)`, `of_thinking(thinking, signature=None)`,
`of_redacted_thinking(data)`, `of_tool_use(id, name, input)`, `of_tool_result(tool_use_id, content, is_error=False)`
(str content → one text block) and helpers `is_media()`, `result_text(joiner="\n")`, `result_media()`;
`Message(role: "user"|"assistant", blocks)` (+`text()`, `tool_uses()`, `tool_results()`);
`ToolDef(name, description="", input_schema={"type":"object","properties":{}})`;
`NormalizedRequest(model, system: List[str], messages, tools, tool_choice: Optional[{"type":"auto"|"any"|"tool"|"none",
"name"?}], disable_parallel_tool_use=False, max_tokens=32000, temperature=None, top_p=None, top_k=None,
stop_sequences=[], stream=True, effort: Optional[EFFORTS], thinking_requested=False, thinking_budget=None,
output_format=None, session_id=None, raw={}, headers={} (lower-case keys, no auth), dropped=[])` with
`tool_names()`, `all_tool_names()`, `tool_use_names_by_id()`, `is_probe()`. `BLOCK_TYPES`,
`EFFORTS = ("none","minimal","low","medium","high","xhigh","max")`.
`fold_system_messages_raw(messages, wrap=True) -> FoldResult(messages, tool_additions, tool_reference_additions,
tool_removals, folded)`; `apply_tool_changes(raw_tools, fold) -> raw_tools`;
`split_effort_from_thinking(thinking) -> (effort, requested, budget)`.

**`errors.py`** — `GatewayError(status, err_type, message, should_retry=False, retry_after=None,
upstream_status=None, upstream_body=None, upstream_headers=None, connection_error=False)` with `.body()`,
`.headers()`, `.to_stream_error()`; `map_upstream_error(status, body_text, headers, provider, model,
context_window=None, est_tokens=None, auth_hint=None) -> GatewayError` (§2.4); `map_error_code(code, message,
provider, model, context_window=None, est_tokens=None, extra=None) -> GatewayError`; `prompt_too_long(n, m)`;
`parse_retry_after(headers) -> Optional[float]`; `ERROR_TYPES`; `TERMINAL_RETRY_DELAY = 300`.

**`signatures.py`** — `encode_signature(target, payload: dict) -> "fgw1.<target>.<b64url json>"`,
`decode_signature(sig) -> Optional[(target, dict)]` (synthetic chat → `("chat", {})`), `signature_target(sig)`,
`synthetic_signature(text) -> "fgw1.chat.<sha1-12>"`, `keep_thinking_for(sig, target) -> bool`; `TARGETS`.
Payload conventions: responses `{"enc": encrypted_content, "summary": [texts]}`; gemini `{"s": thoughtSignature}`.

**`config.py`** — `ModelSpec(id, display_name=None, context=None, max_output=None, reasoning=False, tools=True,
vision=False, effort_param=False, dialect_override=None, target_override=None, path_override=None,
responses_lite=False, extra={})` (`from_dict` accepts a bare id string; unknown keys → `extra`; `to_dict` omits
defaults; `label()`); `ProviderSpec(id, display_name="", dialect="openai_chat", target=None, profile=None,
base_url="", auth={"kind":"none"}, headers={}, allow_unlisted=False, chat_only=False, options={}, models=[],
fallback=None, schema_mode=None, tool_name_regex=None, max_tool_name=None)` with `from_dict(id, d, parent=None)`,
`to_dict(parent=None)` (fallback inherits/omits `display_name, auth, models, allow_unlisted, chat_only`),
`find_model(id)` (exact then case-insensitive), `lists_model`, `model_spec(id)` (catalog or default),
`effective_dialect(ms)`, `effective_target(ms)`, `label()`, `problems()`; `RouteTable(version=1,
picker_prefix="claude-via-", long_context_threshold=60000, providers: Dict[id, ProviderSpec] (ordered), roles,
aliases, provider_aliases)` with `from_dict`, `to_dict`, `copy`, `provider(id)`, `canonical_provider(id)`,
`picker_id(provider_id, model_or_spec)` (+`[1m]` when context ≥ 1e6), `problems() -> List[str]`, `validate()`
(raises `ConfigError(problems)`). `split_route("p,m") -> (p, m)` (first comma; no comma → `(None, s)`);
`upstream_env_name(id, fallback=False)`; `apply_upstream_env_overrides(table, environ=None) -> RouteTable` (copy);
`describe_upstream_env_overrides(table, environ=None) -> List[str]`. `SecretStore(initial=None)`: `get(name,
default=None)`, `set(name, value)` (None/"" deletes), `delete`, `has`, `names()`, `values_for_redaction()`,
`redact(text)`, `in`, `len`; repr shows names only; pickling/JSON raise `TypeError`; copy/deepcopy are in-memory.
Constants `DIALECTS`, `DIALECT_TARGETS`, `AUTH_KINDS`, `AUTH_STYLES`, `SCHEMA_MODES`, `ROLE_NAMES`.

**`auth/base.py`** — `AuthError(message, hint=None, terminal=True)`; `AuthProvider(provider_id="")`: `kind`,
`available()`, `headers(force_refresh=False) -> Dict[str,str]`, `on_unauthorized(used) -> bool`, `describe() -> dict`
(never secrets), `relogin_hint() -> Optional[str]`, `_lock` (RLock); `Token(access_token, expires_at=None,
extra=None)` (`expires_within(s)`); `RefreshingAuth(provider_id)` — implement `_load() -> Optional[Token]`,
`_refresh(token) -> Token` (raise AuthError), optionally `_token_headers(token)`; supplies single-flight
`headers()`/`on_unauthorized()` (a thread whose `used` headers differ from the current token returns True without
refreshing), `refresh_margin = 300`, `current_token()`; `redact_headers(headers)`.
**`auth/static.py`** — `StaticKeyAuth(secret_name, store, style="bearer"|"x-api-key"|"x-goog-api-key"|"none",
provider_id="")` (key read from the store per call; `on_unauthorized` True only if the stored key changed);
`NoAuth(provider_id="")`. **`auth/__init__.py`** — `make_auth(auth_dict, secrets, provider_id)`; lazy kinds
`AUTH_CLASSES = {"codex_chatgpt": ("codex_chatgpt","CodexChatGPTAuth"), "grok_cli": ("grok_cli","GrokCliAuth"),
"gcloud_adc": ("gcloud_adc","GcloudADCAuth")}` constructed as `Cls(provider_id=..., options=auth_dict)`.

**`dialects/base.py`** — `ProviderRuntime(provider_id)`: `lock`, `state`, `get`, `set`, `setdefault(key, factory)`,
`log_once(log, key, msg, *args)`; well-known keys `grok_fallback`, `gemini_schema_fallback`, `ollama_caps`,
`ollama_chat_fallback`, `gemini_sig_lru`, `versions`. `LRU(capacity=4096)`: `get`, `put`, `in`, `len`.
`RequestContext(req, resolution, provider, model, runtime, auth, http, session_id, log=None, requested_model=None,
est_tokens=None, tracer=None, secrets=None)` (+`background`, `trace(name, **fields)`, `with_provider(spec)`).
`Dialect`: `name`, `execute(ctx) -> Iterator[Event]` — raise `GatewayError` before the first yield; afterwards yield
`err.to_stream_error()` and stop; close upstream in `finally`; never retry 429/5xx. `join_url(base, path)`.
`send_with_auth_retry(ctx, method, url, headers, body, stream=True, unauthorized=None, timeout=None) -> Response`
(2xx only; 401 or `unauthorized(status, text)` → `auth.on_unauthorized(used)` → one re-send; errors via
`errors.map_upstream_error(…, context_window=ctx.model.context, est_tokens=ctx.est_tokens,
auth_hint=auth.relogin_hint())`; transport failure → 502 `connection_error=True`; `AuthError` → 401). **Sticky
fallbacks are owned by the dialect**: at the start of `execute`, if its runtime flag is set, delegate to
`get_dialect(fb.dialect).execute(ctx.with_provider(provider.fallback))`; on a qualifying pre-commit failure set the
flag, log once, and delegate the same way. The server always dispatches on the primary provider.
**`dialects/__init__.py`** — `get_dialect(name)` (cached instance; `ValueError` if unknown); `REGISTRY = {
"openai_chat": ("openai_chat","OpenAIChatDialect"), "responses": ("responses","ResponsesDialect"), "gemini":
("gemini","GeminiDialect"), "anthropic_passthrough": ("anthropic_passthrough","AnthropicPassthroughDialect"), "cli":
("cli","CliDialect")}`. Dialect instances are stateless and constructed with no arguments.

**`transport.py`** (interface frozen; agent B replaces internals) — `TransportError(message, url=None, cause=None)`;
`CaseInsensitiveDict`; `HttpClient(timeout=600.0, connect_timeout=30.0, user_agent=None, ssl_context=None,
environ=None)`: `request(method, url, headers=None, body: bytes|str|None=None, stream=True, timeout=None) ->
Response` (never raises on HTTP status; defaults `User-Agent`, `Accept-Encoding: identity`, `Content-Length`),
`proxy_for(url)`, `close()`; `Response`: `status`, `reason`, `headers`, `url`, `read()`, `text()`, `json()`,
`iter_lines()`, `close()` (thread-safe, unblocks readers), `closed`, context manager; `SSEEvent(event="message",
data="", id=None, retry=None)` (+`json()`); `iter_sse(response_or_lines)`; `iter_ndjson(response_or_lines)`;
`HEARTBEAT`; `iter_with_heartbeat(iterable, interval, on_close=None, max_queue=1024)`; `make_ssl_context(cafile=None,
environ=None)`; `proxy_for_url(url, environ=None)`; `bypass_proxy(host, port, no_proxy)`; `DEFAULT_USER_AGENT`.

**`presets.py` / `catalog.py`** (structure frozen; agent F owns content) — `PRESETS[name]` = ProviderSpec fields +
launcher keys `catalog`, `secret_env`, `default_model`, `background_model`, `notes` (`LAUNCHER_KEYS`);
`provider_from_preset(name, overrides=None) -> ProviderSpec`; `route_table_from_presets(names, overrides=None,
roles=None) -> RouteTable`; `preset_names()`, `secret_env_vars(name)`, `default_roles(name, provider_id=None)`;
`DEFAULT_PROVIDER_ALIASES`, `GEMINI_TOOL_NAME_REGEX`. `CATALOGS[family]`, `get_catalog(family) -> List[ModelSpec]`
(copies), `families()`, `find_model(family, id)`, `is_responses_lite(id)`, `is_openai_reasoning(id)`,
`is_xai_reasoning(id, spec=None)`. Presets present: xai, grok, openai, codex, gemini, gemini-vertex, deepseek, kimi,
openrouter, ollama, opencode-zen, opencode-go, opencode, nvidia. Vertex: empty `base_url` ⇒ host from location.

**`testing/__init__.py`** — `FIXTURES_DIR`, `CLAUDE_CODE_FIXTURES = "claude_code_2.1.288"`, `SENTINEL =
"FAKEKEY-SENTINEL"`, `fixture_path(name)`, `load_fixture(name) -> {"method","path","headers","body"}`,
`list_fixtures()`. **`testing/mock_upstreams.py`** — `MockServer(kind=None, brain=None, options=None,
handler=None)` (`start()`, `stop()`, context manager, `url`, `port`, `requests` (records `{method, path, query,
raw_path, headers, body_json, body_raw?}`), `requests_for(prefix, method=None)`, `last_request()`, `clear()`, `state`,
`lock`, `errors`, `options`, `brain`); `register_kind(name, factory)` where `factory(server) -> handle(req, resp)`;
`get_kind_factory(name)` (imports `MOCK_MODULES` = mock_openai_chat, mock_responses, mock_gemini, mock_anthropic,
mock_auth on demand); `MockRequest` (`method, path, query, raw_path, headers, body, json, header(), bearer()`);
`MockResponder` (`send_json`, `send_text`, `send_bytes`, `start_sse() -> SSEWriter`, `start_chunked() -> ChunkWriter`,
`close_connection()`); `SSEWriter.event(name, data)`, `.data(obj|str)`, `.comment()`, `.raw(bytes)`, `.done()`,
`.close()`; `Brain(command="printf hello > out.txt && env", tool="Bash", background_text, thinking)` with
`find_tool(offered)`, `decide(offered_tools, tool_results, background=False) -> BrainReply(kind, tool_name,
arguments, text, thinking)`, `leaks`, `decisions`; `sha8`, `sse_event_bytes`, `sse_data_bytes`,
`anthropic_brain_inputs(body)`; built-in kind `echo`.

**`tools/vendor_gateway.py`** — `python tools/vendor_gateway.py <fry> [--force] [--dry-run] [--check]`; library
`vendor(fry_repo, src_root, force, dry_run, check, out) -> exit code` (0 ok/in sync, 1 out of sync, 2 refused).

### 8.2 Expected from Phase-1 agents (implement exactly these names)
**A — `anthropic_in.py`**: `parse_messages_request(body: dict, headers: Mapping[str, str]) -> NormalizedRequest`
(applies `fold_system_messages_raw` + `apply_tool_changes`; malformed → `GatewayError(400, "invalid_request_error")`).
**A — `anthropic_out.py`**: `from .errors import GatewayError, map_upstream_error, prompt_too_long` (re-export only);
`new_message_id() -> "msg_<24hex>"`; `SSEEmitter(write: Callable[[bytes], None], requested_model: str,
message_id: Optional[str] = None, input_estimate: int = 0)` with `start()`, `ping()`, `feed(event)`, `finish()`
(idempotent), `error(err_type, message)`, `committed: bool`; `Aggregator(requested_model, message_id=None,
input_estimate=0)` with `feed(event)`, `result() -> dict` (Anthropic Message JSON; a fed StreamError makes
`result()` raise the equivalent `GatewayError`); `error_response(err) -> (status, headers, body_bytes)`;
`estimate_tokens(req: NormalizedRequest) -> int` (§2.4 formula).
**A — `router.py`**: `class Resolution` with attributes `provider: ProviderSpec` (primary, never the fallback),
`model: str` (upstream id, `[1m]` stripped), `model_spec: ModelSpec`, `requested: str` (verbatim),
`role: Optional[str]` (`default|background|longContext|subagent` or None for explicit routes), `background: bool`;
`class RouteNotFound(errors.GatewayError)` (404 `not_found_error`, message per §5.2);
`Router(table: RouteTable, discovered: Optional[Dict[str, List[str]]] = None)` with `resolve(requested,
est_tokens=None) -> Resolution`, `set_discovered(provider_id, model_ids)`, `models_response() -> dict`
(`/v1/models` body), `model_entry(model_id) -> Optional[dict]`.
**A — `server.py`**: `Gateway(table: RouteTable, secrets: SecretStore, host="127.0.0.1", port=0, token=None,
log=None, trace_file=None, http=None)` with `start() -> Gateway` (bound + serving on a daemon thread on return),
`stop()`, `url`, `port`, `token`, context manager, `auth_for(provider_id) -> AuthProvider`,
`runtime_for(provider_id) -> ProviderRuntime`. One AuthProvider + ProviderRuntime per provider id (fallback reuses
the parent's AuthProvider when its auth dict is equal). Builds `RequestContext` and dispatches
`get_dialect(provider.effective_dialect(model_spec))`. **A — `tracing.py`**: `setup_logging(level="INFO",
log_file=None, secrets=None) -> logging.Logger` (logger `"ai_gateway"`), `Tracer(path, secrets=None)` callable
`(name, fields)`, `redact(text, secrets)`. **A — `__main__.py`** (`serve` command, §1).
**B — `filelock.py`**: `ExclusiveFileLock(path, timeout=10.0, poll=0.05)` (context manager, `acquire()`,
`release()`, raises `LockTimeout(TimeoutError)`). **B — `atomicio.py`**: `atomic_write_bytes(path, data, mode=0o600)`,
`atomic_write_json(path, obj, indent=2, mode=0o600)`, `restrict_permissions(path)`. **B — `secrets.py`**:
`resolve_many(specs: Dict[str, List[str]], timeout=15.0) -> Dict[str, Optional[str]]` (sources tried in order:
`"env:VAR"`, `"op://…"`, `"literal:value"`, `"json:<path>#<dotted.key>"`; parallel, memoized per process, never
logged) and `load_into(store: SecretStore, specs) -> Dict[str, str]` (name → source label). **B — auth**:
`CodexChatGPTAuth(provider_id, options)` (`options["auth_path"]` optional; headers `Authorization`,
`ChatGPT-Account-ID`; `account_id()`), `GrokCliAuth(provider_id, options)` (`api_key_entry() -> Optional[str]`),
`GcloudADCAuth(provider_id, options)` (`options` project/location/adc_path; headers `Authorization`,
`x-goog-user-project`; `project()`, `location()`), all `RefreshingAuth` subclasses honouring §4 overrides.
**B — testing**: `mock_auth.py` kinds `openai_oauth`, `xai_oidc`, `google_oauth`; `fake_bins.py`
`make_fake_bins(directory, **behaviour) -> str` (dir to prepend to PATH; `.cmd` twins on Windows).
**C — `dialects/openai_chat.py`** `OpenAIChatDialect`; **`chat_profiles.py`** `ChatProfile` + `get_profile(name)`
(None → generic); **`toolnames.py`** `ToolNameMap(names, regex=None, maxlen=64, leading_letter=False)` with
`upstream(name)`, `original(name)` (unknown → unchanged), and `encode_tool_id(id)`, `decode_tool_id(id)`,
`new_tool_id()` (§3.0 Tool ids); **`schema.py`** `scrub(schema, mode) -> dict` (never mutates input);
**`testing/mock_openai_chat.py`** kinds `xai_chat`, `openai_chat`, `moonshot_chat`, `deepseek_chat`, `nvidia_chat`,
`ollama_chat` (incl. `/api/show`, `/api/tags`), `openrouter_chat`, `generic_chat`.
**D — `dialects/responses.py`** `ResponsesDialect`; **`responses_targets.py`** `get_target(name) ->
ResponsesTarget` (url path, headers builder, body rules), `codex_version()`, `grok_client_version()`;
**`testing/mock_responses.py`** kinds `openai_responses`, `chatgpt_codex`, `grok_proxy` (`options["version_gate"]`),
`xai_responses`. **E — `dialects/gemini.py`** `GeminiDialect`; **`testing/mock_gemini.py`** kinds `gemini_api`,
`vertex`. **F — `dialects/anthropic_passthrough.py`** `AnthropicPassthroughDialect`; **`dialects/cli.py`**
`CliDialect`; **`launchkit.py`** `find_claude() -> List[str]` (argv prefix), `build_child_env(base_env, base_url,
token, default_id, background_id, context=None, extra=None) -> Dict[str, str]`, `build_direct_env(base_env,
base_url, token, models: Dict[str, str], extra=None) -> Dict[str, str]`, `run_child(argv, env, cwd=None) -> int`,
`claude_version(argv_prefix=None) -> Optional[str]`; presets/catalog content; **`testing/mock_anthropic.py`** kind
`anthropic`; **`testing/claude_e2e.py`** `run_e2e(launcher_argv, env, workdir, timeout=300) -> E2EResult` +
`claude_available()`.
Shared rule: each agent writes its own tests in `tests/gateway/test_<its module>.py` and registers mock kinds only
in its own `testing/mock_*.py`. Unknown cross-agent needs → ask the coordinator, never edit another owner's file.

## 9. Module ownership (Phase 1)
| Owner | Files |
|---|---|
| Phase 0 (frozen) | `__init__.py`, `VERSION`, `compat.py`, `events.py`, `model.py`, `errors.py`, `signatures.py`, `config.py`, `auth/__init__.py`, `auth/base.py`, `auth/static.py`, `dialects/__init__.py`, `dialects/base.py`, `testing/__init__.py`, `testing/mock_upstreams.py`, `testing/fixtures/`, `tests/__init__.py`, `tests/gateway/{__init__,_pkg,test_foundation,test_errors,test_vendor,test_py38_compat}.py`, `tools/vendor_gateway.py` |
| A core | `anthropic_in.py`, `anthropic_out.py`, `router.py`, `server.py`, `tracing.py`, `__main__.py` |
| B platform | `transport.py` internals (keep-alive pool, stale retry, CONNECT proxy; signatures frozen), `filelock.py`, `atomicio.py`, `secrets.py`, `auth/codex_chatgpt.py`, `auth/grok_cli.py`, `auth/gcloud_adc.py`, `testing/mock_auth.py`, `testing/fake_bins.py` |
| C chat | `dialects/openai_chat.py`, `chat_profiles.py`, `toolnames.py`, `schema.py`, `testing/mock_openai_chat.py` |
| D responses | `dialects/responses.py`, `responses_targets.py`, `testing/mock_responses.py` |
| E gemini | `dialects/gemini.py`, `testing/mock_gemini.py` |
| F edges | `dialects/anthropic_passthrough.py`, `dialects/cli.py`, `launchkit.py`, `presets.py` + `catalog.py` (content), `testing/mock_anthropic.py`, `testing/claude_e2e.py` |
