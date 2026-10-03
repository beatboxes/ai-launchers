#!/usr/bin/env bash
# install.sh — ai-launchers installer (macOS / Linux).
# Creates shims in ~/.ai-launchers/bin (override: AIL_BIN) that run each launcher with an ABSOLUTE
# python3 (>= 3.8) path, and adds that directory to PATH via one line tagged "# ai-launchers" in your
# shell rc file(s). Idempotent. Reversal: ./uninstall.sh
set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${AIL_BIN:-$HOME/.ai-launchers/bin}"
PROVIDERS="grok codex gemini deepseek kimi"
TAG="# ai-launchers"

echo "== ai-launchers install =="
echo "repo:    $REPO"
echo "bin dir: $BIN_DIR"

py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; }
PY=""
for cand in python3 python; do
  p="$(command -v "$cand" 2>/dev/null || true)"
  if [ -n "$p" ] && py_ok "$p"; then PY="$p"; break; fi
done
if [ -z "$PY" ]; then
  echo "ERROR: Python >= 3.8 not found (looked for python3, python). Install Python 3 and re-run." >&2
  exit 1
fi
echo "python:  $PY ($("$PY" -c 'import platform; print(platform.python_version())'))"

if ! command -v claude >/dev/null 2>&1; then
  echo "WARN: Claude Code (claude) is not on PATH — 'launch' will fail until you install it:" >&2
  echo "      npm i -g @anthropic-ai/claude-code" >&2
fi

mkdir -p "$BIN_DIR"
for p in $PROVIDERS; do
  # Every launcher is suffixed "-wrap" so its shim never shadows a native grok/codex binary.
  name="$p-wrap"
  script="$REPO/$p/$name.py"
  if [ ! -f "$script" ]; then echo "  missing $script — skipping $p"; continue; fi
  shim="$BIN_DIR/$name"
  cat > "$shim" <<EOF
#!/bin/sh
exec "$PY" "$script" "\$@"
EOF
  chmod +x "$shim"
  echo "  installed: $shim"
done

LINE="export PATH=\"$BIN_DIR:\$PATH\"  $TAG"
add_line() {  # add_line <rc file>: append the tagged PATH line once (creates the file if needed)
  if [ -f "$1" ] && grep -qxF "$LINE" "$1" 2>/dev/null; then
    echo "PATH: already in $1"
  else
    if [ -s "$1" ] && [ -n "$(tail -c 1 "$1")" ]; then printf '\n' >> "$1"; fi
    printf '%s\n' "$LINE" >> "$1"
    echo "PATH: added $BIN_DIR to $1"
  fi
}

login_shell="$(basename "${SHELL:-sh}")"
added=0
if [ -f "$HOME/.bashrc" ]; then add_line "$HOME/.bashrc"; added=1; fi
if [ -f "$HOME/.zshrc" ] || [ "$login_shell" = "zsh" ]; then
  add_line "$HOME/.zshrc"   # created when zsh is the login shell (macOS default) and it doesn't exist yet
  added=1
fi
if [ "$added" = 0 ]; then
  if [ -f "$HOME/.bash_profile" ]; then add_line "$HOME/.bash_profile"; else add_line "$HOME/.profile"; fi
fi
if [ "$login_shell" = "fish" ]; then
  echo "PATH: fish detected — run once:  fish_add_path $BIN_DIR"
fi

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "Open a new shell (or run: export PATH=\"$BIN_DIR:\$PATH\") to use the launchers." ;;
esac
echo
echo "Done. Try:  grok-wrap --help    then:  grok-wrap doctor"
