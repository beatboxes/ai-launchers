#!/usr/bin/env bash
# install.sh — ai-launchers installer (POSIX). Creates shims in ~/.ai-launchers/bin for
# all 5 launchers and prepends that dir to PATH (via ~/.bashrc append, idempotent).
# Reversal: uninstall.sh.
set -eu
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${AIL_BIN:-$HOME/.ai-launchers/bin}"
PROVIDERS="grok codex gemini deepseek kimi"

echo "== ai-launchers install =="
echo "repo:    $REPO"
echo "bin dir: $BIN_DIR"

if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python3/python not found" >&2; exit 1
fi
PY="$(command -v python3 || command -v python)"
if ! command -v ccr >/dev/null 2>&1; then
  echo "WARN: ccr not on PATH — 'launch' will fail. Install: npm i -g @musistudio/claude-code-router" >&2
fi
if ! command -v claude >/dev/null 2>&1; then
  echo "WARN: claude CLI not on PATH — 'launch' will fail. Install: npm i -g @anthropic-ai/claude-code" >&2
fi

mkdir -p "$BIN_DIR"
for p in $PROVIDERS; do
  # Every launcher is suffixed "-wrap" so its shim doesn't shadow a native
  # grok/codex binary on PATH.
  shimName="$p-wrap"
  py="$REPO/$p/$shimName.py"
  [ -f "$py" ] || { echo "missing $py — skipping $p"; continue; }
  shim="$BIN_DIR/$shimName"
  cat > "$shim" <<EOF
#!/usr/bin/env bash
exec "$PY" "$py" "\$@"
EOF
  chmod +x "$shim"
  echo "  installed: $shim"
done

# add to PATH via ~/.bashrc (idempotent)
if [ -f "$HOME/.bashrc" ] && ! grep -qF "$BIN_DIR" "$HOME/.bashrc" 2>/dev/null; then
  echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.bashrc"
  echo "PATH: added $BIN_DIR to ~/.bashrc (start a new shell to pick it up)"
else
  echo "PATH: $BIN_DIR already in ~/.bashrc (or no .bashrc)"
fi
echo
echo "Done. Try:  grok-wrap --help"