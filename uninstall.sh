#!/usr/bin/env bash
# uninstall.sh — reverse install.sh (POSIX)
set -eu
BIN_DIR="${AIL_BIN:-$HOME/.ai-launchers/bin}"
PROVIDERS="grok codex gemini deepseek kimi"
# Remove both legacy (pre-rename "<p>") and current ("<p>-wrap") shims so an
# upgrade from the old names cleans up cleanly.
for p in $PROVIDERS; do
  for shimName in "$p" "$p-wrap"; do
    shim="$BIN_DIR/$shimName"
    [ -f "$shim" ] && { rm -f "$shim"; echo "removed: $shim"; }
  done
done
if [ -f "$HOME/.bashrc" ]; then
  if grep -qF "$BIN_DIR" "$HOME/.bashrc" 2>/dev/null; then
    # remove the line we added (idempotent best-effort)
    sed -i "\@export PATH=\"$BIN_DIR:\$PATH\"@d" "$HOME/.bashrc"
    echo "PATH: removed $BIN_DIR from ~/.bashrc"
  fi
fi
echo "Uninstall complete."