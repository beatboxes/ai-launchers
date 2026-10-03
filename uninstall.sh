#!/usr/bin/env bash
# uninstall.sh — reverse install.sh (macOS / Linux). Removes the shims and every PATH line tagged
# "# ai-launchers" (plus the untagged line written by v0.1) from your shell rc files.
# Portable: BSD and GNU sed (sed -i.bak), no GNU-only flags. Leaves ~/.ai-launchers config/keys alone.
set -eu

BIN_DIR="${AIL_BIN:-$HOME/.ai-launchers/bin}"
PROVIDERS="grok codex gemini deepseek kimi"
LEGACY="export PATH=\"$BIN_DIR:\$PATH\""

# Current ("<p>-wrap") and pre-rename ("<p>") shim names.
for p in $PROVIDERS; do
  for name in "$p" "$p-wrap"; do
    shim="$BIN_DIR/$name"
    if [ -f "$shim" ]; then rm -f "$shim"; echo "removed: $shim"; fi
  done
done
rmdir "$BIN_DIR" 2>/dev/null || true

for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile" "$HOME/.bash_profile"; do
  [ -f "$rc" ] || continue
  changed=0
  if grep -q '# ai-launchers$' "$rc"; then
    sed -i.bak '/# ai-launchers$/d' "$rc" && rm -f "$rc.bak"
    changed=1
  fi
  if grep -qxF "$LEGACY" "$rc"; then  # v0.1 line (exact match, no regex surprises from the path)
    tmp="$rc.ai-launchers.tmp"
    grep -vxF "$LEGACY" "$rc" > "$tmp" || true
    cat "$tmp" > "$rc" && rm -f "$tmp"
    changed=1
  fi
  if [ "$changed" = 1 ]; then echo "PATH: removed ai-launchers entry from $rc"; fi
done
echo "Uninstall complete. (Config, keys and logs remain in ~/.ai-launchers — delete it to remove them.)"
