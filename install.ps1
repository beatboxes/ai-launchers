# install.ps1 — ai-launchers installer (Windows)
# Creates .cmd shims in ~/.ai-launchers/bin for all 5 launchers, adds that dir to user PATH.
# Foreground, idempotent. Reversal: uninstall.ps1 (or remove shims + PATH entry).
[CmdletBinding()]
param(
    [string]$BinDir = (Join-Path $env:USERPROFILE ".ai-launchers\bin"),
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$providers = @("grok","codex","gemini","deepseek","kimi")

Write-Host "== ai-launchers install =="
Write-Host "repo:    $repo"
Write-Host "bin dir: $BinDir"

# --- dep checks ---
$missing = @()
if (-not (Get-Command py -ErrorAction SilentlyContinue) -and
    -not (Get-Command python -ErrorAction SilentlyContinue)) {
    $missing += "python (py launcher)"
}
if (-not (Get-Command ccr -ErrorAction SilentlyContinue)) {
    Write-Warning "ccr (claude-code-router) not on PATH — `launch` will fail. Install: npm i -g @musistudio/claude-code-router"
}
if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Warning "claude CLI not on PATH — `launch` will fail. Install: npm i -g @anthropic-ai/claude-code"
}
if ($missing.Count -gt 0) {
    Write-Error "Missing: $($missing -join ', ')"
    exit 1
}

if ($DryRun) { Write-Host "[dry-run] would create $($providers.Count) shims in $BinDir"; exit 0 }

if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir -Force | Out-Null }

foreach ($p in $providers) {
    # Every launcher is suffixed "-wrap" so its shim doesn't shadow a native
    # grok/codex binary on PATH.
    $shimName = "$p-wrap"
    $py = Join-Path $repo "$p\$shimName.py"
    if (-not (Test-Path $py)) { Write-Warning "missing $py — skipping $p"; continue }
    $shim = Join-Path $BinDir "$shimName.cmd"
    $body = "@echo off`r`npython `"$py`" %*`r`n"
    Set-Content -Path $shim -Value $body -Encoding ASCII -NoNewline
    Write-Host "  installed: $shim"
}

# add bin dir to user PATH if not present
$userPath = [Environment]::GetEnvironmentVariable("Path","User")
if ($userPath -notlike "*$BinDir*") {
    $newPath = if ($userPath) { "$userPath;$BinDir" } else { $BinDir }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "PATH: added $BinDir (open a new terminal to pick it up)"
} else {
    Write-Host "PATH: $BinDir already present"
}

Write-Host "`nDone. Try:  grok-wrap --help    (after reopening the terminal)"
Write-Host "Reversal:  remove shims in $BinDir and the PATH entry (see uninstall.ps1)."