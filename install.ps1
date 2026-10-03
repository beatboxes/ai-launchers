# install.ps1 - ai-launchers installer (Windows).
# Resolves ONE Python 3.8+ interpreter (the py launcher with -3, else a real python.exe - the
# Microsoft Store "WindowsApps" alias stub is rejected), writes .cmd shims with that absolute path to
# %USERPROFILE%\.ai-launchers\bin and adds that directory to the user PATH. Idempotent.
# Reversal: uninstall.ps1
[CmdletBinding()]
param(
    [string]$BinDir = (Join-Path $env:USERPROFILE ".ai-launchers\bin"),
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$providers = @("grok", "codex", "gemini", "deepseek", "kimi")

Write-Host "== ai-launchers install =="
Write-Host "repo:    $repo"
Write-Host "bin dir: $BinDir"

function Test-Python([string]$Exe, [string[]]$PreArgs) {
    try {
        & $Exe @PreArgs -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

$pyExe = $null
$pyArgs = @()
$launcher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($launcher -and (Test-Python $launcher.Source @("-3"))) {
    $pyExe = $launcher.Source
    $pyArgs = @("-3")
} else {
    foreach ($cand in @(Get-Command python.exe -CommandType Application -All -ErrorAction SilentlyContinue)) {
        if ($cand.Source -like "*\WindowsApps\*") { continue }   # Store alias stub, not a real Python
        if (Test-Python $cand.Source @()) { $pyExe = $cand.Source; break }
    }
}
if (-not $pyExe) {
    Write-Error "Python 3.8+ not found. Install it from https://www.python.org/downloads/ (includes the py launcher) and re-run."
    exit 1
}
$pyDesc = (& $pyExe @pyArgs -c "import platform, sys; print(sys.executable, platform.python_version())")
Write-Host "python:  $pyExe $($pyArgs -join ' ')  ($pyDesc)"

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Warning "Claude Code (claude) is not on PATH - 'launch' will fail until you install it: npm i -g @anthropic-ai/claude-code"
}

if ($DryRun) { Write-Host "[dry-run] would create $($providers.Count) shims in $BinDir using $pyExe"; exit 0 }

if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir -Force | Out-Null }

$prefix = "`"$pyExe`""
if ($pyArgs.Count -gt 0) { $prefix = "$prefix $($pyArgs -join ' ')" }
foreach ($p in $providers) {
    # Every launcher is suffixed "-wrap" so its shim never shadows a native grok/codex binary.
    $name = "$p-wrap"
    $script = Join-Path $repo "$p\$name.py"
    if (-not (Test-Path $script)) { Write-Warning "missing $script - skipping $p"; continue }
    $shim = Join-Path $BinDir "$name.cmd"
    $body = "@echo off`r`n$prefix `"$script`" %*`r`nexit /b %errorlevel%`r`n"
    Set-Content -Path $shim -Value $body -Encoding ASCII -NoNewline
    Write-Host "  installed: $shim"
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$entries = @()
if ($userPath) { $entries = $userPath -split ';' | Where-Object { $_ } }
if ($entries -notcontains $BinDir) {
    $newPath = (@($entries) + $BinDir) -join ';'
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "PATH: added $BinDir (open a new terminal to pick it up)"
} else {
    Write-Host "PATH: $BinDir already present"
}

Write-Host "`nDone. Try:  grok-wrap --help   then:  grok-wrap doctor   (after reopening the terminal)"
Write-Host "Reversal:  .\uninstall.ps1"
