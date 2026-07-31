# uninstall.ps1 — reverse install.ps1 (Windows)
[CmdletBinding()]
param(
    [string]$BinDir = (Join-Path $env:USERPROFILE ".ai-launchers\bin")
)
$ErrorActionPreference = "Continue"
$providers = @("grok","codex","gemini","deepseek","kimi")
# Remove both legacy (pre-rename "<p>.cmd") and current ("<p>-wrap.cmd") shims
# so an upgrade from the old names cleans up cleanly.
foreach ($p in $providers) {
    foreach ($shimName in @($p, "$p-wrap")) {
        $shim = Join-Path $BinDir "$shimName.cmd"
        if (Test-Path $shim) { Remove-Item $shim -Force; Write-Host "removed: $shim" }
    }
}
$userPath = [Environment]::GetEnvironmentVariable("Path","User")
if ($userPath -and ($userPath -like "*$BinDir*")) {
    $parts = $userPath -split ';' | Where-Object { $_ -and $_ -ne $BinDir }
    [Environment]::SetEnvironmentVariable("Path", ($parts -join ';'), "User")
    Write-Host "PATH: removed $BinDir"
}
Write-Host "Uninstall complete."