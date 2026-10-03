# uninstall.ps1 - reverse install.ps1 (Windows). Removes the shims and the user PATH entry.
# Leaves %USERPROFILE%\.ai-launchers config, keys and logs in place.
[CmdletBinding()]
param(
    [string]$BinDir = (Join-Path $env:USERPROFILE ".ai-launchers\bin")
)
$ErrorActionPreference = "Continue"
$providers = @("grok", "codex", "gemini", "deepseek", "kimi")
# Current ("<p>-wrap.cmd") and pre-rename ("<p>.cmd") shim names.
foreach ($p in $providers) {
    foreach ($name in @($p, "$p-wrap")) {
        $shim = Join-Path $BinDir "$name.cmd"
        if (Test-Path $shim) { Remove-Item $shim -Force; Write-Host "removed: $shim" }
    }
}
if ((Test-Path $BinDir) -and -not (Get-ChildItem $BinDir -Force | Select-Object -First 1)) {
    Remove-Item $BinDir -Force
}
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath) {
    $parts = $userPath -split ';' | Where-Object { $_ }
    if ($parts -contains $BinDir) {
        $kept = $parts | Where-Object { $_ -ne $BinDir }
        [Environment]::SetEnvironmentVariable("Path", ($kept -join ';'), "User")
        Write-Host "PATH: removed $BinDir"
    }
}
Write-Host "Uninstall complete. (Config, keys and logs remain in $(Split-Path $BinDir -Parent).)"
