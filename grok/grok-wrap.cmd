@echo off
setlocal
rem grok-wrap shim for cmd/PowerShell: prefer the py launcher (Python 3), fall back to python.
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%~dp0grok-wrap.py" %*
) else (
  python "%~dp0grok-wrap.py" %*
)
exit /b %errorlevel%
