@echo off
setlocal
rem gemini-wrap shim for cmd/PowerShell: prefer the py launcher (Python 3), fall back to python.
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%~dp0gemini-wrap.py" %*
) else (
  python "%~dp0gemini-wrap.py" %*
)
exit /b %errorlevel%
