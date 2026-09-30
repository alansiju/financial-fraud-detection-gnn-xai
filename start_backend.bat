@echo off
REM ============================================================================
REM  FraudShield AI - Backend
REM
REM  Was hardcoded to "C:\Users\Alan siju\...\Python311\python.exe" - another
REM  machine's user account, and a Python version that is not installed here.
REM  Now uses the project's own virtual environment via a relative path, so it
REM  runs wherever the repo is checked out.
REM
REM  Bound to 127.0.0.1 rather than 0.0.0.0: this is a local app, the CORS
REM  policy only allows localhost anyway, and there is no reason to publish it
REM  to the rest of the network.
REM
REM  --reload is deliberately NOT used. Startup loads a 1.2 GB graph and scores
REM  all 590,540 nodes to warm the prediction cache; reloading on every file
REM  save would repeat that each time.
REM ============================================================================
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [ERROR] .venv not found. Create it first:
  echo     py -3.12 -m venv .venv
  echo     .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  echo     .venv\Scripts\python -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

echo ==========================================
echo   FraudShield AI - Backend
echo ==========================================
echo.
echo Loading the graph and warming the prediction cache.
echo First start takes around a minute; the server is not
echo ready until it prints "Application startup complete".
echo.

".venv\Scripts\python.exe" -m uvicorn server.main:app --host 127.0.0.1 --port 8000
pause
