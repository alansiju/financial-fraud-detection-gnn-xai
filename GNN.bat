@echo off
REM ============================================================================
REM  FraudShield AI - one-click launcher
REM
REM  Double-click to run the whole app. It will:
REM    1. check the Python venv, the model artifacts and Node are all present
REM    2. install frontend packages if node_modules is missing
REM    3. start the backend in its own window (or reuse one already running)
REM    4. wait until the backend has actually finished loading - it loads a
REM       1.2 GB graph and warms the prediction cache, about a minute
REM    5. start the frontend in its own window (or reuse one already running)
REM    6. open the dashboard in your browser
REM
REM  Stop the app by closing the two server windows.
REM
REM  Ports are fixed: the backend's CORS policy only allows the frontend on
REM  5173, so if something else holds a port the app cannot work. The script
REM  says so plainly instead of starting a half-broken copy.
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title FraudShield AI - Launcher
set "PYTHONIOENCODING=utf-8"

set "BACKEND=http://127.0.0.1:8000"
set "FRONTEND=http://127.0.0.1:5173"
set "PY=%~dp0.venv\Scripts\python.exe"

echo.
echo  ==========================================
echo    FraudShield AI - Advanced Launcher
echo  ==========================================
echo.

REM -- 1. Pre-flight ---------------------------------------------------------
echo  [1/5] Checking environment...

where curl.exe >nul 2>&1 || (
  echo  [ERROR] curl.exe not found. It ships with Windows 10 1803 and later.
  goto :fail
)

if not exist "%PY%" (
  echo  [ERROR] Python venv not found at .venv
  echo          Create it with:
  echo            py -3.12 -m venv .venv
  echo            .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  echo            .venv\Scripts\python -m pip install -r requirements.txt
  goto :fail
)

set "MISSING="
for %%F in (
  "fraud_graph.pt"
  "outputs\models\best_model.pt"
  "outputs\models\preprocessor.pkl"
  "outputs\models\edge_attr_lookup.pkl"
  "outputs\models\node_meta.pkl"
  "outputs\models\lightgbm.pkl"
  "outputs\models\ensemble.json"
) do (
  if not exist "%%~F" (
    echo  [ERROR] Missing model artifact: %%~F
    set "MISSING=1"
  )
)
if defined MISSING (
  echo          Rebuild with the training pipeline - see FIXES.md.
  goto :fail
)

where node >nul 2>&1 || (
  echo  [ERROR] Node.js not found on PATH. Install it from https://nodejs.org
  goto :fail
)

if not exist "frontend\node_modules" (
  echo  [INFO] Installing frontend packages - first run only...
  pushd frontend
  call npm install || ( popd & echo  [ERROR] npm install failed. & goto :fail )
  popd
)
echo        OK

REM -- 2. Backend ------------------------------------------------------------
echo  [2/5] Backend...
call :is_up "%BACKEND%/docs"
if !UP!==1 (
  echo        already running on %BACKEND% - reusing it
) else (
  call :port_busy 8000
  if !BUSY!==1 (
    echo  [ERROR] Port 8000 is held by another program that is not this backend.
    echo          Close it, or find it with:  netstat -ano ^| findstr :8000
    goto :fail
  )
  start "FraudShield Backend" /D "%~dp0" cmd /k ""%PY%" -m uvicorn server.main:app --host 127.0.0.1 --port 8000"
  echo        started in its own window
)

REM -- 3. Wait for the backend to finish loading -----------------------------
echo  [3/5] Waiting for the model to load (about a minute on first start)...
set /a TRIES=0
:wait_backend
call :is_up "%BACKEND%/docs"
if !UP!==1 goto :backend_ready
set /a TRIES+=1
if !TRIES! geq 180 (
  echo  [ERROR] Backend did not come up within 3 minutes.
  echo          Check the "FraudShield Backend" window for the error.
  goto :fail
)
<nul set /p "=."
ping -n 2 127.0.0.1 >nul
goto :wait_backend
:backend_ready
echo.
echo        ready

REM -- 4. Frontend -----------------------------------------------------------
echo  [4/5] Frontend...
call :is_up "%FRONTEND%/"
if !UP!==1 (
  echo        already running on %FRONTEND% - reusing it
  goto :frontend_ready
)
call :port_busy 5173
if !BUSY!==1 (
  echo  [ERROR] Port 5173 is held by another program.
  echo          The backend only accepts the app from 5173, so it must be free.
  echo          Find it with:  netstat -ano ^| findstr :5173
  goto :fail
)
start "FraudShield Frontend" /D "%~dp0frontend" cmd /k "npm run dev -- --port 5173 --strictPort --host 127.0.0.1"
set /a TRIES=0
:wait_frontend
call :is_up "%FRONTEND%/"
if !UP!==1 goto :frontend_ready
set /a TRIES+=1
if !TRIES! geq 60 (
  echo  [ERROR] Frontend did not come up within 60 seconds.
  echo          Check the "FraudShield Frontend" window for the error.
  goto :fail
)
ping -n 2 127.0.0.1 >nul
goto :wait_frontend
:frontend_ready
echo        ready

REM -- 5. Open ---------------------------------------------------------------
echo  [5/5] Opening the dashboard...
start "" "%FRONTEND%/"

echo.
echo  ==========================================
echo    Running
echo      Dashboard : %FRONTEND%
echo      API docs  : %BACKEND%/docs
echo      Sign in   : admin / fraud123
echo.
echo    To stop: close the "FraudShield Backend"
echo    and "FraudShield Frontend" windows.
echo  ==========================================
echo.
echo  This window closes in 15 seconds.
ping -n 16 127.0.0.1 >nul
exit /b 0

REM -- helpers ---------------------------------------------------------------
:is_up
set "UP=0"
curl.exe -s -o nul -m 2 -f "%~1" && set "UP=1"
exit /b 0

:port_busy
set "BUSY=0"
netstat -ano | findstr /R /C:"127.0.0.1:%1 .*LISTENING" /C:"0.0.0.0:%1 .*LISTENING" /C:"\[::\]:%1 .*LISTENING" /C:"\[::1\]:%1 .*LISTENING" >nul && set "BUSY=1"
exit /b 0

:fail
echo.
echo  Launch aborted.
pause
exit /b 1