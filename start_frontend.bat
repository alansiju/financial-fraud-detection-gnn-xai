@echo off
REM ============================================================================
REM  FraudShield AI - Frontend
REM
REM  Was "cd /d D:\projects\GNN\frontend" - a drive that does not exist on this
REM  machine. Now relative to this script.
REM
REM  Port 5173 is pinned with --strictPort: server/main.py only allows
REM  http://localhost:5173 and http://127.0.0.1:5173 through CORS, so if Vite
REM  silently falls forward to 5174 every API call is blocked and the app looks
REM  broken with no obvious cause.
REM ============================================================================
setlocal
cd /d "%~dp0frontend"

if not exist "node_modules" (
  echo.
  echo [INFO] node_modules missing - installing dependencies first.
  echo.
  call npm install
)

echo ==========================================
echo   FraudShield AI - Frontend
echo ==========================================
echo.
echo   http://localhost:5173
echo.
echo   Sign in with  admin / fraud123
echo.
echo   The backend must already be running on port 8000
echo   (start_backend.bat).
echo.

call npm run dev -- --port 5173 --strictPort --host 127.0.0.1
pause
