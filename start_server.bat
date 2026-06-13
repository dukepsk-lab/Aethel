@echo off
REM ============================================================
REM  Aethel - Windows VPS startup script
REM  Double-click (or run from cmd) to start everything.
REM
REM  Prerequisites (one-time setup):
REM    1. Python 3.11+ installed (python.org)
REM    2. PostgreSQL 16 + pgvector extension installed
REM    3. MetaTrader 5 terminal running and logged in
REM    4. .env file created from .env.example
REM    5. Run setup_windows.bat once first
REM ============================================================

setlocal enabledelayedexpansion
title Aethel Command Center

REM -- locate Python (prefer venv if present) --
if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
    set PIP=venv\Scripts\pip.exe
) else (
    set PYTHON=python
    set PIP=pip
)

REM -- verify .env exists --
if not exist ".env" (
    echo ERROR: .env not found.
    echo Copy .env.example to .env and fill in your API keys.
    pause
    exit /b 1
)

REM -- read API port from .env (default 8000) --
set API_PORT=8000
for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
    if "%%A"=="AETHEL_API_PORT" set API_PORT=%%B
)

echo.
echo  ========================================
echo   Aethel Command Center
echo   API       : http://localhost:%API_PORT%
echo   Dashboard : open frontend in browser
echo  ========================================
echo.

REM -- check PostgreSQL is reachable --
echo [1/4] Checking PostgreSQL...
%PYTHON% -c "import asyncio, asyncpg; asyncio.run(asyncpg.connect('postgresql://aethel:aethel@localhost:5432/aethel'))" 2>nul
if errorlevel 1 (
    echo       PostgreSQL not reachable. Starting service...
    net start postgresql-x64-16 2>nul
    if errorlevel 1 net start postgresql 2>nul
    timeout /t 3 /nobreak >nul
)
echo       OK

REM -- initialise DB tables (safe to re-run) --
echo [2/4] Initialising database...
%PYTHON% -c "import asyncio; from aethel.db.session import init_db; asyncio.run(init_db())"
if errorlevel 1 (
    echo ERROR: DB init failed. Check DATABASE_URL in .env and PostgreSQL service.
    pause
    exit /b 1
)
echo       OK

REM -- start Next.js frontend in a new window --
echo [3/4] Starting frontend...
if not exist "frontend\package.json" goto skip_frontend
if exist "frontend\node_modules" goto start_frontend
echo       Installing node modules (first run, takes a few minutes)...
pushd frontend
call npm install --silent
popd
:start_frontend
start "Aethel Frontend" /min cmd /c "cd /d "%~dp0frontend" && npm start"
echo       Frontend starting at http://localhost:3000
goto after_frontend
:skip_frontend
echo       frontend folder not found - skipping
:after_frontend

REM -- start API + orchestrator --
echo [4/4] Starting Aethel API + orchestrator...
echo.
echo  Press Ctrl+C to stop.
echo.
%PYTHON% -m aethel.api.main
