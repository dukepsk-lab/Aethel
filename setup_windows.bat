@echo off
REM ============================================================
REM  Aethel — Windows VPS one-time setup
REM  Run this ONCE after cloning the repo.
REM
REM  Before running:
REM    1. Install Python 3.11+ from https://python.org
REM       (tick "Add to PATH" during install)
REM    2. Install PostgreSQL 16 from https://www.postgresql.org/download/windows/
REM       - Remember the postgres superuser password
REM    3. Install pgvector for Windows:
REM       https://github.com/pgvector/pgvector/releases
REM       (download the .zip, copy vector.dll + vector.control + vector--*.sql
REM        into your PostgreSQL lib/ and share/extension/ folders)
REM    4. Install MetaTrader 5 and log in
REM ============================================================

setlocal
title Aethel Setup

echo.
echo  ========================================
echo   Aethel — one-time Windows setup
echo  ========================================
echo.

REM ── Python version check ────────────────────────────────────
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause & exit /b 1
)

REM ── create virtual environment ──────────────────────────────
echo [1/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
)
call venv\Scripts\activate.bat

REM ── install Python dependencies ─────────────────────────────
echo [2/5] Installing Python dependencies (this takes a few minutes)...
pip install -e ".[mt5]" --quiet
if errorlevel 1 (
    echo ERROR: pip install failed. Check your internet connection.
    pause & exit /b 1
)
echo       OK

REM ── create .env if missing ──────────────────────────────────
echo [3/5] Creating .env...
if not exist ".env" (
    copy .env.example .env
    echo       .env created from .env.example
    echo       IMPORTANT: open .env and fill in your API keys before starting!
) else (
    echo       .env already exists — skipping
)

REM ── create PostgreSQL database + user ───────────────────────
echo [4/5] Setting up PostgreSQL database...
echo       You will be prompted for the postgres superuser password.
psql -U postgres -c "CREATE USER aethel WITH PASSWORD 'aethel';" 2>nul
psql -U postgres -c "CREATE DATABASE aethel OWNER aethel;" 2>nul
psql -U postgres -d aethel -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>nul
echo       Database ready (ignore 'already exists' warnings)

REM ── initialise DB tables ────────────────────────────────────
echo [5/5] Initialising database tables...
python -c "import asyncio; from aethel.db.session import init_db; asyncio.run(init_db())"
if errorlevel 1 (
    echo WARNING: DB init failed. Edit .env DATABASE_URL and re-run this script.
) else (
    echo       Tables created OK
)

echo.
echo  ========================================
echo   Setup complete!
echo.
echo   Next steps:
echo   1. Open .env and set your API keys:
echo      - AETHEL_DEEPSEEK_API_KEY
echo      - AETHEL_GEMINI_API_KEY
echo      - AETHEL_TELEGRAM_BOT_TOKEN (optional)
echo   2. Make sure MetaTrader 5 is running
echo   3. Double-click start_server.bat
echo  ========================================
echo.
pause
