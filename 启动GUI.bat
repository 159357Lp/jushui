@echo off
rem ============================================================================
rem  LM Loader standalone launcher
rem
rem  Self-contained on purpose: it must NOT depend on whatever happens to be
rem  first on PATH. The previous version called bare `pythonw`, which resolved
rem  to a Python without PySide6 - and since pythonw has no console, the GUI
rem  simply never appeared with no explanation.
rem
rem  Kept ASCII-only (no chcp, no BOM): this machine's console code page is
rem  936, so any non-ASCII text in a .bat gets mangled and can break parsing.
rem  Messages are English for that reason; the GUI itself is localised.
rem
rem  Interpreter resolution order:
rem    1. LMLOADER_PYTHON env var (explicit override)
rem    2. pythonw.exe next to this script        (portable / venv install)
rem    3. the standard per-user install location
rem  PySide6 is verified before launching, so a bad setup reports itself
rem  instead of vanishing.
rem ============================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PYW="

if defined LMLOADER_PYTHON (
    if exist "!LMLOADER_PYTHON!" set "PYW=!LMLOADER_PYTHON!"
)

if not defined PYW (
    if exist "%~dp0.venv\Scripts\pythonw.exe" set "PYW=%~dp0.venv\Scripts\pythonw.exe"
)
if not defined PYW (
    if exist "%~dp0pythonw.exe" set "PYW=%~dp0pythonw.exe"
)

if not defined PYW (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe" (
        set "PYW=%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe"
    )
)

if not defined PYW (
    echo.
    echo   [LM Loader] No usable Python interpreter found.
    echo.
    echo   Tried:
    echo     - env var LMLOADER_PYTHON
    echo     - %~dp0.venv\Scripts\pythonw.exe
    echo     - %LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe
    echo.
    echo   Install Python 3.10+ then run:
    echo     pip install -e "%~dp0."
    echo.
    pause
    exit /b 1
)

rem Probe with the console python.exe: errors are visible there, unlike
rem pythonw which would swallow them.
set "PYEXE=%PYW:pythonw.exe=python.exe%"
if not exist "%PYEXE%" set "PYEXE=%PYW%"

"%PYEXE%" -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [LM Loader] PySide6 is missing from:
    echo     %PYW%
    echo.
    echo   Install the dependencies into that same interpreter:
    echo     "%PYEXE%" -m pip install -e "%~dp0."
    echo.
    pause
    exit /b 1
)

start "" /b "%PYW%" -m lmloader.gui.main
exit /b 0
