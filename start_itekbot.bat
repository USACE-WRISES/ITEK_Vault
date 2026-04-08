@echo off
title ITEKbot – Starting...

echo Checking Python...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not found. Please install from https://www.python.org/downloads/
    pause
    exit /b
)

echo Installing/updating libraries if missing...
pip install -r requirements.txt

echo Starting ITEKbot...
python query_server.py

echo.
echo Press any key to close...
pause >nul