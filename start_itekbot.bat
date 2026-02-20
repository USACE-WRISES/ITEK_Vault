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
pip install --quiet --upgrade fastapi uvicorn langchain-ollama langchain-community faiss-cpu pydantic

echo Starting ITEKbot...
python query_server.py

echo.
echo Press any key to close...
pause >nul