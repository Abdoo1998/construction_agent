@echo off
echo PDF RAG System Setup Script for Windows
echo ==================================

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    exit /b 1
)

REM Check if virtual environment exists, create if it doesn't
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Error: Failed to create virtual environment.
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment and install requirements
echo Installing requirements...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install requirements.
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo To run the RAG API server:
echo   call venv\Scripts\activate.bat
echo   python main.py
echo.
echo To deactivate the virtual environment when done:
echo   deactivate 