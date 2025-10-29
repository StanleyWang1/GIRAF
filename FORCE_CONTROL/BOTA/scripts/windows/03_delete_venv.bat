@echo off
REM Deletes the virtual environment folder named "venv"

if not exist "venv" (
    echo No venv directory found.
    exit /b 0
)

echo Deleting venv directory...
rmdir /s /q "venv"

if exist "venv" (
    echo Failed to delete venv.
    exit /b 1
) else (
    echo venv deleted