@echo off
setlocal enabledelayedexpansion

echo === Python Virtual Environment Setup ===

set "HAS_312="
set "HAS_313="

:: Detect installed Python versions using py launcher
for /f "tokens=*" %%i in ('py -0p') do (
    echo %%i | findstr /C:"3.13" >nul && set HAS_313=1
    echo %%i | findstr /C:"3.12" >nul && set HAS_312=1
)

:: Prompt user for selection if both are available
if defined HAS_312 if defined HAS_313 (
    echo Multiple Python versions detected.
    echo [1] Python 3.13
    echo [2] Python 3.12
    set /p CHOICE="Select Python version (1 or 2): "

    :: Trim whitespace from input
    for /f "tokens=* delims= " %%a in ("!CHOICE!") do set CHOICE=%%a

    if "!CHOICE!"=="1" (
        set PY_CMD=py -3.13
    ) else if "!CHOICE!"=="2" (
        set PY_CMD=py -3.12
    ) else (
        echo !CHOICE! is not a valid option.
        echo Invalid choice. Exiting...
        exit /b 1
    )
) else if defined HAS_313 (
    echo Only Python 3.13 found. Using that.
    set PY_CMD=py -3.13
) else if defined HAS_312 (
    echo Only Python 3.12 found. Using that.
    set PY_CMD=py -3.12
) else (
    echo Neither Python 3.12 nor 3.13 found.
    exit /b 1
)

:: Create virtual environment
set ENV_NAME=venv
echo Creating virtual environment with: %PY_CMD%
%PY_CMD% -m venv %ENV_NAME%

if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
)

:: Activate virtual environment
echo Activating the virtual environment...
call .\%ENV_NAME%\Scripts\activate

if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

:: Install required libraries
echo Installing libraries...
call pip install bota-driver

echo Libraries installed successfully.
deactivate
