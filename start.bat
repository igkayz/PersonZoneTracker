@echo off
chcp 65001 >nul
title Трекер запретных зон (YOLO + SQLite)
cd /d "%~dp0"

set "PY_CMD="
if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PY_CMD=python"
    ) else (
        where py >nul 2>&1
        if %errorlevel%==0 set "PY_CMD=py"
    )
)

if "%PY_CMD%"=="" (
    echo [ОШИБКА] Python не найден! Установите Python 3.10+ с сайта python.org и отметьте "Add Python to PATH".
    pause
    exit /b 1
)

echo [1/2] Проверка библиотек (numpy, opencv-python, ultralytics)...
"%PY_CMD%" -c "import numpy, cv2, ultralytics" >nul 2>&1
if %errorlevel% neq 0 (
    echo Установка недостающих библиотек из requirements.txt...
    "%PY_CMD%" -m pip install -r requirements.txt
)

echo [2/2] ЗАПУСК СЕРВЕРА...
"%PY_CMD%" app_server.py

pause
