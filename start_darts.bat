@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
if exist ".venv\Scripts\python.exe" ( set PY=".venv\Scripts\python.exe" ) else ( set PY=python )

:loop
echo [%date% %time%] indul >> logs\server.log
%PY% server.py --weights runs\tips\weights\best_openvino_model --device intel:gpu --fallback-weights runs\tips\weights\best.pt --cams 0 1 2 --auto-devices --devices 1 0 2 --fast-confirm 0.75 >> logs\server.log 2>&1
set RC=%errorlevel%
if "%RC%"=="42" (
  echo [%date% %time%] a program leallitasat a weboldalrol kertek, nem inditom ujra >> logs\server.log
  exit /b 0
)
echo [%date% %time%] leallt (kod: %RC%), ujrainditas 5 mp mulva >> logs\server.log
timeout /t 5 /nobreak >nul
goto loop
