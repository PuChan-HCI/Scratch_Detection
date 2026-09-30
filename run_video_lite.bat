@echo off
setlocal
cd /d "%~dp0"
set "SCRATCH_PYTHON=C:\Users\PrimaLab\anaconda3\envs\CarDamage\python.exe"
set "SCRATCH_INPUT=%~1"
if not defined SCRATCH_INPUT set "SCRATCH_INPUT=%~dp0video_one_ai.mp4"
"%SCRATCH_PYTHON%" "%~dp0detect_scratches_lite.py" --model "%~dp0scratch_lite.pt" --device auto --input "%SCRATCH_INPUT%" --output "%~dp0scratch_lite_result.mp4"
pause
