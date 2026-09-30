@echo off
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
  py -3 serve.py
  goto end
)
where python >nul 2>nul
if not errorlevel 1 (
  python serve.py
  goto end
)
if exist "%USERPROFILE%\anaconda3\envs\CarDamage\python.exe" (
  "%USERPROFILE%\anaconda3\envs\CarDamage\python.exe" serve.py
  goto end
)
echo Python 3 is required. See README.md.
:end
pause
