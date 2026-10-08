@echo off
setlocal
cd /d "%~dp0"
set "SOURCE=%USERPROFILE%\Downloads\bge-large-en-v1.5"
set "DEST=%~dp0models\bge-large-en-v1.5"
if not exist "%SOURCE%\model.safetensors" (
  echo Model not found in:
  echo %SOURCE%
  echo.
  echo Put your complete bge-large-en-v1.5 folder in Downloads and run this again.
  pause
  exit /b 1
)
if exist "%DEST%" rmdir /s /q "%DEST%"
xcopy "%SOURCE%" "%DEST%" /E /I /H /Y
if errorlevel 1 (
  echo Model copy failed.
  pause
  exit /b 1
)
echo.
echo BGE model copied to:
echo %DEST%
echo.
pause
