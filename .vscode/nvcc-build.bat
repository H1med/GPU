@echo off
setlocal

if "%~1"=="" (
  echo Usage: nvcc-build.bat ^<source.cu^> [output.exe]
  exit /b 1
)

set "SRC=%~f1"
if "%~2"=="" (
  set "OUT=%~dpn1.exe"
) else (
  set "OUT=%~f2"
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo [ERROR] vswhere.exe not found.
  echo         Install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload.
  exit /b 1
)

set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"

if not defined VSDIR (
  echo [ERROR] MSVC C++ toolchain not found.
  echo         Install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload.
  exit /b 1
)

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo [ERROR] Failed to initialize the MSVC environment.
  exit /b 1
)

if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
  echo [ERROR] nvcc.exe not found under "%CUDA_PATH%".
  exit /b 1
)

"%CUDA_PATH%\bin\nvcc.exe" -arch=native -O2 -lineinfo "%SRC%" -o "%OUT%"
if errorlevel 1 exit /b 1

echo [OK] Built "%OUT%"
