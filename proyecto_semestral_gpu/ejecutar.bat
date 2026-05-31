@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  Ejecutando NL-Means Video
echo ========================================

set PATH=%PATH%;C:\vcpkg\installed\x64-windows\bin

set "VIDEO=%~1"
if "!VIDEO!"=="" (
    set count=0
    echo.
    echo Videos disponibles:
    for %%f in (videos\*.mp4 videos\*.avi) do (
        set /a count+=1
        set "video[!count!]=%%f"
        echo   [!count!] %%f
    )
)

if "!VIDEO!"=="" goto pedir_video
goto ejecutar

:pedir_video
set "choice="
set /p choice=Elige un numero:
if "!choice!"=="" ( echo Entrada invalida, intenta de nuevo. & goto pedir_video )
for /f "delims=0123456789" %%i in ("!choice!") do ( echo Entrada invalida, intenta de nuevo. & goto pedir_video )
if !choice! LSS 1 ( echo Entrada invalida, intenta de nuevo. & goto pedir_video )
if !choice! GTR !count! ( echo Entrada invalida, intenta de nuevo. & goto pedir_video )
set idx=!choice!
call set "VIDEO=%%video[!idx!]%%"

:ejecutar
cd /d "%~dp0"
echo.
echo Video: !VIDEO!
echo.

echo [1/3] CPU OpenMP...
cpu\nlmeans_video_cpu_omp.exe !VIDEO!
if !errorlevel! neq 0 ( echo ERROR ejecutando CPU && pause && exit /b 1 )

echo.
echo [2/3] GPU CUDA...
gpu\nlmeans_video_gpu.exe !VIDEO!
if !errorlevel! neq 0 ( echo ERROR ejecutando GPU && pause && exit /b 1 )

echo.
echo [3/3] GPU Shared Memory...
gpu\nlmeans_video_gpu_shared.exe !VIDEO!
if !errorlevel! neq 0 ( echo ERROR ejecutando GPU Shared && pause && exit /b 1 )

echo.
echo ========================================
echo  Ejecucion completada
echo ========================================
pause
