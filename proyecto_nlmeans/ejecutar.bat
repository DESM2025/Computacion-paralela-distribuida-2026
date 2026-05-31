@echo off
echo ========================================
echo  Ejecutando NL-Means Imagenes
echo ========================================

set IMAGEN=%1

echo.
echo [1/4] CPU Secuencial...
cd /d "%~dp0cpu"
if "%IMAGEN%"=="" ( nlmeans_cpu.exe ) else ( nlmeans_cpu.exe %IMAGEN% )
if %errorlevel% neq 0 ( echo ERROR ejecutando CPU Secuencial && pause && exit /b 1 )

echo.
echo [2/4] CPU OpenMP...
if "%IMAGEN%"=="" ( nlmeans_cpu_omp.exe ) else ( nlmeans_cpu_omp.exe %IMAGEN% )
if %errorlevel% neq 0 ( echo ERROR ejecutando CPU OpenMP && pause && exit /b 1 )

echo.
echo [3/4] GPU CUDA...
cd /d "%~dp0gpu"
if "%IMAGEN%"=="" ( nlmeans_gpu.exe ) else ( nlmeans_gpu.exe %IMAGEN% )
if %errorlevel% neq 0 ( echo ERROR ejecutando GPU && pause && exit /b 1 )

echo.
echo [4/4] GPU Shared Memory...
if "%IMAGEN%"=="" ( nlmeans_gpu_shared.exe ) else ( nlmeans_gpu_shared.exe %IMAGEN% )
if %errorlevel% neq 0 ( echo ERROR ejecutando GPU Shared && pause && exit /b 1 )

echo.
echo ========================================
echo  Ejecucion completada
echo ========================================
pause
