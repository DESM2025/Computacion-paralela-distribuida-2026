@echo off
echo ========================================
echo  Compilando NL-Means Imagenes
echo ========================================

echo.
echo [1/4] Compilando CPU Secuencial...
cd /d "%~dp0cpu"
cl /O2 /EHsc /openmp nlmeans_cpu.cpp /Fe:nlmeans_cpu.exe
if %errorlevel% neq 0 ( echo ERROR compilando CPU Secuencial && pause && exit /b 1 )
echo OK

echo.
echo [2/4] Compilando CPU OpenMP...
cl /O2 /EHsc /openmp nlmeans_cpu_omp.cpp /Fe:nlmeans_cpu_omp.exe
if %errorlevel% neq 0 ( echo ERROR compilando CPU OpenMP && pause && exit /b 1 )
echo OK

echo.
echo [3/4] Compilando GPU CUDA...
cd /d "%~dp0gpu"
nvcc -arch=sm_89 nlmeans_gpu.cu -o nlmeans_gpu.exe
if %errorlevel% neq 0 ( echo ERROR compilando GPU && pause && exit /b 1 )
echo OK

echo.
echo [4/4] Compilando GPU Shared Memory...
nvcc -arch=sm_89 nlmeans_gpu_shared.cu -o nlmeans_gpu_shared.exe
if %errorlevel% neq 0 ( echo ERROR compilando GPU Shared && pause && exit /b 1 )
echo OK

echo.
echo ========================================
echo  Compilacion completada exitosamente
echo ========================================
pause
