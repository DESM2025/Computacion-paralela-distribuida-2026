@echo off
echo ========================================
echo  Compilando NL-Means Video
echo ========================================

set OPENCV_INC=C:\vcpkg\installed\x64-windows\include\opencv4
set OPENCV_LIB=C:\vcpkg\installed\x64-windows\lib
set OPENCV_LIBS="%OPENCV_LIB%\opencv_core4.lib" "%OPENCV_LIB%\opencv_videoio4.lib" "%OPENCV_LIB%\opencv_imgproc4.lib" "%OPENCV_LIB%\opencv_imgcodecs4.lib" "%OPENCV_LIB%\opencv_highgui4.lib"

echo.
echo [1/3] Compilando CPU OpenMP...
cd /d "%~dp0cpu"
cl /O2 /EHsc /openmp /std:c++17 nlmeans_video_cpu_omp.cpp /Fe:nlmeans_video_cpu_omp.exe /I"%OPENCV_INC%" /link /LIBPATH:"%OPENCV_LIB%" opencv_core4.lib opencv_videoio4.lib opencv_imgproc4.lib opencv_imgcodecs4.lib opencv_highgui4.lib
if %errorlevel% neq 0 ( echo ERROR compilando CPU && pause && exit /b 1 )
echo OK

echo.
echo [2/3] Compilando GPU CUDA...
cd /d "%~dp0gpu"
nvcc -arch=sm_89 nlmeans_video_gpu.cu -o nlmeans_video_gpu.exe -I"%OPENCV_INC%" %OPENCV_LIBS%
if %errorlevel% neq 0 ( echo ERROR compilando GPU && pause && exit /b 1 )
echo OK

echo.
echo [3/3] Compilando GPU Shared Memory...
nvcc -arch=sm_89 nlmeans_video_gpu_shared.cu -o nlmeans_video_gpu_shared.exe -I"%OPENCV_INC%" %OPENCV_LIBS%
if %errorlevel% neq 0 ( echo ERROR compilando GPU Shared && pause && exit /b 1 )
echo OK

echo.
echo ========================================
echo  Compilacion completada exitosamente
echo ========================================
pause
