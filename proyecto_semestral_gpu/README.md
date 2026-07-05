# NLM-C Video Denoising — Proyecto Semestral

Implementación del filtro Non-Local Means Color (NLM-C) para eliminación de ruido en video, en tres versiones paralelas: CPU OpenMP, GPU CUDA con memoria global y GPU CUDA con memoria compartida.

---

## Requisitos

### Hardware
- GPU NVIDIA con soporte CUDA (probado en RTX 4060, sm_89)
- CPU con soporte multi-hilo (mínimo 16 hilos lógicos recomendado)

### Software
- Windows 10/11 x64
- [Visual Studio 2022](https://visualstudio.microsoft.com/) con componente **"Desarrollo para escritorio con C++"**
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) — asegurarse de que `nvcc` esté en el PATH
- [vcpkg](https://github.com/microsoft/vcpkg) con OpenCV 4.12.0:

```
git clone https://github.com/microsoft/vcpkg
cd vcpkg
bootstrap-vcpkg.bat
vcpkg install opencv4:x64-windows
```

vcpkg debe estar instalado en `C:\vcpkg`. Si lo instalaste en otra ruta, edita la variable `OPENCV_INC` y `OPENCV_LIB` dentro de `compilar.bat`.

### Python (para gráficas y extracción de frames)
- Python 3.9 o superior
- Instalar dependencias:

```
pip install pandas matplotlib opencv-python
```

---

## Videos

Los archivos de video (.mp4) no están incluidos en el repositorio debido a las limitaciones de tamaño de GitHub (archivos de 30–250 MB). Están disponibles en Google Drive:

**[Descargar videos — Google Drive](https://drive.google.com/drive/folders/16nZwVe2y4QAuavekx1UiA4Dz3QVW-ak3?usp=drive_link)**

Para reproducir los resultados, descarga los videos y colócalos en las carpetas correspondientes según la estructura indicada abajo.

---

## Estructura del proyecto

```
proyecto_semestral_gpu/
  cpu/                  → Código fuente CPU OpenMP
  gpu/                  → Código fuente GPU CUDA
  videos/               → Videos de entrada (.mp4) — ver Google Drive
  noise/                → Script para agregar ruido + videos originales — ver Google Drive
  resultados/           → Videos filtrados, métricas y gráficas (dataset principal)
  resultados t-80/      → Resultados de validación con segundo video
  compilar.bat          → Compila las 3 versiones
  ejecutar.bat          → Ejecuta las 3 versiones sobre un video
  graficar.py           → Genera gráficas de rendimiento
  extraer_frames.py     → Extrae frames comparativos de los videos
```

---

## Pasos para reproducir

### 1. Compilar

Abrir **"x64 Native Tools Command Prompt for VS 2022"** (no el terminal normal), navegar a la carpeta del proyecto y ejecutar:

```
compilar.bat
```

Esto compilará las 3 versiones y generará los ejecutables dentro de `cpu/` y `gpu/`.

### 2. Agregar ruido al video (opcional)

Si quieres generar el video ruidoso desde cero:

```
cd noise
python agregar_ruido.py
```

El video ruidoso resultante (`city_ruido_sigma25.mp4`) debe copiarse a la carpeta `videos/`.

### 3. Ejecutar el procesamiento

Desde la raíz del proyecto (con el mismo terminal de VS):

```
ejecutar.bat
```

Se mostrará un menú para elegir el video. El script ejecutará las 3 versiones en orden y guardará los videos filtrados y métricas CSV en `resultados/`.

### 4. Generar gráficas

```
python graficar.py
```

Las gráficas se guardan en `resultados/graficas/`.

### 5. Extraer frames comparativos

```
python extraer_frames.py
```

Los frames se guardan en `resultados/comparacion/`.

---

## Versiones utilizadas

| Componente       | Versión                        |
|------------------|--------------------------------|
| Sistema operativo| Windows 11 x64                 |
| GPU              | NVIDIA RTX 4060 (sm_89)        |
| CUDA Toolkit     | 12.x                           |
| Compilador C++   | MSVC x64 (VS 2022, /O2)        |
| OpenCV           | 4.12.0 (via vcpkg)             |
| Python           | 3.9+                           |
| pandas           | 2.x                            |
| matplotlib       | 3.x                            |

---

## Notas

- El tiempo GPU reportado mide exclusivamente el kernel, sin incluir transferencias de datos CPU↔GPU.
- El tiempo CPU incluye el overhead de gestión de hilos OpenMP.
- Los videos de salida se nombran automáticamente según el video de entrada para evitar sobreescrituras.
- Uso de IA generativa (Claude, Anthropic) declarado en las referencias del informe.
