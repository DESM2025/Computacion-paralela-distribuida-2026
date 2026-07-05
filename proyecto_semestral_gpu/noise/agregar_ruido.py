import cv2
import numpy as np
import os
import glob

def listar_videos(carpeta):
    exts = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    videos = []
    for ext in exts:
        videos += glob.glob(os.path.join(carpeta, ext))
    return sorted(videos)

def elegir_video(videos):
    print("\nVideos disponibles:")
    for i, v in enumerate(videos):
        print(f"  [{i+1}] {os.path.basename(v)}")
    while True:
        try:
            op = int(input("Elige un numero: ")) - 1
            if 0 <= op < len(videos):
                return videos[op]
        except ValueError:
            pass
        print("Entrada invalida, intenta de nuevo.")

def elegir_sigma():
    print("\nNivel de ruido (sigma):")
    print("  [1] Bajo   (sigma=10) — ruido sutil")
    print("  [2] Medio  (sigma=25) — igual que en el proyecto")
    print("  [3] Alto   (sigma=40) — ruido muy visible")
    print("  [4] Custom")
    while True:
        try:
            op = int(input("Elige un numero: "))
            if op == 1: return 10.0
            if op == 2: return 25.0
            if op == 3: return 40.0
            if op == 4:
                s = float(input("Ingresa sigma (1-100): "))
                if 1 <= s <= 100: return s
        except ValueError:
            pass
        print("Entrada invalida, intenta de nuevo.")

# ── Main ──────────────────────────────────────────────────────────────────────
carpeta = os.path.dirname(os.path.abspath(__file__))
videos = listar_videos(carpeta)

if not videos:
    print("No se encontraron videos en la carpeta. Agrega un video .mp4/.avi/.mov/.mkv aqui.")
    exit(1)

video_path = elegir_video(videos)
sigma = elegir_sigma()

nombre_base = os.path.splitext(os.path.basename(video_path))[0]
salida_path = os.path.join(carpeta, f"{nombre_base}_ruido_sigma{int(sigma)}.mp4")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: no se pudo abrir {video_path}")
    exit(1)

ancho  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo: {ancho}x{alto}  FPS: {fps:.2f}  Frames: {total}")
print(f"Sigma: {sigma}")
print(f"Salida: {salida_path}")
print("Procesando...")

writer = cv2.VideoWriter(salida_path,
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (ancho, alto))
if not writer.isOpened():
    print("Error: no se pudo crear el video de salida.")
    exit(1)

np.random.seed(42)  # semilla fija para reproducibilidad
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"  Frame {frame_idx}/{total}...")

    # Agregar ruido Gaussiano
    ruido = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    frame_ruidoso = np.clip(frame.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
    writer.write(frame_ruidoso)

cap.release()
writer.release()
print(f"\nListo — {frame_idx} frames procesados.")
print(f"Video con ruido guardado en: {salida_path}")
