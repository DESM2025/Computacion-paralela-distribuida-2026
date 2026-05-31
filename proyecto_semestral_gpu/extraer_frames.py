import cv2
import os
import random
import glob

def listar_videos(carpeta):
    videos = glob.glob(os.path.join(carpeta, "*.mp4")) + glob.glob(os.path.join(carpeta, "*.avi"))
    return sorted(videos)

def elegir_video(carpeta, mensaje):
    videos = listar_videos(carpeta)
    if not videos:
        print(f"No se encontraron videos en {carpeta}")
        exit(1)
    print(f"\n{mensaje}")
    for i, v in enumerate(videos):
        print(f"  [{i+1}] {os.path.basename(v)}")
    while True:
        try:
            opcion = int(input("Elige un numero: ")) - 1
            if 0 <= opcion < len(videos):
                return videos[opcion]
        except ValueError:
            pass
        print("Opcion invalida, intenta de nuevo.")

# Elegir videos
video_original = elegir_video("videos", "Video ORIGINAL:")
video_limpio   = elegir_video("resultados/videos", "Video LIMPIO a comparar:")

cap_orig   = cv2.VideoCapture(video_original)
cap_limpio = cv2.VideoCapture(video_limpio)

if not cap_orig.isOpened():
    print(f"Error: no se pudo abrir {video_original}")
    exit(1)
if not cap_limpio.isOpened():
    print(f"Error: no se pudo abrir {video_limpio}")
    exit(1)

total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\nTotal frames: {total_frames}")

# Elegir 8 frames al azar evitando inicio y final
frames_elegidos = sorted(random.sample(range(10, total_frames - 10), 8))
print(f"Frames elegidos: {frames_elegidos}")

os.makedirs("resultados/comparacion", exist_ok=True)

for idx in frames_elegidos:
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret1, frame_orig = cap_orig.read()

    cap_limpio.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret2, frame_limpio = cap_limpio.read()

    if not ret1 or not ret2:
        print(f"No se pudo leer frame {idx}, saltando...")
        continue

    cv2.imwrite(f"resultados/comparacion/frame{idx:04d}_original.jpg", frame_orig)
    cv2.imwrite(f"resultados/comparacion/frame{idx:04d}_limpio.jpg", frame_limpio)

    comparacion = cv2.hconcat([frame_orig, frame_limpio])
    cv2.imwrite(f"resultados/comparacion/frame{idx:04d}_comparacion.jpg", comparacion)

    print(f"Frame {idx} guardado")

cap_orig.release()
cap_limpio.release()
print(f"\nListo — imagenes en resultados/comparacion/")
