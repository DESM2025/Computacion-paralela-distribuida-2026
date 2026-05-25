import gzip
import csv
import time
import asyncio
import random
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime, timedelta

# Configuracion
N_LINES    = 3_000_000   # lineas del log sintetico (~300MB descomprimido, ~60MB .gz)
CHUNK_SIZE = 100_000     # lineas por chunk
N_WORKERS  = 4
LOG_FILE   = "logs_sintetico.gz"
RESULTS_DIR = "anexos_benchmark"

ENDPOINTS = ["/api/login", "/api/data", "/api/logout", "/api/upload", "/api/status"]
CODES     = [200, 200, 200, 400, 500]

# ── Generacion de datos ──────────────────────────────────────────────────────

def generar_logs(filename, n_lines):
    print(f"Generando {n_lines:,} lineas en {filename}...")
    base = datetime(2024, 1, 1)
    with gzip.open(filename, "wt", encoding="utf-8") as f:
        for i in range(n_lines):
            ts  = base + timedelta(seconds=i)
            uid = f"user_{random.randint(1, 1000)}"
            ep  = random.choice(ENDPOINTS)
            lat = round(random.uniform(10, 500), 2)
            cod = random.choice(CODES)
            f.write(f"{uid},{ts.strftime('%Y-%m-%d %H:%M:%S')},{ep},{lat},{cod}\n")
    print(f"Archivo generado: {os.path.getsize(filename)/1e6:.1f} MB\n")

# ── Parseo y agregacion (CPU-bound) ──────────────────────────────────────────

def parsear_chunk(lines):
    # Parsea lineas y preagrega metricas por (usuario, endpoint, ventana 15min)
    metricas = defaultdict(lambda: {"count": 0, "latencia_total": 0.0, "errores": 0})
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            uid, ts_str, ep, lat, cod = line.split(",")
            ts  = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            ventana = ts.replace(minute=(ts.minute // 15) * 15, second=0)
            clave   = (uid, ep, ventana.strftime("%Y-%m-%d %H:%M"))
            metricas[clave]["count"]          += 1
            metricas[clave]["latencia_total"] += float(lat)
            metricas[clave]["errores"]        += 1 if cod != "200" else 0
        except Exception:
            continue
    return dict(metricas)

def consolidar(resultados_parciales):
    # Agrega resultados de todos los chunks sin condiciones de carrera
    final = defaultdict(lambda: {"count": 0, "latencia_total": 0.0, "errores": 0})
    for parcial in resultados_parciales:
        for clave, vals in parcial.items():
            final[clave]["count"]          += vals["count"]
            final[clave]["latencia_total"] += vals["latencia_total"]
            final[clave]["errores"]        += vals["errores"]
    return final

def leer_chunks(filename):
    # Lee el archivo .gz y devuelve lista de chunks
    chunks = []
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= CHUNK_SIZE:
                chunks.append(chunk)
                chunk = []
        if chunk:
            chunks.append(chunk)
    return chunks

# ── Configuraciones de benchmark ─────────────────────────────────────────────

def run_secuencial(chunks):
    resultados = [parsear_chunk(c) for c in chunks]
    return consolidar(resultados)

def run_threading(chunks):
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        resultados = list(ex.map(parsear_chunk, chunks))
    return consolidar(resultados)

def run_multiprocessing(chunks):
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        resultados = list(ex.map(parsear_chunk, chunks))
    return consolidar(resultados)

async def run_hibrido(chunks):
    # asyncio coordina el envio de chunks al pool de procesos
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        tareas = [loop.run_in_executor(ex, parsear_chunk, c) for c in chunks]
        resultados = await asyncio.gather(*tareas)
    return consolidar(resultados)

# ── Benchmark ────────────────────────────────────────────────────────────────

def medir(nombre, fn, chunks):
    # Warm-up con primer chunk
    parsear_chunk(chunks[0])
    t0 = time.perf_counter()
    fn(chunks)
    return time.perf_counter() - t0

def medir_hibrido(chunks):
    parsear_chunk(chunks[0])
    t0 = time.perf_counter()
    asyncio.run(run_hibrido(chunks))
    return time.perf_counter() - t0

def guardar_resultados(resultados):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_seq = resultados["Secuencial"]

    # TXT
    with open(f"{RESULTS_DIR}/resultados2.txt", "w") as f:
        f.write("Benchmark Pipeline Concurrente - Ejercicio 2\n")
        f.write(f"Lineas: {N_LINES:,} | Chunk: {CHUNK_SIZE:,} | Workers: {N_WORKERS}\n\n")
        f.write(f"{'Config':<30} {'Tiempo(s)':<12} {'Speedup':<10} {'Eficiencia'}\n")
        f.write("-" * 60 + "\n")
        for nombre, t in resultados.items():
            sp = t_seq / t
            ef = sp / N_WORKERS if nombre != "Secuencial" else 1.0
            f.write(f"{nombre:<30} {t:<12.4f} {sp:<10.2f} {ef:.2f}\n")

    # CSV
    with open(f"{RESULTS_DIR}/resultados2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "tiempo", "speedup", "eficiencia"])
        for nombre, t in resultados.items():
            sp = t_seq / t
            ef = sp / N_WORKERS if nombre != "Secuencial" else 1.0
            writer.writerow([nombre, round(t, 4), round(sp, 4), round(ef, 4)])

    print(f"\nResultados guardados en {RESULTS_DIR}/resultados2.txt y resultados2.csv")

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Generar archivo si no existe
    if not os.path.exists(LOG_FILE):
        generar_logs(LOG_FILE, N_LINES)

    print("Leyendo y particionando archivo...")
    chunks = leer_chunks(LOG_FILE)
    print(f"Total chunks: {len(chunks)}\n")

    configs = {
        "Secuencial":              lambda c: run_secuencial(c),
        "Threading":               lambda c: run_threading(c),
        "Multiprocessing":         lambda c: run_multiprocessing(c),
        "Hibrido asyncio+multiproc": lambda c: None,  # se mide aparte
    }

    resultados = {}

    for nombre, fn in configs.items():
        if nombre == "Hibrido asyncio+multiproc":
            print(f"Corriendo {nombre}...")
            t = medir_hibrido(chunks)
        else:
            print(f"Corriendo {nombre}...")
            t = medir(nombre, fn, chunks)
        print(f"Tiempo: {t:.4f} s\n")
        resultados[nombre] = t

    t_seq = resultados["Secuencial"]
    print(f"{'Config':<30} {'Tiempo(s)':<12} {'Speedup':<10} {'Eficiencia'}")
    print("-" * 60)
    for nombre, t in resultados.items():
        sp = t_seq / t
        ef = sp / N_WORKERS if nombre != "Secuencial" else 1.0
        print(f"{nombre:<30} {t:<12.4f} {sp:<10.2f} {ef:.2f}")

    guardar_resultados(resultados)
