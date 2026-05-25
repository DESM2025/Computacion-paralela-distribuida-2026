import numpy as np
import time
import csv
import os
import sys
from multiprocessing import shared_memory, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

# Necesario para compatibilidad con %run de Jupyter en Windows
if not hasattr(sys.modules['__main__'], '__spec__'):
    sys.modules['__main__'].__spec__ = None

def process_block(start_idx, end_idx, shm_name, shape, dtype, k=10):
    """
    Worker independiente que lee desde la memoria compartida (zero-copy)
    y calcula el Top-K para su bloque asignado.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    X = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    query_block = X[start_idx:end_idx]

    # Producto punto vectorial (BLAS rutinas en C subyacentes)
    sim_matrix = np.dot(query_block, X.T)

    # Filtrado In-Place: argpartition para extraer el Top-K en O(N)
    top_k_indices = np.argpartition(-sim_matrix, k, axis=1)[:, :k]
    top_k_values = np.take_along_axis(sim_matrix, top_k_indices, axis=1)

    # Ordenamiento final local de los k elementos (O(k log k))
    sort_order = np.argsort(-top_k_values, axis=1)
    final_indices = np.take_along_axis(top_k_indices, sort_order, axis=1)
    final_values = np.take_along_axis(top_k_values, sort_order, axis=1)

    existing_shm.close()
    return start_idx, final_indices, final_values

def main():
    N_VECTORS = 20000
    DIM = 128
    K = 10
    BLOCK_SIZE = 2000
    RESULTS_DIR = "anexos_benchmark"

    print("Generando dataset...")
    np.random.seed(42)
    datos_crudos = np.random.rand(N_VECTORS, DIM).astype(np.float32)
    normas = np.sum(np.abs(datos_crudos), axis=1, keepdims=True)
    X_original = datos_crudos / normas

    bytes_size = X_original.nbytes
    shm = shared_memory.SharedMemory(create=True, size=bytes_size)
    X_shared = np.ndarray(X_original.shape, dtype=X_original.dtype, buffer=shm.buf)
    X_shared[:] = X_original[:]

    resultados_indices = np.zeros((N_VECTORS, K), dtype=np.int32)
    resultados_valores = np.zeros((N_VECTORS, K), dtype=np.float32)
    chunks = [(i, min(i + BLOCK_SIZE, N_VECTORS)) for i in range(0, N_VECTORS, BLOCK_SIZE)]

    max_workers = cpu_count()
    test_cores = [1, 2, 3, 4]
    if max_workers > 4:
        test_cores.append(8 if max_workers >= 8 else max_workers)
    test_cores = sorted(list(set(test_cores)))

    t_secuencial = None
    filas_csv = []

    print(f"\nMatriz compartida en RAM: {bytes_size / (1024**2):.2f} MB")
    print("-" * 70)
    print(f"{'Workers':<10} | {'Tiempo (s)':<12} | {'Speedup':<10} | {'Eficiencia':<10}")
    print("-" * 70)

    try:
        for w in test_cores:
            start_time = time.perf_counter()

            with ProcessPoolExecutor(max_workers=w) as executor:
                futures = [
                    executor.submit(process_block, start, end, shm.name, X_shared.shape, X_shared.dtype, K)
                    for start, end in chunks
                ]
                for future in as_completed(futures):
                    start_idx, ind, val = future.result()
                    end_idx = start_idx + ind.shape[0]
                    resultados_indices[start_idx:end_idx] = ind
                    resultados_valores[start_idx:end_idx] = val

            t_exec = time.perf_counter() - start_time

            if w == 1:
                t_secuencial = t_exec

            sp = t_secuencial / t_exec
            ep = sp / w

            print(f"{w:<10} | {t_exec:<12.4f} | {sp:<10.4f} | {ep:<10.4f}")
            filas_csv.append((w, round(t_exec, 4), round(sp, 4), round(ep, 4)))

    finally:
        shm.close()
        shm.unlink()

    # Guardar resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/resultados3.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workers", "tiempo", "speedup", "eficiencia"])
        writer.writerows(filas_csv)
    print(f"\nResultados guardados en {RESULTS_DIR}/resultados3.csv")

if __name__ == '__main__':
    main()
