import concurrent.futures
import time

import matplotlib.pyplot as plt
import numpy as np

# Rango de latencias: 0 a 999 ms, un bin por ms
NUM_BINS = 1000


# Debe estar al nivel del modulo para que ProcessPoolExecutor pueda serializarla
def calcular_histograma_local(chunk):
    # Convierte valores a enteros y cuenta cuantos caen en cada bin
    indices = np.clip(chunk.astype(np.int32), 0, NUM_BINS - 1)
    return np.bincount(indices, minlength=NUM_BINS)


def _calcular_p95(hist, n_total):
    # Recorre el histograma acumulado hasta alcanzar el 95% de los datos
    umbral    = 0.95 * n_total
    acumulado = 0
    for ms, cantidad in enumerate(hist):
        acumulado += cantidad
        if acumulado >= umbral:
            return ms
    return NUM_BINS - 1


def _generar_latencias(n_lecturas, seed=42):
    # Mezcla realista: respuestas rapidas (normal) + cola lenta (exponencial)
    rng     = np.random.default_rng(seed)
    rapidas = rng.normal(loc=50, scale=15, size=int(n_lecturas * 0.85))
    lentas  = rng.exponential(scale=200,   size=int(n_lecturas * 0.15))
    datos   = np.clip(np.concatenate([rapidas, lentas]), 0, NUM_BINS - 1)
    rng.shuffle(datos)
    return datos


def run_latency_secuencial(latencias):
    # Construye el histograma completo en una sola pasada secuencial
    t0 = time.perf_counter()

    indices     = np.clip(latencias.astype(np.int32), 0, NUM_BINS - 1)
    hist_global = np.bincount(indices, minlength=NUM_BINS)
    n_total     = int(hist_global.sum())

    p95_ms      = _calcular_p95(hist_global, n_total)
    n_anomalias = int(hist_global[p95_ms:].sum())

    elapsed = time.perf_counter() - t0

    print("[ SECUENCIAL ]")
    print(f"  Percentil 95 : {p95_ms} ms")
    print(f"  Anomalias    : {n_anomalias:,}  ({n_anomalias / n_total * 100:.1f}%)")
    print(f"  Tiempo       : {elapsed:.3f} s")

    return hist_global, p95_ms, n_anomalias, elapsed


def run_latency_paralelo(latencias, num_workers=4):
    # Variables locales por worker  : hist_local  (array de NUM_BINS enteros)
    # Variables globales post-reduce: hist_global, p95_ms, n_anomalias

    # Particionamiento uniforme: costo por dato homogeneo => balance perfecto
    chunks = np.array_split(latencias, num_workers)

    t0 = time.perf_counter()

    # Fase paralela: cada worker construye su histograma local de forma independiente
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        hist_locales = list(executor.map(calcular_histograma_local, chunks))

    # Reduccion: suma bin a bin (asociativa y conmutativa, sin condicion de carrera)
    hist_global = np.sum(hist_locales, axis=0)
    n_total     = int(hist_global.sum())

    # Calculos globales: solo posibles despues de que la reduccion este completa
    p95_ms      = _calcular_p95(hist_global, n_total)
    n_anomalias = int(hist_global[p95_ms:].sum())

    elapsed = time.perf_counter() - t0

    print(f"[ PARALELO - {num_workers} workers ]")
    print(f"  Percentil 95 : {p95_ms} ms")
    print(f"  Anomalias    : {n_anomalias:,}  ({n_anomalias / n_total * 100:.1f}%)")
    print(f"  Tiempo       : {elapsed:.3f} s")

    return hist_global, hist_locales, p95_ms, n_anomalias, elapsed


def run_latency_analysis(n_lecturas=5_000_000, num_workers=4, seed=42):

    print(f"Lecturas totales: {n_lecturas:,}\n")
    latencias = _generar_latencias(n_lecturas, seed)

    hist_seq, p95_seq, anom_seq, t_seq = run_latency_secuencial(latencias)
    print()
    hist_par, hist_locales, p95_par, anom_par, t_par = run_latency_paralelo(latencias, num_workers)

    speedup = t_seq / t_par
    print(f"\nSpeedup: {speedup:.2f}x")

    # Verificacion: ambas versiones deben producir el mismo P95
    assert p95_seq == p95_par, "Error: P95 secuencial y paralelo no coinciden"
    print("Verificacion OK: P95 secuencial == P95 paralelo\n")

    zoom    = 500
    x       = np.arange(NUM_BINS)
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histograma global con linea del P95
    axes[0].bar(x[:zoom], hist_par[:zoom], width=1, color='#2c3e50', alpha=0.85)
    axes[0].axvline(p95_par, color='#e74c3c', linewidth=2, label=f'P95 = {p95_par} ms')
    axes[0].set_title('Distribución global de latencias')
    axes[0].set_xlabel('Latencia [ms]')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Histogramas parciales superpuestos por worker
    for i, hist in enumerate(hist_locales):
        axes[1].plot(x[:zoom], hist[:zoom], alpha=0.75,
                     label=f'Worker {i + 1}', color=colores[i])
    axes[1].axvline(p95_par, color='black', linewidth=1.5,
                    linestyle='--', label=f'P95 = {p95_par} ms')
    axes[1].set_title('Histogramas parciales por worker')
    axes[1].set_xlabel('Latencia [ms]')
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Comparacion de tiempos secuencial vs paralelo
    axes[2].bar(['Secuencial', f'Paralelo\n({num_workers} workers)'],
                [t_seq, t_par],
                color=['#c0392b', '#27ae60'], edgecolor='#1f2937', width=0.5)
    axes[2].set_title(f'Tiempo de ejecución (speedup: {speedup:.2f}x)')
    axes[2].set_ylabel('Tiempo [s]')
    axes[2].grid(True, linestyle='--', alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()

    return {
        'p95_ms':       p95_par,
        'n_anomalias':  anom_par,
        'n_total':      n_lecturas,
        'hist_global':  hist_par,
        'hist_locales': hist_locales,
        't_secuencial': t_seq,
        't_paralelo':   t_par,
        'speedup':      speedup,
    }


if __name__ == '__main__':
    run_latency_analysis(num_workers=4)
