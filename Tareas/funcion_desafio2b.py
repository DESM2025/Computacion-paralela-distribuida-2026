import concurrent.futures
import time

import matplotlib.pyplot as plt
import numpy as np

# Desviacion estandar del kernel gaussiano
SIGMA = 2.5


def _gaussian_kernel(k, sigma):
    # Genera kernel gaussiano de tamaño 2k+1 normalizado
    j = np.arange(-k, k + 1, dtype=np.float64)
    w = np.exp(-j**2 / (2 * sigma**2))
    return w / w.sum()


# Funcion ejecutada por cada worker (nivel de modulo para evitar el GIL)
def apply_filter_local(args):
    # Recibe chunk con halos ya incluidos y devuelve solo los resultados del centro
    chunk, kernel, k = args
    n_out = len(chunk) - 2 * k
    window_size = 2 * k + 1
    result = np.empty(n_out, dtype=np.float64)
    for i in range(n_out):
        result[i] = np.dot(kernel, chunk[i : i + window_size])
    return result


def _filtro_secuencial(signal_pad, kernel, k, n):
    # Recorre todos los n puntos aplicando el kernel centrado en cada uno
    window_size = 2 * k + 1
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = np.dot(kernel, signal_pad[i : i + window_size])
    return result


def run_iot_temperature_filter(n_readings=200_000, k=15, num_workers=4, seed=42):

    # Generacion de datos sinteticos: temperatura IoT con variacion estacional y ruido
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * np.pi, n_readings)
    signal = 20 + 5 * np.sin(t) + 2 * np.sin(5 * t) + rng.normal(0, 2.0, n_readings)

    kernel = _gaussian_kernel(k, SIGMA)

    # Padding con valores extremos para manejar las fronteras del arreglo
    signal_pad = np.pad(signal, k, mode='edge')

    # Version secuencial
    t0 = time.perf_counter()
    result_seq = _filtro_secuencial(signal_pad, kernel, k, n_readings)
    t_seq = time.perf_counter() - t0

    print("[ SECUENCIAL ]")
    print(f"  Tiempo : {t_seq:.3f} s")

    # Version paralela con halos
    # Cada worker recibe signal_pad[start : end + 2k], que incluye:
    #   - k elementos ANTES del bloque util  -> halo izquierdo
    #   - el bloque util propio del worker
    #   - k elementos DESPUES del bloque util -> halo derecho
    indices   = np.array_split(np.arange(n_readings), num_workers)
    args_list = []
    for idx in indices:
        start = int(idx[0])
        end   = int(idx[-1]) + 1
        chunk_con_halos = signal_pad[start : end + 2 * k]
        args_list.append((chunk_con_halos, kernel, k))

    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        parciales = list(executor.map(apply_filter_local, args_list))
    result_par = np.concatenate(parciales)
    t_par = time.perf_counter() - t0

    speedup   = t_seq / t_par
    max_error = float(np.max(np.abs(result_seq - result_par)))

    print(f"[ PARALELO - {num_workers} workers ]")
    print(f"  Tiempo      : {t_par:.3f} s")
    print(f"  Speedup     : {speedup:.2f}x")
    print(f"  Error maximo secuencial vs paralelo: {max_error:.2e}")

    # Graficos
    zoom     = 2000   # primeras 2000 muestras para grafico 1
    x        = np.arange(n_readings)
    colores  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Grafico 1: señal original vs filtrada secuencial (zoom primeras 2000 muestras)
    axes[0].plot(x[:zoom], signal[:zoom],
                 color='#aec6e8', linewidth=0.8, alpha=0.7, label='Original (ruidosa)')
    axes[0].plot(x[:zoom], result_seq[:zoom],
                 color='#1f77b4', linewidth=1.8, label=f'Filtrada (k={k})')
    axes[0].set_title('Señal original vs filtro gaussiano')
    axes[0].set_xlabel('Muestra')
    axes[0].set_ylabel('Temperatura [C]')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Grafico 2: resultado de cada worker coloreado en el arreglo completo
    # Muestra los 4 segmentos del arreglo filtrado con un color distinto por worker
    offset = 0
    for i, parcial in enumerate(parciales):
        n_chunk = len(parcial)
        axes[1].plot(x[offset : offset + n_chunk], parcial,
                     color=colores[i], linewidth=0.6,
                     label=f'Worker {i + 1}', alpha=0.9)
        offset += n_chunk
    axes[1].set_title('Segmento de cada worker (señal filtrada completa)')
    axes[1].set_xlabel('Muestra')
    axes[1].set_ylabel('Temperatura filtrada [C]')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Grafico 3: comparacion de tiempos
    axes[2].bar(['Secuencial', f'Paralelo\n({num_workers} workers)'],
                [t_seq, t_par],
                color=['#27ae60', '#c0392b'], edgecolor='#1f2937', width=0.5)
    axes[2].set_title(f'Tiempo de ejecución (speedup: {speedup:.2f}x)')
    axes[2].set_ylabel('Tiempo [s]')
    axes[2].grid(True, linestyle='--', alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()

    return {
        'result_seq':   result_seq,
        'result_par':   result_par,
        'max_error':    max_error,
        't_secuencial': t_seq,
        't_paralelo':   t_par,
        'speedup':      speedup,
    }


if __name__ == '__main__':
    run_iot_temperature_filter(num_workers=4)
