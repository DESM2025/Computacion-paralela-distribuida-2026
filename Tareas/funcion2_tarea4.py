import concurrent.futures
import time

import matplotlib.pyplot as plt
import numpy as np


def calculate_partial_sum(chunk):
    return np.sum(chunk)


def run_reduction_damage_demo(size=50_000_000, num_workers=4, min_damage=0, max_damage=500):
    damage_logs = np.random.randint(min_damage, max_damage, size=size, dtype=np.int32)
    chunks = np.array_split(damage_logs, num_workers)

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        partial_sums = list(executor.map(calculate_partial_sum, chunks))

    total_damage = sum(partial_sums)
    elapsed = time.time() - start_time

    print(f"Tiempo: {elapsed:.2f}s")
    print(f"Daño Total: {total_damage}")

    workers = [f'Worker {i+1}' for i in range(num_workers)]

    plt.figure(figsize=(8, 5))
    plt.bar(workers, partial_sums, color='#2c3e50', edgecolor='#1a252f', width=0.6)
    plt.title('Sumas parciales por proceso (Patrón Reduction)')
    plt.ylabel('Daño acumulado')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.show()

    return total_damage, partial_sums


if __name__ == '__main__':
    run_reduction_damage_demo(num_workers=4)