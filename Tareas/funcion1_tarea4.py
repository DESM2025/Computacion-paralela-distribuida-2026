import concurrent.futures
import hashlib
import random
import string
import time
from collections import Counter

import matplotlib.pyplot as plt


def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()


def run_map_password_hash_demo(num_passwords=100000, password_length=8, num_workers=4):
    passwords = [
        ''.join(random.choices(string.ascii_letters + string.digits, k=password_length))
        for _ in range(num_passwords)
    ]

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        hashes = list(executor.map(hash_password, passwords))

    elapsed = time.time() - start_time
    print(f"Tiempo: {elapsed:.2f}s")
    print(f"Total procesado: {len(hashes)}")

    first_chars = [h[0] for h in hashes]
    counts = Counter(first_chars)
    labels, values = zip(*sorted(counts.items()))

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='#1f77b4', edgecolor='#08306b')
    plt.title('Distribución del primer carácter hexadecimal (Patrón Map)')
    plt.xlabel('Carácter hexadecimal')
    plt.ylabel('Frecuencia')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return hashes


if __name__ == '__main__':
    run_map_password_hash_demo(num_workers=4)