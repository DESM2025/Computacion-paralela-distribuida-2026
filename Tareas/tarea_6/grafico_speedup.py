#!/usr/bin/env python3
"""
grafico_speedup.py — Genera gráfico de speedup a partir de los resultados del benchmark.

Corre en Windows (requiere matplotlib).
Instalar: pip install matplotlib

Uso:
  1. Copia benchmark_2.json, benchmark_4.json, benchmark_6.json desde nodo0 a esta carpeta
  2. python grafico_speedup.py

Genera: speedup_comparativo.png
"""

from __future__ import annotations

import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("Falta matplotlib. Instálalo con:  pip install matplotlib")
    sys.exit(1)


def cargar_benchmark(n: int) -> dict | None:
    path = f"benchmark_{n}.json"
    if not os.path.exists(path):
        print(f"  No encontrado: {path} — omitiendo {n} workers")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    configs = [2, 4, 6]
    datos   = {n: cargar_benchmark(n) for n in configs}
    datos   = {n: d for n, d in datos.items() if d is not None}

    if not datos:
        print("No se encontró ningún archivo benchmark_N.json.")
        print("Corre benchmark.py en nodo0 primero.")
        sys.exit(1)

    # Extraer nombres de tareas del primer benchmark disponible
    primer = next(iter(datos.values()))
    tareas = [t["tarea"] for t in primer["tareas"]]
    labels = {t["tarea"]: t["label"] for t in primer["tareas"]}

    colores = ["#3A86FF", "#06D6A0", "#EF476F"]
    markers = ["o", "s", "^"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Análisis de Rendimiento — Orquestador Genérico QEMU", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#F8F9FA")

    # ── Gráfico 1: Speedup por tarea (barras agrupadas) ──
    ax1 = axes[0]
    ns_disponibles = sorted(datos.keys())

    import numpy as np
    x     = np.arange(len(tareas))
    ancho1 = 0.8 / len(ns_disponibles)
    bar_colores = ["#3A86FF", "#FF6B6B"]

    for j, n in enumerate(ns_disponibles):
        speedups = []
        for tarea in tareas:
            sp = next((t["speedup"] for t in datos[n]["tareas"] if t["tarea"] == tarea), 0)
            speedups.append(sp or 0)
        offset = (j - len(ns_disponibles) / 2 + 0.5) * ancho1
        ax1.bar(x + offset, speedups, width=ancho1,
                label=f"{n} workers", color=bar_colores[j % len(bar_colores)], alpha=0.85)

    ax1.axhline(y=1, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="Speedup = 1")
    ax1.set_xlabel("Tarea", fontsize=12)
    ax1.set_ylabel("Speedup (T_secuencial / T_distribuido)", fontsize=12)
    ax1.set_title("Speedup por tarea", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels.get(t, t) for t in tareas], fontsize=9, rotation=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(bottom=0)

    # ── Gráfico 2: Tiempo distribuido vs secuencial ───
    ax2 = axes[1]
    x    = range(len(tareas))
    ancho = 0.2

    # Secuencial (solo del primer config disponible, es el mismo)
    t_seq = []
    for tarea in tareas:
        for t in primer["tareas"]:
            if t["tarea"] == tarea:
                t_seq.append(t["t_secuencial"])

    ax2.bar([xi - ancho for xi in x], t_seq, width=ancho,
            label="Secuencial", color="#AAAAAA", alpha=0.85)

    for j, n in enumerate(ns_disponibles):
        t_dist = []
        for tarea in tareas:
            for t in datos[n]["tareas"]:
                if t["tarea"] == tarea:
                    t_dist.append(t["t_distribuido"])
        offset = (j - len(ns_disponibles)/2 + 0.5) * ancho
        bar_col2 = ["#3A86FF", "#FF6B6B"]
        ax2.bar([xi + offset + ancho for xi in x], t_dist, width=ancho,
                label=f"{n} workers", color=bar_col2[j % 2], alpha=0.85)

    ax2.set_xticks(list(x))
    ax2.set_xticklabels([labels.get(t, t) for t in tareas], fontsize=9, rotation=10)
    ax2.set_ylabel("Tiempo (segundos)", fontsize=12)
    ax2.set_title("Tiempo: secuencial vs distribuido", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    out = "speedup_comparativo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Gráfico guardado: {out}")

    # ── Tabla de resumen en consola ───────────────────
    print("\nResumen de speedup:")
    print(f"{'Tarea':25s}", end="")
    for n in ns_disponibles:
        print(f"  {n} workers", end="")
    print()
    print("-" * (25 + 12 * len(ns_disponibles)))

    for tarea in tareas:
        print(f"{labels.get(tarea, tarea):25s}", end="")
        for n in ns_disponibles:
            sp = next(
                (t["speedup"] for t in datos[n]["tareas"] if t["tarea"] == tarea),
                None
            )
            print(f"  {sp:>8.2f}x " if sp else "       N/A  ", end="")
        print()

    plt.show()


if __name__ == "__main__":
    main()
