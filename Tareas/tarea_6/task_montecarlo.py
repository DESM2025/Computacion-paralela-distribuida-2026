#!/usr/bin/env python3
"""
task_montecarlo.py — Tarea: estimación de π por método Monte Carlo.

Cada worker simula un lote de muestras aleatorias (semilla fija por chunk
para reproducibilidad) y reporta cuántas cayeron dentro del cuarto de círculo.
El coordinador agrega los resultados y calcula la estimación de π.

Payload esperado:
  {"samples": 2000000}             # muestras totales
  {"samples": 2000000, "seed": 42} # semilla base opcional (default 42)

Ejemplo de uso:
  python3 coordinator_generic.py \\
      --task task_montecarlo.py \\
      --payload '{"samples": 2000000}' \\
      --workers workers.json
"""

from __future__ import annotations
import math
import random


# ──────────────────────────────────────────────
# Contrato de tarea
# ──────────────────────────────────────────────

def split(payload: dict, workers: list) -> list[dict]:
    """
    Divide el total de muestras entre los workers.
    Cada chunk tiene una semilla distinta para resultados independientes.
    """
    total   = int(payload.get("samples", 1_000_000))
    seed    = int(payload.get("seed", 42))
    n       = len(workers)
    base    = total // n
    remainder = total % n

    chunks = []
    for i in range(n):
        samples = base + (1 if i < remainder else 0)
        chunks.append({
            "samples": samples,
            "seed":    seed + i,   # semilla distinta por chunk → varianza real
        })

    return chunks


def run(chunk: dict) -> dict:
    """
    Simula `samples` puntos aleatorios en [0,1)² con la semilla dada.
    Cuenta cuántos caen dentro del cuarto de círculo de radio 1.
    """
    samples = int(chunk["samples"])
    seed    = int(chunk.get("seed", 42))

    rng    = random.Random(seed)
    inside = sum(
        1
        for _ in range(samples)
        if rng.random() ** 2 + rng.random() ** 2 <= 1.0
    )

    return {
        "inside":  inside,
        "samples": samples,
    }


def merge(results: list[dict]) -> dict:
    """
    Estima π = 4 × (total_inside / total_samples).
    Reporta error relativo respecto de math.pi.
    """
    total_inside  = sum(r["inside"]  for r in results)
    total_samples = sum(r["samples"] for r in results)

    pi_estimate  = 4.0 * total_inside / total_samples
    abs_error    = abs(pi_estimate - math.pi)
    rel_error_pct = 100.0 * abs_error / math.pi

    return {
        "pi_estimate":    round(pi_estimate, 8),
        "pi_real":        round(math.pi,     8),
        "abs_error":      round(abs_error,   8),
        "rel_error_pct":  round(rel_error_pct, 4),
        "total_samples":  total_samples,
        "total_inside":   total_inside,
    }
