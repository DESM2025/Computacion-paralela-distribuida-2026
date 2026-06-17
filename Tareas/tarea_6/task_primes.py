#!/usr/bin/env python3
"""
task_primes.py — Tarea: conteo de números primos en un rango.

Contrato split/run/merge para el coordinador genérico.

Payload esperado:
  {"upper": 300000}          # cuenta primos en [2, upper]
  {"upper": 300000, "lower": 2}   # opcional: límite inferior

Ejemplo de uso:
  python3 coordinator_generic.py \\
      --task task_primes.py \\
      --payload '{"upper": 300000}' \\
      --workers workers.json
"""

from __future__ import annotations
import math


# ──────────────────────────────────────────────
# Contrato de tarea
# ──────────────────────────────────────────────

def split(payload: dict, workers: list) -> list[dict]:
    """
    Divide el rango [lower, upper] en un chunk por worker.
    Cada chunk contiene {"start": int, "end": int}.
    """
    upper = int(payload.get("upper", 100_000))
    lower = int(payload.get("lower", 2))
    n     = len(workers)

    if upper < lower:
        raise ValueError(f"upper ({upper}) debe ser >= lower ({lower})")

    total = upper - lower + 1
    size  = total // n

    chunks  = []
    current = lower

    for i in range(n):
        start = current
        end   = (current + size - 1) if i < n - 1 else upper
        chunks.append({"start": start, "end": end})
        current = end + 1

    return chunks


def run(chunk: dict) -> dict:
    """
    Cuenta primos en el rango [start, end].
    Función pura: solo depende del chunk.
    """
    start = int(chunk["start"])
    end   = int(chunk["end"])

    count = sum(1 for n in range(start, end + 1) if _is_prime(n))

    return {
        "prime_count": count,
        "range":       [start, end],
    }


def merge(results: list[dict]) -> dict:
    """
    Suma los prime_count parciales y reporta cobertura.
    """
    total  = sum(r["prime_count"] for r in results)
    ranges = [r["range"] for r in results]

    return {
        "total_primes":    total,
        "chunks_merged":   len(results),
        "ranges_computed": ranges,
    }


# ──────────────────────────────────────────────
# Helper interno (no forma parte del contrato)
# ──────────────────────────────────────────────

def _is_prime(n: int) -> bool:
    if n < 2:         return False
    if n == 2:        return True
    if n % 2 == 0:    return False
    limit = int(math.sqrt(n)) + 1
    for d in range(3, limit, 2):
        if n % d == 0:
            return False
    return True
