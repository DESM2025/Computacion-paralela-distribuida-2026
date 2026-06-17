#!/usr/bin/env python3
"""
task_wordcount.py — Tarea: conteo de frecuencia de palabras.

Genera un corpus sintético (reproducible por semilla) y cuenta
la frecuencia de cada token.  No requiere archivos externos.

Payload esperado:
  {"total_words": 60000}             # tamaño del corpus
  {"total_words": 60000, "seed": 7}  # semilla opcional (default 123)

Ejemplo de uso:
  python3 coordinator_generic.py \\
      --task task_wordcount.py \\
      --payload '{"total_words": 60000}' \\
      --workers workers.json
"""

from __future__ import annotations
import random


# Vocabulario de muestra con dominio computación/matemática
_VOCAB = [
    "computacion", "paralela", "distribuida", "cluster", "nodo", "worker",
    "coordinador", "tarea", "hilo", "proceso", "red", "socket", "protocolo",
    "resultado", "chunk", "rango", "speedup", "latencia", "eficiencia",
    "balanceo", "reintento", "fallo", "sistema", "algoritmo", "funcion",
    "datos", "memoria", "cpu", "gpu", "python", "parametro", "modulo",
    "importar", "serializar", "json", "puerto", "conexion", "mensaje",
    "the", "and", "of", "in", "to", "a", "is", "that", "it", "for",
]


# ──────────────────────────────────────────────
# Contrato de tarea
# ──────────────────────────────────────────────

def split(payload: dict, workers: list) -> list[dict]:
    """
    Genera corpus de `total_words` palabras (reproducible con `seed`)
    y reparte las líneas resultantes entre los workers.
    Cada chunk contiene {"lines": [str, ...]}.
    """
    total_words = int(payload.get("total_words", 50_000))
    seed        = int(payload.get("seed", 123))
    n           = len(workers)

    rng   = random.Random(seed)
    words = [rng.choice(_VOCAB) for _ in range(total_words)]

    # Agrupar en líneas de 10 palabras para mayor naturalidad
    line_size = 10
    all_lines = [
        " ".join(words[i : i + line_size])
        for i in range(0, len(words), line_size)
    ]

    chunk_size = max(1, len(all_lines) // n)
    chunks = []
    for i in range(n):
        start = i * chunk_size
        end   = start + chunk_size if i < n - 1 else len(all_lines)
        lines = all_lines[start:end]
        if lines:
            chunks.append({"lines": lines})

    return chunks


def run(chunk: dict) -> dict:
    """
    Cuenta frecuencia de tokens en las líneas asignadas.
    """
    lines  = chunk.get("lines", [])
    counts: dict[str, int] = {}

    for line in lines:
        for token in line.split():
            token = token.strip(".,;:!?\"'()-").lower()
            if token:
                counts[token] = counts.get(token, 0) + 1

    return {
        "word_counts":     counts,
        "lines_processed": len(lines),
    }


def merge(results: list[dict]) -> dict:
    """
    Consolida los diccionarios parciales en un conteo global.
    Reporta top-10 palabras más frecuentes.
    """
    total_counts: dict[str, int] = {}
    total_lines = 0

    for r in results:
        total_lines += r.get("lines_processed", 0)
        for word, count in r.get("word_counts", {}).items():
            total_counts[word] = total_counts.get(word, 0) + count

    top10 = dict(
        sorted(total_counts.items(), key=lambda x: -x[1])[:10]
    )

    return {
        "total_tokens":  sum(total_counts.values()),
        "unique_words":  len(total_counts),
        "lines_total":   total_lines,
        "top_10":        top10,
    }
