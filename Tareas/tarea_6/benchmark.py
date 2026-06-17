#!/usr/bin/env python3
"""
benchmark.py — Mide el rendimiento del cluster con N workers.

Corre en nodo0. Ejecuta las 3 tareas en modo distribuido
y en modo secuencial, y guarda los tiempos en un JSON.

Uso (desde nodo0, con las VMs ya encendidas):
  python3 benchmark.py --workers workers-2.json --password 1234
  python3 benchmark.py --workers workers-4.json --password 1234
  python3 benchmark.py --workers workers-6.json --password 1234

Resultado: benchmark_2.json / benchmark_4.json / benchmark_6.json
Luego copia esos JSON a Windows y corre grafico_speedup.py.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time

import coordinator_generic as cg

# ── Configuración de tareas a correr ──────────────
TAREAS = [
    {
        "task_file": "task_primes.py",
        "task_name": "task_primes",
        "payload":   {"upper": 300_000},
        "label":     "Primos (upper=300000)",
    },
    {
        "task_file": "task_wordcount.py",
        "task_name": "task_wordcount",
        "payload":   {"total_words": 100_000},
        "label":     "Wordcount (100k palabras)",
    },
    {
        "task_file": "task_montecarlo.py",
        "task_name": "task_montecarlo",
        "payload":   {"samples": 2_000_000},
        "label":     "Monte Carlo (2M muestras)",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark del orquestador genérico")
    parser.add_argument("--workers",  required=True, help="JSON de workers (ej: workers-4.json)")
    parser.add_argument("--password", default="",    help="Contraseña SSH")
    parser.add_argument("--no-deploy", action="store_true",
                        help="Omitir despliegue SSH (workers ya configurados)")
    args = parser.parse_args()

    with open(args.workers, encoding="utf-8") as f:
        workers = json.load(f)

    n_workers = len(workers)
    output_file = f"benchmark_{n_workers}.json"

    password = args.password
    if not password and not args.no_deploy:
        password = getpass.getpass(f"Contraseña SSH para {n_workers} workers: ")

    print(f"\n{'='*55}")
    print(f"  BENCHMARK — {n_workers} workers")
    print(f"{'='*55}")

    resultados = {
        "n_workers": n_workers,
        "workers":   [w["name"] for w in workers],
        "tareas":    [],
    }

    for t in TAREAS:
        print(f"\n── {t['label']} ──────────────────────────────")

        task = cg.load_task(t["task_file"])
        job_id_dist = f"bench-dist-{n_workers}w-{t['task_name']}"
        job_id_seq  = f"bench-seq-{t['task_name']}"

        # ── Modo distribuido ─────────────────────────
        if not args.no_deploy:
            print("Desplegando...")
            for w in workers:
                cg.deploy_to_worker(w, ["worker_agent.py", t["task_file"]], password)

        print("Corriendo distribuido...")
        summary_dist = cg.run_job(task, t["task_name"], workers, t["payload"], job_id_dist)
        t_dist = summary_dist["elapsed"]
        print(f"  Tiempo distribuido : {t_dist:.3f}s")

        # ── Modo secuencial (baseline) ───────────────
        print("Corriendo secuencial...")
        summary_seq = cg.run_sequential(task, t["task_name"], workers, t["payload"], job_id_seq)
        t_seq = summary_seq["elapsed"]
        print(f"  Tiempo secuencial  : {t_seq:.3f}s")

        speedup = round(t_seq / t_dist, 3) if t_dist > 0 else None
        print(f"  Speedup            : {speedup}x")

        resultados["tareas"].append({
            "tarea":          t["task_name"],
            "label":          t["label"],
            "payload":        t["payload"],
            "t_secuencial":   t_seq,
            "t_distribuido":  t_dist,
            "speedup":        speedup,
            "resultado_dist": summary_dist["result"],
            "resultado_seq":  summary_seq["result"],
            "chunks":         summary_dist["chunks_total"],
            "completados":    summary_dist["completed"],
        })

        # No redesplegar en siguiente tarea si ya está listo
        args.no_deploy = True

    # ── Guardar resultados ───────────────────────────
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"  Resultados guardados en: {output_file}")
    print(f"{'='*55}")

    print("\nResumen:")
    for t in resultados["tareas"]:
        print(f"  {t['tarea']:20s}  seq={t['t_secuencial']:.3f}s  "
              f"dist={t['t_distribuido']:.3f}s  "
              f"speedup={t['speedup']}x")

    print(f"\nCopia {output_file} a Windows y ejecuta grafico_speedup.py")


if __name__ == "__main__":
    main()
