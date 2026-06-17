#!/usr/bin/env python3
"""
worker_agent.py — Agente genérico de ejecución de tareas.

Escucha en TCP :9000 y ejecuta cualquier tarea que cumpla el contrato
split/run/merge.  El agente no conoce el dominio del problema; solo
importa la tarea indicada por task_name y llama a run(chunk).

Protocolo de entrada (una línea JSON):
  {
    "job_id":    "job-2026-001",
    "chunk_id":  7,
    "task_name": "task_primes",
    "chunk":     {"start": 100001, "end": 150000}
  }

Protocolo de salida (una línea JSON):
  {
    "ok":       true,
    "chunk_id": 7,
    "result":   {"prime_count": 4256},
    "seconds":  0.83
  }
  o en caso de error:
  {
    "ok":       false,
    "chunk_id": 7,
    "error":    "descripción del error"
  }

Uso:
  python3 worker_agent.py          # puerto por defecto 9000
  AGENT_PORT=9001 python3 worker_agent.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import socketserver
import sys
import time
import traceback

TASK_DIR = "/root/orchestrator"
AGENT_VERSION = "generic-v1"


# ──────────────────────────────────────────────
# Carga dinámica de módulo de tarea
# ──────────────────────────────────────────────

def load_task(task_name: str):
    """
    Importa task_name desde TASK_DIR.
    task_name puede ser "task_primes" o "task_primes.py".
    Valida que el módulo exponga la función 'run'.
    """
    fname = task_name if task_name.endswith(".py") else f"{task_name}.py"
    path = os.path.join(TASK_DIR, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tarea no encontrada: {path}  "
            f"(TASK_DIR={TASK_DIR}, archivos disponibles: "
            f"{os.listdir(TASK_DIR) if os.path.isdir(TASK_DIR) else 'directorio no existe'})"
        )

    spec = importlib.util.spec_from_file_location(task_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "run"):
        raise AttributeError(
            f"La tarea '{task_name}' no define la función 'run(chunk) -> dict'"
        )

    return mod


# ──────────────────────────────────────────────
# Handler de conexión TCP
# ──────────────────────────────────────────────

class AgentHandler(socketserver.StreamRequestHandler):
    def handle(self):
        client = self.client_address[0]
        chunk_id = -1
        req = {}

        try:
            raw = self.rfile.readline().decode("utf-8").strip()
            if not raw:
                return

            req = json.loads(raw)
            chunk_id = req.get("chunk_id", -1)
            task_name = req["task_name"]

            # Mensaje de health-check: responder sin cargar ninguna tarea
            if task_name == "probe":
                resp = {
                    "ok": True,
                    "chunk_id": chunk_id,
                    "result": {"probe": "ok", "version": AGENT_VERSION},
                    "seconds": 0.0,
                }
                self._send(resp)
                return

            # Cargar tarea y ejecutar run(chunk)
            task = load_task(task_name)
            chunk = req["chunk"]

            t0 = time.time()
            result = task.run(chunk)
            elapsed = round(time.time() - t0, 4)

            if not isinstance(result, dict):
                raise TypeError(
                    f"run() debe retornar dict, recibió {type(result).__name__}"
                )

            resp = {
                "ok": True,
                "chunk_id": chunk_id,
                "result": result,
                "seconds": elapsed,
            }

        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            resp = {
                "ok": False,
                "chunk_id": chunk_id,
                "error": str(exc),
            }

        self._send(resp)

    def _send(self, data: dict):
        try:
            self.wfile.write((json.dumps(data) + "\n").encode("utf-8"))
            self.wfile.flush()
        except Exception:
            pass


# ──────────────────────────────────────────────
# Servidor TCP con soporte de múltiples conexiones
# ──────────────────────────────────────────────

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    port = int(os.environ.get("AGENT_PORT", 9000))

    os.makedirs(TASK_DIR, exist_ok=True)

    print(f"Worker Agent v{AGENT_VERSION}", flush=True)
    print(f"Escuchando en 0.0.0.0:{port}", flush=True)
    print(f"Directorio de tareas: {TASK_DIR}", flush=True)
    print(f"Hostname: {socket.gethostname()}", flush=True)

    server = ThreadedTCPServer(("0.0.0.0", port), AgentHandler)
    server.serve_forever()
