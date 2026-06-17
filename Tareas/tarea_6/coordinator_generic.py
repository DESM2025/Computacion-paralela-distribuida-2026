#!/usr/bin/env python3
"""
coordinator_generic.py — Coordinador genérico de tareas distribuidas.

Construido sobre la demo de primos del cluster QEMU.
Implementa las versiones progresivas V1-V5 del ejercicio 8.3.

  V1 — Carga dinámica de módulo de tarea (importlib)
  V2 — Protocolo JSON explícito con job_id / chunk_id / ok
  V3 — Despliegue automático vía SSH (paramiko, hash, logs)
  V4 — Cola dinámica: workers reciben chunks según disponibilidad
  V5 — Máquina de estados por chunk: reintentos con límite

Uso desde nodo0:
  python3 coordinator_generic.py \\
      --task     task_primes.py \\
      --payload  '{"upper": 300000}' \\
      --workers  workers.json

  python3 coordinator_generic.py \\
      --task     task_wordcount.py \\
      --payload  '{"total_words": 60000}' \\
      --workers  workers.json

  # Modo secuencial (baseline para speedup):
  python3 coordinator_generic.py \\
      --task      task_primes.py \\
      --payload   '{"upper": 300000}' \\
      --workers   workers.json \\
      --sequential
"""

from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

MAX_RETRIES = 3
REMOTE_DIR  = "/root/orchestrator"
SSH_USER    = "root"

# Estados del ciclo de vida de un chunk (V5)
PENDING   = "pending"
RUNNING   = "running"
COMPLETED = "completed"
FAILED    = "failed"


# ──────────────────────────────────────────────
# V1 — Carga dinámica del módulo de tarea
# ──────────────────────────────────────────────

def load_task(task_path: str):
    """
    Importa el módulo de tarea desde task_path usando importlib.
    Valida que exponga las tres funciones del contrato: split, run, merge.
    """
    if not os.path.exists(task_path):
        _die(f"No existe el archivo de tarea: {task_path}")

    spec = importlib.util.spec_from_file_location("task", task_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        _die(f"Error al importar {task_path}: {exc}")

    for fn in ("split", "run", "merge"):
        if not hasattr(mod, fn):
            _die(
                f"La tarea '{task_path}' no define la función '{fn}'.\n"
                f"Contrato requerido: split(payload, workers), "
                f"run(chunk), merge(results)"
            )

    return mod


# ──────────────────────────────────────────────
# V2 — Protocolo JSON de comunicación
# ──────────────────────────────────────────────

def send_chunk(worker: Dict, msg: Dict, timeout: float = 60.0) -> Tuple[bool, Any, float]:
    """
    Envía msg al worker y espera la respuesta JSON.
    Retorna (ok, result_o_error, segundos_elapsed).
    """
    t0 = time.time()
    try:
        host = worker["task_host"]
        port = int(worker["task_port"])
        payload = json.dumps(msg) + "\n"

        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(payload.encode("utf-8"))

            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk

        elapsed = time.time() - t0

        if not data:
            return False, f"{worker['name']}: conexión cerrada sin respuesta", elapsed

        resp = json.loads(data.decode("utf-8"))

        if resp.get("ok"):
            return True, resp["result"], elapsed
        else:
            return False, resp.get("error", "respuesta con ok=false sin detalle"), elapsed

    except json.JSONDecodeError as exc:
        return False, f"JSON inválido en respuesta: {exc}", time.time() - t0
    except Exception as exc:
        return False, str(exc), time.time() - t0


def worker_healthy(worker: Dict, timeout: float = 5.0) -> bool:
    """
    Prueba funcional: envía un probe y verifica respuesta correcta.
    NO se fía solo de que el puerto esté abierto.
    """
    probe = {"job_id": "probe", "chunk_id": -1, "task_name": "probe", "chunk": {}}
    ok, _, _ = send_chunk(worker, probe, timeout)
    return ok


# ──────────────────────────────────────────────
# V3 — Despliegue automático vía SSH
# ──────────────────────────────────────────────

def _require_paramiko():
    try:
        import paramiko
        return paramiko
    except ImportError:
        _die(
            "Falta paramiko. Instálalo en nodo0:\n"
            "    apk add py3-paramiko\n"
            "  o bien:\n"
            "    python3 -m pip install paramiko\n"
            "Luego reintenta, o usa --no-deploy si los workers ya están listos."
        )


def _ssh_connect(worker: Dict, password: str):
    paramiko = _require_paramiko()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=worker["ssh_host"],
        port=int(worker["ssh_port"]),
        username=SSH_USER,
        password=password,
        timeout=15,
        look_for_keys=False,
        allow_agent=False,
        banner_timeout=15,
        auth_timeout=15,
    )
    return client


def _ssh_exec(client, cmd: str, check: bool = True) -> Tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if check and exit_code != 0:
        raise RuntimeError(
            f"Comando remoto falló (código {exit_code}):\n"
            f"CMD : {cmd}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        )
    return exit_code, out, err


def _upload_file_if_changed(sftp, local_path: str, remote_path: str, name: str) -> bool:
    """Compara hashes y copia solo si el contenido cambió. Retorna True si copió."""
    with open(local_path, "rb") as f:
        local_bytes = f.read()

    try:
        with sftp.open(remote_path, "rb") as rf:
            remote_bytes = rf.read()
        if remote_bytes == local_bytes:
            return False   # sin cambios
    except OSError:
        pass   # archivo no existe remotamente

    with sftp.open(remote_path, "wb") as rf:
        rf.write(local_bytes)
    return True


def _restart_agent(client, worker: Dict):
    """Mata worker-agent anterior e inicia uno nuevo. Espera hasta que responda."""
    name     = worker["name"]
    pid_path = f"/tmp/agent_{name}.pid"
    log_path = f"/tmp/agent_{name}.log"
    agent    = f"{REMOTE_DIR}/worker_agent.py"

    print(f"  [{name}] reiniciando worker-agent...")

    kill_cmd = f"""
if [ -f {pid_path} ]; then
  kill "$(cat {pid_path})" 2>/dev/null || true
  rm -f {pid_path}
fi
for pid in $(ps 2>/dev/null | grep -v grep | grep -E 'worker_agent|worker\\.py' | awk '{{print $1}}'); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.5
"""
    _ssh_exec(client, kill_cmd, check=False)

    start_cmd = (
        f"nohup python3 {agent} "
        f"> {log_path} 2>&1 "
        f"& echo $! > {pid_path}"
    )
    _ssh_exec(client, start_cmd)

    # Esperar hasta 25 s que el agente responda
    deadline = time.time() + 25
    while time.time() < deadline:
        if worker_healthy(worker, timeout=2.0):
            print(f"  [{name}] worker-agent en línea ✓")
            return
        time.sleep(0.8)

    # Si no responde, mostrar log
    _, log_out, _ = _ssh_exec(client, f"tail -30 {log_path} 2>/dev/null || true", check=False)
    if log_out.strip():
        print(f"  [{name}] log del agente:\n{log_out}")

    raise RuntimeError(f"[{name}] worker-agent no respondió tras reinicio")


def deploy_to_worker(worker: Dict, files: List[str], password: str):
    """
    Copia archivos al worker vía SFTP (solo los que cambiaron) y
    reinicia el agente si alguno cambió o si no está saludable.
    """
    name = worker["name"]
    print(f"  [{name}] conectando por SSH a {worker['ssh_host']}:{worker['ssh_port']}...")

    client = _ssh_connect(worker, password)
    try:
        _ssh_exec(client, f"mkdir -p {REMOTE_DIR}")

        sftp = client.open_sftp()
        any_changed = False

        try:
            for local_path in files:
                fname = os.path.basename(local_path)
                remote_path = f"{REMOTE_DIR}/{fname}"
                changed = _upload_file_if_changed(sftp, local_path, remote_path, fname)
                if changed:
                    print(f"  [{name}] copiado: {fname} (SHA256: {_file_hash(local_path)[:12]}…)")
                    any_changed = True
                else:
                    print(f"  [{name}] sin cambios: {fname}")
        finally:
            sftp.close()

        if any_changed or not worker_healthy(worker, timeout=3.0):
            _restart_agent(client, worker)
        else:
            print(f"  [{name}] agente ya estaba en línea, no se reinicia.")

    finally:
        client.close()


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ──────────────────────────────────────────────
# V4 + V5 — Cola dinámica + estados + reintentos
# ──────────────────────────────────────────────

def run_job(
    task,
    task_name: str,
    workers: List[Dict],
    payload: Dict,
    job_id: str,
) -> Dict:
    """
    Ejecuta el job distribuido con cola dinámica (V4) y
    máquina de estados por chunk (V5).

    Flujo:
      1. split(payload, workers)   → lista de chunks
      2. Envío paralelo a workers con protocolo JSON (V2)
      3. Chunks fallidos → reintento hasta MAX_RETRIES (V5)
      4. merge(partial_results)    → resultado final
    """
    # ── split (V1) ──────────────────────────────
    chunks_data = task.split(payload, workers)
    n = len(chunks_data)

    print(f"\n[{job_id}] Tarea   : {task_name}")
    print(f"[{job_id}] Workers : {[w['name'] for w in workers]}")
    print(f"[{job_id}] Chunks  : {n}")
    print()

    # ── estado inicial de cada chunk (V5) ────────
    states = [
        {
            "chunk_id": i,
            "chunk":    c,
            "state":    PENDING,
            "result":   None,
            "error":    None,
            "retries":  0,
            "worker":   None,
            "seconds":  None,
        }
        for i, c in enumerate(chunks_data)
    ]

    pending       = list(range(n))
    total_retries = 0
    t0_total      = time.time()

    # ── cola dinámica con reintentos (V4 + V5) ───
    while pending:
        # Asignación round-robin de workers a chunks pendientes
        assignments = {
            chunk_id: workers[i % len(workers)]
            for i, chunk_id in enumerate(pending)
        }

        # Marcar chunks como 'running'
        for chunk_id, w in assignments.items():
            states[chunk_id]["state"]  = RUNNING
            states[chunk_id]["worker"] = w["name"]

        # Envío concurrente: todos los pendientes en paralelo (V4)
        futures: Dict[concurrent.futures.Future, int] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending)) as ex:
            for chunk_id, w in assignments.items():
                cs  = states[chunk_id]
                msg = {
                    "job_id":    job_id,
                    "chunk_id":  chunk_id,
                    "task_name": task_name,
                    "chunk":     cs["chunk"],
                }
                f = ex.submit(send_chunk, w, msg)
                futures[f] = chunk_id

            next_pending = []
            for f in concurrent.futures.as_completed(futures):
                chunk_id = futures[f]
                cs       = states[chunk_id]
                ok, result_or_err, elapsed = f.result()

                if ok:
                    cs["state"]   = COMPLETED
                    cs["result"]  = result_or_err
                    cs["seconds"] = round(elapsed, 4)
                    print(
                        f"  ✓ [{cs['worker']}] chunk {chunk_id} "
                        f"completado ({elapsed:.3f}s)"
                    )
                else:
                    cs["retries"] += 1
                    total_retries += 1

                    if cs["retries"] < MAX_RETRIES:
                        cs["state"] = PENDING
                        next_pending.append(chunk_id)
                        print(
                            f"  ↻ [{cs['worker']}] chunk {chunk_id} "
                            f"falló (intento {cs['retries']}/{MAX_RETRIES}): "
                            f"{result_or_err}"
                        )
                    else:
                        cs["state"] = FAILED
                        cs["error"] = str(result_or_err)
                        print(
                            f"  ✗ [{cs['worker']}] chunk {chunk_id} "
                            f"FALLÓ definitivamente: {result_or_err}"
                        )

        pending = next_pending

    elapsed_total = time.time() - t0_total

    # ── merge (V1) ───────────────────────────────
    completed_results = [s["result"] for s in states if s["state"] == COMPLETED]
    failed_chunks     = [
        {"chunk_id": s["chunk_id"], "error": s["error"]}
        for s in states if s["state"] == FAILED
    ]

    final_result = task.merge(completed_results)

    summary = {
        "job_id":        job_id,
        "task":          task_name,
        "workers":       [w["name"] for w in workers],
        "chunks_total":  n,
        "completed":     len(completed_results),
        "failed":        len(failed_chunks),
        "failed_chunks": failed_chunks,
        "total_retries": total_retries,
        "elapsed":       round(elapsed_total, 3),
        "result":        final_result,
        "chunk_details": [
            {
                "chunk_id": s["chunk_id"],
                "state":    s["state"],
                "worker":   s["worker"],
                "seconds":  s["seconds"],
                "retries":  s["retries"],
            }
            for s in states
        ],
    }

    return summary


def run_sequential(task, task_name: str, workers: List[Dict], payload: Dict, job_id: str) -> Dict:
    """
    Modo secuencial (baseline): ejecuta todo en el coordinador sin red.
    Sirve para calcular speedup vs. la versión distribuida.
    """
    print(f"\n[{job_id}] MODO SECUENCIAL (baseline)")
    chunks_data = task.split(payload, [workers[0]])  # simula 1 solo worker
    t0 = time.time()
    results = [task.run(c) for c in chunks_data]
    elapsed = time.time() - t0
    final   = task.merge(results)
    return {
        "job_id":   job_id,
        "task":     task_name,
        "mode":     "sequential",
        "chunks":   len(chunks_data),
        "elapsed":  round(elapsed, 3),
        "result":   final,
    }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _die(msg: str):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _log_result(summary: Dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResultado guardado en: {output_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Coordinador genérico de tareas distribuidas sobre cluster QEMU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python3 coordinator_generic.py \\
      --task task_primes.py --payload '{"upper":300000}' --workers workers.json

  python3 coordinator_generic.py \\
      --task task_wordcount.py --payload '{"total_words":60000}' --workers workers.json

  python3 coordinator_generic.py \\
      --task task_primes.py --payload '{"upper":300000}' --workers workers.json \\
      --sequential    # baseline secuencial
""",
    )
    parser.add_argument("--task",         required=True,  help="Archivo .py con split/run/merge")
    parser.add_argument("--payload",      required=True,  help="JSON con parámetros de la tarea")
    parser.add_argument("--workers",      required=True,  help="JSON con lista de workers")
    parser.add_argument("--output",       default="resultado.json", help="Archivo de salida")
    parser.add_argument("--agent",        default="worker_agent.py",
                        help="Ruta al worker_agent.py a desplegar (default: ./worker_agent.py)")
    parser.add_argument("--ssh-password", default="",
                        help="Contraseña SSH (si se omite, se pide interactivamente)")
    parser.add_argument("--no-deploy",    action="store_true",
                        help="Omitir despliegue SSH (asumir workers ya configurados)")
    parser.add_argument("--sequential",   action="store_true",
                        help="Ejecutar en modo secuencial local (baseline para speedup)")
    args = parser.parse_args()

    # ── Cargar tarea (V1) ───────────────────────
    task      = load_task(args.task)
    task_name = os.path.splitext(os.path.basename(args.task))[0]

    # ── Cargar workers y payload ─────────────────
    with open(args.workers, encoding="utf-8") as f:
        workers = json.load(f)

    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        _die(f"--payload no es JSON válido: {exc}")

    job_id = f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    print("=" * 50)
    print("  Coordinador Genérico  —  Ejercicio 8.3")
    print("=" * 50)
    print(f"Job ID  : {job_id}")
    print(f"Tarea   : {args.task}")
    print(f"Payload : {args.payload}")
    print(f"Workers : {args.workers}")

    # ── Modo secuencial (baseline) ───────────────
    if args.sequential:
        summary = run_sequential(task, task_name, workers, payload, job_id)
        print(f"\nResultado (secuencial):")
        print(json.dumps(summary["result"], indent=2, ensure_ascii=False))
        print(f"Tiempo secuencial: {summary['elapsed']}s")
        _log_result(summary, args.output)
        return

    # ── Despliegue vía SSH (V3) ──────────────────
    if not args.no_deploy:
        agent_path = args.agent
        if not os.path.exists(agent_path):
            _die(
                f"No encuentro worker_agent.py en: {agent_path}\n"
                f"Especifica la ruta con --agent o usa --no-deploy."
            )

        ssh_password = args.ssh_password
        if not ssh_password:
            ssh_password = getpass.getpass(f"Contraseña SSH para usuario '{SSH_USER}': ")

        print("\n── Desplegando archivos ──────────────────────")
        files_to_deploy = [agent_path, args.task]

        for worker in workers:
            try:
                deploy_to_worker(worker, files_to_deploy, ssh_password)
            except Exception as exc:
                _die(
                    f"Fallo al desplegar en {worker['name']}: {exc}\n"
                    f"Verifica que la VM esté encendida y que el puerto SSH "
                    f"{worker['ssh_port']} esté accesible."
                )
    else:
        print("\n[--no-deploy] Omitiendo despliegue. Verificando workers...")
        for worker in workers:
            if worker_healthy(worker, timeout=5.0):
                print(f"  [{worker['name']}] OK")
            else:
                print(
                    f"  [{worker['name']}] NO responde. "
                    f"Ejecuta sin --no-deploy para desplegar automáticamente."
                )

    # ── Ejecutar job distribuido (V2 + V4 + V5) ─
    print("\n── Ejecutando job ────────────────────────────")
    summary = run_job(task, task_name, workers, payload, job_id)

    # ── Mostrar resultados ───────────────────────
    print(f"\n── Resultado ─────────────────────────────────")
    print(json.dumps(summary["result"], indent=2, ensure_ascii=False))
    print(f"\nChunks completados : {summary['completed']}/{summary['chunks_total']}")
    if summary["failed"]:
        print(f"Chunks fallidos    : {summary['failed']}")
        for fc in summary["failed_chunks"]:
            print(f"  chunk {fc['chunk_id']}: {fc['error']}")
    if summary["total_retries"]:
        print(f"Reintentos totales : {summary['total_retries']}")
    print(f"Tiempo distribuido : {summary['elapsed']}s")

    _log_result(summary, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        sys.exit(130)
