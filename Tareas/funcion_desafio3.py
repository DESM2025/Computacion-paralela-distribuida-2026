import concurrent.futures
import random
import re
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

RUTAS = [
    '/index.html', '/api/v1/users', '/api/v1/orders', '/api/v1/products',
    '/login', '/logout', '/static/app.js', '/static/style.css',
    '/dashboard', '/admin/panel', '/api/v1/search', '/health',
]
METODOS = ['GET', 'GET', 'GET', 'POST', 'POST', 'PUT', 'DELETE']
ESTADOS = [200, 200, 200, 200, 301, 304, 400, 403, 404, 500, 502]
PATRON_LOG = re.compile(
    r'(\S+) - - \[(.+?)\] "(\S+) (\S+) \S+" (\d+) (\d+)'
)


# Etapa 1: parsear líneas crudas a diccionarios estructurados
def parsear_lote(lineas):
    registros = []
    for linea in lineas:
        m = PATRON_LOG.match(linea)
        if m:
            ip, timestamp, metodo, ruta, estado, tamaño = m.groups()
            hora = int(timestamp.split(':')[1])
            registros.append({
                'ip':     ip,
                'hora':   hora,
                'metodo': metodo,
                'ruta':   ruta,
                'estado': int(estado),
                'tamaño': int(tamaño),
            })
    return registros


# Etapa 2: añadir categoría HTTP a cada registro
def enriquecer_lote(registros):
    for r in registros:
        estado = r['estado']
        if estado < 300:
            r['categoria'] = 'éxito'
        elif estado < 400:
            r['categoria'] = 'redirección'
        elif estado < 500:
            r['categoria'] = 'error cliente'
        else:
            r['categoria'] = 'error servidor'
    return registros


# Etapa 3: contar ocurrencias locales por IP, ruta, hora y categoría
def agregar_lote(registros):
    return {
        'por_ip':        Counter(r['ip']        for r in registros),
        'por_ruta':      Counter(r['ruta']       for r in registros),
        'por_hora':      Counter(r['hora']       for r in registros),
        'por_categoria': Counter(r['categoria']  for r in registros),
    }


def generar_logs(n, seed=42):
    # Genera n líneas de log HTTP sintéticas con distribución de tráfico por hora
    rng = random.Random(seed)
    ips = [f'192.168.{rng.randint(0, 10)}.{rng.randint(1, 254)}' for _ in range(500)]
    pesos_hora = [1]*8 + [5]*4 + [8]*4 + [6]*4 + [3]*2 + [1]*2
    horas = list(range(24))
    lineas = []
    for _ in range(n):
        ip      = rng.choice(ips)
        hora    = rng.choices(horas, weights=pesos_hora)[0]
        minuto  = rng.randint(0, 59)
        segundo = rng.randint(0, 59)
        metodo  = rng.choice(METODOS)
        ruta    = rng.choice(RUTAS)
        estado  = rng.choice(ESTADOS)
        tamaño  = rng.randint(200, 15000)
        ts = f'01/Jan/2025:{hora:02d}:{minuto:02d}:{segundo:02d} +0000'
        lineas.append(
            f'{ip} - - [{ts}] "{metodo} {ruta} HTTP/1.1" {estado} {tamaño}'
        )
    return lineas


def run_http_pipeline(n_lineas=300_000, num_workers=4, seed=42):

    print(f'Generando {n_lineas:,} líneas de log HTTP...')
    lineas = generar_logs(n_lineas, seed)
    chunks = [lineas[i::num_workers] for i in range(num_workers)]

    tiempos = {}

    # Etapa 1: parseo paralelo (Map — cada worker parsea su chunk de forma independiente)
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        resultados_etapa1 = list(ex.map(parsear_lote, chunks))
    tiempos['parseo'] = time.perf_counter() - t0

    # Barrera: etapa 1 completa; se consolidan los registros parseados
    total_parseados = sum(len(r) for r in resultados_etapa1)
    print(f'[Etapa 1 - Parseo]      {tiempos["parseo"]:.3f} s  |  {total_parseados:,} registros')

    # Etapa 2: enriquecimiento paralelo (Map — cada worker añade la categoría a sus registros)
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        resultados_etapa2 = list(ex.map(enriquecer_lote, resultados_etapa1))
    tiempos['enriquecimiento'] = time.perf_counter() - t0

    # Barrera: etapa 2 completa
    print(f'[Etapa 2 - Enriquec.]   {tiempos["enriquecimiento"]:.3f} s')

    # Etapa 3: agregación paralela (Map) + reducción (suma de Counters parciales)
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        resultados_etapa3 = list(ex.map(agregar_lote, resultados_etapa2))
    tiempos['agregacion'] = time.perf_counter() - t0

    # Barrera: etapa 3 completa; se reduce sumando los Counters de cada worker
    agg_global = {
        'por_ip':        Counter(),
        'por_ruta':      Counter(),
        'por_hora':      Counter(),
        'por_categoria': Counter(),
    }
    for parcial in resultados_etapa3:
        for clave in agg_global:
            agg_global[clave] += parcial[clave]

    print(f'[Etapa 3 - Agregación]  {tiempos["agregacion"]:.3f} s')
    print(f'Tiempo total pipeline:  {sum(tiempos.values()):.3f} s')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Gráfico 1: tráfico por hora del día
    horas_ord = sorted(agg_global['por_hora'].keys())
    conteos_h = [agg_global['por_hora'][h] for h in horas_ord]
    axes[0].bar(horas_ord, conteos_h, color='#2c3e50', edgecolor='#1a252f', width=0.8)
    axes[0].set_title('Tráfico por hora del día')
    axes[0].set_xlabel('Hora')
    axes[0].set_ylabel('Número de peticiones')
    axes[0].set_xticks(horas_ord[::2])
    axes[0].grid(True, linestyle='--', alpha=0.5, axis='y')

    # Gráfico 2: top 8 rutas más solicitadas
    top_rutas = agg_global['por_ruta'].most_common(8)
    rutas_lbl = [r for r, _ in top_rutas]
    rutas_cnt = [c for _, c in top_rutas]
    colores_r = plt.cm.Blues(np.linspace(0.4, 0.9, len(rutas_lbl)))
    axes[1].barh(rutas_lbl, rutas_cnt, color=colores_r, edgecolor='#08306b')
    axes[1].set_title('Top 8 rutas más solicitadas')
    axes[1].set_xlabel('Número de peticiones')
    axes[1].invert_yaxis()
    axes[1].grid(True, linestyle='--', alpha=0.5, axis='x')

    # Gráfico 3: tiempo por etapa del pipeline
    etapas    = ['Parseo\n(Etapa 1)', 'Enriquec.\n(Etapa 2)', 'Agregación\n(Etapa 3)']
    t_vals    = [tiempos['parseo'], tiempos['enriquecimiento'], tiempos['agregacion']]
    colores_e = ['#1f77b4', '#2ca02c', '#d62728']
    axes[2].bar(etapas, t_vals, color=colores_e, edgecolor='#1f2937', width=0.5)
    axes[2].set_title(f'Tiempo por etapa del pipeline ({num_workers} workers)')
    axes[2].set_ylabel('Tiempo [s]')
    axes[2].grid(True, linestyle='--', alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()

    return {
        'total_registros': total_parseados,
        'por_hora':        dict(agg_global['por_hora']),
        'por_ruta':        dict(agg_global['por_ruta']),
        'por_categoria':   dict(agg_global['por_categoria']),
        'tiempos':         tiempos,
    }


if __name__ == '__main__':
    run_http_pipeline(num_workers=4)
