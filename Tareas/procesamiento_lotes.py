import time
import math
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def simular_estres_termico(id_aleacion):
    # Simula el desgaste termico de una aleacion independiente
    nivel_fatiga = id_aleacion * 0.5
    for ciclo in range(1, 4000000):
        nivel_fatiga += math.sin(ciclo * 0.01) * math.sqrt(ciclo) * 0.001
    return nivel_fatiga

def ejecutar_ejercicio_3():
    print("Simulacion de procesamiento de lotes en metalurgia\n")
    lotes = list(range(8)) 
    procesos = [1]
    tiempos = []
    
    inicio = time.perf_counter()
    for lote in lotes:
        simular_estres_termico(lote)
    t_sec = time.perf_counter() - inicio
    tiempos.append(t_sec)
    print(f"Tiempo Secuencial: {t_sec:.4f} s")

    for p in [2, 4]:
        inicio = time.perf_counter()
        with ProcessPoolExecutor(max_workers=p) as exec:
            list(exec.map(simular_estres_termico, lotes))
        t_par = time.perf_counter() - inicio
        
        s = t_sec / t_par
        e = s / p
        procesos.append(p)
        tiempos.append(t_par)
        print(f"Procesos p={p}: T={t_par:.4f}s | speedup={s:.2f}x | eficiencia={e:.2f}")

    plt.figure(figsize=(8, 4))
    plt.bar(procesos, tiempos, color='#0A2342', width=0.6, label='Tiempo de ejecucion')
    plt.title('Rendimiento en Procesamiento por Lotes')
    plt.xlabel('Cantidad de procesos (p)')
    plt.ylabel('Tiempo de ejecucion (segundos)')
    plt.xticks(procesos)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('grafico_procesamiento_lotes.png', dpi=90)
    plt.show()

if __name__ == '__main__':
    ejecutar_ejercicio_3()