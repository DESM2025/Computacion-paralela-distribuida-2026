import time
import math
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def prediccion_numerica(semilla_matematica):
    #Simula un modelo de prediccion climatica pesado
    resultado = 0.0
    for i in range(1, 5000000):
        resultado += math.sin(i) * math.cos(semilla_matematica)
    return resultado

def ejecutar_parte_2b():
    print("\nSimulación de prediccion con procesos\n")
    tareas = [10, 20, 30, 40] 
    procesos = [1]
    tiempos = []
    
    inicio_sec = time.perf_counter()
    for t in tareas:
        prediccion_numerica(t)
    t_secuencial = time.perf_counter() - inicio_sec
    tiempos.append(t_secuencial)
    print(f"Tiempo Secuencial: {t_secuencial:.4f} s")

    for p in [2, 4]:
        inicio_par = time.perf_counter()
        with ProcessPoolExecutor(max_workers=p) as executor:
            executor.map(prediccion_numerica, tareas)
        t_paralelo = time.perf_counter() - inicio_par
        
        speedup = t_secuencial / t_paralelo
        eficiencia = speedup / p
        procesos.append(p)
        tiempos.append(t_paralelo)
        
        print(f"Procesos {p}: T={t_paralelo:.4f} s | speedup={speedup:.2f}x | eficiencia={eficiencia:.2f}")

    plt.figure(figsize=(8, 4))
    plt.bar(procesos, tiempos, color='#0A2342', width=0.6, label='Tiempo de ejecucion')
    plt.title('Rendimiento CPU-bound con Procesos')
    plt.xlabel('Cantidad de procesos (p)')
    plt.ylabel('Tiempo de ejecucion (segundos)')
    plt.xticks(procesos)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('grafico_ejmulti.png', dpi=90)
    plt.show()

if __name__ == '__main__':
    ejecutar_parte_2b()