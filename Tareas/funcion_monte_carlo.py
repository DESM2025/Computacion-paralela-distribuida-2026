import random

def simular_monte_carlo(dardos):
    dentro_circulo = 0
    for _ in range(dardos):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            dentro_circulo += 1
    return dentro_circulo

def procesar_lote_simulaciones(lote):
    return [simular_monte_carlo(d) for d in lote]