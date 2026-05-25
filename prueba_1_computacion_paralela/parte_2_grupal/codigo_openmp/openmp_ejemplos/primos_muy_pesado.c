#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Determina si un numero entero es primo sin usar sqrt, evitando una llamada
   extra a la biblioteca matematica en cada evaluacion. */
static int es_primo(int numero) {
    if (numero < 2) {
        return 0;
    }

    if (numero == 2) {
        return 1;
    }

    if (numero % 2 == 0) {
        return 0;
    }

    for (int divisor = 3; divisor <= numero / divisor; divisor += 2) {
        if (numero % divisor == 0) {
            return 0;
        }
    }

    return 1;
}

/* Cuenta todos los numeros primos hasta el limite usando un ciclo serial,
   es decir, ejecutado por un solo hilo. */
static int contar_primos_serial(int limite) {
    int total = 0;

    for (int numero = 2; numero <= limite; numero++) {
        total += es_primo(numero);
    }

    return total;
}

/* Cuenta todos los numeros primos hasta el limite usando OpenMP con una
   cantidad configurable de hilos. */
static int contar_primos_paralelo(int limite, int hilos) {
    int total = 0;

    /* Paraleliza el conteo usando exactamente la cantidad de hilos indicada.
       schedule(dynamic, 1000) ayuda a equilibrar el trabajo porque algunos
       numeros tardan mas que otros en evaluarse como primos. */
    #pragma omp parallel for num_threads(hilos) schedule(dynamic, 1000) reduction(+:total)
    for (int numero = 2; numero <= limite; numero++) {
        total += es_primo(numero);
    }

    return total;
}

/* Ejecuta la prueba completa: mide la version serial y luego compara la
   version paralela con distintas cantidades de hilos. */
int main(int argc, char *argv[]) {
    int limite = 50000000;

    if (argc > 1) {
        limite = atoi(argv[1]);
    }

    if (limite < 2) {
        fprintf(stderr, "El limite debe ser mayor o igual a 2.\n");
        return 1;
    }

    int max_hilos = omp_get_max_threads();
    int pruebas_hilos[] = {1, 2, 4, 8, 12, 16, 18, 20, 32};
    int cantidad_pruebas = (int)(sizeof(pruebas_hilos) / sizeof(pruebas_hilos[0]));

    printf("Comparacion pesada con OpenMP: conteo de numeros primos\n");
    printf("Limite: %d\n", limite);
    printf("Procesadores disponibles: %d\n", omp_get_num_procs());
    printf("Maximo de hilos OpenMP: %d\n\n", max_hilos);

    double inicio_serial = omp_get_wtime();
    int primos_serial = contar_primos_serial(limite);
    double fin_serial = omp_get_wtime();
    double tiempo_serial = fin_serial - inicio_serial;

    printf("Resultado serial: %d primos\n", primos_serial);
    printf("Tiempo serial: %.6f segundos\n\n", tiempo_serial);

    printf("%8s %14s %14s %14s\n", "Hilos", "Primos", "Tiempo", "Aceleracion");
    printf("%8s %14s %14s %14s\n", "-----", "------", "------", "-----------");

    for (int i = 0; i < cantidad_pruebas; i++) {
        int hilos = pruebas_hilos[i];

        if (hilos > max_hilos) {
            continue;
        }

        double inicio_paralelo = omp_get_wtime();
        int primos_paralelo = contar_primos_paralelo(limite, hilos);
        double fin_paralelo = omp_get_wtime();
        double tiempo_paralelo = fin_paralelo - inicio_paralelo;
        double aceleracion = tiempo_serial / tiempo_paralelo;

        printf("%8d %14d %11.6f s %12.2fx\n",
               hilos,
               primos_paralelo,
               tiempo_paralelo,
               aceleracion);

        if (primos_paralelo != primos_serial) {
            printf("Advertencia: el resultado con %d hilos no coincide.\n", hilos);
        }
    }

    return 0;
}
