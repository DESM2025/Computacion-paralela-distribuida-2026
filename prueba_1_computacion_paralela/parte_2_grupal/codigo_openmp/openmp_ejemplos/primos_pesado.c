#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Determina si un numero entero es primo probando divisores impares
   hasta la raiz cuadrada del numero. */
static int es_primo(int n) {
    if (n < 2) {
        return 0;
    }

    if (n == 2) {
        return 1;
    }

    if (n % 2 == 0) {
        return 0;
    }

    int limite = (int)sqrt((double)n);

    for (int divisor = 3; divisor <= limite; divisor += 2) {
        if (n % divisor == 0) {
            return 0;
        }
    }

    return 1;
}

/* Cuenta numeros primos primero en serial y luego en paralelo para comparar
   los tiempos de ejecucion de ambas versiones. */
int main(int argc, char *argv[]) {
    int limite = 25000000;

    if (argc > 1) {
        limite = atoi(argv[1]);
    }

    if (limite < 2) {
        fprintf(stderr, "El limite debe ser mayor o igual a 2.\n");
        return 1;
    }

    printf("Conteo de numeros primos hasta %d\n", limite);
    printf("Procesadores disponibles: %d\n", omp_get_num_procs());
    printf("Maximo de hilos OpenMP: %d\n\n", omp_get_max_threads());

    double inicio_serial = omp_get_wtime();
    int primos_serial = 0;

    for (int numero = 2; numero <= limite; numero++) {
        primos_serial += es_primo(numero);
    }

    double fin_serial = omp_get_wtime();

    double inicio_paralelo = omp_get_wtime();
    int primos_paralelo = 0;

    /* Paraleliza el ciclo que prueba cada numero. schedule(dynamic, 1000)
       reparte bloques de 1000 iteraciones a medida que los hilos terminan,
       y reduction suma los conteos parciales de primos. */
    #pragma omp parallel for schedule(dynamic, 1000) reduction(+:primos_paralelo)
    for (int numero = 2; numero <= limite; numero++) {
        primos_paralelo += es_primo(numero);
    }

    double fin_paralelo = omp_get_wtime();

    double tiempo_serial = fin_serial - inicio_serial;
    double tiempo_paralelo = fin_paralelo - inicio_paralelo;

    printf("Primos encontrados en serial: %d\n", primos_serial);
    printf("Tiempo serial: %.6f segundos\n\n", tiempo_serial);

    printf("Primos encontrados en paralelo: %d\n", primos_paralelo);
    printf("Tiempo paralelo: %.6f segundos\n\n", tiempo_paralelo);

    if (primos_serial != primos_paralelo) {
        printf("Advertencia: los resultados no coinciden.\n");
    }

    if (tiempo_paralelo > 0.0) {
        printf("Aceleracion: %.2fx\n", tiempo_serial / tiempo_paralelo);
    }

    return 0;
}
