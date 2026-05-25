#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Ejecuta una comparacion simple entre una suma serial y una suma paralela
   usando OpenMP sobre un arreglo grande de valores double. */
int main(void) {
    const int cantidad = 50000000;
    double *valores = malloc((size_t)cantidad * sizeof(double));

    if (valores == NULL) {
        fprintf(stderr, "No se pudo reservar memoria.\n");
        return 1;
    }

    for (int i = 0; i < cantidad; i++) {
        valores[i] = 1.0;
    }

    printf("Ejemplo OpenMP en C\n");
    printf("Cantidad de valores: %d\n", cantidad);
    printf("Procesadores disponibles: %d\n", omp_get_num_procs());
    printf("Maximo de hilos OpenMP: %d\n\n", omp_get_max_threads());

    /* Crea una region paralela: OpenMP lanza varios hilos y cada uno
       ejecuta este mismo bloque de codigo. */
    #pragma omp parallel
    {
        int hilo = omp_get_thread_num();
        int total_hilos = omp_get_num_threads();

        /* Evita que varios hilos escriban en pantalla al mismo tiempo. */
        #pragma omp critical
        {
            printf("Hola desde el hilo %d de %d\n", hilo, total_hilos);
        }
    }

    double inicio_serial = omp_get_wtime();
    double suma_serial = 0.0;

    for (int i = 0; i < cantidad; i++) {
        suma_serial += valores[i];
    }

    double fin_serial = omp_get_wtime();

    double inicio_paralelo = omp_get_wtime();
    double suma_paralela = 0.0;

    /* Reparte las iteraciones del for entre varios hilos. La clausula
       reduction combina las sumas parciales de cada hilo en suma_paralela. */
    #pragma omp parallel for reduction(+:suma_paralela)
    for (int i = 0; i < cantidad; i++) {
        suma_paralela += valores[i];
    }

    double fin_paralelo = omp_get_wtime();

    double tiempo_serial = fin_serial - inicio_serial;
    double tiempo_paralelo = fin_paralelo - inicio_paralelo;

    printf("\nSuma serial: %.0f\n", suma_serial);
    printf("Tiempo serial: %.6f segundos\n", tiempo_serial);
    printf("\nSuma paralela: %.0f\n", suma_paralela);
    printf("Tiempo paralelo: %.6f segundos\n", tiempo_paralelo);

    if (tiempo_paralelo > 0.0) {
        printf("\nAceleracion: %.2fx\n", tiempo_serial / tiempo_paralelo);
    }

    free(valores);
    return 0;
}
