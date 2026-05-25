#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 50000000
#define D 16

// Ejecuta la normalizacion con un numero fijo de hilos y retorna el tiempo
double ejecutar(int hilos, float *X, float *Z) {

    omp_set_num_threads(hilos);

    double suma[D]={0}, suma2[D]={0};
    long count[D]={0}, atipicos[D]={0};

    double t0 = omp_get_wtime();

    // Pasada 1: media ignorando NaN
    #pragma omp parallel for schedule(static) reduction(+: suma[:D], count[:D])
    for (long i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (!isnan(X[i*D+j])) { suma[j] += X[i*D+j]; count[j]++; }

    double media[D], var[D];
    for (int j = 0; j < D; j++)
        media[j] = count[j] > 0 ? suma[j]/count[j] : 0.0;

    // Pasada 2: varianza ignorando NaN
    #pragma omp parallel for schedule(static) reduction(+: suma2[:D])
    for (long i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (!isnan(X[i*D+j])) { double d = X[i*D+j]-media[j]; suma2[j] += d*d; }

    for (int j = 0; j < D; j++)
        var[j] = count[j] > 1 ? suma2[j]/count[j] : 1.0;

    // Pasada 3: normalizar y contar atipicos |z|>3
    #pragma omp parallel for schedule(static) reduction(+: atipicos[:D])
    for (long i = 0; i < N; i++)
        for (int j = 0; j < D; j++) {
            if (isnan(X[i*D+j])) { Z[i*D+j] = NAN; continue; }
            double z = (X[i*D+j] - media[j]) / sqrt(var[j]);
            Z[i*D+j] = (float)z;
            if (fabs(z) > 3.0) atipicos[j]++;
        }

    double t1 = omp_get_wtime();

    // Imprimir estadisticas solo con 1 hilo (referencia)
    if (hilos == 1) {
        printf("Col  Media     Varianza  Validos   Atipicos\n");
        for (int j = 0; j < D; j++)
            printf("%-4d %-9.4f %-9.4f %-9ld %ld\n", j, media[j], var[j], count[j], atipicos[j]);
        printf("\n");
    }

    return t1 - t0;
}

int main(void) {

    float *X = (float*)malloc((long)N * D * sizeof(float));
    float *Z = (float*)malloc((long)N * D * sizeof(float));
    if (!X || !Z) { fprintf(stderr, "Sin memoria\n"); return 1; }

    // Dataset sintetico: 5% NaN, resto valores aleatorios [0,10]
    srand(42);
    for (long i = 0; i < (long)N * D; i++)
        X[i] = ((float)rand()/RAND_MAX < 0.05f) ? NAN : (float)rand()/RAND_MAX * 10.0f;

    int configs[] = {1, 2, 4, 8, 16};
    double tiempos[5];

    printf("=== Benchmark Normalizacion Masiva OpenMP ===\n");
    printf("Matriz: %d x %d | NaN: 5%%\n\n", N, D);

    // Correr con cada configuracion de hilos
    for (int i = 0; i < 5; i++) {
        printf("Corriendo con %d hilo(s)...\n", configs[i]);
        tiempos[i] = ejecutar(configs[i], X, Z);
        printf("Tiempo: %.4f s\n\n", tiempos[i]);
    }

    // Tabla de resultados
    double t1 = tiempos[0]; // tiempo serial
    printf("Hilos  Tiempo(s)  Speedup  Eficiencia\n");
    printf("--------------------------------------\n");
    for (int i = 0; i < 5; i++) {
        double sp = t1 / tiempos[i];
        double ef = sp / configs[i];
        printf("%-6d %-10.4f %-8.2f %.2f\n", configs[i], tiempos[i], sp, ef);
    }

    // Guardar resultados.txt
    FILE *f = fopen("../anexos_benchmark/resultados.txt", "w");
    if (f) {
        fprintf(f, "Benchmark Normalizacion Masiva OpenMP\n");
        fprintf(f, "Matriz: %d x %d | NaN: 5%%\n\n", N, D);
        fprintf(f, "Hilos  Tiempo(s)  Speedup  Eficiencia\n");
        fprintf(f, "--------------------------------------\n");
        for (int i = 0; i < 5; i++) {
            double sp = t1 / tiempos[i];
            double ef = sp / configs[i];
            fprintf(f, "%-6d %-10.4f %-8.2f %.2f\n", configs[i], tiempos[i], sp, ef);
        }
        fclose(f);
        printf("\nResultados guardados en anexos_benchmark/resultados.txt\n");
    }

    // Guardar resultados1.csv
    FILE *csv = fopen("../anexos_benchmark/resultados1.csv", "w");
    if (csv) {
        fprintf(csv, "hilos,tiempo,speedup,eficiencia\n");
        for (int i = 0; i < 5; i++) {
            double sp = t1 / tiempos[i];
            double ef = sp / configs[i];
            fprintf(csv, "%d,%.4f,%.4f,%.4f\n", configs[i], tiempos[i], sp, ef);
        }
        fclose(csv);
        printf("Resultados guardados en anexos_benchmark/resultados1.csv\n");
    }

    free(X); free(Z);
    return 0;
}
