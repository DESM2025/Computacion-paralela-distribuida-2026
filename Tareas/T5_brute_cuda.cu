/*
 * T5_brute_cuda.cu — Fuerza bruta MD5 con CUDA.
 *
 * Compilar : nvcc -O2 -o T5_brute_cuda.exe T5_brute_cuda.cu
 * Ejecutar : T5_brute_cuda.exe
 *
 * Contraseñas objetivo:
 *   longitud 4 → "cake"   MD5: a1b8585122e1ad60623d6a74d3eb3b6a
 *   longitud 5 → "hello"  MD5: 5d41402abc4b2a76b9719d911017c592
 *   longitud 6 → "monkey" MD5: d0763edaa9d9bd2a9516280e9044d885
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Tiempo de pared en milisegundos usando timespec_get (C11).
 * Mismo método que OpenCL → comparación de setup justa. */
static double wall_ms()
{
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1.0e6;
}

#include "T5_md5.h"

static const uint8_t TARGET_4[16] = {
    0xa1,0xb8,0x58,0x51, 0x22,0xe1,0xad,0x60,
    0x62,0x3d,0x6a,0x74, 0xd3,0xeb,0x3b,0x6a
};
static const uint8_t TARGET_5[16] = {
    0x5d,0x41,0x40,0x2a, 0xbc,0x4b,0x2a,0x76,
    0xb9,0x71,0x9d,0x91, 0x10,0x17,0xc5,0x92
};
static const uint8_t TARGET_6[16] = {
    0xd0,0x76,0x3e,0xda, 0xa9,0xd9,0xbd,0x2a,
    0x95,0x16,0x28,0x0e, 0x90,0x44,0xd8,0x85
};

#define BLOCK_SIZE 256

/* Macro para verificar errores CUDA */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA ERROR %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
        }                                                                       \
    } while(0)

/* ---------------------------------------------------------------------
 * Kernel GPU.
 * total y tid usan unsigned int (26^6 = 308M < 2^32).
 * Evitar unsigned long long como parámetro de kernel mejora la estabilidad.
 * --------------------------------------------------------------------- */
__global__ void md5_brute_kernel(int          pass_len,
                                  unsigned int total,
                                  const uint8_t* __restrict__ target,
                                  int*         found,
                                  char*        found_pass)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    /* Convertir tid a contraseña base-26 (a-z) */
    char pass[8];
    unsigned int n = tid;
    for (int i = pass_len - 1; i >= 0; i--) {
        pass[i] = 'a' + (int)(n % 26u);
        n /= 26u;
    }

    /* Calcular MD5 y comparar */
    uint32_t hash[4];
    md5_compute(pass, pass_len, hash);

    const uint8_t* h = (const uint8_t*)hash;
    int match = 1;
    for (int i = 0; i < 16; i++) {
        if (h[i] != target[i]) { match = 0; break; }
    }
    if (match) {
        atomicExch(found, 1);
        for (int i = 0; i < pass_len; i++) found_pass[i] = pass[i];
    }
}

static void run_length(int pass_len, const uint8_t* target_h, FILE* csv)
{
    unsigned int total = 1;
    for (int i = 0; i < pass_len; i++) total *= 26u;

    cudaEvent_t k0, k1;
    CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));

    /* Setup: alloc + H2D — medido con timespec_get (CPU wall-clock, igual que OpenCL) */
    double t_s0 = wall_ms();

    uint8_t* d_target; int* d_found; char* d_pass;
    CUDA_CHECK(cudaMalloc(&d_target, 16));
    CUDA_CHECK(cudaMalloc(&d_found,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pass,   8));
    CUDA_CHECK(cudaMemcpy(d_target, target_h, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pass,  0, 8));

    double setup_ms = wall_ms() - t_s0;

    /* Kernel */
    unsigned int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaEventRecord(k0));
    md5_brute_kernel<<<grid, BLOCK_SIZE>>>(pass_len, total, d_target, d_found, d_pass);
    CUDA_CHECK(cudaEventRecord(k1));
    CUDA_CHECK(cudaEventSynchronize(k1));
    CUDA_CHECK(cudaGetLastError()); /* captura errores del kernel */
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, k0, k1));

    /* Resultados */
    int  found_h = 0;
    char pass_h[8] = {0};
    CUDA_CHECK(cudaMemcpy(&found_h, d_found, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pass_h,   d_pass,  8,           cudaMemcpyDeviceToHost));

    cudaFree(d_target); cudaFree(d_found); cudaFree(d_pass);
    cudaEventDestroy(k0); cudaEventDestroy(k1);

    double hashrate = (total / 1.0e6) / ((double)kernel_ms / 1000.0);

    printf("  len=%-2d  hilos=%-12u  setup=%8.3f ms  kernel=%8.3f ms  %10.1f MH/s",
           pass_len, total, setup_ms, (double)kernel_ms, hashrate);
    if (found_h) printf("  >> ENCONTRADA: \"%.*s\"", pass_len, pass_h);
    printf("\n");

    fprintf(csv, "CUDA,%d,%u,%.4f,%.4f,%.2f\n",
            pass_len, total, setup_ms, (double)kernel_ms, hashrate);
}

int main()
{
    cudaEvent_t i0, i1;
    CUDA_CHECK(cudaEventCreate(&i0)); CUDA_CHECK(cudaEventCreate(&i1));
    CUDA_CHECK(cudaEventRecord(i0));
    CUDA_CHECK(cudaFree(0)); /* fuerza inicialización del runtime */
    CUDA_CHECK(cudaEventRecord(i1));
    CUDA_CHECK(cudaEventSynchronize(i1));
    float init_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&init_ms, i0, i1));
    cudaEventDestroy(i0); cudaEventDestroy(i1);

    /* "w" sobreescribe el archivo en cada ejecución */
    FILE* csv = fopen("T5_resultados_cuda.csv", "w");
    if (!csv) { fprintf(stderr, "ERROR: no se puede abrir T5_resultados_cuda.csv\n"); return 1; }
    fprintf(csv, "Tecnologia,Longitud_Pass,Hilos_Totales,Tiempo_Setup_ms,Tiempo_Kernel_ms,Hashrate_MHs\n");

    fprintf(csv, "CUDA,0,0,%.4f,0.0000,0.00\n", (double)init_ms);


    printf("=== CUDA MD5 Brute-Force Benchmark ===\n");
    printf("  Init runtime CUDA: %.2f ms\n\n", (double)init_ms);

    const uint8_t* targets[3] = {TARGET_4, TARGET_5, TARGET_6};
    int            lengths[3] = {4, 5, 6};
    for (int i = 0; i < 3; i++)
        run_length(lengths[i], targets[i], csv);

    fclose(csv);
    printf("\nResultados guardados en T5_resultados_cuda.csv\n");
    return 0;
}
