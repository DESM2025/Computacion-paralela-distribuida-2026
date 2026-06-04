/*
 * T5_brute_opencl.cpp — Fuerza bruta MD5 con OpenCL.
 *
 * Compilar Linux : g++ -O2 -std=c++14 -o T5_brute_opencl T5_brute_opencl.cpp -lOpenCL
 * Compilar Windows (MSVC): cl /O2 /std:c++14 T5_brute_opencl.cpp OpenCL.lib
 *   (asegurarse de tener el include path del SDK de CUDA/OpenCL en INCLUDE y LIB)
 *
 * Ejecutar: ./T5_brute_opencl  (debe estar en el mismo directorio que T5_md5.h y T5_kernel_opencl.cl)
 *
 * Contraseñas objetivo hardcodeadas (mismas que la versión CUDA):
 *   longitud 4 → "cake"   MD5: a1b8585122e1ad60623d6a74d3eb3b6a
 *   longitud 5 → "hello"  MD5: 5d41402abc4b2a76b9719d911017c592
 *   longitud 6 → "monkey" MD5: d0763edaa9d9bd2a9516280e9044d885
 */

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* Tiempo de pared en milisegundos usando timespec_get (C11).
 * Mismo método que CUDA → comparación de setup justa. */
static double wall_ms()
{
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1.0e6;
}

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

#define LOCAL_SIZE 256

/* Estado global de OpenCL (inicializado una sola vez) */
static cl_context       g_ctx;
static cl_device_id     g_device;
static cl_command_queue g_queue;
static cl_program       g_program;
static cl_kernel        g_kernel;

/* Lee un archivo de texto y devuelve un buffer malloc'd. */
static char* read_file(const char* path, size_t* out_len)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: no se puede leer '%s'\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    size_t len = (size_t)ftell(f);
    rewind(f);
    char* buf = (char*)malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);
    *out_len = len;
    return buf;
}

/* ---------------------------------------------------------------------
 * Inicialización única de OpenCL:
 *   plataforma → dispositivo GPU → contexto → cola → compilar kernel.
 * Retorna el tiempo en ms de todo este proceso.
 * --------------------------------------------------------------------- */
static double opencl_init()
{
    double t0 = wall_ms();

    cl_int err;

    /* Seleccionar primera plataforma y primer GPU */
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: no se encontró GPU OpenCL (err=%d)\n", err);
        exit(1);
    }

    /* Imprimir nombre del dispositivo */
    char dev_name[256];
    clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("  Dispositivo: %s\n", dev_name);

    /* Contexto y cola de comandos con profiling habilitado */
    g_ctx   = clCreateContext(NULL, 1, &g_device, NULL, NULL, &err);
    /* clCreateCommandQueue es deprecada en OpenCL 2.0+ pero funciona en 1.x y 2.x */
    g_queue = clCreateCommandQueue(g_ctx, g_device, CL_QUEUE_PROFILING_ENABLE, &err);

    /* Construir fuente: T5_md5.h + T5_kernel_opencl.cl */
    size_t md5_len, kern_len;
    char* md5_src  = read_file("T5_md5.h", &md5_len);
    char* kern_src = read_file("T5_kernel_opencl.cl", &kern_len);
    if (!md5_src || !kern_src) exit(1);

    const char* sources[2] = { md5_src,  kern_src  };
    size_t      sizes[2]   = { md5_len,  kern_len  };

    g_program = clCreateProgramWithSource(g_ctx, 2, sources, sizes, &err);
    err = clBuildProgram(g_program, 1, &g_device, "-cl-std=CL1.2 -cl-mad-enable", NULL, NULL);

    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char* log = (char*)malloc(log_sz);
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        fprintf(stderr, "ERROR compilando kernel OpenCL:\n%s\n", log);
        free(log);
        exit(1);
    }

    g_kernel = clCreateKernel(g_program, "md5_brute_kernel", &err);
    free(md5_src); free(kern_src);

    return wall_ms() - t0;
}

/* ---------------------------------------------------------------------
 * Ejecuta un benchmark para una longitud de contraseña dada.
 * Mide: setup (alloc + H2D + setargs) y tiempo de kernel (event profiling).
 * --------------------------------------------------------------------- */
static void run_length(int pass_len, const uint8_t* target_h, FILE* csv)
{
    unsigned long long total = 1;
    for (int i = 0; i < pass_len; i++) total *= 26;

    cl_int err;

    /* --- Tiempo de setup: crear buffers + setear argumentos --- */
    double t0 = wall_ms();

    cl_mem d_target = clCreateBuffer(g_ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16, (void*)target_h, &err);

    int  found_h = 0;
    char pass_h[8] = {0};
    cl_mem d_found = clCreateBuffer(g_ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &found_h, &err);
    cl_mem d_pass = clCreateBuffer(g_ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 8, pass_h, &err);

    cl_long total_cl = (cl_long)total;
    clSetKernelArg(g_kernel, 0, sizeof(int),     &pass_len);
    clSetKernelArg(g_kernel, 1, sizeof(cl_long), &total_cl);
    clSetKernelArg(g_kernel, 2, sizeof(cl_mem),  &d_target);
    clSetKernelArg(g_kernel, 3, sizeof(cl_mem),  &d_found);
    clSetKernelArg(g_kernel, 4, sizeof(cl_mem),  &d_pass);

    double setup_ms = wall_ms() - t0;

    /* --- Lanzar kernel y medir con event profiling --- */
    /* El global size debe ser múltiplo del local size */
    size_t global_size = (size_t)(((total + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE);
    size_t local_size  = LOCAL_SIZE;

    cl_event event;
    err = clEnqueueNDRangeKernel(g_queue, g_kernel, 1,
                                  NULL, &global_size, &local_size,
                                  0, NULL, &event);
    clWaitForEvents(1, &event);

    cl_ulong t_start = 0, t_end = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &t_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &t_end,   NULL);
    double kernel_ms = (double)(t_end - t_start) * 1.0e-6;

    /* --- Leer resultados --- */
    clEnqueueReadBuffer(g_queue, d_found, CL_TRUE, 0, sizeof(int), &found_h, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, d_pass,  CL_TRUE, 0, 8,           pass_h,  0, NULL, NULL);

    clReleaseEvent(event);
    clReleaseMemObject(d_target);
    clReleaseMemObject(d_found);
    clReleaseMemObject(d_pass);

    double hashrate = (total / 1.0e6) / (kernel_ms / 1000.0);

    printf("  len=%-2d  hilos=%-12llu  setup=%8.2f ms  kernel=%8.2f ms  %8.1f MH/s",
           pass_len, (unsigned long long)total, setup_ms, kernel_ms, hashrate);
    if (found_h) printf("  >> ENCONTRADA: \"%.*s\"", pass_len, pass_h);
    printf("\n");

    fprintf(csv, "OpenCL,%d,%llu,%.4f,%.4f,%.2f\n",
            pass_len, (unsigned long long)total, setup_ms, kernel_ms, hashrate);
}

int main()
{
    /* "w" sobreescribe el archivo en cada ejecución */
    FILE* csv = fopen("T5_resultados_opencl.csv", "w");
    if (!csv) { fprintf(stderr, "ERROR: no se puede abrir T5_resultados_opencl.csv\n"); return 1; }
    fprintf(csv, "Tecnologia,Longitud_Pass,Hilos_Totales,Tiempo_Setup_ms,Tiempo_Kernel_ms,Hashrate_MHs\n");

    printf("=== OpenCL MD5 Brute-Force Benchmark ===\n");
    double init_ms = opencl_init();
    printf("  Init OpenCL (plataforma + contexto + compilación): %.2f ms\n\n", init_ms);

    /* Fila de inicialización (longitud 0 = sin kernel) */
    fprintf(csv, "OpenCL,0,0,%.4f,0.0000,0.00\n", init_ms);

    const uint8_t* targets[3] = {TARGET_4, TARGET_5, TARGET_6};
    int            lengths[3] = {4, 5, 6};

    for (int i = 0; i < 3; i++)
        run_length(lengths[i], targets[i], csv);

    /* Liberar recursos OpenCL */
    clReleaseKernel(g_kernel);
    clReleaseProgram(g_program);
    clReleaseCommandQueue(g_queue);
    clReleaseContext(g_ctx);

    fclose(csv);
    printf("\nResultados guardados en T5_resultados_opencl.csv\n");
    return 0;
}
