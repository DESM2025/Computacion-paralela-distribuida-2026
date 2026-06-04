/*
 * T5_kernel_opencl.cl — Kernel OpenCL para fuerza bruta MD5.
 *
 * NOTA: el host (T5_brute_opencl.cpp) prepende T5_md5.h a este archivo
 * antes de compilarlo, por lo que md5_compute() está disponible aquí.
 *
 * Cada work-item convierte su global ID a una contraseña en base-26 (a-z),
 * calcula su MD5 y lo compara con el objetivo.
 */

__kernel void md5_brute_kernel(int               pass_len,
                                long              total,
                                __global const uchar* target,
                                __global int*     found,
                                __global char*    found_pass)
{
    long tid = (long)get_global_id(0);
    if (tid >= total) return;

    /* Convertir tid a contraseña en base-26 */
    char pass[8];
    long n = tid;
    for (int i = pass_len - 1; i >= 0; i--) {
        pass[i] = 'a' + (int)(n % 26);
        n /= 26;
    }

    /* Calcular MD5 */
    md5_u32 hash[4];
    md5_compute(pass, pass_len, hash);

    /* Comparar byte a byte con el objetivo */
    const uchar* h = (const uchar*)hash;
    int match = 1;
    for (int i = 0; i < 16; i++) {
        if (h[i] != target[i]) { match = 0; break; }
    }

    if (match) {
        atomic_xchg(found, 1);
        for (int i = 0; i < pass_len; i++) found_pass[i] = pass[i];
    }
}
