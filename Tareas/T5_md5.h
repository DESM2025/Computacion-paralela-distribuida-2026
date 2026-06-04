#ifndef T5_MD5_H
#define T5_MD5_H

/*
 * T5_md5.h — Implementación MD5 para kernels GPU.
 * Compatible con CUDA (.cu) y OpenCL (.cl) via compilación condicional.
 * Restricción: longitud de entrada <= 55 bytes (un solo bloque de 512 bits).
 *
 * Diseño: empaqueta los bytes del input directamente en M[] sin buffer
 * intermedio, para minimizar el uso de registros y evitar spilling en GPU.
 */

#ifdef __OPENCL_VERSION__
    typedef uint  md5_u32;
    #define GPU_FUNC
#else
    #include <stdint.h>
    typedef uint32_t md5_u32;
    #ifdef __CUDACC__
        #define GPU_FUNC __device__ __forceinline__
    #else
        #define GPU_FUNC
    #endif
#endif

#define MD5_F(b,c,d) (((b)&(c))|(~(b)&(d)))
#define MD5_G(b,c,d) (((b)&(d))|((c)&~(d)))
#define MD5_H(b,c,d) ((b)^(c)^(d))
#define MD5_I(b,c,d) ((c)^((b)|~(d)))
#define ROTL32(x,n)  (((x)<<(n))|((x)>>(32-(n))))

#define FF(a,b,c,d,x,s,t) a = b + ROTL32(a + MD5_F(b,c,d) + (x) + (t), s)
#define GG(a,b,c,d,x,s,t) a = b + ROTL32(a + MD5_G(b,c,d) + (x) + (t), s)
#define HH(a,b,c,d,x,s,t) a = b + ROTL32(a + MD5_H(b,c,d) + (x) + (t), s)
#define II(a,b,c,d,x,s,t) a = b + ROTL32(a + MD5_I(b,c,d) + (x) + (t), s)

GPU_FUNC void md5_compute(const char* input, int len, md5_u32 out[4])
{
    int i;

    /* Construir el bloque de 16 palabras directamente (sin buffer de bytes intermedio).
     * Los accesos a M[] en las rondas usan índices constantes → el compilador los
     * mantiene en registros en lugar de derramarlos a memoria local. */
    md5_u32 M[16];
    for (i = 0; i < 16; i++) M[i] = 0u;

    /* Empaquetar bytes del input en little-endian */
    for (i = 0; i < len; i++)
        M[i >> 2] |= ((md5_u32)((unsigned char)input[i])) << ((i & 3) << 3);

    /* Padding: byte 0x80 justo después del mensaje */
    M[len >> 2] |= (md5_u32)0x80u << ((len & 3) << 3);

    /* Longitud en bits (little-endian 64 bits; cabe en 32 bits para passwords cortas) */
    M[14] = (md5_u32)(len << 3);

    md5_u32 a = 0x67452301u, b = 0xEFCDAB89u,
            c = 0x98BADCFEu, d = 0x10325476u;

    /* Ronda 1 */
    FF(a,b,c,d, M[ 0], 7,0xD76AA478u); FF(d,a,b,c, M[ 1],12,0xE8C7B756u);
    FF(c,d,a,b, M[ 2],17,0x242070DBu); FF(b,c,d,a, M[ 3],22,0xC1BDCEEEu);
    FF(a,b,c,d, M[ 4], 7,0xF57C0FAFu); FF(d,a,b,c, M[ 5],12,0x4787C62Au);
    FF(c,d,a,b, M[ 6],17,0xA8304613u); FF(b,c,d,a, M[ 7],22,0xFD469501u);
    FF(a,b,c,d, M[ 8], 7,0x698098D8u); FF(d,a,b,c, M[ 9],12,0x8B44F7AFu);
    FF(c,d,a,b, M[10],17,0xFFFF5BB1u); FF(b,c,d,a, M[11],22,0x895CD7BEu);
    FF(a,b,c,d, M[12], 7,0x6B901122u); FF(d,a,b,c, M[13],12,0xFD987193u);
    FF(c,d,a,b, M[14],17,0xA679438Eu); FF(b,c,d,a, M[15],22,0x49B40821u);

    /* Ronda 2 */
    GG(a,b,c,d, M[ 1], 5,0xF61E2562u); GG(d,a,b,c, M[ 6], 9,0xC040B340u);
    GG(c,d,a,b, M[11],14,0x265E5A51u); GG(b,c,d,a, M[ 0],20,0xE9B6C7AAu);
    GG(a,b,c,d, M[ 5], 5,0xD62F105Du); GG(d,a,b,c, M[10], 9,0x02441453u);
    GG(c,d,a,b, M[15],14,0xD8A1E681u); GG(b,c,d,a, M[ 4],20,0xE7D3FBC8u);
    GG(a,b,c,d, M[ 9], 5,0x21E1CDE6u); GG(d,a,b,c, M[14], 9,0xC33707D6u);
    GG(c,d,a,b, M[ 3],14,0xF4D50D87u); GG(b,c,d,a, M[ 8],20,0x455A14EDu);
    GG(a,b,c,d, M[13], 5,0xA9E3E905u); GG(d,a,b,c, M[ 2], 9,0xFCEFA3F8u);
    GG(c,d,a,b, M[ 7],14,0x676F02D9u); GG(b,c,d,a, M[12],20,0x8D2A4C8Au);

    /* Ronda 3 */
    HH(a,b,c,d, M[ 5], 4,0xFFFA3942u); HH(d,a,b,c, M[ 8],11,0x8771F681u);
    HH(c,d,a,b, M[11],16,0x6D9D6122u); HH(b,c,d,a, M[14],23,0xFDE5380Cu);
    HH(a,b,c,d, M[ 1], 4,0xA4BEEA44u); HH(d,a,b,c, M[ 4],11,0x4BDECFA9u);
    HH(c,d,a,b, M[ 7],16,0xF6BB4B60u); HH(b,c,d,a, M[10],23,0xBEBFBC70u);
    HH(a,b,c,d, M[13], 4,0x289B7EC6u); HH(d,a,b,c, M[ 0],11,0xEAA127FAu);
    HH(c,d,a,b, M[ 3],16,0xD4EF3085u); HH(b,c,d,a, M[ 6],23,0x04881D05u);
    HH(a,b,c,d, M[ 9], 4,0xD9D4D039u); HH(d,a,b,c, M[12],11,0xE6DB99E5u);
    HH(c,d,a,b, M[15],16,0x1FA27CF8u); HH(b,c,d,a, M[ 2],23,0xC4AC5665u);

    /* Ronda 4 */
    II(a,b,c,d, M[ 0], 6,0xF4292244u); II(d,a,b,c, M[ 7],10,0x432AFF97u);
    II(c,d,a,b, M[14],15,0xAB9423A7u); II(b,c,d,a, M[ 5],21,0xFC93A039u);
    II(a,b,c,d, M[12], 6,0x655B59C3u); II(d,a,b,c, M[ 3],10,0x8F0CCC92u);
    II(c,d,a,b, M[10],15,0xFFEFF47Du); II(b,c,d,a, M[ 1],21,0x85845DD1u);
    II(a,b,c,d, M[ 8], 6,0x6FA87E4Fu); II(d,a,b,c, M[15],10,0xFE2CE6E0u);
    II(c,d,a,b, M[ 6],15,0xA3014314u); II(b,c,d,a, M[13],21,0x4E0811A1u);
    II(a,b,c,d, M[ 4], 6,0xF7537E82u); II(d,a,b,c, M[11],10,0xBD3AF235u);
    II(c,d,a,b, M[ 2],15,0x2AD7D2BBu); II(b,c,d,a, M[ 9],21,0xEB86D391u);

    out[0] = a + 0x67452301u;
    out[1] = b + 0xEFCDAB89u;
    out[2] = c + 0x98BADCFEu;
    out[3] = d + 0x10325476u;
}

#endif /* T5_MD5_H */
