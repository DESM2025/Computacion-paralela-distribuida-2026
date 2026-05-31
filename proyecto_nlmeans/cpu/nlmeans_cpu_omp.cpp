#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

void agregarRuido(std::vector<float>& imagen, int n, float sigma) {
    srand(42);
    for (int i = 0; i < n; i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float ruido = sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        imagen[i] = fmaxf(0.0f, fminf(255.0f, imagen[i] + ruido));
    }
}

void nlMeansCPU_OMP(const std::vector<float>& entrada,
                    std::vector<float>& salida,
                    int ancho, int alto,
                    int tamPatch, int tamVentana, float h) {
    int mitadPatch   = tamPatch / 2;
    int mitadVentana = tamVentana / 2;
    float h2 = h * h;

    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(16)
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            float sumaPesos = 0.0f;
            float sumaValor = 0.0f;
            for (int vy = -mitadVentana; vy <= mitadVentana; vy++) {
                for (int vx = -mitadVentana; vx <= mitadVentana; vx++) {
                    int ny = y + vy;
                    int nx = x + vx;
                    if (ny < 0 || ny >= alto || nx < 0 || nx >= ancho) continue;
                    float dist2 = 0.0f;
                    for (int py = -mitadPatch; py <= mitadPatch; py++) {
                        for (int px = -mitadPatch; px <= mitadPatch; px++) {
                            int ay = std::max(0, std::min(alto  - 1, y  + py));
                            int ax = std::max(0, std::min(ancho - 1, x  + px));
                            int by = std::max(0, std::min(alto  - 1, ny + py));
                            int bx = std::max(0, std::min(ancho - 1, nx + px));
                            float diff = entrada[ay * ancho + ax] - entrada[by * ancho + bx];
                            dist2 += diff * diff;
                        }
                    }
                    float peso = expf(-dist2 / (49.0f * h2));
                    sumaPesos += peso;
                    sumaValor += peso * entrada[ny * ancho + nx];
                }
            }
            salida[y * ancho + x] = sumaValor / sumaPesos;
        }
    }
}

int main(int argc, char* argv[]) {
    const char* rutaImagen = argc > 1 ? argv[1] : "../imagenes/Original_lena512.jpg";

    int ancho, alto, canalesOrig;
    unsigned char* datosOrig = stbi_load(rutaImagen, &ancho, &alto, &canalesOrig, 0);
    if (!datosOrig) { std::cerr << "Error cargando imagen." << std::endl; return 1; }

    int n = ancho * alto;
    int canales = (canalesOrig >= 3) ? 3 : 1;

    std::cout << "=== NL-Means CPU OpenMP ===" << std::endl;
    std::cout << "Imagen: " << ancho << "x" << alto << "  Canales: " << canales << (canales == 3 ? " (color)" : " (gris)") << std::endl;
    std::cout << "Patch: 7x7  Ventana: 21x21  Hilos: 16" << std::endl;

    int tamPatch = 7, tamVentana = 21;
    float h = 25.0f;

    std::vector<std::vector<float>> imgCanales(canales, std::vector<float>(n));
    for (int i = 0; i < n; i++)
        for (int c = 0; c < canales; c++)
            imgCanales[c][i] = (float)datosOrig[i * canalesOrig + c];
    stbi_image_free(datosOrig);

    for (int c = 0; c < canales; c++)
        agregarRuido(imgCanales[c], n, 25.0f);

    std::vector<std::vector<float>> imgLimpia(canales, std::vector<float>(n, 0.0f));

    auto inicio = std::chrono::high_resolution_clock::now();
    for (int c = 0; c < canales; c++)
        nlMeansCPU_OMP(imgCanales[c], imgLimpia[c], ancho, alto, tamPatch, tamVentana, h);
    auto fin = std::chrono::high_resolution_clock::now();

    double tiempoMs = std::chrono::duration<double, std::milli>(fin - inicio).count();
    std::cout << "Tiempo: " << tiempoMs << " ms (" << tiempoMs / 1000.0 << " s)" << std::endl;

    std::vector<unsigned char> salida(n * canales);
    for (int i = 0; i < n; i++)
        for (int c = 0; c < canales; c++)
            salida[i * canales + c] = (unsigned char)fmaxf(0.0f, fminf(255.0f, imgLimpia[c][i]));

    stbi_write_jpg("../resultados/imagenes/limpia_cpu_omp.jpg", ancho, alto, canales, salida.data(), 95);

    std::ofstream csv("../resultados/metricas/metricas_cpu_omp.csv");
    csv << "version,resolucion,canales,hilos,tiempo_ms,tiempo_s\n";
    csv << "CPU_OpenMP," << ancho << "x" << alto << "," << canales << ",16," << tiempoMs << "," << tiempoMs / 1000.0 << "\n";
    csv.close();

    std::cout << "Imagen guardada en resultados/imagenes/limpia_cpu_omp.jpg" << std::endl;
    std::cout << "Metricas guardadas en resultados/metricas/metricas_cpu_omp.csv" << std::endl;
    return 0;
}
