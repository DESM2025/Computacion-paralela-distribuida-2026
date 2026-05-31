#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <io.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#define TAM_PATCH   7
#define TAM_VENTANA 21
#define H_PARAM     25.0f
#define BLOCK_SIZE  16

__global__ void nlMeansKernel(const float* entrada, float* salida, int ancho, int alto) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ancho || y >= alto) return;

    const int mp = TAM_PATCH / 2;
    const int mv = TAM_VENTANA / 2;
    const float h2 = H_PARAM * H_PARAM;

    float sumaPesos = 0.0f, sumaValor = 0.0f;
    for (int vy = -mv; vy <= mv; vy++) {
        for (int vx = -mv; vx <= mv; vx++) {
            int ny = y + vy, nx = x + vx;
            if (ny < 0 || ny >= alto || nx < 0 || nx >= ancho) continue;
            float dist2 = 0.0f;
            for (int py = -mp; py <= mp; py++) {
                for (int px = -mp; px <= mp; px++) {
                    int ay = max(0, min(alto-1,  y+py)), ax = max(0, min(ancho-1, x+px));
                    int by = max(0, min(alto-1, ny+py)), bx = max(0, min(ancho-1, nx+px));
                    float diff = entrada[ay*ancho+ax] - entrada[by*ancho+bx];
                    dist2 += diff * diff;
                }
            }
            float peso = expf(-dist2 / (49.0f * h2));
            sumaPesos += peso;
            sumaValor += peso * entrada[ny*ancho+nx];
        }
    }
    salida[y*ancho+x] = sumaValor / sumaPesos;
}

void agregarRuido(std::vector<float>& imagen, int n, float sigma) {
    srand(42);
    for (int i = 0; i < n; i++) {
        float u1 = ((float)rand()+1.0f)/((float)RAND_MAX+1.0f);
        float u2 = ((float)rand()+1.0f)/((float)RAND_MAX+1.0f);
        float ruido = sigma * sqrtf(-2.0f*logf(u1)) * cosf(2.0f*3.14159265f*u2);
        imagen[i] = fmaxf(0.0f, fminf(255.0f, imagen[i]+ruido));
    }
}

#define CUDA_CHECK(call) { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(_err) << " en linea " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

void procesarCanal(const std::vector<float>& entrada, std::vector<float>& salida, int ancho, int alto, float& tiempoMs) {
    int n = ancho * alto;
    float *d_entrada, *d_salida;
    CUDA_CHECK(cudaMalloc(&d_entrada, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_salida,  n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_entrada, entrada.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((ancho+BLOCK_SIZE-1)/BLOCK_SIZE, (alto+BLOCK_SIZE-1)/BLOCK_SIZE);

    cudaEvent_t ini, fin;
    cudaEventCreate(&ini); cudaEventCreate(&fin);
    cudaEventRecord(ini);
    nlMeansKernel<<<gridDim, blockDim>>>(d_entrada, d_salida, ancho, alto);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(fin);
    cudaEventSynchronize(fin);

    float t = 0.0f;
    cudaEventElapsedTime(&t, ini, fin);
    tiempoMs += t;

    CUDA_CHECK(cudaMemcpy(salida.data(), d_salida, n*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_entrada); cudaFree(d_salida);
    cudaEventDestroy(ini); cudaEventDestroy(fin);
}

std::string elegirImagen() {
    std::vector<std::string> imagenes;
    struct _finddata_t fi; intptr_t h;
    const char* exts[] = {"../imagenes/*.jpg","../imagenes/*.png","../imagenes/*.bmp"};
    for (auto& ext : exts) {
        h = _findfirst(ext, &fi);
        if (h != -1) { do { imagenes.push_back(std::string("../imagenes/") + fi.name); } while (_findnext(h, &fi) == 0); _findclose(h); }
    }
    if (imagenes.empty()) { std::cerr << "No se encontraron imagenes en ../imagenes/" << std::endl; exit(1); }
    std::cout << "\nImagenes disponibles:" << std::endl;
    for (int i = 0; i < (int)imagenes.size(); i++)
        std::cout << "  [" << i+1 << "] " << imagenes[i] << std::endl;
    int op = 0;
    while (op < 1 || op > (int)imagenes.size()) {
        std::cout << "Elige un numero: ";
        if (!(std::cin >> op)) { std::cin.clear(); std::cin.ignore(1000, '\n'); op = 0; }
        if (op < 1 || op > (int)imagenes.size()) std::cout << "Entrada invalida, intenta de nuevo." << std::endl;
    }
    return imagenes[op - 1];
}

bool preguntarRuido() {
    int op = -1;
    while (op != 0 && op != 1) {
        std::cout << "¿Agregar ruido artificial? (1=Si / 0=No): ";
        if (!(std::cin >> op)) { std::cin.clear(); std::cin.ignore(1000, '\n'); op = -1; }
        if (op != 0 && op != 1) std::cout << "Entrada invalida, intenta de nuevo." << std::endl;
    }
    return op == 1;
}

int main(int argc, char* argv[]) {
    std::string rutaImagenStr = (argc > 1) ? argv[1] : elegirImagen();
    const char* rutaImagen = rutaImagenStr.c_str();

    int ancho, alto, canalesOrig;
    unsigned char* datosOrig = stbi_load(rutaImagen, &ancho, &alto, &canalesOrig, 0);
    if (!datosOrig) { std::cerr << "Error cargando imagen." << std::endl; return 1; }

    int n = ancho * alto;
    int canales = (canalesOrig >= 3) ? 3 : 1;

    std::cout << "=== NL-Means GPU CUDA ===" << std::endl;
    std::cout << "Imagen: " << ancho << "x" << alto << "  Canales: " << canales << (canales==3?" (color)":" (gris)") << std::endl;

    std::vector<std::vector<float>> imgCanales(canales, std::vector<float>(n));
    for (int i = 0; i < n; i++)
        for (int c = 0; c < canales; c++)
            imgCanales[c][i] = (float)datosOrig[i*canalesOrig+c];
    stbi_image_free(datosOrig);

    // Nombre base para archivos de salida
    std::string rutaStr(rutaImagen);
    size_t sep = rutaStr.find_last_of("/\\");
    std::string nombreBase = (sep == std::string::npos) ? rutaStr : rutaStr.substr(sep + 1);
    size_t punto = nombreBase.find_last_of('.');
    if (punto != std::string::npos) nombreBase = nombreBase.substr(0, punto);

    bool conRuido = preguntarRuido();
    if (conRuido) {
        for (int c = 0; c < canales; c++)
            agregarRuido(imgCanales[c], n, 25.0f);
        std::string rutaRuidosa = "../resultados/imagenes/ruidosa_" + nombreBase + ".jpg";
        std::vector<unsigned char> imgRuidosaOut(n * canales);
        for (int i = 0; i < n; i++)
            for (int c = 0; c < canales; c++)
                imgRuidosaOut[i * canales + c] = (unsigned char)fmaxf(0.0f, fminf(255.0f, imgCanales[c][i]));
        stbi_write_jpg(rutaRuidosa.c_str(), ancho, alto, canales, imgRuidosaOut.data(), 95);
        std::cout << "Imagen ruidosa guardada en " << rutaRuidosa << std::endl;
    }

    std::vector<std::vector<float>> imgLimpia(canales, std::vector<float>(n));
    float tiempoMs = 0.0f;

    std::cout << "Ejecutando NL-Means en GPU..." << std::endl;
    for (int c = 0; c < canales; c++)
        procesarCanal(imgCanales[c], imgLimpia[c], ancho, alto, tiempoMs);

    std::cout << "Tiempo GPU: " << tiempoMs << " ms (" << tiempoMs/1000.0f << " s)" << std::endl;

    std::vector<unsigned char> salida(n*canales);
    for (int i = 0; i < n; i++)
        for (int c = 0; c < canales; c++)
            salida[i*canales+c] = (unsigned char)fmaxf(0.0f, fminf(255.0f, imgLimpia[c][i]));

    stbi_write_jpg("../resultados/imagenes/limpia_gpu.jpg", ancho, alto, canales, salida.data(), 95);

    std::ofstream csv("../resultados/metricas/metricas_gpu.csv");
    csv << "version,resolucion,canales,cuda_cores,tiempo_ms,tiempo_s\n";
    csv << "GPU_CUDA," << ancho << "x" << alto << "," << canales << ",3072," << tiempoMs << "," << tiempoMs/1000.0f << "\n";
    csv.close();

    std::cout << "Imagen guardada en resultados/imagenes/limpia_gpu.jpg" << std::endl;
    std::cout << "Metricas guardadas en resultados/metricas/metricas_gpu.csv" << std::endl;
    return 0;
}
