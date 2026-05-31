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
#define MITAD_V     (TAM_VENTANA / 2)
#define MITAD_P     (TAM_PATCH / 2)
#define HALO        (MITAD_V + MITAD_P)
#define TILE_SIZE   (BLOCK_SIZE + 2 * HALO)

__global__ void nlMeansKernelShared(const float* entrada, float* salida, int ancho, int alto) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x  = blockIdx.x * BLOCK_SIZE + tx;
    int y  = blockIdx.y * BLOCK_SIZE + ty;
    int x0 = blockIdx.x * BLOCK_SIZE - HALO;
    int y0 = blockIdx.y * BLOCK_SIZE - HALO;

    for (int dy = ty; dy < TILE_SIZE; dy += BLOCK_SIZE) {
        for (int dx = tx; dx < TILE_SIZE; dx += BLOCK_SIZE) {
            int gx = max(0, min(ancho-1, x0+dx));
            int gy = max(0, min(alto-1,  y0+dy));
            tile[dy][dx] = entrada[gy*ancho+gx];
        }
    }
    __syncthreads();

    if (x >= ancho || y >= alto) return;

    const float h2 = H_PARAM * H_PARAM;
    float sumaPesos = 0.0f, sumaValor = 0.0f;
    int lx = tx + HALO, ly = ty + HALO;

    for (int vy = -MITAD_V; vy <= MITAD_V; vy++) {
        for (int vx = -MITAD_V; vx <= MITAD_V; vx++) {
            int ny = y+vy, nx = x+vx;
            if (ny < 0 || ny >= alto || nx < 0 || nx >= ancho) continue;
            int lny = ly+vy, lnx = lx+vx;
            float dist2 = 0.0f;
            for (int py = -MITAD_P; py <= MITAD_P; py++) {
                for (int px = -MITAD_P; px <= MITAD_P; px++) {
                    float a = tile[ly+py][lx+px];
                    float b;
                    int tny = lny+py, tnx = lnx+px;
                    if (tny>=0 && tny<TILE_SIZE && tnx>=0 && tnx<TILE_SIZE)
                        b = tile[tny][tnx];
                    else
                        b = entrada[max(0,min(alto-1,ny+py))*ancho + max(0,min(ancho-1,nx+px))];
                    float diff = a - b;
                    dist2 += diff * diff;
                }
            }
            float peso = expf(-dist2 / (49.0f * h2));
            sumaPesos += peso;
            sumaValor += peso * tile[lny][lnx];
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

void procesarCanal(const std::vector<float>& entrada, std::vector<float>& salida, int ancho, int alto, float& tiempoMs) {
    int n = ancho * alto;
    float *d_entrada, *d_salida;
    cudaMalloc(&d_entrada, n*sizeof(float));
    cudaMalloc(&d_salida,  n*sizeof(float));
    cudaMemcpy(d_entrada, entrada.data(), n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((ancho+BLOCK_SIZE-1)/BLOCK_SIZE, (alto+BLOCK_SIZE-1)/BLOCK_SIZE);

    cudaEvent_t ini, fin;
    cudaEventCreate(&ini); cudaEventCreate(&fin);
    cudaEventRecord(ini);
    nlMeansKernelShared<<<gridDim, blockDim>>>(d_entrada, d_salida, ancho, alto);
    cudaEventRecord(fin);
    cudaEventSynchronize(fin);

    float t = 0.0f;
    cudaEventElapsedTime(&t, ini, fin);
    tiempoMs += t;

    cudaMemcpy(salida.data(), d_salida, n*sizeof(float), cudaMemcpyDeviceToHost);
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

    std::cout << "=== NL-Means GPU CUDA + Shared Memory ===" << std::endl;
    std::cout << "Imagen: " << ancho << "x" << alto << "  Canales: " << canales << (canales==3?" (color)":" (gris)") << std::endl;
    std::cout << "Tile: " << TILE_SIZE << "x" << TILE_SIZE << "  Shared por bloque: " << TILE_SIZE*TILE_SIZE*4 << " bytes" << std::endl;

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

    std::cout << "Ejecutando NL-Means GPU + Shared Memory..." << std::endl;
    for (int c = 0; c < canales; c++)
        procesarCanal(imgCanales[c], imgLimpia[c], ancho, alto, tiempoMs);

    std::cout << "Tiempo GPU Shared: " << tiempoMs << " ms (" << tiempoMs/1000.0f << " s)" << std::endl;

    std::vector<unsigned char> salida(n*canales);
    for (int i = 0; i < n; i++)
        for (int c = 0; c < canales; c++)
            salida[i*canales+c] = (unsigned char)fmaxf(0.0f, fminf(255.0f, imgLimpia[c][i]));

    stbi_write_jpg("../resultados/imagenes/limpia_gpu_shared.jpg", ancho, alto, canales, salida.data(), 95);

    std::ofstream csv("../resultados/metricas/metricas_gpu_shared.csv");
    csv << "version,resolucion,canales,cuda_cores,tile_size,shared_bytes,tiempo_ms,tiempo_s\n";
    csv << "GPU_Shared," << ancho << "x" << alto << "," << canales << ",3072,"
        << TILE_SIZE << "x" << TILE_SIZE << "," << TILE_SIZE*TILE_SIZE*4 << ","
        << tiempoMs << "," << tiempoMs/1000.0f << "\n";
    csv.close();

    std::cout << "Imagen guardada en resultados/imagenes/limpia_gpu_shared.jpg" << std::endl;
    std::cout << "Metricas guardadas en resultados/metricas/metricas_gpu_shared.csv" << std::endl;
    return 0;
}
