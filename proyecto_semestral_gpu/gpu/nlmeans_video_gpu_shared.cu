#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <direct.h>
#include <io.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define TAM_PATCH   7
#define TAM_VENTANA 21
#define H_PARAM     10.0f
#define BLOCK_SIZE  16
#define MITAD_V     (TAM_VENTANA / 2)
#define MITAD_P     (TAM_PATCH / 2)
#define HALO        (MITAD_V + MITAD_P)
#define TILE_SIZE   (BLOCK_SIZE + 2 * HALO)

#define CUDA_CHECK(call) { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(_err) << " en linea " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// NLM-C con Shared Memory: 3 tiles cargados en memoria compartida
__global__ void nlMeansColorKernelShared(
    const float* ch0, const float* ch1, const float* ch2,
    float* out0, float* out1, float* out2,
    int ancho, int alto) {

    __shared__ float tile0[TILE_SIZE][TILE_SIZE];
    __shared__ float tile1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile2[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x  = blockIdx.x * BLOCK_SIZE + tx;
    int y  = blockIdx.y * BLOCK_SIZE + ty;
    int x0 = blockIdx.x * BLOCK_SIZE - HALO;
    int y0 = blockIdx.y * BLOCK_SIZE - HALO;

    for (int dy = ty; dy < TILE_SIZE; dy += BLOCK_SIZE) {
        for (int dx = tx; dx < TILE_SIZE; dx += BLOCK_SIZE) {
            int gx = max(0, min(ancho-1, x0+dx));
            int gy = max(0, min(alto-1,  y0+dy));
            tile0[dy][dx] = ch0[gy*ancho+gx];
            tile1[dy][dx] = ch1[gy*ancho+gx];
            tile2[dy][dx] = ch2[gy*ancho+gx];
        }
    }
    __syncthreads();

    if (x >= ancho || y >= alto) return;

    const float h2 = H_PARAM * H_PARAM;
    float sumaPesos = 0.0f;
    float sv0 = 0.0f, sv1 = 0.0f, sv2 = 0.0f;
    int lx = tx + HALO, ly = ty + HALO;

    for (int vy = -MITAD_V; vy <= MITAD_V; vy++) {
        for (int vx = -MITAD_V; vx <= MITAD_V; vx++) {
            int ny = y+vy, nx = x+vx;
            if (ny < 0 || ny >= alto || nx < 0 || nx >= ancho) continue;
            int lny = ly+vy, lnx = lx+vx;
            float dist2 = 0.0f;
            for (int py = -MITAD_P; py <= MITAD_P; py++) {
                for (int px = -MITAD_P; px <= MITAD_P; px++) {
                    float a0 = tile0[ly+py][lx+px];
                    float a1 = tile1[ly+py][lx+px];
                    float a2 = tile2[ly+py][lx+px];
                    float b0, b1, b2;
                    int tny = lny+py, tnx = lnx+px;
                    if (tny>=0 && tny<TILE_SIZE && tnx>=0 && tnx<TILE_SIZE) {
                        b0 = tile0[tny][tnx];
                        b1 = tile1[tny][tnx];
                        b2 = tile2[tny][tnx];
                    } else {
                        int gy2 = max(0,min(alto-1,ny+py));
                        int gx2 = max(0,min(ancho-1,nx+px));
                        b0 = ch0[gy2*ancho+gx2];
                        b1 = ch1[gy2*ancho+gx2];
                        b2 = ch2[gy2*ancho+gx2];
                    }
                    float d0 = a0-b0, d1 = a1-b1, d2 = a2-b2;
                    dist2 += d0*d0 + d1*d1 + d2*d2;
                }
            }
            float peso = expf(-dist2 / (3.0f * 49.0f * h2));
            sumaPesos += peso;
            sv0 += peso * tile0[lny][lnx];
            sv1 += peso * tile1[lny][lnx];
            sv2 += peso * tile2[lny][lnx];
        }
    }
    out0[y*ancho+x] = sv0 / sumaPesos;
    out1[y*ancho+x] = sv1 / sumaPesos;
    out2[y*ancho+x] = sv2 / sumaPesos;
}

void procesarFrame(
    float* d_ch0, float* d_ch1, float* d_ch2,
    float* d_out0, float* d_out1, float* d_out2,
    const std::vector<float>& ch0, const std::vector<float>& ch1, const std::vector<float>& ch2,
    std::vector<float>& out0, std::vector<float>& out1, std::vector<float>& out2,
    int ancho, int alto, float& tiempoMs) {

    int n = ancho * alto;
    CUDA_CHECK(cudaMemcpy(d_ch0, ch0.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ch1, ch1.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ch2, ch2.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((ancho+BLOCK_SIZE-1)/BLOCK_SIZE, (alto+BLOCK_SIZE-1)/BLOCK_SIZE);

    cudaEvent_t ev_ini, ev_fin;
    cudaEventCreate(&ev_ini); cudaEventCreate(&ev_fin);
    cudaEventRecord(ev_ini);
    nlMeansColorKernelShared<<<gridDim, blockDim>>>(d_ch0, d_ch1, d_ch2, d_out0, d_out1, d_out2, ancho, alto);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(ev_fin);
    cudaEventSynchronize(ev_fin);

    float t = 0.0f;
    cudaEventElapsedTime(&t, ev_ini, ev_fin);
    tiempoMs += t;

    CUDA_CHECK(cudaMemcpy(out0.data(), d_out0, n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out1.data(), d_out1, n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out2.data(), d_out2, n*sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventDestroy(ev_ini); cudaEventDestroy(ev_fin);
}

std::string elegirVideo() {
    std::vector<std::string> videos;
    struct _finddata_t fi;
    intptr_t h;
    h = _findfirst("videos/*.mp4", &fi);
    if (h != -1) { do { videos.push_back(std::string("videos/") + fi.name); } while (_findnext(h, &fi) == 0); _findclose(h); }
    h = _findfirst("videos/*.avi", &fi);
    if (h != -1) { do { videos.push_back(std::string("videos/") + fi.name); } while (_findnext(h, &fi) == 0); _findclose(h); }
    if (videos.empty()) { std::cerr << "No se encontraron videos en videos/" << std::endl; exit(1); }
    std::cout << "\nVideos disponibles:" << std::endl;
    for (int i = 0; i < (int)videos.size(); i++)
        std::cout << "  [" << i+1 << "] " << videos[i] << std::endl;
    int opcion = 0;
    while (opcion < 1 || opcion > (int)videos.size()) {
        std::cout << "Elige un numero: ";
        if (!(std::cin >> opcion)) { std::cin.clear(); std::cin.ignore(1000, '\n'); opcion = 0; }
        if (opcion < 1 || opcion > (int)videos.size()) std::cout << "Entrada invalida, intenta de nuevo." << std::endl;
    }
    return videos[opcion - 1];
}

int main(int argc, char* argv[]) {
    std::string rutaVideoStr = (argc > 1) ? argv[1] : elegirVideo();
    const char* rutaVideo = rutaVideoStr.c_str();

    cv::VideoCapture cap(rutaVideo);
    if (!cap.isOpened()) {
        std::cerr << "Error: no se pudo abrir el video: " << rutaVideo << std::endl;
        return 1;
    }

    int ancho   = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int alto    = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps  = cap.get(cv::CAP_PROP_FPS);
    int nFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    int n       = ancho * alto;
    int canales = 3;

    std::cout << "=== NLM-C Video GPU CUDA + Shared Memory (Color NL-Means) ===" << std::endl;
    std::cout << "Video : " << ancho << "x" << alto
              << "  FPS: " << fps
              << "  Frames: " << nFrames << std::endl;
    std::cout << "Patch: 7x7  Ventana: 21x21  h: 10" << std::endl;
    std::cout << "Tile: " << TILE_SIZE << "x" << TILE_SIZE
              << "  Shared por bloque: " << 3*TILE_SIZE*TILE_SIZE*4 << " bytes (3 canales)" << std::endl;

    // Nombre de salida derivado del input
    std::string rutaStr(rutaVideo);
    size_t sep = rutaStr.find_last_of("/\\");
    std::string nombreBase = (sep == std::string::npos) ? rutaStr : rutaStr.substr(sep + 1);
    size_t punto = nombreBase.find_last_of('.');
    if (punto != std::string::npos) nombreBase = nombreBase.substr(0, punto);

    _mkdir("../resultados");
    _mkdir("resultados/videos");
    _mkdir("resultados/metricas");

    std::string rutaSalida = "resultados/videos/limpia_gpu_shared_" + nombreBase + ".mp4";

    cv::VideoWriter writer(rutaSalida,
                           cv::VideoWriter::fourcc('m','p','4','v'),
                           fps, cv::Size(ancho, alto));
    if (!writer.isOpened()) {
        std::cerr << "Error: no se pudo crear el video de salida." << std::endl;
        return 1;
    }

    // Pre-alocar memoria GPU una sola vez (3 canales entrada + 3 salida)
    float *d_ch0, *d_ch1, *d_ch2, *d_out0, *d_out1, *d_out2;
    CUDA_CHECK(cudaMalloc(&d_ch0,  n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ch1,  n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ch2,  n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out0, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out1, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out2, n*sizeof(float)));

    float tiempoTotalMs = 0.0f;
    int frameIdx = 0;
    cv::Mat frame, frameF;

    while (cap.read(frame)) {
        frameIdx++;
        if (frameIdx % 30 == 0)
            std::cout << "Procesando frame " << frameIdx << "/" << nFrames << "..." << std::endl;

        frame.convertTo(frameF, CV_32F);

        std::vector<cv::Mat> bgr(canales);
        cv::split(frameF, bgr);

        std::vector<std::vector<float>> imgCanales(canales, std::vector<float>(n));
        std::vector<std::vector<float>> imgLimpia(canales, std::vector<float>(n));

        for (int c = 0; c < canales; c++)
            memcpy(imgCanales[c].data(), (float*)bgr[c].data, n * sizeof(float));

        procesarFrame(d_ch0, d_ch1, d_ch2, d_out0, d_out1, d_out2,
                      imgCanales[0], imgCanales[1], imgCanales[2],
                      imgLimpia[0],  imgLimpia[1],  imgLimpia[2],
                      ancho, alto, tiempoTotalMs);

        std::vector<cv::Mat> bgrLimpia(canales);
        for (int c = 0; c < canales; c++) {
            bgrLimpia[c] = cv::Mat(alto, ancho, CV_32F);
            memcpy((float*)bgrLimpia[c].data, imgLimpia[c].data(), n * sizeof(float));
        }

        cv::Mat frameLimpio;
        cv::merge(bgrLimpia, frameLimpio);
        frameLimpio.convertTo(frameLimpio, CV_8U);
        writer.write(frameLimpio);
    }

    cap.release();
    writer.release();
    cudaFree(d_ch0); cudaFree(d_ch1); cudaFree(d_ch2);
    cudaFree(d_out0); cudaFree(d_out1); cudaFree(d_out2);

    double tiempoPorFrame = tiempoTotalMs / frameIdx;

    std::cout << "Frames procesados : " << frameIdx << std::endl;
    std::cout << "Tiempo total      : " << tiempoTotalMs << " ms (" << tiempoTotalMs / 1000.0f << " s)" << std::endl;
    std::cout << "Tiempo por frame  : " << tiempoPorFrame << " ms" << std::endl;
    std::cout << "Video guardado en " << rutaSalida << std::endl;

    std::ofstream csv("resultados/metricas/metricas_gpu_shared.csv");
    csv << "version,resolucion,fps,frames,cuda_cores,tile_size,shared_bytes,tiempo_total_ms,tiempo_total_s,tiempo_por_frame_ms\n";
    csv << "GPU_Shared," << ancho << "x" << alto << "," << fps << ","
        << frameIdx << ",3072,"
        << TILE_SIZE << "x" << TILE_SIZE << "," << TILE_SIZE*TILE_SIZE*4 << ","
        << tiempoTotalMs << "," << tiempoTotalMs / 1000.0f << "," << tiempoPorFrame << "\n";
    csv.close();

    std::cout << "Metricas guardadas en resultados/metricas/metricas_gpu_shared.csv" << std::endl;
    return 0;
}
