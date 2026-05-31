#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include <omp.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// NLM-C: distancia calculada con los 3 canales juntos
void nlMeansColor_OMP(
    const std::vector<float>& ch0, const std::vector<float>& ch1, const std::vector<float>& ch2,
    std::vector<float>& out0, std::vector<float>& out1, std::vector<float>& out2,
    int ancho, int alto, int tamPatch, int tamVentana, float h) {

    int mitadPatch   = tamPatch / 2;
    int mitadVentana = tamVentana / 2;
    float h2 = h * h;

    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(16)
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            float sumaPesos = 0.0f;
            float sv0 = 0.0f, sv1 = 0.0f, sv2 = 0.0f;
            for (int vy = -mitadVentana; vy <= mitadVentana; vy++) {
                for (int vx = -mitadVentana; vx <= mitadVentana; vx++) {
                    int ny = y + vy, nx = x + vx;
                    if (ny < 0 || ny >= alto || nx < 0 || nx >= ancho) continue;
                    float dist2 = 0.0f;
                    for (int py = -mitadPatch; py <= mitadPatch; py++) {
                        for (int px = -mitadPatch; px <= mitadPatch; px++) {
                            int ay = std::max(0, std::min(alto-1,  y+py));
                            int ax = std::max(0, std::min(ancho-1, x+px));
                            int by = std::max(0, std::min(alto-1,  ny+py));
                            int bx = std::max(0, std::min(ancho-1, nx+px));
                            float d0 = ch0[ay*ancho+ax] - ch0[by*ancho+bx];
                            float d1 = ch1[ay*ancho+ax] - ch1[by*ancho+bx];
                            float d2 = ch2[ay*ancho+ax] - ch2[by*ancho+bx];
                            dist2 += d0*d0 + d1*d1 + d2*d2;
                        }
                    }
                    float peso = expf(-dist2 / (3.0f * 49.0f * h2));
                    sumaPesos += peso;
                    sv0 += peso * ch0[ny*ancho+nx];
                    sv1 += peso * ch1[ny*ancho+nx];
                    sv2 += peso * ch2[ny*ancho+nx];
                }
            }
            out0[y*ancho+x] = sv0 / sumaPesos;
            out1[y*ancho+x] = sv1 / sumaPesos;
            out2[y*ancho+x] = sv2 / sumaPesos;
        }
    }
}

std::string elegirVideo() {
    std::vector<std::string> videos;
    for (const auto& entry : fs::directory_iterator("videos")) {
        std::string ext = entry.path().extension().string();
        if (ext == ".mp4" || ext == ".avi" || ext == ".MP4" || ext == ".AVI")
            videos.push_back(entry.path().string());
    }
    if (videos.empty()) { std::cerr << "No se encontraron videos en videos/" << std::endl; exit(1); }
    std::sort(videos.begin(), videos.end());
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

    std::cout << "=== NLM-C Video CPU OpenMP (Color NL-Means) ===" << std::endl;
    std::cout << "Video : " << ancho << "x" << alto
              << "  FPS: " << fps
              << "  Frames: " << nFrames
              << "  Hilos: 16" << std::endl;
    std::cout << "Patch: 7x7  Ventana: 21x21  h: 10" << std::endl;

    // Nombre de salida derivado del input
    std::string rutaStr(rutaVideo);
    size_t sep = rutaStr.find_last_of("/\\");
    std::string nombreBase = (sep == std::string::npos) ? rutaStr : rutaStr.substr(sep + 1);
    size_t punto = nombreBase.find_last_of('.');
    if (punto != std::string::npos) nombreBase = nombreBase.substr(0, punto);

    fs::create_directories("resultados/videos");
    fs::create_directories("resultados/metricas");

    std::string rutaSalida = "resultados/videos/limpia_cpu_omp_" + nombreBase + ".mp4";

    cv::VideoWriter writer(rutaSalida,
                           cv::VideoWriter::fourcc('m','p','4','v'),
                           fps, cv::Size(ancho, alto));
    if (!writer.isOpened()) {
        std::cerr << "Error: no se pudo crear el video de salida." << std::endl;
        return 1;
    }

    int tamPatch = 7, tamVentana = 21;
    float h = 10.0f;

    double tiempoTotalMs = 0.0;
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
        std::vector<std::vector<float>> imgLimpia(canales, std::vector<float>(n, 0.0f));

        for (int c = 0; c < canales; c++)
            memcpy(imgCanales[c].data(), (float*)bgr[c].data, n * sizeof(float));

        auto inicio = std::chrono::high_resolution_clock::now();
        nlMeansColor_OMP(imgCanales[0], imgCanales[1], imgCanales[2],
                         imgLimpia[0],  imgLimpia[1],  imgLimpia[2],
                         ancho, alto, tamPatch, tamVentana, h);
        auto fin_t = std::chrono::high_resolution_clock::now();

        tiempoTotalMs += std::chrono::duration<double, std::milli>(fin_t - inicio).count();

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

    double tiempoPorFrame = tiempoTotalMs / frameIdx;

    std::cout << "Frames procesados : " << frameIdx << std::endl;
    std::cout << "Tiempo total      : " << tiempoTotalMs << " ms (" << tiempoTotalMs / 1000.0 << " s)" << std::endl;
    std::cout << "Tiempo por frame  : " << tiempoPorFrame << " ms" << std::endl;
    std::cout << "Video guardado en " << rutaSalida << std::endl;

    std::ofstream csv("resultados/metricas/metricas_cpu_omp.csv");
    csv << "version,resolucion,fps,frames,hilos,tiempo_total_ms,tiempo_total_s,tiempo_por_frame_ms\n";
    csv << "CPU_OpenMP," << ancho << "x" << alto << "," << fps << ","
        << frameIdx << ",16," << tiempoTotalMs << ","
        << tiempoTotalMs / 1000.0 << "," << tiempoPorFrame << "\n";
    csv.close();

    std::cout << "Metricas guardadas en resultados/metricas/metricas_cpu_omp.csv" << std::endl;
    return 0;
}
