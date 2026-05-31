import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

C1 = "#1a3a5c"
C2 = "#2e6da4"
C3 = "#5ba3d9"
C4 = "#a8d4f5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#f7f9fc",
    "figure.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.color": "#d0dce8",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})

os.makedirs("metricas", exist_ok=True)
os.makedirs("graficas", exist_ok=True)
os.makedirs("imagenes", exist_ok=True)

df_seq   = pd.read_csv("metricas/metricas_cpu_secuencial.csv")
t_seq    = df_seq["tiempo_ms"].values[0]
resolucion = df_seq["resolucion"].values[0]
t_omp    = pd.read_csv("metricas/metricas_cpu_omp.csv")["tiempo_ms"].values[0]
t_gpu    = pd.read_csv("metricas/metricas_gpu.csv")["tiempo_ms"].values[0]
t_shared = pd.read_csv("metricas/metricas_gpu_shared.csv")["tiempo_ms"].values[0]

versiones = [
    "CPU Secuencial\n(1 hilo)",
    "CPU OpenMP\n(16 hilos)",
    "GPU CUDA\n(Mem. Global)",
    "GPU CUDA\n(Shared Memory)"
]
tiempos  = [t_seq, t_omp, t_gpu, t_shared]
speedups = [1.0, t_seq/t_omp, t_seq/t_gpu, t_seq/t_shared]
colores  = [C1, C2, C3, C4]

# ── Grafica 1: Tiempo ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(versiones, tiempos, color=colores, width=0.5, edgecolor="white", linewidth=1.2)
for bar, t in zip(bars, tiempos):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + t_seq*0.012,
            f"{t:,.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1a1a2e")
ax.set_title(f"NL-Means — Tiempo de Ejecución ({resolucion})", fontsize=14, fontweight="bold", pad=15, color="#1a1a2e")
ax.set_ylabel("Tiempo (ms)", fontsize=12, color="#1a1a2e")
ax.set_ylim(0, t_seq * 1.2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("graficas/grafica_tiempo.png", dpi=150)
plt.close()
print("Guardada: graficas/grafica_tiempo.png")

# ── Grafica 2: Speedup ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(versiones, speedups, color=colores, width=0.5, edgecolor="white", linewidth=1.2)
for bar, s in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups)*0.012,
            f"{s:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1a1a2e")
ax.set_title(f"NL-Means — Speedup vs CPU Secuencial ({resolucion})", fontsize=14, fontweight="bold", pad=15, color="#1a1a2e")
ax.set_ylabel("Speedup (veces más rápido)", fontsize=12, color="#1a1a2e")
ax.set_ylim(0, max(speedups) * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("graficas/grafica_speedup.png", dpi=150)
plt.close()
print("Guardada: graficas/grafica_speedup.png")

# ── Grafica 3: Tiempo log ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(versiones, tiempos, color=colores, width=0.5, edgecolor="white", linewidth=1.2)
for bar, t in zip(bars, tiempos):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.5,
            f"{t:,.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1a1a2e")
ax.set_yscale("log")
ax.set_title(f"NL-Means — Tiempo en Escala Logarítmica ({resolucion})", fontsize=14, fontweight="bold", pad=15, color="#1a1a2e")
ax.set_ylabel("Tiempo (ms) — escala log", fontsize=12, color="#1a1a2e")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("graficas/grafica_tiempo_log.png", dpi=150)
plt.close()
print("Guardada: graficas/grafica_tiempo_log.png")

print(f"\nResumen:")
print(f"  CPU Secuencial  : {t_seq:>10,.1f} ms  —  1.0x")
print(f"  CPU OpenMP      : {t_omp:>10,.1f} ms  —  {speedups[1]:.1f}x")
print(f"  GPU Mem. Global : {t_gpu:>10,.1f} ms  —  {speedups[2]:.1f}x")
print(f"  GPU Shared Mem  : {t_shared:>10,.1f} ms  —  {speedups[3]:.1f}x")
