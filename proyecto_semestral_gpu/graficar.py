import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

C1 = "#2e6da4"
C2 = "#5ba3d9"
C3 = "#a8d4f5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#f7f9fc",
    "figure.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.color": "#d0dce8",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})

os.makedirs("resultados/metricas", exist_ok=True)
os.makedirs("resultados/graficas", exist_ok=True)

df_omp    = pd.read_csv("resultados/metricas/metricas_cpu_omp.csv")
df_gpu    = pd.read_csv("resultados/metricas/metricas_gpu.csv")
df_shared = pd.read_csv("resultados/metricas/metricas_gpu_shared.csv")

t_omp    = df_omp["tiempo_total_ms"].values[0]
t_gpu    = df_gpu["tiempo_total_ms"].values[0]
t_shared = df_shared["tiempo_total_ms"].values[0]

tp_omp    = df_omp["tiempo_por_frame_ms"].values[0]
tp_gpu    = df_gpu["tiempo_por_frame_ms"].values[0]
tp_shared = df_shared["tiempo_por_frame_ms"].values[0]

resolucion = df_omp["resolucion"].values[0]
fps        = df_omp["fps"].values[0]
frames     = int(df_omp["frames"].values[0])

versiones = [
    "CPU OpenMP\n(16 hilos)",
    "GPU CUDA\n(Mem. Global)",
    "GPU CUDA\n(Shared Memory)"
]
tiempos_total  = [t_omp, t_gpu, t_shared]
tiempos_frame  = [tp_omp, tp_gpu, tp_shared]
speedups       = [1.0, t_omp/t_gpu, t_omp/t_shared]
colores        = [C1, C2, C3]

# ── Grafica 1: Tiempo total ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(versiones, tiempos_total, color=colores, width=0.5, edgecolor="white", linewidth=1.2)
for bar, t in zip(bars, tiempos_total):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tiempos_total)*0.012,
            f"{t:,.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1a1a2e")
ax.set_title(f"NL-Means Video — Tiempo Total ({resolucion}, {frames} frames)", fontsize=14, fontweight="bold", pad=15, color="#1a1a2e")
ax.set_ylabel("Tiempo (ms)", fontsize=12, color="#1a1a2e")
ax.set_ylim(0, max(tiempos_total) * 1.2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("resultados/graficas/grafica_tiempo_total.png", dpi=150)
plt.close()
print("Guardada: resultados/graficas/grafica_tiempo_total.png")

# ── Grafica 2: Tiempo por frame ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(versiones, tiempos_frame, color=colores, width=0.5, edgecolor="white", linewidth=1.2)
for bar, t in zip(bars, tiempos_frame):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tiempos_frame)*0.012,
            f"{t:,.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1a1a2e")
ax.set_title(f"NL-Means Video — Tiempo por Frame ({resolucion}, {fps:.2f} fps)", fontsize=14, fontweight="bold", pad=15, color="#1a1a2e")
ax.set_ylabel("Tiempo por frame (ms)", fontsize=12, color="#1a1a2e")
ax.set_ylim(0, max(tiempos_frame) * 1.2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("resultados/graficas/grafica_tiempo_frame.png", dpi=150)
plt.close()
print("Guardada: resultados/graficas/grafica_tiempo_frame.png")

# ── Grafica 3: Speedup ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(versiones, speedups, color=colores, width=0.5, edgecolor="white", linewidth=1.2)
for bar, s in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups)*0.012,
            f"{s:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1a1a2e")
ax.set_title(f"NL-Means Video — Speedup vs CPU OpenMP ({resolucion})", fontsize=14, fontweight="bold", pad=15, color="#1a1a2e")
ax.set_ylabel("Speedup (veces más rápido)", fontsize=12, color="#1a1a2e")
ax.set_ylim(0, max(speedups) * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("resultados/graficas/grafica_speedup.png", dpi=150)
plt.close()
print("Guardada: resultados/graficas/grafica_speedup.png")

# ── Grafica 4: Tiempo total escala log ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(versiones, tiempos_total, color=colores, width=0.5, edgecolor="white", linewidth=1.2)
for bar, t in zip(bars, tiempos_total):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.5,
            f"{t:,.1f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1a1a2e")
ax.set_yscale("log")
ax.set_title(f"NL-Means Video — Tiempo en Escala Logarítmica ({resolucion})", fontsize=14, fontweight="bold", pad=30, color="#1a1a2e")
ax.set_ylabel("Tiempo (ms) — escala log", fontsize=12, color="#1a1a2e")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("resultados/graficas/grafica_tiempo_log.png", dpi=150)
plt.close()
print("Guardada: resultados/graficas/grafica_tiempo_log.png")

print(f"\nResumen ({resolucion}, {frames} frames a {fps:.2f} fps):")
print(f"  CPU OpenMP (16 hilos) : {t_omp:>12,.1f} ms total  —  {tp_omp:,.1f} ms/frame  —  1.0x")
print(f"  GPU Mem. Global       : {t_gpu:>12,.1f} ms total  —  {tp_gpu:,.1f} ms/frame  —  {speedups[1]:.1f}x")
print(f"  GPU Shared Memory     : {t_shared:>12,.1f} ms total  —  {tp_shared:,.1f} ms/frame  —  {speedups[2]:.1f}x")
