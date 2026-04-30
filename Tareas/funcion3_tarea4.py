import concurrent.futures
import hashlib
import random
import string
import time

try:
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except Exception:
	plt = None


def _hash_item(item):
	idx, pwd = item
	return idx, hashlib.sha256(pwd.encode()).hexdigest()


def _hash_chunk(chunk):
	return [_hash_item(item) for item in chunk]


def _particion_bloques(items, num_workers):
	n = len(items)
	base, extra = divmod(n, num_workers)
	out = []
	start = 0
	for w in range(num_workers):
		size = base + (1 if w < extra else 0)
		out.append(items[start:start + size])
		start += size
	return out


def _particion_ciclica(items, num_workers):
	return [items[i::num_workers] for i in range(num_workers)]


def _particion_irregular(items, num_workers):
	bins = [[] for _ in range(num_workers)]
	carga = [0] * num_workers
	for item in sorted(items, key=lambda x: len(x[1]), reverse=True):
		w = min(range(num_workers), key=carga.__getitem__)
		bins[w].append(item)
		carga[w] += len(item[1])
	return bins


def _ejecutar_esquema(nombre, items, particionador, num_workers):
	chunks = [c for c in particionador(items, num_workers) if c]
	cargas = [sum(len(pwd) for _, pwd in c) for c in chunks]

	t0 = time.perf_counter()
	backend = "processes"
	try:
		with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
			pares = [par for bloque in ex.map(_hash_chunk, chunks) for par in bloque]
	except Exception:
		backend = "threads_fallback"
		with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
			pares = [par for bloque in ex.map(_hash_chunk, chunks) for par in bloque]
	dt = time.perf_counter() - t0

	pares.sort(key=lambda x: x[0])
	hashes = [h for _, h in pares]

	prom = sum(cargas) / len(cargas)
	desbalance = max(cargas) / prom if prom else 0.0

	return {
		"esquema": nombre,
		"tiempo_s": dt,
		"desbalance": desbalance,
		"total": len(hashes),
		"hashes": hashes,
		"backend": backend,
	}


def run_partitioning_hash_demo(num_workers=4, num_passwords=120_000, seed=42, show_plot=True):
	rng = random.Random(seed)
	alfabeto = string.ascii_letters + string.digits
	longitudes = [8, 8, 8, 12, 16, 24, 32, 48, 64]

	passwords = [
		"".join(rng.choices(alfabeto, k=rng.choice(longitudes)))
		for _ in range(num_passwords)
	]
	items = list(enumerate(passwords))

	resultados = [
		_ejecutar_esquema("bloques", items, _particion_bloques, num_workers),
		_ejecutar_esquema("ciclico", items, _particion_ciclica, num_workers),
		_ejecutar_esquema("irregular", items, _particion_irregular, num_workers),
	]

	ref = resultados[0]["hashes"]
	for r in resultados:
		r["ok"] = r["hashes"] == ref

	base = resultados[0]["tiempo_s"]
	print("Comparacion de particionamiento para hashing")
	print(f"workers={num_workers} | passwords={num_passwords}")
	print("-" * 96)
	print(f"{'esquema':<12}{'tiempo[s]':>12}{'speedup':>12}{'desbalance':>14}{'total':>12}{'ok':>8}{'backend':>16}")
	print("-" * 96)
	for r in sorted(resultados, key=lambda x: x["tiempo_s"]):
		speedup = base / r["tiempo_s"]
		print(
			f"{r['esquema']:<12}{r['tiempo_s']:>12.3f}{speedup:>12.2f}{r['desbalance']:>14.3f}{r['total']:>12}{str(r['ok']):>8}{r['backend']:>16}"
		)

	if show_plot and plt is not None:
		labels = [r["esquema"] for r in resultados]
		tiempos = [r["tiempo_s"] for r in resultados]
		plt.figure(figsize=(8, 5))
		plt.bar(labels, tiempos, color=["#1f77b4", "#2ca02c", "#d62728"], edgecolor="#1f2937")
		plt.title("Tiempos por esquema de particionamiento")
		plt.xlabel("Esquema")
		plt.ylabel("Tiempo [s]")
		plt.grid(True, linestyle="--", alpha=0.6, axis="y")
		plt.tight_layout()
		plt.show()

	salida = []
	for r in resultados:
		salida.append({
			"esquema": r["esquema"],
			"tiempo_s": r["tiempo_s"],
			"desbalance": r["desbalance"],
			"total": r["total"],
			"ok": r["ok"],
			"backend": r["backend"],
		})
	return salida


if __name__ == "__main__":
	run_partitioning_hash_demo(num_workers=4)
