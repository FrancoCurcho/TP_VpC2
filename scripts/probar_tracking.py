"""Prueba rápida de la Etapa 2 (tracking con ByteTrack) sobre frames reales.

    PYTHONPATH=src uv run --no-project \
        --with ultralytics --with lapx --with opencv-python \
        python scripts/probar_tracking.py [SNMOT-XXX] [N_FRAMES]

Corre tracking sobre los primeros N frames de un clip, reporta la estabilidad de
los IDs (cuántos IDs únicos por clase) y dibuja los track_id sobre un frame.
"""

import sys
from collections import Counter, defaultdict

import cv2

sys.path.insert(0, "src")
from futbol_vpc import config, detection, tracking  # noqa: E402

DEVICE = "mps"
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "SNMOT-108"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 150
FRAMES = config.DATA_DIR / "soccernet" / "tracking" / "train" / CLIP_NAME / "img1"

print(f"Cargando modelo y trackeando {N_FRAMES} frames de {CLIP_NAME}...")
modelo = detection.cargar_modelo()

frames = [cv2.imread(str(FRAMES / f"{i:06d}.jpg")) for i in range(1, N_FRAMES + 1)]
por_frame = tracking.trackear(modelo, frames, device=DEVICE)

# --- Estabilidad de IDs: qué IDs únicos aparecen por clase ---
ids_por_clase = defaultdict(set)
apariciones = Counter()  # cuántos frames vive cada track_id
for dets in por_frame:
    for d in dets:
        if d.track_id is not None:
            ids_por_clase[d.cls].add(d.track_id)
            apariciones[d.track_id] += 1

print("\n=== IDs únicos por clase ===")
for cls, ids in sorted(ids_por_clase.items()):
    print(f"  {cls}: {len(ids)} IDs")

vida_media = sum(apariciones.values()) / len(apariciones) if apariciones else 0
print(f"\nTotal IDs: {len(apariciones)} | vida media: {vida_media:.0f}/{N_FRAMES} frames")
print(f"IDs efímeros (<5 frames): {sum(1 for v in apariciones.values() if v < 5)}")

# --- Dibujar track_id sobre el último frame ---
COL = {"player": (0, 165, 255), "ball": (0, 255, 255), "goalkeeper": (0, 255, 0), "referee": (0, 255, 0)}
out = frames[-1].copy()
for d in por_frame[-1]:
    x1, y1, x2, y2 = map(int, d.bbox)
    color = COL.get(d.cls, (200, 200, 200))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    if d.track_id is not None:
        cv2.putText(out, f"{d.track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

config.OUTPUTS_DIR.mkdir(exist_ok=True)
salida = config.OUTPUTS_DIR / f"prueba_tracking_{CLIP_NAME}.jpg"
cv2.imwrite(str(salida), out)
print("Frame anotado guardado en:", salida)
