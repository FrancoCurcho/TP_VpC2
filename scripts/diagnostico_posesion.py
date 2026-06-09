"""Diagnóstico: ¿por qué hay tan pocos frames con posesión?

Trackea una vez y mide: (a) en cuántos frames se detecta la pelota, y (b) cuántos
frames tienen poseedor para varios umbrales. Aísla si el cuello es la DETECCIÓN de
la pelota o el UMBRAL de distancia.

    PYTHONPATH=src uv run --no-project \
        --with ultralytics --with lapx --with opencv-python \
        --with "git+https://github.com/roboflow/sports.git" \
        --with transformers --with umap-learn --with supervision \
        python scripts/diagnostico_posesion.py [SNMOT-XXX] [N_FRAMES]
"""

import sys

import cv2

sys.path.insert(0, "src")
from futbol_vpc import config, detection, possession, team_assign, tracking  # noqa: E402

DEVICE = "mps"
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "SNMOT-108"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else None
CONF = float(sys.argv[4]) if len(sys.argv) > 4 else None
FRAMES = config.DATA_DIR / "soccernet" / "tracking" / "train" / CLIP_NAME / "img1"

print(f"Trackeando {N_FRAMES} frames de {CLIP_NAME} (imgsz={IMGSZ}, conf={CONF})...")
modelo = detection.cargar_modelo()
frames = [cv2.imread(str(FRAMES / f"{i:06d}.jpg")) for i in range(1, N_FRAMES + 1)]
por_frame = tracking.trackear(modelo, frames, device=DEVICE, imgsz=IMGSZ, conf=CONF)
team_assign.TeamAssigner(device=DEVICE).asignar(por_frame, frames)

# (a) ¿en cuántos frames se detecta la pelota?
con_pelota = sum(1 for dets in por_frame if any(d.cls == "ball" for d in dets))
print(f"\nFrames con pelota detectada: {con_pelota}/{N_FRAMES} ({con_pelota/N_FRAMES:.0%})")

# (b) frames con poseedor (crudo, sin suavizar) para varios umbrales
UMBRALES = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
print("\numbral_rel | frames con poseedor")
for u in UMBRALES:
    n = sum(1 for dets in por_frame if possession.poseedor_en_frame(dets, umbral_rel=u))
    print(f"   {u:<7} | {n}/{N_FRAMES} ({n/N_FRAMES:.0%})")

# (c) con interpolación de la pelota en huecos cortos
possession.interpolar_pelota(por_frame, max_gap=5)
con_pelota2 = sum(1 for dets in por_frame if any(d.cls == "ball" for d in dets))
print(f"\nCON interpolación — frames con pelota: {con_pelota2}/{N_FRAMES} "
      f"({con_pelota2/N_FRAMES:.0%})")
print("umbral_rel | frames con poseedor (interpolado)")
for u in UMBRALES:
    n = sum(1 for dets in por_frame if possession.poseedor_en_frame(dets, umbral_rel=u))
    print(f"   {u:<7} | {n}/{N_FRAMES} ({n/N_FRAMES:.0%})")
