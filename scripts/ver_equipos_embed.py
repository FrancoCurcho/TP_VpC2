"""Visualización de la asignación de equipo por EMBEDDINGS (módulo team_assign).

    PYTHONPATH=src uv run --no-project \
        --with ultralytics --with lapx --with opencv-python \
        --with "git+https://github.com/roboflow/sports.git" \
        --with transformers --with umap-learn --with supervision \
        python scripts/ver_equipos_embed.py [SNMOT-XXX] [N_FRAMES] [FRAME_DIBUJO]

Trackea, asigna equipo por track con embeddings SigLIP, y dibuja un frame con los
equipos para inspección visual (rojo = equipo_A, azul = equipo_B).
"""

import sys
from collections import Counter

import cv2

sys.path.insert(0, "src")
from futbol_vpc import config, detection, team_assign, tracking  # noqa: E402

DEVICE = "mps"
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "SNMOT-108"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
FRAME_DIBUJO = int(sys.argv[3]) if len(sys.argv) > 3 else 375
FRAMES = config.DATA_DIR / "soccernet" / "tracking" / "train" / CLIP_NAME / "img1"

print(f"Trackeando {N_FRAMES} frames de {CLIP_NAME}...")
modelo = detection.cargar_modelo()
frames = [cv2.imread(str(FRAMES / f"{i:06d}.jpg")) for i in range(1, N_FRAMES + 1)]
por_frame = tracking.trackear(modelo, frames, device=DEVICE)

print("Asignando equipos por embeddings (SigLIP)...")
ta = team_assign.TeamAssigner(device=DEVICE)
ta.asignar(por_frame, frames)
print("Tracks asignados:", dict(Counter(ta.equipo_por_track.values())))

# --- Dibujar el frame elegido ---
idx = FRAME_DIBUJO - 1
COLORES = {"equipo_A": (0, 0, 255), "equipo_B": (255, 0, 0), None: (128, 128, 128)}
out = frames[idx].copy()
for d in por_frame[idx]:
    x1, y1, x2, y2 = map(int, d.bbox)
    if d.cls == "player":
        color = COLORES[d.team]
    elif d.cls == "ball":
        color = (0, 255, 255)
    else:
        color = (0, 255, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

config.OUTPUTS_DIR.mkdir(exist_ok=True)
salida = config.OUTPUTS_DIR / f"equipos_embed_{CLIP_NAME}_f{FRAME_DIBUJO}.jpg"
cv2.imwrite(str(salida), out)
print("Frame anotado guardado en:", salida)
