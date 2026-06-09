"""Prueba de la Etapa 3 (posesión + métricas) sobre un clip real.

    PYTHONPATH=src uv run --no-project \
        --with ultralytics --with lapx --with opencv-python \
        --with "git+https://github.com/roboflow/sports.git" \
        --with transformers --with umap-learn --with supervision \
        python scripts/probar_posesion.py [SNMOT-XXX] [N_FRAMES] [FRAME_DIBUJO]

Pipeline completo: detección → tracking → equipos (embeddings) → posesión.
Imprime las métricas agregadas y dibuja un frame marcando al poseedor.
"""

import sys

import cv2

sys.path.insert(0, "src")
from futbol_vpc import config, detection, metrics, possession, team_assign, tracking  # noqa: E402

DEVICE = "mps"
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "SNMOT-108"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
FRAME_DIBUJO = int(sys.argv[3]) if len(sys.argv) > 3 else 375
FRAMES = config.DATA_DIR / "soccernet" / "tracking" / "train" / CLIP_NAME / "img1"

print(f"Pipeline sobre {N_FRAMES} frames de {CLIP_NAME}...")
modelo = detection.cargar_modelo()
frames = [cv2.imread(str(FRAMES / f"{i:06d}.jpg")) for i in range(1, N_FRAMES + 1)]
por_frame = tracking.trackear(modelo, frames, device=DEVICE)
team_assign.TeamAssigner(device=DEVICE).asignar(por_frame, frames)

serie = possession.secuencia_posesion(por_frame)
met = metrics.resumen(serie, fps=25)

print("\n=== Métricas de posesión ===")
for equipo, frac in met["porcentaje_posesion"].items():
    print(f"  {equipo}: {frac:.1%}")
print(f"  cambios de posesión: {met['cambios_de_posesion']}")
print(f"  duración media de secuencia: {met['duracion_media_seg']:.1f} s")
frames_con = sum(1 for t in serie if t is not None)
print(f"  frames con posesión: {frames_con}/{len(serie)}")

# --- Dibujar el frame marcando al poseedor ---
idx = FRAME_DIBUJO - 1
COL = {"equipo_A": (0, 0, 255), "equipo_B": (255, 0, 0), None: (128, 128, 128)}
out = frames[idx].copy()
for d in por_frame[idx]:
    x1, y1, x2, y2 = map(int, d.bbox)
    if d.cls == "player":
        c = COL[d.team]
    elif d.cls == "ball":
        c = (0, 255, 255)
    else:
        c = (0, 255, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)

poseedor = possession.poseedor_en_frame(por_frame[idx])
pelotas = [d for d in por_frame[idx] if d.cls == "ball"]
if poseedor is not None and pelotas:
    px, py = map(int, possession._pies(poseedor.bbox))
    bx, by = map(int, max(pelotas, key=lambda d: d.conf).centro)
    cv2.circle(out, (px, py), 14, (0, 255, 255), 3)        # marca al poseedor
    cv2.line(out, (px, py), (bx, by), (0, 255, 255), 2)    # línea a la pelota
    cv2.putText(out, "POSESION", (px - 30, py + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

config.OUTPUTS_DIR.mkdir(exist_ok=True)
salida = config.OUTPUTS_DIR / f"posesion_{CLIP_NAME}_f{FRAME_DIBUJO}.jpg"
cv2.imwrite(str(salida), out)
print("\nFrame anotado guardado en:", salida)
